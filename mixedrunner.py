import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
import random
from collections import defaultdict

from rfgap import RFGAP
from SupportFunctions.imbalancer import ImbalanceHandler
from rfoversampleJ import RFOversampler


def train_and_evaluate_rf(X_train, y_train, X_test, y_test, label):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    f1 = f1_score(y_test, preds, average='weighted')
    print(f"[F1 - {label}] Weighted F1 Score: {f1:.4f}")
    return f1


def load_and_preprocess(dataset_name, seed):
    """Load data, skip if any class has fewer than 2 samples, else split and ONE-HOT encode (aligned)."""
    with open("prepared_datasets.pkl", "rb") as f:
        datasets = joblib.load(f)
    data_entry = datasets[dataset_name]
    df = data_entry["data"]
    categorical_indices = data_entry["categorical_indices"]
    target_col = df.columns[0]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Skip dataset if any class has fewer than 2 samples
    class_counts = y.value_counts()
    if class_counts.min() < 2:
        print(f"Skipping dataset '{dataset_name}' (seed={seed}): insufficient class samples:\n{class_counts}")
        return None, None, None, None, None

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed
    )

    cat_columns = [X.columns[i] for i in categorical_indices] if categorical_indices else []
    if cat_columns:
        X_train_full[cat_columns] = X_train_full[cat_columns].astype("category")
        X_test[cat_columns] = X_test[cat_columns].astype("category")

    X_train_oh = pd.get_dummies(X_train_full, columns=cat_columns, dtype=int)
    X_test_oh = pd.get_dummies(X_test, columns=cat_columns, dtype=int)
    X_test_oh = X_test_oh.reindex(columns=X_train_oh.columns, fill_value=0)

    return X_train_oh, y_train_full, X_test_oh, y_test, cat_columns


def run_baseline(datasets, seeds):
    all_results = []
    for dataset in datasets:
        baseline_scores = []
        for seed in seeds:
            X_train, y_train, X_test, y_test, _cat_cols = load_and_preprocess(dataset, seed)
            if X_train is None:
                print(f"Skipping baseline for dataset '{dataset}'.")
                baseline_scores = []
                break
            f1 = train_and_evaluate_rf(X_train, y_train, X_test, y_test, "Baseline")
            baseline_scores.append(f1)
        if not baseline_scores:
            continue
        mean_f1 = np.mean(baseline_scores)
        std_err = np.std(baseline_scores) / np.sqrt(len(baseline_scores))
        all_results.append({
            "Dataset": dataset,
            "Method": "Baseline",
            "Mean_Weighted_F1": mean_f1,
            "Standard_Error": std_err,
            "Ratio_SMOTE": np.nan,
            "Imbalance_Ratio": np.nan,
            "Seed_Count": len(baseline_scores)
        })
    return all_results


def run_unbalanced(X_imb, y_imb, X_test, y_test):
    return train_and_evaluate_rf(X_imb, y_imb, X_test, y_test, "Unbalanced")


def run_smote(X_imb, y_imb, X_test, y_test, seed):
    sm = SMOTE(random_state=seed)
    X_smote, y_smote = sm.fit_resample(X_imb, y_imb)
    return train_and_evaluate_rf(X_smote, y_smote, X_test, y_test, "SMOTE"), X_smote, y_smote


def run_rfoversample(X_imb, y_imb, X_test, y_test, seed, cat_columns):

    # Normalize cat_columns to NAMES (the new oversampler groups by prefixes)
    if cat_columns:
        first = cat_columns[0]
        if isinstance(first, (int, np.integer)):
            cat_columns = [X_imb.columns[i] for i in cat_columns]
    else:
        cat_columns = None

    has_cats = bool(cat_columns)

    rf_oversampler = RFOversampler(
        X_imb, y_imb,
        contains_categoricals=has_cats,
        encoded=has_cats,                # we're passing one-hot features when has_cats=True
        cat_cols=cat_columns,            # NAMES (e.g., ["color","shape"]), not ints
        K=7,
        add_noise=True,
        noise_scale=0.15,
        cat_prob_sample=True,
        random_state=seed,
        enforce_domains=True,
        binary_strategy="bernoulli",     # use "threshold" for deterministic binaries
        hybrid_perturb_frac=0.25         # small diversity boost to reduce mean-collapse
    )

    X_resampled, y_resampled = rf_oversampler.fit()
    return train_and_evaluate_rf(X_resampled, y_resampled, X_test, y_test, "RFOversample")

def append_results_to_csv(results_list, results_csv_path):
    df_new = pd.DataFrame(results_list)
    if os.path.exists(results_csv_path):
        df_existing = pd.read_csv(results_csv_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True).drop_duplicates()
    else:
        df_combined = df_new
    df_combined.to_csv(results_csv_path, index=False)


def run_all_experiments(datasets, seeds, imbalance_ratios, results_csv_path):
    all_scores = defaultdict(list)

    # Baseline
    baseline_results = run_baseline(datasets, seeds)
    for res in baseline_results:
        key = (res["Dataset"], res["Method"], None)
        all_scores[key].append(res["Mean_Weighted_F1"])

    # Imbalance experiments
    for dataset in datasets:
        for imbalance_ratio in imbalance_ratios:
            for seed in seeds:
                X_train_enc, y_train, X_test_enc, y_test, cat_columns = load_and_preprocess(dataset, seed)
                if X_train_enc is None:
                    continue

                imbalancer = ImbalanceHandler(
                    X_train_enc, y_train,
                    imbalance_ratio=imbalance_ratio,
                    batch_size=200,
                    random_state=seed
                )
                X_imb, y_imb = imbalancer.introduce_imbalance()

                # Unbalanced
                f1 = run_unbalanced(X_imb, y_imb, X_test_enc, y_test)
                all_scores[(dataset, "Unbalanced", imbalance_ratio)].append(f1)

                # SMOTE
                f1, _, _ = run_smote(X_imb, y_imb, X_test_enc, y_test, seed)
                all_scores[(dataset, "SMOTE", imbalance_ratio)].append(f1)

                # RF Oversample
                f1 = run_rfoversample(X_imb, y_imb, X_test_enc, y_test, seed, cat_columns)
                all_scores[(dataset, "RFOversample", imbalance_ratio)].append(f1)

    # Aggregate and save
    results_to_save = []
    for key, scores in all_scores.items():
        dataset, method, imb_ratio = key
        mean_score = np.mean(scores)
        std_err = np.std(scores) / np.sqrt(len(scores)) if len(scores) > 1 else np.nan

        results_to_save.append({
            "Dataset": dataset,
            "Method": method,
            "Mean_Weighted_F1": mean_score,
            "Standard_Error": std_err,
            "Ratio_SMOTE": np.nan,
            "Imbalance_Ratio": imb_ratio if imb_ratio is not None else np.nan,
            "Seed_Count": len(scores)
        })

    append_results_to_csv(results_to_save, results_csv_path)


if __name__ == "__main__":
    datasets = [
        "titanic", "heart_disease", "heart_failure", "sonar"
        # add more datasets here if needed
    ]

    imbalance_ratios = [0.1, 0.15, 0.2]
    seeds = random.sample(range(1, 10000), 15)

    results_csv_path = "my_experiment_results.csv"
    if os.path.exists(results_csv_path):
        os.remove(results_csv_path)

    run_all_experiments(datasets, seeds, imbalance_ratios, results_csv_path)

    final_results = pd.read_csv(results_csv_path)
    print(final_results)
