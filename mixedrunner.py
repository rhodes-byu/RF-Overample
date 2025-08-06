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
from rfoversample import RFOversampler


def train_and_evaluate_rf(X_train, y_train, X_test, y_test, label):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    f1 = f1_score(y_test, preds, average='weighted')
    print(f"[F1 - {label}] Weighted F1 Score: {f1:.4f}")
    return f1


def load_and_preprocess(dataset_name, seed):
    """Load data, skip if any class has fewer than 2 samples, else split and encode."""
    with open("prepared_datasets.pkl", "rb") as f:
        datasets = joblib.load(f)
    data_entry = datasets[dataset_name]
    df = data_entry["data"]
    categorical_indices = data_entry["categorical_indices"]
    target_col = df.columns[0]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Skip this dataset if any class has fewer than 2 samples
    class_counts = y.value_counts()
    if class_counts.min() < 2:
        print(f"Skipping dataset '{dataset_name}' (seed={seed}): insufficient class samples:\n{class_counts}")
        return None, None, None, None

    # Safe to stratify
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y,
        test_size=0.3,
        stratify=y,
        random_state=seed
    )

    # Encode categorical features
    cat_columns = [X.columns[i] for i in categorical_indices]
    train_categories = {}
    for col in cat_columns:
        X_train_full[col] = X_train_full[col].astype("category")
        train_categories[col] = X_train_full[col].cat.categories

    for col in cat_columns:
        X_train_full[col] = X_train_full[col].cat.codes
        X_test[col] = pd.Categorical(
            X_test[col], categories=train_categories[col]
        ).codes

    return X_train_full, y_train_full, X_test, y_test


def run_baseline(datasets, seeds):
    all_results = []
    for dataset in datasets:
        baseline_scores = []
        for seed in seeds:
            X_train, y_train, X_test, y_test = load_and_preprocess(dataset, seed)
            if X_train is None:
                # Skip entire dataset if any seed fails the class-count check
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


def run_rfoversample(X_imb, y_imb, X_test, y_test, seed):
    rf_oversampler = RFOversampler(X_imb, y_imb, num_samples=3)
    X_resampled, y_resampled = rf_oversampler.fit()
    return train_and_evaluate_rf(X_resampled, y_resampled, X_test, y_test, "RFOversample")


def run_hybrid(X_imb, y_imb, X_test, y_test, n_synthetic, ratio_smote, seed):
    minority_class = y_imb.value_counts().idxmin()

    # SMOTE portion
    sm = SMOTE(random_state=seed)
    X_smote_all, y_smote_all = sm.fit_resample(X_imb, y_imb)
    n_original = len(X_imb)
    synthetic_mask = np.arange(len(X_smote_all)) >= n_original
    X_synthetic_smote_all = pd.DataFrame(X_smote_all[synthetic_mask], columns=X_imb.columns)

    n_smote = int(n_synthetic * ratio_smote)
    n_rf = n_synthetic - n_smote

    if len(X_synthetic_smote_all) > n_smote:
        chosen_indices = np.random.choice(len(X_synthetic_smote_all), size=n_smote, replace=False)
        X_synthetic_smote = X_synthetic_smote_all.iloc[chosen_indices].to_numpy()
    else:
        X_synthetic_smote = X_synthetic_smote_all.to_numpy()

    # RF Oversample portion
    if n_rf > 0:
        rf_oversampler = RFOversampler(X_imb, y_imb, num_samples=3)
        X_rf_resampled, y_rf_resampled = rf_oversampler.fit()
        # Extract synthetic samples (those beyond original data)
        synthetic_rf_samples = X_rf_resampled.iloc[n_original:].head(n_rf)
    else:
        synthetic_rf_samples = pd.DataFrame(columns=X_imb.columns)

    X_aug = pd.concat([
        X_imb,
        pd.DataFrame(X_synthetic_smote, columns=X_imb.columns),
        synthetic_rf_samples.reset_index(drop=True)
    ], ignore_index=True)

    y_aug = pd.concat([
        y_imb,
        pd.Series([minority_class] * len(X_synthetic_smote)),
        pd.Series([minority_class] * len(synthetic_rf_samples))
    ], ignore_index=True)

    return train_and_evaluate_rf(
        X_aug, y_aug, X_test, y_test, f"Hybrid_RFoversample_{ratio_smote}_{n_synthetic}"
    )


def append_results_to_csv(results_list, results_csv_path):
    df_new = pd.DataFrame(results_list)
    if os.path.exists(results_csv_path):
        df_existing = pd.read_csv(results_csv_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True).drop_duplicates()
    else:
        df_combined = df_new
    df_combined.to_csv(results_csv_path, index=False)


def run_all_experiments(datasets, seeds, imbalance_ratios, ratio_smotes, n_synthetics, results_csv_path):
    all_scores = defaultdict(list)

    # Baseline
    baseline_results = run_baseline(datasets, seeds)
    for res in baseline_results:
        key = (res["Dataset"], res["Method"], None, None, None)
        all_scores[key].append(res["Mean_Weighted_F1"])

    # Imbalance experiments
    for dataset in datasets:
        for imbalance_ratio in imbalance_ratios:
            for seed in seeds:
                X_train_enc, y_train, X_test_enc, y_test = load_and_preprocess(dataset, seed)
                if X_train_enc is None:
                    # skip this seed/dataset
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
                all_scores[(dataset, "Unbalanced", imbalance_ratio, None, None)].append(f1)

                # SMOTE
                f1, _, _ = run_smote(X_imb, y_imb, X_test_enc, y_test, seed)
                all_scores[(dataset, "SMOTE", imbalance_ratio, None, None)].append(f1)

                # RF Oversample
                f1 = run_rfoversample(X_imb, y_imb, X_test_enc, y_test, seed)
                all_scores[(dataset, "RFOversample", imbalance_ratio, None, None)].append(f1)

                # Hybrid
                for ratio in ratio_smotes:
                    for n_syn in n_synthetics:
                        f1 = run_hybrid(X_imb, y_imb, X_test_enc, y_test, n_syn, ratio, seed)
                        all_scores[(dataset, f"Hybrid_RFoversample_{ratio}_{n_syn}", imbalance_ratio, ratio, n_syn)].append(f1)

    # Aggregate and save
    results_to_save = []
    for key, scores in all_scores.items():
        dataset, method, imb_ratio, ratio_smote, n_syn = key
        mean_score = np.mean(scores)
        std_err = np.std(scores) / np.sqrt(len(scores)) if len(scores) > 1 else np.nan

        results_to_save.append({
            "Dataset": dataset,
            "Method": method,
            "Mean_Weighted_F1": mean_score,
            "Standard_Error": std_err,
            "Ratio_SMOTE": ratio_smote if ratio_smote is not None else np.nan,
            "Imbalance_Ratio": imb_ratio if imb_ratio is not None else np.nan,
            "Seed_Count": len(scores)
        })

    append_results_to_csv(results_to_save, results_csv_path)


if __name__ == "__main__":
    datasets = [

    "sonar", "chess", "diabetes"

    ]
    # "treeData", "waveform", "wine" "artificial_tree", "audiology", "balance_scale", "breast_cancer", "car", 
    # "chess", "crx", "diabetes", "ecoli_5", "flare1", "glass", "heart_disease",
    # "heart_failure", "hepatitis", "hill_valley", "ionosphere", "iris",
    # "lymphography", "parkinsons", "seeds", "segmentation", "sonar",
    # "tic-tac-toe", 

    imbalance_ratios = [0.1, 0.15, 0.2]
    ratio_smotes = [0.5]
    n_synthetics = [100]
    seeds = random.sample(range(1, 10000), 30)

    results_csv_path = "my_experiment_results.csv"
    if os.path.exists(results_csv_path):
        os.remove(results_csv_path)

    run_all_experiments(datasets, seeds, imbalance_ratios, ratio_smotes, n_synthetics, results_csv_path)

    final_results = pd.read_csv(results_csv_path)
    print(final_results)

