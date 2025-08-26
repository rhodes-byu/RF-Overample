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

# Your libs
from rfgap import RFGAP
from SupportFunctions.imbalancer import ImbalanceHandler
from rfoversampleJ import RFOversampler


# -------------------------
# Model & eval helpers
# -------------------------
def train_and_evaluate_rf(X_train, y_train, X_test, y_test, label):
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    f1 = f1_score(y_test, preds, average='weighted')
    print(f"[F1 - {label}] Weighted F1 Score: {f1:.4f}")
    return f1


def load_and_preprocess(dataset_name, seed):

    with open("prepared_datasets.pkl", "rb") as f:
        datasets = joblib.load(f)

    data_entry = datasets[dataset_name]
    df = data_entry["data"]
    categorical_indices = data_entry["categorical_indices"]
    target_col = df.columns[0]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed
    )

    cat_columns = [X.columns[i] for i in categorical_indices] if categorical_indices else []
    if cat_columns:
        X_train_full[cat_columns] = X_train_full[cat_columns].astype("category")
        X_test[cat_columns] = X_test[cat_columns].astype("category")

    X_train_oh = pd.get_dummies(X_train_full, columns=cat_columns, dtype=int)
    X_test_oh  = pd.get_dummies(X_test,        columns=cat_columns, dtype=int)
    X_test_oh  = X_test_oh.reindex(columns=X_train_oh.columns, fill_value=0)

    return X_train_oh, y_train_full, X_test_oh, y_test, cat_columns


# -------------------------
# Baseline across seeds
# -------------------------
def run_baseline(datasets, seeds):
    """
    Run Baseline over multiple seeds and return raw per-seed scores per dataset.
    Returns: dict { dataset_name: [f1_seed0, f1_seed1, ...] }
    """
    baseline_scores = {}
    for dataset in datasets:
        per_seed = []
        for seed in seeds:
            X_train, y_train, X_test, y_test, _ = load_and_preprocess(dataset, seed)
            if X_train is None:
                print(f"Skipping baseline for dataset '{dataset}'.")
                per_seed = []
                break
            f1 = train_and_evaluate_rf(X_train, y_train, X_test, y_test, "Baseline")
            per_seed.append(f1)
        if per_seed:
            baseline_scores[dataset] = per_seed
    return baseline_scores


# -------------------------
# Methods under test
# -------------------------
def run_unbalanced(X_imb, y_imb, X_test, y_test):
    return train_and_evaluate_rf(X_imb, y_imb, X_test, y_test, "Unbalanced")

def run_class_weighted(X_imb, y_imb, X_test, y_test):
    
    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
    clf.fit(X_imb, y_imb)
    preds = clf.predict(X_test)
    f1 = f1_score(y_test, preds, average='weighted')
    print(f"[F1 - ClassWeighted] Weighted F1 Score: {f1:.4f}")
    return f1

def run_smote(X_imb, y_imb, X_test, y_test, seed):
    sm = SMOTE(random_state=seed)
    X_smote, y_smote = sm.fit_resample(X_imb, y_imb)
    return train_and_evaluate_rf(X_smote, y_smote, X_test, y_test, "SMOTE"), X_smote, y_smote


def run_rfoversample(X_imb, y_imb, X_test, y_test, seed, cat_columns):
    # Normalize cat_columns to NAMES (oversampler groups by prefixes)
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
        encoded=has_cats,
        cat_cols=cat_columns,
        K=5,
        add_noise=True,
        noise_scale=0.15,
        cat_prob_sample=True,
        random_state=seed,
        enforce_domains=False,
        binary_strategy="bernoulli",
        hybrid_perturb_frac=0.20,
        boundary_strategy="nearest"
    )
    X_resampled, y_resampled = rf_oversampler.fit()
    return train_and_evaluate_rf(X_resampled, y_resampled, X_test, y_test, "RFOversample")


# -------------------------
# Results persistence
# -------------------------
def append_results_to_csv(results_list, results_csv_path):
    df_new = pd.DataFrame(results_list)
    if os.path.exists(results_csv_path):
        df_existing = pd.read_csv(results_csv_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True).drop_duplicates()
    else:
        df_combined = df_new
    df_combined.to_csv(results_csv_path, index=False)


# -------------------------
# Orchestration
# -------------------------
def run_all_experiments(datasets, seeds, imbalance_ratios, results_csv_path):
    all_scores = defaultdict(list)
    results_to_save = []

    # ---- Baseline (no induced imbalance)
    baseline_scores = run_baseline(datasets, seeds)
    for dataset, scores in baseline_scores.items():
        n = len(scores)
        mean_score = float(np.mean(scores))
        std_dev = float(np.std(scores, ddof=1)) if n > 1 else 0.0
        std_err = float(std_dev / np.sqrt(n)) if n > 0 else np.nan

        results_to_save.append({
            "Dataset": dataset,
            "Method": "Baseline",
            "Mean_Weighted_F1": mean_score,
            "Standard_Deviation": std_dev,
            "Standard_Error": std_err,
            "Ratio_SMOTE": np.nan,
            "Imbalance_Ratio": np.nan,
            "Seed_Count": n
        })

    # ---- Induce imbalance & evaluate methods
    for dataset in datasets:
        for imbalance_ratio in imbalance_ratios:
            for seed in seeds:
                X_train_enc, y_train, X_test_enc, y_test, cat_columns = load_and_preprocess(dataset, seed)
                if X_train_enc is None:
                    continue

                # NEW: Pairwise majority-relative cap (no batch_size/min-minority)
                imbalancer = ImbalanceHandler(
                    X_train_enc, y_train,
                    imbalance_ratio=imbalance_ratio,
                    random_state=seed,
                    floor_base=20,
                    K=5
                )
                X_imb, y_imb = imbalancer.introduce_imbalance()

                # Unbalanced
                f1 = run_unbalanced(X_imb, y_imb, X_test_enc, y_test)
                all_scores[(dataset, "Unbalanced", imbalance_ratio)].append(f1)

                f1 = run_class_weighted(X_imb, y_imb, X_test_enc, y_test)
                all_scores[(dataset, "ClassWeighted", imbalance_ratio)].append(f1)
                # SMOTE
                f1, _, _ = run_smote(X_imb, y_imb, X_test_enc, y_test, seed)
                all_scores[(dataset, "SMOTE", imbalance_ratio)].append(f1)

                # RF Oversample
                f1 = run_rfoversample(X_imb, y_imb, X_test_enc, y_test, seed, cat_columns)
                all_scores[(dataset, "RFOversample", imbalance_ratio)].append(f1)

    # ---- Aggregate across seeds and persist
    for key, scores in all_scores.items():
        dataset, method, imb_ratio = key
        n = len(scores)
        mean_score = float(np.mean(scores))
        std_dev = float(np.std(scores, ddof=1)) if n > 1 else 0.0
        std_err = float(std_dev / np.sqrt(n)) if n > 0 else np.nan

        results_to_save.append({
            "Dataset": dataset,
            "Method": method,
            "Mean_Weighted_F1": mean_score,
            "Standard_Deviation": std_dev,
            "Standard_Error": std_err,
            "Ratio_SMOTE": np.nan,
            "Imbalance_Ratio": imb_ratio if imb_ratio is not None else np.nan,
            "Seed_Count": n
        })

    append_results_to_csv(results_to_save, results_csv_path)


if __name__ == "__main__":
    datasets = [
        "waveform", "optdigits",
        "titanic", "treeData",
        "breast_cancer", "diabetes",
        "crx", "heart_failure",
    ]

    imbalance_ratios = [0.05, 0.1, 0.15]

    seeds = random.sample(range(1, 10000), 30)

    results_csv_path = "my_experiment_results.csv"
    if os.path.exists(results_csv_path):
        os.remove(results_csv_path)

    run_all_experiments(datasets, seeds, imbalance_ratios, results_csv_path)

    final_results = pd.read_csv(results_csv_path)
    print(final_results)
