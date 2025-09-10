# mixedrunner.py
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
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    f1 = f1_score(y_test, preds, average='weighted')
    print(f"[F1 - {label}] Weighted F1 Score: {f1:.4f}")
    return f1


def load_and_preprocess(prepared, dataset_name, seed):
    """
    Loads a single dataset from the already-loaded 'prepared' dict and returns:
      X_train_oh, y_train, X_test_oh, y_test, cat_names
    Expects 'categorical_names' to exist in the pickle (no index fallback).
    """
    if dataset_name not in prepared:
        raise KeyError(f"Dataset '{dataset_name}' not found in prepared dictionary")

    data_entry = prepared[dataset_name]
    df = data_entry["data"].copy() 
    target_col = df.columns[0]

    cat_names = data_entry.get("categorical_names", []) or []

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed
    )

    if cat_names:
        for subset in (X_train_full, X_test):
            present = [c for c in cat_names if c in subset.columns]
            subset[present] = subset[present].astype("category")

    # One-hot encode only declared categorical columns
    X_train_oh = pd.get_dummies(X_train_full, columns=cat_names, dtype=int)
    X_test_oh  = pd.get_dummies(X_test,        columns=cat_names, dtype=int)

    # Align columns (fill missing with 0)
    X_test_oh = X_test_oh.reindex(columns=X_train_oh.columns, fill_value=0)

    return X_train_oh, y_train_full, X_test_oh, y_test, cat_names

# Methods
def run_baseline(prepared, datasets, seeds):
    """
    Run Baseline over multiple seeds and return raw per-seed scores per dataset.
    Returns: dict { dataset_name: [f1_seed0, f1_seed1, ...] }
    """
    baseline_scores = {}
    for dataset in datasets:
        per_seed = []
        for seed in seeds:
            X_train, y_train, X_test, y_test, _ = load_and_preprocess(prepared, dataset, seed)
            f1 = train_and_evaluate_rf(X_train, y_train, X_test, y_test, "Baseline")
            per_seed.append(f1)
        if per_seed:
            baseline_scores[dataset] = per_seed
    return baseline_scores


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


def run_rfoversample(X_imb, y_imb, X_test, y_test, seed, cat_names):
    """
    RFOversampler expects categorical columns; we pass NAMES (no indices).
    'encoded=True' because we already one-hot encoded inputs.
    """
    has_cats = bool(cat_names)

    rf_oversampler = RFOversampler(
        X_imb, y_imb,
        contains_categoricals=has_cats,
        encoded=has_cats,      
        cat_cols=cat_names,   
        K=5,
        add_noise=True,
        noise_scale=0.15,
        cat_prob_sample=False,
        random_state=seed,
        enforce_domains=False,
        binary_strategy="bernoulli",
        hybrid_perturb_frac=0.2,
        boundary_strategy="nearest"
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


def run_all_experiments(prepared, datasets, seeds, imbalance_ratios, results_csv_path):
    all_scores = defaultdict(list)
    results_to_save = []

    # ---- Baseline
    baseline_scores = run_baseline(prepared, datasets, seeds)
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
                X_train_enc, y_train, X_test_enc, y_test, cat_names = load_and_preprocess(prepared, dataset, seed)

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

                # Class Weights
                f1 = run_class_weighted(X_imb, y_imb, X_test_enc, y_test)
                all_scores[(dataset, "ClassWeighted", imbalance_ratio)].append(f1)

                # SMOTE
                f1, _, _ = run_smote(X_imb, y_imb, X_test_enc, y_test, seed)
                all_scores[(dataset, "SMOTE", imbalance_ratio)].append(f1)

                # RF Oversample
                f1 = run_rfoversample(X_imb, y_imb, X_test_enc, y_test, seed, cat_names)
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
    # Load once
    with open("prepared_datasets.pkl", "rb") as f:
        prepared = joblib.load(f)

    # Run all avaliable source openml-cc18 datasets
    datasets = [
        name for name, entry in prepared.items()
        if isinstance(entry, dict) and entry.get("origin") == "openml-cc18"
    ]

    if not datasets:
        raise RuntimeError(
            "No OpenML-CC18 datasets found in prepared_datasets.pkl. "
        )

    imbalance_ratios = [0.05, 0.1, 0.15]
    seeds = random.sample(range(1, 10000), 5)

    results_csv_path = "my_experiment_results.csv"
    if os.path.exists(results_csv_path):
        os.remove(results_csv_path)

    run_all_experiments(prepared, datasets, seeds, imbalance_ratios, results_csv_path)

    final_results = pd.read_csv(results_csv_path)
    print(final_results)


# past running using dataset naming specifically
""" if __name__ == "__main__":
    datasets = [
        "waveform", "optdigits",
        "titanic", "treeData",
        "breast_cancer", "diabetes",
        "crx", "heart_failure",
    ]

    imbalance_ratios = [0.05, 0.1, 0.15]

    seeds = random.sample(range(1, 10000), 15)

    results_csv_path = "my_experiment_results.csv"
    if os.path.exists(results_csv_path):
        os.remove(results_csv_path)

    run_all_experiments(datasets, seeds, imbalance_ratios, results_csv_path)

    final_results = pd.read_csv(results_csv_path)
    print(final_results)
 """