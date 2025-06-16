import numpy as np
import pandas as pd
from joblib import Parallel, delayed, dump
from sklearn.model_selection import StratifiedKFold

# Custom Support Functions
from SupportFunctions.model_trainer import ModelTrainer
from SupportFunctions.imbalancer import ImbalanceHandler
from SupportFunctions.prepare_datasets import DatasetPreprocessor
from SupportFunctions.load_datasets import load_selected_datasets
from SupportFunctions.apply_AA import (
    find_minority_archetypes,
    merge_archetypes_with_minority,
)
from SupportFunctions.visualizer import clean_results


def run_cross_validation(
    dataset,
    target_column,
    encoding_method,
    method,
    imbalance_ratio,
    archetype_setting,
    minority_sample_setting,
    use_archetypes,
    n_folds,
    seed,
    categorical_indices=None,
):
    """Return one DataFrame row per fold, tagged with seed & fold."""
    print(
        f"\n[INFO] CV | method={method:>15s}  enc={encoding_method:<7s} "
        f"seed={seed}"
    )

    apply_imbalance = method != "baseline"
    apply_archetypes = use_archetypes and method in {"smote", "adasyn"}

    pre = DatasetPreprocessor(
        dataset,
        target_column=target_column,
        encoding_method=encoding_method,
        random_state=seed,
        method=method,
        categorical_indices=categorical_indices,
    )
    X_full = pd.concat([pre.x_train, pre.x_test], ignore_index=True)
    y_full = pd.concat([pre.y_train, pre.y_test], ignore_index=True)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_reports = []

    for fold, (tri, tei) in enumerate(skf.split(X_full, y_full), start=1):
        x_tr, y_tr = X_full.iloc[tri], y_full.iloc[tri]
        x_te, y_te = X_full.iloc[tei], y_full.iloc[tei]

        if apply_imbalance:
            ih = ImbalanceHandler(
                x_tr,
                y_tr,
                imbalance_ratio=imbalance_ratio,
                random_state=seed,
            )
            x_tr, y_tr = ih.introduce_imbalance()

        if apply_archetypes:
            archetypes = find_minority_archetypes(
                x_tr,
                y_tr,
                **(archetype_setting or {}),
            )
            x_tr, y_tr = merge_archetypes_with_minority(
                x_tr,
                y_tr,
                archetypes,
                random_state=seed,
                **(minority_sample_setting or {}),
            )

        trainer = ModelTrainer(
            x_tr,
            y_tr,
            x_te,
            y_te,
            random_state=seed,
            categorical_indices=categorical_indices,
            categorical_column_names=[
                x_tr.columns[i] for i in categorical_indices
            ]
            if categorical_indices
            else [],
            contains_categoricals=bool(categorical_indices),
            encoded=(encoding_method == "onehot"),
        )

        report = trainer.train_and_evaluate(method=method)
        if report is None or report.empty:
            print(f"[WARN] Fold {fold} returned empty report → skipped.")
            continue

        report["fold"] = [fold] * len(report)
        report["seed"] = [seed] * len(report)

        weighted_row = report[report["class"] == "weighted avg"]
        if not weighted_row.empty:
            weighted_f1 = weighted_row["f1-score"].values[0]
        else:
            weighted_f1 = None
        report["Weighted F1 Score"] = [weighted_f1] * len(report)

        fold_reports.append(report)

    return pd.concat(fold_reports, axis=0) if fold_reports else pd.DataFrame()


def run_experiment(cfg):
    all_datasets = load_selected_datasets(cfg)

    jobs = []
    for seed in cfg["random_states"]:  # one job block per seed
        for dataset_name, ds in all_datasets.items():
            dataset = ds["data"]
            cat_idx = ds["categorical_indices"]

            for method in cfg["methods"]:
                valid_enc = (
                    ["ordinal"]
                    if method in {"rfoversample", "smotenc"}
                    else cfg["encoding_methods"]
                )
                if method == "smotenc" and not cat_idx:
                    continue

                for enc in valid_enc:
                    for ratio in cfg["imbalance_ratios"]:
                        for use_arch in cfg["use_archetypes"]:
                            arch_iter = (
                                cfg["archetype_settings"] if use_arch else [None]
                            )
                            min_iter = (
                                cfg["minority_sample_settings"]
                                if use_arch
                                else [None]
                            )
                            for arch_set in arch_iter:
                                for min_set in min_iter:
                                    jobs.append(
                                        dict(
                                            dataset_name=dataset_name,
                                            dataset=dataset,
                                            categorical_indices=cat_idx,
                                            encoding_method=enc,
                                            method=method,
                                            imbalance_ratio=ratio,
                                            archetype_setting=arch_set,
                                            minority_sample_setting=min_set,
                                            use_archetypes=use_arch,
                                            seed=seed,
                                            n_folds=cfg["n_folds"],
                                        )
                                    )

    def worker(j):
        tgt_col = cfg.get("target_column") or j["dataset"].columns[0]

        try:
            report = run_cross_validation(
                dataset=j["dataset"],
                target_column=tgt_col,
                encoding_method=j["encoding_method"],
                method=j["method"],
                imbalance_ratio=j["imbalance_ratio"],
                archetype_setting=j["archetype_setting"],
                minority_sample_setting=j["minority_sample_setting"],
                use_archetypes=j["use_archetypes"],
                n_folds=j["n_folds"],
                seed=j["seed"],
                categorical_indices=j["categorical_indices"],
            )

            trimmed = {k: v for k, v in j.items() if k != "dataset"}
            trimmed["classification_report"] = report

            return trimmed

        except Exception as e:
            print(f"[ERROR] job failed → {e}")
            trimmed = {k: v for k, v in j.items() if k != "dataset"}
            trimmed.update({"classification_report": pd.DataFrame(), "error": str(e)})
            return trimmed

    raw_results = Parallel(n_jobs=cfg.get("n_jobs", -1))(
        delayed(worker)(job) for job in jobs
    )

    results_df = clean_results(raw_results)
    dump(results_df, cfg.get("results_file", "experiment_results.pkl"))
    print(f"[INFO] Saved results to {cfg.get('results_file')}")
    return results_df
