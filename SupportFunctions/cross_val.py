import numpy as np
import pandas as pd
from joblib import Parallel, delayed, dump
from sklearn.model_selection import StratifiedKFold

# Custom Support Functions
from SupportFunctions.model_trainer import ModelTrainer
from SupportFunctions.imbalancer import ImbalanceHandler
from SupportFunctions.prepare_datasets import DatasetPreprocessor
from SupportFunctions.load_datasets import load_selected_datasets
from SupportFunctions.apply_AA import find_minority_archetypes, merge_archetypes_with_minority
from SupportFunctions.visualizer import clean_results


def run_cross_validation(
    dataset,
    target_column,
    encoding_method,
    method,
    imbalance_ratio,
    archetype_proportion,
    reintroduced_minority,
    use_archetypes,
    n_folds,
    seed,
    categorical_indices=None,
):
    print(
        f"\n[INFO] CV | method={method:>15s}  enc={encoding_method:<7s} "
        f"seed={seed}"
    )

    apply_imbalance = method != "baseline"
    apply_archetypes = use_archetypes and method in {"smote", "adasyn", "rfoversample"}

    preprocessor = DatasetPreprocessor(
        dataset,
        target_column=target_column,
        encoding_method=encoding_method,
        random_state=seed,
        method=method,
        categorical_indices=categorical_indices,
    )

    X_full = pd.concat([preprocessor.x_train, preprocessor.x_test], ignore_index=True)
    y_full = pd.concat([preprocessor.y_train, preprocessor.y_test], ignore_index=True)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_reports = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_full, y_full), start=1):
        x_train, y_train = X_full.iloc[train_idx], y_full.iloc[train_idx]
        x_test, y_test = X_full.iloc[test_idx], y_full.iloc[test_idx]

        if apply_imbalance:
            imbalancer = ImbalanceHandler(
                x_train,
                y_train,
                imbalance_ratio=imbalance_ratio,
                random_state=seed,
            )
            x_train, y_train = imbalancer.introduce_imbalance()

        if apply_archetypes:
            archetypes = find_minority_archetypes(
                x_train,
                y_train,
                archetype_proportion=archetype_proportion
            )
            x_train, y_train = merge_archetypes_with_minority(
                x_train,
                y_train,
                archetypes,
                sample_percentage=reintroduced_minority,
                random_state=seed
            )

        trainer = ModelTrainer(
            x_train,
            y_train,
            x_test,
            y_test,
            random_state=seed,
            categorical_indices=categorical_indices,
            categorical_column_names=[
                x_train.columns[i] for i in categorical_indices
            ] if categorical_indices else [],
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
        weighted_f1 = weighted_row["f1-score"].values[0] if not weighted_row.empty else None
        report["Weighted F1 Score"] = [weighted_f1] * len(report)

        fold_reports.append(report)

    return pd.concat(fold_reports, axis=0) if fold_reports else pd.DataFrame()


def run_experiment(cfg):
    all_datasets = load_selected_datasets(cfg)

    jobs = []
    for seed in cfg["random_states"]:
        for dataset_name, ds in all_datasets.items():
            dataset = ds["data"]
            categorical_indices = ds["categorical_indices"]

            for method in cfg["methods"]:
                valid_encodings = (
                    ["ordinal"]
                    if method in {"rfoversample", "smotenc"}
                    else cfg["encoding_methods"]
                )
                if method == "smotenc" and not categorical_indices:
                    continue

                for encoding_method in valid_encodings:
                    for imbalance_ratio in cfg["imbalance_ratios"]:
                        for use_archetypes in cfg["use_archetypes"]:
                            arch_iter = (
                                cfg.get("archetype_proportions", [None]) if use_archetypes else [None]
                            )
                            minority_iter = (
                                cfg.get("reintroduced_minority", [None]) if use_archetypes else [None]
                            )

                            for archetype_proportion in arch_iter:
                                for reintroduced_minority in minority_iter:
                                    jobs.append(
                                        dict(
                                            dataset_name=dataset_name,
                                            dataset=dataset,
                                            categorical_indices=categorical_indices,
                                            encoding_method=encoding_method,
                                            method=method,
                                            imbalance_ratio=imbalance_ratio,
                                            archetype_proportion=archetype_proportion,
                                            reintroduced_minority=reintroduced_minority,
                                            use_archetypes=use_archetypes,
                                            seed=seed,
                                            n_folds=cfg["n_folds"],
                                        )
                                    )

    def worker(job):
        target_column = cfg.get("target_column") or job["dataset"].columns[0]

        try:
            report = run_cross_validation(
                dataset=job["dataset"],
                target_column=target_column,
                encoding_method=job["encoding_method"],
                method=job["method"],
                imbalance_ratio=job["imbalance_ratio"],
                archetype_proportion=job["archetype_proportion"],
                reintroduced_minority=job["reintroduced_minority"],
                use_archetypes=job["use_archetypes"],
                n_folds=job["n_folds"],
                seed=job["seed"],
                categorical_indices=job["categorical_indices"],
            )

            trimmed = {k: v for k, v in job.items() if k != "dataset"}
            trimmed["classification_report"] = report

            return trimmed

        except Exception as e:
            print(f"[ERROR] job failed → {e}")
            trimmed = {k: v for k, v in job.items() if k != "dataset"}
            trimmed.update({"classification_report": pd.DataFrame(), "error": str(e)})
            return trimmed

    raw_results = Parallel(n_jobs=cfg.get("n_jobs", -1))(
        delayed(worker)(job) for job in jobs
    )

    results_df = clean_results(raw_results)
    dump(results_df, cfg.get("results_file", "experiment_results.pkl"))
    print(f"[INFO] Saved results to {cfg.get('results_file')}")
    return results_df
