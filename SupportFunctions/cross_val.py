import os
import re
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, dump
from sklearn.model_selection import StratifiedKFold

from SupportFunctions.model_trainer import ModelTrainer, ResamplingHandler
from SupportFunctions.imbalancer import ImbalanceHandler
from SupportFunctions.prepare_datasets import DatasetPreprocessor
from SupportFunctions.load_datasets import load_datasets
from SupportFunctions.apply_AA import find_minority_archetypes, merge_archetypes_with_minority
from SupportFunctions.visualizer import clean_results

def run_cross_validation(dataset, target_column, encoding_method, method, imbalance_ratio,
                         archetype_setting, minority_sample_setting, use_archetypes,
                         n_folds, seed, random_state):

    preprocessor = DatasetPreprocessor(dataset, target_column=target_column, 
                                       encoding_method=encoding_method, random_state=random_state, method=method)

    X_full = pd.concat([preprocessor.x_train, preprocessor.x_test], ignore_index=True)
    Y_full = pd.concat([preprocessor.y_train, preprocessor.y_test], ignore_index=True)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_reports = []

    for train_index, val_index in skf.split(X_full, Y_full):
        x_train_fold = X_full.iloc[train_index]
        y_train_fold = Y_full.iloc[train_index]
        x_val_fold = X_full.iloc[val_index]
        y_val_fold = Y_full.iloc[val_index]

        ih = ImbalanceHandler(x_train_fold, y_train_fold, imbalance_ratio, random_state=seed)
        x_train_fold, y_train_fold = ih.introduce_imbalance()

        if method in ["smote", "adasyn"] and use_archetypes:
            if "archetype_proportion" in archetype_setting:
                archetypes = find_minority_archetypes(x_train_fold, y_train_fold,
                                                      archetype_proportion=archetype_setting["archetype_proportion"])
            else:
                archetypes = find_minority_archetypes(x_train_fold, y_train_fold,
                                                      n_archetypes=archetype_setting.get("n_archetypes", 10))

            if "sample_percentage" in minority_sample_setting:
                x_train_fold, y_train_fold = merge_archetypes_with_minority(
                    x_train_fold, y_train_fold, archetypes,
                    sample_percentage=minority_sample_setting["sample_percentage"],
                    random_state=seed
                )
            else:
                x_train_fold, y_train_fold = merge_archetypes_with_minority(
                    x_train_fold, y_train_fold, archetypes,
                    sample_number=minority_sample_setting.get("sample_number", 0),
                    random_state=seed
                )

        trainer = ModelTrainer(x_train_fold, y_train_fold, x_val_fold, y_val_fold, random_state=random_state)
        report = trainer.train_and_evaluate(method=method)
        fold_reports.append(report)

    return aggregate_fold_reports(fold_reports)

def aggregate_fold_reports(reports):
    aggregated = pd.concat(reports, axis=0)
    return aggregated.groupby(aggregated.index).mean()

def run_experiment(config):
    all_datasets = load_datasets(config.get("dataset_folder", "datasets"),
                                 selected_datasets=config.get("selected_datasets", []))

    jobs = []
    n_iterations = config.get("n_iterations", 1)
    use_archetypes = config.get("use_archetypes", [True])
    if not isinstance(use_archetypes, list):
        use_archetypes = [use_archetypes]

    for iteration in range(1, n_iterations + 1):
        seed = iteration
        for use_arch in use_archetypes:
            for dataset_name, dataset in all_datasets.items():
                for enc in config.get("encoding_methods", ["onehot"]):
                    for method in config.get("methods", ["none"]):
                        if method != "archetypal" or use_arch:
                            for ratio in config.get("imbalance_ratios", [0.1]):
                                for arch_setting in config.get("archetype_settings", [{"archetype_proportion": 0.2}]):
                                    for min_setting in config.get("minority_sample_settings", [{"sample_percentage": 0.5}]):
                                        jobs.append({
                                            "dataset_name": dataset_name,
                                            "dataset": dataset,
                                            "encoding_method": enc,
                                            "method": method,
                                            "imbalance_ratio": ratio,
                                            "archetype_setting": arch_setting,
                                            "minority_sample_setting": min_setting,
                                            "use_archetypes": use_arch,
                                            "seed": seed,
                                            "n_folds": config.get("n_folds", 1),
                                            "random_state": config.get("random_state", 42)
                                        })

    def process_job(job_config):
        target_column = config.get("target_column") or job_config["dataset"].columns[0]
        try:
            cv_report = run_cross_validation(
                dataset=job_config["dataset"],
                target_column=target_column,
                encoding_method=job_config["encoding_method"],
                method=job_config["method"],
                imbalance_ratio=job_config["imbalance_ratio"],
                archetype_setting=job_config["archetype_setting"],
                minority_sample_setting=job_config["minority_sample_setting"],
                use_archetypes=job_config["use_archetypes"],
                n_folds=job_config["n_folds"],
                seed=job_config["seed"],
                random_state=job_config["random_state"]
            )
            return {
                "classification_report": cv_report,
                "dataset": job_config["dataset_name"],
                "encoding_method": job_config["encoding_method"],
                "method": job_config["method"],
                "imbalance_ratio": job_config["imbalance_ratio"],
                "archetype_setting": job_config["archetype_setting"],
                "minority_sample_setting": job_config["minority_sample_setting"],
                "use_archetypes": job_config["use_archetypes"],
                "iteration_seed": job_config["seed"]
            }
        except Exception as e:
            return {
                "classification_report": None,
                "dataset": job_config["dataset_name"],
                "encoding_method": job_config["encoding_method"],
                "method": job_config["method"],
                "imbalance_ratio": job_config["imbalance_ratio"],
                "archetype_setting": job_config["archetype_setting"],
                "minority_sample_setting": job_config["minority_sample_setting"],
                "use_archetypes": job_config["use_archetypes"],
                "iteration_seed": job_config["seed"],
                "error": str(e)
            }

    results = Parallel(n_jobs=config.get("n_jobs", -1))(
        delayed(process_job)(job) for job in jobs
    )

    results_df = clean_results(results)
    dump(results_df, config.get("results_file", "experiment_results.pkl"))
    return results_df
