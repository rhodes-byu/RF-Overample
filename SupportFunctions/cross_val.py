import os
import re
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, dump
from sklearn.model_selection import StratifiedKFold

# Import support functions from our project.
from SupportFunctions.model_trainer import ModelTrainer, ResamplingHandler
from SupportFunctions.imbalancer import ImbalanceHandler
from SupportFunctions.prepare_datasets import DatasetPreprocessor
from SupportFunctions.load_datasets import load_datasets
from SupportFunctions.apply_AA import find_minority_archetypes, merge_archetypes_with_minority
from SupportFunctions.visualizer import clean_results

def run_cross_validation(dataset, target_column, encoding_method, method, imbalance_ratio,
                         archetype_setting, minority_sample_setting, use_archetypes,
                         n_folds, seed, random_state):
    """
    Run k-fold cross validation for a given configuration on a dataset.
    
    Performs:
      1. Preprocessing with DatasetPreprocessor.
      2. Splitting the data into k folds using StratifiedKFold.
      3. For each fold:
         - Introduces imbalance via ImbalanceHandler.
         - Optionally applies archetypal analysis (if applicable).
         - Trains and evaluates a model using ModelTrainer.
      4. Aggregates the classification reports by averaging metrics.
      
    Returns:
        pd.DataFrame: The aggregated (averaged) classification report.
    """
    # Preprocess the dataset.
    preprocessor = DatasetPreprocessor(dataset, target_column=target_column, 
                                       encoding_method=encoding_method, random_state=random_state)
    X_full = preprocessor.dataset.drop(columns=[preprocessor.target_column])
    Y_full = preprocessor.dataset[preprocessor.target_column]

    # Set up stratified k-fold splitting.
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_reports = []

    for train_index, val_index in skf.split(X_full, Y_full):
        # Create training and validation splits.
        x_train_fold = X_full.iloc[train_index]
        y_train_fold = Y_full.iloc[train_index]
        x_val_fold = X_full.iloc[val_index]
        y_val_fold = Y_full.iloc[val_index]

        # Introduce imbalance on training data.
        ih = ImbalanceHandler(x_train_fold, y_train_fold, imbalance_ratio, random_state=seed)
        x_train_fold, y_train_fold = ih.introduce_imbalance()

        # Optionally, if method requires and archetypes are used, perform archetypal analysis.
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

        # Train and evaluate the model on the current fold.
        trainer = ModelTrainer(x_train_fold, y_train_fold, x_val_fold, y_val_fold, random_state=random_state)
        report = trainer.train_and_evaluate(method=method)
        fold_reports.append(report)

    # Aggregate the fold reports by averaging the metrics.
    aggregated_report = aggregate_fold_reports(fold_reports)
    return aggregated_report

def aggregate_fold_reports(reports):
    """
    Aggregates classification reports from each fold by averaging numeric metrics.
    
    Args:
        reports (list): List of pandas DataFrames, each a classification report.
    
    Returns:
        pd.DataFrame: The aggregated (averaged) classification report.
    """
    aggregated = pd.concat(reports, axis=0)
    aggregated = aggregated.groupby(aggregated.index).mean()
    return aggregated

def run_experiment(config):
    """
    Runs the entire experiment across a grid of parameter combinations using cross-validation,
    with parallel processing via joblib.
    
    The function:
      1. Loads the selected datasets using load_datasets.
      2. Creates a list of job configurations representing each unique combination of parameters.
      3. Processes each job in parallel (using Parallel/delayed) by calling run_cross_validation.
      4. Attaches identifying metadata (e.g., dataset name, method, imbalance ratio, etc.) to each result.
      5. Aggregates and cleans the results via clean_results, prints debug output, and saves the final DataFrame.
    
    Returns:
        pd.DataFrame: The cleaned DataFrame of experiment results.
    """
    # Load only the selected datasets.
    all_datasets = load_datasets(config.get("dataset_folder", "datasets"),
                                 selected_datasets=config.get("selected_datasets", []))
    print(f"[DEBUG] Loaded datasets: {list(all_datasets.keys())}")

    # Build a list of job configurations.
    jobs = []
    n_iterations = config.get("n_iterations", 1)
    use_archetypes = config.get("use_archetypes", [True])
    if not isinstance(use_archetypes, list):
        use_archetypes = [use_archetypes]
        
    for iteration in range(1, n_iterations + 1):
        seed = iteration  # Sequential seeding.
        for use_arch in use_archetypes:
            for dataset_name, dataset in all_datasets.items():
                for enc in config.get("encoding_methods", ["onehot"]):
                    for method in config.get("methods", ["none"]):
                        if method != "archetypal" or use_arch:
                            for ratio in config.get("imbalance_ratios", [0.1]):
                                for arch_setting in config.get("archetype_settings", [{"archetype_proportion": 0.2}]):
                                    for min_setting in config.get("minority_sample_settings", [{"sample_percentage": 0.5}]):
                                        job_config = {
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
                                        }
                                        jobs.append(job_config)

    # Define a helper function for processing a single job.
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
            result = {
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
            return result
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

    # Process jobs in parallel.
    results = Parallel(n_jobs=config.get("n_jobs", -1))(
        delayed(process_job)(job) for job in jobs
    )

    # Clean and aggregate the results.
    results_df = clean_results(results)

    # DEBUG: Print a sample of the cleaned results.
    print("DEBUG: Cleaned Experiment Results (first 5 rows):")
    print(results_df.head())
    print("DEBUG: Cleaned Results Columns:", results_df.columns.tolist())

    # Save the final results.
    dump(results_df, config.get("results_file", "experiment_results.pkl"))
    print("Experiment results saved to '{}'".format(config.get("results_file", "experiment_results.pkl")))

    return results_df