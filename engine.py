from joblib import Parallel, delayed, dump
import pandas as pd
import numpy as np

from SupportFunctions.model_trainer import ModelTrainer, ResamplingHandler
from SupportFunctions.imbalancer import ImbalanceHandler
from SupportFunctions.prepare_datasets import DatasetPreprocessor
from SupportFunctions.load_datasets import load_datasets
from SupportFunctions.apply_AA import find_minority_archetypes, merge_archetypes_with_minority
from SupportFunctions.visualizer import clean_results
from SupportFunctions.cross_val import run_cross_validation


class ExperimentRunner:
    def __init__(self, target_column=None, n_jobs=-1, random_state=42, use_archetypes=True, 
                 archetype_setting=None, minority_sample_setting=None, n_iterations=1, n_folds=1):
        """
        Args:
            target_column (str, optional): The target column to be used. Defaults to the first column.
            n_jobs (int): Number of jobs for parallel processing.
            random_state (int): Random seed for reproducibility.
            use_archetypes (bool or list): Whether to apply archetypal analysis. Can be a list (e.g. [True, False]).
            archetype_setting (dict, optional): Default archetype setting dictionary.
            minority_sample_setting (dict, optional): Default minority sample setting dictionary.
            n_iterations (int): Number of iterations (each with its own seed).
            n_folds (int): Number of folds for cross validation within each iteration.
        """
        self.target_column = target_column
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.n_iterations = n_iterations
        self.n_folds = n_folds
        
        # Ensure use_archetypes is always a list.
        if not isinstance(use_archetypes, list):
            self.use_archetypes = [use_archetypes]
        else:
            self.use_archetypes = use_archetypes

        self.archetype_setting = (archetype_setting if archetype_setting is not None 
                                  else {"n_archetypes": 10})
        self.minority_sample_setting = (minority_sample_setting if minority_sample_setting is not None 
                                        else {"sample_percentage": 0.2})
        
    def run_multiple_configs(self, datasets, methods, imbalance_ratios, encoding_methods, 
                             archetype_settings=None, minority_sample_settings=None):
        arch_list = (archetype_settings if archetype_settings is not None 
                     else [self.archetype_setting])
        ms_list = (minority_sample_settings if minority_sample_settings is not None 
                   else [self.minority_sample_setting])
        
        results = []
        # Loop over iterations (each with a sequential seed).
        for iteration in range(1, self.n_iterations + 1):
            current_seed = iteration  # Sequential seeding.
            for use_arch in self.use_archetypes:
                for name, dataset in datasets.items():
                    for enc in encoding_methods:
                        for method in methods:
                            if method != "archetypal" or use_arch:
                                for ratio in imbalance_ratios:
                                    for arch in arch_list:
                                        for ms in ms_list:
                                            res = self._run_single_config_with_cv(
                                                dataset_name=name,
                                                dataset=dataset,
                                                method=method,
                                                imbalance_ratio=ratio,
                                                encoding_method=enc,
                                                archetype_setting=arch,
                                                minority_sample_setting=ms,
                                                use_archetypes=use_arch,
                                                seed=current_seed
                                            )
                                            results.append(res)
        return results

    def _run_single_config_with_cv(self, dataset_name, dataset, method, imbalance_ratio, encoding_method, 
                                   archetype_setting, minority_sample_setting, use_archetypes, seed):
        try:
            target_column = self.target_column if self.target_column else dataset.columns[0]

            cv_report = run_cross_validation(
                dataset=dataset,
                target_column=target_column,
                encoding_method=encoding_method,
                method=method,
                imbalance_ratio=imbalance_ratio,
                archetype_setting=archetype_setting,
                minority_sample_setting=minority_sample_setting,
                use_archetypes=use_archetypes,
                n_folds=self.n_folds,
                seed=seed,
                random_state=self.random_state
            )

            # Debug print: inspect the beginning of cv_report.
            print(f"[DEBUG] dataset: {dataset_name}, method: {method}, seed: {seed}")
            if isinstance(cv_report, pd.DataFrame):
                print(f"[DEBUG] cv_report head:\n{cv_report.head()}")
            else:
                print(f"[DEBUG] cv_report: {cv_report}")

            # Wrap the cv_report under 'classification_report' key.
            cv_report_dict = {"classification_report": cv_report}
            cv_report_dict["dataset"] = dataset_name
            cv_report_dict["encoding_method"] = encoding_method
            cv_report_dict["method"] = method
            cv_report_dict["imbalance_ratio"] = imbalance_ratio
            cv_report_dict["archetype_setting"] = archetype_setting
            cv_report_dict["minority_sample_setting"] = minority_sample_setting
            cv_report_dict["use_archetypes"] = use_archetypes
            cv_report_dict["iteration_seed"] = seed

            return cv_report_dict

        except Exception as e:
            print(f"[ERROR] Exception for dataset: {dataset_name}, method: {method}, seed: {seed}: {e}")
            return {
                "classification_report": None,
                "dataset": dataset_name,
                "encoding_method": encoding_method,
                "method": method,
                "imbalance_ratio": imbalance_ratio,
                "archetype_setting": archetype_setting,
                "minority_sample_setting": minority_sample_setting,
                "use_archetypes": use_archetypes,
                "iteration_seed": seed,
                "error": str(e),
            }


def run_experiment(config):
    """
    Loads datasets (using the new load_datasets which filters by selected names), 
    instantiates the ExperimentRunner, executes the experiments using joblib for parallel processing,
    outputs cleaned results for debugging, and saves the results.
    
    Args:
        config (dict): Configuration dictionary with parameters.
        
    Returns:
        results_df (pd.DataFrame): Cleaned DataFrame of experiment results.
    """
    # Load only the selected datasets using the updated load_datasets.
    all_datasets = load_datasets(
        config.get("dataset_folder", "datasets"),
        selected_datasets=config.get("selected_datasets", [])
    )

    selected_datasets = all_datasets

    # Instantiate ExperimentRunner.
    runner = ExperimentRunner(
        target_column=config.get("target_column", None),
        n_jobs=config.get("n_jobs", -1),
        random_state=config.get("random_state", 42),
        use_archetypes=config.get("use_archetypes", True),
        archetype_setting=config.get("archetype_settings", {"n_archetypes": 10}),
        minority_sample_setting=config.get("minority_sample_settings", {"sample_percentage": 0.2}),
        n_iterations=config.get("n_iterations", 1),
        n_folds=config.get("n_folds", 1)
    )

    # Run experiments across all configurations.
    experiment_results = runner.run_multiple_configs(
        datasets=selected_datasets,
        methods=config.get("methods", ["none"]),
        imbalance_ratios=config.get("imbalance_ratios", [0.1]),
        encoding_methods=config.get("encoding_methods", ["onehot"]),
        archetype_settings=config.get("archetype_settings", [{"archetype_proportion": 0.2}]),
        minority_sample_settings=config.get("minority_sample_settings", [{"sample_percentage": 0.5}])
    )

    # Clean and aggregate results.
    results_df = clean_results(experiment_results)

    # DEBUG: Print the first few rows and the column names.
    print("DEBUG: Cleaned Experiment Results (first 5 rows):")
    print(results_df)
    print("DEBUG: Cleaned Results Columns:", results_df.columns.tolist())

    # Save results.
    dump(results_df, config.get("results_file", "experiment_results.pkl"))
    print("Experiment results saved to '{}'".format(config.get("results_file", "experiment_results.pkl")))

    return results_df