from joblib import Parallel, delayed, dump, load
import pandas as pd
import numpy as np

# Import custom modules from the SupportFunctions package
from SupportFunctions.model_trainer import ModelTrainer, ResamplingHandler
from SupportFunctions.imbalancer import ImbalanceHandler
from SupportFunctions.prepare_datasets import DatasetPreprocessor
from SupportFunctions.load_datasets import load_datasets
from SupportFunctions.apply_AA import find_minority_archetypes, merge_archetypes_with_minority
from SupportFunctions.visualizer import clean_results, plot_f1_scores

class ExperimentRunner:
    """Runs multiple dataset configurations with different imbalance techniques, encoding methods, and archetypal settings."""
    
    def __init__(self, target_column=None, n_jobs=-1, random_state=42, use_archetypes=True, 
                 archetype_setting=None, minority_sample_setting=None):
        """
        Args:
            target_column (str, optional): The target column to be used. Defaults to the first column.
            n_jobs (int): Number of jobs for parallel processing.
            random_state (int): Random seed for reproducibility.
            use_archetypes (bool): Whether to apply archetypal analysis as a preprocessing step.
            archetype_setting (dict, optional): Default archetype setting dictionary (e.g., {"n_archetypes": 10} or {"archetype_proportion": 0.3}).
            minority_sample_setting (dict, optional): Default minority sample setting dictionary (e.g., {"sample_percentage": 0.2} or {"sample_number": 50}).
        """
        self.target_column = target_column
        self.n_jobs = n_jobs  
        self.random_state = random_state
        self.use_archetypes = use_archetypes
        self.archetype_setting = archetype_setting if archetype_setting is not None else {"n_archetypes": 10}
        self.minority_sample_setting = minority_sample_setting if minority_sample_setting is not None else {"sample_percentage": 0.2}

    def run_multiple_configs(self, datasets, methods, imbalance_ratios, encoding_methods, 
                             archetype_settings=None, minority_sample_settings=None):
        """
        Runs experiments over datasets, encoding methods, resampling methods, imbalance ratios,
        and archetypal settings.
        
        Args:
            datasets (dict): Dictionary of dataset names -> DataFrames.
            methods (list): List of resampling methods (e.g., ["none", "class_weights", "smote", "adasyn", "random_undersampling", "easy_ensemble"]).
            imbalance_ratios (list): List of imbalance ratios (e.g., [0.2, 0.25]).
            encoding_methods (list): List of categorical encoding methods (e.g., ["ordinal", "onehot", "frequency", "barycentric"]).
            archetype_settings (list, optional): List of archetype setting dictionaries. If not provided, defaults to [self.archetype_setting].
            minority_sample_settings (list, optional): List of minority sample setting dictionaries. If not provided, defaults to [self.minority_sample_setting].
        
        Returns:
            list: A list of dictionaries containing experiment results.
        """
        # If not provided, use the default setting in a one-element list
        arch_list = archetype_settings if archetype_settings is not None else [self.archetype_setting]
        ms_list = minority_sample_settings if minority_sample_settings is not None else [self.minority_sample_setting]
        
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._run_single_config)(name, dataset, method, ratio, enc, arch, ms)
            for name, dataset in datasets.items()
            for enc in encoding_methods
            for method in methods
            if method != "archetypal" or self.use_archetypes  # skip standalone archetypal method if disabled
            for ratio in imbalance_ratios
            for arch in arch_list
            for ms in ms_list
        )
        return results

    def _run_single_config(self, dataset_name, dataset, method, imbalance_ratio, encoding_method, archetype_setting, minority_sample_setting):
        try:
            # Determine target column (default to first column if not specified)
            target_column = self.target_column if self.target_column else dataset.columns[0]
            
            # Step 1: Preprocess the dataset using the specified encoding method
            preprocessor = DatasetPreprocessor(dataset, target_column=target_column, encoding_method=encoding_method)
            x_train, x_test, y_train, y_test = (
                preprocessor.x_train,
                preprocessor.x_test,
                preprocessor.y_train,
                preprocessor.y_test,
            )
            
            # Step 2: Introduce class imbalance using ImbalanceHandler
            imbalance_handler = ImbalanceHandler(x_train, y_train, imbalance_ratio, random_state=self.random_state)
            x_train, y_train = imbalance_handler.introduce_imbalance()
            
            # Step 3: If the method generates points (SMOTE/ADASYN) and archetypal analysis is enabled,
            # apply archetypal analysis as a preprocessing step.
            if method in ["smote", "adasyn"] and self.use_archetypes:
                if "archetype_proportion" in archetype_setting:
                    archetypes = find_minority_archetypes(x_train, y_train, archetype_proportion=archetype_setting["archetype_proportion"])
                else:
                    archetypes = find_minority_archetypes(x_train, y_train, n_archetypes=archetype_setting.get("n_archetypes", 10))
                
                if "sample_percentage" in minority_sample_setting:
                    x_train, y_train = merge_archetypes_with_minority(
                        x_train, y_train, archetypes, sample_percentage=minority_sample_setting["sample_percentage"],
                        random_state=self.random_state
                    )
                else:
                    x_train, y_train = merge_archetypes_with_minority(
                        x_train, y_train, archetypes, sample_number=minority_sample_setting.get("sample_number", 0),
                        random_state=self.random_state
                    )
            
            # Step 4: Apply resampling methods if needed
            if method in ["smote", "adasyn", "random_undersampling"]:
                resampler = ResamplingHandler(x_train, y_train, random_state=self.random_state)
                if method == "smote":
                    x_train, y_train = resampler.apply_smote()
                elif method == "adasyn":
                    x_train, y_train = resampler.apply_adasyn()
                elif method == "random_undersampling":
                    x_train, y_train = resampler.apply_random_undersampling()
            
            # For "none", "class_weights", or "easy_ensemble", no extra resampling is applied.
            trainer = ModelTrainer(x_train, y_train, x_test, y_test, random_state=self.random_state)
            result = trainer.train_and_evaluate(method=method)
            
            return {
                "dataset": dataset_name,
                "encoding_method": encoding_method,
                "method": method,
                "imbalance_ratio": imbalance_ratio,
                "archetype_setting": archetype_setting,
                "minority_sample_setting": minority_sample_setting,
                "classification_report": result,
            }
        except Exception as e:
            return {
                "dataset": dataset_name,
                "encoding_method": encoding_method,
                "method": method,
                "imbalance_ratio": imbalance_ratio,
                "archetype_setting": archetype_setting,
                "minority_sample_setting": minority_sample_setting,
                "error": str(e),
            }

# Specify datasets
all_datasets = load_datasets()
selected = ["crx", "titanic"]
selected_datasets = {name: df for name, df in all_datasets.items() if name in selected}

if __name__ == "__main__":
    datasets = selected_datasets  # Expects a dict of dataset names -> DataFrames
    methods = ["none", "class_weights", "smote", "adasyn", "random_undersampling", "easy_ensemble"]
    imbalance_ratios = [0.2, 0.25]
    encoding_methods = ["ordinal", "onehot", "frequency", "barycentric"]
    use_archetypes = True  # Set to False to disable archetypal analysis as a preprocessing step

    # Define lists of settings for archetypes and minority samples.
    # For archetypes, for example, try proportions 0.3 and 0.4:
    archetype_settings = [{"archetype_proportion": 0.3}, {"archetype_proportion": 0.4}]
    # For minority samples, try merging 20% and 30% of the original minority points:
    minority_sample_settings = [{"sample_percentage": 0.2}, {"sample_percentage": 0.3}]

    runner = ExperimentRunner(target_column=None, n_jobs=-1, random_state=42, use_archetypes=use_archetypes)
    experiment_results = runner.run_multiple_configs(datasets, methods, imbalance_ratios, encoding_methods, 
                                                       archetype_settings=archetype_settings, minority_sample_settings=minority_sample_settings)
    
    # Clean results and dump to disk
    results_df = clean_results(experiment_results)
    dump(results_df, "experiment_results.pkl")
    print("Experiment results saved to 'experiment_results.pkl'")
    
    # Load results from file and plot
    loaded_results_df = load("experiment_results.pkl")
    print("\nLoaded Experiment Results:")
    print(loaded_results_df[["Dataset", "encoding_method", "Method", "Imbalance Ratio", "Weighted F1 Score"]])
    plot_f1_scores(loaded_results_df)
