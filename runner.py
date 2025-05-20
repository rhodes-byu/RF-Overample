from SupportFunctions.cross_val import run_experiment
import os

if __name__ == "__main__":
    config = {
        "dataset_folder": "datasets",
        "selected_datasets": [f.split(".")[0] for f in os.listdir("datasets") if f.endswith(".csv")],
        "methods": ['none', 'class_weights', 'smote', 'adasyn', 'random_undersampling', 'rfoversample', 'smotenc'],
        "imbalance_ratios": [0.2],
        "encoding_methods": ["onehot"], # or ordinal
        "use_archetypes": [False],
        "archetype_settings": [{"archetype_proportion": 0.2}],
        "minority_sample_settings": [{"sample_percentage": 0.5}],
        "n_iterations": 5,
        "n_folds": 5,
        "n_jobs": -1,
        "random_state": 42,
        "results_file": "experiment_results.pkl"
    }

    run_experiment(config)