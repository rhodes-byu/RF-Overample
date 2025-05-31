from SupportFunctions.cross_val import run_experiment
import os

if __name__ == "__main__":
    config = {

        # Datasets and Methods
        "selected_datasets": ['seeds'],
        "methods": ['none', 'class_weights', 'smote', 'adasyn', 'random_undersampling', 'rfoversample', 'smotenc'],
        "encoding_methods": ["ordinal"],  # onehot
        "imbalance_ratios": [0.2],

        # Archetypes
        "use_archetypes": [False],
        "archetype_settings": [{"archetype_proportion": 0.2}],
        "minority_sample_settings": [{"sample_percentage": 0.5}],

        # Cross-validation
        "n_iterations": 5,
        "n_folds": 5,
        "n_jobs": -1,
        "random_state": 42,

        # Save Results
        "results_file": "experiment_results.pkl"
    }

    run_experiment(config)
