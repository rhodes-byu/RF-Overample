from SupportFunctions.cross_val import run_experiment

if __name__ == "__main__":
    config = {
        "dataset_folder": "datasets",
        "selected_datasets": ["titanic"],
        "methods": ["none", "smote", "class_weights"],
        # Model compatability: "adasyn", "random_undersampling" (if applicable)
        "imbalance_ratios": [0.2],
        "encoding_methods": ["onehot"], # or ordinal
        "use_archetypes": [True],
        "archetype_settings": [{"archetype_proportion": 0.2}],
        "minority_sample_settings": [{"sample_percentage": 0.5}],
        "n_iterations": 3,
        "n_folds": 3,
        "n_jobs": -1,
        "random_state": 42,
        "results_file": "experiment_results.pkl"
    }

    run_experiment(config)
