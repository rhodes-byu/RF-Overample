from SupportFunctions.cross_val import run_experiment

if __name__ == "__main__":
    config = {
        "dataset_folder": "datasets",
        "selected_datasets": ["diabetes"],
        "methods": ["rfoversample"],
        # Model compatability: "adasyn", "random_undersampling" (if applicable)
        "imbalance_ratios": [0.2],
<<<<<<< HEAD
        "encoding_methods": ["onehot"], # or ordinal
=======
        "encoding_methods": ["other"], # or ordinal
>>>>>>> 1c5862261008c1288b97717ad7b674e7bcfa003f
        "use_archetypes": [False],
        "archetype_settings": [{"archetype_proportion": 0.2}],
        "minority_sample_settings": [{"sample_percentage": 0.5}],
        "n_iterations": 3,
        "n_folds": 3,
        "n_jobs": -1,
        "random_state": 42,
        "results_file": "experiment_results.pkl"
    }

    run_experiment(config)
