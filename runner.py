from engine import run_experiment

if __name__ == "__main__":
    config = {
        "dataset_folder": "datasets",
        "selected_datasets": ["titanic"],  # Only load the Titanic dataset.
        "methods": ["none", "smote", "class_weights"],
        # Other options: "adasyn", "random_undersampling" (if applicable)
        "imbalance_ratios": [0.2],
        "encoding_methods": ["onehot"],  # Alternatives: "ordinal"
        "use_archetypes": [True],
        "archetype_settings": [{"archetype_proportion": 0.2}],
        "minority_sample_settings": [{"sample_percentage": 0.5}],
        "n_iterations": 10,
        "n_folds": 5,
        "n_jobs": -1,  # Parallel processing flag (if enabled in engine.py)
        "random_state": 42,
        "results_file": "experiment_results.pkl"
    }

    run_experiment(config)
