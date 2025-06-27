from SupportFunctions.cross_val import run_experiment
import json

if __name__ == "__main__":
    with open("random_seeds.json", "r") as f:
        random_seeds = json.load(f)

    print(f"[INFO] Loaded random seeds: {random_seeds}")

    config = {
        # Datasets and Methods
<<<<<<< HEAD
        "selected_datasets": ["titanic"],
        "methods": ['smote'],  # none, class_weights, adasyn, random_undersampling, rfoversample, smotenc
        "encoding_methods": ["ordinal"],
        "imbalance_ratios": [0.15, 0.2, 0.25],

        # Archetype Parameters
        "use_archetypes": [True, False],
        "archetype_proportions": [0.1, 0.2, 0.3],
        "reintroduced_minority": [0.1, 0.3, 0.5, 0.7, 0.9],
=======
        "selected_datasets": ['crx'],
        "methods": ['rfoversample'], #none, class_weights, adasyn, random_undersampling, rfoversample, smotenc
        "encoding_methods": ["ordinal"],
        "imbalance_ratios": [0.15, 0.2, 0.25],

        # Archetypes
        "use_archetypes": [False, False],
        "archetype_settings": [{"archetype_proportion": 0.2}],
        "minority_sample_settings":[{"sample_percentage": 0.5}],
>>>>>>> 0dd340e3cd2800e2454f571b7e849081c2bf7c14

        # Cross-validation
        "random_states": random_seeds,
        "n_folds": 3,
        "n_jobs": -1,

        # Results
        "results_file": "experiment_results.pkl"
    }

    run_experiment(config)
