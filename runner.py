from SupportFunctions.cross_val import run_experiment
import json

if __name__ == "__main__":
    with open("random_seeds.json", "r") as f:
        random_seeds = json.load(f)

    print(f"[INFO] Loaded random seeds: {random_seeds}")

    config = {
        # Datasets and Methods
        "selected_datasets": "all",
        "methods": ['smote'],  # none, class_weights, adasyn, random_undersampling, rfoversample, smotenc
        "encoding_methods": ["ordinal"],
        "imbalance_ratios": [0.2],

        # Archetype Parameters
        "use_archetypes": [True, False],
        "archetype_proportions": [0.3],
        "reintroduced_minority": [1.0],

        # Cross-validation
        "random_states": random_seeds,
        "n_folds": 3,
        "n_jobs": -1,

        # Results
        "results_file": "experiment_results.pkl"
    }

    run_experiment(config)
