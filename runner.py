from SupportFunctions.cross_val import run_experiment
import json

if __name__ == "__main__":
    with open("random_seeds.json", "r") as f:
        random_seeds = json.load(f)

    print(f"[INFO] Loaded random seeds: {random_seeds}")

    config = {
        # Datasets and Methods
        "selected_datasets": ["titanic", "sonar", "diabetes", "heart_disease", "heart_failure", "ionosphere", "parkinsons"],
        "methods": ['none', "smote", "class_weights", "adasyn", "random_undersampling", "rfoversample", "baseline"], #none, class_weights, adasyn, random_undersampling, rfoversample, smotenc
        "encoding_methods": ["ordinal"],
        "imbalance_ratios": [0.15, 0.2, 0.25, 0.3, 0.35],

        # Archetypes
        "use_archetypes": [True, False],
        "archetype_settings": [{"archetype_proportion": 0.2}],
        "minority_sample_settings":[{"sample_percentage": 0.5}],

        # Cross-validation using pre-generated seeds
        "random_states": random_seeds,
        "n_folds": 5,
        "n_jobs": -1,

        # Save Results
        "results_file": "experiment_results.pkl"
    }

    run_experiment(config)
