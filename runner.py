from SupportFunctions.cross_val import run_experiment

if __name__ == "__main__":
    config = {
        "dataset_folder": "datasets",
        "selected_datasets": ["hill_valley", 'artificial_tree', 'balance_scale', 'breast_cancer', 
                              'diabetes', 'ecoli_5', 'glass', 'iris', 'mnist_test', 'optdigits',
                              'parkinsons', 'seeds', 'segmentation', 'sonar', 'treeData', 'waveform',
                              'wine'],
        "methods": ['none', 'class_weights', 'class_weights', 'smote', 'adasyn', 
                    'random_undersampling', 'easy_ensemble', 'rfoversample'],
        # Model compatability: "adasyn", "random_undersampling" (if applicable)
        "imbalance_ratios": [0.2],
        "encoding_methods": ["onehot"], # or ordinal
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
