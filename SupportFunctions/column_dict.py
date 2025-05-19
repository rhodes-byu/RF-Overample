from SupportFunctions.load_datasets import load_datasets

# Categorical Columns
categorical_cols = {
    'artificial_tree': None,
    'audiology': 'all',
    'balance_scale': 'all',
    'breast_cancer': 'all',
    'car': 'all',
    'chess': 'all',
    'crx': [0, 2, 3, 4, 5, 7, 8, 10, 11],
    'diabetes': None,
    'ecoli_5': None,
    'flare1': 'all',
    'glass': None,
    'heart_disease': [1, 2, 5, 6, 8, 10, 11, 12],
    'heart_failure': [1, 2, 4, 7, 8],
    'hepatitis': [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    'hill_valley': None,
    'ionosphere': [0, 1],
    'iris': None,
    'lymphography': 'all',
    'mnist_test': None,
    'optdigits': None,
    'parkinsons': None,
    'seeds': None,
    'segmentation': None,
    'sonar': None,
    'tic-tac-toe': 'all',
    'titanic': [0, 1, 3, 4, 6],
    'treeData': None,
    'waveform': None,
    'wine': None,
    'zoo': 'all'
}

# Build Dictionary
def build_dataset_dict(data_folder):
    raw_datasets = load_datasets(data_folder)
    dataset_dict = {}

    for name, df in raw_datasets.items():
        X = df.drop(columns=df.columns[0])  # temporarily drop target
        cat_spec = categorical_cols.get(name)

        if cat_spec == 'all':
            cat_indices = list(range(X.shape[1]))
        elif cat_spec is None:
            cat_indices = []
        else:
            cat_indices = cat_spec

        dataset_dict[name] = {
            "data": df,
            "categorical_indices": cat_indices
        }

    return dataset_dict
