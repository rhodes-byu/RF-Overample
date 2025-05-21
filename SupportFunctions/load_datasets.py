import os
import pandas as pd

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
    'flare1': [0, 1, 2],
    'glass': None,
    'heart_disease': [1, 2, 5, 6, 8, 10, 11, 12],
    'heart_failure': [1, 2, 4, 7, 8],
    'hepatitis': [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    'hill_valley': None,
    'ionosphere': [0],
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

def load_and_prepare_datasets(folder_path="datasets", selected_datasets=None):
    dataset_dict = {}

    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            dataset_name = os.path.splitext(file)[0]

            if selected_datasets is not None and dataset_name not in selected_datasets:
                continue

            file_path = os.path.join(folder_path, file)

            try:
                df = pd.read_csv(file_path)
                X = df.drop(columns=df.columns[0])

                cat_spec = categorical_cols.get(dataset_name)
                if cat_spec == 'all':
                    cat_indices = list(range(X.shape[1]))
                elif cat_spec is None:
                    cat_indices = []
                else:
                    cat_indices = cat_spec

                dataset_dict[dataset_name] = {
                    "data": df,
                    "categorical_indices": cat_indices
                }

                print(f"Loaded: {file} ({df.shape[0]} rows, {df.shape[1]} columns)")

            except Exception as e:
                print(f"Failed to load '{file}': {e}")

    print(f"\nTotal datasets loaded successfully: {len(dataset_dict)}")
    return dataset_dict