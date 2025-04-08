import os
import pandas as pd

def load_datasets(folder_path="datasets", selected_datasets=None):

    datasets = {}

    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            dataset_name = os.path.splitext(file)[0]
            if selected_datasets is not None and dataset_name not in selected_datasets:
                continue

            file_path = os.path.join(folder_path, file)
            try:
                df = pd.read_csv(file_path)
                datasets[dataset_name] = df
                print(f"Loaded: {file} ({df.shape[0]} rows, {df.shape[1]} columns)")
            except Exception as e:
                print(f"Failed to load '{file}'")

    print(f"\nTotal datasets loaded successfully: {len(datasets)}")
    return datasets
