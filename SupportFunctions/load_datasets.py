import os
import pandas as pd

def load_datasets(folder_path="datasets"):
    """
    Load all CSV datasets from a specified folder into a dictionary.
    
    Args:
        folder_path (str): Path to the folder containing CSV datasets. Defaults to 'datasets'.
    
    Returns:
        dict: Dictionary where keys are dataset names (without extension) and values are DataFrames.
    """
    datasets = {}
    
    # Ensure folder exists
    if not os.path.exists(folder_path):
        print(f"Warning: Folder '{folder_path}' does not exist.")
        return datasets

    # Load datasets
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            try:
                datasets[os.path.splitext(file)[0]] = pd.read_csv(file_path)
                print(f"Loaded: {file} ({datasets[file[:-4]].shape[0]} rows, {datasets[file[:-4]].shape[1]} columns)")
            except Exception as e:
                print(f"Warning: Failed to load '{file}' - {e}")
    
    print(f"\nTotal datasets loaded successfully: {len(datasets)}")
    return datasets