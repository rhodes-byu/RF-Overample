from SupportFunctions.load_datasets import load_and_prepare_datasets
import joblib

if __name__ == "__main__":
    datasets = load_and_prepare_datasets(folder_path="datasets")
    joblib.dump(datasets, "prepared_datasets.pkl")
    print("\nSaved preprocessed datasets to prepared_datasets.pkl")
