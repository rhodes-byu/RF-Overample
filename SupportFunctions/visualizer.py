import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
import os

def extract_f1_score(report_df):
    """
    Extracts the weighted F1 score from a classification report DataFrame.
    """
    try:
        if isinstance(report_df, pd.DataFrame) and "weighted avg" in report_df.index:
            return report_df.loc["weighted avg", "f1-score"]
        return np.nan
    except Exception:
        return np.nan

def clean_results(results):
    """
    Converts a list of result dictionaries into a cleaned DataFrame with standardized columns.
    """
    valid_results = [res for res in results if "classification_report" in res]
    results_df = pd.DataFrame(valid_results)
    results_df["Weighted F1 Score"] = results_df["classification_report"].apply(extract_f1_score)
    results_df.rename(columns={
        "dataset": "Dataset", 
        "method": "Method", 
        "imbalance_ratio": "Imbalance Ratio"
    }, inplace=True)
    results_df["Weighted F1 Score"] = pd.to_numeric(results_df["Weighted F1 Score"], errors="coerce")
    results_df["Imbalance Ratio"] = results_df["Imbalance Ratio"].astype(str).replace("N/A", "Original")
    return results_df

def plot_f1_scores(results_df):
    """
    Visualizes the weighted F1 scores for each dataset, aggregating the scores
    (by mean) for each resampling method and imbalance ratio.
    """
    datasets = results_df["Dataset"].unique()
    for dataset in datasets:
        df = results_df[results_df["Dataset"] == dataset]
        # Create a pivot table: index = Method, columns = Imbalance Ratio, values = mean Weighted F1 Score
        pivot = df.groupby(["Method", "Imbalance Ratio"])["Weighted F1 Score"].mean().unstack("Imbalance Ratio")
        pivot = pivot.fillna(0)
        
        # Plot the pivot table as a grouped bar chart
        ax = pivot.plot(kind="bar", figsize=(12, 7))
        plt.title(f"F1 Score Comparison for {dataset}", fontsize=14)
        plt.xlabel("Resampling Method", fontsize=12)
        plt.ylabel("Weighted F1 Score", fontsize=12)
        plt.ylim(0, 1)
        plt.legend(title="Imbalance Ratio")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Allow visualize.py to run independently.
    results_file = "experiment_results.pkl"
    if os.path.exists(results_file):
        results_df = load(results_file)
        print("Loaded experiment results from", results_file)
        print(results_df[["Dataset", "Method", "Imbalance Ratio", "Weighted F1 Score"]])
        plot_f1_scores(results_df)
    else:
        print(f"Results file '{results_file}' not found. Please run runner.py first to generate experiment results.")
