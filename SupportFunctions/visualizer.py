import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
import os

def extract_f1_score(report_df):
    try:
        if isinstance(report_df, pd.DataFrame) and "weighted avg" in report_df.index:
            return report_df.loc["weighted avg", "f1-score"]
        return np.nan
    except Exception:
        return np.nan

def clean_results(results):

    valid_results = [res for res in results if "classification_report" in res]
    results_df = pd.DataFrame(valid_results)

    # Extract weighted F1 Score
    results_df["Weighted F1 Score"] = results_df["classification_report"].apply(extract_f1_score)

    # Rename columns for consistency
    results_df.rename(columns={
        "dataset": "Dataset", 
        "method": "Method", 
        "imbalance_ratio": "Imbalance Ratio",
        "encoding_method": "Encoding Method",
        "archetype_setting": "Archetype Setting",
        "minority_sample_setting": "Minority Sample Setting"
    }, inplace=True)

    if "Minority Sample Setting" in results_df.columns:
        results_df["Minority Sample Setting"] = results_df["Minority Sample Setting"].astype(str)

    if "Archetype Setting" in results_df.columns:
        results_df["Archetype Setting"] = results_df["Archetype Setting"].astype(str)

    # Ensure missing columns are added with default values
    for col in ["Archetype Setting", "Minority Sample Setting"]:
        if col not in results_df.columns:
            print(f"[DEBUG] Adding missing column: {col}")  
            results_df[col] = "N/A"

    print("\n[DEBUG] Final Columns in Cleaned DataFrame:", results_df.columns)  # Debugging
    return results_df


def plot_f1_scores(results_df):
    """
    Visualizes the weighted F1 scores for each dataset, grouped by resampling method and imbalance ratio.
    """
    datasets = results_df["Dataset"].unique()
    for dataset in datasets:
        df = results_df[results_df["Dataset"] == dataset]
        
        pivot = df.groupby(["Method", "Imbalance Ratio"])["Weighted F1 Score"].mean().unstack("Imbalance Ratio")
        pivot = pivot.fillna(0)

        ax = pivot.plot(kind="bar", figsize=(12, 7))
        plt.title(f"F1 Score Comparison for {dataset} (Averaged Over Encoding Methods)", fontsize=14)
        plt.xlabel("Resampling Method", fontsize=12)
        plt.ylabel("Weighted F1 Score", fontsize=12)
        plt.ylim(0, 1)
        plt.legend(title="Imbalance Ratio")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()

def plot_f1_by_encoding(results_df):
    """
    Visualizes the impact of encoding method on F1 score.
    """
    datasets = results_df["Dataset"].unique()
    for dataset in datasets:
        df = results_df[results_df["Dataset"] == dataset]

        pivot = df.groupby(["Method", "Encoding Method"])["Weighted F1 Score"].mean().unstack("Encoding Method")
        pivot = pivot.fillna(0)

        ax = pivot.plot(kind="bar", figsize=(12, 7))
        plt.title(f"F1 Score Comparison for {dataset} (Separated by Encoding Method)", fontsize=14)
        plt.xlabel("Resampling Method", fontsize=12)
        plt.ylabel("Weighted F1 Score", fontsize=12)
        plt.ylim(0, 1)
        plt.legend(title="Encoding Method")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()

def plot_f1_by_archetype_setting(results_df):
    """
    Visualizes the impact of different archetype settings on F1 Score.
    """
    datasets = results_df["Dataset"].unique()
    for dataset in datasets:
        df = results_df[results_df["Dataset"] == dataset]

        pivot = df.groupby(["Method", "Archetype Setting"])["Weighted F1 Score"].mean().unstack("Archetype Setting")
        pivot = pivot.fillna(0)

        ax = pivot.plot(kind="bar", figsize=(12, 7))
        plt.title(f"F1 Score Comparison for {dataset} (Separated by Archetype Setting)", fontsize=14)
        plt.xlabel("Resampling Method", fontsize=12)
        plt.ylabel("Weighted F1 Score", fontsize=12)
        plt.ylim(0, 1)
        plt.legend(title="Archetype Setting")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()

def plot_f1_by_minority_sample_setting(results_df):
    """
    Visualizes the impact of different minority sample settings on F1 Score.
    """
    datasets = results_df["Dataset"].unique()
    for dataset in datasets:
        df = results_df[results_df["Dataset"] == dataset]

        pivot = df.groupby(["Method", "Minority Sample Setting"])["Weighted F1 Score"].mean().unstack("Minority Sample Setting")
        pivot = pivot.fillna(0)

        ax = pivot.plot(kind="bar", figsize=(12, 7))
        plt.title(f"F1 Score Comparison for {dataset} (Separated by Minority Sample Setting)", fontsize=14)
        plt.xlabel("Resampling Method", fontsize=12)
        plt.ylabel("Weighted F1 Score", fontsize=12)
        plt.ylim(0, 1)
        plt.legend(title="Minority Sample Setting")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    results_file = "experiment_results.pkl"

    if os.path.exists(results_file):
        results_df = load(results_file)
        print("Loaded experiment results from", results_file)

        print(results_df[["Dataset", "Encoding Method", "Method", "Imbalance Ratio", 
                          "Archetype Setting", "Minority Sample Setting", "Weighted F1 Score"]])

        # Run all visualizations
        plot_f1_scores(results_df)
        plot_f1_by_encoding(results_df)
        plot_f1_by_archetype_setting(results_df)
        plot_f1_by_minority_sample_setting(results_df)

    else:
        print(f"Results file '{results_file}' not found. Please run runner.py first to generate experiment results.")
