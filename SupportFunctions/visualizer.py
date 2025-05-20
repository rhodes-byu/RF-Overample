import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load

def process_plot(save_fig=False, output_dir='graphs', filename='plot.png'):
    if save_fig:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath)
        print(f"Graph saved to {filepath}")
        plt.close()
    else:
        plt.show()

def extract_f1_score(report_df):
    if not isinstance(report_df, pd.DataFrame):
        print("[WARN] Report is not a DataFrame.")
        return np.nan
    if "weighted avg" not in report_df.index:
        print("[WARN] Missing 'weighted avg' in classification report.")
        print(report_df)
        return np.nan
    return report_df.loc["weighted avg", "f1-score"]


def clean_results(results):
    valid_results = [res for res in results if "classification_report" in res]
    results_df = pd.DataFrame(valid_results)

    results_df["Weighted F1 Score"] = results_df["classification_report"].apply(extract_f1_score)

    results_df.rename(columns={
        "dataset": "Dataset",
        "method": "Method",
        "imbalance_ratio": "Imbalance Ratio",
        "encoding_method": "Encoding Method",
        "archetype_setting": "Archetype Setting",
        "minority_sample_setting": "Minority Sample Setting",
        "use_archetypes": "Use Archetypes"
    }, inplace=True)

    if "Minority Sample Setting" in results_df.columns:
        results_df["Minority Sample Setting"] = results_df["Minority Sample Setting"].astype(str)
    if "Archetype Setting" in results_df.columns:
        results_df["Archetype Setting"] = results_df["Archetype Setting"].astype(str)

    for col in ["Archetype Setting", "Minority Sample Setting"]:
        if col not in results_df.columns:
            results_df[col] = "N/A"

    return results_df

def annotate_bars(ax):
    """
    Annotates each bar in a bar chart with its height (F1 score value).
    """
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=9)

def plot_f1_scores(results_df, save_fig=False, output_dir='graphs'):
    datasets = results_df["Dataset"].unique()
    for dataset in datasets:
        df = results_df[results_df["Dataset"] == dataset]
        pivot = df.groupby(["Method", "Imbalance Ratio"])["Weighted F1 Score"].mean().unstack("Imbalance Ratio").fillna(0)

        ax = pivot.plot(kind="bar", figsize=(12, 7))
        annotate_bars(ax)

        plt.title(f"F1 Score Comparison for {dataset} (Averaged Over Encoding Methods)", fontsize=14)
        plt.xlabel("Resampling Method", fontsize=12)
        plt.ylabel("Weighted F1 Score", fontsize=12)
        plt.ylim(0, 1)
        plt.legend(title="Imbalance Ratio")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()

        filename = f"f1_scores_{dataset}.png"
        process_plot(save_fig=save_fig, output_dir=output_dir, filename=filename)
        plt.clf()

def plot_f1_by_encoding(results_df, save_fig=False, output_dir='graphs'):
    datasets = results_df["Dataset"].unique()
    for dataset in datasets:
        df = results_df[results_df["Dataset"] == dataset]
        pivot = df.groupby(["Method", "Encoding Method"])["Weighted F1 Score"].mean().unstack("Encoding Method").fillna(0)

        ax = pivot.plot(kind="bar", figsize=(12, 7))
        annotate_bars(ax)

        plt.title(f"F1 Score Comparison for {dataset} (Separated by Encoding Method)", fontsize=14)
        plt.xlabel("Resampling Method", fontsize=12)
        plt.ylabel("Weighted F1 Score", fontsize=12)
        plt.ylim(0, 1)
        plt.legend(title="Encoding Method")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()

        filename = f"f1_by_encoding_{dataset}.png"
        process_plot(save_fig=save_fig, output_dir=output_dir, filename=filename)
        plt.clf()

def plot_f1_by_archetype_setting(results_df, save_fig=False, output_dir='graphs'):
    datasets = results_df["Dataset"].unique()
    for dataset in datasets:
        df = results_df[results_df["Dataset"] == dataset]
        pivot = df.groupby(["Method", "Archetype Setting"])["Weighted F1 Score"].mean().unstack("Archetype Setting").fillna(0)

        ax = pivot.plot(kind="bar", figsize=(12, 7))
        annotate_bars(ax)

        plt.title(f"F1 Score Comparison for {dataset} (Separated by Archetype Setting)", fontsize=14)
        plt.xlabel("Resampling Method", fontsize=12)
        plt.ylabel("Weighted F1 Score", fontsize=12)
        plt.ylim(0, 1)
        plt.legend(title="Archetype Setting")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()

        filename = f"f1_by_archetype_{dataset}.png"
        process_plot(save_fig=save_fig, output_dir=output_dir, filename=filename)
        plt.clf()

def plot_f1_by_minority_sample_setting(results_df, save_fig=False, output_dir='graphs'):
    datasets = results_df["Dataset"].unique()
    for dataset in datasets:
        df = results_df[results_df["Dataset"] == dataset]
        pivot = df.groupby(["Method", "Minority Sample Setting"])["Weighted F1 Score"].mean().unstack("Minority Sample Setting").fillna(0)

        ax = pivot.plot(kind="bar", figsize=(12, 7))
        annotate_bars(ax)

        plt.title(f"F1 Score Comparison for {dataset} (Separated by Minority Sample Setting)", fontsize=14)
        plt.xlabel("Resampling Method", fontsize=12)
        plt.ylabel("Weighted F1 Score", fontsize=12)
        plt.ylim(0, 1)
        plt.legend(title="Minority Sample Setting")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()

        filename = f"f1_by_minority_{dataset}.png"
        process_plot(save_fig=save_fig, output_dir=output_dir, filename=filename)
        plt.clf()

def plot_f1_by_use_of_archetypes(results_df, save_fig=False, output_dir='graphs'):
    datasets = results_df["Dataset"].unique()
    for dataset in datasets:
        df = results_df[results_df["Dataset"] == dataset]
        pivot = df.groupby(["Method", "Use Archetypes"])["Weighted F1 Score"].mean().unstack("Use Archetypes").fillna(0)

        ax = pivot.plot(kind="bar", figsize=(12, 7))
        annotate_bars(ax)

        plt.title(f"F1 Score Comparison for {dataset} (Use of Archetypes)", fontsize=14)
        plt.xlabel("Resampling Method", fontsize=12)
        plt.ylabel("Weighted F1 Score", fontsize=12)
        plt.ylim(0, 1)
        plt.legend(title="Use Archetypes")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()

        filename = f"f1_by_use_archetypes_{dataset}.png"
        process_plot(save_fig=save_fig, output_dir=output_dir, filename=filename)
        plt.clf()

if __name__ == "__main__":
    results_file = "experiment_results.pkl"

    if os.path.exists(results_file):
        results_df = load(results_file)
        print("Loaded experiment results from", results_file)

        print(results_df[["Dataset", "Encoding Method", "Method", "Imbalance Ratio",
                          "Archetype Setting", "Minority Sample Setting", "Weighted F1 Score"]])

        plot_f1_scores(results_df, save_fig=True)
        plot_f1_by_encoding(results_df, save_fig=True)
        plot_f1_by_archetype_setting(results_df, save_fig=True)
        plot_f1_by_minority_sample_setting(results_df, save_fig=True)
        plot_f1_by_use_of_archetypes(results_df, save_fig=True)
    else:
        print(f"Results file '{results_file}' not found. Please run runner.py first to generate experiment results.")
