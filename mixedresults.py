import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ===== CONFIG =====
results_csv = "my_experiment_results.csv"  # path to your results CSV
save_fig_path = "experiment_results.png"   # where to save the plot
figsize = (14, 10)                         # size of the whole figure
# ==================

# Load results
df = pd.read_csv(results_csv)

# Ensure proper sorting
df.sort_values(by=["Dataset", "Imbalance_Ratio", "Method"], inplace=True)

# Get unique datasets and imbalance ratios
datasets = df["Dataset"].unique()
imbalance_ratios = sorted(df["Imbalance_Ratio"].dropna().unique())

# Create figure
fig, axes = plt.subplots(len(datasets), 1, figsize=figsize, sharex=True)
if len(datasets) == 1:
    axes = [axes]  # make iterable for 1 dataset

# Plot for each dataset
for ax, dataset in zip(axes, datasets):
    subset = df[df["Dataset"] == dataset]

    # Baseline
    baseline_row = subset[subset["Method"] == "Baseline"]
    if not baseline_row.empty:
        baseline_f1 = baseline_row["Mean_Weighted_F1"].values[0]
        ax.axhline(y=baseline_f1, color="gray", linestyle="--", label="Baseline")

    # Methods (excluding baseline)
    methods = subset["Method"].unique()
    methods = [m for m in methods if m != "Baseline"]

    for method in methods:
        method_data = subset[subset["Method"] == method]
        ax.errorbar(
            method_data["Imbalance_Ratio"],
            method_data["Mean_Weighted_F1"],
            yerr=method_data["Standard_Error"],
            label=method,
            marker="o",
            capsize=4
        )

    ax.set_title(f"Dataset: {dataset}")
    ax.set_ylabel("Weighted F1 Score")
    ax.legend()

# Shared X label
axes[-1].set_xlabel("Imbalance Ratio")

plt.tight_layout()
plt.savefig(save_fig_path, dpi=300)
plt.show()
