#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mixedresults2.py (SD-only)
- Loads experiment results
- Aggregates to one value per (Dataset, Method)
- Produces:
    1) Summary Δ vs SMOTE (per dataset, averaged — trivial if single ratio)
    2) Method performance distribution across datasets (box + jitter with ±SD)
    3) Average rank across datasets (lower is better)
    4) Per-dataset bar charts comparing methods (Mean ± SD)
Notes:
- This script uses ONLY Standard Deviation (SD) for variability visualization.
- It assumes you're currently working with a single imbalance ratio; aggregation
  still works if more are added later.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# --- Config ---
# =========================
CSV_PATH = "my_experiment_results.csv"
OUT_DIR = Path("graphs")
PER_DATASET_DIR = OUT_DIR / "per_dataset"
OUT_DIR.mkdir(exist_ok=True)
PER_DATASET_DIR.mkdir(parents=True, exist_ok=True)

# Consistent method order for plots
order_methods = ["Baseline", "Unbalanced", "SMOTE", "RFOversample"]

# =========================
# --- Load ---
# =========================
df = pd.read_csv(CSV_PATH)

# Require SD (we standardize on SD-only)
required = {"Dataset", "Method", "Mean_Weighted_F1", "Standard_Deviation"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"CSV missing required columns: {missing}")

# Optional columns we won't rely on for plotting, but keep if present
if "Imbalance_Ratio" not in df.columns:
    df["Imbalance_Ratio"] = np.nan

# Harmonize method names
df["Method"] = df["Method"].replace({
    "OversamplerJ": "RFOversample",
    "RFoversample": "RFOversample",
    "rfoversampler": "RFOversample",
    "SMOTE": "SMOTE",
    "Unbalanced": "Unbalanced",
    "Baseline": "Baseline",
})

# =========================
# Aggregation helpers
# =========================
def combine_mean_sd(g: pd.DataFrame) -> pd.Series:
    """
    Combine potentially multiple rows per (Dataset, Method) (e.g., multiple runs/ratios)
    Mean = arithmetic mean of Mean_Weighted_F1
    SD   = RMS of Standard_Deviation (conservative combination across rows)
    """
    m = g["Mean_Weighted_F1"].mean()
    sd = np.sqrt(np.nanmean(np.square(g["Standard_Deviation"].fillna(0.0))))
    return pd.Series({"Mean": m, "SD": sd})

# Aggregate to one row per (Dataset, Method)
agg_dataset_method = (
    df.groupby(["Dataset", "Method"], dropna=False)
      .apply(combine_mean_sd, include_groups=False)
      .reset_index()
)

# =========================
# 1) Summary: Δ vs SMOTE (per dataset)
# =========================
smote = agg_dataset_method[agg_dataset_method["Method"] == "SMOTE"][["Dataset", "Mean"]].set_index("Dataset")
rf    = agg_dataset_method[agg_dataset_method["Method"] == "RFOversample"][["Dataset", "Mean"]].set_index("Dataset")

if not smote.empty and not rf.empty:
    delta = (rf["Mean"] - smote["Mean"]).dropna().sort_values()
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(delta)) if len(delta) else 4))
    colors = np.where(delta.values >= 0, "tab:green", "tab:red") if len(delta) else []
    ax.barh(delta.index if len(delta) else [], delta.values if len(delta) else [], color=colors)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("RFOversample − SMOTE (Mean Weighted F1)")
    ax.set_ylabel("Dataset")
    ax.set_title("Improvement of RFOversample over SMOTE (per dataset)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "summary_delta_vs_smote.png", dpi=200)
    plt.close()

# =========================
# 2) Method performance distribution across datasets (box + jitter with ±SD)
# =========================
summ = agg_dataset_method.copy()
summ["Method"] = pd.Categorical(summ["Method"], categories=order_methods, ordered=True)

data = [summ.loc[summ["Method"] == m, "Mean"].dropna().values for m in order_methods]
fig, ax = plt.subplots(figsize=(8, 5))
positions = np.arange(1, len(order_methods) + 1)
ax.boxplot(data, positions=positions, showfliers=False)

# jittered points with ±SD per dataset-method
for i, m in enumerate(order_methods, start=1):
    sub = summ.loc[summ["Method"] == m, ["Mean", "SD"]].dropna(subset=["Mean"])
    if sub.empty:
        continue
    y = sub["Mean"].to_numpy()
    sd = sub["SD"].to_numpy()
    x = np.random.normal(loc=i, scale=0.05, size=len(y))
    ax.plot(x, y, "o", alpha=0.6)
    ax.errorbar(x, y, yerr=sd, fmt="none", elinewidth=1, capsize=2, alpha=0.6)

ax.set_xticks(positions)
ax.set_xticklabels(order_methods)
ax.set_ylabel("Mean Weighted F1 (per dataset)")
ax.set_title("Method performance distribution across datasets (Mean ± SD)")
plt.tight_layout()
plt.savefig(OUT_DIR / "summary_method_distribution.png", dpi=200)
plt.close()

# =========================
# 3) Average ranks across datasets (lower is better)
# =========================
def ranks_per_dataset(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    g["Rank"] = (-g["Mean"]).rank(method="average")  # rank 1 = best
    return g

ranked = agg_dataset_method.groupby("Dataset", dropna=False)\
    .apply(ranks_per_dataset, include_groups=False).reset_index(drop=True)

avg_ranks = (
    ranked.groupby("Method", dropna=False)["Rank"].mean()
          .reindex(order_methods)
          .dropna()
)

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(avg_ranks.index.astype(str), avg_ranks.values)
ax.set_ylabel("Average rank (lower is better)")
ax.set_title("Average method rank across datasets")
for i, v in enumerate(avg_ranks.values):
    ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR / "summary_average_ranks.png", dpi=200)
plt.close()

# =========================
# 4) Per-dataset bars (Mean ± SD)
#     - Clean when only a single imbalance ratio is used (no lines/ratios)
# =========================
for ds, g in agg_dataset_method.groupby("Dataset", dropna=False):
    g = g.copy()
    g["Method"] = pd.Categorical(g["Method"], categories=order_methods, ordered=True)
    g = g.sort_values("Method").dropna(subset=["Mean"])
    if g.empty:
        continue

    fig, ax = plt.subplots(figsize=(7, 4))
    methods = g["Method"].astype(str).tolist()
    means = g["Mean"].to_numpy()
    sds   = g["SD"].to_numpy()

    ax.bar(methods, means, yerr=sds, capsize=3)
    ax.set_ylim(0, 1)  # adjust if your metric differs
    ax.set_ylabel("Mean Weighted F1 (±SD)")
    ax.set_title(ds)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(PER_DATASET_DIR / f"{ds}_methods_mean_sd.png", dpi=200)
    plt.close()

print(f"Saved figures in: {OUT_DIR.resolve()}")
print(f"Per-dataset figures in: {PER_DATASET_DIR.resolve()}")
