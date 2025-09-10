from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# comparison methods
PRIMARY_A = "RFOversample"
PRIMARY_B = "SMOTE"

CSV_PATH = "my_experiment_results.csv"
OUT_DIR = Path("graphs")
OUT_DIR.mkdir(exist_ok=True)
PER_DS_DIR = OUT_DIR / "per_dataset2"
PER_DS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- helpers ----------
def ensure_num(s):
    return pd.to_numeric(s, errors="coerce")

def se_from_sd_n(sd, n):
    n = max(int(n), 1)
    return sd / np.sqrt(n)

def welch_t_from_summary(m1, sd1, n1, m2, sd2, n2):
    """Welch's t-test from summary (two-sided). Returns (t, df, p)."""
    from scipy.stats import t
    v1 = (sd1**2) / n1
    v2 = (sd2**2) / n2
    se = np.sqrt(v1 + v2)
    if se == 0:
        return np.nan, np.nan, np.nan
    t_stat = (m1 - m2) / se
    df = (v1 + v2)**2 / ((v1**2)/(n1-1) + (v2**2)/(n2-1)) if n1 > 1 and n2 > 1 else np.nan
    p = 2 * (1 - t.cdf(abs(t_stat), df)) if df == df else np.nan
    return t_stat, df, p

# ---------- load & filter ----------
df = pd.read_csv(CSV_PATH)

required = ["Dataset", "Method", "Mean_Weighted_F1", "Standard_Deviation", "Seed_Count"]
for c in required:
    if c not in df.columns:
        raise ValueError(f"Missing required column: {c}")
if "Imbalance_Ratio" not in df.columns:
    df["Imbalance_Ratio"] = np.nan

# Coerce numerics
df["Mean_Weighted_F1"]   = ensure_num(df["Mean_Weighted_F1"])
df["Standard_Deviation"] = ensure_num(df["Standard_Deviation"])
df["Seed_Count"]         = ensure_num(df["Seed_Count"])
df["Imbalance_Ratio"]    = ensure_num(df["Imbalance_Ratio"])

# keep only the two methods of interest
df2 = df[df["Method"].isin([PRIMARY_A, PRIMARY_B])].copy()

# ---------- compute per-dataset×ratio deltas, CI, Welch ----------
rows = []
group_cols = ["Dataset", "Imbalance_Ratio"]
for (ds, ratio), g in df2.groupby(group_cols, dropna=False):
    A = g[g["Method"] == PRIMARY_A]
    B = g[g["Method"] == PRIMARY_B]
    if A.empty or B.empty:
        continue

    mA, sdA, nA = float(A["Mean_Weighted_F1"].values[0]), float(A["Standard_Deviation"].values[0]), int(A["Seed_Count"].values[0])
    mB, sdB, nB = float(B["Mean_Weighted_F1"].values[0]), float(B["Standard_Deviation"].values[0]), int(B["Seed_Count"].values[0])
    seA, seB = se_from_sd_n(sdA, nA), se_from_sd_n(sdB, nB)

    delta = mA - mB
    se_delta = np.sqrt(seA**2 + seB**2)
    ci95 = 1.96 * se_delta

    t_stat, df_w, p_val = welch_t_from_summary(mA, sdA, nA, mB, sdB, nB)

    rows.append({
        "Dataset": ds,
        "Imbalance_Ratio": ratio,
        f"{PRIMARY_A}_mean": mA,
        f"{PRIMARY_A}_SD": sdA,
        f"{PRIMARY_A}_SE": seA,
        f"{PRIMARY_B}_mean": mB,
        f"{PRIMARY_B}_SD": sdB,
        f"{PRIMARY_B}_SE": seB,
        "Delta_A_minus_B": delta,
        "SE_Delta": se_delta,
        "CI95_Delta": ci95,
        "Welch_t": t_stat,
        "Welch_df": df_w,
        "Welch_p": p_val,
        "Significant_p<0.05": (p_val < 0.05) if p_val == p_val else False
    })

delta_df = pd.DataFrame(rows).sort_values(["Imbalance_Ratio", "Delta_A_minus_B"])
out_csv = OUT_DIR / f"delta_{PRIMARY_A}_vs_{PRIMARY_B}.csv"
delta_df.to_csv(out_csv, index=False)
print(f"[Saved] Δ table -> {out_csv.resolve()}")

# ---------- quick win-rate summary ----------
summary = (
    delta_df
    .assign(Win = np.where(delta_df["Delta_A_minus_B"] > 0, 1, 0),
            Win_sig = np.where((delta_df["Delta_A_minus_B"] > 0) & (delta_df["Welch_p"] < 0.05), 1, 0))
    .groupby("Imbalance_Ratio", dropna=False)[["Win","Win_sig"]]
    .mean()
    .rename(columns={"Win":"WinRate_A>B", "Win_sig":"WinRate_A>B_significant"})
)
print("\nWin rates by ratio (fraction of datasets):")
print(summary.to_string(float_format=lambda x: f"{x:.2f}"))

# ---------- plots: forest-style Δ with 95% CI (per ratio) ----------
for ratio, g in delta_df.groupby("Imbalance_Ratio", dropna=False):
    g = g.sort_values("Delta_A_minus_B")
    labels = g["Dataset"].tolist()
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35*len(labels))))
    ax.barh(y, g["Delta_A_minus_B"].to_numpy())
    ax.errorbar(g["Delta_A_minus_B"].to_numpy(), y, xerr=g["CI95_Delta"].to_numpy(),
                fmt="none", capsize=2)
    ax.axvline(0.0, color="k", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    title = f"{PRIMARY_A} − {PRIMARY_B} (Δ F1, 95% CI)" + (f" | Imbalance={ratio:g}" if ratio == ratio else "")
    ax.set_title(title)
    ax.set_xlabel("Δ Mean Weighted F1 (A − B)")
    plt.tight_layout()
    fname = OUT_DIR / f"delta_{PRIMARY_A}_minus_{PRIMARY_B}__imb_{'NA' if ratio != ratio else str(ratio).replace('.','p')}.png"
    plt.savefig(fname, dpi=200)
    plt.close()
    print(f"[Saved] {fname}")

# ---------- plots: per-dataset two-bar (A vs B) with ±SE and AUTO-ZOOM Y ----------
for (ds, ratio), g in df2.groupby(group_cols, dropna=False):
    A = g[g["Method"] == PRIMARY_A]
    B = g[g["Method"] == PRIMARY_B]
    if A.empty or B.empty:
        continue

    means = np.array([float(A["Mean_Weighted_F1"].values[0]), float(B["Mean_Weighted_F1"].values[0])], dtype=float)
    ses   = np.array([
        se_from_sd_n(float(A["Standard_Deviation"].values[0]), int(A["Seed_Count"].values[0])),
        se_from_sd_n(float(B["Standard_Deviation"].values[0]), int(B["Seed_Count"].values[0]))
    ], dtype=float)
    labels = [PRIMARY_A, PRIMARY_B]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, means, yerr=ses, capsize=3)

    # ---- AUTO-ZOOM Y-AXIS (with safety clamps and minimum span) ----
    ymin = float(np.min(means - ses))
    ymax = float(np.max(means + ses))
    # pad by 5% of range (or small epsilon if flat)
    rng = max(ymax - ymin, 1e-6)
    pad = 0.05 * rng
    y_low = max(0.0, ymin - pad)
    y_high = min(1.0, ymax + pad)
    # ensure a minimum visible span (e.g., 0.05) so error bars are readable
    min_span = 0.05
    if (y_high - y_low) < min_span:
        mid = 0.5 * (y_high + y_low)
        y_low = max(0.0, mid - 0.5 * min_span)
        y_high = min(1.0, mid + 0.5 * min_span)
    ax.set_ylim(y_low, y_high)

    ax.set_ylabel("Mean Weighted F1 (±SE)")
    title = f"{ds}" + (f" | Imbalance={ratio:g}" if ratio == ratio else "")
    ax.set_title(title)
    plt.tight_layout()
    fname = PER_DS_DIR / f"{ds}__imb_{'NA' if ratio != ratio else str(ratio).replace('.','p')}_{PRIMARY_A}_vs_{PRIMARY_B}.png"
    plt.savefig(fname, dpi=200)
    plt.close()

print(f"[Done] Figures saved under {OUT_DIR.resolve()} and {PER_DS_DIR.resolve()}")
