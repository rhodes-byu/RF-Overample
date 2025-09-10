# mixedresults.py  — numeric-only summaries (no plots)
from pathlib import Path
import numpy as np
import pandas as pd
from itertools import combinations

CSV_PATH = "my_experiment_results.csv"
OUT_DIR = Path("graphs")  # keep existing path; we'll only write CSVs here
OUT_DIR.mkdir(exist_ok=True, parents=True)

# ---------- helpers ----------
def ensure_num(s: pd.Series):
    return pd.to_numeric(s, errors="coerce")

def se_from_sd_n(sd, n):
    n = max(int(n), 1)
    return sd / np.sqrt(n)

def anova_from_summary(means, sds, ns, labels):
    """
    Classic one-way ANOVA computed from summary stats.
    Returns dict with SS_between, SS_within, df_between, df_within, F, p (NaN if invalid).
    """
    import numpy as np
    k = len(means)
    if k < 2:
        return None
    N = np.sum(ns)
    gm = np.sum(ns * means) / N
    ss_between = np.sum(ns * (means - gm) ** 2)
    ss_within = np.sum((ns - 1) * (sds ** 2))
    df_between = k - 1
    df_within = N - k
    ms_between = ss_between / df_between if df_between > 0 else np.nan
    ms_within = ss_within / df_within if df_within > 0 else np.nan
    F = ms_between / ms_within if (ms_within is not None and ms_within > 0) else np.nan
    try:
        from scipy.stats import f
        p = 1 - f.cdf(F, df_between, df_within) if not (np.isnan(F) or np.isnan(df_between) or np.isnan(df_within)) else np.nan
    except Exception:
        p = np.nan
    return {
        "k": int(k),
        "N": int(N),
        "grand_mean": float(gm),
        "SS_between": float(ss_between),
        "SS_within": float(ss_within),
        "df_between": int(df_between),
        "df_within": int(df_within),
        "F": float(F) if F == F else np.nan,
        "p_value": float(p) if p == p else np.nan,
        "labels": list(labels),
    }

def welch_t_from_summary(m1, sd1, n1, m2, sd2, n2):
    """Welch's t-test from summary data; returns (t, df, p) two-sided."""
    from scipy.stats import t
    v1 = (sd1 ** 2) / n1
    v2 = (sd2 ** 2) / n2
    se = np.sqrt(v1 + v2)
    if se == 0:
        return np.nan, np.nan, np.nan
    t_stat = (m1 - m2) / se
    df = (v1 + v2) ** 2 / ((v1 ** 2) / (n1 - 1) + (v2 ** 2) / (n2 - 1)) if n1 > 1 and n2 > 1 else np.nan
    p = 2 * (1 - t.cdf(abs(t_stat), df)) if df == df else np.nan
    return t_stat, df, p

def holm_bonferroni(pvals):
    """Holm-Bonferroni step-down adjustment. Returns adjusted p-values aligned to input order."""
    pvals = np.array(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    prev = 0.0
    for i, idx in enumerate(order):
        adj_p = (m - i) * pvals[idx]
        adj[idx] = max(prev, min(1.0, adj_p))
        prev = adj[idx]
    return adj

# ---------- load ----------
df = pd.read_csv(CSV_PATH)

print("[Info] SD (sample) computed with ddof=1 across seeds; SE = SD / sqrt(n).")

# required columns & coercion
required = ["Dataset", "Method", "Mean_Weighted_F1", "Standard_Deviation", "Seed_Count"]
for c in required:
    if c not in df.columns:
        raise ValueError(f"Missing required column: {c}")
if "Imbalance_Ratio" not in df.columns:
    df["Imbalance_Ratio"] = np.nan

df["Mean_Weighted_F1"]   = ensure_num(df["Mean_Weighted_F1"])
df["Standard_Deviation"] = ensure_num(df["Standard_Deviation"])
df["Seed_Count"]         = ensure_num(df["Seed_Count"])
df["Imbalance_Ratio"]    = ensure_num(df["Imbalance_Ratio"])
df["Standard_Error"]     = ensure_num(df.get("Standard_Error", df["Standard_Deviation"] / np.sqrt(df["Seed_Count"].replace(0, np.nan))))

# ---------- per-stratum (Dataset × Imbalance_Ratio) ANOVA + pairwise Welch ----------
anova_rows = []
pairwise_rows = []

group_cols = ["Dataset", "Imbalance_Ratio"]
for (ds, ratio), g in df.groupby(group_cols, dropna=False):
    g = g.dropna(subset=["Mean_Weighted_F1"])
    if g.empty:
        continue

    methods = g["Method"].tolist()
    means = g["Mean_Weighted_F1"].to_numpy(dtype=float)
    sds   = g["Standard_Deviation"].fillna(0.0).to_numpy(dtype=float)
    ns    = g["Seed_Count"].to_numpy(dtype=int)

    # ANOVA from summary
    res = anova_from_summary(means, sds, ns, methods)
    if res:
        row = {"Dataset": ds, "Imbalance_Ratio": ratio}
        row.update({k: v for k, v in res.items() if k not in ("labels",)})
        anova_rows.append(row)

    # Pairwise Welch with Holm
    idx_pairs = list(combinations(range(len(methods)), 2))
    pvals = []
    tmp = []
    for i, j in idx_pairs:
        t_stat, df_w, p = welch_t_from_summary(means[i], sds[i], ns[i], means[j], sds[j], ns[j])
        tmp.append((methods[i], methods[j], t_stat, df_w, p))
        pvals.append(p if p == p else 1.0)
    if pvals:
        adj = holm_bonferroni(pvals)
        for (mi, mj, t_stat, df_w, p), p_adj in zip(tmp, adj):
            pairwise_rows.append({
                "Dataset": ds,
                "Imbalance_Ratio": ratio,
                "Method_A": mi,
                "Method_B": mj,
                "Welch_t": t_stat,
                "Welch_df": df_w,
                "p_value": p,
                "p_value_holm": float(p_adj),
                "Significant_p<0.05_Holm": (p_adj < 0.05) if p_adj == p_adj else False
            })

anova_df = pd.DataFrame(anova_rows)
pairwise_df = pd.DataFrame(pairwise_rows)

# ---------- overall method summaries across all strata ----------
# 1) Per-method distribution stats
meth_stats = (
    df.groupby("Method", as_index=False)["Mean_Weighted_F1"]
      .agg(n="count", mean="mean", std="std", median="median",
           q25=lambda s: np.nanpercentile(s, 25),
           q75=lambda s: np.nanpercentile(s, 75))
)
meth_stats["iqr"] = meth_stats["q75"] - meth_stats["q25"]

# 2) Per-stratum rank (higher F1 = better rank 1), then summarize ranks
ranked = df.copy()
ranked["Rank"] = ranked.groupby(group_cols, dropna=False)["Mean_Weighted_F1"]\
                       .rank(method="average", ascending=False)
rank_summary = (
    ranked.groupby("Method", as_index=False)["Rank"]
          .agg(mean_rank="mean", std_rank="std", median_rank="median", n_strata="count")
          .sort_values("mean_rank", ascending=True)
)

# 3) Top-k counts (k=1,3) — counts over strata
def kth_smallest(arr, k):
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return np.nan
    k = max(1, min(k, a.size))
    return np.partition(a, k - 1)[k - 1]

cutoff_k1 = ranked.groupby(group_cols, dropna=False)["Rank"].transform(lambda s: kth_smallest(s.values, 1))
cutoff_k3 = ranked.groupby(group_cols, dropna=False)["Rank"].transform(lambda s: kth_smallest(s.values, 3))
ranked["Top1"] = (ranked["Rank"] <= cutoff_k1).astype(int)
ranked["Top3"] = (ranked["Rank"] <= cutoff_k3).astype(int)
topk = ranked.groupby("Method", as_index=False)[["Top1", "Top3"]].sum().sort_values(["Top1", "Top3"], ascending=[False, False])

# 4) Head-to-head win / tie / loss counts by strata (using mean F1)
methods = sorted(df["Method"].unique().tolist())
m_index = {m: i for i, m in enumerate(methods)}
# initialize counts
win = np.zeros((len(methods), len(methods)), dtype=int)
tie = np.zeros_like(win)
loss = np.zeros_like(win)

for _, g in df.groupby(group_cols, dropna=False):
    # per stratum means by method
    s = g.set_index("Method")["Mean_Weighted_F1"]
    for i, j in combinations(s.index, 2):
        mi, mj = m_index[i], m_index[j]
        if pd.isna(s[i]) or pd.isna(s[j]):
            continue
        if np.isclose(s[i], s[j], atol=1e-12):
            tie[mi, mj] += 1
            tie[mj, mi] += 1
        elif s[i] > s[j]:
            win[mi, mj] += 1
            loss[mj, mi] += 1
        else:
            win[mj, mi] += 1
            loss[mi, mj] += 1

head2head = []
for i, a in enumerate(methods):
    for j, b in enumerate(methods):
        if i == j: 
            continue
        head2head.append({
            "Method_A": a,
            "Method_B": b,
            "Wins_A_over_B": int(win[i, j]),
            "Ties": int(tie[i, j]),
            "Losses_A_to_B": int(loss[i, j]),
        })
head2head_df = pd.DataFrame(head2head)

# 5) Head-to-head SIGNIFICANT win counts (Welch Holm p<0.05)
sig_pairs = pairwise_df[pairwise_df["Significant_p<0.05_Holm"] == True]
sig_counts = (
    sig_pairs.groupby(["Method_A", "Method_B"], as_index=False)
             .size()
             .rename(columns={"size": "Sig_Wins_A_over_B"})
)
# Make a square matrix-like listing (optional long form)
sig_full = []
for a in methods:
    for b in methods:
        if a == b:
            continue
        n_sig = int(sig_counts.loc[(sig_counts["Method_A"]==a) & (sig_counts["Method_B"]==b), "Sig_Wins_A_over_B"].sum())
        sig_full.append({"Method_A": a, "Method_B": b, "Sig_Wins_A_over_B": n_sig})
sig_head2head_df = pd.DataFrame(sig_full)

# ---------- write CSV outputs ----------
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

anova_path       = RESULTS_DIR / "anova_by_dataset.csv"
pairwise_path    = RESULTS_DIR / "pairwise_welch_by_dataset_holm.csv"
stats_path       = RESULTS_DIR / "overall_method_stats.csv"
ranks_path       = RESULTS_DIR / "overall_method_ranks.csv"
topk_path        = RESULTS_DIR / "overall_method_topk_counts.csv"
h2h_path         = RESULTS_DIR / "headtohead_counts.csv"
h2h_sig_path     = RESULTS_DIR / "headtohead_significant_counts.csv"

anova_df.to_csv(anova_path, index=False)
pairwise_df.to_csv(pairwise_path, index=False)
meth_stats.to_csv(stats_path, index=False)
rank_summary.to_csv(ranks_path, index=False)
topk.to_csv(topk_path, index=False)
head2head_df.to_csv(h2h_path, index=False)
sig_head2head_df.to_csv(h2h_sig_path, index=False)

print(f"[Saved] ANOVA table -> {anova_path.resolve()}")
print(f"[Saved] Pairwise Welch (Holm) -> {pairwise_path.resolve()}")
print(f"[Saved] Method stats -> {stats_path.resolve()}")
print(f"[Saved] Rank summary -> {ranks_path.resolve()}")
print(f"[Saved] Top-k counts -> {topk_path.resolve()}")
print(f"[Saved] Head-to-head -> {h2h_path.resolve()}")
print(f"[Saved] Head-to-head (significant) -> {h2h_sig_path.resolve()}")
