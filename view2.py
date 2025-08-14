import os
import random
import numpy as np
import pandas as pd

from SupportFunctions.imbalancer import ImbalanceHandler
from imblearn.over_sampling import SMOTE

# Import BOTH oversamplers (unchanged files)
from rfoversample import RFOversampler as RFOversamplerLegacy
from rfoversampleJ import RFOversampler as RFOversamplerNew

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns


# ------------- I/O helpers -------------
def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def _savefig(path):
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print(f"[saved] {path}")


# ------------- data helpers -------------
def _one_hot_like_mixedrunner(df, categorical_indices):
    """Return X_oh, y, cat_columns (base names)"""
    target_col = df.columns[0]
    X = df.drop(columns=[target_col])
    y = df[target_col]
    cat_columns = [X.columns[i] for i in (categorical_indices or [])]
    if cat_columns:
        X[cat_columns] = X[cat_columns].astype("category")
    X_oh = pd.get_dummies(X, columns=cat_columns, dtype=int)
    return X_oh, y, cat_columns

def _load_dataset_entry(dataset_name):
    import joblib
    with open("prepared_datasets.pkl", "rb") as f:
        datasets = joblib.load(f)
    return datasets[dataset_name]

def _align_columns(*dfs):
    all_cols = sorted(set().union(*[df.columns for df in dfs]))
    return [df.reindex(columns=all_cols, fill_value=0) for df in dfs]

def _standardize_joint(*dfs):
    aligned = _align_columns(*dfs)
    mat = np.vstack([a.values for a in aligned])
    scaler = StandardScaler(with_mean=True, with_std=True)
    Z = scaler.fit_transform(mat)
    out, start = [], 0
    for a in aligned:
        n = len(a)
        out.append(pd.DataFrame(Z[start:start+n, :], columns=a.columns, index=a.index))
        start += n
    return out

def _extract_synthetic_tail(X_resampled, X_imb):
    n0 = len(X_imb)
    return X_resampled.iloc[n0:].reset_index(drop=True)


# ------------- plotting (custom labels + saving) -------------
def _pca_scatter(A, B, C, labels, title, outpath):
    # A/B/C are aligned & standardized DataFrames
    combined = pd.concat([
        A.assign(Source=labels[0]),
        B.assign(Source=labels[1]),
        C.assign(Source=labels[2])
    ], ignore_index=True)
    X = combined.drop(columns="Source").to_numpy()
    reduced = PCA(n_components=2, random_state=42).fit_transform(X)
    combined["PC1"] = reduced[:, 0]
    combined["PC2"] = reduced[:, 1]
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=combined, x="PC1", y="PC2", hue="Source", alpha=0.6, s=30)
    plt.title(title)
    _savefig(outpath)
    plt.show()

def _tsne_scatter(A, B, C, labels, title, outpath):
    combined = pd.concat([
        A.assign(Source=labels[0]),
        B.assign(Source=labels[1]),
        C.assign(Source=labels[2])
    ], ignore_index=True)
    X = combined.drop(columns="Source").to_numpy()
    # keep tsne stable
    perplexity = max(5, min(30, (len(X) // 3) - 1))
    reduced = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                   init="pca", learning_rate="auto").fit_transform(X)
    combined["Dim1"] = reduced[:, 0]
    combined["Dim2"] = reduced[:, 1]
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=combined, x="Dim1", y="Dim2", hue="Source", alpha=0.6, s=30)
    plt.title(f"{title} (t-SNE, perplexity={perplexity})")
    _savefig(outpath)
    plt.show()

def _nn_hist(A_orig_ref, B_new, C_new, labels, title, outpath):
    # A_orig_ref is the reference cloud to which we measure NN distances
    Xo = A_orig_ref.to_numpy()
    Xb = B_new.to_numpy()
    Xc = C_new.to_numpy()
    k = min(1, len(Xo)) or 1
    nbrs = NearestNeighbors(n_neighbors=k).fit(Xo)
    dist_b, _ = nbrs.kneighbors(Xb)
    dist_c, _ = nbrs.kneighbors(Xc)
    plt.figure(figsize=(10, 5))
    plt.hist(dist_b.flatten(), bins=30, alpha=0.6, label=labels[1])
    plt.hist(dist_c.flatten(), bins=30, alpha=0.6, label=labels[2])
    plt.title(title)
    plt.xlabel("Distance"); plt.ylabel("Count"); plt.legend()
    _savefig(outpath)
    plt.show()


# ------------- core pipeline for one dataset -------------
def run_one_dataset(dataset_name="titanic", imbalance_ratio=0.15, seed=42, outdir="graphs"):
    outdir = _ensure_dir(outdir)

    # 1) Load
    data_entry = _load_dataset_entry(dataset_name)
    df = data_entry["data"]
    categorical_indices = data_entry["categorical_indices"]

    class_counts = df.iloc[:, 0].value_counts()
    if class_counts.min() < 2:
        print(f"Skipping '{dataset_name}': insufficient class samples:\n{class_counts}")
        return

    # 2) Split + one-hot
    X_full_oh, y_full, cat_columns = _one_hot_like_mixedrunner(df, categorical_indices)
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        X_full_oh, y_full, test_size=0.3, stratify=y_full, random_state=seed
    )
    X_full_train = X_train_full.copy()

    # 3) Introduce imbalance
    imbalancer = ImbalanceHandler(
        X_train_full, y_train_full,
        imbalance_ratio=imbalance_ratio,
        batch_size=200,
        random_state=seed
    )
    X_imb, y_imb = imbalancer.introduce_imbalance()

    # Minority slice for reference
    minority_class = y_imb.value_counts().idxmin()
    X_minority_unbalanced = X_imb[y_imb == minority_class].reset_index(drop=True)

    # 4) SMOTE
    sm = SMOTE(random_state=seed)
    X_smote_all, y_smote_all = sm.fit_resample(X_imb, y_imb)
    X_smote_synth = _extract_synthetic_tail(X_smote_all, X_imb)

    # 5) RF Oversampler (Old)
    rfos_legacy = RFOversamplerLegacy(
        X_imb, y_imb,
        num_samples=3,
        contains_categoricals=bool(cat_columns),
        encoded=True,
        cat_cols=cat_columns
    )
    X_rf_old_all, y_rf_old_all = rfos_legacy.fit()
    X_rf_old_synth = _extract_synthetic_tail(X_rf_old_all, X_imb)

    # 6) RF Oversampler (New / J)
    rfos_new = RFOversamplerNew(
        X_imb, y_imb,
        contains_categoricals=bool(cat_columns),
        encoded=True,
        cat_cols=cat_columns,
        K=7, add_noise=True, noise_scale=0.15,
        cat_prob_sample=True, random_state=seed,
        enforce_domains=True, binary_strategy="bernoulli",
        hybrid_perturb_frac=0.25
    )
    X_rf_new_all, y_rf_new_all = rfos_new.fit()
    X_rf_new_synth = _extract_synthetic_tail(X_rf_new_all, X_imb)

    # 7) Standardize/align for fair visuals
    (X_full_train_z,
     X_minority_unbalanced_z,
     X_smote_synth_z,
     X_rf_old_synth_z,
     X_rf_new_synth_z) = _standardize_joint(
        X_full_train, X_minority_unbalanced, X_smote_synth, X_rf_old_synth, X_rf_new_synth
    )

    # --- VIEW A: Original vs RF-Old vs SMOTE ---
    labels_A = ["Original", "RF Oversample (Old)", "SMOTE"]
    base_A = f"{dataset_name}_orig_rfOld_smote"
    _pca_scatter(X_full_train_z, X_rf_old_synth_z, X_smote_synth_z,
                 labels_A, f"PCA – {dataset_name}: Original vs RF-Old vs SMOTE",
                 os.path.join(outdir, f"{base_A}_pca.png"))
    _tsne_scatter(X_full_train_z, X_rf_old_synth_z, X_smote_synth_z,
                  labels_A, f"t-SNE – {dataset_name}: Original vs RF-Old vs SMOTE",
                  os.path.join(outdir, f"{base_A}_tsne.png"))
    _nn_hist(X_full_train_z, X_rf_old_synth_z, X_smote_synth_z,
             labels_A, f"NN Distances to Original – {dataset_name}: RF-Old vs SMOTE",
             os.path.join(outdir, f"{base_A}_nn.png"))

    # --- VIEW B: Original vs RF-New vs SMOTE ---
    labels_B = ["Original", "RF Oversample (New)", "SMOTE"]
    base_B = f"{dataset_name}_orig_rfNew_smote"
    _pca_scatter(X_full_train_z, X_rf_new_synth_z, X_smote_synth_z,
                 labels_B, f"PCA – {dataset_name}: Original vs RF-New vs SMOTE",
                 os.path.join(outdir, f"{base_B}_pca.png"))
    _tsne_scatter(X_full_train_z, X_rf_new_synth_z, X_smote_synth_z,
                  labels_B, f"t-SNE – {dataset_name}: Original vs RF-New vs SMOTE",
                  os.path.join(outdir, f"{base_B}_tsne.png"))
    _nn_hist(X_full_train_z, X_rf_new_synth_z, X_smote_synth_z,
             labels_B, f"NN Distances to Original – {dataset_name}: RF-New vs SMOTE",
             os.path.join(outdir, f"{base_B}_nn.png"))

    # --- VIEW C: Original vs RF-Old vs RF-New ---
    labels_C = ["Original", "RF Oversample (Old)", "RF Oversample (New)"]
    base_C = f"{dataset_name}_orig_rfOld_rfNew"
    _pca_scatter(X_full_train_z, X_rf_old_synth_z, X_rf_new_synth_z,
                 labels_C, f"PCA – {dataset_name}: Original vs RF-Old vs RF-New",
                 os.path.join(outdir, f"{base_C}_pca.png"))
    _tsne_scatter(X_full_train_z, X_rf_old_synth_z, X_rf_new_synth_z,
                  labels_C, f"t-SNE – {dataset_name}: Original vs RF-Old vs RF-New",
                  os.path.join(outdir, f"{base_C}_tsne.png"))
    _nn_hist(X_full_train_z, X_rf_old_synth_z, X_rf_new_synth_z,
             labels_C, f"NN Distances to Original – {dataset_name}: RF-Old vs RF-New",
             os.path.join(outdir, f"{base_C}_nn.png"))

    # --- Bonus: focus on minority cloud ---
    labels_min = ["Unbalanced Minority", "RF Oversample (New)", "SMOTE"]
    base_min = f"{dataset_name}_minority_rfNew_smote"
    _pca_scatter(X_minority_unbalanced_z, X_rf_new_synth_z, X_smote_synth_z,
                 labels_min, f"PCA – {dataset_name}: Minority vs RF-New vs SMOTE",
                 os.path.join(outdir, f"{base_min}_pca.png"))
    _tsne_scatter(X_minority_unbalanced_z, X_rf_new_synth_z, X_smote_synth_z,
                  labels_min, f"t-SNE – {dataset_name}: Minority vs RF-New vs SMOTE",
                  os.path.join(outdir, f"{base_min}_tsne.png"))
    _nn_hist(X_minority_unbalanced_z, X_rf_new_synth_z, X_smote_synth_z,
             labels_min, f"NN Distances to Minority – {dataset_name}: RF-New vs SMOTE",
             os.path.join(outdir, f"{base_min}_nn.png"))


def main(datasets=None, imbalance_ratio=0.15, seed=42, outdir="graphs"):
    datasets = datasets or ["titanic", "heart_disease", "heart_failure", "sonar"]
    outdir = _ensure_dir(outdir)
    for ds in datasets:
        try:
            run_one_dataset(dataset_name=ds, imbalance_ratio=imbalance_ratio, seed=seed, outdir=outdir)
        except Exception as e:
            print(f"[WARN] Skipping {ds} due to error: {e}")


if __name__ == "__main__":
    main(datasets=["titanic", "heart_disease"], imbalance_ratio=0.15, seed=42, outdir="graphs")
