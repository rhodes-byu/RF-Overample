import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from rfgap import RFGAP
from SupportFunctions.imbalancer import ImbalanceHandler

# Customized Parameters
dataset_name = "titanic"
imbalance_ratio = 0.05
xlim = (-5, 5)
ylim = (-5, 5)

def plot_2d_scatter(X, y, title, filename, synthetic_points=None, minority_class=None):
    plt.figure(figsize=(8,6))
    colors = plt.colormaps.get_cmap('tab10')
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        if synthetic_points is not None:
            if isinstance(synthetic_points, (pd.DataFrame, pd.Series)):
                syn_df = synthetic_points
            else:
                syn_df = pd.DataFrame(synthetic_points, columns=X.columns)
            syn_2d = pca.transform(syn_df)
    else:
        X_2d = X.values if hasattr(X, "values") else X
        if synthetic_points is not None:
            if isinstance(synthetic_points, (pd.DataFrame, pd.Series)):
                syn_2d = synthetic_points.values
            else:
                syn_2d = synthetic_points
    classes = np.unique(y)
    for i, cls in enumerate(classes):
        cls_mask = (y == cls)
        plt.scatter(X_2d[cls_mask, 0], X_2d[cls_mask, 1],
                    label=f"Class {cls}", alpha=0.6, s=30, color=colors(i))
    if synthetic_points is not None:
        plt.scatter(syn_2d[:, 0], syn_2d[:, 1],
                    label="Synthetic Samples", c='red', marker='x', s=60)
    plt.title(title)
    plt.legend()
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.tight_layout()
    os.makedirs("graphs", exist_ok=True)
    plt.savefig(os.path.join("graphs", filename))
    plt.close()

def train_and_evaluate_rf(X_train, y_train, X_test, y_test, label):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    f1 = f1_score(y_test, preds, average='weighted')
    print(f"[F1 - {label}] Weighted F1 Score: {f1:.4f}")
    return f1

def run_experiment(seed, n_synthetic=100, ratio_smote=0.5, k_neighbors=5):
    with open("prepared_datasets.pkl", "rb") as f:
        datasets = joblib.load(f)

    data_entry = datasets[dataset_name]
    df = data_entry["data"]
    categorical_indices = data_entry["categorical_indices"]
    target_col = df.columns[0]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Ordinal encoding
    cat_columns = [X.columns[i] for i in categorical_indices]
    for col in cat_columns:
        X[col] = X[col].astype("category").cat.codes

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed)

    minority_class = y_train_full.value_counts().idxmin()

    plot_2d_scatter(X_train_full, y_train_full,
                    title="Original Training Data",
                    filename=f"original_train_seed{seed}.png",
                    minority_class=minority_class)

    results = {}
    results['Original'] = train_and_evaluate_rf(X_train_full, y_train_full, X_test, y_test, "Original")

    imbalancer = ImbalanceHandler(X_train_full, y_train_full,
                                  imbalance_ratio=imbalance_ratio,
                                  batch_size=200, random_state=seed)
    X_imb, y_imb = imbalancer.introduce_imbalance()

    plot_2d_scatter(X_imb, y_imb,
                    title="Imbalanced Training Data",
                    filename=f"imbalanced_train_seed{seed}.png",
                    minority_class=minority_class)

    results['Imbalanced'] = train_and_evaluate_rf(X_imb, y_imb, X_test, y_test, "Imbalanced Only")

    # Number of synthetic samples by SMOTE and Proximity based on ratio
    n_smote = int(n_synthetic * ratio_smote)
    n_prox = n_synthetic - n_smote

    # Classic SMOTE samples
    sm = SMOTE(random_state=seed)
    X_smote, y_smote = sm.fit_resample(X_imb, y_imb)

    minority_class = y_imb.value_counts().idxmin()
    X_minority_original = X_imb[y_imb == minority_class]
    n_original = len(X_imb)
    synthetic_mask = np.arange(len(X_smote)) >= n_original
    X_synthetic_smote_all = pd.DataFrame(X_smote[synthetic_mask], columns=X_imb.columns)

    # Sample n_smote synthetic samples randomly from all SMOTE-generated samples
    if len(X_synthetic_smote_all) > n_smote:
        chosen_indices = np.random.choice(len(X_synthetic_smote_all), size=n_smote, replace=False)
        X_synthetic_smote = X_synthetic_smote_all.iloc[chosen_indices].to_numpy()
    else:
        X_synthetic_smote = X_synthetic_smote_all.to_numpy()

    # For plotting, convert synthetic smote samples back to DataFrame
    X_synthetic_smote_df = pd.DataFrame(X_synthetic_smote, columns=X_imb.columns)

    plot_2d_scatter(
        pd.concat([X_minority_original, X_synthetic_smote_df]),
        [minority_class] * (len(X_minority_original) + len(X_synthetic_smote)),
        title="Classic SMOTE: Original Minority + Synthetic Points",
        filename=f"smote_synthetic_seed{seed}.png",
        synthetic_points=X_synthetic_smote_df,
        minority_class=minority_class
    )
    results['SMOTE'] = train_and_evaluate_rf(np.vstack([X_imb, X_synthetic_smote]),
                                             np.hstack([y_imb, [minority_class]*len(X_synthetic_smote)]),
                                             X_test, y_test, "SMOTE")

    # Proximity-guided synthetic samples
    min_mask = y_imb == minority_class
    X_min = X_imb[min_mask]
    y_min = y_imb[min_mask]

    rfgap = RFGAP(n_estimators=100, random_state=seed)
    rfgap.fit(X_min, y_min)
    prox_matrix = rfgap.get_proximities()
    if hasattr(prox_matrix, "toarray"):
        prox_matrix = prox_matrix.toarray()

    neighbors_idx = np.argsort(prox_matrix, axis=1)[:, -k_neighbors-1:-1]

    synthetic_samples_prox = []
    for _ in range(n_prox):
        i = np.random.choice(len(X_min))
        base = X_min.iloc[i].values
        neighbor_indices = neighbors_idx[i]
        neighbor_prox = prox_matrix[i, neighbor_indices]
        weights = neighbor_prox / neighbor_prox.sum() if neighbor_prox.sum() > 0 else np.ones_like(neighbor_prox) / len(neighbor_prox)
        chosen_neighbor_idx = np.random.choice(neighbor_indices, p=weights)
        neighbor = X_min.iloc[chosen_neighbor_idx].values
        alpha = np.random.rand()
        synthetic = base + alpha * (neighbor - base)
        synthetic_samples_prox.append(synthetic)

    synthetic_array_prox = np.array(synthetic_samples_prox)

    # Combine imbalanced + SMOTE + proximity synthetic samples
    X_aug = pd.concat([
        X_imb,
        X_synthetic_smote_df,
        pd.DataFrame(synthetic_array_prox, columns=X_imb.columns)
    ], ignore_index=True)

    y_aug = pd.concat([
        y_imb,
        pd.Series([minority_class] * len(X_synthetic_smote)),
        pd.Series([minority_class] * len(synthetic_samples_prox))
    ], ignore_index=True)

    plot_2d_scatter(X_aug, y_aug,
                    title="Hybrid SMOTE + Proximity-Guided Synthetic Sampling",
                    filename=f"hybrid_synthetic_seed{seed}.png",
                    synthetic_points=pd.DataFrame(synthetic_array_prox, columns=X_imb.columns),
                    minority_class=minority_class)

    results['Hybrid'] = train_and_evaluate_rf(X_aug, y_aug, X_test, y_test, "Hybrid")

    return results


if __name__ == "__main__":
    all_results = {'Original': [], 'Imbalanced': [], 'SMOTE': [], 'Hybrid': []}

    for seed in range(20, 26):
        print(f"\nRunning Experiment [SEED {seed}]")
        result = run_experiment(seed=seed, n_synthetic=100, ratio_smote=0.2)  # Example: 60% SMOTE, 40% Prox
        for key in all_results:
            all_results[key].append(result.get(key, None))

    print("\nAveraged F1 Scores Across Seeds")
    for method, scores in all_results.items():
        filtered_scores = [s for s in scores if s is not None]
        avg_f1 = np.mean(filtered_scores)
        std_f1 = np.std(filtered_scores)
        print(f"{method}: {avg_f1:.4f} ± {std_f1:.4f}")
