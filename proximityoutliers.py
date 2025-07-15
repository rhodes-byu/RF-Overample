import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from rfgap import RFGAP
from SupportFunctions.imbalancer import ImbalanceHandler

# Customized Parameters
dataset_name = "titanic"
imbalance_ratio = 0.1
n_outliers = 10
xlim = (-5, 5)
ylim = (-5, 5)

def plot_2d_scatter(X, y, title, filename, outliers_idx=None, synthetic_points=None, minority_class=None):

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

    if outliers_idx is not None:
        plt.scatter(X_2d[outliers_idx, 0], X_2d[outliers_idx, 1],
                    edgecolors='k', facecolors='none',
                    s=100, linewidths=1.5, label="Outliers")

    if synthetic_points is not None:
        plt.scatter(syn_2d[:, 0], syn_2d[:, 1],
                    label="Synthetic Samples", c='red', marker='x', s=60)

    plt.title(title)
    plt.legend()
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.tight_layout()
    plt.savefig(os.path.join("graphs", filename))
    plt.close()

def train_and_evaluate_rf(X_train, y_train, X_test, y_test, label):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    f1 = f1_score(y_test, preds, average='weighted')
    print(f"[F1 - {label}] Weighted F1 Score: {f1:.4f}")
    return f1

def run_experiment(seed, n_synthetic=100, k_neighbors=5):

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

    # Split train/test
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed
    )

    minority_class = y_train_full.value_counts().idxmin()

    plot_2d_scatter(X_train_full, y_train_full,
                    title="Original Training Data",
                    filename=f"original_train_seed{seed}.png",
                    minority_class=minority_class)

    results = {}

    # Model 1: Original
    results['Original'] = train_and_evaluate_rf(X_train_full, y_train_full, X_test, y_test, "Original")

    # Apply Imbalance
    imbalancer = ImbalanceHandler(X_train_full, y_train_full,
                                  imbalance_ratio=imbalance_ratio,
                                  batch_size=200, random_state=seed)
    X_imb, y_imb = imbalancer.introduce_imbalance()

    # Visualize imbalanced data
    plot_2d_scatter(X_imb, y_imb,
                    title="Imbalanced Training Data",
                    filename=f"imbalanced_train_seed{seed}.png",
                    minority_class=minority_class)

    # Model 2: Imbalanced Only
    results['Imbalanced'] = train_and_evaluate_rf(X_imb, y_imb, X_test, y_test, "Imbalanced Only")

    # Model 3: Classic SMOTE
    sm = SMOTE(random_state=seed)
    X_smote, y_smote = sm.fit_resample(X_imb, y_imb)

    # Identify original minority points in imbalanced data
    minority_class = y_imb.value_counts().idxmin()
    X_minority_original = X_imb[y_imb == minority_class]

    # SMOTE appends synthetic minority samples at the end
    n_original = len(X_imb)
    synthetic_mask = np.arange(len(X_smote)) >= n_original
    X_synthetic_smote = X_smote[synthetic_mask]

    # Visualize original minority + synthetic points from SMOTE
    plot_2d_scatter(
        pd.concat([X_minority_original, pd.DataFrame(X_synthetic_smote, columns=X_minority_original.columns)]),
        [minority_class] * (len(X_minority_original) + len(X_synthetic_smote)),
        title="Classic SMOTE: Original Minority + Synthetic Points",
        filename=f"smote_synthetic_seed{seed}.png",
        synthetic_points=pd.DataFrame(X_synthetic_smote, columns=X_minority_original.columns),
        minority_class=minority_class
    )

    results['SMOTE'] = train_and_evaluate_rf(X_smote, y_smote, X_test, y_test, "SMOTE")


    # Model 4: Proximity-Guided SMOTE
    min_mask = y_imb == minority_class
    X_min = X_imb[min_mask]
    y_min = y_imb[min_mask]

    rfgap = RFGAP(n_estimators=100, random_state=seed)
    rfgap.fit(X_min, y_min)
    outlier_scores = rfgap.get_outlier_scores(y_min, scaling='normalize')
    outlier_indices = np.argsort(outlier_scores)[-n_outliers:]
    X_outliers = X_min.iloc[outlier_indices]

    # Visualize outliers
    plot_2d_scatter(X_min, y_min,
                    title="Minority Outliers",
                    filename=f"outliers_seed{seed}.png",
                    outliers_idx=outlier_indices,
                    minority_class=minority_class)

    nn = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(X_outliers)
    _, neighbors = nn.kneighbors(X_outliers)

    synthetic_samples = []
    for _ in range(n_synthetic):
        i = np.random.choice(len(X_outliers))
        base = X_outliers.iloc[i].values
        neighbor = X_outliers.iloc[np.random.choice(neighbors[i][1:])].values
        alpha = np.random.rand()
        synthetic = base + alpha * (neighbor - base)
        synthetic_samples.append(synthetic)

    synthetic_array = np.array(synthetic_samples)
    X_aug = pd.concat([X_imb, pd.DataFrame(synthetic_array, columns=X_imb.columns)], ignore_index=True)
    y_aug = pd.concat([y_imb, pd.Series([minority_class] * len(synthetic_samples))], ignore_index=True)

    # Visualize Generated Points
    plot_2d_scatter(X_aug, y_aug,
                    title="Proximity-SMOTE Augmented Data",
                    filename=f"augmented_seed{seed}.png",
                    synthetic_points=pd.DataFrame(synthetic_array, columns=X_imb.columns),
                    minority_class=minority_class)

    results['Proximity-SMOTE'] = train_and_evaluate_rf(X_aug, y_aug, X_test, y_test, "Proximity-SMOTE")

    return results


if __name__ == "__main__":
    all_results = {'Original': [], 'Imbalanced': [], 'SMOTE': [], 'Proximity-SMOTE': []}

    for seed in range(42, 47):
        print(f"\nRunning Experiment [SEED {seed}]")
        result = run_experiment(seed=seed)
        for key in all_results:
            all_results[key].append(result[key])

    print("\nAveraged F1 Scores Across Seeds")
    for method, scores in all_results.items():
        avg_f1 = np.mean(scores)
        std_f1 = np.std(scores)
        print(f"{method}: {avg_f1:.4f} ± {std_f1:.4f}")
