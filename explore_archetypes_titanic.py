import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

from SupportFunctions.apply_AA import find_minority_archetypes

# Load preprocessed datasets
pkl_path = "prepared_datasets.pkl"

if not os.path.exists(pkl_path):
    raise FileNotFoundError(f"Expected pickle file at {pkl_path}. Please run load_datasets.py first.")

with open(pkl_path, "rb") as f:
    datasets = joblib.load(f)

# Titanic dataset
if "titanic" not in datasets:
    raise KeyError("Titanic dataset not found in prepared_datasets.pkl")

data_entry = datasets["titanic"]
df = data_entry["data"]
categorical_indices = data_entry["categorical_indices"]
target_col = df.columns[0]

X = df.drop(columns=[target_col])
y = df[target_col]

# Ordinal encode categoricals
cat_columns = [X.columns[i] for i in categorical_indices]
X_encoded = X.copy()
for col in cat_columns:
    X_encoded[col] = X_encoded[col].astype("category").cat.codes

# Scale all features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)
X_scaled_df = pd.DataFrame(X_scaled, columns=X_encoded.columns)

# Extract minority class and apply AA
minority_class = y.value_counts().idxmin()
X_minority_scaled = X_scaled_df[y == minority_class]
archetypes_array = find_minority_archetypes(X_minority_scaled, y[y == minority_class], n_archetypes=10)
archetypes_df = pd.DataFrame(archetypes_array, columns=X_encoded.columns)

# Outlier detection among archetypes
minority_mean = X_minority_scaled.mean().to_numpy()
archetype_dists = euclidean_distances(archetypes_df, [minority_mean]).flatten()
threshold = np.percentile(archetype_dists, 90)

print("\n[INFO] Archetype distances from minority class mean (scaled):")
for i, dist in enumerate(archetype_dists):
    flag = " <-- POSSIBLE OUTLIER" if dist > threshold else ""
    print(f"  Archetype {i+1}: Distance = {dist:.2f}{flag}")

# PCA visualization (on scaled full dataset)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
Z_pca = pca.transform(archetypes_df)

y_labels = y.map(lambda label: "Minority" if label == minority_class else "Majority")
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=["orange" if lbl == "Minority" else "gray" for lbl in y_labels], alpha=0.3, s=20, label="All Data")
X_min_idx = y[y == minority_class].index
plt.scatter(X_pca[X_min_idx, 0], X_pca[X_min_idx, 1], c="blue", alpha=0.5, label="Minority Points", s=40)
plt.scatter(Z_pca[:, 0], Z_pca[:, 1], c="red", edgecolor="black", marker="X", s=120, label="Archetypes")

plt.title("PCA (Scaled): All Data vs. Minority vs. Archetypes (Titanic)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Distribution comparison (on scaled features)
X_minority_scaled["type"] = "minority"
archetypes_df["type"] = "archetype"

combined = pd.concat([X_minority_scaled, archetypes_df], ignore_index=True)
melted = pd.melt(combined, id_vars="type", var_name="Feature", value_name="Value")

plt.figure(figsize=(14, 8))
sns.violinplot(data=melted, x="Feature", y="Value", hue="type", split=True)
plt.xticks(rotation=45)
plt.title("Feature Value Distributions (Scaled): Minority vs Archetypes")
plt.tight_layout()
plt.show()
