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
from SupportFunctions.imbalancer import ImbalanceHandler

# settings
n_archetypes = 20

# load dataset
with open("prepared_datasets.pkl", "rb") as f:
    datasets = joblib.load(f)

df = datasets["titanic"]["data"]
categorical_indices = datasets["titanic"]["categorical_indices"]
target_col = df.columns[0]

X = df.drop(columns=[target_col])
y = df[target_col]

# encoding to numerical codes
cat_columns = [X.columns[i] for i in categorical_indices]
X_encoded = X.copy()
for col in cat_columns:
    X_encoded[col] = X_encoded[col].astype("category").cat.codes

print(f"\n[INFO] Original class distribution: {y.value_counts().to_dict()}")

# introduce imbalance
imbalancer = ImbalanceHandler(X_encoded, y, imbalance_ratio=0.2, batch_size=200, random_state=42)
X_imb, y_imb = imbalancer.introduce_imbalance()

# standize dataset for later graphing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imb)
X_scaled_df = pd.DataFrame(X_scaled, columns=X_encoded.columns)

# extract minority class
minority_class = y_imb.value_counts().idxmin()
X_minority = X_scaled_df[y_imb == minority_class].copy()
y_minority = y_imb[y_imb == minority_class]

# apply AA
archetypes_array = find_minority_archetypes(X_minority, y_minority, n_archetypes=n_archetypes)
archetypes_df = pd.DataFrame(archetypes_array, columns=X_encoded.columns)

if len(archetypes_df) < n_archetypes:
    print(f"[WARN] Only {len(archetypes_df)} archetypes were generated (requested {n_archetypes}).")
else:
    print(f"[INFO] Successfully generated {len(archetypes_df)} archetypes.")

# try and identify potential outlier archetypes
minority_mean = X_minority.mean().to_numpy()
dists = euclidean_distances(archetypes_df, [minority_mean]).flatten()
threshold = np.percentile(dists, 90)

print("\n[INFO] Archetype distances from minority class mean (scaled):")
for i, dist in enumerate(dists):
    flag = " <-- POSSIBLE OUTLIER" if dist > threshold else ""
    print(f"  Archetype {i+1}: Distance = {dist:.2f}{flag}")

# PCA transform
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
Z_pca = pca.transform(archetypes_df.values)

# output file
desktop = os.path.join(os.path.expanduser("~"), "Desktop")
output_dir = desktop if os.path.exists(desktop) else os.path.join(os.path.expanduser("~"), "AA_Visualizations")
os.makedirs(output_dir, exist_ok=True)

# plot 1
minority_mask = y_imb == minority_class
X_min_pca = X_pca[minority_mask]
X_maj_pca = X_pca[~minority_mask]

plt.figure(figsize=(10, 6))
plt.scatter(X_maj_pca[:, 0], X_maj_pca[:, 1], c="lightgray", alpha=0.3, s=20, label="Majority Class")
plt.scatter(X_min_pca[:, 0], X_min_pca[:, 1], c="blue", alpha=0.5, s=30, label="Minority Class")
plt.scatter(Z_pca[:, 0], Z_pca[:, 1], c="red", edgecolor="black", marker="X", s=120, label="Archetypes")

# add numbers
for i, (x, y_pt) in enumerate(Z_pca):
    plt.text(x + 0.1, y_pt, str(i + 1), fontsize=9, color="red")

plt.title("PCA (Scaled): Majority vs Minority vs Archetypes")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend()
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='datalim')  # Ensure consistent axis scale
plt.tight_layout()

pca_path = os.path.join(output_dir, "pca_archetypes_titanic.png")
plt.savefig(pca_path)
print(f"[INFO] PCA plot saved to: {pca_path}")
plt.close()

# distribution plot
X_minority["type"] = "minority"
archetypes_df["type"] = "archetype"
combined = pd.concat([X_minority, archetypes_df], ignore_index=True)
melted = pd.melt(combined, id_vars="type", var_name="Feature", value_name="Value")

plt.figure(figsize=(14, 8))
sns.violinplot(data=melted, x="Feature", y="Value", hue="type", split=True)
plt.xticks(rotation=45)
plt.title("Feature Distributions (Scaled): Minority vs Archetypes")
plt.tight_layout()
violin_path = os.path.join(output_dir, "violin_archetypes_titanic.png")
plt.savefig(violin_path)
print(f"[INFO] Violin plot saved to: {violin_path}")
plt.close()
