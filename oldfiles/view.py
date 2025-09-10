import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from matplotlib.colors import ListedColormap

def plot_pca_scatter(X_orig, X_rf_synth, X_smote_synth, y_orig=None):
    combined = pd.concat([
        X_orig.assign(Source='Original'),
        X_rf_synth.assign(Source='RF Oversample'),
        X_smote_synth.assign(Source='SMOTE')
    ], ignore_index=True)

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(combined.drop(columns='Source'))
    combined['PC1'] = reduced[:, 0]
    combined['PC2'] = reduced[:, 1]

    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=combined, x='PC1', y='PC2', hue='Source', alpha=0.7, s=60)
    plt.title('PCA Projection of Original and Synthetic Samples')
    plt.show()

def plot_tsne_scatter(X_orig, X_rf_synth, X_smote_synth, y_orig=None, perplexity=30):
    combined = pd.concat([
        X_orig.assign(Source='Original'),
        X_rf_synth.assign(Source='RF Oversample'),
        X_smote_synth.assign(Source='SMOTE')
    ], ignore_index=True)

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    reduced = tsne.fit_transform(combined.drop(columns='Source'))
    combined['Dim1'] = reduced[:, 0]
    combined['Dim2'] = reduced[:, 1]

    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=combined, x='Dim1', y='Dim2', hue='Source', alpha=0.7, s=60)
    plt.title('t-SNE Projection of Original and Synthetic Samples')
    plt.show()

def plot_pairplot_features(X_orig, X_rf_synth, X_smote_synth, feature_list=None):
    if feature_list is None:
        feature_list = X_orig.columns[:4]

    combined = pd.concat([
        X_orig[feature_list].assign(Source='Original'),
        X_rf_synth[feature_list].assign(Source='RF Oversample'),
        X_smote_synth[feature_list].assign(Source='SMOTE')
    ], ignore_index=True)

    sns.pairplot(combined, hue='Source', diag_kind='kde', plot_kws={'alpha': 0.6, 's': 30})
    plt.suptitle('Pairplot of Selected Features', y=1.02)
    plt.show()

def plot_feature_distributions(X_orig, X_rf_synth, X_smote_synth, feature_list=None):
    if feature_list is None:
        feature_list = X_orig.columns[:4]

    for feature in feature_list:
        plt.figure(figsize=(8, 4))
        sns.kdeplot(X_orig[feature], label='Original', fill=True)
        sns.kdeplot(X_rf_synth[feature], label='RF Oversample', fill=True)
        sns.kdeplot(X_smote_synth[feature], label='SMOTE', fill=True)
        plt.title(f'Distribution of Feature: {feature}')
        plt.legend()
        plt.show()

def plot_nearest_neighbor_distance_hist(X_orig, X_rf_synth, X_smote_synth, k=1):
    nbrs = NearestNeighbors(n_neighbors=k).fit(X_orig)

    dist_rf, _ = nbrs.kneighbors(X_rf_synth)
    dist_smote, _ = nbrs.kneighbors(X_smote_synth)

    plt.figure(figsize=(10, 5))
    plt.hist(dist_rf.flatten(), bins=30, alpha=0.6, label='RF Oversample')
    plt.hist(dist_smote.flatten(), bins=30, alpha=0.6, label='SMOTE')
    plt.title('Histogram of Nearest Neighbor Distances to Original Samples')
    plt.xlabel('Distance')
    plt.ylabel('Count')
    plt.legend()
    plt.show()

def plot_decision_boundaries(X_orig, y_orig, X_rf_synth, X_smote_synth, method_labels=['Original', 'RF Oversample', 'SMOTE'], reduction='pca'):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    X_all = pd.concat([
        X_orig.assign(Source=method_labels[0]),
        X_rf_synth.assign(Source=method_labels[1]),
        X_smote_synth.assign(Source=method_labels[2])
    ], ignore_index=True)
    
    y_all = pd.Series([0]*len(X_orig) + [1]*len(X_rf_synth) + [2]*len(X_smote_synth))

    if reduction == 'pca':
        reducer = PCA(n_components=2)
    elif reduction == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:
        raise ValueError("reduction must be 'pca' or 'tsne'")

    X_proj = reducer.fit_transform(X_all.drop(columns='Source'))
    X_all['Dim1'] = X_proj[:,0]
    X_all['Dim2'] = X_proj[:,1]

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_all[['Dim1', 'Dim2']], y_all)

    x_min, x_max = X_all['Dim1'].min() - 1, X_all['Dim1'].max() + 1
    y_min, y_max = X_all['Dim2'].min() - 1, X_all['Dim2'].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    Z = clf.predict(grid_points)
    Z = Z.reshape(xx.shape)

    cmap_light = ListedColormap(['#AAAAFF','#FFAAAA','#AAFFAA'])
    cmap_bold = ['#0000FF','#FF0000','#00AA00']

    plt.figure(figsize=(12,8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)

    for i, label in enumerate(method_labels):
        subset = X_all[X_all['Source'] == label]
        plt.scatter(subset['Dim1'], subset['Dim2'], c=cmap_bold[i], label=label, edgecolor='k', s=60, alpha=0.8)

    plt.title(f'Decision Boundary on 2D {reduction.upper()} Projection')
    plt.xlabel('Dim1')
    plt.ylabel('Dim2')
    plt.legend()
    plt.show()
