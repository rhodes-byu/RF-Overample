import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('/mnt/data/my_experiment_results.csv')

# Filter for the two methods of interest
methods = ['Hybrid_0.5_100', 'SMOTE']
df_filtered = df[df['Method'].isin(methods)]

# Pivot so each row is a dataset + imbalance ratio, with columns for each method's performance
df_pivot = df_filtered.pivot_table(
    index=['Dataset', 'Imbalance_Ratio'],
    columns='Method',
    values='Mean_Weighted_F1'
).reset_index()

# Create scatter plot
plt.figure()
sc = plt.scatter(
    df_pivot['Hybrid_0.5_100'],
    df_pivot['SMOTE'],
    c=df_pivot['Imbalance_Ratio']
)
plt.colorbar(sc, label='Imbalance Ratio')
plt.xlabel('Hybrid_0.5_100 Mean Weighted F1')
plt.ylabel('SMOTE Mean Weighted F1')
plt.title('Performance Comparison: Hybrid_0.5_100 vs SMOTE')
for _, row in df_pivot.iterrows():
    plt.annotate(
        row['Dataset'],
        (row['Hybrid_0.5_100'], row['SMOTE']),
        fontsize=8,
        alpha=0.7
    )
plt.tight_layout()
plt.show()
