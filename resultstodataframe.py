import pandas as pd
from joblib import load
from SupportFunctions.visualizer import clean_results

# Load raw results
results_file = "experiment_results.pkl"
raw_results = load(results_file)

# Clean and flatten into a DataFrame
if isinstance(raw_results, pd.DataFrame):
    results_df = raw_results
else:
    results_df = clean_results(raw_results)

# Optional: Save as CSV for external use
results_df.to_csv("experiment_results_summary.csv", index=False)

# Show a preview
print(results_df.head())
