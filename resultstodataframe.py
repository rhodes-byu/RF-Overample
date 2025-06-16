import pandas as pd
from joblib import load
from SupportFunctions.visualizer import clean_results

results_file = "experiment_results.pkl"
raw_results = load(results_file)

results_df = (
    raw_results if isinstance(raw_results, pd.DataFrame) else clean_results(raw_results)
)

if "Dataset" not in results_df.columns and "dataset_name" in results_df.columns:
    results_df = results_df.rename(columns={"dataset_name": "Dataset"})

if "Dataset" in results_df.columns:
    results_df = results_df[results_df["Dataset"] != "zoo"]

sort_cols = [c for c in ["Dataset", "Method"] if c in results_df.columns]
if sort_cols:
    results_df = results_df.sort_values(by=sort_cols).reset_index(drop=True)


output_file = "experiment_results_summary.csv"
try:
    results_df.to_csv(output_file, index=False)
    print(f"[INFO] Saved tidy results to {output_file}")
except PermissionError:
    print(f"[WARN] Permission denied: close '{output_file}' if it's open and try again.")

print(results_df.head(10))
