import pandas as pd
from joblib import load
from SupportFunctions.visualizer import clean_results

results_file = "experiment_results.pkl"
raw_results = load(results_file)

results_df = clean_results(raw_results) if not isinstance(raw_results, pd.DataFrame) else raw_results

results_df = results_df.sort_values(by="Dataset").reset_index(drop=True)

results_df.to_csv("cleaned_experiment_results.csv", index=False)

print(results_df.head())
