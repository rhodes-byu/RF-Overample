from joblib import load
import os
import pandas as pd
from SupportFunctions.visualizer import (
    plot_f1_scores,
    plot_f1_by_encoding,
    plot_f1_by_archetype_setting,
    plot_f1_by_minority_sample_setting,
    plot_f1_by_use_of_archetypes,
    clean_results
)

results_file = "experiment_results.pkl"

if os.path.exists(results_file):
    raw_results = load(results_file)

    # Auto-detect and clean only if needed
    if isinstance(raw_results, pd.DataFrame):
        results_df = raw_results
    else:
        results_df = clean_results(raw_results)

    print("Available columns in the dataset:\n", results_df.columns)

    print("\nLoaded Experiment Results (sample rows):")
    try:
        print(results_df[["Dataset", "Encoding Method", "Method", "Imbalance Ratio", "Weighted F1 Score"]].head())
    except KeyError as e:
        print(f"Error: {e}")
        print("\nColumn names available:", results_df.columns)

    plot_f1_scores(results_df, save_fig=True)
    # plot_f1_by_encoding(results_df, save_fig=True)
    # plot_f1_by_archetype_setting(results_df, save_fig=True)
    # plot_f1_by_minority_sample_setting(results_df, save_fig=True)
    # plot_f1_by_use_of_archetypes(results_df, save_fig=True)

else:
    print(f"Results file '{results_file}' not found. Please run runner.py first to generate experiment results.")
