from joblib import load
import os
from SupportFunctions.visualizer import (
    plot_f1_scores,
    plot_f1_by_encoding,
    plot_f1_by_archetype_setting,
    plot_f1_by_minority_sample_setting,
    plot_f1_by_use_of_archetypes
)

# Path to the results file.
results_file = "experiment_results.pkl"

if os.path.exists(results_file):
    loaded_results_df = load(results_file)
    
    # Print available columns to verify the structure.
    print("Available columns in the dataset:\n", loaded_results_df.columns)
    
    print("\nLoaded Experiment Results (sample rows):")
    try:
        # Print key columns in a user-friendly table.
        print(loaded_results_df[["Dataset", "Encoding Method", "Method", "Imbalance Ratio", "Weighted F1 Score"]].head())
    except KeyError as e:
        print(f"Error: {e}")
        print("\nColumn names available:", loaded_results_df.columns)
    
    plot_f1_scores(loaded_results_df, save_fig=True)
    # plot_f1_by_encoding(loaded_results_df, save_fig=True)
    # plot_f1_by_archetype_setting(loaded_results_df, save_fig=True)
    # plot_f1_by_minority_sample_setting(loaded_results_df, save_fig=True)
    # plot_f1_by_use_of_archetypes(loaded_results_df, save_fig=True)

else:
    print(f"Results file '{results_file}' not found. Please run runner.py first to generate experiment results.")
