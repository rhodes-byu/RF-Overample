from joblib import load
import pandas as pd
import os
from SupportFunctions.visualizer import plot_f1_scores, plot_f1_by_encoding, plot_f1_by_archetype_setting, plot_f1_by_minority_sample_setting

# Check if the results file exists
results_file = "experiment_results.pkl"

if os.path.exists(results_file):
    # Load results from file
    loaded_results_df = load(results_file)
    
    # Print available columns for debugging
    print("Available columns in the dataset:\n", loaded_results_df.columns)

    # Print experiment results
    print("\nLoaded Experiment Results:")
    try:
        print(loaded_results_df[["Dataset", "Encoding Method", "Method", "Imbalance Ratio", "Weighted F1 Score"]])
    except KeyError as e:
        print(f"Error: {e}")
        print("\nColumn names available:", loaded_results_df.columns)

    # Plot F1 scores
    plot_f1_scores(loaded_results_df)
    plot_f1_by_encoding(loaded_results_df)
    plot_f1_by_minority_sample_setting(loaded_results_df)
    plot_f1_by_archetype_setting(loaded_results_df)

else:
    print(f"Results file '{results_file}' not found. Please run runner.py first to generate experiment results.")
