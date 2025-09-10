import os
import pandas as pd

# Path to the folder containing datasets
folder_path = "datasets/openml-cc18"

results = []

for filename in sorted(os.listdir(folder_path)):
    if not filename.endswith(".csv"):
        continue

    file_path = os.path.join(folder_path, filename)

    try:
        df = pd.read_csv(file_path)

        # Skip datasets with any missing values
        if df.isnull().values.any():
            print(f"Skipping {filename}: contains missing values")
            continue

        # Final column is the target
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]

        # One-hot encode features only (numeric cols pass through)
        X_enc = pd.get_dummies(X, dummy_na=False)

        # Recombine (target unchanged)
        encoded_df = pd.concat([X_enc, y], axis=1)

        # Report shape after encoding (rows, cols)
        shape = encoded_df.shape
        print(f"{filename}: shape after one-hot = {shape[0]} rows x {shape[1]} cols "
              f"(features: {X_enc.shape[1]}, target: 1)")

        results.append({"dataset": filename,
                        "rows": shape[0],
                        "cols_after_onehot": shape[1],
                        "feature_cols_after_onehot": X_enc.shape[1]})

    except Exception as e:
        print(f"Could not process {filename}: {e}")