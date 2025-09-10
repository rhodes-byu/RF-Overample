# load_datasets.py
import os
from pathlib import Path
from typing import List, Tuple, Iterator
import pandas as pd
import joblib

INT_CATEGORICAL_MAX_UNIQUE = 50  # treat int columns with <= 50 uniques as categorical

def _move_last_col_to_first(df: pd.DataFrame) -> pd.DataFrame:
    if df.shape[1] <= 1:
        return df
    cols = list(df.columns)
    target = cols[-1]
    return df[[target] + cols[:-1]]

def _infer_categorical_indices(df: pd.DataFrame) -> List[int]:

    if df.shape[1] <= 1:
        return []

    target_col = df.columns[0]
    X = df.drop(columns=[target_col])

    cat_cols = []
    for c in X.columns:
        s = X[c]
        kind = s.dtype.kind  # 'O'=object, 'b'=bool, 'i'/'u'=int, 'f'=float

        if kind == "O" or str(s.dtype) == "category":          # object/category
            cat_cols.append(c)
        elif kind == "b":                                      # bool
            cat_cols.append(c)
        elif kind in ("i", "u"):                               # int/uint
            if s.nunique(dropna=True) <= INT_CATEGORICAL_MAX_UNIQUE:
                cat_cols.append(c)
        # floats are left as continuous by default

    return [X.columns.get_loc(c) for c in cat_cols]

def _iter_dataset_csvs(root: Path) -> Iterator[Tuple[str, Path, bool]]:

    base = root / "datasets"
    cc18 = base / "openml-cc18"

    # top-level csvs (preformatted: first column is target)
    for p in sorted(base.glob("*.csv")):
        yield p.stem, p, False

    # openml-cc18 subfolder (last column is target)
    if cc18.exists():
        for p in sorted(cc18.glob("*.csv")):
            yield p.stem, p, True

def load_and_prepare_datasets(folder_path: str = "datasets"):

    dataset_dict = {}
    root = Path(".")

    for dataset_name, file_path, is_openml in _iter_dataset_csvs(root):
        try:
            df = pd.read_csv(file_path, low_memory=False)

            # Handle target position based on source
            if is_openml:
                # OpenML dumps: LAST column is the target -> move to FIRST
                df = _move_last_col_to_first(df)
                origin = "openml-cc18"
            else:
                # Local/top-level: already FIRST column is target
                origin = "local"

            # Infer categoricals (relative to X)
            cat_indices = _infer_categorical_indices(df)
            X = df.drop(columns=df.columns[0])
            cat_names = [X.columns[i] for i in cat_indices] if cat_indices else []

            dataset_dict[dataset_name] = {
                "data": df,
                "categorical_indices": cat_indices,
                "categorical_names": cat_names,
                "origin": origin,
                "source_path": str(file_path),
            }

            print(
                f"Loaded: {dataset_name} "
                f"({df.shape[0]} rows, {df.shape[1]} cols) | "
                f"from={origin} | "
                f"categoricals={len(cat_indices)}"
            )

        except Exception as e:
            print(f"Failed to load '{file_path}': {e}")

    print(f"\nTotal datasets loaded: {len(dataset_dict)}")
    return dataset_dict

def load_selected_datasets(config, pickle_path: str = "prepared_datasets.pkl"):
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Pickle file not found at: {pickle_path}")

    with open(pickle_path, "rb") as f:
        all_datasets = joblib.load(f)

    selected_names = config.get("selected_datasets", [])
    if isinstance(selected_names, str) and selected_names.lower() == "all":
        selected_names = list(all_datasets.keys())

    selected_data = {}
    for name in selected_names:
        if name in all_datasets:
            selected_data[name] = all_datasets[name]
        else:
            raise KeyError(f"Dataset '{name}' not found in {pickle_path}")

    return selected_data

if __name__ == "__main__":
    output_path = "prepared_datasets.pkl"
    datasets = load_and_prepare_datasets()
    joblib.dump(datasets, output_path)
    print(f"\nSaved preprocessed datasets to: {output_path}")
