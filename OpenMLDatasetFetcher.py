# fetch_openml_cc18_to_csv.py
from pathlib import Path
import os
import openml
import pandas as pd


openml.config.apikey = os.getenv(" "APIKEYGOESHERE"", "")
openml.config.cache_directory = "openml_cache"

OUT = Path("data/openml-cc18")
OUT.mkdir(parents=True, exist_ok=True)

suite = openml.study.get_suite("OpenML-CC18") 
print(f"{len(suite.data)} datasets in CC-18")

for did in suite.data:
    ds = openml.datasets.get_dataset(did)
    X, y, categorical_indicator, attr_names = ds.get_data(
        target=ds.default_target_attribute,
        dataset_format="dataframe" 
    )
    df = X.copy()
    df[ds.default_target_attribute] = y

    safe_name = f"{did}_{ds.name}".replace("/", "_")
    out_path = OUT / f"{safe_name}.csv"
    df.to_csv(out_path, index=False)
    print("saved:", out_path)
