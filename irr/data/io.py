# irr/data/splits.py
# data ingestion utility that loads multiple CSVs and concatenates them
# includes featur and label checks, drops NaNs, ensures label is binary {0,1}

from pathlib import Path
import glob, numpy as np, pandas as pd
from irr.constants import FEATURES, LABEL_COL



def load_csvs(data_glob: str) -> pd.DataFrame:
    paths = sorted(glob.glob(data_glob))
    if not paths:
        raise FileNotFoundError(f"No CSV files matched: {data_glob}")
    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        missing = [c for c in FEATURES + [LABEL_COL] if c not in df.columns]
        if missing:
            raise ValueError(f"{p} missing columns: {missing}")
        dfs.append(df)
    full = pd.concat(dfs, ignore_index=True)
    full = full.dropna(subset=FEATURES + [LABEL_COL])
    full[LABEL_COL] = (full[LABEL_COL].astype(np.int64) > 0).astype(np.int64)
    return full
