# irr/data/io.py
# Load multiple CSVs, validate feature/label columns, and concatenate.

from __future__ import annotations
import glob
import numpy as np
import pandas as pd
from irr.constants import FEATURES, LABEL_COL


def load_csvs(data_glob: str) -> pd.DataFrame:
    paths = sorted(glob.glob(data_glob))
    if not paths:
        raise FileNotFoundError(f"No CSV files matched: {data_glob}")

    dfs: list[pd.DataFrame] = []
    for p in paths:
        df = pd.read_csv(p)
        missing = [c for c in FEATURES + [LABEL_COL] if c not in df.columns]
        if missing:
            raise ValueError(f"{p} missing required columns: {missing}")
        dfs.append(df)

    full = pd.concat(dfs, ignore_index=True)

    # Drop rows with missing feature or label values
    full = full.dropna(subset=FEATURES + [LABEL_COL]).copy()

    # Coerce label to 0 or 1
    full[LABEL_COL] = pd.to_numeric(full[LABEL_COL], errors="coerce").fillna(0).astype(np.int64)
    full[LABEL_COL] = (full[LABEL_COL] > 0).astype(np.int64)

    # Normalize string columns 
    if ".geo" in full.columns:
        full[".geo"] = full[".geo"].astype(str)
    if "state" in full.columns:
        full["state"] = full["state"].astype(str)
    if "county_fips" in full.columns:
        full["county_fips"] = full["county_fips"].astype(str).str.zfill(3)

    return full
