import pandas as pd, glob
from pathlib import Path

# Adjust if your paths differ
TRAIN_GLOB = "../raw_data/*unbalanced_20*.csv"          # will include *_unbalanced_*.csv
VAL_GLOB   = "../raw_data/*unbalanced_val*.csv"      # val files

# 1) Read files
train_files = sorted(p for p in glob.glob(TRAIN_GLOB) if "val" not in Path(p).name)
val_files   = sorted(glob.glob(VAL_GLOB))

train = pd.concat((pd.read_csv(p) for p in train_files), ignore_index=True)
val   = pd.concat((pd.read_csv(p) for p in val_files),   ignore_index=True)

# 2) Drop the first column if it's an index column in your CSVs
train_n = train.iloc[:, [-1]]
val_n   = val.iloc[:, [-1]]

# 3) Drop duplicates within each set (so we count unique rows)
train_u = train_n.drop_duplicates()
val_u   = val_n.drop_duplicates()

# 4) Find exact overlaps (inner join on all columns)
common_cols = [c for c in train_u.columns if c in val_u.columns]
overlap = pd.merge(train_u[common_cols], val_u[common_cols], how="inner").drop_duplicates()

print(f"Train rows: {len(train):,} (unique {len(train_u):,})")
print(f"Val rows  : {len(val):,} (unique {len(val_u):,})")
print(f"Exact duplicate rows across sets: {len(overlap):,}")
print(f"Leakage % of train unique: {100*len(overlap)/max(1,len(train_u)):.3f}%")
print(f"Leakage % of val unique  : {100*len(overlap)/max(1,len(val_u)):.3f}%")

# %%


# Load
train_files = sorted(glob.glob("../raw_data/*unbalanced_2*.csv"))
val_files   = sorted(glob.glob("../raw_data/*unbalanced_val*.csv"))

train = pd.concat((pd.read_csv(p) for p in train_files), ignore_index=True)
val   = pd.concat((pd.read_csv(p) for p in val_files),   ignore_index=True)

# Make sure we’re actually using the `.geo` col explicitly (not just "last column")
geo_col = ".geo"
assert geo_col in train.columns and geo_col in val.columns

# Normalize a touch to avoid whitespace artifacts
t_geo = train[geo_col].astype(str).str.strip()
v_geo = val[geo_col].astype(str).str.strip()

# Unique location counts
t_unique = set(t_geo)
v_unique = set(v_geo)
overlap  = t_unique & v_unique

print(f"Train rows: {len(train):,} | unique geos: {len(t_unique):,}")
print(f"Val rows  : {len(val):,} | unique geos: {len(v_unique):,}")
print(f"Location overlap (geos in both): {len(overlap):,}")
print(f"Leakage by location — of train: {100*len(overlap)/max(1,len(t_unique)):.3f}%")
print(f"Leakage by location — of val  : {100*len(overlap)/max(1,len(v_unique)):.3f}%")

# How many years per location? (quick sanity)
t_per_loc = t_geo.value_counts()
v_per_loc = v_geo.value_counts()
print("\nTrain — rows per location (value_counts of counts):\n", t_per_loc.value_counts().sort_index())
print("\nVal   — rows per location (value_counts of counts):\n", v_per_loc.value_counts().sort_index())

# %%
