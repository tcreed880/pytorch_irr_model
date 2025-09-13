# %% [markdown]
# # Explore IrrMapper AlphaEarth CSVs
# Quick EDA: class counts and counts by county

# %%
import pandas as pd
from pathlib import Path

# Path to raw data folder
data_dir = Path("../raw_data")

# Your CSVs: one per year 2018–2022
files = sorted(data_dir.glob("WA_alphaearth_irrmapper_unbalanced_*.csv"))
files

# %%
# Peek at one file to understand columns
sample_df = pd.read_csv(files[0], nrows=5)
sample_df.head()

# %%
# Function to load and summarize a file
def summarize_file(path: Path, target_col="label", county_col="county"):
    df = pd.read_csv(path)
    print(f"=== {path.name} ===")
    # Overall 0/1 counts
    counts = df[target_col].value_counts().sort_index()
    print("Class counts:\n", counts, "\n")
    # Counts by county (top 10)
    if county_col in df.columns:
        ccounts = df.groupby(county_col)[target_col].value_counts().unstack(fill_value=0)
        print("Counts by county (first 10 rows):")
        print(ccounts.head(10))
    else:
        print(f"No '{county_col}' column found.")
    return df

# %%
# Loop through all files
dfs = {}
for f in files:
    dfs[f.stem] = summarize_file(f)

# %%
# Combine all years into one DataFrame (optional)
all_df = pd.concat(dfs.values(), keys=dfs.keys(), names=["year", "row"])
print("Combined shape:", all_df.shape)

# %%
# Plot class counts per year
import matplotlib.pyplot as plt

year_counts = {
    year: df["label"].value_counts().sort_index()
    for year, df in dfs.items()
}
count_df = pd.DataFrame(year_counts).T.fillna(0)

count_df.plot(kind="bar", stacked=True)
plt.title("Class balance per year")
plt.xlabel("Year")
plt.ylabel("Count")
plt.legend(title="Label")
plt.tight_layout()

# %%
