import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load data
# -----------------------------
df = pd.read_csv("/home/diegopab/projects/ForestFire_GT/paperwriting/ForestFireGT/data/indata/data_trainingset_2017-2024/allvars_2017-2024.csv" )

# -----------------------------
# 2. Clean dataset
# -----------------------------
# Drop non-informative / indexing columns
drop_cols = ["lat", "lon", "month", "year", "fire"]
df = df.drop(columns=drop_cols, errors="ignore")

# Optional: rename variables for nicer labels (recommended for paper)
rename_dict = {
    "pr": "Precipitation",
    "tas": "Mean Temp",
    "tasmax": "Max Temp",
    "tasmin": "Min Temp",
    "rsds": "Solar Radiation",
    "PET": "PET",
    "climbal": "Clim. Water Balance",
    "distRoads": "Dist. Roads",
    "distUrban": "Dist. Urban",
    "distWater": "Dist. Water",
    "elev": "Elevation",
    "slope": "Slope",
    "aspect_sin": "Aspect Sine",
    "aspect_cos": "Aspect Cosine"
}
df = df.rename(columns=rename_dict)

# -----------------------------
# 3. Correlation matrix
# -----------------------------
corr = df.corr(method="pearson")

# Mask upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# -----------------------------
# 4. Plot
# -----------------------------
sns.set_style("white")

plt.figure(figsize=(10, 8))

heatmap = sns.heatmap(
    corr,
    mask=mask,
    cmap="coolwarm",
    vmin=-1, vmax=1,
    center=0,
    square=True,
    linewidths=0.5,
    annot=True,
    fmt=".2f",
    annot_kws={"size": 9},
    cbar_kws={"shrink": 0.8, "label": "Pearson correlation"}
)

# Improve labels
plt.xticks(rotation=45, ha="right", fontsize=12)
plt.yticks(rotation=0, fontsize=12)

# Title (keep subtle for publications)
plt.title("Pearson Correlation Matrix", fontsize=14)

plt.tight_layout()

# -----------------------------
# 5. Save figure
# -----------------------------
os.makedirs("../../figures/01_data_exploration", exist_ok=True)
plt.savefig("../../figures/01_data_exploration/correlation_heatmap.png", dpi=300, bbox_inches="tight")

plt.show()
