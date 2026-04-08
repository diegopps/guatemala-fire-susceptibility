"""
plot_pca.py
-----------
Performs PCA on all predictor variables and generates three
figures:
  1. scree_plot.png       — variance explained per component
  2. loadings_heatmap.png — variable contributions to each component
  3. biplot.png           — variables projected in PC1/PC2 space

The data is standardised (zero mean, unit variance) before PCA,
as variables are on very different scales.

Dependencies:
    pip install pandas matplotlib seaborn scikit-learn numpy
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

INPUT_FILE = "/home/diegopab/projects/ForestFire_GT/paperwriting/data/data_trainingset_2017-2024/allvars_2017-2024_v2.csv"
OUTPUT_DIR = "./figures/pca"

# ---------------------------------------------------------------------------
# Variables (fire excluded — predictors only)
# ---------------------------------------------------------------------------

VARIABLES = {
    "tas":       "Mean Temp (°C)",
    "tasmax":    "Max Temp (°C)",
    "tasmin":    "Min Temp (°C)",
    "pr":        "Precipitation",
    "rsds":      "Solar Radiation",
    "PET":       "PET",
    "climbal":   "Climatic Water Balance",
    "elev":      "Elevation",
    "slope":     "Slope",
    "aspect_sin":    "Aspect Sine",
    "aspect_cos":    "Aspect Cosine",
    "distRoads": "Dist. to Roads",
    "distUrban": "Dist. to Urban",
    "distWater": "Dist. to Water",
}

# ---------------------------------------------------------------------------
# Seaborn / matplotlib theme
# ---------------------------------------------------------------------------

sns.set_theme(style="whitegrid", font="DejaVu Sans")
plt.rcParams.update({
    "axes.titlesize":  12,
    "axes.labelsize":  10,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "figure.dpi":      150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

COLOR      = "#2E86AB"
COLOR_CUM  = "#E84855"

# ---------------------------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------------------------

def load_and_scale(path: str):
    """Load CSV, apply unit conversions, standardise, return scaled array + labels."""
    df = pd.read_csv(path)

    # Unit conversions (consistent with other scripts)
    for col in ["tas", "tasmax", "tasmin"]:
        df[col] = df[col] - 273.15
    df["pr"] = df["pr"] * 2_592_000

    cols = list(VARIABLES.keys())
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in CSV: {missing}")

    X = df[cols].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, list(VARIABLES.values())

# ---------------------------------------------------------------------------
# Plot 1 — Scree plot
# ---------------------------------------------------------------------------

def plot_scree(pca: PCA, output_path: str):
    n = len(pca.explained_variance_ratio_)
    components = np.arange(1, n + 1)
    var_exp    = pca.explained_variance_ratio_ * 100
    cum_var    = np.cumsum(var_exp)

    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    ax1.bar(components, var_exp, color=COLOR, alpha=0.85, label="Per component")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance (%)", color=COLOR)
    ax1.tick_params(axis="y", labelcolor=COLOR)
    ax1.set_xticks(components)
    ax1.set_ylim(0, max(var_exp) * 1.25)

    ax2 = ax1.twinx()
    ax2.plot(components, cum_var, color=COLOR_CUM, marker="o",
             linewidth=1.8, markersize=5, label="Cumulative")
    ax2.axhline(90, color=COLOR_CUM, linewidth=0.8, linestyle="--", alpha=0.6)
    ax2.set_ylabel("Cumulative Variance (%)", color=COLOR_CUM)
    ax2.tick_params(axis="y", labelcolor=COLOR_CUM)
    ax2.set_ylim(0, 105)

    # Combined legend
    handles = [
        mpatches.Patch(color=COLOR, alpha=0.85, label="Per component"),
        plt.Line2D([0], [0], color=COLOR_CUM, marker="o", markersize=5, label="Cumulative"),
    ]
    ax1.legend(handles=handles, loc="center right", framealpha=0.9)
    ax1.set_title("PCA — Explained Variance per Component")
    ax1.grid(axis="y", linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")

# ---------------------------------------------------------------------------
# Plot 2 — Loadings heatmap
# ---------------------------------------------------------------------------

def plot_loadings(pca: PCA, labels: list, output_path: str):
    # Show only components that together explain ≥ 90% of variance
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    n_components = int(np.searchsorted(cum_var, 0.90)) + 1
    n_components = min(n_components, pca.n_components_)

    loadings = pd.DataFrame(
        pca.components_[:n_components].T,
        index=labels,
        columns=[f"PC{i+1}" for i in range(n_components)],
    )

    fig, ax = plt.subplots(figsize=(n_components * 0.9 + 3, len(labels) * 0.55 + 1.5))
    sns.heatmap(
        loadings,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-1, vmax=1,
        annot=True, fmt=".2f",
        annot_kws={"size": 8},
        linewidths=0.4,
        cbar_kws={"label": "Loading", "shrink": 0.8},
    )
    ax.set_title("PCA Loadings — Variable Contributions per Component")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")

# ---------------------------------------------------------------------------
# Plot 3 — Biplot (PC1 vs PC2)
# ---------------------------------------------------------------------------

def plot_biplot(pca: PCA, labels: list, output_path: str):
    loadings = pca.components_[:2].T   # shape (n_vars, 2)
    var_exp  = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(7, 7))

    # Draw unit circle for reference
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="#cccccc",
            linewidth=0.8, linestyle="--", zorder=0)

    # Draw axes through origin
    ax.axhline(0, color="#cccccc", linewidth=0.6, zorder=0)
    ax.axvline(0, color="#cccccc", linewidth=0.6, zorder=0)

    # Plot loading vectors
    for i, (x, y) in enumerate(loadings):
        ax.annotate(
            "", xy=(x, y), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color=COLOR,
                            lw=1.6, mutation_scale=12),
        )
        # Offset label slightly beyond arrow tip
        offset = 0.07
        ax.text(
            x * (1 + offset), y * (1 + offset),
            labels[i],
            fontsize=8, ha="center", va="center",
            color="#222222",
        )

    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}% variance explained)", fontsize=10)
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}% variance explained)", fontsize=10)
    ax.set_title("PCA Biplot — Variable Loadings (PC1 vs PC2)")
    ax.set_aspect("equal")
    ax.grid(linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading and scaling data...")
    X_scaled, labels = load_and_scale(INPUT_FILE)
    print(f"  {X_scaled.shape[0]:,} samples, {X_scaled.shape[1]} variables.")

    print("Fitting PCA...")
    pca = PCA(n_components=X_scaled.shape[1])
    pca.fit(X_scaled)

    print("Plotting scree plot...")
    plot_scree(pca, os.path.join(OUTPUT_DIR, "scree_plot.png"))

    print("Plotting loadings heatmap...")
    plot_loadings(pca, labels, os.path.join(OUTPUT_DIR, "loadings_heatmap.png"))

    print("Plotting biplot...")
    plot_biplot(pca, labels, os.path.join(OUTPUT_DIR, "biplot.png"))

    print("Done.")


if __name__ == "__main__":
    main()
