import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


# ---------------------------------------------------------------------
# Resolve project directories
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURE_DIR = os.path.join(BASE_DIR, "figures")

os.makedirs(FIGURE_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# Compute absolute residual
# ---------------------------------------------------------------------
def compute_absolute_residual(df, results_df):
    """
    Compute |Eg_exp - Eg_ML|
    """
    df["Abs_delta_Eg"] = np.abs(
        results_df["Eg_exp"] - results_df["Eg_ML"]
    )
    return df


# ---------------------------------------------------------------------
# Create quartile bins
# ---------------------------------------------------------------------
def create_quartile_bins(df, descriptors):
    """
    Create quartile bins (equal-frequency) for descriptors.
    """

    for desc in descriptors:
        df[f"{desc}_bin"] = pd.qcut(
            df[desc],
            q=4,
            duplicates="drop"
        )

    return df


# ---------------------------------------------------------------------
# Helper: Format numeric range
# ---------------------------------------------------------------------
def _format_interval(interval):
    left = round(interval.left, 2)
    right = round(interval.right, 2)
    return f"{left}–{right}"


# ---------------------------------------------------------------------
# Plot regime analysis
# ---------------------------------------------------------------------
def plot_regime_analysis(df, descriptors):
    """
    Generate regime analysis figure:
    3 panels on top, 2 on bottom.
    """

    colors = [
        "#4C72B0",
        "#DD8452",
        "#55A868",
        "#C44E52",
        "#8172B3"
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
    axes = axes.flatten()

    for i, desc in enumerate(descriptors):

        grouped = (
            df.groupby(f"{desc}_bin", observed=False)["Abs_delta_Eg"]
            .mean()
        )

        # Compute Spearman properly (avoid NaN)
        valid = df[[desc, "Abs_delta_Eg"]].dropna()
        rho, pval = spearmanr(valid[desc], valid["Abs_delta_Eg"])

        ax = axes[i]

        grouped.plot(
            kind="bar",
            ax=ax,
            color=colors[i],
            edgecolor="black",
            linewidth=0.8
        )

        # Numeric range labels
        range_labels = [
            _format_interval(interval)
            for interval in grouped.index
        ]

        ax.set_xticklabels(
            range_labels,
            rotation=30,
            ha="right",
            fontsize=10
        )

        ax.set_title(
            f"({chr(97+i)}) {desc}",
            fontsize=12,
            fontweight="bold"
        )

        ax.set_ylabel("Mean |ΔEg| (eV)", fontsize=12)
        ax.tick_params(axis="y", labelsize=11)

        ax.text(
            0.5,
            0.88,
            rf"$\rho$ = {rho:.2f}" + "\n" + rf"$p$ = {pval:.1e}",
            transform=ax.transAxes,
            ha="center",
            fontsize=11
        )

    # Remove unused 6th subplot
    fig.delaxes(axes[-1])

    plt.tight_layout()

    save_path = os.path.join(FIGURE_DIR, "Figure2_Regime_Analysis.png")
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()

    print(f"Figure 2 saved to: {save_path}")