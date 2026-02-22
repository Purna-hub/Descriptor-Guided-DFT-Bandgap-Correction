import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import os


def compute_absolute_residual(df, results_df):

    df = df.copy()

    df["Abs_delta_Eg"] = np.nan
    df.loc[results_df.index, "Abs_delta_Eg"] = np.abs(
        results_df["Eg_exp"] -
        (results_df["Eg_DFT"] + results_df["Predicted_delta_Eg"])
    )

    return df


def create_quartile_bins(df, descriptors):

    quartile_ranges = []

    for desc in descriptors:

        # Compute quartile edges
        quartiles = df[desc].quantile([0.0, 0.25, 0.5, 0.75, 1.0]).values

        quartile_ranges.append([
            desc,
            quartiles[0], quartiles[1],
            quartiles[1], quartiles[2],
            quartiles[2], quartiles[3],
            quartiles[3], quartiles[4]
        ])

        df[f"{desc}_bin"] = pd.qcut(
            df[desc],
            q=4,
            duplicates="drop"
        )

    # Save Supplementary Table S1
    os.makedirs("results", exist_ok=True)

    quartile_df = pd.DataFrame(
        quartile_ranges,
        columns=[
            "Descriptor",
            "Q1_min", "Q1_max",
            "Q2_min", "Q2_max",
            "Q3_min", "Q3_max",
            "Q4_min", "Q4_max"
        ]
    )

    quartile_df.to_csv(
        "results/Supplementary_Table_S1_Quartile_Ranges.csv",
        index=False
    )

    return df


def plot_regime_analysis(df, descriptors):

    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

    fig, axes = plt.subplots(
        1,
        len(descriptors),
        figsize=(16, 4.5),
        sharey=True
    )

    regime_stats = []

    for i, desc in enumerate(descriptors):

        # Compute quartile edges
        edges = df[desc].quantile([0, 0.25, 0.5, 0.75, 1]).values
        edges = np.round(edges, 2)

        # Create formatted ranges
        ranges = [
            f"{edges[0]:.2f}–{edges[1]:.2f}",
            f"{edges[1]:.2f}–{edges[2]:.2f}",
            f"{edges[2]:.2f}–{edges[3]:.2f}",
            f"{edges[3]:.2f}–{edges[4]:.2f}",
        ]

        regime = (
            df.groupby(f"{desc}_bin", observed=False)["Abs_delta_Eg"]
            .mean()
        )

        rho, pval = spearmanr(
            df[desc],
            df["Abs_delta_Eg"],
            nan_policy="omit"
        )

        regime_stats.append([
            desc,
            round(rho, 3),
            "{:.2e}".format(pval)
        ])

        quartile_labels = ["Q1", "Q2", "Q3", "Q4"]

        bars = axes[i].bar(
            quartile_labels,
            regime.values,
            color=colors[i],
            edgecolor="black",
            linewidth=0.8
        )

        axes[i].set_title(
            f"({chr(97+i)}) {desc}",
            fontsize=10,
            fontweight="bold"
        )

        axes[i].set_xlabel("Quartile", fontsize=9)

        # Add range text below each bar
        for j, bar in enumerate(bars):
            axes[i].text(
                bar.get_x() + bar.get_width()/2,
                -0.05,  # slightly below x-axis
                ranges[j],
                ha="center",
                va="top",
                fontsize=7,
                rotation=45,
                transform=axes[i].get_xaxis_transform()
            )

        # Add rho and p-value
        axes[i].text(
            0.5,
            0.85,
            f"$\\rho$={rho:.2f}\n$p$={pval:.1e}",
            transform=axes[i].transAxes,
            ha="center",
            fontsize=8
        )

        axes[i].tick_params(axis="both", direction="in")

    axes[0].set_ylabel(
        "Mean |Residual| (eV)",
        fontsize=10,
        fontweight="bold"
    )

    plt.tight_layout()
    plt.savefig(
        "figures/Figure2_Regime_Analysis.png",
        dpi=600,
        bbox_inches="tight"
    )
    plt.close()

    print("Regime analysis completed.")