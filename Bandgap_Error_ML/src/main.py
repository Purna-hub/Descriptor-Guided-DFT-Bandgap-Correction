import pandas as pd

from utils import set_seed, create_directories
from config import RANDOM_SEED, FEATURE_COLUMNS
from feature_engineering import compute_features
from model_training import train_models
from evaluation import plot_parity_and_residual
from shap_analysis import run_shap
from regime_analysis import (
    compute_absolute_residual,
    create_quartile_bins,
    plot_regime_analysis
)


def main():

    print("\n========== Descriptor-Guided ML Correction Pipeline ==========\n")

    # ------------------------------------------------------------------
    # 1️⃣ Reproducibility
    # ------------------------------------------------------------------
    set_seed(RANDOM_SEED)
    create_directories()

    # ------------------------------------------------------------------
    # 2️⃣ Load Dataset
    # ------------------------------------------------------------------
    print("Loading dataset...")
    df = pd.read_csv("data/raw/bandgap_dataset.csv")

    required_columns = ["Composition", "Eg_exp", "Eg_DFT"]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in dataset.")

    print(f"Dataset loaded successfully. Total samples: {len(df)}")

    # ------------------------------------------------------------------
    # 3️⃣ Feature Engineering
    # ------------------------------------------------------------------
    print("Performing feature engineering...")
    df = compute_features(df)

    df.to_csv(
        "data/processed/feature_engineered_dataset.csv",
        index=False
    )

    print("Feature engineering completed.")

    # ------------------------------------------------------------------
    # 4️⃣ Model Training & Comparison
    # ------------------------------------------------------------------
    print("Training models and performing comparison...")

    best_model, best_model_name, results_df, X_test = train_models(df)

    print(f"\nBest performer selected: {best_model_name}\n")

    # ------------------------------------------------------------------
    # 5️⃣ Figure 1 – Parity + Residual
    # ------------------------------------------------------------------
    print("Generating Figure 1 (Parity + Residual)...")

    plot_parity_and_residual(results_df)

    print("Figure 1 saved successfully.")

    # ------------------------------------------------------------------
    # 6️⃣ Figure 2 – Regime Analysis
    # ------------------------------------------------------------------
    print("Performing regime analysis (quartile-based)...")

    descriptors_for_regime = [
        "delta_chi",
        "delta_r",
        "VEC",
        "Z_avg",
        "delta_atomic_volume"
    ]

    df = compute_absolute_residual(df, results_df)
    df = create_quartile_bins(df, descriptors_for_regime)

    plot_regime_analysis(df, descriptors_for_regime)

    print("Figure 2 saved successfully.")

    # ------------------------------------------------------------------
    # 7️⃣ Figure 3 – SHAP Analysis (Best Model Only)
    # ------------------------------------------------------------------
    print("Running SHAP interpretability on best model...")

    run_shap(best_model, X_test, FEATURE_COLUMNS)

    print("Figure 3 saved successfully.")

    # ------------------------------------------------------------------
    # 8️ Final Summary
    # ------------------------------------------------------------------
    print("\n================= PIPELINE EXECUTED SUCCESSFULLY =================")
    print("Outputs generated in:")
    print("  → figures/")
    print("  → results/")
    print("  → data/processed/")
    print("=================================================================\n")


if __name__ == "__main__":
    main()