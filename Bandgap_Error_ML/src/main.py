import os
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

    set_seed(RANDOM_SEED)
    create_directories()

    # Resolve dataset path properly
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(BASE_DIR, "data", "raw", "bandgap_dataset.csv")

    df = pd.read_csv(dataset_path)

    required_columns = ["Composition", "Eg_exp", "Eg_DFT"]
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = compute_features(df)

    processed_path = os.path.join(
        BASE_DIR,
        "data",
        "processed",
        "feature_engineered_dataset.csv"
    )

    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)

    best_model, best_model_name, results_df, X_test = train_models(df)

    plot_parity_and_residual(results_df)

    descriptors = [
        "delta_chi",
        "delta_r",
        "VEC",
        "Z_avg",
        "delta_atomic_volume"
    ]

    df = compute_absolute_residual(df, results_df)
    df = create_quartile_bins(df, descriptors)

    plot_regime_analysis(df, descriptors)

    run_shap(best_model, X_test, FEATURE_COLUMNS)


if __name__ == "__main__":
    main()