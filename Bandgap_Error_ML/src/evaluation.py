import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.metrics import r2_score
import pandas as pd
import os


def plot_parity_and_residual(results_df):

    results_df["Eg_ML"] = (
        results_df["Eg_DFT"] +
        results_df["Predicted_delta_Eg"]
    )

    Eg_exp = results_df["Eg_exp"].values
    Eg_DFT = results_df["Eg_DFT"].values
    Eg_ML  = results_df["Eg_ML"].values

    mae_dft = np.mean(np.abs(Eg_exp - Eg_DFT))
    mae_ml  = np.mean(np.abs(Eg_exp - Eg_ML))

    r2_dft = r2_score(Eg_exp, Eg_DFT)
    r2_ml  = r2_score(Eg_exp, Eg_ML)

    # Save metrics
    os.makedirs("results", exist_ok=True)

    metrics_df = pd.DataFrame({
        "Model": ["DFT", "ML-corrected"],
        "MAE": [mae_dft, mae_ml],
        "R2": [r2_dft, r2_ml]
    })

    metrics_df.to_csv("results/model_performance.csv", index=False)

    res_dft = Eg_exp - Eg_DFT
    res_ml  = Eg_exp - Eg_ML

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Parity
    ax = axes[0]
    ax.scatter(Eg_exp, Eg_DFT, alpha=0.6, label="DFT (PBE)")
    ax.scatter(Eg_exp, Eg_ML, alpha=0.6, label="ML-corrected")

    min_val = min(Eg_exp.min(), Eg_DFT.min(), Eg_ML.min())
    max_val = max(Eg_exp.max(), Eg_DFT.max(), Eg_ML.max())

    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    ax.set_xlabel("Experimental Band Gap (eV)")
    ax.set_ylabel("Predicted Band Gap (eV)")
    ax.legend()
    ax.set_title("(a)")

    # Residual
    ax = axes[1]
    x_grid = np.linspace(min(res_dft.min(), res_ml.min()),
                         max(res_dft.max(), res_ml.max()), 500)

    kde_dft = gaussian_kde(res_dft)
    kde_ml  = gaussian_kde(res_ml)

    ax.plot(x_grid, kde_dft(x_grid), label="DFT (PBE)")
    ax.plot(x_grid, kde_ml(x_grid), label="ML-corrected")
    ax.axvline(0.0, linestyle="--")

    ax.set_xlabel("Residual (eV)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_title("(b)")

    plt.tight_layout()
    plt.savefig("figures/Figure1_Parity_Residual.png", dpi=600)
    plt.close()

    return mae_ml