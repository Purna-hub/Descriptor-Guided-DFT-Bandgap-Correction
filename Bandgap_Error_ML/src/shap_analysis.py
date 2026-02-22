import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def run_shap(model, X_test, feature_names):

    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    X_test = pd.DataFrame(X_test, columns=feature_names)

    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)

    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig("figures/Figure3_SHAP_summary.png", dpi=600)
    plt.close()

    mean_shap = np.abs(shap_values.values).mean(axis=0)

    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Absolute_SHAP": mean_shap
    }).sort_values(by="Mean_Absolute_SHAP", ascending=False)

    shap_df.to_csv("results/shap_feature_importance.csv", index=False)