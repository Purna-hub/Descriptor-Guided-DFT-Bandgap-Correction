from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import xgboost as xgb
import pandas as pd
import numpy as np
import os

from config import FEATURE_COLUMNS, TARGET_COLUMN, TEST_SIZE, RANDOM_SEED


def train_models(df):

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED
    )

    models = {
        "Linear_Regression": LinearRegression(),
        "Random_Forest": RandomForestRegressor(random_state=RANDOM_SEED),
        "Gradient_Boosting": GradientBoostingRegressor(random_state=RANDOM_SEED),
        "XGBoost": xgb.XGBRegressor(random_state=RANDOM_SEED),
        "SVR": SVR()
    }

    results = []

    # Create output directories
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/individual_models", exist_ok=True)

    for name, model in models.items():

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        cv_scores = cross_val_score(
            model,
            X,
            y,
            scoring="neg_mean_absolute_error",
            cv=5
        )

        cv_mae = -cv_scores.mean()
        cv_std = cv_scores.std()

        # Save individual model performance
        individual_df = pd.DataFrame({
            "Metric": ["MAE", "RMSE", "R2", "CV_MAE", "CV_STD"],
            "Value": [mae, rmse, r2, cv_mae, cv_std]
        })

        individual_df.to_csv(
            f"results/individual_models/{name}_performance.csv",
            index=False
        )

        results.append([name, mae, rmse, r2, cv_mae, cv_std])

    # Save overall comparison table
    metrics_df = pd.DataFrame(
        results,
        columns=["Model", "MAE", "RMSE", "R2", "CV_MAE", "CV_STD"]
    )

    metrics_df.to_csv("results/model_comparison.csv", index=False)

    # Select best model (lowest MAE)
    best_row = metrics_df.sort_values("MAE").iloc[0]
    best_model_name = best_row["Model"]
    best_model = models[best_model_name]

    print(f"Best model selected: {best_model_name}")

    # Retrain best model
    best_model.fit(X_train, y_train)
    y_pred_best = best_model.predict(X_test)

    results_df = df.loc[X_test.index].copy()
    results_df["Predicted_delta_Eg"] = y_pred_best

    return best_model, best_model_name, results_df, X_test