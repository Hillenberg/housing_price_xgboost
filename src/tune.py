# src/tune.py
"""
Hyperparameter-Tuning für Housing Price Prediction.
Verwendet RandomizedSearchCV auf der bestehenden Pipeline.
"""
import os
import numpy as np
import joblib

from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV, KFold

import mlflow
import mlflow.sklearn

from src.config import PROC_DIR, MODEL_DIR, N_SPLITS, SEED, MODEL_TYPE
from src.data import load_processed_data
from src.pipeline import build_pipeline

def main():
    # 1) MLflow-Experiment
    mlflow.set_experiment("house-price-hpo")
    with mlflow.start_run():
        mlflow.log_param("model_type", MODEL_TYPE)
        mlflow.log_param("n_splits", N_SPLITS)
        mlflow.log_param("seed", SEED)

        # 2) Daten laden
        df = load_processed_data("train_clean.csv")
        X = df.drop(columns=["Id", "SalePrice"])
        y = np.log1p(df["SalePrice"])

        # 3) Pipeline bauen
        num_feats = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_feats = X.select_dtypes(include=["object", "category"]).columns.tolist()
        base_pipe = build_pipeline(num_feats, cat_feats, use_power_transform=True)

        # 4) Suchraum definieren
        param_dist = {
            "model__max_depth":       randint(4, 6),
            "model__n_estimators":    randint(400, 900),
            "model__learning_rate":   uniform(0.01, 0.06),
            "model__subsample":       uniform(0.6, 0.4),
            "model__colsample_bytree":uniform(0.6, 0.4),
            "model__reg_alpha":       uniform(0, 0.1),         # L1 Regularisierung
            "model__reg_lambda":      uniform(2, 4),           # L2 Regularisierung
        }

        # 5) RandomizedSearchCV einrichten
        cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        search = RandomizedSearchCV(
            estimator=base_pipe,
            param_distributions=param_dist,
            n_iter=200,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=-1,
            random_state=SEED,
            verbose=2
        )

        # 6) Suche durchführen
        search.fit(X, y)

        # 7) Bestes Ergebnis loggen
        best_rmse = -search.best_score_
        best_params = search.best_params_
        mlflow.log_metric("best_cv_rmse", best_rmse)
        mlflow.log_params(best_params)

        print(f"Best CV RMSE: {best_rmse:.4f}")
        print("Best Params:", best_params)

        # 8) Final mit den besten Parametern trainieren und speichern
        best_pipe = search.best_estimator_
        best_pipe.fit(X, y)
        os.makedirs(MODEL_DIR, exist_ok=True)
        out_path = os.path.join(MODEL_DIR, "pipeline_hpo.pkl")
        joblib.dump(best_pipe, out_path)
        mlflow.sklearn.log_model(best_pipe, name="model_hpo")
        print(f"Tuned pipeline saved to {out_path}")

if __name__ == "__main__":
    main()