# src/train.py
"""
Training-Skript für das Housing-Price-Projekt mit MLflow-Integration.
Führt Cross-Validation durch, loggt Parameter und Metriken in MLflow,
trainiert das finale Modell auf dem gesamten Datensatz und speichert die Pipeline.
"""
import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from src.config import PROC_DIR, MODEL_DIR, N_SPLITS, SEED, MODEL_TYPE, XGB_PARAMS, LGBM_PARAMS, LASSO_PARAMS
from src.data import load_processed_data
from src.pipeline import build_pipeline


def main():
    # 1) MLflow Experiment konfigurieren
    mlflow.set_experiment("house-price-prediction")
    with mlflow.start_run():
        mlflow.log_param("model_type", MODEL_TYPE)
        # 2) Daten laden
        df = load_processed_data("train_clean.csv")
        X = df.drop(columns=["Id", "SalePrice"])
        y = np.log1p(df["SalePrice"])

        # 3) Features ermitteln
        num_feats = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_feats = X.select_dtypes(include=["object", "category"]).columns.tolist()

        # 4) Pipeline bauen
        pipeline = build_pipeline(num_feats, cat_feats, use_power_transform=True)

        # 5) Parameter loggen
        mlflow.log_param("n_splits", N_SPLITS)
        mlflow.log_param("seed", SEED)
        # logge Hyperparameter je nach Modelltyp
        if MODEL_TYPE == "xgb":
            mlflow.log_params(XGB_PARAMS)
        elif MODEL_TYPE == "lgbm":
            mlflow.log_params(LGBM_PARAMS)
        elif MODEL_TYPE == "lasso":
            mlflow.log_params(LASSO_PARAMS)
        else:
            raise ValueError(f"Unknown MODEL_TYPE {MODEL_TYPE}")

        # 6) Cross-Validation
        cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        scores = cross_val_score(
            pipeline, X, y,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1
        )
        rmse_scores = -scores
        mean_rmse = rmse_scores.mean()

        # 7) Metriken loggen
        mlflow.log_metric("mean_cv_rmse", mean_rmse)
        for i, score in enumerate(rmse_scores, start=1):
            mlflow.log_metric(f"rmse_fold_{i}", score)

        print(f"CV RMSE scores: {rmse_scores}")
        print(f"Mean CV RMSE: {mean_rmse:.4f}")

        # 8) Finales Modell trainieren & speichern
        pipeline.fit(X, y)

        # Feature-Importances aus dem Modell holen
        modeltype = pipeline.named_steps["model"]
        importances = modeltype.feature_importances_
        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

        # in Series verpacken und sortieren
        fi = pd.Series(importances, index=feature_names)
        fi_top10 = fi.sort_values(ascending=False).head(10)

        # Plot erzeugen
        fig, ax = plt.subplots(figsize=(8, 6))
        fi_top10.sort_values().plot.barh(ax=ax)
        ax.set_title("Top 10 Feature Importances")
        ax.set_xlabel("Importance")
        plt.tight_layout()

        # Plot als Artefakt loggen
        mlflow.log_figure(fig, "feature_importances_top10.png")

        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, "pipeline.pkl")
        joblib.dump(pipeline, model_path)
        print(f"Model pipeline saved to {model_path}")

        input_example = X.head(5)
        signature = infer_signature(input_example, pipeline.predict(input_example))

        # 9) Modell als MLflow Artefakt loggen
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=signature,
            input_example=input_example
        )
        mlflow.log_artifact(model_path)

if __name__ == "__main__":
    main()
