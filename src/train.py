# src/train.py
"""
Training-Skript für das Housing-Price-Projekt mit MLflow-Integration.
Führt Cross-Validation durch, loggt Parameter und Metriken in MLflow,
trainiert das finale Modell auf dem gesamten Datensatz und speichert die Pipeline.
"""
import os
import numpy as np
import joblib
from sklearn.model_selection import KFold, cross_val_score

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from src.config import PROC_DIR, MODEL_DIR, N_SPLITS, SEED, XGB_PARAMS
from src.data import load_processed_data
from src.pipeline import build_pipeline


def main():
    # 1) MLflow Experiment konfigurieren
    mlflow.set_experiment("house-price-prediction")
    with mlflow.start_run():
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
        mlflow.log_params(XGB_PARAMS)

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
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, "pipeline.pkl")
        joblib.dump(pipeline, model_path)
        print(f"Model pipeline saved to {model_path}")

        # 9) Modell als MLflow Artefakt loggen
        # Beispiel-Inputs als DataFrame (z.B. die ersten 5 Zeilen deines Trainings-Featuresets)
        input_example = X.head(5)

        # Automatisch Signatur (Schema) aus Input und Prediction ermitteln
        signature = infer_signature(input_example, pipeline.predict(input_example))

        mlflow.sklearn.log_model(
             pipeline,
             name="model",
             signature=signature,
             input_example=input_example
         )

        mlflow.log_artifact(model_path)

if __name__ == "__main__":
    main()