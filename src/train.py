# src/train.py
"""
Training-Skript für das Housing-Price-Projekt.
Führt Cross-Validation durch, loggt die RMSE-Werte, trainiert das finale Modell auf dem gesamten Datensatz
und speichert die Pipeline inklusive Modell.
"""
import os
import numpy as np
import joblib
from sklearn.model_selection import KFold, cross_val_score

from src.config import PROC_DIR, MODEL_DIR, N_SPLITS, SEED
from src.data import load_processed_data
from src.pipeline import build_pipeline


def main():
    # 1) Geladene bereinigte Trainingsdaten
    df = load_processed_data("train_clean.csv")

    # 2) Merkmale und Ziel definieren (ID fallen lassen)
    X = df.drop(columns=["Id", "SalePrice"])
    y = np.log1p(df["SalePrice"])  # Ziel log-transformieren

    # 3) Feature-Listen automatisch ableiten
    num_feats = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_feats = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # 4) Pipeline bauen
    model = build_pipeline(num_feats, cat_feats, use_power_transform=True)

    # 5) Cross-Validation
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    scores = cross_val_score(
        model, X, y,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )
    rmse_scores = -scores  # negierte Scores positiv machen
    print(f"CV RMSE scores: {rmse_scores}")
    print(f"Mean CV RMSE: {rmse_scores.mean():.4f}")

    # 6) Finales Modell trainieren und speichern
    model.fit(X, y)
    os.makedirs(MODEL_DIR, exist_ok=True)
    out_path = os.path.join(MODEL_DIR, "pipeline.pkl")
    joblib.dump(model, out_path)
    print(f"Model pipeline saved to {out_path}")


if __name__ == "__main__":
    main()