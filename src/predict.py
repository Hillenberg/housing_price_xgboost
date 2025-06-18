# src/predict.py
"""
Erzeugt eine Kaggle-Submission aus dem gespeicherten Pipeline-Modell.
Lädt das Training-Pipeline-Objekt, wendet es auf den Test-Datensatz an
und speichert die Vorhersagen als submission.csv.
"""
import os
import numpy as np
import pandas as pd
import joblib

from src.config import RAW_DIR, MODEL_DIR
from src.data import load_raw_data


def main():
    # 1) Lade das trainierte Pipeline-Modell
    model_path = os.path.join(MODEL_DIR, "pipeline.pkl")
    model = joblib.load(model_path) #Model ist quasi die gesamte Pipeline
    print(f"Loaded model from {model_path}")

    # 2) Test-Daten einlesen
    test_df = load_raw_data("test.csv")
    # ID-Spalte zwischenspeichern
    ids = test_df["Id"]
    # ID nicht als Feature nutzen
    X_test = test_df.drop(columns=["Id"])

    # 3) Vorhersagen auf Log-Skala und Rücktransformation
    preds_log = model.predict(X_test)
    preds = np.expm1(preds_log)

    # 4) Submission-DataFrame erstellen
    submission = pd.DataFrame({
        "Id": ids,
        "SalePrice": preds
    })

    # 5) Als CSV speichern
    out_file = "submission.csv"
    submission.to_csv(out_file, index=False)
    print(f"Submission saved to {out_file}")


if __name__ == "__main__":
    main()