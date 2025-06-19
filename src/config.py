# src/config.py
"""
Konfigurationsdatei für das Housing-Price-Projekt.
Definiert Projekt-Root, Pfade, Random-Seeds und grundsätzliche Parameter.
"""
import os

# Projekt-Root ermitteln (Verzeichnis eine Ebene oberhalb von src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Verzeichnisse relativ zum Projekt-Root
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")      # Rohdaten
PROC_DIR = os.path.join(PROJECT_ROOT, "data", "processed")  # Verarbeitete Daten
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")         # Modelle und Artefakte

# Reproduzierbarkeit
SEED = 42  # Zufalls-Seed für NumPy, scikit-learn, XGBoost

# Cross-Validation
N_SPLITS = 5  # Anzahl Splits für K-Fold CV

# Welches Modell soll die Pipeline benutzen? Optionen: "xgb", "lgbm", "lasso"
MODEL_TYPE = "xgb"

# Parameter-Dicts für jedes Modell
LGBM_PARAMS  = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "random_state": SEED
}

LASSO_PARAMS = {
    "alpha": 1.0,
    "random_state": SEED
}

# Default-Hyperparameter für XGBRegressor (kann später überschrieben werden)
XGB_PARAMS = {
    "n_estimators": 1500,
    "learning_rate": 0.015,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "random_state": SEED,
    "eval_metric": "rmse"
}