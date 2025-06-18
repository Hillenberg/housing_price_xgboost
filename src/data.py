# src/data.py
"""
Funktionen für Daten-Input, -Output und -Aufteilung.
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import RAW_DIR, PROC_DIR, SEED


def load_raw_data(filename: str) -> pd.DataFrame: # Type hints
    """
    Lädt einen CSV-Datensatz aus dem RAW_DIR.
    :param filename: Dateiname im RAW_DIR (z.B. 'train.csv').
    :return: DataFrame mit Rohdaten.
    """
    path = os.path.join(RAW_DIR, filename)
    return pd.read_csv(path)


def save_processed_data(df: pd.DataFrame, filename: str):
    """
    Speichert einen DataFrame als CSV im PROC_DIR.
    :param df: DataFrame, das gespeichert werden soll.
    :param filename: Dateiname im PROC_DIR (z.B. 'train_clean.csv').
    """
    os.makedirs(PROC_DIR, exist_ok=True)
    path = os.path.join(PROC_DIR, filename)
    df.to_csv(path, index=False)


def load_processed_data(filename: str) -> pd.DataFrame:
    """
    Lädt einen CSV-Datensatz aus dem PROC_DIR.
    :param filename: Dateiname im PROC_DIR.
    :return: DataFrame mit verarbeiteten Daten.
    """
    path = os.path.join(PROC_DIR, filename)
    return pd.read_csv(path)


def split_train_val(df: pd.DataFrame, target: str, test_size: float = 0.2, random_state: int = SEED):
    """
    Teilt DataFrame in Trainings- und Validierungs-Subset.
    :param df: Input-DataFrame mit Target-Spalte.
    :param target: Name der Zielspalte (z.B. 'SalePrice').
    :param test_size: Anteil für Validierung.
    :param random_state: Seed für Reproduzierbarkeit.
    :return: X_train, X_val, y_train, y_val
    """
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
