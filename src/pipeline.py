# src/pipeline.py
"""
Preprocessing- und Modell-Pipeline für das Housing-Price-Projekt.
Definiert einen ColumnTransformer für numerische und kategoriale Features
eingebettet in eine vollständige Pipeline mit XGBRegressor.
"""
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from xgboost import XGBRegressor

from src.config import XGB_PARAMS


def build_pipeline(num_feats, cat_feats, use_power_transform=True):
    """
    Baut eine ML-Pipeline auf:
      - Preprocessing (Imputation, Skalierung, optionaler Power-Transform)
      - XGBRegressor mit Default-Hyperparametern aus config

    :param num_feats: Liste numerischer Feature-Namen
    :param cat_feats: Liste kategorialer Feature-Namen
    :param use_power_transform: Wenn True, wird PowerTransformer auf numerische Features angewendet
    :return: sklearn.pipeline.Pipeline-Objekt
    """
    # Numerische Transformationen
    num_steps = [
        ('imputer', SimpleImputer(strategy='median'))
    ]
    if use_power_transform:
        num_steps.append(('power', PowerTransformer(method='yeo-johnson')))
    num_steps.append(('scaler', StandardScaler()))
    numeric_transformer = Pipeline(steps=num_steps)

    # Kategoriale Transformationen
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # ColumnTransformer kombiniert numerische und kategoriale Pfade
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, num_feats),
        ('cat', categorical_transformer, cat_feats)
    ], remainder='drop')

    # Vollständige Pipeline mit Vorverarbeitung und Modell
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', XGBRegressor(**XGB_PARAMS))
    ])

    return pipeline
