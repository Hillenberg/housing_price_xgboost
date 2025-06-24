# Housing Price Prediction 📦🏠

Dieses Repository enthält ein vollständiges End-to-End-Projekt zur Vorhersage von Immobilienpreisen mit Python, scikit-learn, XGBoost/LightGBM/Lasso und MLflow-Tracking. Das Projekt ist containerisiert mit Docker und ermöglicht einfache Batch- und Online-Inferenz.

---

## ⚙️ Projektstruktur

```text
housing-price/
├── src/                  # Quellcode
│   ├── config.py         # zentrale Konfiguration
│   ├── data.py           # Datenlade-Funktionen
│   ├── pipeline.py       # Preprocessing- und ML-Pipeline
│   ├── train.py          # Training + MLflow-Logging
│   ├── tune.py           # Hyperparameter-Tuning (RandomizedSearchCV)
│   └── predict.py        # Batch-Inferenz via CLI
├── data/
│   └── raw/              # rohes Kaggle-Dataset (train.csv, test.csv)
├── models/               # gespeicherte Pipelines (.pkl)
├── mlruns/               # MLflow-Tracking-Store
├── requirements.txt      # Python-Abhängigkeiten
├── Dockerfile            # Container-Build-Anweisungen
└── README.md             # Projektbeschreibung (diese Datei)
```

---

## 🚀 Installation & Setup (lokal)

1. Repository klonen:

   ```bash
   git clone <repo-url>
   cd housing-price
   ```
2. Virtuelle Umgebung erstellen & aktivieren:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate     # macOS/Linux
   # .\.venv\Scripts\activate # Windows
   ```
3. Abhängigkeiten installieren:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. MLflow-UI starten (optional):

   ```bash
   mlflow ui --port 5000
   ```

   Browser: [http://localhost:5000](http://localhost:5000)

---

## 🎯 Training

```bash
python -m src.train
```

* Lädt `data/raw/train_clean.csv`, führt K-Fold CV durch, loggt Parameter & Metriken in MLflow.
* Trainiert final auf gesamtem Datensatz mit Early Stopping (bei XGBoost) und speichert `models/pipeline.pkl`.

---

## 🔍 Hyperparameter-Tuning

```bash
python -m src.tune
```

* Führt RandomizedSearchCV durch, loggt `best_cv_rmse` und `best_params` in MLflow-Experiment `house-price-hpo`.
* Speichert getunte Pipeline in `models/pipeline_hpo.pkl`.

---

## 🚀 Inferenz

Standardmäßig werden Vorhersagen für das Kaggle-Testset (`data/raw/test.csv`) erzeugt und in `submission.csv` geschrieben:

```bash
python -m src.predict
```

## 🐳 Docker

1. Image bauen:

   ```bash
   docker build -t housing-price:latest .
   ```
2. MLflow-UI im Container:

   ```bash
   docker run -d --name hp-mlflow -p 5001:5000 -v $(pwd)/mlruns:/app/mlruns housing-price:latest
   # UI: http://localhost:5001
   ```
3. Batch-Predict im Container:

   ```bash
   docker run --rm -v $(pwd)/models:/app/models -v $(pwd)/data/raw:/app/data/raw housing-price:latest python -m src.predict
   ```

---

## 🎛 Konfiguration

Alle Hyperparameter, Pfade und Settings zentral in `src/config.py`. Einfach dort z.B. `MODEL_TYPE = "lgbm"` umstellen oder XGB\_PARAMS anpassen.

---

## 📊 MLflow-Tracking

* Experimente unter `mlruns/` persistiert
* UI: `mlflow ui` (host/port konfigurierbar)
* Artefakte (Plots, Modelle) direkt im Run verfügbar
<img width="1498" alt="image" src="https://github.com/user-attachments/assets/b9c84df2-f39d-4951-b053-7184b0da4cfa" />

---

## 📂 Gitignore

Empfohlen, folgende Dateien/Ordner zu ignorieren (`.gitignore`):

```
.venv/
__pycache__/
mlruns/
models/
```

---

## 📈 Ergebnisse

* **Best RMSE:** erzielt mit **XGBRegressor**.<br>
* **Top 130**-Platzierung auf dem Kaggle-Leaderboard der Housing Price Challenge.
<img width="1182" alt="image" src="https://github.com/user-attachments/assets/528f1fab-1517-4614-9207-d61ed0504f81" />


---

