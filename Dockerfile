# 1) Basis-Image: schlankes Python-Image
FROM python:3.11-slim

# 2) Arbeitsverzeichnis im Container
WORKDIR /app

# 3) Abhängigkeiten installieren
#    Kopiere nur requirements.txt, um Caching zu nutzen
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 4) Deinen gesamten Projekt-Code ins Image kopieren
COPY . .

# 5) Port freigeben (für MLflow UI)
EXPOSE 5000

# 6) Standardbefehl: Starte MLflow-UI auf Port 5000
CMD ["mlflow", "ui", "--host", "0.0.0.0", "--port", "5000"]