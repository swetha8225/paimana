#!/usr/bin/env bash
# PAIMANA one-command launcher (Linux/macOS).
# Installs backend deps if needed, builds the DB + models if missing,
# then serves the app (API + built frontend) on http://localhost:8000
set -e
cd "$(dirname "$0")"

echo "== PAIMANA launcher =="

# 1. python deps
if ! python3 -c "import fastapi, catboost, lifelines, shap" 2>/dev/null; then
  echo "[setup] installing backend dependencies..."
  pip install -q -r backend/requirements.txt
fi

# 2. dataset (built from the canonical panel CSV)
if [ ! -f data/paimana_panel.csv ]; then
  echo "[data] building panel from parsed source CSVs..."
  (cd data_pipeline && python3 build_dataset.py)
fi

# 3. train models + build DB (skip if artifacts already exist)
if [ ! -f backend/artifacts/models.joblib ] || [ ! -f backend/data/paimana.db ]; then
  echo "[ml] training pipeline (~1 min)..."
  (cd backend && python3 -m app.ml.pipeline)
fi

# 4. frontend build (skip if already built)
if [ ! -f backend/app/static/index.html ]; then
  echo "[frontend] building React app..."
  (cd frontend && npm install --no-audit --no-fund && npm run build)
fi

echo
echo "== PAIMANA ready: http://localhost:8000 =="
echo "   (retrain any time: cd backend && python3 -m app.ml.pipeline)"
cd backend && python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
