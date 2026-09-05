"""Application configuration: paths and configurable risk-score weights."""
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'data')
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts')
DB_PATH = os.path.join(BASE_DIR, 'data', 'paimana.db')
PANEL_CSV = os.path.join(DATA_DIR, 'paimana_panel.csv')
MANIFEST_JSON = os.path.join(DATA_DIR, 'source_manifest.json')
STATIC_DIR = os.path.join(BASE_DIR, 'app', 'static')

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# Default component weights of the transparent implementation-risk score.
# Configurable at runtime via PUT /api/config/risk-weights (persisted below).
DEFAULT_RISK_WEIGHTS = {
    'cost_risk': 0.35,
    'schedule_risk': 0.35,
    'expenditure_risk': 0.20,
    'reporting_risk': 0.10,
}
WEIGHTS_FILE = os.path.join(ARTIFACTS_DIR, 'risk_weights.json')

RISK_LEVELS = [
    (30, 'Low'), (55, 'Moderate'), (75, 'High'), (101, 'Critical'),
]

# Labels / thresholds
RECALL_PRIORITY = True          # early warning: missing a risky project is worse
MIN_REGRESSION_OBS = 150        # below this, refuse to train magnitude models
MIN_SURVIVAL_EVENTS = 150       # below this, survival analysis not appropriate
MIN_FORECAST_OBS = 12           # monthly points needed for per-project stats models


def load_risk_weights() -> dict:
    if os.path.exists(WEIGHTS_FILE):
        try:
            with open(WEIGHTS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return dict(DEFAULT_RISK_WEIGHTS)


def save_risk_weights(w: dict) -> None:
    with open(WEIGHTS_FILE, 'w') as f:
        json.dump(w, f, indent=2)


def risk_level(score: float) -> str:
    for cap, name in RISK_LEVELS:
        if score < cap:
            return name
    return 'Critical'
