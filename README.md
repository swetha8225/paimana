# PAIMANA

**Pro-active Analytics for Infrastructure Monitoring and Assessment (National Analytics)**
— SIH 2026 problem statement SIH26103 — built on **real, public MoSPI data**.

PAIMANA ingests the Ministry of Statistics & Programme Implementation (MoSPI)
**Quarterly Progress Information System Reports (QPISR)** and **Flash Reports**
on central-sector infrastructure projects (₹150 Cr+), builds a clean project
panel, trains leak-safe overrun-prediction models, and serves a full-stack
analytics dashboard with early-warning risk scores, transparent warnings,
explainability (SHAP), what-if simulation, benchmarking and a natural-language
assistant.

---

## Quick start

```bash
./run.sh                 # installs deps if needed, trains, serves on :8000
```

Or step by step:

```bash
# 1. dataset (optional — data/paimana_panel.csv is already committed)
cd data_pipeline && python3 build_dataset.py

# 2. train all models + build SQLite DB (~45 s)
cd ../backend && python3 -m app.ml.pipeline

# 3. serve API + built frontend
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Frontend development mode: `cd frontend && npm install && npm run dev`
(Vite on :5173, proxies `/api` to :8000). The production build is committed at
`backend/app/static/`.

## Layout

```
data_source/pdfs/         10 public MoSPI PDFs (6 QPISR + 4 flash reports)
data_pipeline/            PDF parsers + canonical panel builder
  parse_flash.py          Apr/May 2024 flash layout
  parse_qpisr.py          QPISR-style census/completed tables (x-band
                          calibration anchored to each PDF's own headers)
  build_dataset.py        canonical panel: dedup, sector/ministry/state fixes,
                          derived fields, validations, source manifest
  interim/                per-report parsed CSVs
data/
  paimana_panel.csv       11,121 project-month rows · 2,163 projects · 6 months
  source_manifest.json    sources, formulas, validation results
backend/
  app/ml/pipeline.py      trains everything, scores projects, writes DB+card
  app/ml/models.py        classification (5 models), regression (5), survival
  app/ml/risk.py          transparent risk score + warnings + recommendations
  app/ml/explain.py       SHAP global/local explanations
  app/ml/forecast.py      trend/forecast honesty layer
  app/services/assistant.py  rule-based NL interface over the DB
  app/api.py              REST API
  artifacts/              model_card.json (all metrics), models.joblib
  data/paimana.db         SQLite (panel, scores, warnings)
frontend/                 React + TypeScript + Recharts dashboard
screenshots/              rendered-page captures
```

## Dataset (all public, parsed from source PDFs)

| Month | Report | Ongoing | Completed | Closed unfinished |
|---|---|---|---|---|
| Apr 2024 | Flash Report | 1,817 | 51 | 11 |
| May 2024 | Flash Report | 1,789 | 16 | 14 |
| Jun 2024 | QPISR Q1 2024-25 | 1,933 | 105 | — |
| Aug 2024 | Flash Report (QPISR-style census) | 1,778 | 16 | 12 (frozen) |
| Sep 2024 | Flash Report (QPISR-style census) | 1,725 | 13 | — |
| Mar 2025 | QPISR Q4 2024-25 | 1,764 | 77 | — |

**Parse validation** (computed vs printed in the PDFs): cost-overrun % match
100% within 1pp (n=1,470); time-overrun 99.5% within 1 month (n=1,568);
original cost consistent across months for 95.2% of projects (the rest are
genuine source restatements). See the Data & Sources page in the app.

## Models (all metrics computed on the actual dataset — nothing hard-coded)

- **Leakage policy**: outcome-derived fields (revised/anticipated cost, overrun
  %, actual completion) are never features. Approval-stage features: sector,
  ministry, state, log original cost, planned duration, approval year.
  Monitoring-stage adds only information accumulated by the report month.
- **Time-aware validation**: Apr–Aug 2024 train → Sep 2024 validation
  (threshold + calibration) → Mar 2025 test.
- **Cost-overrun classification** — 5 candidates (Logistic Regression as the
  statistical baseline, Random Forest, XGBoost, LightGBM, CatBoost),
  probability-calibrated (isotonic) on validation. Winner by validation
  PR-AUC: **LightGBM** (test ROC-AUC 0.922). **Strict project-disjoint
  validation (unseen projects): AUC 0.737** — this is the honest screening
  figure and PAIMANA displays it alongside the temporal split.
- **Overrun magnitude regression** — Huber (statistical) vs 4 GBMs:
  **XGBoost**, test MAE 9.1pp (Huber 18.0).
- **Schedule delay** — survival analysis (Cox PH, Weibull AFT) was evaluated
  with 5-fold project-disjoint CV; concordance ≤ 0.55 (only 66 completed
  projects with dates), so the system **automatically falls back** to
  regression on reported delay (XGBoost, MAE 6.6 months). The decision and its
  reason are surfaced in the UI.
- **Forecasting** — per-project ARIMA/ETS honestly infeasible (≤6 monthly
  observations/project; ≥12 required). A global panel model (CatBoost with lag
  features) is evaluated against a naive persistence baseline on the most
  recent month: it does **not** beat the baseline (MAE 7.5 vs 6.1), so the app
  shows the observed trend and the comparison instead of a fake forecast.
- **Anomaly detection** — Isolation Forest (3% contamination) on latest-month
  cost/schedule/expenditure indicators: 60 flagged.
- **SHAP** — global and per-project explanations, framed strictly as
  "influenced the model's prediction" (association, not causation).

## Risk score (transparent, configurable)

`0.35·cost + 0.35·schedule + 0.20·expenditure + 0.10·reporting`, each
component a documented formula (`app/ml/risk.py`, Risk Methodology page,
editable via `PUT /api/config/risk-weights`). Levels: Low <30, Moderate 30–55,
High 55–75, Critical ≥75. Current distribution: 1,364 Low / 687 Moderate /
97 High / 15 Critical. The score is a monitoring aid, **not** an official
government rating.

## API

`/api/dashboard`, `/api/projects` (+`/{code}`, `/{code}/whatif`),
`/api/model-card`, `/api/risk/methodology`, `/api/data-quality`,
`/api/forecast`, `/api/anomalies`, `/api/assistant?q=…`, `/api/meta`.
Interactive docs at `/docs` (Swagger).

## Honesty rules implemented

- Insufficient data → explicit "Insufficient data available for this
  analysis." (never fabricated numbers).
- What-if results labelled **model simulation**, not government forecast.
- Statistical-vs-ML winners computed from held-out metrics on identical
  splits, never hard-coded.
- Variables absent from public data (contractor performance, structured
  delay reasons, land-acquisition milestones, …) are documented as
  **future PAIMANA collection recommendations**, never invented features.
- Assistant answers are computed from the DB via SQL; it never claims
  official decisions. Physical progress is NaN (shown as "not reported") in
  months where MoSPI does not publish it.

## Known limitations

6 report months only (public extracts currently available); physical progress
absent in Apr/May 2024; ~19% of projects lack approval dates in source data
(reflected in reporting-risk); Apr/May closed-event projects carry no codes
and are name-matched ("UNM-" ids where ambiguous).
