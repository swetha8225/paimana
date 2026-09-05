# PAIMANA — Session State (auto-maintained working notes)

## Current state: END-TO-END APP COMPLETE AND RUNNING (as of 2026-09-03)

**Live**: `uvicorn app.main:app --port 8000` serving API + built frontend.
Verify: `curl localhost:8000/api/health` → `{"status":"ok","db":true,"models":true}`.
Relaunch anytime: `cd /home/user/paimana && ./run.sh` (or start uvicorn directly).

### Completed this session
1. **Aug/Sep band bug FIXED** — `calibrate_bands()` in parse_qpisr.py anchors x-bands
   to each PDF's own header words (Approval/Commissioning/Cost/Expenditure/Progress);
   original_cost cross-month consistency 13% → **95.2%**. QPISR months unaffected.
   Sep census-table selection confirmed correct (1,725 rows = full census, not NE subset).
2. **Panel final**: 11,121 rows, 2,163 projects, 6 months (Apr/May/Jun/Aug/Sep 2024,
   Mar 2025). Validations: COR 100% ≤1pp (n=1470), TOR 99.5% ≤1m (n=1568).
3. **Backend complete** (`/home/user/paimana/backend/app/`):
   - ml/features.py — leakage-safe, approval vs monitoring stages, STATE_ZONES, time_split_months
   - ml/models.py — classification_task (LR/RF/XGB/LGBM/CatBoost + isotonic calibration
     + recall-priority threshold), regression_task (Huber/RF/XGB/LGBM/CatBoost),
     survival_task (Cox+Weibull, 5-fold project-disjoint CV, auto-reject at c≤0.55),
     ColumnTransformerWithNames + AsCatFrame (custom transformers ARE sklearn-cloneable;
     CatBoost needs int cat columns → AsCatFrame DataFrame step)
   - ml/risk.py — transparent 4-component score (0.35/0.35/0.20/0.10), warnings
     (10 rule types), recommendations mapping; levels Low<30/Mod<55/High<75/Critical
   - ml/explain.py — SHAP walks all pipeline steps; linear coef fallback
   - ml/forecast.py — per-project ARIMA infeasible note (≤6 obs, need 12); global
     panel model vs naive baseline honest comparison
   - ml/pipeline.py — orchestrator + project_disjoint_eval + anomaly + similarity +
     scoring + model_card.json + models.joblib (46s runtime)
   - services/assistant.py — rule-based NL over DB (top-risk/aggregate/project/
     drivers/data-quality intents)
   - api.py (14 endpoints, sanitize() NaN→None), main.py (static SPA serving)
4. **Frontend complete** (`/home/user/paimana/frontend/`, built to backend/app/static):
   React+TS+Recharts, 8 pages: Dashboard, Projects (+detail with SHAP/warnings/
   peers/history/what-if), Risk Methodology (editable weights), Model Performance
   (stat-vs-ML, curves, project-disjoint panel, auto-decisions), Trends, Data &
   Sources, Assistant. Verified via headless Chromium: 0 JS errors, all pages render.
5. README.md + run.sh + screenshots/ (final_*.png).

### Final model results (model_card.json — computed, not hard-coded)
- Classification approval: **LightGBM** test ROC-AUC 0.922 / PR-AUC 0.887 / recall
  0.92; XGBoost 0.926, CatBoost 0.915, RF 0.910, LR 0.793 (stat baseline).
  **Project-disjoint AUC 0.737** (honest new-project screening figure).
- Regression: cost XGBoost MAE 9.08pp (Huber 17.98); time XGBoost MAE 6.61mo (Huber 11.21).
- Survival: REJECTED (CV concordance ≤0.55, 66 events) → regression fallback, shown in UI.
- Forecast: panel model MAE 7.52 vs naive 6.12 → does NOT beat baseline → observed
  trend only, comparison displayed.
- Anomaly: 60/1,969 flagged. Risk dist: 1,364 Low / 687 Moderate / 97 High / 15 Critical.
- SHAP global: approval_year > log_original_cost > state > planned_duration > sector.

### Gotchas for future edits
- **Workspace snapshot cap ~128MB**: keep total under it or files get silently
  dropped (models.joblib was lost once). Fixes applied: joblib.dump(compress=3)
  (33MB→12.5MB); unused 2023-24 QPISR PDFs + 0-byte stubs deleted (re-download
  from mospi.gov.in/sites/default/files/publication_reports/QPSR_{2nd,3rd,4th}_QTR_2023-24.pdf,
  browser UA required). Current total ~72MB. If artifacts vanish again:
  `cd backend && python3 -m app.ml.pipeline` regenerates everything in ~50s.
- Installed pip packages / playwright chromium do NOT persist across sessions:
  `pip install -r backend/requirements.txt playwright && python3 -m playwright
  install --with-deps chromium`, then restart uvicorn.
- Custom sklearn transformers MUST inherit BaseEstimator/TransformerMixin (clone).
- CatBoost cat_features must be tuple (not list) AND int-typed columns (AsCatFrame).
- lifelines: bool dummies → int; time-split by last-obs month puts all events in
  train (completions leave panel early) → use project-disjoint CV.
- FastAPI responses: wrap in sanitize() (NaN not JSON compliant).
- uvicorn does NOT reload; restart after backend edits.
- API server caches models.joblib + model_card.json at module level.

### Possible next steps (not blocking)
- Optional Ollama hook for assistant phrasing (env var) — rule engine is complete.
- More report months (older QPISR quarters) to extend the panel.
- Demo script for SIH judging (walkthrough order: Dashboard → project detail →
  what-if → Model Performance honesty panels → Assistant).
