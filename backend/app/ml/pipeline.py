"""PAIMANA ML pipeline orchestrator.

Run:  python -m app.ml.pipeline
Loads the canonical panel, trains all models with time-aware validation,
scores every project, generates warnings, and writes the SQLite DB plus
artifacts (model card, fitted models).
"""
import json
import os
import sys
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from ..config import (ARTIFACTS_DIR, DEFAULT_RISK_WEIGHTS, PANEL_CSV,
                      load_risk_weights)
from ..db import init_db
from .features import (FEATURE_DOCS, LEAKAGE_NOTES, build_features, state_zone,
                       time_split_months)
from .models import classification_task, regression_task, survival_task
from .risk import compute_risk, generate_warnings, recommendations
from .forecast import forecast_task

FUTURE_VARIABLES = [
    'Contractor / implementing-agency identity and past performance',
    'Detailed structured delay reasons (currently only headline categories '
    'from flash-report tables)',
    'Sanctioned vs released funds and sanction-to-award lag',
    'Land acquisition and statutory clearance milestone dates',
    'Physical milestone completion (currently only cumulative % progress)',
    'Monthly (rather than quarterly/flash) reporting frequency',
]


def log(msg):
    print(f'[pipeline] {msg}', flush=True)


def load_panel():
    df = pd.read_csv(PANEL_CSV, dtype={'project_code': str, 'report_month': str,
                                       'approval_date': str})
    log(f'panel loaded: {len(df)} rows, {df.project_code.nunique()} projects, '
        f'months {sorted(df.report_month.unique())}')
    return df


def latest_rows(panel):
    return panel.sort_values('report_month').groupby('project_code').tail(1).copy()


def project_disjoint_eval(panel, best_model_name, stage='approval'):
    """Strict robustness check: hold out ENTIRE projects (not months) to measure
    generalisation to projects the model has never seen - the honest figure for
    'screen a newly approved project' claims."""
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score, roc_auc_score)

    from .features import FEATURES_MONITOR
    from .models import make_classifiers
    d = panel[(panel['event'] == 'ongoing_report') &
              panel['cost_overrun_flag'].notna() &
              (panel['original_cost'] > 0) &
              panel['planned_duration_months'].notna() &
              panel['approval_date'].notna()].copy()
    if len(d) < 500:
        return {'available': False}
    codes = d['project_code'].unique()
    rng = np.random.RandomState(42)
    rng.shuffle(codes)
    n = len(codes)
    test_codes = set(codes[:int(n * 0.25)])
    val_codes = set(codes[int(n * 0.25):int(n * 0.40)])
    train_codes = set(codes) - test_codes - val_codes
    is_tr = d['project_code'].isin(train_codes).values
    is_va = d['project_code'].isin(val_codes).values
    is_te = d['project_code'].isin(test_codes).values
    X = build_features(d, stage)
    y = d['cost_overrun_flag'].astype(int).values
    try:
        pipe = make_classifiers()[best_model_name]
        pipe.fit(X[is_tr], y[is_tr])
        pva = pipe.predict_proba(X[is_va])[:, 1]
        pte = pipe.predict_proba(X[is_te])[:, 1]
        from .models import _threshold_for_recall
        thr = _threshold_for_recall(y[is_va], pva)
        yhat = (pte >= thr).astype(int)
        return {
            'available': True,
            'description': 'Entire projects held out (train 60% / val 15% / '
                           'test 25% of project codes) - measures generalisation '
                           'to projects never seen during training. The primary '
                           'time-split metrics above share projects across '
                           'months (panel data), which inflates scores by '
                           'partially recognising the same project.',
            'model': best_model_name,
            'n_test_rows': int(is_te.sum()),
            'n_test_projects': len(test_codes),
            'roc_auc': float(roc_auc_score(y[is_te], pte)),
            'accuracy': float(accuracy_score(y[is_te], yhat)),
            'precision': float(precision_score(y[is_te], yhat, zero_division=0)),
            'recall': float(recall_score(y[is_te], yhat, zero_division=0)),
            'f1': float(f1_score(y[is_te], yhat, zero_division=0)),
            'threshold': thr,
        }
    except Exception as e:
        return {'available': False, 'error': str(e)[:200]}


# --------------------------------------------------------------- anomaly model
def anomaly_task(panel):
    from sklearn.ensemble import IsolationForest
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    latest = latest_rows(panel)
    latest = latest[latest['event'] != 'completed']
    feats = pd.DataFrame(index=latest.index)
    feats['log_original_cost'] = np.log(latest['original_cost'].astype(float)
                                        .clip(lower=.01))
    feats['log_latest_cost'] = np.log(latest['latest_cost'].astype(float)
                                      .clip(lower=.01))
    for c in ['cost_overrun_pct', 'expenditure_over_original_pct',
              'elapsed_fraction', 'time_overrun_months', 'project_age_months']:
        feats[c] = pd.to_numeric(latest[c], errors='coerce')
    feats = feats.replace([np.inf, -np.inf], np.nan)
    if len(feats) < 100:
        return {'available': False, 'reason': 'Insufficient data available for '
                'this analysis.', 'n': int(len(feats))}, None, pd.Series(dtype=bool)
    pipe = Pipeline([('imp', SimpleImputer(strategy='median')),
                     ('sc', StandardScaler()),
                     ('if', IsolationForest(contamination=0.03, random_state=42))])
    labels = pipe.fit_predict(feats)
    anom = pd.Series(labels == -1, index=latest.index)
    summary = {
        'available': True, 'n_scored': int(len(feats)),
        'n_flagged': int(anom.sum()),
        'flag_rate_pct': float(100 * anom.mean()),
        'features': list(feats.columns),
        'method': 'Isolation Forest (3% contamination) on scaled cost/schedule/'
                  'expenditure indicators of the latest report month per project.',
    }
    return summary, pipe, anom


# ------------------------------------------------------------- similarity index
def similarity_task(panel):
    from sklearn.impute import SimpleImputer
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    latest = latest_rows(panel)
    f = pd.DataFrame(index=latest.index)
    for c in ['sector', 'ministry', 'state']:
        f[c] = latest[c].astype('object').where(latest[c].notna(), 'Unknown')
    f = pd.get_dummies(f, columns=['sector', 'ministry', 'state'])
    f['log_original_cost'] = np.log(latest['original_cost'].astype(float)
                                    .clip(lower=.01))
    f['planned_duration_months'] = pd.to_numeric(
        latest['planned_duration_months'], errors='coerce')
    f = f.replace([np.inf, -np.inf], np.nan)
    med = f.median(numeric_only=True).fillna(0)
    f = f.fillna(med)
    mat = StandardScaler().fit_transform(f)
    nn = NearestNeighbors(n_neighbors=11).fit(mat)
    return {
        'available': True, 'n_projects': int(len(latest)),
        'features': 'sector + ministry + state (one-hot), log original cost, '
                    'planned duration (scaled)',
        'method': 'k-NN (k=10 peers) in standardized feature space',
    }, nn, mat, latest['project_code'].tolist()


# ------------------------------------------------------------------ survival
def survival_predictions(surv, latest):
    """Delay probability and expected delay for ongoing projects from the
    fitted survival model."""
    out = pd.DataFrame(index=latest.index,
                       columns=['delay_prob', 'expected_delay_months'])
    if not surv or not surv.get('available'):
        return out
    m = surv['serving_model']
    cov_cols = surv.get('train_columns') or []
    if not cov_cols:
        return out
    d = latest[(latest['event'] == 'ongoing_report') &
               latest['approval_date'].notna() &
               (pd.to_numeric(latest['original_cost'], errors='coerce') > 0) &
               latest['planned_duration_months'].notna()].copy()
    if not len(d):
        return out

    def dur(row):
        try:
            a = str(row['approval_date'])
            ay, am = int(a[:4]), int(a[5:7])
            ey, em = int(str(row['report_month'])[:4]), int(str(row['report_month'])[5:7])
            return max(0.0, (ey * 12 + em) - (ay * 12 + am))
        except Exception:
            return 0.0

    X = pd.DataFrame({
        'log_original_cost': np.log(d['original_cost'].astype(float).clip(lower=.01)),
        'planned_duration_months': d['planned_duration_months'].astype(float),
        'zone': d['state'].map(state_zone),
    })
    X = pd.get_dummies(X, columns=['zone'], drop_first=True)
    X = X.reindex(columns=cov_cols, fill_value=0).astype(float)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    if not len(X):
        return out

    sf = m.predict_survival_function(X, conditional_after=d.apply(dur, axis=1))
    for i, idx in enumerate(d.index):
        try:
            planned = float(d.loc[idx, 'planned_duration_months'])
            s = sf.iloc[:, i].dropna()
            if len(s) == 0:
                continue
            # P(duration > planned) = probability commissioning slips beyond
            # the original target, given survival so far
            times = s.index.values.astype(float)
            surv_at_planned = float(np.interp(planned, times, s.values))
            out.loc[idx, 'delay_prob'] = 1.0 - surv_at_planned
            # expected additional months vs plan: median remaining time
            med = float(np.interp(0.5, s.values[::-1], times[::-1]))
            out.loc[idx, 'expected_delay_months'] = med - planned
        except Exception:
            continue
    return out


# ------------------------------------------------------------------ main run
def run():
    t0 = datetime.now(timezone.utc)
    panel = load_panel()
    init_db(panel)

    months = sorted(panel['report_month'].unique())
    latest = latest_rows(panel)

    log('training classification (approval stage)...')
    cls_appr = classification_task(panel, stage='approval')
    log(f"  best={cls_appr.get('best_model')} "
        f"test_auc={cls_appr.get('best_metrics', {}).get('roc_auc')}")
    if cls_appr.get('available'):
        pd_eval = project_disjoint_eval(panel, cls_appr['best_model'], 'approval')
        cls_appr['project_disjoint'] = pd_eval
        log(f"  project-disjoint AUC={pd_eval.get('roc_auc')}")

    log('training classification (monitoring stage)...')
    cls_mon = classification_task(panel, stage='monitoring')
    log(f"  best={cls_mon.get('best_model')} "
        f"test_auc={cls_mon.get('best_metrics', {}).get('roc_auc')}")

    log('training cost-overrun regression...')
    reg_cost = regression_task(panel, 'cost_overrun_pct')
    log(f"  best={reg_cost.get('best_model')} test_mae="
        f"{reg_cost.get('best_metrics', {}).get('mae')}")

    log('training time-overrun regression...')
    reg_time = regression_task(panel, 'time_overrun_months')
    log(f"  best={reg_time.get('best_model')} test_mae="
        f"{reg_time.get('best_metrics', {}).get('mae')}")

    log('fitting survival models...')
    surv = survival_task(panel)
    if surv.get('available'):
        log(f"  best={surv.get('best_model')} c-index="
            f"{surv.get('best_metrics', {}).get('concordance_index')}")
    else:
        log(f"  unavailable: {surv.get('reason')}")

    log('evaluating forecast models...')
    fc = forecast_task(panel, 'cost_overrun_pct')
    log(f"  available={fc.get('available')}")

    log('fitting anomaly detector...')
    anom_summary, anom_model, anom_flags = anomaly_task(panel)
    log(f"  flagged={anom_summary.get('n_flagged')}")

    log('building similarity index...')
    sim_summary, nn_model, sim_matrix, sim_codes = similarity_task(panel)

    # ---------------------------------------------------------- scoring
    log('scoring latest month per project...')
    scores = []
    warnings_rows = []
    surv_pred = survival_predictions(surv, latest)
    anomaly_by_code = {}
    if anom_flags is not None and len(anom_flags):
        idx_map = latest['project_code'].to_dict()
        anomaly_by_code = {idx_map[i]: bool(v) for i, v in anom_flags.items()}

    X_appr = build_features(latest, 'approval')
    X_mon = build_features(latest, 'monitoring')
    p_appr = (cls_appr['serving_model'].predict_proba(X_appr)[:, 1]
              if cls_appr.get('available') else [np.nan] * len(latest))
    p_mon = (cls_mon['serving_model'].predict_proba(X_mon)[:, 1]
             if cls_mon.get('available') else [np.nan] * len(latest))
    pred_cor = (reg_cost['serving_model'].predict(X_appr)
                if reg_cost.get('available') else [np.nan] * len(latest))
    pred_tor = (reg_time['serving_model'].predict(X_appr)
                if reg_time.get('available') else [np.nan] * len(latest))

    latest_codes = set(latest['project_code'])
    latest_month = months[-1]
    for i, (idx, row) in enumerate(latest.iterrows()):
        code = row['project_code']
        pa, pm = float(p_appr[i]), float(p_mon[i])
        dp = surv_pred.loc[idx, 'delay_prob'] if idx in surv_pred.index else np.nan
        ed = (surv_pred.loc[idx, 'expected_delay_months']
              if idx in surv_pred.index else np.nan)
        anomaly = anomaly_by_code.get(code, False)
        risk = compute_risk(row, pm if not np.isnan(pm) else pa, dp, anomaly)
        not_latest = (row['report_month'] != latest_month and
                      row['event'] == 'ongoing_report')
        row_d = row.to_dict()
        row_d['not_in_latest'] = not_latest
        ws = generate_warnings(row_d, pm if not np.isnan(pm) else pa, risk,
                               anomaly) if row['event'] != 'completed' else []
        scores.append({
            'project_code': code, 'report_month': row['report_month'],
            'event': row['event'],
            'pred_prob_approval': None if np.isnan(pa) else round(pa, 4),
            'pred_prob_monitoring': None if np.isnan(pm) else round(pm, 4),
            'pred_cost_overrun_pct': None if np.isnan(pred_cor[i])
            else round(float(pred_cor[i]), 1),
            'pred_time_overrun_months': None if np.isnan(pred_tor[i])
            else round(float(pred_tor[i]), 1),
            'delay_prob': None if pd.isna(dp) else round(float(dp), 4),
            'expected_delay_months': None if pd.isna(ed) else round(float(ed), 1),
            'risk_total': round(risk['total'], 1), 'risk_level': risk['level'],
            'risk_components': json.dumps(risk['components']),
            'anomaly_flag': 1 if anomaly else 0,
            'top_warning': ws[0]['what'] if ws else None,
            'recommendations': json.dumps(recommendations(ws)),
        })
        for w in ws:
            warnings_rows.append((code, row['report_month'], w['warning_type'],
                                  w['severity'], w['what'], w['reason'],
                                  w['action']))

    scores_df = pd.DataFrame(scores)
    log(f'scored {len(scores_df)} projects '
        f'({(scores_df.risk_level == "Critical").sum()} critical)')

    # ------------------------------------------------------------ write DB
    from ..db import get_conn
    conn = get_conn()
    conn.execute('DROP TABLE IF EXISTS project_scores')
    scores_df.to_sql('project_scores', conn, if_exists='replace', index=False)
    conn.executemany('INSERT INTO warnings VALUES (?,?,?,?,?,?,?)', warnings_rows)
    conn.commit()
    conn.close()
    log(f'wrote {len(warnings_rows)} warnings to DB')

    # ------------------------------------------------------------ SHAP
    log('computing SHAP global importance...')
    shap_global = None
    try:
        from .explain import global_importance
        stage = cls_appr.get('stage', 'approval')
        shap_global = global_importance(cls_appr['base_model'], latest, stage)
    except Exception as e:
        log(f'  SHAP failed: {e}')

    # ------------------------------------------------- statistical vs ML card
    def stat_vs_ml_cls(task):
        if not task.get('available'):
            return None
        res = task['model_results']
        stat = res.get('Logistic Regression', {})
        ml_best = task['best_metrics']
        if 'error' in stat:
            return None
        winner = ('Machine Learning' if (ml_best.get('roc_auc') or 0) >
                  (stat.get('roc_auc') or 0) else 'Statistical baseline')
        return {
            'statistical': {'model': 'Logistic Regression', **stat},
            'ml': {'model': task['best_model'], **ml_best},
            'winner_on_test_roc_auc': winner,
            'metric_note': 'Models compared on the same time-based test split '
                           '(most recent report month); comparison computed from '
                           'the actual dataset.',
        }

    def stat_vs_ml_reg(task):
        if not task.get('available'):
            return None
        res = task['model_results']
        stat = res.get('Huber Regression (statistical baseline)', {})
        ml_best = task['best_metrics']
        if 'error' in stat:
            return None
        winner = ('Machine Learning' if (ml_best.get('mae') or 9e9) <
                  (stat.get('mae') or 9e9) else 'Statistical baseline')
        return {
            'statistical': {'model': 'Huber Regression', **stat},
            'ml': {'model': task['best_model'], **ml_best},
            'winner_on_test_mae': winner,
        }

    model_card = {
        'generated_at': t0.isoformat(),
        'dataset': {
            'source': 'MoSPI Quarterly Progress Information System Reports '
                      '(QPISR) and Flash Reports (public data), parsed from '
                      'official PDFs',
            'report_months': months,
            'panel_rows': int(len(panel)),
            'unique_projects': int(panel['project_code'].nunique()),
            'per_month_counts': panel.groupby(['report_month', 'event'])
            .size().unstack(fill_value=0).to_dict('index'),
        },
        'split_policy': {
            'description': 'Time-aware splits: earlier report months train, '
                           'middle months validate (threshold & calibration '
                           'selection), the most recent month is held out for '
                           'testing. Project-month rows use only information '
                           'available at that month (see leakage notes).',
            'leakage_notes': LEAKAGE_NOTES,
            'feature_docs': FEATURE_DOCS,
        },
        'tasks': {
            'classification_approval_stage': _slim(cls_appr),
            'classification_monitoring_stage': _slim(cls_mon),
            'regression_cost_overrun_pct': _slim(reg_cost),
            'regression_time_overrun_months': _slim(reg_time),
            'survival_time_overrun': _slim(surv),
            'forecast_cost_overrun_pct': {k: v for k, v in fc.items()
                                          if k != 'model'},
            'anomaly_detection': anom_summary,
            'similarity_benchmarking': sim_summary,
        },
        'stat_vs_ml': {
            'classification_approval': stat_vs_ml_cls(cls_appr),
            'classification_monitoring': stat_vs_ml_cls(cls_mon),
            'regression_cost_overrun_pct': stat_vs_ml_reg(reg_cost),
            'regression_time_overrun_months': stat_vs_ml_reg(reg_time),
        },
        'risk_score': {
            'weights': load_risk_weights(),
            'defaults': DEFAULT_RISK_WEIGHTS,
            'components': 'cost_risk, schedule_risk, expenditure_risk, '
                          'reporting_risk - each a documented formula (see '
                          '/api/risk/methodology or the Risk Methodology page)',
        },
        'shap_global': shap_global,
        'future_variables': FUTURE_VARIABLES,
        'future_variables_note': 'These variables are NOT present in the '
                                 'current public MoSPI extracts. They are '
                                 'recommended future PAIMANA collections.',
    }
    with open(os.path.join(ARTIFACTS_DIR, 'model_card.json'), 'w') as f:
        json.dump(model_card, f, indent=2, default=str)

    # ------------------------------------------------------------ save models
    joblib.dump({
        'cls_approval': cls_appr.get('serving_model'),
        'cls_approval_base': cls_appr.get('base_model'),
        'cls_monitoring': cls_mon.get('serving_model'),
        'cls_monitoring_base': cls_mon.get('base_model'),
        'reg_cost': reg_cost.get('serving_model'),
        'reg_time': reg_time.get('serving_model'),
        'anomaly': anom_model,
        'nn_model': nn_model, 'sim_matrix': sim_matrix, 'sim_codes': sim_codes,
        'survival': surv.get('serving_model') if surv.get('available') else None,
        'survival_cols': surv.get('train_columns'),
        'forecast': fc.get('model'),
        'stages': {'approval': cls_appr.get('stage'),
                   'monitoring': cls_mon.get('stage')},
    }, os.path.join(ARTIFACTS_DIR, 'models.joblib'), compress=3)
    log(f'pipeline complete in '
        f'{(datetime.now(timezone.utc) - t0).total_seconds():.0f}s -> '
        f'{ARTIFACTS_DIR}')


def _slim(task):
    """Drop non-serialisable model objects from a task bundle."""
    if not task or not task.get('available'):
        return {k: v for k, v in (task or {}).items()
                if k not in ('serving_model', 'base_model')}
    out = {k: v for k, v in task.items()
           if k not in ('serving_model', 'base_model')}
    return out


if __name__ == '__main__':
    sys.exit(run())
