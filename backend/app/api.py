"""PAIMANA REST API (FastAPI). All numbers come from the SQLite DB and model
artifacts - no fabricated values."""
import json
import math

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from .config import (ARTIFACTS_DIR, DEFAULT_RISK_WEIGHTS, MANIFEST_JSON,
                     load_risk_weights, risk_level, save_risk_weights)
from .db import query_df
from .ml.features import build_features
from .ml.risk import compute_risk, recommendations

router = APIRouter(prefix='/api')

_MODELS = None
_CARD = None


def sanitize(obj):
    """Make any response JSON-compliant: NaN/inf/NaT -> None."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if obj is pd.NaT:
        return None
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj


def _models():
    global _MODELS
    if _MODELS is None:
        _MODELS = joblib.load(f'{ARTIFACTS_DIR}/models.joblib')
    return _MODELS


def _card():
    global _CARD
    if _CARD is None:
        with open(f'{ARTIFACTS_DIR}/model_card.json') as f:
            _CARD = json.load(f)
    return _CARD


def _scored_panel():
    return query_df("""
        SELECT ps.project_code, ps.report_month, ps.pred_prob_approval,
               ps.pred_prob_monitoring, ps.pred_cost_overrun_pct,
               ps.pred_time_overrun_months, ps.delay_prob,
               ps.expected_delay_months, ps.risk_total, ps.risk_level,
               ps.risk_components, ps.anomaly_flag, ps.top_warning,
               ps.event,
               pa.project_name, pa.sector, pa.ministry, pa.state,
               pa.agency, pa.approval_date, pa.original_cost, pa.revised_cost,
               pa.anticipated_cost, pa.latest_cost, pa.cumulative_expenditure,
               pa.physical_progress_pct, pa.cost_overrun_pct,
               pa.expenditure_over_original_pct, pa.time_overrun_months,
               pa.original_completion_target, pa.revised_completion_target,
               pa.anticipated_completion_target, pa.actual_completion_date,
               pa.planned_duration_months, pa.project_age_months,
               pa.elapsed_fraction, pa.delay_reasons_reported
        FROM project_scores ps
        JOIN panel pa ON pa.project_code = ps.project_code
                     AND pa.report_month = ps.report_month
    """)


# ------------------------------------------------------------------ dashboard
@router.get('/filters')
def filters():
    """Distinct filter values for the projects page."""
    p = query_df('SELECT DISTINCT sector, ministry, state FROM panel')
    return sanitize({
        'sectors': sorted(x for x in p['sector'].dropna().unique() if x),
        'ministries': sorted(x for x in p['ministry'].dropna().unique() if x),
        'states': sorted(x for x in p['state'].dropna().unique() if x),
    })


@router.get('/dashboard')
def dashboard():
    p = query_df('SELECT * FROM panel')
    latest_month = p['report_month'].max()
    cur = p[p['report_month'] == latest_month]
    scored = _scored_panel()
    ongoing = scored[scored['event'] != 'completed']
    over = ongoing[ongoing['cost_overrun_pct'] > 0]
    tor = ongoing[ongoing['time_overrun_months'] > 0]
    trend = p[p['event'] == 'ongoing_report'].groupby('report_month').agg(
        ongoing=('project_code', 'nunique'),
        orig_cr=('original_cost', 'sum'),
        latest_cr=('latest_cost', 'sum'),
        exp_cr=('cumulative_expenditure', 'sum'),
        over_n=('cost_overrun_flag', 'sum')).reset_index()
    sec = ongoing.groupby('sector').agg(
        n=('project_code', 'nunique'),
        orig=('original_cost', 'sum'), latest=('latest_cost', 'sum'),
        over=('cost_overrun_pct', lambda x: (x > 0).sum()),
        avg_cor=('cost_overrun_pct', 'mean')).reset_index()
    return sanitize({
        'latest_month': latest_month,
        'months': sorted(p['report_month'].unique()),
        'kpis': {
            'ongoing_projects': int(ongoing['project_code'].nunique()),
            'original_cost_cr': float(cur['original_cost'].sum()),
            'latest_cost_cr': float(cur['latest_cost'].sum()),
            'expenditure_cr': float(cur['cumulative_expenditure'].sum()),
            'cost_overrun_projects': int(len(over)),
            'cost_overrun_share_pct': float(100 * len(over) / max(1, len(ongoing))),
            'avg_cost_overrun_pct': float(over['cost_overrun_pct'].mean())
            if len(over) else None,
            'schedule_overrun_projects': int(len(tor)),
            'avg_schedule_overrun_months': float(tor['time_overrun_months'].mean())
            if len(tor) else None,
        },
        'risk_distribution': ongoing['risk_level'].value_counts().to_dict(),
        'trend': trend.to_dict('records'),
        'sectors': sec.sort_values('n', ascending=False).to_dict('records'),
        'top_risk': ongoing.nlargest(10, 'risk_total')[
            ['project_code', 'project_name', 'sector', 'risk_total',
             'risk_level', 'cost_overrun_pct', 'time_overrun_months',
             'pred_prob_monitoring']].to_dict('records'),
        'warning_counts': query_df(
            'SELECT warning_type, severity, COUNT(*) n FROM warnings '
            'GROUP BY warning_type, severity').to_dict('records'),
    })


# ------------------------------------------------------------------- projects
@router.get('/projects')
def projects(search: str = '', sector: str = '', ministry: str = '',
             state: str = '', level: str = '', event: str = 'ongoing',
             sort: str = 'risk', page: int = 1, size: int = 25):
    s = _scored_panel()
    if event == 'ongoing':
        s = s[s['event'] != 'completed']
    elif event == 'completed':
        s = s[s['event'] == 'completed']
    if search:
        mask = (s['project_name'].str.contains(search, case=False, na=False) |
                s['project_code'].str.contains(search, case=False, na=False))
        s = s[mask]
    for col, val in [('sector', sector), ('ministry', ministry),
                     ('state', state), ('risk_level', level)]:
        if val:
            s = s[s[col] == val]
    sort_col = {'risk': 'risk_total', 'cor': 'cost_overrun_pct',
                'tor': 'time_overrun_months', 'cost': 'original_cost',
                'name': 'project_name'}.get(sort, 'risk_total')
    asc = sort == 'name'
    s = s.sort_values(sort_col, ascending=asc, na_position='last')
    total = len(s)
    s = s.iloc[(page - 1) * size: page * size]
    return sanitize({'total': int(total), 'page': page, 'size': size,
            'items': s.to_dict('records')})


@router.get('/projects/{code}')
def project_detail(code: str):
    hist = query_df('SELECT * FROM panel WHERE project_code = ? '
                    'ORDER BY report_month', (code,))
    if not len(hist):
        raise HTTPException(404, 'Project not found')
    s = query_df('SELECT * FROM project_scores WHERE project_code = ?',
                 (code,))
    w = query_df('SELECT * FROM warnings WHERE project_code = ?', (code,))
    recs = []
    if len(s):
        recs = recommendations(sorted(
            w.to_dict('records'), key=lambda x: ['critical', 'high', 'medium',
                                                 'low'].index(x['severity'])))
    out = sanitize({
        'code': code,
        'history': hist.to_dict('records'),
        'score': s.to_dict('records')[0] if len(s) else None,
        'warnings': w.to_dict('records'),
        'recommendations': recs,
    })
    # local SHAP explanation
    try:
        from .ml.explain import project_explanation
        latest = hist.iloc[[-1]]
        expl = project_explanation(_models()['cls_approval_base'], latest,
                                   'approval')
        out['shap'] = expl
    except Exception:
        out['shap'] = {'available': False}
    # peers
    try:
        out['peers'] = _peers(code)
    except Exception:
        out['peers'] = {'available': False}
    return sanitize(out)


def _peers(code):
    m = _models()
    codes = m['sim_codes']
    if code not in codes:
        return {'available': False}
    i = codes.index(code)
    mat = m['sim_matrix']
    dist, idx = m['nn_model'].kneighbors(mat[i].reshape(1, -1))
    peer_codes = [codes[j] for j in idx[0] if codes[j] != code][:10]
    peers = query_df(
        'SELECT DISTINCT project_code, project_name, sector, state, original_cost, '
        'cost_overrun_pct, time_overrun_months, event FROM panel '
        f"WHERE project_code IN ({','.join('?'*len(peer_codes))}) "
        'AND report_month = (SELECT MAX(report_month) FROM panel p2 '
        'WHERE p2.project_code = panel.project_code)', peer_codes)
    return {'available': True,
            'note': 'Similar projects by sector, ministry, state, original '
                    'cost and planned duration.',
            'peers': peers.to_dict('records')}


# -------------------------------------------------------------------- what-if
@router.post('/projects/{code}/whatif')
def whatif(code: str, body: dict):
    """Model simulation: override project attributes and see the predicted
    change. This is a model simulation, not a government forecast."""
    hist = query_df('SELECT * FROM panel WHERE project_code = ? '
                    'ORDER BY report_month', (code,))
    if not len(hist):
        raise HTTPException(404, 'Project not found')
    row = hist.iloc[-1].copy()
    overrides = {k: v for k, v in (body or {}).items()
                 if k in ('sector', 'ministry', 'state', 'original_cost',
                          'planned_duration_months', 'approval_date',
                          'cumulative_expenditure', 'physical_progress_pct',
                          'project_age_months')}
    sim = row.copy()
    for k, v in overrides.items():
        sim[k] = v
    # recompute derived fields affected by overrides
    try:
        oc = float(sim['original_cost'])
        lc = float(sim['latest_cost']) if pd.notna(sim['latest_cost']) else oc
        sim['latest_cost'] = lc
        sim['cost_overrun_pct'] = (lc - oc) / oc * 100 if oc > 0 else np.nan
        ex = pd.to_numeric(sim['cumulative_expenditure'], errors='coerce')
        sim['expenditure_over_original_pct'] = (ex - oc) / oc * 100 if oc > 0 \
            else np.nan
        pd_ = pd.to_numeric(sim['planned_duration_months'], errors='coerce')
        age = pd.to_numeric(sim['project_age_months'], errors='coerce')
        if pd.notna(pd_) and pd_ > 0 and pd.notna(age):
            sim['elapsed_fraction'] = age / pd_
    except Exception:
        pass

    m = _models()
    base_row = row.to_frame().T
    sim_row = sim.to_frame().T

    def predict(models_key, stage):
        if not m.get(models_key):
            return None
        X = build_features(sim_row, stage)
        return float(m[models_key].predict_proba(X)[0, 1])

    p_appr = predict('cls_approval', 'approval')
    p_mon = predict('cls_monitoring', 'monitoring')
    base_appr = float(m['cls_approval'].predict_proba(
        build_features(base_row, 'approval'))[0, 1])
    base_mon = float(m['cls_monitoring'].predict_proba(
        build_features(base_row, 'monitoring'))[0, 1]) \
        if m.get('cls_monitoring') else base_appr

    cor_pred = None
    if m.get('reg_cost'):
        cor_pred = float(m['reg_cost'].predict(
            build_features(sim_row, 'approval'))[0])
    risk = compute_risk(sim, p_mon if p_mon is not None else p_appr, None,
                        False)
    base_risk = compute_risk(row, base_mon, None, False)
    return sanitize({
        'label': 'Model simulation (not a government forecast)',
        'overrides': overrides,
        'before': {'pred_prob_approval': round(base_appr, 4),
                   'pred_prob_monitoring': round(base_mon, 4),
                   'risk': base_risk},
        'after': {'pred_prob_approval': None if p_appr is None else round(p_appr, 4),
                  'pred_prob_monitoring': None if p_mon is None else round(p_mon, 4),
                  'pred_cost_overrun_pct': None if cor_pred is None
                  else round(cor_pred, 1),
                  'risk': risk},
        'note': 'Probabilities from the calibrated classification model; risk '
                'recomputed with the transparent component formulas.',
    })


# ---------------------------------------------------------------- model pages
@router.get('/model-card')
def model_card():
    return _card()


@router.get('/model-performance')
def model_performance():
    return _card()


@router.get('/risk/methodology')
def risk_methodology():
    w = load_risk_weights()
    return {
        'weights': w, 'defaults': DEFAULT_RISK_WEIGHTS,
        'levels': {'Low': '<30', 'Moderate': '30-55', 'High': '55-75',
                   'Critical': '>=75'},
        'components': {
            'cost_risk': '0.5 x 100 x calibrated overrun probability + '
                         '0.5 x min(cost_overrun_pct, 100). '
                         'A 100% overrun saturates the realized part.',
            'schedule_risk': 'weighted mean of: delay probability (0.4, where '
                             'a survival model is available), months past the '
                             'original target / 30 (0.3), anticipated delay '
                             'months / 24 (0.3).',
            'expenditure_risk': 'weighted mean of: expenditure above original '
                                'cost % (0.4), spending ahead of schedule '
                                'fraction (0.3), financial-vs-physical progress '
                                'mismatch (0.3).',
            'reporting_risk': 'missing approval date (+35), missing original '
                              'cost (+25), no physical progress reported '
                              '(+10), statistical anomaly flag (+40).',
        },
        'disclaimer': 'The risk score is a transparent monitoring aid computed '
                      'from reported data and model outputs - it is not an '
                      'official government rating.',
    }


@router.put('/config/risk-weights')
def update_risk_weights(body: dict):
    try:
        w = {k: float(body[k]) for k in
             ('cost_risk', 'schedule_risk', 'expenditure_risk', 'reporting_risk')}
    except (KeyError, TypeError, ValueError):
        raise HTTPException(400, 'Body must contain numeric cost_risk, '
                                 'schedule_risk, expenditure_risk, reporting_risk')
    if not 0.95 <= sum(w.values()) <= 1.05:
        raise HTTPException(400, 'Weights must sum to ~1.0')
    save_risk_weights(w)
    # recompute totals/levels from stored components
    scores = query_df('SELECT project_code, risk_components FROM project_scores')
    from .db import get_conn
    conn = get_conn()
    for _, r in scores.iterrows():
        try:
            c = json.loads(r['risk_components'])
            total = (w['cost_risk'] * c['cost_risk'] +
                     w['schedule_risk'] * c['schedule_risk'] +
                     w['expenditure_risk'] * c['expenditure_risk'] +
                     w['reporting_risk'] * c['reporting_risk'])
            conn.execute('UPDATE project_scores SET risk_total=?, risk_level=? '
                         'WHERE project_code=?',
                         (round(total, 1), risk_level(total), r['project_code']))
        except Exception:
            continue
    conn.commit()
    conn.close()
    return {'weights': w, 'updated': True}


# ---------------------------------------------------------------- data quality
@router.get('/data-quality')
def data_quality():
    p = query_df('SELECT * FROM panel')
    with open(MANIFEST_JSON) as f:
        manifest = json.load(f)
    per_month = p.groupby(['report_month', 'event']).size().unstack(fill_value=0)
    nulls = {}
    for c in ['approval_date', 'original_cost', 'planned_duration_months',
              'physical_progress_pct', 'cumulative_expenditure', 'sector',
              'state', 'time_overrun_months']:
        if c in p.columns:
            nulls[c] = float(100 * p[c].isna().mean())
    # cross-month consistency of original cost
    cons = p.dropna(subset=['original_cost']).groupby('project_code')[
        'original_cost'].nunique()
    return {
        'manifest': manifest,
        'per_month': per_month.reset_index().to_dict('records'),
        'null_share_pct': nulls,
        'original_cost_consistency_pct': float(100 * (cons == 1).mean()),
        'known_limitations': [
            'Physical progress is reported only in QPISR-style months (incl. '
            'Aug/Sep 2024 flash reports); it is honestly absent (NaN) for '
            'Apr/May 2024 flash months.',
            'Apr/May 2024 closed-event projects arrive without project codes '
            'and are matched by name where unique ("UNM-" otherwise).',
            'Delay reasons exist only for closed-event tables; ongoing census '
            'rows have no structured reason field - a documented future '
            'PAIMANA collection need.',
            'The panel is a snapshot of 6 report months (Apr-Sep 2024, '
            'Mar 2025) - trends are short and forecasting is limited '
            'accordingly.',
        ],
    }


# ------------------------------------------------------------------- forecast
@router.get('/forecast')
def forecast():
    card = _card()
    p = query_df('SELECT * FROM panel WHERE event = "ongoing_report"')
    series = p.groupby('report_month').agg(
        avg_cor=('cost_overrun_pct', 'mean'),
        share_over=('cost_overrun_flag', 'mean'),
        exp_cr=('cumulative_expenditure', 'sum'),
        latest_cr=('latest_cost', 'sum')).reset_index()
    return sanitize({
        'feasibility': card['tasks']['forecast_cost_overrun_pct'].get(
            'feasibility'),
        'evaluation': {k: v for k, v in
                       card['tasks']['forecast_cost_overrun_pct'].items()
                       if k in ('test_month', 'n_test', 'model_mae', 'naive_mae',
                                'model_beats_naive', 'available', 'reason',
                                'note')},
        'monthly': series.to_dict('records'),
        'disclaimer': 'Trend views show observed MoSPI data. Where the panel '
                      'model does not beat the naive baseline on hold-out '
                      'evaluation, no model forecast is shown - only the '
                      'observed trend and the honest model-vs-baseline '
                      'comparison.',
    })


@router.get('/anomalies')
def anomalies():
    s = _scored_panel()
    a = s[s['anomaly_flag'] == 1]
    return sanitize({'count': int(len(a)),
                     'note': 'Isolation Forest on latest-month cost/schedule/'
                             'expenditure indicators (3% contamination). Verify '
                             'flagged figures with the implementing agency.',
                     'items': a[['project_code', 'project_name', 'sector',
                                 'original_cost', 'latest_cost',
                                 'cost_overrun_pct', 'time_overrun_months',
                                 'risk_total', 'risk_level']].to_dict('records')})


# ------------------------------------------------------------------ assistant
@router.get('/assistant')
def assistant(q: str = Query('', description='Natural-language question')):
    from .services.assistant import answer
    a = answer(q, card=_card())
    a['disclaimer'] = ('Answers are computed from the MoSPI panel in the PAIMANA '
                       'database. The assistant does not state or imply official '
                       'government decisions.')
    return sanitize(a)


@router.get('/meta')
def meta():
    card = _card()
    return {
        'name': 'PAIMANA - Pro-active Analytics for Infrastructure '
                'Monitoring and Assessment (National Analytics)',
        'dataset': card['dataset'],
        'generated_at': card['generated_at'],
        'tasks_available': {k: v.get('available', True)
                            for k, v in card['tasks'].items()},
    }
