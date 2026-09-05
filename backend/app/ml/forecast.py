"""Expenditure / cost-overrun trend modelling and forecasting.

Per-project statistical forecasting (ARIMA/ETS) needs >= 12 monthly
observations; the current public panel has 6 report months per project, so the
system honestly reports that and instead uses a GLOBAL panel model (gradient
boosting with lag features, evaluated against a naive persistence baseline) -
the approach is selected empirically, never assumed.
"""
import numpy as np
import pandas as pd

from .features import month_index

MIN_POINTS_PER_PROJECT = 12   # for ARIMA/ETS


def per_project_feasibility(panel: pd.DataFrame):
    counts = panel.groupby('project_code')['report_month'].nunique()
    return {
        'projects': int(len(counts)),
        'max_observations_per_project': int(counts.max()) if len(counts) else 0,
        'min_required': MIN_POINTS_PER_PROJECT,
        'feasible': bool(counts.max() >= MIN_POINTS_PER_PROJECT) if len(counts)
        else False,
        'note': f'Per-project ARIMA/ETS forecasting requires at least '
                f'{MIN_POINTS_PER_PROJECT} monthly observations per project; the '
                f'panel provides at most {int(counts.max()) if len(counts) else 0}. '
                f'PAIMANA therefore uses a global panel model with lag features '
                f'(evaluated against a naive baseline) until longer series are '
                f'available.',
    }


def _build_supervised(panel: pd.DataFrame, value_col: str):
    """Frame -> (features, target, next_month) rows for next-observation
    prediction. Consecutive report gaps up to 8 months are allowed (quarterly
    cadence) and the gap length is itself a feature."""
    rows = []
    p = panel.sort_values(['project_code', 'report_month'])
    for code, g in p.groupby('project_code', sort=False):
        g = g.dropna(subset=[value_col])
        vals = g[value_col].astype(float).tolist()
        months = g['report_month'].tolist()
        for i in range(2, len(vals)):
            gap = month_index(months[i]) - month_index(months[i - 1])
            if np.isnan(gap) or gap < 1 or gap > 8:
                continue          # same month or too big a gap between reports
            rows.append({
                'project_code': code, 'next_month': months[i], 'target': vals[i],
                'lag1': vals[i - 1], 'lag2': vals[i - 2],
                'lag1_diff': vals[i - 1] - vals[i - 2],
                'months_ahead': float(gap),
                'sector': g['sector'].iloc[i - 1],
                'log_cost': np.log(float(g['original_cost'].iloc[i - 1] or 1)),
            })
    return pd.DataFrame(rows)


def forecast_task(panel: pd.DataFrame, value_col: str = 'cost_overrun_pct'):
    feas = per_project_feasibility(panel)
    sup = _build_supervised(panel, value_col)
    months = sorted(sup['next_month'].unique()) if len(sup) else []
    if len(sup) < 300 or len(months) < 2:
        return {'available': False,
                'reason': 'Insufficient sequential observations for trend '
                          'modelling (needs >= 2 consecutive pairs and >= 300 '
                          'project-month rows).',
                'feasibility': feas}
    # hold out the most recent observed month as test
    test_month = months[-1]
    train = sup[sup['next_month'] != test_month]
    test = sup[sup['next_month'] == test_month]
    if len(test) < 50 or len(train) < 100:
        return {'available': False,
                'reason': 'Too few rows in the most recent month for an honest '
                          'hold-out evaluation.', 'feasibility': feas}

    from catboost import CatBoostRegressor
    from sklearn.metrics import mean_absolute_error
    cat_cols = ['sector']
    feats = ['lag1', 'lag2', 'lag1_diff', 'months_ahead', 'sector', 'log_cost']
    m = CatBoostRegressor(iterations=400, depth=5, learning_rate=0.06,
                          random_seed=42, verbose=False)
    m.fit(train[feats], train['target'], cat_features=cat_cols)
    pred = m.predict(test[feats])
    ytest = test['target'].astype(float)
    mae_model = float(mean_absolute_error(ytest, pred))
    mae_naive = float(mean_absolute_error(ytest, test['lag1']))
    return {
        'available': True, 'value_col': value_col, 'test_month': test_month,
        'n_test': int(len(test)), 'model_mae': mae_model, 'naive_mae': mae_naive,
        'model_beats_naive': bool(mae_model < mae_naive),
        'feasibility': feas,
        'model': m,
        'note': 'Global panel model (CatBoost with lag features). The naive '
                'baseline predicts next month = current month; the panel model '
                'is used only where it beats the baseline on the hold-out month.',
    }
