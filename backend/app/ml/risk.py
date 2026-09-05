"""Transparent implementation-risk score, rule-based warnings and
recommendations.

The risk score is deliberately NOT a black box: every component is a formula
over observable quantities and model outputs, documented here and in the UI.
Component weights are configurable (config.load_risk_weights).
"""
import numpy as np
import pandas as pd

from ..config import load_risk_weights, risk_level


# ------------------------------------------------------------------- helpers
def _num(x):
    try:
        v = float(x)
        return np.nan if np.isinf(v) else v
    except (TypeError, ValueError):
        return np.nan


def months_between(a, b):
    """Months from 'YYYY-MM' a to 'YYYY-MM' b (b - a)."""
    try:
        ay, am = int(str(a)[:4]), int(str(a)[5:7])
        by, bm = int(str(b)[:4]), int(str(b)[5:7])
        return (by * 12 + bm) - (ay * 12 + am)
    except Exception:
        return np.nan


# ------------------------------------------------------------ risk components
def cost_risk(row, pred_prob):
    """0-100. Model's calibrated overrun probability blended with the overrun
    already visible in reported costs (if any). 100% overrun -> full realized
    component of 100."""
    p = _num(pred_prob)
    cor = _num(row.get('cost_overrun_pct'))
    realized = np.nan
    if not np.isnan(cor):
        realized = float(np.clip(cor, -10, 100))
    if not np.isnan(p) and not np.isnan(realized):
        return 0.5 * 100 * p + 0.5 * realized
    if not np.isnan(p):
        return 100 * p
    if not np.isnan(realized):
        return realized
    return 50.0


def schedule_risk(row, delay_prob):
    """0-100. Model's delay probability (where a survival model is available),
    how far past the original target the project already is (30 months past
    saturates), and the current anticipated delay (24 months saturates)."""
    p = _num(delay_prob)
    past = months_between(row.get('original_completion_target'),
                          row.get('report_month'))
    past_component = (np.clip(past / 30.0, 0, 1) * 100
                      if not np.isnan(past) else np.nan)
    tor = _num(row.get('time_overrun_months'))
    tor_component = (np.clip(tor / 24.0, 0, 1) * 100
                     if not np.isnan(tor) else np.nan)
    parts, weights = [], []
    if not np.isnan(p):
        parts.append(100 * p); weights.append(0.4)
    if not np.isnan(past_component):
        parts.append(past_component); weights.append(0.3)
    if not np.isnan(tor_component):
        parts.append(tor_component); weights.append(0.3)
    if not parts:
        return 50.0
    return float(np.average(parts, weights=weights))


def expenditure_risk(row):
    """0-100. Expenditure above original cost, spending ahead of schedule
    fraction, and a financial-vs-physical progress mismatch check."""
    exp_over = _num(row.get('expenditure_over_original_pct'))
    elapsed = _num(row.get('elapsed_fraction'))
    exp_frac = latest_exp_frac(row)
    parts, weights = [], []
    if not np.isnan(exp_over):
        parts.append(float(np.clip(exp_over, 0, 100))); weights.append(0.4)
    if not np.isnan(elapsed) and not np.isnan(exp_frac):
        # spending faster than schedule (both as fractions of plan)
        gap = float(np.clip((exp_frac - elapsed) * 150, 0, 100))
        parts.append(gap); weights.append(0.3)
    prog = _num(row.get('physical_progress_pct'))
    if not np.isnan(prog) and not np.isnan(exp_frac):
        # classic MoSPI red flag: low physical progress, high financial progress
        mismatch = float(np.clip((exp_frac * 100 - prog) * 1.5, 0, 100))
        parts.append(mismatch); weights.append(0.3)
    if not parts:
        return 50.0
    return float(np.average(parts, weights=weights))


def latest_exp_frac(row):
    lc = _num(row.get('latest_cost'))
    ex = _num(row.get('cumulative_expenditure'))
    if np.isnan(lc) or np.isnan(ex) or lc <= 0:
        return np.nan
    return float(np.clip(ex / lc, 0, 1.5))


def reporting_risk(row, anomaly):
    """0-100. Data/reporting quality issues that limit monitoring itself."""
    score = 0.0
    if row.get('approval_date') in (None, '', 'NaT') or \
            pd.isna(row.get('approval_date')):
        score += 35
    if pd.isna(row.get('original_cost')) or _num(row.get('original_cost', np.nan)) <= 0:
        score += 25
    if pd.isna(_num(row.get('physical_progress_pct', np.nan))):
        score += 10                     # progress not reported in this month
    if anomaly:
        score += 40
    return float(min(score, 100))


def compute_risk(row, pred_prob, delay_prob, anomaly=False):
    w = load_risk_weights()
    cr = cost_risk(row, pred_prob)
    sr = schedule_risk(row, delay_prob)
    er = expenditure_risk(row)
    rr = reporting_risk(row, anomaly)
    total = (w['cost_risk'] * cr + w['schedule_risk'] * sr +
             w['expenditure_risk'] * er + w['reporting_risk'] * rr)
    return {
        'total': float(np.clip(total, 0, 100)),
        'level': risk_level(total),
        'components': {'cost_risk': round(cr, 1), 'schedule_risk': round(sr, 1),
                       'expenditure_risk': round(er, 1), 'reporting_risk': round(rr, 1)},
    }


# ------------------------------------------------------------------- warnings
SEVERITY_ORDER = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}

WARN_RULES = {
    'cost_overrun_projected': {
        'severity': 'high', 'what': 'Cost overrun projected / present',
        'action': 'Review cost-to-benefit, expedite revised-cost approval, tighten '
                  'expenditure monitoring.'},
    'severe_cost_overrun': {
        'severity': 'critical', 'what': 'Severe cost overrun (>100%)',
        'action': 'Escalate for appraisal; consider scope re-optimisation.'},
    'schedule_overrun': {
        'severity': 'high', 'what': 'Behind original schedule',
        'action': 'Schedule recovery plan, review tendering/land-acquisition '
                  'bottlenecks.'},
    'severe_schedule_overrun': {
        'severity': 'critical', 'what': 'Schedule overrun exceeds 24 months',
        'action': 'High-level review; consider phased commissioning.'},
    'spend_progress_mismatch': {
        'severity': 'medium', 'what': 'Financial progress far ahead of physical '
        'progress', 'action': 'Audit expenditure certification.'},
    'expenditure_above_original': {
        'severity': 'medium', 'what': 'Expenditure already exceeds original cost',
        'action': 'Verify revised-cost proposal status.'},
    'model_high_risk': {
        'severity': 'high', 'what': 'Model flags high overrun probability',
        'action': 'Pre-emptive review at monitoring agency level.'},
    'anomaly': {
        'severity': 'medium', 'what': 'Unusual pattern detected (statistical '
        'anomaly)', 'action': 'Verify reported figures with the implementing '
        'agency.'},
    'missing_approval_date': {
        'severity': 'low', 'what': 'Approval date missing in source data',
        'action': 'Backfill from administrative records.'},
    'no_recent_report': {
        'severity': 'low', 'what': 'Not reported in the latest census month',
        'action': 'Confirm project status (completed/frozen/dropped).'},
}


def generate_warnings(row, pred_prob, risk, anomaly=False):
    out = []
    cor = _num(row.get('cost_overrun_pct'))
    tor = _num(row.get('time_overrun_months'))
    exp_over = _num(row.get('expenditure_over_original_pct'))
    prog = _num(row.get('physical_progress_pct'))
    exp_frac = latest_exp_frac(row)

    def add(key, reason):
        r = WARN_RULES[key]
        out.append({'warning_type': key, 'severity': r['severity'], 'what': r['what'],
                    'reason': reason, 'action': r['action']})

    if not np.isnan(cor) and cor > 100:
        add('severe_cost_overrun', f'Reported cost overrun is {cor:.0f}%.')
    elif (not np.isnan(cor) and cor > 0) or (pred_prob is not None and
                                             not np.isnan(_num(pred_prob)) and
                                             pred_prob >= 0.7):
        if not np.isnan(cor) and cor > 0:
            add('cost_overrun_projected', f'Reported cost overrun is {cor:.0f}%.')
        else:
            add('model_high_risk',
                f'Model estimates {pred_prob*100:.0f}% probability of cost overrun.')

    if not np.isnan(tor) and tor > 24:
        add('severe_schedule_overrun', f'Anticipated schedule overrun is '
                                       f'{tor:.0f} months.')
    elif not np.isnan(tor) and tor > 0:
        add('schedule_overrun', f'Anticipated schedule overrun is {tor:.0f} months.')

    if not np.isnan(exp_over) and exp_over > 0:
        add('expenditure_above_original',
            f'Expenditure exceeds original cost by {exp_over:.0f}%.')
    if not np.isnan(prog) and not np.isnan(exp_frac) and exp_frac * 100 - prog > 25:
        add('spend_progress_mismatch',
            f'Financial progress {exp_frac*100:.0f}% vs physical progress '
            f'{prog:.0f}%.')
    if anomaly:
        add('anomaly', 'Flagged as an outlier relative to peer projects '
                       '(isolation-forest score).')
    if pd.isna(row.get('approval_date')):
        add('missing_approval_date', 'Approval date is absent in MoSPI source data.')
    if row.get('not_in_latest', False):
        add('no_recent_report', 'Project absent from the most recent report month '
                                'in the dataset.')
    out.sort(key=lambda w: SEVERITY_ORDER[w['severity']])
    return out


# ------------------------------------------------------------- recommendations
REC_MAP = {
    'severe_cost_overrun': 'Commission an independent cost-benefit re-appraisal '
                           'before further sanction.',
    'cost_overrun_projected': 'Fast-track the revised-cost approval cycle and '
                              'enforce monthly expenditure ceilings.',
    'severe_schedule_overrun': 'Undertake a time-overrun review with the '
                               'implementing agency; consider milestone-based '
                               'monitoring.',
    'schedule_overrun': 'Identify binding bottlenecks (land, tendering, statutory '
                        'clearances) and set recovery milestones.',
    'spend_progress_mismatch': 'Order an expenditure-certification audit before '
                               'further disbursement.',
    'expenditure_above_original': 'Regularise the excess through a revised-cost '
                                  'proposal at the competent authority.',
    'model_high_risk': 'Place the project on a pre-emptive watch list with '
                       'quarterly review.',
    'anomaly': 'Cross-verify the reported financial/physical progress with the '
               'project authority.',
    'missing_approval_date': 'Retrieve the approval record from the sanctioning '
                             'authority to enable age-based analytics.',
    'no_recent_report': 'Confirm whether the project has been completed, frozen '
                        'or dropped from the census.',
}


def recommendations(warnings):
    seen, recs = set(), []
    for w in warnings:
        t = w['warning_type']
        if t in REC_MAP and t not in seen:
            seen.add(t)
            recs.append({'based_on': w['what'], 'recommendation': REC_MAP[t],
                         'warning_type': t})
    if not recs:
        recs.append({'based_on': 'No active warnings',
                     'recommendation': 'Continue standard monitoring cadence.',
                     'warning_type': 'none'})
    return recs
