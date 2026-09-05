"""
Leakage-safe feature construction.

Two information sets are distinguished explicitly:

- APPROVAL STAGE: information known when the project is approved
  (sector, ministry, state, original cost, planned duration, approval year).
  Nothing that happens after approval is included.

- MONITORING STAGE: adds information that accumulates during execution and is
  known at the report month (project age, elapsed fraction of schedule,
  expenditure pattern). Post-outcome fields (revised/anticipated cost,
  cost-overrun %, time overrun, actual completion) are NEVER features.

The classification/regression targets (cost overrun flag / %) are derived from
revised/anticipated vs original cost and are therefore excluded from features.
"""
import numpy as np
import pandas as pd

CAT_FEATURES = ['sector', 'ministry', 'state']
NUM_APPROVAL = ['log_original_cost', 'planned_duration_months', 'approval_year']
NUM_MONITOR = ['project_age_months', 'elapsed_fraction', 'expenditure_over_original_pct',
               'report_month_idx']

FEATURES_APPROVAL = CAT_FEATURES + NUM_APPROVAL
FEATURES_MONITOR = CAT_FEATURES + NUM_APPROVAL + NUM_MONITOR

# Documentation surfaced in the UI
FEATURE_DOCS = {
    'sector': ('Sector of the project', 'approval'),
    'ministry': ('Administering ministry/department (derived from sector)', 'approval'),
    'state': ('State where the project is located', 'approval'),
    'log_original_cost': ('log( original approved cost in Rs crore )', 'approval'),
    'planned_duration_months': ('Original completion target - approval date (months)',
                                'approval'),
    'approval_year': ('Year of approval', 'approval'),
    'project_age_months': ('Report month - approval date (months)', 'monitoring'),
    'elapsed_fraction': ('Project age / planned duration', 'monitoring'),
    'expenditure_over_original_pct': ('(Cumulative expenditure - original cost) / original '
                                      'cost x 100, as known at the report month',
                                      'monitoring'),
    'report_month_idx': ('Report month as a numeric index', 'monitoring'),
}

# Explicit leakage statement for the Model Performance page
LEAKAGE_NOTES = [
    'Excluded as features: revised cost, anticipated cost, latest cost, '
    'cost-overrun %, time overrun, cost-overrun flag, actual/revised completion '
    'dates, physical progress-linked outcome fields.',
    'These fields define the prediction targets; using them as inputs would '
    'leak the outcome into the model.',
    'Monitoring-stage features use only information accumulated up to the '
    'report month (age, elapsed schedule fraction, expenditure pattern).',
    'Validation is time-based: earlier report months train, later months '
    'validate, the most recent month is held out for testing.',
]


# Zone mapping keeps the survival model parsimonious: with a limited number
# of completed projects, states are grouped into planning regions.
STATE_ZONES = {
    'North': ['Jammu & Kashmir', 'Himachal Pradesh', 'Punjab', 'Haryana',
              'Uttarakhand', 'Delhi', 'Chandigarh', 'Ladakh'],
    'South': ['Andhra Pradesh', 'Karnataka', 'Kerala', 'Tamil Nadu',
              'Telangana', 'Puducherry', 'Lakshadweep'],
    'East': ['Bihar', 'Jharkhand', 'Odisha', 'West Bengal',
             'Andaman & Nicobar'],
    'West': ['Rajasthan', 'Gujarat', 'Maharashtra', 'Goa', 'Daman & Diu',
             'Dadra & Nagar Haveli'],
    'Central': ['Uttar Pradesh', 'Madhya Pradesh', 'Chhattisgarh'],
    'Northeast': ['Assam', 'Meghalaya', 'Manipur', 'Mizoram', 'Nagaland',
                  'Tripura', 'Arunachal Pradesh', 'Sikkim'],
}
_STATE_TO_ZONE = {s: z for z, ss in STATE_ZONES.items() for s in ss}


def state_zone(state) -> str:
    if state is None or (isinstance(state, float) and np.isnan(state)):
        return 'Other'
    s = str(state).strip()
    if s in _STATE_TO_ZONE:
        return _STATE_TO_ZONE[s]
    if not s or s.lower() in ('nan', 'unknown', 'multiple states', 'national',
                              'multi-state'):
        return 'Other'
    return 'Other'


def month_index(ym) -> float:
    if ym is None or (isinstance(ym, float) and np.isnan(ym)):
        return np.nan
    try:
        y, m = str(ym).split('-')[:2]
        return int(y) * 12 + int(m) - 1
    except Exception:
        return np.nan


def build_features(df: pd.DataFrame, stage: str = 'approval') -> pd.DataFrame:
    """Return feature frame aligned to df index."""
    out = pd.DataFrame(index=df.index)
    for c in CAT_FEATURES:
        out[c] = df[c].astype('object').where(df[c].notna(), 'Unknown')
    out['log_original_cost'] = np.log(df['original_cost'].astype(float).clip(lower=0.01))
    out['planned_duration_months'] = df['planned_duration_months'].astype(float)
    out['approval_year'] = df['approval_date'].map(
        lambda s: float(str(s)[:4]) if isinstance(s, str) and len(str(s)) >= 4 else np.nan)
    if stage == 'monitoring':
        out['project_age_months'] = df['project_age_months'].astype(float)
        out['elapsed_fraction'] = df['elapsed_fraction'].astype(float)
        out['expenditure_over_original_pct'] = \
            df['expenditure_over_original_pct'].astype(float)
        out['report_month_idx'] = df['report_month'].map(month_index)
    return out


def time_split_months(months: pd.Series, train_frac=0.6, val_frac=0.25):
    """Split the sorted unique report months into train / validation / test.

    Earlier months -> train, later -> validation, most recent -> test.
    Returns (train_months, val_months, test_months)."""
    uniq = sorted(m for m in months.dropna().unique())
    if len(uniq) < 3:
        # fall back: chronological leave-last-out if possible
        if len(uniq) == 2:
            return [uniq[0]], [], [uniq[1]]
        return uniq, [], []
    n = len(uniq)
    n_train = max(1, int(round(n * train_frac)))
    n_val = max(1, int(round(n * val_frac)))
    n_val = min(n_val, n - n_train - 1)
    return uniq[:n_train], uniq[n_train:n_train + n_val], uniq[n_train + n_val:]
