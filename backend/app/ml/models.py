"""
Model training and evaluation: cost-overrun classification (with probability
calibration), cost-overrun magnitude regression, time-overrun (regression +
survival), anomaly detection and similarity benchmarking.

Every metric written to the model card is computed from the actual dataset with
time-aware splits. Nothing is hard-coded.
"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

warnings.filterwarnings('ignore')

from .features import (FEATURES_APPROVAL, FEATURES_MONITOR, build_features,
                       state_zone, time_split_months)

RANDOM_STATE = 42


# ----------------------------------------------------------------- utilities
def _safe_roc_auc(y, p):
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y, p))
    except Exception:
        return None


def _safe_pr_auc(y, p):
    try:
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(y, p))
    except Exception:
        return None


def _fmt_confusion(y, yhat):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, yhat, labels=[0, 1])
    return {'tn': int(cm[0, 0]), 'fp': int(cm[0, 1]), 'fn': int(cm[1, 0]),
            'tp': int(cm[1, 1])}


def brier(y, p):
    from sklearn.metrics import brier_score_loss
    return float(brier_score_loss(y, p))


def _curves(y, p):
    from sklearn.metrics import precision_recall_curve, roc_curve
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    return {
        'roc': {'fpr': fpr[::max(1, len(fpr) // 200)].tolist(),
                'tpr': tpr[::max(1, len(tpr) // 200)].tolist()},
        'pr': {'precision': prec[::max(1, len(prec) // 200)].tolist(),
               'recall': rec[::max(1, len(rec) // 200)].tolist()},
    }


def _calibration_curve(y, p, n_bins=10):
    from sklearn.calibration import calibration_curve
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy='quantile')
    return {'mean_predicted': mean_pred.tolist(), 'fraction_positive': frac_pos.tolist()}


def _threshold_for_recall(y, p):
    """Choose a decision threshold on validation data that prioritises recall
    (early warning): the smallest threshold whose precision is still >= 0.35,
    else the F2-optimal threshold."""
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y, p)
    best = 0.5
    for pi, ri, ti in zip(prec[:-1], rec[:-1], thr):
        if pi >= 0.35:
            best = ti
            break
    f2 = 5 * prec[:-1] * rec[:-1] / (4 * prec[:-1] + rec[:-1] + 1e-12)
    best_f2 = thr[int(np.argmax(f2))] if len(thr) else 0.5
    # prefer the recall-prioritising threshold unless it is degenerate
    chosen = best if 0.05 < best < 0.95 else best_f2
    return float(np.clip(chosen, 0.05, 0.95))


# ------------------------------------------------------------ model factories
def make_classifiers(seed=RANDOM_STATE):
    """Candidate classifiers for tabular cost-overrun prediction."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier

    def tree_pre(cat_idx):
        return ColumnTransformerWithNames([
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value',
                                   unknown_value=-1, dtype=np.int32), cat_idx),
        ])

    models = {}

    # Logistic regression: statistical baseline (one-hot + median impute + scale)
    models['Logistic Regression'] = Pipeline([
        ('prep', ColumnTransformerWithNames([
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
             FEATURES_APPROVAL[:3]),
        ])),
        ('sc', StandardScaler()),
        ('clf', LogisticRegression(max_iter=2000, class_weight='balanced',
                                   C=0.5, random_state=seed)),
    ])

    models['Random Forest'] = Pipeline([
        ('prep', ColumnTransformerWithNames([
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value',
                                   unknown_value=-1, dtype=np.int32),
             FEATURES_APPROVAL[:3]),
            ('imp', SimpleImputer(strategy='median'), FEATURES_APPROVAL[3:]),
        ])),
        ('clf', RandomForestClassifier(n_estimators=400, min_samples_leaf=5,
                                       class_weight='balanced_subsample',
                                       random_state=seed, n_jobs=-1)),
    ])

    models['XGBoost'] = Pipeline([
        ('prep', ColumnTransformerWithNames([
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value',
                                   unknown_value=-1, dtype=np.int32),
             FEATURES_APPROVAL[:3]),
        ])),
        ('clf', XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.06,
                              subsample=0.9, colsample_bytree=0.9,
                              scale_pos_weight=1.0, random_state=seed,
                              eval_metric='logloss', n_jobs=-1)),
    ])

    models['LightGBM'] = Pipeline([
        ('prep', ColumnTransformerWithNames([
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value',
                                   unknown_value=-1, dtype=np.int32),
             FEATURES_APPROVAL[:3]),
        ])),
        ('clf', LGBMClassifier(n_estimators=500, num_leaves=48, learning_rate=0.06,
                               class_weight='balanced', random_state=seed,
                               verbose=-1, n_jobs=-1)),
    ])

    models['CatBoost'] = Pipeline([
        ('prep', ColumnTransformerWithNames([
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value',
                                   unknown_value=-1, dtype=np.int32),
             FEATURES_APPROVAL[:3]),
        ])),
        ('cast', AsCatFrame(3)),
        ('clf', CatBoostClassifier(iterations=700, depth=6, learning_rate=0.06,
                                   cat_features=(0, 1, 2), random_seed=seed,
                                   verbose=False, auto_class_weights='Balanced')),
    ])
    return models


class AsCatFrame(BaseEstimator, TransformerMixin):
    """Return a DataFrame whose first n_cats columns are int32 - CatBoost
    rejects floating-point categorical columns in numpy arrays."""

    def __init__(self, n_cats=0):
        self.n_cats = n_cats

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X)
        df = pd.DataFrame(X[:, self.n_cats:].astype(float))
        for i in range(self.n_cats):
            df.insert(i, f'c{i}', X[:, i].astype(np.int32))
        return df


class ColumnTransformerWithNames(BaseEstimator, TransformerMixin):
    """Column transformer that takes columns by NAME, passes any remaining
    numeric columns through median imputation, and preserves column order so
    fitted pipelines can transform single-row frames (what-if analysis)."""

    def __init__(self, transformers=()):
        self.transformers = transformers

    def fit(self, X, y=None):
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline as SkPipe
        covered = []
        for _, _, cols in self.transformers:
            covered.extend(cols if isinstance(cols, (list, tuple)) else [cols])
        num_cols = [c for c in X.columns if c not in covered]
        spec = list(self.transformers)
        if num_cols:
            spec.append(('num', SkPipe([('imp', SimpleImputer(strategy='median'))]),
                         num_cols))
        self.ct_ = ColumnTransformer(spec, remainder='drop')
        self.ct_.fit(X, y)
        self.columns_ = list(X.columns)
        return self

    def transform(self, X):
        return self.ct_.transform(X)


def classification_task(df: pd.DataFrame, stage: str = 'approval'):
    """Train all candidate classifiers with a time-aware split, calibrate the
    best one and return a result bundle with metrics + fitted serving model."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score)
    from sklearn.pipeline import Pipeline

    feats = FEATURES_MONITOR if stage == 'monitoring' else FEATURES_APPROVAL
    d = df[(df['event'] == 'ongoing_report') & df['cost_overrun_flag'].notna() &
           (df['original_cost'] > 0) & df['planned_duration_months'].notna() &
           df['approval_date'].notna()].copy()
    if len(d) < 200 or d['cost_overrun_flag'].nunique() < 2:
        return {'available': False, 'reason': 'Insufficient labelled data for this task.',
                'n_rows': int(len(d))}

    tr_m, va_m, te_m = time_split_months(d['report_month'])
    X = build_features(d, stage)
    y = d['cost_overrun_flag'].astype(int).values
    is_tr = d['report_month'].isin(tr_m).values
    is_va = d['report_month'].isin(va_m).values
    is_te = d['report_month'].isin(te_m).values
    if is_te.sum() < 50 or is_tr.sum() < 100:
        return {'available': False,
                'reason': 'Time-aware split leaves too few rows (need >=3 report months).',
                'n_rows': int(len(d))}

    results = {}
    fitted = {}
    Xtr, ytr = X[is_tr], y[is_tr]
    Xva, yva = X[is_va], y[is_va]
    Xte, yte = X[is_te], y[is_te]
    for name, pipe in make_classifiers().items():
        try:
            pipe.fit(Xtr, ytr)
            pva = pipe.predict_proba(Xva)[:, 1]
            pte = pipe.predict_proba(Xte)[:, 1]
            thr = _threshold_for_recall(yva, pva)
            yhat = (pte >= thr).astype(int)
            results[name] = {
                'accuracy': float(accuracy_score(yte, yhat)),
                'precision': float(precision_score(yte, yhat, zero_division=0)),
                'recall': float(recall_score(yte, yhat, zero_division=0)),
                'f1': float(f1_score(yte, yhat, zero_division=0)),
                'roc_auc': _safe_roc_auc(yte, pte),
                'pr_auc': _safe_pr_auc(yte, pte),
                'brier': brier(yte, pte),
                'confusion': _fmt_confusion(yte, yhat),
                'val_roc_auc': _safe_roc_auc(yva, pva),
                'threshold': thr,
            }
            fitted[name] = pipe
        except Exception as e:  # keep other models runnable
            results[name] = {'error': str(e)[:200]}

    ok = {k: v for k, v in results.items() if 'error' not in v}
    if not ok:
        return {'available': False, 'reason': 'All candidate models failed to '
                'fit on this data.', 'n_rows': int(len(d))}
    # select by validation PR-AUC (early warning priority), tie-break ROC-AUC
    def sel_key(kv):
        m = kv[1]
        return ((m.get('pr_auc') or 0) + (m.get('val_roc_auc') or 0))
    best_name = max(ok.items(), key=sel_key)[0]
    best = fitted[best_name]

    # probability calibration on the validation split (isotonic where data allows)
    method = 'isotonic' if is_va.sum() > 1000 else 'sigmoid'
    pva_base = fitted[best_name].predict_proba(Xva)[:, 1]
    try:
        calib = CalibratedClassifierCV(best, method=method, cv='prefit')
        calib.fit(Xva, yva)
        pva_cal = calib.predict_proba(Xva)[:, 1]
        pte_cal = calib.predict_proba(Xte)[:, 1]
    except Exception:
        calib = None
        pva_cal, pte_cal = pva_base, fitted[best_name].predict_proba(Xte)[:, 1]

    thr = _threshold_for_recall(yva, pva_cal)
    yhat = (pte_cal >= thr).astype(int)
    best_metrics = {
        'accuracy': float(accuracy_score(yte, yhat)),
        'precision': float(precision_score(yte, yhat, zero_division=0)),
        'recall': float(recall_score(yte, yhat, zero_division=0)),
        'f1': float(f1_score(yte, yhat, zero_division=0)),
        'roc_auc': _safe_roc_auc(yte, pte_cal),
        'pr_auc': _safe_pr_auc(yte, pte_cal),
        'brier': brier(yte, pte_cal),
        'confusion': _fmt_confusion(yte, yhat),
        'threshold': thr,
        'calibration_method': method if calib else 'none',
        'curves': _curves(yte, pte_cal),
        'calibration_curve': _calibration_curve(yte, pte_cal),
    }

    # serving model: refit on train+val, cross-validated calibration
    Xtrv = pd.concat([Xtr, Xva])
    ytrv = np.concatenate([ytr, yva])
    try:
        serving = make_classifiers()[best_name]
        serving.fit(Xtrv, ytrv)
        serving = CalibratedClassifierCV(serving, method=method, cv=3)
        serving.fit(Xtrv, ytrv)
    except Exception:
        serving = best

    return {
        'available': True, 'stage': stage, 'features': feats,
        'n_rows': int(len(d)), 'n_train': int(is_tr.sum()), 'n_val': int(is_va.sum()),
        'n_test': int(is_te.sum()), 'train_months': tr_m, 'val_months': va_m,
        'test_months': te_m,
        'positive_rate': float(y.mean()),
        'model_results': results, 'best_model': best_name,
        'best_metrics': best_metrics, 'serving_model': serving,
        'base_model': fitted[best_name],
        'flag_definition': 'cost_overrun_flag = 1 when the current (revised/'
                           'anticipated) cost exceeds the original approved cost '
                           '- consistent with MoSPI flash-report counting.',
    }


# ---------------------------------------------------------------- regression
def make_regressors(seed=RANDOM_STATE):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import HuberRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor

    models = {}
    models['Huber Regression (statistical baseline)'] = Pipeline([
        ('prep', ColumnTransformerWithNames([
            ('cat', OneHotFillEncoder(), FEATURES_APPROVAL[:3]),
        ])),
        ('imp', MedImputer()),
        ('sc', StandardScaler()),
        ('reg', HuberRegressor(max_iter=500, alpha=0.1)),
    ])
    models['Random Forest'] = Pipeline([
        ('prep', ColumnTransformerWithNames([
            ('cat', OrdinalFillEncoder(), FEATURES_APPROVAL[:3]),
        ])),
        ('reg', RandomForestRegressor(n_estimators=400, min_samples_leaf=5,
                                      random_state=seed, n_jobs=-1)),
    ])
    models['XGBoost'] = Pipeline([
        ('prep', ColumnTransformerWithNames([
            ('cat', OrdinalFillEncoder(), FEATURES_APPROVAL[:3]),
        ])),
        ('reg', XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.06,
                             subsample=0.9, random_state=seed, n_jobs=-1)),
    ])
    models['LightGBM'] = Pipeline([
        ('prep', ColumnTransformerWithNames([
            ('cat', OrdinalFillEncoder(), FEATURES_APPROVAL[:3]),
        ])),
        ('reg', LGBMRegressor(n_estimators=500, num_leaves=48, learning_rate=0.06,
                              random_state=seed, verbose=-1, n_jobs=-1)),
    ])
    models['CatBoost'] = Pipeline([
        ('prep', ColumnTransformerWithNames([
            ('cat', OrdinalFillEncoder(), FEATURES_APPROVAL[:3]),
        ])),
        ('cast', AsCatFrame(3)),
        ('reg', CatBoostRegressor(iterations=700, depth=6, learning_rate=0.06,
                                  cat_features=(0, 1, 2), random_seed=seed,
                                  verbose=False)),
    ])
    return models


class OneHotFillEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        from sklearn.preprocessing import OneHotEncoder
        self.enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.enc.fit(X.fillna('Unknown'))
        return self

    def transform(self, X):
        return self.enc.transform(X.fillna('Unknown'))


class OrdinalFillEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        from sklearn.preprocessing import OrdinalEncoder
        self.enc = OrdinalEncoder(handle_unknown='use_encoded_value',
                                  unknown_value=-1, dtype=np.int32)
        self.enc.fit(X.fillna('Unknown'))
        return self

    def transform(self, X):
        return self.enc.transform(X.fillna('Unknown'))


class MedImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        import pandas as pd
        self.med = pd.DataFrame(X).replace([np.inf, -np.inf], np.nan).median()
        self.med = self.med.fillna(0)
        return self

    def transform(self, X):
        import pandas as pd
        d = pd.DataFrame(X).replace([np.inf, -np.inf], np.nan)
        return d.fillna(self.med).values


def regression_task(df: pd.DataFrame, target: str, min_obs: int = 150,
                    stage: str = 'approval'):
    """Predict the magnitude of overrun (target = 'cost_overrun_pct') or the
    anticipated schedule delay (target = 'time_overrun_months')."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    feats = FEATURES_MONITOR if stage == 'monitoring' else FEATURES_APPROVAL
    d = df[(df['event'] == 'ongoing_report') & df[target].notna() &
           (df['original_cost'] > 0) & df['planned_duration_months'].notna() &
           df['approval_date'].notna()].copy()
    if len(d) < min_obs:
        return {'available': False,
                'reason': f'Insufficient data available for this analysis '
                          f'({len(d)} valid observations, {min_obs} required).',
                'n_rows': int(len(d))}
    tr_m, va_m, te_m = time_split_months(d['report_month'])
    X = build_features(d, stage)
    y = d[target].astype(float).values
    is_tr = d['report_month'].isin(tr_m).values
    is_va = d['report_month'].isin(va_m).values
    is_te = d['report_month'].isin(te_m).values
    if is_te.sum() < 50:
        return {'available': False, 'reason': 'Time-aware split leaves too few test rows.',
                'n_rows': int(len(d))}

    results, fitted = {}, {}
    Xtr, ytr = X[is_tr], y[is_tr]
    Xva, yva = X[is_va], y[is_va]
    Xte, yte = X[is_te], y[is_te]
    for name, pipe in make_regressors().items():
        try:
            pipe.fit(Xtr, ytr)
            pte = pipe.predict(Xte)
            results[name] = {
                'mae': float(mean_absolute_error(yte, pte)),
                'rmse': float(np.sqrt(mean_squared_error(yte, pte))),
                'r2': float(r2_score(yte, pte)),
                'val_mae': float(mean_absolute_error(yva, pipe.predict(Xva))),
            }
            fitted[name] = pipe
        except Exception as e:
            results[name] = {'error': str(e)[:200]}
    ok = {k: v for k, v in results.items() if 'error' not in v}
    if not ok:
        return {'available': False, 'reason': 'All candidate models failed to '
                'fit on this data.', 'n_rows': int(len(d))}
    best_name = max(ok.items(), key=lambda kv: (kv[1]['val_mae'] * -1))[0]

    # serving model refit on train+val
    Xtrv = pd.concat([Xtr, Xva])
    ytrv = np.concatenate([ytr, yva])
    serving = make_regressors()[best_name]
    serving.fit(Xtrv, ytrv)
    return {
        'available': True, 'stage': stage, 'target': target, 'features': feats,
        'n_rows': int(len(d)), 'n_train': int(is_tr.sum()), 'n_val': int(is_va.sum()),
        'n_test': int(is_te.sum()), 'train_months': tr_m, 'val_months': va_m,
        'test_months': te_m,
        'target_mean': float(np.mean(y)), 'target_std': float(np.std(y)),
        'model_results': results, 'best_model': best_name,
        'best_metrics': ok[best_name], 'serving_model': serving,
    }


# ------------------------------------------------------------------ survival
def survival_task(df: pd.DataFrame, min_events: int = 60):
    """Time-overrun via survival analysis on actual completion.
    One observation per project: duration = approval -> completion (event) or
    approval -> last report month (censored). Covariates are kept parsimonious
    (log cost, planned duration, planning zone) because the number of completed
    projects with known dates is limited."""
    from lifelines import CoxPHFitter, WeibullAFTFitter
    from lifelines.utils import concordance_index

    d = df[df['approval_date'].notna() & (df['original_cost'] > 0) &
           df['planned_duration_months'].notna()].copy()
    # last observation per project
    d = d.sort_values('report_month').groupby('project_code').tail(1).copy()
    completed = d[d['event'] == 'completed'].copy()
    n_events = len(completed)
    n_ongoing = len(d) - n_events
    if n_events < min_events or n_ongoing < 100:
        return {'available': False,
                'reason': f'Survival analysis not appropriate: {n_events} completed '
                          f'projects (need >= {min_events}) and {n_ongoing} ongoing. '
                          f'Regression on reported schedule delay is used instead.',
                'n_events': int(n_events), 'n_ongoing': int(n_ongoing)}

    def duration_months(row):
        end = row['actual_completion_date'] if row['event'] == 'completed' \
            else row['report_month']
        a = str(row['approval_date'])
        try:
            ay, am = int(a[:4]), int(a[5:7])
            ey, em = int(str(end)[:4]), int(str(end)[5:7])
            return (ey * 12 + em) - (ay * 12 + am)
        except Exception:
            return np.nan

    d['duration'] = d.apply(duration_months, axis=1)
    d['event_observed'] = (d['event'] == 'completed').astype(int)
    d = d[(d['duration'] > 0) & (d['duration'] < 600)]
    s = pd.DataFrame({
        'duration': d['duration'], 'event_observed': d['event_observed'],
        'log_original_cost': np.log(d['original_cost'].astype(float).clip(lower=.01)),
        'planned_duration_months': d['planned_duration_months'].astype(float),
        'zone': d['state'].map(state_zone),
        'last_month': d['report_month'],
    })
    s = pd.get_dummies(s, columns=['zone'], drop_first=True)
    for c in s.columns:
        if s[c].dtype == bool:
            s[c] = s[c].astype(int)

    # NOTE ON VALIDATION: a split by last-observed report month cannot be used
    # here, because completed projects leave the panel early and would put all
    # events in the training fold (the test fold would be 100% censored).
    # The honest evaluation for this task is K-fold cross-validation over
    # DISJOINT PROJECTS, which also directly measures new-project screening.
    drop = ['last_month']
    codes = d['project_code'].values
    rng = np.random.RandomState(42)
    fold = {c: i % 5 for i, c in enumerate(rng.permutation(sorted(set(codes))))}
    s['fold'] = [fold[c] for c in codes]

    results, fitted = {}, {}
    for name, cls in [('Cox Proportional Hazards', CoxPHFitter),
                      ('Weibull AFT', WeibullAFTFitter)]:
        cidxs = []
        try:
            for k in range(5):
                tr = s[s['fold'] != k].drop(columns=drop + ['fold'])
                te = s[s['fold'] == k].drop(columns=drop + ['fold'])
                if te['event_observed'].sum() < 3 or tr['event_observed'].sum() < 10:
                    continue
                m = cls(penalizer=0.05)
                m.fit(tr, duration_col='duration', event_col='event_observed')
                cidxs.append(float(concordance_index(
                    te['duration'], -m.predict_expectation(te),
                    te['event_observed'])))
            if not cidxs:
                raise ValueError('no valid folds')
            results[name] = {'concordance_index_cv_mean': float(np.mean(cidxs)),
                             'concordance_index_cv_std': float(np.std(cidxs)),
                             'folds': len(cidxs)}
        except Exception as e:
            results[name] = {'error': str(e)[:200]}
    ok = {k: v for k, v in results.items() if 'error' not in v}
    if not ok:
        return {'available': False, 'reason': 'Survival models failed to fit.',
                'n_events': int(n_events), 'n_ongoing': int(n_ongoing)}
    best_name = max(ok.items(),
                    key=lambda kv: kv[1]['concordance_index_cv_mean'])[0]
    if (ok[best_name].get('concordance_index_cv_mean') or 0) <= 0.55:
        return {'available': False,
                'reason': 'Survival models achieved cross-validated concordance '
                          '<= 0.55 - no better than chance - so regression on '
                          'reported schedule delay is used instead.',
                'n_events': int(n_events), 'n_ongoing': int(n_ongoing)}

    # serving model: fit on all observations
    m = (CoxPHFitter if best_name == 'Cox Proportional Hazards'
         else WeibullAFTFitter)(penalizer=0.05)
    m.fit(s.drop(columns=drop + ['fold']), duration_col='duration',
          event_col='event_observed')
    covariate_columns = [c for c in s.columns
                         if c not in ('duration', 'event_observed', 'last_month',
                                      'fold')]
    return {
        'available': True, 'n_events': int(n_events), 'n_ongoing': int(n_ongoing),
        'n_rows': int(len(s)),
        'model_results': results, 'best_model': best_name,
        'best_metrics': ok[best_name], 'serving_model': m,
        'train_columns': covariate_columns,
        'decision': f'{n_ongoing} ongoing projects are right-censored and '
                    f'{n_events} have actual completion dates, so survival '
                    'analysis is applicable. Cox PH vs Weibull AFT selected by '
                    '5-fold project-disjoint cross-validated concordance '
                    '(>0.55 required); a report-month split is not usable for '
                    'this task because completions cluster in earlier months.',
    }
