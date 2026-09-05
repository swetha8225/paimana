"""SHAP explanations for the winning classification model.

Framing rule: SHAP values are presented as 'how strongly a feature influenced
the model's prediction' - an association statement, never a causal claim.
"""
import numpy as np
import pandas as pd

from .features import build_features

CAVEAT = ('SHAP values show how strongly each feature influenced the model\'s '
          'prediction for this project. They are associations learned from '
          'historical data, not causal effects.')


def _shap_matrix(base_model, X: pd.DataFrame, max_rows=2000):
    """Walk the pipeline's non-final steps (prep, optional cat cast) and return
    (Xt, clf, raw_feature_names)."""
    Xs = X.iloc[:max_rows]
    Xt = Xs
    for name, step in base_model.named_steps.items():
        if name == base_model.steps[-1][0]:
            break
        Xt = step.transform(Xt)
    return Xt, base_model.steps[-1][1], list(Xs.columns)


def _prep_and_explain(base_model, X: pd.DataFrame, max_rows=2000):
    """Explain the underlying (uncalibrated) model with TreeExplainer; fall
    back to coefficient-based contributions for linear models."""
    Xs = X.iloc[:max_rows]
    Xt, clf, raw_names = _shap_matrix(base_model, X, max_rows)
    Xt_arr = np.asarray(Xt, dtype=float)
    try:
        import shap
        sv = shap.TreeExplainer(clf).shap_values(Xt)
        if isinstance(sv, list):          # binary classifiers may return a list
            sv = sv[1]
        if sv is not None and getattr(sv, 'ndim', 0) == 3:
            sv = sv[:, :, 1]
        if sv is None:
            raise ValueError('no shap values')
        sv = np.asarray(sv)
        # ordinal-encoded path preserves the raw feature order
        names = raw_names if sv.shape[1] == len(raw_names) else \
            [f'f{i}' for i in range(sv.shape[1])]
        return pd.DataFrame(sv, columns=names, index=Xs.index)
    except Exception:
        # linear fallback: signed coefficient contributions beta_i * (x_i - mean)
        try:
            coefs = np.asarray(clf.coef_).ravel()
            if len(coefs) != Xt_arr.shape[1]:
                return None
            sv = (Xt_arr - Xt_arr.mean(axis=0)) * coefs
            names = _onehot_names(base_model, Xt_arr)
            return pd.DataFrame(sv, columns=names, index=Xs.index)
        except Exception:
            return None


def _onehot_names(base_model, Xt):
    """Names for one-hot expanded matrices (linear models)."""
    out = []
    try:
        prep = base_model.named_steps['prep']
        for name, trans, cols in prep.ct_.transformers_:
            if trans == 'drop' or trans is None or name == 'remainder':
                continue
            col_list = cols if isinstance(cols, (list, tuple)) else [cols]
            if type(trans).__name__ == 'OneHotEncoder':
                for c, cats in zip(col_list, trans.categories_):
                    out += [f'{c}={a}' for a in cats]
            else:
                out += list(col_list)
    except Exception:
        pass
    if len(out) != Xt.shape[1]:
        out = [f'f{i}' for i in range(Xt.shape[1])]
    return out


def global_importance(base_model, df: pd.DataFrame, stage='approval'):
    X = build_features(df, stage)
    sv = _prep_and_explain(base_model, X)
    if sv is None:
        return None
    means = sv.abs().mean().sort_values(ascending=False)
    return {
        'features': means.head(15).index.tolist(),
        'mean_abs_shap': means.head(15).round(4).tolist(),
        'caveat': CAVEAT,
    }


def project_explanation(base_model, project_row: pd.DataFrame, stage='approval'):
    """SHAP explanation for a single project row (a one-row frame with the raw
    panel columns)."""
    X = build_features(project_row, stage)
    sv = _prep_and_explain(base_model, X)
    if sv is None:
        return {'available': False}
    s = sv.iloc[0].sort_values(key=np.abs, ascending=False)
    top = s.head(8)
    return {
        'available': True,
        'caveat': CAVEAT,
        'features': [{'feature': k, 'shap': float(v),
                      'direction': 'increases risk' if v > 0 else 'decreases risk'}
                     for k, v in top.items()],
    }
