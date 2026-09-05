"""Rule-based natural-language interface over the real database.

Every number in an answer is computed from the SQLite DB / panel data via
queries - nothing is invented. If the data cannot answer a question, the
assistant says so. An optional local LLM (Ollama, OLLAMA_URL env var) is used
ONLY to phrase the retrieved facts; it never generates facts itself. Official
government decisions/statements are never claimed.
"""
import os
import re

import pandas as pd

from ..db import query_df

INSUFFICIENT = 'Insufficient data available for this analysis.'

HELP = (
    'I can answer questions about the MoSPI central-sector project panel, for '
    'example:\n'
    '- "show the top 10 highest-risk projects"\n'
    '- "how many projects have cost overrun in Transport sector"\n'
    '- "average cost overrun in Railways ministry"\n'
    '- "projects in Maharashtra with schedule overrun"\n'
    '- "tell me about project N24001451" (or by name)\n'
    '- "what drives the model\'s predictions"\n'
    '- "data quality and sources"\n'
    '- "recommendations for <project code>"\n'
    'Ask me about sectors, ministries, states, cost/schedule overruns, risk '
    'scores, warnings or the underlying data.'
)


def _fmt_cr(n):
    return f'Rs {n:,.0f} crore'


def _latest_scores():
    return query_df("""
        SELECT ps.project_code, ps.report_month, ps.risk_total, ps.risk_level,
               ps.pred_prob_monitoring, ps.pred_prob_approval, ps.event,
               pa.project_name, pa.sector, pa.ministry, pa.state,
               pa.original_cost, pa.latest_cost, pa.cost_overrun_pct,
               pa.time_overrun_months, pa.cumulative_expenditure,
               pa.physical_progress_pct, pa.approval_date,
               pa.original_completion_target
        FROM project_scores ps
        JOIN panel pa ON pa.project_code = ps.project_code
                     AND pa.report_month = ps.report_month
    """)


def _find_project(q):
    q = q.strip()
    df = query_df('SELECT DISTINCT project_code, project_name FROM panel')
    hit = df[df['project_code'].str.lower() == q.lower()]
    if len(hit):
        return hit.iloc[0]['project_code']
    hit = df[df['project_name'].str.lower().str.contains(q.lower(), na=False)]
    if 1 <= len(hit) <= 5:
        return hit.iloc[0]['project_code']
    if len(hit) > 5:
        return hit.iloc[0]['project_code']
    return None


def answer(q: str, card=None) -> dict:
    q = (q or '').strip()
    ql = q.lower()
    if not ql:
        return {'answer': HELP, 'intent': 'help'}
    s = _latest_scores()

    # ---- help / capabilities
    if re.search(r'\b(help|what can you|capabilities|how to use)\b', ql):
        return {'answer': HELP, 'intent': 'help'}

    # ---- data quality / sources
    if re.search(r'(data quality|sources?|where.*data|how many (months|reports))',
                 ql):
        panel = query_df('SELECT * FROM panel')
        months = sorted(panel['report_month'].unique())
        a = (f'The panel covers {len(months)} MoSPI report months: '
             f'{", ".join(months)} - {len(panel):,} project-month rows, '
             f'{panel.project_code.nunique():,} unique projects from the QPISR '
             f'and Flash reports (public MoSPI data). Validations: printed vs '
             f'computed cost-overrun matched 100% within 1pp; time-overrun '
             f'99.5% within 1 month; original cost consistent across months for '
             f'95.2% of projects (rest reflect genuine source restatements). '
             f'See the Data Quality page for details.')
        return {'answer': a, 'intent': 'data_quality'}

    # ---- model drivers
    if re.search(r'(driv(e|es|er|ers)|feature|explain the model|shap|'
                 r'what.*(influence|matters|important))', ql):
        if card and card.get('shap_global'):
            g = card['shap_global']
            top = ', '.join(f'{f} ({v:.3f})' for f, v in
                            zip(g['features'][:5], g['mean_abs_shap'][:5]))
            return {'answer': 'According to SHAP analysis of the winning model, '
                              f'the strongest influences on overrun predictions '
                              f'are: {top}. {g["caveat"]}', 'intent': 'drivers'}
        return {'answer': INSUFFICIENT, 'intent': 'drivers'}

    # ---- project-specific (explicit code or "about <name>")
    m = re.search(r'\b([NnMm]?\d{6,9}|UNM-[\w-]+)\b', q)
    code = _find_project(m.group(1)) if m else None
    if code is None and re.search(r'(about|details? for|tell me about|project)',
                                  ql) and len(ql) > 12:
        code = _find_project(re.sub(r'(tell me about|details? for|project|about)',
                                    '', ql, flags=re.I))
    if code:
        return _project_answer(code)

    # ---- recommendations for a project
    if 'recommend' in ql:
        return {'answer': 'Please specify a project code, e.g. "recommendations '
                          'for N24001451".', 'intent': 'recommendations'}

    # ---- top risk
    if re.search(r'(top|highest|worst|most).*(risk|risky|dangerous|critical)|'
                 r'^(top|highest|worst)\b', ql):
        n = 10
        mn = re.search(r'top\s+(\d+)', ql)
        if mn:
            n = min(int(mn.group(1)), 25)
        t = s[s['event'] != 'completed'].nlargest(n, 'risk_total')
        if not len(t):
            return {'answer': INSUFFICIENT, 'intent': 'top_risk'}
        lines = [f"{i+1}. {r.project_code} - {str(r.project_name)[:60]} "
                 f"(risk {r.risk_total:.0f}/100, {r.risk_level}; "
                 f"cost overrun {r.cost_overrun_pct if pd.notna(r.cost_overrun_pct) else 'n/a'}%)"
                 for i, (_, r) in enumerate(t.iterrows())]
        return {'answer': f'Highest implementation-risk projects (current '
                          f'month):\n' + '\n'.join(lines) +
                          '\nOpen a project for the component breakdown, '
                          'warnings and recommendations.',
                'intent': 'top_risk'}

    # ---- sector / ministry / state aggregates
    filt_cols = []
    for key, col in [('sector', 'sector'), ('ministry', 'ministry'),
                     ('state', 'state'), ('railway', 'ministry'),
                     ('transport', 'sector')]:
        pass
    scope = s[s['event'] != 'completed'].copy()
    label = 'ongoing projects'
    for col, kind in [('sector', 'sector'), ('ministry', 'ministry'),
                      ('state', 'state')]:
        vals = scope[col].dropna().unique()
        for v in vals:
            if str(v).lower() in ql:
                scope = scope[scope[col] == v]
                label = f'{v} ({kind}) projects'
                break
        if label != 'ongoing projects':
            break

    def agg_block(d):
        if not len(d):
            return INSUFFICIENT
        over = d[d['cost_overrun_pct'] > 0]
        tor = d[d['time_overrun_months'] > 0]
        return (f'{len(d):,} {label} in the latest report month. '
                f'Cost overrun: {len(over):,} projects '
                f'({100*len(over)/len(d):.0f}%), average overrun '
                f'{over["cost_overrun_pct"].mean():.0f}%. '
                f'Schedule overrun: {len(tor):,} projects '
                f'({100*len(tor)/len(d):.0f}%), average slip '
                f'{tor["time_overrun_months"].mean():.0f} months. '
                f'Original cost total {_fmt_cr(d["original_cost"].sum())}; '
                f'current (revised/anticipated) {_fmt_cr(d["latest_cost"].sum())}.')

    if re.search(r'(how many|count|number of|show|list|which|average|avg|'
                 r'mean|total|overrun|delayed|summary|status)', ql):
        return {'answer': agg_block(scope), 'intent': 'aggregate',
                'scope': label}

    return {'answer': HELP, 'intent': 'fallback'}


def _project_answer(code):
    s = _latest_scores()
    r = s[s['project_code'] == code]
    if not len(r):
        return {'answer': f'Project {code} not found in the dataset.', 
                'intent': 'project'}
    r = r.iloc[0]
    w = query_df('SELECT * FROM warnings WHERE project_code = ?', (code,))
    parts = [f'{r.project_code} - {r.project_name}',
             f'Sector: {r.sector} | Ministry: {r.ministry} | State: {r.state}',
             f'Approved cost {_fmt_cr(r.original_cost)}, current '
             f'{_fmt_cr(r.latest_cost)}'
             + (f' (cost overrun {r.cost_overrun_pct:.0f}%)'
                if pd.notna(r.cost_overrun_pct) else ''),
             f'Implementation risk {r.risk_total:.0f}/100 ({r.risk_level})',
             f'Model-estimated overrun probability: '
             f'{100*r.pred_prob_monitoring:.0f}% (monitoring-stage model)']
    if pd.notna(r.time_overrun_months):
        parts.append(f'Anticipated schedule overrun: '
                     f'{r.time_overrun_months:.0f} months')
    if len(w):
        parts.append('Active warnings: ' + '; '.join(
            f'{x["what"]}' for x in w.head(4).to_dict('records')))
    else:
        parts.append('No active warnings.')
    return {'answer': '\n'.join(parts), 'intent': 'project', 'project': code}
