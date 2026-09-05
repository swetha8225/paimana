"""
Build the canonical PAIMANA-style project-month panel from parsed MoSPI reports.

Sources (all publicly published by MoSPI):
- Monthly Flash Reports (April/May/August/September 2024): ongoing-project
  census annexures + closed-event table.
- QPISR quarterly reports (Q1 2024-25 = June 2024, Q4 2024-25 = March 2025):
  full ongoing census (Table 7) + completed lists (Table 3).

All derived fields are computed here with explicit, documented formulas.
Nothing is imputed or fabricated; missing source values stay missing.
"""
import json
import os
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
INTERIM = os.path.join(HERE, 'interim')
OUTDIR = os.path.join(HERE, '..', 'data')
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------------- mappings ---
SECTOR_FIX = {
    'TELECOMMUNICATIONS': 'TELECOMMUNICATIONS', 'COMMUNICATIONS': 'TELECOMMUNICATIONS',
    'TELECOMMUNI CATIONS': 'TELECOMMUNICATIONS',
    'SHIPPING AND PORTS': 'PORTS AND SHIPPING', 'PORTS': 'PORTS AND SHIPPING',
    'DEPARTMENT OF HIGHER EDUCATION': 'HIGHER EDUCATION', 'EDUCATION': 'HIGHER EDUCATION',
    'DONER': 'DEVELOPMENT OF NORTH EASTERN REGION',
    'FAMILY WELFARE': 'HEALTH AND FAMILY WELFARE', 'HEALTH': 'HEALTH AND FAMILY WELFARE',
    'DEFENCE': 'DEFENCE PRODUCTION',
    'ROAD TRANSPORT': 'ROAD TRANSPORT AND HIGHWAYS',
    'TRANSPORT HIGHWAYS': 'ROAD TRANSPORT AND HIGHWAYS',
    'ROAD AND HIGHWAYS': 'ROAD TRANSPORT AND HIGHWAYS',
}

MINISTRY_MAP = {
    'ATOMIC ENERGY': 'Department of Atomic Energy',
    'CIVIL AVIATION': 'Ministry of Civil Aviation',
    'COAL': 'Ministry of Coal',
    'TELECOMMUNICATIONS': 'Ministry of Communications',
    'DEFENCE PRODUCTION': 'Dept. of Defence Production (M/o Defence)',
    'DEVELOPMENT OF NORTH EASTERN REGION': 'M/o Development of North Eastern Region',
    'HEALTH AND FAMILY WELFARE': 'Dept. of Health and Family Welfare',
    'FINANCE': 'Ministry of Finance',
    'HIGHER EDUCATION': 'Dept. of Higher Education (M/o Education)',
    'HOUSING': 'Ministry of Housing and Urban Affairs',
    'MINES': 'Ministry of Mines',
    'PETROLEUM': 'M/o Petroleum and Natural Gas',
    'PORTS AND SHIPPING': 'M/o Ports, Shipping and Waterways',
    'POWER': 'Ministry of Power',
    'RAILWAYS': 'Ministry of Railways',
    'RENEWABLE ENERGY': 'M/o New and Renewable Energy',
    'ROAD TRANSPORT AND HIGHWAYS': 'M/o Road Transport and Highways',
    'STEEL': 'Ministry of Steel',
    'URBAN DEVELOPMENT': 'Ministry of Housing and Urban Affairs',
    'WATER RESOURCES': 'Dept. of Water Resources, RD & GR',
    'HOME AFFAIRS': 'Ministry of Home Affairs',
    'SOCIAL JUSTICE': 'M/o Social Justice and Empowerment',
    'DPIIT': 'DPIIT (M/o Commerce and Industry)',
}

STATE_FIX = {
    'Telegana': 'Telangana', 'New Delhi': 'Delhi', 'Orissa': 'Odisha',
    'Pondicherry': 'Puducherry', 'Uttaranchal': 'Uttarakhand',
    'Andaman And Islands': 'Andaman and Nicobar Islands',
    'Andaman Islands': 'Andaman and Nicobar Islands',
    'Jammu And': 'Jammu and Kashmir', 'Jammu And Kashmir And Ladakh': 'Jammu and Kashmir',
    'Not Applicable': 'Multiple States', 'Multiple State': 'Multiple States',
    'Dadra And Nagar Haveli': 'Dadra and Nagar Haveli and Daman and Diu',
    'Nctu Delhi': 'Delhi',
}

SOURCES = {
    '2024-04': ('Flash Report on Central Sector Infrastructure Projects, April 2024',
                'https://www.mospi.gov.in/sites/default/files/publication_reports/FlashReport_April_2024.pdf'),
    '2024-05': ('Flash Report on Central Sector Infrastructure Projects, May 2024',
                'https://www.mospi.gov.in/sites/default/files/publication_reports/FlashReport_May_2024.pdf'),
    '2024-06': ('Quarterly Project Implementation Status Report, Q1 2024-25 (Apr-Jun 2024)',
                'https://www.mospi.gov.in/sites/default/files/publication_reports/QPISR_1st_QTR_2024-25.pdf'),
    '2024-08': ('Flash Report on Central Sector Infrastructure Projects, August 2024',
                'https://www.mospi.gov.in/sites/default/files/publication_reports/FlashReport_August_2024.pdf'),
    '2024-09': ('Flash Report on Central Sector Infrastructure Projects, September 2024',
                'https://www.mospi.gov.in/sites/default/files/publication_reports/FlashReport_September_2024.pdf'),
    '2025-03': ('Quarterly Project Implementation Status Report, Q4 2024-25 (Jan-Mar 2025)',
                'https://www.mospi.gov.in/sites/default/files/publication_reports/QPISR_4th_QTR_2024-25.pdf'),
}

ANNEX_LABEL = {
    'ahead_orig': 'Annexure IV (ahead of schedule)',
    'onsched_orig': 'Annexure V (on schedule)',
    'delayed_orig': 'Annexure VII (delayed, w.r.t. original schedule)',
    'no_doc': 'Annexure IX (without date of commissioning)',
    'no_orig_doc': 'Annexure X (without original date of commissioning)',
    'ongoing_census': 'Table 7 (census of ongoing projects)',
}

REASON_RULES = [
    ('Land acquisition', r'land\s*acquis|land\s*issue|site\s*hand|right\s*of\s*way|row\b'),
    ('Environmental / forest clearance', r'environment|forest|wildlife|clearance'),
    ('Funding / budget constraints', r'fund|budget|financial\s*constraint|non-?availab.*fund|payment'),
    ('Law and order / security', r'law\s*and\s*order|naxal|insurg|security|militan'),
    ('Contractual / tender issues', r'contract|tender|bid|arbitrat|vendor'),
    ('Supply chain / equipment', r'supply|equipment|material|machinery|logistic'),
    ('Rehabilitation & resettlement', r'\br\s*&\s*r\b|rehabilitat|resettlement'),
    ('Litigation / court orders', r'court|litigat|stay\s*order|legal'),
    ('COVID-19 / epidemic', r'covid|pandemic|epidemic|lockdown'),
    ('Weather / geological / natural', r'flood|cyclone|weather|geolog|mining\s*condition|landslide|rain|earth'),
    ('Utility shifting / power supply', r'utility|power\s*supply|shifting\s*of|electric|transmission\s*line'),
    ('Slow progress / contractor performance', r'slow\s*progress|poor\s*progress|slow\s*pace|performance\s*of\s*contractor|pace\s*of\s*work'),
    ('Approvals / pending sanctions', r'approval|sanction|pending\s*clearance|permission|licen'),
    ('Design / scope changes', r'design\s*change|scope\s*change|change\s*in\s*scope|modif|revised\s*scope'),
    ('Rail/level crossing/ROB coordination', r'level\s*crossing|rob\b|railway\s*crossing|siding'),
]


def norm_sector(s):
    if pd.isna(s):
        return None
    t = re.sub(r'\s+', ' ', str(s).upper().strip('. '))
    t = SECTOR_FIX.get(t, t)
    return t or None


def norm_state(s):
    if pd.isna(s) or str(s).strip() in ('', 'nan'):
        return None
    t = re.sub(r'\s+', ' ', str(s).strip())
    t = ' '.join(w.capitalize() if not w.isupper() else w for w in t.split())
    t = t.title()
    t = STATE_FIX.get(t, t)
    # keep 'Andaman and Nicobar Islands' style casing tidy
    t = t.replace(' And ', ' and ')
    return t or None


def categorize_reasons(txt):
    if not txt or pd.isna(txt) or str(txt).strip().lower() in ('nil', 'nan', '-'):
        return None
    cats = []
    low = str(txt).lower()
    for cat, pat in REASON_RULES:
        if re.search(pat, low):
            cats.append(cat)
    return '; '.join(cats) if cats else 'Other reported reasons'


def month_index(ym):
    if not ym or pd.isna(ym):
        return np.nan
    y, m = str(ym).split('-')[:2]
    return int(y) * 12 + int(m) - 1


def months_diff(a, b):
    """b - a in months"""
    ia, ib = month_index(a), month_index(b)
    if np.isnan(ia) or np.isnan(ib):
        return np.nan
    return ib - ia


def load_interim():
    frames = []
    for f in sorted(os.listdir(INTERIM)):
        path = os.path.join(INTERIM, f)
        if f.startswith('flash_') and f.endswith('_ongoing.csv'):
            df = pd.read_csv(path, dtype={'project_code': str})
            df['report_month'] = f.split('_')[1]
            df['report_kind'] = 'flash_census'
            frames.append(df)
        elif f.startswith('qpisr_') and f.endswith('_ongoing.csv'):
            df = pd.read_csv(path, dtype={'project_code': str})
            df['report_kind'] = 'qpisr_census'
            frames.append(df)
    ongoing = pd.concat(frames, ignore_index=True)

    ev_frames = []
    for f in sorted(os.listdir(INTERIM)):
        path = os.path.join(INTERIM, f)
        if f.startswith('flash_') and f.endswith('_closed.csv'):
            df = pd.read_csv(path, dtype={'project_code': str})
            df['source_kind'] = 'flash_closed'
            ev_frames.append(df)
        elif f.startswith('qpisr_') and f.endswith('_completed.csv'):
            df = pd.read_csv(path, dtype={'project_code': str})
            df['source_kind'] = 'report_completed_list'
            ev_frames.append(df)
    events = pd.concat(ev_frames, ignore_index=True) if ev_frames else pd.DataFrame()
    return ongoing, events


def build_panel():
    ongoing, events = load_interim()

    # ---- ongoing census rows -> canonical panel ----
    p = pd.DataFrame()
    p['report_month'] = ongoing['report_month']
    p['project_code'] = ongoing['project_code'].astype(str).str.strip()
    p['project_name'] = ongoing['project_name'].astype(str).str.strip()
    p['sector'] = ongoing['sector'].map(norm_sector)
    p['ministry'] = p['sector'].map(lambda s: MINISTRY_MAP.get(s) if s else None)
    p['state'] = ongoing['state'].map(norm_state)
    p['agency'] = ongoing['agency'].astype(str).str.strip().replace('nan', None)
    p['approval_date'] = ongoing['approval_date']
    p['original_completion_target'] = ongoing['original_doc']
    p['revised_completion_target'] = ongoing['revised_doc']
    p['anticipated_completion_target'] = ongoing['anticipated_doc']
    p['original_cost'] = ongoing['original_cost']
    p['revised_cost'] = ongoing['revised_cost']
    p['anticipated_cost'] = ongoing['anticipated_cost']
    p['cumulative_expenditure'] = ongoing['expenditure']
    p['physical_progress_pct'] = ongoing.get('physical_progress_pct')
    p['schedule_status'] = ongoing['schedule_status']
    p['delay_reasons_reported'] = ongoing.get('delay_reasons')
    p['reported_cost_overrun_pct'] = ongoing.get('cost_overrun_pct')
    p['reported_time_overrun_months'] = ongoing.get('time_overrun_months')

    # latest (current) estimates: anticipated, else revised, else original
    p['latest_cost'] = p['anticipated_cost'].fillna(p['revised_cost']).fillna(p['original_cost'])
    p['latest_completion_target'] = p['anticipated_completion_target'].fillna(
        p['revised_completion_target']).fillna(p['original_completion_target'])

    # ---- derived metrics (documented formulas) ----
    p['cost_overrun_pct'] = np.where(
        p['latest_cost'].notna() & p['original_cost'].notna() & (p['original_cost'] > 0),
        (p['latest_cost'] - p['original_cost']) / p['original_cost'] * 100, np.nan)
    p['expenditure_over_original_pct'] = np.where(
        p['cumulative_expenditure'].notna() & p['original_cost'].notna() & (p['original_cost'] > 0),
        (p['cumulative_expenditure'] - p['original_cost']) / p['original_cost'] * 100, np.nan)
    p['time_overrun_months'] = [
        months_diff(a, b) for a, b in zip(p['original_completion_target'],
                                          p['latest_completion_target'])]
    # MoSPI counts a project as showing cost overrun when its current
    # (revised/anticipated) cost exceeds the original approved cost.
    p['cost_overrun_flag'] = np.where(
        p['cost_overrun_pct'].notna(), (p['cost_overrun_pct'] > 0).astype(float), np.nan)
    p['project_age_months'] = [
        months_diff(a, b) for a, b in zip(p['approval_date'], p['report_month'])]
    p['planned_duration_months'] = [
        months_diff(a, b) for a, b in zip(p['approval_date'],
                                          p['original_completion_target'])]
    p['elapsed_fraction'] = np.where(
        p['project_age_months'].notna() & (p['planned_duration_months'] > 0),
        p['project_age_months'] / p['planned_duration_months'], np.nan)
    p['event'] = 'ongoing_report'
    p['actual_completion_date'] = None
    p['delay_reason_categories'] = p['delay_reasons_reported'].map(categorize_reasons)
    p['source_report'] = p['report_month'].map(
        lambda m: SOURCES.get(m, ('', ''))[0])
    p['source_detail'] = ongoing['schedule_status'].map(ANNEX_LABEL)
    p['source_url'] = p['report_month'].map(lambda m: SOURCES.get(m, ('', ''))[1])
    p['report_kind'] = ongoing['report_kind']

    # ---- dedupe (code, report_month): prefer QPISR rows ----
    p = p.sort_values(['report_month', 'report_kind'],
                      key=lambda s: s.map({'qpisr_census': 0, 'flash_census': 1})
                      if s.name == 'report_kind' else s)
    dup_mask = p.duplicated(subset=['project_code', 'report_month'], keep='first')
    n_dups = int(dup_mask.sum())
    p = p[~dup_mask].copy()

    # ---- closed / completed events ----
    ev_rows = []
    name_lookup = {}
    for _, r in p.iterrows():
        key = re.sub(r'[^A-Z0-9]', '', str(r['project_name']).upper())[:60]
        if key and r['project_code']:
            name_lookup.setdefault(key, set()).add(r['project_code'])

    for _, e in events.iterrows():
        rm = e.get('report_month')
        code = str(e.get('project_code') or '').strip()
        if not code or code in ('', 'nan'):
            key = re.sub(r'[^A-Z0-9]', '', str(e.get('project_name', '')).upper())[:60]
            cands = name_lookup.get(key, set())
            if len(cands) == 1:
                code = next(iter(cands))
            else:
                code = f"UNM-{rm}-{len(ev_rows)}"
        is_flash = e.get('source_kind') == 'flash_closed'
        if is_flash:
            # flash TABLE 13: anticipated DOC at closure ~ completion when it is
            # in the past; otherwise the project left the shelf unfinished
            antic = e.get('anticipated_doc')
            if str(e.get('event')) == 'completed':
                event, completion = 'completed', antic or rm
            else:
                event, completion = str(e.get('event')), None
            if event == 'completed' and antic and month_index(antic) > month_index(rm):
                event, completion = 'closed_unfinished', None
        else:
            event, completion = 'completed', e.get('completion_month') or rm
        ev_rows.append({
            'report_month': rm, 'project_code': code,
            'project_name': str(e.get('project_name') or '').strip(),
            'sector': norm_sector(e.get('sector')),
            'ministry': None, 'state': norm_state(e.get('state')),
            'agency': e.get('agency') if pd.notna(e.get('agency')) else None,
            'approval_date': e.get('approval_date'),
            'original_completion_target': e.get('original_doc'),
            'revised_completion_target': None,
            'anticipated_completion_target': e.get('anticipated_doc'),
            'original_cost': e.get('original_cost'),
            'revised_cost': None,
            'anticipated_cost': e.get('anticipated_cost'),
            'cumulative_expenditure': e.get('expenditure'),
            'physical_progress_pct': None,
            'schedule_status': 'closed',
            'latest_cost': (e.get('anticipated_cost') if pd.notna(e.get('anticipated_cost'))
                            else e.get('original_cost')),
            'latest_completion_target': e.get('anticipated_doc') or e.get('original_doc'),
            'delay_reasons_reported': None,
            'reported_cost_overrun_pct': None,
            'reported_time_overrun_months': None,
            'event': event,
            'actual_completion_date': completion,
            'project_age_months': months_diff(e.get('approval_date'), completion),
            'planned_duration_months': months_diff(e.get('approval_date'), e.get('original_doc')),
            'report_kind': 'event',
            'source_report': SOURCES.get(rm, ('', ''))[0],
            'source_detail': 'Flash Report TABLE-13 (closed during month)' if is_flash
                             else 'QPISR Table 3 (completed during quarter)',
            'source_url': SOURCES.get(rm, ('', ''))[1],
        })
    ev_df = pd.DataFrame(ev_rows)
    if len(ev_df):
        ev_df['cost_overrun_pct'] = np.where(
            ev_df['latest_cost'].notna() & ev_df['original_cost'].notna() &
            (ev_df['original_cost'] > 0),
            (ev_df['latest_cost'] - ev_df['original_cost']) / ev_df['original_cost'] * 100,
            np.nan)
        ev_df['expenditure_over_original_pct'] = np.where(
            ev_df['cumulative_expenditure'].notna() & ev_df['original_cost'].notna() &
            (ev_df['original_cost'] > 0),
            (ev_df['cumulative_expenditure'] - ev_df['original_cost']) /
            ev_df['original_cost'] * 100, np.nan)
        ev_df['time_overrun_months'] = [
            months_diff(a, b) for a, b in zip(ev_df['original_completion_target'],
                                              ev_df['actual_completion_date'])]
        ev_df['cost_overrun_flag'] = np.where(
            ev_df['cost_overrun_pct'].notna(),
            (ev_df['cost_overrun_pct'] > 0).astype(float), np.nan)
        ev_df['elapsed_fraction'] = np.nan
        ev_df['delay_reason_categories'] = None
        panel = pd.concat([p, ev_df], ignore_index=True)
    else:
        panel = p

    # fill ministry for event rows from sector
    panel['ministry'] = panel.apply(
        lambda r: MINISTRY_MAP.get(r['sector']) if pd.notna(r.get('sector')) else r.get('ministry'),
        axis=1)

    panel = panel.sort_values(['project_code', 'report_month']).reset_index(drop=True)
    return panel, n_dups


def validate(panel):
    checks = {}
    v = panel.dropna(subset=['cost_overrun_pct', 'reported_cost_overrun_pct'])
    v = v[v['reported_cost_overrun_pct'].notna()]
    if len(v):
        diff = (v['cost_overrun_pct'] - v['reported_cost_overrun_pct']).abs()
        checks['computed_vs_reported_cost_overrun'] = {
            'n': int(len(v)), 'match_within_1pp_pct': round(float((diff <= 1).mean() * 100), 1)}
    v2 = panel.dropna(subset=['time_overrun_months', 'reported_time_overrun_months'])
    if len(v2):
        diff2 = (v2['time_overrun_months'] - v2['reported_time_overrun_months']).abs()
        checks['computed_vs_reported_time_overrun'] = {
            'n': int(len(v2)), 'match_within_1m_pct': round(float((diff2 <= 1).mean() * 100), 1)}
    # cross-month consistency of original cost for the same project
    g = panel[panel['event'] == 'ongoing_report'].dropna(subset=['original_cost']) \
        .groupby('project_code')['original_cost'].nunique()
    checks['original_cost_consistent_across_months'] = {
        'projects': int(len(g)),
        'consistent_pct': round(float((g == 1).mean() * 100), 1)}
    checks['report_months'] = sorted(panel['report_month'].unique().tolist())
    checks['rows'] = int(len(panel))
    checks['unique_projects'] = int(panel['project_code'].nunique())
    return checks


def main():
    panel, n_dups = build_panel()
    panel.to_csv(os.path.join(OUTDIR, 'paimana_panel.csv'), index=False)
    checks = validate(panel)
    meta = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'sources': {m: {'report': t[0], 'url': t[1]} for m, t in SOURCES.items()},
        'notes': [
            'All rows are parsed from MoSPI-published PDF reports (Flash Reports and QPISR).',
            'cost_overrun_pct = (latest_cost - original_cost)/original_cost*100, where '
            'latest_cost = anticipated cost if reported, else revised, else original.',
            'time_overrun_months = latest_completion_target - original_completion_target '
            'in months (actual completion date for completed projects).',
            'cost_overrun_flag = 1 when latest cost exceeds original approved cost '
            '(consistent with MoSPI flash-report counting).',
            'Physical progress % is published only in QPISR quarterly reports, not in '
            'monthly flash reports; it is missing for flash-only months.',
            'Flash TABLE-13 closed projects carry no project codes in the source; they are '
            'matched to coded panel rows by normalized project name where unique.',
            'For flash TABLE-13 rows, completion month is approximated by the reported '
            '"now anticipated" date of commissioning when it is in the past, else the '
            'project is marked closed_unfinished.',
            'Ministry is derived from sector via a documented GoI administrative mapping '
            '(shown in the app Data Quality page).',
            f'{n_dups} duplicate (project, month) rows were dropped, keeping QPISR first.',
        ],
        'validation': checks,
    }
    with open(os.path.join(OUTDIR, 'source_manifest.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(checks, indent=2))
    print(f"\npanel rows={len(panel)}, projects={panel['project_code'].nunique()}")
    print(panel.groupby(['report_month', 'event']).size().unstack(fill_value=0))


if __name__ == '__main__':
    main()
