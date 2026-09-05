"""SQLite access layer. The canonical panel CSV is imported on pipeline run."""
import os
import sqlite3

import pandas as pd

from .config import DB_PATH

PANEL_COLUMNS = [
    'report_month', 'project_code', 'project_name', 'sector', 'ministry', 'state',
    'agency', 'approval_date', 'original_completion_target',
    'revised_completion_target', 'anticipated_completion_target',
    'actual_completion_date', 'original_cost', 'revised_cost', 'anticipated_cost',
    'latest_cost', 'latest_completion_target', 'cumulative_expenditure',
    'physical_progress_pct', 'cost_overrun_pct', 'expenditure_over_original_pct',
    'time_overrun_months', 'cost_overrun_flag', 'project_age_months',
    'planned_duration_months', 'elapsed_fraction', 'schedule_status', 'event',
    'delay_reasons_reported', 'delay_reason_categories',
    'reported_cost_overrun_pct', 'reported_time_overrun_months',
    'source_report', 'source_detail', 'source_url', 'report_kind',
]


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(panel: pd.DataFrame):
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    conn = get_conn()
    panel.to_sql('panel', conn, if_exists='replace', index=False)
    conn.execute("""
        CREATE TABLE project_scores AS
        SELECT * FROM panel WHERE 0
    """)
    conn.execute("""
        CREATE TABLE warnings (
            project_code TEXT, report_month TEXT, warning_type TEXT,
            severity TEXT, what TEXT, reason TEXT, action TEXT
        )
    """)
    conn.commit()
    conn.close()


def query_df(sql, params=()):
    conn = get_conn()
    try:
        return pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()


def execute(sql, params=()):
    conn = get_conn()
    try:
        conn.execute(sql, params)
        conn.commit()
    finally:
        conn.close()
