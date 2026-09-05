"""
Parse MoSPI monthly Flash Report PDFs (2024 layout).

Extracts, per report month:
- Ongoing-project census from Annexures IV (ahead of schedule), V (on schedule),
  VII (delayed w.r.t. original schedule, incl. agency-reported reasons for delay),
  IX (without date of commissioning), X (without original DOC).
- Completed/dropped/frozen events from TABLE-13 (no codes in source; matched later).

Column positions differ per annexure, so x-column clusters are calibrated per
annexure from the data tokens themselves. Only published MoSPI data is read.
"""
import json
import os
import re
import sys
import tempfile
import warnings

import pdfplumber
from pypdf import PdfReader, PdfWriter

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from common import FLASH_SECTORS, STATES, parse_month_year, parse_number, norm_text

DATE_RE = re.compile(r'^\d{1,2}/\d{4}$')
NUM_RE = re.compile(r'^[\d,]+(?:\.\d+)?$')
INT_RE = re.compile(r'^\d{1,4}$')
CODE_RE = re.compile(r'\[([A-Z]{0,2}\d{5,10})\]?')

SECTOR_SET = set(FLASH_SECTORS)
STATE_SET = set(STATES)

# lines that are page/column headers or noise (never project data)
HEADER_PAT = re.compile(
    r'(Reasons for Delay|As Reported|Implementing Agenc|Date of|Commissioning|'
    r'Original/|Anticip|Cumulative|Expenditure|Approval|Overrun|Crore|Month/Year|'
    r'Central Sector|Costing Rs|Details of|Annexure|^S\.?\s?No|Table|Grand Total|'
    r'SI\.?No|Now anticipated|Final|Project\s*$|DOA|DOC)', re.I)
COLNUM_PAT = re.compile(r'^(\d{1,2}\s+){3,}\d{1,2}\s*$')
PAGENUM_PAT = re.compile(r'^\d{1,4}$')
MONTH_SECTION_PAT = re.compile(
    r'^((?:January|February|March|April|May|June|July|August|September|October|'
    r'November|December),?\s*\d{4})$', re.I)


def page_texts(pdf_path):
    r = PdfReader(pdf_path)
    return [(p.extract_text() or '') for p in r.pages]


def find_pages(texts, pattern, min_page=0):
    return [i for i, t in enumerate(texts)
            if i >= min_page and re.search(pattern, t, re.I) and '…' not in t]


def slice_pdf_pages(pdf_path, pages, tmpdir):
    r = PdfReader(pdf_path)
    w = PdfWriter()
    for i in pages:
        if 0 <= i < len(r.pages):
            w.add_page(r.pages[i])
    out = os.path.join(tmpdir, f"slice_{abs(hash(tuple(pages))) % 10**8}.pdf")
    with open(out, 'wb') as f:
        w.write(f)
    return out


def get_lines(pdf_path, pages):
    """pdfplumber words grouped into visual lines, for the given pages."""
    all_lines = []
    with tempfile.TemporaryDirectory() as tmpdir:
        CHUNK = 30
        for st in range(0, len(pages), CHUNK):
            chunk = pages[st:st + CHUNK]
            sp = slice_pdf_pages(pdf_path, chunk, tmpdir)
            with pdfplumber.open(sp) as pdf:
                for p in pdf.pages:
                    words = p.extract_words(keep_blank_chars=False)
                    lines = []
                    for w in sorted(words, key=lambda w: (w['top'], w['x0'])):
                        placed = False
                        for ln in lines:
                            if abs(ln['top'] - w['top']) <= 2.5:
                                ln['words'].append(w)
                                placed = True
                                break
                        if not placed:
                            lines.append({'top': w['top'], 'words': [w]})
                    for ln in lines:
                        ln['words'].sort(key=lambda w: w['x0'])
                        ln['text'] = ' '.join(w['text'] for w in ln['words'])
                    all_lines.extend(lines)
    return all_lines


def is_sector_header(text):
    t = norm_text(text).upper().rstrip(' .0123456789')
    return t in SECTOR_SET and len(t) > 3


def parse_label(txt):
    """'[N04000074]AAI,TAMIL NADU' -> (code, agency, state)"""
    m = CODE_RE.search(txt)
    if not m:
        return None
    code = m.group(1)
    after = txt[m.end():].strip(' ,-')
    comma = after.find(',')
    if comma >= 0:
        agency = after[:comma].strip()
        state = after[comma + 1:].strip()
    else:
        agency, state = after, ''
    # state may run into trailing data printed on the same line: keep only
    # leading alphabetic tokens and strip ' - ...' tails
    state = re.split(r'\s+-\s+', state)[0].strip()
    words = []
    for w in state.split():
        if NUM_RE.match(w) or DATE_RE.match(w) or re.match(r'^[\d.,]+$', w):
            break
        words.append(w)
    state = ' '.join(words)
    agency = re.split(r'\s+-\s+', agency)[0].strip()
    return code, norm_text(agency), norm_text(state)


def complete_state(state, next_line_text):
    """If state is a partial prefix of a known state, extend with next-line words."""
    if not state:
        return state, False
    up = state.upper()
    candidates = [s for s in STATE_SET if s.startswith(up) and s != up]
    if not candidates:
        return state, False
    rest = norm_text(next_line_text or '').upper()
    for c in sorted(candidates, key=len, reverse=True):
        remainder = c[len(up):].strip()
        if remainder and rest.startswith(remainder[:min(len(remainder), 12)]):
            return c.title().replace(' And ', ' and ').replace('NADU', 'Nadu'), True
    return state, False


def cluster_1d(xs, gap=16, min_count=1, max_spread=35):
    """Cluster x positions; drop clusters with fewer than min_count members or
    wider than max_spread (phantom columns from stray numerals inside project
    names chain together into broad clusters; real table columns are tight)."""
    if not xs:
        return []
    xs = sorted(xs)
    cl = [[xs[0]]]
    for x in xs[1:]:
        if x - cl[-1][-1] <= gap:
            cl[-1].append(x)
        else:
            cl.append([x])
    return [sum(c) / len(c) for c in cl
            if len(c) >= min_count and (c[-1] - c[0]) <= max_spread]


def nearest_idx(x, centers, tol=25):
    best, bd = None, 1e9
    for i, c in enumerate(centers):
        d = abs(x - c)
        if d < bd:
            best, bd = i, d
    return best if bd <= tol else None


def parse_census(pdf_path, pages, annex_key):
    """Parse one ongoing-projects annexure (IV/V/VII/IX/X)."""
    if not pages:
        return []
    lines = get_lines(pdf_path, pages)

    # --- state machine: blocks of lines, one block per project ---
    blocks = []
    sector = None
    open_block = None

    def close():
        nonlocal open_block
        open_block = None

    for ln in lines:
        t = ln['text'].strip()
        if not t or PAGENUM_PAT.match(t) or COLNUM_PAT.match(t) or HEADER_PAT.search(t):
            continue
        if is_sector_header(t):
            close()
            sector = norm_text(t).upper().rstrip(". ")
            continue
        if re.match(r'^(Grand\s+)?Total\b', t, re.I):
            close()
            continue
        words = ln['words']
        first = words[0]
        has_label = bool(CODE_RE.search(t))
        if has_label and (open_block is None or open_block.get('label')):
            # new project that starts with its label line (page-break case)
            close()
            open_block = {'sector': sector, 'sno': None, 'lines': [ln], 'label': ln}
            blocks.append(open_block)
            continue
        if INT_RE.match(first['text']) and first['x0'] < 110 and len(words) > 2:
            # S.No line starts a new project
            close()
            open_block = {'sector': sector, 'sno': int(first['text']),
                          'lines': [ln], 'label': None}
            blocks.append(open_block)
            continue
        if open_block is not None:
            if has_label and open_block.get('label') is None:
                open_block['label'] = ln
            open_block['lines'].append(ln)

    # --- calibrate column clusters from data tokens across the annexure ---
    # only clusters appearing in many rows are real columns; stray numerals
    # inside project names form small phantom clusters that get filtered out.
    date_xs, num_xs = [], []
    for b in blocks:
        for ln in b['lines']:
            for w in ln['words']:
                if DATE_RE.match(w['text']) and w['x0'] > 150:
                    date_xs.append(w['x0'])
                elif NUM_RE.match(w['text']) and w['x0'] > 150:
                    num_xs.append(w['x0'])
    min_ct = max(15, int(0.2 * max(1, len(blocks))))
    date_centers = cluster_1d(date_xs, min_count=min_ct)
    num_centers = cluster_1d(num_xs, min_count=min_ct)
    if not date_centers:
        return []
    approval_center = date_centers[0]           # DOA is always the leftmost date column
    doc_centers = [c for c in date_centers[1:] if c > approval_center + 20]
    # number columns left of the first doc column are costs; right are TOR/COR etc.
    first_doc_x = doc_centers[0] if doc_centers else 10 ** 9
    # cost columns always sit RIGHT of the approval-date column (project names,
    # whose stray numerals form phantom clusters, end left of it)
    cost_centers = [c for c in num_centers
                    if approval_center + 15 < c < first_doc_x - 10]
    tail_centers = [c for c in num_centers if c >= first_doc_x - 10]

    rows = []
    for b in blocks:
        label_ln = b.get('label')
        if label_ln is None:
            continue
        parsed = parse_label(label_ln['text'])
        if parsed is None:
            continue
        code, agency, state = parsed
        state = re.split(r'\s+-\s+', state)[0].strip()
        # state wrap: the line after the label may complete a partial state name
        idx = b['lines'].index(label_ln)
        if idx + 1 < len(b['lines']):
            state2, extended = complete_state(state.upper(), b['lines'][idx + 1]['text'])
            if extended:
                state = state2

        def ym(v):
            return f"{v[0]}-{v[1]:02d}" if v else None

        toks = []
        for ln in b['lines']:
            for w in ln['words']:
                toks.append({'t': w['text'], 'x': w['x0'], 'top': ln['top']})
        # data tokens only (exclude name region x<165 and sno)
        data = [tk for tk in toks
                if tk['x'] > 165 and (DATE_RE.match(tk['t']) or NUM_RE.match(tk['t']))]

        rec = {
            'project_code': code, 'agency': agency, 'state': state,
            'sector': b['sector'], 'schedule_status': annex_key,
            'approval_date': None, 'original_cost': None, 'revised_cost': None,
            'anticipated_cost': None, 'expenditure': None,
            'original_doc': None, 'revised_doc': None, 'anticipated_doc': None,
            'time_overrun_months': None, 'cost_overrun_pct': None,
            'delay_reasons': '',
        }
        # approval date
        for tk in data:
            if DATE_RE.match(tk['t']) and abs(tk['x'] - approval_center) <= 25:
                rec['approval_date'] = ym(parse_month_year(tk['t']))
                break
        # doc columns: rightmost doc cluster = anticipated; left clusters = orig/revised
        doc_seen = {ci: [] for ci in range(len(doc_centers))}
        for tk in data:
            if not DATE_RE.match(tk['t']):
                continue
            if abs(tk['x'] - approval_center) <= 25:
                continue
            best, bd = None, 1e9
            for ci, c in enumerate(doc_centers):
                if abs(tk['x'] - c) < bd:
                    best, bd = ci, abs(tk['x'] - c)
            if best is not None and bd <= 22:
                doc_seen[best].append(tk)
        for ci in range(len(doc_centers) - 1):        # orig/revised clusters
            for tk in doc_seen[ci]:
                if rec['original_doc'] is None:
                    rec['original_doc'] = ym(parse_month_year(tk['t']))
                elif rec['revised_doc'] is None:
                    rec['revised_doc'] = ym(parse_month_year(tk['t']))
        if doc_centers:
            for tk in doc_seen[len(doc_centers) - 1]:  # anticipated cluster
                if rec['anticipated_doc'] is None:
                    rec['anticipated_doc'] = ym(parse_month_year(tk['t']))
        # cost columns: cost_centers[0] = orig/revised, [1] = anticipated, [2] = expenditure
        for ci, c in enumerate(cost_centers[:3]):
            vals = [tk for tk in data if NUM_RE.match(tk['t']) and abs(tk['x'] - c) <= 22]
            for tk in vals:
                v = parse_number(tk['t'])
                if ci == 0:
                    if rec['original_cost'] is None:
                        rec['original_cost'] = v
                    elif rec['revised_cost'] is None:
                        rec['revised_cost'] = v
                elif ci == 1:
                    if rec['anticipated_cost'] is None:
                        rec['anticipated_cost'] = v
                elif ci == 2:
                    if rec['expenditure'] is None:
                        rec['expenditure'] = v
        # tail columns (TOR, COR) by x order
        for ci, c in enumerate(tail_centers[:2]):
            vals = [tk for tk in data if NUM_RE.match(tk['t']) and abs(tk['x'] - c) <= 22]
            v = parse_number(vals[0]['t']) if vals else None
            if ci == 0 and v is not None and rec['time_overrun_months'] is None:
                rec['time_overrun_months'] = v
            elif ci == 1 and v is not None and rec['cost_overrun_pct'] is None:
                rec['cost_overrun_pct'] = v
        # reasons: words right of the last tail/doc column
        reason_x = max([c for c in tail_centers] + [c for c in doc_centers] + [400]) + 25
        parts = []
        for ln in b['lines']:
            for w in ln['words']:
                if w['x0'] >= reason_x and w['text'] not in ('Nil', 'NIL', '-', ')', '('):
                    parts.append(w['text'])
                elif parts and 165 <= w['x0'] < reason_x and w['text'] not in ('-',) \
                        and not DATE_RE.match(w['text']) and not NUM_RE.match(w['text']) \
                        and not CODE_RE.search(w['text']) and w['x0'] > 300:
                    parts.append(w['text'])
        rec['delay_reasons'] = norm_text(' '.join(parts))[:400]
        # name: words in the name region from lines BEFORE the label line
        name_parts = []
        for ln in b['lines']:
            if ln is label_ln or (b['lines'].index(ln) > idx):
                continue
            for w in ln['words']:
                if 40 <= w['x0'] < 175 and w['text'] != '-' \
                        and not DATE_RE.match(w['text']) and not NUM_RE.match(w['text']):
                    name_parts.append(w['text'])
        # name can also sit on the label line itself (before the code bracket)
        pre = label_ln['text'][:CODE_RE.search(label_ln['text']).start()]
        name_parts.extend([w for w in pre.split() if w not in ('-',)])
        rec['project_name'] = norm_text(' '.join(name_parts))[:300]
        rows.append(rec)
    return rows


def parse_table13(pdf_path, pages, report_month):
    """TABLE-13: projects completed/dropped/frozen during the month.
    Columns: SI.No, Project, DOA, Original Cost, Original DOC,
    Now anticipated Cost, Now anticipated DOC, Final Expenditure. No codes."""
    rows = []
    if not pages:
        return rows
    lines = get_lines(pdf_path, pages)
    sector = None
    cur = None
    for ln in lines:
        t = ln['text'].strip()
        if not t or PAGENUM_PAT.match(t) or COLNUM_PAT.match(t) or HEADER_PAT.search(t):
            continue
        if is_sector_header(t):
            sector = norm_text(t).upper().rstrip(". ")
            cur = None
            continue
        words = ln['words']
        first = words[0]
        if INT_RE.match(first['text']) and first['x0'] < 110 and len(words) > 2:
            cur = {'sector': sector, 'lines': [ln], 'name_parts': []}
            rows.append(cur)
            continue
        if cur is not None:
            cur['lines'].append(ln)
    out = []
    for b in rows:
        toks = [w for ln in b['lines'] for w in ln['words'] if w['x0'] > 150]
        dates = [w for w in toks if DATE_RE.match(w['text'])]
        nums = [w for w in toks if NUM_RE.match(w['text'])]
        name_words = []
        for ln in b['lines']:
            for w in ln['words']:
                if 40 < w['x0'] < 160 and w['text'] != '-' and not DATE_RE.match(w['text']):
                    name_words.append(w['text'])

        def ym(v):
            return f"{v[0]}-{v[1]:02d}" if v else None

        # default event = completed; sections detected below override
        out.append({
            'project_code': '', 'project_name': norm_text(' '.join(name_words))[:300],
            'sector': b['sector'],
            'approval_date': ym(parse_month_year(dates[0]['text'])) if len(dates) > 0 else None,
            'original_cost': parse_number(nums[0]['text']) if len(nums) > 0 else None,
            'original_doc': ym(parse_month_year(dates[1]['text'])) if len(dates) > 1 else None,
            'anticipated_cost': parse_number(nums[1]['text']) if len(nums) > 1 else None,
            'anticipated_doc': ym(parse_month_year(dates[2]['text'])) if len(dates) > 2 else None,
            'expenditure': parse_number(nums[-1]['text']) if len(nums) > 2 else None,
            'event': 'completed', 'report_month': report_month,
        })
    return out


def parse_flash(pdf_path, report_month):
    texts = page_texts(pdf_path)
    n = len(texts)

    # ---- locate all annexure start pages (skip TOC pages with dots) ----
    starts = {}
    for num in ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI',
                'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII']:
        pat = rf'Annexure\s*[-–]?\s*{num}\b'
        pgs = find_pages(texts, pat, min_page=10)
        if pgs:
            starts[num] = pgs[0]
    order = sorted(starts.items(), key=lambda kv: kv[1])

    def annex_range(num):
        if num not in starts:
            return []
        st = starts[num]
        later = [p for _, p in order if p > st]
        en = min(later) - 1 if later else n - 1
        # annexure content can't run past the last page
        return list(range(st, en + 1))

    ongoing = []
    for key, num in [('ahead_orig', 'IV'), ('onsched_orig', 'V'), ('delayed_orig', 'VII'),
                     ('no_doc', 'IX'), ('no_orig_doc', 'X')]:
        rng = annex_range(num)
        rows = parse_census(pdf_path, rng, key)
        ongoing.extend(rows)

    # ---- TABLE-13 closed events ----
    t13 = find_pages(texts, r'TABLE-?13\b', min_page=8)
    t13_pages = []
    if t13:
        st = t13[0]
        t13_pages = [st]
        for i in range(st + 1, min(st + 10, n)):
            if re.search(r'TABLE-?14\b', texts[i]):
                break
            t13_pages.append(i)
    closed = parse_table13(pdf_path, t13_pages, report_month)

    summary = {'annexure_starts': starts}
    return ongoing, closed, summary


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--pdf', required=True)
    ap.add_argument('--month', required=True)
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()

    ongoing, closed, summary = parse_flash(args.pdf, args.month)
    os.makedirs(args.outdir, exist_ok=True)
    import csv
    with open(f'{args.outdir}/flash_{args.month}_ongoing.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(ongoing[0].keys()) if ongoing else ['empty'])
        w.writeheader()
        w.writerows(ongoing)
    with open(f'{args.outdir}/flash_{args.month}_closed.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(closed[0].keys()) if closed else ['empty'])
        w.writeheader()
        w.writerows(closed)
    with open(f'{args.outdir}/flash_{args.month}_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"{args.month}: ongoing={len(ongoing)} closed={len(closed)}")


if __name__ == '__main__':
    main()
