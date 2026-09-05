"""
Parse MoSPI QPISR (Quarterly Project Implementation Status Report) PDFs, 2024-25
layout.

Extracts:
- Table 7  : full census of ongoing projects (state, sector, agency, code, name,
             approval date, original/revised/anticipated cost and completion
             dates, cumulative expenditure, physical progress %).
- Table 3  : projects completed during the quarter.

The 2024-25 QPISR layout marks revised values in parentheses (x) and
anticipated values in braces {x}, which makes column assignment reliable.
"""
import csv
import os
import re
import sys
import warnings

import pdfplumber
from pypdf import PdfReader

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from common import parse_month_year, parse_number, norm_text

CODE_LINE = re.compile(r'^\(\s*([A-Z]{0,2}\d{5,10})\s*\)$')
CODE_ANYWHERE = re.compile(r'\(\s*([A-Z]{0,2}\d{5,10})\s*\)')
AGENCY_LINE = re.compile(r'^\(\s*[A-Z][A-Za-z0-9 .&/()\-]{1,80}\)$')
HEADER_PAT = re.compile(
    r'(MOSPI_|Table:-|State\s+Sector|Project\s+Name|Date\s+of|Commissioning|'
    r'Approval|Expenditure|Progress|Central Sector|Sl\.?\s*No|Rs\.?\s*Crore|'
    r'Anticipated|Original|Cumulative|AGENCY_NAME)', re.I)
DATEISH = re.compile(r'^\d{1,2}[-/]\d{4}$|^[A-Za-z]{3}[- ]\d{2,4}$')


def get_pages(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        return pdf.pages


def page_lines(page):
    words = page.extract_words(keep_blank_chars=False)
    lines = {}
    for w in sorted(words, key=lambda w: (w['top'], w['x0'])):
        key = round(w['top'] / 3)
        if key not in lines:
            lines[key] = {'top': w['top'], 'words': []}
        lines[key]['words'].append(w)
    out = []
    for key in sorted(lines):
        ln = lines[key]
        ln['words'].sort(key=lambda w: w['x0'])
        ln['text'] = ' '.join(w['text'] for w in ln['words'])
        out.append(ln)
    return out


def clean_paren(tok):
    """'(May-23)' / '{6/2023}' / '(N.A.)' / '{N.A.}' -> inner text or None."""
    t = tok.strip()
    if t.startswith('(') and t.endswith(')'):
        inner = t[1:-1].strip()
    elif t.startswith('{') and t.endswith('}'):
        inner = t[1:-1].strip()
    else:
        inner = t
    if inner in ('N.A.', 'NA', 'N.A', '-', ''):
        return None, None
    return inner, t[0] if t[0] in '({' else None


def calibrate_bands(pdf_path, pages_idx):
    """Anchor column x-bands to this document's own header words so that
    layout shifts between reports don't misassign columns."""
    anchors = {}
    with pdfplumber.open(pdf_path) as pdf:
        for pi in pages_idx[:5]:
            if pi >= len(pdf.pages):
                continue
            for w in pdf.pages[pi].extract_words(keep_blank_chars=False):
                t = w['text'].strip('():')
                if t in ('Approval', 'Commissioning', 'Cost', 'Expenditure', 'Progress') \
                        and t not in anchors:
                    anchors[t] = w['x0']
            if len(anchors) == 5:
                break
    if len(anchors) < 5:
        return {'approval': (293, 350), 'doc': (355, 428), 'cost': (428, 497),
                'exp': (497, 556), 'prog': (556, 625)}  # QPISR 2024-25 defaults
    a, c, k, e, p = (anchors['Approval'], anchors['Commissioning'], anchors['Cost'],
                     anchors['Expenditure'], anchors['Progress'])
    doc_end = (c + k) / 2
    return {
        'approval': (a - 40, doc_end - 78 if doc_end - 78 > a + 30 else a + 45),
        'doc': (doc_end - 78, doc_end),
        'cost': (doc_end, (k + e) / 2),
        'exp': ((k + e) / 2, (e + p) / 2),
        'prog': ((e + p) / 2, p + 90),
    }


def parse_table7(pdf_path, pages_idx, report_month):
    rows = []
    state = sector = None
    open_block = None
    bands = calibrate_bands(pdf_path, pages_idx)
    with pdfplumber.open(pdf_path) as pdf:
        for pi in pages_idx:
            if pi >= len(pdf.pages):
                break
            for ln in page_lines(pdf.pages[pi]):
                t = ln['text'].strip()
                if not t or HEADER_PAT.search(t):
                    continue
                if re.match(r'^\d{1,3}$', t):
                    continue  # page number
                m = CODE_LINE.match(t)
                if m and open_block is not None:
                    open_block['code'] = m.group(1)
                    rows.append(open_block)
                    open_block = None
                    continue
                if open_block is not None:
                    open_block['lines'].append(ln)
                    continue
                # start of a new block: a line that has either state/sector
                # columns or a Sl.No at x~135-155 followed by name at x~156
                has_sno = any(130 <= w['x0'] < 158 and re.match(r'^\d{1,4}$', w['text'])
                              for w in ln['words'])
                has_name = any(w['x0'] >= 156 for w in ln['words'])
                if has_sno and has_name:
                    open_block = {'lines': [ln], 'code': None, 'page': pi + 1}
                elif not has_sno and any(w['x0'] < 75 for w in ln['words']) and \
                        all(w['x0'] < 135 for w in ln['words']):
                    # continuation line with only state/sector words (rare)
                    pass
    out = []
    for b in rows:
        if not b['code']:
            continue
        # state words: x < 75 ; sector words: 75 <= x < 135
        st_words, sec_words, name_words = [], [], []
        for ln in b['lines']:
            txt = ln['text'].strip()
            am = AGENCY_LINE.match(txt)
            cm = CODE_LINE.match(txt)
            for w in ln['words']:
                x, tok = w['x0'], w['text']
                if x < 75 and re.match(r'^[A-Z.&()\-]+$', tok):
                    st_words.append(tok)
                elif 75 <= x < 135 and re.match(r'^[A-Z.&()\-]+$', tok):
                    sec_words.append(tok)
                elif 156 <= x < 300 and not am and not cm:
                    name_words.append(tok)
        if st_words:
            state = norm_text(' '.join(st_words)).title()
        if sec_words:
            sector = norm_text(' '.join(sec_words)).upper()

        approval = orig_doc = rev_doc = antic_doc = None
        orig_cost = rev_cost = antic_cost = expenditure = progress = None

        def split_values(seg):
            """Split a band segment into ()-revised, {}-anticipated and plain
            values (some PDFs split tokens into single characters)."""
            vals = []
            cur, kind = '', 'plain'
            for ch in seg:
                if ch == '(':
                    if cur.strip():
                        vals.append((kind, cur.strip()))
                    cur, kind = '', 'paren'
                elif ch == '{':
                    if cur.strip():
                        vals.append((kind, cur.strip()))
                    cur, kind = '', 'brace'
                elif ch in (')', '}'):
                    if cur.strip():
                        vals.append((kind, cur.strip()))
                    cur, kind = '', 'plain'
                else:
                    cur += ch
            if cur.strip():
                vals.append((kind, cur.strip()))
            return vals

        for ln in b['lines']:
            for band, (lo, hi) in bands.items():
                seg_words = [w for w in ln['words'] if lo <= w['x0'] < hi]
                if not seg_words:
                    continue
                seg = ''.join(w['text'] for w in seg_words)
                for kind, val in split_values(seg):
                    if band == 'approval' and approval is None:
                        approval = parse_month_year(val)
                    elif band == 'doc':
                        if kind == 'paren' and rev_doc is None:
                            v = parse_month_year(val)
                            rev_doc = rev_doc or v
                        elif kind == 'brace' and antic_doc is None:
                            v = parse_month_year(val)
                            antic_doc = antic_doc or v
                        elif kind == 'plain' and orig_doc is None:
                            orig_doc = parse_month_year(val)
                    elif band == 'cost':
                        if kind == 'paren' and rev_cost is None:
                            rev_cost = parse_number(val)
                        elif kind == 'brace' and antic_cost is None:
                            antic_cost = parse_number(val)
                        elif kind == 'plain' and orig_cost is None:
                            orig_cost = parse_number(val)
                    elif band == 'exp' and expenditure is None:
                        expenditure = parse_number(val)
                    elif band == 'prog' and progress is None:
                        progress = parse_number(val)

        agency = None
        for ln in b['lines']:
            am = AGENCY_LINE.match(ln['text'].strip())
            if am:
                agency = norm_text(ln['text'].strip()[1:-1])
                break

        def ym(v):
            return f"{v[0]}-{v[1]:02d}" if v else None

        out.append({
            'project_code': b['code'], 'project_name': norm_text(' '.join(name_words))[:300],
            'sector': sector, 'state': state, 'agency': agency,
            'approval_date': ym(approval),
            'original_cost': orig_cost, 'revised_cost': rev_cost,
            'anticipated_cost': antic_cost, 'expenditure': expenditure,
            'original_doc': ym(orig_doc), 'revised_doc': ym(rev_doc),
            'anticipated_doc': ym(antic_doc), 'physical_progress_pct': progress,
            'schedule_status': 'ongoing_census',
            'report_month': report_month,
        })
    return out


def parse_table3(pdf_path, pages_idx, report_month, completion_month, event='completed'):
    """Completed-during-quarter list. Two source layouts:
    Q1-2024-25: 'NAME... cost doc expenditure' / (agency) / (CODE) / (STATE)
    Q4-2024-25: name / (agency) / 'sno (CODE)' / (STATE)
    Completion month approximated to the quarter end (documented)."""
    from common import STATES
    state_set = {s.upper() for s in STATES}
    rows = []
    sector = None
    sector_buf = []
    open_block = None

    def flush_sector():
        nonlocal sector, sector_buf
        if sector_buf:
            sector = norm_text(' '.join(sector_buf)).upper()
            sector_buf = []

    with pdfplumber.open(pdf_path) as pdf:
        for pi in pages_idx:
            if pi >= len(pdf.pages):
                break
            for ln in page_lines(pdf.pages[pi]):
                t = ln['text'].strip()
                if not t or HEADER_PAT.search(t):
                    continue
                if re.match(r'^\d{1,3}$', t):
                    continue
                m = CODE_ANYWHERE.search(t)
                if m:
                    flush_sector()
                    if open_block is None:
                        open_block = {'lines': [], 'code': None, 'sector': sector}
                    open_block['code'] = m.group(1)
                    open_block['sector'] = open_block['sector'] or sector
                    rows.append(open_block)
                    open_block = None
                    continue
                # state line directly after a code line
                cand = t.strip('()').upper() if re.match(r'^\([A-Z][A-Z .&\-]{1,50}\)$', t) \
                    else (t.upper() if re.match(r'^[A-Z][A-Z .&\-]{1,50}$', t) else None)
                if cand is not None and cand in state_set:
                    for prev in reversed(rows):
                        if prev.get('code') and not prev.get('state_line'):
                            prev['state_line'] = cand
                            break
                    continue
                # wrapped sector header: consecutive ALL-CAPS lines between records
                if open_block is None and re.match(r'^[A-Z][A-Z0-9 .&\-]{1,60}$', t) \
                        and len(t.split()) <= 6 and t not in ('Total',) \
                        and not any(re.match(r'^[\d,.]+$', w['text'])
                                    or re.match(r'^\d{1,2}[-/]\d{4}$', w['text'])
                                    for w in ln['words']):
                    sector_buf.append(norm_text(t))
                    continue
                flush_sector()
                if open_block is None and any(w['x0'] >= 130 for w in ln['words']):
                    # sector may sit inline in a left column on the first line
                    lead = [w for w in ln['words'] if w['x0'] < 130
                            and re.match(r'^[A-Z][A-Z.&()\-]*$', w['text'])]
                    if lead and all(w['x0'] < 130 for w in lead):
                        cand = norm_text(' '.join(w['text'] for w in lead)).upper()
                        if len(cand) > 3 and not any(
                                re.match(r'^[\d,.]+$', w['text']) for w in ln['words']
                                if w['x0'] < 130):
                            sector = cand
                    open_block = {'lines': [ln], 'code': None, 'sector': sector}
                elif open_block is not None:
                    open_block['lines'].append(ln)
        if open_block is not None and open_block.get('code'):
            rows.append(open_block)

    out = []
    for b in rows:
        if not b.get('code'):
            continue
        toks = []          # (token, flag) flag: True=date, False=number, None=word
        agency = None
        for ln in b.get('lines', []):
            txt = ln['text'].strip()
            am = AGENCY_LINE.match(txt)
            if am and not re.search(r'\d{5,}', txt):
                agency = norm_text(txt[1:-1])
                continue
            for w in ln['words']:
                if w['x0'] < 130:
                    continue  # sector column
                tok = w['text']
                if re.match(r'^\d{1,2}[-/]\d{4}$', tok):
                    toks.append((tok, True))
                elif re.match(r'^[\d,]+(\.\d+)?$', tok):
                    toks.append((tok, False))
                else:
                    toks.append((tok, None))
        # structure: NAME... original_cost original_doc expenditure (name wraps
        # may follow). Anchor on the date token.
        date_tok, orig_cost, exp, name_cut = None, None, None, None
        di = next((i for i, (tk, f) in enumerate(toks) if f is True), None)
        if di is not None:
            date_tok = toks[di][0]
            oi = next((i for i in range(di - 1, -1, -1) if toks[i][1] is False), None)
            ei = next((i for i in range(di + 1, len(toks)) if toks[i][1] is False), None)
            if oi is not None:
                orig_cost = parse_number(toks[oi][0])
                name_cut = oi
            if ei is not None:
                exp = parse_number(toks[ei][0])
        else:
            nums = [i for i, (tk, f) in enumerate(toks) if f is False]
            if len(nums) >= 2:
                orig_cost = parse_number(toks[nums[-2]][0])
                exp = parse_number(toks[nums[-1]][0])
                name_cut = nums[-2]
        if name_cut is not None:
            name = ' '.join(tk for tk, f in toks[:name_cut])
        else:
            name = ' '.join(tk for tk, f in toks if f is None)
        name = re.sub(r'^\d{1,3}(?=[A-Z])', '', name).strip(' -')
        ym = None
        if date_tok:
            v = parse_month_year(date_tok)
            ym = f"{v[0]}-{v[1]:02d}" if v else None
        state = b.get('state_line')
        out.append({
            'project_code': b['code'], 'project_name': norm_text(name)[:300],
            'sector': b.get('sector'), 'state': state.title() if state else None,
            'agency': agency,
            'original_cost': orig_cost, 'original_doc': ym, 'expenditure': exp,
            'event': event,
            'completion_month': completion_month,
            'report_month': report_month,
        })
    return out


def find_table_pages(pdf_path, pattern):
    r = PdfReader(pdf_path)
    out = []
    for i, p in enumerate(r.pages):
        t = p.extract_text() or ''
        if '…' in t or '.....' in t or 'AGENCY_NAME' in t:
            continue  # skip table-of-contents and agency-legend pages
        if re.search(pattern, t, re.I):
            out.append(i)
    return out


def parse_qpisr(pdf_path, report_month, completion_month, census_table=7,
                closed_event=None):
    t7 = find_table_pages(pdf_path, rf'Table\s*:?-?\s*{census_table}\b')
    t3 = find_table_pages(pdf_path, r'Table\s*:?-?\s*3\b')
    ongoing, completed = [], []
    if t7:
        # census table runs to the end of the document, stopping at the legend
        n = PdfReader(pdf_path).get_num_pages()
        end = n
        r = PdfReader(pdf_path)
        for i in range(t7[0], n):
            t = r.pages[i].extract_text() or ''
            if 'AGENCY_NAME' in t:
                end = i
                break
        ongoing = parse_table7(pdf_path, list(range(t7[0], end)), report_month)
    if t3:
        t4 = find_table_pages(pdf_path, r'Table\s*:?-?\s*4\b')
        end = t4[0] if t4 else t3[0] + 15
        completed = parse_table3(pdf_path, list(range(t3[0], min(end, t3[0] + 15))),
                                 report_month, completion_month)
    # optional frozen/deleted table (August 2024 layout: Table 5)
    frozen = []
    if closed_event:
        t5 = find_table_pages(pdf_path, rf'Table\s*:?-?\s*5\b')
        if t5:
            t6 = find_table_pages(pdf_path, r'Table\s*:?-?\s*6\b')
            end = t6[0] if t6 else t5[0] + 6
            frozen = parse_table3(pdf_path, list(range(t5[0], min(end, t5[0] + 6))),
                                  report_month, completion_month, event=closed_event)
    return ongoing, completed + frozen


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--pdf', required=True)
    ap.add_argument('--month', required=True, help='census report month YYYY-MM')
    ap.add_argument('--completion-month', required=True,
                    help='quarter-end month used as completion month for Table 3')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--census-table', type=int, default=7)
    ap.add_argument('--closed-event', default=None,
                    help="parse Table 5 as a frozen/deleted event list")
    args = ap.parse_args()
    ongoing, completed = parse_qpisr(args.pdf, args.month, args.completion_month,
                                     census_table=args.census_table,
                                     closed_event=args.closed_event)
    os.makedirs(args.outdir, exist_ok=True)
    with open(f'{args.outdir}/qpisr_{args.month}_ongoing.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(ongoing[0].keys()) if ongoing else ['empty'])
        w.writeheader()
        w.writerows(ongoing)
    with open(f'{args.outdir}/qpisr_{args.month}_completed.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(completed[0].keys()) if completed else ['empty'])
        w.writeheader()
        w.writerows(completed)
    print(f"{args.month}: ongoing={len(ongoing)} completed={len(completed)}")


if __name__ == '__main__':
    main()
