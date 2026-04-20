"""
KCET Cutoff PDF Parser — Dual-Format Support (v3)
=================================================
Handles:
  - 2021-2024: "N E001 College Name City" headers, 2-letter branch codes
  - 2025 R1:   "College: (E001)Name" headers, 24 categories, full branch names
  - 2025 R2/R3:"College: E001 Name"  headers, 28 categories (adds GMP/NRI/OPN/OTH)

Fixes applied:
  - Dynamic category parsing from header line (handles 24 or 28 cols)
  - College regex handles both (E001) and E001 without parens
  - Split-decimal merging: "143346." + "5" -> "143346.5"
  - Stray page-number tokens appended to branch name lines are stripped
"""

import pdfplumber
import csv
import re
from pathlib import Path

# ── Standard 24 categories (2021-2024 + 2025 R1) ─────────────────────────
CATEGORIES_24 = [
    '1G','1K','1R',
    '2AG','2AK','2AR',
    '2BG','2BK','2BR',
    '3AG','3AK','3AR',
    '3BG','3BK','3BR',
    'GM','GMK','GMR',
    'SCG','SCK','SCR',
    'STG','STK','STR',
]

# ── Extended 28 categories (2025 R2/R3) ───────────────────────────────────
CATEGORIES_28 = [
    '1G','1K','1R',
    '2AG','2AK','2AR',
    '2BG','2BK','2BR',
    '3AG','3AK','3AR',
    '3BG','3BK','3BR',
    'GM','GMK','GMP','GMR',
    'NRI','OPN','OTH',
    'SCG','SCK','SCR',
    'STG','STK','STR',
]

# All valid category codes
ALL_CATEGORY_CODES = set(CATEGORIES_24 + CATEGORIES_28)

# ── Branch name normalization ──────────────────────────────────────────────
BRANCH_NAME_FIXES = [
    (re.compile(r'DAT\s+A\s+SCIENCE',    re.I), 'DATA SCIENCE'),
    (re.compile(r'TELECOMMUNICAT\s+ION', re.I), 'TELECOMMUNICATION'),
    (re.compile(r'COMMUNICAT\s+ION',     re.I), 'COMMUNICATION'),
    (re.compile(r'INFORMAT\s+ION',       re.I), 'INFORMATION'),
    (re.compile(r'INSTRUMENTAT\s+ION',   re.I), 'INSTRUMENTATION'),
    (re.compile(r'APPLICAT\s+ION',       re.I), 'APPLICATION'),
    (re.compile(r'PRODUCT\s+ION',        re.I), 'PRODUCTION'),
    (re.compile(r'\s{2,}',               re.I), ' '),
]

def normalize_branch_name(name: str) -> str:
    for pattern, replacement in BRANCH_NAME_FIXES:
        name = pattern.sub(replacement, name)
    return name.strip()

# ── 2-letter branch codes (2021-2024) ─────────────────────────────────────
OLD_BRANCH_CODES = {
    'AE','AI','AR','AT','AU','BM','BT','CA','CB','CD','CE',
    'CH','CM','CN','CS','CV','CY','EC','EE','EI','EL','EM',
    'EN','ET','EV','FD','GE','IM','IE','IS','IT','MA','MB',
    'MC','ME','MH','ML','MN','MT','MU','NC','PH','PS','RA',
    'RO','ST','TE','TX','UD',
}

# ── Regex patterns ─────────────────────────────────────────────────────────
# 2021-2024 college: "12 E042 College Name"
OLD_COLLEGE_RE = re.compile(r'^\d+\s+(E\d{3,4})\s+(.+)$')

# 2025 college - handles BOTH:
#   "College: (E002)Name..."   (R1 style)
#   "College: E002 Name..."    (R2/R3 style)
NEW_COLLEGE_RE = re.compile(
    r'^College:\s*\(?([Ee]\d{3,4})\)?\s*(.+)$'
)

# Old column header (2021-2024): starts with "1G"
OLD_HEADER_RE  = re.compile(r'^1G\s+1K\s+1R\s+')
# New column header (2025): "Course Name 1G 1K 1R ..."
NEW_HEADER_RE  = re.compile(r'^Course\s+Name\s+1G\s+1K\s+1R')

# A rank-bearing token: a number (possibly decimal/split) or "--"
RANK_TOKEN_RE  = re.compile(r'(\d[\d.]*|--)')
# Stray page-number suffix: e.g. "ENGINEERING & 5" — a lone short int at end of name
STRAY_NUM_RE   = re.compile(r'\s+\d{1,3}$')
# Page-number-only line
PAGE_NUM_RE    = re.compile(r'^\d{1,3}$')
# Footer
FOOTER_RE      = re.compile(r'^Generated on:', re.IGNORECASE)
# Title / header lines to skip in new format
TITLE_RE       = re.compile(
    r'^(KARNATAKA EXAMINATIONS|Non-Interactive|UGCET-\d{4}|Seat Type:)', re.IGNORECASE
)


# ─────────────────────────────────────────────────────────────────────────
# Token helpers
# ─────────────────────────────────────────────────────────────────────────

def extract_and_merge_tokens(line: str) -> list:
    """
    Extract rank tokens, merging split decimals.
    e.g. tokens ["143346.", "5", "--", "99."] + next "7" -> ["143346.5", "--", "99.7"]
    We merge in a second pass: if token[i] ends with '.' and token[i+1] is pure digits,
    merge them into one float string.
    """
    raw = RANK_TOKEN_RE.findall(line)
    merged = []
    i = 0
    while i < len(raw):
        tok = raw[i]
        if tok.endswith('.') and i + 1 < len(raw) and re.match(r'^\d+$', raw[i+1]):
            merged.append(tok + raw[i+1])
            i += 2
        else:
            merged.append(tok)
            i += 1
    return merged


def is_rank_line(line: str) -> bool:
    """True if ≥3 rank-like tokens found."""
    return len(RANK_TOKEN_RE.findall(line)) >= 3


def parse_rank(val: str):
    """'--' -> None, number string -> int (rounded from float)."""
    v = val.strip().rstrip('.')
    if v in ('--', '', '-'):
        return None
    try:
        return round(float(v))
    except ValueError:
        return None


def pad_to(lst: list, n: int, fill='--') -> list:
    while len(lst) < n:
        lst.append(fill)
    return lst[:n]


def parse_header_cats(line: str) -> list:
    """
    Extract ordered category list from a header line.
    'Course Name 1G 1K 1R 2AG ...' -> ['1G','1K','1R','2AG',...]
    """
    tokens = line.split()
    # Skip 'Course' and 'Name' prefix words
    cats = [t for t in tokens if t in ALL_CATEGORY_CODES]
    return cats if len(cats) >= 20 else CATEGORIES_24  # fallback


def strip_stray_page_num(name: str) -> str:
    """Remove trailing stray page number e.g. 'ENGINEERING & 5' -> 'ENGINEERING &'."""
    return STRAY_NUM_RE.sub('', name).strip()


def emit_rows(writer_fn, year, round_num, college_code, college_name,
              branch_code, branch_name, tokens, categories):
    """Write one CSV row per category that has a valid cutoff."""
    tokens     = pad_to(tokens, len(categories))
    branch_name = normalize_branch_name(branch_name)
    college_name = college_name.strip()
    for cat, val in zip(categories, tokens):
        rank = parse_rank(val)
        if rank is not None:
            writer_fn({
                'year':         year,
                'round':        round_num,
                'college_code': college_code.upper(),
                'college_name': college_name,
                'branch_code':  branch_code.strip(),
                'branch_name':  branch_name,
                'category':     cat,
                'cutoff_rank':  rank,
            })


# ─────────────────────────────────────────────────────────────────────────
# 2021–2024 parser
# ─────────────────────────────────────────────────────────────────────────

def parse_old_format(lines: list, state: dict, writer_fn):
    """State machine for 2021-2024 PDFs (2-letter branch codes, 24 cats)."""
    categories = state.get('categories', CATEGORIES_24)
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1

        if not line or PAGE_NUM_RE.match(line) or FOOTER_RE.match(line):
            continue

        # Column header
        if OLD_HEADER_RE.match(line):
            state['pending'] = None
            continue

        # College header
        m = OLD_COLLEGE_RE.match(line)
        if m:
            state['college_code'] = m.group(1)
            state['college_name'] = m.group(2).strip()
            state['pending'] = None
            continue

        if not state['college_code']:
            continue

        first = line.split()[0] if line.split() else ''

        # Branch data line
        if first in OLD_BRANCH_CODES:
            tokens = extract_and_merge_tokens(line)
            after_code = line[len(first):].strip()
            nm = re.search(r'(\d[\d.]*|--)\s*([\d.\s\-]+)$', after_code)
            if nm and len(tokens) >= 3:
                branch_name = after_code[:nm.start()].strip()
                # Check for name continuation on next non-blank line
                j = i
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j < len(lines):
                    nxt = lines[j].strip()
                    nxt_first = nxt.split()[0] if nxt.split() else ''
                    if (not is_rank_line(nxt)
                            and not OLD_COLLEGE_RE.match(nxt)
                            and not OLD_HEADER_RE.match(nxt)
                            and nxt_first not in OLD_BRANCH_CODES
                            and not PAGE_NUM_RE.match(nxt)):
                        branch_name = (branch_name + ' ' + nxt).strip()
                        i = j + 1

                emit_rows(writer_fn, state['year'], state['round'],
                          state['college_code'], state['college_name'],
                          first, branch_name, tokens, categories)
                state['pending'] = None
            else:
                state['pending'] = {'code': first, 'name': after_code, 'tokens': []}
            continue

        # Continuation of pending branch
        if state.get('pending'):
            pb = state['pending']
            if is_rank_line(line):
                pb['tokens'] = extract_and_merge_tokens(line)
                emit_rows(writer_fn, state['year'], state['round'],
                          state['college_code'], state['college_name'],
                          pb['code'], pb['name'], pb['tokens'], categories)
                state['pending'] = None
            else:
                pb['name'] += ' ' + line
            continue


# ─────────────────────────────────────────────────────────────────────────
# 2025 parser (all rounds)
# ─────────────────────────────────────────────────────────────────────────

def parse_new_format(lines: list, state: dict, writer_fn):
    """
    State machine for 2025 PDFs (all rounds).
    - Dynamically detects category count from header line (24 or 28)
    - Handles college code with or without parentheses
    - Merges split decimals and strips stray page numbers
    """
    # Use per-PDF categories (set when header line is parsed)
    if 'categories' not in state:
        state['categories'] = CATEGORIES_24

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1

        if not line or PAGE_NUM_RE.match(line) or FOOTER_RE.match(line):
            continue

        # Skip title/metadata lines
        if TITLE_RE.match(line):
            continue

        # Column header line — parse categories dynamically
        if NEW_HEADER_RE.match(line):
            cats = parse_header_cats(line)
            if cats:
                state['categories'] = cats
            state['pending'] = None
            continue

        # College header (handles both "(E001)" and "E001")
        m = NEW_COLLEGE_RE.match(line)
        if m:
            state['college_code'] = m.group(1).upper()
            # Strip trailing address info from college name
            raw_name = m.group(2).strip()
            state['college_name'] = raw_name
            state['pending'] = None
            continue

        if not state['college_code']:
            continue

        categories = state['categories']

        # ── Rank-bearing line ─────────────────────────────────────────────
        if is_rank_line(line):
            tokens = extract_and_merge_tokens(line)
            # Branch name is everything before the first numeric/-- token
            rm = re.search(r'(\d[\d.]*|--)', line)
            branch_name_part = line[:rm.start()].strip() if rm else ''
            # Strip any stray page number from name part
            branch_name_part = strip_stray_page_num(branch_name_part)

            if state.get('pending'):
                pb = state['pending']
                full_name = (pb['name'] + ' ' + branch_name_part).strip()
                full_name = strip_stray_page_num(full_name)
                emit_rows(writer_fn, state['year'], state['round'],
                          state['college_code'], state['college_name'],
                          '', full_name, tokens, categories)
                state['pending'] = None
            else:
                # Peek ahead for multi-line name continuations (no rank tokens)
                j = i
                while j < len(lines) and not lines[j].strip():
                    j += 1
                cont_parts = []
                while j < len(lines):
                    nxt = lines[j].strip()
                    if not nxt or PAGE_NUM_RE.match(nxt):
                        j += 1; continue
                    if (not is_rank_line(nxt)
                            and not NEW_COLLEGE_RE.match(nxt)
                            and not NEW_HEADER_RE.match(nxt)
                            and not FOOTER_RE.match(nxt)
                            and not TITLE_RE.match(nxt)):
                        cont_parts.append(strip_stray_page_num(nxt))
                        j += 1
                    else:
                        break
                full_name = (branch_name_part + ' ' + ' '.join(cont_parts)).strip()
                i = j
                emit_rows(writer_fn, state['year'], state['round'],
                          state['college_code'], state['college_name'],
                          '', full_name, tokens, categories)
                state['pending'] = None
            continue

        # ── Non-rank line — buffer as branch name start ───────────────────
        if (not NEW_COLLEGE_RE.match(line)
                and not NEW_HEADER_RE.match(line)
                and not FOOTER_RE.match(line)
                and not TITLE_RE.match(line)):
            clean_line = strip_stray_page_num(line)
            if state.get('pending'):
                state['pending']['name'] += ' ' + clean_line
            else:
                state['pending'] = {'name': clean_line, 'code': '', 'tokens': []}
            continue


# ─────────────────────────────────────────────────────────────────────────
# PDF entry point
# ─────────────────────────────────────────────────────────────────────────

def parse_pdf(pdf_path: str, year: int, round_num: int, writer_fn):
    """Parse one PDF, streaming rows via writer_fn. One page at a time."""
    is_2025  = (year == 2025)
    parse_fn = parse_new_format if is_2025 else parse_old_format

    state = {
        'year': year, 'round': round_num,
        'college_code': None, 'college_name': None,
        'pending': None,
        # categories will be set by header-line parsing for 2025
        'categories': CATEGORIES_24,
    }
    row_count = [0]

    def counting_writer(row):
        writer_fn(row)
        row_count[0] += 1

    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        for page_num, page in enumerate(pdf.pages):
            print(f"\r  [{year} R{round_num}] Page {page_num+1}/{total} - {row_count[0]} rows",
                  end='', flush=True)
            text = page.extract_text(x_tolerance=3, y_tolerance=3)
            if text:
                parse_fn(text.splitlines(), state, counting_writer)

    print(f"\r  [{year} R{round_num}] [OK] Done - {row_count[0]} rows          ")
    return row_count[0]


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main():
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    data_dir.mkdir(exist_ok=True)
    output_path = data_dir / 'cutoffs.csv'

    fieldnames = ['year','round','college_code','college_name',
                  'branch_code','branch_name','category','cutoff_rank']

    years  = [2021, 2022, 2023, 2024, 2025]
    rounds = [1, 2, 3]
    total  = 0

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        def write_row(row):
            writer.writerow(row)

        for year in years:
            for rnd in rounds:
                pdf_file = base_dir / str(year) / f'R{rnd}.pdf'
                if not pdf_file.exists():
                    print(f"  [SKIP] {pdf_file} not found")
                    continue
                count = parse_pdf(str(pdf_file), year, rnd, write_row)
                total += count
                f.flush()  # flush to disk after each PDF

    print(f"\n[DONE] Total records written: {total:,}")
    print(f"   Output: {output_path}")


if __name__ == '__main__':
    main()
