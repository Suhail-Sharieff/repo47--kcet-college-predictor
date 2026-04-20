"""Quick sanity test — 3 pages each from 2025 R2 and R3."""
import sys
sys.path.insert(0, 'src')
from parse_pdfs import parse_new_format, CATEGORIES_24

import pdfplumber

for rnd in [2, 3]:
    rows = []
    def capture(row): rows.append(row)

    state = {
        'year': 2025, 'round': rnd,
        'college_code': None, 'college_name': None,
        'pending': None, 'categories': CATEGORIES_24,
    }

    with pdfplumber.open(f'2025/R{rnd}.pdf') as pdf:
        for page_num in range(min(3, len(pdf.pages))):
            page = pdf.pages[page_num]
            text = page.extract_text(x_tolerance=3, y_tolerance=3)
            if text:
                parse_new_format(text.splitlines(), state, capture)

    print(f"\n=== 2025 R{rnd}: {len(rows)} rows from 3 pages ===")
    print(f"Categories detected: {state['categories']}")
    print(f"\nFirst 5 rows:")
    for r in rows[:5]:
        print(f"  {r['college_code']} | {r['branch_name'][:35]} | {r['category']} | {r['cutoff_rank']}")

    print(f"\nUnique categories in data:")
    print(f"  {sorted(set(r['category'] for r in rows))}")

    print(f"\nSample colleges:")
    for c in list(set((r['college_code'], r['college_name'][:50]) for r in rows))[:4]:
        print(f"  {c[0]}: {c[1]}")
