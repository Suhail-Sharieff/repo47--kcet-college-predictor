"""
KCET Data Cleaner
=================
Fixes:
  1. College name deduplication — uses college_code as canonical key
  2. Branch name normalization — removes stray digits, unifies near-duplicates
  3. Merges duplicate summary rows produced by the above normalizations

Run:  venv\\Scripts\\python src\\clean_data.py
"""

import pandas as pd
import re
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'


# =============================================================================
# PART 1: College name deduplication
# =============================================================================

# Karnataka cities — used to truncate college names after the city token
KA_CITIES = re.compile(
    r'\b(Bangalore|Bengaluru|Mysore|Mysuru|Hubli|Dharwad|Belgaum|Belagavi|'
    r'Mangalore|Mangaluru|Kalaburagi|Gulbarga|Davanagere|Bellary|Ballari|'
    r'Shimoga|Shivamogga|Tumkur|Tumakuru|Bijapur|Vijayapura|Hassan|Udupi|'
    r'Raichur|Chitradurga|Mandya|Chikkaballapur|Chickballapur|Ramanagara|'
    r'Chikmagalur|Kodagu|Bagalkot|Bidar|KGF|Kolar|Tiptur|Nitte|Surathkal|'
    r'Yelahanka|Hesaraghatta|Anekal|Devanahalli)\b',
    re.IGNORECASE
)


def clean_college_name(name: str) -> str:
    """Produce a short, address-free college name."""
    name = str(name).strip()

    # Remove parenthetical institutional qualifiers
    name = re.sub(
        r'\s*\((AUTONOMOUS|DEEMED|A STATE AUTONOMOUS[^)]*|PUBLIC UNIV[^)]*'
        r'|SOUTH[^)]*|RING[^)]*)\)',
        '', name, flags=re.IGNORECASE
    ).strip()

    # Remove everything after known address keywords
    name = re.sub(
        r'\s+(P\.?\s?B\.?\s?(NO\.?|BOX)|POST BOX|OUTER RING|BULL TEMPLE|'
        r'VIDYA SOUDHA|SHAVIGE|KUMARASWAMY|AMBEDKAR VEEDHI|K\.R\.\s*CIRCLE|'
        r'NEAR ITPB|KRISHNADEVA|MARALUR|MANDYA:\s*\d|JSS TECHNICAL|'
        r'B\.H\.|00RGAUM|BANGARAPET|BANGALORE\s*-\s*MYSORE|R\.V\. VIDYANIKETAN|'
        r'MSR NAGAR|MYSORE ROAD|RING ROAD|CAMPUS|LAYOUT|NAGAR,|PIN\s*\d).*',
        '', name, flags=re.IGNORECASE
    ).strip()

    # Truncate after city name
    m = KA_CITIES.search(name)
    if m:
        name = name[:m.end()].strip()

    name = re.sub(r'\s{2,}', ' ', name)
    return name.rstrip('.,- ')


def build_canonical_names(df: pd.DataFrame) -> dict:
    """
    Returns {college_code: canonical_name}.
    Priority: 2021-2024 names (short, clean) > 2025 cleaned name.
    """
    canonical = {}

    # First: 2021-2024 data — vote for most common cleaned name
    old = df[df['year'] < 2025]
    for code, group in old.groupby('college_code'):
        names = [clean_college_name(n) for n in group['college_name']]
        best = Counter(names).most_common(1)[0][0]
        canonical[code] = best

    # Second: fill codes first introduced in 2025
    df25 = df[df['year'] == 2025]
    for code, group in df25.groupby('college_code'):
        if code not in canonical:
            names = [clean_college_name(n) for n in group['college_name'].unique()]
            canonical[code] = max(names, key=len)

    return canonical


# =============================================================================
# PART 2: Branch name normalization
# =============================================================================

# Stray-digit regexes (PDF decimal-split artifact)
# "Artificial Intelligence 5 And Machine Learning" -> "...And Machine Learning"
STRAY_MID_RE   = re.compile(r'(?<=[A-Za-z\)])\s+\d{1,3}\s+(?=[A-Za-z\(])')
STRAY_TRAIL_RE = re.compile(r'\s+\d{1,3}\s*$')

# All uppercase key -> canonical Title Case name
BRANCH_MERGE = {
    # Aerospace
    'AERO SPACE ENGINEERING':                            'Aerospace Engineering',
    'AERONAUTICAL ENGINEERING':                          'Aerospace Engineering',
    'AERONAUTICS ENGINEERING':                           'Aerospace Engineering',
    'AEROSPACE ENGINEERING':                             'Aerospace Engineering',
    # Electronics & Communication
    'ELECTRONICS AND COMMUNICATION ENGG':                'Electronics And Communication Engineering',
    'ELECTRONICS & COMMUNICATION ENGINEERING':           'Electronics And Communication Engineering',
    'ELECTRONICS AND COMMUNICATION ENGINEERING':         'Electronics And Communication Engineering',
    # Electrical
    'ELECTRICAL & ELECTRONICS ENGINEERING':              'Electrical And Electronics Engineering',
    'ELECTRICAL AND ELECTRONICS ENGINEERING':            'Electrical And Electronics Engineering',
    'ELECTRICAL AND ELECTRONICS ENGG':                   'Electrical And Electronics Engineering',
    'ELECTRICAL ENGINEERING':                            'Electrical And Electronics Engineering',
    # Telecom
    'ELECTRONICS AND TELECOMMUNICATION ENGINEERING':     'Electronics And Telecommunication Engineering',
    'ELECTRONICS & TELECOMMUNICATION ENGINEERING':       'Electronics And Telecommunication Engineering',
    # Information Science
    'INFORMATION SCIENCE AND ENGINEERING':               'Information Science And Engineering',
    'INFORMATION SCIENCE & ENGINEERING':                 'Information Science And Engineering',
    # Industrial
    'INDUSTRIAL ENGINEERING & MANAGEMENT':               'Industrial Engineering And Management',
    'INDUSTRIAL ENGINEERING AND MANAGEMENT':             'Industrial Engineering And Management',
    'INDUSTRIAL ENGINEERING AND MGMT':                   'Industrial Engineering And Management',
    'INDUSTRIAL ENGINEERING':                            'Industrial Engineering And Management',
    # Instrumentation
    'ELECTRONICS AND INSTRUMENTATION ENGINEERING':       'Electronics And Instrumentation Engineering',
    'ELECTRONICS & INSTRUMENTATION ENGINEERING':         'Electronics And Instrumentation Engineering',
    # AI / ML
    'ARTIFICIAL INTELLIGENCE AND MACHINE LEARNING':      'Artificial Intelligence And Machine Learning',
    'ARTIFICIAL INTELLIGENCE':                           'Artificial Intelligence And Machine Learning',
    'ARTIFICIAL INTELLIGENCE ENGG':                      'Artificial Intelligence And Machine Learning',
    'COMPUTER SCIENCE AND ENGG(ARTIFICIAL INTELLIGENCE AND MACHINE LEARNING)':
                                                         'Computer Science And Engineering (Ai And Ml)',
    'COMPUTER SCIENCE AND ENGINEERING (AI & ML)':        'Computer Science And Engineering (Ai And Ml)',
    'COMPUTER SCIENCE AND ENGINEERING (AI AND ML)':      'Computer Science And Engineering (Ai And Ml)',
    'COMPUTER SCIENCE AND ENGINEERING(ARTIFICIAL INTELLIGENCE AND MACHINE LEARNING)':
                                                         'Computer Science And Engineering (Ai And Ml)',
    # Data Science
    'COMPUTER SCIENCE AND ENGINEERING(DATA SCIENCE)':    'Computer Science And Engineering (Data Science)',
    'COMPUTER SCIENCE AND ENGINEERING (DATA SCIENCE)':   'Computer Science And Engineering (Data Science)',
    'COMPUTER SCIENCE AND ENGINEERING(D ATA SCIENCE)':   'Computer Science And Engineering (Data Science)',
    'COMPUTER SCIENCE AND ENGINEERING(DAT A SCIENCE)':   'Computer Science And Engineering (Data Science)',
    # AI + Data Science
    'ARTIFICIAL INTELLIGENCE AND DATA SCIENCE':          'Artificial Intelligence And Data Science',
    # Cyber Security
    'COMPUTER SCIENCE AND ENGINEERING (CYBER SECURITY)': 'Computer Science And Engineering (Cyber Security)',
    'COMPUTER SCIENCE AND ENGINEERING(CYBER SECURITY)':  'Computer Science And Engineering (Cyber Security)',
    # Computer Science
    'COMPUTER SCIENCE AND ENGINEERING':                  'Computer Science And Engineering',
    'COMPUTER SCIENCE AND ENGG':                         'Computer Science And Engineering',
    'COMPUTER SCIENCE & ENGINEERING':                    'Computer Science And Engineering',
    # Information Technology
    'INFORMATION TECHNOLOGY':                            'Information Technology',
    # Computer Networking
    'COMPUTER NETWORKING':                               'Computer Networking',
    # Civil
    'CIVIL ENGINEERING':                                 'Civil Engineering',
    'CIVIL ENGG':                                        'Civil Engineering',
    # Mechanical
    'MECHANICAL ENGINEERING':                            'Mechanical Engineering',
    'MECHANICAL ENGG':                                   'Mechanical Engineering',
    # Chemical
    'CHEMICAL ENGINEERING':                              'Chemical Engineering',
    'CHEMICAL ENGG':                                     'Chemical Engineering',
    # Biotechnology
    'BIO-TECHNOLOGY':                                    'Biotechnology',
    'BIOTECHNOLOGY':                                     'Biotechnology',
    'BIO TECHNOLOGY':                                    'Biotechnology',
    # Textiles
    'TEXTILES TECHNOLOGY':                               'Textile Technology',
    'TEXTILE TECHNOLOGY':                                'Textile Technology',
    'TEXTILE ENGINEERING':                               'Textile Technology',
    # Silk
    'SILK TECHNOLOGY':                                   'Silk Technology',
    # Medical Electronics
    'MEDICAL ELECTRONICS ENGINEERING':                   'Medical Electronics Engineering',
    # Robotics
    'ROBOTICS AND AUTOMATION':                           'Robotics And Automation',
    'AUTOMATION AND ROBOTICS':                           'Robotics And Automation',
    # Agriculture
    'AGRICULTURE ENGINEERING':                           'Agriculture Engineering',
    # Mining
    'MINING ENGINEERING':                                'Mining Engineering',
    # Metallurgy
    'METALLURGICAL ENGINEERING':                         'Metallurgical Engineering',
    # Mechatronics
    'MECHATRONICS':                                      'Mechatronics',
    # Nano Tech
    'NANO TECHNOLOGY':                                   'Nano Technology',
    # Architecture
    'ARCHITECTURE':                                      'Architecture',
    # Food Tech
    'FOOD TECHNOLOGY':                                   'Food Technology',
    # Biomedical
    'BIOMEDICAL ENGINEERING':                            'Biomedical Engineering',
    # Environmental
    'ENVIRONMENTAL ENGINEERING':                         'Environmental Engineering',
    # Automobile
    'AUTOMOBILE ENGINEERING':                            'Automobile Engineering',
    'AUTOMOTIVE ENGINEERING':                            'Automobile Engineering',
}


def normalize_branch_canonical(name: str) -> str:
    """Strip stray digits and map to canonical branch name."""
    name = name.strip()
    name = STRAY_MID_RE.sub(' ', name)          # remove mid-name stray digit
    name = STRAY_TRAIL_RE.sub('', name).strip() # remove trailing stray digit
    name = re.sub(r'\s{2,}', ' ', name).strip()
    return BRANCH_MERGE.get(name.upper(), name.title())


# =============================================================================
# PART 3: Apply and save
# =============================================================================

def main():
    print("Loading data...")
    df = pd.read_csv(DATA_DIR / 'cutoffs.csv', low_memory=False)
    df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)
    summary = pd.read_csv(DATA_DIR / 'summary.csv', low_memory=False)

    # ── Step 1: canonical college names ──────────────────────────────────
    print("\nBuilding canonical college names...")
    canonical_names = build_canonical_names(df)
    print(f"  {len(canonical_names)} college codes")

    # Report inconsistencies
    changed = []
    for code, canon in canonical_names.items():
        old_names = df[df['college_code'] == code]['college_name'].unique()
        unique_cleaned = set(clean_college_name(n) for n in old_names)
        if len(unique_cleaned) > 1:
            changed.append((code, list(unique_cleaned), canon))

    print(f"  {len(changed)} colleges had inconsistent names:")
    for code, variants, canon in sorted(changed)[:15]:
        print(f"  {code}  =>  {canon}")
        for v in variants[:2]:
            if v != canon:
                print(f"         was: {v}")
    if len(changed) > 15:
        print(f"  ... and {len(changed)-15} more (all fixed)")

    # ── Step 2: patch summary.csv ─────────────────────────────────────────
    print("\nPatching summary.csv...")
    summary['college_clean'] = summary['college_code'].map(canonical_names).fillna(
        summary['college_clean']
    )
    summary['branch_canonical'] = summary['branch_canonical'].apply(normalize_branch_canonical)

    before = len(summary)
    agg_cols = {
        'mean_cutoff':    'mean',
        'min_cutoff':     'min',
        'max_cutoff':     'max',
        'std_cutoff':     'mean',
        'n_years':        'max',
        'latest_cutoff':  'last',
        'trend':          'mean',
        'predicted_2026': 'min',
    }
    summary = summary.groupby(
        ['college_code', 'college_clean', 'branch_canonical', 'category'],
        as_index=False
    ).agg(agg_cols)
    summary['mean_cutoff']    = summary['mean_cutoff'].round().astype(int)
    summary['min_cutoff']     = summary['min_cutoff'].astype(int)
    summary['max_cutoff']     = summary['max_cutoff'].astype(int)
    summary['predicted_2026'] = summary['predicted_2026'].astype(int)

    after = len(summary)
    print(f"  Rows: {before:,} -> {after:,} (merged {before - after:,} duplicates)")
    summary.to_csv(DATA_DIR / 'summary.csv', index=False)
    print("  Saved: summary.csv")

    # ── Step 3: colleges.csv ─────────────────────────────────────────────
    print("\nSaving colleges.csv...")
    colleges = pd.DataFrame([
        {'college_code': k, 'college_name': v}
        for k, v in sorted(canonical_names.items())
    ])
    colleges.to_csv(DATA_DIR / 'colleges.csv', index=False)
    print(f"  {len(colleges)} colleges saved")

    # ── Step 4: branches.csv ─────────────────────────────────────────────
    print("\nSaving branches.csv...")
    branches = pd.DataFrame({
        'branch_name': sorted(summary['branch_canonical'].unique())
    })
    branches.to_csv(DATA_DIR / 'branches.csv', index=False)
    print(f"  {len(branches)} branches saved")

    print("\n[DONE] Data cleaned. Restart the server to apply changes.")


if __name__ == '__main__':
    main()
