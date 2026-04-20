"""
KCET College Predictor — Prediction Engine
===========================================
Given user_rank + category → returns ranked list of (college, branch)
with confidence levels based on historical cutoff analysis.
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from functools import lru_cache

BASE_DIR  = Path(__file__).parent.parent
DATA_DIR  = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'models'


@lru_cache(maxsize=1)
def load_assets():
    """Load summary CSV and model bundle once, cache in memory."""
    summary = pd.read_csv(DATA_DIR / 'summary.csv')
    colleges = pd.read_csv(DATA_DIR / 'colleges.csv')
    branches = pd.read_csv(DATA_DIR / 'branches.csv')
    with open(DATA_DIR / 'categories.json') as f:
        categories = json.load(f)

    bundle = None
    bundle_path = MODEL_DIR / 'model_bundle.pkl'
    if bundle_path.exists():
        bundle = joblib.load(bundle_path)

    return summary, colleges, branches, categories, bundle


def admission_confidence(user_rank, mean_cutoff, std_cutoff, predicted_2026):
    """
    Returns a confidence label: 'Safe', 'Moderate', 'Ambitious', 'Reach'
    based on how the user's rank compares to the historical distribution.
    Lower CET rank = better.
    """
    # If user rank is BELOW (smaller number = better) the prediction
    gap = predicted_2026 - user_rank  # positive = user is better
    if std_cutoff <= 0:
        std_cutoff = max(mean_cutoff * 0.1, 1000)

    z_score = gap / std_cutoff  # how many SDs above the cutoff
    if z_score >= 1.5:
        return 'Safe'
    elif z_score >= 0.3:
        return 'Moderate'
    elif z_score >= -0.5:
        return 'Ambitious'
    else:
        return 'Reach'


def predict(user_rank: int, category: str, branch_filter: list = None,
            top_n: int = 100) -> list:
    """
    Main prediction function.
    Returns list of dicts sorted from best (most competitive) to easiest admission.
    """
    summary, colleges, branches, categories, bundle = load_assets()

    # Filter by category
    mask = summary['category'] == category
    df = summary[mask].copy()

    if df.empty:
        return []

    # If branch filter provided
    if branch_filter:
        bf_upper = [b.upper() for b in branch_filter]
        df = df[df['branch_canonical'].str.upper().isin(bf_upper)]

    # Keep only rows where user rank <= predicted_2026 cutoff
    # (user's rank number is smaller → better → they qualify)
    eligible = df[df['predicted_2026'] >= user_rank].copy()

    if eligible.empty:
        # Fallback: return closest matches even if slightly out of range
        eligible = df.nsmallest(top_n, 'predicted_2026')
        eligible = eligible[eligible['predicted_2026'] >= user_rank * 0.7]

    # Score: colleges with SMALLER predicted cutoff are more prestigious
    # We sort by predicted_2026 ascending (most competitive first)
    eligible = eligible.sort_values('predicted_2026')

    results = []
    for _, row in eligible.head(top_n).iterrows():
        confidence = admission_confidence(
            user_rank,
            row['mean_cutoff'],
            row['std_cutoff'],
            row['predicted_2026'],
        )
        results.append({
            'college_code':    row['college_code'],
            'college_name':    row['college_clean'],
            'branch':          row['branch_canonical'].title(),
            'category':        row['category'],
            'predicted_cutoff': int(row['predicted_2026']),
            'mean_cutoff':     int(row['mean_cutoff']),
            'min_cutoff':      int(row['min_cutoff']),
            'max_cutoff':      int(row['max_cutoff']),
            'n_years':         int(row['n_years']),
            'confidence':      confidence,
            'trend':           round(float(row['trend']), 1),
        })

    return results


def get_trends(college_code: str, branch: str, category: str) -> dict:
    """
    Return year-wise cutoff trend for a specific college+branch+category.
    Reads the raw CSV and matches by branch name (case-insensitive).
    """
    df = pd.read_csv(DATA_DIR / 'cutoffs.csv')

    mask = (
        (df['college_code'].str.upper() == college_code.upper()) &
        (df['category'] == category) &
        (df['branch_name'].str.upper() == branch.upper())
    )
    subset = df[mask].copy()

    # If no exact match on branch_name, try matching the summary branch name
    if subset.empty:
        mask2 = (
            (df['college_code'].str.upper() == college_code.upper()) &
            (df['category'] == category)
        )
        subset2 = df[mask2].copy()
        # Fuzzy: find any row whose branch_name is contained in the query branch
        branch_upper = branch.upper()
        subset = subset2[
            subset2['branch_name'].str.upper().apply(
                lambda n: n in branch_upper or branch_upper in n
            )
        ]

    if subset.empty:
        return {}

    # Group by year, take min cutoff across rounds (round 1 most competitive)
    trend = subset.groupby('year')['cutoff_rank'].min().reset_index()
    return {int(r['year']): int(r['cutoff_rank']) for _, r in trend.iterrows()}


if __name__ == '__main__':
    # Quick test
    results = predict(user_rank=5000, category='1G', top_n=10)
    for r in results:
        print(f"{r['college_code']} | {r['college_name'][:40]} | {r['branch'][:30]} | "
              f"Cutoff: {r['predicted_cutoff']} | {r['confidence']}")
