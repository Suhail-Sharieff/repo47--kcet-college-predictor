"""
KCET College Predictor — ML Model Trainer
==========================================
Strategy:
  - For each (college_code, branch_name_normalized, category) triplet,
    we have up to 15 data points (5 years × 3 rounds).
  - We treat the MOST RECENT round cutoff per year as the "final" cutoff.
  - Model 1: GradientBoosting regressor to predict next cutoff (trend)
  - Model 2: Simple percentile-based lookup for direct rank comparison
  - Prediction: given user rank + category → find all seats where
    predicted_cutoff >= user_rank → scored + sorted list
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error

BASE_DIR  = Path(__file__).parent.parent
DATA_DIR  = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'models'
MODEL_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────
# Normalize branch names: unify 2021-2024 codes with 2025 full names
# ─────────────────────────────────────────────────────────────────────────

# Map 2-letter codes → canonical full name
BRANCH_MAP = {
    'AE': 'AEROSPACE ENGINEERING',
    'AI': 'ARTIFICIAL INTELLIGENCE AND MACHINE LEARNING',
    'AR': 'ARCHITECTURE',
    'AT': 'AUTOMATION AND ROBOTICS',
    'AU': 'AUTOMOBILE ENGINEERING',
    'BM': 'BIOMEDICAL ENGINEERING',
    'BT': 'BIO-TECHNOLOGY',
    'CA': 'COMPUTER APPLICATIONS',
    'CB': 'COMPUTER SCIENCE AND BUSINESS SYSTEMS',
    'CD': 'COMPUTER SCIENCE AND DESIGN',
    'CE': 'CIVIL ENGINEERING',
    'CH': 'CHEMICAL ENGINEERING',
    'CM': 'COMPUTER SCIENCE AND ENGINEERING (CYBER SECURITY)',
    'CN': 'COMPUTER NETWORKING',
    'CS': 'COMPUTER SCIENCE AND ENGINEERING',
    'CV': 'CIVIL ENGINEERING',
    'CY': 'COMPUTER SCIENCE AND ENGINEERING (CYBER SECURITY)',
    'EC': 'ELECTRONICS AND COMMUNICATION ENGG',
    'EE': 'ELECTRICAL & ELECTRONICS ENGINEERING',
    'EI': 'ELECTRONICS AND INSTRUMENTATION ENGINEERING',
    'EL': 'ELECTRICAL ENGINEERING',
    'EM': 'ELECTRONICS AND MEDIA TECHNOLOGY',
    'EN': 'ENVIRONMENTAL ENGINEERING',
    'ET': 'ELECTRONICS AND TELECOMMUNICATION ENGINEERING',
    'EV': 'ENVIRONMENTAL ENGINEERING',
    'FD': 'FOOD TECHNOLOGY',
    'GE': 'GEOLOGICAL ENGINEERING',
    'IE': 'INFORMATION SCIENCE AND ENGINEERING',
    'IM': 'INDUSTRIAL ENGINEERING & MANAGEMENT',
    'IS': 'INFORMATION SCIENCE AND ENGINEERING',
    'IT': 'INFORMATION TECHNOLOGY',
    'MA': 'MANUFACTURING ENGINEERING',
    'MB': 'MEDICAL ELECTRONICS ENGINEERING',
    'MC': 'MECHATRONICS',
    'ME': 'MECHANICAL ENGINEERING',
    'MH': 'MECHANICAL ENGINEERING',
    'ML': 'MACHINE LEARNING',
    'MN': 'MINING ENGINEERING',
    'MT': 'METALLURGICAL ENGINEERING',
    'MU': 'MECHATRONICS',
    'NC': 'NANO TECHNOLOGY',
    'PH': 'PHARMACEUTICAL ENGINEERING',
    'PS': 'POLYMER SCIENCE AND TECHNOLOGY',
    'RA': 'ROBOTICS AND AUTOMATION',
    'RO': 'ROBOTICS',
    'ST': 'SILK TECHNOLOGY',
    'TE': 'TEXTILE ENGINEERING',
    'TX': 'TEXTILES TECHNOLOGY',
    'UD': 'URBAN DESIGN',
}

# Also normalize common 2025 variants → canonical
BRANCH_ALIASES = {
    'AERO SPACE ENGINEERING': 'AEROSPACE ENGINEERING',
    'COMPUTER SCIENCE AND ENGG(ARTIFICIAL INTELLIGENCE AND MACHINE LEARNING)': 'COMPUTER SCIENCE AND ENGINEERING (AI & ML)',
    'COMPUTER SCIENCE AND ENGINEERING(DATA SCIENCE)': 'COMPUTER SCIENCE AND ENGINEERING (DATA SCIENCE)',
    'ARTIFICIAL INTELLIGENCE AND DATA SCIENCE': 'ARTIFICIAL INTELLIGENCE AND DATA SCIENCE',
    'ELECTRICAL & ELECTRONICS ENGINEERING': 'ELECTRICAL & ELECTRONICS ENGINEERING',
    'ELECTRONICS AND COMMUNICATION ENGG': 'ELECTRONICS AND COMMUNICATION ENGINEERING',
    'ELECTRONICS AND TELECOMMUNICATION ENGINEERING': 'ELECTRONICS AND TELECOMMUNICATION ENGINEERING',
    'INDUSTRIAL ENGINEERING & MANAGEMENT': 'INDUSTRIAL ENGINEERING & MANAGEMENT',
    'INFORMATION SCIENCE AND ENGINEERING': 'INFORMATION SCIENCE AND ENGINEERING',
}


def normalize_branch(code: str, name: str) -> str:
    """Return canonical branch name."""
    code = str(code).strip().upper()
    name = str(name).strip().upper()
    if code and code in BRANCH_MAP:
        canonical = BRANCH_MAP[code]
        # Check aliases for the canonical
        return BRANCH_ALIASES.get(canonical, canonical)
    # For 2025 (no code), use name directly with alias lookup
    return BRANCH_ALIASES.get(name, name)


# ─────────────────────────────────────────────────────────────────────────
# Load & prepare data
# ─────────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    path = DATA_DIR / 'cutoffs.csv'
    print(f"Loading {path} ...")
    df = pd.read_csv(path, low_memory=False)
    df['cutoff_rank'] = pd.to_numeric(df['cutoff_rank'], errors='coerce')
    df['branch_code'] = df['branch_code'].fillna('').astype(str)
    df = df.dropna(subset=['cutoff_rank'])
    df['cutoff_rank'] = df['cutoff_rank'].astype(int)
    print(f"  Raw rows: {len(df):,}")

    # Normalize branch names
    df['branch_canonical'] = df.apply(
        lambda r: normalize_branch(r['branch_code'], r['branch_name']), axis=1
    )

    # Clean college names — strip trailing address junk (everything after 2nd comma)
    def clean_college(name: str) -> str:
        name = str(name).strip()
        parts = name.split(',')
        # Keep college name + city (first 2 parts), remove address
        return ', '.join(parts[:2]).strip()

    df['college_clean'] = df['college_name'].apply(clean_college)

    print(f"  Unique colleges: {df['college_code'].nunique()}")
    print(f"  Unique branches: {df['branch_canonical'].nunique()}")
    print(f"  Categories:      {df['category'].nunique()}")
    print(f"  Years:           {sorted(df['year'].unique())}")
    return df


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a summary table: for each (college_code, branch_canonical, category)
    compute statistics across years/rounds for the prediction engine.
    """
    # Use Round 1 cutoff as the main reference (most competitive)
    r1 = df[df['round'] == 1].copy()

    # Group by (college, branch, category, year) → take min cutoff per year-round combo
    g = df.groupby(['college_code', 'college_clean', 'branch_canonical', 'category', 'year'])['cutoff_rank'].min().reset_index()

    summary = g.groupby(['college_code', 'college_clean', 'branch_canonical', 'category']).agg(
        mean_cutoff   = ('cutoff_rank', 'mean'),
        min_cutoff    = ('cutoff_rank', 'min'),
        max_cutoff    = ('cutoff_rank', 'max'),
        std_cutoff    = ('cutoff_rank', 'std'),
        n_years       = ('cutoff_rank', 'count'),
        latest_cutoff = ('cutoff_rank', 'last'),   # most recent year available
    ).reset_index()

    # Fill NaN std with 0
    summary['std_cutoff'] = summary['std_cutoff'].fillna(0)

    # Compute trend: slope of cutoff vs year
    def year_trend(group):
        if len(group) < 2:
            return 0.0
        years = group['year'].values.astype(float)
        cutoffs = group['cutoff_rank'].values.astype(float)
        # Simple linear regression slope
        x = years - years.mean()
        slope = np.dot(x, cutoffs) / (np.dot(x, x) + 1e-9)
        return slope

    trend = g.groupby(['college_code', 'branch_canonical', 'category']).apply(
        year_trend, include_groups=False
    ).reset_index()
    trend.columns = ['college_code', 'branch_canonical', 'category', 'trend']

    summary = summary.merge(trend, on=['college_code', 'branch_canonical', 'category'], how='left')
    summary['trend'] = summary['trend'].fillna(0)

    # Predicted 2026 cutoff = latest + trend (capped to valid rank range)
    summary['predicted_2026'] = (summary['latest_cutoff'] + summary['trend'])
    # Clip to [1, 250000], fill any NaN/inf with mean_cutoff as fallback
    summary['predicted_2026'] = summary['predicted_2026'].replace(
        [float('inf'), float('-inf')], float('nan')
    )
    summary['predicted_2026'] = summary['predicted_2026'].fillna(summary['mean_cutoff'])
    summary['predicted_2026'] = summary['predicted_2026'].clip(lower=1, upper=250000)
    summary['predicted_2026'] = summary['predicted_2026'].round().astype(int)

    print(f"Summary table: {len(summary):,} rows")
    return summary


# ─────────────────────────────────────────────────────────────────────────
# ML model (optional, for trend/confidence scoring)
# ─────────────────────────────────────────────────────────────────────────

def train_ml_model(df: pd.DataFrame):
    """Train a GradientBoosting model to predict cutoff given features."""
    print("\nTraining ML model ...")

    # Features: college (encoded), branch (encoded), category (encoded), year, round
    le_college  = LabelEncoder()
    le_branch   = LabelEncoder()
    le_cat      = LabelEncoder()

    df2 = df.dropna(subset=['cutoff_rank']).copy()
    df2 = df2[df2['cutoff_rank'] > 0]   # ensure positive for log transform
    df2['col_enc']    = le_college.fit_transform(df2['college_code'])
    df2['branch_enc'] = le_branch.fit_transform(df2['branch_canonical'])
    df2['cat_enc']    = le_cat.fit_transform(df2['category'])

    X = df2[['col_enc', 'branch_enc', 'cat_enc', 'year', 'round']].values
    y = np.log1p(df2['cutoff_rank'].values.astype(float))

    # Drop any NaN/inf that crept in
    valid = np.isfinite(y)
    X, y = X[valid], y[valid]

    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
        verbose=0,
    )
    model.fit(X, y)

    # Quick CV score
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
    mae_log = -scores.mean()
    # Back to rank scale
    y_pred = model.predict(X)
    mae_rank = mean_absolute_error(np.expm1(y), np.expm1(y_pred))
    print(f"  Training MAE (rank): {mae_rank:.0f}")
    print(f"  CV MAE (log scale):  {mae_log:.4f}")

    return model, le_college, le_branch, le_cat


# ─────────────────────────────────────────────────────────────────────────
# Save everything
# ─────────────────────────────────────────────────────────────────────────

def main():
    df = load_data()
    summary = build_summary(df)

    # Save summary (used by the prediction engine at runtime)
    summary_path = DATA_DIR / 'summary.csv'
    summary.to_csv(summary_path, index=False)
    print(f"\n[OK] Summary saved: {summary_path}")

    # Build lookup metadata for the UI
    colleges_meta = (
        df[['college_code','college_clean']]
        .drop_duplicates()
        .sort_values('college_code')
        .rename(columns={'college_clean': 'college_name'})
    )
    colleges_meta.to_csv(DATA_DIR / 'colleges.csv', index=False)

    branches_meta = (
        df[['branch_canonical']]
        .drop_duplicates()
        .sort_values('branch_canonical')
        .rename(columns={'branch_canonical': 'branch_name'})
    )
    branches_meta.to_csv(DATA_DIR / 'branches.csv', index=False)

    categories = sorted(df['category'].unique().tolist())
    with open(DATA_DIR / 'categories.json', 'w') as f:
        json.dump(categories, f)

    # Train ML model
    model, le_col, le_branch, le_cat = train_ml_model(df)

    # Save model bundle
    bundle = {
        'model':      model,
        'le_college': le_col,
        'le_branch':  le_branch,
        'le_cat':     le_cat,
    }
    joblib.dump(bundle, MODEL_DIR / 'model_bundle.pkl', compress=3)
    print(f"[OK] Model saved: {MODEL_DIR / 'model_bundle.pkl'}")
    print("\nDone! Ready to serve predictions.")


if __name__ == '__main__':
    main()
