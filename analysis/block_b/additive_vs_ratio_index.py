#!/usr/bin/env python3
"""
加法指標による並行分析（頑健性検証）
レビュー指摘: 「理論は加法なのにマクロは比率I/Eで始める」ズレの解消

比率指標:  R = Internet / Education  (第一近似)
加法指標:  A = z(Internet) - z(Education)  (理論整合的)

主要結果（TWFE符号反転、閾値構造）が指標形式に対して頑健かを検証。

依存: pandas, numpy, scipy, statsmodels
データ: data/macro/ (panel data)
"""

import os
import sys
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "macro")


def load_panel():
    candidates = [
        os.path.join(DATA_DIR, "panel_merged.csv"),
        os.path.join(DATA_DIR, "panel_with_inactivity.csv"),
        os.path.join(DATA_DIR, "macro_panel.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return pd.read_csv(path)
    print(f"ERROR: No panel data in {DATA_DIR}")
    sys.exit(1)


def construct_indices(df):
    """Construct both ratio and additive indices."""
    # Identify columns
    inet_col = next((c for c in ['internet', 'internet_pct', 'IT.NET.USER.ZS'] if c in df.columns), None)
    edu_col = next((c for c in ['education', 'edu_years', 'mean_schooling'] if c in df.columns), None)
    entity_col = next((c for c in ['country', 'entity', 'iso3'] if c in df.columns), None)
    year_col = next((c for c in ['year', 'time'] if c in df.columns), None)

    if not all([inet_col, edu_col, entity_col, year_col]):
        print(f"Missing columns. Available: {list(df.columns)}")
        return None

    df = df.rename(columns={inet_col: 'internet', edu_col: 'education',
                            entity_col: 'entity', year_col: 'year'})

    # Drop missing
    df = df.dropna(subset=['internet', 'education'])
    df = df[df['education'] > 0]

    # Ratio index (first approximation)
    df['R_ratio'] = df['internet'] / df['education']

    # Additive index: within-country z-score difference
    # z-score within each country's time series
    df['z_internet'] = df.groupby('entity')['internet'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )
    df['z_education'] = df.groupby('entity')['education'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )
    df['A_additive'] = df['z_internet'] - df['z_education']

    # Also: cross-sectional z-score per year
    df['zx_internet'] = df.groupby('year')['internet'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )
    df['zx_education'] = df.groupby('year')['education'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )
    df['Ax_additive'] = df['zx_internet'] - df['zx_education']

    print(f"  Constructed indices for {df['entity'].nunique()} countries, "
          f"{df['year'].nunique()} years, {len(df)} obs")
    print(f"  R_ratio range: [{df['R_ratio'].min():.2f}, {df['R_ratio'].max():.2f}]")
    print(f"  A_additive range: [{df['A_additive'].min():.2f}, {df['A_additive'].max():.2f}]")

    return df


def run_fe_comparison(df, dep_col):
    """Compare ratio vs additive index in FE and TWFE models."""
    import statsmodels.api as sm

    df = df.dropna(subset=[dep_col, 'R_ratio', 'A_additive', 'entity', 'year'])

    # Entity demeaning for FE
    for col in [dep_col, 'R_ratio', 'A_additive']:
        df[f'{col}_dm'] = df[col] - df.groupby('entity')[col].transform('mean')

    # TWFE: also demean by year
    for col in [dep_col, 'R_ratio', 'A_additive']:
        df[f'{col}_twfe'] = df[f'{col}_dm'] - df.groupby('year')[f'{col}_dm'].transform('mean')

    results = {}
    for index_name, index_col in [('Ratio (R=I/E)', 'R_ratio'), ('Additive (z(I)-z(E))', 'A_additive')]:
        for spec_name, suffix in [('Country FE', '_dm'), ('TWFE', '_twfe')]:
            y = df[f'{dep_col}{suffix}'].values
            X = df[f'{index_col}{suffix}'].values.reshape(-1, 1)

            model = sm.OLS(y, X)
            res = model.fit(cov_type='HC1')

            key = f"{index_name} | {spec_name}"
            results[key] = {
                'β': res.params[0],
                'SE': res.bse[0],
                't': res.tvalues[0],
                'p': res.pvalues[0],
                'N': len(y),
            }

    return results


def print_comparison(results, outcome_label):
    print(f"\n{'='*70}")
    print(f" Index Comparison: {outcome_label}")
    print(f"{'='*70}")
    print(f"  {'Specification':<40s} {'β':>10s} {'SE':>10s} {'t':>8s} {'p':>10s}")
    print(f"  {'-'*78}")
    for key, vals in results.items():
        sig = '***' if vals['p'] < 0.001 else '**' if vals['p'] < 0.01 else '*' if vals['p'] < 0.05 else ''
        print(f"  {key:<40s} {vals['β']:10.4f} {vals['SE']:10.4f} {vals['t']:8.2f} {vals['p']:10.4f} {sig}")

    # Key comparison: does TWFE reversal hold for both indices?
    ratio_fe = [v for k, v in results.items() if 'Ratio' in k and 'Country FE' in k][0]
    ratio_twfe = [v for k, v in results.items() if 'Ratio' in k and 'TWFE' in k][0]
    add_fe = [v for k, v in results.items() if 'Additive' in k and 'Country FE' in k][0]
    add_twfe = [v for k, v in results.items() if 'Additive' in k and 'TWFE' in k][0]

    print(f"\n  TWFE Sign Reversal Check:")
    ratio_reversal = (ratio_fe['β'] > 0 and ratio_twfe['β'] < 0) or (ratio_fe['β'] < 0 and ratio_twfe['β'] > 0)
    add_reversal = (add_fe['β'] > 0 and add_twfe['β'] < 0) or (add_fe['β'] < 0 and add_twfe['β'] > 0)
    print(f"    Ratio:    FE β={ratio_fe['β']:+.4f} → TWFE β={ratio_twfe['β']:+.4f}  Reversal: {'YES' if ratio_reversal else 'NO'}")
    print(f"    Additive: FE β={add_fe['β']:+.4f} → TWFE β={add_twfe['β']:+.4f}  Reversal: {'YES' if add_reversal else 'NO'}")
    print(f"    → {'ROBUST: Reversal holds for both index forms' if (ratio_reversal == add_reversal) else 'DIVERGENT: Index form matters'}")


def main():
    print("=" * 70)
    print(" Additive vs Ratio Index: Robustness Comparison")
    print("=" * 70)

    print("\nLoading data...")
    df = load_panel()
    df = construct_indices(df)
    if df is None:
        return

    # Find depression column
    dep_col = next((c for c in ['dep', 'dep_rate', 'depression', 'depression_rate']
                    if c in df.columns), None)
    sui_col = next((c for c in ['suicide', 'suicide_rate', 'sui_rate']
                    if c in df.columns), None)

    if dep_col:
        results = run_fe_comparison(df, dep_col)
        print_comparison(results, f"Depression ({dep_col})")

    if sui_col:
        results = run_fe_comparison(df, sui_col)
        print_comparison(results, f"Suicide ({sui_col})")

    print(f"\n{'='*70}")
    print(" INTERPRETATION")
    print(f"{'='*70}")
    print("""
  If TWFE sign reversal holds for BOTH ratio and additive indices:
    → The finding is robust to index functional form
    → Theoretical consistency: additive model L = α₁·I − α₂·C
      is supported at macro level

  If reversal holds for ratio but NOT additive (or vice versa):
    → Index form matters for inference
    → Need to report both and discuss discrepancy
    """)


if __name__ == "__main__":
    main()
