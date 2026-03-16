#!/usr/bin/env python3
"""
Block B: プラセボ検定 (Construct Specificity Test)
===================================================
Review 14 対応: AdProxy が「あらゆるアウトカムを予測する万能変数」ではなく、
メンタルヘルス指標に特異的に関連することを示す。

方法:
  AdProxy → 心血管疾患有病率 / 糖尿病有病率 (プラセボアウトカム) の回帰を実行。
  メンタルヘルスアウトカム (depression, suicide) と比較して、
  プラセボでは効果が消失 or 大幅に縮小することを示す。

解釈:
  - AdProxy → depression: 有意 (主要結果)
  - AdProxy → cardiovascular: 非有意 or 符号反転 → 構成概念特異性を支持
  - AdProxy → diabetes: 非有意 or 符号反転 → 構成概念特異性を支持

依存: pandas, numpy, statsmodels
データ: data/macro/panel_merged.csv
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "macro")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")


def load_panel():
    for fname in ["panel_merged.csv", "panel_with_inactivity.csv"]:
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"Loaded: {fname} ({len(df):,} rows)")
            return df
    print("ERROR: No panel data found.")
    sys.exit(1)


def run_twfe(df, outcome, treatment="ad_proxy", entity="code", time="year"):
    """Run TWFE with entity + year FE, cluster-robust SE."""
    sub = df.dropna(subset=[outcome, treatment]).copy()
    sub[outcome] = pd.to_numeric(sub[outcome], errors="coerce")
    sub[treatment] = pd.to_numeric(sub[treatment], errors="coerce")
    sub = sub.dropna(subset=[outcome, treatment])
    if len(sub) < 100:
        return None

    sub[entity] = sub[entity].astype(str)
    entity_dummies = pd.get_dummies(sub[entity], prefix="fe", drop_first=True, dtype=float)
    year_dummies = pd.get_dummies(sub[time].astype(str), prefix="yr", drop_first=True, dtype=float)

    X = pd.concat([sub[[treatment]].astype(float), entity_dummies, year_dummies], axis=1)
    X = sm.add_constant(X)

    try:
        m = OLS(sub[outcome].astype(float), X).fit(cov_type="cluster",
                                      cov_kwds={"groups": sub[entity]})
        outcome_std = sub[outcome].std()
        return {
            "beta": float(m.params[treatment]),
            "se": float(m.bse[treatment]),
            "t": float(m.tvalues[treatment]),
            "pvalue": float(m.pvalues[treatment]),
            "ci_lo": float(m.conf_int().loc[treatment, 0]),
            "ci_hi": float(m.conf_int().loc[treatment, 1]),
            "cohen_d": float(m.params[treatment] / outcome_std) if outcome_std > 0 else np.nan,
            "n": int(m.nobs),
            "n_entities": int(sub[entity].nunique()),
            "r2_within": float(m.rsquared),
        }
    except Exception as e:
        print(f"  TWFE failed for {outcome}: {e}")
        return None


def run_first_diff(df, outcome, treatment="ad_proxy", entity="code", time="year"):
    """Run first-difference model."""
    sub = df.dropna(subset=[outcome, treatment]).sort_values([entity, time]).copy()
    sub[outcome] = pd.to_numeric(sub[outcome], errors="coerce")
    sub[treatment] = pd.to_numeric(sub[treatment], errors="coerce")
    sub[entity] = sub[entity].astype(str)
    sub[f"d_{outcome}"] = sub.groupby(entity)[outcome].diff().astype(float)
    sub[f"d_{treatment}"] = sub.groupby(entity)[treatment].diff().astype(float)
    fd = sub.dropna(subset=[f"d_{outcome}", f"d_{treatment}"])

    if len(fd) < 50:
        return None

    X = sm.add_constant(fd[[f"d_{treatment}"]].astype(float))
    try:
        m = OLS(fd[f"d_{outcome}"].astype(float), X).fit(cov_type="cluster",
                                             cov_kwds={"groups": fd[entity]})
        outcome_std = fd[f"d_{outcome}"].std()
        return {
            "beta": float(m.params[f"d_{treatment}"]),
            "se": float(m.bse[f"d_{treatment}"]),
            "t": float(m.tvalues[f"d_{treatment}"]),
            "pvalue": float(m.pvalues[f"d_{treatment}"]),
            "ci_lo": float(m.conf_int().loc[f"d_{treatment}", 0]),
            "ci_hi": float(m.conf_int().loc[f"d_{treatment}", 1]),
            "cohen_d": float(m.params[f"d_{treatment}"] / outcome_std) if outcome_std > 0 else np.nan,
            "n": int(m.nobs),
            "n_entities": int(fd[entity].nunique()),
        }
    except Exception as e:
        print(f"  FD failed for {outcome}: {e}")
        return None


def main():
    print("=" * 70)
    print("Placebo Test: AdProxy Construct Specificity (Review 14)")
    print("=" * 70)

    panel = load_panel()

    # Define outcomes: mental health (target) vs non-mental-health (placebo)
    outcomes = {
        # ── Target outcomes (should be significant) ──
        "depression_prevalence": {"label": "Depression prevalence", "type": "target"},
        "suicide": {"label": "Suicide rate", "type": "target"},
        "depression_dalys": {"label": "Depression DALYs", "type": "target"},
        # ── Placebo outcomes (should NOT be significant) ──
        "cardiovascular_prevalence": {"label": "Cardiovascular prevalence", "type": "placebo"},
        "diabetes_prevalence": {"label": "Diabetes prevalence", "type": "placebo"},
    }

    # Check availability
    available = {}
    for col, info in outcomes.items():
        if col in panel.columns and panel[col].notna().sum() > 100:
            n = panel[col].notna().sum()
            available[col] = info
            print(f"  [{info['type']:7s}] {info['label']}: {n:,} obs")
        else:
            print(f"  [MISS  ] {info['label']}: not available")

    if not any(v["type"] == "placebo" for v in available.values()):
        print("\n[WARNING] No placebo outcomes available.")
        print("Download from IHME GBD: cardiovascular diseases + diabetes")
        print("Run: python data/download_ihme_extended.py")
        return

    print()

    # ── Run models ──
    all_results = {}
    for col, info in available.items():
        print(f"\n{'─' * 50}")
        print(f"[{info['type'].upper()}] {info['label']}")
        print(f"{'─' * 50}")

        twfe = run_twfe(panel, col)
        fd = run_first_diff(panel, col)

        if twfe:
            sig = "***" if twfe["pvalue"] < 0.001 else "**" if twfe["pvalue"] < 0.01 else "*" if twfe["pvalue"] < 0.05 else "n.s."
            print(f"  TWFE: β={twfe['beta']:+.6f} [{twfe['ci_lo']:.6f}, {twfe['ci_hi']:.6f}] "
                  f"p={twfe['pvalue']:.4f} {sig} d={twfe['cohen_d']:.4f}")

        if fd:
            sig = "***" if fd["pvalue"] < 0.001 else "**" if fd["pvalue"] < 0.01 else "*" if fd["pvalue"] < 0.05 else "n.s."
            print(f"  FD:   β={fd['beta']:+.6f} [{fd['ci_lo']:.6f}, {fd['ci_hi']:.6f}] "
                  f"p={fd['pvalue']:.4f} {sig} d={fd['cohen_d']:.4f}")

        all_results[col] = {
            "label": info["label"],
            "type": info["type"],
            "twfe": twfe,
            "first_diff": fd,
        }

    # ── Save results ──
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, "placebo_test.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved: {output_path}")

    # ── Summary table ──
    print(f"\n{'=' * 70}")
    print("PLACEBO TEST SUMMARY (TWFE)")
    print(f"{'=' * 70}")
    print(f"{'Outcome':<30} {'Type':>8} {'β':>12} {'p-value':>10} {'d':>8} {'Sig':>6}")
    print("─" * 76)
    for col, res in all_results.items():
        twfe = res.get("twfe")
        if twfe:
            sig = "***" if twfe["pvalue"] < 0.001 else "**" if twfe["pvalue"] < 0.01 else "*" if twfe["pvalue"] < 0.05 else "n.s."
            print(f"  {res['label']:<28} {res['type']:>8} {twfe['beta']:>+12.6f} "
                  f"{twfe['pvalue']:>10.4f} {twfe['cohen_d']:>8.4f} {sig:>6}")

    # ── Interpretation ──
    target_sig = []
    placebo_sig = []
    for col, res in all_results.items():
        twfe = res.get("twfe")
        if twfe and twfe["pvalue"] < 0.05:
            if res["type"] == "target":
                target_sig.append(res["label"])
            else:
                placebo_sig.append(res["label"])

    print(f"\n{'─' * 70}")
    print("Interpretation:")
    if target_sig and not placebo_sig:
        print("  ✓ CONSTRUCT SPECIFICITY SUPPORTED")
        print(f"    AdProxy significantly predicts: {', '.join(target_sig)}")
        print(f"    AdProxy does NOT predict placebo outcomes")
    elif target_sig and placebo_sig:
        print("  △ PARTIAL SPECIFICITY")
        print(f"    Significant targets: {', '.join(target_sig)}")
        print(f"    Significant placebos: {', '.join(placebo_sig)}")
        print("    AdProxy may capture general development rather than mental-health-specific effects")
    else:
        print("  ? INCONCLUSIVE (insufficient significant results)")


if __name__ == "__main__":
    main()
