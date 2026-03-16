#!/usr/bin/env python3
"""
Block B: DALYs三角測量分析
===========================
Review 14 対応: depression prevalence に加え DALYs (Disability-Adjusted Life Years)
を代替アウトカムとして使用し、主要結果の頑健性を三角測量で確認する。

目的:
  depression_prevalence → depression_dalys に差し替えて、
  FE / TWFE / 1階差分の主要結果が再現されるかを検証。
  DALYs は有病率 (prevalence) と重症度 (severity) の積で構成されるため、
  診断基準の変化に対して有病率よりも頑健な指標である。

依存: pandas, numpy, statsmodels
データ: data/macro/panel_merged.csv (depression_dalys 列が必要)
"""

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
    """Load macro panel with extended outcomes."""
    for fname in ["panel_merged.csv", "panel_with_inactivity.csv"]:
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"Loaded: {fname} ({len(df):,} rows)")
            return df
    print("ERROR: No panel data found.")
    sys.exit(1)


def run_fe_model(df, outcome, treatment="ad_proxy", entity="code", time="year"):
    """Run entity-FE and TWFE models, return results dict."""
    sub = df.dropna(subset=[outcome, treatment]).copy()
    sub[outcome] = pd.to_numeric(sub[outcome], errors="coerce")
    sub[treatment] = pd.to_numeric(sub[treatment], errors="coerce")
    sub = sub.dropna(subset=[outcome, treatment])
    if len(sub) < 100:
        return None

    sub[entity] = sub[entity].astype(str)

    # Entity dummies
    entity_dummies = pd.get_dummies(sub[entity], prefix="fe", drop_first=True, dtype=float)
    year_dummies = pd.get_dummies(sub[time].astype(str), prefix="yr", drop_first=True, dtype=float)

    results = {}

    # ── Pooled OLS ──
    X_pooled = sm.add_constant(sub[[treatment]].astype(float))
    try:
        m_pooled = OLS(sub[outcome].astype(float), X_pooled).fit(cov_type="cluster",
                                                     cov_kwds={"groups": sub[entity]})
        results["pooled"] = {
            "beta": m_pooled.params[treatment],
            "se": m_pooled.bse[treatment],
            "pvalue": m_pooled.pvalues[treatment],
            "ci_lo": m_pooled.conf_int().loc[treatment, 0],
            "ci_hi": m_pooled.conf_int().loc[treatment, 1],
            "n": int(m_pooled.nobs),
            "r2": m_pooled.rsquared,
        }
    except Exception as e:
        print(f"  Pooled OLS failed: {e}")

    # ── Entity FE ──
    X_fe = pd.concat([sub[[treatment]].astype(float), entity_dummies], axis=1)
    X_fe = sm.add_constant(X_fe)
    try:
        m_fe = OLS(sub[outcome].astype(float), X_fe).fit(cov_type="cluster",
                                            cov_kwds={"groups": sub[entity]})
        results["entity_fe"] = {
            "beta": m_fe.params[treatment],
            "se": m_fe.bse[treatment],
            "pvalue": m_fe.pvalues[treatment],
            "ci_lo": m_fe.conf_int().loc[treatment, 0],
            "ci_hi": m_fe.conf_int().loc[treatment, 1],
            "n": int(m_fe.nobs),
            "r2_within": m_fe.rsquared,
        }
    except Exception as e:
        print(f"  Entity FE failed: {e}")

    # ── TWFE ──
    X_twfe = pd.concat([sub[[treatment]].astype(float), entity_dummies, year_dummies], axis=1)
    X_twfe = sm.add_constant(X_twfe)
    try:
        m_twfe = OLS(sub[outcome].astype(float), X_twfe).fit(cov_type="cluster",
                                                 cov_kwds={"groups": sub[entity]})
        results["twfe"] = {
            "beta": m_twfe.params[treatment],
            "se": m_twfe.bse[treatment],
            "pvalue": m_twfe.pvalues[treatment],
            "ci_lo": m_twfe.conf_int().loc[treatment, 0],
            "ci_hi": m_twfe.conf_int().loc[treatment, 1],
            "n": int(m_twfe.nobs),
            "r2_within": m_twfe.rsquared,
        }
    except Exception as e:
        print(f"  TWFE failed: {e}")

    # ── First Difference ──
    sub_sorted = sub.sort_values([entity, time])
    sub_sorted[f"d_{outcome}"] = sub_sorted.groupby(entity)[outcome].diff()
    sub_sorted[f"d_{treatment}"] = sub_sorted.groupby(entity)[treatment].diff()
    fd = sub_sorted.dropna(subset=[f"d_{outcome}", f"d_{treatment}"])

    if len(fd) > 50:
        X_fd = sm.add_constant(fd[[f"d_{treatment}"]].astype(float))
        try:
            m_fd = OLS(fd[f"d_{outcome}"].astype(float), X_fd).fit(cov_type="cluster",
                                                       cov_kwds={"groups": fd[entity]})
            results["first_diff"] = {
                "beta": m_fd.params[f"d_{treatment}"],
                "se": m_fd.bse[f"d_{treatment}"],
                "pvalue": m_fd.pvalues[f"d_{treatment}"],
                "ci_lo": m_fd.conf_int().loc[f"d_{treatment}", 0],
                "ci_hi": m_fd.conf_int().loc[f"d_{treatment}", 1],
                "n": int(m_fd.nobs),
                "r2": m_fd.rsquared,
            }
        except Exception as e:
            print(f"  FD failed: {e}")

    return results


def compute_cohen_d(beta, outcome_std):
    """Compute standardized effect size."""
    if outcome_std > 0:
        return beta / outcome_std
    return np.nan


def main():
    print("=" * 70)
    print("DALYs Triangulation Analysis (Review 14)")
    print("=" * 70)

    panel = load_panel()

    outcomes = {
        "depression_prevalence": "Depression prevalence (per 100k)",
        "depression_dalys": "Depression DALYs (per 100k)",
    }

    # Check data availability
    available = {}
    for col, label in outcomes.items():
        if col in panel.columns and panel[col].notna().sum() > 100:
            n_obs = panel[col].notna().sum()
            n_countries = panel.dropna(subset=[col])["code"].nunique()
            available[col] = label
            print(f"  {label}: {n_obs:,} obs, {n_countries} countries")
        else:
            print(f"  {label}: NOT AVAILABLE (run download_ihme_extended.py)")

    if "depression_dalys" not in available:
        print("\n[WARNING] depression_dalys not available.")
        print("Download from IHME GBD Results Tool:")
        print("  Cause=Depressive disorders, Measure=DALYs, Metric=Rate")
        print("  Save as: data/macro/ihme_depression_dalys.csv")
        print("  Then run: python data/download_ihme_extended.py")
        print("  Then run: python data/build_macro_panel.py")
        if "depression_prevalence" not in available:
            print("\nNo outcomes available. Exiting.")
            return
        print("\nRunning with depression_prevalence only as baseline...")

    print()

    all_results = {}
    for outcome_col, outcome_label in available.items():
        print(f"\n{'─' * 50}")
        print(f"Outcome: {outcome_label}")
        print(f"{'─' * 50}")

        outcome_std = panel[outcome_col].std()

        results = run_fe_model(panel, outcome_col)
        if results is None:
            print("  Insufficient data.")
            continue

        for model_name, res in results.items():
            d = compute_cohen_d(res["beta"], outcome_std)
            sign = "+" if res["beta"] > 0 else ""
            sig = "***" if res["pvalue"] < 0.001 else "**" if res["pvalue"] < 0.01 else "*" if res["pvalue"] < 0.05 else "n.s."
            print(f"  {model_name:12s}: β={sign}{res['beta']:.6f} "
                  f"[{res['ci_lo']:.6f}, {res['ci_hi']:.6f}] "
                  f"p={res['pvalue']:.4f} {sig} "
                  f"d={d:.4f} N={res['n']:,}")

        all_results[outcome_col] = results

    # ── Save results ──
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, "dalys_triangulation.json")

    import json

    # Convert to serializable format
    serializable = {}
    for outcome, models in all_results.items():
        serializable[outcome] = {}
        for model_name, res in models.items():
            serializable[outcome][model_name] = {
                k: float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in res.items()
            }

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved: {output_path}")

    # ── Comparison summary ──
    if len(all_results) >= 2:
        print(f"\n{'=' * 70}")
        print("TRIANGULATION COMPARISON")
        print(f"{'=' * 70}")
        print(f"{'Model':<15} {'Prevalence β':>15} {'DALYs β':>15} {'Sign match':>12}")
        print("─" * 60)
        for model_name in ["pooled", "entity_fe", "twfe", "first_diff"]:
            prev = all_results.get("depression_prevalence", {}).get(model_name)
            dalys = all_results.get("depression_dalys", {}).get(model_name)
            if prev and dalys:
                match = "✓" if (prev["beta"] > 0) == (dalys["beta"] > 0) else "✗"
                print(f"  {model_name:<13} {prev['beta']:>+15.6f} {dalys['beta']:>+15.6f} {match:>12}")


if __name__ == "__main__":
    main()
