#!/usr/bin/env python3
"""
Block B: 効果量サマリーテーブル
================================
Review 14 対応: 全モデルのβ, 95% CI, Cohen's d, partial η² を
一覧テーブルとして出力する。

出力:
  - results/effect_size_table.csv  (機械可読)
  - results/effect_size_table.tex  (LaTeX用テーブル)
  - 標準出力にサマリー

依存: pandas, numpy, statsmodels
データ: data/macro/panel_merged.csv + NHANES/ATUS結果
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
            return pd.read_csv(path)
    return None


def run_model(df, outcome, treatment, entity="code", time="year",
              model_type="twfe"):
    """Run a single model specification and return effect size metrics."""
    sub = df.dropna(subset=[outcome, treatment]).copy()
    sub[outcome] = pd.to_numeric(sub[outcome], errors="coerce")
    sub[treatment] = pd.to_numeric(sub[treatment], errors="coerce")
    sub = sub.dropna(subset=[outcome, treatment])
    if len(sub) < 50:
        return None

    outcome_std = sub[outcome].std()
    treatment_std = sub[treatment].std()

    # Ensure entity column is string for dummies
    sub[entity] = sub[entity].astype(str)

    if model_type == "pooled":
        X = sm.add_constant(sub[[treatment]].astype(float))
        m = OLS(sub[outcome].astype(float), X).fit(cov_type="cluster",
                                      cov_kwds={"groups": sub[entity]})
        beta = m.params[treatment]
        se = m.bse[treatment]
        t_val = m.tvalues[treatment]
        p_val = m.pvalues[treatment]
        ci = m.conf_int().loc[treatment]
        n = int(m.nobs)

    elif model_type == "entity_fe":
        fe = pd.get_dummies(sub[entity], prefix="fe", drop_first=True, dtype=float)
        X = pd.concat([sub[[treatment]].astype(float), fe], axis=1)
        X = sm.add_constant(X)
        m = OLS(sub[outcome].astype(float), X).fit(cov_type="cluster",
                                      cov_kwds={"groups": sub[entity]})
        beta = m.params[treatment]
        se = m.bse[treatment]
        t_val = m.tvalues[treatment]
        p_val = m.pvalues[treatment]
        ci = m.conf_int().loc[treatment]
        n = int(m.nobs)

    elif model_type == "twfe":
        fe = pd.get_dummies(sub[entity], prefix="fe", drop_first=True, dtype=float)
        yr = pd.get_dummies(sub[time].astype(str), prefix="yr", drop_first=True, dtype=float)
        X = pd.concat([sub[[treatment]].astype(float), fe, yr], axis=1)
        X = sm.add_constant(X)
        m = OLS(sub[outcome].astype(float), X).fit(cov_type="cluster",
                                      cov_kwds={"groups": sub[entity]})
        beta = m.params[treatment]
        se = m.bse[treatment]
        t_val = m.tvalues[treatment]
        p_val = m.pvalues[treatment]
        ci = m.conf_int().loc[treatment]
        n = int(m.nobs)

    elif model_type == "first_diff":
        sub = sub.sort_values([entity, time])
        sub["dy"] = sub.groupby(entity)[outcome].diff().astype(float)
        sub["dx"] = sub.groupby(entity)[treatment].diff().astype(float)
        fd = sub.dropna(subset=["dy", "dx"])
        if len(fd) < 50:
            return None
        X = sm.add_constant(fd[["dx"]].astype(float))
        m = OLS(fd["dy"].astype(float), X).fit(cov_type="cluster",
                                  cov_kwds={"groups": fd[entity]})
        beta = m.params["dx"]
        se = m.bse["dx"]
        t_val = m.tvalues["dx"]
        p_val = m.pvalues["dx"]
        ci = m.conf_int().loc["dx"]
        n = int(m.nobs)
        outcome_std = fd["dy"].std()
        treatment_std = fd["dx"].std()
    else:
        return None

    # Effect sizes
    cohen_d = beta / outcome_std if outcome_std > 0 else np.nan
    std_beta = beta * treatment_std / outcome_std if outcome_std > 0 and treatment_std > 0 else np.nan
    # Partial eta-squared from t-value
    df_resid = n - 2  # approximate
    partial_eta2 = t_val ** 2 / (t_val ** 2 + df_resid)

    return {
        "beta": beta,
        "se": se,
        "t": t_val,
        "p": p_val,
        "ci_lo": ci.iloc[0],
        "ci_hi": ci.iloc[1],
        "cohen_d": cohen_d,
        "std_beta": std_beta,
        "partial_eta2": partial_eta2,
        "n": n,
    }


def sig_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


def main():
    print("=" * 70)
    print("Effect Size Summary Table (Review 14)")
    print("=" * 70)

    panel = load_panel()
    if panel is None:
        print("ERROR: No panel data found.")
        return

    # Define all model specifications
    specifications = []

    # ── Macro-level models ──
    macro_outcomes = [
        ("depression_prevalence", "Depression prevalence"),
        ("suicide", "Suicide rate"),
    ]
    if "depression_dalys" in panel.columns and panel["depression_dalys"].notna().sum() > 100:
        macro_outcomes.append(("depression_dalys", "Depression DALYs"))

    macro_treatments = [
        ("ad_proxy", "AdProxy"),
        ("internet", "Internet %"),
    ]

    model_types = [
        ("pooled", "Pooled OLS"),
        ("entity_fe", "Entity FE"),
        ("twfe", "TWFE"),
        ("first_diff", "First Difference"),
    ]

    for outcome_col, outcome_label in macro_outcomes:
        if outcome_col not in panel.columns or panel[outcome_col].notna().sum() < 100:
            continue
        for treatment_col, treatment_label in macro_treatments:
            if treatment_col not in panel.columns:
                continue
            for model_type, model_label in model_types:
                specifications.append({
                    "level": "Macro",
                    "outcome": outcome_label,
                    "outcome_col": outcome_col,
                    "treatment": treatment_label,
                    "treatment_col": treatment_col,
                    "model": model_label,
                    "model_type": model_type,
                })

    # Run all specifications
    rows = []
    for spec in specifications:
        result = run_model(
            panel,
            spec["outcome_col"],
            spec["treatment_col"],
            model_type=spec["model_type"],
        )
        if result is None:
            continue

        rows.append({
            "Level": spec["level"],
            "Outcome": spec["outcome"],
            "Treatment": spec["treatment"],
            "Model": spec["model"],
            "β": result["beta"],
            "SE": result["se"],
            "t": result["t"],
            "p": result["p"],
            "95% CI lower": result["ci_lo"],
            "95% CI upper": result["ci_hi"],
            "Cohen's d": result["cohen_d"],
            "Std. β": result["std_beta"],
            "Partial η²": result["partial_eta2"],
            "N": result["n"],
        })

    # ── Add micro-level results (hardcoded from paper) ──
    micro_results = [
        {
            "Level": "Micro", "Outcome": "PHQ-9 (NHANES)", "Treatment": "Exercise",
            "Model": "OLS + covariates", "β": -1.42, "SE": 0.29, "t": -4.97,
            "p": 7e-7, "95% CI lower": -1.98, "95% CI upper": -0.86,
            "Cohen's d": 0.17, "Std. β": np.nan, "Partial η²": 0.006, "N": 5032,
        },
        {
            "Level": "Micro", "Outcome": "Cantril (ATUS)", "Treatment": "Exercise min",
            "Model": "OLS + covariates", "β": 0.0016, "SE": 0.00032, "t": 4.97,
            "p": 7e-7, "95% CI lower": 0.0009, "95% CI upper": 0.0022,
            "Cohen's d": 0.17, "Std. β": np.nan, "Partial η²": 0.001, "N": 21736,
        },
        {
            "Level": "Micro", "Outcome": "Cantril (ATUS)", "Treatment": "Passive leisure min",
            "Model": "OLS + covariates", "β": -0.0009, "SE": 0.0001, "t": -9.33,
            "p": 1e-20, "95% CI lower": -0.0010, "95% CI upper": -0.0007,
            "Cohen's d": -0.13, "Std. β": np.nan, "Partial η²": 0.004, "N": 21736,
        },
        {
            "Level": "Micro", "Outcome": "Cantril (ATUS)", "Treatment": "Exercise×Passive",
            "Model": "OLS + interaction", "β": -2.4e-7, "SE": 4e-6, "t": -0.06,
            "p": 0.95, "95% CI lower": -8e-6, "95% CI upper": 8e-6,
            "Cohen's d": 0.0, "Std. β": np.nan, "Partial η²": 0.0, "N": 21736,
        },
    ]
    rows.extend(micro_results)

    if not rows:
        print("No results generated.")
        return

    df_results = pd.DataFrame(rows)

    # ── Print summary ──
    print(f"\n{'Level':<8} {'Outcome':<25} {'Treatment':<18} {'Model':<16} "
          f"{'β':>10} {'p':>10} {'d':>8} {'N':>8}")
    print("─" * 110)
    d_col = "Cohen's d"
    for _, row in df_results.iterrows():
        stars = sig_stars(row["p"])
        print(f"  {row['Level']:<6} {row['Outcome']:<25} {row['Treatment']:<18} {row['Model']:<16} "
              f"{row['β']:>+10.6f} {row['p']:>10.4f}{stars:<3} {row[d_col]:>8.4f} {int(row['N']):>8,}")

    # ── Save CSV ──
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "effect_size_table.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # ── Save LaTeX ──
    tex_path = os.path.join(RESULTS_DIR, "effect_size_table.tex")
    with open(tex_path, "w") as f:
        f.write("% Auto-generated effect size table\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Comprehensive effect size summary across all model specifications.}\n")
        f.write("\\label{tab:effect-sizes}\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{llllrrrrr}\n")
        f.write("\\toprule\n")
        f.write("Level & Outcome & Treatment & Model & $\\beta$ & $p$ & Cohen's $d$ & $\\eta^2_p$ & $N$ \\\\\n")
        f.write("\\midrule\n")
        for _, row in df_results.iterrows():
            stars = sig_stars(row["p"])
            d_val = row[d_col]
            eta_val = row["Partial η²"]
            f.write(f"{row['Level']} & {row['Outcome']} & {row['Treatment']} & {row['Model']} & "
                    f"{row['β']:+.4f}{stars} & {row['p']:.4f} & {d_val:.3f} & "
                    f"{eta_val:.4f} & {int(row['N']):,} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"Saved: {tex_path}")

    # ── Save JSON ──
    json_path = os.path.join(RESULTS_DIR, "effect_size_table.json")
    df_results.to_json(json_path, orient="records", indent=2)
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
