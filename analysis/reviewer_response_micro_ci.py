#!/usr/bin/env python3
"""
Reviewer Response: Micro-Level Confidence Intervals & Spline Nonlinearity Check
================================================================================
Addresses Stanford Reviewer Questions 3 & 4:

  Q3: Effect sizes with 95% confidence intervals for all predictors
  Q4: Spline nonlinearity check (restricted cubic spline) for continuous
      predictors, with F-test and AIC comparison

Analyses:
  A) NHANES 2017-2018 — PHQ-9 ~ exercise + covariates
     - OLS with β + 95% CI for each predictor
     - Cohen's d + 95% CI for exercise effect
     - Spline note: exercise is binary → spline not applicable

  B) ATUS Wellbeing Module 2010-2013 — Cantril ~ time-use + covariates
     - OLS with β + 95% CI for each predictor
     - Cohen's d + 95% CI for exercise vs no-exercise
     - Restricted cubic spline (3 knots) for exercise_minutes
     - Restricted cubic spline (3 knots) for passive_minutes
     - F-test for nonlinearity, AIC comparison

Output:
  - JSON results → results/reviewer_micro_ci.json
  - Formatted summary table (stdout)

Usage:
  python analysis/reviewer_response_micro_ci.py
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from patsy import dmatrix

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# Paths
# ============================================================
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
NHANES_DIR = os.path.join(BASE_DIR, "data", "nhanes")
ATUS_DIR = os.path.join(BASE_DIR, "data", "atus")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# Utility: Cohen's d with 95% CI (Hedges' approximation)
# ============================================================
def cohens_d_ci(group1, group2, alpha=0.05):
    """
    Compute Cohen's d (group1 - group2) and its 95% CI using
    the non-central t distribution approach.
    """
    n1, n2 = len(group1), len(group2)
    m1, m2 = group1.mean(), group2.mean()
    s1, s2 = group1.std(ddof=1), group2.std(ddof=1)

    # Pooled SD
    sp = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    d = (m1 - m2) / sp

    # SE of d (Hedges & Olkin, 1985)
    se_d = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))

    z = stats.norm.ppf(1 - alpha / 2)
    ci_lo = d - z * se_d
    ci_hi = d + z * se_d

    return {
        "d": float(d),
        "se": float(se_d),
        "ci_lower": float(ci_lo),
        "ci_upper": float(ci_hi),
        "n1": int(n1),
        "n2": int(n2),
    }


def ols_with_ci(X, y, var_names, alpha=0.05):
    """
    Run OLS using statsmodels and extract β, SE, t, p, 95% CI for each predictor.
    Returns (result_object, list_of_dicts).
    """
    model = sm.OLS(y, X, missing="drop")
    res = model.fit(cov_type="HC1")  # robust SEs
    ci = res.conf_int(alpha=alpha)

    rows = []
    for i, name in enumerate(var_names):
        rows.append({
            "variable": name,
            "beta": float(res.params[i]),
            "se": float(res.bse[i]),
            "t": float(res.tvalues[i]),
            "p": float(res.pvalues[i]),
            "ci_lower": float(ci[i, 0]),
            "ci_upper": float(ci[i, 1]),
        })
    return res, rows


# ############################################################
#  PART A: NHANES 2017-2018
# ############################################################
def run_nhanes():
    print("=" * 72)
    print("PART A: NHANES 2017-2018 — PHQ-9 ~ exercise + covariates")
    print("=" * 72)

    # ------ Load ------
    required = ["DEMO_J.XPT", "DPQ_J.XPT", "PAQ_J.XPT", "BMX_J.XPT", "HIQ_J.XPT"]
    missing = [f for f in required if not os.path.exists(os.path.join(NHANES_DIR, f))]
    if missing:
        print(f"  ERROR: missing files in {NHANES_DIR}: {missing}")
        return None

    demo = pd.read_sas(os.path.join(NHANES_DIR, "DEMO_J.XPT"), format="xport")
    dpq  = pd.read_sas(os.path.join(NHANES_DIR, "DPQ_J.XPT"), format="xport")
    paq  = pd.read_sas(os.path.join(NHANES_DIR, "PAQ_J.XPT"), format="xport")
    bmx  = pd.read_sas(os.path.join(NHANES_DIR, "BMX_J.XPT"), format="xport")
    hiq  = pd.read_sas(os.path.join(NHANES_DIR, "HIQ_J.XPT"), format="xport")

    df = demo.merge(dpq, on="SEQN", how="inner")
    df = df.merge(paq, on="SEQN", how="inner")
    df = df.merge(bmx[["SEQN", "BMXBMI"]], on="SEQN", how="left")
    df = df.merge(hiq[["SEQN", "HIQ011"]], on="SEQN", how="left")

    # PHQ-9
    phq_cols = [f"DPQ0{i}0" for i in range(1, 10)]
    for col in phq_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").round()
        df[col] = df[col].replace({7: np.nan, 9: np.nan})
        df[col] = df[col].where(df[col].isin([0, 1, 2, 3]))
    df["phq9"] = df[phq_cols].sum(axis=1, min_count=len(phq_cols))

    # Exercise (binary: leisure vigorous OR moderate)
    df["exercise"] = ((df["PAQ650"] == 1) | (df["PAQ665"] == 1)).astype(int)

    # Covariates
    df["age"] = pd.to_numeric(df["RIDAGEYR"], errors="coerce")
    df["female"] = (df["RIAGENDR"] == 2).astype(int)
    df["education"] = df["DMDEDUC2"].replace({7: np.nan, 9: np.nan})
    df["poverty_income_ratio"] = pd.to_numeric(df["INDFMPIR"], errors="coerce")
    df["insured"] = np.where(df["HIQ011"] == 1, 1, np.where(df["HIQ011"] == 2, 0, np.nan))
    df["bmi"] = pd.to_numeric(df["BMXBMI"], errors="coerce")

    # Filter
    keep_vars = ["phq9", "exercise", "age", "female", "education",
                 "poverty_income_ratio", "insured", "bmi"]
    df = df[df["age"] >= 18].copy()
    df = df.dropna(subset=keep_vars)
    df = df[df["education"].isin([1, 2, 3, 4, 5])]
    print(f"  Analytic sample: N = {len(df):,}")

    # ------ OLS with 95% CI ------
    y = df["phq9"].values
    var_names = ["intercept", "exercise", "age", "female", "education",
                 "poverty_income_ratio", "insured", "bmi"]
    X = sm.add_constant(df[["exercise", "age", "female", "education",
                            "poverty_income_ratio", "insured", "bmi"]].values)

    res, coef_rows = ols_with_ci(X, y, var_names)

    print(f"\n  OLS: PHQ-9 ~ exercise + age + female + education + PIR + insured + BMI")
    print(f"  N = {int(res.nobs):,}, R² = {res.rsquared:.4f}, AIC = {res.aic:.1f}")
    print(f"\n  {'Variable':<25} {'β':>8} {'SE':>8} {'t':>8} {'p':>10} {'95% CI':>20}")
    print("  " + "-" * 82)
    for r in coef_rows:
        sig = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else ""
        print(f"  {r['variable']:<25} {r['beta']:>8.4f} {r['se']:>8.4f} "
              f"{r['t']:>8.2f} {r['p']:>10.2e} [{r['ci_lower']:>8.4f}, {r['ci_upper']:>8.4f}] {sig}")

    # ------ Cohen's d for exercise ------
    ex_yes = df.loc[df["exercise"] == 1, "phq9"]
    ex_no  = df.loc[df["exercise"] == 0, "phq9"]
    d_info = cohens_d_ci(ex_no, ex_yes)  # no-exercise minus exercise (positive = exercise is protective)
    print(f"\n  Cohen's d (no-exercise minus exercise, positive = exercise protective):")
    print(f"    d = {d_info['d']:.3f}  [{d_info['ci_lower']:.3f}, {d_info['ci_upper']:.3f}]")
    print(f"    N(exercise=1) = {d_info['n2']:,}, N(exercise=0) = {d_info['n1']:,}")

    # ------ Spline note ------
    print(f"\n  SPLINE CHECK:")
    print(f"    Exercise is a binary indicator (leisure vigorous or moderate activity: yes/no).")
    print(f"    NHANES PAQ module asks 'Did you do any vigorous/moderate recreational")
    print(f"    activities?' (PAQ650/PAQ665) as yes/no → no continuous dose available")
    print(f"    in this dataset for a spline specification.")
    print(f"    Conclusion: Spline test is NOT APPLICABLE for NHANES exercise variable.")

    return {
        "n": int(len(df)),
        "r_squared": float(res.rsquared),
        "aic": float(res.aic),
        "coefficients": coef_rows,
        "cohens_d_exercise": d_info,
        "spline_note": (
            "Exercise is binary (PAQ650/PAQ665: yes/no). "
            "No continuous physical activity dose variable is available in this "
            "NHANES module for a spline specification. Spline test not applicable."
        ),
    }


# ############################################################
#  PART B: ATUS Wellbeing Module 2010-2013
# ############################################################
def run_atus():
    print("\n" + "=" * 72)
    print("PART B: ATUS 2010-2013 — Cantril ~ time-use + covariates")
    print("=" * 72)

    # ------ Load ------
    def _find(candidates):
        for f in candidates:
            p = os.path.join(ATUS_DIR, f)
            if os.path.exists(p):
                return p
        return None

    sum_path = _find(["atussum_0324.dat", "atussum-0324.dat",
                       "atussum_0323.dat", "atussum-0323.dat"])
    wb_path  = _find(["wbresp_1013.dat", "wbresp-1013.dat"])

    if sum_path is None or wb_path is None:
        print(f"  ERROR: ATUS data files not found in {ATUS_DIR}")
        return None

    print(f"  Summary: {sum_path}")
    print(f"  Wellbeing: {wb_path}")

    wb = pd.read_csv(wb_path, low_memory=False)
    wb.columns = [c.upper() for c in wb.columns]
    atus = pd.read_csv(sum_path, low_memory=False)
    atus.columns = [c.upper() for c in atus.columns]

    # Filter to WB module years
    atus_wb = atus[atus["TUYEAR"].isin([2010, 2012, 2013])].copy()

    # Merge
    cantril_col = "WECANTRIL" if "WECANTRIL" in wb.columns else "WBLADDER"
    wb_cols = ["TUCASEID", cantril_col]
    if "WEGENHTH" in wb.columns:
        wb_cols.append("WEGENHTH")
    df = atus_wb.merge(wb[wb_cols], on="TUCASEID", how="inner")

    # Activity variables (from T-columns in summary file)
    # Passive: TV + leisure computer
    passive_cols = ["T120303", "T120306"]
    df["passive_min"] = df[[c for c in passive_cols if c in df.columns]].sum(axis=1)

    # Active cognitive leisure
    active_cols = [
        "T120101", "T120201", "T120202",
        "T120307", "T120308", "T120309", "T120310",
        "T120311", "T120312", "T120313",
        "T120401", "T120402", "T120403", "T120404", "T120405",
    ]
    df["active_cognitive_min"] = df[[c for c in active_cols if c in df.columns]].sum(axis=1)

    # Exercise
    exercise_cols = [f"T1301{i:02d}" for i in range(1, 30)]
    df["exercise_min"] = df[[c for c in exercise_cols if c in df.columns]].sum(axis=1)

    # Wellbeing
    df["cantril"] = pd.to_numeric(df[cantril_col], errors="coerce")
    df.loc[df["cantril"] < 0, "cantril"] = np.nan

    # Demographics
    df["age"] = pd.to_numeric(df["TEAGE"], errors="coerce")
    df["female"] = (pd.to_numeric(df["TESEX"], errors="coerce") == 2).astype(float)
    df["education"] = pd.to_numeric(df["PEEDUCA"], errors="coerce")
    # TRERNWA: weekly earnings (cents * 100), -1 = not available
    df["income"] = pd.to_numeric(df.get("TRERNWA", pd.Series(dtype=float)), errors="coerce")
    df.loc[df["income"] < 0, "income"] = np.nan
    # Convert from cents to dollars (TRERNWA is in cents * 100)
    df["income"] = df["income"] / 100.0

    # Clean
    clean = df.dropna(subset=["cantril", "age", "female", "education"]).copy()
    clean = clean[(clean["cantril"] >= 0) & (clean["cantril"] <= 10)]

    # For models with income: use subset that has income, but run main model without it
    has_income = clean["income"].notna().sum()
    print(f"  Analytic sample: N = {len(clean):,}")
    print(f"  Observations with income data: {has_income:,}")
    print(f"  Cantril mean = {clean['cantril'].mean():.2f}, SD = {clean['cantril'].std():.2f}")
    print(f"  Exercise: mean = {clean['exercise_min'].mean():.1f} min, "
          f"zero-rate = {100 * (clean['exercise_min'] == 0).mean():.1f}%")
    print(f"  Passive: mean = {clean['passive_min'].mean():.1f} min, "
          f"zero-rate = {100 * (clean['passive_min'] == 0).mean():.1f}%")
    print(f"  Active cognitive: mean = {clean['active_cognitive_min'].mean():.1f} min, "
          f"zero-rate = {100 * (clean['active_cognitive_min'] == 0).mean():.1f}%")

    # ---- B1: Main OLS with 95% CI ----
    print(f"\n  --- B1: OLS with 95% CIs ---")
    var_names_main = ["intercept", "exercise_min", "active_cognitive_min",
                      "passive_min", "age", "female", "education"]
    X_main = sm.add_constant(clean[["exercise_min", "active_cognitive_min",
                                     "passive_min", "age", "female", "education"]].values)
    y = clean["cantril"].values

    res_main, coef_main = ols_with_ci(X_main, y, var_names_main)
    print(f"  N = {int(res_main.nobs):,}, R² = {res_main.rsquared:.4f}, AIC = {res_main.aic:.1f}")
    print(f"\n  {'Variable':<25} {'β':>9} {'SE':>8} {'t':>8} {'p':>10} {'95% CI':>22}")
    print("  " + "-" * 85)
    for r in coef_main:
        sig = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else ""
        print(f"  {r['variable']:<25} {r['beta']:>9.5f} {r['se']:>8.5f} "
              f"{r['t']:>8.2f} {r['p']:>10.2e} [{r['ci_lower']:>9.5f}, {r['ci_upper']:>9.5f}] {sig}")

    # Also run with income for the subset that has it
    sub_inc = clean.dropna(subset=["income"]).copy()
    aic_with_income = None
    coef_income = None
    if len(sub_inc) > 100:
        print(f"\n  --- B1b: OLS with income covariate (N = {len(sub_inc):,}) ---")
        var_names_inc = ["intercept", "exercise_min", "active_cognitive_min",
                         "passive_min", "age", "female", "education", "income"]
        X_inc = sm.add_constant(sub_inc[["exercise_min", "active_cognitive_min",
                                          "passive_min", "age", "female",
                                          "education", "income"]].values)
        y_inc = sub_inc["cantril"].values
        res_inc, coef_income = ols_with_ci(X_inc, y_inc, var_names_inc)
        aic_with_income = float(res_inc.aic)
        print(f"  N = {int(res_inc.nobs):,}, R² = {res_inc.rsquared:.4f}, AIC = {res_inc.aic:.1f}")
        for r in coef_income:
            sig = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else ""
            print(f"  {r['variable']:<25} {r['beta']:>9.5f} {r['se']:>8.5f} "
                  f"{r['t']:>8.2f} {r['p']:>10.2e} [{r['ci_lower']:>9.5f}, {r['ci_upper']:>9.5f}] {sig}")

    # ---- B2: Cohen's d for exercise vs no-exercise ----
    print(f"\n  --- B2: Cohen's d for exercise (any vs none) ---")
    has_ex = clean.loc[clean["exercise_min"] > 0, "cantril"]
    no_ex  = clean.loc[clean["exercise_min"] == 0, "cantril"]
    d_ex = cohens_d_ci(has_ex, no_ex)
    print(f"  d = {d_ex['d']:.3f}  [{d_ex['ci_lower']:.3f}, {d_ex['ci_upper']:.3f}]")
    print(f"  N(exercise>0) = {d_ex['n1']:,}, N(exercise=0) = {d_ex['n2']:,}")
    print(f"  Mean Cantril: exercise = {has_ex.mean():.3f}, no exercise = {no_ex.mean():.3f}")

    # ---- B3: Spline nonlinearity check — exercise_minutes ----
    print(f"\n  --- B3: Restricted Cubic Spline — exercise_minutes (3 knots) ---")

    # Subset to exercisers only for spline (since ~75% have 0 minutes,
    # also run on full sample with 0s)
    # Full sample spline
    ex_vals = clean["exercise_min"].values
    # Use patsy to create restricted cubic spline basis (natural spline / cr)
    # 3 knots placed at 10th, 50th, 90th percentiles of non-zero values
    nonzero_ex = ex_vals[ex_vals > 0]
    if len(nonzero_ex) > 50:
        knots_ex = np.percentile(nonzero_ex, [10, 50, 90])
        print(f"  Knots for exercise_min (10th, 50th, 90th of nonzero): "
              f"{knots_ex[0]:.0f}, {knots_ex[1]:.0f}, {knots_ex[2]:.0f}")

        # Spline basis using patsy cr() — restricted cubic spline
        try:
            spline_basis_ex = dmatrix(
                f"cr(exercise_min, knots=[{knots_ex[0]}, {knots_ex[1]}, {knots_ex[2]}]) "
                f"+ active_cognitive_min + passive_min + age + female + education",
                data=clean,
                return_type="dataframe",
            )
            spline_names_ex = list(spline_basis_ex.columns)

            res_spline_ex = sm.OLS(y, spline_basis_ex.values).fit(cov_type="HC1")
            aic_spline_ex = float(res_spline_ex.aic)

            # F-test for nonlinearity: compare linear model vs spline model
            # The spline model has extra df from the nonlinear spline terms
            # We test: are the extra spline terms jointly significant?
            # Linear model has 1 df for exercise; spline has df_spline for exercise
            # RSS comparison
            rss_linear = float(res_main.ssr)
            rss_spline = float(res_spline_ex.ssr)
            df_linear = int(res_main.df_model) + 1  # total params including intercept
            df_spline = int(res_spline_ex.df_model) + 1
            n_obs = int(res_main.nobs)

            extra_df = df_spline - df_linear
            if extra_df > 0:
                f_stat = ((rss_linear - rss_spline) / extra_df) / (rss_spline / (n_obs - df_spline))
                p_f = 1 - stats.f.cdf(f_stat, extra_df, n_obs - df_spline)
            else:
                f_stat = np.nan
                p_f = np.nan

            print(f"  Linear model AIC = {res_main.aic:.1f}")
            print(f"  Spline model AIC = {aic_spline_ex:.1f}")
            print(f"  ΔAIC (linear - spline) = {res_main.aic - aic_spline_ex:.1f} "
                  f"(positive = spline preferred)")
            print(f"  F-test for nonlinearity: F({extra_df}, {n_obs - df_spline}) = {f_stat:.3f}, "
                  f"p = {p_f:.4f}")
            if p_f < 0.05:
                print(f"  → Significant nonlinearity detected (p < 0.05).")
            else:
                print(f"  → No significant nonlinearity (p >= 0.05). Linear specification adequate.")

            spline_ex_result = {
                "knots": [float(k) for k in knots_ex],
                "aic_linear": float(res_main.aic),
                "aic_spline": aic_spline_ex,
                "delta_aic": float(res_main.aic - aic_spline_ex),
                "f_stat": float(f_stat) if not np.isnan(f_stat) else None,
                "f_df1": int(extra_df),
                "f_df2": int(n_obs - df_spline),
                "f_p": float(p_f) if not np.isnan(p_f) else None,
                "nonlinear_significant": bool(p_f < 0.05) if not np.isnan(p_f) else None,
            }
        except Exception as e:
            print(f"  ERROR in exercise spline: {e}")
            spline_ex_result = {"error": str(e)}
    else:
        print(f"  Insufficient non-zero exercise observations for spline.")
        spline_ex_result = {"error": "Insufficient non-zero observations"}

    # ---- B4: Spline nonlinearity check — passive_minutes ----
    print(f"\n  --- B4: Restricted Cubic Spline — passive_minutes (3 knots) ---")

    nonzero_passive = clean.loc[clean["passive_min"] > 0, "passive_min"].values
    if len(nonzero_passive) > 50:
        knots_pas = np.percentile(clean["passive_min"].values, [25, 50, 75])
        print(f"  Knots for passive_min (25th, 50th, 75th): "
              f"{knots_pas[0]:.0f}, {knots_pas[1]:.0f}, {knots_pas[2]:.0f}")

        try:
            spline_basis_pas = dmatrix(
                f"exercise_min + active_cognitive_min "
                f"+ cr(passive_min, knots=[{knots_pas[0]}, {knots_pas[1]}, {knots_pas[2]}]) "
                f"+ age + female + education",
                data=clean,
                return_type="dataframe",
            )

            res_spline_pas = sm.OLS(y, spline_basis_pas.values).fit(cov_type="HC1")
            aic_spline_pas = float(res_spline_pas.aic)

            rss_spline_pas = float(res_spline_pas.ssr)
            df_spline_pas = int(res_spline_pas.df_model) + 1
            extra_df_pas = df_spline_pas - df_linear if 'df_linear' in dir() else df_spline_pas - (len(var_names_main))
            n_obs_p = int(res_spline_pas.nobs)

            if extra_df_pas > 0:
                f_stat_pas = ((rss_linear - rss_spline_pas) / extra_df_pas) / \
                             (rss_spline_pas / (n_obs_p - df_spline_pas))
                p_f_pas = 1 - stats.f.cdf(f_stat_pas, extra_df_pas, n_obs_p - df_spline_pas)
            else:
                f_stat_pas = np.nan
                p_f_pas = np.nan

            print(f"  Linear model AIC = {res_main.aic:.1f}")
            print(f"  Spline model AIC = {aic_spline_pas:.1f}")
            print(f"  ΔAIC (linear - spline) = {res_main.aic - aic_spline_pas:.1f} "
                  f"(positive = spline preferred)")
            print(f"  F-test for nonlinearity: F({extra_df_pas}, {n_obs_p - df_spline_pas}) = "
                  f"{f_stat_pas:.3f}, p = {p_f_pas:.4f}")
            if p_f_pas < 0.05:
                print(f"  → Significant nonlinearity detected (p < 0.05).")
            else:
                print(f"  → No significant nonlinearity (p >= 0.05). Linear specification adequate.")

            spline_pas_result = {
                "knots": [float(k) for k in knots_pas],
                "aic_linear": float(res_main.aic),
                "aic_spline": aic_spline_pas,
                "delta_aic": float(res_main.aic - aic_spline_pas),
                "f_stat": float(f_stat_pas) if not np.isnan(f_stat_pas) else None,
                "f_df1": int(extra_df_pas),
                "f_df2": int(n_obs_p - df_spline_pas),
                "f_p": float(p_f_pas) if not np.isnan(p_f_pas) else None,
                "nonlinear_significant": bool(p_f_pas < 0.05) if not np.isnan(p_f_pas) else None,
            }
        except Exception as e:
            print(f"  ERROR in passive spline: {e}")
            spline_pas_result = {"error": str(e)}
    else:
        print(f"  Insufficient non-zero passive observations for spline.")
        spline_pas_result = {"error": "Insufficient non-zero observations"}

    # ---- B5: Combined spline model (both exercise and passive) ----
    print(f"\n  --- B5: Combined Spline Model (exercise + passive splines) ---")
    try:
        spline_basis_both = dmatrix(
            f"cr(exercise_min, knots=[{knots_ex[0]}, {knots_ex[1]}, {knots_ex[2]}]) "
            f"+ active_cognitive_min "
            f"+ cr(passive_min, knots=[{knots_pas[0]}, {knots_pas[1]}, {knots_pas[2]}]) "
            f"+ age + female + education",
            data=clean,
            return_type="dataframe",
        )
        res_spline_both = sm.OLS(y, spline_basis_both.values).fit(cov_type="HC1")
        aic_spline_both = float(res_spline_both.aic)

        rss_spline_both = float(res_spline_both.ssr)
        df_spline_both = int(res_spline_both.df_model) + 1
        extra_df_both = df_spline_both - df_linear
        n_obs_b = int(res_spline_both.nobs)

        if extra_df_both > 0:
            f_stat_both = ((rss_linear - rss_spline_both) / extra_df_both) / \
                          (rss_spline_both / (n_obs_b - df_spline_both))
            p_f_both = 1 - stats.f.cdf(f_stat_both, extra_df_both, n_obs_b - df_spline_both)
        else:
            f_stat_both = np.nan
            p_f_both = np.nan

        print(f"  Linear model AIC = {res_main.aic:.1f}")
        print(f"  Combined spline AIC = {aic_spline_both:.1f}")
        print(f"  ΔAIC (linear - combined spline) = {res_main.aic - aic_spline_both:.1f}")
        print(f"  F-test: F({extra_df_both}, {n_obs_b - df_spline_both}) = {f_stat_both:.3f}, "
              f"p = {p_f_both:.4f}")

        spline_both_result = {
            "aic_combined_spline": aic_spline_both,
            "delta_aic": float(res_main.aic - aic_spline_both),
            "f_stat": float(f_stat_both) if not np.isnan(f_stat_both) else None,
            "f_df1": int(extra_df_both),
            "f_df2": int(n_obs_b - df_spline_both),
            "f_p": float(p_f_both) if not np.isnan(p_f_both) else None,
        }
    except Exception as e:
        print(f"  ERROR in combined spline: {e}")
        spline_both_result = {"error": str(e)}

    # ---- Summary table ----
    print(f"\n  {'='*72}")
    print(f"  ATUS AIC COMPARISON SUMMARY")
    print(f"  {'='*72}")
    print(f"  {'Model':<45} {'AIC':>10} {'ΔAIC':>10}")
    print(f"  {'-'*65}")
    print(f"  {'Linear (baseline)':<45} {res_main.aic:>10.1f} {'---':>10}")
    if 'aic_spline_ex' in dir():
        print(f"  {'Spline on exercise_min (3 knots)':<45} {aic_spline_ex:>10.1f} "
              f"{res_main.aic - aic_spline_ex:>+10.1f}")
    if 'aic_spline_pas' in dir():
        print(f"  {'Spline on passive_min (3 knots)':<45} {aic_spline_pas:>10.1f} "
              f"{res_main.aic - aic_spline_pas:>+10.1f}")
    if 'aic_spline_both' in dir():
        print(f"  {'Combined splines (exercise + passive)':<45} {aic_spline_both:>10.1f} "
              f"{res_main.aic - aic_spline_both:>+10.1f}")

    return {
        "n": int(len(clean)),
        "r_squared": float(res_main.rsquared),
        "aic_linear": float(res_main.aic),
        "coefficients": coef_main,
        "coefficients_with_income": coef_income,
        "aic_with_income": aic_with_income,
        "cohens_d_exercise": d_ex,
        "spline_exercise": spline_ex_result,
        "spline_passive": spline_pas_result,
        "spline_combined": spline_both_result,
    }


# ############################################################
#  MAIN
# ############################################################
def main():
    print("*" * 72)
    print("  Reviewer Response: Effect Sizes with CIs & Spline Nonlinearity")
    print("  Questions 3 & 4 — Stanford Reviewer")
    print("*" * 72)

    results = {}

    # Part A: NHANES
    nhanes_results = run_nhanes()
    if nhanes_results is not None:
        results["nhanes"] = nhanes_results

    # Part B: ATUS
    atus_results = run_atus()
    if atus_results is not None:
        results["atus"] = atus_results

    # Save JSON
    json_path = os.path.join(RESULTS_DIR, "reviewer_micro_ci.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n{'='*72}")
    print(f"Results saved: {json_path}")
    print(f"{'='*72}")

    # ---- Final summary ----
    print(f"\n{'*'*72}")
    print("EXECUTIVE SUMMARY FOR REVIEWER")
    print(f"{'*'*72}")

    if "nhanes" in results:
        r = results["nhanes"]
        ex_coef = next(c for c in r["coefficients"] if c["variable"] == "exercise")
        print(f"\n  NHANES (N = {r['n']:,}):")
        print(f"    Exercise β = {ex_coef['beta']:.4f} [{ex_coef['ci_lower']:.4f}, {ex_coef['ci_upper']:.4f}]")
        print(f"    Cohen's d = {r['cohens_d_exercise']['d']:.3f} "
              f"[{r['cohens_d_exercise']['ci_lower']:.3f}, {r['cohens_d_exercise']['ci_upper']:.3f}]")
        print(f"    Spline: {r['spline_note']}")

    if "atus" in results:
        r = results["atus"]
        ex_coef = next(c for c in r["coefficients"] if c["variable"] == "exercise_min")
        pas_coef = next(c for c in r["coefficients"] if c["variable"] == "passive_min")
        print(f"\n  ATUS (N = {r['n']:,}):")
        print(f"    Exercise_min β = {ex_coef['beta']:.5f} [{ex_coef['ci_lower']:.5f}, {ex_coef['ci_upper']:.5f}]")
        print(f"    Passive_min  β = {pas_coef['beta']:.5f} [{pas_coef['ci_lower']:.5f}, {pas_coef['ci_upper']:.5f}]")
        print(f"    Cohen's d (exercise any vs none) = {r['cohens_d_exercise']['d']:.3f} "
              f"[{r['cohens_d_exercise']['ci_lower']:.3f}, {r['cohens_d_exercise']['ci_upper']:.3f}]")

        se = r.get("spline_exercise", {})
        if "f_p" in se and se["f_p"] is not None:
            print(f"    Spline exercise: F = {se['f_stat']:.3f}, p = {se['f_p']:.4f}, "
                  f"ΔAIC = {se['delta_aic']:+.1f}")
        sp = r.get("spline_passive", {})
        if "f_p" in sp and sp["f_p"] is not None:
            print(f"    Spline passive:  F = {sp['f_stat']:.3f}, p = {sp['f_p']:.4f}, "
                  f"ΔAIC = {sp['delta_aic']:+.1f}")


if __name__ == "__main__":
    main()
