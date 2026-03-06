#!/usr/bin/env python3
"""
ATUS Wellbeing Module: 共変量投入版 + サーベイウェイト適用版
対応セクション: Appendix A.2 (共変量投入版・ウェイト適用版)

ベースライン分析（02_atus_wellbeing_analysis.py）との比較:
  1. 共変量なし（ベースライン）
  2. 共変量投入（年齢・性別・教育・所得）
  3. サーベイウェイト適用（TUFNWGTP）
  4. 共変量 + ウェイト（フルモデル）

加法モデル vs 交互作用モデルの結論が頑健かを検証。

依存: pandas, numpy, scipy, statsmodels
データ: data/atus/ (download_atus.py で取得)
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "atus")


def load_atus():
    """Load ATUS activity, wellbeing, and respondent files."""
    # Try multiple possible file patterns
    act_candidates = [
        "atusact_0324.dat",
        "atusact-0324.dat",
        "atusact_0313.dat",
        "atusact-0313.dat",
        "atusact_0324.csv",
        "atusact_0313.csv",
    ]
    wb_candidates = [
        "wbresp_1013.dat",
        "wbresp-1013.dat",
        "atuswb_0313.dat",
        "atuswb-0313.dat",
        "wbresp_1013.csv",
        "atuswb_0313.csv",
    ]
    resp_candidates = [
        "atusresp_0324.dat",
        "atusresp-0324.dat",
        "atusresp_0313.dat",
        "atusresp-0313.dat",
        "atusresp_0324.csv",
        "atusresp_0313.csv",
    ]

    def find_file(candidates, label):
        for f in candidates:
            path = os.path.join(DATA_DIR, f)
            if os.path.exists(path):
                return path
        print(f"ERROR: {label} file not found in {DATA_DIR}")
        print(f"  Tried: {candidates}")
        print("  Run: python data/download_atus.py first")
        sys.exit(1)

    act_path = find_file(act_candidates, "Activity")
    wb_path = find_file(wb_candidates, "Wellbeing")
    resp_path = find_file(resp_candidates, "Respondent")

    print(f"  Activity: {act_path}")
    print(f"  Wellbeing: {wb_path}")
    print(f"  Respondent: {resp_path}")

    # Read (auto-detect separator)
    for sep in [',', '\t', '|']:
        try:
            act = pd.read_csv(act_path, sep=sep, low_memory=False)
            if len(act.columns) > 2:
                break
        except:
            continue

    for sep in [',', '\t', '|']:
        try:
            wb = pd.read_csv(wb_path, sep=sep, low_memory=False)
            if len(wb.columns) > 2:
                break
        except:
            continue

    for sep in [',', '\t', '|']:
        try:
            resp = pd.read_csv(resp_path, sep=sep, low_memory=False)
            if len(resp.columns) > 2:
                break
        except:
            continue

    return act, wb, resp


def construct_time_variables(act):
    """Aggregate activity minutes into 3 categories per respondent."""
    # Activity code mapping (6-digit TUCASEID + TUACTIVITY_N)
    act["code"] = act.get("TRCODE", act.get("TUTIER1CODE", 0))
    act["minutes"] = act.get("TUACTDUR24", act.get("DURATION", 0))

    # Category definitions (see Appendix A.2)
    passive_codes = [120303, 120306]  # TV + leisure PC
    cognitive_codes = list(range(120101, 120102)) + \
                      list(range(120201, 120300)) + \
                      list(range(120307, 120314)) + \
                      list(range(120401, 120500))
    exercise_codes = list(range(130101, 130200))

    grouped = act.groupby("TUCASEID").apply(
        lambda g: pd.Series({
            "passive_min": g.loc[g["code"].isin(passive_codes), "minutes"].sum(),
            "cognitive_min": g.loc[g["code"].isin(cognitive_codes), "minutes"].sum(),
            "exercise_min": g.loc[g["code"].isin(exercise_codes), "minutes"].sum(),
        })
    ).reset_index()

    return grouped


def prepare_dataset(act, wb, resp):
    """Merge and construct analysis-ready dataset."""
    time_vars = construct_time_variables(act)

    # Merge with wellbeing
    id_col = "TUCASEID"
    df = time_vars.merge(wb[[id_col, "WBLADDER", "WEGESSION"]], on=id_col, how="inner")
    df = df.rename(columns={"WBLADDER": "cantril"})

    # Merge with respondent demographics
    demo_cols = [id_col, "TEAGE", "TESEX", "PEEDUCA", "HEFAMINC", "TUFNWGTP"]
    available = [c for c in demo_cols if c in resp.columns]
    df = df.merge(resp[available], on=id_col, how="left")

    # Covariates
    if "TEAGE" in df.columns:
        df["age"] = df["TEAGE"]
    if "TESEX" in df.columns:
        df["female"] = (df["TESEX"] == 2).astype(int)
    if "PEEDUCA" in df.columns:
        df["education"] = df["PEEDUCA"]  # 31-46 scale
    if "HEFAMINC" in df.columns:
        df["income_cat"] = df["HEFAMINC"]  # 1-16 categorical
    if "TUFNWGTP" in df.columns:
        df["weight"] = df["TUFNWGTP"]

    # Binary indicators
    df["has_exercise"] = (df["exercise_min"] > 0).astype(int)
    df["has_cognitive"] = (df["cognitive_min"] > 0).astype(int)
    df["has_processing"] = ((df["has_exercise"] == 1) | (df["has_cognitive"] == 1)).astype(int)

    return df


def run_model(df, y_col, x_cols, weight_col=None, label=""):
    """Run OLS or WLS and print results."""
    import statsmodels.api as sm

    sub = df.dropna(subset=[y_col] + x_cols)
    if weight_col:
        sub = sub.dropna(subset=[weight_col])
    X = sm.add_constant(sub[x_cols].values)
    y = sub[y_col].values

    if weight_col:
        model = sm.WLS(y, X, weights=sub[weight_col].values)
    else:
        model = sm.OLS(y, X)

    result = model.fit(cov_type="HC1")

    print(f"\n  [{label}] N={len(sub):,}, R²={result.rsquared:.4f}, AIC={result.aic:.1f}")
    for i, name in enumerate(["const"] + x_cols):
        print(f"    {name:25s} β={result.params[i]:8.4f}  t={result.tvalues[i]:7.2f}  p={result.pvalues[i]:.2e}")

    return result


def main():
    print("=" * 60)
    print("ATUS Wellbeing: 共変量投入版 + ウェイト適用版")
    print("=" * 60)

    print("\nLoading data...")
    try:
        act, wb, resp = load_atus()
    except Exception as e:
        print(f"\nERROR loading data: {e}")
        print("This script requires ATUS data files.")
        print("Run: python data/download_atus.py")
        print("\nAlternatively, if data format differs, adjust load_atus().")
        return

    print("Constructing variables...")
    df = prepare_dataset(act, wb, resp)
    print(f"Analytic sample: N = {len(df):,}")

    # ---- Core variables ----
    base_x = ["exercise_min", "cognitive_min", "passive_min"]
    interaction_x = base_x + ["exercise_x_cognitive"]
    df["exercise_x_cognitive"] = df["exercise_min"] * df["cognitive_min"]

    # Covariate sets
    demo_vars = []
    if "age" in df.columns:
        demo_vars.append("age")
    if "female" in df.columns:
        demo_vars.append("female")
    if "education" in df.columns:
        demo_vars.append("education")
    if "income_cat" in df.columns:
        demo_vars.append("income_cat")

    has_weights = "weight" in df.columns
    has_covariates = len(demo_vars) > 0

    # ---- Comparison: 4 specifications ----
    print("\n" + "=" * 60)
    print("Additive model comparison across specifications")
    print("=" * 60)

    # 1. Baseline (no covariates, no weights)
    print("\n--- Specification 1: No covariates, no weights (baseline) ---")
    r1_add = run_model(df, "cantril", base_x, label="Additive")
    r1_int = run_model(df, "cantril", interaction_x, label="Interaction")
    print(f"  → ΔAIC = {r1_add.aic - r1_int.aic:.1f} (positive = additive preferred)")

    # 2. With covariates
    if has_covariates:
        print(f"\n--- Specification 2: + Covariates ({', '.join(demo_vars)}) ---")
        r2_add = run_model(df, "cantril", base_x + demo_vars, label="Additive+cov")
        r2_int = run_model(df, "cantril", interaction_x + demo_vars, label="Interaction+cov")
        print(f"  → ΔAIC = {r2_add.aic - r2_int.aic:.1f}")

    # 3. With weights only
    if has_weights:
        print("\n--- Specification 3: Weighted (TUFNWGTP), no covariates ---")
        r3_add = run_model(df, "cantril", base_x, weight_col="weight", label="Additive+wt")
        r3_int = run_model(df, "cantril", interaction_x, weight_col="weight", label="Interaction+wt")
        print(f"  → ΔAIC = {r3_add.aic - r3_int.aic:.1f}")

    # 4. Full: covariates + weights
    if has_covariates and has_weights:
        print(f"\n--- Specification 4: Full (covariates + weights) ---")
        r4_add = run_model(df, "cantril", base_x + demo_vars, weight_col="weight", label="Full additive")
        r4_int = run_model(df, "cantril", interaction_x + demo_vars, weight_col="weight", label="Full interaction")
        print(f"  → ΔAIC = {r4_add.aic - r4_int.aic:.1f}")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY: 加法モデル vs 交互作用モデルの頑健性")
    print("=" * 60)
    print("交互作用が全仕様で非有意（p > 0.05）かつΔAIC > 0であれば、")
    print("加法モデルの優位性はウェイト・共変量の選択に対して頑健。")
    print("係数の大きさの変動はSES交絡の程度を示す。")


if __name__ == "__main__":
    main()
