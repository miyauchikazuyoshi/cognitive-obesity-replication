#!/usr/bin/env python3
"""
固定効果モデル仕様比較 (Section 2.1 — Table 5相当)
===================================================
Pooled OLS → Country FE → TWFE → TWFE+covariates の段階的投入により
ad_proxy → depression の関係がどの仕様で頑健かを示す。

Models:
  1) Pooled OLS:  depression ~ ad_proxy
  2) Country FE:  depression ~ ad_proxy + country_i
  3) TWFE:        depression ~ ad_proxy + country_i + year_t
  4) TWFE + Cov:  depression ~ ad_proxy + GDP + education + country_i + year_t

Inputs:
  - data/macro/panel_merged.csv (or compatible macro panel)

Outputs:
  - results/block_a_fe_comparison.json
  - console output
"""

from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import statsmodels.api as sm

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "macro")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# DATA LOADING
# ============================================================
def load_panel() -> pd.DataFrame:
    candidates = [
        "panel_with_inactivity.csv",
        "panel_merged.csv",
        "full_panel_rebuilt.csv",
        "macro_panel.csv",
    ]
    for fname in candidates:
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"Loaded panel: {path} (rows={len(df):,})")
            return df

    print(f"ERROR: No macro panel CSV found under {DATA_DIR}")
    print("Expected one of:")
    for fname in candidates:
        print(f"  - data/macro/{fname}")
    print("\nSee data/README_data.md for data acquisition/assembly instructions.")
    sys.exit(1)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {c.lower(): c for c in df.columns}

    def find(candidates):
        for c in candidates:
            if c.lower() in col_map:
                return col_map[c.lower()]
        return None

    rename = {}
    country_col = find(["country", "entity", "location", "country_name"])
    year_col = find(["year", "time"])
    internet_col = find(["internet", "internet_pct", "it.net.user.zs"])
    gdp_col = find(["gdp", "gdp_pc", "gdp_per_capita", "ny.gdp.pcap.cd"])
    dep_col = find(["depression_prevalence", "dep", "dep_rate",
                     "depression_rate", "depression"])
    edu_col = find(["education", "mean_years_schooling", "schooling"])

    if country_col is None or year_col is None:
        print("ERROR: cannot identify country/year columns in panel.")
        sys.exit(1)

    rename[country_col] = "country"
    rename[year_col] = "year"
    if internet_col:
        rename[internet_col] = "internet"
    if gdp_col:
        rename[gdp_col] = "gdp"
    if dep_col:
        rename[dep_col] = "depression_prevalence"
    if edu_col:
        rename[edu_col] = "education"

    df = df.rename(columns=rename)
    return df


# ============================================================
# WITHIN TRANSFORMATION (entity demeaning)
# ============================================================
def demean_by(df: pd.DataFrame, cols: list[str], group: str) -> pd.DataFrame:
    """Group-mean subtraction (within transformation)."""
    out = df.copy()
    for col in cols:
        out[col] = out.groupby(group)[col].transform(lambda x: x - x.mean())
    return out


# ============================================================
# MODEL ESTIMATION
# ============================================================
def estimate_model(
    df: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    country_fe: bool = False,
    year_fe: bool = False,
    label: str = "",
) -> dict:
    """
    Estimate OLS/FE model. Country FE via within transformation,
    year FE via year dummies.
    """
    sub = df.dropna(subset=[y_col] + x_cols).copy()

    if country_fe:
        # Within transformation for country FE
        sub = demean_by(sub, [y_col] + x_cols, "country")

    if year_fe:
        year_dummies = pd.get_dummies(sub["year"], prefix="yr", drop_first=True,
                                       dtype=float)
        if country_fe:
            # Also demean year dummies
            for col in year_dummies.columns:
                sub[col] = year_dummies[col]
            sub = demean_by(sub, list(year_dummies.columns), "country")
            x_all = x_cols + list(year_dummies.columns)
        else:
            for col in year_dummies.columns:
                sub[col] = year_dummies[col]
            x_all = x_cols + list(year_dummies.columns)
    else:
        x_all = x_cols

    X = sm.add_constant(sub[x_all].values)
    y = sub[y_col].values

    model = sm.OLS(y, X)
    # Clustered SE by country (if possible)
    try:
        result = model.fit(cov_type="cluster",
                           cov_kwds={"groups": sub["country"].values})
    except Exception:
        result = model.fit(cov_type="HC1")

    n_obs = int(result.nobs)
    n_countries = sub["country"].nunique()

    # Extract ad_proxy coefficient (index 1 = first regressor after const)
    beta = float(result.params[1])
    se = float(result.bse[1])
    t = float(result.tvalues[1])
    p = float(result.pvalues[1])

    aic = float(result.aic)
    bic = float(result.bic)
    r2 = float(result.rsquared)

    return {
        "label": label,
        "n_obs": n_obs,
        "n_countries": n_countries,
        "beta_proxy": round(beta, 6),
        "se": round(se, 6),
        "t": round(t, 2),
        "p": round(p, 4),
        "r2": round(r2, 4),
        "aic": round(aic, 1),
        "bic": round(bic, 1),
        "country_fe": country_fe,
        "year_fe": year_fe,
    }


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("BLOCK A — 固定効果モデル仕様比較 (Section 2.1, Table 5)")
    print("=" * 70)

    df = normalize_columns(load_panel())

    # -- Ad proxy construction --
    internet_pct = pd.to_numeric(df.get("internet"), errors="coerce")
    gdp = pd.to_numeric(df.get("gdp"), errors="coerce")
    if internet_pct is None or gdp is None:
        print("ERROR: internet or gdp column not found.")
        sys.exit(1)
    if internet_pct.max(skipna=True) <= 1.5:
        internet_pct = internet_pct * 100.0
    df["ad_proxy"] = internet_pct * gdp / 1000.0

    # Ensure numeric
    for col in ["depression_prevalence", "ad_proxy", "gdp", "education"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Scale ad_proxy for readability (per 1000 units)
    df["ad_proxy_k"] = df["ad_proxy"] / 1000.0

    required = ["depression_prevalence", "ad_proxy_k"]
    core = df.dropna(subset=required).copy()
    print(f"\nCore sample: N={len(core):,}, Countries={core['country'].nunique()}, "
          f"Years={core['year'].nunique()}")

    # ---- Model 1: Pooled OLS ----
    m1 = estimate_model(core, "depression_prevalence", ["ad_proxy_k"],
                        country_fe=False, year_fe=False,
                        label="Model 1: Pooled OLS")

    # ---- Model 2: Country FE ----
    m2 = estimate_model(core, "depression_prevalence", ["ad_proxy_k"],
                        country_fe=True, year_fe=False,
                        label="Model 2: Country FE")

    # ---- Model 3: TWFE (Country + Year FE) ----
    m3 = estimate_model(core, "depression_prevalence", ["ad_proxy_k"],
                        country_fe=True, year_fe=True,
                        label="Model 3: TWFE")

    # ---- Model 4: TWFE + Covariates ----
    has_cov = all(c in df.columns for c in ["gdp", "education"])
    m4 = None
    if has_cov:
        # Log GDP for better scaling
        core_cov = core.dropna(subset=["gdp", "education"]).copy()
        core_cov["log_gdp"] = np.log(core_cov["gdp"].clip(lower=100))
        if len(core_cov) > 100:
            m4 = estimate_model(core_cov, "depression_prevalence",
                                ["ad_proxy_k", "log_gdp", "education"],
                                country_fe=True, year_fe=True,
                                label="Model 4: TWFE + Covariates")
        else:
            print("\n  SKIP Model 4: insufficient observations with covariates")

    # ---- Print comparison table ----
    models = [m for m in [m1, m2, m3, m4] if m is not None]

    print(f"\n{'=' * 70}")
    print(f"FIXED EFFECTS MODEL COMPARISON")
    print(f"{'=' * 70}")
    print(f"{'Model':<30s} {'β(proxy)':>10s} {'SE':>10s} {'t':>8s} {'p':>8s} "
          f"{'R²':>7s} {'N':>7s}")
    print("-" * 82)
    for m in models:
        sig = "***" if m["p"] < 0.001 else "**" if m["p"] < 0.01 else "*" if m["p"] < 0.05 else ""
        print(f"  {m['label']:<28s} {m['beta_proxy']:>10.6f} {m['se']:>10.6f} "
              f"{m['t']:>8.2f} {m['p']:>7.4f}{sig} {m['r2']:>7.4f} {m['n_obs']:>7,d}")

    print(f"\n--- AIC / BIC ---")
    for m in models:
        print(f"  {m['label']:<28s} AIC={m['aic']:>10.1f}  BIC={m['bic']:>10.1f}")

    # ---- Interpretation ----
    print(f"\n--- Interpretation ---")
    if len(models) >= 3:
        sign_changes = []
        for i in range(1, len(models)):
            if np.sign(models[i]["beta_proxy"]) != np.sign(models[i-1]["beta_proxy"]):
                sign_changes.append((models[i-1]["label"], models[i]["label"]))
        if sign_changes:
            print("  Sign reversal detected between:")
            for a, b in sign_changes:
                print(f"    {a} → {b}")
            print("  → OVB or Simpson's paradox; FE specification matters.")
        else:
            print("  No sign reversal across specifications.")
            print("  → Direction of proxy effect is robust to FE structure.")

    # ---- Save JSON ----
    output = {"models": models}
    out_path = os.path.join(RESULTS_DIR, "block_a_fe_comparison.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
