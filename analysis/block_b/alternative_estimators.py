#!/usr/bin/env python3
"""
Block B: 代替推定量による頑健性検証
対応: Stanford Agentic Reviewer Q1, Q2 (both rounds)

実装する3つの代替推定量:
  1. Driscoll-Kraay SE (cross-sectional dependence + serial correlation)
  2. Country-specific linear trends in FE
  3. First-difference GMM (Arellano-Bond style)

目的: TWFE / 1階差分の主要結果が、
  - cross-sectional dependenceに対して頑健か
  - 国固有のトレンドを入れても生き残るか
  - 動的パネルバイアス（Nickell bias）を補正しても維持されるか

依存: pandas, numpy, statsmodels, linearmodels (pip install linearmodels)
データ: data/macro/panel_with_inactivity.csv (or equivalent)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "macro")


def load_panel():
    """Load macro panel data. Adjust path/columns as needed."""
    candidates = [
        os.path.join(DATA_DIR, "panel_with_inactivity.csv"),
        os.path.join(DATA_DIR, "panel_merged.csv"),
        os.path.join(DATA_DIR, "macro_panel.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"  Loaded: {path} ({len(df):,} rows)")
            return df

    print(f"ERROR: No panel data found in {DATA_DIR}")
    print("  Expected one of: panel_with_inactivity.csv, panel_merged.csv, macro_panel.csv")
    print("  See data/README_data.md for assembly instructions.")
    sys.exit(1)


def ensure_panel_structure(df):
    """Ensure entity/time index and required columns."""
    # Identify entity and time columns
    entity_col = None
    time_col = None

    for c in ["country", "entity", "iso3", "country_code"]:
        if c in df.columns:
            entity_col = c
            break

    for c in ["year", "time"]:
        if c in df.columns:
            time_col = c
            break

    if entity_col is None or time_col is None:
        print("ERROR: Cannot identify entity/time columns")
        print(f"  Available: {list(df.columns)}")
        sys.exit(1)

    df = df.rename(columns={entity_col: "entity", time_col: "year"})

    # Required variable columns (flexible naming)
    var_map = {}
    for target, candidates in [
        ("dep", ["dep_rate", "depression", "depression_rate", "dep"]),
        ("suicide", ["suicide_rate", "suicide", "sui_rate"]),
        ("proxy", ["ad_proxy", "proxy", "adproxy"]),
        ("gdp", ["gdp_pc", "gdp", "gdp_per_capita", "NY.GDP.PCAP.CD"]),
        ("internet", ["internet", "internet_pct", "IT.NET.USER.ZS"]),
    ]:
        for c in candidates:
            if c in df.columns:
                var_map[target] = c
                break

    for key, col in var_map.items():
        if key != col:
            df = df.rename(columns={col: key})

    return df


# ====================================================================
# 1. Driscoll-Kraay Standard Errors
# ====================================================================

def driscoll_kraay_fe(df, y_var, x_vars, max_lag=3):
    """
    Entity FE regression with Driscoll-Kraay standard errors.

    DK-SE are robust to:
      - Heteroscedasticity
      - Arbitrary serial correlation (up to max_lag)
      - Cross-sectional (spatial) dependence

    Implementation: Newey-West on time-series of cross-sectional averages
    of the score (moment conditions), following Hoechle (2007).
    """
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant

    sub = df.dropna(subset=[y_var] + x_vars + ["entity", "year"]).copy()

    # Entity demeaning (within transformation)
    for col in [y_var] + x_vars:
        group_mean = sub.groupby("entity")[col].transform("mean")
        sub[f"{col}_dm"] = sub[col] - group_mean

    y = sub[f"{y_var}_dm"].values
    X = sub[[f"{v}_dm" for v in x_vars]].values

    # OLS on demeaned data
    model = OLS(y, X)
    result_ols = model.fit()
    resid = result_ols.resid
    N = sub["entity"].nunique()
    T = sub["year"].nunique()

    # Score matrix: X_it * e_it
    scores = X * resid[:, np.newaxis]
    sub["_resid"] = resid
    for j, v in enumerate(x_vars):
        sub[f"_score_{j}"] = scores[:, j]

    # Time-series of cross-sectional sums of scores
    k = len(x_vars)
    S_t = np.zeros((T, k))
    years = sorted(sub["year"].unique())
    for t_idx, yr in enumerate(years):
        mask = sub["year"] == yr
        S_t[t_idx] = sub.loc[mask, [f"_score_{j}" for j in range(k)]].sum().values

    # Newey-West on S_t
    # Γ_0
    S_centered = S_t  # already mean-zero under OLS
    Gamma_0 = S_centered.T @ S_centered / T

    # Γ_j for j=1..max_lag
    V_dk = Gamma_0.copy()
    for lag in range(1, max_lag + 1):
        w = 1 - lag / (max_lag + 1)  # Bartlett kernel
        Gamma_j = S_centered[lag:].T @ S_centered[:-lag] / T
        V_dk += w * (Gamma_j + Gamma_j.T)

    # Bread: (X'X)^{-1}
    XtX_inv = np.linalg.inv(X.T @ X)

    # Sandwich
    V_beta = XtX_inv @ (V_dk * T) @ XtX_inv

    se_dk = np.sqrt(np.diag(V_beta))
    t_dk = result_ols.params / se_dk
    from scipy.stats import t as t_dist
    p_dk = 2 * (1 - t_dist.cdf(np.abs(t_dk), df=T - k - 1))

    return result_ols.params, se_dk, t_dk, p_dk, len(sub)


# ====================================================================
# 2. Country-Specific Linear Trends
# ====================================================================

def fe_with_country_trends(df, y_var, x_vars):
    """
    Entity FE + entity-specific linear time trends.

    Model: y_it = α_i + δ_i·t + β·X_it + ε_it

    This absorbs country-specific linear trends that could be
    confounded with the treatment (internet/ad proxy expansion).
    """
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant

    sub = df.dropna(subset=[y_var] + x_vars + ["entity", "year"]).copy()

    # Create entity dummies × time interaction
    # More efficient: demean after partialling out entity + entity×time
    entities = sub["entity"].unique()
    entity_map = {e: i for i, e in enumerate(entities)}
    sub["entity_id"] = sub["entity"].map(entity_map)

    # Within-entity demeaning of y and X, also removing entity-specific trend
    def detrend_group(g):
        t = g["year"].values - g["year"].values.mean()
        for col in [y_var] + x_vars:
            vals = g[col].values
            # Remove group mean and group-specific linear trend
            mean_val = vals.mean()
            # OLS of col on t within group
            if len(t) > 1 and np.std(t) > 0:
                slope = np.sum((t - t.mean()) * (vals - mean_val)) / np.sum((t - t.mean())**2)
            else:
                slope = 0
            g[f"{col}_dt"] = vals - mean_val - slope * t
        return g

    sub = sub.groupby("entity", group_keys=False).apply(detrend_group)

    y = sub[f"{y_var}_dt"].values
    X = sub[[f"{v}_dt" for v in x_vars]].values

    model = OLS(y, X)
    result = model.fit(cov_type="HC1")

    return result.params, result.bse, result.tvalues, result.pvalues, len(sub)


# ====================================================================
# 3. First-Difference GMM (simplified Arellano-Bond)
# ====================================================================

def first_difference_with_lags(df, y_var, x_vars, n_lags=1):
    """
    First-difference model with lagged dependent variable.

    Δy_it = ρ·Δy_{i,t-1} + β·ΔX_it + Δε_it

    Uses 2SLS with y_{i,t-2} as instrument for Δy_{i,t-1}
    (Arellano-Bond logic, simplified).

    NOTE: Full system GMM requires linearmodels or specialized package.
    This provides the core identification check.
    """
    from statsmodels.regression.linear_model import OLS
    from scipy.stats import t as t_dist

    sub = df.dropna(subset=[y_var] + x_vars + ["entity", "year"]).copy()
    sub = sub.sort_values(["entity", "year"])

    # First differences
    for col in [y_var] + x_vars:
        sub[f"d_{col}"] = sub.groupby("entity")[col].diff()

    # Lagged dependent variable (in differences)
    sub[f"d_{y_var}_lag1"] = sub.groupby("entity")[f"d_{y_var}"].shift(1)

    # Instrument: y_{t-2} level
    sub[f"{y_var}_lag2"] = sub.groupby("entity")[y_var].shift(2)

    sub = sub.dropna(subset=[f"d_{y_var}", f"d_{y_var}_lag1", f"{y_var}_lag2"] +
                     [f"d_{v}" for v in x_vars])

    if len(sub) < 50:
        print("  WARNING: Too few observations for FD-IV estimation")
        return None, None, None, None, len(sub)

    # Stage 1: Δy_{t-1} = π·y_{t-2} + γ·ΔX + v
    y_stage1 = sub[f"d_{y_var}_lag1"].values
    Z = sub[[f"{y_var}_lag2"] + [f"d_{v}" for v in x_vars]].values

    model_s1 = OLS(y_stage1, Z)
    res_s1 = model_s1.fit()
    fitted_dy_lag = res_s1.fittedvalues

    # First-stage F-statistic
    f_stat_first_stage = res_s1.fvalue
    print(f"  First-stage F = {f_stat_first_stage:.2f} (>10 = strong instrument)")

    # Stage 2: Δy_t = ρ·Δŷ_{t-1} + β·ΔX + ε
    y_stage2 = sub[f"d_{y_var}"].values
    X_stage2 = np.column_stack([fitted_dy_lag] +
                                [sub[f"d_{v}"].values for v in x_vars])

    model_s2 = OLS(y_stage2, X_stage2)
    res_s2 = model_s2.fit(cov_type="HC1")

    var_names = [f"Δ{y_var}(t-1)"] + [f"Δ{v}" for v in x_vars]

    return res_s2.params, res_s2.bse, res_s2.tvalues, res_s2.pvalues, len(sub)


# ====================================================================
# Main
# ====================================================================

def print_results(label, var_names, params, se, t, p, n):
    print(f"\n  [{label}] N = {n:,}")
    if params is None:
        print("  (estimation failed)")
        return
    for i, name in enumerate(var_names):
        sig = ""
        if p[i] < 0.001:
            sig = "***"
        elif p[i] < 0.01:
            sig = "**"
        elif p[i] < 0.05:
            sig = "*"
        print(f"    {name:25s}  β={params[i]:10.6f}  SE={se[i]:10.6f}  t={t[i]:7.2f}  p={p[i]:.4f} {sig}")


def main():
    print("=" * 70)
    print(" Block B: Alternative Estimators for Robustness")
    print(" (Driscoll-Kraay SE / Country Trends / FD-IV)")
    print("=" * 70)

    print("\nLoading panel data...")
    df = load_panel()
    df = ensure_panel_structure(df)

    available = [c for c in ["dep", "suicide", "proxy", "gdp", "internet"] if c in df.columns]
    print(f"  Available variables: {available}")
    print(f"  Entities: {df['entity'].nunique()}, Years: {df['year'].nunique()}")

    if "proxy" not in df.columns and "internet" in df.columns and "gdp" in df.columns:
        df["proxy"] = df["internet"] * df["gdp"] / 1000
        print("  Constructed proxy = internet × gdp / 1000")

    # ============================================================
    # Analysis 1: Depression ~ proxy (core model)
    # ============================================================
    if "dep" in df.columns and "proxy" in df.columns:
        x_vars = ["proxy"]
        if "gdp" in df.columns:
            x_vars.append("gdp")

        print("\n" + "=" * 70)
        print(f" Outcome: Depression | X = {x_vars}")
        print("=" * 70)

        print("\n--- 1. Driscoll-Kraay SE (max_lag=3) ---")
        params, se, t, p, n = driscoll_kraay_fe(df, "dep", x_vars, max_lag=3)
        print_results("DK-SE", x_vars, params, se, t, p, n)

        print("\n--- 2. Country-Specific Linear Trends ---")
        params, se, t, p, n = fe_with_country_trends(df, "dep", x_vars)
        print_results("FE+Trends", x_vars, params, se, t, p, n)

        print("\n--- 3. First-Difference IV (Arellano-Bond logic) ---")
        params, se, t, p, n = first_difference_with_lags(df, "dep", x_vars, n_lags=1)
        var_names_fd = ["Δdep(t-1)"] + [f"Δ{v}" for v in x_vars]
        print_results("FD-IV", var_names_fd, params, se, t, p, n)

    # ============================================================
    # Analysis 2: Suicide ~ proxy
    # ============================================================
    if "suicide" in df.columns and "proxy" in df.columns:
        x_vars_s = ["proxy"]
        if "gdp" in df.columns:
            x_vars_s.append("gdp")

        print("\n" + "=" * 70)
        print(f" Outcome: Suicide | X = {x_vars_s}")
        print("=" * 70)

        print("\n--- 1. Driscoll-Kraay SE (max_lag=3) ---")
        params, se, t, p, n = driscoll_kraay_fe(df, "suicide", x_vars_s, max_lag=3)
        print_results("DK-SE", x_vars_s, params, se, t, p, n)

        print("\n--- 2. Country-Specific Linear Trends ---")
        params, se, t, p, n = fe_with_country_trends(df, "suicide", x_vars_s)
        print_results("FE+Trends", x_vars_s, params, se, t, p, n)

        print("\n--- 3. First-Difference IV ---")
        params, se, t, p, n = first_difference_with_lags(df, "suicide", x_vars_s, n_lags=1)
        var_names_fd = ["Δsuicide(t-1)"] + [f"Δ{v}" for v in x_vars_s]
        print_results("FD-IV", var_names_fd, params, se, t, p, n)

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print(" INTERPRETATION GUIDE")
    print("=" * 70)
    print("""
  Driscoll-Kraay SE:
    If significance survives → results robust to cross-sectional dependence.
    If SE inflates substantially → standard TWFE inference was anticonservative.

  Country-Specific Trends:
    If proxy remains significant → effect is not driven by country-specific
    linear trends (e.g., secular improvements in healthcare).
    If proxy loses significance → the effect may be confounded with trends.

  First-Difference IV:
    If Δproxy significant after instrumenting Δy(t-1) → survives Nickell bias.
    First-stage F > 10 indicates strong instrument.
    If ρ (Δy lag) is large → strong persistence; dynamic structure matters.
    """)


if __name__ == "__main__":
    main()
