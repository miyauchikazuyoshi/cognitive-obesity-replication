#!/usr/bin/env python3
"""
Ad Proxy Validation: Internet × GDP/capita vs External Benchmarks
================================================================
Validates the ad_proxy = Internet(%) × GDP/capita / 1000 construct
against available external benchmarks.

Three layers of validation:
  Layer 1 — Cross-sectional rank correlation (Spearman ρ)
            with known country-level digital ad spend per capita
  Layer 2 — Time-series validation (2022-2024): does the proxy
            track actual ad spend *changes* over time?
  Layer 3 — Bootstrap CI + leave-one-out robustness

External data sources (all publicly available):
  - eMarketer/Insider Intelligence via B2BHouse (2024 update):
    29 countries × 3 years (2022-2024), total digital ad spend in B USD
  - Oberlo/eMarketer (2023): Top 10 + growth rates
  - GroupM "This Year Next Year" 2023/2024, IAB 2023
  - Statista Digital Market Outlook (2024) summaries
  - DataReportal Digital 2025: per-user social media ad spend

This script does NOT require proprietary WARC/GroupM microdata.

Inputs:
  - data/macro/panel_merged.csv (or compatible macro panel)

Outputs:
  - results/proxy_validation.json
  - results/figures/proxy_validation_rank.png
  - results/figures/proxy_validation_timeseries.png
  - console output
"""

from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "macro")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# ============================================================
# DATA LOADING
# ============================================================
def load_panel() -> pd.DataFrame:
    candidates = [
        "panel_merged.csv",
        "panel_with_inactivity.csv",
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
    gdp_col = find(["gdp", "gdp_pc", "gdp_per_capita", "ny.gdp.pcap.cd",
                     "gdp_current_usd", "gdp_ppp_2017"])
    pop_col = find(["population", "pop", "sp.pop.totl"])

    if country_col is None or year_col is None:
        print("ERROR: cannot identify country/year columns.")
        sys.exit(1)

    rename[country_col] = "country"
    rename[year_col] = "year"
    if internet_col: rename[internet_col] = "internet"
    if gdp_col: rename[gdp_col] = "gdp"
    if pop_col: rename[pop_col] = "population"
    return df.rename(columns=rename)


def ensure_proxy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    internet = pd.to_numeric(out.get("internet"), errors="coerce")
    gdp = pd.to_numeric(out.get("gdp"), errors="coerce")
    internet_pct = internet * 100.0 if internet.max(skipna=True) <= 1.5 else internet
    out["internet_pct"] = internet_pct
    out["ad_proxy"] = internet_pct * gdp / 1000.0
    return out


# ============================================================
# REFERENCE DATA: Multi-year digital ad spend by country
# ============================================================
# Source: eMarketer/Insider Intelligence via B2BHouse (2024 update)
# + Oberlo/eMarketer (2023) + Statista Digital Market Outlook
# All values: total digital advertising spend in BILLION USD
#
# Citation: "Digital Ad Spend - Statistics and Trends (2024 Update)"
#   B2BHouse / Tamarind, sourcing eMarketer/Insider Intelligence
#   URL: https://www.theb2bhouse.com/digital-ad-spend-statistics-and-trends/

DIGITAL_AD_SPEND_TOTAL = {
    # (country, year): total digital ad spend in billion USD
    # --- Layer 1: B2BHouse/eMarketer multi-year (29 countries) ---
    ("Australia", 2022): 10.66, ("Australia", 2023): 11.38, ("Australia", 2024): 11.95,
    ("China", 2022): 129.59, ("China", 2023): 140.10, ("China", 2024): 153.46,
    ("India", 2022): 3.43, ("India", 2023): 3.99, ("India", 2024): 4.60,
    ("Indonesia", 2022): 1.08, ("Indonesia", 2023): 1.21, ("Indonesia", 2024): 1.33,
    ("Japan", 2022): 20.81, ("Japan", 2023): 22.09, ("Japan", 2024): 23.16,
    ("South Korea", 2022): 6.18, ("South Korea", 2023): 6.44, ("South Korea", 2024): 6.69,
    ("Thailand", 2022): 1.02, ("Thailand", 2023): 1.14, ("Thailand", 2024): 1.22,
    ("New Zealand", 2022): 1.24, ("New Zealand", 2023): 1.30, ("New Zealand", 2024): 1.34,
    ("Singapore", 2022): 0.74, ("Singapore", 2023): 0.81, ("Singapore", 2024): 0.88,
    ("Malaysia", 2022): 0.46, ("Malaysia", 2023): 0.50, ("Malaysia", 2024): 0.53,
    ("Hong Kong", 2022): 1.04, ("Hong Kong", 2023): 1.09, ("Hong Kong", 2024): 1.12,
    ("Philippines", 2022): 0.56, ("Philippines", 2023): 0.61, ("Philippines", 2024): 0.66,
    ("Taiwan", 2022): 1.49, ("Taiwan", 2023): 1.54, ("Taiwan", 2024): 1.59,
    ("Vietnam", 2022): 0.37, ("Vietnam", 2023): 0.40, ("Vietnam", 2024): 0.44,
    ("Argentina", 2022): 0.67, ("Argentina", 2023): 0.80, ("Argentina", 2024): 0.96,
    ("Brazil", 2022): 6.93, ("Brazil", 2023): 7.72, ("Brazil", 2024): 8.72,
    ("Mexico", 2022): 3.16, ("Mexico", 2023): 3.62, ("Mexico", 2024): 4.16,
    ("Peru", 2022): 0.25, ("Peru", 2023): 0.30, ("Peru", 2024): 0.36,
    ("Chile", 2022): 0.57, ("Chile", 2023): 0.66, ("Chile", 2024): 0.78,
    ("Colombia", 2022): 0.65, ("Colombia", 2023): 0.75, ("Colombia", 2024): 0.88,
    ("Canada", 2022): 10.92, ("Canada", 2023): 12.19, ("Canada", 2024): 13.59,
    ("United States", 2022): 248.72, ("United States", 2023): 278.59, ("United States", 2024): 311.46,
    ("Russia", 2022): 2.23, ("Russia", 2023): 2.53, ("Russia", 2024): 2.85,
    ("Turkey", 2022): 0.36, ("Turkey", 2023): 0.40, ("Turkey", 2024): 0.44,
    ("France", 2022): 7.60, ("France", 2023): 8.37, ("France", 2024): 9.29,
    ("Germany", 2022): 11.29, ("Germany", 2023): 12.08, ("Germany", 2024): 13.23,
    ("Italy", 2022): 3.59, ("Italy", 2023): 3.84, ("Italy", 2024): 4.20,
    ("Spain", 2022): 2.93, ("Spain", 2023): 3.19, ("Spain", 2024): 3.54,
    ("United Kingdom", 2022): 30.53, ("United Kingdom", 2023): 33.21, ("United Kingdom", 2024): 36.70,
    # --- Layer 2: Statista/GroupM/IAB estimates for additional countries ---
    # Total market sizes from GroupM TYNY 2023, Statista DMO 2024
    ("Netherlands", 2022): 4.0, ("Netherlands", 2023): 4.4,
    ("Sweden", 2022): 3.2, ("Sweden", 2023): 3.5,
    ("Norway", 2022): 1.8, ("Norway", 2023): 2.0,
    ("Denmark", 2022): 1.9, ("Denmark", 2023): 2.1,
    ("Switzerland", 2022): 2.4, ("Switzerland", 2023): 2.6,
    ("Belgium", 2022): 1.6, ("Belgium", 2023): 1.8,
    ("Austria", 2022): 1.2, ("Austria", 2023): 1.3,
    ("Finland", 2022): 0.9, ("Finland", 2023): 1.0,
    ("Poland", 2022): 2.0, ("Poland", 2023): 2.2,
    ("Czech Republic", 2022): 0.8, ("Czech Republic", 2023): 0.9,
    ("Ireland", 2022): 0.8, ("Ireland", 2023): 0.9,
    ("Israel", 2022): 1.4, ("Israel", 2023): 1.5,
    # --- Layer 3: Emerging markets from Statista/regional reports ---
    ("South Africa", 2022): 0.6, ("South Africa", 2023): 0.7,
    ("Nigeria", 2022): 0.3, ("Nigeria", 2023): 0.35,
    ("Egypt", 2022): 0.2, ("Egypt", 2023): 0.25,
    ("Saudi Arabia", 2022): 1.0, ("Saudi Arabia", 2023): 1.2,
    ("United Arab Emirates", 2022): 0.8, ("United Arab Emirates", 2023): 0.9,
    ("Pakistan", 2022): 0.15, ("Pakistan", 2023): 0.18,
    ("Bangladesh", 2022): 0.08, ("Bangladesh", 2023): 0.10,
}

# Population estimates (millions, ~2022-2023) from World Bank
POPULATION = {
    "United States": 334.9, "China": 1425.9, "India": 1428.6, "Indonesia": 277.5,
    "Pakistan": 240.5, "Bangladesh": 172.9, "Brazil": 216.4, "Nigeria": 223.8,
    "Russia": 144.2, "Mexico": 128.9, "Japan": 124.5, "Germany": 84.5,
    "United Kingdom": 67.7, "France": 68.2, "Italy": 58.9, "Canada": 40.1,
    "South Korea": 51.7, "Spain": 48.1, "Australia": 26.4, "Netherlands": 17.6,
    "Saudi Arabia": 36.9, "Turkey": 85.3, "Switzerland": 8.8, "Taiwan": 23.9,
    "Poland": 37.6, "Thailand": 71.8, "South Africa": 60.4, "Colombia": 52.1,
    "Argentina": 46.7, "Malaysia": 34.3, "Peru": 34.0, "Chile": 19.6,
    "Czech Republic": 10.8, "Belgium": 11.7, "Sweden": 10.5, "Denmark": 5.9,
    "Finland": 5.6, "Norway": 5.5, "Austria": 9.1, "Ireland": 5.2,
    "Israel": 9.8, "Singapore": 5.9, "Hong Kong": 7.5, "New Zealand": 5.2,
    "Philippines": 117.3, "Vietnam": 99.5, "Egypt": 112.7,
    "United Arab Emirates": 9.4, "Bangladesh": 172.9, "Pakistan": 240.5,
}


# ============================================================
# VALIDATION
# ============================================================
def main():
    print("=" * 70)
    print("AD PROXY VALIDATION (Enhanced)")
    print("proxy = Internet(%) × GDP/capita / 1000")
    print("=" * 70)

    df = ensure_proxy(normalize_columns(load_panel()))
    results = {}

    # ---- Build reference dataframe ----
    ref_rows = []
    for (country, year), spend_bn in DIGITAL_AD_SPEND_TOTAL.items():
        pop = POPULATION.get(country)
        if pop:
            ref_rows.append({
                "country": country,
                "year": year,
                "ad_spend_bn": spend_bn,
                "ad_spend_per_capita": spend_bn * 1e9 / (pop * 1e6),
            })
    ref_df = pd.DataFrame(ref_rows)
    n_countries_total = ref_df["country"].nunique()
    n_years = ref_df["year"].nunique()
    print(f"\nReference data: {n_countries_total} countries × {n_years} years")
    print(f"Total data points: {len(ref_df)}")

    # ---- Match with proxy ----
    # Use yearly proxy values (match year-by-year)
    proxy_annual = (
        df.groupby(["country", "year"])["ad_proxy"]
        .mean().reset_index()
    )
    merged = ref_df.merge(proxy_annual, on=["country", "year"], how="inner")
    merged = merged.dropna(subset=["ad_proxy", "ad_spend_per_capita"])
    print(f"Matched: {len(merged)} country-year observations")
    print(f"Unique countries matched: {merged['country'].nunique()}")

    # ==================================================================
    # TEST 1: Cross-sectional rank correlation (2023 snapshot)
    # ==================================================================
    print("\n" + "=" * 70)
    print("TEST 1: Cross-Sectional Rank Correlation (2023)")
    print("=" * 70)

    snap = merged[merged["year"] == 2023].copy()
    if len(snap) < 10:
        snap = merged[merged["year"] == 2022].copy()

    n_cs = len(snap)
    rho_cs, p_cs = stats.spearmanr(snap["ad_proxy"], snap["ad_spend_per_capita"])
    r_loglog, p_loglog = stats.pearsonr(
        np.log(snap["ad_proxy"].clip(lower=0.01)),
        np.log(snap["ad_spend_per_capita"].clip(lower=0.01))
    )
    print(f"N = {n_cs} countries")
    print(f"Spearman ρ = {rho_cs:.4f} (p = {p_cs:.2e})")
    print(f"Pearson r (log-log) = {r_loglog:.4f} (p = {p_loglog:.2e})")

    results["cross_sectional_2023"] = {
        "n_countries": n_cs,
        "spearman_rho": round(float(rho_cs), 4),
        "spearman_p": float(p_cs),
        "pearson_loglog_r": round(float(r_loglog), 4),
        "pearson_loglog_p": float(p_loglog),
        "countries": sorted(snap["country"].tolist()),
    }

    # Rank comparison
    snap["rank_proxy"] = snap["ad_proxy"].rank(ascending=False)
    snap["rank_adspend"] = snap["ad_spend_per_capita"].rank(ascending=False)
    snap["rank_diff"] = (snap["rank_proxy"] - snap["rank_adspend"]).abs()
    print(f"Mean absolute rank difference: {snap['rank_diff'].mean():.1f}")

    # ==================================================================
    # TEST 2: Bootstrap 95% CI for Spearman ρ
    # ==================================================================
    print("\n" + "=" * 70)
    print("TEST 2: Bootstrap 95% CI for Spearman ρ")
    print("=" * 70)

    n_boot = 10000
    boot_rhos = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        idx = rng.choice(len(snap), size=len(snap), replace=True)
        r, _ = stats.spearmanr(
            snap["ad_proxy"].values[idx],
            snap["ad_spend_per_capita"].values[idx]
        )
        boot_rhos.append(r)
    boot_rhos = np.array(boot_rhos)
    ci_lo, ci_hi = np.percentile(boot_rhos, [2.5, 97.5])
    print(f"Bootstrap ρ: median = {np.median(boot_rhos):.4f}")
    print(f"95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")

    results["bootstrap_ci"] = {
        "n_bootstrap": n_boot,
        "median_rho": round(float(np.median(boot_rhos)), 4),
        "ci_95_lower": round(float(ci_lo), 4),
        "ci_95_upper": round(float(ci_hi), 4),
    }

    # ==================================================================
    # TEST 3: Leave-one-out robustness
    # ==================================================================
    print("\n" + "=" * 70)
    print("TEST 3: Leave-One-Out Robustness")
    print("=" * 70)

    loo_rhos = []
    loo_countries = []
    for i in range(len(snap)):
        mask = np.ones(len(snap), dtype=bool)
        mask[i] = False
        r, _ = stats.spearmanr(
            snap["ad_proxy"].values[mask],
            snap["ad_spend_per_capita"].values[mask]
        )
        loo_rhos.append(r)
        loo_countries.append(snap["country"].values[i])

    loo_rhos = np.array(loo_rhos)
    print(f"LOO ρ range: [{loo_rhos.min():.4f}, {loo_rhos.max():.4f}]")
    print(f"LOO ρ mean: {loo_rhos.mean():.4f} ± {loo_rhos.std():.4f}")

    # Most influential countries
    influence = np.abs(loo_rhos - rho_cs)
    top_k = 5
    top_idx = np.argsort(influence)[-top_k:][::-1]
    print(f"\nMost influential countries (removing changes ρ most):")
    for idx in top_idx:
        delta = loo_rhos[idx] - rho_cs
        print(f"  {loo_countries[idx]:<20s}: ρ = {loo_rhos[idx]:.4f} (Δ = {delta:+.4f})")

    results["leave_one_out"] = {
        "rho_min": round(float(loo_rhos.min()), 4),
        "rho_max": round(float(loo_rhos.max()), 4),
        "rho_mean": round(float(loo_rhos.mean()), 4),
        "rho_std": round(float(loo_rhos.std()), 4),
    }

    # ==================================================================
    # TEST 4: Time-series validation (2022 → 2024)
    # ==================================================================
    print("\n" + "=" * 70)
    print("TEST 4: Time-Series Validation (Δproxy tracks Δad_spend)")
    print("=" * 70)

    # Countries with both 2022 and 2024 data
    countries_multi = merged.groupby("country")["year"].nunique()
    countries_3yr = countries_multi[countries_multi >= 2].index.tolist()

    ts_rows = []
    for c in countries_3yr:
        cdata = merged[merged["country"] == c].sort_values("year")
        if len(cdata) >= 2:
            y_first, y_last = cdata["year"].min(), cdata["year"].max()
            row_first = cdata[cdata["year"] == y_first].iloc[0]
            row_last = cdata[cdata["year"] == y_last].iloc[0]
            if row_first["ad_proxy"] > 0 and row_first["ad_spend_per_capita"] > 0:
                ts_rows.append({
                    "country": c,
                    "delta_log_proxy": np.log(row_last["ad_proxy"]) - np.log(row_first["ad_proxy"]),
                    "delta_log_adspend": np.log(row_last["ad_spend_per_capita"]) - np.log(row_first["ad_spend_per_capita"]),
                })

    if len(ts_rows) >= 10:
        ts_df = pd.DataFrame(ts_rows)
        rho_ts, p_ts = stats.spearmanr(ts_df["delta_log_proxy"], ts_df["delta_log_adspend"])
        r_ts, p_r_ts = stats.pearsonr(ts_df["delta_log_proxy"], ts_df["delta_log_adspend"])
        print(f"N = {len(ts_df)} countries with multi-year data")
        print(f"Spearman ρ(Δlog_proxy, Δlog_adspend) = {rho_ts:.4f} (p = {p_ts:.2e})")
        print(f"Pearson r(Δlog_proxy, Δlog_adspend) = {r_ts:.4f} (p = {p_r_ts:.2e})")

        results["time_series"] = {
            "n_countries": len(ts_df),
            "spearman_rho_delta": round(float(rho_ts), 4),
            "spearman_p_delta": float(p_ts),
            "pearson_r_delta": round(float(r_ts), 4),
            "pearson_p_delta": float(p_r_ts),
        }
    else:
        print(f"Insufficient multi-year matches ({len(ts_rows)})")

    # ==================================================================
    # TEST 5: Variance decomposition
    # ==================================================================
    print("\n" + "=" * 70)
    print("TEST 5: Variance Decomposition (Internet vs GDP)")
    print("=" * 70)

    recent = df[(df["year"] >= 2020) & (df["year"] <= 2023)].copy()
    valid = recent.dropna(subset=["internet_pct", "gdp", "ad_proxy"])
    valid = valid[valid["ad_proxy"] > 0]
    if len(valid) > 50:
        log_proxy = np.log(valid["ad_proxy"])
        log_internet = np.log(valid["internet_pct"].clip(lower=0.1))
        log_gdp = np.log(valid["gdp"].clip(lower=100))

        r_internet, _ = stats.pearsonr(log_internet, log_proxy)
        r_gdp, _ = stats.pearsonr(log_gdp, log_proxy)

        print(f"R²(Internet → proxy) = {r_internet**2:.3f}")
        print(f"R²(GDP → proxy)      = {r_gdp**2:.3f}")
        pct_inet = r_internet**2 / (r_internet**2 + r_gdp**2) * 100
        pct_gdp = r_gdp**2 / (r_internet**2 + r_gdp**2) * 100
        print(f"Relative: Internet {pct_inet:.0f}%, GDP {pct_gdp:.0f}%")

        results["variance_decomposition"] = {
            "r2_internet": round(float(r_internet**2), 4),
            "r2_gdp": round(float(r_gdp**2), 4),
            "pct_internet": round(float(pct_inet), 1),
            "pct_gdp": round(float(pct_gdp), 1),
        }

    # ==================================================================
    # TEST 6: Residual proxy (orthogonal to GDP)
    # ==================================================================
    print("\n" + "=" * 70)
    print("TEST 6: Residual Proxy (GDP-orthogonalized)")
    print("=" * 70)

    try:
        import statsmodels.api as sm
        proxy_avg = snap.set_index("country")["ad_proxy"]
        gdp_avg = df[(df["year"] >= 2020) & (df["year"] <= 2023)].groupby("country")["gdp"].mean()
        common = proxy_avg.index.intersection(gdp_avg.dropna().index)
        if len(common) >= 15:
            X = sm.add_constant(np.log(gdp_avg[common].clip(lower=100)))
            y = np.log(proxy_avg[common].clip(lower=0.01))
            resid = sm.OLS(y, X).fit().resid

            adspend_ref = snap.set_index("country")["ad_spend_per_capita"]
            common2 = resid.index.intersection(adspend_ref.index)
            rho_resid, p_resid = stats.spearmanr(resid[common2], adspend_ref[common2])
            print(f"ρ(residual_proxy, ad_spend/cap) = {rho_resid:.4f} (p = {p_resid:.2e})")
            print(f"N = {len(common2)}")
            print("→ Internet component captures ad-market variation beyond GDP" if rho_resid > 0.3 else "")

            results["residual_proxy"] = {
                "spearman_rho": round(float(rho_resid), 4),
                "spearman_p": float(p_resid),
                "n_countries": len(common2),
            }
    except ImportError:
        print("  (statsmodels not available, skipping)")

    # ==================================================================
    # FIGURES
    # ==================================================================
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)

    # Figure 1: Enhanced proxy vs ad spend scatter
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: log-log scatter
    ax = axes[0]
    x = np.log10(snap["ad_proxy"].clip(lower=0.01))
    y = np.log10(snap["ad_spend_per_capita"].clip(lower=0.01))

    # Color by income tier
    colors = []
    for _, row in snap.iterrows():
        pc = row["ad_spend_per_capita"]
        if pc > 200: colors.append("#2563eb")      # high
        elif pc > 20: colors.append("#059669")      # mid
        else: colors.append("#dc2626")              # low
    ax.scatter(x, y, c=colors, alpha=0.7, s=50, edgecolors="white", linewidth=0.5)

    # Label select countries
    label_countries = {"United States", "China", "India", "Japan", "United Kingdom",
                       "Nigeria", "Pakistan", "Brazil", "Germany", "Australia",
                       "South Korea", "Saudi Arabia", "Indonesia", "Singapore"}
    for _, row in snap.iterrows():
        if row["country"] in label_countries or row["rank_diff"] > 10:
            ax.annotate(row["country"],
                       (np.log10(max(row["ad_proxy"], 0.01)),
                        np.log10(max(row["ad_spend_per_capita"], 0.01))),
                       fontsize=6.5, alpha=0.85,
                       xytext=(3, 3), textcoords="offset points")

    z = np.polyfit(x, y, 1)
    xline = np.linspace(x.min() - 0.1, x.max() + 0.1, 100)
    ax.plot(xline, np.polyval(z, xline), "k--", alpha=0.3, linewidth=1)
    ax.set_xlabel("log₁₀(Ad Proxy)", fontsize=10)
    ax.set_ylabel("log₁₀(Digital Ad Spend per capita, USD)", fontsize=10)
    ax.set_title(f"Proxy vs Actual Ad Spend/Capita\nN = {n_cs}, Spearman ρ = {rho_cs:.3f}, "
                 f"95% CI [{ci_lo:.3f}, {ci_hi:.3f}]", fontsize=10, fontweight="bold")

    # Right: rank comparison
    ax = axes[1]
    ax.scatter(snap["rank_proxy"], snap["rank_adspend"],
               c=colors, alpha=0.7, s=50, edgecolors="white", linewidth=0.5)
    ax.plot([0, n_cs + 1], [0, n_cs + 1], "k--", alpha=0.3)

    for _, row in snap.iterrows():
        if row["rank_diff"] > 8:
            ax.annotate(row["country"],
                       (row["rank_proxy"], row["rank_adspend"]),
                       fontsize=6.5, alpha=0.85)

    ax.set_xlabel("Proxy Rank", fontsize=10)
    ax.set_ylabel("Actual Ad Spend/Capita Rank", fontsize=10)
    ax.set_title(f"Rank Comparison\nMean |ΔRank| = {snap['rank_diff'].mean():.1f}", fontsize=10, fontweight="bold")
    ax.invert_xaxis(); ax.invert_yaxis()

    plt.tight_layout()
    fig1_path = os.path.join(FIG_DIR, "proxy_validation_rank.png")
    plt.savefig(fig1_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig1_path}")

    # Figure 2: Time-series proxy trajectories
    fig, ax = plt.subplots(figsize=(10, 6))
    showcase = ["United States", "Japan", "United Kingdom", "Germany",
                "China", "India", "Brazil", "South Korea", "Australia", "Nigeria"]
    cmap = plt.cm.tab10(np.linspace(0, 1, len(showcase)))

    for country, color in zip(showcase, cmap):
        cdata = df[(df["country"] == country) & (df["ad_proxy"].notna())]
        if len(cdata) > 3:
            ax.plot(cdata["year"], cdata["ad_proxy"], label=country,
                    color=color, linewidth=1.5)

    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel("Ad Proxy (Internet% × GDP/cap / 1000)", fontsize=10)
    ax.set_title("Ad Proxy Trajectories — Selected Countries", fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.set_xlim(1990, 2023)
    plt.tight_layout()
    fig2_path = os.path.join(FIG_DIR, "proxy_validation_timeseries.png")
    plt.savefig(fig2_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig2_path}")

    # ==================================================================
    # SUMMARY
    # ==================================================================
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    cs = results.get("cross_sectional_2023", {})
    bs = results.get("bootstrap_ci", {})
    ts = results.get("time_series", {})
    loo = results.get("leave_one_out", {})

    print(f"\n  Cross-sectional:  ρ = {cs.get('spearman_rho', 'N/A')}, "
          f"N = {cs.get('n_countries', 'N/A')}")
    print(f"  Bootstrap 95% CI: [{bs.get('ci_95_lower', 'N/A')}, {bs.get('ci_95_upper', 'N/A')}]")
    print(f"  Leave-one-out:    ρ ∈ [{loo.get('rho_min', 'N/A')}, {loo.get('rho_max', 'N/A')}]")
    if ts:
        print(f"  Time-series Δ:    ρ = {ts.get('spearman_rho_delta', 'N/A')}, "
              f"N = {ts.get('n_countries', 'N/A')}")

    verdict = ("STRONG" if abs(cs.get("spearman_rho", 0)) > 0.85
               else "MODERATE" if abs(cs.get("spearman_rho", 0)) > 0.7
               else "WEAK")
    print(f"\n  VERDICT: {verdict} proxy validity")
    results["verdict"] = verdict

    # Save
    json_path = os.path.join(RESULTS_DIR, "proxy_validation.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {json_path}")


if __name__ == "__main__":
    main()
