#!/usr/bin/env python3
"""
Platform Extraction Phase: Meta ARPU × User Attitudes Divergence Analysis
=========================================================================

Tests whether Meta's increasing monetization intensity (ARPU) correlates with
declining user attitudes toward social media, using publicly available data.

Data sources:
  - Meta 10-K SEC filings (2012-2024): ARPU, DAU, ad impressions
  - Pew Research Center (2018-2024): "Social media mostly good/bad for society"
  - Reuters Digital News Report (2012-2024): Trust in news via Facebook

Connection to Cognitive Obesity:
  Extraction phase = platforms inflate I (information exposure) while suppressing
  C (cognitive control) in L = α₁·I − α₂·C, resulting in net utility loss L
  despite increased time-on-platform.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ---- Paths ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..", "..", "..")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


# ======================================================================
# DATA: Compiled from public SEC filings and survey reports
# ======================================================================

def build_meta_financial_data():
    """
    Meta Platforms 10-K filings (SEC EDGAR).
    ARPU = Average Revenue Per User (worldwide, annual).
    Source: investor.atmeta.com / SEC Form 10-K
    """
    data = {
        "year": [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
        # Worldwide ARPU (USD, annual) — from 10-K filings
        "meta_arpu_world": [5.32, 6.64, 9.45, 11.96, 15.98, 20.21, 24.96, 29.25, 32.03, 40.96, 36.49, 40.02, 48.85],
        # US & Canada ARPU (USD, annual)
        "meta_arpu_us": [20.04, 26.76, 38.63, 49.05, 62.23, 84.41, 111.97, 130.14, 143.47, 175.44, 164.92, 178.36, 218.52],
        # DAU worldwide (millions, Q4 of each year)
        "meta_dau_m": [618, 757, 890, 1038, 1227, 1401, 1523, 1657, 1845, 1929, 1984, 2110, 2200],
        # Ad impressions YoY growth (%)
        "ad_impressions_growth_pct": [None, None, None, None, None, None, 34, 33, 33, 10, 18, 28, 11],
        # Ad price YoY growth (%)
        "ad_price_growth_pct": [None, None, None, None, None, None, -2, -5, -1, 24, -16, 3, 10],
    }
    return pd.DataFrame(data)


def build_pew_attitudes_data():
    """
    Pew Research Center: "Is social media mostly good or mostly bad for society?"
    Source: ATP Wave data + published topline reports
    Note: Not surveyed every year; available years shown.
    """
    data = {
        "year": [2018, 2020, 2022, 2024],
        # % saying "mostly bad for society"
        "pew_bad_pct": [55, 64, 64, 68],
        # % saying "mostly good for society"
        "pew_good_pct": [45, 36, 36, 32],
    }
    return pd.DataFrame(data)


def build_reuters_trust_data():
    """
    Reuters Institute Digital News Report: Trust in news found on social media.
    Global average (all markets covered that year).
    Source: Annual DNR reports 2012-2024
    Note: Scale and question wording evolved; values are approximate midpoints.
    """
    data = {
        "year": [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
        # % who trust news on social media (approximate global average)
        "reuters_trust_social_pct": [32, 27, 24, 23, 23, 24, 22, 23, 22, 20],
        # % who actively avoid news (global average)
        "news_avoidance_pct": [None, None, None, None, 29, 32, 36, 38, 36, 39],
    }
    return pd.DataFrame(data)


def build_platform_events():
    """Key platform events for annotation."""
    return [
        (2017.0, "YouTube\nAdpocalypse"),
        (2018.1, "Facebook MSI\nalgorithm"),
        (2018.3, "Cambridge\nAnalytica"),
        (2021.0, "Apple ATT\n(iOS 14.5)"),
        (2022.0, "Meta stock\n-65%"),
    ]


# ======================================================================
# ANALYSIS
# ======================================================================

def main():
    print("=" * 70)
    print("PILOT: Platform Investment-to-Extraction Phase Analysis")
    print("=" * 70)

    meta = build_meta_financial_data()
    pew = build_pew_attitudes_data()
    reuters = build_reuters_trust_data()
    events = build_platform_events()

    # ---- Analysis 1: ARPU Trend ----
    print("\n" + "-" * 50)
    print("Analysis 1: Meta ARPU Growth Trajectory")
    print("-" * 50)

    # Log-linear regression: ARPU = a * exp(b * year)
    log_arpu = np.log(meta["meta_arpu_us"].values)
    years_centered = meta["year"].values - 2012
    slope, intercept, r, p, se = stats.linregress(years_centered, log_arpu)
    print(f"  US ARPU log-linear trend: growth rate = {slope*100:.1f}%/year")
    print(f"  R² = {r**2:.3f}, p = {p:.2e}")
    print(f"  2012: ${meta['meta_arpu_us'].iloc[0]:.2f} → 2024: ${meta['meta_arpu_us'].iloc[-1]:.2f}")
    print(f"  Ratio: {meta['meta_arpu_us'].iloc[-1]/meta['meta_arpu_us'].iloc[0]:.1f}x in 12 years")

    # DAU growth for comparison
    slope_dau, _, r_dau, _, _ = stats.linregress(years_centered, np.log(meta["meta_dau_m"].values))
    print(f"\n  DAU log-linear growth: {slope_dau*100:.1f}%/year (R²={r_dau**2:.3f})")
    print(f"  ARPU grows {slope/slope_dau:.1f}x faster than DAU → extraction acceleration")

    # ---- Analysis 2: Divergence Index ----
    print("\n" + "-" * 50)
    print("Analysis 2: Engagement-Satisfaction Divergence Index")
    print("-" * 50)

    # Merge datasets on year
    merged = meta.merge(reuters, on="year", how="inner")

    # Standardize: ARPU (proxy for engagement/extraction) vs trust (satisfaction)
    arpu_z = (merged["meta_arpu_us"] - merged["meta_arpu_us"].mean()) / merged["meta_arpu_us"].std()
    trust_z = (merged["reuters_trust_social_pct"] - merged["reuters_trust_social_pct"].mean()) / merged["reuters_trust_social_pct"].std()

    merged["divergence_index"] = arpu_z.values - trust_z.values

    print("  Year  | ARPU(US$) | Trust(%) | z(ARPU) | z(Trust) | DI")
    print("  ------|-----------|----------|---------|----------|-----")
    for _, row in merged.iterrows():
        idx = merged.index.get_loc(_)
        print(f"  {int(row['year'])}  | ${row['meta_arpu_us']:8.2f} | {row['reuters_trust_social_pct']:6.0f}%  | "
              f"{arpu_z.iloc[idx]:+6.2f}  | {trust_z.iloc[idx]:+6.2f}   | {row['divergence_index']:+5.2f}")

    # DI trend
    di_slope, di_int, di_r, di_p, _ = stats.linregress(
        merged["year"].values, merged["divergence_index"].values
    )
    print(f"\n  DI trend: slope = {di_slope:+.3f}/year, R² = {di_r**2:.3f}, p = {di_p:.4f}")
    print(f"  → Divergence is {'increasing' if di_slope > 0 else 'decreasing'} over time")

    # ---- Analysis 3: Pew Attitudes ----
    print("\n" + "-" * 50)
    print("Analysis 3: Pew 'Social Media Bad for Society' Trend")
    print("-" * 50)

    for _, row in pew.iterrows():
        bar = "█" * int(row["pew_bad_pct"] / 2)
        print(f"  {int(row['year'])}: {row['pew_bad_pct']:.0f}% bad | {bar}")

    pew_slope, _, pew_r, pew_p, _ = stats.linregress(pew["year"], pew["pew_bad_pct"])
    print(f"\n  Trend: +{pew_slope:.1f} pp/year (R²={pew_r**2:.3f}, p={pew_p:.3f})")

    # ---- Analysis 4: ARPU growth vs attitude decline correlation ----
    print("\n" + "-" * 50)
    print("Analysis 4: ARPU × Attitudes Cross-Correlation")
    print("-" * 50)

    # Merge ARPU and Pew on year
    arpu_pew = meta[["year", "meta_arpu_us"]].merge(pew[["year", "pew_bad_pct"]], on="year")
    if len(arpu_pew) >= 3:
        r_val, p_val = stats.pearsonr(arpu_pew["meta_arpu_us"], arpu_pew["pew_bad_pct"])
        print(f"  Pearson r(ARPU_US, pew_bad_pct) = {r_val:.3f}, p = {p_val:.3f}")
        print(f"  → {'Significant' if p_val < 0.05 else 'Not significant (small N)'} correlation")
        print(f"  N = {len(arpu_pew)} (limited by Pew survey years)")
    else:
        r_val, p_val = np.nan, np.nan
        print("  Insufficient overlapping years for correlation")

    # ---- Visualization ----
    print("\n" + "-" * 50)
    print("Analysis 5: Visualization")
    print("-" * 50)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Platform Investment → Extraction Phase Transition", fontsize=14, fontweight="bold")

    # Panel A: ARPU trajectory
    ax = axes[0, 0]
    ax.plot(meta["year"], meta["meta_arpu_us"], "o-", color="#1877F2", linewidth=2, markersize=6, label="US & Canada")
    ax.plot(meta["year"], meta["meta_arpu_world"], "s--", color="#1877F2", alpha=0.5, linewidth=1.5, label="Worldwide")
    ax.set_xlabel("Year")
    ax.set_ylabel("ARPU (USD/year)")
    ax.set_title("A) Meta ARPU: Extraction Intensity")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    # Add event annotations
    for yr, label in events:
        if 2012 <= yr <= 2024:
            ax.axvline(yr, color="red", alpha=0.3, linestyle="--")

    # Panel B: Pew attitudes
    ax = axes[0, 1]
    years_pew = pew["year"]
    ax.bar(years_pew - 0.2, pew["pew_bad_pct"], width=0.4, color="#FF4444", alpha=0.7, label="Mostly bad")
    ax.bar(years_pew + 0.2, pew["pew_good_pct"], width=0.4, color="#44AA44", alpha=0.7, label="Mostly good")
    ax.set_xlabel("Year")
    ax.set_ylabel("% of respondents")
    ax.set_title("B) Pew: Social Media Good/Bad for Society")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 80)

    # Panel C: Divergence Index
    ax = axes[1, 0]
    ax.plot(merged["year"], merged["divergence_index"], "o-", color="#8B0000", linewidth=2, markersize=8)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.fill_between(merged["year"], 0, merged["divergence_index"],
                     where=merged["divergence_index"] > 0, alpha=0.2, color="red", label="Extraction > Satisfaction")
    ax.fill_between(merged["year"], 0, merged["divergence_index"],
                     where=merged["divergence_index"] <= 0, alpha=0.2, color="green", label="Satisfaction > Extraction")
    # Trend line
    trend_years = np.linspace(merged["year"].min(), merged["year"].max(), 100)
    ax.plot(trend_years, di_slope * trend_years + di_int, ":", color="#8B0000", alpha=0.5)
    ax.set_xlabel("Year")
    ax.set_ylabel("Divergence Index (z)")
    ax.set_title(f"C) Engagement–Satisfaction Divergence (slope={di_slope:+.3f}/yr)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    # Event annotations
    for yr, label in events:
        if merged["year"].min() <= yr <= merged["year"].max():
            ax.axvline(yr, color="orange", alpha=0.4, linestyle="--")
            ax.text(yr, ax.get_ylim()[1] * 0.9, label, fontsize=6, ha="center", va="top",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.8))

    # Panel D: Reuters trust + news avoidance
    ax = axes[1, 1]
    ax.plot(reuters["year"], reuters["reuters_trust_social_pct"], "o-", color="#0066CC", linewidth=2, label="Trust in social media news (%)")
    ax2 = ax.twinx()
    avoidance = reuters.dropna(subset=["news_avoidance_pct"])
    ax2.plot(avoidance["year"], avoidance["news_avoidance_pct"], "s-", color="#CC6600", linewidth=2, label="News avoidance (%)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Trust (%)", color="#0066CC")
    ax2.set_ylabel("News Avoidance (%)", color="#CC6600")
    ax.set_title("D) Reuters DNR: Trust & News Avoidance")
    ax.grid(True, alpha=0.3)
    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center left")

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "pilot_platform_extraction.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {fig_path}")

    # ---- Results JSON ----
    results = {
        "study": "Platform Investment-to-Extraction Phase",
        "meta_arpu": {
            "us_2012": float(meta["meta_arpu_us"].iloc[0]),
            "us_2024": float(meta["meta_arpu_us"].iloc[-1]),
            "growth_ratio": float(meta["meta_arpu_us"].iloc[-1] / meta["meta_arpu_us"].iloc[0]),
            "log_linear_growth_rate_pct_per_year": float(slope * 100),
            "arpu_vs_dau_growth_ratio": float(slope / slope_dau),
        },
        "pew_attitudes": {
            "bad_2018": float(pew.loc[pew["year"] == 2018, "pew_bad_pct"].values[0]),
            "bad_2024": float(pew.loc[pew["year"] == 2024, "pew_bad_pct"].values[0]),
            "trend_pp_per_year": float(pew_slope),
        },
        "divergence_index": {
            "trend_slope_per_year": float(di_slope),
            "trend_r_squared": float(di_r ** 2),
            "trend_p_value": float(di_p),
        },
        "arpu_attitude_correlation": {
            "pearson_r": float(r_val) if not np.isnan(r_val) else None,
            "p_value": float(p_val) if not np.isnan(p_val) else None,
            "n_observations": int(len(arpu_pew)),
        },
    }
    results_path = os.path.join(RESULTS_DIR, "pilot_platform_extraction.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {results_path}")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    print(f"  1. Meta US ARPU grew {meta['meta_arpu_us'].iloc[-1]/meta['meta_arpu_us'].iloc[0]:.0f}x "
          f"in 12 years ({slope*100:.0f}%/yr compound)")
    print(f"  2. ARPU grows {slope/slope_dau:.1f}x faster than user base → extraction acceleration")
    print(f"  3. 'Social media bad for society': {pew.iloc[0]['pew_bad_pct']:.0f}% (2018) → "
          f"{pew.iloc[-1]['pew_bad_pct']:.0f}% (2024)")
    print(f"  4. Divergence Index trend: {di_slope:+.3f}/yr (p={di_p:.4f})")
    print(f"  5. Trust in social media news: {reuters.iloc[0]['reuters_trust_social_pct']}% "
          f"({int(reuters.iloc[0]['year'])}) → {reuters.iloc[-1]['reuters_trust_social_pct']}% "
          f"({int(reuters.iloc[-1]['year'])})")
    print()


if __name__ == "__main__":
    main()
