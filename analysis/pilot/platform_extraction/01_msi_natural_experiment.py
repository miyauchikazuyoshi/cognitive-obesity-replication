#!/usr/bin/env python3
"""
Study 2: Facebook MSI Algorithm Change — Natural Experiment Analysis
=====================================================================

Tests whether Facebook's January 2018 "Meaningful Social Interactions" (MSI)
algorithm change functioned as a measurable extraction-phase event that
increased engagement while degrading user experience.

Design: Interrupted time-series / pre-post comparison using published survey data.

Data sources (all published aggregate data):
  - Pew Research Center: "Social media good/bad for society" (2018, 2020, 2022, 2024)
  - Pew Research Center: "Internet good/bad for society" (2014, 2018)
  - Pew Research Center: Platform-specific usage (2012-2024)
  - Reuters Institute DNR: Trust in news via social media (2012-2024)
  - Meta 10-K: ARPU, DAU, engagement metrics

The MSI change (Jan 11, 2018) is a sharp, exogenous policy shift that provides
a natural experiment breakpoint.

NOTE: For individual-level analysis, Pew ATP microdata (Waves 25, 28, 51)
      requires free registration at pewresearch.org. This script uses published
      aggregate-level data. When ATP microdata is available, uncomment the
      microdata analysis section.

Connection to Cognitive Obesity:
  The MSI change was framed as reducing passive consumption (lowering I),
  but internal documents (Haugen 2021) suggest it actually amplified
  engagement-maximizing content (inflammatory, divisive) — inflating I
  while degrading information quality, thus worsening L = α₁·I − α₂·C.
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
# DATA: Compiled from published sources
# ======================================================================

def build_pew_internet_attitudes():
    """
    Pew: "Has the internet been a good thing or bad thing for society?"
    Source: Pew Research Center, April 2018 report
    "Declining Majority of Online Americans Say the Internet Has Been Good for Society"
    """
    data = {
        "year": [2014, 2018],
        "internet_good_pct": [76, 70],
        "internet_bad_pct": [15, 25],
        "internet_both_pct": [8, 4],
    }
    return pd.DataFrame(data)


def build_pew_social_media_attitudes():
    """
    Pew: "Overall, social media has had a [mostly positive / mostly negative]
    effect on the way things are going in the country today."
    Source: Pew ATP surveys, published toplines
    """
    data = {
        "year": [2020, 2022, 2024],
        # % saying "mostly negative effect"
        "sm_negative_pct": [64, 64, 68],
        # % saying "mostly positive effect"
        "sm_positive_pct": [10, 10, 8],
        # Note: 2018 used different wording ("good/bad for society")
        # so treated separately
    }
    return pd.DataFrame(data)


def build_pew_platform_adoption():
    """
    Pew: "Do you use [platform]?" — % of US adults
    Source: Pew Social Media Fact Sheet (annual)
    """
    data = {
        "year": [2012, 2013, 2014, 2015, 2016, 2018, 2019, 2021, 2023, 2024],
        "facebook_pct": [54, 57, 58, 62, 68, 68, 69, 69, 68, 67],
        "youtube_pct": [None, None, None, None, None, 73, 73, 81, 83, 85],
        "instagram_pct": [None, 13, 17, 21, 28, 35, 37, 40, 47, 48],
        "twitter_pct": [None, 16, 18, 19, 21, 24, 22, 23, 22, 22],
    }
    return pd.DataFrame(data)


def build_facebook_engagement_metrics():
    """
    Meta 10-K: Facebook-specific engagement metrics.
    DAU/MAU ratio = "stickiness" proxy (higher = more habitual use)

    Source: Meta 10-K SEC filings (10-K reports Facebook DAU and MAU separately)
    Note: From 2024, Meta stopped reporting Facebook-specific DAU/MAU,
          switching to "Family" DAP/MAP metrics.
    """
    data = {
        "year": [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
        # Facebook DAU (millions, Q4) — verified against SEC 10-K filings
        "fb_dau_m": [757, 890, 1038, 1227, 1401, 1523, 1657, 1840, 1929, 2000, 2110],
        # Facebook MAU (millions, Q4)
        "fb_mau_m": [1228, 1393, 1591, 1860, 2129, 2320, 2498, 2797, 2912, 2963, 3065],
    }
    df = pd.DataFrame(data)
    df["stickiness"] = df["fb_dau_m"] / df["fb_mau_m"]
    return df


def build_reuters_trust_timeseries():
    """
    Reuters DNR: Trust in news found via social media (global average %).
    Source: Reuters Institute Digital News Report (annual)
    """
    data = {
        "year": [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
        "trust_social_pct": [32, 27, 24, 23, 23, 24, 22, 23, 22, 20],
    }
    return pd.DataFrame(data)


# ======================================================================
# ANALYSIS
# ======================================================================

def main():
    print("=" * 70)
    print("STUDY 2: Facebook MSI Algorithm Change — Natural Experiment")
    print("=" * 70)
    print("  Event: Jan 11, 2018 — Facebook MSI algorithm change")
    print("  Design: Pre-post comparison with multiple indicator time series")

    # Load data
    internet_att = build_pew_internet_attitudes()
    sm_att = build_pew_social_media_attitudes()
    platform = build_pew_platform_adoption()
    fb_engage = build_facebook_engagement_metrics()
    reuters = build_reuters_trust_timeseries()

    # ---- Analysis 1: Pre-post Internet Attitudes ----
    print("\n" + "-" * 50)
    print("Analysis 1: Internet 'Good for Society' — Pre vs Post MSI")
    print("-" * 50)

    print("  Pew: 'Has the internet been a good thing for society?'")
    print(f"    2014 (pre-MSI):  {internet_att.iloc[0]['internet_good_pct']}% good, "
          f"{internet_att.iloc[0]['internet_bad_pct']}% bad")
    print(f"    2018 (post-MSI): {internet_att.iloc[1]['internet_good_pct']}% good, "
          f"{internet_att.iloc[1]['internet_bad_pct']}% bad")
    good_change = internet_att.iloc[1]['internet_good_pct'] - internet_att.iloc[0]['internet_good_pct']
    bad_change = internet_att.iloc[1]['internet_bad_pct'] - internet_att.iloc[0]['internet_bad_pct']
    print(f"    Change: good {good_change:+.0f}pp, bad {bad_change:+.0f}pp")
    print(f"    Note: 4-year window includes MSI + Cambridge Analytica (Mar 2018)")

    # ---- Analysis 2: Facebook Stickiness (DAU/MAU) ----
    print("\n" + "-" * 50)
    print("Analysis 2: Facebook Stickiness (DAU/MAU Ratio)")
    print("-" * 50)

    # Pre-MSI: 2013-2017, Post-MSI: 2018-2023
    pre_msi = fb_engage[fb_engage["year"] <= 2017]
    post_msi = fb_engage[fb_engage["year"] >= 2018]

    pre_sticky = pre_msi["stickiness"].values
    post_sticky = post_msi["stickiness"].values

    print(f"  Pre-MSI (2013-2017):  mean stickiness = {pre_sticky.mean():.4f} "
          f"(range: {pre_sticky.min():.4f}-{pre_sticky.max():.4f})")
    print(f"  Post-MSI (2018-2023): mean stickiness = {post_sticky.mean():.4f} "
          f"(range: {post_sticky.min():.4f}-{post_sticky.max():.4f})")

    # t-test
    t_stat, t_p = stats.ttest_ind(pre_sticky, post_sticky, equal_var=False)
    print(f"  Welch t-test: t = {t_stat:.3f}, p = {t_p:.4f}")
    if t_p < 0.05:
        direction = "higher" if post_sticky.mean() > pre_sticky.mean() else "lower"
        print(f"  → Stickiness is significantly {direction} post-MSI")
    else:
        print(f"  → No significant change in stickiness")

    # Trend analysis
    for label, subset in [("Pre-MSI", pre_msi), ("Post-MSI", post_msi)]:
        if len(subset) >= 3:
            slope, _, r, p, _ = stats.linregress(subset["year"], subset["stickiness"])
            print(f"  {label} trend: slope = {slope:+.5f}/yr, R² = {r**2:.3f}")

    # ---- Analysis 3: Trust Decline — Structural Break at 2018 ----
    print("\n" + "-" * 50)
    print("Analysis 3: Reuters Trust — Structural Break Detection")
    print("-" * 50)

    trust_pre = reuters[reuters["year"] <= 2017]
    trust_post = reuters[reuters["year"] >= 2018]

    # Linear trend pre and post
    if len(trust_pre) >= 2:
        slope_pre, int_pre, r_pre, _, _ = stats.linregress(
            trust_pre["year"], trust_pre["trust_social_pct"]
        )
        print(f"  Pre-MSI (2015-2017): slope = {slope_pre:+.2f}pp/yr, R² = {r_pre**2:.3f}")

    if len(trust_post) >= 2:
        slope_post, int_post, r_post, _, _ = stats.linregress(
            trust_post["year"], trust_post["trust_social_pct"]
        )
        print(f"  Post-MSI (2018-2024): slope = {slope_post:+.2f}pp/yr, R² = {r_post**2:.3f}")

    # Chow-like test: compare pooled vs separate regressions
    # Full model: one regression for all data
    slope_full, int_full, r_full, _, _ = stats.linregress(
        reuters["year"], reuters["trust_social_pct"]
    )
    print(f"  Full period (2015-2024): slope = {slope_full:+.2f}pp/yr, R² = {r_full**2:.3f}")

    # Residuals comparison
    rss_full = sum((reuters["trust_social_pct"] - (slope_full * reuters["year"] + int_full)) ** 2)
    rss_pre = sum((trust_pre["trust_social_pct"] - (slope_pre * trust_pre["year"] + int_pre)) ** 2) if len(trust_pre) >= 2 else 0
    rss_post = sum((trust_post["trust_social_pct"] - (slope_post * trust_post["year"] + int_post)) ** 2) if len(trust_post) >= 2 else 0
    rss_split = rss_pre + rss_post

    print(f"\n  RSS (full model): {rss_full:.2f}")
    print(f"  RSS (split at 2018): {rss_split:.2f}")
    print(f"  RSS reduction: {(1 - rss_split/rss_full)*100:.1f}%")

    # Simple Chow F-test approximation
    n = len(reuters)
    k = 2  # intercept + slope
    if rss_split > 0 and (n - 2*k) > 0:
        f_stat = ((rss_full - rss_split) / k) / (rss_split / (n - 2*k))
        print(f"  Chow F-stat ≈ {f_stat:.2f} (df1={k}, df2={n-2*k})")

    # ---- Analysis 4: Facebook Adoption Plateau ----
    print("\n" + "-" * 50)
    print("Analysis 4: Platform Adoption — Facebook Saturation")
    print("-" * 50)

    fb_data = platform.dropna(subset=["facebook_pct"])
    fb_pre = fb_data[fb_data["year"] <= 2017]
    fb_post = fb_data[fb_data["year"] >= 2018]

    if len(fb_pre) >= 2:
        slope_fb_pre, _, _, _, _ = stats.linregress(fb_pre["year"], fb_pre["facebook_pct"])
        print(f"  Facebook adoption growth pre-MSI: {slope_fb_pre:+.1f}pp/yr")

    if len(fb_post) >= 2:
        slope_fb_post, _, _, _, _ = stats.linregress(fb_post["year"], fb_post["facebook_pct"])
        print(f"  Facebook adoption growth post-MSI: {slope_fb_post:+.1f}pp/yr")

    print(f"\n  Facebook adoption trajectory:")
    for _, row in fb_data.iterrows():
        bar = "█" * int(row["facebook_pct"])
        print(f"    {int(row['year'])}: {row['facebook_pct']:.0f}% {bar}")

    # ---- Visualization ----
    print("\n" + "-" * 50)
    print("Analysis 5: Visualization")
    print("-" * 50)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Study 2: Facebook MSI Algorithm Change (Jan 2018) — Natural Experiment",
                 fontsize=13, fontweight="bold")

    msi_date = 2018.04  # Jan 11, 2018

    # Panel A: Facebook stickiness
    ax = axes[0, 0]
    ax.plot(fb_engage["year"], fb_engage["stickiness"], "o-", color="#1877F2",
            linewidth=2, markersize=7)
    ax.axvline(msi_date, color="red", linestyle="--", alpha=0.7, label="MSI change")
    ax.set_xlabel("Year")
    ax.set_ylabel("DAU/MAU Ratio")
    ax.set_title("A) Facebook Stickiness (DAU/MAU)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.55, 0.75)

    # Panel B: Reuters trust
    ax = axes[0, 1]
    ax.plot(reuters["year"], reuters["trust_social_pct"], "o-", color="#0066CC",
            linewidth=2, markersize=7)
    ax.axvline(msi_date, color="red", linestyle="--", alpha=0.7, label="MSI change")
    # Pre and post trend lines
    if len(trust_pre) >= 2:
        x_pre = np.linspace(trust_pre["year"].min(), 2017.5, 50)
        ax.plot(x_pre, slope_pre * x_pre + int_pre, ":", color="green", alpha=0.7, label="Pre-MSI trend")
    if len(trust_post) >= 2:
        x_post = np.linspace(2017.5, trust_post["year"].max(), 50)
        ax.plot(x_post, slope_post * x_post + int_post, ":", color="red", alpha=0.7, label="Post-MSI trend")
    ax.set_xlabel("Year")
    ax.set_ylabel("Trust in Social Media News (%)")
    ax.set_title("B) Reuters: Trust in News via Social Media")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel C: Platform adoption
    ax = axes[1, 0]
    fb_clean = platform.dropna(subset=["facebook_pct"])
    yt_clean = platform.dropna(subset=["youtube_pct"])
    ig_clean = platform.dropna(subset=["instagram_pct"])
    ax.plot(fb_clean["year"], fb_clean["facebook_pct"], "o-", color="#1877F2",
            linewidth=2, label="Facebook")
    if len(yt_clean) > 0:
        ax.plot(yt_clean["year"], yt_clean["youtube_pct"], "s-", color="#FF0000",
                linewidth=2, label="YouTube")
    if len(ig_clean) > 0:
        ax.plot(ig_clean["year"], ig_clean["instagram_pct"], "^-", color="#E1306C",
                linewidth=2, label="Instagram")
    ax.axvline(msi_date, color="red", linestyle="--", alpha=0.5, label="MSI change")
    ax.set_xlabel("Year")
    ax.set_ylabel("% of US Adults Using Platform")
    ax.set_title("C) Platform Adoption (Pew)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel D: Social media attitudes over time
    ax = axes[1, 1]
    # Combine internet attitudes (2014, 2018) and social media attitudes (2020, 2022, 2024)
    all_years = [2014, 2018, 2020, 2022, 2024]
    # Internet bad + SM negative (different questions but related construct)
    bad_pct = [15, 25, 64, 64, 68]
    labels_q = ["Internet\nbad", "Internet\nbad", "SM\nnegative", "SM\nnegative", "SM\nnegative"]
    colors = ["#FFB347", "#FFB347", "#FF4444", "#FF4444", "#FF4444"]
    bars = ax.bar(all_years, bad_pct, width=0.6, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.axvline(msi_date, color="red", linestyle="--", alpha=0.7, label="MSI change")
    for bar, label, yr in zip(bars, labels_q, all_years):
        ax.text(yr, bar.get_height() + 1, label, ha="center", va="bottom", fontsize=7)
    ax.set_xlabel("Year")
    ax.set_ylabel("% Negative Attitude")
    ax.set_title("D) Negative Attitudes Toward Internet/Social Media")
    ax.set_ylim(0, 80)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "pilot_msi_natural_experiment.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {fig_path}")

    # ---- Results JSON ----
    results = {
        "study": "MSI Algorithm Change Natural Experiment",
        "event_date": "2018-01-11",
        "internet_attitudes": {
            "good_2014": int(internet_att.iloc[0]["internet_good_pct"]),
            "good_2018": int(internet_att.iloc[1]["internet_good_pct"]),
            "bad_2014": int(internet_att.iloc[0]["internet_bad_pct"]),
            "bad_2018": int(internet_att.iloc[1]["internet_bad_pct"]),
        },
        "facebook_stickiness": {
            "pre_msi_mean": float(pre_sticky.mean()),
            "post_msi_mean": float(post_sticky.mean()),
            "welch_t": float(t_stat),
            "welch_p": float(t_p),
        },
        "reuters_trust": {
            "pre_msi_slope_pp_yr": float(slope_pre) if len(trust_pre) >= 2 else None,
            "post_msi_slope_pp_yr": float(slope_post) if len(trust_post) >= 2 else None,
            "rss_full": float(rss_full),
            "rss_split": float(rss_split),
            "rss_reduction_pct": float((1 - rss_split / rss_full) * 100),
        },
        "facebook_adoption": {
            "pre_msi_growth_pp_yr": float(slope_fb_pre) if len(fb_pre) >= 2 else None,
            "post_msi_growth_pp_yr": float(slope_fb_post) if len(fb_post) >= 2 else None,
        },
        "data_note": "Uses published aggregate data. For individual-level DiD analysis, "
                     "download Pew ATP Waves 25, 28, 51 from pewresearch.org (free, requires registration).",
    }
    results_path = os.path.join(RESULTS_DIR, "pilot_msi_natural_experiment.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {results_path}")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    print(f"  1. Internet 'bad for society': {internet_att.iloc[0]['internet_bad_pct']}% (2014) "
          f"→ {internet_att.iloc[1]['internet_bad_pct']}% (2018)")
    print(f"  2. SM 'mostly negative effect': {sm_att.iloc[0]['sm_negative_pct']}% (2020) "
          f"→ {sm_att.iloc[-1]['sm_negative_pct']}% (2024)")
    print(f"  3. Facebook stickiness: {pre_sticky.mean():.4f} (pre) vs "
          f"{post_sticky.mean():.4f} (post), p={t_p:.4f}")
    print(f"  4. Trust decline: {slope_pre:+.2f}pp/yr (pre) vs {slope_post:+.2f}pp/yr (post)")
    print(f"  5. Facebook adoption: {slope_fb_pre:+.1f}pp/yr (pre) → "
          f"{slope_fb_post:+.1f}pp/yr (post) = saturation")
    print()
    print("  INTERPRETATION:")
    print("  The MSI change coincides with a plateau in Facebook adoption,")
    print("  a flattening of trust decline (already low), and sustained high")
    print("  stickiness — consistent with the extraction-phase hypothesis:")
    print("  fewer new users + same engagement = more monetization per user.")
    print()


if __name__ == "__main__":
    main()
