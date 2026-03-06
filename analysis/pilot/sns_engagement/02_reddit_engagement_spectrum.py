#!/usr/bin/env python3
"""
Pilot SNS Engagement — Reddit Engagement Spectrum Analysis
==========================================================
Analyzes the relationship between posting frequency and mental health
subreddit participation using Reddit data.

Primary data: Reddit Mental Health Dataset (Zenodo: 10.5281/zenodo.3941387)
Fallback:     Synthetic data from 00_synthetic_data.py

Analyses:
  1. Engagement tier classification (Lurker → Light → Moderate → Heavy)
  2. MH subreddit participation by tier
  3. Logistic regression: MH_participation ~ log(posts) + account_age
  4. Engagement spectrum visualization

Outputs:
  - results/pilot_reddit_engagement.json
  - results/figures/pilot_reddit_engagement_spectrum.png
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

import statsmodels.api as sm
import statsmodels.formula.api as smf

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..")
DATA_DIR = os.path.join(BASE_DIR, "data", "pilot")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Mental health subreddits (from the Zenodo dataset documentation)
MH_SUBREDDITS = {
    "depression", "anxiety", "suicidewatch", "bipolar", "mentalhealth",
    "bpd", "ptsd", "adhd", "schizophrenia", "ocd",
    "socialanxiety", "panicattack", "eatingdisorders",
    "selfharm", "stopselfharm",
}


# ============================================================
# DATA LOADING
# ============================================================
def load_data() -> tuple[pd.DataFrame, str]:
    """
    Load Reddit Mental Health data or fall back to synthetic.

    Returns:
        (DataFrame, source_label)
    """
    # 1. Try real Reddit data (pre-aggregated per user)
    reddit_dir = os.path.join(DATA_DIR, "reddit_mental_health")
    reddit_candidates = [
        "user_engagement_summary.csv",   # Pre-processed per-user summary
        "posts_aggregated.csv",
    ]
    for fname in reddit_candidates:
        path = os.path.join(reddit_dir, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"Loaded Reddit data: {path} (rows={len(df):,})")
            return df, "reddit_real"

    # 2. Try raw Reddit data (need to aggregate)
    raw_candidates = [
        os.path.join(reddit_dir, "posts.csv"),
        os.path.join(reddit_dir, "reddit_mental_health_posts.csv"),
    ]
    for path in raw_candidates:
        if os.path.exists(path):
            print(f"Found raw Reddit data: {path}")
            print("  Aggregating per user... (this may take a moment)")
            df = aggregate_reddit_raw(path)
            print(f"  Aggregated: {len(df):,} users")
            return df, "reddit_real"

    # 3. Fallback to synthetic
    synth_path = os.path.join(DATA_DIR, "reddit_synthetic.csv")
    if os.path.exists(synth_path):
        df = pd.read_csv(synth_path)
        print(f"Loaded synthetic Reddit data: {synth_path} (rows={len(df):,})")
        print("  NOTE: Using synthetic data. For real analysis, download from:")
        print("    https://zenodo.org/records/3941387")
        print(f"  Place files in: {reddit_dir}/")
        return df, "synthetic"

    # 4. No data found
    print("ERROR: No data found.")
    print(f"  Expected: {reddit_dir}/ or {synth_path}")
    print("  Run 00_synthetic_data.py first, or download Reddit data from Zenodo.")
    sys.exit(1)


def aggregate_reddit_raw(path: str) -> pd.DataFrame:
    """
    Aggregate raw post-level Reddit data to per-user summary.
    """
    df = pd.read_csv(path, usecols=lambda c: c.lower() in {
        "author", "subreddit", "created_utc", "id",
    })
    df.columns = [c.lower() for c in df.columns]

    # Remove deleted/removed
    df = df[~df["author"].isin(["[deleted]", "AutoModerator"])]

    # Per-user aggregation
    user_stats = df.groupby("author").agg(
        total_posts=("id", "count"),
        mh_subreddit_posts=("subreddit", lambda x: sum(s.lower() in MH_SUBREDDITS for s in x)),
    ).reset_index()

    user_stats.rename(columns={"author": "user_id"}, inplace=True)
    user_stats["mh_subreddit_pct"] = (
        user_stats["mh_subreddit_posts"] / user_stats["total_posts"]
    ).fillna(0)

    # Estimate account age from first post
    first_post = df.groupby("author")["created_utc"].min().reset_index()
    first_post.columns = ["user_id", "first_post_utc"]
    user_stats = user_stats.merge(first_post, on="user_id", how="left")
    if "first_post_utc" in user_stats.columns:
        max_utc = df["created_utc"].max()
        user_stats["account_age_days"] = (
            (max_utc - user_stats["first_post_utc"]) / 86400
        ).fillna(365)
    else:
        user_stats["account_age_days"] = 365  # default

    return user_stats


def normalize_reddit(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Normalize column names."""
    col_map = {c.lower(): c for c in df.columns}

    renames = {}
    for std_name, candidates in {
        "user_id": ["user_id", "author", "username"],
        "total_posts": ["total_posts", "post_count", "num_posts"],
        "total_comments": ["total_comments", "comment_count", "num_comments"],
        "mh_subreddit_posts": ["mh_subreddit_posts", "mh_posts"],
        "mh_subreddit_pct": ["mh_subreddit_pct", "mh_pct", "mh_fraction"],
        "account_age_days": ["account_age_days", "account_age"],
        "engagement_tier": ["engagement_tier", "tier", "user_tier"],
    }.items():
        for c in candidates:
            if c.lower() in col_map:
                renames[col_map[c.lower()]] = std_name
                break

    df = df.rename(columns=renames)

    required = ["total_posts"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        sys.exit(1)

    # Derive engagement tier if missing
    if "engagement_tier" not in df.columns:
        df["engagement_tier"] = np.where(
            df["total_posts"] == 0, "Lurker",
            np.where(df["total_posts"] <= 5, "Light",
                     np.where(df["total_posts"] <= 50, "Moderate", "Heavy")))

    # Derive MH pct if missing
    if "mh_subreddit_pct" not in df.columns and "mh_subreddit_posts" in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["mh_subreddit_pct"] = np.where(
                df["total_posts"] > 0,
                df["mh_subreddit_posts"] / df["total_posts"],
                0.0,
            )

    return df


# ============================================================
# ANALYSIS 1: ENGAGEMENT TIER DISTRIBUTION
# ============================================================
def analyze_tiers(df: pd.DataFrame) -> dict:
    """Describe engagement tier distribution and MH participation."""
    tier_order = ["Lurker", "Light", "Moderate", "Heavy"]
    results = {}

    for tier in tier_order:
        sub = df[df["engagement_tier"] == tier]
        if len(sub) == 0:
            continue

        mh_rate = sub["mh_subreddit_pct"].mean() if "mh_subreddit_pct" in sub.columns else 0
        mh_any = (sub["mh_subreddit_pct"] > 0).mean() if "mh_subreddit_pct" in sub.columns else 0

        results[tier] = {
            "n": int(len(sub)),
            "pct_of_total": float(len(sub) / len(df) * 100),
            "mean_posts": float(sub["total_posts"].mean()),
            "median_posts": float(sub["total_posts"].median()),
            "mh_subreddit_mean_pct": float(mh_rate),
            "mh_any_participation": float(mh_any),
        }

        print(f"    {tier:10s}: n={len(sub):6,}  "
              f"mean_posts={sub['total_posts'].mean():7.1f}  "
              f"MH_any={mh_any:.1%}  "
              f"MH_mean_pct={mh_rate:.3f}")

    return results


# ============================================================
# ANALYSIS 2: LOGISTIC REGRESSION
# ============================================================
def run_logistic(df: pd.DataFrame) -> dict:
    """
    Logistic regression: MH_participation ~ log(1+posts) + account_age.
    """
    sub = df[df["total_posts"] > 0].copy()  # Exclude lurkers (no posts → no MH posts)

    if "mh_subreddit_pct" not in sub.columns:
        print("  SKIP: mh_subreddit_pct column not available")
        return {}

    sub["mh_any"] = (sub["mh_subreddit_pct"] > 0).astype(int)
    sub["log_posts"] = np.log1p(sub["total_posts"])

    if "account_age_days" in sub.columns:
        sub["log_account_age"] = np.log1p(sub["account_age_days"])
        formula = "mh_any ~ log_posts + log_account_age"
    else:
        formula = "mh_any ~ log_posts"

    try:
        model = smf.logit(formula, data=sub).fit(disp=0)
    except Exception as e:
        print(f"  Logistic regression failed: {e}")
        return {}

    print(f"\n  Logistic Regression (n={int(model.nobs):,}):")
    print(f"  Pseudo R-squared = {model.prsquared:.4f}")
    print(f"  AIC = {model.aic:.1f}")
    for var in model.params.index:
        b = model.params[var]
        se = model.bse[var]
        z = model.tvalues[var]
        p = model.pvalues[var]
        odds_ratio = np.exp(b)
        print(f"    {var:20s}: beta={b:7.3f}, SE={se:.3f}, z={z:6.2f}, "
              f"p={p:.3e}, OR={odds_ratio:.3f}")

    result = {
        "n": int(model.nobs),
        "pseudo_r_squared": float(model.prsquared),
        "aic": float(model.aic),
        "coefficients": {},
    }
    for var in model.params.index:
        result["coefficients"][var] = {
            "beta": float(model.params[var]),
            "se": float(model.bse[var]),
            "z_stat": float(model.tvalues[var]),
            "p_value": float(model.pvalues[var]),
            "odds_ratio": float(np.exp(model.params[var])),
        }

    return result


# ============================================================
# ANALYSIS 3: SPECTRUM VISUALIZATION
# ============================================================
def plot_spectrum(df: pd.DataFrame, source: str) -> str:
    """Plot engagement spectrum with MH participation overlay."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left: Post count distribution (log scale) ---
    active = df[df["total_posts"] > 0]
    axes[0].hist(
        np.log10(active["total_posts"] + 1),
        bins=40,
        color="#1f77b4",
        edgecolor="white",
        alpha=0.8,
    )
    axes[0].set_xlabel("log10(Post Count + 1)")
    axes[0].set_ylabel("Number of Users")
    axes[0].set_title("Post Count Distribution")
    axes[0].axvline(np.log10(2), color="gray", linestyle="--", alpha=0.5, label="Light/Lurker")
    axes[0].axvline(np.log10(6), color="orange", linestyle="--", alpha=0.5, label="Light/Moderate")
    axes[0].axvline(np.log10(51), color="red", linestyle="--", alpha=0.5, label="Moderate/Heavy")
    axes[0].legend(fontsize=8)

    # --- Right: MH participation rate by post count bins ---
    if "mh_subreddit_pct" in df.columns:
        # Create post count bins (adaptive to data range)
        max_posts = int(df["total_posts"].max())
        all_edges = [0, 1, 2, 5, 10, 20, 50, 100, 500, 5000]
        all_labels = ["0", "1", "2-4", "5-9", "10-19", "20-49", "50-99", "100-499", "500+"]

        # Keep only edges up to max_posts, then add final edge
        valid_bins = [e for e in all_edges if e <= max_posts]
        valid_bins.append(max_posts + 1)
        # Deduplicate while preserving order
        seen = set()
        unique_bins = []
        for b in valid_bins:
            if b not in seen:
                seen.add(b)
                unique_bins.append(b)
        valid_bins = unique_bins
        bin_labels = all_labels[: len(valid_bins) - 1]

        df_plot = df.copy()
        df_plot["post_bin"] = pd.cut(
            df_plot["total_posts"],
            bins=valid_bins,
            labels=bin_labels,
            right=False,
            duplicates="drop",
        )

        mh_by_bin = df_plot.groupby("post_bin", observed=True).agg(
            mh_any_rate=("mh_subreddit_pct", lambda x: (x > 0).mean()),
            mh_mean_pct=("mh_subreddit_pct", "mean"),
            n=("total_posts", "count"),
        )

        ax2_twin = axes[1].twinx()

        # Bar: sample size
        axes[1].bar(
            range(len(mh_by_bin)),
            mh_by_bin["n"],
            alpha=0.3,
            color="#1f77b4",
            label="N users",
        )
        axes[1].set_ylabel("Number of Users", color="#1f77b4")

        # Line: MH participation rate
        ax2_twin.plot(
            range(len(mh_by_bin)),
            mh_by_bin["mh_any_rate"] * 100,
            "o-",
            color="#d62728",
            linewidth=2,
            markersize=6,
            label="MH subreddit participation %",
        )
        ax2_twin.set_ylabel("MH Subreddit Participation (%)", color="#d62728")

        axes[1].set_xticks(range(len(mh_by_bin)))
        axes[1].set_xticklabels(mh_by_bin.index, rotation=45, ha="right", fontsize=8)
        axes[1].set_xlabel("Post Count Range")
        axes[1].set_title("MH Participation by Engagement Level")

        # Combined legend
        lines1, labels1 = axes[1].get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        axes[1].legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

    source_label = "Synthetic" if source == "synthetic" else "Reddit Mental Health Dataset"
    fig.suptitle(f"Reddit Engagement Spectrum ({source_label})", fontsize=13, fontweight="bold")
    plt.tight_layout()

    fig_path = os.path.join(FIG_DIR, "pilot_reddit_engagement_spectrum.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {fig_path}")

    return fig_path


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    print("=" * 70)
    print("PILOT: Reddit Engagement Spectrum & Mental Health")
    print("=" * 70)

    # Load data
    df, source = load_data()
    df = normalize_reddit(df, source)

    print(f"\n  Total users: {len(df):,}")
    print(f"  Users with posts > 0: {(df['total_posts'] > 0).sum():,}")
    if "mh_subreddit_pct" in df.columns:
        print(f"  Users with any MH participation: {(df['mh_subreddit_pct'] > 0).sum():,}")

    # Analysis 1: Tier distribution
    print("\n" + "-" * 50)
    print("Analysis 1: Engagement Tier Distribution")
    print("-" * 50)
    tier_results = analyze_tiers(df)

    # Analysis 2: Logistic regression
    print("\n" + "-" * 50)
    print("Analysis 2: Logistic Regression")
    print("-" * 50)
    logistic_results = run_logistic(df)

    # Analysis 3: Visualization
    print("\n" + "-" * 50)
    print("Analysis 3: Engagement Spectrum Visualization")
    print("-" * 50)
    fig_path = plot_spectrum(df, source)

    # Key finding
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    if "Moderate" in tier_results and "Light" in tier_results:
        mod_mh = tier_results["Moderate"]["mh_any_participation"]
        light_mh = tier_results["Light"]["mh_any_participation"]
        print(f"  Moderate vs Light MH participation: {mod_mh:.1%} vs {light_mh:.1%}")
    if logistic_results and "coefficients" in logistic_results:
        log_posts = logistic_results["coefficients"].get("log_posts", {})
        if log_posts:
            print(f"  log(posts) → MH participation: OR={log_posts.get('odds_ratio', 0):.3f}, "
                  f"p={log_posts.get('p_value', 1):.3e}")

    # Save JSON
    output = {
        "source": source,
        "n_users": len(df),
        "tier_distribution": tier_results,
        "logistic_regression": logistic_results,
        "figure": fig_path,
    }

    json_path = os.path.join(RESULTS_DIR, "pilot_reddit_engagement.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {json_path}")

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
