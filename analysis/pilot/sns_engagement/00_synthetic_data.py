#!/usr/bin/env python3
"""
Pilot SNS Engagement — Synthetic Data Generator
================================================
Generates synthetic survey + Reddit-style data for pipeline testing.
Real data sources:
  - Understanding Society Wave 11 (UK Data Service)
  - Reddit Mental Health Dataset (Zenodo)

Outputs:
  - data/pilot/sns_engagement_synthetic.csv   (survey-style)
  - data/pilot/reddit_synthetic.csv           (Reddit-style)
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..")
DATA_DIR = os.path.join(BASE_DIR, "data", "pilot")
os.makedirs(DATA_DIR, exist_ok=True)

SEED = 42


# ============================================================
# 1. SURVEY-STYLE SYNTHETIC DATA (Understanding Society mimic)
# ============================================================
def generate_survey_data(n: int = 2000) -> pd.DataFrame:
    """
    Generate synthetic survey data mimicking Understanding Society Wave 11.

    Variables:
      respondent_id   — unique identifier
      age             — 16-85
      sex             — 0=male, 1=female
      posting_freq    — 0-7 scale (how often post on social media)
      viewing_freq    — 0-7 scale (how often view social media)
      ghq12_score     — 0-36 (General Health Questionnaire, higher = worse)
      platform_primary — Instagram / Twitter / Facebook / TikTok / Reddit / Other
      push_on         — bool (push notifications enabled)
      perceived_ad_freq — 1-5 (how many ads perceived)
      engagement_type — Writer / ROM / Mixed (derived)
    """
    rng = np.random.default_rng(SEED)

    age = rng.integers(16, 86, size=n)
    sex = rng.binomial(1, 0.52, size=n)  # slight female skew

    # Posting freq: right-skewed (most people post rarely)
    posting_raw = rng.exponential(1.5, size=n)
    posting_freq = np.clip(np.round(posting_raw), 0, 7).astype(int)

    # Viewing freq: left-skewed (most people view often)
    viewing_raw = 7 - rng.exponential(1.2, size=n)
    viewing_freq = np.clip(np.round(viewing_raw), 0, 7).astype(int)

    # GHQ-12: base + effects
    # Hypothesized: high viewing + low posting → worst mental health
    # (passive consumption without expression outlet)
    ghq_base = 10 + 0.05 * age  # age baseline
    ghq_sex = 1.5 * sex  # female slightly higher
    ghq_viewing = 0.8 * viewing_freq  # passive consumption effect
    ghq_posting = -0.3 * posting_freq  # posting has mild protective effect
    ghq_interaction = 0.4 * viewing_freq * (7 - posting_freq) / 7  # ROM penalty
    ghq_noise = rng.normal(0, 4, size=n)

    ghq12_score = np.clip(
        np.round(ghq_base + ghq_sex + ghq_viewing + ghq_posting + ghq_interaction + ghq_noise),
        0, 36,
    ).astype(int)

    # Platform
    platforms = ["Instagram", "Twitter", "Facebook", "TikTok", "Reddit", "Other"]
    platform_weights = [0.30, 0.15, 0.25, 0.15, 0.10, 0.05]
    platform_primary = rng.choice(platforms, size=n, p=platform_weights)

    # Push notifications
    push_on = rng.binomial(1, 0.65, size=n).astype(bool)

    # Perceived ad frequency
    perceived_ad_freq = np.clip(rng.poisson(2.5, size=n) + 1, 1, 5)

    # Engagement type (derived)
    engagement_type = np.where(
        posting_freq >= 3,
        "Writer",
        np.where(posting_freq <= 1, "ROM", "Mixed"),
    )

    df = pd.DataFrame({
        "respondent_id": np.arange(1, n + 1),
        "age": age,
        "sex": sex,
        "posting_freq": posting_freq,
        "viewing_freq": viewing_freq,
        "ghq12_score": ghq12_score,
        "platform_primary": platform_primary,
        "push_on": push_on,
        "perceived_ad_freq": perceived_ad_freq,
        "engagement_type": engagement_type,
    })

    return df


# ============================================================
# 2. REDDIT-STYLE SYNTHETIC DATA
# ============================================================
def generate_reddit_data(n_users: int = 5000) -> pd.DataFrame:
    """
    Generate synthetic Reddit-style engagement data.

    Variables:
      user_id         — unique identifier
      total_posts     — total post count (power-law distributed)
      total_comments  — total comment count
      mh_subreddit_posts — posts in mental-health subreddits
      mh_subreddit_pct   — fraction of posts in MH subreddits
      account_age_days   — days since account creation
      engagement_tier    — Lurker / Light / Moderate / Heavy
    """
    rng = np.random.default_rng(SEED + 1)

    # Post counts: power-law (most users post very little)
    total_posts = np.round(rng.pareto(1.5, size=n_users) * 2).astype(int)
    total_posts = np.clip(total_posts, 0, 5000)

    # Comments: correlated with posts but higher volume
    total_comments = np.round(total_posts * rng.uniform(1.5, 8.0, size=n_users)).astype(int)

    # MH subreddit participation: higher for moderate posters (not heavy)
    # Hypothesis: moderate engagement correlates with help-seeking
    base_mh_rate = 0.02 + 0.05 * np.exp(-((np.log1p(total_posts) - 3) ** 2) / 4)
    mh_subreddit_posts = rng.binomial(np.maximum(total_posts, 1), base_mh_rate)
    with np.errstate(divide="ignore", invalid="ignore"):
        mh_subreddit_pct = np.where(
            total_posts > 0,
            mh_subreddit_posts / total_posts,
            0.0,
        )

    # Account age
    account_age_days = rng.integers(30, 3650, size=n_users)

    # Engagement tier
    engagement_tier = np.where(
        total_posts == 0,
        "Lurker",
        np.where(
            total_posts <= 5,
            "Light",
            np.where(total_posts <= 50, "Moderate", "Heavy"),
        ),
    )

    df = pd.DataFrame({
        "user_id": np.arange(1, n_users + 1),
        "total_posts": total_posts,
        "total_comments": total_comments,
        "mh_subreddit_posts": mh_subreddit_posts,
        "mh_subreddit_pct": mh_subreddit_pct,
        "account_age_days": account_age_days,
        "engagement_tier": engagement_tier,
    })

    return df


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    print("=" * 70)
    print("PILOT SNS ENGAGEMENT — Synthetic Data Generator")
    print("=" * 70)

    # --- Survey data ---
    survey = generate_survey_data(n=2000)
    survey_path = os.path.join(DATA_DIR, "sns_engagement_synthetic.csv")
    survey.to_csv(survey_path, index=False)
    print(f"\nSurvey data: {survey_path}")
    print(f"  rows = {len(survey):,}")
    print(f"  columns = {list(survey.columns)}")
    print(f"\n  Engagement type distribution:")
    for etype, count in survey["engagement_type"].value_counts().items():
        print(f"    {etype:8s}: {count:5d} ({count/len(survey)*100:.1f}%)")
    print(f"\n  GHQ-12 by engagement type:")
    for etype, grp in survey.groupby("engagement_type")["ghq12_score"]:
        print(f"    {etype:8s}: mean={grp.mean():.1f}, sd={grp.std():.1f}")

    # --- Reddit data ---
    reddit = generate_reddit_data(n_users=5000)
    reddit_path = os.path.join(DATA_DIR, "reddit_synthetic.csv")
    reddit.to_csv(reddit_path, index=False)
    print(f"\nReddit data: {reddit_path}")
    print(f"  rows = {len(reddit):,}")
    print(f"  columns = {list(reddit.columns)}")
    print(f"\n  Engagement tier distribution:")
    for tier, count in reddit["engagement_tier"].value_counts().items():
        print(f"    {tier:10s}: {count:5d} ({count/len(reddit)*100:.1f}%)")
    print(f"\n  MH subreddit participation by tier:")
    for tier, grp in reddit.groupby("engagement_tier")["mh_subreddit_pct"]:
        print(f"    {tier:10s}: mean={grp.mean():.3f}, sd={grp.std():.3f}")

    print("\n" + "=" * 70)
    print("Done. Synthetic data ready for pipeline testing.")
    print("=" * 70)


if __name__ == "__main__":
    main()
