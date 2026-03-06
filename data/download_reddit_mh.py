#!/usr/bin/env python3
"""
Download Reddit Mental Health Dataset from Zenodo
==================================================
Source: Low et al. (2020) — "Natural Language Processing Reveals Vulnerable
        Mental Health Support Groups and Heightened Health Anxiety on Reddit
        During COVID-19"
Zenodo: https://zenodo.org/records/3941387
License: CC-BY-4.0

Downloads the 2019 time window (Jan 1 - Apr 20, 2019) for all 28 subreddits,
then aggregates into a per-user engagement summary for pilot analysis.

Output:
  data/pilot/reddit_mental_health/user_engagement_summary.csv
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pandas as pd
import numpy as np

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(BASE_DIR, "data", "pilot", "reddit_mental_health")
RAW_DIR = os.path.join(OUT_DIR, "raw_2019")
os.makedirs(RAW_DIR, exist_ok=True)

ZENODO_BASE = "https://zenodo.org/api/records/3941387/files"

# All 28 subreddits in the dataset
SUBREDDITS = [
    # Mental health (15)
    "addiction", "adhd", "alcoholism", "anxiety", "autism",
    "bipolarreddit", "bpd", "depression", "EDAnonymous",
    "healthanxiety", "lonely", "mentalhealth", "ptsd",
    "schizophrenia", "socialanxiety", "suicidewatch",
    # Non-mental-health controls (12)
    "conspiracy", "divorce", "fitness", "guns", "jokes",
    "legaladvice", "meditation", "parenting", "personalfinance",
    "relationships", "teaching",
    # COVID-specific (only has 'post' period, skip for 2019)
    # "COVID19_support",
]

MH_SUBREDDITS = {
    "addiction", "adhd", "alcoholism", "anxiety", "autism",
    "bipolarreddit", "bpd", "depression", "edanonymous",
    "healthanxiety", "lonely", "mentalhealth", "ptsd",
    "schizophrenia", "socialanxiety", "suicidewatch",
}

# We download only the "2019" period (Jan 1 - Apr 20, 2019)
PERIOD = "2019"


# ============================================================
# DOWNLOAD
# ============================================================
def download_file(url: str, dest: str, max_retries: int = 2) -> bool:
    """Download a file using curl (more robust for large files)."""
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        size_mb = os.path.getsize(dest) / 1e6
        print(f"  SKIP (exists, {size_mb:.1f} MB): {os.path.basename(dest)}")
        return True

    for attempt in range(max_retries):
        print(f"  Downloading: {os.path.basename(dest)} ... ", end="", flush=True)
        try:
            t0 = time.time()
            result = subprocess.run(
                ["curl", "-sL", "--max-time", "300", "-o", dest, url],
                capture_output=True, text=True, timeout=360,
            )
            if result.returncode == 0 and os.path.exists(dest) and os.path.getsize(dest) > 1000:
                size_mb = os.path.getsize(dest) / 1e6
                elapsed = time.time() - t0
                print(f"OK ({size_mb:.1f} MB, {elapsed:.0f}s)")
                return True
            else:
                print(f"FAIL (attempt {attempt+1})")
                if os.path.exists(dest):
                    os.remove(dest)
        except subprocess.TimeoutExpired:
            print(f"TIMEOUT (attempt {attempt+1})")
            if os.path.exists(dest):
                os.remove(dest)
        except Exception as e:
            print(f"ERROR: {e}")
            if os.path.exists(dest):
                os.remove(dest)

    return False


def download_all() -> list[str]:
    """Download all 2019 subreddit files. Returns list of downloaded paths."""
    print("=" * 70)
    print(f"Downloading Reddit MH Dataset ({PERIOD} period)")
    print(f"Source: https://zenodo.org/records/3941387")
    print(f"Output: {RAW_DIR}")
    print("=" * 70)

    downloaded = []
    failed = []

    for sub in SUBREDDITS:
        fname = f"{sub}_{PERIOD}_features_tfidf_256.csv"
        url = f"{ZENODO_BASE}/{fname}/content"
        dest = os.path.join(RAW_DIR, fname)

        if download_file(url, dest):
            downloaded.append(dest)
        else:
            failed.append(sub)

    print(f"\n  Downloaded: {len(downloaded)} / {len(SUBREDDITS)}")
    if failed:
        print(f"  Failed: {failed}")

    return downloaded


# ============================================================
# AGGREGATE
# ============================================================
def aggregate_to_user_summary(file_paths: list[str]) -> pd.DataFrame:
    """
    Aggregate per-subreddit CSVs into a per-user engagement summary.

    For each user, we compute:
      - total_posts: total posts across all subreddits
      - n_subreddits: number of distinct subreddits posted in
      - mh_subreddit_posts: posts in mental health subreddits
      - non_mh_subreddit_posts: posts in non-MH subreddits
      - mh_subreddit_pct: fraction of posts in MH subreddits
      - subreddits_list: comma-separated list of subreddits
      - primary_subreddit: subreddit with most posts
    """
    print("\n" + "=" * 70)
    print("Aggregating per-user engagement summary...")
    print("=" * 70)

    # Collect (author, subreddit, count) tuples
    records = []
    total_posts = 0

    for path in file_paths:
        fname = os.path.basename(path)
        # Only need author and subreddit columns
        try:
            df = pd.read_csv(path, usecols=["author", "subreddit"])
        except (ValueError, KeyError):
            # Try reading first few cols if column names differ
            df = pd.read_csv(path, nrows=0)
            cols = [c for c in df.columns if c.lower() in ("author", "subreddit")]
            if len(cols) < 2:
                print(f"  SKIP (no author/subreddit): {fname}")
                continue
            df = pd.read_csv(path, usecols=cols)

        # Drop deleted users
        df = df[~df["author"].isin(["[deleted]", "AutoModerator", "[removed]"])]

        # Per-user-subreddit counts
        counts = df.groupby(["author", "subreddit"]).size().reset_index(name="n_posts")
        records.append(counts)
        total_posts += len(df)
        print(f"  {fname}: {len(df):,} posts, {df['author'].nunique():,} users")

    if not records:
        print("ERROR: No records to aggregate.")
        sys.exit(1)

    # Combine all
    all_counts = pd.concat(records, ignore_index=True)

    # Group by user across subreddits
    user_sub = all_counts.groupby(["author", "subreddit"])["n_posts"].sum().reset_index()

    print(f"\n  Total posts: {total_posts:,}")
    print(f"  Unique users: {user_sub['author'].nunique():,}")
    print(f"  Unique subreddits: {user_sub['subreddit'].nunique()}")

    # Per-user summary
    user_summary = user_sub.groupby("author").agg(
        total_posts=("n_posts", "sum"),
        n_subreddits=("subreddit", "nunique"),
        subreddits_list=("subreddit", lambda x: ",".join(sorted(x))),
    ).reset_index()

    # MH vs non-MH split
    user_sub["is_mh"] = user_sub["subreddit"].str.lower().isin(MH_SUBREDDITS)
    mh_counts = user_sub[user_sub["is_mh"]].groupby("author")["n_posts"].sum().rename("mh_subreddit_posts")
    non_mh_counts = user_sub[~user_sub["is_mh"]].groupby("author")["n_posts"].sum().rename("non_mh_subreddit_posts")

    user_summary = user_summary.merge(mh_counts, on="author", how="left")
    user_summary = user_summary.merge(non_mh_counts, on="author", how="left")
    user_summary["mh_subreddit_posts"] = user_summary["mh_subreddit_posts"].fillna(0).astype(int)
    user_summary["non_mh_subreddit_posts"] = user_summary["non_mh_subreddit_posts"].fillna(0).astype(int)

    user_summary["mh_subreddit_pct"] = (
        user_summary["mh_subreddit_posts"] / user_summary["total_posts"]
    ).fillna(0)

    # Primary subreddit (most posts)
    primary = user_sub.loc[user_sub.groupby("author")["n_posts"].idxmax()][["author", "subreddit"]]
    primary.columns = ["author", "primary_subreddit"]
    user_summary = user_summary.merge(primary, on="author", how="left")

    # Engagement tier
    user_summary["engagement_tier"] = np.where(
        user_summary["total_posts"] <= 1, "Lurker",
        np.where(user_summary["total_posts"] <= 5, "Light",
                 np.where(user_summary["total_posts"] <= 50, "Moderate", "Heavy")))

    # Rename author → user_id for consistency
    user_summary.rename(columns={"author": "user_id"}, inplace=True)

    return user_summary


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    # Step 1: Download
    downloaded = download_all()

    if not downloaded:
        print("\nERROR: No files downloaded. Check network connection.")
        sys.exit(1)

    # Step 2: Aggregate
    user_summary = aggregate_to_user_summary(downloaded)

    # Step 3: Save
    out_path = os.path.join(OUT_DIR, "user_engagement_summary.csv")
    user_summary.to_csv(out_path, index=False)

    print(f"\n{'=' * 70}")
    print(f"DONE: {out_path}")
    print(f"  Users: {len(user_summary):,}")
    print(f"  Columns: {list(user_summary.columns)}")
    print(f"\n  Engagement tier distribution:")
    for tier, n in user_summary["engagement_tier"].value_counts().items():
        print(f"    {tier:10s}: {n:7,} ({n/len(user_summary)*100:.1f}%)")
    print(f"\n  MH participation:")
    has_mh = (user_summary["mh_subreddit_pct"] > 0).sum()
    print(f"    Any MH subreddit: {has_mh:,} ({has_mh/len(user_summary)*100:.1f}%)")
    print(f"    MH-only users: {(user_summary['mh_subreddit_pct'] == 1).sum():,}")
    print(f"    Non-MH-only users: {(user_summary['mh_subreddit_pct'] == 0).sum():,}")
    print(f"    Mixed users: {((user_summary['mh_subreddit_pct'] > 0) & (user_summary['mh_subreddit_pct'] < 1)).sum():,}")
    print(f"{'=' * 70}")

    print(f"\n  Next: run the analysis pipeline:")
    print(f"    python3 analysis/pilot/sns_engagement/02_reddit_engagement_spectrum.py")


if __name__ == "__main__":
    main()
