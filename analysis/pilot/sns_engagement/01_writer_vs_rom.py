#!/usr/bin/env python3
"""
Pilot SNS Engagement — Writer vs ROM (Read-Only Member) Analysis
================================================================
Decomposes social media mental health impact by engagement type:
  - Writer (frequent posters)
  - ROM (passive consumers / lurkers)
  - Mixed

Primary data: Understanding Society Wave 11 (UK Household Longitudinal Study)
Fallback:     Synthetic data from 00_synthetic_data.py

Analyses:
  1. 2x2 typology (high/low viewing x high/low posting)
  2. GHQ-12 group comparison (ANOVA + Tukey HSD)
  3. OLS regression: GHQ-12 ~ posting + viewing + interaction + demographics
  4. Interaction plot

Outputs:
  - results/pilot_writer_vs_rom.json
  - results/figures/pilot_writer_vs_rom_interaction.png
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

import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..")
DATA_DIR = os.path.join(BASE_DIR, "data", "pilot")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# ============================================================
# DATA LOADING
# ============================================================
def load_data() -> tuple[pd.DataFrame, str]:
    """
    Load Understanding Society data or fall back to synthetic.

    Returns:
        (DataFrame, source_label)
    """
    # 1. Try Understanding Society data
    usoc_dir = os.path.join(DATA_DIR, "understanding_society")
    usoc_candidates = [
        "k_indresp.csv",      # Wave 11 individual response
        "wave11_indresp.csv",
        "usoc_wave11.csv",
    ]
    for fname in usoc_candidates:
        path = os.path.join(usoc_dir, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"Loaded Understanding Society: {path} (rows={len(df):,})")
            return df, "understanding_society"

    # 2. Fallback to synthetic
    synth_path = os.path.join(DATA_DIR, "sns_engagement_synthetic.csv")
    if os.path.exists(synth_path):
        df = pd.read_csv(synth_path)
        print(f"Loaded synthetic data: {synth_path} (rows={len(df):,})")
        print("  NOTE: Using synthetic data. For real analysis, place")
        print("  Understanding Society Wave 11 data in:")
        print(f"    {usoc_dir}/")
        print("  Download from: https://www.understandingsociety.ac.uk/")
        return df, "synthetic"

    # 3. No data found
    print("ERROR: No data found.")
    print(f"  Expected: {usoc_dir}/ or {synth_path}")
    print("  Run 00_synthetic_data.py first, or download Understanding Society.")
    sys.exit(1)


def normalize_survey(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """
    Normalize column names for analysis.
    Understanding Society has different naming than synthetic data.
    """
    if source == "synthetic":
        return df  # already clean

    # Understanding Society normalization
    col_map = {c.lower(): c for c in df.columns}

    def find(candidates: list[str]) -> str | None:
        for c in candidates:
            if c.lower() in col_map:
                return col_map[c.lower()]
        return None

    rename = {}
    # Map USOC variable names to our standard
    age_col = find(["k_age_dv", "k_dvage", "age_dv", "age"])
    sex_col = find(["k_sex", "k_sex_dv", "sex"])
    ghq_col = find(["k_scghq2_dv", "k_scghq1_dv", "scghq2_dv", "ghq12_score"])
    post_col = find(["k_socmedia2", "posting_freq"])  # USOC: frequency of posting
    view_col = find(["k_socmedia1", "viewing_freq"])  # USOC: frequency of viewing

    if age_col:
        rename[age_col] = "age"
    if sex_col:
        rename[sex_col] = "sex"
    if ghq_col:
        rename[ghq_col] = "ghq12_score"
    if post_col:
        rename[post_col] = "posting_freq"
    if view_col:
        rename[view_col] = "viewing_freq"

    df = df.rename(columns=rename)

    required = ["age", "sex", "ghq12_score", "posting_freq", "viewing_freq"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns after normalization: {missing}")
        print(f"  Available: {list(df.columns)[:30]}")
        sys.exit(1)

    return df


# ============================================================
# ANALYSIS 1: 2x2 TYPOLOGY
# ============================================================
def build_typology(df: pd.DataFrame) -> pd.DataFrame:
    """Create 2x2 typology: high/low viewing x high/low posting."""
    posting_med = df["posting_freq"].median()
    viewing_med = df["viewing_freq"].median()

    conditions = [
        (df["posting_freq"] > posting_med) & (df["viewing_freq"] > viewing_med),
        (df["posting_freq"] > posting_med) & (df["viewing_freq"] <= viewing_med),
        (df["posting_freq"] <= posting_med) & (df["viewing_freq"] > viewing_med),
        (df["posting_freq"] <= posting_med) & (df["viewing_freq"] <= viewing_med),
    ]
    labels = [
        "High-Post/High-View",
        "High-Post/Low-View",
        "Low-Post/High-View",   # ← ROM: this is the hypothesized risk group
        "Low-Post/Low-View",
    ]
    df = df.copy()
    df["typology"] = np.select(conditions, labels, default="Unclassified")
    return df


# ============================================================
# ANALYSIS 2: ANOVA + TUKEY
# ============================================================
def run_anova(df: pd.DataFrame) -> dict:
    """One-way ANOVA on GHQ-12 by typology, with Tukey HSD."""
    groups = [grp["ghq12_score"].dropna().values for _, grp in df.groupby("typology")]
    f_stat, p_value = stats.f_oneway(*groups)

    print(f"\n  ANOVA: F = {f_stat:.2f}, p = {p_value:.2e}")

    tukey = pairwise_tukeyhsd(df["ghq12_score"], df["typology"], alpha=0.05)
    print(f"\n  Tukey HSD:\n{tukey}")

    # Extract pairwise comparisons
    pairwise = []
    for i in range(len(tukey.summary().data) - 1):
        row = tukey.summary().data[i + 1]
        pairwise.append({
            "group1": str(row[0]),
            "group2": str(row[1]),
            "meandiff": float(row[2]),
            "p_adj": float(row[3]),
            "reject": bool(row[5]) if len(row) > 5 else float(row[3]) < 0.05,
        })

    return {
        "f_statistic": float(f_stat),
        "p_value": float(p_value),
        "tukey_pairwise": pairwise,
    }


# ============================================================
# ANALYSIS 3: OLS REGRESSION
# ============================================================
def run_regression(df: pd.DataFrame) -> dict:
    """OLS: GHQ-12 ~ posting + viewing + posting:viewing + age + sex."""
    sub = df.dropna(subset=["ghq12_score", "posting_freq", "viewing_freq", "age", "sex"])

    formula = "ghq12_score ~ posting_freq + viewing_freq + posting_freq:viewing_freq + age + C(sex)"
    model = smf.ols(formula, data=sub).fit()

    print(f"\n  OLS Regression (n={int(model.nobs):,}):")
    print(f"  R-squared = {model.rsquared:.3f}, Adj. R-squared = {model.rsquared_adj:.3f}")
    print(f"  AIC = {model.aic:.1f}")
    print()
    for var in ["posting_freq", "viewing_freq", "posting_freq:viewing_freq", "age"]:
        if var in model.params:
            b = model.params[var]
            se = model.bse[var]
            t = model.tvalues[var]
            p = model.pvalues[var]
            print(f"    {var:30s}: beta={b:7.3f}, SE={se:.3f}, t={t:6.2f}, p={p:.3e}")

    result = {
        "n": int(model.nobs),
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "aic": float(model.aic),
        "coefficients": {},
    }
    for var in model.params.index:
        result["coefficients"][var] = {
            "beta": float(model.params[var]),
            "se": float(model.bse[var]),
            "t_stat": float(model.tvalues[var]),
            "p_value": float(model.pvalues[var]),
        }

    return result


# ============================================================
# ANALYSIS 4: INTERACTION PLOT
# ============================================================
def plot_interaction(df: pd.DataFrame, source: str) -> str:
    """Plot posting_freq × viewing_freq → GHQ-12 interaction."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Mean GHQ-12 by typology
    typology_means = df.groupby("typology")["ghq12_score"].agg(["mean", "sem"])
    typology_means = typology_means.sort_values("mean", ascending=False)

    colors = {
        "Low-Post/High-View": "#d62728",   # Red — ROM risk group
        "High-Post/High-View": "#ff7f0e",
        "Low-Post/Low-View": "#2ca02c",
        "High-Post/Low-View": "#1f77b4",
    }
    bar_colors = [colors.get(t, "#999999") for t in typology_means.index]

    axes[0].barh(
        range(len(typology_means)),
        typology_means["mean"],
        xerr=typology_means["sem"] * 1.96,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.5,
    )
    axes[0].set_yticks(range(len(typology_means)))
    axes[0].set_yticklabels(typology_means.index, fontsize=9)
    axes[0].set_xlabel("Mean GHQ-12 Score (95% CI)")
    axes[0].set_title("Mental Health by Engagement Typology")
    axes[0].invert_yaxis()

    # Right: Interaction surface — viewing_freq as x-axis, lines by posting level
    posting_bins = pd.cut(df["posting_freq"], bins=[-.1, 1, 3, 7], labels=["Low (0-1)", "Mid (2-3)", "High (4-7)"])
    df_plot = df.copy()
    df_plot["posting_level"] = posting_bins

    for level, color in zip(["Low (0-1)", "Mid (2-3)", "High (4-7)"], ["#d62728", "#ff7f0e", "#1f77b4"]):
        sub = df_plot[df_plot["posting_level"] == level]
        means = sub.groupby("viewing_freq")["ghq12_score"].mean()
        axes[1].plot(means.index, means.values, "o-", color=color, label=f"Posting: {level}", markersize=5)

    axes[1].set_xlabel("Viewing Frequency (0-7)")
    axes[1].set_ylabel("Mean GHQ-12 Score")
    axes[1].set_title("Posting × Viewing Interaction")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    source_label = "Synthetic" if source == "synthetic" else "Understanding Society Wave 11"
    fig.suptitle(f"Writer vs ROM Analysis ({source_label})", fontsize=13, fontweight="bold")
    plt.tight_layout()

    fig_path = os.path.join(FIG_DIR, "pilot_writer_vs_rom_interaction.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {fig_path}")

    return fig_path


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    print("=" * 70)
    print("PILOT: Writer vs ROM — SNS Engagement & Mental Health")
    print("=" * 70)

    # Load data
    df, source = load_data()
    df = normalize_survey(df, source)

    # Drop missing
    required = ["ghq12_score", "posting_freq", "viewing_freq", "age", "sex"]
    before = len(df)
    df = df.dropna(subset=required)
    print(f"\n  After dropping missing: {len(df):,} / {before:,} rows")

    # Build typology
    df = build_typology(df)
    print("\n  Typology distribution:")
    for typ, count in df["typology"].value_counts().items():
        mean_ghq = df[df["typology"] == typ]["ghq12_score"].mean()
        print(f"    {typ:25s}: n={count:5d}, mean GHQ-12={mean_ghq:.1f}")

    # Analysis 1: ANOVA + Tukey
    print("\n" + "-" * 50)
    print("Analysis 1: ANOVA + Tukey HSD")
    print("-" * 50)
    anova_results = run_anova(df)

    # Analysis 2: OLS regression
    print("\n" + "-" * 50)
    print("Analysis 2: OLS Regression")
    print("-" * 50)
    regression_results = run_regression(df)

    # Analysis 3: Interaction plot
    print("\n" + "-" * 50)
    print("Analysis 3: Interaction Plot")
    print("-" * 50)
    fig_path = plot_interaction(df, source)

    # Key finding summary
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Find ROM group (Low-Post/High-View) vs others
    rom = df[df["typology"] == "Low-Post/High-View"]["ghq12_score"]
    writer = df[df["typology"] == "High-Post/High-View"]["ghq12_score"]
    if len(rom) > 0 and len(writer) > 0:
        t, p = stats.ttest_ind(rom, writer)
        d = (rom.mean() - writer.mean()) / np.sqrt((rom.std()**2 + writer.std()**2) / 2)
        print(f"  ROM vs Writer (High-View): d={d:.2f}, t={t:.2f}, p={p:.3e}")

    interaction_coef = regression_results["coefficients"].get("posting_freq:viewing_freq", {})
    if interaction_coef:
        print(f"  Interaction (posting×viewing): beta={interaction_coef['beta']:.3f}, "
              f"p={interaction_coef['p_value']:.3e}")

    # Save JSON
    output = {
        "source": source,
        "n": len(df),
        "typology_summary": {
            typ: {
                "n": int(count),
                "ghq12_mean": float(df[df["typology"] == typ]["ghq12_score"].mean()),
                "ghq12_sd": float(df[df["typology"] == typ]["ghq12_score"].std()),
            }
            for typ, count in df["typology"].value_counts().items()
        },
        "anova": anova_results,
        "regression": regression_results,
        "figure": fig_path,
    }

    json_path = os.path.join(RESULTS_DIR, "pilot_writer_vs_rom.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {json_path}")

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
