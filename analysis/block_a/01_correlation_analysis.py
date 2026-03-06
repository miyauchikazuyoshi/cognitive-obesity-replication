#!/usr/bin/env python3
"""
国別時系列相関分析 (Section 2.1)
================================
各国について ad_proxy と depression_prevalence の時系列相関を算出し、
相関係数の分布と国別ランキングを可視化する。

対応: paper_figure_table_map.md Block A — 相関図・国別ランキング

Inputs:
  - data/macro/panel_merged.csv (or compatible macro panel)

Outputs:
  - results/block_a_correlations.json
  - results/figures/fig_a1_correlation_distribution.png
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
    suicide_col = find(["suicide", "suicide_rate", "sui_rate"])

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
    if suicide_col:
        rename[suicide_col] = "suicide"

    df = df.rename(columns=rename)
    return df


# ============================================================
# COUNTRY-LEVEL CORRELATIONS
# ============================================================
def compute_country_correlations(
    df: pd.DataFrame,
    x_var: str = "ad_proxy",
    y_var: str = "depression_prevalence",
    min_obs: int = 10,
) -> pd.DataFrame:
    """各国の時系列相関を算出。最低 min_obs 年分のデータがある国のみ。"""
    records = []
    for country, grp in df.groupby("country"):
        sub = grp[[x_var, y_var]].dropna()
        if len(sub) < min_obs:
            continue
        r, p = stats.pearsonr(sub[x_var], sub[y_var])
        records.append({
            "country": country,
            "n_years": len(sub),
            "r": r,
            "p_value": p,
        })
    return pd.DataFrame(records).sort_values("r", ascending=False)


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("BLOCK A — 国別時系列相関分析 (Section 2.1)")
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

    required = ["depression_prevalence", "ad_proxy"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: required columns missing: {', '.join(missing)}")
        sys.exit(1)

    # ---- 1) Ad proxy vs Depression ----
    print(f"\n--- Ad Proxy × Depression Prevalence ---")
    corr_dep = compute_country_correlations(df, "ad_proxy", "depression_prevalence")
    print(f"Countries with ≥10 years: {len(corr_dep)}")
    print(f"Mean r = {corr_dep['r'].mean():.4f}, Median r = {corr_dep['r'].median():.4f}")
    print(f"Positive: {(corr_dep['r'] > 0).sum()}, Negative: {(corr_dep['r'] < 0).sum()}")
    print(f"Significant (p<0.05): {(corr_dep['p_value'] < 0.05).sum()}")

    # Top/Bottom 10
    print(f"\nTop 10 (strongest positive):")
    for _, row in corr_dep.head(10).iterrows():
        sig = "*" if row["p_value"] < 0.05 else ""
        print(f"  {row['country']:<30s} r={row['r']:+.4f} (n={row['n_years']}) {sig}")

    print(f"\nBottom 10 (strongest negative):")
    for _, row in corr_dep.tail(10).iterrows():
        sig = "*" if row["p_value"] < 0.05 else ""
        print(f"  {row['country']:<30s} r={row['r']:+.4f} (n={row['n_years']}) {sig}")

    # ---- 2) Ad proxy vs Suicide (if available) ----
    corr_sui = None
    if "suicide" in df.columns:
        print(f"\n--- Ad Proxy × Suicide Rate ---")
        corr_sui = compute_country_correlations(df, "ad_proxy", "suicide")
        if len(corr_sui) > 0:
            print(f"Countries with ≥10 years: {len(corr_sui)}")
            print(f"Mean r = {corr_sui['r'].mean():.4f}, "
                  f"Median r = {corr_sui['r'].median():.4f}")
            print(f"Positive: {(corr_sui['r'] > 0).sum()}, "
                  f"Negative: {(corr_sui['r'] < 0).sum()}")
        else:
            print("  Not enough data for country-level suicide correlations.")
            corr_sui = None

    # ---- 3) Figure: Correlation distribution ----
    fig, axes = plt.subplots(1, 2 if corr_sui is not None else 1,
                             figsize=(12 if corr_sui is not None else 7, 5))
    if corr_sui is None:
        axes = [axes]

    # Depression panel
    ax = axes[0]
    ax.hist(corr_dep["r"], bins=30, color="#c23616", alpha=0.7, edgecolor="white")
    ax.axvline(corr_dep["r"].mean(), color="black", linestyle="--", linewidth=1.5,
               label=f"Mean r = {corr_dep['r'].mean():.3f}")
    ax.axvline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Pearson r (ad proxy vs depression)")
    ax.set_ylabel("Number of countries")
    ax.set_title("Country-Level Time-Series Correlations\n(Ad Proxy × Depression)")
    ax.legend()

    # Suicide panel (if available)
    if corr_sui is not None:
        ax2 = axes[1]
        ax2.hist(corr_sui["r"], bins=30, color="#0652DD", alpha=0.7, edgecolor="white")
        ax2.axvline(corr_sui["r"].mean(), color="black", linestyle="--", linewidth=1.5,
                    label=f"Mean r = {corr_sui['r'].mean():.3f}")
        ax2.axvline(0, color="gray", linestyle=":", linewidth=1)
        ax2.set_xlabel("Pearson r (ad proxy vs suicide)")
        ax2.set_ylabel("Number of countries")
        ax2.set_title("Country-Level Time-Series Correlations\n(Ad Proxy × Suicide)")
        ax2.legend()

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "fig_a1_correlation_distribution.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved figure: {fig_path}")

    # ---- 4) Save JSON ----
    output = {
        "depression": {
            "n_countries": len(corr_dep),
            "mean_r": round(float(corr_dep["r"].mean()), 4),
            "median_r": round(float(corr_dep["r"].median()), 4),
            "n_positive": int((corr_dep["r"] > 0).sum()),
            "n_negative": int((corr_dep["r"] < 0).sum()),
            "n_significant_05": int((corr_dep["p_value"] < 0.05).sum()),
            "top10": corr_dep.head(10).to_dict(orient="records"),
            "bottom10": corr_dep.tail(10).to_dict(orient="records"),
        },
    }
    if corr_sui is not None and len(corr_sui) > 0:
        output["suicide"] = {
            "n_countries": len(corr_sui),
            "mean_r": round(float(corr_sui["r"].mean()), 4),
            "median_r": round(float(corr_sui["r"].median()), 4),
            "n_positive": int((corr_sui["r"] > 0).sum()),
            "n_negative": int((corr_sui["r"] < 0).sum()),
        }

    out_path = os.path.join(RESULTS_DIR, "block_a_correlations.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
