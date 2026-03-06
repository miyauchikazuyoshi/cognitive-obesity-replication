#!/usr/bin/env python3
"""
Depression vs Homicide: Global Analysis & Internalization-Externalization
=========================================================================
Generates two key figures cut from v9 that should be restored in v9.4:

  (1) fig_depression_homicide_global.png
      4-panel: global time series, cross-sectional scatter,
      within-country correlation distribution, key-country divergence

  (2) fig_internalization_externalization.png
      Internalization-Externalization symmetry analysis:
      within-country Pearson r distributions for dep-hom, sui-hom, dep-sui
      + global Z-score time series

Inputs:
  - data/macro/panel_merged.csv

Outputs:
  - results/figures/fig_depression_homicide_global.png
  - results/figures/fig_internalization_externalization.png
  - results/block_a_depression_homicide.json
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
from matplotlib.gridspec import GridSpec

# ============================================================
# PATHS
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "macro")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})


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
    sys.exit(1)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Use specific column names; avoid duplicates
    renames = {}
    seen_targets = set()
    for c in df.columns:
        cl = c.lower().strip()
        if cl == "depression_prevalence" and "depression" not in seen_targets:
            renames[c] = "depression"
            seen_targets.add("depression")
        elif cl == "homicide" and "homicide" not in seen_targets:
            renames[c] = "homicide"
            seen_targets.add("homicide")
        elif cl == "suicide" and "suicide" not in seen_targets:
            renames[c] = "suicide"
            seen_targets.add("suicide")
        elif cl in ("country", "entity", "country_name") and "country" not in seen_targets:
            renames[c] = "country"
            seen_targets.add("country")

    df = df.rename(columns=renames)
    # Drop duplicate columns if any
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def classify_income(row_gdp: float) -> str:
    """Rough income classification based on GDP per capita (2017 PPP)."""
    if pd.isna(row_gdp):
        return "Unknown"
    if row_gdp >= 20000:
        return "High"
    elif row_gdp >= 6000:
        return "Upper-middle"
    elif row_gdp >= 2000:
        return "Lower-middle"
    else:
        return "Low"


# ============================================================
# WITHIN-COUNTRY CORRELATIONS
# ============================================================
def compute_within_country_corr(df: pd.DataFrame, var1: str, var2: str,
                                min_years: int = 10):
    """Compute within-country temporal Pearson r for each country."""
    results = []
    for country, grp in df.groupby("country"):
        sub = grp.dropna(subset=[var1, var2])
        if len(sub) < min_years:
            continue
        r, p = stats.pearsonr(sub[var1], sub[var2])
        # Get median GDP for income classification
        med_gdp = grp["gdp"].median() if "gdp" in grp.columns else np.nan
        results.append({
            "country": country,
            "n_years": len(sub),
            "r": r,
            "p_value": p,
            "income": classify_income(med_gdp),
        })
    return pd.DataFrame(results)


# ============================================================
# FIGURE 1: Depression vs Homicide Global Analysis
# ============================================================
def fig_depression_homicide_global(df: pd.DataFrame) -> dict:
    """Generate 4-panel Depression vs Homicide global analysis."""

    fig = plt.figure(figsize=(14, 11))
    gs = GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # ---- Panel A: Global time series ----
    ax_a = fig.add_subplot(gs[0, 0])
    global_ts = (df.groupby("year")[["depression", "homicide"]]
                 .mean().dropna(how="all"))
    global_ts = global_ts.loc[1990:]

    color_dep = "#2166AC"
    color_hom = "#B2182B"

    ax_a.plot(global_ts.index, global_ts["depression"], '-o', color=color_dep,
              markersize=3, label="Depression prevalence (%)", linewidth=1.5)
    ax_a.set_ylabel("Depression prevalence (%)", color=color_dep)
    ax_a.tick_params(axis='y', labelcolor=color_dep)

    ax2 = ax_a.twinx()
    ax2.plot(global_ts.index, global_ts["homicide"], '-s', color=color_hom,
             markersize=3, label="Homicide rate (per 100k)", linewidth=1.5)
    ax2.set_ylabel("Homicide rate (per 100k)", color=color_hom)
    ax2.tick_params(axis='y', labelcolor=color_hom)

    # Compute global correlation
    both = global_ts.dropna()
    if len(both) > 3:
        r_global, p_global = stats.pearsonr(both["depression"], both["homicide"])
    else:
        r_global, p_global = np.nan, np.nan

    ax_a.set_title(f"A. Global Trends (1990–2023)\n"
                   f"Pearson r = {r_global:.3f}, p = {p_global:.3f}")
    ax_a.set_xlabel("Year")

    # ---- Panel B: Cross-sectional scatter (latest year with data) ----
    ax_b = fig.add_subplot(gs[0, 1])
    latest = df[df["year"] >= 2019].copy()
    cross = (latest.groupby("country")[["depression", "homicide"]]
             .mean().dropna())

    ax_b.scatter(cross["depression"], cross["homicide"],
                 alpha=0.5, s=20, color="#7570B3", edgecolors="none")
    if len(cross) > 5:
        r_cross, p_cross = stats.pearsonr(cross["depression"],
                                           cross["homicide"])
        # Log scale for homicide (large range)
        ax_b.set_yscale("log")
    else:
        r_cross, p_cross = np.nan, np.nan

    ax_b.set_title(f"B. Cross-Sectional (2019+)\n"
                   f"Pearson r = {r_cross:.3f}, p = {p_cross:.3f}")
    ax_b.set_xlabel("Depression prevalence (%)")
    ax_b.set_ylabel("Homicide rate per 100k (log scale)")

    # ---- Panel C: Within-country correlation distribution ----
    ax_c = fig.add_subplot(gs[1, 0])
    corr_df = compute_within_country_corr(df, "depression", "homicide")

    if len(corr_df) > 0:
        corr_sorted = corr_df.sort_values("r")
        n_total = len(corr_sorted)
        colors = []
        for _, row in corr_sorted.iterrows():
            if row["p_value"] < 0.05 and row["r"] < 0:
                colors.append("#B2182B")  # sig negative = red
            elif row["p_value"] < 0.05 and row["r"] > 0:
                colors.append("#2166AC")  # sig positive = blue
            else:
                colors.append("#999999")  # non-significant = grey

        ax_c.bar(range(n_total), corr_sorted["r"].values, color=colors,
                 width=1.0, edgecolor="none")
        med_r = corr_sorted["r"].median()
        ax_c.axhline(med_r, color="red", linestyle="--", linewidth=1,
                     label=f"Median r = {med_r:.3f}")
        ax_c.axhline(0, color="black", linewidth=0.5)

        n_sig_neg = sum(1 for c in colors if c == "#B2182B")
        n_sig_pos = sum(1 for c in colors if c == "#2166AC")
        n_ns = sum(1 for c in colors if c == "#999999")

        ax_c.set_title(f"C. Within-Country Correlations (n={n_total})\n"
                       f"Red=sig. negative, Blue=sig. positive, Grey=n.s.")
        ax_c.set_xlabel("Countries (sorted by r)")
        ax_c.set_ylabel("Within-country Pearson r")
        ax_c.legend(fontsize=8)
    else:
        ax_c.text(0.5, 0.5, "Insufficient data", ha='center', va='center',
                  transform=ax_c.transAxes)

    # ---- Panel D: Key countries divergence time series ----
    ax_d = fig.add_subplot(gs[1, 1])
    key_countries = ["United States", "Japan", "Brazil",
                     "Germany", "South Africa", "South Korea"]

    for country in key_countries:
        cdata = df[df["country"] == country].copy()
        cdata = cdata.dropna(subset=["depression", "homicide"]).sort_values("year")
        if len(cdata) < 5:
            continue
        # Z-score divergence: Z(depression) - Z(homicide)
        if cdata["depression"].std() > 0 and cdata["homicide"].std() > 0:
            z_dep = (cdata["depression"] - cdata["depression"].mean()) / cdata["depression"].std()
            z_hom = (cdata["homicide"] - cdata["homicide"].mean()) / cdata["homicide"].std()
            divergence = z_dep - z_hom
            ax_d.plot(cdata["year"], divergence, '-', linewidth=1.2, label=country)

    ax_d.axhline(0, color="black", linewidth=0.5, linestyle="-")
    ax_d.set_title("D. Depression–Homicide Divergence\n"
                   "(positive = depression up, homicide down)")
    ax_d.set_xlabel("Year")
    ax_d.set_ylabel("Z(Depression) − Z(Homicide)")
    ax_d.legend(fontsize=7, loc="best", ncol=2)

    fig.suptitle("Depression Prevalence vs Homicide Rate: Global Analysis (1990–2023)",
                 fontsize=13, fontweight="bold", y=0.98)

    outpath = os.path.join(FIG_DIR, "fig_depression_homicide_global.png")
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"Saved: {outpath}")

    return {
        "global_r": float(r_global) if not np.isnan(r_global) else None,
        "global_p": float(p_global) if not np.isnan(p_global) else None,
        "cross_sectional_r": float(r_cross) if not np.isnan(r_cross) else None,
        "n_countries_corr": len(corr_df),
        "median_within_r": float(corr_df["r"].median()) if len(corr_df) > 0 else None,
    }


# ============================================================
# FIGURE 2: Internalization-Externalization Symmetry
# ============================================================
def fig_internalization_externalization(df: pd.DataFrame) -> dict:
    """Generate Internalization-Externalization symmetry analysis.

    Top row: within-country Pearson r distributions for 3 variable pairs
             (Depression-Homicide, Suicide-Homicide, Depression-Suicide)
    Bottom row: global Z-score time series for same 3 pairs
    """
    pairs = [
        ("depression", "homicide", "Depression vs Homicide"),
        ("suicide", "homicide", "Suicide vs Homicide"),
        ("depression", "suicide", "Depression vs Suicide"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    json_out = {}

    for col_idx, (v1, v2, title) in enumerate(pairs):
        # --- Top row: correlation distribution ---
        ax_top = axes[0, col_idx]
        corr_df = compute_within_country_corr(df, v1, v2)

        if len(corr_df) == 0:
            ax_top.text(0.5, 0.5, f"No data for\n{v1} vs {v2}",
                        ha='center', va='center', transform=ax_top.transAxes)
            axes[1, col_idx].text(0.5, 0.5, "No data",
                                  ha='center', va='center',
                                  transform=axes[1, col_idx].transAxes)
            continue

        corr_sorted = corr_df.sort_values("r")
        n = len(corr_sorted)
        colors = []
        for _, row in corr_sorted.iterrows():
            if row["p_value"] < 0.05 and row["r"] < 0:
                colors.append("#B2182B")
            elif row["p_value"] < 0.05 and row["r"] > 0:
                colors.append("#2166AC")
            else:
                colors.append("#999999")

        ax_top.bar(range(n), corr_sorted["r"].values, color=colors,
                   width=1.0, edgecolor="none")
        med_r = corr_sorted["r"].median()
        n_sig_neg = sum(1 for c in colors if c == "#B2182B")
        n_sig_pos = sum(1 for c in colors if c == "#2166AC")

        ax_top.axhline(med_r, color="orange", linestyle="--", linewidth=1.5)
        ax_top.axhline(0, color="black", linewidth=0.5)
        ax_top.set_ylim(-1, 1)

        ax_top.text(0.02, 0.95,
                    f"Median r = {med_r:.3f}\n"
                    f"Sig−: {n_sig_neg}, Sig+: {n_sig_pos}",
                    transform=ax_top.transAxes, fontsize=8,
                    va='top', bbox=dict(boxstyle='round,pad=0.3',
                                        facecolor='white', alpha=0.8))
        ax_top.set_title(title, fontweight='bold')
        if col_idx == 0:
            ax_top.set_ylabel("Within-country Pearson r")
        ax_top.set_xlabel(f"Countries (n={n})")

        # --- Bottom row: global Z-score time series ---
        ax_bot = axes[1, col_idx]
        global_ts = (df.groupby("year")[[v1, v2]]
                     .mean().dropna(how="all").loc[1990:])

        if len(global_ts) > 5:
            for var, color, label in [(v1, "#2166AC", v1.title()),
                                      (v2, "#B2182B", v2.title())]:
                series = global_ts[var].dropna()
                if len(series) > 2 and series.std() > 0:
                    z = (series - series.mean()) / series.std()
                    ax_bot.plot(z.index, z.values, '-', linewidth=1.5,
                                color=color, label=label)

            r_ts, p_ts = stats.pearsonr(
                global_ts[v1].dropna().values[:min(len(global_ts[v1].dropna()),
                                                    len(global_ts[v2].dropna()))],
                global_ts[v2].dropna().values[:min(len(global_ts[v1].dropna()),
                                                    len(global_ts[v2].dropna()))]
            ) if len(global_ts.dropna()) > 3 else (np.nan, np.nan)

            both_ts = global_ts.dropna()
            if len(both_ts) > 3:
                r_ts, p_ts = stats.pearsonr(both_ts[v1], both_ts[v2])
            else:
                r_ts, p_ts = np.nan, np.nan

            ax_bot.set_title(f"Global trends (r = {r_ts:.3f}, p = {p_ts:.3f})",
                             fontsize=9)
        else:
            r_ts, p_ts = np.nan, np.nan

        ax_bot.axhline(0, color="black", linewidth=0.5, linestyle="-")
        ax_bot.legend(fontsize=8)
        ax_bot.set_xlabel("Year")
        if col_idx == 0:
            ax_bot.set_ylabel("Z-score (global mean)")

        json_out[f"{v1}_vs_{v2}"] = {
            "n_countries": n,
            "median_r": float(med_r),
            "n_sig_negative": n_sig_neg,
            "n_sig_positive": n_sig_pos,
            "global_ts_r": float(r_ts) if not np.isnan(r_ts) else None,
        }

    fig.suptitle("Internalization-Externalization Symmetry Analysis (1990–2023)",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    outpath = os.path.join(FIG_DIR, "fig_internalization_externalization.png")
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"Saved: {outpath}")

    return json_out


# ============================================================
# JAPAN CASE STUDY STATS
# ============================================================
def japan_case_study(df: pd.DataFrame) -> dict:
    """Extract Japan-specific statistics for paper anchor text."""
    jp = df[df["country"] == "Japan"].copy().sort_values("year")
    jp = jp.dropna(subset=["depression", "homicide"])

    if len(jp) < 5:
        print("WARNING: Insufficient Japan data")
        return {}

    r, p = stats.pearsonr(jp["depression"], jp["homicide"])

    # Get start/end values
    earliest = jp.iloc[0]
    latest = jp.iloc[-1]

    # Rank among all countries
    corr_all = compute_within_country_corr(df, "depression", "homicide")
    corr_all_sorted = corr_all.sort_values("r")
    rank = None
    for i, (_, row) in enumerate(corr_all_sorted.iterrows()):
        if row["country"] == "Japan":
            rank = i + 1
            break

    result = {
        "country": "Japan",
        "n_years": len(jp),
        "r": float(r),
        "p_value": float(p),
        "depression_start": float(earliest["depression"]),
        "depression_end": float(latest["depression"]),
        "depression_year_start": int(earliest["year"]),
        "depression_year_end": int(latest["year"]),
        "homicide_start": float(earliest["homicide"]),
        "homicide_end": float(latest["homicide"]),
        "rank_of_n": f"{rank}/{len(corr_all)}" if rank else None,
    }

    print("\n=== Japan Case Study ===")
    for k, v in result.items():
        print(f"  {k}: {v}")

    return result


# ============================================================
# MAIN
# ============================================================
def main():
    df = load_panel()
    df = normalize_columns(df)

    # Filter to valid years
    df = df[df["year"].between(1990, 2023)]

    print(f"\nPanel: {df['country'].nunique()} countries, "
          f"{df['year'].nunique()} years, {len(df):,} obs")
    print(f"Depression coverage: {int(df['depression'].notna().sum()):,}")
    print(f"Homicide coverage: {int(df['homicide'].notna().sum()):,}")
    print(f"Suicide coverage: {int(df['suicide'].notna().sum()):,}")

    # Generate figures
    print("\n--- Figure: Depression vs Homicide Global ---")
    stats1 = fig_depression_homicide_global(df)

    print("\n--- Figure: Internalization-Externalization ---")
    stats2 = fig_internalization_externalization(df)

    print("\n--- Japan Case Study ---")
    jp_stats = japan_case_study(df)

    # Save JSON
    output = {
        "depression_homicide_global": stats1,
        "internalization_externalization": stats2,
        "japan_case_study": jp_stats,
    }
    json_path = os.path.join(RESULTS_DIR, "block_a_depression_homicide.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {json_path}")


if __name__ == "__main__":
    main()
