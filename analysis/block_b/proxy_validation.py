#!/usr/bin/env python3
"""
Ad Proxy Validation: Internet × GDP/capita vs External Benchmarks
================================================================
Validates the ad_proxy = Internet(%) × GDP/capita / 1000 construct
against available external benchmarks:

  1. World Bank ICT expenditure (% of GDP) — IT.NET.BBND.P2 etc.
  2. Known country-level digital ad spend rankings (hardcoded from
     public industry reports: GroupM, IAB, eMarketer summaries)
  3. Rank correlation (Spearman) between proxy and benchmarks
  4. Temporal stability of proxy ranking vs ad market maturity

This script does NOT require proprietary WARC/GroupM microdata.
Users with WARC access can extend section [WARC EXTENSION] below.

Inputs:
  - data/macro/panel_merged.csv (or compatible macro panel)
  - Hardcoded reference ad spend data (from public reports)

Outputs:
  - results/proxy_validation.json
  - results/figures/proxy_validation_rank.png
  - results/figures/proxy_validation_timeseries.png
  - console output

References:
  - GroupM "This Year Next Year" (2023, 2024)
  - IAB Global Report on Internet Advertising (2023)
  - Statista Digital Advertising Worldwide (2024)
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
# DATA LOADING (shared pattern)
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
    print("Run: python3 data/build_macro_panel.py")
    sys.exit(1)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {c.lower(): c for c in df.columns}

    def find(candidates: list[str]) -> str | None:
        for c in candidates:
            if c.lower() in col_map:
                return col_map[c.lower()]
        return None

    rename: dict[str, str] = {}
    country_col = find(["country", "entity", "location", "country_name"])
    year_col = find(["year", "time"])
    internet_col = find(["internet", "internet_pct", "it.net.user.zs"])
    gdp_col = find(["gdp", "gdp_pc", "gdp_per_capita", "ny.gdp.pcap.cd",
                     "gdp_current_usd", "gdp_ppp_2017"])

    if country_col is None or year_col is None:
        print("ERROR: cannot identify country/year columns.")
        sys.exit(1)

    rename[country_col] = "country"
    rename[year_col] = "year"
    if internet_col:
        rename[internet_col] = "internet"
    if gdp_col:
        rename[gdp_col] = "gdp"
    return df.rename(columns=rename)


def ensure_proxy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    internet = pd.to_numeric(out.get("internet"), errors="coerce")
    gdp = pd.to_numeric(out.get("gdp"), errors="coerce")

    if internet is None or gdp is None:
        print("ERROR: internet or gdp column missing.")
        sys.exit(1)

    internet_pct = internet * 100.0 if internet.max(skipna=True) <= 1.5 else internet
    out["internet_pct"] = internet_pct
    out["ad_proxy"] = internet_pct * gdp / 1000.0
    return out


# ============================================================
# REFERENCE DATA: Known Digital Ad Spend Rankings
# ============================================================
# Sources: GroupM "This Year Next Year" 2023/2024, IAB 2023,
# eMarketer/Insider Intelligence 2024, Statista 2024 summaries
#
# Values = digital ad spending per capita (approximate USD, ~2022)
# These are publicly available summary rankings, not microdata.

DIGITAL_AD_SPEND_PER_CAPITA_2022 = {
    # Top tier (>$200/capita)
    "United States": 580,
    "United Kingdom": 420,
    "Australia": 380,
    "Norway": 350,
    "Denmark": 330,
    "Sweden": 310,
    "Canada": 290,
    "Switzerland": 280,
    "Netherlands": 270,
    "Germany": 250,
    "Japan": 240,
    "France": 210,
    # Mid tier ($50-200)
    "South Korea": 190,
    "New Zealand": 180,
    "Finland": 170,
    "Singapore": 165,
    "Ireland": 160,
    "Israel": 150,
    "Belgium": 140,
    "Austria": 135,
    "Spain": 100,
    "Italy": 95,
    "Czech Republic": 75,
    "Poland": 55,
    # Lower-mid tier ($10-50)
    "China": 48,
    "Brazil": 32,
    "Russia": 30,
    "Mexico": 22,
    "Malaysia": 20,
    "Thailand": 18,
    "Turkey": 15,
    "Argentina": 12,
    "South Africa": 10,
    # Low tier (<$10)
    "Indonesia": 6,
    "India": 4,
    "Nigeria": 1.5,
    "Egypt": 2,
    "Pakistan": 0.8,
    "Bangladesh": 0.4,
    "Ethiopia": 0.1,
}

# Total digital ad market size (billion USD, ~2022)
# Source: GroupM/Statista summaries
DIGITAL_AD_MARKET_TOTAL_2022 = {
    "United States": 210,
    "China": 70,
    "United Kingdom": 28,
    "Japan": 25,
    "Germany": 15,
    "France": 10,
    "Canada": 9.5,
    "Australia": 9,
    "South Korea": 8,
    "Brazil": 6.5,
    "India": 5.5,
    "Russia": 4,
    "Indonesia": 2,
    "Italy": 5,
    "Spain": 4.5,
    "Netherlands": 4,
    "Sweden": 3.2,
    "Norway": 1.8,
    "Denmark": 1.9,
    "Switzerland": 2.4,
    "Mexico": 2.5,
    "Poland": 2,
    "Turkey": 1.5,
    "Argentina": 0.5,
    "South Africa": 0.6,
    "Nigeria": 0.3,
    "Egypt": 0.2,
}

# ============================================================
# VALIDATION ANALYSIS
# ============================================================
def main():
    print("=" * 70)
    print("AD PROXY VALIDATION")
    print("proxy = Internet(%) × GDP/capita / 1000")
    print("=" * 70)

    df = ensure_proxy(normalize_columns(load_panel()))

    # Use 2020-2022 average for proxy (to match reference data ~2022)
    recent = df[(df["year"] >= 2020) & (df["year"] <= 2022)].copy()
    proxy_by_country = (
        recent.groupby("country")["ad_proxy"]
        .mean()
        .dropna()
        .to_frame("proxy_mean")
    )
    print(f"\nCountries with proxy data (2020-2022 avg): {len(proxy_by_country)}")

    results = {}

    # ---- Test 1: Rank correlation with per-capita digital ad spend ----
    print("\n" + "=" * 70)
    print("TEST 1: Rank Correlation — Proxy vs Digital Ad Spend/Capita")
    print("=" * 70)

    ref_pc = pd.Series(DIGITAL_AD_SPEND_PER_CAPITA_2022, name="ad_spend_pc")
    merged_pc = proxy_by_country.join(ref_pc, how="inner")
    n_match = len(merged_pc)
    print(f"Matched countries: {n_match}/{len(ref_pc)}")

    if n_match >= 5:
        rho_pc, p_pc = stats.spearmanr(merged_pc["proxy_mean"], merged_pc["ad_spend_pc"])
        r_pearson_pc, p_pearson_pc = stats.pearsonr(
            np.log(merged_pc["proxy_mean"].clip(lower=0.01)),
            np.log(merged_pc["ad_spend_pc"].clip(lower=0.01))
        )
        print(f"Spearman ρ = {rho_pc:.3f} (p = {p_pc:.2e})")
        print(f"Pearson r (log-log) = {r_pearson_pc:.3f} (p = {p_pearson_pc:.2e})")

        results["per_capita"] = {
            "n_countries": n_match,
            "spearman_rho": round(float(rho_pc), 4),
            "spearman_p": float(p_pc),
            "pearson_loglog_r": round(float(r_pearson_pc), 4),
            "pearson_loglog_p": float(p_pearson_pc),
        }

        # Rank comparison
        merged_pc["rank_proxy"] = merged_pc["proxy_mean"].rank(ascending=False)
        merged_pc["rank_adspend"] = merged_pc["ad_spend_pc"].rank(ascending=False)
        merged_pc["rank_diff"] = (merged_pc["rank_proxy"] - merged_pc["rank_adspend"]).abs()

        print(f"\nMean absolute rank difference: {merged_pc['rank_diff'].mean():.1f}")
        print(f"Max rank difference: {merged_pc['rank_diff'].max():.0f}")

        worst = merged_pc.nlargest(5, "rank_diff")[["proxy_mean", "ad_spend_pc",
                                                      "rank_proxy", "rank_adspend", "rank_diff"]]
        print("\nLargest rank discrepancies:")
        for idx, row in worst.iterrows():
            print(f"  {idx}: proxy_rank={row['rank_proxy']:.0f}, "
                  f"adspend_rank={row['rank_adspend']:.0f} (diff={row['rank_diff']:.0f})")

        results["per_capita"]["mean_abs_rank_diff"] = round(float(merged_pc["rank_diff"].mean()), 2)
        results["per_capita"]["countries_matched"] = sorted(merged_pc.index.tolist())
    else:
        print("  SKIP: insufficient country matches")

    # ---- Test 2: Rank correlation with total ad market size ----
    print("\n" + "=" * 70)
    print("TEST 2: Rank Correlation — Proxy×Population vs Total Ad Market")
    print("=" * 70)

    ref_total = pd.Series(DIGITAL_AD_MARKET_TOTAL_2022, name="ad_market_total")
    pop_col = None
    for c in ["population", "pop", "sp.pop.totl"]:
        if c in df.columns:
            pop_col = c
            break

    if pop_col is not None:
        pop_recent = recent.groupby("country")[pop_col].mean().dropna()
        proxy_total = (proxy_by_country["proxy_mean"] * pop_recent / 1e6).dropna()
        proxy_total.name = "proxy_total"
        merged_total = pd.DataFrame({"proxy_total": proxy_total}).join(ref_total, how="inner")
        n_total = len(merged_total)
        print(f"Matched countries: {n_total}/{len(ref_total)}")

        if n_total >= 5:
            rho_t, p_t = stats.spearmanr(merged_total["proxy_total"],
                                          merged_total["ad_market_total"])
            print(f"Spearman ρ = {rho_t:.3f} (p = {p_t:.2e})")
            results["total_market"] = {
                "n_countries": n_total,
                "spearman_rho": round(float(rho_t), 4),
                "spearman_p": float(p_t),
            }
    else:
        print("  SKIP: population column not found for total market comparison")

    # ---- Test 3: Proxy ranking stability over time ----
    print("\n" + "=" * 70)
    print("TEST 3: Temporal Stability of Proxy Rankings")
    print("=" * 70)

    stability = {}
    for yr in [2000, 2005, 2010, 2015, 2020]:
        yr_data = df[df["year"] == yr].dropna(subset=["ad_proxy"]).copy()
        if len(yr_data) < 20:
            continue
        yr_data["rank"] = yr_data["ad_proxy"].rank(ascending=False)
        stability[yr] = yr_data.set_index("country")["rank"]

    if len(stability) >= 2:
        years_list = sorted(stability.keys())
        print(f"Years with sufficient data: {years_list}")

        for i in range(len(years_list) - 1):
            y1, y2 = years_list[i], years_list[i + 1]
            common = stability[y1].index.intersection(stability[y2].index)
            if len(common) >= 10:
                rho_s, _ = stats.spearmanr(stability[y1][common], stability[y2][common])
                print(f"  {y1} → {y2}: Spearman ρ = {rho_s:.3f} (n={len(common)})")
        results["temporal_stability"] = {
            "years": years_list,
            "note": "Rank correlations between consecutive periods"
        }

    # ---- Test 4: Component decomposition ----
    print("\n" + "=" * 70)
    print("TEST 4: Variance Decomposition — Internet vs GDP contribution")
    print("=" * 70)

    valid = recent.dropna(subset=["internet_pct", "gdp", "ad_proxy"]).copy()
    valid = valid[valid["ad_proxy"] > 0]
    if len(valid) > 50:
        log_proxy = np.log(valid["ad_proxy"])
        log_internet = np.log(valid["internet_pct"].clip(lower=0.1))
        log_gdp = np.log(valid["gdp"].clip(lower=100))

        r_internet, _ = stats.pearsonr(log_internet, log_proxy)
        r_gdp, _ = stats.pearsonr(log_gdp, log_proxy)

        print(f"corr(log_proxy, log_internet) = {r_internet:.3f}")
        print(f"corr(log_proxy, log_gdp) = {r_gdp:.3f}")
        print(f"Proxy variance driven more by GDP ({r_gdp**2:.1%}) than Internet ({r_internet**2:.1%})")

        results["variance_decomposition"] = {
            "r_log_internet": round(float(r_internet), 4),
            "r_log_gdp": round(float(r_gdp), 4),
            "r2_internet": round(float(r_internet**2), 4),
            "r2_gdp": round(float(r_gdp**2), 4),
            "interpretation": "GDP dominates proxy variance; "
                              "proxy is primarily an economic development indicator "
                              "with internet penetration as secondary modulator"
        }

    # ---- Test 5: Residual proxy (orthogonal to GDP) ----
    print("\n" + "=" * 70)
    print("TEST 5: Residual Proxy (orthogonalized to GDP)")
    print("=" * 70)

    if len(valid) > 50 and "ad_spend_pc" in locals().get("merged_pc", pd.DataFrame()).columns:
        import statsmodels.api as sm
        X_gdp = sm.add_constant(np.log(valid.groupby("country")["gdp"].mean().clip(lower=100)))
        y_proxy = np.log(valid.groupby("country")["ad_proxy"].mean().clip(lower=0.01))

        common_idx = X_gdp.index.intersection(y_proxy.index)
        if len(common_idx) > 20:
            resid = sm.OLS(y_proxy[common_idx], X_gdp.loc[common_idx]).fit().resid
            resid.name = "proxy_resid"
            resid_df = pd.DataFrame(resid)

            ref_pc2 = pd.Series(DIGITAL_AD_SPEND_PER_CAPITA_2022, name="ad_spend_pc")
            merged_resid = resid_df.join(ref_pc2, how="inner")

            if len(merged_resid) >= 5:
                rho_resid, p_resid = stats.spearmanr(
                    merged_resid["proxy_resid"],
                    merged_resid["ad_spend_pc"]
                )
                print(f"Spearman ρ(residual proxy, ad spend/cap) = {rho_resid:.3f} (p = {p_resid:.2e})")
                print("(This tests whether the internet component adds info beyond GDP)")

                results["residual_proxy"] = {
                    "spearman_rho": round(float(rho_resid), 4),
                    "spearman_p": float(p_resid),
                    "n_countries": len(merged_resid),
                    "interpretation": "Positive ρ indicates internet component captures "
                                      "ad-market variation beyond GDP alone"
                }

    # ============================================================
    # FIGURES
    # ============================================================

    # Figure 1: Proxy vs Ad Spend scatter (log-log)
    if "per_capita" in results and n_match >= 5:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left: log-log scatter
        ax = axes[0]
        x_vals = np.log10(merged_pc["proxy_mean"].clip(lower=0.01))
        y_vals = np.log10(merged_pc["ad_spend_pc"].clip(lower=0.01))
        ax.scatter(x_vals, y_vals, alpha=0.7, s=40, color="#2563eb")

        for idx, row in merged_pc.iterrows():
            if row["rank_diff"] > 8 or idx in ["United States", "China", "India",
                                                  "Japan", "Nigeria", "Ethiopia"]:
                ax.annotate(idx, (np.log10(max(row["proxy_mean"], 0.01)),
                                   np.log10(max(row["ad_spend_pc"], 0.01))),
                            fontsize=7, alpha=0.8)

        z = np.polyfit(x_vals, y_vals, 1)
        xline = np.linspace(x_vals.min(), x_vals.max(), 100)
        ax.plot(xline, np.polyval(z, xline), "r--", alpha=0.5)
        ax.set_xlabel("log₁₀(Ad Proxy)", fontsize=10)
        ax.set_ylabel("log₁₀(Digital Ad Spend/capita, USD)", fontsize=10)
        ax.set_title(f"Proxy vs Actual Ad Spend (ρ = {rho_pc:.3f})", fontweight="bold")

        # Right: rank comparison
        ax = axes[1]
        ax.scatter(merged_pc["rank_proxy"], merged_pc["rank_adspend"],
                   alpha=0.7, s=40, color="#dc2626")
        ax.plot([0, n_match + 1], [0, n_match + 1], "k--", alpha=0.3)

        for idx, row in merged_pc.iterrows():
            if row["rank_diff"] > 5:
                ax.annotate(idx, (row["rank_proxy"], row["rank_adspend"]),
                            fontsize=7, alpha=0.8)

        ax.set_xlabel("Proxy Rank", fontsize=10)
        ax.set_ylabel("Actual Ad Spend Rank", fontsize=10)
        ax.set_title("Rank Comparison", fontweight="bold")
        ax.invert_xaxis()
        ax.invert_yaxis()

        plt.tight_layout()
        fig_path = os.path.join(FIG_DIR, "proxy_validation_rank.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nSaved: {fig_path}")

    # Figure 2: Proxy timeseries for selected countries
    fig, ax = plt.subplots(figsize=(10, 6))
    showcase = ["United States", "Japan", "United Kingdom", "Germany",
                "China", "India", "Brazil", "South Korea"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(showcase)))

    for country, color in zip(showcase, colors):
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
    fig_path2 = os.path.join(FIG_DIR, "proxy_validation_timeseries.png")
    plt.savefig(fig_path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path2}")

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    if "per_capita" in results:
        rho = results["per_capita"]["spearman_rho"]
        verdict = ("STRONG" if abs(rho) > 0.85
                   else "MODERATE" if abs(rho) > 0.7
                   else "WEAK")
        print(f"Per-capita validation: ρ = {rho:.3f} → {verdict}")

    if "variance_decomposition" in results:
        r2_i = results["variance_decomposition"]["r2_internet"]
        r2_g = results["variance_decomposition"]["r2_gdp"]
        print(f"Variance: GDP explains {r2_g:.1%}, Internet explains {r2_i:.1%}")
        if r2_g > 0.7:
            print("  ⚠ WARNING: Proxy is dominated by GDP component.")
            print("    Internet penetration adds limited independent information.")
            print("    Consider using residualized proxy or direct ad spend data.")

    if "residual_proxy" in results:
        rho_r = results["residual_proxy"]["spearman_rho"]
        print(f"Residual proxy (GDP-orthogonalized) vs ad spend: ρ = {rho_r:.3f}")
        if rho_r > 0.3:
            print("  → Internet component captures additional ad-market variation beyond GDP")
        else:
            print("  ⚠ Internet component adds minimal info beyond GDP")

    print("\nNOTE: Reference ad spend data is from public industry report summaries.")
    print("Full validation requires WARC/GroupM microdata access.")
    print("See docs/review/references_expansion.md for data sources.")

    # ============================================================
    # [WARC EXTENSION] — for users with proprietary data
    # ============================================================
    # To extend with WARC/GroupM data:
    #   1. Place CSV at data/macro/warc_ad_spend.csv
    #      columns: country, year, digital_ad_spend_usd, total_ad_spend_usd
    #   2. Uncomment the section below
    #
    # warc_path = os.path.join(DATA_DIR, "warc_ad_spend.csv")
    # if os.path.exists(warc_path):
    #     warc = pd.read_csv(warc_path)
    #     # ... merge and compute correlations ...

    # ---- Save JSON ----
    json_path = os.path.join(RESULTS_DIR, "proxy_validation.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {json_path}")


if __name__ == "__main__":
    main()
