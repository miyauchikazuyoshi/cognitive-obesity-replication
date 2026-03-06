#!/usr/bin/env python3
"""
Ad Proxy External Validation Dataset
Stanford Reviewer Q3/Q8: "Expand validation beyond 10 countries"

Compiles publicly available digital ad spending data by country
to validate the proxy (Internet% × GDP/capita) against actual ad expenditure.

Sources (free / partially free):
  1. Oberlo/eMarketer (2023): Top 10 countries, absolute ad spend
  2. Statista free tier: Selected country data points
  3. IAB reports: Regional aggregates
  4. GroupM "This Year Next Year" reports: Global estimates (PDF)

Limitation: Most granular, multi-year, multi-country ad spend data
requires paid subscriptions (WARC, Statista Premium, eMarketer Pro).
This script provides the framework; users with paid access can extend.

Output: data/macro/ad_spend_validation.csv
"""

import os
import pandas as pd
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "macro")

# ============================================================
# Available free data points (compiled from public sources)
# ============================================================

# 2023 digital ad spend by country (Oberlo/eMarketer, billion USD)
AD_SPEND_2023 = {
    "United States": 263.89,
    "China": 136.10,
    "United Kingdom": 32.66,
    "Japan": 21.20,
    "Germany": 11.74,
    "Canada": 10.77,
    "Australia": 10.52,
    "France": 7.97,
    "Brazil": 7.73,
    "South Korea": 5.81,
}

# 2023 population (World Bank, millions) - for per-capita calculation
POPULATION_2023 = {
    "United States": 334.9,
    "China": 1425.9,
    "United Kingdom": 67.7,
    "Japan": 124.5,
    "Germany": 84.5,
    "Canada": 40.1,
    "Australia": 26.4,
    "France": 68.2,
    "Brazil": 216.4,
    "South Korea": 51.7,
}

# 2023 internet penetration (%) - ITU/World Bank estimates
INTERNET_2023 = {
    "United States": 92.0,
    "China": 73.0,
    "United Kingdom": 97.0,
    "Japan": 93.0,
    "Germany": 93.0,
    "Canada": 95.0,
    "Australia": 96.0,
    "France": 92.0,
    "Brazil": 81.0,
    "South Korea": 98.0,
}

# 2023 GDP per capita (World Bank, current USD)
GDP_PC_2023 = {
    "United States": 80035,
    "China": 12720,
    "United Kingdom": 48913,
    "Japan": 33950,
    "Germany": 52824,
    "Canada": 53247,
    "Australia": 63487,
    "France": 44408,
    "Brazil": 8917,
    "South Korea": 32423,
}

# Growth rates 2023 (Oberlo, %)
GROWTH_2023 = {
    "United States": 12.0,
    "Peru": 20.0,
    "Argentina": 19.2,
    "Chile": 17.0,
    "India": 16.5,
    "Colombia": 16.0,
    "Mexico": 14.5,
    "Russia": 13.3,
    "Indonesia": 12.1,
    "Turkey": 12.0,
}


def build_validation_dataset():
    """Construct validation dataset from available data."""
    os.makedirs(OUT_DIR, exist_ok=True)

    rows = []
    for country in AD_SPEND_2023:
        ad_spend = AD_SPEND_2023[country]
        pop = POPULATION_2023.get(country)
        inet = INTERNET_2023.get(country)
        gdp = GDP_PC_2023.get(country)

        if pop and inet and gdp:
            ad_per_capita = ad_spend * 1e9 / (pop * 1e6)  # USD per person
            proxy = inet * gdp / 1000  # Our proxy formula

            rows.append({
                "country": country,
                "year": 2023,
                "digital_ad_spend_bn_usd": ad_spend,
                "population_mn": pop,
                "ad_spend_per_capita_usd": round(ad_per_capita, 2),
                "internet_pct": inet,
                "gdp_per_capita_usd": gdp,
                "proxy_inet_x_gdp_div1000": round(proxy, 1),
                "growth_pct": GROWTH_2023.get(country),
                "source": "Oberlo/eMarketer 2023",
            })

    df = pd.DataFrame(rows)

    # Correlation analysis
    from scipy.stats import pearsonr, spearmanr

    r_pearson, p_pearson = pearsonr(df["ad_spend_per_capita_usd"], df["proxy_inet_x_gdp_div1000"])
    r_spearman, p_spearman = spearmanr(df["ad_spend_per_capita_usd"], df["proxy_inet_x_gdp_div1000"])

    print("=" * 60)
    print("Ad Proxy External Validation (N = %d countries, 2023)" % len(df))
    print("=" * 60)
    print(f"\n  Proxy = Internet% × GDP/capita / 1000")
    print(f"  Validation target: Digital ad spend per capita (USD)")
    print(f"\n  Pearson r = {r_pearson:.4f} (p = {p_pearson:.2e})")
    print(f"  Spearman ρ = {r_spearman:.4f} (p = {p_spearman:.2e})")

    print(f"\n  Country-level comparison:")
    print(f"  {'Country':<20s} {'Ad$/cap':>8s} {'Proxy':>10s}")
    print(f"  {'-'*40}")
    for _, row in df.sort_values("ad_spend_per_capita_usd", ascending=False).iterrows():
        print(f"  {row['country']:<20s} ${row['ad_spend_per_capita_usd']:>7.1f} {row['proxy_inet_x_gdp_div1000']:>10.1f}")

    # Save
    out_path = os.path.join(OUT_DIR, "ad_spend_validation.csv")
    df.to_csv(out_path, index=False)
    print(f"\n  Saved to: {out_path}")

    # Limitations
    print(f"\n  LIMITATIONS:")
    print(f"  - Only 10 countries (top spenders) for 1 year (2023)")
    print(f"  - No time-series variation")
    print(f"  - Ad spend = total digital, not per-user ad load")
    print(f"  - Paid WARC/eMarketer data would provide:")
    print(f"    * 50+ countries × 10+ years")
    print(f"    * Per-user ad impressions")
    print(f"    * Format breakdown (display, search, social, video)")
    print(f"    * Platform-specific metrics")

    return df


# ============================================================
# Template for extending with paid data
# ============================================================

PAID_DATA_TEMPLATE = """
# If you have access to WARC/eMarketer/GroupM data, add rows here:
# Format: country, year, digital_ad_spend_bn_usd, ad_spend_per_user_usd, 
#         ad_impressions_per_user, notification_density, source

# Example:
# Germany, 2018, 8.2, 142.5, NA, NA, WARC
# Germany, 2019, 9.1, 156.3, NA, NA, WARC
# Germany, 2020, 9.8, 168.2, NA, NA, WARC
"""


if __name__ == "__main__":
    build_validation_dataset()

    # Save template
    template_path = os.path.join(OUT_DIR, "ad_spend_extended_TEMPLATE.csv")
    with open(template_path, 'w') as f:
        f.write("country,year,digital_ad_spend_bn_usd,ad_spend_per_user_usd,"
                "ad_impressions_per_user,notification_density,source\n")
        f.write("# Add rows from WARC/eMarketer/GroupM here\n")
    print(f"\n  Template saved to: {template_path}")
    print(f"  Fill with paid data sources to extend validation beyond N=10.")
