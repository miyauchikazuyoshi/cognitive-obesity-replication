#!/usr/bin/env python3
"""
記述統計: マクロパネルの概要 (Section 2.1)
==========================================
パネルデータの基本構造（国数・年数・観測数・欠損率）と
主要変数の要約統計量を算出する。

対応: paper_figure_table_map.md Block A — 記述統計表

Inputs:
  - data/macro/panel_merged.csv (or compatible macro panel)

Outputs:
  - results/block_a_descriptive.json
  - console output
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "macro")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# DATA LOADING (shared pattern with Block B)
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
    edu_col = find(["education", "mean_years_schooling", "schooling"])
    services_col = find(["employment_services", "services_employment",
                         "sl.srv.empl.zs"])

    if country_col is None or year_col is None:
        print("ERROR: cannot identify country/year columns in panel.")
        print(f"Columns: {list(df.columns)[:40]} ...")
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
    if edu_col:
        rename[edu_col] = "education"
    if services_col:
        rename[services_col] = "employment_services"

    df = df.rename(columns=rename)
    return df


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("BLOCK A — 記述統計: マクロパネルの概要 (Section 2.1)")
    print("=" * 70)

    df = normalize_columns(load_panel())

    # -- Ad proxy construction --
    internet_pct = pd.to_numeric(df.get("internet"), errors="coerce")
    gdp = pd.to_numeric(df.get("gdp"), errors="coerce")
    if internet_pct is not None and gdp is not None:
        if internet_pct.max(skipna=True) <= 1.5:
            internet_pct = internet_pct * 100.0
        df["ad_proxy"] = internet_pct * gdp / 1000.0

    # -- Panel structure --
    n_countries = df["country"].nunique()
    years = df["year"].dropna().astype(int)
    year_range = f"{years.min()}-{years.max()}" if len(years) > 0 else "N/A"
    n_years = df["year"].nunique()

    print(f"\nPanel structure:")
    print(f"  Countries: {n_countries}")
    print(f"  Years:     {n_years} ({year_range})")
    print(f"  Total obs: {len(df):,}")

    # -- Variable availability & missing rates --
    key_vars = ["internet", "gdp", "depression_prevalence", "suicide",
                "ad_proxy", "education", "employment_services"]
    available_vars = [v for v in key_vars if v in df.columns]

    print(f"\n{'Variable':<25s} {'N':>7s} {'Missing%':>9s} {'Mean':>12s} {'SD':>12s} {'Min':>12s} {'Max':>12s}")
    print("-" * 90)

    stats_out = {}
    for var in available_vars:
        col = pd.to_numeric(df[var], errors="coerce")
        n_valid = col.notna().sum()
        n_miss = col.isna().sum()
        pct_miss = n_miss / len(df) * 100
        desc = col.describe()
        print(f"  {var:<23s} {n_valid:>7,d} {pct_miss:>8.1f}% {desc['mean']:>12.2f} "
              f"{desc['std']:>12.2f} {desc['min']:>12.2f} {desc['max']:>12.2f}")
        stats_out[var] = {
            "n": int(n_valid),
            "missing_pct": round(float(pct_miss), 2),
            "mean": round(float(desc["mean"]), 4),
            "std": round(float(desc["std"]), 4),
            "min": round(float(desc["min"]), 4),
            "max": round(float(desc["max"]), 4),
            "p25": round(float(desc["25%"]), 4),
            "median": round(float(desc["50%"]), 4),
            "p75": round(float(desc["75%"]), 4),
        }

    # -- Temporal coverage --
    print(f"\n--- Temporal coverage (obs per year) ---")
    if "depression_prevalence" in df.columns:
        yearly = df.dropna(subset=["depression_prevalence"]).groupby("year").size()
        print(f"  Depression data: {yearly.min()}-{yearly.max()} countries/year "
              f"(median={yearly.median():.0f})")
    if "internet" in df.columns:
        yearly_i = df.dropna(subset=["internet"]).groupby("year").size()
        print(f"  Internet data:   {yearly_i.min()}-{yearly_i.max()} countries/year "
              f"(median={yearly_i.median():.0f})")

    # -- Complete-case sample for core analysis --
    core_vars = ["depression_prevalence", "ad_proxy"]
    core_available = [v for v in core_vars if v in df.columns]
    if len(core_available) == len(core_vars):
        core = df.dropna(subset=core_vars)
        print(f"\n--- Core analysis sample (complete cases: depression + ad_proxy) ---")
        print(f"  N = {len(core):,}, Countries = {core['country'].nunique()}, "
              f"Years = {core['year'].nunique()}")

    # -- Save JSON --
    output = {
        "panel": {
            "n_countries": n_countries,
            "n_years": n_years,
            "year_range": year_range,
            "total_obs": len(df),
        },
        "variables": stats_out,
    }
    out_path = os.path.join(RESULTS_DIR, "block_a_descriptive.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
