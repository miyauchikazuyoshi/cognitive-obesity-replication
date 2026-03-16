#!/usr/bin/env python3
"""
IHME GBD 拡張データダウンロード・処理スクリプト
=================================================
Review 14 対応: DALYs三角測量 + プラセボ検定用アウトカム

IHME GBD Results Tool (https://vizhub.healthdata.org/gbd-results/) から
手動ダウンロードしたCSVを処理し、分析用CSVに変換する。

必要なダウンロード (すべて Age-standardized, Both sexes, Rate, 1990-2023):
  1. depression_dalys:  Cause=Depressive disorders, Measure=DALYs
  2. cardiovascular:    Cause=Cardiovascular diseases, Measure=Prevalence
  3. diabetes:          Cause=Diabetes mellitus type 2, Measure=Prevalence

ダウンロード手順:
  1. https://vizhub.healthdata.org/gbd-results/ にアクセス
  2. 以下の設定で各アウトカムをダウンロード:
     - GBD Estimate: Cause of death or injury
     - Location: Select all countries (not regions/aggregates)
     - Year: 1990-2023
     - Context: Cause
     - Age: Age-standardized
     - Sex: Both
     - Metric: Rate (per 100,000)
  3. 各ファイルをダウンロードし、以下の名前でdata/macro/に配置:
     - ihme_depression_dalys.csv  (Measure=DALYs, Cause=Depressive disorders)
     - ihme_cardiovascular.csv    (Measure=Prevalence, Cause=Cardiovascular diseases)
     - ihme_diabetes.csv          (Measure=Prevalence, Cause=Diabetes mellitus type 2)

出力 (data/macro/):
  - ihme_depression_dalys_clean.csv
  - ihme_cardiovascular_clean.csv
  - ihme_diabetes_clean.csv
"""

import os
import sys

import pandas as pd

# Reuse build_macro_panel's load_ihme function
sys.path.insert(0, os.path.dirname(__file__))
from build_macro_panel import load_ihme, canonical_country

MACRO_DIR = os.path.join(os.path.dirname(__file__), "macro")

DOWNLOADS = [
    {
        "input": "ihme_depression_dalys.csv",
        "output": "ihme_depression_dalys_clean.csv",
        "value_name": "depression_dalys",
        "expected_cause": "Depressive disorders",
        "expected_measure": "DALYs",
        "expected_metric": "Rate",
    },
    {
        "input": "ihme_cardiovascular.csv",
        "output": "ihme_cardiovascular_clean.csv",
        "value_name": "cardiovascular_prevalence",
        "expected_cause": "Cardiovascular diseases",
        "expected_measure": "Prevalence",
        "expected_metric": "Rate",
    },
    {
        "input": "ihme_diabetes.csv",
        "output": "ihme_diabetes_clean.csv",
        "value_name": "diabetes_prevalence",
        "expected_cause": "Diabetes",
        "expected_measure": "Prevalence",
        "expected_metric": "Rate",
    },
]


def process_ihme_file(spec: dict) -> bool:
    """Process a single IHME download file."""
    input_path = os.path.join(MACRO_DIR, spec["input"])
    output_path = os.path.join(MACRO_DIR, spec["output"])

    if not os.path.exists(input_path):
        print(f"  [SKIP] {spec['input']} not found. Download from IHME GBD Results Tool.")
        return False

    try:
        df = load_ihme(
            input_path,
            spec["value_name"],
            expected_cause=spec["expected_cause"],
            expected_measure=spec["expected_measure"],
            expected_metric=spec["expected_metric"],
        )
        df.to_csv(output_path, index=False)
        n_countries = df["country_key"].nunique()
        n_years = df["year"].nunique()
        print(f"  [OK] {spec['output']}: {len(df):,} rows, {n_countries} countries, {n_years} years")
        return True
    except Exception as e:
        print(f"  [ERROR] {spec['input']}: {e}")
        return False


def print_download_instructions():
    """Print step-by-step download instructions."""
    print("\n" + "=" * 70)
    print("IHME GBD Extended Data: Download Instructions")
    print("=" * 70)
    print()
    print("URL: https://vizhub.healthdata.org/gbd-results/")
    print()
    print("Common settings for ALL downloads:")
    print("  - Location: All countries (deselect regions/aggregates)")
    print("  - Year: 1990-2023")
    print("  - Age: Age-standardized")
    print("  - Sex: Both")
    print("  - Metric: Rate (per 100,000)")
    print()

    for i, spec in enumerate(DOWNLOADS, 1):
        print(f"  Download {i}: {spec['value_name']}")
        print(f"    Cause:   {spec['expected_cause']}")
        print(f"    Measure: {spec['expected_measure']}")
        print(f"    Save as: data/macro/{spec['input']}")
        print()

    print("After downloading, run this script again to process:")
    print("  python data/download_ihme_extended.py")
    print()
    print("Then rebuild the macro panel:")
    print("  python data/build_macro_panel.py")
    print("=" * 70)


if __name__ == "__main__":
    print("=== IHME GBD Extended Data Processing ===\n")

    success_count = 0
    for spec in DOWNLOADS:
        ok = process_ihme_file(spec)
        if ok:
            success_count += 1

    if success_count == 0:
        print("\nNo IHME extended files found.")
        print_download_instructions()
    elif success_count < len(DOWNLOADS):
        missing = [s["input"] for s in DOWNLOADS
                    if not os.path.exists(os.path.join(MACRO_DIR, s["input"]))]
        print(f"\nProcessed {success_count}/{len(DOWNLOADS)} files.")
        print(f"Missing: {', '.join(missing)}")
        print_download_instructions()
    else:
        print(f"\nAll {success_count} files processed successfully.")
        print("Next: python data/build_macro_panel.py")
