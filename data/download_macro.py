#!/usr/bin/env python3
"""
マクロパネルデータ取得スクリプト
対応セクション: Section 2.1-2.2 (Block A/B)

World Bank WDI の一部は API 経由で自動取得可能。
IHME GBD（鬱病）は手動ダウンロードが必要（再配布制約）。

補助データ（自動）:
  - WHO GHO API: 身体不活発率 (NCD_PAC), 自殺率 (MH_12) → download_who_gho.py
  - OWID grapher: 平均就学年数, UNODC殺人率 → download_owid.py

出力先: data/macro/
"""

import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "macro")

# World Bank indicators used
WB_INDICATORS = {
    "IT.NET.USER.ZS": "Internet users (% of population)",
    "NY.GDP.PCAP.CD": "GDP per capita (current US$)",
    "NY.GDP.PCAP.PP.KD": "GDP per capita, PPP (constant 2017 international $)",
    "SP.POP.TOTL": "Population, total",
    "SL.SRV.EMPL.ZS": "Employment in services (% of total)",
}

# IHME data (manual download required)
IHME_SOURCES = {
    "depression": "IHME GBD Results: cause=Depressive disorders, metric=Rate, measure=Prevalence",
    "suicide": "IHME GBD Results: cause=Self-harm, metric=Rate, measure=Deaths",
    "homicide": "IHME GBD Results: cause=Interpersonal violence, metric=Rate, measure=Deaths",
}

# WHO data
WHO_SOURCES = {
    "NCD_PAC": "WHO NCD Country Profiles: Insufficient physical activity",
}


def download_worldbank():
    """World Bank WDI via wbgapi (if available) or manual instructions."""
    os.makedirs(OUT_DIR, exist_ok=True)

    out_path = os.path.join(OUT_DIR, "worldbank_wdi.csv")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"  [skip] {out_path} already exists")
        return True

    try:
        import wbgapi as wb
        print("  wbgapi found. Downloading World Bank indicators...")
        import pandas as pd

        frames = []
        for code, desc in WB_INDICATORS.items():
            print(f"    {code}: {desc}")
            try:
                df = wb.data.DataFrame(code, time=range(1990, 2024), labels=True)
                df = df.reset_index()
                df['indicator'] = code
                frames.append(df)
            except Exception as e:
                print(f"    [FAIL] {code}: {e}")

        if frames:
            combined = pd.concat(frames, ignore_index=True)
            combined.to_csv(out_path, index=False)
            print(f"\n  Saved to {out_path}")
        return True

    except ImportError:
        print("  wbgapi not installed. Install with: pip install wbgapi")
        print("  Or download manually from: https://databank.worldbank.org/")
        print(f"\n  Required indicators:")
        for code, desc in WB_INDICATORS.items():
            print(f"    {code}: {desc}")
        return False


def print_manual_instructions():
    """Print instructions for data that requires manual download."""
    print("\n=== Manual Download Required ===\n")

    print("1. IHME Global Burden of Disease (GBD) Results")
    print("   URL: https://vizhub.healthdata.org/gbd-results/")
    print("   Settings:")
    print("     - Location: All countries")
    print("     - Year: 1990-2023")
    print("     - Age: Age-standardized (recommended)")
    print("     - Sex: Both")
    print("   Download separately:")
    print("     - depression (required): Depressive disorders, Prevalence, Rate")
    print("     - suicide (optional):    Self-harm, Deaths, Rate")
    print("     - homicide (optional):   Interpersonal violence, Deaths, Rate")
    print("   Save (rename) to:")
    print(f"     - {OUT_DIR}/ihme_depression.csv")
    print(f"     - {OUT_DIR}/ihme_suicide.csv   (optional)")
    print(f"     - {OUT_DIR}/ihme_homicide.csv  (optional)")

    print("\n2. If you want to minimize manual steps:")
    print("   - WHO GHO API (auto): python data/download_who_gho.py")
    print("   - OWID grapher (auto): python data/download_owid.py")

    print(f"\n3. After placing IHME CSVs, build the analysis panel:")
    print(f"   python data/build_macro_panel.py")


if __name__ == "__main__":
    print("=== Macro Panel Data Acquisition ===\n")
    print("--- World Bank WDI (automated) ---")
    download_worldbank()
    print_manual_instructions()
