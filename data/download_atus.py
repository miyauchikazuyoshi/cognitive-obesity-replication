#!/usr/bin/env python3
"""
ATUS Wellbeing Module データダウンロードスクリプト
対応セクション: Section 2.3.3, Appendix A.2

Bureau of Labor Statistics の ATUS マイクロデータ。
本リポジトリの分析では、以下を組み合わせる:
  - 基本ATUS (multi-year): Activity + Respondent（2003-2024 を使用）
  - Well-Being module (multi-year): 2010/2012/2013 の WB Respondent

NOTE:
  BLS は bot 対策で自動ダウンロードを拒否する場合があります (HTTP 403 / Access Denied)。
  その場合はブラウザで ZIP を手動ダウンロードし、`data/atus/` に配置した上で
  このスクリプトを実行すると、ZIP の展開だけを行います。

ダウンロードファイル:
  - atusact-0324.zip    Activity file (2003-2024)
  - atusresp-0324.zip   Respondent file (2003-2024; includes TUFNWGTP weight)
  - wbresp_1013.zip     WB Respondent file (2010/2012/2013; includes Cantril ladder)

出力先: data/atus/
"""

import os
import urllib.request
import zipfile

BASE_URL = "https://www.bls.gov/tus/datafiles"
FILES = [
    ("atusact-0324.zip", "Activity file (2003-2024)"),
    ("atusresp-0324.zip", "Respondent file (2003-2024)"),
    ("wbresp_1013.zip", "Well-Being module respondent file (2010/2012/2013)"),
]

OUT_DIR = os.path.join(os.path.dirname(__file__), "atus")


def download_and_extract():
    os.makedirs(OUT_DIR, exist_ok=True)
    for fname, desc in FILES:
        url = f"{BASE_URL}/{fname}"
        dest = os.path.join(OUT_DIR, fname)
        if not os.path.exists(dest):
            print(f"  Downloading {desc} ({fname}) ...")
            try:
                urllib.request.urlretrieve(url, dest)
                size_mb = os.path.getsize(dest) / 1e6
                print(f"  [done] {fname} ({size_mb:.1f} MB)")
            except Exception as e:
                print(f"  [FAIL] {fname}: {e}")
                print(f"  Manual download: {url}")
                continue
        else:
            print(f"  [found] {fname} (manual download?)")

        # Extract
        if fname.endswith(".zip"):
            try:
                with zipfile.ZipFile(dest, 'r') as z:
                    z.extractall(OUT_DIR)
                    print(f"  [extracted] {', '.join(z.namelist())}")
            except Exception as e:
                print(f"  [extract failed] {e}")

    print(f"\nAll ATUS files saved to {OUT_DIR}/")
    print("\nNOTE: BLS may change URL structure. If downloads fail,")
    print("use the official pages for manual download:")
    print("  - ATUS multi-year (2003-2024): https://www.bls.gov/tus/data/datafiles-0324.htm")
    print("  - WB module (2010/2012/2013 multi-year): https://www.bls.gov/tus/data/wbdatafiles_1013.htm")
    print("If direct ZIP links fail, BLS recommends right-click → 'Save link as...' on those pages.")


if __name__ == "__main__":
    print("=== ATUS Wellbeing Module Download ===")
    download_and_extract()
