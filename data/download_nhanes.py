#!/usr/bin/env python3
"""
NHANES 2017-2018 データダウンロードスクリプト
対応セクション: Section 2.3.2, Appendix A.1

ダウンロードファイル:
  - DEMO_J.XPT  人口統計
  - DPQ_J.XPT   PHQ-9 抑うつスクリーニング
  - PAQ_J.XPT   身体活動
  - BMX_J.XPT   身体計測（BMI）
  - HIQ_J.XPT   健康保険

出力先: data/nhanes/
"""

import os
import tempfile
import urllib.request

# NOTE:
# Older NHANES direct links under /Nchs/Nhanes/2017-2018/ may return an HTML
# "Page Not Found" payload with HTTP 200. Use the canonical Public DataFiles
# endpoint instead.
BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles"
FILES = ["DEMO_J", "DPQ_J", "PAQ_J", "BMX_J", "HIQ_J"]

OUT_DIR = os.path.join(os.path.dirname(__file__), "nhanes")

XPT_MAGIC = b"HEADER RECORD***"


def _looks_like_xpt(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(len(XPT_MAGIC))
        return head == XPT_MAGIC
    except Exception:
        return False


def _download(url: str, dest: str) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=os.path.dirname(dest)) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    os.replace(tmp_path, dest)


def download():
    os.makedirs(OUT_DIR, exist_ok=True)
    for name in FILES:
        url = f"{BASE_URL}/{name}.xpt"
        dest_name = f"{name}.XPT"
        dest = os.path.join(OUT_DIR, dest_name)
        if os.path.exists(dest) and _looks_like_xpt(dest):
            print(f"  [skip] {dest_name} already exists")
            continue
        if os.path.exists(dest) and not _looks_like_xpt(dest):
            print(f"  [re-download] {dest_name} exists but is not an XPORT file")
        print(f"  Downloading {dest_name} ...")
        _download(url, dest)
        if not _looks_like_xpt(dest):
            raise RuntimeError(
                f"Downloaded file does not look like a SAS XPORT file: {dest}. "
                f"Please download manually from NHANES (see script docstring)."
            )
        size_mb = os.path.getsize(dest) / 1e6
        print(f"  [done] {dest_name} ({size_mb:.1f} MB)")
    print(f"\nAll NHANES files saved to {OUT_DIR}/")


if __name__ == "__main__":
    print("=== NHANES 2017-2018 Download ===")
    download()
