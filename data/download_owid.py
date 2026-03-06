#!/usr/bin/env python3
"""
Our World in Data (OWID) grapher downloader.

Downloads:
  - Mean years of schooling (UNDP/HDR-derived series on OWID)
  - Homicide rate (UNODC) series on OWID

Outputs (data/macro/):
  - owid_mean_years_of_schooling.csv
  - owid_homicide_rate_unodc.csv
"""

from __future__ import annotations

import os
import tempfile
import urllib.request

OUT_DIR = os.path.join(os.path.dirname(__file__), "macro")
BASE_URL = "https://ourworldindata.org/grapher"

UA_HEADERS = {"User-Agent": "Mozilla/5.0 (replication; contact in repo README)"}

FILES = {
    "mean-years-of-schooling.csv": "owid_mean_years_of_schooling.csv",
    "homicide-rate-unodc.csv": "owid_homicide_rate_unodc.csv",
}


def download(url: str, dest: str) -> None:
    req = urllib.request.Request(url, headers=UA_HEADERS)
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=os.path.dirname(dest)) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    os.replace(tmp_path, dest)


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== OWID grapher download ===\n")
    for remote_name, local_name in FILES.items():
        url = f"{BASE_URL}/{remote_name}"
        out_path = os.path.join(OUT_DIR, local_name)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            print(f"[skip] {local_name} already exists")
            continue

        print(f"Downloading {remote_name} ...")
        try:
            download(url, out_path)
        except Exception as e:
            print(f"  [FAIL] {remote_name}: {e}")
            continue
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"  Saved: {out_path} ({size_mb:.1f} MB)")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
