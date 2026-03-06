#!/usr/bin/env python3
"""
WHO Global Health Observatory (GHO) API downloader.

Downloads (country-year panel; both sexes):
  - NCD_PAC: Prevalence of insufficient physical activity among adults aged 18+ (%), crude estimate
  - MH_12:   Age-standardized suicide rates (per 100 000 population)

Outputs (data/macro/):
  - who_ncd_pac.csv
  - who_suicide_mh12.csv
"""

from __future__ import annotations

import os
import time
from typing import Any
from urllib.parse import quote, urlencode

import pandas as pd
import requests

OUT_DIR = os.path.join(os.path.dirname(__file__), "macro")
BASE_URL = "https://ghoapi.azureedge.net/api"

UA_HEADERS = {"User-Agent": "Mozilla/5.0 (replication; contact in repo README)"}


def fetch_odata(endpoint: str, params: dict[str, str]) -> list[dict[str, Any]]:
    base = f"{BASE_URL}/{endpoint}"
    rows: list[dict[str, Any]] = []
    next_url: str | None = base
    next_params: dict[str, str] | None = params

    while next_url:
        if next_params is not None:
            query = urlencode(next_params, quote_via=quote, safe="(),$=:")
            req_url = f"{next_url}?{query}"
        else:
            req_url = next_url

        resp = requests.get(req_url, headers=UA_HEADERS, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        rows.extend(payload.get("value", []))
        next_url = payload.get("@odata.nextLink")
        next_params = None
        time.sleep(0.05)  # be polite

    return rows


def save_panel(rows: list[dict[str, Any]], out_path: str, value_name: str) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No rows returned for {os.path.basename(out_path)} (check indicator/filter).")

    keep = [c for c in ["SpatialDim", "TimeDim", "NumericValue", "Low", "High"] if c in df.columns]
    df = df[keep].copy()
    df = df.rename(columns={"SpatialDim": "code", "TimeDim": "year", "NumericValue": value_name})
    df["code"] = df["code"].astype(str)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")
    for c in ["Low", "High"]:
        if c in df.columns:
            df[c.lower()] = pd.to_numeric(df[c], errors="coerce")
            df = df.drop(columns=[c])

    df = df[df["code"].str.len() == 3].copy()
    df = df.groupby(["code", "year"], as_index=False)[[value_name] + [c for c in ["low", "high"] if c in df.columns]].mean()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=== WHO GHO API download ===\n")

    # ---- Physical inactivity (NCD_PAC) ----
    pac_path = os.path.join(OUT_DIR, "who_ncd_pac.csv")
    if os.path.exists(pac_path) and os.path.getsize(pac_path) > 0:
        print(f"[skip] {os.path.basename(pac_path)} already exists")
    else:
        print("Downloading WHO physical inactivity (NCD_PAC)...")
        pac_rows = fetch_odata(
            "NCD_PAC",
            {
                "$select": "SpatialDim,TimeDim,NumericValue,Low,High,Dim1,Dim2",
                "$filter": (
                    "SpatialDimType eq 'COUNTRY' and "
                    "Dim1 eq 'SEX_BTSX' and "
                    "Dim2 eq 'AGEGROUP_YEARS18-PLUS' and "
                    "TimeDim ge 1990 and TimeDim le 2023"
                ),
                "$top": "1000",
            },
        )
        save_panel(pac_rows, pac_path, value_name="physical_inactivity")
        print(f"  Saved: {pac_path}")

    # ---- Suicide rate (MH_12) ----
    sui_path = os.path.join(OUT_DIR, "who_suicide_mh12.csv")
    if os.path.exists(sui_path) and os.path.getsize(sui_path) > 0:
        print(f"[skip] {os.path.basename(sui_path)} already exists")
    else:
        print("Downloading WHO suicide rates (MH_12; age-standardized)...")
        sui_rows = fetch_odata(
            "MH_12",
            {
                "$select": "SpatialDim,TimeDim,NumericValue,Low,High,Dim1",
                "$filter": (
                    "SpatialDimType eq 'COUNTRY' and "
                    "Dim1 eq 'SEX_BTSX' and "
                    "TimeDim ge 1990 and TimeDim le 2023"
                ),
                "$top": "1000",
            },
        )
        save_panel(sui_rows, sui_path, value_name="suicide")
        print(f"  Saved: {sui_path}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
