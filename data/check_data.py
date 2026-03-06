#!/usr/bin/env python3
"""
Data presence checker + next-step instructions.

Run from repo root:
  python data/check_data.py

This script does NOT download anything by itself; it only reports what's missing
and which command (or manual download) to do next.
"""

from __future__ import annotations

import os
import textwrap


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _path(*parts: str) -> str:
    return os.path.join(REPO_ROOT, *parts)


def _exists(path: str) -> bool:
    return os.path.exists(path)


def _looks_like_xpt(path: str) -> bool:
    magic = b"HEADER RECORD***"
    try:
        with open(path, "rb") as f:
            head = f.read(len(magic))
        return head == magic
    except Exception:
        return False


def _any_exists(dir_path: str, candidates: list[str]) -> str | None:
    for fname in candidates:
        p = os.path.join(dir_path, fname)
        if os.path.exists(p):
            return p
    return None


def check_nhanes() -> list[str]:
    missing: list[str] = []
    nhanes_dir = _path("data", "nhanes")
    required = ["DEMO_J.XPT", "DPQ_J.XPT", "PAQ_J.XPT", "BMX_J.XPT", "HIQ_J.XPT"]

    print("\n=== NHANES (Block C) ===")
    if not _exists(nhanes_dir):
        missing.extend([os.path.join("data", "nhanes", f) for f in required])
        print("MISSING: data/nhanes/ (directory not found)")
        print("Next:    python data/download_nhanes.py")
        return missing

    bad = []
    for f in required:
        p = os.path.join(nhanes_dir, f)
        if not _exists(p):
            missing.append(os.path.join("data", "nhanes", f))
        elif not _looks_like_xpt(p):
            bad.append(os.path.join("data", "nhanes", f))

    if not missing and not bad:
        print(f"OK: {len(required)}/{len(required)} files present")
        return []

    if missing:
        print("MISSING:")
        for m in missing:
            print(f"  - {m}")
    if bad:
        print("INVALID (not SAS XPORT):")
        for b in bad:
            print(f"  - {b}")

    print("Next: python data/download_nhanes.py")
    return missing + bad


def check_atus() -> list[str]:
    missing: list[str] = []
    atus_dir = _path("data", "atus")
    print("\n=== ATUS (Block C) ===")

    if not _exists(atus_dir):
        print("MISSING: data/atus/ (directory not found)")
        print("Next:    python data/download_atus.py")
        return ["data/atus/"]

    act = _any_exists(atus_dir, ["atusact_0324.dat", "atusact-0324.dat", "atusact_0313.dat", "atusact-0313.dat"])
    wb = _any_exists(atus_dir, ["wbresp_1013.dat", "wbresp-1013.dat", "atuswb_0313.dat", "atuswb-0313.dat"])
    resp = _any_exists(atus_dir, ["atusresp_0324.dat", "atusresp-0324.dat", "atusresp_0313.dat", "atusresp-0313.dat"])

    if act and wb and resp:
        print("OK: raw .dat files present")
        return []

    print("MISSING: required raw ATUS files for analysis/block_c/02_atus_*.py")
    if not act:
        missing.append("data/atus/atusact-0324.zip → atusact_0324.dat")
    if not wb:
        missing.append("data/atus/wbresp_1013.zip → wbresp_1013.dat")
    if not resp:
        missing.append("data/atus/atusresp-0324.zip → atusresp_0324.dat")

    for m in missing:
        print(f"  - {m}")

    zips_present = [
        f
        for f in [
            "atusact-0324.zip",
            "atusresp-0324.zip",
            "wbresp_1013.zip",
            # legacy names (older drafts)
            "atusact-0313.zip",
            "atusresp-0313.zip",
            "atuswb-0313.zip",
            "atuswgtp-0313.zip",
        ]
        if _exists(os.path.join(atus_dir, f))
    ]
    if zips_present:
        print("\nFound ZIP(s) in data/atus/ (will only extract):")
        for z in zips_present:
            print(f"  - data/atus/{z}")

    print("\nNext (try automated; may 403 due to BLS bot protection):")
    print("  python data/download_atus.py")
    print("\nIf you see HTTP 403, download these ZIPs via browser, keep filenames as-is,")
    print("place them under data/atus/, then rerun python data/download_atus.py:")
    print("  - atusact-0324.zip")
    print("  - atusresp-0324.zip")
    print("  - wbresp_1013.zip")
    print("  (Optional) wbact_1013.zip if you want WB activity-level data.")
    print("\nSee: data/README_data.md")
    return missing


def check_macro() -> list[str]:
    missing: list[str] = []
    macro_dir = _path("data", "macro")
    print("\n=== Macro panel (Block B / Block C-macro) ===")

    required_auto = [
        ("data/macro/worldbank_wdi.csv", "python data/download_macro.py"),
        ("data/macro/owid_mean_years_of_schooling.csv", "python data/download_owid.py"),
        ("data/macro/owid_homicide_rate_unodc.csv", "python data/download_owid.py"),
        ("data/macro/who_ncd_pac.csv", "python data/download_who_gho.py"),
        ("data/macro/who_suicide_mh12.csv", "python data/download_who_gho.py"),
    ]

    for rel, how in required_auto:
        if not _exists(_path(*rel.split("/"))):
            missing.append(rel)
            print(f"MISSING: {rel}")
            print(f"Next:    {how}")

    # Manual (IHME)
    ihme_required = "data/macro/ihme_depression.csv"
    ihme_path = _path(*ihme_required.split("/"))
    if not _exists(ihme_path):
        missing.append(ihme_required)
        print(f"\nMISSING (manual): {ihme_required}")
        print("Download from IHME GBD Results:")
        print("  - https://vizhub.healthdata.org/gbd-results/")
        print("  - Cause: Depressive disorders")
        print("  - Measure: Prevalence")
        print("  - Metric: Rate")
        print("  - Age: Age-standardized")
        print("  - Sex: Both")
        print("  - Location: All countries and territories")
        print("  - Year: 1990-2023")
        print("Export CSV and rename to: data/macro/ihme_depression.csv")
    else:
        print(f"\nFOUND: {ihme_required}")
        try:
            import pandas as pd

            def _read_delimited(path: str) -> "pd.DataFrame":
                for sep in [",", "\t", "|"]:
                    try:
                        df = pd.read_csv(path, sep=sep, low_memory=False)
                        if df.shape[1] > 2:
                            return df
                    except Exception:
                        continue
                return pd.read_csv(path, low_memory=False)

            df = _read_delimited(ihme_path)
            col_map = {c.lower(): c for c in df.columns}

            def _col(cands: list[str]) -> str | None:
                for c in cands:
                    if c.lower() in col_map:
                        return col_map[c.lower()]
                return None

            cause_col = _col(["cause_name", "cause"])
            measure_col = _col(["measure_name", "measure"])
            metric_col = _col(["metric_name", "metric"])
            sex_col = _col(["sex_name", "sex"])
            age_col = _col(["age_name", "age"])

            def _uniq(col: str, limit: int = 6) -> list[str]:
                vals = df[col].astype(str).str.strip().str.lower()
                out = sorted([v for v in vals.unique().tolist() if v])[:limit]
                return out

            extra = []
            for label, col in [("cause", cause_col), ("measure", measure_col), ("metric", metric_col), ("sex", sex_col), ("age", age_col)]:
                if col is None:
                    continue
                u = _uniq(col)
                if len(u) > 1:
                    extra.append(label)
                print(f"  {label:7s}: {', '.join(u) if u else '(none)'}")

            if all(c is not None for c in [cause_col, measure_col, metric_col, sex_col, age_col]):
                s_cause = df[cause_col].astype(str).str.strip().str.lower()
                s_measure = df[measure_col].astype(str).str.strip().str.lower()
                s_metric = df[metric_col].astype(str).str.strip().str.lower()
                s_sex = df[sex_col].astype(str).str.strip().str.lower()
                s_age = df[age_col].astype(str).str.strip().str.lower()

                mask = (
                    s_cause.str.contains("depressive disorders", na=False)
                    & (s_measure == "prevalence")
                    & (s_metric == "rate")
                    & s_sex.str.startswith("both", na=False)
                    & s_age.str.contains("standard", na=False)
                )
                n_match = int(mask.sum())
                print(f"  matched: {n_match:,} rows for the intended slice")
                if n_match == 0:
                    print("  WARN: Intended slice not found; re-export from IHME with the exact filters in data/README_data.md.")

            if extra:
                print("  NOTE: This file contains multiple slices (" + ", ".join(extra) + ").")
                print("        data/build_macro_panel.py will try to auto-filter to:")
                print("        cause=Depressive disorders, measure=Prevalence, metric=Rate, sex=Both, age=Age-standardized.")
        except Exception as e:
            print(f"  [WARN] Could not inspect IHME depression CSV: {e}")

    # Panel build
    panel_candidates = ["panel_with_inactivity.csv", "panel_merged.csv"]
    has_panel = any(_exists(os.path.join(macro_dir, f)) for f in panel_candidates)
    if not has_panel:
        if _exists(os.path.join(macro_dir, "worldbank_wdi.csv")) and _exists(os.path.join(macro_dir, "ihme_depression.csv")):
            print("\nNOTE: Macro inputs found, but merged panel is missing.")
            print("Next: python data/build_macro_panel.py")
        else:
            print("\nNOTE: Macro panel build is skipped until WDI + IHME depression are present.")
            print("After placing files, run: python data/build_macro_panel.py")
    else:
        print("\nOK: macro panel CSV found:")
        for f in panel_candidates:
            p = os.path.join(macro_dir, f)
            if _exists(p):
                print(f"  - data/macro/{f}")

    return missing


def main() -> int:
    print("============================================================")
    print(" Data check: Cognitive Obesity replication")
    print("============================================================")

    missing_all: list[str] = []
    missing_all.extend(check_nhanes())
    missing_all.extend(check_atus())
    missing_all.extend(check_macro())

    print("\n============================================================")
    if not missing_all:
        print("All required files are present for the full pipeline.")
        print("Next: bash run_all.sh")
        return 0

    # Deduplicate while keeping order
    seen = set()
    dedup = []
    for m in missing_all:
        if m in seen:
            continue
        seen.add(m)
        dedup.append(m)

    print("Missing items detected:")
    for m in dedup:
        print(f"  - {m}")

    print("\nNext recommended steps:")
    print("  1) Run the download scripts shown above (NHANES/WDI/OWID/WHO).")
    print("  2) Manually download IHME depression CSV (see instructions above).")
    print("  3) Build macro panel: python data/build_macro_panel.py")
    print("  4) Run: bash run_all.sh")

    print("\nFor details: data/README_data.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
