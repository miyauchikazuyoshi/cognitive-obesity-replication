#!/usr/bin/env python3
"""
First-Difference Identification: proxy vs GDP (Section 2.2.5.2)
==============================================================
Use time-series variation (within-country first differences) plus year FE
to separate ad-ecosystem expansion (proxy) from pure economic growth (GDP).

Model (paper-style):
  Δy_it = β1·Δlog(proxy)_it + β2·Δlog(GDP)_it + Year FE + ε_it

where:
  proxy = Internet(%) × GDP/capita($) / 1000

Outputs:
  - results/figures/first_difference_horse_race.png
  - results/first_difference_proxy_vs_gdp.json
"""

from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "macro")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


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
    dep_col = find(["depression_prevalence", "dep", "dep_rate", "depression_rate", "depression"])
    suicide_col = find(["suicide", "suicide_rate", "sui_rate"])
    internet_col = find(["internet", "internet_pct", "it.net.user.zs"])
    gdp_col = find(["gdp", "gdp_pc", "gdp_per_capita", "ny.gdp.pcap.cd", "gdp_current_usd", "gdp_ppp_2017"])
    proxy_col = find(["ad_proxy", "proxy", "adproxy"])

    if country_col is None or year_col is None or dep_col is None:
        print("ERROR: cannot identify required columns (country/year/depression) in panel.")
        print(f"Columns: {list(df.columns)[:60]} ...")
        sys.exit(1)

    rename[country_col] = "country"
    rename[year_col] = "year"
    rename[dep_col] = "depression_prevalence"
    if suicide_col is not None:
        rename[suicide_col] = "suicide"
    if internet_col is not None:
        rename[internet_col] = "internet"
    if gdp_col is not None:
        rename[gdp_col] = "gdp"
    if proxy_col is not None:
        rename[proxy_col] = "ad_proxy"

    return df.rename(columns=rename)


def ensure_proxy(df: pd.DataFrame) -> pd.DataFrame:
    if "gdp" not in df.columns or df["gdp"].isna().all():
        print("ERROR: gdp column missing. Run data/download_macro.py and data/build_macro_panel.py")
        sys.exit(1)
    if "internet" not in df.columns or df["internet"].isna().all():
        print("ERROR: internet column missing. Run data/download_macro.py and data/build_macro_panel.py")
        sys.exit(1)

    out = df.copy()
    internet = pd.to_numeric(out["internet"], errors="coerce")
    out["internet_pct"] = internet * 100.0 if internet.max(skipna=True) <= 1.5 else internet
    out["internet_share"] = out["internet_pct"] / 100.0

    if "ad_proxy" not in out.columns or out["ad_proxy"].isna().all():
        gdp = pd.to_numeric(out["gdp"], errors="coerce")
        out["ad_proxy"] = out["internet_pct"] * gdp / 1000.0
    return out


def run_fd_year_fe(df: pd.DataFrame, outcome_diff: str) -> dict[str, float | int | None]:
    sub = df.dropna(subset=[outcome_diff, "d_log_proxy", "d_log_gdp", "year"]).copy()
    if len(sub) < 500 or sub["year"].nunique() < 10:
        return {"beta_proxy": None, "t_proxy": None, "beta_gdp": None, "t_gdp": None, "n": int(len(sub))}

    m = smf.ols(f"{outcome_diff} ~ d_log_proxy + d_log_gdp + C(year)", data=sub).fit()
    return {
        "beta_proxy": float(m.params.get("d_log_proxy", np.nan)),
        "t_proxy": float(m.tvalues.get("d_log_proxy", np.nan)),
        "beta_gdp": float(m.params.get("d_log_gdp", np.nan)),
        "t_gdp": float(m.tvalues.get("d_log_gdp", np.nan)),
        "n": int(m.nobs),
    }


df = ensure_proxy(normalize_columns(load_panel()))

base = df.dropna(subset=["country", "year", "depression_prevalence", "ad_proxy", "gdp"]).copy()
base = base[(base["ad_proxy"] > 0) & (base["gdp"] > 0)].copy()
base = base.sort_values(["country", "year"])

base["log_proxy"] = np.log(base["ad_proxy"])
base["log_gdp"] = np.log(base["gdp"])

base["d_dep"] = base.groupby("country")["depression_prevalence"].diff()
base["d_suicide"] = base.groupby("country")["suicide"].diff() if "suicide" in base.columns else np.nan
base["d_log_proxy"] = base.groupby("country")["log_proxy"].diff()
base["d_log_gdp"] = base.groupby("country")["log_gdp"].diff()

# Correlations of Δlog series
fd_corr = base.dropna(subset=["d_log_proxy", "d_log_gdp"]).copy()
overall_r = float(np.corrcoef(fd_corr["d_log_proxy"], fd_corr["d_log_gdp"])[0, 1]) if len(fd_corr) > 10 else np.nan

fd_90s = fd_corr[(fd_corr["year"] >= 1991) & (fd_corr["year"] <= 1999)]
r_90s = float(np.corrcoef(fd_90s["d_log_proxy"], fd_90s["d_log_gdp"])[0, 1]) if len(fd_90s) > 10 else np.nan

vif = None
if np.isfinite(overall_r):
    vif = float(1.0 / (1.0 - overall_r**2))

print("=" * 70)
print("Δlog(proxy) vs Δlog(GDP) correlation")
print("=" * 70)
print(f"Overall r = {overall_r:.2f} | VIF ≈ {vif:.1f}" if vif is not None else f"Overall r = {overall_r:.2f}")
print(f"1990s r = {r_90s:.2f}" if np.isfinite(r_90s) else "1990s r = NA")

# ----------------------------------------------------------------------
# Full-sample horse race (Δ + Year FE)
# ----------------------------------------------------------------------
full_dep = run_fd_year_fe(base, "d_dep")
print("\n" + "=" * 70)
print("First-difference + Year FE (Depression)")
print("=" * 70)
if full_dep["beta_proxy"] is not None:
    print(f"Δlog(proxy): β = {full_dep['beta_proxy']:.3f}, t = {full_dep['t_proxy']:.2f}")
    print(f"Δlog(GDP):   β = {full_dep['beta_gdp']:.3f}, t = {full_dep['t_gdp']:.2f}")
    print(f"N = {full_dep['n']:,}")

full_suicide = None
has_suicide = "suicide" in base.columns and base["suicide"].notna().any()
if has_suicide:
    full_suicide = run_fd_year_fe(base, "d_suicide")
    print("\n" + "=" * 70)
    print("First-difference + Year FE (Suicide)")
    print("=" * 70)
    if full_suicide["beta_proxy"] is not None:
        print(f"Δlog(proxy): β = {full_suicide['beta_proxy']:.3f}, t = {full_suicide['t_proxy']:.2f}")
        print(f"Δlog(GDP):   β = {full_suicide['beta_gdp']:.3f}, t = {full_suicide['t_gdp']:.2f}")
        print(f"N = {full_suicide['n']:,}")

# ----------------------------------------------------------------------
# High-penetration subsample (>70%)
# ----------------------------------------------------------------------
hi = base[base["internet_pct"] > 70].copy()
hi_dep = run_fd_year_fe(hi, "d_dep")
hi_suicide = run_fd_year_fe(hi, "d_suicide") if has_suicide else None

print("\n" + "=" * 70)
print("High penetration subsample (internet > 70%)")
print("=" * 70)
if hi_dep["beta_proxy"] is not None:
    print(f"[Dep] Δlog(proxy): β = {hi_dep['beta_proxy']:.3f}, t = {hi_dep['t_proxy']:.2f}")
    print(f"[Dep] Δlog(GDP):   β = {hi_dep['beta_gdp']:.3f}, t = {hi_dep['t_gdp']:.2f}")
    print(f"[Dep] N = {hi_dep['n']:,}")
if has_suicide and hi_suicide and hi_suicide["beta_proxy"] is not None:
    print(f"[Sui] Δlog(proxy): β = {hi_suicide['beta_proxy']:.3f}, t = {hi_suicide['t_proxy']:.2f}")
    print(f"[Sui] Δlog(GDP):   β = {hi_suicide['beta_gdp']:.3f}, t = {hi_suicide['t_gdp']:.2f}")
    print(f"[Sui] N = {hi_suicide['n']:,}")

# ----------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

def plot_panel(ax, title, r):
    if r["beta_proxy"] is None:
        ax.axis("off")
        ax.text(0.5, 0.5, f"{title}\nSKIPPED (insufficient data)", ha="center", va="center", fontweight="bold")
        return

    labels = ["Δlog(proxy)", "Δlog(GDP)"]
    tvals = [r["t_proxy"], r["t_gdp"]]
    colors = ["#dc2626", "#94a3b8"]
    ax.barh(labels, tvals, color=colors, edgecolor="white")
    ax.axvline(x=1.96, color="gray", linestyle="--", alpha=0.6)
    ax.axvline(x=-1.96, color="gray", linestyle="--", alpha=0.6)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("t-statistic")
    ax.set_xlim(min(-5, min(tvals) - 1), max(5, max(tvals) + 1))

plot_panel(axes[0], "Full sample: Δ + Year FE", full_dep)
plot_panel(axes[1], "High penetration (>70%): Δ + Year FE", hi_dep)

plt.tight_layout()
fig_path = os.path.join(FIG_DIR, "first_difference_horse_race.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {fig_path}")

# ----------------------------------------------------------------------
# Save JSON
# ----------------------------------------------------------------------
out = {
    "corr": {"overall_r_dlog_proxy_dlog_gdp": overall_r, "r_90s": r_90s, "vif": vif},
    "full": {"depression": full_dep, "suicide": full_suicide},
    "high_penetration": {"depression": hi_dep, "suicide": hi_suicide},
}

json_path = os.path.join(RESULTS_DIR, "first_difference_proxy_vs_gdp.json")
with open(json_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"Saved: {json_path}")

