#!/usr/bin/env python3
"""
Threshold Sweep Analyses (Section 2.2.4 / 2.2.5)
================================================
Exploratory "phase transition" checks by progressively restricting the sample
to higher-exposure environments and re-estimating a TWFE model.

Implements:
  A) Internet penetration threshold sweep (30% → 85%)
     TWFE: outcome ~ internet_share + country FE + year FE

  B) Ad exposure proxy threshold sweep
     proxy = Internet(%) × GDP/capita($) / 1000
     TWFE: outcome ~ log(proxy) + country FE + year FE

Outputs:
  - results/figures/phase_transition_threshold.png
  - results/figures/ad_proxy_phase_transition.png
  - results/threshold_sweep.json
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

    if country_col is None or year_col is None:
        print("ERROR: cannot identify country/year columns in panel.")
        print(f"Columns: {list(df.columns)[:60]} ...")
        sys.exit(1)

    rename[country_col] = "country"
    rename[year_col] = "year"
    if dep_col is not None:
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


def ensure_internet_and_proxy(df: pd.DataFrame) -> pd.DataFrame:
    if "internet" not in df.columns or df["internet"].isna().all():
        print("ERROR: internet column missing. Run data/download_macro.py and data/build_macro_panel.py")
        sys.exit(1)

    out = df.copy()
    internet = pd.to_numeric(out["internet"], errors="coerce")
    out["internet_pct"] = internet * 100.0 if internet.max(skipna=True) <= 1.5 else internet
    out["internet_share"] = out["internet_pct"] / 100.0

    if "ad_proxy" not in out.columns or out["ad_proxy"].isna().all():
        if "gdp" not in out.columns or out["gdp"].isna().all():
            print("ERROR: ad_proxy missing and cannot be constructed (need gdp).")
            sys.exit(1)
        gdp = pd.to_numeric(out["gdp"], errors="coerce")
        out["ad_proxy"] = out["internet_pct"] * gdp / 1000.0

    return out


def twfe_ols(df: pd.DataFrame, outcome: str, x: str) -> dict[str, float | int | None]:
    sub = df.dropna(subset=["country", "year", outcome, x]).copy()
    if len(sub) < 200 or sub["country"].nunique() < 20 or sub["year"].nunique() < 5:
        return {"beta": None, "t": None, "p": None, "n": int(len(sub)), "nc": int(sub["country"].nunique())}

    # Classical OLS SE (matches the style of other replication scripts here).
    model = smf.ols(f"{outcome} ~ {x} + C(country) + C(year)", data=sub).fit()
    return {
        "beta": float(model.params.get(x, np.nan)),
        "t": float(model.tvalues.get(x, np.nan)),
        "p": float(model.pvalues.get(x, np.nan)),
        "n": int(model.nobs),
        "nc": int(sub["country"].nunique()),
    }


def point_style(beta: float | None, p: float | None) -> tuple[str, float]:
    if beta is None or p is None or not np.isfinite(beta) or not np.isfinite(p):
        return ("#9ca3af", 0.5)  # gray
    if p >= 0.05:
        return ("#f59e0b", 0.8)  # orange (n.s.)
    return ("#16a34a", 0.9) if beta < 0 else ("#dc2626", 0.9)  # green / red


df = ensure_internet_and_proxy(normalize_columns(load_panel()))

if "depression_prevalence" not in df.columns or df["depression_prevalence"].isna().all():
    print("ERROR: depression_prevalence column missing. Place IHME depression CSV and rebuild panel.")
    sys.exit(1)

has_suicide = "suicide" in df.columns and df["suicide"].notna().any()

# ----------------------------------------------------------------------
# A) Internet threshold sweep
# ----------------------------------------------------------------------
internet_thresholds = list(range(30, 86))
internet_dep = []
internet_sui = []

for thr in internet_thresholds:
    sub = df[df["internet_pct"] > thr].copy()
    r_dep = twfe_ols(sub, "depression_prevalence", "internet_share")
    internet_dep.append({"threshold_pct": thr, **r_dep})

    if has_suicide:
        r_sui = twfe_ols(sub, "suicide", "internet_share")
        internet_sui.append({"threshold_pct": thr, **r_sui})

print("=" * 70)
print("INTERNET THRESHOLD SWEEP (TWFE)")
print("=" * 70)
for thr in [30, 50, 70, 85]:
    r = next((x for x in internet_dep if x["threshold_pct"] == thr), None)
    if r and r["beta"] is not None:
        print(f"Depression | internet>{thr}%: β={r['beta']:.3f}, t={r['t']:.2f}, N={r['n']:,}")
    if has_suicide:
        rs = next((x for x in internet_sui if x["threshold_pct"] == thr), None)
        if rs and rs["beta"] is not None:
            print(f"Suicide    | internet>{thr}%: β={rs['beta']:.3f}, t={rs['t']:.2f}, N={rs['n']:,}")

fig, axes = plt.subplots(1, 2 if has_suicide else 1, figsize=(12 if has_suicide else 6, 4))
if not isinstance(axes, np.ndarray):
    axes = np.array([axes])

ax = axes[0]
xs = [d["threshold_pct"] for d in internet_dep]
ys = [d["beta"] if d["beta"] is not None else np.nan for d in internet_dep]
colors = []
alphas = []
for d in internet_dep:
    c, a = point_style(d["beta"], d["p"])
    colors.append(c)
    alphas.append(a)
ax.scatter(xs, ys, c=colors, alpha=0.9, s=18)
ax.plot(xs, ys, color="#111827", alpha=0.25, linewidth=1)
ax.axhline(0, color="#6b7280", linewidth=1, alpha=0.6)
ax.set_title("Depression (TWFE): internet effect by threshold", fontweight="bold")
ax.set_xlabel("Internet penetration threshold (%)")
ax.set_ylabel("β on internet_share")

if has_suicide:
    ax = axes[1]
    xs = [d["threshold_pct"] for d in internet_sui]
    ys = [d["beta"] if d["beta"] is not None else np.nan for d in internet_sui]
    colors = []
    for d in internet_sui:
        c, _ = point_style(d["beta"], d["p"])
        colors.append(c)
    ax.scatter(xs, ys, c=colors, alpha=0.9, s=18)
    ax.plot(xs, ys, color="#111827", alpha=0.25, linewidth=1)
    ax.axhline(0, color="#6b7280", linewidth=1, alpha=0.6)
    ax.set_title("Suicide (TWFE): internet effect by threshold", fontweight="bold")
    ax.set_xlabel("Internet penetration threshold (%)")
    ax.set_ylabel("β on internet_share")

plt.tight_layout()
fig8_path = os.path.join(FIG_DIR, "phase_transition_threshold.png")
plt.savefig(fig8_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {fig8_path}")

# ----------------------------------------------------------------------
# B) Ad proxy threshold sweep (log proxy)
# ----------------------------------------------------------------------
proxy_base = df.dropna(subset=["ad_proxy", "gdp"]).copy()
proxy_base = proxy_base[(proxy_base["ad_proxy"] > 0)].copy()
proxy_base["log_ad_proxy"] = np.log(proxy_base["ad_proxy"])

# Sweep across proxy percentiles + paper anchor thresholds
quant_thr = np.unique(np.percentile(proxy_base["ad_proxy"].values, np.linspace(5, 95, 60)))
anchor_thr = np.array([55, 148, 403, 1097, 8103, 22026], dtype=float)
proxy_thresholds = np.unique(np.concatenate([quant_thr, anchor_thr]))
proxy_thresholds = proxy_thresholds[np.isfinite(proxy_thresholds)]
proxy_thresholds = proxy_thresholds[proxy_thresholds > 0]
proxy_thresholds = sorted(float(x) for x in proxy_thresholds)

proxy_dep = []
proxy_sui = []
for thr in proxy_thresholds:
    sub = proxy_base[proxy_base["ad_proxy"] > thr].copy()
    r_dep = twfe_ols(sub, "depression_prevalence", "log_ad_proxy")
    proxy_dep.append({"threshold_proxy": thr, **r_dep})
    if has_suicide:
        r_sui = twfe_ols(sub, "suicide", "log_ad_proxy")
        proxy_sui.append({"threshold_proxy": thr, **r_sui})

print("\n" + "=" * 70)
print("AD PROXY THRESHOLD SWEEP (TWFE; X=log(proxy))")
print("=" * 70)
for thr in [55, 403, 1097, 8103, 22026]:
    r = min(proxy_dep, key=lambda d: abs(d["threshold_proxy"] - thr))
    if r["beta"] is not None:
        print(f"Depression | proxy>{thr:,.0f}: β={r['beta']:.3f}, t={r['t']:.2f}, N={r['n']:,}")

fig, axes = plt.subplots(1, 2 if has_suicide else 1, figsize=(12 if has_suicide else 6, 4))
if not isinstance(axes, np.ndarray):
    axes = np.array([axes])

ax = axes[0]
xs = [d["threshold_proxy"] for d in proxy_dep]
ys = [d["beta"] if d["beta"] is not None else np.nan for d in proxy_dep]
colors = []
for d in proxy_dep:
    c, _ = point_style(d["beta"], d["p"])
    colors.append(c)
ax.scatter(xs, ys, c=colors, alpha=0.9, s=18)
ax.plot(xs, ys, color="#111827", alpha=0.25, linewidth=1)
ax.axhline(0, color="#6b7280", linewidth=1, alpha=0.6)
ax.set_xscale("log")
ax.set_title("Depression (TWFE): log(proxy) effect by threshold", fontweight="bold")
ax.set_xlabel("Proxy threshold (log scale)")
ax.set_ylabel("β on log(proxy)")

if has_suicide:
    ax = axes[1]
    xs = [d["threshold_proxy"] for d in proxy_sui]
    ys = [d["beta"] if d["beta"] is not None else np.nan for d in proxy_sui]
    colors = []
    for d in proxy_sui:
        c, _ = point_style(d["beta"], d["p"])
        colors.append(c)
    ax.scatter(xs, ys, c=colors, alpha=0.9, s=18)
    ax.plot(xs, ys, color="#111827", alpha=0.25, linewidth=1)
    ax.axhline(0, color="#6b7280", linewidth=1, alpha=0.6)
    ax.set_xscale("log")
    ax.set_title("Suicide (TWFE): log(proxy) effect by threshold", fontweight="bold")
    ax.set_xlabel("Proxy threshold (log scale)")
    ax.set_ylabel("β on log(proxy)")

plt.tight_layout()
fig9_path = os.path.join(FIG_DIR, "ad_proxy_phase_transition.png")
plt.savefig(fig9_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {fig9_path}")

# ----------------------------------------------------------------------
# Save JSON
# ----------------------------------------------------------------------
out = {
    "internet_sweep": {
        "x": "internet_share",
        "thresholds_pct": internet_thresholds,
        "depression": internet_dep,
        "suicide": internet_sui if has_suicide else None,
    },
    "proxy_sweep": {
        "x": "log_ad_proxy",
        "thresholds_proxy": proxy_thresholds,
        "depression": proxy_dep,
        "suicide": proxy_sui if has_suicide else None,
    },
}

json_path = os.path.join(RESULTS_DIR, "threshold_sweep.json")
with open(json_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"Saved: {json_path}")

