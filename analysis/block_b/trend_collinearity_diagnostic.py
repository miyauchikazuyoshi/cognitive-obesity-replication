#!/usr/bin/env python3
"""
Trend-Collinearity Diagnostic (Robustness Appendix)
====================================================
Demonstrates why adding country-specific linear trends causes the
AdProxy coefficient to lose significance: an over-controlling /
collinearity problem, not evidence of no effect.

For most countries, within-country variation in internet penetration
and ad_proxy is almost entirely captured by a simple linear trend in
year.  Adding country × year fixed effects therefore absorbs the very
variation used to identify the coefficient.

Outputs
-------
- docs/paper/en/latex/figures/trend_collinearity.pdf
- docs/paper/ja_pre/latex/figures/trend_collinearity.pdf
- results/trend_collinearity_diagnostic.json
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
from matplotlib.ticker import PercentFormatter

# ── paths ────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data", "macro", "panel_merged.csv")
FIG_EN = os.path.join(ROOT, "docs", "paper", "en", "latex", "figures", "trend_collinearity.pdf")
FIG_JA = os.path.join(ROOT, "docs", "paper", "ja_pre", "latex", "figures", "trend_collinearity.pdf")
RESULTS = os.path.join(ROOT, "results", "trend_collinearity_diagnostic.json")

for p in [FIG_EN, FIG_JA, RESULTS]:
    os.makedirs(os.path.dirname(p), exist_ok=True)

# ── load data ────────────────────────────────────────────────────────
df = pd.read_csv(DATA)
print(f"Loaded {len(df)} rows, {df['code'].nunique()} countries")

# ── helper: per-country R² of variable ~ year (linear trend) ────────
def country_r2(df_in: pd.DataFrame, var: str) -> pd.DataFrame:
    """Return per-country R² of `var` regressed on year (OLS)."""
    records = []
    for code, g in df_in.groupby("code"):
        sub = g[["year", var]].dropna()
        if len(sub) < 4:
            continue
        y = sub[var].values
        x = sub["year"].values.astype(float)
        # demean for numerical stability
        xm = x - x.mean()
        ym = y - y.mean()
        ss_tot = (ym ** 2).sum()
        if ss_tot == 0:
            continue
        slope = (xm * ym).sum() / (xm ** 2).sum()
        resid = ym - slope * xm
        ss_res = (resid ** 2).sum()
        r2 = 1.0 - ss_res / ss_tot
        records.append({"code": code, "r2": r2, "n_obs": len(sub)})
    return pd.DataFrame(records)


def summarise_r2(r2_df: pd.DataFrame) -> dict:
    v = r2_df["r2"]
    return {
        "median": round(float(v.median()), 4),
        "mean": round(float(v.mean()), 4),
        "p25": round(float(v.quantile(0.25)), 4),
        "p75": round(float(v.quantile(0.75)), 4),
        "n_countries": int(len(v)),
    }


# ── step 2: compute per-country R² ──────────────────────────────────
r2_internet = country_r2(df, "internet")
r2_proxy = country_r2(df, "ad_proxy")
r2_dep = country_r2(df, "depression_prevalence")

sum_internet = summarise_r2(r2_internet)
sum_proxy = summarise_r2(r2_proxy)
sum_dep = summarise_r2(r2_dep)

print(f"\nInternet  R²: median={sum_internet['median']:.3f}  "
      f"mean={sum_internet['mean']:.3f}  (n={sum_internet['n_countries']})")
print(f"AdProxy   R²: median={sum_proxy['median']:.3f}  "
      f"mean={sum_proxy['mean']:.3f}  (n={sum_proxy['n_countries']})")
print(f"Depression R²: median={sum_dep['median']:.3f}  "
      f"mean={sum_dep['mean']:.3f}  (n={sum_dep['n_countries']})")

# ── step 4: variance ratio after de-trending ─────────────────────────
#   For each country, demean ad_proxy (within FE), then remove the
#   country-specific linear trend.  Compare residual variance to
#   demeaned variance.
var_demeaned_total = 0.0
var_residual_total = 0.0
n_total = 0

for code, g in df.groupby("code"):
    sub = g[["year", "ad_proxy"]].dropna()
    if len(sub) < 4:
        continue
    y = sub["ad_proxy"].values
    x = sub["year"].values.astype(float)
    ym = y - y.mean()  # within-country demeaned
    xm = x - x.mean()
    ss_x = (xm ** 2).sum()
    if ss_x == 0:
        continue
    slope = (xm * ym).sum() / ss_x
    resid = ym - slope * xm  # after removing linear trend

    var_demeaned_total += (ym ** 2).sum()
    var_residual_total += (resid ** 2).sum()
    n_total += len(sub)

variance_ratio = var_residual_total / var_demeaned_total
pct_absorbed = (1 - variance_ratio) * 100
print(f"\nVariance ratio (residual / demeaned): {variance_ratio:.4f}")
print(f"  → Linear trends absorb {pct_absorbed:.1f}% of within-country AdProxy variance")

# ── step 3: interpretation string ────────────────────────────────────
interp = (
    f"On average, {sum_proxy['mean']*100:.0f}% of within-country AdProxy variation "
    f"is captured by a linear trend (median R² = {sum_proxy['median']:.2f}). "
    f"Adding country-specific trends absorbs {pct_absorbed:.0f}% of demeaned variance, "
    f"leaving insufficient residual signal for coefficient estimation — "
    f"an over-controlling problem rather than evidence of null effect."
)
print(f"\n{interp}")

# ── step 5: figure ───────────────────────────────────────────────────

# Select representative countries for panel (a)
# Mix of early/late adopters, different income levels, different regions
REPRESENTATIVE = [
    "USA", "GBR", "JPN", "DEU", "KOR",   # early adopters, high-income
    "BRA", "MEX", "TUR", "THA",           # middle-income
    "IND", "NGA", "KEN", "BGD",           # late adopters, lower-income
    "CHN", "RUS",                          # large transition economies
]

# Filter to countries that exist in data with enough internet obs
rep_df = df[df["code"].isin(REPRESENTATIVE)].copy()
rep_codes = [c for c in REPRESENTATIVE
             if rep_df[rep_df["code"] == c]["internet"].notna().sum() >= 5]

# Colour palette: a professional qualitative palette (grayscale-friendly via linestyle)
cmap = plt.cm.tab20
colours = {c: cmap(i / len(rep_codes)) for i, c in enumerate(rep_codes)}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

# ── Panel (a): Internet Penetration Trajectories ────────────────────
ax = axes[0]
for code in rep_codes:
    sub = rep_df[rep_df["code"] == code][["year", "internet"]].dropna().sort_values("year")
    if len(sub) < 4:
        continue
    x = sub["year"].values.astype(float)
    y = sub["internet"].values
    # scatter
    ax.plot(x, y, "o", color=colours[code], markersize=2.5, alpha=0.7)
    # linear trend overlay — clip to observed year range
    xm = x - x.mean()
    ym = y - y.mean()
    slope = (xm * ym).sum() / (xm ** 2).sum()
    intercept = y.mean() - slope * x.mean()
    x_line = np.array([x.min(), x.max()])
    y_line = np.clip(slope * x_line + intercept, 0, None)
    ax.plot(x_line, y_line, "-", color=colours[code],
            linewidth=1.2, alpha=0.85, label=code)

ax.set_ylim(-2, 105)
ax.set_xlabel("Year", fontsize=10)
ax.set_ylabel("Internet penetration (%)", fontsize=10)
ax.set_title("(a) Internet Penetration Trajectories", fontsize=11, fontweight="bold")
ax.legend(fontsize=6, ncol=3, loc="upper left", frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(labelsize=9)

# ── Panel (b): Histogram of per-country R² for ad_proxy ~ year ──────
ax = axes[1]
vals = r2_proxy["r2"].values
bins = np.linspace(0, 1, 26)
ax.hist(vals, bins=bins, color="#4878A8", edgecolor="white", linewidth=0.5, alpha=0.85)
med = np.median(vals)
ax.axvline(med, color="#C44E52", linewidth=1.8, linestyle="--",
           label=f"Median $R^2$ = {med:.2f}")
ax.set_xlabel("$R^2$ (ad_proxy $\\sim$ year, within country)", fontsize=10)
ax.set_ylabel("Number of countries", fontsize=10)
ax.set_title("(b) Within-Country Variance\nAbsorbed by Linear Trends", fontsize=11, fontweight="bold")
ax.legend(fontsize=9, frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(labelsize=9)

# ── save figure ──────────────────────────────────────────────────────
for path in [FIG_EN, FIG_JA]:
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
plt.close(fig)

# ── step 6: save JSON results ────────────────────────────────────────
results = {
    "internet_r2": sum_internet,
    "ad_proxy_r2": sum_proxy,
    "depression_r2": sum_dep,
    "variance_ratio_after_detrend": round(float(variance_ratio), 4),
    "pct_variance_absorbed": round(float(pct_absorbed), 2),
    "interpretation": interp,
}

with open(RESULTS, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"Saved: {RESULTS}")

print("\nDone.")
