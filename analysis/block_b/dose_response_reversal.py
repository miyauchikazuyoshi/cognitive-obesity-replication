#!/usr/bin/env python3
"""
Dose-Response Reversal Analysis (Section 2.2.6)
================================================
After Hansen PTR finds no discontinuous threshold for depression,
test whether the relationship between ad exposure proxy and outcomes
is better described as a continuous inverted-U (quadratic) pattern.

Implements:
  1) Linear vs quadratic (country FE; within transformation)
  2) Restricted cubic spline (3 knots; country FE)
  3) Reversal point (marginal effect = 0) + cluster bootstrap CI
  4) Suicide as a hard-outcome comparison (if available)

Inputs (data/macro/):
  - panel_merged.csv (preferred) or any compatible macro panel CSV.

Outputs:
  - results/figures/dose_response_reversal.png
  - results/dose_response_reversal.json
"""

from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from numpy.linalg import lstsq
from scipy import stats

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "macro")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def load_panel() -> pd.DataFrame:
    candidates = [
        "panel_with_inactivity.csv",
        "panel_merged.csv",
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
    print("Expected one of:")
    for fname in candidates:
        print(f"  - data/macro/{fname}")
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
    proxy_col = find(["ad_proxy", "proxy", "adproxy"])
    internet_col = find(["internet", "internet_pct", "it.net.user.zs"])
    gdp_col = find(["gdp", "gdp_pc", "gdp_per_capita", "ny.gdp.pcap.cd", "gdp_current_usd", "gdp_ppp_2017"])

    if country_col is None or year_col is None or dep_col is None:
        print("ERROR: cannot identify required columns (country/year/depression) in panel.")
        print(f"Columns: {list(df.columns)[:60]} ...")
        sys.exit(1)

    rename[country_col] = "country"
    rename[year_col] = "year"
    rename[dep_col] = "depression_prevalence"
    if suicide_col is not None:
        rename[suicide_col] = "suicide"
    if proxy_col is not None:
        rename[proxy_col] = "ad_proxy"
    if internet_col is not None:
        rename[internet_col] = "internet"
    if gdp_col is not None:
        rename[gdp_col] = "gdp"

    return df.rename(columns=rename)


def ensure_ad_proxy(df: pd.DataFrame) -> pd.DataFrame:
    if "ad_proxy" in df.columns and df["ad_proxy"].notna().any():
        return df

    if "internet" not in df.columns or "gdp" not in df.columns:
        print("ERROR: 'ad_proxy' missing and cannot be constructed (need internet & gdp).")
        print("Available columns:", ", ".join(map(str, df.columns)))
        sys.exit(1)

    internet = pd.to_numeric(df["internet"], errors="coerce")
    gdp = pd.to_numeric(df["gdp"], errors="coerce")

    # proxy = Internet(%) × GDP/capita($) / 1000
    internet_pct = internet * 100.0 if internet.max(skipna=True) <= 1.5 else internet
    df = df.copy()
    df["ad_proxy"] = internet_pct * gdp / 1000.0
    return df


def rcs_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """
    Restricted cubic spline basis (Harrell). For 3 knots, returns 2 columns:
      [x, cubic_term]
    """
    basis = [x]
    kn = knots
    for j in range(len(knots) - 2):
        term = (
            np.maximum(0, (x - kn[j]) ** 3)
            - np.maximum(0, (x - kn[-2]) ** 3) * (kn[-1] - kn[j]) / (kn[-1] - kn[-2])
            + np.maximum(0, (x - kn[-1]) ** 3) * (kn[-2] - kn[j]) / (kn[-1] - kn[-2])
        )
        basis.append(term)
    return np.column_stack(basis)


df = ensure_ad_proxy(normalize_columns(load_panel()))

panel = df.dropna(subset=["country", "year", "depression_prevalence", "ad_proxy"]).copy()
panel["ad_proxy_sq"] = panel["ad_proxy"] ** 2

# Country-demean (within transformation)
for col in ["depression_prevalence", "ad_proxy", "ad_proxy_sq"]:
    panel[f"{col}_dm"] = panel.groupby("country")[col].transform(lambda x: x - x.mean())

y = panel["depression_prevalence_dm"].values
x1 = panel["ad_proxy_dm"].values
x2 = panel["ad_proxy_sq_dm"].values

# Linear model
X_lin = x1.reshape(-1, 1)
b_lin, _, _, _ = lstsq(X_lin, y, rcond=None)
rss_lin = np.sum((y - X_lin @ b_lin) ** 2)

# Quadratic model
X_quad = np.column_stack([x1, x2])
b_quad, _, _, _ = lstsq(X_quad, y, rcond=None)
rss_quad = np.sum((y - X_quad @ b_quad) ** 2)

n = len(y)
nc = panel["country"].nunique()

F_quad = ((rss_lin - rss_quad) / 1) / (rss_quad / (n - nc - 2))
p_quad = 1 - stats.f.cdf(F_quad, 1, n - nc - 2)

aic_lin = n * np.log(rss_lin / n) + 2 * (nc + 1)
aic_quad = n * np.log(rss_quad / n) + 2 * (nc + 2)

beta1 = float(b_quad[0])
beta2 = float(b_quad[1])
reversal = None
if beta2 != 0:
    reversal = float(-beta1 / (2 * beta2))

print("=" * 70)
print("1. QUADRATIC MODEL (DEPRESSION)")
print("=" * 70)
print(f"N = {n:,}, Countries = {nc:,}")
print(f"β₁ = {beta1:.8f}")
print(f"β₂ = {beta2:.12f}")
print(f"F-test (quad vs linear): F = {F_quad:.2f}, p = {p_quad:.6e}")
print(f"AIC: linear = {aic_lin:.1f}, quadratic = {aic_quad:.1f}")
if reversal is not None:
    print(f"Reversal point (marginal effect = 0): proxy ≈ {reversal:.0f}")

# ----------------------------------------------------------------------
# Restricted cubic spline (3 knots)
# ----------------------------------------------------------------------
proxy_vals = panel["ad_proxy"].values
knots = np.percentile(proxy_vals, [25, 50, 75])
spline_raw = rcs_basis(proxy_vals, knots)
spline_dm = np.zeros_like(spline_raw)
for col_idx in range(spline_raw.shape[1]):
    tmp = pd.Series(spline_raw[:, col_idx], index=panel.index, name=f"_sp{col_idx}")
    tmp_dm = tmp.groupby(panel["country"]).transform(lambda x: x - x.mean())
    spline_dm[:, col_idx] = tmp_dm.values

X_spl = spline_dm
b_spl, _, _, _ = lstsq(X_spl, y, rcond=None)
rss_spl = np.sum((y - X_spl @ b_spl) ** 2)
aic_spl = n * np.log(rss_spl / n) + 2 * (nc + X_spl.shape[1])

F_spl = ((rss_lin - rss_spl) / max(X_spl.shape[1] - 1, 1)) / (rss_spl / (n - nc - X_spl.shape[1]))
p_spl = 1 - stats.f.cdf(F_spl, max(X_spl.shape[1] - 1, 1), n - nc - X_spl.shape[1])

print("\n" + "=" * 70)
print("2. RESTRICTED CUBIC SPLINE (DEPRESSION)")
print("=" * 70)
print(f"Knots: {knots[0]:.1f}, {knots[1]:.1f}, {knots[2]:.1f}")
print(f"F-test (spline vs linear): F = {F_spl:.2f}, p = {p_spl:.6e}")
print(f"AIC: linear = {aic_lin:.1f}, quad = {aic_quad:.1f}, spline = {aic_spl:.1f}")

# ----------------------------------------------------------------------
# Suicide dose-response (optional)
# ----------------------------------------------------------------------
suicide_stats = None
if "suicide" in df.columns and df["suicide"].notna().any():
    panel_s = df.dropna(subset=["country", "year", "suicide", "ad_proxy"]).copy()
    panel_s["ad_proxy_sq"] = panel_s["ad_proxy"] ** 2
    for col in ["suicide", "ad_proxy", "ad_proxy_sq"]:
        panel_s[f"{col}_dm"] = panel_s.groupby("country")[col].transform(lambda x: x - x.mean())

    y_s = panel_s["suicide_dm"].values
    x1_s = panel_s["ad_proxy_dm"].values
    x2_s = panel_s["ad_proxy_sq_dm"].values

    X_lin_s = x1_s.reshape(-1, 1)
    b_lin_s, _, _, _ = lstsq(X_lin_s, y_s, rcond=None)
    rss_lin_s = np.sum((y_s - X_lin_s @ b_lin_s) ** 2)

    X_quad_s = np.column_stack([x1_s, x2_s])
    b_quad_s, _, _, _ = lstsq(X_quad_s, y_s, rcond=None)
    rss_quad_s = np.sum((y_s - X_quad_s @ b_quad_s) ** 2)

    n_s = len(y_s)
    nc_s = panel_s["country"].nunique()
    F_quad_s = ((rss_lin_s - rss_quad_s) / 1) / (rss_quad_s / (n_s - nc_s - 2))
    p_quad_s = 1 - stats.f.cdf(F_quad_s, 1, n_s - nc_s - 2)

    beta1_s = float(b_quad_s[0])
    beta2_s = float(b_quad_s[1])
    reversal_s = None
    if beta2_s != 0:
        reversal_s = float(-beta1_s / (2 * beta2_s))

    print("\n" + "=" * 70)
    print("3. QUADRATIC MODEL (SUICIDE)")
    print("=" * 70)
    print(f"N = {n_s:,}, Countries = {nc_s:,}")
    print(f"β₁ = {beta1_s:.8f}, β₂ = {beta2_s:.12f}")
    print(f"F (quad vs linear): F = {F_quad_s:.2f}, p = {p_quad_s:.6e}")
    if reversal_s is not None:
        print(f"Reversal point: proxy ≈ {reversal_s:.0f}")

    suicide_stats = {
        "n": int(n_s),
        "nc": int(nc_s),
        "beta1": beta1_s,
        "beta2": beta2_s,
        "F_quad": float(F_quad_s),
        "p_quad": float(p_quad_s),
        "reversal_proxy": reversal_s,
    }

# ----------------------------------------------------------------------
# Bootstrap CI for reversal point (depression)
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("4. BOOTSTRAP CI FOR REVERSAL POINT (DEPRESSION)")
print("=" * 70)

np.random.seed(42)
n_boot = 1000
reversal_boot: list[float] = []
unique_c = panel["country"].unique()

for b in range(n_boot):
    bc = np.random.choice(unique_c, size=len(unique_c), replace=True)
    frames = []
    for j, c in enumerate(bc):
        tmp = panel[panel["country"] == c].copy()
        tmp = tmp.assign(country_boot=f"{c}_{j}")
        frames.append(tmp)
    bd = pd.concat(frames, ignore_index=True)

    # Within transformation using bootstrap cluster id
    for col in ["depression_prevalence", "ad_proxy", "ad_proxy_sq"]:
        bd[f"{col}_dm"] = bd.groupby("country_boot")[col].transform(lambda x: x - x.mean())

    yb = bd["depression_prevalence_dm"].values
    xb = np.column_stack([bd["ad_proxy_dm"].values, bd["ad_proxy_sq_dm"].values])

    try:
        bb, _, _, _ = lstsq(xb, yb, rcond=None)
        b1 = float(bb[0])
        b2 = float(bb[1])
        if b2 != 0:
            rev = -b1 / (2 * b2)
            if np.isfinite(rev) and 0 < rev < 200_000:
                reversal_boot.append(float(rev))
    except Exception:
        pass

    if (b + 1) % 250 == 0:
        print(f"  Bootstrap {b+1}/{n_boot}...")

reversal_boot_arr = np.array(reversal_boot, dtype=float)
boot_summary = None
if len(reversal_boot_arr) >= 20:
    boot_summary = {
        "n_boot_kept": int(len(reversal_boot_arr)),
        "median": float(np.median(reversal_boot_arr)),
        "mean": float(np.mean(reversal_boot_arr)),
        "ci95_low": float(np.percentile(reversal_boot_arr, 2.5)),
        "ci95_high": float(np.percentile(reversal_boot_arr, 97.5)),
        "sd": float(np.std(reversal_boot_arr)),
    }
    print(f"\nReversal bootstrap (kept = {boot_summary['n_boot_kept']:,}):")
    print(f"  Median = {boot_summary['median']:.0f}")
    print(f"  95% CI = [{boot_summary['ci95_low']:.0f}, {boot_summary['ci95_high']:.0f}]")
else:
    print("\nWARN: Too few bootstrap draws kept for a stable CI.")

# ----------------------------------------------------------------------
# Visualization (paper-style summary figure)
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# A: Depression dose-response (binned means)
ax = axes[0, 0]
bins = pd.qcut(panel["ad_proxy"], 20, duplicates="drop")
binned = (
    panel.groupby(bins)
    .agg(proxy_mean=("ad_proxy", "mean"), dep_mean=("depression_prevalence", "mean"), n=("depression_prevalence", "count"))
    .reset_index(drop=True)
)
ax.scatter(
    binned["proxy_mean"],
    binned["dep_mean"],
    s=binned["n"] / 5,
    alpha=0.6,
    color="steelblue",
    edgecolor="black",
    linewidth=0.3,
)
z = np.polyfit(binned["proxy_mean"], binned["dep_mean"], 2)
x_fit = np.linspace(0, binned["proxy_mean"].max(), 200)
y_fit = np.polyval(z, x_fit)
ax.plot(x_fit, y_fit, "r-", linewidth=2, label="Quadratic fit (binned)")
ax.set_xlabel("Ad Exposure Proxy")
ax.set_ylabel("Depression Prevalence (%)")
ax.set_title("A. Depression: Continuous Dose-Response", fontweight="bold")
ax.legend(fontsize=9)

# B: Marginal effect curve (quadratic FE)
ax = axes[0, 1]
if reversal is not None:
    x_pred = np.linspace(0, np.nanmax(panel["ad_proxy"].values), 1000)
    dy = beta1 + 2.0 * beta2 * x_pred
    ax.plot(x_pred, dy * 1000, "b-", linewidth=2)
    ax.axhline(y=0, color="red", linestyle="--", linewidth=1)
    ax.axvline(x=reversal, color="green", linestyle=":", linewidth=1.5, label=f"ME = 0 at proxy ≈ {reversal:.0f}")
    ax.set_xlabel("Ad Exposure Proxy")
    ax.set_ylabel("Marginal Effect (×1000)")
    ax.set_title("B. Depression: Marginal Effect (FE quadratic)", fontweight="bold")
    ax.legend(fontsize=9)
else:
    ax.axis("off")
    ax.text(0.5, 0.5, "B: Marginal Effect\nSKIPPED (β₂=0)", ha="center", va="center", fontweight="bold")

# C: Suicide dose-response (binned means)
ax = axes[1, 0]
if "suicide" in df.columns and df["suicide"].notna().any():
    panel_s_plot = df.dropna(subset=["suicide", "ad_proxy"]).copy()
    bins_s = pd.qcut(panel_s_plot["ad_proxy"], 20, duplicates="drop")
    binned_s = (
        panel_s_plot.groupby(bins_s)
        .agg(proxy_mean=("ad_proxy", "mean"), sui_mean=("suicide", "mean"), n=("suicide", "count"))
        .reset_index(drop=True)
    )
    ax.scatter(
        binned_s["proxy_mean"],
        binned_s["sui_mean"],
        s=binned_s["n"] / 5,
        alpha=0.6,
        color="darkorange",
        edgecolor="black",
        linewidth=0.3,
    )
    if len(binned_s) > 3:
        z_s = np.polyfit(binned_s["proxy_mean"], binned_s["sui_mean"], 2)
        x_fit_s = np.linspace(0, binned_s["proxy_mean"].max(), 200)
        y_fit_s = np.polyval(z_s, x_fit_s)
        ax.plot(x_fit_s, y_fit_s, "r-", linewidth=2, label="Quadratic fit (binned)")
    ax.set_xlabel("Ad Exposure Proxy")
    ax.set_ylabel("Suicide Rate (per 100k)")
    ax.set_title("C. Suicide: Continuous Dose-Response", fontweight="bold")
    ax.legend(fontsize=9)
else:
    ax.axis("off")
    ax.text(0.5, 0.5, "C: Suicide\nSKIPPED (missing column)", ha="center", va="center", fontweight="bold")

# D: Model comparison
ax = axes[1, 1]
models = ["Linear", "Quadratic", "Spline (3 knots)"]
aic_vals = [aic_lin, aic_quad, aic_spl]
colors = ["#2196F3", "#4CAF50", "#FF9800"]
bars = ax.barh(models, aic_vals, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
ax.set_xlabel("AIC (lower = better)")
ax.set_title("D. Model Comparison (Depression)", fontweight="bold")
for i, v in enumerate(aic_vals):
    ax.text(v + 5, i, f"{v:.0f}", va="center", fontsize=9)

fig.suptitle("Dose-Response Analysis: Reversal Point Detection", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()

fig_path = os.path.join(FIG_DIR, "dose_response_reversal.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {fig_path}")

# ----------------------------------------------------------------------
# Save JSON summary for paper insertion
# ----------------------------------------------------------------------
out_json = {
    "depression": {
        "n": int(n),
        "nc": int(nc),
        "beta1": beta1,
        "beta2": beta2,
        "F_quad": float(F_quad),
        "p_quad": float(p_quad),
        "aic_lin": float(aic_lin),
        "aic_quad": float(aic_quad),
        "aic_spline": float(aic_spl),
        "reversal_proxy": reversal,
        "bootstrap": boot_summary,
    },
    "spline": {
        "knots": [float(k) for k in knots],
        "F_spline": float(F_spl),
        "p_spline": float(p_spl),
    },
    "suicide": suicide_stats,
}

json_path = os.path.join(RESULTS_DIR, "dose_response_reversal.json")
with open(json_path, "w") as f:
    json.dump(out_json, f, indent=2)
print(f"Saved: {json_path}")

