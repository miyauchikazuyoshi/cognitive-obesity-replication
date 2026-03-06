#!/usr/bin/env python3
"""
Service Sector Moderation (Section 2.2.7)
=========================================
Test whether the ad exposure proxy → depression relationship is moderated
by "cognitive density" proxied by service-sector employment share.

Implements:
  1) Country-mean tercile split by service employment share
     - quadratic dose-response by tercile
     - reversal point per tercile
  2) Interaction model:
       dep ~ proxy + proxy^2 + service + proxy×service + proxy^2×service (country FE)
     with an F-test comparing against the baseline quadratic model.

Inputs (data/macro/):
  - panel_merged.csv (preferred) with columns:
      country, year, depression_prevalence, internet, gdp, service_employment

Outputs:
  - results/figures/service_sector_moderation.png
  - results/service_sector_moderation.json
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
    svc_col = find(["service_employment", "sl.srv.empl.zs", "services_employment"])
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
    if svc_col is not None:
        rename[svc_col] = "service_employment"
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
    internet_pct = internet * 100.0 if internet.max(skipna=True) <= 1.5 else internet
    df = df.copy()
    df["ad_proxy"] = internet_pct * gdp / 1000.0
    return df


df = ensure_ad_proxy(normalize_columns(load_panel()))

required = ["country", "year", "depression_prevalence", "ad_proxy", "service_employment"]
missing = [c for c in required if c not in df.columns]
if missing:
    print("ERROR: required columns missing:", ", ".join(missing))
    print("Available columns:", ", ".join(map(str, df.columns)))
    sys.exit(1)

panel = df.dropna(subset=required).copy()
panel["ad_proxy_sq"] = panel["ad_proxy"] ** 2
print(f"Panel: N={len(panel):,}, Countries={panel['country'].nunique():,}")

# ----------------------------------------------------------------------
# 1) Tercile split by country-mean service employment
# ----------------------------------------------------------------------
svc_by_country = panel.groupby("country")["service_employment"].mean()
q33 = float(svc_by_country.quantile(0.33))
q66 = float(svc_by_country.quantile(0.66))

def svc_group(country: str) -> str:
    v = float(svc_by_country.get(country, np.nan))
    if not np.isfinite(v):
        return "NA"
    if v < q33:
        return "Low"
    if v >= q66:
        return "High"
    return "Mid"

panel["svc_group"] = panel["country"].map(svc_group)

group_results: dict[str, dict[str, float | int]] = {}
for group in ["Low", "Mid", "High"]:
    sub = panel[panel["svc_group"] == group].copy()
    if len(sub) < 50:
        continue

    for col in ["depression_prevalence", "ad_proxy", "ad_proxy_sq"]:
        sub[f"{col}_dm"] = sub.groupby("country")[col].transform(lambda x: x - x.mean())

    y = sub["depression_prevalence_dm"].values
    x1 = sub["ad_proxy_dm"].values
    x2 = sub["ad_proxy_sq_dm"].values

    X_lin = x1.reshape(-1, 1)
    b_lin, _, _, _ = lstsq(X_lin, y, rcond=None)
    rss_lin = np.sum((y - X_lin @ b_lin) ** 2)

    X_quad = np.column_stack([x1, x2])
    b_quad, _, _, _ = lstsq(X_quad, y, rcond=None)
    rss_quad = np.sum((y - X_quad @ b_quad) ** 2)

    nc = int(sub["country"].nunique())
    n = int(len(y))
    F = ((rss_lin - rss_quad) / 1) / (rss_quad / (n - nc - 2))
    p = 1 - stats.f.cdf(F, 1, n - nc - 2)

    beta1 = float(b_quad[0])
    beta2 = float(b_quad[1])
    reversal = None
    if beta2 != 0:
        reversal = float(-beta1 / (2 * beta2))

    group_results[group] = {
        "n": n,
        "nc": nc,
        "mean_service": float(sub["service_employment"].mean()),
        "beta1": beta1,
        "beta2": beta2,
        "F_quad": float(F),
        "p_quad": float(p),
        "mean_proxy": float(sub["ad_proxy"].mean()),
        "reversal_proxy": reversal,
    }

print("=" * 70)
print("TERCILE RESULTS (Quadratic within-country FE)")
print("=" * 70)
print(f"Terciles by country mean service employment: <{q33:.1f}%, {q33:.1f}-{q66:.1f}%, >{q66:.1f}%")
for group in ["Low", "Mid", "High"]:
    r = group_results.get(group)
    if not r:
        continue
    sig = "***" if r["p_quad"] < 0.001 else "**" if r["p_quad"] < 0.01 else "*" if r["p_quad"] < 0.05 else "n.s."
    rev_s = f"{r['reversal_proxy']:.0f}" if r["reversal_proxy"] is not None else "NA"
    print(f"\n{group} service ({r['mean_service']:.0f}%): N={r['n']:,}, countries={r['nc']:,}")
    print(f"  β₁ = {r['beta1']:.8f}, β₂ = {r['beta2']:.12f}")
    print(f"  Quadratic F = {r['F_quad']:.2f} ({sig})")
    print(f"  Reversal proxy ≈ {rev_s}")

# ----------------------------------------------------------------------
# 2) Interaction model: proxy + proxy^2 moderated by service_employment
# ----------------------------------------------------------------------
full = panel.copy()
full["proxy_x_svc"] = full["ad_proxy"] * full["service_employment"]
full["proxysq_x_svc"] = full["ad_proxy_sq"] * full["service_employment"]

for col in [
    "depression_prevalence",
    "ad_proxy",
    "ad_proxy_sq",
    "service_employment",
    "proxy_x_svc",
    "proxysq_x_svc",
]:
    full[f"{col}_dm"] = full.groupby("country")[col].transform(lambda x: x - x.mean())

y = full["depression_prevalence_dm"].values

X_quad = np.column_stack([full["ad_proxy_dm"].values, full["ad_proxy_sq_dm"].values])
b_q, _, _, _ = lstsq(X_quad, y, rcond=None)
rss_q = np.sum((y - X_quad @ b_q) ** 2)

X_int = np.column_stack(
    [
        full["ad_proxy_dm"].values,
        full["ad_proxy_sq_dm"].values,
        full["service_employment_dm"].values,
        full["proxy_x_svc_dm"].values,
        full["proxysq_x_svc_dm"].values,
    ]
)
b_int, _, _, _ = lstsq(X_int, y, rcond=None)
rss_int = np.sum((y - X_int @ b_int) ** 2)

nc = int(full["country"].nunique())
n = int(len(y))
F_int = ((rss_q - rss_int) / 3) / (rss_int / (n - nc - 5))
p_int = 1 - stats.f.cdf(F_int, 3, n - nc - 5)

print("\n" + "=" * 70)
print("INTERACTION MODEL (proxy × service employment)")
print("=" * 70)
print(f"F-test (add 3 interaction terms): F = {F_int:.2f}, p = {p_int:.6e}")

# ----------------------------------------------------------------------
# Visualization (paper-style)
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
colors = {"Low": "#2196F3", "Mid": "#FF9800", "High": "#F44336"}

ax = axes[0]
for group in ["Low", "Mid", "High"]:
    sub = panel[panel["svc_group"] == group].copy()
    if len(sub) < 50:
        continue
    bins = pd.qcut(sub["ad_proxy"], 15, duplicates="drop")
    binned = (
        sub.groupby(bins)
        .agg(x=("ad_proxy", "mean"), y=("depression_prevalence", "mean"), n=("depression_prevalence", "count"))
        .reset_index(drop=True)
    )
    ax.scatter(
        binned["x"],
        binned["y"],
        s=binned["n"] / 3,
        alpha=0.5,
        color=colors[group],
        edgecolor="black",
        linewidth=0.3,
    )
    if len(binned) > 3:
        z = np.polyfit(binned["x"], binned["y"], 2)
        x_fit = np.linspace(binned["x"].min(), binned["x"].max(), 100)
        y_fit = np.polyval(z, x_fit)
        r = group_results.get(group, {})
        label = f"{group} svc"
        if r and isinstance(r.get("mean_service"), float):
            label = f"{group} svc ({r['mean_service']:.0f}%)"
        ax.plot(x_fit, y_fit, "-", color=colors[group], linewidth=2, label=label)

ax.set_xlabel("Ad Exposure Proxy")
ax.set_ylabel("Depression Prevalence (%)")
ax.set_title("A. Dose-Response by Service Sector Intensity", fontweight="bold")
ax.legend(fontsize=9)

ax = axes[1]
groups = ["Low", "Mid", "High"]
reversals = [group_results.get(g, {}).get("reversal_proxy") for g in groups]
proxy_means = [group_results.get(g, {}).get("mean_proxy") for g in groups]

bars = ax.bar(
    groups,
    [r if r is not None else np.nan for r in reversals],
    color=[colors[g] for g in groups],
    alpha=0.8,
    edgecolor="black",
    linewidth=0.5,
)
for i, pm in enumerate(proxy_means):
    if pm is None:
        continue
    ax.scatter(i, pm, color="black", s=100, zorder=5, marker="D", label="Mean proxy" if i == 0 else "")
ax.set_ylabel("Ad Exposure Proxy")
ax.set_title("B. Reversal Point by Service Intensity", fontweight="bold")
ax.legend(fontsize=9)

fig.suptitle("Service Sector Moderation of Cognitive Exposure Effects", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
fig_path = os.path.join(FIG_DIR, "service_sector_moderation.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {fig_path}")

# ----------------------------------------------------------------------
# Save JSON summary for paper insertion
# ----------------------------------------------------------------------
out = {
    "terciles": {"q33": q33, "q66": q66},
    "by_group": group_results,
    "interaction": {"F": float(F_int), "p": float(p_int)},
    "n": int(len(panel)),
    "nc": int(panel["country"].nunique()),
}

json_path = os.path.join(RESULTS_DIR, "service_sector_moderation.json")
with open(json_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"Saved: {json_path}")

