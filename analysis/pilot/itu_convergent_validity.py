#!/usr/bin/env python3
"""
ITU Mobile Broadband Convergent Validity Analysis
==================================================
Tests whether the AdProxy construct converges with an independent
measure of information-access density: mobile cellular subscriptions
per 100 inhabitants (World Bank indicator IT.CEL.SETS.P2).

Analyses:
  A. Convergent validity — Pearson/Spearman correlation (levels & FD)
  B. Substitution test — TWFE & FD models with mobile_cellular
  C. Horse-race — AdProxy + mobile_cellular in the same model
  D. Factor analysis — do both load on one latent factor?

Inputs:
  - World Bank API (IT.CEL.SETS.P2)
  - data/macro/panel_merged.csv

Outputs:
  - results/itu_convergent_validity.json
  - docs/paper/en/latex/figures/itu_convergent.pdf
  - docs/paper/ja_pre/latex/figures/itu_convergent.pdf
"""

from __future__ import annotations

import json
import os
import sys
import warnings
import urllib.request
import ssl

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ──────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PANEL_PATH = os.path.join(ROOT, "data", "macro", "panel_merged.csv")
RESULTS_DIR = os.path.join(ROOT, "results")
FIG_EN = os.path.join(ROOT, "docs", "paper", "en", "latex", "figures")
FIG_JA = os.path.join(ROOT, "docs", "paper", "ja_pre", "latex", "figures")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_EN, exist_ok=True)
os.makedirs(FIG_JA, exist_ok=True)


# ── Step 1: Download World Bank data ──────────────────────────────

def fetch_wb_indicator(indicator: str, start: int = 1990, end: int = 2023) -> pd.DataFrame:
    """Download a World Bank indicator via the v2 JSON API."""
    url = (
        f"https://api.worldbank.org/v2/country/all/indicator/{indicator}"
        f"?date={start}:{end}&format=json&per_page=20000"
    )
    print(f"  Fetching {indicator} from World Bank API ...")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={"User-Agent": "Python/cognitive-obesity"})
    with urllib.request.urlopen(req, context=ctx, timeout=60) as resp:
        raw = json.loads(resp.read().decode("utf-8"))

    if len(raw) < 2 or raw[1] is None:
        raise RuntimeError(f"No data returned for {indicator}")

    records = []
    for entry in raw[1]:
        val = entry.get("value")
        if val is None:
            continue
        iso3 = entry.get("countryiso3code", "")
        if not iso3:
            continue  # skip aggregates without ISO3
        records.append({
            "code": iso3,
            "country_wb": entry["country"]["value"],
            "year": int(entry["date"]),
            indicator: float(val),
        })
    df = pd.DataFrame(records)
    print(f"    -> {len(df)} observations, {df['code'].nunique()} entities")
    return df


def download_mobile_data() -> pd.DataFrame:
    """Try multiple WB indicators for mobile/broadband access."""
    # Primary: mobile cellular subscriptions per 100 people
    try:
        df = fetch_wb_indicator("IT.CEL.SETS.P2")
        df = df.rename(columns={"IT.CEL.SETS.P2": "mobile_cellular"})
        return df
    except Exception as e:
        print(f"  IT.CEL.SETS.P2 failed: {e}")

    # Fallback: fixed broadband per 100
    try:
        df = fetch_wb_indicator("IT.NET.BBND.P2")
        df = df.rename(columns={"IT.NET.BBND.P2": "mobile_cellular"})
        print("  Using IT.NET.BBND.P2 (fixed broadband) as fallback")
        return df
    except Exception as e:
        print(f"  IT.NET.BBND.P2 failed: {e}")

    raise RuntimeError("Could not fetch any mobile/broadband indicator from World Bank")


# ── Step 2: Merge with panel ──────────────────────────────────────

def load_and_merge() -> pd.DataFrame:
    """Load panel + mobile data, merge on code × year."""
    panel = pd.read_csv(PANEL_PATH)
    print(f"Panel: {len(panel)} rows, {panel['code'].nunique()} entities")

    mobile = download_mobile_data()

    # World Bank may return ISO2 or ISO3 — panel uses ISO3
    # Try ISO3 merge first; if very few matches, try ISO2→ISO3 mapping
    merged = panel.merge(mobile[["code", "year", "mobile_cellular"]],
                         on=["code", "year"], how="inner")
    print(f"Merged (inner): {len(merged)} rows, {merged['code'].nunique()} entities")

    if len(merged) < 100:
        # Fallback: WB returned ISO2 codes; use pycountry or manual mapping
        print("  Few matches — trying ISO2-to-ISO3 conversion ...")
        try:
            import pycountry
            def iso2_to_iso3(c):
                try:
                    return pycountry.countries.get(alpha_2=c).alpha_3
                except:
                    return None
            mobile["code3"] = mobile["code"].apply(iso2_to_iso3)
            mobile2 = mobile.dropna(subset=["code3"]).copy()
            mobile2["code"] = mobile2["code3"]
            merged = panel.merge(mobile2[["code", "year", "mobile_cellular"]],
                                 on=["code", "year"], how="inner")
            print(f"  Merged after ISO2→ISO3: {len(merged)} rows, {merged['code'].nunique()} entities")
        except ImportError:
            print("  pycountry not available; skipping ISO2 fallback")

    return merged


# ── Step 3: Analysis functions ────────────────────────────────────

def convergent_validity(df: pd.DataFrame) -> dict:
    """A. Pearson & Spearman correlations (levels and first-differences)."""
    results = {}
    sub = df.dropna(subset=["ad_proxy", "mobile_cellular"]).copy()
    n = len(sub)

    # Levels
    r_p, p_p = stats.pearsonr(sub["ad_proxy"], sub["mobile_cellular"])
    r_s, p_s = stats.spearmanr(sub["ad_proxy"], sub["mobile_cellular"])
    results["levels"] = {
        "pearson_r": round(r_p, 4), "pearson_p": float(f"{p_p:.2e}"),
        "spearman_rho": round(r_s, 4), "spearman_p": float(f"{p_s:.2e}"),
        "n": n,
    }
    print(f"\n  Levels — Pearson r = {r_p:.3f} (p = {p_p:.2e}), "
          f"Spearman rho = {r_s:.3f} (p = {p_s:.2e}), n = {n}")

    # First-differences
    sub = sub.sort_values(["code", "year"])
    sub["d_ad"] = sub.groupby("code")["ad_proxy"].diff()
    sub["d_mob"] = sub.groupby("code")["mobile_cellular"].diff()
    fd = sub.dropna(subset=["d_ad", "d_mob"])
    n_fd = len(fd)
    if n_fd > 10:
        r_p2, p_p2 = stats.pearsonr(fd["d_ad"], fd["d_mob"])
        r_s2, p_s2 = stats.spearmanr(fd["d_ad"], fd["d_mob"])
        results["first_diff"] = {
            "pearson_r": round(r_p2, 4), "pearson_p": float(f"{p_p2:.2e}"),
            "spearman_rho": round(r_s2, 4), "spearman_p": float(f"{p_s2:.2e}"),
            "n": n_fd,
        }
        print(f"  FD    — Pearson r = {r_p2:.3f} (p = {p_p2:.2e}), "
              f"Spearman rho = {r_s2:.3f} (p = {p_s2:.2e}), n = {n_fd}")
    return results


def run_twfe(df: pd.DataFrame, x_var: str, y_var: str) -> dict | None:
    """Run entity + year TWFE via OLS with dummies (statsmodels)."""
    import statsmodels.api as sm

    sub = df.dropna(subset=[x_var, y_var]).copy()
    if len(sub) < 50:
        return None

    # Entity dummies
    entity_dummies = pd.get_dummies(sub["code"], prefix="e", drop_first=True).astype(float)
    year_dummies = pd.get_dummies(sub["year"], prefix="y", drop_first=True).astype(float)
    X = pd.concat([sub[[x_var]].reset_index(drop=True),
                    entity_dummies.reset_index(drop=True),
                    year_dummies.reset_index(drop=True)], axis=1)
    X = sm.add_constant(X)
    y = sub[y_var].reset_index(drop=True)

    model = sm.OLS(y, X).fit(cov_type="HC1")
    coef = model.params[x_var]
    se = model.bse[x_var]
    p = model.pvalues[x_var]
    ci = model.conf_int().loc[x_var].tolist()
    return {
        "coef": round(float(coef), 6),
        "se": round(float(se), 6),
        "p": float(f"{p:.4e}"),
        "ci95": [round(ci[0], 6), round(ci[1], 6)],
        "n": int(model.nobs),
        "r2_adj": round(float(model.rsquared_adj), 4),
    }


def run_fd(df: pd.DataFrame, x_var: str, y_var: str) -> dict | None:
    """First-difference OLS: Δy ~ Δx."""
    import statsmodels.api as sm

    sub = df.dropna(subset=[x_var, y_var]).sort_values(["code", "year"]).copy()
    sub[f"d_{x_var}"] = sub.groupby("code")[x_var].diff()
    sub[f"d_{y_var}"] = sub.groupby("code")[y_var].diff()
    fd = sub.dropna(subset=[f"d_{x_var}", f"d_{y_var}"])
    if len(fd) < 30:
        return None

    X = sm.add_constant(fd[[f"d_{x_var}"]])
    y = fd[f"d_{y_var}"]
    model = sm.OLS(y, X).fit(cov_type="HC1")
    coef = model.params[f"d_{x_var}"]
    se = model.bse[f"d_{x_var}"]
    p = model.pvalues[f"d_{x_var}"]
    ci = model.conf_int().loc[f"d_{x_var}"].tolist()
    return {
        "coef": round(float(coef), 6),
        "se": round(float(se), 6),
        "p": float(f"{p:.4e}"),
        "ci95": [round(ci[0], 6), round(ci[1], 6)],
        "n": int(model.nobs),
        "r2_adj": round(float(model.rsquared_adj), 4),
    }


def substitution_test(df: pd.DataFrame) -> dict:
    """B. Replace AdProxy with mobile_cellular in TWFE & FD models."""
    results = {}
    for outcome in ["depression_prevalence", "suicide"]:
        res = {}
        # AdProxy baseline
        twfe_ad = run_twfe(df, "ad_proxy", outcome)
        fd_ad = run_fd(df, "ad_proxy", outcome)
        # Mobile substitute
        twfe_mob = run_twfe(df, "mobile_cellular", outcome)
        fd_mob = run_fd(df, "mobile_cellular", outcome)

        res["adproxy_twfe"] = twfe_ad
        res["adproxy_fd"] = fd_ad
        res["mobile_twfe"] = twfe_mob
        res["mobile_fd"] = fd_mob

        results[outcome] = res
        print(f"\n  {outcome}:")
        if twfe_ad:
            print(f"    AdProxy  TWFE: coef={twfe_ad['coef']:.6f}, p={twfe_ad['p']:.4e}")
        if twfe_mob:
            print(f"    Mobile   TWFE: coef={twfe_mob['coef']:.6f}, p={twfe_mob['p']:.4e}")
        if fd_ad:
            print(f"    AdProxy  FD:   coef={fd_ad['coef']:.6f}, p={fd_ad['p']:.4e}")
        if fd_mob:
            print(f"    Mobile   FD:   coef={fd_mob['coef']:.6f}, p={fd_mob['p']:.4e}")

    return results


def horse_race(df: pd.DataFrame) -> dict:
    """C. Both AdProxy and mobile_cellular in one TWFE model."""
    import statsmodels.api as sm

    results = {}
    for outcome in ["depression_prevalence", "suicide"]:
        sub = df.dropna(subset=["ad_proxy", "mobile_cellular", outcome]).copy()
        if len(sub) < 50:
            results[outcome] = None
            continue

        entity_dummies = pd.get_dummies(sub["code"], prefix="e", drop_first=True).astype(float)
        year_dummies = pd.get_dummies(sub["year"], prefix="y", drop_first=True).astype(float)
        X = pd.concat([sub[["ad_proxy", "mobile_cellular"]].reset_index(drop=True),
                        entity_dummies.reset_index(drop=True),
                        year_dummies.reset_index(drop=True)], axis=1)
        X = sm.add_constant(X)
        y = sub[outcome].reset_index(drop=True)

        model = sm.OLS(y, X).fit(cov_type="HC1")

        res = {}
        for var in ["ad_proxy", "mobile_cellular"]:
            coef = model.params[var]
            se = model.bse[var]
            p = model.pvalues[var]
            ci = model.conf_int().loc[var].tolist()
            res[var] = {
                "coef": round(float(coef), 6),
                "se": round(float(se), 6),
                "p": float(f"{p:.4e}"),
                "ci95": [round(ci[0], 6), round(ci[1], 6)],
            }
        res["n"] = int(model.nobs)
        res["r2_adj"] = round(float(model.rsquared_adj), 4)
        results[outcome] = res

        print(f"\n  Horse race — {outcome}:")
        print(f"    AdProxy:  coef={res['ad_proxy']['coef']:.6f}, p={res['ad_proxy']['p']:.4e}")
        print(f"    Mobile:   coef={res['mobile_cellular']['coef']:.6f}, p={res['mobile_cellular']['p']:.4e}")

    return results


def factor_analysis(df: pd.DataFrame) -> dict | None:
    """D. Factor analysis on AdProxy & mobile_cellular."""
    from sklearn.decomposition import FactorAnalysis

    sub = df.dropna(subset=["ad_proxy", "mobile_cellular"]).copy()
    if len(sub) < 50:
        return None

    X = sub[["ad_proxy", "mobile_cellular"]].values
    # Standardize
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)

    fa = FactorAnalysis(n_components=1, random_state=42)
    fa.fit(X_std)
    loadings = fa.components_[0]
    explained = 1 - fa.noise_variance_

    result = {
        "loading_ad_proxy": round(float(loadings[0]), 4),
        "loading_mobile_cellular": round(float(loadings[1]), 4),
        "communality_ad_proxy": round(float(explained[0]), 4),
        "communality_mobile_cellular": round(float(explained[1]), 4),
        "n": len(sub),
    }
    print(f"\n  Factor loadings: AdProxy={loadings[0]:.3f}, Mobile={loadings[1]:.3f}")
    print(f"  Communalities:   AdProxy={explained[0]:.3f}, Mobile={explained[1]:.3f}")
    return result


# ── Step 4: Figures ───────────────────────────────────────────────

def make_figure(df: pd.DataFrame, sub_results: dict, horse_results: dict):
    """Two-panel publication figure."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # ── Panel (a): Scatter ──
    ax = axes[0]
    sub = df.dropna(subset=["ad_proxy", "mobile_cellular"]).copy()
    ax.scatter(sub["ad_proxy"], sub["mobile_cellular"],
               alpha=0.15, s=8, color="#2c7bb6", edgecolors="none")

    # Regression line
    mask = np.isfinite(sub["ad_proxy"]) & np.isfinite(sub["mobile_cellular"])
    slope, intercept, r, p, _ = stats.linregress(sub.loc[mask, "ad_proxy"],
                                                   sub.loc[mask, "mobile_cellular"])
    x_range = np.linspace(sub["ad_proxy"].min(), sub["ad_proxy"].max(), 100)
    ax.plot(x_range, intercept + slope * x_range, color="#d7191c", linewidth=1.5)

    rho, p_rho = stats.spearmanr(sub["ad_proxy"], sub["mobile_cellular"])
    ax.text(0.05, 0.95, f"Spearman $\\rho$ = {rho:.2f}\n$n$ = {len(sub):,}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_xlabel("AdProxy (Internet % $\\times$ GDP pc / 1000)", fontsize=9)
    ax.set_ylabel("Mobile cellular subs. per 100 inh.", fontsize=9)
    ax.set_title("(a) AdProxy vs Mobile Cellular", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)

    # ── Panel (b): Coefficient comparison ──
    ax = axes[1]
    # Collect coefficients for depression_prevalence
    outcome = "depression_prevalence"
    labels = []
    coefs = []
    errs = []
    colors_bar = []

    if sub_results.get(outcome, {}).get("adproxy_twfe"):
        r_ = sub_results[outcome]["adproxy_twfe"]
        labels.append("AdProxy\n(alone)")
        coefs.append(r_["coef"])
        errs.append(r_["se"] * 1.96)
        colors_bar.append("#2c7bb6")

    if sub_results.get(outcome, {}).get("mobile_twfe"):
        r_ = sub_results[outcome]["mobile_twfe"]
        labels.append("Mobile\n(alone)")
        coefs.append(r_["coef"])
        errs.append(r_["se"] * 1.96)
        colors_bar.append("#abd9e9")

    if horse_results.get(outcome) and horse_results[outcome].get("ad_proxy"):
        r_ = horse_results[outcome]["ad_proxy"]
        labels.append("AdProxy\n(horse race)")
        coefs.append(r_["coef"])
        errs.append(r_["se"] * 1.96)
        colors_bar.append("#d7191c")

    if horse_results.get(outcome) and horse_results[outcome].get("mobile_cellular"):
        r_ = horse_results[outcome]["mobile_cellular"]
        labels.append("Mobile\n(horse race)")
        coefs.append(r_["coef"])
        errs.append(r_["se"] * 1.96)
        colors_bar.append("#fdae61")

    x_pos = np.arange(len(labels))
    ax.bar(x_pos, coefs, yerr=errs, color=colors_bar, edgecolor="black",
           linewidth=0.5, capsize=4, width=0.6, zorder=3)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", zorder=1)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("TWFE coefficient (depression prev.)", fontsize=9)
    ax.set_title("(b) Coefficient Comparison", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)

    plt.tight_layout()

    for out_dir in [FIG_EN, FIG_JA]:
        path = os.path.join(out_dir, "itu_convergent.pdf")
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"  Saved figure: {path}")

    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("ITU Mobile Broadband — Convergent Validity Analysis")
    print("=" * 60)

    # Load & merge
    print("\n[1] Loading data ...")
    df = load_and_merge()

    # A. Convergent validity
    print("\n[2] Convergent validity (correlation) ...")
    conv = convergent_validity(df)

    # B. Substitution test
    print("\n[3] Substitution test (TWFE & FD) ...")
    sub_res = substitution_test(df)

    # C. Horse race
    print("\n[4] Horse race (both in model) ...")
    horse_res = horse_race(df)

    # D. Factor analysis
    print("\n[5] Factor analysis ...")
    try:
        fa_res = factor_analysis(df)
    except Exception as e:
        print(f"  Factor analysis skipped: {e}")
        fa_res = None

    # Compile results
    all_results = {
        "analysis": "ITU Mobile Broadband Convergent Validity",
        "indicator": "IT.CEL.SETS.P2 (mobile cellular subs per 100 inh.)",
        "source": "World Bank Open Data API",
        "convergent_validity": conv,
        "substitution_test": sub_res,
        "horse_race": horse_res,
        "factor_analysis": fa_res,
    }

    out_path = os.path.join(RESULTS_DIR, "itu_convergent_validity.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved: {out_path}")

    # Figure
    print("\n[6] Generating figure ...")
    make_figure(df, sub_res, horse_res)

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
