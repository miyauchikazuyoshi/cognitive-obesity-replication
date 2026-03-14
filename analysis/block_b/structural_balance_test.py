#!/usr/bin/env python3
"""
Structural Balance Test: L = α₁·I − α₂·C
==========================================
Direct estimation of the cognitive obesity balance equation using
the 177-country macro panel (1990-2023).

The core theoretical prediction is:
  Depression = α₁·Internet − α₂·Education + controls + FE
where α₁ > 0 (information exposure increases depression)
and   α₂ < 0 (education/processing capacity decreases depression)

Models:
  1. Baseline:    Depression ~ Internet + Country FE
  2. Additive:    Depression ~ Internet + Education + Country FE
  3. TWFE:        Depression ~ Internet + Education + Country FE + Year FE
  4. Controls:    Depression ~ Internet + Education + log(GDP) + Country FE + Year FE
  5. Interaction: Depression ~ Internet + Education + Internet×Education + Country FE + Year FE

Additional analyses:
  - Ratio model (R = I/E) vs additive for AIC comparison
  - Education-tercile subgroups
  - High-internet subsample (>30%)
  - Suicide replication
  - LaTeX table output

Data: panel_merged.csv (Block A/B pipeline)

Outputs:
  - results/structural_balance_test.json
  - results/figures/fig_structural_balance_coefficients.png
  - results/figures/fig_education_moderation.png
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

from scipy import stats

# ============================================================
# Configuration
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "macro")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def load_panel() -> pd.DataFrame:
    candidates = ["panel_merged.csv", "panel_with_inactivity.csv", "macro_panel.csv"]
    for fname in candidates:
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"Loaded: {path} ({len(df):,} rows)")
            return df
    print(f"ERROR: No panel CSV found in {DATA_DIR}")
    sys.exit(1)


# ============================================================
# 1. Load and prepare data
# ============================================================
panel = load_panel()

# Ensure columns exist
required = ["country", "year", "internet", "education", "depression_prevalence"]
missing = [c for c in required if c not in panel.columns]
if missing:
    print(f"ERROR: Missing columns: {missing}")
    print(f"Available: {list(panel.columns)}")
    sys.exit(1)

# Build analysis sample
sub = panel.dropna(subset=["depression_prevalence", "internet", "education"]).copy()
sub["log_gdp"] = np.log(sub["gdp"].clip(lower=1)) if "gdp" in sub.columns else np.nan

# Interaction term
sub["internet_x_education"] = sub["internet"] * sub["education"]

# Ratio (for comparison)
sub["ratio"] = sub["internet"] / sub["education"].clip(lower=0.1)

print(f"\n{'='*70}")
print(f"STRUCTURAL BALANCE TEST: L = α₁·I − α₂·C")
print(f"{'='*70}")
print(f"Panel: N={len(sub):,}, Countries={sub['country'].nunique()}, "
      f"Years={sub['year'].min()}-{sub['year'].max()}")
print(f"Depression prevalence: mean={sub['depression_prevalence'].mean():.3f}, "
      f"sd={sub['depression_prevalence'].std():.3f}")
print(f"Internet (%): mean={sub['internet'].mean():.1f}, sd={sub['internet'].std():.1f}")
print(f"Education (years): mean={sub['education'].mean():.1f}, sd={sub['education'].std():.1f}")


# ============================================================
# 2. Regression utilities
# ============================================================
def country_demean(df, cols):
    """Within-group demeaning for Country FE."""
    out = df.copy()
    for col in cols:
        out[f"{col}_dm"] = out.groupby("country")[col].transform(lambda x: x - x.mean())
    return out


def twfe_demean(df, cols):
    """Demean for TWFE (country + year)."""
    out = df.copy()
    for col in cols:
        cm = out.groupby("country")[col].transform("mean")
        ym = out.groupby("year")[col].transform("mean")
        gm = out[col].mean()
        out[f"{col}_twfe"] = out[col] - cm - ym + gm
    return out


def ols_fit(X, y, n_fe):
    """OLS with FE-adjusted degrees of freedom."""
    from numpy.linalg import lstsq as np_lstsq
    b, _, _, _ = np_lstsq(X, y, rcond=None)
    resid = y - X @ b
    n = len(y)
    k = X.shape[1]
    df_resid = n - n_fe - k
    rss = float(np.sum(resid ** 2))
    tss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - rss / tss if tss > 0 else 0
    se = np.sqrt(rss / max(df_resid, 1))
    xtx_inv = np.linalg.inv(X.T @ X + np.eye(k) * 1e-12)
    se_b = np.sqrt(se ** 2 * np.diag(xtx_inv))
    t_vals = b / np.where(se_b > 0, se_b, 1e-12)
    p_vals = 2 * (1 - stats.t.cdf(np.abs(t_vals), max(df_resid, 1)))
    aic = n * np.log(rss / n) + 2 * (n_fe + k)
    bic = n * np.log(rss / n) + np.log(n) * (n_fe + k)
    return {
        "b": b, "se": se_b, "t": t_vals, "p": p_vals,
        "r2": r2, "rss": rss, "aic": aic, "bic": bic,
        "n": n, "k": k, "n_fe": n_fe,
    }


def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return ""


# ============================================================
# 3. Prepare demeaned variables
# ============================================================
all_vars = ["depression_prevalence", "internet", "education",
            "internet_x_education", "ratio"]
if sub["log_gdp"].notna().any():
    all_vars.append("log_gdp")

sub = country_demean(sub, all_vars)
sub = twfe_demean(sub, all_vars)

nc = sub["country"].nunique()
n = len(sub)

# ============================================================
# 4. Five-model comparison
# ============================================================
print(f"\n{'='*70}")
print("MODEL COMPARISON: Depression ~ f(Internet, Education)")
print(f"{'='*70}\n")

y_dm = sub["depression_prevalence_dm"].values
y_twfe = sub["depression_prevalence_twfe"].values

model_specs = {}

# Model 1: Internet only (Country FE)
X1 = sub["internet_dm"].values.reshape(-1, 1)
m1 = ols_fit(X1, y_dm, nc)
model_specs["M1: Internet only (FE)"] = (m1, ["Internet"], "FE")

# Model 2: Additive (Country FE)
X2 = np.column_stack([sub["internet_dm"].values, sub["education_dm"].values])
m2 = ols_fit(X2, y_dm, nc)
model_specs["M2: Additive (FE)"] = (m2, ["Internet (α₁)", "Education (α₂)"], "FE")

# Model 3: Additive (TWFE)
X3 = np.column_stack([sub["internet_twfe"].values, sub["education_twfe"].values])
ny = sub["year"].nunique()
m3 = ols_fit(X3, y_twfe, nc + ny)
model_specs["M3: Additive (TWFE)"] = (m3, ["Internet (α₁)", "Education (α₂)"], "TWFE")

# Model 4: Controls (TWFE + log GDP)
if "log_gdp_twfe" in sub.columns:
    sub_m4 = sub.dropna(subset=["log_gdp_twfe"])
    X4 = np.column_stack([sub_m4["internet_twfe"].values, sub_m4["education_twfe"].values,
                          sub_m4["log_gdp_twfe"].values])
    y4 = sub_m4["depression_prevalence_twfe"].values
    m4 = ols_fit(X4, y4, nc + ny)
    model_specs["M4: Controls (TWFE+GDP)"] = (m4, ["Internet (α₁)", "Education (α₂)", "log(GDP)"], "TWFE+GDP")

# Model 5: Interaction (TWFE)
X5 = np.column_stack([sub["internet_twfe"].values, sub["education_twfe"].values,
                       sub["internet_x_education_twfe"].values])
m5 = ols_fit(X5, y_twfe, nc + ny)
model_specs["M5: Interaction (TWFE)"] = (m5, ["Internet (α₁)", "Education (α₂)", "Internet×Education (α₃)"], "TWFE+Int")

# Ratio model for AIC comparison
X_ratio = sub["ratio_twfe"].values.reshape(-1, 1)
m_ratio = ols_fit(X_ratio, y_twfe, nc + ny)
model_specs["M_R: Ratio R=I/E (TWFE)"] = (m_ratio, ["Ratio (I/E)"], "TWFE")

# Print results
json_models = {}
for name, (m, var_names, fe_type) in model_specs.items():
    print(f"\n{name} (N={m['n']:,}, R²={m['r2']:.4f}, AIC={m['aic']:.1f})")
    print("-" * 55)
    model_json = {
        "n": int(m["n"]), "r2": float(m["r2"]),
        "aic": float(m["aic"]), "bic": float(m["bic"]),
        "fe_type": fe_type, "vars": {}
    }
    for i, vname in enumerate(var_names):
        stars = sig_stars(m["p"][i])
        print(f"  {vname:<30} β={m['b'][i]:>10.6f}  SE={m['se'][i]:.6f}  "
              f"t={m['t'][i]:>7.2f}  p={m['p'][i]:.4f} {stars}")
        model_json["vars"][vname] = {
            "beta": float(m["b"][i]), "se": float(m["se"][i]),
            "t": float(m["t"][i]), "p": float(m["p"][i]),
        }
    json_models[name] = model_json

# AIC comparison table
print(f"\n{'='*70}")
print("AIC COMPARISON")
print(f"{'='*70}")
all_aics = {k: v["aic"] for k, v in json_models.items()}
best_aic = min(all_aics.values())
print(f"\n{'Model':<35} {'AIC':>12} {'ΔAIC':>8}  {'R²':>8}")
print("-" * 70)
for name in model_specs:
    m_json = json_models[name]
    delta = m_json["aic"] - best_aic
    marker = " ← BEST" if delta < 0.1 else ""
    print(f"{name:<35} {m_json['aic']:>12.1f} {delta:>8.1f}  {m_json['r2']:>8.4f}{marker}")

# ============================================================
# 5. Hypothesis tests
# ============================================================
print(f"\n{'='*70}")
print("HYPOTHESIS TESTS (from M3: Additive TWFE)")
print(f"{'='*70}")

m3_r = model_specs["M3: Additive (TWFE)"][0]
alpha1 = m3_r["b"][0]
alpha2 = m3_r["b"][1]
t1 = m3_r["t"][0]
t2 = m3_r["t"][1]
p1 = m3_r["p"][0]
p2 = m3_r["p"][1]

h1_pass = alpha1 > 0 and p1 < 0.05
h2_pass = alpha2 < 0 and p2 < 0.05
print(f"\nH1: α₁ > 0 (Internet ↑ → Depression ↑)")
print(f"    α₁ = {alpha1:.6f}, t = {t1:.2f}, p = {p1:.4f}")
print(f"    Result: {'SUPPORTED ✓' if h1_pass else 'NOT SUPPORTED ✗'}")

print(f"\nH2: α₂ < 0 (Education ↑ → Depression ↓)")
print(f"    α₂ = {alpha2:.6f}, t = {t2:.2f}, p = {p2:.4f}")
print(f"    Result: {'SUPPORTED ✓' if h2_pass else 'NOT SUPPORTED ✗'}")

if "M5: Interaction (TWFE)" in model_specs:
    m5_r = model_specs["M5: Interaction (TWFE)"][0]
    alpha3 = m5_r["b"][2]
    t3 = m5_r["t"][2]
    p3 = m5_r["p"][2]
    h3_pass = alpha3 < 0 and p3 < 0.05
    print(f"\nH3: α₃ < 0 (Education buffers Internet effect)")
    print(f"    α₃ = {alpha3:.6f}, t = {t3:.2f}, p = {p3:.4f}")
    print(f"    Result: {'SUPPORTED ✓' if h3_pass else 'NOT SUPPORTED ✗'}")

# H4: Additive > Ratio (AIC)
aic_add = json_models["M3: Additive (TWFE)"]["aic"]
aic_rat = json_models["M_R: Ratio R=I/E (TWFE)"]["aic"]
h4_pass = aic_add < aic_rat
print(f"\nH4: Additive model preferred over Ratio (AIC)")
print(f"    AIC(Additive) = {aic_add:.1f}, AIC(Ratio) = {aic_rat:.1f}")
print(f"    ΔAIC = {aic_rat - aic_add:.1f}")
print(f"    Result: {'SUPPORTED ✓' if h4_pass else 'NOT SUPPORTED ✗'}")

hypothesis_results = {
    "H1_alpha1_positive": {"alpha1": float(alpha1), "t": float(t1), "p": float(p1), "supported": h1_pass},
    "H2_alpha2_negative": {"alpha2": float(alpha2), "t": float(t2), "p": float(p2), "supported": h2_pass},
    "H4_additive_preferred": {"aic_additive": float(aic_add), "aic_ratio": float(aic_rat),
                              "delta_aic": float(aic_rat - aic_add), "supported": h4_pass},
}
if "M5: Interaction (TWFE)" in model_specs:
    hypothesis_results["H3_interaction_negative"] = {
        "alpha3": float(alpha3), "t": float(t3), "p": float(p3), "supported": h3_pass
    }

# ============================================================
# 6. Education-tercile subgroup analysis
# ============================================================
print(f"\n{'='*70}")
print("EDUCATION TERCILE SUBGROUP ANALYSIS")
print(f"{'='*70}")

country_means = sub.groupby("country").agg(
    edu_mean=("education", "mean"),
    internet_mean=("internet", "mean"),
    dep_mean=("depression_prevalence", "mean")
).reset_index()

edu_t1 = country_means["edu_mean"].quantile(0.33)
edu_t2 = country_means["edu_mean"].quantile(0.67)

tercile_labels = {
    "Low education": country_means[country_means["edu_mean"] <= edu_t1]["country"].tolist(),
    "Medium education": country_means[(country_means["edu_mean"] > edu_t1) &
                                       (country_means["edu_mean"] <= edu_t2)]["country"].tolist(),
    "High education": country_means[country_means["edu_mean"] > edu_t2]["country"].tolist(),
}

tercile_results = {}
for label, countries in tercile_labels.items():
    s = sub[sub["country"].isin(countries)].copy()
    if len(s) < 50:
        continue
    s = country_demean(s, ["depression_prevalence", "internet"])
    X_t = s["internet_dm"].values.reshape(-1, 1)
    y_t = s["depression_prevalence_dm"].values
    nc_t = s["country"].nunique()
    m_t = ols_fit(X_t, y_t, nc_t)
    stars = sig_stars(m_t["p"][0])
    print(f"\n{label} (N={len(s):,}, {nc_t} countries, mean edu={country_means[country_means['country'].isin(countries)]['edu_mean'].mean():.1f} yrs):")
    print(f"  Internet → Depression: β={m_t['b'][0]:.6f}, t={m_t['t'][0]:.2f}, p={m_t['p'][0]:.4f} {stars}")
    tercile_results[label] = {
        "n": int(len(s)), "n_countries": nc_t,
        "beta_internet": float(m_t["b"][0]), "t": float(m_t["t"][0]),
        "p": float(m_t["p"][0]), "r2": float(m_t["r2"]),
    }

# ============================================================
# 7. High-internet subsample (>30%)
# ============================================================
print(f"\n{'='*70}")
print("HIGH-INTERNET SUBSAMPLE (Internet > 30%)")
print(f"{'='*70}")

hi = sub[sub["internet"] > 30].copy()
hi = country_demean(hi, ["depression_prevalence", "internet", "education"])
X_hi = np.column_stack([hi["internet_dm"].values, hi["education_dm"].values])
y_hi = hi["depression_prevalence_dm"].values
nc_hi = hi["country"].nunique()
m_hi = ols_fit(X_hi, y_hi, nc_hi)
print(f"\nN={len(hi):,}, Countries={nc_hi}")
for i, v in enumerate(["Internet (α₁)", "Education (α₂)"]):
    stars = sig_stars(m_hi["p"][i])
    print(f"  {v:<25} β={m_hi['b'][i]:.6f}, t={m_hi['t'][i]:.2f}, p={m_hi['p'][i]:.4f} {stars}")

high_internet_results = {
    "n": int(len(hi)), "n_countries": nc_hi,
    "alpha1": {"beta": float(m_hi["b"][0]), "t": float(m_hi["t"][0]), "p": float(m_hi["p"][0])},
    "alpha2": {"beta": float(m_hi["b"][1]), "t": float(m_hi["t"][1]), "p": float(m_hi["p"][1])},
}

# ============================================================
# 8. Suicide replication
# ============================================================
suicide_results = None
if "suicide" in sub.columns and sub["suicide"].notna().sum() > 500:
    print(f"\n{'='*70}")
    print("SUICIDE REPLICATION")
    print(f"{'='*70}")

    sub_s = sub.dropna(subset=["suicide"]).copy()
    sub_s = country_demean(sub_s, ["suicide", "internet", "education"])
    sub_s = twfe_demean(sub_s, ["suicide", "internet", "education"])

    X_s = np.column_stack([sub_s["internet_twfe"].values, sub_s["education_twfe"].values])
    y_s = sub_s["suicide_twfe"].values
    nc_s = sub_s["country"].nunique()
    ny_s = sub_s["year"].nunique()
    m_s = ols_fit(X_s, y_s, nc_s + ny_s)

    print(f"N={len(sub_s):,}, Countries={nc_s}")
    for i, v in enumerate(["Internet (α₁)", "Education (α₂)"]):
        stars = sig_stars(m_s["p"][i])
        print(f"  {v:<25} β={m_s['b'][i]:.6f}, t={m_s['t'][i]:.2f}, p={m_s['p'][i]:.4f} {stars}")

    suicide_results = {
        "n": int(len(sub_s)), "n_countries": nc_s,
        "alpha1": {"beta": float(m_s["b"][0]), "t": float(m_s["t"][0]), "p": float(m_s["p"][0])},
        "alpha2": {"beta": float(m_s["b"][1]), "t": float(m_s["t"][1]), "p": float(m_s["p"][1])},
    }

# ============================================================
# 9. LaTeX table output
# ============================================================
print(f"\n{'='*70}")
print("LaTeX TABLE")
print(f"{'='*70}\n")

latex_lines = [
    r"\begin{table}[htbp]",
    r"\centering",
    r"\caption{Structural balance test: $L = \alpha_1 \cdot I - \alpha_2 \cdot C$}",
    r"\label{tab:structural-balance}",
    r"\small",
    r"\begin{tabular}{lccccc}",
    r"\toprule",
    r"& M1 & M2 & M3 & M4 & M5 \\",
    r"& Internet & Additive & Additive & +Controls & +Interaction \\",
    r"& FE & FE & TWFE & TWFE & TWFE \\",
    r"\midrule",
]

model_keys = ["M1: Internet only (FE)", "M2: Additive (FE)", "M3: Additive (TWFE)",
              "M4: Controls (TWFE+GDP)", "M5: Interaction (TWFE)"]

# Internet row
row_i = "Internet ($\\alpha_1$)"
for mk in model_keys:
    if mk in json_models:
        mv = json_models[mk]["vars"]
        key = [k for k in mv if "Internet" in k and "×" not in k][0] if any("Internet" in k and "×" not in k for k in mv) else None
        if key:
            b = mv[key]["beta"]
            t = mv[key]["t"]
            s = sig_stars(mv[key]["p"])
            row_i += f" & {b:.4f}{s}"
        else:
            row_i += " & "
    else:
        row_i += " & "
row_i += r" \\"
latex_lines.append(row_i)

# t-stats for internet
row_ti = ""
for mk in model_keys:
    if mk in json_models:
        mv = json_models[mk]["vars"]
        key = [k for k in mv if "Internet" in k and "×" not in k][0] if any("Internet" in k and "×" not in k for k in mv) else None
        if key:
            row_ti += f" & ({mv[key]['t']:.2f})"
        else:
            row_ti += " & "
    else:
        row_ti += " & "
row_ti += r" \\"
latex_lines.append(row_ti)

# Education row
row_e = "Education ($\\alpha_2$)"
for mk in model_keys:
    if mk in json_models:
        mv = json_models[mk]["vars"]
        key = [k for k in mv if "Education" in k and "×" not in k][0] if any("Education" in k and "×" not in k for k in mv) else None
        if key:
            b = mv[key]["beta"]
            s = sig_stars(mv[key]["p"])
            row_e += f" & {b:.4f}{s}"
        else:
            row_e += " & "
    else:
        row_e += " & "
row_e += r" \\"
latex_lines.append(row_e)

# t-stats for education
row_te = ""
for mk in model_keys:
    if mk in json_models:
        mv = json_models[mk]["vars"]
        key = [k for k in mv if "Education" in k and "×" not in k][0] if any("Education" in k and "×" not in k for k in mv) else None
        if key:
            row_te += f" & ({mv[key]['t']:.2f})"
        else:
            row_te += " & "
    else:
        row_te += " & "
row_te += r" \\"
latex_lines.append(row_te)

# log(GDP) row
row_g = "log(GDP)"
for mk in model_keys:
    if mk in json_models:
        mv = json_models[mk]["vars"]
        if "log(GDP)" in mv:
            b = mv["log(GDP)"]["beta"]
            s = sig_stars(mv["log(GDP)"]["p"])
            row_g += f" & {b:.4f}{s}"
        else:
            row_g += " & "
    else:
        row_g += " & "
row_g += r" \\"
latex_lines.append(row_g)

# Interaction row
row_int = "Internet $\\times$ Education ($\\alpha_3$)"
for mk in model_keys:
    if mk in json_models:
        mv = json_models[mk]["vars"]
        key = [k for k in mv if "×" in k][0] if any("×" in k for k in mv) else None
        if key:
            b = mv[key]["beta"]
            s = sig_stars(mv[key]["p"])
            row_int += f" & {b:.6f}{s}"
        else:
            row_int += " & "
    else:
        row_int += " & "
row_int += r" \\"
latex_lines.append(row_int)

latex_lines.append(r"\midrule")

# N, R², AIC
row_n = "N"
row_r2 = "$R^2$"
row_aic = "AIC"
for mk in model_keys:
    if mk in json_models:
        mj = json_models[mk]
        row_n += f" & {mj['n']:,}"
        row_r2 += f" & {mj['r2']:.4f}"
        row_aic += f" & {mj['aic']:.0f}"
    else:
        row_n += " & "
        row_r2 += " & "
        row_aic += " & "

latex_lines.append(row_n + r" \\")
latex_lines.append(row_r2 + r" \\")
latex_lines.append(row_aic + r" \\")
latex_lines.extend([
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
])

for line in latex_lines:
    print(line)

# ============================================================
# 10. Figures
# ============================================================

# Figure 1: Coefficient comparison across models
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: α₁ (Internet) across models
ax = axes[0]
model_labels = []
betas = []
cis_lo = []
cis_hi = []
for name in ["M1: Internet only (FE)", "M2: Additive (FE)", "M3: Additive (TWFE)",
             "M4: Controls (TWFE+GDP)", "M5: Interaction (TWFE)"]:
    if name not in json_models:
        continue
    mv = json_models[name]["vars"]
    key = [k for k in mv if "Internet" in k and "×" not in k]
    if key:
        v = mv[key[0]]
        model_labels.append(name.split(":")[0])
        betas.append(v["beta"])
        cis_lo.append(v["beta"] - 1.96 * v["se"])
        cis_hi.append(v["beta"] + 1.96 * v["se"])

y_pos = range(len(model_labels))
ax.barh(y_pos, betas, color="#ef4444", alpha=0.7, edgecolor="white")
ax.errorbar(betas, y_pos, xerr=[np.array(betas) - np.array(cis_lo),
            np.array(cis_hi) - np.array(betas)],
            fmt="none", color="black", capsize=3)
ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(model_labels)
ax.set_xlabel("Coefficient (β)")
ax.set_title("α₁: Internet → Depression", fontweight="bold")

# Panel B: α₂ (Education) across models
ax = axes[1]
model_labels2 = []
betas2 = []
cis_lo2 = []
cis_hi2 = []
for name in ["M2: Additive (FE)", "M3: Additive (TWFE)",
             "M4: Controls (TWFE+GDP)", "M5: Interaction (TWFE)"]:
    if name not in json_models:
        continue
    mv = json_models[name]["vars"]
    key = [k for k in mv if "Education" in k and "×" not in k]
    if key:
        v = mv[key[0]]
        model_labels2.append(name.split(":")[0])
        betas2.append(v["beta"])
        cis_lo2.append(v["beta"] - 1.96 * v["se"])
        cis_hi2.append(v["beta"] + 1.96 * v["se"])

y_pos2 = range(len(model_labels2))
ax.barh(y_pos2, betas2, color="#3b82f6", alpha=0.7, edgecolor="white")
ax.errorbar(betas2, y_pos2, xerr=[np.array(betas2) - np.array(cis_lo2),
            np.array(cis_hi2) - np.array(betas2)],
            fmt="none", color="black", capsize=3)
ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
ax.set_yticks(y_pos2)
ax.set_yticklabels(model_labels2)
ax.set_xlabel("Coefficient (β)")
ax.set_title("α₂: Education → Depression", fontweight="bold")

plt.tight_layout()
fig1_path = os.path.join(FIG_DIR, "fig_structural_balance_coefficients.png")
plt.savefig(fig1_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {fig1_path}")

# Figure 2: Education moderation
fig, ax = plt.subplots(figsize=(8, 5))
colors = {"Low education": "#ef4444", "Medium education": "#f59e0b", "High education": "#22c55e"}
for label, r in tercile_results.items():
    ax.barh(label, r["beta_internet"], color=colors.get(label, "gray"), alpha=0.8, edgecolor="white")
    ax.text(r["beta_internet"], label,
            f'  β={r["beta_internet"]:.4f} (t={r["t"]:.1f})',
            va="center", fontsize=9)

ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("Internet → Depression coefficient (β)")
ax.set_title("Education Moderates Internet-Depression Link", fontweight="bold")
plt.tight_layout()
fig2_path = os.path.join(FIG_DIR, "fig_education_moderation.png")
plt.savefig(fig2_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {fig2_path}")

# ============================================================
# 11. Save JSON
# ============================================================
output = {
    "description": "Structural balance test: L = alpha_1 * I - alpha_2 * C",
    "panel": {"n": int(n), "n_countries": int(nc), "year_range": [int(sub["year"].min()), int(sub["year"].max())]},
    "models": json_models,
    "hypotheses": hypothesis_results,
    "education_terciles": tercile_results,
    "high_internet_subsample": high_internet_results,
    "suicide_replication": suicide_results,
}

json_path = os.path.join(RESULTS_DIR, "structural_balance_test.json")
with open(json_path, "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nSaved: {json_path}")

# ============================================================
# 12. Summary
# ============================================================
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"\nThe cognitive obesity balance equation L = α₁·I − α₂·C predicts:")
print(f"  α₁ > 0 (information exposure increases depression)")
print(f"  α₂ < 0 (education/processing capacity decreases depression)")
print(f"\nResults from TWFE estimation (M3):")
print(f"  α₁ (Internet)  = {alpha1:.6f}, t = {t1:.2f} → {'CONFIRMED' if h1_pass else 'NOT CONFIRMED'}")
print(f"  α₂ (Education) = {alpha2:.6f}, t = {t2:.2f} → {'CONFIRMED' if h2_pass else 'NOT CONFIRMED'}")
if h4_pass:
    print(f"\n  Additive model preferred over ratio model (ΔAIC = {aic_rat - aic_add:.1f})")
print(f"\n  Education moderation: {'CONFIRMED' if h3_pass else 'NOT CONFIRMED'} in M5")
