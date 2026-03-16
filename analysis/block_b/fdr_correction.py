#!/usr/bin/env python3
"""
FDR (False Discovery Rate) Multiple Testing Correction
=======================================================
Collects all p-values from panel model specifications, groups them into
hypothesis families, and applies Benjamini-Hochberg FDR correction
(with Bonferroni for comparison).

Addresses Stanford reviewer request for explicit multiple-testing strategy.
"""

import json
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(BASE, "results")

EFFECT_FILE = os.path.join(RESULTS, "effect_size_table.json")
DALYS_FILE = os.path.join(RESULTS, "dalys_triangulation.json")
PLACEBO_FILE = os.path.join(RESULTS, "placebo_test.json")
MICRO_CI_FILE = os.path.join(RESULTS, "reviewer_micro_ci.json")
ATUS_FILE = os.path.join(RESULTS, "atus_wellbeing.json")

OUT_JSON = os.path.join(RESULTS, "fdr_correction.json")
OUT_TEX = os.path.join(RESULTS, "fdr_correction.tex")


# ---------------------------------------------------------------------------
# Benjamini-Hochberg procedure (pure numpy, no statsmodels dependency)
# ---------------------------------------------------------------------------
def benjamini_hochberg(p_values):
    """Return BH-adjusted p-values (same order as input)."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([])
    order = np.argsort(p)
    ranked = np.empty(n)
    ranked[order] = np.arange(1, n + 1)
    adjusted = p * n / ranked
    # enforce monotonicity (descending through ranks)
    adjusted_sorted = adjusted[order]
    for i in range(n - 2, -1, -1):
        adjusted_sorted[i + 1] = min(adjusted_sorted[i + 1], 1.0)
        adjusted_sorted[i] = min(adjusted_sorted[i], adjusted_sorted[i + 1])
    adjusted_sorted[-1] = min(adjusted_sorted[-1], 1.0)
    result = np.empty(n)
    result[order] = adjusted_sorted
    return result


def bonferroni(p_values):
    """Return Bonferroni-adjusted p-values."""
    p = np.asarray(p_values, dtype=float)
    return np.minimum(p * len(p), 1.0)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_json(path):
    if not os.path.exists(path):
        print(f"  [WARN] File not found: {path}")
        return None
    with open(path) as f:
        return json.load(f)


effect = load_json(EFFECT_FILE)
dalys = load_json(DALYS_FILE)
placebo = load_json(PLACEBO_FILE)
micro_ci = load_json(MICRO_CI_FILE)
atus_wb = load_json(ATUS_FILE)


# ---------------------------------------------------------------------------
# Collect p-values into families
# ---------------------------------------------------------------------------
families = {}

# Helper: extract from effect_size_table.json rows
def rows_by_filter(data, level, outcome):
    return [r for r in data if r["Level"] == level and r["Outcome"] == outcome]


# --- Family 1: Macro panel – Depression prevalence ---
if effect:
    dep_rows = rows_by_filter(effect, "Macro", "Depression prevalence")
    families["macro_depression"] = {
        "label": "Macro panel: Depression prevalence",
        "tests": [
            {"spec": f"{r['Treatment']} / {r['Model']}", "p": r["p"]}
            for r in dep_rows
        ],
    }

# --- Family 2: Macro panel – Suicide rate ---
if effect:
    sui_rows = rows_by_filter(effect, "Macro", "Suicide rate")
    families["macro_suicide"] = {
        "label": "Macro panel: Suicide rate",
        "tests": [
            {"spec": f"{r['Treatment']} / {r['Model']}", "p": r["p"]}
            for r in sui_rows
        ],
    }

# --- Family 3: Macro panel – Depression DALYs ---
if effect:
    daly_rows = rows_by_filter(effect, "Macro", "Depression DALYs")
    families["macro_dalys"] = {
        "label": "Macro panel: Depression DALYs",
        "tests": [
            {"spec": f"{r['Treatment']} / {r['Model']}", "p": r["p"]}
            for r in daly_rows
        ],
    }

# --- Family 4: Placebo tests ---
if placebo:
    placebo_tests = []
    for outcome_key, outcome_data in placebo.items():
        if outcome_data.get("type") != "placebo":
            continue
        label = outcome_data.get("label", outcome_key)
        for model_key in ["twfe", "first_diff"]:
            if model_key in outcome_data:
                placebo_tests.append({
                    "spec": f"{label} / {model_key.replace('_', ' ').title()}",
                    "p": outcome_data[model_key]["pvalue"],
                })
    # Also include target outcomes from placebo_test.json as reference
    # (these overlap with Families 1-3 but the placebo file tests them
    #  under the same specification for comparability)
    families["placebo"] = {
        "label": "Placebo / falsification tests",
        "tests": placebo_tests,
    }

# --- Family 5: Micro-level (NHANES + ATUS) ---
micro_tests = []
if micro_ci:
    # NHANES: exercise coefficient
    nhanes_coeffs = micro_ci.get("nhanes", {}).get("coefficients", [])
    for c in nhanes_coeffs:
        if c["variable"] == "exercise":
            micro_tests.append({
                "spec": "NHANES: Exercise -> PHQ-9",
                "p": c["p"],
            })
    # ATUS: main coefficients (exercise_min, passive_min, active_cognitive_min)
    atus_coeffs = micro_ci.get("atus", {}).get("coefficients", [])
    for c in atus_coeffs:
        if c["variable"] in ("exercise_min", "passive_min", "active_cognitive_min"):
            micro_tests.append({
                "spec": f"ATUS: {c['variable']} -> Cantril",
                "p": c["p"],
            })

# ATUS wellbeing: key OLS tests
if atus_wb:
    ols = atus_wb.get("ols_cantril", {})
    for key_suffix, label in [
        ("exercise", "Exercise (binary)"),
        ("active_leisure", "Active leisure (binary)"),
        ("high_passive", "High passive (binary)"),
    ]:
        p_key = f"p_{key_suffix}"
        if p_key in ols:
            micro_tests.append({
                "spec": f"ATUS OLS: {label} -> Cantril",
                "p": ols[p_key],
            })

if micro_tests:
    families["micro"] = {
        "label": "Micro-level (NHANES + ATUS)",
        "tests": micro_tests,
    }


# ---------------------------------------------------------------------------
# Apply corrections
# ---------------------------------------------------------------------------
alpha = 0.05
output = {
    "method": "Benjamini-Hochberg",
    "alpha": alpha,
    "families": {},
}

for fam_key, fam in families.items():
    raw_p = [t["p"] for t in fam["tests"]]
    specs = [t["spec"] for t in fam["tests"]]

    # Handle p = 0.0 (machine-zero from floating point)
    raw_p_safe = [max(p, 1e-300) for p in raw_p]

    bh = benjamini_hochberg(raw_p_safe)
    bonf = bonferroni(raw_p_safe)

    n_sig_raw = sum(1 for p in raw_p if p < alpha)
    n_sig_bh = sum(1 for p in bh if p < alpha)
    n_sig_bonf = sum(1 for p in bonf if p < alpha)

    output["families"][fam_key] = {
        "label": fam["label"],
        "n_tests": len(raw_p),
        "specifications": specs,
        "raw_p_values": raw_p,
        "bh_adjusted_p_values": [round(float(p), 10) for p in bh],
        "bonferroni_adjusted_p_values": [round(float(p), 10) for p in bonf],
        "reject_bh_005": [bool(p < alpha) for p in bh],
        "reject_bonferroni_005": [bool(p < alpha) for p in bonf],
        "n_significant_raw": n_sig_raw,
        "n_significant_bh": n_sig_bh,
        "n_significant_bonferroni": n_sig_bonf,
    }

# Global summary
total_tests = sum(f["n_tests"] for f in output["families"].values())
total_sig_raw = sum(f["n_significant_raw"] for f in output["families"].values())
total_sig_bh = sum(f["n_significant_bh"] for f in output["families"].values())
total_sig_bonf = sum(f["n_significant_bonferroni"] for f in output["families"].values())

output["summary"] = (
    f"{total_sig_bh} of {total_tests} tests remain significant after "
    f"BH-FDR correction at α = {alpha} "
    f"({total_sig_raw} significant before correction; "
    f"{total_sig_bonf} survive Bonferroni)"
)

# ---------------------------------------------------------------------------
# Save JSON
# ---------------------------------------------------------------------------
with open(OUT_JSON, "w") as f:
    json.dump(output, f, indent=2)
print(f"Saved: {OUT_JSON}")


# ---------------------------------------------------------------------------
# Generate LaTeX table
# ---------------------------------------------------------------------------
def fmt_p(p):
    """Format a p-value for LaTeX."""
    if p < 1e-10:
        return f"$< 10^{{-10}}$"
    elif p < 0.001:
        exp = int(np.floor(np.log10(p)))
        mantissa = p / (10 ** exp)
        return f"${mantissa:.1f} \\times 10^{{{exp}}}$"
    else:
        return f"{p:.4f}"


def star(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


lines = []
lines.append(r"\begin{table}[htbp]")
lines.append(r"\centering")
lines.append(r"\caption{Multiple Testing Correction: Benjamini--Hochberg FDR}")
lines.append(r"\label{tab:fdr}")
lines.append(r"\footnotesize")
lines.append(r"\begin{tabular}{llccc}")
lines.append(r"\toprule")
lines.append(r"Family & Specification & Raw $p$ & BH $q$ & Bonferroni $p$ \\")
lines.append(r"\midrule")

for fam_key, fam_data in output["families"].items():
    label = fam_data["label"]
    n = fam_data["n_tests"]
    specs = fam_data["specifications"]
    raw = fam_data["raw_p_values"]
    bh_adj = fam_data["bh_adjusted_p_values"]
    bonf_adj = fam_data["bonferroni_adjusted_p_values"]

    for i in range(n):
        fam_col = label if i == 0 else ""
        spec_short = specs[i].replace("_", r"\_").replace("%", r"\%")
        raw_s = fmt_p(raw[i]) + star(raw[i])
        bh_s = fmt_p(bh_adj[i]) + star(bh_adj[i])
        bonf_s = fmt_p(bonf_adj[i]) + star(bonf_adj[i])
        lines.append(f"  {fam_col} & {spec_short} & {raw_s} & {bh_s} & {bonf_s} \\\\")

    lines.append(r"\addlinespace")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(
    r"\par\smallskip\noindent\textit{Note.} "
    r"$^{*}p<.05$; $^{**}p<.01$; $^{***}p<.001$. "
    r"BH $q$-values computed within each family using the "
    r"Benjamini--Hochberg step-up procedure. "
    + output["summary"].replace("%", r"\%")
    + "."
)
lines.append(r"\end{table}")

tex = "\n".join(lines)
with open(OUT_TEX, "w") as f:
    f.write(tex)
print(f"Saved: {OUT_TEX}")

# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("FDR MULTIPLE TESTING CORRECTION — SUMMARY")
print("=" * 70)
for fam_key, fam_data in output["families"].items():
    label = fam_data["label"]
    n = fam_data["n_tests"]
    n_raw = fam_data["n_significant_raw"]
    n_bh = fam_data["n_significant_bh"]
    n_bonf = fam_data["n_significant_bonferroni"]
    print(f"\n  {label} ({n} tests)")
    print(f"    Raw significant:        {n_raw}/{n}")
    print(f"    BH-FDR significant:     {n_bh}/{n}")
    print(f"    Bonferroni significant:  {n_bonf}/{n}")
    for i, spec in enumerate(fam_data["specifications"]):
        raw_p = fam_data["raw_p_values"][i]
        bh_p = fam_data["bh_adjusted_p_values"][i]
        sig = "✓" if fam_data["reject_bh_005"][i] else "✗"
        print(f"      {sig} {spec:45s}  p={raw_p:.2e}  q={bh_p:.2e}")

print(f"\n  {output['summary']}")
print("=" * 70)
