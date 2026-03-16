#!/usr/bin/env python3
"""
Parse R CCE/IFE estimator output log into structured JSON.

Reads: results/r_cce_ife_output.log
Writes: results/cce_ife_comparison.json

Format:
{
  outcome: {
    estimator: {
      variable: {beta, se, p, significant}
    }
  }
}
"""

import re
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
LOG_FILE = ROOT / "results" / "r_cce_ife_output.log"
OUT_FILE = ROOT / "results" / "cce_ife_comparison.json"


def parse_coef_line(line: str):
    """Parse a coefficient line from R summary output.

    Handles formats like:
      proxy  1.0329e-04  4.6631e-06 22.1496 < 2.2e-16 ***
      proxy  3.1178e-05  2.8449e-06  10.959 < 2.2e-16 ***
      gdp   -4.6409e-06  1.0098e-06 -4.5959 4.421e-06 ***
    Also handles coeftest output:
      proxy  3.1178e-05  5.1303e-06  6.0774 1.319e-09 ***
    """
    line = line.strip()
    if not line or line.startswith("---") or line.startswith("Signif"):
        return None

    # Match variable name followed by numeric values
    # Variable names may contain underscores and dots
    m = re.match(
        r'^(\S+)\s+'           # variable name
        r'([+-]?\d+\.\d+e[+-]\d+)\s+'  # estimate
        r'([+-]?\d+\.\d+e[+-]\d+)\s+'  # std error
        r'([+-]?\d+\.\d+)\s+'          # t-value
        r'(.+)$',                       # p-value + signif codes
        line
    )
    if not m:
        return None

    var_name = m.group(1)
    beta = float(m.group(2))
    se = float(m.group(3))
    p_str = m.group(5).strip()

    # Parse p-value: handle "< 2.2e-16" and "< 2e-16" forms
    p_match = re.match(r'<\s*(\S+)', p_str)
    if p_match:
        p = float(p_match.group(1))
    else:
        p_num = re.match(r'([+-]?\d+\.?\d*(?:e[+-]?\d+)?)', p_str)
        if p_num:
            p = float(p_num.group(1))
        else:
            p = None

    sig = "***" in p_str or "**" in p_str or ("*" in p_str and "." not in p_str.replace("***","").replace("**","").replace("*",""))
    # Simpler: check significance markers
    stars = p_str.rstrip()
    if "***" in stars:
        sig_level = "***"
    elif "**" in stars:
        sig_level = "**"
    elif stars.endswith("*") or " * " in stars or "\t*" in stars:
        sig_level = "*"
    elif stars.endswith("."):
        sig_level = "."
    else:
        sig_level = ""

    return {
        "variable": var_name,
        "beta": beta,
        "se": se,
        "p": p,
        "significant": p is not None and p < 0.05,
        "stars": sig_level,
    }


def parse_log(log_text: str) -> dict:
    """Parse the full R log into structured results."""
    results = {}
    lines = log_text.split("\n")
    i = 0

    current_outcome = None
    current_estimator = None

    while i < len(lines):
        line = lines[i]

        # Detect outcome block
        outcome_m = re.match(r'\s*OUTCOME:\s+(\S+)', line)
        if outcome_m:
            current_outcome = outcome_m.group(1).lower()
            if current_outcome not in results:
                results[current_outcome] = {}
            i += 1
            continue

        # Detect estimator sections
        if "--- 2a. Entity FE" in line:
            current_estimator = "FE"
        elif "--- 2b. Two-Way FE" in line:
            current_estimator = "TWFE"
        elif "TWFE with Driscoll-Kraay SE:" in line:
            current_estimator = "DK-SE"
        elif "--- 4. Pesaran CCE" in line:
            current_estimator = "CCE"
        elif "--- 5. Bai IFE" in line:
            current_estimator = "IFE"

        # Parse coefficient lines when we have context
        if current_outcome and current_estimator:
            parsed = parse_coef_line(line)
            if parsed:
                var_name = parsed["variable"]
                # Only capture substantive variables (proxy, gdp), not factor/bar vars for main table
                # But store all for completeness
                if current_outcome not in results:
                    results[current_outcome] = {}
                if current_estimator not in results[current_outcome]:
                    results[current_outcome][current_estimator] = {}

                results[current_outcome][current_estimator][var_name] = {
                    "beta": parsed["beta"],
                    "se": parsed["se"],
                    "p": parsed["p"],
                    "significant": parsed["significant"],
                    "stars": parsed["stars"],
                }

        i += 1

    return results


def add_meta(results: dict) -> dict:
    """Add metadata and summary."""
    # Build a summary of sign consistency
    summary = {}
    for outcome, estimators in results.items():
        summary[outcome] = {}
        # Collect signs for proxy and gdp across estimators
        for var in ["proxy", "gdp"]:
            signs = []
            betas = []
            for est_name, est_vars in estimators.items():
                if var in est_vars:
                    b = est_vars[var]["beta"]
                    signs.append("+" if b > 0 else "-")
                    betas.append(b)
            if signs:
                consistent = len(set(signs)) == 1
                summary[outcome][var] = {
                    "sign_consistent": consistent,
                    "direction": signs[0] if consistent else "mixed",
                    "beta_range": [min(betas), max(betas)],
                    "all_significant": all(
                        est_vars.get(var, {}).get("significant", False)
                        for est_vars in estimators.values()
                    ),
                }

    return {
        "description": "CCE/IFE estimator comparison (R plm/phtt)",
        "estimators": {
            "FE": "Entity fixed effects (within)",
            "TWFE": "Two-way fixed effects (entity + time)",
            "DK-SE": "TWFE with Driscoll-Kraay standard errors (vcovSCC)",
            "CCE": "Pesaran Common Correlated Effects (manual, cross-sectional averages)",
            "IFE": "Interactive Fixed Effects (PCA-augmented proxy)",
        },
        "results": results,
        "robustness_summary": summary,
    }


def main():
    if not LOG_FILE.exists():
        print(f"ERROR: Log file not found: {LOG_FILE}", file=sys.stderr)
        sys.exit(1)

    log_text = LOG_FILE.read_text(encoding="utf-8")
    results = parse_log(log_text)

    if not results:
        print("ERROR: No results parsed from log file.", file=sys.stderr)
        sys.exit(1)

    output = add_meta(results)

    OUT_FILE.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote: {OUT_FILE}")
    print(f"Outcomes: {list(results.keys())}")
    for outcome, estimators in results.items():
        print(f"\n  {outcome.upper()}:")
        for est, vars_dict in estimators.items():
            core_vars = {k: v for k, v in vars_dict.items() if k in ("proxy", "gdp")}
            parts = []
            for k, v in core_vars.items():
                b = v["beta"]
                p = v["p"]
                parts.append(f"{k}={b:.6e} (p={p:.2e})")
            print(f"    {est}: {', '.join(parts)}")


if __name__ == "__main__":
    main()
