#!/bin/bash
# =============================================================
# Cognitive Obesity: Full Replication Pipeline
# =============================================================
# Usage: bash run_all.sh
#
# Prerequisites:
#   pip install -r requirements.txt
#   IHME/WHO data manually placed in data/macro/ (see data/README_data.md)
# =============================================================

set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

if [ -z "${PYTHON:-}" ]; then
  if [ -x "$ROOT_DIR/.venv/bin/python" ]; then
    PYTHON="$ROOT_DIR/.venv/bin/python"
  else
    PYTHON="python3"
  fi
fi

if [ ! -x "$PYTHON" ] && ! command -v "$PYTHON" >/dev/null 2>&1; then
  PYTHON="python"
fi
if [ ! -x "$PYTHON" ] && ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "ERROR: python not found. Install Python 3 and/or create a venv:"
  echo "  python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

echo "=============================================="
echo " Cognitive Obesity Replication Pipeline"
echo "=============================================="

# ---- Step 1: Data acquisition ----
echo ""
echo "[Step 1/5] Downloading data..."

echo "  NHANES 2017-2018..."
$PYTHON data/download_nhanes.py

echo "  ATUS 2010-2013..."
$PYTHON data/download_atus.py

echo "  World Bank WDI..."
$PYTHON data/download_macro.py

echo "  OWID (schooling years, homicide)..."
$PYTHON data/download_owid.py || echo "  WARN: download_owid.py failed (continuing)."

echo "  WHO GHO (inactivity, suicide)..."
$PYTHON data/download_who_gho.py || echo "  WARN: download_who_gho.py failed (continuing)."

echo "  Building macro panel (panel_merged.csv)..."
if [ -f "data/macro/worldbank_wdi.csv" ] && [ -f "data/macro/ihme_depression.csv" ]; then
  $PYTHON data/build_macro_panel.py || echo "  WARN: build_macro_panel.py failed (macro analyses may be skipped)."
else
  echo "  NOTE: Skip build_macro_panel.py (missing worldbank_wdi.csv and/or IHME files)."
fi

echo ""
echo "  NOTE: IHME GBD data requires manual download."
echo "  See data/README_data.md for instructions."
echo "  After placing IHME/WHO CSVs, rerun this script or run: python3 data/build_macro_panel.py"

# ---- Step 2: Block A (first approximation, requires macro data) ----
echo ""
echo "[Step 2/5] Block A: First approximation (Section 2.1)..."

if [ ! -f "data/macro/panel_merged.csv" ] && [ ! -f "data/macro/panel_with_inactivity.csv" ]; then
    echo "  SKIP: panel data not found in data/macro/"
else
    echo "  Descriptive statistics..."
    $PYTHON analysis/block_a/00_descriptive_stats.py 2>&1 | tee results/block_a_descriptive.log

    echo "  Country-level correlations..."
    $PYTHON analysis/block_a/01_correlation_analysis.py 2>&1 | tee results/block_a_correlations.log

    echo "  Fixed effects model comparison..."
    $PYTHON analysis/block_a/02_fixed_effects_comparison.py 2>&1 | tee results/block_a_fe_comparison.log
fi

# ---- Step 3: Block C (individual-level, no manual data needed) ----
echo ""
echo "[Step 3/5] Block C: Individual-level analyses..."

echo "  NHANES PHQ-9 × Exercise (unweighted)..."
$PYTHON analysis/block_c/01_nhanes_phq9_exercise.py 2>&1 | tee results/nhanes_unweighted.log

echo "  NHANES PHQ-9 × Exercise (weighted)..."
$PYTHON analysis/block_c/01_nhanes_weighted.py 2>&1 | tee results/nhanes_weighted.log

echo "  ATUS Wellbeing (baseline — Summary File approach)..."
HAS_SUM=0
HAS_WB=0
for f in data/atus/atussum_0324.dat data/atus/atussum-0324.dat data/atus/atussum_0323.dat data/atus/atussum-0323.dat; do [ -f "$f" ] && HAS_SUM=1; done
for f in data/atus/wbresp_1013.dat data/atus/wbresp-1013.dat data/atus/atuswb_0313.dat data/atus/atuswb-0313.dat; do [ -f "$f" ] && HAS_WB=1; done

if [ "$HAS_SUM" -eq 1 ] && [ "$HAS_WB" -eq 1 ]; then
  $PYTHON analysis/block_c/02_atus_wellbeing_analysis.py 2>&1 | tee results/atus_baseline.log
else
  echo "  SKIP: atussum + wbresp not found in data/atus/ (see data/README_data.md)" | tee results/atus_baseline.log
fi

echo "  ATUS Ratio Test (additive vs ratio model)..."
if [ "$HAS_SUM" -eq 1 ] && [ "$HAS_WB" -eq 1 ]; then
  $PYTHON analysis/block_c/02b_atus_ratio_test.py 2>&1 | tee results/atus_ratio_test.log
else
  echo "  SKIP: atussum + wbresp not found in data/atus/" | tee results/atus_ratio_test.log
fi

echo "  ATUS Wellbeing (covariates + weights)..."
# Legacy: also check for atusact/atusresp for the covariates variant
HAS_ACT=0
HAS_RESP=0
for f in data/atus/atusact_0324.dat data/atus/atusact-0324.dat data/atus/atusact_0313.dat data/atus/atusact-0313.dat; do [ -f "$f" ] && HAS_ACT=1; done
for f in data/atus/atusresp_0324.dat data/atus/atusresp-0324.dat data/atus/atusresp_0313.dat data/atus/atusresp-0313.dat; do [ -f "$f" ] && HAS_RESP=1; done

if [ "$HAS_ACT" -eq 1 ] && [ "$HAS_WB" -eq 1 ] && [ "$HAS_RESP" -eq 1 ]; then
  $PYTHON analysis/block_c/02_atus_with_covariates.py 2>&1 | tee results/atus_full.log
else
  echo "  SKIP: ATUS raw files not found in data/atus/ (see data/README_data.md)" | tee results/atus_full.log
fi

# ---- Step 3: Block B (requires macro data) ----
echo ""
echo "[Step 4/5] Block B: Macro panel analyses..."

if [ ! -f "data/macro/panel_merged.csv" ] && [ ! -f "data/macro/panel_with_inactivity.csv" ]; then
    echo "  SKIP: panel_merged.csv not found in data/macro/"
    echo "  Download IHME/WHO data and run: python3 data/build_macro_panel.py"
else
    echo "  Hansen PTR + dose-response..."
    $PYTHON analysis/block_b/hansen_ptr.py 2>&1 | tee results/hansen_ptr.log

    echo "  Lag analysis..."
    $PYTHON analysis/block_b/lag_analysis.py 2>&1 | tee results/lag_analysis.log

    echo "  Stationarity tests..."
    $PYTHON analysis/block_b/stationarity.py 2>&1 | tee results/stationarity.log

    echo "  Robustness checks..."
    $PYTHON analysis/block_b/robust_tests.py 2>&1 | tee results/robust_tests.log

    echo "  Alternative estimators (DK-SE, country trends, FD-IV)..."
    $PYTHON analysis/block_b/alternative_estimators.py 2>&1 | tee results/alternative_estimators.log

    echo "  Threshold sweeps (internet & ad proxy)..."
    $PYTHON analysis/block_b/threshold_sweep.py 2>&1 | tee results/threshold_sweep.log

    echo "  First-difference identification (Δlog proxy vs Δlog GDP)..."
    $PYTHON analysis/block_b/first_difference_proxy_vs_gdp.py 2>&1 | tee results/first_difference_proxy_vs_gdp.log

    echo "  Dose-response reversal (quadratic/spline + reversal bootstrap)..."
    $PYTHON analysis/block_b/dose_response_reversal.py 2>&1 | tee results/dose_response_reversal.log

    echo "  Service-sector moderation..."
    $PYTHON analysis/block_b/service_sector_moderation.py 2>&1 | tee results/service_sector_moderation.log

    echo "  Ad proxy validation..."
    $PYTHON analysis/block_b/proxy_validation.py 2>&1 | tee results/proxy_validation.log

    echo "  CCE / IFE estimators (R)..."
    if command -v Rscript &> /dev/null; then
      Rscript analysis/block_b/r_cce_ife_estimators.R 2>&1 | tee results/r_estimators.log
    else
      echo "  SKIP: Rscript not found. Install R and run manually:"
      echo "    Rscript analysis/block_b/r_cce_ife_estimators.R"
    fi
fi

# ---- Step 4: Macro processing capacity (Block C macro) ----
echo ""
echo "[Step 5/5] Block C macro: Processing capacity..."

if [ ! -f "data/macro/panel_with_inactivity.csv" ]; then
    echo "  SKIP: panel_with_inactivity.csv not found"
else
    $PYTHON analysis/block_c/03_macro_processing_capacity.py 2>&1 | tee results/macro_processing.log
fi

# ---- Done ----
echo ""
echo "=============================================="
echo " Pipeline complete."
echo " Results saved to results/*.log"
echo " See results/paper_figure_table_map.md for"
echo " correspondence between outputs and paper."
echo "=============================================="
