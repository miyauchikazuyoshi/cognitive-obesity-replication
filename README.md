# Cognitive Obesity: Replication Materials

**Paper:** "Cognitive Obesity: Cognitive-Experiential Imbalance as a Unifying Framework for Depression, Violence, and Information Overload"

**Project Page:** https://miyauchikazuyoshi.github.io/cognitive-obesity-replication/

|  | Link |
|--|------|
| Preprint (EN) | [PDF](docs/paper/en/cognitive_obesity_preprint.pdf) |
| Preprint (JA) | [PDF](docs/paper/ja_pre/cognitive_obesity_preprint.pdf) |
| LaTeX source (EN) | [docs/paper/en/latex/](docs/paper/en/latex/) |
| LaTeX source (JA) | [docs/paper/ja_pre/latex/](docs/paper/ja_pre/latex/) |

**Status:** Preprint — Available on [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6527598).

## Overview

This repository provides all analysis scripts needed to reproduce the quantitative findings in the paper. The core model is an additive balance equation:

**L = α₁·I − α₂·C**

- **Term 1 (α₁·I):** Cognitive Input — information processing with low sensorimotor loop closure
- **Term 2 (α₂·C):** Experiential Processing — bodily/sensory processing with high sensorimotor loop closure

## Repository Structure

```
cognitive-obesity-replication/
├── README.md                  ← This file
├── LICENSE                    ← MIT
├── requirements.txt           ← Python dependencies
├── run_all.sh                 ← Full pipeline (bash run_all.sh)
│
├── data/
│   ├── README_data.md         ← Data sources & download instructions
│   ├── download_nhanes.py     ← NHANES 2017-2018 (automated)
│   ├── download_atus.py       ← ATUS Wellbeing 2010-2013 (may require manual)
│   ├── download_macro.py      ← World Bank + manual IHME/WHO
│   ├── download_who_gho.py    ← WHO GHO API (inactivity, suicide)
│   └── download_owid.py       ← OWID (schooling years, homicide)
│
├── analysis/
│   ├── block_a/               ← First approximation (Section 2.1)
│   │   ├── 00_descriptive_stats.py       ← Panel overview & summary statistics
│   │   ├── 01_correlation_analysis.py    ← Country-level time-series correlations
│   │   └── 02_fixed_effects_comparison.py ← FE model specification comparison (Table 5)
│   │
│   ├── block_b/               ← Macro panel (177 countries × 1990-2023)
│   │   ├── hansen_ptr.py      ← Section 2.2.6: Threshold regression
│   │   ├── lag_analysis.py    ← Section 2.2.1: Granger causality
│   │   ├── stationarity.py    ← Section 2.2.8: Unit root tests
│   │   ├── robust_tests.py    ← Section 2.2.8: Robustness checks
│   │   ├── alternative_estimators.py ← DK-SE, country trends, FD-IV
│   │   ├── proxy_validation.py      ← Ad proxy construct validation
│   │   ├── adproxy_validation.py    ← ITU convergent validity for AdProxy
│   │   ├── fdr_correction.py        ← FDR (Benjamini-Hochberg) correction
│   │   ├── itu_convergent_validity.py ← ITU mobile broadband validation
│   │   ├── first_difference_proxy_vs_gdp.py ← FD proxy vs GDP comparison
│   │   ├── trend_collinearity_diagnostic.py ← Country trend diagnostics
│   │   └── r_cce_ife_estimators.R   ← CCE (Pesaran), IFE (Bai) [R]
│   │
│   ├── block_c/               ← Individual-level validation
│   │   ├── 01_nhanes_phq9_exercise.py       ← Section 2.3.2 (baseline)
│   │   ├── 01_nhanes_weighted.py            ← Appendix A.1 (survey weights)
│   │   ├── nhanes_sleep_moderation.py       ← Sleep deprivation moderation
│   │   ├── 02_atus_wellbeing_analysis.py    ← Section 2.3.3 (baseline)
│   │   ├── 02b_atus_ratio_test.py           ← Additive vs. ratio model test
│   │   ├── 02_atus_with_covariates.py       ← Appendix A.2 (covariates + weights)
│   │   └── 03_macro_processing_capacity.py  ← Section 2.3.1
│   │
│   └── pilot/                 ← Pilot studies (extensions)
│       └── sns_engagement/    ← SNS toxicity decomposition
│           ├── 00_synthetic_data.py           ← Pipeline test data generator
│           ├── 01_writer_vs_rom.py            ← Writer vs ROM (passive consumer)
│           └── 02_reddit_engagement_spectrum.py ← Reddit posting frequency × MH
│
├── docs/
│   ├── paper/                 ← EN/JA preprints + LaTeX source
│   └── experiment_design/     ← Future experiment designs
│       └── sns_decomposition_design.md  ← Evaluated vs non-evaluated + push×ads 2×2
│
└── results/
    └── paper_figure_table_map.md  ← Paper claim → script correspondence
```

## Quick Start

```bash
# 1. Install dependencies
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

# 2. Download data
# NHANES/WDI: automated. ATUS may require manual download (see data/README_data.md).
python3 data/download_nhanes.py
python3 data/download_atus.py
python3 data/download_macro.py
python3 data/download_who_gho.py
python3 data/download_owid.py
python3 data/check_data.py  # shows any missing manual items (IHME/ATUS)

# 3. Run individual-level analyses (no manual data needed)
python3 analysis/block_c/01_nhanes_phq9_exercise.py
python3 analysis/block_c/02_atus_wellbeing_analysis.py

# 4. (Optional) Macro panel analyses require manual IHME/WHO downloads
# See data/README_data.md, then build the panel:
python3 data/build_macro_panel.py

# Or run the full pipeline
bash run_all.sh
```

## Key Results

### Block B: Macro Panel (Section 2.2)
- **Hansen PTR:** Suicide threshold γ*=25.7 (F=213, significant); Depression threshold not detected (p=0.654)
- **Dose-response:** Continuous inverted-U for depression (quadratic F=127.0), reversal at proxy ≈ 9,200
- **First-difference identification (internet >30%):** Δproxy t=3.16 (significant); ΔGDP t=−4.46 (significant) — proxy carries independent predictive power beyond GDP

### Block C: Individual-Level (Section 2.3)
- **NHANES (N=5,032):** Balance model exercise β=−1.19 (t=−8.56); full cov β=−0.84 (t=−5.40); interaction n.s. (p=0.71)
- **ATUS (N=21,736):** Additive model decisively preferred (ΔAIC=73.6; interaction p=0.95)
- **Ratio test:** Additive L=α₁·I−α₂·C beats ratio R=I/C model (ΔAIC=86); validates theoretical framework
- **Balance test:** Highest passive-to-active ratio quintile shows 1.8× the Fair/Poor health rate (Q5=25.2% vs Q1=13.8%)

## Known Limitations

Documented in the paper (Section 8.4) and Appendix A.3:

1. **Cross-sectional individual data** — causal direction unidentified
2. **Survey weights not applied** in baseline analyses (weighted versions provided)
3. **ATUS covariates not included** in baseline (covariate version provided)
4. **US-only individual data** — cross-cultural generalizability unknown
5. **Ecological fallacy** constrains all macro-level findings
6. **No natural experiment** directly tied to advertising ecosystems
7. **Ad proxy construct validity** limited (indirect measure)

## Pilot Studies

Extending the Cognitive Obesity framework to decompose SNS toxicity:

| Study | Data | Script | Status |
|-------|------|--------|--------|
| Writer vs ROM | Understanding Society Wave 11 | `analysis/pilot/sns_engagement/01_writer_vs_rom.py` | Pipeline ready |
| Reddit engagement spectrum | Reddit MH Dataset (Zenodo) | `analysis/pilot/sns_engagement/02_reddit_engagement_spectrum.py` | Pipeline ready |
| Evaluated vs non-evaluated SNS | *Design only* | — | [`docs/experiment_design/`](docs/experiment_design/sns_decomposition_design.md) |
| Push notification × advertising | *Design only* | — | [`docs/experiment_design/`](docs/experiment_design/sns_decomposition_design.md) |
| **Platform extraction phase** | Meta 10-K, Pew, Reuters | `analysis/pilot/platform_extraction/00_meta_arpu_attitudes.py` | **Results available** |
| ARPU × Attitudes divergence | SEC filings + surveys | (above) | DI slope=+0.60/yr, p=.0001 |
| GSS wellbeing divergence | GSS 1972–2024 | `analysis/pilot/platform_extraction/01_gss_wellbeing_divergence.py` | Pipeline ready |

Run `python3 analysis/pilot/sns_engagement/00_synthetic_data.py` to generate test data, then run the analysis scripts.

## Planned Additions

- [x] Driscoll-Kraay standard errors for cross-sectional dependence
- [x] Country-specific linear trends in fixed effects
- [x] First-difference IV (Arellano-Bond logic) for Nickell bias
- [x] Pesaran CCE (Common Correlated Effects) estimator [R]
- [x] Bai IFE (Interactive Fixed Effects) via phtt [R]
- [ ] Full system GMM (Blundell-Bond) with instrument diagnostics
- [ ] Alternative ad proxy validation with WARC/GroupM data
- [ ] ELSA longitudinal validation (UK panel)
- [ ] Suicide quality-weighted subset analysis (WHO quality scores)
- [ ] Event-study around quasi-exogenous shocks (GDPR, ATT, cable landings)

## Citation

```
Miyauchi, K. (2026). Cognitive Obesity: Cognitive-Experiential Imbalance as a
Unifying Framework for Depression, Violence, and Information Overload.
SSRN. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6527598
```

See also: [CITATION.cff](CITATION.cff)

## License

Code: MIT License. Data: See individual source licenses in `data/README_data.md`.

## Dependencies

**Python:**
```
python3 -m pip install -r requirements.txt  # pandas, numpy, scipy, statsmodels, matplotlib, wbgapi
```

**R** (for CCE/IFE estimators):
```r
install.packages(c("plm", "phtt", "lmtest", "sandwich"))
```
