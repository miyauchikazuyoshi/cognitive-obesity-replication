# 論文の数値 → スクリプト対応表

論文中の全定量的主張と、それを再現するスクリプト・出力箇所の対応。

## Block A: 第一近似（Section 2.1）

| 論文の主張 | 数値 | スクリプト | 出力変数 |
|---|---|---|---|
| パネル構造（国数・年数・N） | 記述統計 | analysis/block_a/00_descriptive_stats.py | results/block_a_descriptive.json |
| 主要変数の要約統計量 | mean, SD, min, max | analysis/block_a/00_descriptive_stats.py | results/block_a_descriptive.json |
| 国別時系列相関（ad_proxy × depression） | r, p per country | analysis/block_a/01_correlation_analysis.py | results/block_a_correlations.json |
| 相関係数分布（Figure） | histogram | analysis/block_a/01_correlation_analysis.py | results/figures/fig_a1_correlation_distribution.png |
| 国別ランキング（Top/Bottom 10） | r values | analysis/block_a/01_correlation_analysis.py | results/block_a_correlations.json |
| FE仕様比較（Table 5相当） | β, SE, t, p, AIC | analysis/block_a/02_fixed_effects_comparison.py | results/block_a_fe_comparison.json |
| Pooled OLS → Country FE → TWFE → +Cov | 係数変化 | analysis/block_a/02_fixed_effects_comparison.py | results/block_a_fe_comparison.json |

## Block B: 認知入力の飽和構造（Section 2.2）

| 論文の主張 | 数値 | スクリプト | 出力変数 |
|---|---|---|---|
| Granger-style: R→D (1y/2y) | t-stat | analysis/block_b/lag_analysis.py | results/lag_results.json |
| Reverse direction: D→R | t-stat | analysis/block_b/lag_analysis.py | results/lag_results.json |
| Non-stationarity + 1st diff check | t-stat | analysis/block_b/stationarity.py | console output |
| Internet閾値掃引（Figure 8） | β,t by threshold | analysis/block_b/threshold_sweep.py | results/threshold_sweep.json |
| Ad proxy閾値掃引（Figure 9） | β,t by threshold | analysis/block_b/threshold_sweep.py | results/threshold_sweep.json |
| 1階差分+Year FE（Δlog proxy vs Δlog GDP） | β,t, r, VIF | analysis/block_b/first_difference_proxy_vs_gdp.py | results/first_difference_proxy_vs_gdp.json |
| Hansen PTR: suicide threshold | γ* = 25.7, F = 213 | analysis/block_b/hansen_ptr.py | PTR suicide output |
| Hansen PTR: depression | p = 0.654, n.s. | analysis/block_b/hansen_ptr.py | PTR depression output |
| Dose-response（二次/スプライン） | F, AIC, 反転点CI | analysis/block_b/dose_response_reversal.py | results/dose_response_reversal.json |
| サービス業修飾（Figure 13） | 交互作用F,p | analysis/block_b/service_sector_moderation.py | results/service_sector_moderation.json |
| Ratio vs components AIC | AIC values | analysis/block_b/robust_tests.py | console output |
| Ad proxy構成妥当性 | Spearman ρ, rank diff | analysis/block_b/proxy_validation.py | results/proxy_validation.json |
| Proxy分散分解（Internet vs GDP寄与） | r², 解釈 | analysis/block_b/proxy_validation.py | results/proxy_validation.json |
| 残差プロキシ（GDP直交化） | ρ, p | analysis/block_b/proxy_validation.py | results/proxy_validation.json |

## Block C: 体験的処理とメンタルヘルス（Section 2.3）

| 論文の主張 | 数値 | スクリプト | 出力変数 |
|---|---|---|---|
| NHANES: Balance (exercise effect) | β = −1.19, t = −8.56 | analysis/block_c/01_nhanes_phq9_exercise.py | Model 1 |
| NHANES: Balance + full cov | β = −0.84, t = −5.40 | analysis/block_c/01_nhanes_phq9_exercise.py | Model 2 |
| NHANES: 30% attenuation | (1.19−0.84)/1.19 | analysis/block_c/01_nhanes_phq9_exercise.py | Model 1 vs 2 |
| NHANES: Interaction n.s. (high sedentary) | p = 0.71 | analysis/block_c/01_nhanes_phq9_exercise.py | Model 1 interaction |
| ATUS: Interaction n.s. | t = −0.06, p = 0.95 | analysis/block_c/02_atus_wellbeing_analysis.py | OLS interaction |
| ATUS: ΔAIC additive pref. | 73.6 | analysis/block_c/02_atus_wellbeing_analysis.py | AIC comparison |
| ATUS: Full balance test | t = 8.06, p = 8.57e-16 | analysis/block_c/02_atus_wellbeing_analysis.py | Balance test |
| ATUS: Fair/Poor ratio | 25.2% vs 12.5% (≈2x) | analysis/block_c/02_atus_wellbeing_analysis.py | Health rate comparison |
| ATUS: 3.1x worst/best | 20.9% / 6.7% | analysis/block_c/02_atus_wellbeing_analysis.py | 2×2 quadrant |
| Macro processing: TWFE flip | β sign reversal | analysis/block_c/03_macro_processing_capacity.py | TWFE model |

## Robustness / Sensitivity

| 分析 | スクリプト | 備考 |
|---|---|---|
| NHANES weighted | analysis/block_c/01_nhanes_weighted.py | Appendix A.1 対応 |
| ATUS + covariates + weights | analysis/block_c/02_atus_with_covariates.py | Appendix A.2 対応 |
| Unit root / cointegration | analysis/block_b/stationarity.py | Non-stationarity check |
| Income-stratified robustness | analysis/block_b/robust_tests.py | By income tercile |

## 未実装（今後追加予定）

- Chow検定（所得層間の構造変化）
- Suicide quality-weighted subset analysis（WHO品質スコア）

## Alternative Estimators (Stanford Reviewer対応)

| 分析 | スクリプト | 対応指摘 |
|---|---|---|
| Driscoll-Kraay SE | analysis/block_b/alternative_estimators.py | Cross-sectional dependence |
| Country-specific linear trends | analysis/block_b/alternative_estimators.py | Trend confounding |
| First-difference IV (AB logic) | analysis/block_b/alternative_estimators.py | Nickell bias |
| Pesaran CCE (Mean Group) | analysis/block_b/r_cce_ife_estimators.R | Latent common factors |
| Bai IFE (Interactive FE) | analysis/block_b/r_cce_ife_estimators.R | Unit-specific factor loadings |
| Driscoll-Kraay SE (R/plm) | analysis/block_b/r_cce_ife_estimators.R | vcovSCC implementation |
| NHANES weighted | analysis/block_c/01_nhanes_weighted.py | Survey design |
| ATUS + covariates + weights | analysis/block_c/02_atus_with_covariates.py | Confounding |
