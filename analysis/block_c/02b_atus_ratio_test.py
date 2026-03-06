#!/usr/bin/env python3
"""
ATUS Ratio Test: 認知肥満比率 R = I/C の直接検証
==================================================
理論の核心: ループが閉じた体験的処理(C) vs 受動的認知入力(I) の比率が
主観的幸福・健康を予測する。

モデル比較:
  Model A: Cantril ~ passive + exercise + active_leisure (加法: 分単位)
  Model B: Cantril ~ R = passive / (exercise + active_leisure + 1) (比率)
  Model C: Cantril ~ log(R) (対数比率)

atussum_0324.dat + wbresp_1013.dat を使用（binary版と同一データ）

出力:
  - AIC比較（加法 vs 比率 vs 対数比率）
  - Rの五分位用量反応
  - 連続dose-response曲線
  - 閾値検出（Rがいくつを超えると悪影響？）
"""

import pandas as pd
import numpy as np
from scipy import stats
from numpy.linalg import lstsq
import os
import sys

# ============================================================
# Configuration
# ============================================================
DATA_DIR = os.environ.get(
    "ATUS_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "atus"),
)
DATA_DIR = os.path.abspath(DATA_DIR)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def _find_first(candidates):
    for fname in candidates:
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            return path
    return None


# ============================================================
# 1. Load & Merge (identical to 02_atus)
# ============================================================
sum_path = _find_first([
    "atussum_0324.dat", "atussum-0324.dat",
    "atussum_0323.dat", "atussum-0323.dat",
])
wb_path = _find_first([
    "wbresp_1013.dat", "wbresp-1013.dat",
    "atuswb_0313.dat", "atuswb-0313.dat",
])

if sum_path is None or wb_path is None:
    print("ERROR: atussum or wbresp not found in", DATA_DIR)
    sys.exit(1)

print(f"Summary: {sum_path}")
print(f"WB:      {wb_path}")

wb = pd.read_csv(wb_path, low_memory=False)
wb.columns = [c.upper() for c in wb.columns]
atus = pd.read_csv(sum_path, low_memory=False)
atus.columns = [c.upper() for c in atus.columns]
atus_wb = atus[atus['TUYEAR'].isin([2010, 2012, 2013])].copy()

cantril_col = 'WECANTRIL' if 'WECANTRIL' in wb.columns else 'WBLADDER'
wb_cols = ['TUCASEID', cantril_col]
if 'WEGENHTH' in wb.columns:
    wb_cols.append('WEGENHTH')

df = atus_wb.merge(wb[wb_cols], on='TUCASEID', how='inner')

# Activity variables
passive_cols = ['T120303', 'T120306']
df['passive'] = df[[c for c in passive_cols if c in df.columns]].sum(axis=1)

active_cols = [
    'T120101', 'T120201', 'T120202',
    'T120307', 'T120308', 'T120309', 'T120310',
    'T120311', 'T120312', 'T120313',
    'T120401', 'T120402', 'T120403', 'T120404', 'T120405',
]
df['active_leisure'] = df[[c for c in active_cols if c in df.columns]].sum(axis=1)

exercise_cols = [f'T1301{i:02d}' for i in range(1, 30)]
df['exercise'] = df[[c for c in exercise_cols if c in df.columns]].sum(axis=1)

# Denominator = experiential processing (loop-closed)
df['denominator'] = df['exercise'] + df['active_leisure']

# Wellbeing
df['cantril'] = pd.to_numeric(df[cantril_col], errors='coerce')
df.loc[df['cantril'] < 0, 'cantril'] = np.nan
df['health'] = pd.to_numeric(df.get('WEGENHTH', pd.Series(dtype=float)), errors='coerce')
df.loc[df['health'] < 1, 'health'] = np.nan
df['poor_health'] = (df['health'] >= 4).astype(float)

# Demographics
df['age'] = pd.to_numeric(df['TEAGE'], errors='coerce')
df['female'] = (pd.to_numeric(df['TESEX'], errors='coerce') == 2).astype(float)

clean = df.dropna(subset=['cantril', 'health']).copy()
clean = clean[(clean['cantril'] >= 0) & (clean['cantril'] <= 10)]
clean = clean[clean['health'].between(1, 5)]
clean['age_c'] = clean['age'] - clean['age'].mean()

print(f"\nN = {len(clean)}")

# ============================================================
# 2. Compute Cognitive Obesity Ratio
# ============================================================
# R = I / C = passive / (exercise + active_leisure)
# +1 smoothing to avoid division by zero
clean['R'] = clean['passive'] / (clean['denominator'] + 1)
clean['log_R'] = np.log1p(clean['R'])

# Also compute the inverse for intuition: C/I ratio (higher = better)
clean['CI_ratio'] = (clean['denominator'] + 1) / (clean['passive'] + 1)
clean['log_CI'] = np.log1p(clean['CI_ratio'])

print(f"R (cognitive obesity ratio) stats:")
print(f"  mean={clean['R'].mean():.2f}, median={clean['R'].median():.2f}")
print(f"  min={clean['R'].min():.2f}, max={clean['R'].max():.2f}")
print(f"  passive=0 (R=0): {(clean['passive']==0).sum()} ({100*(clean['passive']==0).mean():.1f}%)")
print(f"  denominator=0:   {(clean['denominator']==0).sum()} ({100*(clean['denominator']==0).mean():.1f}%)")


# ============================================================
# 3. OLS Model Comparison
# ============================================================
def ols_fit(X, y, names, label):
    n, k = len(y), X.shape[1]
    b, _, _, _ = lstsq(X, y, rcond=None)
    resid = y - X @ b
    rss = np.sum(resid**2)
    tss = np.sum((y - y.mean())**2)
    r2 = 1 - rss / tss
    se = np.sqrt(rss / (n - k))
    xtx_inv = np.linalg.inv(X.T @ X)
    se_b = np.sqrt(se**2 * np.diag(xtx_inv))
    t = b / se_b
    p = 2 * (1 - stats.t.cdf(np.abs(t), n - k))
    aic = n * np.log(rss / n) + 2 * k
    bic = n * np.log(rss / n) + k * np.log(n)
    print(f"\n{label} (N={n}, R²={r2:.4f}, AIC={aic:.1f}, BIC={bic:.1f})")
    print(f"{'Variable':<25} {'β':>10} {'SE':>10} {'t':>8} {'p':>10}")
    print("-" * 68)
    for i, nm in enumerate(names):
        sig = '***' if p[i] < 0.001 else ('**' if p[i] < 0.01 else ('*' if p[i] < 0.05 else ''))
        print(f"{nm:<25} {b[i]:>10.5f} {se_b[i]:>10.5f} {t[i]:>8.2f} {p[i]:>10.2e} {sig}")
    return {'b': b, 'se': se_b, 't': t, 'p': p, 'r2': r2, 'aic': aic, 'bic': bic,
            'names': names, 'rss': rss}


y = clean['cantril'].values
controls = [clean['age_c'].values, clean['female'].values]
ctrl_names = ['Age', 'Female']

print(f"\n{'='*70}")
print("MODEL COMPARISON: Additive vs Ratio vs Log-Ratio")
print("="*70)

# Model A: Additive (continuous minutes)
X_a = np.column_stack([np.ones(len(clean)),
                        clean['passive'].values,
                        clean['exercise'].values,
                        clean['active_leisure'].values] + controls)
res_a = ols_fit(X_a, y,
    ['Intercept', 'Passive(min)', 'Exercise(min)', 'ActiveLeisure(min)'] + ctrl_names,
    'Model A: Additive (continuous minutes)')

# Model B: Ratio R = I/C
X_b = np.column_stack([np.ones(len(clean)),
                        clean['R'].values] + controls)
res_b = ols_fit(X_b, y,
    ['Intercept', 'R = I/C'] + ctrl_names,
    'Model B: Ratio R = passive/(denominator+1)')

# Model C: log(R)
X_c = np.column_stack([np.ones(len(clean)),
                        clean['log_R'].values] + controls)
res_c = ols_fit(X_c, y,
    ['Intercept', 'log(1+R)'] + ctrl_names,
    'Model C: Log-Ratio log(1 + R)')

# Model D: log(R) + log(R)² (quadratic to detect optimal zone)
X_d = np.column_stack([np.ones(len(clean)),
                        clean['log_R'].values,
                        clean['log_R'].values**2] + controls)
res_d = ols_fit(X_d, y,
    ['Intercept', 'log(1+R)', 'log(1+R)²'] + ctrl_names,
    'Model D: Quadratic log-Ratio')

# Model E: Binary (original paper approach) for reference
clean['has_ex'] = (clean['exercise'] > 0).astype(float)
clean['has_al'] = (clean['active_leisure'] > 0).astype(float)
clean['high_passive'] = (clean['passive'] > clean['passive'].median()).astype(float)
X_e = np.column_stack([np.ones(len(clean)),
                        clean['has_ex'].values,
                        clean['has_al'].values,
                        clean['high_passive'].values] + controls)
res_e = ols_fit(X_e, y,
    ['Intercept', 'HasExercise', 'HasActiveLeisure', 'HighPassive'] + ctrl_names,
    'Model E: Binary indicators (paper approach)')

print(f"\n{'='*70}")
print("AIC / BIC / R² COMPARISON")
print("="*70)
models = [('A: Additive', res_a), ('B: Ratio R', res_b),
          ('C: log(R)', res_c), ('D: Quadratic log(R)', res_d),
          ('E: Binary (paper)', res_e)]
print(f"{'Model':<25} {'AIC':>10} {'BIC':>10} {'R²':>10} {'k':>5}")
print("-" * 65)
for name, res in models:
    k = len(res['b'])
    print(f"{name:<25} {res['aic']:>10.1f} {res['bic']:>10.1f} {res['r2']:>10.4f} {k:>5}")

best_aic = min(r['aic'] for _, r in models)
print(f"\nBest AIC: {best_aic:.1f}")
for name, res in models:
    delta = res['aic'] - best_aic
    print(f"  {name}: ΔAIC = {delta:+.1f}")


# ============================================================
# 4. Quintile Dose-Response for R
# ============================================================
print(f"\n{'='*70}")
print("QUINTILE DOSE-RESPONSE for R = passive / (denominator + 1)")
print("="*70)

clean['R_quintile'] = pd.qcut(clean['R'], 5, labels=['Q1\n(Lowest R)', 'Q2', 'Q3', 'Q4', 'Q5\n(Highest R)'])
q_means = clean.groupby('R_quintile', observed=False).agg(
    cantril_mean=('cantril', 'mean'),
    cantril_se=('cantril', lambda x: x.std() / np.sqrt(len(x))),
    health_mean=('health', 'mean'),
    poor_pct=('poor_health', lambda x: x.mean() * 100),
    n=('cantril', 'count'),
    R_median=('R', 'median'),
).reset_index()

print(f"\n{'Quintile':<15} {'N':>6} {'R median':>10} {'Cantril':>8} {'Health':>8} {'PoorH%':>8}")
print("-" * 60)
for _, row in q_means.iterrows():
    ql = str(row['R_quintile']).replace('\n', ' ')
    print(f"{ql:<15} {row['n']:>6} {row['R_median']:>10.2f} "
          f"{row['cantril_mean']:>8.2f} {row['health_mean']:>8.2f} {row['poor_pct']:>7.1f}%")

# Effect size Q1 vs Q5
q1 = clean[clean['R_quintile'] == 'Q1\n(Lowest R)']['cantril']
q5 = clean[clean['R_quintile'] == 'Q5\n(Highest R)']['cantril']
t_q, p_q = stats.ttest_ind(q1, q5)
d_q = (q1.mean() - q5.mean()) / np.sqrt((q1.std()**2 + q5.std()**2) / 2)
print(f"\nQ1 vs Q5: Δ={q1.mean()-q5.mean():.2f}, t={t_q:.2f}, p={p_q:.2e}, d={d_q:.2f}")


# ============================================================
# 5. Decile dose-response for smooth curve
# ============================================================
clean['R_decile'] = pd.qcut(clean['R'], 10, labels=False, duplicates='drop')
dec_means = clean.groupby('R_decile', observed=False).agg(
    cantril=('cantril', 'mean'),
    cantril_se=('cantril', lambda x: x.std() / np.sqrt(len(x))),
    poor_pct=('poor_health', lambda x: x.mean() * 100),
    R_median=('R', 'median'),
    n=('cantril', 'count'),
).reset_index()


# ============================================================
# 6. Health outcome (General Health 1-5)
# ============================================================
print(f"\n{'='*70}")
print("HEALTH OUTCOME: General Health (1=excellent, 5=poor)")
print("="*70)

y_h = clean['health'].values

X_h_ratio = np.column_stack([np.ones(len(clean)),
                              clean['log_R'].values] + controls)
res_h_ratio = ols_fit(X_h_ratio, y_h,
    ['Intercept', 'log(1+R)'] + ctrl_names,
    'Health ~ log(R) + controls')

X_h_add = np.column_stack([np.ones(len(clean)),
                            clean['passive'].values,
                            clean['exercise'].values,
                            clean['active_leisure'].values] + controls)
res_h_add = ols_fit(X_h_add, y_h,
    ['Intercept', 'Passive', 'Exercise', 'ActiveLeisure'] + ctrl_names,
    'Health ~ additive + controls')

print(f"\nHealth AIC: Ratio={res_h_ratio['aic']:.1f}, Additive={res_h_add['aic']:.1f}, "
      f"ΔAIC={res_h_ratio['aic'] - res_h_add['aic']:+.1f}")


# ============================================================
# 7. Visualization
# ============================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle(f'ATUS 2010-2013: Cognitive Obesity Ratio Test (N={len(clean):,})\n'
             'R = Passive Input / (Exercise + Active Cognitive Leisure + 1)',
             fontsize=14, fontweight='bold', y=1.02)

# A: Quintile dose-response (Cantril)
ax = axes[0, 0]
colors_q = ['#4CAF50', '#8BC34A', '#FFC107', '#FF9800', '#F44336']
bars = ax.bar(range(5), q_means['cantril_mean'], yerr=1.96*q_means['cantril_se'],
              color=colors_q, edgecolor='black', linewidth=0.5, capsize=4, alpha=0.85)
for i, row in q_means.iterrows():
    ql = str(row['R_quintile']).replace('\n', ' ')
    ax.text(i, row['cantril_mean'] + 1.96*row['cantril_se'] + 0.03,
            f"{row['cantril_mean']:.2f}\n(R̃={row['R_median']:.1f})",
            ha='center', fontsize=8, fontweight='bold')
ax.set_xticks(range(5))
ax.set_xticklabels(['Q1\nLowest R\n(Balanced)', 'Q2', 'Q3', 'Q4',
                     'Q5\nHighest R\n(Cogn. Obese)'], fontsize=8)
ax.set_ylabel('Mean Cantril Ladder', fontsize=11)
ax.set_ylim(6.8, 7.4)
ax.set_title(f'A. Quintile Dose-Response\nQ1 vs Q5: d={d_q:.2f}***', fontsize=12, fontweight='bold')

# B: Decile smooth curve
ax = axes[0, 1]
ax.errorbar(dec_means['R_median'], dec_means['cantril'],
            yerr=1.96*dec_means['cantril_se'],
            fmt='o-', color='#1976D2', markersize=6, capsize=3, linewidth=2)
ax.set_xlabel('R = Passive / (Denominator + 1)', fontsize=11)
ax.set_ylabel('Mean Cantril Ladder', fontsize=11)
ax.set_title('B. Continuous Dose-Response', fontsize=12, fontweight='bold')
# Add trend line
z = np.polyfit(dec_means['R_median'], dec_means['cantril'], 2)
x_smooth = np.linspace(dec_means['R_median'].min(), dec_means['R_median'].max(), 100)
ax.plot(x_smooth, np.polyval(z, x_smooth), '--', color='#F44336', alpha=0.7, label='Quadratic fit')
ax.legend(fontsize=9)

# C: Model comparison (AIC)
ax = axes[1, 0]
model_names = ['A: Additive\n(minutes)', 'B: Ratio\nR=I/C',
               'C: log(R)', 'D: Quad\nlog(R)', 'E: Binary\n(paper)']
aics = [res_a['aic'], res_b['aic'], res_c['aic'], res_d['aic'], res_e['aic']]
delta_aics = [a - min(aics) for a in aics]
colors_m = ['#9E9E9E', '#2196F3', '#4CAF50', '#FF9800', '#9E9E9E']
bars_m = ax.bar(model_names, delta_aics, color=colors_m, edgecolor='black', linewidth=0.5, alpha=0.85)
ax.set_ylabel('ΔAIC from best', fontsize=11)
ax.set_title('C. Model Comparison (lower = better)', fontsize=12, fontweight='bold')
for i, (d, a) in enumerate(zip(delta_aics, aics)):
    ax.text(i, d + 0.5, f'ΔAIC={d:.0f}\nR²={models[i][1]["r2"]:.4f}',
            ha='center', fontsize=8, fontweight='bold')
ax.axhline(y=0, color='black', linewidth=0.5)

# D: Poor health by quintile
ax = axes[1, 1]
ax.bar(range(5), q_means['poor_pct'], color=colors_q,
       edgecolor='black', linewidth=0.5, alpha=0.85)
for i, row in q_means.iterrows():
    ax.text(i, row['poor_pct'] + 0.5, f"{row['poor_pct']:.1f}%",
            ha='center', fontsize=9, fontweight='bold')
ax.set_xticks(range(5))
ax.set_xticklabels(['Q1\nBalanced', 'Q2', 'Q3', 'Q4', 'Q5\nCogn. Obese'], fontsize=8)
ax.set_ylabel('Fair/Poor Health (%)', fontsize=11)
ax.set_title('D. Health by Cognitive Obesity Quintile', fontsize=12, fontweight='bold')

plt.tight_layout()
fig_path = os.path.join(FIG_DIR, "atus_ratio_test.png")
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"\nFigure saved: {fig_path}")

# ============================================================
# 8. Save JSON
# ============================================================
results = {
    'n': int(len(clean)),
    'R_stats': {
        'mean': float(clean['R'].mean()),
        'median': float(clean['R'].median()),
        'std': float(clean['R'].std()),
    },
    'model_comparison': {
        name: {
            'aic': float(res['aic']),
            'bic': float(res['bic']),
            'r2': float(res['r2']),
            'delta_aic': float(res['aic'] - best_aic),
        }
        for name, res in models
    },
    'quintile_dose_response': {
        f'Q{i+1}': {
            'n': int(row['n']),
            'R_median': float(row['R_median']),
            'cantril': float(row['cantril_mean']),
            'poor_health_pct': float(row['poor_pct']),
        }
        for i, (_, row) in enumerate(q_means.iterrows())
    },
    'Q1_vs_Q5': {
        'delta': float(q1.mean() - q5.mean()),
        't': float(t_q),
        'p': float(p_q),
        'd': float(d_q),
    },
    'log_R_beta_cantril': float(res_c['b'][1]),
    'log_R_t_cantril': float(res_c['t'][1]),
    'log_R_beta_health': float(res_h_ratio['b'][1]),
    'log_R_t_health': float(res_h_ratio['t'][1]),
}

json_path = os.path.join(RESULTS_DIR, "atus_ratio_test.json")
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Results saved: {json_path}")

print(f"\n{'='*70}")
print("SUMMARY")
print("="*70)
print(f"Best model: {min(models, key=lambda x: x[1]['aic'])[0]}")
print(f"Ratio hypothesis: R ↑ → Cantril ↓ (β_logR = {res_c['b'][1]:.4f}, t={res_c['t'][1]:.2f})")
print(f"Quintile gradient: Q1={q_means.iloc[0]['cantril_mean']:.2f} → Q5={q_means.iloc[4]['cantril_mean']:.2f}")
print(f"Cohen's d (Q1 vs Q5): {d_q:.2f}")
