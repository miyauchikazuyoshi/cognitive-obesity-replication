#!/usr/bin/env python3
"""
Section 2.3.3: ATUS Wellbeing Module 2010-2013 (N ≈ 21,736)
------------------------------------------------------------
Cantrilラダーと体験的処理の2成分構造の検証

データ: BLS American Time Use Survey
  - atussum: 活動時間要約（分/日）— T列に事前集計された活動分数を使用
  - wbresp_1013: Wellbeing Module 回答者（2010, 2012, 2013）

活動コード (Summary File T-columns):
  受動的余暇（認知入力proxy）:
    T120303 = TV視聴
    T120306 = レジャー用PC
  能動的認知余暇（体験的処理:認知的）:
    T120101 = 社交
    T1202xx = 社会的イベント
    T120307-T120313 = ゲーム・手芸・趣味・読書・執筆・音楽演奏
    T1204xx = 芸術鑑賞
  身体運動（体験的処理:身体的）:
    T1301xx = スポーツ・エクササイズ

OLS uses BINARY indicators (has_exercise, has_active_leisure) with TEAGE/TESEX controls.

出力:
  - 2×2対偶検証（身体運動有無 × 認知余暇有無）
  - 3×3用量反応
  - OLS回帰（年齢・性別統制、交互作用項検定）
  - フルバランステスト
  - 加法 vs ratio モデル比較（ΔAIC）

再現手順:
  1. data/atus/ に atussum_0324.dat と wbresp_1013.dat を配置
  2. python analysis/block_c/02_atus_wellbeing_analysis.py
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
# 1. Load & Merge — Summary File approach
# ============================================================
print("=" * 70)
print("ATUS Wellbeing Module — Summary File Approach")
print("=" * 70)

# Summary file (pre-computed T-columns with minutes per activity code)
sum_path = _find_first([
    "atussum_0324.dat", "atussum-0324.dat",
    "atussum_0323.dat", "atussum-0323.dat",
    "atussum_0324.csv", "atussum_0323.csv",
])

# Wellbeing respondent file (Cantril ladder, general health)
wb_path = _find_first([
    "wbresp_1013.dat", "wbresp-1013.dat",
    "atuswb_0313.dat", "atuswb-0313.dat",
])

if sum_path is None:
    print(f"ERROR: Cannot find ATUS Summary File in {DATA_DIR}")
    print("Expected: atussum_0324.dat or atussum_0323.dat")
    print("Download from https://www.bls.gov/tus/data.htm")
    sys.exit(1)

if wb_path is None:
    print(f"ERROR: Cannot find Wellbeing respondent file in {DATA_DIR}")
    print("Expected: wbresp_1013.dat")
    print("Download from https://www.bls.gov/tus/data.htm")
    sys.exit(1)

print(f"Summary:    {sum_path}")
print(f"Wellbeing:  {wb_path}")

# Load WB respondent data
print("\nLoading WB respondent data (2010, 2012, 2013)...")
wb = pd.read_csv(wb_path, low_memory=False)
wb.columns = [c.upper() for c in wb.columns]
print(f"WB respondents: {len(wb)}")

# Load Summary File
print("Loading time use summary...")
atus = pd.read_csv(sum_path, low_memory=False)
atus.columns = [c.upper() for c in atus.columns]

# Filter to WB module years
atus_wb = atus[atus['TUYEAR'].isin([2010, 2012, 2013])].copy()
print(f"ATUS 2010/12/13: {len(atus_wb)} respondents")

# Merge
wb_cols = ['TUCASEID']
cantril_col = 'WECANTRIL' if 'WECANTRIL' in wb.columns else 'WBLADDER'
wb_cols.append(cantril_col)
if 'WEGENHTH' in wb.columns:
    wb_cols.append('WEGENHTH')
if 'WEREST' in wb.columns:
    wb_cols.append('WEREST')
if 'WUFINLWGT' in wb.columns:
    wb_cols.append('WUFINLWGT')

df = atus_wb.merge(wb[wb_cols], on='TUCASEID', how='inner')
print(f"Merged: {len(df)}")

# ============================================================
# 2. Construct Activity Variables from T-columns
# ============================================================
# PASSIVE leisure (numerator proxy: TV + computer leisure)
passive_cols = ['T120303', 'T120306']
df['passive_min'] = df[[c for c in passive_cols if c in df.columns]].sum(axis=1)

# ACTIVE cognitive leisure (denominator: cognitive exercise)
active_cols = [
    'T120101',  # Socializing
    'T120201', 'T120202',  # Social events
    'T120307', 'T120308', 'T120309', 'T120310',  # Games, Crafts, Hobbies, Reading
    'T120311', 'T120312', 'T120313',  # Writing, Music performance
    'T120401', 'T120402', 'T120403', 'T120404', 'T120405',  # Arts attendance
]
df['active_leisure_min'] = df[[c for c in active_cols if c in df.columns]].sum(axis=1)

# PHYSICAL exercise (denominator: physical exercise)
exercise_cols = [f'T1301{i:02d}' for i in range(1, 30)]
df['exercise_min'] = df[[c for c in exercise_cols if c in df.columns]].sum(axis=1)

# Total denominator
df['denominator_min'] = df['active_leisure_min'] + df['exercise_min']

# BINARY indicators (key difference from activity-level approach)
df['has_exercise'] = (df['exercise_min'] > 0).astype(int)
df['has_active_leisure'] = (df['active_leisure_min'] > 0).astype(int)
df['has_any_denominator'] = ((df['exercise_min'] > 0) | (df['active_leisure_min'] > 0)).astype(int)
df['high_passive'] = (df['passive_min'] > df['passive_min'].median()).astype(int)

# Wellbeing outcomes
df['cantril'] = pd.to_numeric(df[cantril_col], errors='coerce')
df.loc[df['cantril'] < 0, 'cantril'] = np.nan
if 'WEGENHTH' in df.columns:
    df['health'] = pd.to_numeric(df['WEGENHTH'], errors='coerce')
    df.loc[df['health'] < 1, 'health'] = np.nan
    df['poor_health'] = (df['health'] >= 4).astype(float)
else:
    df['health'] = np.nan
    df['poor_health'] = np.nan

# Demographics from Summary File (TEAGE, TESEX)
if 'TEAGE' in df.columns:
    df['age'] = pd.to_numeric(df['TEAGE'], errors='coerce')
else:
    df['age'] = np.nan
if 'TESEX' in df.columns:
    df['female'] = (pd.to_numeric(df['TESEX'], errors='coerce') == 2).astype(float)
else:
    df['female'] = np.nan

# Clean: drop invalid cantril and health
clean = df.dropna(subset=['cantril', 'health']).copy()
clean = clean[(clean['cantril'] >= 0) & (clean['cantril'] <= 10)]
clean = clean[clean['health'].between(1, 5)]

HAS_DEMOGRAPHICS = clean['age'].notna().any() and clean['female'].notna().any()
HAS_HEALTH = clean['health'].notna().any()

print(f"\nClean N = {len(clean)}")
print(f"Cantril mean: {clean['cantril'].mean():.2f}")
print(f"Passive median: {clean['passive_min'].median():.0f} min/day")
print(f"Active leisure: mean={clean['active_leisure_min'].mean():.0f} min, "
      f"zero-rate={100*(clean['active_leisure_min']==0).mean():.1f}%")
print(f"Exercise: mean={clean['exercise_min'].mean():.0f} min, "
      f"zero-rate={100*(clean['exercise_min']==0).mean():.1f}%")
print(f"Demographics: {'TEAGE + TESEX available' if HAS_DEMOGRAPHICS else 'NOT available'}")

# ============================================================
# 3. 2×2 Contrapositive Test
# ============================================================
print(f"\n{'='*70}")
print("2×2 CONTRAPOSITIVE: Exercise × Active Cognitive Leisure")
print("=" * 70)

quadrants = {
    'Exercise + ActiveLeisure (分母充実)':
        (clean['has_exercise'] == 1) & (clean['has_active_leisure'] == 1),
    'Exercise only':
        (clean['has_exercise'] == 1) & (clean['has_active_leisure'] == 0),
    'ActiveLeisure only':
        (clean['has_exercise'] == 0) & (clean['has_active_leisure'] == 1),
    'Neither (分母枯渇)':
        (clean['has_exercise'] == 0) & (clean['has_active_leisure'] == 0),
}

print(f"\n{'Quadrant':<40} {'N':>6} {'Cantril':>8} {'Health':>8} {'PoorH%':>8}")
print("-" * 74)
q_results = {}
for name, mask in quadrants.items():
    q = clean[mask]
    q_results[name] = {
        'n': int(len(q)),
        'cantril': float(q['cantril'].mean()),
        'health': float(q['health'].mean()),
        'poor_health': float(q['poor_health'].mean() * 100),
    }
    print(f"{name:<40} {len(q):>6} {q['cantril'].mean():>8.2f} {q['health'].mean():>8.2f} "
          f"{q['poor_health'].mean()*100:>7.1f}%")

both = clean[(clean['has_exercise'] == 1) & (clean['has_active_leisure'] == 1)]
neither = clean[(clean['has_exercise'] == 0) & (clean['has_active_leisure'] == 0)]
t_c, p_c = stats.ttest_ind(both['cantril'], neither['cantril'])
d_c = (both['cantril'].mean() - neither['cantril'].mean()) / \
      np.sqrt((both['cantril'].std()**2 + neither['cantril'].std()**2) / 2)
t_h, p_h = stats.ttest_ind(neither['health'], both['health'])
d_h = (neither['health'].mean() - both['health'].mean()) / \
      np.sqrt((both['health'].std()**2 + neither['health'].std()**2) / 2)

print(f"\nBoth vs Neither:")
print(f"  Cantril: Δ={both['cantril'].mean()-neither['cantril'].mean():.2f}, "
      f"t={t_c:.2f}, p={p_c:.2e}, d={d_c:.2f}")
print(f"  Health:  Δ={neither['health'].mean()-both['health'].mean():.2f}, "
      f"t={t_h:.2f}, p={p_h:.2e}, d={d_h:.2f}")

# ============================================================
# 4. 3×3 Dose-Response
# ============================================================
print(f"\n{'='*70}")
print("3×3 DOSE-RESPONSE")
print("=" * 70)

clean['ex_level'] = 'None'
ex_pos = clean[clean['exercise_min'] > 0]['exercise_min']
if len(ex_pos) > 10:
    ex_med = ex_pos.median()
    clean.loc[(clean['exercise_min'] > 0) & (clean['exercise_min'] <= ex_med), 'ex_level'] = 'Low'
    clean.loc[clean['exercise_min'] > ex_med, 'ex_level'] = 'High'

clean['al_level'] = 'None'
al_pos = clean[clean['active_leisure_min'] > 0]['active_leisure_min']
if len(al_pos) > 10:
    al_med = al_pos.median()
    clean.loc[(clean['active_leisure_min'] > 0) & (clean['active_leisure_min'] <= al_med), 'al_level'] = 'Low'
    clean.loc[clean['active_leisure_min'] > al_med, 'al_level'] = 'High'

print(f"\n{'Exercise':<10} {'ActiveLeis':<12} {'N':>6} {'Cantril':>8} {'PoorH%':>8}")
print("-" * 50)
dose_results = {}
for ex in ['None', 'Low', 'High']:
    for al in ['None', 'Low', 'High']:
        q = clean[(clean['ex_level'] == ex) & (clean['al_level'] == al)]
        if len(q) > 20:
            key = f"{ex}_{al}"
            dose_results[key] = {
                'n': int(len(q)),
                'cantril': float(q['cantril'].mean()),
                'poor_health': float(q['poor_health'].mean() * 100),
            }
            print(f"{ex:<10} {al:<12} {len(q):>6} {q['cantril'].mean():>8.2f} "
                  f"{q['poor_health'].mean()*100:>7.1f}%")

# ============================================================
# 5. OLS Regression — BINARY indicators + demographics
# ============================================================
print(f"\n{'='*70}")
print("OLS REGRESSION: Cantril ← binary activities + controls")
print("=" * 70)

# Center age
if HAS_DEMOGRAPHICS:
    clean['age_c'] = clean['age'] - clean['age'].mean()
clean['interaction_ex_al'] = clean['has_exercise'] * clean['has_active_leisure']


def ols(X, y, names, label):
    n, k = len(y), X.shape[1]
    b, _, _, _ = lstsq(X, y, rcond=None)
    resid = y - X @ b
    rss = np.sum(resid**2)
    se = np.sqrt(rss / (n - k))
    xtx_inv = np.linalg.inv(X.T @ X)
    se_b = np.sqrt(se**2 * np.diag(xtx_inv))
    t = b / se_b
    p = 2 * (1 - stats.t.cdf(np.abs(t), n - k))
    aic = n * np.log(rss / n) + 2 * k
    print(f"\n{label} (N={n}, AIC={aic:.1f})")
    print(f"{'Variable':<30} {'β':>8} {'SE':>8} {'t':>8}")
    print("-" * 58)
    for i, nm in enumerate(names):
        sig = '***' if p[i] < 0.001 else ('**' if p[i] < 0.01 else ('*' if p[i] < 0.05 else ''))
        print(f"{nm:<30} {b[i]:>8.4f} {se_b[i]:>8.4f} {t[i]:>8.2f} {sig}")
    return b, se_b, t, p, aic


# Build design matrix
y = clean['cantril'].values

# Model A: additive with binary indicators
x_cols = [
    np.ones(len(clean)),
    clean['has_exercise'].values.astype(float),
    clean['has_active_leisure'].values.astype(float),
    clean['interaction_ex_al'].values.astype(float),
    clean['high_passive'].values.astype(float),
]
x_names = ['Intercept', 'Exercise(身体)', 'ActiveLeisure(認知)',
           'Exercise×Leisure', 'HighPassive(分子)']

if HAS_DEMOGRAPHICS:
    x_cols.append(clean['age_c'].values)
    x_cols.append(clean['female'].values)
    x_names.extend(['Age', 'Female'])

X = np.column_stack(x_cols)
b_a, se_a, t_a, p_a, aic_a = ols(X, y, x_names,
    "Cantril Ladder" + (" — with Age/Sex controls" if HAS_DEMOGRAPHICS else ""))

# Also run for Health outcome
y_h = clean['health'].values
b_h, se_h, t_h_ols, p_h_ols, aic_h = ols(X, y_h, x_names,
    "General Health (1-5, higher=worse)" + (" — with Age/Sex controls" if HAS_DEMOGRAPHICS else ""))

# Model B: additive without interaction (for AIC comparison)
x_cols_noint = [
    np.ones(len(clean)),
    clean['has_exercise'].values.astype(float),
    clean['has_active_leisure'].values.astype(float),
    clean['high_passive'].values.astype(float),
]
x_names_noint = ['Intercept', 'Exercise(身体)', 'ActiveLeisure(認知)', 'HighPassive(分子)']
if HAS_DEMOGRAPHICS:
    x_cols_noint.append(clean['age_c'].values)
    x_cols_noint.append(clean['female'].values)
    x_names_noint.extend(['Age', 'Female'])

X_noint = np.column_stack(x_cols_noint)
b_ni, se_ni, t_ni, p_ni, aic_ni = ols(X_noint, y, x_names_noint,
    "Additive model (no interaction)")

print(f"\nΔAIC (interaction - no interaction) = {aic_a - aic_ni:.1f}")

# ============================================================
# 6. Full Ratio Test: HighPassive × NoDenominator
# ============================================================
print(f"\n{'='*70}")
print("FULL RATIO: HighPassive(分子) × NoDenominator(分母)")
print("=" * 70)

for name, mask in [
    ('LowPassive + Denom (最良)', (clean['high_passive'] == 0) & (clean['has_any_denominator'] == 1)),
    ('LowPassive + NoDenom', (clean['high_passive'] == 0) & (clean['has_any_denominator'] == 0)),
    ('HighPassive + Denom', (clean['high_passive'] == 1) & (clean['has_any_denominator'] == 1)),
    ('HighPassive + NoDenom (最悪)', (clean['high_passive'] == 1) & (clean['has_any_denominator'] == 0)),
]:
    q = clean[mask]
    print(f"{name:<35} N={len(q):>5}, Cantril={q['cantril'].mean():.2f}, "
          f"PoorH={q['poor_health'].mean()*100:.1f}%")

best_bal = clean[(clean['high_passive'] == 0) & (clean['has_any_denominator'] == 1)]['cantril']
worst_bal = clean[(clean['high_passive'] == 1) & (clean['has_any_denominator'] == 0)]['cantril']
t_bal, p_bal = stats.ttest_ind(best_bal, worst_bal)
d_bal = (best_bal.mean() - worst_bal.mean()) / np.sqrt(
    (best_bal.std()**2 + worst_bal.std()**2) / 2)
print(f"\nBest vs Worst: t={t_bal:.2f}, p={p_bal:.2e}, d={d_bal:.2f}")

# ============================================================
# 7. Visualization
# ============================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# A: 2×2 Cantril
ax = axes[0, 0]
labels_q = ['Exercise○\nLeisure○\n(Enriched)', 'Exercise○\nLeisure×',
            'Exercise×\nLeisure○', 'Exercise×\nLeisure×\n(Depleted)']
cantril_vals = [q_results[k]['cantril'] for k in quadrants]
ns = [q_results[k]['n'] for k in quadrants]
colors = ['#4CAF50', '#8BC34A', '#FFC107', '#F44336']
ax.bar(range(4), cantril_vals, color=colors, edgecolor='black', linewidth=0.5)
ax.set_xticks(range(4))
ax.set_xticklabels(labels_q, fontsize=8)
ax.set_ylabel('Cantril Ladder (0-10)', fontsize=11)
ax.set_ylim(min(cantril_vals) - 0.5, max(cantril_vals) + 0.5)
ax.set_title(f'A. Life Satisfaction\nt={t_c:.1f}, d={d_c:.2f}', fontsize=12, fontweight='bold')
for i, (c, n) in enumerate(zip(cantril_vals, ns)):
    ax.text(i, c + 0.05, f'{c:.2f}\n(n={n:,})', ha='center', fontsize=9)

# B: Poor health
ax = axes[0, 1]
poor_vals = [q_results[k]['poor_health'] for k in quadrants]
ax.bar(range(4), poor_vals, color=colors, edgecolor='black', linewidth=0.5)
ax.set_xticks(range(4))
ax.set_xticklabels(labels_q, fontsize=8)
ax.set_ylabel('Fair/Poor Health (%)', fontsize=11)
ax.set_title(f'B. Self-Rated Health\nt={t_h:.1f}, d={d_h:.2f}', fontsize=12, fontweight='bold')
for i, (r, n) in enumerate(zip(poor_vals, ns)):
    ax.text(i, r + 0.5, f'{r:.1f}%', ha='center', fontsize=9)

# C: 3×3 heatmap
ax = axes[1, 0]
grid = np.zeros((3, 3))
for i, ex in enumerate(['None', 'Low', 'High']):
    for j, al in enumerate(['None', 'Low', 'High']):
        q = clean[(clean['ex_level'] == ex) & (clean['al_level'] == al)]
        grid[2 - i, j] = q['cantril'].mean() if len(q) > 20 else np.nan
im = ax.imshow(grid, cmap='RdYlGn', aspect='auto', vmin=6.0, vmax=7.5)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['None', 'Low', 'High'])
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(['High', 'Low', 'None'])
ax.set_xlabel('Active Cognitive Leisure (認知運動)', fontsize=11)
ax.set_ylabel('Physical Exercise (身体運動)', fontsize=11)
ax.set_title('C. 3×3 Dose-Response', fontsize=12, fontweight='bold')
for i in range(3):
    for j in range(3):
        if not np.isnan(grid[i, j]):
            ax.text(j, i, f'{grid[i, j]:.2f}', ha='center', va='center',
                    fontsize=11, fontweight='bold')
plt.colorbar(im, ax=ax)

# D: OLS Coefficients (binary indicators)
ax = axes[1, 1]
coef_labels = ['Exercise\n(Binary)', 'Active Leisure\n(Binary)', 'Interaction', 'High Passive']
coef_idx = [1, 2, 3, 4]  # Indices in b_a
coef_vals = [b_a[i] for i in coef_idx]
coef_errs = [1.96 * se_a[i] for i in coef_idx]
coef_colors = ['#4CAF50', '#2196F3', '#9C27B0', '#F44336']
ax.barh(range(4), coef_vals, xerr=coef_errs, color=coef_colors, alpha=0.8,
        edgecolor='black', linewidth=0.5, capsize=4)
ax.set_yticks(range(4))
ax.set_yticklabels(coef_labels, fontsize=9)
ax.axvline(x=0, color='black', linewidth=0.5)
ax.set_xlabel('Effect on Cantril (β)', fontsize=11)
ctrl_str = 'w/ age, sex controls' if HAS_DEMOGRAPHICS else 'no demog. controls'
ax.set_title(f'D. OLS Coefficients ({ctrl_str})', fontsize=12, fontweight='bold')
for i, idx in enumerate(coef_idx):
    x_pos = coef_vals[i] + coef_errs[i] * 1.1 if coef_vals[i] > 0 else coef_vals[i] - coef_errs[i] * 1.3
    ax.text(x_pos, i, f't={t_a[idx]:.1f}', va='center', fontsize=9, fontweight='bold')

fig.suptitle(f'ATUS 2010-2013 Wellbeing Module (N={len(clean):,})\n'
             'Denominator Contrapositive: Exercise × Cognitive Leisure → Wellbeing',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig_path = os.path.join(FIG_DIR, "atus_wellbeing_analysis.png")
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"\nFigure saved: {fig_path}")

# ============================================================
# 8. Save Results JSON
# ============================================================
results = {
    'n': int(len(clean)),
    'cantril_mean': float(clean['cantril'].mean()),
    'summary_file_used': True,
    'binary_indicators': True,
    'has_demographics': bool(HAS_DEMOGRAPHICS),
    'quadrant_results': q_results,
    'contrapositive': {
        'delta_cantril': float(both['cantril'].mean() - neither['cantril'].mean()),
        't_cantril': float(t_c),
        'p_cantril': float(p_c),
        'd_cantril': float(d_c),
        'delta_health': float(neither['health'].mean() - both['health'].mean()),
        't_health': float(t_h),
        'p_health': float(p_h),
        'd_health': float(d_h),
    },
    'ols_cantril': {
        'beta_exercise': float(b_a[1]),
        'se_exercise': float(se_a[1]),
        't_exercise': float(t_a[1]),
        'p_exercise': float(p_a[1]),
        'beta_active_leisure': float(b_a[2]),
        'se_active_leisure': float(se_a[2]),
        't_active_leisure': float(t_a[2]),
        'p_active_leisure': float(p_a[2]),
        'beta_interaction': float(b_a[3]),
        't_interaction': float(t_a[3]),
        'p_interaction': float(p_a[3]),
        'beta_high_passive': float(b_a[4]),
        't_high_passive': float(t_a[4]),
        'p_high_passive': float(p_a[4]),
    },
    'ols_health': {
        'beta_exercise': float(b_h[1]),
        't_exercise': float(t_h_ols[1]),
        'beta_active_leisure': float(b_h[2]),
        't_active_leisure': float(t_h_ols[2]),
    },
    'balance_test': {
        'best_cantril': float(best_bal.mean()),
        'worst_cantril': float(worst_bal.mean()),
        't': float(t_bal),
        'p': float(p_bal),
        'd': float(d_bal),
    },
    'aic_with_interaction': float(aic_a),
    'aic_no_interaction': float(aic_ni),
}

if HAS_DEMOGRAPHICS:
    results['ols_cantril']['beta_age'] = float(b_a[5])
    results['ols_cantril']['t_age'] = float(t_a[5])
    results['ols_cantril']['beta_female'] = float(b_a[6])
    results['ols_cantril']['t_female'] = float(t_a[6])

json_path = os.path.join(RESULTS_DIR, "atus_wellbeing.json")
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Results saved: {json_path}")

print(f"\n{'='*70}")
print("ANALYSIS COMPLETE")
print(f"{'='*70}")
