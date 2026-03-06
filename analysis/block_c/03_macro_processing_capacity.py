#!/usr/bin/env python3
"""
Section 2.3.1: マクロレベルでの操作化の構造的限界 + ratio vs additive 国レベル検証
----------------------------------------------------------------------
WHO身体不活発指標（NCD_PAC）を処理容量proxyとして、国レベルパネルで
ad_proxy × physical_inactivity の交互作用・比率・加法モデルを比較。

データ: panel_with_inactivity.csv（Block A/Bで構築済み）
  - depression_prevalence: IHME GBD 鬱病有病率
  - internet: ITU インターネット普及率
  - gdp: World Bank GDP per capita
  - physical_inactivity: WHO NCD PAC 身体不活発率
  - suicide: WHO 自殺率

出力:
  - Country FE 5モデル比較（Ad proxy only, Inactivity only, Ratio, Additive, Interaction）
  - AIC比較
  - 2×2象限分析（鬱病 & 自殺）
  - 1階差分 交互作用テスト

再現手順:
  1. panel_with_inactivity.csv をBlock A/Bパイプラインで生成
  2. python 03_macro_processing_capacity.py
"""

import pandas as pd
import numpy as np
from numpy.linalg import lstsq
from scipy import stats
import os
import sys

# ============================================================
# Configuration
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "macro")
PANEL_PATH = os.environ.get(
    "MACRO_PANEL_PATH",
    os.path.join(DATA_DIR, "panel_with_inactivity.csv"),
)
PANEL_PATH = os.path.abspath(PANEL_PATH)

if not os.path.exists(PANEL_PATH):
    print(f"ERROR: Macro panel not found: {PANEL_PATH}")
    print("Place panel_with_inactivity.csv under data/macro/ (see data/README_data.md),")
    print("or set MACRO_PANEL_PATH to the CSV path.")
    sys.exit(1)

# ============================================================
# 1. Load panel data
# ============================================================
panel = pd.read_csv(PANEL_PATH)
panel['ad_proxy'] = panel['internet'] * panel['gdp'] / 1000

sub = panel.dropna(subset=['depression_prevalence', 'ad_proxy', 'physical_inactivity']).copy()
print(f"Panel: N={len(sub)}, Countries={sub['country'].nunique()}")

# ============================================================
# 2. Construct variables
# ============================================================
sub['active'] = 100 - sub['physical_inactivity']
sub['ratio'] = sub['ad_proxy'] / sub['active']
sub['interaction'] = sub['ad_proxy'] * sub['physical_inactivity']

# Demean for Country FE
for col in ['depression_prevalence', 'ratio', 'ad_proxy', 'physical_inactivity', 'interaction']:
    sub[f'{col}_dm'] = sub.groupby('country')[col].transform(lambda x: x - x.mean())

y = sub['depression_prevalence_dm'].values
nc = sub['country'].nunique()
n = len(y)

# ============================================================
# 3. FE regression utility
# ============================================================
def fe_reg(X, y, nc, n):
    b, _, _, _ = lstsq(X, y, rcond=None)
    resid = y - X @ b
    k = X.shape[1]
    rss = np.sum(resid**2)
    se = np.sqrt(rss / (n - nc - k))
    xtx_inv = np.linalg.inv(X.T @ X)
    se_b = np.sqrt(se**2 * np.diag(xtx_inv))
    t = b / se_b
    p = 2 * (1 - stats.t.cdf(np.abs(t), n - nc - k))
    aic = n * np.log(rss / n) + 2 * (nc + k)
    return b, se_b, t, p, rss, aic

# ============================================================
# 4. Five-model comparison (Country FE)
# ============================================================
print(f"\n{'='*70}")
print("MODEL COMPARISON (Country FE)")
print(f"{'='*70}")

models = {
    'A: Ad proxy only': [sub['ad_proxy_dm'].values.reshape(-1, 1), ['ad_proxy']],
    'B: Inactivity only': [sub['physical_inactivity_dm'].values.reshape(-1, 1), ['inactivity']],
    'C: Ratio (proxy/active)': [sub['ratio_dm'].values.reshape(-1, 1), ['ratio']],
    'D: Additive (proxy + inact)': [
        np.column_stack([sub['ad_proxy_dm'].values, sub['physical_inactivity_dm'].values]),
        ['proxy', 'inactivity']
    ],
    'E: Interaction (proxy × inact)': [
        np.column_stack([sub['ad_proxy_dm'].values, sub['physical_inactivity_dm'].values,
                         sub['interaction_dm'].values]),
        ['proxy', 'inactivity', 'proxy×inact']
    ],
}

results = {}
for name, (X, vars_) in models.items():
    b, se_b, t, p, rss, aic = fe_reg(X, y, nc, n)
    results[name] = {'b': b, 't': t, 'p': p, 'aic': aic}
    print(f"\n{name} (AIC = {aic:.1f}):")
    for i, v in enumerate(vars_):
        sig = '***' if p[i] < 0.001 else '**' if p[i] < 0.01 else '*' if p[i] < 0.05 else ''
        print(f"  {v:<20} β = {b[i]:.8f}, t = {t[i]:.2f} {sig}")

# AIC comparison
best_aic = min(r['aic'] for r in results.values())
print(f"\n{'Model':<40} {'AIC':<12} {'ΔAIC'}")
print("-" * 60)
for name, r in results.items():
    delta = r['aic'] - best_aic
    marker = ' ← BEST' if delta == 0 else ''
    print(f"{name:<40} {r['aic']:<12.1f} {delta:.1f}{marker}")

# ============================================================
# 5. 2×2 Quadrant analysis
# ============================================================
print(f"\n{'='*70}")
print("2×2 QUADRANT ANALYSIS")
print(f"{'='*70}")

cm = sub.groupby('country').agg(
    proxy=('ad_proxy', 'mean'),
    inact=('physical_inactivity', 'mean'),
    dep=('depression_prevalence', 'mean')
).reset_index()

proxy_med = cm['proxy'].median()
inact_med = cm['inact'].median()

quads = [
    ('Low proxy + Active',   (cm['proxy'] < proxy_med) & (cm['inact'] < inact_med)),
    ('Low proxy + Inactive', (cm['proxy'] < proxy_med) & (cm['inact'] >= inact_med)),
    ('High proxy + Active',  (cm['proxy'] >= proxy_med) & (cm['inact'] < inact_med)),
    ('High proxy + Inactive',(cm['proxy'] >= proxy_med) & (cm['inact'] >= inact_med)),
]

print(f"Medians: proxy = {proxy_med:.0f}, inactivity = {inact_med:.1f}%")
print(f"\n{'Quadrant':<30} {'N':>5} {'Mean Dep':>10} {'SD':>8}")
print("-" * 55)
for label, mask in quads:
    q = cm[mask]
    print(f"{label:<30} {len(q):>5} {q['dep'].mean():>10.3f} {q['dep'].std():>8.3f}")

worst = cm[(cm['proxy'] >= proxy_med) & (cm['inact'] >= inact_med)]['dep']
best = cm[(cm['proxy'] < proxy_med) & (cm['inact'] < inact_med)]['dep']
t_q, p_q = stats.ttest_ind(worst, best)
d_q = (worst.mean() - best.mean()) / np.sqrt((worst.std()**2 + best.std()**2) / 2)
print(f"\nWorst vs Best: t = {t_q:.2f}, p = {p_q:.4f}, d = {d_q:.2f}")

# ============================================================
# 6. Same for suicide (hard outcome)
# ============================================================
print(f"\n{'='*70}")
print("SUICIDE QUADRANT & INTERACTION")
print(f"{'='*70}")

sub_s = panel.dropna(subset=['suicide', 'ad_proxy', 'physical_inactivity']).copy()
sub_s['active'] = 100 - sub_s['physical_inactivity']
sub_s['interaction'] = sub_s['ad_proxy'] * sub_s['physical_inactivity']

for col in ['suicide', 'ad_proxy', 'physical_inactivity', 'interaction']:
    sub_s[f'{col}_dm'] = sub_s.groupby('country')[col].transform(lambda x: x - x.mean())

y_s = sub_s['suicide_dm'].values
X_int_s = np.column_stack([sub_s['ad_proxy_dm'].values, sub_s['physical_inactivity_dm'].values,
                            sub_s['interaction_dm'].values])
b_s, _, t_s, p_s, _, _ = fe_reg(X_int_s, y_s, sub_s['country'].nunique(), len(y_s))

print(f"Interaction model (suicide):")
for i, v in enumerate(['proxy', 'inactivity', 'proxy×inact']):
    print(f"  {v:<20} β = {b_s[i]:.8f}, t = {t_s[i]:.2f}")

# ============================================================
# 7. First-difference interaction
# ============================================================
print(f"\n{'='*70}")
print("FIRST-DIFFERENCE INTERACTION")
print(f"{'='*70}")

sub_fd = sub.sort_values(['country', 'year'])
sub_fd['d_dep'] = sub_fd.groupby('country')['depression_prevalence'].diff()
sub_fd['d_proxy'] = sub_fd.groupby('country')['ad_proxy'].diff()
sub_fd['d_inact'] = sub_fd.groupby('country')['physical_inactivity'].diff()
fd = sub_fd.dropna(subset=['d_dep', 'd_proxy', 'd_inact']).copy()
fd['d_int'] = fd['d_proxy'] * fd['d_inact']

X_fd = np.column_stack([fd['d_proxy'].values, fd['d_inact'].values, fd['d_int'].values])
y_fd = fd['d_dep'].values
b_fd, _, _, _ = lstsq(X_fd, y_fd, rcond=None)
resid_fd = y_fd - X_fd @ b_fd
se_fd = np.sqrt(np.sum(resid_fd**2) / (len(y_fd) - 3))
xtx_fd = np.linalg.inv(X_fd.T @ X_fd)
se_b_fd = np.sqrt(se_fd**2 * np.diag(xtx_fd))
t_fd = b_fd / se_b_fd

print(f"N = {len(y_fd)}")
for i, name in enumerate(['Δproxy', 'Δinactivity', 'Δproxy×Δinact']):
    print(f"  {name:<20} β = {b_fd[i]:.8f}, t = {t_fd[i]:.2f}")

print(f"\n{'='*70}")
print("結論")
print(f"{'='*70}")
print("マクロレベルの体験的処理の操作化は構造的限界を持つ。")
print("WHO身体不活発指標は「体験的処理容量」の一側面にすぎない。")
print("→ 個人レベルデータ（NHANES, ATUS）で2成分構造を直接検証。")
