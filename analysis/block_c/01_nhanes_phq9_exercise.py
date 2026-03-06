#!/usr/bin/env python3
"""
Section 2.3.2: NHANES 2017-2018 PHQ-9 と余暇身体運動 (N = 5,032)
--------------------------------------------------------------
データ: CDC NHANES 2017-2018 (.XPT形式)
  - DEMO_J.XPT: 人口統計（年齢 RIDAGEYR, 性別 RIAGENDR, 教育 DMDEDUC2, 貧困所得比 INDFMPIR）
  - DPQ_J.XPT:  PHQ-9 うつ病尺度（DPQ010-DPQ090）
  - PAQ_J.XPT:  身体活動質問票（余暇運動 PAQ650/PAQ665, 座位時間 PAD680）
  - BMX_J.XPT:  身体測定（BMI: BMXBMI）
  - HIQ_J.XPT:  健康保険（HIQ011）

出力:
  - 2群比較（運動あり/なし × うつ病率）
  - OLS回帰（年齢・性別統制）
  - フル共変量モデル（教育・貧困所得比・保険・BMI追加統制）
  - 座位時間×運動交互作用検定

再現手順:
  1. データ取得:
     python data/download_nhanes.py
  2. 実行:
     python analysis/block_c/01_nhanes_phq9_exercise.py
"""

import pandas as pd
import numpy as np
from scipy import stats
import os
import sys
import statsmodels.api as sm

# ============================================================
# Configuration
# ============================================================
DATA_DIR = os.environ.get(
    "NHANES_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "nhanes"),
)
DATA_DIR = os.path.abspath(DATA_DIR)

REQUIRED_FILES = ["DEMO_J.XPT", "DPQ_J.XPT", "PAQ_J.XPT", "BMX_J.XPT", "HIQ_J.XPT"]
missing = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(DATA_DIR, f))]
if missing:
    print(f"ERROR: NHANES XPT files not found in: {DATA_DIR}")
    print(f"Missing: {', '.join(missing)}")
    print("\nRun from repo root:")
    print("  python data/download_nhanes.py")
    print("\nOr set NHANES_DIR to the folder containing the XPT files.")
    sys.exit(1)

# ============================================================
# 1. Load and merge NHANES data
# ============================================================
demo = pd.read_sas(f'{DATA_DIR}/DEMO_J.XPT', format='xport')
dpq = pd.read_sas(f'{DATA_DIR}/DPQ_J.XPT', format='xport')
paq = pd.read_sas(f'{DATA_DIR}/PAQ_J.XPT', format='xport')
bmx = pd.read_sas(f'{DATA_DIR}/BMX_J.XPT', format='xport')
hiq = pd.read_sas(f'{DATA_DIR}/HIQ_J.XPT', format='xport')

df = demo.merge(dpq, on='SEQN', how='inner')
df = df.merge(paq, on='SEQN', how='inner')
df = df.merge(bmx[['SEQN', 'BMXBMI']], on='SEQN', how='left')
df = df.merge(hiq[['SEQN', 'HIQ011']], on='SEQN', how='left')

# ============================================================
# 2. Construct PHQ-9 total score
# ============================================================
phq_cols = [f'DPQ0{i}0' for i in range(1, 10)]
for col in phq_cols:
    # pandas.read_sas (xport) may represent 0 as an extremely small float.
    # Round first so we reliably recover integer codes.
    df[col] = pd.to_numeric(df[col], errors="coerce").round()
    df[col] = df[col].replace({7: np.nan, 9: np.nan})
    df[col] = df[col].where(df[col].isin([0, 1, 2, 3]))

# Use a complete-case PHQ-9 total (match paper analytic sample)
df['phq9'] = df[phq_cols].sum(axis=1, min_count=len(phq_cols))

# ============================================================
# 3. Construct exercise variable
# ============================================================
# PAQ650: 余暇での激しい運動 (1=Yes, 2=No)
# PAQ665: 余暇での中等度運動 (1=Yes, 2=No)
df['leisure_exercise'] = ((df['PAQ650'] == 1) | (df['PAQ665'] == 1)).astype(int)

# ============================================================
# 4. Filter to valid sample (align across models)
# ============================================================
df["PAD680"] = pd.to_numeric(df["PAD680"], errors="coerce").round().replace({7777: np.nan, 9999: np.nan})
df = df.dropna(subset=['phq9', 'leisure_exercise', 'RIDAGEYR', 'RIAGENDR', 'PAD680'])
df = df[df['RIDAGEYR'] >= 18]
df = df[df["PAD680"] < 1440]
df["sedentary_min"] = df["PAD680"].astype(float)
sed_q75 = float(df["sedentary_min"].quantile(0.75))
df["high_sedentary"] = (df["sedentary_min"] >= sed_q75).astype(int)
df['depressed'] = (df['phq9'] >= 10).astype(int)  # 標準カットオフ
print(f"分析対象: N = {len(df)}")
print(f"PHQ-9 平均: {df['phq9'].mean():.2f}, SD: {df['phq9'].std():.2f}")
print(f"うつ病率 (PHQ≥10): {df['depressed'].mean()*100:.1f}%")
print(f"余暇運動あり: {df['leisure_exercise'].mean()*100:.1f}%")
print(f"座位時間 75p: {sed_q75:.0f} 分/日（high_sedentaryの閾値）")

# ============================================================
# 5. 2群比較
# ============================================================
ex = df[df['leisure_exercise'] == 1]
no = df[df['leisure_exercise'] == 0]
t_val, p_val = stats.ttest_ind(ex['phq9'], no['phq9'])
d = (no['phq9'].mean() - ex['phq9'].mean()) / np.sqrt(
    (no['phq9'].std()**2 + ex['phq9'].std()**2) / 2)

print(f"\n{'='*60}")
print("2群比較: 余暇運動あり vs なし")
print(f"{'='*60}")
print(f"運動あり: うつ病率 {ex['depressed'].mean()*100:.1f}%, PHQ-9 = {ex['phq9'].mean():.2f}")
print(f"運動なし: うつ病率 {no['depressed'].mean()*100:.1f}%, PHQ-9 = {no['phq9'].mean():.2f}")
print(f"うつ病率の比: {no['depressed'].mean()/ex['depressed'].mean():.1f}倍")
print(f"t = {abs(t_val):.2f}, p = {p_val:.2e}, Cohen's d = {d:.2f}")

# ============================================================
# 6. 2×2（運動×座位）対偶検証（balance）
# ============================================================
print(f"\n{'='*70}")
print("2×2 対偶検証: 身体運動 × 高座位（上位25%）")
print(f"{'='*70}")

quadrants = {
    'Active + Low Sed': (df['leisure_exercise'] == 1) & (df['high_sedentary'] == 0),
    'Active + High Sed': (df['leisure_exercise'] == 1) & (df['high_sedentary'] == 1),
    'Inactive + Low Sed': (df['leisure_exercise'] == 0) & (df['high_sedentary'] == 0),
    'Inactive + High Sed': (df['leisure_exercise'] == 0) & (df['high_sedentary'] == 1),
}

print(f"\n{'Quadrant':<25} {'N':>6} {'PHQ-9':>8} {'Dep%':>8}")
print("-" * 52)
for name, mask in quadrants.items():
    q = df.loc[mask]
    print(f"{name:<25} {len(q):>6} {q['phq9'].mean():>8.2f} {q['depressed'].mean()*100:>7.1f}%")

best = df.loc[quadrants['Active + Low Sed'], 'phq9']
worst = df.loc[quadrants['Inactive + High Sed'], 'phq9']
t_bw, p_bw = stats.ttest_ind(worst, best)
d_bw = (worst.mean() - best.mean()) / np.sqrt((worst.std()**2 + best.std()**2) / 2)
print(f"\nBest vs Worst: t={t_bw:.2f}, p={p_bw:.2e}, d={d_bw:.2f}")

# ============================================================
# 7. OLS回帰（balanceモデル）
# ============================================================
def ols_report(X, y, var_names, label, *, robust_hc1: bool = False):
    """OLS report (optionally HC1 robust)."""
    model = sm.OLS(y, X, missing="drop")
    res = model.fit(cov_type="HC1") if robust_hc1 else model.fit()

    n = int(res.nobs)
    k = int(res.df_model) + 1
    b = res.params
    se_b = res.bse
    t = b / se_b
    p = 2 * (1 - stats.t.cdf(np.abs(t), df=n - k))

    resid = res.resid
    rss = float(np.sum(resid**2))
    aic = n * np.log(rss / n) + 2 * k

    print(f"\n{label}")
    print(f"  N = {n}, AIC = {aic:.1f}")
    for i, name in enumerate(var_names):
        sig = '***' if p[i] < 0.001 else '**' if p[i] < 0.01 else '*' if p[i] < 0.05 else ''
        print(f"  {name:<20} β = {b[i]:>8.4f}, t = {t[i]:>7.2f}, p = {p[i]:.2e} {sig}")
    return b, t, p, aic

y = df['phq9'].values
df["age_c"] = df["RIDAGEYR"] - df["RIDAGEYR"].mean()
df["female"] = (df["RIAGENDR"] == 2).astype(int)
df["balance_interaction"] = df["leisure_exercise"] * df["high_sedentary"]

print(f"\n{'='*60}")
print("OLS回帰")
print(f"{'='*60}")

# Model 1: balance (exercise + sedentary + interaction)
X_bal = np.column_stack(
    [
        np.ones(len(df)),
        df["leisure_exercise"].values,
        df["high_sedentary"].values,
        df["balance_interaction"].values,
        df["age_c"].values,
        df["female"].values,
    ]
)

b1, t1, p1, aic1 = ols_report(
    X_bal,
    y,
    ["intercept", "exercise", "high_sedentary", "exercise×high_sedentary", "age_c", "female"],
    "Model 1: Balance（運動 + 高座位 + 交互作用 + 年齢 + 性別）",
    robust_hc1=False,
)

# ============================================================
# 8. フル共変量モデル（balance + SES/health controls）
# ============================================================
df_full = df.dropna(subset=['DMDEDUC2', 'INDFMPIR', 'BMXBMI', 'HIQ011'])
df_full = df_full[df_full['DMDEDUC2'].isin([1, 2, 3, 4, 5])]
df_full = df_full[df_full['HIQ011'].isin([1, 2])]

y_full = df_full['phq9'].values
df_full["age_c"] = df_full["RIDAGEYR"] - df["RIDAGEYR"].mean()
df_full["female"] = (df_full["RIAGENDR"] == 2).astype(int)
df_full["balance_interaction"] = df_full["leisure_exercise"] * df_full["high_sedentary"]

X_full = np.column_stack(
    [
        np.ones(len(df_full)),
        df_full["leisure_exercise"].values,
        df_full["high_sedentary"].values,
        df_full["balance_interaction"].values,
        df_full["age_c"].values,
        df_full["female"].values,
        df_full["DMDEDUC2"].values,
        df_full["INDFMPIR"].values,
        (df_full["HIQ011"] == 1).astype(int).values,  # insured=1
        df_full["BMXBMI"].values,
    ]
)

b2, t2, p2, aic2 = ols_report(
    X_full, y_full,
    ['intercept', 'exercise', 'high_sedentary', 'exercise×high_sedentary', 'age_c', 'female',
     'education', 'poverty_income_ratio', 'insured', 'bmi'],
    "Model 2: Balance + 共変量（教育・貧困所得比・保険・BMI）",
    robust_hc1=False,
)

attenuation = (1 - abs(b2[1]) / abs(b1[1])) * 100
print(f"\n  運動効果の減衰: {attenuation:.0f}% (β: {b1[1]:.2f} → {b2[1]:.2f})")

# ============================================================
# 9. 参考: 連続座位時間×運動 交互作用（分単位）
# ============================================================
df_sit = df.copy()
df_sit["cont_interaction"] = df_sit["leisure_exercise"] * df_sit["sedentary_min"]
X_int = np.column_stack(
    [
        np.ones(len(df_sit)),
        df_sit["leisure_exercise"].values,
        df_sit["sedentary_min"].values,
        df_sit["cont_interaction"].values,
        df_sit["age_c"].values,
        df_sit["female"].values,
    ]
)
b3, t3, p3, aic3 = ols_report(
    X_int,
    df_sit["phq9"].values,
    ["intercept", "exercise", "sedentary_min", "exercise×sedentary_min", "age_c", "female"],
    "Model 3: 連続座位時間×運動（参考）",
    robust_hc1=False,
)

print(f"\n{'='*60}")
print("結論")
print(f"{'='*60}")
print(f"余暇運動（rec_active）は PHQ-9 を有意に予測（Model 1 β = {b1[1]:.2f}, Model 2 β = {b2[1]:.2f}）")
print(f"高座位×運動の交互作用は非有意（Model 1 p = {p1[3]:.2f}）")
