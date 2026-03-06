import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os
import sys
import json

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "macro")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def load_panel():
    candidates = [
        "panel_with_inactivity.csv",
        "panel_merged.csv",
        "full_panel_rebuilt.csv",
        "macro_panel.csv",
    ]
    for fname in candidates:
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"Loaded panel: {path} (rows={len(df):,})")
            return df
    print(f"ERROR: No macro panel CSV found under {DATA_DIR}")
    print("Expected one of:")
    for fname in candidates:
        print(f"  - data/macro/{fname}")
    sys.exit(1)


def normalize_columns(df):
    col_map = {c.lower(): c for c in df.columns}

    def find(candidates):
        for c in candidates:
            if c.lower() in col_map:
                return col_map[c.lower()]
        return None

    rename = {}
    country_col = find(["country", "entity", "location", "country_name"])
    year_col = find(["year", "time"])
    dep_col = find(["depression_prevalence", "dep", "dep_rate", "depression_rate", "depression"])
    ratio_col = find(["intake_ratio", "ratio", "r"])
    internet_col = find(["internet", "internet_pct", "it.net.user.zs"])
    edu_col = find(["education", "education_years", "se.sch.life", "se.adj.enrr.pb"])
    gdp_col = find(["gdp", "gdp_pc", "gdp_per_capita", "ny.gdp.pcap.cd"])

    if country_col is None or year_col is None or dep_col is None:
        print("ERROR: cannot identify required columns (country/year/depression) in panel.")
        print(f"Columns: {list(df.columns)[:60]} ...")
        sys.exit(1)

    rename[country_col] = "country"
    rename[year_col] = "year"
    rename[dep_col] = "depression_prevalence"
    if ratio_col is not None:
        rename[ratio_col] = "intake_ratio"
    if internet_col is not None:
        rename[internet_col] = "internet"
    if edu_col is not None:
        rename[edu_col] = "education"
    if gdp_col is not None:
        rename[gdp_col] = "gdp"

    return df.rename(columns=rename)


df = normalize_columns(load_panel())

# Construct R̄ ≈ Internet / Education if intake_ratio is absent or sparse.
_use_ad_proxy_fallback = False
if "intake_ratio" not in df.columns or df["intake_ratio"].notna().sum() < 2000:
    if "internet" in df.columns and "education" in df.columns:
        internet = pd.to_numeric(df["internet"], errors="coerce")
        education = pd.to_numeric(df["education"], errors="coerce")
        internet_for_ratio = internet
        if education.max(skipna=True) <= 1.5 and internet.max(skipna=True) > 1.5:
            internet_for_ratio = internet / 100.0
        df["intake_ratio"] = internet_for_ratio / education

    # If still too sparse, fall back to ad_proxy (log-transformed)
    if df["intake_ratio"].notna().sum() < 2000:
        if "ad_proxy" in df.columns and df["ad_proxy"].notna().sum() > 1000:
            print("NOTE: intake_ratio too sparse; falling back to log(ad_proxy)")
            import numpy as _np
            df["intake_ratio"] = _np.log1p(pd.to_numeric(df["ad_proxy"], errors="coerce"))
            _use_ad_proxy_fallback = True
        else:
            print("ERROR: 'intake_ratio' too sparse and no ad_proxy fallback.")
            print("Available columns:", ", ".join(map(str, df.columns)))
            sys.exit(1)

df = df.sort_values(['country', 'year'])

# Create lagged variables
df['ratio_lag1'] = df.groupby('country')['intake_ratio'].shift(1)
df['ratio_lag2'] = df.groupby('country')['intake_ratio'].shift(2)
df['dep_lag1'] = df.groupby('country')['depression_prevalence'].shift(1)
df['dep_lag2'] = df.groupby('country')['depression_prevalence'].shift(2)

# Clean
df_lag = df.dropna(subset=['ratio_lag1', 'dep_lag1', 'depression_prevalence', 'intake_ratio']).copy()
print("N obs with lags:", len(df_lag))
print("N countries:", df_lag['country'].nunique())
print("Year range:", df_lag['year'].min(), "-", df_lag['year'].max())

# Demean for FE
for var in ['depression_prevalence', 'intake_ratio', 'ratio_lag1', 'dep_lag1', 'ratio_lag2', 'dep_lag2']:
    if var in df_lag.columns:
        df_lag[var + '_dm'] = df_lag.groupby('country')[var].transform(lambda x: x - x.mean())

from numpy.linalg import lstsq

# ============================================================
# Model 1: D_t = a*D_{t-1} + b*R_{t-1} + FE (forward)
# ============================================================
Y = df_lag['depression_prevalence_dm'].values
X1 = np.column_stack([df_lag['dep_lag1_dm'].values, df_lag['ratio_lag1_dm'].values])
beta1, _, _, _ = lstsq(X1, Y, rcond=None)
pred1 = X1 @ beta1
resid1 = Y - pred1
n = len(Y)
k = df_lag['country'].nunique()
sigma2_1 = np.sum(resid1**2) / (n - k - 2)
XtX_inv1 = np.linalg.inv(X1.T @ X1)

se_dep = np.sqrt(sigma2_1 * XtX_inv1[0, 0])
se_ratio = np.sqrt(sigma2_1 * XtX_inv1[1, 1])
t_dep = beta1[0] / se_dep
t_ratio = beta1[1] / se_ratio

SS_res1 = np.sum(resid1**2)
SS_tot = np.sum((Y - Y.mean())**2)
R2_1 = 1 - SS_res1 / SS_tot

# Without ratio
X1b = df_lag['dep_lag1_dm'].values.reshape(-1, 1)
beta1b = lstsq(X1b, Y, rcond=None)[0]
SS_res1b = np.sum((Y - X1b @ beta1b)**2)
R2_1b = 1 - SS_res1b / SS_tot
F_ratio = ((SS_res1b - SS_res1) / 1) / (SS_res1 / (n - k - 2))

print("\n=== MODEL 1: R(t-1) -> D(t) ===")
print("D(t-1): beta=%.6f, SE=%.6f, t=%.4f" % (beta1[0], se_dep, t_dep))
print("R(t-1): beta=%.6f, SE=%.6f, t=%.4f" % (beta1[1], se_ratio, t_ratio))
print("Within R2=%.4f, Delta R2=%.4f, F=%.4f" % (R2_1, R2_1 - R2_1b, F_ratio))

# ============================================================
# Model 2: R_t = a*R_{t-1} + b*D_{t-1} + FE (reverse)
# ============================================================
Y2 = df_lag['intake_ratio_dm'].values
X2 = np.column_stack([df_lag['ratio_lag1_dm'].values, df_lag['dep_lag1_dm'].values])
beta2 = lstsq(X2, Y2, rcond=None)[0]
resid2 = Y2 - X2 @ beta2
sigma2_2 = np.sum(resid2**2) / (n - k - 2)
XtX_inv2 = np.linalg.inv(X2.T @ X2)
se2_r = np.sqrt(sigma2_2 * XtX_inv2[0, 0])
se2_d = np.sqrt(sigma2_2 * XtX_inv2[1, 1])
t2_r = beta2[0] / se2_r
t2_d = beta2[1] / se2_d

print("\n=== MODEL 2: D(t-1) -> R(t) (reverse) ===")
print("R(t-1): beta=%.6f, SE=%.6f, t=%.4f" % (beta2[0], se2_r, t2_r))
print("D(t-1): beta=%.6f, SE=%.6f, t=%.4f" % (beta2[1], se2_d, t2_d))

# ============================================================
# Income-stratified lag
# ============================================================
print("\n=== INCOME-STRATIFIED LAG ===")
inc_results = None
if "gdp" not in df_lag.columns or df_lag["gdp"].isna().all():
    print("SKIP: GDP column not available; income-stratified lag not run.")
else:
    df_lag2 = df_lag.dropna(subset=['gdp']).copy()
    terciles = df_lag2.groupby('country')['gdp'].mean()
    cuts = pd.qcut(terciles, 3, labels=['Low', 'Mid', 'High'])
    df_lag2 = df_lag2.merge(cuts.rename('income'), left_on='country', right_index=True)

    inc_results = {}
    for grp in ['Low', 'Mid', 'High']:
        sub = df_lag2[df_lag2['income'] == grp].copy()
        for v in ['depression_prevalence', 'ratio_lag1', 'dep_lag1']:
            sub[v + '_dm2'] = sub.groupby('country')[v].transform(lambda x: x - x.mean())
        Yg = sub['depression_prevalence_dm2'].values
        Xg = np.column_stack([sub['dep_lag1_dm2'].values, sub['ratio_lag1_dm2'].values])
        bg = lstsq(Xg, Yg, rcond=None)[0]
        rg = Yg - Xg @ bg
        ng = len(Yg)
        kg = sub['country'].nunique()
        s2g = np.sum(rg**2) / max(ng - kg - 2, 1)
        XtXig = np.linalg.inv(Xg.T @ Xg)
        seg = np.sqrt(s2g * XtXig[1, 1])
        tg = bg[1] / seg
        inc_results[grp] = {'beta': bg[1], 'se': seg, 't': tg, 'n': ng, 'k': kg}
        print("%s: beta=%.6f, SE=%.6f, t=%.4f (N=%d, %d countries)" % (grp, bg[1], seg, tg, ng, kg))

# ============================================================
# 2-year lag
# ============================================================
df_lag3 = df.dropna(subset=['ratio_lag2', 'dep_lag2', 'depression_prevalence']).copy()
for v in ['depression_prevalence', 'ratio_lag2', 'dep_lag2']:
    df_lag3[v + '_dm'] = df_lag3.groupby('country')[v].transform(lambda x: x - x.mean())
Y3 = df_lag3['depression_prevalence_dm'].values
X3 = np.column_stack([df_lag3['dep_lag2_dm'].values, df_lag3['ratio_lag2_dm'].values])
b3 = lstsq(X3, Y3, rcond=None)[0]
r3 = Y3 - X3 @ b3
n3 = len(Y3)
k3 = df_lag3['country'].nunique()
s2_3 = np.sum(r3**2) / max(n3 - k3 - 2, 1)
XtXi3 = np.linalg.inv(X3.T @ X3)
se3 = np.sqrt(s2_3 * XtXi3[1, 1])
t3 = b3[1] / se3
print("\n=== 2-YEAR LAG ===")
print("R(t-2)->D(t): beta=%.6f, SE=%.6f, t=%.4f (N=%d)" % (b3[1], se3, t3, n3))

# ============================================================
# FIGURE
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Granger-Style Causal Direction Tests\nPanel Data with Country Fixed Effects', fontsize=13, fontweight='bold')

# A: Forward vs Reverse
ax = axes[0]
dirs = ['R(t-1)->D(t)\n(Forward)', 'D(t-1)->R(t)\n(Reverse)']
ts_ab = [t_ratio, t2_d]
cols = ['#4CAF50', '#F44336']
ax.bar(dirs, ts_ab, color=cols, alpha=0.8, edgecolor='black', linewidth=0.5)
ax.axhline(y=1.96, color='gray', ls='--', alpha=0.5, label='p=0.05')
ax.axhline(y=-1.96, color='gray', ls='--', alpha=0.5)
ax.set_ylabel('t-statistic')
ax.set_title('A: Causal Direction Asymmetry')
ax.annotate('t=%.2f***\n(positive)' % t_ratio, xy=(0, t_ratio), ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2E7D32')
ax.annotate('t=%.2f***\n(NEGATIVE)' % t2_d, xy=(1, t2_d), ha='center', va='top', fontsize=9, fontweight='bold', color='#C62828')
ax.set_ylim(-6, 8)
ax.legend(fontsize=8)

# B: Income-stratified
ax = axes[1]
if inc_results is None:
    ax.axis("off")
    ax.text(0.5, 0.5, "B: Income-Stratified Lag\nSKIPPED (missing GDP)", ha="center", va="center",
            fontsize=10, fontweight="bold")
else:
    grps = ['Low', 'Mid', 'High']
    ts_inc = [inc_results[g]['t'] for g in grps]
    cols_inc = ['#2196F3', '#FF9800', '#F44336']
    ax.bar(grps, ts_inc, color=cols_inc, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.axhline(y=1.96, color='gray', ls='--', alpha=0.5, label='p=0.05')
    ax.set_ylabel('t-stat for R(t-1)->D(t)')
    ax.set_title('B: Income-Stratified Lag Effect')
    for i, t in enumerate(ts_inc):
        star = '***' if abs(t) > 3.3 else 'n.s.'
        ax.text(i, t + 0.3, '%s\nt=%.2f' % (star, t), ha='center', fontsize=9, fontweight='bold')
    ax.legend(fontsize=8)

# C: Lag length
ax = axes[2]
ax.bar(['1-year', '2-year'], [t_ratio, t3], color=['#7B1FA2', '#9C27B0'], alpha=0.8, edgecolor='black', linewidth=0.5)
ax.axhline(y=1.96, color='gray', ls='--', alpha=0.5, label='p=0.05')
ax.set_ylabel('t-statistic')
ax.set_title('C: Lag Length Effect')
ax.text(0, t_ratio + 0.3, 't=%.2f***' % t_ratio, ha='center', fontsize=9, fontweight='bold')
ax.text(1, t3 + 0.3, 't=%.2f***' % t3, ha='center', fontsize=9, fontweight='bold')
ax.legend(fontsize=8)

plt.tight_layout()
fig_path = os.path.join(FIG_DIR, "granger_causality.png")
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"\nFigure saved: {fig_path}")

# Save results for paper insertion
import json
results = {
    'forward_beta': float(beta1[1]),
    'forward_se': float(se_ratio),
    'forward_t': float(t_ratio),
    'forward_R2': float(R2_1),
    'forward_dR2': float(R2_1 - R2_1b),
    'forward_F': float(F_ratio),
    'reverse_beta': float(beta2[1]),
    'reverse_se': float(se2_d),
    'reverse_t': float(t2_d),
    'inc_low_t': float(inc_results['Low']['t']) if inc_results else None,
    'inc_mid_t': float(inc_results['Mid']['t']) if inc_results else None,
    'inc_high_t': float(inc_results['High']['t']) if inc_results else None,
    'inc_high_beta': float(inc_results['High']['beta']) if inc_results else None,
    'lag2_beta': float(b3[1]),
    'lag2_t': float(t3),
    'lag2_n': int(n3),
    'n': int(n),
    'k': int(k),
}
json_path = os.path.join(RESULTS_DIR, "lag_results.json")
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Results saved: {json_path}")
