import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import os
import sys

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "macro")


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
    print("See data/README_data.md for data acquisition/assembly instructions.")
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

    return df.rename(columns=rename)


df = normalize_columns(load_panel())

if "intake_ratio" not in df.columns:
    if "internet" not in df.columns or "education" not in df.columns:
        print("ERROR: 'intake_ratio' missing and cannot be constructed (need internet & education).")
        print("Available columns:", ", ".join(map(str, df.columns)))
        sys.exit(1)

    internet = pd.to_numeric(df["internet"], errors="coerce")
    education = pd.to_numeric(df["education"], errors="coerce")
    internet_for_ratio = internet
    if education.max(skipna=True) <= 1.5 and internet.max(skipna=True) > 1.5:
        internet_for_ratio = internet / 100.0
    df["intake_ratio"] = internet_for_ratio / education

df = df.sort_values(['country', 'year'])
df["ratio_lag1"] = df.groupby("country")["intake_ratio"].shift(1)

# ============================================================
# 1. STATIONARITY: Im-Pesaran-Shin style panel unit root
# We use ADF on each country, then combine p-values
# ============================================================
from scipy import stats

def adf_test_simple(series):
    """Simple ADF: regress dy on y_{t-1} and dy_{t-1}"""
    y = series.dropna().values
    if len(y) < 8:
        return np.nan
    dy = np.diff(y)
    y_lag = y[:-1]
    n = len(dy)
    if n < 5:
        return np.nan
    # dy_t = alpha + beta*y_{t-1} + eps
    X = np.column_stack([np.ones(n), y_lag[:-1] if n > 1 else y_lag])
    # Actually simpler: just test correlation of dy with y_lag
    X = np.column_stack([np.ones(len(dy)), y_lag[:len(dy)]])
    try:
        beta = np.linalg.lstsq(X, dy, rcond=None)[0]
        resid = dy - X @ beta
        se = np.sqrt(np.sum(resid**2) / (len(dy) - 2) / np.sum((y_lag[:len(dy)] - y_lag[:len(dy)].mean())**2))
        t_stat = beta[1] / se
        return t_stat
    except:
        return np.nan

print("="*60)
print("PANEL STATIONARITY TESTS")
print("="*60)

for var_name in ['depression_prevalence', 'intake_ratio', 'internet', 'education']:
    if var_name not in df.columns:
        continue
    t_stats = []
    for country, grp in df.groupby('country'):
        series = grp[var_name].dropna()
        if len(series) >= 10:
            t = adf_test_simple(series)
            if not np.isnan(t):
                t_stats.append(t)
    
    if t_stats:
        # IPS test: average t-stat, standardize
        mean_t = np.mean(t_stats)
        n_countries = len(t_stats)
        # Under H0, E[ADF_t] approx -1.5, Var approx 1.0 for T~20
        z = np.sqrt(n_countries) * (mean_t - (-1.52)) / 0.74  # approximate critical values
        print("\n%s:" % var_name)
        print("  N countries: %d" % n_countries)
        print("  Mean ADF t-stat: %.3f" % mean_t)
        print("  IPS Z-stat: %.3f" % z)
        # How many individual countries reject H0 at 5%?
        reject_count = sum(1 for t in t_stats if t < -2.86)  # ADF 5% critical value ~-2.86
        print("  Countries rejecting unit root (5%%): %d / %d (%.1f%%)" % (reject_count, n_countries, 100*reject_count/n_countries))

# ============================================================
# 2. FIRST-DIFFERENCE ROBUSTNESS
# ============================================================
print("\n" + "="*60)
print("FIRST-DIFFERENCE ROBUSTNESS CHECK")
print("="*60)

df['d_dep'] = df.groupby('country')['depression_prevalence'].diff()
df['d_ratio'] = df.groupby('country')['intake_ratio'].diff()
df['d_ratio_lag1'] = df.groupby('country')['d_ratio'].shift(1)
df['d_dep_lag1'] = df.groupby('country')['d_dep'].shift(1)

df_fd = df.dropna(subset=['d_dep', 'd_ratio_lag1', 'd_dep_lag1']).copy()
print("N obs (first-diff): %d" % len(df_fd))

from numpy.linalg import lstsq

# Forward: d_D(t) = a*d_D(t-1) + b*d_R(t-1)
Y = df_fd['d_dep'].values
X = np.column_stack([df_fd['d_dep_lag1'].values, df_fd['d_ratio_lag1'].values])
beta = lstsq(X, Y, rcond=None)[0]
resid = Y - X @ beta
n = len(Y)
sigma2 = np.sum(resid**2) / (n - 2)
XtX_inv = np.linalg.inv(X.T @ X)
se_r = np.sqrt(sigma2 * XtX_inv[1, 1])
t_r = beta[1] / se_r
print("\nForward (first-diff): d_R(t-1) -> d_D(t)")
print("  beta = %.6f, SE = %.6f, t = %.4f" % (beta[1], se_r, t_r))

# Reverse: d_R(t) = a*d_R(t-1) + b*d_D(t-1)
df['d_ratio_curr'] = df.groupby('country')['intake_ratio'].diff()
df_fd2 = df.dropna(subset=['d_ratio_curr', 'd_ratio_lag1', 'd_dep_lag1']).copy()
Y2 = df_fd2['d_ratio_curr'].values
X2 = np.column_stack([df_fd2['d_ratio_lag1'].values, df_fd2['d_dep_lag1'].values])
beta2 = lstsq(X2, Y2, rcond=None)[0]
resid2 = Y2 - X2 @ beta2
sigma2_2 = np.sum(resid2**2) / (len(Y2) - 2)
XtX_inv2 = np.linalg.inv(X2.T @ X2)
se2 = np.sqrt(sigma2_2 * XtX_inv2[1, 1])
t2 = beta2[1] / se2
print("\nReverse (first-diff): d_D(t-1) -> d_R(t)")
print("  beta = %.6f, SE = %.6f, t = %.4f" % (beta2[1], se2, t2))

# ============================================================
# 3. ENDOGENEITY: Internet and Education separately vs ratio
# ============================================================
print("\n" + "="*60)
print("ENDOGENEITY: RATIO vs COMPONENTS")
print("="*60)

df['internet_lag1'] = df.groupby('country')['internet'].shift(1)
df['education_lag1'] = df.groupby('country')['education'].shift(1)
df['dep_lag1'] = df.groupby('country')['depression_prevalence'].shift(1)

df_endo = df.dropna(subset=['depression_prevalence', 'dep_lag1', 'internet_lag1', 'education_lag1', 'ratio_lag1']).copy()

# Demean
for v in ['depression_prevalence', 'dep_lag1', 'internet_lag1', 'education_lag1', 'ratio_lag1']:
    df_endo[v + '_dm'] = df_endo.groupby('country')[v].transform(lambda x: x - x.mean())

# Model A: D(t) ~ D(t-1) + Internet(t-1) + Education(t-1) + FE
Y = df_endo['depression_prevalence_dm'].values
Xa = np.column_stack([
    df_endo['dep_lag1_dm'].values,
    df_endo['internet_lag1_dm'].values,
    df_endo['education_lag1_dm'].values,
])
beta_a = lstsq(Xa, Y, rcond=None)[0]
pred_a = Xa @ beta_a
SS_a = np.sum((Y - pred_a)**2)
SS_tot = np.sum((Y - Y.mean())**2)
R2_a = 1 - SS_a / SS_tot
n = len(Y)
k = df_endo['country'].nunique()

resid_a = Y - pred_a
sigma2_a = np.sum(resid_a**2) / (n - k - 3)
XtX_inv_a = np.linalg.inv(Xa.T @ Xa)
for j, name in enumerate(['D(t-1)', 'Internet(t-1)', 'Education(t-1)']):
    se = np.sqrt(sigma2_a * XtX_inv_a[j, j])
    t = beta_a[j] / se
    print("Components model: %s: beta=%.6f, t=%.4f" % (name, beta_a[j], t))
print("  R2 = %.4f" % R2_a)

# Model B: D(t) ~ D(t-1) + Ratio(t-1) + FE
df_endo['ratio_lag1_dm'] = df_endo.groupby('country')['ratio_lag1'].transform(lambda x: x - x.mean())
Xb = np.column_stack([
    df_endo['dep_lag1_dm'].values,
    df_endo['ratio_lag1_dm'].values,
])
beta_b = lstsq(Xb, Y, rcond=None)[0]
SS_b = np.sum((Y - Xb @ beta_b)**2)
R2_b = 1 - SS_b / SS_tot

resid_b = Y - Xb @ beta_b
sigma2_b = np.sum(resid_b**2) / (n - k - 2)
XtX_inv_b = np.linalg.inv(Xb.T @ Xb)
for j, name in enumerate(['D(t-1)', 'Ratio(t-1)']):
    se = np.sqrt(sigma2_b * XtX_inv_b[j, j])
    t = beta_b[j] / se
    print("Ratio model: %s: beta=%.6f, t=%.4f" % (name, beta_b[j], t))
print("  R2 = %.4f" % R2_b)

# AIC comparison
aic_a = n * np.log(SS_a / n) + 2 * (3 + k)
aic_b = n * np.log(SS_b / n) + 2 * (2 + k)
print("\nAIC components: %.1f" % aic_a)
print("AIC ratio: %.1f" % aic_b)
print("Delta AIC: %.1f (negative = ratio better)" % (aic_b - aic_a))

# Model C: D(t) ~ D(t-1) + Ratio(t-1) + Internet(t-1) + Education(t-1) + FE
# Does ratio add beyond components?
Xc = np.column_stack([
    df_endo['dep_lag1_dm'].values,
    df_endo['internet_lag1_dm'].values,
    df_endo['education_lag1_dm'].values,
    df_endo['ratio_lag1_dm'].values,
])
beta_c = lstsq(Xc, Y, rcond=None)[0]
SS_c = np.sum((Y - Xc @ beta_c)**2)
R2_c = 1 - SS_c / SS_tot

resid_c = Y - Xc @ beta_c
sigma2_c = np.sum(resid_c**2) / (n - k - 4)
XtX_inv_c = np.linalg.inv(Xc.T @ Xc)
for j, name in enumerate(['D(t-1)', 'Internet(t-1)', 'Education(t-1)', 'Ratio(t-1)']):
    se = np.sqrt(sigma2_c * XtX_inv_c[j, j])
    t = beta_c[j] / se
    print("Full model: %s: beta=%.6f, t=%.4f" % (name, beta_c[j], t))
print("  R2 = %.4f" % R2_c)

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("1. Stationarity: Check ADF results above")
print("2. First-diff Granger: Forward t=%.2f, Reverse t=%.2f" % (t_r, t2))
print("3. Components vs Ratio: R2_components=%.4f, R2_ratio=%.4f" % (R2_a, R2_b))
