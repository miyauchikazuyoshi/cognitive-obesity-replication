import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from numpy.linalg import lstsq
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

def adf_t(y):
    y = np.array(y)
    if len(y) < 8:
        return np.nan
    dy = np.diff(y)
    y_lag = y[:-1]
    n = len(dy)
    X = np.column_stack([np.ones(n), y_lag])
    b = lstsq(X, dy, rcond=None)[0]
    r = dy - X @ b
    se = np.sqrt(np.sum(r**2)/(n-2) * np.linalg.inv(X.T @ X)[1,1])
    return b[1] / se if se > 0 else np.nan

print("=" * 60)
print("1. PANEL STATIONARITY (ADF per country)")
print("=" * 60)

for var in ['depression_prevalence', 'intake_ratio']:
    ts = []
    for c, g in df.groupby('country'):
        s = g[var].dropna()
        if len(s) >= 10:
            t = adf_t(s.values)
            if not np.isnan(t):
                ts.append(t)
    mean_t = np.mean(ts)
    n_c = len(ts)
    reject = sum(1 for t in ts if t < -2.86)
    print("%s (N=%d):" % (var, n_c))
    print("  Mean ADF t: %.3f" % mean_t)
    print("  Reject unit root 5%%: %d/%d (%.0f%%)" % (reject, n_c, 100*reject/n_c))

print()
print("=" * 60)
print("2. FIRST-DIFFERENCE GRANGER")
print("=" * 60)

df['d_dep'] = df.groupby('country')['depression_prevalence'].diff()
df['d_ratio'] = df.groupby('country')['intake_ratio'].diff()
df['d_ratio_lag'] = df.groupby('country')['d_ratio'].shift(1)
df['d_dep_lag'] = df.groupby('country')['d_dep'].shift(1)

fd = df.dropna(subset=['d_dep', 'd_ratio', 'd_ratio_lag', 'd_dep_lag']).copy()
print("N = %d" % len(fd))

Y = fd['d_dep'].values
X = np.column_stack([fd['d_dep_lag'].values, fd['d_ratio_lag'].values])
b = lstsq(X, Y, rcond=None)[0]
r = Y - X @ b
s2 = np.sum(r**2) / (len(Y) - 2)
se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1,1])
t_fwd = b[1] / se
print("Forward d_R(t-1)->d_D(t): beta=%.6f, t=%.4f" % (b[1], t_fwd))

Y2 = fd['d_ratio'].values
X2 = np.column_stack([fd['d_ratio_lag'].values, fd['d_dep_lag'].values])
b2 = lstsq(X2, Y2, rcond=None)[0]
r2 = Y2 - X2 @ b2
s2_2 = np.sum(r2**2) / (len(Y2) - 2)
se2 = np.sqrt(s2_2 * np.linalg.inv(X2.T @ X2)[1,1])
t_rev = b2[1] / se2
print("Reverse d_D(t-1)->d_R(t): beta=%.6f, t=%.4f" % (b2[1], t_rev))

print()
print("=" * 60)
print("3. COMPONENTS vs RATIO")
print("=" * 60)

df['ratio_lag1'] = df.groupby('country')['intake_ratio'].shift(1)
df['dep_lag1'] = df.groupby('country')['depression_prevalence'].shift(1)
df['inet_lag1'] = df.groupby('country')['internet'].shift(1)
df['edu_lag1'] = df.groupby('country')['education'].shift(1)

de = df.dropna(subset=['depression_prevalence','dep_lag1','inet_lag1','edu_lag1','ratio_lag1']).copy()
print("N=%d, countries=%d" % (len(de), de['country'].nunique()))

for v in ['depression_prevalence','dep_lag1','inet_lag1','edu_lag1','ratio_lag1']:
    de[v+'_dm'] = de.groupby('country')[v].transform(lambda x: x - x.mean())

Y = de['depression_prevalence_dm'].values
n = len(Y)
k = de['country'].nunique()
SS_tot = np.sum((Y - Y.mean())**2)

def fit(X, Y, names, label):
    b = lstsq(X, Y, rcond=None)[0]
    r = Y - X @ b
    SS = np.sum(r**2)
    R2 = 1 - SS / SS_tot
    s2 = SS / max(len(Y) - k - len(names), 1)
    Xi = np.linalg.inv(X.T @ X)
    print("%s (R2=%.4f):" % (label, R2))
    for j, nm in enumerate(names):
        se = np.sqrt(s2 * Xi[j,j])
        t = b[j] / se
        sig = '***' if abs(t)>3.3 else ('*' if abs(t)>1.96 else 'n.s.')
        print("  %s: b=%.6f t=%.3f %s" % (nm, b[j], t, sig))
    return SS, R2

Xa = np.column_stack([de['dep_lag1_dm'].values, de['inet_lag1_dm'].values, de['edu_lag1_dm'].values])
SS_a, R2_a = fit(Xa, Y, ['D_lag','Internet_lag','Education_lag'], 'Components')

Xb = np.column_stack([de['dep_lag1_dm'].values, de['ratio_lag1_dm'].values])
SS_b, R2_b = fit(Xb, Y, ['D_lag','Ratio_lag'], 'Ratio')

Xc = np.column_stack([de['dep_lag1_dm'].values, de['inet_lag1_dm'].values, de['edu_lag1_dm'].values, de['ratio_lag1_dm'].values])
SS_c, R2_c = fit(Xc, Y, ['D_lag','Internet_lag','Education_lag','Ratio_lag'], 'Full')

aic_a = n*np.log(SS_a/n) + 2*3
aic_b = n*np.log(SS_b/n) + 2*2
aic_c = n*np.log(SS_c/n) + 2*4
print("\nAIC: Comp=%.1f Ratio=%.1f Full=%.1f" % (aic_a, aic_b, aic_c))

F_ratio = ((SS_a - SS_c) / 1) / (SS_c / max(n-k-4, 1))
F_comp = ((SS_b - SS_c) / 2) / (SS_c / max(n-k-4, 1))
print("F ratio beyond comp: %.3f" % F_ratio)
print("F comp beyond ratio: %.3f" % F_comp)

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print("First-diff: Fwd t=%.2f (%s), Rev t=%.2f (%s)" % (
    t_fwd, '+' if t_fwd>0 else '-', t_rev, '+' if t_rev>0 else '-'))
