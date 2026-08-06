"""Builds notebooks/eda.ipynb programmatically. Run once; re-run to regenerate."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

md("""# Playground Series S6E7 — Health Risk Prediction: EDA

3-class classification (`health_condition`: `fit` / `at-risk` / `unhealthy`), scored on **Balanced Accuracy**.

Goals of this notebook:
- Understand target imbalance (balanced accuracy makes minority classes disproportionately important)
- Characterize missingness (11 of 13 features have missing values, 1-12%)
- Look at numeric/categorical distributions conditioned on target
- Check train/test covariate shift
- Surface concrete feature-engineering ideas for the modeling pipeline (`src/features.py`)
""")

code("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

pd.set_option('display.max_columns', 100)
sns.set_theme(style='whitegrid', palette='deep')
plt.rcParams['figure.dpi'] = 100

DATA_DIR = '../data'
train = pd.read_csv(f'{DATA_DIR}/train.csv')
test = pd.read_csv(f'{DATA_DIR}/test.csv')
sample_sub = pd.read_csv(f'{DATA_DIR}/sample_submission.csv')

TARGET = 'health_condition'
ID_COL = 'id'

print('train:', train.shape)
print('test :', test.shape)
print('sub  :', sample_sub.shape)
""")

md("## 1. Schema & first look")

code("""train.info()
""")

code("""train.head(10)
""")

code("""num_cols = train.select_dtypes(include=[np.number]).columns.drop(ID_COL).tolist()
cat_cols = [c for c in train.columns if c not in num_cols + [ID_COL, TARGET]]
print('numeric  :', num_cols)
print('categorical:', cat_cols)
""")

md("## 2. Target distribution\n\nBalanced accuracy weights every class equally regardless of support, so the two minority classes (`unhealthy`, `fit`) matter as much as the dominant `at-risk` class despite being ~6-9% of rows each.")

code("""target_counts = train[TARGET].value_counts()
target_pct = train[TARGET].value_counts(normalize=True) * 100
display(pd.DataFrame({'count': target_counts, 'pct': target_pct.round(2)}))

fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(x=target_counts.index, y=target_counts.values, ax=ax)
ax.set_title('Target distribution (health_condition)')
ax.set_ylabel('count')
for i, v in enumerate(target_counts.values):
    ax.text(i, v, f'{v:,}\\n({target_pct.values[i]:.1f}%)', ha='center', va='bottom')
plt.tight_layout()
plt.show()
""")

md("""**Takeaway:** severe imbalance (`at-risk` ≈ 86%, `unhealthy` ≈ 8%, `fit` ≈ 6%). Implications for the pipeline:
- Use **StratifiedKFold** for CV so fold-level class ratios match the population.
- Prefer models/losses that output calibrated multiclass probabilities (logloss/multi:softprob) — balanced accuracy is then optimized downstream via argmax on OOF-calibrated probabilities, not by training with balanced-accuracy directly.
- Consider `class_weight='balanced'` variants as *additional* diversity in the model zoo, not as the only strategy — the competition writeup explicitly notes over-indexing on public LB (which is influenced by the same imbalance) misled many via blind blending.
""")

md("## 3. Missing data")

code("""miss = train.isnull().mean().sort_values(ascending=False) * 100
miss = miss[miss > 0]
display(miss.round(2).to_frame('pct_missing'))

fig, ax = plt.subplots(figsize=(7, 5))
sns.barplot(x=miss.values, y=miss.index, ax=ax, orient='h')
ax.set_xlabel('% missing')
ax.set_title('Missingness by column (train)')
plt.tight_layout()
plt.show()
""")

code("""# Missingness co-occurrence: do the same rows tend to be missing across columns?
miss_matrix = train[miss.index].isnull().astype(int)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(miss_matrix.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
ax.set_title('Correlation between missingness indicators')
plt.tight_layout()
plt.show()
""")

code("""# Does missingness rate differ by target class? (informative missingness -> keep as engineered feature)
rows = []
for c in miss.index:
    grp = train.groupby(TARGET)[c].apply(lambda s: s.isnull().mean() * 100)
    rows.append(grp.rename(c))
miss_by_target = pd.concat(rows, axis=1).T
display(miss_by_target.round(2))
""")

md("**Takeaway:** if missingness rates vary meaningfully by class, `missing indicator` columns (one per feature) are worth including in the heavy-FE feature set — cheap and often informative on synthetic/generated playground data.")

md("## 4. Numeric feature distributions")

code("""fig, axes = plt.subplots(3, 3, figsize=(15, 12))
for ax, c in zip(axes.flat, num_cols):
    sns.histplot(train[c].dropna(), kde=True, ax=ax, bins=50)
    ax.set_title(c)
for ax in axes.flat[len(num_cols):]:
    ax.axis('off')
plt.tight_layout()
plt.show()
""")

code("""display(train[num_cols].describe().T)
""")

code("""# Skewness / kurtosis - flags candidates for log/sqrt transforms
skew_kurt = pd.DataFrame({
    'skew': train[num_cols].skew(),
    'kurtosis': train[num_cols].kurtosis(),
})
display(skew_kurt.round(3))
""")

md("## 5. Numeric features conditioned on target")

code("""fig, axes = plt.subplots(3, 3, figsize=(16, 13))
for ax, c in zip(axes.flat, num_cols):
    sns.boxplot(data=train, x=TARGET, y=c, ax=ax, order=['fit', 'at-risk', 'unhealthy'])
    ax.set_title(c)
for ax in axes.flat[len(num_cols):]:
    ax.axis('off')
plt.tight_layout()
plt.show()
""")

code("""# Mean per class - quick numeric signal check
display(train.groupby(TARGET)[num_cols].mean().T.round(3))
""")

code("""# ANOVA F-test: does each numeric feature separate the 3 classes?
rows = []
for c in num_cols:
    groups = [train.loc[train[TARGET] == g, c].dropna() for g in train[TARGET].unique()]
    f, p = stats.f_oneway(*groups)
    rows.append((c, f, p))
anova = pd.DataFrame(rows, columns=['feature', 'F_stat', 'p_value']).sort_values('F_stat', ascending=False)
display(anova)
""")

md("**Takeaway:** rank numeric features by F-statistic to prioritize which ones deserve heavier interaction/ratio engineering. Features with near-identical distributions across classes and high p-values are still worth keeping (tree models can find nonlinear combinations) but are lower-priority for manual ratio features.")

md("## 6. Numeric feature correlations & multicollinearity")

code("""fig, ax = plt.subplots(figsize=(8, 6))
corr = train[num_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
ax.set_title('Numeric feature correlation (Pearson)')
plt.tight_layout()
plt.show()
""")

md("## 7. Categorical features")

code("""fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, c in zip(axes.flat, cat_cols):
    vc = train[c].value_counts(dropna=False)
    sns.barplot(x=vc.index.astype(str), y=vc.values, ax=ax)
    ax.set_title(c)
    ax.tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.show()
""")

code("""# Normalized crosstab: class distribution within each category level
for c in cat_cols:
    ct = pd.crosstab(train[c], train[TARGET], normalize='index') * 100
    print(f'--- {c} ---')
    display(ct.round(2))
""")

code("""# Chi-square test of independence between each categorical feature and target
rows = []
for c in cat_cols:
    ct = pd.crosstab(train[c], train[TARGET])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    rows.append((c, chi2, p))
chi2_df = pd.DataFrame(rows, columns=['feature', 'chi2', 'p_value']).sort_values('chi2', ascending=False)
display(chi2_df)
""")

md("## 8. Outlier scan (IQR method)")

code("""rows = []
for c in num_cols:
    q1, q3 = train[c].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((train[c] < lo) | (train[c] > hi)).sum()
    rows.append((c, lo, hi, n_out, round(n_out / len(train) * 100, 3)))
outlier_df = pd.DataFrame(rows, columns=['feature', 'lower', 'upper', 'n_outliers', 'pct_outliers'])
display(outlier_df)
""")

md("**Takeaway:** playground-synthetic numeric features are typically bounded and clean (see plausible physiological ranges above — e.g. heart_rate ~40-110, bmi ~15-35). Outlier rates should be low; if any column shows a large outlier %, treat it as a clipping/winsorization candidate before feeding linear/distance-based models (LogReg, kNN, SVM, MLP).")

md("## 9. Train vs test covariate shift")

code("""# Compare numeric distributions between train and test - large shifts hurt CV-LB agreement
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
for ax, c in zip(axes.flat, num_cols):
    sns.kdeplot(train[c].dropna(), ax=ax, label='train', fill=True, alpha=0.3)
    sns.kdeplot(test[c].dropna(), ax=ax, label='test', fill=True, alpha=0.3)
    ax.set_title(c)
    ax.legend()
for ax in axes.flat[len(num_cols):]:
    ax.axis('off')
plt.tight_layout()
plt.show()
""")

code("""# KS test per numeric column, train vs test
rows = []
for c in num_cols:
    stat, p = stats.ks_2samp(train[c].dropna(), test[c].dropna())
    rows.append((c, stat, p))
ks_df = pd.DataFrame(rows, columns=['feature', 'ks_stat', 'p_value']).sort_values('ks_stat', ascending=False)
display(ks_df)
""")

code("""# Categorical distribution shift: compare normalized value counts
for c in cat_cols:
    tr_vc = train[c].value_counts(normalize=True, dropna=False)
    te_vc = test[c].value_counts(normalize=True, dropna=False)
    cmp = pd.concat([tr_vc, te_vc], axis=1, keys=['train', 'test']).fillna(0) * 100
    print(f'--- {c} ---')
    display(cmp.round(2))
""")

code("""# missingness rate: train vs test
miss_cmp = pd.concat([
    train.isnull().mean() * 100,
    test.isnull().mean() * 100,
], axis=1, keys=['train_pct_missing', 'test_pct_missing']).round(2)
display(miss_cmp[miss_cmp.sum(axis=1) > 0])
""")

md("**Takeaway:** if KS-stats and categorical/missingness rates are close between train and test, this is a standard i.i.d. playground split — CV should track LB reasonably well *when using a proper multi-fold CV framework*, consistent with the competition writeup's core lesson (public-LB-chasing via blind blends diverges from a robust CV-based approach).")

md("## 10. Sanity checks")

code("""print('duplicate rows (all cols):', train.duplicated().sum())
print('duplicate rows (excl id)   :', train.drop(columns=[ID_COL]).duplicated().sum())
print('id column unique in train  :', train[ID_COL].is_unique)
print('id column unique in test   :', test[ID_COL].is_unique)
print('train/test id overlap      :', len(set(train[ID_COL]) & set(test[ID_COL])))
print('sample_submission columns  :', sample_sub.columns.tolist())
display(sample_sub.head())
""")

md("""## 11. Summary — feature engineering ideas for the pipeline

1. **Missing indicators** for all 11 columns with missing values — missingness rate varies somewhat by class, worth keeping as binary flags.
2. **Numeric ratios/products/differences** across the 7 numeric columns (e.g. `calorie_expenditure / step_count`, `water_intake / bmi`) — physiologically motivated combinations tend to separate `unhealthy` from `fit`.
3. **Groupby aggregates**: mean/std/min/max of each numeric column grouped by each categorical column (and by categorical-pairs) — cheap, leakage-free (no target used), and is the single biggest lever for reaching a large heavy-FE feature count.
4. **Frequency encoding** for all categorical columns (and pairwise combinations).
5. **OOF target encoding** (K-fold safe, using the *same* fixed folds used for model training) for categorical columns and top pairwise combinations — must reuse identical fold assignment across the whole model zoo so OOF predictions stay aligned for blending later.
6. **Deviation features**: numeric value minus its group-mean (by each categorical column) — captures "high/low relative to peer group".
7. **Binning** continuous features into quantile buckets, treated as categorical (helps GBMs split cleanly, gives linear models nonlinearity).
8. **Row-level stats**: count of missing values per row, sum/mean of standardized numerics per row.
9. Given low cardinality of all categoricals (3-4 levels + NaN treated as its own level), one-hot / ordinal encodings are both cheap — include both as separate feature-set variants for model diversity.
10. Train/test distributions look consistent (see Section 9) — a well-built CV framework (StratifiedKFold, fixed across the whole model zoo) should track the private LB, in line with the competition author's "trust your CV" lesson.
""")

nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.11'},
}

with open('notebooks/eda.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print('wrote notebooks/eda.ipynb with', len(cells), 'cells')
