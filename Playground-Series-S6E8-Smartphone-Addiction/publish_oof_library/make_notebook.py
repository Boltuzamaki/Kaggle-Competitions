"""Build the public notebook for the 47-stream own-model stack."""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kaggle": {"accelerator": "none", "dataSources": [], "dockerImageVersionId": None},
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}

cells = []

cells.append(nbf.v4.new_markdown_cell("""# Stacking 47 Models | Public LB 0.97073

This notebook combines out-of-fold predictions from 47 models into a single
submission that scores **0.97073** on the public leaderboard.

The base models were trained separately on the official competition data, across six
families: XGBoost, LightGBM, CatBoost, an exact-value lookup transformer, several
other neural architectures, and a set of fold-safe target encoding models. Their
out-of-fold and test predictions are attached as a dataset, so this notebook covers
the stacking step only and runs in about a minute.

What it covers:

1. Rank normalisation, so models with different calibration can be compared
2. A correlation check for deciding which models are worth adding
3. Cross-fitted logistic regression for the blend weights
4. How local cross-validation maps onto the public leaderboard in this competition"""))

cells.append(nbf.v4.new_markdown_cell("""## What is in the library

Six families, all trained under honest five or ten fold out-of-fold protocols so the
predictions below are always for rows the model did not train on.

| family | streams | what it is |
| --- | --- | --- |
| other neural | 11 | TabM, RealMLP, DeepFM, DCNv2, GANDALF, TabR, FT-Transformer |
| XGBoost | 9 | nested target encoding, depth diversity, seed bags |
| LightGBM and HistGB | 8 | pair lattice, driver reconstruction, raw bags |
| exact value lookup transformer | 8 | a neural net over embeddings of exact repeated values |
| CatBoost | 6 | dual view, ordered boosting, pair evidence |
| fold-safe target encoding | 5 | added most recently, including the best single model |

Best single model is 0.9684256 out of fold. The stack reaches about 0.96975, which
scored 0.97073 on the public leaderboard. These counts are recomputed from the data
later in the notebook."""))

cells.append(nbf.v4.new_code_cell("""from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtri
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

TARGET = 'addicted_label'
SEED = 20260806


def find(name):
    for base in [Path('/kaggle/input'), Path('.'), Path('data')]:
        if not base.exists():
            continue
        hits = list(base.rglob(name))
        if hits:
            return hits[0]
    raise FileNotFoundError(name)


oof = pd.read_parquet(find('oof_predictions.parquet'))
test = pd.read_parquet(find('test_predictions.parquet'))
labels = pd.read_parquet(find('train_labels.parquet'))
index = pd.read_csv(find('stream_index.csv'))

assert oof.id.equals(labels.id)
y = labels[TARGET].to_numpy('int8')
names = [c for c in oof.columns if c != 'id']

print(f'{len(names)} streams, {len(oof)} train rows, {len(test)} test rows')
print(index.head(10).to_string(index=False))"""))

cells.append(nbf.v4.new_markdown_cell("""## Put every model on the same scale

A tree model and a neural network can rank rows almost identically and still produce
completely different numbers. One might output probabilities clustered around 0.7,
the other spread across the whole interval. A linear blend fed those raw values
spends its weights fixing calibration instead of deciding who to trust.

Converting each stream to normal scores by rank removes the problem. Only the
ordering survives, which is all AUC cares about."""))

cells.append(nbf.v4.new_code_cell("""def gauss_rank(values):
    r = pd.Series(values).rank(method='average').to_numpy()
    return ndtri(r / (len(r) + 1.0)).astype('float32')


X = np.column_stack([gauss_rank(oof[n].to_numpy()) for n in names])
X_test = np.column_stack([gauss_rank(test[n].to_numpy()) for n in names])
print('meta matrix:', X.shape)"""))

cells.append(nbf.v4.new_markdown_cell("""## Which models are actually worth having

Before looking at any blend, check how similar the models are. This is the cheapest
useful diagnostic in the whole pipeline and it is easy to skip.

Every gradient boosted model in the library sits above 0.99 rank correlation with
every other one, no matter which library trained it or how the features were built.
The lookup transformer sits around 0.976 against the same models. That difference
decides where GPU time is worth spending.

As a concrete example, one CatBoost variant here cost 26 GPU minutes and moved the
blend by 0.0000059, because it sat at 0.99 correlation with models already present.
A weaker model built on a different idea was worth considerably more."""))

cells.append(nbf.v4.new_code_cell("""probe = 'foldsafe_te_xgb' if 'foldsafe_te_xgb' in names else names[0]
base_rank = pd.Series(oof[probe].to_numpy()).rank(pct=True)

corr = []
for n in names:
    if n == probe:
        continue
    corr.append({'stream': n,
                 'corr_with_' + probe: base_rank.corr(pd.Series(oof[n].to_numpy()).rank(pct=True)),
                 'oof_auc': roc_auc_score(y, oof[n].to_numpy())})
corr = pd.DataFrame(corr).sort_values('corr_with_' + probe)

print('LEAST correlated with', probe)
print(corr.head(8).to_string(index=False))
print('\\nMOST correlated (these add the least)')
print(corr.tail(6).to_string(index=False))"""))

cells.append(nbf.v4.new_markdown_cell("""## The stack

Two choices matter here.

**Cross fitting.** The logistic regression is trained on four fifths of the rows and
scores the fifth it has not seen, rotating through all five. Fitting once on
everything and scoring the same rows gives a higher number that means nothing,
because the weights have already seen the answers.

**Unconstrained weights.** About twenty of the forty-seven weights come out negative,
which looks wrong at first. It is not. When two models are 0.99 correlated, the pair
is useful mainly through the small part where they disagree, and a negative weight on
one of them is how a linear model extracts that difference. Forcing weights to be
non negative scores 0.9689379 against 0.9697477 unconstrained, a loss of 0.0008,
which is large on a leaderboard this compressed.

**Bagging.** Refitting on resampled subsets and averaging the result is worth roughly
0.00002. Small, but free."""))

cells.append(nbf.v4.new_code_cell("""folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y))
C = 0.1

plain = np.zeros(len(y))
for fit_i, val_i in folds:
    model = LogisticRegression(C=C, max_iter=2000).fit(X[fit_i], y[fit_i])
    plain[val_i] = model.decision_function(X[val_i])

bagged = np.zeros(len(y))
for fit_i, val_i in folds:
    acc = np.zeros(len(val_i))
    for b in range(5):
        rng = np.random.default_rng(SEED + b)
        sub = rng.choice(fit_i, int(0.7 * len(fit_i)), replace=False)
        acc += LogisticRegression(C=C, max_iter=2000).fit(
            X[sub], y[sub]).decision_function(X[val_i]) / 5
    bagged[val_i] = acc

best_single = max(roc_auc_score(y, oof[n].to_numpy()) for n in names)
print(f'best single stream : {best_single:.7f}')
print(f'cross fitted stack : {roc_auc_score(y, plain):.7f}')
print(f'bagged stack       : {roc_auc_score(y, bagged):.7f}')
print(f'gain over best single: {roc_auc_score(y, bagged) - best_single:+.7f}')"""))

cells.append(nbf.v4.new_markdown_cell("""## Where the weight goes

Grouping the weights by family shows what the stack is really leaning on. Eight
lookup transformer streams carry about 21 percent of the total absolute weight, while
nine XGBoost streams carry under 12 percent between them. Models built on one good
idea beat a larger pile built on a crowded one.

The largest single weight is worth noticing. It goes to the fold-safe target encoding
model built over all 66 column pairs, which scores 0.968155 on its own and is beaten
by several streams next to it. It earns the weight by being wrong in different places
rather than by being right more often. Adding it moved the leaderboard score from
0.97064 to 0.97073."""))

cells.append(nbf.v4.new_code_cell("""final = LogisticRegression(C=C, max_iter=2000).fit(X, y)
w = final.coef_[0]

table = pd.DataFrame({'stream': names, 'weight': w,
                      'oof_auc': [roc_auc_score(y, oof[n].to_numpy()) for n in names]})
print('largest positive weights')
print(table.sort_values('weight', ascending=False).head(10).to_string(index=False))
print('\\nlargest negative weights (corrections, not bad models)')
print(table.sort_values('weight').head(6).to_string(index=False))


def family(n):
    if n.startswith('lookup'):
        return 'lookup transformer'
    if n.startswith('foldsafe'):
        return 'fold-safe TE'
    if n.startswith('xgb'):
        return 'xgboost'
    if n.startswith('cat'):
        return 'catboost'
    if n.startswith(('lgb', 'histgb', 'repr')):
        return 'lightgbm / histgb'
    return 'other neural'


table['family'] = table.stream.map(family)
share = table.assign(abs_w=table.weight.abs()).groupby('family').agg(
    streams=('stream', 'size'), share=('abs_w', 'sum'))
share['share'] = (100 * share['share'] / share['share'].sum()).round(1)
print('\\nshare of total absolute weight')
print(share.sort_values('share', ascending=False).to_string())"""))

cells.append(nbf.v4.new_markdown_cell("""## Cross validation predicts the leaderboard here

This competition has an unusually stable relationship between honest out-of-fold AUC
and the public score. Across five submissions the public score came in at the
out-of-fold number plus 0.00099, every time within 0.00002:

| stack | OOF | predicted | actual |
| --- | --- | --- | --- |
| earlier rank blend | 0.9692864 | 0.97028 | 0.97029 |
| 40 streams | 0.9696223 | 0.97061 | 0.97060 |
| 42 streams | 0.9696482 | 0.97064 | 0.97062 |
| 45 streams | 0.9696691 | 0.97066 | 0.97064 |
| 47 streams | 0.9697477 | 0.97074 | 0.97073 |

Once that relationship held twice, submissions stopped being the way to find out
whether an idea worked. Measuring offline and only submitting candidates that had
already cleared the bar is the difference between testing five ideas a day and
testing as many as there is compute for.

The same arithmetic should apply to any blend built from this library. Whatever
out-of-fold number it reaches, add 0.00099 for the expected public score."""))

cells.append(nbf.v4.new_code_cell("""test_score = final.decision_function(X_test)

submission = pd.DataFrame({'id': test.id.to_numpy(),
                           TARGET: pd.Series(test_score).rank(pct=True).to_numpy()})

assert len(submission) == len(test)
assert submission[TARGET].notna().all()
assert np.isfinite(submission[TARGET]).all()
assert submission[TARGET].between(0, 1).all()
assert submission.id.is_unique

submission.to_csv('submission.csv', index=False)
print(submission.shape)
submission.head()"""))

cells.append(nbf.v4.new_markdown_cell("""## What made the biggest difference

**Correlation matters more than score when choosing what to add.** Most of the GPU
time in this project went into models sitting at 0.99 correlation with something
already in the library. The correlation table takes seconds to produce and is a
better guide than another round of tuning.

**More folds beats more models.** Moving one XGBoost from five folds to ten was worth
0.00018, because each model then trains on 90 percent of the rows instead of 80. That
is more than an entire additional model family contributed.

**Weak and different beats strong and similar.** The largest jump in this stack came
from a model that scored worse on its own than the one beside it.

**Target-free architectures underperform on this data.** DCNv2, GANDALF, TabR and an
FT-Transformer all landed between 0.938 and 0.940, roughly 0.03 below anything with
access to target statistics. The lookup transformer works because its exact value
embeddings encode those statistics implicitly.

The library is attached to this notebook, so it can be used as a starting point for
a different blend. Comments and corrections are welcome."""))

nb["cells"] = cells
HERE.mkdir(parents=True, exist_ok=True)
nbf.write(nb, HERE / "s6e8_47_stream_stack.ipynb")
print("wrote", HERE / "s6e8_47_stream_stack.ipynb", "cells:", len(cells))
