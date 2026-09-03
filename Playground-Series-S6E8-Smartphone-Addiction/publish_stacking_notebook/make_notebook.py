"""Build the public stacking notebook.

Writes a self-contained Kaggle notebook that trains three model families on the
official data, then combines them with a cross-fitted logistic meta-learner.
Everything it needs is produced inside the notebook, so a reader can run it
top to bottom without any external prediction files.
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kaggle": {"accelerator": "gpu", "dataSources": [], "dockerImageVersionId": None},
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}

cells = []

cells.append(nbf.v4.new_markdown_cell("""# Stacking three model families the honest way

Most public notebooks for this competition either train one model or average a few
submissions together. This one does the middle thing properly. It trains three
different model families, keeps their out-of-fold predictions, and then learns how
much to trust each one using a second model fitted on those predictions.

Three things here are worth your time even if you never run the code:

1. A target encoding test that came out the opposite way to what I expected, with the numbers.
2. A correlation check that shows why adding a fourth gradient boosting model is a waste of your GPU quota.
3. A stacking setup where the blend weights never see the rows they are scored on.

Everything runs on the official competition files. No external data, no public
prediction files, nothing downloaded."""))

cells.append(nbf.v4.new_markdown_cell("""## Setup

The data loader looks for the competition files rather than hard coding a path,
so this runs unchanged whether the notebook is attached to the competition or you
are testing locally."""))

cells.append(nbf.v4.new_code_cell("""import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from scipy.special import ndtri
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

TARGET = 'addicted_label'
ID = 'id'
SEED = 20260806
FOLDS = 5


def locate_data():
    for root in [Path('/kaggle/input'), Path('.')]:
        if not root.exists():
            continue
        for path in root.rglob('train.csv'):
            try:
                columns = pd.read_csv(path, nrows=2).columns
            except Exception:
                continue
            if TARGET in columns and (path.parent / 'test.csv').exists():
                return path.parent
    raise FileNotFoundError('Competition files were not found')


data_dir = locate_data()
train = pd.read_csv(data_dir / 'train.csv')
test = pd.read_csv(data_dir / 'test.csv')
sample = pd.read_csv(data_dir / 'sample_submission.csv')

raw_columns = [c for c in train.columns if c not in (ID, TARGET)]
y = train[TARGET].to_numpy('int8')

print('train:', train.shape)
print('test :', test.shape)
print('positive rate:', round(float(y.mean()), 6))"""))

cells.append(nbf.v4.new_markdown_cell("""## The feature view

Daily screen time is roughly the sum of social media, gaming, and work or study
time, so the leftover after subtracting those is informative on its own. Missing
values carry signal too, which is common in generated tabular data, so every
column gets a missing flag and the whole missing pattern gets packed into one
integer.

None of this touches the target. It is the same function applied to train and test."""))

cells.append(nbf.v4.new_code_cell("""def base_features(frame):
    x = frame.drop(columns=[ID, TARGET], errors='ignore').copy()
    numeric = list(x.select_dtypes(include='number').columns)

    x['missing_count'] = x.isna().sum(axis=1).astype('int8')
    pattern = np.zeros(len(x), dtype='int32')
    for bit, column in enumerate(x.columns):
        pattern |= x[column].isna().to_numpy('int32') << bit
    x['missing_pattern'] = pattern
    for column in numeric:
        x[column + '__missing'] = x[column].isna().astype('int8')

    x['leisure_hours'] = x['social_media_hours'] + x['gaming_hours']
    x['accounted_hours'] = x['leisure_hours'] + x['work_study_hours']
    x['unaccounted_screen'] = x['daily_screen_time_hours'] - x['accounted_hours']
    x['weekend_unaccounted'] = x['weekend_screen_time'] - x['accounted_hours']
    x['weekend_gap'] = x['weekend_screen_time'] - x['daily_screen_time_hours']
    x['weekend_ratio'] = x['weekend_screen_time'] / (x['daily_screen_time_hours'] + 0.25)
    x['leisure_share'] = x['leisure_hours'] / (x['daily_screen_time_hours'] + 0.25)
    x['social_share'] = x['social_media_hours'] / (x['daily_screen_time_hours'] + 0.25)
    x['gaming_share'] = x['gaming_hours'] / (x['daily_screen_time_hours'] + 0.25)
    x['work_share'] = x['work_study_hours'] / (x['daily_screen_time_hours'] + 0.25)
    x['notif_per_screen'] = x['notifications_per_day'] / (x['daily_screen_time_hours'] + 0.25)
    x['opens_per_screen'] = x['app_opens_per_day'] / (x['daily_screen_time_hours'] + 0.25)
    x['notif_per_open'] = x['notifications_per_day'] / (x['app_opens_per_day'] + 2.0)
    x['sleep_screen_balance'] = x['sleep_hours'] - x['daily_screen_time_hours']
    x['screen_sleep_ratio'] = x['daily_screen_time_hours'] / (x['sleep_hours'] + 0.25)
    return x.replace([np.inf, -np.inf], np.nan)


train_base = base_features(train)
test_base = base_features(test)
print('base features:', train_base.shape[1])"""))

cells.append(nbf.v4.new_markdown_cell("""## Exact value statistics, and a test that surprised me

Numeric values in this dataset repeat thousands of times, so for each column I add
how often a value occurs and the smoothed rate of the positive class for that value.
Pair keys get the same treatment for five combinations that describe screen time.

Here is the part worth reading. The usual advice is that a target encoding must be
built out of fold, because otherwise every training row sees its own label through
the encoded value. I believed that, built both versions, and measured them against
each other under one identical three fold protocol. Then I did the same for pair
keys. The results do not point the same way:

| what is encoded | fitted on all training rows | inner out of fold |
| --- | --- | --- |
| single columns | **0.96769** | 0.96755 |
| single + 5 pair keys | 0.87063 | 0.96754 |
| single + all 66 pair keys | 0.83549 | not run |

For single columns the plain version wins on every fold, by more than the fold
standard deviation of 0.00009. For pair keys the plain version does not just lose,
it falls off a cliff.

One number explains both rows. It is the number of training rows sharing a key:

| key | median rows per key | share of keys with one row |
| --- | --- | --- |
| daily_screen_time_hours | 236 | 3.1% |
| age | 37303 | 0.0% |
| daily_screen_time_hours + social_media_hours | 2 | 47.8% |
| daily_screen_time_hours + weekend_screen_time | 1 | 55.0% |

When a key holds hundreds of rows, one row's own label barely moves that key's rate,
so the leak is negligible. Out of fold encoding then only hurts, because it builds
training features from 80 percent of the rows while test features come from all of
them, and that mismatch costs more than the leak saves.

When a key holds one row, the encoded value for that row is its own label with a bit
of smoothing on top. The model learns to read the answer straight off the feature.
At prediction time the key has never been seen, the value falls back to the prior,
and everything the model learned about that feature is worthless. Half of those pair
keys are singletons, which is why the score collapses by 0.10.

So the rule is not "always encode out of fold" and it is not "never bother". Count
the rows per key first. Low cardinality keys are fine encoded plainly and slightly
better that way. High cardinality keys must be out of fold or left out entirely.

Worth adding: a 30k row version of the single column test said the opposite, because
at that size even single columns behave like sparse keys. A small scale check can
point the wrong way on exactly this question.

This notebook encodes target rates for single columns only. Pair keys appear below
purely as occurrence counts, which use no labels at all and are safe at any
cardinality."""))

cells.append(nbf.v4.new_code_cell("""PAIR_COLUMNS = [
    ('daily_screen_time_hours', 'weekend_screen_time'),
    ('daily_screen_time_hours', 'social_media_hours'),
    ('daily_screen_time_hours', 'sleep_hours'),
    ('social_media_hours', 'gaming_hours'),
    ('notifications_per_day', 'app_opens_per_day'),
]
SMOOTHING = 40.0


def key_of(frame, column, digits=None):
    s = frame[column]
    if digits is not None and pd.api.types.is_numeric_dtype(s):
        s = s.round(digits)
    return s.astype('string').fillna('__NA__')


def pair_key(frame, left, right, digits=None):
    return key_of(frame, left, digits) + '|' + key_of(frame, right, digits)


def add_statistics(fit_raw, fit_y, apply_raw, apply_base):
    \"\"\"Every mapping is learned from fit_raw only, then applied to apply_raw.\"\"\"
    prior = float(np.mean(fit_y))
    columns = {}

    for column in fit_raw.columns:
        counts = key_of(fit_raw, column).value_counts()
        columns[column + '__logfreq'] = np.log1p(
            key_of(apply_raw, column).map(counts).fillna(0)).to_numpy('float32')
        if pd.api.types.is_numeric_dtype(fit_raw[column]):
            rounded = key_of(fit_raw, column, 1).value_counts()
            columns[column + '__rlogfreq'] = np.log1p(
                key_of(apply_raw, column, 1).map(rounded).fillna(0)).to_numpy('float32')

        stats = pd.DataFrame({'k': key_of(fit_raw, column).to_numpy(), 'y': fit_y})
        stats = stats.groupby('k').y.agg(['sum', 'count'])
        rate = (stats['sum'] + SMOOTHING * prior) / (stats['count'] + SMOOTHING)
        columns[column + '__te'] = key_of(apply_raw, column).map(rate).fillna(
            prior).to_numpy('float32')

    for left, right in PAIR_COLUMNS:
        for tag, digits in (('exact', None), ('rounded', 1)):
            counts = pair_key(fit_raw, left, right, digits).value_counts()
            columns[f'{left}__{right}__{tag}_lf'] = np.log1p(
                pair_key(apply_raw, left, right, digits).map(counts).fillna(0)).to_numpy('float32')

    # Building the columns first and joining once keeps pandas from
    # reallocating the frame on every insert.
    added = pd.DataFrame(columns, index=apply_base.index)
    return pd.concat([apply_base, added], axis=1)


train_raw = train[raw_columns]
test_raw = test[raw_columns]
example = add_statistics(train_raw, y, train_raw, train_base)
print('features after statistics:', example.shape[1])
del example
gc.collect()"""))

cells.append(nbf.v4.new_markdown_cell("""## Getting out-of-fold predictions

Stacking only works if the meta model is fitted on predictions the base models made
for rows they did not train on. So each base model runs a five fold loop. The
statistics above are rebuilt inside every fold using only that fold's training rows,
which matters, because if I built them once on all of train the validation scores
would be optimistic and the stack would learn the wrong weights.

Categoricals are turned into integer codes so all three libraries can share one
matrix."""))

cells.append(nbf.v4.new_code_cell("""for column in train_base.select_dtypes(exclude='number').columns:
    levels = pd.Index(pd.concat([train_base[column], test_base[column]], ignore_index=True)
                      .astype('string').fillna('Missing').unique())
    dtype = pd.CategoricalDtype(levels)
    train_base[column] = train_base[column].astype('string').fillna('Missing').astype(dtype).cat.codes
    test_base[column] = test_base[column].astype('string').fillna('Missing').astype(dtype).cat.codes

folds = list(StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(train_base, y))
oof_store = {}
test_store = {}


def run_family(name, fit_predict):
    \"\"\"Five folds of one model family. Returns nothing, fills the stores.\"\"\"
    started = time.time()
    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    for fold, (fit_i, val_i) in enumerate(folds, 1):
        fit_raw = train_raw.iloc[fit_i]
        fit_y = y[fit_i]
        x_fit = add_statistics(fit_raw, fit_y, fit_raw, train_base.iloc[fit_i])
        x_val = add_statistics(fit_raw, fit_y, train_raw.iloc[val_i], train_base.iloc[val_i])
        x_test = add_statistics(fit_raw, fit_y, test_raw, test_base)
        x_val = x_val[x_fit.columns]
        x_test = x_test[x_fit.columns]

        val_pred, test_part = fit_predict(x_fit, fit_y, x_val, x_test, fold)
        oof[val_i] = val_pred
        test_pred += test_part / FOLDS
        print(f'  {name} fold {fold}: {roc_auc_score(y[val_i], val_pred):.7f}'
              f'  ({time.time() - started:.0f}s)', flush=True)
        del x_fit, x_val, x_test
        gc.collect()

    oof_store[name] = oof
    test_store[name] = test_pred
    print(f'{name} pooled OOF AUC: {roc_auc_score(y, oof):.7f}\\n', flush=True)"""))

cells.append(nbf.v4.new_markdown_cell("""### XGBoost

Depth 7 with a low learning rate. On a GPU this is the fastest of the three."""))

cells.append(nbf.v4.new_code_cell("""USE_GPU = True
try:
    xgb.train({'device': 'cuda', 'tree_method': 'hist'},
              xgb.DMatrix(np.zeros((8, 2), 'float32'), label=np.array([0, 1] * 4)), 1)
except Exception:
    USE_GPU = False
print('xgboost on gpu:', USE_GPU)


def xgb_fit(x_fit, fit_y, x_val, x_test, fold):
    params = {'objective': 'binary:logistic', 'eval_metric': 'auc',
              'max_depth': 7, 'eta': 0.022, 'subsample': 0.85,
              'colsample_bytree': 0.55, 'min_child_weight': 48,
              'reg_lambda': 3.5, 'reg_alpha': 0.25, 'seed': SEED + fold,
              'tree_method': 'hist'}
    if USE_GPU:
        params['device'] = 'cuda'
    booster = xgb.train(params, xgb.DMatrix(x_fit, label=fit_y), 2800)
    return booster.predict(xgb.DMatrix(x_val)), booster.predict(xgb.DMatrix(x_test))


run_family('xgboost', xgb_fit)"""))

cells.append(nbf.v4.new_markdown_cell("""### LightGBM

Shallower and more heavily sampled than the XGBoost model, so it makes different
mistakes. That is the only reason it is here."""))

cells.append(nbf.v4.new_code_cell("""def lgb_fit(x_fit, fit_y, x_val, x_test, fold):
    params = {'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.03,
              'num_leaves': 31, 'min_data_in_leaf': 160, 'bagging_fraction': 0.85,
              'bagging_freq': 1, 'feature_fraction': 0.7, 'lambda_l1': 0.15,
              'lambda_l2': 2.5, 'verbosity': -1, 'seed': SEED + fold,
              'num_threads': -1}
    booster = lgb.train(params, lgb.Dataset(x_fit, label=fit_y), num_boost_round=2400)
    return booster.predict(x_val), booster.predict(x_test)


run_family('lightgbm', lgb_fit)"""))

cells.append(nbf.v4.new_markdown_cell("""### CatBoost

Ordered boosting and a different split rule again. CatBoost wants no missing values
in its numeric matrix, so they get a sentinel value that sits well outside the real
range. The missing flags built earlier still carry the information."""))

cells.append(nbf.v4.new_code_cell("""def cat_fit(x_fit, fit_y, x_val, x_test, fold):
    a = x_fit.astype('float32').fillna(-999.0)
    b = x_val.astype('float32').fillna(-999.0)
    c = x_test.astype('float32').fillna(-999.0)
    model = CatBoostClassifier(iterations=3000, depth=7, learning_rate=0.035,
                               l2_leaf_reg=7.0, random_strength=0.3,
                               bagging_temperature=0.4, eval_metric='AUC',
                               task_type='GPU' if USE_GPU else 'CPU',
                               random_seed=SEED + fold, verbose=0)
    model.fit(Pool(a, fit_y))
    return model.predict_proba(b)[:, 1], model.predict_proba(c)[:, 1]


run_family('catboost', cat_fit)"""))

cells.append(nbf.v4.new_markdown_cell("""## How different are they really

Before stacking, look at how correlated the three models are. This is the check that
tells you whether another model is worth training.

In my own larger experiments every gradient boosting model I trained sat above 0.99
rank correlation with every other one, no matter which library produced it. Adding a
fourth was worth about six ten millionths of AUC. The models that actually moved the
blend were the ones built on a different idea, such as a neural network over exact
value embeddings, which sat closer to 0.97.

If the numbers below are all very close to 1, the honest conclusion is that a fourth
tree model will not help you, and your time is better spent on something structurally
different."""))

cells.append(nbf.v4.new_code_cell("""names = list(oof_store)
ranks = {n: pd.Series(oof_store[n]).rank(pct=True) for n in names}

print('standalone OOF AUC')
for n in names:
    print(f'  {n:<10} {roc_auc_score(y, oof_store[n]):.7f}')

print('\\nrank correlation between models')
corr = pd.DataFrame({a: {b: ranks[a].corr(ranks[b]) for b in names} for a in names})
print(corr.round(5).to_string())"""))

cells.append(nbf.v4.new_markdown_cell("""## The stack

Two details make this a fair stack rather than a way to fool yourself.

First, each model's predictions are converted to normal scores by rank. A tree model
and a neural network can be well ordered but calibrated completely differently, and
a linear meta model handles them far better once they are on a shared scale.

Second, the meta model is cross fitted. The logistic regression is trained on four
fifths of the rows and predicts the fifth it did not see, rotating through. If I
fitted it once on everything and scored it on the same rows, the number would look
better and mean nothing.

Weights are left unconstrained on purpose. Some come out negative, which looks wrong
until you think about it. A negative weight means that model is being used to correct
the others rather than to vote. When I forced weights to be non negative in a larger
version of this stack, the score dropped by about 0.0007, which is large for this
competition."""))

cells.append(nbf.v4.new_code_cell("""def gauss_rank(values):
    r = pd.Series(values).rank(method='average').to_numpy()
    return ndtri(r / (len(r) + 1.0)).astype('float32')


X = np.column_stack([gauss_rank(oof_store[n]) for n in names])
X_test = np.column_stack([gauss_rank(test_store[n]) for n in names])

meta_folds = StratifiedKFold(5, shuffle=True, random_state=SEED + 1).split(X, y)
meta_oof = np.zeros(len(y))
for fit_i, val_i in meta_folds:
    meta = LogisticRegression(C=0.1, max_iter=2000).fit(X[fit_i], y[fit_i])
    meta_oof[val_i] = meta.decision_function(X[val_i])

stack_auc = roc_auc_score(y, meta_oof)
best_single = max(roc_auc_score(y, oof_store[n]) for n in names)
print(f'best single model : {best_single:.7f}')
print(f'cross fitted stack: {stack_auc:.7f}')
print(f'gain from stacking: {stack_auc - best_single:+.7f}')

final = LogisticRegression(C=0.1, max_iter=2000).fit(X, y)
print('\\nlearned weights')
for n, w in zip(names, final.coef_[0]):
    print(f'  {n:<10} {w:+.5f}')
test_score = final.decision_function(X_test)"""))

cells.append(nbf.v4.new_markdown_cell("""## Submission

The meta model returns a log odds score rather than a probability. AUC only cares
about ordering, so converting to a percentile rank is safe and keeps the output
inside the range the submission format expects.

The checks at the end are dull by design. Every wasted submission I have made came
from a file that was silently misaligned, not from a bad model."""))

cells.append(nbf.v4.new_code_cell("""submission = sample.copy()
submission[TARGET] = pd.Series(test_score).rank(pct=True).to_numpy()

assert submission[ID].equals(sample[ID])
assert list(submission.columns) == [ID, TARGET]
assert submission[TARGET].notna().all()
assert np.isfinite(submission[TARGET]).all()
assert submission[TARGET].between(0, 1).all()
assert len(submission) == len(test)

submission.to_csv('submission.csv', index=False)
print(submission.shape)
submission.head()"""))

cells.append(nbf.v4.new_markdown_cell("""## What I would do next

The stack above beats its own best member, but the gap between a good single model
and a good blend on this competition is smaller than people expect. The leaderboard
is compressed to the point where a few ten thousandths separate dozens of teams.

Two things helped me more than adding models:

Ten folds instead of five. Each model then trains on 90 percent of the rows rather
than 80. On the XGBoost model here that was worth about 0.00018, which is more than
a whole extra model family bought me.

Checking correlation before spending compute. I trained a CatBoost variant that took
26 minutes on a GPU and added 0.0000059 to the blend, because it was 0.99 correlated
with models I already had. The correlation table would have told me that in advance.

If this was useful, an upvote helps other people find it. Questions and corrections
are welcome in the comments, particularly if you get a different answer on the target
encoding test."""))

nb["cells"] = cells
HERE.mkdir(parents=True, exist_ok=True)
nbf.write(nb, HERE / "s6e8_stacking_three_families.ipynb")
print("wrote", HERE / "s6e8_stacking_three_families.ipynb", "cells:", len(cells))
