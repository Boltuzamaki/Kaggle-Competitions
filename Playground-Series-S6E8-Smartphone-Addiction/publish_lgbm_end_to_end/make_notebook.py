"""Build the private review notebook for the original LightGBM submission."""
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
cells.append(nbf.v4.new_markdown_cell("""# End to End, No Model Blend | LightGBM Public LB 0.96949

This is the full pipeline behind my first strong standalone model family for the Smartphone Addiction competition.

**Public leaderboard ROC AUC: 0.96949**

There is no stacking and no mixing of different model types here. The final prediction is the average of three LightGBM seeds using the same features and parameters. I use the seed average only to make the result a little less sensitive to randomness.

The useful part of this approach was not a huge parameter search. It was treating repeated values, missing fields, and screen time accounting as useful structure instead of feeding only the raw columns into a model."""))

cells.append(nbf.v4.new_markdown_cell("""## Setup

The notebook reads only the official competition files. It does not load public predictions, external labels, or another notebook's submission."""))

cells.append(nbf.v4.new_code_cell("""from pathlib import Path
import gc
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

TARGET = 'addicted_label'
ID = 'id'
SEEDS = [3407, 7117, 9919]
PAIR_COLUMNS = [
    ('daily_screen_time_hours', 'weekend_screen_time'),
    ('daily_screen_time_hours', 'social_media_hours'),
    ('daily_screen_time_hours', 'sleep_hours'),
    ('social_media_hours', 'gaming_hours'),
    ('notifications_per_day', 'app_opens_per_day'),
]

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

print('train:', train.shape)
print('test :', test.shape)
print('target mean:', round(train[TARGET].mean(), 6))"""))

cells.append(nbf.v4.new_markdown_cell("""## A compact feature view

Daily screen time is closely related to social media, gaming, and work or study time. I keep the original values and add a few accounting features around that relationship. Missing values are also informative in this synthetic dataset, so every numeric column gets a missing flag."""))

cells.append(nbf.v4.new_code_cell("""def make_base_features(frame):
    x = frame.drop(columns=[ID, TARGET], errors='ignore').copy()
    numeric = list(x.select_dtypes(include='number').columns)

    x['missing_count'] = x.isna().sum(axis=1).astype('int8')
    missing_pattern = np.zeros(len(x), dtype='int32')
    for bit, column in enumerate(x.columns):
        missing_pattern |= x[column].isna().to_numpy('int32') << bit
    x['missing_pattern'] = missing_pattern
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

x_train_base = make_base_features(train)
x_test_base = make_base_features(test)
raw_columns = [c for c in train.columns if c not in [ID, TARGET]]
print('base feature count:', x_train_base.shape[1])"""))

cells.append(nbf.v4.new_markdown_cell("""## Exact value statistics

Many numeric values repeat thousands of times. For each original column I add two simple summaries:

1. How often the exact value appears in train
2. The smoothed target rate for that value

The test table never contributes labels. All mappings applied to test are learned from train only.

One practical note: this cell reproduces the representation used for the recorded leaderboard submission. Because the train target statistic contains the row itself, it should not be used as proof of cross validation quality. The public score quoted above is the actual recorded competition score."""))

cells.append(nbf.v4.new_code_cell("""def add_exact_statistics(fit_x, apply_x, y, columns, smoothing=40.0):
    out = apply_x.copy()
    prior = float(np.mean(y))

    for column in columns:
        fit_key = fit_x[column].astype('string').fillna('__NA__')
        apply_key = apply_x[column].astype('string').fillna('__NA__')
        stats = pd.DataFrame({'key': fit_key, 'y': np.asarray(y)}).groupby('key').y.agg(['sum', 'count'])

        rate = (stats['sum'] + smoothing * prior) / (stats['count'] + smoothing)
        out[column + '__te'] = apply_key.map(rate).fillna(prior).astype('float32')
        out[column + '__logfreq'] = np.log1p(apply_key.map(stats['count']).fillna(0)).astype('float32')

        if pd.api.types.is_numeric_dtype(fit_x[column]):
            rounded_fit = fit_x[column].round(1).astype('string').fillna('__NA__')
            rounded_apply = apply_x[column].round(1).astype('string').fillna('__NA__')
            rounded_count = rounded_fit.value_counts(dropna=False)
            out[column + '__rounded_logfreq'] = np.log1p(
                rounded_apply.map(rounded_count).fillna(0)).astype('float32')

    # Pair support is target free. It tells the tree whether a combination is
    # common without adding another target-rate column.
    for left, right in PAIR_COLUMNS:
        for suffix, digits in [('exact', None), ('rounded', 1)]:
            left_fit = fit_x[left] if digits is None else fit_x[left].round(digits)
            right_fit = fit_x[right] if digits is None else fit_x[right].round(digits)
            left_apply = apply_x[left] if digits is None else apply_x[left].round(digits)
            right_apply = apply_x[right] if digits is None else apply_x[right].round(digits)
            fit_key = left_fit.astype('string').fillna('__NA__') + '|' + right_fit.astype('string').fillna('__NA__')
            apply_key = left_apply.astype('string').fillna('__NA__') + '|' + right_apply.astype('string').fillna('__NA__')
            count = fit_key.value_counts(dropna=False)
            out[f'{left}__{right}__{suffix}_logfreq'] = np.log1p(
                apply_key.map(count).fillna(0)).astype('float32')

    return out

y = train[TARGET].astype('int8')
x_train = add_exact_statistics(x_train_base, x_train_base, y, raw_columns)
x_test = add_exact_statistics(x_train_base, x_test_base, y, raw_columns)

print('final feature count:', x_train.shape[1])"""))

cells.append(nbf.v4.new_markdown_cell("""## Categorical columns

LightGBM can use the original categorical fields directly. I align their category levels once so train and test have the same schema."""))

cells.append(nbf.v4.new_code_cell("""categorical_columns = list(x_train.select_dtypes(exclude='number').columns)
for column in categorical_columns:
    levels = pd.Index(pd.concat([x_train[column], x_test[column]], ignore_index=True)
                      .astype('string').fillna('Missing').unique())
    dtype = pd.CategoricalDtype(levels)
    x_train[column] = x_train[column].astype('string').fillna('Missing').astype(dtype)
    x_test[column] = x_test[column].astype('string').fillna('Missing').astype(dtype)

print('categorical columns:', categorical_columns)"""))

cells.append(nbf.v4.new_markdown_cell("""## Train one LightGBM recipe

The parameters are fixed. I train the same model with three random seeds and average them. This is seed averaging within one model family, not a model blend."""))

cells.append(nbf.v4.new_code_cell("""def build_model(seed):
    return lgb.LGBMClassifier(
        objective='binary',
        n_estimators=3654,
        learning_rate=0.025,
        num_leaves=31,
        min_child_samples=160,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.86,
        reg_alpha=0.15,
        reg_lambda=2.5,
        max_bin=255,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )

predictions = []
importance = []

for seed in SEEDS:
    print('training seed', seed)
    model = build_model(seed)
    model.fit(x_train, y, categorical_feature=categorical_columns)
    predictions.append(model.predict_proba(x_test)[:, 1])
    importance.append(pd.Series(model.feature_importances_, index=x_train.columns, name=str(seed)))
    del model
    gc.collect()

test_prediction = np.mean(predictions, axis=0)
print('prediction range:', float(test_prediction.min()), float(test_prediction.max()))"""))

cells.append(nbf.v4.new_markdown_cell("""## What the model used

The importance chart is a quick sanity check rather than a causal explanation. The model consistently relies on screen time, social and gaming usage, notification behavior, and the repeated value statistics."""))

cells.append(nbf.v4.new_code_cell("""importance_table = pd.concat(importance, axis=1)
importance_table['mean'] = importance_table.mean(axis=1)
top = importance_table.sort_values('mean', ascending=False).head(20).sort_values('mean')
ax = top['mean'].plot.barh(figsize=(9, 7), title='Top LightGBM features')
ax.set_xlabel('split importance averaged over three seeds')"""))

cells.append(nbf.v4.new_markdown_cell("""## Submission

The final checks are intentionally boring. IDs must stay in sample order, every prediction must be finite, and the output must contain exactly the two expected columns."""))

cells.append(nbf.v4.new_code_cell("""submission = sample.copy()
submission[TARGET] = test_prediction

assert submission[ID].equals(sample[ID])
assert list(submission.columns) == [ID, TARGET]
assert submission[TARGET].notna().all()
assert np.isfinite(submission[TARGET]).all()
assert submission[TARGET].between(0, 1).all()

submission.to_csv('/kaggle/working/submission.csv', index=False)
submission.head()"""))

cells.append(nbf.v4.new_markdown_cell("""## Closing thought

This result came from a fairly small LightGBM and a feature view that matches the structure of the data. It reached **0.96949** on the public leaderboard without stacking different model families.

There is still room above this score, but this notebook is a useful clean baseline when you want one understandable pipeline from train.csv to submission.csv."""))

nb["cells"] = cells
HERE.mkdir(parents=True, exist_ok=True)
nbf.write(nb, HERE / "s6e8_end_to_end_no_model_blend.ipynb")
