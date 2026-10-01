# Playground Series S6E9: Predicting Electric Vehicle Purchases

Kaggle `playground-series-s6e9`. Binary tabular classification scored on ROC AUC:
given a 13-column consumer survey row, predict the probability that the respondent
buys an electric vehicle.

Submission format is `id,Will_Buy_EV`, where the second column is a probability in
[0, 1]. Only the ordering matters for AUC, so calibration is irrelevant here.

## Result

| | CV AUC | Public LB |
|---|---|---|
| Stock LightGBM on the 13 raw columns | 0.941776 | 0.94163 |
| Same model plus the income digit features | 0.945685 | 0.94581 |
| Tuned XGBoost, 10 folds, 3 seeds | 0.946094 | 0.94612 |
| Seven-model weighted rank blend | 0.946339 | **0.94636** |

Cross-validation tracked the public leaderboard to within 0.0002 on all six
submissions, so model selection was done on CV throughout.

The stock baseline sat on the leaderboard median at the time it was submitted.
Everything above it came from one observation about how the data was generated,
and almost nothing from tuning.

## The data

668,665 training rows and 286,571 test rows. No missing values in either file.
No duplicate feature rows. `id` is pure noise (AUC 0.49999). Train and test
marginals are effectively identical: every numeric mean differs by less than
0.005 standard deviations, every categorical by less than 0.002 total variation.
There is no cleaning to do and no drift to correct, which is part of why tuning
feels so unrewarding on this dataset.

| Column | Type | Domain | Role |
|---|---|---|---|
| `Age` | numeric | 25 to 69 | weak, non-monotonic |
| `Annual_Income_USD` | numeric | integer, floored at 30,000 | second strongest (r = 0.226), and carries the artifact |
| `Daily_Commute_km` | numeric | one decimal, floored at 5.0 | weak negative |
| `Number_of_Cars_Owned` | numeric | 1 to 4 | negligible |
| `Charging_Stations_Near_Home` | numeric | 0 to 14 | negligible |
| `Charging_Stations_Near_Work` | numeric | 0 to 19 | negligible |
| `Environmental_Concern_Level` | numeric | 1 to 5 | strongest single driver (r = 0.464) |
| `Gender` | categorical | Male / Female / Other | no measurable effect |
| `City_Type` | categorical | Urban / Suburban / Rural | mild |
| `Current_Car_Type` | categorical | Hatchback / Sedan / SUV / Truck | mild, truck owners buy least |
| `Home_Charging_Possible` | categorical | Yes / No | moderate (12.7% vs 19.6%) |
| `Subsidy_Available` | categorical | Yes / No | hard gate (0.6% vs 27.5%) |
| `Range_Anxiety_Level` | categorical | Low / Medium / High | hard gate (18.9% / 4.2% / 0.1%) |

17.46% of training rows are positive. The provided `sample_submission.csv` is that
base rate repeated on every row, which scores 0.5.

### Two gates and one dial

`Subsidy_Available` and `Range_Anxiety_Level` multiply rather than add. Close either
one and the purchase rate falls below one percent regardless of income, age or
charging access.

| | Low anxiety | Medium | High |
|---|---|---|---|
| No subsidy | 0.63% | 0.10% | 0.00% |
| Subsidy | 29.57% | 6.94% | 0.23% |

Inside the open cell (subsidy available, low anxiety, 381,386 rows) the rate is
29.6%, and two features do nearly all the remaining work:
`Environmental_Concern_Level` sweeps it from 1.0% to 71.9%, and income decile
sweeps it from 8.0% to 52.2%. Everything else is second order.

Two columns are floor-censored: 9.2% of rows sit exactly at
`Annual_Income_USD = 30000` and 21.6% exactly at `Daily_Commute_km = 5.0`. Both get
an explicit flag rather than being treated as ordinary values.

## Why a stock model stalls

A stock LightGBM scores 0.9418. Three checks say that is not a tuning problem.

1. A capacity sweep saturates. Learning rates from 0.01 to 0.05 and 16 to 256 leaves
   all land within 0.0007 of each other on fold 0, and the shallowest configuration
   wins. The target function is smooth and low order, not deep and
   interaction heavy. (`src/diagnostics/capacity_ceiling.py`)
2. A GPU MLP with categorical embeddings reaches only 0.9372, below the trees. So the
   gap is not a failure to model smoothness either.
3. The model is already at its own ceiling. Sampling labels from its own out-of-fold
   probabilities and re-scoring gives 0.9416, which is the best AUC obtainable if
   those probabilities were the truth. It achieved 0.9418.
   (`src/diagnostics/bayes_ceiling.py`)

That third number is the useful one. It does not say the task is capped at 0.9416,
it says the features as given are. The representation had to change.

## What the generator leaked

The competition data is synthetic, generated from a real 10,000-row survey that is
public on Kaggle (`itzzomkar/ev-adoption-behavior-and-range-anxiety`). The generating
model left three separable fingerprints on `Annual_Income_USD`.

### 1. The digits

Group training rows by a base-10 digit position of income and measure how far the
purchase rate spreads across the ten digits, against the binomial null corridor
(the spread expected if the digit meant nothing).

| Digit position | Spread, competition data | Null corridor | Ratio |
|---|---|---|---|
| units | 7.59 pp | 0.46 pp | 16.7x |
| tens | 10.01 pp | 0.46 pp | 22.0x |
| hundreds | 11.58 pp | 0.46 pp | 25.4x |
| thousands | 10.60 pp | 0.46 pp | 23.3x |

The control is what makes this a finding rather than a fishing expedition. Run the
identical test on the real survey the data was generated from and the ratios are 1.4
to 1.6, which is nothing. Same test, same column, same units. The only difference is
which dataset a generative model sat in front of. The hundreds digit of a household
income cannot cause anyone to buy a car.

Worth roughly 0.0025 AUC. (`src/diagnostics/income_digits.py`)

### 2. The jaggedness

The income to P(buy) curve is not smooth at any scale. Binned at $1,000 and compared
against its own local trend (an 11-bin rolling median), more than thirty bands
deviate by over 6 sigma and several by over 15. The $60,000 to $63,000 range sits at
1 to 3% where its neighbourhood says 7 to 9%; $68,000 sits at 24% where its
neighbourhood says 11%. Income therefore needs encoding at several resolutions at
once rather than one.

### 3. Hard rules

Three regions are effectively deterministic.

| Region | Train rows | Purchase rate | Test rows |
|---|---|---|---|
| `income >= 170537` | 393 | 1.0000 | 156 |
| `38000 <= income <= 42000` | 1,277 | 0.0016 | 503 |
| `income == 30000` (mode-collapse spike) | 61,605 | 0.0443 | 26,276 |

The cliff at 170,537 overrides both gates. All 393 rows buy regardless of subsidy,
range anxiety or environmental concern, and the highest income carrying a "No" label
anywhere in training is 169,972.

## Features and validation

150 columns before fold-level encoding, built in two stages so that the
fold-dependent part can be rebuilt inside every fold.

Fold independent (`features.base_frame`):

* the 13 raw columns, ordinal encoded, minus `Number_of_Cars_Owned` which carries
  nothing
* base-10 digits of income, commute and age at powers 0 through 5
* flags for the cliff, the dead zone, the $30,000 spike and the commute floor, plus
  distance to the cliff
* multi-resolution keys: income exact, income // 100, income // 1000, commute exact,
  commute integer
* real-survey anchors, which map each value to its purchase rate in the original
  10,000-row survey. This gives the model the true smooth curve alongside the
  generator's jagged one, and carries no competition label
* frequency encodings computed over train and test pooled. This is unsupervised, so
  pooling is legitimate and gives a sharper estimate of how often each exact income
  value was emitted

Fold dependent (`features.fold_transform`): smoothed mean-target encodings of all 29
base, digit and key columns at three smoothing strengths (5, 20, 100), so the model
picks its own bias and variance trade-off rather than having one picked for it.

Two rules keep those honest, and skipping either inflates CV by several thousandths
while buying nothing on the board:

1. encodings are fitted inside each fold, on that fold's training rows only
2. the training-side copy uses an inner 5-fold, so no row is ever encoded with a
   statistic its own label helped compute

Validation is one `StratifiedKFold(shuffle, seed=42)` split shared by every
experiment, so the out-of-fold matrix is coherent and blend weights are fitted on
honest predictions. Exploration ran at 5 folds. The final suite uses 10, worth about
0.00026, because each model sees more rows and the in-fold target encoding is
estimated from 90% of the data instead of 80%. Set `FOLDS=10`.

## Models and blending

Optuna TPE on pre-transformed folds, 45 trials for LightGBM and 30 for XGBoost. Both
converged independently on very shallow models: LightGBM at 19 leaves and lr 0.017,
XGBoost at depth 4 and lr 0.021. That is the same conclusion the capacity sweep
reached from the other direction.

Only models trained in this folder go into the blend. No public submission files are
used. Predictions are converted to ranks, then combined four ways, and the best
out-of-fold blender is submitted: plain rank mean, greedy hill climbing with
replacement, those weights polished by a continuous non-negative Nelder-Mead search,
and a cross-validated logistic stack as a sanity check.

Each model is also re-run over a different partition of the same rows (`FOLD_SEED=7`).
Every run's out-of-fold estimate stays honest, since it is still out-of-fold for its
own split, and averaging across partitions removes split variance from the test
predictions.

| Model | Folds | CV AUC |
|---|---|---|
| `final_lgb_v2_fs7` LightGBM | 10 | 0.946241 |
| `final_lgb_v2` LightGBM | 10 | 0.946223 |
| `final_xgb_v2_fs7` XGBoost | 10 | 0.946227 |
| `final_xgb_v2` XGBoost | 10 | 0.946212 |
| `final_lgbdiv_v2` LightGBM, blinded to exact-income TE | 10 | 0.946168 |
| `final_xgb_v1` XGBoost, earlier feature set | 10 | 0.946094 |
| `final_cat_v2` CatBoost | 10 | 0.946005 |
| `final_lgbxt_v2` LightGBM, extremely randomised splits | 10 | 0.945718 |
| `final_mlp_v2` MLP with embeddings | 10 | 0.942540 |
| Weighted rank blend | 10 | **0.946339** |

### The blend ceiling is structural

This is the part worth reading. Every gradient-boosted model correlates at 0.994 to
0.999 on ranks, so eight models together beat the best single model by 0.0001, and
models nine and ten changed the blend by nothing at all (0.946339, then 0.946338,
then 0.946339).

That was tested rather than assumed. `final_lgbdiv_v2` is a LightGBM deliberately
blinded to the exact-income target encodings, the columns every other model leans on,
to force it to rebuild the signal from raw digits and coarser keys. It scored
0.946168, essentially unchanged, and came back correlated at 0.9994 with the standard
LightGBM, which is higher than XGBoost is. The artifact is encoded so redundantly
across digits, multi-scale keys and smoothings that any competent model converges on
the same ranking. No amount of reweighting fixes that.

The MLP is the only genuinely decorrelated member at rank correlation 0.968, which
makes it exactly the blend member one would want, and it is too weak to carry weight.

## What did not work

Measured on the same folds, and kept here rather than quietly dropped.

| Tried | Result |
|---|---|
| Window encodings: rolling target mean over the k nearest distinct income values | -0.00008. The three smoothing strengths already smooth at multiple scales, so the windows were redundant. Code kept behind `USE_WINDOWS = False` |
| Digits declared categorical to LightGBM | -0.00008. Ordinally meaningless, but the model isolates single digit values anyway |
| The real 10k survey as extra training rows | +0.0002 at unit weight, negative above it. Its value was as a control, not as data |
| Ratio features: income per car, km per station, income times commute | -0.0001. More ways to split on noise |
| Wider, deeper and longer-scheduled MLPs | All worse than the baseline MLP. 0.9436 against 0.9451 for trees on the same fold |
| Forcing the cliff rows to p = 1 | +0.000001. The model already ranks 97% of them in the top 1% |
| Retuning XGBoost on the wider matrix | Better on the two tuning folds, 0.00005 worse on full CV. Two-fold tuning overfits those folds |

One of these is worth singling out. A residual scan over roughly forty candidate keys,
ranked by statistical significance, put `Age` at the top at nearly 19 sigma. It is
worth essentially nothing: with about 14,800 rows per age value, a plus or minus
0.8 percentage point wobble is overwhelming evidence that the wobble is real and no
evidence at all that it is useful. Significance scales with n, AUC does not. Rank
candidates by effect size. (`src/diagnostics/residual_scan.py`)

## Layout

```
src/common.py        data loading, the shared fold split, leak-free target encoding,
                     the experiment runner (writes OOF, test preds and a JSONL log)
src/features.py      base_frame() for fold-independent columns,
                     fold_transform() for the fold-dependent encodings
src/eda.py           EDA figures and reports/eda_stats.json
src/exp01, exp02     the baseline and the first digit model, kept to show the jump
src/exp_final.py     every final model: lgb, xgb, cat, hgb, mlp, lgbxt, lgbdiv
src/tune.py          Optuna search over pre-transformed folds
src/blend.py         rank mean, hill climbing, weight polish, logistic stack
src/report_data.py   computes every number the report quotes
src/make_report.py   renders reports/report.html
src/make_notebook.py builds the Kaggle notebook
src/diagnostics/     the measurements behind each claim above
notebook/            the Kaggle notebook, runs end to end
reports/report.html  standalone analysis report
artifacts/           experiment log and tuned parameters
```

Competition data, prediction arrays and generated submissions are not committed. See
`.gitignore`.

## Reproducing

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python pandas numpy scikit-learn matplotlib seaborn \
    lightgbm xgboost catboost optuna nbformat
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu124

kaggle competitions download -c playground-series-s6e9 -p data
unzip -o 'data/*.zip' -d data
kaggle datasets download -d itzzomkar/ev-adoption-behavior-and-range-anxiety \
    -p data/original --unzip
```

Then, from `src/`:

```bash
../.venv/bin/python eda.py                      # figures and stats
FOLDS=10 ../.venv/bin/python exp_final.py lgb 3 # any of lgb xgb cat hgb mlp lgbxt lgbdiv
FOLDS=10 FOLD_SEED=7 ../.venv/bin/python exp_final.py lgb 3
../.venv/bin/python blend.py                    # blends whatever OOFs exist
../.venv/bin/python report_data.py && ../.venv/bin/python make_report.py
```

Submitting:

```bash
kaggle competitions submit -c playground-series-s6e9 \
    -f submissions/<file>.csv -m "<message>"
```

## A note on the leaderboard

At the 2026-09-28 snapshot the public board read 0.94945 at the top (a single
outlier), then a cluster at 0.94694, 0.94681, 0.94679, with 3,331 teams. The best
self-trained single model published publicly was LB 0.94638.

Scores above that cluster are, on inspection, chains of other people's public
submission CSVs re-blended. One such notebook documents combining
`0.50 x author_A + 0.50 x author_B`, then `0.47 x that + 0.54 x author_C`. Nothing
here does that. Every model in this folder is trained from the competition data, and
the blend only ever combines those, which puts this solution at the honest
single-model frontier rather than at the top of the visible board.

Playground Series competitions do not award competition medals for leaderboard rank.
Medals there come from notebook and discussion upvotes, which is why `notebook/` is a
first-class deliverable here rather than an afterthought.
