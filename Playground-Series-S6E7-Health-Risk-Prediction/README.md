# Playground Series S6E7 — Health Risk Prediction

Kaggle **Playground Series, Season 6 Episode 7** (`playground-series-s6e7`) — a
3-class tabular classification problem scored on **balanced accuracy**.

This folder holds the full solution: a local model zoo, ~34 Kaggle GPU/CPU
kernels, the ensembling and audit tooling, and the EDA that motivated it all.
Data, prediction arrays and generated submissions are excluded (see §7).

---

## 1. The problem

Predict `health_condition ∈ {fit, at-risk, unhealthy}` for each person from 13
lifestyle/biometric features.

| | |
|---|---|
| train / test rows | 690,088 / 295,753 |
| features | 7 numeric, 6 categorical — 11 of the 13 have missing values (1–12%) |
| classes | `at-risk` 85.9%, `unhealthy` 8.4%, `fit` 5.8% |
| metric | **balanced accuracy** — mean of per-class recall, so the two minority classes matter as much as the 86% majority |
| teams | 2,852 |

Features: `sleep_duration`, `heart_rate`, `bmi`, `calorie_expenditure`,
`step_count`, `exercise_duration`, `water_intake` (numeric);
`diet_type`, `stress_level`, `sleep_quality`, `physical_activity_level`,
`smoking_alcohol`, `gender` (categorical).

The dataset is synthetic, generated from a real student-health survey. That
detail turns out to be the single most important fact in the whole competition
— see §3.2.

`notebooks/eda.ipynb` has the full exploration: target imbalance, missingness
structure, per-feature distributions, and a train/test covariate-shift check
(the split is clean and i.i.d., so no adversarial-validation work was needed).

## 2. Result

| | balanced accuracy |
|---|---|
| Best public-LB submission (snapshot 2026-07-26) | **0.95070** — rank 379 / 2,852 |
| Best nested-CV-audited ensemble (OOF) | **0.95074** |
| Best single model (OOF) | **0.95065** — RealMLP, exact-value TE features |
| Best local-zoo model without exact-value TE (OOF) | 0.88083 — LightGBM |

The leaderboard figure comes from a public-LB snapshot taken on 2026-07-26.
Work continued past that date; the final standing is not captured in this repo.

The gap between rows 3 and 4 — **+0.07 balanced accuracy** — is the whole story
of this solution, and it came from one feature-engineering idea, not from
model tuning. §3.2.

## 3. Approach

The work ran in three phases. Each one is still in this folder.

### 3.1 Phase 1 — a broad local zoo, to find the honest baseline

Design constraints, taken from a prior season's winning writeup (public-LB rank
chasing via blind blending decoupled from CV; trusting CV instead moved that
author from rank 344 public to rank 1 private):

- **One fixed CV split, reused by every model** (`src/folds.py`, cached to
  `artifacts/folds.npy`). Every model's out-of-fold predictions are row-aligned,
  so any subset can be blended later by plain array averaging — no retraining.
- **Feature sets built once, not per model.** `src/features.py` materialises 4
  cached numeric matrices that every model spec picks from:

  | name | dims | NaNs | categoricals | consumers |
  |---|---|---|---|---|
  | `raw_nan` | 13 | preserved | ordinal | NaN-native models (LGBM/XGB/CatBoost/HistGB/RF/ET/DT) |
  | `raw_imputed` | 31 | median-imputed + standardised | one-hot | linear / distance / NB / MLP |
  | `fe_heavy_nan` | 450 | preserved | ordinal | NaN-native, heavy-FE variant |
  | `fe_heavy_imputed` | 450 | imputed + standardised | ordinal + one-hot | linear / distance / NB / MLP, heavy-FE variant |

  The heavy-FE block (~440 columns) is missing indicators, pairwise numeric
  ratios/diffs/products, log1p and square transforms, quantile bins, groupby
  aggregates by each categorical, deviation-from-group-mean, frequency encoding
  (including pairwise), and out-of-fold multiclass target encoding keyed to the
  **same fixed folds** — so it is leakage-free and every OOF score stays honest.
- **Diversity over tuning.** `src/model_zoo.py` generates 434 deterministic
  `ModelSpec`s (stable id = hash of family + feature set + params + seed) across
  22 families. Hand-picked presets chosen for difference of *shape*, not a blind
  grid.
- **Resumable by construction.** `src/train.py` writes `artifacts/manifest.csv`
  *after* each model's OOF/test arrays land on disk via atomic rename. Kill it
  any time; rerunning skips every `success` id. Registry order is seed-shuffled
  per run so an interrupted run samples across families instead of finishing all
  LightGBM variants first. `--max-models` / `--time-budget-min` bound a run.

**333 runs completed, 13 failed, 27 families** — the ledger is committed at
`artifacts/manifest.csv`.

And the ceiling was **0.8808**. Every gradient-boosted tree landed in
0.87–0.88 regardless of hyperparameters; the heavy-FE block bought almost
nothing. That flatness was the signal that the problem was not a tuning problem.

### 3.2 Phase 2 — the two things that actually worked

**(a) Per-value exact-value target encoding.** Replace every one of the 13
columns — the 6 categoricals *and* the 7 numerics — with, for each class, the
mean of that class's indicator over rows sharing its **exact value**. 13 columns
× 3 classes = 39 new features.

Encoding a numeric per exact value looks wrong: `step_count` has 12,807 distinct
values. It works here because the synthetic generator **resampled real discrete
values** from the source survey rather than adding continuous noise — so each
"numeric" is secretly a high-cardinality categorical, and at 690k rows there are
enough rows per value for the mean to carry real signal.

Scale is load-bearing. Screened on a 70k subsample the same feature measures as
noise (−0.0017) and was written off on that evidence; re-measured at full scale,
paired across folds, it is worth **+0.0012 on FT-Transformer, on all 5 folds**,
and it is what lifts the tree models from 0.88 to 0.9504. Two lessons paid for
in wall-clock: *a per-value encoder cannot be screened at small n*, and *a lever
tested only inside a composition has not been tested* (this one was first
bundled with a feature costing −0.0069, and the two cancelled to a convincing
zero).

The encoder is fit **inside each training fold** (out-of-fold within that fold,
so a row never encodes itself) and only *transforms* validation and test rows.

**(b) Modern tabular architectures, run as Kaggle kernels.** The local box could
not host these, so ~34 notebooks were generated programmatically by the
`scripts/build_*_kaggle.py` family, pushed to Kaggle, and their outputs pulled
back by a cron-driven monitor (`scripts/kaggle_monitor.py`, which rotates
between two Kaggle accounts via `scripts/kaggle_profile.py` to widen the GPU
quota — credentials live in `~/.kaggle-profiles/` and are never written to disk
by this repo).

Families explored: **RealMLP** (5 seeds + a capacity variant), **FT-Transformer**
(2 seeds, balanced-loss variant), **TabM**, **TabNet**, **TabPFN v3**, **TabFM**,
**YDF**, **EBM**, AutoGluon, Optuna-tuned HGB/XGB, TE-encoded
XGB/CatBoost/ExtraTrees/HGB seed-bags, an ordinal-decomposition HGB, a
missing-pattern router, a rule-lookup model, and an OOF-disagreement meta-model.
Each kernel directory holds its `.ipynb` and `kernel-metadata.json`.

Top OOF scores from this phase:

| model | OOF balanced accuracy |
|---|---|
| RealMLP (kernel, `kernel_fe_te`) | 0.950646 |
| RealMLP (external seed variant) | 0.950625 |
| TE-HGB seed-bag (`raw_exact_value_te`) | 0.950407 |
| XGB (rule features) | 0.950081 |
| YDF GBT | 0.949775 |
| TabM | 0.949150 |
| TabNet | 0.948404 |

Note how tightly clustered these are — 0.9484 to 0.9506 across wildly different
architectures. Once the exact-value TE features are present, the architecture
barely matters.

### 3.3 Phase 3 — ensembling, audited

With every model's OOF row-aligned, blending is array averaging.
`src/ensemble.py` supports random-subset search, top-k, all, manual and a
logistic-regression-on-logits stacker (leak-free via `PredefinedSplit`):

```bash
python -m src.ensemble --mode random --n-models 20 --n-trials 200   # search random N-model blends
python -m src.ensemble --mode topk   --n-models 15 --make-submission
python -m src.ensemble --mode stack  --n-models 40 --make-submission
python -m src.ensemble --mode all    --make-submission
```

The trap at this stage is choosing *which* models and *what* weights on the same
OOF rows you then report the score from. `scripts/audit_stable_ensemble.py`
avoids it with a **nested meta-fold audit**: inclusion and weights are fitted on
six deterministic meta-folds and scored on an untouched seventh. Deliberately
stricter than optimising on all OOF rows and quoting that number back.

Results (`artifacts/stable_ensemble_audit.json`,
`artifacts/strong_submission_1_check.json`):

| | balanced accuracy |
|---|---|
| best single model | 0.950646 |
| 11-model blend, weights fitted and scored on all OOF | 0.950868 |
| same blend, **nested audit** (held-out meta-fold) | 0.950740 |

So the honest ensembling gain over the best single model is **+0.0001**
(0.950646 → 0.950740) — not the +0.0002 the unaudited fit would have claimed.
The winning weights are consistently ~0.35 RealMLP / ~0.32 TE-HGB /
~0.25 RealMLP-seed2 / ~0.08 XGB-rule — three architectures, nothing exotic.

The last week's scripts are conservative post-processing on top of a strong
public anchor, each writing a JSON audit of exactly which rows it flips and why:

- `build_anchor_consensus_candidates.py` — keep the anchor everywhere except
  rows where an architecture-diverse panel (only models ≥0.9495 OOF, weighted so
  RealMLP can't win by seed count) disagrees.
- `build_final_push_candidates.py` — a stricter version: a flip needs agreement
  between two independently seeded FT-Transformers *and* a five-seed RealMLP
  bag, swept over margin thresholds 0.4–0.9.
- `build_inverse_frontier_probes.py` — disjoint single-tier probes designed so
  each LB submission measures one specific evidence tier rather than a mixture.
- `build_source_rule_blend.py` — blends in class rates from the real 50k source
  survey, keyed on the three decisive features. Nested-CV verdict: +0.00001,
  39 rows changed. Correctly *not* submitted.
- `build_public_95307_candidate.py` — reconstructs a published post-processing
  result from scratch and hash-verifies it, to establish what the public
  frontier actually was rather than trusting a score claim.

## 4. Live dashboard

```bash
python -m src.dashboard          # http://localhost:8765
python -m src.dashboard --port 9000
```

Stdlib `http.server` only, no extra dependencies. Re-reads
`artifacts/manifest.csv` on every poll (client refetches every 5s), so it tracks
`src.train` in real time across restarts and resumes: overall progress and ETA,
best CV so far, per-family breakdown, an OOF trend chart, recent completions,
and the watchdog's last log lines. `run_dashboard.bat` is the Windows shortcut.

## 5. Running it

```bash
pip install numpy pandas scipy scikit-learn pyyaml xgboost lightgbm catboost \
            matplotlib seaborn optuna pytorch-tabnet

# 1. put the competition data in data/
kaggle competitions download -c playground-series-s6e7 -p data && unzip -d data data/*.zip

# 2. build the 4 cached feature sets (~2 min on 690k rows)
python -m src.features

# 3. see what's pending without training
python -m src.train --dry-run

# 4. train — resumable, safe to Ctrl-C and rerun
python -m src.train
python -m src.train --max-models 50          # or in bounded chunks
python -m src.train --time-budget-min 180

# 5. explore ensembles over whatever has finished
python -m src.ensemble --mode random --n-models 20 --n-trials 200 --make-submission

# 6. submit
kaggle competitions submit -c playground-series-s6e7 -f submissions/<file>.csv -m "<message>"
```

**GPU:** flip `device.mode: gpu` in `config.yaml` and rerun `python -m src.train`.
LightGBM, XGBoost, CatBoost and TabNet derive their device params from that one
flag (`src/model_zoo.py::_device_params`); everything else is CPU-only either
way. Completed models are skipped via the manifest, so nothing is wasted.

**Parallelism:** `run.max_parallel_workers` in `config.yaml` trains N models in
separate processes, each capped to `cpu_count()//N` threads, with feature arrays
memory-mapped so they are shared rather than duplicated. Peak working set per
worker on the 450-column feature set is 1–2 GB, so `free_RAM_GB / 2` is a sane
ceiling.

The Kaggle-side experiments are generated, not hand-written — e.g.
`python scripts/build_realmlp_kaggle_experiment.py` writes
`kaggle_kernels/realmlp_seed2027_gpu/`, which is then pushed with
`kaggle kernels push`. Those kernels install their own extras
(`pytabkit`, `rtdl`, `tabm`, `tabpfn`, `ydf`, `interpret`, `catstat`,
`autogluon`) inside the Kaggle image.

## 6. Layout

```
config.yaml                  single source of truth: paths, folds, device, run limits
notebooks/eda.ipynb          full EDA
src/
  config.py                  loads config.yaml
  data.py                    raw I/O, target encode/decode
  folds.py                   fixed StratifiedKFold assignment
  features.py                builds and caches the 4 feature sets
  model_zoo.py               generates the 434 ModelSpecs
  estimators.py              ModelSpec -> fittable estimator, GPU/CPU param switching
  train.py                   resumable training loop (main entry point)
  ensemble.py                random / topk / all / manual / stack blending + submission writer
  dashboard.py               live progress dashboard
scripts/
  build_*_kaggle*.py         generate the Kaggle kernel notebooks
  build_*_candidates.py      build and audit conservative submission candidates
  audit_*.py                 nested-CV audits of ensembles and routers
  kaggle_monitor.py          cron poller: collect finished kernel outputs
  kaggle_profile.py          atomic Kaggle credential-profile switch
  watchdog.ps1 / install_watchdog.ps1   keep the local training loop alive (Windows)
kaggle_kernels/<name>/       one directory per Kaggle kernel: .ipynb + kernel-metadata.json
artifacts/
  manifest.csv               the results ledger — one row per attempted model
  *_audit.json               how each candidate submission was built, row by row
```

At runtime, `artifacts/` also fills with `folds.npy`, `features/`,
`oof/{model_id}.npy`, `test_preds/{model_id}.npy` and `logs/`; submissions land
in `submissions/`. None of that is committed.

## 7. Not in this repo

Excluded by `.gitignore`, all regenerable or third-party:

- **`data/`** — competition CSVs, plus `data/original_source/` (the real 50k
  student-health survey the synthetic data was generated from, used by
  `build_source_rule_blend.py`). Download from Kaggle.
- **`artifacts/features|oof|test_preds|logs/`** (~7.7 GB) and
  **`submissions/`** (~250 MB) — rebuilt by `src.features` / `src.train` /
  `src.ensemble`.
- **`kaggle_kernels/*/output/` and `kaggle_kernels/monitor_downloads/`**
  (~7 GB) — kernel run outputs pulled back by the monitor. The notebooks that
  produce them are committed.
- **`reference_notebooks/` and `public_research/`** — other people's public
  notebooks, studied during the run and referenced by some Phase-3 scripts.
  Fetch them with `kaggle kernels pull <slug>`:
  `artkomissar/students-be-healthy`,
  `najiama/post-processing-calibration-lb-0-95307`,
  `najiama/post-processing-calibration-lb-0-95288`,
  `yw8837/public-lb-0-95306-reproducible-29-row-cleanup`,
  `crystalbaby/lb-0-95289`,
  `anhadmahajan06/s6e7-post-processing-ensemble`,
  `anhadmahajan06/s6e7-knn-manifold-label-smoothing`,
  `yaaangzhou/kernel-2-realmlp-pytorch-implementation-cpu`,
  `nawfeelrahman1124444/ps-s6-ep6-realmlp-0-95090`,
  `beicicc/student-health-risk-public-ensemble`.

## 8. What generalises

- **A flat hyperparameter response is a message.** Twenty-two families all
  stuck at 0.88 meant the features were wrong, not the models.
- **Screen features at the scale you'll use them.** The winning encoder reads
  as pure noise on a 70k subsample and is worth +0.07 at 690k.
- **Test one lever at a time.** Two features worth +0.0012 and −0.0069 measure
  as zero together, and you conclude the wrong thing about both.
- **Fix the CV split before the first model.** Everything downstream —
  blending, stacking, nested audits, seed bags — is only possible because every
  OOF array is row-aligned to the same folds.
- **Audit the ensemble on rows the weights never saw.** The unaudited blend
  reads 0.95087; the honest number is 0.95074. Small here, but it is exactly
  the gap that turns into a private-LB drop.
