# Predicting Smartphone Addiction — Kaggle Playground S6E8

Binary classification: estimate the probability that a participant is
smartphone-addicted (`addicted_label`) from demographic, screen-time, app-use,
sleep, stress and academic/work-impact variables. 691,369 training rows,
296,302 test rows, 12 predictors, ROC AUC.

**Public leaderboard: 0.97080**, from a 72-stream cross-fitted logistic stack.
The first CatBoost baseline in this repo scored 0.96332, so the ensemble work is
worth about +0.0075 AUC on a leaderboard where the top score is 0.97115 and a
large cluster sits at 0.97086.

## Approach

Every model writes an out-of-fold prediction over the training rows and a
prediction over the test rows. A cross-fitted L2 logistic regression over the
gauss-ranked streams produces the final submission. All streams are built from
official competition data only.

The interesting problem is not accuracy, it is **selection**: with 78 candidate
streams whose mutual rank correlations reach 0.99, most additions are worth
between +0.000001 and +0.00001, while the cross-fitted estimate itself has a
noise floor around 0.0000033. A single before/after refit cannot resolve that.
`ensemble_original/validation_audit.py` therefore runs the whole stack with and
without each candidate on identical folds across eight meta-fold seeds and
reports a paired t-statistic. A stream is kept only if its paired gain is
positive with |t| > 3.

## What the experiments actually showed

The full ledger is in [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md), including every
rejected idea. The results worth repeating elsewhere:

**A rejected architecture is often a rejected *view*.** Four neural
architectures were recorded as failures at 0.938–0.940: FT-Transformer, DCNv2,
GANDALF and TabR, all trained target-free. Re-running FT-Transformer unchanged
on a fold-safe target-encoded view scored **0.9673** — the same architecture,
+0.027, and the strongest network in the inventory. The deficit belonged to the
features, not the model. Not every architecture was rescued this way: DCNv2 came
back at 0.9484 and stayed rejected.

**Decorrelation only pays inside a narrow accuracy band.** A LightGBM within
0.0002 of the best stream contributed nothing (t=+2.8). Bagged ExtraTrees
sitting 0.0067 *below* it contributed more than twice as much (t=+5.8). But a
kernel machine at 0.9515 and an additive linear model at 0.9578 both made the
stack worse. Being different is only valuable above a floor somewhere around
0.960; below it, the disagreements are noise rather than information.

**Standalone accuracy does not predict stack contribution.** The FT-Transformer
is the most accurate network here (0.9673) but pays only +0.0000020 (t=+4.3),
while a weaker residual MLP (0.9664) pays +0.0000068 (t=+9.9). Candidates must
be ranked by paired t, never by leaderboard-style accuracy.

**More rows per model is the most reliable lever.** Moving from 5 to 10 outer
folds was worth +0.00018 on XGBoost, +0.00025 on the lookup transformer and
+0.00016 on the FT-Transformer. Twenty folds was tried on the strongest family
and is recorded in the ledger.

**Two things that produced nothing, tested properly.** A feature-engineering
screen of four independent groups (peer-relative deviation, percentile rank,
fitted residuals, duplicate-row counts) returned a best delta of +0.0000094 with
two groups negative. A 15-trial random search over the network family found
nothing the hand-set architecture did not already have. The remaining headroom
was in model families, not in columns or hyperparameters.

**The leaderboard offset is not a constant.** Public LB tracks cross-fitted OOF
with an offset that shrinks as OOF rises: +0.00100, then +0.00098, then +0.00096.
Estimates built on a stale offset overstate the leaderboard by more than several
recent gains were worth.

## Layout

```
ensemble_original/     stacking, weight fitting, and the paired validation audit
cpu_kernel_arch_te/    multi-architecture kernel family (resnet, cnn1d, dae,
                       FT-Transformer, gated, DCNv2, NODE) over one shared encoder
cpu_kernel_*/          one self-contained Kaggle script kernel per model
gpu_*/, local_*/       local GPU experiments, mostly the lookup transformer
publish_*/             notebooks prepared for publication on Kaggle
EXPERIMENT_LOG.md      append-only ledger: what was tried, what it scored, verdict
NOTES.md               working notes
```

Kernel directories whose `experiment.py` carries a `# generated` marker are
emitted by the neighbouring `build_kernels.py`; a Kaggle script kernel must be a
single self-contained file, so each variant is a full copy by necessity.

## Reproduce

Competition data is not committed. Download it from Kaggle first:

```bash
kaggle competitions download -c playground-series-s6e8 -p . && unzip playground-series-s6e8.zip
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python pandas numpy scikit-learn lightgbm xgboost catboost torch
.venv/bin/python train_baseline.py                        # CatBoost baseline, LB 0.96332
.venv/bin/python ensemble_original/stack_weights.py       # fit stack weights over available streams
.venv/bin/python ensemble_original/validation_audit.py X  # paired t-test for candidate stream X
```

The stacking scripts read per-stream OOF/test CSVs under `artifacts/`, which are
regenerated by the kernels rather than committed.

## Not included

Prediction files, trained weights, and competition data are excluded — they are
large and reproducible. Third-party material is also excluded: several public
notebooks and prediction files from other Kaggle users were downloaded during
this work for study and benchmarking, and republishing them here is not mine to
do. Where their ideas informed an experiment, the ledger says so by name.
