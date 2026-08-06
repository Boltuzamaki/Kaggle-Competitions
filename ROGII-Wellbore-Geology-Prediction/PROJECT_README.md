# ROGII — Wellbore Geology Prediction

Kaggle featured competition, $50k, closed 2026-08-05. Predict **TVT** (the
wellbore's stratigraphic position inside the rock column) along the hidden
~75% of each horizontal lateral, from the well's gamma-ray log, its XYZ
trajectory, and a paired vertical typewell.

Metric: pooled row-wise RMSE in feet, over 14,151 hidden rows.

## Result

| | score |
|---|---|
| Final public LB | **6.463** |
| Best honest scratch-only stack | 8.129 |
| Hold-last-known-TVT baseline | 15.883 |

## What is in here

| path | contents |
|---|---|
| `EXPERIMENT_LEDGER.md` | the authoritative research log — every accepted, rejected and invalid experiment, with scores and diagnoses |
| `README.md` | task description, EDA findings, baseline model |
| `STRATEGY.md` | early strategy and the honest-vs-overlap analysis |
| `SUBMISSION_PROTOCOL.md` | submission checklist and audit requirements |
| `history.md` | chronological narrative of the work |
| `HANDOFF_FINAL_DAY.md` | final-day state and decision framework |
| `exp/*.py` | ~200 experiment scripts (CV harnesses, trackers, feature families) |
| `exp/results/**` | per-experiment JSON/markdown summaries |
| `kernels/**` | Kaggle kernel builders and metadata |
| `notebooks/` | self-contained baseline notebook |

Competition data is **not** committed — download it with
`kaggle competitions download -c rogii-wellbore-geology-prediction -p data`.

## Findings worth keeping

Most of the value here is negative results and diagnostics, which are recorded
in full in `EXPERIMENT_LEDGER.md`. The ones that generalise:

**The leaderboard has a measurable noise floor of ~0.037 ft.** Resubmitting an
unchanged kernel returns a different score. Two single draws need ~0.10 ft
before a difference means anything. About one team in ten sat within ±0.037 ft
of a single score, so ordering in that band is largely an artefact.

**CV and LB can be anti-correlated on this task.** One documented case: an
out-of-fold score improved 8.248 → 7.623 while the public score moved the wrong
way, 6.675 → 6.924, at rank correlation −0.243.

**Public score claims did not reproduce.** A notebook advertising 6.390 scored
6.465 when rerun verbatim from a different account.

**The visible test files are placeholders.** They are byte-identical copies of
three training wells, but the scored data is different — proven by submitting a
guarded exact-match model, which correctly refused to fire and returned exactly
the hold-anchor baseline of 15.883.

**A quantified identifiability barrier.** Curve capacity is ample (per-well
oracles reach 3–6 ft) but the coherent per-well datum/slope is not predictable
out-of-well from any legal feature set tried: R² ≈ 0 against a required ≈0.32.
This is the reason the honest ceiling sat near 8.

**Two of my own mid-project conclusions were wrong and were retracted in the
ledger** — an inferred "pure twin copy scores ~7.04", and a shift-versus-score
slope fitted across notebooks that turned out to be confounded. Both retractions
are kept in place deliberately, with the evidence that overturned them.

## Attribution

Parts of this work study, reimplement or fork public Kaggle notebooks from the
competition (the Harshini, Pilkwang, Fleongg, ravaghi and related lineages).
Where a script reimplements a public method it is named for that lineage.
Verbatim copies of other authors' notebooks are not included in this
repository; see `EXPERIMENT_LEDGER.md` for the provenance audits, including
which public artifacts were rejected as inadmissible model sources.
