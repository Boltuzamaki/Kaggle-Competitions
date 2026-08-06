# ROGII Wellbore Competition: Experiment History

Last updated: 2026-07-23

This document records the research, implementations, Kaggle runs, validation
results, submissions, negative results, and current conclusions from this
workspace. It is intended to prevent repeating experiments and to distinguish:

- a real Kaggle leaderboard result;
- an offline cross-validation result;
- an output inferred to be strong but not yet submitted;
- a public-leaderboard shortcut that may not generalize.

No Kaggle API credentials or other secrets are recorded here.

---

## Kaggle accounts used

Two Kaggle accounts were used to run this project's notebooks. Their usernames
are easy to confuse because they differ by one letter:

| Project label | Exact Kaggle username | Primary use |
|---|---|---|
| Main account / Boltuzamaki | `boltuzamaki` | Baselines, submissions, contact pipeline, R3/R4, R5, and R6 |
| Secondary account / EzioAuditore | `boltuzmaki` | Parallel E01-E06 experiment matrix and combined external-data CV |

### Main account: `boltuzamaki`

The currently configured Kaggle CLI profile belongs to:

```text
boltuzamaki
```

Its local authentication file is outside this repository:

```text
C:\Users\chand\.kaggle\kaggle.json
```

The file contains credentials and must not be copied into this workspace,
committed to Git, or pasted into this history.

Main-account kernels represented in this workspace include:

- `rogii-wellbore-geo-particle-filter-no-leak`;
- `rogii-wellbore-geo-physics-stack`;
- `rogii-wellbore-geo-physics-stack-infer`;
- `rogii-contact-gated-stratigraphic-alignment`;
- `rogii-deterministic-contact-base`;
- `rogii-decorrelated-pf-ensemble`;
- `rogii-external-geometry-transfer-gpu-r2`;
- `rogii-current-neural-stack-integration-r3`;
- `rogii-r4-validated-neural-contact-probe`;
- `rogii-r5-clean-stack-audit`;
- `rogii-r5-anchored-surface-cnn`;
- `rogii-r5-prefix-feature-contact-v2`;
- `rogii-r6-warp-cross-attention`.

The verified competition submission table in Section 5 belongs to this main
account.

### Secondary account: `boltuzmaki`

The user-facing label used for this account was `EzioAuditore`. Its exact
Kaggle username in the notebook metadata is:

```text
boltuzmaki
```

This is not the same username as `boltuzamaki`; the secondary username omits
the second `a`.

The account was authenticated for the parallel experiment batch through the
`KAGGLE_API_TOKEN` environment variable. The token value is intentionally not
stored in this document. A fresh shell may require the environment variable to
be supplied again.

Secondary-account kernels represented in this workspace are:

- `rogii-combined-geology-cv-r1`;
- `rogii-e01-stack-contact-control`;
- `rogii-e02-multiref-contact`;
- `rogii-e03-prefix-bounded`;
- `rogii-e04-heel-bimodal`;
- `rogii-e05-spatial-surface`;
- `rogii-e06-gpu-candidate-ranker`.

Their notebook definitions are under `kernels/ezio_matrix/`, and downloaded
results are under `kernels/ezio_results/`. No secondary-account leaderboard
submission is recorded in this workspace.

### Public source accounts

The following usernames appear under `ilog/` because their public notebooks
were downloaded for research. They are source authors, not accounts controlled
by this project:

- `ravaghi`;
- `fleongg`;
- `pilkwang`;
- `foysalemonshanto`;
- `canqiang`;
- `chiekhalloul`;
- `beicicc`;
- `yusuketogashi`;
- `wbfranci`;
- `kaiwalyaatulraut`;
- `ayodejiibrahimlateef`.

These public authors should not be confused with the two accounts used to run
our own experiments.

---

## 1. Competition understanding

The task is to predict `TVT` for rows in each horizontal well after the
prediction-start point, where `TVT_input` becomes missing.

Inputs available at prediction time include:

- measured depth `MD`;
- trajectory coordinates `X`, `Y`, and `Z`;
- horizontal-well gamma ray `GR`;
- the visible `TVT_input` prefix;
- a paired typewell with `TVT`, `GR`, and formation information;
- formation/contact columns such as `ANCC`, `ASTNU`, `ASTNL`, `EGFDU`,
  `EGFDL`, and `BUDA`.

The output is a two-column submission:

```text
id,tvt
```

The competition metric is pooled, row-weighted RMSE. Long hidden tails
therefore have more influence than short wells.

Early in the project, the data flow and input/output relationship were
explained with diagrams and EDA figures. The main local figures are under
`eda_figs/`, and the strategy summary is in `STRATEGY.md`.

---

## 2. Foundational observations

### 2.1 Predict the residual, not absolute TVT

The useful target for ordinary models is:

```text
dTVT = TVT - last_visible_TVT
```

The flat last-known-TVT trajectory is a strong baseline. A useful model must
correct drifting wells without introducing movement on already-flat wells.

### 2.2 Structural-surface decomposition

Define:

```text
U = TVT + Z
```

The `-Z` component supplies much of the local wiggle. The difficult part is
forecasting the slowly varying structural surface `U` far beyond the visible
prefix. This became the main interpretation of the Kaggle write-up:
"The Wiggle Is Free, the Trend Is the Wall."

### 2.3 Public test overlap

All 14,151 scored rows in the current public test belong to three well IDs that
also appear in the training data:

- `000d7d20`: 3,836 rows;
- `00bbac68`: 6,014 rows;
- `00e12e8b`: 4,301 rows.

Several public notebooks reconstruct the hidden test trajectories using
same-well training TVT or a contact formula calibrated with same-well training
TVT. This is very strong on the public surface but is not evidence that the
method will work on unseen private wells.

### 2.4 Contact reconstruction

For a formation/contact `c`, the frequently used formula is:

```text
TVT_contact = TVT_typewell_contact - (Z - contact_column) + well_bias
```

The important distinction is how `well_bias` is estimated:

- public-aggressive version: estimate it from the matching training well's
  true `TVT`;
- target-free version: estimate it using only contact features and the
  visible test `TVT_input` prefix.

---

## 3. Initial baselines and local model screening

### 3.1 Tabular baselines

Models were trained on grouped whole-well splits. The results in
`exp/results/` were:

| Model | Mean per-well RMSE | Pooled RMSE | Verdict |
|---|---:|---:|---|
| Ridge | 12.6333 | 15.5924 | Weak |
| XGBoost | 11.9662 | 14.8352 | Better than Ridge, still weak |
| CatBoost | 11.9681 | 14.8867 | Similar to XGBoost |
| LightGBM | 11.9984 | 14.8774 | Similar to XGBoost |

Plain tabular models did not capture the trajectory physics sufficiently.

### 3.2 Sequence models

The following sequence architectures were trained:

| Model | Mean per-well RMSE | Pooled RMSE | Verdict |
|---|---:|---:|---|
| ConvGRU | 10.3393 | 13.2168 | Best standalone NN in the first screen |
| TCN | 10.9572 | 13.8256 | Useful but behind ConvGRU |
| TCN + Transformer | 11.3363 | 14.1265 | No improvement |
| Transformer | 11.4715 | 14.1557 | No improvement |

The sequence models beat the simple tabular models but did not beat the
particle-filter pipeline.

### 3.3 Formation-plane prior

An honest leave-one-well-out spatial formation-plane prior was tested. It was
very poor, around 38 ft RMSE in the early evaluation. The strong "formation"
signal in several public notebooks was not coming from spatial interpolation;
it was coming from each test well's own contact columns or matching train well.

Verdict: do not use an ordinary cross-well formation-plane prior as a primary
trajectory.

---

## 4. Particle-filter and physics pipeline

The main honest pipeline developed through:

- momentum particle filtering;
- multiple GR likelihood scales;
- beam/DP continuity paths;
- last-known-TVT hold;
- PF/beam blending;
- multi-seed averaging;
- smoothing and bounded extrapolation.

Important local scripts include:

- `exp/pf_sweep.py`;
- `exp/pf_final_cv.py`;
- `exp/pf_tracker.py`;
- `exp/beam_tracker.py`;
- `exp/compute_signals.py`;
- `wellbore_lib.py`.

The strongest early honest public submission was the PF + beam blend at
**9.072**. Increasing particle count, seeds, and changing the blend did not
consistently improve the leaderboard.

---

## 5. Verified Kaggle submissions from our account

The submission history verified through the Kaggle API is:

| Date | Submission | Public RMSE | Result |
|---|---|---:|---|
| 2026-07-01 | LightGBM `dTVT` baseline | 14.864 | Weak baseline |
| 2026-07-01 | Visible-well/direct train-twin trajectory | No displayed score | Not used as evidence |
| 2026-07-01 | Honest PF + beam blend | **9.072** | Best early honest submission |
| 2026-07-01 | Direct train-TVT copy resubmission | No displayed score | Not used as evidence |
| 2026-07-01 | Tuned PF, scale 12, 1,000 particles, 160 seeds | 9.284 | Worse than 9.072 |
| 2026-07-18 | Contact-gated structural alignment v1 | **8.940** | Best submitted score on our account |
| 2026-07-21 | 90% contact + 10% validated neural correction | 9.430 | Regression; rejected |

The current best verified score on our account is therefore **8.940**.

---

## 6. Public notebook and discussion research

Kaggle discussions, notebooks, and full end-to-end pipelines were inspected.
Pure blends of other people's submission files were treated as low-value
evidence. The main code lineages studied were:

- Ravaghi Ridge, LightGBM, and hill-climbing baselines;
- Fleongg pretrained trajectory models;
- Pilkwang dual-track prefix-calibrated pipeline;
- Foysal PF/contact/gold-calibration notebooks;
- Canqiang `rogii-det-mha140b`;
- Chiekhalloul `rogii-det-mha140b-exact-r1`;
- Beicicc MHA240/MHA260 variants;
- the public "Wiggle Is Free, Trend Is the Wall" write-up;
- later V586/V599/PF-branch continuation notebooks.

Downloaded source notebooks are under `ilog/`.

### 6.1 Meaning of MHA in these notebooks

In the inspected MHA240 notebook, MHA did not mean a multi-head-attention
network. It meant a midpoint-hedge strength/alpha setting.

For the MHA240 run:

```text
midpoint hedge: 0/3 wells qualify
```

The named MHA240 modification therefore changed no rows. Its final score was
caused by upstream contact reconstruction and a global bias change.

### 6.2 Same-well contact override audit

The public pipeline applied same-well contact reconstruction redundantly in
multiple places:

1. an early same-well physical trajectory;
2. a guarded contact override;
3. a contact candidate inside gold calibration;
4. a final contact re-application.

The visible-prefix contact fits were approximately:

| Well | Prefix RMSE | Replaced rows |
|---|---:|---:|
| `000d7d20` | 0.0101 | 3,836 |
| `00bbac68` | 0.0090 | 6,014 |
| `00e12e8b` | 0.0079 | 4,301 |

All public rows were replaced.

### 6.3 MHA240 score investigation

The Beicicc version 4 notebook was verified in Kaggle as scoring **6.848**.
Its final `submission.csv` hash is:

```text
7F035F1B633601455E9BAB9AEA0862A87B414218503CD71F8B77E9FAD7A9264C
```

That file is byte-for-byte identical to the locally downloaded
Chiekhalloul Exact-R1 output:

`ilog/chiekhalloul_det_mha140b_exact_r1_output/submission.csv`

The notebook applies a global `-0.40 ft` correction. Its own comments state
that the shift was inferred using a public-leaderboard probe, so it is not a
private-safe calibration.

### 6.4 Later public frontier

Several later notebooks are not independent new models. They share the same
large PF/contact/gold pipeline and make small controlled changes to one well.

Observed artifacts:

| Artifact | Change | Hash / score evidence |
|---|---|---|
| V586 no-bias contact artifact | Removes the `-0.40 ft` shift | Advertised around 6.768 |
| PF branch artifact | Adds exactly `+2.0 ft` to all 4,301 rows of `00e12e8b` | Kaggle best version reported 6.593 |
| PF branch continuation | Adds `+2.5 ft` instead | Latest version scored 6.624, worse |

The `+2.0 ft` branch was selected using a bimodal PF seed distribution:

```text
well        separation   mass_low   mass_high   applied shift
00e12e8b    29.439 ft    0.7213     0.2787      +2.0 ft
```

The other two wells did not qualify. The regression from `+2.0` to `+2.5`
shows that continuing to tune the shift on the public leaderboard is
overfitting.

Downloaded frontier notebooks and outputs are under:

`ilog/frontier_20260722/`

---

## 7. Contact-gated stratigraphic alignment

The production-style notebook combined:

1. ridge/PF trajectory;
2. selector/PF trajectory;
3. an SP45-style blend;
4. robust projection of `U = TVT + Z`;
5. pretrained learned trajectory models;
6. guarded contact reconstruction;
7. visible-prefix calibration;
8. final submission audit.

The primary implementation is under:

`kernels/contact_train/`

The contact-gated v1 submission scored **8.940**.

The main lesson was that contact overrides dominate earlier pipeline stages.
If the final contact re-application accepts a well, changes made by PF, neural,
visible-prefix, or spatial stages before that point may disappear from the
final CSV.

---

## 8. Ezio six-experiment Kaggle matrix

Six experiments were designed to run across the available Kaggle CPU/GPU
quota. Their completed outputs are under `kernels/ezio_results/`.

### E01: stack/contact control

- Reproduced the contact-gated control.
- All three wells passed the EGFDU prefix guard.
- Final hash:
  `2B86386F19279E79E7184096F353CCF2B97785DE67B268CAA56AA5F85405A815`.

### E02: multi-reference contact

- Tested all six formation references.
- Selected:
  - `EGFDL` for `000d7d20`;
  - `ANCC` for `00bbac68`;
  - `ASTNU` for `00e12e8b`.
- Prefix RMSE remained around 0.008-0.010.
- Produced a different final hash, but it was not promoted because there was
  no honest suffix evidence that the alternative contacts improved hidden TVT.

### E03: bounded visible-prefix correction

- Surface-trend correction passed on two wells.
- Proposed moves were bounded to 20% of the candidate correction.
- The later contact re-application restored the contact trajectory.
- Final CSV was identical to E01.

### E04: heel/bimodal GR hedge

- Tested bimodal datum/GR ambiguity logic.
- No accepted final change survived the contact layer.
- Final CSV was identical to E01.

### E05: spatial structural surface

- Tested a spatially informed surface continuation.
- It did not survive the final contact policy.
- Final CSV was identical to E01.

### E06: GPU candidate ranker

- Ranker OOF RMSE: 11.1406 versus stack OOF 11.2355.
- Absolute gain: 0.0949 ft.
- Worst-decile RMSE worsened from 14.8079 to 15.6669.
- Promotion gate failed.
- Final test CSV was identical to E01 because contact remained final.

Overall conclusion: the matrix did not produce a safe improvement over the
contact anchor.

---

## 9. External-data investigation

The following external sources were downloaded or inspected:

### 9.1 Geology Forecast Challenge

Local location:

`external_data/geology_forecast_challenge/`

Findings:

- 123 independent raw structural horizon curves;
- no gamma ray;
- geometry and slope statistics resemble the competition's `TVT + Z`
  structural surface;
- useful for near-prefix geometry continuation;
- recursive long-horizon forecasts drift.

Against simple linear extrapolation, the external geometry Ridge branch
improved:

| Horizon | Linear RMSE | External candidate RMSE |
|---:|---:|---:|
| 300 ft | 2.634 | 1.814 |
| 600 ft | 4.753 | 3.838 |
| 1,200 ft | 9.121 | 8.033 |
| 2,400 ft | 18.081 | 16.863 |
| 4,800 ft | 34.743 | 33.182 |

These gains were against a simple extrapolator, not the full PF/contact stack.

### 9.2 Combined-data nested CV R1

Current-well surface windows were trained together with external curves.
Five outer folds were grouped by current competition well.

| Horizon | Current only | Combined |
|---:|---:|---:|
| 300 ft | 1.7828 | 1.7811 |
| 600 ft | 3.7266 | 3.7262 |

Four of five folds selected the minimum external weight. The improvement was
negligible, so R1 was rejected.

Artifacts:

`external_data/full_cv_r1/`

### 9.3 External neural pretraining R2

An MLP was pretrained on the 123 external curves and fine-tuned on all 773
competition wells using five grouped outer folds.

| Metric | Scratch | Transfer |
|---|---:|---:|
| 300-ft RMSE | 1.7522 | 1.7539 |
| 600-ft RMSE | 3.6280 | 3.6615 |
| 1,200-ft decay RMSE | 7.6065 | 7.7300 |
| 2,400-ft decay RMSE | 16.0403 | 16.2905 |

Transfer was worse at every promotion horizon and won on only 48.25% of wells
at 600 ft. It was rejected.

Artifacts:

`kernels/external_pretrain_gpu_r2/output_v2/`

### 9.4 FORCE 2020

Local location:

`external_data/force_2020/`

Findings:

- 118 LAS wells;
- roughly 2.30 million usable GR samples;
- substantial basin/acquisition domain shift;
- potentially useful for masked-GR self-supervised pretraining;
- unsuitable as direct TVT supervision.

### 9.5 Geosteering World Cup 2021

Local location:

`external_data/geosteering_world_cup_2021/`

Findings:

- thousands of human interpretation snapshots;
- only two independent geological scenarios;
- no raw GR in the interpretation CSVs;
- possibly useful for uncertainty envelopes or trajectory smoothness;
- not thousands of independent training wells.

### 9.6 ROGII/GWC 2020 typelog

The original Dataverse site was unreliable, but a local repository copy was
available during the audit. It contains one synthetic typelog and associated
GR/log data.

Best proposed use:

- datum shifts;
- local stretch/compression;
- gain and offset drift;
- smoothing/dropout;
- nonlinear GR warp augmentation.

It is not enough data for direct supervised TVT training.

Full audit:

`external_data/USABILITY_AUDIT.md`

---

## 10. Current-only neural integration R3 and R4

### R3: grouped OOF integration

A current-only MLP was used as a small distance-decayed correction to the
PF/beam/GR residual stack.

| Metric | Stack | Neural blend |
|---|---:|---:|
| Full pooled RMSE | 11.2118 | 11.1043 |
| 600-ft RMSE | 3.5597 | 3.3673 |
| 1,200-ft RMSE | 5.3217 | 5.1008 |
| Per-well p90 | 14.9182 | 14.6460 |

All five folds improved and 58.7% of wells improved. R3 passed the offline
promotion gate.

Artifacts:

`kernels/neural_stack_integration_r3/output_v1/`

### R4: leaderboard integration

The promoted R3 correction was refit and applied conservatively:

```text
90% contact + 10% neural correction
```

Public score:

```text
9.430
```

This was worse than the 8.940 contact submission. The offline gain did not
transfer when mixed into the contact-dominated public trajectory.

Conclusion: do not blend this neural correction into the current contact
submission in the same way.

---

## 11. "Wiggle Is Free" R5/R6 experiment series

The local plan is documented in:

- `kernels/wiggle_r5/WRITEUP_TO_6_PLAN.md`;
- `kernels/wiggle_r5/EXPERIMENT_QUEUE.md`.

### R5 clean-stack audit

- Strict no-contact baseline.
- Pooled RMSE: 11.2789.
- Mean well RMSE: 8.6876.
- No promotable improvement.

Artifacts:

`kernels/wiggle_r5/rogii-r5-clean-stack-audit/output_v1/`

### R5 anchored surface CNN

The network predicted an anchored structural surface.

| Metric | Stack | CNN blend |
|---|---:|---:|
| Full pooled RMSE | 11.2043 | 11.1044 |
| Absolute gain | | 0.09993 |
| Fold wins | | 4/5 |
| Per-well p90 | 14.8836 | 14.9109 |

It narrowly missed the 0.10-ft promotion threshold and worsened p90. The
standalone geometry network was much worse at long range. It was rejected.

Artifacts:

`kernels/wiggle_r5/rogii-r5-anchored-surface-cnn/output_v1/`

### R6 WARP cross-attention

The WARP branch used trajectory windows and typewell/GR cross-attention.

| Metric | Stack | WARP blend |
|---|---:|---:|
| Full pooled RMSE | 11.2065 | 11.1212 |
| Absolute gain | | 0.0853 |
| Relative gain | | 0.761% |
| Fold wins | | 5/5 |
| 600-ft RMSE | 3.5618 | 3.3711 |
| 1,200-ft RMSE | 5.3269 | 5.1052 |
| Per-well p90 | 14.8684 | 14.8958 |

The signal was consistent across folds, but the absolute gain was too small and
p90 slightly worsened. The current form was rejected.

Artifacts:

`kernels/warp_r6/output_v1/`

Important lesson: a future WARP model should predict residuals relative to the
final contact anchor and be applied after contact/gold calibration, rather than
predicting the complete trajectory before a contact layer that may overwrite
it.

---

## 12. R5 target-free prefix/contact experiment

Notebook:

`kernels/wiggle_r5/rogii-r5-prefix-feature-contact/`

This was the most important new result from the R5/R6 queue.

It estimated a contact trajectory without reading same-well training TVT:

| Well | Accepted | Prefix audit RMSE | Training TVT used? | Rows replaced |
|---|---:|---:|---:|---:|
| `000d7d20` | Yes | 0.01209 | No | 3,836 |
| `00bbac68` | Yes | 0.00924 | No | 6,014 |
| `00e12e8b` | Yes | 0.01166 | No | 4,301 |

Reference formation:

```text
EGFDU
```

The final output differs from the published no-bias V586 artifact by only a
constant, well-specific amount:

| Well | Mean difference |
|---|---:|
| `000d7d20` | -0.00862 ft |
| `00bbac68` | -0.00607 ft |
| `00e12e8b` | -0.00859 ft |

This file has not yet been submitted from our account:

`kernels/wiggle_r5/rogii-r5-prefix-feature-contact/output_v2/submission.csv`

It is the strongest target-free/public candidate currently produced by our own
experiment queue. Its near identity to published strong artifacts suggests it
should score far better than 8.940, but that remains an estimate until an
actual submission is scored.

Generalization caveat: it has excellent prefix self-verification on the three
current test wells, but it still requires rolling suffix validation across the
773 training wells before being called private-safe.

---

## 13. Main negative results

The following ideas were tried or investigated and should not be repeated
without a materially different formulation:

- plain Ridge/XGB/CatBoost/LightGBM trajectory prediction;
- standalone Transformer-heavy sequence models;
- honest spatial formation-plane interpolation;
- direct external-data row concatenation;
- external geometry neural pretraining;
- full-trajectory anchored CNN;
- the current complete-trajectory WARP implementation;
- adding a small neural correction to the public contact trajectory;
- wall/tail shrink toward the last visible TVT;
- `pf_z` as a third correction after WARP and physics postprocessing;
- large kitchen-sink PF ensembles;
- unrestricted high-degree structural trend extrapolation;
- blindly selecting the best contact from prefix RMSE alone;
- increasing the public PF branch shift from `+2.0` to `+2.5`;
- making corrections before a final contact layer that later overwrites them.

---

## 14. What appears to work

The strongest recurring signals are:

1. a robust particle-filter trajectory;
2. averaging decorrelated PF seeds/configurations;
3. using the visible prefix for calibration and rejection;
4. the physical contact equation;
5. predicting or smoothing `U = TVT + Z`;
6. small, decorrelated residual corrections rather than replacing the full
   trajectory;
7. applying useful corrections after gold/contact calibration;
8. using PF seed spread and bimodality as uncertainty evidence.

---

## 15. Current recommended path

### Immediate score check

Submit the R5 feature-only contact output to obtain a real leaderboard score:

`kernels/wiggle_r5/rogii-r5-prefix-feature-contact/output_v2/submission.csv`

This has not been submitted as of this document's update.

### Generalization experiments

1. **Rolling-suffix contact CV**
   - Evaluate 50%, 65%, 75%, and competition-matched cut points.
   - Fit contact bias only from the allowed visible prefix.
   - Report pooled RMSE, per-well p90, and worst wells.

2. **Multi-contact consensus**
   - Generate EGFDU, EGFDL, ASTNL, ANCC, and BUDA candidates.
   - Select or weight them using held-out visible-prefix suffixes.
   - Reject candidates when formations disagree.

3. **Generalized PF branch hedge**
   - Detect two PF modes from seed distributions.
   - Cross-fit branch strength and cap.
   - Never hardcode a well ID or a leaderboard-derived shift.

4. **Contact-residual WARP**
   - Target `TVT - TVT_feature_contact`.
   - Use small weights such as 0.05, 0.10, 0.20, and 0.30.
   - Apply after the final contact/gold layer.

5. **Post-contact structural residual smoothing**
   - Fit a robust low-degree residual model in `U = TVT + Z`.
   - Use a warm-up ramp and strict extrapolation caps.

### What not to spend quota on next

- another generic sequence architecture;
- more external pretraining without a new transfer objective;
- more full-trajectory CNNs;
- more public shift tuning;
- another broad blend of published submission files.

---

## 16. Key workspace locations

| Purpose | Location |
|---|---|
| Overall strategy | `STRATEGY.md` |
| Main README | `README.md` |
| Initial experiments | `exp/` |
| Honest/PF notebooks | `kernels/honest/`, `kernels/stack/` |
| Contact training | `kernels/contact_train/` |
| Ezio experiment matrix | `kernels/ezio_results/` |
| External-data audit | `external_data/USABILITY_AUDIT.md` |
| External pretraining | `kernels/external_pretrain_gpu_r2/` |
| Neural integration | `kernels/neural_stack_integration_r3/` |
| R4 contact/neural probe | `kernels/neural_contact_probe_r4/` |
| R5 experiments | `kernels/wiggle_r5/` |
| R6 WARP | `kernels/warp_r6/` |
| Downloaded public notebooks | `ilog/` |
| Current frontier downloads | `ilog/frontier_20260722/` |
| Kaggle write-up copy | `The Wiggle Is Free, the Trend Is the Wall _ Kaggle.html` |

---

## 17. Current state in one paragraph

The best verified submission from our account is the 8.940 contact-gated
trajectory. Generic tabular models, sequence models, external transfer,
anchored CNN, and the first WARP attempt did not produce a safe improvement.
The strongest newly completed result is the target-free feature-contact
trajectory: it uses no same-well training TVT, passes very tight visible-prefix
audits on all three test wells, and is numerically almost identical to a
published strong public artifact. The next decision should be to score that
file, then concentrate future Kaggle quota on rolling-suffix contact
validation, a generalized PF bimodal hedge, and residual modeling after the
contact anchor.
