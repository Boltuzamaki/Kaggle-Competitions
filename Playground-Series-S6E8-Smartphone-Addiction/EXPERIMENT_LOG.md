# S6E8 experiment log

Append-only record of what was tried, what it scored, and whether it was kept.
Negative results are as important as positive ones here: most of the compute in
this project has gone into ideas that did not work, and re-running them is the
main way to waste a day.

Read this before proposing a new experiment. If an idea appears below with
verdict REJECTED, do not repeat it without a stated reason why the conditions
have changed.

Append with `./log_experiment.py` or by hand. Keep the columns.

---

## Standing facts

**OOF to LB calibration.** Public LB = honest cross-fitted OOF + 0.00099.
Confirmed five times, each within 0.00002:

| Stack | OOF | Predicted LB | Actual LB |
| --- | --- | --- | --- |
| S003 rank blend | 0.9692864 | 0.97028 | 0.97029 |
| 40-stream logistic | 0.9696223 | 0.97061 | 0.97060 |
| 42-stream bagged | 0.9696482 | 0.97064 | 0.97062 |
| 45-stream bagged | 0.9696691 | 0.97066 | 0.97064 |
| 47-stream bagged | 0.9697477 | 0.97074 | 0.97073 |
| 72-stream logistic | 0.9698402 | 0.97082 | 0.97080 |

The offset shrinks monotonically as OOF rises: +0.00100 at the bottom of that
table, +0.00098 in the middle, +0.00096 at the top. That is what a compressing
leaderboard does, and it means each additional point of OOF buys slightly less
LB than the last. Use **+0.00096** in the current range.

Never spend a submission on a candidate that does not clear
`OOF >= target - 0.00096` offline first. To beat the 0.97086 cluster that means
**OOF >= 0.9699000** — a moving target that recedes as we approach it, so treat
any estimate built on an older offset as optimistic.

**Medals.** Playground competitions award no competition medals or ranking
points. Only notebook upvotes count toward tier progression, at 5 / 20 / 50 for
bronze / silver / gold, and only votes from Expert tier or above count.
Notebooks stop accruing after 90 days.

**Leaderboard shape.** Extremely compressed. Top 0.97115, large cluster at
0.97086, public 55-model benchmark 0.97069. Differences of 0.0002 OOF are worth
many places.

**Tree family is saturated.** Every GBDT sits at 0.99+ rank correlation with
every other, regardless of library. Rank correlation of fold-safe TE XGBoost
with: xgb_hpo_d7 0.9961, lgb_pair_lattice 0.9950, cat_dual_view 0.9903, but
lookup_v2_s42 only 0.9756. A 50/50 blend with the lookup transformer reaches
0.9691730 versus 0.9684932 with the existing XGBoost. Spend GPU time on the
lookup transformer seed bag, not on another boosted variant.

**Target encoding depends on rows per key, not dogma.** See E043.

**Decorrelation only pays inside a narrow accuracy band.** `foldsafe_te_wide` was
weaker than its neighbours by 0.0003 and took the largest weight in the stack.
`cpu_linear` was the most decorrelated stream ever built here (0.930 against the
lookup transformer, versus 0.99 among trees) but sat 0.0106 below the best
stream, and the stack gave it a below-median weight. A model has to be close
enough in accuracy for its disagreements to carry information rather than noise.
Do not add a much weaker family just because it is different.

**But family matters more than accuracy inside that band.** Measured on the same
paired test, on the same night, with 50 streams loaded:

| stream | standalone | gap to best | paired t | verdict |
| --- | --- | --- | --- | --- |
| cpu_lgb10fold | 0.9682515 | -0.0002 | +2.8 | rejected |
| cpu_extratrees | 0.9617301 | -0.0067 | +5.8 | kept |
| cpu_linear | 0.9578187 | -0.0106 | -2.3 | rejected |

A LightGBM within 0.0002 of the best stream contributes nothing, while bagged
randomised trees sitting 0.0067 below it contribute more than twice as much
signal. So the band has a floor somewhere between 0.958 and 0.962, and inside
that band the ordering is by how different the family is, not by how accurate it
is. The practical consequence for CPU time: stop building boosted variants, and
spend the slots on learners whose errors have a different shape.

**How to decide whether a stream belongs.** Do not compare two independently-run
stack numbers. The cross-fitted estimate has a noise floor of about 0.0000017
(std over eight meta-fold seeds), so anything under roughly 0.0000033 is not
evidence, and a single before/after refit cannot resolve the gains being chased
here. Use `ensemble_original/validation_audit.py`, which runs the stack with and
without a candidate on identical folds per seed. The paired design cancels fold
noise and gives a t-statistic. Measured examples: `foldsafe_te_wide` t=+88.7,
`lookup_v2_s20260902` t=+11.9, `foldsafe_te_cat` t=+0.6, `cpu_linear` t=-2.3.
Keep a stream only if the paired gain is positive with |t| > 3.

**The stack is not overfit to OOF or LB.** With weights fitted on one random half
of the rows and scored on the other, the stack reaches 0.9695542 against 0.9682409
for the best single stream on those same untouched rows, a gain of +0.0013132 with
no selection touching the scored rows. Selection throughout has been on OOF, with
the leaderboard used only to verify calibration.

**Environment.** Local `.venv` is pandas 3.0.5, Kaggle images are pandas 2.x.
`Series.round()` on an object column is a silent no-op in pandas 3 and raises
`TypeError` in pandas 2. Smoke-test every kernel with `.venv_kaggle/bin/python`
before `kaggle kernels push`.

**Kaggle CPU budget for networks.** Ten outer folds times two seeds overruns the
12-hour limit for anything wider or deeper than the 512-256-128 MLP. Measured:
`resnet_te` (d=256, 4 blocks) finished 10 folds in 7.7h, but `gated_te` (d=512)
managed 8 of 10 in 9.6h and `resnet_te_deep` (8 blocks, 24 epochs) only 4 of 10
in 8.6h. Both wrote `partial_oof_*`, which the registry silently skips, so the
failure looks exactly like a stream that was never built. Before pushing a
network at 10 folds, check it against the 7.7h reference; when in doubt use five
folds, because a complete five-fold stream enters the stack and a partial
ten-fold one does not.

**Standalone accuracy does not predict stack contribution.** Measured on the
78-stream matrix: `cpu_ftt_te` is the best network in the inventory at 0.9671574
but pays only +0.0000020 (t=+4.3), while `cpu_resnet_te` at 0.9663595 pays
+0.0000068 (t=+9.9) and `lookup_v2_10fold_s20260912` pays +0.0000093 (t=+10.2).
Rank candidates by paired t, never by standalone AUC.

**Job chaining.** Never wait on a `pgrep -f <name>` pattern: the waiting shell's
own command line contains the pattern, so it matches itself and never exits.
The same applies to `pkill -f`, which kills the shell running it. Wait on a
captured PID instead, and verify the real process exists after launching.

---

## Ledger

| id | date | experiment | metric | verdict | note |
| --- | --- | --- | --- | --- | --- |
| S004 | 2026-08-06 | 40-stream cross-fitted logistic stack | OOF 0.9696223, LB 0.97060 | ADOPTED | replaced hand-tuned 10-stream rank average |
| S005 | 2026-08-06 | bagged logistic weights + fold-safe TE XGB | OOF 0.9696482, LB 0.97062 | ADOPTED | bagging worth about +0.00002 |
| S006 | 2026-08-06 | 45 streams (+10-fold TE XGB, multi-smoothing, CatBoost) | OOF 0.9696691, LB 0.97064 | ADOPTED | |
| S007 | 2026-08-06 | 47 streams (+wide-key TE XGB, 7th lookup seed) | OOF 0.9697477, LB 0.97073 | ADOPTED | first own result above the 0.97069 public benchmark |
| E042 | 2026-08-06 | fold-safe OOF TE XGBoost, rich composition view, 5 fold | OOF 0.9682418 | ADOPTED | beat every existing XGBoost stream, 20 min on Kaggle GPU |
| E042b | 2026-08-06 | same recipe at 10 outer folds | OOF 0.9684256 | ADOPTED | +0.00018 over 5-fold; best single own model. More rows per model is a real lever |
| E043 | 2026-08-06 | self-including vs out-of-fold TE, single columns | 0.96768759 vs 0.96755293 | REJECTED (the fix) | self-including WINS on every fold. See mechanism below |
| E043b | 2026-08-06 | self-including TE applied to 5 pair keys | OOF 0.87063 | REJECTED | catastrophic, -0.097 |
| E043c | 2026-08-06 | self-including TE applied to all 66 pair keys | OOF 0.83549 | REJECTED | catastrophic, -0.132 |
| E043d | 2026-08-06 | LightGBM capacity 63 leaves on published view | OOF 0.96745965 | REJECTED | -0.00023 versus 31 leaves |
| E044 | 2026-08-06 | fold-safe TE CatBoost | OOF 0.9677370 | MARGINAL | standalone below existing cat_dual_view; added +0.0000059 to the stack. 26 GPU minutes for nothing |
| E045 | 2026-08-06 | fold-safe TE XGBoost, multi-smoothing 10/40/150 | OOF 0.9681489 | MARGINAL | below single-smoothing 0.9682418 |
| E046 | 2026-08-06 | fold-safe TE XGBoost, all 66 pair keys | OOF 0.9681547 | KEPT FOR BLEND | weaker standalone than the 7-pair version but contributed real blend value in S007 |
| E047 | 2026-08-06 | non-negative least squares stack weights | OOF 0.9689379 | REJECTED | far worse than unconstrained logistic. Negative weights are corrections, not noise |
| E048 | 2026-08-06 | hill-climb rank blend over 40 streams | OOF 0.9694472 | REJECTED | below logistic 0.9696223 |
| E049 | 2026-08-06 | additional lookup transformer seeds (20260901-04) | seed 20260901 OOF 0.9683441 | ADOPTED | 39 min per seed on local 4060, the only decorrelated family |
| E032 | 2026-08-04 | nested FT-Transformer (target-free) | OOF 0.9401087 | REJECTED | |
| E035 | 2026-08-04 | nested pair-evidence CatBoost | OOF 0.9664626 | MARGINAL | |
| E037 | 2026-08-04 | TabR retrieval (target-free) | OOF 0.9379940 | REJECTED | |
| E039 | 2026-08-04 | DCNv2 cross network (target-free) | OOF 0.9384981 | REJECTED | |
| E039b | 2026-08-04 | ExtraTrees on hierarchical support | OOF 0.9432187 | REJECTED | 4.5 CPU hours |
| E040 | 2026-08-04 | GANDALF GFLU (target-free) | OOF 0.9387141 | REJECTED | |
| E050 | 2026-08-07 | three-family stacking notebook (xgb+lgb+cat, all trees) | stack 0.9684600, best single 0.9683983, gain +0.0000617 | REJECTED | members 0.99 correlated so stacking has nothing to combine; empirical proof of tree saturation |
| E051 | 2026-08-07 | lookup transformer seed 20260902 | OOF 0.9684095 | ADOPTED | strongest lookup seed so far |
| E052 | 2026-08-07 | CPU additive logistic model on fold-safe TE features | OOF 0.9578187, stack weight rank 39/49 | REJECTED | most decorrelated stream in library (0.930 vs lookup) but 0.0106 below best; decorrelation only pays inside a narrow accuracy band |
| E053 | 2026-08-07 | 49-stream stack (+cpu_linear, +lookup seed 20260902) | OOF 0.9697605, implied LB 0.97075 | MARGINAL | +0.0000128 over 47 streams, mostly from the lookup seed; not submitted, inside noise |
| V001 | 2026-08-07 | validation audit: noise floor, paired with/without, split-half | noise std 0.0000017; split-half stack 0.9695542 vs best single 0.9682409 | ADOPTED | stacking gain +0.0013 holds on rows with zero selection; CV is sound, not LB-overfit |
| V002 | 2026-08-07 | paired t-test on stream inclusion | wide t=+88.7, lookup_s902 t=+11.9, foldsafe_cat t=+0.6, cpu_linear t=-2.3 | ADOPTED | drop cpu_linear and foldsafe_te_cat; neither survives a paired test |
| E054 | 2026-08-07 | CPU 10-fold LightGBM on fold-safe TE | OOF 0.9682515 | REJECTED | paired t=+2.8, below the t>3 bar. A GBDT within 0.0002 of the best stream still adds nothing; tree saturation confirmed a third time |
| E055 | 2026-08-07 | CPU ExtraTrees on fold-safe TE | OOF 0.9617301 | ADOPTED | paired t=+5.8 despite sitting 0.0067 below the best stream. Bagged randomised trees are decorrelated enough to pay where a near-best GBDT is not |
| E051b | 2026-08-07 | lookup transformer seed 20260903 | OOF 0.9684260 | ADOPTED | ties the best single own model; paired t=+9.2 |
| E056 | 2026-08-07 | higher-capacity meta-learners over the 49-stream matrix | regime-interacted 0.9697594 (t=-4.8), shallow LGBM 0.9696688 (t=-32.3), logistic C=1.0 +0.0000008 (t=+0.8) | REJECTED | incumbent L2 logistic 0.9697707 is not beaten by anything. Per-regime weights overfit despite E019 finding regime signal with 4 models; a GBDT meta-learner is far worse. Stop trying to improve the meta-learner and spend the time on streams |
| E058 | 2026-08-07 | 63-stream stack: all 8 lookup seeds, the 10-fold lookup, both HPO trees, three MLP-on-TE variants, RF and ExtraTrees | OOF 0.9698134 (logistic C=0.1) | ADOPTED | +0.0000352 over the 55-stream incumbent. Best single stream is now lookup_v2_10fold at 0.9685902, ahead of every boosted tree |
| E059 | 2026-08-07 | Nystroem RBF kernel machine added to the 63-stream stack | standalone 0.9515181; stack 0.9698134 -> 0.9698083 | REJECTED | third confirmation of the band floor. A genuinely novel inductive bias does not earn a slot on novelty; it must clear the accuracy band first. cpu_extratrees pays at 0.9617, tabm_lattice fails at 0.9592, this fails at 0.9515 |

| E060 | 2026-08-08 | rejected neural architectures re-run on the fold-safe TE view | resnet 0.9663595, dae 0.9658961, cnn1d 0.9626764 | ADOPTED | the four target-free rejections (FTT 0.9401, DCNv2 0.9385, GANDALF 0.9387, TabR 0.9380) were rejections of the *view*, not the architectures. On the view that supplies target statistics all three land in band, and resnet beats the plain MLP |
| E061 | 2026-08-08 | four 10-fold lookup transformer seeds (20260911-14) | 0.9685575, 0.9686765, 0.9685566, 0.9685440 | ADOPTED | seed 20260912 is the best single stream in the library. Ten folds beats five for this family as it did for XGBoost |
| E062 | 2026-08-08 | random search over the MLP-on-TE, 15 trials | best search 0.9650046 vs incumbent 0.9649726; final stream 0.9660810 vs hand-set 0.9661121 | REJECTED (as an improvement) | the hand-set 512-256-128 was already at the family's ceiling. Tuning a family that was never tuned is not automatically free money; kept only as a second configuration of a known-good shape |
| E063 | 2026-08-08 | feature engineering screen: peer deviation, percentile rank, fitted residual, duplicate counts | rank +0.0000094, dup +0.0000000, residual -0.0000268, peer -0.0000418 | REJECTED | the view is finished. Four independent groups, none clearing a coarse +0.0002 bar, two negative. Stop adding columns |
| S008 | 2026-08-08 | 72-stream logistic stack | OOF 0.9698402, LB 0.97080 | ADOPTED | beats the previous best submission 0.97073. Gains came from new families and more folds, not from tuning or features |
| E064 | 2026-08-08 | FT-Transformer on TE view, Kaggle GPU | crashed | ENVIRONMENT | `no kernel image is available for execution on the device`: Kaggle's P100 is sm_60 and the image's torch build has no kernel for it. Not a code fault. Re-run on the local 4060 at batch 1024, since attention over ~120 tokens will not fit 8GB at 8192 |

| E064b | 2026-08-09 | FT-Transformer on the TE view, local RTX 4060 | OOF 0.9671574 | ADOPTED | the strongest network in the inventory, ahead of resnet 0.9663595 and the plain MLP 0.9661121. The ledger recorded this same architecture at 0.9401 target-free, so the view was worth +0.027 to it. The single most valuable consequence of E060 |
| E065 | 2026-08-09 | DCNv2 on the TE view | OOF 0.9483968 | REJECTED | not every rejected architecture is rescued by the view. Bounded-degree crossing lands below every stream the band floor has admitted, so it is excluded without a paired test on the E059 precedent |
| E066 | 2026-08-09 | second wave on the TE view: gated, deeper resnet, 10-fold DAE | gated 0.9646245, resnet_deep 0.9662506, dae_10f 0.9661482 | ADOPTED | all three in band. Ten folds lifted the DAE from 0.9658961 to 0.9661482, the fold lever holding for a third family |
| E067 | 2026-08-09 | four more 10-fold lookup seeds (20260915-18) | 0.9685595, 0.9685803, 0.9685334, 0.9686425 | ADOPTED | the family keeps producing streams at the top of the single-model table |
| S009 | 2026-08-09 | 78-stream logistic stack | OOF 0.9698498 | HELD | +0.0000096 over the 72-stream submission, about four times the noise floor but only ~+0.00001 of implied LB. Not submitted on its own; held for the FT-Transformer fold bag |

| E068 | 2026-08-09 | 10-fold FT-Transformer on the TE view | OOF 0.9673175 | ADOPTED | +0.00016 over the 5-fold run, the fold lever holding for a fourth family |
| E069 | 2026-08-09 | wider FT-Transformer (d_token 96, 4 layers, batch 768) | CUDA OOM | ABANDONED | tried to allocate 4.13GiB with 3.30GiB free on the 8GB card. Not retried at a smaller batch: the paired test puts this family at t=+4.3 against resnet's t=+9.9, so GPU hours belong to the lookup family instead. Recorded so the shape is not re-attempted without a bigger card |
| E070 | 2026-08-09 | third autoencoder, wide bottleneck and heavy swap noise | OOF 0.9658104 | PENDING PAIRED TEST | carried on sufferance; the family's two existing members failed at t=+1.9 and t=+1.1 |

**Pattern across E032/E037/E039/E040:** every target-free neural architecture
lands near 0.938 to 0.940, roughly 0.030 below the models that see target
information. This dataset requires target statistics. The lookup transformer
succeeds because its exact-value embeddings encode them implicitly. Do not spend
more GPU time on target-free architectures however novel they are.

---

## E043 mechanism, worth keeping

Whether a target encoding needs to be built out of fold depends on how many
training rows share a key.

| key | median rows per key | share of keys with one row |
| --- | --- | --- |
| daily_screen_time_hours | 236 | 3.1% |
| age | 37303 | 0.0% |
| stress_level | 207674 | 0.0% |
| notifications_per_day + app_opens_per_day | 10 | 5.2% |
| daily_screen_time_hours + social_media_hours | 2 | 47.8% |
| daily_screen_time_hours + weekend_screen_time | 1 | 55.0% |

With hundreds of rows per key the leak is negligible, and out-of-fold encoding
actively hurts because training features are built from 80% of rows while test
features come from all of them. With one row per key the encoded value is the
row's own label, the model learns to read the answer off the feature, and at
prediction time the unseen key falls back to the prior.

Rule: count rows per key first. Low cardinality keys are better encoded plainly.
High cardinality keys must be out of fold or excluded.

A 30k-row version of this test gave the opposite answer (+0.0039 for the fix)
because at that scale even single columns behave like sparse keys. Small-scale
smoke tests are execution checks, not evidence about full-scale ranking.
