# ROGII Wellbore Geology — Gold-Medal Strategy

_Synthesis of the top public notebooks (Pilkwang target-free geosteering, ravaghi
ridge, mycarta domain toolkit), the discussions, and our own local experiments._

---

## 1. What the task really is
Predict **TVT** (stratigraphic depth of the bit) for the **hidden toe-end tail**
of each horizontal well — every row after the Prediction-Start (PS) point where
`TVT_input` is NaN. Metric = **row-weighted RMSE** of `dTVT = TVT − predicted`.
Because it's row-weighted, **long tails dominate the score** → prioritise the
long wells.

Model the **residual** `ΔTVT = TVT − last_known_TVT`, not raw TVT. The flat
anchor (`ΔTVT = 0`) is a strong baseline; the job is to help *drifting* wells
without disturbing *flat* wells.

## 2. The scoring landscape (public LB)
| Tier | Public RMSE | How |
|---|---:|---|
| Top | ~5.3 | leak-aware ensembles + heavy stacking, tuned to the re-pick |
| Strong public | 7.0–7.3 | same-well contact leak + PF/beam + GBDT + blend |
| Reference (ravaghi) | 7.95 | ridge stack over physics signals + same-well contacts |
| **Our honest PF+beam** | **9.07** | target-free particle filter (no leak) |
| Deotte starters | ~15 | plain XGB/NN on tabular features |

## 3. The decisive insight: leak vs honest, public vs private
**Every one of the 3 public test wells is a byte-identical copy of a train well**
(same MD/X/Y/Z/GR, labels stripped). The hidden LB target is a *different manual
re-pick* of those same wells, so copying the train `TVT` scores ~7.9 (not 0) — a
strong but **overlap-dependent** shortcut.

- `same_well_physical` / `tvt_from_contacts` = **public-aggressive**. Great when
  the scored wells overlap train; **collapses on unseen private wells**.
- Target-free PF/beam/NCC/DTW = **private-safe**. Uses only observed test logs +
  typewell; generalises to unseen wells.

**Gold is decided on the private LB.** The whole bet:
- If private = unseen wells → the overlap-abusers shake out; **the best honest
  target-free model wins**. Trust CV, not the public LB.
- If private = row-split of the same 3 wells → the leak persists and honest tops
  out ~9. Then gold needs the leak (a call you've chosen not to make).

The mycarta toolkit + discussions note the validation wells are **spatially
interleaved with train (interpolation, not extrapolation)** and GroupKFold-by-well
is *more pessimistic* than the real test — i.e. a robust honest model should score
**better** on the true test than in our CV. That favours the honest bet.

> **Our chosen line: build the strongest honest, target-free, CV-trusted model.**
> Accept a mediocre public score; play for the private shake-up.

## 4. What works vs what doesn't (our measured results)
| Signal / method | Honest CV pooled | Verdict |
|---|---:|---|
| Constant anchor | ~15.7 | baseline |
| Momentum **Particle Filter** | **~10.1** | ✅ core signal |
| Beam search | ~15 (alone) | ✅ only as PF blend / feature |
| PF + beam + hold (CV weights) | ~10.4 | ✅ our current submission (9.07 public) |
| Velocity/dip PF (`pf_z`) | ~19 | ❌ over-constrains alone; only as a feature |
| **Formation-plane prior** (honest, imputed) | **~38** | ❌ formation tops don't interpolate spatially |
| Sequence NN (Conv-BiGRU / TCN / Transformer) | 10.3–11.5 | ⚠️ ok, not better than PF |
| Tabular LGBM on plain features | ~15 | ❌ needs the physics signals as features |

**Key negative result:** the honest formation-plane prior is useless — the
reference's strong "contact" signal was the **leak** (each well's own formation
column), not honest spatial imputation. Don't chase it.

## 5. The honest gold pipeline (what we're building)
A **single LightGBM** (end-to-end, no blend of independent solutions) predicting
`ΔTVT`, with **domain-physics signals as features**:

1. **Particle-filter path** `pf_d` (tuned: momentum 0.998, scale 12) — strongest single signal.
2. **Beam-search path** `beam_d`, and `pf_vs_beam` disagreement.
3. **Multi-scale NCC** vs typewell (scales 8/15/25) + match scores — stratigraphic barcode alignment.
4. **Self-correlation** vs the lateral's own known section (GR repeats as the bit re-crosses beds).
5. **Q-3D tortuosity** (rolling curvature of X/Y/Z path) — mycarta's single biggest signal; captures active steering.
6. **Relative trajectory** `d_md, d_z, d_xy, dz/dmd` from PS (never absolute depth — within-lateral TVT⊥Z).
7. **Typewell GR residuals** at PF±offset baselines (`twres_pf{−20…20}`) — local alignment evidence.
8. **GR texture** (rolling mean/std, first diff), **frac-along-lateral**.

Validation: **StratifiedGroupKFold by well** (strata = drilling azimuth × median
TVT × XY grid bin). Postprocess: **shrink toward anchor, clip slope, light
Savitzky-Golay smoothing** (TVT is a smooth curve; kill jumps).

**Rules that keep it private-safe:**
- GroupKFold by well (no same-well rows across folds).
- Fit any spatial reference leave-one-well-out.
- Never read hidden-tail `TVT`; fit GR calibration on prefix only.
- No `same_well_physical`, no train-twin copy.

## 6. Roadmap to gold (ranked by expected value)
1. **Ship the honest LGBM-on-physics-signals stack** (in progress) — expected honest CV ~8–9, and *better on the real interpolation test*.
2. **Add self-correlation + tortuosity** properly (tortuosity was mycarta's top feature) and re-CV.
3. **Postprocessing sweep** (shrink/clip/savgol) chosen on CV, not LB.
4. **Seed/particle scale-up** on the final PF for the 3 test wells (cheap).
5. **Two-submission hedge (allowed, honest):** keep one *pure target-free* submission (private bet) and note the public-aggressive one separately so you can compare shake-out — but select the honest one as final.
6. If the private set turns out to overlap train (watch the forum), only then reconsider the contact shortcut.

## 7. CV discipline (the thing that actually wins medals here)
- **Trust GroupKFold-by-well CV.** The public LB is only 3 wells — extremely
  noisy; do not tune to it.
- Report **both** row-weighted (metric) and per-well RMSE.
- A change ships only if it improves CV **and** doesn't add movement on flat wells.
- Expect the real test to score *better* than CV (interpolation vs pessimistic blocking).

## 8. Current status
- Honest submission on LB: **9.07** (PF+beam+hold).
- Building: single-LGBM physics-signal stack (signals computing over all 773 wells now).
- Next: CV the stack, postprocess, submit the honest candidate.
