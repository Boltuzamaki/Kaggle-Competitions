# Mid-point next-method plan (written 2026-07-20)

## Where the project stands

| Fact | Value |
|---|---|
| Best submission | **229.1051** (`54839276`, V9 N4 measured-family control, 2026-07-19) |
| Other submissions | V1 smoke 300.8064, E33 398.0498 |
| Leaderboard top | ~145.4 (#1 Jason Kim); top-20 cutoff ~169 |
| Completed | `boltuzamaki/neural-debris-ndr-contrastive-v9`; N4 scored 229.1051 and N1 remains unsubmitted |
| Authored, never pushed | `forensics/kaggle_ndr_frontier_v8` (prune×EWC grid + synthetic recovery, 4 candidates, 3 submission variants) |
| Deadline | 22 July 2026 AoE (~2.5 days left) |
| Account | boltuzamaki (kaggle CLI in `.venv/Scripts/kaggle.exe`) |

Internal repair matrices (V1-V3, E28-E54) all hit the same wall: suppression vs
retention trade-off, no candidate passed the joint gates. The score-bearing family
is the **public NDR recipe**: activation-prune 2 cls_subnet layers -> 20-iter
empty-label classifier fine-tune with EWC -> *confidence-collapse rescoring* of the
original model's candidate boxes (`p_poison = 0.9·s_diff + 0.1·s_geo`, demote >=0.55,
keep <=0.25). Our exact repro reproduced its 229.2314 exactly, so the local pipeline
is trustworthy.

## What the evidence says to try next (ranked)

1. **Replica-ensemble survival signal (this notebook, `ndr_v10_ensemble`).**
   The single de-poisoned replica's `s_diff` is noisy - one 20-iteration
   fine-tune on 20 images decides every candidate's fate. Averaging `s_diff`
   across 3 seed-replicas of the same recipe denoises the poison probability.
   This was Stage C item 8 of `forensics/PUBLIC_KAGGLE_RESEARCH_2026-07-18.md`
   and is the cheapest high-confidence improvement over 229.

2. **Per-box diagnostics export -> local post-processing retuning (same notebook).**
   The 229 run only saved per-image aggregates, so every P_HI/P_LO/threshold
   retune costs a full GPU run. V10 exports per-candidate `(box, score, s_diff
   per replica, s_geo, dashedness, linearity)` as one npz. After ONE GPU run you
   can generate unlimited submission variants locally with
   `tools/local_retune.py` - critical with ~2.5 days left.

3. **Morphological dashedness filter + rescue (same notebook, auto-gated).**
   The #1 player's public roadmap (`references/roadmap_226`) says poisoned
   detections look "dashed/segmented" while real streaks are continuous and
   linear, and that shape-filtering + rescuing linear low-confidence streaks is
   what took them below 226. Host clarifications explicitly allow raw test
   pixels in deterministic post-processing. V10 computes a dashedness score per
   candidate box, validates it on public data (20 poison crops vs synthetic
   clean streaks; used only if AUC >= 0.65), and emits dash-weighted and
   rescue variants.

4. **Post-processing grid around P_HI/P_LO/MIN_KEEP** (Stage C item 10) -
   covered by the exported variants (tight / minkeep30) plus local retuning.

5. **V9 contrastive results** - when the running kernel finishes, compare its
   frozen finalists (`N1_center`, `N4_no_amp`, `N2_ampstrict`) against V10's
   variants; the amplifier ratio is an orthogonal ranking signal that can be
   merged into `p_poison` locally via the npz if V9's separation audit passes.

6. **Optional: push V8 frontier** if GPU quota allows - its synthetic-recovery
   candidate (prune -> retention fine-tune on synthetic streaks) is the only
   family that tries to *raise* genuine-streak confidence, which the asymmetric
   metric rewards.

## Submission budget plan (lower is better, ~2.5 days)

- Day 1 (today): run `ndr_v10_ensemble` on Kaggle T4. Preserve the one
  remaining submission slot until the seed-42 anchor and replica-diversity
  audits finish; only then compare `V10_A` and the gate-passing dash variants.
- Day 2: local retune from npz using what the day-1 scores imply
  (each submission is a probe of the hidden reference density); submit 2-3
  retuned CSVs. Merge V9 amplifier signal if its audit passed.
- Final day: submit the best two; keep 229.2314 CSV as the safety pick for
  final selection.

## Rule 7.A guardrails (unchanged)

- No manual/automatic test labels; no external models on test.
- Test pixels only via the provided poisoned model, its repairs, and
  deterministic post-processing (host-permitted).
- All variant rules frozen in `selection_lock.json` before test enumeration.
- Dashedness constants and the synthetic simulator are designed from
  public/unlearn info only, never from inspecting test streaks.
