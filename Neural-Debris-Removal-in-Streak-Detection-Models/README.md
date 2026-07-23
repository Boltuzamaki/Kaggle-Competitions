# Neural Debris Removal in Streak Detection Models (ESA "Secure Your AI")

Machine-unlearning / AI-security competition by Sybilla Technologies, KP Labs and
ESA (ESOC): [Kaggle competition page](https://www.kaggle.com/competitions/neural-debris-removal-in-streak-detection-models).

**Final result: 208.1281 maCADD - rank 61 / 614 teams** (public leaderboard,
lower is better). For reference, the organizers' baselines were: raw poisoned
model **378.02**, empty submission **284.20**, official fine-tuning baseline
**259.92**; the winning team scored ~136-145.

## The task

The organizers trained two RetinaNet (ResNet-50-FPN, single "object" class,
custom anchors) streak detectors on synthetic night-sky imagery:

- a **clean** model - never shared; it is the hidden scoring reference;
- a **poisoned** model (`poisoned_model.pth`) - shared, corrupted with poisoned
  training examples so it detects non-genuine streaks.

Given only the poisoned weights, a 20-image **unlearn set** (with COCO boxes on
the poisoned objects), and 2,000 unlabeled **test images**, the task is to
*de-poison* the model so its test detections match the hidden clean model as
closely as possible. Scoring uses **maCADD** (mean asymmetric Confidence-Aware
Detection Distance): Hungarian-match your boxes to the hidden clean model's
boxes across IoU thresholds 0.2-0.9 and penalize confidence differences -
asymmetrically, so moving confidence in the "correct unlearning direction" costs
10× less. Every unmatched box on either side is penalized by its full
confidence. This makes the metric brutally sensitive to extra boxes and to
confidence calibration, not just to detection accuracy.

## What actually worked (final pipeline)

Direct model repair never produced a submittable detector - every fine-tune /
prune / gradient-ascent variant that suppressed the poison also crushed the
model's confidence on genuine streaks (documented in
`reports/FINAL_FORENSIC_REPORT.md`). The family that worked on the leaderboard
instead treats repaired models as **counterfactual probes** over the *original*
model's detections:

1. **Freeze the candidate bank.** Keep the poisoned model's own boxes (no new
   boxes, no moved boxes - adding boxes is catastrophic under maCADD; V14
   proved it at 477.65).
2. **Train deliberately damaged/unlearned auxiliary models** (activation-pruned
   + EWC empty-label fine-tunes, contrastive unlearners, PCGrad students
   trained against public/external clean-streak data).
3. **Measure confidence collapse.** A candidate box whose confidence collapses
   under the counterfactual models is probably poison; one that survives is
   probably genuine.
4. **Selectively reduce confidence** on only the strongest suspects, in tiers
   (keep / mid 0.21 / epsilon 0.02), never increasing anything.

Score progression of the submitted chain:

| Score | Submission | What changed |
|---:|---|---|
| 300.81 | V1 smoke test | Aggressive head repair; pipeline validation only |
| 229.23 | NDR229 exact | Bug-faithful reproduction of the best public recipe (2-layer activation pruning + 20-iter classifier-only EWC fine-tune + confidence-collapse rescoring) |
| 229.11 | V9 N4 | Contrastive unlearner with synthetic clean-streak retention (survivor-ratio ranking) |
| 224.12 | V10_B | 3-seed replica ensemble of the collapse signal + pixel "dashedness" morphology (gated on public data, AUC 0.764) |
| 216.54 | V12/M1 | Tiered-recovery public artifact: KD-retained student, P3 prototype score, high/mid/epsilon confidence tiers |
| 213.71 | V15_B | Exact V12 bank + hard veto on 71 boxes flagged by a PCGrad counterfactual trained with public external streak data |
| **208.13** | **Step 10C strict** | + 168 extra mid-tier suppressions chosen by a stress-robust 3-of-3 consensus (renderer-fingerprint ensemble × PCGrad × V12 survivor), all gates frozen on public data only |

Negative leaderboard evidence that shaped the final design:

- **E33 (398.05):** the best local pseudo-clean proxy candidate transferred
  terribly - local proxies must never select submissions.
- **V14_C (477.65):** expanding the candidate bank (8,298 boxes) is fatal.
- **V18_A (280.34):** broadly *restoring* low-confidence boxes is also fatal,
  even with 95-98% public cross-domain precision.

## Experiment map

Roughly 40 forensic checks + 30 repair/ranking experiment families were run.
`REPORT.md` is the complete record; the short version:

| Family | Folder | Outcome |
|---|---|---|
| P1-P3 forensics (image, causal, internals) | `kernels/forensics/` | Diagnosis: dynamic *semantic* object-generation poisoning - no fixed patch/border/frequency/location trigger; response concentrated in P3/P4 classification outputs |
| V1-V3 head repair, adversarial unlearning, channel ablation | `kernels/forensics/` | All rejected: hard suppression<->retention trade-off; no isolable "backdoor neuron" |
| E00-E54 maximal experiment matrix | `kernels/forensics/v4_experiment_matrix`, audits in `reports/` | 300+ candidates; zero passed joint gates; exposed proxy-selection failure (E33) |
| NDR229 exact reproduction | `kernels/experiments/ndr229_exact_gpu` | 229.23; first score-bearing family (preserves the public notebook's two-layer pruning index quirk) |
| V9 contrastive unlearner × amplifier | `kernels/experiments/ndr_contrastive_v9` | 229.11; survivor ratio validated, amplifier unvalidated |
| V10 replica ensemble + dashedness | `kernels/experiments/ndr_v10_ensemble` | 224.12; replica diversity real (pairwise corr ~ 0.5); dashedness gated in at AUC 0.764 |
| V11/V12 trajectory + tiered recovery | `kernels/experiments/ndr_v11_v12_trajectory_anchor` | V12/M1 exact artifact 216.54 |
| V13 task-vector / ranker / projection | `kernels/experiments/ndr_v13_breakthrough` | Not submitted; its broad box bank later shown unsafe |
| V14 external-data PCGrad | `kernels/experiments/ndr_v14_external_retain` | 477.65 as a full pipeline - but its PCGrad signal survived as a veto |
| V15 anchor + PCGrad veto | `kernels/experiments/ndr_v15_anchor_veto` | 213.71; 71-box hard veto on the exact V12 bank |
| V16-V17, V20, V22, V23 gated signals | `kernels/experiments/` | All disabled by predeclared bidirectional cross-domain transfer gates (one-way AUC is not enough) |
| V18 broad recovery | `kernels/experiments/ndr_v18_canonical_recovery` | 280.34; broad restoration abandoned |
| V19, V21 stability/consensus audits | `kernels/experiments/` | Confirmed V15_B; exposed that "independent" signals correlate 0.957 |
| Sequential validation steps 1-17 | `kernels/validation/` | Frozen-gate ledger (`reports/VALIDATION_EXECUTION_STATUS.md`); Step 10C strict -> **208.13** |
| Surrogate competition harness | `kernels/surrogate/` | Stage-1 trigger screens; blocked at frozen gates, honestly recorded |

## Key lessons

1. **Repair-as-probe beats repair-as-product.** Under a distance-to-hidden-model
   metric, deploying a repaired detector is riskier than using repairs to
   re-score the original detector's own boxes.
2. **Never add boxes; barely touch confidences.** Unmatched-box penalties and
   asymmetric confidence costs dominate maCADD.
3. **Local proxies lie.** Pseudo-clean maCADD on the 20 public images selected a
   398-scoring candidate. Public cross-domain AUC of 0.97+ still failed on the
   hidden model. Every signal was therefore gated by *bidirectional*
   cross-domain transfer tests frozen before test inference, and gate failures
   meant the candidate reduced to the incumbent - several experiments
   (V16, V17, V20, V22, V23) shipped as deliberate no-ops because of this.
4. **The poison was semantic.** Forensics rejected every fixed-trigger
   hypothesis; the model genuinely learned a "non-genuine streak" concept that
   overlaps the legitimate class, which is why naive unlearning destroys recall.
5. **Reproduce the anchor bug-for-bug.** The best public notebook's pruning had
   an index quirk (only 2 of 4 claimed layers pruned); exact score reproduction
   required preserving it.

## Repository layout

```
├── README.md                  this file
├── REPORT.md                  complete experiment & blocker report (the full story)
├── requirements.txt           local Python environment (CPU; Kaggle kernels install their own deps)
├── eda/                       EDA notebook + written summary of data/model forensics
├── reports/                   forensic reports, per-experiment audits, frozen-gate ledgers,
│                              research queues and the public-notebook research audit
├── kernels/
│   ├── forensics/             P1-P5 forensic program + V1-V3/E-series repair kernels
│   ├── experiments/           NDR229 … V23 (the score-bearing chain and gated experiments)
│   ├── validation/            sequential frozen-gate validation steps (3 -> 10D); Step 10C is the final winner
│   └── surrogate/             surrogate-competition harness (stage 1a/1b/1c)
└── tools/                     local post-processing retuner and audit/alignment helpers
```

Each kernel folder contains the notebook source (`.py` percent-format and/or
`.ipynb`), its `kernel-metadata.json` (exact accelerator + pinned Docker image
digest), and flattened `audit__*.json` files - the run's frozen selection lock
and final report as produced on Kaggle. Model weights, test predictions and
bulk outputs are intentionally excluded.

## Reproducing

Everything ran as private Kaggle notebooks under the account `boltuzamaki`,
on **Tesla T4** with the pinned Kaggle Docker image recorded in each
`kernel-metadata.json` (T4/SM 7.5 matters: Detectron2 is compiled in-kernel and
the scored runs are not reproducible on P100 with the same stack).

1. Join the competition and set up the Kaggle CLI (`pip install kaggle`).
2. Pick a kernel folder, e.g. the first score-bearing model:

   ```bash
   kaggle kernels push -p kernels/experiments/ndr229_exact_gpu
   ```

   Each notebook is self-contained: it installs Detectron2, reads only the
   competition inputs, trains/infers, writes `submission*.csv` plus audit
   JSONs, and **never** calls the submission API.
3. Reproduce the final submission chain in order: `ndr229_exact_gpu` ->
   `ndr_contrastive_v9` -> `ndr_v10_ensemble` -> `ndr_v11_v12_trajectory_anchor`
   (V12 anchor) -> `ndr_v14_external_retain` (PCGrad checkpoints) ->
   `ndr_v15_anchor_veto` -> `validation/step10c_stress_consensus_inference`.
   Later kernels consume earlier kernels' exported artifacts (uploaded as
   Kaggle datasets; each kernel's metadata lists its sources).
4. Post-processing retunes need no GPU: `tools/local_retune.py` regenerates a
   full submission from a per-box diagnostics `.npz` exported by the V10+ runs.
5. `eda/EDA.ipynb` runs locally on CPU against the competition download
   (`requirements.txt` environment).

Deterministic-run details: every experiment wrote a `selection_lock.json`
*before* enumerating test images, logged JSONL heartbeats, hashed its inputs
and outputs (SHA-256), and validated the 2,000-row submission schema. Exact
control variants reproduce the incumbent CSV hash byte-for-byte before any new
variant is trusted.

## Rules compliance (competition Rule 7.A)

No test image was ever labeled - manually, automatically, or via pseudo/soft
labels; no external model annotated test data; test pixels were used only for
normal inference and predeclared deterministic post-processing (explicitly
host-permitted); synthetic streak generators were designed from public/unlearn
data and physics only; all selection rules were frozen before test enumeration;
leaderboard scores were used only for coarse method-family evaluation.

## Credits

Public work this solution built on (used as recipes/signals, cited not copied):

- `sanidhyavijay24/ndr-trial1` - pruning + EWC + confidence-collapse rescoring
  (the 229 anchor recipe)
- `biohack44/ndr-trial-v2` - contrastive ranking and the tiered V12/M1 artifact
- `zaouiyassine/de-poisining-through-prunning-finetuning` - pruning + EWC
- `amerhu/debris-removal-calibrated-rescoring-v3-0` - rescoring / geometry ideas
- `jasonkimmmmmmmm/roadmap-that-got-me-to-226-poisoned-dataset` - thresholding
  and morphology ("dashedness") guidance
- `sanidhyavijay24/streaksyolodataset` (Zenodo `10.5281/zenodo.14047944`) -
  public external clean-streak data for PCGrad retain objectives
- Organizers' citation: Kaczmarek, Ntagiou, Grzywaczewski, Kotowski, Drzał,
  Shendy - *Neural Debris Removal in Streak Detection Models*, Kaggle, 2026.
