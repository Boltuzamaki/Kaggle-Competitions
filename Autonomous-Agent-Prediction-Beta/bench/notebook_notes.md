# Hardened Freeroll Portfolio — Agent Config

This notebook writes an **Agent Config** for *Autonomous Agent Prediction (Beta)*. The
submission is not a prediction file: it is a `agent.yaml` plus prompts and skill scripts that
are dropped into an offline CPU container and must solve an unseen binary-classification
task end to end, inside 60 minutes, 30 submissions and a $2 LLM budget.

## Credit

The deterministic core is the public LB **0.822** portfolio by
[Kun Zhang](https://www.kaggle.com/code/beicicc/deterministic-portfolio-replication-s018),
wrapped in the "Freeroll" persona-chain architecture from
[najiama](https://www.kaggle.com/code/najiama/lb-0-823-the-freeroll-gemini-pro-strategy) (LB 0.823),
which in turn builds on
[Georgy Mamarin's proof](https://www.kaggle.com/code/georgymamarin/your-agent-s-selection-policy-is-worth-0-0005-auc)
that deleting the selection agent is worth ~0.0005 AUC. Everything below is a change on top
of those, and each one is attached to a measurement.

## What this adds, and the evidence for it

The 16 organizer-supplied training tasks were profiled first, because the hidden evaluation
tasks come from the same family. The profile decides which improvements are even applicable:

| Property | Value across all 16 tasks |
|---|---|
| Rows | 500 – 49,432 |
| Features | 8 – 30 |
| Positive rate | 0.493 – 0.511 (balanced everywhere) |
| Max categorical cardinality | **8** |
| High-cardinality categoricals | **0 tasks** |
| Date-like columns | **0 tasks** |

That table is a negative result with teeth. Target encoding, frequency encoding and date
decomposition — the headline features of several competing configs — are **inert on this task
family**, because there is no high-cardinality column for them to act on and no date to
decompose. Class-imbalance machinery is equally inert at a 50% base rate. Effort spent there
buys nothing, which is consistent with those configs scoring 0.814–0.819 rather than 0.822.

So the changes here are aimed at what the profile *does* contain — small-to-medium tables of
mostly numeric plus low-cardinality categorical features:

1. **Two more model families** (XGBoost, HistGradientBoosting) and a **seed-bagged CatBoost**
   for small tables, widening the blend without touching any existing member.
2. **Greedy (Caruana) rank blending** over the out-of-fold matrix, gated by a 5e-4 promotion
   margin and a 1,500-row floor — because a blend that picks its own weights on the OOF it is
   scored against wins the OOF comparison on small tables by fitting fold noise.
3. **A pruned candidate slate.** Every extra near-duplicate candidate is another draw whose
   maximum is upward biased on the public split and not on the private one, so weak candidates
   are dropped rather than submitted.
4. **`engine="pyarrow"` on every read.** Reproduced locally: plain `pd.read_csv` **segfaults**
   (exit 139) on `train_14`, which silently kills the container and forfeits the session. This
   is worth far more in expectation than any AUC delta below.
5. **dtype-safe categorical detection.** Stated as "not numeric and not datetime" rather than
   "is object", so it survives the container's pandas version — under pandas 3 string columns
   arrive as `StringDtype` and an `is_object_dtype` test silently finds zero categoricals and
   feeds raw strings to the numeric path.
6. **No selection agent.** The organizer's own tool documentation states: *"If never called,
   the harness defaults to the best public score."* Leaving the slots empty is therefore the
   documented behaviour, not a trick, and it returns the final minutes to the freeroll stage.

## Measured effect

Each candidate config was run through an offline replica of a full session on all 16 organizer
tasks — real skill scripts, real manifest parsing, and selection simulated the way the harness
behaves when the slots are left empty (top two public kept, better private wins).

Paired against the proven core on the same 16 tasks:

| | proven core | this config |
|---|---:|---:|
| Mean selected private AUC | 0.803996 | **0.804714** |
| Mean delta | — | **+0.000718** |
| Win / tie / loss | — | **7 / 7 / 2** |
| Worst single-task regression | — | **−0.00026** |
| Best single-task gain | — | **+0.00763** (`train_13`, 500 rows) |

The shape of that distribution is the point: seven tasks are reproduced to the last decimal,
the two losses are ~2.6e-4, and the gains run an order of magnitude larger. That is what
"additive" is supposed to look like.

Two challengers were **rejected** by the same harness rather than shipped:

- Letting the extra learners into the fixed-weight blends raised the mean but regressed
  `train_09` and `train_15`, because it silently changed candidates the proven config
  produces. Floor broken; reverted.
- Bagged Caruana selection (averaging greedy weights over random model subsets) *lowered*
  the mean to 0.804299. The dilution cost more than the overfitting it removed. Rejected.
- A heavier variant (three seeds × ten folds on every table plus a seed-averaged LightGBM)
  reached 0.804859, only **+0.000146** over the shipped config — statistically
  indistinguishable at t=1.30 — while tripling runtime and tripling the worst-case
  regression to −0.0009. Held, not promoted: an unverified 3× runtime increase against a
  60-minute hard limit is not worth a delta that cannot move a three-decimal leaderboard.

### Runtime, measured in the deployment environment

A separate notebook ran these exact scripts on the Kaggle CPU image. This matters more than the
local timings, because the container is the thing with the 60-minute limit:

| Task | quick | portfolio | total |
|---|---:|---:|---:|
| `train_12` (49k rows) | 13.4 s | 197.7 s | **211.1 s** |
| `train_11` (29k rows) | 6.2 s | 189.3 s | 195.4 s |
| `train_14` (16 cat cols) | 13.2 s | 143.2 s | 156.3 s |

Worst case is **211 s of a 3,600 s session** — about 6% of the budget, leaving the rest to the
freeroll. The container reports 4 CPUs and **pandas 2.3.3**, which is the same major version the
benchmark above ran under, so these AUC deltas are measurements in the deployment environment
rather than extrapolations from a different one.

## Architecture

```
agent.yaml  (SequentialAgent)
  1. quick       gemini-3.1-flash-lite  ->  fast CatBoost floor, submitted immediately
  2. portfolio   gemini-3.1-flash-lite  ->  runs the frozen skill, submits its slate
  3. pro_mad     gemini-3.1-pro-preview ->  high-temperature feature invention
  4. pro_strict  gemini-3.1-pro-preview ->  low-temperature repair of stage 3's code
  5. pro_balanced gemini-3.1-pro-preview -> safe optimisation until 4 minutes remain
  (no selection agent -> harness keeps the top two public scores)
```

Stages 3–5 cannot lower the result: the deterministic slate is already submitted and scored, so
the worst case is that the freeroll produces nothing that beats it.
