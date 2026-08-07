# 07 — Hardened Freeroll Portfolio

## Outcome: FAILED — `SubmissionStatus.ERROR`, no score

Submitted 2026-08-06 21:16 UTC as submission `55307199`, the last one available. It came back
`ERROR` with no public or private score, and the competition closed at 2026-08-06 23:59 UTC
before the failure was acted on. The standing result is therefore the 2026-08-04 submission:
public 0.822, private 0.780.

The archive passed the organizer's own `validate_submission.py` (YAML, includes, allowed
models, ADK dry-run compile) and its skill scripts had been executed successfully in the real
Kaggle container. The validator compiles the config but does not run a session, so the failure
is somewhere it cannot see — most likely at agent-runtime rather than in the modelling scripts,
which had already run clean in-container. Kaggle exposes no error detail through the API, so
the cause is **unknown**, not diagnosed.

### What the private leaderboard revealed

| Submission | Public | Private |
|---|---:|---:|
| 2026-08-04 | 0.822 | **0.780** |
| 2026-07-20 (v6) | 0.818 | **0.780** |
| 2026-07-19 (v5) | 0.818 | **0.780** |
| 2026-07-18 (v4) | 0.818 | **0.780** |
| 2026-07-17 (v3) | 0.818 | **0.780** |

Five materially different configurations, public scores spanning 0.818–0.822, and an
**identical private score every time**. The public leaderboard movement this whole effort was
chasing was noise; the private metric never responded to any of it. That is consistent with the
offline finding that a perfect oracle over all candidates beats the auto-selection rule by only
+0.000045 — the deterministic core was saturated, and so, apparently, was the private split.

The honest read: the measured +0.00072 offline gain would very likely have produced 0.780 as
well. The submission failing cost little in score, but the process failure below is real
regardless.

### Process failure worth recording

The submission was deliberately made 2h43m early so an `ERROR` could be caught and fixed. It
was not caught in time. Kaggle also reported "0 submissions remaining today" on upload, so a
same-day resubmission would likely have been refused anyway — meaning the early-submission
buffer provided far less protection than assumed. For a one-shot submission the right
procedure is to **poll to a terminal status and confirm a score before standing down**, and to
verify beforehand whether a failed submission returns quota.

---

## Design and evidence (as built)

`sha256(submission.zip) = edc67eb7c4d43b8033368b01d248671bc537db616cea3887dbc1704198b3cdaa`

## Where this came from

Three leaderboard kernels were pulled and read end to end:

| Kernel | LB | What it actually is |
|---|---:|---|
| `najiama/lb-0-823-the-freeroll-gemini-pro-strategy` | 0.823 | `beicicc`'s core **verbatim** + persona-chain freeroll + deleted select agent |
| `beicicc/deterministic-portfolio-replication-s018` | 0.822 | the real engine: CatBoost + LightGBM + ExtraTrees + logistic, 5-fold, rank blends |
| `lucifer19/agentforge-observatory` | 0.822 | same anchor, plus a local falsification harness |
| `crystalbaby/0-819` | 0.814–0.819 | different architecture: more skills, LLM writes more, high-card feature engineering |

The decisive observation is that the 0.823 and 0.822 configs share an **identical**
deterministic core. The entire +0.001 is the no-select rule plus an LLM lottery — so no
amount of model work separates them, and local benchmarking cannot either.

## Why the feature-engineering-heavy configs score lower

All 16 organizer tasks were profiled first, because the hidden evaluation tasks come from
the same family:

| Property | All 16 organizer tasks | External set (titanic/adult/credit_g/phoneme/bank) |
|---|---|---|
| Positive rate | 0.493 – 0.511 | 0.117 – 0.700 |
| Max categorical cardinality | **8** | 40 |
| High-cardinality categoricals | **0 / 16 tasks** | 1 / 5 datasets |
| Date-like columns | **0 / 16 tasks** | — |

Cross-fitted target encoding, frequency encoding, date decomposition and class-imbalance
handling have nothing to act on here. That is a measured explanation for why the configs
built around them sit at 0.814–0.819.

It also means the **external datasets are the wrong benchmark for this competition** — tuning
against titanic/adult pushes you toward exactly the machinery that is inert on the real tasks.
The 16 organizer tasks are the right population, and are what everything below is measured on.

## What this config changes, and the evidence

Measured with an offline replica of a full session on all 16 tasks: real skill scripts, real
manifest parsing, selection simulated as the harness behaves with empty slots (top two public
kept, better private wins).

| | proven core | this config |
|---|---:|---:|
| Mean selected private AUC | 0.803996 | **0.804714** |
| Mean delta | — | **+0.000718** |
| Win / tie / loss | — | **7 / 7 / 2** |
| Worst single-task regression | — | **−0.00026** |
| Best single-task gain | — | **+0.00763** (`train_13`, 500 rows) |

1. **Floor protection is structural, not hoped for.** The four proven models drive
   `rank_top2` and `rank_all`; the added learners can only append a candidate. A check
   (`bench/check_floor.py`) confirms all 112 proven candidate scores are reproduced to
   within 8e-7 — float summation noise in ExtraTrees' threaded averaging.
2. **Added candidates**: XGBoost, HistGradientBoosting, and a seed-bagged CatBoost that uses
   10 folds × 3 seeds on tables under 3,000 rows. That last one is the whole `train_13` gain.
3. **Margin-gated greedy rank blend** (Caruana forward selection), gated by a 5e-4 promotion
   margin and a 1,500-row floor.
4. **`engine="pyarrow"` reads.** Reproduced locally: plain `pd.read_csv` **segfaults**
   (exit 139) on `train_14`. In-container that silently kills the session.
5. **dtype-safe categorical detection** — "not numeric and not datetime" rather than
   "is object", so pandas 3's `StringDtype` cannot empty the categorical list.
6. **No select agent.** Per the organizer's tool docs: *"If never called, the harness
   defaults to the best public score."*
7. **Dataset-informed freeroll prompts** — the Gemini-Pro personas are told the family has no
   high-cardinality or date columns, so they stop wasting turns on inert ideas.

## Runtime, measured in the deployment environment

`kaggle_notebook_parity/` ran these exact scripts on the Kaggle CPU image:

| Task | quick | portfolio | total |
|---|---:|---:|---:|
| `train_12` (49k rows) | 13.4 s | 197.7 s | **211.1 s** |
| `train_11` (29k rows) | 6.2 s | 189.3 s | 195.4 s |
| `train_14` (16 cat cols) | 13.2 s | 143.2 s | 156.3 s |

Worst case is 211 s of a 3,600 s session. The container reports 4 CPUs and **pandas 2.3.3** —
the same major version the benchmark ran under, so the AUC deltas above are measurements in
the deployment environment, not extrapolations.

## Challengers that were rejected

Kept here because the rejections were the most informative part of the work.

| Challenger | Result | Decision |
|---|---|---|
| Extra learners feeding the fixed-weight blends | mean up, but `train_09`/`train_15` regressed — it silently changed proven candidates | ❌ floor broken, reverted |
| Bagged Caruana selection (greedy over random model subsets) | mean **fell** to 0.804299 | ❌ dilution cost more than the overfitting it removed |
| Heavy variant: 3 seeds × 10 folds everywhere + seed-averaged LightGBM | 0.804859, only **+0.000146** over shipped (t=1.30) at 3× runtime and 3× worst-case regression | ❌ held — unverified 3× runtime against a 60-min hard limit for a delta that cannot move a 3-decimal LB |

## Honest limits

- 16 tasks at one seed cannot resolve deltas below roughly 0.0005; the shipped delta
  (+0.00072, t=2.14 at the v9 stage) sits right at that edge.
- The freeroll stage cannot be evaluated offline at all. It is included because it cannot
  lower the result — the deterministic slate is already submitted and scored — not because
  it was measured.
- The public leaderboard is one session on one hidden task, so a 0.001 difference there is
  substantially luck of the draw. The case for this config is expected value per task plus
  the removal of a reproducible crash, not a predicted leaderboard number.

## Reproduce

```bash
.venv/bin/python bench/run_bench.py --package bench/v10_pkg --out out.json --workers 5
.venv/bin/python bench/compare.py    --base research_outputs/bench/base822_full.json --challenger out.json
.venv/bin/python bench/check_floor.py --base research_outputs/bench/base822_full.json --challenger out.json
```
