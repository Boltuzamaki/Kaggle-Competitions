# Autonomous Agent Prediction (Beta)

[Kaggle competition](https://www.kaggle.com/competitions/autonomous-agent-prediction-beta) ·
2026-07-06 → 2026-08-06 · 570 teams · ROC AUC

**Result: public 0.822 / private 0.780 — rank 141 / 570.** The final submission returned
`SubmissionStatus.ERROR` and never scored; the standing result is the 2026-08-04 entry.

The single most useful thing this repository records is not a model. It is this:

| Submission | Public | Private |
|---|---:|---:|
| 2026-08-04 | 0.822 | **0.780** |
| 2026-07-20 (v6) | 0.818 | **0.780** |
| 2026-07-19 (v5) | 0.818 | **0.780** |
| 2026-07-18 (v4) | 0.818 | **0.780** |
| 2026-07-17 (v3) | 0.818 | **0.780** |

Five materially different agent configurations, public scores spanning 0.818–0.822, and an
**identical private score every time**. The public leaderboard this competition was played on
was noise, and the private metric never responded to any configuration change.

---

## The competition

This is Kaggle-in-Kaggle: the submission is not a prediction file, it is an **agent config**.
A `agent.yaml` plus prompts, tool declarations and skill scripts is dropped into a sandboxed
offline CPU container together with an unseen binary-classification task. The agent has to
inspect the schema, train, submit candidate probability files, read back the public-split
score, and choose its finals — alone.

| Constraint | Limit |
|---|---:|
| Wall clock | 60 min per session |
| Prediction submissions | 30 per session |
| LLM spend | $2 per session |
| Execution | CPU, offline, pre-installed packages only |
| Archive | `agent.yaml` must sit at the archive root |

Two sessions are run per submission: one produces the public leaderboard score, the other the
private one. Sixteen labelled training tasks are supplied from the same family as the hidden
evaluations.

## Reading the leaderboard before writing any code

Three high-scoring public kernels were pulled and read end to end:

| Kernel | LB | What it actually is |
|---|---:|---|
| [`najiama/lb-0-823-the-freeroll-gemini-pro-strategy`](https://www.kaggle.com/code/najiama/lb-0-823-the-freeroll-gemini-pro-strategy) | 0.823 | `beicicc`'s core **verbatim** + a Gemini-Pro persona chain + deleted select agent |
| [`beicicc/deterministic-portfolio-replication-s018`](https://www.kaggle.com/code/beicicc/deterministic-portfolio-replication-s018) | 0.822 | the real engine: CatBoost + LightGBM + ExtraTrees + logistic, 5-fold, rank blends |
| [`lucifer19/agentforge-observatory`](https://www.kaggle.com/code/lucifer19/agentforge-observatory) | 0.822 | the same anchor, plus a local falsification harness |
| [`crystalbaby/0-819`](https://www.kaggle.com/code/crystalbaby/0-819) | 0.814–0.819 | a different architecture: more skills, the LLM writes more, high-cardinality feature engineering |

The decisive observation is that the 0.823 and 0.822 configs share a **byte-identical**
deterministic core. The whole +0.001 is the no-select rule plus an LLM lottery — so no amount
of model work separates them, and offline benchmarking never could.

## Profiling the tasks decides which code can possibly work

All 16 organizer tasks were profiled first, because the hidden evaluation tasks are drawn from
the same family:

| Property | Organizer tasks (16) | External set (titanic / adult / credit_g / phoneme / bank) |
|---|---|---|
| Rows | 500 – 49,432 | 600 – 18,000 |
| Positive rate | **0.493 – 0.511** | 0.117 – 0.700 |
| Max categorical cardinality | **8** | 40 |
| High-cardinality categoricals | **0 / 16 tasks** | 1 / 5 datasets |
| Date-like columns | **0 / 16 tasks** | — |

Cross-fitted target encoding, frequency encoding, date decomposition and class-imbalance
machinery have nothing to act on here. That is a measured explanation for why the configs built
around them sit at 0.814–0.819 while the plain portfolio sits at 0.822.

It also means the **usual public tabular datasets are the wrong benchmark for this
competition**: tuning against titanic/adult pushes you toward precisely the machinery that is
inert on the real tasks. The 16 organizer tasks are the right population, and everything below
is measured on them.

## The offline session replica

`bench/run_bench.py` replays a full session against tasks whose labels are known: it copies only
the three files the container exposes, runs the real skill scripts as subprocesses, parses the
real `PORTFOLIO_MANIFEST`, scores every emitted candidate, and simulates selection the way the
harness behaves when the slots are left empty — top two public kept, better private wins.

Two findings from it shaped everything else:

- **Selection has no headroom.** A perfect oracle over all candidates beats the auto-selection
  rule by **+0.000045** across 16 tasks, and public predicts private at r = 0.84. The candidate
  pool, not the selection policy, is the ceiling.
- **The deterministic core is saturated.** The best single candidate leaves a mean regret of
  only 0.00026 against that oracle.

## What the final config changed

The four proven models drive `rank_top2` and `rank_all`; added learners may only *append* a
candidate. `bench/check_floor.py` verifies this structurally — all 112 proven candidate scores
reproduce to within 8e-7, which is float summation noise in ExtraTrees' threaded averaging.

| | proven core | final config |
|---|---:|---:|
| Mean selected private AUC (16 tasks) | 0.803996 | **0.804714** |
| Mean delta | — | **+0.000718** |
| Win / tie / loss | — | **7 / 7 / 2** |
| Worst single-task regression | — | **−0.00026** |
| Best single-task gain | — | **+0.00763** (`train_13`, 500 rows) |

Beyond the added learners (XGBoost, HistGradientBoosting, a 10-fold × 3-seed CatBoost for small
tables) and a margin-gated greedy rank blend, three changes were reliability rather than score:

1. **`engine="pyarrow"` reads.** Reproduced locally: plain `pd.read_csv` **segfaults**
   (exit 139) on `train_14`. In-container that silently kills the session and forfeits it.
2. **dtype-safe categorical detection** — "not numeric and not datetime" rather than
   "is object", so pandas 3's `StringDtype` cannot silently empty the categorical list.
3. **No select agent.** Per the organizer's own tool documentation: *"If never called, the
   harness defaults to the best public score."*

### Runtime, measured in the deployment environment

`notebooks/agent-portfolio-runtime-parity.ipynb` ran these exact scripts on the Kaggle CPU image:

| Task | quick | portfolio | total |
|---|---:|---:|---:|
| `train_12` (49k rows) | 13.4 s | 197.7 s | **211.1 s** |
| `train_11` (29k rows) | 6.2 s | 189.3 s | 195.4 s |
| `train_14` (16 cat cols) | 13.2 s | 143.2 s | 156.3 s |

Worst case is 211 s of a 3,600 s session. The container reports 4 CPUs and **pandas 2.3.3**, the
same major version the offline benchmark ran under — so the deltas above are measurements in the
deployment environment, not extrapolations from a different one.

## Challengers that were rejected

Kept because the rejections were the most informative part of the work.

| Challenger | Result | Decision |
|---|---|---|
| Extra learners feeding the fixed-weight blends | mean up, but `train_09` / `train_15` regressed — it silently changed proven candidates | rejected, floor broken |
| Bagged Caruana selection (greedy over random model subsets) | mean **fell** to 0.804299 | rejected — dilution cost more than the overfitting it removed |
| 3 seeds × 10 folds everywhere + seed-averaged LightGBM | 0.804859, only **+0.000146** over shipped (t = 1.30) at 3× runtime and 3× the worst-case regression | held — an unverified 3× runtime increase against a 60-min hard limit is not worth a delta that cannot move a three-decimal leaderboard |

## Post-mortem

The final submission (`55307199`, uploaded 21:16 UTC, 2h43m before the deadline) returned
`ERROR` with no score, and the deadline passed before it was addressed.

The archive had passed the organizer's own `validate_submission.py` — YAML, `!include`
resolution, allowed-model check and an ADK dry-run compile — and its skill scripts had already
executed cleanly inside the real Kaggle container. The validator compiles the config but never
runs a session, so the failure sits somewhere it cannot see. Kaggle exposes no error detail
through the API, so the cause is **unknown, not diagnosed**.

What went wrong procedurally is clearer. The submission was made early precisely so an `ERROR`
could be caught and fixed, but Kaggle reported *"0 submissions remaining today"* on upload —
so a same-day retry would most likely have been refused anyway, and the buffer was worth far
less than assumed. For a one-shot submission the correct procedure is to **poll to a terminal
status and confirm a score before standing down**, and to establish in advance whether a failed
submission returns quota.

Given every completed configuration scored private 0.780 regardless, the failure cost little in
rank. The process lesson stands on its own.

## Layout

```text
submissions/          agent configs for all seven submissions; 07 is the final one
bench/                offline session replica, comparison tools, notebook builders
notebooks/            the two Kaggle notebooks authored here
results/              raw per-task measurements behind every number above
```

Competition data is **not** committed. Download it into `competition_data/data/` with:

```bash
kaggle competitions download -c autonomous-agent-prediction-beta
```

## Reproduce

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python numpy "pandas<3" scikit-learn catboost lightgbm xgboost pyarrow

# Replay a full session for a config against all 16 labelled tasks
.venv/bin/python bench/run_bench.py \
    --package submissions/07_hardened_freeroll/agent \
    --out results/rerun.json --workers 5

# Paired per-task comparison, and the floor guarantee
.venv/bin/python bench/compare.py     --base results/base822_full.json --challenger results/rerun.json
.venv/bin/python bench/check_floor.py --base results/base822_full.json --challenger results/rerun.json
```

`results/base822_full.json` is the leaderboard-proven core measured on the same 16 tasks, so any
new config can be compared against it directly. The submission archive is not committed; it is
regenerated deterministically (fixed timestamps, sorted entries) by
`notebooks/hardened-freeroll-portfolio.ipynb`, whose output matched
`sha256 = edc67eb7c4d43b8033368b01d248671bc537db616cea3887dbc1704198b3cdaa`.
