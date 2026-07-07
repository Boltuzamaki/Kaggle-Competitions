# The 2026 NeuroGolf Championship

Solutions and tooling for [NeuroGolf 2026](https://www.kaggle.com/competitions/neurogolf-2026), a Kaggle competition where you build the **smallest possible ONNX network** that solves each of 400 ARC-AGI visual reasoning tasks. Score per task is `max(1, 25 − ln(memory + params))`, and a task only counts if the network is **100% correct** on every train, test, and arc-gen example — any mistake zeroes that task.

**Current best: 7243.96** (public leaderboard, 2026-07-07).

---

## Repository layout

```
data/                   Task definitions (task001.json ... task400.json) + the official scorer
  neurogolf_utils/       neurogolf_utils.py — sanitize_model / score_network / run_network etc.
repairs/                Our current best ONNX file per task (400 files) + the tracker database
  task001.onnx ...       The actual submitted models
  user_code/             Per-task Python source that builds/repairs each ONNX graph
  tracker.db             SQLite source of truth: state / points / cost / n_fail / notes per task
  catalog.csv            Full 400-task catalog: cost, points, ops, DSL rule
  autosolved.json        Tasks solved by a generic templated rule (no per-task script)
webapp/                 Flask tracker app (Docker) — browse/edit/audit all 400 tasks, build + submit
scripts/                Reusable analysis/audit/golf scripts
arc_dsl_ref/            Reference ARC-DSL solvers used to cross-check task rules
*.ipynb                 Working notebooks (see below)
```

Not tracked in git (present locally, see `.gitignore`): `baseline_v22/` (the public baseline this
project builds on top of — third-party, not ours), `submissions/` (downloaded candidate
zips/folders from public notebooks, used for comparison), and various scratch/output/log
directories from day-to-day iteration.

### Notebooks

- `from_scratch.ipynb` — from-scratch task builds with math derivations and plots
- `reverse_engineer_all.ipynb` — a browsable catalog of all 400 tasks (graph + rule + cost)
- `onnx_from_scratch_tutorial.ipynb` — a standalone ONNX-from-numpy phrasebook (doesn't solve any task)
- `neurogolf_best_7171.ipynb` — submission repro notebook (name is legacy; actual best is well above 7171 now)

---

## How a task gets solved

1. Read the task's train/test examples, figure out the transformation rule.
2. Build (or repair) an ONNX graph that implements it, respecting the competition's constraints:
   static shapes only, `input`/`output` tensors named exactly that, no `Loop`/`Scan`/`NonZero`/
   `Unique`/`Compress`/`Sequence*`/custom domains/subgraphs.
3. **Audit it for real** — run it through the exact scorer logic the competition uses
   (`neurogolf_utils.sanitize_model` + `score_network`, `onnx.checker.check_model(full_check=True)`,
   all train+test+arc-gen examples) on a pinned `onnxruntime==1.27.0` environment. Only `nfail == 0`
   counts as solved.
4. If it's a genuine improvement over the current best for that task, it gets merged into `repairs/`
   and `tracker.db` records why (cost before/after, what changed, any caveats).

The webapp (`webapp/`) wraps this whole loop with a UI: paste/edit code or wire up a visual graph
editor per task, see the audit result live, browse version history, and build/submit the final
`submission.zip` (always assembles the best verified file per task — `repairs/` vs the public
baseline — and refuses to ship an incomplete or corrupt zip).

```
cd webapp && docker compose up -d --build
# -> http://localhost:5000
```

---

## Hard-won lessons (read before golfing further)

These cost real leaderboard points to learn, so they're written down instead of re-learned:

- **A clean local audit (`nfail == 0` on every cached example) is necessary but not sufficient.**
  Two "verified" external patches this project tried passed the full local train+test+arc-gen audit
  perfectly, then scored ~0 on the real grader. Root cause: the local `arc-gen` snapshot is an
  incomplete sample of whatever the production grader actually checks against. Pure structural/
  algebraic rewrites (De Morgan folds, TopK-tiebreak-to-scalar collapse, Gather→Split folding) have
  held up every time; anything that depends on an *empirical* claim about the data ("this branch is
  always true," "this tensor is always zero here," "this alias is verified on available examples")
  needs an isolated single-task Kaggle submission before it's trusted, no matter how clean the local
  pass looks.
- **Fewer ONNX nodes does not mean lower score.** The cost formula is `memory + params` (total tensor
  bytes), not node count. Several "optimizations" that shrank the node count *increased* the byte
  cost by materializing a bigger intermediate tensor or consolidating small ops into one with a
  larger footprint.
- **Never trust a stale or indirect number when a fresh direct comparison is possible.** Every
  regression this project hit came from some version of this: comparing a new candidate against a
  cached `tracker.db` value instead of re-auditing fresh, or assuming an older "historically proven"
  file must still beat a newer, already-real-world-confirmed download without checking. The fix is
  always the same — audit both candidates with the same methodology, right before deciding, every time.
- **The 14 negative-pads/checker-only-failure tasks are a special case.** Some ONNX models with
  negative `pads` (Conv/ConvTranspose/MaxPool) get rejected by `onnx.checker.check_model(full_check=True)`
  locally, but the real Kaggle grader scores them anyway. Don't assume these are 0 from a local
  audit alone — the only way to compare two candidates for these specific tasks is an isolated
  submission.

---

## Setup

```
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # onnxruntime>=1.27, for local dev/webapp
# a second pinned env (not committed) mirrors the production scorer's onnxruntime==1.24.4 for
# parity-checking golf candidates before submitting
```

Kaggle CLI must be configured (`kaggle.json`) to submit directly from `webapp` or the scripts in
`scripts/`.
