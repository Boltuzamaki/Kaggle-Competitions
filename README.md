# The 2026 NeuroGolf Championship

Solutions and tooling for [NeuroGolf 2026](https://www.kaggle.com/competitions/neurogolf-2026), a Kaggle competition where you build the **smallest possible ONNX network** that solves each of 400 ARC-AGI visual reasoning tasks. Score per task is `max(1, 25 - ln(memory + params))`, and a task only counts if the network is **100% correct** on every train, test, and arc-gen example. Any mistake zeroes that task.

**Current best: 7272.64** (public leaderboard, 2026-07-12).

---

## Repository layout

```
data/                   Task definitions (task001.json ... task400.json) + the official scorer
  neurogolf_utils/       neurogolf_utils.py: sanitize_model / score_network / run_network etc.
repairs/                Our current best ONNX file per task (400 files) + the tracker database
  task001.onnx ...       The actual submitted models
  user_code/             Per-task Python source that builds/repairs each ONNX graph
  tracker.db             SQLite source of truth: state / points / cost / n_fail / notes per task
  catalog.csv            Full 400-task catalog: cost, points, ops, DSL rule
  autosolved.json        Tasks solved by a generic templated rule (no per-task script)
other_model_onnx/        Candidate ONNX files (ours or found elsewhere) awaiting or after audit
webapp/                 Flask tracker app (Docker): browse/edit/audit all 400 tasks, build + submit
scripts/                Reusable analysis/audit/golf scripts
arc_dsl_ref/            Reference ARC-DSL solvers used to cross-check task rules
*.ipynb                 Working notebooks (see below)
```

Not tracked in git (present locally, see `.gitignore`): `baseline_v22/` (the public baseline this
project builds on top of, third-party and not ours), `submissions/` (downloaded candidate
zips/folders from public notebooks, used for comparison), and various scratch/output/log
directories from day-to-day iteration.

### Notebooks

- `from_scratch.ipynb`: from-scratch task builds with math derivations and plots
- `reverse_engineer_all.ipynb`: a browsable catalog of all 400 tasks (graph + rule + cost)
- `onnx_from_scratch_tutorial.ipynb`: a standalone ONNX-from-numpy phrasebook (doesn't solve any task)
- `neurogolf_best_7171.ipynb`: submission repro notebook (name is legacy; actual best is well above 7171 now)

---

## How a task gets solved

1. Read the task's train/test examples, figure out the transformation rule.
2. Build (or repair) an ONNX graph that implements it, respecting the competition's constraints:
   static shapes only, `input`/`output` tensors named exactly that, no `Loop`/`Scan`/`NonZero`/
   `Unique`/`Compress`/`Sequence*`/custom domains/subgraphs.
3. **Audit it for real.** Run it through the exact scorer logic the competition uses
   (`neurogolf_utils.sanitize_model` + `score_network`, `onnx.checker.check_model(full_check=True)`,
   all train+test+arc-gen examples) on a pinned `onnxruntime==1.27.0` environment. Only `nfail == 0`
   counts as solved.
4. If it's a genuine improvement over the current best for that task, it gets merged into `repairs/`
   and `tracker.db` records why: cost before/after, what changed, any caveats.

The webapp (`webapp/`) wraps this whole loop with a UI: paste/edit code or wire up a visual graph
editor per task, see the audit result live, browse version history, and build/submit the final
`submission.zip`. It always assembles the best verified file per task (`repairs/` vs the public
baseline) and refuses to ship an incomplete or corrupt zip. There's also a bucket comparison page
(`/buckets`) for pasting another team's per-bucket score table and finding the biggest gaps.

```
cd webapp && docker compose up -d --build
# -> http://localhost:5000
```

---

## Cost-reduction strategies that have actually worked

The cost formula is `memory + params`, added together as raw byte/element counts, then the whole
sum goes through `25 - ln(cost)`. Two consequences worth internalizing before golfing anything:
cost is dominated by whichever single tensor is biggest, so look there first; and because the score
is log-scaled, cutting a cost in half is worth the same points (`ln(2) ≈ 0.69`) whether the task
started at 60 or 60000. Small, cheap-looking tasks are not automatically low priority.

- **Quantize counting and matching operations.** `Conv` on a float input forces a float32
  (4 bytes/element) output. If the actual computed values are small non-negative integers, a
  `QLinearConv` doing the same arithmetic can output `uint8` directly (1 byte/element) for a flat
  4x cut on that tensor, with zero precision loss as long as the values never leave the uint8 range.
  Same idea applies to `ConvInteger`, which is *forced* to int32 output by the ONNX spec; swapping
  it for `QLinearConv` recovers the same 4x.
- **Trim initializer data that's provably never used.** Two of the biggest single-task wins this
  project found were exactly this: a Conv kernel with several entire input channels that were
  always zero (dropped, since a zero channel contributes nothing to the sum), and an Einsum with
  two "different" constant vectors that turned out to be identical and got collapsed into one
  shared initializer. Both cases roughly halved that task's cost for a clean, provable win.
- **Bake a static shape around a dynamic position.** If a `Slice`'s start position depends on the
  input but its size (end minus start) is always the same fixed value, hardcode that size as a
  constant offset instead of leaving both ends dynamic. The result keeps a fully static shape (which
  the scorer requires) while still adapting to wherever the input actually puts the interesting
  region.
- **Fold a negative-pad Conv over a separate Slice/crop**, but only when it's provably replacing a
  real crop step that the network would otherwise need anyway. This has held up every time it was
  used to eliminate a genuine step; it has broken things when used speculatively with no clear
  savings mechanism behind it.
- **Prefer algebraic identities over explicit search.** Finding a "second largest value" via
  `sum_of_all - max` is cheaper than masking absent entries with a sentinel and running `ReduceMin`,
  whenever the structure of the task guarantees there are only two distinct values in play. Fewer
  nodes, fewer constants, same result.
- **Drop axes that are always size 1.** Slicing along the batch dimension is usually pointless work
  since every input in this competition has batch size 1; omitting it from the `axes` list of a
  `Slice` costs nothing and occasionally lets shape inference simplify downstream ops too.
- **Watch for a fragmented cost profile before committing to a rewrite.** Some tasks look expensive
  but the cost is spread thin across a couple hundred small tensors rather than concentrated in a
  handful of big ones. A full rewrite of that kind of graph tends to reproduce the same fragmentation
  rather than deliver the order-of-magnitude cut a quick glance at the total cost suggests; profile
  the per-tensor breakdown before investing serious time.

None of the above should be confused with fitting a threshold or weight to match the locally visible
examples. See the next section for why that distinction matters.

---

## Hard-won lessons (read before golfing further)

These cost real leaderboard points to learn, so they're written down instead of re-learned:

- **A clean local audit (`nfail == 0` on every cached example) is necessary but not sufficient.**
  Two external patches this project tried passed the full local train+test+arc-gen audit perfectly,
  then scored badly wrong on the real grader. Root cause: the local `arc-gen` snapshot is an
  incomplete sample of whatever the production grader actually checks against. Pure structural or
  algebraic rewrites (De Morgan folds, TopK-tiebreak-to-scalar collapse, Gather-to-Split folding)
  have held up every time; anything that depends on an *empirical* claim about the data ("this
  branch is always true," "this tensor is always zero here," "this threshold happened to separate
  every example I've seen") needs an isolated single-task Kaggle submission before it's trusted, no
  matter how clean the local pass looks.
- **Fewer ONNX nodes does not mean lower score.** The cost formula is `memory + params` (total
  tensor bytes), not node count. Several "optimizations" that shrank the node count *increased* the
  byte cost by materializing a bigger intermediate tensor or consolidating small ops into one with a
  larger footprint.
- **Never trust a stale or indirect number when a fresh direct comparison is possible.** Every
  regression this project hit came from some version of this: comparing a new candidate against a
  cached `tracker.db` value instead of re-auditing fresh, or assuming an older "historically proven"
  file must still beat a newer, already-confirmed download without checking. The fix is always the
  same: audit both candidates with the same methodology, right before deciding, every time.
- **The negative-pads/checker-only-failure tasks are a special case.** Some ONNX models with
  negative `pads` (Conv/ConvTranspose/MaxPool) get rejected by
  `onnx.checker.check_model(full_check=True)` locally, but the real Kaggle grader scores them
  anyway. Don't assume these are zero from a local audit alone; the only way to compare two
  candidates for these specific tasks is an isolated submission.

---

## Setup

```
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # onnxruntime>=1.27, for local dev/webapp
# a second pinned env (not committed) mirrors the production scorer's onnxruntime==1.24.4 for
# parity-checking golf candidates before submitting
```

Kaggle CLI must be configured (`kaggle.json`) to submit directly from `webapp` or the scripts in
`scripts/`.
