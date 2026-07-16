# The 2026 NeuroGolf Championship

Solutions and tooling for [NeuroGolf 2026](https://www.kaggle.com/competitions/neurogolf-2026) — a Kaggle competition where you build the smallest possible ONNX network that solves each of 400 ARC-AGI visual reasoning tasks. Score per task is `max(1, 25 - ln(memory + params))`, and a task only counts if the network is 100% correct on every train/test/arc-gen example — one wrong output zeroes the whole task.

**Final score: 7440.82** — rank ~193/3059, bronze medal.

Want to learn ONNX from this project instead of just the results? `docs/onnx-learning-guide.html` is a from-scratch tutorial covering the basics, how NumPy/math map to ONNX ops, every op family used here, and the real bugs we hit along the way.

## Layout

```
data/              Task definitions (task001.json ... task400.json) + the official scorer
repairs/           Our best ONNX file per task (400 files) + tracker.db (state/cost/notes per task)
  user_code/         Per-task Python source that builds/repairs each graph
other_model_onnx/  Candidate files (ours or found elsewhere) waiting on or after review
webapp/            Flask tracker app: browse/edit/audit tasks, build + submit
scripts/           Analysis and audit scripts
arc_dsl_ref/       Reference ARC-DSL solvers used to cross-check task rules
*.ipynb            Working notebooks
```

Not tracked in git: `baseline_v22/` (the public baseline this builds on top of), `submissions/`
(downloaded candidate folders used for comparison), and scratch/log directories.

## How a task gets solved

1. Work out the transformation rule from the train/test examples.
2. Build or repair an ONNX graph that implements it, within the competition's constraints (static
   shapes, no `Loop`/`Scan`/`NonZero`/`Unique`/`Compress`/subgraphs).
3. Audit it for real: run the exact scorer logic (`sanitize_model` + `score_network`,
   `onnx.checker.check_model(full_check=True)`, every train/test/arc-gen example) on the pinned
   `onnxruntime` version. Only a 100% pass counts.
4. If it beats the current best for that task, merge it into `repairs/` and note why in `tracker.db`.

The webapp wraps this in a UI — edit a task, see the audit run live, browse history, and build the
final `submission.zip` (always assembled from the best verified file per task).

```
cd webapp && docker compose up -d --build   # -> http://localhost:5000
```

## What actually moved the score

The cost formula is `memory + params` (raw bytes/elements), then `25 - ln(cost)`. Because it's
log-scaled, halving a cost is worth the same points whether it started at 60 or 60,000 — small tasks
aren't automatically low priority.

- **Quantize where the math allows it.** If a `Conv`'s output values are always small non-negative
  integers, `QLinearConv` can emit `uint8` instead of float32 — a free 4x cut on that tensor.
- **Cut initializer data that's provably dead.** Zero input channels in a Conv kernel, or two
  "different" constants that turn out identical — both showed up and both roughly halved a task's
  cost once removed.
- **Collapse a graph into one Einsum** when the transformation is bi/multilinear in its inputs. The
  single best-value pattern in the whole competition — it removes every charged intermediate tensor
  between input and output at once.
- **Check older opsets for attribute-only versions of an op.** `Upsample` takes `scales` as an
  attribute pre-opset-10; `Resize` always wants it as a tensor input (which gets charged as params).
- **A cheaper op isn't a cheaper graph if it needs a second op to feed it.** We tried swapping a
  5-param `MaxRoiPool` for a 0-param `Slice`+`Upsample` pair — "won" on params, lost badly overall
  (36,000 vs 5) because the intermediate tensor between them gets charged in raw bytes. Always measure
  end-to-end cost through the real scorer.
- **`params` is counted by element count, not bytes** — dtype is free for small constant/lookup
  tables, but intermediate tensors are charged `dtype_size × elements`, so use `uint8`/`int8` there
  when it's safe to.

None of this is the same as fitting a rule to match only the examples you can see — that's the
subject of the next section.

## Lessons that cost real points to learn

- **A clean local pass isn't proof.** A couple of candidates passed every cached example and still
  scored wrong on the real grader — our local `arc-gen` sample doesn't cover everything the
  production grader checks. Structural rewrites (De Morgan folds, TopK-tiebreak collapses) held up
  fine; anything resting on an empirical claim about the data ("this branch is always true here")
  needs an isolated Kaggle submission before you trust it.
- **Node count isn't cost.** Cost is total tensor bytes. Several "optimizations" that shrank node
  count actually grew the byte cost by materializing a bigger intermediate.
- **Always re-audit fresh, never trust a cached number.** Every regression we hit traced back to
  comparing against a stale `tracker.db` value instead of re-checking both candidates right before
  deciding.
- **Negative-pad models are a special case.** Some negative-pad `Conv`/`ConvTranspose`/`MaxPool`
  graphs fail the local checker but score fine on the real grader. The only way to know is an
  isolated submission — several turned out to be genuine wins.
- **Watch Conv/ConvTranspose bias length.** If the bias initializer has fewer elements than
  `out_channels`, ONNX Runtime reads past the buffer — undefined behavior that's invisible locally
  (fresh process, zeroed memory) but produces wrong, non-deterministic results on a grader that
  reuses the same process across submissions. This caused every "identical resubmission scores
  differently" mystery we hit. Check `bias_len != out_channels` directly — running the file proves
  nothing. See `scratch_onnx/check_conv_bias.py`, and run it before every submission.
- **Route correctness checks through the official conversion function, not a hand-rolled one.** It
  silently skips any example bigger than 30×30. A stricter test script than the real scorer produces
  false failures just as damaging as a too-lenient one — one task showed 296/400 fake "failures" that
  were all oversized grids the real grader never sees.
- **When a batch of verified changes doesn't score what the sum predicts, suspect your own test
  setup first.** Isolated single-task tests were accurate to within 0.01–0.02 points every time.
  Every "combined batch doesn't match" mystery we hit traced back to a contaminated base folder, not
  a real interaction between unrelated per-task models (there isn't one — scoring is per-task).

## Cost-0 tasks

`points = max(1, 25 - ln(cost))`, so a cost of 1 gives the max score, 25.0. Three tasks here hit that
floor with a single parameter-free op (e.g. a bare `Transpose`), and six more sit at cost 5–10.
Getting there took real dead-ends worth recording: replacing `MaxRoiPool` with a `Slice`+`Upsample`
pair looked cheaper on paper (0 params vs 5) but the intermediate tensor between the two ops costs
36,000 — a 7000x regression, not a win.

## Setup

```
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
```

A second pinned environment (not committed) mirrors the production scorer's `onnxruntime` version for
parity-checking candidates before submitting. Kaggle CLI needs `kaggle.json` configured to submit
directly from the webapp or `scripts/`.

## Citation

```
boltuzamaki, "The 2026 NeuroGolf Championship", 2026.
https://github.com/Boltuzamaki/The-2026-NeuroGolf-Championship
```

## License

[MIT](LICENSE)
