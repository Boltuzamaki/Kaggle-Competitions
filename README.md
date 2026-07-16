# The 2026 NeuroGolf Championship

Solutions and tooling for [NeuroGolf 2026](https://www.kaggle.com/competitions/neurogolf-2026), a Kaggle competition where you build the smallest possible ONNX network that solves each of 400 ARC-AGI visual reasoning tasks. Score per task is `max(1, 25 - ln(memory + params))`, and a task only counts if the network is 100% correct on every train/test/arc-gen example: one wrong output zeroes the whole task.

**Final score: 7440.82** (rank ~193/3059, bronze medal).

Want to learn ONNX from this project instead of just the results? `docs/onnx-learning-guide.html` is a from-scratch tutorial covering the basics, how NumPy/math map to ONNX ops, every op family used here, and the real bugs we hit along the way. For the short version (workflow, what worked, what went wrong) see `docs/SOLUTION_WRITEUP.md`.

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

The full gate sequence a candidate has to clear before step 4, in order:

![Validation flowchart](images/flow_chart_of_validation.png)

### Webapp

The webapp wraps this in a UI: edit a task, see the audit run live, browse history, and build the
final `submission.zip` (always assembled from the best verified file per task).

```
cd webapp && docker compose up -d --build   # -> http://localhost:5000
```

Mission control, all 400 tasks at a glance:

![Mission control](images/homepage.png)

Per-task editor, code, live examples, and notes side by side:

![Task editor](images/code_runner_and_taskwise_notes.png)

Quick Check, a standalone scratch runner, separate from a task's saved history:

![Quick Check](images/code_runner_and_tester.png)

Upload & Verify, score a candidate `.onnx` file before deciding whether to keep it:

![Upload and Verify](images/upload_onnx_and_test.png)

Bucket comparison, paste another team's per-task scores to find where they're beating us:

![Bucket comparison](images/bucket_to_select_onnx.png)

## What actually moved the score

Cost is `memory + params`, then `25 - ln(cost)`. It's log-scaled, so halving a cost is worth the
same points whether it started at 60 or 60,000: cheap-looking tasks aren't low priority.

- Quantize counting/matching ops: `QLinearConv` emits `uint8` instead of float32 when the values
  allow it, a free 4x cut.
- Cut initializer data that's provably dead: zero-value Conv channels, duplicate constants.
- Collapse a graph into one `Einsum` when the transformation is bi/multilinear. Best single pattern
  in the competition; removes every intermediate tensor between input and output.
- Older opsets sometimes hide an attribute-only version of an op. `Upsample` takes `scales` as an
  attribute pre-opset-10; `Resize` charges it as a tensor input instead.
- A cheaper op isn't a cheaper graph if it needs a second op to feed it: a 5-param `MaxRoiPool`
  beats a 0-param `Slice`+`Upsample` pair, because the intermediate between them costs 36,000 bytes.
  Always measure end-to-end through the real scorer.
- `params` is element count, not bytes. dtype is free for small lookup tables, but intermediate
  tensors are charged `dtype_size × elements`, so use `uint8`/`int8` there.

## Lessons that cost real points to learn

- A clean local pass isn't proof. Candidates that passed every cached example still scored wrong on
  the real grader. Anything resting on an empirical claim about the data needs an isolated
  submission before you trust it.
- Node count isn't cost. Several "optimizations" shrank node count but grew byte cost.
- Always re-audit fresh: every regression traced back to comparing against a stale `tracker.db`
  value instead of re-checking both candidates right before deciding.
- Negative-pad models are a special case: some fail the local checker but score fine on the real
  grader. Several turned out to be genuine wins.
- Watch Conv/ConvTranspose bias length. A bias shorter than `out_channels` is an out-of-bounds read,
  invisible locally, but it produces wrong, non-deterministic results on a grader that reuses process
  memory across submissions. See `scratch_onnx/check_conv_bias.py`, run it before every submission.
- Route correctness checks through the official conversion function, not a hand-rolled one. It
  silently skips grids bigger than 30×30, and a stricter test script produces false failures too.
- When a batch's score doesn't match the sum of its predicted gains, suspect your own test setup
  first. Every mismatch traced back to a contaminated base folder, not a real interaction between
  unrelated per-task models.

## Cost-0 tasks

`points = max(1, 25 - ln(cost))`, so a cost of 1 gives the max score, 25.0. Three tasks here hit that
floor with a single parameter-free op (e.g. a bare `Transpose`), and six more sit at cost 5 to 10.
Getting there took real dead-ends worth recording: replacing `MaxRoiPool` with a `Slice`+`Upsample`
pair looked cheaper on paper (0 params vs 5) but the intermediate tensor between the two ops costs
36,000, a 7000x regression, not a win.

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
