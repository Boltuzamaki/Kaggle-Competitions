# NeuroGolf 2026: Solution Writeup

Final score **7440.82** <br>
Rank 196/3059, bronze. <br>

Solo entry, built with two AI collaborators doing
different jobs: ChatGPT generated the actual ONNX candidates and optimization code per task, and
Claude Code was the pair-programmer for everything else, most of the audit tooling, candidate
screening, and merge discipline below was built and run with it rather than by hand.

## Overview

`cost = memory_bytes + params`, `score = max(1, 25 - ln(cost))`, and a task only counts if the
network is 100% correct on every train/test/arc-gen example. Like most solutions here, this
wasn't one clever trick; it was a repeatable audit-then-merge loop applied 400 times, plus a
handful of structural tricks that kept paying off across many tasks at once.

## Workflow

**ChatGPT Plus (WebUI)**: generated the ONNX candidates and optimization code per task. <br>

**Claude Code**: job was narrow but essential: build a local webapp whose only purpose was to
make testing whatever ChatGPT produced fast, drop a candidate in, get pass/fail plus real cost
back in seconds instead of round-tripping through a notebook every time. <br>

Budget was $20 ChatGPT + $20 Claude Code.

The per-task loop, repeated for each task worked on:

1. **Filter from the bucket comparison page**: find where another team's per-task scores beat
   ours the most, that's the shortlist of tasks worth spending time on.
2. **Visualize the task**: look at the train/test grids directly in the UI.
3. **Work out the rule myself**: write it in plain language (e.g. rotate, translate, colour
   remap), not left to the model to guess from the grids alone.
4. **Prompt ChatGPT** with: the current public-best ONNX for that task, my own rule description,
   the notes saved from earlier iterations on that same task (every iteration gets a note saved
   in the UI, so the next attempt starts with what was already tried), and a short brief of the
   problem.
5. **Try ONNX-level optimization first**: tweak/shrink the existing graph rather than starting
   over.
6. **If the gain is under 0.5 points, rewrite the architecture instead**: past that threshold,
   polishing the same graph stops being worth the time. A different representation usually does
   better than a smaller version of the same one.

The notes saved in the UI also fed a second-order pass: once one task in a family was actually
optimized, I'd go back through the saved notes to spot other tasks that looked structurally
similar (same transformation shape, same op pattern) and work through that whole group together,
instead of treating every task as an isolated problem.

![Mission control](https://raw.githubusercontent.com/Boltuzamaki/The-2026-NeuroGolf-Championship/main/images/homepage.png)
![Upload and Verify: score a candidate before deciding whether to keep it](https://raw.githubusercontent.com/Boltuzamaki/The-2026-NeuroGolf-Championship/main/images/upload_onnx_and_test.png)
![Quick Check: standalone scratch runner for fast iteration](https://raw.githubusercontent.com/Boltuzamaki/The-2026-NeuroGolf-Championship/main/images/code_runner_and_tester.png)
![Per-task editor: code, live examples, and notes together](https://raw.githubusercontent.com/Boltuzamaki/The-2026-NeuroGolf-Championship/main/images/code_runner_and_taskwise_notes.png)
![Bucket comparison: where the per-task shortlist in step 1 comes from](https://raw.githubusercontent.com/Boltuzamaki/The-2026-NeuroGolf-Championship/main/images/bucket_to_select_onnx.png)

Beyond the per-task loop, overall task selection (which of the 400 were even worth this much
attention) came from the
[community discussion thread](https://www.kaggle.com/competitions/neurogolf-2026/discussion/708377).
Thanks to **Fritz Cremer** for posting the realistic testing window there: within a given ~5 hour
window, only about 3-5 tasks could actually be taken to a fully optimized, verified state. That
number reframed the whole approach. It's not enough time to touch all 400 seriously (started
late), so the discussion thread's signal was used to prioritize before spending that time.

I only became active in this competition once **ChatGPT 5.6** came out, so the whole effort ran
inside whatever time was left after that, and I got around **150 tasks** through the full loop
above in that window. The other ~250 were never a different kind of problem, just untouched by
this loop for lack of time.

## Validation gate (every candidate, no exceptions)

1. File size under Kaggle's 1.44 MB cap.
2. `Conv`/`ConvTranspose` bias length == `out_channels` (see below, this one mattered a lot).
3. Local correctness, `nfail == 0`, on the full cached train+test+arc-gen set.
4. Cost strictly cheaper than the current best for that task.
5. Hash-memorizer signature scan: `BitShift`/`BitwiseAnd`/`Or`/`Xor` combined with `Gather`, with
   the lookup-table size checked against the local example count.
6. Fresh ARC-GEN generator test: examples never seen in the cached set. A 100% local pass means
   nothing on its own; this is what actually separates a real fix from an overfit one.
7. Timing check for large single-Einsum candidates: a big contraction can look cheap on paper and
   still take 15+ minutes per inference.
8. Cross-check on the pinned `onnxruntime==1.24.4` (not just whatever's newest locally). Catches
   missing kernels for certain dtypes that only show up on the real grader's version.

<img src="https://raw.githubusercontent.com/Boltuzamaki/The-2026-NeuroGolf-Championship/main/images/flow_chart_of_validation.png" alt="Validation flowchart" width="500">

[Full size](https://raw.githubusercontent.com/Boltuzamaki/The-2026-NeuroGolf-Championship/main/images/flow_chart_of_validation.png)

## Techniques that actually moved the score

- **Collapse the whole graph into one Einsum** whenever the transformation is bi/multilinear in
  its inputs. This was the single highest-value pattern in the whole competition: it removes
  every charged intermediate tensor between input and output in one shot, worth anywhere from
  +0.5 to +3 points on tasks where it applied.
- **Quantize counting/matching ops.** If a `Conv`'s real output values are small non-negative
  integers, `QLinearConv` emits `uint8` directly instead of float32, a free 4x cut, zero precision
  loss.
- **Cut initializer data that's provably dead**: zero input channels in a Conv kernel, or two
  "different" constants that turn out identical. Both patterns showed up more than once and both
  roughly halved that task's cost.
- **Check older opsets for attribute-only op variants.** `Upsample` (pre-opset-10) takes `scales`
  as a plain attribute; `Resize`, its replacement, always wants a tensor input, which gets charged
  as params.
- **`params` is element count, not bytes.** dtype is free for small lookup/permutation tables, but
  intermediate tensors are charged `dtype_size × elements`, so reserve `uint8`/`int8` for the
  large *intermediate* tensors, not small constant tables where it makes no difference.
- **A cheaper op isn't a cheaper graph if it needs a second op to feed it.** Swapping a 5-param
  `MaxRoiPool` for a 0-param `Slice`+`Upsample` pair "won" on params but cost 36,000 vs 5 once the
  intermediate tensor between them was counted. Always measure end-to-end through the real
  scorer, never reason about params/ops in isolation.

## Novel ideas, with real before/after numbers

Techniques that improved score (+0.15pt/task or better counted as significant, ~+17.9 points
total across these, plus task067/179/241 hitting the max 25pt outright via the zero-param
direct-output template):

- Collapse the entire rule into one direct-to-output Einsum: task303 1450→68 (+3.060), task197
  1141→144 (+2.070), task041 1787→551 (+1.177)
- Factor dense channel-routing matrices into low-rank codes: task304 1320→260 (+1.625), task032
  910→408 (+0.802)
- Use the input itself as a dynamic convolution kernel: task082 190→28, dynamic-weight
  ConvTranspose (+1.915)
- Bit-pack spatial state instead of materializing masks: task034 2072→645, packed anchors,
  diagonal bit-fields, and BitwiseAnd (+1.167)
- Encode positions as scalars rather than vectors: task036 1589→596, powers-of-16 positional
  encoding and logarithmic decoding (+0.981)
- Write the full output directly: task042 3109→1107, cropped processing plus asymmetric-padding
  final Conv (+1.033)
- Express images as a few outer-product factors: task199 1823→883, three factors in the final
  Einsum (+0.725)
- Polynomial equality/color routing: task197 uses `0.5 - (a-b)^2` (+2.070), task046 uses
  `1 - (x-c)^2` in `QLinearConv` (+0.226)
- Exploit grouped channel topology: task372 710→360, one `Conv(group=2)` (+0.679)
- Share affine bases and constants aggressively: task084 419→205, two affine tensors replaced
  with shared sparse bases (+0.715)
- Reuse the input as multiple operands: task373 60→30 (+0.693), task375 327→227 (+0.365)
- Quantized compact detectors instead of propagation pipelines: task196 4536→3387 (+0.292),
  task031 637→519 (+0.205)
- Replace merged geometric constructions with independent clues: task035 1891→1588, 18 border
  clues and TopK (+0.175)

Blockers (looked like wins, didn't count):

- task303's dynamic-`Range` cost-5 model: functionally correct, but statically unscorable.
- task294 and task033's `com.microsoft::FusedConv` versions: non-standard domain, rejected by
  the scorer.
- Sparse-Conv candidates: rejected by full ONNX checking.
- task001's all-zero inactive-logit version: later found validator-invalid.
- task302's final verified improvement: only ~+0.0096, below the significance bar.
- task304's newer cost-258 builder: promising, but conservatively reported as the already-audited
  cost-260 result above instead.

## What was genuinely problematic

- **Conv/ConvTranspose bias out-of-bounds read.** Bias shorter than `out_channels` reads past the
  buffer: undefined behavior. Invisible locally (fresh process, zeroed heap), but Kaggle's grader
  reuses process memory across submissions, so a previous run's leftover bytes leak in, causing
  wrong, non-deterministic, order-dependent scores. Explained every "identical resubmission scores
  differently" mystery we hit. Only catchable by static inspection, not by running the file.
  Confirmed by several other top teams too, not unique to us.
- **`TopK` silently has no INT8/UINT8 kernel on the real grader's pinned `onnxruntime`.** Passes
  locally without any error on a newer runtime version, so it's completely invisible unless you
  actually run the candidate against the exact pinned version Kaggle uses, not just whatever's
  newest on your own machine.
- **A single large `Einsum` can look cheap on paper and still hang.** A 50-operand contraction
  with a long equation string took 15+ minutes for one inference, purely from contraction-order
  blowup, despite a tiny params/cost footprint. Cost and param count say nothing about actual
  runtime; that only shows up by timing a real inference directly.
- **An over-strict test script is as damaging as a too-lenient one.** The official conversion
  function silently skips grids bigger than 30×30; a hand-rolled check that misses this produces
  huge fake failure rates (one task showed 296/400 "failures" that were just oversized grids the
  real grader never sees). Always route through the exact official function.

## Repo

[github.com/Boltuzamaki/The-2026-NeuroGolf-Championship](https://github.com/Boltuzamaki/The-2026-NeuroGolf-Championship)

## Citation

```
boltuzamaki, "NeuroGolf 2026 Solution Writeup", 2026.
https://github.com/Boltuzamaki/The-2026-NeuroGolf-Championship
```
