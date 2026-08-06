# Exact 6.568 public-parent audit

Date: 2026-08-04

## Identity and integrity

- Kernel: `hjyact/ultimate-pf-config-strategy-a-reproducible-score`
- Kaggle kernel id: `128161011`
- Current ordinal version: `2`
- Referenced scriptVersionId: `337064157`
- Pulled source/output staging directory: `/tmp/hjyact_parent`
- Exact scored-parent `submission.csv` SHA256:
  `b192d3f348ae00680dc4df942b95cef5fd708c636a741f77dfb6b6e89b9ded4a`

The pulled output is byte-identical to
`ilog/frontier_20260722/public_tvt_solution/output/submission.csv`. This closes
the provenance chain used by the downstream Contact and U Restore notebooks.

## What produced the output

The notebook is a public mega-stack, not a from-raw standalone PF solution.
It mounts a ridge artifact dataset, pretrained learned-model packages and a
second model package, and its learned branch permits an id-exact precomputed
submission fallback.

More importantly, the final trajectory uses exact same-well train/test overlap:

| well | contact prefix RMSE | overridden suffix rows |
|---|---:|---:|
| `000d7d20` | 0.010077 | 3,836 |
| `00bbac68` | 0.009044 | 6,014 |
| `00e12e8b` | 0.007855 | 4,301 |

Thus all 14,151 visible scoring rows are replaced by EGFDU contact
reconstruction from their exact training twins. After this, a PF seed-branch
hedge adds exactly +2 ft to all 4,301 rows of `00e12e8b`. Its nominally generic
constants are strength 0.60, minor mass 0.25, separation 4--40 ft and cap 2 ft,
but the intervention was developed from public-well/leaderboard response and
cannot explain a clean generalization score.

The selected visible-prefix profile makes no numerical move. The model-package
correction is auto-disabled. Consequently, the decisive final layers are the
exact overlap reconstruction and the one-well +2-ft branch response.

## Clean-component disposition

The reusable from-raw mechanisms are PF/beam tracking, heel-only affine GR
calibration, likelihood-width changes, normalized-`U=TVT+Z` projection and a
bimodal likelihood scan. They are not new relative to the local backlog:

- affine-calibrated PF reference: 12.161625 -> 14.659568, rejected;
- PF gamma scale 1.3: 12.087 average -> 14.583 across robust seed banks,
  rejected;
- Setchell/normalized structural projection: lost at full-field add-one;
- prefix-calibrated Student-t/PDE update: small pilot gain, failed locked
  confirmation gate;
- public bimodal/midpoint hedge: previously audited as leaderboard-tuned and
  not an independently validated general rule.

No genuinely new clean component remains to add to the immutable 8.086407
level-2 OOF center. Re-running a renamed copy of an already rejected mechanism
would not constitute a new strict-CV experiment.

## Submission monitor

Competition submission `55220585` was still `PENDING` with no score at the
latest check in this audit. No submission was made.
