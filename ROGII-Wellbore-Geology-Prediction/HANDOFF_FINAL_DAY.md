# Handoff — final day (deadline 2026-08-05 23:59 UTC = 05:29 IST on the 6th)

Written 2026-08-04 ~19:50 UTC. No submissions were made after the 05:00 IST cutoff.

## Submission slots

- Yesterday's five are spent (see table below).
- Five fresh slots became available at **00:00 UTC / 05:30 IST**.
- Competition closes **2026-08-05 23:59 UTC**.

## What is on the board

| ref | shift on `00e12e8b` | what it tests | score |
|---|---:|---|---:|
| 55220585 | +4.201 | public Contact-and-U-Restore backup | **6.463** |
| 55234329 | n/a | guarded twin, exact-match only | 15.883 |
| 55239047 | 0.000 | your lineage, cleaned | pending |
| 55245052 | 3.920 | faithful rerun of public 6.390 | pending |
| 55246258 | 3.500 | 6390 lineage, id constant removed | pending |
| 55248032 | 0.000 | 6390 lineage, zero shift, fully clean | pending |

Kaggle's rerun queue was ~8 h behind at the time of writing; that is why none of
the four had scored.

## The one number that decides everything

`55245052` is a faithful rerun of the notebook that reports **6.390**. The bronze
cut was **6.407**. If it reproduces, bronze is already secured and no further
submission is strictly needed.

## How to read the four pending scores

They form a dose-response series on a single constant added to well
`00e12e8b`'s 4,301 rows. Compare `55246258` (3.500) against `55245052` (3.920):
these are the *same pipeline* differing by exactly -0.420 on those rows and
0.00e+00 elsewhere, so their difference measures the slope cleanly.

- **55246258 < 55245052** — the slope is real and negative, the id-targeted
  constant was hurting, and `55248032` (zero shift) should be the best of the
  four. Select it.
- **within ~0.05 of each other** — the constant is inert and the apparent public
  ladder was noise. Select whichever scored best; do not chase it further.
- **55246258 > 55245052** — the probing was directionally right. Keep 6.463 or
  55245052, whichever is lower.

Remember the measured resubmission noise: **sd 0.037 ft**, so two single draws
need about **0.10 ft** before a difference means anything.

## Recommended use of the five fresh slots

1. **Do nothing until at least two of the pending four have scored.** Submitting
   more variants blind adds noise, not information.
2. Do **not** submit further perturbations of the shift constant and keep the
   best draw. That is leaderboard probing, it will not survive to the private
   board, and it is the thing you ruled out.
3. The highest-value remaining action is **final submission selection**, not
   another variant.

## Final selection

Kaggle scores only the submissions you select. Suggested pairing:

- the lowest-scoring submission on the public board, and
- `55248032` (or whichever clean zero-shift variant scored best) as the
  private-safe pick, since it carries no leaderboard-derived constant and
  degrades gracefully.

## Honest position on silver

Silver needs **6.356**; you are at 6.463. That gap is 0.107 ft, about 2.8x the
noise floor, so it cannot arrive by a lucky draw. No public notebook anywhere
reaches it — best public claims are 6.390 and 6.568. Bronze is the realistic
target and may already be in hand via 55245052.

Full detail, including two mid-session corrections where earlier conclusions
were retracted, is in `EXPERIMENT_LEDGER.md` under the 2026-08-04 entries.
