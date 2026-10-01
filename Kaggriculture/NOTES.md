# Kaggriculture - state

## Live
- Submitted **v10** (`56328723`), rating climbing (680 -> 932 within the first hour;
  new submissions start near the old rating and converge over a day or two).
- v10 is the Apache-2.0 shop-router lineage (`aurax7` EXP257) adopted wholesale.
  `main.py` == `champion/main.py` == `gauntlet/base_aurax7.py`.
- Our own v9 is parked at `candidates/v9_ours_REPLACED.py`.

## What happened to v9 (score 645.9, rank 5472 / 9415)
Engine **1.32.7** (Aug 15) gave CARROT, TOMATO and EGG a `hinge` price curve on
the scarcity side: price is calm until market inventory falls past `T`, then runs
away quadratically (`HINGE_GAIN = 8`). CARROT's `below_target` went 0.20 -> 1.00.
Meanwhile the old premium goods still collapse on a glut - MELON, WOOL, MILK and
STRAWBERRY hit the $1 floor about 100-200 units above `I0`.

v9 is a melon / wool / milk / strawberry farm, which is exactly the half of the
board that stopped paying. Worse, the local venv was pinned at **1.32.6**, so every
sweep run after Aug 15 was tuned against a game that no longer existed. The venv is
now on 1.32.7 and `arena.py` keys its cache by engine version.

## Calibration: the public gauntlet is saturated
All nine agents extracted from public notebooks lose **100%** of games to the v10
base. Forcing each of the nine route tapes against two of them scored a flat 1.000
- no resolving power at all. Round-robin over the ten strongest public agents:

    aurax7 1.00 | ahmedberatozer 0.87 | flexonafft 0.76 | thomastschinkel 0.70
    tetsutani 0.55 | boatlee 0.44 | raykkretzschmar 0.31 | kaitofukami 0.20
    prvsiyan 0.09 | pilkwang 0.07 | ours_v9 0.00  (108 games each)

So the **mirror** (candidate vs the unmodified base) is the only local opponent
that can separate candidates, and `promote.py` was rewritten around that: the h2h
paired bootstrap CI is now the deciding gate and the gauntlet is a no-regression
floor. A control variant identical to the base ties every game at margin exactly
0, which is the harness self-check.

## The base is an open-loop plan
13 precomputed 719-step route tapes plus reactive repair layers. `_router` reads
the day-6 shop pair and picks a route; step 648 swaps in the terminal route.
Two defects in the shipped table: five of its fifteen entries name routes 9-12
that were never built (the chassis silently falls back to route 0), and it only
covers pairs containing YARN_STORE - 49 of the 64 possible pairs are unrouted.

## Measured, mirror, 20 games each
    frontrun (front_run + opponent-plan proxy)  0.950   +147 bank
    flag_room_guard_on                          0.800   +623
    flag_clamp_sells_on                         0.650   +106
    control                                     0.500      0   <- self-check
    flag_front_run_on (no opponent plan)        0.500      0   <- layer is inert
    terminal_* (all 8 alternatives)             0.15-0.25
    flag_hand_align_off                         0.100  -33,610

`front_run` ships enabled-able but dead: nothing supplies `opponent_plan`, so the
layer never fires. Feeding it a proxy for the opponent's tape - our own current
route, on the argument that much of the ladder runs this same lineage and both
seats see the same shop draw - turns it on, and it wins 19 of 20.

## Tools
`arena.py` (league, engine-aware cache) · `promote.py` (gate + only thing that
submits) · `sweep/mirror_eval.py` (variant vs base) · `sweep/variants.py` (source
rewrites: router table, terminal route, settings, opponent-plan shim) ·
`sweep/pair_sweep.py` + `sweep/seed_index.py` (stratified router re-fit) ·
`sweep/launch.py` (shard a sweep across the Kaggle notebook quota) ·
`sweep/mine_tapes.py` (harvest opponent tapes from our own episodes)

## Next
1. Combination sweep of frontrun / room_guard / clamp_sells, then gate + submit.
2. Router re-fit from the Kaggle pair sweep (4 of 7 shards running; the 5th CPU
   slot and both GPU slots were refused as "batch session count reached").
3. `mine_tapes.py` on v10's episodes. Our own submission is the sensor: as the
   rating climbs we get matched against stronger agents and their full action
   streams arrive in our replays for free.

## Session log 2026-09-18 (rank 5472 -> 2458)

Four submissions, each gated against the previous champion:

| ver | change | mirror vs prior champion |
|---|---|---|
| v10 | adopt the Apache-2.0 shop-router lineage wholesale | 108-0 vs the public field |
| v11 | feed `front_run` an opponent-plan proxy (it ships dead) | 19-1 |
| v14 | widen the race from 4 glut products to all 8 | 47-3 |
| v16 | add FERTILIZER -- every tradable product | 47-3 |

The whole v11->v16 chain is one finding worked out: the chassis ships a
`front_run` layer that never fires because nothing supplies `opponent_plan`,
and once it fires, the product list it was handed is too narrow.

### Rejected, with the measurement
- **live opponent model** (read their harvest-ready yield off their public
  tiles instead of assuming they run our tape): 9-31 vs v11. front_run needs to
  know *when* they sell, not *whether they hold stock*; standing field yield is
  almost always non-zero, so the layer fired constantly and dumped early.
- **opponent-side lookahead** (3/6/12 steps): byte-identical results at every
  window. Structurally inert -- see below.
- **own-side window** (2/3/4/6/8/12): also byte-identical to control. So the
  binding term in `qty = min(our stock, their qty, own_next)` is neither
  opponent model nor tape plan -- it is **our actual holdings**. front_run
  already moves everything we have, every time it fires. The item list was the
  only real knob, which is why widening it was worth three submissions.
- **herd trim**: dropping SHEEP costs 65% of the bank, COW 30%, GOOSE 26%.
  Wool clearing $1/unit made sheep look unprofitable; the herd's value is the
  fertilizer chain, which a per-item revenue split cannot see.
- terminal routes (all 8), `min_sell_price` (6/15/30), `budget_guard` (3-2-45),
  `dead_stock`. With the race this wide, `sell_lead` and `terminal_liquidation`
  are now inert -- switching either off is byte-identical to control.

### Gate change, flagged
`promote.py` used to require the gauntlet's paired CI lower bound >= 0. That is
unsatisfiable on a metric where 11 of 12 opponents sit pinned at 1.000 and can
only move down: v14 scored delta exactly +0.000, CI [-0.008, +0.008], and the
old gate still rejected it. Replaced with a stated non-inferiority margin,
`GAUNTLET_TOLERANCE = 0.02` (~5 games in 240). It is not tuned to pass a
candidate -- it rejected v14 on the first re-run, which is why that run went to
22 seeds instead.

## 2026-09-21 - rebase onto the current public top base

The public meta moved while we were tuning a three-day-old base. Today's rebase
was worth more than every local improvement combined.

| ver | change | vs prior champion | live rating |
|---|---|---|---|
| v16 | (previous champion, old base) | - | 1775 |
| v17 | rebase to `haideptry` Master Hybrid + front_run + opponent proxy | old champion lost **0/64** to the bare new base | **2633.6** |
| v19 | widen the race to all nine tradable products | 39-0-11 | climbing |
| v21 | opponent proxy answers over a 2-step window | 23-12-5 | climbing |

**v17 = 2633.6 is the proven fallback.** Silver opened at 2631.5, so v17 is a
silver-level agent on the live ladder. It is preserved verbatim at
`candidates/v17_newbase_frontrun.py`; its diff from the public base is 30 lines
(8 comment, one `front_run` flip, one 17-line `_OppPlanProxy`). If v19 or v21
converge lower, resubmit v17 unchanged.

### Sequencing mistake worth not repeating
Kaggle keeps only the **latest two** submissions active. v17 was at 2614.9 and
still climbing toward the silver line when v21 was submitted, which displaced it
from the active pair. Both replacements are measurably stronger, but they restart
from ~600 and need hours. Let a converging submission finish crossing a threshold
before spending a slot that evicts it.

### What the front-run knobs do, and why the answer is base-specific
`front_run` moves `qty = min(our holdings, their planned qty, our own next-step
plan)`. Which term binds is a property of the BASE's tapes, not of the layer:

- old base: our holdings bound. Opponent window and own window were both
  provably inert (windows 3/6/12 byte-identical to control). Only the product
  list mattered, and widening it to all nine was worth 47-3.
- this base: the opponent window is worth 23-12-5 at any width >1, the own
  window is still inert, and the product list is worth 39-0-11.

Re-measure the width against each new base rather than carrying the answer over.

## 2026-09-28 - the mirror gate was wrong, and it cost a week

v19 and v21 each cleared all six promotion gates against the previous champion
with tight CIs over 50+ mirror games. Both converged about **1000 rating points
lower** on the live ladder:

| agent | front_run config | mirror vs prior champ | ladder |
|---|---|---|---|
| v17 | narrow: 4 products, 1-step window | (baseline) | **2621.9** |
| v19 | all nine products | 0.77, CI [+0.154,+0.365] | 1599.4 |
| v21 | all nine + 2-step opponent window | 0.69, CI [+0.087,+0.288] | 1558.3 |

Monotonic in how aggressively we front-run, and the mechanism explains it. The
`front_run` layer is driven by an opponent-plan proxy that assumes *the opponent
is running our own tape*. In a mirror that assumption is exactly true, so
front-running wins reliably. On the ladder it is usually false, and acting on a
wrong opponent model dumps product early at worse prices. **The mirror rewarded
exactly what the ladder punishes.**

The gauntlet could not have caught it either: every extractable public agent
loses ~100% to a current base, so it saturates. Net position: no local metric in
this project was ever validated against ladder outcome. The v11->v16 chain on the
older base is unverified for the same reason -- it was gated the same way.

**Cost.** Only the latest two submissions stay active and team score is the max
over them, so v17 was benched behind v19/v21 for a week. We sat at 1599.4
(rank 2281/10120, out of bronze) while holding a 2621.9 agent.

**Recovery (2026-09-28).** Resubmitted v17 twice from the `v17-silver-2633` tag,
plus one probe of the bare public base with `front_run` untouched -- the trend
predicts zero front-running may beat v17, and the max-over-slots rule makes that
probe free while the other slot holds the floor.

**Rule going forward:** do not promote on a local metric that has not been
checked against live ladder ratings on at least two submissions.

### Which local signals survived, and which did not

Not all local measurement failed - the distinction is whether the opponent in
the test resembles US or resembles the FIELD.

- **Cross-agent h2h transferred.** The rebase decision was made on a 64-0 local
  result against a genuinely different agent, and it gained +790 on the ladder.
  Ranking candidate BASES this way is still trustworthy.
- **Mirror (or near-mirror) h2h did not.** Our front_run proxy assumes the
  opponent runs our own tape, so any opponent that IS our own tape hands it a
  correct model for free. v19/v21 won those and lost ~1000 rating.

The trap: v17 differs from its base by 30 lines, so "v17 vs bare base" is a
near-mirror and scores 0.93 locally. That number says nothing about whether
front_run helps on the ladder; only the live v22 probe can answer it.

2026-09-28 ranking of this week's public bases (cross-agent, trustworthy):
v17 0.93 > haideptry Master Hybrid 0.73 > haideptry Shepherd's Ledger 0.33 >
guruprasaathas Master Engine v5 0.00. No rebase warranted.
