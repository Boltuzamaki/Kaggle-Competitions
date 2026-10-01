# Kaggriculture

Kaggle agent competition, July to September 2026. Two players each run a farm on
a 10x10 grid for 30 in-game days (720 turns), buying seeds and livestock,
planting, watering, harvesting, hiring hands, and selling into a market that both
players share. Most coins at the end wins. The ladder scores win/loss/tie only,
so bank margin is worth nothing except as a diagnostic.

Final: rank 2765 of 10246, score 1362.4. Bronze needed 1836.3. No medal.

## What happened

The agent here went through two distinct phases, and the second one is the
interesting part because it is mostly a record of being wrong.

### Phase 1: our own planner (May to late August)

`agents/v9_capacity_farmer.py` is a hand-written planner: market-capacity-derived
herd sizing, a greedy router for the farmer and hired hands, impact-ordered sell
scheduling, and land purchase timed off a board-occupancy gate. It peaked at 778
on the public ladder in early August and was submitted through nine versions.

Then it collapsed to 645.9, rank 5472 of 9415.

The cause was an engine change. `kaggle-environments` 1.32.7, released 15 August,
gave CARROT, TOMATO and EGG a new `hinge` price curve on the scarcity side: price
stays flat until market inventory falls past a threshold, then climbs
quadratically. CARROT's scarcity premium went from 0.20 to 1.00 of base. The old
premium goods were untouched and still crash to the $1 floor on a glut:

| market inventory vs I0 | -900 | -450 | 0 | +100 | +200 |
|---|---|---|---|---|---|
| TOMATO | 2520 | 414 | 60 | 35 | 24 |
| CARROT | 385 | 70 | 35 | 23 | 19 |
| EGG | 573 | 97 | 50 | 42 | 41 |
| MELON | 310 | 304 | 250 | 150 | 1 |
| WOOL / MILK / STRAWBERRY | 250-420 | 250-340 | 160-200 | 1 | 1 |

v9 was a melon, wool, milk and strawberry farm. It was built for exactly the half
of the board that stopped paying. Worse, the local virtualenv was still pinned at
1.32.6, so every sweep run after 15 August was tuned against a game that no
longer existed.

### Phase 2: adopting the public lineage (September)

By September a shared Apache-2.0 agent lineage had formed in the competition's
public notebooks, passed between a dozen authors. Every strong public agent was a
variant of the same chassis: 13 precomputed 719-step route tapes, a router that
picks a tape from the day-6 shop draw, and reactive repair layers on top.

v9 lost 108 games out of 108 against the ten strongest public agents. So the
agent was rebased onto that lineage. See `submitted/ATTRIBUTION.md`; the
submitted files are 99.7% other people's code.

The one original finding: every top-tier public agent ships the chassis
`front_run` layer with `front_run: False`, and the call site guards on an
`opponent_plan` argument that nothing ever supplies. The layer is dead code as
published. Five independent public agents were checked and all five had it
dormant. Switching it on and feeding it a proxy for the opponent's tape is about
20 lines, and it won against every base it was tried on.

Rebasing was worth about +790 rating. The 20 lines were worth roughly +50.

## Results by submission

| version | change | ladder |
|---|---|---|
| v9 | our own planner, nine iterations | 778 peak, 645.9 after the engine change |
| v10 | adopt the public `aurax7` shop-router lineage | 1531.3 |
| v11 | wake the dormant `front_run` layer | 1401.8 |
| v14 | widen the front-run race to all eight products | 1763.0 |
| v16 | add FERTILIZER to the race | 1781.4 |
| v17 | rebase to `haideptry` Master Hybrid, narrow race | 1491.4 |
| v19 | widen to all nine products | 932.3 |
| v21 | plus a 2-step opponent window | 1019.2 |
| v23 | rebase to `ahmedberatozer` v54, unmodified | 1347.2 |
| v25 | v54 plus the full front-run stack | 1362.4 |

The ratings are not comparable across rows. Kaggle keeps only the latest two
submissions active and a deactivated submission freezes at whatever the field
looked like at the time, so an old high number can be an artefact. This caused a
real mistake, described below.

## What the harness does

`arena.py` plays agents against each other. Matchups are not symmetric because
the market is shared, so every pairing is played in both seats on the same seed,
and results are cached by file content hash so changing one agent only replays
that agent's games.

`promote.py` is the only thing allowed to submit. A candidate must beat the
champion head to head with a paired bootstrap CI excluding a coin flip, must not
regress against the opponent pool, must have no errored game, and must stay
inside the per-step time budget.

`sweep/` holds the rest. `variants.py` builds candidate agents by rewriting the
base source: flipping chassis settings, replacing the router table, swapping the
terminal route, widening the front-run product list, installing the opponent-plan
proxy. `mirror_eval.py` and `pair_sweep.py` run those variants. `launch.py` shards
a sweep across Kaggle notebook sessions. `mine_tapes.py` harvests opponent action
streams from our own episode replays, since each replay contains both players'
full action history.

`analysis/` contains the price-curve and demand arithmetic the strategy was
reasoned from.

`rl/` is an abandoned recurrent self-play experiment: a 24-channel board encoder,
GRU state, masked action scorer, autoregressive market head. It was never trained
to the point of beating the heuristic agent and is included only for the record.

## Three things that went wrong, and what they cost

**An unseeded harness.** `arena.py` passed `seed` into the environment
configuration; the sweep scripts written later did not. "Paired seeds, both seats"
paired nothing for several runs. Caught and fixed, but it invalidated earlier
sweep output.

**A router re-fit built on a bad index.** The plan was to re-fit the shop-pair to
route table, since five of the shipped table's fifteen entries name route tapes
that were never built and 49 of the 64 possible shop pairs are unrouted. Seeds
were indexed by their day-6 shop pair using fast two-PASS-agent games. That index
matched real games 0 times out of 7. The environment RNG is shared, and
weed-spawn checks consume draws in proportion to each player's empty tiles, so
the shop sequence depends on what the agents actually do. Every router variant
targeted a shop pair that never occurred, which is why 3960 Kaggle-hours of games
came back as exact ties. Fixable by bucketing on the observed pair instead of a
predicted one, but it was not fixed before the deadline.

**Gating on the wrong opponent.** This is the expensive one. Candidates were
promoted on mirror matches, meaning candidate against the current champion. Three
versions passed that gate with tight confidence intervals over 50 or more games
and then ranked below their predecessor on the ladder. The mechanism is specific:
the front-run layer is driven by a proxy that assumes the opponent is running our
own tape. In a mirror that assumption is exactly true, so front-running wins
reliably. Against the real field it is usually false, and acting on a wrong
opponent model sells product early at worse prices.

Cross-agent evaluation, meaning the candidate against a panel of genuinely
different agents, did track the ladder. It predicted the rebase correctly. The
distinction is whether the test opponent resembles us or resembles the field, and
it took until the last two days to see it.

There was also a self-inflicted error on top of that. v17 showed 2621.9 while two
newer agents showed about 1580, so v17 looked a thousand points better and was
rolled back to. That 2621.9 was frozen from a week earlier against a weaker
field. Resubmitting v17 fresh converged to 1491.4, below both. The rollback made
the agent weaker and burned a day.

## Why the ceiling was where it was

Our best agents saturate around 1400 to 1600. The authors of the notebooks they
are built on sit at 2045 to 2345 with agents they do not publish. The `aurax7`
notebook converged to 1531 on the ladder while its author scored 2633, a
difference of about 1100 points between what people publish and what they run.

Adopting public work closes the gap to the published frontier quickly and then
stops. Getting past it needed something the public notebooks do not contain,
which for this competition meant either mining the top teams' replay archive for
better route tapes or writing a genuinely better planner. Neither was done in
time.

## Running it

```bash
python -m venv .venv && .venv/bin/pip install -U kaggle-environments
.venv/bin/python arena.py roundrobin agents/v9_capacity_farmer.py submitted/v17_newbase_frontrun.py --seeds 8
.venv/bin/python promote.py --candidate submitted/v25_abo_fr_all_clamp.py --seeds 20
```

`promote.py` expects a `champion/main.py` and a `gauntlet/` directory of opponent
agents. Neither is committed: the gauntlet was other authors' public notebooks
pulled locally for benchmarking. `extract_gauntlet.py` rebuilds it from notebooks
you download yourself.
