# Pokemon TCG AI Battle Challenge

Agent, evaluation harness and full experiment record for the Pokemon Company /
Kaggle *PTCG AI Battle Challenge* (Simulation division, competition slug
`pokemon-tcg-ai-battle`, 16 June to 17 August 2026).

The submission is a Python function that plays complete 60-card Pokemon TCG
games inside the organizers' `cabt` simulator. Agents are matched against each
other continuously and rated on a live ladder.

Best recorded public score in this workspace: **844.4** (Observable Meta Router
V3 lineage, Kaggle row `55056992`). Best score of the final agent line:
**794.3** (Grim Search Family v2). Bronze sat at 837.0 and silver at 911.3 when
the ladder was last sampled on 2026-08-14, at which point the active
submission was 764.8 for a rank of roughly 1159 out of 6834.

---

## 1. The problem

The entire agent is one function:

```python
def agent(obs_dict: dict) -> list[int]:
    return [chosen_option_index]
```

Every decision in a game arrives as a list of legal options: mulligan, place
the active Pokemon, attach energy, evolve, retreat, attack, play a trainer, and
the initial 60-card deck build (a single choice with `maxCount = 60`). The
engine never offers an illegal move, so the agent cannot lose to rules errors.
It can only pick badly.

What makes this hard is not the rules, it is the information structure:

- Hidden state: the opponent's hand, deck order and face-down prizes are not
  observable.
- Randomness: shuffles, draws and coin flips move a large share of outcomes.
- Branching: around 2,000 Standard-legal cards, and a turn is a sequence of
  interacting sub-decisions rather than one move.
- The deck itself is part of the policy. A strong policy on a weak list loses
  to a mediocre policy on a strong list.

That last point was the single most expensive lesson here. The first 45
experiments tuned policy while the agent played the engine's default Abomasnow
list. The 531.2 score was mostly the deck, not the policy.

---

## 2. Results

Ladder scores, in the order the submissions were made. The public score of an
unchanged submission drifts as the ladder re-matches it, so these are snapshots
and not stable measurements (see section 6).

| Date | Submission | Public score |
|---|---|---|
| 2026-07-15 | Heuristic v1, embedded deck | 452.1 |
| 2026-07-15 | v3 shallow determinized search | 531.2 |
| 2026-07-16 | RL net, 48 and 96 sims | 476.2 / 442.2 |
| 2026-08-04 | v3 search on a Garchomp list | 632.5 |
| late July | Observable Meta Router V3 (row `55056992`) | **844.4** |
| 2026-08-07 | Alakazam fork, one search-margin constant changed | 777.1 |
| 2026-08-12 | Grim Search Family v1 (row `55444198`) | 719.8 |
| 2026-08-13 | Ogerpon field07 candidate | ~684 to 697 |
| 2026-08-12 onward | Grim Search Family v2 | 794.3 (converged) |

Two things in that table are worth reading carefully.

The RL net scored *below* the hand-written heuristic it was meant to replace.
Neural policies were retried four more times later (imitation ranker, DAgger,
pairwise advantage, causal-correction doses) and never beat the frozen
hand-written base on a held-out panel.

The 844.4 row predates the final agent line and was never beaten by it. It is
reported here as the honest high-water mark rather than quietly dropped.

---

## 3. The final agent

`agent/grim_search_family_v2_package/` is the agent as submitted.

It is a frozen Grimmsnarl ex control policy with a narrow forward-search layer
bolted on top:

```
main.py                 composite entry point
  grim_base.py          frozen base policy (byte-identical to grim_candy_v2/main.py)
  grim_search_hybrid.py forward search + visible-family gate
```

The gate is the whole idea. Generic search made the agent worse everywhere it
was tried, so search is allowed to run only when a named opponent archetype is
actually visible on the board, and only when it beats the base policy's choice
by a configured board-value margin:

| Visible opponent family | Card IDs | Override margin |
|---|---|---|
| Garchomp | 341, 342, 379, 380, 381, 387 | 10,000 |
| Archaludon | 57, 169, 190, 666 | 20,000 |
| Router / Crustle | 344, 345, 607 | 20,000 |
| anything else | - | search disabled, exact base policy |

Search itself is deliberately small: single-choice MAIN decisions only, top-4
candidates from the domain policy, 2 determinizations, 20-step rollouts. The
rollout continuation uses a separate clean domain policy so the base policy's
stateful memory is never mutated by a counterfactual branch.

Measured against family v1 on common random numbers, both seats:

| Panel | v2 | v1 | discordant seeds | p |
|---|---|---|---|---|
| Gated Router v12, 1200 games | 900 | 892 | 10 / 2 | 0.0386 |
| Visible-field Router, 600 games | 510 | 503 | 7 / 0 | 0.0156 |
| Crustle-counter safety, 600 games | 467 | 467 | 0 / 0 | invariant |
| 12-family broad battery, 2400 games | 1997 | 1989 | - | - |

Every gain is small and no opponent family went negative. That is the shape of
a real but modest improvement, and it is also why the family was eventually
declared finished: eleven further variants moved the validated panel by 1.5
points total, which is inside control drift.

Other agent lineages kept here: `agent/fork/` (the public Alakazam fork whose
single changed constant reached 777.1), `agent/router_v5` through
`agent/router_v13_search` (the Great Tusk / Crustle router lineage that
produced 844.4), and `agent/current_candidates/` (packaged candidates).

---

## 4. The evaluation harness

`tools/crn/` is the part of this project with the longest useful life, and it
took longer to get right than the agent did.

**Common random numbers.** The stock engine draws fresh shuffles per game, so
at n = 120 to 200 almost all observed spread is deck luck. Forty times more
search, tuned weights and a working opponent model all came back inside the
noise band. `crn_export.cpp` exposes a seeded battle start; `paired_eval.py`
then plays both candidates over identical seeds in both seats and reports a
McNemar test on the seeds where they disagree. Discordant-seed counts are the
real signal, not win-rate differences.

**Opponent choice.** Three panels were built, and only the third predicts the
ladder:

| Harness | Opponents | Verdict |
|---|---|---|
| `arena.py` meta-* | our policy piloting mined field decks | saturated, useless above ~700 |
| `band_eval.py` | our policy piloting rating-band decks | measures beating our own impersonation of the field, cross-deck predictions are wrong |
| `public_panel.py` | 13 real published agents piloting their own decks | tracks the ladder |

`public_panel.py` was validated on three known ladder points: family_v2 at
65.4 percent local against 794.3, the fork at 65.1 percent against 777.1, and
the Ogerpon candidate at 50.0 percent against roughly 697. Correct order,
correct magnitude.

Two rules came out of that, both learned by getting them wrong first:

- Use the unweighted average. Weighting cells by observed field share predicted
  the Ogerpon candidate at +26 over family v2. The ladder said -100.
- Never run fewer than about ten opponents. With three opponents the fork read
  +24 over family v2. With thirteen, the true gap is +0.3.

`band_eval.py` mispredicting a cross-deck comparison cost one real submission
slot. It is kept in the repository because the failure is instructive, not
because it should be used.

---

## 5. What did not work

Each of these is a measured negative, not an untried idea. Numbers are in
`docs/experiment-log.md`.

- Imitation ranker and learned rankers, five separate attempts.
- Gated neural router over the hand-written base.
- Scaled search: 40x the simulation budget, inside the noise band.
- Belief-state determinization from a fitted opponent deck model.
- Generic search behind an archetype gate: decisively harmful even on the
  families it was gated to, partly because shared support-card IDs made narrow
  gates fire broadly. Alakazam 161 vs 179, p = 0.0021. Froslass 149 vs 179,
  p = 0.00003.
- Deck tech, 24 arms, best arm +0.8 against control drift of plus or minus 0.8.
- Gate margin sweeps, 6 arms, byte-identical output.
- MCTS and beam variants, 8 arms, byte-identical output, which is how the dead
  code below was found.
- Adopting a public agent claiming 1084.5: it scored 11 of 80 against our
  control's 40 of 80, seed-level 0 vs 29, p below 1e-6. Notebook titles are
  not evidence.

### The Ogerpon postmortem

An Ogerpon search candidate measured 83.7 percent band-weighted against the
shipped agent's 75.5 percent, and was submitted. It scored about 684.

The search had six stacked bugs and had never executed a single step: a raw
string passed to `search_begin`, an ApiResult versus SearchState API mismatch,
`your_deck=[673]*60`, `evaluate_state` missing its seat argument, an
eager-default `len(None)`, and candidates drawn from `choose()` which truncates
to `maxCount = 1`. Every one of them was swallowed by a bare `except` that fell
back to the heuristic. The "83.7 percent" was the fallback policy's score.

After fixing all six, with search actually running at beam 10 and a 69 percent
override rate, the same measurement dropped to 51.4 percent. The evaluator, not
the search, is the weak component.

Two habits came out of this and are now enforced in the harness: a
stock-versus-stock control run that must return exactly 0/0 discordant seeds,
and an override-rate counter so a silently inactive component cannot pass as an
improvement. A related trap worth naming: `option.cardId` is `None` for
PLAY and EVOLVE options, so card-specific policies that read it are silently
generic. Cards must be resolved through `option.index` into `HAND`.

---

## 6. Ladder noise

The displayed public score of an unchanged submission moves substantially. The
RL-net submission was observed between 432.5 and 500.9 across 43 snapshots of
the same row, a net drift of -52.1 points with no change to the code.

Practical consequences, all of which cost something before they were adopted:

- Only the two most recent submissions stay active, and the reported score is
  the max of the active pair. A failing slot costs nothing, but it also cannot
  be diagnosed.
- Submissions need 5 to 10 hours of spacing to converge. Stacking them means
  reading pre-convergence numbers.
- Any single submission's score is one sample. Real candidates were submitted
  twice.
- A local gain must clear the ladder noise floor to be worth a slot. The
  working threshold here was roughly 80 to 90 ladder points.

---

## 7. Layout

```
agent/
  grim_search_family_v2_package/  final agent as submitted
  grim_candy_v2/                  frozen base policy (family v2's grim_base.py)
  fork/                           public Alakazam fork, 777.1 line, plus the top-20 field decks
  router_v5 .. router_v13_search  Great Tusk / Crustle router lineage, 844.4 line
  current_candidates/             packaged submission candidates
  majkel_ogerpon_domain/          Ogerpon candidate, see the postmortem
  *.py                            policy modules: domain, search, hybrid, beliefs, deck tools
  field_decks*.json               field composition mined from replays, by rating band

tools/
  arena.py                        competitor registry, every named agent resolves here
  crn/                            the CRN evaluation harness
    paired_eval.py                seeded paired A/B with McNemar
    public_panel.py               13 real published agents, the only ladder-predictive panel
    band_eval.py                  rating-band panel, kept as a documented failure
    clean_battery.py              generic A/B battery with a control arm
    crn_export.cpp                seeded battle-start shim over the engine
    determinism_scan.py           which opponents reproduce run to run
  rank/                           dataset extraction and ranker training for the neural attempts
  build_*.py                      submission packagers
  build_kaggle_*.py               generators for the remote Kaggle GPU/CPU experiment kernels

docs/
  experiment-log.md               chronological state of play, every measured negative
  experiments-2026-08-04.md       the detailed earlier log, 1100+ lines with raw numbers

references/top_rankers/decks.py   public deck lists with per-deck provenance
```

---

## 8. Running it locally

The engine is not vendored here (see section 9), so it has to be installed
first.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt      # kaggle-environments==1.30.1
```

Copy the simulator package the competition ships (`cg/`, containing `api.py`,
`game.py`, `sim.py`, `utils.py` and the `libcg` binary for your platform) into
whichever agent package you want to run, next to its `main.py`.

The CRN harness additionally needs `libcrn.so`, built from `crn_export.cpp`
against the engine's C++ source:

```bash
c++ -O2 -shared -fPIC -o tools/crn/libcrn.so tools/crn/crn_export.cpp <engine sources>
```

Then, for a paired comparison:

```bash
.venv/bin/python tools/crn/paired_eval.py grim-candy-v2 router-v12 \
    --opponent meta-alakazam --seeds 200
```

and for the ladder-predictive panel:

```bash
PP_SUBJECTS=family_v2_base PP_SEEDS=40 PP_WORKERS=16 \
    .venv/bin/python tools/crn/public_panel.py
```

`public_panel.py` resolves its opponents from `references/`, which holds
downloaded published agents and is not redistributed here. Populate it from the
competition's public notebooks, or point the loader elsewhere. Each cell runs in
its own process on purpose: the engine aborts at C++ level with
`buffer full. capacity:7` when too many battles share one interpreter.

---

## 9. Provenance

This folder mixes original work with adopted public material. The split:

**Written here:** the forward-search and visible-family gate layer
(`grim_search_hybrid.py`), the entire `tools/crn/` harness including the seeded
engine shim, the domain and hybrid policies, the router lineage, the field
mining and deck tooling, and all documentation.

**Adopted from public Kaggle notebooks:** the Grimmsnarl ex control policy stack
that `grim_base.py` and `grim_candy_v2/` contain (`policies/`, `experts/`,
`strategic_policy.py`, `matchup_router.py`, the guard modules and the packaged
`models/`), the Alakazam agent in `agent/fork/fork_main.py`, and the deck lists
in `references/top_rankers/decks.py` and `agent/fork/top20_decks/`, the Ogerpon
policy under `agent/majkel_ogerpon_domain/`, and the notebook-derived packages
under `agent/current_candidates/`. These are included so the submitted agent is
complete and auditable. They are their
authors' work, published in that competition's public notebooks, and are not
claimed here.

**Not included:** the organizers' `cg` simulator package and its `libcg`
binaries, the downloaded corpus of other competitors' agents under `references/`
that the public panel plays against, trained checkpoints (`*.pt`, `*.pth`), and
the raw replay and scratch data. All of it is regenerable or downloadable, and
none of it belongs in version control. The `EN_Card_Data.csv` inside the agent
packages is kept because the packaged policy reads it at load time.

The working rule throughout the project was that public deck lists are fair to
adopt and public policy code is not, since the Strategy division is judged on
originality. Both fork lineages here were used as controls and as measured
baselines. The 794.3 line ships original changes on an adopted base, which is
disclosed rather than presented as original.
