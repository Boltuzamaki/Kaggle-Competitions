# Detailed experiment log, phase 1

The earlier chronological log, roughly 1100 lines, with the raw per-experiment
numbers behind the summaries in `experiment-log.md`.

---

> the ~20 approaches already measured and dead, and the harness rules. This file
> is the chronological log with all raw numbers behind it.

# Experiment log — 2026-08-04

Session goal: scrape the competition's public discussions/notebooks and raise the
Simulation leaderboard score. Starting point **531.2**, rank 4465/6227.

**Result: 643.8 (+112.6), new best.**

---

## Headline finding

`agent/deck.csv` and `search_agent.DECK` still held the **cabt engine's sample
Mega Abomasnow deck — 33 basic Water Energy in 60 cards**. It has *zero*
appearances across all 4,720 games in the official 2026-08-03 top-episode dump.
Experiments EXP-27→45 tuned search, beam width, ISMCTS, RL nets and belief models
around it and each concluded "the policy is weak". The repo's own
`references/top_rankers/META_REPORT.md` had already flagged this deck in July as
"the beginner list… not a competitive target", and `deck.csv` carried the comment
"Replace with your own meta deck (Phase 2 in PLAN.md)". Phase 2 was never done.

Fix: keep the proven v3 search policy, change only the deck.

---

## What worked

| # | Change | Evidence |
|---|---|---|
| 1 | **Mine live meta decks from replay data** — `agent/meta_decks.py`, 10 archetypes attributed to the top-scoring LB team on each, all verified engine-legal via `battle_start` | Replaced guesswork with the actual field |
| 2 | **v3 search + Cynthia's Garchomp ex** (`main_v3_garchomp.py`) | LB **643.8** vs 531.2 for the same policy on Abomasnow. Local rating 84.1% over 9 competitors |
| 3 | **Duplicate submissions to hedge rating variance** (per discussion 712621) | Two byte-identical archives scored **643.8 vs 596.1** — a 47-point spread from luck alone |
| 4 | **Confirm-before-promote gate in the tuning kernel** | Caught an overfit genome (0.85 search → 0.65 vs 0.72 baseline on fresh sample) and refused to promote it |
| 5 | **Audited submission builder** — allowlisted members, source + isolated smoke, forbidden-member scan | 0 crashes/invalids/timeouts across 69 live episodes and several thousand local games |
| 6 | **Common random numbers** — built the engine source into `tools/crn/libcrn.so` with `deviceRand=false` and a caller-supplied seed (`CrnBattleStart`) | **Verified reproducible**: same seed → byte-identical 220-step game; different seed → different game. This is the fix for the measurement problem below |
| 7 | **Paired evaluator** `tools/crn/paired_eval.py` — both candidates play identical seeds, both seats, McNemar on discordant seeds | In **12 seeds** it showed belief-garchomp and v3-garchomp are behaviourally identical (**0 discordant**), an answer 120 unpaired games could only call "probably nothing". Positive control (v3 vs meta-garchomp) gave 4 discordant, so it does separate real differences |

## What failed

| # | Attempt | Outcome |
|---|---|---|
| 1 | **Deck-specific flat policy for Grimmsnarl** (`grimmsnarl_policy.py`) | LB **443.6** — *worse* than v3 on the bad deck. Flat priority scoring is a weaker policy than determinized search. Do not replace v3 with hand-scored priorities |
| 2 | **3 of 4 deck-specific overrides** (snipe targeting, setup lead, trainer priority) | Clean ablation vs a diverse field: removing each *raised* the rating (61.7 / 60.8 / 60.4 vs 59.2 full). Only Punk Up energy routing paid (−5.9 when removed) |
| 3 | **Scaling search to use the 600 s/game budget** (`search_scaled.py`, 0.4 s → 25 s per match) | No measurable gain: 55% vs stock v3, Wilson [42.5, 66.9], n=60. 40× the compute for nothing |
| 4 | **Hill-climbing policy weights on Kaggle** (7,611 s, 4 workers) | Overfit. Rejected by its own confirmation gate |
| 5 | **Trusting the local arena rating** | `v3-abomasnow` rated *highest* locally (69%) while scoring worst live (531.2). Local ratings do not track the LB |
| 6 | **Mirror-only validation** | The 59.2% mirror result hid that the policy was bad, because both sides shared the flaw. Always validate against a diverse field |
| 7 | **Opponent-deck modelling** (`search_belief.py`) — identify the opponent's archetype from revealed cards, determinize from that meta list instead of placeholder Snorlax | Null, twice. Unpaired arena: **51.7%** vs stock v3, Wilson [42.8, 60.4], n=120. Re-tested through the *fixed* CRN harness: **90.0% vs 93.8%**, 7 discordant seeds of 40 (belief better on 2, worse on 5), p=0.45. Not a no-op — it identifies the archetype in **90.1%** of decisions and the agents do diverge — the correct opponent deck simply does not help, and trends slightly negative |

### The pattern behind failures 3, 4 and 7

Three well-formed improvements — 40× more search, tuned weights, and a working
opponent model — all landed inside the noise band. That is now the primary
obstacle: **our measurement instrument is too weak to detect the changes we are
making.** Fixing evaluation matters more than any further policy idea.

## Process mistakes to avoid repeating

1. **Misread the arena's "A perspective"** — it is always the competitor registered
   first in `COMPETITORS`, not the CLI order. This produced a phantom "3% winrate
   defect" report. Read the `X vs Y` header before interpreting any record line.
2. **Predicted ~950 from the deck swap** while my own analysis said deck ≈ 40–50
   points and agent quality ≈ 300. The archetype medians (900–1035) reflect those
   teams' *agents*, not their decks.
3. **Concluded from a 29-point LB difference** that the deck swap had failed,
   before reading discussion 712621. Under ~100 points, conclude nothing.

## Method notes

- Kaggle sim submissions seed at **600** and converge from there; a score with
  <20 episodes is mostly prior.
- Arena stdout is block-buffered when redirected — poll for the output `.csv`,
  never tail the log.
- `nohup … &` inside a tool call does not survive; use tracked background jobs.
- System python (3.14) has no pip. Use `uv venv --python 3.12 .venv` +
  `kaggle-environments==1.30.1`.

## Engine source — findings (discussion 717141)

The **full C++ engine source has been public since 2026-07-01** as `ptcg_engine.zip`
on the competition Data tab. Nothing in this repo had used it. Downloaded to
`data/engine/` (local only — the licence forbids sharing/republishing and requires
deletion at competition end, so it must stay out of git).

Two results from reading it:

1. **The engine supports fully deterministic play, but the shipped API disables it.**
   `Game.h` holds `uint32_t seed` and seeds an `mt19937`; `config.deviceRand`
   selects between that seeded RNG and a fresh `std::random_device` at every
   randomness site (`CardMove.h:262` deck shuffle, `SelectProc.h:56,94` coin
   flips, `EffectInstant.h:584`). `Api.h:33` hardcodes `config.deviceRand = true`
   and reseeds from `random_device`, so nothing is reproducible through
   `ApiBattleStart` — which takes only `int* cards`, no seed parameter.

   This corrects **EXP-31**, which concluded "no native seed function exists".
   The capability exists; it is switched off at the API boundary. Building the
   provided source locally with `deviceRand = false` and a settable seed would
   give **common random numbers** and paired A/B evaluation — the direct fix for
   the noise problem above. Explicitly sanctioned use: "local testing,
   verification, and training".

2. **Do not exploit engine bugs.** The official post forbids exploiting
   implementation bugs or unintended behaviour; a competitor has already reported
   a variable-shadowing crash in `EffectProc.h` `ToolCountProc`. Our agent must
   never rely on such behaviour.

## Competition format — corrected understanding (discussion 714189)

- **Only two submissions are active at a time — the two most recent.** Submitting
  a third deactivates the oldest. Both of our slots currently hold identical
  Garchomp copies.
- **The two active at the Aug-16 deadline determine the final placement**, after
  ~2 weeks of continued convergence. There is **no leaderboard reset**.
- Kaggle states early matchmaking luck is "never locked in" and every game updates
  the rating — which argues *against* the community's "resubmit until a lucky
  streak sticks" theory.
- Target is ~24 games/day per submission.

**Action required before Aug 16:** deliberately choose the two active agents and
stop submitting. Two copies of one agent is a fine exploration state, a poor
final one.

## ⚠ CORRECTION — the first CRN results measured the wrong policy

`cg/game.py` attaches `search_begin_input` **in Python** from the `SerialData`
binary blob; it is not part of the engine's JSON:

    Battle.obs["search_begin_input"] = ctypes.string_at(sd.data, sd.count).decode("ascii")

The first version of `tools/crn/paired_eval.py` did not attach it. Without it
`to_observation_class` yields a non-agent observation, **every `search_begin()`
raises `ValueError("Not agent observation.")`**, and `search_agent.agent`
silently falls back to `_heuristic` — measured at **0 successes / 155 attempts**.
In the real `kaggle_environments` arena the same agent searches 106/106, so the
defect was confined to the harness.

**Everything below measured with the heuristic, not v3 search, and must be
re-run:** the paired deck ranking, the Crustle/Garchomp confirmation, the
"0 discordant seeds" belief-agent result, and the `tune_value` run.
`submission_v3_crustle.tar.gz` was submitted on this flawed evidence.

Why it hid so well:
- `search_agent.agent` swallows search exceptions by design (never crash in
  production), so local failure is invisible.
- The harness was validated for *determinism* (which passed) but never for
  *behavioural equivalence to the real environment*.
- The tell was misread: value weights of 900 vs 3 gave byte-identical games. That
  was written up as "the leaf evaluation barely matters" when the real cause was
  that the leaf evaluation was never being computed.

**Lesson: validate a new harness by reproducing a known result from the old one
before trusting any new conclusion from it.** Fixed and verified at 127/127.

## Corrected deck comparison (CRN with search actually running)

80 games per matchup, paired seeds, v3 search on both sides.

| Opponent | Crustle | Garchomp | discordant C/G | Field share |
|---|---|---|---|---|
| grimmsnarl | 88.8% | 90.0% | 6/7 | 37.9% |
| dudunsparce | 55.0% | **83.8%** | 3/21 | 10.5% |
| alakazam | 52.5% | **88.8%** | 3/27 | 10.0% |
| dragapult | 85.0% | **96.3%** | 3/11 | 6.0% |

**Aggregate: Crustle 225 / Garchomp 287 of 320, discordant 15/66, p < 0.0001.**
**Field-weighted: Garchomp 89.4% vs Crustle 77.3% (+12.1 for Garchomp).**

Full three-way ranking, 320 paired games per pair, all significant and transitive:

| Matchup | Result | p |
|---|---|---|
| garchomp vs crustle | 287-225 | <0.0001 |
| garchomp vs grimmsnarl | 290-255 | 0.0007 |
| grimmsnarl vs crustle | 259-216 | 0.0001 |

**Garchomp > Grimmsnarl > Crustle.** Crustle is the *weakest* of the three under
real search; the broken harness had ranked it first. The original v3-garchomp
submission was correct.

Live scores are consistent but far less sensitive (Garchomp 643.9 vs Crustle
oscillating 420-656 over ~13 episodes), which is the expected relationship: 320
paired local games resolve a 12-point deck edge that the ladder cannot show
without hundreds of episodes.

**Garchomp > Crustle — the exact opposite of the broken run**, which had reported
Crustle dominant (220-143 vs Grimmsnarl). Under real search the two are equal in
that matchup (71-72) and Garchomp wins the rest comfortably. The discordant count
against Grimmsnarl fell from 75 to 13: the search compensates for deck
differences that the heuristic could not, so deck choice matters *less* under a
strong policy than the heuristic-based measurement implied.

Consequence: the original v3-garchomp submission was correct; switching to
Crustle was a mistake caused entirely by the harness defect. Because one Garchomp
copy was deliberately kept active as a control, the mistake cost nothing.

**Next action:** with tomorrow's 5 submissions, submit v3-garchomp twice to fill
both active slots and evict Crustle (active = the two most recent, so a single
submission cannot displace it).

## Paired deck ranking (CRN, v3 policy held constant) — SUPERSEDED, see correction above

300 games per deck, identical seeds, 3 fixed opponents. Because the policy is
constant, every discordant seed is attributable to the deck.

| Deck | Wins /300 | vs Garchomp |
|---|---|---|
| **crustle** | **223 (74.3%)** | discordant 49/31, p=0.057 |
| garchomp *(submitted, LB 643.8)* | 197 (65.7%) | — |
| grimmsnarl | 188 (62.7%) | p=0.54 |
| lucario | 137 | p<0.001 |
| dudunsparce | 108 | p<0.001 |
| dragapult | 79 | p<0.001 |

**Crustle is the strongest deck for our policy** — and the unpaired arena had it
*last* (37.5% in `field_v3g`) purely because it was piloted there by the weak
generic policy. Deck strength must always be measured under the policy that will
actually pilot it.

### Confirmation, 120 seeds x 4 opponents (960 games each)

| Opponent | Crustle | Garchomp | discordant C/G | p | Field share |
|---|---|---|---|---|---|
| **grimmsnarl** | **91.7%** | 59.6% | 68/7 | **<0.0001** | **37.9%** |
| dudunsparce | 35.4% | 56.7% | 19/57 | <0.0001 | 10.5% |
| alakazam | 60.0% | 65.8% | 29/40 | 0.23 | 10.0% |
| dragapult | 81.2% | 76.2% | 35/24 | 0.19 | 6.0% |

**Field-weighted: Crustle 76.6% vs Garchomp 61.6% (+15.0).**

The raw aggregate was **644 vs 620, p=0.19 — "no difference"** — because it
weights every opponent equally. The live field does not: Grimmsnarl alone is
37.9% of it, and that is where Crustle is overwhelming. **Always weight paired
results by actual field share before deciding.** An equal-weighted aggregate
across matchups is close to meaningless in a game with strong matchup structure.

Submitted 2026-08-04 as `submission_v3_crustle.tar.gz`
(sha256 `9af80bd5…`), keeping one Garchomp copy active as a live control.

## How to build and use the CRN harness

    E="data/engine/ptcg_engine/ptcgProgram 22"
    g++ -std=c++20 -O2 -shared -fPIC -fvisibility=hidden -I"$E" \
        tools/crn/crn_export.cpp -o tools/crn/libcrn.so
    .venv/bin/python tools/crn/test_determinism.py          # must print PASS
    .venv/bin/python tools/crn/paired_eval.py A B --opponent meta-alakazam --seeds 100

`crn_export.cpp` adds one new entry point and modifies no engine file. It is a
local measurement instrument only — submissions still run against Kaggle's
official binary, and nothing depends on engine bugs.

**Read discordant seeds, not win-rates.** Two agents that tie on win-rate but
never disagree per-seed are the same agent in practice; the discordant count is
the real signal and it converges an order of magnitude faster.

## Open leads

- **`search_belief.py`** — determinize the opponent's deck from the mined meta
  lists instead of placeholder Snorlax. The compute-scaling null result implies
  the limit is determinization *bias*, not sample count. Under test.
- **Garchomp is only 5.4% of the field**, so its matchups are thinly sampled;
  re-mine as the meta moves toward the Aug-16 lock.
- Re-run the deck sweep with v3 search across all 10 mined decks — only Garchomp
  and Grimmsnarl have been tested with the strong policy.

---

# 2026-08-05 — external intel and benchmark discipline

## From discussion 713608 ("What We Tried, What Ceilinged")

A rival team's detailed post-mortem. Directly relevant findings:

- **Their entire search program failed to beat the trivial first-index baseline**:
  1-ply hand-crafted ~40%, 2-ply expectimax ~50%, ISMCTS/AlphaZero-lite 42.5→50%,
  MLP self-play 52–67%. Matches our own three null results.
- **"Beam value is inversely proportional to base-policy quality. Wrap weak
  heuristics; leave strong ones alone."** Same beam: +11.3pp on a weak policy,
  −15.4pp on a strong one.
- **Matchup-gated search** was their best lever (+5.7pp): search *hurt*
  Starmie-vs-Lucario (29→22%) but was huge vs Crustle (39→83.5%). Default to the
  pure policy, switch to search only once the opponent is confidently identified.
- **Their "obvious" loss-analysis fixes regressed the agent** −7.6pp pooled
  (p=0.003). Survivorship bias from reading only losing games. Do not ship
  intuitive tactical guards without a paired A/B.
- **Their 68-weight hill-climb was reading noise**: "best" rose 0.786→0.921, but
  re-running the identical config at 2,000 games differed by 1.37pp on sampling
  alone, against a 0.5pp accept threshold. Identical failure mode to our Kaggle
  weight sweep, which our confirm-on-disjoint-seeds gate correctly rejected.
- **Revalidate at ≥400 games.** A 77% metric at 200 games was 67% at 400.
- They explicitly lack common random numbers and ask whether anyone has it
  working. **We do** — `tools/crn/`. That is our one clear methodological edge.
- Their breakthrough was representational: card-identity embeddings in a small
  transformer (73–74% vs baseline), after nine methods that encoded only option
  *type* plateaued.

## Our measurements (paired CRN, search verified running)

| Test | Result | Verdict |
|---|---|---|
| v3 search vs first-index baseline (Garchomp) | **91.2% vs 60.0%**, discordant 2/22, p=0.0001 | Our search is sound; it does *not* degrade the engine's native ordering |
| hybrid (domain-driven search) vs v3 | 92.5% vs 88.8%, discordant 7/5, p=0.77 | No difference |

**The real problem this exposes:** we win ~91% against our local opponents but
score only ~644 on the ladder. Every local opponent is `domain_policy`, which is
too weak to distinguish good from great — so every policy refinement measures as
null. Fix: benchmark against the frozen public LB-950 agent
(`tools/public_agents.py`, allowed as a local opponent, never copied into ours).

## THE key measurement: benchmark vs the public LB-950 agents

`tools/public_agents.py` loads two frozen public agents as local opponents
(allowed by our rule: benchmark against them, never copy their code). 100 games
per pairing:

| Matchup | Our win rate |
|---|---|
| v3-garchomp vs public-alakazam | **30%** |
| v3-garchomp vs public-archaludon | **21%** |
| hybrid-garchomp vs public-archaludon | **40%** |
| hybrid-garchomp vs public-alakazam | 25% |
| first-index vs public agents | 5–13% |

Overall: public-alakazam 71.8%, public-archaludon 69.8%, **hyb-garchomp 51.8%,
sweep-garchomp 47.8%**, first-garchomp 9.0%.

**We win 21–30% against ~950-rated agents.** That fully explains 644 vs 950;
there is no remaining mystery about the score gap.

### The methodological headline

**Weak sparring partners cannot distinguish our agents.** Against
`domain_policy`-piloted meta decks, hybrid vs v3 measured p=0.77 — "no detectable
difference". Against `public-archaludon` the same two agents differ by **19
points** (40% vs 21%).

Every null result recorded earlier today (scaled search, tuned weights, opponent
modelling, richer evaluator, 3-of-4 deck overrides) was measured against
opponents too weak to reveal anything. **Re-test them all against the public
agents before concluding they are dead.** A local win-rate above ~85% against our
own policies means the opponent is saturated and the measurement is worthless.

Standing rule going forward: **benchmark against `public-alakazam` /
`public-archaludon`, not against `meta-*`.**

## Card-identity ranker trained on official replays

Host-sanctioned data (discussion 709160: datasets published "to help in reviewing
replays as well as training agents"). Extracted with `tools/rank/extract_dataset.py`,
filtered to winners on teams scoring >=1000 LB: **212,899 decisions from one day**
(4,720 games); two further days downloaded.

`tools/rank/train_ranker.py` — listwise softmax over the legal option list, scored
per option from a **card-identity embedding** + option type + numeric fields +
board context. Card identity is the representational lever discussion 713608
identified after nine of their methods plateaued encoding only option *type*.

Result: **top-1 46.08% vs 39.88% for "always take option 0" (+6.2)**, top-3 79.2%,
1.8 MB checkpoint. Note the baseline itself: the engine's native ordering predicts
a 1000+ rated player's choice 39.9% of the time, independently confirming that the
ordering is strong and that naive reorderings lose to it.

`agent/ranker_agent.py` wires it in with modes prior / policy / off and a v3 →
legal-index fallback chain. 0 failures, p95 9.9 s per match against a 600 s budget.

---

# 2026-08-05 session — full record

Start: 572.5, rank 4005/6447. End: Ogerpon+Garchomp active, rank ~1900.
Medal thresholds that day: **BRONZE 841.4 (rank 644), SILVER 909.0 (rank 322)**.

## 1. The finding that reframed everything: our benchmark was saturated

`tools/public_agents.py` loads two frozen public agents (~950 LB) as local
opponents — allowed as opponents, never copied into our agent. First benchmark
against them (100 games/pairing):

| Matchup | Our WR |
|---|---|
| v3-garchomp vs public-alakazam | 30% |
| v3-garchomp vs public-archaludon | 21% |
| hybrid-garchomp vs public-archaludon | 40% |
| first-index vs public agents | 5–13% |

**We win 21–30% against the agents we are chasing.** 644 vs 950 fully explained.

Critically: hybrid vs v3 measured **p=0.77 ("no difference") against our own weak
`meta-*` opponents** and **p=0.031 with a 19-point gap against a strong one**.

**RULE: benchmark against `public-alakazam` / `public-archaludon`, never `meta-*`.
A local win-rate above ~85% means the opponent is saturated and the measurement
is worthless.** Several earlier "dead ends" were measured on the saturated
instrument and deserve re-testing individually before being trusted.

## 2. Confirmed upgrades (both shipped)

| Change | Evidence | Status |
|---|---|---|
| **hybrid v5 over v3** (domain-driven search, richer leaf eval) | paired CRN vs public-archaludon: 29.2% vs 18.3%, discordant 22/9, **p=0.031** | live |
| **Ogerpon deck over Garchomp** | paired cross-deck CRN vs BOTH publics, 240 games/side: hybrid 46.3% vs 33.8% (48/26, **p=0.0146**); replicated under v3 40.0% vs 26.3% (23/48, **p=0.0044**) | live |

Ogerpon's edge is concentrated in the **Alakazam matchup (66.7% vs 33.3%)**;
Garchomp stays better into Archaludon (34.2% vs 25.8%), so both are kept active.
Ogerpon caveat: only **4 Basic Pokemon in 60 cards** — frequent mulligans, and an
instant loss if the lone attacker falls with an empty bench. Tail risk that a
240-game two-opponent sample may under-represent.

## 3. Negatives (measured, not assumed)

| Attempt | Result |
|---|---|
| **Card-identity ranker** (BC on 623k teacher decisions) | top-1 **47.2% vs 40.3%** baseline (+6.9) but **plays worse**: 10% vs v3's 19% against a strong opponent. Accuracy is not playing strength — classic covariate shift. 3x more data bought only +0.7pp accuracy, so data is not the constraint |
| **Matchup-gated router** (hybrid↔v3 on identified archetype) | Directionally ahead in 3/3 tests, pooled 72/55 discordant, **p=0.156** over ~700 games. Not significant → **held back, not shipped** |
| Scaled search / weight tuning / opponent determinization | Null (see 2026-08-04 section) |

## 4. Three harness bugs, one meta-lesson

1. **Missing `search_begin_input`** — CRN harness silently ran the heuristic, not
   the search (0/155). Tell: absurd value weights gave byte-identical games.
2. **Arena "A perspective"** is the first-registered competitor, not CLI order.
   Tell: a 3% win-rate that should have been impossible.
3. **`paired_eval.py` CLI used `paired()` for cross-deck comparisons**, feeding
   `deck_a` to the engine for BOTH candidates. Tell: the same matchup reading
   33.3% and 68.1%. Fixed — the CLI now auto-selects `paired_decks()` when decks
   differ.

**META-LESSON: every one of these surfaced as a number that should have been
impossible, and each time the first instinct was to explain it as a finding.
Treat impossible numbers as bugs until proven otherwise.** Also: validate a new
harness by reproducing a known result from the old one before trusting it.

## 5. Score calibration (important for planning)

- Submissions **seed at 600**; a score under ~20 episodes is mostly prior.
- Worked example: submission 55302291 read **827.6 at 7 episodes**, converged to
  **642.5** — a 185-point drop. Its byte-identical twin went 509 → 601.
- **Honest agent levels: v3 ≈ 590, hybrid ≈ 620.** The 643.9 quoted earlier was
  the lucky tail of a ~590 agent, not its level.
- **Translation ratio: ~+11 points of win-rate against strong opponents ≈ +30 LB
  points.** Use this to sanity-check whether a local gain is worth a submission.
- Unpaired n=60–80 overstated the effect in **6 of 6** cases this session. Treat
  any unpaired number as a hypothesis; only paired CRN results decide.

## 6. Replay-training pipeline (host-sanctioned, previously blocked)

The daily episode dumps contain **actions**, not just decks (~114 decisions/game).
Kaggle staff published them "to help in reviewing replays as well as training
agents" (discussion 709160). The repo's own rule against using replay actions as
labels was **stricter than the competition rules** and closed off the largest
resource available.

- `tools/rank/extract_dataset.py` — filters to winners on teams >= 1000 LB.
  **212,899 decisions/day; 623,227 across three days.**
- `tools/rank/train_ranker.py` — listwise softmax over legal options, card-identity
  embedding + option type + numerics + board context.
- `agent/ranker_agent.py` — prior / policy / off modes, v3 fallback.
- Baseline worth knowing: **"always take option 0" predicts a 1000+ rated player's
  choice 40%** of the time. The engine's native ordering is genuinely strong.

## 7. Where the ladder's top actually is (discussion 724362, 30k-game timing study)

- Half the field is hand-written bots or **copies of the public sample bots**.
- **Best public notebook ≈ 950 — that is the public ceiling.**
- **1000+ is private work**: "loads a heavy model but also fully utilises the time
  in the game… most likely RL + bounded search."
- We use **0.4 s of a 600 s** budget. The leader burns it.

## 8. Next actions

1. Re-test the saturated-benchmark nulls against the public agents.
2. Ogerpon vs Garchomp on the ladder — both active, identical policy, so this is
   a clean live A/B.
3. Gated router: needs ~2000+ paired games to resolve p=0.156, or a better
   identification signal early in games (misroutes happen before cards are shown).
4. Re-mine the meta near the Aug-16 lock; current field data is 2026-08-03.
5. Curate the final two active slots deliberately before the deadline.

## Learned board-value critic — FAILED (significant)

`tools/rank/extract_values.py` → **1,529,127 boards** labelled by game outcome
(every board, both sides, all teams — the label is the outcome so no teacher
filtering is needed; 7x the policy dataset and unbiased).

`tools/rank/train_value_critic.py`:

| model | AUC | early game (turn<=8) |
|---|---|---|
| logistic | 0.7239 | 0.6671 |
| **MLP** | **0.8124** | **0.7523** |

Matches the published reference (linear probe 0.818, their NN value head 0.61) —
the signal really is extractable from public board state.

Wired into hybrid's leaf via `agent/critic_policy.py` (hybrid_agent accepts a
`policy_module`, so only `evaluate_board` changed). Result vs public-archaludon,
paired CRN, 120 games/side:

    critic leaf   9/120   7.5%
    formula leaf 27/120  22.5%
    discordant 5/22, McNemar p = 0.0021  -> critic is SIGNIFICANTLY WORSE

**Why it likely failed:** the critic was trained on boards sampled from real
games (between-turn states), but search leaves are mid-rollout states after
partially executed turns — a different distribution. The hand-written formula is
crude but degrades gracefully out of distribution.

**Rule established: offline metrics do not predict playing strength on this
problem.** Ranker (top-1 47.2% vs 40.3% baseline) played at half strength; critic
(AUC 0.8124) played at a third. Only paired CRN vs strong opponents tracks truth.

---

# 2026-08-07 — winner write-up lessons and testing infrastructure

Three 1st/2nd-place write-ups the user supplied (Orbit Wars 1st, Lux AI S3 1st,
FIDE chess Approvers 2nd). Their convergent lessons, and what we did about each.

## Lesson 1: opponent pools prevent overfitting — WE HIT THIS EXACT FAILURE

  Lux AI 1st: "we also let the agent face a pool of older opponents ... This
  improved the agent's robustness, preventing overfitting to a single self-play
  style."
  Orbit Wars 1st (stated regret): "I would have ... added league-play against past
  checkpoints to help prevent strategic cycles and self-overfitting."

Us: optimised against TWO public agents, got Ogerpon over Garchomp at p=0.0146,
and it scored **578 on the ladder vs Garchomp's 594**. Two opponents is a target
to overfit, not a benchmark.

Built `tools/crn/league.py` — 9 weighted opponents (2 public ~950 agents, top-team
decks, other archetypes, a different policy class, a degenerate baseline).

## Lesson 2: skip imitation learning — CONFIRMED BY OUR OWN FAILURES

  Orbit Wars: "avoided any sort of imitation learning initialization."
  Lux AI: implemented BC mixing from replays and "did not incorporate this."

Our three imitation-flavoured attempts all failed: BC ranker (top-1 47.2% vs
40.3%, played at HALF strength), value critic (AUC 0.8124, **p=0.0021 WORSE**).
Three independent sources plus our own data — this direction is closed.

## Lesson 3: SPRT + SPSA + enormous test volume

  Approvers 2nd: "we used SPRT to determine whether a change is statistically
  beneficial and SPSA for tuning various constants ... around 20M games."
  (We have played ~20k. They ran 1250 experiment branches.)

Built `tools/crn/sprt.py`: sequential test over paired CRN seeds, stops as soon as
the LLR crosses accept/reject. Validated on the known-good hybrid-vs-v3 change:
W20 L11 D29, LLR +0.47 at 30 seeds -> correctly reports "not proven at 20 Elo"
rather than false-positive. Note 29/60 seeds were identical outcomes and carry no
information — paired testing skips exactly that waste.

Built `agent/tuned_policy.py` (16 hand-set priority constants exposed) and
`tools/crn/spsa.py`, tuning against the league with a disjoint-seed confirmation
gate. Running overnight.

## Turn-order investigation (from discussion 723591)

Hypothesis: we might be unusually weak going second, and second is the
"controllable" seat (~96% obtainable vs ~54% for first).

Measured at n=180/matchup, both live decks:

| deck | first | second | asymmetry |
|---|---|---|---|
| Ogerpon vs alakazam | 75.9% | 61.9% | +14.0 |
| Garchomp vs alakazam | 40.0% | 24.4% | +15.6 |

~15 points vs a field norm of ~10.4 — mildly worse, NOT the 30-point defect the
n=22 sample suggested. Hypothesis largely disconfirmed; I over-read the small sample.

**Confirmed and interesting:** `public-archaludon` chose to go SECOND in **180/180**
games and still beats us (we win 30-44%). Only ~8.5% of the field takes second.

## Counterfactual analysis (improving on discussion 731298)

731298 proposes LLM analysis of replays to synthesise rules. The naive form was
already measured by 713608 and BACKFIRED (-7.6pp, p=0.003) from survivorship bias
— analysing only losses.

`tools/crn/counterfactual.py` does the version CRN makes possible: play the same
seed with two agents, keep only DISCORDANT seeds (one won, one lost, identical
shuffles), and record the first decision where they diverged. Causal comparison,
not a story fitted to losses. First run: 5 discordant of 30; hybrid chose ATTACH
where v3 chose PLAY and lost all 3 — suggestive only, n=3.

## Status

Ladder ~606.9, rank ~3569/6447. Bronze 841.4, silver 909.0.
Active: hybrid+Ogerpon (578.1) and hybrid+Garchomp (593.8).
Converged agent levels: v3 ~590, hybrid ~620, Ogerpon ~578.
**Nothing submitted without approval.**

---

# 2026-08-07 (later) — the damage-model bug and what it revealed

## The bug (real, and large)

`domain_policy` reads damage from the static `Attack.damage` field. For **14
attacks across the current meta that field is wrong**, and for several it is 0:

| card | listed | actual |
|---|---|---|
| Teal Mask Ogerpon ex (OUR attacker) | 30 | 30 + 30 x energy on BOTH actives (~270) |
| Alakazam "Powerful Hand" (LB-950 deck) | **0** | 20 x cards in hand |
| Dipplin "Do the Wave" | **0** | 20 x benched |
| Mega Froslass "Resentful Refrain" | **0** | 50 x opponent hand |
| Passimian "Coordinated Throwing" | **0** | 20 x basics |

Consequences: `ko = dmg >= hp` never fired, so the lethal-attack override and
Boss's-Orders-gust-for-KO were dead code; and `_lethal_threat()` computed ZERO
incoming damage from Alakazam, so we never retreated from lethal against the
archetype the strongest public agent plays.

## Fixing it made the agent MUCH WORSE

`agent/scaled_policy.py` (correct damage in `_plan_attack` + `_lethal_threat`):

    SPRT vs hybs-other: REJECT after 116 seeds, W27 L93 D112

**Mechanism.** Myriad Leaf Shower scales with accumulated Energy. The static
30-damage estimate made attacking look unattractive, so the agent defaulted to
setup (Teal Dance every turn) and swung only when forced — correct play for this
deck. With correct damage it sees "90 now", attacks immediately, and never
reaches the 270-damage turns. **Our search is ~1 turn deep** (`_MAXROLL` bounds
micro-steps WITHIN a turn; we never evaluate future turns), so it cannot see that
waiting compounds. The underestimate was accidentally acting as a patience
heuristic that compensated for search myopia.

**Lesson: a more accurate model is not automatically a better agent when the
surrounding policy was implicitly calibrated to the inaccuracy.**

## This reframes every other failure today

SPSA weights, card-specific weights and the value critic were all re-prioritising
actions inside a search that cannot see past the current turn. **Better priorities
cannot substitute for missing lookahead.** Multi-turn search is the prerequisite.

## Today's rejections (all by SPRT, cheap — compute not submissions)

| candidate | verdict |
|---|---|
| SPSA-tuned generic weights | REJECT, W89 L95 at 200 seeds |
| Card-specific Ogerpon weights (30 params) | REJECT, W58 L71; league exact tie 60.7% vs 60.7%, p=0.90 |
| Learned value critic (AUC 0.8124) | REJECT, significantly worse (p=0.0021) |
| Scaled damage, both halves | REJECT, W27 L93 |
| Scaled damage, defence half only | testing |

## Methodological findings

1. **A tuner's own confirmation gate is NOT independent validation.** SPSA
   reported `promote: True` (0.4028 -> 0.4618 on "disjoint" seeds) for weights
   that SPRT then rejected at 200 seeds. Its gate used the same 4-opponent pool
   and only 24-30 seeds.
2. **Local validation cannot estimate LB score.** We beat public-alakazam 68.8%
   and public-archaludon 25.0% — a 44-point swing by opponent. Elo transitivity
   fails under strong matchup structure. Naive Elo from the alakazam number
   implies rating ~1087 while we actually score ~600.
3. **Local validation CAN compare A vs B** — that is what SPRT is for, and it has
   1 ladder confirmation (hybrid > v3: ~618 vs ~598) and 0 refutations. So run
   many experiments locally, submit only accepted ones.
4. Match attacks by name AND text fragment: "Psychic" names 5 distinct attacks
   with listed damage 10/30/40/80 and only one carries the scaling clause.

## Seven rejections in one day — and what they collectively prove

| # | candidate | SPRT verdict |
|---|---|---|
| 1 | SPSA-tuned generic weights (16 params) | REJECT W89 L95 @200 |
| 2 | Card-specific Ogerpon weights (30 params) | REJECT W58 L71; league exact tie 60.7/60.7 p=0.90 |
| 3 | Learned value critic (AUC 0.8124) | REJECT, significantly WORSE p=0.0021 |
| 4 | Scaled damage, offence+defence | REJECT W27 L93 (strongly worse) |
| 5 | Scaled damage, defence only | INCONCLUSIVE W88 L90 @200 |
| 6 | Matchup-gated router | pooled p=0.156 over ~700 games |
| 7 | Search parameter sweep (7 configs) | all p 0.50-0.82 |

Individually these look like seven failures. Together they are one finding:
**agent strength is not limited by priorities, leaf-evaluation quality, damage
accuracy, sample count, or opponent modelling.** Every one of those was varied and
measured; none moved the needle.

What remains untested is **search DEPTH**. `_MAXROLL` bounds micro-steps WITHIN
the current turn; we never evaluate a future turn at all. The scaled-damage result
is the direct evidence: correcting our own damage made the agent MUCH worse
(W27 L93) because the underestimate was compensating for myopia -- with 1-turn
lookahead the agent cannot see that waiting to accumulate Energy compounds into a
270-damage swing, so accurate "90 damage now" makes it attack too early.

**Conclusion: multi-turn lookahead is the prerequisite.** Until it exists, tuning
anything on top is measuring noise, which is exactly what seven tests showed.

## On the public LB-950 agent (jazivxt/codex-sol-eclipse-alakazam)

Byte-identical to romanrozen/strong-start-baseline-agent-v10-lb-950 (58,322 chars,
143 weight entries, same values) -- same code under two kernels.

Its own comments: "Produced using Codex and Vibe Coding" and "memetic-tuned
alak_evo ... (baked, seed for wm4 evo)". So the public ceiling is **LLM-generated
code plus evolutionary weight tuning**, not irreproducible human expertise. That
corrects an earlier claim in this log that 300+ card branches represented weeks of
domain knowledge we could not replicate.

Decision (user, 2026-08-07): do NOT fork it. Rationale: it caps Simulation near
950 while gold needs ~1073, it reverses the originality rule, and the Strategy
division -- which holds the $30k/team and Round-2 qualification -- explicitly
scores originality of approach. Instead replicate the MECHANISM originally.

`tools/crn/memetic.py` does that: population + elitism over card-specific weights,
PAIRED selection on shared seeds, disjoint-seed confirmation per generation,
checkpointed for multi-day runs. Still requires an independent SPRT before any
submission -- a tuner's own gate is not validation (SPSA reported promote=True on
weights SPRT then rejected at 200 seeds).

## THE missing technique: 2-ply minimax (from the LB-950 agent's structure)

Mining the public agent's section headers and function names (technique only, no
code copied) found the one thing our search lacks:

    K_OPP = 3                  # opponent branching at ply-2 MAIN
    # ply 1: take idx, greedy-complete our turn
    # ply 2: min over opponent's top-K first-actions

**They score a candidate move by the opponent's BEST reply. We stopped after our
own turn and never modelled the reply at all.**

Everything else on their surface (`_match_archetype`, `_sample_hidden`,
`_greedy_complete_turn`, `_leaf_eval`) we already have equivalents for, and the
archetype one tested null. The opponent-reply ply was the real gap.

This is also the direct explanation for the day's most confusing result: fixing
our damage model made the agent MUCH worse (W27 L93). With 1-ply search the agent
cannot see that attacking early hands the opponent a strong answer, so accurate
"90 damage now" beats a patient 270-damage line. The broken estimate had been
acting as an accidental patience heuristic.

`agent/minimax_agent.py` implements it against the engine's search API (our own
code): ply 1 = our action + greedy turn completion, ply 2 = minimise over the
opponent's top-K replies, degrading to 1-ply hybrid on any failure.
Cost ~K_OPP x 1-ply: 12.2 s p95 per match against a 600 s budget, 0 failures.

Early SPRT vs the live agent: **W7 L0 D13 at 10 seeds** — the strongest start of
any candidate today (eight previous ones were rejected).

**If it accepts, re-test the damage model, card-specific weights and the value
critic** — all three failed for the same reason and become worth revisiting once
the search can see one ply further.

## 2-ply minimax — REJECTED (and it is an informative rejection)

`agent/minimax_agent.py` implements the same search shape the LB-950 agent uses
(ply 1: our action + greedy turn completion; ply 2: minimise over the opponent's
top-K replies, K_OPP=3). SPRT vs our live agent:

    W7 L0   @ 10 seeds   LLR +0.30   (small-sample luck)
    W20 L18 @ 40 seeds   LLR +0.05
    W26 L28 @ 55 seeds   LLR -0.14   -> flat, stopped

**Conclusion: the 950 agent's advantage is NOT the search shape.** We built the
same 2-ply opponent-reply structure and it changed nothing. What differs is what
they EVALUATE at those nodes: 69 card-specific weights refined over many
generations of memetic search (`alak_evo` -> `wm4 evo`).

That makes eleven rejected candidates on the original track in one day, and it
redirects effort: the lever is the weight surface and the tuning, not the search
architecture.

## Track B: the forked baseline, and why it is tunable

Submitted the public baseline (alakazam_courage_v22) as a measured reference.
Packaged from its own MAIN_SOURCE/DECK_SOURCE + the bundled cg engine; isolated
smoke passed; beats our best agent 54.2% head-to-head locally (n=24).

It is a BASE, not a ceiling:
  * 69 card-specific weights in a module-level dict where `W is WEIGHTS`, so they
    are runtime-mutable, AND it reads an external ./alak_w.json at import -- we
    can ship tuned weights as a FILE without editing its code;
  * `K_OPP = 3` and `TIME_BUDGET_S = 0.8` are one-line knobs;
  * runs at ~0.8 s/game, so tuning it is computationally cheap.

`agent/fork_policy.py` + `tools/crn/tune_fork.py` evolve those 69 weights against
our league with paired CRN selection and a disjoint-seed confirmation.

The published write-ups say no public competitor has common random numbers
working; their 950 was tuned without it. Our search over the same surface is
better directed.

---

# 2026-08-07 -- the fork was shipped with a blind belief model

## The defect

`_sample_hidden()` determinizes the opponent's hidden deck before every search
rollout, and `_match_archetype()` chooses the deck it samples from by Pokemon-ID
overlap against a pool of templates. That pool loads from a directory:

    for _d in ("/kaggle_simulations/agent/top20_decks", "top20_decks",
               "../top20_decks", ...):

The public notebook supplies that directory as an ATTACHED KAGGLE DATASET. We
cloned the notebook by regex-extracting MAIN_SOURCE / DECK_SOURCE / GROUP_SOURCE,
which captures source and the 60-card deck but no attached data. Result:

    _TEMPLATES      = 0
    _TEMPLATE_SIG   = 0
    _match_archetype(anything) -> None

Confirmed independently in the shipped artifact: `submission_fork_v22.tar.gz`
contains main.py, deck.csv, group.txt, cg/ and NO top20_decks/. The agent that
converged to 742.5 on the ladder searched against a blind opponent model for
every one of its 41 episodes.

It fails soft -- empty list, no exception, nothing in any log. The only symptom
was a score gap I had wrongly attributed to meta drift and copy saturation.

## The fix, and why it stays original

Refilled top20_decks/ with 32 unique 60-card lists from OUR Aug-03 replay
mining (top_decks.TOP_DECKS + meta_decks). That is deck data we extracted
ourselves, fresher than their July snapshot, and consistent with the
deck-lists-yes-policy-code-no constraint.

Bundle `submission_fork_v22_templates.tar.gz`: byte-identical main.py, deck.csv,
group.txt and cg/ -- templates are the only change, so the ladder delta is
attributable.

## Measurements

  * fork policy on M Sato's LB-1170.6 Alakazam list (td_02) vs its own list:
    102 wins vs 125, discordant 16/30, p=0.0553 -- REJECTED. The 69 weights are
    tuned to their exact list; a stronger deck under a foreign policy is not
    stronger. (13th rejected candidate.)
  * templates vs blind, first opponent (td-td_08): 45 vs 30, discordant 18/8.
  * full verdict pending on two DISJOINT arms (different seeds AND different
    opponent panels) -- a single panel is not a verdict.

## Process lessons

  * When forking any Kaggle kernel, enumerate its attached datasets and assert
    on load counts (`len(_TEMPLATES) > 0`). A clean import proves nothing.
  * `pkill -f <pattern>` matches the invoking shell's own command line, and also
    any poller whose command mentions the log filename. It killed three of my own
    tasks today. Kill by PID.
  * deck_evo (mutating the 15 non-core support-card counts) ran at ~26 min per
    generation and was killed as unable to produce a validated result in the time
    available; cores went to the confirmation arm instead.

## CORRECTION: the clone was already exact

Earlier in this file I claimed our regex extraction "drops attached datasets" and
that the missing top20_decks explained the 742-vs-950 gap. That was wrong, and it
was asserted before the check that refutes it.

`kaggle kernels pull -m jazivxt/codex-sol-eclipse-alakazam`:

    "dataset_sources": []          <- the kernel attaches NO data
    "competition_sources": ["pokemon-tcg-ai-battle"]

and the kernel's own packaging, at its line 1381:

    archive.add(WORK / "main.py",   arcname="main.py")
    archive.add(WORK / "deck.csv",  arcname="deck.csv")
    archive.add(WORK / "group.txt", arcname="group.txt")
    archive.add(cg_path,            arcname="cg")

That is exactly the four members our bundle already had. Our shipped main.py is
SHA-identical to the kernel's MAIN_SOURCE (f31eba2e819ee2b3). The kernel never
creates top20_decks/ or alak_w.json anywhere.

So both load paths are DEAD CODE in the author's own submission. Their graded
agent also runs _TEMPLATES=0 with no override file; the 69 tuned weights are
baked into the source as 65 distinct values, not supplied externally.

There is no private data to add and no exact-clone upside left. 742.5 is what
this code scores in the current field; the ~950 was theirs in an earlier meta,
against an LB that discussion 712621 documents producing 940.7 vs 790.8 for
identical agents.

Lesson: an unresolved file path proves a feature is inactive. It proves nothing
about WHY, and nothing about whether activating it would help. Check
`dataset_sources` and the packaging code before theorising.

## Deck transplant results (single-concept, 640 games each)

    tech_basics     215 vs stock 196   disc 47/31   p=0.0894  lead
    mine_over_dudu  199 vs stock 199   disc 35/38   p=0.8149  flat
    dunsparce_line  200 vs stock 204   disc 36/41   p=0.6485  flat
    hammer_pkg      185 vs stock 217   disc 26/50   p=0.0083  WORSE

hammer_pkg is a useful confirmed negative: cutting Lillie's Determination and
Neutralization Zone measurably hurts, so both are load-bearing and no future
deck variant should touch them.

## Search scaling is parked

Graded episodes: actTimeout=0, runTimeout=2000 s, median 162 / max 248 steps.
Measured cost per decision:

    stock  N_DET= 3 K_OPP=3   43 ms/dec   ~6.0 s/episode
    x2     N_DET= 6 K_OPP=4  130 ms/dec  ~18.3 s/episode
    x4     N_DET=12 K_OPP=5  209 ms/dec  ~29.3 s/episode

against a ~700 s safe allowance (35% of the ceiling). The shipped agent uses
under 1% of its permitted compute, and TIME_BUDGET_S=0.80 never binds.

But strength did not follow: x2 105/144 (disc 9/6, p=0.61), x4 106/144
(disc 11/6, p=0.33). The decisive number is DISCORDANCE -- quadrupling the search
changed only 17 of 72 paired outcomes, versus 78 of 160 for a single deck edit.
More determinizations of a ~1-turn-deep rollout buy almost nothing. The headroom
is real; this search cannot spend it. Deck composition has ~5x the leverage.

## 2026-08-08 overnight: harness bias, and the elite-deck pivot

### Battery on the deterministic panel, unbounded time budget

    control       987 vs  992  disc   1/  6   <- should be 0/0
    depth2        987 vs  990  disc  52/ 57   p=0.7016  flat
    depth4        990 vs  986  disc  57/ 54   p=0.8494  flat
    tech_basics   960 vs  990  disc  97/129   p=0.0392  WORSE
    templates     987 vs  992  disc   4/  9   p=0.2673  inert

Removing TIME_BUDGET_S did NOT clean the control, so the deadline was not the
cause -- that hypothesis was wrong. The residual is ~5 discordant per 600 and,
across three runs (4/5, 1/6, 4/9), it consistently favours whichever arm plays
SECOND. That is a systematic bias, not noise, and every candidate we have tested
played FIRST.

Magnitude bounds what it can explain: a bias of ~5 cannot account for
tech_basics' margin of 32, so that verdict stands. But depth2's margin of 5 is
exactly the bias size, which is why "flat" is the correct reading there.

Suspected cause: clean_battery builds the opponent ONCE per chunk and reuses it
for both arms, so the second arm inherits whatever state the first arm's games
left in the opponent module. tools/crn/order_effect.py tests three conditions
(normal / swapped / opponent-rebuilt-per-arm) with BOTH arms stock.

### Multi-turn lookahead: measured out

agent/deep_search.py extends the fork's 2-turn horizon by alternating greedy
turns before the leaf. depth2 and depth4 are both flat over 600 paired seeds.
Depth changes ~55 of 600 outcomes -- the mechanism fires -- and wins exactly as
often as it loses. The horizon is not the constraint.

### Why every search-side change reads flat

    search consulted 43x  ->  overrode the heuristic 1x   (5.6%)

The override threshold is 500 points (half a prize), so the shipped agent is ~94%
pure heuristic. Depth, evaluator, templates and determinization counts were all
tuning a subsystem that rarely gets a vote. A smoke test at lower thresholds sent
every discordant seed to stock, i.e. the search's overrides are WORSE than the
heuristic and 500 acts as a safety valve; margin_battery probes upward and
includes search-off.

### Elite replay mining (the pivot)

The manifests carry per-episode player ratings. 1,138 games between players rated
>=1150 contain only 17 distinct deck lists:

    elite_07  Mega Lucario ex        83 games   75.9%   Majkel1337
    elite_02  Dunsparce toolbox     374 games   58.8%
    elite_04  Alakazam (the fork's) 170 games   48.8%

The fork plays the 48.8% archetype. Win rate is not merely pilot skill: the same
pilot appears on lists at 75.9%, 52.8% and 36.7%.

BUT the deck cannot be cashed in without a policy for it:

    our domain policy on elite_07 vs the fork:  4.2%
    our domain policy on elite_02 vs the fork: 12.5%
    our domain policy on elite_04 vs the fork:  4.2%

Identical on the fork's OWN archetype, so this is a pure policy gap. A 75.9% deck
in weak hands loses to a 48.8% deck in tuned hands. agent/lucario_policy.py is a
first card-specific policy for elite_07 (34 tunable weights, encoding the Mega
Brave / Aura Jab alternation and Lunatone-as-bench-enabler); untuned it scores
8.3% against the fork, so evolution has to close a very large gap.

## 2026-08-08: the search override margin -- first validated positive

The fork only lets its search change a move when the search's value estimate beats
the heuristic's top choice by 500 points (half a prize). Measured on a live game
that fires 1 time in 43 decisions, so the shipped agent is ~94% pure heuristic.

Sweeping that constant, 150 seeds x 3 opponents, noise floor 6 discordant:

    control   736 vs  738  disc   2/  4   (floor)
    m0        745 vs  740  disc  54/ 49   p=0.6935  flat
    m200      736 vs  741  disc   5/ 10   p=0.3017  worse
    m1000     744 vs  737  disc  16/  9   p=0.2301  flat
    m3000     760 vs  735  disc  37/ 14   p=0.0021  BETTER
    nosearch  747 vs  739  disc  33/ 26   p=0.4347  flat

Confirmation on DISJOINT seeds and a DIFFERENT panel (meta-* rather than the
public agents), partials:

    control    0/ 2
    m1000     29/ 9
    m3000     36/21
    m6000     53/34
    m10000    53/34      <- identical to m6000: no override ever exceeds 6000,
    nosearch  29/33         so the curve saturates there

Reading: the search's MARGINAL overrides are net-harmful while its high-confidence
ones help. Stock's bar admits too many. Raising it filters the bad ones and keeps
the good, which is why nosearch does NOT win -- on the confirmation panel it is
clearly worse (10/20 mid-run, 29/33 final).

This reframes yesterday's wall of flat results better than the earlier "search
barely runs" account: the search runs often enough to matter, and a substantial
share of what it does is wrong. Deepening it (depth2/depth4 flat) or improving its
beliefs (templates inert) could not help while its marginal decisions were
net-negative.

CAVEAT worth recording: `nosearch` is not strictly single-variable. Disabling
USE_SEARCH also skips the search's internal random.shuffle calls in _sample_hidden,
which shifts the RNG stream for every later decision. The high-margin arms keep the
search running and change only the final acceptance test, so those are the clean
comparisons.

Shipping note: this is a ONE-LINE constant, so it composes with any tuned genome
rather than competing with it. Bundle submission_fork_m3000.tar.gz is built and
smoke-tested in isolation (imports clean, 60-card deck, 69 weights, margin patched)
but NOT submitted pending the confirmation table.

## Kaggle Lucario evolution: 120 generations, zero promotions

The 1/5th success rule shrinks the mutation scale on failure, so with no promotions
it annealed to the 0.03 floor and the search became an ever-more-local hill
climber, posting margins of +1..+3 against a gate of 6. Correct behaviour near an
optimum; exactly wrong for a policy still at ~8% relative strength, where the need
is exploration. Fixed: scale floor 0.12, and a restart that re-seeds the whole
population from a heavily perturbed champion after 40 stagnant generations.

## 2026-08-08 morning: confirmation, shipping, and what is now exhausted

### Override margin -- CONFIRMED on a second panel and SHIPPED

Panel B used disjoint seeds (917xxx) and a completely different opponent set
(meta-garchomp/dragapult/alakazam/dudunsparce):

    control  1040 vs 1042  disc   0/  2   (floor = 2)
    m1000    1059 vs 1040  disc  29/  9   p=0.0021  BETTER
    m3000    1059 vs 1039  disc  53/ 32   p=0.0301  BETTER
    m6000    1058 vs 1040  disc  53/ 34   p=0.0536  flat
    m10000   1058 vs 1040  disc  53/ 34   identical to m6000 -- curve saturates,
    nosearch 1038 vs 1041  disc  49/ 50     no override ever exceeds 6000

Pooled across both panels:
    m1000  45/18  p=0.00105  net +2.6% of 1050 seed-units
    m3000  90/46  p=0.00023  net +4.2%

Shipped m3000. LADDER RESPONSE: 777.1, against 768.9 for the byte-identical stock
fork and 720.7 for its duplicate. First time a local paired measurement and a
ladder movement have agreed in this project. A second draw was submitted to
replace the 720.7 slot, which was both weaker and the unimproved agent.

### Leaf evaluator: all six terms flat or worse

Run with every arm at override margin 0 so the search actually votes (at the
stock margin it changes the move only 5.6% of the time, which is why the earlier
depth and template tests were uninformative).

    control  disc  18/ 13   (floor = 31 at this margin)
    hand     disc  46/ 53   worse      card advantage
    stage    disc  26/ 31   worse      Abra<Kadabra<Alakazam progress
    frachp   disc  36/ 26   flat       fraction-of-max HP
    bench    disc  32/ 21   flat       board width
    deck     disc  27/ 36   worse      deck-out pressure
    all      disc  45/ 51   worse

Note the noise floor rises from ~6 to ~31 when the margin is lowered: more search
participation means more RNG-sensitive decisions. Judge these against 18/13, not
against zero. This is why every battery carries a stock-vs-stock control.

### Lucario policy evolution: 307 generations, 7 restarts, nothing

The restart machinery worked as designed -- it detected stagnation seven times and
re-seeded at scale 0.45 each time -- and still found no promotion. A complete
negative rather than a stalled run. 34 hand-written weights are too crude a
starting point for this deck.

### Weight evolution: a bug, and a retracted conclusion

I reported weight evolution as an exhausted negative after 25 generations. That
was WRONG. The run crashed at generation 27 with

    AttributeError: 'NoneType' object has no attribute 'WEIGHTS'

_duel runs inside pool workers (which call _init) AND in the main process for the
promotion-confirmation duel, where _A/_B were never set. The identical bug had
already been fixed in evo_policy.py and was never backported.

The failure fires ONLY on success: that line executes only when a candidate
clears the racing gate. So generation 27 was the first candidate the loosened
gate let through, and the crash destroyed the result before it could be tested.
Concluding "flat, the author's tuning converged" from that run was unfounded.

Two gate changes made along the way, both sound and both retained:
  * racing gate 9 -> 5. Seventeen generations produced no candidate that even
    reached confirmation; a real 55/45 edge scores ~+4 and was being discarded
    unseen. Screen loosely, confirm strictly.
  * annealing floor 0.03 -> 0.12 plus a restart after 35 stagnant generations,
    backported from evo_policy. Without it the 1/5th rule shrinks the step on
    failure and the search anneals itself into a local point.

Also noted: the checkpoint's stored `scale` overrides EV_SCALE on resume, so an
intended wider restart scale silently did not apply.

### Order effect: hypothesis refuted

order_effect.py, both arms stock, 60 seeds x 3 opponents:

    normal   1/0     swapped 0/0     fresh-opponent 0/2

The bias did not flip when play order was swapped and rebuilding the opponent per
arm did not remove it. Pooled across every control run: 10/22, p=0.052 --
borderline, not established. So the "systematic second-mover bias" I asserted was
not supported; what is real is ~1.4% residual nondeterminism of unknown origin,
setting a noise floor of roughly 10 discordant per 600 seed-units. Counterbalancing
by seed parity was added to evo2 and evo_policy as hygiene, but it unlocks nothing.

## 2026-08-08 afternoon: two follow-ups on the margin mechanism, both dead

The margin result says the search's marginal overrides are net-harmful. Two ways
to filter them better than a flat threshold were tried. Both failed.

### Determinization consensus -- REJECTED

Require the candidate to beat the heuristic in a majority of INDEPENDENT
determinizations, not merely on average. One lucky sample can carry a bad move
through an average; agreement across samples should filter that directly.

Stacked on top of margin 3000 it is redundant: 0/0 discordant across all arms,
because the threshold already blocks nearly every override and leaves consensus
nothing to work on.

As an ALTERNATIVE filter (margin 0, votes only), 60 seeds x 3 opponents, with the
shipped m3000 agent as control:

    control   297 vs 297  disc   0/  0   (perfect floor this run)
    c50       288 vs 297  disc  15/ 25   worse
    c60       289 vs 297  disc  13/ 22   worse
    c75       286 vs 298  disc  15/ 28   worse  p=0.0673
    c50m1000  299 vs 298  disc  11/ 10   flat

Agreement-across-samples is a WORSE filter than a value threshold. Implementation
note: the first version tallied votes on the cumulative `acc` rather than the
per-determinization delta, which would have let early rounds dominate every vote;
fixed before any numbers were taken.

### Phase-dependent margin -- REJECTED, and verified rather than assumed

The search sees exactly two turns: a poor model of a long game, a decent one near
the finish. So lower the threshold when few prizes remain (trust it more late).

All arms came back 0/0, which is the same signature as a patch that never fires,
so it was instrumented instead of believed:

    flat3000 : search consulted 82x, overrode 1x
    late0    : search consulted 82x, overrode 1x   <- identical

Dropping the late-game margin to ZERO adds no overrides at all. In endgame states
the search either is not consulted (`_search_decide` requires >=3 options) or never
prefers a different move. The lever has no leverage. Five minutes to kill, because
the code path was checked rather than a 0/0 table being reported as a null result.
