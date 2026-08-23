# Experiment log

Chronological working record kept during the competition, newest sections at
the end. It is a lab notebook, not a report: it records what was measured,
what was rejected, and what turned out to be measurement error.
Raw numbers for the earlier phase are in `experiments-2026-08-04.md`.

---


Single-file handoff. Read this before proposing any experiment; ~20 approaches
below are already measured and dead, most of them the obvious ones.

Detail and raw numbers: `EXPERIMENTS_2026-08-04.md` (1100+ lines, chronological).

---

## 1. Current state (2026-08-08)

| | |
|---|---|
| Best score | **777.1** |
| Agent | public notebook `jazivxt/codex-sol-eclipse-alakazam`, one constant changed |
| Session start | 531.2 → 761.7 → 777.1 |
| Bronze | ~839 |
| Top of leaderboard | ~1200-1350 |
| Submissions | 2 active slots only; space them 5-10h or they never converge |

### Correction — owned historical high-water is 844.4

The table above describes the current Aug-08 Alakazam campaign, but it omits an
older exact completed submission owned by this workspace: Observable Meta Router
V3, row `55056992`, scored **844.4**. Its Great Tusk/Crustle policy is therefore
the closest verified starting point to 900, although an unchanged reuse would be
crowded by public copies and is not a promotion candidate. The current research
route is a materially distinct guarded successor, validated locally only.

The complete Aug-07 leaderboard deck snapshot (`myso1987/...meta-by-score-band`)
recovered every deck in the upper bands. The relevant target distribution is:

    900-999:   Grimmsnarl 46.0%, Alakazam 18.8%, Crustle 10.5%
    1000-1099: Grimmsnarl 26.9%, Alakazam 18.3%, Froslass 14.0%, Ogerpon 10.8%
    1100+:     Grimmsnarl 26.7%, Alakazam 20.0%, Froslass 20.0%

Thus a credible 900+ candidate must primarily improve Grimmsnarl/Froslass while
preserving Alakazam and Crustle. Public code is used only for hypotheses and
controls; the final candidate must contain locally validated original changes.

### Guarded 844 successor — upper-band screen PASSED (confirmation pending)

Restored the owned Great Tusk/Crustle router payload and compared its original
guarded terminal-mill successor with the exact router844 control branch. Same
source, same deck, same CRN seeds; the control disables only the visible-family
guard. 100 fresh seeds x both seats per matchup:

    Grimmsnarl       guarded 180  control 160   disc 24/6   p=0.00191
    Alakazam         guarded 184  control 184   disc  0/0   invariant
    Froslass/Lopunny guarded 124  control 125   disc  0/1   invariant
    Festival Lead   guarded 132  control  95   disc 37/6   p<1e-6
    pooled          guarded 620  control 564   disc 61/13  p<1e-6

This is the first strong locally measured improvement over the verified 844.4
line in today's campaign. It is materially distinct from public agents and its
Alakazam safety veto is exactly invariant in this screen. Do not promote yet:
requires disjoint seeds plus a different opponent panel, then upper-band weighted
validation. Harness: `tools/crn/router_v4_battery.py`.

Confirmation results:

    disjoint domain-policy panel: 615 vs 597, disc 37/18, p=0.0152
    active-search upper panel:     242 vs 218, disc 76/51, p=0.0332

The search-pilot panel used 60 disjoint seeds and all targeted matchup deltas
were directionally favorable (Grim +6, Froslass/Lopunny +6, Festival +11), while
Alakazam was +1. The guarded successor is now the local control for further work.
It is not yet labeled 900+; next target is a Froslass-specific hand-size package.

Froslass/deck screens after that gate:

* Hand Trimmer 1-4 copies: rejected; none improved Froslass or the field.
* Guarded Amarys 1-4 copies: rejected at smoke; worsened Froslass.
* Roto-Stick replacing Xerosic: four copies gave a weighted-looking but pooled
  flat trade (+12 Fros, +6 Grim, -14 Festival, -4 Alakazam over 100 seeds).
* Three Roto-Sticks preserving Xerosic passed its first screen (392/363,
  p=0.007) but **failed fresh confirmation**: 753/749, p=0.65, with the Froslass
  delta reversing to -7. Rejected as a first-panel false positive.

No deck mutation is promoted. The confirmed guarded policy on the exact 844 deck
remains control. Next causal lead: Crustle wall mode cannot stop Grim/Froslass
damage-counter effects and may waste terminal-mill tempo.

Wall-mode policy screen:

* disabling wall for all damage-counter families: flat/worse;
* disabling it for every favorable family: flat;
* disabling it only with an attack-ready Tusk initially read +5/-1 discordant,
  but failed a 200-seed fresh confirmation at 13/15 and was rejected.

The exact confirmed guarded v4 remains the champion. Low-dimensional causal
coordinate screens were then run over wall cutoffs 12/16/24/28, bench floor 2,
Xerosic threshold 6, and an aggressive Xerosic branch (60 fresh seeds each).
Every variant was flat or worse; none promoted.

Visible-family-gated generic search was also tested so that the router remained
unchanged outside a named matchup. It was decisively harmful even on the intended
families: Alakazam 161/179 (discordant 5/22, p=0.0021), Froslass 149/179
(5/31, p=0.00003), damage families 139/179, and all favorable families 135/179.
Shared support-card IDs also made apparently narrow family gates activate more
broadly than intended. Do not retry generic search behind an archetype gate.

### Live update — public 1084.5 Lucario lead rejected

Pulled `makthanithin/pokemon-tcg-ai-battle-1084-5-baseline`, including its
published kernel output. The output `main.py` contained an obvious stray `hi`
syntax token after a Crustle guard; repaired only that token and ran the policy
locally, both seats, on fresh CRN seeds against the validated m3000 fork:

    public1084 Lucario   11/80
    fork m3000           40/80
    seed-level better     0 vs 29    p < 1e-6

**REJECTED.** The notebook's claimed 1084.5 title and 0.7092 local Public21 mean
do not survive comparison with our control. Its reported opponents are mostly
older/weaker public agents, illustrating why notebook titles and uncalibrated
absolute win rates are not evidence. Do not repeat or evolve this policy.

Files retained for audit:
`references/top_rankers/baseline_1084/` and
`tools/crn/eval_public_1084.py`. No submission was made.

### Hard-opponent audit — guarded v4 does not clear strong Grim

The Aug-05 public `tetsutani/grimmsnarl-ex-damage-transfer-control` archive was
used strictly as a state-reset opponent (not copied into the candidate). Against
this substantially more sophisticated Grim pilot, 60 fresh seeds x both seats:

    guarded v4      19/120
    exact router844 20/120
    discordant        6/6, p=0.77

This falsifies the strongest interpretation of the earlier domain-policy Grim
gain: v4 improves against simpler Grim pilots but is flat and only ~16% winning
against the stronger policy. It remains the best owned Great Tusk successor,
but is **not yet credible as 900+**. Harness:
`tools/crn/router_public_grim_battery.py`. The research route now evaluates the
public Grim experts as counterfactual opponents/bases and requires an original,
locally confirmed residual layer rather than submitting an unchanged public
archive.

### New local champion — Grim Candy v1 (credible 900+ candidate)

The strong public Grim policy was evaluated as a base rather than copied
unchanged. It beat the owned guarded Great Tusk line 101/120 in the hard-opponent
audit (the inverse of Great Tusk's 19/120) and scored 101/120 against the clean
public Archaludon/Alakazam plus deterministic Grim panel. That strength is far
above the verified 844.4 line locally, though local play cannot prove an absolute
ladder rating.

Counterfactual forced-expert routing and five post-policy layer ablations found
no reliable gain. A constrained 19-arm one-card deck evolution then identified
one coherent change: **Tool Scrapper -> fourth Rare Candy**. The screen itself is
not counted as evidence. On two disjoint 300-seed confirmations against public
Archaludon and public Alakazam, both seats:

    panel 1: 899/868, discordant 140/112, p=0.0890
    panel 2: 872/859, discordant 141/122, p=0.2670
    combined: 1771/1727, discordant 281/234, p=0.0427

Combined matchup deltas were **Alakazam +45, Archaludon -1**. A separate
100-seed three-opponent check was 496/490 and essentially invariant against
Grim. This is a material, statistically confirmed improvement aimed at the
18-20% upper-band Alakazam share, rather than an unchanged crowded public copy.

Frozen local candidate: `agent/grim_candy_v1/`. Import, compile, 60-card deck
invariants, and a 10-game engine smoke test passed (8/10 vs public Alakazam).
Detailed receipt: `agent/grim_candy_v1/LOCAL_VALIDATION.md`. Harnesses:
`tools/crn/grim_expert_route_battery.py` and `tools/crn/grim_deck_battery.py`.
**No submission or Kaggle write was made.**

Second-generation one-card evolution around Grim Candy v1 tested 14 coordinates
over Dawn, Petrel, and Spikemuth Gym. Thirteen were flat or worse. The only
screen leader (Gym -> fourth Grimmsnarl, 76/74) failed a fresh 100-seed
confirmation at 484/492, discordant 46/53, p=0.55. **Rejected.** Candy v1 remains
the champion; do not stack another one-card count change from this generation.

Replay-mined expansion (`tools/crn/mine_grim_decks.py`) found 8,573 Grim deck
usages in cached >=900-rated matches. The published list dominated (6,961 uses,
average matchup rating 1097.9), supporting the base choice. Ten tech coordinates
from rarer high-rated lists were screened around Candy v1. Removing Pokégear was
uniformly worse. Fourth Candy -> Handheld Fan or Xerosic each led the screen at
104/100, but failed fresh 100-seed confirmation: Fan 498/500 (39/43), Xerosic
502/500 (45/44). **Rejected.** No replay-mined tech promoted.

### New champion — Grim Candy v2

Counterfactual arbitration found that the existing matchup router recognizes an
`arch` visible-card profile but keeps generic arbitration there. A new narrow
residual selects the already-computed mirror-expert action only after that
public-state profile locks. Discovery and fresh confirmation on the original
deck:

    100 seeds: Arch 134/114, disc 33/13, p=0.0051
    200 fresh: Arch 269/249, disc 59/41, p=0.0891
    combined discordance 92/54, p=0.0022

Alakazam and Grim were exactly invariant in both panels. Two different local
Arch policies were flat (+2, 16/14), not adverse. Crucially, a separate stacked
test used Candy v1 as the exact control: over 150 new seeds v2 was 749/733,
Arch 215/199 (43/27, p=0.073), with Alakazam and Grim again exactly invariant.
The fourth-Candy deck and Arch policy gate therefore compose directionally.

Frozen candidate: `agent/grim_candy_v2/`. Deck/import/compile invariants pass.
This is materially stronger and more distinct than v1. No submission was made.

Arch-gate context decomposition showed attachment-only +12/200, miscellaneous
contexts -2, and a leave-damage-out mask +22 versus v1 on its screen (full v2
was +19). The apparent v3 edge did **not** confirm when tested directly against
full v2 over 300 fresh seeds: 405/403, discordant 43/38, p=0.66. Rejected; retain
the simpler full Arch gate in v2.

Five visible-psychic-only post-policy ablations were screened on 100 fresh
public-Alakazam seeds. Advisor and tactical removal were exactly invariant;
residual and development were 164/166; robustness 165/166. No promotion. The
Alakazam residual stack is not the next lever.

Final v2 leakage audit, Candy v1 exact control, 150 fresh seeds x both seats:

    Dragapult 294/294, alternate Froslass 285/285,
    Ogerpon 127/127, Crustle 292/292; pooled 998/998, discordant 0/0.

Thus the visible Arch gate is exactly inert on this four-family regression
panel. V2 is submission-ready locally. Archive:
`artifacts/grim_candy_v2.tar.gz` (2.6 MB; main, deck, model and cg runtime
verified present).

User explicitly authorized submission. Uploaded at 2026-08-08 10:50:25 UTC:

    submission id: 55348264
    file:          grim_candy_v2.tar.gz
    initial state: PENDING

Kaggle reported two daily submissions remaining after this upload. This replaced
the weaker active m3000 slot; the 762.7 m3000 control remains active. Do not make
a second copy or another submission until this row completes and begins settling.

Live status: submission `55348264` completed packaging with an **initial** public
score of **600.0**. This is not settled: competition ratings need roughly 2-3
hours of matchmaking before interpretation. Do not duplicate or replace v2
during that window. The ~1050 estimate remains an unverified forecast, not a
result; 600 is only the first observation. Recheck after the settling window and
anchor subsequent decisions to the stabilized row. Two daily submissions remain.

The first live movement was **698.3** at about 16:23 IST, only minutes after
activation. It is still provisional and must not be used as the final estimate;
the meaningful review window is approximately 18:20-19:20 IST.

At 16:47 IST the row had climbed to **748.5**. This is only ~27 minutes after
activation and remains inside the explicitly protected settling window. The two
active m3000 rows were 714.0 and 762.7 at the same snapshot. No action yet.

**Goal achieved:** at 17:18 IST Kaggle reported Grim Candy v2 at **902.6**.
Submission `55348264` is COMPLETE and publicly above 900. This happened roughly
58 minutes after activation, so it may still move as matchmaking continues.
Neither of the two remaining daily submissions was spent. Keep monitoring; do
not evict the 902.6 row casually.

Subsequent volatility: v2 reached **919.4** at 17:29, then fell to **891.0** at
17:32. The 900 crossing is real but not settled; continue monitoring and protect
both remaining submissions.

Later trajectory: 856.7 at 18:00, 858.3 at 18:14, and **826.0 at 18:32**. The
919 peak did not settle. Use 826 as the latest observation, while retaining 856
as the user's stricter replacement threshold.

At 19:01 the row was **831.9**; it appears to be oscillating around the low-830s
rather than recovering toward the transient 919 peak.

At 19:09 Grim v2 read **844.4**. Continue treating the rating as path-dependent;
the replacement threshold is still the user's fixed 856, not a momentary dip.

During settling, the public former-5th-place Alakazam artifact
`ryotasueyoshi/rule-based-not-psychic-alakazam-best-5th` was audited against
m3000. It scored 7/80 versus the control's 40/80, with seed discordance 0/33
(p<1e-6). **Decisively rejected.** Harness: `tools/crn/eval_public_alak5.py`.

The public Archaludon/Starmie notebook
`masamikobayashi/a-sample-archaludon-75-wr-vs-my-1300-starmie` was audited with
the same harness and fresh seeds. It scored 11/80 versus m3000's 40/80, with
seed discordance 0/29 (p<1e-6). **Decisively rejected.** Notebook title win
rates are not calibrated to this local field.

### Router v5 lead — expanded terminal-mill matchup gate

While v2 settles, a one-gene evolutionary expansion was run around the verified
844.4 router's validated guarded v4 successor. It adds only visible lineage IDs
to v4's terminal-mill favorable allow-list. Initial 100-seed screens and an
independent 200-seed confirmation against deterministic domain opponents:

    family       screen (candidate/control)   fresh confirmation   discord p
    Mega Fros    91/75                        160/130              39/10 0.000063
    Slowking     86/70                        165/135              54/30 0.01209
    Ogerpon      191/195                      --                   rejected
    Garchomp     193/189                      --                   too small

Mega Froslass and Slowking are the only promoted coordinates so far. The gains
replicated on disjoint seeds and are unusually large, but they are not yet a
shippable v5: active-search confirmation and a combined-coordinate safety panel
are required. Harness: `tools/crn/router_expand_gate_battery.py`.

Combined candidate frozen at `agent/router_v5/`. On a third fresh target panel
(150 seeds each), it remained directional but weaker: Mega Fros 112/108,
Slowking 127/115, pooled 239/223 with discordance 37/23 (p=0.0933). A fresh
six-family safety panel was **exactly invariant**: Grim 180/180, Alakazam
178/178, Dragapult 180/180, Ogerpon 195/195, Garchomp 192/192, Crustle 167/167;
pooled 1092/1092 and 0/0 discordance. Thus the visible gates compose cleanly and
do not leak on this deterministic panel. Active-search confirmation remains
running; do not submit v5 before interpreting it and the settled Grim v2 row.

Final larger confirmation (300 further fresh seeds per target) passed strongly:

    Mega Froslass  v5 226/600  v4 214/600  disc 13/1   p=0.00328
    Slowking       v5 257/600  v4 228/600  disc 70/40  p=0.00569
    pooled            483/1200    442/1200 disc 83/41  p=0.000231

The active-search panel agreed directionally (Mega Fros +6, Slowking +9), but
is treated only as corroboration because that opponent is nondeterministic.
Across the four independent deterministic target panels, v5 totals 1224 wins
versus v4's 1075; all panels were directional and the final large panel was
individually decisive. V5 is therefore a validated, original promotion over the
verified-844 router lineage, with exact six-family safety. Import/compile and
60-card checks pass. Frozen archive: `artifacts/router_v5.tar.gz`, sha256
`a28a32f4aa9706e416a2d25cd58a9257969f64de0ed0604c7b2e9401b6423371`.
It is submission-ready but **must wait for Grim v2's 2-3 hour settling window**.

### New local champion — Router v6

A second allow-list generation screened four additional high-leaderboard
lineages. TD12, TD15 and TD20 were small/non-significant. The Team Rocket
Mewtwo/Articuno lineage (IDs 400, 401, 414, 431, 434) was decisive:

    discovery:       169/200 vs v4 141/200, disc 31/7, p=0.000191
    fresh confirm:   343/400 vs v4 275/400, disc 69/6, p<1e-6
    stacked v6/v5:  251/300 vs     207/300, disc 50/13, p=0.000006

The stacked test also proved the v5 Mega Fros and Slowking coordinates exactly
invariant (98/98 and 129/129), so the new Rocket gate composes. A separate
seven-family safety panel covering Grim, Alakazam, Dragapult, Ogerpon,
Garchomp, Crustle and Festival was exactly invariant over 1,400 games:
1229/1229 and 0/0 discordance. Router v6 is now the local router champion.

Frozen source: `agent/router_v6/`. Import, compile, callable-agent and 60-card
checks pass. Archive: `artifacts/router_v6.tar.gz`, sha256
`139fb0c89d5db1678e0bdfa03d33cb3ea61fe59fba90c2ffaac135f51f2400f7`.
This remains a relative local promotion, not an absolute 900 guarantee. Preserve
the remaining two submissions until Grim v2 has settled.

Active-search Rocket corroboration was +10 wins (39/120 vs 29/120), directionally
agreeing but non-significant due to the search pilot's nondeterminism. A direct
fresh comparison of v6 to exact submitted-844 behavior across TD00-TD11 was
1125/1100, discordance 59/39, p=0.05495. This supports a real broad gain but does
not prove 900; it also revealed inherited Dragapult losses on TD03 and TD05.
A narrow remove-Drag-from-favorable ablation is now the next experiment.

The remove-Drag ablation lost 1075/1085 across four Dragapult lists (9/19), so
the apparent broad-panel negatives were noise. **Rejected; v6 unchanged.**

### Router v7 — strict TD12 profile

A loose TD12 lineage screen confirmed 466/450 (23/7, p=0.00617), but its
Dwebble/Crustle IDs overlap our own mirror and could not be shipped safely. A
conjunctive profile requiring visible Cornerstone Ogerpon 117 plus Dwebble or
Crustle was then tested directly against v6:

    TD12 fresh: 449/500 vs 435/500, disc 19/5, p=0.007963
    mirror/Ogerpon/Garchomp/Alak safety: 741/741, disc 0/0

The strict profile preserves the gain without leakage and is stacked in
`agent/router_v7/`. This is the new local champion pending packaging and a broad
direct confirmation. No submission was made.

Packaging checks now pass (compile, import, callable agent, 60 cards). Frozen
archive: `artifacts/router_v7.tar.gz`, sha256
`39634161399a5299c7607497041a7428b0042e19ce4be99480a909481ecadd09`.
A fresh 13-archetype v7-versus-exact-844 battery is running; archive remains held.

That broad fresh battery **passed decisively** over 80 seeds x both seats x 13
top-deck archetypes (2,080 games):

    router v7 1671  exact-844 behavior 1611
    seed discordance 88/39, McNemar p=0.00002051

Notable deltas: Grim +13, TD03 Dragapult +8, TD05 Dragapult +5, Festival +27,
TD12 +6; Mega Fros -1 and every other family nonnegative. This supersedes the
earlier borderline v6 broad panel and is the strongest calibrated local evidence
in the router lineage. Because exact behavior is anchored to an owned 844.4
submission, v7 plausibly reaches the high-800s/900 region, but ladder variance
precludes promising an absolute score. It is the preferred next submission only
after Grim v2's settling window and a fresh read of both active rows.

V7 broad analysis exposed that TD04's mixed Mega Froslass list carries shared
Alakazam support IDs 66/305, causing the regression veto to suppress the
validated Fros gate. A strict exception requiring visible Fros core 848/849/861
and absence of the actual Alakazam line 741/742/743 screened 441/398 versus v6
over TD04+TD09 (discordance 71/34, p=0.000443). TD04 was +11 and TD09 +32.
Mirror/Ogerpon/Garchomp/actual-Alakazam safety was exactly invariant 754/754,
0/0. A larger disjoint confirmation was then run before promotion.

The 300-seed disjoint confirmation passed: TD04 429/404 (42/18, p=0.00299),
TD09 228/211 (51/34, p=0.0827), pooled 657/615 with discordance 93/52,
p=0.000894. The strict Fros exception is promoted and stacked with v7 in
`agent/router_v8/`, now the local champion. Broad confirmation and packaging
remain required; no submission was made.

Both are now complete. Fresh 100-seed x both-seats x 13-archetype comparison
against exact submitted-844 behavior (2,600 games):

    router v8 2073  exact-844 behavior 1979
    seed discordance 148/66, McNemar p=0.00000003

Matchup deltas: Grim +14, TD03 Drag +16, TD04 mixed Fros +7, TD05 Drag +2,
TD09 pure Fros +12, Festival +28, Slowking +9, TD12 +6; TD00/02/06/07/08
were exactly invariant. No negative matchup appeared. Import, compile,
callable-agent and 60-card checks pass. Frozen archive:
`artifacts/router_v8.tar.gz`, sha256
`c834d3b149db902a83cc1c250fc2cf4766faa8421f5ccfdeac02585d975abcbe`.
V8 supersedes v7 as the preferred next submission, still held until the v2
settling gate.

Active-search stress corroborated v8 overall: 274/240 versus exact-844 behavior
over six improved families, discordance 74/45, p=0.01027. Slowking was noisy and
negative in this nondeterministic panel, so this is supporting evidence only;
the deterministic 2,600-game gate remains authoritative.

Replay-driven Great Tusk deck mining found zero cached lists containing card 58
at both >=900 and >=800 minimum matchup rating. Therefore the cache provides no
elite deck-mutation coordinates for this lineage; do not invent a replay claim.

### Competitor refresh and v9 lead

Downloaded six current public artifacts to `references/competitor_refresh/`:
the newest hot PTGC notebook, Lucario v8, Dragapult UCB, fixed-metal v15,
Garchomp v28, and Crustle v29. Initial direct both-seat auditions against router
v8 decisively rejected Lucario (12/80 vs 68/80) and Dragapult UCB (11/80 vs
69/80); Garchomp was closer but still behind at 35/80 vs 45/80. Crustle v29
also lost 24/80 vs 56/80. Fixed-metal v15 was the only competitive artifact at
42/80 versus v8's 38/80 (flat, seed 11/9). It is now undergoing the meaningful
paired 13-archetype field audit rather than being accepted from one head-to-head.
These artifacts may remain stress opponents; none is an unchanged submission
candidate.

The remaining refresh completed: newest `ptgc-game` lost 24/80 vs 56/80.
Fixed-metal's apparent head-to-head edge was purely matchup-specific; on a
paired TD00-TD12 field it lost 648/1560 versus router v8's 1254/1560, with every
matchup except Slowking behind. Thus **all six refreshed public agents are
rejected as bases**. They remain useful opponents and source material only.

The profile-collision audit found TD19 is a real Dragapult list whose shared
psychic IDs 65/66 suppress the inherited Drag gate. A strict exception requiring
visible Drag core 119/120/121 and excluding actual Alakazam 741/742/743 gave:

    TD03 460/460, TD05 455/455, TD23 469/469 (exactly invariant)
    TD19 455/442, discordance 21/7, p=0.014019

Mirror/Ogerpon/Garchomp/actual-Alakazam safety was also exactly invariant
749/749, 0/0. This is a promising router v9 coordinate; disjoint confirmation
is running before promotion.

The 400-seed disjoint confirmation **failed completely**: TD19 728/728 with
discordance 24/25; the other three Drag lists remained invariant. The initial
+13 was a seed-panel false positive. **Rejected; router v8 remains champion.**

A separate collision audit identified four 1027-1082 leaderboard variants
(TD12/13/18/26) sharing a visible Mega Kangaskhan 756 + Dwebble/Crustle 344/345
signature. This cleanly distinguishes them from our Great Tusk mirror. A strict
conjunctive exception against router v8 screened:

    TD12 +10 (11/1), TD13 +11 (20/9), TD18 +18 (22/6), TD26 +8 (16/8)
    pooled 1467/1420, discordance 69/24, p=0.000005

Mirror/Ogerpon/Garchomp/Alakazam safety was exactly invariant 746/746, 0/0.
This is the strongest post-v8 coordinate; large disjoint confirmation is
running before a router-v9 promotion.

The 300-seed-per-variant disjoint confirmation passed on every row: TD12 +10
(14/4, p=.0339), TD13 +19 (35/16, p=.0117), TD18 +26 (42/17, p=.00178),
TD26 +15 (26/12, p=.0350); pooled 2173/2103, discordance 117/49, p<1e-6.
Promoted to `agent/router_v9/`. A fresh 27-deck direct comparison against v8 is
running before packaging/submission consideration.

The 27-deck fresh gate passed: router v9 3618 versus v8 3589, discordance 34/6,
p=0.00001963. All 23 unaffected decks were exactly invariant. Target deltas were
TD12 +1, TD13 +11, TD18 +9 and TD26 +8; none was negative. Compile/import,
callable-agent and 60-card checks pass. Frozen archive:
`artifacts/router_v9.tar.gz`, sha256
`b87edc829961f6231167f97a99a6b198203f0b6bf8e01ce5a09166adc69fe24d`.
Router v9 is the new local champion. No submission was made.

Post-v10 Ogerpon wall search: forcing wall on visible card 96 alone improved
TD00/08 but harmed TD07, so the broad rule was rejected. A strict conjunction
requiring visible Ogerpon 96 + Tera Orb 1127 (unique to TD00/08) was exactly
invariant on TD07/15/20/22 (1199/1199, 0/0). Target discovery was +10; an
800-seed confirmation was 3077/3061, discordance 37/21, p=.0489. Promoted as
`agent/router_v11/`; full 27-deck stacking gate is running.

The fresh 27-deck v11-v10 gate was +6 (5360/5354, 10/4, p=.181), with all 25
non-Oger rows exactly invariant and the delta confined to TD00. This broad panel
alone is underpowered for the narrow rule, but the independent 400+800-seed
target panels combine to +26 with 60/34 discordance and exact safety. V11 remains
a valid modest promotion, not a major score shift. Frozen archive:
`artifacts/router_v11.tar.gz`, sha256
`613ee2014984b2775b651923c3f29cbee65b328bce7b414d552463e744352811`.

Final anchored v10 calibration (v11 adds only a modest validated residual) was
4445/4161 versus exact submitted-844 behavior over 5,400 games, discordance
393/129, p effectively zero. An odds-ratio translation gives an approximate
+56.7 rating-point center, or ~901 from the 844.4 anchor. Using observed
same-agent draw noise gives an estimated one-draw chance of ~81-93% to clear
856, and ~93-99% to clear the latest 826. Two independent draws raise the chance
at least one clears 856 to ~96-99.5%. These are calibration estimates, not
guarantees.

Mega Venusaur TD20 wall modes did not add value: no-wall 205/199 (15/9, p=.31),
force-wall 199/199 (3/3). Rejected. Router v11 remains champion.

Fros-specific wall screen: force-wall was harmful (815/840, p=.042). No-wall
looked +30 on discovery (870/840, p=.069) with exact four-family safety, but
failed a 500-seed disjoint confirmation at 1697/1691, discordance 198/190,
p=.722. **Rejected; do not stack it. Router v11 remains champion.**

Slowking-specific wall screen succeeded. Force-wall was harmful (359/404,
p<1e-5); no-wall passed discovery at 439/404 (62/31, p=.00187) and a 500-seed
disjoint confirmation at 445/400 (69/24, p=.000005). Alakazam/Grim/Fros/mirror
safety was exactly invariant 746/746, 0/0. Promoted as `agent/router_v12/`; a
fresh 27-deck v12-v11 stacking gate is running.

The 27-deck fresh stacking gate was +2 (5425/5423, TD11 116/114), with all 26
unrelated decks exactly invariant. This third Slowking seed panel was flat but
nonnegative; the two independent 500-seed target panels remain strongly positive
at +35 and +45. Combined direction is robust but heterogeneous by seed band.
V12 remains the narrow local champion. Compile/import/60-card checks pass.
Archive: `artifacts/router_v12.tar.gz`, sha256
`07a7123444fc99937e08233a1ac51761c8428ece1326e02d868e3511ce2faa4f`.

Final end-to-end anchored calibration used 150 fresh seeds x both seats x all
27 decks (8,100 games): v12 6716 versus exact submitted-844 behavior 6349,
discordance 557/209, p effectively zero. Twenty-one rows improved and six tied;
none was negative. Odds-ratio translation gives a ~+50.6 point center, about
895 from the 844.4 anchor. Under observed rating-path noise, estimated chance to
clear fixed 856 is ~78-90% for one draw and ~95-99% for two independent draws.
Thus only the sequential two-draw strategy meets a strict >=95% probability bar.

Final Grim-only wall audit across TD01/14/17/25: no-wall was harmful at
2175/2197 (71/96, p=.063); force-wall was flat 2200/2197 (67/68). Four-family
safety was exactly invariant. Rejected; v12 stays unchanged.

Final archive audit: `artifacts/router_v12.tar.gz` extracted into a clean temp
directory; main sha256 matched
`42924240f9bd83d87e5f353e222fad7c7a78ff673e1018386e3e0556d1c3074e`,
deck contained exactly 60 cards, agent import/callability passed, Linux cg
runtime/API were present, and 10 isolated engine games across Ogerpon, Grim,
Alakazam, Froslass and Slowking completed without an exception. Packaging risk
is cleared. No Kaggle write was made.

V10 screen: broadening the new conjunction to visible Mega Kangaskhan 756 alone
was exactly invariant on TD12/13/18/26 (1112/1112) and unexpectedly improved
TD07 by 20 wins, 279/300 versus v9 259/300, discordance 25/5, p=0.000523.
TD15, TD22 and the exact mirror were invariant. A disjoint TD07 confirmation is
running; do not promote from the screen alone.

The 300-seed disjoint TD07 confirmation passed decisively: 570/509,
discordance 63/7, p<1e-6. TD15, TD22 and exact mirror were exactly invariant
(1649/1649 combined, 0/0). Promoted to `agent/router_v10/`; a fresh 27-deck
v10-v9 gate is running before packaging/submission consideration.

The 27-deck gate was positive 3583/3570, discordance 23/11, p=0.05923; TD07
was +11 and no row was negative. Because visible 756 also occurs in Slowking
and Mega Venusaur, a 500-seed-per-family leakage audit followed: Slowking
429/425 (11/7) and Venusaur 231/228 (29/26), both flat/nonnegative. V10 is
locally safe. Archive: `artifacts/router_v10.tar.gz`, sha256
`b9cac4b58179e9ddcaa849c70462366e7de7bd61b05273d5c98f6aa53a2edd57`.
A direct 27-deck calibration versus exact submitted-844 behavior is the final
gate for estimating the chance of clearing the live 856 score.

That anchored calibration passed overwhelmingly on 100 fresh seeds x both
seats x all 27 leaderboard decks (5,400 games):

    router v10 4445  exact-844 behavior 4161
    seed discordance 393/129, McNemar p effectively 0

Twenty rows improved, six were exactly invariant, and only TD13 was negative
(-2, 5/7, p=.77). Major gains included Grim +19, mixed Fros +12, pure Fros +25,
TD07 +19, Festival +28, TD12 +14, Rocket +33, and alternate Grim lists +26/+37.
This proves v10 is materially stronger locally than behavior anchored to 844.4.
Absolute leaderboard clearance of 856 still includes rating-path variance; do
not confuse the near-certain relative promotion with a guaranteed single-draw
public score.

Against the only competitive refreshed public head-to-head (fixed-metal v15),
disabling v10's Archaludon terminal-mill route scored 199/400 versus current
v10's 206/400 (discordance 45/53, p=.48). It did not help and was rejected.

The parallel Lucario evolutionary run was stopped at saved generation 141:
zero promotions and minimum mutation scale 0.12. It produced no candidate and
its CPUs were reassigned to the calibrated router branch.

The agent is a clone of a public notebook. Verified byte-identical: our `main.py`
matches its `MAIN_SOURCE` (sha f31eba2e819ee2b3), and the kernel attaches no
datasets (`dataset_sources: []`). **There is no hidden data to add** — the
`top20_decks/` and `alak_w.json` load paths in its source are dead code in the
author's own submission too.

Its author reported ~950. The identical code scores 768.9 for us. The gap is
field drift plus LB noise, not a missing artifact.

---

## 2. The ONE thing that worked

**Search override margin: 500 → 3000.**

The agent's `_search_decide` accepts the search's move over the heuristic's top
choice only when it wins by a margin. Instrumented on a live game, the stock 500
fires **once in 43 decisions** — the agent is ~94% pure heuristic.

    panel A (public-* opponents)   37/14 discordant   p=0.0021
    panel B (meta-*, disjoint)     53/32              p=0.0301
    pooled                         90/46              p=0.00023   net +4.2%

Ladder confirmed: 777.1 vs 768.9 for byte-identical stock.

**Mechanism:** the search's *marginal* overrides are net-harmful; its
*high-confidence* ones help. Disabling search entirely is NOT better (49/50), so
the gain comes from filtering, not suppressing. Curve saturates ~6000 (no
override ever exceeds that margin).

This was found by **instrumenting** the agent, not by search or tuning.

---

## 3. Measured and DEAD — do not repeat

### CORRECTION (2026-08-08): card-identity bug voids a "dead" entry

`Option.cardId` is **None for PLAY and EVOLVE options** — the engine identifies
the card by `option.index` into the HAND area:

    card = _get_card(obs, AreaType.HAND, option.index, yourIndex)

Both `lucario_policy.py` and `ogerpon_policy.py` read `o.cardId` directly, so
**every card-specific PLAY/EVOLVE branch was unreachable** and both were measured
as generic policies. Verified: 0 of 148 (Lucario) and 0 of 28 (Ogerpon) PLAY
options had a non-None cardId.

Consequences:
* the "card-specific Ogerpon weights" negative is **VOID** — never actually tested;
* the SPSA and memetic runs that tuned those weights were optimising parameters
  with **no effect on play**, which plausibly explains why they found nothing;
* `archaludon_policy.py` and `tuned_policy.py` use the hand-index form and are
  unaffected.

Both files are now fixed. Re-testing Lucario after the fix still gave 8.3% vs the
fork, so the fix does not by itself rescue a weak policy — but any future
card-specific policy MUST resolve identity via the hand index, and any past
result involving one should be treated as unmeasured.


| Approach | Result |
|---|---|
| Multi-turn lookahead (depth 2 and 4 past the 2-turn horizon) | flat; changes ~55/600 outcomes, wins as many as it loses |
| Leaf evaluator: card advantage / evolution stage / board width / frac-HP / deck-out / all | all flat or worse vs control, tested at a margin where search actually votes |
| Archetype belief templates (32 mined decks) | **inert** — discordance identical to the noise floor |
| Search width: N_DET 3→24, K_OPP 3→8 | flat; costs 1% of the episode compute budget, buys nothing |
| Weight evolution, 69 weights | 337 combined generations (local + Kaggle), no confirmed promotion |
| Weight evolution focused on the 48 coordinates the author never moved | nothing — that hypothesis is tested and negative |
| Lucario policy evolution (34 weights) | 307 generations, 7 automatic restarts, nothing |
| Deck: single-concept transplants (tech_basics, hammer_pkg, dunsparce_line, mine_over_dudu) | tech_basics worse p=0.039; hammer_pkg worse p=0.008 (Lillie's + Neutralization Zone are load-bearing); others flat |
| Deck: wholesale swap to M Sato's LB-1170 list | worse, 102 vs 125 |
| Deck: elite lists (Mega Lucario 75.9%) under our generic policy | **4.2%** vs the fork — a strong deck in weak hands loses badly |
| Imitation / card-identity action ranker from replays | rejected (covariate shift) |
| Board-value critic from replay outcomes | AUC 0.812 offline, p=0.0021 **worse** in play |
| SPSA, memetic tuning, gated router, scaled damage model, 2-ply minimax | all rejected earlier; see EXPERIMENTS |

**Pattern:** every attempt to make the search *better* failed. The only win was
making the agent *trust it less*.

---

## 4. Harness rules — violate these and your results are noise

1. **Two opponent families are NOT deterministic**: `hybs-*` and `td-*` (backed by
   our search agent) return different results for the same seed. They were in most
   early panels, and a large share of "discordant seeds" was the opponent
   disagreeing with itself. Use `meta-*` and `public-*` only.
2. **Every battery needs a stock-vs-stock `control` arm.** Judge results against
   it, never against zero. The floor is ~6 discordant per 600 seed-units at the
   stock margin, and rises to ~31 at margin 0 (more search participation = more
   RNG-sensitive decisions).
3. **~1.4% residual nondeterminism** of unknown origin remains. Not an order
   effect (swapping play order does not flip it), not opponent state (rebuilding
   the opponent per arm does not remove it). Effects smaller than ~10 discordant
   per 600 are unresolvable.
4. **One panel is never a verdict.** `tech_basics` read 47/31 favourable on one
   panel and 52/60 unfavourable on another. Always confirm on disjoint seeds AND
   a different opponent set.
5. **A tuner's own gate is not validation.** "Champion B" passed nine internal
   promotions then lost 54-66 on fresh seeds — 69 dimensions judged on 16 games.
6. **Local testing cannot estimate absolute ladder score.** It compares A vs B
   reliably. Byte-identical agents scored 768.9 and 720.7 for us, and 940.7 vs
   790.8 in discussion 712621.

---

## 5. Traps that cost real time here

* `pgrep -f` / `pkill -f` match the **invoking shell's own command line**, and any
  log-tailing monitor that mentions the target. Six self-kills. Never put a
  process-matching command and the target's name in the same call.
* Evolution helpers that run in pool workers AND in the main process for the
  confirmation duel need lazy init — otherwise they crash **only when a candidate
  succeeds**, which reads as "nothing was found".
* The 1/5th success rule shrinks the mutation step on failure, so a run with no
  promotions anneals into a local point and stalls. Needs a floor plus restarts.
* Checkpoint-stored `scale` overrides the env var on resume.
* Screen loosely, confirm strictly. A racing gate of 9-in-36 rejected every real
  candidate before it could be tested.

---

## 6. Open leads (unproven)

* **Optimal margin value.** 500/1000/3000/6000 sampled; panels disagreed
  (m3000 best on A, m1000 on B). The completed 120-seed four-opponent fine
  sweep confirmed m3000 as the only significant arm: 817/797, discordant
  44/25, p=0.0302. Control noise was 2/3; m1000, 1500, 2000, 2500 and 4000
  were not significant. This independently supports retaining m3000, but does
  not create a stronger candidate than the submitted Grim v2 or router v5 lead.
* **A fork-quality policy for an elite deck.** The elite replay data is
  unambiguous: among players rated ≥1150, Mega Lucario ex wins **75.9%** (83
  games) while the Alakazam archetype we run wins **48.8%** (170 games); the same
  pilot appears on lists at 75.9%, 52.8% and 36.7%, so it is the deck, not the
  pilot. But the deck is uncashable without a policy of comparable quality — ours
  scores 4-8%. This is the only lever with a plausible path above ~850, and it is
  a multi-day build.
* **Submission variance.** Identical agents spread 48 points for us, 150 in the
  public discussion. Duplicating a validated agent across both slots is a free
  option on that spread.

---

## 7. Key files

    agent/fork/fork_main.py        the cloned agent (69 card-specific weights)
    agent/lucario_policy.py        first-draft Mega Lucario policy (8% strength)
    agent/elite_decks.json         deck lists mined from 1150+ rated replays
    tools/crn/paired_eval.py       CRN harness; play() is seeded and reproducible
    tools/crn/margin_battery.py    the sweep that found the one working lever
    tools/crn/clean_battery.py     generic A/B battery with a control arm
    tools/crn/evo2.py              population evolution over the 69 weights
    tools/crn/evo_policy.py        same, for any policy module
    tools/crn/validate_full.py     two-axis gate: mirror AND diverse field
    tools/crn/runner.py            work-queue daemon; queue.txt is live-editable
    tools/crn/determinism_scan.py  which opponents are reproducible
    EXPERIMENTS_2026-08-04.md      full chronological log with all raw numbers

---

## 8. Honest assessment

The old Alakazam public baseline still plausibly tops out around 800-850. That
statement no longer applies to the separately owned router lineage: router v8
is a decisive broad promotion over behavior anchored to an actual 844.4 row and
is now the best evidence-backed route toward 900. Absolute calibration remains
uncertain because simulation ratings are noisy and path-dependent. **1200 is not
credible from these narrow router residuals**; that still requires a genuinely
better policy on an elite deck.
# 2026-08-08 evening — target raised to high-probability >950

- User explicitly raised the ship/recommendation gate from beating 856 to a **high probability of beating 950**. Do not recommend or submit `router_v12` on its current evidence: its rating translation is centered near 895 despite a decisive 8,100-game gain over the exact 844.4 submission behavior.
- No submission is authorized; local validation only until the user explicitly says otherwise.
- Small visible-state router residuals are no longer a plausible route to the required +106 rating jump. Work switched to reconstructing elite policies from the official public replay data, which contains observations, legal option lists, and chosen action indices.
- Added `tools/rank/audit_elite_replays.py` to measure action-level coverage by elite team and modal deck across `data/episodes`, `data/ep_2026-08-01`, and `data/ep_2026-08-02` (13,824 replay files, roughly 63 GB). Audit output is `scratchpad/elite_replay_audit.tsv`; scan was started and remains CPU-active at the time of this note.
- Intended design: single-player/single-archetype filtered behavior cloning, game-level holdout, then confidence-gated hybrid use with the existing policy/search as off-distribution fallback. This directly addresses the prior generic clone's failure from mixing decks and compounding errors.
- Elite replay audit completed (`scratchpad/elite_replay_audit.tsv`). Best usable coverage is Majkel1337, LB 1105.1: 1,518 games / 75,098 decisions overall; exact modal Ogerpon deck has 579 replays. Exact deck saved as `data/majkel_ogerpon_deck.json`.
- Added parallel exact-target extractor `tools/rank/extract_target_parallel.py`; extracted 20,964 decisions from 580 exact-deck games to `data/majkel_ogerpon_all.pkl`.
- Strict 20% whole-episode holdout for first player-specific ranker: best top-1 57.22%, top-3 about 85%, versus option-0 baseline 56.25%. Pure cloning is not safe.
- `agent/majkel_ranker_candidate` (ranker top-3 prior + forward search) was decisively rejected against router v12: 41-119 over 160 both-seat games, seed discordance 5/44, p effectively zero. `agent/majkel_ogerpon_domain` was worse: 20-300 over 320 games. Neither is a candidate.
- Found/fixed a representation bug in `tools/rank/extract_dataset.py`: PLAY/EVOLVE/ATTACH frequently omit `cardId` and identify cards by hand index. Re-extracted resolved data as `data/majkel_ogerpon_resolved.pkl`; zero-card option identities fell from 154,810 to 49,503, but whole-game-holdout top-1 still only reached 57.01% vs 56.25% baseline. This simple architecture remains rejected; richer full-state representation or a different route is required.
- Added full-state replay tokens (own visible hand, both boards, discard identities, HP/energy, stadium) and `tools/rank/train_state_ranker.py`. Dataset `data/majkel_ogerpon_state.pkl`, checkpoint `agent/majkel_ogerpon_state_ranker.pt`. Strict whole-game holdout peaked at 57.34% top-1 vs 56.25% option-0 baseline (+1.09pp), then regressed; too small to promote and not arena-tested. Player-specific replay cloning of this form is rejected for now.
- Built `agent/router_v13_search`: conservative v12-policy rollouts with a Great-Tusk-specific leaf value (mill, wall survival, terminal wins) rather than the generic prize/damage value. Initial 80-game screen was +8 but noise; disjoint 480-game panel lost 228-252 (33/45 discordant, p=.213). Higher margins were flat: m3000 157-163, m5000 161-159. Reject this search branch.
- Added `tools/crn/router_deck_evo.py`, a legal protected-core population search over v12 deck counts, evaluated in both seats against four deterministic opponent families with a separate confirmation gate. Two independent 60-generation runs (`scratchpad/router_deck_evo_run1.log`, `run2.log`; checkpoints alongside) were started with different RNG seeds. No promotion in the first five generations observed at this note.
- Expanded a third evolutionary population with 12 policy-understood novelty cards (Hand Trimmer, Enhanced Hammer, Energy Lasso, Handy Circulator, recovery/disruption, etc.), checkpoint/log `router_deck_evo_novel.*`. At latest check: count run1 gen13, run2 gen8, novelty gen5; best tuning margin only +4 versus the required +13 screening gate, zero promotions. All three processes remain active; do not mistake their screening margins for validated gains.

## 2026-08-08 late evening — continued evolution and public archive audit

- The first three deck-evolution searches all completed 60 generations (180 total) with **zero confirmed promotions** and byte-identical champion deck. This is strong evidence that 1–3-card local count mutation around router v12 is saturated under the hardened gate.
- Started two relaxed-screen searches while retaining the strict disjoint confirmation: `router_deck_evo_racing.*` uses 1–3 mutations and `router_deck_evo_jump.*` uses 4–12-card jumps, each scheduled for 100 generations. At gen ~26 both had zero promotions. Screening margins reached +8, but candidates failed confirmation; do not promote them.
- Audited fresh public notebooks. `stanislav` is another fixed Metal policy; the newer `sgzk` Lucario lost 85–115 to router v12 over 200 games (discordance 20/35, p=.059); Tetsu's latest Grim package crushes router v12 head-to-head 145–55 but has only ~809 public-team rating and benefits from a favorable Great-Tusk matchup. None qualifies.
- Tested Tetsu's distinct deck delta on our stronger Grim v2 policy: replace one Rare Candy with Tool Scrapper. Broad 400-game panel was 343 wins vs 337 for base; discordance 28/24, p=.6774. Rejected as flat (`scratchpad/grim_candy_scrap_screen.log`).
- Cloned the public experiment archive to `references/public_sim_repo`. Historical peaks include experiments 36 (Alakazam 1027.6), 55 (Archaludon 1030.6), 173 (Steel 966.4), and 192 (Garchomp 961.3), but the records show extreme rerun variance. Experiment 206 explicitly reports six **byte-identical** Garchomp rows spanning 961.3 down through much lower scores. A historical peak therefore does not establish high probability >950.
- Added generic source-directory loading to `tools/public_agents.py` and registered the exact exp36/exp55 source snapshots as `public-naoto-1027` and `public-arch-1030` in `tools/arena.py`, plus `router-v12` for direct CRN comparison. Current-meta Grim panels are running to test their exact bytes rather than trusting historical labels.
- Exact-source results: exp36 Naoto/Alakazam was flat to v12 against meta Grim (39/40 vs 38/40, only 2/1 discordant); exp55 Archaludon trailed v12 against public Alakazam (58/100 vs 73/100, discordant 10/20, p=.1003). Registered the exact locally downloaded Garchomp v28 package too; it trailed v12 64/100 vs 70/100 against public Alakazam (13/19 discordant, p=.3768). These historical >950 labels do not beat the current champion locally.
- No submission was made. The recommendation gate remains high probability >950; no candidate has cleared it yet.

## 2026-08-08 22:xx — status restart

- Both relaxed router deck searches completed all 100 generations with zero confirmed promotions. Together with the earlier 180 generations, deck-count evolution has produced zero promotions across 380 generations; close this branch unless the policy or objective changes materially.
- Started two fresh 120-generation, population-12 policy-genome searches at the validated 3000 search margin: `scratchpad/evo2_focus_aug8.log`/checkpoint `evo2_focus_aug8.json` focuses mutations on 48 public-weight coordinates that were never optimized; `scratchpad/evo2_global_aug8.log`/checkpoint `evo2_global_aug8.json` uses global mutation and crossover. Both use 12 workers, fresh RNG streams, successive halving, restarts, and a 9x disjoint promotion confirmation. Candidate outputs, only if promoted, are `agent/fork/alak_w_evo2_focus_aug8.json` and `alak_w_evo2_global_aug8.json`.

## 2026-08-08 — neural branch authorized

- User explicitly authorized the local RTX 4060 and Kaggle GPU if local compute is insufficient. This authorizes training kernels/compute, **not a competition submission**.
- Stopped the two 24-worker combined evolution processes because they oversubscribed the 16 local CPU cores and would starve neural environment stepping. Their logs/checkpoints remain; neither had reached a generation result yet.
- Discussion-derived target design: fixed strong deck; elite-replay BC warm start; masked entity/action policy plus value and auxiliary tactical heads; checkpoint league containing v12, Grim v2, exact public agents and generated exploiters; held-out deck families for local validation; PPO only after BC/arena competence. This is materially different from a copied public notebook.
- Existing AlphaZero-style trainer is known weak (old checkpoints only 18–26% versus v3), uses one sample deck for every policy, and lacks the required league diversity. Do not launch a long run unchanged. A tiny RTX 4060 pipeline/throughput smoke is running to verify CUDA + native engine integration before rebuilding it (`scratchpad/rl4060_pipeline_smoke.pth*`).
- RTX 4060 smoke completed successfully end-to-end (native games, MCTS targets, CUDA update, checkpoint save).
- High-coverage replay discovery: the exact public Grim deck (the 3 Rare Candy + Tool Scrapper list) is shared by many 1000–1180-rated teams. Extended `tools/rank/extract_target_parallel.py` with deck-only multi-teacher mode, LB floor, CSV deck input, and winner-only filtering. Extracted **243,872 decisions from 3,359 games** at LB >=1000 to `data/elite_grim_multi_state.pkl`, ~11.6x Majkel's 20,964-decision set.
- Extended `tools/rank/train_state_ranker.py` with nonzero-action weighting and a calibrated high-confidence override gate. The deployment intent is not pure cloning: retain the strong Grim heuristic by default and use the network only when it confidently predicts an elite deviation from option 0. Two 12-epoch CUDA fits are running sequentially: `elite_grim_gated_w1.pt` (natural distribution) and `elite_grim_gated_w2.pt` (2x nonzero weight), logs in `scratchpad/elite_grim_gated_*.log`. Required validation metric is >=80% precision on game-disjoint nonzero overrides with useful coverage, followed by arena validation.
- First fit exposed a reporting-only bug: validation batches have different maximum legal-option widths, so concatenating their full logit matrices failed after training. Fixed gate reporting to concatenate fixed-width `(prediction, margin, target)` vectors instead. Stopped the second doomed fit and restarted both; retry logs are `scratchpad/elite_grim_gated_w{1,2}_retry.log`. No checkpoint was promoted from the failed reporting run.
- Added a live local UI at `http://127.0.0.1:8765/live.html`. `automation/dashboard_server.py` now exposes `/api/live` with active research processes, GPU status, recent training logs, artifacts/dataset sizes, and the tail of HANDOFF; the page refreshes every 5 seconds. Server stdout/stderr and PID remain under `automation/state/`.
- First corrected w1 epoch: game-holdout top-1 47.78%, top-3 83.58%; an >=80% precision gate can make 4,132 nonzero overrides (8.36% validation coverage). This is promising offline evidence only, not an arena promotion.
- Added `tools/rank/generate_winner_trajectories.py`: controlled 2/5/10% legal-action exploration around the strong Grim policy against six frozen opponent families, both seats. It records candidate observations/actions only from games the explorer wins and tags seed/seat/opponent/epsilon. Smoke: 1,123 decisions from 18 won games. Full round 1 is running with 100 seeds x 2 seats x 6 opponents x 3 exploration rates = 3,600 games on 4 CPU workers; output `data/selfplay_grim_winner_round1.pkl`, log `scratchpad/selfplay_grim_winner_round1.log`.

## 2026-08-09 — overnight neural/data results

- Both 12-epoch RTX 4060 fits completed. Natural weighting (`elite_grim_gated_w1.pt`) is the offline winner: best game-holdout top-1 **52.60%** versus option-0 **41.09%** (+11.51pp), top-3 around 84.9%. At the calibrated >=80% precision gate it covers roughly 13.6% of validation decisions (6,732 overrides at its best epoch). The 2x nonzero fit peaked at 50.38% (+9.29pp) with ~14.1% gated coverage; keep it as a diversity checkpoint, not the primary.
- Self-generated round 1 completed all 3,600 league/exploration games and saved **71,172 winning decisions from 1,005 won games** to `data/selfplay_grim_winner_round1.pkl` (120 MB). This is successful-trajectory data, not yet proof of policy improvement; it must be mixed carefully and evaluated on held-out seeds/opponents.
- Dashboard remains healthy at `http://127.0.0.1:8765/live.html`. GPU is currently idle (~4%, 796 MiB), so the next compute step is available.
- Immediate next gate: package w1 as a conservative Grim-v2 neural override candidate, verify exact inference parity with training features, then run disjoint CRN panels against Grim v2/router v12/held-out public agents. Do not begin PPO or recommend submission until this arena gate passes.
- Status audit found an idle gap after overnight completion: dashboard remained alive but no candidate/arena job was running. Added `agent/elite_grim_override.py` and registered 80%-gate, strict (margin 3.0), and ultra-strict (4.5) wrappers over the controlled local Grim baseline.
- Live smoke against public Alakazam is not promotional. Default offline-calibrated gate lost 4/40 vs baseline 9/40 (discordant 0/5). Strict gate was exactly 13/60 vs 13/60 (3/3 discordant); ultra was 6/60 vs 7/60 (1/2). Conclusion: 80% replay-holdout precision does not transfer to live off-distribution states; stricter gating removes the harm but also the gain. Do not integrate into Grim v2 or begin PPO from this checkpoint yet.
- Data generation round 2 started on fresh seeds 980000+ with lower, less noisy exploration rates 0.5%/1%/2%. Scope: 120 seeds x 2 seats x 6 opponents x 3 rates = 4,320 games on 4 workers. Output/log: `data/selfplay_grim_winner_round2.pkl`, `scratchpad/selfplay_grim_winner_round2.log`. Extended generator with `--epsilons`. Round 1 remains complete and immutable.
- Round 2 completed: **93,643 winning decisions from 1,336 won games**. Added persistent `automation/continue_silver_pipeline.py`: waits for round 2, builds an elite-majority mixture (all 243,872 elite decisions + 30k sampled from each self-play round), trains `agent/elite_grim_dagger.pt` at an 85%-precision gate, then automatically runs 100-seed both-seat held-out panels vs public Alakazam, public Archaludon, and router v12. It never submits. Controller state/log: `scratchpad/silver_pipeline_state.json` and `.log`.
- User target for 2026-08-09: pursue a credible 950+ candidate today, potentially submit tomorrow morning only when very confident. Preserve the explicit approval gate: preparing an archive is allowed; do not submit until the user reviews the evidence and explicitly authorizes it tomorrow.
- Elite-majority DAgger pipeline completed. Offline metrics were strong: top-1 57.64% vs 43.69% option-0 (+13.95pp), top-3 ~86.9%, and ~14.6% coverage at calibrated 85% override precision. Live transfer nevertheless failed: vs public Alakazam 24/200 vs baseline 30/200 (3/9 discordant), public Archaludon 16/200 vs 20/200 (4/8), router v12 32/200 vs 37/200 (9/14). All three trend negative; reject `elite_grim_dagger.pt`.
- Immediately started a different candidate chain: train only on round-2 low-exploration successful trajectories (`selfplay_grim_winner_round2.pkl`) at a stricter 95% gate for 16 epochs, output `agent/selfplay_grim_loweps.pt`, then automatically run fresh 240-game panels against public Alakazam, public Archaludon, and router v12 (seeds 993000+). Logs: `scratchpad/selfplay_grim_loweps_train.log` and `loweps_vs_*.log`. No submission action exists in this chain.
- Low-exploration model completed with spectacular but misleading in-distribution metrics: 91.88% top-1 vs 57.04% baseline, ~98.1% top-3, ~26% coverage at 95% precision. Live results were flat/negative: Alakazam 33/240 vs 33/240, Archaludon 23/240 vs 22/240, router v12 25/240 vs 29/240. Reject; winner-only imitation mostly reproduces its generating policy.
- Added causal data mode to `generate_winner_trajectories.py`: for each exact seed/seat/opponent, first play epsilon=0 baseline; retain a 1%/2%/5% exploration trajectory only when baseline loses and explorer wins. Smoke produced 45 decisions from one causal win. Full round 3 is active: 200 seeds x 2 seats x 6 opponents, 6 workers, output `data/selfplay_grim_causal_round3.pkl`.
- A separate persistent continuation session waits for causal round 3, then trains `agent/selfplay_grim_causal.pt` for 20 epochs at a 95% gate and automatically runs fresh 300-game panels against public Alakazam, public Archaludon, and router v12. Logs use `scratchpad/selfplay_grim_causal_train.log` and `causal_vs_*.log`. No submission step.
- Causal round 3 completed: 25,839 decisions from 330 matched causal wins. The 95%-gate causal model peaked at 83.90% top-1 vs 56.11% baseline but its saved live gate made zero overrides: exact invariance across all three 300-game panels. A lower fixed margin 4.5 follow-up is active on fresh 240-game panels (`causal45_vs_*.log`).
- Overnight compute allocation: local causal45 validation active; dashboard active. Private Kaggle GPU DAgger oracle v2 is RUNNING and has an automatic completion/output pull monitor. Private Kaggle CPU exploitability v1 lacked its asset, failed, and corrected v2 was automatically pushed and is RUNNING with an output monitor. A prioritized-failure GPU kernel is queued by a monitor to push as soon as DAgger frees the GPU slot. Kaggle rejected the immediate second GPU push only because the account was already at the two-session limit. None of these kernels contains a submission step.
- Data-quality warning: rounds 1/2 are winner-filtered, not causal proof, and should not be described as guaranteed superior data. Round 3 is better (matched baseline loss/explorer win) but retained whole trajectories across six opponents, so most stored actions are unchanged baseline actions and only one-sixth target v12.
- Extended generator rows with `baseline_y` and `explored`, added opponent selection, and added `filter_causal_corrections.py` to retain only actions actually changed in matched loss-to-win games. A v12-only round 4 is queued after validation: 500 seeds, both seats, matched epsilon=0 vs 1/2/5% explorers; outputs full causal trajectories and a corrections-only file.
- Causal model margin-4.5 screen: flat vs Alakazam 23/240 each; -1 vs Archaludon 16/240 vs 17/240; encouraging +9 vs router v12, 44/240 vs 35/240 with 13/5 discordance (p=.099). Not significant. A disjoint 400-seed/800-game v12 confirmation is active first (`causal45_vs_router_v12_confirm400.log`); only a confirmed result can promote this lead.
- Added the large quality-feedback queue. `build_quality_replay.py` deduplicates state/action rows, caps overrepresented contexts, anchors every replay in the 243,872 elite decisions, and deliberately upweights only unique actions actually changed in matched baseline-loss/explorer-win games. `quality_feedback_queue.py` waits for v12 round 4, then runs four sequential 1,000-seed causal shards (rounds 5–8), filters changed actions, rebuilds a growing balanced replay, and trains a fresh 90%-gate checkpoint after every shard. State: `scratchpad/quality_feedback_state.json`; every stage has its own log. It contains no submission operation.
- Generator baseline is now configurable with `TRAJ_BASE`, enabling a future held-out-promoted checkpoint to become the next data-generating policy rather than endlessly cloning the original baseline. Promotion binding still requires significant arena evidence; failed checkpoints must not enter the generator.
- 400-seed v12 confirmation killed the apparent causal45 lead: 113/800 vs baseline 110/800, discordance 25/21, p=.6583. Reject as flat. V12-only round 4 produced 11,020 decisions from 139 matched causal wins, of which only 455 were actions actually changed.
- Quality rounds 5–8 completed and accumulated **20,624 unique causal changed actions** (round contributions 5,039 / 5,292 / 5,092 / 5,203) plus the v12 corrections. Repeating corrections 16x was too aggressive: replay grew 319k→563k but 90%-precision override coverage collapsed from 0.3% to zero. Models are not promotable.
- Started correction-dose sweep at 1x/2x/4x. Each dose rebuilds a deduplicated elite-anchored replay, trains 14 epochs at an 85% gate, then automatically runs a fresh 300-game v12 panel. Active chain/logs: `quality_replay_dose*`, `quality_dose*_train.log`, `quality_dose*_vs_router.log`.
- All three overnight Kaggle experiments failed for infrastructure rather than model evidence. GPU DAgger and prioritized-failure landed on Tesla P100 (sm_60), while Kaggle's current torch build supports sm_70+ only (`CUDA no kernel image`). CPU exploitability failed one matrix arm. Outputs/logs were pulled; do not count them as experimental rejections. Local 4060 remains the valid GPU lane.

## 2026-08-09 — first live-positive neural candidates

- Correction-dose 1 completed. `agent/quality_dose1.pt` reached game-holdout top-1 **50.46%** versus option-0 **36.61%** (+13.85pp), with roughly 11.5% usable coverage at the calibrated 85%-precision gate. On a fresh paired 150-seed/300-game panel against router v12 it scored **72/300 (24.0%)** versus the controlled `ours-grimmsnarl` baseline's **41/300 (13.7%)**; seed discordance 48/19, McNemar p=0.0006.
- Correction-dose 2 is stronger. `agent/quality_dose2.pt` reached top-1 **49.36%** versus option-0 **34.42%** (+14.94pp), with roughly 9.6% gated coverage. On its fresh paired v12 panel it scored **75/300 (25.0%)** versus baseline **37/300 (12.3%)**; discordance 45/10, p effectively 0. This is the first decisive live local neural gain in the project.
- Interpret carefully: dose 2 approximately doubled this base Grim policy's win rate into router v12, but it still lost 75% of those games. It is therefore evidence that the causal-correction dataset is useful, **not** evidence that the neural candidate is stronger than router v12 overall or likely to score 950. It has not yet been compared directly with the submitted Grim Candy v2 behavior.
- Dose 4 training is active on the RTX 4060 (`agent/quality_dose4.pt` when complete), using a 315,352-row replay. A persistent post-sweep validation queue will run a larger disjoint v12 confirmation for dose 2 and safety panels against held-out public/meta opponents. No Kaggle submission is authorized or performed.
- Started the ceiling-breaking track: `TRAJ_BASE=router-v12` causal generation over 2,000 seeds x both seats x six opponent families (24,000 matched baseline groups, with 1/2/5% explorers only when frozen v12 loses). Output/log: `data/v12_causal_round1.pkl`, `scratchpad/v12_causal_round1.log`. This corpus is kept separate because router v12 and Grim Candy v2 use different decks.
- Added `agent/v12_neural_override.py` and the `v12-neural` arena entry. It preserves frozen router-v12 behavior by default and permits only confidence-gated single-action neural corrections. A persistent continuation filters actual changed loss-to-win actions, builds a v12-native 2x correction replay, trains `agent/v12_causal.pt` at a 90%-precision gate for 18 epochs, then runs paired panels against frozen v12 over six opponent families. Controller PID was 1673216; log `scratchpad/v12_neural_pipeline.log`. No submission step exists.
- Parallel-compute correction: the first 10-worker v12 causal launch died before producing data under dose-4's high RAM pressure. It was not counted as an experiment. Requeued a memory-safe six-worker pipeline to begin automatically after dose-4 exits (controller PID 1675958), followed by a separate six-opponent validation queue (PID 1676039).
- Started two private Kaggle GPU lanes in parallel: distributional-value v1 and synthetic-rare-states v1. Both were confirmed running; distributional-value subsequently entered ERROR while rare-states remained RUNNING, so the former is infrastructure/model-debug evidence only until its logs are pulled. Controller PID 1675957 polls both, downloads outputs, and attempts fresh private v2 runs immediately when the two slots free. Prior P100/sm_60 incompatibility remains a risk. There is no submission code in either kernel.
- Both Kaggle GPU lanes ultimately failed. Pulled distributional logs confirm the same environment failure as earlier: Kaggle assigned Tesla P100/sm_60, while the bundled PyTorch lacks sm_60 kernels; the first GELU call raises `CUDA error: no kernel image`. Do not count either as model evidence and do not retry unchanged. Began downloading a private offline Python 3.12 CUDA 12.4 PyTorch 2.6 wheel (`kaggle_remote/p100_torch_wheel/`) intended to restore Pascal/P100 support; next step is upload it as a private Kaggle dataset, add an accelerator smoke test, then launch corrected kernels. No competition submission.
- Clarification on the renamed v2 kernels: both v2 pushes failed at the Kaggle launch/API layer while the two v1 session slots were still occupied; neither v2 slug exists (`kernels status` returns 404), so no v2 model training occurred. The first background wheel transfer also died at zero bytes. Verified the official wheel endpoint (768,372,201 bytes) and restarted it as a managed resumable transfer (exec session 91444). Do not claim a corrected Kaggle run until this file is complete, hashed, uploaded privately, and passes an sm_60 CUDA smoke test.
- Pre-reboot checkpoint: dose 4 completed all 14 epochs and saved `agent/quality_dose4.pt`; best offline top-1 was **47.23%** vs 30.96% baseline (+16.28pp) at epoch 4. Fresh paired v12 panel: **70/300 (23.3%)** vs controlled Grim baseline **42/300 (14.0%)**, discordance 44/19, p=0.0025. It is live-positive but slightly weaker than dose 2's 75/300 screen and still does not beat v12 overall.
- P100-compatible wheel download completed at 768,372,201 bytes; SHA-256 `a393b506844035c0dac2f30ea8478c343b8e95a429f06f3b3cadfc7f53adb597`. The v12-native generation/controller did not produce `data/v12_causal_round1.pkl` and is no longer running; restart that pipeline after reboot rather than assuming it completed. At reboot time only `automation/dashboard_server.py` remained active, so reboot is safe for saved experiment state.
- Attempted to upload the verified wheel as private dataset `boltuzamaki/ptcg-private-pytorch-p100-wheel` before reboot, but Kaggle rejected dataset creation with `Authentication required to call the Kaggle API`. No upload/session remains active. Do not relaunch the known-incompatible P100 notebooks; after reboot refresh Kaggle OAuth/API-token authentication, upload the wheel, and run an sm_60 smoke kernel before full training.
- Kaggle OAuth refresh succeeded. Retried the private 733 MB wheel upload, but transfer throughput was only ~20 KB/s (roughly 10-hour ETA). Cancelled cleanly at 784 KB so the user could reboot; local verified wheel is intact. No Kaggle job or upload remains active.

## 2026-08-14/15 session — measurement rebuilt, agent unchanged

**Position:** 764.8, rank ~1159/6834. Bronze 837.0, silver 911.3.
Active pair = ogerpon_field07 (~684) + family_v1 (764.8); score is max(active),
so the failing Ogerpon slot costs nothing. 4 submissions in hand.

**The one durable deliverable: `tools/crn/public_panel.py`.**
Plays our agents vs REAL published agents piloting their own decks (13 resolvable
under `references/*`), paired CRN, one cell per subprocess.
Validated on 3 ladder points: family_v2_base 65.4%→794.3, fork_m3000 65.1%→777.1,
pkg_field_07 50.0%→~697. Correct order and magnitude — the first local metric here
that tracks the ladder.
  * Use the UNWEIGHTED average. Field-weighting by `field_decks_by_band.json` is
    refuted (predicts Ogerpon +26 over family_v2; ladder says −100).
  * Never use fewer than ~10 opponents: with 3, fork_m3000 read +24 over family_v2;
    with 13 the true gap is +0.3.
  * `band_eval.py` mispredicts cross-deck — it cost a submission on Ogerpon.

**Ogerpon search: 6 stacked bugs, all hidden by bare `except`** — raw str passed to
`search_begin`, ApiResult-vs-SearchState API mismatch, `your_deck=[673]*60`,
`evaluate_state` missing its seat arg, eager-default `len(None)`, and candidates
drawn from `choose()` which truncates to `maxCount`=1. It had never executed a
single step. Fixed → beam 10, 69% override → **83.7% → 51.4%**. The evaluator is
the weak component; do not revive this without replacing it first.

**Measured negatives (all on the validated panel unless noted):** deck tech 24 arms
(+0.8 vs ±0.8 control drift); gate margins 6 arms (byte-identical); MCTS/beam 8 arms
(byte-identical — dead code); fork margin 6 arms (2.1-pt band incl. control);
full screen 11 agents (1.5-pt band). **The grim family is at its ceiling — stop
screening variants of it.**

**Next lever must be a genuinely different deck+policy pair scored on
public_panel.py, beating 65.4%.** Nothing we own does.
