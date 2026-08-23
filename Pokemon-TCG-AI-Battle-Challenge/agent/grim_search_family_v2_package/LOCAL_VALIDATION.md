# Grim Candy v2 — local-only candidate

Status: promoted locally on 2026-08-08. **Never submitted.**

V2 retains the V1 deck and adds one public-state policy residual. Once the
existing matchup detector locks visible opposing cards as the Archaludon
family, arbitration selects the already-computed mirror expert action. All
other detected families remain exactly on the V1 path.

Policy-gate validation against public Archaludon, with Alakazam and Grim safety
controls:

| panel | gated | V1 | discordant gated/V1 | p |
|---|---:|---:|---:|---:|
| 100 seeds, seed0 9,100,000 | 495 | 475 | 33/13 | 0.0051 |
| 200 fresh seeds, seed0 9,500,000 | 997 | 977 | 59/41 | 0.0891 |
| combined Arch discordance | | | **92/54** | **0.0022** |

In both panels Alakazam and Grim outcomes were exactly invariant. Two alternate
Archaludon policies were flat (+2 pooled, 16/14 discordant), not adverse.

The required stacked confirmation used Candy v1—not the original public deck—as
the exact control. Over 150 additional fresh seeds, v2 scored 749/733 overall;
Archaludon was 215/199 (discordant 43/27, p=0.073), while Alakazam and Grim were
again exactly invariant. Thus the deck and policy improvements compose without
the interaction reversing.

## V1 deck receipt

This freezes the public Grimmsnarl control policy from
`tetsutani/grimmsnarl-ex-damage-transfer-control` plus one locally selected deck
change: remove the singleton Tool Scrapper (1137) and add the fourth Rare Candy
(1079). The policy source is otherwise unchanged; embedded active deck constants
were synchronized so the packaged selection deck is exactly 60 cards.

The mutation was selected from a 19-arm one-card coordinate screen, then tested
on two disjoint 300-seed confirmations against deterministic public Archaludon
and Alakazam policies, both seats:

| panel | candidate | public deck | discordant candidate/base | p |
|---|---:|---:|---:|---:|
| seed0 6,800,000 | 899 | 868 | 140/112 | 0.0890 |
| seed0 7,200,000 | 872 | 859 | 141/122 | 0.2670 |
| combined | 1771 | 1727 | 281/234 | **0.0427** |

Combined matchup deltas were Alakazam +45 and Archaludon -1. A separate
100-seed three-opponent confirmation was 496/490, with Grimmsnarl invariant.
The initial 10-seed mutation screen is excluded from the combined p-value.

Reproduce with `tools/crn/grim_deck_battery.py`. Local validation cannot prove
an absolute Kaggle skill rating; it establishes a repeatable improvement over
the public deck on the principal non-mirror upper-meta matchup.
