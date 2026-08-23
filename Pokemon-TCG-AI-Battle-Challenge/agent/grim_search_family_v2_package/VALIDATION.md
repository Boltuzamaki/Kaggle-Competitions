# Validation summary — family v2

## V2 improvement over submitted family v1

- Fresh gated Router v12: 900/1200 vs 892/1200 (`+8`), paired
  better/worse seeds 10/2, exact p approximately 0.0386.
- Independent visible-field Router: 510/600 vs 503/600 (`+7`), paired
  better/worse seeds 7/0, exact p 0.015625.
- Crustle-counter safety: 467/600 vs 467/600, zero discordant seeds.
- Fresh 12-family v2-v1 broad battery: 1997/2400 vs 1989/2400 (`+8`);
  no opponent family was negative.

The submitted family-v1 checkpoint is Kaggle row 55444198 (public score 719.8).
Family-v2 has not been submitted.

## Family v1 foundation

Baseline: frozen Grim Candy v2. All comparisons use common-random-number,
two-seat paired evaluation. No competition submission was made.

## Direct deployable-gate check

- Archaludon family: 919/1200 vs 894/1200 (`+25`), paired better/worse
  seeds 26/1, exact binomial p approximately 4.2e-7.
- Public Garchomp check: 119/200 vs 116/200 (`+3`).
- Alakazam, Router v12, Crustle, Grimmsnarl, and FrostWall safety field:
  897/1000 vs 897/1000, zero discordant seeds.

## Fresh final broad battery

Twelve opponent families, 100 fresh seeds and both seats per family:

- Candidate: 2108/2400.
- Grim Candy v2: 2098/2400.
- Net: `+10` wins.
- Nine non-target families had zero discordant seeds.
- Public Archaludon: `+7`; v3 Garchomp: `+5`; public Garchomp: `-2`.

## Component evidence accumulated before the final battery

- Garchomp gate across public, meta, and v3 implementations: `+53` wins;
  paired better/worse seeds 186/130, exact p approximately 0.00193.
- Archaludon 20k search across independent implementations before direct gate
  validation: `+27` wins; paired better/worse seeds 46/20, exact p
  approximately 0.00186.

These results establish a conservative aggregate improvement, not a guaranteed
Kaggle leaderboard score. The family gates deliberately preserve frozen Grim
behavior outside the two validated matchups.
