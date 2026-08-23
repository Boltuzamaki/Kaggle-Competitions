Pokemon TCG AI Battle - Fixed Metal v13

This package uses the reviewed Meta-A Archaludon/Cinderace deck and its
specialist policy for every episode.

The previous 2-Metal/1-Fighting selection was reevaluated after 22 new exact
opponent decks expanded the local library to 285. Across two certification
bands Metal scored 65.03% versus Fighting's 63.17%; a third unseen band
confirmed Metal at 63.48% versus Fighting at 60.67%. The fixed route therefore
adds a small, repeatable expected edge over the former mixture.

No unverified card swap or tactical override is included. Forward search
remains disabled because both our testing and the reviewed public reference
showed its shallow evaluator losing to the deck-specific heuristic. Existing
legal-action fallbacks, low-deck guards, matchup logic, and exact-score tie
handling are retained.

Public source attribution:
  aristophanivan/improved-probabilistic-agent
  llccqq624/ptcg-meta-a-stable-submit
  maulikgajera/the-pok-mon-ptcg-ai-battle-agent

Optional local diagnostic overrides remain available:
  PTCG_DECK=rank1
  PTCG_DECK=meta_a
  PTCG_DECK=water
