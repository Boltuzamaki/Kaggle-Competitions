"""Decks extracted from top public kernels of the Kaggle 'Pokemon TCG AI Battle'
competition (slug: pokemon-tcg-ai-battle).

Each deck is a flat list of exactly 60 integer card IDs (repeated per copy).
Card IDs decode via references/cg-lib  ->  cg.api.all_card_data().

Provenance per deck is noted in the comment above it. Counts were extracted
from the notebook decklist comments; totals were verified == 60.

NOTE on ARCHALUDON: masamikobayashi's kernel loads its 60-card deck.csv from an
external dataset that is NOT bundled with the notebook source, and the inline
docstring lists only the unique cards (not full copy counts, and missing the
Cinderace pre-evolutions). We therefore expose only the identified unique IDs
(ARCHALUDON_UNIQUE) rather than a fabricated 60-card list.
"""


def _expand(pairs):
    out = []
    for cid, n in pairs:
        out += [cid] * n
    return out


# ---------------------------------------------------------------------------
# kiyotah/a-sample-rule-based-agent-mega-lucario-ex-deck   (817 votes, #1)
# Archetype: Mega Lucario ex (Fighting) beatdown. Main attacker 678 (megaEX,
# HP340). Also used by romanrozen/strong-start-baseline-agent-v10 (LB 950+),
# which runs this same deck.csv (dataset kiyotah/mega-lucario-ex-deck).
# ---------------------------------------------------------------------------
MEGA_LUCARIO = _expand([
    (673, 2),   # Makuhita
    (674, 2),   # Hariyama
    (675, 2),   # Lunatone
    (676, 3),   # Solrock
    (677, 3),   # Riolu
    (678, 4),   # Mega Lucario ex  (megaEX, HP340, main attacker)
    (1102, 4),  # Dusk Ball
    (1123, 2),  # Switch
    (1141, 4),  # Premium Power Pro
    (1142, 4),  # Fighting Gong
    (1152, 4),  # Poke Pad
    (1159, 1),  # Hero's Cape
    (1182, 2),  # Boss's Orders
    (1192, 4),  # Carmine
    (1227, 4),  # Lillie's Determination
    (1252, 2),  # Gravity Mountain (stadium)
    (6, 13),    # Basic Fighting Energy
])


# ---------------------------------------------------------------------------
# kiyotah/a-sample-rule-based-agent-dragapult-ex-deck   (310 votes, #4)
# Archetype: Dragapult ex (Dragon) spread/multi-KO. Phantom Dive spreads
# damage to take >=3 prizes in a turn. Fire+Psychic energy base.
# ---------------------------------------------------------------------------
DRAGAPULT = _expand([
    (119, 4),   # Dreepy
    (120, 4),   # Drakloak
    (121, 3),   # Dragapult ex  (EX, HP320, main attacker - Phantom Dive)
    (140, 1),   # Fezandipiti ex
    (184, 1),   # Latias ex
    (235, 2),   # Budew
    (1071, 1),  # Meowth ex
    (1079, 2),  # Rare Candy
    (1080, 1),  # Unfair Stamp
    (1086, 4),  # Buddy-Buddy Poffin
    (1097, 2),  # Night Stretcher
    (1120, 4),  # Crushing Hammer
    (1121, 4),  # Ultra Ball
    (1152, 3),  # Poke Pad
    (1156, 1),  # Lucky Helmet
    (1182, 3),  # Boss's Orders
    (1198, 4),  # Crispin
    (1210, 2),  # Brock's Scouting
    (1227, 4),  # Lillie's Determination
    (1256, 2),  # Team Rocket's Watchtower (stadium)
    (2, 4),     # Basic Fire Energy
    (5, 4),     # Basic Psychic Energy
])


# ---------------------------------------------------------------------------
# kiyotah/a-sample-rule-based-agent-mega-abomasnow-ex-deck   (109 votes)
# Archetype: Mega Abomasnow ex (Water) "Hammer-lanche" beatdown. Beginner
# deck: very high energy count (34 Water), thin trainer line.
# ---------------------------------------------------------------------------
MEGA_ABOMASNOW = _expand([
    (721, 2),   # Kyogre
    (722, 4),   # Snover
    (723, 4),   # Mega Abomasnow ex  (megaEX, HP350, main attacker - Hammer-lanche)
    (1121, 4),  # Ultra Ball
    (1126, 1),  # Precious Trolley
    (1192, 4),  # Carmine
    (1227, 4),  # Lillie's Determination
    (1262, 3),  # Surfing Beach (stadium)
    (3, 34),    # Basic Water Energy
])


# ---------------------------------------------------------------------------
# ryotasueyoshi/rule-based-not-psychic-alakazam-best-5th   (76 votes, was #5 on LB)
# Archetype: Alakazam (Psychic) draw-engine. Powerful Hand = 20 dmg per card
# in hand; big draw engine (Kadabra/Alakazam Psychic Draw, Dudunsparce Run Away
# Draw, Fezandipiti Flip the Script) to maximize hand size before attacking.
# ---------------------------------------------------------------------------
ALAKAZAM = _expand([
    (741, 4),   # Abra
    (742, 4),   # Kadabra
    (743, 3),   # Alakazam  (HP140, main attacker - Powerful Hand)
    (305, 3),   # Dunsparce
    (66, 2),    # Dudunsparce (draw engine)
    (140, 1),   # Fezandipiti ex
    (142, 1),   # Genesect
    (858, 1),   # Psyduck
    (343, 1),   # Shaymin
    (1079, 3),  # Rare Candy
    (1081, 3),  # Enhanced Hammer
    (1086, 4),  # Buddy-Buddy Poffin
    (1097, 1),  # Night Stretcher
    (1129, 1),  # Sacred Ash
    (1152, 4),  # Poke Pad
    (1156, 3),  # Lucky Helmet
    (1182, 2),  # Boss's Orders
    (1225, 4),  # Hilda
    (1231, 4),  # Dawn
    (1264, 4),  # Battle Cage (stadium)
    (5, 2),     # Basic Psychic Energy
    (19, 4),    # Telepath Psychic Energy
    (13, 1),    # Enriching Energy
])


# ---------------------------------------------------------------------------
# masamikobayashi/a-sample-archaludon-75-wr-vs-my-1300-starmie   (92 votes)
# Archetype: Archaludon ex (Metal) tempo. Cinderace Turbo Flare energy-accel
# turn 1 -> Duraludon -> Archaludon ex Metal Defender ({M}{M}{M}=220, no
# Weakness next turn). Full Metal Lab (-30 dmg to Metal) + Hero's Cape walls.
# Full 60-card list NOT recoverable (external deck.csv); unique IDs only:
# ---------------------------------------------------------------------------
ARCHALUDON_UNIQUE = [
    169,   # Duraludon (Basic Metal, HP130)
    190,   # Archaludon ex (Stage1, HP300, main attacker - Metal Defender)
    666,   # Cinderace (Stage2, HP160, Turbo Flare energy accel; pre-evos not listed)
    57,    # Relicanth (Memory Dive - unlock Raging Hammer)
    1152,  # Poke Pad
    1121,  # Ultra Ball
    1122,  # Pokegear 3.0
    1097,  # Night Stretcher
    1147,  # Jumbo Ice Cream
    1159,  # Hero's Cape
    1182,  # Boss's Orders
    1185,  # Explorer's Guidance
    1227,  # Lillie's Determination
    1244,  # Full Metal Lab (stadium, -30 dmg to Metal) x4
    8,     # Basic Metal Energy x11
]


ALL_DECKS = {
    "MEGA_LUCARIO": MEGA_LUCARIO,
    "DRAGAPULT": DRAGAPULT,
    "MEGA_ABOMASNOW": MEGA_ABOMASNOW,
    "ALAKAZAM": ALAKAZAM,
}

if __name__ == "__main__":
    for name, d in ALL_DECKS.items():
        assert len(d) == 60, (name, len(d))
        print(f"{name}: {len(d)} cards OK")
    print(f"ARCHALUDON_UNIQUE: {len(ARCHALUDON_UNIQUE)} unique ids (partial)")
