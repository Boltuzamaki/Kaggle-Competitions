"""Discussion-level metagame deck profiles for hidden-card beliefs.

This file contains card counts only, not any public agent's decision policy.
The profiles are used to form plausible hidden states from cards already seen.
"""


def _expand(pairs):
    cards = []
    for card_id, count in pairs:
        cards.extend([card_id] * count)
    return cards


META_PROFILES = {
    "lucario": _expand([
        (673, 2), (674, 2), (675, 2), (676, 3), (677, 3), (678, 4),
        (1102, 4), (1123, 2), (1141, 4), (1142, 4), (1152, 4),
        (1159, 1), (1182, 2), (1192, 4), (1227, 4), (1252, 2), (6, 13),
    ]),
    "dragapult": _expand([
        (119, 4), (120, 4), (121, 3), (140, 1), (184, 1), (235, 2),
        (1071, 1), (1079, 2), (1080, 1), (1086, 4), (1097, 2),
        (1120, 4), (1121, 4), (1152, 3), (1156, 1), (1182, 3),
        (1198, 4), (1210, 2), (1227, 4), (1256, 2), (2, 4), (5, 4),
    ]),
    "abomasnow": _expand([
        (721, 2), (722, 4), (723, 4), (1121, 4), (1126, 1),
        (1192, 4), (1227, 4), (1262, 3), (3, 34),
    ]),
    "alakazam": _expand([
        (741, 4), (742, 4), (743, 3), (305, 3), (66, 2), (140, 1),
        (142, 1), (858, 1), (343, 1), (1079, 3), (1081, 3),
        (1086, 4), (1097, 1), (1129, 1), (1152, 4), (1156, 3),
        (1182, 2), (1225, 4), (1231, 4), (1264, 4), (5, 2), (19, 4),
        (13, 1),
    ]),
    "archaludon": _expand([
        (169, 4), (190, 4), (666, 4), (57, 1), (1121, 4), (1122, 4),
        (1147, 4), (1152, 4), (1097, 3), (1159, 1), (1182, 3),
        (1185, 4), (1213, 1), (1227, 4), (1244, 4), (8, 11),
    ]),
}


assert all(len(deck) == 60 for deck in META_PROFILES.values())
