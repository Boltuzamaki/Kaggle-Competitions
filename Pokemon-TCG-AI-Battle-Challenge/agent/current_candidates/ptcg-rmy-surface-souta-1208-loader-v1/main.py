"""Clean-room visible-state Mega Lopunny ex / Dudunsparce policy, V2.

The 60-card list was transcribed from public competition replay 89276670.
Gameplay code is independently authored from the official English card table
and released observation API; no public agent source is included or copied.
"""

import cg.api as api


MIST_ENERGY = 11
ENRICHING_ENERGY = 13
SPIKY_ENERGY = 14
DUDUNSPARCE = 66
FAN_ROTOM = 174
DUNSPARCE = 305
BUNEARY = 848
MEGA_LOPUNNY_EX = 849

BUDDY_POFFIN = 1086
ULTRA_BALL = 1121
POKEGEAR = 1122
POKE_PAD = 1152
AIR_BALLOON = 1174
BOSS_ORDERS = 1182
XEROSIC = 1197
HILDA = 1225
LILLIE = 1227
WALLY = 1229

GALE_THRUST = 1225
SPIKY_HOPPER = 1226
RUN_AWAY_DRAW = "Run Away Draw"
FAN_CALL = "Fan Call"

DECK = (
    MIST_ENERGY, MIST_ENERGY, MIST_ENERGY, MIST_ENERGY,
    ENRICHING_ENERGY,
    SPIKY_ENERGY, SPIKY_ENERGY, SPIKY_ENERGY,
    DUDUNSPARCE, DUDUNSPARCE, DUDUNSPARCE, DUDUNSPARCE,
    FAN_ROTOM,
    DUNSPARCE, DUNSPARCE, DUNSPARCE, DUNSPARCE,
    BUNEARY, BUNEARY, BUNEARY, BUNEARY,
    MEGA_LOPUNNY_EX, MEGA_LOPUNNY_EX, MEGA_LOPUNNY_EX,
    BUDDY_POFFIN, BUDDY_POFFIN, BUDDY_POFFIN, BUDDY_POFFIN,
    ULTRA_BALL, ULTRA_BALL, ULTRA_BALL, ULTRA_BALL,
    POKEGEAR, POKEGEAR, POKEGEAR, POKEGEAR,
    POKE_PAD, POKE_PAD, POKE_PAD, POKE_PAD,
    AIR_BALLOON, AIR_BALLOON, AIR_BALLOON, AIR_BALLOON,
    BOSS_ORDERS, BOSS_ORDERS, BOSS_ORDERS,
    XEROSIC,
    HILDA, HILDA, HILDA, HILDA,
    LILLIE, LILLIE, LILLIE, LILLIE,
    WALLY, WALLY, WALLY, WALLY,
)

CARD_DATA = {card.cardId: card for card in api.all_card_data()}
ATTACK_DATA = {attack.attackId: attack for attack in api.all_attack()}
ENERGIES = {MIST_ENERGY, ENRICHING_ENERGY, SPIKY_ENERGY}
SUPPORTERS = {BOSS_ORDERS, XEROSIC, HILDA, LILLIE, WALLY}
GRIM_FAMILY = {646, 647, 648, 860}
EX_DAMAGE_WALLS = {345}  # Crustle's Mysterious Rock Inn blocks Gale Thrust.

def _choose(observation):
    obs = api.to_observation_class(observation)
    select = obs.select
    state = obs.current
    if select is None or state is None:
        return list(DECK)
    if not select.option or select.maxCount == 0:
        return []

    you = state.yourIndex
    opponent_index = 1 - you
    yours = state.players[you]
    opponent = state.players[opponent_index]
    hand = yours.hand or []
    hand_ids = [card.id for card in hand]
    discard_ids = [card.id for card in yours.discard]

    def board(player):
        result = []
        if player.active and player.active[0] is not None:
            result.append((api.AreaType.ACTIVE, 0, player.active[0]))
        for index, pokemon in enumerate(player.bench):
            if pokemon is not None:
                result.append((api.AreaType.BENCH, index, pokemon))
        return result

    own_board = board(yours)
    enemy_board = board(opponent)
    active = yours.active[0] if yours.active and yours.active[0] is not None else None
    enemy_active = opponent.active[0] if opponent.active and opponent.active[0] is not None else None

    def pokemon_at(player_index, area, index):
        if player_index not in (0, 1) or index is None:
            return None
        player = state.players[player_index]
        if area == api.AreaType.ACTIVE and 0 <= index < len(player.active):
            return player.active[index]
        if area == api.AreaType.BENCH and 0 <= index < len(player.bench):
            return player.bench[index]
        return None

    def option_card_id(option):
        if option.cardId not in (None, 0):
            return option.cardId
        index = option.index
        if index is None:
            return None
        player_index = you if option.playerIndex is None else option.playerIndex
        if option.type in (api.OptionType.PLAY, api.OptionType.ATTACH, api.OptionType.EVOLVE):
            cards = state.players[player_index].hand or []
            return cards[index].id if 0 <= index < len(cards) else None
        if option.area == api.AreaType.HAND:
            cards = state.players[player_index].hand or []
            return cards[index].id if 0 <= index < len(cards) else None
        if option.area == api.AreaType.DISCARD:
            cards = state.players[player_index].discard
            return cards[index].id if 0 <= index < len(cards) else None
        if option.area == api.AreaType.DECK and select.deck is not None:
            return select.deck[index].id if 0 <= index < len(select.deck) and select.deck[index] is not None else None
        if option.area == api.AreaType.LOOKING and state.looking is not None:
            return state.looking[index].id if 0 <= index < len(state.looking) and state.looking[index] is not None else None
        pokemon = pokemon_at(player_index, option.area, index)
        return pokemon.id if pokemon is not None else None

    def option_target(option):
        player_index = you if option.playerIndex is None else option.playerIndex
        if option.type in (api.OptionType.ATTACH, api.OptionType.EVOLVE):
            return pokemon_at(you, option.inPlayArea, option.inPlayIndex)
        return pokemon_at(player_index, option.area, option.index)

    def count_field(card_id):
        return sum(pokemon.id == card_id for _, _, pokemon in own_board)

    def energy_count(pokemon):
        return len(pokemon.energies) if pokemon is not None else 0

    def has_tool(pokemon):
        return pokemon is not None and bool(pokemon.tools)

    def ready_lopunny(pokemon):
        return pokemon is not None and pokemon.id == MEGA_LOPUNNY_EX and energy_count(pokemon) >= 1

    buneary_field = count_field(BUNEARY)
    lopunny_field = count_field(MEGA_LOPUNNY_EX)
    dunsparce_field = count_field(DUNSPARCE)
    dudunsparce_field = count_field(DUDUNSPARCE)
    bunny_line_field = buneary_field + lopunny_field
    dunsparce_line_field = dunsparce_field + dudunsparce_field
    lopunny_hand = hand_ids.count(MEGA_LOPUNNY_EX)
    dudunsparce_hand = hand_ids.count(DUDUNSPARCE)
    energy_hand = sum(hand_ids.count(card_id) for card_id in ENERGIES)
    bench_space = max(0, yours.benchMax - len(yours.bench))
    bench_lopunny = [pokemon for area, _, pokemon in own_board if area == api.AreaType.BENCH and ready_lopunny(pokemon)]
    any_grim_visible = any(pokemon.id in GRIM_FAMILY for _, _, pokemon in enemy_board)
    # The released observation exposes this directly.  Staying stateless avoids
    # stale process memory at game boundaries and is faithful to fresh loading.
    moved_this_turn = bool(state.retreated)
    ex_damage_wall_active = enemy_active is not None and enemy_active.id in EX_DAMAGE_WALLS
    early_two_mega_window = state.turn <= 4 and lopunny_field < 2
    draw_engine_safe = yours.deckCount >= 6
    lillie_draw = 8 if len(yours.prize) == 6 else 6
    # Lillie returns the rest of the hand before drawing, but a conservative
    # reserve is valuable in long Dudunsparce loops and against mill decks.
    lillie_safe = yours.deckCount + max(0, len(hand) - 1) >= lillie_draw + 3

    def prize_value(pokemon):
        data = CARD_DATA.get(pokemon.id) if pokemon is not None else None
        if data is None:
            return 1
        if data.megaEx:
            return 3
        if data.ex:
            return 2
        return 1

    def attack_damage(target):
        if active is None or active.id != MEGA_LOPUNNY_EX:
            return 0
        damage = 230 if moved_this_turn else 160 if energy_count(active) >= 2 else 60
        data = CARD_DATA.get(target.id) if target is not None else None
        if data is not None:
            if data.weakness == api.EnergyType.COLORLESS:
                damage *= 2
            elif data.resistance == api.EnergyType.COLORLESS:
                damage = max(0, damage - 30)
        return damage

    def enemy_score(pokemon):
        if pokemon is None:
            return -100000
        damage = attack_damage(pokemon)
        prizes = prize_value(pokemon)
        knockout = damage >= pokemon.hp
        score = prizes * 5000 + len(pokemon.energies) * 260 + (pokemon.maxHp - pokemon.hp) * 3
        if knockout:
            score += 25000 + prizes * 9000
            if len(opponent.prize) <= prizes:
                score += 300000
        else:
            score += min(damage, pokemon.hp) * 10 - pokemon.hp
        return score

    active_enemy_score = enemy_score(enemy_active)
    best_bench_score = max((enemy_score(p) for area, _, p in enemy_board if area == api.AreaType.BENCH), default=-100000)
    useful_gust = active is not None and active.id == MEGA_LOPUNNY_EX and best_bench_score >= active_enemy_score + 5000
    gale_immediate_win = (
        enemy_active is not None
        and not ex_damage_wall_active
        and attack_damage(enemy_active) >= enemy_active.hp
        and len(opponent.prize) <= prize_value(enemy_active)
    )
    spiky_immediate_win = (
        enemy_active is not None
        and enemy_active.hp <= 160
        and len(opponent.prize) <= prize_value(enemy_active)
    )
    damaged_megas = [p for _, _, p in own_board if p.id == MEGA_LOPUNNY_EX and p.hp < p.maxHp]
    max_mega_damage = max((p.maxHp - p.hp for p in damaged_megas), default=0)

    def visible_enemy_attack_ceiling():
        """Conservative printed-damage ceiling from public visible state only."""
        if enemy_active is None:
            return 0
        data = CARD_DATA.get(enemy_active.id)
        if data is None:
            return 0
        available = len(enemy_active.energies)
        ceiling = 0
        for attack_id in data.attacks:
            attack = ATTACK_DATA.get(attack_id)
            if attack is None or len(attack.energies) > available:
                continue
            damage = attack.damage
            if data.energyType == CARD_DATA[MEGA_LOPUNNY_EX].weakness:
                damage *= 2
            ceiling = max(ceiling, damage)
        return ceiling

    enemy_attack_ceiling = visible_enemy_attack_ceiling()
    active_mega_threatened = (
        active is not None
        and active.id == MEGA_LOPUNNY_EX
        and active.hp <= enemy_attack_ceiling
    )
    # A heal is worthwhile at 90+, or earlier when the visible printed attack
    # would otherwise take three prizes.  This is deliberately not an oracle:
    # only current HP, energy and official card text are used.
    wally_trigger = bool(damaged_megas) and (max_mega_damage >= 150 or active_mega_threatened)
    need_core_mega = lopunny_field < 2 and buneary_field > lopunny_hand
    ultra_can_find_missing = (
        state.turn >= 3
        and ULTRA_BALL in hand_ids
        and len(hand) >= 3
        and need_core_mega
    )
    prefer_item_then_lillie = ultra_can_find_missing and LILLIE in hand_ids and lillie_safe

    # Run Away Draw shuffles the engine off the board, so use the visible hand
    # first when it can still establish the paired-Mega rotation.  This is a
    # state predicate, not action history retained across calls.
    pending_evolution = state.turn >= 3 and (
        (MEGA_LOPUNNY_EX in hand_ids and buneary_field > 0)
        or (DUDUNSPARCE in hand_ids and dunsparce_field > 0)
    )
    pending_basic = bench_space > 0 and (
        (BUNEARY in hand_ids and bunny_line_field < 2)
        or (DUNSPARCE in hand_ids and dunsparce_line_field < 2)
        or (FAN_ROTOM in hand_ids and state.turn <= 2 and count_field(FAN_ROTOM) == 0)
    )
    pending_energy = (
        not state.energyAttached
        and energy_hand > 0
        and any(p.id in (BUNEARY, MEGA_LOPUNNY_EX) and energy_count(p) == 0 for _, _, p in own_board)
    )
    pending_balloon = AIR_BALLOON in hand_ids and any(
        p.id in (BUNEARY, MEGA_LOPUNNY_EX) and not has_tool(p) for _, _, p in own_board
    )
    pending_supporter = not state.supporterPlayed and (
        (HILDA in hand_ids and (buneary_field > lopunny_hand or energy_hand == 0))
        or (WALLY in hand_ids and wally_trigger)
        or (LILLIE in hand_ids and lillie_safe and len(hand) <= 6)
    )
    pending_setup_item = (
        (BUDDY_POFFIN in hand_ids and bench_space > 0 and (bunny_line_field < 2 or dunsparce_line_field < 2))
        or (state.turn >= 3 and ULTRA_BALL in hand_ids and len(hand) >= 3 and need_core_mega)
        or (POKE_PAD in hand_ids and (dunsparce_field > dudunsparce_hand or dunsparce_line_field < 2))
        or (POKEGEAR in hand_ids and yours.deckCount >= 4)
    )
    high_value_hand_action_pending = any((
        pending_evolution,
        pending_basic,
        pending_energy,
        pending_balloon,
        pending_supporter,
        pending_setup_item,
    ))
    post_core_refresh = lopunny_field >= 1 and state.turnActionCount >= 8
    run_away_ready = draw_engine_safe and (len(hand) <= 5 or not high_value_hand_action_pending or post_core_refresh)
    pending_run_away = run_away_ready and dudunsparce_field > 0
    development_pending = high_value_hand_action_pending or pending_run_away

    def keep_value(card_id):
        if card_id is None:
            return 0
        if card_id == MEGA_LOPUNNY_EX:
            return 145 if buneary_field > lopunny_hand else 70
        if card_id == BUNEARY:
            return 125 if bunny_line_field + hand_ids.count(BUNEARY) < 2 else 48
        if card_id == DUDUNSPARCE:
            return 105 if dunsparce_field > dudunsparce_hand else 55
        if card_id == DUNSPARCE:
            return 88 if dunsparce_line_field + hand_ids.count(DUNSPARCE) < 2 else 35
        if card_id == FAN_ROTOM:
            return 64 if state.turn <= 2 and count_field(FAN_ROTOM) == 0 else 8
        if card_id == ENRICHING_ENERGY:
            return 140
        if card_id == MIST_ENERGY:
            return 120 if any_grim_visible else 92
        if card_id == SPIKY_ENERGY:
            return 132 if state.turn <= 4 else 100
        if card_id == AIR_BALLOON:
            bunny_tools = sum(has_tool(p) for _, _, p in own_board if p.id in (BUNEARY, MEGA_LOPUNNY_EX))
            if bunny_tools >= 2:
                return 12
            if bunny_tools >= 1 and hand_ids.count(AIR_BALLOON) >= 2:
                return 28
            return 125
        if card_id in (BUDDY_POFFIN, ULTRA_BALL, POKE_PAD):
            return 75
        if card_id == POKEGEAR:
            return 62
        if card_id == HILDA:
            if prefer_item_then_lillie:
                return 25
            return 105 if lopunny_field == 0 and (buneary_field > lopunny_hand or energy_hand == 0) else 32
        if card_id == WALLY:
            return 150 if wally_trigger else 52
        if card_id == BOSS_ORDERS:
            return 110 if useful_gust else 45
        if card_id == XEROSIC:
            return 72
        if card_id == LILLIE:
            return 80 if lillie_safe and len(hand) <= 5 else 5
        return 20

    def own_target_score(pokemon, area=None):
        if pokemon is None:
            return -100000
        score = pokemon.hp + energy_count(pokemon) * 180
        if pokemon.id == MEGA_LOPUNNY_EX:
            score += 30000 + (15000 if ready_lopunny(pokemon) else 0) + (5000 if has_tool(pokemon) else 0)
            score += (pokemon.maxHp - pokemon.hp) * 8
        elif pokemon.id == BUNEARY:
            score += 10000
        elif pokemon.id == DUDUNSPARCE:
            score += 7000
        elif pokemon.id == DUNSPARCE:
            score += 5000
        if area == api.AreaType.ACTIVE and ready_lopunny(pokemon):
            score += 2000
        return score

    def search_score(card_id):
        if card_id is None:
            return -100000
        if card_id == MEGA_LOPUNNY_EX:
            missing = max(0, buneary_field - lopunny_hand)
            return 50000 + missing * 4000 if missing else 4000
        if card_id == BUNEARY:
            return 45000 - bunny_line_field * 7000 if bunny_line_field < 2 and bench_space > 0 else 500
        if card_id == DUDUNSPARCE:
            missing = max(0, dunsparce_field - dudunsparce_hand)
            return 38000 + missing * 3000 if missing else 2500
        if card_id == DUNSPARCE:
            return 32000 - dunsparce_line_field * 5000 if dunsparce_line_field < 2 and bench_space > 0 else 500
        if card_id == FAN_ROTOM:
            return 28000 if state.turn <= 2 and count_field(FAN_ROTOM) == 0 and bench_space > 0 else 100
        if card_id == ENRICHING_ENERGY:
            return 36000
        if card_id == MIST_ENERGY:
            return 33000 if any_grim_visible else 26000
        if card_id == SPIKY_ENERGY:
            return 29000
        if card_id == HILDA:
            return 30000 if buneary_field > lopunny_hand or energy_hand == 0 else 12000
        if card_id == WALLY:
            return 42000 if wally_trigger else 7000
        if card_id == BOSS_ORDERS:
            return 35000 if useful_gust else 8000
        if card_id == XEROSIC:
            opponent_hand = len(opponent.hand or [])
            return 27000 if opponent_hand >= 6 else 6000
        if card_id == LILLIE:
            return 25000 if lillie_safe and len(hand) <= 5 else -5000
        return 1000 - keep_value(card_id)

    def play_score(card_id):
        if card_id is None:
            return -100000
        data = CARD_DATA.get(card_id)
        if data is not None and data.cardType == api.CardType.POKEMON:
            if card_id == BUNEARY:
                return 550000 - bunny_line_field * 8000 if bunny_line_field < 2 and bench_space > 0 else -6000
            if card_id == DUNSPARCE:
                return 545000 - dunsparce_line_field * 9000 if dunsparce_line_field < 2 and bench_space > 0 else -6000
            if card_id == FAN_ROTOM:
                return 505000 if state.turn <= 2 and count_field(FAN_ROTOM) == 0 and bench_space > 0 else -7000
            return -6000
        if card_id == BOSS_ORDERS:
            return 470000 if useful_gust else -5000
        if card_id == WALLY:
            return 585000 if wally_trigger else -5000
        if card_id == HILDA:
            missing_evolution = buneary_field > lopunny_hand
            if prefer_item_then_lillie:
                return 480000
            return 525000 if missing_evolution or energy_hand == 0 else -4000
        if card_id == XEROSIC:
            opponent_hand = len(opponent.hand or [])
            established = lopunny_field >= 1 and energy_hand + sum(energy_count(p) for _, _, p in own_board) >= 1
            return 395000 if established and opponent_hand >= 6 else -4500
        if card_id == LILLIE:
            return 525000 + max(0, 7 - len(hand)) * 3500 if lillie_safe and len(hand) <= 6 else -4500
        if card_id == BUDDY_POFFIN:
            need = (bunny_line_field < 2 or dunsparce_line_field < 2) and bench_space > 0
            return 555000 if need else -6000
        if card_id == ULTRA_BALL:
            return 560000 if state.turn >= 3 and need_core_mega and len(hand) >= 3 else -7000
        if card_id == POKE_PAD:
            need = dunsparce_field > dudunsparce_hand or dunsparce_line_field < 2
            return 525000 if need else -5000
        if card_id == POKEGEAR:
            return 530000 if len(hand) <= 6 and yours.deckCount >= 4 else 180000
        if card_id == AIR_BALLOON:
            return -5000  # Tool use is represented by ATTACH in the released SDK.
        return 1000

    def attach_score(option, card_id):
        target = option_target(option)
        if target is None:
            return -100000
        if card_id == AIR_BALLOON:
            if has_tool(target):
                return -6000
            if target.id == MEGA_LOPUNNY_EX:
                if sum(has_tool(p) for _, _, p in own_board if p.id in (BUNEARY, MEGA_LOPUNNY_EX)) >= 2:
                    return -4000
                return 535000 + (12000 if target is active else 0) + (10000 if early_two_mega_window else 0)
            if target.id == BUNEARY:
                bunny_tools = sum(has_tool(p) for _, _, p in own_board if p.id in (BUNEARY, MEGA_LOPUNNY_EX))
                if bunny_tools >= 2 or lopunny_field >= 2:
                    return -4000
                return 515000 + (65000 if target is active and state.turn <= 2 else 0) + (10000 if early_two_mega_window else 0)
            return 1000
        if card_id not in ENERGIES:
            return 0
        if target.id == MEGA_LOPUNNY_EX:
            score = 540000 if energy_count(target) == 0 else 300000 if energy_count(target) == 1 else 2000
            if target is active:
                score += 8000
                if ex_damage_wall_active and energy_count(target) == 1:
                    score += 230000
            elif early_two_mega_window and energy_count(target) == 0:
                score += 16000
            if card_id == ENRICHING_ENERGY:
                score += 24000
            if card_id == MIST_ENERGY and any_grim_visible:
                score += 22000
            if card_id == SPIKY_ENERGY and target is active:
                score += 17000
            return score
        if target.id == BUNEARY:
            score = 520000 if energy_count(target) == 0 else 3000
            if card_id == ENRICHING_ENERGY:
                score += 22000
            return score
        if target.id in (DUNSPARCE, DUDUNSPARCE) and active is target:
            return 170000 if energy_count(target) == 0 else 1000
        return 500

    def evolve_score(option):
        card_id = option_card_id(option)
        target = option_target(option)
        if target is None:
            return -100000
        if card_id == MEGA_LOPUNNY_EX and target.id == BUNEARY:
            score = 565000 + (12000 if early_two_mega_window else 0) + (9000 if energy_count(target) else 0) + (7000 if has_tool(target) else 0)
            if early_two_mega_window and active is not None and active.id == BUNEARY:
                other_buneary = any(area == api.AreaType.BENCH and pokemon.id == BUNEARY for area, _, pokemon in own_board)
                if other_buneary:
                    score += 28000 if target is not active else -28000
            elif target is active:
                score += 8000
            return score
        if card_id == DUDUNSPARCE and target.id == DUNSPARCE:
            return 555000 + (5000 if target is active else 0)
        return 1000

    def ability_score(option):
        target = option_target(option)
        if target is not None and target.id == DUDUNSPARCE:
            return 515000 if run_away_ready else -9000
        if target is not None and target.id == FAN_ROTOM and state.turn <= 2:
            return 570000
        return 150000

    def option_score(option):
        if option.type == api.OptionType.YES:
            return 10000
        if option.type == api.OptionType.NO:
            return 0
        if option.type == api.OptionType.NUMBER:
            return option.number or 0
        if option.type == api.OptionType.PLAY:
            return play_score(option_card_id(option))
        if option.type == api.OptionType.ATTACH:
            return attach_score(option, option_card_id(option))
        if option.type == api.OptionType.EVOLVE:
            return evolve_score(option)
        if option.type == api.OptionType.ABILITY:
            return ability_score(option)
        if option.type == api.OptionType.RETREAT:
            if development_pending and not bench_lopunny:
                return 120000
            if ex_damage_wall_active:
                if active is not None and energy_count(active) >= 2:
                    return -5000
                two_energy_bench = [pokemon for pokemon in bench_lopunny if energy_count(pokemon) >= 2]
                return 480000 if two_energy_bench else 180000
            if bench_lopunny:
                return 480000
            return -4000
        if option.type == api.OptionType.ATTACK:
            if option.attackId == GALE_THRUST:
                if ex_damage_wall_active:
                    return -8000
                if development_pending and not gale_immediate_win:
                    return 300000
                return 620000 if moved_this_turn else 250000
            if option.attackId == SPIKY_HOPPER:
                if development_pending and not spiky_immediate_win:
                    return 300000
                return 620000 if ex_damage_wall_active else 430000
            return 90000
        if option.type == api.OptionType.END:
            return -10000
        if option.type == api.OptionType.CARD:
            card_id = option_card_id(option)
            player_index = you if option.playerIndex is None else option.playerIndex
            target = option_target(option)
            if select.context == api.SelectContext.SETUP_ACTIVE_POKEMON:
                return {BUNEARY: 50000, DUNSPARCE: 35000, FAN_ROTOM: 18000}.get(card_id, 1000)
            if select.context in (api.SelectContext.SETUP_BENCH_POKEMON, api.SelectContext.TO_BENCH, api.SelectContext.TO_FIELD):
                if select.context == api.SelectContext.TO_BENCH and state.turn <= 2:
                    if card_id == FAN_ROTOM:
                        return 60000 if count_field(FAN_ROTOM) == 0 else -1000
                    if card_id == DUNSPARCE:
                        return 52000 - dunsparce_line_field * 3500
                    if card_id == BUNEARY:
                        return 44000 - bunny_line_field * 4000
                if card_id == BUNEARY:
                    return 48000 - bunny_line_field * 4000
                if card_id == DUNSPARCE:
                    return 42000 - dunsparce_line_field * 3500
                if card_id == FAN_ROTOM:
                    return 30000 if state.turn <= 2 and count_field(FAN_ROTOM) == 0 else -1000
                return 500
            if select.context in (api.SelectContext.TO_ACTIVE, api.SelectContext.SWITCH):
                if player_index == opponent_index:
                    return enemy_score(target)
                if ready_lopunny(target):
                    return 60000 + target.hp + (5000 if has_tool(target) else 0)
                return own_target_score(target, option.area)
            if select.context == api.SelectContext.TO_HAND:
                return search_score(card_id)
            if select.context in (api.SelectContext.DISCARD, api.SelectContext.TO_DECK_BOTTOM):
                return -keep_value(card_id)
            if select.context in (api.SelectContext.DAMAGE, api.SelectContext.DAMAGE_COUNTER, api.SelectContext.DAMAGE_COUNTER_ANY):
                return enemy_score(target) if player_index == opponent_index else -own_target_score(target, option.area)
            if select.context in (api.SelectContext.HEAL, api.SelectContext.REMOVE_DAMAGE_COUNTER):
                if target is not None and target.id == MEGA_LOPUNNY_EX:
                    return (target.maxHp - target.hp) * 100 + (5000 if target is active else 0)
                return target.maxHp - target.hp if target is not None else 0
            if select.context in (api.SelectContext.ATTACH_TO, api.SelectContext.EFFECT_TARGET):
                if player_index == opponent_index:
                    return enemy_score(target)
                return own_target_score(target, option.area)
            return search_score(card_id)
        if option.type in (api.OptionType.ENERGY_CARD, api.OptionType.TOOL_CARD, api.OptionType.ENERGY):
            return option.count or 0
        if option.type == api.OptionType.SKILL:
            return 1000
        if option.type == api.OptionType.SPECIAL_CONDITION:
            return 1
        return 0

    scores = [option_score(option) for option in select.option]
    order = sorted(range(len(scores)), key=lambda index: (scores[index], -index), reverse=True)

    if select.context in (api.SelectContext.DISCARD, api.SelectContext.TO_DECK_BOTTOM):
        take = min(select.maxCount, max(select.minCount, 0))
        return order[:take]

    chosen = []
    selected_buneary = 0
    selected_dunsparce = 0
    for index in order:
        if len(chosen) >= select.maxCount:
            break
        card_id = option_card_id(select.option[index])
        if select.context in (api.SelectContext.SETUP_BENCH_POKEMON, api.SelectContext.TO_BENCH, api.SelectContext.TO_FIELD):
            if card_id == BUNEARY and bunny_line_field + selected_buneary >= 2:
                continue
            if card_id == DUNSPARCE and dunsparce_line_field + selected_dunsparce >= 2:
                continue
            selected_buneary += card_id == BUNEARY
            selected_dunsparce += card_id == DUNSPARCE
        elif select.context == api.SelectContext.TO_HAND:
            # Fan Call's early public line fills a three-Buneary/two-Dunsparce
            # reserve without wasting all three slots on duplicate copies.
            if card_id == BUNEARY and bunny_line_field + hand_ids.count(BUNEARY) + selected_buneary >= 3:
                continue
            if card_id == DUNSPARCE and dunsparce_line_field + hand_ids.count(DUNSPARCE) + selected_dunsparce >= 2:
                continue
            selected_buneary += card_id == BUNEARY
            selected_dunsparce += card_id == DUNSPARCE
        if scores[index] >= 0 or len(chosen) < select.minCount:
            chosen.append(index)

    if len(chosen) < select.minCount:
        for index in order:
            if index not in chosen:
                chosen.append(index)
                if len(chosen) >= select.minCount:
                    break

    return chosen


def mega_lopunny_cleanroom_entrypoint(observation):
    """Absolute-final one-argument competition callable."""
    if not isinstance(observation, dict) or observation.get("select") is None or observation.get("current") is None:
        return list(DECK)
    return _choose(observation)
