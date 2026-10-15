"""Bounded, reproducible forecasts for building and using a scoring engine.

These are policy estimates, never game observations. Each forecast deals a
single shuffled deck without replacement, keeps unplayed cards, and advances
the effects whose counters can be predicted. Draw decisions do not inspect
the next cards in the shuffled pile.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass, replace
from random import Random
from statistics import median

from strategy import (Card, Joker, RunState, SUITS, HAND_ORDER, HAND_RANK,
                      _boss, _face, _card_future_value, _discard_candidates,
                      _preserved_pattern_value, _draw_best_plan, score_hand,
                      next_blind_target, _shop_round_state, _shop_order_variants,
                      _standard_unseen, _discard_cost, classify, _before_scoring,
                      _card_retriggers, _card_debuffed, _effective_suits, _probability, HAND_LEVELS,
                      _play_floor, _joker_global_effect, _legal_play)
from strategy import _rank_windows


CHIP_JOKERS = frozenset({
    'sly_joker', 'wily_joker', 'clever_joker', 'devious_joker', 'crafty_joker',
    'stuntman', 'blue_joker', 'bull', 'runner', 'square_joker', 'ice_cream',
    'stone_joker', 'wee_joker', 'castle', 'banner', 'odd_todd', 'scholar',
    'walkie_talkie', 'arrowhead', 'hiker',
})
MULT_JOKERS = frozenset({
    'joker', 'jolly_joker', 'zany_joker', 'mad_joker', 'crazy_joker',
    'droll_joker', 'half_joker', 'gros_michel', 'abstract_joker', 'supernova',
    'ride_the_bus', 'green_joker', 'spare_trousers', 'fibonacci', 'even_steven',
    'raised_fist', 'shoot_the_moon', 'bootstraps', 'mystic_summit', 'popcorn',
    'fortune_teller', 'flash_card', 'smiley_face', 'lusty_joker', 'greedy_joker',
    'wrathful_joker', 'gluttonous_joker', 'onyx_agate', 'swashbuckler',
})
XMULT_JOKERS = frozenset({
    'the_duo', 'the_trio', 'the_family', 'the_order', 'the_tribe', 'cavendish',
    'card_sharp', 'blackboard', 'joker_stencil', 'hologram', 'constellation',
    'vampire', 'ramen', 'acrobat', 'flower_pot', 'seeing_double', 'steel_joker',
    'obelisk', 'throwback', 'loyalty_card', 'drivers_license', 'campfire',
    'glass_joker', 'lucky_cat', 'hit_the_road', 'madness', 'canio', 'yorick',
    'photograph', 'baron', 'triboulet', 'ancient_joker', 'the_idol', 'bloodstone',
    'baseball_card',
})
RETRIGGER_JOKERS = frozenset({'hanging_chad', 'sock_and_buskin', 'hack', 'mime', 'seltzer'})
GROWTH_JOKERS = frozenset({
    'green_joker', 'ride_the_bus', 'supernova', 'runner', 'square_joker',
    'spare_trousers', 'hologram', 'constellation', 'fortune_teller', 'flash_card',
    'vampire', 'glass_joker', 'lucky_cat', 'wee_joker', 'castle', 'canio', 'yorick', 'obelisk',
})
ECONOMY_JOKERS = frozenset({
    'golden_joker', 'cloud_9', 'rocket', 'mail_in_rebate', 'business_card',
    'reserved_parking', 'egg', 'gift_card', 'delayed_gratification',
    'golden_ticket', 'satellite', 'to_the_moon', 'to_do_list', 'rough_gem', 'trading_card',
})
TEMPORARY_JOKERS = frozenset({'ice_cream', 'popcorn', 'seltzer', 'turtle_bean'})
CONDITIONAL_XMULT_JOKERS = frozenset({
    'flower_pot', 'blackboard', 'card_sharp', 'the_duo', 'the_trio',
    'the_family', 'the_order', 'the_tribe', 'seeing_double', 'acrobat',
    'loyalty_card', 'joker_stencil',
})
PHASE_XMULT_JOKERS = frozenset({'card_sharp', 'acrobat', 'loyalty_card'})


@dataclass(frozen=True)
class EngineProfile:
    chips: int
    additive_mult: int
    multiplicative_mult: int
    retriggers: int
    growth: int
    economy: int
    temporary: int
    preferred_hand: str
    weaknesses: tuple[str, ...]


@dataclass(frozen=True)
class RoundForecast:
    target: float
    boss: str
    mean_score: float
    safe_score: float
    clear_rate: float
    typical_hand: float
    hands: int
    discards: int
    frequencies: tuple[tuple[str, float], ...]
    scores: tuple[float, ...]
    # Training opportunities stop when the blind clears, unlike score-budget
    # samples, which measure how much all available hands could produce.
    growth_per_round: tuple[tuple[str, float], ...] = ()
    plays_per_round: tuple[tuple[str, float], ...] = ()
    level_gains_per_round: tuple[tuple[str, float], ...] = ()
    # Produced at a verified projected clear; use is a future policy estimate.
    # These do not occupy current inventory or level the ongoing blind.
    blue_planets_per_round: float = 0.
    # Estimated activation in plays before the blind clears, not observed rates.
    conditional_xmult_rates: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class ShopForecast:
    next_round: RoundForecast
    boss_round: RoundForecast | None
    sustainable_score: float
    profile: EngineProfile
    fragile_score: float | None = None
    fragile_loss_probability: float = 0
    development_score: float | None = None

    @property
    def weak(self):
        return (self.next_round.safe_score < self.next_round.target * 1.15
                or self.sustainable_score < self.next_round.target * 1.15
                or (self.boss_round is not None and
                    self.boss_round.safe_score < self.boss_round.target * 1.1))

    @property
    def coverage(self):
        values = [min(self.next_round.safe_score, self.sustainable_score) /
                  max(1, self.next_round.target)]
        if self.boss_round is not None:
            values.append(self.boss_round.safe_score / max(1, self.boss_round.target))
        return min(values)

    @property
    def clear_rate(self):
        rates = [self.next_round.clear_rate]
        if self.boss_round is not None:
            rates.append(self.boss_round.clear_rate)
        return min(rates)


def growth_opportunity(joker, state, forecast):
    """Reward an affordable growth path, scaled by likely activation and time."""
    horizon = max(0, 8 - state.ante)
    if joker.key not in GROWTH_JOKERS or not horizon or joker.debuffed:
        return 0.
    frequencies = dict(getattr(forecast.next_round, 'frequencies', ()))
    measured = dict(getattr(forecast.next_round, 'growth_per_round', ()))
    if joker.key in measured and measured[joker.key] <= 0:
        return 0.
    activation = 1.
    if joker.key == 'runner':
        activation = sum(v for k, v in frequencies.items() if 'straight' in k)
    elif joker.key == 'spare_trousers':
        activation = sum(frequencies.get(k, 0) for k in ('two_pair', 'full_house', 'flush_house'))
    elif joker.key == 'ride_the_bus':
        deck = full_deck(state)
        activation = sum(not _face(c, state) for c in deck) / max(1, len(deck))
        activation *= activation
    elif joker.key == 'constellation':
        generated = getattr(forecast.next_round, 'blue_planets_per_round', 0.)
        activation = min(1., state.money / 30 + .35 * state.has('astronomer') + generated)
    elif joker.key == 'hologram':
        activation = min(1., state.money / 40 + state.effect_count('marble_joker'))
    elif joker.key == 'vampire':
        activation = min(1., sum(bool(c.enhancement) for c in full_deck(state)) / 8)
    elif joker.key == 'glass_joker':
        activation = min(1., sum(c.enhancement == 'glass' for c in full_deck(state)) / 5)
    elif joker.key == 'lucky_cat':
        activation = min(1., sum(c.enhancement == 'lucky' for c in full_deck(state)) / 5)
    elif joker.key in ('canio', 'flash_card'):
        activation = 0.  # No known future destruction/rerolls.
    elif joker.key == 'green_joker':
        activation = max(.2, 1 - forecast.next_round.discards / max(1, forecast.next_round.hands) * .3)
    elif joker.key == 'obelisk':
        # A head start in another hand type is necessary to sustain it.
        activation = min(1., max(state.played_types.values(), default=0) / 6)
    return min(10., horizon * 1.5 * activation)


def replacement_reduces_survival(before, after):
    """Protect each imminent requirement, rather than just their minimum."""
    old_next = before.next_round.safe_score / max(1, before.next_round.target)
    new_next = after.next_round.safe_score / max(1, after.next_round.target)
    # A distant Boss improvement cannot pay for crossing into a lethal next
    # blind, or a large drop in its sampled completion rate near the threshold.
    if old_next >= 1 > new_next + .01:
        return True
    if (old_next < 1.1 and new_next + .01 < old_next
            or new_next < 1.15 and after.next_round.clear_rate + .16 < before.next_round.clear_rate):
        return True
    if before.coverage < 1.1 and after.coverage + .01 < before.coverage:
        return True
    if before.boss_round and after.boss_round:
        old = before.boss_round.safe_score / max(1, before.boss_round.target)
        new = after.boss_round.safe_score / max(1, after.boss_round.target)
        if old < 1.1 and new + .01 < old:
            return True
    return False


def full_deck(state):
    return state.deck_cards or state.remaining_deck or tuple(
        Card(rank, suit) for suit in SUITS for rank in range(2, 15))


def engine_profile(state, frequencies=(), conditional_rates=None):
    keys = Counter(j.key for j in state.active_jokers)
    def count(group):
        return sum(keys[key] for key in group)
    chips = count(CHIP_JOKERS) + sum(j.edition == 'foil' for j in state.active_jokers)
    additive = count(MULT_JOKERS) + sum(j.edition == 'holographic' for j in state.active_jokers)
    # Baseball contributes only when at least one physical Uncommon remains.
    multiplicative = count(XMULT_JOKERS) - keys['baseball_card']
    # An unpowered X1 scaler is a growth route, but supplies no multiplier
    # yet. Avoid declaring a late-game engine complete because of its label.
    unpowered = {'hologram', 'constellation', 'vampire', 'lucky_cat', 'glass_joker',
                 'campfire', 'madness', 'canio', 'yorick', 'hit_the_road', 'obelisk'}
    multiplicative -= sum(j.key in unpowered and j.value <= 1 for j in state.active_jokers)
    multiplicative -= sum(j.key == 'ramen' and j.numeric_value_available and j.value <= 1
                          for j in state.active_jokers)
    rates = dict(conditional_rates or ())
    if conditional_rates is not None:
        multiplicative -= sum(j.key in rates and rates[j.key] < .25
                              for j in state.active_jokers)
    deck = full_deck(state)
    for j in state.active_jokers:
        if j.key == 'drivers_license':
            enhanced = (sum(bool(c.enhancement) for c in state.deck_cards)
                        if state.deck_cards and len(state.deck_cards) == state.deck_size
                        else j.get('enhanced_count', sum(bool(c.enhancement) for c in state.remaining_deck)))
            multiplicative -= enhanced < 16
        elif j.key == 'steel_joker':
            steel = (sum(c.enhancement == 'steel' for c in state.deck_cards)
                     if state.deck_cards and len(state.deck_cards) == state.deck_size else
                     j.value > 1 if j.get('observed_value') else
                     max(j.value, sum(c.enhancement == 'steel' for c in state.remaining_deck)))
            multiplicative -= not steel
    if not any(_face(c, state) for c in deck):
        multiplicative -= keys['photograph'] + keys['triboulet']
    # Throwback is computed from verified skipped blinds, not its OCR value.
    if state.skipped_blinds == 0:
        multiplicative -= keys['throwback']
    if any(j.rarity == 'uncommon' for j in state.active_jokers):
        multiplicative += keys['baseball_card']
    multiplicative += sum(j.edition == 'polychrome' for j in state.active_jokers)
    weights = Counter(dict(frequencies))
    for category, plays in state.played_types.items():
        weights[category] += plays / max(8, sum(state.played_types.values())) * .35
    # A newly bought enabler may not yet appear in played-hand history.
    supports = {
        'pair': {'jolly_joker', 'sly_joker', 'the_duo'},
        'two_pair': {'mad_joker', 'clever_joker', 'spare_trousers'},
        'three_kind': {'zany_joker', 'wily_joker', 'the_trio'},
        'straight': {'crazy_joker', 'devious_joker', 'the_order', 'runner', 'shortcut'},
        'flush': {'droll_joker', 'crafty_joker', 'the_tribe', 'smeared_joker'},
        'four_kind': {'the_family'},
    }
    for category, supporting in supports.items():
        weights[category] += .12 * sum(keys[k] for k in supporting)
    for category, level in state.levels.items():
        if level > 1:
            weights[category] += min(.3, (level - 1) * .03)
    preferred = max(weights, key=weights.get) if weights else 'pair'
    weaknesses = []
    if not chips and max(state.levels.values(), default=1) < 3:
        weaknesses.append('low_chips')
    if not additive and max(state.levels.values(), default=1) < 3:
        weaknesses.append('low_mult')
    def repeats_multiplier(c):
        return (c.enhancement == 'glass' or c.edition == 'polychrome'
                or _face(c, state) and (keys['photograph'] or keys['triboulet'])
                or keys['ancient_joker'] or keys['the_idol'] or keys['bloodstone'])
    played_repeat = any(repeats_multiplier(c) and (
        keys['hanging_chad'] or keys['seltzer']
        or keys['sock_and_buskin'] and _face(c, state)
        or keys['hack'] and not c.is_stone and 2 <= c.rank <= 5) for c in deck)
    held_repeat = keys['mime'] and any(c.enhancement == 'steel' or
        not c.is_stone and c.rank == 13 and keys['baron'] for c in deck)
    if state.ante >= 3 and not multiplicative and not (played_repeat or held_repeat):
        weaknesses.append('no_multiplier_or_retrigger')
    conditional_effective = sum(j.key in CONDITIONAL_XMULT_JOKERS and rates.get(j.key, 1.) >= .25
                                for j in state.active_jokers)
    reliable_condition = any(rate >= (.25 if key in PHASE_XMULT_JOKERS else .75)
                             for key, rate in rates.items())
    if rates and not reliable_condition and multiplicative <= conditional_effective:
        weaknesses.append('unreliable_multiplier')
    if count(TEMPORARY_JOKERS):
        weaknesses.append('temporary_support')
    if keys['gros_michel']:
        weaknesses.append('fragile_support')
    growing = count(GROWTH_JOKERS) + keys['burnt_joker'] + keys['space_joker']
    if max(state.played_types.values(), default=0) < 2:
        growing -= keys['obelisk']
    if keys['swashbuckler'] and (keys['egg'] or keys['gift_card']):
        growing += 1
    if (not state.observatory and state.consumables_held < state.consumable_slots
            and any(c.seal.lower() == 'blue' and not c.debuffed and not c.face_down
                    for c in state.deck_cards or state.remaining_deck)):
        growing += 1  # Known recurring level production is also engine growth.
    if not growing and state.ante >= 3:
        weaknesses.append('no_scaling')
    return EngineProfile(chips, additive, multiplicative,
                         count(RETRIGGER_JOKERS), growing,
                         count(ECONOMY_JOKERS), count(TEMPORARY_JOKERS), preferred,
                         tuple(weaknesses))


def _round_state(state, boss, target):
    state = _shop_round_state(state)
    hand_size = state.hand_size
    # A completed Manacle round temporarily reduced the observed hand size.
    if _boss(state) == 'manacle' and state.round_resources_observed:
        hand_size += 1
    probe = replace(state, boss=boss, hand_size=hand_size, target_score=target,
                    round_score=0, round_played_types={}, boss_seen_types=frozenset(),
                    boss_target_hand='', boss_first_hand_done=False, forced_card_index=-1,
                    seen_cards=(), boss_disabled=False, joker_order_unknown=False)
    if _boss(probe) == 'amber_acorn':
        probe = replace(probe, joker_order_unknown=True)
    if _boss(probe) == 'water':
        probe = replace(probe, discards_left=0)
    elif _boss(probe) == 'needle':
        probe = replace(probe, hands_left=max(1, 1 + 3 * probe.effect_count('burglar')
                                            - probe.joker_count('troubadour')))
    elif _boss(probe) == 'manacle':
        probe = replace(probe, hand_size=max(1, hand_size - 1))
    return probe


def advance_projected_blind_start(state):
    """Known Marble production in a forecast only; never write runtime state.

    Stone has no rank/suit for scoring. Its random printed front therefore
    needs no invented identity. Copiers add cards; each physical Hologram
    grows once per added card, not once per scoring copy.
    """
    count = state.effect_count('marble_joker')
    if not count:
        return state
    stones = (Card(2, 'S', enhancement='stone'),) * count
    jokers = []
    for j in state.jokers:
        if not j.debuffed and j.key == 'hologram':
            j = replace(j, value=max(1., j.value) + .25 * count)
        elif not j.debuffed and j.key == 'stone_joker' and j.get('observed_value'):
            j = replace(j, value=j.value + 25 * count)
        jokers.append(j)
    return replace(state, jokers=tuple(jokers), deck_size=state.deck_size + count,
                   deck_cards=state.deck_cards + stones if state.deck_cards else ())


def _hold_potential(hand, keep, state, preferred):
    held = [hand[i] for i in keep]
    ranks = Counter(c.rank for c in held if not c.is_stone and not c.face_down)
    suit_counts = Counter(c.suit for c in held if not c.is_stone and not c.face_down)
    value = _preserved_pattern_value(held, state)
    value += sum(_card_future_value(c, state) for c in held) * .15
    if 'flush' in preferred:
        value += 11 * max(suit_counts.values(), default=0)
    if preferred in ('pair', 'two_pair', 'three_kind', 'full_house', 'four_kind', 'five_kind'):
        value += 10 * max(ranks.values(), default=0)
    if 'straight' in preferred or state.has('shortcut') or state.has('runner'):
        best_chain = 0
        for ace_low in (False, True):
            ordered = sorted(1 if ace_low and r == 14 else r for r in ranks)
            chain = 0
            previous = -100
            for rank in ordered:
                chain = chain + 1 if rank - previous <= (2 if state.has('shortcut') else 1) else 1
                previous = rank
                best_chain = max(best_chain, chain)
        value += 12 * min(5, best_chain)
    if state.has('ride_the_bus'):
        value -= sum(_face(c, state) for c in held) * 5
    if state.has('blackboard'):
        value -= sum(c.suit in ('H', 'D') for c in held) * 5
    if state.has('flower_pot'):
        from strategy import suit_coverage
        # Splash makes all retained cards potential scoring cards. Without
        # it, prefer rank/straight structure; rainbow high cards cannot fire.
        if state.has('splash'):
            value += 16 * suit_coverage(held,state)
        elif 0 < len(held) <= 5:
            pattern = classify(held,state)
            value += 8 * suit_coverage([held[i] for i in pattern.scoring],state)
    # An incomplete three-card core needs room to receive replacements.
    value -= 2 * max(0, len(keep) - 5)
    if any(j.key in ('burnt_joker', 'castle', 'yorick', 'trading_card') for j in state.active_jokers):
        discarded = [c for i, c in enumerate(hand) if i not in keep]
        after = advance_projected_discard(state, discarded)
        for category, level in after.levels.items():
            gain = level - state.level(category)
            if gain > 0:
                chips, mult = HAND_LEVELS[category][2:]
                value += gain * (12 + chips * .25 + mult * 6) * (1 if category == preferred else .25)
        for old, new in zip(state.jokers, after.jokers):
            if old.key == 'castle':
                value += max(0, new.value - old.value) * .7
            elif old.key == 'yorick':
                value += max(0, new.value - old.value) * 25
        value += max(0, after.money - state.money) * 2
    return value


def forecast_hold(hand, state, preferred):
    candidates = _discard_candidates(hand, state)
    if not candidates:
        return tuple(range(len(hand)))
    return max(candidates, key=lambda keep: (_hold_potential(hand, keep, state, preferred), -len(keep)))


def diverse_draw_candidates(hand, state, candidates, limit=14):
    """Keep alternative build plans in 11–30-card hands, within a CPU budget."""
    preferred = engine_profile(state).preferred_hand
    ordered = sorted(candidates, key=lambda keep: _hold_potential(hand, keep, state, preferred), reverse=True)
    selected = ordered[:min(6, limit)]
    # Keep the preferred Burnt upgrade eligible in a 13-card hand even when
    # ordinary retained cores fill the initial shortlist.
    upgrade = next((keep for keep in ordered if candidates[keep] == 'Burnt upgrade ' + preferred), None)
    if upgrade is not None and upgrade not in selected:
        selected.append(upgrade)
    groups = set()
    for keep in ordered:
        reason = candidates[keep]
        if reason.startswith('Burnt upgrade '):
            group = reason
        elif 'Castle' in reason or 'Yorick' in reason or 'Trading Card' in reason:
            group = 'discard_growth'
        elif 'Flower Pot' in reason:
            group = 'distinct_suits'
        elif 'steel' in reason or 'Baron' in reason or 'Shoot' in reason:
            group = 'held_multiplier'
        elif 'straight flush' in reason:
            group = 'straight_flush'
        elif 'straight' in reason:
            group = 'straight'
        elif 'flush' in reason:
            group = 'flush'
        elif 'grow' in reason or 'three of' in reason:
            group = 'rank_groups'
        else:
            group = 'cycle'
        if group not in groups:
            groups.add(group)
            if keep not in selected:
                selected.append(keep)
    for keep in ordered:
        if len(selected) >= limit:
            break
        if keep not in selected:
            selected.append(keep)
    return {keep: candidates[keep] for keep in selected[:limit]}


def first_discard(state):
    if state.round_discards_used >= 0:
        return state.round_discards_used == 0
    # Old checkpoints have no counter. Infer only from the intact capacity.
    return state.discards_left > 0 and state.discards_left == _shop_round_state(state).discards_left


def advance_projected_discard(state, discarded, drawn=None):
    """One shared model for real-plan samples, round budgets and a second draw.

    These are predictions only. Actual action journals still record observed
    counters; an unreadable Yorick counter never becomes an invented counter.
    """
    drawn = len(discarded) if drawn is None else drawn
    levels = dict(state.levels)
    if (first_discard(state) and state.effect_count('burnt_joker')
            and discarded and not any(c.face_down for c in discarded)):
        category = classify(discarded, state).category
        levels[category] = state.level(category) + state.effect_count('burnt_joker')
    jokers = []
    money = state.money
    for j in state.jokers:
        if j.debuffed:
            jokers.append(j)
            continue
        value, data = j.value, dict(j.data)
        if j.key == 'green_joker':
            value = max(0, value - 1)
        elif j.key == 'ramen':
            value = max(1, (value if j.get('observed_value') else value or 2) - .01 * len(discarded))
        elif j.key == 'castle' and 'suit' in data:
            suit = SUITS[int(data['suit']) % 4]
            value += 3 * sum(not _card_debuffed(c, state) and suit in _effective_suits(c, state)
                             for c in discarded)
        elif j.key == 'yorick' and 'discards_remaining' in data:
            remaining = int(data['discards_remaining'])
            threshold = int(data.get('discard_threshold', 0))
            if threshold > 0 and 1 <= remaining <= threshold:
                delta = len(discarded) - remaining
                if delta >= 0:
                    value = max(1, value) + (1 + delta // threshold) * data.get('discard_growth', 1)
                    data['discards_remaining'] = threshold - delta % threshold
                else:
                    data['discards_remaining'] = -delta
        jokers.append(replace(j, value=value, data=tuple(data.items())))
    from economy import discard_income
    money += discard_income(discarded, state)
    return replace(state, jokers=tuple(jokers), levels=levels, money=money,
                   discards_left=max(0, state.discards_left - 1),
                   round_discards_used=max(0, state.round_discards_used) + 1,
                   draw_pile_size=max(0, state.draw_pile_size - drawn) if state.draw_pile_size >= 0 else -1,
                   seen_cards=state.seen_cards + tuple(discarded))


def advance_projected_play(state, hand, indices, estimate):
    """Advance only counters supported by the actual simulated play."""
    category = estimate.category
    original = [hand[i] for i in indices]
    pattern, before = _before_scoring(classify(original, state), state)
    selected = list(pattern.played)
    scored = [selected[i] for i in pattern.scoring]
    hiker = state.effect_count('hiker')
    if hiker:
        for index in pattern.scoring:
            card = selected[index]
            if not _card_debuffed(card, state):
                selected[index] = replace(card, bonus_chips=card.bonus_chips +
                                          5 * hiker * _card_retriggers(card, state, index, pattern))
    deck = list(state.deck_cards)
    if deck:
        for old, changed in zip(original, selected):
            if old != changed and old in deck:
                deck[deck.index(old)] = replace(changed, debuffed=False, face_down=False)
    history = dict(state.played_types)
    history[category] = history.get(category, 0) + 1
    round_history = dict(state.round_played_types)
    round_history[category] = round_history.get(category, 0) + 1
    levels = dict(state.levels)
    if _boss(state) == 'arm':
        # score_hand applies this play's downgrade before scoring. Persist it
        # for subsequent simulated hands, as the real Boss does.
        levels[category] = max(1, state.level(category) - 1)
    jokers = []
    for j in before.jokers:
        key, value, data = j.key, j.value, dict(j.data)
        if j.debuffed:
            jokers.append(j)
            continue
        if key == 'green_joker':
            value += 1
        elif key == 'ride_the_bus':
            value = 0 if any(_face(c, state) for c in scored) else value + 1
        elif key == 'runner' and 'straight' in category:
            value += 15
        elif key == 'square_joker' and len(indices) == 4:
            value += 4
        elif key == 'spare_trousers' and category in ('two_pair', 'full_house', 'flush_house'):
            value += 2
        elif key == 'obelisk':
            # Card.calculate_joker mutates the physical counter before this
            # hand scores. Copiers repeat its effect, not its growth event.
            most = max(state.played_types.values(), default=0)
            value = 1. if state.played(category) >= most else max(1., value) + .2
        elif key == 'ice_cream':
            value = max(0, (value if j.numeric_value_available else value or 100) - 5)
            data['observed_value'] = 1
            if value <= 0 and not data.get('counter_lower_bound'):
                continue
        elif key == 'seltzer':
            data['remaining_hands'] = max(0, data.get('remaining_hands', 10) - 1)
            if data['remaining_hands'] <= 0:
                continue
        elif key == 'loyalty_card':
            remaining = int(data.get('remaining', 5))
            data['remaining'] = 5 if remaining == 0 else remaining - 1
        jokers.append(replace(j, value=value, data=tuple(data.items())))
    return replace(state, jokers=tuple(jokers), levels=levels, deck_cards=tuple(deck), played_types=history,
                   round_played_types=round_history, boss_seen_types=frozenset(round_history),
                   boss_first_hand_done=True, hands_left=max(0, state.hands_left - 1),
                   boss_target_hand=(category if _boss(state) == 'mouth' and
                                     not state.boss_target_hand else state.boss_target_hand),
                   seen_cards=state.seen_cards + tuple(selected))


def _safe_play_score(hand, indices, estimate, state):
    if (_boss(state) == 'hook' or state.has('misprint') or state.has('bloodstone') or state.has('space_joker')
            or any(hand[i].enhancement == 'lucky' for i in indices)):
        held = [c for i, c in enumerate(hand) if i not in indices]
        floor = score_hand([hand[i] for i in indices], replace(state, score_mode='conservative'), held).score
        return .65 * estimate.score + .35 * floor
    return estimate.score


def conditional_multiplier_hits(hand, indices, state):
    """Evaluate the same scoring conditions as score_hand, including copies."""
    selected = [hand[i] for i in indices]
    held = [c for i, c in enumerate(hand) if i not in indices]
    if _boss(state) == 'hook' and len(held) <= 2:
        held = []  # Hook removes them before Blackboard can inspect the hand.
    pattern, scoring = _before_scoring(classify(selected, state), state)
    legal = _legal_play(pattern, state)
    attempts, hits = Counter(), Counter()
    for ability in scoring.abilities:
        if ability is None or ability.key not in CONDITIONAL_XMULT_JOKERS:
            continue
        attempts[ability.key] += 1
        hits[ability.key] += legal and _joker_global_effect(ability, pattern, held, scoring)[2] > 1
    return attempts, hits


def forecast_round(state, boss='', target=0, samples=6):
    from engine_development import blue_planet_yield
    start = _round_state(state, boss, target)
    deck = tuple(replace(c, debuffed=False, face_down=False) for c in full_deck(state))
    # An order is chosen once, before seeing forecast cards. Copy targets stay
    # consistent across the round rather than changing for each random hand.
    examples = (tuple(Card(r, 'H') for r in (2, 3, 4, 5, 6)),
                (Card(14, 'S'), Card(14, 'H'), Card(13, 'C'), Card(13, 'D'), Card(2, 'S')))
    def order_score(js):
        opened = advance_projected_blind_start(replace(start, jokers=js))
        return sum(_draw_best_plan(h, opened)[1].score for h in examples)
    order = max(_shop_order_variants(start), key=order_score)
    start = replace(start, jokers=order)
    marble = start.effect_count('marble_joker')
    start = advance_projected_blind_start(start)
    preferred = engine_profile(start).preferred_hand
    totals, hand_scores, categories = [], [], Counter()
    growth, real_plays, level_gains = Counter(), Counter(), Counter()
    multiplier_attempts, multiplier_hits = Counter(), Counter()
    blue_planets = 0.
    has_blue = any(c.seal.lower() == 'blue' for c in deck)
    for sample in range(samples):
        pile = list(deck)
        rng = Random(724901 + sample * 101)
        rng.shuffle(pile)
        if not state.deck_cards and not state.remaining_deck and state.deck_size > 0:
            # After composition was invalidated, game 2 had 46 cards while the
            # prior still had 52. Unknown identities remain sampled from the
            # prior, but the visible deck count limits draws and Blue Joker.
            if state.deck_size <= len(pile):
                pile = pile[:state.deck_size]
            else:
                pile.extend(rng.choices(deck, k=state.deck_size - len(pile)))
        if marble:
            pile.extend([Card(2, 'S', enhancement='stone')] * marble)
            rng.shuffle(pile)
        hand, pile = pile[:start.hand_size], pile[start.hand_size:]
        probe, total, floor_total = start, 0.0, 0.0
        training_done = False
        for _ in range(start.hands_left):
            if not hand:
                break
            probe = replace(probe, draw_pile_size=len(pile), round_score=total)
            indices, estimate = _draw_best_plan(hand, probe)
            # A discard hold is selected without consulting the draw pile.
            while (probe.discards_left > 0 and pile and estimate.score <
                   max(1, target - total) / max(1, probe.hands_left) * .95):
                keep = forecast_hold(hand, probe, preferred)
                discarded = [c for i, c in enumerate(hand) if i not in keep]
                draw = min(len(discarded), len(pile))
                if not draw:
                    break
                hand = [hand[i] for i in keep] + pile[:draw]
                pile = pile[draw:]
                probe = advance_projected_discard(probe, discarded, draw)
                indices, estimate = _draw_best_plan(hand, probe)
            earned = _safe_play_score(hand, indices, estimate, probe)
            if has_blue and not training_done:
                floor_total += _play_floor(indices, estimate, hand, probe)
            total += earned
            hand_scores.append(earned)
            categories[estimate.category] += 1
            after = advance_projected_play(probe, hand, indices, estimate)
            held = [c for i, c in enumerate(hand) if i not in indices]
            if _boss(probe) == 'hook':
                # Fixed seed, unrelated to the unseen pile's contents.
                hook = Random(1931 + sample * 31 + probe.hands_left)
                for _ in range(min(2, len(held))):
                    held.pop(hook.randrange(len(held)))
            if not training_done:
                attempts, hits = conditional_multiplier_hits(hand, indices, probe)
                multiplier_attempts.update(attempts)
                multiplier_hits.update(hits)
                real_plays[estimate.category] += 1 / samples
                # Count growth in the first playable segment of this blind;
                # later hypothetical hands cannot train a real cleared blind.
                old = Counter()
                new = Counter()
                for j in state.active_jokers:
                    if j.key in GROWTH_JOKERS:
                        old[j.key] += j.value
                for j in after.active_jokers:
                    if j.key in GROWTH_JOKERS:
                        new[j.key] += j.value
                if (total >= target > 0 or after.hands_left == 0
                        or not pile and not held):
                    for key in old.keys() | new.keys():
                        growth[key] += (new[key] - old[key]) / samples
                    for key, value in after.levels.items():
                        level_gains[key] += max(0, value - start.level(key)) / samples
                    if has_blue and floor_total >= target > 0:
                        held_for_reward = [c for i, c in enumerate(hand) if i not in indices]
                        planets = blue_planet_yield(held_for_reward, probe)
                        blue_planets += planets / samples
                        if not probe.observatory:
                            # Observatory may keep the generated Planet for
                            # X1.5 instead. Do not also claim it was consumed.
                            level_gains[estimate.category] += planets / samples
                            growth['constellation'] += (.1 * planets *
                                after.joker_count('constellation') / samples)
                    training_done = True
            draw = min(max(0, start.hand_size - len(held)), len(pile))
            if _boss(probe) == 'serpent':
                draw = min(3, len(pile))
            hand, pile, probe = held + pile[:draw], pile[draw:], after
        totals.append(total)
    ordered = sorted(totals)
    lower = ordered[max(0, len(ordered) // 4)]
    safe = .7 * lower + .3 * median(ordered)
    # Six successful samples are not proof that a conditional engine is safe.
    # Price the observed lower tail, especially when only one hand is allowed.
    conditional = sum(start.has(k) for k in
                      ('flower_pot', 'blackboard', 'banner', 'ancient_joker', 'the_idol', 'bloodstone'))
    rates = tuple(sorted((key, multiplier_hits[key] / count)
                         for key, count in multiplier_attempts.items() if count))
    volatile_condition = any(key not in PHASE_XMULT_JOKERS and rate < .75 for key, rate in rates)
    if start.hands_left == 1 or conditional >= 2 or volatile_condition:
        safe = min(safe, .75 * ordered[0] + .25 * lower)
    if start.hands_left == 1:
        # There is no second hand to compensate for the weakest draw. Avoid
        # letting better samples lift a losing single-hand outcome above pace.
        safe = min(safe, ordered[0])
    frequency = tuple((k, v / max(1, sum(categories.values()))) for k, v in sorted(categories.items()))
    return RoundForecast(target, boss, sum(totals) / len(totals), safe,
                         sum(s >= target for s in totals) / len(totals) if target else 0,
                         median(hand_scores) if hand_scores else 0,
                         start.hands_left, start.discards_left, frequency, tuple(totals),
                         tuple(sorted(growth.items())), tuple(sorted(real_plays.items())),
                         tuple(sorted(level_gains.items())), blue_planets, rates)


def _age_temporary_engine(state):
    """Stress the engine after another round; do not invent purchased support."""
    jokers = []
    hand_size = state.hand_size
    for j in state.jokers:
        if j.key == 'seltzer':
            remaining = j.get('remaining_hands', 10) - max(3, state.hands_left)
            if remaining <= 0:
                continue
            j = replace(j, data=tuple({**dict(j.data), 'remaining_hands': remaining}.items()))
        if j.key == 'ice_cream':
            value = max(0, (j.value if j.numeric_value_available else j.value or 100)
                        - 5 * max(3, state.hands_left))
            if not value and not j.get('counter_lower_bound'):
                continue
            j = replace(j, value=value, data=tuple({**dict(j.data), 'observed_value': 1}.items()))
        elif j.key == 'popcorn':
            value = max(0, (j.value if j.numeric_value_available else j.value or 20) - 4)
            if not value and not j.get('counter_lower_bound'):
                continue
            j = replace(j, value=value, data=tuple({**dict(j.data), 'observed_value': 1}.items()))
        elif j.key == 'turtle_bean':
            bonus = j.get('hand_size_bonus', 5)
            hand_size -= 1
            if bonus <= 1:
                continue
            j = replace(j, data=tuple({**dict(j.data), 'hand_size_bonus': bonus - 1}.items()))
        jokers.append(j)
    return replace(state, jokers=tuple(jokers), hand_size=max(1, hand_size))


_FORECAST_CACHE = OrderedDict()


def shop_forecast(state):
    # These counters reset in _round_state and cannot affect any forecast.
    # Normalize the input too, so observation-only changes reuse a forecast.
    state = replace(_shop_round_state(state), round_score=0, seen_cards=(),
                    round_played_types={}, boss_seen_types=frozenset(),
                    boss_target_hand='', boss_first_hand_done=False, forced_card_index=-1)
    key = repr(state)
    if key in _FORECAST_CACHE:
        _FORECAST_CACHE.move_to_end(key)
        return _FORECAST_CACHE[key]
    next_boss = (state.upcoming_boss if state.blind_index == 1 and
                 state.upcoming_boss_ante == state.ante else '')
    next_samples = 12 if _boss(replace(state,boss=next_boss,boss_disabled=False)) == 'needle' else 6
    next_round = forecast_round(state, next_boss, next_blind_target(state), samples=next_samples)
    boss_round = None
    if (state.blind_index == 0 and state.upcoming_boss_ante == state.ante
            and state.upcoming_boss_target > 0):
        boss_samples = 12 if _boss(replace(state,boss=state.upcoming_boss,boss_disabled=False)) == 'needle' else 6
        boss_round = forecast_round(state, state.upcoming_boss, state.upcoming_boss_target, samples=boss_samples)
    aged = _age_temporary_engine(_shop_round_state(state))
    sustainable = (min(next_round.safe_score, forecast_round(
        aged, next_boss, next_blind_target(state), samples=next_samples).safe_score)
                   if aged != _shop_round_state(state) else next_round.safe_score)
    fragile_score, loss_probability = None, 0.
    bananas = [i for i, j in enumerate(aged.jokers)
               if j.key == 'gros_michel' and not j.debuffed and not j.get('eternal')]
    if len(bananas) == 1:
        # A bridge Joker can disappear before the next Boss. For a single
        # banana, use the local game's destruction probability over up to
        # three rounds; do not treat destruction as an observed fact.
        index = bananas[0]
        failed = replace(aged, jokers=tuple(j for i, j in enumerate(aged.jokers) if i != index))
        fragile_score = forecast_round(failed, next_boss, next_blind_target(state), samples=next_samples).safe_score
        horizon = max(1, min(3, 9 - state.ante))
        loss_probability = 1 - (1 - _probability(state, 6)) ** horizon
        sustainable = min(sustainable, (1 - loss_probability) * sustainable + loss_probability * fragile_score)
    result = ShopForecast(next_round, boss_round, sustainable,
                          engine_profile(state, next_round.frequencies,
                                         getattr(next_round, 'conditional_xmult_rates', ())), fragile_score, loss_probability)
    from engine_development import development_score
    result = replace(result, development_score=development_score(state, result))
    _FORECAST_CACHE[key] = result
    while len(_FORECAST_CACHE) > 80:
        _FORECAST_CACHE.popitem(last=False)
    return result


def deck_target_value(card, state):
    """Persistent card suitability, distinct from the current hand's score."""
    value = _card_future_value(card, state)
    deck = full_deck(state)
    ranks = Counter(c.rank for c in deck if not c.is_stone)
    suits = Counter(c.suit for c in deck if not c.is_stone)
    preferred = engine_profile(state).preferred_hand
    if preferred in ('pair', 'two_pair', 'three_kind', 'full_house', 'four_kind', 'five_kind'):
        value += max(0, ranks[card.rank] - 4) * 5
    if 'flush' in preferred or state.has('the_tribe') or state.has('crafty_joker'):
        value += max(0, suits[card.suit] - len(deck) / 4) * 1.5
    if ('straight' in preferred and not card.is_stone and not state.has('shortcut')):
        # A middle rank participates in more ordinary runs than an endpoint.
        # Weight each route by the *remembered* neighbouring rank counts;
        # do not prefer a middle card whose required neighbours were removed.
        strength = min(1., max(0, state.level('straight') - 1) / 3 +
                       .5 * bool(state.has('runner') or state.has('the_order')))
        size = 4 if state.has('four_fingers') else 5
        support = 0.
        for window in _rank_windows(state):
            if len(window) != size or card.rank not in window:
                continue
            route = 1.
            for rank in window:
                if rank != card.rank:
                    route *= min(1., ranks[rank] / 4)
            support += route
        value += 3 * strength * support
    if card.enhancement == 'steel' and state.has('mime'):
        value += 30
    if card.enhancement == 'glass':
        value += 20 * bool(state.has('hanging_chad') or state.has('hack') or
                           (card.is_face and state.has('sock_and_buskin')))
    if card.enhancement == 'lucky':
        value += 12 * bool(state.has('lucky_cat'))
        value += 8 * bool(state.has('hanging_chad') or state.has('hack'))
    if card.seal.lower() == 'red':
        value += 18 * bool(card.enhancement == 'glass' or card.enhancement == 'steel' or
                           (not card.is_stone and card.rank == 13 and state.has('baron')))
    if card.seal.lower() == 'blue':
        value += 8 * bool(state.level(preferred) > 1 or state.has('constellation'))
    if card.enhancement == 'gold' and state.has('mime'):
        value += 12
    return value


def improve_play_continuation(hand, state, best, contenders):
    """Compare near-best plays by the cards they leave for the following hand."""
    unseen = _standard_unseen(hand, state)
    available = min(len(unseen), state.draw_pile_size if state.draw_pile_size >= 0 else len(unseen))
    if not available:
        return best
    floor = best.estimate.score * .65
    candidates = [c for c in contenders if c.estimate.score >= floor]
    # Scoring-core variations and kickers matter. Rank by immediate utility
    # and retained structure, then reserve one representative per hand type.
    candidates.sort(key=lambda c: c.utility + .5 * _preserved_pattern_value(
        [card for i, card in enumerate(hand) if i not in c.indices], state), reverse=True)
    selected = [best]
    categories = {best.estimate.category}
    for c in candidates:
        if c.estimate.category not in categories:
            selected.append(c)
            categories.add(c.estimate.category)
    for c in candidates:
        if c.indices not in {p.indices for p in selected}:
            selected.append(c)
        if len(selected) >= 8:
            break
    selected = selected[:8]
    permutations = []
    for sample in range(8):
        pile = list(unseen)
        Random(88721 + state.seed + sample * 131).shuffle(pile)
        permutations.append(pile)
    winner, winner_value = best, float('-inf')
    for candidate in selected:
        held = [c for i, c in enumerate(hand) if i not in candidate.indices]
        after = advance_projected_play(state, hand, candidate.play_order or candidate.indices,
                                       candidate.estimate)
        draw_count = min(len(candidate.indices), available)
        next_scores = []
        for sample_index, pile in enumerate(permutations):
            new_hand = held + pile[:draw_count]
            probe = replace(after, draw_pile_size=max(0, available - draw_count),
                            round_score=state.round_score + candidate.estimate.score)
            if _boss(state) == 'hook' and held:
                retained = list(held)
                rng = Random(4177 + sample_index * 59 + state.seed)
                for _ in range(min(2, len(retained))):
                    retained.pop(rng.randrange(len(retained)))
                next_draw = min(available, draw_count + min(2, len(held)))
                new_hand = retained + pile[:next_draw]
                probe = replace(probe, draw_pile_size=max(0, available - next_draw))
            elif _boss(state) == 'serpent':
                next_draw = min(3, available)
                new_hand = held + pile[:next_draw]
                probe = replace(probe, draw_pile_size=max(0, available - next_draw))
            _, estimate = _draw_best_plan(new_hand, probe)
            next_scores.append(estimate.score)
        expected_next = sum(next_scores) / len(next_scores)
        lower_next = sorted(next_scores)[len(next_scores) // 4]
        wins = sum(candidate.estimate.score + value >= state.score_needed for value in next_scores) / len(next_scores)
        value = (candidate.utility + .45 * expected_next + .2 * lower_next
                 + wins * min(300, state.score_needed * .12))
        if value > winner_value:
            winner_value = value
            winner = replace(candidate, utility=value, reason=candidate.reason +
                             f'; next hand {expected_next:.0f}, lower quartile {lower_next:.0f}; preserve useful core')
    return winner


def second_discard_gain(original, keep, draws, first_scores, unseen, sample_state, state):
    """Value one additional discard only after observing the first draw.

    It is an option value: a good first result is played immediately. A bad
    first result may keep a new core, without peeking at the second draw.
    """
    preferred = engine_profile(state).preferred_hand
    pace = state.score_needed / max(1, state.hands_left)
    gains = []
    for sample, (draw, first_score) in enumerate(zip(draws[:8], first_scores[:8])):
        if first_score >= pace * 1.1:
            gains.append(0)
            continue
        pool = list(unseen)
        for c in draw:
            if c in pool:
                pool.remove(c)
        hand = [original[i] for i in keep] + list(draw)
        keep_again = forecast_hold(hand, sample_state, preferred)
        count = len(hand) - len(keep_again)
        limit = len(pool) if sample_state.draw_pile_size < 0 else min(len(pool), sample_state.draw_pile_size)
        count = min(count, limit)
        if not count:
            gains.append(0)
            continue
        after = advance_projected_discard(sample_state, [hand[i] for i in range(len(hand)) if i not in keep_again], count)
        cost = _discard_cost(sample_state, first_score)
        outcomes = []
        for second_sample in range(4):
            new = [hand[i] for i in keep_again] + Random(
                91291 + sample * 97 + second_sample * 997 + state.seed).sample(pool, count)
            outcomes.append(_draw_best_plan(new, after)[1].score)
        gains.append(max(0, sum(outcomes) / len(outcomes) - first_score - cost))
    # A single sampled improvement is not proof; leave most of that option's
    # value unclaimed and charge the future resource before comparing plans.
    return .35 * sum(gains) / max(1, len(gains))
