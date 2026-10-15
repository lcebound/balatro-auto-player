"""Bounded estimates of growth already supported by the observed engine.

These estimates affect policy utility only. They never update runtime state,
invent future shop offers, or authorize an unsafe hand over a verified clear.
"""
from collections import Counter
from dataclasses import replace
from random import Random
from itertools import combinations

from strategy import (HAND_LEVELS, score_hand, _draw_best_plan, _shop_order_variants,
                      _shop_round_state, _card_debuffed, _face, _boss)


def blue_planet_yield(held, state):
    """Bounded end-of-round production from known held cards, not new draws.

    Mime repeats Blue seals. Hook can remove any two held cards, so its
    guaranteed yield loses two sources before applying repeats/slot limits.
    """
    room = max(0, state.consumable_slots - state.consumables_held)
    sources = sum(c.seal.lower() == 'blue' and not c.face_down
                  and not _card_debuffed(c, state) for c in held)
    if _boss(state) == 'hook':
        sources = max(0, sources - 2)
    return min(room, sources * (1 + state.effect_count('mime')))


def development_score(state, forecast):
    from planning import full_deck, _age_temporary_engine
    horizon = min(6., max(0., 8 - state.ante) * 2.) * .65
    base = forecast.sustainable_score
    if not horizon:
        return base
    sample = forecast.next_round
    growth = dict(sample.growth_per_round)
    plays = dict(sample.plays_per_round)
    levels = dict(sample.level_gains_per_round)
    sale_growth = (state.has('swashbuckler') and (state.has('egg') or state.has('gift_card')))
    if not (any(v != 0 for v in growth.values()) or levels
            or (state.has('supernova') and plays) or sale_growth):
        return base
    physical = Counter(j.key for j in state.active_jokers)
    future_jokers = []
    for j in state.jokers:
        gain = growth.get(j.key, 0) / max(1, physical[j.key]) * horizon
        # A Bus reset or Green discard is part of the measured net growth.
        # Do not treat an engine losing counters as a growth investment.
        floor = 1. if j.key in ('hologram', 'constellation', 'vampire', 'lucky_cat',
                              'glass_joker', 'canio', 'yorick', 'obelisk') else 0.
        developed = replace(j, value=max(floor, j.value + gain)) if not j.debuffed else j
        if sale_growth:
            # Swashbuckler reads sale values; Egg's growth is not liquid cash.
            extra = state.joker_count('gift_card') + (3 if j.key == 'egg' and not j.debuffed else 0)
            developed = replace(developed, sell_value=j.sell_value + extra * horizon)
            if j.key == 'swashbuckler' and j.get('observed_value'):
                other_gain = sum(state.joker_count('gift_card') +
                    (3 if old.key == 'egg' and not old.debuffed else 0)
                    for old in state.jokers if old is not j)
                developed = replace(developed, value=developed.value + other_gain * horizon)
        future_jokers.append(developed)
    future = replace(state, jokers=tuple(future_jokers),
                     played_types={**state.played_types, **{
                         k: state.played(k) + v * horizon for k, v in plays.items()}},
                     levels={**state.levels, **{
                         k: state.level(k) + v * horizon for k, v in levels.items()}})
    current = _shop_round_state(state)
    # Don't rely on a consumable bridge to multiply those future counters.
    for _ in range(min(3, max(1, round(horizon)))):
        current = _age_temporary_engine(current)
        future = _age_temporary_engine(future)
    deck = tuple(replace(c, debuffed=False, face_down=False) for c in full_deck(state))
    if not deck:
        return base
    before, after = [], []
    for seed in (19031, 19237, 19507):
        hand = list(deck)
        Random(seed + state.seed).shuffle(hand)
        hand = hand[:max(1, current.hand_size)]
        # Use identical hands for both projections; no imagined future cards.
        for probe, result in ((current, before), (future, after)):
            probe = replace(probe, boss='', boss_disabled=False, round_score=0,
                            round_played_types={}, boss_seen_types=frozenset(),
                            boss_target_hand='', boss_first_hand_done=False,
                            draw_pile_size=max(0, state.deck_size-len(hand)),
                            target_score=0, score_mode='conservative')
            result.append(max(_draw_best_plan(hand, replace(probe, jokers=order))[1].score
                              for order in _shop_order_variants(probe)[:2]))
    # Damp the projection and cap it: three sample hands are directional
    # evidence, not a full simulation of several future Antes.
    ratio = (sum(after) + 20) / (sum(before) + 20)
    return base * min(3., max(.25, ratio))


def play_development_value(indices, estimate, hand, state, *, guaranteed_score=None):
    horizon = min(6., max(0., 8-state.ante))
    guaranteed = estimate.score if guaranteed_score is None else guaranteed_score
    if state.hands_left <= 1 and guaranteed < state.score_needed:
        return 0.
    if not horizon or estimate.score <= 0:
        return 0.
    selected = [hand[i] for i in indices]
    scored = [selected[i] for i in estimate.scoring]
    chips, mult = 0., 0.
    mult += state.effect_count('green_joker')
    if len(selected) == 4:
        chips += 4 * state.effect_count('square_joker')
    if 'straight' in estimate.category:
        chips += 15 * state.effect_count('runner')
    if estimate.category in ('two_pair', 'full_house', 'flush_house'):
        mult += 2 * state.effect_count('spare_trousers')
    if not any(_face(c, state) for c in scored):
        mult += state.effect_count('ride_the_bus')
    # Prefer repeated, supported hand types when they score nearly equally.
    if state.effect_count('supernova'):
        focus = max(state.played_types.values(), default=0)
        if state.played(estimate.category) >= focus:
            mult += state.effect_count('supernova')
    value = estimate.score * horizon * (
        chips / max(20., estimate.chips) + mult / max(5., estimate.multiplier)) * .25
    value = min(estimate.score * .08, value)
    if 0 < state.score_needed <= guaranteed:
        blue = blue_planet_yield([c for i, c in enumerate(hand) if i not in indices], state)
        if blue:
            c, m, dc, dm = HAND_LEVELS[estimate.category]
            level = state.level(estimate.category)-1
            improvement = dc/max(1., c+level*dc) + dm/max(1., m+level*dm)
            # A produced Planet both levels the winning hand and feeds every
            # physical Constellation. Blueprint copies the scoring multiplier,
            # but does not create another physical growth counter.
            xratio = 1.
            for j in state.abilities:
                if j is not None and j.key == 'constellation':
                    xratio *= 1 + .1 * blue / max(1., j.value)
            benefit = improvement * blue + xratio - 1
            value += min(estimate.score * .2, estimate.score * benefit *
                         min(.4, .15 + horizon * .04))
    return value


def safe_training_play(hand, state, current):
    """One early training play backed by a finisher already in the hand."""
    from planning import advance_projected_play
    from economy import investment_rounds
    from strategy import PlayDecision
    if (state.ante > 4 or state.hands_left < 3 or state.round_played_types
            or _boss(state) or state.forced_card_index >= 0
            or any(c.face_down for c in hand)
            or any(state.has(k) for k in ('blackboard', 'raised_fist', 'ice_cream',
                                         'seltzer', 'ramen', 'obelisk'))):
        return current
    scalers = ('green_joker', 'ride_the_bus', 'square_joker', 'runner', 'spare_trousers', 'supernova')
    if not any(state.has(k) for k in scalers) or investment_rounds(state) < 3:
        return current
    probe = replace(state, score_mode='conservative')
    candidates = []
    for size in range(1, min(5, len(hand)) + 1):
        for indices in combinations(range(len(hand)), size):
            held = [c for i, c in enumerate(hand) if i not in indices]
            result = score_hand([hand[i] for i in indices], probe, held)
            if not 0 < result.score < state.score_needed:
                continue
            value = play_development_value(indices, result, hand, state)
            if value > 0 and held:
                candidates.append((value, indices, result, held))
    best, best_value = current, 0.
    for _, indices, result, held in sorted(candidates, key=lambda c: c[0], reverse=True)[:8]:
        after = advance_projected_play(probe, hand, indices, result)
        after = replace(after, target_score=state.score_needed - result.score,
                        round_score=0, draw_pile_size=0)
        finish_indices, finish = _draw_best_plan(held, after)
        if finish.score < after.score_needed * 1.2:
            continue
        before = score_hand([held[i] for i in finish_indices], probe,
                            [c for i, c in enumerate(held) if i not in finish_indices])
        ratio = finish.score / max(1., before.score)
        if ratio < 1.08:
            continue
        value = min(1., ratio - 1.) * investment_rounds(state) - .35
        if value > best_value:
            best_value = value
            best = PlayDecision(indices, result, current.utility + value,
                f'early training; known retained finisher {finish.score:.0f} for {after.score_needed:.0f}; projected growth ratio {ratio:.2f}', indices, training=True)
    return best
