"""Installed-game Boss tactics shared by runtime actions and shop forecasts.

The catalogue describes responses, not promises of beating every seed.
Forecast-only state transitions never authorize a real sale or reveal cards.
"""
from dataclasses import replace
from random import Random

# Keys match RunState._boss_name. Rules verified against local blind.lua and
# localization/en-us.lua on 2026-10-18. Runtime mechanisms remain authoritative.
BOSS_TACTICS = {
    'hook': ('protect held effects from two random removals', 'prefer played-card scoring over fragile held engines'),
    'ox': ('avoid the taxed hand when survival permits', 'protect cash-dependent scoring and diversify hand types'),
    'house': ('cycle unknown opening cards; keep identities unknown', 'value discards and scoring independent of ranks'),
    'wall': ('spend resources on the larger actual target', 'build sufficient sustainable score'),
    'wheel': ('keep face-down cards unknown; replace weak unknowns', 'favor reliable global scoring'),
    'arm': ('apply the level loss before each score', 'favor Joker scoring over a single high-level hand'),
    'club': ('avoid relying on debuffed Club effects', 'diversify suits and global scoring'),
    'fish': ('use visible opening cards; later play draws are unknown', 'value opening-hand strength and discards'),
    'psychic': ('play exactly five cards, retaining useful held cards when possible', 'avoid engines requiring short plays'),
    'goad': ('avoid relying on debuffed Spade effects', 'diversify suits and global scoring'),
    'water': ('never plan a discard', 'value extra hands, draw consistency and hand size'),
    'window': ('avoid relying on debuffed Diamond effects', 'diversify suits and global scoring'),
    'manacle': ('plan with one less card in hand', 'value hand size and compact hand types'),
    'eye': ('score different hand categories; cycle when necessary', 'keep multiple viable hand types'),
    'mouth': ('choose a repeatable opening category, then preserve its lock', 'prefer consistent repeatable scoring'),
    'plant': ('face cards lose their card effects', 'avoid dependence on face-card triggers and held Kings'),
    'serpent': ('draw only three after play or discard', 'prefer efficient cycling and useful retained cards'),
    'pillar': ('ante-used cards are debuffed; preserve fresh scoring cards', 'consider accumulated ante card usage'),
    'needle': ('use discards to prepare the single scoring hand', 'prioritize one-hand reliability and extra-hand effects'),
    'head': ('avoid relying on debuffed Heart effects', 'diversify suits and global scoring'),
    'tooth': ('play fewer cards when the score permits', 'protect cash-dependent engines and purchase reserves'),
    'flint': ('halve only base Chips and Mult before Joker effects', 'favor Joker-provided Chips and Mult'),
    'mark': ('face cards remain unknown until played; discard for visible cards', 'favor non-face and global scoring'),
    'amber_acorn': ('use order-independent bounds; never read hidden Joker titles', 'reduce copier/order dependence'),
    'verdant_leaf': ('sell a verified sellable Joker before drawing for score', 'reserve a low-impact Joker to sell inside the Boss'),
    'violet_vessel': ('plan against the displayed very large target', 'prioritize sustainable multiplication and retriggers'),
    'crimson_heart': ('reobserve the disabled Joker after each play', 'prefer redundant scoring sources over one critical Joker'),
    'cerulean_bell': ('include the observed forced card in every play', 'favor hands tolerant of an extra required card'),
}


def future_boss_weight(ante):
    """Small optional preparation cost; larger in the final two goal Antes."""
    return (.03, .03, .04, .04, .05, .06, .12, .18)[min(8, max(1, ante)) - 1]


def remember_upcoming_boss(state, ante, observation):
    """A missed new-ante title never carries the previous Boss forward."""
    if state.upcoming_boss_ante != ante:
        state = replace(state, upcoming_boss='', upcoming_boss_target=0, upcoming_boss_ante=0)
    if state.ante != ante:
        state = replace(state, pillar_cards=())
    if observation:
        name, target = observation
        state = replace(state, upcoming_boss=name, upcoming_boss_target=target, upcoming_boss_ante=ante)
    return state


def after_joker_sale(state, index, *, disable_leaf=False, project=False):
    from strategy import joker_hand_size_delta
    old = state.jokers[index]
    if old.get('eternal'):
        raise ValueError('Eternal Joker cannot be sold')
    jokers = tuple(j for i, j in enumerate(state.jokers) if i != index)
    if project:
        jokers = tuple(replace(j, value=max(1., j.value) + .25)
                       if j.key == 'campfire' and not j.debuffed else j for j in jokers)
    return replace(state, jokers=jokers,
        total_joker_slots=state.total_joker_slots - (old.edition == 'negative'),
        hand_size=max(1, state.hand_size + joker_hand_size_delta(state.jokers, jokers)),
        # Burglar's already granted hands are not revoked by selling it.
        discards_left=max(0, state.discards_left - {'drunkard': 1, 'merry_andy': 3}.get(old.key, 0)),
        boss_disabled=state.boss_disabled or disable_leaf)


def choose_verdant_sale(state):
    """Rank legal sacrifices using three shared bounded draws, not full rounds."""
    from strategy import _boss, _draw_best_plan, _shop_order_variants, Card, SUITS
    if _boss(state) != 'verdant_leaf':
        return None
    legal = [i for i, j in enumerate(state.jokers) if not j.get('eternal')]
    if not legal:
        return None
    deck = [replace(c, debuffed=False, face_down=False) for c in
            (state.deck_cards or state.remaining_deck or tuple(Card(r, s) for s in SUITS for r in range(2, 15)))]
    rng = Random(18018)
    draws = [tuple(rng.sample(deck, min(state.hand_size, len(deck)))) for _ in range(3)]
    best = None
    for index in legal:
        after = after_joker_sale(state, index, disable_leaf=True, project=True)
        after = replace(after, score_mode='conservative', forced_card_index=-1)
        value = 0.
        # A sale changes copier adjacency. Compare realizable bounded orders,
        # with one order shared by all three draws, just as shop forecasts do.
        for order in _shop_order_variants(after):
            ordered = replace(after, jokers=order)
            scores = [_draw_best_plan(hand[:after.hand_size], ordered)[1].score for hand in draws if hand]
            value = max(value, min(scores, default=0) * .6 + sum(scores) / max(1, len(scores)) * .4)
        candidate = (value, -int(state.jokers[index].key in ('blueprint', 'brainstorm')), -index)
        if best is None or candidate > best[0]:
            best = candidate, index
    return best[1]


def forecast_draw(cards, state, rng, event):
    """Conceal unknown identities before planning; no access to hidden fronts."""
    from strategy import Card, _boss, _face, _probability
    boss = _boss(state)
    if boss not in ('house', 'fish', 'mark', 'wheel'):
        return list(cards)
    return [Card(0, '?', face_down=True) if (
        boss == 'house' and event == 'initial'
        or boss == 'fish' and event == 'play'
        or boss == 'mark' and _face(card, state)
        or boss == 'wheel' and rng.random() < _probability(state, 7)
    ) else card for card in cards]


def forecast_boss_turn(state, hand, rng):
    """Projected random restrictions; each sample retains its own sequence."""
    from strategy import _boss
    boss = _boss(state)
    if boss == 'cerulean_bell':
        return replace(state, forced_card_index=rng.randrange(len(hand)) if hand else -1)
    if boss == 'crimson_heart':
        previous = {i for i, j in enumerate(state.jokers) if j.get('_forecast_heart_disabled')}
        jokers = tuple(replace(j, debuffed=False,
            data=tuple((k, v) for k, v in j.data if k != '_forecast_heart_disabled'))
            if i in previous else j for i, j in enumerate(state.jokers))
        eligible = [i for i, j in enumerate(jokers) if not j.debuffed and i not in previous]
        eligible = eligible or [i for i, j in enumerate(jokers) if not j.debuffed]
        if eligible:
            index = rng.choice(eligible)
            jokers = tuple(replace(j, debuffed=True,
                data=(*j.data, ('_forecast_heart_disabled', 1))) if i == index else j
                for i, j in enumerate(jokers))
        return replace(state, jokers=jokers)
    return state


def discard_draw_count(state, discarded, available):
    from strategy import _boss
    return min(available, 3 if _boss(state) == 'serpent' and discarded else discarded)


def legal_discard_keeps(hand, state, candidates):
    """Bell's forced card is selected for a discard as well as for a play."""
    from strategy import _boss, _card_future_value
    forced = state.forced_card_index
    if _boss(state) != 'cerulean_bell' or not 0 <= forced < len(hand):
        return candidates
    result = {}
    for keep, reason in candidates.items():
        kept = set(keep) - {forced}
        if len(hand) - len(kept) > 5:
            spare = [i for i in range(len(hand)) if i not in kept and i != forced]
            if spare:
                kept.add(max(spare, key=lambda i: _card_future_value(hand[i], state)))
        if 1 <= len(hand) - len(kept) <= 5:
            result[tuple(sorted(kept))] = reason
    return result
