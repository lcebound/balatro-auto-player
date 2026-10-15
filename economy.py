"""Economy policy estimates for an Ante-8 clear, never observed income.

Cash, sale-value growth and scoring growth have separate values. Random
targets in future rounds are averaged, never copied from today's tooltip.
"""
from dataclasses import replace

from strategy import _boss, _card_debuffed, _face, _probability


def investment_rounds(state):
    # The final Boss payout cannot fund another shop before the goal.
    remaining = max(0, (8 - state.ante) * 3 + (1 - state.blind_index
                                             if state.blind_index >= 0 else 2))
    if state.blind_index == 2 and state.upcoming_boss_ante < state.ante:
        remaining = max(0, (9 - state.ante) * 3 - 1)
    return min(6, remaining)


def interest(state, money=None):
    if state.deck_type == 'Green Deck':
        return 0
    return min(max(0, state.money if money is None else money), state.interest_cash_cap) // 5


def delayed_income(state):
    from planning import first_discard
    return (2 * state.discards_left * state.joker_count('delayed_gratification')
            if state.discards_left > 0 and first_discard(state) else 0.)


def discard_income(discarded, state):
    from planning import first_discard
    cash = 3 * state.joker_count('trading_card') if first_discard(state) and len(discarded) == 1 else 0
    for ability in state.abilities:
        if ability is not None and ability.key == 'mail_in_rebate':
            rank = ability.get('rank', -1)
            cash += 5 * sum(c.rank == rank and not c.is_stone and not c.face_down
                            and not _card_debuffed(c, state) for c in discarded)
    return cash


def joker_income(joker, state, forecast):
    """Expected spendable dollars per round, not sale value or a guarantee."""
    from planning import full_deck
    if joker.debuffed:
        return 0.
    deck = full_deck(state)
    size = max(1, len(deck))
    sample = forecast.next_round
    plays = max(1., sum(dict(getattr(sample, 'plays_per_round', ())).values()))
    faces = sum(_face(c, state) for c in deck if not c.face_down) / size
    key = joker.key
    if key == 'golden_joker':
        return 4.
    if key == 'rocket':
        return joker.get('income', 1) + .5 * max(0, investment_rounds(state) - 2)
    if key == 'cloud_9':
        return joker.get('income', sum(c.rank == 9 and not c.is_stone for c in deck))
    if key == 'to_the_moon':
        return float(interest(state))
    if key == 'satellite':
        return joker.get('income', 0)  # No invented Planet-use history.
    if key == 'delayed_gratification':
        capacity = state.base_discards_per_round + state.joker_count('drunkard') + 3 * state.joker_count('merry_andy')
        if state.effect_count('burglar') or _boss(sample_state(state, sample)) == 'water':
            return 0.
        return 2 * capacity * (.75 if not forecast.weak else .2)
    if key == 'mail_in_rebate':
        # Rebate targets change each round. A modest random-target draw
        # estimate rewards the route without promising today's rank tomorrow.
        if not getattr(sample, 'discards', state.base_discards_per_round):
            return 0.
        seen = min(size, state.hand_size + 5 * min(2, getattr(sample, 'discards', state.base_discards_per_round)))
        return 5 * seen / 13 * .55
    if key == 'trading_card':
        return 2.4 if getattr(sample, 'discards', state.base_discards_per_round) and not forecast.weak else 0.
    if key == 'business_card':
        return 2 * _probability(state, 2) * faces * min(5., 2.5 * plays)
    if key == 'reserved_parking':
        return _probability(state, 2) * faces * max(0, state.hand_size - 3) * plays * (1 + state.effect_count('mime'))
    if key == 'golden_ticket':
        return 4 * sum(c.enhancement == 'gold' for c in deck) / size * min(5., 2.5 * plays)
    if key == 'rough_gem':
        return sum(c.suit == 'D' and not c.is_stone for c in deck) / size * min(5., 2.5 * plays)
    if key == 'to_do_list':
        # Future targets are random. Known-target execution is scored by
        # score_hand; buying this is a discounted opportunity, not fixed cash.
        return 4 * plays * .2
    return 0.


def sample_state(state, sample):
    return replace(state, boss=getattr(sample, 'boss', ''), boss_disabled=False)


def economy_value(state, forecast):
    rounds = investment_rounds(state)
    if not rounds:
        return 0.
    liquid = sum(joker_income(j, state, forecast) for j in state.active_jokers)
    # A second income slot is less useful once there is plenty of cash; it
    # must compete with scoring in the complete replacement forecast.
    weight = 1. if state.money < 25 else (.75 if state.money < 50 else .4)
    value = min(38., liquid * rounds * .65) * weight
    for j in state.active_jokers:
        if j.key == 'egg':
            value += 3 * rounds * .3
        elif j.key == 'gift_card':
            value += (len(state.jokers) + state.consumables_held) * rounds * .22
    return value


def safe_income_discard(hand, state, current):
    """Trigger income while keeping a known finishing hand; no lucky draw."""
    from planning import advance_projected_discard
    from strategy import DiscardDecision, _draw_best_plan, _card_future_value
    if state.discards_left <= 0 or _boss(state) in ('water', 'hook', 'fish', 'house', 'wheel'):
        return None
    spare = [i for i, c in enumerate(hand) if not c.face_down]
    targets = {int(a.get('rank', -1)) for a in state.abilities
               if a is not None and a.key == 'mail_in_rebate'}
    matching = sorted((i for i in spare if hand[i].rank in targets
                       and not hand[i].is_stone and not _card_debuffed(hand[i], state)),
                      key=lambda i: _card_future_value(hand[i], state))
    candidates = [tuple(matching[:size]) for size in range(1, min(5, len(matching)) + 1)]
    if state.has('trading_card'):
        expendable = [i for i in spare if not (hand[i].enhancement or hand[i].edition
                       or hand[i].seal or hand[i].bonus_chips)
                      and _card_future_value(hand[i], state) <= 8]
        candidates.extend((i,) for i in sorted(expendable, key=lambda i: _card_future_value(hand[i], state))[:3])
    best = None
    for dropped in candidates:
        if not dropped:
            continue
        cards = [hand[i] for i in dropped]
        if any(c.seal.lower() == 'blue' or c.enhancement in ('gold', 'steel') for c in cards):
            continue
        cash = discard_income(cards, state) - delayed_income(state)
        if cash <= 0:
            continue
        after = advance_projected_discard(state, cards, drawn=0)
        retained = [c for i, c in enumerate(hand) if i not in dropped]
        if not retained:
            continue
        _, result = _draw_best_plan(retained, replace(after, score_mode='conservative'))
        # Keep some margin for unknown drawn cards and replan after observing.
        if result.score < state.score_needed * 1.1:
            continue
        if any(a is not None and a.key in ('blackboard', 'raised_fist') for a in state.abilities):
            continue  # Unknown drawn ranks/suits can weaken held effects.
        utility = cash
        if best is None or utility > best.utility:
            best = DiscardDecision(dropped, result.score, utility,
                f'income discard ${cash:.0f}; retained conservative clear {result.score:.0f}; reobserve before playing')
    return best
