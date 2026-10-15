"""Per-game hand levels, updated by confirmed action journals.

Fixed scoring increments live in strategy.HAND_LEVELS. Never accumulate an
expected Space Joker trigger as an actual level; those rows need observation.
"""
from dataclasses import replace

from strategy import HAND_LEVELS, classify, _boss, _legal_play

ORDINARY = frozenset(tuple(HAND_LEVELS)[:9])


def fresh_levels(state):
    return replace(state, levels={key: 1 for key in HAND_LEVELS},
                   levels_initialized=True, uncertain_levels=(), hand_history_dirty=False)


def needs_observation(state):
    # Secret hands may be invisible in Run Info until played. Keep their
    # uncertainty, but inspect them only if this hand can actually make one.
    visible = ORDINARY | {k for k in HAND_LEVELS if state.played(k) > 0}
    return (not state.levels_initialized or bool(set(state.uncertain_levels) & visible)
            or state.hand_history_dirty)


def secret_candidates(state, cards):
    from itertools import combinations
    wanted = set(state.uncertain_levels) - ORDINARY
    found = {}
    if not wanted:
        return found
    for size in range(1, min(5, len(cards)) + 1):
        for indices in combinations(range(len(cards)), size):
            if state.forced_card_index >= 0 and state.forced_card_index not in indices:
                continue
            selected = [cards[i] for i in indices]
            if any(c.face_down for c in selected):
                continue
            category = classify(selected, state).category
            if category in wanted:
                found.setdefault(category, indices)
    return found


def sync_cache(session, state=None):
    state = state or session.policy_state
    session.levels_cache = dict(state.levels)
    session.levels_dirty = needs_observation(state)
    session.refresh_levels = session.levels_dirty


def uncertain(state, categories):
    return replace(state, uncertain_levels=tuple(sorted(set(state.uncertain_levels) | set(categories))))


def after_discard(before, after, cards):
    count = before.effect_count('burnt_joker')
    # A shuffled, face-down copier can have an unknown target.
    unknown_copy = before.joker_order_unknown and any(
        j.key in ('blueprint', 'brainstorm') for j in before.active_jokers)
    if not cards or before.round_discards_used > 0 or not (count or unknown_copy):
        return after
    if (before.round_discards_used < 0 or unknown_copy
            or any(c.face_down for c in cards)):
        return uncertain(after, HAND_LEVELS)
    category = classify(cards, before).category
    return replace(after, levels={**after.levels, category: before.level(category) + count})


def after_play(before, after, category, cards=None):
    # Arm applies before Space Joker in the installed game. Its reduction is
    # permanent, respects Chicot/disabled bosses, and stops at level one.
    known = category in HAND_LEVELS
    if _boss(before) == 'arm':
        if known:
            after = replace(after, levels={**after.levels,
                            category: max(1, before.level(category) - 1)})
        else:
            after = uncertain(after, HAND_LEVELS)
    unknown_copy = before.joker_order_unknown and any(
        j.key in ('blueprint', 'brainstorm') for j in before.active_jokers)
    legal = not known or cards is None or _legal_play(classify(cards, before), before)
    if legal and (before.effect_count('space_joker') or unknown_copy):
        after = uncertain(after, (category,) if known else HAND_LEVELS)
    if not known:
        after = replace(after, hand_history_dirty=True)
    return after
