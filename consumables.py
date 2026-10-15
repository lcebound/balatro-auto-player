"""Tactical use of Tarot, Planet and Spectral cards.

Pure decisions only. Random transformations are valued as uncertain outcomes;
the controller uses only transformations it can select and verify visibly.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from itertools import combinations

from strategy import (Card, RunState, PLANET_HAND, SUITS, joker_name,
                      decide_play, _draw_estimate, _card_future_value,
                      _card_debuffed, score_hand, _face)
from planning import deck_target_value, full_deck, engine_profile


@dataclass(frozen=True)
class Consumable:
    name: str
    kind: str

    @property
    def key(self):
        return joker_name(self.name)


@dataclass(frozen=True)
class ConsumableDecision:
    action: str = "hold"
    indices: tuple[int, ...] = ()
    utility: float = 0
    expected_score: float = 0
    reason: str = "keep for a useful hand"
    deterministic: bool = True
    play_order: tuple[int, ...] = ()


ENHANCEMENT_TAROTS = {
    "the_magician": ("lucky", 2), "the_empress": ("mult", 2),
    "the_hierophant": ("bonus", 2), "the_lovers": ("wild", 1),
    "the_chariot": ("steel", 1), "justice": ("glass", 1),
    "the_devil": ("gold", 1), "the_tower": ("stone", 1),
}
SUIT_TAROTS = {"the_star": "D", "the_moon": "C", "the_sun": "H", "the_world": "S"}
SEAL_SPECTRALS = {"talisman": "gold", "deja_vu": "red", "trance": "blue", "medium": "purple"}


def transformed_hand(hand, consumable, indices):
    """Exact deterministic card transformations, retaining edition and seal."""
    result = list(hand)
    key = consumable.key
    if key in ENHANCEMENT_TAROTS:
        enhancement, _ = ENHANCEMENT_TAROTS[key]
        for i in indices:
            result[i] = replace(result[i], enhancement=enhancement, rankless=enhancement == "stone")
    elif key in SUIT_TAROTS:
        for i in indices:
            result[i] = replace(result[i], suit=SUIT_TAROTS[key])
    elif key == "strength":
        for i in indices:
            result[i] = replace(result[i], rank=2 if result[i].rank == 14 else result[i].rank + 1)
    elif key == "death":
        if len(indices) != 2:
            raise ValueError("Death needs two selected cards")
        left, right = sorted(indices)
        result[left] = result[right]  # Direction is physical left -> right.
    elif key == "the_hanged_man":
        result = [card for i, card in enumerate(result) if i not in indices]
    elif key in SEAL_SPECTRALS:
        for i in indices:
            result[i] = replace(result[i], seal=SEAL_SPECTRALS[key])
    elif key == "cryptid":
        for i in indices:
            result.extend([result[i], result[i]])
    else:
        raise ValueError(f"Transformation is not deterministic: {consumable.name}")
    return result


def _choices(hand, maximum, state):
    choices = [selection for size in range(1, maximum + 1)
               for selection in combinations(range(len(hand)), size)]
    if len(choices) <= 90:
        return choices
    # Include strong targets and trash targets: Death/Hanged Man need both.
    strong = sorted(range(len(hand)), key=lambda i: deck_target_value(hand[i], state), reverse=True)
    important = set(strong[:6] + strong[-3:])
    # Keep all single targets, all pair targets in ordinary/13-card hands and
    # representatives of every suit. Do not truncate by left-to-right index.
    important.update(next(i for i, c in enumerate(hand) if c.suit == suit)
                     for suit in SUITS if any(c.suit == suit for c in hand))
    return [selection for selection in choices if len(selection) <= 2 or
            set(selection) <= important]


def transformed_state(state, hand, modified):
    """Apply a deterministic deck edit to remembered composition, if known."""
    old = [replace(c, debuffed=False, face_down=False) for c in hand]
    new = [replace(c, debuffed=False, face_down=False) for c in modified]
    removed, added = Counter(old) - Counter(new), Counter(new) - Counter(old)
    uncertain = set(state.unverified_chip_cards)
    if any(card.identity in uncertain for card in old):
        # Strength/suit edits/copies can carry invisible chips to another
        # identity, especially when a debuffed tooltip hides the real value.
        uncertain.update(card.identity for card in added)
    deck = list(state.deck_cards)
    if deck:
        for card, count in removed.items():
            for _ in range(count):
                match = next((i for i, c in enumerate(deck) if c == card), None)
                if match is None:
                    # Unknown duplication/destruction must invalidate exact
                    # composition instead of silently subtracting a wrong card.
                    deck = []
                    break
                deck.pop(match)
            if not deck:
                break
        if deck:
            for card, count in added.items():
                deck.extend([card] * count)
    size = state.deck_size + len(modified) - len(hand)
    jokers = state.jokers
    additions = sum(added.values()) if len(modified) > len(hand) else 0
    if additions:
        jokers = tuple(replace(j, value=max(1, j.value) + .25 * additions)
                       if j.key == 'hologram' and not j.debuffed else j for j in jokers)
    if len(modified) < len(hand):
        faces = sum(count for c, count in removed.items() if _face(c, state))
        jokers = tuple(replace(j, value=max(1, j.value) + faces)
                       if j.key == 'canio' and not j.debuffed else j for j in jokers)
    return replace(state, deck_cards=tuple(deck), deck_size=size, jokers=jokers,
                   unverified_chip_cards=tuple(sorted(uncertain)))


def consumed_engine(state, consumable):
    """Predict deterministic use triggers, including copied scoring targets.

    Physical scalers grow once; copies subsequently read that target value.
    This does not create random Jokers or invent unseen card transformations.
    """
    key = consumable.key
    jokers = []
    for j in state.jokers:
        if not j.debuffed:
            if consumable.kind.lower() == 'tarot' and j.key == 'fortune_teller':
                j = replace(j, value=j.value + 1)
            if key in PLANET_HAND and j.key == 'constellation':
                j = replace(j, value=max(1, j.value) + .1)
        jokers.append(j)
    levels, planets = dict(state.levels), dict(state.planet_cards)
    if key in PLANET_HAND or key == 'black_hole':
        categories = [PLANET_HAND[key]] if key in PLANET_HAND else list(PLANET_HAND.values())
        for category in categories:
            levels[category] = state.level(category) + 1
        if key in PLANET_HAND:
            category = PLANET_HAND[key]
            planets[category] = max(0, planets.get(category, 0) - 1)
    return replace(state, jokers=tuple(jokers), levels=levels, planet_cards=planets)


def state_after_consumable(state, hand, consumable, plan):
    key = consumable.key
    if key in ENHANCEMENT_TAROTS or key in SUIT_TAROTS or key in SEAL_SPECTRALS or key in (
            'strength', 'death', 'the_hanged_man', 'cryptid'):
        state = transformed_state(state, hand, transformed_hand(hand, consumable, plan.indices))
    state = consumed_engine(state, consumable)
    return replace(state, consumables_held=max(0, state.consumables_held - 1))


def decide_consumable(consumable, hand, state):
    key = consumable.key
    current = decide_play(hand, state)
    score = current.estimate.score
    best = ConsumableDecision(expected_score=score)
    space = max(0, state.consumable_slots - state.consumables_held + 1)
    used_engine = consumed_engine(state, consumable)
    if key in PLANET_HAND or key == "black_hole":
        upgraded = used_engine
        improved = decide_play(hand, upgraded).estimate.score
        if state.observatory and key in PLANET_HAND:
            if improved <= score:
                return replace(best, reason="Observatory multiplier is worth more while this planet stays held")
        return ConsumableDecision("use", expected_score=improved,
            utility=max(1, improved - score), reason="permanent hand level upgrade")
    if key == "the_hermit":
        cash = min(20, state.money)
        funded = replace(used_engine, money=state.money + cash)
        improved = _draw_estimate(hand, funded)
        urgent = score < state.score_needed <= _draw_estimate(hand, replace(funded, score_mode='conservative'))
        return ConsumableDecision("use" if urgent or cash >= 10 or state.consumables_held >= state.consumable_slots else "hold",
            utility=cash + max(0, improved - score) + 200 * urgent, expected_score=improved,
            reason=f"double money for ${cash}, capped at $20; cash/growth scoring {score:.0f}->{improved:.0f}")
    if key == "temperance":
        cash = min(50, sum(j.sell_value for j in state.jokers))
        return ConsumableDecision("use" if cash >= 10 else "hold", utility=cash,
            expected_score=score, reason=f"observed Joker sell values pay ${cash}")
    if key in ("the_emperor", "the_high_priestess"):
        return ConsumableDecision("use" if space >= 2 else "hold", utility=6,
            expected_score=score, reason="leave two slots for generated consumables", deterministic=False)
    if key in ("judgement", "the_soul"):
        free = state.total_joker_slots - len(state.jokers)
        return ConsumableDecision("use" if free > 0 else "hold", utility=12 if free > 0 else 0,
            expected_score=score, reason="requires a free Joker slot", deterministic=False)
    if key in ("ankh", "hex"):
        # Destruction of the rest of a developed engine is too expensive.
        if len(state.jokers) != 1 or state.jokers[0].get("eternal"):
            return replace(best, reason="random selection would destroy or fail to change the current engine")
        if key == 'hex' and state.jokers[0].edition:
            return replace(best, reason='Hex requires an unedited Joker')
        if key == 'ankh' and state.total_joker_slots < 2 and state.jokers[0].edition != 'negative':
            return replace(best, reason='Ankh needs room for the second Joker')
        return ConsumableDecision("use", utility=10, expected_score=score,
            reason="only one Joker, so target is known", deterministic=False)
    if key in ("ouija", "ectoplasm"):
        return replace(best, reason="hand-size loss is permanent; needs a deck-specific plan")
    if key in ("immolate", "familiar", "grim", "incantation", "sigil", "aura", "wraith", "the_wheel_of_fortune"):
        return replace(best, reason="random deck or Joker change requires outcome observation", deterministic=False)
    if key == "the_fool":
        return replace(best, reason="requires observed previous Tarot/Planet identity")
    if not hand:
        return best
    if key in ENHANCEMENT_TAROTS:
        maximum = ENHANCEMENT_TAROTS[key][1]
    elif key in SUIT_TAROTS:
        maximum = 3
    elif key in ("strength", "death", "the_hanged_man"):
        maximum = 2
    elif key in SEAL_SPECTRALS or key == "cryptid":
        maximum = 1
    else:
        return best
    choices = [(indices, ()) for indices in _choices(hand, maximum, state)]
    if key == 'death':
        # Also consider copying a valuable card on the left, after a verified
        # physical rearrangement. Plain Death always copies left -> right.
        choices += [(indices, tuple(reversed(indices))) for indices, _ in choices if len(indices) == 2]
    for indices, play_order in choices:
        if key == "death" and len(indices) != 2:
            continue
        if any(hand[i].face_down for i in indices):
            continue
        if (key in ENHANCEMENT_TAROTS or key == 'strength') and any(
                hand[i].is_stone and not 2 <= hand[i].rank <= 14 for i in indices):
            continue
        working = list(hand)
        targets = indices
        if play_order:
            working = [hand[i] for i in play_order] + [c for i, c in enumerate(hand) if i not in indices]
            targets = (0, 1)
        modified = transformed_hand(working, consumable, targets)
        if not modified:
            continue
        projected_state = transformed_state(used_engine, working, modified)
        projected = _draw_estimate(modified, projected_state)
        lasting = sum(deck_target_value(c, projected_state) for c in modified) - sum(deck_target_value(c, state) for c in hand)
        if key == "the_hanged_man":
            average = sum(deck_target_value(c, state) for c in full_deck(state)) / len(full_deck(state))
            # Removing a low card is not automatically useful in a Hack or
            # Fibonacci engine. Compare against this deck's actual average.
            lasting += sum(2 * average - deck_target_value(hand[i], state) for i in indices)
        if key in SEAL_SPECTRALS:
            lasting += 6 * len(indices)
        if key == "cryptid":
            lasting += 20 if any(hand[i].enhancement == "steel" or hand[i].seal for i in indices) else -10
        if key == "death":
            left, right = targets
            counts = Counter(c.rank for c in modified if not c.is_stone)
            if engine_profile(state).preferred_hand in ('pair', 'two_pair', 'three_kind', 'full_house', 'four_kind', 'five_kind'):
                lasting += 10 * max(0, counts[working[right].rank] - 2)
            if _card_debuffed(working[right], state):
                lasting -= 25
        if key == 'the_chariot':
            lasting += 20 * sum(hand[i].rank == 13 for i in indices) * bool(state.has('baron'))
        if key in SUIT_TAROTS:
            preferred = engine_profile(state).preferred_hand
            if 'flush' not in preferred and not (state.has('the_tribe') or state.has('crafty_joker')):
                lasting *= .2
        if key == 'justice' and state.has('vampire'):
            lasting *= .3
        benefit = projected - score + lasting * max(.2, (9 - state.ante) * .12)
        urgent = False
        if projected >= state.score_needed > score:
            urgent = _draw_estimate(modified, replace(projected_state, score_mode='conservative')) >= state.score_needed
        if urgent:
            benefit += 200
        if benefit > best.utility + 2:
            best = ConsumableDecision("use", indices, benefit, projected,
                f"{consumable.name}: expected hand {score:.0f} -> {projected:.0f}; "
                f"deck benefit {lasting:.1f}; targets {indices}", play_order=play_order)
    return best
