"""Balatro scoring and decisions for an Airtest-controlled game.

The existing runner supplies only Card(rank, suit). Optional state captures
observed Jokers, levels, blinds and deck composition when those become
available. Public choose_play and choose_discard retain their old signatures.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from itertools import combinations, permutations, product
from math import comb, isfinite, log2
from random import Random
from functools import lru_cache, cached_property
import re
import unicodedata
from typing import Iterable, Sequence


SUITS = ("S", "H", "D", "C")
HAND_ORDER = (
    "high_card", "pair", "two_pair", "three_kind", "straight", "flush",
    "full_house", "four_kind", "straight_flush", "five_kind",
    "flush_house", "flush_five",
)
HAND_RANK = {name: index for index, name in enumerate(HAND_ORDER)}

# Base chips, base mult, chips per planet level, mult per planet level.
HAND_LEVELS = {
    "high_card": (5, 1, 10, 1),
    "pair": (10, 2, 15, 1),
    "two_pair": (20, 2, 20, 1),
    "three_kind": (30, 3, 20, 2),
    "straight": (30, 4, 30, 3),
    "flush": (35, 4, 15, 2),
    "full_house": (40, 4, 25, 2),
    "four_kind": (60, 7, 30, 3),
    "straight_flush": (100, 8, 40, 4),
    "five_kind": (120, 12, 35, 3),
    "flush_house": (140, 14, 40, 4),
    "flush_five": (160, 16, 50, 3),
}
BASE = {name: values[:2] for name, values in HAND_LEVELS.items()}
PLANET_HAND = {
    "pluto": "high_card", "mercury": "pair", "uranus": "two_pair",
    "venus": "three_kind", "saturn": "straight", "jupiter": "flush",
    "earth": "full_house", "mars": "four_kind", "neptune": "straight_flush",
    "planet_x": "five_kind", "ceres": "flush_house", "eris": "flush_five",
}
HAND_ALIASES = {
    "high card": "high_card", "pair": "pair", "two pair": "two_pair",
    "three of a kind": "three_kind", "straight": "straight",
    "flush": "flush", "full house": "full_house",
    "four of a kind": "four_kind", "straight flush": "straight_flush",
    "five of a kind": "five_kind", "flush house": "flush_house",
    "flush five": "flush_five",
}


def normalize_name(name: str) -> str:
    return " ".join(name.strip().lower().replace("_", " ").replace("-", " ").split())


def hand_name(name: str) -> str:
    key = normalize_name(name)
    return HAND_ALIASES.get(key, key.replace(" ", "_"))


@lru_cache(maxsize=1024)
def joker_name(name: str) -> str:
    value = unicodedata.normalize("NFKD", normalize_name(name))
    value = "".join(c for c in value if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]", "", value).replace(" ", "_")


@dataclass(frozen=True)
class Card:
    rank: int
    suit: str
    enhancement: str = ""
    edition: str = ""
    seal: str = ""
    bonus_chips: float = 0
    debuffed: bool = False
    face_down: bool = False
    rankless: bool = False

    @property
    def chips(self) -> float:
        if self.is_stone:
            return 50 + self.bonus_chips
        return (11 if self.rank == 14 else min(self.rank, 10)) + self.bonus_chips

    @property
    def is_face(self) -> bool:
        return not self.is_stone and self.rank in (11, 12, 13)

    @property
    def is_wild(self) -> bool:
        return self.enhancement == "wild" and not self.debuffed

    @property
    def is_stone(self) -> bool:
        return self.enhancement == "stone" or self.rankless

    @property
    def identity(self) -> tuple[int, str]:
        return self.rank, self.suit.upper()[:1]


@dataclass(frozen=True)
class Joker:
    name: str
    value: float = 0
    edition: str = ""
    debuffed: bool = False
    rarity: str = "common"
    sell_value: int = 0
    data: tuple[tuple[str, float], ...] = ()

    @property
    def key(self) -> str:
        return joker_name(self.name)

    @cached_property
    def _data_lookup(self):
        # Match the original first-match semantics even for duplicate keys.
        return dict(reversed(self.data))

    def get(self, name: str, default: float = 0) -> float:
        return self._data_lookup.get(name, default)

    @property
    def numeric_value_available(self):
        """An observed value or an explicit bound must override defaults, including zero."""
        return bool(self.get('observed_value') or self.get('counter_lower_bound'))


@dataclass(frozen=True)
class RunState:
    jokers: tuple[Joker, ...] = ()
    levels: dict[str, int] = field(default_factory=dict)
    played_types: dict[str, int] = field(default_factory=dict)
    remaining_deck: tuple[Card, ...] = ()
    # Full, remembered composition is separate from an explicitly supplied
    # remaining draw pile. Transformations persist across blind resets.
    deck_cards: tuple[Card, ...] = ()
    seen_cards: tuple[Card, ...] = ()
    hand_size: int = 8
    hands_left: int = 4
    discards_left: int = 3
    money: int = 4
    target_score: float = 0
    round_score: float = 0
    ante: int = 1
    boss: str = ""
    boss_target_hand: str = ""
    boss_seen_types: frozenset[str] = frozenset()
    boss_first_hand_done: bool = False
    skipped_blinds: int = 0
    deck_size: int = 52
    total_joker_slots: int = 5
    consumable_slots: int = 2
    score_mode: str = "expected"
    seed: int = 0
    round_played_types: dict[str, int] = field(default_factory=dict)
    draw_pile_size: int = -1
    consumables_held: int = 0
    total_hands_played: int = 0
    boss_disabled: bool = False
    score_confidence: str = "observed"
    objective: str = "score"
    deck_type: str = "Red Deck"
    forced_card_index: int = -1
    planet_cards: dict[str, int] = field(default_factory=dict)
    observatory: bool = False
    space_applied: bool = False
    pillar_cards: tuple[Card, ...] = ()
    shop_rerolls: int = 0
    base_hands_per_round: int = 4
    base_discards_per_round: int = 3
    round_resources_observed: bool = False
    round_discards_used: int = -1  # -1 means an older checkpoint did not record it.
    blind_index: int = -1
    upcoming_boss: str = ""
    upcoming_boss_target: float = 0
    upcoming_boss_ante: int = 0
    owned_vouchers: tuple[str, ...] = ()
    interest_cash_cap: int = 25
    pack_hand_from_full_deck: bool = False
    joker_order_unknown: bool = False
    # Hiker chips are invisible on the face until the tooltip is read again.
    # Keep uncertainty across shops/blinds, including after selling Hiker.
    unverified_chip_cards: tuple[tuple[int, str], ...] = ()
    # Persist these with the per-game state, including pending transactions.
    levels_initialized: bool = False
    uncertain_levels: tuple[str, ...] = ()
    hand_history_dirty: bool = False

    @cached_property
    def active_jokers(self) -> tuple[Joker, ...]:
        return tuple(j for j in self.jokers if not j.debuffed)

    @cached_property
    def _physical_counts(self):
        return Counter(j.key for j in self.active_jokers)

    def has(self, key: str) -> bool:
        return self._physical_counts.get(key, 0) > 0

    def joker_count(self, key: str) -> int:
        return self._physical_counts.get(key, 0)

    def level(self, category: str) -> int:
        return max(1, self.levels.get(category, 1))

    def played(self, category: str) -> int:
        return self.played_types.get(category, 0)

    @property
    def score_needed(self) -> float:
        return max(0, self.target_score - self.round_score)

    @cached_property
    def abilities(self) -> tuple[Joker | None, ...]:
        if self.joker_order_unknown:
            # A shuffled face-down roster does not reveal copier targets.
            return tuple(None if j.debuffed or j.key in ('blueprint', 'brainstorm')
                         or (j.key == 'seltzer' and j.get('remaining_hands', 10) <= 0)
                         else j for j in self.jokers)
        return resolve_abilities(self.jokers)

    @cached_property
    def _effect_counts(self):
        return Counter(j.key for j in self.abilities if j is not None)

    @cached_property
    def _boss_name(self):
        if self.boss_disabled or self.has('chicot'):
            return ''
        return normalize_name(self.boss).removeprefix('the ').replace(' ', '_')

    def effect_count(self, key: str) -> int:
        return self._effect_counts.get(key, 0)


NONCOPYABLE = frozenset({
    "astronomer", "chaos_the_clown", "chicot", "cloud_9", "credit_card",
    "delayed_gratification", "drunkard", "egg", "four_fingers", "gift_card",
    "golden_joker", "invisible_joker", "juggler", "merry_andy", "midas_mask",
    "mr_bones", "oops_all_6s", "pareidolia", "rocket", "satellite", "shortcut",
    "showman", "sixth_sense", "smeared_joker", "splash", "to_the_moon",
    "trading_card", "troubadour", "turtle_bean",
})


@lru_cache(maxsize=256)
def resolve_abilities(jokers: tuple[Joker, ...]) -> tuple[Joker | None, ...]:
    """Resolve copier chains; editions belong to the physical Joker."""
    def resolve(index, seen):
        if not 0 <= index < len(jokers) or index in seen:
            return None
        joker = jokers[index]
        if joker.debuffed:
            return None
        if joker.key in ("blueprint", "brainstorm"):
            target = index + 1 if joker.key == "blueprint" else 0
            result = resolve(target, seen | {index})
            if result is None or result.key in NONCOPYABLE or not result.get("copyable", 1):
                return None
            return result
        return joker
    return tuple(resolve(i, frozenset()) for i in range(len(jokers)))


def _face(card: Card, state: RunState) -> bool:
    return not card.is_stone and (card.is_face or state.has("pareidolia"))


def _probability(state: RunState, denominator: int) -> float:
    return min(1., 2. ** state.joker_count("oops_all_6s") / denominator)


@dataclass(frozen=True)
class HandPattern:
    category: str
    scoring: tuple[int, ...]
    played: tuple[Card, ...]
    flush_suits: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScoreEstimate:
    score: float
    category: str
    chips: float
    multiplier: float
    scoring: tuple[int, ...]
    cash: float = 0
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlayDecision:
    indices: tuple[int, ...]
    estimate: ScoreEstimate
    utility: float
    reason: str
    play_order: tuple[int, ...] = ()
    training: bool = False


@dataclass(frozen=True)
class DiscardDecision:
    indices: tuple[int, ...]
    expected_score: float
    utility: float
    reason: str


@dataclass(frozen=True)
class ShopItem:
    kind: str
    name: str
    price: int
    joker: Joker | None = None
    edition: str = ""
    pack_contents: tuple[str, ...] = ()


@dataclass(frozen=True)
class ShopDecision:
    action: str
    item_index: int = -1
    sell_index: int = -1
    utility: float = 0
    reason: str = ""


def straight(ranks: list[int]) -> bool:
    unique = set(ranks)
    return len(ranks) == 5 and len(unique) == 5 and (
        max(unique) - min(unique) == 4 or unique == {14, 2, 3, 4, 5}
    )


def _rank_windows(state: RunState) -> list[tuple[int, ...]]:
    windows: list[tuple[int, ...]] = []
    lengths = (4, 5) if state.has("four_fingers") else (5,)
    for length in lengths:
        for low in range(2, 16 - length):
            windows.append(tuple(range(low, low + length)))
        windows.append((14, *range(2, length + 1)))
    return windows


def _straight_matches(ranks: Iterable[int], state: RunState) -> bool:
    ranks = tuple(ranks)
    values = set(ranks)
    minimum = 4 if state.has("four_fingers") else 5
    if len(values) < minimum:
        return False
    if any(len(window) == len(values) and set(window) == values
           for window in _rank_windows(state)):
        return True
    if not state.has("shortcut"):
        return False
    ordered = sorted(values)
    if all(0 < b - a <= 2 for a, b in zip(ordered, ordered[1:])):
        return True
    if 14 in ordered:
        low_ace = sorted(1 if rank == 14 else rank for rank in ordered)
        if all(0 < b - a <= 2 for a, b in zip(low_ace, low_ace[1:])):
            return True
    return False


def _effective_suits(card: Card, state: RunState) -> frozenset[str]:
    if card.is_stone:
        return frozenset()
    if card.is_wild:
        return frozenset(SUITS)
    suit = card.suit.upper()[:1]
    if state.has("smeared_joker"):
        if suit in ("S", "C"):
            return frozenset(("S", "C"))
        if suit in ("H", "D"):
            return frozenset(("H", "D"))
    return frozenset((suit,))


def _flush_suits(cards: Sequence[Card], state: RunState) -> tuple[str, ...]:
    minimum = 4 if state.has("four_fingers") else 5
    return tuple(suit for suit in SUITS if sum(
        suit in _effective_suits(card, state) and not card.face_down
        for card in cards
    ) >= minimum)


def _straight_indices(cards: Sequence[Card], state: RunState) -> tuple[int, ...]:
    eligible = [(i, card.rank) for i, card in enumerate(cards)
                if not card.is_stone and not card.face_down]
    minimum = 4 if state.has("four_fingers") else 5
    for count in range(min(len(eligible), 5), minimum - 1, -1):
        for selection in combinations(eligible, count):
            if _straight_matches((rank for _, rank in selection), state):
                return tuple(i for i, _ in selection)
    return ()


def _flush_indices(cards: Sequence[Card], state: RunState) -> tuple[int, ...]:
    available = _flush_suits(cards, state)
    for suit in ("S", "H", "C", "D"):
        if suit not in available:
            continue
        indices = tuple(i for i, card in enumerate(cards)
                        if suit in _effective_suits(card, state) and not card.face_down)
        return indices
    return ()


def _straight_flush_indices(cards: Sequence[Card], state: RunState) -> tuple[int, ...]:
    # Game combines the straight and flush sets, including Four Fingers.
    run, flush = _straight_indices(cards, state), _flush_indices(cards, state)
    return tuple(sorted(set(run) | set(flush))) if run and flush else ()


_CLASSIFICATION_KEYS = ('four_fingers', 'shortcut', 'smeared_joker', 'splash')


@lru_cache(maxsize=16)
def _classification_state(flags):
    return RunState(jokers=tuple(Joker(key) for key, enabled in zip(_CLASSIFICATION_KEYS,flags) if enabled))


@lru_cache(maxsize=8192)
def _classify_cached(played, flags):
    return _classify_uncached(played, _classification_state(flags))


def classify(cards: Sequence[Card], state: RunState | None = None) -> HandPattern:
    """Reuse exact ordered cards with the four actual classification rules.

    Levels, money and history affect scoring later, not hand classification.
    Every card modifier and its order remain part of the immutable cache key.
    """
    if not 1 <= len(cards) <= 5:
        raise ValueError('A played hand needs 1 to 5 cards')
    state = state or RunState()
    return _classify_cached(tuple(cards), tuple(state.has(k) for k in _CLASSIFICATION_KEYS))


def _classify_uncached(cards: Sequence[Card], state: RunState | None = None) -> HandPattern:
    """Recognize twelve hands and find the cards that actually score."""
    if not 1 <= len(cards) <= 5:
        raise ValueError("A played hand needs 1 to 5 cards")
    state = state or RunState()
    played = tuple(cards)
    by_rank: dict[int, list[int]] = defaultdict(list)
    for index, card in enumerate(played):
        if not card.is_stone and not card.face_down:
            by_rank[card.rank].append(index)
    groups = sorted(by_rank.values(), key=lambda g: (len(g), played[g[0]].rank), reverse=True)
    frequencies = sorted((len(group) for group in groups), reverse=True)
    flush = _flush_indices(played, state)
    sf = _straight_flush_indices(played, state)
    run = _straight_indices(played, state)
    all_ranked = tuple(i for group in groups for i in group)

    if frequencies == [5] and flush:
        category, scoring = "flush_five", all_ranked
    elif frequencies == [3, 2] and flush:
        category, scoring = "flush_house", all_ranked
    elif frequencies == [5]:
        category, scoring = "five_kind", all_ranked
    elif sf:
        category, scoring = "straight_flush", sf
    elif frequencies and frequencies[0] == 4:
        category, scoring = "four_kind", tuple(groups[0])
    elif frequencies == [3, 2]:
        category, scoring = "full_house", all_ranked
    elif flush:
        category, scoring = "flush", flush
    elif run:
        category, scoring = "straight", run
    elif frequencies and frequencies[0] == 3:
        category, scoring = "three_kind", tuple(groups[0])
    elif frequencies.count(2) >= 2:
        category, scoring = "two_pair", tuple(groups[0] + groups[1])
    elif frequencies and frequencies[0] == 2:
        category, scoring = "pair", tuple(groups[0])
    else:
        category = "high_card"
        scoring = (max(range(len(played)), key=lambda i: played[i].chips),)

    scoring = tuple(sorted(set(scoring) | {
        i for i, card in enumerate(played) if card.is_stone
    }))
    if state.has("splash"):
        scoring = tuple(range(len(played)))
    return HandPattern(category, scoring, played, _flush_suits(played, state))


def _boss(state: RunState) -> str:
    return state._boss_name


def _card_debuffed(card: Card, state: RunState) -> bool:
    if card.debuffed:
        return True
    boss = _boss(state)
    suit_bosses = {"goad": "S", "club": "C", "head": "H", "window": "D"}
    if suit_bosses.get(boss) in _effective_suits(card, state):
        return True
    if boss == "plant" and _face(card, state):
        return True
    if boss == "pillar" and card in state.pillar_cards:
        return True
    return False


def locked_mouth_hand(state: RunState) -> str:
    """The Mouth locks the first hand, including across checkpoint resumes."""
    if _boss(state) != "mouth":
        return ""
    if state.boss_target_hand:
        return hand_name(state.boss_target_hand)
    # Migrate old checkpoints without the explicit lock. Dict insertion order
    # preserves the first played type; rejected cycling must not widen it.
    if state.round_played_types:
        return next(iter(state.round_played_types))
    if len(state.boss_seen_types) == 1:
        return next(iter(state.boss_seen_types))
    return ""


def _legal_play(pattern: HandPattern, state: RunState) -> bool:
    boss = _boss(state)
    if boss == "eye" and pattern.category in state.boss_seen_types:
        return False
    mouth_hand = locked_mouth_hand(state)
    if mouth_hand and pattern.category != mouth_hand:
        return False
    if boss == "psychic" and len(pattern.played) != 5:
        return False
    return True


def _card_retriggers(card: Card, state: RunState, index: int,
                     pattern: HandPattern) -> int:
    count = 1 + int(card.seal.lower() == "red")
    if pattern.scoring and index == pattern.scoring[0]:
        count += 2 * state.effect_count("hanging_chad")
    if _face(card, state):
        count += state.effect_count("sock_and_buskin")
    if not card.is_stone and card.rank in (2, 3, 4, 5):
        count += state.effect_count("hack")
    if state.hands_left == 1:
        count += state.effect_count("dusk")
    count += state.effect_count("seltzer")
    return count


def _scored_card_effect(card: Card, state: RunState) -> tuple[float, float, float, float]:
    if _card_debuffed(card, state):
        return 0, 0, 1, 0
    chips, mult, xmult, cash = card.chips, 0.0, 1.0, 0.0
    if card.enhancement == "bonus":
        chips += 30
    elif card.enhancement == "mult":
        mult += 4
    elif card.enhancement == "glass":
        xmult *= 2
    elif card.enhancement == "lucky" and state.score_mode == "expected":
        mult += 20 * _probability(state, 5)
        cash += 20 * _probability(state, 15)
    if card.edition == "foil":
        chips += 50
    elif card.edition == "holographic":
        mult += 10
    elif card.edition == "polychrome":
        xmult *= 1.5
    if card.seal.lower() == "gold":
        cash += 3
    return chips, mult, xmult, cash


def _joker_card_effect(joker: Joker, card: Card, state: RunState) -> tuple[float, float, float, float]:
    key = joker.key
    rank = card.rank
    if card.is_stone:
        return 0, 0, 1, 0
    suit_jokers = {
        "greedy_joker": "D", "lusty_joker": "H",
        "wrathful_joker": "S", "gluttonous_joker": "C",
    }
    if key in suit_jokers and suit_jokers[key] in _effective_suits(card, state):
        return 0, 3, 1, 0
    if key == "scary_face" and _face(card, state):
        return 30, 0, 1, 0
    if key == "smiley_face" and _face(card, state):
        return 0, 5, 1, 0
    if key == "even_steven" and rank <= 10 and rank % 2 == 0:
        return 0, 4, 1, 0
    if key == "odd_todd" and (rank == 14 or rank <= 9 and rank % 2 == 1):
        return 31, 0, 1, 0
    if key == "fibonacci" and rank in (2, 3, 5, 8, 14):
        return 0, 8, 1, 0
    if key == "scholar" and rank == 14:
        return 20, 4, 1, 0
    if key == "walkie_talkie" and rank in (4, 10):
        return 10, 4, 1, 0
    if key == "triboulet" and rank in (12, 13):
        return 0, 0, 2, 0
    if key == "ancient_joker":
        suit = SUITS[int(joker.get("suit", 0)) % 4]
        if suit in _effective_suits(card, state):
            return 0, 0, 1.5, 0
    if key in ("the_idol", "idol"):
        suit = SUITS[int(joker.get("suit", 0)) % 4]
        if rank == int(joker.get("rank", 0)) and suit in _effective_suits(card, state):
            return 0, 0, 2, 0
    if key == "business_card" and _face(card, state):
        return 0, 0, 1, 2 * _probability(state, 2)
    if key == "bloodstone" and "H" in _effective_suits(card, state):
        return 0, 0, 1 + .5 * _probability(state, 2) if state.score_mode == "expected" else 1, 0
    if key == "arrowhead" and "S" in _effective_suits(card, state):
        return 50, 0, 1, 0
    if key == "onyx_agate" and "C" in _effective_suits(card, state):
        return 0, 7, 1, 0
    if key == "rough_gem" and "D" in _effective_suits(card, state):
        return 0, 0, 1, 1
    if key == "hiker":
        return 0, 0, 1, 0  # Permanent chips apply from the next trigger.
    return 0, 0, 1, 0


def _held_effect(card: Card, state: RunState) -> tuple[float, float, float]:
    if _card_debuffed(card, state):
        return 0, 1, 0
    mult, xmult, cash = 0.0, 1.0, 0.0
    if card.enhancement == "steel":
        xmult *= 1.5
    if card.enhancement == "gold":
        cash += 3 / max(1, state.hands_left)
    return mult, xmult, cash


def _contains_pair(pattern: HandPattern) -> bool:
    return any(count >= 2 for count in Counter(
        card.rank for card in pattern.played if not card.is_stone and not card.face_down
    ).values())


def _contains_three(pattern: HandPattern) -> bool:
    return any(count >= 3 for count in Counter(
        card.rank for card in pattern.played if not card.is_stone and not card.face_down
    ).values())


def _contains_four(pattern: HandPattern) -> bool:
    return any(count >= 4 for count in Counter(
        card.rank for card in pattern.played if not card.is_stone and not card.face_down
    ).values())


def _contains_two_pair(pattern: HandPattern) -> bool:
    return sum(count >= 2 for count in Counter(
        card.rank for card in pattern.played if not card.is_stone and not card.face_down
    ).values()) >= 2


def _all_black_held(held: Sequence[Card], state: RunState) -> bool:
    return bool(held) and all(
        bool(_effective_suits(card, state) & {"S", "C"})
        for card in held
    )


def _rainbow_scored(pattern: HandPattern, state: RunState) -> bool:
    scored = [pattern.played[i] for i in pattern.scoring]
    # One card can fill at most one slot. Smeared cards also consume one slot.
    def assign(suits, used):
        if not suits:
            return True
        return any(i not in used and suits[0] in _effective_suits(card, state)
                   and assign(suits[1:], used | {i}) for i, card in enumerate(scored))
    return assign(SUITS, set())


def _joker_global_effect(joker: Joker, pattern: HandPattern,
                         held: Sequence[Card], state: RunState
                         ) -> tuple[float, float, float, float]:
    """Effects applied after scored cards, in the visible Joker order."""
    key = joker.key
    pair = _contains_pair(pattern)
    three = _contains_three(pattern)
    four = _contains_four(pattern)
    two_pair = _contains_two_pair(pattern)
    category = pattern.category
    if key == "joker":
        return 0, 4, 1, 0
    if key == "jolly_joker" and pair:
        return 0, 8, 1, 0
    if key == "zany_joker" and three:
        return 0, 12, 1, 0
    if key == "mad_joker" and two_pair:
        return 0, 10, 1, 0
    if key == "crazy_joker" and "straight" in category:
        return 0, 12, 1, 0
    if key == "droll_joker" and "flush" in category:
        return 0, 10, 1, 0
    if key == "sly_joker" and pair:
        return 50, 0, 1, 0
    if key == "wily_joker" and three:
        return 100, 0, 1, 0
    if key == "clever_joker" and two_pair:
        return 80, 0, 1, 0
    if key == "devious_joker" and "straight" in category:
        return 100, 0, 1, 0
    if key == "crafty_joker" and "flush" in category:
        return 80, 0, 1, 0
    if key == "half_joker" and len(pattern.played) <= 3:
        return 0, 20, 1, 0
    if key == "gros_michel":
        return 0, 15, 1, 0
    if key == "cavendish":
        return 0, 0, 3, 0
    if key == "stuntman":
        return 250, 0, 1, 0
    if key == "blue_joker":
        return 2 * max(0, state.draw_pile_size if state.draw_pile_size >= 0 else state.deck_size - len(state.seen_cards)), 0, 1, 0
    if key == "abstract_joker":
        return 0, 3 * len(state.jokers), 1, 0
    if key == "supernova":
        return 0, state.played(category) + 1, 1, 0
    if key == "ride_the_bus":
        return 0, 0 if any(_face(pattern.played[i], state) for i in pattern.scoring) else joker.value + 1, 1, 0
    if key == "green_joker":
        return 0, joker.value + 1, 1, 0
    if key == "runner":
        return joker.value + 15 * bool(_straight_indices(pattern.played, state)), 0, 1, 0
    if key == "square_joker":
        return joker.value + 4 * (len(pattern.played) == 4), 0, 1, 0
    if key == "spare_trousers":
        return 0, joker.value + 2 * two_pair, 1, 0
    if key == "hologram":
        return 0, 0, max(1, joker.value), 0
    if key == "constellation":
        return 0, 0, max(1, joker.value), 0
    if key == "vampire":
        return 0, 0, max(1, joker.value), 0
    if key == "steel_joker":
        if state.deck_cards and len(state.deck_cards) == state.deck_size:
            return 0, 0, 1 + .2 * sum(c.enhancement == 'steel' for c in state.deck_cards), 0
        if joker.get("observed_value"):
            return 0, 0, max(1, joker.value), 0
        steel_count = sum(card.enhancement == "steel" for card in state.remaining_deck)
        return 0, 0, 1 + 0.2 * max(steel_count, joker.value), 0
    if key == "bull":
        return 2 * state.money, 0, 1, 0
    if key == "bootstraps":
        return 0, 2 * (state.money // 5), 1, 0
    if key == "banner":
        return 30 * state.discards_left, 0, 1, 0
    if key == "mystic_summit" and state.discards_left == 0:
        return 0, 15, 1, 0
    if key == "misprint":
        return 0, 11.5 if state.score_mode == "expected" else 0, 1, 0
    if key == "card_sharp" and state.round_played_types.get(category, 0) > 0:
        return 0, 0, 3, 0
    if key == "the_duo" and pair:
        return 0, 0, 2, 0
    if key == "the_trio" and three:
        return 0, 0, 3, 0
    if key == "the_family" and four:
        return 0, 0, 4, 0
    if key == "the_order" and "straight" in category:
        return 0, 0, 3, 0
    if key == "the_tribe" and "flush" in category:
        return 0, 0, 2, 0
    if key == "blackboard" and _all_black_held(held, state):
        return 0, 0, 3, 0
    if key == "acrobat" and state.hands_left == 1:
        return 0, 0, 3, 0
    if key == "throwback":
        return 0, 0, 1 + 0.25 * state.skipped_blinds, 0
    if key == "flower_pot" and _rainbow_scored(pattern, state):
        return 0, 0, 3, 0
    if key == "seeing_double":
        scored = [pattern.played[i] for i in pattern.scoring if not _card_debuffed(pattern.played[i], state)]
        if any(i != j and "C" in _effective_suits(a, state) and
               bool(_effective_suits(b, state) - {"C"})
               for i, a in enumerate(scored) for j, b in enumerate(scored)):
            return 0, 0, 2, 0
    if key == "loyalty_card" and joker.get("remaining", -1) == 0:
        return 0, 0, 4, 0
    if key == "ramen":
        return 0, 0, max(1, joker.value or 2), 0
    if key == "popcorn":
        return 0, max(0, joker.value if joker.numeric_value_available else joker.value or 20), 1, 0
    if key == "ice_cream":
        return max(0, joker.value if joker.numeric_value_available else joker.value or 100), 0, 1, 0
    if key == "flash_card":
        return 0, joker.value, 1, 0
    if key == "red_card":
        return 0, joker.value, 1, 0
    if key == "fortune_teller":
        return 0, joker.value, 1, 0
    if key == "swashbuckler":
        sell = sum(j.sell_value for j in state.jokers if j is not joker)
        return 0, joker.value if joker.get("observed_value") else sell, 1, 0
    if key == "ceremonial_dagger":
        return 0, joker.value, 1, 0
    if key in ("campfire", "glass_joker", "lucky_cat", "hit_the_road", "madness", "canio", "yorick"):
        return 0, 0, max(1, joker.value), 0
    if key == "obelisk":
        most = max(state.played_types.values(), default=0)
        return 0, 0, 1 if state.played(category) >= most else max(1, joker.value) + .2, 0
    if key == "stone_joker":
        if state.deck_cards and len(state.deck_cards) == state.deck_size:
            return 25 * sum(c.is_stone for c in state.deck_cards), 0, 1, 0
        if joker.get("observed_value"):
            return joker.value, 0, 1, 0
        return 25 * max(joker.value, sum(c.is_stone for c in state.remaining_deck)), 0, 1, 0
    if key in ("wee_joker", "castle"):
        return joker.value, 0, 1, 0
    if key == "erosion":
        return 0, 4 * max(0, joker.get("initial_deck_size", 52) - state.deck_size), 1, 0
    if key == "drivers_license":
        enhanced = (sum(bool(c.enhancement) for c in state.deck_cards)
                    if state.deck_cards and len(state.deck_cards) == state.deck_size else
                    joker.get("enhanced_count", sum(bool(c.enhancement) for c in state.remaining_deck)))
        if enhanced >= 16:
            return 0, 0, 3, 0
    if key == "joker_stencil":
        empty = state.total_joker_slots - len(state.jokers) + state.joker_count("joker_stencil")
        return 0, 0, max(1, empty), 0
    if key == "ticket" or key == "golden_ticket":
        return 0, 0, 1, 0
    if key == "to_do_list" and int(joker.get("hand", -1)) == HAND_RANK[category]:
        return 0, 0, 1, 4
    if key == "vagabond" and state.money <= 4 and state.consumables_held < state.consumable_slots:
        return 0, 0, 1, 3
    return 0, 0, 1, 0


def _joker_edition(joker: Joker) -> tuple[float, float, float]:
    if joker.edition == "foil":
        return 50, 0, 1
    if joker.edition == "holographic":
        return 0, 10, 1
    if joker.edition == "polychrome":
        return 0, 0, 1.5
    return 0, 0, 1


def _joker_card_is_first_face(card: Card, index: int,
                              pattern: HandPattern, state: RunState) -> bool:
    return _face(card, state) and index == next(
        (i for i in pattern.scoring if _face(pattern.played[i], state)), -1
    )


def _before_scoring(pattern: HandPattern, state: RunState):
    """Scalers and enhancement converters run before card triggers."""
    if not any(state.has(key) for key in ('midas_mask', 'vampire', 'wee_joker')):
        return pattern, state
    cards = list(pattern.played)
    jokers = []
    physical = state.jokers
    if state.joker_order_unknown:
        # Vampire before Midas is the smaller possible Vampire growth.
        physical = tuple(sorted(physical, key=lambda j: j.key == 'midas_mask'))
    for joker in physical:
        if joker.debuffed:
            jokers.append(joker)
            continue
        if joker.key == "midas_mask":
            for index in pattern.scoring:
                if _face(cards[index], state) and not _card_debuffed(cards[index], state):
                    cards[index] = replace(cards[index], enhancement="gold")
        if joker.key == "vampire":
            count = 0
            for index in pattern.scoring:
                if cards[index].enhancement and not _card_debuffed(cards[index], state):
                    cards[index] = replace(cards[index], enhancement="", rankless=False)
                    count += 1
            joker = replace(joker, value=max(1, joker.value) + .1 * count)
        if joker.key == "wee_joker":
            count = sum(_card_retriggers(cards[i], state, i, pattern)
                        for i in pattern.scoring if cards[i].rank == 2 and not cards[i].is_stone
                        and not _card_debuffed(cards[i], state))
            joker = replace(joker, value=joker.value + 8 * count)
        jokers.append(joker)
    return replace(pattern, played=tuple(cards)), replace(state, jokers=tuple(jokers))


def _unknown_order_mult(mult, effects):
    """Lower bound for an unknown order of nonnegative +Mult and XMult."""
    additive, factor = 0., 1.
    for added, multiplier in effects:
        additive += max(0., added)
        factor *= max(0., multiplier)
    return mult * factor + additive * min(1., factor)


def score_hand(cards: Sequence[Card], state: RunState | None = None,
               held: Sequence[Card] = ()) -> ScoreEstimate:
    """Score a play, including Hook's discard before held-card triggers."""
    state = state or RunState()
    held_effects = (any(c.enhancement in ('steel', 'gold') or c.seal.lower() == 'blue' for c in held)
                   or any(state.has(key) for key in ('baron', 'shoot_the_moon', 'raised_fist',
                                                     'reserved_parking', 'blackboard')))
    if _boss(state) != 'hook' or not held or not held_effects:
        return _score_hand_core(cards, state, held)
    outcomes = []
    for dropped in combinations(range(len(held)), min(2, len(held))):
        remaining = tuple(c for i, c in enumerate(held) if i not in dropped)
        outcomes.append(_score_hand_core(cards, state, remaining))
    if state.score_mode != 'expected':
        result = min(outcomes, key=lambda e: (e.score, e.cash))
    else:
        result = replace(outcomes[0],
            score=sum(e.score for e in outcomes) / len(outcomes),
            chips=sum(e.chips for e in outcomes) / len(outcomes),
            multiplier=sum(e.multiplier for e in outcomes) / len(outcomes),
            cash=sum(e.cash for e in outcomes) / len(outcomes))
    return replace(result, notes=result.notes +
                   ('Hook removes held cards before scoring; expectation or worst-case bound applied',))


def _score_hand_core(cards: Sequence[Card], state: RunState | None = None,
               held: Sequence[Card] = ()) -> ScoreEstimate:
    """Estimate score in game trigger order, with expected random effects."""
    state = state or RunState()
    pattern = classify(cards, state)
    space = state.effect_count("space_joker")
    if space and state.score_mode == "expected" and not state.space_applied:
        probability = _probability(state, 4)
        outcomes = []
        for gained in range(space + 1):
            weight = comb(space, gained) * probability ** gained * (1 - probability) ** (space - gained)
            start_level = state.level(pattern.category)
            if _boss(state) == 'arm':
                start_level = max(1, start_level - 1)
            levels = {**state.levels, pattern.category: start_level + gained}
            result = _score_hand_core(cards, replace(state, space_applied=True, levels=levels), held)
            outcomes.append((weight, result))
        representative = outcomes[0][1]
        return replace(representative,
            score=sum(p * e.score for p, e in outcomes),
            chips=sum(p * e.chips for p, e in outcomes),
            multiplier=sum(p * e.multiplier for p, e in outcomes),
            cash=sum(p * e.cash for p, e in outcomes),
            notes=representative.notes + ("Space Joker uses a distribution of level-up outcomes",))
    if not _legal_play(pattern, state):
        return ScoreEstimate(0, pattern.category, 0, 0, pattern.scoring,
                             notes=("boss rejects this play",))
    pattern, state = _before_scoring(pattern, state)
    base_chips, base_mult, level_chips, level_mult = HAND_LEVELS[pattern.category]
    level = state.level(pattern.category) - 1
    if _boss(state) == "arm" and not state.space_applied:
        level = max(0, level - 1)
    chips = float(base_chips + level * level_chips)
    mult = float(base_mult + level * level_mult)
    cash = 0.0
    notes: list[str] = []
    if _boss(state) == "flint":
        chips = max(1, int(chips * .5 + .5))
        mult = max(1, int(mult * .5 + .5))
        notes.append("Flint halves base chips and mult")
    if _boss(state) == "ox" and pattern.category == hand_name(state.boss_target_hand):
        cash -= state.money
    if _boss(state) == "tooth":
        cash -= len(cards)

    hiker = state.effect_count('hiker')
    for index in pattern.scoring:
        card = pattern.played[index]
        for repetition in range(_card_retriggers(card, state, index, pattern)):
            upgraded_card = (replace(card, bonus_chips=card.bonus_chips + 5 * repetition * hiker)
                             if repetition and hiker else card)
            added_chips, added_mult, factor, income = _scored_card_effect(upgraded_card, state)
            chips += added_chips
            mult += added_mult
            mult *= factor
            cash += income
            if _card_debuffed(card, state):
                continue
            unknown_effects = []
            for joker in state.abilities:
                if joker is None:
                    continue
                jc, jm, jx, money = _joker_card_effect(joker, card, state)
                chips += jc
                if state.joker_order_unknown:
                    unknown_effects.append((jm, jx))
                else:
                    mult += jm
                    mult *= jx
                cash += money
                if joker.key == "photograph" and _joker_card_is_first_face(
                    card, index, pattern, state
                ):
                    if state.joker_order_unknown:
                        unknown_effects.append((0, 2))
                    else:
                        mult *= 2
                if joker.key == "golden_ticket" and card.enhancement == "gold":
                    cash += 4
            if state.joker_order_unknown:
                mult = _unknown_order_mult(mult, unknown_effects)

    ranked_held = [(i, card) for i, card in enumerate(held) if not card.is_stone]
    lowest = min(ranked_held, key=lambda entry: (entry[1].rank, -entry[0]))[0] if ranked_held else -1
    for index, card in enumerate(held):
        if _card_debuffed(card, state):
            continue
        repeat = 1 + int(card.seal.lower() == "red") + state.effect_count("mime")
        for _ in range(repeat):
            added_mult, factor, income = _held_effect(card, state)
            mult += added_mult
            mult *= factor
            cash += income
            unknown_effects = []
            for joker in state.abilities:
                if joker is None:
                    continue
                if joker.key == "baron" and not card.is_stone and card.rank == 13:
                    if state.joker_order_unknown:
                        unknown_effects.append((0, 1.5))
                    else:
                        mult *= 1.5
                if joker.key == "shoot_the_moon" and not card.is_stone and card.rank == 12:
                    if state.joker_order_unknown:
                        unknown_effects.append((13, 1))
                    else:
                        mult += 13
                if joker.key == "raised_fist" and index == lowest:
                    added = 2 * (11 if card.rank == 14 else min(10, card.rank))
                    if state.joker_order_unknown:
                        unknown_effects.append((added, 1))
                    else:
                        mult += added
                if joker.key == "reserved_parking" and _face(card, state):
                    cash += _probability(state, 2)
            if state.joker_order_unknown:
                mult = _unknown_order_mult(mult, unknown_effects)

    unknown_effects = []
    for physical, ability in zip(state.jokers, state.abilities):
        if physical.debuffed:
            continue
        ec, em, ex = _joker_edition(physical)
        # Foil/holographic precede ability; polychrome follows it.
        chips += ec
        if state.joker_order_unknown:
            unknown_effects.append((em, 1))
        else:
            mult += em
        if ability is not None:
            jc, jm, factor, income = _joker_global_effect(ability, pattern, held, state)
            chips += jc
            if state.joker_order_unknown:
                unknown_effects.append((jm, factor))
            else:
                mult += jm
                mult *= factor
            cash += income
        final_factor = 1.5 ** (state.effect_count("baseball_card") * (physical.rarity == "uncommon")) * ex
        if state.joker_order_unknown:
            unknown_effects.append((0, final_factor))
        else:
            mult *= final_factor

    if state.joker_order_unknown:
        mult = _unknown_order_mult(mult, unknown_effects)
        notes.append('face-down Joker order unknown; conservative order bound, copier targets excluded')

    if state.observatory:
        mult *= 1.5 ** state.planet_cards.get(pattern.category, 0)
    if normalize_name(state.deck_type) == "plasma deck":
        chips = mult = (chips + mult) / 2

    result = max(0.0, chips * mult)
    if not isfinite(result):
        result = 1e18
    if any(card.face_down for card in cards):
        notes.append("face-down ranks are uncertain")
    if state.score_mode == "expected" and (state.has("bloodstone") or any(c.enhancement == "lucky" for c in cards)):
        notes.append("random effects use expectation; not a guaranteed score")
    # A held Blue Seal creates a planet only at end of the round, with space.
    if result >= state.score_needed > 0:
        blue = sum(c.seal.lower() == "blue" and not _card_debuffed(c, state) for c in held)
        cash += 3 * min(blue, max(0, state.consumable_slots - state.consumables_held))
    return ScoreEstimate(result, pattern.category, chips, mult,
                         pattern.scoring, cash, tuple(notes))


def evaluate(cards: list[Card]) -> tuple[float, str]:
    """Compatibility API: score without extra observed game state."""
    result = score_hand(cards)
    return result.score, result.category


def _card_future_value(card: Card, state: RunState) -> float:
    """Value of keeping a card for later plays rather than cycling it."""
    if card.face_down:
        return 0
    rank = 0 if card.is_stone else card.rank
    value = card.chips * 0.45
    if card.enhancement == "steel":
        value += 24
    if card.enhancement == "gold":
        value += 18
    if card.enhancement == "glass":
        value += 7
    if card.enhancement == "mult":
        value += 8
    if card.enhancement == "bonus":
        value += 8
    if card.edition == "polychrome":
        value += 20
    if card.edition == "holographic":
        value += 12
    if card.edition == "foil":
        value += 8
    if card.seal.lower() == "red":
        value += 14
    if rank == 13 and state.has("baron"):
        value += 50
    if rank == 12 and state.has("shoot_the_moon"):
        value += 25
    if rank == 14 and state.has('scholar'):
        value += 12
    if rank in (2, 3, 5, 8, 14) and state.has('fibonacci'):
        value += 10
    if 2 <= rank <= 5 and state.has('hack'):
        value += 8 + 8 * bool(card.enhancement or card.edition or card.seal)
    if card.is_face and (state.has('photograph') or state.has('triboulet')):
        value += 12 + 12 * bool(state.has('sock_and_buskin') or state.has('hanging_chad'))
    if rank in (4, 10) and state.has('walkie_talkie'):
        value += 10
    if 2 <= rank <= 10 and rank % 2 == 0 and state.has('even_steven'):
        value += 8
    if (rank == 14 or (2 <= rank <= 10 and rank % 2)) and state.has('odd_todd'):
        value += 8
    if card.seal.lower() == 'blue':
        value += 16
    if card.seal.lower() == 'purple':
        value += 12
    if _card_debuffed(card, state):
        value *= 0.25
    return value


def _preserved_pattern_value(cards: Sequence[Card], state: RunState) -> float:
    """Estimate what a retained partial hand can make next."""
    if not cards:
        return 0
    ranks = Counter(card.rank for card in cards if not card.is_stone and not card.face_down)
    suits = Counter(card.suit for card in cards if not card.is_stone and not card.face_down)
    value = 0.0
    if ranks:
        top = max(ranks.values())
        if top >= 2:
            value += 16 * (top - 1)
        if sum(count >= 2 for count in ranks.values()) >= 2:
            value += 18
    if suits:
        top_suit = max(suits.values())
        if top_suit >= 3:
            value += 7 * (top_suit - 2)
    for window in _rank_windows(state):
        overlap = len(set(window) & set(ranks))
        if overlap >= 3:
            value = max(value, 12 * (overlap - 2))
    return value


def _play_floor(indices, estimate, hand, state):
    """Conservative model score; hidden cards cannot establish a sure clear."""
    if any(hand[i].face_down for i in indices):
        return 0.
    if state.score_mode == 'expected' and (
            _boss(state) == 'hook' or state.has('misprint') or state.has('bloodstone') or state.has('space_joker')
            or any(hand[i].enhancement == 'lucky' for i in indices)):
        return score_hand([hand[i] for i in indices], replace(state, score_mode='conservative'),
                          [c for i, c in enumerate(hand) if i not in indices]).score
    return estimate.score


def _play_utility(indices: tuple[int, ...], estimate: ScoreEstimate,
                  hand: Sequence[Card], state: RunState, *, floor_score=None) -> float:
    if estimate.score <= 0:
        return float("-inf")
    needed = state.score_needed
    score = estimate.score
    remaining = [card for i, card in enumerate(hand) if i not in indices]
    guaranteed = _play_floor(indices, estimate, hand, state) if floor_score is None else floor_score
    # A one-hand win is preferred to farming extra points.
    if needed and guaranteed >= needed:
        if state.objective == "survival":
            score = needed + .15 * min(score - needed, needed)
        # User prefers higher scores; preserve overshoot in score mode.
        score += max(1000, needed * .1) + 100 * state.hands_left
    deck_left = max(0, state.draw_pile_size if state.draw_pile_size >= 0 else state.deck_size - len(state.seen_cards))
    can_draw = min(len(indices), deck_left)
    # Cycling breaks close ties; it must not turn a pair into repeated singles.
    cycle = 0.35 * can_draw
    preserved = _preserved_pattern_value(remaining, state)
    preserved += sum(_card_future_value(card, state) for card in remaining) * 0.12
    if state.hands_left <= 1:
        preserved *= 0.1
        cycle = 0
    elif state.hands_left == 2:
        preserved *= 0.5
    economy = estimate.cash * (1 if needed > score else 2)
    if state.objective == 'survival' and 0 < needed <= guaranteed:
        # Surplus after a verified clear does not help an Ante-8 goal.
        score = needed + max(1000, needed * .1) + 100 * state.hands_left
        economy = estimate.cash * max(2., needed * .015)
    # Glass destruction is one chance per played card, not per retrigger.
    glass_loss = sum(_card_future_value(hand[i], state) * _probability(state, 4)
                     for i in indices if hand[i].enhancement == "glass")
    if state.hands_left <= 1 or score < needed:
        glass_loss *= .1
    bus_reset = 0
    if state.has("ride_the_bus") and any(_face(hand[indices[i]], state) for i in estimate.scoring):
        bus_reset = sum(j.value for j in state.jokers if j.key == "ride_the_bus") * max(1, 8 - state.ante)
    hook_loss = 0
    if _boss(state) == "hook" and remaining:
        hook_loss = sum(_card_future_value(c, state) for c in remaining) * min(2, len(remaining)) / len(remaining) * .1
    from engine_development import play_development_value
    development = play_development_value(indices, estimate, hand, state, guaranteed_score=guaranteed)
    if state.objective == 'survival' and 0 < needed <= guaranteed:
        development *= min(1., needed / estimate.score)
    return score + min(4.0, preserved * 0.08) + cycle + economy + development - glass_loss - bus_reset - hook_loss


def decide_play(hand: Sequence[Card], state: RunState | None = None,
                *, optimize_order: bool = True, lookahead: bool = True) -> PlayDecision:
    """Search every playable subset, score it, and account for held cards."""
    if not hand:
        raise ValueError("Hand is empty")
    state = state or RunState()
    best: PlayDecision | None = None
    best_clears = False
    mouth_candidates = {}
    contenders = []
    sensitive = optimize_order and (
        state.effect_count("photograph") or state.effect_count("hanging_chad")
        or any(c.enhancement == "glass" or c.edition == "polychrome" for c in hand)
        or any(j is not None and j.key in ("triboulet", "bloodstone", "ancient_joker", "the_idol") for j in state.abilities))
    for size in range(1, min(5, len(hand)) + 1):
        for indices in combinations(range(len(hand)), size):
            if state.forced_card_index >= 0 and state.forced_card_index not in indices:
                continue
            selected = tuple(hand[i] for i in indices)
            held = tuple(card for i, card in enumerate(hand) if i not in indices)
            result = score_hand(selected, state, held)
            guaranteed = _play_floor(indices, result, hand, state)
            clears = state.score_needed > 0 and guaranteed >= state.score_needed
            utility = _play_utility(indices, result, hand, state, floor_score=guaranteed)
            order = indices
            if sensitive and len(indices) > 1:
                # Compare trigger-aware orders for every subset, then exhaust
                # permutations of the winner once. This keeps large hands fast.
                additive = tuple(sorted(indices, key=lambda i: (
                    hand[i].enhancement == "glass" or hand[i].edition == "polychrome",
                    -int(hand[i].enhancement == "mult" or hand[i].edition == "holographic"))))
                chad_first = tuple(sorted(indices, key=lambda i: (
                    -int(_face(hand[i], state) and state.effect_count("photograph") > 0),
                    -int(hand[i].enhancement == "glass"), -int(hand[i].seal.lower() == "red"),
                    -hand[i].chips)))
                for candidate in {additive, chad_first} - {indices}:
                    estimate = score_hand([hand[i] for i in candidate], state, held)
                    candidate_floor = _play_floor(candidate, estimate, hand, state)
                    candidate_clears = state.score_needed > 0 and candidate_floor >= state.score_needed
                    value = _play_utility(candidate, estimate, hand, state, floor_score=candidate_floor)
                    if (candidate_clears, value) > (clears, utility):
                        result, utility, order = estimate, value, candidate
                        clears = candidate_clears
            if best is None or (
                clears, utility, result.score, size,
                -sum(_card_future_value(hand[i], state) for i in indices)
            ) > (
                best_clears, best.utility, best.estimate.score, len(best.indices),
                -sum(_card_future_value(hand[i], state) for i in best.indices)
            ):
                reason = f"{result.category}: expected {result.score:.0f}; chips {result.chips:.1f} x mult {result.multiplier:.2f}; level {state.level(result.category)}"
                best = PlayDecision(indices, result, utility, reason, order)
                best_clears = clears
            if lookahead and state.score_needed > result.score > 0 and state.hands_left > 1:
                contenders.append(PlayDecision(indices, result, utility,
                    f'{result.category}: expected {result.score:.0f}', order))
            if _boss(state) == 'mouth' and not locked_mouth_hand(state) and result.score > 0:
                previous = mouth_candidates.get(result.category)
                if previous is None or utility > previous.utility:
                    mouth_candidates[result.category] = PlayDecision(indices, result, utility,
                        f"{result.category}: expected {result.score:.0f}", order)
    if best is None:
        raise ValueError("No hand satisfies the forced card constraint")
    if best.estimate.score <= 0:
        return _cycle_play(hand, state)
    if mouth_candidates and best.estimate.score < state.score_needed and state.hands_left > 1:
        best = max((_mouth_opening_plan(hand, state, candidate)
                    for candidate in mouth_candidates.values()), key=lambda decision: decision.utility)
    elif lookahead and contenders and state.score_needed > best.estimate.score and not any(c.face_down for c in hand):
        from planning import improve_play_continuation
        best = improve_play_continuation(hand, state, best, contenders)
    if lookahead and best_clears and state.objective == 'survival':
        from engine_development import safe_training_play
        best = safe_training_play(hand, state, best)
    return best


def choose_play(hand: list[Card]) -> tuple[list[int], float, str]:
    """Runner-compatible play choice."""
    decision = decide_play(hand)
    return list(decision.indices), decision.estimate.score, decision.estimate.category


def optimize_play_order(hand: Sequence[Card], state: RunState, decision: PlayDecision) -> PlayDecision:
    """Compare all orders of the chosen five cards, including held-card order.

    This runs once for the real decision, never in Monte Carlo draw samples.
    The controller must physically drag the order before using its score.
    """
    selected = decision.indices
    held = tuple(card for i, card in enumerate(hand) if i not in selected)
    best = decision if decision.play_order else replace(decision, play_order=selected)
    clears = state.score_needed > 0 and _play_floor(best.play_order, best.estimate, hand, state) >= state.score_needed
    for order in permutations(selected):
        estimate = score_hand(tuple(hand[i] for i in order), state, held)
        floor = _play_floor(order, estimate, hand, state)
        order_clears = state.score_needed > 0 and floor >= state.score_needed
        utility = _play_utility(order, estimate, hand, state, floor_score=floor)
        previous_utility = _play_utility(best.play_order, best.estimate, hand, state)
        if (order_clears, utility) > (clears, previous_utility + .01):
            best = replace(best, estimate=estimate, utility=best.utility + utility - previous_utility,
                reason=f"{best.reason}; order {order} improves trigger sequence", play_order=order)
            clears = order_clears
    return best


def optimize_joker_order(hand: Sequence[Card], state: RunState, decision: PlayDecision):
    """Arrange flat Mult before XMult and search Blueprint/Brainstorm targets."""
    original = tuple(range(len(state.jokers)))
    if len(original) < 2 or state.joker_order_unknown or _boss(state) == "amber_acorn":
        return original, decision.estimate.score
    selected_order = decision.play_order or decision.indices
    selected = [hand[i] for i in selected_order]
    held = [card for i, card in enumerate(hand) if i not in decision.indices]
    best_order, best_score = original, score_hand(selected, state, held).score
    preserve_clear = (state.score_needed > 0 and
                      _play_floor(selected_order, decision.estimate, hand, state) >= state.score_needed)
    if len(original) <= 6:
        candidates = permutations(original)
    else:
        # Negative Jokers can make the slot count exceed five. Use local swaps
        # instead of a factorial search in that case.
        candidates = [original]
        for left, right in combinations(original, 2):
            order = list(original)
            order[left], order[right] = order[right], order[left]
            candidates.append(tuple(order))
    for order in candidates:
        arranged = replace(state, jokers=tuple(state.jokers[i] for i in order))
        score = score_hand(selected, arranged, held).score
        if score > best_score * 1.005:
            if preserve_clear and score_hand(selected, replace(arranged, score_mode='conservative'), held).score < state.score_needed:
                continue
            best_order, best_score = tuple(order), score
    return best_order, best_score


def _standard_unseen(hand: Sequence[Card], state: RunState) -> list[Card]:
    """Use the known deck if present; otherwise estimate a normal deck."""
    if state.remaining_deck:
        available = list(state.remaining_deck)
        for card in hand:
            if card in available:
                available.remove(card)
        return available
    if state.deck_cards and len(state.deck_cards) == state.deck_size:
        available = list(state.deck_cards)
        for card in (*hand, *state.seen_cards):
            if card in available:
                available.remove(card)
            else:
                match = next((i for i, other in enumerate(available)
                              if other.identity == card.identity or
                              (card.is_stone and other.is_stone and card.edition == other.edition
                               and card.seal == other.seal and card.bonus_chips == other.bonus_chips)), None)
                if match is not None:
                    available.pop(match)
        return available
    visible = Counter(card.identity for card in (*hand, *state.seen_cards))
    unseen: list[Card] = []
    for suit in SUITS:
        for rank in range(2, 15):
            identity = rank, suit
            if visible[identity] > 0:
                visible[identity] -= 1
            else:
                unseen.append(Card(rank, suit))
    return unseen


def _discard_candidates(hand: Sequence[Card], state: RunState, *, include_discard_effects=True
                        ) -> dict[tuple[int, ...], str]:
    """Generate distinct strategic holds; at most five cards are discarded."""
    count = len(hand)
    candidates: dict[tuple[int, ...], str] = {}
    minimum = max(0, count - 5)

    def offer(keep: Iterable[int], reason: str) -> None:
        selected = set(keep)
        if len(selected) < minimum:
            order = sorted(range(count), key=lambda i: _card_future_value(hand[i], state),
                           reverse=True)
            for index in order:
                if len(selected) >= minimum:
                    break
                selected.add(index)
        if len(selected) < count and count - len(selected) <= 5:
            candidates.setdefault(tuple(sorted(selected)), reason)

    valuable = sorted(range(count), key=lambda i: _card_future_value(hand[i], state),
                      reverse=True)
    offer(valuable[:minimum], "replace weak cards")
    offer(valuable[:max(minimum, 1)], "keep strongest card")
    offer(valuable[:max(minimum, 2)], "keep two strong cards")

    ranks: dict[int, list[int]] = defaultdict(list)
    suits: dict[str, list[int]] = defaultdict(list)
    for index, card in enumerate(hand):
        if not card.is_stone and not card.face_down:
            ranks[card.rank].append(index)
            for suit in _effective_suits(card, state):
                suits[suit].append(index)

    groups = [group for group in ranks.values() if len(group) >= 2]
    for group in groups:
        offer(group, f"grow {hand[group[0]].rank}s")
        if len(group) > 3:
            offer(group[:3], "preserve three of a kind")
    for first, second in combinations(groups, 2):
        offer(first + second, "grow two pair or full house")

    for suit, indices in suits.items():
        if len(indices) >= 2:
            indices = sorted(indices, key=lambda i: _card_future_value(hand[i], state),
                             reverse=True)
            offer(indices[:4], f"draw to {suit} flush")
            if len(indices) >= 3:
                offer(indices[:3], f"keep {suit} flush core")
    for core in flower_pot_cores(hand, state):
        offer(core, 'Flower Pot distinct-suit core')

    for window in _rank_windows(state):
        possible = [ranks[rank] for rank in window if ranks[rank]]
        if len(possible) < 3:
            continue
        kept = [max(group, key=lambda i: _card_future_value(hand[i], state))
                for group in possible]
        offer(kept, "draw to straight")
        for suit in SUITS:
            suited = [i for i in kept if suit in _effective_suits(hand[i], state)]
            if len(suited) >= 3:
                offer(suited, "draw to straight flush")

    if state.has("baron"):
        offer((i for i, card in enumerate(hand) if card.rank == 13),
              "hold kings for Baron")
    if state.has("shoot_the_moon"):
        offer((i for i, card in enumerate(hand) if card.rank == 12),
              "hold queens for Shoot the Moon")
    if any(card.enhancement == "steel" for card in hand):
        offer((i for i, card in enumerate(hand) if card.enhancement == "steel"),
              "hold steel cards")
    if include_discard_effects and any(j.key in ('burnt_joker', 'castle', 'yorick', 'trading_card')
                                       for j in state.active_jokers):
        from planning import first_discard
        first = first_discard(state)
        weakest = tuple(reversed(valuable))
        if first and state.effect_count('burnt_joker'):
            # Ordinary holds preserve made hands. Burnt needs the opposite:
            # intentionally discard a useful type to level it. Reserve one
            # affordable representative per category without subset explosion.
            cores = {(weakest[0],)}
            for group in groups:
                ordered = sorted(group, key=lambda i: _card_future_value(hand[i], state))
                for size in range(2, min(5, len(ordered)) + 1):
                    cores.add(tuple(ordered[:size]))
            for a, b in combinations(groups, 2):
                a = sorted(a, key=lambda i: _card_future_value(hand[i], state))
                b = sorted(b, key=lambda i: _card_future_value(hand[i], state))
                cores.add(tuple(a[:2] + b[:2]))
                if len(a) >= 3:
                    cores.add(tuple(a[:3] + b[:2]))
                if len(b) >= 3:
                    cores.add(tuple(a[:2] + b[:3]))
            for matching in suits.values():
                size = 4 if state.has('four_fingers') else 5
                if len(matching) >= size:
                    cores.add(tuple(sorted(matching, key=lambda i: _card_future_value(hand[i], state))[:size]))
            for window in _rank_windows(state):
                if all(ranks[r] for r in window):
                    cores.add(tuple(min(ranks[r], key=lambda i: _card_future_value(hand[i], state)) for r in window))
            categories = {}
            for core in cores:
                if not core or len(core) > 5 or any(hand[i].face_down for i in core):
                    continue
                category = classify([hand[i] for i in core], state).category
                value = sum(_card_future_value(hand[i], state) for i in core)
                if category not in categories or value < categories[category][0]:
                    categories[category] = value, set(core)
            for category, (_, dropped) in categories.items():
                offer((i for i in range(count) if i not in dropped), 'Burnt upgrade ' + category)
        for j in state.active_jokers:
            if j.key == 'castle' and 'suit' in dict(j.data):
                suit = SUITS[int(j.get('suit')) % 4]
                targets = [i for i in weakest if not _card_debuffed(hand[i], state)
                           and suit in _effective_suits(hand[i], state)]
                for size in range(1, min(5, len(targets)) + 1):
                    dropped = set(targets[:size])
                    offer((i for i in range(count) if i not in dropped), 'grow Castle with matching discards')
            if j.key == 'yorick' and 1 <= j.get('discards_remaining', 0) <= 5:
                size = int(j.get('discards_remaining'))
                if size <= count:
                    dropped = set(weakest[:size])
                    offer((i for i in range(count) if i not in dropped), 'reach Yorick growth threshold')
        if first and state.has('trading_card'):
            offer((i for i in range(count) if i != weakest[0]), 'Trading Card first single discard')
    targets = {int(a.get('rank', -1)) for a in state.abilities
               if a is not None and a.key == 'mail_in_rebate'}
    matching = [i for i in reversed(valuable) if hand[i].rank in targets
                and not hand[i].is_stone and not hand[i].face_down
                and not _card_debuffed(hand[i], state)]
    for size in range(1, min(5, len(matching)) + 1):
        dropped = set(matching[:size])
        offer((i for i in range(count) if i not in dropped), 'Mail-In Rebate matching discard')
    return candidates


def suit_coverage(cards, state):
    """Maximum distinct suit slots; each physical card fills only one."""
    masks = {0}
    for c in cards:
        if c.face_down or c.is_stone or _card_debuffed(c, state):
            continue
        allowed = _effective_suits(c, state)
        masks |= {mask | (1 << i) for mask in tuple(masks)
                  for i, suit in enumerate(SUITS) if suit in allowed and not mask & (1 << i)}
    return max((mask.bit_count() for mask in masks), default=0)


def flower_pot_cores(hand, state):
    """Bounded suit-diverse holds alongside the ordinary rank/flush plans."""
    if not state.has('flower_pot'):
        return ()
    ranks = Counter(c.rank for c in hand if not c.is_stone and not c.face_down)
    pools = {s: sorted((i for i,c in enumerate(hand) if not c.face_down
                       and not _card_debuffed(c,state) and s in _effective_suits(c,state)),
                      key=lambda i: (ranks[hand[i].rank], _card_future_value(hand[i],state)),
                      reverse=True)[:2] for s in SUITS}
    cores = set()
    for size in (3,4):
        for suits in combinations(SUITS,size):
            for chosen in product(*(pools[s] for s in suits)):
                if len(set(chosen)) == size:
                    cores.add(tuple(sorted(chosen)))
    def value(core):
        selected = [hand[i] for i in core]
        return (suit_coverage(selected,state), _preserved_pattern_value(selected,state),
                sum(_card_future_value(c,state) for c in selected))
    # Keep both a complete suit route and partial routes with room to draw.
    return tuple(core for size in (4,3) for core in
                 sorted((c for c in cores if len(c)==size),key=value,reverse=True)[:4])


def _cycle_play(hand: Sequence[Card], state: RunState) -> PlayDecision:
    """Spend a zero-score hand to draw when no scoring subset exists.

    A boss-rejected play still consumes a hand and draws replacements. Score
    stays zero; sampled next hands choose which partial combination to keep.
    """
    candidates = _discard_candidates(hand, state, include_discard_effects=False)
    if not candidates:
        candidates = {(): "replace the current hand"}
    unseen = _standard_unseen(hand, state)
    deck_left = max(0, state.draw_pile_size if state.draw_pile_size >= 0 else len(unseen))
    after = replace(state, hands_left=max(0, state.hands_left - 1))
    rng = Random(_sample_seed(hand, state))
    draws = {}
    best, best_key = None, None
    for keep, reason in candidates.items():
        indices = tuple(i for i in range(len(hand)) if i not in keep)
        if not 1 <= len(indices) <= 5:
            continue
        if state.forced_card_index >= 0 and state.forced_card_index not in indices:
            continue
        retained = [hand[i] for i in keep]
        draw_count = min(len(indices), deck_left, len(unseen),
                         max(0, state.hand_size - len(retained)))
        if draw_count not in draws:
            draws[draw_count] = tuple(tuple(rng.sample(unseen, draw_count)) for _ in range(24))
        scores = [_draw_estimate([*retained, *draw], after)
                  for draw in draws[draw_count]] if after.hands_left > 0 and draw_count else [0.0]
        wins = sum(score >= state.score_needed > 0 for score in scores) / len(scores)
        utility = _survival_utility(scores, after) + wins * max(1000, state.score_needed)
        utility += _preserved_pattern_value(retained, state) * .03
        key = (utility, draw_count, -sum(_card_future_value(hand[i], state) for i in indices))
        if best_key is None or key > best_key:
            estimate = score_hand([hand[i] for i in indices], state, retained)
            best = PlayDecision(indices, estimate, utility,
                f"zero-score cycle: {reason}; draw {draw_count}; preserve a chance to score next hand", indices)
            best_key = key
    if best is None:
        # A forced card can be excluded from every structural hold candidate.
        # The final hand or an empty draw pile must still reach a real outcome.
        indices = ((state.forced_card_index,) if state.forced_card_index >= 0 else (0,))
        held = [card for i, card in enumerate(hand) if i not in indices]
        estimate = score_hand([hand[i] for i in indices], state, held)
        best = PlayDecision(indices, estimate, 0.0,
                            "zero-score cycle: no replacement plan; consume the remaining hand", indices)
    return best


def _sample_seed(hand: Sequence[Card], state: RunState) -> int:
    seed = state.seed + 7919 * state.ante
    for index, card in enumerate(hand):
        seed += (index + 1) * (card.rank * 17 + ord((card.suit or "?").upper()[:1]))
    return seed


def _mouth_opening_plan(hand, state, decision):
    """Value repeatability before committing this blind to one category."""
    retained = [c for i, c in enumerate(hand) if i not in decision.indices]
    unseen = _standard_unseen(hand, state)
    deck_left = state.draw_pile_size if state.draw_pile_size >= 0 else len(unseen)
    draw_count = min(len(decision.indices), len(unseen), max(0, deck_left))
    after = replace(state, hands_left=state.hands_left - 1,
                    boss_target_hand=decision.estimate.category,
                    boss_first_hand_done=True,
                    round_score=state.round_score + decision.estimate.score)
    rng = Random(_sample_seed(hand, state))
    scores = [_draw_estimate([*retained, *rng.sample(unseen, draw_count)], after)
              for _ in range(24)]
    probability = sum(score > 0 for score in scores) / len(scores)
    wins = sum(score >= after.score_needed for score in scores) / len(scores)
    expected = sum(scores) / len(scores)
    utility = decision.estimate.score + .65 * expected * after.hands_left
    utility += wins * state.score_needed * .65
    utility += probability * min(state.score_needed, decision.estimate.score) * .5
    utility -= (1 - probability) * after.score_needed * .2
    return replace(decision, utility=utility,
                   reason=f"{decision.reason}; Mouth repeat probability {probability:.0%}, next score {expected:.0f}")


def _hand_after_draw(kept: Sequence[Card], unseen: Sequence[Card],
                     draw_count: int, rng: Random) -> list[Card]:
    if draw_count <= 0 or not unseen:
        return list(kept)
    sample = rng.sample(list(unseen), min(draw_count, len(unseen)))
    return [*kept, *sample]


def _discard_cost(state: RunState, current_score: float) -> float:
    """Discourages spending the final discard for a small improvement."""
    base = max(8.0, current_score * 0.07)
    if state.discards_left == 1:
        base *= 1.7
    if state.hands_left <= 1:
        base *= 0.5
    if state.has("banner"):
        base += 30 * max(1, state.hands_left)
    if state.has("green_joker"):
        base += 5 * max(1, state.hands_left)
    if state.has("mystic_summit") and state.discards_left == 1:
        base *= 0.4
    if state.has("ramen"):
        base += current_score * .05
    return base


def _survival_utility(scores: Sequence[float], state: RunState) -> float:
    if not scores:
        return float("-inf")
    sorted_scores = sorted(scores)
    mean = sum(scores) / len(scores)
    median = sorted_scores[len(scores) // 2]
    lower = sorted_scores[max(0, len(scores) // 5)]
    needed = state.score_needed / max(1, state.hands_left)
    wins = (sum(score >= state.score_needed for score in scores) / len(scores)
            if state.score_needed > 0 else 0)
    if state.hands_left <= 1:
        # With one hand remaining, surviving the blind dominates average score.
        return 0.40 * mean + 0.35 * median + 0.25 * lower + wins * needed * 2
    if needed > mean:
        return 0.70 * mean + 0.20 * median + 0.10 * lower + wins * needed * 0.4
    return 0.55 * mean + 0.35 * median + 0.10 * lower


def _draw_best_plan(hand: Sequence[Card], state: RunState):
    """Bounded structural search for simulations; real plays stay exhaustive."""
    indices = tuple(range(len(hand)))
    plans = {(i,) for i in indices}
    ranks = defaultdict(list)
    for i, card in enumerate(hand):
        if not card.is_stone and not card.face_down:
            ranks[card.rank].append(i)
    groups = []
    for group in ranks.values():
        for size in range(2, min(5, len(group)) + 1):
            for core in combinations(group, size):
                plans.add(core)
                groups.append(core)
    for first, second in combinations(groups, 2):
        if len(first) + len(second) <= 5 and set(first).isdisjoint(second):
            plans.add(tuple(sorted(first + second)))
    minimum = 4 if state.has("four_fingers") else 5
    for suit in SUITS:
        matching = [i for i in indices if suit in _effective_suits(hand[i], state)]
        for core in combinations(matching, minimum):
            plans.add(core)
        if minimum == 4:
            for core in combinations(matching, 5):
                plans.add(core)
    for window in _rank_windows(state):
        if all(ranks[rank] for rank in window):
            # Different suits or editions can matter: keep both scoring-first
            # and same-suit representatives without exploding draw samples.
            core = tuple(sorted(max(ranks[r], key=lambda i: _card_future_value(hand[i], state)) for r in window))
            plans.add(core)
            for suit in SUITS:
                suited = [next((i for i in ranks[r] if suit in _effective_suits(hand[i], state)), -1) for r in window]
                if all(i >= 0 for i in suited):
                    plans.add(tuple(sorted(suited)))
    # Shortcut's gapped runs require rank subsets outside ordinary windows.
    if state.has("shortcut"):
        for core in combinations(indices, minimum):
            if _straight_indices([hand[i] for i in core], state):
                plans.add(core)
    extra_order = sorted(indices, key=lambda i: _card_future_value(hand[i], state))
    for core in tuple(plans):
        rest = [i for i in extra_order if i not in core]
        if rest:
            plans.add(tuple(sorted((*core, *rest[:5 - len(core)]))))
        if state.has("square_joker") and len(core) < 4:
            plans.add(tuple(sorted((*core, *rest[:4 - len(core)]))))
    candidates = ((core, score_hand([hand[i] for i in core], state,
                                   [card for i, card in enumerate(hand) if i not in core]))
                  for core in sorted(plans) if core)
    risk = (state.score_needed > 0 and state.score_mode == 'expected' and
            (_boss(state) == 'hook' or state.has('misprint') or state.has('bloodstone') or state.has('space_joker')
             or any(c.enhancement == 'lucky' or c.face_down for c in hand)))
    def draw_key(pair):
        core, result = pair
        floor = _play_floor(core, result, hand, state) if risk else result.score
        clears = state.score_needed > 0 and floor >= state.score_needed
        development = 0.
        if clears:
            # Match the real policy's safe growth preferences. No extra draws
            # or full lookahead are needed for this bounded simulation search.
            from engine_development import play_development_value
            development = play_development_value(core, result, hand, state, guaranteed_score=floor)
        return clears, result.score + development, HAND_RANK[result.category], len(core)
    return max(candidates, key=draw_key)


def _draw_best_estimate(hand: Sequence[Card], state: RunState) -> ScoreEstimate:
    return _draw_best_plan(hand, state)[1]


def _draw_estimate(hand: Sequence[Card], state: RunState) -> float:
    return _draw_best_estimate(hand, state).score


def _last_hand_discard(hand, state, current, unseen):
    """Prefer any route to survival over a hand known to lose.

    Keep the current scoring core and valuable held cards available while
    searching. A lower mean can still be the only way to finish the blind.
    """
    candidates = _discard_candidates(hand, state)
    protected = set(current.indices)
    protected.update(i for i, card in enumerate(hand)
                     if card.enhancement == 'steel' or (state.has('baron') and card.rank == 13))
    expendable = sorted((i for i in range(len(hand)) if i not in protected),
                        key=lambda i: _card_future_value(hand[i], state))
    for count in range(1, min(5, len(expendable)) + 1):
        dropped = set(expendable[:count])
        keep = tuple(i for i in range(len(hand)) if i not in dropped)
        candidates.setdefault(keep, 'preserve made hand and cycle spare cards')
    # A single-card out must not disappear in eight random samples or in the
    # pruning used for ordinary 13-card hands.
    for index in range(len(hand)):
        if index not in protected:
            candidates.setdefault(tuple(i for i in range(len(hand)) if i != index),
                                  'look for a single-card out')
    rng = Random(_sample_seed(hand, state))
    draw_limit = min(len(unseen), state.draw_pile_size if state.draw_pile_size >= 0 else len(unseen))
    draws = {}
    best, best_key = None, None
    from planning import advance_projected_discard
    for keep, reason in candidates.items():
        discarded = tuple(i for i in range(len(hand)) if i not in keep)
        draw_count = min(len(discarded), draw_limit)
        if not draw_count:
            continue
        if draw_count not in draws:
            draws[draw_count] = (tuple((card,) for card in unseen) if draw_count == 1
                                 else tuple(tuple(rng.sample(unseen, draw_count)) for _ in range(48)))
        retained = [hand[i] for i in keep]
        sampled = advance_projected_discard(state, [hand[i] for i in discarded], draw_count)
        scores = [_draw_estimate([*retained, *draw], sampled) for draw in draws[draw_count]]
        wins = sum(score >= state.score_needed for score in scores) / len(scores)
        expected = sum(scores) / len(scores)
        # After another miss, keeping the best core and more draw capacity is
        # still preferable to spending the last hand on certain defeat.
        key = (wins, max(scores), expected, -len(discarded))
        if best_key is None or key > best_key:
            best_key = key
            best = DiscardDecision(discarded, expected, wins,
                f'last hand needs {state.score_needed:.0f}; {reason}; '
                f'sampled clear probability {wins:.1%}, mean {expected:.0f}')
    if best is not None:
        return best
    return DiscardDecision((), current.estimate.score, current.utility, 'no legal draw remains')


def decide_discard(hand: Sequence[Card], state: RunState | None = None,
                   samples: int = 16, *, current: PlayDecision | None = None) -> DiscardDecision:
    """Compare draw plans against playing now using a visible-deck posterior."""
    if not hand:
        raise ValueError("Hand is empty")
    state = state or RunState()
    current = current if current is not None else decide_play(hand, state)
    if current.training:
        return DiscardDecision((), current.estimate.score, current.utility,
                               'preserve the verified retained training finisher')
    current_order = current.play_order or current.indices
    guaranteed = score_hand([hand[i] for i in current_order],
                            replace(state, score_mode='conservative'),
                            [card for i, card in enumerate(hand) if i not in current.indices]).score
    if state.discards_left <= 0 or len(hand) < 2:
        return DiscardDecision((), current.estimate.score, current.utility,
                               "no discard available")
    if guaranteed >= state.score_needed > 0:
        if state.objective == 'survival':
            from economy import safe_income_discard
            income = safe_income_discard(hand, state, current)
            if income is not None:
                return income
        return DiscardDecision((), current.estimate.score, current.utility,
                               "play to clear blind now")
    if _boss(state) == "water":
        return DiscardDecision((), current.estimate.score, current.utility,
                               "Water disables discards")
    # A made large hand on pace should not spend every discard fishing.
    pace = state.score_needed / max(1, state.hands_left)
    mouth_opening = _boss(state) == 'mouth' and not locked_mouth_hand(state)
    from planning import first_discard, advance_projected_discard
    income_targets = any(a is not None and a.key == 'mail_in_rebate'
                        and 2 <= a.get('rank', -1) <= 14 for a in state.abilities)
    income_search = income_targets or state.has('trading_card') and first_discard(state)
    burnt_upgrade = state.effect_count('burnt_joker') and first_discard(state)
    if (not mouth_opening and not burnt_upgrade and not income_search and state.hands_left > 1 and guaranteed >= pace * 1.1
            and HAND_RANK[current.estimate.category] >= HAND_RANK["two_pair"]):
        return DiscardDecision((), current.estimate.score, current.utility,
                               "made hand is above required scoring pace; save discards")
    if not mouth_opening and not burnt_upgrade and not income_search and state.has("green_joker") and current.estimate.score >= pace:
        return DiscardDecision((), current.estimate.score, current.utility,
                               "on pace; preserve Green Joker scaling")
    unseen = _standard_unseen(hand, state)
    if not unseen or state.draw_pile_size == 0:
        return DiscardDecision((), current.estimate.score, current.utility,
                               "no cards left to draw")

    if state.hands_left == 1:
        return _last_hand_discard(hand, state, current, unseen)

    candidates = _discard_candidates(hand, state)
    if len(hand) > 10:
        # Large hands make exhaustive draw scoring expensive. Keep structurally
        # distinct plans and sample eight draws, still search all final subsets.
        from planning import diverse_draw_candidates
        candidates = diverse_draw_candidates(hand, state, candidates, limit=14)
        samples = min(samples, 8)
    best = DiscardDecision((), current.estimate.score, current.estimate.score,
                           f"play current {current.estimate.category}")
    rng = Random(_sample_seed(hand, state))
    sample_count = max(8, min(samples, 64))
    cost = _discard_cost(state, current.estimate.score)
    reserve_for_later = (not mouth_opening and not burnt_upgrade and not income_search and state.ante >= 3
        and current.estimate.category in ('high_card', 'pair') and state.active_jokers
        and guaranteed >= pace * 1.3 and pace > 0
        and not any(state.has(key) for key in ('castle', 'yorick', 'trading_card')))
    if reserve_for_later:
        # Real late-game Pair engines spent discards for marginal gains while
        # already above pace. Keep a reserve for a worse subsequent draw;
        # a substantial scoring improvement can still justify discarding.
        cost += guaranteed * .15
        best = replace(best, reason='small-hand engine is above scoring pace; draw gain must justify using a reserve discard')
    if mouth_opening:
        cost *= .15
    baseline = current.estimate.score
    # Every candidate with the same draw count sees the same sampled cards.
    # This reduces variance when comparing close discard plans.
    draw_limit = min(len(unseen), state.draw_pile_size if state.draw_pile_size >= 0 else len(unseen))
    draw_sets = {
        draw_count: tuple(
            tuple(rng.sample(unseen, min(draw_count, draw_limit)))
            for _ in range(sample_count)
        )
        for draw_count in {len(hand) - len(keep) for keep in candidates}
        if draw_limit > 0
    }

    for keep, reason in candidates.items():
        retained = [hand[i] for i in keep]
        draw_count = len(hand) - len(keep)
        if draw_count not in draw_sets:
            continue
        scores: list[float] = []
        discarded_cards = [c for i, c in enumerate(hand) if i not in keep]
        sample_state = advance_projected_discard(state, discarded_cards, min(draw_count, draw_limit))
        for draw in draw_sets[draw_count]:
            new_hand = [*retained, *draw]
            scores.append(_draw_estimate(new_hand, sample_state))
        expected = sum(scores) / len(scores)
        utility = _survival_utility(scores, state) - cost
        follow_up = 0
        if (state.discards_left >= 2 and state.score_needed > 0 and
                current.estimate.score < pace * .8 and len(hand) <= 13):
            from planning import second_discard_gain
            follow_up = second_discard_gain(hand, keep, draw_sets[draw_count], scores,
                                            unseen, sample_state, state)
            utility += follow_up
        if mouth_opening:
            utility += sum(score >= state.score_needed for score in scores) / len(scores) * state.score_needed * .6
        utility += _preserved_pattern_value(retained, state) * 0.03
        purple = sum(hand[i].seal.lower() == "purple" and not _card_debuffed(hand[i], state)
                     for i in range(len(hand)) if i not in keep)
        utility += 3 * min(purple, max(0, state.consumable_slots - state.consumables_held))
        from economy import discard_income, delayed_income
        utility += (discard_income(discarded_cards, state) - delayed_income(state)) * max(1., min(20., pace * .015))
        if utility > max(baseline, best.utility):
            discarded = tuple(i for i in range(len(hand)) if i not in keep)
            best = DiscardDecision(discarded, expected, utility,
                                   f"{reason}; estimated score {expected:.0f}"
                                   + (f'; next-discard option {follow_up:.0f}' if follow_up > 0 else ''))
    return best


def choose_discard(hand: list[Card]) -> tuple[list[int], str]:
    """Runner-compatible discard choice."""
    decision = decide_discard(hand)
    return list(decision.indices), decision.reason


def _money_value(money: int, ante: int, state: RunState | None = None) -> float:
    """Interest and flexibility make early cash more useful than late cash."""
    from economy import interest as cash_interest
    earned = cash_interest(state, money) if state is not None else min(25, max(0, money)) // 5
    horizon = max(1, 9 - ante)
    return money + earned * min(4, horizon) * 0.6


def _joker_standalone_value(joker: Joker, state: RunState) -> float:
    """A rough one-round value before deck-specific synergy is applied."""
    key = joker.key
    fixed_mult = {
        "joker": 12, "jolly_joker": 12, "zany_joker": 10,
        "mad_joker": 9, "crazy_joker": 10, "droll_joker": 10,
        "half_joker": 9, "gros_michel": 17,
        "misprint": 11, "popcorn": 14,
    }
    fixed_chips = {
        "sly_joker": 13, "wily_joker": 14, "clever_joker": 13,
        "devious_joker": 14, "crafty_joker": 13,
        "stuntman": 32, "blue_joker": 18,
        "runner": max(5, joker.value * 0.22),
        "square_joker": max(4, joker.value * 0.22),
        "ice_cream": max(4, (joker.value or 100) * 0.2),
    }
    if key in fixed_mult:
        return fixed_mult[key]
    if key in fixed_chips:
        return fixed_chips[key]
    if key in {'ancient_joker', 'joker_stencil', 'fibonacci', 'raised_fist', 'seltzer', 'supernova', 'scholar'}:
        # These were repeatedly passed over in the recorded shops. Their
        # long-term ceiling complements the marginal next-hand simulation.
        return {'ancient_joker': 25, 'joker_stencil': 22, 'fibonacci': 22,
                'raised_fist': 20, 'seltzer': 15,
                'supernova': 12 + max(state.played_types.values(), default=0) * 1.5,
                'scholar': 15}[key]
    if key in {"the_duo", "the_trio", "the_family", "the_order",
               "the_tribe", "cavendish", "card_sharp", "blackboard"}:
        return 25
    if key in {"hologram", "constellation", "vampire", "ramen",
               "acrobat", "flower_pot", "seeing_double"}:
        return 16
    if key in {"four_fingers", "shortcut", "smeared_joker"}:
        return 13
    if key in {"bull", "bootstraps"}:
        return 10 + state.money * 0.35
    if key in {"baron", "shoot_the_moon", "steel_joker"}:
        return 7
    if key in {"photograph", "hanging_chad", "sock_and_buskin",
               "hack", "mime", "splash"}:
        return 11
    if key in {"blueprint", "brainstorm"}:
        return 18 if state.active_jokers else 5
    if key in {"egg", "gift_card", "trading_card", "dna"}:
        return 7 + max(0, 7 - state.ante)
    if key in {"green_joker", "ride_the_bus", "spare_trousers"}:
        return 12 + joker.value * .6 + max(0, 7 - state.ante)
    if key in {"golden_joker", "cloud_9", "rocket", "mail_in_rebate", "business_card"}:
        return 7 + max(0, 9 - state.ante) * 1.8
    if key in {"juggler", "turtle_bean", "troubadour"}:
        return 10 + (6 if state.hand_size <= 8 else 2)
    if key in {"burglar", "drunkard", "merry_andy"}:
        return 13
    if key in {"burnt_joker", "space_joker", "perkeo"}:
        return 16 + max(0, 8 - state.ante)
    if key in {"canio", "yorick", "triboulet"}:
        return 32
    if key in {"ceremonial_dagger", "madness", "ankh"}:
        return 3 if len(state.jokers) >= 3 else 9
    return 5 + max(0, 5 - state.ante) * 0.5


def _deck_profile(state: RunState) -> dict[str, float]:
    deck = state.deck_cards or state.remaining_deck or tuple(
        Card(rank, suit) for suit in SUITS for rank in range(2, 15)
    )
    count = max(1, len(deck))
    suits = Counter(card.suit for card in deck if not card.is_stone)
    ranks = Counter(card.rank for card in deck if not card.is_stone)
    enhancements = Counter(card.enhancement for card in deck)
    return {
        "face": sum(card.is_face for card in deck) / count,
        "ace": ranks[14] / count,
        "king": ranks[13] / count,
        "queen": ranks[12] / count,
        "steel": enhancements["steel"] / count,
        "glass": enhancements["glass"] / count,
        "enhanced": sum(bool(c.enhancement) for c in deck) / count,
        "wild": enhancements["wild"] / count,
        "spade": suits["S"] / count,
        "heart": suits["H"] / count,
        "diamond": suits["D"] / count,
        "club": suits["C"] / count,
        "rank_density": max(ranks.values(), default=0) / count,
        "suit_density": max(suits.values(), default=0) / count,
    }


def _joker_synergy(joker: Joker, state: RunState,
                   profile: dict[str, float]) -> float:
    key = joker.key
    existing = {j.key for j in state.active_jokers}
    bonus = 0.0
    if key == "baron":
        bonus += max(0, profile["king"] - 4 / 52) * 180
        bonus += 9 if "mime" in existing else 0
    if key == "shoot_the_moon":
        bonus += max(0, profile["queen"] - 4 / 52) * 100
    if key == "steel_joker":
        bonus += profile["steel"] * 100
    if key == 'vampire':
        if 'midas_mask' in existing:
            bonus += 12
        elif joker.value <= 1.05:
            # A new X1 Vampire consumes a slot but has no scoring growth
            # without enhancements. Value only the support already present.
            bonus -= 12 * max(0, 1 - 8 * profile['enhanced'])
    if key == "photograph":
        bonus += 13 if "hanging_chad" in existing else 0
        bonus += profile["face"] * 18
    if key == "hanging_chad":
        bonus += 13 if "photograph" in existing else 0
        bonus += profile["glass"] * 35
    if key == "sock_and_buskin":
        bonus += profile["face"] * 25
        bonus += 10 if "triboulet" in existing else 0
    if key == "hack":
        bonus += sum(8 for card in (state.deck_cards or state.remaining_deck)
                     if 2 <= card.rank <= 5 and not card.is_stone) / 10
    if key == "smeared_joker":
        bonus += 9 if "the_tribe" in existing else 0
        bonus += 5 if "crafty_joker" in existing else 0
    if key == "four_fingers":
        bonus += 9 if "the_order" in existing or "the_tribe" in existing else 0
    if key == "shortcut":
        bonus += 8 if "runner" in existing or "the_order" in existing else 0
    if key in {"the_tribe", "droll_joker", "crafty_joker"}:
        bonus += max(0, profile["suit_density"] - 0.25) * 60
    if key in {"the_trio", "wily_joker", "zany_joker"}:
        bonus += max(0, profile["rank_density"] - 4 / 52) * 110
    if key in {"blueprint", "brainstorm"}:
        best_existing = max((_joker_standalone_value(j, state)
                             for j in state.active_jokers), default=0)
        bonus += best_existing * 0.55
    if key == "bull":
        bonus += min(20, state.money) * 0.4
    if key == "bootstraps":
        bonus += (state.money // 5) * 1.5
    if key == "flower_pot" and "smeared_joker" in existing:
        bonus -= 10
    if key == "blackboard" and profile["suit_density"] > 0.4:
        bonus += 4
    return bonus


def _joker_purchase_value(joker: Joker, state: RunState) -> float:
    profile = _deck_profile(state)
    value = _joker_standalone_value(joker, state)
    value += _joker_synergy(joker, state, profile)
    if joker.edition == "foil":
        value += 9
    if joker.edition == "holographic":
        value += 11
    if joker.edition == "polychrome":
        value += 17
    if joker.edition == "negative":
        value += 14
    if joker.key in {j.key for j in state.active_jokers}:
        if joker.key in {"four_fingers", "shortcut", "smeared_joker", "splash"}:
            value *= 0.25
    if joker.debuffed:
        value *= 0.3
    return value


def _planet_purchase_value(name: str, state: RunState) -> float:
    category = PLANET_HAND.get(joker_name(name))
    if category is None:
        return 2
    level = state.level(category)
    played = state.played(category)
    chips_gain = HAND_LEVELS[category][2]
    mult_gain = HAND_LEVELS[category][3]
    base = 2 + 0.11 * chips_gain + 2.8 * mult_gain
    base *= 1 + min(1.2, played * 0.13)
    base /= 1 + 0.15 * (level - 1)
    if category in {"high_card", "pair"}:
        base *= 1.15
    if category in {"flush_five", "flush_house", "five_kind"} and played == 0:
        base *= 0.25
    defaults = {'pair': .30, 'two_pair': .25, 'flush': .20, 'straight': .15,
                'three_kind': .10, 'high_card': .10, 'full_house': .08}
    # Shrink a lucky opening quad/straight flush toward the deck prior. One
    # spectacular hand is not evidence of a reproducible scoring engine.
    prior = defaults.get(category, .005)
    frequency = (played + 8 * prior) / (sum(state.played_types.values()) + 8)
    base *= min(1.2, .1 + 2 * frequency)
    return base


def _tarot_purchase_value(name: str, state: RunState) -> float:
    key = joker_name(name)
    profile = _deck_profile(state)
    values = {
        "the_fool": 13, "the_magician": 10, "the_high_priestess": 12,
        "the_empress": 10, "the_emperor": 13, "the_hierophant": 8,
        "the_lovers": 7, "the_chariot": 14, "justice": 9,
        "the_hermit": min(20, state.money), "the_wheel_of_fortune": 6,
        "strength": 12, "the_hanged_man": 13, "death": 17,
        "temperance": min(50, sum(j.sell_value for j in state.jokers)),
        "the_devil": 11, "the_tower": 6, "the_star": 8,
        "the_moon": 8, "the_sun": 8, "judgement": 15,
        "the_world": 8,
    }
    value = values.get(key, 5)
    if key in {"the_star", "the_moon", "the_sun", "the_world"}:
        value += max(0, profile["suit_density"] - 0.25) * 30
    if key == "the_chariot" and state.has("baron"):
        value += 10
    if key == "death" and profile["glass"] + profile["steel"] > 0:
        value += 9
    return value


def _voucher_purchase_value(name: str, state: RunState) -> float:
    key = joker_name(name)
    value = {
        "overstock": 18, "overstock_plus": 24,
        "clearance_sale": 18, "liquidation": 26,
        "hone": 13, "glow_up": 19,
        "reroll_surplus": 15, "reroll_glut": 18,
        "crystal_ball": 9, "omen_globe": 12,
        "telescope": 17, "observatory": 21,
        "grabber": 28, "nacho_tong": 30,
        "wasteful": 19, "recyclomancy": 23,
        "tarot_merchant": 12, "tarot_tycoon": 16,
        "planet_merchant": 12, "planet_tycoon": 16,
        "seed_money": 16, "money_tree": 22,
        "blank": 3, "antimatter": 36,
        "paint_brush": 24, "palette": 28,
    }.get(key, 6)
    if key in {"seed_money", "money_tree"}:
        value += min(12, state.money * 0.3)
    return value * max(0.4, min(1.0, (10 - state.ante) / 8))


def _item_value(item: ShopItem, state: RunState) -> float:
    kind = joker_name(item.kind)
    if kind == "joker":
        return _joker_purchase_value(item.joker or Joker(item.name), state)
    if kind == "planet":
        return _planet_purchase_value(item.name, state)
    if kind == "tarot":
        return _tarot_purchase_value(item.name, state)
    if kind == "voucher":
        return _voucher_purchase_value(item.name, state)
    if kind == "spectral":
        return 11 + max(0, 6 - state.ante)
    if kind in {"arcana_pack", "celestial_pack", "standard_pack", "buffoon_pack"}:
        if item.pack_contents:
            content_kind = {"arcana_pack": "tarot", "celestial_pack": "planet",
                            "buffoon_pack": "joker", "standard_pack": "card"}[kind]
            contents = [ShopItem(content_kind, name, 0) for name in item.pack_contents]
            return max((_item_value(content, state) for content in contents), default=5)
        return {"arcana_pack": 10, "celestial_pack": 9,
                "standard_pack": 8, "buffoon_pack": 15}[kind]
    return 3


def next_blind_target(state: RunState) -> float:
    bases = (300, 800, 2000, 5000, 11000, 20000, 35000, 50000)
    base = bases[min(max(1, state.ante), 8) - 1]
    if state.blind_index == 0:
        return base * 1.5
    if state.blind_index == 1:
        if state.upcoming_boss_ante == state.ante and state.upcoming_boss_target > 0:
            return state.upcoming_boss_target
        return base * 2
    if state.blind_index == 2:
        # The HUD may already show the new ante in the cashout/shop.
        next_ante = state.ante if state.upcoming_boss_ante < state.ante else state.ante + 1
        return bases[min(max(1, next_ante), 8) - 1]
    if state.target_score <= base:
        return base * 1.5
    if state.target_score <= base * 1.5:
        return base * 2
    return bases[min(state.ante, 7)] if state.ante < 8 else base * 2


@lru_cache(maxsize=64)
def _shop_probe_hands(deck: tuple[Card, ...], count: int):
    rng = Random(17011991)
    return tuple(tuple(rng.sample(deck, min(count, len(deck)))) for _ in range(20))


def _shop_order_variants(state):
    """Compare realizable orders, keeping copier targets explicit."""
    examples = (
        (Card(14, 'S'), Card(14, 'H'), Card(14, 'D'), Card(14, 'C'), Card(13, 'S')),
        tuple(Card(rank, 'H') for rank in (2, 3, 4, 5, 6)),
        (Card(14, 'S'), Card(14, 'H'), Card(13, 'S'), Card(13, 'H')),
    )
    patterns = [classify(hand, state) for hand in examples]
    last = replace(state, hands_left=1)
    def multiplicative(joker):
        return joker.key == 'card_sharp' or joker.edition == 'polychrome' or any(
            _joker_global_effect(joker, pattern, (Card(13, 'S'),), last)[2] > 1 for pattern in patterns)
    canonical = tuple(sorted(range(len(state.jokers)), key=lambda i: multiplicative(state.jokers[i])))
    orders = {tuple(range(len(state.jokers))), canonical}
    copiers = [i for i, j in enumerate(state.jokers) if j.key in ('blueprint', 'brainstorm')]
    targets = [i for i, j in enumerate(state.jokers) if j.key not in ('blueprint', 'brainstorm')
               and j.key not in NONCOPYABLE and j.get('copyable', 1)]
    targets.sort(key=lambda i: _joker_purchase_value(state.jokers[i], state), reverse=True)
    frontier = [list(canonical)]
    for copier in copiers:
        next_frontier = []
        for order in frontier:
            for target in targets[:3]:
                candidate = [i for i in order if i != copier]
                if state.jokers[copier].key == 'blueprint':
                    candidate.insert(candidate.index(target), copier)
                else:
                    candidate.remove(target)
                    candidate.insert(0, target)
                    candidate.append(copier)
                orders.add(tuple(candidate))
                next_frontier.append(candidate)
        frontier = next_frontier[:9]
    return tuple(tuple(state.jokers[i] for i in order) for order in sorted(orders))


def _shop_round_state(state: RunState) -> RunState:
    """Purchases prepare the next round, whose action counters reset."""
    hands = state.base_hands_per_round + 3 * state.effect_count('burglar') - state.joker_count('troubadour')
    discards = state.base_discards_per_round + state.joker_count('drunkard') + 3 * state.joker_count('merry_andy')
    if state.effect_count('burglar'):
        discards = 0
    return replace(state, hands_left=max(1, hands), discards_left=max(0, discards), round_discards_used=0)


def _shop_scores(state: RunState) -> tuple[float, ...]:
    """Common deck samples compare marginal scoring value, not rarity."""
    state = _shop_round_state(state)
    deck = state.deck_cards or state.remaining_deck or tuple(Card(rank, suit) for suit in SUITS for rank in range(2, 15))
    probe = replace(state, boss='', boss_target_hand='', boss_seen_types=frozenset(),
                    boss_first_hand_done=False, forced_card_index=-1,
                    hands_left=max(1, state.hands_left), discards_left=max(0, state.discards_left // 2), round_played_types={},
                    draw_pile_size=max(0, len(deck) - state.hand_size), seen_cards=(),
                    round_score=0, target_score=0)
    scores = []
    orders = _shop_order_variants(probe)
    histories = [{} for _ in orders]
    for index, hand in enumerate(_shop_probe_hands(tuple(deck), max(1, state.hand_size))):
        # Later hands can activate Card Sharp; first-hand-only jokers still
        # receive a fair share without pretending every play is the first.
        phase = index % 4
        if not phase:
            histories = [{} for _ in orders]
        later = replace(probe,
                        boss_first_hand_done=bool(phase),
                        hands_left=max(1, state.hands_left - round(phase * (state.hands_left - 1) / 3)))
        # A conservative forecast of net Green Joker growth includes expected
        # discards. It does not treat every sampled hand as value zero forever.
        candidates = []
        for order_index, order in enumerate(orders):
            history = histories[order_index]
            estimate = _draw_best_estimate(hand, replace(later, round_played_types=history, jokers=tuple(
                replace(j, value=j.value + .5 * phase) if j.key == 'green_joker' else j for j in order)))
            candidates.append(estimate.score)
            history[estimate.category] = history.get(estimate.category, 0) + 1
        scores.append(max(candidates))
    return tuple(scores)


def _score_summary(scores):
    return .7 * sum(scores) / len(scores) + .3 * sorted(scores)[len(scores) // 2]


def shop_strength(state: RunState) -> dict[str, float | bool]:
    from planning import shop_forecast
    state = _shop_round_state(state)
    forecast = shop_forecast(state)
    return {'typical_hand_score': forecast.next_round.typical_hand, 'next_target': forecast.next_round.target,
            'weak_engine': forecast.weak, 'safe_round_score': forecast.next_round.safe_score,
            'sample_clear_rate': forecast.next_round.clear_rate,
            'sustainable_round_score': forecast.sustainable_score,
            'fragile_support_score': forecast.fragile_score,
            'fragile_loss_probability': forecast.fragile_loss_probability,
            'upcoming_boss': state.upcoming_boss,
            'boss_target': forecast.boss_round.target if forecast.boss_round else
                           (forecast.next_round.target if forecast.next_round.boss else 0),
            'boss_safe_score': forecast.boss_round.safe_score if forecast.boss_round else
                               (forecast.next_round.safe_score if forecast.next_round.boss else 0),
            'next_round_worst_sample': min(forecast.next_round.scores, default=0),
            'next_round_hands': forecast.next_round.hands,
            'deck_composition_known': bool(state.deck_cards or state.remaining_deck),
            'preferred_hand': forecast.profile.preferred_hand,
            'engine_weaknesses': forecast.profile.weaknesses,
            'conditional_multiplier_activation_estimate': getattr(forecast.next_round, 'conditional_xmult_rates', ()),
            'development_score_estimate': forecast.development_score}


def joker_hand_size_delta(before: Sequence[Joker], after: Sequence[Joker]) -> int:
    sizes = {'juggler': 1, 'troubadour': 2, 'stuntman': -2, 'merry_andy': -1}
    def bonus(joker):
        return int(joker.get('hand_size_bonus', 5)) if joker.key == 'turtle_bean' else sizes.get(joker.key, 0)
    return sum(bonus(j) for j in after) - sum(bonus(j) for j in before)


def _purchase_state(item: ShopItem, state: RunState, sell_index=-1) -> RunState:
    if joker_name(item.kind) == 'voucher':
        from shop_extras import voucher_state
        return voucher_state(state, item)
    money = state.money - item.price
    jokers = list(state.jokers)
    slots = state.total_joker_slots
    if sell_index >= 0:
        old = jokers.pop(sell_index)
        money += old.sell_value
        slots -= old.edition == 'negative'
    kind = joker_name(item.kind)
    if kind == 'joker':
        joker = item.joker or Joker(item.name)
        jokers.append(joker)
        slots += joker.edition == 'negative'
    hand_delta = joker_hand_size_delta(state.jokers, jokers)
    hands_delta = 3 * (sum(j.key == 'burglar' for j in jokers) - sum(j.key == 'burglar' for j in state.jokers))
    hands_delta -= sum(j.key == 'troubadour' for j in jokers) - sum(j.key == 'troubadour' for j in state.jokers)
    levels = dict(state.levels)
    if kind == 'planet' and joker_name(item.name) in PLANET_HAND:
        category = PLANET_HAND[joker_name(item.name)]
        levels[category] = state.level(category) + 1
        jokers = [replace(j, value=max(1, j.value) + .1) if j.key == 'constellation' else j for j in jokers]
    discards = 0 if any(j.key == 'burglar' for j in jokers) else state.discards_left
    return replace(state, money=money, jokers=tuple(jokers), levels=levels, total_joker_slots=slots,
                   hand_size=max(1, state.hand_size + hand_delta),
                   hands_left=max(1, state.hands_left + hands_delta), discards_left=discards)


def decide_shop(items: Sequence[ShopItem], state: RunState, *,
                reroll_cost: int | None = None) -> ShopDecision:
    """Compare complete rounds, Boss exposure and permanent engine support."""
    from planning import (shop_forecast, XMULT_JOKERS, GROWTH_JOKERS, ECONOMY_JOKERS,
                          replacement_reduces_survival, growth_opportunity)
    from economy import economy_value
    state = _shop_round_state(state)
    before = shop_forecast(state)
    economy_before = economy_value(state, before)
    baseline = before.next_round.safe_score
    needs_starter = (not state.active_jokers and state.ante <= 2 and
                     max(state.levels.values(), default=1) < 3)
    weak = before.weak or needs_starter
    before_rate = getattr(before, 'clear_rate', getattr(before.next_round, 'clear_rate', 1.))
    emergency = before.coverage < 1. or before_rate < .5
    best = ShopDecision("leave", reason="no affordable improvement justifies its cost")
    best_survival = False
    best_progress = 0.
    for index, item in enumerate(items):
        kind = joker_name(item.kind)
        if kind == 'voucher' or kind.endswith('_pack'):
            from shop_extras import extra_is_worth_considering
            if not extra_is_worth_considering(item, state, before):
                continue
        if kind in ('planet', 'tarot', 'spectral') and state.consumables_held >= state.consumable_slots:
            continue
        joker = item.joker or Joker(item.name)
        if kind == 'joker' and joker.key == 'red_card' and joker.value <= 0 and not joker.edition:
            # The controller does not intentionally skip packs to feed Red
            # Card. A changed sampled draw route cannot turn +0 into an upgrade.
            continue
        full = kind == 'joker' and len(state.jokers) >= state.total_joker_slots + (joker.edition == 'negative')
        options = [i for i, old in enumerate(state.jokers) if not old.get('eternal')]
        # Stencil, hand-size penalties and resource tradeoffs can justify a
        # replacement while a physical slot is still empty.
        replacements = options if full else [-1] + (options if kind == 'joker' else [])
        for replacement in replacements:
            after = _purchase_state(item, state, replacement)
            after = _shop_round_state(after)
            if after.money < 0 or len(after.jokers) > after.total_joker_slots:
                continue
            projected = shop_forecast(after) if (kind in ('joker', 'planet', 'voucher')
                or (kind.endswith('_pack') and (state.has('bull') or state.has('bootstraps')))) else before
            after_score = projected.next_round.safe_score
            gain = 22 * log2((after_score + 20) / (baseline + 20))
            gain += 8 * log2((projected.next_round.mean_score + 20) /
                             (before.next_round.mean_score + 20))
            if before.boss_round and projected.boss_round:
                gain += 15 * log2((projected.boss_round.safe_score + 20) /
                                  (before.boss_round.safe_score + 20))
            gain += 10 * log2((projected.sustainable_score + 20) /
                              (before.sustainable_score + 20))
            immediate_gain = gain
            future_before = getattr(before, 'development_score', None) or before.sustainable_score
            future_after = getattr(projected, 'development_score', None) or projected.sustainable_score
            development_weight = 4 if emergency else 18
            if (kind == 'joker' and joker.key in GROWTH_JOKERS and state.ante <= 3
                    and not emergency and projected.coverage >= 1.25
                    and projected.clear_rate >= .85 and future_after > future_before
                    and dict(projected.next_round.growth_per_round).get(joker.key, 0) > 0):
                # The early Square offer kept >7x coverage, but lost to a
                # minor Tarot because its growth projection had too little weight.
                # Pay for demonstrated growth while immediate survival has
                # headroom; do not invent future offers or activations.
                development_weight = 28
            gain += development_weight * log2((future_after + 20) / (future_before + 20))
            after_rate = getattr(projected, 'clear_rate', getattr(projected.next_round, 'clear_rate', 1.))
            gain += 24 * (after_rate - before_rate)
            gain = max(-100, min(120, gain))
            if kind == 'planet':
                category = PLANET_HAND.get(joker_name(item.name), '')
                frequency = dict(before.next_round.frequencies).get(category, 0)
                # Mars in game 4 changed neither the simulated hand nor the
                # round. Do not pay for unused levels because their base is big.
                if gain < .5 and frequency < .1 and not (state.has('constellation') or state.observatory):
                    continue
                if (state.ante <= 2 and after.money < 5 and before.coverage >= 1.25
                        and 'low_mult' in before.profile.weaknesses
                        and projected.coverage < before.coverage + .35
                        and after_score < baseline * 1.2
                        and not (state.has('constellation') or state.observatory)):
                    # Keep the scoring-Joker/interest budget when a minor
                    # planet upgrade cannot repair the missing additive Mult.
                    continue
            if (replacement >= 0 and projected.coverage < before.coverage * .8
                    and before.coverage < 1.5):
                continue
            if replacement >= 0 and replacement_reduces_survival(before, projected):
                # Do not improve an easy blind at the expense of an already
                # lethal Boss or the engine after temporary support expires.
                continue
            prior = (0. if kind == 'joker' and joker.key in ECONOMY_JOKERS else _item_value(item, state))
            if kind.endswith('_pack'):
                from shop_extras import sealed_pack_value
                prior = sealed_pack_value(item, state, before)
            if replacement >= 0:
                if state.jokers[replacement].key not in ECONOMY_JOKERS:
                    prior -= _joker_purchase_value(state.jokers[replacement], state)
            if kind == 'joker':
                # Constellation at X1, Hallucination without packs, and income
                # alone cannot replace the immediate need to score enough.
                prior *= .35 if gain > 1 else (.15 if weak else .45)
                if joker.key in XMULT_JOKERS and 'no_multiplier_or_retrigger' in before.profile.weaknesses and gain > 1:
                    prior += min(8, max(0, state.ante - 2) * 2)
                elif joker.key in XMULT_JOKERS and state.ante >= 3 and immediate_gain > 4:
                    prior += min(6., immediate_gain * .15)
                if joker.key in GROWTH_JOKERS and gain >= 0:
                    prior += growth_opportunity(joker, after, projected)
                if replacement >= 0 and not emergency:
                    lost = state.jokers[replacement]
                    if lost.key in GROWTH_JOKERS:
                        # Incoming growth already receives an opportunity
                        # value. Selling supported growth must pay its cost too.
                        prior -= 1.5 * growth_opportunity(lost, state, before)
                if (emergency and projected.coverage <= before.coverage + .01
                        and after_rate <= before_rate and immediate_gain < .5):
                    continue  # an income/growth label does not rescue a lethal blind
                if joker.key == 'riff_raff' and state.ante <= 3:
                    # The produced Commons are unknown. Value the free-slot
                    # opportunity without inserting imaginary scoring Jokers
                    # into the round forecast.
                    free = max(0, after.total_joker_slots - len(after.jokers))
                    prior += min(2, free) * (4 + max(0, 4 - state.ante))
                economic_gain = economy_value(after, projected) - economy_before
                scoring_source = (joker.edition in ('foil', 'holographic', 'polychrome')
                    or replacement < 0 and state.has('abstract_joker')
                    or state.has('swashbuckler') and joker.sell_value >
                       (state.jokers[replacement].sell_value if replacement >= 0 else 0))
                resource_gain = after.hand_size > state.hand_size or after.hands_left > state.hands_left
                if (joker.key in ECONOMY_JOKERS and state.ante >= 3
                        and (before.coverage < 1.5 or before_rate < .85)
                        and not (scoring_source or resource_gain)):
                    # A pure income Joker cannot create immediate chips or
                    # Mult by changing the simulated discard route. Require a
                    # real scoring/resource source before trusting that gain.
                    continue
                if (joker.key in ECONOMY_JOKERS and state.ante >= 3
                        and (before.coverage < 1.5 or before_rate < .85)
                        and immediate_gain < .5
                        and projected.coverage <= before.coverage + .03):
                    continue  # an investment must not occupy the rescue budget
                prior += economic_gain * (.25 if emergency else 1.)
            if kind == 'planet':
                prior *= .35
            if kind in ('tarot', 'spectral'):
                prior *= .5 if weak else .8
                # The controller cannot use random destructive Spectrals or
                # the Fool safely; buying them would occupy a slot indefinitely.
                if joker_name(item.name) in ('the_fool', 'the_emperor', 'the_high_priestess',
                    'ouija', 'ectoplasm', 'immolate',
                    'familiar', 'grim', 'incantation', 'sigil', 'aura', 'wraith', 'the_wheel_of_fortune'):
                    continue
                if joker_name(item.name) in ('ankh', 'hex') and (
                        len(state.jokers) != 1 or state.jokers[0].get('eternal')):
                    continue
                if joker_name(item.name) == 'hex' and state.jokers[0].edition:
                    continue
                if joker_name(item.name) == 'ankh' and state.total_joker_slots < 2 and state.jokers[0].edition != 'negative':
                    continue
                if needs_starter and after.money < 12 and joker_name(item.name) not in ('judgement', 'the_soul'):
                    continue
            if kind == 'tarot' and weak and joker_name(item.name) in ('the_hermit', 'temperance'):
                prior *= .6
            cash_cost = _money_value(state.money, state.ante, state) - _money_value(after.money, state.ante, state)
            net = gain * (1.25 if weak else 1) + prior - cash_cost * (1 if weak else 1.3)
            repairs_coverage = before.coverage < 1 <= projected.coverage
            progress = max(0., projected.coverage - before.coverage) + max(0., after_rate - before_rate)
            priority = progress if emergency else 0.
            if ((net > 0 or (emergency and progress >= .08))
                    and (repairs_coverage, priority, net) > (best_survival, best_progress, best.utility)):
                action = 'sell_then_buy' if replacement >= 0 else 'buy'
                best = ShopDecision(action, index, replacement, net,
                    f"{action} {item.name}; safe round {baseline:.0f}->{after_score:.0f}; "
                    f"next clear rate {before.next_round.clear_rate:.2f}->{projected.next_round.clear_rate:.2f}; "
                    f"coverage {before.coverage:.2f}->{projected.coverage:.2f}; "
                    f"sustainable {before.sustainable_score:.0f}->{projected.sustainable_score:.0f}; "
                    f"development estimate {future_before:.0f}->{future_after:.0f}; net {net:.1f}")
                best_survival = repairs_coverage
                best_progress = priority
    # Early failures had no functioning scoring Joker while the old reserve
    # prevented even one affordable reroll. Keep enough cash to buy an offer.
    reserve = (3 if len(state.jokers) < 2 else 10) if weak else state.interest_cash_cap
    if state.deck_type == 'Green Deck':
        reserve = min(reserve, 10)
    needs_headroom = before.coverage < 1.75 or before.sustainable_score < next_blind_target(state) * 1.3
    weaknesses = getattr(getattr(before, 'profile', None), 'weaknesses', ())
    needs_development = (3 <= state.ante < 8 and any(key in weaknesses for key in
                         ('low_chips', 'low_mult', 'no_scaling', 'no_multiplier_or_retrigger', 'unreliable_multiplier')))
    growth_pressure = (4 <= state.ante < 8 and before.coverage < 1.5
                       and any(k in weaknesses for k in ('no_scaling','no_multiplier_or_retrigger','unreliable_multiplier')))
    if needs_development and not weak and state.money >= 31:
        # Pay at most one interest tier to search for a missing engine part;
        # reserve the offered Joker's budget separately. Bull/Bootstraps keep
        # the stronger cash protection below.
        reserve = min(reserve, max(20, state.interest_cash_cap - 5))
    critical = emergency
    max_rerolls = 4 if critical and state.money >= 40 else (3 if needs_headroom and state.money >= 50 else 2)
    if critical and before.coverage < .85 and state.money >= 80:
        # Ante 7 left $118 with zero sampled clears solely at the old four
        # reroll cap. Bound a cash-rich rescue search at eight, keeping the
        # observed price ceiling and a separate purchase/cash-Joker reserve.
        max_rerolls = 8
    purchase_cash = 4 if not state.jokers else 6
    if not state.jokers and weak:
        reserve = 0
    if critical and not (state.has('bull') or state.has('bootstraps')):
        # A recorded Ante-6 run left with $17 and forecast coverage 0.78.
        # Saving $10 prevented a $6 reroll while still leaving purchase cash.
        # Spend that interest reserve when the next round is already lethal;
        # keep the bounded reroll limit and a separate budget for the offer.
        reserve = 0
    elif growth_pressure and not (state.has('bull') or state.has('bootstraps')):
        # Late recorded engines bought small unknown upgrades at $17 while
        # missing scaling/XMult. Keep the offer budget, but allow one search
        # before the next blind becomes outright lethal.
        reserve = min(reserve, 5)
    if state.has('bull') or state.has('bootstraps'):
        reserve = max(reserve, min(25, state.money))
    max_reroll_cost = min(20, max(11, state.money // 4)) if critical else 8
    marginal_pack = (best.action == 'buy' and items[best.item_index].kind in
                     ('celestial_pack', 'arcana_pack', 'standard_pack')
                     and (best.utility < 4 or state.ante >= 3 and needs_development
                          and (before.coverage < 1.35 or before_rate < .9)))
    if marginal_pack:
        pack_kind = items[best.item_index].kind
        marginal_pack = not (pack_kind == 'celestial_pack' and
                            (state.has('constellation') or 'telescope' in state.owned_vouchers)
                            or pack_kind == 'arcana_pack' and state.has('fortune_teller')
                            or pack_kind == 'standard_pack' and state.has('hologram'))
    # A cheap sealed pack with unknown contents used to consume the remaining
    # search budget while the late engine still had no scaling or multiplier.
    # Known scoring upgrades and supported pack engines retain priority.
    marginal_consumable = (emergency and best.action == 'buy' and best.utility < 4
                           and items[best.item_index].kind in ('tarot','spectral'))
    search_missing_part = ((marginal_pack and (needs_development or emergency))
                           or marginal_consumable) and not best_survival and best_progress < .08
    if ((best.action == 'leave' or search_missing_part) and reroll_cost is not None and 0 <= reroll_cost <= max_reroll_cost
            and state.shop_rerolls < max_rerolls and (weak or (needs_headroom and state.money >= 40)
                                                   or (needs_development and state.money >= 31)
                                                   or (growth_pressure and state.shop_rerolls < 1))
            and state.money >= reserve + reroll_cost + purchase_cash):
        best = ShopDecision('reroll', utility=1,
                            reason=f'round/Boss coverage {before.coverage:.2f}; '
                                   f'missing parts {weaknesses}; '
                                   f'growth/XMult support needed {needs_development}; '
                                   f'preemptive growth search {growth_pressure}; '
                                   f'keep purchase cash; limit {max_rerolls} rerolls')
    return best
