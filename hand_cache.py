"""Cache tooltip-proven cards that physically remain in the current blind.

This is deliberately not a rank/suit deck catalogue: two copies may have
different seals or permanent chips hidden by overlap. Only confirmed retained
cards can be reused, after their visible appearance is checked again.
"""
from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np

from joker_cache import round_token, aligned_art
from strategy import Card
from vision import hand_slots


def card_patch(image, position):
    x, y = position
    # Stay inside the exposed left strip even with a thirteen-card hand.
    return image[y + 5:y + 140, x + 5:x + 49].copy()


def same_card_art(image, x, patch):
    if patch.shape[:2] != (135, 44) or float(np.std(patch)) < 8:
        return False
    search = cv2.GaussianBlur(image[655:975, max(0, x - 35):x + 90], (3, 3), 0)
    h, w = patch.shape[:2]
    best = (-1., None, None)
    for angle in (0, -2, 2, -4, 4, -6, 6, -8, 8):
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        candidate = cv2.warpAffine(patch, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
        candidate = cv2.GaussianBlur(candidate, (3, 3), 0)[8:-8, 6:-6]
        scores = cv2.matchTemplate(search, candidate, cv2.TM_CCOEFF_NORMED)
        _, confidence, _, (px, py) = cv2.minMaxLoc(scores)
        if confidence > best[0]:
            best = confidence, candidate, (px, py)
        if confidence < .97:
            continue
        matched = search[py:py + candidate.shape[0], px:px + candidate.shape[1]]
        if float(np.abs(matched.astype(np.float32) - candidate).mean()) <= 12:
            return True
    return aligned_art(search, *best)


@dataclass
class HandCache:
    token: tuple
    cards: tuple
    patches: tuple

    def retain(self, cards):
        remaining = Counter(cards)
        indices = []
        for i, card in enumerate(self.cards):
            if remaining[card]:
                indices.append(i)
                remaining[card] -= 1
        return HandCache(self.token, tuple(self.cards[i] for i in indices),
                         tuple(self.patches[i] for i in indices))

    def verified_identities(self, image, state, count, limit):
        if self.token != round_token(state):
            return [None] * count
        if 948.5 / max(1, max(count, limit) - 1) < 64:
            # Extremely wide hands obscure even the proof strip. Read tooltips
            # rather than matching a crop belonging partly to its neighbour.
            return [None] * count
        # An incoming duplicate can share a retained card's visible corner yet
        # hide a different seal/chip bonus. Only unique identities in a known,
        # reconciled deck can prove that this is the same physical card.
        if not state.deck_cards or len(state.deck_cards) != state.deck_size:
            return [None] * count
        identities_in_deck = Counter(c.identity for c in state.deck_cards)
        matches = []
        for x in hand_slots(count, limit):
            candidates = [i for i, (card, patch) in enumerate(zip(self.cards, self.patches))
                          if not card.face_down and not card.is_stone
                          and identities_in_deck[card.identity] == 1 and same_card_art(image, x, patch)]
            # Similar-looking copies with different hidden metadata are not
            # interchangeable, even when their rank and suit are identical.
            identities = {self.cards[i] for i in candidates}
            matches.append(next(iter(identities)) if len(identities) == 1 else None)
        allowed = Counter(self.cards)
        used = Counter(c for c in matches if c is not None)
        return [c if c is None or used[c] <= allowed[c] else None for c in matches]


def finish_hand_action(session, pending):
    kind, details = pending['kind'], pending.get('details', {})
    cache = getattr(session, 'hand_cache', None)
    if cache is None:
        return
    if kind in ('play', 'discard'):
        retained = details.get('retained_cards')
        session.hand_cache = (cache.retain(Card(**c) for c in retained)
                              if retained is not None else None)
    elif kind == 'consume':
        # Planets/money do not alter retained playing cards. All card-modifying
        # or unfamiliar consumables start fresh, including same-count effects.
        if details.get('consumable_kind') != 'planet' and details.get('name') not in ('The Hermit', 'Temperance'):
            session.hand_cache = None
    elif kind in ('pick_pack', 'open_pack', 'skip_pack'):
        session.hand_cache = None
