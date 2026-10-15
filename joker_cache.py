"""Reuse verified Joker tooltips while checking the visible roster each time.

Unknown effects stay on the full observation path. Counters are never advanced
from predicted scores. All caches belong to one Session and are not persisted.
"""
from dataclasses import dataclass, replace

import cv2
import numpy as np

from perception import joker_slots
from strategy import _boss


# These tooltips contain no counter that changes within a blind. Some effects
# depend on hand/money/history; strategy.py computes those from observed state.
# Blind boundaries still refresh rank/suit targets, lifetime and Boss debuffs.
STABLE_WITHIN_BLIND = frozenset('''
joker greedy_joker lusty_joker wrathful_joker gluttonous_joker
jolly_joker zany_joker mad_joker crazy_joker droll_joker
sly_joker wily_joker clever_joker devious_joker crafty_joker
half_joker joker_stencil four_fingers mime credit_card
banner mystic_summit marble_joker 8_ball dusk raised_fist chaos_the_clown
fibonacci scary_face abstract_joker delayed_gratification hack pareidolia
gros_michel even_steven odd_todd scholar business_card supernova splash
blue_joker sixth_sense space_joker burglar blackboard flower_pot shortcut
golden_joker fortune_teller juggler drunkard acrobat sock_and_buskin
swashbuckler troubadour golden_ticket mr_bones hanging_chad
rough_gem bloodstone arrowhead onyx_agate photograph reserved_parking
mail_in_rebate to_the_moon ancient_joker walkie_talkie the_idol
the_duo the_trio the_family the_order the_tribe stuntman
invisible_joker satellite shoot_the_moon drivers_license cartomancer
astronomer burnt_joker bull bootstraps cavendish card_sharp smeared_joker
baseball_card blueprint brainstorm chicot triboulet
'''.split()) - {'fortune_teller', 'swashbuckler', 'drivers_license'}

# These targets/lifetimes change at cashout or at the next blind. The others
# still need a visible identity/edition/debuff check, but not a new tooltip.
ROUND_CHANGING = frozenset({'ancient_joker', 'the_idol', 'mail_in_rebate',
                           'gros_michel', 'cavendish', 'invisible_joker'})

# Verified triggers in the installed game's card.lua. These counters change
# through another action (or cashout), not merely by playing/discarding cards.
# Transactions and blind boundaries still invalidate them independently.
UNCHANGED_BY_HAND_ACTION = frozenset({'red_card', 'constellation', 'campfire',
    'flash_card', 'throwback', 'fortune_teller', 'to_do_list', 'egg', 'gift_card', 'rocket', 'popcorn'})
UNCHANGED_BY_DISCARD = frozenset({'square_joker', 'ride_the_bus', 'runner',
    'vampire', 'wee_joker', 'obelisk', 'lucky_cat', 'loyalty_card', 'ice_cream', 'seltzer'})


def round_token(state):
    return (state.ante, state.blind_index, state.boss.lower(), state.boss_disabled)


def roster_patch(image, count, index):
    # Interior art, excluding animated borders and badges. Keep an anchored
    # reference rather than accepting gradually drifting images as new truth.
    x = joker_slots(count)[index]
    # Include the lower emblem: several scoring Jokers share the same face
    # and are distinguished by the small hand symbol underneath it.
    return image[125:310, x - 55:x + 55].copy()


def same_slot_art(image, count, index, patch):
    if patch is None or not patch.size or float(np.std(patch)) < 8:
        return False
    x = joker_slots(count)[index]
    search = image[105:330, x - 75:x + 75]
    if search.shape[0] < patch.shape[0] or search.shape[1] < patch.shape[1]:
        return False
    # Balatro continuously tilts cards. Compensate that small affine motion,
    # retaining strict similarity rather than lowering the identity threshold.
    search = cv2.GaussianBlur(search, (3, 3), 0)
    height, width = patch.shape[:2]
    best = (-1., None, None)
    for angle in (0, -2, 2, -4, 4):
        for scale in (1., .97, 1.03):
            transform = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
            candidate = cv2.warpAffine(patch, transform, (width, height),
                                       borderMode=cv2.BORDER_REPLICATE)
            candidate = cv2.GaussianBlur(candidate, (3, 3), 0)[8:-8, 8:-8]
            result = cv2.matchTemplate(search, candidate, cv2.TM_CCOEFF_NORMED)
            _, confidence, _, position = cv2.minMaxLoc(result)
            if confidence > best[0]:
                best = confidence, candidate, position
            if confidence < .97:
                continue
            # Correlation alone can hide a changed edition/debuff colour.
            px, py = position
            matched = search[py:py + candidate.shape[0], px:px + candidate.shape[1]]
            colour_error = float(np.abs(matched.astype(np.float32) - candidate).mean())
            if colour_error <= 12:
                return True
    return aligned_art(search, *best)


def aligned_art(search, confidence, candidate, position):
    """Resolve small CRT/tilt distortion, keeping strict final image proof."""
    if confidence < .85 or candidate is None:
        return False
    x, y = position
    h, w = candidate.shape[:2]
    tile = search[y:y + h, x:x + w]
    gray = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    try:
        _, matrix = cv2.findTransformECC(gray(candidate), gray(tile),
            np.eye(2, 3, dtype=np.float32), cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 25, 1e-4), None, 3)
    except cv2.error:
        return False
    # Alignment corrects small animation, not arbitrary shape differences.
    if (abs(float(matrix[0, 0]) - 1) > .08 or abs(float(matrix[1, 1]) - 1) > .08
            or abs(float(matrix[0, 1])) > .12 or abs(float(matrix[1, 0])) > .12
            or abs(float(matrix[0, 2])) > 6 or abs(float(matrix[1, 2])) > 6):
        return False
    aligned = cv2.warpAffine(tile, matrix, (w, h),
                            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                            borderMode=cv2.BORDER_REPLICATE)
    interior = (slice(5, -5), slice(5, -5))
    a, b = candidate[interior], aligned[interior]
    score = float(cv2.matchTemplate(b, a, cv2.TM_CCOEFF_NORMED)[0, 0])
    return score >= .97 and float(np.abs(a.astype(np.float32) - b).mean()) <= 12


@dataclass
class JokerCache:
    token: tuple
    limit: int
    jokers: tuple
    patches: tuple
    # None is the conservative default for older callers/tests. A freshly
    # collected roster is clean until a confirmed action dirties its effects.
    dirty: frozenset | None = None

    def compatible(self, state, count, limit):
        if (self.token != round_token(state) or self.limit != limit
                or len(self.jokers) != count):
            return False
        # Random disabling and shuffling can change the scoring engine without
        # changing its count. Always inspect these Bosses while active.
        if _boss(state) in ('crimson_heart', 'amber_acorn'):
            return False
        return True

    def verified_slots(self, image, state, count, limit):
        if not self.compatible(state, count, limit):
            return ()
        return tuple(same_slot_art(image, count, i, patch)
                     for i, patch in enumerate(self.patches))

    def reusable(self, image, state, count, limit):
        slots = self.verified_slots(image, state, count, limit)
        return self.compatible(state, count, limit) and all(slots)

    def reconciled_slots(self, image, state, count):
        """Map current slots to proven old cards, including additions/removals.

        Count, slot number and transaction predictions never prove identity.
        Ambiguous visually identical copies with different effects are reread.
        """
        if _boss(state) in ('crimson_heart', 'amber_acorn'):
            return (None,) * count
        matches = []
        for index in range(count):
            candidates = [old for old, patch in enumerate(self.patches)
                          if same_slot_art(image, count, index, patch)]
            if not candidates or len({self.jokers[i] for i in candidates}) != 1:
                matches.append(None)
            else:
                matches.append(candidates[0])
        # One reference cannot authorize two new copies. Identical existing
        # copies can each supply one proof; an additional copy needs a tooltip.
        for old in set(i for i in matches if i is not None):
            slots = [i for i, value in enumerate(matches) if value == old]
            available = sum(j == self.jokers[old] for j in self.jokers)
            if len(slots) > available:
                for index in slots:
                    matches[index] = None
        return tuple(matches)

    def needs_tooltip(self, old, state):
        joker = self.jokers[old]
        if self.dirty is not None and joker.key in self.dirty:
            return True
        if self.token != round_token(state):
            if self.token[2:] != round_token(state)[2:]:
                return True
            return (joker.key not in STABLE_WITHIN_BLIND or joker.key in ROUND_CHANGING
                    or bool(joker.get('perishable')) or any(j.key == 'gift_card' for j in self.jokers))
        if self.dirty is None:
            return joker.key not in STABLE_WITHIN_BLIND
        return joker.key in self.dirty


def dirty_jokers(session, keys=None, *, round_end=False):
    cache = getattr(session, 'joker_cache', None)
    if cache is None:
        return
    if keys is None:
        keys = {j.key for j in cache.jokers if j.key not in STABLE_WITHIN_BLIND}
    if round_end:
        keys = set(keys) | ROUND_CHANGING
        keys |= {j.key for j in cache.jokers if j.get('perishable')}
        if any(j.key == 'gift_card' for j in cache.jokers):
            keys |= {j.key for j in cache.jokers}
    previous = cache.dirty
    if previous is None:
        previous = {j.key for j in cache.jokers if j.key not in STABLE_WITHIN_BLIND}
    session.joker_cache = replace(cache, dirty=frozenset(previous) | frozenset(keys))


def dirty_hand_action(session, kind):
    """Refresh affected counters, preserving already-dirty/unknown effects."""
    cache = getattr(session, 'joker_cache', None)
    if cache is None:
        return
    unchanged = UNCHANGED_BY_HAND_ACTION
    if kind == 'discard':
        unchanged |= UNCHANGED_BY_DISCARD
    keys = {j.key for j in cache.jokers
            if j.key not in STABLE_WITHIN_BLIND and j.key not in unchanged}
    dirty_jokers(session, keys)


def cached_joker(joker, state):
    # This value is purely a function of the freshly observed cash counter;
    # score_hand uses state.money as well. Preserve edition and other metadata.
    if joker.key == 'bull':
        return replace(joker, value=2 * state.money)
    if joker.key == 'bootstraps':
        return replace(joker, value=2 * (state.money // 5))
    if joker.key == 'blue_joker':
        remaining = state.draw_pile_size if state.draw_pile_size >= 0 else state.deck_size - len(state.seen_cards)
        return replace(joker, value=2 * max(0, remaining))
    return joker
