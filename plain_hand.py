"""Directly identify plain cards in the calibrated eight-card layout.

The full exposed face proves rank/suit and ordinary appearance. Deck memory
proves absence of hidden seals/permanent chips; appearance alone cannot.
"""
from collections import Counter
from functools import lru_cache
import io
import zipfile

import cv2
import numpy as np
from PIL import Image

from strategy import Card, _boss
from vision import ROOT, SUITS, _atlas, hand_slots


CARD_CHANGING_BOSSES = frozenset({'goad', 'club', 'head', 'window', 'plant',
    'pillar', 'verdant_leaf', 'house', 'wheel', 'fish', 'mark'})


@lru_cache(maxsize=1)
def plain_faces():
    front = _atlas()
    with zipfile.ZipFile(ROOT / 'Balatro' / 'Balatro.exe') as game:
        centers = Image.open(io.BytesIO(game.read('resources/textures/2x/Enhancers.png'))).convert('RGBA')
    # The installed game's c_base uses Enhancers at (1, 0).
    base = centers.crop((142, 0, 284, 190))
    faces = {}
    for row, suit in enumerate(SUITS):
        for rank in range(2, 15):
            face = base.copy()
            face.alpha_composite(front.crop(((rank - 2) * 142, row * 190,
                                            (rank - 1) * 142, (row + 1) * 190)))
            faces[Card(rank, suit)] = cv2.cvtColor(np.array(face.convert('RGB')), cv2.COLOR_RGB2BGR)[6:155, 6:82]
    return faces


@lru_cache(maxsize=4096)
def face_variant(card, scale, angle):
    face = cv2.resize(plain_faces()[card], None, fx=scale, fy=scale)
    h, w = face.shape[:2]
    face = cv2.warpAffine(face, cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1),
                         (w, h), borderMode=cv2.BORDER_REPLICATE)
    return cv2.GaussianBlur(face, (5, 5), 1)


def best_face(search, card, *, refine=False):
    best = (-1., float('inf'))
    best_tile, best_template = None, None
    scales = (1.175, 1.2, 1.225) if refine else (1.2,)
    angles = (0, -2, 2, -4, 4, -6, 6, -8, 8) if refine else (0, -4, 4)
    for scale in scales:
        for angle in angles:
            template = face_variant(card, scale, angle)
            h, w = template.shape[:2]
            _, score, _, (x, y) = cv2.minMaxLoc(cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED))
            if score > best[0]:
                actual = search[y:y + h, x:x + w]
                best = score, float(np.abs(actual.astype(np.float32) - template).mean())
                best_tile, best_template = actual, template
    if refine and best[0] >= .87:
        # The CRT bends different parts of a card by slightly different
        # amounts. Fit a bounded affine motion for these three contenders;
        # evaluate both the winner and its competitors under the same rules.
        gray = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        try:
            _, matrix = cv2.findTransformECC(gray(best_template), gray(best_tile),
                np.eye(2, 3, dtype=np.float32), cv2.MOTION_AFFINE,
                (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 25, 1e-4), None, 3)
            if (abs(float(matrix[0, 0]) - 1) <= .08 and abs(float(matrix[1, 1]) - 1) <= .08
                    and abs(float(matrix[0, 1])) <= .12 and abs(float(matrix[1, 0])) <= .12
                    and abs(float(matrix[0, 2])) <= 8 and abs(float(matrix[1, 2])) <= 8):
                h, w = best_template.shape[:2]
                aligned = cv2.warpAffine(best_tile, matrix, (w, h),
                    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
                a, b = best_template[5:-5, 5:-5], aligned[5:-5, 5:-5]
                score = float(cv2.matchTemplate(b, a, cv2.TM_CCOEFF_NORMED)[0, 0])
                if score > best[0]:
                    best = score, float(np.abs(a.astype(np.float32) - b).mean())
        except cv2.error:
            pass
    return best


def plain_deck_identities(state, count, limit):
    if (count != 8 or limit != 8 or not state.deck_cards
            or len(state.deck_cards) != state.deck_size or _boss(state) in CARD_CHANGING_BOSSES):
        return {}
    counts = Counter(c.identity for c in state.deck_cards)
    uncertain = set(state.unverified_chip_cards)
    return {c.identity: c for c in state.deck_cards
            if counts[c.identity] == 1 and c.identity not in uncertain
            and c == Card(c.rank, c.suit) and c in plain_faces()}


def read_plain_hand(image, state, count, limit, *, skip_slots=()):
    """Return (identity, face confidence) or None for each slot. No UI input."""
    allowed = plain_deck_identities(state, count, limit)
    result = [None] * count
    if not allowed or image.shape[:2] != (1191, 2064):
        return result
    for index, x in enumerate(hand_slots(count, limit)):
        if index in skip_slots:
            continue
        search = cv2.GaussianBlur(image[655:965, x - 20:x + 125], (5, 5), 1)
        coarse = sorted(((best_face(search, card)[0], card) for card in plain_faces()),
                        key=lambda item: item[0], reverse=True)
        if coarse[0][0] < .85:
            continue
        refined = sorted(((best_face(search, card, refine=True), card) for _, card in coarse[:3]),
                         key=lambda item: item[0][0], reverse=True)
        (score, error), card = refined[0]
        competitor = max(refined[1][0][0], coarse[3][0] + .025)
        if score >= .94 and score - competitor >= .055 and error <= 24 and card.identity in allowed:
            result[index] = card, score
    # No template proof can make a unique physical card appear twice.
    counts = Counter(v[0].identity for v in result if v is not None)
    return [v if v is None or counts[v[0].identity] == 1 else None for v in result]


def mark_unverified_chips(state, played):
    """A confirmed Hiker play can change invisible permanent chip values."""
    if not (state.has('hiker') or any(j.key == 'unknown_hidden_joker' for j in state.jokers)):
        return state
    from dataclasses import replace
    identities = set(state.unverified_chip_cards)
    # Mark all played cards, conservatively including non-scoring/debuffed
    # ones. Never increment an unobserved chip value from a score prediction.
    identities.update(c.identity for c in played if not c.face_down)
    if any(c.face_down for c in played):
        identities.update(c.identity for c in state.deck_cards)
    return replace(state, unverified_chip_cards=tuple(sorted(identities)))
