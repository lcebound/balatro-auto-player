"""Read ordinary card identities from Balatro's visible card corners."""

from __future__ import annotations

import io
import zipfile
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from strategy import Card
from recovery import ObservationError


ROOT = Path(__file__).resolve().parent
SUITS = "HCDS"
RANKS = list(range(2, 15))


def _atlas() -> Image.Image:
    # The atlas belongs to the user's installed copy; no game data is bundled.
    with zipfile.ZipFile(ROOT / "Balatro" / "Balatro.exe") as game:
        return Image.open(io.BytesIO(game.read("resources/textures/2x/8BitDeck.png"))).convert("RGBA")


@lru_cache(maxsize=1)
def card_templates() -> dict[Card, np.ndarray]:
    atlas = _atlas()
    templates = {}
    for row, suit in enumerate(SUITS):
        for col, rank in enumerate(RANKS):
            corner = atlas.crop((col * 142, row * 190, col * 142 + 30, row * 190 + 35))
            white = Image.new("RGBA", corner.size, "white")
            white.alpha_composite(corner)
            templates[Card(rank, suit)] = cv2.cvtColor(np.array(white.convert("RGB")), cv2.COLOR_RGB2BGR)
    return templates


@lru_cache(maxsize=1)
def full_card_templates() -> dict[Card, np.ndarray]:
    atlas = _atlas()
    templates = {}
    for row, suit in enumerate(SUITS):
        for col, rank in enumerate(RANKS):
            corner = atlas.crop((col * 142, row * 190, col * 142 + 35, row * 190 + 55))
            white = Image.new("RGBA", corner.size, "white")
            white.alpha_composite(corner)
            templates[Card(rank, suit)] = cv2.cvtColor(np.array(white.convert("RGB")), cv2.COLOR_RGB2BGR)
    return templates


def match_at(
    image: np.ndarray,
    x: int,
    y: int,
    radius: int = 20,
    templates: dict[Card, np.ndarray] | None = None,
    vertical: int = 70,
    angles: tuple[int, ...] = (0,),
) -> list[tuple[float, Card, float, tuple[int, int]]]:
    """Rank card templates around one visible top-left corner."""
    x0, y0 = max(0, x - radius), max(0, y - vertical)
    roi = image[y0:y + vertical + 100, x0:x + 90]
    candidates = []
    for card, template in (templates or card_templates()).items():
        best = (-1.0, 0.0, (0, 0))
        for scale in (0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8):
            for angle in angles:
                resized = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                if angle:
                    h, w = resized.shape[:2]
                    resized = cv2.warpAffine(resized, cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1),
                                             (w, h), borderMode=cv2.BORDER_REPLICATE)
                if resized.shape[0] > roi.shape[0] or resized.shape[1] > roi.shape[1]:
                    continue
                result = cv2.matchTemplate(roi, resized, cv2.TM_CCOEFF_NORMED)
                left = max(0, x - 15 - x0)
                right = min(result.shape[1], x + 16 - x0)
                top = max(0, y - vertical - y0)
                bottom = min(result.shape[0], y + vertical + 1 - y0)
                if right <= left or bottom <= top:
                    continue
                _, score, _, location = cv2.minMaxLoc(result[top:bottom, left:right])
                location = (location[0] + left, location[1] + top)
                if score > best[0]:
                    best = (score, scale, location)
        candidates.append((best[0], card, best[1], best[2]))
    return sorted(candidates, reverse=True, key=lambda item: item[0])


# Calibrated from the fixed 2064x1191 window after the tutorial. These are
# approximate top-left positions; template matching refines each position.
def hand_slots(count: int, limit: int | None = None) -> list[int]:
    # Balatro spreads a full hand over the same area at every hand limit.
    # A partially filled hand retains spacing for its observed capacity.
    max_cards = max(count, limit if limit is not None else count)
    spacing = 948.5 / max(1, max_cards - 1)
    return [round(1034 + (index - (count - 1) / 2) * spacing) for index in range(count)]


@lru_cache(maxsize=2)
def _stone_position_templates(fine=False):
    """Mask rotation padding and suppress scanlines without discarding cracks."""
    source = next(t for c, t in special_corner_templates().items() if c.is_stone)
    source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    variants = []
    for scale in (.9, 1., 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8):
        resized = cv2.resize(source, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        resized = cv2.GaussianBlur(resized, (5, 5), 1.2)
        h, w = resized.shape
        interior = np.zeros((h, w), np.uint8)
        interior[8:-8, 8:-8] = 255
        angles = (-7, -5, -3, -1, 1, 3, 5, 7) if fine else (0, -8, -6, -4, -2, 2, 4, 6, 8)
        for angle in angles:
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
            rw, rh = int(h * sin + w * cos), int(h * cos + w * sin)
            matrix[0, 2] += (rw - w) / 2
            matrix[1, 2] += (rh - h) / 2
            rotated = cv2.warpAffine(resized, matrix, (rw, rh))
            mask = cv2.warpAffine(interior, matrix, (rw, rh), flags=cv2.INTER_NEAREST)
            variants.append((rotated, mask, matrix[:, 2]))
    return variants


def stone_position(image, x):
    """Locate Stone texture; discovery additionally requires >= .85 confidence."""
    x0, y0 = max(0, x - 30), 655
    roi = image[y0:895, x0:x + 100]
    gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (5, 5), 1.2)
    best = (-1., (x, 725))
    def search(variants):
        nonlocal best
        for template, mask, offset in variants:
            h, w = template.shape
            if h > gray.shape[0] or w > gray.shape[1]:
                continue
            scores = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED, mask=mask)
            scores[~np.isfinite(scores)] = -1
            # Constrain the transformed corner to this slot, rather than allowing
            # a match on the adjacent card's exposed strip.
            left = max(0, int(np.ceil(x - 25 - x0 - offset[0])))
            right = min(scores.shape[1], int(np.floor(x + 25 - x0 - offset[0])) + 1)
            if right <= left:
                continue
            _, score, _, location = cv2.minMaxLoc(scores[:, left:right])
            corner = (round(x0 + left + location[0] + offset[0]),
                      round(y0 + location[1] + offset[1]))
            if score > best[0]:
                best = score, corner

    search(_stone_position_templates())
    # Leftmost tilted Stone cards can fall between the two-degree samples.
    # Refine a plausible texture match; keep both acceptance thresholds and
    # the original location for an already strong match unchanged.
    if .75 <= best[0] < .85:
        search(_stone_position_templates(fine=True))
    if best[0] < .75:
        raise ObservationError(f'Stone corner at {x} is not verified ({best[0]:.2f}); recollect before selecting')
    return best


def read_hand(image: np.ndarray, count: int | None = None, *, identities=None,
              allow_uncertain=False) -> list[tuple[Card, float, tuple[int, int]]]:
    if image.shape[:2] != (1191, 2064):
        raise ObservationError(f"Unsupported screenshot size {image.shape[:2]}")
    from perception import hand_counter
    observed_count, limit = hand_counter(image)
    if count is not None and count != observed_count:
        raise ObservationError(f'Hand changed from {count} to {observed_count}; recollect before selecting')
    count = observed_count
    if not 1 <= count <= 30:
        raise ObservationError(f"No playable cards: visible hand count {count}")
    result = []
    for index, x in enumerate(hand_slots(count, limit)):
            identity_hint = identities[index] if identities is not None else None
            if identity_hint is None:
                backs = {c: template for c, template in special_corner_templates().items() if c.face_down}
                back_score, back, _, position = match_at(image, x, 725, radius=30, templates=backs,
                                                         angles=(0, -8, -4, 4, 8))[0]
                if back_score >= .65:
                    result.append((back, back_score, (x - 30 + position[0], 655 + position[1])))
                    continue
            if identity_hint is not None and identity_hint.face_down:
                backs = {c: template for c, template in special_corner_templates().items() if c.face_down}
                score, card, _, position = match_at(image, x, 725, radius=30, templates=backs,
                                                    angles=(0, -8, -4, 4, 8))[0]
                if score < .55:
                    raise ObservationError(f'Face-down card {index} changed; recollect before selecting')
                result.append((card, score, (x - 30 + position[0], 655 + position[1])))
                continue
            if identity_hint is not None and identity_hint.is_stone:
                # A proven rankless identity must never use ordinary rank/suit
                # templates: stone cracks can correlate with an internal pip,
                # which hides the real top edge and its selected lift.
                score, position = stone_position(image, x)
                result.append((identities[index], score, position))
                continue
            templates = full_card_templates()
            if identity_hint is not None:
                identity = Card(identity_hint.rank, identity_hint.suit)
                if identity in full_card_templates():
                    templates = {identity: full_card_templates()[identity]}
            # With a proven identity, compare rotations even when an upright
            # template looks plausible: large suit pips inside a selected card
            # can otherwise outscore its tilted corner and hide its raised state.
            angles = (0, -8, -4, 4, 8) if identity_hint is not None else (0,)
            score, rank_card, _, local = match_at(image, x, 725, radius=30, templates=templates, angles=angles)[0]
            if score < .53:
                score, rank_card, _, local = match_at(image, x, 725, radius=30, templates=templates, angles=(-8, -4, 4, 8))[0]
            if score < 0.53 or (identity_hint is not None and identity_hint.face_down):
                if identity_hint is None:
                    # A tilted new Stone Card has no rank/suit corner. The
                    # shader can defeat the raw-color special template even
                    # though the masked texture locator is unambiguous.
                    # This only seeds collection: modifiers still need the
                    # real tooltip, and selected state is verified separately.
                    try:
                        stone_score, stone_corner = stone_position(image, x)
                    except ObservationError:
                        pass
                    else:
                        if stone_score >= .85:
                            result.append((Card(0, '?', enhancement='stone', rankless=True),
                                           stone_score, stone_corner))
                            continue
                specials = match_at(image, x, 725, radius=30, templates=special_corner_templates(),
                                    angles=(0, -8, -4, 4, 8))[0]
                special_score, special, _, position = specials
                if special_score >= .55 and (special_score > score + .08 or score < .48):
                    result.append((special, special_score, (x - 30 + position[0], 655 + position[1])))
                    continue
                minimum = .40 if allow_uncertain or identities is not None else .48
                if score < minimum:
                    raise ObservationError(f"Incomplete hand: only {len(result)} of {count} cards recognized")
            card = Card(rank_card.rank, rank_card.suit)
            result.append((card, score, (x - 30 + local[0], 655 + local[1])))
    # Duplicated cards are legal after Death, Cryptid, DNA or Spectral effects.
    return result


@lru_cache(maxsize=1)
def special_corner_templates():
    with zipfile.ZipFile(ROOT / "Balatro" / "Balatro.exe") as game:
        atlas = Image.open(io.BytesIO(game.read("resources/textures/2x/Enhancers.png"))).convert("RGBA")
    templates = {}
    for card, col, row in [(Card(0, "?", enhancement="stone", rankless=True), 5, 0),
                           (Card(0, "?", face_down=True), 0, 0)]:
        # Stone has no rank or suit glyph. Its tiny rounded corner is too
        # featureless under Balatro's shader; include the stone cracks below
        # it while staying inside the exposed strip of overlapping cards.
        width, height = (45, 90) if card.rankless else (35, 55)
        corner = atlas.crop((col * 142, row * 190,
                             col * 142 + width, row * 190 + height))
        white = Image.new("RGBA", corner.size, "white")
        white.alpha_composite(corner)
        templates[card] = cv2.cvtColor(np.asarray(white.convert("RGB")), cv2.COLOR_RGB2BGR)
    return templates


UI_REGIONS = {
    "main": (530, 970, 680, 1050),
    "blind": (180, 210, 450, 330),
    "shop": (150, 180, 475, 320),
    "cashout": (880, 460, 1260, 540),
    "guide": (1800, 375, 1915, 460),
    "loss": (1040, 135, 1590, 250),
    "deck": (1000, 335, 1170, 390),
    "unlock": (895, 300, 1160, 460),
}


@lru_cache(maxsize=1)
def ui_templates() -> dict[str, np.ndarray]:
    return {
        name: cv2.cvtColor(
            np.array(Image.open(ROOT / "assets" / "ui" / f"{name}.png").convert("RGB")),
            cv2.COLOR_RGB2BGR,
        )
        for name in UI_REGIONS
    }


def scene_scores(image: np.ndarray) -> dict[str, float]:
    scores = {}
    for name, (x1, y1, x2, y2) in UI_REGIONS.items():
        # Balatro animates and slightly shifts panels between runs. Search a
        # small neighborhood around each distinctive UI label.
        margin = 35
        current = image[max(0, y1 - margin):min(image.shape[0], y2 + margin),
                        max(0, x1 - margin):min(image.shape[1], x2 + margin)]
        scores[name] = float(cv2.matchTemplate(
            current, ui_templates()[name], cv2.TM_CCOEFF_NORMED
        ).max())
    return scores


def classify_scene(image: np.ndarray) -> str:
    scores = scene_scores(image)
    # Collection unlocks use several different titles and card pictures.
    # Their shared orange Continue bar and white reward panel are stable.
    button = image[815:885, 700:1370]
    reward = image[480:765, 970:1330]
    orange = np.mean((button[:, :, 2] > 170) &
                     (button[:, :, 1] > 75) &
                     (button[:, :, 1] < 210) &
                     (button[:, :, 0] < 90))
    white = np.mean(reward.min(axis=2) > 210)
    if orange > 0.45 and white > 0.65:
        return "unlock_reward"
    # The new-run modal can cover a live shop. Its background SHOP sign is
    # still visible, so the modal must take precedence.
    if scores["deck"] > 0.80:
        return "deck"
    name = max(scores, key=scores.get)
    # Open boosters can retain the shop sign behind their card area. Ordinary
    # hand frames don't pay the additional pack OCR cost.
    if name == 'shop' or scores[name] <= .55:
        from shop_extras import read_pack_view, shop_ready
        if read_pack_view(image) is not None:
            return 'pack'
        if name == 'shop' and not shop_ready(image):
            return 'unknown'
    if scores[name] > 0.55:
        return name
    if playable_hand_visible(image):
        return "hand"
    try:
        read_hand(image, allow_uncertain=True)
        return "hand"
    except RuntimeError:
        from perception import read_text, victory_new_run
        if victory_new_run(read_text(image, (950, 125, 1700, 1100))) is not None:
            return "win"
        return "unknown"


def playable_hand_visible(image: np.ndarray) -> bool:
    """Prove the hand controls without identifying every playing card twice.

    This is scene evidence only. collect_cards still proves card identities,
    attributes and positions, and select_verified checks the actual selection.
    Nonstandard/covered controls retain the existing full-reader fallback.
    """
    from perception import hand_counter
    # Before any selection both buttons are grey, so OCR routinely emits no
    # labels. Match the proven complete control row from a real idle frame;
    # Sort Hand and both distinct action labels must independently agree.
    if not hand_controls_visible(image):
        return False
    try:
        count, limit = hand_counter(image)
    except ObservationError:
        return False
    return 1 <= count <= limit <= 30


@lru_cache(maxsize=1)
def hand_control_templates():
    regions = {'hand_play_control': (800, 1007, 990, 1075),
               'hand_discard_control': (1250, 1007, 1440, 1075),
               'hand_sort_control': (1030, 990, 1215, 1040)}
    return [(region, cv2.cvtColor(np.array(Image.open(
                ROOT / 'assets' / 'ui' / f'{name}.png').convert('RGB')), cv2.COLOR_RGB2BGR))
            for name, region in regions.items()]


def hand_controls_visible(image):
    for (x, y, right, bottom), template in hand_control_templates():
        search = image[y - 12:bottom + 12, x - 12:right + 12]
        if search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
            return False
        score = float(cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED).max())
        if score < .88:
            return False
    return True


def active_blind(image: np.ndarray) -> int:
    """Return 0, 1 or 2 for the orange Select button."""
    choices = []
    for left in (630, 1000, 1370):
        b, g, r = image[399:445, left:left + 30].mean(axis=(0, 1))
        choices.append((float(r - b), float(g)))
    winner = max(range(3), key=lambda i: choices[i][0])
    if choices[winner][0] < 130 or choices[winner][1] < 95:
        raise ObservationError(f"No active blind button: {choices}")
    return winner


def likely_joker_offer(image: np.ndarray) -> int | None:
    """Find a Joker in either shop slot using the installed game's sprite atlas."""
    scores = []
    for left, right in ((1045, 1245), (1240, 1450)):
        region = image[450:725, left:right]
        score = max(float(cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED).max())
                    for template in joker_templates())
        scores.append(score)
    best = max(range(2), key=lambda i: scores[i])
    return best if scores[best] >= 0.27 else None


@lru_cache(maxsize=1)
def joker_templates() -> tuple[np.ndarray, ...]:
    with zipfile.ZipFile(ROOT / "Balatro" / "Balatro.exe") as game:
        atlas = Image.open(io.BytesIO(game.read("resources/textures/2x/Jokers.png"))).convert("RGBA")
    templates = []
    for row in range(atlas.height // 190):
        for col in range(atlas.width // 142):
            tile = atlas.crop((col * 142, row * 190, (col + 1) * 142, (row + 1) * 190))
            background = Image.new("RGBA", tile.size, "white")
            background.alpha_composite(tile)
            rgb = np.array(background.convert("RGB"))
            if rgb.std() < 20:  # Empty atlas cells look like a perfect match.
                continue
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            templates.append(cv2.resize(bgr, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_NEAREST))
    return tuple(templates)
