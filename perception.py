"""Read visible state; uncertain observations never become invented values.

Templates come from the user's installed game. OCR models live in the venv.
This module does not connect to, inspect memory of, or operate the game.
"""
from __future__ import annotations

import io
import re
from difflib import SequenceMatcher
import zipfile
from dataclasses import dataclass, replace
from functools import lru_cache
from collections import OrderedDict, Counter
from hashlib import blake2b
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from strategy import Card, Joker, RunState, ShopItem, SUITS, joker_name, HAND_ALIASES
from recovery import ObservationError, HandPreviewConflict

ROOT = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def ocr_engine():
    from rapidocr import RapidOCR
    return RapidOCR(params={
        "EngineConfig.onnxruntime.intra_op_num_threads": 2,
        "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        "Global.log_level": "error",
    })


_TEXT_CACHE = OrderedDict()
_TEXT_CACHE_BYTES = 0


def victory_new_run(words):
    """Prove the completed victory panel, including split YOU / WIN! OCR."""
    def normalized(text):
        return re.sub(r'[^A-Z]', '', text.upper())

    reliable = [(normalized(text), point) for text, confidence, point in words
                if confidence > .85]
    header = sorted([(text, point) for text, point in reliable
                     if 950 <= point[0] <= 1700 and 125 <= point[1] <= 300],
                    key=lambda entry: entry[1][0])
    proven = any(''.join(text for text, point in header
                        if abs(point[1] - anchor[1]) <= 20) == 'YOUWIN'
                 for _, anchor in header)
    if not proven:
        return None
    # Tutorial text, a shop item mentioning a win, or a New Run modal must
    # never be mistaken for a finished game. Require independent panel rows.
    stats = {text for text, point in reliable
             if 950 <= point[0] <= 1350 and 300 <= point[1] <= 650}
    if len(stats & {'BESTHAND', 'MOSTPLAYEDHAND', 'CARDSPLAYED', 'CARDSDISCARDED'}) < 2:
        return None
    if not any(text == 'ENDLESSMODE' and 950 <= point[0] <= 1700
               and 850 <= point[1] <= 1100 for text, point in reliable):
        return None
    buttons = [point for text, point in reliable if text == 'NEWRUN'
               and 1300 <= point[0] <= 1700 and 580 <= point[1] <= 800]
    return buttons[0] if len(buttons) == 1 else None


def read_text(image, region=None, *, line=False):
    """Return text, confidence and center in canonical screenshot coordinates."""
    x, y = 0, 0
    if region:
        x, y, right, bottom = region
        image = image[y:bottom, x:right]
    if not image.size:
        return []
    engine = ocr_engine()
    pixels = image.tobytes()
    key = (id(engine), image.shape, image.dtype.str, x, y, line,
           blake2b(pixels, digest_size=16).digest())
    cached = _TEXT_CACHE.get(key)
    # Check exact bytes too: a hash never substitutes for pixel equality.
    if cached is not None and cached[0] == pixels and cached[2] is engine:
        _TEXT_CACHE.move_to_end(key)
        return list(cached[1])
    result = engine(image, use_det=not line, use_cls=False)
    if result.txts is None:
        return []
    boxes = result.boxes if getattr(result, "boxes", None) is not None else [None] * len(result.txts)
    words = [(text, float(score),
             (int(box.mean(axis=0)[0]) + x, int(box.mean(axis=0)[1]) + y)
             if box is not None else (x + image.shape[1] // 2, y + image.shape[0] // 2))
            for text, score, box in zip(result.txts, result.scores, boxes)]
    global _TEXT_CACHE_BYTES
    old = _TEXT_CACHE.pop(key, None)
    if old is not None:
        _TEXT_CACHE_BYTES -= len(old[0])
    _TEXT_CACHE[key] = (pixels, tuple(words), engine)
    _TEXT_CACHE_BYTES += len(pixels)
    while len(_TEXT_CACHE) > 64 or _TEXT_CACHE_BYTES > 32 * 1024 * 1024:
        _, old = _TEXT_CACHE.popitem(last=False)
        _TEXT_CACHE_BYTES -= len(old[0])
    return words


def read_offer_price(image, text, confidence, center):
    """Resolve suspicious prices from the existing frame, never a price list.

    Ordinary prices use the existing OCR result. A large value (including
    legitimate inflation) requires two agreeing, reliable narrow reads. No
    fresh screenshot, hover or wait is introduced.
    """
    def parse(label):
        match = re.fullmatch(r'\$(\d[\d,]*)', label.strip())
        return int(match[1].replace(',', '')) if match else None

    original = parse(text)
    if original is None or confidence <= .8:
        raise ObservationError(f'Unreadable shop price: {text!r}')
    if original < 40:
        return original
    x, y = map(int, center)
    height, width = image.shape[:2]
    values = []
    for half_width, half_height, line in ((65, 28, False), (65, 28, True), (55, 25, True)):
        region = (max(0, x - half_width), max(0, y - half_height),
                  min(width, x + half_width), min(height, y + half_height))
        reads = {parse(t) for t, score, (px, py) in read_text(image, region, line=line)
                 if score > .9 and abs(px - x) <= 30 and abs(py - y) <= 18
                 and parse(t) is not None}
        if len(reads) > 1:
            raise ObservationError(f'Conflicting local shop prices: {reads}')
        if reads:
            values.append(reads.pop())
        if len(set(values)) > 1 or confidence > .9 and values and values[-1] != original:
            raise ObservationError(f'Conflicting shop price: {text!r}, local={values}')
        if len(values) >= 2:
            return values[0]
    raise ObservationError(f'Cannot verify suspicious shop price: {text!r}, local={values}')


def numeric(text):
    """Balatro uses commas, scientific notation and coloured pixel digits."""
    cleaned = text.strip().replace(",", "").replace(" ", "").replace("$", "")
    cleaned = cleaned.translate(str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1"}))
    match = re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", cleaned)
    return float(cleaned) if match else None


def score_numeric(text):
    """Integer score displays may have a dot misread in place of a comma."""
    text = text.strip().lstrip('#').strip()
    # Pixel-font commas can be detected as an adjacent dot/comma pair, e.g.
    # the real 38,852 cashout read as 38.,852. Require exact 3-digit groups;
    # ordinary decimals and scientific notation retain numeric() semantics.
    if re.fullmatch(r'\d{1,3}(?:[.,]{1,2}\d{3})+', text):
        return float(text.replace('.', '').replace(',', ''))
    return numeric(text)


def hand_counter(image):
    for text, confidence, _ in read_text(image, (990, 935, 1250, 1005)):
        match = re.search(r"(\d+)\s*/\s*(\d+)", text)
        if match and confidence >= .75:
            count, limit = map(int, match.groups())
            if 0 <= count <= 30 and 1 <= limit <= 30:
                return count, limit
    raise ObservationError("Cannot read the visible hand counter; refusing to guess a smaller hand")


UI_NUMBERS = {
    "target_score": (320, 285, 495, 365),
    "round_score": (320, 470, 495, 540),
    "hands_left": (270, 770, 405, 900),
    "discards_left": (410, 770, 530, 900),
    "money": (295, 925, 495, 985),
    "ante": (280, 1010, 410, 1140),
}


def verified_resource_counter(image, key):
    """Corroborate a missing counter in its digit-only region.

    These are visible pixels, not an inferred starting hand/discard budget.
    All confident numeric views must agree; one read cannot fill the field.
    """
    region = {'hands_left': (295, 815, 392, 884),
              'discards_left': (430, 815, 520, 884)}[key]
    left, top, right, bottom = region
    crop = image[top:bottom, left:right]
    if not crop.size:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if float(np.std(gray)) < 4:
        return None
    contrast = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    views = (crop, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
             cv2.cvtColor(contrast, cv2.COLOR_GRAY2BGR))
    proofs = []
    for view in views:
        numbers = {int(text.strip()) for text, confidence, _ in read_text(view, line=True)
                   if confidence > .94 and re.fullmatch(r'\d{1,2}', text.strip())
                   and 0 <= int(text.strip()) <= 30}
        if len(numbers) > 1:
            return None
        proofs.extend(numbers)
    return proofs[0] if len(proofs) >= 2 and len(set(proofs)) == 1 else None


def read_numbers(image):
    values = {}
    # Keep enough surrounding pixels for detection of large pixel-font digits.
    for text, score, (x, y) in read_text(image, (105, 110, 545, 1150)):
        if score < .75:
            continue
        key = None
        if 350 < x < 520 and 305 < y < 370:
            key = "target_score"
        elif 350 < x < 520 and 465 < y < 535:
            key = "round_score"
        elif 285 < x < 395 and 820 < y < 895:
            key = "hands_left"
        elif 415 < x < 525 and 820 < y < 895:
            key = "discards_left"
        elif 290 < x < 520 and 925 < y < 985:
            key = "money"
        elif 285 < x < 400 and 1060 < y < 1125 and "/" in text:
            key = "ante"
            text = text.split("/")[0]
        value = score_numeric(text) if key in ('target_score', 'round_score') else numeric(text)
        if key and value is not None and value >= 0:
            if key == 'target_score' and value < 100:
                continue
            if key in ("hands_left", "discards_left") and value > 30:
                continue
            values[key] = int(value) if key not in ("target_score", "round_score") else value
    for text, confidence, _ in read_text(image, (1650, 1080, 1960, 1160)):
        match = re.search(r"(\d+)\s*/\s*(\d+)", text)
        if match and confidence > .75:
            values["draw_pile_size"], values["deck_size"] = map(int, match.groups())
    # Pixel-font 1/2/4 occasionally disappears in a large panel crop. Retry
    # its own panel rather than inventing a default or using a guessed counter.
    for key in ("target_score", "round_score", "hands_left", "discards_left", "money", "ante"):
        if key in values:
            continue
        for text, score, _ in read_text(image, UI_NUMBERS[key]):
            if key == "ante" and "/" in text:
                text = text.split("/")[0]
            value = score_numeric(text) if key in ('target_score', 'round_score') else numeric(text)
            limit = float('inf') if key in ('target_score', 'round_score') else (10000 if key == 'money' else 30)
            if value is not None and score > .8 and 0 <= value <= limit:
                if key == 'target_score' and value < 100:
                    continue
                values[key] = value if key in ('target_score', 'round_score') else int(value)
                break
    if 'round_score' not in values:
        # With no scored hands, the chip icon and zero can merge into '#0'.
        # A tight crop proves zero; nonzero values could be truncated here.
        for text, confidence, _ in read_text(image, (385, 460, 530, 550)):
            if numeric(text) == 0 and confidence > .9:
                values['round_score'] = 0.
                break
    for key in ('target_score', 'round_score'):
        if key in values:
            continue
        for text, confidence, _ in read_text(image, UI_NUMBERS[key], line=True):
            value = score_numeric(text)
            if value is not None and confidence > .85 and value >= (100 if key == 'target_score' else 0):
                values[key] = value
                break
    for key in ('hands_left', 'discards_left'):
        if key not in values:
            value = verified_resource_counter(image, key)
            if value is not None:
                values[key] = value
    return values


@dataclass(frozen=True)
class JokerDefinition:
    key: str
    name: str
    tile: tuple[int, int]
    cost: int
    rarity: str
    copyable: bool
    config: dict[str, float]
    soul_tile: tuple[int, int] | None = None
    internal_name: str = ""


@lru_cache(maxsize=1)
def joker_definitions():
    with zipfile.ZipFile(ROOT / "Balatro" / "Balatro.exe") as archive:
        source = archive.read("game.lua").decode("utf8")
        localization = archive.read("localization/en-us.lua").decode("utf8")
    displayed_names = {key: name for key, name in re.findall(
        r'''(?m)^\s*(j_\w+)\s*=\s*\{\s*\n\s*name\s*=\s*["']([^"\n]+?)["'],?\s*\n''', localization)}
    definitions = {}
    for line in source.splitlines():
        if not re.match(r"\s*j_\w+\s*=\s*\{", line):
            continue
        name = re.search(r'''name\s*=\s*["']([^"']+)["']''', line)
        tile = re.search(r"pos\s*=\s*\{\s*x\s*=\s*(\d+),\s*y\s*=\s*(\d+)", line)
        if not name or not tile:
            continue
        internal_id = re.match(r"\s*(j_\w+)", line)[1]
        displayed = displayed_names.get(internal_id, name[1])
        key = joker_name(displayed)
        config = line.split("config", 1)[-1].split("unlock_condition", 1)[0]
        numbers = {k.lower(): float(v) for k, v in re.findall(r"(\w+)\s*=\s*(-?\d+(?:\.\d+)?)", config)}
        rarity = re.search(r"rarity\s*=\s*(\d)", line)
        cost = re.search(r"\bcost\s*=\s*(\d+)", line)
        soul = re.search(r"soul_pos\s*=\s*\{\s*x\s*=\s*(\d+),\s*y\s*=\s*(\d+)", line)
        definitions[key] = JokerDefinition(key, displayed, tuple(map(int, tile.groups())),
            int(cost[1]) if cost else 0,
            {1: "common", 2: "uncommon", 3: "rare", 4: "legendary"}.get(int(rarity[1]) if rarity else 1, "common"),
            bool(re.search(r"blueprint_compat\s*=\s*true", line)), numbers,
            tuple(map(int, soul.groups())) if soul else None, name[1])
    return definitions


@lru_cache(maxsize=1)
def joker_templates():
    with zipfile.ZipFile(ROOT / "Balatro" / "Balatro.exe") as archive:
        atlas = Image.open(io.BytesIO(archive.read("resources/textures/2x/Jokers.png"))).convert("RGBA")
    templates = {}
    for key, definition in joker_definitions().items():
        col, row = definition.tile
        # Ignore the animated border and the vertical JOKER label.
        tile = atlas.crop((col * 142 + 30, row * 190 + 35, col * 142 + 125, row * 190 + 160))
        white = Image.new("RGBA", tile.size, "white")
        white.alpha_composite(tile)
        if definition.soul_tile:
            sx, sy = definition.soul_tile
            soul = atlas.crop((sx * 142 + 30, sy * 190 + 35, sx * 142 + 125, sy * 190 + 160))
            white.alpha_composite(soul)
        templates[key] = cv2.cvtColor(np.asarray(white.convert("RGB")), cv2.COLOR_RGB2BGR)
    return templates


def read_jokers(image, region=(535, 115, 1690, 320), count=None):
    """Match atlas art, preserving left-to-right order and confidence."""
    if count is None:
        return _match_art(image, region, joker_templates())
    observations = []
    for x in joker_slots(count):
        matches = _match_art(image, (max(535, x - 85), 110, min(1670, x + 85), 330), joker_templates())
        if matches:
            observations.append(max(matches, key=lambda entry: entry[0]))
        else:
            # Every physical slot is retained for tooltip identification.
            observations.append((0., "", (x, 220)))
    return observations


def joker_slots(count):
    if count == 1:
        return [1020]
    if count == 2:
        return [842, 1198]
    return [round(664 + i * 712 / max(1, count - 1)) for i in range(count)]


def identify_joker_text(lines, hint="", *, title=None, art_confidence=0):
    """All 150 names, with bounded OCR tolerance and ambiguity rejection."""
    definitions = joker_definitions()
    aliases = joker_aliases()
    exact_aliases = {aliases[_label_key(line)] for line in lines if _label_key(line) in aliases}
    specific = exact_aliases - {'joker'}
    if len(specific) == 1:
        return next(iter(specific)), 1.
    if len(specific) > 1:
        raise ObservationError('Multiple Joker titles in one tooltip')
    # The word Joker in a description or on the card border is not a title.
    # Hidden titles remain recognizable through their localized description.
    from item_identity import identify_description
    try:
        return identify_description(lines, 'joker')
    except RuntimeError:
        pass
    # Misprint's localized body is generated at runtime rather than stored in
    # the catalogue. Its complete short panel must agree with visible artwork.
    compact = [line.strip() for line in lines if not re.search(
        r'not\s+(?:t\s+)?discovered|^common$', line, re.I)]
    if (hint == 'misprint' and art_confidence >= .75 and len(compact) == 1
            and (m := re.fullmatch(r'\+\s*(\d+)\s+Mult', compact[0], re.I))
            and 0 <= int(m[1]) <= 23
            and any(line.strip().casefold() == 'common' for line in lines)):
        return hint, art_confidence
    hidden = any('not discovered' in line.lower() for line in lines)
    if title is not None and _label_key(title) == 'joker' and not hidden:
        fragments = {word.lower() for word in lines if ' ' not in word}
        other_title = any(key != 'joker' and key.endswith('_joker')
                          and key.removesuffix('_joker') in fragments for key in definitions)
        if not other_title:
            return 'joker', 1.
    if exact_aliases == {'joker'} and not hidden and ('Joker' in lines or len(lines) == 1):
        if len(lines) == 1 or any(re.search(r'\+\s*4\s+mult\b', line, re.I) for line in lines):
            return 'joker', 1.
    normalized = [joker_name(line) for line in lines]
    exact = {line for line in normalized if line in definitions}
    exact.discard('joker')
    if len(exact) == 1:
        return next(iter(exact)), 1.
    if hint in exact:
        return hint, 1.
    candidates = []
    for key in definitions:
        if key == 'joker':
            continue
        similarity = max((SequenceMatcher(None, key, line).ratio() for line in normalized), default=0)
        candidates.append((similarity, key))
    candidates.sort(reverse=True)
    best, key = candidates[0]
    if best >= .87 and best - candidates[1][0] >= .08:
        return key, best
    raise ObservationError("Joker tooltip name is unreadable or ambiguous")


def tooltip_labels(words, minimum=.72):
    """Keep OCR fragments and joined rows so split titles remain readable."""
    reliable = sorted((word for word in words if word[1] > minimum), key=lambda w: (w[2][1], w[2][0]))
    rows = []
    for word in reliable:
        row = next((row for row in rows if abs(row[0][2][1] - word[2][1]) < 14), None)
        if row is None:
            rows.append([word])
        else:
            row.append(word)
    labels = [word[0] for word in reliable]
    for row in rows:
        row.sort(key=lambda word: word[2][0])
        content = [word[0] for word in row if not re.fullmatch(r'\s*(?:\d+\s*/\s*\d+|\$[\d,.]+|#)\s*', word[0])]
        if len(content) > 1:
            labels.append(' '.join(content))
    return list(dict.fromkeys(labels))


def tooltip_rows(words, minimum=.72):
    """Join OCR fragments once per row, retaining their reading order."""
    rows = []
    for word in sorted((w for w in words if w[1] > minimum), key=lambda w: (w[2][1], w[2][0])):
        row = next((r for r in rows if abs(r[0][2][1] - word[2][1]) < 14), None)
        if row is None:
            rows.append([word])
        else:
            row.append(word)
    return [' '.join(w[0] for w in sorted(row, key=lambda w: w[2][0])) for row in rows]


def tooltip_title(words):
    candidates = [word for word in words if word[1] > .8
                  and not re.fullmatch(r'[\d\s/$#.,+×x-]+', word[0])
                  and word[0].strip().lower() not in ('next', 'round', 'reroll')]
    if not candidates:
        return None
    first_y = min(word[2][1] for word in candidates)
    row = sorted((word for word in candidates if abs(word[2][1] - first_y) < 14), key=lambda w: w[2][0])
    return ' '.join(word[0] for word in row)


def joker_tooltip_words(words, card_x):
    """Separate the hovered roster tooltip from shop card artwork below it.

    The rarity badge terminates the main tooltip. Edition help panels are
    separate; retain their explicit edition titles, never their bonus amounts.
    """
    endings = [word for word in words if word[1] > .85
               and word[0].strip().lower() in ('common', 'uncommon', 'rare', 'legendary')
               and abs(word[2][0] - card_x) < 180]
    if not endings:
        endings = [word for word in words if word[1] > .85
                   and word[0].strip().lower() == 'debuffed'
                   and abs(word[2][0] - card_x) < 180]
    bottom = min(endings, key=lambda w: abs(w[2][0] - card_x))[2][1] + 24 if endings else 850
    main = [word for word in words if 330 <= word[2][1] <= bottom
            and abs(word[2][0] - card_x) < 260]
    versions = [word for word in words if word[1] > .85
                and word[0].strip().lower() in ('foil', 'holographic', 'polychrome', 'negative', 'eternal')
                and 330 <= word[2][1] <= bottom and abs(word[2][0] - card_x) < 600]
    return main, versions


def _label_key(text):
    return "".join(char for char in text.casefold() if char.isalnum())


@lru_cache(maxsize=1)
def joker_aliases():
    """Map localized display names and internal names to stable policy keys."""
    definitions = joker_definitions()
    aliases = {_label_key(d.name): d.key for d in definitions.values()}
    aliases.update({_label_key(d.internal_name): d.key for d in definitions.values()})
    with zipfile.ZipFile(ROOT / "Balatro" / "Balatro.exe") as archive:
        english = archive.read("localization/en-us.lua").decode("utf8")
        pattern = r'''(?m)^\s*(j_\w+)\s*=\s*\{\s*\n\s*name\s*=\s*["']([^"\n]+?)["'],?\s*\n'''
        ids = {key: joker_name(name) for key, name in re.findall(pattern, english)}
        for filename in archive.namelist():
            if filename.startswith("localization/") and filename.endswith(".lua"):
                for internal_id, name in re.findall(pattern, archive.read(filename).decode("utf8")):
                    if ids.get(internal_id) in definitions:
                        aliases[_label_key(name)] = ids[internal_id]
    return aliases


def _match_art(image, region, templates, skip=frozenset()):
    x, y, right, bottom = region
    roi = cv2.resize(image[y:bottom, x:right], None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
    matches = []
    for key, template in templates.items():
        if key in skip:
            continue
        best = None
        scales = (.65, .7, .75, .8, .84, .9, .98) if key == "wee_joker" else (1., 1.1, 1.2, 1.3, 1.4)
        for scale in scales:
            scaled = cv2.resize(template, None, fx=scale * .5, fy=scale * .5, interpolation=cv2.INTER_AREA)
            if key == "square_joker":
                scaled = cv2.resize(scaled, (scaled.shape[1], round(scaled.shape[0] * 142 / 190)), interpolation=cv2.INTER_AREA)
            if scaled.shape[0] > roi.shape[0] or scaled.shape[1] > roi.shape[1]:
                continue
            result = cv2.matchTemplate(roi, scaled, cv2.TM_CCOEFF_NORMED)
            _, score, _, pos = cv2.minMaxLoc(result)
            if best is None or score > best[0]:
                best = (score, key, (x + 2 * pos[0] + scaled.shape[1],
                                     y + 2 * pos[1] + scaled.shape[0]))
        if best and best[0] >= .68:
            matches.append(best)
    accepted = []
    for score, key, center in sorted(matches, reverse=True):
        if all(abs(center[0] - entry[2][0]) > 85 for entry in accepted):
            accepted.append((score, key, center))
    return sorted(accepted, key=lambda entry: entry[2][0])


@lru_cache(maxsize=1)
def consumable_templates():
    from consumables import Consumable
    with zipfile.ZipFile(ROOT / "Balatro" / "Balatro.exe") as archive:
        source = archive.read("game.lua").decode("utf8")
        atlas = Image.open(io.BytesIO(archive.read("resources/textures/2x/Tarots.png"))).convert("RGBA")
    templates, definitions = {}, {}
    for line in source.splitlines():
        if not re.match(r"\s*c_\w+\s*=\s*\{", line):
            continue
        name = re.search(r'''name\s*=\s*["']([^"']+)["']''', line)
        tile = re.search(r"pos\s*=\s*\{\s*x\s*=\s*(\d+),\s*y\s*=\s*(\d+)", line)
        kind = re.search(r'''set\s*=\s*["'](Tarot|Planet|Spectral)["']''', line)
        if not name or not tile or not kind:
            continue
        col, row = map(int, tile.groups())
        tile = atlas.crop((col * 142 + 25, row * 190 + 35, col * 142 + 125, row * 190 + 160))
        background = Image.new("RGBA", tile.size, "white")
        background.alpha_composite(tile)
        key = joker_name(name[1])
        definitions[key] = Consumable(name[1], kind[1])
        templates[key] = cv2.cvtColor(np.asarray(background.convert("RGB")), cv2.COLOR_RGB2BGR)
    return templates, definitions


def read_consumables(image):
    # Two held cards spread to both edges; the left artwork starts before 1540.
    return _match_art(image, (1450, 80, 2030, 330), consumable_templates()[0])


def consumable_slots(count):
    if count < 1:
        return []
    if count == 1:
        return [1687]
    # CardArea distributes held consumables across a fixed 2.3-card-wide area,
    # even when vouchers or negative editions increase the slot count.
    return [round(1570 + i * 234 / (count - 1)) for i in range(count)]


def consumable_centers(matches, count):
    """Assign each artwork match to one slot, even when cards overlap."""
    slots = consumable_slots(count)
    tolerance = min(85, .45 * (slots[1] - slots[0])) if len(slots) > 1 else 85
    available = list(matches)
    centers = []
    for x in slots:
        nearest = min(available, key=lambda match: abs(match[2][0] - x)) if available else None
        if nearest and abs(nearest[2][0] - x) < tolerance:
            centers.append(nearest[2])
            available.remove(nearest)
        else:
            centers.append((x, 220))
    return centers


def identify_consumable_text(lines, *, title=None):
    definitions = consumable_templates()[1]
    aliases = {}
    for key, definition in definitions.items():
        aliases[_label_key(definition.name)] = key
        aliases[_label_key(definition.name.removeprefix('The '))] = key
    if title is not None and _label_key(title) in aliases:
        return aliases[_label_key(title)], 1.
    found = {aliases[_label_key(line)] for line in lines if _label_key(line) in aliases}
    from item_identity import identify_description
    # The Fool names the previously used planet inside its own effect panel.
    # Even when its title is cropped, that reference is not the offer identity.
    primary, description_error = None, None
    try:
        primary = identify_description(lines, 'consumable')
    except RuntimeError as error:
        description_error = error
    if primary is not None and primary[0] == 'the_fool':
        return primary
    if len(found) == 1:
        return next(iter(found)), 1.
    if primary is not None:
        return primary
    raise ObservationError(str(description_error)) from description_error


def shop_tooltip_region(image, card_x):
    """Locate the grey tooltip panel to the left of a hovered shop card."""
    left, top, right, bottom = 535, 350, max(1670, min(2040, int(card_x + 230))), 900
    hsv = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2HSV)
    # CRT tint can disconnect the grey border or join it to the shop buttons.
    # Anchor on the filled white effect body instead, requiring grey title
    # and footer bands on the same horizontal span. A card face or a thin
    # panel outline alone cannot establish this tooltip geometry.
    body_mask = ((hsv[:, :, 1] < 80) & (hsv[:, :, 2] > 180)).astype(np.uint8) * 255
    body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    _, _, bodies, _ = cv2.connectedComponentsWithStats(body_mask)
    body_candidates = []
    for bx, by, bw, bh, area in bodies[1:]:
        bx, by, bw, bh, area = map(int, (bx, by, bw, bh, area))
        gap = card_x - (left + bx + bw)
        if not (150 < bw < 650 and 45 < bh < 400 and area / (bw * bh) >= .5
                and 45 < gap < 380 and top + by < 740
                and by >= 55 and by + bh + 55 <= hsv.shape[0]):
            continue
        bands = (hsv[by - 55:by - 10, bx:bx + bw],
                 hsv[by + bh + 5:by + bh + 55, bx:bx + bw])
        if not all(float(np.mean((band[:, :, 1] < 90) & (band[:, :, 2] > 60)
                                 & (band[:, :, 2] < 180))) >= .35 for band in bands):
            continue
        body_candidates.append((abs(gap - 100), -area,
            (max(left, left + bx - 12), max(top, top + by - 65),
             min(right, left + bx + bw + 12), min(bottom, top + by + bh + 65))))
    body_region = min(body_candidates)[2] if body_candidates else None
    mask = ((hsv[:, :, 1] < 40) & (hsv[:, :, 2] > 75)).astype(np.uint8) * 255
    _, original_labels, original_components, _ = cv2.connectedComponentsWithStats(mask)
    # A selected card can move its tooltip against a related enhancement
    # panel. Remove their thin glow bridge before choosing the main panel.
    separated = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    _, labels, components, _ = cv2.connectedComponentsWithStats(separated)
    candidates = []
    for component, (x, y, width, height, area) in enumerate(components[1:], 1):
        pixels = original_labels[labels == component]
        original = original_components[np.bincount(pixels).argmax()]
        ox, oy, ow, oh = map(int, original[:4])
        # Keep the original vertical extent, including a detached title. If
        # two panels were joined, restrict OCR to this panel's horizontal span.
        if ow <= width + 30:
            x, width = ox, ow
        else:
            end = min(ox + ow, int(x + width) + 4)
            x = max(ox, int(x) - 4)
            width = end - x
        y, height = oy, oh
        # A white effect body may be disconnected from the thin outer border.
        # Recover the enclosing frame, which also contains title and type.
        # The width bound excludes an adjacent referenced-card help panel.
        enclosing = [(int(ey), int(eh)) for ex, ey, ew, eh, ea in original_components[1:]
                     if 0 <= int(x) - int(ex) <= 30
                     and 0 <= int(ex + ew) - int(x + width) <= 30
                     and int(ey) <= int(y) and int(ey + eh) >= int(y + height)
                     and height < int(eh) < 500 and int(ea) > 3000]
        if enclosing:
            y, height = max(enclosing, key=lambda entry: entry[1])
        x, y = int(x) + left, int(y) + top
        width, height = int(width), int(height)
        gap = card_x - (x + width)
        if (150 < width < 650 and 70 < height < 500 and area > 3000
                and 45 < gap < 380 and y < 740):
            candidates.append((abs(gap - 90), -int(area), (x, y, x + width, y + height)))
    if not candidates:
        if body_region is not None:
            return body_region
        raise ObservationError('Shop tooltip panel is not visible; refusing a whole-screen identity guess')
    x1, y1, x2, y2 = min(candidates)[2]
    # The title and rarity sit on darker grey than the white effect body.
    # Their components can be detached, especially after selection moves the
    # panel. Keep those rows without including a neighbouring card's artwork.
    if y2 - y1 < 135:
        y1, y2 = max(top, y1 - 75), min(bottom, y2 + 80)
    region = max(left, x1 - 12), y1, min(right, x2 + 12), y2
    # Keep an already complete, tightly bounded outer frame. Changing a valid
    # crop can itself perturb OCR (notably the selected Fool title). Replace
    # only a clipped title/footer or a span leaking into neighbouring buttons.
    if body_region is not None:
        bx1, by1, bx2, by2 = body_region
        if (region[0] < bx1 - 20 or region[2] > bx2 + 20
                or region[1] > by1 + 20 or region[3] < by2 - 20):
            return body_region
    return region


def shop_offer_item(labels, price, hint='', *, require_supported=True, art_confidence=0,
                    image=None, words=None):
    """Identify a named offer, including its live effect and safe consumables."""
    joker_identity = None
    if any(label.strip().lower() in ('common', 'uncommon', 'rare', 'legendary') for label in labels):
        joker_identity = identify_joker_text(labels, hint, art_confidence=art_confidence)
    else:
        # Rarity can be outside a cropped tooltip. A distinctive effect still
        # establishes the item type; don't route every such offer to Tarot OCR.
        from item_identity import identify_description
        try:
            joker_identity = identify_description(labels, 'joker')
        except RuntimeError:
            pass
    if joker_identity is not None:
        key, confidence = joker_identity
        joker = parse_joker_tooltip(joker_from_definition(key), labels, image=image, words=words)
        return ShopItem('joker', joker.name, price, joker=joker, edition=joker.edition), confidence
    key, confidence = identify_consumable_text(labels, title=labels[0] if labels else None)
    consumable = consumable_templates()[1][key]
    kind = joker_name(consumable.kind)
    item = ShopItem(kind, consumable.name, price)
    if require_supported and not supported_shop_item(item):
        raise ObservationError('Consumable needs an unsupported use workflow: ' + consumable.name)
    return item, confidence


def offer_art_patch(image, center):
    x, y = map(int, center)
    return image[max(0, y - 60):y + 60, max(0, x - 45):x + 45].copy()


def same_offer_art(image, center, patch):
    """A skipped offer is reused only while its visible artwork still agrees."""
    if patch is None or not patch.size or float(np.std(patch)) < 8:
        return False
    x, y = map(int, center)
    search = image[max(0, y - 75):y + 75, max(0, x - 60):x + 60]
    if search.shape[0] < patch.shape[0] or search.shape[1] < patch.shape[1]:
        return False
    score = cv2.matchTemplate(search, patch, cv2.TM_CCOEFF_NORMED)
    _, confidence, _, (px, py) = cv2.minMaxLoc(score)
    matched = search[py:py + patch.shape[0], px:px + patch.shape[1]]
    return confidence >= .97 and float(np.abs(matched.astype(np.float32) - patch).mean()) <= 12


def supported_shop_item(item):
    """Identity is independent of whether a visible Use workflow is supported."""
    if item.kind not in ('tarot', 'spectral', 'planet'):
        return True
    from consumables import ENHANCEMENT_TAROTS, SUIT_TAROTS, SEAL_SPECTRALS
    safe = set(ENHANCEMENT_TAROTS) | set(SUIT_TAROTS) | set(SEAL_SPECTRALS)
    safe.update(('strength', 'death', 'the_hanged_man', 'the_hermit', 'temperance', 'cryptid',
                 'black_hole', 'judgement', 'the_soul', 'ankh', 'hex'))
    return item.kind == 'planet' or joker_name(item.name) in safe


def read_reroll_button(image):
    words = read_text(image, (605, 575, 905, 760))
    buttons = [p for text, score, p in words if text.strip().lower() == 'reroll' and score > .85]
    button = buttons[0] if len(buttons) == 1 else None
    prices = [int(text.strip()[1:]) for text, score, _ in words
              if score > .85 and re.fullmatch(r'\$\d+', text.strip())]
    if button is None or len(prices) > 1:
        return None
    if len(prices) == 1:
        return button, prices[0]
    # A wide detector read missed $6 (confidence .66) in the Needle shop.
    # Require two agreeing local reads (tight recognition and/or two contrast
    # variants), preserving the original wide-read acceptance threshold.
    x, y = button
    region = (max(0, x - 60), y + 21, min(image.shape[1], x + 60), y + 86)
    tight = [int(m[1]) for text, score, _ in read_text(image, region, line=True)
             if score > .8 and (m := re.fullmatch(r'\s*\$\s*(\d{1,2})\s*', text))]
    if len(tight) > 1:
        return None
    crop = image[region[1]:region[3], region[0]:region[2]]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    proofs = [tight[0]] if tight else []
    for threshold in (200, 190):
        binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)[1]
        contrast = [int(m[1]) for text, score, _ in read_text(
            cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), line=True)
            if score > .9 and (m := re.fullmatch(r'\s*\$\s*(\d{1,2})\s*', text))]
        if len(contrast) > 1:
            return None
        proofs.extend(contrast)
        if len(proofs) >= 2:
            break
    if len(proofs) >= 2 and len(set(proofs)) == 1:
        return button, proofs[0]
    return None


def consumable_counter(image):
    for text, score, _ in read_text(image, (1730, 325, 2010, 395)):
        match = re.search(r"(\d+)\s*/\s*(\d+)", text)
        if match and score > .75:
            return tuple(map(int, match.groups()))
    raise ObservationError("Cannot read consumable slot counter")


def joker_counter(image):
    for text, score, _ in read_text(image, (545, 325, 745, 390)):
        match = re.search(r"(\d+)\s*/\s*(\d+)", text)
        if match and score > .75:
            count, capacity = map(int, match.groups())
            if 0 <= count <= capacity <= 100:
                return count, capacity
    # The slash in a small counter can disappear in panel OCR (3/5 -> 375).
    # Read the isolated line, requiring the actual slash and a plausible count.
    for region in ((570, 337, 700, 378), (578, 337, 641, 378)):
        # The Water's bright background pulse can hide this text from a
        # wide OCR crop. The tight crop excludes its neighbouring glow;
        # retain the wide crop for counters with two-digit capacities.
        for text, score, _ in read_text(image, region, line=True):
            match = re.fullmatch(r'\s*(\d+)\s*/\s*(\d+)\s*', text)
            if match and score > .85:
                count, capacity = map(int, match.groups())
                if 0 <= count <= capacity <= 100:
                    return count, capacity
    raise ObservationError("Cannot read Joker slot counter")


def joker_from_definition(key):
    definition = joker_definitions()[key]
    config = definition.config
    value = config.get("mult", config.get("chips", 0))
    # Multiplicative values consistently mean the full multiplier, never its increment.
    if key in ("hologram", "constellation", "vampire", "campfire", "obelisk", "glass_joker", "lucky_cat", "hit_the_road", "canio", "yorick", "madness"):
        value = 1
    if key == "ramen":
        value = 2
    return Joker(definition.name, value=value, rarity=definition.rarity,
                 data=(("copyable", float(definition.copyable)),))


def read_ramen_multiplier(image, words):
    """Recover a red counter from its proven first effect row, on the same frame.

    Ordinary readable tooltips never call this. Neither the title, shop price
    nor Ramen's initial X2 is evidence of its current counter.
    """
    if not words:
        return None
    titles = [w for w in words if w[0].strip().casefold() == 'ramen' and w[1] >= .8]
    if len(titles) != 1:
        return None
    tx, ty = titles[0][2]
    anchors = [w for w in words if re.search(r'\bmult\b', w[0], re.I)
               and 'loses' not in w[0].casefold() and w[1] >= .8
               and abs(w[2][0] - tx) < 120 and ty + 15 <= w[2][1] <= ty + 100]
    if not anchors:
        return None
    first_y = min(w[2][1] for w in anchors)
    anchors = [w for w in anchors if abs(w[2][1] - first_y) <= 8]
    if len(anchors) != 1:
        return None
    x, y = anchors[0][2]
    a, b, c, e = map(int, (max(525, x - 180), max(330, y - 14),
                           min(2020, x + 250), min(1000, y + 15)))
    pixels = image[b:e, a:c]
    if not pixels.size:
        return None
    blue, green, red = [pixels[:, :, i].astype(np.float32) for i in range(3)]
    mask = ((red > 150) & (red > green * 1.4) & (red > blue * 1.4)).astype(np.uint8) * 255
    _, _, components, _ = cv2.connectedComponentsWithStats(mask)
    boxes = [(int(xx), int(yy), int(w), int(h)) for xx, yy, w, h, _ in components[1:]
             if 18 <= w <= 90 and h >= 20 and abs(a + xx + w / 2 - x) <= 80]
    if len(boxes) != 1:
        return None
    xx, yy, width, height = boxes[0]
    region = (max(a, a + xx - 8), max(330, b - 5), min(c, a + xx + width + 8), e + 5)
    gray = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    contrast = cv2.convertScaleAbs(gray, alpha=1.6, beta=-50)
    glyph = 255 - mask[:, max(0, xx - 6):min(mask.shape[1], xx + width + 6)]
    glyph = cv2.copyMakeBorder(glyph, 7, 7, 7, 7, cv2.BORDER_CONSTANT, value=0)
    glyph = cv2.cvtColor(glyph, cv2.COLOR_GRAY2BGR)
    votes = Counter()
    def add_vote(view, crop):
        labels = [w[0].strip() for w in read_text(view, crop, line=True) if w[1] >= .8]
        text = ' '.join(labels)
        # This crop contains only the red factor field, not the decay/body.
        # OCR can omit X or turn it into H; no decimal is ever inserted.
        match = re.fullmatch(r'[\[({]*\s*[xX×H]?\s*(\d+(?:\.\d+)?)\s*[\])}]*', text)
        if match:
            value = float(match[1])
            if 1 <= value <= 2:
                votes[value] += 1
    add_vote(contrast, region)
    contrast_proven = bool(votes)
    add_vote(glyph, (0, 0, glyph.shape[1], glyph.shape[0]))
    if len(votes) > 1:
        return None
    if sum(votes.values()) < 2:
        if contrast_proven:
            add_vote(image, region)
        else:
            # With no usable first view, two isolated glyph layouts must prove
            # the number. A single confident digit still cannot authorize it.
            padding = (6,) if votes else (8, 10)
            for pad in padding:
                positive = cv2.copyMakeBorder(mask[yy:yy + height, xx:xx + width],
                                             pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
                positive = cv2.resize(positive, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
                positive = cv2.cvtColor(positive, cv2.COLOR_GRAY2BGR)
                add_vote(positive, (0, 0, positive.shape[1], positive.shape[0]))
                if len(votes) > 1:
                    return None
    return next(iter(votes)) if len(votes) == 1 and sum(votes.values()) >= 2 else None


def parse_joker_tooltip(joker, lines, *, effect_lines=None, image=None, words=None):
    """Current counters override static defaults only when explicitly observed."""
    text = " ".join(lines)
    value = joker.value
    data = dict(joker.data)
    if re.search(r'\bEternal\b', text, re.I):
        data['eternal'] = 1
    if joker.key == 'turtle_bean':
        match = re.search(r'\+\s*(\d+)\s+hand\s+size', text, re.I)
        if match:
            data['hand_size_bonus'] = int(match[1])
    if joker.key == 'seltzer':
        match = re.search(r'(?:next|remaining)\s*(\d+)\s*hands?', text, re.I)
        if match:
            data['remaining_hands'] = int(match[1])
            data['lifetime_observed'] = 1
        else:
            # The physical Joker still retriggers now. Its remaining lifetime
            # is unknown, so do not budget it as a fresh ten-hand purchase.
            data['remaining_hands'] = 1
            data['lifetime_observed'] = 0
    current = re.search(r"Currently[:\s]*(?:[xX×]\s*)?([+]?\d+(?:\.\d+)?)\s*(Mult|Chips)", text, re.I)
    if current:
        value = float(current[1])
        data["observed_value"] = 1
    if joker.key == 'fortune_teller':
        # This installed tooltip omits the word Mult in its current-value row.
        # Read the bounded row, never the +1 growth rate in the description.
        amount = current_fortune_mult(text)
        if amount is not None:
            value = amount
            data['observed_value'] = 1
        else:
            data.pop('observed_value', None)
            data['counter_lower_bound'] = 1
            value = 0
    if joker.key == 'to_do_list':
        data.pop('hand', None)
        # Anchor the full category after "is a": Pair must not match Two Pair.
        targets = re.findall(r'\bis\s+a\s+([a-z ]+?)[.,]', text, re.I)
        categories = {HAND_ALIASES.get(' '.join(t.lower().split())) for t in targets}
        categories.discard(None)
        if len(categories) == 1:
            from strategy import HAND_RANK
            data['hand'] = HAND_RANK[categories.pop()]
    if joker.key == 'mail_in_rebate':
        data.pop('rank', None)
        primary = ' '.join(lines if effect_lines is None else effect_lines)
        targets = re.findall(r'\bdiscarded\s+(10|[2-9]|Ace|King|Queen|Jack)\s*[,.;]?\s*rank\b', primary, re.I)
        ranks = {int(t) if t.isdigit() else {'ace': 14, 'king': 13, 'queen': 12, 'jack': 11}[t.lower()] for t in targets}
        if len(ranks) == 1:
            data['rank'] = ranks.pop()
    if joker.key in ('rocket', 'cloud_9', 'satellite'):
        data.pop('income', None)
        primary = ' '.join(lines if effect_lines is None else effect_lines)
        expression = (r'\bEarn\s*\$\s*(\d+)\s+at\s+end\s+of\s+round'
                      if joker.key == 'rocket' else r'\bCurrently\s*[:\s]*\$\s*(\d+)')
        incomes = {int(v) for v in re.findall(expression, primary, re.I)}
        if len(incomes) == 1:
            data['income'] = incomes.pop()
    direct_effects = {
        'popcorn': r'\+\s*(\d+(?:\.\d+)?)\s*Mult\b',
        'ice_cream': r'\+\s*(\d+(?:\.\d+)?)\s*Chips\b',
        'ramen': r'[xX×]\s*(\d+(?:\.\d+)?)\s*Mult\b',
    }
    if joker.key in direct_effects and not current:
        primary = ' '.join(lines if effect_lines is None else effect_lines)
        amounts = {float(amount) for amount in re.findall(direct_effects[joker.key], primary, re.I)}
        if joker.key == 'ramen':
            amounts = {amount for amount in amounts if amount >= 1}  # Exclude X0.01 decay.
            if not amounts and image is not None:
                recovered = read_ramen_multiplier(image, words)
                if recovered is not None:
                    amounts = {recovered}
            if any(amount > 2 for amount in amounts):
                raise ObservationError('Current Ramen effect exceeds its physical maximum')
        if len(amounts) != 1:
            raise ObservationError('Current ' + joker.name + ' effect is unreadable or ambiguous')
        value = amounts.pop()
        data['observed_value'] = 1
    if joker.key == "loyalty_card":
        match = re.search(r"(\d+)\s+remaining", text, re.I)
        if match:
            data["remaining"] = int(match[1])
    if joker.key == 'yorick':
        counter = re.search(r'every\s*(\d+)\s*\[\s*(\d+)\s*\]', text, re.I)
        growth = re.search(r'gains\s*[xX×]\s*(\d+(?:\.\d+)?)\s*Mult', text, re.I)
        if counter and growth and 1 <= int(counter[2]) <= int(counter[1]):
            data['discard_threshold'] = int(counter[1])
            data['discards_remaining'] = int(counter[2])
            data['discard_growth'] = float(growth[1])
    for suit, label in zip(SUITS, ("Spades", "Hearts", "Diamonds", "Clubs")):
        if re.search(r'\b' + label.lower().rstrip('s') + r's?\b', text.lower()) and joker.key in ("ancient_joker", "the_idol", "castle"):
            data["suit"] = SUITS.index(suit)
    if joker.key == "the_idol":
        for label, rank in (("Ace", 14), ("King", 13), ("Queen", 12), ("Jack", 11)):
            if re.search(r"\b" + label + r"\b", text, re.I):
                data["rank"] = rank
        match = re.search(r"(?:played|scored)\s+(10|[2-9])\b", text, re.I)
        if match:
            data["rank"] = int(match[1])
    edition = next((key for key in ("polychrome", "holographic", "foil", "negative") if key in text.lower()), joker.edition)
    if joker.key == "drivers_license":
        match = re.search(r"Currently[:\s]*(\d+)", text, re.I)
        if match:
            data["enhanced_count"] = int(match[1])
    return replace(joker, value=value, edition=edition, data=tuple(data.items()), debuffed="debuffed" in text.lower())


CURRENT_XMULT_JOKERS = frozenset(('hologram', 'constellation', 'vampire', 'campfire',
    'obelisk', 'glass_joker', 'lucky_cat', 'hit_the_road', 'canio', 'yorick', 'madness', 'steel_joker'))


def bounded_temporary_joker(joker, lines):
    """A proven identity with unreadable declining numbers can still score safely."""
    if joker.key not in ('popcorn', 'ice_cream', 'ramen'):
        raise ObservationError('No proven numeric bound for ' + joker.name)
    text = ' '.join(lines).lower()
    data = dict(joker.data)
    data.pop('observed_value', None)
    data['counter_lower_bound'] = 1
    if 'eternal' in text:
        data['eternal'] = 1
    edition = next((key for key in ('polychrome', 'holographic', 'foil', 'negative')
                    if key in text), joker.edition)
    return replace(joker, value=1 if joker.key == 'ramen' else 0, edition=edition,
                   debuffed='debuffed' in text, data=tuple(data.items()))


def current_multiplier(text):
    """Read only a Currently row, allowing an OCR variant of its X prefix."""
    match = re.search(r'Currently[\s:\[({]*[xX×K£E]?\s*(\d+(?:\.\d+)?)\s*Mult\b', text, re.I)
    return float(match[1]) if match else None


def current_fortune_mult(text):
    amounts = {int(v) for v in re.findall(
        r'\bCurrently\s*[:\s]*\+\s*(\d+)\s*(?:Mult\s*)?[)\]]', text, re.I)}
    return float(amounts.pop()) if len(amounts) == 1 else None


def verify_current_multiplier(image, words, joker, state, previous=None):
    """Independent scoped reads and physical growth bounds protect scoring.

    Missing decimals are never inserted by guessing. If views cannot agree,
    use an explicit lower bound rather than treating a default as observed.
    """
    fortune = joker.key == 'fortune_teller'
    if (joker.key not in CURRENT_XMULT_JOKERS and not fortune) or joker.debuffed:
        return joker
    lower, upper = (0. if fortune else 1.), float('inf')
    if fortune and previous is not None and previous.numeric_value_available:
        # All physical Fortune Tellers share a monotonically growing run tally.
        lower = previous.value
    if joker.key == 'madness':
        # Each non-Boss selection grants at most X0.5. Ante-reducing vouchers
        # can revisit earlier Antes, so account for both owned reductions.
        reductions = sum(joker_name(v) in ('hieroglyph', 'petroglyph') for v in state.owned_vouchers)
        selections = 2 * max(0, state.ante - 1 + reductions) + min(2, max(0, state.blind_index + 1))
        upper = min(1 + .5 * selections, 1 + .5 * (state.total_hands_played + 1))
        if (previous is not None and previous.key == joker.key and previous.numeric_value_available
                and 1 <= previous.value <= upper + 1e-6):
            lower = previous.value
    if joker.key == 'obelisk':
        upper = 1 + .2 * max(0, state.total_hands_played)
        # A lower bound cannot bound the next actual value from above. Only a
        # confirmed previous counter on this unique physical copy can do so.
        if (previous is not None and previous.key == joker.key
                and previous.get('observed_value') and 1 <= previous.value <= upper + 1e-6):
            at = previous.get('value_observed_at_hand', -1)
            if 0 <= at <= state.total_hands_played:
                upper = min(upper, previous.value + .2 * (state.total_hands_played - at))
    if joker.key == 'vampire':
        # At most five scoring enhancements can be removed by one play.
        upper = 1 + .5 * state.total_hands_played
        if previous is not None and previous.key == joker.key and 1 <= previous.value <= upper + 1e-6:
            lower = previous.value
            at = previous.get('value_observed_at_hand', -1)
            if at >= 0:
                upper = min(upper, lower + .5 * max(0, state.total_hands_played - at))
    anchors = [w for w in words if 'currently' in w[0].lower() and w[1] >= .75]
    votes = Counter()
    if len(anchors) == 1:
        x, y = anchors[0][2]
        region = (max(525, x - 140), max(330, y - 22), min(2020, x + 300), min(850, y + 22))
        gray = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        views = [(pixels, region, line) for pixels in (image, gray) for line in (True, False)]
        if joker.key == 'madness':
            # Still four reads. Tight views separate the current counter from
            # the destruction line and the neighbouring edition panel.
            tight = (region[0], max(330, y - 15), region[2], min(850, y + 15))
            views = [(image, region, True), (gray, region, False),
                     (image, tight, True), (image, tight, False)]
        for pixels, crop, single_line in views:
            read = read_text(pixels, crop, line=single_line)
            labels = tooltip_rows([w for w in read if w[1] >= .75])
            text = ' '.join(labels)
            if joker.key == 'madness':
                # Scoped Currently row only; H is a verified X OCR variant.
                text = re.sub(r'(Currently[\s:\[({]*)H(?=\s*\d)', r'\1X', text, flags=re.I)
            value = (current_fortune_mult if fortune else current_multiplier)(text)
            if value is not None and lower - 1e-6 <= value <= upper + 1e-6:
                votes[value] += 1
    data = dict(joker.data)
    ordered = votes.most_common()
    if ordered and ordered[0][1] >= 2 and (len(ordered) == 1 or
            (not fortune and joker.key != 'madness' and ordered[0][1] > ordered[1][1])):
        data['observed_value'] = 1
        data['value_observed_at_hand'] = state.total_hands_played
        data.pop('counter_lower_bound', None)
        return replace(joker, value=ordered[0][0], data=tuple(data.items()))
    data.pop('observed_value', None)
    data['counter_lower_bound'] = 1
    if joker.key == 'vampire' and previous is not None and previous.get('value_observed_at_hand', -1) >= 0:
        data['value_observed_at_hand'] = previous.get('value_observed_at_hand')
    return replace(joker, value=lower, data=tuple(data.items()))


def parse_card_tooltip(card, lines, *, require_identity=False):
    text = " ".join(lines).lower()
    text = re.sub(r'\bqveen\b', 'queen', text)
    # Pixel-font OCR can lose the short word 'of' (e.g. 'Ҷf Spades').
    # A suit name plus an explicit base-chip line still proves a card tooltip.
    suit_words = set(re.findall(r'\b(spades|hearts|diamonds|clubs)\b', text))
    chip_line = re.search(r'\+\s*\d+\s+chips\b', text)
    if not card.face_down and not (len(suit_words) == 1 and chip_line) and not re.search(r'\bof\s+(?:spades|hearts|diamonds|clubs)\b|\b(?:stone|bonus|mult|wild|glass|steel|gold|lucky) card\b', text):
        # Slot counters alone do not prove that hover reached the intended card.
        # Modification-only inputs remain useful for existing pure callers.
        if not any(word in text for word in ('seal', 'polychrome', 'holographic', 'foil', 'extra chips')):
            raise ObservationError('Card tooltip contains no readable card identity or modifiers')
    # Joining panel counters with a split title fabricated "2 of Clubs"
    # from "0/2", "of Clubs", "King". Counter/pip-only lines are not titles.
    identity_text = ' '.join(line for line in lines if not re.fullmatch(r'\s*\d+(?:\s*/\s*\d+)?\s*', line)).lower()
    identity_text = re.sub(r'\bqveen\b', 'queen', identity_text)
    titles = set(re.findall(r'(?<![/\d])\b(ace|king|queen|jack|10|[2-9])\s+of\s+(spades|hearts|diamonds|clubs)\b', identity_text))
    rank_proven = len(titles) == 1
    if len(titles) > 1 and require_identity:
        raise ObservationError('Card tooltip has conflicting rank/suit titles')
    if len(titles) == 1 and not card.face_down:
        rank_label, suit_label = next(iter(titles))
        rank = {'ace': 14, 'king': 13, 'queen': 12, 'jack': 11}.get(rank_label)
        card = replace(card, rank=rank if rank is not None else int(rank_label),
                       suit={'spades': 'S', 'hearts': 'H', 'diamonds': 'D', 'clubs': 'C'}[suit_label])
    elif not card.face_down and not card.rankless:
        suits = set(re.findall(r'\bof\s+(spades|hearts|diamonds|clubs)\b', text))
        if not suits and chip_line:
            suits = suit_words
        if len(suits) == 1:
            card = replace(card, suit={'spades': 'S', 'hearts': 'H', 'diamonds': 'D', 'clubs': 'C'}[next(iter(suits))])
            # OCR sometimes drops the single digit in '7 of Spades'. Ordinary
            # 2..9 chip values disambiguate it; ten chips cannot identify a face.
            base_chips = set(re.findall(r'\+\s*(\d+)\s+chips\b', text))
            if len(base_chips) == 1 and not card.enhancement and not re.search(r'\b(?:bonus|stone) card\b', text):
                chips = int(next(iter(base_chips)))
                if 2 <= chips <= 9 or chips == 11:
                    card = replace(card, rank=14 if chips == 11 else chips)
                    rank_proven = True
    enhancement = next((key for key in ("bonus", "mult", "wild", "glass", "steel", "stone", "gold", "lucky")
                        if re.search(r"\b" + key + r" card\b", text)), card.enhancement)
    base_chips = set(re.findall(r'\+\s*(\d+)\s+chips\b', text))
    if (require_identity and rank_proven and len(base_chips) == 1
            and not card.face_down and enhancement not in ('bonus', 'stone')
            and not card.bonus_chips and not re.search(r'\bextra chips\b', text)
            and 'debuffed' not in text):
        chips = int(next(iter(base_chips)))
        if chips in (*range(2, 10), 11) and card.chips != chips:
            raise ObservationError('Card title contradicts its base chips; reread the isolated title')
    if len(base_chips) == 1 and not rank_proven and not card.face_down and enhancement != 'stone':
        chips = int(next(iter(base_chips)))
        if 2 <= chips <= 9 or chips == 11:
            card = replace(card, rank=14 if chips == 11 else chips)
            rank_proven = True
    if require_identity and not card.face_down and enhancement != 'stone' and not rank_proven:
        raise ObservationError('Face-up card rank is not proven by its tooltip; reread instead of retaining a template guess')
    edition = next((key for key in ("polychrome", "holographic", "foil") if key in text), card.edition)
    seal = next((key for key in ("red", "blue", "gold", "purple") if key + " seal" in text), card.seal)
    # The game's tooltip reports ability.bonus + ability.perma_bonus as
    # "extra chips". Card.bonus_chips stores only the permanent component;
    # score_hand adds Bonus Card's 30 chips (or Stone Card's 50) separately.
    bonus = card.bonus_chips
    extra_chips = {int(n) for n in re.findall(r"\+\s*(\d+)\s+extra chips", text)}
    if len(extra_chips) > 1:
        raise ObservationError('Card tooltip has conflicting extra-chip totals')
    if extra_chips:
        total = next(iter(extra_chips))
        fixed = 30 if enhancement == 'bonus' else 0
        if total < fixed:
            raise ObservationError('Card extra chips contradict its enhancement')
        bonus = total - fixed
    elif enhancement == 'stone' and len(base_chips) == 1:
        total = int(next(iter(base_chips)))
        if total < 50:
            raise ObservationError('Stone Card chips are below its fixed bonus')
        bonus = total - 50
    return replace(card, enhancement=enhancement, edition=edition, seal=seal,
                   bonus_chips=bonus, rankless=enhancement == "stone",
                   debuffed=card.debuffed or "debuffed" in text)


def read_card_title(image, position):
    """Read the white rank/suit header independently of its tooltip body."""
    x, y = position
    left, top = max(530, x - 160), max(240, y - 600)
    right, bottom = min(2020, x + 350), min(875, y - 35)
    if right <= left or bottom <= top:
        return []
    hsv = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 1] < 65) & (hsv[:, :, 2] > 185)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    _, _, components, _ = cv2.connectedComponentsWithStats(mask)
    titles = {}
    for bx, by, width, height, area in components[1:]:
        if not (100 < width < 450 and 15 < height < 55 and width / height > 3
                and area > 900 and abs(left + bx + width / 2 - (x + 71)) < 95):
            continue
        region = (int(left + bx - 4), int(top + by - 4),
                  int(left + bx + width + 4), int(top + by + height + 4))
        for title, confidence, _ in read_text(image, region, line=True):
            if confidence > .90 and re.fullmatch(
                    r'\s*(?:Ace|King|Queen|Jack|10|[2-9])\s+of\s+(?:Spades|Hearts|Diamonds|Clubs)\s*',
                    title, re.I):
                titles[title.strip().casefold()] = title.strip()
    # Multiple different headers cannot establish which card was hovered.
    return list(titles.values()) if len(titles) == 1 else []


@lru_cache(maxsize=1)
def blind_names():
    with zipfile.ZipFile(ROOT / 'Balatro' / 'Balatro.exe') as archive:
        source = archive.read('localization/en-us.lua').decode('utf8')
    return tuple(name for name in re.findall(
        r'bl_\w+\s*=\s*\{\s*name\s*=\s*"([^"]+)"', source)
        if name not in ('Small Blind', 'Big Blind'))


def read_future_boss(image):
    """Read the visible Boss column while all three blinds are displayed."""
    words = read_text(image, (1340, 445, 1650, 975))
    titles = [(t, c, p) for t, c, p in words if 445 <= p[1] <= 635]
    candidates = [t for t, c, p in titles if c >= .85] + tooltip_rows(titles, minimum=.85)
    names = {name for name in blind_names() for text in candidates
             if ' '.join(text.split()).casefold() == name.casefold()}
    if not names and titles:
        # A split Verdant / t Leaf title is repaired on the same frame.
        # Similarity chooses only a crop, never a Boss identity.
        anchor = max(titles, key=lambda w: max(SequenceMatcher(None,
            w[0].casefold(), name.casefold()).ratio() for name in blind_names()))[2][1]
        local = read_text(image, (1340, anchor - 31, 1650, anchor + 31), line=True)
        names = {name for text, confidence, _ in local if confidence > .9
                 for name in blind_names() if text.strip().casefold() == name.casefold()}
        body = ' '.join(t.lower() for t, c, p in words if c > .9)
        # Only observed pixel-font confusions, corroborated by unique rule
        # text from this same Boss column. No general fuzzy-title acceptance.
        aliases = (
            ('Verdant Leaf', {'uerdant leaf'}, ('all cards debuffed', 'until 1 joker sold')),
            ('Violet Vessel', {'violet uessel', 'uiolet vessel', 'vialet uessel'}, ('very large blind',)),
            ('The Ox', {'the 0x'}, ('playing a', 'sets money to $0')),
        )
        if not names:
            names = {name for name, variants, rules in aliases
                     if any(t.strip().casefold() in variants and c > .9 for t, c, p in local)
                     and all(rule in body for rule in rules)}
    targets = {score_numeric(text) for text, confidence, (_, y) in words
               if confidence >= .85 and 700 <= y <= 925 and '$' not in text}
    targets = {value for value in targets if value is not None and value >= 100}
    if len(names) != 1 or len(targets) != 1:
        return None
    return names.pop(), targets.pop()


def observed_state(image, previous: RunState):
    values = read_numbers(image)
    count, limit = hand_counter(image)
    values["hand_size"] = limit
    values["consumables_held"], values["consumable_slots"] = consumable_counter(image)
    if "ante" in values and values["ante"] != previous.ante:
        values["pillar_cards"] = ()
    # Missing scores would make both discard and one-hand-win decisions unsafe.
    missing = {"target_score", "round_score", "hands_left", "discards_left"} - values.keys()
    if missing:
        raise ObservationError("Cannot read required game counters: " + ", ".join(sorted(missing)))
    boss_lines = tooltip_labels(read_text(image, (120, 115, 530, 235)), minimum=.8)
    for text in boss_lines:
        text = ' '.join(text.split())
        if text in blind_names():
            values["boss"] = text
    if any("Blind" in line for line in boss_lines):
        values["boss"] = ""
    if replace(previous, **values)._boss_name == 'ox':
        from strategy import hand_name
        targets = {hand_name(match[1]) for line in boss_lines
                   if (match := re.fullmatch(r'Playing\s+(?:a|an)\s+(.+)', line.strip(), re.I))}
        targets &= set(HAND_ALIASES.values())
        if len(targets) == 1:
            values['boss_target_hand'] = targets.pop()
    return replace(previous, **values), count


def parse_levels(words):
    """Pair each Poker Hands row with the level displayed to its left."""
    levels, played = {}, {}
    for text, confidence, (_, y) in words:
        category = recognize_hand_name(text)
        if category is None or confidence < .75:
            continue
        row = sorted([entry for entry in words if abs(entry[2][1] - y) < 18], key=lambda entry: entry[2][0])
        for label, score, _ in row:
            match = re.search(r"(?:Lvl|Lv|Level|lul|lvi)[.\s]*(\d+)", label, re.I)
            if match and score > .75:
                levels[category] = int(match[1])
        # The far right # column shows all-run play counts (Supernova/Obelisk).
        if row:
            value = numeric(row[-1][0])
            if value is not None and row[-1][1] > .8:
                played[category] = int(value)
    return levels, played


def recognize_hand_name(text):
    label = text.strip().lower()
    direct = HAND_ALIASES.get(label)
    if direct:
        return direct
    label = label.translate(str.maketrans({'5': 's', '0': 'o', '1': 'l'}))
    direct = HAND_ALIASES.get(label)
    if direct:
        return direct
    best_by_category = {}
    for alias, category in HAND_ALIASES.items():
        similarity = SequenceMatcher(None, alias, label).ratio()
        best_by_category[category] = max(best_by_category.get(category, 0), similarity)
    ranked = sorted(((score, category) for category, score in best_by_category.items()), reverse=True)
    if ranked and ranked[0][0] >= .9 and ranked[0][0] - ranked[1][0] >= .08:
        return ranked[0][1]
    return None


def read_hand_preview(image, *, contrast=False):
    if contrast:
        # A different observation of the same verified selection, without
        # another screenshot, hover or animation wait. Keep the same bounds
        # and acceptance rules; never condition OCR on the planned category.
        crop = image[545:655, 150:535]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
        words = [(text, score, (x + 150, y + 545)) for text, score, (x, y)
                 in read_text(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR))]
    else:
        words = read_text(image, (150, 545, 535, 655))
    return parse_hand_preview(words)


def read_hand_preview_level(image):
    words = read_text(image, (150, 545, 535, 655))
    category = parse_hand_preview(words)
    labels = ' '.join(text for text, confidence, _ in words if confidence > .85)
    levels = {int(value) for value in re.findall(r'\b(?:lvl|lv|level|lul|lvi)[.\s]*(\d+)\b', labels, re.I)}
    if len(levels) != 1 or min(levels) < 1:
        raise ObservationError('Selected hand preview level is unreadable or conflicting')
    return category, levels.pop()


def parse_hand_preview(words):
    rows = []
    for word in sorted((w for w in words if w[1] > .8), key=lambda w: (w[2][1], w[2][0])):
        row = next((r for r in rows if abs(r[0][2][1] - word[2][1]) < 14), None)
        if row is None:
            rows.append([word])
        else:
            row.append(word)
    categories = set()
    # Parse full rows: "Pair" inside "Two Pair", or "Flush" inside
    # "Straight Flush", is not a second competing hand category.
    for row in rows:
        ordered = sorted(row, key=lambda w: w[2][0])
        text = ' '.join(w[0] for w in ordered)
        title = re.split(r'\b(?:lvl|lv|lul|lvi)[.\s]*\d', text, flags=re.I)[0].strip()
        category = recognize_hand_name(title)
        if category is None and len(ordered) > 1:
            # The small level suffix can read as 141.4 or l41.1 while the
            # complete hand title is exact. This proves only the category;
            # read_hand_preview_level still requires a readable level.
            suffix = ordered[-1][0].strip()
            if (len(suffix) <= 12 and re.fullmatch(r'[lviu\d.\u03c8]+\.\d+', suffix, re.I)
                    and ordered[-1][2][0] > ordered[-2][2][0] + 15):
                category = HAND_ALIASES.get(' '.join(w[0] for w in ordered[:-1]).strip().lower())
        if category:
            categories.add(category)
    if len(categories) > 1:
        raise HandPreviewConflict('Selected hand preview contains conflicting categories: ' + ', '.join(sorted(categories)))
    if not categories:
        raise ObservationError('Selected hand preview category is not readable or unique')
    return categories.pop()
