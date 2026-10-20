"""Visible booster/voucher workflows; the game archive is a read-only catalogue.

No memory inspection or game callbacks are used. A transaction is prepared
only after its identity and button have both been read from a screenshot.
"""
from __future__ import annotations

import io
import math
import re
import zipfile
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from strategy import Card, RunState, ShopItem, PLANET_HAND, joker_name
from recovery import ObservationError, begin_action, confirm_pending

ROOT = Path(__file__).resolve().parent
PACK_TYPES = ('arcana', 'celestial', 'spectral', 'standard', 'buffoon')
# These require additional shop/boss interfaces which are not implemented yet.
DEFERRED_VOUCHERS = frozenset({'magic_trick', 'illusion', 'directors_cut',
    'retcon', 'hieroglyph', 'petroglyph', 'observatory'})


@lru_cache(maxsize=1)
def extra_catalogue():
    """All 32 vouchers and 32 booster artwork variants in the installed copy."""
    definitions, templates = {}, {}
    with zipfile.ZipFile(ROOT / 'Balatro' / 'Balatro.exe') as archive:
        source = archive.read('game.lua').decode('utf8')
        atlases = {kind: Image.open(io.BytesIO(archive.read(path))).convert('RGBA')
            for kind, path in [('voucher', 'resources/textures/2x/Vouchers.png'),
                               ('booster', 'resources/textures/2x/boosters.png')]}
    for line in source.splitlines():
        key = re.match(r'\s*([vp]_\w+)\s*=\s*\{', line)
        name = re.search(r'''name\s*=\s*"([^"]+)"''', line)
        tile = re.search(r'pos\s*=\s*\{x=(\d+),y=(\d+)\}', line)
        if not (key and name and tile):
            continue
        voucher = key[1].startswith('v_')
        kind = 'voucher' if voucher else next((k + '_pack' for k in PACK_TYPES
                                               if k in name[1].lower()), '')
        if not kind:
            continue
        size = re.search(r'extra\s*=\s*(\d+)', line)
        choose = re.search(r'choose\s*=\s*(\d+)', line)
        definitions[key[1]] = dict(name=name[1], kind=kind,
            size=int(size[1]) if size else 0, choose=int(choose[1]) if choose else 1,
            requires=tuple(re.findall(r"'([^']+)'", line.split('requires =', 1)[1].split('}', 1)[0]))
                     if 'requires =' in line else ())
        x, y = map(int, tile.groups())
        patch = atlases['voucher' if voucher else 'booster'].crop(
            (x * 142 + 25, y * 190 + 35, x * 142 + 125, y * 190 + 160))
        bg = Image.new('RGBA', patch.size, 'white')
        bg.alpha_composite(patch)
        templates[key[1]] = cv2.cvtColor(np.asarray(bg.convert('RGB')), cv2.COLOR_RGB2BGR)
    return definitions, templates


@lru_cache(maxsize=1)
def voucher_descriptions():
    from item_identity import _brace_end
    with zipfile.ZipFile(ROOT / 'Balatro' / 'Balatro.exe') as archive:
        source = archive.read('localization/en-us.lua').decode('utf8')
        game = archive.read('game.lua').decode('utf8')
    catalogue = extra_catalogue()[0]
    result = {}
    for match in re.finditer(r'(?m)^\s*(v_\w+)\s*=\s*\{', source):
        key = match[1]
        if key not in catalogue:
            continue
        start = source.index('{', match.start())
        block = source[start:_brace_end(source, start) + 1]
        text = re.search(r'text\s*=\s*\{', block)
        if not text:
            continue
        a = block.index('{', text.start())
        lines = re.findall(r'"((?:[^"\\]|\\.)*)"', block[a:_brace_end(block, a) + 1])
        definition = next(line for line in game.splitlines() if re.match(r'\s*' + key + r'\s*=', line))
        display = re.search(r'extra_disp\s*=\s*(\d+)', definition)
        extra = re.search(r'extra\s*=\s*(\d+(?:\.\d+)?)', definition)
        value = float(display[1] if display else extra[1]) if (display or extra) else 1
        if key in ('v_seed_money', 'v_money_tree'):
            value /= 5  # ability.extra is the cash threshold; tooltip shows interest dollars.
        body = re.sub(r'\{[^}]*\}', '', ' '.join(lines)).replace('#1#', f'{value:g}')
        result[key] = body
    return result


def _effect_words(text):
    return ' '.join(re.findall(r'[a-z]+|\d+(?:\.\d+)?', text.casefold()))


def identify_extra(labels, price, owned_vouchers=()):
    """Prove a name or an undiscovered effect with its numbers/prerequisites."""
    titles = {re.sub(r'\W+', '', label).casefold() for label in labels}
    matches = {(d['kind'], d['name']) for d in extra_catalogue()[0].values()
               if re.sub(r'\W+', '', d['name']).casefold() in titles}
    if len(matches) > 1:
        raise ObservationError('Multiple booster/voucher names are visible')
    if not matches:
        body = ' '.join(dict.fromkeys(labels))
        if not re.search(r'\bnot\s+disco[uv]ered\b', body, re.I):
            raise ObservationError('Booster/voucher title is missing or ambiguous')
        # The game hides undiscovered names, but exposes kind and exact pick
        # counts. Preserve both numbers to distinguish normal/Jumbo/Mega.
        packs = re.findall(r'\bchoose\s+([12])\s+of\s+up\s+to\s+([2-5])\s+'
                           r'(joker|tarot|planet|spectral|playing)\s+cards\b', body, re.I)
        kinds = {'joker': 'buffoon_pack', 'tarot': 'arcana_pack', 'planet': 'celestial_pack',
                 'spectral': 'spectral_pack', 'playing': 'standard_pack'}
        for choose, size, family in packs:
            matches.update((d['kind'], d['name']) for d in extra_catalogue()[0].values()
                if d['kind'] == kinds[family.lower()] and d['choose'] == int(choose) and d['size'] == int(size))
        if not packs:
            observed = ' ' + _effect_words(body) + ' '
            catalogue = extra_catalogue()[0]
            for key, description in voucher_descriptions().items():
                d = catalogue[key]
                requires = [joker_name(catalogue[k]['name']) for k in d['requires']]
                if joker_name(d['name']) in owned_vouchers or not all(k in owned_vouchers for k in requires):
                    continue
                expected = ' ' + _effect_words(description) + ' '
                if expected in observed:
                    matches.add((d['kind'], d['name']))
        if len(matches) != 1:
            raise ObservationError('Undiscovered booster/voucher effect is incomplete or ambiguous')
    kind, name = matches.pop()
    return ShopItem(kind, name, price)


def voucher_state(state, item):
    """Deterministic effects from Card:apply_to_run; prices remain observed."""
    key = joker_name(item.name)
    if key in state.owned_vouchers:
        raise ObservationError('Voucher already redeemed')
    changes = dict(money=state.money - item.price,
                   owned_vouchers=tuple(state.owned_vouchers) + (key,))
    if key in ('grabber', 'nacho_tong'):
        changes.update(base_hands_per_round=state.base_hands_per_round + 1,
                       hands_left=state.hands_left + 1)
    elif key in ('wasteful', 'recyclomancy'):
        changes.update(base_discards_per_round=state.base_discards_per_round + 1,
                       discards_left=state.discards_left + 1)
    elif key in ('paint_brush', 'palette'):
        changes['hand_size'] = state.hand_size + 1
    elif key == 'antimatter':
        changes['total_joker_slots'] = state.total_joker_slots + 1
    elif key == 'crystal_ball':
        changes['consumable_slots'] = state.consumable_slots + 1
    elif key in ('seed_money', 'money_tree'):
        changes['interest_cash_cap'] = 50 if key == 'seed_money' else 100
    elif key == 'observatory':
        changes['observatory'] = True
    elif key in ('hieroglyph', 'petroglyph'):
        changes.update(ante=max(0, state.ante - 1), upcoming_boss='',
                       upcoming_boss_target=0, upcoming_boss_ante=0)
        if key == 'hieroglyph':
            changes.update(base_hands_per_round=max(1, state.base_hands_per_round - 1),
                           hands_left=max(1, state.hands_left - 1))
        else:
            changes.update(base_discards_per_round=max(0, state.base_discards_per_round - 1),
                           discards_left=max(0, state.discards_left - 1))
    return replace(state, **changes)


def extra_is_worth_considering(item, state, forecast):
    key = joker_name(item.name)
    left = state.money - item.price
    if left < 0:
        return False
    if item.kind == 'voucher':
        if key in state.owned_vouchers or key in DEFERRED_VOUCHERS:
            return False
        immediate = key in ('grabber', 'nacho_tong', 'paint_brush', 'palette',
                            'wasteful', 'recyclomancy')
        # Scoring/resource upgrades may spend the reserve to rescue a run.
        if forecast.weak and not immediate and left < 15:
            return False
        if key == 'blank':
            return left >= 40 and not forecast.weak
        if key in ('seed_money', 'money_tree'):
            return left >= state.interest_cash_cap and state.ante <= 6
        return left >= (0 if immediate else 15)
    if (state.ante >= 8 and state.blind_index == 1 and forecast.weak
            and item.kind in ('celestial_pack', 'arcana_pack')):
        # There is no later shop before the goal. When known upgrades/search
        # cannot rescue the Boss, an affordable usable pack may spend the
        # usual interest reserve. Contents stay unknown in the score forecast.
        return True
    if item.kind == 'buffoon_pack':
        return len(state.jokers) < state.total_joker_slots and left >= (
            0 if not state.jokers else 6)
    # Unopened contents are unknown; keep purchasing money for the weak engine.
    if not state.jokers and forecast.weak:
        return False
    if state.has('bull') or state.has('bootstraps'):
        return left >= 25
    if item.kind == 'standard_pack':
        # Feeding an observed Hologram is a supported investment. A safe
        # engine may pay one interest tier; weak engines still need the cash
        # for a known rescue, and cash-scoring Jokers were protected above.
        reserve = 20 if state.has('hologram') and not forecast.weak and state.ante < 8 else 25
        return left >= reserve and (state.has('hologram') or bool(state.deck_cards))
    if item.kind == 'spectral_pack':
        return left >= 25 and not forecast.weak
    return left >= (10 if forecast.weak else 20)


def sealed_pack_value(item, state, forecast):
    """Opportunity value, not a forecast containing imaginary pack rewards."""
    base = {'buffoon_pack': 13 if forecast.weak else 7,
            'celestial_pack': 8 + min(5, state.effect_count('constellation') * 2),
            'arcana_pack': 7 + min(4, state.effect_count('fortune_teller')),
            'spectral_pack': 5, 'standard_pack': 4 + 5 * state.has('hologram')}[item.kind]
    if item.kind == 'celestial_pack' and 'telescope' in state.owned_vouchers:
        base += 4
    if item.name.lower().startswith('mega '):
        base *= 1.6
    elif item.name.lower().startswith('jumbo '):
        base *= 1.15
    if state.ante < 8 and not forecast.weak:
        count = 2 if item.name.lower().startswith('mega ') else 1
        key, step = {'celestial_pack': ('constellation', .1),
                     'standard_pack': ('hologram', .25)}.get(item.kind, ('', 0))
        ratio = 1.
        for j in state.abilities:
            if j is not None and j.key == key:
                ratio *= 1 + count * step / max(1., j.value)
        # Marginal feeding value falls as X grows. This is an opportunity
        # premium, never an invented pack card in the survival forecast.
        base += min(12., (ratio - 1) * 24 * min(1., (8-state.ante)/3))
    return base


def offer_tooltip_regions(image, center, *, hand=False):
    """Isolate nearby grey panels; exclude the hovered card and distant HUD."""
    x, y = center
    left, top, right, bottom = 530, (130 if hand else 330), 2040, 1160
    hsv = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 1] < 45) & (hsv[:, :, 2] > 75)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    _, _, components, _ = cv2.connectedComponentsWithStats(mask)
    regions = []
    for bx, by, w, h, area in components[1:]:
        if not (140 < w < 650 and 55 < h < 550 and area > 3000):
            continue
        bx, by = int(bx) + left, int(by) + top
        if bx <= x <= bx + w and by <= y <= by + h:
            continue
        dx = max(bx - x, 0, x - bx - w)
        dy = max(by - y, 0, y - by - h)
        if dx > 430 or dy > 430:
            continue
        region = (max(left, bx - 12), max(top, by - 85),
                  min(right, bx + int(w) + 12), min(bottom, by + int(h) + 40))
        regions.append((dx + dy, region))
    if not regions:
        raise ObservationError('Nearby offer tooltip is not visible')
    return [region for _, region in sorted(regions)[:3]]


def offer_labels(image, center, *, hand=False):
    from perception import read_text, tooltip_rows
    try:
        labels = [line for region in offer_tooltip_regions(image, center, hand=hand)
                  for line in tooltip_rows(read_text(image, region))]
    except ObservationError:
        labels = []
    if hand or any(re.search(r'not disc(?:overed|ouered)', line, re.I) for line in labels):
        if not labels:
            raise ObservationError('Nearby offer tooltip is not visible')
        return labels
    # Raised white playing cards can join the tooltip mask into one oversized
    # component. Read its actual heading, then retain only centered consecutive
    # body rows. An adjacent enhancement explanation is a separate column.
    x, y = center
    words = read_text(image, (max(530, int(x - 420)), max(330, int(y - 550)),
                              min(2040, int(x + 420)), max(385, int(y - 100))))
    anchors = [(text, score, p) for text, score, p in words
               if score > .9 and re.fullmatch(r'Not Disc(?:overed|ouered)', text.strip(), re.I)
               and abs(p[0] - x) < 200]
    if len(anchors) == 1:
        heading, _, (ax, ay) = anchors[0]
        rows = []
        for word in sorted(words, key=lambda w: (w[2][1], w[2][0])):
            if (word[1] < .8 or abs(word[2][0] - ax) >= 190
                    or not (ay + 20 < word[2][1] < min(y - 100, ay + 350))):
                continue
            if not rows or word[2][1] - rows[-1][0][2][1] > 16:
                rows.append([word])
            else:
                rows[-1].append(word)
        body, last_y = [], ay
        for row in rows:
            row_x = sum(len(t) * p[0] for t, _, p in row) / max(1, sum(len(t) for t, _, _ in row))
            row_y = sum(p[1] for _, _, p in row) / len(row)
            if abs(row_x - ax) > 70:
                continue
            if row_y - last_y > 60:
                break
            body.extend(tooltip_rows(row))
            last_y = row_y
        if len(body) >= 2:
            return [heading, *body]
    if not labels:
        raise ObservationError('Nearby offer tooltip is not visible')
    return labels


def collect_extra_offers(session, image):
    from perception import read_text, read_offer_price
    from runtime_issues import record_issue
    words = read_text(image, (810, 750, 1770, 890))
    prices = []
    for text, confidence, (x, py) in words:
        if not (confidence > .8 and re.fullmatch(r'\$\d+', text.strip())
                and 810 <= x <= 1770 and 750 <= py <= 890):
            continue
        try:
            price = read_offer_price(image, text, confidence, (x, py))
        except ObservationError as error:
            if getattr(session, 'pending_action', None):
                raise
            record_issue(session, 'extra_offer_price_unreadable', level='degraded',
                code='optional_offer_price_unreadable', message='Skip an unverified optional shop price',
                error=str(error), text=text, position=(x, py))
            continue
        if price != int(text.strip()[1:]):
            session.log('shop_price_recovered', original=text, price=price, position=(x, py))
        prices.append((price, (x, py)))
    prices.sort(key=lambda row: row[1][0])
    cache = getattr(session, 'extra_offer_cache', None)
    if cache and len(prices) == len(cache['prices']) and all(
            price == old_price and abs(pos[0] - old_pos[0]) < 8 and abs(pos[1] - old_pos[1]) < 8
            for (price, pos), (old_price, old_pos) in zip(prices, cache['prices'])):
        # Only upper offers reroll. The caller clears this cache on shop entry
        # and before any lower-row transaction; inflation/discount price
        # changes also force fresh tooltip identities here.
        session.log('extra_offers_reused', count=len(cache['items']))
        return list(cache['items']), list(cache['positions'])
    items, positions = [], []
    for price, (x, py) in prices:
        # The price labels locate the three lower slots even with CRT shaders.
        # Their artwork is distorted differently from ordinary Joker cards;
        # do not lower sprite confidence or use a false atlas match as identity.
        center = (x, py + 165)
        session.hover(center[0] / 2064, center[1] / 1191, 'read booster/voucher title')
        try:
            item = identify_extra(offer_labels(session.observe('extra-offer-tooltip'), center),
                                  price, getattr(session, 'policy_state', RunState()).owned_vouchers)
        except ObservationError as error:
            session.log('extra_offer_unreadable', price=price, error=str(error))
            continue
        items.append(item)
        positions.append(center)
        session.log('extra_offer_verified', item=asdict(item), position=center)
    session.hover(.93, .55, 'dismiss extra offer tooltip')
    session.extra_offer_cache = dict(prices=prices, items=items, positions=positions) if len(items) == len(prices) else None
    return items, positions


@dataclass(frozen=True)
class PackView:
    kind: str
    remaining: int
    skip: tuple[int, int]
    title_y: int


def read_pack_view(image):
    from perception import read_text, tooltip_rows
    words = read_text(image, (550, 350, 2040, 1165))
    reliable = [w for w in words if w[1] > .85]
    rows = tooltip_rows(reliable)
    titles = [re.sub(r'\s+', ' ', t).strip()
              for t in rows + [t for t, _, _ in reliable]]
    kinds = [k + '_pack' for k in PACK_TYPES if any(
        re.search(r'\b(?:Jumbo |Mega )?' + k + r' Pack\b', r, re.I) for r in titles)]
    skip = [p for text, score, p in words if text.strip().lower() == 'skip' and score > .85]
    counts = {int(m[1]) for row in titles if (m := re.fullmatch(r'Choose\s+([12])', row, re.I))}
    for text, _, (x, y) in reliable:
        if text.strip().casefold() == 'choose':
            counts.update(int(t.strip()) for t, _, (nx, ny) in reliable
                          if re.fullmatch('[12]', t.strip()) and 0 < nx - x < 180 and abs(ny - y) < 25)
            if not counts:
                # Whole-panel OCR drops the isolated pixel-font 1. Its exact
                # line is read separately, without inferring the count from
                # the purchase plan or from the unrelated deck counter.
                region = (int(x + 80), int(y - 27), int(x + 160), int(y + 28))
                counts.update(int(t.strip()) for t, score, _ in read_text(image, region, line=True)
                              if score > .9 and re.fullmatch('[12]', t.strip()))
    if len(kinds) != 1 or len(skip) != 1 or len(counts) != 1:
        return None
    title_y = [p[1] for text, score, p in words
               if 'pack' in text.lower() and score > .8]
    return PackView(kinds[0], counts.pop(), skip[0], max(title_y, default=skip[0][1]))


def shop_ready(image):
    """The background SHOP sign alone cannot prove a booster has closed."""
    from perception import read_text, tooltip_rows
    words = read_text(image, (550, 360, 915, 710))
    reliable = [w for w in words if w[1] > .85]
    if any(re.fullmatch(r'Next\s+Round', row.strip(), re.I) for row in tooltip_rows(reliable)):
        return True
    # This button usually prints Next and Round on two vertical lines.
    pairs = [(np, rp) for nt, _, np in reliable for rt, _, rp in reliable
             if nt.strip().casefold() == 'next' and rt.strip().casefold() == 'round'
             and abs(np[0] - rp[0]) < 30 and 15 < rp[1] - np[1] < 70]
    return len(pairs) == 1


def extra_outcome_values(image, pending):
    """Read transaction evidence without issuing a second click."""
    from perception import read_text
    view = read_pack_view(image)
    values = {}
    if view:
        values.update(pack_kind=view.kind, pack_remaining=view.remaining)
    if pending['kind'] == 'redeem':
        x = pending['details']['offer_x']
        prices = [t for t, score, _ in read_text(image, (int(x - 90), 750, int(x + 90), 855))
                  if score > .8 and re.fullmatch(r'\$\d+', t.strip())]
        values['voucher_price_visible'] = bool(prices)
    return values


def verified_button(image, label, center):
    from perception import read_text
    x, y = center
    words = read_text(image, (max(530, int(x - 190)), max(330, int(y - 120)),
                             min(2040, int(x + 240)), min(1185, int(y + 250))))
    matches = [p for text, score, p in words
               if text.strip().lower() == label.lower() and score > .85]
    if len(matches) != 1:
        raise ObservationError(f'{label} button is missing or ambiguous')
    return matches[0]


def verify_offer_selection(session, image, center, label, check_identity):
    """Re-read transient selected tooltips without repeating a purchase/use."""
    for attempt in range(3):
        try:
            check_identity(offer_labels(image, center))
            return verified_button(image, label, center)
        except ObservationError as error:
            session.log('extra_selection_read_retry', attempt=attempt + 1, error=str(error))
            if attempt == 2:
                raise
            session.hover(.93, .55, 'dismiss transient extra tooltip')
            session.hover(center[0] / 2064, center[1] / 1191, 'refresh selected extra tooltip')
            image = session.observe('extra-selected-retry')


def purchase_extra(session, item, center, state):
    session.extra_offer_cache = None
    session.tap(center[0] / 2064, center[1] / 1191, 'select named extra offer')
    selected = session.observe('extra-selected')
    def check_identity(labels):
        actual = identify_extra(labels, item.price, state.owned_vouchers)
        if (actual.kind, actual.name) != (item.kind, item.name):
            raise ObservationError('Selected extra changed before purchase')
    label = 'Redeem' if item.kind == 'voucher' else 'Open'
    button = verify_offer_selection(session, selected, center, label, check_identity)
    after = voucher_state(state, item) if item.kind == 'voucher' else replace(state,
        money=state.money - item.price,
        pack_hand_from_full_deck=(item.kind in ('arcana_pack', 'spectral_pack')
                                 and state.draw_pile_size == state.deck_size))
    begin_action(session, 'redeem' if item.kind == 'voucher' else 'open_pack',
                 {'money': state.money}, after_state=after, name=item.name,
                 price=item.price, pack_kind=item.kind, offer_x=center[0])
    session.tap(button[0] / 2064, button[1] / 1191, f'{label} verified {item.name}')
    confirm_pending(session)
    if item.kind != 'voucher':
        handle_pack(session, session.observe('opened-pack'))
    return session.policy_state


@dataclass(frozen=True)
class PackChoice:
    item: ShopItem
    center: tuple[int, int]
    card: Card | None = None


def standard_centers(image, bottom):
    """Card-shaped light rectangles are locators; tooltips prove identity."""
    left, top, right = 550, 700, 1670
    crop = image[top:bottom, left:right]
    mask = (crop.min(axis=2) > 165).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    _, _, components, _ = cv2.connectedComponentsWithStats(mask)
    centers = []
    for x, y, w, h, area in components[1:]:
        if 85 < w < 210 and 125 < h < 295 and .5 < w / h < .85 and area > w * h * .5:
            centers.append((int(left + x + w / 2), int(top + y + h / 2)))
    return sorted(centers)


def pack_choice_locators(image, view):
    from perception import _match_art, consumable_templates, joker_templates
    # Real reward centers are y=853..870 in saved frames. Exclude the raised
    # hand above and the deck back at x>1680, both former false candidates.
    bottom = min(1030, view.title_y - 30)
    if bottom <= 700:
        raise ObservationError('Pack card area is not readable')
    if view.kind == 'standard_pack':
        return [(None, center) for center in standard_centers(image, bottom)]
    templates = joker_templates() if view.kind == 'buffoon_pack' else consumable_templates()[0]
    return [(key, center) for _, key, center in _match_art(image, (550, 700, 1670, bottom), templates)]


def collect_pack_choices(session, image, view):
    from perception import shop_offer_item, parse_card_tooltip, supported_shop_item, read_jokers
    locators = pack_choice_locators(image, view)
    choices = []
    for hint, center in locators:
        for attempt in range(2):
            session.hover(center[0] / 2064, center[1] / 1191, 'read pack choice')
            try:
                labels = offer_labels(session.observe('pack-choice-tooltip'), center)
                if view.kind == 'standard_pack':
                    card = parse_card_tooltip(Card(2, 'S'), labels, require_identity=True)
                    item = ShopItem('card', f'{card.rank}{card.suit}', 0)
                else:
                    art = (min(read_jokers(image, (550, 700, 1670, 1100)),
                               key=lambda m: abs(m[2][0] - center[0]), default=(0, '', None))
                           if hint == 'misprint' else (0, hint or '', None))
                    item, _ = shop_offer_item(labels, 0, art[1],
                        require_supported=False, art_confidence=art[0])
                    if not supported_shop_item(item):
                        session.log('pack_choice_unsupported', position=center, name=item.name,
                            reason='identity verified; use workflow is not supported')
                        break
                    expected = 'joker' if view.kind == 'buffoon_pack' else (
                        'planet' if view.kind == 'celestial_pack' else None)
                    if expected and item.kind != expected and joker_name(item.name) != 'black_hole':
                        raise ObservationError('Pack choice type conflicts with pack')
                    if item.kind == 'joker' and hint and item.joker.key != hint:
                        raise ObservationError('Pack choice title and artwork disagree')
                    card = None
                choices.append(PackChoice(item, center, card))
                break
            except ObservationError as error:
                if str(error).startswith('Consumable needs an unsupported use workflow:'):
                    session.log('pack_choice_unsupported', position=center, reason=str(error))
                    break
                if attempt:
                    session.log('pack_choice_unreadable', position=center, error=str(error))
                else:
                    session.hover(.93, .55, 'dismiss pack choice tooltip before reread')
    session.hover(.93, .55, 'dismiss pack tooltip')
    return choices


def read_pack_hand(image, count, identities=None):
    """The hand above an Arcana/Spectral pack has a different vertical origin."""
    from vision import hand_slots, match_at, full_card_templates, special_corner_templates, card_templates
    if not 1 <= count <= 30:
        raise ObservationError('Pack hand count is outside the supported range')
    found = []
    for index, x in enumerate(hand_slots(count, count)):
        templates = full_card_templates()
        if identities is not None:
            card = identities[index]
            if card.is_stone:
                templates = {c: t for c, t in special_corner_templates().items() if c.is_stone}
            else:
                base = Card(card.rank, card.suit)
                templates = {base: templates[base]}
        best = match_at(image, x, 490, radius=35, vertical=120, templates=templates)[0]
        if best[0] < .53:
            best = match_at(image, x, 490, radius=35, vertical=120,
                            templates=templates, angles=(0, -8, -4, 4, 8))[0]
        if best[0] < .50 and identities is not None and not identities[index].is_stone:
            # Gold/foil backgrounds can obscure the larger white template.
            # Keep the already proven rank/suit and the same confidence gate,
            # but compare a smaller corner containing fewer background pixels.
            base = Card(identities[index].rank, identities[index].suit)
            small = match_at(image, x, 490, radius=35, vertical=120,
                             templates={base: card_templates()[base]}, angles=(0, -8, -4, 4, 8))[0]
            if small[0] > best[0]:
                best = small
        if best[0] < .50 and identities is None:
            special = match_at(image, x, 490, radius=35, vertical=120,
                               templates=special_corner_templates(), angles=(0, -8, -4, 4, 8))[0]
            if special[0] > best[0]:
                best = special
        score, card, _, (px, py) = best
        if score < .50 or card.face_down:
            raise ObservationError(f'Pack hand corner {index} is unreadable')
        found.append((card, score, (max(0, x - 35) + px, 370 + py)))
    return found


def collect_pack_hand(session, image, state):
    from perception import read_numbers, parse_card_tooltip
    for attempt in range(3):
        values = read_numbers(image)
        total, pile = values.get('deck_size'), values.get('draw_pile_size')
        count = total - pile if total is not None and pile is not None else 0
        try:
            found = read_pack_hand(image, count)
            break
        except ObservationError:
            if attempt == 2:
                raise
            image = session.observe('pack-hand-ready-retry')
    enriched = []
    for index, (hint, confidence, (x, y)) in enumerate(found):
        for attempt in range(3):
            session.hover((x + 30) / 2064, (y + 90) / 1191, 'read pack hand card')
            frame = session.observe('pack-hand-tooltip')
            try:
                labels = offer_labels(frame, (x + 30, y + 90), hand=True)
                card = parse_card_tooltip(hint, labels, require_identity=True)
                break
            except ObservationError:
                if attempt == 2:
                    raise
                session.hover(.93, .55, 'dismiss pack hand tooltip before retry')
        enriched.append((card, confidence, (x, y)))
        session.log('pack_hand_card_observed', index=index, card=asdict(card), tooltip=labels)
    session.hover(.93, .55, 'dismiss pack hand tooltip')
    # A provisional artwork match may locate a pip inside the card instead of
    # its corner. After tooltips prove the identities, relocate those exact
    # corners on an unobstructed frame before measuring selection lift.
    for attempt in range(3):
        try:
            baseline = read_pack_hand(session.observe('pack-hand-verified'), count,
                                      identities=[c for c, _, _ in enriched])
            break
        except ObservationError:
            if attempt == 2:
                raise
    return [(card, score, position)
            for (card, _, _), (_, score, position) in zip(enriched, baseline)]


def select_pack_targets(session, found, indices):
    """Verify lift of exactly the requested cards in the elevated pack hand."""
    desired, actual = set(indices), set()
    if len(desired) > 5 or any(i < 0 or i >= len(found) for i in desired):
        raise ObservationError('Invalid pack target indices')
    current = found
    for attempt in range(3):
        for index in sorted(desired ^ actual, reverse=True):
            x, y = current[index][2]
            session.hover(.93, .55, 'dismiss pack target tooltip')
            session.tap((x + 30) / 2064, (y + 90) / 1191, 'select verified pack hand target')
        session.hover(.93, .55, 'move off pack targets')
        frame = session.observe('pack-targets-selected')
        current = read_pack_hand(frame, len(found), identities=[c for c, _, _ in found])
        if any(new[0].identity != old[0].identity for new, old in zip(current, found)):
            raise ObservationError('Pack target hand identities changed before use')
        lifts = [old[2][1] - new[2][1] for new, old in zip(current, found)]
        actual = {i for i, lift in enumerate(lifts) if 28 < lift < 100}
        if actual == desired and all(abs(lift) < 20 for i, lift in enumerate(lifts) if i not in desired):
            session.log('pack_targets_verified', indices=sorted(desired))
            return
    raise ObservationError('Pack target selection could not be verified')


def choose_pack_reward(choices, state, hand=()):
    """Compare permanent upgrades and observed engine synergy at zero price."""
    from planning import shop_forecast, deck_target_value, full_deck
    from strategy import _purchase_state, _shop_round_state, joker_hand_size_delta
    from consumables import Consumable, ConsumableDecision, decide_consumable, state_after_consumable
    before = shop_forecast(_shop_round_state(state))
    candidates = []
    for choice in choices:
        item = choice.item
        plan = None
        if item.kind == 'joker':
            if len(state.jokers) >= state.total_joker_slots + (item.joker.edition == 'negative'):
                continue
            after = _purchase_state(item, state)
            # Purchases in a shop don't persist simulated planet use; Jokers do.
            utility = 1
        elif item.kind == 'card':
            deck = full_deck(state)
            average = sum(deck_target_value(c, state) for c in deck) / max(1, len(deck))
            value = deck_target_value(choice.card, state)
            utility = value - average
            if state.has('hologram'):
                utility += 8
            if utility <= 0:
                continue  # avoid diluting a developed deck with an ordinary card
            jokers = tuple(replace(j, value=max(1, j.value) + .25)
                           if j.key == 'hologram' and not j.debuffed else j for j in state.jokers)
            after = replace(state, deck_size=state.deck_size + 1, jokers=jokers,
                deck_cards=state.deck_cards + (choice.card,) if state.deck_cards else ())
        else:
            consumable = Consumable(item.name, item.kind)
            if consumable.key in PLANET_HAND or consumable.key == 'black_hole':
                plan = ConsumableDecision('use', reason='permanent pack hand-level upgrade')
            elif hand:
                # Pack choices expire now; don't pretend they occupy a held slot.
                tactical = replace(state, target_score=0, round_score=0, boss='',
                    boss_seen_types=frozenset(), round_played_types={}, forced_card_index=-1,
                    consumables_held=state.consumable_slots)
                plan = decide_consumable(consumable, hand, tactical)
            else:
                # These don't need hand targets; supply an observed-context plan.
                key = consumable.key
                if key == 'the_hermit' and state.money > 0:
                    plan = ConsumableDecision('use', utility=min(20, state.money))
                elif key == 'temperance' and sum(j.sell_value for j in state.jokers) > 0:
                    plan = ConsumableDecision('use', utility=min(50, sum(j.sell_value for j in state.jokers)))
                elif key in ('judgement', 'the_soul') and len(state.jokers) < state.total_joker_slots:
                    plan = ConsumableDecision('use', utility=12, deterministic=False)
            if plan is None or plan.action != 'use':
                continue
            # Death reordering uses the normal hand drag verifier, not a guess.
            if plan.play_order:
                continue
            after = state_after_consumable(state, hand, consumable, plan)
            after = replace(after, consumables_held=state.consumables_held,
                            planet_cards=dict(state.planet_cards))
            if consumable.key == 'the_hermit':
                after = replace(after, money=state.money + min(20, state.money))
            elif consumable.key == 'temperance':
                after = replace(after, money=state.money + min(50, sum(j.sell_value for j in state.jokers)))
            utility = max(0, plan.utility) * .03
        projected = shop_forecast(_shop_round_state(after))
        gain = 30 * math.log2((projected.next_round.safe_score + 20) / (before.next_round.safe_score + 20))
        gain += 15 * math.log2((projected.sustainable_score + 20) / (before.sustainable_score + 20))
        if projected.boss_round and before.boss_round:
            gain += 20 * math.log2((projected.boss_round.safe_score + 20) / (before.boss_round.safe_score + 20))
        if item.kind == 'planet' and gain < .01 and not state.has('constellation'):
            utility += state.played(PLANET_HAND.get(joker_name(item.name), '')) * .01
        repairs = before.coverage < 1 <= projected.coverage
        candidates.append(((repairs, gain + utility), choice, plan, after))
    return max(candidates, key=lambda row: row[0])[1:] if candidates else None


def plan_pack_reward(session, image, view, choices):
    from consumables import ENHANCEMENT_TAROTS, SUIT_TAROTS, SEAL_SPECTRALS
    targeted = set(ENHANCEMENT_TAROTS) | set(SUIT_TAROTS) | set(SEAL_SPECTRALS)
    targeted.update(('strength', 'death', 'the_hanged_man', 'cryptid'))
    state = session.policy_state
    found = []
    if (view.kind in ('arcana_pack', 'spectral_pack') and state.pack_hand_from_full_deck
            and any(joker_name(c.item.name) in targeted for c in choices)):
        try:
            found = collect_pack_hand(session, image, state)
        except ObservationError as error:
            # Targeted rewards are optional. Without a verified hand, compare
            # the independently verified targetless rewards (or skip); never
            # press Use on guessed targets or abort the whole game here.
            session.log('pack_hand_unavailable', error=str(error),
                        reason='evaluate verified rewards that require no hand targets')
    reward = choose_pack_reward(choices, state, tuple(c for c, _, _ in found))
    return reward, found


def handle_pack(session, image):
    from perception import shop_offer_item, parse_card_tooltip, read_numbers
    from controller import collect_jokers
    blocked_rewards = set()
    for attempt in range(8):  # up to five rejected optional rewards plus two picks
        view = read_pack_view(image)
        for ready_attempt in range(3):
            if view is not None:
                break
            if attempt > 0 and shop_ready(image):
                return
            image = session.observe('pack-view-ready-retry')
            view = read_pack_view(image)
        if view is None:
            raise ObservationError('Opened pack is not readable')
        state = session.policy_state
        choices = collect_pack_choices(session, session.observe('pack-options'), view)
        choices = [c for c in choices if (c.item.kind, c.item.name) not in blocked_rewards]
        # Inspect the cards only after learning which rewards need targets.
        reward, found = plan_pack_reward(session, session.observe('pack-hand-context'), view, choices)
        if reward is not None and reward[1] and reward[1].indices:
            try:
                select_pack_targets(session, found, reward[1].indices)
            except ObservationError as error:
                current_view = read_pack_view(session.observe('pack-target-fallback'))
                if (current_view is None or current_view.kind != view.kind
                        or current_view.remaining != view.remaining):
                    raise ObservationError('Pack changed during target verification') from error
                view = current_view
                session.log('pack_targets_unavailable', error=str(error),
                            reason='evaluate verified rewards that require no hand targets')
                # No Use click has occurred. Partial target selection is harmless
                # to Hermit/planets and to Skip, which do not consume hand targets.
                reward = choose_pack_reward(choices, state)
                found = []
        session.log('pack_decision', pack=view.kind, remaining=view.remaining,
                    choices=[asdict(c) for c in choices],
                    chosen=reward[0].item.name if reward else 'skip')
        if reward is None:
            begin_action(session, 'skip_pack', {'pack_remaining': view.remaining},
                         after_state=replace(state, pack_hand_from_full_deck=False),
                         pack_kind=view.kind)
            session.tap(view.skip[0] / 2064, view.skip[1] / 1191, 'skip unusable pack choices')
            confirm_pending(session)
            return
        choice, plan, after = reward
        # Transformations can reorder raised cards; random Joker creation can
        # also alter hand capacity. Further Mega picks use targetless effects
        # until that changed hand layout has an independent verification path.
        if view.remaining == 1 or view.kind in ('arcana_pack', 'spectral_pack'):
            after = replace(after, pack_hand_from_full_deck=False)
        x, y = choice.center
        session.tap(x / 2064, y / 1191, 'select verified pack reward')
        selected = session.observe('pack-reward-selected')
        def check_identity(labels):
            if choice.card:
                actual_card = parse_card_tooltip(choice.card, labels, require_identity=True)
                if actual_card != choice.card:
                    raise ObservationError('Selected playing card changed')
            else:
                from perception import read_jokers
                art = (min(read_jokers(selected, (550, 550, 1670, 1100)),
                           key=lambda m: abs(m[2][0] - choice.center[0]), default=(0, '', None))
                       if choice.item.name == 'Misprint' else (0, '', None))
                actual, _ = (shop_offer_item(labels, 0, art[1], art_confidence=art[0])
                             if choice.item.name == 'Misprint' else shop_offer_item(labels, 0))
                if (actual.kind, actual.name, actual.joker) != (choice.item.kind, choice.item.name, choice.item.joker):
                    raise ObservationError('Selected pack reward differs from the plan')
        try:
            button = verify_offer_selection(session, selected, choice.center,
                'Select' if choice.item.kind in ('joker', 'card') else 'Use', check_identity)
        except ObservationError as error:
            if str(error) in ('Selected pack reward differs from the plan',
                              'Selected playing card changed'):
                raise  # positive evidence of a changed identity needs reconciliation
            # Nothing has been used yet. An optional reward can be abandoned
            # while the same independently verified pack is still open.
            session.hover(.93, .55, 'dismiss unavailable pack reward')
            image = session.observe('pack-reward-unavailable')
            current = read_pack_view(image)
            if current is None or (current.kind, current.remaining) != (view.kind, view.remaining):
                raise ObservationError('Pack changed before optional reward fallback') from error
            # Selection can raise/recenter a card. Blocking its identity, not
            # its old coordinates, prevents another cycle on the same reward.
            blocked_rewards.add((choice.item.kind, choice.item.name))
            from runtime_issues import record_issue
            record_issue(session, 'pack_reward_unavailable', level='degraded',
                code='optional_pack_reward_unavailable', message='Evaluate another verified pack reward',
                name=choice.item.name, error=str(error),
                reason='no Use/Select click occurred; evaluate another verified reward or Skip')
            continue
        levels_changed = choice.item.kind == 'planet' or joker_name(choice.item.name) == 'black_hole'
        begin_action(session, 'pick_pack', {'pack_remaining': view.remaining,
                     'joker_count': len(state.jokers), 'deck_size': state.deck_size}, after_state=after,
                     name=choice.item.name, pack_kind=view.kind, reward_kind=choice.item.kind,
                     levels_changed=levels_changed)
        session.tap(button[0] / 2064, button[1] / 1191, 'take verified pack reward')
        confirm_pending(session)
        image = session.observe('pack-after-reward')
        # Random creation/destruction and growth are read back from the game.
        predicted = session.policy_state
        state = collect_jokers(session, image, predicted)
        from strategy import joker_hand_size_delta
        state = replace(state, hand_size=max(1, predicted.hand_size +
                        joker_hand_size_delta(predicted.jokers, state.jokers)))
        numbers = read_numbers(image)
        if choice.item.kind == 'joker':
            key = choice.item.joker.key
            if (len(state.jokers) != len(predicted.jokers)
                    or sum(j.key == key for j in state.jokers) != sum(j.key == key for j in predicted.jokers)):
                raise ObservationError('Pack Joker receipt differs from the verified selection')
        if choice.card and numbers.get('deck_size') != predicted.deck_size:
            raise ObservationError('Standard pack did not confirm its playing-card deck increase')
        state = replace(state, **numbers)
        session.policy_state = state
        from recovery import atomic_json
        atomic_json(session.run_dir / 'policy-state.json', asdict(state))
    raise ObservationError('Pack still open after its allowed selections')
