"""Identify hidden shop names from the installed game's descriptions."""
from functools import lru_cache
from difflib import SequenceMatcher
from pathlib import Path
import re
import zipfile

from strategy import joker_name


def _brace_end(source, start):
    depth, quote, escaped = 0, None, False
    for i in range(start, len(source)):
        char = source[i]
        if quote:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == quote:
                quote = None
        elif char in ('"', "'"):
            quote = char
        elif char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                return i
    raise ValueError('Incomplete localization entry')


@lru_cache(maxsize=1)
def descriptions():
    from perception import joker_definitions, consumable_templates
    root = Path(__file__).resolve().parent
    with zipfile.ZipFile(root / 'Balatro' / 'Balatro.exe') as archive:
        source = archive.read('localization/en-us.lua').decode('utf-8')
    names = {joker_name(d.internal_name): key for key, d in joker_definitions().items()}
    names.update({joker_name(d.name): key for key, d in joker_definitions().items()})
    consumables = consumable_templates()[1]
    catalog = {}
    for match in re.finditer(r'(?m)^\s*((?:j|c)_\w+)\s*=\s*\{', source):
        start = source.index('{', match.start())
        block = source[start:_brace_end(source, start) + 1]
        name = re.search(r'name\s*=\s*["\']([^"\']+)["\']', block)
        text = re.search(r'text\s*=\s*\{', block)
        if name is None or text is None:
            continue
        key = joker_name(name[1])
        kind = 'joker' if match[1].startswith('j_') else 'consumable'
        key = names.get(key, key) if kind == 'joker' else key
        if kind == 'consumable' and key not in consumables:
            continue
        first = block.index('{', text.start())
        body = block[first:_brace_end(block, first) + 1]
        lines = re.findall(r'"((?:[^"\\]|\\.)*)"', body)
        suit = {'greedy_joker': 'Diamond', 'lusty_joker': 'Heart',
                'wrathful_joker': 'Spade', 'gluttonous_joker': 'Club'}.get(key)
        if suit:
            lines = [line.replace('#2#', suit) for line in lines]
        hand_requirement = {
            'jolly_joker': 'Pair', 'sly_joker': 'Pair', 'the_duo': 'Pair',
            'zany_joker': 'Three of a Kind', 'wily_joker': 'Three of a Kind',
            'the_trio': 'Three of a Kind', 'mad_joker': 'Two Pair',
            'clever_joker': 'Two Pair', 'spare_trousers': 'Two Pair',
            'crazy_joker': 'Straight', 'devious_joker': 'Straight', 'the_order': 'Straight',
            'droll_joker': 'Flush', 'crafty_joker': 'Flush', 'the_tribe': 'Flush',
            'the_family': 'Four of a Kind',
        }.get(key) if kind == 'joker' else None
        if hand_requirement:
            lines = [line.replace('#2#', hand_requirement) for line in lines]
        if kind == 'consumable':
            from consumables import SUIT_TAROTS, ENHANCEMENT_TAROTS
            from strategy import PLANET_HAND, HAND_ALIASES
            replacement = None
            if key in SUIT_TAROTS:
                replacement = {'D': 'Diamonds', 'C': 'Clubs', 'H': 'Hearts', 'S': 'Spades'}[SUIT_TAROTS[key]]
            elif key in ENHANCEMENT_TAROTS:
                replacement = ENHANCEMENT_TAROTS[key][0].title() + ' Card'
            elif key in PLANET_HAND:
                replacement = next(name.title() for name, category in HAND_ALIASES.items()
                                   if category == PLANET_HAND[key])
            if replacement:
                lines = [line.replace('#2#', replacement) for line in lines]
        catalog[(kind, key)] = lines
    return catalog


def tokens(text):
    # Observed OCR sometimes adds a stray letter before the word "hand".
    text = re.sub(r'\byhand\b', 'hand', text, flags=re.I)
    text = re.sub(r'\bsyit\b', 'suit', text, flags=re.I)
    # The red multiplier badge can turn X1.5 into K1.5 in a selected
    # tooltip. Normalize only this numeric Mult notation, not arbitrary K
    # words or the identifying words in an undiscovered item's effect.
    text = re.sub(r'\b[kK](?=\s*\d+(?:[.,]\d+)?\s+Mult\b)', 'X', text, flags=re.I)
    text = re.sub(r'\{[^}]*\}|#\d+#', ' ', text)
    text = re.split(r'\bcurrently\b', text, flags=re.I)[0]
    return re.findall(r'[a-z]+', text.lower())


def distinctive_joker_effect(lines):
    """Keep effect numbers that the general word matcher deliberately drops."""
    text = ' '.join(dict.fromkeys(lines)).casefold()
    text = re.sub(r'\bsyit\b', 'suit', text)
    matches = set()
    suit_keys = {'diamond': 'greedy_joker', 'heart': 'lusty_joker',
                 'spade': 'wrathful_joker', 'club': 'gluttonous_joker'}
    for suit, key in suit_keys.items():
        if re.search(r'played cards with\s+' + suit +
                     r'\s+suit give\s*\+\s*3\s+mult when scored', text):
            matches.add(key)
    if re.search(r'played cards with\s+club\s+suit give\s*\+\s*7\s+mult when scored', text):
        matches.add('onyx_agate')
    if (re.search(r'x\s*1[.,]5\s+mult when\s+(?:1\s+)?scored', text)
            and re.search(r'suit changes at\s+(?:t\s+)?end of round', text)):
        matches.add('ancient_joker')
    if (re.search(r'earn\s*\$\s*\d+\s+at end of round', text)
            and re.search(r'payout increases by\s*\$\s*2\s+when boss blind is defeated', text)):
        matches.add('rocket')
    if re.search(r'\b1\s+free reroll\s+per shop\b', text):
        matches.add('chaos_the_clown')
    # Both the complete growth condition and its amount distinguish this
    # from Mad Joker's fixed Two Pair bonus (a perfect substring match).
    if re.search(r'\bthis joker gains\s*\+\s*2\s+mult\s+if played hand contains\s+a two pair\b', text):
        matches.add('spare_trousers')
    # Hack has only four alphabetic tokens. Keep all four identifying ranks;
    # the generic description matcher deliberately removes these numbers.
    # 0r is an observed OCR form of "or" in this numeric row only.
    if re.search(r'\bretrigger\s+each played\s+2[.,\s]+3[.,\s]+4[.,\s]+(?:or|0r)\s+5\b', text):
        matches.add('hack')
    if len(matches) > 1:
        raise RuntimeError('Multiple distinct Joker effects are visible')
    if not matches:
        return None
    key = next(iter(matches))
    from perception import joker_definitions
    rarities = {line.strip().casefold() for line in lines
                if line.strip().casefold() in ('common', 'uncommon', 'rare', 'legendary')}
    if rarities and rarities != {joker_definitions()[key].rarity}:
        raise RuntimeError('Joker effect and rarity disagree')
    return key


def identify_description(lines, kind='joker'):
    short_candidates = []
    if kind == 'consumable':
        # Hermit's description has only four alphabetic tokens and cannot
        # enter the long-description fuzzy matcher. Require its complete
        # effect including the observed dollar cap instead of a loose hint.
        text = ' '.join(lines).lower()
        if re.search(r'\bdoubles\s+money\s*[\[(]\s*max\s+of\s*\$\s*20\s*[\])]', text):
            short_candidates.append((1., 'the_hermit'))
    if kind == 'joker':
        distinctive = distinctive_joker_effect(lines)
        if distinctive:
            return distinctive, 1.
        # These short effects have too few words for a safe fuzzy match. Keep
        # their distinguishing numbers instead of lowering the global margin.
        text = ' '.join(dict.fromkeys(lines)).lower()
        text = re.sub(r'\bhand\s+d\s+size\b', 'hand size', text)
        short_effects = {
            'juggler': r'\+\s*1\s+hand\s+size\b',
            'drunkard': r'\+\s*1\s+discard\s+each\s+round\b',
            'mystic_summit': r'\+\s*15\s+mult\s+when\s+0\s+discards\s+remaining\b',
        }
        matched = [key for key, pattern in short_effects.items() if re.search(pattern, text)]
        if len(matched) == 1:
            return matched[0], 1.
        if len(matched) > 1:
            raise RuntimeError('Multiple short joker descriptions are visible')
    # Repeated panels may share one row. Global row deduplication removed that
    # row from the later, readable paragraph (e.g. Mail-In Rebate's rank line).
    # Matching contiguous windows tolerates repeated complete paragraphs.
    observed = tokens(' '.join(lines))
    candidates = short_candidates
    for (candidate_kind, key), description in descriptions().items():
        if candidate_kind != kind:
            continue
        if key == 'baseball_card' and not any(
                observed[i:i + 2] == ['uncommon', 'jokers']
                for i in range(len(observed) - 1)):
            # A short fuzzy window must not drop this essential condition
            # and accept a different rarity as Baseball Card's identity.
            continue
        expected = tokens(' '.join(description))
        if len(expected) < 5:
            continue
        best = 0.
        for length in range(max(5, len(expected) - 3), len(expected) + 4):
            for start in range(max(1, len(observed) - length + 1)):
                fragment = observed[start:start + length]
                ratio = SequenceMatcher(None, expected, fragment, autojunk=False).ratio()
                best = max(best, ratio)
        candidates.append((best, key))
    candidates.sort(reverse=True)
    if candidates and candidates[0][0] >= .88:
        margin = candidates[0][0] - (candidates[1][0] if len(candidates) > 1 else 0)
        if margin >= .045:
            return candidates[0][1], candidates[0][0]
    raise RuntimeError('Item description is unreadable or ambiguous')
