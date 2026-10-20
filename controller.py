"""Visible information collection and verified actions for the policy."""
from __future__ import annotations

from dataclasses import asdict, replace
from collections import Counter
import re
import time

from perception import (read_text, observed_state, read_jokers, joker_counter,
                        joker_from_definition, parse_joker_tooltip,
                        parse_card_tooltip, parse_levels, joker_definitions,
                        consumable_counter, read_consumables, consumable_templates,
                        identify_joker_text, tooltip_labels, consumable_centers,
                        identify_consumable_text, tooltip_title)
from perception import read_hand_preview
from perception import joker_slots
from perception import hand_counter
from perception import tooltip_rows
from perception import joker_tooltip_words
from perception import verify_current_multiplier
from perception import bounded_temporary_joker
from perception import read_card_title
from consumables import decide_consumable, state_after_consumable
from types import SimpleNamespace
from strategy import Joker, RunState, joker_name, _boss
from vision import read_hand
from recovery import ObservationError, HandPreviewError, HandPreviewConflict, begin_action, finish_action
from runtime_issues import record_issue


def close_run_info(session, back):
    """Confirm the overlay was dismissed before trying to read a hand."""
    for attempt in range(3):
        session.tap(back[0] / 2064, back[1] / 1191, 'close Run Info')
        image = session.observe('after-run-info-close')
        try:
            if hand_counter(image)[0] > 0:
                session.log('run_info_closed', attempts=attempt + 1)
                return
        except ObservationError:
            pass
        words = read_text(image, (520, 110, 1600, 1100))
        back = next((p for text, score, p in words
                     if text.strip().lower() == 'back' and score > .85), None)
        if back is None:
            raise ObservationError('Run Info close outcome is unreadable; recollect before clicking')
        session.log('run_info_close_retry', attempt=attempt + 1)
        time.sleep(.5)
    raise ObservationError('Run Info remained open after three verified close attempts')


def collect_levels(session, state):
    from hand_levels import ORDINARY, sync_cache
    session.tap(.09, .73, "read Poker Hands levels in Run Info")
    levels, played, back = {}, {}, None
    ordinary = set(ORDINARY)
    # A probability trigger only invalidates its affected row. Other rows
    # remain authoritative, even when OCR misses them in the same panel.
    visible = ordinary | {k for k in state.uncertain_levels if state.played(k) > 0}
    required = (set(state.uncertain_levels) & visible if state.levels_initialized
                and not state.hand_history_dirty else ordinary | (set(state.uncertain_levels) & visible))
    if not state.levels_initialized:
        required |= ordinary
    try:
        for attempt in range(3):
            image = session.observe("poker-hand-levels")
            words = read_text(image)
            found_levels, found_played = parse_levels(words)
            levels.update(found_levels)
            played.update(found_played)
            back = next((p for text, score, p in words if text.strip().lower() == 'back' and score > .8), back)
            # Re-read individual rows with margins if whole-panel OCR missed them.
            if required - levels.keys():
                rows = read_text(image, (560, 335, 1500, 910))
                found_levels, found_played = parse_levels(rows)
                levels.update(found_levels)
                played.update(found_played)
            if required <= levels.keys() and back is not None:
                break
            session.log('hand_levels_retry', attempt=attempt + 1, missing=sorted(required - levels.keys()))
            time.sleep(.5)
        missing = required - levels.keys()
        cached = getattr(session, 'levels_cache', {})
        consistent = all(key not in cached or cached[key] == value for key, value in levels.items())
        if (missing and not getattr(session, 'levels_dirty', True) and consistent
                and required <= cached.keys() and not (missing & set(state.uncertain_levels))):
            levels.update({key: cached[key] for key in missing})
            session.log('hand_levels_cached', categories=sorted(missing), reason='previously verified levels; no level-changing action')
        if required - levels.keys():
            raise ObservationError('Missing observed hand levels: ' + ', '.join(sorted(required - levels.keys())))
        if back is None:
            raise ObservationError('Run Info Back button was not recognized')
    finally:
        if back is not None:
            close_run_info(session, back)
    session.log("hand_levels", levels=levels, played_types=played)
    from strategy import HAND_LEVELS
    updates = levels if not state.levels_initialized else {k: v for k, v in levels.items()
                                                        if k in required or k in state.uncertain_levels}
    state = replace(state, levels={**dict.fromkeys(HAND_LEVELS, 1), **state.levels, **updates},
                    played_types=dict(state.played_types, **played), levels_initialized=True,
                    uncertain_levels=tuple(k for k in state.uncertain_levels if k not in updates),
                    hand_history_dirty=False)
    sync_cache(session, state)
    return state


def collect_jokers(session, image, state, *, force=False):
    started = time.perf_counter()
    from joker_cache import (JokerCache, STABLE_WITHIN_BLIND, round_token,
                             roster_patch, cached_joker)
    for attempt in range(3):
        try:
            count, limit = joker_counter(image)
            break
        except ObservationError as error:
            session.log('joker_counter_read_retry', attempt=attempt + 1, error=str(error))
            if attempt == 2:
                raise
            time.sleep(.7)
            image = session.observe('joker-counter-retry')
    cache = getattr(session, 'joker_cache', None)
    if _boss(state) == 'amber_acorn':
        # The installed blind flips and shuffles the complete roster. There is
        # no name tooltip on a back; the pre-blind identities remain a multiset,
        # never a mapping from identities to physical slots.
        token = round_token(state)
        remembered = getattr(session, 'hidden_joker_roster', None)
        if remembered is None or remembered[0] != token:
            # A resumed hidden roster already includes its previous action
            # bounds. Do not apply those same plays/discards for a second time.
            offsets = ((sum(state.round_played_types.values()), state.round_discards_used)
                       if state.joker_order_unknown else (0, 0))
            remembered = (token, limit, tuple(state.jokers), *offsets)
            session.hidden_joker_roster = remembered
        known = remembered[2]
        same_count = len(known) == count and remembered[1] == limit
        if not same_count:
            # Ice Cream, Seltzer, Ramen or a consumable can remove a hidden
            # card. Its identity cannot be deduced from the count alone.
            known = tuple(Joker('Unknown hidden Joker', data=(('copyable', 0),))
                          for _ in range(count))
            session.hidden_joker_roster = (token, limit, known)
        base_played, base_discards = remembered[3:5] if len(remembered) >= 5 else (0, 0)
        played = max(0, sum(state.round_played_types.values()) - base_played)
        discards = (max(0, state.round_discards_used - base_discards)
                    if min(base_discards, state.round_discards_used) >= 0 else None)
        safe = []
        for joker in known:
            data, value = dict(joker.data), joker.value
            if joker.key == 'ice_cream':
                initial = value if joker.numeric_value_available else 0
                value = max(0, initial - 5 * played)
            elif joker.key == 'green_joker':
                value = (max(0, value + played - discards)
                         if joker.numeric_value_available and discards is not None else 0)
            elif joker.key == 'ramen':
                # A discard contains at most five cards. Unknown discarded
                # identities/counts cannot justify a more optimistic effect.
                value = (max(1, value - .05 * discards)
                         if joker.numeric_value_available and discards is not None else 1)
            elif joker.key == 'ride_the_bus':
                value = 0  # An intervening face-card score can reset it.
            elif joker.key in ('obelisk', 'hit_the_road'):
                value = 1
            elif joker.key == 'loyalty_card':
                data['remaining'] = -1
            elif joker.key == 'seltzer':
                data['remaining_hands'] = max(0, data.get('remaining_hands', 10) - played)
            if joker.key in ('ice_cream', 'green_joker', 'ramen', 'ride_the_bus', 'obelisk', 'hit_the_road'):
                data.pop('observed_value', None)
                data['counter_lower_bound'] = 1
            safe.append(replace(joker, value=value, data=tuple(data.items())))
        session.log('hidden_joker_roster', keys=[j.key for j in safe],
                    observed_count=count, remembered_count=len(remembered[2]),
                    roster_known=same_count and all(j.key != 'unknown_hidden_joker' for j in known),
                    order_known=False, reason='active Amber Acorn; preserve identities without guessing shuffled slots')
        return replace(state, jokers=tuple(safe), total_joker_slots=limit,
                       joker_order_unknown=bool(count), score_confidence='hidden_joker_order')
    session.hidden_joker_roster = None
    mapping = (cache.reconciled_slots(image, state, count)
               if not force and cache is not None else (None,) * count)
    verified = tuple(old is not None for old in mapping)
    reuse = any(verified)
    # Atlas matching is only a locator/hint. When a verified roster is intact,
    # its calibrated slot centres suffice for refreshing mutable tooltips.
    matches = ([(1. if verified[i] else 0., cache.jokers[mapping[i]].key if verified[i] else '', (x, 220))
                for i, x in enumerate(joker_slots(count))]
               if reuse else read_jokers(image, count=count))
    jokers = []
    reused, refreshed = [], []
    for confidence, key, (x, y) in matches:
        index = len(jokers)
        if reuse and verified[index] and not cache.needs_tooltip(mapping[index], state):
            jokers.append(cached_joker(cache.jokers[mapping[index]], state))
            reused.append(index)
            continue
        for attempt in range(3):
            session.hover(x / 2064, y / 1191, f"read Joker slot {len(jokers)} tooltip")
            tooltip = session.observe(f"joker-{key}")
            words = read_text(tooltip, (525, 330, 2020, 850))
            words, version_words = joker_tooltip_words(words, x)
            identity_labels = tooltip_labels(words)
            labels = identity_labels + [word[0] for word in version_words
                                        if word[0] not in identity_labels]
            try:
                key, name_confidence = identify_joker_text(identity_labels, key, title=tooltip_title(words))
                joker = joker_from_definition(key)
                effect_lines = None
                if key in ('popcorn', 'ice_cream', 'ramen'):
                    # Separate the main effect from neighbouring edition help
                    # panels, whose +10 Mult/+50 Chips are independent bonuses.
                    effect_lines = tooltip_rows([word for word in words if abs(word[2][0] - x) < 140])
                try:
                    joker = parse_joker_tooltip(joker, labels, effect_lines=effect_lines,
                                               image=tooltip, words=words)
                except ObservationError:
                    # The title has already been proven. After local rereads,
                    # a declining numeric effect can use a lower bound.
                    if attempt < 2 or key not in ('popcorn', 'ice_cream', 'ramen'):
                        raise
                    joker = bounded_temporary_joker(joker, labels)
                # Identity and current numeric value are separate proofs. A
                # readable name must not authorize an OCR-inflated XMult.
                old = [j for j in state.jokers if j.key == key]
                previous = old[0] if len(old) == 1 else None
                if key == 'madness' and not (reuse and verified[index]):
                    # A newly acquired same-name card does not inherit the
                    # destroyed/replaced physical card's monotonic counter.
                    previous = None
                joker = verify_current_multiplier(tooltip, words, joker, state, previous)
                if joker.get('counter_lower_bound'):
                    record_issue(session, 'joker_counter_bounded', level='degraded',
                        code='joker_effect_bounded', message='Continue using a verified scoring lower bound',
                        name=joker.name, value=joker.value,
                        reason='current effect unreadable or conflicting; explicit scoring lower bound')
                break
            except ObservationError:
                if attempt == 2:
                    raise
                session.hover(.93, .55, 'dismiss tooltip before retry')
        jokers.append(joker)
        refreshed.append(index)
        session.log("joker_observed", joker=asdict(joker), confidence=confidence,
                    name_confidence=name_confidence, tooltip=labels)
    if refreshed:
        session.hover(.93, .55, "dismiss tooltip")
    if len(jokers) != count:
        raise ObservationError('Joker roster length differs from the observed counter')
    # A contradictory visual proof is unsafe. A genuinely new/unmatched slot
    # does not invalidate independently proven neighbours.
    if reuse and any(mapping[i] is not None and j.key != cache.jokers[mapping[i]].key
                     for i, j in enumerate(jokers)):
        session.joker_cache = None
        return collect_jokers(session, session.observe('joker-roster-changed'), state, force=True)
    patches = tuple(cache.patches[mapping[i]] if i in reused else roster_patch(image, count, i)
                    for i in range(count))
    session.joker_cache = JokerCache(round_token(state), limit, tuple(jokers), patches, frozenset())
    session.log('joker_collection', reused_slots=reused, refreshed_slots=refreshed,
                roster_verified_from_cache=bool(reuse), seconds=round(time.perf_counter() - started, 3))
    return replace(state, jokers=tuple(jokers), total_joker_slots=limit,
                   joker_order_unknown=False, score_confidence=('bounded_joker_counter'
                       if any(j.get('counter_lower_bound') for j in jokers) else 'observed'))


def collect_cards(session, image, count, *, state=None):
    from hand_cache import HandCache, card_patch
    from joker_cache import round_token
    from plain_hand import read_plain_hand
    started = time.perf_counter()
    state = state or getattr(session, 'policy_state', RunState())
    cache = getattr(session, 'hand_cache', None)
    if cache is not None or count == 8:
        observed_count, limit = hand_counter(image)
        if observed_count != count:
            raise ObservationError('Hand changed before cached card inspection')
    else:
        limit = state.hand_size
    hints = cache.verified_identities(image, state, count, limit) if cache else [None] * count
    plain = read_plain_hand(image, state, count, limit,
                            skip_slots=[i for i, hint in enumerate(hints) if hint is not None])
    hints = [hint if hint is not None else plain[i][0] if plain[i] else None
             for i, hint in enumerate(hints)]
    # Proven retained cards or ordinary faces with known deck metadata can
    # constrain corner matching. Uncertain/special cards use the full reader.
    found = read_hand(image, count, identities=hints if any(hints) else None, allow_uncertain=True)
    enriched = []
    reused, refreshed, direct = [], [], []
    for index, (card, confidence, (x, y)) in enumerate(found):
        if hints[index] is not None:
            enriched.append((hints[index], confidence, (x, y)))
            (direct if plain[index] else reused).append(index)
            session.log('card_observed', index=index, card=asdict(hints[index]),
                        confidence=confidence, tooltip=[],
                        identity_source='verified_plain_face' if plain[index] else 'verified_retained_card',
                        face_confidence=plain[index][1] if plain[index] else None)
            continue
        if card.face_down:
            # A back has no readable rank, suit or modifiers. Neighbouring
            # tooltip text must never be used to enrich its unknown identity.
            enriched.append((card, confidence, (x, y)))
            session.log('card_observed', index=index, card=asdict(card),
                        confidence=confidence, tooltip=[], identity_source='visible_back')
            continue
        for attempt in range(3):
            session.hover((x + 30) / 2064, (y + 90) / 1191, f"read card {index} modifiers")
            tooltip = session.observe(f"card-{index}-tooltip")
            region = (530, 320, 2020, 875) if attempt == 0 else (max(530, x - 180), max(300, y - 450), min(2020, x + 380), min(875, y - 15))
            labels = tooltip_labels(read_text(tooltip, region))
            try:
                card = parse_card_tooltip(card, labels, require_identity=True)
                break
            except ObservationError as error:
                title = read_card_title(tooltip, (x, y))
                if title:
                    recovered_labels = [line for line in labels if not re.search(
                        r'\b(?:ace|king|queen|jack|10|[2-9])\s+of\s+(?:spades|hearts|diamonds|clubs)\b',
                        line, re.I)] + title
                    try:
                        card = parse_card_tooltip(card, recovered_labels, require_identity=True)
                    except ObservationError:
                        pass
                    else:
                        labels = recovered_labels
                        session.log('card_title_recovered', index=index, title=title,
                                    source='isolated_title_line')
                        break
                session.log('card_identity_retry', index=index, attempt=attempt + 1, error=str(error), tooltip=labels)
                if attempt == 2:
                    raise
                session.hover(.93, .55, 'dismiss unreadable card tooltip before retry')
                time.sleep(.3)
        enriched.append((card, confidence, (x, y)))
        refreshed.append(index)
        session.log("card_observed", index=index, card=asdict(card), confidence=confidence, tooltip=labels)
    if refreshed:
        session.hover(.93, .55, "dismiss card tooltip")
    positions_reused = all(hint is not None or (
        confidence >= .53 and not original.is_stone and not actual.is_stone
        and not original.face_down and not actual.face_down and original.identity == actual.identity)
        for hint, (original,confidence,_), (actual,_,_) in zip(hints,found,enriched))
    located = (found if positions_reused else
               read_hand(image, count, identities=[entry[0] for entry in enriched]))
    patches = tuple(next(p for c, p in zip(cache.cards, cache.patches) if c == enriched[i][0])
                    if i in reused else card_patch(image, entry[2]) for i, entry in enumerate(located))
    session.hand_cache = HandCache(round_token(state), tuple(entry[0] for entry in enriched), patches)
    session.log('card_collection', reused_slots=reused, plain_slots=direct, refreshed_slots=refreshed,
                positions_reused=positions_reused,seconds=round(time.perf_counter() - started, 3))
    return [(card, confidence, position) for (card, _, _), (_, confidence, position) in zip(enriched, located)]


def collect_position(session, state, *, refresh_levels=False, initial_image=None):
    started = time.perf_counter()
    from hand_levels import needs_observation
    if needs_observation(state):
        state = collect_levels(session, state)
        initial_image = None
    reuse = getattr(session, 'reuse_observation', None)
    for retry in range(3):
        image = (reuse('decision-state', initial_image)
                 if retry == 0 and initial_image is not None and reuse is not None
                 else session.observe("decision-state"))
        try:
            state, count = observed_state(image, state)
            break
        except ObservationError as error:
            session.log('state_read_retry', attempt=retry + 1, error=str(error))
            if retry == 2:
                raise
            time.sleep(1)
    state_read = time.perf_counter()
    state = collect_jokers(session, image, state)
    jokers_read = time.perf_counter()
    if not state.round_resources_observed:
        # Observe reset resources before any play/discard. Boss-only reductions
        # must not become the assumed capacity of subsequent ordinary rounds.
        boss = state.boss.lower().removeprefix('the ')
        base_hands = state.base_hands_per_round
        base_discards = state.base_discards_per_round
        if boss != 'needle':
            base_hands = max(1, state.hands_left - 3 * state.effect_count('burglar') + state.joker_count('troubadour'))
        if boss != 'water' and not state.effect_count('burglar'):
            base_discards = max(0, state.discards_left - state.joker_count('drunkard') - 3 * state.joker_count('merry_andy'))
        state = replace(state, base_hands_per_round=base_hands,
                        base_discards_per_round=base_discards, round_resources_observed=True)
    image = (reuse('hand-before-inspection', image) if reuse is not None
             else session.observe("hand-before-inspection"))
    found = collect_cards(session, image, count, state=state)
    if state.deck_cards and len(state.deck_cards) != state.deck_size:
        session.log('deck_memory_invalidated', known=len(state.deck_cards), observed=state.deck_size,
                    reason='unobserved deck creation/destruction')
        state = replace(state, deck_cards=())
    elif state.deck_cards:
        deck = list(state.deck_cards)
        uncertain = set(state.unverified_chip_cards)
        # Unique identities can be reconciled from observed enhancements,
        # seals or editions without guessing which duplicate changed.
        identities = Counter(c.identity for c in deck)
        for card, _, _ in found:
            if not card.face_down and identities[card.identity] == 1:
                index = next(i for i, old in enumerate(deck) if old.identity == card.identity)
                deck[index] = replace(card, debuffed=False, face_down=False)
                if not card.debuffed:
                    # Debuffed tooltip can hide its permanent chips. Only a
                    # readable active card retires that invisible uncertainty.
                    uncertain.discard(card.identity)
        state = replace(state, deck_cards=tuple(deck), unverified_chip_cards=tuple(sorted(uncertain)))
    if _boss(state) == "cerulean_bell":
        flags = selection_flags(found)
        if len(flags) != 1:
            raise ObservationError("Cannot identify Cerulean Bell's forced card")
        state = replace(state, forced_card_index=next(iter(flags)))
    else:
        state = replace(state, forced_card_index=-1)
    state, found = collect_secret_levels(session, state, found)
    session.log("observed_state", state=asdict(state), hand_count=count)
    session.log('position_timing', seconds=round(time.perf_counter() - started, 4),
                state_seconds=round(state_read - started, 4),
                joker_seconds=round(jokers_read - state_read, 4),
                cards_and_levels_seconds=round(time.perf_counter() - jokers_read, 4))
    return state, found


def collect_secret_levels(session, state, found):
    """An upgraded secret hand can still be absent from the Run Info list."""
    from hand_levels import secret_candidates, sync_cache
    from perception import read_hand_preview_level
    candidates = secret_candidates(state, [c for c, _, _ in found])
    for category, indices in candidates.items():
        original = selection_flags(found)
        selected = select_verified(session, found, indices, 'inspect uncertain secret-hand level')
        try:
            actual, level = read_hand_preview_level(selected)
            if actual != category:
                raise ObservationError('Secret-hand level preview differs from verified cards')
            state = replace(state, levels={**state.levels, category: level},
                            uncertain_levels=tuple(k for k in state.uncertain_levels if k != category))
            session.log('hand_level_preview', category=category, level=level)
        finally:
            located = read_hand(selected, len(found))
            raised = [(old[0], new[1], new[2]) for old, new in zip(found, located)]
            restored = select_verified(session, raised, original, 'restore selection after level inspection')
            located = read_hand(restored, len(found))
            found = [(old[0], new[1], new[2]) for old, new in zip(found, located)]
    sync_cache(session, state)
    return state, found


def selection_flags(found):
    count = len(found)
    return {i for i, (_, _, (_, y)) in enumerate(found)
            if y < 710 + 22 * abs(2 * i / max(1, count - 1) - 1) - 25}


def hand_selection_point(position):
    # A raised left neighbour can cover the middle of an unraised card.
    # The lower part of its exposed left strip remains clickable. Keep the
    # horizontal offset small for crowded hands; moving right can hit the
    # next card. These positions are refined top-left corners, not centers.
    x, y = position
    return (x + 30) / 2064, (y + 195) / 1191


def select_verified(session, found, indices, reason):
    desired = set(indices)
    if len(desired) > 5 or any(index < 0 or index >= len(found) for index in desired):
        raise ValueError('Invalid card selection')
    already = selection_flags(found)
    current = found
    for attempt in range(3):
        toggles = sorted(desired ^ already, reverse=True)
        if toggles and attempt == 0:
            session.hover(.93, .55, 'dismiss previous card tooltip before selection')
        for index in toggles:
            if attempt:
                session.hover(.93, .55, 'dismiss previous card tooltip before selection')
            x, y = hand_selection_point(current[index][2])
            session.tap(x, y, f"{reason}: card {index}")
        session.hover(.93, .55, 'move off selected cards')
        image = session.observe('verify-selected-cards')
        selected = read_hand(image, len(found), identities=[entry[0] for entry in found])
        actual = selection_flags(selected)
        if actual == desired:
            session.log('selection_verified', indices=sorted(desired), attempts=attempt + 1)
            return image
        session.log('selection_mismatch', expected=sorted(desired), actual=sorted(actual), attempt=attempt + 1)
        already, current = actual, selected
    raise ObservationError('Selected cards still differ from the plan after three checks')


def verify_play_preview(session, image, category, *, hidden=False):
    contrast, contradiction = False, False
    for attempt in range(3):
        actual = None
        try:
            if hidden:
                words = read_text(image, (150, 545, 535, 655))
                # The game intentionally hides this preview when any selected
                # card is face down. Require that visible unknown preview;
                # do not pretend the estimated category has been confirmed.
                if not any(score > .8 and text.strip().count('?') >= 3
                           for text, score, _ in words):
                    raise ObservationError('Selected hidden-hand preview is not readable')
                session.log('play_preview_verified', category='unknown', hidden=True)
                return
            actual = read_hand_preview(image, contrast=True) if contrast else read_hand_preview(image)
            if actual != category:
                contradiction = True
                raise ObservationError(f'Selected preview is {actual}, policy planned {category}')
            session.log('play_preview_verified', category=actual)
            return
        except ObservationError as error:
            if isinstance(error, HandPreviewConflict):
                contradiction = True
            session.log('play_preview_mismatch', expected=category, attempt=attempt + 1, error=str(error))
            if attempt == 2:
                raise HandPreviewError(str(error), cache_safe=not hidden and not contradiction) from error
            if not hidden and attempt == 0 and actual is None and not contradiction:
                # Spend the second of the same three reads on a different
                # representation, replacing a sleep/capture/repeated OCR.
                contrast = True
                session.log('preview_read_variant', variant='same_frame_contrast')
                continue
            contrast = False
            time.sleep(.3)
            image = session.observe('verify-play-preview-again')


def apply_play_order(session, found, decision):
    session.play_order_fallback = False
    order = decision.play_order or decision.indices
    if order == tuple(sorted(order)):
        return found, list(decision.indices)
    desired = list(order) + [i for i in range(len(found)) if i not in order]
    roster = list(range(len(found)))
    originals = list(found)
    current = list(found)
    for dest, original in enumerate(desired):
        source = roster.index(original)
        if source == dest:
            continue
        sx, sy = current[source][2]
        dx, dy = current[dest][2]
        session.drag(((sx + 35) / 2064, (sy + 95) / 1191),
                     ((dx + 15) / 2064, (dy + 95) / 1191), "optimize card scoring order")
        roster.insert(dest, roster.pop(source))
        session.hover(.93, .55, "dismiss drag hover")
        image = session.observe("verify-card-order")
        expected = [originals[i][0].identity for i in roster]
        try:
            current = read_hand(image, len(found))
        except ObservationError:
            current = []
        if [entry[0].identity for entry in current] != expected:
            # Corner guesses (notably 10 -> 9) are insufficient proof that a
            # drag failed. Reread full tooltips before accepting any order.
            current = collect_cards(session, image, len(found))
            if Counter(entry[0].identity for entry in current) != Counter(entry[0].identity for entry in originals):
                raise ObservationError('Card roster changed during reorder; recollect before playing')
            if [entry[0].identity for entry in current] != expected:
                session.play_order_fallback = True
                session.log('card_order_fallback', expected=expected,
                            actual=[entry[0].identity for entry in current],
                            reason='drag did not establish requested order; replan in verified actual order')
                return current, []
            session.log('card_order_verified_by_tooltips', identities=expected)
    enriched = [(originals[original][0], confidence, pos)
                for original, (_, confidence, pos) in zip(roster, current)]
    return enriched, list(range(len(order)))


def wait_consumable_hand(session, image):
    """Wait for the hand to return to its normal position after use animation."""
    for attempt in range(5):
        try:
            count, _ = hand_counter(image)
            if count < 1:
                raise ObservationError('No cards visible after consumable animation')
            # The counter may return before a transformed card has finished
            # moving. Require all card locations before clearing selections.
            read_hand(image, count, allow_uncertain=True)
            return image, count
        except ObservationError as error:
            if attempt == 4:
                raise
            session.log('consumable_animation_wait', attempt=attempt + 1, error=str(error))
            time.sleep(1)
            image = session.observe('after-consumable-ready')


def try_consumable(session, state, found):
    if state.consumables_held <= 0:
        return False
    image = session.observe("held-consumables")
    matches = read_consumables(image)
    definitions = consumable_templates()[1]
    best = None
    for index, center in enumerate(consumable_centers(matches, state.consumables_held)):
        session.hover(center[0] / 2064, center[1] / 1191, 'read consumable title and description')
        tooltip = session.observe('consumable-tooltip')
        words = read_text(tooltip, (1300, 75, 2055, 850))
        labels = tooltip_labels([word for word in words
                                 if word[2][1] > 340 or abs(word[2][0] - center[0]) < 85])
        try:
            key, confidence = identify_consumable_text(labels)
        except ObservationError as error:
            session.log('consumable_unknown', slot=index, tooltip=labels, error=str(error))
            continue
        consumable = definitions[key]
        session.log('consumable_observed', name=consumable.name, confidence=confidence)
        plan = decide_consumable(consumable, [c for c, _, _ in found], state)
        signature = (consumable.key, tuple(sorted(repr(c) for c, _, _ in found)))
        if signature in getattr(session, 'blocked_consumable_orders', set()):
            continue
        session.log("consumable_decision", name=consumable.name, **asdict(plan))
        observable_random = consumable.key in ('judgement', 'the_soul', 'ankh', 'hex')
        if plan.action == "use" and (plan.deterministic or observable_random) and (best is None or plan.utility > best[0].utility):
            best = plan, consumable, center
    if best is None:
        session.hover(.93, .55, 'dismiss consumable tooltip')
        return False
    plan, consumable, (x, y) = best
    if plan.play_order:
        signature = (consumable.key, tuple(sorted(repr(c) for c, _, _ in found)))
        found, targets = apply_play_order(session, found,
            SimpleNamespace(indices=plan.indices, play_order=plan.play_order))
        if getattr(session, 'play_order_fallback', False):
            blocked = getattr(session, 'blocked_consumable_orders', set())
            blocked.add(signature)
            session.blocked_consumable_orders = blocked
            session.log('consumable_order_abandoned', name=consumable.name,
                        reason='Death direction was not established; recollect and replan')
            return True
        plan = replace(plan, indices=tuple(targets), play_order=())
    session.hover(x / 2064, y / 1191, "confirm consumable identity")
    words = read_text(session.observe('consumable-tooltip'), (1300, 75, 2055, 850))
    labels = tooltip_labels([word for word in words if word[2][1] > 340 or abs(word[2][0] - x) < 85])
    if identify_consumable_text(labels)[0] != consumable.key:
        raise ObservationError('Consumable tooltip changed before use')
    session.hover(.93, .55, "dismiss consumable tooltip")
    if plan.indices:
        select_verified(session, found, plan.indices, "consumable targets")
    session.tap(x / 2064, y / 1191, "select consumable")
    words = read_text(session.observe("consumable-use-button"), (1400, 70, 2050, 650))
    button = next((p for text, score, p in words if text.strip().lower() == "use" and score > .85), None)
    if button is None:
        raise ObservationError("Consumable Use button could not be verified")
    begin_action(session, 'consume', {'consumable_count': state.consumables_held},
                 after_state=state_after_consumable(state, [c for c, _, _ in found], consumable, plan),
                 name=consumable.name, levels_changed=(consumable.kind.lower() == 'planet'
                                                     or consumable.key == 'black_hole'),
                 consumable_kind=consumable.kind.lower())
    session.tap(button[0] / 2064, button[1] / 1191, plan.reason)
    time.sleep(2)
    image = session.observe("after-consumable")
    count, _ = consumable_counter(image)
    if count != state.consumables_held - 1:
        raise ObservationError("Consumable use did not change the observed slot count")
    finish_action(session, True, name=consumable.name)
    session.policy_state = replace(session.policy_state, consumables_held=count)
    image, hand_count = wait_consumable_hand(session, image)
    # Strength/Death/enhancement targets may remain raised after use.
    if plan.indices:
        current = read_hand(image, hand_count)
        for index in sorted(selection_flags(current), reverse=True):
            cx, cy = current[index][2]
            session.tap((cx + 30) / 2064, (cy + 45) / 1191, "clear consumed-card selection")
    return True


def apply_joker_order(session, state, order):
    if state.joker_order_unknown:
        session.log('joker_order_skipped', reason='physical identities hidden by Amber Acorn')
        return state
    original = tuple(range(len(state.jokers)))
    if sorted(order) != list(original):
        raise ObservationError('Joker order is not a permutation of the observed roster')
    if tuple(order) == original:
        return state
    # Read the result of every drag. A dropped mouse event must not shift the
    # assumed indices used by the following drag.
    def identity(joker):
        return (joker.key, joker.edition, joker.debuffed, joker.value, joker.data)
    expected = [identity(state.jokers[i]) for i in order]
    roster = Counter(identity(j) for j in state.jokers)
    actual = state
    slots = joker_slots(len(state.jokers))
    for attempt in range(min(10, len(slots) + 3)):
        current = [identity(j) for j in actual.jokers]
        if current == expected:
            session.log('joker_order_verified', keys=[j.key for j in actual.jokers])
            return actual
        dest = next(i for i, (a, b) in enumerate(zip(current, expected)) if a != b)
        source = next(i for i in range(dest + 1, len(current)) if current[i] == expected[dest])
        session.hover(.93, .55, 'dismiss tooltip before Joker drag')
        session.drag((slots[source] / 2064, 220 / 1191),
                     ((slots[dest] - 40) / 2064, 220 / 1191), 'optimize Joker trigger and copy order')
        session.hover(.93, .55, 'dismiss Joker drag')
        image = session.observe('verify-joker-order')
        actual = collect_jokers(session, image, actual)
        if Counter(identity(j) for j in actual.jokers) != roster:
            raise ObservationError('Joker roster changed during ordering')
        session.log('joker_order_drag_observed', attempt=attempt + 1,
                    keys=[j.key for j in actual.jokers])
    if [identity(j) for j in actual.jokers] == expected:
        session.log('joker_order_verified', keys=[j.key for j in actual.jokers])
    else:
        # A fully read unchanged roster is valid gameplay state. The caller
        # recomputes scores and chooses cards using this actual order.
        session.log('joker_order_fallback', keys=[j.key for j in actual.jokers],
                    reason='replan scoring in the verified actual order after bounded drag attempts')
    return actual
