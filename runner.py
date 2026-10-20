"""Airtest runner with observed state and verified card selections."""
from __future__ import annotations
import argparse
import json
import pickle
import re
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from bot import RUNS, Session
from controller import collect_position, select_verified, apply_play_order, collect_jokers, try_consumable, apply_joker_order, verify_play_preview
from perception import (read_text, read_numbers, read_jokers, joker_from_definition,
                        identify_joker_text, tooltip_labels, consumable_counter,
                        shop_offer_item, read_reroll_button, joker_slots, shop_tooltip_region, tooltip_rows)
from perception import read_future_boss, victory_new_run, read_offer_price
from perception import supported_shop_item
from recovery import (ObservationError, DesktopUnavailableError, begin_action, finish_action, confirm_pending,
                      recover_view, run_info_back, atomic_json, state_from_dict)
from strategy import Card, SUITS, RunState, ShopItem, decide_shop, decide_discard, decide_play, optimize_play_order, optimize_joker_order, locked_mouth_hand, shop_strength, joker_hand_size_delta
from strategy import score_hand
from perception import verify_current_multiplier
from perception import joker_tooltip_words, tooltip_title
from vision import active_blind, classify_scene, scene_scores
from run_history import code_version, export_run
from runtime_issues import record_issue, issue_counts, finish_issue_review, recovery_code
from shop_extras import collect_extra_offers, purchase_extra, handle_pack
from hand_levels import after_discard, after_play, fresh_levels, sync_cache


def observe_scene(session, image):
    scene = classify_scene(image)
    if scene == 'unknown':
        back = run_info_back(image)
        if back is not None:
            # Focus can disappear during collect_levels' finally block. The
            # next iteration must close the proven overlay before proceeding.
            session.tap(back[0] / 2064, back[1] / 1191,
                        'dismiss recognized Run Info before scene reconciliation')
            sync_cache(session)
            image = session.observe('scene-after-interrupted-run-info')
            scene = classify_scene(image)
    if scene == 'unknown' and scene_scores(image).get('shop', 0) > .55:
        # The cashout click can leave the cursor on a new shop offer whose
        # tooltip covers Next Round. Moving away is read-only inspection.
        session.hover(.93, .55, 'dismiss tooltip before verifying shop or pack scene')
        image = session.observe('scene-without-tooltip')
        scene = classify_scene(image)
    return image, scene


def record_round_snapshot(session, image, scene):
    labels = None
    if scene in ('loss', 'win'):
        labels = [text for text, confidence, _ in read_text(image, (550, 180, 1720, 1100)) if confidence > .8]
    session.log('round_snapshot', counters=read_numbers(image), settled=True,
                scene=scene, source='runtime_' + scene, end_screen=labels)


def verify_selected_purchase(image, chosen, x):
    words = read_text(image, shop_tooltip_region(image, x))
    labels = tooltip_rows(words)
    labels = [label for label in labels if label.strip() != 'JOKER']
    art = read_jokers(image, (960, 350, 1930, 800)) if chosen.name == 'Misprint' else []
    hint = min(art, key=lambda match: abs(match[2][0] - x)) if art else (0, '', None)
    actual, _ = shop_offer_item(labels, chosen.price, hint[1], art_confidence=hint[0],
                               image=image, words=words)
    if actual.kind != chosen.kind or actual.name != chosen.name:
        raise ObservationError('Selected shop item differs from the purchase plan')


def verify_shop_selection(session, image, chosen, x, y):
    """Refresh transient selected-card tooltips before authorizing a buy."""
    for attempt in range(3):
        try:
            buttons = read_text(image, (960, 550, 1930, 800))
            buy = next((p for text, score, p in buttons
                if text.strip().lower() == 'buy' and score > .85 and abs(p[0] - x) < 110), None)
            if buy is None:
                raise ObservationError('Buy button could not be verified')
            verify_selected_purchase(image, chosen, x)
            return buy, image
        except ObservationError as error:
            session.log('shop_selection_read_retry', attempt=attempt + 1,
                        name=chosen.name, error=str(error))
            if attempt == 2:
                raise
            session.hover(.93, .55, 'dismiss selected shop tooltip before retry')
            session.hover(x / 2064, y / 1191, 'refresh selected shop tooltip')
            image = session.observe('shop-selected-retry')


def collect_shop_jokers(session, image, state):
    actual = collect_jokers(session, image, state)
    return replace(actual, hand_size=max(1, state.hand_size + joker_hand_size_delta(state.jokers, actual.jokers)))


def select_cards(session, found, indices, reason):
    return select_verified(session, found, indices, reason)


def timed_policy(session, phase, calculate):
    started = time.perf_counter()
    result = calculate()
    session.log('policy_timing', phase=phase, seconds=round(time.perf_counter() - started, 4))
    return result


def play_one_hand(session, discards=4, target=300, round_score=0, hands_left=4):
    state = getattr(session, "policy_state", RunState(discards_left=discards,
        target_score=target, round_score=round_score, hands_left=hands_left))
    refresh = getattr(session, "refresh_levels", True)
    initial_image = getattr(session, 'scene_hand_frame', None)
    session.scene_hand_frame = None
    state, initial_image = prepare_boss_action(session, state, initial_image)
    for attempt in range(32):
        state, found = collect_position(session, state, refresh_levels=refresh,
                                        initial_image=initial_image)
        initial_image = None
        session.refresh_levels = refresh = False
        if try_consumable(session, state, found):
            state = session.policy_state
            refresh = getattr(session, 'refresh_levels', True)
            continue
        cards = [entry[0] for entry in found]
        decision = timed_policy(session, 'play',
            lambda: optimize_play_order(cards, state, decide_play(cards, state)))
        order, arranged_score = timed_policy(session, 'joker_order',
            lambda: optimize_joker_order(cards, state, decision))
        if order != tuple(range(len(state.jokers))):
            session.log("joker_order_decision", order=order, expected_score=arranged_score)
            state = apply_joker_order(session, state, order)
            decision = optimize_play_order(cards, state, decide_play(cards, state))
        order_for_score = decision.play_order or decision.indices
        conservative = score_hand([cards[i] for i in order_for_score],
            replace(state, score_mode='conservative'),
            [c for i, c in enumerate(cards) if i not in decision.indices])
        session.log("hand_decision", cards=[f"{c.rank}{c.suit}" for c in cards],
            card_details=[asdict(c) for c in cards], best_indices=list(decision.indices),
            play_order=list(decision.play_order), expected_score=decision.estimate.score,
            conservative_score=conservative.score, category=decision.estimate.category,
            score_confidence=state.score_confidence,
            reason=decision.reason, chips=decision.estimate.chips,
            multiplier=decision.estimate.multiplier, notes=decision.estimate.notes,
            actual_round_score=state.round_score, target_score=state.target_score,
            discards_remaining=state.discards_left, hands_left=state.hands_left,
            jokers=[asdict(j) for j in state.jokers])
        if state.discards_left:
            discard = timed_policy(session, 'discard',
                lambda: decide_discard(cards, state, current=decision))
            session.log("discard_decision", **asdict(discard))
            if discard.indices:
                select_verified(session, found, discard.indices, "discard")
                after = replace(state, seen_cards=state.seen_cards + tuple(cards[i] for i in discard.indices),
                                round_discards_used=max(0, state.round_discards_used) + 1)
                after = after_discard(state, after, [cards[i] for i in discard.indices])
                begin_action(session, 'discard', {'discards_left': state.discards_left}, after_state=after,
                             retained_cards=[asdict(c) for i, c in enumerate(cards) if i not in discard.indices])
                session.tap(.645, .88, "confirm verified discard")
                time.sleep(2)
                confirm_pending(session)
                state = session.policy_state
                refresh = getattr(session, 'refresh_levels', False)
                continue
        if decision.estimate.score <= 0:
            session.log('zero_score_cycle', indices=list(decision.indices), reason=decision.reason,
                        locked_hand=locked_mouth_hand(state), hands_left=state.hands_left,
                        draw_pile_size=state.draw_pile_size)
        found, indices = apply_play_order(session, found, decision)
        if getattr(session, 'play_order_fallback', False):
            cards = [entry[0] for entry in found]
            decision = decide_play(cards, state, optimize_order=False)
            indices = list(decision.indices)
            session.log('hand_decision_after_order_fallback',
                        card_details=[asdict(c) for c in cards], best_indices=indices,
                        play_order=list(decision.play_order), category=decision.estimate.category,
                        expected_score=decision.estimate.score,
                        reason='use verified actual card order without another drag')
        selected_image = select_verified(session, found, indices, f"play {decision.estimate.category}")
        hidden = any(found[i][0].face_down for i in indices)
        if hidden:
            verify_play_preview(session, selected_image, decision.estimate.category, hidden=True)
        else:
            verify_play_preview(session, selected_image, decision.estimate.category)
        history, round_history = dict(state.played_types), dict(state.round_played_types)
        category = 'unknown' if hidden else decision.estimate.category
        if not hidden:
            history[category] = history.get(category, 0) + 1
            round_history[category] = round_history.get(category, 0) + 1
        after = replace(state, played_types=history, round_played_types=round_history,
            boss_seen_types=frozenset(round_history), boss_first_hand_done=True,
            total_hands_played=state.total_hands_played + 1,
            pillar_cards=state.pillar_cards + tuple(cards[i] for i in decision.indices),
            seen_cards=state.seen_cards + tuple(cards[i] for i in decision.indices))
        from plain_hand import mark_unverified_chips
        after = mark_unverified_chips(after, [found[i][0] for i in indices])
        after = after_play(state, after, category, [found[i][0] for i in indices])
        if state.boss.lower().removeprefix('the ') == 'mouth':
            after = replace(after, boss_target_hand=locked_mouth_hand(state) or category)
        begin_action(session, 'play', {'hands_left': state.hands_left, 'round_score': state.round_score},
                     after_state=after, category=category, estimated_score=decision.estimate.score,
                     retained_cards=[asdict(c) for i, (c, _, _) in enumerate(found) if i not in indices])
        session.tap(.43, .88, "play verified hand")
        # Next observation reads the actual score; estimates are never accumulated.
        time.sleep(2)
        confirm_pending(session, session.observe('after-play'))
        return state.discards_left, decision.estimate.score
    raise RuntimeError("Hand did not finish after 32 action cycles")


def prepare_boss_action(session, state, image=None):
    """Resolve required Boss actions before card discovery or discard search."""
    from strategy import _boss
    from perception import observed_state
    from boss_policy import choose_verdant_sale
    if (state.blind_index != 2 or state.boss_disabled or state.has('chicot')
            or state.boss and _boss(state) != 'verdant_leaf'):
        return state, image
    if getattr(session, 'pending_action', None):
        raise ObservationError('Confirm the pending action before preparing a Boss')
    image = image if image is not None else session.observe('boss-preparation-state')
    state, _ = observed_state(image, state)
    if _boss(state) != 'verdant_leaf':
        return state, image
    state = collect_jokers(session, image, state)
    index = choose_verdant_sale(state)
    if index is None:
        session.log('boss_preparation_unavailable', boss=state.boss,
                    reason='no verified sellable Joker; play with the active debuff')
        return state, image
    session.log('boss_preparation_decision', boss=state.boss, action='sell_joker',
                index=index, name=state.jokers[index].name,
                reason='disable Verdant Leaf before spending discards')
    state = sell_shop_joker(session, state, index, disable_leaf=True)
    image = getattr(session, 'boss_after_sale_frame', None)
    session.boss_after_sale_frame = None
    return state, image


def sell_shop_joker(session, state, index, *, disable_leaf=False):
    old = state.jokers[index]
    if old.get('eternal'):
        raise ObservationError('Refusing to sell an Eternal Joker')
    x = joker_slots(len(state.jokers))[index]
    session.tap(x / 2064, 220 / 1191, 'select verified replacement Joker')
    selected = session.observe('shop-joker-selected-for-sale')
    words = read_text(selected, (520, 70, 2040, 650))
    main, versions = joker_tooltip_words(words, x)
    key, _ = identify_joker_text(tooltip_labels(main), title=tooltip_title(main))
    if key != old.key or any(text.strip().lower() == 'eternal' for text, _, _ in main + versions):
        raise ObservationError('Selected Joker identity or sellability differs from the replacement plan')
    button = next((p for text, score, p in words
                   if re.fullmatch(r'sell(?:\s+\$\d+)?', text.strip(), re.I)
                   and score > .85 and abs(p[0] - x) < 260), None)
    if button is None:
        raise ObservationError('Replacement Sell button could not be verified')
    after = replace(state, jokers=tuple(j for i, j in enumerate(state.jokers) if i != index),
                    total_joker_slots=state.total_joker_slots - (old.edition == 'negative'))
    after = replace(after, hand_size=max(1, state.hand_size + joker_hand_size_delta(state.jokers, after.jokers)))
    if disable_leaf:
        from strategy import _boss
        from boss_policy import after_joker_sale
        if _boss(state) != 'verdant_leaf':
            raise ObservationError('Boss sale is only authorized for active Verdant Leaf')
        after = after_joker_sale(state, index, disable_leaf=True)
    begin_action(session, 'sell', {'joker_count': len(state.jokers), 'money': state.money},
                 after_state=after, name=old.name,
                 **({'purpose': 'disable_verdant_leaf'} if disable_leaf else {}))
    session.tap(button[0] / 2064, button[1] / 1191, 'sell weak Joker for a scoring replacement')
    confirm_pending(session)
    if disable_leaf:
        session.hand_cache = None  # The sale changes every card's debuff metadata.
        image = session.observe('boss-after-verified-joker-sale')
        session.boss_after_sale_frame = image
        return replace(session.policy_state, **read_numbers(image))
    image = session.observe('shop-after-joker-sale')
    return collect_shop_jokers(session, image, replace(session.policy_state, **read_numbers(image)))


def read_shop_reroll(session, image, latest_image):
    """Retry a missing price on real newer pixels; never infer a shop cost."""
    reroll = read_reroll_button(image)
    if reroll is not None:
        return reroll
    if latest_image is not image:
        reroll = read_reroll_button(latest_image)
        session.log('shop_reroll_read_retry', source='bottom_offers', verified=reroll is not None)
    if reroll is None:
        # The caller has dismissed the tooltip. One fresh frame also handles
        # transient text animation without weakening read_reroll_button.
        fresh = session.observe('shop-reroll-confirmation')
        reroll = read_reroll_button(fresh)
        session.log('shop_reroll_read_retry', source='fresh', verified=reroll is not None)
    return reroll


def shop(session, image):
    # Cashout advances end-of-round counters before a new blind token exists.
    from joker_cache import dirty_jokers
    dirty_jokers(session, round_end=True)
    session.hand_cache = None
    session.extra_offer_cache = None
    numbers = read_numbers(image)
    if 'money' not in numbers:
        raise ObservationError('Shop money is not readable')
    held, capacity = consumable_counter(image)
    state = replace(getattr(session, "policy_state", RunState()), **numbers,
                    consumables_held=held, consumable_slots=capacity)
    state = collect_shop_jokers(session, image, state)
    blocked_offers = set()
    unsupported_offers = {}
    unreadable_offers = {}
    for attempt in range(14):
        observation_started = time.perf_counter()
        image = session.observe("shop-offers")
        # Overstock vouchers can add a third/fourth upper offer. Prices and
        # tooltip verification locate each item; don't assume two fixed slots.
        matches = read_jokers(image, (960, 450, 1930, 680))
        words = read_text(image, (960, 430, 1930, 750))
        items, positions = [], []
        price_labels = [(text, score, center) for text, score, center in words
                        if text.strip().startswith('$') and score > .8 and center[1] < 500]
        for text, confidence, (x, py) in price_labels:
            try:
                price = read_offer_price(image, text, confidence, (x, py))
            except ObservationError as error:
                if getattr(session, 'pending_action', None):
                    raise
                record_issue(session, 'shop_offer_price_unreadable', level='degraded',
                    code='optional_offer_price_unreadable', message='Skip an unverified optional shop price',
                    error=str(error), text=text, position=(x, py))
                continue
            if price != int(text.strip().replace('$', '').replace(',', '')):
                session.log('shop_price_recovered', original=text, price=price, position=(x, py))
            y = py + 140
            art = min(matches, key=lambda match: abs(match[2][0] - x)) if matches else (0, '', None)
            hint = art[1]
            from perception import offer_art_patch, same_offer_art
            cache_key = (price, round(x / 10), round(y / 10))
            cached = unsupported_offers.get(cache_key)
            if cached and same_offer_art(image, (x, y), cached[1]):
                session.log('shop_offer_unsupported_reused', name=cached[0].name, price=price)
                continue
            rejected = unreadable_offers.get(cache_key)
            if rejected is not None and same_offer_art(image, (x, y), rejected):
                session.log('shop_offer_unreadable_reused', price=price)
                continue
            session.hover(x / 2064, y / 1191, 'confirm shop offer title before evaluation')
            tooltip_image = session.observe('shop-offer-tooltip')
            labels = []
            try:
                tooltip_words = read_text(tooltip_image, shop_tooltip_region(tooltip_image, x))
                labels = tooltip_rows(tooltip_words)
                labels = [label for label in labels if label.strip() != 'JOKER']
                item, name_confidence = shop_offer_item(labels, price, hint,
                    require_supported=False, art_confidence=art[0],
                    image=tooltip_image, words=tooltip_words)
                if item.joker is not None:
                    item = replace(item, joker=verify_current_multiplier(
                        tooltip_image, tooltip_words, item.joker, state))
            except DesktopUnavailableError:
                raise
            except ObservationError as error:
                if getattr(session, 'pending_action', None):
                    raise
                unreadable_offers[cache_key] = offer_art_patch(image, (x, y))
                record_issue(session, 'shop_offer_unknown', level='degraded',
                    code='optional_offer_unreadable', message='Skip an unreadable optional shop offer',
                    error=str(error), price=price, tooltip=labels)
                continue
            if not supported_shop_item(item):
                unsupported_offers[cache_key] = (item, offer_art_patch(image, (x, y)))
                session.log('shop_offer_unsupported', name=item.name, kind=item.kind,
                    price=price, reason='identity verified; use workflow is not supported')
                continue
            session.log('shop_offer_verified', name=item.name, kind=item.kind,
                        price=price, confidence=name_confidence, tooltip=labels)
            items.append(item)
            positions.append((x, y))
        bottom_image = session.observe('shop-bottom-offers')
        extra_items, extra_positions = collect_extra_offers(session, bottom_image)
        items.extend(extra_items)
        positions.extend(extra_positions)
        available = [(item, position) for item, position in zip(items, positions)
                     if (item.kind, item.name, item.price, position) not in blocked_offers]
        items = [item for item, _ in available]
        positions = [position for _, position in available]
        session.hover(.93, .55, 'dismiss shop tooltip')
        reroll = read_shop_reroll(session, image, bottom_image)
        observation_seconds = time.perf_counter() - observation_started
        decision_started = time.perf_counter()
        decision = timed_policy(session, 'shop',
            lambda: decide_shop(items, state, reroll_cost=reroll[1] if reroll else None))
        strength = shop_strength(state)
        session.log("shop_decision", items=[asdict(item) for item in items], decision=asdict(decision),
                    money=state.money, strength=strength, rerolls_used=state.shop_rerolls,
                    observation_seconds=round(observation_seconds, 3),
                    decision_seconds=round(time.perf_counter() - decision_started, 3),
                    reroll_cost=reroll[1] if reroll else None,
                    state=asdict(state))
        if decision.action == 'sell_then_buy':
            chosen = items[decision.item_index]
            before_keys = sorted(j.key for i, j in enumerate(state.jokers) if i != decision.sell_index)
            state = sell_shop_joker(session, state, decision.sell_index)
            session.policy_state = state
            if (sorted(j.key for j in state.jokers) != before_keys or state.money < chosen.price
                    or len(state.jokers) >= state.total_joker_slots +
                       bool(chosen.joker and chosen.joker.edition == 'negative')):
                session.log('replacement_purchase_cancelled', name=chosen.name,
                            reason='observed post-sale inventory or cash differs from the purchase plan')
                continue
            # Complete the evaluated bundle using the same verified offer.
            # The normal selection/identity/receipt checks below still apply.
            # Replanning here used to sell a Joker then leave or reroll without
            # buying the replacement on which the survival forecast depended.
            decision = replace(decision, action='buy', sell_index=-1)
            session.log('replacement_purchase_continued', name=chosen.name,
                        reason='sale confirmed; verify the planned offer before Buy')
        if decision.action == 'reroll' and reroll:
            button, cost = reroll
            after = replace(state, shop_rerolls=state.shop_rerolls + 1, money=state.money - cost)
            begin_action(session, 'reroll', {'money': state.money, 'reroll_cost': cost}, after_state=after)
            session.tap(button[0] / 2064, button[1] / 1191, decision.reason)
            confirm_pending(session)
            state = session.policy_state
            blocked_offers.clear()
            unsupported_offers.clear()
            unreadable_offers.clear()
            continue
        if decision.action != "buy":
            break
        x, y = positions[decision.item_index]
        chosen = items[decision.item_index]
        if chosen.kind == 'voucher' or chosen.kind.endswith('_pack'):
            session.policy_state = state
            state = purchase_extra(session, chosen, (x, y), state)
            image = session.observe('shop-after-extra')
            held, capacity = consumable_counter(image)
            state = collect_shop_jokers(session, image, replace(state, **read_numbers(image),
                              consumables_held=held, consumable_slots=capacity))
            session.policy_state = state
            continue
        session.tap(x / 2064, y / 1191, "select verified " + chosen.kind + " purchase")
        selected = session.observe("shop-selected")
        try:
            buy, selected = verify_shop_selection(session, selected, chosen, x, y)
        except DesktopUnavailableError:
            raise
        except ObservationError as error:
            # No Buy has been clicked and no transaction journal exists here.
            # The optional offer can be skipped without inventing an outcome.
            if getattr(session, 'pending_action', None):
                raise
            blocked_offers.add((chosen.kind, chosen.name, chosen.price, (x, y)))
            record_issue(session, 'shop_purchase_skipped', level='degraded',
                code='optional_purchase_selection_unverified',
                message='Skip an optional purchase before any Buy click',
                name=chosen.name, error=str(error))
            session.hover(.93, .55, 'dismiss unverified optional purchase tooltip')
            continue
        before_count = len(state.jokers)
        before_jokers = state.jokers
        before_consumables = consumable_counter(selected)[0]
        begin_action(session, 'buy', {'joker_count': before_count,
                     'consumable_count': before_consumables, 'money': state.money},
                     name=items[decision.item_index].name, price=items[decision.item_index].price,
                     reward_kind=chosen.kind)
        session.tap(buy[0] / 2064, buy[1] / 1191, decision.reason)
        time.sleep(1)
        image = session.observe("shop-after-purchase")
        confirm_pending(session, image)
        image = session.observe('shop-purchase-confirmed')
        held, capacity = consumable_counter(image)
        state = collect_shop_jokers(session, image, replace(state, **read_numbers(image),
                              consumables_held=held, consumable_slots=capacity))
        expected_jokers = before_count + (chosen.kind == 'joker')
        expected_consumables = before_consumables + (chosen.kind != 'joker')
        if len(state.jokers) != expected_jokers or held != expected_consumables:
            session.log('purchase_type_mismatch', expected=chosen.kind, before=before_count,
                        after=len(state.jokers), consumables=held)
            session.policy_state = state
            break
        actual_names = [joker.name for joker in state.jokers]
        if chosen.kind == 'joker':
            expected_key = chosen.joker.key
            old_count = sum(j.key == expected_key for j in before_jokers)
            new_count = sum(j.key == expected_key for j in state.jokers)
            if new_count != old_count + 1:
                session.policy_state = state
                session.log('purchase_identity_mismatch', expected=chosen.name, actual=actual_names)
                raise ObservationError('Purchased Joker identity was not confirmed by its count increase')
        session.policy_state = state
    session.policy_state = state
    from shop_extras import shop_ready
    next_frame = session.observe('shop-before-next-round')
    if not shop_ready(next_frame):
        session.log('shop_exit_deferred', reason='actual Next Round button is not visible')
        return
    session.tap(.36, .43, "next round")


def prepare_new_run(session, image):
    """Select New Run explicitly; the main menu defaults to Continue."""
    words = read_text(image, (520, 110, 1700, 1100))
    tab = next((p for text, score, p in words
                if text.strip().lower() == 'new run' and score > .85 and p[1] < 400), None)
    if tab is None:
        raise ObservationError('New Run tab is not readable; refusing to continue an old save')
    session.tap(tab[0] / 2064, tab[1] / 1191, 'select New Run tab explicitly')
    words = read_text(session.observe('verify-new-run-setup'), (520, 110, 1700, 1100))
    labels = ' '.join(tooltip_labels(words)).lower()
    play = next((p for text, score, p in words
                 if text.strip().lower() == 'play' and score > .85 and p[1] > 700), None)
    if 'seeded run' not in labels or 'best hand' in labels or play is None:
        raise ObservationError('Fresh run settings are not verified')
    session.log('new_run_setup_verified', labels=tooltip_labels(words))
    return play[0] / 2064, play[1] / 1191


def reset_game_state(session):
    """Clear per-game observations, never an unconfirmed action journal."""
    session.policy_state = RunState(objective='survival')
    session.joker_cache = None
    session.hand_cache = None
    session.extra_offer_cache = None
    session.hidden_joker_roster = None
    session.blocked_consumable_orders = set()
    session.scene_hand_frame = None
    session.boss_after_sale_frame = None
    session.play_order_fallback = False
    session._last_observation = None
    session.refresh_levels = True
    session.levels_dirty = True
    session.levels_cache = {}


def run_batch(session, target_games, max_actions=None, resume=False):
    if max_actions is None:
        max_actions = max(2000, target_games * 2000)
    finished, unknowns, stagnant = 0, 0, 0
    last_signature = None
    consecutive_recoveries, total_recoveries = 0, 0
    awaiting_new_game, game_active = False, False
    starting_fresh = not resume
    reset_game_state(session)
    if resume and (session.run_dir / 'status.json').exists():
        previous = json.loads((session.run_dir / 'status.json').read_text(encoding='utf-8'))
        finished = previous['finished_games']
        awaiting_new_game = previous.get('awaiting_new_game', False)
        game_active = previous.get('game_active', not awaiting_new_game)
        total_recoveries = previous.get('automatic_recoveries', 0)
        starting_fresh = previous.get('starting_fresh', False)
        checkpoint = session.run_dir / 'policy-state.pkl'
        state_json = session.run_dir / 'policy-state.json'
        if state_json.exists():
            session.policy_state = state_from_dict(json.loads(state_json.read_text(encoding='utf-8')))
        elif checkpoint.exists():
            session.policy_state = pickle.loads(checkpoint.read_bytes())
    session.policy_state = replace(session.policy_state, objective='survival')
    sync_cache(session)
    session.log("batch_resume" if resume else "batch_start", target_games=target_games,
                finished_games=finished, action_limit=max_actions, code_version=code_version())
    session.current_game_number = min(finished + 1, target_games)

    def preserve_history():
        try:
            export_run(session.run_dir)
        except Exception as error:
            # Preserve gameplay/logging when a table is open in Excel or the
            # destination is unavailable. The JSONL can be exported afterward.
            session.log('history_export_failed', error=repr(error))

    def save_status(result="running", error=None):
        status = {"target_games": target_games, "finished_games": finished,
            "action_limit": max_actions,
            "current_game": min(finished + 1, target_games), "status": result,
            'awaiting_new_game': awaiting_new_game, 'game_active': game_active,
            'automatic_recoveries': total_recoveries,
            'issue_counts': issue_counts(session),
            'starting_fresh': starting_fresh,
            'pending_action': (getattr(session, 'pending_action', None) or {}).get('kind')}
        if error:
            status["error"] = error
        atomic_json(session.run_dir / 'policy-state.json', asdict(session.policy_state))
        atomic_json(session.run_dir / 'status.json', status)

    def wait_for_desktop(error):
        # Focus can also fail while recovering a different observation error.
        # In both places it must wait without clearing pending transactions.
        record_issue(session, 'desktop_wait_started', level='waiting', code='desktop_unavailable',
            message='Wait until the desktop can receive input', error=str(error))
        save_status('waiting_for_desktop', str(error))
        delay = 5
        while True:
            time.sleep(delay)
            try:
                session.focus()
                break
            except DesktopUnavailableError:
                delay = min(30, delay * 2)
        session.log('desktop_access_restored')
        save_status()

    def finish_game(result):
        nonlocal finished, awaiting_new_game, game_active, consecutive_recoveries
        if awaiting_new_game:
            return finished >= target_games
        finished += 1
        awaiting_new_game, game_active = True, False
        if consecutive_recoveries:
            session.log('recovery_resolved', attempts=consecutive_recoveries, resolved_by='game_end')
            consecutive_recoveries = 0
        session.log("game_end", number=finished, result=result, automatic_recoveries=total_recoveries)
        finish_issue_review(session, finished, result)
        preserve_history()
        reset_game_state(session)
        save_status("complete" if finished >= target_games else "running")
        if finished >= target_games:
            session.log("batch_complete", finished_games=finished)
            preserve_history()
            return True
        session.current_game_number = finished + 1
        return False

    save_status()
    if finished >= target_games:
        save_status('complete')
        return
    try:
        for step in range(max_actions):
            save_status()
            try:
                if getattr(session, 'pending_action', None):
                    confirm_pending(session)
                image = session.observe(f"step-{step}")
                image, scene = observe_scene(session, image)
                session.log("scene", name=scene, scores={k: round(v, 3) for k, v in scene_scores(image).items()})
                signature = (scene, bytes((image[100:1130:35, 120:1950:35] // 32).flatten()))
                stagnant = stagnant + 1 if signature == last_signature else 0
                last_signature = signature
                if stagnant >= 12:
                    raise ObservationError(f"Unchanged {scene} screen persisted")
                if starting_fresh:
                    # A new batch starts from a fresh deck setup. An old save
                    # or an old Game Over screen never counts as its first run.
                    if scene == 'deck':
                        starting_fresh = False
                    elif scene == 'main':
                        session.tap(.29, .85, 'open fresh run setup')
                        continue
                    elif scene == 'loss':
                        session.tap(.63, .79, 'open setup without counting the previous game')
                        continue
                    elif scene == 'win':
                        button = victory_new_run(read_text(image, (950, 125, 1700, 1100)))
                        if button is None:
                            raise ObservationError('Previous victory New Run button is not verified')
                        session.tap(button[0] / 2064, button[1] / 1191,
                                    'open setup without counting the previous victory')
                        continue
                    elif scene in ('hand', 'shop', 'blind', 'cashout', 'unknown'):
                        words = read_text(image, (520, 110, 1700, 1100))
                        button = next((p for text, score, p in words
                                       if text.strip().lower() == 'new run' and score > .85), None)
                        if button:
                            session.tap(button[0] / 2064, button[1] / 1191, 'choose New Run instead of continuing a saved run')
                        elif scene in ('hand', 'shop', 'blind', 'cashout'):
                            session.tap(.09, .85, 'open Options to start a fresh full game')
                        else:
                            time.sleep(1)
                        continue
                if scene == "loss":
                    record_round_snapshot(session, image, scene)
                    if finish_game("loss"):
                        return
                    session.tap(.63, .79, "open New Run setup")
                elif scene == 'win':
                    button = victory_new_run(read_text(image, (950, 125, 1700, 1100)))
                    if button is None:
                        raise ObservationError('Victory panel and New Run button are not verified')
                    record_round_snapshot(session, image, scene)
                    if finish_game('win'):
                        return
                    session.tap(button[0] / 2064, button[1] / 1191, 'open next full game after victory')
                elif scene == "deck":
                    play_button = prepare_new_run(session, image)
                    if not game_active:
                        session.log("game_start", number=finished + 1, deck="Red Deck", stake="White Stake")
                        game_active, awaiting_new_game = True, False
                        session.policy_state = replace(session.policy_state,
                            deck_cards=tuple(Card(rank, suit) for suit in SUITS for rank in range(2, 15)))
                        session.policy_state = fresh_levels(session.policy_state)
                        sync_cache(session)
                    session.tap(*play_button, "start verified fresh run")
                elif scene == "guide":
                    session.tap(.90, .35, "dismiss tutorial prompt")
                elif scene == "unlock":
                    session.tap(.50, .75, "dismiss unlocked Joker notice")
                elif scene == "unlock_reward":
                    session.tap(.50, .71, "dismiss collection unlock")
                elif scene == "main":
                    session.tap(.29, .85, "start game")
                elif scene == "blind":
                    blind = active_blind(image)
                    upcoming = read_future_boss(image)
                    numbers = read_numbers(image)
                    ante = numbers.get('ante', session.policy_state.ante)
                    from boss_policy import BOSS_TACTICS, future_boss_weight, remember_upcoming_boss
                    session.policy_state = remember_upcoming_boss(session.policy_state, ante, upcoming)
                    from strategy import joker_name
                    known = session.policy_state.upcoming_boss
                    tactic = BOSS_TACTICS.get(joker_name(known).removeprefix('the_'))
                    session.log("blind_decision", blind=blind, upcoming_boss=upcoming,
                                boss_tactic=tactic, future_boss_weight=future_boss_weight(ante),
                                reason="play available blind")
                    session.tap((.36, .54, .72)[blind], .35, "select blind")
                    session.policy_state = replace(session.policy_state, round_score=0,
                        round_played_types={}, boss_seen_types=frozenset(),
                        boss_first_hand_done=False, seen_cards=(),
                        boss=(session.policy_state.upcoming_boss if blind == 2 and
                              session.policy_state.upcoming_boss_ante == ante else ""),
                        boss_target_hand="", boss_disabled=False, shop_rerolls=0,
                        round_resources_observed=False, round_discards_used=0, blind_index=blind, ante=ante)
                    sync_cache(session)
                elif scene == "hand":
                    session.scene_hand_frame = image
                    play_one_hand(session)
                elif scene == "cashout":
                    record_round_snapshot(session, image, scene)
                    session.tap(.54, .42, "cash out")
                elif scene == "shop":
                    shop(session, image)
                elif scene == 'pack':
                    handle_pack(session, image)
                else:
                    words = read_text(image, (550, 100, 1700, 1100)) if unknowns >= 2 else []
                    new_run = victory_new_run(words)
                    if new_run:
                        record_round_snapshot(session, image, 'win')
                        if finish_game("win"):
                            return
                        session.tap(new_run[0] / 2064, new_run[1] / 1191, "open next full game")
                    else:
                        unknowns += 1
                        if unknowns >= 20:
                            raise ObservationError("Unknown screen persisted for twenty observations")
                        time.sleep(1)
                        continue
                if consecutive_recoveries:
                    session.log('recovery_resolved', attempts=consecutive_recoveries, resolved_by=scene)
                unknowns, consecutive_recoveries = 0, 0
                time.sleep(1)
            except DesktopUnavailableError as error:
                # Locking the desktop or temporary focus denial is not a failed
                # game action. Keep the journal and retry budget intact.
                wait_for_desktop(error)
                continue
            except ObservationError as error:
                consecutive_recoveries += 1
                total_recoveries += 1
                record_issue(session, 'automatic_recovery', level='recoverable',
                    code=recovery_code(error, getattr(session, 'pending_action', None)),
                    message='Reobserve and recover in the current game',
                    attempt=consecutive_recoveries, error=str(error),
                    pending_kind=(getattr(session, 'pending_action', None) or {}).get('kind'))
                save_status('recovering', str(error))
                if consecutive_recoveries > 5:
                    raise RuntimeError('Automatic recovery exhausted after five attempts') from error
                try:
                    recover_view(session, error=error)
                except DesktopUnavailableError as recovery_error:
                    wait_for_desktop(recovery_error)
                except ObservationError as recovery_error:
                    session.log('recovery_read_failed', error=str(recovery_error))
                unknowns, stagnant, last_signature = 0, 0, None
                time.sleep(1)
                continue
        raise RuntimeError(f"Action limit reached with {finished} completed games")
    except Exception as error:
        record_issue(session, 'batch_error', level='fatal', code='batch_execution_failed',
                     message='Batch cannot safely continue', error=repr(error))
        save_status("failed", repr(error))
        preserve_history()
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["one-hand", "batch"])
    parser.add_argument("--discards", type=int, default=4)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--run-dir", type=str)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-actions", type=int, default=None)
    args = parser.parse_args()
    session = Session(Path(args.run_dir).resolve() if args.run_dir else RUNS / datetime.now().strftime("test-%Y%m%d-%H%M%S"))
    try:
        if args.command == "one-hand":
            play_one_hand(session, args.discards)
        else:
            run_batch(session, args.games, max_actions=args.max_actions, resume=args.resume)
    except Exception as error:
        session.log("fatal", error=repr(error))
        raise


if __name__ == "__main__":
    main()
