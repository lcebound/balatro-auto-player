"""Bounded observation recovery and a durable journal for committed clicks."""
from dataclasses import asdict
from pathlib import Path
import json
import os
import re
import time

from strategy import Card, Joker, RunState, joker_name


class ObservationError(RuntimeError):
    """Read/verification failure: take a new screenshot before taking action."""


class DesktopUnavailableError(ObservationError):
    """The game cannot receive input; wait without spending gameplay retries."""


class HandPreviewError(ObservationError):
    """A pre-click preview failed; only an unreadable title can retain caches."""

    def __init__(self, message, *, cache_safe=False):
        super().__init__(message)
        self.cache_safe = cache_safe


class HandPreviewConflict(ObservationError):
    """Readable competing preview categories must be checked in a new frame."""


def json_default(value):
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f'Cannot serialize {type(value).__name__}')


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   default=json_default), encoding='utf-8')
    os.replace(temporary, path)


def state_from_dict(data):
    data = dict(data)
    for key in ('remaining_deck', 'deck_cards', 'seen_cards', 'pillar_cards'):
        data[key] = tuple(Card(**card) for card in data.get(key, ()))
    data['jokers'] = tuple(Joker(**dict(joker, data=tuple(tuple(v) for v in joker.get('data', ()))))
                           for joker in data.get('jokers', ()))
    data['boss_seen_types'] = frozenset(data.get('boss_seen_types', ()))
    data['owned_vouchers'] = tuple(data.get('owned_vouchers', ()))
    data['uncertain_levels'] = tuple(data.get('uncertain_levels', ()))
    if 'unverified_chip_cards' not in data and data.get('total_hands_played', 0):
        # Old checkpoints have no provenance for invisible chip changes.
        data['unverified_chip_cards'] = tuple(c.identity for c in data['deck_cards'])
    else:
        data['unverified_chip_cards'] = tuple(tuple(v) for v in data.get('unverified_chip_cards', ()))
    return RunState(**data)


def begin_action(session, kind, before, after_state=None, **details):
    if getattr(session, 'pending_action', None):
        raise ObservationError('Previous click still needs outcome confirmation')
    pending = {'kind': kind, 'before': before, 'details': details,
               'after_state': asdict(after_state) if after_state is not None else None}
    atomic_json(session.run_dir / 'pending-action.json', pending)
    session.pending_action = pending
    session.log('action_prepared', kind=kind, before=before, details=details)


def finish_action(session, applied, **details):
    pending = getattr(session, 'pending_action', None)
    if pending is None:
        return
    if applied and pending['kind'] in ('buy', 'sell', 'consume', 'redeem', 'open_pack', 'pick_pack', 'skip_pack'):
        from joker_cache import dirty_jokers
        kind, action_details = pending['kind'], pending.get('details', {})
        reward = action_details.get('reward_kind')
        name = action_details.get('name')
        # Preserve old cards across known additions/removals. The observer
        # reconciles their actual appearances and reads only unmatched slots.
        # Unknown transactions and same-count random replacements stay strict.
        unsafe = (not action_details or (kind == 'buy' and not reward)
                  or (kind == 'consume' and not action_details.get('consumable_kind'))
                  or joker_name(name or '') in ('ankh', 'hex', 'ectoplasm', 'the_soul', 'soul', 'wraith'))
        if unsafe:
            session.joker_cache = None
        elif kind == 'buy' and reward == 'joker':
            dirty_jokers(session, {'swashbuckler'})
        elif kind == 'sell':
            dirty_jokers(session, {'swashbuckler', 'campfire'})
        elif kind == 'skip_pack':
            dirty_jokers(session, {'red_card'})
        elif kind == 'consume' and action_details.get('consumable_kind') == 'planet':
            dirty_jokers(session, {'constellation'})
        elif kind == 'pick_pack' and reward == 'planet':
            dirty_jokers(session, {'constellation'})
        elif kind == 'pick_pack' and reward == 'joker':
            dirty_jokers(session, {'swashbuckler'})
        elif kind in ('consume', 'pick_pack'):
            dirty_jokers(session)
        elif kind == 'redeem' and joker_name(name or '') in ('clearance_sale', 'liquidation'):
            cache = getattr(session, 'joker_cache', None)
            if cache:
                dirty_jokers(session, {j.key for j in cache.jokers})
        session.hidden_joker_roster = None
    if applied:
        from hand_cache import finish_hand_action
        from joker_cache import dirty_jokers, dirty_hand_action
        finish_hand_action(session, pending)
        if pending['kind'] in ('play', 'discard'):
            dirty_hand_action(session, pending['kind'])
        elif pending['kind'] == 'reroll':
            dirty_jokers(session, {'flash_card'})
    if applied and pending.get('after_state') is not None:
        old_levels = dict(getattr(getattr(session, 'policy_state', None), 'levels', {}))
        session.policy_state = state_from_dict(pending['after_state'])
        # Persist the confirmed history before retiring its journal. Replaying
        # the same after_state after a crash is idempotent; losing it is not.
        atomic_json(session.run_dir / 'policy-state.json', asdict(session.policy_state))
        changes = {k: v for k, v in session.policy_state.levels.items() if old_levels.get(k, 1) != v}
        if changes:
            session.log('hand_levels_local', source=pending['kind'], levels=changes,
                        uncertain_levels=session.policy_state.uncertain_levels)
    if applied and hasattr(session, 'policy_state'):
        from hand_levels import sync_cache, uncertain
        # Legacy journals may have no projected state. Do not infer receipt
        # from a planned click or silently trust incomplete old checkpoints.
        if (pending['kind'] in ('consume', 'pick_pack')
                and pending.get('details', {}).get('levels_changed', True)
                and pending.get('after_state') is None):
            from strategy import HAND_LEVELS
            session.policy_state = uncertain(session.policy_state, HAND_LEVELS)
            atomic_json(session.run_dir / 'policy-state.json', asdict(session.policy_state))
        sync_cache(session)
    session.log('action_confirmed' if applied else 'action_not_applied',
                kind=pending['kind'], **details)
    atomic_json(session.run_dir / 'pending-action.json', {'kind': None})
    session.pending_action = None


def action_outcome(pending, scene, values):
    """True applied, False observably unchanged, None cannot decide yet."""
    kind, before = pending['kind'], pending['before']
    if kind == 'open_pack':
        details = pending['details']
        if (scene == 'pack' and values.get('pack_kind') == details['pack_kind']
                and values.get('money') == before['money'] - details['price']):
            return True
        return False if scene == 'shop' and values.get('money') == before['money'] else None
    if kind == 'redeem':
        if (scene == 'shop' and values.get('money') == before['money'] - pending['details']['price']
                and values.get('voucher_price_visible') is False):
            return True
        return False if scene == 'shop' and values.get('money') == before['money'] else None
    if kind in ('pick_pack', 'skip_pack'):
        reward_kind = pending.get('details', {}).get('reward_kind')
        receipt_key = {'joker': 'joker_count', 'card': 'deck_size'}.get(reward_kind)
        if receipt_key and values.get(receipt_key) != before[receipt_key] + 1:
            return None
        if scene == 'shop':
            return True
        if scene == 'pack':
            if values.get('pack_kind') != pending.get('details', {}).get('pack_kind', values.get('pack_kind')):
                return None
            remaining = values.get('pack_remaining')
            if kind == 'pick_pack' and remaining == before['pack_remaining'] - 1:
                return True
            if remaining == before['pack_remaining']:
                return False
        return None
    if kind == 'play' and scene in ('cashout', 'loss', 'shop', 'blind', 'win'):
        return True
    if kind in ('play', 'discard'):
        key = 'hands_left' if kind == 'play' else 'discards_left'
        if key not in values:
            return None
        if values[key] < before[key]:
            return True
        if kind == 'play' and values.get('round_score', -1) > before['round_score']:
            return True
        return False if scene == 'hand' and values[key] == before[key] else None
    if kind == 'buy':
        if scene != 'shop':
            return None
        details = pending.get('details', {})
        if details.get('reward_kind') and 'price' in details:
            joker_delta = int(details['reward_kind'] == 'joker')
            expected = {'joker_count': before['joker_count'] + joker_delta,
                        'consumable_count': before['consumable_count'] + 1 - joker_delta,
                        'money': before['money'] - details['price']}
            if all(values.get(k) == v for k, v in expected.items()):
                return True
            if all(k in values and values[k] == before[k] for k in expected):
                return False
            return None
        if any(values.get(key, -1) > before[key] for key in ('joker_count', 'consumable_count')):
            return True
        if 'money' in values and values['money'] < before['money']:
            return True
        if all(key in values and values[key] == before[key]
               for key in ('money', 'joker_count', 'consumable_count')):
            return False
        return None
    if kind == 'consume':
        if 'consumable_count' not in values:
            return None
        count = values['consumable_count']
        return True if count < before['consumable_count'] else (False if count == before['consumable_count'] else None)
    if (kind == 'sell' and pending.get('details', {}).get('purpose') == 'disable_verdant_leaf'):
        if scene != 'hand' or 'joker_count' not in values:
            return None
        if values['joker_count'] == before['joker_count'] - 1:
            return True
        return False if values['joker_count'] == before['joker_count'] else None
    if kind == 'sell' and scene == 'shop':
        if values.get('joker_count', before['joker_count']) < before['joker_count']:
            return True
        if values.get('money', before['money']) > before['money']:
            return True
        return False if all(key in values and values[key] == before[key] for key in ('money', 'joker_count')) else None
    if kind == 'reroll' and scene == 'shop':
        if values.get('money', before['money']) < before['money']:
            return True
        if values.get('reroll_cost', before['reroll_cost']) > before['reroll_cost']:
            return True
        return False if all(key in values and values[key] == before[key] for key in ('money', 'reroll_cost')) else None
    if kind in ('sell', 'reroll'):
        return None
    raise ValueError(f'Unsupported pending action {kind}')


def confirm_pending(session, image=None):
    """Confirm a click without pressing its button again during animation."""
    from perception import read_numbers, joker_counter, consumable_counter, read_reroll_button
    from vision import classify_scene, scene_scores
    pending = getattr(session, 'pending_action', None)
    if pending is None:
        return
    unchanged = 0
    for attempt in range(4):
        current = image if attempt == 0 and image is not None else session.observe('confirm-pending-action')
        values = None
        if pending['kind'] in ('play', 'discard'):
            # The existing receipt rule accepts an observed resource decrease
            # even during animation. Prove it first instead of identifying all
            # cards just to name the scene. No estimated score is a receipt.
            values = read_numbers(current)
            if action_outcome(pending, 'unknown', values) is True:
                session.log('action_outcome_read', kind=pending['kind'], scene='unknown',
                            counters=values, outcome=True, receipt_source='observed_counters')
                finish_action(session, True, scene='unknown', counters=values)
                return
        scene = classify_scene(current)
        if scene == 'unknown' and scene_scores(current).get('shop', 0) > .55:
            # Selling shifts neighboring Jokers under the cursor; their new
            # tooltip can hide Next Round and hence the scene receipt. Clear
            # that overlay without issuing a second transaction click.
            session.hover(.93, .55, 'dismiss tooltip before verifying transaction receipt')
            current = session.observe('confirm-pending-without-tooltip')
            scene = classify_scene(current)
            values = None
        if values is None:
            values = read_numbers(current)
        if pending['kind'] in ('redeem', 'open_pack', 'pick_pack', 'skip_pack'):
            from shop_extras import extra_outcome_values
            values.update(extra_outcome_values(current, pending))
            if (pending['kind'] == 'pick_pack'
                    and pending.get('details', {}).get('reward_kind') == 'joker'):
                try:
                    values['joker_count'] = joker_counter(current)[0]
                except ObservationError:
                    pass
        if pending['kind'] in ('buy', 'consume', 'sell'):
            try:
                values['consumable_count'] = consumable_counter(current)[0]
                if pending['kind'] in ('buy', 'sell'):
                    values['joker_count'] = joker_counter(current)[0]
            except ObservationError:
                pass
        if pending['kind'] == 'reroll':
            button = read_reroll_button(current)
            if button is not None:
                values['reroll_cost'] = button[1]
        outcome = action_outcome(pending, scene, values)
        session.log('action_outcome_read', kind=pending['kind'], scene=scene, counters=values, outcome=outcome)
        if outcome is True:
            finish_action(session, True, scene=scene, counters=values)
            return
        unchanged = unchanged + 1 if outcome is False else 0
        if unchanged >= 3:
            finish_action(session, False, scene=scene)
            raise ObservationError('Button click did not change visible counters; recollect before retry')
        time.sleep(1)
    raise ObservationError('Click outcome still unreadable; pending action retained')


def run_info_back(image):
    """Recognize the complete Run Info navigation, not an arbitrary Back."""
    from perception import read_text
    words = [word for word in read_text(image, (520, 130, 1600, 270)) if word[1] > .85]
    header = ' '.join(word[0].strip().lower() for word in sorted(words, key=lambda w: w[2][0]))
    # OCR may split Poker / Hands in a narrow header crop.
    if not all(re.search(pattern, header) for pattern in
               (r'\bpoker\s+hands\b', r'\bblinds\b', r'\bvouchers\b')):
        return None
    buttons = [p for text, score, p in read_text(image, (900, 960, 1150, 1075))
               if text.strip().lower() == 'back' and score > .85]
    return buttons[0] if len(buttons) == 1 else None


def recover_view(session, *, error=None):
    """Dismiss recognized overlays and request a fresh decision state."""
    from perception import read_text
    from vision import classify_scene
    preserve = (isinstance(error, HandPreviewError) and error.cache_safe
                and not getattr(session, 'pending_action', None))
    def invalidate():
        session.joker_cache = None
        session.hand_cache = None
        session.extra_offer_cache = None
    if not preserve:
        invalidate()
    try:
        image = session.observe('automatic-recovery')
    except ObservationError:
        if not hasattr(session, 'restore_geometry'):
            raise
        preserve = False
        invalidate()
        session.restore_geometry()
        image = session.observe('automatic-recovery-restored-window')
    scene = classify_scene(image)
    if preserve and scene == 'hand':
        # The failed preview preceded begin_action, so no game action was
        # committed. Existing caches still require their normal fresh-pixel,
        # count, round, metadata and dirty-effect checks on the next read.
        session.hover(.93, .55, 'dismiss tooltip during preview recovery')
        from hand_levels import sync_cache
        sync_cache(session)
        session.log('preview_recovery_cache_retained', scene=scene)
        return
    invalidate()
    modal_buttons = {'guide': (.90, .35), 'unlock': (.50, .75), 'unlock_reward': (.50, .71)}
    if scene in modal_buttons:
        session.tap(*modal_buttons[scene], 'dismiss recognized notice during recovery')
        from hand_levels import sync_cache
        sync_cache(session)
        return
    words = read_text(image, (520, 110, 1600, 1100))
    back = next((p for text, score, p in words if text.strip().lower() == 'back' and score > .85), None)
    if back is not None:
        session.tap(back[0] / 2064, back[1] / 1191, 'close recognized overlay during recovery')
    session.hover(.93, .55, 'dismiss tooltip during recovery')
    from hand_levels import sync_cache
    sync_cache(session)
