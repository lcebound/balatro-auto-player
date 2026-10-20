"""Bounded alternative reads, scoped recovery and isolation between games."""
import json
import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

import runner
from controller import verify_play_preview
from perception import read_hand_preview
from recovery import HandPreviewError, HandPreviewConflict, ObservationError, recover_view, finish_action, atomic_json
from strategy import RunState, ShopItem, ShopDecision
from tests.test_runtime_recovery import FakeSession
from tools.replay_observation_speed import load_frame

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / 'tests/fixtures/recognition_review_20261017.json').read_text('utf-8'))


class ObservedSession(FakeSession):
    def __init__(self, directory):
        super().__init__(directory)
        self.observations = []

    def observe(self, label):
        self.observations.append(label)
        return self.frame


def shop_scenario(mode='same', *, shop_factory=None, real_ocr=False):
    """Two shop evaluations separated by an observed lower-row purchase/reroll."""
    import perception
    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
        session = ObservedSession(directory)
        case = next(c for c in FIXTURE['frames'] if c.get('expected') == 'Flower Pot')
        session.frame = load_frame(ROOT / case['path'])
        state = RunState(money=20)
        session.policy_state = state
        price = [6]
        calls = dict(ocr=0, hovers=0)
        session.hover = lambda *args: calls.__setitem__('hovers', calls['hovers'] + 1)

        def text(image, region=None, **kwargs):
            if region == (960, 430, 1930, 750):
                return [(f'${price[0]}', .99, (1161, 450))]
            calls['ocr'] += 1
            if real_ocr:
                perception._TEXT_CACHE.clear()
                perception._TEXT_CACHE_BYTES = 0
                return perception.read_text(image, region, **kwargs)
            return [('Not Discovered', .99, (890, 500))]

        def purchase(s, *args):
            if mode == 'price':
                price[0] = 5
            elif mode == 'art':
                s.frame = s.frame.copy()
                tile = s.frame[530:650, 1116:1206]
                tile[:] = np.clip(tile.astype(np.int16) - 50, 0, 255).astype(np.uint8)
            return state

        first = ('reroll' if mode == 'reroll' else 'leave' if mode == 'new_shop' else 'buy')
        decisions = [ShopDecision(action=first, item_index=0), ShopDecision(action='leave')]
        for name, value in {
            'read_numbers': {'money': 20}, 'consumable_counter': (0, 2),
            'collect_shop_jokers': state, 'read_jokers': [],
            'shop_tooltip_region': (719, 466, 1068, 716),
            'read_shop_reroll': ((750, 620), 5), 'shop_strength': 1,
            'collect_extra_offers': ([ShopItem('voucher', 'Grabber', 10)], [(900, 800)]),
        }.items():
            stack.enter_context(patch.object(runner, name, return_value=value))
        stack.enter_context(patch.object(runner, 'read_text', side_effect=text))
        item_reader = stack.enter_context(patch.object(runner, 'shop_offer_item',
            side_effect=ObservationError('Joker tooltip name is unreadable or ambiguous')))
        stack.enter_context(patch.object(runner, 'purchase_extra', side_effect=purchase))
        stack.enter_context(patch.object(runner, 'confirm_pending', side_effect=lambda s: finish_action(s, True)))
        stack.enter_context(patch.object(runner, 'decide_shop', side_effect=decisions))
        shop = shop_factory() if shop_factory else runner.shop
        shop(session, session.frame)
        if mode == 'new_shop':
            shop(session, session.frame)
        calls['item_reads'] = item_reader.call_count
        calls['tooltip_captures'] = session.observations.count('shop-offer-tooltip')
        calls['reused'] = sum(e['event'] == 'shop_offer_unreadable_reused' for e in session.events)
        calls['pending'] = session.pending_action
        return calls


class PreviewBudgetTests(unittest.TestCase):
    def test_normal_preview_still_needs_one_read_without_wait_or_new_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            session = ObservedSession(directory)
            with patch('controller.read_hand_preview', return_value='pair') as read, \
                    patch('controller.time.sleep') as wait:
                verify_play_preview(session, session.frame, 'pair')
            read.assert_called_once_with(session.frame)
            wait.assert_not_called()
            self.assertEqual(session.observations, [])

    def test_failed_panel_uses_real_pixels_in_alternative_read_before_new_capture(self):
        case = next(c for c in FIXTURE['frames'] if c['role'] == 'fatal')
        with tempfile.TemporaryDirectory() as directory:
            session = ObservedSession(directory)
            session.frame = load_frame(ROOT / case['path'])
            def reader(image, **kwargs):
                if not kwargs.get('contrast'):
                    raise ObservationError('unreadable title')
                return read_hand_preview(image, **kwargs)
            with patch('controller.read_hand_preview', side_effect=reader) as read, \
                    patch('controller.time.sleep') as wait:
                verify_play_preview(session, session.frame, 'straight_flush')
            self.assertEqual(read.call_count, 2)
            wait.assert_not_called()
            self.assertEqual((session.observations, session.taps), ([], []))

    def test_read_budget_remains_three_and_last_read_uses_a_fresh_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            session = ObservedSession(directory)
            with patch('controller.read_hand_preview', side_effect=[ObservationError('first'),
                    ObservationError('alternative'), 'pair']) as read, patch('controller.time.sleep') as wait:
                verify_play_preview(session, session.frame, 'pair')
            self.assertEqual(read.call_count, 3)
            self.assertEqual(read.call_args_list[1].kwargs, {'contrast': True})
            self.assertEqual(read.call_args_list[2].kwargs, {})
            self.assertEqual(session.observations, ['verify-play-preview-again'])
            wait.assert_called_once_with(.3)

    def test_a_readable_conflicting_category_requires_fresh_observation(self):
        for conflict in ('flush', HandPreviewConflict('Pair and Flush')):
            with self.subTest(conflict=conflict), tempfile.TemporaryDirectory() as directory:
                session = ObservedSession(directory)
                with patch('controller.read_hand_preview', side_effect=[conflict, 'straight_flush']) as read, \
                        patch('controller.time.sleep'):
                    verify_play_preview(session, session.frame, 'straight_flush')
                self.assertTrue(all(not call.kwargs for call in read.call_args_list))
                self.assertEqual(session.observations, ['verify-play-preview-again'])

    def test_exhausted_unreadable_and_conflicting_previews_have_different_scopes(self):
        for outcome, safe in ((ObservationError('unreadable'), True), ('flush', False),
                              (HandPreviewConflict('Pair and Flush'), False),
                              (['flush', ObservationError('later unreadable'), ObservationError('still unreadable')], False)):
            with self.subTest(safe=safe), tempfile.TemporaryDirectory() as directory:
                session = ObservedSession(directory)
                kwargs = {'side_effect': outcome} if isinstance(outcome, (Exception, list)) else {'return_value': outcome}
                with patch('controller.read_hand_preview', **kwargs) as read, patch('controller.time.sleep'):
                    with self.assertRaises(HandPreviewError) as caught:
                        verify_play_preview(session, session.frame, 'straight_flush')
                self.assertEqual(caught.exception.cache_safe, safe)
                self.assertEqual(read.call_count, 3)
                self.assertFalse(session.taps)
                self.assertIsNone(session.pending_action)

    def test_hidden_preview_does_not_infer_a_known_category_or_preserve_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            session = ObservedSession(directory)
            with patch('controller.read_text', return_value=[]), \
                    patch('controller.read_hand_preview') as read, patch('controller.time.sleep'):
                with self.assertRaises(HandPreviewError) as caught:
                    verify_play_preview(session, session.frame, 'pair', hidden=True)
            self.assertFalse(caught.exception.cache_safe)
            read.assert_not_called()
            self.assertEqual(len(session.observations), 2)

    def test_contrast_reader_agrees_on_all_twenty_retained_previews(self):
        for case in FIXTURE['frames']:
            if case['kind'] == 'preview':
                with self.subTest(path=case['path']):
                    self.assertEqual(read_hand_preview(load_frame(ROOT / case['path']), contrast=True), case['expected'])


class ScopedRecoveryTests(unittest.TestCase):
    def test_retained_caches_still_reject_changed_card_and_joker_pixels(self):
        from hand_cache import HandCache, card_patch
        from joker_cache import JokerCache, round_token, roster_patch
        from strategy import Card, Joker, SUITS
        from vision import hand_slots
        case = next(c for c in FIXTURE['frames'] if c['role'] == 'fatal')
        image = load_frame(ROOT / case['path'])
        state = RunState(hand_size=9, jokers=tuple(Joker(n) for n in
            ('Raised Fist', 'Crafty Joker', 'Four Fingers', 'Flower Pot', 'Mad Joker')),
            deck_cards=tuple(Card(rank, suit) for suit in SUITS for rank in range(2, 15)))
        with tempfile.TemporaryDirectory() as directory:
            session = ObservedSession(directory)
            session.frame, session.policy_state = image, state
            session.joker_cache = JokerCache(round_token(state), 5, state.jokers,
                tuple(roster_patch(image, 5, i) for i in range(5)), frozenset({'crafty_joker'}))
            session.hand_cache = HandCache(round_token(state), (Card(2, 'S'),),
                (card_patch(image, (hand_slots(9)[8], 675)),))
            with patch('vision.classify_scene', return_value='hand'):
                recover_view(session, error=HandPreviewError('preview', cache_safe=True))
            changed = image.copy()
            changed[105:330, 535:1670] = 0
            changed[655:975, 535:2020] = 0
            self.assertFalse(session.joker_cache.reusable(changed, state, 5, 5))
            self.assertEqual(session.joker_cache.dirty, frozenset({'crafty_joker'}))
            self.assertEqual(session.hand_cache.verified_identities(changed, state, 9, 9), [None] * 9)

    def test_only_unreadable_precommit_hand_preview_can_retain_verified_caches(self):
        cases = [(HandPreviewError('preview', cache_safe=True), 'hand', None, True),
                 (HandPreviewError('preview', cache_safe=True), 'shop', None, False),
                 (HandPreviewError('preview', cache_safe=True), 'hand', {'kind': 'play'}, False),
                 (HandPreviewError('wrong category'), 'hand', None, False),
                 (ObservationError('unreadable cards'), 'hand', None, False)]
        for error, scene, pending, keep in cases:
            with self.subTest(scene=scene, keep=keep, error=type(error).__name__), tempfile.TemporaryDirectory() as directory:
                session = ObservedSession(directory)
                jokers, cards = object(), object()
                session.joker_cache, session.hand_cache = jokers, cards
                session.pending_action = pending
                with patch('vision.classify_scene', return_value=scene), \
                        patch('perception.read_text', return_value=[]) as text:
                    recover_view(session, error=error)
                self.assertIs(session.joker_cache, jokers if keep else None)
                self.assertIs(session.hand_cache, cards if keep else None)
                self.assertIs(session.pending_action, pending)
                self.assertEqual(text.call_count, 0 if keep else 1)
                self.assertEqual(len(session.observations), 1)
                self.assertFalse(session.taps)

    def test_geometry_restore_discards_cache_even_for_a_scoped_error(self):
        with tempfile.TemporaryDirectory() as directory:
            session = ObservedSession(directory)
            session.joker_cache, session.hand_cache = object(), object()
            session.restore_geometry = Mock()
            with patch.object(session, 'observe', side_effect=[ObservationError('size'), session.frame]), \
                    patch('vision.classify_scene', return_value='hand'), patch('perception.read_text', return_value=[]):
                recover_view(session, error=HandPreviewError('preview', cache_safe=True))
            self.assertIsNone(session.joker_cache)
            self.assertIsNone(session.hand_cache)
            session.restore_geometry.assert_called_once()


class GameIsolationTests(unittest.TestCase):
    def test_second_game_cannot_inherit_blocked_consumables_or_cached_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            session = ObservedSession(directory)
            games = []
            def hand(s):
                games.append(s.current_game_number)
                self.assertEqual(s.blocked_consumable_orders, set())
                self.assertIsNone(s.hidden_joker_roster)
                self.assertIsNone(s.extra_offer_cache)
                self.assertEqual(s.policy_state.played_types, {})
                s.blocked_consumable_orders.add(('death', 'same hand'))
                s.hidden_joker_roster = ('old blind',)
                s.extra_offer_cache = {'items': ['old shop']}
            with patch('runner.prepare_new_run', return_value=(.5, .79)), \
                    patch('runner.classify_scene', side_effect=['deck', 'hand', 'loss', 'deck', 'hand', 'loss']), \
                    patch('runner.scene_scores', return_value={}), patch('runner.time.sleep'), \
                    patch('runner.play_one_hand', side_effect=hand):
                runner.run_batch(session, 2, max_actions=10)
            self.assertEqual(games, [1, 2])
            self.assertEqual(len([e for e in session.events if e['event'] == 'game_end']), 2)

    def test_reset_never_discards_a_pending_transaction(self):
        with tempfile.TemporaryDirectory() as directory:
            session = ObservedSession(directory)
            pending = {'kind': 'buy', 'before': {'money': 20}}
            session.pending_action = pending
            atomic_json(session.run_dir / 'pending-action.json', pending)
            runner.reset_game_state(session)
            self.assertIs(session.pending_action, pending)
            self.assertEqual(json.loads((session.run_dir / 'pending-action.json').read_text('utf-8')), pending)


class UnreadableOfferTests(unittest.TestCase):
    def test_same_rejected_offer_does_not_repeat_hover_capture_and_ocr(self):
        result = shop_scenario()
        self.assertEqual((result['item_reads'], result['tooltip_captures'], result['reused']), (1, 1, 1))
        self.assertIsNone(result['pending'])

    def test_reroll_changed_art_changed_price_and_new_shop_require_a_new_read(self):
        for mode in ('reroll', 'art', 'price', 'new_shop'):
            with self.subTest(mode=mode):
                result = shop_scenario(mode)
                self.assertEqual((result['item_reads'], result['tooltip_captures']), (2, 2))
                self.assertIsNone(result['pending'])


if __name__ == '__main__':
    unittest.main()
