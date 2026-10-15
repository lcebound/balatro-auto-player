"""Offline guards for faster scene/receipt reads and action-specific caches."""
import tempfile
import time
import unittest
from pathlib import Path
from dataclasses import replace
from unittest.mock import Mock, patch

import numpy as np

from bot import Session
from joker_cache import JokerCache, dirty_hand_action, dirty_jokers, round_token
from recovery import ObservationError, begin_action, confirm_pending
from strategy import Joker, RunState
from tests.test_runtime_recovery import FakeSession
from vision import playable_hand_visible, hand_controls_visible, classify_scene
from tools.replay_observation_speed import load_frame


class HandSceneTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((1191, 2064, 3), dtype=np.uint8)
    def visible(self, controls=True, count=(8, 8)):
        with patch('vision.hand_controls_visible', return_value=controls), \
                patch('perception.hand_counter', return_value=count):
            return playable_hand_visible(self.frame)

    def test_controls_and_actual_count_prove_standard_and_large_hands(self):
        for count in ((8, 8), (13, 13), (3, 8)):
            self.assertTrue(self.visible(count=count))

    def test_partial_controls_empty_and_invalid_counts_reject(self):
        self.assertFalse(self.visible(controls=False))
        self.assertFalse(self.visible(count=(0, 8)))
        self.assertFalse(self.visible(count=(13, 8)))

    def test_unreadable_count_keeps_full_reader_fallback(self):
        with patch('vision.hand_controls_visible', return_value=True), \
                patch('perception.hand_counter', side_effect=ObservationError('unreadable')):
            self.assertFalse(playable_hand_visible(self.frame))

    def test_actual_grey_controls_classify_without_full_card_scan(self):
        frame = load_frame(Path(__file__).parent / 'fixtures/hand_latency_ready_20261011.png')
        self.assertTrue(hand_controls_visible(frame))
        with patch('vision.read_hand', side_effect=AssertionError('duplicate full card scan')):
            self.assertEqual(classify_scene(frame), 'hand')

    def test_each_required_control_must_be_visible(self):
        original = load_frame(Path(__file__).parent / 'fixtures/hand_latency_ready_20261011.png')
        for x, y, right, bottom in ((800, 1007, 990, 1075),
                                    (1250, 1007, 1440, 1075), (1030, 990, 1215, 1040)):
            frame = original.copy()
            frame[y-15:bottom+15, x-15:right+15] = 0
            self.assertFalse(hand_controls_visible(frame))


class ReceiptTests(unittest.TestCase):
    def test_decrease_confirms_once_without_card_scan_or_extra_click(self):
        for kind, key in (('play', 'hands_left'), ('discard', 'discards_left')):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                session = FakeSession(directory)
                begin_action(session, kind, {key: 4, 'round_score': 0})
                with patch('perception.read_numbers', return_value={key: 3}), \
                        patch('vision.classify_scene', side_effect=AssertionError('unnecessary scan')):
                    confirm_pending(session)
                    confirm_pending(session)
                self.assertIsNone(session.pending_action)
                self.assertFalse(session.taps)
                self.assertEqual(sum(e['event'] == 'action_confirmed' for e in session.events), 1)

    def test_unchanged_or_missing_resource_requires_scene_receipt(self):
        for numbers in ({'hands_left': 4, 'round_score': 0}, {}):
            with self.subTest(numbers=numbers), tempfile.TemporaryDirectory() as directory:
                session = FakeSession(directory)
                begin_action(session, 'play', {'hands_left': 4, 'round_score': 0})
                with patch('perception.read_numbers', return_value=numbers), \
                        patch('vision.classify_scene', return_value='cashout') as scene:
                    confirm_pending(session)
                scene.assert_called_once()
                self.assertIsNone(session.pending_action)

    def test_animation_without_receipt_retains_transaction_without_click(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            begin_action(session, 'discard', {'discards_left': 4})
            with patch('perception.read_numbers', return_value={}), \
                    patch('vision.classify_scene', return_value='unknown'), \
                    patch('vision.scene_scores', return_value={'shop': 0}), patch('recovery.time.sleep'):
                with self.assertRaises(ObservationError):
                    confirm_pending(session)
            self.assertIsNotNone(session.pending_action)
            self.assertFalse(session.taps)


class FrameReuseTests(unittest.TestCase):
    def setUp(self):
        self.session = Session.__new__(Session)
        self.frame = np.zeros((1191, 2064, 3), dtype=np.uint8)
        self.session.input_revision = 2
        self.session._last_observation = (self.frame, 2, time.monotonic(), 'saved.png', 1376, 794)
        self.session.focus = Mock()
        self.session.log = Mock()
        self.session.observe = Mock(return_value='fresh')

    def test_recent_same_frame_keeps_exact_evidence_path(self):
        result = self.session.reuse_observation('hand-before-inspection', self.frame)
        self.assertIs(result, self.frame)
        self.session.observe.assert_not_called()
        self.assertEqual(self.session.log.call_args.kwargs['screenshot'], 'saved.png')
        self.assertTrue(self.session.log.call_args.kwargs['reused'])

    def test_different_frame_ui_input_and_age_force_new_capture(self):
        self.assertEqual(self.session.reuse_observation('next', self.frame.copy()), 'fresh')
        self.session.input_revision += 1
        self.assertEqual(self.session.reuse_observation('next', self.frame), 'fresh')
        self.session.input_revision = 2
        previous = list(self.session._last_observation)
        previous[2] -= 4
        self.session._last_observation = tuple(previous)
        self.assertEqual(self.session.reuse_observation('next', self.frame), 'fresh')

    def test_focus_change_invalidates_prior_capture(self):
        self.session.focus.side_effect = lambda: setattr(self.session, 'input_revision', 3)
        self.assertEqual(self.session.reuse_observation('next', self.frame), 'fresh')


class JokerTriggerTests(unittest.TestCase):
    def session(self, dirty=frozenset()):
        session = Mock()
        names = ('Ride the Bus', 'Square Joker', 'Red Card', 'Fortune Teller',
                 'Green Joker', 'Castle', 'Yorick', 'Ramen', 'Unknown hidden Joker')
        state = RunState(jokers=tuple(Joker(n) for n in names))
        session.joker_cache = JokerCache(round_token(state), 5, state.jokers, (), dirty)
        return session

    def test_discard_preserves_play_growth_and_refreshes_discard_and_unknown_effects(self):
        session = self.session()
        dirty_hand_action(session, 'discard')
        self.assertEqual(session.joker_cache.dirty,
                         frozenset({'green_joker', 'castle', 'yorick', 'ramen', 'unknown_hidden_joker'}))

    def test_play_refreshes_bus_square_and_preserves_already_dirty_transaction(self):
        session = self.session(frozenset({'red_card', 'fortune_teller'}))
        dirty_hand_action(session, 'play')
        self.assertTrue({'ride_the_bus', 'square_joker', 'red_card', 'fortune_teller'} <= session.joker_cache.dirty)

    def test_unknown_old_cache_is_not_silently_made_clean(self):
        session = self.session(None)
        dirty_hand_action(session, 'discard')
        self.assertTrue({'ride_the_bus', 'square_joker', 'red_card'} <= session.joker_cache.dirty)

    def test_round_end_still_refreshes_static_counter_and_lifetime(self):
        session = self.session()
        dirty_jokers(session, round_end=True)
        self.assertTrue({'red_card', 'fortune_teller', 'ride_the_bus'} <= session.joker_cache.dirty)


if __name__ == '__main__':
    unittest.main()
