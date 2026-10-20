"""Real Royal Flush previews and previously skipped shop tooltips."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from controller import verify_play_preview
from item_identity import identify_description
from perception import (parse_hand_preview, read_hand_preview, read_hand_preview_level,
                        read_text, shop_offer_item, shop_tooltip_region, tooltip_rows)
from recovery import ObservationError
from strategy import HAND_RANK, hand_name
from tests.test_runtime_recovery import FakeSession
from tools.replay_observation_speed import load_frame

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / 'tests/fixtures/recognition_review_20261017.json').read_text('utf-8'))


def frame(case):
    path = ROOT / case['path']
    if hashlib.sha256(path.read_bytes()).hexdigest() != case['sha256']:
        raise AssertionError('Retained pixels changed: ' + str(path))
    return load_frame(path)


class RecognitionReviewTests(unittest.TestCase):
    def test_real_royal_flush_frames_verify_without_clicks_or_recovery(self):
        self.assertEqual(hand_name('Royal Flush'), 'straight_flush')
        for case in FIXTURE['frames']:
            if case['role'] != 'fatal':
                continue
            with self.subTest(path=case['path']), tempfile.TemporaryDirectory() as directory:
                image = frame(case)
                session = FakeSession(directory)
                session.frame = image
                verify_play_preview(session, image, 'straight_flush')
                self.assertEqual(session.events[-1]['event'], 'play_preview_verified')
                self.assertEqual(session.taps, [])
                self.assertIsNone(session.pending_action)

    def test_existing_categories_and_transient_suffix_frames_keep_their_identity(self):
        for case in FIXTURE['frames']:
            if case['kind'] == 'preview' and case['role'] != 'fatal':
                with self.subTest(path=case['path']):
                    self.assertEqual(read_hand_preview(frame(case)), case['expected'])

    def test_unreadable_level_does_not_become_a_guessed_upgrade(self):
        for case in FIXTURE['frames']:
            if case['role'] == 'transient':
                with self.subTest(path=case['path']), self.assertRaises(ObservationError):
                    read_hand_preview_level(frame(case))

    def test_royal_flush_has_a_real_level_but_cannot_verify_a_different_hand(self):
        case = next(c for c in FIXTURE['frames'] if c['role'] == 'fatal')
        image = frame(case)
        self.assertEqual(read_hand_preview_level(image), ('straight_flush', 1))
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            session.frame = image
            with patch('controller.time.sleep'), self.assertRaises(ObservationError):
                verify_play_preview(session, image, 'flush')
            self.assertEqual(session.taps, [])
            self.assertIsNone(session.pending_action)

    def test_all_seven_degraded_shop_frames_resolve_to_the_actual_item(self):
        for case in FIXTURE['frames']:
            if case['kind'] != 'shop':
                continue
            with self.subTest(path=case['path']):
                image = frame(case)
                region = shop_tooltip_region(image, case['card_x'])
                labels = tooltip_rows(read_text(image, region))
                item, confidence = shop_offer_item(labels, case['price'])
                self.assertEqual((item.kind, item.name), ('joker', case['expected']))
                self.assertGreaterEqual(confidence, .88)
                self.assertIn(item.joker.rarity.title(), [s.title() for s in labels])
                if item.name == 'To Do List':
                    self.assertEqual(dict(item.joker.data)['hand'], HAND_RANK['four_kind'])

    def test_hack_requires_all_four_ranks_and_matching_rarity(self):
        self.assertEqual(identify_description(
            ['Not Discovered', 'Retrigger', 'each played', '2.3.4.0r 5', 'Uncommon'])[0], 'hack')
        for text in ('2, 3, 4, or 6', '2, 3, or 5', '2, 3, 4', '12, 3, 4, or 5'):
            with self.subTest(text=text), self.assertRaises(RuntimeError):
                identify_description(['Not Discovered', 'Retrigger each played', text, 'Uncommon'])
        with self.assertRaisesRegex(RuntimeError, 'rarity disagree'):
            identify_description(['Retrigger each played 2, 3, 4, or 5', 'Common'])

    def test_spare_trousers_requires_the_complete_growth_effect(self):
        text = 'This Joker gains +2 Mult if played hand contains a Two Pair'
        self.assertEqual(identify_description(['Not Discovered', text, 'Uncommon'])[0], 'spare_trousers')
        with self.assertRaisesRegex(RuntimeError, 'rarity disagree'):
            identify_description([text, 'Common'])
        self.assertNotEqual(identify_description(
            ['Not Discovered', '+10 Mult if played hand contains a Two Pair', 'Common'])[0], 'spare_trousers')

    def test_multiple_complete_effects_still_conflict(self):
        with self.assertRaisesRegex(RuntimeError, 'Multiple distinct'):
            identify_description(['This Joker gains +2 Mult if played hand contains a Two Pair',
                                  'Retrigger each played 2, 3, 4, or 5', 'Uncommon'])

    def test_unreadable_suffix_cannot_hide_a_second_hand_category(self):
        for titles in (('Pair', 'Flush'), ('Royal',), ('Royal', 'Pair')):
            words = [(word, .99, (200 + i * 80, 600)) for i, word in enumerate(titles)]
            words.append(('141.4', .99, (490, 600)))
            with self.subTest(titles=titles), self.assertRaises(ObservationError):
                parse_hand_preview(words)
        with self.assertRaises(ObservationError):
            parse_hand_preview([('Royal Flush', .99, (300, 590)),
                                ('Pair', .99, (300, 630))])


if __name__ == '__main__':
    unittest.main()
