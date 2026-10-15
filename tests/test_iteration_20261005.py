"""Regressions from the supervised fresh batch, using saved failure frames."""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from item_identity import identify_description, tokens
from perception import read_text, shop_tooltip_region, tooltip_rows, shop_offer_item, joker_counter
from controller import collect_jokers, selection_flags, wait_consumable_hand
from vision import read_hand
from recovery import ObservationError
from strategy import RunState, Card
from tests.test_runtime_recovery import FakeSession
from runner import verify_selected_purchase, verify_shop_selection
from strategy import ShopItem


class SelectedBaseballCardTests(unittest.TestCase):
    def test_numeric_badge_ocr_keeps_the_hidden_baseball_identity(self):
        labels = ['Not Discovered', 'ex', 'Ʒ Uncommon Jokers',
                  'each give K1.5 Mult', 'Rare']
        self.assertEqual(identify_description(labels)[0], 'baseball_card')
        self.assertEqual(tokens('each give K1.5 Mult'), ['each', 'give', 'x', 'mult'])
        self.assertEqual(tokens('King Mult'), ['king', 'mult'])
        with self.assertRaises(RuntimeError):
            identify_description(['Not Discovered', 'Rare Jokers', 'each give K1.5 Mult'])

    def test_real_selected_frame_verifies_baseball_and_rejects_a_different_plan(self):
        path = Path(__file__).parent / 'fixtures/iteration_20261005/baseball-selected.png'
        image = cv2.resize(cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1),
                           (2064, 1191), interpolation=cv2.INTER_NEAREST)
        labels = tooltip_rows(read_text(image, shop_tooltip_region(image, 1162)))
        item, confidence = shop_offer_item(labels, 8)
        self.assertEqual(item.name, 'Baseball Card')
        self.assertGreaterEqual(confidence, .88)
        verify_selected_purchase(image, item, 1162)
        with self.assertRaisesRegex(RuntimeError, 'differs'):
            verify_selected_purchase(image, ShopItem('joker', 'Clever Joker', 8), 1162)


class GlowingCounterTests(unittest.TestCase):
    def test_real_water_glow_counter_has_five_visible_jokers(self):
        path = Path(__file__).parent / 'fixtures/iteration_20261005/water-glow-counter.png'
        image = cv2.resize(cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1),
                           (2064, 1191), interpolation=cv2.INTER_NEAREST)
        self.assertEqual(joker_counter(image), (5, 5))

    def test_transient_counter_failure_reobserves_without_clicking(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            with patch('controller.joker_counter', side_effect=[ObservationError('glow'), (0, 5)]), \
                    patch('controller.time.sleep'):
                actual = collect_jokers(session, session.frame, RunState())
            self.assertEqual(actual.jokers, ())
            self.assertEqual(session.taps, [])
            self.assertEqual(sum(e['event'] == 'joker_counter_read_retry' for e in session.events), 1)

    def test_persistently_unreadable_counter_stops_after_bounded_retries(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            with patch('controller.joker_counter', side_effect=ObservationError('unreadable')) as counter, \
                    patch('controller.time.sleep'):
                with self.assertRaises(ObservationError):
                    collect_jokers(session, session.frame, RunState())
            self.assertEqual(counter.call_count, 3)
            self.assertEqual(session.taps, [])


class StoneHandTests(unittest.TestCase):
    def test_real_tower_transformation_keeps_all_eight_card_locations(self):
        path = Path(__file__).parent / 'fixtures/iteration_20261005/tower-stone-hand.png'
        image = cv2.resize(cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1),
                           (2064, 1191), interpolation=cv2.INTER_NEAREST)
        found = read_hand(image, 8)
        expected = [(14, 'H'), (14, 'D'), (10, 'C'), (0, '?'),
                    (8, 'S'), (5, 'H'), (5, 'C'), (3, 'C')]
        self.assertEqual([c.identity for c, _, _ in found], expected)
        self.assertTrue(found[3][0].rankless)
        self.assertEqual(found[3][0].enhancement, 'stone')
        self.assertFalse(any(c.face_down for c, _, _ in found))
        self.assertGreater(found[3][1], .60)
        verified = read_hand(image, 8, identities=[c for c, _, _ in found])
        self.assertEqual(selection_flags(verified), set())
        self.assertEqual([c.identity for c, _, _ in verified], expected)

    def test_real_target_before_tower_is_not_mistaken_for_stone(self):
        path = Path(__file__).parent / 'fixtures/iteration_20261005/tower-target-before-use.png'
        image = cv2.resize(cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1),
                           (2064, 1191), interpolation=cv2.INTER_NEAREST)
        identities = [Card(r, s) for r, s in [(14, 'H'), (14, 'D'), (10, 'C'), (9, 'H'),
                                              (8, 'S'), (5, 'H'), (5, 'C'), (3, 'C')]]
        found = read_hand(image, 8, identities=identities)
        self.assertEqual(selection_flags(found), {3})
        self.assertFalse(any(c.rankless or c.face_down for c, _, _ in found))

    def test_visible_counter_does_not_end_the_wait_before_cards_are_located(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            with patch('controller.hand_counter', return_value=(8, 8)), \
                    patch('controller.read_hand', side_effect=[ObservationError('moving card'), []]) as locate, \
                    patch('controller.time.sleep'):
                _, count = wait_consumable_hand(session, session.frame)
            self.assertEqual(count, 8)
            self.assertEqual(locate.call_count, 2)
            self.assertEqual(session.taps, [])

    def test_persistent_incomplete_hand_never_authorizes_selection_clicks(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            with patch('controller.hand_counter', return_value=(8, 8)), \
                    patch('controller.read_hand', side_effect=ObservationError('missing card')) as locate, \
                    patch('controller.time.sleep'):
                with self.assertRaises(ObservationError):
                    wait_consumable_hand(session, session.frame)
            self.assertEqual(locate.call_count, 5)
            self.assertEqual(session.taps, [])


class NarrowSelectedPlanetTests(unittest.TestCase):
    def test_real_uranus_panel_and_its_buy_button_are_verified(self):
        path = Path(__file__).parent / 'fixtures/iteration_20261005/uranus-selected.png'
        image = cv2.resize(cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1),
                           (2064, 1191), interpolation=cv2.INTER_NEAREST)
        chosen = ShopItem('planet', 'Uranus', 3)
        verify_selected_purchase(image, chosen, 1162)
        with self.assertRaisesRegex(ObservationError, 'differs'):
            verify_selected_purchase(image, ShopItem('planet', 'Mars', 3), 1162)
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            buy, _ = verify_shop_selection(session, image, chosen, 1162, 700)
            self.assertLess(abs(buy[0] - 1162), 20)
            self.assertGreater(buy[1], 650)
            self.assertEqual(session.taps, [])

    def test_transient_selected_tooltip_failure_refreshes_without_buying(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            with patch('runner.read_text', return_value=[('Buy', .99, (1162, 700))]), \
                    patch('runner.verify_selected_purchase', side_effect=[ObservationError('tooltip moved'), None]):
                buy, _ = verify_shop_selection(session, session.frame, ShopItem('planet', 'Uranus', 3), 1162, 700)
            self.assertEqual(buy, (1162, 700))
            self.assertEqual(session.taps, [])
            self.assertEqual(sum(e['event'] == 'shop_selection_read_retry' for e in session.events), 1)

    def test_wrong_selected_identity_never_authorizes_purchase(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            with patch('runner.read_text', return_value=[('Buy', .99, (1162, 700))]), \
                    patch('runner.verify_selected_purchase', side_effect=ObservationError('different item')) as verify:
                with self.assertRaises(ObservationError):
                    verify_shop_selection(session, session.frame, ShopItem('planet', 'Uranus', 3), 1162, 700)
            self.assertEqual(verify.call_count, 3)
            self.assertEqual(session.taps, [])


if __name__ == '__main__':
    unittest.main()
