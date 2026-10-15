"""Regressions from the six completed games and interrupted seventh game."""
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from controller import apply_play_order, collect_cards
from perception import read_text, shop_tooltip_region, tooltip_rows, shop_offer_item, parse_card_tooltip, joker_counter
from runner import verify_selected_purchase, play_one_hand
from strategy import Card, Joker, RunState, ShopItem, decide_play, decide_discard, decide_shop, score_hand
from tests.test_runtime_recovery import FakeSession
from vision import read_hand

FIXTURES = Path(__file__).parent / 'fixtures/iteration_20261004'


def screenshot(name):
    return cv2.resize(cv2.imdecode(np.fromfile(FIXTURES / name, dtype=np.uint8), 1),
                      (2064, 1191), interpolation=cv2.INTER_NEAREST)


def located(cards):
    return [(card, .9, (550 + i * 140, 720)) for i, card in enumerate(cards)]


class OctoberIterationTests(unittest.TestCase):
    def test_actual_three_of_five_counter_keeps_the_slash(self):
        self.assertEqual(joker_counter(screenshot('2973-decision-state.png')), (3, 5))
        for false_line in ('375', '8/5'):
            with patch('perception.read_text', side_effect=[[('375', .99, (609, 354))],
                                                            [(false_line, .99, (615, 355))],
                                                            [(false_line, .99, (615, 355))]]):
                with self.assertRaises(RuntimeError):
                    joker_counter(np.zeros((1191, 2064, 3), dtype=np.uint8))

    def test_diamond_four_cannot_become_nine_when_base_chips_disagree(self):
        broken = ['9 of Diamonds', '+4 chips']
        with self.assertRaisesRegex(RuntimeError, 'contradicts'):
            parse_card_tooltip(Card(9, 'D'), broken, require_identity=True)
        image = screenshot('2158-card-5-tooltip.png')
        found = [(Card(9, 'D'), .8, (1241, 715))]
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            session.frame = image
            words = [(line, .99, (1310, 570 + 30 * i)) for i, line in enumerate(broken)]
            with patch('controller.read_hand', return_value=found), patch('controller.read_text', return_value=words):
                verified = collect_cards(session, image, 1)
            self.assertEqual(verified[0][0].identity, (4, 'D'))
            self.assertTrue(any(e['event'] == 'card_title_recovered' for e in session.events))

    def test_three_selected_shop_panels_retain_title_and_rarity(self):
        for file, x, name in [('2462-shop-selected.png', 1259, 'Blue Joker'),
                              ('2846-shop-selected.png', 1160, 'Mad Joker'),
                              ('3836-shop-selected.png', 1160, 'Scary Face')]:
            image = screenshot(file)
            labels = tooltip_rows(read_text(image, shop_tooltip_region(image, x)))
            item, _ = shop_offer_item(labels, 4)
            self.assertEqual(item.name, name)
            verify_selected_purchase(image, item, x)
            with self.assertRaises(RuntimeError):
                verify_selected_purchase(image, ShopItem('joker', 'Joker', 4), x)

    def test_mark_hand_detects_four_backs_before_false_rank_matches(self):
        found = read_hand(screenshot('4097-hand-before-inspection.png'), 8, allow_uncertain=True)
        self.assertEqual([i for i, (card, _, _) in enumerate(found) if card.face_down], [1, 2, 3, 4])
        # Corner guesses are provisional; the weak red 3 needs tooltip proof.
        titles = ['Ace of Spades', '10 of Clubs', '3 of Hearts', '2 of Hearts']
        words = [[(title, .99, (1000, 500))] for title in titles]
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            session.frame = screenshot('4097-hand-before-inspection.png')
            with patch('controller.read_text', side_effect=words) as ocr:
                enriched = collect_cards(session, session.frame, 8)
            self.assertEqual(ocr.call_count, 4)
            self.assertEqual([r[0].identity for r in enriched if not r[0].face_down],
                             [(14, 'S'), (10, 'C'), (3, 'H'), (2, 'H')])

    def test_corner_misread_during_drag_is_verified_by_tooltips(self):
        cards = [Card(14, 'H'), Card(13, 'S'), Card(12, 'S')]
        original = located(cards)
        actual = located([cards[1], cards[0], cards[2]])
        wrong_corner = [*actual[:2], (Card(9, 'S'), .6, actual[2][2])]
        plan = replace(decide_play(cards), indices=(0, 1), play_order=(1, 0))
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            session.drag = lambda *args: None
            with patch('controller.read_hand', return_value=wrong_corner), patch('controller.collect_cards', return_value=actual):
                result, indices = apply_play_order(session, original, plan)
            self.assertEqual([r[0].identity for r in result], [r[0].identity for r in actual])
            self.assertEqual(indices, [0, 1])
            self.assertFalse(session.play_order_fallback)
            self.assertTrue(any(e['event'] == 'card_order_verified_by_tooltips' for e in session.events))

    def test_failed_drag_replans_in_actual_order_without_repeated_dragging(self):
        cards = [Card(14, 'H'), Card(13, 'S'), Card(12, 'S')]
        found = located(cards)
        plan = replace(decide_play(cards), indices=(0, 1), play_order=(1, 0))
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            drags = []
            session.drag = lambda *args: drags.append(args)
            with patch('controller.read_hand', return_value=found), patch('controller.collect_cards', return_value=found):
                actual, indices = apply_play_order(session, found, plan)
            self.assertEqual(actual, found)
            self.assertEqual(len(drags), 1)
            self.assertTrue(session.play_order_fallback)

    def test_changed_roster_after_drag_does_not_allow_unverified_play(self):
        cards = [Card(14, 'H'), Card(13, 'S'), Card(12, 'S')]
        found = located(cards)
        wrong = located([Card(2, 'D'), cards[1], cards[2]])
        plan = replace(decide_play(cards), indices=(0, 1), play_order=(1, 0))
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            session.drag = lambda *args: None
            with patch('controller.read_hand', return_value=wrong), patch('controller.collect_cards', return_value=wrong):
                with self.assertRaisesRegex(RuntimeError, 'roster changed'):
                    apply_play_order(session, found, plan)
            self.assertEqual(session.taps, [])

    def test_runner_logs_actual_order_score_after_drag_fallback(self):
        cards = [Card(14, 'S'), Card(13, 'S'), Card(12, 'S'), Card(11, 'S'), Card(10, 'S')]
        found = located(cards)
        state = RunState(jokers=(Joker('Photograph'), Joker('Hanging Chad')), discards_left=0)
        expected = decide_play(cards, state, optimize_order=False)
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            session.policy_state = state
            def failed_order(*args):
                session.play_order_fallback = True
                return found, []
            with patch('runner.collect_position', return_value=(state, found)), patch('runner.try_consumable', return_value=False), patch('runner.apply_play_order', side_effect=failed_order), patch('runner.select_verified'), patch('runner.verify_play_preview') as preview, patch('runner.confirm_pending'), patch('runner.time.sleep'):
                play_one_hand(session)
            event = next(e for e in session.events if e['event'] == 'hand_decision_after_order_fallback')
            self.assertEqual(event['expected_score'], expected.estimate.score)
            self.assertEqual(session.pending_action['details']['estimated_score'], expected.estimate.score)
            preview.assert_called_once()

    def test_lucky_expected_clear_cannot_override_guaranteed_clear(self):
        cards = [Card(2, 'H', enhancement='lucky'), Card(13, 'S')]
        state = RunState(target_score=15, hands_left=1)
        chosen = decide_play(cards, state)
        actual = score_hand([cards[i] for i in chosen.play_order], replace(state, score_mode='conservative'),
                            [c for i, c in enumerate(cards) if i not in chosen.indices])
        self.assertGreaterEqual(actual.score, 15)

    def test_lucky_expectation_does_not_skip_available_discard_as_certain_win(self):
        cards = [Card(2, 'H', enhancement='lucky'), Card(13, 'S')]
        state = RunState(target_score=30, hands_left=1, discards_left=1)
        self.assertGreaterEqual(decide_play(cards, state).estimate.score, 30)
        self.assertNotEqual(decide_discard(cards, state).reason, 'play to clear blind now')

    def test_early_weak_shop_can_reroll_but_reserves_purchase_cash(self):
        state = RunState(money=17, ante=2, target_score=800)
        self.assertEqual(decide_shop([], state, reroll_cost=5).action, 'reroll')
        self.assertEqual(decide_shop([], replace(state, money=8), reroll_cost=5).action, 'leave')
        self.assertEqual(decide_shop([], replace(state, shop_rerolls=2), reroll_cost=5).action, 'leave')


if __name__ == '__main__':
    unittest.main()
