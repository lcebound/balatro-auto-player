"""Retained shop budgets, hidden growth and same-frame numeric proofs."""
import json
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from controller import collect_jokers
from joker_cache import round_token
from perception import (parse_joker_tooltip, read_ramen_multiplier, read_text,
                        shop_offer_item, tooltip_rows, verify_current_multiplier)
from recovery import ObservationError, state_from_dict
from runner import verify_selected_purchase
from strategy import Joker, RunState, ShopItem, decide_shop
from tests.test_speed_and_policy_20261005 import forecast
from tools.replay_observation_speed import load_frame
from tools.review_growth_20261015 import items_from_event

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / 'tests/fixtures/growth_review_20261019.json').read_text('utf-8'))


def ramen_words(image):
    words = read_text(image, (525, 330, 2020, 850))
    x, y = next(w[2] for w in words if w[0].strip().casefold() == 'ramen')
    return [w for w in words if abs(w[2][0] - x) < 145 and y - 22 <= w[2][1] <= y + 210]


class ShopBudgetTests(unittest.TestCase):
    def historical(self, name):
        event = next(c['event'] for c in DATA['shops'] if c['name'] == name)
        return decide_shop(items_from_event(event), state_from_dict(event['state']),
                           reroll_cost=event.get('reroll_cost'))

    def test_actual_empty_starter_prefers_buffoon_to_last_cash_wasteful(self):
        result = self.historical('starter_voucher')
        self.assertEqual((result.action, result.item_index), ('buy', 2))

    def test_actual_lethal_shop_preserves_rescue_cash_instead_of_seed_money(self):
        self.assertEqual(self.historical('dangerous_seed_money').action, 'reroll')

    def test_actual_midgame_growth_and_multiplier_upgrades_are_preserved(self):
        for name in ('green_pickup', 'card_sharp_upgrade'):
            with self.subTest(name=name):
                result = self.historical(name)
                self.assertEqual((result.action, result.item_index, result.sell_index),
                                 ('sell_then_buy', 1, 0))

    def test_a_real_immediate_resource_rescue_can_still_spend_last_cash(self):
        before, after = forecast(safe=500, target=600), forecast(safe=1000, target=600)
        with patch('planning.shop_forecast', side_effect=[before, after]):
            result = decide_shop([ShopItem('voucher', 'Grabber', 10)], RunState(ante=1, money=10))
        self.assertEqual(result.action, 'buy')

    def test_a_funded_nonlethal_investment_is_still_eligible(self):
        before = forecast(safe=5000, target=600)
        with patch('planning.shop_forecast', return_value=before):
            result = decide_shop([ShopItem('voucher', 'Seed Money', 10)],
                                 RunState(ante=2, money=49, jokers=(Joker('Joker'),)))
        self.assertEqual(result.action, 'buy')


class HiddenGrowthTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((1191, 2064, 3), np.uint8)
        self.session = SimpleNamespace(hidden_joker_roster=None, joker_cache=None, log=Mock())

    def observe(self, state):
        with patch('controller.joker_counter', return_value=(len(state.jokers), 5)), \
             patch('controller.read_jokers') as atlas, patch('controller.read_text') as ocr:
            result = collect_jokers(self.session, self.frame, state)
        atlas.assert_not_called()
        ocr.assert_not_called()
        self.assertTrue(result.joker_order_unknown)
        return result

    def test_retained_amber_counters_keep_confirmed_green_growth_without_slot_mapping(self):
        for case in DATA['hidden']:
            with self.subTest(index=case['index']):
                state = state_from_dict(case['state'])
                prior = state_from_dict(dict(case['state'], jokers=case['prior_roster'])).jokers
                self.session.hidden_joker_roster = (round_token(state), 5, prior)
                result = self.observe(state)
                green = next(j for j in result.jokers if j.key == 'green_joker')
                self.assertEqual(green.value, max(0, 28 + sum(state.round_played_types.values())
                                                 - state.round_discards_used))
                self.assertTrue(green.get('counter_lower_bound'))
                self.assertFalse(green.get('observed_value'))

    def test_action_bounds_use_frozen_baseline_and_do_not_accumulate_twice(self):
        known = lambda name, value: Joker(name, value, data=(('observed_value', 1),))
        state = RunState(boss='Amber Acorn', round_discards_used=0,
                         jokers=(known('Green Joker', 28), known('Ramen', 1.9), known('Ice Cream', 80)))
        first = self.observe(state)
        later = replace(first, round_played_types={'pair': 2}, round_discards_used=3)
        result = self.observe(later)
        self.assertEqual([j.value for j in result.jokers], [27, 1.75, 70])
        self.assertEqual(self.observe(result).jokers, result.jokers)

    def test_resumed_hidden_bounds_are_not_applied_to_old_actions_again(self):
        state = RunState(boss='Amber Acorn', joker_order_unknown=True, round_discards_used=2,
                         round_played_types={'pair': 3}, jokers=(
                             Joker('Green Joker', 29, data=(('counter_lower_bound', 1),)),))
        result = self.observe(state)
        self.assertEqual(result.jokers[0].value, 29)
        self.assertEqual(self.observe(replace(result, round_played_types={'pair': 4})).jokers[0].value, 30)

    def test_missing_proof_or_unknown_discards_cannot_create_a_fresh_counter(self):
        state = RunState(boss='Amber Acorn', jokers=(Joker('Green Joker', 28), Joker('Ice Cream'), Joker('Ramen')))
        result = self.observe(state)
        self.assertEqual([j.value for j in result.jokers], [0, 0, 1])
        self.assertTrue(all(j.get('counter_lower_bound') for j in result.jokers))


class ObeliskBoundsTests(unittest.TestCase):
    def verify(self, value, hands, previous=None):
        words = [(f'Currently X{value} Mult', .99, (900, 500))]
        with patch('perception.read_text', return_value=words) as read:
            result = verify_current_multiplier(np.zeros((1191, 2064, 3), np.uint8), words,
                                              Joker('Obelisk', value), RunState(total_hands_played=hands), previous)
        self.assertEqual(read.call_count, 4)
        return result

    def test_agreement_cannot_authorize_a_physically_impossible_decimal_loss(self):
        for value, hands in ((22, 21), (12, 48)):
            result = self.verify(value, hands)
            self.assertEqual(result.value, 1)
            self.assertTrue(result.get('counter_lower_bound'))
            self.assertFalse(result.get('observed_value'))

    def test_verified_previous_value_limits_single_hand_growth_but_allows_reset(self):
        old = Joker('Obelisk', 1.2, data=(('observed_value', 1), ('value_observed_at_hand', 47)))
        self.assertEqual(self.verify(2.2, 48, old).value, 1)
        for value in (1, 1.2, 1.4):
            self.assertTrue(self.verify(value, 48, old).get('observed_value'))

    def test_previous_lower_bound_cannot_be_used_as_an_upper_bound(self):
        old = Joker('Obelisk', 1, data=(('counter_lower_bound', 1), ('value_observed_at_hand', 47)))
        self.assertEqual(self.verify(2.2, 48, old).value, 2.2)
        self.assertEqual(self.verify(12, 70).value, 12)


class RamenProofTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = []
        for case in DATA['ramen']:
            image = load_frame(ROOT / case['proof']['path'])
            cls.cases.append((case, image, ramen_words(image)))

    def test_retained_shop_and_selection_frames_prove_live_multiplier(self):
        for case, image, words in self.cases:
            with self.subTest(index=case['index']):
                item, _ = shop_offer_item(tooltip_rows(words), 6, image=image, words=words)
                self.assertEqual((item.name, item.joker.value), ('Ramen', 2))
                self.assertTrue(item.joker.get('observed_value'))

    def test_selected_purchase_keeps_identity_gate_and_uses_same_frame(self):
        case, image, words = self.cases[4]
        with patch('runner.shop_tooltip_region', return_value=(525, 330, 2020, 850)), \
             patch('runner.read_text', return_value=words):
            verify_selected_purchase(image, ShopItem('joker', 'Ramen', 6), 1250)
            with self.assertRaises(ObservationError):
                verify_selected_purchase(image, ShopItem('joker', 'Joker', 2), 1250)

    def test_actual_selected_card_coordinates_keep_the_real_narrow_tooltip_path(self):
        # Coordinates are from the original select-purchase taps, not the
        # card's unselected slot or an artificially widened tooltip region.
        coordinates = {1517: 1158, 1683: 1259, 13200: 1259, 20304: 1161}
        for case, image, words in self.cases:
            if case['index'] in coordinates:
                with self.subTest(index=case['index']):
                    verify_selected_purchase(image, ShopItem('joker', 'Ramen', 6), coordinates[case['index']])

    def test_readable_normal_counter_uses_no_extra_ocr(self):
        with patch('perception.read_text') as read:
            result = parse_joker_tooltip(Joker('Ramen'), ['Ramen', 'X1.72 Mult', 'loses X0.01 Mult'],
                                        image=self.cases[0][1], words=self.cases[0][2])
        self.assertEqual(result.value, 1.72)
        read.assert_not_called()

    def test_no_image_or_wrong_title_cannot_fill_in_initial_x2(self):
        with self.assertRaises(ObservationError):
            parse_joker_tooltip(Joker('Ramen'), ['Ramen', '2 Mult', 'loses X0.01 Mult'])
        case, image, words = self.cases[0]
        self.assertIsNone(read_ramen_multiplier(image, [w for w in words if w[0] != 'Ramen']))
        with self.assertRaises(ObservationError):
            parse_joker_tooltip(Joker('Ramen'), ['Ramen', 'X20 Mult', 'loses X0.01 Mult'])

    def test_a_single_vote_low_confidence_or_conflict_cannot_authorize_factor(self):
        case, image, words = self.cases[0]
        for responses in (([('X2', .99, (1, 1))], [], []),
                          ([('X2', .79, (1, 1))],) * 4,
                          ([('X2', .99, (1, 1))], [('X1.9', .99, (1, 1))])):
            with self.subTest(responses=responses), patch('perception.read_text', side_effect=responses) as read:
                self.assertIsNone(read_ramen_multiplier(image, words))
                self.assertLessEqual(read.call_count, 4)


if __name__ == '__main__':
    unittest.main()
