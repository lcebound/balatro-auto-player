"""Recorded offer failures and shop risk cases; no live device or game."""
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from item_identity import identify_description
from perception import shop_offer_item, supported_shop_item, read_reroll_button, read_jokers
from planning import shop_forecast, forecast_round, growth_opportunity, _FORECAST_CACHE
from recovery import ObservationError, state_from_dict
from shop_extras import handle_pack, PackView, PackChoice
from strategy import Card, Joker, RunState, ShopItem, decide_shop
from tests.test_runtime_recovery import FakeSession
from tests.test_speed_and_policy_20261005 import forecast

FIXTURES = Path(__file__).resolve().parent / 'fixtures'


def frame(name):
    return cv2.imdecode(np.frombuffer((FIXTURES / name).read_bytes(), np.uint8), cv2.IMREAD_COLOR)


class RecordedOfferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recorded = json.loads((FIXTURES / 'offer_and_policy_20261010.json').read_text('utf-8'))

    def test_all_nineteen_logged_offer_failures_have_verified_identities(self):
        art = read_jokers(frame('misprint_shop_20261010.png'), (960, 450, 1930, 680))
        misprint = max((a for a in art if a[1] == 'misprint'), key=lambda a: a[0])
        for case in self.recorded['offers']:
            with self.subTest(time=case['time']):
                hint = misprint if case['expected_name'] == 'Misprint' else (0, '', None)
                item, _ = shop_offer_item(case['labels'], 5, hint[1],
                    art_confidence=hint[0], require_supported=False)
                self.assertEqual(item.name, case['expected_name'])
                self.assertEqual(supported_shop_item(item), case['expected_supported'])

    def test_club_mult_numbers_and_rarity_distinguish_the_two_jokers(self):
        self.assertEqual(identify_description(['Played cards with', 'Club suit give',
            '+7 Mult when scored', 'Uncommon'])[0], 'onyx_agate')
        self.assertEqual(identify_description(['Played cards with', 'Club syit give',
            '+3 Mult when scored', 'Common'])[0], 'gluttonous_joker')
        with self.assertRaises(RuntimeError):
            identify_description(['Played cards with Club suit give +3 Mult when scored',
                                  'Played cards with Club suit give +7 Mult when scored'])
        with self.assertRaises(RuntimeError):
            identify_description(['Played cards with Club suit give +7 Mult when scored', 'Common'])

    def test_short_misprint_effect_needs_matching_artwork(self):
        labels = ['Not t Discovered', '+15 Mult', 'Common']
        for hint, confidence in [('gros_michel', .99), ('misprint', .7), ('', 0)]:
            with self.assertRaises(ObservationError):
                shop_offer_item(labels, 4, hint, art_confidence=confidence)

    def test_unsupported_identity_does_not_authorize_its_use(self):
        case = next(c for c in self.recorded['offers'] if c['expected_name'] == 'The High Priestess')
        with self.assertRaisesRegex(ObservationError, 'unsupported use workflow'):
            shop_offer_item(case['labels'], 3)
        item, _ = shop_offer_item(case['labels'], 3, require_supported=False)
        self.assertEqual(item.name, 'The High Priestess')
        self.assertFalse(supported_shop_item(item))

    def test_actual_needle_reroll_price_is_recovered_from_saved_pixels(self):
        button, price = read_reroll_button(frame('needle_reroll_20261010.png'))
        self.assertEqual(price, 6)
        self.assertEqual(button, (759, 633))

    def test_conflicting_price_proofs_and_multiple_buttons_are_rejected(self):
        image = frame('needle_reroll_20261010.png')
        with patch('perception.read_text', side_effect=[
            [('Reroll', .99, (759, 633)), ('$6', .66, (759, 683))],
            [('$6', .82, (759, 683))], [('$8', .96, (60, 32))]]):
            self.assertIsNone(read_reroll_button(image))
        with patch('perception.read_text', return_value=[
            ('Reroll', .99, (759, 633)), ('Reroll', .99, (750, 700)), ('$6', .99, (759, 683))]):
            self.assertIsNone(read_reroll_button(image))

    def test_recorded_needle_shop_uses_available_reroll_instead_of_leaving(self):
        state = state_from_dict(self.recorded['needle_shop'])
        self.assertEqual(state.money, 52)
        self.assertEqual(state.upcoming_boss, 'The Needle')
        self.assertEqual(decide_shop([], state, reroll_cost=6).action, 'reroll')
        self.assertEqual(decide_shop([], replace(state, money=11), reroll_cost=6).action, 'leave')

    def test_conditional_engine_forecast_prices_the_low_scoring_tail(self):
        result = shop_forecast(state_from_dict(self.recorded['conditional_shop']))
        scores = sorted(result.next_round.scores)
        self.assertLessEqual(result.next_round.safe_score, .75 * scores[0] + .25 * scores[len(scores)//4])

    def test_recorded_second_game_can_spend_interest_reserve_to_repair_a_deficit(self):
        state = state_from_dict(self.recorded['conditional_shop'])
        self.assertEqual(state.money, 18)
        self.assertLess(shop_forecast(state).coverage, 1)
        self.assertEqual(decide_shop([], state, reroll_cost=5).action, 'reroll')


class PolicyRecoveryTests(unittest.TestCase):
    def test_urgent_shop_chooses_scoring_progress_over_unactivated_income(self):
        state = RunState(money=40, ante=5)
        items = [ShopItem('joker', 'Golden Joker', 5, Joker('Golden Joker')),
                 ShopItem('joker', 'Joker', 2, Joker('Joker'))]
        before = forecast(safe=4000, target=10000)
        after = forecast(safe=8000, target=10000)
        def projected(s):
            return after if s.has('joker') else before
        with patch('planning.shop_forecast', side_effect=projected), patch('strategy._item_value', return_value=100):
            decision = decide_shop(items, state)
        self.assertEqual(decision.action, 'buy')
        self.assertEqual(decision.item_index, 1)

    def test_growth_reward_depends_on_activation_and_remaining_antes(self):
        result = forecast()
        state = RunState(ante=3, money=35)
        self.assertGreater(growth_opportunity(Joker('Green Joker'), state, result), 0)
        self.assertEqual(growth_opportunity(Joker('Runner'), state, result), 0)
        self.assertEqual(growth_opportunity(Joker('Green Joker'), replace(state, ante=8), result), 0)

    def test_observation_only_changes_reuse_the_same_forecast(self):
        _FORECAST_CACHE.clear()
        state = RunState(ante=2, money=25)
        sample = forecast().next_round
        with patch('planning.forecast_round', return_value=sample) as compute:
            first = shop_forecast(state)
            second = shop_forecast(replace(state, round_score=999, seen_cards=(Card(14, 'S'),),
                                           round_played_types={'pair': 4}, boss_first_hand_done=True))
        self.assertIs(first, second)
        self.assertEqual(compute.call_count, 1)

    def test_unknown_composition_still_respects_the_observed_deck_count(self):
        # 46 was the actual failed game's deck count. The forecast must not
        # award Blue Joker chips for six cards which are no longer in the deck.
        captured = []
        from strategy import score_hand
        def plan(hand, state):
            captured.append(state.draw_pile_size)
            return (0,), score_hand([hand[0]], state, hand[1:])
        state = RunState(deck_size=46, hand_size=8, jokers=(Joker('Blue Joker'),))
        with patch('planning._draw_best_plan', side_effect=plan):
            forecast_round(state, target=0, samples=1)
        self.assertEqual(captured[-4:], [38, 37, 36, 35])

    def test_missing_use_button_abandons_reward_before_any_use_then_skips(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            view = PackView('celestial_pack', 1, (1500, 1000), 1000)
            choices = [PackChoice(ShopItem('planet', 'Mercury', 0), (1100, 850))]
            with patch('shop_extras.read_pack_view', return_value=view), \
                 patch('shop_extras.collect_pack_choices', return_value=choices), \
                 patch('shop_extras.verify_offer_selection', side_effect=ObservationError('Use button is missing or ambiguous')), \
                 patch('vision.classify_scene', return_value='shop'), \
                 patch('perception.read_numbers', return_value={}):
                handle_pack(session, session.frame)
            self.assertIsNone(session.pending_action)
            self.assertEqual([e['kind'] for e in session.events if e['event'] == 'action_prepared'], ['skip_pack'])
            self.assertEqual(session.policy_state.levels, {})
            self.assertEqual(len(session.taps), 2)  # select, then independently verified Skip
            self.assertTrue(any(e['event'] == 'pack_reward_unavailable' for e in session.events))


if __name__ == '__main__':
    unittest.main()
