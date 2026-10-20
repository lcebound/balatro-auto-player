"""Real missed investments, survival boundaries and suspicious price pixels."""
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from economy import safe_economy_investment
from perception import read_offer_price
from recovery import ObservationError, state_from_dict
from shop_extras import collect_extra_offers
from strategy import Joker, RunState, ShopItem, decide_shop, _purchase_state
from tests.test_runtime_recovery import FakeSession
from tests.test_speed_and_policy_20261005 import forecast
from tools.replay_observation_speed import load_frame
from tools.review_growth_20261015 import items_from_event

ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads((ROOT / 'tests/fixtures/growth_decisions_20261017.json').read_text('utf-8'))['cases']
PRICES = json.loads((ROOT / 'tests/fixtures/shop_price_20261017.json').read_text('utf-8'))['cases']


def historical(name):
    event = next(c['event'] for c in CASES if c['name'] == name)
    return event, state_from_dict(event['state']), items_from_event(event)


class HistoricalGrowthTests(unittest.TestCase):
    def decide(self, name):
        event, state, items = historical(name)
        return decide_shop(items, state, reroll_cost=event.get('reroll_cost'))

    def test_early_golden_precedes_a_minor_tarot_and_preserves_cash(self):
        event, state, items = historical('early_economy_before_tower')
        decision = self.decide('early_economy_before_tower')
        self.assertEqual((decision.action, decision.item_index, decision.sell_index), ('buy', 1, -1))
        self.assertGreaterEqual(_purchase_state(items[1], state).money, 5)
        self.assertEqual(self.decide('early_golden_stencil').action, 'buy')

    def test_midgame_green_replaces_a_static_support_not_the_multiplier(self):
        decision = self.decide('green_growth_skipped')
        self.assertEqual((decision.action, decision.item_index, decision.sell_index), ('sell_then_buy', 1, 4))
        _, state, items = historical('green_growth_skipped')
        after = _purchase_state(items[1], state, 4)
        self.assertTrue(after.has('flower_pot'))
        self.assertTrue(after.has('green_joker'))

    def test_late_unstarted_growth_does_not_take_the_rescue_budget(self):
        self.assertEqual(self.decide('late_burnt_available').action, 'reroll')
        self.assertEqual(self.decide('burnt_growth_skipped').action, 'leave')
        self.assertEqual(self.decide('late_red_card').action, 'leave')
        self.assertEqual(self.decide('conditional_obelisk').action, 'reroll')

    def test_castles_affordability_is_upstream_of_growth_evaluation(self):
        event, state, items = historical('castle_price_suspicious')
        self.assertEqual(items[1].price, 99)  # Keep the original logged error.
        self.assertEqual(decide_shop(items, state, reroll_cost=event['reroll_cost']).action, 'reroll')
        case = PRICES[0]
        price = read_offer_price(load_frame(ROOT / case['proof']['path']),
                                 case['original'], case['confidence'], case['center'])
        items[1] = replace(items[1], price=price)
        decision = decide_shop(items, state, reroll_cost=event['reroll_cost'])
        self.assertEqual((decision.action, decision.item_index, decision.sell_index), ('sell_then_buy', 1, 0))


class EconomyInvestmentBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.state = RunState(ante=2, money=14, jokers=(Joker('Joker Stencil'),))
        self.joker = Joker('Golden Joker')
        self.after = replace(self.state, money=8, jokers=self.state.jokers + (self.joker,))
        self.before = forecast(safe=3000, target=800)
        self.projected = forecast(safe=2250, target=800)

    def eligible(self, **changes):
        args = dict(state=self.state, after=self.after, before=self.before,
                    projected=self.projected, joker=self.joker, sell_index=-1)
        args.update(changes)
        return safe_economy_investment(**args)

    def test_funded_surplus_is_eligible_without_changing_any_forecast(self):
        self.assertTrue(self.eligible())
        self.assertEqual((self.projected.next_round.safe_score, self.after.money), (2250, 8))

    def test_late_investment_and_spending_the_last_interest_tier_are_rejected(self):
        self.assertFalse(self.eligible(state=replace(self.state, ante=4)))
        self.assertFalse(self.eligible(after=replace(self.after, money=4)))

    def test_immediate_sustainable_and_probability_risk_are_not_discounted(self):
        cases = [forecast(safe=1100, target=800), forecast(safe=2250, target=800, sustainable=1000),
                 forecast(safe=2000, target=800)]
        cases.append(replace(self.projected, next_round=replace(self.projected.next_round, clear_rate=.8)))
        for projected in cases:
            with self.subTest(projected=projected):
                self.assertFalse(self.eligible(projected=projected))
        self.assertFalse(self.eligible(before=forecast(safe=700, target=800)))

    def test_dangerous_boss_blocks_the_surplus_discount(self):
        before = forecast(safe=3000, target=800, boss_safe=60000)
        projected = forecast(safe=2250, target=800, boss_safe=42000)
        self.assertFalse(self.eligible(before=before, projected=projected))

    def test_income_assets_and_a_slow_payback_are_not_a_spendable_engine(self):
        for joker in (Joker('Egg'), Joker('Gift Card'), Joker('Satellite'),
                      replace(self.joker, debuffed=True)):
            with self.subTest(joker=joker.name):
                self.assertFalse(self.eligible(joker=joker,
                    after=replace(self.after, jokers=self.state.jokers + (joker,))))
        self.assertFalse(self.eligible(state=replace(self.state, money=25)))

    def test_existing_income_and_growth_cores_are_protected(self):
        state = replace(self.state, jokers=(Joker('Golden Joker'),))
        self.assertFalse(self.eligible(state=state))
        for name in ('Green Joker', 'Castle', 'Burnt Joker', 'Space Joker'):
            self.assertFalse(self.eligible(state=replace(self.state, jokers=(Joker(name),)), sell_index=0))


class PriceEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((1191, 2064, 3), np.uint8)
        self.center = (1349, 453)

    def labels(self, value, confidence=.98):
        return [(value, confidence, self.center)]

    def test_real_castle_scholar_and_buffoon_prices(self):
        for case in PRICES:
            with self.subTest(path=case['proof']['path']):
                path = ROOT / case['proof']['path']
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), case['proof']['sha256'])
                self.assertEqual(read_offer_price(load_frame(path), case['original'],
                    case['confidence'], case['center']), case['expected'])

    def test_normal_prices_have_zero_additional_ocr_calls(self):
        with patch('perception.read_text') as reader:
            for price in (0, 4, 9, 10, 20, 39):
                self.assertEqual(read_offer_price(self.image, f'${price}', .98, self.center), price)
            reader.assert_not_called()

    def test_legitimate_large_prices_are_verified_without_capping(self):
        for price in (40, 99, 105):
            with patch('perception.read_text', return_value=self.labels(f'${price}')) as reader:
                self.assertEqual(read_offer_price(self.image, f'${price}', .98, self.center), price)
                self.assertEqual(reader.call_count, 2)

    def test_only_two_agreeing_reliable_local_reads_allow_a_correction(self):
        with patch('perception.read_text', side_effect=[self.labels('$99', .81),
                self.labels('$9'), self.labels('$9')]) as reader:
            self.assertEqual(read_offer_price(self.image, '$99', .81, self.center), 9)
            self.assertEqual(reader.call_count, 3)
        with patch('perception.read_text', side_effect=[[], self.labels('$9'), []]) as reader:
            with self.assertRaises(ObservationError):
                read_offer_price(self.image, '$99', .81, self.center)
            self.assertEqual(reader.call_count, 3)

    def test_contradictions_do_not_become_invented_prices(self):
        for reads, confidence in (([self.labels('$9'), self.labels('$8')], .81),
                                  ([self.labels('$9')], .98),
                                  ([[('$9', .98, self.center), ('$8', .98, self.center)]], .81)):
            with self.subTest(reads=reads), patch('perception.read_text', side_effect=reads):
                with self.assertRaises(ObservationError):
                    read_offer_price(self.image, '$99', confidence, self.center)

    def test_unreadable_or_nearby_other_price_is_not_evidence(self):
        for reads in ([], [('$9', .8, self.center)], [('$9', .99, (1400, 453))]):
            with patch('perception.read_text', return_value=reads):
                with self.assertRaises(ObservationError):
                    read_offer_price(self.image, '$99', .81, self.center)

    def test_lower_offer_uses_recovered_price_instead_of_raw_ocr(self):
        frame = self.image
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            session.hover = Mock()
            session.observe = Mock(return_value=frame)
            session.log = Mock()
            item = ShopItem('buffoon_pack', 'Buffoon Pack', 4)
            with patch('perception.read_text', return_value=[('$54', .82, (1267, 787))]), \
                    patch('perception.read_offer_price', return_value=4), \
                    patch('shop_extras.offer_labels', return_value=['Buffoon Pack']), \
                    patch('shop_extras.identify_extra', return_value=item) as identify:
                items, positions = collect_extra_offers(session, frame)
            self.assertEqual(identify.call_args.args[1], 4)
            self.assertEqual(items, [item])
            self.assertEqual(positions, [(1267, 952)])

    def test_price_failure_is_optional_only_when_no_transaction_is_pending(self):
        for pending in (None, {'kind': 'buy'}):
            with self.subTest(pending=pending), tempfile.TemporaryDirectory() as directory:
                session = FakeSession(directory)
                session.pending_action = pending
                session.hover = Mock()
                with patch('perception.read_text', return_value=[('$54', .82, (1267, 787))]), \
                        patch('perception.read_offer_price', side_effect=ObservationError('conflict')), \
                        patch('shop_extras.identify_extra') as identify:
                    if pending:
                        with self.assertRaises(ObservationError):
                            collect_extra_offers(session, self.image)
                    else:
                        self.assertEqual(collect_extra_offers(session, self.image), ([], []))
                        self.assertEqual(session.runtime_issues[0]['level'], 'degraded')
                    identify.assert_not_called()


if __name__ == '__main__':
    unittest.main()
