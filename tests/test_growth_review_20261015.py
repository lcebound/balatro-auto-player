"""Run-backed growth transitions, early investment and bounded rescue search."""
import json
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from planning import advance_projected_play, engine_profile, forecast_round, growth_opportunity
from recovery import state_from_dict
from runner import read_shop_reroll
from strategy import Card, Joker, RunState, ShopItem, decide_shop, score_hand, classify
from tests.test_speed_and_policy_20261005 import forecast

ROOT = Path(__file__).resolve().parents[1]


class ObeliskTests(unittest.TestCase):
    def setUp(self):
        self.hand = [Card(2, 'S')]
        self.state = RunState(jokers=(Joker('Obelisk', 1.8),),
            played_types={'flush': 6, 'high_card': 4}, hands_left=3)

    def advance(self, state, hand=None):
        hand = hand or self.hand
        score = score_hand(hand, state)
        return advance_projected_play(state, hand, tuple(range(len(hand))), score)

    def test_growth_persists_then_resets_at_tied_most_played(self):
        first = self.advance(self.state)
        self.assertAlmostEqual(first.jokers[0].value, 2.)
        self.assertEqual(first.played('high_card'), 5)
        second = self.advance(first)
        self.assertAlmostEqual(second.jokers[0].value, 2.2)
        third = self.advance(second)
        self.assertEqual(third.jokers[0].value, 1.)
        self.assertEqual(self.state.jokers[0].value, 1.8)

    def test_reset_persists_into_next_hand_not_the_old_multiplier(self):
        flush = [Card(r, 'C') for r in (2, 3, 5, 8, 10)]
        after = self.advance(self.state, flush)
        self.assertEqual(after.jokers[0].value, 1.)
        score = score_hand(self.hand, after)
        self.assertAlmostEqual(score.multiplier, 1.2)

    def test_blueprint_copies_effect_but_does_not_double_physical_growth(self):
        state = replace(self.state, jokers=(Joker('Blueprint'), self.state.jokers[0]))
        after = self.advance(state)
        self.assertAlmostEqual(after.jokers[1].value, 2.)
        self.assertEqual(after.jokers[0].value, 0)
        self.assertEqual(after.effect_count('obelisk'), 2)

    def test_debuffed_obelisk_counter_does_not_change(self):
        state = replace(self.state, jokers=(replace(self.state.jokers[0], debuffed=True),))
        self.assertEqual(self.advance(state).jokers[0].value, 1.8)

    def test_empty_history_resets_and_is_not_supported_growth(self):
        state = replace(self.state, ante=3, played_types={})
        self.assertEqual(self.advance(state).jokers[0].value, 1.)
        self.assertIn('no_scaling', engine_profile(state).weaknesses)
        self.assertNotIn('no_scaling', engine_profile(replace(self.state, ante=3)).weaknesses)

    def test_forecast_measures_actual_obelisk_net_growth(self):
        state = replace(self.state, deck_cards=tuple(Card(r,'S') for r in (2,5,8,11,14)),
                        hand_size=5, discards_left=0, base_discards_per_round=0)
        result = forecast_round(state, target=1000, samples=2)
        self.assertIn('obelisk', dict(result.growth_per_round))


class RescueBudgetTests(unittest.TestCase):
    def setUp(self):
        self.state = RunState(ante=7, money=118, shop_rerolls=4,
                             jokers=(Joker('Joker',4),), objective='survival')
        self.weak = forecast(safe=43000, target=70000)

    def decision(self, state=None, projection=None, price=9, items=()):
        with patch('planning.shop_forecast',return_value=projection or self.weak):
            return decide_shop(items, state or self.state, reroll_cost=price)

    def test_cash_rich_lethal_run_can_search_after_four_rerolls(self):
        decision = self.decision()
        self.assertEqual(decision.action, 'reroll')
        self.assertIn('limit 8 rerolls', decision.reason)

    def test_eight_remains_a_hard_cap(self):
        self.assertEqual(self.decision(replace(self.state,money=200,shop_rerolls=8)).action,'leave')

    def test_existing_smaller_budget_cap_and_offer_cash_are_preserved(self):
        self.assertEqual(self.decision(replace(self.state,money=59)).action,'leave')
        self.assertEqual(self.decision(replace(self.state,money=14,shop_rerolls=0)).action,'leave')

    def test_safe_engine_and_unknown_or_expensive_price_do_not_search(self):
        self.assertEqual(self.decision(projection=forecast(safe=150000,target=70000)).action,'leave')
        self.assertEqual(self.decision(price=None).action,'leave')
        self.assertEqual(self.decision(price=21).action,'leave')

    def test_rich_bull_keeps_its_cash_reserve_and_price_ceiling(self):
        state=replace(self.state,jokers=(Joker('Bull'),))
        self.assertEqual(self.decision(state).action,'reroll')
        self.assertEqual(self.decision(replace(state,money=30,shop_rerolls=0)).action,'leave')

    def test_known_scoring_rescue_still_wins_over_unknown_search(self):
        item=ShopItem('joker','Cavendish',4,Joker('Cavendish'))
        with patch('planning.shop_forecast',side_effect=lambda s:
                   forecast(safe=120000,target=70000) if s.has('cavendish') else self.weak):
            d=decide_shop([item],self.state,reroll_cost=9)
        self.assertEqual(d.action,'buy')


class RerollObservationTests(unittest.TestCase):
    def test_confirmed_original_price_needs_no_new_frame(self):
        session=Mock(); initial,latest=object(),object()
        with patch('runner.read_reroll_button',return_value=((755,635),6)) as reader:
            self.assertEqual(read_shop_reroll(session,initial,latest)[1],6)
        reader.assert_called_once_with(initial)
        session.observe.assert_not_called()

    def test_existing_newer_frame_can_recover_price_without_new_capture(self):
        session=Mock(); initial,latest=object(),object()
        with patch('runner.read_reroll_button',side_effect=[None,((755,635),6)]) as reader:
            self.assertEqual(read_shop_reroll(session,initial,latest)[1],6)
        self.assertEqual(reader.call_args_list[-1].args,(latest,))
        session.observe.assert_not_called()

    def test_one_fresh_confirmation_and_still_unknown_is_not_invented(self):
        session=Mock(); frame=object()
        with patch('runner.read_reroll_button',side_effect=[None,None]):
            self.assertIsNone(read_shop_reroll(session,frame,frame))
        session.observe.assert_called_once_with('shop-reroll-confirmation')

    def test_fresh_confirmation_can_supply_a_verified_price(self):
        session=Mock(); initial,latest=object(),object()
        with patch('runner.read_reroll_button',side_effect=[None,None,((755,635),7)]):
            self.assertEqual(read_shop_reroll(session,initial,latest)[1],7)
        session.observe.assert_called_once()


class EarlyGrowthTests(unittest.TestCase):
    def test_growth_investment_requires_measured_growth_and_safe_coverage(self):
        state=RunState(ante=2,money=15,jokers=(Joker('Joker',4),),total_joker_slots=1)
        item=ShopItem('joker','Square Joker',4,Joker('Square Joker'))
        old=replace(forecast(safe=10000,target=2000),development_score=10000)
        new=replace(forecast(safe=9000,target=2000),development_score=16000)
        new=replace(new,next_round=replace(new.next_round,growth_per_round=(('square_joker',4.),)))
        with patch('planning.shop_forecast',side_effect=lambda s:new if s.has('square_joker') else old):
            d=decide_shop([item],state)
        self.assertEqual(d.action,'sell_then_buy')
        lethal=replace(new,next_round=replace(new.next_round,safe_score=1000,clear_rate=0.))
        with patch('planning.shop_forecast',side_effect=lambda s:lethal if s.has('square_joker') else old):
            self.assertEqual(decide_shop([item],state).action,'leave')

    def test_history_early_square_has_real_growth_but_needs_no_imaginary_shop(self):
        from tools.review_growth_20261015 import items_from_event
        fixture=json.loads((ROOT/'tests/fixtures/growth_review_20261015.json').read_text('utf8'))
        case=next(r for r in fixture['cases'] if r['name']=='early_square')
        event=case['event'];state=state_from_dict(event['state'])
        d=decide_shop(items_from_event(event),state,reroll_cost=event['reroll_cost'])
        self.assertEqual((d.action,d.item_index,d.sell_index),('sell_then_buy',1,3))


if __name__ == '__main__':
    unittest.main()
