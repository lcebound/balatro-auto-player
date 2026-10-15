"""Growth is an investment with conditions, never an observed future score."""
import unittest
from dataclasses import asdict, replace
from unittest.mock import patch

from engine_development import development_score, play_development_value
from planning import forecast_round, engine_profile
from strategy import Card, Joker, RunState, ShopItem, ScoreEstimate, decide_play, decide_shop, score_hand
from tests.test_speed_and_policy_20261005 import forecast


class ScoreGrowthTests(unittest.TestCase):
    def test_level_scalers_count_as_growth_and_unpowered_x1_is_not_active_mult(self):
        for name in ('Burnt Joker', 'Space Joker'):
            profile = engine_profile(RunState(ante=4, jokers=(Joker(name),)))
            self.assertGreater(profile.growth, 0)
            self.assertNotIn('no_scaling', profile.weaknesses)
        state = RunState(ante=4, jokers=(Joker('Constellation', 1),))
        self.assertEqual(engine_profile(state).multiplicative_mult, 0)
        self.assertIn('no_multiplier_or_retrigger', engine_profile(state).weaknesses)
        self.assertEqual(engine_profile(replace(state, jokers=(Joker('Constellation', 1.3),))).multiplicative_mult, 1)

    def test_forecast_trains_only_until_actual_clear_not_all_budgeted_hands(self):
        state = RunState(ante=2, jokers=(Joker('Green Joker'),), base_hands_per_round=4,
                         hands_left=4, discards_left=0, base_discards_per_round=0)
        estimate = ScoreEstimate(100, 'high_card', 10, 10, (0,))
        with patch('planning._draw_best_plan', return_value=((0,), estimate)):
            result = forecast_round(state, target=1, samples=2)
        self.assertEqual(result.mean_score, 400)
        self.assertEqual(dict(result.growth_per_round)['green_joker'], 1)
        self.assertEqual(dict(result.plays_per_round), {'high_card': 1})

    def test_growth_projection_changes_only_estimate_and_scales_with_available_rounds(self):
        state = RunState(ante=3, jokers=(Joker('Supernova'),), played_types={'pair': 8}, money=30)
        f = forecast(safe=12000)
        f = replace(f, next_round=replace(f.next_round, plays_per_round=(('pair', 2.),)))
        original = asdict(state)
        value = development_score(state, f)
        self.assertGreater(value, f.sustainable_score)
        self.assertEqual(asdict(state), original)
        self.assertEqual(development_score(replace(state, ante=8), f), f.sustainable_score)

    def test_growth_requiring_unknown_future_purchases_is_not_invented(self):
        for name in ('Constellation', 'Hologram', 'Lucky Cat'):
            state = RunState(ante=3, jokers=(Joker(name, 1),), money=50)
            f = forecast()
            self.assertEqual(development_score(state, f), f.sustainable_score)

    def test_immediate_survival_cannot_be_replaced_by_an_optimistic_growth_estimate(self):
        state = RunState(ante=4, money=30, jokers=(Joker('Joker', 4),))
        old = replace(forecast(safe=4000, target=10000), development_score=8000)
        fake_future = replace(old, development_score=100000)
        with patch('planning.shop_forecast', side_effect=lambda s: fake_future if s.has('green_joker') else old), \
                patch('strategy._item_value', return_value=100):
            result = decide_shop([ShopItem('joker', 'Green Joker', 4, Joker('Green Joker'))], state)
        self.assertEqual(result.action, 'leave')

    def test_near_equal_upgrade_does_not_sell_supported_growth(self):
        state = RunState(ante=3, money=30, total_joker_slots=1,
                         jokers=(Joker('Supernova'),), played_types={'pair': 10})
        old = replace(forecast(safe=20000, target=10000), development_score=30000)
        after = replace(forecast(safe=21000, target=10000), development_score=21000)
        with patch('planning.shop_forecast', side_effect=[old, after]), \
                patch('strategy._item_value', return_value=0), patch('strategy._joker_purchase_value', return_value=0):
            result = decide_shop([ShopItem('joker', 'Abstract Joker', 4, Joker('Abstract Joker'))], state)
        self.assertEqual(result.action, 'leave')

    def test_no_growth_roster_can_search_while_current_blind_is_safe(self):
        state = RunState(ante=4, money=45, jokers=(Joker('Joker', 4),))
        with patch('planning.shop_forecast', return_value=forecast(safe=25000, target=10000)):
            self.assertEqual(decide_shop([], state, reroll_cost=5).action, 'reroll')
            self.assertEqual(decide_shop([], replace(state, money=30), reroll_cost=5).action, 'leave')
            self.assertEqual(decide_shop([], replace(state, shop_rerolls=2), reroll_cost=5).action, 'leave')

    def test_blue_seal_is_kept_when_another_card_already_clears(self):
        cards = [Card(14, 'S', seal='blue'), Card(13, 'H'), Card(12, 'D'), Card(8, 'C'),
                 Card(6, 'S'), Card(4, 'H'), Card(2, 'D'), Card(3, 'C')]
        state = RunState(ante=3, target_score=10, hands_left=4)
        decision = decide_play(cards, state, lookahead=False)
        self.assertGreaterEqual(decision.estimate.score, state.score_needed)
        self.assertNotIn(0, decision.indices)
        full = decide_play(cards, replace(state, consumables_held=2), lookahead=False)
        self.assertIn(0, full.indices)

    def test_last_hand_preserves_the_only_confirmed_clear_even_if_blue_is_scored(self):
        cards = [Card(14, 'S', seal='blue'), Card(13, 'H'), Card(12, 'D'), Card(8, 'C')]
        state = RunState(ante=3, target_score=16, hands_left=1)
        decision = decide_play(cards, state, lookahead=False)
        self.assertIn(0, decision.indices)
        self.assertGreaterEqual(decision.estimate.score, 16)

    def test_uncertain_clear_gets_no_guaranteed_blue_seal_reward(self):
        hand = [Card(13, 'H'), Card(14, 'S', seal='blue')]
        estimate = score_hand(hand[:1])
        state = RunState(ante=3, target_score=10)
        self.assertEqual(play_development_value((0,), estimate, hand, state, guaranteed_score=0), 0)


if __name__ == '__main__':
    unittest.main()
