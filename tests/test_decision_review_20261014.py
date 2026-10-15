"""Log-backed resource budgets and conditional multiplier reliability."""
import unittest
from dataclasses import replace
from unittest.mock import patch

from strategy import Card, Joker, RunState, ShopItem, ScoreEstimate, decide_shop
from planning import (conditional_multiplier_hits, engine_profile, forecast_round,
                      ShopForecast, RoundForecast)


def projection(state, score=12000, target=10000, rate=1., frequencies=()):
    sample = RoundForecast(target, '', score, score, rate, score / 4, 4, 3,
                           frequencies, (score,))
    return ShopForecast(sample, None, score, engine_profile(state))


class ConditionalMultiplierTests(unittest.TestCase):
    def setUp(self):
        self.rainbow = [Card(14, 'S'), Card(14, 'H'), Card(13, 'D'), Card(13, 'C')]
        self.state = RunState(ante=5, jokers=(Joker('Flower Pot'),))

    def test_four_scoring_suits_activate_but_three_do_not(self):
        attempts, hits = conditional_multiplier_hits(self.rainbow, (0, 1, 2, 3), self.state)
        self.assertEqual((attempts['flower_pot'], hits['flower_pot']), (1, 1))
        cards = self.rainbow[:3] + [Card(13, 'S')]
        self.assertEqual(conditional_multiplier_hits(cards, (0, 1, 2, 3), self.state)[1]['flower_pot'], 0)

    def test_rainbow_high_card_needs_splash(self):
        hand = [Card(14, 'S'), Card(10, 'H'), Card(7, 'D'), Card(3, 'C')]
        self.assertEqual(conditional_multiplier_hits(hand, (0, 1, 2, 3), self.state)[1]['flower_pot'], 0)
        splash = replace(self.state, jokers=self.state.jokers + (Joker('Splash'),))
        self.assertEqual(conditional_multiplier_hits(hand, (0, 1, 2, 3), splash)[1]['flower_pot'], 1)

    def test_copies_share_condition_without_inflating_probability(self):
        state = replace(self.state, jokers=(Joker('Blueprint'), Joker('Flower Pot')))
        attempts, hits = conditional_multiplier_hits(self.rainbow, (0, 1, 2, 3), state)
        self.assertEqual((attempts['flower_pot'], hits['flower_pot']), (2, 2))
        self.assertEqual(hits['flower_pot'] / attempts['flower_pot'], 1)

    def test_disabled_effects_are_not_counted(self):
        state = replace(self.state, jokers=(Joker('Flower Pot', debuffed=True),))
        self.assertFalse(conditional_multiplier_hits(self.rainbow, (0, 1, 2, 3), state)[0])

    def test_card_sharp_requires_this_round_history_and_a_legal_play(self):
        state = RunState(jokers=(Joker('Card Sharp'),))
        self.assertEqual(conditional_multiplier_hits(self.rainbow, (0,), state)[1]['card_sharp'], 0)
        state = replace(state, round_played_types={'high_card': 1})
        self.assertEqual(conditional_multiplier_hits(self.rainbow, (0,), state)[1]['card_sharp'], 1)
        state = replace(state, boss='The Eye', boss_seen_types=frozenset({'high_card'}))
        self.assertEqual(conditional_multiplier_hits(self.rainbow, (0,), state)[1]['card_sharp'], 0)

    def test_hook_empty_hold_cannot_supply_blackboard(self):
        state = RunState(jokers=(Joker('Blackboard'),), boss='The Hook')
        hand = [Card(14, 'H'), Card(13, 'S'), Card(10, 'C')]
        self.assertEqual(conditional_multiplier_hits(hand, (0,), state)[1]['blackboard'], 0)
        hand.append(Card(4, 'C'))
        self.assertEqual(conditional_multiplier_hits(hand, (0,), state)[1]['blackboard'], 1)

    def test_rare_condition_does_not_fill_multiplier_role(self):
        profile = engine_profile(self.state, conditional_rates=(('flower_pot', .1),))
        self.assertEqual(profile.multiplicative_mult, 0)
        self.assertIn('no_multiplier_or_retrigger', profile.weaknesses)
        self.assertIn('unreliable_multiplier', profile.weaknesses)

    def test_supported_condition_fills_role(self):
        profile = engine_profile(self.state, conditional_rates=(('flower_pot', .8),))
        self.assertEqual(profile.multiplicative_mult, 1)
        self.assertNotIn('unreliable_multiplier', profile.weaknesses)

    def test_moderate_flower_pot_activation_is_not_stable_support(self):
        profile = engine_profile(self.state, conditional_rates=(('flower_pot', .625),))
        self.assertEqual(profile.multiplicative_mult, 1)
        self.assertIn('unreliable_multiplier', profile.weaknesses)

    def test_normal_card_sharp_first_play_is_not_a_random_failure(self):
        state = RunState(ante=5, jokers=(Joker('Card Sharp'),))
        profile = engine_profile(state, conditional_rates=(('card_sharp', 2 / 3),))
        self.assertNotIn('unreliable_multiplier', profile.weaknesses)

    def test_other_reliable_multiplier_preserves_complete_role(self):
        state = replace(self.state, jokers=self.state.jokers + (Joker('Cavendish'),))
        profile = engine_profile(state, conditional_rates=(('flower_pot', 0.),))
        self.assertEqual(profile.multiplicative_mult, 1)
        self.assertNotIn('no_multiplier_or_retrigger', profile.weaknesses)
        self.assertNotIn('unreliable_multiplier', profile.weaknesses)

    def test_ramen_verified_x1_bound_is_not_powered_but_new_default_is(self):
        bounded = Joker('Ramen', 1, data=(('counter_lower_bound', 1),))
        self.assertEqual(engine_profile(RunState(ante=5, jokers=(bounded,))).multiplicative_mult, 0)
        self.assertEqual(engine_profile(RunState(ante=5, jokers=(Joker('Ramen'),))).multiplicative_mult, 1)

    def test_forecast_activation_uses_actual_scoring_cards(self):
        state = replace(self.state, deck_cards=tuple(self.rainbow), deck_size=4,
                        hand_size=4, base_discards_per_round=0)
        result = forecast_round(state, target=1, samples=2)
        self.assertEqual(dict(result.conditional_xmult_rates)['flower_pot'], 1)
        wrong = replace(state, deck_cards=tuple(self.rainbow[:3] + [Card(13, 'S')]))
        self.assertEqual(dict(forecast_round(wrong, target=1, samples=2).conditional_xmult_rates)['flower_pot'], 0)

    def test_post_clear_hypothetical_plays_do_not_inflate_activation(self):
        state = RunState(jokers=(Joker('Card Sharp'),), deck_cards=(Card(14, 'S'),) * 12,
                         deck_size=12, hand_size=4, base_hands_per_round=3, base_discards_per_round=0)
        result = ScoreEstimate(10, 'high_card', 10, 1, (0,))
        with patch('planning._draw_best_plan', return_value=((0,), result)):
            clear_first = forecast_round(state, target=1, samples=1)
            need_three = forecast_round(state, target=30, samples=1)
        self.assertEqual(dict(clear_first.conditional_xmult_rates)['card_sharp'], 0)
        self.assertAlmostEqual(dict(need_three.conditional_xmult_rates)['card_sharp'], 2 / 3)


class ShopBudgetTests(unittest.TestCase):
    def setUp(self):
        self.state = RunState(ante=4, money=30, jokers=(Joker('Joker', 4),), objective='survival')

    def test_unfed_zero_red_card_cannot_replace_a_scoring_joker(self):
        state = replace(self.state, total_joker_slots=1)
        item = ShopItem('joker', 'Red Card', 5, Joker('Red Card', 0))
        with patch('planning.shop_forecast', side_effect=lambda s: projection(s, 40000 if s.has('red_card') else 12000)):
            self.assertEqual(decide_shop([item], state).action, 'leave')

    def test_grown_red_card_can_still_be_bought(self):
        item = ShopItem('joker', 'Red Card', 5, Joker('Red Card', 18))
        with patch('planning.shop_forecast', side_effect=lambda s: projection(s, 40000 if s.has('red_card') else 12000)):
            self.assertEqual(decide_shop([item], self.state).action, 'buy')

    def test_zero_red_card_with_scoring_edition_is_not_blank(self):
        item = ShopItem('joker', 'Red Card', 5, Joker('Red Card', 0, edition='foil'))
        with patch('planning.shop_forecast', side_effect=lambda s: projection(s, 40000 if s.has('red_card') else 12000)):
            self.assertEqual(decide_shop([item], self.state).action, 'buy')

    def test_thin_late_engine_searches_before_buying_income(self):
        item = ShopItem('joker', 'Golden Joker', 4, Joker('Golden Joker'))
        with patch('planning.shop_forecast', side_effect=lambda s: projection(s, rate=.83)):
            decision = decide_shop([item], self.state, reroll_cost=5)
        self.assertEqual(decision.action, 'reroll')

    def test_income_cannot_claim_improved_score_from_a_different_sampled_route(self):
        item = ShopItem('joker', 'Mail-In Rebate', 4, Joker('Mail-In Rebate'))
        state = replace(self.state, jokers=(Joker('Clever Joker'),), total_joker_slots=1)
        with patch('planning.shop_forecast', side_effect=lambda s: projection(s,
                   13000 if s.has('mail_in_rebate') else 12000, rate=.83)):
            self.assertEqual(decide_shop([item], state, reroll_cost=5).action, 'reroll')

    def test_income_can_supply_real_abstract_mult_when_it_fills_a_slot(self):
        item = ShopItem('joker', 'Golden Joker', 4, Joker('Golden Joker'))
        state = replace(self.state, jokers=(Joker('Abstract Joker'),))
        with patch('planning.shop_forecast', side_effect=lambda s: projection(s,
                   15000 if s.has('golden_joker') else 12000, rate=.83)):
            self.assertEqual(decide_shop([item], state, reroll_cost=5).action, 'buy')

    def test_safe_late_and_early_engines_can_invest(self):
        item = ShopItem('joker', 'Golden Joker', 4, Joker('Golden Joker'))
        for state in (self.state, replace(self.state, ante=1, money=15)):
            with patch('planning.shop_forecast', side_effect=lambda s: projection(s, score=20000)):
                self.assertEqual(decide_shop([item], state).action, 'buy')

    def test_income_joker_scoring_edition_can_rescue_an_emergency(self):
        item = ShopItem('joker', 'Golden Joker', 4, Joker('Golden Joker', edition='holographic'))
        with patch('planning.shop_forecast', side_effect=lambda s: projection(s, 18000 if s.has('golden_joker') else 8000)):
            self.assertEqual(decide_shop([item], self.state, reroll_cost=5).action, 'buy')

    def test_unknown_pack_at_exact_old_boundary_does_not_take_search_budget(self):
        pack = ShopItem('celestial_pack', 'Celestial Pack', 4)
        with patch('planning.shop_forecast', side_effect=lambda s: projection(s, rate=.83)), \
             patch('shop_extras.sealed_pack_value', return_value=12.):
            self.assertEqual(decide_shop([pack], self.state, reroll_cost=5).action, 'reroll')

    def test_constellation_feeding_keeps_priority(self):
        state = replace(self.state, jokers=(Joker('Constellation', 1.5),))
        pack = ShopItem('celestial_pack', 'Celestial Pack', 4)
        with patch('planning.shop_forecast', side_effect=lambda s: projection(s, rate=.83)), \
             patch('shop_extras.sealed_pack_value', return_value=12.):
            self.assertEqual(decide_shop([pack], state, reroll_cost=5).action, 'buy')

    def test_search_requires_separate_purchase_budget(self):
        state = replace(self.state, money=10)
        pack = ShopItem('celestial_pack', 'Celestial Pack', 4)
        with patch('planning.shop_forecast', side_effect=lambda s: projection(s, rate=.83)):
            self.assertNotEqual(decide_shop([pack], state, reroll_cost=5).action, 'reroll')

    def test_small_early_planet_gain_does_not_spend_last_scoring_budget(self):
        state = RunState(ante=1, money=5, jokers=(Joker('Wily Joker'),))
        item = ShopItem('planet', 'Jupiter', 3)
        with patch('planning.shop_forecast', side_effect=lambda s: projection(s,
                   1100 if s.level('flush') > 1 else 1000, target=450, frequencies=(('flush', .1),))):
            self.assertEqual(decide_shop([item], state).action, 'leave')

    def test_planet_that_repairs_coverage_can_use_last_cash(self):
        state = RunState(ante=1, money=3, jokers=(Joker('Wily Joker'),))
        item = ShopItem('planet', 'Jupiter', 3)
        with patch('planning.shop_forecast', side_effect=lambda s: projection(s,
                   650 if s.level('flush') > 1 else 400, target=500, frequencies=(('flush', .3),))):
            self.assertEqual(decide_shop([item], state).action, 'buy')

    def test_low_frequency_planet_without_scoring_gain_is_skipped(self):
        item = ShopItem('planet', 'Jupiter', 3)
        with patch('planning.shop_forecast', side_effect=lambda s: projection(s, frequencies=(('flush', .06),))), \
             patch('strategy._item_value', return_value=100):
            self.assertEqual(decide_shop([item], self.state).action, 'leave')

    def test_known_constellation_route_can_use_low_frequency_planet(self):
        state = replace(self.state, jokers=(Joker('Constellation', 1.5),))
        item = ShopItem('planet', 'Jupiter', 3)
        with patch('planning.shop_forecast', side_effect=lambda s: projection(s, frequencies=(('flush', .06),))), \
             patch('strategy._item_value', return_value=100):
            self.assertEqual(decide_shop([item], state).action, 'buy')


if __name__ == '__main__':
    unittest.main()
