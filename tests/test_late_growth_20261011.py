"""Known producers, net losses, feeding budgets, and safe growth choices."""
import unittest
from dataclasses import asdict, replace
from unittest.mock import patch

from engine_development import blue_planet_yield, development_score, play_development_value
from planning import advance_projected_blind_start, forecast_round, engine_profile
from shop_extras import extra_is_worth_considering, sealed_pack_value
from strategy import Card, Joker, RunState, ShopItem, ScoreEstimate, decide_play, _draw_best_plan, SUITS
from tests.test_speed_and_policy_20261005 import forecast


def blue_engine(**changes):
    state = RunState(ante=3, money=30, hand_size=2, deck_size=2,
                     deck_cards=(Card(14, 'S', seal='blue'), Card(13, 'H')),
                     jokers=(Joker('Constellation', 1), Joker('Mime')),
                     discards_left=0, base_discards_per_round=0)
    return replace(state, **changes)


class BlueProductionTests(unittest.TestCase):
    def test_mime_copiers_and_slots_limit_production(self):
        held = [Card(14, 'S', seal='blue')]
        state = blue_engine(consumable_slots=4,
                            jokers=(Joker('Blueprint'), Joker('Mime')))
        self.assertEqual(blue_planet_yield(held, state), 3)
        self.assertEqual(blue_planet_yield(held, replace(state, consumables_held=3)), 1)
        self.assertEqual(blue_planet_yield(held, replace(state, consumables_held=4)), 0)

    def test_debuff_hidden_and_hook_are_not_promised_sources(self):
        state = blue_engine()
        self.assertEqual(blue_planet_yield([Card(14, 'S', seal='blue', debuffed=True)], state), 0)
        self.assertEqual(blue_planet_yield([Card(14, 'S', seal='blue', face_down=True)], state), 0)
        self.assertEqual(blue_planet_yield(list(state.deck_cards), replace(state, boss='The Hook')), 0)
        self.assertEqual(blue_planet_yield([state.deck_cards[0]] * 3,
                                         replace(state, boss='The Hook')), 2)

    def test_real_and_forecast_policy_keep_blue_on_a_confirmed_clear(self):
        state = blue_engine(target_score=1)
        hand = list(state.deck_cards)
        actual = decide_play(hand, state, lookahead=False)
        simulated, estimate = _draw_best_plan(hand, state)
        self.assertNotIn(0, actual.indices)
        self.assertNotIn(0, simulated)
        self.assertGreaterEqual(estimate.score, 1)

    def test_clear_forecast_levels_and_feeds_only_once_after_blind(self):
        state = blue_engine()
        original = asdict(state)
        result = forecast_round(state, target=1, samples=2)
        self.assertEqual(result.blue_planets_per_round, 2)
        self.assertEqual(dict(result.level_gains_per_round)['high_card'], 2)
        self.assertAlmostEqual(dict(result.growth_per_round)['constellation'], .2)
        self.assertEqual(sum(dict(result.plays_per_round).values()), 1)
        self.assertEqual(asdict(state), original)

    def test_each_physical_constellation_grows_but_blueprint_is_not_a_third(self):
        state = blue_engine(jokers=(Joker('Blueprint'), Joker('Constellation', 1),
                                    Joker('Constellation', 1)), consumable_slots=1)
        result = forecast_round(state, target=1, samples=2)
        self.assertEqual(result.blue_planets_per_round, 1)
        self.assertAlmostEqual(dict(result.growth_per_round)['constellation'], .2)

    def test_full_slots_loss_or_unproven_random_clear_get_no_feed(self):
        for state, target in ((blue_engine(consumables_held=2), 1),
                              (blue_engine(), 1_000_000)):
            result = forecast_round(state, target=target, samples=2)
            self.assertEqual(result.blue_planets_per_round, 0)
            self.assertEqual(dict(result.growth_per_round).get('constellation', 0), 0)
        estimate = ScoreEstimate(100, 'high_card', 10, 10, (0,))
        def made(hand, state):
            return ((next((i for i, c in enumerate(hand) if not c.seal), 0),), estimate)
        with patch('planning._draw_best_plan', side_effect=made), \
                patch('planning._play_floor', return_value=0):
            result = forecast_round(blue_engine(), target=1, samples=2)
        self.assertEqual(result.blue_planets_per_round, 0)

    def test_observatory_does_not_count_both_holding_and_consuming_planets(self):
        result = forecast_round(blue_engine(observatory=True), target=1, samples=2)
        self.assertGreater(result.blue_planets_per_round, 0)
        self.assertEqual(dict(result.level_gains_per_round).get('high_card', 0), 0)
        self.assertEqual(dict(result.growth_per_round).get('constellation', 0), 0)

    def test_production_changes_future_value_not_current_survival_score(self):
        state = blue_engine()
        sample = forecast_round(state, target=1, samples=2)
        projected = replace(forecast(), next_round=sample)
        self.assertGreater(development_score(state, projected), projected.sustainable_score)
        self.assertEqual(development_score(replace(state, ante=8), projected), projected.sustainable_score)

    def test_known_blue_level_producer_is_a_growth_path_even_without_scaling_joker(self):
        state = blue_engine(jokers=())
        self.assertNotIn('no_scaling', engine_profile(state).weaknesses)
        self.assertIn('no_scaling', engine_profile(replace(state, consumables_held=2)).weaknesses)
        self.assertIn('no_multiplier_or_retrigger', engine_profile(state).weaknesses)

    def test_full_deck_pair_engine_levels_only_its_winning_hand(self):
        cards = tuple(Card(r, s, seal='blue' if r in (2, 3) else '')
                      for s in SUITS for r in range(2, 15))
        state = RunState(ante=3, money=30, deck_cards=cards,
            jokers=(Joker('Half Joker', 20), Joker('Sly Joker', 50),
                    Joker('Constellation', 1), Joker('Mime')))
        result = forecast_round(state, target=2000, samples=6)
        self.assertGreater(result.blue_planets_per_round, 0)
        self.assertEqual(set(dict(result.level_gains_per_round)), {'pair'})
        self.assertLessEqual(result.blue_planets_per_round, state.consumable_slots)
        self.assertAlmostEqual(dict(result.growth_per_round)['constellation'],
                               .1 * result.blue_planets_per_round)

    def test_losing_last_hand_never_sacrifices_score_to_save_blue(self):
        state = blue_engine(target_score=16, hands_left=1, jokers=())
        play = decide_play(state.deck_cards, state, lookahead=False)
        self.assertIn(0, play.indices)
        self.assertGreaterEqual(play.estimate.score, 16)


class AutomaticGrowthTests(unittest.TestCase):
    def test_marble_copies_grow_each_physical_hologram_and_add_stones(self):
        state = blue_engine(jokers=(Joker('Blueprint'), Joker('Marble Joker'),
                                    Joker('Hologram', 1.5), Joker('Hologram', 2)))
        original = asdict(state)
        after = advance_projected_blind_start(state)
        self.assertEqual(after.deck_size, 4)
        self.assertEqual(sum(c.is_stone for c in after.deck_cards), 2)
        self.assertEqual([j.value for j in after.jokers if j.key == 'hologram'], [2., 2.5])
        self.assertEqual(asdict(state), original)

    def test_debuffed_producers_do_not_add_cards(self):
        state = blue_engine(jokers=(Joker('Marble Joker', debuffed=True), Joker('Hologram', 1)))
        self.assertEqual(advance_projected_blind_start(state), state)

    def test_stone_joker_observed_chip_counter_and_known_deck_stay_consistent(self):
        state = blue_engine(jokers=(Joker('Marble Joker'),
                                    Joker('Stone Joker', 50, data=(('observed_value', 1),))))
        after = advance_projected_blind_start(state)
        self.assertEqual(after.jokers[1].value, 75)
        self.assertEqual(len(after.deck_cards), after.deck_size)

    def test_blind_start_growth_is_measured_against_original_roster(self):
        state = blue_engine(jokers=(Joker('Marble Joker'), Joker('Hologram', 1)),
                            deck_cards=(Card(14, 'S'), Card(13, 'H')))
        result = forecast_round(state, target=1, samples=2)
        self.assertAlmostEqual(dict(result.growth_per_round)['hologram'], .25)

    def test_empty_draw_pile_does_not_end_training_while_held_cards_can_play(self):
        state = blue_engine(jokers=(Joker('Green Joker'),))
        estimate = ScoreEstimate(10, 'high_card', 10, 1, (0,))
        with patch('planning._draw_best_plan', return_value=((0,), estimate)):
            result = forecast_round(state, target=20, samples=2)
        self.assertEqual(dict(result.growth_per_round)['green_joker'], 2)
        self.assertEqual(sum(dict(result.plays_per_round).values()), 2)

    def test_negative_net_growth_is_not_clamped_into_a_stable_engine(self):
        state = RunState(ante=3, jokers=(Joker('Ride the Bus', 20),))
        f = forecast()
        f = replace(f, next_round=replace(f.next_round, growth_per_round=(('ride_the_bus', -20),)))
        original = asdict(state)
        self.assertLess(development_score(state, f), f.sustainable_score)
        self.assertEqual(asdict(state), original)

    def test_no_producer_or_unknown_purchases_create_imaginary_growth(self):
        state = RunState(ante=3, money=100, jokers=(Joker('Hologram', 1),))
        self.assertEqual(development_score(state, forecast()), forecast().sustainable_score)


class FeedingBudgetTests(unittest.TestCase):
    def test_safe_hologram_can_spend_one_interest_tier_but_not_rescue_money(self):
        item = ShopItem('standard_pack', 'Standard Pack', 4)
        state = blue_engine(money=24, jokers=(Joker('Hologram', 1.5),))
        safe = forecast(safe=20000)
        self.assertTrue(extra_is_worth_considering(item, state, safe))
        self.assertFalse(extra_is_worth_considering(item, replace(state, money=23), safe))
        self.assertFalse(extra_is_worth_considering(item, state, forecast(safe=5000)))
        for name in ('Bull', 'Bootstraps'):
            self.assertFalse(extra_is_worth_considering(item, replace(state,
                jokers=state.jokers + (Joker(name),)), safe))

    def test_marginal_feeding_value_decreases_with_existing_multiplier(self):
        safe = forecast(safe=20000)
        for kind, name, joker in (('standard_pack', 'Standard Pack', 'Hologram'),
                                  ('celestial_pack', 'Celestial Pack', 'Constellation')):
            item = ShopItem(kind, name, 4)
            young = blue_engine(jokers=(Joker(joker, 1),))
            mature = replace(young, jokers=(Joker(joker, 5),))
            self.assertGreater(sealed_pack_value(item, young, safe), sealed_pack_value(item, mature, safe))
            self.assertGreater(sealed_pack_value(ShopItem(kind, 'Mega ' + name, 8), young, safe),
                               sealed_pack_value(item, young, safe))


if __name__ == '__main__':
    unittest.main()
