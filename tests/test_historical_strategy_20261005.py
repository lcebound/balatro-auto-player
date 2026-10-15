"""Strategy regressions from the twelve completed games; no device control."""
import unittest
from collections import OrderedDict
from dataclasses import replace
from unittest.mock import patch

from consumables import (Consumable, ConsumableDecision, consumed_engine,
                         state_after_consumable, transformed_state, decide_consumable)
from perception import parse_joker_tooltip
from planning import (advance_projected_discard, advance_projected_play,
                      first_discard, shop_forecast)
from strategy import (Card, Joker, RunState, ShopItem, decide_play, score_hand,
                      optimize_play_order, _play_floor, _standard_unseen)
from strategy import _draw_best_plan, _discard_candidates
from tests.test_speed_and_policy_20261005 import forecast


class SurvivalChoiceTests(unittest.TestCase):
    def test_a_known_clear_beats_a_higher_random_mean(self):
        hand = [Card(r, 'S') for r in (2, 7, 10, 12, 14)] + [Card(r, 'H') for r in (3, 4, 5, 8, 11)]
        state = RunState(jokers=(Joker('Bloodstone'),), levels={'flush': 10},
                         target_score=4500, hands_left=1, discards_left=0)
        risky = score_hand(hand[5:], state, hand[:5])
        safe = score_hand(hand[:5], state, hand[5:])
        self.assertGreater(risky.score, safe.score + 1000)
        self.assertLess(_play_floor(tuple(range(5, 10)), risky, hand, state), state.score_needed)
        chosen = decide_play(hand, state)
        self.assertEqual(set(chosen.indices), set(range(5)))
        chosen = optimize_play_order(hand, state, chosen)
        self.assertGreaterEqual(_play_floor(chosen.play_order, chosen.estimate, hand, state), state.score_needed)

    def test_misprint_is_not_a_guaranteed_bonus(self):
        hand = [Card(14, 'S')]
        state = RunState(jokers=(Joker('Misprint'),), target_score=100)
        estimate = score_hand(hand, state)
        self.assertGreater(estimate.score, 100)
        self.assertEqual(_play_floor((0,), estimate, hand, state), 16)

    def test_discard_and_shop_samples_use_the_same_safe_clear_rule(self):
        hand = [Card(r, 'S') for r in (2, 7, 10, 12, 14)] + [Card(r, 'H') for r in (3, 4, 5, 8, 11)]
        state = RunState(jokers=(Joker('Bloodstone'),), levels={'flush': 10},
                         target_score=4500, hands_left=1)
        indices, result = _draw_best_plan(hand, state)
        self.assertEqual(set(indices), set(range(5)))
        self.assertGreaterEqual(_play_floor(indices, result, hand, state), state.score_needed)

    def test_shop_repairs_boss_coverage_before_chasing_easy_blind_points(self):
        from strategy import decide_shop
        state = RunState(ante=5, money=50, blind_index=0)
        before = forecast(safe=20000, target=20000, boss_safe=12000)
        risky = forecast(safe=90000, target=20000, boss_safe=39000)
        balanced = forecast(safe=43000, target=20000, boss_safe=43000)
        items = [ShopItem('joker', 'Cavendish', 5, Joker('Cavendish')),
                 ShopItem('joker', 'Joker', 5, Joker('Joker', 4))]
        with patch('planning.shop_forecast', side_effect=[before, risky, balanced]):
            choice = decide_shop(items, state)
        self.assertEqual((choice.action, choice.item_index), ('buy', 1))


class DiscardGrowthTests(unittest.TestCase):
    def test_burnt_search_can_discard_the_pair_instead_of_only_preserving_it(self):
        hand = [Card(2, 'S'), Card(2, 'H')] + [Card(r, 'D') for r in (4, 7, 9, 10, 12, 14)]
        state = RunState(jokers=(Joker('Burnt Joker'),), round_discards_used=0)
        keep = tuple(range(2, 8))
        candidates = _discard_candidates(hand, state)
        self.assertEqual(candidates[keep], 'Burnt upgrade pair')
        self.assertNotIn(keep, _discard_candidates(hand, state, include_discard_effects=False))

    def test_large_hand_shortlist_keeps_preferred_burnt_upgrade(self):
        from planning import diverse_draw_candidates
        hand = [Card(2, 'S'), Card(2, 'H')] + [Card(r, 'D') for r in range(3, 14)]
        state = RunState(jokers=(Joker('Burnt Joker'),), round_discards_used=0,
                         played_types={'pair': 20}, hand_size=13)
        candidates = _discard_candidates(hand, state)
        short = diverse_draw_candidates(hand, state, candidates)
        self.assertIn('Burnt upgrade pair', short.values())
        self.assertLessEqual(len(short), 14)

    def test_round_forecast_hold_values_the_upgrade_not_only_retained_cards(self):
        from planning import _hold_potential
        hand = [Card(2, 'S'), Card(2, 'H')] + [Card(r, 'D') for r in (4, 7, 9, 10, 12, 14)]
        state = RunState(jokers=(Joker('Burnt Joker'),), round_discards_used=0)
        keep = tuple(range(2, 8))
        enabled = _hold_potential(hand, keep, state, 'pair')
        after_first = _hold_potential(hand, keep, replace(state, round_discards_used=1), 'pair')
        self.assertGreater(enabled, after_first + 20)

    def test_burnt_upgrades_discarded_type_only_on_first_discard(self):
        hand = [Card(7, 'S'), Card(7, 'H')]
        state = RunState(jokers=(Joker('Blueprint'), Joker('Burnt Joker')),
                         round_discards_used=0, discards_left=3)
        first = advance_projected_discard(state, hand)
        self.assertEqual(first.level('pair'), 3)
        self.assertEqual(first.round_discards_used, 1)
        second = advance_projected_discard(first, hand)
        self.assertEqual(second.level('pair'), 3)
        self.assertEqual(state.level('pair'), 1)

    def test_old_checkpoint_uses_discard_capacity_not_hand_play_history(self):
        self.assertTrue(first_discard(RunState(discards_left=3, boss_first_hand_done=True)))
        self.assertFalse(first_discard(RunState(discards_left=2, boss_first_hand_done=False)))

    def test_castle_yorick_ramen_and_green_counters_advance_together(self):
        jokers = (Joker('Castle', 7, data=(('suit', 0),)),
                  Joker('Yorick', 2, data=(('discards_remaining', 2), ('discard_threshold', 23), ('discard_growth', 1))),
                  Joker('Ramen', 2), Joker('Green Joker', 4))
        state = RunState(jokers=jokers, discards_left=3, draw_pile_size=20)
        cards = [Card(5, 'D', enhancement='wild'), Card(2, 'S'), Card(3, 'S', debuffed=True)]
        after = advance_projected_discard(state, cards)
        self.assertEqual(after.jokers[0].value, 13)
        self.assertEqual(after.jokers[1].value, 3)
        self.assertEqual(after.jokers[1].get('discards_remaining'), 22)
        self.assertAlmostEqual(after.jokers[2].value, 1.97)
        self.assertEqual(after.jokers[3].value, 3)
        self.assertEqual((after.discards_left, after.draw_pile_size), (2, 17))
        self.assertEqual(state.jokers[0].value, 7)

    def test_unknown_yorick_counter_and_debuffed_growth_are_not_invented(self):
        state = RunState(jokers=(Joker('Yorick', 2), Joker('Green Joker', 4, debuffed=True),
                                 Joker('Castle', 12)), discards_left=3)
        after = advance_projected_discard(state, [Card(2, 'S')])
        self.assertEqual(after.jokers, state.jokers)

    def test_yorick_counter_is_read_only_with_explicit_remaining_information(self):
        j = parse_joker_tooltip(Joker('Yorick'),
            ['Yorick', 'This Joker gains X1 Mult every 23 [2]', 'cards discarded', '(Currently X2 Mult)'])
        self.assertEqual(j.value, 2)
        self.assertEqual(j.get('discards_remaining'), 2)
        ambiguous = parse_joker_tooltip(Joker('Yorick'), ['Yorick', 'every 23 cards discarded', 'Currently X2 Mult'])
        self.assertNotIn('discards_remaining', dict(ambiguous.data))

    def test_trading_card_triggers_on_first_discard_even_after_playing(self):
        state = RunState(jokers=(Joker('Trading Card'), Joker('Bull')), money=10,
                         boss_first_hand_done=True, round_discards_used=0, discards_left=3)
        first = advance_projected_discard(state, [Card(2, 'S')])
        self.assertEqual(first.money, 13)
        self.assertEqual(advance_projected_discard(first, [Card(3, 'S')]).money, 13)


class PersistentGrowthTests(unittest.TestCase):
    def test_vampire_wee_hiker_and_red_seal_persist_for_next_play(self):
        card = Card(2, 'H', enhancement='mult', seal='red')
        other = Card(8, 'S')
        state = RunState(deck_cards=(card, other), deck_size=2,
                         jokers=(Joker('Vampire', 1.4), Joker('Wee Joker', 8), Joker('Hiker')))
        after = advance_projected_play(state, [card], (0,), score_hand([card], state))
        self.assertAlmostEqual(after.jokers[0].value, 1.5)
        self.assertEqual(after.jokers[1].value, 24)
        self.assertEqual(after.deck_cards[0].enhancement, '')
        self.assertEqual(after.deck_cards[0].bonus_chips, 10)
        self.assertEqual(after.deck_cards[0].seal, 'red')
        self.assertEqual(state.deck_cards[0], card)
        self.assertEqual(_standard_unseen([], after), [other])

    def test_midas_vampire_order_changes_the_persistent_enhancement(self):
        king = Card(13, 'H')
        state = RunState(deck_cards=(king,), deck_size=1,
                         jokers=(Joker('Midas Mask'), Joker('Vampire', 1)))
        after = advance_projected_play(state, [king], (0,), score_hand([king], state))
        self.assertAlmostEqual(after.jokers[1].value, 1.1)
        self.assertEqual(after.deck_cards[0].enhancement, '')
        reverse = replace(state, jokers=tuple(reversed(state.jokers)))
        after = advance_projected_play(reverse, [king], (0,), score_hand([king], reverse))
        self.assertEqual(after.jokers[0].value, 1)
        self.assertEqual(after.deck_cards[0].enhancement, 'gold')

    def test_canio_values_destroying_a_face_card_without_changing_observed_state(self):
        hand = [Card(13, 'H'), Card(2, 'S')]
        state = RunState(jokers=(Joker('Canio', 2),), deck_cards=tuple(hand), deck_size=2)
        after = transformed_state(state, hand, [hand[1]])
        self.assertEqual(after.jokers[0].value, 3)
        self.assertEqual(state.jokers[0].value, 2)

    def test_tarot_grows_one_physical_fortune_teller_and_copier_reads_it(self):
        state = RunState(jokers=(Joker('Blueprint'), Joker('Fortune Teller', 4)), consumables_held=1)
        after = state_after_consumable(state, [Card(2, 'H')], Consumable('The Magician', 'Tarot'),
                                      ConsumableDecision('use', (0,)))
        self.assertEqual(after.jokers[0].value, 0)
        self.assertEqual(after.jokers[1].value, 5)
        self.assertEqual(after.effect_count('fortune_teller'), 2)
        self.assertEqual(after.consumables_held, 0)

    def test_planet_grows_constellation_and_levels_but_black_hole_does_not_grow_it(self):
        state = RunState(jokers=(Joker('Constellation', 1.2),), planet_cards={'pair': 1})
        after = consumed_engine(state, Consumable('Mercury', 'Planet'))
        self.assertAlmostEqual(after.jokers[0].value, 1.3)
        self.assertEqual(after.level('pair'), 2)
        self.assertEqual(after.planet_cards['pair'], 0)
        black_hole = consumed_engine(state, Consumable('Black Hole', 'Spectral'))
        self.assertEqual(black_hole.jokers[0].value, 1.2)

    def test_hermit_can_save_a_bull_hand_before_reaching_the_usual_ten_dollar_gain(self):
        state = RunState(jokers=(Joker('Bull'),), money=4, target_score=30,
                         hands_left=1, discards_left=0, consumables_held=1)
        hand = [Card(14, 'S')]
        self.assertEqual(score_hand(hand, state).score, 24)
        choice = decide_consumable(Consumable('The Hermit', 'Tarot'), hand, state)
        self.assertEqual(choice.action, 'use')
        self.assertEqual(choice.expected_score, 32)


class FragileEngineTests(unittest.TestCase):
    def test_single_banana_loss_is_a_future_risk_not_an_observed_destruction(self):
        state = RunState(ante=4, blind_index=0, money=30, jokers=(Joker('Gros Michel', 15),))
        before = forecast(safe=12000)
        failed = forecast(safe=5000)
        with patch('planning._FORECAST_CACHE', OrderedDict()), \
                patch('planning.forecast_round', side_effect=[before.next_round, failed.next_round]):
            result = shop_forecast(state)
        probability = 1 - (5 / 6) ** 3
        self.assertAlmostEqual(result.fragile_loss_probability, probability)
        self.assertAlmostEqual(result.sustainable_score, 12000 * (1 - probability) + 5000 * probability)
        self.assertTrue(result.weak)
        self.assertEqual(state.jokers, (Joker('Gros Michel', 15),))


if __name__ == '__main__':
    unittest.main()
