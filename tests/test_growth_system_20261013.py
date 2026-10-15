"""Ante-8 economics, safe triggers, training and actual scoring support."""
import unittest
from dataclasses import asdict, replace
from unittest.mock import patch

from economy import (investment_rounds, interest, delayed_income, discard_income,
                     joker_income, economy_value, safe_income_discard)
from engine_development import development_score
from perception import parse_joker_tooltip
from planning import (advance_projected_discard, advance_projected_play,
                      engine_profile, growth_opportunity, shop_forecast)
from strategy import (Card, Joker, RunState, ShopItem, HAND_RANK, _discard_candidates,
                      decide_play, decide_discard, decide_shop, score_hand, _money_value)
from tests.test_speed_and_policy_20261005 import forecast


def rebate(rank=7):
    return Joker('Mail-In Rebate', data=(('rank', rank),))


class IncomeTests(unittest.TestCase):
    def test_horizon_stops_before_final_boss_cashout(self):
        for index, expected in ((0, 1), (1, 0), (2, 0)):
            self.assertEqual(investment_rounds(RunState(ante=8, blind_index=index, upcoming_boss_ante=8)), expected)
        self.assertEqual(investment_rounds(RunState(ante=1)), 6)
        self.assertEqual(investment_rounds(RunState(ante=8, blind_index=2, upcoming_boss_ante=7)), 2)

    def test_interest_caps_and_green_deck(self):
        state = RunState(money=48, interest_cash_cap=50)
        self.assertEqual(interest(state), 9)
        self.assertEqual(interest(replace(state, money=90)), 10)
        self.assertEqual(interest(replace(state, deck_type='Green Deck')), 0)
        self.assertGreater(_money_value(50, 2, state) - _money_value(46, 2, state), 4)

    def test_delayed_gratification_red_deck_four_discards_once_at_end(self):
        state = RunState(jokers=(Joker('Delayed Gratification'),), discards_left=4,
                         base_discards_per_round=4, round_discards_used=0)
        self.assertEqual(delayed_income(state), 8)
        self.assertEqual(delayed_income(replace(state, discards_left=3, round_discards_used=1)), 0)
        self.assertEqual(score_hand([Card(14, 'S')], state).cash, 0)
        self.assertEqual(delayed_income(replace(state, discards_left=0)), 0)

    def test_rebate_copies_only_known_live_rank_and_no_stone(self):
        state = RunState(jokers=(Joker('Blueprint'), rebate()), money=10)
        cards = [Card(7, 'S'), Card(7, 'H'), Card(7, 'D', debuffed=True),
                 Card(7, 'C', enhancement='stone'), Card(7, 'S', face_down=True)]
        self.assertEqual(discard_income(cards, state), 20)
        self.assertEqual(advance_projected_discard(state, cards).money, 30)
        self.assertEqual(discard_income(cards, replace(state, jokers=(Joker('Mail-In Rebate'),))), 0)

    def test_trading_first_single_only_noncopyable(self):
        state = RunState(jokers=(Joker('Blueprint'), Joker('Trading Card')), round_discards_used=0)
        self.assertEqual(discard_income([Card(2, 'S')], state), 3)
        self.assertEqual(discard_income([Card(2, 'S')]*2, state), 0)
        self.assertEqual(discard_income([Card(2, 'S')], replace(state, round_discards_used=1)), 0)

    def test_economy_uses_real_support_and_liquid_vs_sale_growth(self):
        f = forecast(safe=20000)
        zero = RunState(deck_cards=(Card(2, 'S'),)*20, money=10)
        self.assertEqual(joker_income(Joker('Cloud 9'), zero, f), 0)
        self.assertEqual(joker_income(Joker('Golden Ticket'), zero, f), 0)
        self.assertEqual(joker_income(Joker('Egg'), zero, f), 0)
        self.assertEqual(joker_income(Joker('Satellite'), zero, f), 0)
        self.assertEqual(joker_income(Joker('Golden Joker', debuffed=True), zero, f), 0)
        self.assertEqual(joker_income(Joker('Cloud 9', data=(('income', 7),)), zero, f), 7)

    def test_random_next_target_is_not_todays_profitable_target(self):
        state = RunState(ante=2, money=10)
        a = Joker('To Do List', data=(('hand', HAND_RANK['pair']),))
        b = replace(a, data=(('hand', HAND_RANK['straight_flush']),))
        self.assertEqual(joker_income(a, state, forecast()), joker_income(b, state, forecast()))
        self.assertEqual(joker_income(rebate(7), state, forecast()), joker_income(rebate(14), state, forecast()))

    def test_late_money_has_no_future_investment_value(self):
        state = RunState(ante=8, blind_index=1, jokers=(Joker('Golden Joker'),))
        self.assertEqual(economy_value(state, forecast()), 0)
        early = replace(state, ante=1, blind_index=0, money=10)
        self.assertGreater(economy_value(early, forecast()), economy_value(replace(early, money=70), forecast()))


class TooltipTests(unittest.TestCase):
    def test_rebate_anchor_recognizes_number_and_face(self):
        for title, rank in (('7', 7), ('Ace', 14), ('King', 13), ('10', 10)):
            j = parse_joker_tooltip(Joker('Mail-In Rebate'),
                [f'Earn $5 for each discarded {title}, rank changes every round'])
            self.assertEqual(j.get('rank'), rank)

    def test_unreadable_or_conflicting_new_target_clears_old_target(self):
        for text in ('Earn $5 rank changes every round',
                     'discarded 7, rank discarded Ace, rank'):
            self.assertEqual(parse_joker_tooltip(rebate(), [text]).get('rank', -1), -1)
        todo = Joker('To Do List', data=(('hand', 1),))
        self.assertEqual(parse_joker_tooltip(todo, ['changes every round']).get('hand', -1), -1)

    def test_scoped_income_does_not_confuse_growth_with_current_payout(self):
        rocket = parse_joker_tooltip(Joker('Rocket'),
            ['Earn $7 at end of round', 'Payout increases by $2 when Boss Blind is defeated'])
        self.assertEqual(rocket.get('income'), 7)
        cloud = parse_joker_tooltip(Joker('Cloud 9'), ['Earn $1 for each 9', '(Currently $0)'])
        self.assertEqual(cloud.get('income', -1), 0)
        self.assertEqual(parse_joker_tooltip(cloud, ['Earn $1 for each 9']).get('income', -1), -1)


class TriggerTests(unittest.TestCase):
    def setUp(self):
        self.hand = [Card(14, 'S'), Card(7, 'H'), Card(3, 'C'), Card(2, 'D')]
        self.state = RunState(jokers=(Joker('Joker', 4), rebate()), target_score=60,
                              objective='survival', round_discards_used=0)
        self.current = decide_play(self.hand, self.state, lookahead=False)

    def test_rebate_discards_spare_rank_and_keeps_known_clear(self):
        d = decide_discard(self.hand, self.state, current=self.current)
        self.assertEqual(d.indices, (1,))
        self.assertIn('retained conservative clear', d.reason)
        self.assertGreaterEqual(d.expected_score, 66)

    def test_draw_candidates_include_all_target_ranks(self):
        candidates = _discard_candidates(self.hand, self.state)
        self.assertTrue(any(1 not in keep for keep in candidates))

    def test_farming_never_spends_the_only_known_scoring_card(self):
        state = replace(self.state, jokers=(Joker('Joker', 4), rebate(14)))
        current = decide_play(self.hand, state, lookahead=False)
        self.assertIsNone(safe_income_discard(self.hand, state, current))

    def test_delayed_payout_and_special_cards_are_preserved(self):
        state = replace(self.state, discards_left=4, base_discards_per_round=4,
                        jokers=self.state.jokers + (Joker('Delayed Gratification'),))
        self.assertIsNone(safe_income_discard(self.hand, state, self.current))
        for changes in ({'seal': 'blue'}, {'enhancement': 'gold'}, {'enhancement': 'steel'}):
            hand = list(self.hand)
            hand[1] = replace(hand[1], **changes)
            self.assertIsNone(safe_income_discard(hand, self.state, self.current))

    def test_uncertain_draw_can_disable_held_multiplier(self):
        for joker in ('Blackboard', 'Raised Fist'):
            state = replace(self.state, jokers=self.state.jokers + (Joker(joker),))
            self.assertIsNone(safe_income_discard(self.hand, state, self.current))

    def test_final_hand_can_take_guaranteed_rebate_but_not_unknown_boss(self):
        state = replace(self.state, hands_left=1)
        self.assertIsNotNone(safe_income_discard(self.hand, state, self.current))
        for boss in ('The Hook', 'The Water', 'The Fish'):
            self.assertIsNone(safe_income_discard(self.hand, replace(state, boss=boss), self.current))

    def test_safe_to_do_list_clear_beats_unprofitable_overshoot(self):
        hand = [Card(14, 'S'), Card(14, 'H'), Card(8, 'D'), Card(8, 'C')]
        state = RunState(objective='survival', target_score=50,
            jokers=(Joker('Joker', 4), Joker('To Do List', data=(('hand', HAND_RANK['high_card']),))))
        decision = decide_play(hand, state, lookahead=False)
        self.assertEqual(decision.estimate.category, 'high_card')
        self.assertEqual(decision.estimate.cash, 4)
        self.assertGreaterEqual(decision.estimate.score, state.score_needed)


class TrainingAndShopTests(unittest.TestCase):
    def setUp(self):
        self.hand = [Card(r, s) for r, s in ((14,'S'),(14,'H'),(8,'D'),(8,'C'),
                                          (2,'H'),(3,'D'),(4,'C'),(7,'S'))]
        self.state = RunState(jokers=(Joker('Ride the Bus',2), Joker('Joker',4)),
                              target_score=150, objective='survival', ante=1)

    def test_one_early_training_play_retains_a_known_finisher(self):
        original = asdict(self.state)
        d = decide_play(self.hand, self.state)
        self.assertTrue(d.training)
        self.assertFalse(decide_discard(self.hand, self.state, current=d).indices)
        after = advance_projected_play(self.state, self.hand, d.indices, d.estimate)
        after = replace(after, round_score=d.estimate.score, draw_pile_size=0)
        held = [c for i,c in enumerate(self.hand) if i not in d.indices]
        finish = decide_play(held, after)
        self.assertFalse(finish.training)
        self.assertGreaterEqual(finish.estimate.score, after.score_needed)
        self.assertEqual(asdict(self.state), original)

    def test_no_training_late_boss_last_two_hands_or_repeated(self):
        for changes in ({'ante': 5}, {'hands_left': 2}, {'boss': 'The Needle'},
                        {'round_played_types': {'high_card': 1}}, {'forced_card_index': 0}):
            self.assertFalse(decide_play(self.hand, replace(self.state, **changes)).training)

    def test_no_training_if_known_cards_cannot_finish(self):
        self.assertFalse(decide_play([Card(14,'S')], replace(self.state, target_score=60)).training)

    def test_early_affordable_income_pays_for_itself_with_actual_forecast(self):
        state = RunState(jokers=(Joker('Joker',4),), money=10, ante=1,
                         blind_index=0, base_discards_per_round=4, objective='survival')
        item = ShopItem('joker', 'Golden Joker', 4, Joker('Golden Joker'))
        self.assertEqual(decide_shop([item], state).action, 'buy')

    def test_no_support_cloud9_and_final_economy_are_not_bought(self):
        f = forecast(safe=20000)
        state = RunState(jokers=(Joker('Joker',4),), money=10,
                         deck_cards=(Card(2,'S'),)*20)
        with patch('planning.shop_forecast', return_value=f):
            item = ShopItem('joker', 'Cloud 9', 5, Joker('Cloud 9'))
            self.assertEqual(decide_shop([item], state).action, 'leave')
            item = ShopItem('joker', 'Golden Joker', 4, Joker('Golden Joker'))
            self.assertEqual(decide_shop([item], replace(state, ante=8, blind_index=1)).action, 'leave')

    def test_economy_cannot_rescue_lethal_blind_or_sell_scorer(self):
        item = ShopItem('joker', 'Golden Joker', 4, Joker('Golden Joker'))
        state = RunState(money=20, jokers=(Joker('Joker',4),), total_joker_slots=1)
        with patch('planning.shop_forecast', return_value=forecast(safe=3000)):
            self.assertEqual(decide_shop([item], state).action, 'leave')

    def test_negative_training_gets_no_growth_label_bonus(self):
        f = forecast(safe=20000)
        f = replace(f, next_round=replace(f.next_round, growth_per_round=(('ride_the_bus',-5),)))
        self.assertEqual(growth_opportunity(Joker('Ride the Bus',10), self.state, f), 0)

    def test_egg_swash_growth_is_scoring_and_not_cash(self):
        state = RunState(ante=2, jokers=(Joker('Egg', sell_value=5), Joker('Swashbuckler')))
        self.assertGreater(development_score(state, forecast()), forecast().sustainable_score)
        self.assertNotIn('no_scaling', engine_profile(replace(state, ante=4)).weaknesses)
        observed = replace(state, jokers=(state.jokers[0], Joker('Swashbuckler', 5, data=(('observed_value',1),))))
        self.assertGreater(development_score(observed, forecast()), forecast().sustainable_score)

    def test_inactive_x1_and_empty_retriggers_do_not_hide_missing_multiplier(self):
        for name in ('Canio', 'Madness', 'Campfire', 'Drivers License', 'Hanging Chad', 'Mime'):
            state = RunState(ante=4, jokers=(Joker(name,1),))
            self.assertIn('no_multiplier_or_retrigger', engine_profile(state).weaknesses)
        state = RunState(ante=4, jokers=(Joker('Photograph'), Joker('Hanging Chad')))
        self.assertNotIn('no_multiplier_or_retrigger', engine_profile(state).weaknesses)
        license_state = RunState(ante=4, jokers=(Joker('Drivers License', data=(('enhanced_count',16),)),))
        self.assertNotIn('no_multiplier_or_retrigger', engine_profile(license_state).weaknesses)

    def test_retrigger_needs_matching_played_or_held_source(self):
        state = RunState(ante=4, jokers=(Joker('Mime'),),
                         deck_cards=(Card(2,'S', enhancement='glass'),))
        self.assertIn('no_multiplier_or_retrigger', engine_profile(state).weaknesses)
        state = replace(state, deck_cards=(Card(2,'S', enhancement='steel'),))
        self.assertNotIn('no_multiplier_or_retrigger', engine_profile(state).weaknesses)


if __name__ == '__main__':
    unittest.main()
