"""Observed scoring/engine tradeoffs and replacement purchase boundaries."""
import json
import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from joker_cache import JokerCache, dirty_hand_action, round_token
from perception import current_fortune_mult, parse_joker_tooltip, verify_current_multiplier
from planning import engine_profile, replacement_reduces_survival, deck_target_value
from recovery import ObservationError
from runner import shop
from strategy import Card, Joker, RunState, ShopItem, ShopDecision, decide_shop, score_hand
from tests.test_runtime_recovery import FakeSession
from tests.test_speed_and_policy_20261005 import forecast
from tools.replay_strategy_review_20261011 import corrected_state, replay_counter_pixels

INPUTS = json.loads((Path(__file__).parent / 'fixtures/strategy_review_20261011.json').read_text('utf-8'))


class ObservedCounterTests(unittest.TestCase):
    def test_three_actual_tooltip_images_confirm_current_counters(self):
        rows = replay_counter_pixels(INPUTS)
        self.assertEqual([r['value'] for r in rows], [1, 7, 11])
        self.assertTrue(all(r['verified'] and not r['bound'] for r in rows))

    def test_fortune_uses_current_total_not_growth_rate(self):
        for row in INPUTS['tooltips']:
            j = parse_joker_tooltip(Joker('Fortune Teller'), row['tooltip'])
            self.assertIn(j.value, (1, 7, 11))
            self.assertTrue(j.get('observed_value'))
        j = parse_joker_tooltip(Joker('Fortune Teller'), ['+1 Mult per Tarot', '(Currently +11)'])
        self.assertEqual(j.value, 11)
        self.assertIsNone(current_fortune_mult('+1 Mult per Tarot card used this run'))
        self.assertIsNone(current_fortune_mult('(Currently +7) (Currently +11)'))
        self.assertIsNone(current_fortune_mult('(Currently +1.1)'))

    def test_missing_or_conflicting_fortune_is_an_explicit_bound(self):
        for labels in (['+1 Mult per Tarot'], ['(Currently +7)', '(Currently +11)']):
            j = parse_joker_tooltip(Joker('Fortune Teller'), labels)
            self.assertEqual(j.value, 0)
            self.assertTrue(j.get('counter_lower_bound'))
            self.assertFalse(j.get('observed_value'))

    def test_independent_current_reads_confirm_total(self):
        image = np.zeros((1191, 2064, 3), dtype=np.uint8)
        words = [('(Currently +11)', .99, (670, 485))]
        with patch('perception.read_text', return_value=words):
            j = verify_current_multiplier(image, words, Joker('Fortune Teller'), RunState())
        self.assertEqual(j.value, 11)
        self.assertTrue(j.get('observed_value'))

    def test_counter_conflict_or_regression_preserves_verified_lower_bound(self):
        image = np.zeros((1191, 2064, 3), dtype=np.uint8)
        old = Joker('Fortune Teller', 7, data=(('observed_value', 1),))
        words = [('(Currently +11)', .99, (670, 485))]
        views = [[('(Currently +11)', .99, (670, 485))],
                 [('(Currently +12)', .99, (670, 485))]] * 2
        for responses in (views, [[('(Currently +3)', .99, (670, 485))]] * 4):
            with patch('perception.read_text', side_effect=responses):
                j = verify_current_multiplier(image, words, Joker('Fortune Teller'), RunState(), old)
            self.assertEqual(j.value, 7)
            self.assertTrue(j.get('counter_lower_bound'))

    def test_logged_straight_and_flush_match_independent_actual_scores(self):
        # +11 is read from the preceding tooltip, never fitted to these scores.
        for fragment, actual in (('08:08:48', 15015), ('08:09:40', 4408), ('07:44:23', 7830)):
            row = next(r for r in INPUTS['scores'] if fragment in r['time'])
            cards = [Card(**c) for c in row['cards']]
            result = score_hand([cards[i] for i in row['order']], corrected_state(row),
                                [c for i, c in enumerate(cards) if i not in row['order']])
            self.assertEqual(row['actual'], actual)
            self.assertEqual(result.score, actual)

    def test_to_do_list_category_is_exact_and_cash_is_conditional(self):
        j = parse_joker_tooltip(Joker('To Do List'), INPUTS['todo_offer']['tooltip'])
        pair = [Card(7, 'H'), Card(7, 'S')]
        state = RunState(jokers=(j,))
        self.assertEqual(score_hand(pair, state).cash, 4)
        self.assertEqual(score_hand(pair + [Card(8, 'H'), Card(8, 'S')], state).cash, 0)
        two = parse_joker_tooltip(Joker('To Do List'), ['is a Two Pair,'])
        self.assertNotEqual(j.get('hand'), two.get('hand'))
        self.assertEqual(parse_joker_tooltip(Joker('To Do List'), ['poker hand changes']).get('hand', -1), -1)

    def test_to_do_list_target_does_not_refresh_per_play(self):
        state = RunState(jokers=(Joker('To Do List'),))
        session = type('Session', (), {})()
        session.joker_cache = JokerCache(round_token(state), 5, state.jokers, (), frozenset())
        dirty_hand_action(session, 'play')
        self.assertEqual(session.joker_cache.dirty, frozenset())


class EngineRiskTests(unittest.TestCase):
    def test_developed_straight_deck_values_real_middle_rank_routes(self):
        state = RunState(levels={'straight': 6})
        self.assertGreater(deck_target_value(Card(7, 'H'), state), deck_target_value(Card(14, 'H'), state))
        # Face-card multipliers can make an endpoint worth keeping.
        photo = replace(state, jokers=(Joker('Photograph'),))
        self.assertGreater(deck_target_value(Card(13, 'H'), photo), deck_target_value(Card(7, 'H'), photo))

    def test_missing_straight_neighbours_and_no_investment_disable_middle_bias(self):
        state = RunState(levels={'straight': 6}, deck_cards=tuple(Card(r, 'H') for r in (7, 10, 11, 12, 13, 14)))
        self.assertGreater(deck_target_value(Card(14, 'H'), state), deck_target_value(Card(7, 'H'), state))
        self.assertGreater(deck_target_value(Card(14, 'H'), RunState()), deck_target_value(Card(7, 'H'), RunState()))

    def test_recorded_troubadour_swap_cannot_sacrifice_next_blind_for_boss(self):
        old = forecast(safe=57981, target=52500, sustainable=47044, boss_safe=31095)
        old = replace(old, next_round=replace(old.next_round, clear_rate=5/6),
                      boss_round=replace(old.boss_round, target=70000))
        new = forecast(safe=35544, target=52500, boss_safe=38532)
        new = replace(new, next_round=replace(new.next_round, clear_rate=.5),
                      boss_round=replace(new.boss_round, target=70000))
        self.assertGreater(new.coverage, old.coverage)
        self.assertTrue(replacement_reduces_survival(old, new))

    def test_unsafe_imminent_blind_still_protected_when_boss_is_worst(self):
        old = forecast(safe=9000, boss_safe=4000)
        new = forecast(safe=7000, boss_safe=7000)
        self.assertGreater(new.coverage, old.coverage)
        self.assertTrue(replacement_reduces_survival(old, new))

    def test_safe_headroom_can_pay_for_a_needed_boss_upgrade(self):
        old = forecast(safe=40000, boss_safe=30000)
        new = forecast(safe=25000, boss_safe=45000)
        self.assertFalse(replacement_reduces_survival(old, new))

    def test_unskipped_throwback_is_not_a_working_xmult(self):
        state = RunState(ante=4, jokers=(Joker('Throwback'),))
        self.assertEqual(engine_profile(state).multiplicative_mult, 0)
        self.assertEqual(engine_profile(replace(state, skipped_blinds=1)).multiplicative_mult, 1)
        poly = replace(state, jokers=(Joker('Throwback', edition='polychrome'),))
        self.assertEqual(engine_profile(poly).multiplicative_mult, 1)

    def test_search_budget_protects_purchase_cash_and_two_reroll_limit(self):
        state = RunState(ante=4, money=33, jokers=(Joker('Joker'),))
        f = forecast(safe=20000)
        with patch('planning.shop_forecast', return_value=f):
            self.assertEqual(decide_shop([], state, reroll_cost=5).action, 'reroll')
            self.assertEqual(decide_shop([], replace(state, money=30), reroll_cost=5).action, 'leave')
            self.assertEqual(decide_shop([], replace(state, shop_rerolls=2), reroll_cost=5).action, 'leave')
            self.assertEqual(decide_shop([], state, reroll_cost=8).action, 'leave')

    def test_cash_dependent_engine_keeps_full_reserve(self):
        state = RunState(ante=4, money=33, jokers=(Joker('Bull'),))
        with patch('planning.shop_forecast', return_value=forecast(safe=20000)):
            self.assertEqual(decide_shop([], state, reroll_cost=5).action, 'leave')

    def test_growing_xmult_engine_still_searches_when_chips_are_missing(self):
        state = RunState(ante=4, money=33, jokers=(Joker('Ride the Bus', 5), Joker('Blackboard')))
        f = replace(forecast(safe=6158, target=5000), profile=engine_profile(state))
        self.assertEqual(f.profile.weaknesses, ('low_chips',))
        pack = ShopItem('celestial_pack', 'Jumbo Celestial Pack', 6)
        with patch('planning.shop_forecast', return_value=f), \
                patch('shop_extras.sealed_pack_value', return_value=9.2):
            self.assertEqual(decide_shop([pack], state, reroll_cost=5).action, 'reroll')

    def test_marginal_sealed_pack_does_not_drain_missing_engine_search_budget(self):
        state = RunState(ante=6, money=16, jokers=(Joker('Joker'),))
        pack = ShopItem('celestial_pack', 'Celestial Pack', 4)
        with patch('planning.shop_forecast', return_value=forecast(safe=7000)), \
                patch('shop_extras.sealed_pack_value', return_value=8):
            self.assertEqual(decide_shop([pack], state, reroll_cost=5).action, 'reroll')
            self.assertEqual(decide_shop([pack], replace(state, shop_rerolls=2), reroll_cost=5).action, 'buy')
            self.assertEqual(decide_shop([pack], replace(state, owned_vouchers=('telescope',)), reroll_cost=5).action, 'buy')


class ReplacementExecutionTests(unittest.TestCase):
    def run_shop(self, *, selection_error=False, changed_roster=False):
        initial = RunState(money=13, jokers=tuple(Joker(n) for n in
                          ('Throwback', 'Ride the Bus', 'Gluttonous Joker', 'Blackboard', 'Wrathful Joker')))
        after = replace(initial, money=15, jokers=tuple(j for i, j in enumerate(initial.jokers) if i != 2))
        if changed_roster:
            after = replace(after, jokers=after.jokers[:-1])
        item = ShopItem('joker', 'To Do List', 4, Joker('To Do List'))
        bought = replace(after, money=11, jokers=after.jokers + (item.joker,))
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            session = FakeSession(directory)
            session.policy_state = initial
            values = {'read_numbers': {'money': 13}, 'consumable_counter': (0, 2),
                      'read_jokers': [], 'read_text': [('$4', .99, (1100, 460))],
                      'shop_offer_item': (item, 1.), 'collect_extra_offers': ([], []),
                      'shop_tooltip_region': (530, 330, 1800, 850),
                      'read_reroll_button': None, 'shop_strength': 0}
            for name, value in values.items():
                stack.enter_context(patch('runner.' + name, return_value=value))
            stack.enter_context(patch('runner.collect_shop_jokers', side_effect=[initial, bought]))
            stack.enter_context(patch('runner.sell_shop_joker', return_value=after))
            stack.enter_context(patch('runner.decide_shop', side_effect=[
                ShopDecision('sell_then_buy', 0, 2, 1, 'recorded replacement'), ShopDecision('leave')]))
            verify = stack.enter_context(patch('runner.verify_shop_selection',
                         side_effect=ObservationError('different offer') if selection_error else None,
                         return_value=((1100, 650), session.frame)))
            begin = stack.enter_context(patch('runner.begin_action'))
            stack.enter_context(patch('runner.confirm_pending'))
            stack.enter_context(patch('runner.record_issue'))
            stack.enter_context(patch('runner.time.sleep'))
            stack.enter_context(patch('shop_extras.shop_ready', return_value=True))
            shop(session, session.frame)
            return session, begin, verify

    def test_confirmed_sale_completes_original_purchase_despite_next_policy_leave(self):
        session, begin, verify = self.run_shop()
        begin.assert_called_once()
        self.assertEqual(begin.call_args.args[1], 'buy')
        verify.assert_called_once()
        self.assertTrue(session.policy_state.has('to_do_list'))
        self.assertTrue(any(e['event'] == 'replacement_purchase_continued' for e in session.events))

    def test_unverified_target_never_authorizes_buy(self):
        session, begin, verify = self.run_shop(selection_error=True)
        verify.assert_called_once()
        begin.assert_not_called()
        self.assertFalse(session.policy_state.has('to_do_list'))

    def test_unexpected_inventory_after_sale_cancels_bundle(self):
        session, begin, verify = self.run_shop(changed_roster=True)
        verify.assert_not_called()
        begin.assert_not_called()
        self.assertTrue(any(e['event'] == 'replacement_purchase_cancelled' for e in session.events))


if __name__ == '__main__':
    unittest.main()
