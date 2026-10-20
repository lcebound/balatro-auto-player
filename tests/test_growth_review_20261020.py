"""Physical destructive growth, verified counters and final-shop economics."""
import json
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from joker_cache import JokerCache, round_token, dirty_hand_action, dirty_jokers
from perception import read_text, joker_tooltip_words, verify_current_multiplier
from planning import advance_projected_madness, forecast_round, _round_state
from recovery import state_from_dict
from shop_extras import extra_is_worth_considering
from strategy import Joker, RunState, ShopItem, _shop_round_state, decide_shop
from tests.test_speed_and_policy_20261005 import forecast
from tools.replay_observation_speed import load_frame
from tools.review_growth_20261015 import items_from_event

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / 'tests/fixtures/growth_review_20261020.json').read_text('utf8'))


def historical(name):
    event = next(c['event'] for c in DATA['shops'] if c['name'] == name)
    return event, state_from_dict(event['state']), items_from_event(event)


class DestructiveGrowthTests(unittest.TestCase):
    def opened(self, jokers, sample=0, **kwargs):
        state = _shop_round_state(RunState(ante=4, blind_index=2, jokers=tuple(jokers), **kwargs))
        return state, advance_projected_madness(state, sample)

    def test_each_branch_removes_one_real_candidate_and_never_pays_sale_income(self):
        jokers = [Joker('Madness', 2), Joker('Joker', sell_value=10), Joker('Card Sharp', sell_value=20),
                  Joker('Campfire', 1.5), Joker('Runner', 60)]
        victims = set()
        for branch in range(4):
            before, after = self.opened(jokers, branch, money=9)
            victims.add((set(j.key for j in before.jokers) - set(j.key for j in after.jokers)).pop())
            self.assertEqual(len(after.jokers), 4)
            self.assertEqual(after.money, 9)
            self.assertEqual(next(j.value for j in after.jokers if j.key == 'madness'), 2.5)
            self.assertTrue(all(j.value == 1.5 for j in after.jokers if j.key == 'campfire'))
            self.assertEqual(len(before.jokers), 5)
        self.assertEqual(victims, {'joker', 'card_sharp', 'campfire', 'runner'})

    def test_eternal_targets_are_protected_but_empty_roster_still_grows(self):
        for others in ([], [Joker('Joker', data=(('eternal', 1),))]):
            before, after = self.opened([Joker('Madness', 1)] + others)
            self.assertEqual(len(after.jokers), len(before.jokers))
            self.assertEqual(after.jokers[0].value, 1.5)

    def test_boss_even_when_unknown_or_disabled_does_not_trigger_growth_or_destruction(self):
        before, _ = self.opened([Joker('Madness', 2), Joker('Joker')])
        for state in (replace(before, blind_index=1), replace(before, boss='The Club'),
                      replace(before, boss='The Club', boss_disabled=True)):
            self.assertEqual(advance_projected_madness(state), state)

    def test_copies_do_not_duplicate_growth_or_destruction(self):
        before, after = self.opened([Joker('Blueprint', data=(('eternal', 1),)), Joker('Madness', 2),
                                     Joker('Joker', data=(('eternal', 1),))])
        self.assertEqual(len(after.jokers), 3)
        self.assertEqual(after.jokers[1].value, 2.5)

    def test_another_madness_can_be_destroyed_and_cannot_act_after_removal(self):
        _, after = self.opened([Joker('Madness', 1), Joker('Madness', 4), Joker('Runner', 20)])
        self.assertEqual([j.key for j in after.jokers], ['madness', 'runner'])
        self.assertEqual(after.jokers[0].value, 1.5)

    def test_debuffed_madness_does_not_act_but_is_a_legal_victim(self):
        before, after = self.opened([Joker('Madness', 2, debuffed=True), Joker('Joker')])
        self.assertEqual(after, before)
        _, after = self.opened([Joker('Madness', 2), Joker('Madness', 2, debuffed=True), Joker('Joker')])
        self.assertEqual(len(after.jokers), 2)
        self.assertTrue(all(not j.debuffed for j in after.jokers))

    def test_destroyed_capacity_and_resource_cards_are_not_kept_in_the_forecast(self):
        _, after = self.opened([Joker('Madness', 2), Joker('Stuntman')], hand_size=6)
        self.assertEqual(after.hand_size, 8)
        before, after = self.opened([Joker('Madness', 2), Joker('Burglar')])
        self.assertEqual((before.hands_left, before.discards_left), (7, 0))
        self.assertEqual((after.hands_left, after.discards_left), (4, 0))
        _, after = self.opened([Joker('Madness', 2), Joker('Joker', edition='negative')], total_joker_slots=6)
        self.assertEqual(after.total_joker_slots, 5)

    def test_round_forecast_uses_same_six_draw_samples_and_six_destruction_branches(self):
        state = RunState(ante=4, blind_index=2, jokers=(Joker('Madness', 2), Joker('Joker')))
        with patch('planning.advance_projected_madness', wraps=advance_projected_madness) as opening:
            result = forecast_round(state, target=2000, samples=6)
        self.assertEqual(len(result.scores), 6)
        self.assertEqual(opening.call_count, 6)
        self.assertEqual([c.args[1] for c in opening.call_args_list], list(range(6)))


class MadnessCounterTests(unittest.TestCase):
    def test_retained_newly_proven_values_agree_with_visible_current_effect_not_growth_rate(self):
        recovered = 0
        for case in DATA['counters']:
            if case['event']['joker']['name'] != 'Madness':
                continue
            index = case['index']
            expected = (2 if index < 30828 else 2.5 if index <= 31065 else
                        3 if index <= 31403 else 3.5 if index == 31821 else 4)
            image = load_frame(ROOT / case['proof']['path'])
            words, _ = joker_tooltip_words(read_text(image, (525, 330, 2020, 850)), case['x'])
            value = verify_current_multiplier(image, words, Joker('Madness'), state_from_dict(case['state']))
            with self.subTest(index=index):
                self.assertLessEqual(value.value, expected)
                if value.get('observed_value'):
                    self.assertEqual(value.value, expected)
                    recovered += 1
                else:
                    self.assertTrue(value.get('counter_lower_bound'))
        self.assertGreaterEqual(recovered, 12)

    def test_unreadable_growth_keeps_only_a_proven_same_copy_lower_bound(self):
        old = Joker('Madness', 3, data=(('observed_value', 1),))
        result = verify_current_multiplier(np.zeros((1191, 2064, 3), np.uint8), [],
                                           Joker('Madness'), RunState(ante=5, blind_index=0, total_hands_played=30), old)
        self.assertEqual(result.value, 3)
        self.assertFalse(result.get('observed_value'))
        self.assertTrue(result.get('counter_lower_bound'))

    def test_physical_upper_bound_accounts_for_ante_reducing_vouchers(self):
        words = [('Currently X7 Mult', .99, (1000, 500))]
        state = RunState(ante=5, blind_index=1, total_hands_played=40)
        with patch('perception.read_text', return_value=words) as read:
            rejected = verify_current_multiplier(np.zeros((1191, 2064, 3), np.uint8), words, Joker('Madness'), state)
            accepted = verify_current_multiplier(np.zeros((1191, 2064, 3), np.uint8), words,
                Joker('Madness'), replace(state, owned_vouchers=('hieroglyph', 'petroglyph')))
        self.assertEqual(read.call_count, 8)
        self.assertEqual(rejected.value, 1)
        self.assertEqual(accepted.value, 7)

    def test_conflicting_current_values_never_use_a_majority_as_a_proof(self):
        responses = [[('Currently X2 Mult', .99, (1000, 500))]] * 3 + [[('Currently X2.5 Mult', .99, (1000, 500))]]
        with patch('perception.read_text', side_effect=responses):
            result = verify_current_multiplier(np.zeros((1191, 2064, 3), np.uint8), responses[0],
                Joker('Madness'), RunState(ante=5, blind_index=0, total_hands_played=30))
        self.assertEqual(result.value, 1)
        self.assertFalse(result.get('observed_value'))


class MadnessCacheTests(unittest.TestCase):
    def test_confirmed_hand_actions_keep_counter_but_blind_change_and_transactions_refresh(self):
        state = RunState(ante=5, blind_index=0, jokers=(Joker('Madness', 2),))
        session = SimpleNamespace(joker_cache=JokerCache(round_token(state), 5, state.jokers, (), frozenset()))
        for kind in ('play', 'discard'):
            dirty_hand_action(session, kind)
            self.assertFalse(session.joker_cache.needs_tooltip(0, state))
        self.assertTrue(session.joker_cache.needs_tooltip(0, replace(state, blind_index=1)))
        dirty_jokers(session, {'madness'})
        self.assertTrue(session.joker_cache.needs_tooltip(0, state))


class FinalShopTests(unittest.TestCase):
    def test_actual_final_boss_gold_cash_arrives_too_late(self):
        event, state, items = historical('last_boss_gold')
        result = decide_shop(items, state, reroll_cost=event['reroll_cost'])
        self.assertFalse(result.action == 'buy' and items[result.item_index].name == 'The Devil')

    def test_prior_round_and_known_use_or_scoring_synergies_remain_eligible(self):
        event, state, _ = historical('last_boss_gold')
        cases = [replace(state, ante=7)]
        cases += [replace(state, jokers=state.jokers[:-1] + (Joker(name),))
                  for name in ('Vampire', 'Golden Ticket', "Driver's License", 'Fortune Teller')]
        item = ShopItem('tarot', 'The Devil', 0)
        for state in cases:
            with self.subTest(ante=state.ante, joker=state.jokers[-1].name), \
                 patch('planning.shop_forecast', return_value=forecast(safe=500000, target=300000)):
                self.assertEqual(decide_shop([item], state).action, 'buy')

    def test_known_holographic_income_joker_still_counts_as_immediate_scoring(self):
        event, state, items = historical('holographic_egg')
        result = decide_shop(items, state, reroll_cost=event['reroll_cost'])
        self.assertEqual((result.action, result.item_index), ('buy', 0))

    def test_final_lethal_pack_can_spend_the_reserve_without_inventing_contents(self):
        weak = forecast(safe=100000, target=300000)
        state = RunState(ante=8, blind_index=1, money=4, jokers=(Joker('Joker'),))
        for kind in ('celestial_pack', 'arcana_pack'):
            item = ShopItem(kind, 'Celestial Pack' if kind == 'celestial_pack' else 'Arcana Pack', 4)
            self.assertTrue(extra_is_worth_considering(item, state, weak))
            self.assertFalse(extra_is_worth_considering(item, replace(state, money=3), weak))
            self.assertFalse(extra_is_worth_considering(item, replace(state, ante=7), weak))
            self.assertFalse(extra_is_worth_considering(item, state, forecast(safe=600000, target=300000)))
            with patch('planning.shop_forecast', return_value=weak):
                decision = decide_shop([item], state, reroll_cost=5)
            self.assertEqual(decision.action, 'buy')

    def test_a_known_rescue_beats_unknown_pack_contents(self):
        state = RunState(ante=8, blind_index=1, money=10, jokers=(Joker('Joker'),))
        before, rescue = forecast(safe=100000, target=300000), forecast(safe=400000, target=300000)
        items = [ShopItem('arcana_pack', 'Arcana Pack', 4), ShopItem('joker', 'Card Sharp', 6)]
        with patch('planning.shop_forecast', side_effect=lambda s: rescue if s.has('card_sharp') else before):
            decision = decide_shop(items, state, reroll_cost=5)
        self.assertEqual((decision.action, decision.item_index), ('buy', 1))


if __name__ == '__main__':
    unittest.main()
