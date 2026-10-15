"""Offline strategy regressions from the screenshots of the October batch."""
import json
import unittest
import tempfile
from collections import Counter
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from consumables import (Consumable, decide_consumable, transformed_hand,
                         transformed_state, state_after_consumable, ConsumableDecision)
from controller import try_consumable
from tests.test_runtime_recovery import FakeSession
from perception import read_future_boss, parse_joker_tooltip, joker_from_definition
from planning import (shop_forecast, forecast_round, advance_projected_play,
                      deck_target_value, engine_profile)
from recovery import state_from_dict
from strategy import (Card, Joker, RunState, ShopItem, SUITS, decide_shop,
                      decide_play, score_hand, next_blind_target, _standard_unseen)
from strategy import _card_future_value


FIXTURES = Path(__file__).parent / 'fixtures'


def items_from_dict(items):
    result = []
    for item in items:
        item = dict(item)
        if item.get('joker'):
            item['joker'] = Joker(**dict(item['joker'], data=tuple(
                tuple(pair) for pair in item['joker'].get('data', ()))))
        item['pack_contents'] = tuple(item.get('pack_contents', ()))
        result.append(ShopItem(**item))
    return result


class StrategyDepthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = json.loads((FIXTURES / 'strategy_depth_20261004.json').read_text('utf8'))

    def test_recorded_opening_preserves_cash_instead_of_unused_mars_death(self):
        c = self.cases['shops']['4']
        decision = decide_shop(items_from_dict(c['items']), state_from_dict(c['state']), reroll_cost=5)
        self.assertEqual(decision.action, 'reroll')
        self.assertEqual(decide_shop([], replace(state_from_dict(c['state']), money=8), reroll_cost=5).action, 'leave')

    def test_recorded_stencil_can_replace_before_slots_are_full(self):
        c = self.cases['shops']['7']
        state = state_from_dict(c['state'])
        self.assertEqual((len(state.jokers), state.total_joker_slots), (4, 5))
        decision = decide_shop(items_from_dict(c['items']), state)
        self.assertEqual(decision.action, 'sell_then_buy')
        self.assertEqual(state.jokers[decision.sell_index].name, 'Greedy Joker')
        self.assertIn('coverage', decision.reason)

    def test_recorded_wall_is_detected_before_small_blind(self):
        path = FIXTURES / 'strategy_depth_20261004/early_wall.png'
        image = cv2.resize(cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR),
                           (2064, 1191), interpolation=cv2.INTER_NEAREST)
        self.assertEqual(read_future_boss(image), ('The Wall', 20000))

    def test_ambiguous_boss_column_does_not_invent_a_target(self):
        with patch('perception.read_text', return_value=[('The Wall', .99, (1500, 550)),
                    ('20,000', .99, (1500, 800)), ('2,000', .99, (1500, 880))]):
            self.assertIsNone(read_future_boss(None))

    def test_wall_changes_shop_forecast_and_spends_surplus_to_improve(self):
        c = self.cases['shops']['6']
        state = replace(state_from_dict(c['state']), blind_index=1,
                        upcoming_boss='The Wall', upcoming_boss_target=20000,
                        upcoming_boss_ante=4)
        forecast = shop_forecast(state)
        self.assertEqual(next_blind_target(state), 20000)
        self.assertLess(forecast.next_round.safe_score, 20000)
        self.assertLess(abs(forecast.next_round.mean_score - 14053), 14053 * .25)
        self.assertEqual(decide_shop([], state, reroll_cost=5).action, 'reroll')
        self.assertIn('no_multiplier_or_retrigger', forecast.profile.weaknesses)

    def test_wall_is_budgeted_in_shop_after_small_blind(self):
        state = RunState(ante=4, blind_index=0, target_score=5000,
                         upcoming_boss='The Wall', upcoming_boss_target=20000,
                         upcoming_boss_ante=4, jokers=(Joker('Gros Michel'),), money=50)
        forecast = shop_forecast(state)
        self.assertEqual((forecast.next_round.target, forecast.boss_round.target), (7500, 20000))
        self.assertTrue(forecast.weak)

    def test_new_ante_shop_does_not_skip_forecasting_small_blind(self):
        state = RunState(ante=2, blind_index=2, target_score=600, upcoming_boss_ante=1)
        self.assertEqual(next_blind_target(state), 800)
        self.assertEqual(next_blind_target(replace(state, ante=1, upcoming_boss_ante=1)), 800)

    def test_forecast_applies_water_and_needle_resources(self):
        state = RunState(base_hands_per_round=4, base_discards_per_round=4,
                         jokers=(Joker('Burglar'),))
        self.assertEqual(forecast_round(state, 'The Needle', 1000, samples=2).hands, 4)
        plain = replace(state, jokers=())
        self.assertEqual(forecast_round(plain, 'The Needle', 1000, samples=2).hands, 1)
        self.assertEqual(forecast_round(plain, 'The Water', 1000, samples=2).discards, 0)

    def test_seltzer_lifetime_and_ice_cream_decay_are_not_permanent(self):
        joker = parse_joker_tooltip(joker_from_definition('seltzer'),
                                   ['Seltzer', 'Retrigger all cards played', 'for the next 1 hands'])
        self.assertEqual(joker.get('remaining_hands'), 1)
        hand = [Card(14, 'S')]
        state = RunState(jokers=(joker, Joker('Ice Cream', 10, data=(('observed_value', 1),))))
        estimate = score_hand(hand, state)
        after = advance_projected_play(state, hand, (0,), estimate)
        self.assertFalse(after.has('seltzer'))
        self.assertEqual(after.jokers[0].value, 5)
        forecast = shop_forecast(RunState(jokers=(Joker('Ice Cream', 40,
                                              data=(('observed_value', 1),)),), money=30,
                                             base_discards_per_round=0))
        self.assertLess(forecast.sustainable_score, forecast.next_round.safe_score)

    def test_unreadable_seltzer_lifetime_does_not_assume_ten_more_hands(self):
        joker = parse_joker_tooltip(joker_from_definition('seltzer'),
                                   ['Seltzer', 'Retrigger all cards played'])
        self.assertEqual(joker.get('remaining_hands'), 1)
        self.assertEqual(joker.get('lifetime_observed'), 0)

    def test_round_projection_advances_scaling_without_mutating_observations(self):
        hand = [Card(14, 'S'), Card(14, 'H')]
        state = RunState(jokers=(Joker('Green Joker', 3), Joker('Ride the Bus', 4)),
                         played_types={'pair': 6}, hands_left=3)
        after = advance_projected_play(state, hand, (0, 1), score_hand(hand, state))
        self.assertEqual(after.played_types['pair'], 7)
        self.assertEqual(tuple(j.value for j in after.jokers), (4, 5))
        self.assertEqual(state.played_types['pair'], 6)

    def test_early_riff_raff_has_value_when_generated_jokers_have_space(self):
        state = RunState(money=9, ante=1)
        decision = decide_shop([ShopItem('joker', 'Riff-Raff', 4, Joker('Riff-Raff'))], state)
        self.assertEqual(decision.action, 'buy')

    def test_recorded_water_keeps_even_steven_card_instead_of_useless_kicker(self):
        c = self.cases['hands']['3'][0]
        state = state_from_dict(c['state'])
        hand = [Card(**card) for card in c['event']['card_details']]
        plan = decide_play(hand, state)
        self.assertEqual(plan.estimate.score, c['event']['expected_score'])
        self.assertNotIn(5, plan.indices)  # Retain 6H, which receives Even Steven.
        self.assertIn(3, plan.indices)     # Cycle 10D as the non-scoring kicker.
        self.assertIn('next hand', plan.reason)

    def test_final_hand_and_a_guaranteed_clear_keep_immediate_score_priority(self):
        hand = [Card(14, 'S'), Card(14, 'H'), Card(13, 'S'), Card(13, 'H'), Card(2, 'C')]
        state = RunState(target_score=50, hands_left=4)
        self.assertEqual(decide_play(hand, state).estimate.category, 'two_pair')
        final = replace(state, hands_left=1, target_score=1000)
        self.assertEqual(decide_play(hand, final), decide_play(hand, final, lookahead=False))

    def test_death_can_copy_left_steel_red_king_after_verified_order(self):
        king = Card(13, 'H', enhancement='steel', seal='red')
        hand = [king, Card(2, 'S'), Card(5, 'D'), Card(8, 'C')]
        state = RunState(jokers=(Joker('Baron'), Joker('Mime')), target_score=5000,
                         deck_cards=tuple(hand), deck_size=4, hands_left=2)
        plan = decide_consumable(Consumable('Death', 'Tarot'), hand, state)
        self.assertEqual(plan.action, 'use')
        self.assertEqual(plan.play_order, (1, 0))
        arranged = [hand[i] for i in plan.play_order] + hand[2:]
        after = transformed_hand(arranged, Consumable('Death', 'Tarot'), (0, 1))
        self.assertEqual(after[:2], [king, king])

    def test_large_hand_strength_includes_middle_queen_for_baron(self):
        hand = [Card(10, suit) for suit in SUITS] + [Card(11, suit) for suit in SUITS]
        hand += [Card(9, 'S'), Card(12, 'D'), Card(8, 'H'), Card(7, 'C'), Card(6, 'D')]
        state = RunState(jokers=(Joker('Baron'),), hand_size=13, target_score=50000)
        plan = decide_consumable(Consumable('Strength', 'Tarot'), hand, state)
        self.assertEqual(plan.action, 'use')
        self.assertIn(9, plan.indices)

    def test_deck_memory_preserves_death_duplicate_and_excludes_used_cards(self):
        king = Card(13, 'H', enhancement='steel', seal='red')
        deck = tuple(king if (rank, suit) == king.identity else Card(rank, suit)
                     for suit in SUITS for rank in range(2, 15))
        hand = [Card(2, 'S'), king]
        modified = transformed_hand(hand, Consumable('Death', 'Tarot'), (0, 1))
        state = transformed_state(RunState(deck_cards=deck), hand, modified)
        self.assertEqual(Counter(state.deck_cards)[king], 2)
        self.assertNotIn(Card(2, 'S'), state.deck_cards)
        unseen = _standard_unseen(modified, state)
        self.assertEqual(len(unseen), 50)
        self.assertNotIn(king, unseen)
        self.assertEqual(sum(c.rank == 13 for c in unseen), 3)

    def test_cryptid_updates_hologram_and_remembered_deck_size(self):
        king = Card(13, 'H', enhancement='steel')
        state = RunState(deck_cards=(king,), deck_size=1,
                         jokers=(Joker('Hologram', 1.5),), consumables_held=1)
        plan = decide_consumable(Consumable('Cryptid', 'Spectral'), [king], state)
        after = state_after_consumable(state, [king], Consumable('Cryptid', 'Spectral'), plan)
        self.assertEqual((after.deck_size, len(after.deck_cards)), (3, 3))
        self.assertEqual(after.jokers[0].value, 2)
        self.assertEqual(after.consumables_held, 0)

    def test_red_seal_steel_king_is_more_valuable_in_baron_mime_build(self):
        king = Card(13, 'H', enhancement='steel', seal='red')
        state = RunState(jokers=(Joker('Baron'), Joker('Mime')))
        self.assertGreater(deck_target_value(king, state), deck_target_value(king, RunState()) + 50)

    def test_cannot_buy_an_unsupported_consumable_that_would_stay_held(self):
        state = RunState(money=50, jokers=(Joker('Gros Michel'), Joker('Cavendish')))
        self.assertNotEqual(decide_shop([ShopItem('tarot', 'The Fool', 3)], state).action, 'buy')
        self.assertNotEqual(decide_shop([ShopItem('spectral', 'Ankh', 4)], state).action, 'buy')

    def test_stone_and_unknown_cards_do_not_inherit_rank_build_bonuses(self):
        stone = Card(13, 'H', enhancement='stone', rankless=True)
        self.assertEqual(_card_future_value(stone, RunState(jokers=(Joker('Baron'),))),
                         _card_future_value(stone, RunState()))
        hidden = Card(0, '?', face_down=True)
        self.assertEqual(_card_future_value(hidden, RunState(jokers=(Joker('Hack'),))), 0)

    def test_remembered_steel_and_stone_counts_affect_their_jokers(self):
        steel = Card(13, 'H', enhancement='steel')
        stone = Card(10, 'D', enhancement='stone', rankless=True)
        played = Card(2, 'S')
        state = RunState(deck_cards=(steel, steel, stone, played), deck_size=4,
                         jokers=(Joker('Steel Joker', 1, data=(('observed_value', 1),)),))
        plain = score_hand([played], replace(state, jokers=())).score
        self.assertAlmostEqual(score_hand([played], state).score, plain * 1.4)
        state = replace(state, jokers=(Joker('Stone Joker', 0, data=(('observed_value', 1),)),))
        self.assertAlmostEqual(score_hand([played], state).score, plain + 25)

    def test_transform_can_activate_drivers_license_before_the_next_ocr(self):
        deck = tuple(Card(rank, suit, enhancement='mult' if index < 15 else '')
                     for index, (rank, suit) in enumerate(
                         (rank, suit) for suit in SUITS for rank in range(2, 15)))
        hand = [deck[15]]
        state = RunState(deck_cards=deck, jokers=(Joker('Driver\'s License',
                        data=(('enhanced_count', 15),)),))
        modified = transformed_hand(hand, Consumable('The Empress', 'Tarot'), (0,))
        after = transformed_state(state, hand, modified)
        plain = score_hand(modified, replace(after, jokers=())).score
        self.assertEqual(score_hand(modified, after).score, plain * 3)


class ConsumableExecutionTests(unittest.TestCase):
    def run_consumable(self, session, consumable, plan, found, arranged=None, fallback=False):
        def reorder(*args):
            session.play_order_fallback = fallback
            return arranged or found, [0, 1] if not fallback else []
        with patch('controller.read_consumables', return_value=[]), \
             patch('controller.consumable_centers', return_value=[(1600, 200)]), \
             patch('controller.consumable_templates', return_value=({}, {consumable.key: consumable})), \
             patch('controller.read_text', return_value=[('Use', .99, (1600, 500))]), \
             patch('controller.identify_consumable_text', return_value=(consumable.key, .99)), \
             patch('controller.decide_consumable', return_value=plan), \
             patch('controller.apply_play_order', side_effect=reorder), \
             patch('controller.select_verified', return_value=session.frame), \
             patch('controller.consumable_counter', return_value=(0, 2)), \
             patch('controller.wait_consumable_hand', return_value=(session.frame, len(found))), \
             patch('controller.read_hand', return_value=arranged or found), \
             patch('controller.time.sleep'):
            return try_consumable(session, session.policy_state, found)

    def test_death_journal_records_the_verified_right_card_and_persists_duplicates(self):
        king = Card(13, 'H', enhancement='steel', seal='red')
        trash = Card(2, 'S')
        found = [(king, 1, (850, 720)), (trash, 1, (1000, 720))]
        arranged = [(trash, 1, (850, 720)), (king, 1, (1000, 720))]
        plan = ConsumableDecision('use', (0, 1), 100, reason='copy left King', play_order=(1, 0))
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            session.policy_state = RunState(deck_cards=(king, trash), deck_size=2, consumables_held=1)
            self.assertTrue(self.run_consumable(session, Consumable('Death', 'Tarot'), plan, found, arranged))
            self.assertEqual(session.policy_state.deck_cards, (king, king))
            self.assertEqual(session.policy_state.consumables_held, 0)
            saved = state_from_dict(json.loads((session.run_dir / 'policy-state.json').read_text('utf8')))
            self.assertEqual(saved.deck_cards, (king, king))

    def test_failed_death_direction_does_not_consume_and_does_not_repeat_forever(self):
        found = [(Card(13, 'H', enhancement='steel'), 1, (850, 720)),
                 (Card(2, 'S'), 1, (1000, 720))]
        plan = ConsumableDecision('use', (0, 1), 100, play_order=(1, 0))
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            session.policy_state = RunState(consumables_held=1)
            self.assertTrue(self.run_consumable(session, Consumable('Death', 'Tarot'), plan, found, fallback=True))
            self.assertFalse(self.run_consumable(session, Consumable('Death', 'Tarot'), plan, found, fallback=True))
            self.assertFalse(session.taps)
            self.assertIsNone(session.pending_action)
            self.assertEqual(session.policy_state.consumables_held, 1)

    def test_judgement_is_consumed_and_generated_joker_is_left_for_observation(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            session.policy_state = RunState(consumables_held=1)
            plan = ConsumableDecision('use', utility=12, deterministic=False)
            self.assertTrue(self.run_consumable(session, Consumable('Judgement', 'Tarot'), plan, []))
            self.assertEqual(session.policy_state.consumables_held, 0)
            self.assertEqual(session.policy_state.jokers, ())
            self.assertTrue(session.taps)


if __name__ == '__main__':
    unittest.main()
