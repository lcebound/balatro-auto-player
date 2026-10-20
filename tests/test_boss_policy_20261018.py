"""Boss mechanics, real missed previews and a pre-inspection Leaf sale."""
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from random import Random
from unittest.mock import Mock, patch

import numpy as np

import runner
from boss_policy import (BOSS_TACTICS, after_joker_sale, choose_verdant_sale,
    discard_draw_count, forecast_boss_turn, forecast_draw, future_boss_weight,
    remember_upcoming_boss)
from perception import blind_names, read_future_boss
from planning import _round_state, forecast_round
from recovery import ObservationError, action_outcome, begin_action, finish_action, state_from_dict
from strategy import (Card, Joker, RunState, _card_debuffed, _discard_candidates,
    _draw_best_plan, decide_play, score_hand, joker_name)
from tests.test_runtime_recovery import FakeSession
from tests.test_speed_and_policy_20261005 import forecast
from tools.replay_observation_speed import load_frame

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / 'tests/fixtures/boss_review_20261018.json').read_text('utf-8'))
LEAF = state_from_dict(DATA['cases'][-1]['state'])
HAND = [Card(r, s) for r, s in ((14, 'S'), (14, 'H'), (13, 'D'), (12, 'C'),
                               (10, 'H'), (8, 'S'), (6, 'D'), (3, 'C'))]


class PreviewTests(unittest.TestCase):
    def test_all_three_real_ante_eight_leaf_previews_are_recovered(self):
        for case in DATA['blinds'][-3:]:
            self.assertEqual(read_future_boss(load_frame(ROOT / case['proof']['path'])), ('Verdant Leaf', 100000))

    def test_normal_title_needs_no_additional_ocr_and_conflicts_remain_unknown(self):
        image = np.zeros((1191, 2064, 3), np.uint8)
        words = [('The Plant', .99, (1500, 560)), ('40,000', .99, (1500, 870))]
        with patch('perception.read_text', return_value=words) as reader:
            self.assertEqual(read_future_boss(image), ('The Plant', 40000))
            self.assertEqual(reader.call_count, 1)
        with patch('perception.read_text', return_value=words + [('The Eye', .99, (1500, 610))]):
            self.assertIsNone(read_future_boss(image))

    def test_uv_alias_needs_independent_rule_text_and_a_unique_target(self):
        image = np.zeros((1191, 2064, 3), np.uint8)
        words = [('Uerdant Leaf', .99, (1500, 560)), ('100,000', .99, (1500, 870))]
        with patch('perception.read_text', side_effect=[words, words[:1]]):
            self.assertIsNone(read_future_boss(image))

    def test_all_retained_failed_previews_including_ox_and_violet_are_readable(self):
        for case in DATA['blinds']:
            with self.subTest(index=case['index']):
                self.assertIsNotNone(read_future_boss(load_frame(ROOT / case['proof']['path'])))

    def test_new_ante_cannot_reuse_the_previous_boss(self):
        state = replace(LEAF, ante=8, upcoming_boss='The Serpent', upcoming_boss_ante=7)
        result = remember_upcoming_boss(state, 8, None)
        self.assertEqual((result.upcoming_boss, result.upcoming_boss_target, result.upcoming_boss_ante), ('', 0, 0))
        result = remember_upcoming_boss(state, 8, ('Verdant Leaf', 100000))
        self.assertEqual((result.upcoming_boss, result.upcoming_boss_ante), ('Verdant Leaf', 8))

    def test_same_ante_missing_preview_keeps_known_boss_and_pillar_history(self):
        state = replace(LEAF, upcoming_boss_ante=8, pillar_cards=(Card(14, 'S'),))
        self.assertEqual(remember_upcoming_boss(state, 8, None), state)
        unknown = replace(state, upcoming_boss='', upcoming_boss_ante=0)
        self.assertEqual(remember_upcoming_boss(unknown, 8, None).pillar_cards, state.pillar_cards)
        self.assertEqual(remember_upcoming_boss(state, 9, None).pillar_cards, ())


class LeafTests(unittest.TestCase):
    def test_sacrifice_is_legal_and_projection_accounts_for_lost_joker(self):
        index = choose_verdant_sale(LEAF)
        self.assertIn(index, range(len(LEAF.jokers)))
        after = _round_state(LEAF, 'Verdant Leaf', 100000)
        self.assertTrue(after.boss_disabled)
        self.assertEqual(len(after.jokers), len(LEAF.jokers) - 1)
        self.assertEqual(len(LEAF.jokers), 5)
        self.assertFalse(LEAF.boss_disabled)

    def test_eternals_no_jokers_disabled_boss_and_chicot_need_no_sale(self):
        for state in (replace(LEAF, jokers=()), replace(LEAF, boss_disabled=True),
                      replace(LEAF, jokers=(Joker('Chicot'),)),
                      replace(LEAF, jokers=(Joker('Baron', data=(('eternal', 1),)),))):
            self.assertIsNone(choose_verdant_sale(state))
        state = replace(LEAF, jokers=(Joker('Baron', data=(('eternal', 1),)), Joker('Golden Joker')))
        self.assertEqual(choose_verdant_sale(state), 1)

    def test_new_draws_and_stone_are_debuffed_until_a_real_sale(self):
        for card in (Card(13, 'S'), Card(2, 'H', enhancement='stone')):
            self.assertTrue(_card_debuffed(card, LEAF))
            self.assertFalse(_card_debuffed(card, replace(LEAF, boss_disabled=True)))
        no_sale = _round_state(replace(LEAF, jokers=()), 'Verdant Leaf', 100000)
        self.assertFalse(no_sale.boss_disabled)

    def test_ordinary_low_scoring_hands_remain_playable_under_leaf(self):
        state = replace(LEAF, discards_left=0, jokers=(), target_score=100000)
        decision = decide_play(HAND, state, lookahead=False)
        self.assertTrue(decision.indices)
        self.assertGreater(decision.estimate.score, 0)  # Debuff does not ban playing.

    def test_sale_does_not_revoke_burglars_already_granted_hands(self):
        state = replace(LEAF, jokers=(Joker('Burglar'),), hands_left=7)
        self.assertEqual(after_joker_sale(state, 0, disable_leaf=True).hands_left, 7)

    def test_boss_sale_needs_an_exact_count_receipt_not_only_cash(self):
        p = dict(kind='sell', before=dict(joker_count=5, money=33), details=dict(purpose='disable_verdant_leaf'))
        self.assertIsNone(action_outcome(p, 'hand', {'money': 40}))
        self.assertIsNone(action_outcome(p, 'hand', {'joker_count': 3, 'money': 40}))
        self.assertIsNone(action_outcome(p, 'shop', {'joker_count': 4, 'money': 40}))
        self.assertFalse(action_outcome(p, 'hand', {'joker_count': 5, 'money': 33}))
        self.assertTrue(action_outcome(p, 'hand', {'joker_count': 4, 'money': 33}))
        p['details'] = {}
        self.assertIsNone(action_outcome(p, 'hand', {'joker_count': 4}))

    def test_only_confirmed_sale_commits_disable_and_replay_is_idempotent(self):
        for applied in (False, True):
            with self.subTest(applied=applied), tempfile.TemporaryDirectory() as directory:
                session = FakeSession(directory)
                session.policy_state = LEAF
                after = after_joker_sale(LEAF, 4, disable_leaf=True)
                begin_action(session, 'sell', {'joker_count': 5, 'money': 33}, after_state=after,
                             name='Baron', purpose='disable_verdant_leaf')
                self.assertFalse(session.policy_state.boss_disabled)
                finish_action(session, applied)
                self.assertEqual(session.policy_state.boss_disabled, applied)
                finish_action(session, applied)
                self.assertEqual(len(session.policy_state.jokers), 4 if applied else 5)

    def test_sale_precedes_card_collection_even_when_the_old_frame_would_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            session.policy_state = LEAF
            image = load_frame(ROOT / DATA['failures'][-1]['proof']['path'])
            session.scene_hand_frame = image
            actions = []
            def sell(s, state, index, **kwargs):
                actions.append('sale')
                self.assertTrue(kwargs['disable_leaf'])
                s.boss_after_sale_frame = image
                return after_joker_sale(state, index, disable_leaf=True)
            def collect(s, state, **kwargs):
                actions.append('collect')
                self.assertTrue(state.boss_disabled)
                raise ObservationError('stop before simulated card collection')
            with patch('perception.observed_state', return_value=(LEAF, 8)), \
                    patch('runner.collect_jokers', return_value=LEAF), \
                    patch('runner.sell_shop_joker', side_effect=sell), \
                    patch('runner.collect_position', side_effect=collect):
                with self.assertRaises(ObservationError):
                    runner.play_one_hand(session)
            self.assertEqual(actions, ['sale', 'collect'])

    def test_pending_action_blocks_preparation_and_a_second_sale(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            session.pending_action = {'kind': 'sell'}
            with patch('runner.sell_shop_joker') as sell:
                with self.assertRaises(ObservationError):
                    runner.prepare_boss_action(session, LEAF)
                sell.assert_not_called()


class BossMechanicsTests(unittest.TestCase):
    def test_catalogue_covers_all_twenty_eight_installed_bosses(self):
        self.assertEqual({joker_name(n).removeprefix('the_') for n in blind_names()}, set(BOSS_TACTICS))

    def test_suit_face_and_pillar_effects_do_not_disable_unrelated_cards(self):
        for boss, suit in [('The Club', 'C'), ('The Goad', 'S'), ('The Head', 'H'), ('The Window', 'D')]:
            state = RunState(boss=boss)
            self.assertTrue(_card_debuffed(Card(2, suit), state))
            self.assertFalse(_card_debuffed(Card(2, 'D' if suit != 'D' else 'H'), state))
        self.assertTrue(_card_debuffed(Card(13, 'S'), RunState(boss='The Plant')))
        self.assertFalse(_card_debuffed(Card(10, 'S'), RunState(boss='The Plant')))
        self.assertTrue(_card_debuffed(Card(2, 'S'), RunState(boss='The Pillar', pillar_cards=(Card(2, 'S'),))))

    def test_psychic_eye_mouth_and_bell_choose_legal_scoring_shapes(self):
        self.assertEqual(len(decide_play(HAND, RunState(boss='The Psychic'), lookahead=False).indices), 5)
        state = RunState(boss='The Eye', boss_seen_types=frozenset({'pair'}))
        self.assertNotEqual(decide_play(HAND, state, lookahead=False).estimate.category, 'pair')
        mouth = RunState(boss='The Mouth', boss_target_hand='pair')
        self.assertEqual(decide_play(HAND, mouth, lookahead=False).estimate.category, 'pair')
        bell = RunState(boss='Cerulean Bell', forced_card_index=7)
        self.assertIn(7, _draw_best_plan(HAND, bell)[0])
        for keep in _discard_candidates(HAND, bell):
            self.assertNotIn(7, keep)
            self.assertLessEqual(len(HAND) - len(keep), 5)

    def test_water_needle_manacle_and_serpent_resources(self):
        s = RunState(hand_size=8, base_hands_per_round=4, base_discards_per_round=3)
        self.assertEqual(_round_state(s, 'The Water', 1000).discards_left, 0)
        self.assertEqual(_round_state(s, 'The Needle', 1000).hands_left, 1)
        self.assertEqual(_round_state(s, 'The Manacle', 1000).hand_size, 7)
        for dropped in (1, 2, 5):
            self.assertEqual(discard_draw_count(replace(s, boss='The Serpent'), dropped, 20), 3)
        self.assertEqual(discard_draw_count(replace(s, boss='The Serpent'), 5, 2), 2)

    def test_house_fish_mark_and_wheel_do_not_reveal_hidden_fronts(self):
        house = RunState(boss='The House')
        self.assertTrue(all(c.face_down and c.rank == 0 for c in forecast_draw(HAND, house, Random(1), 'initial')))
        self.assertEqual(forecast_draw(HAND, house, Random(1), 'discard'), HAND)
        fish = RunState(boss='The Fish')
        self.assertEqual(forecast_draw(HAND, fish, Random(1), 'initial'), HAND)
        self.assertTrue(all(c.face_down for c in forecast_draw(HAND, fish, Random(1), 'play')))
        mark = forecast_draw(HAND, RunState(boss='The Mark'), Random(1), 'discard')
        self.assertEqual([c.face_down for c in mark], [False, False, True, True, False, False, False, False])
        wheel = forecast_draw(HAND * 10, RunState(boss='The Wheel'), Random(1), 'initial')
        self.assertTrue(any(c.face_down for c in wheel))
        self.assertTrue(all(c.rank == 0 for c in wheel if c.face_down))

    def test_heart_disables_one_source_and_rotates_without_disabling_permanent_effects(self):
        s = RunState(boss='Crimson Heart', jokers=(Joker('Joker'), Joker('Baron'), Joker('Egg', debuffed=True)))
        first = forecast_boss_turn(s, HAND, Random(1))
        second = forecast_boss_turn(first, HAND, Random(2))
        self.assertEqual(sum(bool(j.get('_forecast_heart_disabled')) for j in first.jokers), 1)
        self.assertNotEqual([j.name for j in first.jokers if j.get('_forecast_heart_disabled')],
                            [j.name for j in second.jokers if j.get('_forecast_heart_disabled')])
        self.assertTrue(second.jokers[2].debuffed)
        self.assertFalse(s.jokers[0].debuffed)

    def test_flint_arm_tooth_and_ox_apply_their_actual_costs(self):
        cards = [Card(14, 'S'), Card(14, 'H')]
        base = score_hand(cards, RunState()).score
        self.assertLess(score_hand(cards, RunState(boss='The Flint')).score, base)
        self.assertLess(score_hand(cards, RunState(boss='The Arm', levels={'pair': 3})).score,
                        score_hand(cards, RunState(levels={'pair': 3})).score)
        self.assertEqual(score_hand(cards, RunState(boss='The Tooth')).cash, -2)
        self.assertEqual(score_hand(cards, RunState(boss='The Ox', boss_target_hand='pair', money=33)).cash, -33)

    def test_ox_target_is_read_from_visible_rule_not_guessed_from_history(self):
        from perception import observed_state
        image = np.zeros((1191, 2064, 3), np.uint8)
        state = RunState(boss='The Ox', played_types={'high_card': 50})
        with patch('perception.read_numbers', return_value=dict(target_score=70000,round_score=0,hands_left=4,discards_left=3)), \
                patch('perception.hand_counter', return_value=(8, 8)), \
                patch('perception.consumable_counter', return_value=(0, 2)), \
                patch('perception.read_text', return_value=[]), \
                patch('perception.tooltip_labels', return_value=['The Ox', 'Playing a Two Pair']):
            observed, _ = observed_state(image, state)
        self.assertEqual(observed.boss_target_hand, 'two_pair')

    def test_boss_cash_deductions_affect_cash_jokers_before_scoring(self):
        cards = HAND[:2]
        state = RunState(money=10, jokers=(Joker('Bull'), Joker('Bootstraps')))
        ox = score_hand(cards, replace(state, boss='The Ox', boss_target_hand='pair'))
        self.assertEqual(ox.score, score_hand(cards, replace(state, money=0)).score)
        tooth = score_hand(cards, replace(state, boss='The Tooth'))
        self.assertEqual(tooth.score, score_hand(cards, replace(state, money=8)).score)
        negative = score_hand(cards, replace(state, money=0, boss='The Tooth'))
        self.assertEqual(negative.score, score_hand(cards, replace(state, money=0)).score)

    def test_future_weight_is_small_then_rises_without_softening_the_next_boss(self):
        weights = [future_boss_weight(a) for a in range(1, 9)]
        self.assertLessEqual(max(weights[:6]), .06)
        self.assertEqual(weights[-2:], [.12, .18])
        baseline = forecast(safe=20000, target=10000, boss_safe=4000)
        early = replace(baseline, boss_weight=weights[0])
        late = replace(baseline, boss_weight=weights[-1])
        self.assertGreater(early.coverage, late.coverage)
        self.assertGreater(early.coverage, 1.8)
        imminent = replace(forecast(safe=4000, target=40000), boss_weight=.03)
        self.assertLess(imminent.coverage, 1)
        self.assertTrue(imminent.weak)

    def test_every_boss_forecast_terminates_with_finite_nonnegative_scores(self):
        # All 28 strategies execute the same bounded round simulator.
        for name in blind_names():
            with self.subTest(boss=name):
                result = forecast_round(RunState(jokers=(Joker('Joker'),)), name, 600, samples=1)
                self.assertGreaterEqual(result.safe_score, 0)
                self.assertLess(result.safe_score, float('inf'))


if __name__ == '__main__':
    unittest.main()
