"""Offline regressions for the nine interrupted acceptance attempts."""
import tempfile
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from item_identity import identify_description
from recovery import ObservationError, action_outcome
from shop_extras import (PackChoice, PackView, collect_pack_choices,
                         handle_pack, pack_choice_locators, plan_pack_reward)
from strategy import Card, Joker, RunState, ShopItem, decide_shop
from tests.test_runtime_recovery import FakeSession

ROOT = Path(__file__).resolve().parents[1]


class LogReviewTests(unittest.TestCase):
    def test_four_actual_bonus_score_residuals_match_after_tooltip_correction(self):
        from perception import parse_card_tooltip
        from recovery import state_from_dict
        from strategy import score_hand
        cases = json.loads((ROOT / 'tests/fixtures/bonus_score_residuals_20261006.json').read_text('utf-8'))
        self.assertEqual(len(cases), 4)
        for case in cases:
            cards = [Card(**card) for card in case['cards']]
            for index, labels in case['bonus_tooltips'].items():
                i = int(index)
                cards[i] = parse_card_tooltip(cards[i], labels, require_identity=True)
            order = case['order']
            result = score_hand([cards[i] for i in order], state_from_dict(case['state']),
                                [card for i, card in enumerate(cards) if i not in order])
            with self.subTest(time=case['time']):
                self.assertLessEqual(abs(result.score - case['actual']), .5)
                self.assertGreater(case['old_estimate'] - case['actual'], 400)

    def test_bonus_tooltip_separates_enhancement_from_permanent_chips(self):
        from perception import parse_card_tooltip
        from strategy import score_hand
        for enhancement, total, fixed in [('bonus', 30, 30), ('bonus', 40, 30), ('steel', 10, 0)]:
            card = parse_card_tooltip(Card(3, 'S'),
                ['3 of Spades', '+3 chips', f'{enhancement.title()} Card', f'+{total} extra chips'],
                require_identity=True)
            self.assertEqual(card.bonus_chips, total - fixed)
            self.assertEqual(score_hand([card]).chips, 5 + 3 + total)
        with self.assertRaises(ObservationError):
            parse_card_tooltip(Card(3, 'S'), ['3 of Spades', 'Bonus Card', '+5 extra chips'])

    def test_hiker_stone_tooltip_subtracts_the_fixed_fifty_chips(self):
        from perception import parse_card_tooltip
        from strategy import score_hand
        card = parse_card_tooltip(Card(0, '?'), ['Stone Card', '+65 Chips', 'no rank or suit'],
                                  require_identity=True)
        self.assertEqual(card.bonus_chips, 15)
        self.assertEqual(score_hand([card]).chips, 70)

    def test_emergency_reroll_uses_interest_reserve_but_keeps_purchase_budget(self):
        state = RunState(money=17, ante=6, blind_index=1, shop_rerolls=1,
                         jokers=(Joker('Supernova'), Joker('Banner')))
        forecast = SimpleNamespace(coverage=.78, weak=True, sustainable_score=31268,
                                   next_round=SimpleNamespace(safe_score=31268))
        with patch('planning.shop_forecast', return_value=forecast):
            self.assertEqual(decide_shop([], state, reroll_cost=6).action, 'reroll')
            from dataclasses import replace
            self.assertEqual(decide_shop([], replace(state, money=11), reroll_cost=6).action, 'leave')
            self.assertEqual(decide_shop([], replace(state, shop_rerolls=2), reroll_cost=6).action, 'leave')
            self.assertEqual(decide_shop([], replace(state, jokers=(Joker('Bull'),)), reroll_cost=6).action, 'leave')
        forecast.coverage = 1.2
        with patch('planning.shop_forecast', return_value=forecast):
            self.assertEqual(decide_shop([], state, reroll_cost=6).action, 'leave')

    def test_mail_in_rebate_is_classified_and_valued_as_income(self):
        from planning import engine_profile
        from strategy import _joker_purchase_value
        state = RunState(ante=2, jokers=(Joker('Mail-In Rebate'),))
        self.assertEqual(engine_profile(state).economy, 1)
        self.assertGreater(_joker_purchase_value(state.jokers[0], state), 15)

    def test_short_hermit_requires_the_complete_effect_and_cap(self):
        self.assertEqual(identify_description(
            ['Not Discovered', 'Doubles money', '(Max of $20]'], 'consumable'),
            ('the_hermit', 1.))
        for text in ['Doubles money', 'Doubles money (Max of $10)',
                     'Doubles money (Max of $200)', 'money (Max of $20)']:
            with self.subTest(text=text), self.assertRaises(RuntimeError):
                identify_description([text], 'consumable')

    def test_two_distinct_complete_effects_are_ambiguous(self):
        with self.assertRaises(RuntimeError):
            identify_description(['Doubles money (Max of $20)',
                'Enhances 2 selected cards to Mult Cards'], 'consumable')

    def test_repeated_tooltip_rows_preserve_the_readable_paragraph(self):
        rows = ['tarh $5 for each', 'discarded Jack, rank', 'changes severy ground',
                'Common', 'Not Discovered', 'Earn $5 for each',
                'discarded Jack, rank', 'changes every round', 'Common', 'TOKER']
        self.assertEqual(identify_description(rows)[0], 'mail_in_rebate')

    def test_actual_failed_hidden_tooltips_now_identify_the_reward(self):
        from shop_extras import offer_labels
        from perception import shop_offer_item
        for filename, center, expected in [
            ('hidden-hermit.png', (1312, 861), 'The Hermit'),
            ('hidden-mail-in-rebate.png', (1264, 744), 'Mail-In Rebate')]:
            path = ROOT / 'tests/fixtures/shop_extras_20261006' / filename
            frame = cv2.resize(cv2.imdecode(np.fromfile(path, np.uint8), 1),
                               (2064, 1191), interpolation=cv2.INTER_NEAREST)
            self.assertEqual(shop_offer_item(offer_labels(frame, center), 0)[0].name, expected)

    def test_buffoon_locators_find_the_actual_two_rewards(self):
        from shop_extras import read_pack_view
        path = ROOT / 'tests/fixtures/shop_extras_20261006/buffoon-locators.png'
        frame = cv2.resize(cv2.imdecode(np.fromfile(path, np.uint8), 1),
                           (2064, 1191), interpolation=cv2.INTER_NEAREST)
        candidates = pack_choice_locators(frame, read_pack_view(frame))
        self.assertEqual({key for key, _ in candidates}, {'dna', 'mail_in_rebate'})
        self.assertEqual(len(candidates), 2)
        self.assertTrue(all(1000 < x < 1260 and 840 < y < 890 for _, (x, y) in candidates))

    def test_real_pack_locators_exclude_deck_and_raised_hand(self):
        path = ROOT / 'tests/fixtures/shop_extras_20261006/open-arcana-12-cards.png'
        frame = cv2.resize(cv2.imdecode(np.fromfile(path, np.uint8), 1),
                           (2064, 1191), interpolation=cv2.INTER_NEAREST)
        from shop_extras import read_pack_view
        view = read_pack_view(frame)
        self.assertIsNotNone(view)
        candidates = pack_choice_locators(frame, view)
        self.assertEqual(len(candidates), 3)
        self.assertTrue(all(550 < x < 1670 and 800 < y < 950
                            for _, (x, y) in candidates))

    def test_targetless_reward_does_not_inspect_the_pack_hand(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            session.policy_state = RunState(money=25, pack_hand_from_full_deck=True)
            choices = [PackChoice(ShopItem('tarot', 'The Hermit', 0), (1100, 850))]
            view = PackView('arcana_pack', 1, (1500, 1050), 1000)
            with patch('shop_extras.collect_pack_hand') as read_hand:
                reward, found = plan_pack_reward(session, session.frame, view, choices)
            read_hand.assert_not_called()
            self.assertEqual(reward[0].item.name, 'The Hermit')
            self.assertEqual(reward[2].money, 45)
            self.assertEqual(found, [])
            self.assertEqual(session.taps, [])

    def test_unreadable_pack_hand_uses_only_verified_targetless_rewards(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            session.policy_state = RunState(money=25, pack_hand_from_full_deck=True)
            target = PackChoice(ShopItem('tarot', 'The Empress', 0), (900, 850))
            hermit = PackChoice(ShopItem('tarot', 'The Hermit', 0), (1200, 850))
            view = PackView('arcana_pack', 1, (1500, 1050), 1000)
            for choices, expected in [([target, hermit], 'The Hermit'), ([target], None)]:
                with patch('shop_extras.collect_pack_hand', side_effect=ObservationError('unreadable corner')):
                    reward, found = plan_pack_reward(session, session.frame, view, choices)
                self.assertEqual(reward[0].item.name if reward else None, expected)
                self.assertEqual(found, [])
            self.assertTrue(any(e['event'] == 'pack_hand_unavailable' for e in session.events))
            self.assertEqual(session.taps, [])

    def test_failed_target_verification_skips_without_pressing_use(self):
        from consumables import ConsumableDecision
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            target = PackChoice(ShopItem('tarot', 'The Empress', 0), (900, 850))
            view = PackView('arcana_pack', 1, (1500, 1050), 1000)
            planned = (target, ConsumableDecision('use', indices=(0, 1)), session.policy_state)
            with patch('shop_extras.read_pack_view', return_value=view), \
                 patch('shop_extras.collect_pack_choices', return_value=[target]), \
                 patch('shop_extras.plan_pack_reward', return_value=(planned, [])), \
                 patch('shop_extras.select_pack_targets', side_effect=ObservationError('lift not verified')), \
                 patch('vision.classify_scene', return_value='shop'), \
                 patch('perception.read_numbers', return_value={}):
                handle_pack(session, session.frame)
            self.assertEqual([tap[2] for tap in session.taps], ['skip unusable pack choices'])
            self.assertIsNone(session.pending_action)
            self.assertTrue(any(e['event'] == 'pack_targets_unavailable' for e in session.events))

    def test_transient_choice_tooltip_retries_and_unsupported_is_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            view = PackView('arcana_pack', 1, (1500, 1050), 1000)
            with patch('shop_extras.pack_choice_locators', return_value=[('the_hermit', (1100, 850))]), \
                 patch('shop_extras.offer_labels', side_effect=[ObservationError('panel missing'), ['The Hermit']]):
                choices = collect_pack_choices(session, session.frame, view)
            self.assertEqual(choices[0].item.name, 'The Hermit')
            with patch('shop_extras.pack_choice_locators', return_value=[('the_high_priestess', (1100, 850))]), \
                 patch('shop_extras.offer_labels', return_value=['The High Priestess']) as read:
                self.assertEqual(collect_pack_choices(session, session.frame, view), [])
            self.assertEqual(read.call_count, 1)
            self.assertTrue(any(e['event'] == 'pack_choice_unsupported' for e in session.events))
            self.assertFalse(any(e['event'] == 'pack_choice_unreadable' for e in session.events))
            self.assertEqual(session.taps, [])

    def test_purchase_receipt_requires_matching_type_count_and_debit(self):
        before = dict(money=30, joker_count=2, consumable_count=1)
        pending = dict(kind='buy', before=before, details=dict(reward_kind='joker', price=5))
        expected = dict(money=25, joker_count=3, consumable_count=1)
        self.assertTrue(action_outcome(pending, 'shop', expected))
        self.assertFalse(action_outcome(pending, 'shop', before))
        for incomplete in [dict(money=25), dict(money=25, joker_count=2, consumable_count=2),
                           dict(money=30, joker_count=3, consumable_count=1),
                           dict(money=24, joker_count=3, consumable_count=1)]:
            with self.subTest(receipt=incomplete):
                self.assertIsNone(action_outcome(pending, 'shop', incomplete))
        pending['details'] = dict(reward_kind='tarot', price=0)
        self.assertTrue(action_outcome(pending, 'shop', dict(money=30, joker_count=2, consumable_count=2)))
        self.assertIsNone(action_outcome(pending, 'shop', dict(money=30, joker_count=3, consumable_count=1)))


if __name__ == '__main__':
    unittest.main()
