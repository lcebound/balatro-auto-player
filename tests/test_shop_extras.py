"""Booster/voucher regressions; never create a live Airtest Session."""
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from recovery import ObservationError, action_outcome, begin_action, confirm_pending, state_from_dict
from strategy import Card, Joker, RunState, ShopItem
from shop_extras import (PackChoice, PackView, extra_catalogue, identify_extra, voucher_state,
    extra_is_worth_considering, read_pack_view, purchase_extra, handle_pack,
    choose_pack_reward, collect_extra_offers, verified_button)
from tests.test_runtime_recovery import FakeSession

ROOT = Path(__file__).resolve().parents[1]


def pending(kind, before=None, **details):
    return dict(kind=kind, before=before or {'pack_remaining': 2}, details=details)


class ExtrasTests(unittest.TestCase):
    def test_all_catalogue_names_and_sizes(self):
        definitions, _ = extra_catalogue()
        self.assertEqual(sum(d['kind'] == 'voucher' for d in definitions.values()), 32)
        self.assertEqual(sum(d['kind'] != 'voucher' for d in definitions.values()), 32)
        for d in definitions.values():
            item = identify_extra([d['name']], 10)
            self.assertEqual(item.name, d['name'])
            self.assertEqual(item.kind, d['kind'])
            if d['kind'] != 'voucher':
                self.assertEqual(d['choose'], 2 if d['name'].startswith('Mega') else 1)

    def test_effect_only_and_conflicting_titles_rejected(self):
        for rows in [['+1 hand each round'], ['Grabber', 'Wasteful'], ['Pack']]:
            with self.assertRaises(ObservationError):
                identify_extra(rows, 10)

    def test_resource_vouchers_update_base_and_current_once(self):
        for name, field in [('Grabber', 'base_hands_per_round'), ('Wasteful', 'base_discards_per_round'),
                            ('Paint Brush', 'hand_size'), ('Crystal Ball', 'consumable_slots'),
                            ('Antimatter', 'total_joker_slots')]:
            state = RunState(money=40)
            item = ShopItem('voucher', name, 10)
            upgraded = voucher_state(state, item)
            self.assertEqual(getattr(upgraded, field), getattr(state, field) + 1)
            self.assertEqual(upgraded.money, 30)
            self.assertEqual(len(upgraded.owned_vouchers), 1)
            with self.assertRaises(ObservationError):
                voucher_state(upgraded, item)
            self.assertEqual(state_from_dict(asdict(upgraded)), upgraded)

    def test_interest_is_cash_threshold_not_interest_dollars(self):
        state = RunState(money=100)
        self.assertEqual(voucher_state(state, ShopItem('voucher', 'Seed Money', 10)).interest_cash_cap, 50)
        self.assertEqual(voucher_state(state, ShopItem('voucher', 'Money Tree', 10)).interest_cash_cap, 100)

    def test_no_unknown_pack_rewards_in_purchase_forecast(self):
        from strategy import _purchase_state
        state = RunState(money=30)
        after = _purchase_state(ShopItem('celestial_pack', 'Celestial Pack', 4), state)
        self.assertEqual(after.money, 26)
        self.assertEqual(after.levels, {})
        self.assertEqual(after.jokers, ())

    def test_reserve_and_full_joker_slots(self):
        weak = SimpleNamespace(weak=True)
        self.assertTrue(extra_is_worth_considering(ShopItem('buffoon_pack', 'Buffoon Pack', 4), RunState(money=4), weak))
        self.assertFalse(extra_is_worth_considering(ShopItem('arcana_pack', 'Arcana Pack', 4), RunState(money=4), weak))
        full = RunState(money=80, jokers=tuple(Joker('Joker') for _ in range(5)))
        self.assertFalse(extra_is_worth_considering(ShopItem('buffoon_pack', 'Buffoon Pack', 4), full, weak))
        self.assertFalse(extra_is_worth_considering(ShopItem('voucher', 'Retcon', 10), full, weak))
        self.assertFalse(extra_is_worth_considering(ShopItem('voucher', 'Blank', 10), replace(full, money=30), weak))

    def test_pack_parser_requires_title_skip_and_remaining(self):
        words = [('Celestial Pack', .99, (1030, 1000)), ('Choose', .99, (970, 1050)),
                 ('2', .99, (1080, 1050)), ('Skip', .99, (1450, 1060))]
        with patch('perception.read_text', return_value=words):
            view = read_pack_view(np.zeros((1191, 2064, 3), np.uint8))
        self.assertEqual(view.kind, 'celestial_pack')
        self.assertEqual(view.remaining, 2)
        for missing in (words[:-1], [words[0], words[-1]], words[1:]):
            with patch('perception.read_text', return_value=missing):
                self.assertIsNone(read_pack_view(np.zeros((1191, 2064, 3), np.uint8)))

    def test_pack_scene_has_priority_over_shop(self):
        from vision import classify_scene
        with patch('vision.scene_scores', return_value={'shop': .9, 'deck': 0}), \
             patch('shop_extras.read_pack_view', return_value=PackView('celestial_pack', 1, (1400, 1000), 1000)):
            self.assertEqual(classify_scene(np.zeros((1191, 2064, 3), np.uint8)), 'pack')

    def test_open_requires_expected_pack_and_exact_debit(self):
        p = pending('open_pack', {'money': 40}, price=4, pack_kind='celestial_pack')
        self.assertTrue(action_outcome(p, 'pack', dict(money=36, pack_kind='celestial_pack')))
        self.assertIsNone(action_outcome(p, 'pack', dict(money=36, pack_kind='arcana_pack')))
        self.assertIsNone(action_outcome(p, 'pack', dict(money=35, pack_kind='celestial_pack')))
        self.assertFalse(action_outcome(p, 'shop', dict(money=40)))

    def test_redeem_requires_removed_offer_not_cash_alone(self):
        p = pending('redeem', {'money': 40}, price=10)
        self.assertIsNone(action_outcome(p, 'shop', dict(money=30, voucher_price_visible=True)))
        self.assertTrue(action_outcome(p, 'shop', dict(money=30, voucher_price_visible=False)))
        self.assertIsNone(action_outcome(p, 'pack', dict(money=30, voucher_price_visible=False)))

    def test_mega_pick_progress_and_inventory_does_not_confirm(self):
        p = pending('pick_pack', pack_kind='celestial_pack')
        self.assertTrue(action_outcome(p, 'pack', dict(pack_kind='celestial_pack', pack_remaining=1)))
        self.assertFalse(action_outcome(p, 'pack', dict(pack_kind='celestial_pack', pack_remaining=2, consumable_count=0)))
        self.assertIsNone(action_outcome(p, 'pack', dict(pack_kind='arcana_pack', pack_remaining=1)))
        self.assertTrue(action_outcome(p, 'shop', {}))
        self.assertFalse(action_outcome(pending('skip_pack'), 'pack', dict(pack_remaining=2)))

    def test_resume_confirms_once_and_marks_levels_dirty(self):
        with tempfile.TemporaryDirectory() as directory:
            s = FakeSession(directory)
            after = RunState(money=36, levels={'pair': 2})
            begin_action(s, 'pick_pack', {'pack_remaining': 2}, after_state=after,
                         pack_kind='celestial_pack', levels_changed=True)
            with patch('vision.classify_scene', return_value='pack'), \
                 patch('perception.read_numbers', return_value={'money': 36}), \
                 patch('shop_extras.extra_outcome_values', return_value=dict(pack_kind='celestial_pack', pack_remaining=1)):
                confirm_pending(s)
                confirm_pending(s)
            self.assertEqual(s.policy_state.level('pair'), 2)
            self.assertTrue(s.levels_dirty)
            self.assertEqual(s.taps, [])
            self.assertIsNone(s.pending_action)

    def test_selected_wrong_offer_never_clicks_redeem(self):
        with tempfile.TemporaryDirectory() as directory:
            s = FakeSession(directory)
            with patch('shop_extras.offer_labels', return_value=['Wasteful']):
                with self.assertRaises(ObservationError):
                    purchase_extra(s, ShopItem('voucher', 'Grabber', 10), (900, 970), RunState(money=40))
            self.assertEqual(len(s.taps), 1)  # selection only; no financial click
            self.assertIsNone(s.pending_action)

    def test_ambiguous_button_rejected(self):
        with patch('perception.read_text', return_value=[('Use', .99, (1200, 700)), ('Use', .99, (1300, 710))]):
            with self.assertRaises(ObservationError):
                verified_button(np.zeros((1191, 2064, 3), np.uint8), 'Use', (1200, 600))

    def test_old_shop_prices_locate_three_extra_offers(self):
        from perception import read_text
        path = ROOT / 'tests/fixtures/shop_extras_20261005/late-celestial-offers.png'
        image = cv2.resize(cv2.imdecode(np.fromfile(path, np.uint8), 1), (2064, 1191), interpolation=cv2.INTER_NEAREST)
        prices = [w for w in read_text(image, (810, 750, 1770, 890)) if w[0].startswith('$')]
        self.assertEqual(sorted(w[0] for w in prices), ['$10', '$4', '$4'])
        def labels(frame, center):
            return ['Grabber'] if center[0] < 1000 else ['Celestial Pack']
        with tempfile.TemporaryDirectory() as directory:
            s = FakeSession(directory)
            with patch('shop_extras.offer_labels', side_effect=labels):
                items, positions = collect_extra_offers(s, image)
            self.assertEqual(len(items), 3)
            self.assertEqual(sum(i.kind == 'celestial_pack' for i in items), 2)
            self.assertTrue(all(900 < p[1] < 1050 for p in positions))
            self.assertEqual(s.taps, [])

    def test_planet_pack_uses_best_engine_upgrade_not_largest_base_hand(self):
        state = RunState(money=40, ante=4, jokers=(Joker('Half Joker', value=20),),
                         levels={'pair': 4}, played_types={'pair': 20})
        options = [PackChoice(ShopItem('planet', 'Mars', 0), (1000, 600)),
                   PackChoice(ShopItem('planet', 'Mercury', 0), (1300, 600))]
        choice, plan, after = choose_pack_reward(options, state)
        self.assertEqual(choice.item.name, 'Mercury')
        self.assertEqual(after.level('pair'), 5)
        self.assertEqual(after.consumables_held, state.consumables_held)

    def test_pack_planet_preserves_held_observatory_inventory(self):
        state = RunState(money=40, jokers=(Joker('Constellation', value=1.2),),
                         consumables_held=2, planet_cards={'pair': 1})
        choice, plan, after = choose_pack_reward([PackChoice(ShopItem('planet', 'Mercury', 0), (1000, 600))], state)
        self.assertEqual(after.planet_cards, {'pair': 1})
        self.assertEqual(after.consumables_held, 2)
        self.assertAlmostEqual(after.jokers[0].value, 1.3)

    def test_full_joker_bar_can_only_take_negative(self):
        state = RunState(money=50, jokers=tuple(Joker('Joker', value=4) for _ in range(5)))
        ordinary = PackChoice(ShopItem('joker', 'Joker', 0, joker=Joker('Joker', value=4)), (1000, 600))
        self.assertIsNone(choose_pack_reward([ordinary], state))
        negative = PackChoice(ShopItem('joker', 'Joker', 0, joker=Joker('Joker', value=4, edition='negative')), (1300, 600))
        self.assertEqual(choose_pack_reward([negative], state)[2].total_joker_slots, 6)

    def test_hologram_standard_card_growth_and_deck_persistence(self):
        card = Card(13, 'S', enhancement='steel', seal='red')
        state = RunState(money=40, jokers=(Joker('Hologram', value=1.5),), deck_cards=(Card(2, 'H'),), deck_size=1)
        choice, plan, after = choose_pack_reward([PackChoice(ShopItem('card', '13S', 0), (1000, 600), card)], state)
        self.assertEqual(after.deck_size, 2)
        self.assertEqual(after.deck_cards[-1], card)
        self.assertAlmostEqual(after.jokers[0].value, 1.75)

    def test_unsafe_spectral_is_skipped(self):
        choice = PackChoice(ShopItem('spectral', 'Ectoplasm', 0), (1000, 600))
        self.assertIsNone(choose_pack_reward([choice], RunState(money=50)))

    def test_empty_pack_choices_skip_with_verified_transition(self):
        with tempfile.TemporaryDirectory() as directory:
            s = FakeSession(directory)
            view = PackView('celestial_pack', 1, (1500, 1000), 1000)
            with patch('shop_extras.read_pack_view', return_value=view), \
                 patch('shop_extras.collect_pack_choices', return_value=[]), \
                 patch('vision.classify_scene', return_value='shop'), \
                 patch('perception.read_numbers', return_value={}):
                handle_pack(s, s.frame)
            self.assertEqual(len(s.taps), 1)
            self.assertIsNone(s.pending_action)
            self.assertEqual(s.policy_state.levels, {})

    def test_mega_workflow_finishes_two_distinct_uses_and_returns_to_shop(self):
        class MegaSession(FakeSession):
            def __init__(self, directory):
                super().__init__(directory)
                self.stage = 0

            def observe(self, label):
                return np.full((1191, 2064, 3), self.stage, np.uint8)

            def tap(self, x, y, reason):
                super().tap(x, y, reason)
                if reason == 'take verified pack reward':
                    self.stage += 1

        def view(image):
            stage = int(image[0, 0, 0])
            return PackView('celestial_pack', 2 - stage, (1500, 1000), 1000) if stage < 2 else None

        def counters(image):
            return {'money': 32}

        def choices(session, image, pack):
            name = 'Mercury' if session.stage == 0 else 'Jupiter'
            return [PackChoice(ShopItem('planet', name, 0), (1100, 600))]

        def offer(image, center):
            return ['Mercury' if image[0, 0, 0] == 0 else 'Jupiter']

        def parse(labels, price, hint=''):
            return ShopItem('planet', labels[0], 0), 1.

        with tempfile.TemporaryDirectory() as directory:
            s = MegaSession(directory)
            s.policy_state = RunState(money=32, consumables_held=2)
            with patch('shop_extras.read_pack_view', side_effect=view), \
                 patch('shop_extras.collect_pack_choices', side_effect=choices), \
                 patch('shop_extras.offer_labels', side_effect=offer), \
                 patch('shop_extras.verified_button', return_value=(1100, 750)), \
                 patch('perception.shop_offer_item', side_effect=parse), \
                 patch('controller.collect_jokers', side_effect=lambda session, image, state, **kw: state), \
                 patch('shop_extras.shop_ready', side_effect=lambda image: image[0, 0, 0] == 2), \
                 patch('vision.classify_scene', side_effect=lambda image: 'shop' if image[0, 0, 0] == 2 else 'pack'), \
                 patch('perception.read_numbers', side_effect=counters):
                handle_pack(s, s.observe('initial'))
            self.assertEqual(s.stage, 2)
            self.assertEqual(s.policy_state.level('pair'), 2)
            self.assertEqual(s.policy_state.level('flush'), 2)
            self.assertEqual(s.policy_state.consumables_held, 2)
            self.assertEqual(sum(e['event'] == 'action_confirmed' for e in s.events), 2)
            self.assertIsNone(s.pending_action)

    def test_wrong_selected_pack_reward_never_uses_it(self):
        with tempfile.TemporaryDirectory() as directory:
            s = FakeSession(directory)
            view = PackView('celestial_pack', 1, (1500, 1000), 1000)
            choices = [PackChoice(ShopItem('planet', 'Mercury', 0), (1100, 600))]
            with patch('shop_extras.read_pack_view', return_value=view), \
                 patch('shop_extras.collect_pack_choices', return_value=choices), \
                 patch('shop_extras.offer_labels', return_value=['Jupiter']), \
                 patch('perception.shop_offer_item', return_value=(ShopItem('planet', 'Jupiter', 0), 1.)):
                with self.assertRaises(ObservationError):
                    handle_pack(s, s.frame)
            self.assertEqual(len(s.taps), 1)
            self.assertIsNone(s.pending_action)

    def test_hermit_pack_predicts_cash_without_consuming_a_held_slot(self):
        state = RunState(money=12, consumables_held=1)
        _, _, after = choose_pack_reward([PackChoice(ShopItem('tarot', 'The Hermit', 0), (1100, 600))], state)
        self.assertEqual(after.money, 24)
        self.assertEqual(after.consumables_held, 1)

    def test_pack_card_and_joker_require_independent_receipt_counter(self):
        for kind, counter, value in [('joker', 'joker_count', 3), ('card', 'deck_size', 52)]:
            p = pending('pick_pack', {'pack_remaining': 1, counter: value}, reward_kind=kind)
            self.assertIsNone(action_outcome(p, 'shop', {}))
            self.assertIsNone(action_outcome(p, 'shop', {counter: value}))
            self.assertTrue(action_outcome(p, 'shop', {counter: value + 1}))

    def test_transient_extra_tooltip_retries_without_clicking(self):
        from shop_extras import verify_offer_selection
        def check(rows):
            if identify_extra(rows, 10).name != 'Grabber':
                raise ObservationError('wrong title')
        with tempfile.TemporaryDirectory() as directory:
            s = FakeSession(directory)
            with patch('shop_extras.offer_labels', side_effect=[[], ['Grabber']]), \
                 patch('shop_extras.verified_button', return_value=(910, 1050)):
                button = verify_offer_selection(s, s.frame, (910, 970), 'Redeem', check)
            self.assertEqual(button, (910, 1050))
            self.assertEqual(s.taps, [])

    def test_same_shop_bottom_offers_reuse_titles_but_new_prices_invalidate(self):
        with tempfile.TemporaryDirectory() as directory:
            s = FakeSession(directory)
            prices = [('$10', .99, (910, 810)), ('$4', .99, (1270, 790))]
            def labels(image, center):
                return ['Grabber'] if center[0] < 1000 else ['Celestial Pack']
            with patch('perception.read_text', return_value=prices) as ocr, \
                 patch('shop_extras.offer_labels', side_effect=labels) as titles:
                first = collect_extra_offers(s, s.frame)
                self.assertEqual(collect_extra_offers(s, s.frame), first)
                self.assertEqual(titles.call_count, 2)
                ocr.return_value = [('$10', .99, (910, 810)), ('$0', .99, (1270, 790))]
                changed, _ = collect_extra_offers(s, s.frame)
                self.assertEqual(titles.call_count, 4)
                self.assertEqual(changed[1].price, 0)

    def test_undiscovered_shop_names_from_real_tooltips(self):
        from shop_extras import offer_labels
        for filename, x, name, kind in [
            ('reroll-surplus.png', 910, 'Reroll Surplus', 'voucher'),
            ('buffoon.png', 1269, 'Buffoon Pack', 'buffoon_pack'),
            ('arcana.png', 1472, 'Arcana Pack', 'arcana_pack')]:
            path = ROOT / 'tests/fixtures/shop_extras_20261006' / filename
            image = cv2.resize(cv2.imdecode(np.fromfile(path, np.uint8), 1), (2064, 1191), interpolation=cv2.INTER_NEAREST)
            item = identify_extra(offer_labels(image, (x, 955)), 10 if kind == 'voucher' else 4)
            self.assertEqual((item.name, item.kind), (name, kind))

    def test_undiscovered_pack_numbers_distinguish_jumbo_and_mega(self):
        for count, size, name in [(1, 2, 'Buffoon Pack'), (1, 4, 'Jumbo Buffoon Pack'), (2, 4, 'Mega Buffoon Pack')]:
            item = identify_extra(['Not Discovered', f'Choose {count} of up to {size} Joker cards'], 4)
            self.assertEqual(item.name, name)
        with self.assertRaises(ObservationError):
            identify_extra(['Not Discovered', 'Choose of up to Joker cards'], 4)

    def test_identical_voucher_effects_require_owned_prerequisite(self):
        rows = ['Not Discovered', 'Rerolls cost', '$2 less']
        self.assertEqual(identify_extra(rows, 10).name, 'Reroll Surplus')
        self.assertEqual(identify_extra(rows, 10, ('reroll_surplus',)).name, 'Reroll Glut')
        with self.assertRaises(ObservationError):
            identify_extra(rows, 10, ('reroll_surplus', 'reroll_glut'))
        self.assertEqual(identify_extra(['Not Discovered', '+1 hand size'], 10).name, 'Paint Brush')
        self.assertEqual(identify_extra(['Not Discovered', '+1 hand size'], 10, ('paint_brush',)).name, 'Palette')

    def test_voucher_percent_and_interest_numbers_are_not_dropped(self):
        self.assertEqual(identify_extra(['Not Discovered', 'All cards and packs in shop are 25% off'], 10).name, 'Clearance Sale')
        self.assertEqual(identify_extra(['Not Discovered', 'All cards and packs in shop are 50% off'], 10,
                         ('clearance_sale',)).name, 'Liquidation')
        self.assertEqual(identify_extra(['Not Discovered', 'Raise the cap on interest earned in each round to $10'], 10).name, 'Seed Money')
        with self.assertRaises(ObservationError):
            identify_extra(['Not Discovered', 'Raise the cap on interest earned in each round to $20'], 10)

    def test_real_open_arcana_has_priority_over_background_shop_and_reads_choose_one(self):
        from vision import classify_scene
        path = ROOT / 'tests/fixtures/shop_extras_20261006/open-arcana-12-cards.png'
        image = cv2.resize(cv2.imdecode(np.fromfile(path, np.uint8), 1), (2064, 1191), interpolation=cv2.INTER_NEAREST)
        view = read_pack_view(image)
        self.assertEqual((view.kind, view.remaining), ('arcana_pack', 1))
        self.assertEqual(classify_scene(image), 'pack')
        from perception import read_numbers
        values = read_numbers(image)
        self.assertEqual(values['deck_size'] - values['draw_pile_size'], 12)
        from shop_extras import read_pack_hand
        found = read_pack_hand(image, 12)
        self.assertEqual(len(found), 12)
        self.assertTrue(all(390 < entry[2][1] < 560 for entry in found))
        self.assertEqual([c.suit for c, _, _ in found], ['S', 'C', 'S', 'S', 'D', 'D', 'S', 'S', 'H', 'D', 'S', 'H'])

    def test_elevated_pack_targets_require_exact_lift_and_unchanged_other_cards(self):
        from shop_extras import select_pack_targets
        found = [(Card(14, 'S'), .9, (560, 490)), (Card(12, 'C'), .9, (650, 470)),
                 (Card(11, 'S'), .9, (740, 450))]
        selected = [(c, score, (x, y - (40 if i == 1 else 0))) for i, (c, score, (x, y)) in enumerate(found)]
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            with patch('shop_extras.read_pack_hand', return_value=selected):
                select_pack_targets(session, found, (1,))
            self.assertEqual(len(session.taps), 1)
            self.assertTrue(any(e['event'] == 'pack_targets_verified' for e in session.events))
            wrong = [(Card(2, 'H'), .9, (560, 450)), *selected[1:]]
            with patch('shop_extras.read_pack_hand', return_value=wrong):
                with self.assertRaises(ObservationError):
                    select_pack_targets(session, found, (1,))

    def test_real_empress_tooltip_overlapping_raised_hand_remains_verified(self):
        from shop_extras import offer_labels, verified_button
        from perception import shop_offer_item
        for name in ('empress-selected-over-hand', 'empress-hover-over-hand'):
            path = ROOT / 'tests/fixtures/shop_extras_20261006' / (name + '.png')
            image = cv2.resize(cv2.imdecode(np.fromfile(path, np.uint8), 1),
                               (2064, 1191), interpolation=cv2.INTER_NEAREST)
            labels = offer_labels(image, (1132, 857))
            self.assertEqual(shop_offer_item(labels, 0)[0].name, 'The Empress')
            self.assertFalse(any('+4' in line for line in labels))
            if 'selected' in name:
                x, y = verified_button(image, 'Use', (1132, 857))
                self.assertTrue(1050 < x < 1200 and 900 < y < 1000)

    def test_verified_pack_corner_baseline_corrects_provisional_pip_location(self):
        from shop_extras import read_pack_hand, select_pack_targets
        identities = [Card(14, 'C'), Card(13, 'S'), Card(13, 'C'), Card(11, 'S'),
                      Card(11, 'C'), Card(10, 'C'), Card(7, 'D'), Card(2, 'S')]
        images = []
        for name in ('magician-target-baseline', 'magician-target-selected'):
            path = ROOT / 'tests/fixtures/shop_extras_20261006' / (name + '.png')
            images.append(cv2.resize(cv2.imdecode(np.fromfile(path, np.uint8), 1),
                                     (2064, 1191), interpolation=cv2.INTER_NEAREST))
        baseline = read_pack_hand(images[0], 8, identities=identities)
        selected = read_pack_hand(images[1], 8, identities=identities)
        lifts = [a[2][1] - b[2][1] for a, b in zip(baseline, selected)]
        self.assertEqual({i for i, dy in enumerate(lifts) if 28 < dy < 100}, {1, 2})
        self.assertTrue(all(abs(dy) < 20 for i, dy in enumerate(lifts) if i not in (1, 2)))
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            session.frame = images[1]
            select_pack_targets(session, baseline, (1, 2))
            self.assertEqual(len(session.taps), 2)
            self.assertTrue(any(e['event'] == 'pack_targets_verified' for e in session.events))

    def test_real_pack_title_extra_whitespace_does_not_expose_shop_actions(self):
        from shop_extras import shop_ready
        from vision import classify_scene
        path = ROOT / 'tests/fixtures/shop_extras_20261006/arcana-spaced-title.png'
        image = cv2.resize(cv2.imdecode(np.fromfile(path, np.uint8), 1),
                           (2064, 1191), interpolation=cv2.INTER_NEAREST)
        self.assertEqual(read_pack_view(image).kind, 'arcana_pack')
        self.assertFalse(shop_ready(image))
        self.assertEqual(classify_scene(image), 'pack')
        # If a title/count is transiently unreadable, remain blocked from
        # shop actions rather than accepting its still visible background.
        with patch('shop_extras.read_pack_view', return_value=None):
            self.assertEqual(classify_scene(image), 'unknown')
            with tempfile.TemporaryDirectory() as directory:
                session = FakeSession(directory)
                session.frame = image
                with self.assertRaisesRegex(ObservationError, 'Opened pack'):
                    handle_pack(session, image)
                self.assertEqual(session.taps, [])

    def test_shop_exit_requires_actual_two_line_next_round_button(self):
        from shop_extras import shop_ready
        frame = np.zeros((1191, 2064, 3), dtype=np.uint8)
        for words, expected in [
            ([('Next', .99, (761, 504)), ('Round', .99, (761, 539))], True),
            ([('Next', .99, (761, 504)), ('Round', .99, (850, 539))], False),
            ([('SHOP', .99, (761, 504))], False),
            ([('Next', .75, (761, 504)), ('Round', .99, (761, 539))], False)]:
            with patch('perception.read_text', return_value=words):
                self.assertEqual(shop_ready(frame), expected)

    def test_actual_shop_tooltip_covering_exit_is_dismissed_before_scene_choice(self):
        from runner import observe_scene
        from vision import classify_scene
        images = []
        for name in ('shop-exit-covered', 'shop-exit-uncovered'):
            path = ROOT / 'tests/fixtures/shop_extras_20261006' / (name + '.png')
            images.append(cv2.resize(cv2.imdecode(np.fromfile(path, np.uint8), 1),
                                     (2064, 1191), interpolation=cv2.INTER_NEAREST))
        self.assertEqual(classify_scene(images[0]), 'unknown')
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            session.frame = images[1]
            _, scene = observe_scene(session, images[0])
            self.assertEqual(scene, 'shop')
            self.assertEqual(session.taps, [])

    def test_real_gold_and_steel_pack_cards_retain_known_identity_and_corner_gate(self):
        from shop_extras import read_pack_hand
        path = ROOT / 'tests/fixtures/shop_extras_20261006/pack-gold-steel-hand.png'
        image = cv2.resize(cv2.imdecode(np.fromfile(path, np.uint8), 1),
                           (2064, 1191), interpolation=cv2.INTER_NEAREST)
        cards = [Card(13, 'D', enhancement='gold'), Card(12, 'H'), Card(11, 'D'), Card(8, 'C'),
                 Card(7, 'D'), Card(6, 'S'), Card(6, 'H', enhancement='steel'), Card(5, 'C')]
        found = read_pack_hand(image, 8, identities=cards)
        self.assertEqual([c.identity for c, _, _ in found], [c.identity for c in cards])
        self.assertTrue(all(score >= .5 for _, score, _ in found))
        self.assertTrue(440 < found[0][2][1] < 500)

    def test_sale_receipt_dismisses_new_neighbor_tooltip_without_selling_twice(self):
        images = []
        for name in ('sale-receipt-covered', 'sale-receipt-uncovered'):
            path = ROOT / 'tests/fixtures/shop_extras_20261006' / (name + '.png')
            images.append(cv2.resize(cv2.imdecode(np.fromfile(path, np.uint8), 1),
                                     (2064, 1191), interpolation=cv2.INTER_NEAREST))
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            session.frame = images[1]
            begin_action(session, 'sell', {'joker_count': 5, 'money': 28}, name='Smiley Face')
            confirm_pending(session, images[0])
            self.assertIsNone(session.pending_action)
            self.assertEqual(session.taps, [])
            self.assertTrue(any(e['event'] == 'action_confirmed' for e in session.events))


if __name__ == '__main__':
    unittest.main()
