"""Offline cache proofs and action lifecycle, including real retained cards."""
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from controller import collect_cards, collect_jokers
from hand_cache import HandCache, card_patch
from joker_cache import JokerCache, dirty_jokers, roster_patch, round_token
from perception import joker_slots
from recovery import begin_action, finish_action, state_from_dict
from strategy import Card, Joker, RunState
from tests.test_runtime_recovery import FakeSession
from vision import read_hand, hand_slots

FIXTURES = Path(__file__).parent / 'fixtures'


class JokerCacheLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.random.default_rng(94).integers(0, 256, (1191, 2064, 3), dtype=np.uint8)
        self.state = RunState(jokers=(Joker('Scholar'), Joker('Square Joker', 8)), ante=2, blind_index=0)
        self.cache = JokerCache(round_token(self.state), 5, self.state.jokers,
            tuple(roster_patch(self.frame, 2, i) for i in range(2)), frozenset())

    def action(self, session, kind, **details):
        begin_action(session, kind, {}, **details)
        finish_action(session, True)

    def test_planet_purchase_does_not_hover_owned_jokers_including_growth_counter(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            session.joker_cache = self.cache
            self.action(session, 'buy', reward_kind='planet', name='Jupiter')
            with patch('controller.joker_counter', return_value=(2, 5)), \
                 patch('controller.read_jokers') as atlas, patch('controller.read_text') as ocr:
                result = collect_jokers(session, self.frame, self.state)
            self.assertEqual(result.jokers, self.state.jokers)
            ocr.assert_not_called()
            atlas.assert_not_called()
            self.assertEqual(session.events[-1]['refreshed_slots'], [])

    def test_added_joker_only_reads_new_slot_with_existing_cards_recentered(self):
        frame = self.frame.copy()
        for index, old in ((0, 0), (1, 1)):
            x = joker_slots(3)[index]
            frame[125:310, x - 55:x + 55] = self.cache.patches[old]
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            session.joker_cache = self.cache
            self.action(session, 'buy', reward_kind='joker', name='Juggler')
            with patch('controller.joker_counter', return_value=(3, 6)), \
                 patch('controller.read_jokers') as atlas, patch('controller.read_text', return_value=[]), \
                 patch('controller.identify_joker_text', return_value=('juggler', 1)), \
                 patch('controller.joker_from_definition', return_value=Joker('Juggler')), \
                 patch('controller.parse_joker_tooltip', return_value=Joker('Juggler')) as tooltip:
                result = collect_jokers(session, frame, self.state)
            tooltip.assert_called_once()
            atlas.assert_not_called()
            self.assertEqual(result.jokers, self.state.jokers + (Joker('Juggler'),))
            self.assertEqual(session.events[-1]['refreshed_slots'], [2])

    def test_sell_maps_remaining_card_after_layout_change(self):
        frame = np.zeros_like(self.frame)
        x = joker_slots(1)[0]
        frame[125:310, x - 55:x + 55] = self.cache.patches[0]
        self.assertEqual(self.cache.reconciled_slots(frame, self.state, 1), (0,))

    def test_ambiguous_duplicate_jokers_with_different_values_are_not_reused(self):
        patch = self.cache.patches[0]
        cache = replace(self.cache, jokers=(Joker('Square Joker', 4), Joker('Square Joker', 8)), patches=(patch, patch))
        frame = self.frame.copy()
        for x in joker_slots(2):
            frame[125:310, x - 55:x + 55] = patch
        self.assertEqual(cache.reconciled_slots(frame, self.state, 2), (None, None))

    def test_confirmed_play_refreshes_growth_counter_and_unconfirmed_action_does_not(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            session.joker_cache = self.cache
            begin_action(session, 'play', {})
            finish_action(session, False)
            self.assertFalse(session.joker_cache.needs_tooltip(1, self.state))
            self.action(session, 'play')
            self.assertTrue(session.joker_cache.needs_tooltip(1, self.state))
            self.assertFalse(session.joker_cache.needs_tooltip(0, self.state))

    def test_static_card_survives_ordinary_boundary_but_targets_perishables_and_boss_changes_refresh(self):
        next_state = replace(self.state, blind_index=1)
        self.assertFalse(self.cache.needs_tooltip(0, next_state))
        self.assertTrue(self.cache.needs_tooltip(1, next_state))
        for joker in (Joker('Ancient Joker'), Joker('Scholar', data=(('perishable', 1),))):
            cache = replace(self.cache, jokers=(joker, self.state.jokers[1]))
            self.assertTrue(cache.needs_tooltip(0, next_state))
        self.assertTrue(self.cache.needs_tooltip(0, replace(next_state, boss='The Plant')))

    def test_gift_card_cashout_refreshes_other_sell_values_and_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            session.joker_cache = replace(self.cache, jokers=(Joker('Scholar'), Joker('Gift Card')))
            dirty_jokers(session, round_end=True)
            self.assertTrue(session.joker_cache.needs_tooltip(0, self.state))
            session.joker_cache = replace(self.cache, jokers=(Joker('Ancient Joker'), Joker('Scholar')))
            dirty_jokers(session, round_end=True)
            self.assertTrue(session.joker_cache.needs_tooltip(0, self.state))
            self.assertFalse(session.joker_cache.needs_tooltip(1, self.state))

    def test_skip_pack_refreshes_red_card_and_reroll_refreshes_only_flash_card(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            session.joker_cache = replace(self.cache, jokers=(Joker('Red Card', 3), Joker('Flash Card', 4)))
            self.action(session, 'skip_pack', pack_kind='arcana')
            self.assertEqual(session.joker_cache.dirty, frozenset({'red_card'}))
            session.joker_cache = replace(session.joker_cache, dirty=frozenset())
            self.action(session, 'reroll', price=5)
            self.assertEqual(session.joker_cache.dirty, frozenset({'flash_card'}))

    def test_planet_consumption_refreshes_constellation_without_square_and_hex_clears_roster(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            session.joker_cache = replace(self.cache, jokers=(Joker('Constellation', 1.2), Joker('Square Joker', 8)))
            self.action(session, 'consume', name='Jupiter', consumable_kind='planet')
            self.assertTrue(session.joker_cache.needs_tooltip(0, self.state))
            self.assertFalse(session.joker_cache.needs_tooltip(1, self.state))
            self.action(session, 'consume', name='Hex', consumable_kind='spectral')
            self.assertIsNone(session.joker_cache)


class RetainedHandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((FIXTURES / 'cache_hand_20261010.json').read_text(encoding='utf-8'))
        cls.state = state_from_dict(cls.data['state'])
        cls.before_cards = tuple(Card(**c) for c in cls.data['before']['cards'])
        cls.after_cards = tuple(Card(**c) for c in cls.data['after']['cards'])
        cls.before = cv2.imdecode(np.frombuffer((FIXTURES / 'cache_hand_before_20261010.png').read_bytes(), np.uint8), 1)
        cls.after = cv2.imdecode(np.frombuffer((FIXTURES / 'cache_hand_after_20261010.png').read_bytes(), np.uint8), 1)
        cls.cache = HandCache(round_token(cls.state), cls.before_cards,
            tuple(card_patch(cls.before, pos) for pos in cls.data['before']['positions']))
        cls.retained = tuple(c for i, c in enumerate(cls.before_cards) if i not in cls.data['played_indices'])

    def test_saved_play_reuses_all_three_retained_cards_and_does_not_guess_new_draws(self):
        hints = self.cache.retain(self.retained).verified_identities(self.after, self.state, 8, 8)
        self.assertEqual([i for i, c in enumerate(hints) if c], [0, 2, 3])
        for i, c in enumerate(hints):
            if c:
                self.assertEqual(c, self.after_cards[i])

    def test_collection_reads_five_new_cards_instead_of_all_eight(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            session.hand_cache = self.cache.retain(self.retained)
            session.policy_state = self.state
            session.frame = self.after
            found = [(c, .99, tuple(p)) for c, p in zip(self.after_cards, self.data['after']['positions'])]
            new_cards = [self.after_cards[i] for i in (1, 4, 5, 6, 7)]
            with patch('controller.hand_counter', return_value=(8, 8)), \
                 patch('plain_hand.read_plain_hand', return_value=[None] * 8), \
                 patch('controller.read_hand', return_value=found), patch('controller.read_text', return_value=[]), \
                 patch('controller.parse_card_tooltip', side_effect=new_cards) as tooltip:
                result = collect_cards(session, self.after, 8)
            self.assertEqual(tuple(c for c, _, _ in result), self.after_cards)
            self.assertEqual(tooltip.call_count, 5)
            self.assertEqual(session.events[-1]['reused_slots'], [0, 2, 3])

    def test_partial_identity_hints_work_with_real_corner_reader(self):
        hints = self.cache.retain(self.retained).verified_identities(self.after, self.state, 8, 8)
        with patch('perception.hand_counter', return_value=(8, 8)):
            found = read_hand(self.after, 8, identities=hints, allow_uncertain=True)
        self.assertEqual([c.identity for c, _, _ in found], [c.identity for c in self.after_cards])

    def test_unknown_deck_duplicate_identity_and_new_blind_cannot_reuse_hidden_metadata(self):
        cache = self.cache.retain(self.retained)
        for state in (replace(self.state, deck_cards=()), replace(self.state, blind_index=1)):
            self.assertFalse(any(cache.verified_identities(self.after, state, 8, 8)))
        # A new duplicate could share the corner while hiding another seal.
        duplicate = replace(self.retained[0], seal='red')
        state = replace(self.state, deck_cards=self.state.deck_cards + (duplicate,), deck_size=53)
        hints = cache.verified_identities(self.after, state, 8, 8)
        self.assertIsNone(hints[0])

    def test_confirmed_action_retains_only_survivors_and_card_modification_clears_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            session.hand_cache = self.cache
            begin_action(session, 'play', {}, retained_cards=[asdict(c) for c in self.retained])
            finish_action(session, True)
            self.assertEqual(session.hand_cache.cards, self.retained)
            begin_action(session, 'consume', {}, name='Jupiter', consumable_kind='planet')
            finish_action(session, True)
            self.assertIsNotNone(session.hand_cache)
            begin_action(session, 'consume', {}, name='The Empress', consumable_kind='tarot')
            finish_action(session, True)
            self.assertIsNone(session.hand_cache)

    def test_changed_card_face_cannot_keep_a_cached_enhancement_or_debuff_status(self):
        changed = self.after.copy()
        changed[655:975, 520:1770] = 0
        self.assertFalse(any(self.cache.retain(self.retained).verified_identities(changed, self.state, 8, 8)))

    def test_thirteen_card_capacity_reuses_only_confirmed_retained_cards(self):
        cards = tuple(Card(rank, 'H') for rank in range(2, 15))
        state = replace(self.state, hand_size=13, deck_size=13, deck_cards=cards)
        patches = tuple(np.random.default_rng(i).integers(0, 256, (135, 44, 3), dtype=np.uint8)
                        for i in range(13))
        image = np.zeros_like(self.after)
        for x, patch_image in zip(hand_slots(13, 13), patches):
            image[735:870, x + 5:x + 49] = patch_image
        cache = HandCache(round_token(state), cards, patches).retain(cards[:4])
        hints = cache.verified_identities(image, state, 13, 13)
        self.assertEqual(hints[:4], list(cards[:4]))
        self.assertEqual(hints[4:], [None] * 9)
        self.assertFalse(any(cache.verified_identities(image, state, 20, 20)))


if __name__ == '__main__':
    unittest.main()
