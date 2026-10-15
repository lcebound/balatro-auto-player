"""Normal eight-card hands use image proof; invisible modifiers stay guarded."""
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

from controller import collect_cards
from plain_hand import read_plain_hand, plain_deck_identities, mark_unverified_chips
from recovery import begin_action, finish_action, state_from_dict
from strategy import Card, Joker
from tests.test_runtime_recovery import FakeSession

FIXTURES = Path(__file__).parent / 'fixtures'


class PlainHandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((FIXTURES / 'cache_hand_20261010.json').read_text(encoding='utf-8'))
        cls.state = state_from_dict(cls.data['state'])
        cls.frames = {tag: cv2.imdecode(np.frombuffer(
            (FIXTURES / f'cache_hand_{tag}_20261010.png').read_bytes(), np.uint8), 1)
            for tag in ('before', 'after')}
        cls.cards = {tag: tuple(Card(**c) for c in cls.data[tag]['cards']) for tag in cls.frames}

    def test_both_saved_eight_card_hands_are_identified_with_zero_tooltips(self):
        for tag, frame in self.frames.items():
            with self.subTest(tag=tag):
                result = read_plain_hand(frame, self.state, 8, 8)
                self.assertTrue(all(result))
                self.assertEqual(tuple(value[0] for value in result), self.cards[tag])
                self.assertTrue(all(value[1] >= .94 for value in result))

    def test_controller_skips_hover_and_ocr_for_a_fresh_eight_card_plain_hand(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            session.policy_state = self.state
            session.hover = Mock()
            with patch('controller.hand_counter', return_value=(8, 8)), \
                 patch('perception.hand_counter', return_value=(8, 8)), \
                 patch('controller.read_text') as ocr, patch('controller.parse_card_tooltip') as tooltip:
                result = collect_cards(session, self.frames['before'], 8)
            self.assertEqual(tuple(c for c, _, _ in result), self.cards['before'])
            session.hover.assert_not_called()
            ocr.assert_not_called()
            tooltip.assert_not_called()
            self.assertEqual(session.events[-1]['plain_slots'], list(range(8)))
            self.assertEqual(session.events[-1]['refreshed_slots'], [])

    def test_mixed_hand_reads_only_the_card_with_a_known_hidden_seal(self):
        original = self.cards['before'][0]
        special = replace(original, seal='red')
        deck = tuple(special if c.identity == original.identity else c for c in self.state.deck_cards)
        state = replace(self.state, deck_cards=deck)
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            session.policy_state = state
            session.frame = self.frames['before']
            with patch('controller.hand_counter', return_value=(8, 8)), \
                 patch('perception.hand_counter', return_value=(8, 8)), \
                 patch('controller.read_text', return_value=[]), \
                 patch('controller.parse_card_tooltip', return_value=special) as tooltip:
                result = collect_cards(session, session.frame, 8)
            tooltip.assert_called_once()
            self.assertEqual(result[0][0], special)
            self.assertEqual(session.events[-1]['plain_slots'], list(range(1, 8)))
            self.assertEqual(session.events[-1]['refreshed_slots'], [0])

    def test_all_special_attributes_block_plain_shortcut(self):
        original = self.cards['before'][0]
        for attributes in ({'enhancement': 'bonus'}, {'enhancement': 'glass'}, {'edition': 'foil'},
                           {'seal': 'red'}, {'bonus_chips': 5}, {'debuffed': True}, {'face_down': True}):
            special = replace(original, **attributes)
            deck = tuple(special if c.identity == original.identity else c for c in self.state.deck_cards)
            with self.subTest(attributes=attributes):
                self.assertNotIn(original.identity, plain_deck_identities(replace(self.state, deck_cards=deck), 8, 8))

    def test_duplicates_unknown_deck_and_nonstandard_capacity_use_conservative_reader(self):
        duplicate = replace(self.cards['before'][0], seal='blue')
        state = replace(self.state, deck_cards=self.state.deck_cards + (duplicate,), deck_size=53)
        self.assertNotIn(duplicate.identity, plain_deck_identities(state, 8, 8))
        for state, count, limit in ((replace(self.state, deck_cards=()), 8, 8),
                                    (self.state, 13, 13), (self.state, 8, 13)):
            self.assertEqual(plain_deck_identities(state, count, limit), {})

    def test_card_debuff_and_flip_bosses_block_direct_plain_proof(self):
        for boss in ('The Plant', 'The Pillar', 'The Fish', 'The Mark', 'The Verdant Leaf'):
            state = replace(self.state, boss=boss)
            self.assertEqual(plain_deck_identities(state, 8, 8), {})
            self.assertTrue(plain_deck_identities(replace(state, boss_disabled=True), 8, 8))

    def test_hiker_marks_only_played_identities_and_persists_after_selling(self):
        hiker = replace(self.state, jokers=(Joker('Hiker'),))
        state = mark_unverified_chips(hiker, self.cards['before'][:2])
        self.assertEqual(set(state.unverified_chip_cards), {c.identity for c in self.cards['before'][:2]})
        state = replace(state, jokers=(), blind_index=1)
        allowed = plain_deck_identities(state, 8, 8)
        self.assertNotIn(self.cards['before'][0].identity, allowed)
        self.assertIn(self.cards['before'][2].identity, allowed)
        result = read_plain_hand(self.frames['before'], state, 8, 8)
        self.assertIsNone(result[0])
        self.assertIsNone(result[1])
        self.assertTrue(all(result[2:]))

    def test_hiker_uncertainty_survives_action_journal_and_checkpoint(self):
        state = mark_unverified_chips(replace(self.state, jokers=(Joker('Hiker'),)), self.cards['before'][:1])
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            begin_action(session, 'play', {}, after_state=state)
            finish_action(session, True)
            reloaded = state_from_dict(json.loads((Path(directory) / 'policy-state.json').read_text(encoding='utf-8')))
            self.assertEqual(reloaded.unverified_chip_cards, state.unverified_chip_cards)
            self.assertEqual(reloaded, state)

    def test_legacy_checkpoint_does_not_claim_invisible_chip_provenance(self):
        old = asdict(replace(self.state, total_hands_played=4))
        del old['unverified_chip_cards']
        state = state_from_dict(old)
        self.assertEqual(plain_deck_identities(state, 8, 8), {})
        self.assertEqual(len(state.unverified_chip_cards), len(state.deck_cards))

    def test_strength_keeps_invisible_chip_uncertainty_when_rank_changes(self):
        from consumables import Consumable, ConsumableDecision, state_after_consumable
        card = Card(9, 'S', debuffed=True)
        # Make the resulting 10 of Spades unique, so duplicate detection alone
        # cannot protect against the hidden chip value moving to a new rank.
        deck = tuple(c for c in self.state.deck_cards if c.identity != (10, 'S'))
        state = replace(self.state, deck_cards=deck, deck_size=51,
                        unverified_chip_cards=((9, 'S'),))
        after = state_after_consumable(state, [card], Consumable('Strength', 'Tarot'),
                                       ConsumableDecision('use', indices=(0,)))
        self.assertEqual(len(after.deck_cards), 51)
        self.assertIn((10, 'S'), after.unverified_chip_cards)
        self.assertNotIn((10, 'S'), plain_deck_identities(after, 8, 8))

    def test_empty_and_obstructed_pixels_cannot_authorize_plain_cards(self):
        frame = np.zeros_like(self.frames['before'])
        self.assertEqual(read_plain_hand(frame, self.state, 8, 8), [None] * 8)


if __name__ == '__main__':
    unittest.main()
