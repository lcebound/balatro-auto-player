"""The real final-boss frame must progress without reading face-down titles."""
import json
import tempfile
import unittest
from dataclasses import replace
from itertools import permutations
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from controller import collect_jokers, collect_position, apply_joker_order
from recovery import state_from_dict
from strategy import Card, Joker, RunState, decide_play, score_hand, optimize_joker_order
from tests.test_runtime_recovery import FakeSession

ROOT = Path(__file__).resolve().parents[1]


class AmberAcornTests(unittest.TestCase):
    def test_real_boss_frame_collects_position_without_hovering_hidden_jokers(self):
        path = ROOT / 'tests/fixtures/amber_acorn_hidden_20261009.png'
        image = cv2.resize(cv2.imdecode(np.fromfile(path, np.uint8), 1),
                           (2064, 1191), interpolation=cv2.INTER_NEAREST)
        previous = state_from_dict(json.loads(
            (ROOT / 'tests/fixtures/amber_acorn_roster_20261009.json').read_text('utf-8')))
        # The retained roster predates explicit level-cache initialization.
        # This case tests hidden Joker observation, not a Run Info panel.
        previous = replace(previous, levels_initialized=True)
        cards = [Card(11, 'H'), Card(10, 'H'), Card(9, 'D'), Card(9, 'D'),
                 Card(6, 'C'), Card(4, 'D'), Card(4, 'D'), Card(3, 'D')]
        found = [(c, 1., (600 + i * 100, 725)) for i, c in enumerate(cards)]
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            session.frame = image
            with patch('controller.read_jokers') as atlas, \
                 patch('controller.collect_cards', return_value=found):
                actual, read = collect_position(session, previous)
            atlas.assert_not_called()
            self.assertEqual(actual.boss, 'Amber Acorn')
            self.assertEqual(actual.target_score, 100000)
            self.assertEqual(actual.jokers, previous.jokers)
            self.assertTrue(actual.joker_order_unknown)
            self.assertEqual(read, found)
            self.assertFalse(session.taps)
            decision = decide_play(cards, actual)
            self.assertGreater(decision.estimate.score, 0)
            self.assertTrue(any('order unknown' in n for n in decision.estimate.notes))

    def test_unknown_order_bound_never_exceeds_any_actual_order(self):
        scenarios = [
            ([Card(10, 'S'), Card(10, 'H')], [Card(2, 'D')],
             (Joker('Joker'), Joker('Card Sharp', edition='holographic'),
              Joker('Popcorn', 15), Joker('Blue Joker'))),
            ([Card(13, 'S')], [Card(13, 'H'), Card(12, 'D')],
             (Joker('Smiley Face'), Joker('Triboulet'), Joker('Photograph'),
              Joker('Baron'), Joker('Shoot the Moon'), Joker('Raised Fist'))),
            ([Card(14, 'S'), Card(14, 'H')], [Card(2, 'C')],
             (Joker('Blueprint', edition='polychrome'), Joker('Brainstorm'),
              Joker('Scholar'), Joker('Card Sharp'))),
        ]
        for cards, held, jokers in scenarios:
            state = RunState(boss='Amber Acorn', jokers=jokers,
                             round_played_types={'pair': 1}, draw_pile_size=35)
            bound = score_hand(cards, replace(state, joker_order_unknown=True), held).score
            for order in permutations(jokers):
                with self.subTest(keys=tuple(j.key for j in order)):
                    actual = score_hand(cards, replace(state, jokers=order), held).score
                    self.assertLessEqual(bound, actual + 1e-7)

    def test_hidden_count_change_does_not_guess_the_removed_joker(self):
        state = RunState(boss='Amber Acorn', jokers=(Joker('Ice Cream', 5), Joker('Card Sharp')))
        frame = np.zeros((1191, 2064, 3), np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            with patch('controller.joker_counter', side_effect=[(2, 5), (1, 5)]), \
                 patch('controller.read_jokers') as atlas:
                first = collect_jokers(session, frame, state)
                changed = collect_jokers(session, frame, first)
            atlas.assert_not_called()
            self.assertEqual(len(changed.jokers), 1)
            self.assertFalse(changed.has('card_sharp'))
            self.assertFalse(changed.has('ice_cream'))
            self.assertFalse(session.events[-1]['roster_known'])

    def test_hidden_mutable_values_do_not_repeatedly_subtract_from_old_estimates(self):
        state = RunState(boss='Amber Acorn', jokers=(
            Joker('Ice Cream', 100, data=(('observed_value', 1),)), Joker('Ramen', 1.9),
            Joker('Green Joker', 25), Joker('Loyalty Card', data=(('remaining', 0),))))
        frame = np.zeros((1191, 2064, 3), np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            with patch('controller.joker_counter', return_value=(4, 5)):
                first = collect_jokers(session, frame, replace(state, round_played_types={'pair': 2}))
                second = collect_jokers(session, frame, replace(first, round_played_types={'pair': 3}))
            self.assertEqual(first.jokers[0].value, 90)
            self.assertEqual(second.jokers[0].value, 85)
            self.assertEqual(second.jokers[1].value, 1)
            self.assertEqual(second.jokers[2].value, 0)
            self.assertEqual(second.jokers[3].get('remaining'), -1)

    def test_unknown_order_cannot_authorize_a_drag(self):
        cards = [Card(10, 'S'), Card(10, 'H')]
        state = RunState(boss='Amber Acorn', joker_order_unknown=True,
                         jokers=(Joker('Card Sharp'), Joker('Joker')))
        decision = decide_play(cards, state)
        self.assertEqual(optimize_joker_order(cards, state, decision)[0], (0, 1))
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            session.drag = lambda *args: self.fail('must not drag hidden identities')
            self.assertEqual(apply_joker_order(session, state, (1, 0)), state)

    def test_disabled_boss_still_reads_visible_roster_and_restores_known_order(self):
        state = RunState(boss='Amber Acorn', boss_disabled=True,
                         joker_order_unknown=True, jokers=(Joker('Joker'),))
        frame = np.zeros((1191, 2064, 3), np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            with patch('controller.joker_counter', return_value=(1, 5)), \
                 patch('controller.read_jokers', return_value=[(1., 'joker', (1020, 220))]), \
                 patch('controller.read_text', return_value=[]), \
                 patch('controller.identify_joker_text', return_value=('joker', 1)), \
                 patch('controller.parse_joker_tooltip', return_value=Joker('Joker')):
                actual = collect_jokers(session, frame, state)
            self.assertFalse(actual.joker_order_unknown)
            self.assertEqual(actual.score_confidence, 'observed')


if __name__ == '__main__':
    unittest.main()
