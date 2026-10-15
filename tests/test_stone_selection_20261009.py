"""Actual failed selection frames, including clicks that toggled Stone off."""
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
from controller import selection_flags, select_verified
from recovery import ObservationError
from strategy import Card
from vision import read_hand, stone_position, hand_slots
from tests.test_runtime_recovery import FakeSession

ROOT = Path(__file__).resolve().parents[1]
CARDS = [Card(13, 'H'), Card(13, 'D', debuffed=True), Card(11, 'H'), Card(10, 'S'),
         Card(6, 'H'), Card(6, 'D', debuffed=True), Card(2, 'C'),
         Card(0, '?', enhancement='stone', rankless=True)]

def frame(name):
    path = ROOT / 'tests/fixtures' / f'stone_selection_{name}_20261009.png'
    return cv2.resize(cv2.imdecode(np.fromfile(path, np.uint8), 1), (2064, 1191),
                       interpolation=cv2.INTER_NEAREST)

class StoneSelectionTests(unittest.TestCase):
    def test_real_raised_and_unraised_stone_are_distinct(self):
        for name, expected in [('before', set()), ('first', {0, 1, 7}),
                               ('toggle_off', {0, 1}), ('final', {0, 1, 7})]:
            with self.subTest(frame=name):
                found = read_hand(frame(name), 8, identities=CARDS)
                self.assertEqual(found[7][0], CARDS[7])
                self.assertGreater(found[7][1], .85)
                self.assertEqual(selection_flags(found), expected)

    def test_verified_selection_does_not_toggle_the_selected_stone_again(self):
        found = read_hand(frame('before'), 8, identities=CARDS)
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            session.frame = frame('first')
            select_verified(session, found, [0, 1, 7], 'offline selection replay')
            self.assertEqual(len(session.taps), 3)
            self.assertEqual(session.events[-1]['event'], 'selection_verified')
            self.assertEqual(session.events[-1]['attempts'], 1)
            self.assertFalse(any(e['event'] == 'selection_mismatch' for e in session.events))

    def test_missing_stone_texture_cannot_be_accepted_as_selected(self):
        image = frame('first')
        image[655:895, 1470:1620] = 0
        with self.assertRaises(ObservationError):
            stone_position(image, hand_slots(8)[7])

if __name__ == '__main__':
    unittest.main()
