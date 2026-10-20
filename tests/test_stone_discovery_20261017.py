"""Two Stone cards on opposite ends of a real eight-card hand."""
import hashlib
import json
import unittest
from pathlib import Path

from controller import selection_flags
from strategy import Card
from tools.replay_observation_speed import load_frame
from vision import classify_scene, hand_slots, read_hand, stone_position

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / 'tests/fixtures/stone_discovery_20261017.json').read_text('utf-8'))
IDENTITIES = [Card(0, '?', enhancement='stone', rankless=True), Card(11, 'S'),
              Card(7, 'C'), Card(6, 'H'), Card(4, 'S'), Card(4, 'C'), Card(3, 'H'),
              Card(0, '?', enhancement='stone', rankless=True)]


class TwoStoneDiscoveryTests(unittest.TestCase):
    def test_real_failure_frames_discover_both_stones_without_identity_hints(self):
        for case in FIXTURE['frames']:
            with self.subTest(frame=case['path']):
                self.assertEqual(hashlib.sha256((ROOT / case['path']).read_bytes()).hexdigest(), case['sha256'])
                found = read_hand(load_frame(case['path']), 8, allow_uncertain=True)
                self.assertEqual(len(found), 8)
                self.assertEqual([i for i, (c, _, _) in enumerate(found) if c.is_stone], [0, 7])
                self.assertTrue(all(found[i][1] >= .85 for i in (0, 7)))
                self.assertEqual(selection_flags(found), set())

    def test_confirmed_identities_relocate_both_stones_and_ordinary_cards(self):
        for case in FIXTURE['frames']:
            found = read_hand(load_frame(case['path']), 8, identities=IDENTITIES)
            self.assertEqual([c for c, _, _ in found], IDENTITIES)
            self.assertEqual(selection_flags(found), set())
            self.assertTrue(all(abs(found[i][2][0] - x) <= 25
                                for i, x in enumerate(hand_slots(8))))

    def test_stable_scene_is_hand_instead_of_repeating_unknown(self):
        for case in FIXTURE['frames']:
            self.assertEqual(classify_scene(load_frame(case['path'])), 'hand')

    def test_ordinary_cards_do_not_cross_stone_discovery_threshold(self):
        from recovery import ObservationError
        image = load_frame(FIXTURE['frames'][-1]['path'])
        for x in hand_slots(8)[1:-1]:
            try:
                confidence, _ = stone_position(image, x)
            except ObservationError:
                continue
            self.assertLess(confidence, .85)


if __name__ == '__main__':
    unittest.main()
