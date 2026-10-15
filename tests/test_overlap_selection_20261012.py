"""Real raised-neighbour overlap and verified differential selection."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np

from controller import hand_selection_point, selection_flags, select_verified
from strategy import Card
from vision import read_hand
from tests.test_runtime_recovery import FakeSession

ROOT = Path(__file__).resolve().parents[1]
CARDS = [Card(13, 'C'), Card(12, 'H'), Card(8, 'S'), Card(7, 'S'),
         Card(5, 'S'), Card(4, 'H'), Card(4, 'C'), Card(3, 'D')]


def found_frame():
    path = ROOT / 'tests/fixtures/overlap_selection_20261012/0144-verify-selected-cards.png'
    image = cv2.resize(cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), 1),
                       (2064, 1191), interpolation=cv2.INTER_NEAREST)
    return image, read_hand(image, 8, identities=CARDS)


class OverlapSelectionTests(unittest.TestCase):
    def test_real_failed_frame_proves_missing_right_card(self):
        _, found = found_frame()
        self.assertEqual(selection_flags(found), {0, 1, 5, 6})
        self.assertEqual(found[7][0], Card(3, 'D'))

    def test_new_point_inside_target_but_outside_raised_neighbour(self):
        _, found = found_frame()
        # Polygon outlines inspected on the retained original 1376x794 frame.
        target = np.array([(1008, 480), (1119, 488), (1106, 639), (994, 633)], np.float32)
        neighbour = np.array([(918, 448), (1025, 454), (1030, 600), (915, 595)], np.float32)
        x, y = found[7][2]
        old = ((x + 30) / 1.5, (y + 105) / 1.5)
        tx, ty = hand_selection_point((x, y))
        new = (tx * 1376, ty * 794)
        self.assertGreater(cv2.pointPolygonTest(target, old, False), 0)
        self.assertGreater(cv2.pointPolygonTest(neighbour, old, False), 0)
        self.assertGreater(cv2.pointPolygonTest(target, new, False), 0)
        self.assertLess(cv2.pointPolygonTest(neighbour, new, False), 0)

    def test_only_missing_card_is_toggled_and_result_is_still_verified(self):
        image, found = found_frame()
        raised = list(found)
        c, confidence, (x, y) = raised[7]
        raised[7] = (c, confidence, (x, y - 50))
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            session.frame = image
            with patch('controller.read_hand', return_value=raised):
                select_verified(session, found, [0, 1, 5, 6, 7], 'discard')
            self.assertEqual(len(session.taps), 1)
            self.assertEqual(session.taps[0][:2], hand_selection_point(found[7][2]))
            self.assertEqual(session.events[-1]['event'], 'selection_verified')


if __name__ == '__main__':
    unittest.main()
