"""The Fish's seven-hand counter: real missed frames and rejecting ambiguity."""
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from perception import read_numbers, verified_resource_counter

ROOT = Path(__file__).resolve().parents[1]


def frame(name='1102-decision-state.png'):
    path = ROOT / 'tests/fixtures/resource_counter_20261013' / name
    return cv2.resize(cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), 1),
                       (2064, 1191), interpolation=cv2.INTER_NEAREST)


def word(text, score=.999):
    return [(text, score, (48, 34))]


class ResourceCounterTests(unittest.TestCase):
    def test_three_failed_runtime_frames_now_prove_seven_hands(self):
        for name in ('1102-decision-state.png', '1106-decision-state.png',
                     '1107-automatic-recovery.png'):
            with self.subTest(name=name):
                image = frame(name)
                self.assertEqual(verified_resource_counter(image, 'hands_left'), 7)
                self.assertEqual(read_numbers(image)['hands_left'], 7)
                self.assertEqual(read_numbers(image)['discards_left'], 0)

    def test_existing_good_observation_is_accepted_without_extra_ocr(self):
        with patch('perception.verified_resource_counter', side_effect=AssertionError('unneeded reread')):
            values = read_numbers(frame('1109-decision-state.png'))
        self.assertEqual(values['hands_left'], 7)
        self.assertEqual(values['discards_left'], 0)

    def test_zero_and_thirty_are_real_resource_values(self):
        for value in (0, 30):
            with patch('perception.read_text', return_value=word(str(value))):
                self.assertEqual(verified_resource_counter(frame(), 'hands_left'), value)

    def test_one_confident_read_does_not_fill_missing_counter(self):
        with patch('perception.read_text', side_effect=[word('7'), [], []]):
            self.assertIsNone(verified_resource_counter(frame(), 'hands_left'))

    def test_conflict_rejected_even_with_two_agreeing_views(self):
        with patch('perception.read_text', side_effect=[word('7'), word('7'), word('1')]):
            self.assertIsNone(verified_resource_counter(frame(), 'hands_left'))

    def test_weak_results_are_not_promoted_to_proof(self):
        with patch('perception.read_text', return_value=word('7', .94)):
            self.assertIsNone(verified_resource_counter(frame(), 'hands_left'))

    def test_out_of_range_or_other_labels_rejected(self):
        for text in ('31', '99', '-1', 'Hands 7', '7/9', 'I'):
            with patch('perception.read_text', return_value=word(text)):
                self.assertIsNone(verified_resource_counter(frame(), 'hands_left'))

    def test_hidden_counter_is_not_guessed_from_the_burglar_or_hand_capacity(self):
        image = frame()
        image[815:884, 295:392] = 0
        self.assertIsNone(verified_resource_counter(image, 'hands_left'))
        self.assertNotIn('hands_left', read_numbers(image))


if __name__ == '__main__':
    unittest.main()
