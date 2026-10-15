"""Real split-title victory and full-game counting regressions; no device."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from perception import read_text, victory_new_run
from runner import run_batch
from tests.test_runtime_recovery import FakeSession
from vision import classify_scene

ROOT = Path(__file__).resolve().parents[1]


def panel_words():
    return [('YOU', .99, (1144, 225)), ('WIN!', .99, (1445, 226)),
            ('Best Hand', .99, (1153, 332)),
            ('Cards Played', .99, (1124, 482)),
            ('New Run', .99, (1498, 673)),
            ('Endless Mode', .99, (1310, 959))]


class VictoryTests(unittest.TestCase):
    def test_real_victory_frame_classifies_and_exposes_correct_new_run(self):
        p = ROOT / 'tests/fixtures/victory_20261011.png'
        frame = cv2.resize(cv2.imdecode(np.frombuffer(p.read_bytes(), np.uint8), 1),
                           (2064, 1191), interpolation=cv2.INTER_NEAREST)
        words = read_text(frame, (950, 125, 1700, 1100))
        self.assertFalse(any('you win' in text.lower() for text, _, _ in words))
        self.assertEqual(victory_new_run(words), (1498, 673))
        self.assertEqual(classify_scene(frame), 'win')

    def test_single_and_split_titles_both_work(self):
        self.assertEqual(victory_new_run(panel_words()), (1498, 673))
        self.assertEqual(victory_new_run([('YOU WIN!', .99, (1300, 225))]
                                        + panel_words()[2:]), (1498, 673))

    def test_missing_panel_evidence_cannot_authorize_a_victory(self):
        for remove in (0, 1, 2, 3, 4, 5):
            with self.subTest(remove=remove):
                self.assertIsNone(victory_new_run([v for i, v in enumerate(panel_words())
                                                   if i != remove]))

    def test_wrong_row_or_low_confidence_title_is_rejected(self):
        for replacement in [('WIN!', .99, (1445, 400)),
                            ('WIN!', .70, (1445, 226)),
                            ('OVER', .99, (1445, 226))]:
            self.assertIsNone(victory_new_run([panel_words()[0], replacement]
                                             + panel_words()[2:]))

    def test_duplicate_new_run_buttons_are_ambiguous(self):
        self.assertIsNone(victory_new_run(panel_words() + [('New Run', .99, (1530, 720))]))

    def test_repeated_victory_only_counts_once_before_next_game(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            scenes = ['deck', 'win', 'win', 'deck', 'loss']
            with patch('runner.prepare_new_run', return_value=(.5, .79)), \
                 patch('runner.classify_scene', side_effect=scenes), \
                 patch('runner.scene_scores', return_value={}), \
                 patch('runner.read_text', return_value=panel_words()), \
                 patch('runner.read_numbers', return_value={}), \
                 patch('runner.time.sleep'):
                run_batch(session, 2, max_actions=10)
            ends = [e for e in session.events if e['event'] == 'game_end']
            self.assertEqual([(e['number'], e['result']) for e in ends], [(1, 'win'), (2, 'loss')])
            victory_taps = [t for t in session.taps if t[2] == 'open next full game after victory']
            self.assertTrue(victory_taps)
            self.assertEqual(victory_taps[0][:2], (1498 / 2064, 673 / 1191))

    def test_old_victory_is_not_counted_in_a_new_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            with patch('runner.prepare_new_run', return_value=(.5, .79)), \
                 patch('runner.classify_scene', side_effect=['win', 'deck', 'win']), \
                 patch('runner.scene_scores', return_value={}), \
                 patch('runner.read_text', return_value=panel_words()), \
                 patch('runner.read_numbers', return_value={}), \
                 patch('runner.time.sleep'):
                run_batch(session, 1, max_actions=10)
            ends = [e for e in session.events if e['event'] == 'game_end']
            self.assertEqual([(e['number'], e['result']) for e in ends], [(1, 'win')])
            self.assertEqual(session.taps[0][:2], (1498 / 2064, 673 / 1191))


if __name__ == '__main__':
    unittest.main()
