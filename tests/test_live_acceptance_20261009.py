"""Regressions from supervised games; never create a real device Session."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from recovery import DesktopUnavailableError, run_info_back
from runner import observe_scene, run_batch
from tests.test_runtime_recovery import FakeSession
from perception import (read_text, joker_tooltip_words, tooltip_labels, tooltip_title,
                        identify_joker_text, joker_from_definition, parse_joker_tooltip,
                        shop_tooltip_region, tooltip_rows, shop_offer_item, identify_consumable_text)

ROOT = Path(__file__).resolve().parents[1]


class LiveAcceptanceTests(unittest.TestCase):
    def test_fool_main_panel_is_identified_before_a_referenced_planet_is_bought(self):
        from recovery import ObservationError
        for name in ('fool_nested_planet_offer_20261009.png',
                     'fool_nested_planet_selected_20261009.png'):
            image = cv2.resize(cv2.imdecode(np.fromfile(ROOT / 'tests/fixtures' / name, np.uint8), 1),
                               (2064, 1191), interpolation=cv2.INTER_NEAREST)
            labels = tooltip_rows(read_text(image, shop_tooltip_region(image, 1353)))
            self.assertEqual(labels[0], 'The Fool')
            self.assertIn('Tarot', labels)
            self.assertEqual(identify_consumable_text(labels, title=labels[0])[0], 'the_fool')
            with self.assertRaisesRegex(ObservationError, 'unsupported use workflow: The Fool'):
                shop_offer_item(labels, 3)

    def test_a_referenced_planet_does_not_override_a_complete_fool_effect(self):
        labels = ['Creates the last', 'Tarot or Planet card', 'used during this run',
                  'The Fool excluded', 'Uranus']
        self.assertEqual(identify_consumable_text(labels)[0], 'the_fool')

    def test_shop_card_name_outside_hover_tooltip_cannot_change_roster_identity(self):
        path = ROOT / 'tests/fixtures/roster_tooltip_shop_collision_20261009.png'
        image = cv2.resize(cv2.imdecode(np.fromfile(path, np.uint8), 1),
                           (2064, 1191), interpolation=cv2.INTER_NEAREST)
        words = read_text(image, (525, 330, 2020, 850))
        from recovery import ObservationError
        with self.assertRaises(ObservationError):
            identify_joker_text(tooltip_labels(words), title=tooltip_title(words))
        main, versions = joker_tooltip_words(words, 844)
        key, _ = identify_joker_text(tooltip_labels(main), title=tooltip_title(main))
        self.assertEqual(key, 'shoot_the_moon')
        self.assertFalse(versions)
        self.assertFalse(any('loyalty' in label.lower() for label in tooltip_labels(main)))

    def test_scoped_real_roster_tooltip_preserves_holographic_edition(self):
        path = ROOT / 'tests/fixtures/roster_tooltip_holographic_20261009.png'
        image = cv2.resize(cv2.imdecode(np.fromfile(path, np.uint8), 1),
                           (2064, 1191), interpolation=cv2.INTER_NEAREST)
        main, versions = joker_tooltip_words(read_text(image, (525, 330, 2020, 850)), 1139)
        labels = tooltip_labels(main)
        key, _ = identify_joker_text(labels, title=tooltip_title(main))
        actual = parse_joker_tooltip(joker_from_definition(key), labels + [w[0] for w in versions])
        self.assertEqual(actual.key, 'card_sharp')
        self.assertEqual(actual.edition, 'holographic')

    def test_edition_help_amount_does_not_replace_main_popcorn_value(self):
        words = [('Popcorn', .99, (1000, 380)), ('+8 Mult', .99, (1000, 425)),
                 ('Common', .99, (1000, 550)), ('Holographic', .99, (650, 400)),
                 ('+10 Mult', .99, (650, 450)), ('Loyalty Card', .99, (1250, 600))]
        main, versions = joker_tooltip_words(words, 1000)
        labels = tooltip_labels(main)
        key, _ = identify_joker_text(labels, title=tooltip_title(main))
        actual = parse_joker_tooltip(joker_from_definition(key), labels + [w[0] for w in versions],
                                    effect_lines=labels)
        self.assertEqual(actual.value, 8)
        self.assertEqual(actual.edition, 'holographic')

    def test_focus_interruption_reconciles_overlay_without_counting_a_game_error(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            session.focus = lambda: None
            with patch('runner.prepare_new_run', return_value=(.5, .79)), \
                 patch('runner.classify_scene', side_effect=['deck', 'hand', 'unknown', 'hand', 'loss']), \
                 patch('runner.scene_scores', return_value={}), \
                 patch('runner.run_info_back', return_value=(1031, 1029)), \
                 patch('runner.time.sleep'), \
                 patch('runner.play_one_hand', side_effect=[DesktopUnavailableError('focus denied'), None]):
                run_batch(session, 1, max_actions=10)
            self.assertEqual(len([e for e in session.events if e['event'] == 'game_end']), 1)
            self.assertFalse(any(e['event'] == 'automatic_recovery' for e in session.events))
            self.assertTrue(any('Run Info before scene' in tap[2] for tap in session.taps))

    def test_interrupted_real_run_info_is_closed_by_the_runner(self):
        path = ROOT / 'tests/fixtures/run_info_interrupted_20261009.png'
        image = cv2.resize(cv2.imdecode(np.fromfile(path, np.uint8), 1),
                           (2064, 1191), interpolation=cv2.INTER_NEAREST)
        back = run_info_back(image)
        self.assertIsNotNone(back)
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            with patch('runner.classify_scene', side_effect=['unknown', 'hand']):
                _, scene = observe_scene(session, image)
            self.assertEqual(scene, 'hand')
            self.assertEqual(len(session.taps), 1)
            self.assertTrue(session.refresh_levels)

    def test_an_arbitrary_back_button_is_not_run_info(self):
        image = np.zeros((1191, 2064, 3), np.uint8)
        for tabs in [[('Back', .99, (1030, 1030))],
                     [('Poker Hands', .99, (800, 200)), ('Blinds', .99, (1000, 200))]]:
            with patch('perception.read_text', return_value=tabs):
                self.assertIsNone(run_info_back(image))


if __name__ == '__main__':
    unittest.main()
