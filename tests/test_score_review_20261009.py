"""Offline tests from score residuals and corrupted live multiplier OCR."""
from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from perception import (read_text, joker_tooltip_words, parse_joker_tooltip,
                        joker_from_definition, tooltip_labels, verify_current_multiplier)
from strategy import Card, Joker, RunState, score_hand, decide_play, _play_floor
from controller import collect_jokers
from tests.test_runtime_recovery import FakeSession
import tempfile
import json
from recovery import state_from_dict
from strategy import PlayDecision, decide_discard, _joker_purchase_value
from perception import identify_joker_text, tooltip_title, joker_definitions
from joker_cache import STABLE_WITHIN_BLIND

ROOT = Path(__file__).resolve().parents[1]

class CounterProofTests(unittest.TestCase):
    def test_corrected_actual_vampire_input_reproduces_the_real_score(self):
        cases = json.loads((ROOT / 'tests/fixtures/score_residuals_20261009.json').read_text('utf-8'))
        case = next(c for c in cases if c['actual'] == 2280)
        state = state_from_dict(case['state'])
        corrected = replace(state, jokers=tuple(replace(j, value=1.1) if j.key == 'vampire' else j for j in state.jokers))
        cards = [Card(**c) for c in case['cards']]
        selected = [cards[i] for i in case['order']]
        held = [c for i, c in enumerate(cards) if i not in case['order']]
        self.assertEqual(score_hand(selected, state, held).score, 21090)
        self.assertAlmostEqual(score_hand(selected, corrected, held).score, case['actual'])

    def test_real_roster_collection_cannot_publish_the_broad_ocr_x11(self):
        path = ROOT / 'tests/fixtures/vampire_decimal_lost_20261009.png'
        image = cv2.resize(cv2.imdecode(np.fromfile(path, np.uint8), 1), (2064, 1191),
                           interpolation=cv2.INTER_NEAREST)
        prior = Joker('Vampire', 1.1, data=(('observed_value', 1), ('value_observed_at_hand', 12)))
        state = RunState(total_hands_played=13, jokers=(prior,))
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            session.frame = image
            with patch('controller.joker_counter', return_value=(1, 5)), \
                 patch('controller.read_jokers', return_value=[(1., 'vampire', (1211, 220))]):
                actual = collect_jokers(session, image, state)
        self.assertEqual(actual.jokers[0].value, 1.1)
        self.assertTrue(actual.jokers[0].get('observed_value'))

    def test_actual_decimal_and_prefix_errors_recover_from_scoped_reads(self):
        prior = Joker('Vampire', 1.1, data=(('observed_value', 1), ('value_observed_at_hand', 12)))
        state = RunState(total_hands_played=13, jokers=(prior,))
        for name, expected in [('vampire_decimal_lost', 1.1),
                               ('vampire_prefix_misread', 1.2),
                               ('vampire_decimal_lost_12', 1.2)]:
            with self.subTest(frame=name):
                path = ROOT / 'tests/fixtures' / f'{name}_20261009.png'
                image = cv2.resize(cv2.imdecode(np.fromfile(path, np.uint8), 1), (2064, 1191),
                                   interpolation=cv2.INTER_NEAREST)
                words, _ = joker_tooltip_words(read_text(image, (525, 330, 2020, 850)), 1211)
                raw = parse_joker_tooltip(joker_from_definition('vampire'), tooltip_labels(words))
                self.assertGreater(raw.value, 2)
                proven = verify_current_multiplier(image, words, raw, state, prior)
                self.assertEqual(proven.value, expected)
                self.assertTrue(proven.get('observed_value'))
                self.assertFalse(proven.get('counter_lower_bound'))

    def test_agreed_but_physically_impossible_number_still_cannot_enter_scoring(self):
        prior = Joker('Vampire', 1.1, data=(('observed_value', 1), ('value_observed_at_hand', 10)))
        state = RunState(total_hands_played=11)
        words = [('Currently X11 Mult', .99, (1189, 518))]
        image = np.zeros((1191, 2064, 3), np.uint8)
        with patch('perception.read_text', return_value=words):
            actual = verify_current_multiplier(image, words, Joker('Vampire', 11), state, prior)
        self.assertEqual(actual.value, 1.1)
        self.assertFalse(actual.get('observed_value'))
        self.assertTrue(actual.get('counter_lower_bound'))

    def test_a_legitimate_large_multiplier_is_preserved_with_history(self):
        prior = Joker('Vampire', 10.9, data=(('observed_value', 1), ('value_observed_at_hand', 30)))
        state = RunState(total_hands_played=31)
        words = [('Currently X11 Mult', .99, (1189, 518))]
        image = np.zeros((1191, 2064, 3), np.uint8)
        with patch('perception.read_text', return_value=words):
            actual = verify_current_multiplier(image, words, Joker('Vampire', 11), state, prior)
        self.assertEqual(actual.value, 11)
        self.assertTrue(actual.get('observed_value'))

    def test_unreadable_counter_is_an_explicit_bound_and_cannot_reset_a_monotonic_scaler(self):
        prior = Joker('Vampire', 1.3, data=(('observed_value', 1), ('value_observed_at_hand', 20)))
        state = RunState(total_hands_played=21)
        result = verify_current_multiplier(np.zeros((1191, 2064, 3), np.uint8), [],
                                           Joker('Vampire', 1), state, prior)
        self.assertEqual(result.value, 1.3)
        self.assertFalse(result.get('observed_value'))
        self.assertTrue(result.get('counter_lower_bound'))

class HookScoreTests(unittest.TestCase):
    def test_actual_queen_mime_hand_cannot_claim_a_sure_clear(self):
        # Recorded prediction 1536; actual 704 after Hook discarded the Queen.
        cards = [Card(14, 'C'), Card(14, 'D'), Card(12, 'S'), Card(11, 'C'),
                 Card(10, 'D'), Card(6, 'H'), Card(4, 'S'), Card(3, 'D')]
        state = RunState(boss='The Hook', target_score=1200, hands_left=3,
                         jokers=(Joker('Shoot the Moon'), Joker('Mime'), Joker('Half Joker')))
        selected, held = [cards[i] for i in (0, 1, 3)], [cards[i] for i in (2, 4, 5, 6, 7)]
        expected = score_hand(selected, state, held)
        floor = score_hand(selected, replace(state, score_mode='conservative'), held)
        self.assertEqual(floor.score, 704)
        self.assertAlmostEqual(expected.score, 1203.2)
        self.assertEqual(_play_floor((0, 1, 3), expected, cards, state), 704)
        self.assertFalse(_play_floor((0, 1, 3), expected, cards, state) >= state.score_needed)

    def test_hook_removing_red_cards_can_enable_blackboard(self):
        played = [Card(10, 'S')]
        state = RunState(boss='The Hook', jokers=(Joker('Blackboard'),))
        held = [Card(2, 'H'), Card(3, 'D'), Card(4, 'C')]
        base = score_hand(played, replace(state, boss=''), held)
        self.assertAlmostEqual(score_hand(played, state, held).score, base.score * (1 + 2 / 3))
        self.assertEqual(score_hand(played, replace(state, score_mode='conservative'), held).score, base.score)

    def test_hook_removes_the_only_held_card_and_disabled_hook_does_not(self):
        played, held = [Card(10, 'S')], [Card(13, 'C', enhancement='steel')]
        state = RunState(boss='The Hook', jokers=(Joker('Baron'),))
        self.assertEqual(score_hand(played, state, held).score, 15)
        self.assertEqual(score_hand(played, replace(state, boss_disabled=True), held).score, 33.75)


class SmallEnginePolicyTests(unittest.TestCase):
    def test_stable_cache_uses_the_catalogue_keys_for_mail_and_eight_ball(self):
        self.assertFalse(STABLE_WITHIN_BLIND - set(joker_definitions()))
        self.assertIn('mail_in_rebate', STABLE_WITHIN_BLIND)
        self.assertIn('8_ball', STABLE_WITHIN_BLIND)

    def test_selected_sale_identity_excludes_an_unrelated_shop_title(self):
        path = ROOT / 'tests/fixtures/selected_roster_sale_20261009.png'
        image = cv2.resize(cv2.imdecode(np.fromfile(path, np.uint8), 1), (2064, 1191),
                           interpolation=cv2.INTER_NEAREST)
        words = read_text(image, (520, 70, 2040, 650))
        words.append(('Loyalty Card', .99, (1200, 630)))
        main, _ = joker_tooltip_words(words, 1028)
        self.assertEqual(identify_joker_text(tooltip_labels(main), title=tooltip_title(main))[0], 'odd_todd')
        self.assertFalse(any('Loyalty' in w[0] for w in main))

    def test_actual_on_pace_pair_saves_a_discard_but_large_gain_still_uses_one(self):
        rows = json.loads((ROOT / 'tests/fixtures/discard_pace_20261009.json').read_text('utf-8'))
        for case, should_discard in zip(rows, (False, True)):
            with self.subTest(time=case['time']):
                state = state_from_dict(case['state'])
                cards = [Card(**c) for c in case['cards']]
                order = tuple(case['order'])
                estimate = score_hand([cards[i] for i in order], state,
                                      [c for i, c in enumerate(cards) if i not in order])
                current = PlayDecision(order, estimate, estimate.score, case['reason'], order)
                choice = decide_discard(cards, state, samples=8, current=current)
                self.assertEqual(bool(choice.indices), should_discard)

    def test_new_vampire_is_valued_with_actual_enhancement_or_midas_support(self):
        plain = tuple(Card(r, 'H') for r in range(2, 15))
        state = RunState(deck_cards=plain, deck_size=len(plain))
        vampire = Joker('Vampire', 1)
        unsupported = _joker_purchase_value(vampire, state)
        enhanced = replace(state, deck_cards=tuple(replace(c, enhancement='mult') for c in plain))
        midas = replace(state, jokers=(Joker('Midas Mask'),))
        self.assertGreater(_joker_purchase_value(vampire, enhanced), unsupported)
        self.assertGreater(_joker_purchase_value(vampire, midas), unsupported)

if __name__ == '__main__':
    unittest.main()
