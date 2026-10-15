"""Offline regressions from the October batch; never connect to a device."""
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from controller import collect_levels, select_verified, close_run_info, collect_jokers, collect_cards
from perception import (parse_levels, tooltip_labels, identify_joker_text,
                        identify_consumable_text, read_text, read_consumables,
                        consumable_slots, consumable_centers, read_card_title)
from perception import score_numeric
from perception import parse_card_tooltip
from recovery import (ObservationError, action_outcome, begin_action,
                      confirm_pending, state_from_dict)
from recovery import DesktopUnavailableError
from runner import run_batch, prepare_new_run, play_one_hand
from strategy import Card, Joker, RunState, score_hand

ROOT = Path(__file__).resolve().parents[1]


class FakeSession:
    def __init__(self, directory):
        self.run_dir = Path(directory)
        self.policy_state = RunState()
        self.pending_action = None
        self.events, self.taps = [], []
        self.frame = np.zeros((1191, 2064, 3), dtype=np.uint8)

    def log(self, event, **details):
        self.events.append(dict(event=event, **details))

    def observe(self, label):
        return self.frame

    def hover(self, *args):
        pass

    def tap(self, *args):
        self.taps.append(args)


def level_words(missing=()):
    names = ['Straight Flush', 'Four of a Kind', 'Full House', 'Flush', '5traight',
             'Three of a Kind', 'Two Pair', 'Pair', 'High Card']
    words = [('Back', 1., (1032, 1027))]
    for i, name in enumerate(names):
        if i in missing:
            continue
        y = 374 + i * 60
        words += [('lul.1', .95, (641, y)), (name, .99, (915, y)), ('0', .99, (1440, y))]
    return words


class RuntimeRecoveryTests(unittest.TestCase):
    def test_joker_order_reconciles_dropped_and_unexpected_drag_results(self):
        from controller import apply_joker_order
        state = RunState(jokers=tuple(Joker(n) for n in
                         ('Mad Joker', 'Clever Joker', 'Card Sharp', 'Sly Joker', 'Popcorn')))
        order = (0, 1, 3, 4, 2)
        partial = replace(state, jokers=tuple(state.jokers[i] for i in (0, 1, 2, 4, 3)))
        goal = replace(state, jokers=tuple(state.jokers[i] for i in order))
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            moves = []
            session.drag = lambda *args: moves.append(args)
            with patch('controller.collect_jokers', side_effect=[state, partial, goal]):
                result = apply_joker_order(session, state, order)
            self.assertEqual(result.jokers, goal.jokers)
            self.assertEqual(moves[0][:2], moves[1][:2])
            self.assertNotEqual(moves[1][0], moves[2][0])

    def test_joker_order_uses_verified_actual_order_after_bounded_failures(self):
        from controller import apply_joker_order
        state = RunState(jokers=(Joker('Card Sharp'), Joker('Popcorn')))
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            moves = []
            session.drag = lambda *args: moves.append(args)
            with patch('controller.collect_jokers', return_value=state):
                actual = apply_joker_order(session, state, (1, 0))
            self.assertEqual(actual, state)
            self.assertLessEqual(len(moves), 5)
            self.assertTrue(any(e['event'] == 'joker_order_fallback' for e in session.events))

    def test_joker_order_rejects_a_changed_roster_and_invalid_permutation(self):
        from controller import apply_joker_order
        state = RunState(jokers=(Joker('Card Sharp'), Joker('Popcorn')))
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            session.drag = lambda *args: None
            changed = replace(state, jokers=(Joker('Card Sharp'), Joker('Sly Joker')))
            with patch('controller.collect_jokers', return_value=changed):
                with self.assertRaisesRegex(ObservationError, 'roster changed'):
                    apply_joker_order(session, state, (1, 0))
            with self.assertRaisesRegex(ObservationError, 'permutation'):
                apply_joker_order(session, state, (0, 0))

    def test_joker_order_distinguishes_duplicate_growth_joker_values(self):
        from controller import apply_joker_order
        state = RunState(jokers=(Joker('Ride the Bus', value=3), Joker('Ride the Bus', value=8)))
        goal = replace(state, jokers=state.jokers[::-1])
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            session.drag = lambda *args: None
            with patch('controller.collect_jokers', return_value=goal):
                actual = apply_joker_order(session, state, (1, 0))
            self.assertEqual([j.value for j in actual.jokers], [8, 3])

    def test_tower_selection_excludes_adjacent_stone_explanation(self):
        from runner import verify_selected_purchase
        from perception import shop_tooltip_region, tooltip_rows, shop_offer_item
        from strategy import ShopItem
        for name in ('tower_selected_adjacent_panel.png', 'tower_hover_adjacent_panel.png'):
            path = ROOT / 'tests/fixtures' / name
            image = cv2.resize(cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1),
                               (2064, 1191), interpolation=cv2.INTER_NEAREST)
            region = shop_tooltip_region(image, 1161)
            labels = tooltip_rows(read_text(image, region))
            self.assertFalse(any('no rank or suit' in line.lower() for line in labels))
            self.assertEqual(shop_offer_item(labels, 3)[0].name, 'The Tower')
            verify_selected_purchase(image, ShopItem('tarot', 'The Tower', 3), 1161)
            with self.assertRaises(ObservationError):
                verify_selected_purchase(image, ShopItem('tarot', 'The Star', 3), 1161)

    def test_three_actual_debuffed_nine_titles_are_read_without_chip_values(self):
        broken = ['of Diamonds', 'Scores no chips', 'and all abilities are disabled', 'Debuffed']
        for number in (477, 478, 479):
            path = ROOT / f'tests/fixtures/debuffed_nine_title_{number}.png'
            image = cv2.resize(cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1),
                               (2064, 1191), interpolation=cv2.INTER_NEAREST)
            title = read_card_title(image, (973, 711))
            card = parse_card_tooltip(Card(9, 'S'), [*broken, *title], require_identity=True)
            self.assertEqual((card.rank, card.suit, card.debuffed), (9, 'D', True))

    def test_card_collection_recovers_missing_rank_from_isolated_title(self):
        path = ROOT / 'tests/fixtures/debuffed_nine_title_479.png'
        image = cv2.resize(cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1),
                           (2064, 1191), interpolation=cv2.INTER_NEAREST)
        found = [(Card(9, 'D'), .55, (973, 711))]
        words = [('of Diamonds', .99, (1050, 490)), ('Scores no chips', .99, (1050, 530)),
                 ('Debuffed', .99, (1050, 640))]
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            session.frame = image
            with patch('controller.read_hand', return_value=found), patch('controller.read_text', return_value=words):
                result = collect_cards(session, image, 1)
            self.assertEqual((result[0][0].identity, result[0][0].debuffed), ((9, 'D'), True))
            self.assertTrue(any(row['event'] == 'card_title_recovered' for row in session.events))
            self.assertFalse(any(row['event'] == 'card_identity_retry' for row in session.events))

    def test_title_fallback_rejects_incomplete_or_low_confidence_titles(self):
        path = ROOT / 'tests/fixtures/debuffed_nine_title_479.png'
        image = cv2.resize(cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1),
                           (2064, 1191), interpolation=cv2.INTER_NEAREST)
        for title, confidence in [('of Diamonds', .99), ('9 of Diamonds', .8), ('0/2', .99)]:
            with patch('perception.read_text', return_value=[(title, confidence, (1050, 490))]):
                self.assertEqual(read_card_title(image, (973, 711)), [])

    def test_ten_games_default_budget_outlasts_2000_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            observations = 0
            def observe(label):
                nonlocal observations
                observations += 1
                session.frame[100, 120] = (observations % 4) * 64
                return session.frame
            session.observe = observe
            scenes = [scene for _ in range(10) for scene in ['deck', *(['hand'] * 210), 'loss']]
            with patch('runner.prepare_new_run', return_value=(.5, .79)), patch('runner.classify_scene', side_effect=scenes), patch('runner.scene_scores', return_value={}), patch('runner.time.sleep'), patch('runner.play_one_hand'), patch('runner.export_run'):
                run_batch(session, 10)
            status = json.loads((session.run_dir / 'status.json').read_text('utf8'))
            self.assertEqual((status['status'], status['finished_games']), ('complete', 10))
            self.assertEqual(status['action_limit'], 20000)
            self.assertGreater(observations, 2000)

    def test_actual_popcorn_tooltip_overrides_static_initial_value(self):
        path = ROOT / 'tests/fixtures/popcorn_decayed_20261004.png'
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        image = cv2.resize(image, (2064, 1191), interpolation=cv2.INTER_NEAREST)
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            session.frame = image
            state = collect_jokers(session, image, RunState())
            self.assertEqual(len(state.jokers), 1)
            self.assertEqual((state.jokers[0].name, state.jokers[0].value), ('Popcorn', 16))

    def test_focus_denial_waits_without_exhausting_recovery_or_counting_a_game(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            original_observe = session.observe
            calls = 0
            def observe(label):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise DesktopUnavailableError('Cannot focus Balatro')
                return original_observe(label)
            session.observe = observe
            failed_focus = 0
            def focus():
                nonlocal failed_focus
                waiting = json.loads((session.run_dir / 'status.json').read_text())
                self.assertEqual(waiting['status'], 'waiting_for_desktop')
                self.assertEqual(waiting['finished_games'], 0)
                self.assertEqual(waiting['automatic_recoveries'], 0)
                self.assertEqual(session.taps, [])
                failed_focus += 1
                if failed_focus <= 7:
                    raise DesktopUnavailableError('Cannot focus Balatro')
            session.focus = focus
            with patch('runner.prepare_new_run', return_value=(.5, .79)), patch('runner.classify_scene', side_effect=['deck', 'hand', 'loss']), patch('runner.scene_scores', return_value={}), patch('runner.time.sleep'), patch('runner.play_one_hand'):
                run_batch(session, 1, max_actions=10)
            status = json.loads((session.run_dir / 'status.json').read_text())
            self.assertEqual((status['status'], status['finished_games'], status['automatic_recoveries']), ('complete', 1, 0))
            self.assertTrue(any(e['event'] == 'desktop_access_restored' for e in session.events))

    def test_run_info_close_retries_a_missed_click_before_reading_hand(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            with patch('controller.hand_counter', side_effect=[ObservationError('Overlay still open'), (8, 8)]), patch('controller.read_text', return_value=level_words()), patch('controller.time.sleep'):
                close_run_info(session, (1032, 1000))
            self.assertEqual(len(session.taps), 2)
            self.assertEqual(session.events[-1]['event'], 'run_info_closed')

    def test_run_info_unknown_outcome_does_not_repeat_an_unverified_click(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            with patch('controller.hand_counter', side_effect=ObservationError('No hand')), patch('controller.read_text', return_value=[]):
                with self.assertRaises(ObservationError):
                    close_run_info(session, (1032, 1000))
            self.assertEqual(len(session.taps), 1)

    def test_consumable_state_survives_recollection_and_play(self):
        hand = [Card(14, 'S'), Card(14, 'H')]
        found = [(card, 1.0, (600 + i * 100, 725)) for i, card in enumerate(hand)]
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            session.policy_state = RunState(consumables_held=1, discards_left=0)
            states = []

            def collect(s, state, **kwargs):
                states.append(state)
                return state, found

            def consume(s, state, cards):
                if len(states) == 1:
                    s.policy_state = replace(state, consumables_held=0)
                    return True
                return False

            with patch('runner.collect_position', side_effect=collect), patch('runner.try_consumable', side_effect=consume), patch('runner.select_verified'), patch('runner.verify_play_preview'), patch('runner.apply_play_order', side_effect=lambda s, f, d: (f, list(d.indices))), patch('runner.time.sleep'), patch('vision.classify_scene', return_value='hand'), patch('perception.read_numbers', return_value={'hands_left': 3, 'round_score': 50}):
                play_one_hand(session)
            self.assertEqual(states[1].consumables_held, 0)
            self.assertEqual(session.policy_state.consumables_held, 0)

    def test_zero_score_click_is_confirmed_and_preserves_mouth_lock(self):
        fixture = json.loads((ROOT / 'tests/fixtures/mouth_no_scoring_hand.json').read_text(encoding='utf-8'))
        state = state_from_dict(fixture['state'])
        hand = [Card(**card) for card in fixture['cards']]
        found = [(card, 1.0, (600 + i * 100, 725)) for i, card in enumerate(hand)]
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            session.policy_state = state
            with patch('runner.collect_position', return_value=(state, found)), patch('runner.try_consumable', return_value=False), patch('runner.select_verified'), patch('runner.verify_play_preview'), patch('runner.apply_play_order', side_effect=lambda s, f, d: (f, list(d.indices))), patch('runner.time.sleep'), patch('vision.classify_scene', return_value='hand'), patch('perception.read_numbers', return_value={'hands_left': 2, 'round_score': 1539}):
                _, score = play_one_hand(session)
            self.assertEqual(score, 0)
            self.assertEqual(session.policy_state.boss_target_hand, 'straight')
            self.assertIn('straight', session.policy_state.boss_seen_types)
            self.assertIsNone(session.pending_action)
            self.assertEqual(sum(e['event'] == 'action_confirmed' for e in session.events), 1)
            self.assertEqual(sum(e['event'] == 'zero_score_cycle' for e in session.events), 1)
            self.assertTrue(any(tap[2].startswith('play verified') for tap in session.taps))

    def test_chip_and_suit_prove_card_when_of_is_garbled(self):
        card = parse_card_tooltip(Card(8, 'H'), ['0/5', '1/2', 'Ҷf', 'Spades', '+8 chips'])
        self.assertEqual((card.rank, card.suit), (8, 'S'))
        with self.assertRaises(ObservationError):
            parse_card_tooltip(Card(8, 'H'), ['0/5', '1/2', '8'])

    def test_new_run_tab_requires_fresh_settings_before_play(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            tab = [('New Run', .99, (800, 210))]
            fresh = tab + [('Seeded Run', .99, (660, 918)), ('PLAY', .99, (1032, 918))]
            with patch('runner.read_text', side_effect=[tab, fresh]):
                self.assertEqual(prepare_new_run(session, session.frame), (1032/2064, 918/1191))
            continued = tab + [('Best Hand', .99, (1000, 450)), ('PLAY', .99, (1032, 918))]
            with patch('runner.read_text', side_effect=[tab, continued]):
                with self.assertRaises(ObservationError):
                    prepare_new_run(session, session.frame)

    def test_actual_misread_straight_is_not_a_missing_level(self):
        levels, _ = parse_levels(level_words())
        self.assertEqual(len(levels), 9)
        self.assertEqual(levels['straight'], 1)

    def test_score_separator_and_missing_zeros_are_distinct(self):
        self.assertEqual(score_numeric('1.200'), 1200)
        self.assertEqual(score_numeric('#1,600'), 1600)
        self.assertEqual(score_numeric('1.6'), 1.6)

    def test_basic_joker_missing_mult_digit_requires_a_verified_title(self):
        lines = ['4/5', 'Joker', 'Mult', 'Common']
        with self.assertRaises(ObservationError):
            identify_joker_text(lines)
        self.assertEqual(identify_joker_text(lines, title='Joker')[0], 'joker')

    def test_square_title_fragments_ignore_neighboring_counters(self):
        words = [('4/5', 1, (608, 375)), ('Joker', 1, (1100, 375)),
                 ('Square', 1, (990, 377)), ('1/2', 1, (1850, 374))]
        self.assertEqual(identify_joker_text(tooltip_labels(words))[0], 'square_joker')

    def test_correct_square_value_removes_fake_four_mult(self):
        cards = [Card(12, 'H'), Card(12, 'D'), Card(5, 'S'), Card(5, 'D')]
        state = RunState(jokers=(Joker('Business Card'), Joker('Joker', 4),
                                 Joker('Even Steven'), Joker('Square Joker', 0)))
        self.assertEqual(score_hand(cards, state).score, 324)

    def test_hidden_shop_title_identified_by_description(self):
        self.assertEqual(identify_joker_text(['Not Discovered', 'Played cards with',
                         'Diamond suit give', '+3 Mult when scored', 'Common'])[0], 'greedy_joker')

    def test_card_border_cannot_make_planet_into_generic_joker(self):
        with self.assertRaises(ObservationError):
            identify_joker_text(['JOKER', 'Not Discovered', 'PLUTO', 'Level up',
                                 'High Card', '+1 Mult and', '+10 chips', 'Planet'])

    def test_consumable_printed_title_without_the(self):
        self.assertEqual(identify_consumable_text(['HIEROPHANT'])[0], 'the_hierophant')

    def test_partial_levels_retry_closes_menu_and_preserves_history(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            with patch('controller.read_text', side_effect=[level_words((4,)), level_words((4,)), level_words()]), patch('controller.time.sleep'), patch('controller.hand_counter', return_value=(8, 8)):
                state = collect_levels(session, RunState(levels={'five_kind': 3}))
            self.assertEqual(state.level('straight'), 1)
            self.assertEqual(state.level('five_kind'), 3)
            self.assertEqual(len(session.taps), 2)
            self.assertTrue(any(e['event'] == 'hand_levels_retry' for e in session.events))

    def test_verified_level_cache_can_cover_one_ocr_omission(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            session.levels_cache = parse_levels(level_words())[0]
            session.levels_dirty = False
            with patch('controller.read_text', return_value=level_words((4,))), patch('controller.time.sleep'), patch('controller.hand_counter', return_value=(8, 8)):
                state = collect_levels(session, RunState())
            self.assertEqual(state.levels['straight'], 1)
            self.assertTrue(any(e['event'] == 'hand_levels_cached' for e in session.events))

    def test_level_change_invalidates_cache_and_does_not_invent_a_level(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            session.levels_cache = parse_levels(level_words())[0]
            session.levels_dirty = True
            with patch('controller.read_text', return_value=level_words((4,))), patch('controller.time.sleep'), patch('controller.hand_counter', return_value=(8, 8)):
                with self.assertRaises(ObservationError):
                    collect_levels(session, RunState())
            self.assertEqual(len(session.taps), 2)

    def test_selection_retry_toggles_only_missing_card(self):
        cards = [(Card(i + 2, 'S'), .9, (560 + i * 135, 740)) for i in range(8)]
        first = [cards[i] if i != 0 else (cards[i][0], .9, (560, 670)) for i in range(8)]
        second = list(first); second[7] = (cards[7][0], .9, (1505, 670))
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            with patch('controller.read_hand', side_effect=[first, second]):
                select_verified(session, cards, (0, 7), 'discard')
            self.assertEqual(len(session.taps), 3)
            self.assertEqual(session.events[-1]['event'], 'selection_verified')

    def test_play_counter_proves_click_during_animation(self):
        pending = {'kind': 'play', 'before': {'hands_left': 4, 'round_score': 0}}
        self.assertTrue(action_outcome(pending, 'unknown', {'hands_left': 3, 'round_score': 0}))
        self.assertIsNone(action_outcome(pending, 'unknown', {'hands_left': 4, 'round_score': 0}))

    def test_buy_detects_wrong_item_type_without_buying_twice(self):
        pending = {'kind': 'buy', 'before': {'joker_count': 0, 'consumable_count': 0, 'money': 10}}
        self.assertTrue(action_outcome(pending, 'shop', {'joker_count': 0, 'consumable_count': 1, 'money': 7}))

    def test_action_is_confirmed_once_and_restores_history(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            after = RunState(total_hands_played=1, seen_cards=(Card(14, 'S'),))
            begin_action(session, 'play', {'hands_left': 4, 'round_score': 0}, after_state=after)
            with patch('vision.classify_scene', return_value='unknown'), patch('perception.read_numbers', return_value={'hands_left': 3}), patch('recovery.time.sleep'):
                confirm_pending(session)
                confirm_pending(session)
            self.assertEqual(session.policy_state.total_hands_played, 1)
            self.assertEqual(session.policy_state.seen_cards, (Card(14, 'S'),))
            restored = state_from_dict(json.loads((session.run_dir / 'policy-state.json').read_text(encoding='utf-8')))
            self.assertEqual(restored.total_hands_played, 1)
            self.assertEqual(restored.seen_cards, (Card(14, 'S'),))
            self.assertIsNone(session.pending_action)
            self.assertEqual(session.taps, [])
            self.assertEqual(sum(e['event'] == 'action_confirmed' for e in session.events), 1)

    def test_batch_recovers_and_counts_each_full_game_only_once(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            scenes = ['deck', 'hand', 'hand', 'loss', 'loss', 'deck', 'hand', 'loss']
            with patch('runner.prepare_new_run', return_value=(.5, .79)), patch('runner.classify_scene', side_effect=scenes), patch('runner.scene_scores', return_value={}), patch('runner.time.sleep'), patch('runner.recover_view') as recover, patch('runner.play_one_hand', side_effect=[ObservationError('Only 8 levels'), None, None]):
                run_batch(session, 2, max_actions=20)
            self.assertEqual(recover.call_count, 1)
            self.assertEqual([e['number'] for e in session.events if e['event'] == 'game_end'], [1, 2])
            status = json.loads((session.run_dir / 'status.json').read_text(encoding='utf-8'))
            self.assertEqual((status['status'], status['finished_games']), ('complete', 2))

    def test_recovery_exhaustion_does_not_count_a_game_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            with patch('runner.classify_scene', return_value='hand'), patch('runner.scene_scores', return_value={}), patch('runner.time.sleep'), patch('runner.recover_view'), patch('runner.play_one_hand', side_effect=ObservationError('Unreadable cards')):
                with self.assertRaisesRegex(RuntimeError, 'recovery exhausted'):
                    run_batch(session, 10, max_actions=20, resume=True)
            status = json.loads((session.run_dir / 'status.json').read_text(encoding='utf-8'))
            self.assertEqual(status['finished_games'], 0)
            self.assertEqual(status['status'], 'failed')

    def test_new_batch_does_not_count_previous_game_over(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            with patch('runner.prepare_new_run', return_value=(.5, .79)), patch('runner.classify_scene', side_effect=['loss', 'deck', 'hand', 'loss']), patch('runner.scene_scores', return_value={}), patch('runner.time.sleep'), patch('runner.play_one_hand'):
                run_batch(session, 1, max_actions=10)
            self.assertEqual(len([e for e in session.events if e['event'] == 'game_end']), 1)
            self.assertTrue(any('previous game' in tap[2] for tap in session.taps))

    def test_actual_termination_screenshot_all_nine_levels(self):
        path = ROOT / 'runs/test-20261001-001157/1069-poker-hand-levels.png'
        if not path.exists():
            self.skipTest('Historical screenshot removed')
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        image = cv2.resize(image, (2064, 1191), interpolation=cv2.INTER_NEAREST)
        self.assertEqual(len(parse_levels(read_text(image))[0]), 9)

    def test_actual_hierophant_screenshot_is_no_longer_cropped_out(self):
        path = ROOT / 'runs/test-20261001-001157/1068-step-54.png'
        if not path.exists():
            self.skipTest('Historical screenshot removed')
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        image = cv2.resize(image, (2064, 1191), interpolation=cv2.INTER_NEAREST)
        matches = read_consumables(image)
        self.assertTrue(any(key == 'the_hierophant' for _, key, _ in matches))

    def test_two_held_consumables_use_the_full_area(self):
        path = ROOT / 'tests/fixtures/held_consumables_20261003.png'
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        image = cv2.resize(image, (2064, 1191), interpolation=cv2.INTER_NEAREST)
        matches = read_consumables(image)
        self.assertEqual([key for _, key, _ in matches], ['the_hierophant', 'the_lovers'])
        for slot, (_, _, center) in zip(consumable_slots(2), matches):
            self.assertLess(abs(slot - center[0]), 20)
        self.assertEqual(consumable_slots(0), [])
        self.assertLess(abs(consumable_slots(1)[0] - 1688), 20)
        self.assertEqual(consumable_slots(6)[::5], consumable_slots(2))
        self.assertEqual(consumable_centers(matches, 2), [match[2] for match in matches])
        # Crowded slots must not all reuse the nearest visible neighbour.
        centers = consumable_centers(matches[:1], 4)
        self.assertEqual(len(set(centers)), 4)
        self.assertEqual([x for x, _ in centers][1:], consumable_slots(4)[1:])


if __name__ == '__main__':
    unittest.main()
