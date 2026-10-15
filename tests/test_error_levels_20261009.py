"""Exercise degraded, recoverable and fatal outcomes without a real device."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from controller import collect_jokers
from perception import bounded_temporary_joker
from planning import _age_temporary_engine, advance_projected_play
from recovery import ObservationError, DesktopUnavailableError
from runner import run_batch, shop
from runtime_issues import record_issue, finish_issue_review, terminal_failures, unresolved_recovery
from strategy import Card, Joker, RunState, ShopItem, ShopDecision, score_hand
from tests.test_runtime_recovery import FakeSession
from tools.monitor_batch import probe, main as monitor_main


class ErrorLevelTests(unittest.TestCase):
    def test_repeated_degraded_issues_are_durable_and_grouped_for_game_end(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            session.current_game_number = 1
            for value in (1, 1.1, 1.2):
                record_issue(session, 'joker_counter_bounded', level='degraded', code='multiplier',
                             message='Use a bound', value=value)
            self.assertFalse((Path(directory) / 'issue-reviews').exists())
            restored = FakeSession(directory)
            report = json.loads(finish_issue_review(restored, 1, 'loss').read_text('utf-8'))
            self.assertEqual(report['counts']['degraded'], 3)
            self.assertEqual(len(report['issues']), 1)
            self.assertEqual(report['issues'][0]['last_context']['value'], 1.2)
            self.assertFalse(terminal_failures(session.events))

    def test_recovered_game_finishes_without_restart_and_writes_review_after_end(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            calls = 0
            def hand(*args):
                nonlocal calls
                self.assertFalse((Path(directory) / 'issue-reviews').exists())
                calls += 1
                if calls == 1:
                    raise ObservationError('Selected cards are unreadable')
                record_issue(session, 'joker_counter_bounded', level='degraded', code='multiplier',
                             message='Use a lower bound', value=1)
            with patch('runner.prepare_new_run', return_value=(.5, .79)), \
                 patch('runner.classify_scene', side_effect=['deck', 'hand', 'hand', 'loss']), \
                 patch('runner.scene_scores', return_value={}), patch('runner.time.sleep'), \
                 patch('runner.recover_view'), patch('runner.play_one_hand', side_effect=hand):
                run_batch(session, 1, max_actions=10)
            self.assertEqual(sum(e['event'] == 'game_start' for e in session.events), 1)
            self.assertEqual(sum(e['event'] == 'game_end' for e in session.events), 1)
            self.assertIsNone(unresolved_recovery(session.events))
            self.assertFalse(terminal_failures(session.events))
            review = json.loads((Path(directory) / 'issue-reviews/game-0001.json').read_text('utf-8'))
            self.assertEqual(review['counts']['recoverable'], 1)
            self.assertEqual(review['counts']['degraded'], 1)
            self.assertLess(next(i for i, e in enumerate(session.events) if e['event'] == 'game_end'),
                            next(i for i, e in enumerate(session.events) if e['event'] == 'game_issue_review'))

    def test_focus_failure_during_recovery_waits_without_restarting_the_game(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            session.focus = lambda: None
            with patch('runner.prepare_new_run', return_value=(.5, .79)), \
                 patch('runner.classify_scene', side_effect=['deck', 'hand', 'hand', 'loss']), \
                 patch('runner.scene_scores', return_value={}), patch('runner.time.sleep'), \
                 patch('runner.play_one_hand', side_effect=[ObservationError('Unreadable hand'), None]), \
                 patch('runner.recover_view', side_effect=DesktopUnavailableError('Focus denied')):
                run_batch(session, 1, max_actions=10)
            status = json.loads((session.run_dir / 'status.json').read_text('utf-8'))
            self.assertEqual(status['status'], 'complete')
            self.assertEqual(status['automatic_recoveries'], 1)
            self.assertEqual(status['issue_counts']['waiting'], 1)
            self.assertEqual(status['issue_counts']['fatal'], 0)
            self.assertEqual(sum(e['event'] == 'game_start' for e in session.events), 1)
            self.assertFalse(terminal_failures(session.events))

    def test_monitor_separates_recovery_from_terminal_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = [dict(event='automatic_recovery', attempt=1, error='Temporary OCR'),
                      dict(event='recovery_resolved', attempts=1)]
            (root / 'events.jsonl').write_text('\n'.join(json.dumps(e) for e in events), 'utf-8')
            (root / 'status.json').write_text(json.dumps(dict(status='running')), 'utf-8')
            with patch('tools.monitor_batch.psutil.pid_exists', return_value=True):
                result = probe(root, 123)
            self.assertFalse(result['failures'])
            self.assertEqual(len(result['recoveries']), 1)
            self.assertIsNone(result['unresolved_recovery'])
            events.append(dict(event='batch_error', error='Recovery exhausted'))
            self.assertEqual(len(terminal_failures(events)), 1)

    def test_monitor_does_not_return_early_on_a_recovery_notification(self):
        recovering = dict(process_alive=True, failures=[], scene='hand', status=dict(status='recovering'))
        complete = dict(process_alive=True, failures=[], scene='loss', status=dict(status='complete'))
        with patch('sys.argv', ['monitor', '--run-dir', 'unused', '--pid', '123', '--seconds', '10']), \
             patch('tools.monitor_batch.probe', side_effect=[recovering, complete]) as mock_probe, \
             patch('tools.monitor_batch.time.monotonic', side_effect=[0, 0, 0]), \
             patch('tools.monitor_batch.time.sleep'), patch('builtins.print'):
            monitor_main()
        self.assertEqual(mock_probe.call_count, 2)

    def test_declining_numeric_effect_can_degrade_but_unknown_identity_cannot(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            words = [('Popcorn', .99, (1100, 380)), ('Foil', .99, (800, 400))]
            with patch('controller.joker_counter', return_value=(1, 5)), \
                 patch('controller.read_jokers', return_value=[(1., 'popcorn', (1100, 220))]), \
                 patch('controller.read_text', return_value=words), \
                 patch('controller.joker_tooltip_words', return_value=(words[:1], words[1:])), \
                 patch('controller.identify_joker_text', return_value=('popcorn', 1.)), \
                 patch('controller.parse_joker_tooltip', side_effect=ObservationError('Unreadable current effect')):
                state = collect_jokers(session, session.frame, RunState())
            joker = state.jokers[0]
            self.assertEqual(joker.value, 0)
            self.assertEqual(joker.edition, 'foil')
            self.assertTrue(joker.get('counter_lower_bound'))
            self.assertFalse(joker.get('observed_value'))
            self.assertEqual(score_hand([Card(14, 'S')], state).score, 66)
            self.assertEqual(session.events[-2]['event'], 'joker_observed')
            self.assertTrue(any(e.get('severity') == 'degraded' for e in session.events))
            with patch('controller.joker_counter', return_value=(1, 5)), \
                 patch('controller.read_jokers', return_value=[(1., 'popcorn', (1100, 220))]), \
                 patch('controller.read_text', return_value=[]), \
                 patch('controller.identify_joker_text', side_effect=ObservationError('Unknown title')):
                with self.assertRaises(ObservationError):
                    collect_jokers(FakeSession(directory), session.frame, RunState())

    def test_zero_bound_does_not_restore_defaults_or_invent_a_free_stencil_slot(self):
        for key in ('Popcorn', 'Ice Cream', 'Ramen'):
            joker = bounded_temporary_joker(Joker(key), [key])
            state = RunState(jokers=(joker, Joker('Joker Stencil')), total_joker_slots=5)
            estimate = score_hand([Card(10, 'S')], state)
            self.assertEqual(estimate.score, 60)
            aged = _age_temporary_engine(state)
            self.assertEqual(len(aged.jokers), 2)
            after = advance_projected_play(state, [Card(10, 'S')], (0,), estimate)
            self.assertEqual(len(after.jokers), 2)


class OptionalShopTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.session = FakeSession(self.directory.name)
        self.item = ShopItem('planet', 'Mercury', 3)
        patches = [patch('runner.read_numbers', return_value={'money': 20}),
                   patch('runner.consumable_counter', return_value=(0, 2)),
                   patch('runner.collect_shop_jokers', side_effect=lambda s, i, st: st),
                   patch('runner.read_jokers', return_value=[]),
                   patch('runner.read_text', return_value=[('$3', .99, (1200, 450))]),
                   patch('runner.shop_tooltip_region', return_value=(0, 0, 1, 1)),
                   patch('runner.shop_offer_item', return_value=(self.item, 1.)),
                   patch('runner.collect_extra_offers', return_value=([], [])),
                   patch('runner.read_reroll_button', return_value=None),
                   patch('runner.shop_strength', return_value={}),
                   patch('runner.time.sleep'),
                   patch('runner.decide_shop', side_effect=lambda items, *a, **kw:
                         ShopDecision('buy', item_index=0) if items else ShopDecision('leave'))]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_unverified_optional_purchase_is_skipped_and_does_not_click_buy(self):
        with patch('runner.verify_shop_selection', side_effect=ObservationError('Wrong selected title')) as verify, \
             patch('runner.confirm_pending') as confirm:
            shop(self.session, self.session.frame)
        self.assertEqual(verify.call_count, 1)
        confirm.assert_not_called()
        self.assertIsNone(self.session.pending_action)
        self.assertTrue(any(e['event'] == 'shop_purchase_skipped' and e['severity'] == 'degraded'
                            for e in self.session.events))
        self.assertEqual(sum('select verified' in tap[2] for tap in self.session.taps), 1)

    def test_an_unconfirmed_buy_click_retains_the_journal_for_recovery(self):
        with patch('runner.verify_shop_selection', return_value=((1200, 700), self.session.frame)), \
             patch('runner.confirm_pending', side_effect=ObservationError('Click outcome unreadable')):
            with self.assertRaises(ObservationError):
                shop(self.session, self.session.frame)
        self.assertEqual(self.session.pending_action['kind'], 'buy')
        self.assertFalse(any(e['event'] == 'shop_purchase_skipped' for e in self.session.events))

    def test_desktop_loss_is_not_mistaken_for_an_optional_offer_failure(self):
        with patch('runner.verify_shop_selection', side_effect=DesktopUnavailableError('Focus denied')):
            with self.assertRaises(DesktopUnavailableError):
                shop(self.session, self.session.frame)
        self.assertFalse(any(e['event'] == 'shop_purchase_skipped' for e in self.session.events))


if __name__ == '__main__':
    unittest.main()
