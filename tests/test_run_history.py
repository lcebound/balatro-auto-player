"""Regression tests for experiment statistics; never import the game runner."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from run_history import summarize_events, save_history, read_events
from tools.export_history import merge_saved_snapshots


def rows():
    events = [
        dict(event='batch_start', target_games=10, code_version={'id': 'a'}),
        dict(event='game_start', number=1, deck='Red Deck', stake='White Stake'),
        dict(event='blind_decision', blind=0),
        dict(event='observed_state', state={'ante': 1, 'target_score': 300, 'round_score': 0, 'money': 4}),
        dict(event='action_prepared', kind='play', before={'round_score': 0}, details={'category': 'pair', 'estimated_score': 320}),
        dict(event='action_confirmed', kind='play', counters={'round_score': 0}),
        dict(event='scene', name='cashout'),
        dict(event='round_snapshot', counters={'round_score': 400}, settled=True),
        dict(event='blind_decision', blind=1),
        dict(event='observed_state', state={'ante': 1, 'target_score': 450, 'round_score': 0}),
        dict(event='action_prepared', kind='play', before={'round_score': 0}, details={'category': 'pair', 'estimated_score': 180}),
        dict(event='action_confirmed', kind='play', counters={'round_score': 0}),
        dict(event='round_snapshot', counters={'round_score': 150}, settled=True,
             end_screen=['Best Hand', '400', 'Cards Played', '4', 'Seed', '76JWMHDF']),
        dict(event='game_end', number=1, result='loss'),
        dict(event='game_start', number=2, deck='Red Deck', stake='White Stake'),
        dict(event='blind_decision', blind=0),
        dict(event='observed_state', state={'ante': 1, 'target_score': 300, 'round_score': 250}),
        dict(event='batch_error', error='Observation failed'),
    ]
    return [dict(time=f'2026-10-02T00:00:{i:02d}+08:00', **row) for i, row in enumerate(events)]


class RunHistoryTests(unittest.TestCase):
    def test_partial_batch_means_use_only_completed_games(self):
        summary = summarize_events(rows(), 'sample')
        batch = summary['batch']
        self.assertEqual((batch['completed_games'], batch['interrupted_games'], batch['not_started_games']), (1, 1, 8))
        self.assertEqual(batch['mean_ante_completed'], 1)
        self.assertEqual(batch['mean_blinds_cleared_completed'], 1)
        self.assertFalse(batch['is_full_batch'])
        self.assertIsNone(batch['mean_ante_full_batch'])
        self.assertEqual(summary['games'][0]['total_score_exact'], 550)
        self.assertIsNone(summary['games'][1]['total_score_exact'])
        self.assertEqual(summary['games'][0]['run_seed'], '76JWMHDF')

    def test_animation_read_is_not_a_zero_scoring_hand(self):
        game = summarize_events(rows(), 'sample')['games'][0]
        self.assertEqual(game['best_hand_score_observed'], 400)
        self.assertEqual(game['observed_scored_plays'], 2)
        self.assertEqual(game['plays'], 2)

    def test_full_ten_game_batch_populates_full_batch_averages(self):
        events = []
        for number in range(1, 11):
            for row in rows()[:14]:
                copied = dict(row)
                if copied['event'] in ('game_start', 'game_end'):
                    copied['number'] = number
                events.append(copied)
        batch = summarize_events(events, 'sample')['batch']
        self.assertTrue(batch['is_full_batch'])
        self.assertEqual(batch['completed_games'], 10)
        self.assertEqual(batch['mean_ante_full_batch'], 1)
        self.assertEqual(batch['mean_total_score_full_batch'], 550)

    def test_boss_cashout_does_not_change_the_finished_blind_ante(self):
        events = rows()[:8]
        events[2]['blind'] = 2
        events[7]['counters']['ante'] = 2
        blind = summarize_events(events, 'sample')['blinds'][0]
        self.assertEqual(blind['ante'], 1)

    def test_conflicting_final_score_does_not_become_an_exact_total(self):
        events = rows()
        events.insert(12, dict(time=events[12]['time'], event='observed_state', state={'round_score': 200}))
        summary = summarize_events(events, 'sample')
        self.assertFalse(summary['games'][0]['score_complete'])
        self.assertIsNone(summary['games'][0]['total_score_exact'])

    def test_cached_snapshot_precedes_same_timestamp_game_end(self):
        events = rows()[:13]
        snapshot = dict(events.pop(), scene='loss', source='saved_settled_frame_ocr')
        snapshot['time'] = events[-1]['time']
        events += [dict(time=snapshot['time'], event='scene', name='loss'),
                   dict(time=snapshot['time'], event='game_end', number=1, result='loss')]
        merged = merge_saved_snapshots(events, [snapshot])
        game = summarize_events(merged, 'sample')['games'][0]
        self.assertEqual(game['last_blind_score'], 150)
        self.assertTrue(game['score_complete'])

    def test_export_is_idempotent_and_preserves_other_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            save_history(summarize_events(rows(), 'first'), directory)
            save_history(summarize_events(rows(), 'second'), directory)
            save_history(summarize_events(rows(), 'first'), directory)
            with (Path(directory) / 'games.csv').open(encoding='utf-8-sig', newline='') as stream:
                records = list(csv.DictReader(stream))
            self.assertEqual(len(records), 20)
            self.assertEqual(sum(r['batch_id'] == 'first' for r in records), 10)
            self.assertTrue((Path(directory) / 'first/summary.json').exists())

    def test_version_change_on_resume_is_explicit(self):
        events = rows()
        events.insert(16, dict(time=events[16]['time'], event='batch_resume', code_version={'id': 'b'}, target_games=10))
        summary = summarize_events(events, 'sample')
        self.assertEqual(summary['batch']['code_version'], 'mixed')
        self.assertEqual(summary['games'][0]['code_version'], 'a')
        self.assertEqual(summary['games'][1]['code_version'], 'mixed')

    def test_truncated_last_line_is_marked_and_interior_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'events.jsonl'
            path.write_text(json.dumps(rows()[0]) + '\n{"event":', encoding='utf-8')
            records, warnings = read_events(path)
            self.assertEqual(len(records), 1)
            self.assertEqual(len(warnings), 1)
            path.write_text('{\n' + json.dumps(rows()[0]) + '\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                read_events(path)


if __name__ == '__main__':
    unittest.main()
