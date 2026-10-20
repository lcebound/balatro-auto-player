"""Verify one autonomous whole game and the next game's independent start."""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_history import read_events
from runtime_issues import terminal_failures, unresolved_recovery
from recovery import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings-record', default='analysis/retained_run_evidence/game-speed-20261014/setting.json')
    args = parser.parse_args()
    settings_path = (ROOT / args.settings_record).resolve()
    assert settings_path.is_relative_to(ROOT) and settings_path.is_file()
    settings = json.loads(settings_path.read_text('utf-8-sig'))
    speed = settings.get('game_speed', settings.get('speed'))
    assert speed == 4, 'Expected verified 4x settings evidence'
    pointer = json.loads((ROOT / 'runs/active-batch.json').read_text('utf-8-sig'))
    run = Path(pointer['run_dir']).resolve()
    assert run.parent == (ROOT / 'runs').resolve()
    events, warnings = read_events(run / 'events.jsonl')
    status = json.loads((run / 'status.json').read_text('utf-8'))
    ends = [e for e in events if e.get('event') == 'game_end']
    first = next(e for e in ends if e['number'] == 1)
    assert first['result'] in ('win', 'loss')
    second = next(e for e in events if e.get('event') == 'game_start' and e['number'] == 2)
    after = [e for e in events if e['time'] > second['time']]
    assert any(e.get('event') == 'scene' and e.get('name') == 'hand' for e in after)
    second_frame = next(e['screenshot'] for e in after if e.get('event') == 'observe'
                        and e.get('label') == 'hand-before-inspection')
    assert status['status'] == 'running' and status['finished_games'] >= 1
    assert not warnings and not terminal_failures(events)
    assert unresolved_recovery(events) is None
    before = events[:events.index(first)]
    terminal_frame = next(e['screenshot'] for e in reversed(before) if e.get('event') == 'observe')
    summary = json.loads((ROOT / 'results' / run.name / 'summary.json').read_text('utf-8'))
    game = next(g for g in summary['games'] if g['game_number'] == 1)
    assert game['completed'] and game['result'] == first['result']
    report = dict(status='passed_first_whole_game_remaining_running', batch=run.name,
        runner_pid=pointer['pid'], updated_at=datetime.now().astimezone().isoformat(),
        code_version=summary['batch']['code_version'], game_speed=4,
        verified_full_games=1, target_games=10, remaining_at_handoff=10-status['finished_games'],
        first_game=game, first_end_event=first, terminal_screenshot=terminal_frame,
        second_game_start=second, second_hand_screenshot=second_frame,
        handoff_status=status, manual_gameplay_interventions=0,
        settings_navigation_record=str(settings_path.relative_to(ROOT)))
    atomic_json(ROOT / 'analysis' / (run.name + '-acceptance.json'), report)
    print(json.dumps(dict(batch=run.name, result=first['result'],
                         finished=status['finished_games'], current_game=status['current_game'])))


if __name__ == '__main__':
    main()
