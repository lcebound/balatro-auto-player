"""Read-only status probe for supervising full autonomous games."""
import argparse
import json
import time
import sys
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_history import summarize_events
from runtime_issues import terminal_failures, unresolved_recovery


def probe(directory, pid):
    status_path = directory / 'status.json'
    status = json.loads(status_path.read_text('utf8')) if status_path.exists() else {}
    events = []
    log_path = directory / 'events.jsonl'
    if log_path.exists():
        for line in log_path.read_text('utf8').splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # Writer may be partway through its final line.
    recent = {}
    current_game_events = []
    for row in events:
        recent[row['event']] = row
        if row['event'] == 'game_start':
            current_game_events = []
        current_game_events.append(row)
    state = recent.get('observed_state', {}).get('state', {})
    decision = recent.get('hand_decision', {})
    failures = terminal_failures(events)
    recoveries = [row for row in events if row['event'] == 'automatic_recovery']
    summary = summarize_events(events, directory.name, status.get('target_games', 10))
    current_game = next((game for game in summary['games']
                         if game['game_number'] == status.get('current_game')), {})
    return {
        'process_alive': psutil.pid_exists(pid),
        'status': status,
        'last_event': {k: events[-1].get(k) for k in ('time', 'event', 'reason', 'label')} if events else None,
        'scene': recent.get('scene', {}).get('name'),
        'position': {k: state.get(k) for k in ('ante', 'boss', 'round_score', 'target_score', 'hands_left', 'discards_left')},
        'last_decision': {k: decision.get(k) for k in ('category', 'expected_score')},
        'current_jokers': [(joker['name'], joker['value']) for joker in state.get('jokers', [])],
        'recent_score_comparisons': current_game.get('score_comparisons', [])[-3:],
        'games_started': [row for row in events if row['event'] == 'game_start'],
        'games_ended': [row for row in events if row['event'] == 'game_end'],
        'blinds_cleared_current_game': sum(row['event'] == 'round_snapshot' and
                                         row.get('scene') == 'cashout'
                                         for row in current_game_events),
        'last_settled_snapshot': recent.get('round_snapshot'),
        'failures': failures[-3:],
        'recoveries': recoveries[-3:],
        'unresolved_recovery': unresolved_recovery(events),
        'issue_counts': status.get('issue_counts', {}),
        'last_screenshot': recent.get('observe', {}).get('screenshot'),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--seconds', type=float, default=0)
    parser.add_argument('--poll-seconds', type=float, default=15)
    parser.add_argument('--compact', action='store_true')
    args = parser.parse_args()
    deadline = time.monotonic() + args.seconds
    while True:
        result = probe(args.run_dir, args.pid)
        status = result['status']
        if (result['failures'] or not result['process_alive'] or
                status.get('status') in ('failed', 'complete', 'waiting_for_desktop') or
                (status.get('finished_games', 0) >= 3 and status.get('game_active') and
                 result['scene'] in ('hand', 'blind')) or time.monotonic() >= deadline):
            if args.compact:
                result = {k: result[k] for k in ('process_alive', 'scene', 'position',
                    'blinds_cleared_current_game', 'failures', 'recoveries',
                    'unresolved_recovery', 'issue_counts', 'last_event', 'last_screenshot')} | {
                    'games': f"{status.get('finished_games', 0)}/{status.get('target_games', 10)}",
                    'current_game': status.get('current_game'), 'status': status.get('status'),
                    'game_active': status.get('game_active')}
            print(json.dumps(result, ensure_ascii=True), flush=True)
            return
        time.sleep(min(max(1, min(60, args.poll_seconds)), max(0, deadline - time.monotonic())))


if __name__ == '__main__':
    main()
