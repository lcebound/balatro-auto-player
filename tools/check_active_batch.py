"""Compact read-only snapshot for intermittent batch supervision."""
import json
from collections import Counter
from pathlib import Path

root = Path(__file__).resolve().parents[1]
pointer = json.loads((root / 'runs/active-batch.json').read_text('utf-8-sig'))
run = Path(pointer['run_dir'])
events = []
for line in (run / 'events.jsonl').read_text('utf-8').splitlines():
    try:
        events.append(json.loads(line))
    except json.JSONDecodeError:
        pass  # A concurrent append may leave the last line incomplete.
status = json.loads((run / 'status.json').read_text('utf-8'))
state = json.loads((run / 'policy-state.json').read_text('utf-8'))
rounds = [e for e in events if e['event'] == 'round_snapshot']
ends = [e for e in events if e['event'] == 'game_end']
errors = [dict(time=e['time'], event=e['event'], error=e.get('error'))
          for e in events if e['event'] in ('automatic_recovery', 'fatal', 'batch_error')]
print(json.dumps(dict(batch=run.name, pid=pointer['pid'], status=status['status'],
    finished=status['finished_games'], game=status['current_game'], ante=state.get('ante'),
    score=state.get('round_score'), target=state.get('target_score'),
    hands=state.get('hands_left'), discards=state.get('discards_left'),
    last=dict(time=events[-1]['time'], event=events[-1]['event']),
    scenes=Counter(e.get('name') for e in events if e['event'] == 'scene'),
    clears=len(rounds), jokers=[j['name'] for j in state.get('jokers', [])],
    issues=status.get('issue_counts'), ends=ends, errors=errors[-2:]), ensure_ascii=True))
