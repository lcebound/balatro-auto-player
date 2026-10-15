"""Extract the real malformed Vampire observation and its nearby frame."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
rows = [json.loads(line) for line in (ROOT / 'runs/autonomous-20261009-062544/events.jsonl').read_text('utf-8').splitlines()]
previous = None
failures = []
for i, row in enumerate(rows):
    if row['event'] == 'joker_observed' and row['joker']['name'] == 'Vampire':
        if row['joker']['value'] > 2 or 'observed_value' not in dict(row['joker']['data']):
            frame = next((r['screenshot'] for r in reversed(rows[:i]) if r['event'] == 'observe'), None)
            failures.append(dict(time=row['time'], value=row['joker']['value'], tooltip=row['tooltip'],
                                 previous=previous, screenshot=frame))
        previous = dict(time=row['time'], value=row['joker']['value'], tooltip=row['tooltip'])
(ROOT / 'analysis/vampire_value_failures_20261009.json').write_text(json.dumps(failures, indent=2), 'utf-8')
for case in failures:
    if case['value'] > 2:
        print(case['time'], case['value'], Path(case['screenshot']).name, case['tooltip'])
