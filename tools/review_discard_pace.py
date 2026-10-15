"""Find real discards of small hands already above safe scoring pace."""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recovery import state_from_dict
from strategy import Card, score_hand
from dataclasses import replace

cases = []
for directory in sorted((ROOT / 'runs').glob('autonomous-20261009-*')):
    state = decision = None
    for line in (directory / 'events.jsonl').read_text('utf-8').splitlines():
        row = json.loads(line)
        if row['event'] == 'observed_state':
            state = row['state']
        elif row['event'] == 'hand_decision':
            decision = row
            state = {**state, 'jokers': row['jokers']}
        elif row['event'] == 'discard_decision' and row['indices'] and decision and state:
            if decision['category'] not in ('pair', 'high_card'):
                continue
            probe = state_from_dict(state)
            cards = [Card(**c) for c in decision['card_details']]
            order = decision['play_order'] or decision['best_indices']
            selected = [cards[i] for i in order]
            held = [c for i, c in enumerate(cards) if i not in order]
            floor = score_hand(selected, replace(probe, score_mode='conservative'), held).score
            pace = probe.score_needed / max(1, probe.hands_left)
            if probe.hands_left > 1 and pace > 0 and floor >= 1.1 * pace:
                cases.append(dict(batch=directory.name, time=row['time'], state=state, cards=decision['card_details'],
                                  order=order, category=decision['category'], floor=floor, pace=pace,
                                  discard=row, reason=decision['reason']))
(ROOT / 'analysis/discard_pace_review_20261009.json').write_text(json.dumps(cases, indent=2), 'utf-8')
print('on-pace small-hand discards', len(cases))
for case in cases[:6]:
    print(case['time'], case['category'], round(case['floor']), round(case['pace']),
          [j['name'] for j in case['state']['jokers']], case['discard']['reason'])
