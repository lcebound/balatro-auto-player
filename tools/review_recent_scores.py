"""Match settled score deltas to their recorded policy inputs, offline only."""
import json
import sys
from pathlib import Path
from collections import Counter
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recovery import state_from_dict
from strategy import Card, score_hand

def cases(directory):
    state = decision = pending = waiting = None
    for line in (directory / 'events.jsonl').read_text('utf-8').splitlines():
        row = json.loads(line)
        event = row['event']
        if event == 'observed_state':
            state = row['state']
        elif event in ('hand_decision', 'hand_decision_after_order_fallback'):
            decision = row
            if state is not None and 'jokers' in row:
                # Reordering may have been verified since observed_state.
                state = {**state, 'jokers': row['jokers']}
        elif event == 'action_prepared':
            pending = row
        elif event == 'action_not_applied':
            pending = None
        elif event == 'action_confirmed' and row['kind'] == 'play' and pending:
            waiting = pending, decision, state
        if waiting and event in ('observed_state', 'round_snapshot'):
            value = row.get('state', row.get('counters', {})).get('round_score')
            if value is None:
                continue
            action, played, played_state = waiting
            actual = value - action['before']['round_score']
            if actual < 0:
                continue
            cards = [Card(**c) for c in played['card_details']]
            order = played['play_order'] or played['best_indices']
            estimate = score_hand([cards[i] for i in order], state_from_dict(played_state),
                                  [c for i, c in enumerate(cards) if i not in order])
            yield dict(batch=directory.name, time=action['time'], actual=actual,
                       logged=action['details'].get('estimated_score', 0),
                       replay=estimate.score, state=played_state, cards=played['card_details'],
                       order=order, category=estimate.category, notes=estimate.notes)
            waiting = None

if __name__ == '__main__':
    rows = [c for directory in sorted((ROOT / 'runs').glob('autonomous-20261009-*'))
            for c in cases(directory)]
    path = ROOT / 'analysis/score_review_20261009.json'
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    print('plays', len(rows), 'positive deltas', sum(c['actual'] > 0 for c in rows))
    ranked = sorted(rows, key=lambda c: abs(c['logged'] - c['actual']) / max(1, c['actual']), reverse=True)
    for c in ranked[:18]:
        print(json.dumps(dict(batch=c['batch'], time=c['time'], actual=c['actual'], logged=c['logged'],
                              replay=c['replay'], category=c['category'],
                              jokers=[(j['name'], j['value'], j['data']) for j in c['state']['jokers']]),
                         ensure_ascii=False))
