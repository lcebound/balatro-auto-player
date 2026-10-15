"""Preserve and replay actual Bonus Card score residuals without device access."""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perception import parse_card_tooltip
from recovery import state_from_dict
from strategy import Card, score_hand


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'runs/autonomous-20261006-030032')
    parser.add_argument('--output', type=Path, default=ROOT / 'tests/fixtures/bonus_score_residuals_20261006.json')
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.run / 'events.jsonl').read_text('utf-8').splitlines()]
    observed, state, decision, pending, waiting = {}, None, None, None, None
    cases = []
    for row in rows:
        event = row['event']
        if event == 'game_end':
            break
        if event == 'card_observed':
            observed[row['index']] = row
        elif event == 'observed_state':
            state = row['state']
        elif event == 'hand_decision':
            decision = row
        elif event == 'action_prepared':
            pending = row
        elif event == 'action_confirmed' and row['kind'] == 'play' and pending:
            waiting = (pending, decision, state, dict(observed))
        if waiting and event in ('observed_state', 'round_snapshot'):
            value = row.get('state', row.get('counters', {})).get('round_score')
            if value is None:
                continue
            action, played, played_state, tooltips = waiting
            actual = value - action['before']['round_score']
            if actual < 0:
                continue
            estimated = action['details'].get('estimated_score', 0)
            if estimated - actual > max(20, actual * .1):
                cards = [Card(**c) for c in played['card_details']]
                corrected = [parse_card_tooltip(c, tooltips[i]['tooltip'], require_identity=True)
                             if c.enhancement == 'bonus' else c for i, c in enumerate(cards)]
                order = played['play_order'] or played['best_indices']
                estimate = score_hand([corrected[i] for i in order], state_from_dict(played_state),
                                      [c for i, c in enumerate(corrected) if i not in order])
                case = dict(time=action['time'], old_estimate=estimated, actual=actual,
                            corrected_estimate=estimate.score, state=played_state,
                            cards=[asdict(c) for c in cards], order=order,
                            bonus_tooltips={str(i): tooltips[i]['tooltip'] for i, c in enumerate(cards)
                                            if c.enhancement == 'bonus'})
                cases.append(case)
                print(json.dumps({k: case[k] for k in ('time', 'old_estimate', 'actual', 'corrected_estimate')}))
            waiting = None
    args.output.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Preserved {len(cases)} residuals: {args.output}')


if __name__ == '__main__':
    main()
