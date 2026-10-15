"""Replay selected actual discard inputs; never performs UI actions."""
import json
import sys
from dataclasses import replace
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recovery import state_from_dict
from strategy import Card, PlayDecision, score_hand, decide_discard
rows = json.loads((ROOT / 'analysis/discard_pace_review_20261009.json').read_text('utf-8'))
results = []
for case in (rows[0], rows[1], rows[2]):
    state = state_from_dict(case['state'])
    cards = [Card(**c) for c in case['cards']]
    order = tuple(case['order'])
    estimate = score_hand([cards[i] for i in order], state, [c for i, c in enumerate(cards) if i not in order])
    current = PlayDecision(order, estimate, estimate.score, case['reason'], order)
    choice = decide_discard(cards, state, samples=8, current=current)
    results.append(dict(time=case['time'], category=case['category'], current=estimate.score,
                        score_needed=state.score_needed, hands=state.hands_left,
                        old_discard=case['discard']['indices'], new_discard=choice.indices,
                        old_mean=case['discard']['expected_score'], new_mean=choice.expected_score,
                        new_reason=choice.reason))
    print(json.dumps(results[-1]))
(ROOT / 'analysis/discard_policy_replay_20261009.json').write_text(json.dumps(results, indent=2), 'utf-8')
