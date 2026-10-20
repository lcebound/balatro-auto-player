"""Compare retained pre-Boss shops using exact before sources or current code."""
import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('label', choices=('before', 'after'))
label = parser.parse_args().label
data = json.loads((ROOT / 'tests/fixtures/boss_shop_weights_20261018.json').read_text('utf-8'))
sys.path.insert(0, str(ROOT))
if label == 'before':
    before = ROOT / 'analysis/retained_run_evidence/autonomous-20261018-001410/boss-review-20261018/before'
    for path in before.glob('*.py'):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == data['before_version']['files'][path.name]
    sys.path.insert(0, str(before))
from recovery import state_from_dict
from strategy import decide_shop
from planning import shop_forecast
from tools.review_growth_20261015 import items_from_event
from run_history import code_version

version = data['before_version']['id'] if label == 'before' else code_version()['id']
results = []
for case in data['cases']:
    e = case['event']
    state = state_from_dict(e['state'])
    decision = decide_shop(items_from_event(e), state, reroll_cost=e.get('reroll_cost'))
    f = shop_forecast(state)
    results.append(dict(index=case['index'], ante=state.ante, boss=state.upcoming_boss,
        weight=getattr(f, 'boss_weight', 1.), coverage=f.coverage, clear_rate=f.clear_rate,
        next_safe=f.next_round.safe_score, boss_safe=f.boss_round.safe_score if f.boss_round else None,
        decision=asdict(decision)))
    print(results[-1], flush=True)
(ROOT / f'analysis/boss_shop_weights_20261018_{label}.json').write_text(json.dumps(
    dict(version=version, scope='offline fixed observations; no purchases executed', cases=results),
    ensure_ascii=False, indent=2) + '\n', 'utf-8')
