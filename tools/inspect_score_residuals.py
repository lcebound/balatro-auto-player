"""Compact detailed inputs for materially wrong real-play predictions."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
rows = json.loads((ROOT / 'analysis/score_review_20261009.json').read_text('utf-8'))
for c in rows:
    if abs(c['logged'] - c['actual']) <= max(20, c['actual'] * .07):
        continue
    state = c['state']
    print(json.dumps(dict(time=c['time'], actual=c['actual'], logged=c['logged'],
                          boss=state['boss'], ante=state['ante'], order=c['order'],
                          cards=[(v['rank'], v['suit'], v['enhancement'], v['debuffed']) for v in c['cards']],
                          state={k:state[k] for k in ('draw_pile_size', 'hands_left', 'discards_left', 'levels')},
                          notes=c['notes']), ensure_ascii=False))
