"""Replay retained observations; never connects to a game or predicts a win."""
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perception import parse_joker_tooltip
from planning import shop_forecast
from recovery import state_from_dict
from run_history import code_version
from strategy import Card, Joker, ShopItem, decide_shop, score_hand


def corrected_state(row):
    state = state_from_dict(row['state'])
    proof = row.get('fortune_proof')
    if proof:
        state = replace(state, jokers=tuple(
            parse_joker_tooltip(j, proof[2]) if j.key == 'fortune_teller' else j
            for j in state.jokers))
    return state


def replay_counter_pixels(inputs):
    import cv2
    import numpy as np
    from perception import read_text, joker_tooltip_words, verify_current_multiplier
    from strategy import RunState
    rows = []
    for row in inputs['tooltips']:
        path = ROOT / row['retained_frame']
        frame = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), 1)
        frame = cv2.resize(frame, (2064, 1191), interpolation=cv2.INTER_NEAREST)
        # The first saved roster contains one Joker; the others contain five.
        center = 1020 if path.name.startswith('0054-') else 660
        words, _ = joker_tooltip_words(read_text(frame, (525, 330, 2020, 850)), center)
        observed = parse_joker_tooltip(Joker('Fortune Teller'), row['tooltip'])
        verified = verify_current_multiplier(frame, words, observed, RunState())
        rows.append(dict(frame=row['retained_frame'], time=row['time'], slot_x=center,
                         value=verified.value, verified=bool(verified.get('observed_value')),
                         bound=bool(verified.get('counter_lower_bound'))))
    return rows


def main():
    inputs = json.loads((ROOT / 'tests/fixtures/strategy_review_20261011.json').read_text('utf-8'))
    counter_pixels = replay_counter_pixels(inputs)
    (ROOT / 'analysis/fortune_counter_replay_20261011.json').write_text(
        json.dumps(counter_pixels, ensure_ascii=False, indent=2), encoding='utf-8')
    scores, shops = [], []
    for row in inputs['scores']:
        state = corrected_state(row)
        cards = [Card(**c) for c in row['cards']]
        result = score_hand([cards[i] for i in row['order']], state,
                            [c for i, c in enumerate(cards) if i not in row['order']])
        scores.append(dict(time=row['time'], actual=row['actual'], old=row['logged'],
                           corrected=result.score, observed_fortune=next(
                               j.value for j in state.jokers if j.key == 'fortune_teller'),
                           proof_time=row['fortune_proof'][0], category=result.category))
    for row in inputs['shops']:
        state = corrected_state(row)
        items = [ShopItem(**{**v, 'joker': Joker(**{**v['joker'], 'data': tuple(
                    tuple(d) for d in v['joker'].get('data', ()))}) if v.get('joker') else None})
                 for v in row['items']]
        if '09:34:43' in row['time']:
            # The recorded offer identity is retained; its target was missing
            # from the old state even though the tooltip explicitly says Pair.
            items = [replace(i, joker=parse_joker_tooltip(i.joker, inputs['todo_offer']['tooltip']))
                     if i.name == 'To Do List' else i for i in items]
        before = shop_forecast(state)
        decision = decide_shop(items, state, reroll_cost=row.get('reroll_cost'))
        record = dict(time=row['time'], old=row['decision'], revised=asdict(decision),
                      predicted_before=asdict(before))
        shops.append(record)
        print(json.dumps(dict(time=row['time'], old=row['decision']['reason'],
                              revised=decision.reason), ensure_ascii=True), flush=True)
    result = dict(version=code_version()['id'], scope='offline; no new gameplay',
                  scores=scores, shops=shops, counter_pixels=counter_pixels,
                  score_checks=dict(count=len(scores),
                      old_within_one=sum(abs(r['actual']-r['old']) <= 1 for r in scores),
                      corrected_within_one=sum(abs(r['actual']-r['corrected']) <= 1 for r in scores)))
    (ROOT / 'analysis/strategy_replay_20261011.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result['score_checks']), flush=True)


if __name__ == '__main__':
    main()
