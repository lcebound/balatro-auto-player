"""Retain bounded historical inputs and compare policies without game access."""
import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_history import read_events, code_version
from recovery import state_from_dict
from strategy import ShopItem, decide_shop
from planning import shop_forecast

BATCH = 'autonomous-20261013-232757'
FIXTURE = ROOT / 'tests/fixtures/decision_review_20261014.json'
TIMES = {
    'irrelevant_early_planet': '2026-10-14T01:45:34.971223+08:00',
    'thin_margin_pack': '2026-10-14T02:18:40.502274+08:00',
    'late_economy': '2026-10-14T02:19:42.713481+08:00',
    'unsupported_zero_scaler': '2026-10-14T00:39:50.387403+08:00',
    'conditional_multiplier': '2026-10-14T04:23:06.656119+08:00',
    'known_multiplier_rescue': '2026-10-14T00:49:34.249011+08:00',
}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', 'utf-8')


def prepare():
    if FIXTURE.exists():
        raise RuntimeError('Historical inputs already retained; do not overwrite them')
    source = ROOT / 'runs' / BATCH / 'events.jsonl'
    events, warnings = read_events(source)
    assert not warnings
    kept = ROOT / 'analysis/retained_run_evidence' / BATCH / 'decision-review-20261014'
    kept.mkdir(parents=True, exist_ok=True)
    proofs, cases, last, hand_frame, last_state, game = [], [], None, None, None, 0

    def retain(frame, role):
        src = Path(frame['screenshot'])
        target = kept / src.name
        shutil.copy2(src, target)
        proof = dict(role=role, source=str(src), retained=str(target),
                     sha256=hashlib.sha256(src.read_bytes()).hexdigest())
        proofs.append(proof)
        return proof

    for event in events:
        if event['event'] == 'game_start':
            game, hand_frame, last_state = event['number'], None, None
        elif event['event'] == 'observe':
            last = event
            if event.get('label') == 'hand-before-inspection':
                hand_frame = event
        elif event['event'] == 'observed_state':
            last_state = event
        if event['time'] in TIMES.values() and event['event'] == 'shop_decision':
            name = next(k for k, v in TIMES.items() if v == event['time'])
            cases.append(dict(name=name, game=game, event=event,
                              proof=retain(last, name)))
        if event['event'] == 'game_end' and game in (2, 3):
            cases.append(dict(name=f'game_{game}_loss', game=game, event=event,
                              state=last_state, proof=retain(last, f'game_{game}_terminal'),
                              hand_proof=retain(hand_frame, f'game_{game}_last_hand')))
    assert len(cases) == 8
    summary = json.loads((ROOT / 'results' / BATCH / 'summary.json').read_text('utf-8'))
    write(FIXTURE, dict(batch=BATCH, actual_version=summary['batch']['code_version'],
                       source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), cases=cases))
    write(kept / 'manifest.json', dict(batch=BATCH, proofs=proofs,
                                     source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    shutil.copy2(ROOT / 'results' / BATCH / 'summary.json', kept / 'summary.json')
    print(json.dumps(dict(prepared=len(cases), retained=str(kept)), ensure_ascii=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--case', action='append')
    args = parser.parse_args()
    if args.prepare:
        prepare()
        return
    data = json.loads(FIXTURE.read_text('utf-8'))
    version = code_version()['id']
    name = 'decision_review_baseline_20261014.json' if args.baseline else 'decision_review_replay_20261014.json'
    path = ROOT / 'analysis' / name
    if args.baseline and path.exists():
        raise RuntimeError('Baseline already retained')
    rows = []
    if args.resume and path.exists():
        previous = json.loads(path.read_text('utf-8'))
        assert previous['version'] == version, 'Cannot mix different replay versions'
        rows = previous['cases']
    finished = {row['name'] for row in rows}
    for case in data['cases']:
        event = case['event']
        if event['event'] != 'shop_decision' or args.case and case['name'] not in args.case:
            continue
        if case['name'] in finished:
            continue
        state = state_from_dict(event['state'])
        items = [ShopItem(item['kind'], item['name'], item['price'],
                          state_from_dict({'jokers': [item['joker']]}).jokers[0]
                          if item.get('joker') else None,
                          item.get('edition', ''), tuple(item.get('pack_contents', ())))
                 for item in event['items']]
        forecast = shop_forecast(state)
        decision = decide_shop(items, state, reroll_cost=event.get('reroll_cost'))
        row = dict(name=case['name'], decision=asdict(decision),
                   coverage=forecast.coverage, clear_rate=forecast.clear_rate,
                   weaknesses=forecast.profile.weaknesses,
                   conditional_rates=getattr(forecast.next_round, 'conditional_xmult_rates', ()))
        rows.append(row)
        write(path, dict(version=version, batch=BATCH, actual_version=data['actual_version'],
                         scope='offline historical inputs; no new gameplay or measured improvement',
                         complete=False, cases=rows))
        print(json.dumps(row, ensure_ascii=True), flush=True)
    assert code_version()['id'] == version, 'Code changed during replay'
    expected = {case['name'] for case in data['cases'] if case['event']['event'] == 'shop_decision'}
    write(path, dict(version=version, batch=BATCH, actual_version=data['actual_version'],
                     scope='offline historical inputs; no new gameplay or measured improvement',
                     complete=expected <= {row['name'] for row in rows}, cases=rows))


if __name__ == '__main__':
    main()
