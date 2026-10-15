"""Bounded real-run inputs, policy replay and growth diagnostics; no game UI."""
import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recovery import state_from_dict
from run_history import code_version, read_events
from strategy import Card, ShopItem, decide_play, decide_discard, decide_shop
from planning import shop_forecast

BATCH = 'autonomous-20261014-234452'
FIXTURE = ROOT / 'tests/fixtures/growth_review_20261015.json'
SHOPS = {713: 'early_square', 2309: 'early_green', 15048: 'visible_constellation_supply',
         17878: 'lethal_pack', 34155: 'wealthy_lethal', 38841: 'late_green_needle',
         28751: 'holographic_income_guard', 22481: 'known_chad_combo',
         19996: 'water_round_budget'}
HANDS = {20220: 'water_growing_obelisk', 20298: 'water_reset_obelisk',
         20339: 'water_last_hand', 34581: 'arm_last_hand',
         39102: 'needle_discard'}


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', 'utf8')


def prepare():
    if FIXTURE.exists():
        raise RuntimeError('Historical fixture already exists')
    source = ROOT / 'runs' / BATCH / 'events.jsonl'
    rows, warnings = read_events(source)
    assert not warnings
    evidence = ROOT / 'analysis/retained_run_evidence' / BATCH / 'growth-review-20261015'
    evidence.mkdir(parents=True, exist_ok=True)
    proofs = {}

    def retain(event):
        src = ROOT / 'runs' / BATCH / Path(event['screenshot'].replace('\\', '/')).name
        dest = evidence / src.name
        if not dest.exists():
            shutil.copy2(src, dest)
        assert src.read_bytes() == dest.read_bytes()
        proof = dict(path=dest.relative_to(ROOT).as_posix(),
                     sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
        proofs[proof['path']] = proof
        return proof

    game, last_frame, hand_frame, state = 0, None, None, None
    cases, outcomes, runtime = [], [], []
    for index, row in enumerate(rows):
        kind = row['event']
        if kind == 'game_start':
            game = row['number']
        if kind == 'observe':
            last_frame = row
            if row['label'] == 'hand-before-inspection':
                hand_frame = row
        if kind == 'observed_state':
            state = row['state']
        if index in SHOPS:
            assert kind == 'shop_decision'
            cases.append(dict(name=SHOPS[index], kind='shop', game=game,
                event_index=index, event=row, proof=retain(last_frame)))
        if index in HANDS:
            assert kind == 'hand_decision'
            # The event's Joker roster is the exact roster used to decide.
            cases.append(dict(name=HANDS[index], kind='hand', game=game,
                event_index=index, event=row, state=dict(state, jokers=row['jokers']),
                proof=retain(hand_frame)))
        if kind == 'game_end':
            outcomes.append(dict(game=game, event=row, proof=retain(last_frame)))
        if kind in ('automatic_recovery', 'recovery_resolved', 'batch_error', 'fatal'):
            runtime.append(dict(event_index=index, event=row))
    summary = json.loads((ROOT / 'results' / BATCH / 'summary.json').read_text('utf8'))
    write(FIXTURE, dict(batch=BATCH, actual_version=summary['batch']['code_version'],
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), cases=cases,
        outcomes=outcomes))
    write(evidence / 'manifest.json', dict(proofs=list(proofs.values()),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    write(evidence / 'runtime-events.json', runtime)
    shutil.copy2(ROOT / 'results' / BATCH / 'summary.json', evidence / 'summary.json')
    for src in (ROOT / 'runs' / BATCH / 'issue-reviews').glob('*.json'):
        shutil.copy2(src, evidence / src.name)
    print(json.dumps(dict(cases=len(cases), frames=len(proofs)), ensure_ascii=False))


def items_from_event(event):
    return [ShopItem(item['kind'], item['name'], item['price'],
        state_from_dict({'jokers': [item['joker']]}).jokers[0] if item.get('joker') else None,
        item.get('edition', ''), tuple(item.get('pack_contents', ()))) for item in event['items']]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--case', action='append')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
        return
    data = json.loads(FIXTURE.read_text('utf8'))
    version = code_version()['id']
    path = ROOT / 'analysis' / ('growth_baseline_20261015.json' if args.baseline
                              else 'growth_replay_20261015.json')
    done = []
    if path.exists():
        if not args.resume:
            raise RuntimeError('Use --resume to preserve completed cases')
        previous = json.loads(path.read_text('utf8'))
        assert previous['version'] == version, 'Replay version changed'
        done = previous['cases']
    completed = {r['name'] for r in done}
    for case in data['cases']:
        if case['name'] in completed or args.case and case['name'] not in args.case:
            continue
        event = case['event']
        state = state_from_dict(event['state'] if case['kind'] == 'shop' else case['state'])
        if case['kind'] == 'shop':
            forecast = shop_forecast(state)
            decision = decide_shop(items_from_event(event), state, reroll_cost=event.get('reroll_cost'))
            row = dict(name=case['name'], decision=asdict(decision), coverage=forecast.coverage,
                clear_rate=forecast.clear_rate, growth=forecast.next_round.growth_per_round,
                development=forecast.development_score)
        else:
            hand = [Card(**c) for c in event['card_details']]
            decision = decide_play(hand, state)
            row = dict(name=case['name'], decision=asdict(decision))
            if case['name'] == 'needle_discard':
                row['discard'] = asdict(decide_discard(hand, state, current=decision))
        done.append(row)
        assert code_version()['id'] == version, 'Code changed during replay'
        write(path, dict(batch=BATCH, actual_version=data['actual_version'], version=version,
            scope='offline fixed inputs; scores and growth are predictions, no gameplay',
            complete=len(done) == len(data['cases']), cases=done))
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
