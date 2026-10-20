"""Retain real shop inputs and compare growth policies without a device."""
import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from planning import shop_forecast
from recovery import state_from_dict
from run_history import code_version, summarize_events
from strategy import decide_shop, _purchase_state
from tools.review_growth_20261015 import items_from_event

BATCH = 'autonomous-20261017-012806'
FIXTURE = ROOT / 'tests/fixtures/growth_decisions_20261017.json'
EVIDENCE = ROOT / 'analysis/retained_run_evidence' / BATCH / 'growth-decisions-20261017'
SELECTED = {543: 'early_economy_before_tower', 561: 'early_golden_stencil',
            2032: 'conditional_obelisk', 2721: 'castle_price_suspicious',
            3604: 'rebate_or_pack', 4140: 'late_discard_voucher',
            4830: 'late_red_card', 5371: 'late_burnt_available',
            6870: 'rocket_purchase', 8202: 'rocket_replacement',
            8258: 'green_growth_skipped', 9285: 'burnt_growth_skipped',
            9725: 'late_small_pack', 9788: 'late_jumbo_pack'}
PROBES = {'early_golden_stencil': [(0, -1)], 'green_growth_skipped': [(1, 4), (1, 0)],
          'burnt_growth_skipped': [(0, 4), (0, 0)], 'late_burnt_available': [(0, 3)]}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', 'utf-8')


def prepare():
    if FIXTURE.exists():
        raise RuntimeError('Historical inputs already retained')
    path = ROOT / 'runs' / BATCH / 'events.jsonl'
    events = [json.loads(s) for s in path.read_text('utf-8').splitlines()]
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    game, last, cases = 0, None, []
    for index, event in enumerate(events):
        if event['event'] == 'game_start':
            game = event['number']
        elif event['event'] == 'observe':
            last = event
        if index in SELECTED:
            assert event['event'] == 'shop_decision'
            source = Path(last['screenshot'])
            target = EVIDENCE / source.name
            shutil.copy2(source, target)
            cases.append(dict(name=SELECTED[index], event_index=index, game=game, event=event,
                proof=dict(path=target.relative_to(ROOT).as_posix(),
                           sha256=hashlib.sha256(target.read_bytes()).hexdigest())))
    for name in ('strategy.py', 'planning.py', 'engine_development.py', 'economy.py'):
        target = EVIDENCE / 'before' / name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(ROOT / name, target)
    write(FIXTURE, dict(batch=BATCH, actual_version=events[0]['code_version']['id'],
        before_version=code_version(), original_events_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), cases=cases))
    write(EVIDENCE / 'summary.json', summarize_events(events, BATCH))
    print('retained', len(cases), 'real shop states', flush=True)


def forecast_details(f):
    return dict(coverage=f.coverage, clear_rate=f.clear_rate,
        safe_score=f.next_round.safe_score, target=f.next_round.target,
        sustainable=f.sustainable_score, development=f.development_score,
        weaknesses=f.profile.weaknesses, frequencies=f.next_round.frequencies,
        growth=f.next_round.growth_per_round, plays=f.next_round.plays_per_round,
        levels=f.next_round.level_gains_per_round, conditional=f.next_round.conditional_xmult_rates)


def replay(label):
    data = json.loads(FIXTURE.read_text('utf-8'))
    version = code_version()
    if label == 'baseline' and version != data['before_version']:
        raise RuntimeError('Baseline requires the exact retained source version')
    path = ROOT / ('analysis/growth_decisions_20261017_' + label + '.json')
    output = []
    for case in data['cases']:
        state = state_from_dict(case['event']['state'])
        items = items_from_event(case['event'])
        decision = decide_shop(items, state, reroll_cost=case['event'].get('reroll_cost'))
        detail = forecast_details(shop_forecast(state))
        probes = []
        for item_index, sell in PROBES.get(case['name'], []):
            projected = _purchase_state(items[item_index], state, sell)
            probes.append(dict(item=items[item_index].name, sell=sell, **forecast_details(shop_forecast(projected))))
        result = dict(name=case['name'], old_decision=case['event']['decision'],
                      decision=asdict(decision), before=detail, probes=probes)
        output.append(result)
        assert code_version() == version, 'Source changed during fixed-input replay'
        write(path, dict(batch=BATCH, actual_version=data['actual_version'], version=version['id'],
            scope='offline policy estimates; no replayed decisions were executed',
            complete=len(output) == len(data['cases']), cases=output))
        print(json.dumps(dict(name=case['name'], action=decision.action, item=decision.item_index,
            sell=decision.sell_index, coverage=round(detail['coverage'], 3),
            development=round(detail['development'], 1), weaknesses=detail['weaknesses']), ensure_ascii=False), flush=True)
    if label == 'after':
        replay_prices(data, version)


def replay_prices(data, version):
    """Keep raw logged states separate from screenshot-corrected observations."""
    from perception import read_offer_price
    from tools.replay_observation_speed import load_frame
    prices = json.loads((ROOT / 'tests/fixtures/shop_price_20261017.json').read_text('utf-8'))
    observations = []
    for case in prices['cases']:
        path = ROOT / case['proof']['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == case['proof']['sha256']
        value = read_offer_price(load_frame(path), case['original'], case['confidence'], case['center'])
        assert value == case['expected']
        observations.append(dict(original=case['original'], recovered=value, proof=case['proof']))
    case = next(c for c in data['cases'] if c['name'] == 'castle_price_suspicious')
    items = items_from_event(case['event'])
    from dataclasses import replace
    items[1] = replace(items[1], price=observations[0]['recovered'])
    decision = decide_shop(items, state_from_dict(case['event']['state']),
                           reroll_cost=case['event'].get('reroll_cost'))
    assert code_version() == version
    write(ROOT / 'analysis/growth_prices_20261017_after.json', dict(version=version['id'],
        scope='offline same-frame price OCR and corrected-input policy; no purchases executed',
        observations=observations, castle_corrected_price_decision=asdict(decision),
        corrected_price_before=json.loads((ROOT / 'analysis/castle_corrected_price_baseline_20261017.json').read_text('utf-8'))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--replay', choices=('baseline', 'after'))
    args = parser.parse_args()
    if args.prepare:
        prepare()
    if args.replay:
        replay(args.replay)
