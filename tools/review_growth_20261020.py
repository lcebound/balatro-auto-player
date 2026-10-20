"""Retain and replay P2's completed batch without a game connection."""
import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perception import (read_text, joker_tooltip_words, verify_current_multiplier)
from recovery import state_from_dict
from run_history import code_version, read_events, export_run
from strategy import Joker, decide_shop
from tools.review_growth_20261015 import items_from_event
from tools.replay_observation_speed import load_frame

BATCH = 'autonomous-20261019-234127'
BASE = ROOT / 'analysis/retained_run_evidence' / BATCH / 'growth-review-20261020'
FIXTURE = ROOT / 'tests/fixtures/growth_review_20261020.json'
SHOPS = {2390: 'holographic_rescue', 5276: 'early_obelisk', 12921: 'conditional_multiplier',
         14011: 'runner_growth', 14339: 'blueprint_upgrade', 17671: 'temperance_unpriced',
         18455: 'last_boss_gold', 18477: 'last_boss_sun', 19316: 'early_green',
         21711: 'supported_runner', 25493: 'late_blueprint', 27744: 'holographic_egg',
         29147: 'funded_hermit', 30147: 'destructive_madness', 32307: 'known_walkie',
         36212: 'burnt_resource_trade', 37015: 'marginal_strength'}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', 'utf8')


def retain():
    assert not FIXTURE.exists(), 'Do not overwrite original review inputs'
    run = ROOT / 'runs' / BATCH
    rows, warnings = read_events(run / 'events.jsonl')
    assert not warnings
    version = code_version()
    assert version['id'] == '1ebc06766cf2b99a'
    BASE.mkdir(parents=True, exist_ok=True)
    for name in version['files']:
        target = BASE / 'before' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    for name in ('status.json', 'policy-state.json', 'pending-action.json', 'issue-state.json',
                 'environment.json', 'stderr.log'):
        if (run / name).exists():
            shutil.copy2(run / name, BASE / name)
    write(BASE / 'summary.json', export_run(run))

    def proof(index):
        event = next(e for e in reversed(rows[:index + 1]) if e['event'] == 'observe')
        source = run / Path(event['screenshot']).name
        target = BASE / source.name
        shutil.copy2(source, target)
        return dict(path=target.relative_to(ROOT).as_posix(),
                    sha256=hashlib.sha256(source.read_bytes()).hexdigest())

    data = dict(batch=BATCH, before_version=version,
                events_sha256=hashlib.sha256((run / 'events.jsonl').read_bytes()).hexdigest(),
                shops=[], counters=[], ends=[], issues=[])
    for i, event in enumerate(rows):
        if i in SHOPS:
            assert event['event'] == 'shop_decision'
            data['shops'].append(dict(index=i, name=SHOPS[i], event=event, proof=proof(i)))
        if event['event'] == 'joker_counter_bounded':
            observed = next(e for e in rows[i + 1:i + 5] if e['event'] == 'joker_observed')
            state = next(e['state'] for e in reversed(rows[:i]) if e['event'] == 'observed_state')
            hover = next(e for e in reversed(rows[:i]) if e['event'] == 'hover' and 'Joker slot' in e.get('reason', ''))
            data['counters'].append(dict(index=i, event=observed, state=state,
                x=hover['x'] * 2064, proof=proof(i)))
        if event['event'] == 'game_end':
            state = next(e['state'] for e in reversed(rows[:i]) if e['event'] == 'observed_state')
            data['ends'].append(dict(index=i, event=event, state=state, proof=proof(i)))
        if event.get('severity') in ('degraded', 'recoverable', 'fatal') or event['event'] == 'fatal':
            data['issues'].append(dict(index=i, event=event, proof=proof(i)))
    # Keep the actual pre-purchase Madness hover too, even though it bounded
    # silently in the shop path rather than emitting joker_counter_bounded.
    index = 30147
    search_start = next(j for j in range(index - 1, -1, -1)
                        if rows[j]['event'] == 'action_confirmed' and rows[j].get('kind') == 'reroll')
    hover_index = next(j for j in range(search_start + 1, index)
                       if rows[j]['event'] == 'observe' and rows[j].get('label') == 'shop-offer-tooltip')
    data['madness_offer'] = dict(index=hover_index, proof=proof(hover_index))
    write(FIXTURE, data)
    print('retained', len(data['shops']), len(data['counters']), len(data['ends']), flush=True)


def replay(label):
    data = json.loads(FIXTURE.read_text('utf8'))
    version = code_version()
    if label == 'before':
        assert version == data['before_version']
    result = dict(version=version, live_run=False, shops=[], counters=[])
    for case in data['shops']:
        e = case['event']
        decision = decide_shop(items_from_event(e), state_from_dict(e['state']),
                               reroll_cost=e.get('reroll_cost'))
        result['shops'].append(dict(name=case['name'], index=case['index'], decision=asdict(decision)))
        print(case['name'], decision.action, decision.item_index, decision.sell_index, flush=True)
    for case in data['counters']:
        raw = case['event']['joker']
        joker = Joker(**{**raw, 'data': tuple(map(tuple, raw['data']))})
        state = state_from_dict(case['state'])
        image = load_frame(ROOT / case['proof']['path'])
        words, _ = joker_tooltip_words(read_text(image, (525, 330, 2020, 850)), case['x'])
        # Independent current-frame proof; previous lower bounds are tested
        # separately and must not substitute for a screenshot success here.
        after = verify_current_multiplier(image, words, joker, state)
        result['counters'].append(dict(index=case['index'], name=joker.name, joker=asdict(after)))
    assert code_version() == version, 'Runtime sources changed during replay'
    result['complete'] = True
    write(ROOT / f'analysis/growth_replay_20261020_{label}.json', result)
    print('complete', version['id'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('retain', 'before', 'after'))
    action = parser.parse_args().action
    retain() if action == 'retain' else replay(action)
