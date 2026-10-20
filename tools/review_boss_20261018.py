"""Replay retained Boss-related inputs; never connect to or operate the game."""
import argparse
import hashlib
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_history import code_version
from recovery import state_from_dict
from strategy import decide_shop, decide_play, decide_discard, Card
from tools.review_growth_20261015 import items_from_event
from tools.replay_observation_speed import load_frame
from perception import read_future_boss, read_numbers


def replay(label):
    data = json.loads((ROOT / 'tests/fixtures/boss_review_20261018.json').read_text('utf-8'))
    version = code_version()
    if label == 'before':
        assert version == data['before_version']
    output = dict(version=version, actual_version=data['actual_version'],
                  scope='offline fixed inputs; predictions only; no decisions executed', cases=[], blinds=[])
    for case in data['cases']:
        e = case['event']
        state = state_from_dict(e.get('state', case.get('state')))
        if e['event'] == 'shop_decision':
            decision = decide_shop(items_from_event(e), state, reroll_cost=e.get('reroll_cost'))
            result = dict(index=case['index'], game=case['game'], decision=asdict(decision))
        else:
            cards = [Card(**c) for c in e['card_details']]
            decision = decide_play(cards, state)
            discard = decide_discard(cards, state, current=decision)
            result = dict(index=case['index'], game=case['game'], play=asdict(decision), discard=asdict(discard))
        output['cases'].append(result)
        print(case['index'], str(result)[:300], flush=True)
    for case in data['blinds']:
        image = load_frame(ROOT / case['proof']['path'])
        result = read_future_boss(image)
        output['blinds'].append(dict(index=case['index'], game=case['game'],
            ante=read_numbers(image).get('ante'), result=result))
    if label == 'after':
        output['corrected_observation_cases'] = []
        for case in data['cases']:
            if case['event']['event'] != 'shop_decision':
                continue
            state = state_from_dict(case['event']['state'])
            known = [b for b in output['blinds'] if b['result'] and b['game'] == case['game']
                     and b['index'] < case['index'] and b['ante'] == state.ante]
            if not known or state.upcoming_boss_ante == state.ante:
                continue
            source = max(known, key=lambda b: b['index'])
            name, target = source['result']
            corrected = replace(state, upcoming_boss=name, upcoming_boss_target=target,
                                upcoming_boss_ante=state.ante)
            decision = decide_shop(items_from_event(case['event']), corrected,
                                  reroll_cost=case['event'].get('reroll_cost'))
            output['corrected_observation_cases'].append(dict(index=case['index'],
                recovered_from_blind=source['index'], boss=name, target=target, decision=asdict(decision)))
        from boss_policy import choose_verdant_sale
        state = state_from_dict(data['cases'][-1]['state'])
        sale = choose_verdant_sale(state)
        output['leaf_preparation'] = dict(index=sale, name=state.jokers[sale].name if sale is not None else None,
            scope='projected choice only; runtime must verify the sale receipt before disabling')
    assert code_version() == version, 'Sources changed during replay'
    output['complete'] = True
    (ROOT / f'analysis/boss_replay_20261018_{label}.json').write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + '\n', 'utf-8')
    print('completed', version['id'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label', choices=('before', 'after'))
    replay(parser.parse_args().label)
