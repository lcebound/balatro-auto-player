"""Replay recorded decisions without connecting to or operating Balatro."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import time
from dataclasses import asdict, replace

import cv2
import numpy as np

from perception import read_future_boss, parse_joker_tooltip
from recovery import state_from_dict
from run_history import code_version
from strategy import Card, Joker, ShopItem, decide_shop, decide_play, _purchase_state
from planning import shop_forecast

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / 'runs/autonomous-20261004-020553'


def items_from_dict(items):
    parsed = []
    for item in items:
        item = dict(item)
        if item.get('joker'):
            item['joker'] = Joker(**dict(item['joker'], data=tuple(
                tuple(pair) for pair in item['joker'].get('data', ()))))
        item['pack_contents'] = tuple(item.get('pack_contents', ()))
        parsed.append(ShopItem(**item))
    return parsed


def main():
    cases = json.loads((ROOT / 'analysis/strategy_cases_20261004.json').read_text('utf8'))
    rows = [json.loads(line) for line in (BATCH / 'events.jsonl').read_text('utf8').splitlines()]
    context, latest, game, joker_tooltips = {}, {}, 0, {}
    screenshot = None
    for e in rows:
        if e['event'] == 'game_start':
            game, latest, joker_tooltips = e['number'], {}, {}
        if e['event'] == 'observe':
            screenshot = e['screenshot'].split('\\')[-1]
        if e['event'] == 'blind_decision':
            latest = dict(blind=e['blind'], board=screenshot)
        if e['event'] == 'joker_observed':
            joker_tooltips[e['joker']['name']] = e['tooltip']
        if e['event'] == 'shop_decision':
            context[e['time']] = dict(latest, joker_tooltips=dict(joker_tooltips))
    selected = [('1', 0), ('3', 8), ('3', 9), ('4', 0), ('5', 2),
                ('5', 15), ('5', 21), ('5', 22), ('6', 16), ('6', -1), ('7', -1)]
    boss_cache, shops = {}, []
    for g, index in selected:
        c = cases[g]['shops'][index]
        info = context[c['time']]
        board = info.get('board')
        if board and board not in boss_cache:
            image = cv2.imdecode(np.frombuffer((BATCH / board).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            image = cv2.resize(image, (2064, 1191), interpolation=cv2.INTER_NEAREST)
            boss_cache[board] = read_future_boss(image)
        state = state_from_dict(c['state'])
        # Legacy checkpoints omitted Seltzer's lifetime although the tooltip
        # screenshot/log recorded it. Replay the information now observed by
        # the new parser rather than pretending that every held Seltzer is new.
        state = replace(state, jokers=tuple(parse_joker_tooltip(j, info['joker_tooltips'][j.name])
            if j.key == 'seltzer' and j.name in info['joker_tooltips'] else j for j in state.jokers))
        state = replace(state, blind_index=info.get('blind', -1))
        boss = boss_cache.get(board)
        if boss and state.blind_index != 2:
            state = replace(state, upcoming_boss=boss[0], upcoming_boss_target=boss[1],
                            upcoming_boss_ante=state.ante)
        elif state.blind_index == 2:
            # The most recent Boss belongs to the ante already completed.
            state = replace(state, upcoming_boss_ante=max(1, state.ante - 1))
        items = items_from_dict(c['items'])
        start = time.monotonic()
        new = decide_shop(items, state, reroll_cost=5)
        before = shop_forecast(state)
        after = (shop_forecast(_purchase_state(items[new.item_index], state, new.sell_index))
                 if new.action in ('buy', 'sell_then_buy') else None)
        result = dict(game=int(g), time=c['time'], items=[i.name for i in items],
                      money=state.money, board=board, observed_boss=boss,
                      old=c['decision'], new=asdict(new), before=asdict(before),
                      after=asdict(after) if after else None, seconds=time.monotonic() - start)
        shops.append(result)
        print(json.dumps({k: result[k] for k in ('game', 'items', 'old', 'new', 'seconds')},
                         ensure_ascii=False), flush=True)
    hands = []
    fixture = json.loads((ROOT / 'tests/fixtures/strategy_depth_20261004.json').read_text('utf8'))
    for g, hand_cases in fixture['hands'].items():
        for c in hand_cases:
            hand = [Card(**card) for card in c['event']['card_details']]
            new = decide_play(hand, state_from_dict(c['state']))
            hands.append(dict(game=int(g), time=c['event']['time'],
                              old_indices=c['event']['best_indices'], old_score=c['event']['expected_score'],
                              new=asdict(new)))
    output = dict(source_batch=BATCH.name, kind='offline_policy_replay', code_version=code_version(),
                  boss_screenshots=boss_cache, shops=shops, hands=hands,
                  limitation='Predictions on recorded states; no new games were played.')
    path = ROOT / 'analysis/strategy_depth_replay_20261004.json'
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), 'utf8')
    print(str(path), flush=True)


if __name__ == '__main__':
    main()
