"""Retain and replay economy/growth shop inputs; never touches the game UI."""
import hashlib
import json
import sys
import zipfile
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from economy import economy_value, investment_rounds
from planning import ECONOMY_JOKERS, XMULT_JOKERS, engine_profile, shop_forecast
from recovery import state_from_dict
from run_history import code_version
from strategy import Card, Joker, RunState, ShopItem, decide_shop, decide_play, decide_discard


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
        default=lambda x: sorted(x) if isinstance(x, frozenset) else x), encoding='utf-8')


def main():
    fixture = ROOT / 'tests/fixtures/growth_system_20261013.json'
    batch = 'autonomous-20261013-015847'
    if not fixture.exists():
        events_file = ROOT / 'runs' / batch / 'events.jsonl'
        chosen = []
        groups = set()
        for line in events_file.read_text('utf-8').splitlines():
            row = json.loads(line)
            if row.get('event') != 'shop_decision' or 'state' not in row:
                continue
            names = {Joker(i['name']).key for i in row['items'] if i['kind'] == 'joker'}
            group = ('economy' if names & ECONOMY_JOKERS else 'xmult'
                     if names & XMULT_JOKERS else 'other')
            ante = row['state']['ante']
            bucket = (group, min(ante, 5))
            if bucket in groups or len(chosen) >= 12:
                continue
            groups.add(bucket)
            chosen.append(dict(batch=batch, time=row['time'], state=row['state'],
                               items=row['items'], reroll_cost=row['reroll_cost'],
                               historical_decision=row['decision']))
        write(fixture, dict(source=f'runs/{batch}/events.jsonl',
                           source_sha256=hashlib.sha256(events_file.read_bytes()).hexdigest(),
                           actual_version='931965e09f711eeb', shops=chosen))
    retained = json.loads(fixture.read_text('utf-8'))
    rows = []
    for row in retained['shops']:
        state = replace(state_from_dict(row['state']), objective='survival')
        items = []
        for item in row['items']:
            args = dict(item)
            if args.get('joker'):
                args['joker'] = Joker(**dict(args['joker'], data=tuple(map(tuple, args['joker'].get('data', ())))))
            args['pack_contents'] = tuple(args.get('pack_contents', ()))
            items.append(ShopItem(**args))
        f = shop_forecast(state)
        rows.append(dict(time=row['time'], ante=state.ante, money=state.money,
            profile=asdict(engine_profile(state, f.next_round.frequencies)),
            historical_decision=row['historical_decision'],
            current_offline_decision=asdict(decide_shop(items, state, reroll_cost=row['reroll_cost'])),
            predicted_coverage=f.coverage, economy_policy_value=economy_value(state, f),
            investment_rounds=investment_rounds(state)))
    hand = [Card(r,s) for r,s in ((14,'S'),(14,'H'),(8,'D'),(8,'C'),(2,'H'),(3,'D'),(4,'C'),(7,'S'))]
    training = RunState(jokers=(Joker('Ride the Bus',2), Joker('Joker',4)),
                        target_score=150, objective='survival', ante=1)
    play = decide_play(hand, training)
    example = dict(state=asdict(training), hand=[asdict(c) for c in hand],
                   play=asdict(play), discard=asdict(decide_discard(hand, training, current=play)))
    local_rules = {}
    with zipfile.ZipFile(ROOT / 'Balatro/Balatro.exe') as archive:
        for name in ('card.lua', 'localization/en-us.lua'):
            data = archive.read(name)
            lines = data.decode('utf-8').splitlines()
            needles = ('j_mail=', 'j_rocket=', 'j_cloud_9=', 'j_satellite=',
                       "self.ability.name == 'Mail-In Rebate' and",
                       "self.ability.name == 'Delayed Gratification' and G.GAME.current_round",
                       "self.ability.name == 'Rocket' and G.GAME.blind.boss")
            excerpts = [dict(line=i+1, text='\n'.join(lines[i:i+9]))
                        for i,l in enumerate(lines) if any(n in l for n in needles)]
            local_rules[name] = dict(sha256=hashlib.sha256(data).hexdigest(), excerpts=excerpts)
    report = dict(starting_version='931965e09f711eeb', version=code_version()['id'],
        scope='offline replay of retained historical shops plus a synthetic training example; no gameplay',
        fixture=str(fixture.relative_to(ROOT)).replace('\\','/'),
        historical_shop_replays=rows, synthetic_training=example, installed_rules=local_rules,
        limitations=['Income and growth values are policy estimates, never actual dollars or scores.',
                     'Historical actions were executed by the old version; current actions are offline suggestions.',
                     'Fixed forecast samples are not measured win rates.',
                     'No new Win/Loss, UI actions, log deletion, or external process startup.'])
    output = ROOT / 'analysis/growth_system_replay_20261013.json'
    write(output, report)
    print(json.dumps(dict(version=report['version'], shops=len(rows),
                         changes=sum(r['historical_decision'] != r['current_offline_decision'] for r in rows),
                         training=play.training), ensure_ascii=False))


if __name__ == '__main__':
    main()
