"""Compare retained decisions before/after this log review, offline only."""
import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from planning import shop_forecast, forecast_hold
from recovery import state_from_dict
from run_history import code_version
from strategy import Card, Joker, ShopItem, decide_shop, decide_play, decide_discard, _discard_candidates


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2,
        default=lambda v: sorted(v) if isinstance(v, frozenset) else v), encoding='utf8')


def retain():
    fixture = ROOT / 'tests/fixtures/decision_tree_20261013.json'
    if fixture.exists():
        return json.loads(fixture.read_text('utf8'))
    batch = 'autonomous-20261013-015847'
    source = ROOT / 'runs' / batch / 'events.jsonl'
    records = [json.loads(l) for l in source.read_text('utf8').splitlines()]
    ids = {14464: 'late_growth_pack', 14989: 'serpent_magician',
           19381: 'wall_arcana', 19806: 'wall_celestial', 8884: 'needle_shop',
           9081: 'needle_last_discard'}
    rows = []
    game, state, frame = 0, None, None
    evidence = ROOT / 'analysis/retained_run_evidence' / batch / 'decision-tree-20261013'
    evidence.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(records):
        event = row['event']
        if event == 'game_start':
            game = row['number']
        if event == 'observed_state':
            state = row['state']
        if event == 'observe':
            frame = ROOT / 'runs' / batch / Path(row['screenshot'].replace('\\','/')).name
        if index not in ids:
            continue
        retained = dict(id=ids[index], event_index=index, game=game, event=row)
        if event == 'hand_decision':
            retained['state'] = dict(state, jokers=row['jokers'])
        if frame is not None and frame.exists():
            target = evidence / frame.name
            shutil.copyfile(frame, target)
            retained['frame'] = str(target.relative_to(ROOT)).replace('\\','/')
            retained['frame_sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
        rows.append(retained)
    summary = json.loads((ROOT / 'results' / batch / 'summary.json').read_text('utf8'))
    data = dict(batch=batch, actual_version=records[0]['code_version']['id'],
        events_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        completed_losses=[{k:g.get(k) for k in ('game_number','result','ante_reached',
            'last_boss','last_blind_score','last_blind_target','best_hand_score_observed',
            'automatic_recoveries')} for g in summary['games'] if g['completed']],
        exit_events=[r for r in records if r['event'] in ('automatic_recovery','fatal')], cases=rows)
    write(fixture, data)
    return data


def evaluate(case):
    row = case['event']
    state = replace(state_from_dict(row['state'] if row['event']=='shop_decision' else case['state']),
                    objective='survival')
    if row['event'] == 'shop_decision':
        items=[]
        for entry in row['items']:
            args = dict(entry)
            if args.get('joker'):
                args['joker'] = Joker(**dict(args['joker'], data=tuple(map(tuple,args['joker'].get('data',())))))
            args['pack_contents'] = tuple(args.get('pack_contents', ()))
            items.append(ShopItem(**args))
        f = shop_forecast(state)
        d = decide_shop(items, state, reroll_cost=row['reroll_cost'])
        return dict(id=case['id'], decision=asdict(d), coverage=f.coverage,
            next_scores=f.next_round.scores, next_sample_clear_rate=f.next_round.clear_rate,
            boss_scores=f.boss_round.scores if f.boss_round else None,
            weaknesses=f.profile.weaknesses)
    hand = [Card(**c) for c in row['card_details']]
    play = decide_play(hand,state,lookahead=False)
    d = decide_discard(hand,state,current=play)
    return dict(id=case['id'],play=asdict(play),discard=asdict(d),
        candidate_count=len(_discard_candidates(hand,state)),
        forecast_hold=forecast_hold(hand,state,'full_house'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline',action='store_true')
    parser.add_argument('--ablation-only',action='store_true')
    args=parser.parse_args()
    data=retain()
    if args.ablation_only:
        from unittest.mock import patch
        from strategy import suit_coverage, classify
        import planning
        case=next(c for c in data['cases'] if c['id']=='needle_shop')
        state=replace(state_from_dict(case['event']['state']),objective='survival')
        potential=planning._hold_potential
        def previous_potential(hand,keep,state,preferred):
            held=[hand[i] for i in keep]
            value=potential(hand,keep,state,preferred)
            if state.has('flower_pot'):
                if state.has('splash'):
                    value-=16*suit_coverage(held,state)
                elif 0<len(held)<=5:
                    pattern=classify(held,state)
                    value-=8*suit_coverage([held[i] for i in pattern.scoring],state)
            return value
        planning._FORECAST_CACHE.clear()
        with patch('strategy.flower_pot_cores',return_value=()), \
             patch('planning._hold_potential',side_effect=previous_potential):
            f=shop_forecast(state)
        output=ROOT / 'analysis/decision_tree_needle_sampling_20261013.json'
        prior=json.loads(output.read_text('utf8')) if output.exists() else None
        if prior and 'earlier_sampling_projection' in prior:
            prior=prior['earlier_sampling_projection']
        row=dict(version=code_version()['id'],
            scope='Controlled previous-hold ablation: new 12 samples and single-hand minimum; not old executable or measured score',
            earlier_sampling_projection=prior,
            current_fixed_previous_holds=dict(next_scores=f.next_round.scores,
                safe=f.next_round.safe_score,sustainable=f.sustainable_score,
                coverage=f.coverage,clear_rate=f.next_round.clear_rate))
        write(output,row)
        print(json.dumps(row['current_fixed_previous_holds'],ensure_ascii=False))
        return
    baseline_path=ROOT / 'analysis/decision_tree_baseline_20261013.json'
    if args.baseline and baseline_path.exists():
        raise SystemExit('Original baseline already retained; refusing to overwrite it with another version.')
    rows=[]
    for case in data['cases']:
        rows.append(evaluate(case))
        print(json.dumps(rows[-1],ensure_ascii=False),flush=True)
    name='decision_tree_baseline_20261013.json' if args.baseline else 'decision_tree_replay_20261013.json'
    write(ROOT / 'analysis' / name,dict(version=code_version()['id'],
        actual_version=data['actual_version'], scope='offline old inputs; no gameplay or measured improvement',cases=rows))


if __name__=='__main__':
    main()
