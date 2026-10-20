"""Retain last-run failure pixels and compare policies without game access."""
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
from run_history import read_events, code_version
from tools.review_growth_20261015 import items_from_event
from strategy import Card, decide_play, decide_discard, decide_shop
from planning import shop_forecast

BATCH = 'autonomous-20261015-232222'
FIXTURE = ROOT / 'tests/fixtures/decision_review_20261016.json'
SELECTED = {117:'starter_pack',168:'early_cash_reserve',1344:'inactive_credit_card',
            2767:'funded_reroll_voucher',3218:'midgame_search',4709:'idle_runner',
            6339:'known_chip_upgrade',6818:'window_last_discard',6990:'window_last_play'}


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', 'utf8')


def prepare():
    if FIXTURE.exists():
        raise RuntimeError('Do not overwrite preserved historical inputs')
    source = ROOT / 'runs' / BATCH / 'events.jsonl'
    rows, warnings = read_events(source)
    assert not warnings
    evidence = ROOT / 'analysis/retained_run_evidence' / BATCH / 'decision-review-20261016'
    evidence.mkdir(parents=True, exist_ok=True)
    proofs = {}

    def retain(event):
        src = ROOT / 'runs' / BATCH / Path(event['screenshot'].replace('\\','/')).name
        dst = evidence / src.name
        if not dst.exists():
            shutil.copy2(src,dst)
        assert src.read_bytes() == dst.read_bytes()
        proof = dict(path=dst.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(dst.read_bytes()).hexdigest())
        proofs[proof['path']] = proof
        return proof

    last = hand_frame = state = None
    game = 0
    cases, failures, runtime, outcomes = [], [], [], []
    for i,row in enumerate(rows):
        kind = row['event']
        if kind == 'game_start':game = row['number']
        if kind == 'observe':
            last = row
            if row['label'] == 'hand-before-inspection':hand_frame=row
        if kind == 'observed_state':state=row['state']
        if i in SELECTED:
            assert kind in ('hand_decision','shop_decision')
            case=dict(name=SELECTED[i],kind='hand' if kind=='hand_decision' else 'shop',
                      game=game,event_index=i,event=row,proof=retain(hand_frame if kind=='hand_decision' else last))
            if kind=='hand_decision':case['state']=dict(state,jokers=row['jokers'])
            cases.append(case)
        if kind=='automatic_recovery':
            failures.append(dict(event_index=i,event=row,proof=retain(last)))
        if kind in ('automatic_recovery','recovery_resolved','batch_error','fatal','game_end'):
            runtime.append(row)
        if kind=='game_end':outcomes.append(dict(event=row,proof=retain(last)))
    write(FIXTURE,dict(batch=BATCH,actual_version=rows[0]['code_version']['id'],
                      source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      cases=cases,failures=failures,outcomes=outcomes))
    write(evidence/'manifest.json',dict(proofs=list(proofs.values())))
    write(evidence/'runtime-events.json',runtime)
    for name in ('issue-state.json','checkpoint.json'):
        src=ROOT/'runs'/BATCH/name
        if src.exists():shutil.copy2(src,evidence/name)
    shutil.copy2(ROOT/'results'/BATCH/'summary.json',evidence/'summary.json')
    for src in (ROOT/'runs'/BATCH/'issue-reviews').glob('*.json'):shutil.copy2(src,evidence/src.name)
    before=evidence/'before-code';before.mkdir(exist_ok=True)
    for name in ('vision.py','strategy.py','planning.py','economy.py'):shutil.copy2(ROOT/name,before/name)
    print(json.dumps(dict(cases=len(cases),failure_frames=len(failures),unique_frames=len(proofs))))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--baseline',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.prepare:prepare();return
    fixture=json.loads(FIXTURE.read_text('utf8'));version=code_version()['id']
    target=ROOT/'analysis'/('decision_baseline_20261016.json' if args.baseline else 'decision_replay_20261016.json')
    output=[]
    if target.exists():
        if not args.resume:raise RuntimeError('Use --resume to preserve existing results')
        prior=json.loads(target.read_text('utf8'));assert prior['version']==version
        output=prior['cases']
    for case in fixture['cases']:
        if case['name'] in {c['name'] for c in output}:continue
        event=case['event'];s=state_from_dict(event['state'] if case['kind']=='shop' else case['state'])
        if case['kind']=='shop':
            f=shop_forecast(s);decision=decide_shop(items_from_event(event),s,reroll_cost=event.get('reroll_cost'))
            result=dict(coverage=f.coverage,growth=f.next_round.growth_per_round,weaknesses=f.profile.weaknesses)
        else:
            hand=[Card(**c) for c in event['card_details']];decision=decide_play(hand,s);result={}
            if case['name']=='window_last_discard':result['discard']=asdict(decide_discard(hand,s,current=decision))
        result.update(name=case['name'],decision=asdict(decision));output.append(result)
        assert code_version()['id']==version,'Code changed during replay'
        write(target,dict(batch=BATCH,actual_version=fixture['actual_version'],version=version,
                         scope='offline fixed inputs, not new observed scores',
                         complete=len(output)==len(fixture['cases']),cases=output))
        print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
