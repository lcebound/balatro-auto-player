"""Fixed-input policy and saved-pixel collection timing, without a game."""
import argparse
import cProfile
import hashlib
import io
import json
import pstats
import re
import shutil
import sys
import time
import tempfile
import statistics
from dataclasses import asdict, replace
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from recovery import state_from_dict
from run_history import code_version
from strategy import Card,Joker,ShopItem,decide_shop,decide_play,_draw_best_plan
from planning import _FORECAST_CACHE


def write(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,
        default=lambda v: sorted(v) if isinstance(v,frozenset) else v),encoding='utf8')


def retain():
    target=ROOT/'tests/fixtures/speed_20261014.json'
    if target.exists():return json.loads(target.read_text('utf8'))
    batch='autonomous-20261013-232757'
    source=ROOT/'runs'/batch/'events.jsonl'
    records=[json.loads(l) for l in source.read_text('utf8').splitlines()]
    shops=[r for r in records if r['event']=='shop_decision']
    shops.sort(key=lambda r:r['decision_seconds'])
    selected={r['time'] for r in (shops[len(shops)//2],shops[-1])}
    states=[];hands=[];state=None;frame=None;tooltip_frames={};collect_input=None
    evidence=ROOT/'analysis/retained_run_evidence'/batch/'speed-20261014'
    evidence.mkdir(parents=True,exist_ok=True)
    def save(path):
        src=ROOT/'runs'/batch/Path(path.replace('\\','/')).name
        dest=evidence/src.name
        if not dest.exists():shutil.copyfile(src,dest)
        return dict(path=str(dest.relative_to(ROOT)).replace('\\','/'),sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
    collections=[]
    for index,row in enumerate(records):
        if row['event']=='observe':
            frame=row['screenshot']
            if row['label']=='hand-before-inspection':
                collect_input=dict(frame=frame,tooltips={},state=state,index=index)
            match=re.fullmatch(r'card-(\d+)-tooltip',row['label'])
            if match and collect_input is not None:
                collect_input['tooltips'].setdefault(match[1],[]).append(frame)
        if row['event']=='card_collection' and collect_input is not None:
            collect_input['logged_seconds']=row['seconds']
        if row['event']=='observed_state':
            state=row['state']
            if collect_input is not None:
                collect_input['state']=state
                collect_input['count']=row['hand_count']
        if row['event']=='hand_decision' and state is not None:
            hands.append(dict(event_index=index,time=row['time'],state=dict(state,jokers=row['jokers']),cards=row['card_details']))
            if collect_input is not None and len(collect_input['tooltips'])>=len(row['card_details']):
                collect_input['cards']=row['card_details']
                collections.append(collect_input)
            collect_input=None
        if row['event']=='shop_decision' and row['time'] in selected:
            states.append(dict(event_index=index,event=row))
    large=max(hands,key=lambda r:len(r['cards']))
    normal=next(r for r in hands if len(r['cards'])==8)
    pixel=next((r for r in collections if r['count']==13),None)
    if pixel is not None:
        pixel=dict(pixel,frame=save(pixel['frame']),tooltips={
            k:[save(path) for path in paths] for k,paths in pixel['tooltips'].items()})
    data=dict(batch=batch,actual_version=records[0]['code_version']['id'],
              events_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
              shops=states,hands=[normal,large],collection=pixel)
    write(target,data)
    return data


def shop_items(entries):
    result=[]
    for entry in entries:
        args=dict(entry)
        if args.get('joker'):
            args['joker']=Joker(**dict(args['joker'],data=tuple(map(tuple,args['joker'].get('data',())))))
        args['pack_contents']=tuple(args.get('pack_contents',()))
        result.append(ShopItem(**args))
    return result


def pixels(case):
    import cv2
    import numpy as np
    from controller import collect_cards
    from perception import _TEXT_CACHE, ocr_engine
    def image(entry):
        frame=cv2.imdecode(np.frombuffer((ROOT/entry['path']).read_bytes(),np.uint8),1)
        return cv2.resize(frame,(2064,1191),interpolation=cv2.INTER_NEAREST)
    main=image(case['frame'])
    tooltips={int(k):[image(e) for e in entries] for k,entries in case['tooltips'].items()}
    class Replay:
        def __init__(self):
            self.policy_state=state_from_dict(case['state']);self.events=[];self.which=0;self.calls={}
        def hover(self,x,y,reason):
            match=re.search(r'read card (\d+) modifiers',reason)
            if match:self.which=int(match[1])
        def observe(self,label):
            count=self.calls.get(self.which,0);self.calls[self.which]=count+1
            return tooltips[self.which][min(count,len(tooltips[self.which])-1)]
        def log(self,event,**data):self.events.append(dict(event=event,**data))
    ocr_engine();_TEXT_CACHE.clear()
    session=Replay();start=time.perf_counter()
    found=collect_cards(session,main,case['count'],state=session.policy_state)
    return dict(seconds=time.perf_counter()-start,cards=[asdict(c) for c,_,_ in found],
                positions=[p for _,_,p in found],collection=session.events[-1])


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=('baseline','optimized','pixels','png'),required=True)
    args=p.parse_args();data=retain()
    output=ROOT/'analysis'/f'speed_{args.phase}_20261014.json'
    if output.exists() and args.phase=='baseline':raise SystemExit('Baseline already preserved.')
    rows=[]
    if args.phase=='png':
        import cv2
        import numpy as np
        from airtest.aircv import imwrite
        from bot import write_observation_png
        cases=[data['collection']['frame'],data['collection']['tooltips']['0'][0]]
        for entry in cases:
            raw=cv2.imdecode(np.frombuffer((ROOT/entry['path']).read_bytes(),np.uint8),1)
            samples={'old':[],'new':[]};sizes={};identical=True
            with tempfile.TemporaryDirectory() as directory:
                for _ in range(3):
                    for key,save in (('old',imwrite),('new',write_observation_png)):
                        path=Path(directory)/f'观测-{key}.png'
                        start=time.perf_counter();save(str(path),raw);samples[key].append(time.perf_counter()-start)
                        buf=path.read_bytes();sizes[key]=len(buf)
                        identical &= np.array_equal(raw,cv2.imdecode(np.frombuffer(buf,np.uint8),1))
            rows.append(dict(frame=entry['path'],median_seconds={k:statistics.median(v) for k,v in samples.items()},
                file_bytes=sizes,pixels_identical=bool(identical)))
        write(output,dict(version=code_version()['id'],scope='Same retained pixels, lossless PNG write only; no live capture',rows=rows))
        print(json.dumps(rows,ensure_ascii=False));return
    if args.phase=='pixels':
        output=ROOT/'analysis'/f"speed_pixels_{code_version()['id']}_20261014.json"
        if data['collection'] is None:raise RuntimeError('No complete retained collection case')
        result=pixels(data['collection']);write(output,dict(version=code_version()['id'],pixel=result))
        print(json.dumps(dict(seconds=result['seconds'],cards=len(result['cards']))));return
    for number,case in enumerate(data['hands']):
        state=state_from_dict(case['state']);hand=[Card(**c) for c in case['cards']]
        start=time.perf_counter();d=decide_play(hand,state)
        rows.append(dict(kind='play',number=number,seconds=time.perf_counter()-start,result=asdict(d)))
        print(json.dumps({k:rows[-1][k] for k in ('kind','number','seconds')}),flush=True)
    profiler=cProfile.Profile();case=data['hands'][-1]
    profiler.runcall(_draw_best_plan,[Card(**c) for c in case['cards']],state_from_dict(case['state']))
    stream=io.StringIO();pstats.Stats(profiler,stream=stream).sort_stats('cumtime').print_stats(25)
    (ROOT/'analysis'/f'speed_profile_{args.phase}_20261014.txt').write_text(stream.getvalue(),encoding='utf8')
    for number,case in enumerate(data['shops']):
        row=case['event'];state=state_from_dict(row['state']);_FORECAST_CACHE.clear()
        start=time.perf_counter();d=decide_shop(shop_items(row['items']),state,reroll_cost=row['reroll_cost'])
        rows.append(dict(kind='shop',number=number,seconds=time.perf_counter()-start,result=asdict(d)))
        print(json.dumps({k:rows[-1][k] for k in ('kind','number','seconds')}),flush=True)
    write(output,dict(version=code_version()['id'],scope='Offline fixed inputs; no live actions or win-rate estimate',rows=rows))


if __name__=='__main__':main()
