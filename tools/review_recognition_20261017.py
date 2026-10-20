"""Retain and replay the October 17 recognition failures without a game."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BATCH = 'autonomous-20261017-012806'
RETAINED = ROOT / 'analysis/retained_run_evidence' / BATCH / 'recognition-review-20261017'
FIXTURE = ROOT / 'tests/fixtures/recognition_review_20261017.json'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def capture():
    from run_history import code_version, summarize_events
    events_path = ROOT / 'runs' / BATCH / 'events.jsonl'
    events = [json.loads(line) for line in events_path.read_text('utf-8').splitlines()]
    if FIXTURE.exists():
        raise RuntimeError('Retained baseline already exists; use --replay instead')
    RETAINED.mkdir(parents=True, exist_ok=True)
    selected, controls = {}, set()
    last_frame, last_hover = None, None
    issue_events = []
    names = {'0658': 'Flower Pot', '0759': 'To Do List', '1357': 'Spare Trousers',
             '1609': 'Hack', '1618': 'Hack', '1639': 'Hack', '1650': 'Hack'}
    for index, event in enumerate(events):
        kind = event['event']
        if kind == 'observe':
            last_frame = event['screenshot']
        elif kind == 'hover':
            last_hover = event
        elif kind == 'play_preview_mismatch':
            fatal = event['expected'] == 'straight_flush'
            if event['attempt'] == 1 or index > len(events) - 20:
                selected[last_frame] = dict(kind='preview', expected=event['expected'],
                    role='fatal' if fatal else 'transient', event_index=index)
        elif kind == 'play_preview_verified' and event['category'] not in controls:
            controls.add(event['category'])
            selected[last_frame] = dict(kind='preview', expected=event['category'],
                                       role='control', event_index=index)
        elif kind == 'shop_offer_unknown':
            stem = Path(last_frame).name[:4]
            selected[last_frame] = dict(kind='shop', expected=names[stem], role='degraded',
                card_x=round(last_hover['x'] * 2064), price=event['price'],
                logged_labels=event['tooltip'], event_index=index)
        if kind in ('batch_start', 'game_start', 'game_end', 'automatic_recovery',
                    'batch_error', 'fatal', 'shop_offer_unknown', 'play_preview_mismatch'):
            issue_events.append(event)
    frames = []
    for source, case in selected.items():
        source = Path(source)
        target = RETAINED / source.name
        shutil.copy2(source, target)
        frames.append(dict(case, path=target.relative_to(ROOT).as_posix(),
                           source=source.relative_to(ROOT).as_posix(),
                           sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    for name in ('perception.py', 'item_identity.py', 'controller.py', 'strategy.py'):
        target = RETAINED / 'before' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    for name in ('status.json', 'issue-state.json', 'policy-state.json', 'pending-action.json'):
        shutil.copy2(ROOT / 'runs' / BATCH / name, RETAINED / name)
    write_json(RETAINED / 'events.json', issue_events)
    write_json(RETAINED / 'summary.json', summarize_events(events, BATCH))
    write_json(FIXTURE, dict(batch=BATCH, actual_version=events[0]['code_version']['id'],
        before_version=code_version(), original_events_sha256=hashlib.sha256(events_path.read_bytes()).hexdigest(),
        frames=frames))
    print('Retained', len(frames), 'frames', flush=True)


def replay(label):
    from tools.replay_observation_speed import load_frame
    from perception import (read_text, read_hand_preview, shop_tooltip_region,
                            tooltip_rows, shop_offer_item)
    from run_history import code_version
    fixture = json.loads(FIXTURE.read_text('utf-8'))
    version = code_version()
    if label == 'before' and version['id'] != fixture['before_version']['id']:
        raise RuntimeError('The original replay requires the retained pre-fix source version')
    output = []
    for case in fixture['frames']:
        path = ROOT / case['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == case['sha256']
        image = load_frame(path)
        result = dict(case)
        try:
            if case['kind'] == 'preview':
                result['words'] = read_text(image, (150, 545, 535, 655))
                result['actual'] = read_hand_preview(image)
            else:
                region = shop_tooltip_region(image, case['card_x'])
                result['region'] = region
                result['words'] = read_text(image, region)
                result['labels'] = tooltip_rows(result['words'])
                item, confidence = shop_offer_item(result['labels'], case['price'])
                result['actual'], result['confidence'] = item.name, confidence
                result['joker_data'] = list(item.joker.data) if item.joker else None
            result['matches'] = result['actual'] == case['expected']
        except RuntimeError as error:
            result['error'] = str(error)
            result['matches'] = False
        output.append(result)
        print(path.name, case['role'], result.get('actual', result.get('error')), flush=True)
    write_json(ROOT / 'analysis' / ('recognition_replay_20261017_' + label + '.json'),
               dict(code_version=version, frames=output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', action='store_true')
    parser.add_argument('--replay', choices=('before', 'after'))
    args = parser.parse_args()
    if args.capture:
        capture()
    if args.replay:
        replay(args.replay)
