"""Attribute saved action delays and benchmark scene/receipt changes offline."""
import argparse
import json
import statistics
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_history import code_version, read_events
from recovery import action_outcome
from tools.replay_observation_speed import load_frame, records, sampled


def describe(values):
    if not values:
        return {'n': 0}
    ordered = sorted(values)
    return dict(n=len(values), mean=round(statistics.mean(values), 4),
                median=round(statistics.median(values), 4),
                p90=round(ordered[min(len(ordered) - 1, int(len(ordered) * .9))], 4),
                maximum=round(max(values), 4), total_seconds=round(sum(values), 4))


def review(run):
    events, warnings = read_events(run / 'events.jsonl')
    buckets, cases = {}, []
    stamp = lambda e: datetime.fromisoformat(e['time']).timestamp()
    for i, event in enumerate(events):
        kind = event['event']
        if kind in ('card_collection', 'joker_collection'):
            buckets.setdefault(kind, []).append(event['seconds'])
        if kind != 'tap' or event.get('reason') not in ('play verified hand', 'confirm verified discard'):
            continue
        action = 'play' if event['reason'] == 'play verified hand' else 'discard'
        case = dict(time=event['time'], action=action, receipt_seconds=None,
                    inspection_seconds=None, collection_seconds=None)
        for later in events[i + 1:]:
            if later['event'] in ('game_end', 'automatic_recovery', 'action_prepared'):
                break
            # A cashout/shop/new blind is real game progress, not time spent
            # waiting for this action's replacement hand. Never mix them.
            if later['event'] == 'scene' and later.get('name') not in ('unknown', 'hand'):
                break
            if later['event'] == 'action_confirmed' and later.get('kind') == action:
                case['receipt_seconds'] = stamp(later) - stamp(event)
            if later['event'] == 'observe' and later.get('label') == 'hand-before-inspection':
                case['inspection_seconds'] = stamp(later) - stamp(event)
                case['hand_screenshot'] = later['screenshot']
            if later['event'] == 'card_collection':
                case['collection_seconds'] = stamp(later) - stamp(event)
                break
        for field in ('receipt_seconds', 'inspection_seconds', 'collection_seconds'):
            if case[field] is not None:
                buckets.setdefault(action + '_to_' + field, []).append(case[field])
        cases.append(case)
    for a, b in zip(events, events[1:]):
        dt = stamp(b) - stamp(a)
        if b['event'] == 'hand_decision' and a['event'] == 'observed_state':
            buckets.setdefault('play_compute_inferred', []).append(dt)
        if b['event'] == 'discard_decision' and a['event'] == 'hand_decision':
            buckets.setdefault('discard_compute_inferred', []).append(dt)
        if b['event'] == 'shop_decision' and a.get('reason') == 'dismiss shop tooltip':
            buckets.setdefault('shop_compute_inferred', []).append(dt)
    cards = [e for e in events if e['event'] == 'card_collection']
    return dict(batch=run.name, warnings=warnings,
                metrics={key: describe(values) for key, values in buckets.items()},
                card_reads={key: sum(len(e.get(key, [])) for e in cards)
                            for key in ('reused_slots', 'plain_slots', 'refreshed_slots')},
                action_cases=cases), events


def benchmark(events, count):
    from perception import read_numbers, ocr_engine
    from vision import classify_scene, playable_hand_visible
    from hand_cache import HandCache, card_patch
    from joker_cache import round_token
    from vision import read_hand
    ocr_engine()
    hands, _ = records(events)
    hands = [h for h in hands if h.get('cards') and h.get('state')]
    cases, cache_cases, receipts = [], [], []
    for case in sampled(hands, count):
        frame = load_frame(case['frame'])
        # Warm OCR identically. Measurements isolate repeated full-corner
        # matching versus the controls check, not model initialization.
        classify_scene(frame)
        start = time.perf_counter()
        with patch('vision.playable_hand_visible', return_value=False):
            old = classify_scene(frame)
        baseline = time.perf_counter() - start
        start = time.perf_counter()
        new = classify_scene(frame)
        changed = time.perf_counter() - start
        cases.append(dict(screenshot=case['frame'], baseline_scene=old, new_scene=new,
                          fast_controls_proven=playable_hand_visible(frame),
                          baseline_seconds=round(baseline, 4), new_seconds=round(changed, 4)))
        print(json.dumps({'scene_case': Path(case['frame']).name, 'old': old, 'new': new}), flush=True)
    pairs = [(a, b) for a, b in zip(hands, hands[1:])
             if a.get('action') in ('play', 'discard') and a['game'] == b['game']
             and round_token(a['state']) == round_token(b['state'])]
    for before, after in sampled(pairs, count):
        a, b = load_frame(before['frame']), load_frame(after['frame'])
        found = read_hand(a, len(before['cards']), identities=before['cards'])
        cache = HandCache(round_token(before['state']), before['cards'],
                          tuple(card_patch(a, entry[2]) for entry in found))
        cache = cache.retain(c for i, c in enumerate(before['cards']) if i not in before.get('indices', []))
        start = time.perf_counter()
        hints = cache.verified_identities(b, after['state'], len(after['cards']), after['state'].hand_size)
        cache_cases.append(dict(screenshot=after['frame'], seconds=round(time.perf_counter() - start, 4),
                                reused=sum(c is not None for c in hints),
                                mismatches=[i for i, c in enumerate(hints)
                                            if c is not None and c != after['cards'][i]]))
        print(json.dumps({'cache_case': Path(after['frame']).name}), flush=True)
    pending, shot = None, None
    for event in events:
        if event['event'] == 'action_prepared':
            pending = {'kind': event['kind'], 'before': event['before']}
        if event['event'] == 'observe':
            shot = event['screenshot']
        if (event['event'] == 'action_outcome_read' and pending is not None and shot
                and pending['kind'] in ('play', 'discard') and event.get('outcome') is True):
            receipts.append(dict(screenshot=shot, pending=pending))
    receipt_cases = []
    for case in sampled(receipts, count):
        frame = load_frame(case['screenshot'])
        read_numbers(frame)
        start = time.perf_counter()
        with patch('vision.playable_hand_visible', return_value=False):
            scene = classify_scene(frame)
        baseline_result = action_outcome(case['pending'], scene, read_numbers(frame))
        baseline = time.perf_counter() - start
        start = time.perf_counter()
        values = read_numbers(frame)
        new = action_outcome(case['pending'], 'unknown', values)
        if new is not True:
            new = action_outcome(case['pending'], classify_scene(frame), values)
        new_seconds = time.perf_counter() - start
        receipt_cases.append(dict(**case, baseline_outcome=baseline_result, new_outcome=new,
                                  baseline_seconds=round(baseline, 4), new_seconds=round(new_seconds, 4)))
        print(json.dumps({'receipt_case': Path(case['screenshot']).name,
                          'old': baseline_result, 'new': new}), flush=True)
    return dict(scope='Offline saved-frame replay with warmed OCR; not live game speed.',
                code_version=code_version()['id'], scenes=cases, receipts=receipt_cases,
                retained_card_matching=cache_cases,
                scene_baseline=describe([c['baseline_seconds'] for c in cases]),
                scene_new=describe([c['new_seconds'] for c in cases]),
                receipt_baseline=describe([c['baseline_seconds'] for c in receipt_cases]),
                receipt_new=describe([c['new_seconds'] for c in receipt_cases]),
                retained_matching=describe([c['seconds'] for c in cache_cases]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replay', type=int, default=0)
    args = parser.parse_args()
    report, events = review(args.batch)
    if args.replay:
        report['offline_replay'] = benchmark(events, args.replay)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'batch': report['batch'], 'metrics': report['metrics'],
                      'offline': {k: v for k, v in report.get('offline_replay', {}).items()
                                  if k not in ('scenes', 'receipts', 'retained_card_matching')}}, ensure_ascii=True))


if __name__ == '__main__':
    main()
