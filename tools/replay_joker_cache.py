"""Audit the new cache against saved frames, without connecting to Balatro."""
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from joker_cache import JokerCache, STABLE_WITHIN_BLIND, round_token, roster_patch, cached_joker
from recovery import state_from_dict


def replay(directory):
    directory = Path(directory)
    cache, frame_path = None, None
    totals = Counter()
    mismatches = []
    for line in (directory / 'events.jsonl').open(encoding='utf8'):
        event = json.loads(line)
        if event['event'] in ('game_start', 'game_end') or (
                event['event'] == 'action_confirmed' and event.get('kind') in ('buy', 'sell', 'consume')):
            cache = None
        if event['event'] == 'observe' and event.get('label') == 'decision-state':
            frame_path = directory / Path(event['screenshot']).name
        if event['event'] != 'observed_state' or frame_path is None:
            continue
        state = state_from_dict(event['state'])
        raw = cv2.imdecode(np.fromfile(frame_path, dtype=np.uint8), 1)
        image = cv2.resize(raw, (2064, 1191), interpolation=cv2.INTER_NEAREST)
        count = len(state.jokers)
        verified = cache.verified_slots(image, state, count, state.total_joker_slots) if cache else ()
        reuse = bool(verified) and any(verified)
        totals['positions'] += 1
        totals['old_tooltip_reads'] += count
        if reuse:
            totals['reusable_positions'] += 1
            for index, actual in enumerate(state.jokers):
                if verified[index] and actual.key in STABLE_WITHIN_BLIND:
                    totals['avoided_tooltip_reads'] += 1
                    saved = cached_joker(cache.jokers[index], state)
                    # A state-derived current counter may have stale display
                    # data; compare all metadata that matters for that effect.
                    if actual.key in ('bull', 'bootstraps', 'blue_joker', 'banner', 'supernova', 'abstract_joker'):
                        actual = replace_value(actual, saved.value)
                    if saved != actual:
                        mismatches.append({'frame': frame_path.name, 'slot': index, 'key': actual.key})
        reused = {i for i, j in enumerate(state.jokers)
                  if reuse and verified[i] and j.key in STABLE_WITHIN_BLIND}
        # A newly observed different identity makes the controller rebuild all.
        if reuse and any(j.key != old.key for j, old in zip(state.jokers, cache.jokers)):
            totals['avoided_tooltip_reads'] -= len(reused)
            reused = set()
        patches = tuple(cache.patches[i] if i in reused else roster_patch(image, count, i)
                        for i in range(count))
        jokers = tuple(cached_joker(cache.jokers[i], state) if i in reused else j
                       for i, j in enumerate(state.jokers))
        cache = JokerCache(round_token(state), state.total_joker_slots, jokers, patches)
        frame_path = None
    totals['new_tooltip_reads'] = totals['old_tooltip_reads'] - totals['avoided_tooltip_reads']
    return {'batch': directory.name, 'counts': dict(totals), 'metadata_mismatches': mismatches,
            'scope': 'saved decision frames only; estimates read avoidance, not live elapsed time'}


def replace_value(joker, value):
    from dataclasses import replace
    return replace(joker, value=value)


if __name__ == '__main__':
    result = replay(sys.argv[1])
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(result, ensure_ascii=True))
