"""Measure retained-card/Joker cache savings on saved frames, without a game.

The ordinary-face shortcut is validated separately by the plain-hand tests
and analysis/plain_hand_replay_20261010.json.
"""
import argparse
import json
import sys
import time
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np

from hand_cache import HandCache, card_patch
from joker_cache import JokerCache, roster_patch, round_token
from recovery import state_from_dict
from run_history import code_version
from strategy import Card, Joker, RunState
from vision import read_hand


def load_frame(path):
    image = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8), 1)
    return cv2.resize(image, (2064, 1191), interpolation=cv2.INTER_NEAREST)


def records(events):
    hands, rosters = [], []
    hand, roster = None, []
    screenshot, roster_frame, game = None, None, 0
    state = RunState()
    in_roster = False
    current_jokers = []
    epoch, roster_can_update = 0, False
    for event in events:
        kind = event['event']
        if kind == 'game_start':
            game += 1
            roster, state = [], RunState()
        elif kind == 'scene' and event.get('name') == 'shop':
            epoch += 1
        elif kind == 'observe':
            screenshot = event['screenshot']
            if event.get('label') == 'hand-before-inspection':
                hand = dict(frame=screenshot, game=game)
                hands.append(hand)
        elif kind == 'hover' and event.get('reason', '').startswith('read Joker slot'):
            if not in_roster:
                roster_frame = screenshot
                in_roster = True
        elif kind == 'joker_observed':
            # Refreshed slots are committed together at joker_collection.
            pass
        elif kind == 'joker_collection':
            fresh = event.get('refreshed_slots', [])
            if not fresh:
                roster_frame = screenshot
            # Filled below from the compact streaming list of fresh values.
            for index, joker in zip(fresh, current_jokers):
                while len(roster) <= index:
                    roster.append(None)
                roster[index] = joker
            count = len(fresh) + len(event.get('reused_slots', []))
            roster = roster[:count]
            if roster_frame and count and all(roster):
                rosters.append(dict(frame=roster_frame, jokers=tuple(roster),
                                    state=state, game=game, epoch=epoch, tooltip_reads=len(fresh)))
                roster_can_update = True
            current_jokers = []
            in_roster = False
        elif kind == 'observed_state':
            state = state_from_dict(event['state'])
            if rosters and roster_can_update:
                rosters[-1]['state'] = state
            if hand:
                hand['state'] = state
                hand['count'] = event['hand_count']
        elif kind == 'hand_decision' and hand:
            hand['cards'] = tuple(Card(**c) for c in event['card_details'])
            hand['play'] = event['best_indices']
        elif kind == 'discard_decision' and hand:
            hand['discard'] = event['indices']
        elif kind == 'action_prepared' and hand and not hand.get('action'):
            action = event['kind']
            if action in ('play', 'discard', 'consume'):
                hand['action'] = action
                hand['indices'] = hand.get(action, [])
        if kind == 'action_prepared':
            roster_can_update = False
        if kind == 'joker_observed':
            value = dict(event['joker'])
            value['data'] = tuple(tuple(v) for v in value.get('data', []))
            current_jokers.append(Joker(**value))
    return hands, rosters


def sampled(items, count):
    if len(items) <= count:
        return items
    return [items[i] for i in np.linspace(0, len(items) - 1, count, dtype=int)]


def replay(batch, count):
    events = [json.loads(line) for line in (batch / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
    hands, rosters = records(events)
    pairs = [(a, b) for a, b in zip(hands, hands[1:])
             if a.get('action') in ('play', 'discard') and a.get('cards') and b.get('cards')
             and a.get('state') and b.get('state') and a['game'] == b['game']
             and round_token(a['state']) == round_token(b['state'])]
    hand_results = []
    for before, after in sampled(pairs, count):
        a, b = load_frame(before['frame']), load_frame(after['frame'])
        with patch('perception.hand_counter', return_value=(len(before['cards']), before['state'].hand_size)):
            located = read_hand(a, len(before['cards']), identities=before['cards'])
        cache = HandCache(round_token(before['state']), before['cards'],
                          tuple(card_patch(a, entry[2]) for entry in located))
        cache = cache.retain(c for i, c in enumerate(before['cards']) if i not in before['indices'])
        started = time.perf_counter()
        hints = cache.verified_identities(b, after['state'], len(after['cards']), after['state'].hand_size)
        hand_results.append(dict(before=Path(before['frame']).name, after=Path(after['frame']).name,
            full_reads=len(after['cards']), old_reads=sum(not c.face_down for c in after['cards']),
            new_reads=sum(c is None and not after['cards'][i].face_down for i, c in enumerate(hints)),
            retained=len(cache.cards), reused_indices=[i for i, c in enumerate(hints) if c is not None],
            mismatches=[i for i, c in enumerate(hints) if c is not None and c != after['cards'][i]],
            matching_seconds=round(time.perf_counter() - started, 4)))
    roster_pairs = [(a, b) for a, b in zip(rosters, rosters[1:])
                    if a['game'] == b['game'] and a['jokers'] and b['jokers']]
    joker_results = []
    for before, after in sampled(roster_pairs, count):
        a, b = load_frame(before['frame']), load_frame(after['frame'])
        cache = JokerCache(round_token(before['state']), 5, before['jokers'],
            tuple(roster_patch(a, len(before['jokers']), i) for i in range(len(before['jokers']))))
        # Match the runtime's cashout invalidation and freshly observed HUD.
        from joker_cache import dirty_jokers
        from perception import read_numbers
        observed_state = (replace(after['state'], **read_numbers(b))
                          if any(j.key in ('bull', 'bootstraps', 'blue_joker') for j in before['jokers'])
                          else after['state'])
        if before['epoch'] != after['epoch']:
            session = SimpleNamespace(joker_cache=cache)
            dirty_jokers(session, round_end=True)
            cache = session.joker_cache
        started = time.perf_counter()
        mapping = cache.reconciled_slots(b, observed_state, len(after['jokers']))
        reused = [i for i, old in enumerate(mapping) if old is not None
                  and not cache.needs_tooltip(old, observed_state)]
        # Numerical cash-derived values are allowed to change. Everything else
        # in a reused static Joker must agree with the observed later tooltip.
        from joker_cache import cached_joker
        joker_results.append(dict(before=Path(before['frame']).name, after=Path(after['frame']).name,
            full_reads=len(after['jokers']), old_reads=after['tooltip_reads'],
            new_reads=len(after['jokers']) - len(reused),
            reused_indices=reused, count_changed=len(before['jokers']) != len(after['jokers']),
            mismatches=[i for i in reused if cached_joker(cache.jokers[mapping[i]], observed_state)
                        != after['jokers'][i]], matching_seconds=round(time.perf_counter() - started, 4)))
    def summary(rows):
        old = sum(r['old_reads'] for r in rows)
        new = sum(r['new_reads'] for r in rows)
        return dict(samples=len(rows), old_reads=old, new_reads=new, avoided=old - new,
                    full_reads=sum(r['full_reads'] for r in rows),
                    reduction=round((old - new) / old, 4) if old else 0,
                    mismatches=sum(len(r['mismatches']) for r in rows),
                    matching_seconds=round(sum(r['matching_seconds'] for r in rows), 3))
    return dict(batch=batch.name, code_version=code_version()['id'],
        scope='Retained-card/Joker cache replay versus recorded tooltip counts; ordinary-face shortcut excluded. Not measured live game duration. full_reads shows the full-roster baseline.',
        hands=summary(hand_results), jokers=summary(joker_results),
        hand_cases=hand_results, joker_cases=joker_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, default=ROOT / 'runs/autonomous-20261010-001133')
    parser.add_argument('--samples', type=int, default=24)
    parser.add_argument('--output', type=Path, default=ROOT / 'analysis/speed_replay_20261010.json')
    args = parser.parse_args()
    result = replay(args.batch, args.samples)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('hands', 'jokers')}, ensure_ascii=True))
