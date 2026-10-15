"""Replay three real score residuals; save compact fixtures before raw cleanup."""
import json
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perception import read_text, tooltip_labels, parse_card_tooltip, read_hand_preview
from recovery import state_from_dict
from strategy import Card, decide_play


def image(path):
    loaded = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv2.resize(loaded, (2064, 1191), interpolation=cv2.INTER_NEAREST)


def main():
    directory = ROOT / 'runs/autonomous-20261002-000231'
    if not (directory / 'events.jsonl').exists():
        cases = json.loads((ROOT / 'tests/fixtures/score_identity_residuals.json').read_text(encoding='utf-8'))
        for case in cases:
            cards = [Card(**card) for card in case['corrected_cards']]
            choice = decide_play(cards, state_from_dict(case['state']))
            print(json.dumps(dict(mode='compact_fixture_replay', action_time=case['action_time'],
                actual_old_score=case['actual_old_score'], decision=asdict(choice)), ensure_ascii=False))
        return
    targets = {'2026-10-02T01:24:36.428919+08:00': 60,
               '2026-10-02T01:26:40.598610+08:00': 296,
               '2026-10-02T02:13:35.568545+08:00': 240}
    cards, state, decision, frame, hover, cases = {}, None, None, None, None, []
    records = [json.loads(line) for line in (directory / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
    for row in records:
        if row['event'] == 'blind_decision':
            cards = {}
        elif row['event'] == 'hover':
            hover = row
        elif row['event'] == 'observe':
            frame = directory / row['screenshot'].replace('\\', '/').split('/')[-1]
        elif row['event'] == 'card_observed':
            cards[row['index']] = dict(record=row, frame=frame, hover=hover)
        elif row['event'] == 'observed_state':
            state = row['state']
        elif row['event'] == 'hand_decision':
            decision = row
        elif row['event'] == 'action_prepared' and row['time'] in targets:
            corrected, tooltips = [], []
            for index in range(len(decision['card_details'])):
                observation = cards[index]
                old = Card(**observation['record']['card'])
                labels = observation['record']['tooltip']
                retry = None
                try:
                    new = parse_card_tooltip(old, labels, require_identity=True)
                except RuntimeError:
                    x = observation['hover']['x'] * 2064 - 30
                    y = observation['hover']['y'] * 1191 - 90
                    crop = tuple(map(round, (max(530, x - 180), max(300, y - 450), min(2020, x + 380), min(875, y - 15))))
                    retry = tooltip_labels(read_text(image(observation['frame']), crop))
                    new = parse_card_tooltip(old, retry, require_identity=True)
                corrected.append(new)
                tooltips.append(dict(index=index, old=asdict(old), corrected=asdict(new), labels=labels, retry_labels=retry))
            choice = decide_play(corrected, state_from_dict(state))
            case = dict(action_time=row['time'], state=state, old_cards=decision['card_details'],
                old_indices=decision['best_indices'], old_category=decision['category'],
                old_estimated_score=decision['expected_score'], actual_old_score=targets[row['time']],
                original_preview_category=read_hand_preview(image(frame)),
                corrected_cards=[asdict(card) for card in corrected],
                corrected_decision=asdict(choice), tooltips=tooltips,
                source_selected_frame=frame.name)
            cases.append(case)
            print(json.dumps({key: case[key] for key in ('action_time', 'old_category', 'old_estimated_score',
                'actual_old_score', 'original_preview_category', 'corrected_decision')}, ensure_ascii=False), flush=True)
    if len(cases) != 3:
        raise RuntimeError('Not all three score residuals were reconstructed')
    output = ROOT / 'tests/fixtures/score_identity_residuals.json'
    output.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
