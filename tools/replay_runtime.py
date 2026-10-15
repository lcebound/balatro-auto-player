"""Replay old logs/screenshots offline. Does not import or connect Airtest."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from perception import (identify_joker_text, parse_joker_tooltip, joker_from_definition,
                        read_text, tooltip_title, tooltip_labels, parse_levels,
                        read_consumables, read_numbers)


def load_image(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot read saved image {path}')
    return cv2.resize(image, (2064, 1191), interpolation=cv2.INTER_NEAREST)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', default='runs/test-20261001-001157')
    args = parser.parse_args()
    directory = (ROOT / args.run_dir).resolve()
    records = [json.loads(line) for line in (directory / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
    previous = None
    results, rescans = [], []
    for record in records:
        if record['event'] == 'observe':
            previous = directory / Path(record['screenshot'].replace('\\', '/')).name
        if record['event'] != 'joker_observed':
            continue
        row = {'time': record['time'], 'old': record['joker']['name']}
        try:
            try:
                key, confidence = identify_joker_text(record['tooltip'])
            except RuntimeError:
                words = read_text(load_image(previous), (525, 330, 2020, 850))
                key, confidence = identify_joker_text(tooltip_labels(words), title=tooltip_title(words))
                rescans.append(previous.name)
            joker = parse_joker_tooltip(joker_from_definition(key), record['tooltip'])
            row.update(new=joker.name, value=joker.value, confidence=confidence)
        except Exception as error:
            row['error'] = str(error)
        results.append(row)
    corrections = Counter((row['old'], row['new']) for row in results
                          if 'new' in row and row['old'] != row['new'])
    report = {'mode': 'offline_only', 'source_run': str(directory), 'samples': len(results),
              'recognized': sum('new' in row for row in results), 'rescanned_screenshots': rescans,
              'corrected': [dict(old=old, new=new, count=count) for (old, new), count in corrections.items()],
              'failures': [row for row in results if 'error' in row], 'level_frames': [], 'counter_frames': []}
    for name in ('0009-poker-hand-levels.png', '0065-poker-hand-levels.png', '1069-poker-hand-levels.png'):
        path = directory / name
        if path.exists():
            levels, _ = parse_levels(read_text(load_image(path)))
            report['level_frames'].append(dict(screenshot=name, count=len(levels), levels=levels))
    for name in ('0035-decision-state.png', '0171-decision-state.png', '1068-step-54.png'):
        path = directory / name
        if path.exists():
            report['counter_frames'].append(dict(screenshot=name, counters=read_numbers(load_image(path))))
    path = directory / '1068-step-54.png'
    if path.exists():
        report['consumable_matches'] = read_consumables(load_image(path))
    destination = ROOT / 'analysis/runtime_identity_replay.json'
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('samples', 'recognized', 'corrected', 'failures', 'level_frames')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
