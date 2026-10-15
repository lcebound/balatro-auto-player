"""Recheck archived failed pack tooltips without opening or controlling a game."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perception import shop_offer_item
from recovery import ObservationError
from shop_extras import offer_labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'analysis/pack_failures_replay_20261006.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'analysis/pack_failures_fixed_20261006.json')
    args = parser.parse_args()
    rows = json.loads(args.input.read_text('utf-8'))
    results = []
    for row in rows:
        frame = cv2.resize(cv2.imdecode(np.fromfile(row['image'], np.uint8), 1),
                           (2064, 1191), interpolation=cv2.INTER_NEAREST)
        result = dict(row, before_review=row['now'])
        try:
            labels = offer_labels(frame, row['center'])
            item, confidence = shop_offer_item(labels, 0)
            result.update(now=item.name, status='recognized_supported', confidence=confidence, labels=labels)
        except ObservationError as error:
            status = 'recognized_unsupported' if str(error).startswith(
                'Consumable needs an unsupported use workflow:') else 'unreadable'
            result.update(now=str(error), status=status)
        result['outside_reward_region'] = not (550 <= row['center'][0] <= 1670 and row['center'][1] >= 700)
        results.append(result)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(screenshots=len(results), outcomes=dict(Counter(r['status'] for r in results)),
                         outside_reward_region=sum(r['outside_reward_region'] for r in results),
                         new_identities=[r['now'] for r in results if r['status']=='recognized_supported'
                                         and r['now'] != r['before_review']], output=str(args.output)), ensure_ascii=False))


if __name__ == '__main__':
    main()
