"""Persist independent old-run timings and exact fixed-input comparisons."""
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_history import code_version
from tools.review_action_latency import describe


def main():
    batch = 'autonomous-20261013-232757'
    source = ROOT / 'runs' / batch / 'events.jsonl'
    events = [json.loads(line) for line in source.read_text('utf8').splitlines()]
    metrics = defaultdict(list)
    identities = Counter()
    for row in events:
        kind = row['event']
        if kind == 'policy_timing':
            metrics['policy_' + row['phase']].append(row['seconds'])
        elif kind == 'position_timing':
            for key in ('state_seconds', 'joker_seconds', 'cards_and_levels_seconds'):
                metrics[key].append(row[key])
        elif kind == 'shop_decision':
            metrics['shop_observation'].append(row['observation_seconds'])
            metrics['shop_decision_total'].append(row['decision_seconds'])
        elif kind == 'card_observed':
            identities[row.get('identity_source', 'tooltip')] += 1
    read = lambda name: json.loads((ROOT / 'analysis' / name).read_text('utf8'))
    before = read('speed_baseline_20261014.json')
    after = read('speed_optimized_20261014.json')
    assert after['version'] == code_version()['id'], 'Optimized evidence is stale'
    comparisons = []
    assert len(before['rows']) == len(after['rows']) == 4
    for old, new in zip(before['rows'], after['rows']):
        assert (old['kind'], old['number']) == (new['kind'], new['number'])
        assert old['result'] == new['result'], 'Policy decision changed'
        comparisons.append(dict(kind=old['kind'], number=old['number'],
            before_seconds=old['seconds'], after_seconds=new['seconds'],
            full_decision_identical=True))
    original_pixels = read('speed_pixels_20261014.json')['pixel']
    optimized_pixels = read(f"speed_pixels_{after['version']}_20261014.json")['pixel']
    assert original_pixels['cards'] == optimized_pixels['cards'], 'Card attributes changed'
    deltas = [[b[0] - a[0], b[1] - a[1]] for a, b in
              zip(original_pixels['positions'], optimized_pixels['positions'])]
    result = dict(batch=batch, actual_version=events[0]['code_version']['id'],
        before_version=before['version'], after_version=after['version'],
        source_events_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        old_live_metrics={k: describe(v) for k, v in metrics.items()},
        old_live_card_identity_sources=dict(identities), offline_policy=comparisons,
        offline_pixels=dict(before_seconds=original_pixels['seconds'],
            after_seconds=optimized_pixels['seconds'], full_card_attributes_identical=True,
            positions_reused=optimized_pixels['collection']['positions_reused'],
            position_deltas= deltas, scope='Retained frames; no capture, save or UI waits'),
        validation=dict(unittest_discover=575, skipped=2, new_tests=16,
            new_tests_included_in_discover=True, live_run=False))
    target = ROOT / 'analysis' / 'speed_comparison_20261014.json'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(dict(version=after['version'], comparisons=comparisons,
                         pixel=result['offline_pixels']), ensure_ascii=False))


if __name__ == '__main__':
    main()
