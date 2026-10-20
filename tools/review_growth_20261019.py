"""Replay retained growth/identity failures and shop decisions, without a device."""
import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from controller import collect_jokers
from joker_cache import round_token
from perception import read_text, verify_current_multiplier, shop_offer_item, tooltip_rows, shop_tooltip_region
from recovery import state_from_dict, ObservationError
from run_history import code_version
from strategy import decide_shop, Joker
from tools.review_growth_20261015 import items_from_event
from tools.replay_observation_speed import load_frame


class Session:
    joker_cache = None
    hidden_joker_roster = None
    def log(self, *args, **kwargs):
        pass


def replay(label):
    data = json.loads((ROOT / 'tests/fixtures/growth_review_20261019.json').read_text('utf-8'))
    version = code_version()
    if label == 'before':
        assert version == data['before_version']
    result = dict(version=version, actual_version=data['actual_version'],
                  scope='offline predictions and screenshot reads; no game actions', shops=[], hidden=[], counters=[], ramen=[])
    for case in data['shops']:
        e = case['event']
        d = decide_shop(items_from_event(e), state_from_dict(e['state']), reroll_cost=e.get('reroll_cost'))
        result['shops'].append(dict(name=case['name'], index=case['index'], decision=asdict(d)))
        print(case['name'], d.action, d.item_index, d.sell_index, flush=True)
    for case in data['hidden']:
        state = state_from_dict(case['state'])
        prior = state_from_dict(dict(case['state'], jokers=case['prior_roster'])).jokers
        session = Session()
        session.hidden_joker_roster = (round_token(state), state.total_joker_slots, prior)
        image = load_frame(ROOT / case['proof']['path'])
        with patch('controller.joker_counter', return_value=(len(state.jokers), state.total_joker_slots)):
            after = collect_jokers(session, image, state)
        result['hidden'].append(dict(index=case['index'], plays=sum(state.round_played_types.values()),
            discards_used=state.round_discards_used, jokers=[asdict(j) for j in after.jokers]))
    for case in data['counters']:
        raw = case['event']['joker']
        joker = Joker(**{**raw, 'data': tuple(tuple(p) for p in raw['data'])})
        state = state_from_dict(case['state'])
        state = replace(state, total_hands_played=int(joker.get('value_observed_at_hand', state.total_hands_played)))
        previous = next((j for j in state.jokers if j.key == joker.key), None)
        image = load_frame(ROOT / case['proof']['path'])
        after = verify_current_multiplier(image, read_text(image, (525, 330, 2020, 850)), joker, state, previous)
        result['counters'].append(dict(index=case['index'], joker=asdict(after)))
    for case in data['ramen']:
        image = load_frame(ROOT / case['proof']['path'])
        # Selected cards can move the panel. Isolate the visible title/body
        # on this retained frame rather than assuming a shop-slot coordinate.
        words = read_text(image, (525, 330, 2020, 850))
        titles = [w for w in words if w[0].strip().lower() == 'ramen']
        assert len(titles) == 1, (case['index'], words)
        x, y = titles[0][2]
        words = [w for w in words if abs(w[2][0] - x) < 145 and y - 22 <= w[2][1] <= y + 210]
        labels = [t for t in tooltip_rows(words) if t.strip() != 'JOKER']
        try:
            kwargs = dict(image=image, words=words) if label == 'after' else {}
            item, _ = shop_offer_item(labels, case['event'].get('price', 6), **kwargs)
            found = dict(item=asdict(item))
        except ObservationError as error:
            found = dict(error=str(error))
        result['ramen'].append(dict(index=case['index'], **found))
    assert code_version() == version, 'Source changed during fixed-input replay'
    result['complete'] = True
    (ROOT / f'analysis/growth_review_20261019_{label}.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n', 'utf-8')
    print('completed', version['id'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label', choices=('before', 'after'))
    replay(parser.parse_args().label)
