"""Replay recorded shop choices using saved screenshots, without Airtest."""
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perception import read_numbers, read_text, tooltip_labels, shop_offer_item
from recovery import state_from_dict
from strategy import decide_shop, shop_strength, joker_name, _purchase_state


def saved_image(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv2.resize(image, (2064, 1191), interpolation=cv2.INTER_NEAREST)


def main():
    directory = ROOT / 'runs/autonomous-20261002-000231'
    if not (directory / 'events.jsonl').exists():
        report = ROOT / 'analysis/score_policy_replay.json'
        if report.exists():
            archived = json.loads(report.read_text(encoding='utf-8'))
            print(json.dumps(dict(mode='archived_comparison', comparisons=len(archived['comparisons']),
                note='Raw screenshots/logs were removed after exporting history; showing the preserved comparison, not rerunning OCR.'), ensure_ascii=False))
            return
        raise FileNotFoundError('Raw batch log was removed and no archived comparison is available')
    records = [json.loads(line) for line in (directory / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
    interesting = {'Ancient Joker', 'Joker Stencil', 'Fibonacci', 'Seltzer', 'Raised Fist', 'Scholar', 'Supernova'}
    state, image_path, hover_x = None, None, None
    offers, latest_jokers, results, mismatches = [], {}, [], []
    game = 0
    for row in records:
        event = row['event']
        if event == 'game_start':
            game = row['number']
            latest_jokers, offers = {}, []
        if event == 'observed_state':
            state = state_from_dict(row['state'])
        if event == 'observe':
            image_path = directory / Path(row['screenshot'].replace('\\', '/')).name
        if event == 'hover':
            hover_x = row['x'] * 2064
        if event == 'joker_observed':
            latest_jokers[joker_name(row['joker']['name'])] = row['joker']
        if event in ('shop_offer_verified', 'shop_offer_non_joker', 'shop_offer_unknown'):
            offers.append(dict(record=row, image=image_path, x=hover_x))
        if event != 'shop_decision':
            continue
        selected = interesting & {item['name'] for item in row['items']}
        if state is None or not selected:
            offers = []
            continue
        frame = saved_image(image_path)
        numbers = read_numbers(frame)
        if 'money' not in numbers:
            mismatches.append(dict(time=row['time'], error='Shop money unreadable; comparison skipped'))
            offers = []
            continue
        updated = []
        for joker in state.jokers:
            value = latest_jokers.get(joker.key, asdict(joker))
            updated.append(type(joker)(**dict(value, data=tuple(tuple(v) for v in value.get('data', ())))))
        observed = replace(state, **numbers, jokers=tuple(updated))
        items, rejected = [], []
        for offer in offers:
            original = offer['record']
            labels = original.get('tooltip')
            if not labels:
                words = read_text(saved_image(offer['image']), (535, 350, 1670, 900))
                labels = tooltip_labels([word for word in words if abs(word[2][0] - offer['x']) > 90])
            labels = [label for label in labels if label.strip() != 'JOKER']
            try:
                item, confidence = shop_offer_item(labels, original['price'])
                items.append(item)
            except RuntimeError as error:
                rejected.append(dict(price=original['price'], tooltip=labels, error=str(error)))
        new = decide_shop(items, observed)
        shopping_list, available, model = [], list(items), observed
        for _ in range(6):
            choice = decide_shop(available, model)
            if choice.action not in ('buy', 'sell_then_buy'):
                break
            bought = available.pop(choice.item_index)
            shopping_list.append(dict(name=bought.name, kind=bought.kind, price=bought.price,
                                      action=choice.action, balance_before=model.money))
            after = _purchase_state(bought, model, choice.sell_index)
            if bought.kind != 'joker':
                # Held cards consume capacity now. Planet levels/Constellation
                # change only when the later hand controller uses the planet.
                after = replace(after, levels=model.levels, jokers=model.jokers,
                                consumables_held=model.consumables_held + 1)
            model = after
        result = dict(game=game, time=row['time'], screenshot=image_path.name,
                      money=observed.money, ante=observed.ante,
                      owned_jokers=[asdict(j) for j in observed.jokers],
                      old_items=row['items'], new_items=[asdict(i) for i in items],
                      old_decision=row['decision'], new_decision=asdict(new),
                      strength=shop_strength(observed), planned_purchases=shopping_list, rejected=rejected)
        results.append(result)
        print(json.dumps({key: result[key] for key in ('game', 'screenshot', 'money', 'old_decision', 'new_decision')}, ensure_ascii=False), flush=True)
        offers = []
    output = ROOT / 'analysis/score_policy_replay.json'
    output.write_text(json.dumps(dict(mode='offline_saved_frames', comparisons=results,
        skipped=mismatches, limitation='Marginal shop comparisons on old states; not a simulated full game or measured score improvement'),
        ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
