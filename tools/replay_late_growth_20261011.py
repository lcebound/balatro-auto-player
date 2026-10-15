"""Reproducible known-production estimates. Never connects to the game."""
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine_development import development_score
from planning import forecast_round, engine_profile, ShopForecast
from recovery import state_from_dict
from run_history import code_version
from shop_extras import sealed_pack_value, extra_is_worth_considering
from strategy import Card, Joker, RunState, ShopItem, decide_play, SUITS


def serializable(value):
    return json.loads(json.dumps(value, default=lambda v: sorted(v) if isinstance(v, frozenset) else v))


def main():
    base = RunState(ante=3, money=30, hand_size=2, deck_size=2,
                    deck_cards=(Card(14, 'S', seal='blue'), Card(13, 'H')),
                    jokers=(Joker('Constellation', 1), Joker('Mime')),
                    discards_left=0, base_discards_per_round=0)
    cases = [('blue_mime_constellation', base, 1),
             ('blue_full_slots', replace(base, consumables_held=2), 1),
             ('blue_cannot_clear', base, 1_000_000),
             ('blue_observatory', replace(base, observatory=True), 1),
             ('marble_hologram', replace(base, jokers=(Joker('Marble Joker'),
                 Joker('Hologram', 1)), deck_cards=(Card(14, 'S'), Card(13, 'H'))), 1)]
    complete = tuple(Card(r, s, seal='blue' if r in (2, 3) else '')
                     for s in SUITS for r in range(2, 15))
    cases.append(('full_52_pair_engine', replace(base, hand_size=8, deck_size=52,
        deck_cards=complete, jokers=(Joker('Half Joker', 20), Joker('Sly Joker', 50),
        Joker('Constellation', 1), Joker('Mime')), discards_left=3,
        base_discards_per_round=3), 2000))
    inputs = [dict(id=name, state=serializable(asdict(state)), target=target)
              for name, state, target in cases]
    fixture = ROOT / 'tests/fixtures/late_growth_20261011.json'
    fixture.write_text(json.dumps(inputs, ensure_ascii=False, indent=2), encoding='utf-8')
    rows = []
    # Read serialized inputs back so this also verifies a preserved replay.
    for case in json.loads(fixture.read_text('utf-8')):
        state = state_from_dict(case['state'])
        round_result = forecast_round(state, target=case['target'], samples=6)
        forecast = ShopForecast(round_result, None, round_result.safe_score, engine_profile(state))
        # Removing production measures the contribution of these inputs,
        # not an execution of the old code or a historical game score.
        no_training = replace(forecast, next_round=replace(round_result,
            growth_per_round=(), plays_per_round=(), level_gains_per_round=()))
        rows.append(dict(id=case['id'], forecast=asdict(round_result),
                         development_estimate=development_score(state, forecast),
                         no_training_estimate=development_score(state, no_training),
                         current_known_hand_choice=asdict(decide_play(state.deck_cards[:state.hand_size],
                             replace(state, target_score=case['target']), lookahead=False))))
    sample = forecast_round(base, target=1, samples=2)
    safe = ShopForecast(sample, None, sample.safe_score, engine_profile(base))
    pack = ShopItem('standard_pack', 'Standard Pack', 4)
    pack_rows = []
    for value in (1., 5.):
        state = replace(base, money=24, jokers=(Joker('Hologram', value),))
        pack_rows.append(dict(x=value, cash=state.money, price=pack.price,
                             eligible=extra_is_worth_considering(pack, state, safe),
                             opportunity_value=sealed_pack_value(pack, state, safe)))
    historical = json.loads((ROOT / 'analysis/score_policy_inputs_20261011.json').read_text('utf-8'))
    historical_rows = []
    for batch, condition in (
        ('autonomous-20261011-013724', lambda s: any(j['name'] == 'Baseball Card' for j in s['jokers'])),
        ('autonomous-20261011-064759', lambda s: s['ante'] == 7)):
        candidates = [(i, row) for i, row in enumerate(historical)
                      if row['batch'] == batch and condition(row['state'])]
        if candidates:
            index, row = candidates[-1]
            state = state_from_dict(row['state'])
            historical_rows.append(dict(batch=batch, time=row['time'],
                input_index=index, evidence='analysis/score_policy_inputs_20261011.json',
                observed_jokers=[dict(name=j.name, value=j.value, edition=j.edition,
                    value_is_lower_bound=bool(j.get('counter_lower_bound')),
                    data=dict(j.data)) for j in state.jokers],
                observed_levels=state.levels, profile=asdict(engine_profile(state)),
                actual_hand_score=row['actual']))
    report = dict(version=code_version()['id'], starting_version='0b9f0a6e16310c19',
                  scope='offline synthetic production boundaries and retained historical inputs; no gameplay',
                  fixture=str(fixture.relative_to(ROOT)), cases=rows, feeding_budget=pack_rows,
                  historical_inputs=historical_rows,
                  limitations=['Six shuffled samples are estimates, not a measured win rate.',
                               'Blue Planet use is projected after clearing, never during the scored blind.',
                               'No-training values are a controlled counterfactual, not the previous code.',
                               'Later Marble deck dilution and unknown purchases are not simulated by development_score.'])
    output = ROOT / 'analysis/late_growth_replay_20261011.json'
    output.write_text(json.dumps(serializable(report), ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(version=report['version'], cases=[dict(id=r['id'],
        blue=r['forecast']['blue_planets_per_round'], growth=r['forecast']['growth_per_round'],
        development=round(r['development_estimate'], 2),
        no_training=round(r['no_training_estimate'], 2)) for r in rows],
        packs=pack_rows), ensure_ascii=True))


if __name__ == '__main__':
    main()
