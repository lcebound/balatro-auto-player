"""Persistent experiment tables, independent of Airtest and disposable logs."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean

SCHEMA_VERSION = 1
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
BLIND_NAMES = ('small', 'big', 'boss')


def code_version():
    files = ('strategy.py', 'planning.py', 'consumables.py', 'runner.py', 'bot.py', 'controller.py',
             'perception.py', 'item_identity.py', 'recovery.py', 'vision.py', 'joker_cache.py', 'hand_cache.py', 'plain_hand.py', 'shop_extras.py',
             'hand_levels.py', 'engine_development.py', 'economy.py', 'runtime_issues.py', 'run_history.py',
             'assets/ui/hand_play_control.png', 'assets/ui/hand_discard_control.png',
             'assets/ui/hand_sort_control.png')
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:16]
    return {'id': digest, 'files': hashes}


def read_events(path):
    records, warnings = [], []
    lines = Path(path).read_text(encoding='utf-8-sig').splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise ValueError(f'Invalid event at line {index + 1}: {path}')
            warnings.append('Incomplete last event excluded after process termination')
    return records, warnings


def _duration(start, end):
    if not start or not end:
        return None
    return round(max(0, (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()), 3)


def _end_stat(labels, label):
    if label not in labels:
        return None
    index = labels.index(label) + 1
    if index < len(labels) and re.fullmatch(r'\d+(?:[.,]\d{3})*', labels[index]):
        return int(labels[index].replace('.', '').replace(',', ''))
    return None


def summarize_events(records, batch_id, target_games=10):
    games, current, blind, pending, waiting_play = {}, None, None, None, None
    policy_version = 'legacy_unrecorded'
    versions = []
    first_time = records[0].get('time') if records else None
    last_time = records[-1].get('time') if records else None

    def observe(values, settled=False, source='event'):
        nonlocal waiting_play
        if current is None:
            return
        current['last_state'].update(values)
        if blind is None:
            return
        for key in ('ante', 'target_score', 'boss'):
            if key in values:
                # Cashout after a boss can already display the NEXT Ante.
                if key == 'ante' and blind['ante'] is not None:
                    continue
                blind[key] = values[key]
        score = values.get('round_score')
        if isinstance(score, (int, float)) and score >= 0:
            blind['score_observed'] = max(blind['score_observed'], score)
            if settled and score >= blind['score_observed']:
                blind['score_settled'] = score
                blind['score_source'] = source
            if waiting_play is not None:
                delta = score - waiting_play['before']
                if delta >= 0:
                    current['play_scores'].append(dict(actual=delta, estimated=waiting_play['estimated'],
                                                       category=waiting_play['category'], source=source))
                    waiting_play = None

    for row in records:
        event = row.get('event')
        if event in ('batch_start', 'batch_resume'):
            target_games = row.get('target_games', target_games)
            policy_version = row.get('code_version', {}).get('id', policy_version)
            if policy_version not in versions:
                versions.append(policy_version)
        if event == 'game_start':
            number = row['number']
            current = games.setdefault(number, dict(number=number, started_at=row.get('time'),
                ended_at=None, result='interrupted', deck=row.get('deck'), stake=row.get('stake'),
                last_state={}, blinds=[], plays=0, discards=0, purchases=[], sells=0, rerolls=0,
                consumables_used=0, recoveries=0, unknown_scenes=0, decisions=0,
                categories=Counter(), play_scores=[], error=None, code_versions=[]))
            blind, pending, waiting_play = None, None, None
            continue
        if current is None:
            continue
        if policy_version not in current['code_versions']:
            current['code_versions'].append(policy_version)
        if event == 'blind_decision':
            # Repeated selection attempts before the first hand are one blind.
            if blind is None or blind['has_hand'] or blind['cleared']:
                blind = dict(number=len(current['blinds']) + 1, blind=BLIND_NAMES[row['blind']],
                    started_at=row.get('time'), ended_at=None, ante=None, boss='',
                    target_score=None, score_observed=0, score_settled=None, score_source=None,
                    cleared=False, has_hand=False, plays=0, discards=0)
                current['blinds'].append(blind)
        elif event == 'observed_state':
            if blind is not None:
                blind['has_hand'] = True
            observe(row['state'], source='observed_state')
        elif event == 'round_snapshot':
            observe(row.get('counters', {}), settled=row.get('settled', False),
                    source=row.get('source', 'runtime_snapshot'))
            current['end_screen'] = row.get('end_screen', current.get('end_screen'))
        elif event == 'hand_decision':
            current['decisions'] += 1
        elif event == 'action_prepared':
            pending = row
        elif event == 'action_not_applied':
            pending = None
        elif event == 'action_confirmed':
            kind = row['kind']
            details = pending.get('details', {}) if pending else {}
            if kind == 'play':
                current['plays'] += 1
                if blind:
                    blind['plays'] += 1
                category = details.get('category', 'unknown')
                current['categories'][category] += 1
                before = pending.get('before', {}).get('round_score') if pending else None
                if before is not None:
                    waiting_play = dict(before=before, estimated=details.get('estimated_score'), category=category)
            elif kind == 'discard':
                current['discards'] += 1
                if blind:
                    blind['discards'] += 1
            elif kind == 'buy':
                current['purchases'].append(dict(time=row.get('time'), **details))
            elif kind in ('sell', 'reroll', 'consume'):
                current[{'sell': 'sells', 'reroll': 'rerolls', 'consume': 'consumables_used'}[kind]] += 1
            # Animation counters can still contain the previous score. Only
            # subsequent ready-hand/settled snapshots feed score statistics.
            current['last_state'].update({k: v for k, v in row.get('counters', {}).items() if k != 'round_score'})
            pending = None
        elif event == 'scene':
            if row['name'] == 'cashout' and blind:
                blind['cleared'] = True
                blind['ended_at'] = row.get('time')
            if row['name'] == 'unknown':
                current['unknown_scenes'] += 1
        elif event == 'automatic_recovery':
            current['recoveries'] += 1
        elif event in ('fatal', 'batch_error'):
            current['error'] = row.get('error')
        elif event == 'game_end':
            current['result'] = row['result']
            current['ended_at'] = row.get('time')
            if blind:
                blind['ended_at'] = row.get('time')
                if row['result'] == 'win':
                    blind['cleared'] = True
            current, blind, pending, waiting_play = None, None, None, None

    game_rows, blind_rows = [], []
    for number in range(1, max(target_games, max(games, default=0)) + 1):
        g = games.get(number)
        version_list = g['code_versions'] if g else versions
        version = version_list[0] if len(version_list) == 1 else 'mixed' if version_list else policy_version
        base = dict(schema_version=SCHEMA_VERSION, batch_id=batch_id, code_version=version,
                    code_versions=version_list, game_number=number)
        if not g:
            game_rows.append(dict(base, result='not_started', completed=False))
            continue
        state, last_blind = g['last_state'], g['blinds'][-1] if g['blinds'] else {}
        ante = last_blind.get('ante') or state.get('ante')
        scores = g['play_scores']
        compared = [s for s in scores if isinstance(s['estimated'], (int, float))]
        settled = sum(b['score_settled'] is not None for b in g['blinds'])
        completed = g['result'] in ('win', 'loss')
        score_complete = bool(g['blinds']) and completed and settled == len(g['blinds'])
        end = g['ended_at'] or last_time
        labels = g.get('end_screen') or []
        seed_index = labels.index('Seed') + 1 if 'Seed' in labels else len(labels)
        seed = labels[seed_index] if seed_index < len(labels) and re.fullmatch(r'[A-Z0-9]{8}', labels[seed_index]) else None
        entry = dict(base, result=g['result'], completed=completed, started_at=g['started_at'], ended_at=g['ended_at'],
            last_observed_at=end, duration_seconds=_duration(g['started_at'], end), deck=g['deck'], stake=g['stake'],
            ante_reached=ante, last_blind=last_blind.get('blind'), last_boss=last_blind.get('boss'),
            stage_reached=((ante - 1) * 3 + BLIND_NAMES.index(last_blind['blind']) + 1) if ante and last_blind else None,
            blinds_entered=len(g['blinds']), blinds_cleared=sum(b['cleared'] for b in g['blinds']),
            last_blind_score=last_blind.get('score_settled') if last_blind.get('score_settled') is not None else last_blind.get('score_observed'),
            last_blind_target=last_blind.get('target_score'),
            last_score_quality='settled_snapshot' if last_blind.get('score_settled') is not None else 'observed_lower_bound',
            best_hand_score_observed=max((s['actual'] for s in scores), default=None), observed_scored_plays=len(scores),
            compared_plays=len(compared),
            mean_abs_prediction_error=round(mean(abs(s['estimated'] - s['actual']) for s in compared), 6) if compared else None,
            mean_prediction_bias=round(mean(s['estimated'] - s['actual'] for s in compared), 6) if compared else None,
            materially_overestimated_plays=sum(s['estimated'] - s['actual'] > max(20, s['actual'] * .1) for s in compared),
            score_comparisons=compared,
            best_hand_score_reported=_end_stat(labels, 'Best Hand'), run_seed=seed,
            cards_played_reported=_end_stat(labels, 'Cards Played'), cards_discarded_reported=_end_stat(labels, 'Cards Discarded'),
            round_reached_reported=_end_stat(labels, 'Round'), cards_purchased_reported=_end_stat(labels, 'Cards Purchased'),
            rerolls_reported=_end_stat(labels, 'Times Rerolled'),
            sum_blind_scores_observed=sum(b['score_observed'] for b in g['blinds']),
            total_score_exact=sum(b['score_settled'] for b in g['blinds']) if score_complete else None,
            score_complete=score_complete, settled_blinds=settled, plays=g['plays'], discards=g['discards'],
            purchases=len(g['purchases']), money_spent=sum(p.get('price', 0) for p in g['purchases']),
            sells=g['sells'], rerolls=g['rerolls'], consumables_used=g['consumables_used'],
            final_money=state.get('money'), final_hands_left=state.get('hands_left'), final_discards_left=state.get('discards_left'),
            final_jokers=state.get('jokers', []), final_levels=state.get('levels', {}),
            played_categories=dict(g['categories']), purchase_details=g['purchases'],
            automatic_recoveries=g['recoveries'], unknown_scenes=g['unknown_scenes'],
            hand_decisions=g['decisions'], error=g['error'], end_screen=g.get('end_screen'))
        game_rows.append(entry)
        for b in g['blinds']:
            blind_rows.append(dict(base, blind_number=b['number'], ante=b['ante'], blind=b['blind'], boss=b['boss'],
                cleared=b['cleared'], started_at=b['started_at'], ended_at=b['ended_at'],
                score_observed=b['score_observed'], score_settled=b['score_settled'], target_score=b['target_score'],
                score_source=b['score_source'], plays=b['plays'], discards=b['discards']))

    started = [g for g in game_rows if g['result'] != 'not_started']
    ended = [g for g in game_rows if g['completed']]
    full = len(ended) == target_games and target_games > 0
    def average(rows, key):
        values = [g[key] for g in rows if g.get(key) is not None]
        return round(mean(values), 6) if values else None
    exact_games = [g for g in ended if g.get('score_complete')]
    batch_version = versions[0] if len(versions) == 1 else 'mixed' if versions else policy_version
    batch = dict(schema_version=SCHEMA_VERSION, batch_id=batch_id, code_version=batch_version, code_versions=versions,
        started_at=first_time, last_observed_at=last_time, target_games=target_games, started_games=len(started),
        completed_games=len(ended), interrupted_games=sum(g['result'] == 'interrupted' for g in game_rows),
        not_started_games=sum(g['result'] == 'not_started' for g in game_rows),
        wins=sum(g['result'] == 'win' for g in ended), losses=sum(g['result'] == 'loss' for g in ended),
        is_full_batch=full, completion_rate=len(ended) / target_games if target_games else None,
        win_rate_completed=sum(g['result'] == 'win' for g in ended) / len(ended) if ended else None,
        mean_ante_completed=average(ended, 'ante_reached'), mean_blinds_cleared_completed=average(ended, 'blinds_cleared'),
        mean_stage_completed=average(ended, 'stage_reached'), mean_best_hand_observed_completed=average(ended, 'best_hand_score_observed'),
        mean_duration_seconds_completed=average(ended, 'duration_seconds'),
        mean_total_score_exact_completed=average(exact_games, 'total_score_exact'), exact_score_games=len(exact_games),
        mean_ante_full_batch=average(ended, 'ante_reached') if full else None,
        mean_blinds_cleared_full_batch=average(ended, 'blinds_cleared') if full else None,
        mean_total_score_full_batch=average(ended, 'total_score_exact') if full and len(exact_games) == target_games else None,
        error=next((g['error'] for g in reversed(started) if g.get('error')), None))
    return dict(schema_version=SCHEMA_VERSION, batch=batch, games=game_rows, blinds=blind_rows)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def _csv_rows(path, replacements, batch_id):
    previous = []
    if path.exists():
        with path.open(encoding='utf-8-sig', newline='') as stream:
            previous = [r for r in csv.DictReader(stream) if r['batch_id'] != batch_id]
    rows = sorted(previous + replacements, key=lambda r: (r['batch_id'], int(r.get('game_number', 0)), int(r.get('blind_number', 0))))
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})
    os.replace(temporary, path)


def save_history(summary, destination=RESULTS):
    destination = Path(destination)
    batch_id = summary['batch']['batch_id']
    if Path(batch_id).name != batch_id or batch_id in ('.', '..'):
        raise ValueError('Batch ID must be a single directory name')
    destination.mkdir(parents=True, exist_ok=True)
    _write_json(destination / batch_id / 'summary.json', summary)
    _csv_rows(destination / 'rounds.csv', [summary['batch']], batch_id)
    _csv_rows(destination / 'games.csv', summary['games'], batch_id)
    _csv_rows(destination / 'blinds.csv', summary['blinds'], batch_id)


def export_run(run_dir, destination=RESULTS):
    run_dir = Path(run_dir)
    path = run_dir / 'events.jsonl'
    if not path.exists():
        return None  # FakeSession and pre-start setup failures have no log.
    records, warnings = read_events(path)
    summary = summarize_events(records, run_dir.name)
    summary['warnings'] = warnings
    save_history(summary, destination)
    return summary
