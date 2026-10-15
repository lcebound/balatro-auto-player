"""Classify operational errors and defer code review until a game ends.

This module records issues only. It never clicks, changes game files, or
edits/reloads policy code while a game is running.
"""
from collections import Counter
from datetime import datetime
import json

from recovery import atomic_json


LEVELS = frozenset(('degraded', 'recoverable', 'waiting', 'fatal'))


def _queue(session):
    rows = getattr(session, 'runtime_issues', None)
    if rows is None:
        path = session.run_dir / 'issue-state.json'
        rows = json.loads(path.read_text('utf-8')) if path.exists() else []
        session.runtime_issues = rows
    return rows


def record_issue(session, event, *, level, code, message, **context):
    """Keep repeated issues compact, with durable evidence in events.jsonl."""
    if level not in LEVELS:
        raise ValueError(f'Unsupported issue level {level}')
    number = getattr(session, 'current_game_number', 1)
    now = datetime.now().astimezone().isoformat()
    rows = _queue(session)
    row = next((r for r in rows if (r['game_number'], r['level'], r['code'], r['message']) ==
                (number, level, code, message)), None)
    if row is None:
        row = dict(game_number=number, level=level, code=code, message=message,
                   count=0, first_seen=now)
        rows.append(row)
    row.update(count=row['count'] + 1, last_seen=now, last_context=context)
    atomic_json(session.run_dir / 'issue-state.json', rows)
    session.log(event, severity=level, issue_code=code, message=message,
                deferred_fix=level in ('degraded', 'recoverable'), **context)


def issue_counts(session, number=None):
    rows = _queue(session)
    return dict(Counter({level: sum(r['count'] for r in rows if r['level'] == level
                       and (number is None or r['game_number'] == number)) for level in LEVELS}))


def finish_issue_review(session, number, result):
    """Publish the game-specific review after a verified Win/Loss only."""
    rows = [r for r in _queue(session) if r['game_number'] == number]
    directory = session.run_dir / 'issue-reviews'
    directory.mkdir(exist_ok=True)
    path = directory / f'game-{number:04d}.json'
    counts = issue_counts(session, number)
    atomic_json(path, dict(game_number=number, result=result, counts=counts, issues=rows,
                          review_after_game=True))
    session.log('game_issue_review', number=number, counts=counts,
                report=str(path), review_after_game=True)
    return path


def recovery_code(error, pending=None):
    if pending:
        return 'action_receipt_unreadable'
    text = str(error).lower()
    if 'selected cards' in text or 'selection' in text or 'preview' in text:
        return 'card_selection_unverified'
    if 'joker' in text:
        return 'joker_observation_unreadable'
    if 'hand' in text or 'card' in text or 'level' in text:
        return 'hand_observation_unreadable'
    return 'scene_observation_unreadable'


def unresolved_recovery(events):
    """A recovery notification is not a terminal failure."""
    pending = None
    for row in events:
        if row.get('event') == 'automatic_recovery':
            pending = row
        elif row.get('event') == 'recovery_resolved':
            pending = None
    return pending


def terminal_failures(events):
    return [row for row in events if row.get('event') in ('batch_error', 'fatal')
            or row.get('severity') == 'fatal']
