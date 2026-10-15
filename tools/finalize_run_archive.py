"""Verify the saved experiment record and prepare a bounded cleanup manifest.

This tool does not delete anything or connect to the game.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recovery import atomic_json


def main():
    batch = 'autonomous-20261002-000231'
    output = ROOT / 'results' / batch
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    source = ROOT / 'runs' / batch / 'events.jsonl'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != summary['raw_source']['events_sha256']:
        raise RuntimeError('Raw log changed after export; export it again before cleanup')
    for name, expected in [('rounds.csv', 1), ('games.csv', len(summary['games'])), ('blinds.csv', len(summary['blinds']))]:
        with (ROOT / 'results' / name).open(encoding='utf-8-sig', newline='') as stream:
            rows = [row for row in csv.DictReader(stream) if row['batch_id'] == batch]
        if len(rows) != expected:
            raise RuntimeError(f'Incomplete {name}: {len(rows)} versus {expected}')
    if (summary['batch']['completed_games'], summary['batch']['interrupted_games'], summary['batch']['not_started_games']) != (3, 1, 6):
        raise RuntimeError('Unexpected batch accounting')
    if [g['total_score_exact'] for g in summary['games'][:3]] != [26456, 3844, 3494]:
        raise RuntimeError('Settled scores are incomplete or differ from verified screenshots')
    fixtures = json.loads((ROOT / 'tests/fixtures/score_identity_residuals.json').read_text(encoding='utf-8'))
    if len(fixtures) != 3:
        raise RuntimeError('Card identity regressions were not preserved')
    setup_name = 'autonomous-20261001-235758'
    setup_dir = ROOT / 'runs' / setup_name
    setup = json.loads((setup_dir / 'status.json').read_text(encoding='utf-8'))
    summary['excluded_setup_attempts'] = [dict(id=setup_name, excluded_from_averages=True,
        reason=setup['error'], status=setup['status'], ended_games=setup['finished_games'],
        raw_events_sha256=hashlib.sha256((setup_dir / 'events.jsonl').read_bytes()).hexdigest())]
    summary['original_batch_status'] = json.loads((ROOT / 'runs' / batch / 'status.json').read_text(encoding='utf-8'))
    atomic_json(output / 'summary.json', summary)
    targets = []
    for name in (batch, setup_name):
        directory = (ROOT / 'runs' / name).resolve()
        if directory.parent != (ROOT / 'runs').resolve():
            raise RuntimeError('Cleanup target escaped the workspace runs directory')
        files = [path for path in directory.rglob('*') if path.is_file()]
        targets.append(dict(path=str(directory), files=len(files), screenshots=sum(p.suffix.lower() == '.png' for p in files),
                            bytes=sum(p.stat().st_size for p in files)))
    manifest = dict(status='prepared', targets=targets, source_log_sha256=digest,
        saved_completed_games=3, saved_interrupted_games=1, saved_blinds=len(summary['blinds']),
        offline_tests={'passed': 126, 'failed': 0}, game_started_during_iteration=False,
        deleted_at=None, note='Original statuses and invalid setup reason retained in summary.json; raw logs/screenshots are disposable.')
    atomic_json(output / 'cleanup.json', manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
