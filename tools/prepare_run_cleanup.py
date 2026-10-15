"""Preserve bounded diagnostic evidence and prepare the safe cleanup script."""
import argparse
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_history import export_run, read_events
from recovery import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('batches', nargs='+')
    args = parser.parse_args()
    targets, evidence = [], []
    run_root = (ROOT / 'runs').resolve()
    for batch in args.batches:
        assert re.fullmatch(r'(?:(autonomous|batch|test)-[0-9-]+|test-[0-9]{8}-setup)', batch)
        run = (run_root / batch).resolve()
        assert run.parent == run_root and not run.is_symlink()
        source = run / 'events.jsonl'
        events, warnings = read_events(source)
        assert not warnings, warnings
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        kept = ROOT / 'analysis/retained_run_evidence' / batch / 'cleanup-state'
        kept.mkdir(parents=True, exist_ok=True)
        for name in ('status.json', 'policy-state.json', 'pending-action.json', 'issue-state.json',
                     'supervision.json'):
            if (run / name).exists():
                shutil.copy2(run / name, kept / name)
        reviews = run / 'issue-reviews'
        if reviews.exists():
            shutil.copytree(reviews, kept / 'issue-reviews', dirs_exist_ok=True)
        salient, shots = [], []
        last_observation = None
        for event in events:
            if event.get('event') == 'observe':
                last_observation = event
            if re.search(r'issue|recovery|game_start|game_end|batch_|fatal', event.get('event', '')):
                salient.append(event)
                if event.get('event') in ('game_end', 'fatal') and last_observation:
                    shots.append(last_observation.get('screenshot'))
        if last_observation:
            shots.append(last_observation.get('screenshot'))
        retained_shots = []
        for name in dict.fromkeys(shots):
            if not name:
                continue
            image = Path(name).resolve()
            assert image.parent == run
            if image.exists():
                target = kept / image.name
                shutil.copy2(image, target)
                retained_shots.append(dict(path=str(target.relative_to(ROOT)),
                    source=str(image), sha256=hashlib.sha256(image.read_bytes()).hexdigest()))
        atomic_json(kept / 'diagnostic-events.json', dict(raw_sha256=digest,
            events=salient, screenshots=retained_shots))
        summary = export_run(run)
        summary['raw_source'] = dict(events_sha256=digest,
                                    preserved_diagnostics=str(kept.relative_to(ROOT)))
        atomic_json(ROOT / 'results' / batch / 'summary.json', summary)
        files = [p for p in run.iterdir() if p.is_file() and
                 (re.fullmatch(r'\d+-.*\.png', p.name) or
                  p.name in ('events.jsonl', 'stdout.log', 'stderr.log'))]
        targets.append(dict(batch=batch, path=str(run), files=len(files),
            events_sha256=digest, history_preserved=True,
            bytes=sum(p.stat().st_size for p in files)))
        evidence.append(str(kept.relative_to(ROOT)))
    manifest = dict(status='prepared', runs_root=str(run_root), targets=targets,
        prepared_at=datetime.now().astimezone().isoformat(), preserved_evidence=evidence,
        note='Existing analysis, results, fixtures and runtime checkpoints are retained.')
    atomic_json(ROOT / args.manifest, manifest)
    print(json.dumps(dict(files=sum(t['files'] for t in targets),
        bytes=sum(t['bytes'] for t in targets), evidence=evidence), ensure_ascii=True))


if __name__ == '__main__':
    main()
