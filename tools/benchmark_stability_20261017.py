"""Compare read/UI budgets and offline CPU time; never connect to the game."""
import ast
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import controller
import perception
import recovery
import runner
from run_history import code_version
from tests.test_stability_budget_20261017 import ObservedSession, shop_scenario, FIXTURE
from tools.replay_observation_speed import load_frame

BASE = ROOT / 'analysis/retained_run_evidence/stability-budget-20261017'


def baseline_function(module, name):
    source = (BASE / 'before' / (module.__name__ + '.py')).read_text('utf-8')
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = dict(vars(module))
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<retained-baseline>', 'exec'), namespace)
    return namespace[name]


def clear_ocr():
    perception._TEXT_CACHE.clear()
    perception._TEXT_CACHE_BYTES = 0


def preview_sample(case, before, fault):
    frame = load_frame(ROOT / case['path'])
    with tempfile.TemporaryDirectory() as directory:
        session = ObservedSession(directory)
        session.frame = frame
        clear_ocr()
        calls, waits = [], []
        actual_reader = perception.read_hand_preview
        def read(image, **kwargs):
            calls.append(kwargs.get('contrast', False))
            if fault == 'all' or fault == 'first' and len(calls) == 1:
                raise recovery.ObservationError('Injected unreadable panel; preserve real fallback pixels')
            return actual_reader(image, **kwargs)
        with patch('controller.read_hand_preview', side_effect=read), patch('controller.time.sleep', side_effect=waits.append):
            verify = baseline_function(controller, 'verify_play_preview') if before else controller.verify_play_preview
            started = time.perf_counter()
            rejected = False
            try:
                verify(session, frame, case['expected'])
            except recovery.ObservationError:
                rejected = True
            seconds = time.perf_counter() - started
        assert rejected == (fault == 'all')
        return dict(seconds=seconds, reads=len(calls), screenshots=len(session.observations),
                    wait_seconds=sum(waits), rejected=rejected)


def recovery_sample(case, before):
    frame = load_frame(ROOT / case['path'])
    with tempfile.TemporaryDirectory() as directory:
        session = ObservedSession(directory)
        session.frame = frame
        session.joker_cache, session.hand_cache = object(), object()
        clear_ocr()
        # Classifier and UI delivery are common to both versions. Measure the
        # recovery work after classification using an actual retained frame.
        with patch('vision.classify_scene', return_value='hand'), \
                patch('perception.read_text', wraps=perception.read_text) as text:
            recover = baseline_function(recovery, 'recover_view') if before else recovery.recover_view
            started = time.perf_counter()
            if before:
                recover(session)
            else:
                recover(session, error=recovery.HandPreviewError('preview', cache_safe=True))
            seconds = time.perf_counter() - started
        return dict(seconds=seconds, full_panel_ocr=text.call_count,
                    screenshots=len(session.observations), cache_retained=session.joker_cache is not None)


def summary(samples):
    return dict(median_seconds=statistics.median(s['seconds'] for s in samples), samples=samples)


def main():
    controls = [c for c in FIXTURE['frames'] if c['role'] == 'control']
    fatal = next(c for c in FIXTURE['frames'] if c['role'] == 'fatal')
    # Warm model initialization is excluded equally; all measured OCR crops
    # are cold so exact-frame text caching cannot hide repeated work.
    perception.read_hand_preview(load_frame(ROOT / controls[0]['path']))
    results = {}
    for fault, cases in [('none', controls), ('first', [fatal] * 3), ('all', [fatal] * 3)]:
        samples = {'before': [], 'after': []}
        for index, case in enumerate(cases):
            for before in ((True, False) if index % 2 == 0 else (False, True)):
                samples['before' if before else 'after'].append(preview_sample(case, before, fault))
        for old, new in zip(samples['before'], samples['after']):
            assert new['reads'] <= old['reads']
            assert new['screenshots'] <= old['screenshots']
            assert new['wait_seconds'] <= old['wait_seconds']
        results['preview_' + fault] = {key: summary(value) for key, value in samples.items()}
        print('preview', fault, {k: round(v['median_seconds'], 6) for k, v in results['preview_' + fault].items()}, flush=True)
    results['preview_recovery'] = {
        key: summary([recovery_sample(fatal, before) for _ in range(3)])
        for key, before in [('before', True), ('after', False)]}
    shop_samples = {'before': [], 'after': []}
    for index in range(3):
        for before in ((True, False) if index % 2 == 0 else (False, True)):
            factory = (lambda: baseline_function(runner, 'shop')) if before else None
            started = time.perf_counter()
            counts = shop_scenario(shop_factory=factory, real_ocr=True)
            shop_samples['before' if before else 'after'].append(dict(seconds=time.perf_counter() - started, **counts))
    results['unchanged_rejected_offer'] = {key: summary(value) for key, value in shop_samples.items()}
    output = dict(before_version=json.loads((BASE / 'before-version.json').read_text('utf-8'))['id'],
        after_version=code_version()['id'], game_run=False,
        scope='offline cold-crop OCR/CPU and operation counts; first/all read failures injected; desktop delivery and common scene classifier excluded',
        results=results)
    path = ROOT / 'analysis/stability_budget_benchmark_20261017.json'
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', 'utf-8')
    for key, result in results.items():
        print(key, {k: round(v['median_seconds'], 6) for k, v in result.items()}, flush=True)


if __name__ == '__main__':
    main()
