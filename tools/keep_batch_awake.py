"""Temporary idle-sleep request while a specific game batch is running.

Does not change the power plan, unlock Windows, or send any game input.
"""
import argparse
import ctypes
import json
import os
import time
from datetime import datetime
from pathlib import Path
import psutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    root = Path(__file__).resolve().parents[1]
    assert run.parent == (root / 'runs').resolve()
    process = psutil.Process(args.pid)
    birth = process.create_time()
    assert 'runner.py' in ' '.join(process.cmdline())
    api = ctypes.WinDLL('kernel32', use_last_error=True).SetThreadExecutionState
    api.argtypes = [ctypes.c_uint32]
    api.restype = ctypes.c_uint32
    held = False
    failure = None

    def record(status, **extra):
        target = run / 'desktop-awake.json'
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(status=status, helper_pid=os.getpid(),
            runner_pid=args.pid, updated_at=datetime.now().astimezone().isoformat(),
            mechanism='SetThreadExecutionState', **extra), indent=2), encoding='utf-8')
        temporary.replace(target)

    try:
        while process.is_running() and process.create_time() == birth:
            if not (run / 'status.json').exists():
                time.sleep(2)
                continue
            state = json.loads((run / 'status.json').read_text('utf-8'))
            if state['status'] in ('complete', 'failed'):
                break
            want = state['status'] != 'waiting_for_desktop'
            if want != held:
                if not api(0x80000003 if want else 0x80000000):
                    raise ctypes.WinError(ctypes.get_last_error())
                held = want
                record('held' if held else 'suspended_for_desktop_wait')
            time.sleep(15)
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        pass
    except Exception as error:
        failure = repr(error)
        record('failed', error=repr(error))
        raise
    finally:
        api(0x80000000)
        record('released', **({'error': failure} if failure else {}))


if __name__ == '__main__':
    main()
