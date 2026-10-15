"""Inspect native save format read-only, without loading or executing Lua."""
import json
import os
from pathlib import Path
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
save = Path(os.environ['APPDATA']) / 'Balatro/1/save.jkr'
raw = save.read_bytes()
decoded = None
compression = None
for bits in (15, -15, 31):
    try:
        decoded = zlib.decompress(raw, bits).decode('utf-8')
        compression = bits
        break
    except (zlib.error, UnicodeError):
        pass
with zipfile.ZipFile(ROOT / 'Balatro/Balatro.exe') as game:
    sources = {}
    for file in ('engine/save_manager.lua', 'functions/misc_functions.lua', 'functions/state_events.lua'):
        if file not in game.namelist():
            continue
        text = game.read(file).decode('utf-8')
        for marker in ('function get_compressed', 'function compress_and_save', 'save_run()', 'save.jkr'):
            at = text.find(marker)
            if at >= 0:
                sources[f'{file}:{text[:at].count(chr(10)) + 1}'] = text[at:at + 700]
report = dict(path=str(save), bytes=len(raw), mtime=save.stat().st_mtime,
              decompressed_bytes=len(decoded.encode('utf-8')) if decoded else None,
              compression_wbits=compression,
              visible_state_fields={key: key in (decoded or '') for key in
                ('cardAreas', 'jokers', 'hand', 'current_round', 'hands_left', 'discards_left', 'x_mult', 'BLIND', 'STATE')},
              serializer_sources=sources, connected_to_runtime=False,
              reason='Native checkpoints may be delayed; freshness and UI agreement must be verified before use')
(ROOT / 'analysis/native_save_feasibility_20261009.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), 'utf-8')
print(json.dumps({k: v for k, v in report.items() if k != 'serializer_sources'}, ensure_ascii=False))
