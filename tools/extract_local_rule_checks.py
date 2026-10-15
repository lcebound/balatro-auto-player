"""Record the installed game's rules used in this offline review."""
import json
import zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
with zipfile.ZipFile(ROOT / 'Balatro/Balatro.exe') as game:
    blind = game.read('blind.lua').decode('utf-8')
    card = game.read('card.lua').decode('utf-8')
    rules = []
    for file, source, marker in [('blind.lua', blind, 'self.name == "The Hook"'),
                                  ('card.lua', card, "self.ability.name == 'Vampire' and not context.blueprint")]:
        at = source.find(marker)
        if at >= 0:
            rules.append(dict(file=file, line=source[:at].count('\n') + 1,
                              excerpt=source[at:at + 1600]))
    (ROOT / 'analysis/local_rules_20261009.json').write_text(json.dumps(rules, indent=2), 'utf-8')
    for rule in rules:
        print(rule['file'], rule['line'])
