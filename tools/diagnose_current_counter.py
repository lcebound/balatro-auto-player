"""Compare scoped OCR of real malformed current multiplier frames."""
import sys
from pathlib import Path
import cv2
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perception import read_text, joker_tooltip_words, tooltip_rows

for name in ('vampire_decimal_lost', 'vampire_prefix_misread', 'vampire_decimal_lost_12'):
    path = ROOT / 'tests/fixtures' / f'{name}_20261009.png'
    image = cv2.resize(cv2.imdecode(np.fromfile(path, np.uint8), 1), (2064, 1191),
                       interpolation=cv2.INTER_NEAREST)
    words, _ = joker_tooltip_words(read_text(image, (525, 330, 2020, 850)), 1211)
    print(name, words)
    current = next(w for w in words if 'currently' in w[0].lower())
    x, y = current[2]
    region = (max(525, x - 140), y - 22, min(2020, x + 300), y + 22)
    print('row full', read_text(image, region, line=True))
    print('row detect', read_text(image, region))
    gray = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    print('gray full', read_text(gray, region, line=True))
    print('gray detect', read_text(gray, region))
