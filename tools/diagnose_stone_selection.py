"""Offline localization diagnostics; never connects to a game device."""
import sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vision import special_corner_templates
import io
import zipfile
from PIL import Image

template = next(t for c, t in special_corner_templates().items() if c.is_stone)
with zipfile.ZipFile(ROOT / 'Balatro/Balatro.exe') as game:
    atlas = Image.open(io.BytesIO(game.read('resources/textures/2x/Enhancers.png'))).convert('RGBA')
stone = np.asarray(atlas.crop((710, 0, 852, 190)))
cv2.imencode('.png', cv2.cvtColor(stone, cv2.COLOR_RGBA2BGRA))[1].tofile(ROOT / 'analysis/stone_atlas.png')
for name in ('before', 'first', 'toggle_off', 'final'):
    path = ROOT / 'tests/fixtures' / f'stone_selection_{name}_20261009.png'
    image = cv2.resize(cv2.imdecode(np.fromfile(path, np.uint8), 1), (2064, 1191),
                       interpolation=cv2.INTER_NEAREST)
    roi = image[630:900, 1440:1660]
    candidates = []
    for scale in np.arange(.9, 1.81, .1):
        for angle in range(-20, 21, 2):
            resized = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            h, w = resized.shape[:2]
            rotated = cv2.warpAffine(resized, cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1),
                                     (w, h), borderMode=cv2.BORDER_REPLICATE)
            result = cv2.matchTemplate(roi, rotated, cv2.TM_CCOEFF_NORMED)
            _, score, _, (x, y) = cv2.minMaxLoc(result)
            candidates.append((round(score, 4), round(float(scale), 1), angle, x + 1440, y + 630))
    print(name, sorted(candidates, reverse=True)[:5])
    cv2.imencode('.png', roi)[1].tofile(ROOT / 'analysis' / f'stone_{name}.png')
    for width, height in ((45, 90), (70, 160), (142, 190)):
        source = cv2.cvtColor(stone[:height, :width, :3], cv2.COLOR_RGB2GRAY)
        candidates = []
        gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (5, 5), 1.2)
        for scale in np.arange(.9, 1.51, .1):
            resized = cv2.resize(source, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            resized = cv2.GaussianBlur(resized, (5, 5), 1.2)
            h, w = resized.shape
            for angle in range(-10, 11, 2):
                matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
                cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
                rw, rh = int(h * sin + w * cos), int(h * cos + w * sin)
                matrix[0, 2] += (rw - w) / 2
                matrix[1, 2] += (rh - h) / 2
                rotated = cv2.warpAffine(resized, matrix, (rw, rh))
                mask = np.zeros((h, w), np.uint8)
                mask[8:-8, 8:-8] = 255
                mask = cv2.warpAffine(mask, matrix, (rw, rh), flags=cv2.INTER_NEAREST)
                if rh > gray.shape[0] or rw > gray.shape[1]:
                    continue
                result = cv2.matchTemplate(gray, rotated, cv2.TM_CCOEFF_NORMED, mask=mask)
                result[~np.isfinite(result)] = -1
                _, score, _, (x, y) = cv2.minMaxLoc(result)
                corner = matrix @ np.array((0, 0, 1))
                candidates.append((round(score, 4), round(float(scale), 1), angle,
                                   round(x + 1440 + corner[0]), round(y + 630 + corner[1])))
        print(name, (width, height), sorted(candidates, reverse=True)[:3])
