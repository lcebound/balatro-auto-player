"""Offline atlas and tooltip-name audit. Never opens or controls Balatro."""
from __future__ import annotations
import argparse
import io
import json
from dataclasses import asdict
from pathlib import Path
import sys
import zipfile
import re
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from perception import joker_definitions, read_jokers, identify_joker_text, read_text


def audit(ocr=False):
    with zipfile.ZipFile(ROOT / "Balatro" / "Balatro.exe") as archive:
        atlas = Image.open(io.BytesIO(archive.read("resources/textures/2x/Jokers.png"))).convert("RGBA")
        font = ImageFont.truetype(io.BytesIO(archive.read("resources/fonts/m6x11plus.ttf")), 44)
    definitions = list(joker_definitions().values())
    rows = []
    started = time.monotonic()
    for index, definition in enumerate(definitions):
        col, row = definition.tile
        tile = atlas.crop((col * 142, row * 190, (col + 1) * 142, (row + 1) * 190))
        base = Image.new("RGBA", tile.size, "white")
        base.alpha_composite(tile)
        if definition.soul_tile:
            sx, sy = definition.soul_tile
            base.alpha_composite(atlas.crop((sx * 142, sy * 190, (sx + 1) * 142, (sy + 1) * 190)))
        scale = .84 if definition.key == "wee_joker" else 1.2
        width = round(142 * scale)
        height = width if definition.key == "square_joker" else round(190 * scale)
        tile = base.convert("RGB").resize((width, height), Image.Resampling.NEAREST)
        board = Image.new("RGB", (2064, 1191), (20, 65, 55))
        board.paste(tile, (1020 - width // 2, 225 - height // 2))
        matches = read_jokers(cv2.cvtColor(np.asarray(board), cv2.COLOR_RGB2BGR), count=1)
        score, predicted, _ = matches[0]
        text_key, _ = identify_joker_text([definition.name])
        rows.append({"key": definition.key, "name": definition.name, "art_prediction": predicted,
                     "art_score": round(score, 4), "art_pass": predicted == definition.key,
                     "name_pass": text_key == definition.key, "definition": asdict(definition)})
        if (index + 1) % 15 == 0:
            print(f"Atlas: {index + 1}/150; passed {sum(r['art_pass'] for r in rows)}", flush=True)
    if ocr:
        for start in range(0, len(rows), 15):
            page = Image.new("RGB", (950, 15 * 72 + 40), (28, 32, 36))
            draw = ImageDraw.Draw(page)
            group = rows[start:start + 15]
            for index, row in enumerate(group):
                draw.text((25, 20 + index * 72), row["name"], font=font, fill="white")
            words = read_text(cv2.cvtColor(np.asarray(page), cv2.COLOR_RGB2BGR))
            for index, row in enumerate(group):
                labels = [label for label, score, (_, y) in words if abs(y - (45 + index * 72)) < 32 and score > .65]
                try:
                    prediction, confidence = identify_joker_text(labels)
                except RuntimeError:
                    prediction, confidence = "", 0
                row["ocr_prediction"] = prediction
                row["ocr_labels"] = labels
                row["ocr_pass"] = prediction == row["key"]
            print(f"OCR names: {start + len(group)}/150", flush=True)
    result = {"total": len(rows), "atlas_pass": sum(r["art_pass"] for r in rows),
              "name_lookup_pass": sum(r["name_pass"] for r in rows),
              "ocr_pass": sum(r.get("ocr_pass", False) for r in rows) if ocr else None,
              "scope": "offline atlas renders and game-font names, not 150 live in-game screenshots",
              "elapsed_seconds": round(time.monotonic() - started, 2), "jokers": rows}
    output = ROOT / "analysis" / "joker_coverage.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps({k: v for k, v in result.items() if k != "jokers"}, ensure_ascii=False), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr", action="store_true")
    args = parser.parse_args()
    audit(args.ocr)
