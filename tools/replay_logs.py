"""Audit existing JSONL and replay representative screenshots offline."""
from __future__ import annotations
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from vision import read_hand
from perception import observed_state, read_jokers, joker_counter, joker_from_definition
from strategy import RunState, decide_play


def main():
    summaries, frames = [], []
    for directory in sorted((ROOT / "runs").iterdir()):
        log = directory / "events.jsonl"
        if not log.exists():
            continue
        records = [json.loads(line) for line in log.read_text(encoding="utf8").splitlines()]
        hands = [record for record in records if record["event"] == "hand_decision"]
        summaries.append({"run": directory.name,
            "games_completed": [record for record in records if record["event"] == "game_end"],
            "fatal": [record for record in records if record["event"] == "fatal"],
            "hand_decisions": len(hands), "recognized_counts": dict(Counter(len(h["cards"]) for h in hands)),
            "categories": dict(Counter(h["category"] for h in hands))})
        selected = set()
        latest = None
        for record in records:
            if record["event"] == "observe":
                latest = directory / Path(record["screenshot"].replace("\\", "/")).name
            if record["event"] != "hand_decision" or latest is None:
                continue
            count = len(record["cards"])
            key = (count, record["category"])
            if key in selected or len(selected) >= 6:
                continue
            selected.add(key)
            frames.append((latest, record))
    important = ["0043-hand-3.png", "0069-hand-2.png", "0092-hand-0.png"]
    directory = ROOT / "runs" / "test-20260930-090648"
    frames = [(directory / name, {}) for name in important] + frames
    results = []
    for path, old in frames:
        if not path.exists():
            continue
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        image = cv2.resize(image, (2064, 1191), interpolation=cv2.INTER_NEAREST)
        row = {"screenshot": str(path.relative_to(ROOT)), "old_cards": old.get("cards"), "old_category": old.get("category")}
        try:
            state, count = observed_state(image, RunState())
            found = read_hand(image, count)
            joker_count_value, _ = joker_counter(image)
            jokers = read_jokers(image, count=joker_count_value)
            from dataclasses import replace
            state = replace(state, jokers=tuple(joker_from_definition(key) for _, key, _ in jokers if key))
            decision = decide_play([card for card, _, _ in found], state)
            row.update(count=count, cards=[f"{c.rank}{c.suit}" for c, _, _ in found],
                jokers=[j.name for j in state.jokers], category=decision.estimate.category,
                expected_score=decision.estimate.score, indices=decision.indices,
                actual_score=state.round_score, target=state.target_score,
                limitation="historical screenshots lack tooltip counters and card modifiers")
        except Exception as error:
            row["error"] = str(error)
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    output = ROOT / "analysis" / "log_replay.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps({"runs": summaries, "replays": results}, ensure_ascii=False, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
