"""Export tabular history before removing a batch's raw logs/screenshots."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_history import read_events, summarize_events, save_history


def enrich_frames(records, run_dir):
    import cv2
    import numpy as np
    from perception import read_numbers, read_text
    output, frame_path, missing = [], None, []
    for record in records:
        output.append(record)
        if record.get('event') == 'observe':
            frame_path = run_dir / record['screenshot'].replace('\\', '/').split('/')[-1]
        if record.get('event') != 'scene' or record.get('name') not in ('cashout', 'loss', 'win'):
            continue
        if frame_path is None or not frame_path.exists():
            missing.append(str(frame_path))
            continue
        image = cv2.imdecode(np.fromfile(frame_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            missing.append(str(frame_path))
            continue
        image = cv2.resize(image, (2064, 1191), interpolation=cv2.INTER_NEAREST)
        counters = read_numbers(image)
        end_screen = None
        if record['name'] in ('loss', 'win'):
            end_screen = [text for text, confidence, _ in read_text(image, (550, 180, 1720, 1100)) if confidence > .8]
        output.append(dict(time=record['time'], event='round_snapshot', counters=counters,
            settled='round_score' in counters, source='saved_settled_frame_ocr',
            scene=record['name'], screenshot=frame_path.name, end_screen=end_screen))
        print(json.dumps(dict(frame=frame_path.name, scene=record['name'], counters=counters), ensure_ascii=False), flush=True)
    return output, missing


def merge_saved_snapshots(records, snapshots):
    # Wall-clock timestamps may be identical for scene/game_end. Preserve the
    # original event order and attach its snapshot before the game ends.
    cached = {}
    for snapshot in snapshots:
        if snapshot.get('source') == 'saved_settled_frame_ocr':
            cached.setdefault((snapshot['time'], snapshot['scene']), []).append(snapshot)
    merged = []
    for row in records:
        merged.append(row)
        if row.get('event') == 'scene':
            merged.extend(cached.get((row['time'], row['name']), ()))
    return merged


def describe(summary):
    b = summary['batch']
    lines = [f"# 运行记录：{b['batch_id']}", '',
        f"目标 {b['target_games']} 整局；启动 {b['started_games']} 局；正常结束 {b['completed_games']} 局；"
        f"脚本中断 {b['interrupted_games']} 局；未开始 {b['not_started_games']} 局。", '',
        f"正常结束局平均到达 Ante：{b['mean_ante_completed']}；平均通过盲注数：{b['mean_blinds_cleared_completed']}。", '',
        '**这是未完成批次。完整十局的平均值为空，不把未开始局填成零，也不把脚本中断计成输局。**', '',
        '| 局 | 结果 | Ante / 最后盲注 | 通过盲注 | 最后盲注实际得分 / 目标 | 耗时（分钟） | 出牌 / 弃牌 | 买入 | 末尾现金 |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for g in summary['games']:
        if g['result'] == 'not_started':
            lines.append(f"| {g['game_number']} | 未开始 | — | — | — | — | — | — | — |")
            continue
        lines.append(f"| {g['game_number']} | {g['result']} | {g['ante_reached']} / {g['last_blind']} | "
            f"{g['blinds_cleared']} | {g['last_blind_score']} / {g['last_blind_target']} | "
            f"{g['duration_seconds'] / 60:.2f} | {g['plays']} / {g['discards']} | {g['purchases']} | ${g['final_money']} |")
    lines += ['', '## 逐局补充', '']
    for g in summary['games']:
        if g['result'] == 'not_started':
            continue
        lines += [f"### 第 {g['game_number']} 局", '',
            f"- 时间：{g['started_at']} 至 {g['ended_at'] or g['last_observed_at']}。",
            f"- 牌组 / 难度：{g['deck']} / {g['stake']}；最后 Boss：{g['last_boss'] or '无'}。",
            f"- 种子：{g['run_seed'] or '未取得'}；结束面板最高单手得分：{g['best_hand_score_reported']}。",
            f"- 结束面板统计：打出牌数 {g['cards_played_reported']}，弃掉牌数 {g['cards_discarded_reported']}，到达 Round {g['round_reached_reported']}。",
            f"- 已记录最高单手实际得分：{g['best_hand_score_observed']}；实际得分已读回的出牌数：{g['observed_scored_plays']}/{g['plays']}。",
            f"- 各盲注观察到的累计得分之和：{g['sum_blind_scores_observed']}；是否全部取得结算得分：{g['score_complete']}。",
            f"- 购买：{json.dumps(g['purchase_details'], ensure_ascii=False)}。",
            f"- 末尾小丑：{json.dumps(g['final_jokers'], ensure_ascii=False)}。",
            f"- 牌型等级：{json.dumps(g['final_levels'], ensure_ascii=False)}。",
            f"- 出牌牌型分布：{json.dumps(g['played_categories'], ensure_ascii=False)}。",
            f"- 自动恢复 {g['automatic_recoveries']} 次；unknown 场景 {g['unknown_scenes']} 次；异常：{g['error'] or '无'}。", '']
    lines += ['## 统计口径', '',
        '- 一轮为计划的十整局；`is_full_batch=True` 才表示十局均有游戏胜负结果。脚本中断单列。',
        '- `ante_reached` 为最后到达的底注，`blinds_cleared` 为实际通过的盲注数；二者不要混用。',
        '- `stage_reached=(Ante-1)*3+盲注序号`，小盲/大盲/Boss 的序号分别为 1/2/3。',
        '- `mean_*_completed` 只对正常结束局求平均，分母为 `completed_games`；`mean_*_full_batch` 在十局未完时为空。',
        '- 分数使用界面实际累计分及相邻出牌后的增量，不累加策略预测分。',
        '- `last_score_quality=observed_lower_bound` 表示没有最终结算读回，可能漏最后一手；`total_score_exact` 只有每个盲注均取得结算分且该局正常结束才填。',
        '- `best_hand_score_observed` 是已取得读回的单手最高值；缺少读回时不能当成游戏统计面板保证的最高值。',
        '- 旧批次没有记录代码哈希，版本为 `legacy_unrecorded`；不能把后来改好的代码标成旧批次使用的版本。',
        '- 详细数据在同目录 `summary.json`；跨批次追加的表在上级 `rounds.csv`、`games.csv`、`blinds.csv`。CSV 使用 UTF-8 BOM，便于 Excel 导入。',
        '- 原始截图和 JSONL 清理后，保留本记录、结构化汇总及离线回归用的牌面数据。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output', default=str(ROOT / 'results'))
    parser.add_argument('--read-saved-frames', action='store_true')
    parser.add_argument('--reuse-saved-snapshots', action='store_true')
    args = parser.parse_args()
    directory = Path(args.run_dir).resolve()
    records, warnings = read_events(directory / 'events.jsonl')
    if args.read_saved_frames:
        records, missing = enrich_frames(records, directory)
        warnings += [f'Missing saved frame: {name}' for name in missing]
    elif args.reuse_saved_snapshots:
        archived = json.loads((Path(args.output) / directory.name / 'summary.json').read_text(encoding='utf-8'))
        records = merge_saved_snapshots(records, archived['settled_snapshots'])
    summary = summarize_events(records, directory.name)
    summary['warnings'] = warnings
    log_path = directory / 'events.jsonl'
    summary['raw_source'] = dict(events_sha256=hashlib.sha256(log_path.read_bytes()).hexdigest(),
        events_bytes=log_path.stat().st_size, screenshots=len(list(directory.glob('*.png'))),
        code_version_recorded_at_runtime=summary['batch']['code_version'] != 'legacy_unrecorded')
    summary['settled_snapshots'] = [r for r in records if r.get('event') == 'round_snapshot']
    save_history(summary, args.output)
    report = Path(args.output) / directory.name / 'review.md'
    report.write_text(describe(summary), encoding='utf-8')
    print(json.dumps(summary['batch'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
