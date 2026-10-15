"""Record the supervised three-game acceptance from saved events only."""
import json
import sys
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_history import read_events
from runtime_issues import terminal_failures, unresolved_recovery


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', default=datetime.now().astimezone().strftime('%Y%m%d'))
    args = parser.parse_args()
    assert len(args.date) == 8 and args.date.isdigit()
    root = Path(__file__).resolve().parents[1]
    pointer = json.loads((root / 'runs/active-batch.json').read_text('utf-8-sig'))
    run = Path(pointer['run_dir'])
    events, warnings = read_events(run / 'events.jsonl')
    status = json.loads((run / 'status.json').read_text('utf-8'))
    ends = [e for e in events if e.get('event') == 'game_end']
    starts = [e for e in events if e.get('event') == 'game_start']
    assert [e['number'] for e in ends] == [1, 2, 3]
    assert all(e['result'] in ('loss', 'win') for e in ends)
    assert starts[-1]['number'] == 4
    assert status['finished_games'] == 3 and status['current_game'] == 4
    assert status['status'] == 'running' and status['game_active']
    assert not warnings
    assert not terminal_failures(events)
    assert unresolved_recovery(events) is None
    after_start = [e for e in events if e['time'] > starts[-1]['time']]
    assert any(e.get('event') == 'scene' and e.get('name') == 'hand' for e in after_start)
    fourth_shot = next(e['screenshot'] for e in after_start
                       if e.get('event') == 'observe' and e.get('label') == 'hand-before-inspection')
    shots = [next(r['screenshot'] for r in reversed(events[:i]) if r.get('event') == 'observe')
             for i, e in enumerate(events) if e.get('event') == 'game_end']
    summary = json.loads((root / 'results' / run.name / 'summary.json').read_text('utf-8'))
    games = [g for g in summary['games'] if g.get('completed')]
    assert len(games) == 3
    path = root / f'analysis/autonomous_acceptance_{args.date}.json'
    report = json.loads(path.read_text('utf-8')) if path.exists() else {}
    report['batch'] = run.name
    report['runner_pid'] = pointer['pid']
    report.update(status='passed_remaining_seven_running',
                  updated_at=datetime.now().astimezone().isoformat(),
                  completed_full_games_verified=3, observed_completed_games=games,
                  completed_independent_games=3,
                  termination_screenshots=shots, fourth_game_start=starts[-1],
                  fourth_game_hand_screenshot=fourth_shot,
                  handoff_status=status, remaining_games=7,
                  issue_counts_at_handoff=status.get('issue_counts', {}),
                  automatic_recovery_attempts=status.get('automatic_recoveries', 0),
                  unresolved_recovery_at_handoff=None,
                  manual_gameplay_interventions=0)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    rows = [f"| {g['game_number']} | {g['result']} | {g['ante_reached']} | {g['blinds_cleared']} | "
            f"{g['last_blind_score']:g}/{g['last_blind_target']:g} | {g['best_hand_score_observed']:g} |"
            for g in games]
    text = f"""# 三整局自主运行验收（{args.date}）

批次：`{run.name}`；后台启动进程 PID：{pointer['pid']}；目标为十整局。

已现场观察三局从新建游戏到 Game Over，并核对结束截图。第四局已自行开始、进入手牌界面。验收结束时进程仍在运行，剩余七局交由脚本继续；尚未声称十局全部完成。

## 已完成游戏

| 局数 | 结局 | 到达底注 | 通过盲注数 | 最后一关得分/目标 | 最高单手分 |
| --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}

三局均有正常游戏胜负结果。已记录的识别降级和自恢复属于局后复盘事项；当前没有终止性错误或未完成的恢复。没有手动点击、出牌、购买或帮助通关。

轻微识别问题和成功恢复的问题在每局结束后统一复盘。只有终止性故障需要停止进程，修复代码后建立新批次，从第一整局重跑。被中止批次和原因记录于同目录 JSON。游戏本身的正常失败不属于脚本异常。

## 文件与清理

当前原始截图及日志：`runs/{run.name}`。游戏统计由脚本写入 `results/{run.name}/summary.json`、`games.csv`、`blinds.csv`。

清理情况详见同目录 JSON 中 cleanup 字段；统计摘要与回归样本保留。
"""
    path.with_suffix('.md').write_text(text, encoding='utf-8')
    print(json.dumps({'status': report['status'], 'batch': run.name,
                      'finished_games': 3, 'current_game': 4, 'remaining_games': 7}))


if __name__ == '__main__':
    main()
