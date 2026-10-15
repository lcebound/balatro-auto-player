"""Save this review and accurate stopped-batch status without starting a game."""
import json
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_history import read_events, summarize_events, code_version

analysis = ROOT / 'analysis'
path = analysis / 'autonomous_acceptance_20261009.json'
acceptance = json.loads(path.read_text('utf-8'))
latest = 'autonomous-20261009-062544'
if not any(r['batch'] == latest for r in acceptance['restarts']):
    acceptance['restarts'].append(dict(batch=latest,
        reason='Stone Card was visibly raised but ordinary corner templates matched its internal texture; selection repeatedly toggled it and failed verification.',
        completed_games=0, manual_gameplay_interventions=0, ante_reached=4, blinds_cleared=11))
acceptance.update(status='offline_fixes_verified_not_live_accepted',
    reviewed_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
    runner_pid=None, completed_independent_games=0, code_version=code_version(),
    verification='Offline screenshots, scoring replays and regression tests only; live three-game acceptance remains unmet')
acceptance['offline_review'] = dict(report='analysis/log_review_20261009.md',
    changed_files=['vision.py', 'perception.py', 'controller.py', 'strategy.py', 'planning.py', 'runner.py', 'joker_cache.py'],
    tested_suites=[dict(tests=62, skipped=2), dict(tests=70, skipped=0), dict(tests=120, skipped=0),
                   dict(tests=16, skipped=0)],
    game_restarted=False, manual_gameplay_interventions=0)
path.write_text(json.dumps(acceptance, ensure_ascii=False, indent=2), 'utf-8')

table = []
reasons = {r['batch']: r['reason'] for r in acceptance['restarts']}
for directory in sorted((ROOT / 'runs').glob('autonomous-20261009-*')):
    records, warnings = read_events(directory / 'events.jsonl')
    summary = summarize_events(records, directory.name)
    game = next(g for g in summary['games'] if g['result'] != 'not_started')
    failure = next((r for r in reversed(records) if r['event'] in ('automatic_recovery', 'batch_error', 'fatal')), {})
    table.append(dict(batch=directory.name, completed=summary['batch']['completed_games'],
                      ante=game['ante_reached'], blinds=game['blinds_cleared'],
                      minutes=round(game['duration_seconds'] / 60, 1), error=failure.get('error'),
                      diagnosis=reasons.get(directory.name), warnings=warnings))
(analysis / 'runtime_failure_review_20261009.json').write_text(json.dumps(table, ensure_ascii=False, indent=2), 'utf-8')
scores = json.loads((analysis / 'score_review_20261009.json').read_text('utf-8'))
lines = [
'# 2026-10-09 日志复盘与代码改进', '',
f"代码版本：`{acceptance['code_version']['id']}`。本次仅修改代码与离线验证，游戏脚本当前已停止。", '',
'## 为什么多次中断', '',
'完整游戏完成数为 0，三局独立运行验收尚未通过。最近五次中断发生在不同的界面分支：焦点中断后菜单未关闭、终局 Boss 隐藏小丑、提示框混入商店卡名、愚者说明引用行星、石头牌选中位置误判。前四类已在此前逐次重启时修复并保存回归样本；第五类是本次主要定位与修复的停止原因。', '',
'一局包含大量读图、点击、动画和界面切换。现有视觉与状态验证仍有未覆盖的分支；必须识别的牌或操作结果读不准时，脚本会拒绝继续。与此同时，小丑名称读对并不能证明倍率读对，数值误读此前会直接进入预测模型，影响出牌和购买。', '',
'| 批次 | 已完成整局 | 到达底注 | 已过盲注 | 记录时长（分钟） | 最后异常 |',
'| --- | ---: | ---: | ---: | ---: | --- |']
for row in table:
    ante = row['ante'] if row['ante'] is not None else '未读到'
    error = row['error'] or '界面识别未知，监督停止'
    lines.append(f"| {row['batch']} | {row['completed']} | {ante} | {row['blinds']} | {row['minutes']} | {error} |")
lines += ['', '## 本次稳定性修复', '',
'1. **石头牌位置**：已经通过提示框证明是石头牌的槽位，只用石头纹理定位。灰度平滑抑制 CRT 扫描线，旋转模板保留完整范围，遮罩排除补边；从变换恢复真实左上角，以此判断是否抬起。仍要求纹理和位置得到验证，缺失纹理会拒绝点击。真实四帧回放中，石头牌的相关度均超过 0.85；此前两张选中帧的普通模板会把内部纹理当作角标。',
'2. **当前倍率独立复核**：对多个成长倍率小丑，单独读取 Currently 行，比较彩色/灰度、整行/分段四种 OCR 输出，需要至少两个结果一致且无并列冲突。吸血鬼额外检查每次出牌最多增长 ×0.5 的上界及累计出牌上界。小数点不会按猜测补入；无法证明的倍率明确记为下界。吸血鬼已有的可信增长不会因一次漏读重置到 ×1。',
'3. **商店与出售身份**：新报价中的成长倍率也经过独立复核；出售前只读取目标小丑的主提示框名称，防止其他商店标题混入。购买、出售及选牌的身份和结果验证仍保留。',
'4. **稳定小丑缓存**：纠正 Mail-In Rebate 与 8 Ball 的内部键名，使它们能走已有的缓存复用路径。所有缓存白名单键都在 150 张小丑的本地目录中。', '',
'## 评分和决策树', '',
f"离线匹配 {len(scores)} 次已确认出牌的后续得分记录，其中 {sum(c['actual'] > 0 for c in scores)} 次增量为正。分析保存了原始策略输入，未将中断的游戏计入十局完成率或完成游戏平均值。", '',
'- **吸血鬼误读**：日志曾记录 ×11、×12，甚至 ×612。实际 ×1.1/×1.2 的三张失败截图经新读取恢复正确。一个真实出牌案例旧预测 21,090、实际 2,280；用截图证明的 ×1.1 重算，得到 2,280。',
'- **钩子的计分顺序**：先随机弃掉最多两张留手牌，再计算钢铁、男爵、射月、哑剧、举拳和黑板等留手效果。期望模式枚举所有合法丢牌结果；保守模式取最坏结果。实际案例旧预测 1,536、实际 704；修正后期望 1,203.2，下界 704，不能再把留手皇后的增益当成必然触发。Boss 被禁用时恢复通常计分。下一手模拟也修复了仅留一张牌时的补牌数量。',
'- **小牌组弃牌成本**：后期高牌/对子引擎已达到所需节奏时，提高消耗弃牌次数的成本，保留应付下一手差牌的资源；明显的大幅增益仍能触发弃牌。Castle、Yorick、Trading Card、Burnt Joker 的成长机会不走这项额外惩罚。',
'- **吸血鬼购买支持**：新 ×1 吸血鬼根据已有增强牌和点金手获得长期价值；缺少支持时降低购买先验，已有倍率和真实整轮预测仍参与比较。',
'- **运行耗时与日志准确性**：保守分数计算实际选中的牌，不再重新搜索另一套最优手牌。减少一次完整组合搜索，同时避免日志把另一种选牌的分数写到当前决策上。', '',
'幸运牌、血石和太空小丑的实际随机结果仍可能偏离期望值；这些偏差不能通过强行让期望值贴合某一次结果来消除。', '',
'## 验证与限制', '',
'已通过相关离线回归组：62 项（其中 2 项跳过）、70 项、120 项，以及最终新增案例组 16 项；这些组存在重复用例，不能相加当成独立总数。另外确认消费后状态、嘴巴零分循环和真实吸血鬼分数案例；修改的运行模块通过编译检查。测试使用 FakeSession 和保存的截图，不连接游戏设备。', '',
'关键回归素材在 `tests/fixtures/`，原始记录保留在 `runs/`，评分/弃牌回放和异常清单在 `analysis/`。', '',
'**本次没有重跑游戏，不能据此宣称已稳定完成三整局，也不能断言修改已经提高了实测平均底注或得分。** 下一次现场验收仍应使用新批次，并按既定要求在脚本故障修复后从第一局重新开始。', '']
(analysis / 'log_review_20261009.md').write_text('\n'.join(lines), 'utf-8')
print(json.dumps(dict(report=str(analysis / 'log_review_20261009.md'), code_version=acceptance['code_version']['id'],
                      batches=len(table), completed=sum(r['completed'] for r in table)), ensure_ascii=False))
