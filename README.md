# Balatro Auto Player · 小丑牌视觉自动化与决策脚本

基于 Python、Airtest、OpenCV 与 OCR 的 Windows 桌面自动游玩脚本，通过游戏窗口完成观察、决策、操作与结果校验。

## 主要能力

- **策略决策**：综合牌型、卡牌属性、小丑效果与触发顺序、Boss 限制评估出牌；结合抽牌预测选择弃牌，依据经济预算、成长潜力与协同关系决定购买和构筑。
- **视觉执行**：识别手牌、商品与动态数值，使用属性缓存和增量识别减少重复观察，校验选牌、购买、出售及消耗牌操作结果。
- **状态与恢复**：保存动作事务和局内状态，区分降级、恢复、致命退出与正常胜负，支持批量运行与断点恢复。
- **离线工具**：提供策略回放、性能分析、运行监督及独立实验统计导出工具。

## 环境与本地资源

- Windows；本地开发使用 Python 3.12。
- 自行安装并运行 Balatro。当前脚本要求英文界面，默认批次使用 Red Deck / White Stake。
- 当前实现从项目内 `Balatro/Balatro.exe` 读取游戏规则与图集；请自行在本地准备该路径，仓库不分发游戏文件。
- 视觉坐标按 2064 × 1191 截图标定；窗口初始化使用 1376 × 794 逻辑像素，对应本地 150% 显示缩放。其他显示环境需要重新标定。
- 仓库已包含 `assets/ui/` 下的界面识别与出牌、弃牌、排序控件模板，克隆后可直接使用。

## 安装与运行

在 PowerShell 中执行：

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

准备本地游戏，启动游戏并保持桌面可交互后：

```powershell
# 观察与连接诊断
.venv\Scripts\python.exe bot.py probe

# 执行一手
.venv\Scripts\python.exe runner.py one-hand

# 从新游戏开始，目标十整局
.venv\Scripts\python.exe runner.py batch --games 10

# 恢复指定本地批次
.venv\Scripts\python.exe runner.py batch --games 10 --run-dir runs/<batch-name> --resume
```

也可使用 `tools/start_batch.ps1 -Games 10` 后台启动，或在自行安装的 AirtestIDE 中打开 `balatro_ide.air/balatro_ide.py`。

运行期间会控制鼠标与游戏窗口。日志和状态默认写入本地 `runs/`，独立统计写入 `results/`；上述目录均被 Git 忽略。只在真实 Win/Loss 后计一整局，预测得分不作为实测得分。

## 代码结构

| 文件 | 用途 |
| --- | --- |
| `runner.py` / `bot.py` | 批次主循环、游戏窗口连接与观察 |
| `strategy.py` / `planning.py` | 得分模型、出牌弃牌与商店决策 |
| `economy.py` / `engine_development.py` | 经济预算与成长评估 |
| `perception.py` / `vision.py` / `item_identity.py` | OCR、图像识别与身份解析 |
| `controller.py` / `shop_extras.py` / `consumables.py` | 操作执行、商店与消耗牌交互 |
| `hand_cache.py` / `joker_cache.py` / `hand_levels.py` | 手牌、小丑与牌型等级缓存 |
| `recovery.py` / `runtime_issues.py` / `run_history.py` | 状态恢复、事件与实验统计 |
| `tests/` / `tools/` | 测试源码与辅助工具 |
