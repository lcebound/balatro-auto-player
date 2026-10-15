# Balatro Auto Player

基于 Python、Airtest、OpenCV 与 OCR 的小丑牌视觉自动化与决策系统，通过游戏窗口完成状态识别、策略规划、操作执行和结果校验。

## 功能

- **出牌与弃牌**：根据牌型收益、卡牌属性、小丑效果与触发顺序、Boss 限制评估出牌，结合抽牌预测选择弃牌。
- **购买与构筑**：综合经济预算、即时收益、成长潜力与小丑协同，规划购买、出售、刷新及消耗牌使用。
- **视觉识别**：识别手牌、商品和动态数值，通过属性缓存与增量识别减少重复观察。
- **操作与恢复**：校验选牌和交易结果，保存局内状态与待确认动作，支持异常恢复、批量运行及断点续跑。

## 运行环境

- Windows、Python 3.12。
- Balatro 英文界面，默认使用 Red Deck / White Stake。
- 将本地游戏目录放置或链接到项目根目录的 `Balatro/`，使 `Balatro/Balatro.exe` 可访问。脚本从该文件读取游戏规则与图集。
- 显示缩放设为 150%。窗口初始化尺寸为 1376 × 794 逻辑像素，识别坐标对应 2064 × 1191 截图；使用其他分辨率或缩放时需调整坐标与模板。

界面与操作控件的识别模板位于 `assets/ui/`。

## 快速开始

在 PowerShell 中执行：

```powershell
git clone https://github.com/lcebound/balatro-auto-player.git
cd balatro-auto-player
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

准备好游戏目录后启动 Balatro，保持游戏窗口可见、桌面可交互。

```powershell
# 检查游戏窗口连接并采集观察帧
.venv\Scripts\python.exe bot.py probe

# 执行一手决策与操作
.venv\Scripts\python.exe runner.py one-hand

# 自动游玩十局
.venv\Scripts\python.exe runner.py batch --games 10
```

指定批次目录并断点续跑：

```powershell
.venv\Scripts\python.exe runner.py batch --games 10 --run-dir runs/my-batch
.venv\Scripts\python.exe runner.py batch --games 10 --run-dir runs/my-batch --resume
```

后台运行可使用 `tools/start_batch.ps1 -Games 10`；AirtestIDE 入口为 `balatro_ide.air/balatro_ide.py`。脚本运行期间会控制鼠标和游戏窗口。

## 代码结构

| 模块 | 职责 |
| --- | --- |
| `runner.py` / `bot.py` | 批次调度、窗口连接与观察循环 |
| `strategy.py` / `planning.py` | 计分模型、出牌弃牌与商店决策 |
| `economy.py` / `engine_development.py` | 经济预算与成长评估 |
| `perception.py` / `vision.py` / `item_identity.py` | OCR、图像识别与身份解析 |
| `controller.py` / `shop_extras.py` / `consumables.py` | 操作执行、商店与消耗牌交互 |
| `hand_cache.py` / `joker_cache.py` / `hand_levels.py` | 手牌、小丑与牌型等级缓存 |
| `recovery.py` / `runtime_issues.py` / `run_history.py` | 状态恢复、运行事件与批次统计 |
| `assets/ui/` | 界面与操作控件模板 |
| `tests/` / `tools/` | 测试与辅助工具 |
