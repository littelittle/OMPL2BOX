# Project Init

## 1) 这个项目在做什么

这是一个基于 **PyBullet + OMPL + VAMP** 的机器人操作仿真项目，主要目标是：

- 在仿真中加载箱体（折叠箱 / mailer box）；
- 用 Franka Panda 机械臂执行 flap 抓取与关盖动作；
- 在关节空间做碰撞检测和轨迹规划（OMPL / VAMP）。

当前主流程入口是 `main.py`，默认执行“关闭双 flap”的 demo。

## 2) 快速运行

```bash
python main.py --gui --mode unpack --robot panda
```

可用参数：

- `--config`：配置文件，默认 `config/defaults.json`
- `--mode`：`unpack` / `pick-place`（当前仅 `unpack` 可用）
- `--gui` / `--nogui`
- `--robot`：`kuka` / `panda`（当前仅 `panda` 可用）

注意：

- 程序末尾是无限 `while True` 仿真循环，需手动停止。
- `pick-place` 分支和 `kuka` 分支在 `main.py` 中目前是 `NotImplementedError`。

## 3) 代码结构速览

```text
.
├── main.py                     # 主入口：建场景 + 加箱体 + 调 planner 任务
├── test_env.py                 # 调试脚本（含大量实验逻辑和断点）
├── trial.py                    # PyBullet 连接小试验
├── config/
│   └── defaults.json           # 默认配置
├── robot_sim/
│   ├── __init__.py             # 对外导出
│   ├── sim_context.py          # 仿真连接/物理参数/地面/台座
│   ├── foldable_box.py         # 四 flap 折叠箱模型与几何 oracle
│   ├── generic_planner.py      # 通用配置/碰撞检测/轨迹执行基类
│   ├── grip_planner.py         # Panda 夹爪 planner（主力）
│   ├── suck_planner.py         # KUKA 吸附风格 planner（旧主线）
│   ├── assets/                 # URDF/mesh 资源（含 mailer box 101）
│   └── utils/                  # 向量、路径、接触帧等工具
└── temp/                       # 历史实验脚本
```

## 4) 当前主执行链路（main）

1. 读取 `config/defaults.json`
2. `make_sim(...)` 创建 PyBullet 世界
3. `create_pedestal(...)` 放置静态台座
4. `FoldableBox(...)` 加载折叠箱 URDF
5. `PandaGripperPlanner(...)` 加载 Panda + 规划器
6. `planner.close_double_flap()` 依次关两个 flap
7. 进入仿真循环持续 step

## 5) 关键类职责

- `FoldableBox`：负责折叠箱模型和 `get_flap_keypoint_pose` 几何 oracle（key/normal/axis/extended/angle）。
- `GenericPlanner`：关节配置读写、状态合法性检查、轨迹执行底座。
- `PandaGripperPlanner`：
  - IK + 碰撞检查
  - OMPL / VAMP 规划
  - gripper 开闭与抓取判定
  - 任务编排（`close_flap`、`close_double_flap`）
- `KukaOmplPlanner`：KUKA 版本规划器，保留较多旧流程实现。

## 6) 依赖与环境

代码里直接依赖：

- `pybullet`
- `pybullet_data`
- `numpy`
- `ompl`（Python 绑定）
- `vamp`

目前仓库内没有 `requirements.txt` / `pyproject.toml`，环境需手动准备。

## 7) 现状与风险点（接手时优先知道）

- `main.py` 的可用路径比较窄：只支持 `panda + unpack`。
- `config/defaults.json` 的 `robot` 字段目前基本不会生效（CLI 默认值会覆盖）。
- `grip_planner.py` 里存在调试残留（`ipdb`、打印较多、重复定义 `close_flap`）。
- `test_env.py` 也是实验脚本，含断点和无限循环，不属于稳定入口。
- 资产目录里有新增/实验文件（如 `assets/101`、zip、临时脚本），需按实际任务甄别。

## 8) 建议的下一步整理

1. 补一个可复现环境文件（`requirements.txt` 或 `pyproject.toml`）。
2. 清理 `main.py` 参数语义（避免默认参数覆盖 config）。
3. 把 `grip_planner.py` 的实验/调试逻辑拆分为模块或脚本。
4. 给 `close_double_flap` 主链路补最小可回归测试（哪怕是 smoke test）。

