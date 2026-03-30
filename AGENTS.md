# Repo Agent Guide

## 1) 仓库定位

这是一个基于 **PyBullet + OMPL + VAMP** 的机器人操作仿真仓库。当前对代理最重要的事实只有一条：

- 目前唯一相对稳定、可作为默认验证路径的是 `main.py` 中的 `panda + "Flap BoxTask"` demo。

这条主链路会：

- 创建 PyBullet 世界；
- 加载 pedestal 与四 flap 折叠箱；
- 初始化 Franka Panda 规划器；
- 调用 `close_double_flap()` 执行双 flap 关盖；
- 最后进入无限仿真循环。

下面这些路径目前都不应被当成“正式产品化入口”：

- `"mailer box task"`：CLI 里已出现，但 `main.py` 中尚未实现完整任务；
- `test_env.py`：实验/调试脚本，不是回归入口；
- `temp/`、`exp/`：历史试验与可视化残留；
- `kuka`：代码保留，但 `main.py` 主路径当前只支持 Panda。

## 2) 标准运行方式

默认从仓库根目录运行：

```bash
python main.py --gui --mode "4-flap box task" --robot panda
```

当前 `main.py` 支持的 CLI 参数：

- `--config`：默认 `config/defaults.json`
- `--mode`：`"4-flap box task"` / `"mailer box task"`
- `--gui` / `--nogui`
- `--robot`：`kuka` / `panda`

当前实际行为：

- `"4-flap box task"` 是主路径；
- `"mailer box task"` 目前没有实质任务流程；
- `--robot kuka` 在 `main.py` 主路径下会走到 `NotImplementedError`；
- 程序结尾有无限 `while True` 仿真循环，验证时需要手动停止。

## 3) 仓库结构

```text
.
├── main.py                   # 当前唯一规范入口
├── config/
│   └── defaults.json         # 默认配置
├── models/
│   ├── foldable_box.py       # 四 flap 折叠箱模型与几何 oracle
│   ├── mailer_box.py         # 较早期 mailer box 实验模型
│   └── mailer_box_101.py     # 当前导出的 MailerBox 实验模型
├── planners/
│   ├── generic_planner.py    # 关节配置、碰撞检查、轨迹执行基类
│   ├── grip_planner.py       # Panda 夹爪规划器与任务编排主实现
│   └── suck_planner.py       # 旧 KUKA/吸附风格规划器
├── scene/
│   └── sim_context.py        # PyBullet 连接、物理参数、plane、pedestal
├── utils/                    # 向量、路径、接触帧、点云等工具
├── assets/                   # URDF 与 mesh 资产
├── test_env.py               # mailer box 相关实验脚本
├── trial.py                  # PyBullet 连接小试验
├── exp/                      # 局部实验脚本
├── temp/                     # 历史实验代码与图片
└── notes_refactor.md         # 重构过程记录
```

## 4) 当前主执行链路

`main.py` 的 `"4-flap box task"` 路径当前流程如下：

1. 读取 `config/defaults.json` 或 `--config` 指定的 JSON。
2. 用 `scene.make_sim()` 和 `scene.physics_from_config()` 创建仿真环境。
3. 创建 pedestal。
4. 创建 `models.FoldableBox`。
5. 创建 `planners.PandaGripperPlanner`，并把 box oracle 与碰撞体 id 传进去。
6. 调用 `planner.close_double_flap()`。
7. 进入无限 `stepSimulation()` 循环。

如果代理要验证主功能，优先围绕这条链路理解和排查，不要先从 `test_env.py` 切入。

## 5) 关键模块职责

- `scene.sim_context`
  - 管理 PyBullet 连接、物理参数、plane 加载与 pedestal 创建。
- `models.FoldableBox`
  - 加载四 flap 折叠箱 URDF；
  - 解析箱体尺寸；
  - 提供 `get_flap_keypoint_pose()` 作为 flap 几何 oracle。
- `models.MailerBox`
  - 当前由 `models/__init__.py` 导出自 `mailer_box_101.py`；
  - 主要服务于 mailer box 实验路径，不是主流程依赖。
- `planners.GenericPlanner`
  - 提供机器人无关的关节读写、状态合法性检查、轨迹执行基础能力。
- `planners.PandaGripperPlanner`
  - 当前主力规划器；
  - 负责 Panda 机器人加载、IK、碰撞检查、OMPL/VAMP 规划、夹爪控制和 flap 任务编排；
  - 主链路最终调用 `close_double_flap()`。
- `planners.KukaOmplPlanner`
  - 保留较多旧实现；
  - 可作为历史参考，但不是 `main.py` 当前支持的主路径。

## 6) 稳定路径 vs 实验路径

建议代理按下面的优先级理解代码：

- 稳定优先：
  - `main.py`
  - `config/defaults.json`
  - `scene/`
  - `models/foldable_box.py`
  - `planners/grip_planner.py`

- 实验/非规范入口：
  - `test_env.py`
  - `models/mailer_box.py`
  - `models/mailer_box_101.py`
  - `exp/`
  - `temp/`
  - `trial.py`

实验路径可以用来补上下文，但不应默认当成“正确行为”的定义来源。

## 7) 代理协作规则

- 修改或排查主流程时，默认先验证 `python main.py --gui --mode "4-flap box task" --robot panda`。
- 不要把 `test_env.py` 当成回归测试或行为基准；它包含实验逻辑。
- 看到程序“卡住”时，先确认是否只是进入了 `main.py` 末尾的无限仿真循环。
- 改 CLI 或配置逻辑前，先核对 `main.py` 中 CLI 参数与 `config/defaults.json` 的优先级关系。
- 需要理解箱体几何时，优先看 `FoldableBox.get_flap_keypoint_pose()`，不要先从实验脚本里的可视化逻辑倒推。
- 需要判断某条路径是否是正式入口时，以 `main.py` 是否直接调用为准，而不是以文件是否存在为准。

## 8) 当前已知缺口

- 仓库内仍没有 `requirements.txt` 或 `pyproject.toml`，环境需要手工准备。
- 代码里直接依赖 `pybullet`、`pybullet_data`、`numpy`、`ompl`、`vamp`。
- `config/defaults.json` 里的 `robot` 字段当前基本不生效，因为 `main.py` 的 `--robot` 默认值是 `"panda"`，会覆盖配置回退逻辑。
- `main.py` 虽然暴露了 `"mailer box task"`，但该分支当前没有完整任务实现。
- `planners/grip_planner.py` 仍包含明显的实验/调试痕迹，例如较多打印、调试残留与重复定义。
- `planners/suck_planner.py` 仍保留旧路径代码，但不要假设它和当前主链路等价。
