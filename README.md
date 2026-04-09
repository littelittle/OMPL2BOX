# box

PyBullet robot-manipulation demos for Panda-based box interaction tasks.

The maintained execution path is centered on `main.py`. The current default demo is `MailerBoxTask`, which sets up a Panda robot in PyBullet and runs a mailer-box interaction pipeline. `FlapBoxTask` is still supported, but the active path in this repo is effectively Panda-only.

## What This Repo Runs

The canonical flow in `main.py` is:

1. Parse CLI arguments.
2. Load a JSON config.
3. Apply runtime overrides for mode, GUI, planning method, and mailer-box parameters.
4. Create a PyBullet simulation.
5. Instantiate `FlapBoxTask` or `MailerBoxTask`.
6. Call `task.setup_scene()`.
7. Call `task.run()`.

Current repo assumptions:

- `main.py` is the only canonical entry point.
- Supported mode names are Python task class names: `FlapBoxTask` and `MailerBoxTask`.
- `MailerBoxTask` is the default path.
- The active execution path is Panda-only.
- The old natural-language CLI modes are gone.
- The infinite post-run simulation loop is currently disabled; the process exits after `task.run()`.

## Setup

Using Conda:

```bash
conda env create -f environment.yml
conda activate box
```

Or with an existing Python 3.10 environment:

```bash
pip install -r requirements.txt
```

Core runtime dependencies:

- `pybullet`
- `numpy`
- `matplotlib`
- `ompl`
- `vamp`
Note: vamp is optional (just comment out all the import vamp and vamp related stuff), and is a variant of the original vamp(with weighted cost)

## Quick Start

Run the default mailer-box task from the repository root:

```bash
conda activate box
python main.py --nogui
```

Useful explicit variants:

```bash
python main.py 
python main.py --nogui
python main.py --box_yaw 20 --box_cloased
```

## CLI Arguments

`main.py` currently supports:

- `--config`: path to JSON config file. Default: `config/MailerBoxTask.json`
- `--mode`: `FlapBoxTask` or `MailerBoxTask`
- `--gui` / `--nogui`
- `--method`: `Sampling`, `Iteration`, or `RRT`
- `--box_scaling`: float override for mailer-box scale
- `--box_pos`: list-like override for mailer-box position
- `--box_yaw`: float override for mailer-box yaw(degree)
- `--box_file_path`: path to the mailer-box URDF
- `--box_closed` / `--box_open`

## Configuration

Relevant config files:

- `config/MailerBoxTask.json`: default config used by `main.py`
- `config/defaults.json`: flap-box oriented config

Current default mailer config:

```json
{
  "mode": "MailerBoxTask",
  "robot": "panda",
  "gui": true,
  "box_pos": [0.6, 0.1, 0.4],
  "box_yaw": 0.0,
  "box_closed": true,
  "box_scaling": 1.0,
  "method": "Iteration",
  "box_file_path": "assets/101/mailerbox_simple_viewer_safe_flap_closed_lid.urdf"
}
```

Common fields used by the current task code:

- Shared/runtime: `mode`, `gui`
- Mailer box task: `robot`, `box_pos`, `box_yaw`, `box_closed`, `box_scaling`, `method`, `box_file_path`
- Flap box task: `robot`, `foldable_box_pos`, `foldable_box_orn`, `pedestal_h`
- Optional physics block: `pybullet.gravity`, `pybullet.time_step`, `pybullet.num_sub_steps`, `pybullet.num_solver_iterations`

## Repository Layout

```text
.
├── main.py
├── config/
│   ├── defaults.json
│   └── MailerBoxTask.json
├── tasks/
│   ├── Task.py
│   ├── FlapBoxTask.py
│   └── MailerBoxTask.py
├── scene/
│   ├── sim_context.py
│   └── build_stuff.py
├── models/
│   ├── __init__.py
│   ├── foldable_box.py
│   ├── mailer_box.py
│   └── mailer_box_101.py
├── planners/
│   ├── generic_planner.py
│   ├── grip_planner.py
│   └── suck_planner.py
├── utils/
├── assets/
├── scripts/
├── test_env.py
├── trial.py
└── notes_refactor.md
```

Source-of-truth notes:

- `models/__init__.py` currently exports `MailerBox` from `models/mailer_box_101.py`.
- `FlapBoxTask` explicitly rejects non-Panda robots.
- `MailerBoxTask` directly constructs `PandaGripperPlanner`, so it should also be treated as Panda-only.

## Non-Canonical Paths

These files exist, but should not be treated as the maintained top-level execution path:

- `test_env.py`: experimental/debug script; still provides helpers imported by `MailerBoxTask`
- `trial.py`: one-off PyBullet contact-force experiment
- `templete.py`: scratch script with stale imports
- `models/mailer_box.py`: older mailer-box implementation
- `planners/suck_planner.py`: legacy KUKA / OMPL path not wired into `main.py`

## Notes

- The default demo path is `MailerBoxTask`.
- If you are trying to understand behavior, start from `main.py`, then trace into `tasks/`, `scene/`, `models/`, and `planners/`.
- For the exported mailer-box model, use `models/__init__.py` as the source of truth.
