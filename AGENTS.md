Repo Agent Guide

1) Repository Positioning

This is a PyBullet robot manipulation repository centered on Panda-based box interaction tasks.

The canonical execution flow lives in `main.py` and currently does the following:

1. Parse CLI arguments.
2. Load a JSON config file.
3. Resolve runtime overrides for mode, GUI, planning method, box scaling, box position, and box yaw.
4. Create a PyBullet simulation via `scene.make_sim(...)`.
5. Select a task class from `tasks.FlapBoxTask` / `tasks.MailerBoxTask`.
6. Call `task.setup_scene()`.
7. Call `task.run()`.

Important current-state notes:

- `main.py` is the only canonical entry point.
- The old natural-language CLI modes are gone. The supported mode names are now Python class names: `FlapBoxTask` and `MailerBoxTask`.
- But mainly focus on MailerBoxTask right now!
- `main.py` no longer exposes a `--robot` CLI flag.
- The post-task infinite simulation loop is currently commented out. The program returns after `task.run()` finishes.
- The active main path is effectively Panda-only even though legacy KUKA code still exists in the repo.

2) Standard Execution

Run from the repository root:

```bash
conda activate box
python main.py --nogui
```

Useful explicit variants:

```bash
python main.py --config config/defaults.json --mode FlapBoxTask --nogui
python main.py --config config/MailerBoxTask.json --mode MailerBoxTask --nogui
```

Currently supported CLI arguments in `main.py`:

- `--config`: path to JSON config. Current default is `config/MailerBoxTask.json`.
- `--mode`: `FlapBoxTask` or `MailerBoxTask`.
- `--gui` / `--nogui`
- `--method`: `Sampling` or `Iteration`
- `--scaling`: float
- `--box_pos`: list-like override for mailer box position
- `--box_yaw`: float

Behavior that matters when reading or editing the code:

- `FlapBoxTask` is a real main path.
- `MailerBoxTask` is a real main path and is the current default config path.
- Both tasks currently instantiate `PandaGripperPlanner`.
- `FlapBoxTask` explicitly raises `NotImplementedError` if config `robot != "panda"`.
- `MailerBoxTask` does not branch on robot type in practice; it directly constructs the Panda planner, so treat it as Panda-only.
- `KukaOmplPlanner` still exists under `planners/suck_planner.py`, but it is not wired into `main.py`.

3) Configuration

Active configs:

- `config/defaults.json`: default flap-box config, with `mode = "FlapBoxTask"`.
- `config/MailerBoxTask.json`: default mailer-box config used by `main.py` unless overridden.

Relevant config fields used by current task code:

- Shared/runtime: `mode`, `gui`
- Flap box task: `robot`, `foldable_box_pos`, `foldable_box_orn`, `pedestal_h`
- Mailer box task: `robot`, `box_pos`, `box_yaw`, `closed`, `scaling`, `method`, `file_path`
- Optional physics block: `pybullet.gravity`, `pybullet.time_step`, `pybullet.num_sub_steps`, `pybullet.num_solver_iterations`

4) Repository Structure

```text
.
├── main.py                     # Canonical entry point
├── AGENTS.md                   # This guide
├── config/
│   ├── defaults.json           # FlapBoxTask-oriented config
│   └── MailerBoxTask.json      # Current default config for main.py
├── tasks/
│   ├── Task.py                 # Minimal base task
│   ├── FlapBoxTask.py          # Foldable 4-flap box task
│   └── MailerBoxTask.py        # Mailer box task, current default path
├── scene/
│   ├── sim_context.py          # SimContext + physics setup
│   └── build_stuff.py          # Scene helpers such as pedestal creation
├── models/
│   ├── foldable_box.py         # Foldable box model + geometric oracle
│   ├── mailer_box_101.py       # Current exported MailerBox implementation
│   └── mailer_box.py           # Older experimental mailer box model
├── planners/
│   ├── generic_planner.py      # Shared planner utilities
│   ├── grip_planner.py         # Panda gripper planner
│   └── suck_planner.py         # Legacy KUKA / OMPL planner path
├── utils/                      # Vectors, paths, yaw search, plotting helpers, etc.
├── assets/
│   ├── foldable_box*.urdf      # Foldable box assets
│   ├── 101/                    # Mailer box asset set used by current config
│   └── 103/                    # Additional mailer box asset set
├── test_env.py                 # Experimental script; also provides helpers imported by MailerBoxTask
├── trial.py                    # Standalone PyBullet contact-force experiment
├── templete.py                 # Scratch script with stale imports
├── exp/                        # Experiment outputs and figures
├── temp/                       # Temporary outputs
├── data/                       # Local generated data / artifacts
└── notes_refactor.md           # Refactor notes
```

5) Non-Canonical Paths

The following should not be treated as production-ready entry points:

- `test_env.py`: experimental/debug-oriented script. Important nuance: `MailerBoxTask.py` imports helper functions from here, so it is not dead code, but it is still not the canonical top-level entry point.
- `trial.py`: one-off PyBullet experiment for gripper contact-force inspection.
- `templete.py`: scratch script with stale imports; not part of the maintained execution path.
- `exp/`, `temp/`, `data/`: local outputs, figures, and artifacts, not stable source-of-truth code paths.
- `models/mailer_box.py`: older mailer-box implementation; the exported `models.MailerBox` currently comes from `models/mailer_box_101.py`.
- `planners/suck_planner.py`: legacy KUKA/OMPL path present in the repo but not connected to `main.py`.

6) Agent Working Guidance

When making changes in this repo, prefer these assumptions unless the code proves otherwise:

- Start from `main.py`, then trace into `tasks/`, then `scene/`, `models/`, and `planners/`.
- Treat Panda as the supported robot in the active execution path.
- Treat `MailerBoxTask` as the default demo because `main.py` defaults to `config/MailerBoxTask.json`.
- Do not describe `test_env.py`, `trial.py`, `templete.py`, `exp/`, or `temp/` as regression entry points.
- If you need to reason about the mailer box model exposed to the rest of the app, use `models/__init__.py` as the source of truth for which implementation is exported.
