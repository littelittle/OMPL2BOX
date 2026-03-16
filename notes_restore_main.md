# restore main.py(4-flap box) notes

## Target command
python main.py

## Run 1
Error:
the specified grasp pose is not correct
something wrong with: planner.prim_acquire_pinch()
approach_dist is too large

Solution:
tune approach_dist to 0(*The differences between the previous and current versions are still unclear, which is why `approach_dist` must be 0 for the current version to function correctly.*)