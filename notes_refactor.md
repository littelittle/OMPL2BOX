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

<br><br>

# refactor test_env.py notes

## Traget command
python main.py (merge what's in test_env to main.py entry point)

## Step 1
dig into main.py

# refactor GenericPlanner&PandaGripperPlanner notes

## _quat_from_normal_and_yaw
- for yaw search
- [x] disabled from the GenericPlanner's method(raise RuntimeError)
- [x] moved to the utils/vector.py

## sample_redundant
- disabled from the GenericPlanner's method
- moved to the task_constraint_planner


<br><br>


# A Glimpse of Panda
(0, b'panda_joint1', 0, 7, 6, 1, 0.0, 0.0, -2.9671, 2.9671, 87.0, 2.175, b'panda_link1', (0.0, 0.0, 1.0), (0.0, 0.0, 0.28300000000000003), (0.0, 0.0, 0.0, 1.0), -1)

(1, b'panda_joint2', 0, 8, 7, 1, 0.0, 0.0, -1.8326, 1.8326, 87.0, 2.175, b'panda_link2', (0.0, 0.0, 1.0), (0.0, 0.04, 0.05), (0.7071067811848163, 0.0, 0.0, 0.7071067811882787), 0)

(2, b'panda_joint3', 0, 9, 8, 1, 0.0, 0.0, -2.9671, 2.9671, 87.0, 2.175, b'panda_link3', (0.0, 0.0, 1.0), (0.0, -0.276, -0.06), (-0.7071067811848163, 0.0, 0.0, 0.7071067811882787), 1)

(3, b'panda_joint4', 0, 10, 9, 1, 0.0, 0.0, -3.1416, 0.0, 87.0, 2.175, b'panda_link4', (0.0, 0.0, 1.0), (0.07250000000000001, -0.01, 0.05), (-0.7071067811848163, 0.0, 0.0, 0.7071067811882787), 2)

(4, b'panda_joint5', 0, 11, 10, 1, 0.0, 0.0, -2.9671, 2.9671, 12.0, 2.61, b'panda_link5', (0.0, 0.0, 1.0), (-0.052500000000000005, 0.354, -0.02), (0.7071067811848163, 0.0, 0.0, 0.7071067811882787), 3)

(5, b'panda_joint6', 0, 12, 11, 1, 0.0, 0.0, -0.0873, 3.8223, 12.0, 2.61, b'panda_link6', (0.0, 0.0, 1.0), (0.0, -0.04, 0.12), (-0.7071067811848163, 0.0, 0.0, 0.7071067811882787), 4)

(6, b'panda_joint7', 0, 13, 12, 1, 0.0, 0.0, -2.9671, 2.9671, 12.0, 2.61, b'panda_link7', (0.0, 0.0, 1.0), (0.047999999999999994, 0.0, 0.0), (-0.7071067811848163, 0.0, 0.0, 0.7071067811882787), 5)

(7, b'panda_joint8', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'panda_link8', (0.0, 0.0, 0.0), (0.0, 0.0, 0.026999999999999996), (0.0, 0.0, 0.0, 1.0), 6)

(8, b'panda_hand_joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'panda_hand', (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.38268343236488267, 0.9238795325113726), 7)

(9, b'panda_finger_joint1', 1, 14, 13, 1, 0.0, 0.0, 0.0, 0.04, 20.0, 0.2, b'panda_leftfinger', (0.0, 1.0, 0.0), (0.0, 0.0, 0.0184), (0.0, 0.0, 0.0, 1.0), 8)

(10, b'panda_finger_joint2', 1, 15, 14, 1, 0.0, 0.0, 0.0, 0.04, 20.0, 0.2, b'panda_rightfinger', (0.0, -1.0, 0.0), (0.0, 0.0, 0.0184), (0.0, 0.0, 0.0, 1.0), 8)

(11, b'panda_grasptarget_hand', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'panda_grasptarget', (0.0, 0.0, 0.0), (0.0, 0.0, 0.065), (0.0, 0.0, 0.0, 1.0), 8)