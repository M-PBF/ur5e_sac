# ur_rl_sb3

A resume-oriented RL extension package for `ur_simulation_gz`, using Stable-Baselines3 to train a policy for UR joint-goal reaching in Gazebo.

## What this package adds

- A `Gymnasium` environment: `URJointGoalEnv`
- Continuous control training script with `SAC`
- Evaluation script with success-rate and final-error metrics
- A launch file that starts UR simulation with `forward_position_controller`
- RViz goal marker publishing on `/ur_rl_sb3/goal_marker`
- Per-episode end reason logging (`goal_reached` / `max_steps`)

## 中文教程（零基础到面试）

- `docs/00_学习路径与总览.md`
- `docs/01_ROS2_零基础入门.md`
- `docs/02_项目架构与RL原理.md`
- `docs/03_从零跑通训练与可视化.md`
- `docs/04_实习面试高频问答.md`

## Dependencies

In addition to ROS 2 dependencies, install Python RL dependencies:

```bash
pip install gymnasium stable-baselines3
```

## Build

```bash
cd /home/bophy/d2lros2/ur5/ur_gz
colcon build --symlink-install
source install/setup.bash
```

## Run simulation (RL-friendly controller)

```bash
ros2 launch ur_rl_sb3 ur_rl_sim.launch.py
```

## Train

In another terminal (after sourcing workspace):

```bash
ros2 run ur_rl_sb3 train_sac --timesteps 150000
```

Model is saved to `models/ur_joint_reach_sac.zip` by default.

During training, each episode end prints a summary like:

```text
[Episode 12] reason=max_steps, final_error=0.1842, ep_reward=-31.553, ep_len=200
```

## Evaluate

```bash
ros2 run ur_rl_sb3 eval_sac --model-path models/ur_joint_reach_sac.zip --episodes 20
```

Evaluation also prints each episode end reason and final error.

## Visualize random target in RViz

If you launch with RViz enabled, add a `Marker` display and set topic to:

```text
/ur_rl_sb3/goal_marker
```

You will see the random target point as a red sphere in `base_link` frame.

## Suggested resume description

- Designed and implemented a ROS2-Gazebo reinforcement learning pipeline for UR5, wrapping simulator control into a custom Gymnasium environment.
- Trained continuous-control policies with Stable-Baselines3 (SAC), and evaluated policy quality via success rate and final joint-error metrics.
- Built reproducible training/evaluation tooling and launch configuration for headless simulation experiments.
