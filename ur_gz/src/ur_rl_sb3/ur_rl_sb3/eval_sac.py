import argparse
import numpy as np

from stable_baselines3 import SAC

from ur_rl_sb3.joint_goal_env import URJointGoalEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained SAC policy.")
    parser.add_argument("--model-path", type=str, default="models/ur_joint_reach_sac.zip")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--step-dt", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--action-scale", type=float, default=0.08)
    parser.add_argument("--goal-tolerance", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = URJointGoalEnv(
        step_dt=args.step_dt,
        max_steps=args.max_steps,
        action_scale=args.action_scale,
        goal_tolerance=args.goal_tolerance,
    )
    model = SAC.load(args.model_path)

    successes = 0
    final_errors = []

    try:
        for ep in range(args.episodes):
            obs, _ = env.reset()
            terminated = False
            truncated = False
            info = {}

            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)

            final_error = info.get("error_norm", np.nan)
            end_reason = info.get("episode_end_reason", "unknown")
            success = bool(terminated)
            successes += int(success)
            final_errors.append(final_error)
            print(
                f"Episode {ep + 1}: reason={end_reason}, success={success}, final_error={final_error:.4f}"
            )
    finally:
        env.close()

    success_rate = successes / max(args.episodes, 1)
    mean_error = float(np.nanmean(final_errors)) if final_errors else float("nan")
    print(f"Success rate: {success_rate * 100:.1f}%")
    print(f"Mean final error: {mean_error:.4f}")


if __name__ == "__main__":
    main()
