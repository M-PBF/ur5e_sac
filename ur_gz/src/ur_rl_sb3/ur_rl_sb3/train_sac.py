import argparse
import os

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from ur_rl_sb3.joint_goal_env import URJointGoalEnv


class EpisodeEndLogger(BaseCallback):
    def __init__(self) -> None:
        super().__init__()
        self.episodes = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for done, info in zip(dones, infos):
            if not done:
                continue

            self.episodes += 1
            end_reason = info.get("episode_end_reason", "unknown")
            error_norm = float(info.get("error_norm", float("nan")))
            episode = info.get("episode", {})
            ep_reward = episode.get("r", float("nan"))
            ep_length = episode.get("l", -1)

            print(
                f"[Episode {self.episodes}] reason={end_reason}, final_error={error_norm:.4f}, "
                f"ep_reward={ep_reward:.3f}, ep_len={ep_length}"
            )

        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC policy for UR joint-goal reaching.")
    parser.add_argument("--timesteps", type=int, default=150_000)
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--log-dir", type=str, default="tb_logs")
    parser.add_argument("--step-dt", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--action-scale", type=float, default=0.08)
    parser.add_argument("--goal-tolerance", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = URJointGoalEnv(
        step_dt=args.step_dt,
        max_steps=args.max_steps,
        action_scale=args.action_scale,
        goal_tolerance=args.goal_tolerance,
    )
    env = Monitor(env)

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        verbose=1,
        tensorboard_log=args.log_dir,
    )

    try:
        model.learn(total_timesteps=args.timesteps, callback=EpisodeEndLogger())
    finally:
        model_path = os.path.join(args.model_dir, "ur_joint_reach_sac")
        model.save(model_path)
        env.close()
        print(f"Saved model to: {model_path}.zip")


if __name__ == "__main__":
    main()
