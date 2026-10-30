"""Shared setup, checkpoint evaluation and metrics for Reacher-v5."""
from pathlib import Path
import csv
import numpy as np
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback

ENV_ID = "Reacher-v5"
EVAL_SEEDS = (1000, 1001, 1002, 1003, 1004)
FIELDS = ("algorithm", "seed", "step", "episode", "return", "mean_distance", "final_distance", "mean_control_cost")


def make_env(render_mode=None):
    return gym.make(ENV_ID, render_mode=render_mode)


def rollout(model, episode_seed, render=False):
    """Evaluate one episode using a deterministic policy and fixed reset seed.

    reward_dist == -fingertip distance under default Reacher-v5 weights.
    reward_ctrl == -0.1 * squared action norm under default weights.
    """
    env = make_env(render_mode="human" if render else None)
    try:
        obs, _ = env.reset(seed=int(episode_seed))
        total, distances, control_costs = 0.0, [], []
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            distances.append(-float(info["reward_dist"]))
            control_costs.append(-float(info["reward_ctrl"]))
            done = terminated or truncated
        return {"return": total,
                "mean_distance": float(np.mean(distances)),
                "final_distance": distances[-1],
                "mean_control_cost": float(np.mean(control_costs))}
    finally:
        env.close()


def evaluate_rows(model, algorithm, seed, step, eval_seeds=EVAL_SEEDS):
    return [dict(algorithm=algorithm, seed=seed, step=step, episode=i,
                 **rollout(model, eval_seed))
            for i, eval_seed in enumerate(eval_seeds)]


def append_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if f.tell() == 0:
            writer.writeheader()
        writer.writerows(rows)


class CheckpointEvaluator(BaseCallback):
    """Evaluate checkpoints every `frequency` actual env steps (single env)."""
    def __init__(self, algorithm, seed, frequency, output_csv):
        super().__init__()
        self.algorithm, self.seed = algorithm, seed
        self.frequency, self.output_csv = frequency, output_csv

    def _on_step(self):
        if self.num_timesteps % self.frequency == 0:
            rows = evaluate_rows(self.model, self.algorithm, self.seed, self.num_timesteps)
            append_csv(self.output_csv, rows)
            avg = np.mean([r["return"] for r in rows])
            print(f"[{self.algorithm} seed={self.seed}] {self.num_timesteps} steps, eval return {avg:.2f}")
        return True
