"""Train PPO and/or SAC, saving models and checkpoint evaluation CSVs."""
import argparse
from pathlib import Path
import random
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from common import make_env, CheckpointEvaluator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alg", choices=["ppo", "sac", "both"], default="both")
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--eval-freq", type=int, default=5_000)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--out", type=Path, default=Path("."))
    args = p.parse_args()
    if args.steps <= 0 or args.eval_freq <= 0:
        p.error("--steps and --eval-freq must be positive")
    algs = ["ppo", "sac"] if args.alg == "both" else [args.alg]
    for alg in algs:
        for seed in args.seeds:
            np.random.seed(seed)
            random.seed(seed)
            folder = args.out / "results" / f"{alg}_seed{seed}"
            folder.mkdir(parents=True, exist_ok=True)
            env = Monitor(make_env(), filename=str(folder / "train_monitor.csv"))
            env.reset(seed=seed)
            env.action_space.seed(seed)
            cls = PPO if alg == "ppo" else SAC
            # Keep the same policy architecture; algorithm defaults otherwise differ.
            model = cls("MlpPolicy", env, seed=seed, device="cpu", verbose=0,
                        policy_kwargs=dict(net_arch=[64, 64]))
            callback = CheckpointEvaluator(alg, seed, args.eval_freq,
                                            folder / "checkpoints.csv")
            print(f"Training {alg.upper()} seed={seed} for {args.steps} steps (CPU)")
            try:
                model.learn(total_timesteps=args.steps, callback=callback)
                dest = args.out / "models" / f"{alg}_seed{seed}"
                dest.parent.mkdir(parents=True, exist_ok=True)
                model.save(str(dest))
                print(f"Saved {dest}.zip, actual training steps: {model.num_timesteps}")
            finally:
                env.close()


if __name__ == "__main__":
    main()
