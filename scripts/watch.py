"""Watch a trained policy; run on a machine with a graphical display."""
import argparse
from pathlib import Path
from stable_baselines3 import PPO, SAC
from common import rollout


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alg", choices=["ppo", "sac"], default="ppo")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("."))
    args = p.parse_args()
    cls = PPO if args.alg == "ppo" else SAC
    path = args.out / "models" / f"{args.alg}_seed{args.seed}.zip"
    model = cls.load(str(path), device="cpu")
    print(rollout(model, episode_seed=42_000, render=True))


if __name__ == "__main__":
    main()
