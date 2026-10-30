"""Evaluate saved policies on identical, held-out episode seeds."""
import argparse
from pathlib import Path
from stable_baselines3 import PPO, SAC
from common import evaluate_rows, append_csv, FIELDS
import csv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alg", choices=["ppo", "sac", "both"], default="both")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--out", type=Path, default=Path("."))
    args = p.parse_args()
    if args.episodes < 1:
        p.error("--episodes must be >= 1")
    path = args.out / "results" / "final_evaluation.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Rebuild rather than silently duplicating old evaluations.
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writeheader()
    for alg in (["ppo", "sac"] if args.alg == "both" else [args.alg]):
        for seed in args.seeds:
            model_path = args.out / "models" / f"{alg}_seed{seed}.zip"
            cls = PPO if alg == "ppo" else SAC
            model = cls.load(str(model_path), device="cpu")
            # Separate from training and checkpoint evaluation episodes.
            rows = evaluate_rows(model, alg, seed, step=model.num_timesteps,
                                 eval_seeds=range(20_000, 20_000 + args.episodes))
            append_csv(path, rows)
            mean_return = sum(r["return"] for r in rows) / len(rows)
            print(f"{alg.upper()} seed={seed}: mean return={mean_return:.3f}")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
