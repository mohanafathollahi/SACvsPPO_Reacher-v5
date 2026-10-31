"""Create comparison plots and descriptive final-performance table."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = ["return", "mean_distance", "final_distance", "mean_control_cost"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("."))
    args = p.parse_args()
    results = args.out / "results"
    plots = results / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(results.glob("*_seed*/checkpoints.csv"))
    if not checkpoints:
        p.error("No checkpoints found; run scripts/train.py first")
    df = pd.concat([pd.read_csv(f) for f in checkpoints], ignore_index=True)
    # First average episodes within each training seed; then average training seeds.
    per_seed = df.groupby(["algorithm", "seed", "step"], as_index=False)[METRICS].mean()
    for metric, ylabel in [("return", "Mean episode return"),
                          ("mean_distance", "Mean fingertip distance"),
                          ("mean_control_cost", "Mean control cost")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for alg, group in per_seed.groupby("algorithm"):
            agg = group.groupby("step")[metric].agg(["mean", "std"]).sort_index()
            ax.plot(agg.index, agg["mean"], label=alg.upper())
            if group["seed"].nunique() > 1:
                sd = agg["std"].fillna(0)
                ax.fill_between(agg.index, agg["mean"] - sd, agg["mean"] + sd,
                                alpha=0.2)
        ax.set(xlabel="Environment training steps", ylabel=ylabel)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots / f"learning_{metric}.png", dpi=160)
        plt.close(fig)

    final = results / "final_evaluation.csv"
    if final.exists():
        ev = pd.read_csv(final)
        per_training_seed = ev.groupby(["algorithm", "seed"], as_index=False)[METRICS].mean()
        summaries = per_training_seed.groupby("algorithm")[METRICS].agg(["mean", "std"])
        summaries.to_csv(results / "final_summary.csv")
        print("Final evaluation (mean ± SD across training seeds; NaN SD if one seed):")
        print(summaries.to_string())
        fig, ax = plt.subplots(figsize=(7, 5))
        means = per_training_seed.groupby("algorithm")["return"].mean()
        errs = per_training_seed.groupby("algorithm")["return"].std().fillna(0)
        ax.bar(means.index.str.upper(), means.values, yerr=errs.values, capsize=5)
        ax.set(ylabel="Mean held-out episode return (higher is better)",
               title="Final policy comparison")
        fig.tight_layout()
        fig.savefig(plots / "final_return.png", dpi=160)
        plt.close(fig)
    print(f"Plots saved to {plots}")


if __name__ == "__main__":
    main()
