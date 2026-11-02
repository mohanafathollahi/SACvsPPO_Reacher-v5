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

        # Increase font size of x and y axis labels
        ax.set_xlabel("Environment training steps", fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)

        # Increase font size of axis numbers (tick labels)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=16)
        fig.tight_layout()
        fig.savefig(plots / f"learning_{metric}.png", dpi=160)
        plt.close(fig)

    print(f"Plots saved to {plots}")


if __name__ == "__main__":
    main()
