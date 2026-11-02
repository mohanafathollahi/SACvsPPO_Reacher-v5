import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load final evaluation results
df = pd.read_csv("results/final_evaluation.csv")

output_dir = Path("results/plots")
output_dir.mkdir(parents=True, exist_ok=True)

# One row per independently trained model
model_results = (
    df.groupby(["algorithm", "seed"], as_index=False)
    .agg(
        **{
            "return": ("return", "mean"),
            "mean_distance": ("mean_distance", "mean"),
            "final_distance": ("final_distance", "mean"),
            "mean_control_cost": ("mean_control_cost", "mean"),
        }
    )
)

metrics = {
    "return": "Average Episode Return",
    "mean_distance": "Mean Distance to Target",
    "final_distance": "Final Distance to Target",
    "mean_control_cost": "Mean Control Cost",
}

colors = {
    "ppo": "royalblue",
    "sac": "darkorange",
}

# Plot 1: Individual seed results for each metric
for metric, ylabel in metrics.items():

    fig, ax = plt.subplots(figsize=(8, 5))

    for x, alg in enumerate(["ppo", "sac"]):

        subset = model_results[
            model_results["algorithm"].str.lower() == alg
        ].sort_values("seed")

        values = subset[metric].to_numpy()

        # Slight horizontal offsets keep points visible
        offsets = np.linspace(-0.08, 0.08, len(values))

        ax.scatter(
            x + offsets,
            values,
            s=100,
            color=colors[alg],
            label=alg.upper(),
            zorder=3,
        )

        # Label each point with its training seed
        for offset, (_, row) in zip(offsets, subset.iterrows()):
            ax.annotate(
                f"Seed {int(row['seed'])}",
                (x + offset, row[metric]),
                xytext=(5, 7),
                textcoords="offset points",
                fontsize=8,
            )

        # Mean across trained models
        mean_value = values.mean()

        ax.hlines(
            mean_value,
            x - 0.22,
            x + 0.22,
            color="black",
            linewidth=2,
            label="Mean" if x == 0 else None,
        )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["PPO", "SAC"])

    ax.set_ylabel(ylabel)
    ax.set_title(f"PPO vs SAC — {ylabel}")

    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / f"scatter_{metric}.png", dpi=160)
    plt.close(fig)


# Plot 2: Reaching accuracy vs control effort
fig, ax = plt.subplots(figsize=(8, 6))

for alg in ["ppo", "sac"]:

    subset = model_results[
        model_results["algorithm"].str.lower() == alg
    ]

    ax.scatter(
        subset["mean_distance"],
        subset["mean_control_cost"],
        s=120,
        label=alg.upper(),
        color=colors[alg],
    )

    for _, row in subset.iterrows():
        ax.annotate(
            f"Seed {int(row['seed'])}",
            (row["mean_distance"], row["mean_control_cost"]),
            xytext=(6, 6),
            textcoords="offset points",
        )

ax.set_xlabel("Mean Distance to Target (lower is better)")
ax.set_ylabel("Mean Control Cost (lower is better)")

ax.set_title("PPO vs SAC — Reaching Accuracy vs Control Cost")

ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(output_dir / "distance_vs_control.png", dpi=160)
plt.close(fig)