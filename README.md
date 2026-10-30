# PPO vs SAC on MuJoCo Reacher-v5 🤖

A small, CPU-friendly, **reproducible learning project** comparing two reinforcement-learning approaches to controlling a two-jointed robotic arm. It trains **PPO (on-policy)** and **SAC (off-policy)** with [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) in [Gymnasium's Reacher-v5](https://gymnasium.farama.org/environments/mujoco/reacher/).

> **Project status:** Runnable experiment code and analysis workflow are included. **No benchmark results are fabricated or precomputed.** Run the commands below on your laptop to generate actual plots and numerical results.

## 1. Research question

Given the same number of **environment interactions**, how do PPO and SAC compare at reaching a target in Reacher-v5?

We compare:

- **Learning progress:** average evaluation return vs. environment steps.
- **Reaching precision:** mean and final fingertip-to-target distance (lower is better).
- **Control effort:** mean penalized torque cost (lower means less action magnitude).
- **Variability:** results across independent training seeds, each evaluated on identical episode-reset seeds.

Training on CPU is supported; **no dedicated GPU or physical robot is required**.

## 2. The robotics / RL problem

Reacher-v5 models a **two-joint arm**. Each policy observes a **10-element observation** describing the arm's state and target geometry, and outputs **2 bounded joint-torque commands** in `[-1, 1]`. MuJoCo applies the physics and returns a reward. The default episode lasts **50 simulation steps** (time-limit truncation).

The default reward is:

\[
r_t = -\|p_\mathrm{finger} - p_\mathrm{target}\|_2 - 0.1\|a_t\|_2^2.
\]

The first term rewards reaching; the second discourages unnecessarily high torque. By default, `info["reward_dist"]` contains the **negative distance** and `info["reward_ctrl"]` contains the **negative weighted control cost**. We store their magnitudes as positive distance/cost metrics.

**Important:** returns are typically negative; a *higher* return (closer to zero) is better. Lower distance and lower control cost are desirable, but may trade off.

## 3. PPO vs SAC: conceptual comparison

| Property | PPO | SAC |
|---|---|---|
| Data collection | On-policy | Off-policy with replay buffer |
| Policy optimization | Clipped policy-gradient objective | Entropy-regularized actor-critic |
| Experience reuse | Limited (latest rollout) | Reuses past transitions |
| Continuous actions | Yes | Yes |
| Memory requirement | Relatively modest | Replay buffer costs more RAM |
| Main question | Can it learn robust reaching? | Does reusing transitions help at the same interaction budget? |

**Fair-comparison scope:** same task, reward, network hidden-layer sizes (`[64, 64]`), device (`cpu`), training-step budget and evaluation seeds. **Not identical optimization effort:** each algorithm retains its own Stable-Baselines3 defaults for batch sizes, update counts, entropy parameters, and learning rates. SAC may take more wall-clock time; a fixed interaction budget does *not* imply equal compute. Report both if measuring runtime. PPO can slightly exceed the requested step budget because it collects full rollouts; use the recorded `step` values when interpreting plots.

## 4. Repository structure

```text
reacher-ppo-vs-sac/
├── README.md                  # Walkthrough and analysis instructions
├── requirements.txt           # Python dependencies
├── scripts/
│   ├── common.py              # Reacher env, rollouts, checkpoint evaluator
│   ├── train.py               # PPO/SAC training
│   ├── evaluate.py            # Held-out final evaluation
│   ├── plot.py                # Learning curves + final performance table
│   └── watch.py               # Visualize a saved policy
├── tests/test_smoke.py        # Quick integration smoke test
├── models/                    # Generated model ZIPs (ignored by Git)
└── results/                   # Generated CSVs and PNGs (ignored by Git)
```

## 5. Set up on your laptop

Recommended: **Python 3.10–3.12**, an up-to-date operating system, ~8 GB+ RAM. For the first run, a CPU is sufficient. MuJoCo installs through Gymnasium's extra.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Windows PowerShell

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If Windows prevents activating scripts, you can invoke `.venv\Scripts\python.exe` directly instead of activating. If MuJoCo fails to install, check your Python version and the [MuJoCo installation guide](https://mujoco.readthedocs.io/en/stable/python.html).

Verify environment and metrics are working:

```bash
python -m unittest discover -s tests -v
```

## 6. Step-by-step experiments

Run commands **from the repository root**. Training happens *without rendering* to keep it fast.

### Step A — Fast trial (single seed)

```bash
python scripts/train.py --alg both --steps 10000 --eval-freq 5000 --seeds 0
python scripts/evaluate.py --alg both --seeds 0 --episodes 10
python scripts/plot.py
```

This is a **pipeline smoke run**, **not** a reliable algorithm ranking. Check whether two `models/*.zip` files and plots in `results/plots/` appear.

### Step B — Main experiment (three seeds)

For a more meaningful comparison, clear or rename existing `results/` and `models/` outputs first (CSV checkpoints are appended when training is rerun). Then run:

```bash
python scripts/train.py --alg both --steps 200000 --eval-freq 10000 --seeds 0 1 2
python scripts/evaluate.py --alg both --seeds 0 1 2 --episodes 30
python scripts/plot.py
```

This trains six policies; on a laptop, consider starting with one seed and expanding later. The same **30 held-out evaluation seeds** (20000–20029) are applied to every final model. Periodic checkpoints use a *different fixed set* (1000–1004) so they do not coincide with final test episodes. Independent training seeds are 0, 1, 2.

**Note on checkpoints:** The callback evaluates models every `--eval-freq` steps and stores episode-level rows; it does not save intermediate model weights. Saved weights represent the **final** models. Checkpoint evaluation adds simulation interactions *outside* the training budget, equally for both algorithms; they are not used for learning.

### Step C — Watch the trained policies

Requires a local graphical desktop session:

```bash
python scripts/watch.py --alg ppo --seed 0
python scripts/watch.py --alg sac --seed 0
```

Each command plays one deterministic 50-step episode, then closes the viewer. For headless systems, omit this step.

## 7. Read the results

Generated data:

| Output | Meaning |
|---|---|
| `results/ppo_seed0/checkpoints.csv` | Per-episode checkpoint metrics for PPO seed 0 |
| `results/sac_seed0/checkpoints.csv` | Per-episode checkpoint metrics for SAC seed 0 |
| `results/*/train_monitor.csv` | Training episode lengths and returns recorded by SB3 Monitor |
| `results/final_evaluation.csv` | Per-episode held-out results for each final policy |
| `results/final_summary.csv` | Means and standard deviations **across training seeds** |
| `results/plots/learning_return.png` | Checkpoint evaluation return vs. environment steps |
| `results/plots/learning_mean_distance.png` | Precision vs. environment steps |
| `results/plots/learning_mean_control_cost.png` | Control effort vs. environment steps |
| `results/plots/final_return.png` | Final mean return with across-seed SD bars |

Each checkpoint curve first averages over evaluation episodes **within each training seed**, then averages across training seeds. Shaded areas are **±1 standard deviation across training seeds**, not confidence intervals. With only one training seed, uncertainty bands cannot be estimated. The final table similarly averages evaluation episodes within each seed, then reports the cross-seed mean and SD. Report number of seeds and episodes alongside results.

### Analysis questions to answer in your report

1. **Learning:** Which approach improves return earlier when compared at *equal environment steps*?
2. **Final performance:** Which approach has the higher held-out return after training?
3. **Precision:** Is the return improvement due to moving closer to the goal, or mainly lower control cost?
4. **Stability:** Does the ranking stay consistent across 3+ independent training seeds?
5. **Efficiency:** Does SAC's experience reuse give better *sample efficiency*, even if its wall-clock training is slower?
6. **Qualitative inspection:** Does the robot reach accurately, oscillate, or apply abrupt actions? (The current quantitative plots measure control magnitude, **not** action smoothness.)

### Fill in after running (do not guess)

| Measure | PPO | SAC | Interpretation |
|---|---|---|---|
| Final held-out mean return ↑ | _run experiment_ | _run experiment_ | Higher is better |
| Mean fingertip distance ↓ | _run experiment_ | _run experiment_ | Lower is better |
| Final fingertip distance ↓ | _run experiment_ | _run experiment_ | Lower is better |
| Mean weighted control cost ↓ | _run experiment_ | _run experiment_ | Lower is better |
| Between-seed variation | _run experiment_ | _run experiment_ | Lower variation may suggest more robust training |

**How to interpret without overclaiming:** Do not conclude that one algorithm is universally better from one random seed or one environment. Consider seed-to-seed variability, tuning effort, compute time, and interactions. An improvement in return can arise from either distance or action-penalty terms.

## 8. Suggested learning extensions

- **More seeds:** Increase to 5–10 independent training seeds for stronger evidence.
- **Hyperparameter study:** Try a small learning-rate sweep; compare each method with similar tuning effort.
- **Reward trade-off:** Change `reward_control_weight` through `gym.make` and see how reaching vs. torque changes. Keep the same weight for both algorithms in each experiment.
- **Smoothness metric:** Add a metric based on `||a_t - a_{t-1}||²` to measure jitter; it is *not* the same as control-cost magnitude.
- **Compute comparison:** Measure training wall time and peak RAM. That is a *separate* efficiency axis from sample efficiency.

## 9. Troubleshooting

- **`ModuleNotFoundError`**: Activate the correct virtual environment and reinstall requirements.
- **MuJoCo viewer fails in a remote session**: Train/evaluate without rendering; the viewer requires a display.
- **Windows/macOS window does not open**: Check MuJoCo's platform-specific viewer instructions; on some macOS configurations, launch using `mjpython` for rendering.
- **Runs seem slow**: Reduce `--steps`, use one training seed, and avoid rendering during training.
- **Results have duplicate checkpoints**: Use a fresh output directory (`--out ./experiment2`) or delete previous `results/` for a clean rerun.
- **No final summary**: Run `scripts/evaluate.py` before `scripts/plot.py`.

## References

- [Gymnasium Reacher-v5 task definition](https://gymnasium.farama.org/environments/mujoco/reacher/)
- [Stable-Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- [Stable-Baselines3 SAC](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)
- [MuJoCo documentation](https://mujoco.readthedocs.io/)

## License

MIT (see `LICENSE`).
