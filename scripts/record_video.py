import argparse
from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO, SAC


def record_video(algorithm, model_path, seed=42, episodes=3):
    output_dir = Path("videos") / algorithm.lower()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create Reacher-v5 with image rendering enabled.
    env = gym.make(
        "Reacher-v5",
        render_mode="rgb_array",
        width=640,
        height=480,
    )

    # Record every episode.
    env = RecordVideo(
        env,
        video_folder=str(output_dir),
        episode_trigger=lambda episode: True,
        name_prefix=algorithm.lower(),
    )

    # Load the trained policy.
    model_class = PPO if algorithm.lower() == "ppo" else SAC
    model = model_class.load(model_path, device="cpu")

    for episode in range(episodes):
        # Same seed for PPO and SAC = same initial conditions.
        obs, info = env.reset(seed=seed + episode)

        terminated = False
        truncated = False
        episode_reward = 0.0

        while not (terminated or truncated):
            action, _ = model.predict(
                obs,
                deterministic=True,
            )

            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward

        print(
            f"{algorithm.upper()} | "
            f"Episode {episode + 1} | "
            f"Reward: {episode_reward:.3f}"
        )

    env.close()
    print(f"Videos saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--alg",
        choices=["ppo", "sac"],
        required=True,
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained model ZIP",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
    )

    args = parser.parse_args()

    record_video(
        algorithm=args.alg,
        model_path=args.model,
        seed=args.seed,
        episodes=args.episodes,
    )