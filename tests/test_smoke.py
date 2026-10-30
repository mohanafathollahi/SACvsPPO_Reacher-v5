"""Optional smoke test: python -m unittest discover -s tests."""
import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import make_env, rollout
from stable_baselines3 import PPO


class SmokeTest(unittest.TestCase):
    def test_environment_and_rollout(self):
        env = make_env()
        try:
            obs, info = env.reset(seed=123)
            self.assertEqual(obs.shape, (10,))
            self.assertEqual(env.action_space.shape, (2,))
            model = PPO("MlpPolicy", env, device="cpu", n_steps=64, batch_size=32,
                        policy_kwargs=dict(net_arch=[64, 64]), verbose=0)
            metrics = rollout(model, episode_seed=20000)
            self.assertEqual(set(metrics), {"return", "mean_distance", "final_distance", "mean_control_cost"})
            self.assertGreaterEqual(metrics["mean_distance"], 0)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
