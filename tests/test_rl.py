import tempfile
import unittest
from pathlib import Path

from rl.config import TrainConfig
from rl.obs import ObsSpec, ObservationEncoder
from rl.rewards import rank_to_reward
from rl.trainer import RankPPOTrainer
from tronk_ml import MLTronkEnv


class RLTests(unittest.TestCase):
    def test_observation_encoder_shape(self) -> None:
        env = MLTronkEnv(seed=1, max_steps=30)
        enc = ObservationEncoder(ObsSpec(radius=4, include_threat_map=True))
        obs = enc.encode(env, 0)
        self.assertEqual(obs.shape[0], enc.obs_dim)

    def test_rank_reward_interpolation(self) -> None:
        cfg = TrainConfig().rewards
        r2 = rank_to_reward(2.0, cfg)
        r3 = rank_to_reward(3.0, cfg)
        r25 = rank_to_reward(2.5, cfg)
        self.assertAlmostEqual(r25, (r2 + r3) / 2.0, places=6)

    def test_trainer_smoke_one_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            cfg = TrainConfig(
                seed=7,
                total_updates=1,
                rollout_episodes_per_update=2,
                eval_matches=2,
                eval_interval_updates=1,
                output_dir=base / "out",
                checkpoint_dir=base / "out" / "checkpoints",
                replay_dir=base / "replays",
            )
            cfg.env.max_steps = 20
            cfg.ppo.minibatch_size = 16
            cfg.ppo.epochs = 1
            cfg.progress_replay_every_updates = 1
            cfg.progress_replay_matches = 1

            trainer = RankPPOTrainer(cfg)
            trainer.train()

            self.assertTrue((cfg.output_dir / "final_policy.pt").exists())
            self.assertTrue((cfg.output_dir / "metrics.jsonl").exists())
            self.assertTrue((cfg.replay_dir / "mlprog_u000001_m01.json").exists())

    def test_trainer_reports_single_rollout_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            cfg = TrainConfig(
                seed=9,
                total_updates=1,
                rollout_episodes_per_update=2,
                eval_matches=2,
                eval_interval_updates=10,
                output_dir=base / "out",
                checkpoint_dir=base / "out" / "checkpoints",
                replay_dir=base / "replays",
            )
            cfg.env.max_steps = 20
            cfg.env.use_c_core = True
            cfg.ppo.minibatch_size = 16
            cfg.ppo.epochs = 1
            cfg.progress_replay_every_updates = 0
            cfg.save_replay_every_updates = 0
            cfg.rewards.turn_streak_limit = 6
            cfg.rewards.turn_streak_penalty = -0.01

            trainer = RankPPOTrainer(cfg)
            trainer.train()

            metrics_path = cfg.output_dir / "metrics.jsonl"
            row = metrics_path.read_text(encoding="utf-8").strip().splitlines()[-1]
            self.assertIn('"rollout_pool_mode": "single"', row)

    def test_trainer_smoke_planning_arch_one_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            cfg = TrainConfig(
                seed=11,
                total_updates=1,
                rollout_episodes_per_update=2,
                policy_arch="planning",
                policy_hidden_dim=128,
                policy_mem_dim=16,
                eval_matches=2,
                eval_interval_updates=10,
                output_dir=base / "out",
                checkpoint_dir=base / "out" / "checkpoints",
                replay_dir=base / "replays",
            )
            cfg.env.max_steps = 20
            cfg.ppo.minibatch_size = 16
            cfg.ppo.epochs = 1
            cfg.progress_replay_every_updates = 0
            cfg.save_replay_every_updates = 0

            trainer = RankPPOTrainer(cfg)
            trainer.train()

            self.assertTrue((cfg.output_dir / "final_policy.pt").exists())
            metrics_path = cfg.output_dir / "metrics.jsonl"
            self.assertTrue(metrics_path.exists())
            row = metrics_path.read_text(encoding="utf-8").strip().splitlines()[-1]
            self.assertIn('"policy_arch": "planning"', row)


if __name__ == "__main__":
    unittest.main()
