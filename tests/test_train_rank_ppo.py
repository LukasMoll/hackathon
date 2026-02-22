import tempfile
import unittest
from pathlib import Path

from train_rank_ppo import archive_existing_ml_runs, build_arg_parser


class TrainRankPPOTests(unittest.TestCase):
    def test_default_training_args_use_planning_rank_terminal_setup(self) -> None:
        args = build_arg_parser().parse_args([])
        self.assertEqual(args.policy_arch, "planning")
        self.assertEqual(args.survival_alpha, 0.0)
        self.assertEqual(args.history_benchmark_points, "10,20,30,40,50,60,70,80,90")
        self.assertEqual(args.history_benchmark_matches, 120)
        self.assertEqual(args.history_benchmark_neighbor_radius, 1)

    def test_archive_existing_ml_runs_moves_top_level_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            replay_dir = Path(tmpdir) / "ml_runs"
            replay_dir.mkdir(parents=True, exist_ok=True)

            (replay_dir / "old1.json").write_text("{}", encoding="utf-8")
            (replay_dir / "old2.json").write_text("{}", encoding="utf-8")
            (replay_dir / "keep.txt").write_text("x", encoding="utf-8")

            moved, archive_dir = archive_existing_ml_runs(replay_dir)

            self.assertEqual(moved, 2)
            self.assertIsNotNone(archive_dir)
            assert archive_dir is not None
            self.assertTrue((archive_dir / "old1.json").exists())
            self.assertTrue((archive_dir / "old2.json").exists())
            self.assertEqual(list(replay_dir.glob("*.json")), [])
            self.assertTrue((replay_dir / "keep.txt").exists())


if __name__ == "__main__":
    unittest.main()
