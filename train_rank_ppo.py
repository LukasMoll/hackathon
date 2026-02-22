from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

from benchmark_latest_vs_history import DEFAULT_PERCENT_POINTS, parse_percent_points, write_latest_vs_history_outputs
from plot_training_survival import write_default_plots
from rl.config import TrainConfig
from rl.trainer import RankPPOTrainer


def archive_existing_ml_runs(replay_dir: Path) -> Tuple[int, Optional[Path]]:
    replay_dir.mkdir(parents=True, exist_ok=True)
    old_runs = sorted([p for p in replay_dir.glob("*.json") if p.is_file()])
    if not old_runs:
        return (0, None)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_dir = replay_dir / "_archive" / stamp
    archive_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    for src in old_runs:
        dst = archive_dir / src.name
        if dst.exists():
            idx = 1
            while True:
                candidate = archive_dir / f"{src.stem}_{idx:03d}{src.suffix}"
                if not candidate.exists():
                    dst = candidate
                    break
                idx += 1
        shutil.move(str(src), str(dst))
        moved += 1

    return (moved, archive_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train rank-optimizing multi-agent PPO on Tronk ML env")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--total-updates", type=int, default=500)
    p.add_argument("--rollout-episodes", type=int, default=64)
    p.add_argument("--policy-arch", choices=["gru", "planning"], default="planning")
    p.add_argument("--policy-hidden-dim", type=int, default=128)
    p.add_argument("--policy-mem-dim", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--radius", type=int, default=5)
    p.add_argument("--survival-alpha", type=float, default=0.0)
    p.add_argument("--policy-lr", type=float, default=3e-4)
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--output-dir", type=str, default="training_outputs")
    p.add_argument("--eval-interval", type=int, default=10)
    p.add_argument("--eval-matches", type=int, default=50)
    p.add_argument("--snapshot-interval", type=int, default=10)
    p.add_argument("--snapshot-pool-size", type=int, default=64)
    p.add_argument("--progress-replay-interval", type=int, default=10)
    p.add_argument("--progress-replay-matches", type=int, default=2)
    p.add_argument("--progress-replay-stochastic", action="store_true")
    p.add_argument("--no-progress-bar", action="store_true")
    p.add_argument("--progress-prefix", type=str, default="mlprog")
    p.add_argument("--save-replay-interval", type=int, default=20)
    p.add_argument("--save-replay-prefix", type=str, default="mltrain")
    p.add_argument("--engine", choices=["c", "python"], default="c")
    p.add_argument("--require-c-core", action="store_true")
    p.add_argument("--entropy-coef", type=float, default=0.02)
    p.add_argument("--loop-cycle-penalty", type=float, default=-0.002)
    p.add_argument("--stagnation-penalty", type=float, default=-0.001)
    p.add_argument("--stagnation-window", type=int, default=20)
    p.add_argument("--stagnation-unique-threshold", type=int, default=8)
    p.add_argument("--turn-streak-penalty", type=float, default=-0.01)
    p.add_argument("--turn-streak-limit", type=int, default=6)
    p.add_argument("--no-auto-plots", action="store_true")
    p.add_argument(
        "--history-benchmark-points",
        type=str,
        default=",".join(str(p) for p in DEFAULT_PERCENT_POINTS),
        help="Comma-separated historical percent points for latest-vs-history benchmark (e.g. 10,20,30,40,50)",
    )
    p.add_argument(
        "--history-benchmark-matches",
        type=int,
        default=120,
        help="Matches per historical benchmark point",
    )
    p.add_argument(
        "--history-benchmark-neighbor-radius",
        type=int,
        default=1,
        help="Neighbor radius for smoothing each benchmark point (1 => p-1,p,p+1)",
    )
    p.add_argument(
        "--no-history-benchmark",
        action="store_true",
        help="Skip latest-vs-history benchmark graph generation at end of training",
    )
    p.add_argument(
        "--no-archive-old-ml-runs",
        action="store_true",
        help="Keep existing top-level ml_runs JSON files instead of archiving them before training",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    output_dir = Path(args.output_dir)
    cfg = TrainConfig(
        seed=args.seed,
        total_updates=args.total_updates,
        rollout_episodes_per_update=args.rollout_episodes,
        policy_arch=args.policy_arch,
        policy_hidden_dim=args.policy_hidden_dim,
        policy_mem_dim=args.policy_mem_dim,
        eval_matches=args.eval_matches,
        eval_interval_updates=args.eval_interval,
        output_dir=output_dir,
        checkpoint_dir=output_dir / "checkpoints",
        replay_dir=Path("ml_runs"),
    )
    cfg.env.max_steps = args.max_steps
    cfg.env.radius = args.radius
    cfg.env.use_c_core = args.engine != "python"
    cfg.env.require_c_core = args.require_c_core

    cfg.rewards.survival_alpha = args.survival_alpha
    cfg.rewards.loop_cycle_penalty = args.loop_cycle_penalty
    cfg.rewards.stagnation_penalty = args.stagnation_penalty
    cfg.rewards.stagnation_window = args.stagnation_window
    cfg.rewards.stagnation_unique_threshold = args.stagnation_unique_threshold
    cfg.rewards.turn_streak_penalty = args.turn_streak_penalty
    cfg.rewards.turn_streak_limit = max(1, args.turn_streak_limit)

    cfg.ppo.policy_lr = args.policy_lr
    cfg.ppo.minibatch_size = args.minibatch_size
    cfg.ppo.epochs = args.epochs
    cfg.ppo.entropy_coef = args.entropy_coef

    cfg.league.snapshot_interval_updates = args.snapshot_interval
    cfg.league.snapshot_pool_size = args.snapshot_pool_size
    cfg.progress_replay_every_updates = args.progress_replay_interval
    cfg.progress_replay_matches = args.progress_replay_matches
    cfg.progress_replay_deterministic = not args.progress_replay_stochastic
    cfg.progress_replay_prefix = args.progress_prefix
    cfg.show_progress_bar = not args.no_progress_bar
    cfg.save_replay_every_updates = args.save_replay_interval
    cfg.save_replay_prefix = args.save_replay_prefix

    if not args.no_archive_old_ml_runs:
        moved, archive_dir = archive_existing_ml_runs(cfg.replay_dir)
        if moved > 0 and archive_dir is not None:
            print(f"Archived {moved} old ml run files to {archive_dir}")

    trainer = RankPPOTrainer(cfg)
    trainer.train()

    if not args.no_auto_plots:
        try:
            written = write_default_plots(cfg.output_dir / "metrics.jsonl", cfg.output_dir, ma_window=10)
            print(f"Wrote {written['survival']}")
            print(f"Wrote {written['placement']}")
        except Exception as exc:
            print(f"Skipping auto-plot generation: {exc}")

        if not args.no_history_benchmark:
            try:
                points = parse_percent_points(args.history_benchmark_points)
                written_hist = write_latest_vs_history_outputs(
                    output_dir=cfg.output_dir,
                    percent_points=points,
                    matches_per_point=max(1, int(args.history_benchmark_matches)),
                    neighbor_radius=max(0, int(args.history_benchmark_neighbor_radius)),
                    seed=cfg.seed + 900_000,
                    deterministic=True,
                    radius=cfg.env.radius,
                    max_steps=cfg.env.max_steps,
                    engine=("c" if cfg.env.use_c_core else "python"),
                    require_c_core=cfg.env.require_c_core,
                )
                print(f"Wrote {written_hist['json']}")
                print(f"Wrote {written_hist['svg']}")
            except Exception as exc:
                print(f"Skipping latest-vs-history benchmark generation: {exc}")


if __name__ == "__main__":
    main()
