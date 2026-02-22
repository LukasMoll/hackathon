from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RankRewardConfig:
    first: float = 1.00
    second: float = 0.60
    third: float = 0.25
    fourth: float = 0.00
    fifth: float = -0.25
    sixth: float = -0.60
    survival_alpha: float = 0.0
    loop_cycle_penalty: float = -0.002
    loop_cycle_lengths: tuple[int, ...] = (4, 6, 8, 10)
    stagnation_penalty: float = -0.001
    stagnation_window: int = 20
    stagnation_unique_threshold: int = 8
    turn_streak_penalty: float = -0.01
    turn_streak_limit: int = 6


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    policy_lr: float = 3e-4
    epochs: int = 4
    minibatch_size: int = 256
    entropy_coef: float = 0.02
    value_coef: float = 0.5
    max_grad_norm: float = 0.5


@dataclass
class LeagueConfig:
    snapshot_interval_updates: int = 10
    snapshot_pool_size: int = 64
    p_recent: float = 0.5
    p_old: float = 0.3
    p_scripted: float = 0.2


@dataclass
class EnvConfig:
    radius: int = 5
    max_steps: int = 300
    randomize_starts: bool = True
    randomize_facings: bool = True
    use_c_core: bool = True
    require_c_core: bool = False


@dataclass
class TrainConfig:
    seed: int = 42
    total_updates: int = 500
    rollout_episodes_per_update: int = 64
    policy_arch: str = "planning"
    policy_hidden_dim: int = 128
    policy_mem_dim: int = 16
    eval_matches: int = 50
    eval_interval_updates: int = 10
    save_replay_every_updates: int = 20
    save_replay_prefix: str = "mltrain"
    progress_replay_every_updates: int = 10
    progress_replay_matches: int = 2
    progress_replay_deterministic: bool = True
    progress_replay_prefix: str = "mlprog"
    show_progress_bar: bool = True

    output_dir: Path = Path("training_outputs")
    checkpoint_dir: Path = Path("training_outputs/checkpoints")
    replay_dir: Path = Path("ml_runs")

    ppo: PPOConfig = field(default_factory=PPOConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    rewards: RankRewardConfig = field(default_factory=RankRewardConfig)
    league: LeagueConfig = field(default_factory=LeagueConfig)
