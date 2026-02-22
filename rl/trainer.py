from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from rl.config import TrainConfig
from rl.league import LeaguePool, SnapshotEntry
from rl.obs import ObsSpec, ObservationEncoder
from rl.policy import RandomScriptedPolicy, SafeGreedyScriptedPolicy, build_policy
from rl.ppo import PPOBatch, ppo_update
from rl.progress import run_self_play_match, write_progress_replay
from rl.rewards import rank_to_reward
from tronk_ml import MLTronkEnv


ACTION_TO_TURN = [-1, 0, 1]


class RankPPOTrainer:
    def __init__(self, config: TrainConfig):
        self.cfg = config
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.replay_dir.mkdir(parents=True, exist_ok=True)

        self.rng = random.Random(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = ObservationEncoder(ObsSpec(radius=self.cfg.env.radius, include_threat_map=True))
        self.policy = build_policy(
            self.cfg.policy_arch,
            self.encoder.obs_dim,
            hidden_dim=self.cfg.policy_hidden_dim,
            mem_dim=self.cfg.policy_mem_dim,
        ).to(self.device)
        self.hidden_dim = int(getattr(self.policy, "hidden_dim", 128))
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.cfg.ppo.policy_lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: max(0.1, 1.0 - (step / max(1, self.cfg.total_updates))),
        )

        self.league = LeaguePool(self.cfg.league)
        self.scripted = [SafeGreedyScriptedPolicy(), RandomScriptedPolicy()]
        self.snapshot_cache: Dict[str, torch.nn.Module] = {}

        self.metrics_path = self.cfg.output_dir / "metrics.jsonl"
        self.config_path = self.cfg.output_dir / "train_config.json"
        self.config_path.write_text(json.dumps(self._config_to_json(), indent=2), encoding="utf-8")
        self._progress_is_tty = sys.stdout.isatty()

    def _config_to_json(self) -> Dict[str, Any]:
        data = asdict(self.cfg)
        for key in ["output_dir", "checkpoint_dir", "replay_dir"]:
            data[key] = str(data[key])
        return data

    def _load_snapshot_policy(self, entry: SnapshotEntry) -> torch.nn.Module:
        key = str(entry.path)
        if key in self.snapshot_cache:
            return self.snapshot_cache[key]
        model = build_policy(
            self.cfg.policy_arch,
            self.encoder.obs_dim,
            hidden_dim=self.cfg.policy_hidden_dim,
            mem_dim=self.cfg.policy_mem_dim,
        ).to(self.device)
        state = torch.load(entry.path, map_location=self.device)
        model.load_state_dict(state)
        model.eval()
        self.snapshot_cache[key] = model
        return model

    def _sample_opponent_type(self) -> Tuple[str, Optional[SnapshotEntry], Optional[int]]:
        r = self.rng.random()
        if r < self.cfg.league.p_scripted:
            return ("scripted", None, self.rng.randrange(len(self.scripted)))

        snap = self.league.sample_snapshot(self.rng)
        if snap is not None:
            return ("snapshot", snap, None)

        # Fallback to scripted when snapshot pool is empty.
        return ("scripted", None, self.rng.randrange(len(self.scripted)))

    def _act_with_policy(
        self,
        model: torch.nn.Module,
        obs: np.ndarray,
        hidden: np.ndarray,
        deterministic: bool,
    ) -> Tuple[int, float, float, np.ndarray]:
        # The policy classes expose act(...) with a unified interface.
        return model.act(obs, hidden, self.device, deterministic=deterministic)

    def _make_env(self, seed: int) -> MLTronkEnv:
        return MLTronkEnv(
            seed=seed,
            max_steps=self.cfg.env.max_steps,
            randomize_starts=self.cfg.env.randomize_starts,
            randomize_facings=self.cfg.env.randomize_facings,
            use_c_core=self.cfg.env.use_c_core,
            require_c_core=self.cfg.env.require_c_core,
        )

    def _run_single_episode(
        self,
        update_idx: int,
        episode_idx: int,
        deterministic_learner: bool = False,
    ) -> Tuple[PPOBatch, Dict[str, float], Dict[str, Any]]:
        env = self._make_env(seed=self.rng.randint(0, 2_000_000_000))
        learner_pid = self.rng.randrange(6)

        seat_type: Dict[int, str] = {learner_pid: "learner"}
        seat_snapshot: Dict[int, Optional[SnapshotEntry]] = {learner_pid: None}
        seat_scripted: Dict[int, Optional[int]] = {learner_pid: None}

        for pid in range(6):
            if pid == learner_pid:
                continue
            t, snap, scripted_idx = self._sample_opponent_type()
            seat_type[pid] = t
            seat_snapshot[pid] = snap
            seat_scripted[pid] = scripted_idx

        opp_hidden = {pid: np.zeros((self.hidden_dim,), dtype=np.float32) for pid in range(6)}
        learner_hidden = np.zeros((self.hidden_dim,), dtype=np.float32)

        batch = PPOBatch()
        learner_alive = True
        survival_steps = 0
        head_history: List[Tuple[int, int]] = [env.players[learner_pid].head]
        repeated_turn_streak = 0
        repeated_turn_value = 0

        while True:
            if not learner_alive:
                break
            if env.done:
                break

            step_actions: List[int] = [0] * 6

            learner_obs = self.encoder.encode(env, learner_pid)
            hidden_before = learner_hidden.copy()
            a, logp, val, learner_hidden = self._act_with_policy(
                self.policy,
                learner_obs,
                learner_hidden,
                deterministic=deterministic_learner,
            )
            learner_turn = ACTION_TO_TURN[a]
            step_actions[learner_pid] = learner_turn

            if learner_turn in (-1, 1):
                if repeated_turn_value == learner_turn:
                    repeated_turn_streak += 1
                else:
                    repeated_turn_value = learner_turn
                    repeated_turn_streak = 1
            else:
                repeated_turn_value = 0
                repeated_turn_streak = 0

            for pid in range(6):
                if pid == learner_pid:
                    continue
                if not env.players[pid].alive:
                    step_actions[pid] = 0
                    continue

                stype = seat_type[pid]
                if stype == "scripted":
                    scripted_idx = seat_scripted[pid]
                    assert scripted_idx is not None
                    action_idx = self.scripted[scripted_idx].act(env, pid)
                    step_actions[pid] = ACTION_TO_TURN[action_idx]
                    continue

                if stype == "snapshot":
                    snap = seat_snapshot[pid]
                    assert snap is not None
                    opp_model = self._load_snapshot_policy(snap)
                    obs = self.encoder.encode(env, pid)
                    action_idx, _, _, new_h = self._act_with_policy(
                        opp_model,
                        obs,
                        opp_hidden[pid],
                        deterministic=False,
                    )
                    opp_hidden[pid] = new_h
                    step_actions[pid] = ACTION_TO_TURN[action_idx]
                    continue

                step_actions[pid] = 0

            before_alive = env.players[learner_pid].alive
            env.step(step_actions)
            after_alive = env.players[learner_pid].alive

            reward = self.cfg.rewards.survival_alpha if before_alive else 0.0
            done = False
            if repeated_turn_streak >= self.cfg.rewards.turn_streak_limit:
                reward += self.cfg.rewards.turn_streak_penalty

            if after_alive:
                curr_head = env.players[learner_pid].head
                for cycle_len in self.cfg.rewards.loop_cycle_lengths:
                    if len(head_history) >= cycle_len and curr_head == head_history[-cycle_len]:
                        reward += self.cfg.rewards.loop_cycle_penalty
                        break
                head_history.append(curr_head)

                if len(head_history) >= self.cfg.rewards.stagnation_window:
                    window = head_history[-self.cfg.rewards.stagnation_window :]
                    if len(set(window)) <= self.cfg.rewards.stagnation_unique_threshold:
                        reward += self.cfg.rewards.stagnation_penalty

            if before_alive and not after_alive:
                done = True
            elif env.done and after_alive:
                done = True

            if done:
                rank = env.compute_ranks()[learner_pid]
                reward += rank_to_reward(rank, self.cfg.rewards)

            batch.add(
                obs=learner_obs,
                action=a,
                log_prob=logp,
                value=val,
                reward=reward,
                done=done,
                hidden=hidden_before,
            )

            survival_steps += 1
            learner_alive = after_alive

            if done:
                break

        final_rank = float(env.compute_ranks()[learner_pid])
        metrics = {
            "rank": final_rank,
            "survival_steps": float(survival_steps),
            "win": 1.0 if final_rank <= 1.0 else 0.0,
        }
        aux = {
            "env_snapshot": env.snapshot(),
            "learner_pid": learner_pid,
            "seat_type": seat_type,
            "core_mode": env.core_mode,
        }
        return batch, metrics, aux

    def _save_replay(self, update_idx: int, episode_idx: int, snapshot: Dict[str, Any]) -> None:
        replay_id = f"{self.cfg.save_replay_prefix}_u{update_idx:06d}_ep{episode_idx:04d}"
        payload = {
            "session_id": replay_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "snapshot": snapshot,
        }
        path = self.cfg.replay_dir / f"{replay_id}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def export_progress_replays_for_current_policy(
        self,
        update_idx: int,
        matches: int,
        *,
        deterministic: bool = True,
        prefix: str = "mlprog",
        checkpoint_name: str = "",
    ) -> List[Path]:
        self.policy.eval()
        written: List[Path] = []
        for match_idx in range(1, matches + 1):
            seed = (self.cfg.seed * 1_000_000) + (update_idx * 100) + match_idx
            snapshot = run_self_play_match(
                self.policy,
                self.encoder,
                self.device,
                seed=seed,
                max_steps=self.cfg.env.max_steps,
                randomize_starts=self.cfg.env.randomize_starts,
                randomize_facings=self.cfg.env.randomize_facings,
                use_c_core=self.cfg.env.use_c_core,
                require_c_core=self.cfg.env.require_c_core,
                deterministic=deterministic,
            )
            path = write_progress_replay(
                output_dir=self.cfg.replay_dir,
                prefix=prefix,
                update_idx=update_idx,
                match_idx=match_idx,
                seed=seed,
                snapshot=snapshot,
                deterministic=deterministic,
                checkpoint_name=checkpoint_name,
            )
            written.append(path)
        return written

    def _evaluate(self, matches: int) -> Dict[str, float]:
        ranks: List[float] = []
        wins = 0
        survivals: List[float] = []

        for i in range(matches):
            _, metrics, _ = self._run_single_episode(
                update_idx=-1,
                episode_idx=i,
                deterministic_learner=True,
            )
            ranks.append(metrics["rank"])
            survivals.append(metrics["survival_steps"])
            if metrics["win"] > 0.5:
                wins += 1

        return {
            "eval_avg_rank": float(np.mean(ranks)) if ranks else 0.0,
            "eval_win_rate": wins / max(1, matches),
            "eval_avg_survival_steps": float(np.mean(survivals)) if survivals else 0.0,
        }

    @staticmethod
    def _format_duration(seconds: float) -> str:
        s = max(0, int(seconds))
        h = s // 3600
        m = (s % 3600) // 60
        sec = s % 60
        return f"{h:02d}:{m:02d}:{sec:02d}"

    def _render_progress(self, update_idx: int, started_at: float, metrics_row: Dict[str, Any]) -> None:
        if not self.cfg.show_progress_bar:
            return

        total = max(1, self.cfg.total_updates)
        elapsed = time.perf_counter() - started_at
        avg_per_update = elapsed / max(1, update_idx)
        remaining_updates = max(0, total - update_idx)
        eta_seconds = avg_per_update * remaining_updates

        progress = min(1.0, update_idx / total)
        bar_width = 28
        filled = int(progress * bar_width)
        if filled >= bar_width:
            bar = "=" * bar_width
        else:
            bar = ("=" * filled) + ">" + ("." * (bar_width - filled - 1))

        line = (
            f"[{bar}] {update_idx:4d}/{total:4d} {progress * 100:5.1f}% "
            f"ETA {self._format_duration(eta_seconds)} "
            f"| surv {metrics_row['avg_survival_steps']:.2f} "
            f"rank {metrics_row['avg_rank']:.2f} "
            f"win {metrics_row['win_rate']:.2f}"
        )

        if self._progress_is_tty:
            print(f"\r{line}", end="", flush=True)
        else:
            print(line, flush=True)

    def train(self) -> None:
        started_at = time.perf_counter()
        for update_idx in range(1, self.cfg.total_updates + 1):
            rollout = PPOBatch()
            ranks: List[float] = []
            survivals: List[float] = []
            wins = 0
            core_modes: List[str] = []

            for ep in range(self.cfg.rollout_episodes_per_update):
                batch, metrics, aux = self._run_single_episode(update_idx, ep, deterministic_learner=False)
                if len(batch) > 0:
                    rollout.extend(batch)

                ranks.append(metrics["rank"])
                survivals.append(metrics["survival_steps"])
                wins += int(metrics["win"] > 0.5)
                core_modes.append(str(aux.get("core_mode", "unknown")))

                if (
                    self.cfg.save_replay_every_updates > 0
                    and update_idx % self.cfg.save_replay_every_updates == 0
                    and metrics["rank"] <= 2.0
                ):
                    self._save_replay(update_idx, ep, aux["env_snapshot"])

            if self.cfg.env.require_c_core and core_modes and any(mode != "c" for mode in core_modes):
                raise RuntimeError("C core required for rollouts, but non-C mode was observed")

            if len(rollout) == 0:
                continue

            self.policy.train()
            ppo_stats = ppo_update(self.policy, self.optimizer, rollout, self.cfg.ppo, self.device)
            self.scheduler.step()

            eval_stats: Dict[str, float] = {}
            if update_idx % self.cfg.eval_interval_updates == 0:
                self.policy.eval()
                eval_stats = self._evaluate(self.cfg.eval_matches)

            if update_idx % self.cfg.league.snapshot_interval_updates == 0:
                self.league.add_snapshot(update_idx, self.policy.state_dict(), self.cfg.checkpoint_dir)

            if (
                self.cfg.progress_replay_every_updates > 0
                and self.cfg.progress_replay_matches > 0
                and update_idx % self.cfg.progress_replay_every_updates == 0
            ):
                self.export_progress_replays_for_current_policy(
                    update_idx=update_idx,
                    matches=self.cfg.progress_replay_matches,
                    deterministic=self.cfg.progress_replay_deterministic,
                    prefix=self.cfg.progress_replay_prefix,
                )

            lr = self.optimizer.param_groups[0]["lr"]
            metrics_row: Dict[str, Any] = {
                "update": update_idx,
                "samples": len(rollout),
                "avg_rank": float(np.mean(ranks)) if ranks else 0.0,
                "win_rate": wins / max(1, len(ranks)),
                "avg_survival_steps": float(np.mean(survivals)) if survivals else 0.0,
                "loss": ppo_stats.loss,
                "policy_loss": ppo_stats.policy_loss,
                "value_loss": ppo_stats.value_loss,
                "entropy": ppo_stats.entropy,
                "lr": lr,
                "snapshot_pool": len(self.league.snapshots),
                "env_core_mode": core_modes[0] if core_modes else "unknown",
                "rollout_pool_mode": "single",
                "policy_arch": self.cfg.policy_arch,
            }
            metrics_row.update(eval_stats)

            with self.metrics_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(metrics_row) + "\n")

            self._render_progress(update_idx, started_at, metrics_row)

        if self.cfg.show_progress_bar and self._progress_is_tty:
            print()
        final_path = self.cfg.output_dir / "final_policy.pt"
        torch.save(self.policy.state_dict(), final_path)
        print(f"Training complete. Final policy: {final_path}")
