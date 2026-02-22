from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from rl.tree_surrogate import (
    ACTION_TO_TURN,
    FeatureSchema,
    TronkscriptInferencePolicy,
    benchmark_tronkscript_vs_model,
    extract_features,
    load_policy,
    validate_tronkscript_bot,
)
from tronk_engine import TronkscriptParser
from tronk_ml import MLTronkEnv


TURN_VALUES = (-1, 0, 1)
TURN_TO_INDEX = {-1: 0, 0: 1, 1: 2}
INDEX_TO_TURN = {0: -1, 1: 0, 2: 1}


def split_train_test(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    train_frac: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if x.shape[0] != y.shape[0]:
        raise ValueError("x/y size mismatch")
    n = x.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 samples")
    rng = np.random.default_rng(seed)
    order = np.arange(n, dtype=np.int32)
    rng.shuffle(order)
    cut = int(max(1, min(n - 1, round(float(train_frac) * n))))
    train_idx = order[:cut]
    test_idx = order[cut:]
    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]


def split_train_test_with_logits(
    x: np.ndarray,
    y: np.ndarray,
    logits: np.ndarray,
    *,
    seed: int,
    train_frac: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if logits.shape[0] != x.shape[0]:
        raise ValueError("logits/x size mismatch")
    if logits.shape[1] != 3:
        raise ValueError("teacher logits must have shape [N, 3]")
    n = x.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 samples")
    rng = np.random.default_rng(seed)
    order = np.arange(n, dtype=np.int32)
    rng.shuffle(order)
    cut = int(max(1, min(n - 1, round(float(train_frac) * n))))
    train_idx = order[:cut]
    test_idx = order[cut:]
    return (
        x[train_idx],
        y[train_idx],
        logits[train_idx],
        x[test_idx],
        y[test_idx],
        logits[test_idx],
    )


class CompactStudentNet(nn.Module):
    def __init__(self, input_dim: int, hidden1: int, hidden2: int, action_dim: int = 3):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden1 = int(hidden1)
        self.hidden2 = int(hidden2)
        self.action_dim = int(action_dim)

        self.fc1 = nn.Linear(self.input_dim, self.hidden1)
        self.fc2 = nn.Linear(self.hidden1, self.hidden2)
        self.fc_out = nn.Linear(self.hidden2, self.action_dim)
        self.value_head = nn.Linear(self.hidden2, 1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return h2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h2 = self.forward_features(x)
        return self.fc_out(h2)

    def forward_with_value(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h2 = self.forward_features(x)
        logits = self.fc_out(h2)
        value = self.value_head(h2).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def predict_indices(self, x_np: np.ndarray, device: torch.device, batch_size: int = 4096) -> np.ndarray:
        self.eval()
        out: List[np.ndarray] = []
        for start in range(0, x_np.shape[0], batch_size):
            xb = torch.from_numpy(x_np[start : start + batch_size].astype(np.float32)).to(device)
            logits = self.forward(xb)
            pred = torch.argmax(logits, dim=-1).cpu().numpy().astype(np.int16)
            out.append(pred)
        return np.concatenate(out, axis=0) if out else np.zeros((0,), dtype=np.int16)

    @torch.no_grad()
    def predict_turn_and_scores_one(self, row: np.ndarray, device: torch.device) -> Tuple[int, Dict[int, float]]:
        self.eval()
        xb = torch.from_numpy(row.astype(np.float32)).to(device).unsqueeze(0)
        logits = self.forward(xb).squeeze(0).detach().cpu().numpy()
        scores = {-1: float(logits[0]), 0: float(logits[1]), 1: float(logits[2])}
        action_idx = int(np.argmax(logits))
        return int(INDEX_TO_TURN[action_idx]), scores


@dataclass
class TrainConfig:
    epochs: int = 12
    batch_size: int = 1024
    lr: float = 1e-3
    weight_decay: float = 1e-5
    seed: int = 123


@dataclass
class DistillConfig:
    ce_coef: float = 1.0
    kl_coef: float = 0.0
    temperature: float = 1.0


@dataclass
class PPOFineTuneConfig:
    updates: int = 0
    rollout_episodes_per_update: int = 24
    max_steps: int = 160
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    ppo_epochs: int = 3
    minibatch_size: int = 512
    policy_lr: float = 2e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    survival_alpha: float = 0.0
    seed: int = 123


def train_student_model(
    x_train: np.ndarray,
    y_train_turn: np.ndarray,
    x_test: np.ndarray,
    y_test_turn: np.ndarray,
    *,
    hidden1: int,
    hidden2: int,
    cfg: TrainConfig,
    distill_cfg: Optional[DistillConfig],
    teacher_logits_train: Optional[np.ndarray],
    teacher_logits_test: Optional[np.ndarray],
    device: torch.device,
) -> Tuple[CompactStudentNet, Dict[str, Any]]:
    if x_train.ndim != 2:
        raise ValueError("x_train must be 2D")
    if y_train_turn.ndim != 1:
        raise ValueError("y_train_turn must be 1D")

    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    model = CompactStudentNet(input_dim=x_train.shape[1], hidden1=hidden1, hidden2=hidden2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    criterion = nn.CrossEntropyLoss()
    distill = distill_cfg or DistillConfig()
    use_kl = (
        float(distill.kl_coef) > 0.0
        and teacher_logits_train is not None
        and teacher_logits_test is not None
        and teacher_logits_train.shape[0] == x_train.shape[0]
        and teacher_logits_test.shape[0] == x_test.shape[0]
    )

    y_train_idx = np.asarray([TURN_TO_INDEX[int(v)] for v in y_train_turn], dtype=np.int64)
    y_test_idx = np.asarray([TURN_TO_INDEX[int(v)] for v in y_test_turn], dtype=np.int64)

    n = x_train.shape[0]
    order = np.arange(n, dtype=np.int32)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_test_acc = -1.0
    history: List[Dict[str, float]] = []

    for epoch in range(max(1, int(cfg.epochs))):
        np.random.shuffle(order)
        model.train()
        losses: List[float] = []
        for start in range(0, n, int(cfg.batch_size)):
            idx = order[start : start + int(cfg.batch_size)]
            xb = torch.from_numpy(x_train[idx].astype(np.float32)).to(device)
            yb = torch.from_numpy(y_train_idx[idx]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            ce_loss = criterion(logits, yb)
            loss = float(distill.ce_coef) * ce_loss
            if use_kl:
                tlog = torch.from_numpy(teacher_logits_train[idx].astype(np.float32)).to(device)
                t = max(1e-6, float(distill.temperature))
                student_logp = F.log_softmax(logits / t, dim=-1)
                teacher_prob = F.softmax(tlog / t, dim=-1)
                kl_loss = F.kl_div(student_logp, teacher_prob, reduction="batchmean") * (t * t)
                loss = loss + (float(distill.kl_coef) * kl_loss)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        model.eval()
        train_pred = model.predict_indices(x_train, device=device)
        test_pred = model.predict_indices(x_test, device=device)
        train_acc = float(np.mean(train_pred == y_train_idx))
        test_acc = float(np.mean(test_pred == y_test_idx))
        avg_loss = float(np.mean(losses)) if losses else 0.0
        train_kl = 0.0
        test_kl = 0.0
        if use_kl:
            with torch.no_grad():
                logits_train = []
                logits_test = []
                bs = max(256, int(cfg.batch_size))
                for start in range(0, x_train.shape[0], bs):
                    xb = torch.from_numpy(x_train[start : start + bs].astype(np.float32)).to(device)
                    logits_train.append(model(xb).cpu().numpy())
                for start in range(0, x_test.shape[0], bs):
                    xb = torch.from_numpy(x_test[start : start + bs].astype(np.float32)).to(device)
                    logits_test.append(model(xb).cpu().numpy())
                st_train = np.concatenate(logits_train, axis=0)
                st_test = np.concatenate(logits_test, axis=0)
                t = max(1e-6, float(distill.temperature))
                p_train = F.log_softmax(torch.from_numpy(st_train) / t, dim=-1)
                q_train = F.softmax(torch.from_numpy(teacher_logits_train) / t, dim=-1)
                p_test = F.log_softmax(torch.from_numpy(st_test) / t, dim=-1)
                q_test = F.softmax(torch.from_numpy(teacher_logits_test) / t, dim=-1)
                train_kl = float(F.kl_div(p_train, q_train, reduction="batchmean").item() * (t * t))
                test_kl = float(F.kl_div(p_test, q_test, reduction="batchmean").item() * (t * t))
        history.append(
            {
                "epoch": float(epoch + 1),
                "loss": avg_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "train_kl": float(train_kl),
                "test_kl": float(test_kl),
            }
        )
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()

    train_pred = model.predict_indices(x_train, device=device)
    test_pred = model.predict_indices(x_test, device=device)
    metrics = {
        "train_acc": float(np.mean(train_pred == y_train_idx)),
        "test_acc": float(np.mean(test_pred == y_test_idx)),
        "history": history,
    }
    return model, metrics


def _teacher_logits_and_turn(
    model: nn.Module,
    obs: np.ndarray,
    hidden_np: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, int]:
    obs_t = torch.from_numpy(obs.astype(np.float32)).to(device).unsqueeze(0)
    h_t = torch.from_numpy(hidden_np.astype(np.float32)).to(device)
    if h_t.dim() == 1:
        h_t = h_t.unsqueeze(0).unsqueeze(0)
    elif h_t.dim() == 2:
        h_t = h_t.unsqueeze(0)
    with torch.no_grad():
        logits_t, _value_t, new_h_t = model(obs_t, h_t)  # type: ignore[misc]
    logits = logits_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
    action_idx = int(np.argmax(logits))
    turn = int(ACTION_TO_TURN[action_idx])
    new_h = new_h_t.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
    return logits, new_h, turn


def collect_teacher_dataset_with_logits(
    *,
    policy_path: Path,
    schema: FeatureSchema,
    target_samples: int,
    seed: int,
    max_steps: int,
    use_c_core: bool,
    require_c_core: bool,
    deterministic_teacher: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    if target_samples <= 0:
        raise ValueError("target_samples must be > 0")

    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, encoder = load_policy(policy_path, device=device)
    hidden_dim = int(getattr(model, "hidden_dim", 128))

    x_rows: List[np.ndarray] = []
    y_rows: List[int] = []
    logits_rows: List[np.ndarray] = []
    episodes = 0
    steps = 0

    while len(x_rows) < target_samples:
        env = MLTronkEnv(
            seed=int(rng.integers(0, 2_000_000_000)),
            max_steps=max_steps,
            randomize_starts=True,
            randomize_facings=True,
            use_c_core=use_c_core,
            require_c_core=require_c_core,
        )
        hidden = [np.zeros((hidden_dim,), dtype=np.float32) for _ in range(6)]
        prev_actions = [0] * 6

        while not env.done and len(x_rows) < target_samples:
            actions = [0] * 6
            for pid in range(6):
                if not env.players[pid].alive:
                    continue
                feat = extract_features(env, pid, prev_actions[pid], schema)
                obs = encoder.encode(env, pid)
                teacher_logits, new_h, teacher_turn = _teacher_logits_and_turn(model, obs, hidden[pid], device)
                hidden[pid] = new_h

                x_rows.append(feat)
                y_rows.append(int(teacher_turn))
                logits_rows.append(teacher_logits)

                actions[pid] = int(teacher_turn)
                prev_actions[pid] = int(teacher_turn)
            env.step(actions)
            steps += 1
        episodes += 1

    x = np.asarray(x_rows[:target_samples], dtype=np.int16)
    y = np.asarray(y_rows[:target_samples], dtype=np.int8)
    logits = np.asarray(logits_rows[:target_samples], dtype=np.float32)
    meta = {
        "episodes": int(episodes),
        "env_steps": int(steps),
        "samples": int(x.shape[0]),
        "feature_count": int(x.shape[1]),
    }
    return x, y, logits, meta


def collect_dagger_dataset_with_logits(
    *,
    policy_path: Path,
    student_model: CompactStudentNet,
    schema: FeatureSchema,
    target_samples: int,
    seed: int,
    max_steps: int,
    use_c_core: bool,
    require_c_core: bool,
    teacher_mix_prob: float,
    use_safety_fallback: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    if target_samples <= 0:
        raise ValueError("target_samples must be > 0")

    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, encoder = load_policy(policy_path, device=device)
    hidden_dim = int(getattr(model, "hidden_dim", 128))
    student_model = student_model.to(device)
    student_model.eval()
    mix = float(max(0.0, min(1.0, teacher_mix_prob)))

    x_rows: List[np.ndarray] = []
    y_rows: List[int] = []
    logits_rows: List[np.ndarray] = []
    episodes = 0
    steps = 0
    teacher_action_count = 0
    student_action_count = 0

    while len(x_rows) < target_samples:
        env = MLTronkEnv(
            seed=int(rng.integers(0, 2_000_000_000)),
            max_steps=max_steps,
            randomize_starts=True,
            randomize_facings=True,
            use_c_core=use_c_core,
            require_c_core=require_c_core,
        )
        hidden = [np.zeros((hidden_dim,), dtype=np.float32) for _ in range(6)]
        prev_actions = [0] * 6

        while not env.done and len(x_rows) < target_samples:
            actions = [0] * 6
            for pid in range(6):
                if not env.players[pid].alive:
                    continue
                feat = extract_features(env, pid, prev_actions[pid], schema)
                obs = encoder.encode(env, pid)
                teacher_logits, new_h, teacher_turn = _teacher_logits_and_turn(model, obs, hidden[pid], device)
                hidden[pid] = new_h

                x_rows.append(feat)
                y_rows.append(int(teacher_turn))
                logits_rows.append(teacher_logits)

                student_turn, student_scores = student_model.predict_turn_and_scores_one(feat, device)
                if use_safety_fallback:
                    student_turn = apply_safety_fallback(env, pid, student_turn, student_scores)

                use_teacher = bool(rng.random() < mix)
                chosen = int(teacher_turn if use_teacher else student_turn)
                actions[pid] = chosen
                prev_actions[pid] = chosen
                if use_teacher:
                    teacher_action_count += 1
                else:
                    student_action_count += 1

            env.step(actions)
            steps += 1
        episodes += 1

    x = np.asarray(x_rows[:target_samples], dtype=np.int16)
    y = np.asarray(y_rows[:target_samples], dtype=np.int8)
    logits = np.asarray(logits_rows[:target_samples], dtype=np.float32)
    meta = {
        "episodes": int(episodes),
        "env_steps": int(steps),
        "samples": int(x.shape[0]),
        "feature_count": int(x.shape[1]),
        "teacher_mix_prob": float(mix),
        "teacher_action_count": int(teacher_action_count),
        "student_action_count": int(student_action_count),
    }
    return x, y, logits, meta


def _compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lam: float,
) -> Tuple[np.ndarray, np.ndarray]:
    n = rewards.shape[0]
    adv = np.zeros((n,), dtype=np.float32)
    gae = 0.0
    next_value = 0.0
    for t in reversed(range(n)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        adv[t] = gae
        next_value = values[t]
    returns = adv + values
    return adv, returns


def ppo_finetune_student(
    *,
    model: CompactStudentNet,
    schema: FeatureSchema,
    cfg: PPOFineTuneConfig,
    use_c_core: bool,
    require_c_core: bool,
) -> Dict[str, Any]:
    if cfg.updates <= 0:
        return {"updates": 0, "history": []}

    rng = np.random.default_rng(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.policy_lr))

    history: List[Dict[str, float]] = []
    for update_idx in range(1, int(cfg.updates) + 1):
        batch_obs: List[np.ndarray] = []
        batch_actions: List[int] = []
        batch_old_logp: List[float] = []
        batch_values: List[float] = []
        batch_rewards: List[float] = []
        batch_dones: List[float] = []
        ep_ranks: List[float] = []
        ep_surv: List[float] = []

        for _ep in range(max(1, int(cfg.rollout_episodes_per_update))):
            env = MLTronkEnv(
                seed=int(rng.integers(0, 2_000_000_000)),
                max_steps=max(1, int(cfg.max_steps)),
                randomize_starts=True,
                randomize_facings=True,
                use_c_core=use_c_core,
                require_c_core=require_c_core,
            )
            prev_actions = [0] * 6
            player_indices: Dict[int, List[int]] = {pid: [] for pid in range(6)}
            survival = [0] * 6

            while not env.done:
                actions = [0] * 6
                for pid in range(6):
                    if not env.players[pid].alive:
                        continue
                    feat = extract_features(env, pid, prev_actions[pid], schema).astype(np.float32)
                    xb = torch.from_numpy(feat).to(device).unsqueeze(0)
                    with torch.no_grad():
                        logits, value = model.forward_with_value(xb)
                        dist = Categorical(logits=logits)
                        action_idx = int(dist.sample().item())
                        logp = float(dist.log_prob(torch.tensor([action_idx], device=device)).item())
                        v = float(value.squeeze(0).item())
                    turn = int(INDEX_TO_TURN[action_idx])
                    actions[pid] = turn
                    prev_actions[pid] = turn

                    batch_obs.append(feat.astype(np.float32))
                    batch_actions.append(action_idx)
                    batch_old_logp.append(logp)
                    batch_values.append(v)
                    batch_rewards.append(float(cfg.survival_alpha))
                    batch_dones.append(0.0)
                    player_indices[pid].append(len(batch_obs) - 1)
                    survival[pid] += 1

                env.step(actions)

            final_ranks = env.compute_ranks()
            for pid in range(6):
                idxs = player_indices[pid]
                if not idxs:
                    continue
                last = idxs[-1]
                rank = float(final_ranks[pid])
                # Default rank objective used by the main trainer.
                if rank <= 1.0:
                    terminal = 1.0
                elif rank <= 2.0:
                    terminal = 0.60
                elif rank <= 3.0:
                    terminal = 0.25
                elif rank <= 4.0:
                    terminal = 0.00
                elif rank <= 5.0:
                    terminal = -0.25
                else:
                    terminal = -0.60
                batch_rewards[last] += float(terminal)
                batch_dones[last] = 1.0
                ep_ranks.append(float(final_ranks[pid]))
                ep_surv.append(float(survival[pid]))

        if not batch_obs:
            continue

        obs_t = torch.from_numpy(np.stack(batch_obs).astype(np.float32)).to(device)
        actions_t = torch.from_numpy(np.asarray(batch_actions, dtype=np.int64)).to(device)
        old_logp_t = torch.from_numpy(np.asarray(batch_old_logp, dtype=np.float32)).to(device)
        values_np = np.asarray(batch_values, dtype=np.float32)
        rewards_np = np.asarray(batch_rewards, dtype=np.float32)
        dones_np = np.asarray(batch_dones, dtype=np.float32)

        adv_np, ret_np = _compute_gae(
            rewards_np,
            values_np,
            dones_np,
            gamma=float(cfg.gamma),
            lam=float(cfg.gae_lambda),
        )
        adv_t = torch.from_numpy(adv_np).to(device)
        ret_t = torch.from_numpy(ret_np).to(device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        idx_all = np.arange(obs_t.shape[0], dtype=np.int32)
        total_loss = 0.0
        total_policy = 0.0
        total_value = 0.0
        total_entropy = 0.0
        num_batches = 0

        for _ in range(max(1, int(cfg.ppo_epochs))):
            rng.shuffle(idx_all)
            mb = max(32, int(cfg.minibatch_size))
            for start in range(0, obs_t.shape[0], mb):
                ids = idx_all[start : start + mb]
                mb_obs = obs_t[ids]
                mb_actions = actions_t[ids]
                mb_old_logp = old_logp_t[ids]
                mb_adv = adv_t[ids]
                mb_ret = ret_t[ids]

                logits, value_pred = model.forward_with_value(mb_obs)
                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logp - mb_old_logp)
                s1 = ratio * mb_adv
                s2 = torch.clamp(ratio, 1.0 - float(cfg.clip_ratio), 1.0 + float(cfg.clip_ratio)) * mb_adv
                policy_loss = -torch.min(s1, s2).mean()
                value_loss = F.mse_loss(value_pred, mb_ret)
                loss = policy_loss + (float(cfg.value_coef) * value_loss) - (float(cfg.entropy_coef) * entropy)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.max_grad_norm))
                optimizer.step()

                total_loss += float(loss.item())
                total_policy += float(policy_loss.item())
                total_value += float(value_loss.item())
                total_entropy += float(entropy.item())
                num_batches += 1

        history.append(
            {
                "update": float(update_idx),
                "loss": float(total_loss / max(1, num_batches)),
                "policy_loss": float(total_policy / max(1, num_batches)),
                "value_loss": float(total_value / max(1, num_batches)),
                "entropy": float(total_entropy / max(1, num_batches)),
                "avg_rank": float(np.mean(ep_ranks)) if ep_ranks else 0.0,
                "avg_survival_steps": float(np.mean(ep_surv)) if ep_surv else 0.0,
            }
        )

    model.eval()
    return {"updates": int(cfg.updates), "history": history}


@dataclass
class QuantizedCompactPolicy:
    input_dim: int
    hidden1: int
    hidden2: int
    scale: int
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    w3: np.ndarray
    b3: np.ndarray

    @staticmethod
    def _div_trunc(a: int, b: int) -> int:
        if b == 0:
            return 0
        return int(a / b)

    def _dense_relu(
        self,
        x: Sequence[int],
        w: np.ndarray,
        b: np.ndarray,
    ) -> List[int]:
        out: List[int] = []
        for row_idx in range(w.shape[0]):
            acc = int(b[row_idx])
            row = w[row_idx]
            for col_idx in range(w.shape[1]):
                acc += int(row[col_idx]) * int(x[col_idx])
            acc = self._div_trunc(acc, self.scale)
            if acc < 0:
                acc = 0
            out.append(int(acc))
        return out

    def _dense_linear(
        self,
        x: Sequence[int],
        w: np.ndarray,
        b: np.ndarray,
    ) -> List[int]:
        out: List[int] = []
        for row_idx in range(w.shape[0]):
            acc = int(b[row_idx])
            row = w[row_idx]
            for col_idx in range(w.shape[1]):
                acc += int(row[col_idx]) * int(x[col_idx])
            acc = self._div_trunc(acc, self.scale)
            out.append(int(acc))
        return out

    def logits(self, row: np.ndarray) -> Tuple[int, int, int]:
        if row.shape[0] != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {row.shape[0]}")
        x = [int(v) for v in row]
        h1 = self._dense_relu(x, self.w1, self.b1)
        h2 = self._dense_relu(h1, self.w2, self.b2)
        out = self._dense_linear(h2, self.w3, self.b3)
        return int(out[0]), int(out[1]), int(out[2])

    def predict_with_scores_one(self, row: np.ndarray) -> Tuple[int, Dict[int, int]]:
        log_left, log_fwd, log_right = self.logits(row)
        scores = {-1: int(log_left), 0: int(log_fwd), 1: int(log_right)}
        best = 0
        best_score = scores[0]
        if scores[-1] > best_score:
            best = -1
            best_score = scores[-1]
        if scores[1] > best_score:
            best = 1
        return int(best), scores

    def predict_one(self, row: np.ndarray) -> int:
        action, _ = self.predict_with_scores_one(row)
        return int(action)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_dim": int(self.input_dim),
            "hidden1": int(self.hidden1),
            "hidden2": int(self.hidden2),
            "scale": int(self.scale),
            "w1": self.w1.astype(int).tolist(),
            "b1": self.b1.astype(int).tolist(),
            "w2": self.w2.astype(int).tolist(),
            "b2": self.b2.astype(int).tolist(),
            "w3": self.w3.astype(int).tolist(),
            "b3": self.b3.astype(int).tolist(),
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "QuantizedCompactPolicy":
        return QuantizedCompactPolicy(
            input_dim=int(data["input_dim"]),
            hidden1=int(data["hidden1"]),
            hidden2=int(data["hidden2"]),
            scale=int(data["scale"]),
            w1=np.asarray(data["w1"], dtype=np.int32),
            b1=np.asarray(data["b1"], dtype=np.int32),
            w2=np.asarray(data["w2"], dtype=np.int32),
            b2=np.asarray(data["b2"], dtype=np.int32),
            w3=np.asarray(data["w3"], dtype=np.int32),
            b3=np.asarray(data["b3"], dtype=np.int32),
        )


def quantize_student_model(model: CompactStudentNet, scale: int) -> QuantizedCompactPolicy:
    if scale <= 0:
        raise ValueError("scale must be > 0")
    with torch.no_grad():
        w1 = np.rint(model.fc1.weight.detach().cpu().numpy() * scale).astype(np.int32)
        b1 = np.rint(model.fc1.bias.detach().cpu().numpy() * scale).astype(np.int32)
        w2 = np.rint(model.fc2.weight.detach().cpu().numpy() * scale).astype(np.int32)
        b2 = np.rint(model.fc2.bias.detach().cpu().numpy() * scale).astype(np.int32)
        w3 = np.rint(model.fc_out.weight.detach().cpu().numpy() * scale).astype(np.int32)
        b3 = np.rint(model.fc_out.bias.detach().cpu().numpy() * scale).astype(np.int32)
    return QuantizedCompactPolicy(
        input_dim=model.input_dim,
        hidden1=model.hidden1,
        hidden2=model.hidden2,
        scale=int(scale),
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        w3=w3,
        b3=b3,
    )


def _emit_tile_helpers(lines: List[str]) -> None:
    lines.append("my_id = getPlayerId()")
    lines.append("mem_prev_action = 0")
    lines.append("")
    lines.append("function tile_code(q, r) do")
    lines.append("    exists, isEmpty, playerId, isGem = getTileRel(q, r)")
    lines.append("    if (exists == 0) then")
    lines.append("        value = -2")
    lines.append("    else if (isGem == 1) then")
    lines.append("        value = -1")
    lines.append("    else if (isEmpty == 1) then")
    lines.append("        value = 0")
    lines.append("    else if (playerId == my_id) then")
    lines.append("        value = 2")
    lines.append("    else")
    lines.append("        value = 1")
    lines.append("    end if")
    lines.append("    return value")
    lines.append("end function")
    lines.append("")
    lines.append("function wait_for_next_move() do")
    lines.append("    tick_now = getTick()")
    lines.append("    rem = tick_now % 10000")
    lines.append("    if (rem == 0) then")
    lines.append("        wait_ticks = 50")
    lines.append("    else")
    lines.append("        wait_ticks = 10000 - rem")
    lines.append("        wait_ticks = wait_ticks + 50")
    lines.append("    end if")
    lines.append("    wait(wait_ticks)")
    lines.append("end function")
    lines.append("")


def _emit_feature_load(lines: List[str], schema: FeatureSchema) -> List[str]:
    feature_vars: List[str] = []
    idx = 0
    for q, r in schema.tile_coords:
        var = f"f{idx}"
        lines.append(f"    {var} = tile_code({q}, {r})")
        feature_vars.append(var)
        idx += 1

    if schema.include_turn:
        var = f"f{idx}"
        lines.append(f"    {var} = getTurn()")
        feature_vars.append(var)
        idx += 1

    if schema.include_length:
        var = f"f{idx}"
        lines.append("    alive, headQ, headR, headFacing, myLength = getPlayerInfo(my_id)")
        lines.append(f"    {var} = myLength")
        lines.append(f"    if ({var} > {schema.length_cap}) then")
        lines.append(f"        {var} = {schema.length_cap}")
        lines.append("    end if")
        feature_vars.append(var)
        idx += 1

    if schema.include_prev_action:
        var = f"f{idx}"
        lines.append(f"    {var} = mem_prev_action")
        feature_vars.append(var)
        idx += 1

    if schema.include_player_info:
        for pid in range(6):
            lines.append(f"    p{pid}_alive, p{pid}_q, p{pid}_r, p{pid}_facing, p{pid}_length = getPlayerInfo({pid})")
            var_alive = f"f{idx}"
            lines.append(f"    {var_alive} = p{pid}_alive")
            feature_vars.append(var_alive)
            idx += 1

            var_q = f"f{idx}"
            lines.append(f"    {var_q} = p{pid}_q")
            feature_vars.append(var_q)
            idx += 1

            var_r = f"f{idx}"
            lines.append(f"    {var_r} = p{pid}_r")
            feature_vars.append(var_r)
            idx += 1

            var_face = f"f{idx}"
            lines.append(f"    {var_face} = p{pid}_facing")
            feature_vars.append(var_face)
            idx += 1

            var_len = f"f{idx}"
            lines.append(f"    {var_len} = p{pid}_length")
            lines.append(f"    if ({var_len} > {schema.length_cap}) then")
            lines.append(f"        {var_len} = {schema.length_cap}")
            lines.append("    end if")
            feature_vars.append(var_len)
            idx += 1
    return feature_vars


def _emit_safety_fallback(lines: List[str]) -> None:
    lines.append("    safe_left = 0")
    lines.append("    safe_fwd = 0")
    lines.append("    safe_right = 0")
    lines.append("    eL, emL, pL, gL = getTileRel(-1, 0)")
    lines.append("    if (eL == 1) then")
    lines.append("        if (emL == 1) then")
    lines.append("            safe_left = 1")
    lines.append("        else if (gL == 1) then")
    lines.append("            safe_left = 1")
    lines.append("        end if")
    lines.append("    end if")
    lines.append("    eF, emF, pF, gF = getTileRel(0, -1)")
    lines.append("    if (eF == 1) then")
    lines.append("        if (emF == 1) then")
    lines.append("            safe_fwd = 1")
    lines.append("        else if (gF == 1) then")
    lines.append("            safe_fwd = 1")
    lines.append("        end if")
    lines.append("    end if")
    lines.append("    eR, emR, pR, gR = getTileRel(1, -1)")
    lines.append("    if (eR == 1) then")
    lines.append("        if (emR == 1) then")
    lines.append("            safe_right = 1")
    lines.append("        else if (gR == 1) then")
    lines.append("            safe_right = 1")
    lines.append("        end if")
    lines.append("    end if")
    lines.append("    chosen_safe = 0")
    lines.append("    if (action == -1) then")
    lines.append("        if (safe_left == 1) then")
    lines.append("            chosen_safe = 1")
    lines.append("        end if")
    lines.append("    else if (action == 0) then")
    lines.append("        if (safe_fwd == 1) then")
    lines.append("            chosen_safe = 1")
    lines.append("        end if")
    lines.append("    else if (action == 1) then")
    lines.append("        if (safe_right == 1) then")
    lines.append("            chosen_safe = 1")
    lines.append("        end if")
    lines.append("    end if")
    lines.append("    if (chosen_safe == 0) then")
    lines.append("        best_action = action")
    lines.append("        best_score = -999999")
    lines.append("        if (safe_fwd == 1) then")
    lines.append("            best_action = 0")
    lines.append("            best_score = score_fwd")
    lines.append("        end if")
    lines.append("        if (safe_left == 1) then")
    lines.append("            if (score_left > best_score) then")
    lines.append("                best_action = -1")
    lines.append("                best_score = score_left")
    lines.append("            end if")
    lines.append("        end if")
    lines.append("        if (safe_right == 1) then")
    lines.append("            if (score_right > best_score) then")
    lines.append("                best_action = 1")
    lines.append("                best_score = score_right")
    lines.append("            end if")
    lines.append("        end if")
    lines.append("        if (best_score > -999999) then")
    lines.append("            action = best_action")
    lines.append("        end if")
    lines.append("    end if")


def _emit_weight_add(lines: List[str], target: str, source_var: str, weight: int) -> None:
    w = int(weight)
    if w == 0:
        return
    if w > 0:
        lines.append(f"    tmp = {source_var} * {w}")
        lines.append(f"    {target} = {target} + tmp")
        return
    lines.append(f"    tmp = {source_var} * {abs(w)}")
    lines.append(f"    {target} = {target} - tmp")


def generate_nn_tronkscript(
    model: QuantizedCompactPolicy,
    schema: FeatureSchema,
    *,
    include_main: bool,
    infer_function_name: str = "infer_action",
    wait_ticks: int = 1200,
    safety_fallback: bool = True,
) -> str:
    lines: List[str] = []
    lines.append("-- Auto-generated quantized compact neural policy.")
    lines.append("-- Integer-only fixed-point inference for Tronkscript.")
    lines.append(f"-- hidden={model.hidden1}x{model.hidden2}, scale={model.scale}")
    _emit_tile_helpers(lines)

    lines.append(f"function {infer_function_name}() do")
    feature_vars = _emit_feature_load(lines, schema)
    if len(feature_vars) != model.input_dim:
        raise ValueError(f"Feature/model mismatch: feature_dim={len(feature_vars)} input_dim={model.input_dim}")

    h1_vars: List[str] = []
    for i in range(model.hidden1):
        hv = f"h1_{i}"
        h1_vars.append(hv)
        lines.append(f"    {hv} = {int(model.b1[i])}")
        for j, fv in enumerate(feature_vars):
            _emit_weight_add(lines, hv, fv, int(model.w1[i, j]))
        lines.append(f"    {hv} = {hv} / {int(model.scale)}")
        lines.append(f"    if ({hv} < 0) then")
        lines.append(f"        {hv} = 0")
        lines.append("    end if")

    h2_vars: List[str] = []
    for i in range(model.hidden2):
        hv = f"h2_{i}"
        h2_vars.append(hv)
        lines.append(f"    {hv} = {int(model.b2[i])}")
        for j, h1v in enumerate(h1_vars):
            _emit_weight_add(lines, hv, h1v, int(model.w2[i, j]))
        lines.append(f"    {hv} = {hv} / {int(model.scale)}")
        lines.append(f"    if ({hv} < 0) then")
        lines.append(f"        {hv} = 0")
        lines.append("    end if")

    lines.append(f"    log_left = {int(model.b3[0])}")
    for j, h2v in enumerate(h2_vars):
        _emit_weight_add(lines, "log_left", h2v, int(model.w3[0, j]))
    lines.append(f"    log_left = log_left / {int(model.scale)}")

    lines.append(f"    log_fwd = {int(model.b3[1])}")
    for j, h2v in enumerate(h2_vars):
        _emit_weight_add(lines, "log_fwd", h2v, int(model.w3[1, j]))
    lines.append(f"    log_fwd = log_fwd / {int(model.scale)}")

    lines.append(f"    log_right = {int(model.b3[2])}")
    for j, h2v in enumerate(h2_vars):
        _emit_weight_add(lines, "log_right", h2v, int(model.w3[2, j]))
    lines.append(f"    log_right = log_right / {int(model.scale)}")

    lines.append("    score_left = log_left")
    lines.append("    score_fwd = log_fwd")
    lines.append("    score_right = log_right")

    lines.append("    action = 0")
    lines.append("    best_score = log_fwd")
    lines.append("    if (log_left > best_score) then")
    lines.append("        action = -1")
    lines.append("        best_score = log_left")
    lines.append("    end if")
    lines.append("    if (log_right > best_score) then")
    lines.append("        action = 1")
    lines.append("        best_score = log_right")
    lines.append("    end if")

    if safety_fallback:
        _emit_safety_fallback(lines)

    lines.append("    return action")
    lines.append("end function")

    if include_main:
        lines.append("")
        lines.append("function main() do")
        lines.append("    while (1 == 1) do")
        lines.append(f"        action = {infer_function_name}()")
        lines.append("        turn(action)")
        lines.append("        mem_prev_action = action")
        lines.append("        wait_for_next_move()")
        lines.append("    end while")
        lines.append("end function")
        lines.append("")
        lines.append("main()")

    return "\n".join(lines) + "\n"


def parse_tronkscript_source(source: str) -> None:
    parser = TronkscriptParser(source)
    parser.parse()


def _is_walkable_tile(tile: Tuple[int, int, int, int]) -> bool:
    exists, is_empty, _occ_pid, is_gem = tile
    return int(exists) == 1 and (int(is_empty) == 1 or int(is_gem) == 1)


def apply_safety_fallback(
    env: MLTronkEnv,
    player_id: int,
    action: int,
    scores: Dict[int, float],
) -> int:
    safe = {
        -1: _is_walkable_tile(env.get_tile_rel(player_id, -1, 0)),
        0: _is_walkable_tile(env.get_tile_rel(player_id, 0, -1)),
        1: _is_walkable_tile(env.get_tile_rel(player_id, 1, -1)),
    }
    if safe.get(action, False):
        return int(action)

    best_action: Optional[int] = None
    best_score = -10**9
    for candidate in (0, -1, 1):
        if not safe[candidate]:
            continue
        candidate_score = float(scores.get(candidate, 0.0))
        if candidate_score > best_score:
            best_score = candidate_score
            best_action = candidate
    if best_action is None:
        return int(action)
    return int(best_action)


def evaluate_conversion_parity(
    *,
    tronkscript_source_no_main: str,
    predictor: QuantizedCompactPolicy,
    schema: FeatureSchema,
    samples: int,
    seed: int,
    max_steps: int,
    use_c_core: bool,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    ts_policy = TronkscriptInferencePolicy(tronkscript_source_no_main, infer_function_name="infer_action")

    total = 0
    matches = 0
    tick_cost_sum = 0
    tick_cost_max = 0

    while total < samples:
        env = MLTronkEnv(
            seed=int(rng.integers(0, 2_000_000_000)),
            max_steps=max(1, int(max_steps)),
            randomize_starts=True,
            randomize_facings=True,
            use_c_core=use_c_core,
            require_c_core=False,
        )
        ts_policy.reset(seed=int(rng.integers(0, 2_000_000_000)), max_steps=max_steps, use_c_core=use_c_core)
        prev_actions = [0] * 6

        while not env.done and total < samples:
            actions = [0] * 6
            for pid in range(6):
                if not env.players[pid].alive:
                    continue

                feat = extract_features(env, pid, prev_actions[pid], schema)
                py_action_raw, py_scores = predictor.predict_with_scores_one(feat)
                py_action = apply_safety_fallback(env, pid, py_action_raw, py_scores)

                ts_action, ts_cost = ts_policy.act(env, pid)
                actions[pid] = int(ts_action)
                prev_actions[pid] = int(ts_action)

                if int(py_action) == int(ts_action):
                    matches += 1
                total += 1
                tick_cost_sum += int(ts_cost)
                tick_cost_max = max(tick_cost_max, int(ts_cost))
                if total >= samples:
                    break

            env.step(actions)

    return {
        "samples": float(total),
        "parity_rate": float(matches / max(1, total)),
        "avg_tick_cost": float(tick_cost_sum / max(1, total)),
        "max_tick_cost": float(tick_cost_max),
    }


@dataclass
class CandidateResult:
    hidden1: int
    hidden2: int
    train_acc: float
    test_acc: float
    parity_rate: float
    avg_tick_cost: float
    max_tick_cost: float
    fits_tick_budget: bool
    parse_ok: bool
    lib_source: str
    bot_source: str
    model_float: CompactStudentNet
    model_quantized: QuantizedCompactPolicy
    train_history: List[Dict[str, float]]

    def to_summary(self) -> Dict[str, Any]:
        return {
            "hidden1": int(self.hidden1),
            "hidden2": int(self.hidden2),
            "train_acc": float(self.train_acc),
            "test_acc": float(self.test_acc),
            "parity_rate": float(self.parity_rate),
            "avg_tick_cost": float(self.avg_tick_cost),
            "max_tick_cost": float(self.max_tick_cost),
            "fits_tick_budget": bool(self.fits_tick_budget),
            "parse_ok": bool(self.parse_ok),
        }


def fit_candidate(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    logits_train: Optional[np.ndarray],
    x_test: np.ndarray,
    y_test: np.ndarray,
    logits_test: Optional[np.ndarray],
    hidden1: int,
    hidden2: int,
    train_cfg: TrainConfig,
    distill_cfg: Optional[DistillConfig],
    quant_scale: int,
    schema: FeatureSchema,
    wait_ticks: int,
    parity_samples: int,
    parity_seed: int,
    parity_max_steps: int,
    use_c_core: bool,
    tick_budget: float,
    device: torch.device,
) -> CandidateResult:
    model_float, metrics = train_student_model(
        x_train,
        y_train,
        x_test,
        y_test,
        hidden1=hidden1,
        hidden2=hidden2,
        cfg=train_cfg,
        distill_cfg=distill_cfg,
        teacher_logits_train=logits_train,
        teacher_logits_test=logits_test,
        device=device,
    )
    model_quant = quantize_student_model(model_float, scale=quant_scale)

    lib_code = generate_nn_tronkscript(
        model_quant,
        schema,
        include_main=False,
        wait_ticks=wait_ticks,
        safety_fallback=True,
    )
    bot_code = generate_nn_tronkscript(
        model_quant,
        schema,
        include_main=True,
        wait_ticks=wait_ticks,
        safety_fallback=True,
    )

    parse_ok = True
    try:
        parse_tronkscript_source(lib_code)
        parse_tronkscript_source(bot_code)
    except Exception:
        parse_ok = False

    if parse_ok:
        parity = evaluate_conversion_parity(
            tronkscript_source_no_main=lib_code,
            predictor=model_quant,
            schema=schema,
            samples=max(100, int(parity_samples)),
            seed=int(parity_seed),
            max_steps=max(20, int(parity_max_steps)),
            use_c_core=use_c_core,
        )
    else:
        parity = {"parity_rate": 0.0, "avg_tick_cost": 1e9, "max_tick_cost": 1e9}

    return CandidateResult(
        hidden1=int(hidden1),
        hidden2=int(hidden2),
        train_acc=float(metrics["train_acc"]),
        test_acc=float(metrics["test_acc"]),
        parity_rate=float(parity["parity_rate"]),
        avg_tick_cost=float(parity["avg_tick_cost"]),
        max_tick_cost=float(parity["max_tick_cost"]),
        fits_tick_budget=bool(float(parity["avg_tick_cost"]) <= float(tick_budget)),
        parse_ok=bool(parse_ok),
        lib_source=lib_code,
        bot_source=bot_code,
        model_float=model_float,
        model_quantized=model_quant,
        train_history=[dict(x) for x in metrics["history"]],
    )


def build_candidate_from_model(
    *,
    model_float: CompactStudentNet,
    hidden1: int,
    hidden2: int,
    train_acc: float,
    test_acc: float,
    train_history: Sequence[Dict[str, float]],
    quant_scale: int,
    schema: FeatureSchema,
    wait_ticks: int,
    parity_samples: int,
    parity_seed: int,
    parity_max_steps: int,
    use_c_core: bool,
    tick_budget: float,
) -> CandidateResult:
    model_quant = quantize_student_model(model_float, scale=quant_scale)

    lib_code = generate_nn_tronkscript(
        model_quant,
        schema,
        include_main=False,
        wait_ticks=wait_ticks,
        safety_fallback=True,
    )
    bot_code = generate_nn_tronkscript(
        model_quant,
        schema,
        include_main=True,
        wait_ticks=wait_ticks,
        safety_fallback=True,
    )

    parse_ok = True
    try:
        parse_tronkscript_source(lib_code)
        parse_tronkscript_source(bot_code)
    except Exception:
        parse_ok = False

    if parse_ok:
        parity = evaluate_conversion_parity(
            tronkscript_source_no_main=lib_code,
            predictor=model_quant,
            schema=schema,
            samples=max(100, int(parity_samples)),
            seed=int(parity_seed),
            max_steps=max(20, int(parity_max_steps)),
            use_c_core=use_c_core,
        )
    else:
        parity = {"parity_rate": 0.0, "avg_tick_cost": 1e9, "max_tick_cost": 1e9}

    return CandidateResult(
        hidden1=int(hidden1),
        hidden2=int(hidden2),
        train_acc=float(train_acc),
        test_acc=float(test_acc),
        parity_rate=float(parity["parity_rate"]),
        avg_tick_cost=float(parity["avg_tick_cost"]),
        max_tick_cost=float(parity["max_tick_cost"]),
        fits_tick_budget=bool(float(parity["avg_tick_cost"]) <= float(tick_budget)),
        parse_ok=bool(parse_ok),
        lib_source=lib_code,
        bot_source=bot_code,
        model_float=model_float,
        model_quantized=model_quant,
        train_history=[dict(x) for x in train_history],
    )


def choose_best_candidate(candidates: Sequence[CandidateResult]) -> CandidateResult:
    if not candidates:
        raise ValueError("No candidates to choose from")

    valid = [c for c in candidates if c.parse_ok]
    if not valid:
        return candidates[0]

    budget_valid = [c for c in valid if c.fits_tick_budget]
    pool = budget_valid if budget_valid else valid

    # Primary: higher test accuracy, then higher parity, then lower avg ticks.
    best = max(pool, key=lambda c: (c.test_acc, c.parity_rate, -c.avg_tick_cost))
    return best


def write_nn_artifacts(
    *,
    output_dir: Path,
    selected: CandidateResult,
    schema: FeatureSchema,
    wait_ticks: int,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    float_path = output_dir / "nn_student_float.pt"
    quant_path = output_dir / "nn_student_quantized.json"
    lib_path = output_dir / "nn_policy_lib.tronkscript"
    bot_path = output_dir / "nn_policy_bot.tronkscript"

    torch.save(selected.model_float.state_dict(), float_path)

    quant_payload = {
        "model_type": "quantized_compact_nn",
        "quantized_model": selected.model_quantized.to_dict(),
        "schema": {
            "tile_coords": [[int(q), int(r)] for q, r in schema.tile_coords],
            "include_turn": bool(schema.include_turn),
            "include_length": bool(schema.include_length),
            "include_prev_action": bool(schema.include_prev_action),
            "include_player_info": bool(schema.include_player_info),
            "length_cap": int(schema.length_cap),
            "names": schema.names,
        },
        "wait_ticks": int(wait_ticks),
    }
    quant_path.write_text(json.dumps(quant_payload, indent=2), encoding="utf-8")

    lib_path.write_text(selected.lib_source, encoding="utf-8")
    bot_path.write_text(selected.bot_source, encoding="utf-8")
    return {
        "float_model": float_path,
        "quantized_model": quant_path,
        "lib_tronkscript": lib_path,
        "bot_tronkscript": bot_path,
    }


def run_final_benchmark(
    *,
    policy_path: Path,
    selected: CandidateResult,
    schema: FeatureSchema,
    matches: int,
    seed: int,
    max_steps: int,
    use_c_core: bool,
    require_c_core: bool,
) -> Dict[str, Any]:
    return benchmark_tronkscript_vs_model(
        policy_path=policy_path,
        tronkscript_source_no_main=selected.lib_source,
        schema=schema,
        predictor=selected.model_quantized,
        matches=max(1, int(matches)),
        seed=int(seed),
        max_steps=max(1, int(max_steps)),
        use_c_core=use_c_core,
        require_c_core=bool(require_c_core),
    )


def run_bot_validation(source_with_main: str, *, steps: int, seeds: Iterable[int]) -> Dict[str, Any]:
    return validate_tronkscript_bot(source_with_main, steps=max(1, int(steps)), seeds=seeds)
