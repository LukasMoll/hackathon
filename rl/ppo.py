from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from rl.config import PPOConfig
from rl.policy import RecurrentActorCritic


@dataclass
class PPOBatch:
    obs: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[float] = field(default_factory=list)
    hiddens: List[np.ndarray] = field(default_factory=list)

    def add(
        self,
        obs: np.ndarray,
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
        hidden: np.ndarray,
    ) -> None:
        self.obs.append(obs)
        self.actions.append(int(action))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.dones.append(1.0 if done else 0.0)
        self.hiddens.append(hidden.astype(np.float32))

    def __len__(self) -> int:
        return len(self.obs)

    def extend(self, other: "PPOBatch") -> None:
        self.obs.extend(other.obs)
        self.actions.extend(other.actions)
        self.log_probs.extend(other.log_probs)
        self.values.extend(other.values)
        self.rewards.extend(other.rewards)
        self.dones.extend(other.dones)
        self.hiddens.extend(other.hiddens)


@dataclass
class PPOStats:
    loss: float
    policy_loss: float
    value_loss: float
    entropy: float


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lam: float,
) -> Dict[str, np.ndarray]:
    n = len(rewards)
    adv = np.zeros(n, dtype=np.float32)
    gae = 0.0
    next_value = 0.0

    for t in reversed(range(n)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        adv[t] = gae
        next_value = values[t]

    returns = adv + values
    return {"advantages": adv, "returns": returns}


def ppo_update(
    policy: RecurrentActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: PPOBatch,
    cfg: PPOConfig,
    device: torch.device,
) -> PPOStats:
    obs = torch.from_numpy(np.stack(batch.obs).astype(np.float32)).to(device)
    actions = torch.from_numpy(np.array(batch.actions, dtype=np.int64)).to(device)
    old_log_probs = torch.from_numpy(np.array(batch.log_probs, dtype=np.float32)).to(device)
    values = np.array(batch.values, dtype=np.float32)
    rewards = np.array(batch.rewards, dtype=np.float32)
    dones = np.array(batch.dones, dtype=np.float32)
    hiddens = torch.from_numpy(np.stack(batch.hiddens).astype(np.float32)).to(device)

    gae = compute_gae(rewards, values, dones, cfg.gamma, cfg.gae_lambda)
    advantages = torch.from_numpy(gae["advantages"]).to(device)
    returns = torch.from_numpy(gae["returns"]).to(device)

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    n = obs.shape[0]
    idx_all = np.arange(n)

    total_loss = 0.0
    total_policy = 0.0
    total_value = 0.0
    total_entropy = 0.0
    count = 0

    for _ in range(cfg.epochs):
        np.random.shuffle(idx_all)
        for start in range(0, n, cfg.minibatch_size):
            end = min(start + cfg.minibatch_size, n)
            mb_idx = idx_all[start:end]
            mb_obs = obs[mb_idx]
            mb_actions = actions[mb_idx]
            mb_old_log_probs = old_log_probs[mb_idx]
            mb_adv = advantages[mb_idx]
            mb_returns = returns[mb_idx]
            mb_hidden = hiddens[mb_idx].unsqueeze(0)

            logits, value_pred, _ = policy(mb_obs, mb_hidden)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(mb_actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - mb_old_log_probs)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(value_pred, mb_returns)
            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()

            total_loss += float(loss.item())
            total_policy += float(policy_loss.item())
            total_value += float(value_loss.item())
            total_entropy += float(entropy.item())
            count += 1

    if count == 0:
        return PPOStats(loss=0.0, policy_loss=0.0, value_loss=0.0, entropy=0.0)

    return PPOStats(
        loss=total_loss / count,
        policy_loss=total_policy / count,
        value_loss=total_value / count,
        entropy=total_entropy / count,
    )
