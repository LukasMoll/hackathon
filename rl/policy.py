from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class RecurrentActorCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 128, action_dim: int = 3):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(obs_dim, 128)
        self.gru = nn.GRU(128, hidden_dim, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, 128)
        self.policy_head = nn.Linear(128, action_dim)
        self.value_head = nn.Linear(128, 1)

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(1, batch_size, self.hidden_dim, device=device)

    def forward(self, obs: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # obs: [B, obs_dim], h: [1, B, hidden_dim]
        x = torch.relu(self.fc1(obs))
        x = x.unsqueeze(1)
        out, h_new = self.gru(x, h)
        out = out.squeeze(1)
        out = torch.relu(self.fc2(out))
        logits = self.policy_head(out)
        value = self.value_head(out).squeeze(-1)
        return logits, value, h_new

    @torch.no_grad()
    def act(
        self,
        obs_np: np.ndarray,
        h_np: np.ndarray,
        device: torch.device,
        deterministic: bool = False,
    ) -> Tuple[int, float, float, np.ndarray]:
        obs = torch.from_numpy(obs_np.astype(np.float32)).to(device).unsqueeze(0)
        h = torch.from_numpy(h_np.astype(np.float32)).to(device)
        if h.dim() == 1:
            h = h.unsqueeze(0).unsqueeze(0)
        elif h.dim() == 2:
            h = h.unsqueeze(0)
        elif h.dim() != 3:
            raise ValueError(f"Unexpected hidden state shape: {tuple(h.shape)}")
        logits, value, h_new = self.forward(obs, h)
        dist = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return (
            int(action.item()),
            float(log_prob.item()),
            float(value.item()),
            h_new.squeeze(0).squeeze(0).cpu().numpy(),
        )


class PlanningActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        mem_dim: int = 16,
        hidden_dim: int = 128,
        action_dim: int = 3,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = mem_dim
        self.mem_dim = mem_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(obs_dim + mem_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

        self.mem_head = nn.Linear(hidden_dim, mem_dim)
        self.mem_gate_head = nn.Linear(hidden_dim, mem_dim)

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(1, batch_size, self.mem_dim, device=device)

    def forward(self, obs: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # obs: [B, obs_dim], h: [1, B, mem_dim] (or [B, mem_dim])
        if h.dim() == 3:
            mem = h.squeeze(0)
        elif h.dim() == 2:
            mem = h
        elif h.dim() == 1:
            mem = h.unsqueeze(0)
        else:
            raise ValueError(f"Unexpected hidden state shape: {tuple(h.shape)}")

        x = torch.cat([obs, mem], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)

        mem_candidate = torch.tanh(self.mem_head(x))
        mem_gate = torch.sigmoid(self.mem_gate_head(x))
        mem_next = ((1.0 - mem_gate) * mem) + (mem_gate * mem_candidate)
        return logits, value, mem_next.unsqueeze(0)

    @torch.no_grad()
    def act(
        self,
        obs_np: np.ndarray,
        h_np: np.ndarray,
        device: torch.device,
        deterministic: bool = False,
    ) -> Tuple[int, float, float, np.ndarray]:
        obs = torch.from_numpy(obs_np.astype(np.float32)).to(device).unsqueeze(0)
        h = torch.from_numpy(h_np.astype(np.float32)).to(device)
        if h.dim() == 1:
            h = h.unsqueeze(0).unsqueeze(0)
        elif h.dim() == 2:
            h = h.unsqueeze(0)
        elif h.dim() != 3:
            raise ValueError(f"Unexpected hidden state shape: {tuple(h.shape)}")
        logits, value, h_new = self.forward(obs, h)
        dist = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return (
            int(action.item()),
            float(log_prob.item()),
            float(value.item()),
            h_new.squeeze(0).squeeze(0).cpu().numpy(),
        )


def build_policy(
    arch: str,
    obs_dim: int,
    *,
    hidden_dim: int = 128,
    mem_dim: int = 16,
    action_dim: int = 3,
) -> nn.Module:
    key = arch.strip().lower()
    if key == "gru":
        return RecurrentActorCritic(obs_dim=obs_dim, hidden_dim=hidden_dim, action_dim=action_dim)
    if key == "planning":
        return PlanningActorCritic(
            obs_dim=obs_dim,
            mem_dim=mem_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
        )
    raise ValueError(f"Unsupported policy architecture '{arch}'")


@dataclass
class ScriptedActionResult:
    action_index: int


class ScriptedPolicy:
    name = "scripted"

    def act(self, env, player_id: int) -> int:
        raise NotImplementedError


class RandomScriptedPolicy(ScriptedPolicy):
    name = "scripted_random"

    def act(self, env, player_id: int) -> int:
        return int(env.random.choice([0, 1, 2]))


class SafeGreedyScriptedPolicy(ScriptedPolicy):
    name = "scripted_safe_greedy"

    @staticmethod
    def _action_to_turn(action_index: int) -> int:
        return [-1, 0, 1][action_index]

    @staticmethod
    def _turn_to_action(turn: int) -> int:
        return {-1: 0, 0: 1, 1: 2}[turn]

    def act(self, env, player_id: int) -> int:
        p = env.players[player_id]
        if not p.alive:
            return 1

        safe_actions = []
        gem_actions = []

        for action_index, turn in enumerate([-1, 0, 1]):
            nf = env.wrap_facing(p.facing + turn)
            dq, dr = [(0, -1), (1, -1), (1, 0), (0, 1), (-1, 1), (-1, 0)][nf]
            tq, tr = p.head[0] + dq, p.head[1] + dr
            exists, is_empty, _, has_gem = env.get_tile_abs(tq, tr)
            if exists == 0:
                continue
            if is_empty == 0 and has_gem == 0:
                continue
            safe_actions.append(action_index)
            if has_gem == 1:
                gem_actions.append(action_index)

        if gem_actions:
            return gem_actions[0]
        if safe_actions:
            return safe_actions[0]
        return 1
