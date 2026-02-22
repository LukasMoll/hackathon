from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from rl.obs import ObsSpec, ObservationEncoder
from rl.policy import build_policy
from tronk_ml import MLTronkEnv


ACTION_TO_TURN = [-1, 0, 1]


def infer_policy_config(policy_path: str, arch: str, hidden_dim: int, mem_dim: int) -> tuple[str, int, int]:
    if arch != "auto":
        return arch, hidden_dim, mem_dim
    cfg_path = Path(policy_path).parent / "train_config.json"
    if not cfg_path.exists():
        return "gru", hidden_dim, mem_dim
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        return (
            str(cfg.get("policy_arch", "gru")),
            int(cfg.get("policy_hidden_dim", hidden_dim)),
            int(cfg.get("policy_mem_dim", mem_dim)),
        )
    except Exception:
        return "gru", hidden_dim, mem_dim


def evaluate(
    policy_path: str,
    matches: int = 200,
    radius: int = 5,
    max_steps: int = 300,
    seed: int = 123,
    engine: str = "c",
    arch: str = "auto",
    hidden_dim: int = 128,
    mem_dim: int = 16,
) -> None:
    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch, hidden_dim, mem_dim = infer_policy_config(policy_path, arch, hidden_dim, mem_dim)

    encoder = ObservationEncoder(ObsSpec(radius=radius, include_threat_map=True))
    model = build_policy(arch, obs_dim=encoder.obs_dim, hidden_dim=hidden_dim, mem_dim=mem_dim).to(device)
    state = torch.load(policy_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    model_hidden_dim = int(getattr(model, "hidden_dim", 128))

    rank_counter = Counter()
    wins = 0

    for _ in range(matches):
        env = MLTronkEnv(
            seed=int(rng.integers(0, 2_000_000_000)),
            max_steps=max_steps,
            randomize_starts=True,
            randomize_facings=True,
            use_c_core=(engine != "python"),
        )
        hidden = [np.zeros((model_hidden_dim,), dtype=np.float32) for _ in range(6)]

        while not env.done:
            actions = [0] * 6
            for pid in range(6):
                if not env.players[pid].alive:
                    actions[pid] = 0
                    continue
                obs = encoder.encode(env, pid)
                action, _, _, new_h = model.act(obs, hidden[pid], device=device, deterministic=True)
                hidden[pid] = new_h
                actions[pid] = ACTION_TO_TURN[action]
            env.step(actions)

        ranks = env.compute_ranks()
        for rank in ranks:
            rank_counter[round(rank, 2)] += 1
        if min(ranks) <= 1.0:
            wins += 1

    print(f"matches={matches}")
    print(f"win_rate_any_seat={wins / max(1, matches):.4f}")
    print("rank_histogram:")
    for rank_key in sorted(rank_counter.keys()):
        print(f"  {rank_key}: {rank_counter[rank_key]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained Tronk rank PPO policy")
    parser.add_argument("--policy", required=True, type=str)
    parser.add_argument("--matches", type=int, default=200)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--engine", choices=["c", "python"], default="c")
    parser.add_argument("--arch", choices=["auto", "gru", "planning"], default="auto")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--mem-dim", type=int, default=16)
    args = parser.parse_args()
    evaluate(
        policy_path=args.policy,
        matches=args.matches,
        radius=args.radius,
        max_steps=args.max_steps,
        seed=args.seed,
        engine=args.engine,
        arch=args.arch,
        hidden_dim=args.hidden_dim,
        mem_dim=args.mem_dim,
    )
