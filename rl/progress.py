from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from rl.obs import ObservationEncoder
from tronk_ml import MLTronkEnv


ACTION_TO_TURN = [-1, 0, 1]


def run_self_play_match(
    model: torch.nn.Module,
    encoder: ObservationEncoder,
    device: torch.device,
    *,
    seed: int,
    max_steps: int,
    randomize_starts: bool,
    randomize_facings: bool,
    use_c_core: bool = True,
    require_c_core: bool = False,
    deterministic: bool = True,
) -> Dict[str, Any]:
    env = MLTronkEnv(
        seed=seed,
        max_steps=max_steps,
        randomize_starts=randomize_starts,
        randomize_facings=randomize_facings,
        use_c_core=use_c_core,
        require_c_core=require_c_core,
    )
    hidden_dim = int(getattr(model, "hidden_dim", 128))
    hiddens = [np.zeros((hidden_dim,), dtype=np.float32) for _ in range(6)]

    while not env.done:
        actions = [0] * 6
        for pid in range(6):
            if not env.players[pid].alive:
                actions[pid] = 0
                continue
            obs = encoder.encode(env, pid)
            action_idx, _, _, next_hidden = model.act(
                obs,
                hiddens[pid],
                device=device,
                deterministic=deterministic,
            )
            hiddens[pid] = next_hidden
            actions[pid] = ACTION_TO_TURN[action_idx]
        env.step(actions)

    return env.snapshot()


def write_progress_replay(
    *,
    output_dir: Path,
    prefix: str,
    update_idx: int,
    match_idx: int,
    seed: int,
    snapshot: Dict[str, Any],
    deterministic: bool,
    checkpoint_name: str = "",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{prefix}_u{update_idx:06d}_m{match_idx:02d}.json"
    path = output_dir / filename
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "session_id": f"{prefix}_u{update_idx:06d}_m{match_idx:02d}",
        "created_at": now,
        "saved_at": now,
        "snapshot": snapshot,
        "meta": {
            "type": "training_progress_replay",
            "update": update_idx,
            "match_idx": match_idx,
            "seed": seed,
            "deterministic": deterministic,
            "checkpoint": checkpoint_name,
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
