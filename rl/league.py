from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

from rl.config import LeagueConfig


@dataclass
class SnapshotEntry:
    update_idx: int
    path: Path


class LeaguePool:
    def __init__(self, cfg: LeagueConfig):
        self.cfg = cfg
        self.snapshots: List[SnapshotEntry] = []

    def add_snapshot(self, update_idx: int, model_state: Dict[str, torch.Tensor], checkpoint_dir: Path) -> SnapshotEntry:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / f"snapshot_u{update_idx:06d}.pt"
        torch.save(model_state, path)
        entry = SnapshotEntry(update_idx=update_idx, path=path)
        self.snapshots.append(entry)
        if len(self.snapshots) > self.cfg.snapshot_pool_size:
            oldest = self.snapshots.pop(0)
            try:
                oldest.path.unlink(missing_ok=True)
            except Exception:
                pass
        return entry

    def sample_snapshot(self, rng: random.Random) -> Optional[SnapshotEntry]:
        if not self.snapshots:
            return None

        r = rng.random()
        recent_count = max(1, min(10, len(self.snapshots)))
        recent = self.snapshots[-recent_count:]
        old = self.snapshots[:-recent_count] if len(self.snapshots) > recent_count else []

        if r < self.cfg.p_recent:
            return rng.choice(recent)
        if r < self.cfg.p_recent + self.cfg.p_old and old:
            return rng.choice(old)
        if r < self.cfg.p_recent + self.cfg.p_old:
            return rng.choice(recent)
        return None
