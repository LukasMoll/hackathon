from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from tronk_engine import rotate_axial
from tronk_ml import MLTronkEnv


@dataclass
class ObsSpec:
    radius: int = 5
    include_threat_map: bool = True


def iter_local_hex(radius: int) -> List[Tuple[int, int]]:
    coords: List[Tuple[int, int]] = []
    for q in range(-radius, radius + 1):
        for r in range(-radius, radius + 1):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) <= radius:
                coords.append((q, r))
    coords.sort(key=lambda x: (x[1], x[0]))
    return coords


class ObservationEncoder:
    def __init__(self, spec: ObsSpec):
        self.spec = spec
        self.local_cells = iter_local_hex(spec.radius)
        self.local_index: Dict[Tuple[int, int], int] = {
            coord: idx for idx, coord in enumerate(self.local_cells)
        }
        self._rotated_local = {
            facing: [rotate_axial(lq, lr, facing) for (lq, lr) in self.local_cells]
            for facing in range(6)
        }

        self.base_channels = 11  # outside,my_head,my_body,enemy_head,enemy_body,enemy_dir6
        self.channels = self.base_channels + (1 if spec.include_threat_map else 0)
        self.global_features = 4
        self.obs_dim = self.channels * len(self.local_cells) + self.global_features

    @staticmethod
    def wrap_facing(f: int) -> int:
        out = f % 6
        if out < 0:
            out += 6
        return out

    def encode(self, env: MLTronkEnv, player_id: int) -> np.ndarray:
        p = env.players[player_id]
        cell_count = len(self.local_cells)
        spatial = np.zeros((self.channels, cell_count), dtype=np.float32)

        occ = env.occupied_positions()
        gems = env.gems
        eff_facing = env.effective_facing(p)
        hq, hr = p.head
        rotated_cells = self._rotated_local[eff_facing]

        # occupancy and board masks
        for idx, ((lq, lr), (rq, rr)) in enumerate(zip(self.local_cells, rotated_cells)):
            aq, ar = hq + rq, hr + rr
            if not env.is_inside(aq, ar):
                spatial[0, idx] = 1.0
                continue

            occ_pid = occ.get((aq, ar), -1)
            if occ_pid == player_id:
                if lq == 0 and lr == 0:
                    spatial[1, idx] = 1.0
                else:
                    spatial[2, idx] = 1.0
            elif occ_pid >= 0:
                occ_player = env.players[occ_pid]
                # Heads for alive opponents; all other occupied enemy cells as body.
                if occ_player.alive and occ_player.head == (aq, ar):
                    spatial[3, idx] = 1.0
                else:
                    spatial[4, idx] = 1.0

            if (aq, ar) in gems:
                pass  # Gem intentionally not used in reward; can still be inferred via occupancy channel.

        # enemy direction channels at enemy heads
        for enemy in env.players:
            if enemy.player_id == player_id or not enemy.alive:
                continue
            ehq, ehr = enemy.head
            dq = ehq - hq
            dr = ehr - hr
            rq, rr = rotate_axial(dq, dr, -eff_facing)
            loc = (rq, rr)
            idx = self.local_index.get(loc)
            if idx is None:
                continue
            rel_facing = self.wrap_facing(env.effective_facing(enemy) - eff_facing)
            spatial[5 + rel_facing, idx] = 1.0

        # threat map: where enemy could step next
        if self.spec.include_threat_map:
            threat_channel = self.channels - 1
            for enemy in env.players:
                if enemy.player_id == player_id or not enemy.alive:
                    continue
                ef = env.effective_facing(enemy)
                ehq, ehr = enemy.head
                for turn in (-1, 0, 1):
                    nf = self.wrap_facing(ef + turn)
                    dq, dr = [(0, -1), (1, -1), (1, 0), (0, 1), (-1, 1), (-1, 0)][nf]
                    tq, tr = ehq + dq, ehr + dr
                    mdq, mdr = tq - hq, tr - hr
                    lq, lr = rotate_axial(mdq, mdr, -eff_facing)
                    idx = self.local_index.get((lq, lr))
                    if idx is not None:
                        spatial[threat_channel, idx] = 1.0

        alive_opponents = sum(1 for x in env.players if x.player_id != player_id and x.alive)
        global_vec = np.array(
            [
                alive_opponents / 5.0,
                env.step_count / max(1, env.max_steps),
                p.length / 91.0,
                1.0 if p.alive else 0.0,
            ],
            dtype=np.float32,
        )

        return np.concatenate([spatial.reshape(-1), global_vec], axis=0)
