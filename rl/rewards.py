from __future__ import annotations

import math
from typing import List

from rl.config import RankRewardConfig


def rank_to_reward(rank: float, cfg: RankRewardConfig) -> float:
    table = {
        1: cfg.first,
        2: cfg.second,
        3: cfg.third,
        4: cfg.fourth,
        5: cfg.fifth,
        6: cfg.sixth,
    }

    if rank <= 1:
        return table[1]
    if rank >= 6:
        return table[6]

    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return table[lo]

    w = rank - lo
    return table[lo] * (1.0 - w) + table[hi] * w


def ranks_to_rewards(ranks: List[float], cfg: RankRewardConfig) -> List[float]:
    return [rank_to_reward(r, cfg) for r in ranks]
