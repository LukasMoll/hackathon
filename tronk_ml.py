from __future__ import annotations

import json
import random
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from tronk_engine import (
    BOARD_RADIUS,
    CORNER_FACINGS,
    CORNER_SPAWNS,
    DIRECTIONS,
    INITIAL_GEMS,
    PLAYER_COLORS,
    rotate_axial,
)
try:
    from tronk_ctypes import CCoreUnavailable, TronkCCore
except Exception:  # pragma: no cover - C core is optional
    CCoreUnavailable = RuntimeError  # type: ignore[assignment]
    TronkCCore = None  # type: ignore[assignment]


ActionType = Union[int, str]


@dataclass
class MLPlayer:
    player_id: int
    color: str
    body: List[Tuple[int, int]]
    facing: int
    pending_turn: int = 0
    growth_pending: int = 0
    alive: bool = True
    error: Optional[str] = None

    @property
    def head(self) -> Tuple[int, int]:
        return self.body[0]

    @property
    def length(self) -> int:
        return len(self.body)


class MLTronkEnv:
    def __init__(
        self,
        seed: int = 1,
        max_steps: int = 300,
        randomize_starts: bool = False,
        randomize_facings: bool = False,
        use_c_core: bool = False,
        mirror_queries: bool = True,
        require_c_core: bool = False,
    ):
        self.seed = seed
        self.random = random.Random(seed)
        self.max_steps = max_steps
        self.randomize_starts = randomize_starts
        self.randomize_facings = randomize_facings
        self.use_c_core = use_c_core
        self.mirror_queries = mirror_queries
        self.require_c_core = require_c_core
        self.core_mode = "python"
        self._c_core: Optional[TronkCCore] = None

        self.step_count = 0
        self.no_gem_steps = 0
        self.done = False
        self.winner_id: Optional[int] = None
        self.death_step: List[Optional[int]] = [None] * 6

        self.players: List[MLPlayer] = []
        self.gems = set()
        self.frames: List[Dict[str, Any]] = []
        self._occupied_cache: Optional[Dict[Tuple[int, int], int]] = None
        if self.use_c_core and TronkCCore is not None:
            try:
                self._c_core = TronkCCore()
                self.core_mode = "c"
            except CCoreUnavailable as exc:
                if self.require_c_core:
                    raise RuntimeError(f"C core required but unavailable: {exc}") from exc
                self._c_core = None
                self.core_mode = "python"
        elif self.use_c_core and self.require_c_core:
            raise RuntimeError("C core required but ctypes bridge is unavailable")
        self.reset()

    def reset(self) -> Dict[str, Any]:
        self.step_count = 0
        self.no_gem_steps = 0
        self.done = False
        self.winner_id = None
        self.death_step = [None] * 6

        spawn_order = list(range(6))
        if self.randomize_starts:
            self.random.shuffle(spawn_order)

        if self._c_core is not None:
            self._c_core.init()
            for i in range(6):
                spawn_idx = spawn_order[i]
                facing = CORNER_FACINGS[spawn_idx]
                if self.randomize_facings:
                    facing = self.random.randint(0, 5)
                q, r = CORNER_SPAWNS[spawn_idx]
                self._c_core.config_player(i, 1, q, r, facing)
            self._sync_from_c_state()
        else:
            self.players = []
            for i in range(6):
                spawn_idx = spawn_order[i]
                facing = CORNER_FACINGS[spawn_idx]
                if self.randomize_facings:
                    facing = self.random.randint(0, 5)
                self.players.append(
                    MLPlayer(
                        player_id=i,
                        color=PLAYER_COLORS[i],
                        body=[CORNER_SPAWNS[spawn_idx]],
                        facing=facing,
                    )
                )
            self.gems = set(INITIAL_GEMS)
        self._occupied_cache = None
        self.frames = []
        self._capture_frame(events=["game_start"])
        return self.snapshot()

    def _sync_from_c_state(self) -> None:
        if self._c_core is None:
            return
        prev_errors = {p.player_id: p.error for p in self.players}
        players: List[MLPlayer] = []
        for pid in range(6):
            state = self._c_core.get_player_state(pid)
            body = list(state["body"])
            if not body:
                body = [CORNER_SPAWNS[pid]]
            players.append(
                MLPlayer(
                    player_id=pid,
                    color=PLAYER_COLORS[pid],
                    body=body,
                    facing=int(state["facing"]),
                    pending_turn=int(state["pending"]),
                    growth_pending=0,
                    alive=bool(int(state["alive"])),
                    error=prev_errors.get(pid),
                )
            )
        self.players = players
        self.gems = set(self._c_core.get_gems())
        self._occupied_cache = None

    def alive_players(self) -> List[MLPlayer]:
        return [p for p in self.players if p.alive]

    def wrap_facing(self, facing: int) -> int:
        return ((facing % 6) + 6) % 6

    def effective_facing(self, player: MLPlayer) -> int:
        return self.wrap_facing(player.facing + player.pending_turn)

    def is_inside(self, q: int, r: int) -> bool:
        s = -q - r
        return max(abs(q), abs(r), abs(s)) <= BOARD_RADIUS

    def occupied_positions(self) -> Dict[Tuple[int, int], int]:
        if self._occupied_cache is not None:
            return self._occupied_cache
        occ: Dict[Tuple[int, int], int] = {}
        for p in self.players:
            for tile in p.body:
                occ[tile] = p.player_id
        self._occupied_cache = occ
        return occ

    def parse_action(self, action: ActionType) -> Optional[int]:
        if isinstance(action, int):
            return action if action in (-1, 0, 1) else None
        if not isinstance(action, str):
            return None
        v = action.strip().lower()
        if v in {"left", "l", "-1"}:
            return -1
        if v in {"forward", "f", "0"}:
            return 0
        if v in {"right", "r", "1"}:
            return 1
        return None

    def set_actions(self, actions: Sequence[ActionType]) -> None:
        if len(actions) != 6:
            raise ValueError("actions must have length 6")
        for pid, raw_action in enumerate(actions):
            p = self.players[pid]
            if not p.alive:
                continue
            parsed = self.parse_action(raw_action)
            if parsed is None:
                continue
            p.pending_turn = parsed
            if self._c_core is not None:
                self._c_core.set_turn(pid, parsed)

    def rel_to_abs(self, player_id: int, q: int, r: int) -> Tuple[int, int]:
        if self._c_core is not None and not self.mirror_queries:
            return self._c_core.rel_to_abs(player_id, q, r)
        p = self.players[player_id]
        rq, rr = rotate_axial(q, r, self.effective_facing(p))
        hq, hr = p.head
        return hq + rq, hr + rr

    def get_tile_abs(self, q: int, r: int) -> Tuple[int, int, int, int]:
        if self._c_core is not None and not self.mirror_queries:
            return self._c_core.get_tile_abs(q, r)
        if not self.is_inside(q, r):
            return (0, 0, -1, 0)

        occ = self.occupied_positions()
        tile = (q, r)
        if tile in occ:
            return (1, 0, occ[tile], 0)

        gem = 1 if tile in self.gems else 0
        is_empty = 0 if gem else 1
        return (1, is_empty, -1, gem)

    def get_tile_rel(self, player_id: int, q: int, r: int) -> Tuple[int, int, int, int]:
        aq, ar = self.rel_to_abs(player_id, q, r)
        return self.get_tile_abs(aq, ar)

    def get_turn(self, player_id: int) -> int:
        if self._c_core is not None and not self.mirror_queries:
            return self._c_core.get_turn(player_id)
        return self.players[player_id].pending_turn

    def get_player_info(self, player_id: int) -> Tuple[int, int, int, int, int]:
        if self._c_core is not None and not self.mirror_queries:
            return self._c_core.get_player_info(player_id)
        if player_id < 0 or player_id >= len(self.players):
            return (0, 0, 0, 0, 0)
        p = self.players[player_id]
        hq, hr = p.head
        return (1 if p.alive else 0, hq, hr, self.effective_facing(p), p.length)

    def get_player_id(self, player_id: int) -> int:
        return player_id

    def get_tick(self) -> int:
        # Kept for compatibility with query interface; each ML step is one tick unit.
        return self.step_count

    def call(self, player_id: int, function_name: str, args: Sequence[int]) -> Any:
        fn = function_name.strip()
        if fn == "getPlayerId":
            return self.get_player_id(player_id)
        if fn == "getTurn":
            return self.get_turn(player_id)
        if fn == "getTileAbs":
            q = args[0] if len(args) > 0 else 0
            r = args[1] if len(args) > 1 else 0
            return self.get_tile_abs(q, r)
        if fn == "getTileRel":
            q = args[0] if len(args) > 0 else 0
            r = args[1] if len(args) > 1 else 0
            return self.get_tile_rel(player_id, q, r)
        if fn == "relToAbs":
            q = args[0] if len(args) > 0 else 0
            r = args[1] if len(args) > 1 else 0
            return self.rel_to_abs(player_id, q, r)
        if fn == "getTick":
            return self.get_tick()
        if fn == "getPlayerInfo":
            pid = args[0] if len(args) > 0 else -1
            return self.get_player_info(pid)
        raise ValueError(f"Unsupported ML function call: {function_name}")

    def _spawn_gem(self) -> bool:
        occupied = set(self.occupied_positions().keys())
        candidates = [
            (q, r)
            for q in range(-BOARD_RADIUS, BOARD_RADIUS + 1)
            for r in range(-BOARD_RADIUS, BOARD_RADIUS + 1)
            if self.is_inside(q, r) and (q, r) not in occupied and (q, r) not in self.gems
        ]
        if not candidates:
            return False
        self.gems.add(self.random.choice(candidates))
        return True

    def step(self, actions: Sequence[ActionType]) -> Dict[str, Any]:
        if self.done:
            return self.snapshot()

        self.set_actions(actions)

        events: List[str] = []

        if self._c_core is not None:
            reason_map = {
                1: "Hit wall",
                2: "Hit occupied tile",
                3: "Contested destination",
            }
            alive_before = [p.alive for p in self.players]
            res = self._c_core.resolve_step()
            death_reasons = res["death_reasons"]
            gem_picked = res["gem_picked"]

            spawn_pickups = int(res["spawn_from_pickups"])
            spawn_starvation = int(res["spawn_from_starvation"])
            for _ in range(spawn_pickups):
                candidates = self._c_core.list_spawn_candidates()
                if not candidates:
                    break
                q, r = self.random.choice(candidates)
                self._c_core.add_gem(q, r)
            if spawn_starvation > 0:
                candidates = self._c_core.list_spawn_candidates()
                if candidates:
                    q, r = self.random.choice(candidates)
                    self._c_core.add_gem(q, r)
                    events.append("extra_gem_spawn")

            if sum(gem_picked) > 0:
                self.no_gem_steps = 0
            else:
                self.no_gem_steps += 1

            self.step_count += 1
            self._sync_from_c_state()

            for pid in range(6):
                if gem_picked[pid]:
                    events.append(f"p{pid}_gem")
                if alive_before[pid] and not self.players[pid].alive:
                    reason = reason_map.get(int(death_reasons[pid]), "Died")
                    self.players[pid].error = reason
                    self.death_step[pid] = self.step_count
                    events.append(f"p{pid}_dead:{reason}")
        else:
            occupied = self.occupied_positions()

            desired_face: Dict[int, int] = {}
            desired_pos: Dict[int, Tuple[int, int]] = {}

            for p in self.players:
                if not p.alive:
                    continue
                face = self.wrap_facing(p.facing + p.pending_turn)
                dq, dr = DIRECTIONS[face]
                hq, hr = p.head
                desired_face[p.player_id] = face
                desired_pos[p.player_id] = (hq + dq, hr + dr)

            doomed: Dict[int, str] = {}

            for pid, dest in desired_pos.items():
                if not self.is_inside(dest[0], dest[1]):
                    doomed[pid] = "Hit wall"
                    continue
                if dest in occupied:
                    doomed[pid] = "Hit occupied tile"

            contested: Dict[Tuple[int, int], List[int]] = {}
            for pid, dest in desired_pos.items():
                if pid in doomed:
                    continue
                contested.setdefault(dest, []).append(pid)

            for dest, pids in contested.items():
                if len(pids) > 1:
                    for pid in pids:
                        doomed[pid] = f"Contested tile {dest}"

            consumed_gems: List[Tuple[int, int]] = []

            for pid, reason in doomed.items():
                p = self.players[pid]
                p.alive = False
                p.error = reason
                self.death_step[pid] = self.step_count + 1
                events.append(f"p{pid}_dead:{reason}")

            for p in self.players:
                if not p.alive:
                    continue
                pid = p.player_id
                if pid in doomed:
                    continue

                dest = desired_pos[pid]
                p.facing = desired_face[pid]
                p.pending_turn = 0
                p.body.insert(0, dest)

                if dest in self.gems:
                    consumed_gems.append(dest)
                    p.growth_pending += 1
                    events.append(f"p{pid}_gem")

                if p.growth_pending > 0:
                    p.growth_pending -= 1
                else:
                    p.body.pop()

            # Movement changed occupied tiles, so refresh cache on next query.
            self._occupied_cache = None

            if consumed_gems:
                for gem in consumed_gems:
                    self.gems.discard(gem)
                for _ in consumed_gems:
                    self._spawn_gem()
                self.no_gem_steps = 0
            else:
                self.no_gem_steps += 1
                if self.no_gem_steps >= 5:
                    if self._spawn_gem():
                        events.append("extra_gem_spawn")

            self.step_count += 1

        alive = self.alive_players()
        if len(alive) <= 1:
            self.done = True
            self.winner_id = alive[0].player_id if len(alive) == 1 else None
            events.append("game_end")
        elif self.step_count >= self.max_steps:
            self.done = True
            self.winner_id = None
            events.append("max_steps")

        self._capture_frame(events=events)
        return self.snapshot()

    def compute_ranks(self) -> List[float]:
        """Return per-player rank (1.0 best, 6.0 worst), with tie averaging."""
        # Alive at episode end is best rank.
        groups: Dict[int, List[int]] = {}
        alive_group = 10**9
        for pid, p in enumerate(self.players):
            if p.alive:
                groups.setdefault(alive_group, []).append(pid)
            else:
                death = self.death_step[pid]
                groups.setdefault(death if death is not None else -1, []).append(pid)

        ordered_keys = sorted(groups.keys(), reverse=True)
        ranks: List[float] = [6.0] * 6
        current_rank = 1
        for key in ordered_keys:
            pids = groups[key]
            start = current_rank
            end = current_rank + len(pids) - 1
            avg_rank = (start + end) / 2.0
            for pid in pids:
                ranks[pid] = avg_rank
            current_rank = end + 1
        return ranks

    def _capture_frame(self, events: Optional[List[str]] = None) -> None:
        frame_players: List[Dict[str, Any]] = []
        for p in self.players:
            hq, hr = p.head
            frame_players.append(
                {
                    "id": p.player_id,
                    "alive": p.alive,
                    "q": hq,
                    "r": hr,
                    "facing": self.effective_facing(p),
                    "pendingTurn": p.pending_turn,
                    "length": p.length,
                    "color": p.color,
                    "body": [[x, y] for x, y in p.body],
                    "error": p.error,
                }
            )

        self.frames.append(
            {
                "step": self.step_count,
                "players": frame_players,
                "gems": [[q, r] for q, r in sorted(self.gems)],
                "events": events or [],
            }
        )

    def snapshot(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "core_mode": self.core_mode,
            "step": self.step_count,
            "max_steps": self.max_steps,
            "done": self.done,
            "winner_id": self.winner_id,
            "death_step": self.death_step,
            "ranks": self.compute_ranks() if self.done else None,
            "frames": self.frames,
        }


@dataclass
class MLSession:
    session_id: str
    env: MLTronkEnv
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class MLSessionStore:
    def __init__(self, save_dir: Path):
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.sessions: Dict[str, MLSession] = {}

    def new_session(self, seed: int = 1, max_steps: int = 300) -> MLSession:
        sid = uuid.uuid4().hex[:12]
        env = MLTronkEnv(seed=seed, max_steps=max_steps, use_c_core=True)
        session = MLSession(session_id=sid, env=env)
        self.sessions[sid] = session
        return session

    def get_session(self, sid: str) -> MLSession:
        if sid not in self.sessions:
            raise KeyError(f"Unknown session '{sid}'")
        return self.sessions[sid]

    def save_session(self, sid: str, name: str) -> Path:
        session = self.get_session(sid)
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
        if not safe_name:
            safe_name = "ml_run"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = self.save_dir / f"mlmanual_{safe_name}_{timestamp}.json"

        payload = {
            "session_id": sid,
            "created_at": session.created_at,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "snapshot": session.env.snapshot(),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def list_saved(self) -> List[str]:
        return sorted([p.name for p in self.save_dir.glob("*.json")])

    def load_saved(self, filename: str) -> Dict[str, Any]:
        if "/" in filename or "\\" in filename:
            raise ValueError("Invalid filename")
        path = self.save_dir / filename
        if not path.exists():
            raise FileNotFoundError(filename)
        return json.loads(path.read_text(encoding="utf-8"))
