from __future__ import annotations

import ctypes
import os
import platform
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple


class CCoreUnavailable(RuntimeError):
    pass


class TronkCCore:
    PLAYER_COUNT = 6
    MAX_BODY_LEN = 91
    MAX_TILES = 91

    def __init__(self) -> None:
        self._lib = self._load_library()
        self._bind()

    def _load_library(self) -> ctypes.CDLL:
        root = Path(__file__).resolve().parent
        src = root / "c_core" / "tronk_core.c"
        if not src.exists():
            raise CCoreUnavailable(f"Missing C source: {src}")

        system = platform.system().lower()
        if "darwin" in system:
            lib_name = "libtronk_core.dylib"
        elif "windows" in system:
            lib_name = "tronk_core.dll"
        else:
            lib_name = "libtronk_core.so"

        out = root / "c_core" / lib_name

        if not out.exists() or src.stat().st_mtime > out.stat().st_mtime:
            self._build(src, out)

        try:
            return ctypes.CDLL(str(out))
        except OSError as exc:
            raise CCoreUnavailable(f"Failed to load C core library: {exc}") from exc

    def _build(self, src: Path, out: Path) -> None:
        system = platform.system().lower()

        if "darwin" in system:
            cmd = [
                "cc",
                "-O3",
                "-std=c99",
                "-dynamiclib",
                "-o",
                str(out),
                str(src),
            ]
        elif "windows" in system:
            raise CCoreUnavailable("Windows build is not configured for the C core")
        else:
            cmd = [
                "cc",
                "-O3",
                "-std=c99",
                "-shared",
                "-fPIC",
                "-o",
                str(out),
                str(src),
            ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", errors="replace")
            raise CCoreUnavailable(f"Failed to build C core: {stderr}") from exc

    def _bind(self) -> None:
        lib = self._lib

        lib.tc_init.argtypes = []
        lib.tc_init.restype = None

        lib.tc_mark_dead.argtypes = [ctypes.c_int]
        lib.tc_mark_dead.restype = None

        lib.tc_set_turn.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.tc_set_turn.restype = None

        lib.tc_set_turn_raw.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.tc_set_turn_raw.restype = None

        lib.tc_config_player.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.tc_config_player.restype = None

        lib.tc_get_turn.argtypes = [ctypes.c_int]
        lib.tc_get_turn.restype = ctypes.c_int

        lib.tc_rel_to_abs.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.tc_rel_to_abs.restype = None

        lib.tc_get_tile_abs.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.tc_get_tile_abs.restype = None

        lib.tc_get_tile_rel.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.tc_get_tile_rel.restype = None

        lib.tc_get_player_info.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.tc_get_player_info.restype = None

        lib.tc_resolve_step.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.tc_resolve_step.restype = None

        lib.tc_list_spawn_candidates.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]
        lib.tc_list_spawn_candidates.restype = ctypes.c_int

        lib.tc_add_gem.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.tc_add_gem.restype = ctypes.c_int

        lib.tc_get_player_state.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]
        lib.tc_get_player_state.restype = None

        lib.tc_get_gems.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]
        lib.tc_get_gems.restype = ctypes.c_int

    def init(self) -> None:
        self._lib.tc_init()

    def mark_dead(self, pid: int) -> None:
        self._lib.tc_mark_dead(int(pid))

    def set_turn(self, pid: int, rel_turn: int) -> None:
        self._lib.tc_set_turn(int(pid), int(rel_turn))

    def set_turn_raw(self, pid: int, value: int) -> None:
        self._lib.tc_set_turn_raw(int(pid), int(value))

    def config_player(self, pid: int, alive: int, q: int, r: int, facing: int) -> None:
        self._lib.tc_config_player(int(pid), int(alive), int(q), int(r), int(facing))

    def get_turn(self, pid: int) -> int:
        return int(self._lib.tc_get_turn(int(pid)))

    def rel_to_abs(self, pid: int, q: int, r: int) -> Tuple[int, int]:
        out_q = ctypes.c_int(0)
        out_r = ctypes.c_int(0)
        self._lib.tc_rel_to_abs(int(pid), int(q), int(r), ctypes.byref(out_q), ctypes.byref(out_r))
        return int(out_q.value), int(out_r.value)

    def get_tile_abs(self, q: int, r: int) -> Tuple[int, int, int, int]:
        exists = ctypes.c_int(0)
        is_empty = ctypes.c_int(0)
        player_id = ctypes.c_int(-1)
        has_gem = ctypes.c_int(0)
        self._lib.tc_get_tile_abs(
            int(q),
            int(r),
            ctypes.byref(exists),
            ctypes.byref(is_empty),
            ctypes.byref(player_id),
            ctypes.byref(has_gem),
        )
        return int(exists.value), int(is_empty.value), int(player_id.value), int(has_gem.value)

    def get_tile_rel(self, pid: int, q: int, r: int) -> Tuple[int, int, int, int]:
        exists = ctypes.c_int(0)
        is_empty = ctypes.c_int(0)
        player_id = ctypes.c_int(-1)
        has_gem = ctypes.c_int(0)
        self._lib.tc_get_tile_rel(
            int(pid),
            int(q),
            int(r),
            ctypes.byref(exists),
            ctypes.byref(is_empty),
            ctypes.byref(player_id),
            ctypes.byref(has_gem),
        )
        return int(exists.value), int(is_empty.value), int(player_id.value), int(has_gem.value)

    def get_player_info(self, pid: int) -> Tuple[int, int, int, int, int]:
        alive = ctypes.c_int(0)
        head_q = ctypes.c_int(0)
        head_r = ctypes.c_int(0)
        head_facing = ctypes.c_int(0)
        length = ctypes.c_int(0)
        self._lib.tc_get_player_info(
            int(pid),
            ctypes.byref(alive),
            ctypes.byref(head_q),
            ctypes.byref(head_r),
            ctypes.byref(head_facing),
            ctypes.byref(length),
        )
        return int(alive.value), int(head_q.value), int(head_r.value), int(head_facing.value), int(length.value)

    def resolve_step(self) -> Dict[str, object]:
        death_reasons = (ctypes.c_int * self.PLAYER_COUNT)()
        gem_picked = (ctypes.c_int * self.PLAYER_COUNT)()
        spawn_from_pickups = ctypes.c_int(0)
        spawn_from_starvation = ctypes.c_int(0)
        self._lib.tc_resolve_step(
            death_reasons,
            gem_picked,
            ctypes.byref(spawn_from_pickups),
            ctypes.byref(spawn_from_starvation),
        )
        return {
            "death_reasons": [int(death_reasons[i]) for i in range(self.PLAYER_COUNT)],
            "gem_picked": [int(gem_picked[i]) for i in range(self.PLAYER_COUNT)],
            "spawn_from_pickups": int(spawn_from_pickups.value),
            "spawn_from_starvation": int(spawn_from_starvation.value),
        }

    def list_spawn_candidates(self) -> List[Tuple[int, int]]:
        qs = (ctypes.c_int * self.MAX_TILES)()
        rs = (ctypes.c_int * self.MAX_TILES)()
        n = int(self._lib.tc_list_spawn_candidates(qs, rs, self.MAX_TILES))
        n = min(n, self.MAX_TILES)
        return [(int(qs[i]), int(rs[i])) for i in range(n)]

    def add_gem(self, q: int, r: int) -> bool:
        return bool(self._lib.tc_add_gem(int(q), int(r)))

    def get_player_state(self, pid: int) -> Dict[str, object]:
        alive = ctypes.c_int(0)
        facing = ctypes.c_int(0)
        pending = ctypes.c_int(0)
        length = ctypes.c_int(0)
        body_q = (ctypes.c_int * self.MAX_BODY_LEN)()
        body_r = (ctypes.c_int * self.MAX_BODY_LEN)()

        self._lib.tc_get_player_state(
            int(pid),
            ctypes.byref(alive),
            ctypes.byref(facing),
            ctypes.byref(pending),
            ctypes.byref(length),
            body_q,
            body_r,
            self.MAX_BODY_LEN,
        )

        n = max(0, min(int(length.value), self.MAX_BODY_LEN))
        body = [(int(body_q[i]), int(body_r[i])) for i in range(n)]

        return {
            "alive": int(alive.value),
            "facing": int(facing.value),
            "pending": int(pending.value),
            "length": n,
            "body": body,
        }

    def get_gems(self) -> List[Tuple[int, int]]:
        qs = (ctypes.c_int * self.MAX_TILES)()
        rs = (ctypes.c_int * self.MAX_TILES)()
        n = int(self._lib.tc_get_gems(qs, rs, self.MAX_TILES))
        n = min(n, self.MAX_TILES)
        return [(int(qs[i]), int(rs[i])) for i in range(n)]
