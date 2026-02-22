from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from rl.obs import ObsSpec, ObservationEncoder
from rl.policy import build_policy
from tronk_ml import MLTronkEnv


ACTION_TO_TURN = [-1, 0, 1]
DEFAULT_PERCENT_POINTS: tuple[int, ...] = (10, 20, 30, 40, 50, 60, 70, 80, 90)
METRIC_KEYS: tuple[str, ...] = (
    "latest_avg_rank",
    "opponent_avg_rank",
    "latest_rank_advantage",
    "latest_win_rate_per_agent",
    "opponent_win_rate_per_agent",
    "latest_win_rate_advantage",
    "latest_top2_rate_per_agent",
    "opponent_top2_rate_per_agent",
    "latest_avg_survival_steps",
    "opponent_avg_survival_steps",
)


@dataclass(frozen=True)
class SnapshotRef:
    update: int
    path: Path


def parse_percent_points(text: str) -> List[int]:
    points: List[int] = []
    seen: set[int] = set()
    for raw in text.split(","):
        token = raw.strip()
        if not token:
            continue
        value = int(token)
        if value < 1 or value > 100:
            raise ValueError(f"Percent point out of range [1,100]: {value}")
        if value in seen:
            continue
        seen.add(value)
        points.append(value)
    if not points:
        raise ValueError("At least one percent point is required")
    return points


def neighbor_percent_points(center: int, radius: int) -> List[int]:
    r = max(0, int(radius))
    values: List[int] = []
    for p in range(center - r, center + r + 1):
        if 1 <= p <= 100:
            values.append(int(p))
    return values


def split_matches(total: int, buckets: int) -> List[int]:
    n = max(1, int(total))
    k = max(1, int(buckets))
    base = n // k
    rem = n % k
    return [base + (1 if i < rem else 0) for i in range(k)]


def weighted_average(metric_rows: Sequence[Dict[str, Any]], key: str) -> float:
    weights = [int(row.get("matches", 0)) for row in metric_rows]
    denom = sum(weights)
    if denom <= 0:
        return 0.0
    numer = sum(float(row[key]) * int(row.get("matches", 0)) for row in metric_rows)
    return float(numer / denom)


def parse_snapshot_update(path: Path) -> int:
    m = re.search(r"snapshot_u(\d+)\.pt$", path.name)
    if not m:
        raise ValueError(f"Unrecognized snapshot filename: {path.name}")
    return int(m.group(1))


def list_snapshots(checkpoint_dir: Path) -> List[SnapshotRef]:
    snaps: List[SnapshotRef] = []
    for path in sorted(checkpoint_dir.glob("snapshot_u*.pt")):
        snaps.append(SnapshotRef(update=parse_snapshot_update(path), path=path))
    return snaps


def choose_nearest_snapshot(snapshots: Sequence[SnapshotRef], target_update: int) -> SnapshotRef:
    if not snapshots:
        raise ValueError("No snapshots available")
    # Tie-break toward earlier updates.
    return min(snapshots, key=lambda s: (abs(s.update - target_update), s.update))


def load_metrics_rows(metrics_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not metrics_path.exists():
        return rows
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def infer_train_config(output_dir: Path) -> Dict[str, Any]:
    cfg_path = output_dir / "train_config.json"
    if not cfg_path.exists():
        return {}
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def infer_total_updates(output_dir: Path, cfg: Dict[str, Any]) -> int:
    # Prefer the configured run length when available. metrics.jsonl can contain
    # appended rows from previous runs if output_dir is reused.
    if "total_updates" in cfg:
        return int(cfg["total_updates"])
    rows = load_metrics_rows(output_dir / "metrics.jsonl")
    if rows:
        return int(max(int(row.get("update", 0)) for row in rows))
    raise RuntimeError(f"Could not infer total updates from {output_dir}")


def infer_policy_config(cfg: Dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(cfg.get("policy_arch", "planning")),
        int(cfg.get("policy_hidden_dim", 128)),
        int(cfg.get("policy_mem_dim", 16)),
    )


def build_loaded_model(
    *,
    arch: str,
    obs_dim: int,
    hidden_dim: int,
    mem_dim: int,
    state_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, int]:
    model = build_policy(
        arch=arch,
        obs_dim=obs_dim,
        hidden_dim=hidden_dim,
        mem_dim=mem_dim,
    ).to(device)
    state = torch.load(state_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, int(getattr(model, "hidden_dim", 128))


def run_head_to_head(
    *,
    latest_model: torch.nn.Module,
    latest_hidden_dim: int,
    opponent_model: torch.nn.Module,
    opponent_hidden_dim: int,
    encoder: ObservationEncoder,
    device: torch.device,
    matches: int,
    seed: int,
    max_steps: int,
    use_c_core: bool,
    require_c_core: bool,
    deterministic: bool,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)

    latest_rank_sum = 0.0
    latest_win = 0
    latest_top2 = 0
    latest_survival_sum = 0.0

    opp_rank_sum = 0.0
    opp_win = 0
    opp_top2 = 0
    opp_survival_sum = 0.0

    latest_agents = 0
    opp_agents = 0

    for _ in range(matches):
        env = MLTronkEnv(
            seed=int(rng.integers(0, 2_000_000_000)),
            max_steps=max_steps,
            randomize_starts=True,
            randomize_facings=True,
            use_c_core=use_c_core,
            require_c_core=require_c_core,
        )

        seats = np.arange(6)
        rng.shuffle(seats)
        group_for_pid = [""] * 6
        for i, pid in enumerate(seats):
            group_for_pid[int(pid)] = "latest" if i < 3 else "opponent"

        hidden: List[np.ndarray] = []
        for pid in range(6):
            if group_for_pid[pid] == "latest":
                hidden.append(np.zeros((latest_hidden_dim,), dtype=np.float32))
            else:
                hidden.append(np.zeros((opponent_hidden_dim,), dtype=np.float32))

        while not env.done:
            actions = [0] * 6
            for pid in range(6):
                if not env.players[pid].alive:
                    continue
                obs = encoder.encode(env, pid)
                if group_for_pid[pid] == "latest":
                    action, _, _, new_h = latest_model.act(obs, hidden[pid], device=device, deterministic=deterministic)
                else:
                    action, _, _, new_h = opponent_model.act(
                        obs,
                        hidden[pid],
                        device=device,
                        deterministic=deterministic,
                    )
                hidden[pid] = new_h
                actions[pid] = ACTION_TO_TURN[action]
            env.step(actions)

        ranks = env.compute_ranks()
        for pid, rank in enumerate(ranks):
            death = env.death_step[pid]
            survival = float(death if death is not None else env.step_count)
            if group_for_pid[pid] == "latest":
                latest_agents += 1
                latest_rank_sum += float(rank)
                latest_win += 1 if rank <= 1.0 else 0
                latest_top2 += 1 if rank <= 2.0 else 0
                latest_survival_sum += survival
            else:
                opp_agents += 1
                opp_rank_sum += float(rank)
                opp_win += 1 if rank <= 1.0 else 0
                opp_top2 += 1 if rank <= 2.0 else 0
                opp_survival_sum += survival

    if latest_agents == 0 or opp_agents == 0:
        raise RuntimeError("Head-to-head benchmark produced empty agent counts")

    latest_avg_rank = latest_rank_sum / latest_agents
    opp_avg_rank = opp_rank_sum / opp_agents
    latest_win_rate = latest_win / latest_agents
    opp_win_rate = opp_win / opp_agents

    return {
        "latest_avg_rank": float(latest_avg_rank),
        "opponent_avg_rank": float(opp_avg_rank),
        "latest_rank_advantage": float(opp_avg_rank - latest_avg_rank),
        "latest_win_rate_per_agent": float(latest_win_rate),
        "opponent_win_rate_per_agent": float(opp_win_rate),
        "latest_win_rate_advantage": float(latest_win_rate - opp_win_rate),
        "latest_top2_rate_per_agent": float(latest_top2 / latest_agents),
        "opponent_top2_rate_per_agent": float(opp_top2 / opp_agents),
        "latest_avg_survival_steps": float(latest_survival_sum / latest_agents),
        "opponent_avg_survival_steps": float(opp_survival_sum / opp_agents),
    }


def render_rank_advantage_svg(points: Sequence[Dict[str, Any]], output_path: Path) -> None:
    if not points:
        raise RuntimeError("No benchmark points to render")

    width = 1200
    height = 700
    margin_l = 95
    margin_r = 50
    margin_t = 70
    margin_b = 85
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    xs = [float(p["percent"]) for p in points]
    ys = [float(p["latest_rank_advantage"]) for p in points]

    xmin = min(xs)
    xmax = max(xs)
    ymin = min(min(ys), 0.0)
    ymax = max(max(ys), 0.0)
    pad = max(0.02, (ymax - ymin) * 0.12 if ymax > ymin else 0.1)
    ymin -= pad
    ymax += pad

    def sx(x: float) -> float:
        if xmax == xmin:
            return margin_l + plot_w / 2.0
        return margin_l + ((x - xmin) / (xmax - xmin)) * plot_w

    def sy(y: float) -> float:
        return margin_t + plot_h - ((y - ymin) / (ymax - ymin)) * plot_h

    xticks = sorted(set(int(x) for x in xs))
    yticks = 8
    grid: List[str] = []
    labels: List[str] = []

    for xval in xticks:
        xpix = sx(float(xval))
        grid.append(
            f'<line x1="{xpix:.2f}" y1="{margin_t}" x2="{xpix:.2f}" y2="{margin_t + plot_h}" '
            f'stroke="#ececec" stroke-width="1"/>'
        )
        labels.append(
            f'<text x="{xpix:.2f}" y="{height - 35}" text-anchor="middle" font-size="14" fill="#444">{xval}%</text>'
        )

    for i in range(yticks + 1):
        yval = ymin + (ymax - ymin) * (i / yticks)
        ypix = sy(yval)
        grid.append(
            f'<line x1="{margin_l}" y1="{ypix:.2f}" x2="{margin_l + plot_w}" y2="{ypix:.2f}" '
            f'stroke="#ececec" stroke-width="1"/>'
        )
        labels.append(
            f'<text x="{margin_l - 12}" y="{ypix + 5:.2f}" text-anchor="end" font-size="14" fill="#444">{yval:.3f}</text>'
        )

    zero_y = sy(0.0)
    polyline = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in zip(xs, ys))

    point_marks = []
    update_labels = []
    for p in points:
        x = sx(float(p["percent"]))
        y = sy(float(p["latest_rank_advantage"]))
        point_marks.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.8" fill="#0f4fa8"/>')
        update_labels.append(
            f'<text x="{x:.2f}" y="{max(margin_t + 14, y - 10):.2f}" text-anchor="middle" font-size="12" fill="#2c2c2c">'
            f"u{int(p['checkpoint_update'])}</text>"
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>
  <text x="{width/2:.1f}" y="36" text-anchor="middle" font-size="24" fill="#111" font-family="Arial,sans-serif">Latest Policy vs Historical Snapshots</text>
  {"".join(grid)}
  <line x1="{margin_l}" y1="{margin_t + plot_h}" x2="{margin_l + plot_w}" y2="{margin_t + plot_h}" stroke="#222" stroke-width="2"/>
  <line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t + plot_h}" stroke="#222" stroke-width="2"/>
  <line x1="{margin_l}" y1="{zero_y:.2f}" x2="{margin_l + plot_w}" y2="{zero_y:.2f}" stroke="#8f8f8f" stroke-width="1.5" stroke-dasharray="5,4"/>
  {"".join(labels)}
  <polyline fill="none" stroke="#0f4fa8" stroke-width="3" points="{polyline}"/>
  {"".join(point_marks)}
  {"".join(update_labels)}
  <rect x="{width - 450}" y="78" width="420" height="96" fill="#fff" stroke="#bbb"/>
  <line x1="{width - 430}" y1="104" x2="{width - 360}" y2="104" stroke="#0f4fa8" stroke-width="4"/>
  <text x="{width - 350}" y="109" font-size="14" fill="#222">Rank advantage = opponent avg rank - latest avg rank</text>
  <text x="{width - 430}" y="133" font-size="14" fill="#222">Higher is better for latest policy</text>
  <text x="{width - 430}" y="157" font-size="14" fill="#222">Point label shows checkpoint update number</text>
  <text x="{width/2:.1f}" y="{height - 10}" text-anchor="middle" font-size="16" fill="#222">Historical checkpoint (% of training)</text>
  <text transform="translate(28,{height/2:.1f}) rotate(-90)" text-anchor="middle" font-size="16" fill="#222">Latest rank advantage (higher better)</text>
</svg>
"""
    output_path.write_text(svg, encoding="utf-8")


def run_latest_vs_history_benchmark(
    *,
    output_dir: Path,
    percent_points: Sequence[int],
    matches_per_point: int,
    neighbor_radius: int,
    seed: int,
    deterministic: bool,
    radius: int | None = None,
    max_steps: int | None = None,
    engine: str | None = None,
    require_c_core: bool | None = None,
) -> Dict[str, Any]:
    cfg = infer_train_config(output_dir)
    total_updates = infer_total_updates(output_dir, cfg)

    arch, hidden_dim, mem_dim = infer_policy_config(cfg)
    env_cfg = cfg.get("env", {}) if isinstance(cfg, dict) else {}
    resolved_radius = int(radius if radius is not None else env_cfg.get("radius", 5))
    resolved_max_steps = int(max_steps if max_steps is not None else env_cfg.get("max_steps", 300))
    resolved_use_c_core = (engine != "python") if engine is not None else bool(env_cfg.get("use_c_core", True))
    resolved_require_c_core = (
        bool(require_c_core) if require_c_core is not None else bool(env_cfg.get("require_c_core", False))
    )

    checkpoint_dir = output_dir / "checkpoints"
    all_snapshots = list_snapshots(checkpoint_dir)
    if not all_snapshots:
        raise RuntimeError(f"No snapshots found in {checkpoint_dir}; cannot build historical benchmark")

    final_policy = output_dir / "final_policy.pt"
    if not final_policy.exists():
        raise RuntimeError(f"Final policy not found: {final_policy}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = ObservationEncoder(ObsSpec(radius=resolved_radius, include_threat_map=True))

    latest_model, latest_hdim = build_loaded_model(
        arch=arch,
        obs_dim=encoder.obs_dim,
        hidden_dim=hidden_dim,
        mem_dim=mem_dim,
        state_path=final_policy,
        device=device,
    )

    # Filter snapshots:
    # 1) keep milestones from current configured run range
    # 2) keep only checkpoints that can be loaded by the current architecture
    candidate_snaps = [s for s in all_snapshots if s.update <= total_updates]
    if not candidate_snaps:
        candidate_snaps = all_snapshots

    compatible_snaps: List[SnapshotRef] = []
    for snap in candidate_snaps:
        try:
            _model, _hdim = build_loaded_model(
                arch=arch,
                obs_dim=encoder.obs_dim,
                hidden_dim=hidden_dim,
                mem_dim=mem_dim,
                state_path=snap.path,
                device=device,
            )
            compatible_snaps.append(snap)
        except Exception:
            continue

    if not compatible_snaps:
        raise RuntimeError(
            f"No compatible snapshots found in {checkpoint_dir} for arch={arch}, "
            f"hidden_dim={hidden_dim}, mem_dim={mem_dim}, radius={resolved_radius}"
        )

    points_data: List[Dict[str, Any]] = []
    for idx, pct in enumerate(percent_points):
        target_update = max(1, int(round(total_updates * (pct / 100.0))))
        center_snap = choose_nearest_snapshot(compatible_snaps, target_update)

        neighbor_points = neighbor_percent_points(int(pct), int(neighbor_radius))
        match_split = split_matches(matches_per_point, len(neighbor_points))
        neighbor_rows: List[Dict[str, Any]] = []

        for nidx, npct in enumerate(neighbor_points):
            n_matches = int(match_split[nidx])
            if n_matches <= 0:
                continue
            n_target_update = max(1, int(round(total_updates * (npct / 100.0))))
            n_snap = choose_nearest_snapshot(compatible_snaps, n_target_update)

            opponent_model, opponent_hdim = build_loaded_model(
                arch=arch,
                obs_dim=encoder.obs_dim,
                hidden_dim=hidden_dim,
                mem_dim=mem_dim,
                state_path=n_snap.path,
                device=device,
            )
            stats = run_head_to_head(
                latest_model=latest_model,
                latest_hidden_dim=latest_hdim,
                opponent_model=opponent_model,
                opponent_hidden_dim=opponent_hdim,
                encoder=encoder,
                device=device,
                matches=n_matches,
                seed=seed + (idx * 100_003) + (nidx * 9_973),
                max_steps=resolved_max_steps,
                use_c_core=resolved_use_c_core,
                require_c_core=resolved_require_c_core,
                deterministic=deterministic,
            )
            nrow: Dict[str, Any] = {
                "neighbor_percent": int(npct),
                "target_update": int(n_target_update),
                "checkpoint_update": int(n_snap.update),
                "checkpoint_name": n_snap.path.name,
                "matches": int(n_matches),
            }
            nrow.update(stats)
            neighbor_rows.append(nrow)

        if not neighbor_rows:
            continue

        row = {
            "percent": int(pct),
            "target_update": int(target_update),
            "checkpoint_update": int(center_snap.update),
            "checkpoint_name": center_snap.path.name,
            "matches": int(sum(int(r["matches"]) for r in neighbor_rows)),
            "neighbor_radius": int(neighbor_radius),
            "neighbors": neighbor_rows,
        }
        for key in METRIC_KEYS:
            row[key] = weighted_average(neighbor_rows, key)
        points_data.append(row)

    return {
        "meta": {
            "output_dir": str(output_dir),
            "total_updates": int(total_updates),
            "policy_arch": arch,
            "policy_hidden_dim": hidden_dim,
            "policy_mem_dim": mem_dim,
            "radius": resolved_radius,
            "max_steps": resolved_max_steps,
            "use_c_core": resolved_use_c_core,
            "require_c_core": resolved_require_c_core,
            "deterministic": deterministic,
            "matches_per_point": int(matches_per_point),
            "neighbor_radius": int(neighbor_radius),
            "percent_points": [int(p) for p in percent_points],
            "seed": int(seed),
        },
        "points": points_data,
    }


def write_latest_vs_history_outputs(
    *,
    output_dir: Path,
    percent_points: Sequence[int],
    matches_per_point: int,
    neighbor_radius: int,
    seed: int,
    deterministic: bool = True,
    radius: int | None = None,
    max_steps: int | None = None,
    engine: str | None = None,
    require_c_core: bool | None = None,
) -> Dict[str, Path]:
    result = run_latest_vs_history_benchmark(
        output_dir=output_dir,
        percent_points=percent_points,
        matches_per_point=matches_per_point,
        neighbor_radius=neighbor_radius,
        seed=seed,
        deterministic=deterministic,
        radius=radius,
        max_steps=max_steps,
        engine=engine,
        require_c_core=require_c_core,
    )

    json_path = output_dir / "latest_vs_history.json"
    svg_path = output_dir / "latest_vs_history.svg"

    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    render_rank_advantage_svg(result["points"], svg_path)
    return {"json": json_path, "svg": svg_path}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Benchmark final policy against historical checkpoints at percentage milestones."
    )
    p.add_argument("--output-dir", type=str, required=True, help="Training output directory")
    p.add_argument("--percent-points", type=str, default="10,20,30,40,50,60,70,80,90")
    p.add_argument("--matches-per-point", type=int, default=120)
    p.add_argument(
        "--neighbor-radius",
        type=int,
        default=1,
        help="Local averaging radius around each point (e.g. 1 uses p-1, p, p+1)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--radius", type=int, default=-1, help="Override radius; default uses train config")
    p.add_argument("--max-steps", type=int, default=-1, help="Override max steps; default uses train config")
    p.add_argument("--engine", choices=["auto", "c", "python"], default="auto")
    p.add_argument("--require-c-core", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    percent_points = parse_percent_points(args.percent_points)
    engine = None if args.engine == "auto" else args.engine
    radius = None if args.radius <= 0 else args.radius
    max_steps = None if args.max_steps <= 0 else args.max_steps
    written = write_latest_vs_history_outputs(
        output_dir=Path(args.output_dir),
        percent_points=percent_points,
        matches_per_point=args.matches_per_point,
        neighbor_radius=max(0, int(args.neighbor_radius)),
        seed=args.seed,
        deterministic=not args.stochastic,
        radius=radius,
        max_steps=max_steps,
        engine=engine,
        require_c_core=args.require_c_core,
    )
    print(f"Wrote {written['json']}")
    print(f"Wrote {written['svg']}")


if __name__ == "__main__":
    main()
