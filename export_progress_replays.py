from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from rl.obs import ObsSpec, ObservationEncoder
from rl.policy import build_policy
from rl.progress import run_self_play_match, write_progress_replay


def parse_update_idx(path: Path) -> int:
    m = re.search(r"snapshot_u(\d+)\.pt$", path.name)
    if not m:
        raise ValueError(f"Unrecognized checkpoint name: {path.name}")
    return int(m.group(1))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export frontend-loadable progression replays from training checkpoints."
    )
    p.add_argument("--checkpoint-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="ml_runs")
    p.add_argument("--matches-per-checkpoint", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--radius", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prefix", type=str, default="mlprog")
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--arch", choices=["auto", "gru", "planning"], default="auto")
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--mem-dim", type=int, default=16)
    p.add_argument("--engine", choices=["c", "python"], default="c")
    p.add_argument("--require-c-core", action="store_true")
    return p


def infer_policy_config(checkpoint_dir: Path, arch: str, hidden_dim: int, mem_dim: int) -> tuple[str, int, int]:
    if arch != "auto":
        return arch, hidden_dim, mem_dim
    cfg_path = checkpoint_dir.parent / "train_config.json"
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


def main() -> None:
    args = build_arg_parser().parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    checkpoints = sorted(checkpoint_dir.glob("snapshot_u*.pt"))
    if not checkpoints:
        raise SystemExit(f"No checkpoints found in {checkpoint_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = ObservationEncoder(ObsSpec(radius=args.radius, include_threat_map=True))
    arch, hidden_dim, mem_dim = infer_policy_config(checkpoint_dir, args.arch, args.hidden_dim, args.mem_dim)
    model = build_policy(arch, obs_dim=encoder.obs_dim, hidden_dim=hidden_dim, mem_dim=mem_dim).to(device)
    model.eval()

    total_written = 0
    for ckpt in checkpoints:
        update_idx = parse_update_idx(ckpt)
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state)
        model.eval()

        for match_idx in range(1, args.matches_per_checkpoint + 1):
            seed = (args.seed * 1_000_000) + (update_idx * 100) + match_idx
            snapshot = run_self_play_match(
                model,
                encoder,
                device,
                seed=seed,
                max_steps=args.max_steps,
                randomize_starts=True,
                randomize_facings=True,
                use_c_core=(args.engine != "python"),
                require_c_core=args.require_c_core,
                deterministic=not args.stochastic,
            )
            write_progress_replay(
                output_dir=output_dir,
                prefix=args.prefix,
                update_idx=update_idx,
                match_idx=match_idx,
                seed=seed,
                snapshot=snapshot,
                deterministic=not args.stochastic,
                checkpoint_name=ckpt.name,
            )
            total_written += 1

    print(
        f"Wrote {total_written} progression replay files from {len(checkpoints)} checkpoints to {output_dir}"
    )


if __name__ == "__main__":
    main()
