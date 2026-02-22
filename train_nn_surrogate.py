from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from rl.obs import iter_local_hex
from rl.nn_surrogate import (
    DistillConfig,
    PPOFineTuneConfig,
    TrainConfig,
    build_candidate_from_model,
    choose_best_candidate,
    collect_dagger_dataset_with_logits,
    collect_teacher_dataset_with_logits,
    fit_candidate,
    ppo_finetune_student,
    run_bot_validation,
    run_final_benchmark,
    split_train_test_with_logits,
    write_nn_artifacts,
)
from rl.tree_surrogate import FeatureSchema, default_feature_schema


def parse_arch_candidates(text: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for token in text.split(","):
        t = token.strip().lower()
        if not t:
            continue
        if "x" not in t:
            raise ValueError(f"Invalid candidate '{token}'. Use format like 24x12")
        left, right = t.split("x", 1)
        h1 = int(left)
        h2 = int(right)
        if h1 <= 0 or h2 <= 0:
            raise ValueError(f"Invalid candidate '{token}'. hidden sizes must be > 0")
        out.append((h1, h2))
    if not out:
        raise ValueError("No valid architecture candidates")
    return out


def parse_seed_list(text: str) -> List[int]:
    out: List[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    return out if out else [1, 2, 3]


def build_feature_schema(
    *,
    base_radius: int,
    front_radius: int,
    length_cap: int,
    include_player_info: bool,
) -> FeatureSchema:
    base = max(1, int(base_radius))
    front = max(0, int(front_radius))
    if front <= base:
        schema = default_feature_schema(local_radius=base)
        return FeatureSchema(
            tile_coords=schema.tile_coords,
            include_turn=True,
            include_length=True,
            include_prev_action=True,
            include_player_info=bool(include_player_info),
            length_cap=max(1, int(length_cap)),
        )

    coords: List[Tuple[int, int]] = []
    for q, r in iter_local_hex(front):
        if q == 0 and r == 0:
            continue
        s = -q - r
        dist = max(abs(q), abs(r), abs(s))
        if dist <= base:
            coords.append((q, r))
            continue
        # "Front cone" relative to local forward direction:
        # include tiles in the 120-degree forward sector (front-left, forward, front-right).
        if r <= 0 and (q + r) <= 0:
            coords.append((q, r))

    return FeatureSchema(
        tile_coords=tuple(coords),
        include_turn=True,
        include_length=True,
        include_prev_action=True,
        include_player_info=bool(include_player_info),
        length_cap=max(1, int(length_cap)),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Distill a compact neural policy from final_policy.pt, convert to Tronkscript, "
            "and validate conversion parity + match quality."
        )
    )
    p.add_argument("--policy", type=str, required=True, help="Path to final_policy.pt (teacher)")
    p.add_argument("--output-dir", type=str, default="nn_surrogate_outputs")

    p.add_argument("--samples", type=int, default=120_000)
    p.add_argument("--collect-seed", type=int, default=42)
    p.add_argument("--collect-max-steps", type=int, default=160)
    p.add_argument("--engine", choices=["c", "python"], default="c")
    p.add_argument("--require-c-core", action="store_true")

    p.add_argument("--feature-radius", type=int, default=2)
    p.add_argument(
        "--feature-front-radius",
        type=int,
        default=0,
        help=(
            "Optional front-extension radius. If > feature-radius, uses feature-radius as base and "
            "extends only forward-sector tiles out to this radius."
        ),
    )
    p.add_argument("--length-cap", type=int, default=20)
    p.add_argument("--include-player-info", action="store_true")

    p.add_argument("--candidate-arches", type=str, default="16x8,24x12,32x16,40x20")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--train-seed", type=int, default=123)
    p.add_argument("--train-frac", type=float, default=0.9)
    p.add_argument("--quant-scale", type=int, default=64)
    p.add_argument("--distill-ce-coef", type=float, default=1.0)
    p.add_argument("--distill-kl-coef", type=float, default=0.5)
    p.add_argument("--distill-temperature", type=float, default=2.0)
    p.add_argument("--dagger-iters", type=int, default=0)
    p.add_argument("--dagger-samples", type=int, default=30_000)
    p.add_argument("--dagger-teacher-mix", type=float, default=0.15)
    p.add_argument("--dagger-use-safety-fallback", action="store_true")
    p.add_argument("--ppo-finetune-updates", type=int, default=0)
    p.add_argument("--ppo-rollout-episodes", type=int, default=24)
    p.add_argument("--ppo-lr", type=float, default=2e-4)
    p.add_argument("--ppo-epochs", type=int, default=3)
    p.add_argument("--ppo-minibatch-size", type=int, default=512)
    p.add_argument("--ppo-entropy-coef", type=float, default=0.01)
    p.add_argument("--ppo-value-coef", type=float, default=0.5)
    p.add_argument("--ppo-survival-alpha", type=float, default=0.0)

    p.add_argument("--tick-budget", type=float, default=5000.0, help="Avg infer tick budget per call")
    p.add_argument("--parity-samples", type=int, default=1200)
    p.add_argument("--parity-seed", type=int, default=777)
    p.add_argument("--parity-max-steps", type=int, default=120)

    p.add_argument("--wait-ticks", type=int, default=1200)
    p.add_argument("--validate-steps", type=int, default=30)
    p.add_argument("--validate-seeds", type=str, default="1,2,3")

    p.add_argument("--benchmark-matches", type=int, default=120)
    p.add_argument("--benchmark-seed", type=int, default=999)
    p.add_argument("--benchmark-max-steps", type=int, default=160)
    p.add_argument("--skip-benchmark", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    policy_path = Path(args.policy)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    schema = build_feature_schema(
        base_radius=max(1, int(args.feature_radius)),
        front_radius=max(0, int(args.feature_front_radius)),
        length_cap=max(1, int(args.length_cap)),
        include_player_info=bool(args.include_player_info),
    )

    use_c_core = args.engine != "python"
    print("Collecting teacher dataset...")
    x, y, teacher_logits, collect_meta = collect_teacher_dataset_with_logits(
        policy_path=policy_path,
        schema=schema,
        target_samples=max(1, int(args.samples)),
        seed=int(args.collect_seed),
        max_steps=max(1, int(args.collect_max_steps)),
        use_c_core=use_c_core,
        require_c_core=bool(args.require_c_core),
        deterministic_teacher=True,
    )
    print(f"Collected samples: {x.shape[0]} (features={x.shape[1]})")

    x_train, y_train, logits_train, x_test, y_test, logits_test = split_train_test_with_logits(
        x,
        y,
        teacher_logits,
        seed=int(args.train_seed),
        train_frac=float(args.train_frac),
    )

    train_cfg = TrainConfig(
        epochs=max(1, int(args.epochs)),
        batch_size=max(32, int(args.batch_size)),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        seed=int(args.train_seed),
    )
    distill_cfg = DistillConfig(
        ce_coef=float(args.distill_ce_coef),
        kl_coef=float(args.distill_kl_coef),
        temperature=max(1e-6, float(args.distill_temperature)),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidates = parse_arch_candidates(args.candidate_arches)

    candidate_results = []
    for idx, (h1, h2) in enumerate(candidates):
        print(f"Fitting candidate {idx + 1}/{len(candidates)}: {h1}x{h2}")
        result = fit_candidate(
            x_train=x_train,
            y_train=y_train,
            logits_train=logits_train,
            x_test=x_test,
            y_test=y_test,
            logits_test=logits_test,
            hidden1=h1,
            hidden2=h2,
            train_cfg=train_cfg,
            distill_cfg=distill_cfg,
            quant_scale=max(1, int(args.quant_scale)),
            schema=schema,
            wait_ticks=max(1, int(args.wait_ticks)),
            parity_samples=max(100, int(args.parity_samples)),
            parity_seed=int(args.parity_seed) + idx,
            parity_max_steps=max(20, int(args.parity_max_steps)),
            use_c_core=use_c_core,
            tick_budget=float(args.tick_budget),
            device=device,
        )
        candidate_results.append(result)
        print(
            "  "
            f"test_acc={result.test_acc:.4f} "
            f"parity={result.parity_rate:.4f} "
            f"avg_tick={result.avg_tick_cost:.1f} "
            f"fits_budget={'yes' if result.fits_tick_budget else 'no'}"
        )

    best = choose_best_candidate(candidate_results)
    print(
        "Selected architecture: "
        f"{best.hidden1}x{best.hidden2} "
        f"(test_acc={best.test_acc:.4f}, parity={best.parity_rate:.4f}, avg_tick={best.avg_tick_cost:.1f})"
    )

    dagger_rounds = []
    for dagger_idx in range(max(0, int(args.dagger_iters))):
        print(
            f"Collecting DAgger data ({dagger_idx + 1}/{int(args.dagger_iters)}) "
            f"with {int(args.dagger_samples)} samples..."
        )
        x_new, y_new, logits_new, dagger_meta = collect_dagger_dataset_with_logits(
            policy_path=policy_path,
            student_model=best.model_float,
            schema=schema,
            target_samples=max(1, int(args.dagger_samples)),
            seed=int(args.collect_seed) + 1000 + dagger_idx,
            max_steps=max(1, int(args.collect_max_steps)),
            use_c_core=use_c_core,
            require_c_core=bool(args.require_c_core),
            teacher_mix_prob=float(args.dagger_teacher_mix),
            use_safety_fallback=bool(args.dagger_use_safety_fallback),
        )
        x = np.concatenate([x, x_new], axis=0)
        y = np.concatenate([y, y_new], axis=0)
        teacher_logits = np.concatenate([teacher_logits, logits_new], axis=0)

        x_train, y_train, logits_train, x_test, y_test, logits_test = split_train_test_with_logits(
            x,
            y,
            teacher_logits,
            seed=int(args.train_seed) + dagger_idx + 1,
            train_frac=float(args.train_frac),
        )

        print(
            f"Refitting selected architecture {best.hidden1}x{best.hidden2} "
            f"after DAgger round {dagger_idx + 1}..."
        )
        best = fit_candidate(
            x_train=x_train,
            y_train=y_train,
            logits_train=logits_train,
            x_test=x_test,
            y_test=y_test,
            logits_test=logits_test,
            hidden1=int(best.hidden1),
            hidden2=int(best.hidden2),
            train_cfg=train_cfg,
            distill_cfg=distill_cfg,
            quant_scale=max(1, int(args.quant_scale)),
            schema=schema,
            wait_ticks=max(1, int(args.wait_ticks)),
            parity_samples=max(100, int(args.parity_samples)),
            parity_seed=int(args.parity_seed) + 500 + dagger_idx,
            parity_max_steps=max(20, int(args.parity_max_steps)),
            use_c_core=use_c_core,
            tick_budget=float(args.tick_budget),
            device=device,
        )
        print(
            "  "
            f"test_acc={best.test_acc:.4f} "
            f"parity={best.parity_rate:.4f} "
            f"avg_tick={best.avg_tick_cost:.1f}"
        )
        dagger_rounds.append(
            {
                "round": int(dagger_idx + 1),
                "collect": dagger_meta,
                "selected_after_round": best.to_summary(),
                "samples_total": int(x.shape[0]),
            }
        )

    ppo_finetune = {"updates": 0, "history": []}
    if int(args.ppo_finetune_updates) > 0:
        print(f"Running PPO fine-tune for {int(args.ppo_finetune_updates)} updates...")
        ppo_cfg = PPOFineTuneConfig(
            updates=max(0, int(args.ppo_finetune_updates)),
            rollout_episodes_per_update=max(1, int(args.ppo_rollout_episodes)),
            max_steps=max(1, int(args.benchmark_max_steps)),
            policy_lr=float(args.ppo_lr),
            ppo_epochs=max(1, int(args.ppo_epochs)),
            minibatch_size=max(32, int(args.ppo_minibatch_size)),
            entropy_coef=float(args.ppo_entropy_coef),
            value_coef=float(args.ppo_value_coef),
            survival_alpha=float(args.ppo_survival_alpha),
            seed=int(args.train_seed) + 9000,
        )
        ppo_finetune = ppo_finetune_student(
            model=best.model_float,
            schema=schema,
            cfg=ppo_cfg,
            use_c_core=use_c_core,
            require_c_core=bool(args.require_c_core),
        )
        best = build_candidate_from_model(
            model_float=best.model_float,
            hidden1=int(best.hidden1),
            hidden2=int(best.hidden2),
            train_acc=float(best.train_acc),
            test_acc=float(best.test_acc),
            train_history=best.train_history,
            quant_scale=max(1, int(args.quant_scale)),
            schema=schema,
            wait_ticks=max(1, int(args.wait_ticks)),
            parity_samples=max(100, int(args.parity_samples)),
            parity_seed=int(args.parity_seed) + 9999,
            parity_max_steps=max(20, int(args.parity_max_steps)),
            use_c_core=use_c_core,
            tick_budget=float(args.tick_budget),
        )
        print(
            "After PPO fine-tune: "
            f"parity={best.parity_rate:.4f} avg_tick={best.avg_tick_cost:.1f}"
        )

    written = write_nn_artifacts(
        output_dir=output_dir,
        selected=best,
        schema=schema,
        wait_ticks=max(1, int(args.wait_ticks)),
    )
    print(f"Wrote {written['float_model']}")
    print(f"Wrote {written['quantized_model']}")
    print(f"Wrote {written['lib_tronkscript']}")
    print(f"Wrote {written['bot_tronkscript']}")

    validate = run_bot_validation(
        best.bot_source,
        steps=max(1, int(args.validate_steps)),
        seeds=parse_seed_list(args.validate_seeds),
    )
    print(f"Validation fatal errors: {validate['fatal_error_count']}")

    benchmark = None
    if not args.skip_benchmark:
        print("Running 50/50 Tronkscript-vs-teacher benchmark...")
        benchmark = run_final_benchmark(
            policy_path=policy_path,
            selected=best,
            schema=schema,
            matches=max(1, int(args.benchmark_matches)),
            seed=int(args.benchmark_seed),
            max_steps=max(1, int(args.benchmark_max_steps)),
            use_c_core=use_c_core,
            require_c_core=bool(args.require_c_core),
        )
        print(
            "Benchmark avg_rank (lower better): "
            f"teacher={benchmark['model']['avg_rank']:.4f} "
            f"nn_ts={benchmark['tronkscript']['avg_rank']:.4f}"
        )

    report = {
        "policy_path": str(policy_path),
        "collect": collect_meta,
        "distillation": {
            "ce_coef": float(args.distill_ce_coef),
            "kl_coef": float(args.distill_kl_coef),
            "temperature": float(args.distill_temperature),
        },
        "dagger": {
            "iters": int(args.dagger_iters),
            "samples_per_iter": int(max(1, int(args.dagger_samples))),
            "teacher_mix_prob": float(args.dagger_teacher_mix),
            "use_safety_fallback": bool(args.dagger_use_safety_fallback),
            "rounds": dagger_rounds,
        },
        "ppo_finetune": ppo_finetune,
        "data": {
            "samples_total": int(x.shape[0]),
            "samples_train": int(x_train.shape[0]),
            "samples_test": int(x_test.shape[0]),
            "feature_dim": int(x.shape[1]),
        },
        "search": {
            "candidates": [f"{h1}x{h2}" for h1, h2 in candidates],
            "tick_budget": float(args.tick_budget),
            "results": [c.to_summary() for c in candidate_results],
        },
        "selected": best.to_summary(),
        "feature_schema": {
            "tile_coords": [[int(q), int(r)] for q, r in schema.tile_coords],
            "include_turn": bool(schema.include_turn),
            "include_length": bool(schema.include_length),
            "include_prev_action": bool(schema.include_prev_action),
            "include_player_info": bool(schema.include_player_info),
            "length_cap": int(schema.length_cap),
            "names": schema.names,
        },
        "validation": validate,
        "benchmark": benchmark,
        "artifacts": {k: str(v) for k, v in written.items()},
    }
    report_path = output_dir / "nn_surrogate_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
