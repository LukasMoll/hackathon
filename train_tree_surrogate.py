from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rl.tree_surrogate import (
    DecisionTreeParams,
    benchmark_tronkscript_vs_model,
    collect_dagger_dataset,
    collect_teacher_dataset,
    default_feature_schema,
    evaluate_accuracy,
    train_bagged_ensemble,
    validate_tronkscript_bot,
    write_tree_artifacts,
)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train a decision-tree surrogate from a policy checkpoint and compile it to Tronkscript."
    )
    p.add_argument("--policy", type=str, required=True, help="Path to final_policy.pt")
    p.add_argument("--output-dir", type=str, default="tree_surrogate_outputs")
    p.add_argument("--samples", type=int, default=120_000, help="Teacher-labeled samples to collect")
    p.add_argument("--collect-seed", type=int, default=42)
    p.add_argument("--collect-max-steps", type=int, default=160)
    p.add_argument("--engine", choices=["c", "python"], default="c")
    p.add_argument("--require-c-core", action="store_true")
    p.add_argument("--feature-radius", type=int, default=2)
    p.add_argument("--length-cap", type=int, default=20)
    p.add_argument("--include-player-info", action="store_true")

    p.add_argument("--max-depth", type=int, default=10)
    p.add_argument("--min-samples-split", type=int, default=128)
    p.add_argument("--min-samples-leaf", type=int, default=64)
    p.add_argument("--min-gain", type=float, default=1e-4)
    p.add_argument("--ensemble-size", type=int, default=1)
    p.add_argument("--train-seed", type=int, default=123)
    p.add_argument("--train-frac", type=float, default=0.9)
    p.add_argument("--dagger-iters", type=int, default=0, help="Additional DAgger iterations after initial fit")
    p.add_argument("--dagger-samples", type=int, default=60_000, help="Samples per DAgger iteration")
    p.add_argument(
        "--dagger-teacher-mix",
        type=float,
        default=0.1,
        help="Teacher action probability while rolling out DAgger data",
    )

    p.add_argument("--wait-ticks", type=int, default=1200)
    p.add_argument("--validate-steps", type=int, default=30)
    p.add_argument("--validate-seeds", type=str, default="1,2,3")

    p.add_argument("--benchmark-matches", type=int, default=300)
    p.add_argument("--benchmark-seed", type=int, default=777)
    p.add_argument("--benchmark-max-steps", type=int, default=160)
    p.add_argument("--skip-benchmark", action="store_true")
    return p


def parse_seed_list(text: str) -> list[int]:
    out: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    return out if out else [1, 2, 3]


def main() -> None:
    args = build_arg_parser().parse_args()
    policy_path = Path(args.policy)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    schema = default_feature_schema(local_radius=args.feature_radius)
    schema = type(schema)(
        tile_coords=schema.tile_coords,
        include_turn=True,
        include_length=True,
        include_prev_action=True,
        include_player_info=bool(args.include_player_info),
        length_cap=max(1, int(args.length_cap)),
    )

    use_c_core = args.engine != "python"
    print("Collecting teacher dataset...")
    x, y, collect_meta = collect_teacher_dataset(
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

    params = DecisionTreeParams(
        max_depth=max(1, int(args.max_depth)),
        min_samples_split=max(2, int(args.min_samples_split)),
        min_samples_leaf=max(1, int(args.min_samples_leaf)),
        min_gain=float(args.min_gain),
    )
    ensemble_size = max(1, int(args.ensemble_size))

    x_all = x
    y_all = y
    dagger_rounds: list[dict] = []
    dagger_iters = max(0, int(args.dagger_iters))
    model = None
    train_metrics: dict[str, float] = {"accuracy": 0.0}
    test_metrics: dict[str, float] = {"accuracy": 0.0}
    x_train = x_all
    y_train = y_all
    x_test = x_all[:1]
    y_test = y_all[:1]

    for round_idx in range(dagger_iters + 1):
        rng = np.random.default_rng(int(args.train_seed) + round_idx)
        n = x_all.shape[0]
        order = np.arange(n, dtype=np.int32)
        rng.shuffle(order)
        cut = int(max(1, min(n - 1, round(n * float(args.train_frac)))))
        train_idx = order[:cut]
        test_idx = order[cut:]

        x_train, y_train = x_all[train_idx], y_all[train_idx]
        x_test, y_test = x_all[test_idx], y_all[test_idx]

        print(f"Training tree surrogate (round {round_idx}/{dagger_iters})...")
        model = train_bagged_ensemble(
            x_train,
            y_train,
            n_estimators=ensemble_size,
            params=params,
            seed=int(args.train_seed) + round_idx,
        )

        y_pred_train = model.predict(x_train)
        y_pred_test = model.predict(x_test)
        train_metrics = evaluate_accuracy(y_train, y_pred_train)
        test_metrics = evaluate_accuracy(y_test, y_pred_test)
        print(f"Round {round_idx} train acc: {train_metrics['accuracy']:.4f}")
        print(f"Round {round_idx} test  acc: {test_metrics['accuracy']:.4f}")

        round_meta = {
            "round": int(round_idx),
            "samples_total": int(x_all.shape[0]),
            "samples_train": int(x_train.shape[0]),
            "samples_test": int(x_test.shape[0]),
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
        }
        dagger_rounds.append(round_meta)

        if round_idx >= dagger_iters:
            break

        print(
            f"Collecting DAgger data (round {round_idx + 1}/{dagger_iters}) "
            f"with {int(args.dagger_samples)} samples..."
        )
        x_new, y_new, dagger_meta = collect_dagger_dataset(
            policy_path=policy_path,
            schema=schema,
            predictor=model,
            target_samples=max(1, int(args.dagger_samples)),
            seed=int(args.collect_seed) + 1000 + round_idx,
            max_steps=max(1, int(args.collect_max_steps)),
            use_c_core=use_c_core,
            require_c_core=bool(args.require_c_core),
            deterministic_teacher=True,
            teacher_mix_prob=float(args.dagger_teacher_mix),
        )
        x_all = np.concatenate([x_all, x_new], axis=0)
        y_all = np.concatenate([y_all, y_new], axis=0)
        round_meta["dagger_collect"] = dagger_meta

    assert model is not None

    written = write_tree_artifacts(
        output_dir=output_dir,
        model=model,
        schema=schema,
        wait_ticks=max(1, int(args.wait_ticks)),
    )
    print(f"Wrote {written['tree_json']}")
    print(f"Wrote {written['lib_tronkscript']}")
    print(f"Wrote {written['bot_tronkscript']}")

    bot_source = written["bot_tronkscript"].read_text(encoding="utf-8")
    lib_source = written["lib_tronkscript"].read_text(encoding="utf-8")
    validate = validate_tronkscript_bot(
        source_with_main=bot_source,
        steps=max(1, int(args.validate_steps)),
        seeds=parse_seed_list(args.validate_seeds),
    )
    print(f"Validation fatal errors: {validate['fatal_error_count']}")

    benchmark: dict | None = None
    if not args.skip_benchmark:
        print("Running 50/50 Tronkscript-vs-model benchmark...")
        benchmark = benchmark_tronkscript_vs_model(
            policy_path=policy_path,
            tronkscript_source_no_main=lib_source,
            schema=schema,
            predictor=model,
            matches=max(1, int(args.benchmark_matches)),
            seed=int(args.benchmark_seed),
            max_steps=max(1, int(args.benchmark_max_steps)),
            use_c_core=use_c_core,
            require_c_core=bool(args.require_c_core),
        )
        print(
            "Benchmark avg_rank (lower better): "
            f"model={benchmark['model']['avg_rank']:.4f} "
            f"tronkscript={benchmark['tronkscript']['avg_rank']:.4f}"
        )

    report = {
        "policy_path": str(policy_path),
        "collect": collect_meta,
        "dagger": {
            "iters": int(dagger_iters),
            "samples_per_iter": int(max(1, int(args.dagger_samples))),
            "teacher_mix": float(args.dagger_teacher_mix),
            "rounds": dagger_rounds,
        },
        "train": {
            "params": {
            "max_depth": params.max_depth,
            "min_samples_split": params.min_samples_split,
            "min_samples_leaf": params.min_samples_leaf,
            "min_gain": params.min_gain,
            "ensemble_size": ensemble_size,
        },
        "samples_total": int(x_all.shape[0]),
        "samples_train": int(x_train.shape[0]),
        "samples_test": int(x_test.shape[0]),
        "tree_max_depth": int(model.max_depth()),
        "tree_leaf_count": int(model.leaf_count()),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    },
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
    report_path = output_dir / "surrogate_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
