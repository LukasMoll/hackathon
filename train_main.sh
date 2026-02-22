#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Reusable defaults (override via env vars or CLI flags).
UPDATES="${UPDATES:-4000}"
OUTPUT_DIR="${OUTPUT_DIR:-training_outputs_main}"
ROLLOUT_EPISODES="${ROLLOUT_EPISODES:-48}"
MAX_STEPS="${MAX_STEPS:-160}"
RADIUS="${RADIUS:-4}"
ENGINE="${ENGINE:-c}"
REQUIRE_C_CORE="${REQUIRE_C_CORE:-1}"
POLICY_ARCH="${POLICY_ARCH:-planning}"
POLICY_HIDDEN_DIM="${POLICY_HIDDEN_DIM:-128}"
POLICY_MEM_DIM="${POLICY_MEM_DIM:-16}"
SURVIVAL_ALPHA="${SURVIVAL_ALPHA:-0.0}"
POLICY_LR="${POLICY_LR:-0.0003}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-256}"
EPOCHS="${EPOCHS:-4}"
EVAL_INTERVAL="${EVAL_INTERVAL:-20}"
EVAL_MATCHES="${EVAL_MATCHES:-24}"
SNAPSHOT_INTERVAL="${SNAPSHOT_INTERVAL:-20}"
SNAPSHOT_POOL_SIZE="${SNAPSHOT_POOL_SIZE:-64}"
PROGRESS_REPLAY_INTERVAL="${PROGRESS_REPLAY_INTERVAL:-50}"
PROGRESS_REPLAY_MATCHES="${PROGRESS_REPLAY_MATCHES:-1}"
PROGRESS_REPLAY_STOCHASTIC="${PROGRESS_REPLAY_STOCHASTIC:-0}"
PROGRESS_PREFIX="${PROGRESS_PREFIX:-mlprog}"
SAVE_REPLAY_INTERVAL="${SAVE_REPLAY_INTERVAL:-0}"
SAVE_REPLAY_PREFIX="${SAVE_REPLAY_PREFIX:-mltrain}"
ENTROPY_COEF="${ENTROPY_COEF:-0.01}"
LOOP_CYCLE_PENALTY="${LOOP_CYCLE_PENALTY:--0.002}"
STAGNATION_PENALTY="${STAGNATION_PENALTY:--0.001}"
STAGNATION_WINDOW="${STAGNATION_WINDOW:-20}"
STAGNATION_UNIQUE_THRESHOLD="${STAGNATION_UNIQUE_THRESHOLD:-8}"
TURN_STREAK_LIMIT="${TURN_STREAK_LIMIT:-6}"
TURN_STREAK_PENALTY="${TURN_STREAK_PENALTY:--0.02}"
HISTORY_BENCHMARK_MATCHES="${HISTORY_BENCHMARK_MATCHES:-120}"
HISTORY_BENCHMARK_POINTS="${HISTORY_BENCHMARK_POINTS:-2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90}"
HISTORY_BENCHMARK_NEIGHBOR_RADIUS="${HISTORY_BENCHMARK_NEIGHBOR_RADIUS:-0}"
NO_HISTORY_BENCHMARK="${NO_HISTORY_BENCHMARK:-0}"
NO_AUTO_PLOTS="${NO_AUTO_PLOTS:-0}"
NO_ARCHIVE_OLD_ML_RUNS="${NO_ARCHIVE_OLD_ML_RUNS:-0}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage:
  ./train_main.sh [additional train_rank_ppo.py args...]

Examples:
  ./train_main.sh
  UPDATES=100 OUTPUT_DIR=training_outputs_quick ./train_main.sh
  ./train_main.sh --total-updates 100 --output-dir training_outputs_custom

Notes:
  - Any args you pass are appended and will override defaults if duplicated.
  - Progress bar with ETA is enabled by default.
  - Set NO_HISTORY_BENCHMARK=1 to skip latest-vs-history graph generation.
EOF
  exit 0
fi

cd "$ROOT_DIR"

if [[ "${SKIP_DEPS_CHECK:-0}" != "1" ]]; then
  if ! "$PYTHON_BIN" - <<'PY'
import importlib.util
import sys
missing = [name for name in ("numpy", "torch") if importlib.util.find_spec(name) is None]
if missing:
    print("MISSING:" + ",".join(missing))
    sys.exit(1)
PY
  then
    echo "Installing missing Python dependencies from requirements.txt..."
    "$PYTHON_BIN" -m pip install --upgrade pip >/dev/null
    "$PYTHON_BIN" -m pip install -r requirements.txt
  fi
fi

cmd=(
  "$PYTHON_BIN" train_rank_ppo.py
  --output-dir "$OUTPUT_DIR"
  --total-updates "$UPDATES"
  --rollout-episodes "$ROLLOUT_EPISODES"
  --policy-arch "$POLICY_ARCH"
  --policy-hidden-dim "$POLICY_HIDDEN_DIM"
  --policy-mem-dim "$POLICY_MEM_DIM"
  --survival-alpha "$SURVIVAL_ALPHA"
  --policy-lr "$POLICY_LR"
  --minibatch-size "$MINIBATCH_SIZE"
  --epochs "$EPOCHS"
  --max-steps "$MAX_STEPS"
  --radius "$RADIUS"
  --engine "$ENGINE"
  --eval-interval "$EVAL_INTERVAL"
  --eval-matches "$EVAL_MATCHES"
  --snapshot-interval "$SNAPSHOT_INTERVAL"
  --snapshot-pool-size "$SNAPSHOT_POOL_SIZE"
  --progress-replay-interval "$PROGRESS_REPLAY_INTERVAL"
  --progress-replay-matches "$PROGRESS_REPLAY_MATCHES"
  --progress-prefix "$PROGRESS_PREFIX"
  --save-replay-interval "$SAVE_REPLAY_INTERVAL"
  --save-replay-prefix "$SAVE_REPLAY_PREFIX"
  --entropy-coef "$ENTROPY_COEF"
  --loop-cycle-penalty "$LOOP_CYCLE_PENALTY"
  --stagnation-penalty "$STAGNATION_PENALTY"
  --stagnation-window "$STAGNATION_WINDOW"
  --stagnation-unique-threshold "$STAGNATION_UNIQUE_THRESHOLD"
  --turn-streak-limit "$TURN_STREAK_LIMIT"
  --turn-streak-penalty "$TURN_STREAK_PENALTY"
  --history-benchmark-matches "$HISTORY_BENCHMARK_MATCHES"
  --history-benchmark-points "$HISTORY_BENCHMARK_POINTS"
  --history-benchmark-neighbor-radius "$HISTORY_BENCHMARK_NEIGHBOR_RADIUS"
)

if [[ "$REQUIRE_C_CORE" == "1" ]]; then
  cmd+=(--require-c-core)
fi
if [[ "$PROGRESS_REPLAY_STOCHASTIC" == "1" ]]; then
  cmd+=(--progress-replay-stochastic)
fi
if [[ "$NO_HISTORY_BENCHMARK" == "1" ]]; then
  cmd+=(--no-history-benchmark)
fi
if [[ "$NO_AUTO_PLOTS" == "1" ]]; then
  cmd+=(--no-auto-plots)
fi
if [[ "$NO_ARCHIVE_OLD_ML_RUNS" == "1" ]]; then
  cmd+=(--no-archive-old-ml-runs)
fi

cmd+=("$@")
exec "${cmd[@]}"
