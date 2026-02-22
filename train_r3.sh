#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Radius-specific defaults; can still be overridden by env or CLI args.
export RADIUS="${RADIUS:-3}"
export OUTPUT_DIR="${OUTPUT_DIR:-training_outputs_main_r3}"

exec "$ROOT_DIR/train_main.sh" "$@"
