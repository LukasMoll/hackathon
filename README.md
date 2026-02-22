# Tronk Engine Replica (Python Backend + Web Frontend)

This project implements a local replica of the Tronk game engine described in `exports/` and exposes it through a Python backend and a browser UI where you can load code for 6 players and run simulations.

## What is implemented

- Hex board with radius `5` (91 tiles).
- 6 player spawns on board corners.
- Simultaneous movement every `10,000` ticks.
- Collision handling:
  - move out of board -> die
  - move onto occupied tile -> die
  - two or more players moving to same destination tile -> all die
- Dead player bodies remain on the board.
- Gem behavior:
  - 6 initial gems near center
  - pickup grows player by 1 and respawns one gem on a random empty tile
  - if no gem pickup for 50,000 ticks, one extra gem spawns every movement step until next pickup
- Tick-based bot execution model with per-function tick costs.
- Tronkscript parser/interpreter with:
  - global + local scopes
  - `if / else if / else`
  - `while`
  - `for (i = a to b)`
  - functions + return (single and multi-value)
  - assignments and compound assignments
  - built-ins from docs (`print`, `wait`, `min`, `max`, etc.)
  - Tronk game API (`turnLeft`, `turnRight`, `turnForward`, `turn`, `getTurn`, `getTileAbs`, `getTileRel`, `relToAbs`, `getTick`, `getPlayerId`, `getPlayerInfo`)

## API

- `GET /api/health`
- `GET /api/default-bot`
- `GET /api/board`
- `POST /api/simulate`
- `POST /api/compare`
- `POST /api/ml/new`
- `GET /api/ml/state/{session_id}`
- `POST /api/ml/step`
- `POST /api/ml/call`
- `POST /api/ml/save`
- `GET /api/ml/saved`
- `POST /api/ml/load`

`POST /api/simulate` body:

```json
{
  "bots": ["...", "...", "...", "...", "...", "..."],
  "seed": 1,
  "max_steps": 220,
  "engine": "c"
}
```

`engine` options:

- `"c"`: use the ctypes-backed C core (default)
- `"python"`: use the pure Python core
- `"auto"`: treated as `"c"` (falls back to Python if C core is unavailable)

`POST /api/compare` runs both Python and C cores with the same inputs and returns a parity report.

## ML interface (no tick budgeting, direct actions)

The ML interface is a separate environment intended for training:

- No code upload and no interpreter execution.
- Every step receives direct actions for all 6 players: `left`, `forward`, or `right` (or `-1/0/1`).
- Movement/collision/gem rules remain the same as Tronk.
- No function tick costs are applied.

Open the UI at:

- [http://127.0.0.1:8000/ml](http://127.0.0.1:8000/ml)

Create a session:

```json
POST /api/ml/new
{
  "seed": 1,
  "max_steps": 300
}
```

Step the environment:

```json
POST /api/ml/step
{
  "session_id": "abc123...",
  "actions": ["left", "forward", "right", "forward", "left", "right"]
}
```

Query board/player functions (same semantics as Tronk docs, no tick costs):

```json
POST /api/ml/call
{
  "session_id": "abc123...",
  "player_id": 0,
  "function": "getTileRel",
  "args": [0, -1]
}
```

Supported query functions:

- `getPlayerId`
- `getTurn`
- `getTileAbs`
- `getTileRel`
- `relToAbs`
- `getTick` (returns ML step counter)
- `getPlayerInfo`

Save and load episodes:

```json
POST /api/ml/save
{
  "session_id": "abc123...",
  "name": "experiment_01"
}
```

- Saved files are written to `/Users/user/Documents/projects/botbattle/ml_runs`.
- List via `GET /api/ml/saved`.
- Load via `POST /api/ml/load` with `{ "filename": "..." }`.

## Run

1. Create a virtual environment and install dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Start the server.

```bash
uvicorn app:app --reload
```

3. Open:

- [http://127.0.0.1:8000](http://127.0.0.1:8000)

## Compatibility notes

The implementation is aligned with the updated function-level docs in `exports/botbattle.be_docs_tronk.md` and `exports/botbattle.be_docs_tronkscript.md`, including:

- `turn(dir)` only accepts relative direction values `-1`, `0`, `1`.
- `getTurn()` returns relative direction `-1`, `0`, `1`.
- `getPlayerInfo(id)` returns five values: `(alive, headQ, headR, headFacing, length)`.
- `getTileAbs/getTileRel` return `(exists, isEmpty, playerId, isGem)` where `isEmpty=0` when a gemstone is present.
- `reverse(x)` performs 32-bit bit-reversal.
- `rand()` defaults to `[0, 0xffff]`.

If you still observe divergence with the official simulator, provide one reproducible fixture (all 6 bot files + seed + first mismatch step), and it can be narrowed down quickly.

## C core

The simulator includes a native core implementation in `/Users/user/Documents/projects/botbattle/c_core/tronk_core.c` loaded through `ctypes` via `/Users/user/Documents/projects/botbattle/tronk_ctypes.py`.

- The shared library is auto-built on first use.
- If the C build/load fails, simulation automatically falls back to the Python core.
- Both cores can be compared through `/api/compare`.

## PPO training system (rank-first)

The project now includes a dedicated multi-agent PPO training stack for the ML interface:

- Shared planning actor-critic policy (Dense 128 -> Dense 128 -> heads + learned memory update)
- Action space: `left`, `forward`, `right`
- Rank-first terminal reward (default) with optional per-step shaping
- Anti-loop shaping to reduce circling local optima
- League self-play with snapshot pool and scripted opponents
- Training replay export to `ml_runs/` for frontend review at `/ml`

Code locations:

- `/Users/user/Documents/projects/botbattle/train_rank_ppo.py` (training entrypoint)
- `/Users/user/Documents/projects/botbattle/evaluate_rank_ppo.py` (evaluation entrypoint)
- `/Users/user/Documents/projects/botbattle/rl/config.py` (all hyperparameters)
- `/Users/user/Documents/projects/botbattle/rl/obs.py` (egocentric rotated observation encoder)
- `/Users/user/Documents/projects/botbattle/rl/policy.py` (recurrent shared actor-critic)
- `/Users/user/Documents/projects/botbattle/rl/ppo.py` (PPO update + GAE)
- `/Users/user/Documents/projects/botbattle/rl/league.py` (snapshot pool logic)
- `/Users/user/Documents/projects/botbattle/rl/trainer.py` (full training loop)

### Observation design (training)

- Egocentric hex disk, configurable radius (default `R=5`)
- Rotated so agent-facing direction is always forward in the input
- Per-cell channels:
  - `outside_board_mask`
  - `my_head`
  - `my_body`
  - `enemy_head`
  - `enemy_body`
  - 6 enemy-facing one-hot channels
  - threat map channel (recommended)
- Global features:
  - alive opponents count
  - phase (`step / max_steps`)
  - own length
  - alive flag

### Reward design (default)

Terminal rank rewards:

- 1st: `+1.00`
- 2nd: `+0.60`
- 3rd: `+0.25`
- 4th: `0.00`
- 5th: `-0.25`
- 6th: `-0.60`

- Optional per-step survival reward:
  - `+alpha` (default `0.0`, disabled)

No gem reward is used.
Anti-loop shaping (turn streak / cycle / stagnation penalties) remains enabled by default.

### Train

```bash
python3 train_rank_ppo.py \
  --total-updates 500 \
  --rollout-episodes 64 \
  --policy-arch planning \
  --max-steps 300 \
  --radius 5 \
  --engine c \
  --require-c-core \
  --survival-alpha 0.0 \
  --entropy-coef 0.02 \
  --loop-cycle-penalty -0.002 \
  --stagnation-penalty -0.001 \
  --progress-replay-interval 10 \
  --progress-replay-matches 2 \
  --progress-prefix mlprog \
  --save-replay-prefix mltrain \
  --output-dir training_outputs
```

Artifacts:

- `training_outputs/final_policy.pt`
- `training_outputs/metrics.jsonl`
- `training_outputs/survival_progress.svg`
- `training_outputs/placement_progress.svg`
- `training_outputs/latest_vs_history.svg` (final policy benchmarked vs historical checkpoints)
- `training_outputs/latest_vs_history.json` (raw benchmark metrics)
- `training_outputs/checkpoints/snapshot_u*.pt`
- training replays in `ml_runs/mltrain_u*_ep*.json` (loadable from `/ml`)
- progression replays in `ml_runs/mlprog_u*_m*.json` (loadable from `/ml`)

`metrics.jsonl` now includes `env_core_mode` so you can verify training is running on the C core.
`metrics.jsonl` also includes `rollout_pool_mode` (`single`) to reflect the deterministic single-process trainer.
`placement_progress.svg` is the recommended late-stage metric: it tracks normalized rank-objective score (0-100).
`latest_vs_history.svg` is the fixed-opponent benchmark: it compares the final policy against snapshots nearest to
10/20/30/.../90% training progress to remove moving-opponent bias.
By default each point is smoothed with local neighbors (`p-1,p,p+1`) to reduce variance.

To run this benchmark manually:

```bash
python3 benchmark_latest_vs_history.py \
  --output-dir training_outputs \
  --percent-points 10,20,30,40,50,60,70,80,90 \
  --matches-per-point 120 \
  --neighbor-radius 1
```

You can skip this benchmark during training with `--no-history-benchmark` or tune smoothing with
`--history-benchmark-neighbor-radius`.

### Visualize policy progress in frontend

The trainer now exports progression matches every `N` updates (`--progress-replay-interval`) with
`--progress-replay-matches` matches each. Files are written to `ml_runs/mlprog_u*_m*.json`.

To generate progression replays from existing checkpoints without retraining:

```bash
python3 export_progress_replays.py \
  --checkpoint-dir training_outputs/checkpoints \
  --output-dir ml_runs \
  --matches-per-checkpoint 2 \
  --max-steps 300
```

Then open `/ml`, click `Refresh saved`, and load any `mlprog_u...` file from the saved list.

### Evaluate

```bash
python3 evaluate_rank_ppo.py --policy training_outputs/final_policy.pt --matches 200 --radius 5 --engine c
```

### Compile To Tronkscript (Tree Surrogate)

To compile a policy checkpoint into a large integer-only `if/else` Tronkscript policy and
benchmark it 50/50 against the original ML policy:

```bash
python3 train_tree_surrogate.py \
  --policy training_outputs/final_policy.pt \
  --output-dir tree_surrogate_outputs \
  --samples 120000 \
  --collect-max-steps 160 \
  --feature-radius 2 \
  --include-player-info \
  --max-depth 12 \
  --min-samples-split 64 \
  --min-samples-leaf 24 \
  --ensemble-size 7 \
  --dagger-iters 2 \
  --dagger-samples 60000 \
  --dagger-teacher-mix 0.15 \
  --benchmark-matches 300 \
  --benchmark-max-steps 160 \
  --engine c \
  --require-c-core
```

Generated artifacts:

- `tree_surrogate_outputs/tree_model.json`
- `tree_surrogate_outputs/tree_policy_lib.tronkscript` (inference function only)
- `tree_surrogate_outputs/tree_policy_bot.tronkscript` (full runnable bot)
- `tree_surrogate_outputs/surrogate_report.json` (train/test imitation + 50/50 benchmark)

Notes:

- `validation.fatal_error_count` in the report counts only parse/runtime/program-exit failures.
- Non-fatal game deaths (wall/collision) are expected gameplay outcomes and are listed in `validation.runs[*].errors`.
- Generated Tronkscript includes a safety fallback that overrides immediate wall/body collisions using the tree votes.
- If benchmark runtime is too high with large trees/ensembles, reduce `--benchmark-matches` first.

### Compile To Tronkscript (Compact NN Surrogate)

To distill the teacher policy into a compact neural network that is explicitly
constrained for Tronkscript conversion, run:

```bash
python3 train_nn_surrogate.py \
  --policy training_outputs/final_policy.pt \
  --output-dir nn_surrogate_outputs \
  --samples 120000 \
  --collect-max-steps 160 \
  --feature-radius 2 \
  --candidate-arches 16x8,24x12,32x16,40x20 \
  --distill-kl-coef 0.5 \
  --distill-temperature 2.0 \
  --dagger-iters 2 \
  --dagger-samples 30000 \
  --dagger-teacher-mix 0.15 \
  --dagger-use-safety-fallback \
  --ppo-finetune-updates 12 \
  --ppo-rollout-episodes 24 \
  --tick-budget 5000 \
  --parity-samples 1200 \
  --benchmark-matches 120 \
  --benchmark-max-steps 160 \
  --engine c \
  --require-c-core
```

This workflow:

- collects teacher-labeled data from `final_policy.pt`
- searches compact architectures that fit a Tronkscript tick budget
- supports soft-logit distillation (KL) in addition to hard action labels
- supports DAgger rounds on student-induced states
- supports short rank-aware PPO fine-tuning on the selected compact student
- quantizes the selected NN to fixed-point integers
- compiles NN inference to Tronkscript
- validates Python-quantized vs Tronkscript parity
- benchmarks 50/50 versus the teacher model

Generated artifacts:

- `nn_surrogate_outputs/nn_student_float.pt`
- `nn_surrogate_outputs/nn_student_quantized.json`
- `nn_surrogate_outputs/nn_policy_lib.tronkscript`
- `nn_surrogate_outputs/nn_policy_bot.tronkscript`
- `nn_surrogate_outputs/nn_surrogate_report.json`

### Plot survival progress

To visualize how average survival changes across training updates:

```bash
python3 plot_training_survival.py \
  --metrics training_outputs/metrics.jsonl \
  --output training_outputs/survival_progress.svg \
  --ma-window 10
```
