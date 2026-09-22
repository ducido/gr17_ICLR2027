#!/usr/bin/env bash
# Run a LIBERO-plus perturbation sweep against an already-running GR00T policy
# server (see examples/LIBERO_plus/README.md for the server command).
#
# Usage: scripts/libero_plus.sh <suite> <pertube> <port> [batch_size]
#   suite:      libero_10 | libero_spatial | libero_object | libero_goal
#               (LIBERO-plus does not publish perturbed variants for libero_90)
#   pertube:    background | camera_view | language | layout | light | noise | robot_init_state
#   port:       port the policy server (gr00t/eval/run_gr00t_server.py) is listening on
#   batch_size: tasks per vectorized batch (default 8); see rollout_batch.py's
#               module docstring for why batching -- not more server/client
#               processes -- is what actually parallelizes this sweep
#               (PolicyServer is single-threaded and blocking).
#
# Adapted from the working reference implementation at
# gr17_tta/scripts/libero_plus.sh for this repo's architecture:
#   - The task-listing helper lives at gr00t/eval/sim/LIBERO_plus/list_tasks.py
#     (no top-level scripts/plus/ dir in this repo).
#   - Adds the "light" -> "Light Conditions" mapping, which was missing from
#     the reference script even though task_classification.json defines it.
#   - Runs the whole filtered task list through ONE
#     gr00t/eval/sim/LIBERO_plus/rollout_batch.py process (vectorized batches
#     of `batch_size` DIFFERENT tasks per policy request) instead of spawning
#     a fresh gr00t/eval/rollout_policy.py process per task: the reference
#     script's one-python-process-per-task loop pays LIBERO/robosuite/mujoco
#     import and scene-load overhead on every single task and never batches
#     the GPU forward pass, which dominates wall time for a sweep of
#     hundreds-to-thousands of single-episode tasks.

export CUDA_VISIBLE_DEVICES=0
# gr00t.eval.sim.LIBERO_plus unconditionally imports `wand`, which dlopen()s
# libMagickWand at import time -- every task fails with "MagickWand shared
# library not found" without it. On this HPC cluster (Lmod modules, no
# apt/sudo) that means loading the ImageMagick module; do it here rather than
# relying on the caller's shell already having it loaded. `command -v module`
# guards this on machines without Lmod (e.g. apt-based, where
# libmagickwand-dev already puts the library on the default search path).
if command -v module >/dev/null 2>&1; then
    module load imagemagick/7.1.1-39
fi

suite=$1
pertube=$2
PORT=$3
BATCH_SIZE=${4:-8}

PYTHON=gr00t/eval/sim/LIBERO_plus/libero_plus_uv/.venv/bin/python

if [ "$pertube" == "background" ]; then
    CATEGORY="Background Textures"
elif [ "$pertube" == "noise" ]; then
    CATEGORY="Sensor Noise"
elif [ "$pertube" == "robot_init_state" ]; then
    CATEGORY="Robot Initial States"
elif [ "$pertube" == "language" ]; then
    CATEGORY="Language Instructions"
elif [ "$pertube" == "camera_view" ]; then
    CATEGORY="Camera Viewpoints"
elif [ "$pertube" == "layout" ]; then
    CATEGORY="Objects Layout"
elif [ "$pertube" == "light" ]; then
    CATEGORY="Light Conditions"
else
    echo "pertube $pertube does not belong to one of the allowed values"
    exit 1
fi

TASKS_FILE=$(mktemp)
trap 'rm -f "$TASKS_FILE"' EXIT
"$PYTHON" gr00t/eval/sim/LIBERO_plus/list_tasks.py --category "$CATEGORY" --suite "$suite" > "$TASKS_FILE" || exit 1

N_TASKS=$(wc -l < "$TASKS_FILE")
if [ "$N_TASKS" -eq 0 ]; then
    echo "No tasks found for category '$CATEGORY' in suite '$suite'" >&2
    exit 1
fi
echo "Loaded $N_TASKS tasks for category '$CATEGORY'; running in batches of $BATCH_SIZE"

action_horizon=8
max_episode_steps=720

LOG_ROOT="eval_logs/libero_plus/${pertube}/${suite}/baseline_20k_${max_episode_steps}steps_eps1_ah${action_horizon}"
mkdir -p "$LOG_ROOT"

"$PYTHON" gr00t/eval/sim/LIBERO_plus/rollout_batch.py \
    --tasks-file "$TASKS_FILE" \
    --log-root "$LOG_ROOT" \
    --policy-client-host 127.0.0.1 \
    --policy-client-port $PORT \
    --max-episode-steps $max_episode_steps \
    --n-action-steps $action_horizon \
    --batch-size $BATCH_SIZE \
    --suite "$suite" \
    --pertube "$pertube" \
    --category "$CATEGORY" \
    2>&1 | tee "$LOG_ROOT/sweep.log"
