#!/usr/bin/env bash
# Run a LIBERO-plus perturbation sweep against an already-running GR00T policy
# server (see examples/LIBERO_plus/README.md for the server command).
#
# Usage: scripts/libero_plus.sh <suite> <pertube> <port>
#   suite:   libero_10 | libero_spatial | libero_object | libero_goal
#            (LIBERO-plus does not publish perturbed variants for libero_90)
#   pertube: background | camera_view | language | layout | light | noise | robot_init_state
#   port:    port the policy server (gr00t/eval/run_gr00t_server.py) is listening on
#
# Adapted from the working reference implementation at
# gr17_tta/scripts/libero_plus.sh for this repo's architecture:
#   - gr00t/eval/rollout_policy.py is a single generic rollout driver here
#     (no separate rollout_policy_libero_plus.py -- dispatch is by env_name
#     prefix, see gr00t/eval/sim/env_utils.py), and its CLI flags are
#     hyphenated (tyro dataclass), not underscored.
#   - The task-listing helper lives at gr00t/eval/sim/LIBERO_plus/list_tasks.py
#     (no top-level scripts/plus/ dir in this repo).
#   - Adds the "light" -> "Light Conditions" mapping, which was missing from
#     the reference script even though task_classification.json defines it.

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

mapfile -t TASKS < <("$PYTHON" gr00t/eval/sim/LIBERO_plus/list_tasks.py --category "$CATEGORY" --suite "$suite") || exit 1

if [ ${#TASKS[@]} -eq 0 ]; then
    echo "No tasks found for category '$CATEGORY' in suite '$suite'" >&2
    exit 1
fi
echo "Loaded ${#TASKS[@]} tasks for category '$CATEGORY'"

action_horizon=8
EPISODES=1
N_envs=1
max_episode_steps=720

for TASK in "${TASKS[@]}"; do
    NAME=$(basename "$TASK")

    LOG_DIR="eval_logs/libero_plus/${pertube}/${suite}/baseline_20k_${max_episode_steps}steps_eps${EPISODES}_ah${action_horizon}/$NAME"
    VIDEO_DIR="$LOG_DIR/videos"
    mkdir -p "$LOG_DIR"
    mkdir -p "$VIDEO_DIR"

    echo "Running task: $TASK"

    "$PYTHON" gr00t/eval/rollout_policy.py \
        --n-episodes $EPISODES \
        --policy-client-host 127.0.0.1 \
        --policy-client-port $PORT \
        --max-episode-steps $max_episode_steps \
        --env-name "$TASK" \
        --n-action-steps $action_horizon \
        --n-envs $N_envs \
        --video-dir "$VIDEO_DIR" \
        > "$LOG_DIR/${NAME}.txt" 2>&1

    echo "Finished task: $TASK"
    echo ""
done
