#!/usr/bin/env bash
# Run a LIBERO-plus sweep across ALL 7 perturbation categories against an
# already-running GR00T policy server (see examples/LIBERO_plus/README.md for
# the server command). Same per-category logic as scripts/libero_plus.sh --
# this just loops it over every pertube value instead of taking one, so the
# category-mapping/task-listing/batching logic lives in one place.
#
# Usage: scripts/libero_plus_all_pertubation.sh <suite> <port> [batch_size]
#   suite:      libero_10 | libero_spatial | libero_object | libero_goal
#               (LIBERO-plus does not publish perturbed variants for libero_90)
#   port:       port the policy server (gr00t/eval/run_gr00t_server.py) is listening on
#   batch_size: tasks per vectorized batch (default 8); see rollout_batch.py's
#               module docstring for why batching -- not more server/client
#               processes -- is what actually parallelizes this sweep
#               (PolicyServer is single-threaded and blocking).

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

suite=$1
PORT=$2
BATCH_SIZE=${3:-8}

# PERTUBATIONS=(background camera_view language layout light noise robot_init_state)
PERTUBATIONS=(background camera_view)

for pertube in "${PERTUBATIONS[@]}"; do
    echo "=== Perturbation: $pertube ==="
    bash "$SCRIPT_DIR/libero_plus.sh" "$suite" "$pertube" "$PORT" "$BATCH_SIZE"
    echo ""
done
