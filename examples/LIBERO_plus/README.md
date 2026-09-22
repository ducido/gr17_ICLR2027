# LIBERO-plus

[LIBERO-plus](https://github.com/sylvestf/LIBERO-plus) perturbs each base [LIBERO](../LIBERO/README.md) task along 7 independent
dimensions — Background Textures, Camera Viewpoints, Language Instructions, Light Conditions,
Objects Layout, Robot Initial States, Sensor Noise — to stress-test how much a policy's success rate
depends on visual/language nuisance factors rather than the underlying skill. It reuses the same Panda
robot/action space as stock LIBERO (`EmbodimentTag.LIBERO_PANDA`), so a checkpoint finetuned on stock
LIBERO data (see [../LIBERO/README.md](../LIBERO/README.md)) can be evaluated directly against
LIBERO-plus's perturbed task variants — no separate finetune is required.

LIBERO-plus vendors the `libero` package name itself (its `setup.py` declares `name="libero"`), so it
runs in its own isolated uv venv, kept separate from stock LIBERO's venv to avoid the two shadowing
each other. Gym environments are registered under the `libero_plus_sim/` prefix (stock LIBERO uses
`libero_sim/`) so `gr00t/eval/rollout_policy.py` can tell which `register_libero*_envs()` to call even
though both share the same `EmbodimentTag.LIBERO_PANDA`.

# One-time setup

## System prerequisites

`libero.libero.envs` unconditionally imports the `wand` package (used for one of the perturbation
categories' motion-blur augmentation), which dynamically loads ImageMagick's MagickWand library at
**import** time — without it, `from libero.libero.envs import OffScreenRenderEnv` raises `ImportError`
even for unrelated tasks. This must be available both when running the setup script and every time you
later launch an eval client.

- **Machine with apt/sudo:**
  ```bash
  sudo apt update
  sudo apt install -y libegl1-mesa-dev libglu1-mesa \
      libexpat1 libfontconfig1-dev libpython3-stdlib libmagickwand-dev
  ```
- **This HPC cluster (Lmod modules, no apt/sudo):** libEGL is already present system-wide; load
  ImageMagick for MagickWand:
  ```bash
  module load imagemagick/7.1.1-39
  ```

## LIBERO-plus assets

LIBERO-plus's extra objects/textures/scenes (multi-GB, not in the git repo) must be downloaded from
HuggingFace and unzipped into the vendored clone before running tasks that use them:
```bash
uv run hf download Sylvest/LIBERO-plus --repo-type dataset --local-dir /tmp/libero_plus_assets
unzip /tmp/libero_plus_assets/assets.zip -d external_dependencies/LIBERO-plus/libero/libero/assets
```

## Build the LIBERO-plus venv

Complete the [one-time simulation environment setup](../../README.md#one-time-simulation-environment-setup)
mentioned by the other benchmark READMEs if you haven't already, then (with the module/system
prerequisites above active) run this benchmark's setup script — it clones LIBERO-plus into
`external_dependencies/LIBERO-plus` (a plain gitignored clone, not a git submodule) and builds the
isolated venv (only needed once):
```bash
module load imagemagick/7.1.1-39   # skip on a machine with libmagickwand-dev installed via apt
bash gr00t/eval/sim/LIBERO_plus/setup_libero_plus.sh
```

# Evaluate checkpoint

Download a LIBERO-finetuned checkpoint (same checkpoints used for stock LIBERO eval — see
[../LIBERO/README.md](../LIBERO/README.md#evaluate-checkpoint)):
```bash
uv run hf download nvidia/GR00T-N1.7-LIBERO --include "libero_10/config.json" "libero_10/embodiment_id.json" "libero_10/model-*.safetensors" "libero_10/model.safetensors.index.json" "libero_10/processor_config.json" "libero_10/statistics.json" --local-dir checkpoints/GR00T-N1.7-LIBERO
```

Run client/server evaluation under the project root directory in separate terminals:

**Terminal 1 - Server** (top-level GR00T `.venv`, same as stock LIBERO — the server is
benchmark-agnostic since LIBERO-plus reuses `EmbodimentTag.LIBERO_PANDA`):
```bash
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
    --embodiment-tag LIBERO_PANDA \
    --use-sim-policy-wrapper
```

**Terminal 2 - Client** (the LIBERO-plus venv's interpreter, `--env-name` uses the
`libero_plus_sim/` prefix; load MagickWand first -- see
[System prerequisites](#system-prerequisites) -- `scripts/libero_plus.sh` does this for you, but a
direct `rollout_policy.py` invocation like this one does not):
```bash
module load imagemagick/7.1.1-39   # skip on a machine with libmagickwand-dev installed via apt
gr00t/eval/sim/LIBERO_plus/libero_plus_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n-episodes 10 \
    --policy-client-host 127.0.0.1 \
    --policy-client-port 5555 \
    --max-episode-steps 720 \
    --env-name libero_plus_sim/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_table_1 \
    --n-action-steps 8 \
    --n-envs 5
```

# Selecting tasks by perturbation category

Each suite has ~2,400-2,600 task variants (10-90x the size of the stock LIBERO suites), so tasks are
looked up by perturbation category rather than enumerated by hand. `list_tasks.py` reads LIBERO-plus's
own `task_classification.json` and prints matching `--env-name` values:

```bash
# List available categories/suites
gr00t/eval/sim/LIBERO_plus/libero_plus_uv/.venv/bin/python gr00t/eval/sim/LIBERO_plus/list_tasks.py --list-categories

# All libero_10 tasks perturbed with Sensor Noise
gr00t/eval/sim/LIBERO_plus/libero_plus_uv/.venv/bin/python gr00t/eval/sim/LIBERO_plus/list_tasks.py \
    --category "Sensor Noise" --suite libero_10
```

Categories: `Background Textures`, `Camera Viewpoints`, `Language Instructions`, `Light Conditions`,
`Objects Layout`, `Robot Initial States`, `Sensor Noise`.

Suites: `libero_10`, `libero_spatial`, `libero_object`, `libero_goal` (LIBERO-plus does not publish
perturbed variants for `libero_90`, the pretrain-only suite; its stock 90 tasks are still registered
under `libero_plus_sim/` for completeness but `list_tasks.py` won't return them since
`task_classification.json` has no `libero_90` entry).

# Running a full perturbation-category sweep

`scripts/libero_plus.sh <suite> <pertube> <port> [batch_size]` runs every task in one
suite/category against an already-running server (Terminal 1 above), writing per-task logs and
videos under `eval_logs/libero_plus/`:
```bash
bash scripts/libero_plus.sh libero_10 noise 5555
```
`pertube` is one of: `background`, `camera_view`, `language`, `layout`, `light`, `noise`,
`robot_init_state` (mapped to the 7 categories above).

**Why this is batched, not one process per task:** `gr00t.policy.server_client.PolicyServer` is a
single-threaded, blocking ZeroMQ loop (one request in flight at a time) -- running more client
processes against one server would only serialize behind each other, not parallelize. The actual
lever is batch size: `gr00t/eval/sim/LIBERO_plus/rollout_batch.py` (which the sweep script calls
once for the whole task list, rather than looping `rollout_policy.py` per task) vectorizes
`batch_size` *different* tasks per macro-step into one request, so one server-side GPU forward pass
covers the whole batch -- the same mechanism `rollout_policy.py --n-envs` uses to batch multiple
episodes of one task, generalized to batch multiple different tasks (what a sweep of ~2,500
single-episode task variants actually needs). Tune the 4th argument to fit your GPU/CPU:
```bash
bash scripts/libero_plus.sh libero_10 noise 5555 16   # 16 tasks per batch instead of the default 8
```

To sweep all 7 perturbation categories for a suite in one go, use
`scripts/libero_plus_all_pertubation.sh <suite> <port> [batch_size]`, which just loops
`scripts/libero_plus.sh` over every category:
```bash
bash scripts/libero_plus_all_pertubation.sh libero_10 5555
```

# Checking results across suites and perturbations

Each sweep run (one `scripts/libero_plus.sh` invocation, i.e. one suite x perturbation) writes, under
its `eval_logs/libero_plus/<pertube>/<suite>/.../` log root:
- `results.csv` -- one row per task (`suite,pertube,category,task,success,episode_length,episode_reward`),
  rewritten fresh each run so re-running a sweep doesn't accumulate stale/duplicate rows.
- `summary.json` -- that run's aggregate (`n_tasks`, `n_success`, `success_rate`).
- `<task_basename>/<task_basename>.txt` and `<task_basename>/videos/` -- unchanged, per-task detail.

To see a suite x perturbation success-rate table across every sweep you've run (no LIBERO/robosuite
import or venv needed -- plain `python3`):
```bash
python3 gr00t/eval/sim/LIBERO_plus/summarize_results.py
```
Add `--combine-csv /path/to/all_results.csv` to also concatenate every run's `results.csv` into one
file (e.g. for a pandas/Excel deep-dive on individual task failures across suites/perturbations).
