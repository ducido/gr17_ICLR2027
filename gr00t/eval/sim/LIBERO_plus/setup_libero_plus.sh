#!/usr/bin/env bash
set -euxo pipefail

# One-time setup for the LIBERO-plus benchmark (https://github.com/sylvestf/LIBERO-plus).
# Mirrors gr00t/eval/sim/LIBERO/setup_libero.sh (same isolated-uv-venv, same
# py3.12/torch2.9.0/mujoco pins that keep robosuite==1.4.0 working on py3.12),
# but points at the vendored LIBERO-plus clone instead of stock LIBERO's
# submodule, since LIBERO-plus is not wired as a git submodule here (see
# LIBERO_REPO below).
#
# System prerequisites (not done by this script):
#   On a machine with apt/sudo:
#     sudo apt update
#     sudo apt install -y libegl1-mesa-dev libglu1-mesa \
#         libexpat1 libfontconfig1-dev libpython3-stdlib libmagickwand-dev
#   On this HPC cluster (no apt/sudo, Lmod modules instead): libEGL is already
#   present system-wide, but MagickWand is not -- `module load
#   imagemagick/7.1.1-39` below covers it for this script's own smoke test;
#   scripts/libero_plus.sh and any other later invocation must load it too
#   (libero.libero.envs unconditionally imports `wand`, which dlopen()s
#   libMagickWand at import time -- without it, `from libero.libero.envs
#   import OffScreenRenderEnv` raises ImportError even for unrelated tasks).
# Verified on this cluster: `module load imagemagick/7.1.1-39` makes
# `from wand.api import library` succeed (resolves
# libMagickWand-7.Q16HDRI.so via LD_LIBRARY_PATH); it is not needed at
# `pip install`-time, only at import-time.
if command -v module >/dev/null 2>&1; then
  module load imagemagick/7.1.1-39
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set paths relative to script location
LIBERO_REPO="$SCRIPT_DIR/../../../../external_dependencies/LIBERO-plus"
PROJECT_REPO="$SCRIPT_DIR/../../../.."
LIBERO_UV_ENV="$SCRIPT_DIR/libero_plus_uv"

# LIBERO-plus is vendored as a plain (gitignored) clone rather than a git
# submodule -- `setup.py` there declares `name="libero"`, so installing it
# editable shadows the stock `libero` package for any interpreter running
# inside this venv. Clone it if it isn't already present.
if [ ! -d "$LIBERO_REPO/.git" ]; then
  git clone https://github.com/sylvestf/LIBERO-plus "$LIBERO_REPO"
fi

rm -rf $LIBERO_UV_ENV
mkdir -p $LIBERO_UV_ENV
uv venv $LIBERO_UV_ENV/.venv --python 3.12
source $LIBERO_UV_ENV/.venv/bin/activate

# robomimic (pulled by LIBERO-plus's requirements) depends on egl-probe==1.0.2,
# which builds a CMake extension whose CMakeLists.txt requires a pre-3.5 CMake
# minimum-required version -- removed entirely in CMake>=4.0. Put an older
# cmake on PATH (venv bin/ is prepended to PATH) before it gets built.
uv pip install "cmake==3.18.4.post1"

# LIBERO-plus's pinned requirements predate Python 3.12 (it forked from stock
# LIBERO). Patch only the pins that otherwise build from source or pull
# source-only transitive deps on py3.12 -- same replacement table as
# setup_libero.sh, kept in sync deliberately.
PATCHED_REQUIREMENTS="$LIBERO_UV_ENV/requirements-py312.txt"
LIBERO_REPO="$LIBERO_REPO" PATCHED_REQUIREMENTS="$PATCHED_REQUIREMENTS" python - <<'PY'
import os
from pathlib import Path

# py3.12: wandb 0.13.1 -> pathtools -> imports the removed stdlib `imp` module.
replacements = {
    "hydra-core": "hydra-core==1.3.2",
    "numpy": "numpy==1.26.4",
    "transformers": "transformers==4.57.3",
    "opencv-python": "opencv-python==4.10.0.84",
    "matplotlib": "matplotlib==3.9.4",
    "wandb": "wandb==0.18.7",
    "thop": "thop==0.1.1.post2209072238",  # 0.1.1-2209072238 isn't PEP 440
}

src = Path(os.environ["LIBERO_REPO"]) / "requirements.txt"
dst = Path(os.environ["PATCHED_REQUIREMENTS"])
lines = []
for raw in src.read_text().splitlines():
    stripped = raw.strip()
    if not stripped or stripped.startswith("#"):
        lines.append(raw)
        continue
    name = stripped.split("==", 1)[0].strip().lower()
    lines.append(replacements.get(name, stripped))
dst.write_text("\n".join(lines) + "\n")
PY
uv pip install --requirements $PATCHED_REQUIREMENTS
uv pip install -e $LIBERO_REPO --config-settings editable_mode=compat
# py3.12 pins: stop the resolver backtracking numba/llvmlite to the 3.10-only build
uv pip install torch==2.9.0 torchvision==0.24.0 pydantic av tianshou==0.5.1 numba==0.65.1 llvmlite==0.47.0 tyro pandas dm_tree einops==0.8.1 albumentations==1.4.18 zmq
uv pip install transformers==4.57.3 msgpack==1.1.0 msgpack-numpy==0.4.8 gymnasium==0.29.1
uv pip install --requirements $LIBERO_REPO/extra_requirements.txt
# Pin mujoco: robosuite 1.4.0 (pulled by LIBERO-plus's requirements) calls
# mj_fullM(model, dst, M), whose signature changed in mujoco 3.10.0 (2026-06-22)
# to mj_fullM(model, data, dst). mujoco is otherwise unpinned here, so it floats
# to the latest release and crashes env creation. Pin below the break (matches
# the stock-LIBERO island's existing mujoco==3.3.1 pin).
uv pip install numpy==1.26.4 mujoco==3.3.1

# Expose gr00t from the repo root via a .pth: no dependency re-resolution, and
# the island supplies gr00t's runtime deps itself (matches the stock-LIBERO island).
python -c "import sysconfig, pathlib; pathlib.Path(sysconfig.get_path('purelib'), 'gr00t.pth').write_text(pathlib.Path('$PROJECT_REPO').resolve().as_posix() + '\n')"

# LIBERO-plus's extra objects/textures/scenes (~multi-GB) are not in the repo;
# download from HuggingFace and unzip into libero/libero/assets/ per
# https://github.com/sylvestf/LIBERO-plus before running the smoke test below:
#   uv run hf download Sylvest/LIBERO-plus --repo-type dataset --local-dir /tmp/libero_plus_assets
#   unzip /tmp/libero_plus_assets/assets.zip -d "$LIBERO_REPO/libero/libero/assets"

rm -rf $HOME/.libero
printf 'n\n' | python -c "from gr00t.eval.sim.LIBERO_plus.libero_plus_env import register_libero_plus_envs"
python - <<'PY'
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
from gr00t.eval.sim.LIBERO_plus.libero_plus_env import register_libero_plus_envs
register_libero_plus_envs()
import gymnasium as gym
env = gym.make("libero_plus_sim/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_table_1")
env.reset()
env.close()
print("Env OK:", type(env))
PY
