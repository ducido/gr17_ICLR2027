# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
import sys
import time
from typing import Any
import uuid

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.deployment.modes import InferenceMode
from gr00t.eval._horizon_contract import PolicyHorizonSpec
from gr00t.eval.sim.env_utils import get_embodiment_tag_from_env_name
from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper
from gr00t.policy import BasePolicy
from gr00t.utils.determinism import seed_everything
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import tyro


ROBOCASA_PANDA_RECORD_VIDEO_KEYS = (
    "video.res256_image_side_0",
    "video.res256_image_side_1",
    "video.res256_image_wrist_0",
)

ROBOCASA365_PANDA_RECORD_VIDEO_KEYS = (
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
)

ROBOCASA_RECORD_VIDEO_KEYS_BY_PREFIX = {
    "robocasa_panda_omron": ROBOCASA_PANDA_RECORD_VIDEO_KEYS,
    "robocasa365_panda_omron": ROBOCASA365_PANDA_RECORD_VIDEO_KEYS,
}

# Canonical per-episode step budget for the default (LIBERO) backend. Kept in one
# place so the CLI, video, and multi-step defaults cannot drift.
DEFAULT_MAX_EPISODE_STEPS = 720


@dataclass
class VideoConfig:
    """Configuration for video recording settings.

    Attributes:
        video_dir: Directory to save videos (if None, no videos are saved)
        steps_per_render: Number of steps between each call to env.render() while recording
            during rollout
        fps: Frames per second for the output video
        codec: Video codec to use for compression
    """

    video_dir: str | None = None
    steps_per_render: int = 2
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS
    fps: int = 20
    codec: str = "h264"
    overlay_text: bool = True
    record_video_keys: tuple[str, ...] | None = None


@dataclass
class MultiStepConfig:
    """Configuration for multi-step environment settings.

    Attributes:
        contract: policy-resolved :class:`PolicyHorizonSpec` carrying
            ``n_action_steps`` and the video / state delta-indices.
        max_episode_steps: Maximum number of steps per episode.
        terminate_on_success: End the episode once the task is reported solved.
    """

    contract: PolicyHorizonSpec
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS
    terminate_on_success: bool = False


@dataclass
class WrapperConfigs:
    """Container for various environment wrapper configurations.

    Attributes:
        multistep: Configuration for multi-step processing (required; carries
            the policy-resolved horizon contract).
        video: Configuration for video recording.
    """

    multistep: MultiStepConfig
    video: VideoConfig = field(default_factory=VideoConfig)


def get_simpler_env_fn(
    env_name: str,
):
    def env_fn():
        from gr00t.eval.sim.SimplerEnv.simpler_env import register_simpler_envs

        register_simpler_envs()
        return gym.make(env_name)

    return env_fn


def get_libero_env_fn(
    env_name: str,
):
    def env_fn():
        from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs

        register_libero_envs()
        return gym.make(env_name)

    return env_fn


def get_libero_plus_env_fn(
    env_name: str,
):
    def env_fn():
        from gr00t.eval.sim.LIBERO_plus.libero_plus_env import register_libero_plus_envs

        register_libero_plus_envs()
        return gym.make(env_name)

    return env_fn


def get_robocasa_env_fn(
    env_name: str,
    robocasa_split: str = "",
):
    def env_fn():
        kwargs = {}
        if env_name.startswith("robocasa365_panda_omron/"):
            import gr00t.eval.sim.robocasa365.gymnasium_groot  # noqa: F401

            if robocasa_split:
                kwargs["split"] = robocasa_split
        else:
            import robocasa  # noqa: F401
            import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401

        return gym.make(env_name, enable_render=True, **kwargs)

    return env_fn


def get_gym_env(env_name: str, env_idx: int, total_n_envs: int, robocasa_split: str = ""):
    """Create Ray environment factory function without wrappers."""

    env_embodiment = get_embodiment_tag_from_env_name(env_name)
    env_prefix = env_name.split("/")[0]

    if env_prefix in ("robocasa_panda_omron", "robocasa365_panda_omron", "gr1_unified"):
        env_fn = get_robocasa_env_fn(env_name, robocasa_split=robocasa_split)

    elif env_embodiment in (EmbodimentTag.SIMPLER_ENV_GOOGLE, EmbodimentTag.SIMPLER_ENV_WIDOWX):
        env_fn = get_simpler_env_fn(env_name)

    # LIBERO and LIBERO-plus share EmbodimentTag.LIBERO_PANDA (same Panda robot/
    # action space), so the prefix -- not the embodiment tag -- decides which
    # register_libero*_envs() gets called; check the more specific prefix first.
    elif env_prefix == "libero_plus_sim":
        env_fn = get_libero_plus_env_fn(env_name)

    elif env_embodiment in (EmbodimentTag.LIBERO_PANDA,):
        env_fn = get_libero_env_fn(env_name)

    else:
        raise ValueError(f"Invalid environment name: {env_name}")

    return env_fn()


def create_eval_env(
    env_name: str,
    env_idx: int,
    total_n_envs: int,
    wrapper_configs: WrapperConfigs,
    robocasa_split: str = "",
    video_dir_override: Path | None = None,
) -> gym.Env:
    """Create a single evaluation environment with wrappers.

    Args:
        env_name: Name of the gymnasium environment to use
        idx: Environment index (used to determine video recording)
        wrapper_configs: Configuration for environment wrappers
        video_dir_override: When set, used instead of
            ``wrapper_configs.video.video_dir`` for this env only. Lets a
            heterogeneous batch (see :func:`run_rollout_gymnasium_policy_batch`)
            route each sub-env's videos to its own task-specific directory
            while sharing one ``wrapper_configs`` for everything else.
    Returns:
        Wrapped gymnasium environment
    """

    env = get_gym_env(env_name, env_idx, total_n_envs, robocasa_split=robocasa_split)
    video_dir = video_dir_override if video_dir_override is not None else wrapper_configs.video.video_dir
    if video_dir is not None:
        from gr00t.eval.sim.wrapper.video_recording_wrapper import VideoRecordingWrapper

        record_video_keys = wrapper_configs.video.record_video_keys
        env_prefix = env_name.split("/")[0]
        if record_video_keys is None:
            record_video_keys = ROBOCASA_RECORD_VIDEO_KEYS_BY_PREFIX.get(env_prefix)

        env = VideoRecordingWrapper(
            env,
            video_dir=Path(video_dir),
            steps_per_render=wrapper_configs.video.steps_per_render,
            max_episode_steps=wrapper_configs.video.max_episode_steps,
            fps=wrapper_configs.video.fps,
            codec=wrapper_configs.video.codec,
            overlay_text=wrapper_configs.video.overlay_text,
            record_video_keys=record_video_keys,
        )

    env = MultiStepWrapper(
        env,
        contract=wrapper_configs.multistep.contract,
        max_episode_steps=wrapper_configs.multistep.max_episode_steps,
        terminate_on_success=wrapper_configs.multistep.terminate_on_success,
    )
    return env


class _RobustAsyncVectorEnv(gym.vector.AsyncVectorEnv):
    """AsyncVectorEnv that tolerates variable-shaped info arrays across envs.

    Gymnasium's default _add_info pre-allocates a numpy array based on the
    first env's value shape and then assigns subsequent envs into it.  When
    envs return differently-shaped values (e.g. variable-length contact arrays)
    the assignment raises ValueError.  We catch that and fall back to a plain
    Python list for that key so the rest of the step can proceed normally.
    """

    def _add_info(self, infos, info, env_num):
        for k, v in info.items():
            if k not in infos:
                infos[k] = [None] * self.num_envs
                infos[f"_{k}"] = np.zeros(self.num_envs, dtype=bool)
            if isinstance(infos[k], np.ndarray):
                try:
                    infos[k][env_num] = v
                except (ValueError, TypeError):
                    lst = list(infos[k])
                    lst[env_num] = v
                    infos[k] = lst
            else:
                infos[k][env_num] = v
            infos[f"_{k}"][env_num] = True
        return infos


def _macro_step_env_steps(env_infos: dict, env_idx: int) -> int:
    """Inner env-steps advanced by one macro-step for ``env_idx``.

    ``MultiStepWrapper`` records the number of inner ``super().step()`` calls in
    ``info["n_env_steps"]``. The vector env relocates that info depending on the
    gymnasium version:

    * gymnasium 0.29.1 (sim venvs) uses inline autoreset: a terminating step's
      info is moved into ``final_info`` while the top-level info describes the
      freshly reset env, which has **no** ``n_env_steps``.
    * gymnasium >=1.0 keeps the terminating step's info at the top level.

    ``final_info`` is therefore consulted first; otherwise the terminal
    macro-step is silently counted as 0 env-steps and ``episode_length``
    collapses to 0, tripping the zero-length-episode invariant downstream.
    """
    final_info = env_infos.get("final_info")
    if final_info is not None and final_info[env_idx] is not None:
        step_info = final_info[env_idx]
        if "n_env_steps" in step_info:
            return int(step_info["n_env_steps"])
    if "n_env_steps" in env_infos:
        # The key can be present with a None entry for this env; count 0 rather
        # than calling int(None).
        n_env_steps = env_infos["n_env_steps"][env_idx]
        if n_env_steps is not None:
            return int(n_env_steps)
    return 0


def _normalize_success(env_success: Any) -> bool:
    """Coerce an env's ``success`` value (bool / int / list / ndarray of
    per-condition flags) to a single bool, matching each success type the sim
    benchmarks are known to emit. Raises on an unrecognized type so a new
    benchmark's novel success shape fails loudly here instead of silently
    miscounting successes.
    """
    if isinstance(env_success, list):
        return bool(np.any(env_success))
    elif isinstance(env_success, np.ndarray):
        return bool(np.any(env_success))
    elif isinstance(env_success, (bool, int)):
        return bool(env_success)
    else:
        raise ValueError(f"Unknown success dtype: {type(env_success)}")


def _collect_rollout_episodes(
    env,
    policy: BasePolicy,
    n_episodes: int,
    n_envs: int,
    seed: int | None,
):
    """Run the rollout loop and return ``(successes, lengths, rewards, infos)``.

    Owns the progress bar and the initial / inter-episode env resets; the
    caller owns env teardown so ``env.close()`` still runs if this raises.
    """
    episode_lengths: list[int] = []
    episode_rewards: list[float] = []
    current_rewards = [0.0] * n_envs
    current_lengths = [0] * n_envs
    completed_episodes = 0
    current_successes = [False] * n_envs
    episode_successes = []
    episode_infos = defaultdict(list)

    # Initial reset; if a seed is provided, give each sub-env a distinct but
    # deterministic seed so that parallel workers don't all start from the
    # same initial state while still being run-to-run reproducible.
    if seed is not None:
        reset_seeds = [int(seed) + i for i in range(n_envs)]
        observations, _ = env.reset(seed=reset_seeds)
    else:
        observations, _ = env.reset()
    policy.reset()

    pbar = tqdm(total=n_episodes, desc="Episodes")
    try:
        while completed_episodes < n_episodes:
            actions, _ = policy.get_action(observations)
            next_obs, rewards, terminations, truncations, env_infos = env.step(actions)
            # NOTE (FY): Currently we don't properly handle policy reset. For now, our policy are stateless,
            # but in the future if we need policy to be stateful, we need to detect env reset and call policy.reset()
            for env_idx in range(n_envs):
                if "success" in env_infos:
                    env_success = env_infos["success"][env_idx]
                    if isinstance(env_success, list):
                        env_success = np.any(env_success)
                    elif isinstance(env_success, np.ndarray):
                        env_success = np.any(env_success)
                    elif isinstance(env_success, bool):
                        env_success = env_success
                    elif isinstance(env_success, int):
                        env_success = bool(env_success)
                    else:
                        raise ValueError(f"Unknown success dtype: {type(env_success)}")
                    current_successes[env_idx] |= bool(env_success)
                else:
                    current_successes[env_idx] = False

                if "final_info" in env_infos and env_infos["final_info"][env_idx] is not None:
                    env_success = env_infos["final_info"][env_idx]["success"]
                    if isinstance(env_success, list):
                        env_success = any(env_success)
                    elif isinstance(env_success, np.ndarray):
                        env_success = np.any(env_success)
                    elif isinstance(env_success, bool):
                        env_success = env_success
                    elif isinstance(env_success, int):
                        env_success = bool(env_success)
                    else:
                        raise ValueError(f"Unknown success dtype: {type(env_success)}")
                    current_successes[env_idx] |= bool(env_success)
                current_rewards[env_idx] += rewards[env_idx]
                current_lengths[env_idx] += _macro_step_env_steps(env_infos, env_idx)

                # If episode ended, store results
                if terminations[env_idx] or truncations[env_idx]:
                    if "final_info" in env_infos:
                        current_successes[env_idx] |= any(
                            env_infos["final_info"][env_idx]["success"]
                        )
                    if "task_progress" in env_infos:
                        episode_infos["task_progress"].append(
                            env_infos["task_progress"][env_idx][-1]
                        )
                    if "q_score" in env_infos:
                        episode_infos["q_score"].append(np.max(env_infos["q_score"][env_idx]))
                    if "valid" in env_infos:
                        episode_infos["valid"].append(all(env_infos["valid"][env_idx]))
                    # Accumulate per-episode results. Both lists are captured
                    # BEFORE the per-env trackers are reset to 0 below — without
                    # this ordering downstream consumers silently see
                    # episode_length=0 / episode_reward=0.0.
                    episode_lengths.append(current_lengths[env_idx])
                    episode_rewards.append(float(current_rewards[env_idx]))
                    episode_successes.append(current_successes[env_idx])
                    # Reset trackers for this environment.
                    current_successes[env_idx] = False
                    # only update completed_episodes if valid
                    if "valid" in episode_infos:
                        if episode_infos["valid"][-1]:
                            completed_episodes += 1
                            pbar.update(1)
                    else:
                        # envs don't return valid
                        completed_episodes += 1
                        pbar.update(1)
                    # Reset with `0.0` to match the `[0.0] * n_envs` init and the
                    # `float(...)` cast above; otherwise the per-env entry's
                    # static type silently flips int <-> float across iterations.
                    current_rewards[env_idx] = 0.0
                    current_lengths[env_idx] = 0
            observations = next_obs

        # Best-effort: a failed final reset must not skip the caller's env.close().
        try:
            env.reset()
        except Exception as reset_err:
            print(f"Final env.reset() before close failed; closing env anyway: {reset_err}")
    finally:
        pbar.close()

    return episode_successes, episode_lengths, episode_rewards, episode_infos


def _collect_one_episode_per_env(
    env,
    policy: BasePolicy,
    n_envs: int,
    seed: int | None,
):
    """Like :func:`_collect_rollout_episodes`, but for a *heterogeneous* batch
    where each sub-env is a different task and exactly one episode per sub-env
    is wanted (see :func:`run_rollout_gymnasium_policy_batch`).

    Unlike the shared-task loop, completion is tracked per sub-env index
    rather than by a single ``n_episodes`` counter, and results are returned
    positionally (index i == the i-th sub-env), not in completion order.
    Sub-envs that finish early keep auto-resetting and stepping (gymnasium
    vector envs step all sub-envs together; there's no way to pause one), but
    any episode after a sub-env's first is discarded -- this wastes some
    compute on early finishers but keeps the vectorized stepping simple and
    correct: exactly one recorded result per sub-env.
    """
    episode_lengths: list[int | None] = [None] * n_envs
    episode_rewards: list[float | None] = [None] * n_envs
    episode_successes: list[bool | None] = [None] * n_envs
    current_rewards = [0.0] * n_envs
    current_lengths = [0] * n_envs
    current_successes = [False] * n_envs
    contributed = [False] * n_envs

    if seed is not None:
        reset_seeds = [int(seed) + i for i in range(n_envs)]
        observations, _ = env.reset(seed=reset_seeds)
    else:
        observations, _ = env.reset()
    policy.reset()

    pbar = tqdm(total=n_envs, desc="Tasks")
    try:
        while not all(contributed):
            actions, _ = policy.get_action(observations)
            next_obs, rewards, terminations, truncations, env_infos = env.step(actions)
            for env_idx in range(n_envs):
                if contributed[env_idx]:
                    continue

                if "success" in env_infos:
                    current_successes[env_idx] |= _normalize_success(env_infos["success"][env_idx])
                else:
                    current_successes[env_idx] = False

                if "final_info" in env_infos and env_infos["final_info"][env_idx] is not None:
                    current_successes[env_idx] |= _normalize_success(
                        env_infos["final_info"][env_idx]["success"]
                    )

                current_rewards[env_idx] += rewards[env_idx]
                current_lengths[env_idx] += _macro_step_env_steps(env_infos, env_idx)

                if terminations[env_idx] or truncations[env_idx]:
                    episode_lengths[env_idx] = current_lengths[env_idx]
                    episode_rewards[env_idx] = float(current_rewards[env_idx])
                    episode_successes[env_idx] = current_successes[env_idx]
                    contributed[env_idx] = True
                    pbar.update(1)
            observations = next_obs

        try:
            env.reset()
        except Exception as reset_err:
            print(f"Final env.reset() before close failed; closing env anyway: {reset_err}")
    finally:
        pbar.close()

    return episode_successes, episode_lengths, episode_rewards


def run_rollout_gymnasium_policy_batch(
    env_names: list[str],
    policy: BasePolicy,
    wrapper_configs: WrapperConfigs,
    video_dirs: list[str | None] | None = None,
    seed: int | None = None,
) -> Any:
    """Run exactly one episode for each of ``len(env_names)`` *different*
    tasks in a single vectorized env, so every macro-step sends ONE batched
    request to the (possibly remote) policy -- one GPU forward pass covers
    the whole batch, unlike calling :func:`run_rollout_gymnasium_policy` once
    per task in a loop (which is what the naive per-task sweep does).

    All tasks in a batch must share the same embodiment (e.g. every
    ``libero_plus_sim/...`` task shares ``EmbodimentTag.LIBERO_PANDA``) since
    they're served by the same policy instance.

    Args:
        env_names: One gym env_name per sub-env; determines batch size.
        video_dirs: Optional per-task video directory, positionally aligned
            with ``env_names`` (``None`` entries disable video for that task).
            Overrides ``wrapper_configs.video.video_dir``.
    Returns:
        ``(env_names, episode_successes, episode_infos)``, positionally
        aligned with ``env_names`` (index i is env_names[i]'s result) --
        unlike :func:`run_rollout_gymnasium_policy`, which returns results in
        completion order because multiple sub-envs there share one task.
    """
    n_envs = len(env_names)
    if video_dirs is not None and len(video_dirs) != n_envs:
        raise ValueError(f"video_dirs has {len(video_dirs)} entries, expected {n_envs}")

    start_time = time.time()
    print(f"Running 1 episode each for {n_envs} tasks in one batch")

    env_fns = [
        partial(
            create_eval_env,
            env_idx=idx,
            env_name=env_names[idx],
            total_n_envs=n_envs,
            wrapper_configs=wrapper_configs,
            video_dir_override=(video_dirs[idx] if video_dirs is not None else None),
        )
        for idx in range(n_envs)
    ]

    if n_envs == 1:
        env = gym.vector.SyncVectorEnv(env_fns)
    else:
        env = _RobustAsyncVectorEnv(env_fns, shared_memory=False, context="spawn")

    try:
        episode_successes, episode_lengths, episode_rewards = _collect_one_episode_per_env(
            env, policy, n_envs, seed
        )
    finally:
        try:
            env.close()
        except Exception as close_err:
            print(f"env.close() during teardown failed: {close_err}")

    print(f"Batch of {n_envs} tasks took {time.time() - start_time:.1f}s")

    episode_infos = {
        "episode_lengths": episode_lengths,
        "episode_rewards": episode_rewards,
    }
    return env_names, episode_successes, episode_infos


def run_rollout_gymnasium_policy(
    env_name: str,
    policy: BasePolicy,
    wrapper_configs: WrapperConfigs,
    n_episodes: int = 10,
    n_envs: int = 1,
    seed: int | None = None,
    robocasa_split: str = "",
) -> Any:
    """Run policy rollouts in parallel environments.

    Args:
        env_name: Name of the gymnasium environment to use
        policy: Policy instance
        n_episodes: Number of episodes to run
        n_envs: Number of parallel environments
        wrapper_configs: Configuration for environment wrappers
        seed: If set, forwards per-env seeds (``seed+i``) to the first
            ``env.reset`` so each sub-env is reproducible. Should be paired
            with :func:`gr00t.utils.determinism.seed_everything` upstream to
            also constrain policy-side RNGs.
    Returns:
        ``(env_name, episode_successes, episode_infos)``. ``episode_lengths``
        in ``episode_infos`` is in **env-steps** (inner ``super().step()``
        calls), the same unit as ``MultiStepWrapper.max_episode_steps``.
    """
    start_time = time.time()
    n_episodes = max(n_episodes, n_envs)
    print(f"Running collecting {n_episodes} episodes for {env_name} with {n_envs} vec envs")

    env_fns = [
        partial(
            create_eval_env,
            env_idx=idx,
            env_name=env_name,
            total_n_envs=n_envs,
            wrapper_configs=wrapper_configs,
            robocasa_split=robocasa_split,
        )
        for idx in range(n_envs)
    ]

    if n_envs == 1:
        env = gym.vector.SyncVectorEnv(env_fns)
    else:
        env = _RobustAsyncVectorEnv(
            env_fns,
            shared_memory=False,
            context="spawn",
        )

    # Reap the vector env (and the ffmpeg / async-worker children it owns) on
    # every exit path; a leaked child blocks the next eval shard's ports/GPUs.
    try:
        episode_successes, episode_lengths, episode_rewards, episode_infos = (
            _collect_rollout_episodes(env, policy, n_episodes, n_envs, seed)
        )
    finally:
        # Don't let a teardown error mask an in-flight rollout exception.
        try:
            env.close()
        except Exception as close_err:
            print(f"env.close() during teardown failed: {close_err}")
    print(f"Collecting {n_episodes} episodes took {time.time() - start_time} seconds")

    assert len(episode_successes) >= n_episodes, (
        f"Expected at least {n_episodes} episodes, got {len(episode_successes)}"
    )

    # Every captured episode ran >= 1 env-step, so episode_length >= 1.
    assert all(length >= 1 for length in episode_lengths), (
        f"Internal invariant violated: rollout produced zero-length episode(s) "
        f"in {episode_lengths!r}."
    )

    # Surface the per-episode length and reward that were tracked locally so
    # downstream metrics (SimplerEnv / LIBERO / Robocasa / Wholebody) can
    # read them off episode_infos instead of silently falling back to 0.
    # Planted BEFORE the "valid" filter so they get filtered in lockstep
    # with the other episode_infos fields.
    episode_infos["episode_lengths"] = episode_lengths
    episode_infos["episode_rewards"] = episode_rewards

    episode_infos = dict(episode_infos)  # Convert defaultdict to dict
    for key, value in episode_infos.items():
        assert len(value) == len(episode_successes), (
            f"Length of {key} is not equal to the number of episodes"
        )

    # process valid results
    if "valid" in episode_infos:
        valids = episode_infos["valid"]
        valid_idxs = np.where(valids)[0]
        episode_successes = [episode_successes[i] for i in valid_idxs]
        episode_infos = {k: [v[i] for i in valid_idxs] for k, v in episode_infos.items()}

    return env_name, episode_successes, episode_infos


def create_gr00t_sim_policy(
    model_path: str,
    embodiment_tag: EmbodimentTag,
    policy_client_host: str = "",
    policy_client_port: int | None = None,
    trt_engine_path: str = "",
    trt_mode: InferenceMode = InferenceMode.n17_full_pipeline,
) -> BasePolicy:
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper

    if policy_client_host and policy_client_port:
        from gr00t.policy.server_client import PolicyClient

        policy = PolicyClient(host=policy_client_host, port=policy_client_port)
    else:
        gr00t_policy = Gr00tPolicy(
            embodiment_tag=embodiment_tag,
            model_path=model_path,
            device=0,
        )
        if trt_engine_path:
            deploy_dir = str(Path(__file__).resolve().parents[2] / "scripts" / "deployment")
            if deploy_dir not in sys.path:
                sys.path.insert(0, deploy_dir)
            from trt_model_forward import close_tensorrt_engines, setup_tensorrt_engines

            try:
                setup_tensorrt_engines(gr00t_policy, trt_engine_path, mode=trt_mode)
            except BaseException:
                # Free engines attached before the failing one so a partial
                # setup doesn't leak GPU memory.
                close_tensorrt_engines(gr00t_policy)
                raise
        policy = Gr00tSimPolicyWrapper(gr00t_policy)
    return policy


def run_gr00t_sim_policy(
    env_name: str,
    n_episodes: int,
    max_episode_steps: int,
    model_path: str = "",
    policy_client_host: str = "",
    policy_client_port: int | None = None,
    n_envs: int = 8,
    n_action_steps: int = 8,
    video_dir: str | None = None,
    trt_engine_path: str = "",
    trt_mode: InferenceMode = InferenceMode.n17_full_pipeline,
    seed: int | None = None,
    robocasa_split: str = "",
):
    # seed_everything resolves `None` via the GR00T_EVAL_SEED env var and is a
    # no-op when that is also unset, so the historical non-deterministic
    # behavior is preserved by default. Returns the effective seed (or None)
    # which we forward to env.reset below.
    seed = seed_everything(seed)

    embodiment_tag = get_embodiment_tag_from_env_name(env_name)

    if video_dir is None:
        if model_path:
            parts = model_path.split("/")
            model_slug = parts[-3] if len(parts) >= 3 else parts[-1]
            video_dir = f"/tmp/sim_eval_videos_{model_slug}_ac{n_action_steps}_{uuid.uuid4()}"
        else:
            video_dir = f"/tmp/sim_eval_videos_{env_name}_ac{n_action_steps}_{uuid.uuid4()}"
    policy = create_gr00t_sim_policy(
        model_path,
        embodiment_tag,
        policy_client_host,
        policy_client_port,
        trt_engine_path=trt_engine_path,
        trt_mode=trt_mode,
    )

    # Release TRT engines explicitly on every exit path: the sim-eval entrypoint
    # hard-exits via os._exit, which skips Engine.__del__ and would leak GPU memory.
    if trt_engine_path:
        deploy_dir = str(Path(__file__).resolve().parents[2] / "scripts" / "deployment")
        if deploy_dir not in sys.path:
            sys.path.insert(0, deploy_dir)
        from trt_model_forward import closing_tensorrt_engines

        engine_cleanup = closing_tensorrt_engines(policy)
    else:
        engine_cleanup = nullcontext()

    with engine_cleanup:
        # Resolve the horizon contract from the policy *before* building the
        # wrapper config. The video / state delta-indices are sourced from the
        # policy's modality config (no policy-independent defaults), and
        # ``n_action_steps`` is the receding-horizon execution length validated
        # against the policy's action horizon. A mismatch now raises here at
        # construction instead of surfacing as an IndexError / cryptic
        # check_observation assert deep inside the rollout loop.
        contract = PolicyHorizonSpec.from_policy(policy, n_action_steps=n_action_steps)

        wrapper_configs = WrapperConfigs(
            multistep=MultiStepConfig(
                contract=contract,
                max_episode_steps=max_episode_steps,
                terminate_on_success=True,
            ),
            video=VideoConfig(
                video_dir=video_dir,
                max_episode_steps=max_episode_steps,
            ),
        )

        results = run_rollout_gymnasium_policy(
            env_name=env_name,
            policy=policy,
            wrapper_configs=wrapper_configs,
            n_episodes=n_episodes,
            n_envs=n_envs,
            seed=seed,
            robocasa_split=robocasa_split,
        )
        print("Video saved to: ", wrapper_configs.video.video_dir)
        return results


def run_gr00t_sim_policy_batch(
    env_names: list[str],
    max_episode_steps: int,
    model_path: str = "",
    policy_client_host: str = "",
    policy_client_port: int | None = None,
    n_action_steps: int = 8,
    video_dirs: list[str | None] | None = None,
    seed: int | None = None,
):
    """Batched counterpart of :func:`run_gr00t_sim_policy`: one episode each
    for every task in ``env_names``, sent to the policy as one vectorized
    batch per macro-step instead of one ``run_gr00t_sim_policy`` process per
    task. All tasks must share one embodiment (checked below), since they're
    served by the same policy instance -- true for a LIBERO-plus sweep, where
    every task uses ``EmbodimentTag.LIBERO_PANDA``.
    """
    seed = seed_everything(seed)

    embodiment_tags = {get_embodiment_tag_from_env_name(name) for name in env_names}
    if len(embodiment_tags) != 1:
        raise ValueError(
            f"run_gr00t_sim_policy_batch requires all tasks to share one embodiment "
            f"(one policy instance serves the whole batch); got {embodiment_tags}"
        )
    embodiment_tag = embodiment_tags.pop()

    policy = create_gr00t_sim_policy(
        model_path,
        embodiment_tag,
        policy_client_host,
        policy_client_port,
    )

    contract = PolicyHorizonSpec.from_policy(policy, n_action_steps=n_action_steps)
    wrapper_configs = WrapperConfigs(
        multistep=MultiStepConfig(
            contract=contract,
            max_episode_steps=max_episode_steps,
            terminate_on_success=True,
        ),
        video=VideoConfig(max_episode_steps=max_episode_steps),
    )

    return run_rollout_gymnasium_policy_batch(
        env_names=env_names,
        policy=policy,
        wrapper_configs=wrapper_configs,
        video_dirs=video_dirs,
        seed=seed,
    )


@dataclass
class RolloutConfig:
    """Configuration for rollout policy evaluation."""

    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS
    """Maximum number of steps per episode."""

    n_episodes: int = 50
    """Number of episodes to run."""

    model_path: str = ""
    """Path to model checkpoint."""

    policy_client_host: str = ""
    """Host for policy client."""

    policy_client_port: int | None = None
    """Port for policy client."""

    env_name: str = "libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"
    """Environment name."""

    n_envs: int = 8
    """Number of parallel environments."""

    n_action_steps: int = 8
    """Number of action steps."""

    video_dir: str | None = None
    """Directory to save videos. If None, uses /tmp/sim_eval_videos_<env>_<uuid>."""

    trt_engine_path: str = ""
    """Path to TRT engine directory. If set, uses TRT inference instead of PyTorch."""

    trt_mode: InferenceMode = InferenceMode.n17_full_pipeline
    """TRT mode: which engine subset to swap in (e.g. 'n17_full_pipeline' for all
    engines, 'vit_llm_only', 'action_head', or 'dit_only')."""

    seed: int | None = None
    """Optional seed for deterministic evaluation. When set, seeds Python /
    NumPy / torch / cuda RNGs, enables cuDNN determinism, and forwards
    per-env seeds to the sim envs. If left as ``None``, falls back to the
    ``GR00T_EVAL_SEED`` env var; if that is also unset, keeps the historical
    non-deterministic behavior."""

    robocasa_split: str = ""
    """Optional RoboCasa/RoboCasa365 split forwarded to the simulator."""


if __name__ == "__main__":
    args = tyro.cli(RolloutConfig)

    # validate policy configuration
    assert (args.model_path and not (args.policy_client_host or args.policy_client_port)) or (
        not args.model_path and args.policy_client_host and args.policy_client_port is not None
    ), (
        "Invalid policy configuration: You must provide EITHER model_path OR (policy_client_host & policy_client_port), not both.\n"
        "If all 3 arguments are provided, explicitly choose one:\n"
        '  - To use policy client: set --policy-client-host and --policy-client-port, and set --model-path ""\n'
        '  - To use model path: set --model-path, and set --policy-client-host "" (and leave --policy-client-port unset)'
    )

    results = run_gr00t_sim_policy(
        env_name=args.env_name,
        n_episodes=args.n_episodes,
        max_episode_steps=args.max_episode_steps,
        model_path=args.model_path,
        policy_client_host=args.policy_client_host,
        policy_client_port=args.policy_client_port,
        n_envs=args.n_envs,
        n_action_steps=args.n_action_steps,
        video_dir=args.video_dir,
        trt_engine_path=args.trt_engine_path,
        trt_mode=args.trt_mode,
        seed=args.seed,
        robocasa_split=args.robocasa_split,
    )
    print("results: ", results)
    print("success rate: ", np.mean(results[1]))
