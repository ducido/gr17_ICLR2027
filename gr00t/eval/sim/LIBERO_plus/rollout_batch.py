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

"""
Batched LIBERO-plus sweep: run one episode for each task in a list, chunked
into vectorized batches so every macro-step sends ONE request (covering a
whole batch of *different* tasks) to the policy server instead of one
gr00t/eval/rollout_policy.py process per task.

gr00t.eval.rollout_policy.PolicyServer is a single-threaded, blocking ZeroMQ
REP loop (one request in flight at a time) -- running several client
processes against one server would only serialize behind each other, not
parallelize. The actual lever is batch size: gr00t.eval.rollout_policy's
run_gr00t_sim_policy_batch vectorizes across tasks so one server-side forward
pass covers `--batch-size` tasks, which is the same mechanism
`--n-envs` uses to batch multiple episodes of ONE task -- generalized here to
batch multiple DIFFERENT tasks, which is what a LIBERO-plus sweep needs
(each of its ~2,500 task variants per suite wants exactly one episode, not
many episodes of the same task).

Usage (mirrors scripts/libero_plus.sh, which drives this):
    gr00t/eval/sim/LIBERO_plus/list_tasks.py --category "Sensor Noise" --suite libero_10 \
        > /tmp/tasks.txt
    gr00t/eval/sim/LIBERO_plus/libero_plus_uv/.venv/bin/python \
        gr00t/eval/sim/LIBERO_plus/rollout_batch.py \
        --tasks-file /tmp/tasks.txt \
        --log-root eval_logs/libero_plus/noise/libero_10/baseline \
        --policy-client-host 127.0.0.1 --policy-client-port 5555 \
        --batch-size 8
"""

import csv
from dataclasses import dataclass
import json
from pathlib import Path

from gr00t.eval.rollout_policy import run_gr00t_sim_policy_batch
import numpy as np
import tyro


@dataclass
class BatchRolloutConfig:
    tasks_file: str
    """Newline-separated list of env_names (e.g. from list_tasks.py)."""

    log_root: str
    """Base directory. Each task gets <log_root>/<task_basename>/{<task_basename>.txt,videos/}.
    Also where the run-level results.csv and summary.json land -- see module docstring."""

    policy_client_host: str = ""
    """Host for policy client."""

    policy_client_port: int | None = None
    """Port for policy client."""

    model_path: str = ""
    """Path to model checkpoint (alternative to policy_client_host/port)."""

    batch_size: int = 8
    """Tasks per vectorized batch (one server request per macro-step covers this many)."""

    max_episode_steps: int = 720
    """Maximum number of steps per episode."""

    n_action_steps: int = 8
    """Number of action steps (receding-horizon execution length)."""

    seed: int | None = None
    """Optional seed; see gr00t.eval.rollout_policy.RolloutConfig.seed."""

    record_video: bool = True
    """If False, skip video recording entirely (faster, no ffmpeg overhead)."""

    suite: str = ""
    """LIBERO-plus suite (e.g. libero_10), stamped into results.csv/summary.json for later
    aggregation across runs -- purely a label, doesn't affect which tasks run."""

    pertube: str = ""
    """Perturbation short-name (e.g. noise), stamped into results.csv/summary.json -- see suite."""

    category: str = ""
    """Perturbation category (e.g. "Sensor Noise"), stamped into results.csv/summary.json."""


_CSV_FIELDS = ["suite", "pertube", "category", "task", "success", "episode_length", "episode_reward"]


def _write_task_log(log_dir: Path, task_basename: str, task: str, success, length, reward) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{task_basename}.txt").write_text(
        f"task: {task}\nsuccess: {success}\nepisode_length: {length}\nepisode_reward: {reward}\n"
    )


def main(config: BatchRolloutConfig) -> None:
    assert (
        config.model_path and not (config.policy_client_host or config.policy_client_port)
    ) or (
        not config.model_path
        and config.policy_client_host
        and config.policy_client_port is not None
    ), (
        "Invalid policy configuration: provide EITHER model_path OR "
        "(policy_client_host & policy_client_port), not both."
    )

    tasks = [
        line.strip() for line in Path(config.tasks_file).read_text().splitlines() if line.strip()
    ]
    if not tasks:
        raise ValueError(f"No tasks found in {config.tasks_file}")

    log_root = Path(config.log_root)
    log_root.mkdir(parents=True, exist_ok=True)

    # Fresh file per run (not append): re-running this exact sweep should
    # replace stale rows, not accumulate duplicates alongside them. Opened
    # once up front and flushed after every batch, so a crash mid-sweep still
    # leaves results.csv usable for whatever completed so far.
    csv_path = log_root / "results.csv"
    csv_file = csv_path.open("w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=_CSV_FIELDS)
    csv_writer.writeheader()

    all_successes: list[bool] = []
    n_batches = (len(tasks) + config.batch_size - 1) // config.batch_size
    try:
        for batch_idx in range(0, len(tasks), config.batch_size):
            batch = tasks[batch_idx : batch_idx + config.batch_size]
            print(
                f"=== Batch {batch_idx // config.batch_size + 1}/{n_batches} "
                f"({len(batch)} tasks) ==="
            )
            basenames = [task.split("/")[-1] for task in batch]
            video_dirs = (
                [str(log_root / name / "videos") for name in basenames]
                if config.record_video
                else None
            )

            env_names, successes, infos = run_gr00t_sim_policy_batch(
                env_names=batch,
                max_episode_steps=config.max_episode_steps,
                model_path=config.model_path,
                policy_client_host=config.policy_client_host,
                policy_client_port=config.policy_client_port,
                n_action_steps=config.n_action_steps,
                video_dirs=video_dirs,
                seed=config.seed,
            )

            for i, (task, name) in enumerate(zip(env_names, basenames)):
                success = successes[i]
                length = infos["episode_lengths"][i]
                reward = infos["episode_rewards"][i]
                _write_task_log(log_root / name, name, task, success, length, reward)
                csv_writer.writerow(
                    {
                        "suite": config.suite,
                        "pertube": config.pertube,
                        "category": config.category,
                        "task": task,
                        "success": success,
                        "episode_length": length,
                        "episode_reward": reward,
                    }
                )
                print(f"  {name}: success={success}")
            csv_file.flush()
            all_successes.extend(successes)
    finally:
        csv_file.close()

    success_rate = float(np.mean(all_successes)) if all_successes else 0.0
    print(f"\nOverall: {sum(all_successes)}/{len(all_successes)} ({100 * success_rate:.1f}%) succeeded")

    (log_root / "summary.json").write_text(
        json.dumps(
            {
                "suite": config.suite,
                "pertube": config.pertube,
                "category": config.category,
                "n_tasks": len(all_successes),
                "n_success": int(sum(all_successes)),
                "success_rate": success_rate,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main(tyro.cli(BatchRolloutConfig))
