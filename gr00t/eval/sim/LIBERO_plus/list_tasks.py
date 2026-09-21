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
List LIBERO-plus ``env_name`` values (for gr00t.eval.rollout_policy
--env-name) filtered by perturbation category and/or task suite.

LIBERO-plus perturbs each base task along 7 independent dimensions
(Background Textures, Camera Viewpoints, Language Instructions, Light
Conditions, Objects Layout, Robot Initial States, Sensor Noise); each suite
has ~2,400-2,600 task variants, so this reads
external_dependencies/LIBERO-plus/libero/libero/benchmark/task_classification.json
directly (pure stdlib, no libero import needed) rather than requiring the
LIBERO-plus venv to enumerate them by hand.

Usage:
    python gr00t/eval/sim/LIBERO_plus/list_tasks.py --category "Sensor Noise" --suite libero_10
    python gr00t/eval/sim/LIBERO_plus/list_tasks.py --list-categories
"""

import json
from pathlib import Path

import tyro


CATEGORIES = (
    "Background Textures",
    "Camera Viewpoints",
    "Language Instructions",
    "Light Conditions",
    "Objects Layout",
    "Robot Initial States",
    "Sensor Noise",
)

SUITES = ("libero_10", "libero_spatial", "libero_object", "libero_goal")

TASK_CLASSIFICATION_PATH = (
    Path(__file__).resolve().parents[4]
    / "external_dependencies"
    / "LIBERO-plus"
    / "libero"
    / "libero"
    / "benchmark"
    / "task_classification.json"
)


def main(
    category: str | None = None,
    suite: str | None = None,
    list_categories: bool = False,
) -> None:
    """
    Args:
        category: One of CATEGORIES. If unset, tasks from all categories are listed.
        suite: One of SUITES. If unset, tasks from all suites are listed.
        list_categories: If set, print the available categories/suites and exit.
    """
    if list_categories:
        print("Categories:", *CATEGORIES, sep="\n  ")
        print("Suites:", *SUITES, sep="\n  ")
        return

    if category is not None and category not in CATEGORIES:
        raise ValueError(f"Unknown category {category!r}. Valid: {CATEGORIES}")
    if suite is not None and suite not in SUITES:
        raise ValueError(f"Unknown suite {suite!r}. Valid: {SUITES}")
    if not TASK_CLASSIFICATION_PATH.exists():
        raise FileNotFoundError(
            f"{TASK_CLASSIFICATION_PATH} not found. Run "
            "gr00t/eval/sim/LIBERO_plus/setup_libero_plus.sh first to clone LIBERO-plus."
        )

    task_classification = json.loads(TASK_CLASSIFICATION_PATH.read_text())
    suites = [suite] if suite is not None else list(task_classification)
    for suite_name in suites:
        for task in task_classification[suite_name]:
            if category is not None and task["category"] != category:
                continue
            print(f"libero_plus_sim/{task['name']}")


if __name__ == "__main__":
    tyro.cli(main)
