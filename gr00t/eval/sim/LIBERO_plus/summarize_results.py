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
Summarize LIBERO-plus sweep results into a suite x perturbation table.

Reads the summary.json each gr00t/eval/sim/LIBERO_plus/rollout_batch.py run
writes to its --log-root (one per suite/perturbation sweep run by
scripts/libero_plus.sh or scripts/libero_plus_all_pertubation.sh), so this
needs no LIBERO/robosuite import and no venv -- plain python3 is enough.

Usage:
    python3 gr00t/eval/sim/LIBERO_plus/summarize_results.py
    python3 gr00t/eval/sim/LIBERO_plus/summarize_results.py --eval-root eval_logs/libero_plus
    # Concatenate every run's per-task results.csv into one file for pandas/Excel:
    python3 gr00t/eval/sim/LIBERO_plus/summarize_results.py --combine-csv /tmp/all_results.csv
"""

import argparse
import csv
import json
from pathlib import Path


def find_summaries(eval_root: Path) -> list[dict]:
    summaries = []
    for path in sorted(eval_root.glob("**/summary.json")):
        data = json.loads(path.read_text())
        data["_path"] = str(path)
        summaries.append(data)
    return summaries


def print_table(summaries: list[dict]) -> None:
    if not summaries:
        print("No summary.json files found.")
        return

    suites = sorted({s["suite"] for s in summaries})
    pertubes = sorted({s["pertube"] for s in summaries})
    cell = {(s["suite"], s["pertube"]): s for s in summaries}

    col_width = max(14, max((len(p) for p in pertubes), default=0) + 2)
    suite_width = max(12, max((len(s) for s in suites), default=0) + 2)

    header = "suite".ljust(suite_width) + "".join(p.ljust(col_width) for p in pertubes)
    print(header)
    print("-" * len(header))
    for suite in suites:
        row = suite.ljust(suite_width)
        for pertube in pertubes:
            s = cell.get((suite, pertube))
            if s is None:
                row += "-".ljust(col_width)
            else:
                text = f"{s['n_success']}/{s['n_tasks']} ({100 * s['success_rate']:.1f}%)"
                row += text.ljust(col_width)
        print(row)

    total_tasks = sum(s["n_tasks"] for s in summaries)
    total_success = sum(s["n_success"] for s in summaries)
    overall = 100 * total_success / total_tasks if total_tasks else 0.0
    print("-" * len(header))
    print(f"Overall: {total_success}/{total_tasks} ({overall:.1f}%) across {len(summaries)} sweep(s)")


def combine_csv(eval_root: Path, out_path: Path) -> None:
    rows = []
    fieldnames = None
    for path in sorted(eval_root.glob("**/results.csv")):
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = fieldnames or reader.fieldnames
            rows.extend(reader)
    if not rows:
        print(f"No results.csv files found under {eval_root}")
        return
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows from results.csv files to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-root",
        default="eval_logs/libero_plus",
        help="Root directory to scan for summary.json / results.csv files (default: %(default)s)",
    )
    parser.add_argument(
        "--combine-csv",
        default=None,
        help="If set, also concatenate every run's results.csv into this one file.",
    )
    args = parser.parse_args()

    eval_root = Path(args.eval_root)
    summaries = find_summaries(eval_root)
    print_table(summaries)

    if args.combine_csv:
        combine_csv(eval_root, Path(args.combine_csv))


if __name__ == "__main__":
    main()
