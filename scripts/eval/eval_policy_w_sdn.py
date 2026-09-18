# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import warnings
from dataclasses import dataclass, field
from typing import List, Literal

import numpy as np
import tyro
import types

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.policy.server_client import PolicyServer, PolicyClient

from gr00t.utils.eval import calc_mse_for_single_trajectory

warnings.simplefilter("ignore", category=FutureWarning)

"""
Example command:

NOTE: provide --model_path to load up the model checkpoint in this script,
        else it will use the default host and port via RobotInferenceClient

python scripts/eval_policy.py --plot --model-path nvidia/GR00T-N1.5-3B
"""


@dataclass
class ArgsConfig:
    """Configuration for evaluating a policy."""

    host: str = "localhost"
    """Host to connect to."""

    port: int = 5556
    """Port to connect to."""

    plot: bool = False
    """Whether to plot the images."""

    modality_keys: List[str] = field(default_factory=lambda: ["single_arm","gripper"])#["right_arm", "left_arm"])
    # modality_keys: List[str] = field(default_factory=lambda: ["left_arm","right_arm", 'left_hand', 'right_hand'])#["right_arm", "left_arm"])

    """Modality keys to evaluate."""

    data_config: str = "fourier_gr1_arms_only"
    """
    Data config to use, e.g. so100, fourier_gr1_arms_only, unitree_g1, etc.
    Or a path to a custom data config file. e.g. "module:ClassName" format.
    See gr00t/experiment/data_config.py for more details.
    """

    steps: int = 50
    """Number of steps to evaluate."""

    trajs: int = 1
    """Number of trajectories to evaluate."""

    start_traj: int = 0
    """Start trajectory to evaluate."""

    action_horizon: int = None
    """Action horizon to evaluate. If None, will use the data config's action horizon."""

    video_backend: Literal["decord", "torchvision_av", "torchcodec"] = "torchvision_av"
    """Video backend to use for various codec options. h264: decord or av: torchvision_av"""

    dataset_path: str = "demo_data/robot_sim.PickNPlace/"
    """Path to the dataset."""

    save_plot_path: str = None
    """Path to save the plot."""

    plot_state: bool = False
    """Whether to plot the state."""


def main(args: ArgsConfig):
    policy = PolicyClient(host=args.host, port=args.port)

    # Get the supported modalities for the policy
    modality = policy.get_modality_config()
    print("Current modality config: \n", modality)

    # Create the dataset
    dataset = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=modality,
    )

    print(len(dataset))
    # Make a prediction
    # obs = dataset[0]
    # for k, v in obs.items():
    #     if isinstance(v, np.ndarray):
    #         print(k, v.shape)
    #     else:
    #         print(k, v)

    # for k, v in dataset.get_step_data(0, 0).items():
    #     if isinstance(v, np.ndarray):
    #         print(k, v.shape)
    #     else:
    #         print(k, v)

    print("Total trajectories:", len(dataset))

    all_mse = []
    for traj_id in range(args.start_traj, args.start_traj + args.trajs):
        print("Running trajectory:", traj_id)
        mse = calc_mse_for_single_trajectory(
            policy,
            dataset,
            traj_id,
            modality_keys=args.modality_keys,
            steps=args.steps,
            action_horizon=args.action_horizon,
            plot=args.plot,
            plot_state=args.plot_state,
            save_plot_path=args.save_plot_path + f"{traj_id}",
            modality_config=modality
        )
        print("MSE:", mse)
        all_mse.append(mse)
    print("Average MSE across all trajs:", np.mean(all_mse))
    print("Done")
    exit()


if __name__ == "__main__":
    # Parse arguments using tyro
    config = tyro.cli(ArgsConfig)
    main(config)