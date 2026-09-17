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

"""
GR00T Inference Service

This script provides both ZMQ and HTTP server/client implementations for deploying GR00T models.
The HTTP server exposes a REST API for easy integration with web applications and other services.

1. Default is zmq server.

Run server: python scripts/inference_service.py --server
Run client: python scripts/inference_service.py --client

2. Run as Http Server:

Dependencies for `http_server` mode:
    => Server (runs GR00T model on GPU): `pip install uvicorn fastapi json-numpy`
    => Client: `pip install requests json-numpy`

HTTP Server Usage:
    python scripts/inference_service.py --server --http-server --port 8000

HTTP Client Usage (assuming a server running on 0.0.0.0:8000):
    python scripts/inference_service.py --client --http-server --host 0.0.0.0 --port 8000

You can use bore to forward the port to your client: `159.223.171.199` is bore.pub.
    bore local 8000 --to 159.223.171.199

3. TensorRT Support:

For accelerated inference using TensorRT, first build the TensorRT engines using the deployment scripts,
then run the server with the --use-tensorrt flag:

TensorRT Server Usage:
    python scripts/inference_service.py --server --use-tensorrt --trt-engine-path gr00t_engine

TensorRT HTTP Server Usage:
    python scripts/inference_service.py --server --http-server --use-tensorrt --trt-engine-path gr00t_engine --port 8000

Note: TensorRT engines must be built before running with --use-tensorrt flag.
See deployment_scripts/README.md for instructions on building TensorRT engines.
"""

import time
import types
from dataclasses import dataclass
from typing import Literal

import numpy as np
import tyro
import torch

# from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING
# from gr00t.eval.robot import RobotInferenceClient, RobotInferenceServer
# from gr00t.experiment.data_config import load_data_config
from gr00t.policy.server_client import PolicyServer, PolicyClient
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
# # from gr00t.wrapper.acg.gr00t_acg import Gr00tPolicy_ACG, GR00T_N1_ACG, FlowmatchingActionHead_ACG
# # from gr00t.wrapper.cfg.gr00t_cfg import Gr00tPolicy_CFG, GR00T_N1_CFG, FlowmatchingActionHead_CFG
# from gr00t.wrapper.motion.gr00t_motion import Gr00tPolicy_Motion, GR00T_N1_Motion
# from gr00t.wrapper.knn.gr00t_knn import Gr00tPolicy_KNN, GR00T_N1_KNN
from gr00t.wrapper.sdn.gr00t_sdn import Gr00tPolicy_SDN, Gr00tN1d7_SDN


@dataclass
class ArgsConfig:
    """Command line arguments for the inference service."""

    model_path: str = "nvidia/GR00T-N1.5-3B"
    """Path to the model checkpoint directory."""

    embodiment_tag: str = "gr1"
    """The embodiment tag for the model."""

    data_config: str = "vrh3_sim"
    """
    The name of the data config to use, e.g. so100, fourier_gr1_arms_only, unitree_g1, etc.

    Or a path to a custom data config file. e.g. "module:ClassName" format.
    See gr00t/experiment/data_config.py for more details.
    """

    port: int = 5556
    """The port number for the server."""

    host: str = "localhost"
    """The host address for the server."""

    server: bool = False
    """Whether to run the server."""

    client: bool = False
    """Whether to run the client."""

    guidance_option: Literal["acg", "cfg", ""] = ""
    """Guidance option for the action generation. Options are 'acg' for action coherent guidance, 'cfg' for classifier-free guidance or empty string for no guidance"""

    bestofN_option: Literal["motion", "knn", "sdn", ""] = ""

    smooth_option: Literal["te", "rtc", "training-time-rtc", ""] = ""
    """Smooth option for the action generation. Options are 'te' for temporal encoding, 'rtc' for real-time chunking, 'training-time-rtc' for training-time rtc or empty string for no smoothing."""

    denoising_steps: int = 4
    """The number of denoising steps to use."""

    api_token: str = None
    """API token for authentication. If not provided, authentication is disabled."""

    http_server: bool = False
    """Whether to run it as HTTP server. Default is ZMQ server."""

    use_tensorrt: bool = False
    """Whether to use TensorRT for inference. Requires TensorRT engines to be built."""

    trt_engine_path: str = "gr00t_engine"
    """Path to the TensorRT engine directory. Only used when use_tensorrt is True."""

    vit_dtype: Literal["fp16", "fp8"] = "fp8"
    """ViT model dtype (fp16, fp8). Only used when use_tensorrt is True."""

    llm_dtype: Literal["fp16", "nvfp4", "fp8"] = "nvfp4"
    """LLM model dtype (fp16, nvfp4, fp8). Only used when use_tensorrt is True."""

    dit_dtype: Literal["fp16", "fp8"] = "fp8"
    """DiT model dtype (fp16, fp8). Only used when use_tensorrt is True."""

    action_horizon: int | None = None


#####################################################################################


def build_contrast_image_generator():
    import sys
    sys.path.append('/projects/extern/kisski/kisski-spath/dir.project/VLA_Imit/gr17_ICLR2027')
    from contrast_utils import get_contrast_image_generator

    temp_config = {
        "camera_name": None,
        "by": "grounded_sam_tracking",
        "inpaint_mode": "lama",
        "color": "auto",
        "sigma": 5,
        "version": 2,
        "get_all_parts": False,
    }
    contrast_image_generator = get_contrast_image_generator(temp_config)
    contrast_image_generator.reset()
    return contrast_image_generator

def _example_zmq_client_call(obs: dict, host: str, port: int, api_token: str):
    """
    Example ZMQ client call to the server.
    """
    # Original ZMQ client mode
    # Create a policy wrapper
    policy_client = PolicyClient(host=host, port=port)

    # print("Available modality config available:")
    # modality_configs = policy_client.get_modality_config()
    # print(modality_configs.keys())

    time_start = time.time()
    action, _ = policy_client.get_action(obs)
    print(f"Total time taken to get action from server: {time.time() - time_start} seconds")
    return action


def _example_http_client_call(obs: dict, host: str, port: int, api_token: str):
    """
    Example HTTP client call to the server.
    """
    import json_numpy

    json_numpy.patch()
    import requests

    # Send request to HTTP server
    print("Testing HTTP server...")

    time_start = time.time()
    response = requests.post(f"http://{host}:{port}/act", json={"observation": obs})
    print(f"Total time taken to get action from HTTP server: {time.time() - time_start} seconds")

    if response.status_code == 200:
        action = response.json()
        return action
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return {}


def main(args: ArgsConfig):
    if args.server:
        # Create a policy
        # The `Gr00tPolicy` class is being used to create a policy object that encapsulates
        # the model path, transform name, embodiment tag, and denoising steps for the robot
        # inference system. This policy object is then utilized in the server mode to start
        # the Robot Inference Server for making predictions based on the specified model and
        # configuration.

        # we will use an existing data config to create the modality config and transform
        # if a new data config is specified, this expect user to
        # construct your own modality config and transform
        # see gr00t/utils/data.py for more details
        # data_config = load_data_config(args.data_config)
        # modality_config = data_config.modality_config()
        # modality_transform = data_config.transform()

        # policy = Gr00tPolicy(
        #     model_path=args.model_path,
        #     modality_config=modality_config,
        #     modality_transform=modality_transform,
        #     embodiment_tag=args.embodiment_tag,
        #     denoising_steps=args.denoising_steps,
        #     smooth_option=args.smooth_option
        # )

        policy = Gr00tPolicy(
            model_path=args.model_path,
            embodiment_tag=args.embodiment_tag,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            strict=True,
        )

        if args.guidance_option == "acg":
            policy.get_action = types.MethodType(Gr00tPolicy_ACG.get_action, policy)
            policy._get_action_from_normalized_input = types.MethodType(Gr00tPolicy_ACG._get_action_from_normalized_input, policy)
            policy.model.get_action = types.MethodType(GR00T_N1_ACG.get_action, policy.model)
            policy.model.action_head.get_action = types.MethodType(FlowmatchingActionHead_ACG.get_action, policy.model.action_head)
        elif args.guidance_option == "cfg":
            policy.get_action = types.MethodType(Gr00tPolicy_CFG.get_action, policy)
            policy._get_action_from_normalized_input = types.MethodType(Gr00tPolicy_CFG._get_action_from_normalized_input, policy)
            policy.model.get_action = types.MethodType(GR00T_N1_CFG.get_action, policy.model)
            policy.model.action_head.get_action = types.MethodType(FlowmatchingActionHead_CFG.get_action, policy.model.action_head)
        elif args.bestofN_option == "motion":
            policy.get_action = types.MethodType(Gr00tPolicy_Motion.get_action, policy)
            policy._get_action_from_normalized_input = types.MethodType(Gr00tPolicy_Motion._get_action_from_normalized_input, policy)
            policy.model.get_action = types.MethodType(GR00T_N1_Motion.get_action, policy.model)
        elif args.bestofN_option == "knn":
            policy.get_action = types.MethodType(Gr00tPolicy_KNN.get_action, policy)
            policy._get_action_from_normalized_input = types.MethodType(Gr00tPolicy_KNN._get_action_from_normalized_input, policy)
            policy.model.get_action = types.MethodType(GR00T_N1_KNN.get_action, policy.model)

            ### init contrast model
            policy.contrast_image_generator = build_contrast_image_generator()
            policy.config = {
                "n_candidates": 12,
                "knn_k": 10,
                "action_horizon": args.action_horizon,
            }
        elif args.bestofN_option == "sdn":
            policy._get_action = types.MethodType(Gr00tPolicy_SDN._get_action, policy)
            policy.model.get_action = types.MethodType(Gr00tN1d7_SDN.get_action, policy.model)

            ### init contrast model
            policy.contrast_image_generator = build_contrast_image_generator()
            policy.config = {
                "n_candidates": 12,
                "knn_k": 10,
                "action_horizon": args.action_horizon,
                "top_k": 5,
                "long_ah": 16
            }


        # with PolicyServer(policy=policy, host=args.host, port=args.port) as server:
        #     try:
        #         server.run()
        #     except KeyboardInterrupt:
        #         print("\nShutting down server...")


        import numpy as np

        front_frame = np.random.randint(0, 256, (1, 480, 640, 3), dtype=np.uint8)
        wrist_frame = np.random.randint(0, 256, (1, 480, 640, 3), dtype=np.uint8)
        joints = np.random.uniform(-np.pi, np.pi, (1,6,)).astype(np.float32)
        gripper = np.random.uniform(0.0, 1.0, (1,1,)).astype(np.float32)
        instruction = "Use the right gripper to pick up the banana and place it into the pot. Then pick up the lid with the right gripper and place it on top of the pot to close it."


        obs = {
            "video": {
                "front": front_frame[None, ...],    # (1, 480, 640, 3) uint8, RGB — overhead cam
                "wrist": wrist_frame[None, ...],    # (1, 480, 640, 3) uint8, RGB — right wrist cam
            },
            "state": {
                "single_arm": joints[None, :],      # (1, 6) float — right arm joint positions, radians
                "gripper": gripper[None, :],        # (1, 1) float — gripper joint position
            },
            "language": {
                "annotation.human.task_description": [[instruction]]
            }
        }


        action, info = policy._get_action(obs)

        for key, value in action.items():
            print(f"Action: {key}: {value.shape}")
            print(np.sum(value))
        


    # Here is mainly a testing code
    elif args.client:
        # In this mode, we will send a random observation to the server and get an action back
        # This is useful for testing the server and client connection

        # Making prediction...
        # - obs: video.ego_view: (1, 256, 256, 3)
        # - obs: state.left_arm: (1, 7)
        # - obs: state.right_arm: (1, 7)
        # - obs: state.left_hand: (1, 6)
        # - obs: state.right_hand: (1, 6)
        # - obs: state.waist: (1, 3)

        # - action: action.left_arm: (16, 7)
        # - action: action.right_arm: (16, 7)
        # - action: action.left_hand: (16, 6)
        # - action: action.right_hand: (16, 6)
        # - action: action.waist: (16, 3)

        import numpy as np

        front_frame = np.random.randint(0, 256, (1, 480, 640, 3), dtype=np.uint8)
        wrist_frame = np.random.randint(0, 256, (1, 480, 640, 3), dtype=np.uint8)
        joints = np.random.uniform(-np.pi, np.pi, (1,6,)).astype(np.float32)
        gripper = np.random.uniform(0.0, 1.0, (1,1,)).astype(np.float32)
        instruction = "Use the right gripper to pick up the banana and place it into the pot. Then pick up the lid with the right gripper and place it on top of the pot to close it."


        obs = {
            "video": {
                "front": front_frame[None, ...],    # (1, 480, 640, 3) uint8, RGB — overhead cam
                "wrist": wrist_frame[None, ...],    # (1, 480, 640, 3) uint8, RGB — right wrist cam
            },
            "state": {
                "single_arm": joints[None, :],      # (1, 6) float — right arm joint positions, radians
                "gripper": gripper[None, :],        # (1, 1) float — gripper joint position
            },
            "language": {
                "annotation.human.task_description": [[instruction]]
            }
        }

        print(obs['video']['front'].shape)
        

        # obs = {
        #     # video keys (from video_keys in config)
        #     "video.cam_head": np.random.randint(0, 256, (1, 480, 640, 3), dtype=np.uint8),
        #     "video.cam_left": np.random.randint(0, 256, (1, 480, 640, 3), dtype=np.uint8),
        #     "video.cam_right": np.random.randint(0, 256, (1, 480, 640, 3), dtype=np.uint8),

        #     # state keys (13 keys, using the start:end ranges in modality.json)
        #     "state.left_arm": np.random.rand(1, 7),
        #     "state.right_arm": np.random.rand(1, 7),
        #     "state.left_hand": np.random.rand(1, 6),
        #     "state.right_hand": np.random.rand(1, 6),
        #     "state.velocity_left_arm": np.random.rand(1, 7),
        #     "state.velocity_right_arm": np.random.rand(1, 7),
        #     "state.velocity_left_hand": np.random.rand(1, 6),
        #     "state.velocity_right_hand": np.random.rand(1, 6),
        #     "state.effort_left_arm": np.random.rand(1, 7),
        #     "state.effort_right_arm": np.random.rand(1, 7),
        #     "state.effort_left_hand": np.random.rand(1, 6),
        #     "state.effort_right_hand": np.random.rand(1, 6),

        #     # action keys are omitted here since they are similar to state (but could be added similarly)

        #     # annotation
        #     "annotation.human.task_description": ["do your thing!"],
        # }
        for i in range(100):
            if args.http_server:
                action = _example_http_client_call(obs, args.host, args.port, args.api_token)
            else:
                action = _example_zmq_client_call(obs, args.host, args.port, args.api_token)

            for key, value in action.items():
                print(f"Action: {key}: {value.shape}")
                print(value)
    else:
        raise ValueError("Please specify either --server or --client")


if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)
    main(config)
