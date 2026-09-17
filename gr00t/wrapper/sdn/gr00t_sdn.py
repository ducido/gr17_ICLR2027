

from pathlib import Path
from typing import Any
import copy
from PIL import Image
import os

import numpy as np
import torch
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7, Gr00tN1d7ActionHead
from transformers.feature_extraction_utils import BatchFeature
from gr00t.data.types import MessageType, ModalityConfig, VLAStepData

class Gr00tPolicy_SDN(Gr00tPolicy):

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Internal method to compute actions from observations.

        Pipeline:
        1. Unbatch observations into individual samples
        2. Convert each to VLAStepData and process
        3. Collate into model input batch
        4. Run model inference
        5. Decode and unnormalize actions

        Args:
            observation: Batched observation dictionary
            options: Optional parameters (currently unused)

        Returns:
            Tuple of (actions_dict, info_dict)
        """
        contrast_observation = copy.deepcopy(observation)

        static_image = contrast_observation['video']['front'][0][0]
        wrist_image = contrast_observation['video']['wrist'][0][0]
        prompt = contrast_observation['language']['annotation.human.task_description'][0][0]

        contrast_image = self.contrast_image_generator.generate(static_image, prompt, is_inpaint=True)
        contrast_wrist_image = self.contrast_image_generator.generate(wrist_image, prompt, is_inpaint=True)


        # idx = len(os.listdir("debug_images"))
        # debug_dir = '/projects/extern/kisski/kisski-spath/dir.project/VLA_Imit/gr17_ICLR2027/debug_images'
        # img = Image.fromarray(contrast_image)
        # img.save(f"{debug_dir}/contrast_image_{idx}.png")

        # wrist_img = Image.fromarray(contrast_wrist_image)
        # wrist_img.save(f"{debug_dir}/contrast_wrist_image_{idx}.png")

        # breakpoint()

        contrast_observation['video']['front'] = contrast_image[None, None, ...]
        contrast_observation['video']['wrist'] = contrast_wrist_image[None, None, ...]


        # Step 1: Split batched observation into individual observations
        unbatched_observations = self._unbatch_observation(observation)
        contrast_unbatched_observations = self._unbatch_observation(contrast_observation)
        processed_inputs = []
        contrast_processed_inputs = []  

        # Step 2: Process each observation through the VLA processor
        states = []
        for obs in unbatched_observations:
            vla_step_data = self._to_vla_step_data(obs)
            states.append(vla_step_data.states)  # dict[str, np.ndarray[np.float32, (T, D)]]
            messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
            processed_inputs.append(self.processor(messages))

        # Step 2: Process each observation through the VLA processor
        contrast_states = []
        for contrast_obs in contrast_unbatched_observations:
            vla_step_data = self._to_vla_step_data(contrast_obs)
            contrast_states.append(vla_step_data.states)  # dict[str, np.ndarray[np.float32, (T, D)]]
            messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
            contrast_processed_inputs.append(self.processor(messages))

        # Step 3: Collate processed inputs into a single batch for model
        collated_inputs = self.collate_fn(processed_inputs)
        collated_inputs = _rec_to_dtype(collated_inputs, dtype=torch.bfloat16)

        # Step 3: Collate processed inputs into a single batch for model
        collated_contrast_inputs = self.collate_fn(contrast_processed_inputs)
        collated_contrast_inputs = _rec_to_dtype(collated_contrast_inputs, dtype=torch.bfloat16)

        # Step 4: Run model inference to predict actions
        with torch.inference_mode():
            model_pred = self.model.get_action(inputs=collated_inputs['inputs'], contrast_inputs=collated_contrast_inputs['inputs'], options=self.config)
        normalized_action = model_pred["action_pred"].float()

        # Step 5: Decode actions from normalized space back to physical units
        batched_states = {}
        for k in self.modality_configs["state"].modality_keys:
            batched_states[k] = np.stack([s[k] for s in states], axis=0)  # (B, T, D)
        unnormalized_action = self.processor.decode_action(
            normalized_action.cpu().numpy(), self.embodiment_tag, batched_states
        )

        # Cast all actions to float32 for consistency
        casted_action = {
            key: value.astype(np.float32) for key, value in unnormalized_action.items()
        }
        return casted_action, {}

class Gr00tN1d7_SDN(Gr00tN1d7):

    def get_action(self, inputs: dict, contrast_inputs: dict, options: dict[str, Any] | None = None) -> BatchFeature:
        """
        Generate actions using the complete model.
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        contrast_backbone_inputs, contrast_action_inputs = self.prepare_input(contrast_inputs)

        # Forward through backbone
        backbone_outputs = self.backbone(backbone_inputs)
        contrast_backbone_outputs = self.backbone(contrast_backbone_inputs)

        for k in backbone_outputs.keys():
            x = torch.cat([backbone_outputs[k], contrast_backbone_outputs[k]], dim=0)
            backbone_outputs[k] = split_repeat_concat(x, options['n_candidates'])

        for k in action_inputs.keys():
            if k != 'pixel_values':
                x = torch.cat([action_inputs[k], contrast_action_inputs[k]], dim=0)
                action_inputs[k] = split_repeat_concat(x, options['n_candidates'])
            else:
                action_inputs[k] = action_inputs[k] + contrast_action_inputs[k]

        action_outputs = self.action_head.get_action(backbone_outputs, action_inputs, options)

        raw_action_outputs = {}
        contrast_action_outputs = {}
        for k in action_outputs.keys():
            raw_action_outputs[k], contrast_action_outputs[k] = torch.chunk(action_outputs[k], 2, dim=0)
        raw = raw_action_outputs['action_pred'][:,:,:7]
        contrast = contrast_action_outputs['action_pred'][:,:,:7]

        ### best-of-N
        _, _, top_indices = cd_with_knn_topK(raw[:,:options['action_horizon']], contrast[:,:options['action_horizon']], knn_k=options['knn_k'], top_k=options['top_k'])
        top_K_best_action = raw[top_indices]
        print("top_K_best_action.shape:", top_K_best_action.shape)

        ### Smoothness
        _, jerk_rms, best_idx = jerk_smoothest_action(top_K_best_action[:, :options['long_ah']])
        print("jerk_rms:", jerk_rms)
        print("best_idx:", best_idx)


        action_outputs["action_pred"] = top_K_best_action[best_idx:best_idx+1][:,:options['action_horizon']]

        return action_outputs



# class Gr00tN1d7ActionHead_SDN(Gr00tN1d7ActionHead)


def cd_with_knn_topK(actions, contrast_actions, knn_k, top_k, eps=1e-8):
    """
    actions: (N, T, D)
    contrast_actions: (C, T, D)

    return:
        best_action: (1, T, D)
    """
    print(f"Best-of-N KNN k = {knn_k}, top_k = {top_k}")

    N = actions.shape[0]
    C = contrast_actions.shape[0]

    A = actions.reshape(N, -1).float()  # (N, TD)
    B = contrast_actions.reshape(C, -1).float()  # (C, TD)

    # kNN in B
    dist_AB = torch.cdist(A, B, p=2) ** 2   # (N, C)
    knn_dist_AB, _ = torch.topk(dist_AB, k=knn_k, largest=False, dim=1)
    R_B = knn_dist_AB.sum(dim=1)  # (N,)

    # kNN in A (exclude self)
    dist_AA = torch.cdist(A, A, p=2) ** 2   # (N, N)
    inf_mask = torch.eye(N, device=A.device) * 1e9
    dist_AA = dist_AA + inf_mask
    knn_dist_AA, _ = torch.topk(dist_AA, k=knn_k, largest=False, dim=1)
    R_A = knn_dist_AA.sum(dim=1)  # (N,)

    # score
    scores = torch.log(R_B + eps) - torch.log(R_A + eps)

    # lấy top K
    top_scores, top_indices = torch.topk(scores, k=top_k, largest=True)

    top_actions = actions[top_indices]  # (top_k, T, D)

    return top_actions, top_scores, top_indices

def jerk_smoothest_action(long_action):
    """
    long_action: torch.Tensor of shape (C, T, D)

    Returns:
        best_idx: int
        best_action: (T, D)
        jerk_rms: (C,)
    """
    assert long_action.ndim == 3
    C, T, D = long_action.shape
    assert T >= 4, "Need at least 4 timesteps to compute jerk"

    # Normalize long_action
    std = long_action.std(dim=(0,1), keepdim=True)
    norm_long_action = long_action / (std + 1e-8)

    # Compute jerk (C, T-3, D)
    jerk = (
        norm_long_action[:, 3:, :-1]
        - 3 * norm_long_action[:, 2:-1, :-1]
        + 3 * norm_long_action[:, 1:-2, :-1]
        - norm_long_action[:, :-3, :-1]
    )

    # ||j||^2 over action dim → (C, T-3)
    jerk_sq = (jerk ** 2).sum(dim=-1)
    # mean over time → (C,)
    jerk_mean = jerk_sq.mean(dim=1)
    # RMS → (C,)
    jerk_rms = torch.sqrt(jerk_mean + 1e-8)  # tránh nan
    # best candidate
    best_idx = torch.argmin(jerk_rms)
    best_action = long_action[best_idx:best_idx+1]
    return best_action, jerk_rms, best_idx

def split_repeat_concat(x, num_repeats):
    # split thành list 10 tensors [1, ...]
    xs = torch.split(x, 1, dim=0)

    out = []
    for xi in xs:
        num_dims = xi.dim()
        xi = xi.repeat(num_repeats, *[1] * (num_dims - 1))
        out.append(xi)

    return torch.cat(out, dim=0)




def _rec_to_dtype(x: Any, dtype: torch.dtype) -> Any:
    """Recursively convert all floating point tensors in a nested structure to the given dtype.

    Args:
        x: Input data structure (tensor, dict, list, or other)
        dtype: Target torch dtype for floating point tensors

    Returns:
        Data structure with floating point tensors converted to target dtype

    Warning:
        Non-floating point tensors will be left as is.
    """
    if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
        return x.to(dtype=dtype)
    # Handle dict-like objects (tianshou.BatchFeature is not dict but has items() method)
    elif isinstance(x, dict) or hasattr(x, "items"):
        return {k: _rec_to_dtype(v, dtype) for k, v in x.items()}  # type: ignore
    elif isinstance(x, list):
        return [_rec_to_dtype(v, dtype) for v in x]
    else:
        return x

