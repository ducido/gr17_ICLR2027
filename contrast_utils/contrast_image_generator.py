import cv2
import numpy as np
import os
import torch

from .inpainters import build_inpainter
from .instruction_templates import get_objects_from_instruction
from .mask_predictors import build_predictor, predict_masks_with_predictor
from .properties import _ROBOT_NAMES
from .utils import visualize_multi_objects



def mask_center_bbox_zero(rgb_image, ratio):
    """
    Tạo bbox ở giữa ảnh với kích thước = ratio * H, ratio * W và set vùng đó = 0
    """
    masked_image = rgb_image.copy()
    H, W = rgb_image.shape[:2]

    # kích thước bbox = ratio * ảnh
    box_h = int(H * ratio)
    box_w = int(W * ratio)

    # center
    center_y = int(H // (1.5))
    center_x = int(W // 2)

    # bbox
    y_min = max(0, center_y - box_h // 2)
    y_max = min(H - 1, center_y + box_h // 2)

    x_min = max(0, center_x - box_w // 2)
    x_max = min(W - 1, center_x + box_w // 2)

    # zero vùng bbox
    masked_image[y_min:y_max+1, x_min:x_max+1] = 0

    return masked_image


def mask_with_bbox_noise(rbg_image, mask, pad=10):
    """
    rbg_image: (H, W, 3)
    mask: (H, W) binary (0/1 hoặc bool)
    pad: số pixel mở rộng bbox
    """

    masked_image = rbg_image.copy()

    ys, xs = np.where(mask > 0)

    # nếu không có object thì return ảnh gốc
    if len(xs) == 0 or len(ys) == 0:
        return masked_image

    # bounding box
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    # padding
    H, W = mask.shape
    x_min = max(0, x_min - pad)
    x_max = min(W - 1, x_max + pad)
    y_min = max(0, y_min - pad)
    y_max = min(H - 1, y_max + pad)

    # tạo noise
    noise = np.random.randint(
        0, 256,
        size=(y_max - y_min + 1, x_max - x_min + 1, 3),
        dtype=np.uint8
    )

    # fill rectangle bằng noise
    masked_image[y_min:y_max+1, x_min:x_max+1] = noise

    return masked_image

def mask_with_bbox_zero(rbg_image, mask, pad=10):
    if mask is None:
        # không có mask → return ảnh gốc
        return rbg_image

    masked_image = rbg_image.copy()

    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return masked_image

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    H, W = mask.shape
    x_min = max(0, x_min - pad)
    x_max = min(W - 1, x_max + pad)
    y_min = max(0, y_min - pad)
    y_max = min(H - 1, y_max + pad)

    masked_image[y_min:y_max+1, x_min:x_max+1] = 0

    return masked_image

def mask_to_points(mask):
    if not mask.any():
        return None
    points = np.argwhere(mask)
    # sample 5 points
    if len(points) > 5:
        points = points[np.random.choice(len(points), 5, replace=False)]
    # y,x -> x,y
    return points[:, [1, 0]]


def mask_to_bbox(mask):
    if not mask.any():
        return None
    y, x = np.where(mask)
    return np.array([x.min(), y.min(), x.max(), y.max()])
    

def name_to_alias(name):
    s = name.split('_')
    rm_list = ['opened', 'light', 'generated', 'modified', 'objaverse', 'bridge', 'baked', 'v2']
    cleaned = []
    for w in s:
        if w[-2:] == "cm":
            # object size in object name
            continue
        if w not in rm_list:
            cleaned.append(w)
    return ' '.join(cleaned)


def my_print(*args):
    # get gpu id
    gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', '-1')
    if gpu_id == '0':
        print(args)


class ContrastImageGenerator:
    def __init__(self, 
                 camera_name=None, 
                 by="gt",
                 inpaint_mode="lama",
                 color="auto",
                 sigma=5,
                 version=2,
                 get_all_parts=False):
        self.camera_name = camera_name
        self.by = by
        self.color = color
        self.sigma = sigma
        assert version in [2, 3]
        self.version = version
        self.get_all_parts = get_all_parts
        
        self.mask_objects = None
        self.keep_objects = None
        self.task_description = None
        self.predictor = None
        self.inpainter = build_inpainter(inpaint_mode)
    
    def generate(self, rgb_image, task_description, is_inpaint=True):

        # if 'wrist' in image_key:
        #     rgb = self._get_rgb_image(obs, image_key)
        #     masked_image = mask_center_bbox_zero(rgb, ratio=2/3)
        #     return masked_image

        print(task_description, self.task_description)
        if task_description != self.task_description:
            self.reset_mask_and_keep_object_names(task_description)
            self.task_description = task_description

            self.predictor = build_predictor(self.by)
            if self.by in ["point_tracking", "box_tracking"]:
                raise NotImplementedError("point_tracking and box_tracking are not implemented yet")
                self._set_points_or_boxes(obs)
    
        mask, excluded_mask = self.get_mask_by_predictor(rgb_image, reverse_mask=False)

        if is_inpaint:
            print("is_inpaint")
            image = self.inpainter.inpaint(rgb_image, mask, excluded_mask)
        else:

            print("No inpainting, masking objects")
            masked_image = np.where(mask[..., None] == 0, rbg_image, 0)
            image = masked_image

            # print("No inpainting, masking objects with bbox noise")
            # rbg_image = self._get_rgb_image(obs)
            # masked_image = mask_with_bbox_noise(rbg_image, mask, pad=10)
            # image = masked_image

            # masked_image = mask_with_bbox_noise(rbg_image, mask, pad=10)
            # print("No inpainting, masking objects with bbox zero")
            # rbg_image = self._get_rgb_image(obs, image_key)
            # masked_image = mask_with_bbox_zero(rbg_image, mask, pad=5)
            # image = masked_image
        return image
    

    def simple_generate(self, obs, image_key):
        rbg_image = self._get_rgb_image(obs, image_key)
        masked_image = mask_center_bbox_zero(rbg_image, ratio=1/2)
        return masked_image

    def reset(self):
        self.task_description = None
        
    def reset_mask_and_keep_object_names(self, task_description):
        self.mask_objects = get_objects_from_instruction(task_description, self.get_all_parts)
        self.keep_objects = ["robot"]
        # my_print('reset:', self.mask_objects, self.keep_objects)

    
    def get_mask_by_predictor(self, image, reverse_mask=False):
        objs = self.mask_objects + self.keep_objects
        masks = predict_masks_with_predictor(image, objs, self.predictor)
        # my_print('get:', len(masks), objs)
        mask_obj_masks, keep_obj_masks = masks[:len(self.mask_objects)], masks[len(self.mask_objects):len(self.mask_objects) + len(self.keep_objects)]
        robot_mask = masks[objs.index('robot')] if 'robot' in objs else None
        mask = self._add_reserve_keep_mask(image.shape[:2], mask_obj_masks, reverse_mask, keep_obj_masks)
        return mask, robot_mask
    
    def _set_points_or_boxes(self, obs):
        assert self.by in ["point_tracking", "box_tracking"]
        assert self.predictor is not None, "Predictor is not initialized"
        seg = self._get_segmentation(obs)
        name2id = self._get_name_to_id()

        masks = []
        for obj_name in self.mask_objects + self.keep_objects:
            if obj_name == 'robot':
                robot_mask = np.zeros_like(seg, dtype=bool)
                for robot_name in _ROBOT_NAMES:
                    robot_mask |= self._get_object_mask_by_gt(seg, name2id, robot_name)
                masks.append(robot_mask)
            else:
                masks.append(self._get_object_mask_by_gt(seg, name2id, obj_name))

        if self.by == "point_tracking":
            points = [mask_to_points(mask) for mask in masks]
            self.predictor.predictor.set_points(points)
        elif self.by == "box_tracking":
            boxes = [mask_to_bbox(mask) for mask in masks]
            self.predictor.predictor.set_boxes(boxes)

    def _add_reserve_keep_mask(self, shape, masks, reverse_mask, keep_masks):
        mask = np.zeros(shape, dtype=bool)
        for obj_mask in masks:
            if obj_mask is not None:
                mask |= obj_mask

        if reverse_mask:
            mask = ~mask

        for obj_mask in keep_masks:
            if obj_mask is not None:
                mask[obj_mask] = False

        return mask
    
    def _get_rgb_image(self, obs, image_key):
        image = obs[image_key]
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if len(image.shape) == 4 and image.shape[0] == 1:
            image = image[0] 
        if len(image.shape) == 5 and image.shape[0] == 1:
            image = image[0][0]
        return image