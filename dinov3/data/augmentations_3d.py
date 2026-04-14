# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math
import random
import logging
import numpy as np
import torch
from torchvision.transforms import v2

logger = logging.getLogger("dinov3")

class MaskingGenerator3D:
    def __init__(self, input_size, num_masking_patches=None, min_num_patches=4, max_num_patches=None, min_aspect=0.3, max_aspect=None):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 3
        self.frames, self.height, self.width = input_size
        self.num_patches = self.frames * self.height * self.width
        self.num_masking_patches = num_masking_patches if num_masking_patches is not None else self.num_patches
        self.min_num_patches = min_num_patches
        self.max_num_patches = max_num_patches if max_num_patches is not None else self.num_patches
        self.max_aspect = max_aspect or (1 / min_aspect)
        self.min_aspect = min_aspect

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for attempt in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = random.uniform(self.min_aspect, self.max_aspect)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            t = random.randint(1, self.frames)
            
            if h < self.height and w < self.width and t <= self.frames:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)
                time = random.randint(0, self.frames - t)

                num_masked = mask[time: time + t, top: top + h, left: left + w].sum()
                if 0 < h * w * t - num_masked <= max_mask_patches:
                    for i in range(time, time + t):
                        for j in range(top, top + h):
                            for k in range(left, left + w):
                                if mask[i, j, k] == 0:
                                    mask[i, j, k] = 1
                                    delta += 1
                if delta > 0:
                    break
        return delta

    def __call__(self):
        mask = np.zeros((self.frames, self.height, self.width), dtype=bool)
        mask_count = 0
        while mask_count < self.max_num_patches:
            max_mask_patches = self.max_num_patches - mask_count
            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta
        return mask.flatten()


class DataAugmentationDINO3D:
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        **kwargs
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.normalize = v2.Normalize(mean=mean, std=std)

    def _apply_spatial_transforms(self, clip, crop_size):
        C, D, H, W = clip.shape
        clip = torch.nn.functional.interpolate(clip.unsqueeze(0), size=(D, crop_size, crop_size), mode='trilinear', align_corners=False).squeeze(0)
        return self.normalize(clip)

    def __call__(self, image):
        global_crops = [self._apply_spatial_transforms(image, self.global_crops_size) for _ in range(2)]
        local_crops = [self._apply_spatial_transforms(image, self.local_crops_size) for _ in range(self.local_crops_number)]
        return global_crops, local_crops