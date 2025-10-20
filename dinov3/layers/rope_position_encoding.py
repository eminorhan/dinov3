# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn


# RoPE positional embedding with no mixing of coordinates (axial) and no learnable weights
# Supports two parametrizations of the rope parameters: either using `base` or `min_period` and `max_period`.
class RopePositionEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float | None = 100.0,
        min_period: float | None = None,
        max_period: float | None = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        assert embed_dim % (4 * num_heads) == 0
        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError("Either `base` or `min_period`+`max_period` must be provided.")

        D_head = embed_dim // num_heads
        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.D_head = D_head
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords

        # Needs persistent=True because we do teacher.load_state_dict(student.state_dict()) to initialize the teacher
        self.dtype = dtype  # Don't rely on self.periods.dtype
        self.register_buffer("periods", torch.empty(D_head // 4, device=device, dtype=dtype), persistent=True)
        self._init_weights()

    def forward(self, *, H: int, W: int) -> tuple[Tensor, Tensor]:
        device = self.periods.device
        dtype = self.dtype
        dd = {"device": device, "dtype": dtype}

        # Prepare coords in range [-1, +1]
        if self.normalize_coords == "max":
            max_HW = max(H, W)
            coords_h = torch.arange(0.5, H, **dd) / max_HW  # [H]
            coords_w = torch.arange(0.5, W, **dd) / max_HW  # [W]
        elif self.normalize_coords == "min":
            min_HW = min(H, W)
            coords_h = torch.arange(0.5, H, **dd) / min_HW  # [H]
            coords_w = torch.arange(0.5, W, **dd) / min_HW  # [W]
        elif self.normalize_coords == "separate":
            coords_h = torch.arange(0.5, H, **dd) / H  # [H]
            coords_w = torch.arange(0.5, W, **dd) / W  # [W]
        else:
            raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)  # [H, W, 2]
        coords = coords.flatten(0, 1)  # [HW, 2]
        coords = 2.0 * coords - 1.0  # Shift range [0, 1] to [-1, +1]

        # Shift coords by adding a uniform value in [-shift, shift]
        if self.training and self.shift_coords is not None:
            shift_hw = torch.empty(2, **dd).uniform_(-self.shift_coords, self.shift_coords)
            coords += shift_hw[None, :]

        # Jitter coords by multiplying the range [-1, 1] by a log-uniform value in [1/jitter, jitter]
        if self.training and self.jitter_coords is not None:
            jitter_max = np.log(self.jitter_coords)
            jitter_min = -jitter_max
            jitter_hw = torch.empty(2, **dd).uniform_(jitter_min, jitter_max).exp()
            coords *= jitter_hw[None, :]

        # Rescale coords by multiplying the range [-1, 1] by a log-uniform value in [1/rescale, rescale]
        if self.training and self.rescale_coords is not None:
            rescale_max = np.log(self.rescale_coords)
            rescale_min = -rescale_max
            rescale_hw = torch.empty(1, **dd).uniform_(rescale_min, rescale_max).exp()
            coords *= rescale_hw

        # Prepare angles and sin/cos
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]  # [HW, 2, D//4]
        angles = angles.flatten(1, 2)  # [HW, D//2]
        angles = angles.tile(2)  # [HW, D]
        cos = torch.cos(angles)  # [HW, D]
        sin = torch.sin(angles)  # [HW, D]

        return (sin, cos)  # 2 * [HW, D]

    def _init_weights(self):
        device = self.periods.device
        dtype = self.dtype
        if self.base is not None:
            periods = self.base ** (2 * torch.arange(self.D_head // 4, device=device, dtype=dtype) / (self.D_head // 2))  # [D//4]
        else:
            base = self.max_period / self.min_period
            exponents = torch.linspace(0, 1, self.D_head // 4, device=device, dtype=dtype)  # [D//4] range [0, 1]
            periods = base**exponents  # range [1, max_period / min_period]
            periods = periods / base  # range [min_period / max_period, 1]
            periods = periods * self.max_period  # range [min_period, max_period]
        self.periods.data = periods


# 3D RoPE positional embedding with no mixing of coordinates (axial) and no learnable weights.
# This class is an adaptation of the 2D RoPE implementation above for 3D inputs like volumetric EM data.
# Supports two parametrizations of the rope parameters: either using `base` or `min_period` and `max_period`.
class RopePositionEmbedding3D(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float | None = 100.0,
        min_period: float | None = None,
        max_period: float | None = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        """
        Initializes the 3D RoPE module.

        Args:
            embed_dim (int): The embedding dimension.
            num_heads (int): The number of attention heads.
            base (float, optional): The base for the geometric progression of periods. Defaults to 100.0.
            min_period (float, optional): The minimum period for the sinusoidal embeddings.
            max_period (float, optional): The maximum period for the sinusoidal embeddings.
            normalize_coords (Literal["min", "max", "separate"]): The method for normalizing coordinates.
                "max": Normalize by the maximum of (D, H, W).
                "min": Normalize by the minimum of (D, H, W).
                "separate": Normalize each dimension independently.
            shift_coords (float, optional): If not None, shifts coordinates by a random value in [-shift, shift] during training.
            jitter_coords (float, optional): If not None, jitters coordinates by a log-uniform value during training.
            rescale_coords (float, optional): If not None, rescales coordinates by a log-uniform value during training.
            dtype (torch.dtype, optional): The data type for tensors.
            device (torch.device, optional): The device to place tensors on.
        """
        super().__init__()
        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError("Either `base` or `min_period`+`max_period` must be provided.")

        D_head = embed_dim // num_heads
        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.D_head = D_head
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords
        self.dtype = dtype

        # For 3D axial RoPE, the dimension must be split into 3 for D, H, W.
        # If D_head isn't divisible by 6, we use the largest multiple of 6 for RoPE
        # and leave the rest of the channels untouched (identity-mapped).
        self.D_rope = (D_head // 6) * 6

        # The number of periods is D_rope // 6, as we have 3 dimensions and each requires a sin/cos pair.
        self.register_buffer("periods", torch.empty(self.D_rope // 6, device=device, dtype=dtype), persistent=True)
        self._init_weights()

    def forward(self, *, D: int, H: int, W: int) -> tuple[Tensor, Tensor]:
        """
        Generates the sin and cos tensors for RoPE.

        Args:
            D (int): The depth of the input volume.
            H (int): The height of the input volume.
            W (int): The width of the input volume.

        Returns:
            A tuple of (sin, cos) tensors, each of shape [D*H*W, embed_dim // num_heads].
        """
        device = self.periods.device
        dtype = self.dtype
        dd = {"device": device, "dtype": dtype}

        # Prepare coords in range [0, 1] before shifting to [-1, +1]
        if self.normalize_coords == "max":
            max_DHW = max(D, H, W)
            coords_d = torch.arange(0.5, D, **dd) / max_DHW
            coords_h = torch.arange(0.5, H, **dd) / max_DHW
            coords_w = torch.arange(0.5, W, **dd) / max_DHW
        elif self.normalize_coords == "min":
            min_DHW = min(D, H, W)
            coords_d = torch.arange(0.5, D, **dd) / min_DHW
            coords_h = torch.arange(0.5, H, **dd) / min_DHW
            coords_w = torch.arange(0.5, W, **dd) / min_DHW
        elif self.normalize_coords == "separate":
            coords_d = torch.arange(0.5, D, **dd) / D
            coords_h = torch.arange(0.5, H, **dd) / H
            coords_w = torch.arange(0.5, W, **dd) / W
        else:
            raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")

        # Create a 3D grid of coordinates
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"), dim=-1)  # [D, H, W, 3]
        coords = coords.flatten(0, 2)  # [DHW, 3]
        coords = 2.0 * coords - 1.0  # Shift range [0, 1] to [-1, +1]

        # --- Coordinate Augmentations (during training) ---
        if self.training and self.shift_coords is not None:
            shift_dhw = torch.empty(3, **dd).uniform_(-self.shift_coords, self.shift_coords)
            coords += shift_dhw[None, :]

        if self.training and self.jitter_coords is not None:
            jitter_max = np.log(self.jitter_coords)
            jitter_dhw = torch.empty(3, **dd).uniform_(-jitter_max, jitter_max).exp()
            coords *= jitter_dhw[None, :]

        if self.training and self.rescale_coords is not None:
            rescale_max = np.log(self.rescale_coords)
            rescale = torch.empty(1, **dd).uniform_(-rescale_max, rescale_max).exp()
            coords *= rescale

        # Prepare angles and sin/cos
        # coords is [DHW, 3], periods is [D_rope//6]
        # angles becomes [DHW, 3, D_rope//6]
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        # Flatten the last two dimensions to combine pos info from D, H, W
        # angles becomes [DHW, 3 * D_rope//6] = [DHW, D_rope//2]
        angles = angles.flatten(1, 2)
        # Tile for the sin/cos pairs, resulting in [DHW, D_rope]
        angles = angles.tile(1, 2)
        cos = torch.cos(angles)  # [DHW, D_rope]
        sin = torch.sin(angles)  # [DHW, D_rope]

        # Pad sin/cos to the full head dimension if D_head is not divisible by 6
        if self.D_rope < self.D_head:
            pad_width = self.D_head - self.D_rope
            # We pad the last dimension on the right side.
            padding = (0, pad_width)
            sin = F.pad(sin, padding)
            cos = F.pad(cos, padding)

        return (sin, cos)

    def _init_weights(self):
        device = self.periods.device
        dtype = self.dtype
        if self.base is not None:
            # The denominator is D_rope // 3, which is 2 * (D_rope // 6)
            periods = self.base ** (2 * torch.arange(self.D_rope // 6, device=device, dtype=dtype) / (self.D_rope // 3))
        else:
            base = self.max_period / self.min_period
            exponents = torch.linspace(0, 1, self.D_rope // 6, device=device, dtype=dtype)
            periods = base**exponents
            periods = periods / base
            periods = periods * self.max_period
        self.periods.data = periods
