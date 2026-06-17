# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from functools import partial

import torch

from dinov3.eval.segmentation.models.backbone.dinov3_adapter import DINOv3_Adapter
from dinov3.eval.segmentation.models.heads.linear_head import LinearHead, LinearHead3D
from dinov3.eval.segmentation.models.heads.mask2former_head import Mask2FormerHead
from dinov3.eval.utils import ModelWithIntermediateLayers
from dinov3.layers import PatchEmbed


def _get_backbone_out_indices(model: torch.nn.Module, backbone_out_layers: str = "four_even_intervals"):
    """
    Get indices for output layers of the ViT backbone. For now there are 3 options available:

    "last" : only extract the last layer, used in segmentation tasks with a bn head.
    "four_last" : extract the four last layers
    "four_even_intervals" : extract outputs every 1/4 of the total number of blocks

    ViT/S (12 blocks): [2, 5, 8, 11]
    ViT/B (12 blocks): [2, 5, 8, 11]
    ViT/L (24 blocks): [5, 11, 17, 23] (classic), [4, 11, 17, 23] (used in the paper)
    ViT/g (40 blocks): [9, 19, 29, 39]
    """
    n_blocks = getattr(model, "n_blocks", 1)
    if backbone_out_layers == "last":
        out_indices = [n_blocks - 1]
    elif backbone_out_layers == "four_last":
        out_indices = [i for i in range(n_blocks - 4, n_blocks)]
    elif backbone_out_layers == "four_even_intervals":
        # Take indices that were used in the paper (for ViT/L only)
        if n_blocks == 24:
            out_indices = [4, 11, 17, 23]
        else:
            out_indices = [i * (n_blocks // 4) - 1 for i in range(1, 5)]
    else:
        # Failsafe: Catch typos or invalid string arguments
        raise ValueError(f"Unsupported backbone_out_layers value: '{backbone_out_layers}'. Valid options are: 'last', 'four_last', 'four_even_intervals'.")

    assert all([out_index < n_blocks for out_index in out_indices])
    
    return out_indices


class FeatureDecoder(torch.nn.Module):
    def __init__(self, segmentation_model: torch.nn.ModuleList):
        super().__init__()
        self.segmentation_model = segmentation_model

    def forward(self, inputs):
        for module in self.segmentation_model:
            inputs = module.forward(inputs)
        return inputs

    def predict(self, inputs, rescale_to=(512, 512)):
        with torch.inference_mode():
            out = self.segmentation_model[0](inputs)  # backbone forward
            out = self.segmentation_model[1].predict(out, rescale_to=rescale_to)  # decoder head prediction
        return out


def build_segmentation_decoder(
    backbone_model,
    backbone_out_layers: str = "four_even_intervals",
    decoder_type="linear",
    hidden_dim=2048,
    num_classes=150,
):
    backbone_indices_to_use = _get_backbone_out_indices(backbone_model, backbone_out_layers)
    if decoder_type == "m2f":
        backbone_model = DINOv3_Adapter(backbone_model, interaction_indexes=backbone_indices_to_use)
        embed_dim = backbone_model.backbone.embed_dim
        patch_size = backbone_model.patch_size
        decoder = Mask2FormerHead(
            input_shape={
                "1": [embed_dim, patch_size * 4, patch_size * 4, 4],
                "2": [embed_dim, patch_size * 2, patch_size * 2, 4],
                "3": [embed_dim, patch_size, patch_size, 4],
                "4": [embed_dim, int(patch_size / 2), int(patch_size / 2), 4],
            },
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            ignore_value=255,
        )
    elif decoder_type == "linear":
        backbone_model = ModelWithIntermediateLayers(
            backbone_model,
            n=backbone_indices_to_use,
            reshape=True,
            return_class_token=False,
        )
        # Important: we freeze the backbone
        embed_dim = backbone_model.feature_model.embed_dim
        if isinstance(embed_dim, int):
            if backbone_out_layers in ["four_last", "four_even_intervals"]:
                embed_dim = [embed_dim] * 4
            else:
                embed_dim = [embed_dim]
        # pick 2D or 3D head based on the patch_embed class of the backbone        
        if isinstance(backbone_model.feature_model.patch_embed, PatchEmbed):
            decoder = LinearHead(in_channels=embed_dim, n_output_channels=num_classes)
        else:
            decoder = LinearHead3D(in_channels=embed_dim, n_output_channels=num_classes)
    else:
        raise ValueError(f'Unsupported decoder "{decoder_type}"')

    segmentation_model = FeatureDecoder(torch.nn.ModuleList([backbone_model, decoder]))
    return segmentation_model
