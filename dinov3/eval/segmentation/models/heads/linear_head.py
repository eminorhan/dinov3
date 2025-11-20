# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearHead(nn.Module):
    """Linear layer for semantic segmentation."""

    def __init__(
        self,
        in_channels,
        n_output_channels,
        use_cls_token=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.channels = sum(in_channels)
        if use_cls_token:
            self.channels *= 2  # concatenate CLS to patch tokens
        self.n_output_channels = n_output_channels
        self.use_cls_token = use_cls_token
        self.conv = nn.Conv2d(self.channels, self.n_output_channels, kernel_size=1, padding=0, stride=1)
        self.dropout = nn.Dropout2d(0.1)
        nn.init.normal_(self.conv.weight, mean=0, std=0.01)
        nn.init.constant_(self.conv.bias, 0)

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """
        inputs = [
            torch.nn.functional.interpolate(
                input=x,
                size=inputs[0].shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            for x in inputs
        ]
        inputs = torch.cat(inputs, dim=1)
        return inputs

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels, H, W) which is feature map for last layer of decoder head.
        """
        # accept lists (for cls token)
        inputs = list(inputs)
        for i, x in enumerate(inputs):
            if self.use_cls_token:
                assert len(x) == 2, "Missing class tokens"
                x, cls_token = x[0], x[1]
                if len(x.shape) == 2:
                    x = x[:, :, None, None]
                cls_token = cls_token[:, :, None, None].expand_as(x)
                inputs[i] = torch.cat((x, cls_token), 1)
            else:
                if len(x.shape) == 2:
                    x = x[:, :, None, None]
                inputs[i] = x
        x = self._transform_inputs(inputs)
        return x

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.dropout(output)
        output = F.layer_norm(output, output.shape[-3:])
        output = self.conv(output)
        return output

    def predict(self, x, rescale_to=(512, 512)):
        """
        Predict function used in evaluation.
        No dropout is used, and the output is rescaled to the ground truth
        for computing metrics.
        """
        x = self._forward_feature(x)
        x = F.layer_norm(x, x.shape[-3:])
        x = self.conv(x)
        x = F.interpolate(input=x, size=rescale_to, mode="bilinear")
        return x


class LinearHead3D(nn.Module):
    """
    Linear layer for 3D semantic segmentation on volumetric data.
    """

    def __init__(
        self,
        in_channels,
        n_output_channels,
        use_cls_token=False,
    ):
        """
        Args:
            in_channels (list[int]): A list of the number of input channels from the ViT backbone's feature maps.
            n_output_channels (int): The number of output channels for the segmentation mask (e.g., number of classes).
            use_cls_token (bool): Whether to concatenate the class token to the patch tokens.
        """
        super().__init__()
        self.in_channels = in_channels
        self.channels = sum(in_channels)
        if use_cls_token:
            self.channels *= 2  # concatenate CLS to patch tokens
        self.n_output_channels = n_output_channels
        self.use_cls_token = use_cls_token

        # Use 3D convolution for volumetric data
        self.conv = nn.Conv3d(self.channels, self.n_output_channels, kernel_size=1, padding=0, stride=1)

        # Initialize weights
        nn.init.normal_(self.conv.weight, mean=0, std=0.01)
        nn.init.constant_(self.conv.bias, 0)

    def _transform_inputs(self, inputs):
        """
        Upsamples and concatenates the list of feature maps.

        Args:
            inputs (list[Tensor]): List of multi-level 3D features from the backbone. Expected shape: [(B, C_i, D_i, H_i, W_i), ...].

        Returns:
            Tensor: The transformed and concatenated inputs of shape (B, C_sum, D, H, W).
        """
        # Upsample all features to the spatial size of the first feature map using trilinear interpolation
        target_size = inputs[0].shape[2:]  # Get (D, H, W) from the first feature map
        inputs = [
            torch.nn.functional.interpolate(
                input=x,
                size=target_size,
                mode="trilinear",
                align_corners=False,
            )
            for x in inputs
        ]
        # Concatenate along the channel dimension
        inputs = torch.cat(inputs, dim=1)
        return inputs

    def _forward_feature(self, inputs):
        """
        Processes inputs, handling the class token if specified, before the final convolution.

        Args:
            inputs (list[Tensor]): List of multi-level 3D features.

        Returns:
            feats (Tensor): A tensor of shape (B, self.channels, D, H, W) ready for the final layer.
        """
        # Ensure inputs are in a list
        inputs = list(inputs)
        for i, x in enumerate(inputs):
            if self.use_cls_token:
                assert len(x) == 2, "Input must be a tuple of (patch_tokens, cls_token) when use_cls_token=True"
                x, cls_token = x[0], x[1]
                # Add spatial dims to tokens if they are flat (B, C) -> (B, C, 1, 1, 1)
                if len(x.shape) == 2:
                    x = x[:, :, None, None, None]
                # Expand class token to match spatial dimensions of patch tokens
                cls_token = cls_token[:, :, None, None, None].expand_as(x)
                inputs[i] = torch.cat((x, cls_token), 1)
            else:
                # Add spatial dims if necessary
                if len(x.shape) == 2:
                    x = x[:, :, None, None, None]
                inputs[i] = x
        
        x = self._transform_inputs(inputs)
        return x

    def forward(self, inputs):
        """
        Forward pass for training.
        
        Args:
            inputs (list[Tensor]): List of feature maps from the ViT backbone.

        Returns:
            Tensor: The output segmentation logits of shape (B, n_output_channels, D, H, W).
        """
        output = self._forward_feature(inputs)
        # Normalize over the spatial dimensions (D, H, W) for each channel
        output = F.layer_norm(output, output.shape[-4:])
        output = self.conv(output)
        return output

    def predict(self, x, rescale_to=(512, 512, 512)):
        """
        Predict function for evaluation/inference. No dropout is applied.

        Args:
            x (list[Tensor]): The input features from the backbone.
            rescale_to (tuple): The target (D, H, W) to resize the output segmentation to.

        Returns:
            Tensor: The final, rescaled segmentation logits.
        """
        x = self._forward_feature(x)
        x = F.layer_norm(x, x.shape[-4:])
        x = self.conv(x)
        # Upsample the final output to the desired size for evaluation
        x = F.interpolate(input=x, size=rescale_to, mode="trilinear", align_corners=False)
        return x