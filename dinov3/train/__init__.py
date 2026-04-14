# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from .multidist_meta_arch import MultiDistillationMetaArch
from .ssl_meta_arch import SSLMetaArch, SSLMetaArch3D
from .train import get_args_parser, main
