# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""Euclidean distance transform (CPU-only; matches batched cv2.DIST_L2 behavior)."""

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


def edt_triton(data: torch.Tensor) -> torch.Tensor:
    """
    Batched EDT for binary masks (B, H, W). Same role as the former Triton path; runs on CPU.
    """
    assert data.dim() == 3
    device = data.device
    dtype = data.dtype
    arr = data.detach().float().cpu().numpy()
    b, h, w = arr.shape
    out = np.empty((b, h, w), dtype=np.float32)
    for i in range(b):
        fg = (arr[i] > 0.5).astype(np.float32)
        # Nearest distance to a zero pixel (background); matches OpenCV DIST_L2 on binary mask
        out[i] = distance_transform_edt(fg, sampling=1.0)
    return torch.from_numpy(out).to(device=device, dtype=dtype)
