# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""CPU-only fork: same function signatures as upstream; placement is always CPU."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Optional, Union

import torch


def resolve_torch_device(
    explicit: Optional[Union[str, torch.device]] = None,
) -> torch.device:
    """Upstream accepts ``device``; this fork always uses CPU."""
    return torch.device("cpu")


def default_build_device_string() -> str:
    return "cpu"


def non_blocking_to(device: torch.device) -> bool:
    return False


def inference_bf16_autocast(device: torch.device):
    return nullcontext()


def enter_persistent_bf16_autocast(device: torch.device):
    ctx = inference_bf16_autocast(device)
    ctx.__enter__()
    return ctx
