# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Integration test: load SAM 3.1 weights from Hugging Face and run one image + text prompt."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

_FIXTURES = Path(__file__).resolve().parent / "fixtures"
TRUCK_JPG = _FIXTURES / "truck.jpg"


class SAMSegmentExtractor:
    def __init__(self, device: str = "cpu"):
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        model = build_sam3_image_model(
            device=device,
            load_from_HF=True,
            hf_checkpoint_version="sam3.1",
        )
        self._processor = Sam3Processor(model, device=device)

    @torch.no_grad()
    def extract(
        self,
        image: Image.Image | np.ndarray,
        latent_h: int,
        latent_w: int,
        prompts: list[str] | None = None,
    ) -> tuple[np.ndarray, int]:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        if not prompts:
            prompts = ["objects"]

        inference_state = self._processor.set_image(image)
        all_masks: list[np.ndarray] = []

        for prompt in prompts:
            output = self._processor.set_text_prompt(
                prompt=prompt,
                state=inference_state,
            )
            masks = output.get("masks", [])
            scores = output.get("scores", [])

            if isinstance(masks, torch.Tensor):
                mlist = [masks[i] for i in range(masks.shape[0])]
            else:
                mlist = list(masks)

            for i, m in enumerate(mlist):
                if i < len(scores) and float(scores[i]) < 0.5:
                    continue
                mask_np = m.cpu().numpy() if isinstance(m, torch.Tensor) else np.array(m)
                if mask_np.ndim == 3:
                    mask_np = mask_np.squeeze()
                min_area = mask_np.shape[0] * mask_np.shape[1] * 0.01
                if mask_np.sum() >= min_area:
                    all_masks.append(mask_np)

        if not all_masks:
            return np.zeros((latent_h, latent_w), dtype=np.uint8), 0

        all_masks.sort(key=lambda m: m.sum(), reverse=True)
        n_objects = min(len(all_masks), 255)
        h, w = all_masks[0].shape[:2]
        seg_map_full = np.zeros((h, w), dtype=np.uint8)
        for rank, mask in enumerate(all_masks[:n_objects]):
            seg_map_full[mask > 0] = rank + 1

        seg_map_tensor = torch.from_numpy(seg_map_full).unsqueeze(0).unsqueeze(0).float()
        seg_map_resized = F.interpolate(
            seg_map_tensor,
            size=(latent_h, latent_w),
            mode="nearest",
        )
        return seg_map_resized.squeeze().byte().numpy(), n_objects


@pytest.fixture
def truck_image():
    assert TRUCK_JPG.is_file(), f"Missing test fixture: {TRUCK_JPG}"
    return Image.open(TRUCK_JPG).convert("RGB")


def test_sam3_image_cpu_hub_sam3_1_weights(truck_image):
    """Downloads facebook/sam3.1 (default), loads checkpoint, runs set_image + set_text_prompt."""
    ex = SAMSegmentExtractor()
    seg, n = ex.extract(truck_image, latent_h=8, latent_w=8, prompts=["truck"])
    assert seg.shape == (8, 8)
    assert seg.dtype == np.uint8
