# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""Hungarian matching for training-style forward paths (vendored from upstream train stack)."""

import numpy as np
import torch
from sam3.model.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from scipy.optimize import linear_sum_assignment
from torch import nn


def _do_matching(cost, repeats=1, return_tgt_indices=False, do_filtering=False):
    if repeats > 1:
        cost = np.tile(cost, (1, repeats))

    i, j = linear_sum_assignment(cost)
    if do_filtering:
        valid_thresh = 1e8
        valid_ijs = [(ii, jj) for ii, jj in zip(i, j) if cost[ii, jj] < valid_thresh]
        i, j = zip(*valid_ijs) if len(valid_ijs) > 0 else ([], [])
        i, j = np.array(i, dtype=np.int64), np.array(j, dtype=np.int64)
    if return_tgt_indices:
        return i, j
    order = np.argsort(j)
    return i[order]


class BinaryHungarianMatcherV2(nn.Module):
    """
    Assignment between predictions and targets (used when training or when
    num_interactive_steps_val > 0 requires matching).
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        focal: bool = False,
        alpha: float = 0.25,
        gamma: float = 2.0,
        stable: bool = False,
        remove_samples_with_0_gt: bool = True,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.norm = nn.Sigmoid()
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, (
            "all costs cant be 0"
        )
        self.focal = focal
        if focal:
            self.alpha = alpha
            self.gamma = gamma
            self.stable = stable
        self.remove_samples_with_0_gt = remove_samples_with_0_gt

    @torch.no_grad()
    def forward(
        self,
        outputs,
        batched_targets,
        repeats=1,
        repeat_batch=1,
        out_is_valid=None,
        target_is_valid_padded=None,
    ):
        _, num_queries = outputs["pred_logits"].shape[:2]

        out_score = outputs["pred_logits"].squeeze(-1)  # (B, Q)
        out_bbox = outputs["pred_boxes"]  # (B, Q, 4))

        device = out_score.device

        num_boxes = batched_targets["num_boxes"].cpu()
        tgt_bbox = batched_targets["boxes_padded"]
        if self.remove_samples_with_0_gt:
            batch_keep = num_boxes > 0
            num_boxes = num_boxes[batch_keep]
            tgt_bbox = tgt_bbox[batch_keep]
            if target_is_valid_padded is not None:
                target_is_valid_padded = target_is_valid_padded[batch_keep]
        if repeat_batch > 1:
            num_boxes = num_boxes.repeat(repeat_batch)
            tgt_bbox = tgt_bbox.repeat(repeat_batch, 1, 1)
            if target_is_valid_padded is not None:
                target_is_valid_padded = target_is_valid_padded.repeat(repeat_batch, 1)

        if self.remove_samples_with_0_gt:
            if repeat_batch > 1:
                batch_keep = batch_keep.repeat(repeat_batch)
            out_score = out_score[batch_keep]
            out_bbox = out_bbox[batch_keep]
            if out_is_valid is not None:
                out_is_valid = out_is_valid[batch_keep]
        assert out_bbox.shape[0] == tgt_bbox.shape[0]
        assert out_bbox.shape[0] == num_boxes.shape[0]

        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
        )

        out_prob = self.norm(out_score)
        if not self.focal:
            cost_class = -out_prob.unsqueeze(-1).expand_as(cost_bbox)
        else:
            if self.stable:
                rescaled_giou = (-cost_giou + 1) / 2
                out_prob = out_prob.unsqueeze(-1).expand_as(cost_bbox) * rescaled_giou
                cost_class = -self.alpha * (1 - out_prob) ** self.gamma * torch.log(
                    out_prob
                ) + (1 - self.alpha) * out_prob**self.gamma * torch.log(1 - out_prob)
            else:
                log_out_prob = torch.nn.functional.logsigmoid(out_score)
                log_one_minus_out_prob = torch.nn.functional.logsigmoid(-out_score)
                cost_class = (
                    -self.alpha * (1 - out_prob) ** self.gamma * log_out_prob
                    + (1 - self.alpha) * out_prob**self.gamma * log_one_minus_out_prob
                )
            if not self.stable:
                cost_class = cost_class.unsqueeze(-1).expand_as(cost_bbox)

        assert cost_class.shape == cost_bbox.shape

        C = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        )
        do_filtering = out_is_valid is not None or target_is_valid_padded is not None
        if out_is_valid is not None:
            C = torch.where(out_is_valid[:, :, None], C, 1e9)
        if target_is_valid_padded is not None:
            C = torch.where(target_is_valid_padded[:, None, :], C, 1e9)
        C = C.cpu().numpy()
        costs = [C[i, :, :s] for i, s in enumerate(num_boxes.tolist())]
        return_tgt_indices = (
            do_filtering or torch.any(num_queries < num_boxes * max(repeats, 1)).item()
        )
        if len(costs) == 0:
            indices = []
            tgt_idx = torch.zeros(0).long().to(device) if return_tgt_indices else None
        elif return_tgt_indices:
            indices, tgt_indices = zip(
                *(
                    _do_matching(
                        c,
                        repeats=repeats,
                        return_tgt_indices=return_tgt_indices,
                        do_filtering=do_filtering,
                    )
                    for c in costs
                )
            )
            tgt_indices = list(tgt_indices)
            sizes = torch.cumsum(num_boxes, -1)[:-1]
            for i in range(1, len(tgt_indices)):
                tgt_indices[i] += sizes[i - 1].item()
            tgt_idx = torch.from_numpy(np.concatenate(tgt_indices)).long().to(device)
        else:
            indices = [
                _do_matching(c, repeats=repeats, do_filtering=do_filtering)
                for c in costs
            ]
            tgt_idx = None

        if self.remove_samples_with_0_gt:
            kept_inds = batch_keep.nonzero().squeeze(1)
            batch_idx = torch.as_tensor(
                sum([[kept_inds[i]] * len(src) for i, src in enumerate(indices)], []),
                dtype=torch.long,
                device=device,
            )
        else:
            batch_idx = torch.as_tensor(
                sum([[i] * len(src) for i, src in enumerate(indices)], []),
                dtype=torch.long,
                device=device,
            )

        if len(indices) > 0:
            src_idx = torch.from_numpy(np.concatenate(indices)).long().to(device)
        else:
            src_idx = torch.empty(0, dtype=torch.long, device=device)
        return batch_idx, src_idx, tgt_idx
