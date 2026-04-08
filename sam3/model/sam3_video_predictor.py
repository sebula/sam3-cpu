# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import torch
from sam3.logger import get_logger
from sam3.model.sam3_base_predictor import Sam3BasePredictor

logger = get_logger(__name__)


class Sam3VideoPredictor(Sam3BasePredictor):
    def __init__(
        self,
        checkpoint_path=None,
        bpe_path=None,
        has_presence_token=True,
        geo_encoder_use_img_cross_attn=True,
        strict_state_dict_loading=True,
        async_loading_frames=False,
        video_loader_type="cv2",
        apply_temporal_disambiguation: bool = True,
        compile: bool = False,
        device=None,
    ):
        super().__init__()
        self.async_loading_frames = async_loading_frames
        self.video_loader_type = video_loader_type
        from sam3.model_builder import build_sam3_video_model

        self.model = (
            build_sam3_video_model(
                checkpoint_path=checkpoint_path,
                bpe_path=bpe_path,
                has_presence_token=has_presence_token,
                geo_encoder_use_img_cross_attn=geo_encoder_use_img_cross_attn,
                strict_state_dict_loading=strict_state_dict_loading,
                apply_temporal_disambiguation=apply_temporal_disambiguation,
                compile=compile,
                device="cpu",
            )
            .to(torch.device("cpu"))
            .eval()
        )

    def remove_object(
        self,
        session_id: str,
        frame_idx: int = 0,
        obj_id: int = 0,
        is_user_action: bool = True,
    ):
        """Remove an object from tracking (SAM3 uses a simpler remove_object API)."""
        session = self._get_session(session_id)
        inference_state = session["state"]

        self.model.remove_object(
            inference_state=inference_state,
            obj_id=obj_id,
            is_user_action=is_user_action,
        )
        return {"is_success": True}

    def _get_session_stats(self):
        """Get a statistics string for live sessions."""
        live_session_strs = []
        for sid, s in self._all_inference_states.items():
            nf = s["state"]["num_frames"]
            live_session_strs.append(f"'{sid}' ({nf} frames)")
        joined = ", ".join(live_session_strs)
        return f"live sessions: [{joined}]"

    def _get_torch_and_gpu_properties(self):
        """Get a string for PyTorch properties."""
        return f"torch: {torch.__version__} (cpu-only fork)"


class Sam3VideoPredictorMultiGPU(Sam3VideoPredictor):
    """
    CPU-only fork: same constructor as upstream (``gpus_to_use`` is accepted and ignored).
    """

    def __init__(self, *model_args, gpus_to_use=None, **model_kwargs):
        super().__init__(*model_args, **model_kwargs)
        self.gpus_to_use = [] if gpus_to_use is None else list(gpus_to_use)
        self.rank = 0
        self.world_size = 1
        self.rank_str = "rank=0 (cpu-only fork)"
        self.device = torch.device("cpu")
        self.has_shutdown = False

    def shutdown(self):
        self.has_shutdown = True
        super().shutdown()
