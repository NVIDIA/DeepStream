# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F


class BboxCropPostprocessor:
    """Postprocessor on PeopleNet: crop detected bboxes for C-RADIO.

    Runs after DeepStream inference on peoplenet.  Receives the
    DS_METADATA ``detections`` dict and the DS_IMAGE
    ``decoded_frames`` (RGB uint8 HWC tensor extracted from the
    DeepStream pipeline).

    For each detected bounding box the processor:
      1. Crops the bbox region from ``decoded_frames``.
      2. Resizes to 256x256 (bicubic).
      3. Normalises with CLIP mean/std.

    Returns:
      * ``pixel_values`` — list of N ``[3, 256, 256]`` float32 tensors
      * ``detections``   — the original detection dict (passthrough)

    The list of crops is passed to the cradio ModelOperator, which
    batches them via ``stack_tensors_in_dict`` so the TRT engine
    receives ``[N, 3, 256, 256]``.

    If no bboxes are detected a single dummy crop is produced so the
    downstream model always receives a valid batch.
    """
    name = "bbox-crop-postprocessor"

    def __init__(self, config):
        # must not exceed cradio max_batch_size
        self._max_crops = int(config.get("max_crops", 16))
        self._mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        self._std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

    def __call__(self, detections, decoded_frames):
        # decoded_frames is an RGB uint8 HWC tensor from DS_IMAGE output
        if decoded_frames is None:
            raise ValueError("decoded_frames is None")
        if not isinstance(decoded_frames, torch.Tensor):
            try:
                decoded_frames = torch.as_tensor(decoded_frames)
            except (TypeError, ValueError) as e:
                raise ValueError(f"decoded_frames could not be converted to tensor: {e}") from e
        if decoded_frames.ndim != 3:
            raise ValueError(
                f"decoded_frames must be 3D (HWC or CHW), got ndim={decoded_frames.ndim}, shape={decoded_frames.shape}"
            )
        if decoded_frames.ndim == 3 and decoded_frames.shape[0] == 3:
            decoded_frames = decoded_frames.permute(1, 2, 0)  # CHW → HWC
        # Expect HWC: (height, width, 3)
        if decoded_frames.shape[2] != 3:
            raise ValueError(
                f"decoded_frames must have 3 channels (RGB), got shape={decoded_frames.shape}"
            )
        frame_h, frame_w = decoded_frames.shape[0], decoded_frames.shape[1]
        # -- Crop each bbox ---------------------------------------------------
        bboxes = detections.get("bboxes", [])
        meta_shape = detections.get("shape", [frame_h, frame_w])
        scale_y = frame_h / meta_shape[0] if meta_shape[0] > 0 else 1.0
        scale_x = frame_w / meta_shape[1] if meta_shape[1] > 0 else 1.0

        crops = []
        for bbox in bboxes[: self._max_crops]:
            left, top, right, bottom = bbox
            left = int(left * scale_x)
            top = int(top * scale_y)
            right = int(right * scale_x)
            bottom = int(bottom * scale_y)

            left = max(0, min(left, frame_w - 1))
            top = max(0, min(top, frame_h - 1))
            right = max(left + 1, min(right, frame_w))
            bottom = max(top + 1, min(bottom, frame_h))

            crop = decoded_frames[top:bottom, left:right, :]  # HWC
            crop = crop.permute(2, 0, 1).float() / 255.0       # CHW [0,1]
            crop = F.interpolate(
                crop.unsqueeze(0), size=(256, 256),
                mode="bicubic", align_corners=False,
            ).squeeze(0)  # [3, 256, 256]

            mean = self._mean.to(crop.device)
            std = self._std.to(crop.device)
            crop = (crop - mean) / std
            crops.append(crop)

        if len(crops) == 0:
            crops = [torch.zeros(3, 256, 256)]

        # Return list of [3, 256, 256] crops — the model operator will
        # batch them via stack_tensors_in_dict into [N, 3, 256, 256].
        return (crops, detections)
