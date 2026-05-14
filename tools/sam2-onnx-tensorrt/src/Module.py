#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import time
from torch import nn
from typing import Any
from sam2.build_sam import build_sam2
from sam2.modeling.sam2_base import SAM2Base
from sam2.modeling.sam2_utils import get_1d_sine_pe
from sam2.utils.misc import fill_holes_in_mask_scores

class ImageEncoder(nn.Module):
    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__()
        self.model = sam_model
        self.no_mem_embed = sam_model.no_mem_embed  # [1,1,256]
        self.image_encoder = sam_model.image_encoder
        self.num_feature_levels = sam_model.num_feature_levels
        self.prepare_backbone_features = sam_model._prepare_backbone_features

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        start_time = time.time()
        backbone_out = self.image_encoder(image)  # {"vision_features","vision_pos_enc","backbone_fpn"}
        backbone_out["backbone_fpn"][0] = self.model.sam_mask_decoder.conv_s0(backbone_out["backbone_fpn"][0])
        backbone_out["backbone_fpn"][1] = self.model.sam_mask_decoder.conv_s1(backbone_out["backbone_fpn"][1])

        vision_pos_enc = backbone_out["vision_pos_enc"]  # three tensors
        backbone_fpn = backbone_out["backbone_fpn"]      # three tensor
        pix_feat = backbone_out["vision_features"]       # one tensor

        expanded_backbone_out = {
            "backbone_fpn": backbone_fpn,
            "vision_pos_enc": vision_pos_enc,
        }
        for i, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
            expanded_backbone_out["backbone_fpn"][i] = feat.expand(1, -1, -1, -1)
        for i, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
            expanded_backbone_out["vision_pos_enc"][i] = pos.expand(1, -1, -1, -1)

        (_, current_vision_feats, current_vision_pos_embeds, _) = self.prepare_backbone_features(expanded_backbone_out)

        current_vision_feat = current_vision_feats[-1] + self.no_mem_embed
        current_vision_feat2 = current_vision_feat.reshape(64, 64, 1, 256).permute(2, 3, 0, 1)  # [1,256,64,64]

        # flatten HWxNxC -> NxCxHxW
        high_res_features_0 = current_vision_feats[0].reshape(256, 256, 1, 32).permute(2, 3, 0, 1)  # [1, 32, 256, 256]
        high_res_features_1 = current_vision_feats[1].reshape(128, 128, 1, 64).permute(2, 3, 0, 1)  # [1, 64, 128, 128]

        # pix_feat              [1, 256, 64, 64]
        # high_res_features_0   [1, 32, 256, 256]
        # high_res_features_1   [1, 64, 128, 128]
        # current_vision_feat   [1, 256, 64, 64]
        # current_vision_pos_embed2 [4096, 1, 256]
        end_time = time.time()

        return pix_feat, high_res_features_0, high_res_features_1, current_vision_feat2, current_vision_pos_embeds[-1]

class MemAttention(nn.Module):
    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__()
        self.model = sam_model
        self.no_mem_embed = sam_model.no_mem_embed
        self.memory_attention = sam_model.memory_attention
        self.obj_ptr_tpos_proj = sam_model.obj_ptr_tpos_proj

    # @torch.no_grad()
    def forward(
        self,
        current_vision_feat: torch.Tensor,      # [1, 256, 64, 64]
        current_vision_pos_embed: torch.Tensor,  # [4096, 1, 256]
        memory_0: torch.Tensor,                  # [batch,num_obj_ptr,256]->[batch,num_obj_ptr,4,64]->[4*num_obj_ptr,batch,64]
        memory_1: torch.Tensor,                  # [batch,num_masks,64,64,64]->[batch,num_masks,64,4096]->[4096*num_masks,batch,64]
        memory_pos_embed: torch.Tensor,          # [y*4096,1,64]
        cond_frame_id_diff: torch.Tensor,        # single float, current frame id - first cond frame id
    ) -> tuple[Any]:
        start_time = time.time()
        num_obj_ptr_tokens = memory_0.shape[1] * 4  # old: shape[0]
        current_vision_feat = current_vision_feat.permute(2, 3, 0, 1).reshape(4096, 1, 256)
        current_vision_feat = current_vision_feat - self.no_mem_embed

        batch_size = memory_0.size()[0]
        num_obj_ptr = memory_0.size()[1]
        num_masks = memory_1.size()[1]

        current_vision_feat = current_vision_feat.repeat(1, batch_size, 1)
        current_vision_pos_embed = current_vision_pos_embed.repeat(1, batch_size, 1)

        # [20,16,256] -> [20,16,4,64] -> [16,4,20,64] -> [64,20,64]
        memory_0 = memory_0.reshape(batch_size, -1, 4, 64)
        memory_0 = memory_0.permute(1, 2, 0, 3).flatten(0, 1)

        # old [16,256] -> [16,1,4,64] -> [16,4,1,64] -> [64,1,64]
        # memory_0 = memory_0.reshape(-1,1,4,64)
        # memory_0 = memory_0.permute(0, 2, 1, 3).flatten(0, 1)

        # [20,7,64,64,64] -> [20,7,64,64*64] -> [7,64*64,20,64] -> [7*64*64,20, 64]
        memory_1 = memory_1.view(batch_size, -1, 64, 64*64).permute(1, 3, 0, 2)
        memory_1 = memory_1.reshape(-1, batch_size, 64)

        # old [7,64,64,64] -> [7,64,64*64] -> [7,64*64,64] -> [7*64*64,1, 64]
        # memory_1 = memory_1.view(-1, 64, 64*64).permute(0,2,1)
        # memory_1 = memory_1.reshape(-1,1,64)

        # [20,7*4096+64,64] -> [7*4096+64,20,64]
        memory_pos_embed = memory_pos_embed.permute(1, 0, 2)

        if True:
            # generate obj_pos as add_tpos_enc_to_obj_ptrs is used in SAM 2.1
            # add it to memory_pos_embed

            # obj_pos frame id is like [cond_frame_id_diff, 1, 2, 3,...,15]
            obj_pos = torch.cat([cond_frame_id_diff.unsqueeze(0),
                                 torch.arange(1, num_obj_ptr, dtype=torch.float32)])
            t_diff_max = 15.0
            obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=256)  # 256 channels # [num_obj_ptr,256]
            obj_pos = self.obj_ptr_tpos_proj(obj_pos)
            obj_pos = obj_pos.unsqueeze(1).expand(-1, batch_size, 64)  # [num_obj_ptr,batch_size,64]
            obj_pos = obj_pos.repeat_interleave(4, dim=0)  # [4*num_obj_ptr,batch_size,64]
            memory_pos_embed[num_masks*4096 : num_masks*4096+4*num_obj_ptr] = obj_pos


        memory = torch.cat((memory_1, memory_0), dim=0)
        pix_feat_with_mem = self.memory_attention(
            curr=current_vision_feat,
            curr_pos=current_vision_pos_embed,
            memory=memory,
            memory_pos=memory_pos_embed,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        # reshape the output (HW)xBxC => BxCxHxW
        image_embed = pix_feat_with_mem.permute(1, 2, 0).view(batch_size, 256, 64, 64)  # [1,256,64,64]
        end_time = time.time()

        return image_embed  # [1,256,64,64]

class MemEncoder(nn.Module):
    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__()
        self.model = sam_model
        self.maskmem_tpos_enc = sam_model.maskmem_tpos_enc
        self.feat_sizes = [(256, 256), (128, 128), (64, 64)]
    @torch.no_grad()
    def forward(
        self,
        mask_for_mem: torch.Tensor,  # [1,1,1024,1024]
        pix_feat: torch.Tensor,       # [1,256,64,64]
        occ_logit: torch.Tensor,      # [1,1]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start_time = time.time()

        batch_size = mask_for_mem.shape[0]
        pix_feat = pix_feat.repeat(batch_size, 1, 1, 1)

        maskmem_features, maskmem_pos_enc = self.model._encode_new_memory(
            current_vision_feats=pix_feat,
            feat_sizes=self.feat_sizes,
            pred_masks_high_res=mask_for_mem,
            is_mask_from_pts=True,
            object_score_logits=occ_logit,
        )
        # maskmem_features = maskmem_features.view(1, 64, 64*64) # .permute(2, 0, 1)
        maskmem_pos_enc = maskmem_pos_enc[0].view(batch_size, 64, 64*64).permute(0, 2, 1)  # .permute(2, 0, 1)


        end_time = time.time()


        return maskmem_features, maskmem_pos_enc, self.maskmem_tpos_enc

class MaskDecoder(nn.Module):
    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__()
        self.model = sam_model
        self.sigmoid_scale_for_mem_enc = sam_model.sigmoid_scale_for_mem_enc
        self.sigmoid_bias_for_mem_enc = sam_model.sigmoid_bias_for_mem_enc
    @torch.no_grad()
    def forward(
        self,
        point_coords: torch.Tensor,   # [num_labels,num_points,2]
        point_labels: torch.Tensor,   # [num_labels,num_points]
        # frame_size: torch.Tensor,   # [2]
        image_embed: torch.Tensor,    # [1,256,64,64]
        high_res_feats_0: torch.Tensor,  # [1, 32, 256, 256]
        high_res_feats_1: torch.Tensor,  # [1, 64, 128, 128]
    ):
        start_time = time.time()
        frame_size = [256, 256]
        point_inputs = {"point_coords": point_coords, "point_labels": point_labels}

        batch_size = point_coords.size()[0]
        if True:
            # image_embed = image_embed.repeat(batch_size, 1, 1, 1) image embedding can be different for each object when using memory bank
            high_res_feats_0 = high_res_feats_0.repeat(batch_size, 1, 1, 1)
            high_res_feats_1 = high_res_feats_1.repeat(batch_size, 1, 1, 1)
        high_res_feats = [high_res_feats_0, high_res_feats_1]

        sam_outputs = self.model._forward_sam_heads(
            backbone_features=image_embed,
            point_inputs=point_inputs,
            mask_inputs=None,
            high_res_features=high_res_feats,
            multimask_output=True
        )
        (
            _,
            _,
            ious,           # [1,3]
            low_res_masks,   # [1,1,256,256]
            high_res_masks,  # [1,1,1024,1024]
            obj_ptr,         # [1,256]
            occ_logit,       # [1,1]
        ) = sam_outputs
        # high resolution mask
        mask_for_mem = torch.sigmoid(high_res_masks)
        mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc
        # fill holes
        low_res_masks = fill_holes_in_mask_scores(low_res_masks, 8)
        # rescaling
        pred_mask = torch.nn.functional.interpolate(
            low_res_masks,
            size=(frame_size[0], frame_size[1]),
            mode="bilinear",
            align_corners=False,
        )
        # Pick highest IOU
        iou = torch.max(ious, dim=-1, keepdim=True)[0]


        end_time = time.time()

        return obj_ptr, mask_for_mem, pred_mask, iou, occ_logit
