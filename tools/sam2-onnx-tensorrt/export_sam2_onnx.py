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
import onnx
import argparse
from src.Module import ImageEncoder
from src.Module import MemAttention
from src.Module import MemEncoder
from src.Module import MaskDecoder
from sam2.build_sam import build_sam2


def export_image_encoder(model,onnx_path):
    print(">>> Exporting Image Encoder...")
    input_img = torch.randn(1, 3,1024, 1024).cpu()
    out = model(input_img)
    output_names = ["pix_feat","high_res_feat0","high_res_feat1","vision_feats","vision_pos_embed"]
    torch.onnx.export(
        model,
        input_img,
        onnx_path+"image_encoder.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["image"],
        output_names=output_names,
    )
    onnx_model = onnx.load(onnx_path+"image_encoder.onnx")
    onnx.checker.check_model(onnx_model)
    print("[SUCCESS] Image Encoder exported successfully!")


def export_memory_attention(model,onnx_path):
    print(">>> Exporting Memory Attention...")
    batch_size = 1
    current_vision_feat = torch.randn(1,256,64,64)      #[1, 256, 64, 64]
    current_vision_pos_embed = torch.randn(4096,1,256)  #[4096, 1, 256]
    memory_0 = torch.randn(batch_size,16,256) # [batch size, num obj ptr, feature size]
    memory_1 = torch.randn(batch_size,7,64,64,64)
    memory_pos_embed = torch.randn(batch_size,7*4096+64,64)      #[y*4096,1,64]
    cond_frame_id_diff = torch.tensor(10.0)
    out = model(
            current_vision_feat = current_vision_feat,
            current_vision_pos_embed = current_vision_pos_embed,
            memory_0 = memory_0,
            memory_1 = memory_1,
            memory_pos_embed = memory_pos_embed,
            cond_frame_id_diff = cond_frame_id_diff,
        )
    input_name = ["current_vision_feat",
                "current_vision_pos_embed",
                "memory_0",
                "memory_1",
                "memory_pos_embed",
                "cond_frame_id_diff",]
    dynamic_axes = {
        "memory_0": {0: "batch_size", 1: "num"},
        "memory_1": {0: "batch_size", 1: "buff_size"},
        "memory_pos_embed": {0: "batch_size", 1: "buff_size_embed"}
    }
    torch.onnx.export(
        model,
        (current_vision_feat,current_vision_pos_embed,memory_0,memory_1,memory_pos_embed,cond_frame_id_diff),
        onnx_path+"memory_attention.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names= input_name,
        output_names=["image_embed"],
        dynamic_axes = dynamic_axes
    )
    # original_model = onnx.load(onnx_path+"memory_attention.onnx")
    # simplified_model, check = simplify(original_model)
    # onnx.save(simplified_model, onnx_path+"memory_attention.onnx")
    onnx_model = onnx.load(onnx_path+"memory_attention.onnx")
    onnx.checker.check_model(onnx_model)
    print("[SUCCESS] Memory Attention exported successfully!")


def export_mask_decoder(model,onnx_path):
    print(">>> Exporting Mask Decoder...")
    batch_size = 20
    point_coords = torch.randn(batch_size,2,2).cpu()
    point_labels = torch.randn(batch_size,2).cpu()
    # point_coords = torch.randn(1,2,2).cpu()
    # point_labels = torch.randn(1,2).cpu()
    # frame_size = torch.tensor([1024,1024],dtype=torch.int64)
    image_embed = torch.randn(batch_size,256,64,64).cpu()
    high_res_feats_0 = torch.randn(1,32,256,256).cpu()
    high_res_feats_1 = torch.randn(1,64,128,128).cpu()

    out = model(
        point_coords = point_coords,
        point_labels = point_labels,
    #    frame_size = frame_size,
        image_embed = image_embed,
        high_res_feats_0 = high_res_feats_0,
        high_res_feats_1 = high_res_feats_1
    )
    # input_name = ["point_coords","point_labels","frame_size","image_embed","high_res_feats_0","high_res_feats_1"]
    input_name = ["point_coords","point_labels","image_embed","high_res_feats_0","high_res_feats_1"]
    output_name = ["obj_ptr","mask_for_mem","pred_mask", "iou", "occ_logit"]
    dynamic_axes = {
        "point_coords":{0: "batch_size",1:"num_points"},
        "point_labels": {0: "batch_size",1:"num_points"},
        "image_embed": {0: "batch_size"},
        # "obj_ptr": {0: "batch_size"},
        # "mask_for_mem": {0: "batch_size"},
        # "pred_mask": {0: "batch_size"}
    }
    torch.onnx.export(
        model,
    #    (point_coords,point_labels,frame_size,image_embed,high_res_feats_0,high_res_feats_1),
        (point_coords,point_labels,image_embed,high_res_feats_0,high_res_feats_1),
        onnx_path+"mask_decoder.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names= input_name,
        output_names=output_name,
        dynamic_axes = dynamic_axes
    )
    onnx_model = onnx.load(onnx_path+"mask_decoder.onnx")
    onnx.checker.check_model(onnx_model)
    print("[SUCCESS] Mask Decoder exported successfully!")


def export_memory_encoder(model,onnx_path):
    print(">>> Exporting Memory Encoder...")
    batch_size = 1
    mask_for_mem = torch.randn(batch_size,1,1024,1024)
    pix_feat = torch.randn(1,256,64,64)
    occ_logit = torch.randn(1,1)
    dynamic_axes = {
        "mask_for_mem":{0: "batch_size"},
        "occ_logit":{0: "batch_size"}
    }

    out = model(mask_for_mem = mask_for_mem,pix_feat = pix_feat,occ_logit = occ_logit)

    input_names = ["mask_for_mem","pix_feat","occ_logit"]
    output_names = ["maskmem_features","maskmem_pos_enc","temporal_code"]
    torch.onnx.export(
        model,
        (mask_for_mem,pix_feat,occ_logit),
        onnx_path+"memory_encoder.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names= input_names,
        output_names= output_names,
        dynamic_axes = dynamic_axes
    )
    onnx_model = onnx.load(onnx_path+"memory_encoder.onnx")
    onnx.checker.check_model(onnx_model)
    print("[SUCCESS] Memory Encoder exported successfully!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="export SAM2.1 to onnx")
    parser.add_argument("--model",type=str,choices=["tiny", "small", "base_plus", "large"],
        default="tiny",required=False,help="SAM2 model type. Choose one of: tiny, small, base_plus, large")
    args = parser.parse_args()

    config_suffix_dict = {
        "tiny": "t",
        "small": "s",
        "base_plus": "b+",
        "large": "l"
    }

    model_type = args.model
    outdir = "checkpoints/{}/".format(model_type)
    config = "configs/sam2.1/sam2.1_hiera_{}.yaml".format(config_suffix_dict[model_type])
    checkpoint = "checkpoints/sam2.1_hiera_{}.pt".format(model_type)

    sam2_model = build_sam2(config, checkpoint, device="cpu")

    image_encoder = ImageEncoder(sam2_model).cpu()
    export_image_encoder(image_encoder,outdir)

    mask_decoder = MaskDecoder(sam2_model).cpu()
    export_mask_decoder(mask_decoder,outdir)

    mem_attention = MemAttention(sam2_model).cpu()
    export_memory_attention(mem_attention,outdir)

    mem_encoder   = MemEncoder(sam2_model).cpu()
    export_memory_encoder(mem_encoder,outdir)
