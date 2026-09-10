# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


from pathlib import Path

import open_clip
import torch
from PIL import Image
import os

# Register the model config
open_clip.add_model_config("configs")

MODEL = os.environ.get("MODEL_NAME", "NVCLIP_224_700M_ViTH14")
model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL,
    pretrained=f"/workspace/checkpoints/baseline/{os.environ.get("CHECKPOINT_NAME")}",
)
tokenizer = open_clip.get_tokenizer(MODEL)

text = tokenizer(["a diagram", "a dog", "a cat"])
image = preprocess(Image.open("CLIP.png")).unsqueeze(0)

# Export to onnx
npx = 224
dummy_image = torch.randn(10, 3, npx, npx)
model.forward(dummy_image,text) # Original CLIP result (1)

# Vision model
torch.onnx.export(model, (dummy_image, None),
  f"/workspace/checkpoints/baseline/nvclip_clipa_vit_h14_700M_vision.onnx",
  export_params=True,
  input_names=["IMAGE"],
  output_names=["LOGITS_PER_IMAGE"],
  opset_version=19,
  dynamic_axes={
      "IMAGE": {
          0: "image_batch_size",
      },
      "LOGITS_PER_IMAGE": {
          0: "image_batch_size",
          1: "text_batch_size",
      },
  },
  verbose=True,
  dynamo=False
)


# Text model
torch.onnx.export(model, (None, text),
  f"/workspace/checkpoints/baseline/nvclip_clipa_vit_h14_700M_text.onnx",
  export_params=True,
  input_names=["TEXT"],
  output_names=["LOGITS_PER_TEXT"],
  opset_version=19,
  dynamic_axes={
      "TEXT": {
          0: "text_batch_size",
      },
      "LOGITS_PER_TEXT": {
          0: "text_batch_size",
          1: "image_batch_size",
      },
  },
  verbose=True,
  dynamo=False
)
