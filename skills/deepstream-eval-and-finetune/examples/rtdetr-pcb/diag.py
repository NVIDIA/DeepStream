# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from PIL import Image
from transformers import AutoModelForObjectDetection, AutoImageProcessor
ck = "runs/rtdetr_pcb/checkpoints/final"
# Local checkpoint path, so this is inert here — set explicitly anyway so the Hub is never
# consulted at a floating revision if `ck` is repointed at a Hub id.
REVISION = "main"
m = AutoModelForObjectDetection.from_pretrained(ck, revision=REVISION).eval().cuda()
print("num_labels:", m.config.num_labels, "id2label:", m.config.id2label)
ip = AutoImageProcessor.from_pretrained(ck, revision=REVISION, do_resize=True, size={"height": 640, "width": 640})
img = Image.open("models/rtdetr_pcb_ft/eval/eval_set/images/000000.jpg").convert("RGB")
enc = ip(images=img, return_tensors="pt").to("cuda")
with torch.no_grad():
    out = m(**enc)
logits = out.logits[0]
probs = logits.sigmoid()
print("raw logits: min%.3f max%.3f mean%.3f" % (logits.min(), logits.max(), logits.mean()))
print("sigmoid max %.4f ; top-5:" % probs.max().item(),
      [round(t, 4) for t in probs.flatten().topk(5).values.tolist()])
res = ip.post_process_object_detection(out, threshold=0.0, target_sizes=torch.tensor([[640, 640]]).cuda())[0]
sc = res["scores"]
print("post_process top scores:", [round(x, 4) for x in sc.topk(min(5, len(sc))).values.tolist()])
