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

# Reorder annotations_test.json so images are interleaved across the 6 defect classes
# (the source split is sorted by class; capping at first-N would otherwise cover only 1-2).
# Images already on disk; only the image list order in the json changes.
import json, collections
p = "data/pcb_defect/annotations_test.json"
c = json.load(open(p))
by_img = collections.defaultdict(list)
for a in c["annotations"]:
    by_img[a["image_id"]].append(a["category_id"])

def domcat(im):
    cats = by_img.get(im["id"], [])
    return collections.Counter(cats).most_common(1)[0][0] if cats else -1

groups = collections.defaultdict(list)
for im in c["images"]:
    groups[domcat(im)].append(im)
print("per-class test images:", {k: len(v) for k, v in sorted(groups.items())})
lists = [groups[k] for k in sorted(groups) if k >= 0]
order, i = [], 0
while any(i < len(l) for l in lists):
    for l in lists:
        if i < len(l):
            order.append(l[i])
    i += 1
c["images"] = order
json.dump(c, open(p, "w"))
cov = collections.Counter(domcat(im) for im in order[:200])
print("first-200 class coverage:", dict(sorted(cov.items())), "total:", len(order))
