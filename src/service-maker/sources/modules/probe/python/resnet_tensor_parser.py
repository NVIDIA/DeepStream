# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Python equivalent of the C++ resnet_tensor_parser plugin.

Parses the raw output tensors from the ResNet10 detection model
(output_bbox/BiasAdd:0 and output_cov/Sigmoid:0), runs DBSCAN clustering
and creates ObjectMetadata for downstream OSD / analytics use.

Importing this module registers "resnet_tensor_parser" in CommonFactory:

    pipeline.attach("pgie", "resnet_tensor_parser", "tensor_parser")

Supported properties (mirror C++ probe):
  network-width   : model input width  (default 960)
  network-height  : model input height (default 544)
  stream-width    : source stream width  (default 1280)
  stream-height   : source stream height (default 720)
  num-classes     : number of classes (default 4)
"""

import math
import numpy as np
import cupy as cp

from pyservicemaker import probe, BatchMetadataOperator
from pyservicemaker._pydeepstream import osd as _osd
from pyservicemaker._pydeepstream import TensorOutputUserMetadata

PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_PERSON  = 2

# Layer names in the ResNet10 model
BBOX_LAYER_NAME = "output_bbox/BiasAdd:0"
COV_LAYER_NAME  = "output_cov/Sigmoid:0"

# Default network / stream parameters (match C++ probe defaults)
DEFAULT_NETWORK_WIDTH  = 960
DEFAULT_NETWORK_HEIGHT = 544
DEFAULT_STREAM_WIDTH   = 1280
DEFAULT_STREAM_HEIGHT  = 720
DEFAULT_NUM_CLASSES    = 4

# Per-class precluster confidence thresholds (match C++ probe)
PRECLUSTER_THRESHOLD = [0.2, 0.2, 0.2, 0.2]

# DBSCAN parameters (match C++ probe)
DBSCAN_EPS       = 0.95
DBSCAN_MIN_BOXES = 3
DBSCAN_MIN_SCORE = 0.5
BBOX_NORM_X      = 35.0
BBOX_NORM_Y      = 35.0


def _iou(a, b):
    """Compute IoU between two boxes [left, top, width, height]."""
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(ax2, bx2),   min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a[2]*a[3] + b[2]*b[3] - inter
    return inter / union if union > 0 else 0.0


def _dbscan_cluster(boxes, eps=DBSCAN_EPS, min_boxes=DBSCAN_MIN_BOXES):
    """
    IoU-based DBSCAN cluster matching C++ NvDsInferDBScanCluster.
    boxes: list of [left, top, width, height, confidence]
    Returns: list of merged boxes [left, top, width, height, confidence]
    """
    if not boxes:
        return []

    n = len(boxes)
    visited  = [False] * n
    clusters = []

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        # Find all neighbours with IoU > eps
        neighbours = [i]
        for j in range(n):
            if not visited[j] and _iou(boxes[i], boxes[j]) >= eps:
                neighbours.append(j)
                visited[j] = True

        if len(neighbours) < min_boxes:
            continue

        # Merge cluster: weighted average by confidence
        total_w  = sum(boxes[k][4] for k in neighbours)
        if total_w == 0:
            continue
        ml = sum(boxes[k][0] * boxes[k][4] for k in neighbours) / total_w
        mt = sum(boxes[k][1] * boxes[k][4] for k in neighbours) / total_w
        mw = sum(boxes[k][2] * boxes[k][4] for k in neighbours) / total_w
        mh = sum(boxes[k][3] * boxes[k][4] for k in neighbours) / total_w
        mc = max(boxes[k][4] for k in neighbours)
        clusters.append([ml, mt, mw, mh, mc])

    return clusters


def _parse_resnet_output(layers, network_width, network_height, num_classes,
                         precluster_thresh=PRECLUSTER_THRESHOLD):
    """
    Python equivalent of NvDsInferParseCustomResnet().
    layers: dict[layer_name -> numpy float32 array (already on host)]
    Returns: list of per-class lists: [{left,top,width,height,confidence}, ...]
    """
    if BBOX_LAYER_NAME not in layers or COV_LAYER_NAME not in layers:
        return [[] for _ in range(num_classes)]

    bbox_buf = np.array(layers[BBOX_LAYER_NAME], dtype=np.float32)
    cov_buf  = np.array(layers[COV_LAYER_NAME],  dtype=np.float32)

    # cov shape: [num_classes, gridH, gridW]
    # bbox shape: [num_classes * 4, gridH, gridW]
    num_c   = cov_buf.shape[0]
    grid_h  = cov_buf.shape[1]
    grid_w  = cov_buf.shape[2]

    stride_x = math.ceil(network_width  / grid_w)
    stride_y = math.ceil(network_height / grid_h)

    # Grid cell centres (normalised by bboxNorm)
    gc_x = (np.arange(grid_w) * stride_x + 0.5) / BBOX_NORM_X   # shape [gridW]
    gc_y = (np.arange(grid_h) * stride_y + 0.5) / BBOX_NORM_Y   # shape [gridH]

    num_to_parse = min(num_c, num_classes)
    per_class    = [[] for _ in range(num_classes)]

    for c in range(num_to_parse):
        thresh  = precluster_thresh[c] if c < len(precluster_thresh) else 0.2
        cov     = cov_buf[c]                           # [gridH, gridW]
        bbox_c  = bbox_buf[c * 4: c * 4 + 4]          # [4, gridH, gridW]

        mask = cov >= thresh
        hs, ws = np.where(mask)

        for h, w in zip(hs, ws):
            gx = gc_x[w]
            gy = gc_y[h]

            x1 = (bbox_c[0, h, w] - gx) * -BBOX_NORM_X
            y1 = (bbox_c[1, h, w] - gy) * -BBOX_NORM_Y
            x2 = (bbox_c[2, h, w] + gx) *  BBOX_NORM_X
            y2 = (bbox_c[3, h, w] + gy) *  BBOX_NORM_Y

            left   = float(np.clip(x1, 0, network_width  - 1))
            top    = float(np.clip(y1, 0, network_height - 1))
            right  = float(np.clip(x2, 0, network_width  - 1))
            bottom = float(np.clip(y2, 0, network_height - 1))

            width  = right  - left + 1
            height = bottom - top  + 1

            per_class[c].append([left, top, width, height, float(cov[h, w])])

    return per_class


@probe(
    name="resnet_tensor_parser",
    params={
        "network-width":  ("integer", DEFAULT_NETWORK_WIDTH,  "model input width"),
        "network-height": ("integer", DEFAULT_NETWORK_HEIGHT, "model input height"),
        "stream-width":   ("integer", DEFAULT_STREAM_WIDTH,   "source stream width"),
        "stream-height":  ("integer", DEFAULT_STREAM_HEIGHT,  "source stream height"),
        "num-classes":    ("integer", DEFAULT_NUM_CLASSES,    "number of detection classes"),
    },
)
class TensorMetaParser(BatchMetadataOperator):

    def __init__(self):
        super().__init__()
        self._network_width  = DEFAULT_NETWORK_WIDTH
        self._network_height = DEFAULT_NETWORK_HEIGHT
        self._stream_width   = DEFAULT_STREAM_WIDTH
        self._stream_height  = DEFAULT_STREAM_HEIGHT
        self._num_classes    = DEFAULT_NUM_CLASSES
        self._params_loaded  = False

    def handle_metadata(self, batch_meta):
        if not self._params_loaded:
            nw = self.get_property("network-width")
            nh = self.get_property("network-height")
            sw = self.get_property("stream-width")
            sh = self.get_property("stream-height")
            nc = self.get_property("num-classes")
            if nw: self._network_width  = int(nw)
            if nh: self._network_height = int(nh)
            if sw: self._stream_width   = int(sw)
            if sh: self._stream_height  = int(sh)
            if nc: self._num_classes    = int(nc)
            self._params_loaded = True
        scale_x = self._stream_width  / self._network_width
        scale_y = self._stream_height / self._network_height

        for frame_meta in batch_meta.frame_items:
            for user_meta in frame_meta.tensor_items:
                tensor_meta = user_meta.as_tensor_output()
                layers = {name: cp.from_dlpack(t).get() for name, t in tensor_meta.get_layers().items()}

                per_class = _parse_resnet_output(
                    layers, self._network_width, self._network_height, self._num_classes
                )

                vehicle_count = 0
                person_count  = 0

                for c, boxes in enumerate(per_class):
                    clustered = _dbscan_cluster(boxes)
                    for box in clustered:
                        left, top, width, height, conf = box
                        obj = batch_meta.acquire_object_meta()
                        if not obj:
                            continue
                        obj.class_id   = c
                        obj.confidence = conf
                        r = _osd.Rect()
                        r.left        = left   * scale_x
                        r.top         = top    * scale_y
                        r.width       = width  * scale_x
                        r.height      = height * scale_y
                        r.border_width = 1
                        r.border_color = _osd.Color(1.0, 0.0, 0.0, 1.0)
                        obj.rect_params = r
                        frame_meta.append(obj)

                        if c == PGIE_CLASS_ID_VEHICLE:
                            vehicle_count += 1
                        elif c == PGIE_CLASS_ID_PERSON:
                            person_count += 1

                print(
                    f"Object Counter:  Pad Idx = {frame_meta.pad_index}"
                    f"  Frame Number = {frame_meta.frame_number}"
                    f"  Vehicle Count = {vehicle_count}"
                    f"  Person Count = {person_count}"
                )
