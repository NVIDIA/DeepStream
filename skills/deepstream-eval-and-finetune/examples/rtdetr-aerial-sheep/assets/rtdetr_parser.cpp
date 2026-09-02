// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/* Copyright (c) 2026, NVIDIA CORPORATION.  Licensed under Apache-2.0.
 * RT-DETR nvinfer parser: logits [Q, C] (sigmoid, NO background), pred_boxes [Q,4]
 * (cx,cy,w,h normalized). Works for any C (80 stock / N fine-tuned).
 *
 * Post-processing matches HuggingFace RTDetrImageProcessor.post_process_object_detection
 * with use_focal_loss=True: global top-k over flattened (query x class) sigmoid scores,
 * NOT per-query argmax (which skews class balance and lowers mAP vs PyTorch). */
#include <vector>
#include <cmath>
#include <algorithm>
#include "nvdsinfer_custom_impl.h"

static inline float sigmoid(float x) { return 1.f / (1.f + expf(-x)); }

struct RtdetrCand {
    int query;
    int cls;
    float score;
};

static void emit_box(const float* boxes, int q, float netW, float netH,
                     int cls, float score, std::vector<NvDsInferObjectDetectionInfo>& objectList)
{
    const float* bx = boxes + (size_t) q * 4;
    float left = (bx[0] - bx[2] * 0.5f) * netW, top = (bx[1] - bx[3] * 0.5f) * netH;
    float w = bx[2] * netW, h = bx[3] * netH;
    if (left < 0.f) { w += left; left = 0.f; }
    if (top < 0.f) { h += top; top = 0.f; }
    if (left >= netW || top >= netH) return;
    w = std::min(w, netW - left); h = std::min(h, netH - top);
    if (w <= 0.f || h <= 0.f) return;
    NvDsInferObjectDetectionInfo o = {};
    o.classId = (unsigned) cls;
    o.detectionConfidence = score;
    o.left = left; o.top = top; o.width = w; o.height = h;
    objectList.push_back(o);
}

extern "C" bool NvDsInferParseCustomRtdetr(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList)
{
    const float* logits = nullptr; const float* boxes = nullptr;
    int numQueries = 0, numClasses = 0;
    for (auto const& l : outputLayersInfo) {
        unsigned int nd = l.inferDims.numDims; if (nd < 1) continue;
        unsigned int last = l.inferDims.d[nd - 1];
        if (last == 4) boxes = (const float*) l.buffer;
        else { logits = (const float*) l.buffer; numClasses = (int) last;
               numQueries = (nd >= 2) ? (int) l.inferDims.d[nd - 2] : 0; }
    }
    if (!logits || !boxes || numQueries <= 0 || numClasses <= 0) return false;
    const float netW = (float) networkInfo.width, netH = (float) networkInfo.height;

    // HF focal-loss path: topk over flattened sigmoid(logits), k = numQueries.
    std::vector<RtdetrCand> cands;
    cands.reserve((size_t) numQueries * numClasses);
    for (int q = 0; q < numQueries; ++q) {
        const float* lg = logits + (size_t) q * numClasses;
        for (int c = 0; c < numClasses; ++c)
            cands.push_back({q, c, sigmoid(lg[c])});
    }
    const int topK = numQueries;
    if ((int) cands.size() > topK)
        std::nth_element(cands.begin(), cands.begin() + topK, cands.end(),
                         [](const RtdetrCand& a, const RtdetrCand& b) { return a.score > b.score; });
    const int n = std::min(topK, (int) cands.size());
    for (int i = 0; i < n; ++i)
        emit_box(boxes, cands[i].query, netW, netH, cands[i].cls, cands[i].score, objectList);
    return true;
}
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomRtdetr);
