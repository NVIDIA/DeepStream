/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2020 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "nvdsinfer_context_impl.h"
#include <cstdio>

/* This file implements the C interface for the NvDsInferContext class. The
 * interface is a simple wrapper over the C++ interface. */

using namespace std;

#define NULL_PARAM_CHECK(param, retvalue) \
    if (param == nullptr) \
    { \
        fprintf(stderr, "Warning: NULL parameter " #param " passed to %s\n", \
                __func__); \
        return retvalue; \
    }


NvDsInferStatus
NvDsInferContext_Create(NvDsInferContextHandle *handle,
        NvDsInferContextInitParams *initParams, void *userCtx,
        NvDsInferContextLoggingFunc logFunc)
{
    NULL_PARAM_CHECK(handle, NVDSINFER_INVALID_PARAMS);
    NULL_PARAM_CHECK(initParams, NVDSINFER_INVALID_PARAMS);

    return createNvDsInferContext(handle, *initParams, userCtx, logFunc);
}

void
NvDsInferContext_Destroy(NvDsInferContextHandle handle)
{
    NULL_PARAM_CHECK(handle, );

    handle->destroy();
}

NvDsInferStatus
NvDsInferContext_QueueInputBatch(NvDsInferContextHandle handle,
        NvDsInferContextBatchInput *batchInput)
{
    NULL_PARAM_CHECK(handle, NVDSINFER_INVALID_PARAMS);
    NULL_PARAM_CHECK(batchInput, NVDSINFER_INVALID_PARAMS);

    return handle->queueInputBatch(*batchInput);
}

NvDsInferStatus
NvDsInferContext_DequeueOutputBatch(NvDsInferContextHandle handle,
        NvDsInferContextBatchOutput *batchOutput)
{
    NULL_PARAM_CHECK(handle, NVDSINFER_INVALID_PARAMS);
    NULL_PARAM_CHECK(batchOutput, NVDSINFER_INVALID_PARAMS);

    return handle->dequeueOutputBatch(*batchOutput);
}

void
NvDsInferContext_ReleaseBatchOutput(NvDsInferContextHandle handle,
        NvDsInferContextBatchOutput *batchOutput)
{
    NULL_PARAM_CHECK(handle, );
    NULL_PARAM_CHECK(batchOutput, );

    return handle->releaseBatchOutput(*batchOutput);
}

unsigned int
NvDsInferContext_GetNumLayersInfo(NvDsInferContextHandle handle)
{
    NULL_PARAM_CHECK(handle, 0);

    std::vector<NvDsInferLayerInfo> layersInfo;
    handle->fillLayersInfo(layersInfo);

    return layersInfo.size();
}

void
NvDsInferContext_FillLayersInfo(NvDsInferContextHandle handle,
        NvDsInferLayerInfo *layersInfo)
{
    NULL_PARAM_CHECK(handle, );
    NULL_PARAM_CHECK(layersInfo, );

    std::vector<NvDsInferLayerInfo> layersInfoVec;
    handle->fillLayersInfo(layersInfoVec);
    for (unsigned int i = 0; i < layersInfoVec.size(); i++)
        layersInfo[i] = layersInfoVec[i];
}

void
NvDsInferContext_GetNetworkInfo(NvDsInferContextHandle handle,
        NvDsInferNetworkInfo *networkInfo)
{
    NULL_PARAM_CHECK(handle, );
    NULL_PARAM_CHECK(networkInfo, );

    return handle->getNetworkInfo(*networkInfo);
}

const char*
NvDsInferContext_GetLabel(NvDsInferContextHandle handle, unsigned int id,
        unsigned int value)
{
    NULL_PARAM_CHECK(handle, nullptr);

    auto labels = handle->getLabels();
    if (labels.size() > id && labels[id].size() > value)
        return labels[id][value].c_str();

    return nullptr;
}
