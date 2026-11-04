/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>

#include "modules/NvMultiObjectTracker/nvdstracker.h"

#include "modules/NvMultiObjectTracker/context.hpp"
#include "modules/NvMultiObjectTracker/NvTrackerParams.hpp"

using namespace std;

/**
 * @brief Query tracker library capabilities and requirements.
 * @param[in] customConfigFilePathSize Length of the custom configuration file's pathname.
 * @param[in] pCustomConfigFilePath A pointer to the custom configuration file's pathname.
 * @param[out] pQuery A pointer to a query structure to be filled by the tracker.
 * @return Status of the query.
 */
NvMOTStatus NvMOT_Query(uint16_t customConfigFilePathSize, char* pCustomConfigFilePath, NvMOTQuery *pQuery)
{
    if (pQuery == NULL)
    {
        cout << "NvMOT_Query failed. NvMOTQuery pointer is NULL" << endl;
        return NvMOTStatus_Error;
    }

    pQuery->computeConfig = NVMOTCOMP_GPU;
    pQuery->numTransforms = 0;
    pQuery->supportPastFrame = true;
    pQuery->batchMode = NvMOTBatchMode_Batch;

    /// Preserve preprocessed RGBA buffers and otherwise request NV12 input.
    if (pQuery->colorFormats[0] != NVBUF_COLOR_FORMAT_NV12 &&
        pQuery->colorFormats[0] != NVBUF_COLOR_FORMAT_RGBA)
    {
        pQuery->colorFormats[0] = NVBUF_COLOR_FORMAT_NV12;
    }

    #ifdef __aarch64__
        pQuery->memType = NVBUF_MEM_DEFAULT;
    #else
        /// vGPU does not support unified memory.
        pQuery->memType = NVBUF_MEM_CUDA_DEVICE;
    #endif

    if (pQuery->contextHandle)
    {
        pQuery->contextHandle->queryParams(*pQuery);
        return NvMOTStatus_OK;
    }
    else
    {
        cout << "NvMOT_Query failed. Context handle is NULL." << endl;
        return NvMOTStatus_Error;
    }
}

/**
 * @brief Initialize a tracking context for a batch of image streams.
 * @param[in] pConfigIn A pointer to a structure specifying the configuration.
 * @param[in,out] pContextHandle A pointer whose existing handle is retired and whose replacement handle is stored for later calls.
 * @param[out] pConfigResponse A pointer to a structure that describes the operation's status.
 * @return The outcome of the initialization attempt.
 */
NvMOTStatus NvMOT_Init(NvMOTConfig *pConfigIn, NvMOTContextHandle *pContextHandle, NvMOTConfigResponse *pConfigResponse)
{
    if(pContextHandle != nullptr)
    {
        NvMOT_DeInit(*pContextHandle);
    }

    NvMOTContext *pContext = nullptr;

    try
    {
        pContext = new NvMOTContext(*pConfigIn, *pConfigResponse);
    }
    catch (exception& e)
    {
        cout << "An exception occurred. " << e.what() << endl;
        if (nullptr != pContext)
            delete pContext;

        return NvMOTStatus_Error;
    }
    catch (...)
    {
        if (nullptr != pContext)
            delete pContext;

        return NvMOTStatus_Error;
    }

    *pContextHandle = pContext;

    return NvMOTStatus_OK;
}

/**
 * @brief Deinitialize a stream context.
 * @param[in] contextHandle The handle for the stream context to retire.
 */
void NvMOT_DeInit(NvMOTContextHandle contextHandle)
{
    try
    {
        delete contextHandle;
    }
    catch (...)
    {
    }
}

/**
 * @brief Process a batch of frames.
 * @param[in] contextHandle A context handle obtained from NvMOT_Init().
 * @param[in] pParams A pointer to parameters for the batch to be processed.
 * @param[out] pTrackedObjectsBatch A pointer to client-allocated tracked-object slots that receive bounding boxes scaled to the first input transform.
 * @return The status of batch processing.
 */
NvMOTStatus NvMOT_Process(
    NvMOTContextHandle contextHandle, NvMOTProcessParams *pParams, NvMOTTrackedObjBatch *pTrackedObjectsBatch
)
{
    try
    {
        if (contextHandle == nullptr)
            throw runtime_error("Invalid context!");

        contextHandle->processFrame(pParams, pTrackedObjectsBatch);
    }
    catch(exception& e)
    {
        cout << "An exception occurred. " << e.what() << endl;
        return NvMOTStatus_Error;
    }
    catch (...)
    {
        return NvMOTStatus_Error;
    }

    return NvMOTStatus_OK;
}

/**
 * @brief Retrieve past-frame, terminated-track, and shadow-track data from the low-level tracker.
 * @param[in] contextHandle The context handle obtained from NvMOT_Init().
 * @param[in] pParams A pointer to parameters for the batch of frames with the available stream ID.
 * @param[out] pTrackerMiscData Miscellaneous data from the low-level tracker to store in user metadata.
 * @return The status of the retrieval.
 */
NvMOTStatus NvMOT_RetrieveMiscData(
    NvMOTContextHandle contextHandle, NvMOTProcessParams *pParams, NvMOTTrackerMiscData *pTrackerMiscData
)
{
    try
    {
        if (contextHandle == nullptr)
            throw runtime_error("Invalid context!");

        contextHandle->retrieveMiscData(pParams, pTrackerMiscData);
    }
    catch(exception& e)
    {
        cout << "An exception occurred. " << e.what() << endl;
        return NvMOTStatus_Error;
    }
    catch (...)
    {
        return NvMOTStatus_Error;
    }

    return NvMOTStatus_OK;
}

/**
 * @brief Remove streams selected by an ID mask from a batch.
 * @param[in] contextHandle The context handle obtained from NvMOTInit().
 * @param[in] streamIdMask A mask that selects streams satisfying (streamId & streamIdMask) == streamIdMask.
 * @return Status of stream removal.
 */
NvMOTStatus NvMOT_RemoveStreams(NvMOTContextHandle contextHandle, NvMOTStreamId streamIdMask)
{
    try
    {
        if (contextHandle == nullptr)
            throw runtime_error("Invalid context!");
        contextHandle->removeStream(streamIdMask);
    }
    catch(exception& e)
    {
        cout << "An exception occurred. " << e.what() << endl;
        return NvMOTStatus_Error;
    }
    catch (...)
    {
        return NvMOTStatus_Error;
    }
    return NvMOTStatus_OK;
}

/**
 * @brief Update dynamic parameters for the low-level tracker.
 * @param[in] contextHandle The context handle obtained from NvMOTInit().
 * @param[in] configStr Stringified configuration for the tracker.
 * @return Status of the configuration update.
 */
NvMOTStatus NvMOT_UpdateParams(NvMOTContextHandle contextHandle, const std::string &configStr)
{
    try
    {
        if (contextHandle == nullptr)
            throw runtime_error("Invalid context!");
        contextHandle->updateParams(configStr);
    }
    catch(exception& e)
    {
        cout << "An exception occurred. " << e.what() << endl;
        return NvMOTStatus_Error;
    }
    catch (...)
    {
        return NvMOTStatus_Error;
    }
    return NvMOTStatus_OK;
}
