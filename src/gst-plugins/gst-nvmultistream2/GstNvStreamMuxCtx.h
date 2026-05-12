/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

/**
 * @file GstNvStreamMuxCtx.h
 * @brief  StreamMux heler context class
 */


#ifndef _GST_NVSTREAMMUXCTX_H_
#define _GST_NVSTREAMMUXCTX_H_

#include "nvbufaudio.h"
#include <unordered_map>
#include <mutex>


class GstNvStreamMuxCtx
{
    public:
    GstNvStreamMuxCtx();
    void SaveAudioParams(uint32_t padId, uint32_t sourceId, NvBufAudioParams audioParams);
    NvBufAudioParams GetAudioParams(uint32_t padId);
    void SetMemTypeNVMM(uint32_t padId, bool isNVMM);
    bool IsMemTypeNVMM(uint32_t padId);

    private:
    std::mutex              mutex;
    std::unordered_map<uint32_t, NvBufAudioParams> audioParamsMap;
    std::unordered_map<uint32_t, bool> isNVMMMap;
};

#endif /**< _GST_NVSTREAMMUXCTX_H_ */

