/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

