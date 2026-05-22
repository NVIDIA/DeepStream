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

#ifndef __NVDSINFER_SERVER_PROTO_UTILS_H__
#define __NVDSINFER_SERVER_PROTO_UTILS_H__

#include <dlfcn.h>
#include <stdarg.h>
#include <string.h>
#include <unistd.h>
#include <cassert>
#include <condition_variable>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>

#include "nvbufsurftransform.h"
#include <infer_datatypes.h>

#pragma GCC diagnostic push
#if __GNUC__ >= 8
#pragma GCC diagnostic ignored "-Wrestrict"
#endif
#include "nvdsinferserver_config.pb.h"
#include "nvdsinferserver_plugin.pb.h"
#pragma GCC diagnostic pop

namespace ic = nvdsinferserver::config;

namespace nvdsinferserver {

InferDataType dataTypeFromDsProto(ic::TensorDataType dt);

InferTensorOrder tensorOrderFromDsProto(ic::TensorOrder o);
InferMediaFormat mediaFormatFromDsProto(ic::MediaFormat f);
InferMemType memTypeFromDsProto(ic::MemoryType t);

NvBufSurfTransform_Compute computeHWFromDsProto(ic::FrameScalingHW h);
NvBufSurfTransform_Inter scalingFilterFromDsProto(uint32_t filter);

bool validateProtoConfig(ic::InferenceConfig& c, const std::string& path);
bool compareModelRepo(
    const ic::TritonModelRepo& repoA, const ic::TritonModelRepo& repoB);

inline bool
hasTriton(const ic::BackendParams& params)
{
    return params.has_triton() || params.has_trt_is();
}
inline const ic::TritonParams&
getTritonParam(const ic::BackendParams& params)
{
    if (params.has_triton()) {
        return params.triton();
    } else {
        assert(params.has_trt_is());
        return params.trt_is();
    }
}
inline ic::TritonParams*
mutableTriton(ic::BackendParams& params)
{
    if (params.has_triton()) {
        return params.mutable_triton();
    } else {
        assert(params.has_trt_is());
        return params.mutable_trt_is();
    }
}

} // namespace nvdsinferserver

#endif
