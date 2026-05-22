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

#ifndef __NVDSINFER_LOGGER_H__
#define __NVDSINFER_LOGGER_H__

#include <memory>
#include <mutex>
#include <string>
#include <array>

#include <NvInfer.h>
#include <nvdsinfer.h>

#if defined(NDEBUG)
#define INFER_LOG_FORMAT_(fmt) fmt
#else
#define INFER_LOG_FORMAT_(fmt) "%s:%d " fmt, __FILE__, __LINE__
#endif

#define dsInferError(fmt, ...)                                           \
    do {                                                                 \
        dsInferLogPrint__(                                               \
            NVDSINFER_LOG_ERROR, INFER_LOG_FORMAT_(fmt), ##__VA_ARGS__); \
    } while (0)

#define dsInferWarning(fmt, ...)                                           \
    do {                                                                   \
        dsInferLogPrint__(                                                 \
            NVDSINFER_LOG_WARNING, INFER_LOG_FORMAT_(fmt), ##__VA_ARGS__); \
    } while (0)

#define dsInferInfo(fmt, ...)                                           \
    do {                                                                \
        dsInferLogPrint__(                                              \
            NVDSINFER_LOG_INFO, INFER_LOG_FORMAT_(fmt), ##__VA_ARGS__); \
    } while (0)

#define dsInferDebug(fmt, ...)                                           \
    do {                                                                 \
        dsInferLogPrint__(                                               \
            NVDSINFER_LOG_DEBUG, INFER_LOG_FORMAT_(fmt), ##__VA_ARGS__); \
    } while (0)

namespace nvdsinfer {
void dsInferLogPrint__(NvDsInferLogLevel level, const char* fmt, ...);
}

extern std::unique_ptr<nvinfer1::ILogger> gTrtLogger;
#endif
