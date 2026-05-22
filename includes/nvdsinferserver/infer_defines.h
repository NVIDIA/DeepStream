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

#ifndef __NVDSINFERSERVER_DEFINES_H__
#define __NVDSINFERSERVER_DEFINES_H__

#include <stdarg.h>
#include <cassert>
#include <condition_variable>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <queue>

#define DISABLE_CLASS_COPY(NoCopyClass)       \
    NoCopyClass(const NoCopyClass&) = delete; \
    void operator=(const NoCopyClass&) = delete

#define SIMPLE_MOVE_COPY(Cls)    \
    Cls& operator=(Cls&& o) {    \
        move_copy(std::move(o)); \
        return *this;            \
    }                            \
    Cls(Cls&& o) { move_copy(std::move(o)); }

#define INFER_UNUSED(a) (void)(a)

#if defined(NDEBUG)
#define INFER_LOG_FORMAT_(fmt) fmt
#else
#define INFER_LOG_FORMAT_(fmt) "%s:%d " fmt, __FILE__, __LINE__
#endif

#define INFER_EXPORT_API __attribute__((__visibility__("default")))

#define InferError(fmt, ...)                                             \
    do {                                                                 \
        dsInferLogPrint__(                                               \
            NVDSINFER_LOG_ERROR, INFER_LOG_FORMAT_(fmt), ##__VA_ARGS__); \
    } while (0)

#define InferWarning(fmt, ...)                                             \
    do {                                                                   \
        dsInferLogPrint__(                                                 \
            NVDSINFER_LOG_WARNING, INFER_LOG_FORMAT_(fmt), ##__VA_ARGS__); \
    } while (0)

#define InferInfo(fmt, ...)                                             \
    do {                                                                \
        dsInferLogPrint__(                                              \
            NVDSINFER_LOG_INFO, INFER_LOG_FORMAT_(fmt), ##__VA_ARGS__); \
    } while (0)

#define InferDebug(fmt, ...)                                             \
    do {                                                                 \
        dsInferLogPrint__(                                               \
            NVDSINFER_LOG_DEBUG, INFER_LOG_FORMAT_(fmt), ##__VA_ARGS__); \
    } while (0)

#define RETURN_IF_FAILED(condition, ret, fmt, ...) \
    do {                                           \
        if (!(condition)) {                        \
            InferError(fmt, ##__VA_ARGS__);        \
            return ret;                            \
        }                                          \
    } while (0)

#define CHECK_NVINFER_ERROR_PRINT(err, action, logPrint, fmt, ...)     \
    do {                                                               \
        NvDsInferStatus ifStatus = (err);                              \
        if (ifStatus != NVDSINFER_SUCCESS) {                           \
            auto errStr = NvDsInferStatus2Str(ifStatus);               \
            logPrint(fmt ", nvinfer error:%s", ##__VA_ARGS__, errStr); \
            action;                                                    \
        }                                                              \
    } while (0)

#define CHECK_NVINFER_ERROR(err, action, fmt, ...) \
    CHECK_NVINFER_ERROR_PRINT(err, action, InferError, fmt, ##__VA_ARGS__)

#define RETURN_NVINFER_ERROR(err, fmt, ...) \
    CHECK_NVINFER_ERROR(err, return ifStatus, fmt, ##__VA_ARGS__)

#define CONTINUE_NVINFER_ERROR(err, fmt, ...) \
    CHECK_NVINFER_ERROR(err, , fmt, ##__VA_ARGS__)


#define CHECK_CUDA_ERR_W_ACTION(err, action, logPrint, fmt, ...)                    \
    do {                                                                  \
        cudaError_t errnum = (err);                                       \
        if (errnum != cudaSuccess) {                                      \
            logPrint(fmt ", cuda err_no:%d, err_str:%s", ##__VA_ARGS__, \
                (int)errnum, cudaGetErrorName(errnum));                   \
            action;                                                       \
        }                                                                 \
    } while (0)

#define CHECK_CUDA_ERR_NO_ACTION(err, fmt, ...) \
    CHECK_CUDA_ERR_W_ACTION(err, , InferError, fmt, ##__VA_ARGS__)

#define RETURN_CUDA_ERR(err, fmt, ...) \
    CHECK_CUDA_ERR_W_ACTION(           \
        err, return NVDSINFER_CUDA_ERROR, InferError, fmt, ##__VA_ARGS__)

#define CONTINUE_CUDA_ERR(err, fmt, ...) \
    CHECK_CUDA_ERR_NO_ACTION(err, fmt, ##__VA_ARGS__)

#define READ_SYMBOL(lib, func_name) \
    lib->symbol<decltype(&func_name)>(#func_name)

#define DIVIDE_AND_ROUND_UP(a, b) ((a + b - 1) / b)
#define INFER_ROUND_UP(value, align) (((value) + (align)-1) & (~((align)-1)))
#define INFER_ROUND_DOWN(value, align) ((value) & (~((align)-1)))
#define INFER_WILDCARD_DIM_VALUE -1
#define INFER_MEM_ALIGNMENT 1024

#endif
