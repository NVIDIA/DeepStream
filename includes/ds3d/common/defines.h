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

#ifndef _DS3D_COMMON_DEFINES__H
#define _DS3D_COMMON_DEFINES__H

#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>

#define ENABLE_DEBUG NvDs3dEnableDebug()

#ifndef DS3D_DISABLE_CLASS_COPY
#define DS3D_DISABLE_CLASS_COPY(NoCopyClass)  \
    NoCopyClass(const NoCopyClass&) = delete; \
    void operator=(const NoCopyClass&) = delete
#endif

#undef DS_ASSERT
#define DS_ASSERT(...) assert((__VA_ARGS__))

#if defined(NDEBUG)
#define DS3D_FORMAT_(fmt, ...) fmt "\n", ##__VA_ARGS__
#else
#define DS3D_FORMAT_(fmt, ...) "%s:%d, " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__
#endif


#define DS3D_LOG_PRINT_(out, level, fmt, ...)                   \
    fprintf(out, DS3D_FORMAT_(#level ": " fmt, ##__VA_ARGS__)); \
    fflush(out)

#undef LOG_PRINT
#undef LOG_DEBUG
#undef LOG_ERROR
#undef LOG_WARNING
#undef LOG_INFO

#define LOG_PRINT(fmt, ...)              \
    fprintf(stdout, fmt, ##__VA_ARGS__); \
    fflush(stdout)

#define LOG_DEBUG(fmt, ...)                                 \
    if (ENABLE_DEBUG) {                                     \
        DS3D_LOG_PRINT_(stdout, DEBUG, fmt, ##__VA_ARGS__); \
    }

#define LOG_ERROR(fmt, ...) DS3D_LOG_PRINT_(stderr, ERROR, fmt, ##__VA_ARGS__)
#define LOG_WARNING(fmt, ...) DS3D_LOG_PRINT_(stderr, WARNING, fmt, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...) DS3D_LOG_PRINT_(stdout, INFO, fmt, ##__VA_ARGS__)

// check ERROR
#define DS3D_FAILED_RETURN(condition, ret, fmt, ...)         \
    do {                                                     \
        if (!(condition)) {                                  \
            LOG_ERROR(fmt ", check failure", ##__VA_ARGS__); \
            return (ret);                                    \
        }                                                    \
    } while (0)

#define DS3D_ERROR_RETURN(code, fmt, ...)                           \
    do {                                                            \
        ErrCode __cd = (code);                                      \
        DS3D_FAILED_RETURN(isGood(__cd), __cd, fmt, ##__VA_ARGS__); \
    } while (0)

#define DS3D_THROW_ERROR(statement, code, msg) \
    if (!(statement)) {                        \
        throwError(code, msg);                 \
    }

#define DS3D_THROW_ERROR_FMT(statement, code, fmt, ...)           \
    if (!(statement)) {                                           \
        char __smsg[2048] = {0};                                  \
        snprintf(__smsg, sizeof(__smsg) - 1, fmt, ##__VA_ARGS__); \
        throwError(code, __smsg);                                 \
    }

#define DS3D_CHECK_CUDA_ERROR(err, action, fmt, ...)                            \
    do {                                                                        \
        cudaError_t errnum = (err);                                             \
        if (errnum != cudaSuccess) {                                            \
            LOG_ERROR(                                                          \
                fmt ", cuda err_no:%d, err_str:%s", ##__VA_ARGS__, (int)errnum, \
                cudaGetErrorName(errnum));                                      \
            action;                                                             \
        }                                                                       \
    } while (0)

#define DS3D_UNUSED(a) (void)(a)

#undef DS3D_TRY
#undef DS3D_CATCH_ERROR
#undef DS3D_CATCH_ANY
#undef DS3D_CATCH_FULL

#define DS3D_TRY try

#define DS3D_CATCH_ERROR(type, errcode, fmt, ...)                           \
    catch (const type& e)                                                   \
    {                                                                       \
        LOG_ERROR(fmt ", catched " #type ":  %s", ##__VA_ARGS__, e.what()); \
        return errcode;                                                     \
    }

#define DS3D_CATCH_ANY(errcode, fmt, ...)                        \
    catch (...)                                                  \
    {                                                            \
        LOG_ERROR(fmt ", catched unknown error", ##__VA_ARGS__); \
        return errcode;                                          \
    }

#define DS3D_EXPORT_API __attribute__((__visibility__("default")))
#define DS3D_EXTERN_C_BEGIN extern "C" {
#define DS3D_EXTERN_C_END }

#define DS3D_STR_PREFIX "DS3D::"
#define DS3D_KEY_NAME(name) DS3D_STR_PREFIX name

#define REGISTER_TYPE_ID(uint64Id) \
    static constexpr TIdType __typeid() { return uint64Id; }


#define DS3D_ROUND_UP(value, align) (((value) + (align)-1) & (~((align)-1)))

#endif  // _DS3D_COMMON_DEFINES__H