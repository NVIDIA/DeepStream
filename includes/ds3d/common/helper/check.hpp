/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef DS3D_COMMON_HELPER_CHECK_HPP
#define DS3D_COMMON_HELPER_CHECK_HPP

#include <assert.h>
#include <cuda_runtime.h>
#include <stdarg.h>
#include <stdio.h>

#include <string>

namespace ds3d {

#define DS3D_NVUNUSED2(a, b) \
  {                     \
    (void)(a);          \
    (void)(b);          \
  }
#define DS3D_NVUNUSED(a) \
  { (void)(a); }

#ifndef DS3D_INFER_ASSERT
#define DS3D_INFER_ASSERT(expr)                                                \
    do {                                                                       \
        if (!(expr)) {                                                         \
            fprintf(stderr, "%s:%d ASSERT(%s) \n", __FILE__, __LINE__, #expr); \
            std::abort();                                                      \
        }                                                                      \
    } while (0)
#endif

#define checkRuntime(call) ds3d::check_runtime(call, #call, __LINE__, __FILE__)

#define checkKernel(...)                                                        \
  do {                                                                          \
    __VA_ARGS__;                                                                \
    ds3d::check_runtime(cudaPeekAtLastError(), #__VA_ARGS__, __LINE__, __FILE__); \
  } while (false)
#define dprintf(...)

#define checkCudaErrors(cudaErrorCode)                                                             \
    {                                                                                              \
        cudaError_t status = cudaErrorCode;                                                        \
        if (status != 0) {                                                                         \
            std::cout << "Cuda failure: " << cudaGetErrorString(status) << " at line " << __LINE__ \
                      << " in file " << __FILE__ << " error status: " << status << std::endl;      \
            abort();                                                                               \
        }                                                                                          \
    }

#define Assertf(cond, fmt, ...)                                                                                         \
  do {                                                                                                                  \
    if (!(cond)) {                                                                                                      \
      fprintf(stderr, "Assert failed 💀. %s in file %s:%d, message: " fmt "\n", #cond, __FILE__, __LINE__, __VA_ARGS__); \
      abort();                                                                                                          \
    }                                                                                                                   \
  } while (false)

#define Asserts(cond, s)                                                                                 \
  do {                                                                                                   \
    if (!(cond)) {                                                                                       \
      fprintf(stderr, "Assert failed 💀. %s in file %s:%d, message: " s "\n", #cond, __FILE__, __LINE__); \
      abort();                                                                                           \
    }                                                                                                    \
  } while (false)

#define Assert(cond)                                                                     \
  do {                                                                                   \
    if (!(cond)) {                                                                       \
      fprintf(stderr, "Assert failed 💀. %s in file %s:%d\n", #cond, __FILE__, __LINE__); \
      abort();                                                                           \
    }                                                                                    \
  } while (false)

static inline bool check_runtime(cudaError_t e, const char *call, int line, const char *file) {
  if (e != cudaSuccess) {
    fprintf(stderr,
            "CUDA Runtime error %s # %s, code = %s [ %d ] in file "
            "%s:%d\n",
            call, cudaGetErrorString(e), cudaGetErrorName(e), e, file, line);
    abort();
    return false;
  }
  return true;
}

};  // namespace ds3d

#endif  // DS3D_COMMON_HELPER_CHECK_HPP