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