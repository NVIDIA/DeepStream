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

#ifndef NVTRACKER2LIB_CUDA_COMMON_HPP_
#define NVTRACKER2LIB_CUDA_COMMON_HPP_

#include <cstdlib>
#include <iostream>
#include <string>
#include <cassert>

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include "cuda.h"
#include "cufft.h"
#include "cuComplex.h"
#include "cublas_v2.h"

#ifndef _WIN32
#include <vpi/Status.h>
#endif

/**
 * @brief Return the symbolic name of a cuFFT status.
 * @param[in] error cuFFT status value.
 */
static const char *_cufftGetErrorEnum(cufftResult error)
{
    switch (error)
    {
        case CUFFT_SUCCESS: // 0, The cuFFT operation was successful
            return "CUFFT_SUCCESS";

        case CUFFT_INVALID_PLAN: // 1, cuFFT was passed an invalid plan handle
            return "CUFFT_INVALID_PLAN";

        case CUFFT_ALLOC_FAILED: // 2, cuFFT failed to allocate GPU or CPU memory
            return "CUFFT_ALLOC_FAILED";

        case CUFFT_INVALID_TYPE: // 3, No longer used
            return "CUFFT_INVALID_TYPE";

        case CUFFT_INVALID_VALUE: // 4, User specified an invalid pointer or parameter
            return "CUFFT_INVALID_VALUE";

        case CUFFT_INTERNAL_ERROR: // 5, Driver or internal cuFFT library error
            return "CUFFT_INTERNAL_ERROR";

        case CUFFT_EXEC_FAILED: // 6, Failed to execute an FFT on the GPU
            return "CUFFT_EXEC_FAILED";

        case CUFFT_SETUP_FAILED: // 7, The cuFFT library failed to initialize
            return "CUFFT_SETUP_FAILED";

        case CUFFT_INVALID_SIZE: // 8, User specified an invalid transform size
            return "CUFFT_INVALID_SIZE";

        case CUFFT_UNALIGNED_DATA: // 9, No longer used
            return "CUFFT_UNALIGNED_DATA";

        case CUFFT_INVALID_DEVICE: // 11, Execution of a plan was on different GPU than plan creation
            return "CUFFT_INVALID_DEVICE";

        case CUFFT_NO_WORKSPACE: // 13, No workspace has been provided prior to plan execution
            return "CUFFT_NO_WORKSPACE";

        case CUFFT_NOT_IMPLEMENTED: // 14, Function does not implement functionality for parameters given.
            return "CUFFT_NOT_IMPLEMENTED";

        case CUFFT_NOT_SUPPORTED: // 16, Operation is not supported for parameters given.
            return "CUFFT_NOT_SUPPORTED";

        default:
            return "Unknown CUFFT error";
    }
}

/**
 * @brief Return the symbolic name of a cuBLAS status.
 * @param[in] error cuBLAS status value.
 */
static const char *_cublasGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS: // 0
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED: // 1
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED: // 3
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE: // 7
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH: // 8
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR: // 11
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED: // 13
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR: // 14
            return "CUBLAS_STATUS_INTERNAL_ERROR";

        case CUBLAS_STATUS_NOT_SUPPORTED: // 15
            return "CUBLAS_STATUS_NOT_SUPPORTED";

        case CUBLAS_STATUS_LICENSE_ERROR: // 16
            return "CUBLAS_STATUS_LICENSE_ERROR";

        default:
            return "Unknown CUBLAS error";
    }
}

/**
 * @brief Report a CUDA driver error and optionally throw an exception.
 * @param[in] code CUDA driver status to check.
 * @param[in] file Source filename associated with the check.
 * @param[in] line Source line associated with the check.
 * @param[in] abort Enables throwing an exception after reporting an error.
 */
inline void gpuAssert(CUresult code, const char *file, int line, bool abort=true)
{
   if (code != CUDA_SUCCESS)
   {
      const char* errstr = nullptr;
      cuGetErrorString(code, &errstr);
      fprintf(stderr,"GPUassert: %s %s %d\n", errstr, file, line);
      if (abort) { throw std::runtime_error("GPUassert failed"); }
   }
}

/**
 * @brief Report a CUDA runtime error and optionally throw an exception.
 * @param[in] code CUDA runtime status to check.
 * @param[in] file Source filename associated with the check.
 * @param[in] line Source line associated with the check.
 * @param[in] abort Enables throwing an exception after reporting an error.
 */
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) { throw std::runtime_error("GPUassert failed"); }
   }
}

/**
 * @brief Report a cuBLAS error and optionally throw an exception.
 * @param[in] code cuBLAS status to check.
 * @param[in] file Source filename associated with the check.
 * @param[in] line Source line associated with the check.
 * @param[in] abort Enables throwing an exception after reporting an error.
 */
inline void gpuAssert_cublas(cublasStatus_t code, const char *file, int line, bool abort=true)
{
   if (code != CUBLAS_STATUS_SUCCESS)
   {
      fprintf(stderr,"GPUassert_CUBLAS: %s %s %d\n", _cublasGetErrorEnum(code), file, line);
      if (abort) { throw std::runtime_error("GPUassert_CUBLAS failed"); }
   }
}

/**
 * @brief Report a cuFFT error and optionally throw an exception.
 * @param[in] code cuFFT status to check.
 * @param[in] file Source filename associated with the check.
 * @param[in] line Source line associated with the check.
 * @param[in] abort Enables throwing an exception after reporting an error.
 */
inline void gpuAssert_cufft(cufftResult code, const char *file, int line, bool abort=true)
{
   if (code != CUFFT_SUCCESS)
   {
      fprintf(stderr,"GPUassert_CUFFT: %s %s %d\n", _cufftGetErrorEnum(code), file, line);
      if (abort) { throw std::runtime_error("GPUassert_CUFFT failed"); }
   }
}

#ifndef _WIN32
/**
 * @brief Report a VPI error and optionally throw an exception.
 * @param[in] code VPI status to check.
 * @param[in] file Source filename associated with the check.
 * @param[in] line Source line associated with the check.
 * @param[in] abort Enables throwing an exception after reporting an error.
 */
inline void gpuAssert_vpi(VPIStatus code, const char *file, int line, bool abort=true)
{
   if (code != VPI_SUCCESS)
   {
      char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH] = {};
      vpiGetLastStatusMessage(buffer, sizeof(buffer));
      fprintf(stderr,"GPUassert_VPI: %s %s %s %d\n", vpiStatusGetName(code), buffer, file, line);
      if (abort) { throw std::runtime_error("GPUassert_VPI failed"); }
   }
}

#endif

// Check a CUDA, cuBLAS, cuFFT, or VPI status.
#define CUDA_CHECK(condition) \
		gpuAssert(condition, __FILE__, __LINE__);

#define CUBLAS_CHECK(condition) \
		gpuAssert_cublas(condition, __FILE__, __LINE__);

#define CUFFT_CHECK(condition) \
		gpuAssert_cufft(condition, __FILE__, __LINE__);

#ifndef _WIN32
#define VPI_CHECK(STMT)                                 \
        gpuAssert_vpi((STMT), __FILE__, __LINE__);		\

#endif

#endif // NVTRACKER2LIB_CUDA_COMMON_HPP_
