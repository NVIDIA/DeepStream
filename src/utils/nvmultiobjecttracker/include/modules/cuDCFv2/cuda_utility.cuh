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

#ifndef CUDCFLIB_CUDCF_UTILITY_CUH_
#define CUDCFLIB_CUDCF_UTILITY_CUH_

#include <cuda_runtime.h>
#include <cuComplex.h>

namespace CUDA_UTIL
{

/**
 * @brief Get free, used, and total GPU memory in bytes.
 * @param[out] free_db The available GPU memory in bytes.
 * @param[out] used_db The used GPU memory in bytes.
 * @param[out] total_db The total GPU memory in bytes.
 */
void static getGpuFreeMemory(double* free_db, double* used_db, double* total_db)
{
  size_t free_byte ;
  size_t total_byte ;
  cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;
  if ( cudaSuccess != cuda_status )
  {
    printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
    exit(EXIT_FAILURE);
  }

  *free_db = (double)free_byte ;
  *total_db = (double)total_byte ;
  *used_db = *total_db - *free_db ;
  return;
}

/**
 * @brief Atomically store the greater of the current and supplied values.
 * @param[in,out] address The floating-point value updated atomically.
 * @param[in] val The candidate maximum value.
 * @return The value stored before the atomic update.
 */
__device__ static inline float atomicMaxFloat(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed, __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

/**
 * @brief Atomically store the lesser of the current and supplied values.
 * @param[in,out] address The floating-point value updated atomically.
 * @param[in] val The candidate minimum value.
 * @return The value stored before the atomic update.
 */
__device__ static inline float atomicMinFloat(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed, __float_as_int(::fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

/**
 * @brief Atomically replace a floating-point value.
 * @param[out] address The floating-point value updated atomically.
 * @param[in] val The replacement value.
 * @return The value stored before the atomic update.
 */
__device__ static inline float atomicSetFloat(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed, __float_as_int(val));
    } while (assumed != old);
    return __int_as_float(old);
}

/**
 * @brief Compute the absolute value of an integer.
 * @param[in] i The input integer.
 */
static __device__ inline int abs(const int i)
{
    return i < 0 ? (-i) : i;
}

/**
 * @brief Compute a non-negative modulo result, including for a negative dividend.
 * @param[in] a The dividend.
 * @param[in] b The divisor.
 */
static __device__ inline int modul(int a, int b)
{
    return ((a % b) + b) % b;
}

/**
 * @brief Add a real scalar to the real component of a complex value.
 * @param[in] a The complex addend.
 * @param[in] b The real addend.
 */
static __device__ inline cuComplex ComplexAdd(cuComplex a, float b)
{
  cuComplex c;
  c.x = a.x + b;
  c.y = a.y;
  return c;
}

/**
 * @brief Add two complex values.
 * @param[in] a The first complex addend.
 * @param[in] b The second complex addend.
 */
static __device__ inline cuComplex ComplexAdd(cuComplex a, cuComplex b)
{
  cuComplex c;
  c.x = a.x + b.x;
  c.y = a.y + b.y;
  return c;
}

/**
 * @brief Scale a complex value by a real scalar.
 * @param[in] a The complex value.
 * @param[in] s The real scale factor.
 */
static __device__ inline cuComplex ComplexScale(cuComplex a, float s)
{
  cuComplex c;
  c.x = s * a.x;
  c.y = s * a.y;
  return c;
}

/**
 * @brief Multiply a complex value by the conjugate of another complex value.
 * @param[in] a The first complex factor.
 * @param[in] b The complex factor that is conjugated.
 */
static __device__ inline cuComplex ComplexMulConjB(cuComplex a, cuComplex b)
{
  cuComplex c;
  c.x = a.x * b.x + a.y * b.y;
  c.y = a.y * b.x - a.x * b.y;
  return c;
}

/**
 * @brief Multiply two complex values.
 * @param[in] a The first complex factor.
 * @param[in] b The second complex factor.
 */
static __device__ inline cuComplex ComplexMul(cuComplex a, cuComplex b)
{
  cuComplex c;
  c.x = a.x * b.x - a.y * b.y;
  c.y = a.x * b.y + a.y * b.x;
  return c;
}

} // end of namespace CUDA_UTIL

#endif // CUDCFLIB_CUDCF_UTILITY_CUH_
