/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <string.h>
#include <stdio.h>
#include "nvds_version.h"
#include <dlfcn.h>
//#include <cuda.h>
//#include <cuda_runtime.h>
#include <NvInferRuntimeCommon.h>
#include <gst/gst.h>

#define CUDNN_LIBRARY "libcudnn.so"
#define CUDNN_LIBRARY_VERSION "libcudnn.so.7"
#define CUDART_LIBRARY "libcudart.so"

//extern "C" size_t
typedef size_t (*fncudnnGetVersion) (void);
typedef void (*fncudaDriverGetVersion) (int *);
typedef void (*fncudaRuntimeGetVersion) (int *);
static void *cudnn_lib = NULL;
static void *cudart_lib = NULL;
static fncudnnGetVersion cudnnGetVersion;
static fncudaDriverGetVersion cudaDriverGetVersionfn;
static fncudaRuntimeGetVersion cudaRuntimeGetVersionfn;

void
nvds_version (unsigned int * major, unsigned int * minor)
{
  if (major != NULL)
    *major = NVDS_VERSION_MAJOR;
  if (minor != NULL)
    *minor = NVDS_VERSION_MINOR;
}

void
nvds_version_print (void)
{
  unsigned int major, minor;
  nvds_version (&major, &minor);
  printf ("DeepStreamSDK %d.%d.%d\n", major, minor, NVDS_VERSION_MICRO);
}

void
nvds_dependencies_version_print (void)
{
  int driverVersion = 0;
  int runtimeVersion = 0;
  cudnn_lib = dlopen (CUDNN_LIBRARY, RTLD_NOW);
  if (cudnn_lib)
  {
    cudnnGetVersion = (fncudnnGetVersion)dlsym (cudnn_lib, "cudnnGetVersion");
  }
  else
  {
    cudnn_lib = dlopen (CUDNN_LIBRARY_VERSION, RTLD_NOW);
    if (cudnn_lib)
    {
      cudnnGetVersion = (fncudnnGetVersion)dlsym (cudnn_lib, "cudnnGetVersion");
    }
  }
  if (cudnn_lib && (!cudnnGetVersion)) {
    dlclose (cudnn_lib);
    cudnn_lib = NULL; cudnnGetVersion = NULL;
  }
  cudart_lib = dlopen (CUDART_LIBRARY, RTLD_NOW);
  if (cudart_lib)
  {
    cudaDriverGetVersionfn = (fncudaDriverGetVersion)dlsym (cudart_lib, "cudaDriverGetVersion");
    cudaRuntimeGetVersionfn = (fncudaRuntimeGetVersion)dlsym (cudart_lib, "cudaRuntimeGetVersion");
  }
  else
  {
    printf("%s CUDA RT library unavailable in path\n", CUDART_LIBRARY);
  }
  if (cudart_lib && (!cudaDriverGetVersionfn || !cudaRuntimeGetVersionfn)) {
    dlclose (cudart_lib);
    cudart_lib = NULL; cudaDriverGetVersionfn = NULL; cudaRuntimeGetVersionfn = NULL;
  }
  if(cudaDriverGetVersionfn) {
    cudaDriverGetVersionfn(&driverVersion);
    printf ("CUDA Driver Version: %d.%d\n", driverVersion/1000, (driverVersion/10)%100);
  }
  else {
    printf ("CUDA Driver Version not available\n");
  }
  if(cudaRuntimeGetVersionfn) {
    cudaRuntimeGetVersionfn(&runtimeVersion);
    printf ("CUDA Runtime Version: %d.%d\n", runtimeVersion/1000, (runtimeVersion/10)%100);
  }
  else {
    printf ("CUDA Runtime Version not available\n");
  }
  if(getInferLibVersion() >= 10000) {
    printf ("TensorRT Version: %d.%d\n", getInferLibVersion()/10000, (getInferLibVersion()/100)%100);
  }
  else {
    printf ("TensorRT Version: %d.%d\n", getInferLibVersion()/1000, (getInferLibVersion()/100)%10);
  }
  if (cudnnGetVersion) {
    if(cudnnGetVersion() >= 90000) {
      printf ("cuDNN Version: %zu.%zu\n", cudnnGetVersion()/10000, (cudnnGetVersion()/100)%100);
    }
    else {
      printf ("cuDNN Version: %zu.%zu\n", cudnnGetVersion()/1000, (cudnnGetVersion()/100)%10);
    }
  }
  else {
    printf ("cuDNN Version not available\n");
  }

  typedef uint32_t (*fnNvwarpVersion) (void);
  void *warp360_lib = dlopen ("libnvds_dewarper.so", RTLD_NOW);
  if (warp360_lib) {
    fnNvwarpVersion nvwarpVersionFn =
        (fnNvwarpVersion) dlsym (warp360_lib, "nvwarpVersion");
    if (nvwarpVersionFn) {
      unsigned version = nvwarpVersionFn ();
      if (0 != (version & 0xFF))
        printf ("libNVWarp360 Version: %u.%u.%ud%u\n", version >> 24,
            (version >> 16) & 0xFF, (version >> 8) & 0xFF, version & 0xFF);
      else
        printf ("libNVWarp360 Version: %u.%u.%u\n", version >> 24,
            (version >> 16) & 0xFF, (version >> 8) & 0xFF);
    }
    dlclose (warp360_lib);
  } else {
    printf ("libNVWarp360: not found\n");
  }
}
