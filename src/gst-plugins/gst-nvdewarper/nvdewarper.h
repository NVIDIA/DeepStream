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
 * @file nvdewarper.h
 *
 * @b Description: This file declares the core dewarping function calls.
 */
#ifndef _NVDEWARPER_H_
#define _NVDEWARPER_H_
#include "gstnvdewarper.h"

#include "nvbufsurface.h"

inline bool CUDA_CHECK_(gint e, gint iLine, const gchar *szFile) {
  if (e != cudaSuccess) {
    std::cout << "CUDA runtime error " << e << " at line " << iLine << " in file " << szFile << endl;
    return false;
  }
  return true;
}

#define cuda_ck(call) CUDA_CHECK_(call, __LINE__, __FILE__)

#define BAIL_IF_FALSE(x, err, code)  do { if (!(x)) { err = code; goto bail; } } while(0)

/**
 * Function definition of dewarping call for each surface.
 *
 * @param[in] nvdewarper Width of the network input, in pixels.
 * @param[in] in_surface Height of the network input, in pixels.
 * @param[in] out_surface Color format of the buffers in the pool.
 *
 * @return Cuda Error. "cudaSuccess" in case of Success.
 */
cudaError gst_nvdewarper_do_dewarp (Gstnvdewarper *nvdewarper,
    NvBufSurface *in_surface, NvBufSurface *out_surface);

/**
 * Function to get core Dewarper library version.
 *
 * @return  The version number.
 */
uint32_t gst_nvdewarper_version();

#endif
