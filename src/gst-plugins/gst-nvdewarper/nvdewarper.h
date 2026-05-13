/**
 * SPDX-FileCopyrightText: Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
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
