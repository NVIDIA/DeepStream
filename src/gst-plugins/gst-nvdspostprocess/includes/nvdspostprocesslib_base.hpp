/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef __NVDSPOSTPROCESSLIB_BASE_HPP__
#define __NVDSPOSTPROCESSLIB_BASE_HPP__

#include <gst/base/gstbasetransform.h>
#include <gst/video/video.h>

#include "gstnvdsbufferpool.h"
#include "nvdspostprocesslib_interface.hpp"

class DSPostProcessLibraryBase : public IDSPostProcessLibrary
{
public:
    DSPostProcessLibraryBase();

    DSPostProcessLibraryBase(DSPostProcess_CreateParams *params);

    virtual ~DSPostProcessLibraryBase();

    virtual bool HandleEvent (GstEvent *event) = 0;

    virtual bool SetConfigFile (const gchar *config_file) = 0;

    /* Process Incoming Buffer */
    virtual BufferResult ProcessBuffer(GstBuffer *inbuf) = 0;

public:
    /* Gstreamer dsexaple2 plugin's base class reference */
    GstBaseTransform *m_element;

    /** GPU ID on which we expect to execute the algorithm */
    guint m_gpuId;

    gboolean m_preprocessor_support;
    cudaStream_t m_cudaStream;

};


DSPostProcessLibraryBase::DSPostProcessLibraryBase() {
    m_element = NULL;
    m_gpuId = 0;
    m_cudaStream = 0;
    m_preprocessor_support = FALSE;
}

DSPostProcessLibraryBase::DSPostProcessLibraryBase(DSPostProcess_CreateParams *params) {
  if (params){
    m_element = params->m_element;
    m_gpuId = params->m_gpuId;
    m_cudaStream = params->m_cudaStream;
    m_preprocessor_support = params->m_preprocessor_support;
  }
  else {
    m_element = NULL;
    m_gpuId = 0;
    m_cudaStream = 0;
    m_preprocessor_support = 0;
  }
}

DSPostProcessLibraryBase::~DSPostProcessLibraryBase() {
}

/* Helped function to get the NvBufSurface from the GstBuffer */
static NvBufSurface *getNvBufSurface (GstBuffer *inbuf)
{
    GstMapInfo in_map_info = GST_MAP_INFO_INIT;
    NvBufSurface *nvbuf_surface = NULL;

    /* Map the buffer contents and get the pointer to NvBufSurface. */
    if (!gst_buffer_map (inbuf, &in_map_info, GST_MAP_READ)) {
      printf ("%s:gst buffer map to get pointer to NvBufSurface failed", __func__);
      return NULL;
    }

    // Assuming that the plugin uses DS NvBufSurface data structure
    nvbuf_surface = (NvBufSurface *) in_map_info.data;

    gst_buffer_unmap(inbuf, &in_map_info);
    return nvbuf_surface;
}

#endif
