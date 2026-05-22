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
