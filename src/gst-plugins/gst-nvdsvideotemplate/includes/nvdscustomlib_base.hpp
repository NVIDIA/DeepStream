/*
 * SPDX-FileCopyrightText: Copyright (c) 2020 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef __NVDSCUSTOMLIB_BASE_HPP__
#define __NVDSCUSTOMLIB_BASE_HPP__

#include <gst/base/gstbasetransform.h>
#include <gst/video/video.h>

#include "gstnvdsbufferpool.h"
#include "nvdscustomlib_interface.hpp"

/* Buffer Pool Configuration Parameters */
struct BufferPoolConfig {
  gint cuda_mem_type;
  guint gpu_id;
  guint max_buffers;
  gint batch_size;
};

class DSCustomLibraryBase : public IDSCustomLibrary
{
public:
    explicit DSCustomLibraryBase(GstBaseTransform* btrans = nullptr);

    /* Set Init Parameters */
    virtual bool SetInitParams(DSCustom_CreateParams *params);

    virtual ~DSCustomLibraryBase();

    /* Set Custom Properties  of the library */
    virtual bool SetProperty(Property &prop) = 0;

    virtual bool HandleEvent (GstEvent *event) = 0;
    // TODO: Add getProperty as well

    virtual char* QueryProperties () = 0;

    /* Get GetCompatibleOutputCaps */
    virtual GstCaps* GetCompatibleCaps (GstPadDirection direction,
        GstCaps* in_caps, GstCaps* othercaps);

    /* Process Incoming Buffer */
    virtual BufferResult ProcessBuffer(GstBuffer *inbuf) = 0;

    /* Helped function to get the NvBufSurface from the GstBuffer */
    NvBufSurface *getNvBufSurface (GstBuffer *inbuf);

    /* Helper function to create the custom buffer pool */
    GstBufferPool* CreateBufferPool (BufferPoolConfig *pool_config, GstCaps *outcaps);

public:
    /* Gstreamer dsexaple2 plugin's base class reference */
    GstBaseTransform *m_element;

    /** GPU ID on which we expect to execute the algorithm */
    guint m_gpuId;

    gboolean m_dummyMetaInsert;
    
    gboolean m_fillDummyBatchMeta;

    /* Video Information */
    GstVideoInfo m_inVideoInfo;
    GstVideoInfo m_outVideoInfo;

    /* Video Format Information */
    GstVideoFormat m_inVideoFmt;
    GstVideoFormat m_outVideoFmt;

    /* Gst Caps Information */
    GstCaps *m_inCaps;
    GstCaps *m_outCaps;
};


DSCustomLibraryBase::DSCustomLibraryBase(GstBaseTransform* btrans) : m_element(btrans)
{
    m_inCaps = NULL;
    m_outCaps = NULL;
    m_gpuId = 0;
    m_dummyMetaInsert = false;
    m_fillDummyBatchMeta = false;
}

bool DSCustomLibraryBase::SetInitParams(DSCustom_CreateParams *params) {
    m_element = params->m_element;
    m_inCaps = params->m_inCaps;
    m_outCaps = params->m_outCaps;
    m_gpuId = params->m_gpuId;
    m_dummyMetaInsert = params->m_dummyMetaInsert;
    m_fillDummyBatchMeta = params->m_fillDummyBatchMeta;

    gst_video_info_from_caps(&m_inVideoInfo, m_inCaps);
    gst_video_info_from_caps(&m_outVideoInfo, m_outCaps);

    m_inVideoFmt = GST_VIDEO_FORMAT_INFO_FORMAT (m_inVideoInfo.finfo);
    m_outVideoFmt = GST_VIDEO_FORMAT_INFO_FORMAT (m_outVideoInfo.finfo);

    return true;
}

DSCustomLibraryBase::~DSCustomLibraryBase() {
}

GstCaps* DSCustomLibraryBase::GetCompatibleCaps (GstPadDirection direction,
 GstCaps* in_caps, GstCaps* othercaps)
{
  GstCaps* result = NULL;
  GstStructure *s1, *s2;
  gint width = 0, height = 0;
  gint num = 0, denom = 0;
  const gchar *inputFmt = NULL;

  GST_INFO_OBJECT (m_element, "\n----------\ndirection = %d (1=Src, 2=Sink) -> %s:\nCAPS = %s\n",
   direction, __func__, gst_caps_to_string(in_caps));
  GST_INFO_OBJECT (m_element, "%s : OTHERCAPS = %s\n", __func__, gst_caps_to_string(othercaps));

#if 0
  GST_INFO_OBJECT (nvdsvideotemplate, "%s : CAPS = %" GST_PTR_FORMAT "\n\n", __func__, in_caps);
  GST_INFO_OBJECT (nvdsvideotemplate, "%s : OTHER CAPS = %" GST_PTR_FORMAT "\n\n", __func__, othercaps);
#endif

  othercaps = gst_caps_truncate(othercaps);
  othercaps = gst_caps_make_writable(othercaps);

  // TODO:
  // Currently selecting only first caps structure
  {
    s1 = gst_caps_get_structure(in_caps, 0);
    s2 = gst_caps_get_structure(othercaps, 0);

    inputFmt = gst_structure_get_string (s1, "format");
    gst_structure_get_int (s1, "width", &width);
    gst_structure_get_int (s1, "height", &height);

    if (0)
      g_print ("InputFMT = %s \n\n", inputFmt);

    /* otherwise the dimension of the output heatmap needs to be fixated */

    // Here change the width and height on output caps based on the information provided
    // byt the custom library
    gst_structure_fixate_field_nearest_int(s2, "width", width);
    gst_structure_fixate_field_nearest_int(s2, "height", height);
    if (gst_structure_get_fraction(s1, "framerate", &num, &denom))
    {
      gst_structure_fixate_field_nearest_fraction(s2, "framerate", num, denom);
    }

    gst_structure_remove_fields (s2, "width", "height", "format", NULL);

    // TODO: Get width, height, coloutformat, and framerate from customlibrary API
    // set the new properties accordingly
    gst_structure_set (s2, "width", G_TYPE_INT, width,
        "height", G_TYPE_INT, height,
        "format", G_TYPE_STRING, "NV12",
        NULL);

    result = gst_caps_ref(othercaps);
  }

  gst_caps_unref(othercaps);

  GST_INFO_OBJECT (m_element, "%s : Updated OTHERCAPS = %s \n\n", __func__, gst_caps_to_string(othercaps));
#if 0
  GST_INFO_OBJECT (nvdsvideotemplate, "%s : CAPS = %" GST_PTR_FORMAT "\n\n", __func__, othercaps);
  GST_INFO_OBJECT(nvdsvideotemplate, "CAPS fixate: %" GST_PTR_FORMAT ", direction %d",
      result, direction);
#endif
  return result;
}

GstBufferPool* DSCustomLibraryBase::CreateBufferPool
    (BufferPoolConfig *pool_config, GstCaps *outcaps)
{
    GstBufferPool *m_buf_pool= NULL;
    GstStructure *config = NULL;

    m_buf_pool = gst_nvds_buffer_pool_new ();

    config = gst_buffer_pool_get_config (m_buf_pool);

    GST_INFO_OBJECT (m_element, "in videoconvert caps = %" GST_PTR_FORMAT "\n", outcaps);
    gst_buffer_pool_config_set_params (config, outcaps, sizeof (NvBufSurface), pool_config->max_buffers, pool_config->max_buffers+4);

    gst_structure_set (config,
            "memtype", G_TYPE_UINT, pool_config->cuda_mem_type,
            "gpu-id", G_TYPE_UINT, pool_config->gpu_id,
            "batch-size", G_TYPE_UINT, pool_config->batch_size, NULL);

    GST_INFO_OBJECT (m_element, " %s Allocating Buffers in NVM Buffer Pool for Max_Views=%d\n",
            __func__, pool_config->batch_size);

    /* set config for the created buffer pool */
    if (!gst_buffer_pool_set_config (m_buf_pool, config)) {
        GST_WARNING ("bufferpool configuration failed");
        return NULL;
    }

    gboolean is_active = gst_buffer_pool_set_active (m_buf_pool, TRUE);
    if (!is_active) {
        GST_WARNING (" Failed to allocate the buffers inside the output pool");
        return NULL;
    } else {
        GST_DEBUG (" Output buffer pool (%p) successfully created with %d buffers",
                m_buf_pool, pool_config->max_buffers);
    }
    return m_buf_pool;
}

/* Helped function to get the NvBufSurface from the GstBuffer */
NvBufSurface *DSCustomLibraryBase::getNvBufSurface (GstBuffer *inbuf)
{
    GstMapInfo in_map_info = GST_MAP_INFO_INIT;
    NvBufSurface *nvbuf_surface = NULL;

    /* Map the buffer contents and get the pointer to NvBufSurface. */
    if (!gst_buffer_map (inbuf, &in_map_info, GST_MAP_READ)) {
        GST_ELEMENT_ERROR (m_element, STREAM, FAILED,
            ("%s:gst buffer map to get pointer to NvBufSurface failed", __func__), (NULL));
        return NULL;
    }

    // Assuming that the plugin uses DS NvBufSurface data structure
    nvbuf_surface = (NvBufSurface *) in_map_info.data;

    gst_buffer_unmap(inbuf, &in_map_info);
    return nvbuf_surface;
}

#endif
