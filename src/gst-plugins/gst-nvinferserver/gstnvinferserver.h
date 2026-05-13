/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

/**
 * @file gstnvinferserver.h
 *
 * @brief nvdsgst_inferserver plugin header file.
 *
 * This file contains the standard GStreamer boilerplate definitions and
 * declarations for the nvinferserver element and plugin.
 *
 */

#ifndef __GST_NVINFER_SERVER_H__
#define __GST_NVINFER_SERVER_H__

#include <gst/base/gstbasetransform.h>
#include <gst/video/video.h>

#include "gstnvdsinfer.h"
#include "gstnvdsmeta.h"

/* Package and library details required for plugin_init */
#define PACKAGE "nvinferserver"
#define VERSION "1.0"
#define LICENSE "Proprietary"
#define DESCRIPTION "NVIDIA DeepStreamSDK TensorRT Inference Server plugin"
#define BINARY_PACKAGE "NVIDIA DeepStreamSDK TensorRT Inference Server plugin"
#define URL "http://nvidia.com/"

namespace gstnvinferserver {
    class GstNvInferServerImpl;
}

G_BEGIN_DECLS

/* Standard GStreamer boilerplate */
typedef struct _GstNvInferServer GstNvInferServer;
typedef struct _GstNvInferServerClass GstNvInferServerClass;

/* Standard GStreamer boilerplate */
#define GST_TYPE_NVINFER_SERVER (gst_nvinfer_server_get_type())
#define GST_NVINFER_SERVER(obj)  \
    (G_TYPE_CHECK_INSTANCE_CAST( \
        (obj), GST_TYPE_NVINFER_SERVER, GstNvInferServer))
#define GST_NVINFER_SERVER_CLASS(klass) \
    (G_TYPE_CHECK_CLASS_CAST(           \
        (klass), GST_TYPE_NVINFER_SERVER, GstNvInferServerClass))
#define GST_NVINFER_SERVER_GET_CLASS(obj) \
    (G_TYPE_INSTANCE_GET_CLASS(           \
        (obj), GST_TYPE_NVINFER_SERVER, GstNvInferServerClass))
#define GST_IS_NVINFER(obj) \
    (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_NVINFER_SERVER))
#define GST_IS_NVINFER_CLASS(klass) \
    (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_NVINFER_SERVER))
#define GST_NVINFER_SERVER_CAST(obj) ((GstNvInferServer*)(obj))

/**
 * @brief List of GObject properties for the element.
 */
enum GstNvInferServerProperty{
    PROP_0,
    PROP_UNIQUE_ID,
    PROP_PROCESS_MODE,
    PROP_CONFIG_FILE_PATH,
    PROP_BATCH_SIZE,
    PROP_INFER_ON_GIE_ID,
    PROP_OPERATE_ON_CLASS_IDS,
    PROP_INTERVAL,
    PROP_OUTPUT_CALLBACK,
    PROP_OUTPUT_CALLBACK_USERDATA,
    PROP_INPUT_TENSOR_META,
    PROP_LAST
};

/**
 * @brief Opaque structure storing data of the nvinferserver element.
 */
struct _GstNvInferServer {
    /** Base class for GstNvInferServer.
     * Should be the first member when extending from GstBaseTransform. */
    GstBaseTransform base_trans;

    /** Boolean indicating if the bound buffer contents should be written to
     * file. */
    gboolean write_raw_buffers_to_file;

    /** Batch counter for writing buffer contents to file. */
    guint64 file_write_batch_num;

    /** Pointer to the callback function and user data for application access to
     * the bound buffer contents. */
    gst_nvinfer_raw_output_generated_callback output_generated_callback;
    gpointer output_generated_userdata;

    /** GstFlowReturn returned by the latest buffer pad push. */
    GstFlowReturn last_flow_ret;

    /** Current batch number of the input batch. */
    guint64 current_batch_num;

    /** Pointer to the GstNvInferServerImpl object for this instance */
    gstnvinferserver::GstNvInferServerImpl* impl;
};

/**
 * @brief The class structure for the nvinferserver element.
 *
 * Derive nvinferserver from GstBaseTransform.
 * GStreamer boilerplate.
 */
struct _GstNvInferServerClass {
    GstBaseTransformClass parent_class;
};

GType gst_nvinfer_server_get_type(void);

G_END_DECLS

#endif /* __GST_INFER_H__ */
