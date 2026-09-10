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


#ifndef __GST_NVDS_3D_FILTER_H__
#define __GST_NVDS_3D_FILTER_H__

#include <glib-object.h>
#include <gst/base/gstbasetransform.h>

#include <ds3d/common/func_utils.h>
#include <ds3d/common/config.h>
#include <ds3d/common/hpp/obj.hpp>
#include <ds3d/common/hpp/databridge.hpp>
#include <ds3d/common/hpp/yaml_config.hpp>

#include <ds3d/gst/custom_lib_factory.h>
#include <ds3d/gst/nvds3d_gst_plugin.h>
#include <ds3d/gst/nvds3d_gst_ptr.h>
#include <ds3d/gst/nvds3d_meta.h>

/* Package and library details required for plugin_init */
#define PACKAGE "nvds3dbridge"
#define VERSION "1.0"
#define LICENSE "Proprietary"
#define DESCRIPTION "NVIDIA DeepStream Plugin for 3d bridge custom plugin"
#define BINARY_PACKAGE "NVIDIA DeepStream 3D Bridge Plugin"
#define URL "http://nvidia.com/"

G_BEGIN_DECLS
/* Standard boilerplate stuff */
typedef struct _GstNvDs3dBridge GstNvDs3dBridge;
typedef struct _GstNvDs3dBridgeClass GstNvDs3dBridgeClass;

/* Standard boilerplate stuff */
#define GST_TYPE_NVDS3DFILTER (gst_nvds3d_bridge_get_type())
#define GST_NVDS3DFILTER(obj) \
    (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_NVDS3DFILTER, GstNvDs3dBridge))
#define GST_NVDS3DFILTER_CLASS(klass) \
    (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_NVDS3DFILTER, GstNvDs3dBridgeClass))
#define GST_NVDS3DFILTER_GET_CLASS(obj) \
    (G_TYPE_INSTANCE_GET_CLASS((obj), GST_TYPE_NVDS3DFILTER, GstNvDs3dBridgeClass))
#define GST_IS_NVDS3DFILTER(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_NVDS3DFILTER))
#define GST_IS_NVDS3DFILTER_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_NVDS3DFILTER))
#define GST_NVDS3DFILTER_CAST(obj) ((GstNvDs3dBridge*)(obj))

struct GstNvDs3dBridgeImpl;

struct _GstNvDs3dBridge {
    GstBaseTransform base_trans;

    GstNvDs3dBridgeImpl* impl;
};

struct _GstNvDs3dBridgeClass {
    GstBaseTransformClass parent_class;
};

GType gst_nvds3d_bridge_get_type(void);

G_END_DECLS
#endif /* __GST_NVDS_3D_FILTER_H__ */
