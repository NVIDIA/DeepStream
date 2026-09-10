/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef __GST_NVDS3DMIXER_H__
#define __GST_NVDS3DMIXER_H__

#include <gst/gst.h>
#include <gst/video/video.h>

#ifndef PACKAGE
#define PACKAGE "nvds3dmixer"
#endif

#define VERSION "1.0"
#define LICENSE "Proprietary"
#define DESCRIPTION "NVIDIA 3D Mixer Plugin"
#define BINARY_PACKAGE "NVIDIA 3D Mixer Plugin"
#define URL "http://nvidia.com/"


G_BEGIN_DECLS
#define GST_TYPE_NVDS3DMIXER (gst_nvds3dmixer_get_type())
#define GST_NVDS3DMIXER(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_NVDS3DMIXER, GstNvDs3dMixer))
#define GST_NVDS3DMIXER_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_NVDS3DMIXER, GstNvDs3dMixerClass))
#define GST_IS_NVDS3DMIXER(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_NVDS3DMIXER))
#define GST_IS_NVDS3DMIXER_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_NVDS3DMIXER))
typedef struct _GstNvDs3dMixer GstNvDs3dMixer;
typedef struct _GstNvDs3dMixerClass GstNvDs3dMixerClass;

struct GstNvDs3dMixerImpl;

struct _GstNvDs3dMixer {
    GstElement element;
    GstPad* srcpad;
    GstNvDs3dMixerImpl* impl;
};

struct _GstNvDs3dMixerClass {
    GstElementClass parent_class;
};

GType gst_nvds3dmixer_get_type(void);

GType gst_nvds3dmixer_pad_get_type(void);
#define GST_TYPE_NVDS3DMIXER_PAD (gst_nvds3dmixer_pad_get_type())
#define GST_NVDS3DMIXER_PAD(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_NVDS3DMIXER_PAD, GstNvDs3dMixerPad))
#define GST_NVDS3DMIXER_PAD_CLASS(klass) \
    (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_NVDS3DMIXER_PAD, GstNvDs3dMixerPadClass))
#define GST_IS_NVDS3DMIXER_PAD(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_NVDS3DMIXER_PAD))
#define GST_IS_NVDS3DMIXER_PAD_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_NVDS3DMIXER_PAD))
#define GST_NVDS3DMIXER_PAD_CAST(obj) ((GstNvDs3dMixerPad*)(obj))

typedef struct _GstNvDs3dMixerPad GstNvDs3dMixerPad;
typedef struct _GstNvDs3dMixerPadClass GstNvDs3dMixerPadClass;

struct _GstNvDs3dMixerPad {
    GstPad parent;

    gboolean got_eos;
};

struct _GstNvDs3dMixerPadClass {
    GstPadClass parent;
};

G_END_DECLS
#endif
