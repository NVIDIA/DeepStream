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
 * @file nvdspreprocess_lib.h
 * <b>NVIDIA DeepStream Preprocess lib specifications </b>
 *
 * @b Description: This file defines common elements used in the API
 * exposed by the Gst-nvdspreprocess plugin.
 */

/**
 * @defgroup  gstreamer_nvdspreprocess_api NvDsPreProcess Plugin
 * Defines an API for the GStreamer NvDsPreProcess custom lib.
 * @ingroup custom_gstreamer
 * @{
 */

#ifndef __NVDSPREPROCESS_LIB__
#define __NVDSPREPROCESS_LIB__

#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include "nvdspreprocess_interface.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * custom library initialization function
 */
CustomCtx *initLib(CustomInitParams initparams);

/**
 * custom library deinitialization function
 */
void deInitLib(CustomCtx *ctx);

/**
 * Custom transformation function for group
 */
NvDsPreProcessStatus CustomTransformation(NvBufSurface *in_surf, NvBufSurface *out_surf,
                                          CustomTransformParams &params);

/**
 * Custom Asynchronus group transformation function
 */
NvDsPreProcessStatus CustomAsyncTransformation(NvBufSurface *in_surf, NvBufSurface *out_surf,
                                               CustomTransformParams &params);

/**
 * Custom tensor preparation function for NCHW/NHWC network order
 */
NvDsPreProcessStatus CustomTensorPreparation(CustomCtx *ctx, NvDsPreProcessBatch *batch,
                                             NvDsPreProcessCustomBuf *&buf,
                                             CustomTensorParams &tensorParam,
                                             NvDsPreProcessAcquirer *acquirer);

#ifdef __cplusplus
}
#endif

#endif
