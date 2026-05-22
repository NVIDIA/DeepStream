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

#include "nvds_rest_server.h"
#include "gst-nvmultiurisrcbincreator.h"
#include "gst-nvcustomevent.h"

#include "nvds_appctx_server.h"

/* Callback to handle application related REST API requests*/
void s_appinstance_callback_impl (NvDsServerAppInstanceInfo * appinstance_info,
    void *ctx);

/* Callback to handle osd related REST API requests*/
void s_osd_callback_impl (NvDsServerOsdInfo * osd_info, void *ctx);

/* Callback to handle nvstreammux related REST API requests*/
void s_mux_callback_impl (NvDsServerMuxInfo * mux_info, void *ctx);

/* Callback to handle encoder specific REST API requests*/
void s_enc_callback_impl (NvDsServerEncInfo * enc_info, void *ctx);

/* Callback to handle encoder specific REST API requests*/
void s_conv_callback_impl (NvDsServerConvInfo * conv_info, void *ctx);

/* Callback to handle nvinferserver specific REST API requests*/
void s_inferserver_callback_impl (NvDsServerInferServerInfo * inferserver_info,
    void *ctx);

/* Callback to handle nvinfer specific REST API requests*/
void s_infer_callback_impl (NvDsServerInferInfo * infer_info, void *ctx);

/* Callback to handle nvv4l2decoder specific REST API requests*/
void s_dec_callback_impl (NvDsServerDecInfo * dec_info, void *ctx);

/* Callback to handle nvdspreprocess specific REST API requests*/
void s_roi_callback_impl (NvDsServerRoiInfo * roi_info, void *ctx);

/* Callback to handle stream add/remove specific REST API requests*/
void s_stream_callback_impl (NvDsServerStreamInfo * stream_info, void *ctx);