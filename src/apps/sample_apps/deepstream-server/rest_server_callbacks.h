/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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