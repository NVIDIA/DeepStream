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

#include "rest_server_callbacks.h"

void
s_appinstance_callback_impl (NvDsServerAppInstanceInfo * appinstance_info, void *ctx)
{
  AppCtx *serverappctx = (AppCtx *) ctx;
  (void) serverappctx;

  if (appinstance_info->uri.find ("/api/v1/") != std::string::npos) {
    if (!s_force_eos_handle (serverappctx->nvmultiurisrcbinCreator,
            appinstance_info)) {
      if (appinstance_info->appinstance_flag == QUIT_APP) {
        appinstance_info->app_log =
          "QUIT_FAIL, Unable to handle force-pipeline-eos nvmultiurisrcbin";
        appinstance_info->status = QUIT_FAIL;
        appinstance_info->err_info.code = StatusInternalServerError;
      }
    } else {
      if (appinstance_info->appinstance_flag == QUIT_APP) {
        appinstance_info->status = QUIT_SUCCESS;
        appinstance_info->err_info.code = StatusOk;
        appinstance_info->app_log = "QUIT_SUCCESS";
        g_print ("appinstance force quit success\n");
      }
    }
  } else {
    g_print("Unsupported REST API version\n");
  }

  return;
}

void
s_osd_callback_impl (NvDsServerOsdInfo * osd_info, void *ctx)
{
  AppCtx *serverappctx = (AppCtx *) ctx;
  (void) serverappctx;

  guint sourceId = std::stoi (osd_info->stream_id);

  if (osd_info->uri.find ("/api/v1/") != std::string::npos) {
    if (!find_source (serverappctx->nvmultiurisrcbinCreator, sourceId)) {
      if (osd_info->osd_flag == PROCESS_MODE) {
        osd_info->status = PROCESS_MODE_UPDATE_FAIL;
        osd_info->err_info.code = StatusInternalServerError;
        osd_info->osd_log = "PROCESS_MODE_UPDATE_FAIL, Unable to find stream id for osd property updation";
      }
    } else {
      GstEvent *nvevent =
          gst_nvevent_osd_process_mode_update ((char *) osd_info->stream_id.
          c_str (), osd_info->process_mode);
      if (!nvevent) {
        osd_info->status = PROCESS_MODE_UPDATE_FAIL;
        osd_info->osd_log = "PROCESS_MODE_UPDATE_FAIL, nv-osd-process-mode-update event creation failed";
        osd_info->err_info.code = StatusInternalServerError;
      }

      if (!gst_pad_push_event ((GstPad
                  *) (gst_nvmultiurisrcbincreator_get_source_pad (serverappctx->
                      nvmultiurisrcbinCreator)), nvevent)) {
        g_print
            ("[WARN] nv-osd-process-mode-update event not pushed downstream.. !! \n");
        osd_info->status = PROCESS_MODE_UPDATE_FAIL;
        osd_info->osd_log = "PROCESS_MODE_UPDATE_FAIL, nv-osd-process-mode-update event not pushed";
        osd_info->err_info.code = StatusInternalServerError;
      } else {
        osd_info->status = PROCESS_MODE_UPDATE_SUCCESS;
        osd_info->err_info.code = StatusOk;
        osd_info->osd_log = "PROCESS_MODE_UPDATE_SUCCESS";
      }

    }
  } else {
    g_print("Unsupported REST API version\n");
  }

  return;
}

void
s_mux_callback_impl (NvDsServerMuxInfo * mux_info, void *ctx)
{
  AppCtx *serverappctx = (AppCtx *) ctx;
  (void) serverappctx;

  if (mux_info->uri.find ("/api/v1/") != std::string::npos) {
    if (!set_nvuribin_mux_prop (serverappctx->nvmultiurisrcbinCreator,mux_info)) {
      switch (mux_info->mux_flag) {
        case BATCHED_PUSH_TIMEOUT:
          g_print ("[WARN] batched-push-timeout update failed .. !! \n");
          mux_info->mux_log = "BATCHED_PUSH_TIMEOUT_UPDATE_FAIL, batched-push-timeout value not updated";
          mux_info->status = BATCHED_PUSH_TIMEOUT_UPDATE_FAIL;
          mux_info->err_info.code = StatusInternalServerError;
          break;
        case MAX_LATENCY:
          g_print ("[WARN] max-latency update failed .. !! \n");
          mux_info->mux_log = "MAX_LATENCY_UPDATE_FAIL, MAX_LATENCY_UPDATE_FAIL, max-latency value not updated";
          mux_info->status = MAX_LATENCY_UPDATE_FAIL;
          mux_info->err_info.code = StatusInternalServerError;
          break;
        default:
          break;
      }
    } else {
      switch (mux_info->mux_flag) {
        case BATCHED_PUSH_TIMEOUT:
          mux_info->status =
              mux_info->status !=
              BATCHED_PUSH_TIMEOUT_UPDATE_FAIL ?
              BATCHED_PUSH_TIMEOUT_UPDATE_SUCCESS :
              BATCHED_PUSH_TIMEOUT_UPDATE_FAIL;
          if ( mux_info->status == BATCHED_PUSH_TIMEOUT_UPDATE_SUCCESS ){
            mux_info->err_info.code = StatusOk;
            mux_info->mux_log = "BATCHED_PUSH_TIMEOUT_UPDATE_SUCCESS";
          } else{
            mux_info->err_info.code = StatusInternalServerError;
            mux_info->mux_log = "BATCHED_PUSH_TIMEOUT_UPDATE_SUCCESS, Error while setting batched-push-timeout property";
          }
          break;
        case MAX_LATENCY:
          mux_info->status =
              mux_info->status !=
              MAX_LATENCY_UPDATE_FAIL ? MAX_LATENCY_UPDATE_SUCCESS :
              MAX_LATENCY_UPDATE_FAIL;
          if ( mux_info->status == MAX_LATENCY_UPDATE_SUCCESS ){
            mux_info->err_info.code = StatusOk;
            mux_info->mux_log = "MAX_LATENCY_UPDATE_SUCCESS";
          } else{
            mux_info->err_info.code = StatusInternalServerError;
            mux_info->mux_log = "MAX_LATENCY_UPDATE_FAIL, Error while setting max-latency property";
          }
          break;
        default:
          break;
      }
    }

    gst_nvmultiurisrcbincreator_sync_children_states (serverappctx->
      nvmultiurisrcbinCreator);
  } else {
    g_print("Unsupported REST API version\n");
  }

  return;
}

void
s_enc_callback_impl (NvDsServerEncInfo * enc_info, void *ctx)
{
  AppCtx *serverappctx = (AppCtx *) ctx;
  (void) serverappctx;
  guint sourceId = std::stoi (enc_info->stream_id);
  GstEvent *nvevent = NULL;

  if (enc_info->uri.find ("/api/v1/") != std::string::npos) {
    if (!find_source (serverappctx->nvmultiurisrcbinCreator, sourceId)) {
      switch (enc_info->enc_flag) {
        case BITRATE:
          enc_info->enc_log = "BITRATE_UPDATE_FAIL, Not able to find source id";
          enc_info->status = BITRATE_UPDATE_FAIL;
          enc_info->err_info.code = StatusInternalServerError;
          break;
        case FORCE_IDR:
          enc_info->enc_log = "FORCE_IDR_UPDATE_FAIL, Not able to find source id";
          enc_info->status = FORCE_IDR_UPDATE_FAIL;
          enc_info->err_info.code = StatusInternalServerError;
          break;
        case FORCE_INTRA:
          enc_info->enc_log = "FORCE_INTRA_UPDATE_FAIL, Not able to find source id";
          enc_info->status = FORCE_INTRA_UPDATE_FAIL;
          enc_info->err_info.code = StatusInternalServerError;
          break;
        case IFRAME_INTERVAL:
          enc_info->enc_log = "IFRAME_INTERVAL_UPDATE_FAIL, Not able to find source id";
          enc_info->status = IFRAME_INTERVAL_UPDATE_FAIL;
          enc_info->err_info.code = StatusInternalServerError;
          break;
        default:
          break;
      }
    } else {
      switch (enc_info->enc_flag) {
        case BITRATE:
          nvevent =
              gst_nvevent_enc_bitrate_update ((char *) enc_info->stream_id.
              c_str (), enc_info->bitrate);
          /* send nv-enc-bitrate-update event */
          if (!gst_pad_push_event ((GstPad
                      *) (gst_nvmultiurisrcbincreator_get_source_pad
                      (serverappctx->nvmultiurisrcbinCreator)), nvevent)) {
            g_print
                ("[WARN] nv-enc-bitrate-update event not pushed downstream.bitrate update failed on encoder.. !! \n");
            enc_info->enc_log =
                !nvevent ? "BITRATE_UPDATE_FAIL, nv-enc-bitrate-update event creation failed" :
                "BITRATE_UPDATE_FAIL, nv-enc-bitrate-update event not pushed";
            enc_info->status = BITRATE_UPDATE_FAIL;
            enc_info->err_info.code = StatusInternalServerError;
          } else {
            enc_info->status = BITRATE_UPDATE_SUCCESS;
            enc_info->err_info.code = StatusOk;
            enc_info->enc_log = "BITRATE_UPDATE_SUCCESS";
          }
          break;
        case FORCE_IDR:
          nvevent =
              gst_nvevent_enc_force_idr ((char *) enc_info->stream_id.c_str (),
              enc_info->force_idr);
          /* send nv-enc-force-idr event */
          if (!gst_pad_push_event ((GstPad
                      *) (gst_nvmultiurisrcbincreator_get_source_pad
                      (serverappctx->nvmultiurisrcbinCreator)), nvevent)) {
            g_print
                ("[WARN] nv-enc-force-idr event not pushed downstream.force IDR frame failed on encoder .. !! \n");
            enc_info->enc_log =
                !nvevent ? "FORCE_IDR_UPDATE_FAIL, nv-enc-force-idr event creation failed" :
                "FORCE_IDR_UPDATE_FAIL, nv-enc-force-idr event not pushed";
            enc_info->status = FORCE_IDR_UPDATE_FAIL;
            enc_info->err_info.code = StatusInternalServerError;
          } else {
            enc_info->status = FORCE_IDR_UPDATE_SUCCESS;
            enc_info->err_info.code = StatusOk;
            enc_info->enc_log = "FORCE_IDR_UPDATE_SUCCESS";
          }
          break;
        case FORCE_INTRA:
          nvevent =
              gst_nvevent_enc_force_intra ((char *) enc_info->stream_id.c_str (),
              enc_info->force_intra);
          /* send nv-enc-force-intra event */
          if (!gst_pad_push_event ((GstPad
                      *) (gst_nvmultiurisrcbincreator_get_source_pad
                      (serverappctx->nvmultiurisrcbinCreator)), nvevent)) {
            g_print
                ("[WARN] nv-enc-force-intra event not pushed downstream.force intra frame failed on encoder .. !! \n");
            enc_info->enc_log =
                !nvevent ? "FORCE_INTRA_UPDATE_FAIL, nv-enc-force-intra event creation failed" :
                "FORCE_INTRA_UPDATE_FAIL, nv-enc-force-intra event not pushed";
            enc_info->status = FORCE_INTRA_UPDATE_FAIL;
            enc_info->err_info.code = StatusInternalServerError;
          } else {
            enc_info->status = FORCE_INTRA_UPDATE_SUCCESS;
            enc_info->err_info.code = StatusOk;
            enc_info->enc_log = "FORCE_INTRA_UPDATE_SUCCESS";
          }
          break;
        case IFRAME_INTERVAL:
          nvevent =
              gst_nvevent_enc_iframeinterval_update ((char *) enc_info->stream_id.
              c_str (), enc_info->iframeinterval);
          /* send nv-enc-iframeinterval-update event */
          if (!gst_pad_push_event ((GstPad
                      *) (gst_nvmultiurisrcbincreator_get_source_pad
                      (serverappctx->nvmultiurisrcbinCreator)), nvevent)) {
            g_print
                ("[WARN] nv-enc-iframeinterval-update event not pushed downstream.iframe interval update failed on encoder .. !! \n");
            enc_info->enc_log =
                !nvevent ? "IFRAME_INTERVAL_UPDATE_FAIL, nv-enc-iframeinterval-update event creation failed" :
                "IFRAME_INTERVAL_UPDATE_FAIL, nv-enc-iframeinterval-update event not pushed";
            enc_info->status = IFRAME_INTERVAL_UPDATE_FAIL;
            enc_info->err_info.code = StatusInternalServerError;
          } else {
            enc_info->status = IFRAME_INTERVAL_UPDATE_SUCCESS;
            enc_info->err_info.code = StatusOk;
            enc_info->enc_log = "IFRAME_INTERVAL_UPDATE_SUCCESS";
          }
          break;
        default:
          break;
      }
    }
    gst_nvmultiurisrcbincreator_sync_children_states (serverappctx->
      nvmultiurisrcbinCreator);
  } else {
    g_print("Unsupported REST API version\n");
  }

  return;
}

void
s_conv_callback_impl (NvDsServerConvInfo * conv_info, void *ctx)
{
  AppCtx *serverappctx = (AppCtx *) ctx;
  (void) serverappctx;

  guint sourceId = std::stoi (conv_info->stream_id);

  if (conv_info->uri.find ("/api/v1/") != std::string::npos) {
    if (!set_nvuribin_conv_prop (serverappctx->nvmultiurisrcbinCreator, sourceId,
            conv_info)) {
      switch (conv_info->conv_flag) {
        case SRC_CROP:
          g_print ("[WARN] source-crop update failed .. !! \n");
          conv_info->conv_log = "SRC_CROP_UPDATE_FAIL, source-crop update failed";
          conv_info->status = SRC_CROP_UPDATE_FAIL;
          conv_info->err_info.code = StatusInternalServerError;
          break;
        case DEST_CROP:
          g_print ("[WARN] source-crop update failed .. !! \n");
          conv_info->conv_log = "DEST_CROP_UPDATE_FAIL, dest-crop update failed";
          conv_info->status = DEST_CROP_UPDATE_FAIL;
          conv_info->err_info.code = StatusInternalServerError;
          break;
        case FLIP_METHOD:
          g_print ("[WARN] flip-method update failed .. !! \n");
          conv_info->conv_log = "FLIP_METHOD_UPDATE_FAIL, flip-method update failed";
          conv_info->status = FLIP_METHOD_UPDATE_FAIL;
          conv_info->err_info.code = StatusInternalServerError;
          break;
        case INTERPOLATION_METHOD:
          g_print ("[WARN] interpolation-method update failed .. !! \n");
          conv_info->conv_log = "INTERPOLATION_METHOD_UPDATE_FAIL, interpolation-method update failed";
          conv_info->status = INTERPOLATION_METHOD_UPDATE_FAIL;
          conv_info->err_info.code = StatusInternalServerError;
          break;
        default:
          break;
      }
    } else {
      switch (conv_info->conv_flag) {
        case SRC_CROP:
          conv_info->status =
              conv_info->status !=
              SRC_CROP_UPDATE_FAIL ? SRC_CROP_UPDATE_SUCCESS :
              SRC_CROP_UPDATE_FAIL;
              if ( conv_info->status == SRC_CROP_UPDATE_SUCCESS ){
                conv_info->err_info.code = StatusOk;
                conv_info->conv_log = "SRC_CROP_UPDATE_SUCCESS";
              } else{
                conv_info->err_info.code = StatusInternalServerError;
                conv_info->conv_log = "SRC_CROP_UPDATE_FAIL, Error while setting src-crop property";
              }
          break;
        case DEST_CROP:
          conv_info->status =
              conv_info->status !=
              DEST_CROP_UPDATE_FAIL ? DEST_CROP_UPDATE_SUCCESS :
              DEST_CROP_UPDATE_FAIL;
          if ( conv_info->status == DEST_CROP_UPDATE_SUCCESS ){
            conv_info->err_info.code = StatusOk;
            conv_info->conv_log = "DEST_CROP_UPDATE_SUCCESS";
          } else{
            conv_info->err_info.code = StatusInternalServerError;
            conv_info->conv_log = "DEST_CROP_UPDATE_FAIL, Error while setting dest-crop property";
          }
          break;
        case FLIP_METHOD:
          conv_info->status =
              conv_info->status !=
              FLIP_METHOD_UPDATE_FAIL ? FLIP_METHOD_UPDATE_SUCCESS :
              FLIP_METHOD_UPDATE_FAIL;
          if ( conv_info->status == FLIP_METHOD_UPDATE_SUCCESS ){
            conv_info->err_info.code = StatusOk;
            conv_info->conv_log = "FLIP_METHOD_UPDATE_SUCCESS";
          } else{
            conv_info->err_info.code = StatusInternalServerError;
            conv_info->conv_log = "FLIP_METHOD_UPDATE_FAIL, Error while setting flip-method property";
          }
          break;
        case INTERPOLATION_METHOD:
          conv_info->status =
              conv_info->status !=
              INTERPOLATION_METHOD_UPDATE_FAIL ?
              INTERPOLATION_METHOD_UPDATE_SUCCESS :
              INTERPOLATION_METHOD_UPDATE_FAIL;
          if ( conv_info->status == INTERPOLATION_METHOD_UPDATE_SUCCESS ){
            conv_info->err_info.code = StatusOk;
            conv_info->conv_log = "INTERPOLATION_METHOD_UPDATE_SUCCESS";
          } else{
            conv_info->err_info.code = StatusInternalServerError;
            conv_info->conv_log = "INTERPOLATION_METHOD_UPDATE_FAIL, Error while setting interpolation-method property";
          }
          break;
        default:
          break;
      }
    }
    gst_nvmultiurisrcbincreator_sync_children_states (serverappctx->
      nvmultiurisrcbinCreator);
  } else {
    g_print("Unsupported REST API version\n");
  }

  return;
}

void
s_inferserver_callback_impl (NvDsServerInferServerInfo * inferserver_info, void *ctx)
{
  AppCtx *serverappctx = (AppCtx *) ctx;
  (void) serverappctx;

  guint sourceId = std::stoi (inferserver_info->stream_id);

  if (inferserver_info->uri.find ("/api/v1/") != std::string::npos) {
    if (!find_source (serverappctx->nvmultiurisrcbinCreator, sourceId)) {
      if (inferserver_info->inferserver_flag == INFERSERVER_INTERVAL) {
        inferserver_info->status = INFERSERVER_INTERVAL_UPDATE_FAIL;
        inferserver_info->err_info.code = StatusInternalServerError;
        inferserver_info->inferserver_log = "INFERSERVER_INTERVAL_UPDATE_FAIL, Unable to find stream id for infer (inferserver) property updation";
      }
    } else {
      GstEvent *nvevent =
          gst_nvevent_infer_interval_update ((char *) inferserver_info->stream_id.
          c_str (), inferserver_info->interval);
      if (!nvevent) {
        inferserver_info->status = INFERSERVER_INTERVAL_UPDATE_FAIL;
        inferserver_info->inferserver_log =
            "INFERSERVER_INTERVAL_UPDATE_FAIL, nv-infer-interval-update event (inferserver) creation failed";
        inferserver_info->err_info.code = StatusInternalServerError;
      }

      /* send nv-infer-interval-update event */
      if (!gst_pad_push_event ((GstPad
                  *) (gst_nvmultiurisrcbincreator_get_source_pad (serverappctx->
                      nvmultiurisrcbinCreator)), nvevent)) {
        g_print
            ("[WARN] nv-infer-interval-update (inferserver) event not pushed downstream.. !! \n");
        inferserver_info->status = INFERSERVER_INTERVAL_UPDATE_FAIL;
        inferserver_info->inferserver_log =
            "INFERSERVER_INTERVAL_UPDATE_FAIL, nv-infer-interval-update event not pushed";
        inferserver_info->err_info.code = StatusInternalServerError;
      } else {
        inferserver_info->status = INFERSERVER_INTERVAL_UPDATE_SUCCESS;
        inferserver_info->err_info.code = StatusOk;
        inferserver_info->inferserver_log = "INFERSERVER_INTERVAL_UPDATE_SUCCESS";
      }
      gst_nvmultiurisrcbincreator_sync_children_states (serverappctx->
          nvmultiurisrcbinCreator);
    }
  } else {
    g_print("Unsupported REST API version\n");
  }

  return;
}

void
s_infer_callback_impl (NvDsServerInferInfo * infer_info, void *ctx)
{
  AppCtx *serverappctx = (AppCtx *) ctx;
  (void) serverappctx;

  guint sourceId = std::stoi (infer_info->stream_id);

  if (infer_info->uri.find ("/api/v1/") != std::string::npos) {
    if (!find_source (serverappctx->nvmultiurisrcbinCreator, sourceId)) {
      infer_info->status = INFER_INTERVAL_UPDATE_FAIL;
      infer_info->err_info.code = StatusInternalServerError;
      infer_info->infer_log = "INFER_INTERVAL_UPDATE_FAIL, Unable to find stream id for infer property updation";
    } else {
      GstEvent *nvevent =
          gst_nvevent_infer_interval_update ((char *) infer_info->stream_id.
          c_str (), infer_info->interval);
      if (!nvevent) {
        infer_info->status = INFER_INTERVAL_UPDATE_FAIL;
        infer_info->infer_log = "INFER_INTERVAL_UPDATE_FAIL, nv-infer-interval-update event creation failed";
        infer_info->err_info.code = StatusInternalServerError;
      }
      /* send nv-infer-interval-update event */
      if (!gst_pad_push_event ((GstPad
                  *) (gst_nvmultiurisrcbincreator_get_source_pad (serverappctx->
                      nvmultiurisrcbinCreator)), nvevent)) {
        g_print
            ("[WARN] nv-infer-interval-update event not pushed downstream.. !! \n");
        infer_info->status = INFER_INTERVAL_UPDATE_FAIL;
        infer_info->infer_log = "INFER_INTERVAL_UPDATE_FAIL, nv-infer-interval-update event not pushed";
        infer_info->err_info.code = StatusInternalServerError;
      } else {
        infer_info->status = INFER_INTERVAL_UPDATE_SUCCESS;
        infer_info->infer_log = "INFER_INTERVAL_UPDATE_SUCCESS";
        infer_info->err_info.code = StatusOk;
      }
    }
    gst_nvmultiurisrcbincreator_sync_children_states (serverappctx->
      nvmultiurisrcbinCreator);
  } else {
    g_print("Unsupported REST API version\n");
  }

  return;
}

void
s_dec_callback_impl (NvDsServerDecInfo * dec_info, void *ctx)
{
  AppCtx *serverappctx = (AppCtx *) ctx;
  (void) serverappctx;

  guint sourceId = std::stoi (dec_info->stream_id);

  if (dec_info->uri.find ("/api/v1/") != std::string::npos) {
    if (!set_nvuribin_dec_prop (serverappctx->nvmultiurisrcbinCreator, sourceId,
            dec_info)) {
      switch (dec_info->dec_flag) {
        case DROP_FRAME_INTERVAL:
          g_print ("[WARN] drop-frame-interval not set on decoder .. !! \n");
          dec_info->status = DROP_FRAME_INTERVAL_UPDATE_FAIL;
          dec_info->dec_log = "DROP_FRAME_INTERVAL_UPDATE_FAIL, drop-frame-interval not set on decoder";
          dec_info->err_info.code = StatusInternalServerError;
          break;
        case SKIP_FRAMES:
          g_print ("[WARN] skip-frame not set on decoder .. !! \n");
          dec_info->status = SKIP_FRAMES_UPDATE_FAIL;
          dec_info->dec_log = "SKIP_FRAMES_UPDATE_FAIL, skip-frame not set on decoder";
          dec_info->err_info.code = StatusInternalServerError;
          break;
        case LOW_LATENCY_MODE:
          g_print ("[WARN] low-latency-mode not set on decoder .. !! \n");
          dec_info->status = LOW_LATENCY_MODE_UPDATE_FAIL;
          dec_info->dec_log = "LOW_LATENCY_MODE_UPDATE_FAIL, low-latency-mode not set on decoder";
          dec_info->err_info.code = StatusInternalServerError;
          break;
        default:
          break;
      }
    } else {
      switch (dec_info->dec_flag) {
        case DROP_FRAME_INTERVAL:
          dec_info->status =
              dec_info->status !=
              DROP_FRAME_INTERVAL_UPDATE_FAIL ? DROP_FRAME_INTERVAL_UPDATE_SUCCESS
              : DROP_FRAME_INTERVAL_UPDATE_FAIL;
          if ( dec_info->status == DROP_FRAME_INTERVAL_UPDATE_SUCCESS ){
            dec_info->err_info.code = StatusOk;
            dec_info->dec_log = "DROP_FRAME_INTERVAL_UPDATE_SUCCESS";
          } else{
            dec_info->err_info.code = StatusInternalServerError;
            dec_info->dec_log = "DROP_FRAME_INTERVAL_UPDATE_FAIL, Error while setting drop-frame-interval property";
          }
          break;
        case SKIP_FRAMES:
          dec_info->status =
              dec_info->status !=
              SKIP_FRAMES_UPDATE_FAIL ? SKIP_FRAMES_UPDATE_SUCCESS :
              SKIP_FRAMES_UPDATE_FAIL;
          if ( dec_info->status == SKIP_FRAMES_UPDATE_SUCCESS ){
            dec_info->err_info.code = StatusOk;
            dec_info->dec_log = "SKIP_FRAMES_UPDATE_SUCCESS";
          } else{
            dec_info->err_info.code = StatusInternalServerError;
            dec_info->dec_log = "SKIP_FRAMES_UPDATE_FAIL, Error while setting skip-frame property";
          }
          break;
        case LOW_LATENCY_MODE:
          dec_info->status =
              dec_info->status !=
              LOW_LATENCY_MODE_UPDATE_FAIL ? LOW_LATENCY_MODE_UPDATE_SUCCESS :
              LOW_LATENCY_MODE_UPDATE_FAIL;
          if ( dec_info->status == LOW_LATENCY_MODE_UPDATE_SUCCESS ){
            dec_info->err_info.code = StatusOk;
            dec_info->dec_log = "LOW_LATENCY_MODE_UPDATE_SUCCESS";
          } else{
            dec_info->err_info.code = StatusInternalServerError;
            dec_info->dec_log = "LOW_LATENCY_MODE_UPDATE_FAIL, Error while setting skip-frame property";
          }
          break;
        default:
          break;
      }
    }

    gst_nvmultiurisrcbincreator_sync_children_states (serverappctx->
      nvmultiurisrcbinCreator);
  } else {
    g_print("Unsupported REST API version\n");
  }

  return;
}

void
s_roi_callback_impl (NvDsServerRoiInfo * roi_info, void *ctx)
{
  AppCtx *serverappctx = (AppCtx *) ctx;
  (void) serverappctx;

  guint sourceId = std::stoi (roi_info->stream_id);

  if (roi_info->uri.find ("/api/v1/") != std::string::npos) {
    if (!find_source (serverappctx->nvmultiurisrcbinCreator, sourceId)) {
      roi_info->status = ROI_UPDATE_FAIL;
      roi_info->err_info.code = StatusInternalServerError;
      roi_info->roi_log = "ROI_UPDATE_FAIL, Unable to find stream id for ROI updation";
    } else {
      RoiDimension roi_dim[roi_info->roi_count];

      for (int i = 0; i < (int) roi_info->roi_count; i++) {
        g_strlcpy (roi_dim[i].roi_id, roi_info->vect[i].roi_id,
            sizeof (roi_dim[i].roi_id));
        roi_dim[i].left = roi_info->vect[i].left;
        roi_dim[i].top = roi_info->vect[i].top;
        roi_dim[i].width = roi_info->vect[i].width;
        roi_dim[i].height = roi_info->vect[i].height;
      }

      GstEvent *nvevent =
          gst_nvevent_new_roi_update ((char *) roi_info->stream_id.c_str (),
          roi_info->roi_count, roi_dim);

      if (!nvevent) {
        roi_info->roi_log = "ROI_UPDATE_FAIL, nv-roi-update event creation failed";
        roi_info->status = ROI_UPDATE_FAIL;
        roi_info->err_info.code = StatusInternalServerError;
      }
      /* send nv-new_roi_update event */
      if (!gst_pad_push_event ((GstPad
                  *) (gst_nvmultiurisrcbincreator_get_source_pad (serverappctx->
                      nvmultiurisrcbinCreator)), nvevent)) {
        switch (roi_info->roi_flag) {
          case ROI_UPDATE:
            g_print ("[WARN] ROI UPDATE event not pushed downstream.. !! \n");
            roi_info->roi_log = "ROI_UPDATE_FAIL, nv-roi-update event not pushed";
            roi_info->status = ROI_UPDATE_FAIL;
            roi_info->err_info.code = StatusInternalServerError;
            break;
          default:
            break;
        }
      } else {
        switch (roi_info->roi_flag) {
          case ROI_UPDATE:
            roi_info->status = ROI_UPDATE_SUCCESS;
            roi_info->err_info.code = StatusOk;
            roi_info->roi_log = "ROI_UPDATE_SUCCESS";
            break;
          default:
            break;
        }
      }
    }
    gst_nvmultiurisrcbincreator_sync_children_states (serverappctx->
      nvmultiurisrcbinCreator);
  } else {
    g_print ("Unsupported REST API version\n");
  }

  return;
}

void
s_stream_callback_impl (NvDsServerStreamInfo * stream_info, void *ctx)
{
  AppCtx *serverappctx = (AppCtx *) ctx;
  (void) serverappctx;

  g_print ("Inside s_stream_callback_impl callback +++ \n");

  g_mutex_lock (&serverappctx->bincreator_lock);

  if (stream_info->uri.find ("/api/v1/") != std::string::npos) {
    /* check stream_info->value_change to identify stream add/remove */
    if (g_strrstr (stream_info->value_change.c_str (), "add")) {
      g_print ("stream_info->value_change is stream add \n");

      GstDsNvUriSrcConfig **sourceConfigs = NULL;
      guint numSourceConfigs = 0;
      /** Check if we can accomodate more sources */
      if (gst_nvmultiurisrcbincreator_get_active_sources_list (serverappctx->
              nvmultiurisrcbinCreator, &numSourceConfigs, &sourceConfigs)) {
        gst_nvmultiurisrcbincreator_src_config_list_free (serverappctx->
            nvmultiurisrcbinCreator, numSourceConfigs, sourceConfigs);
        if (numSourceConfigs >= serverappctx->muxConfig.maxBatchSize) {
          g_print ("Failed to add sensor id=[%s]; "
              "We have [%d] active sources and max-batch-size is configured to [%d]\n",
              stream_info->value_camera_id.c_str (),
              numSourceConfigs, serverappctx->muxConfig.maxBatchSize);
          stream_info->status = STREAM_ADD_FAIL;
          stream_info->stream_log = "STREAM_ADD_FAIL, Active sources exceded max-batch-size of nvstreammux";
          stream_info->err_info.code = StatusInternalServerError;
          g_mutex_unlock (&serverappctx->bincreator_lock);
          return;
        }
      }
      g_print ("we can accomodate more sources\n");

      /** Check if sensor id already exist */
      GstDsNvUriSrcConfig *sourceConfig = NULL;
      if ((sourceConfig =
              gst_nvmultiurisrcbincreator_get_source_config_by_sensorid
              (serverappctx->nvmultiurisrcbinCreator,
                  stream_info->value_camera_id.c_str ()))) {
        g_print ("Failed to add sensor id=[%s]; Already added\n",
            stream_info->value_camera_id.c_str ());
        stream_info->status = STREAM_ADD_FAIL;
        stream_info->stream_log = "STREAM_ADD_FAIL, Duplicate Camera id, unable to add stream";
        stream_info->err_info.code = StatusInternalServerError;
        g_mutex_unlock (&serverappctx->bincreator_lock);
        gst_nvmultiurisrcbincreator_src_config_free (sourceConfig);
        return;
      }
      g_print ("sensor id don't exist \n");

      /** Add the source */
      serverappctx->config.uri = (gchar *) stream_info->value_camera_url.c_str ();
      serverappctx->config.sensorId =
          (gchar *) stream_info->value_camera_id.c_str ();
      serverappctx->config.sensorName =
          (gchar *) stream_info->value_camera_name.c_str ();
      serverappctx->config.source_id = ++serverappctx->sourceIdCounter;

      g_print ("Adding source now \n");
      gboolean ret =
          gst_nvmultiurisrcbincreator_add_source (serverappctx->
          nvmultiurisrcbinCreator, &serverappctx->config);
      if (ret == FALSE) {
        g_print ("Failed to add sensor id=[%s] uri=[%s]\n",
            serverappctx->config.sensorId, serverappctx->config.uri);
        stream_info->status = STREAM_ADD_FAIL;
        stream_info->stream_log = "STREAM_ADD_FAIL, Failed to add source stream";
        stream_info->err_info.code = StatusInternalServerError;
      } else {
        g_print ("Successfully added sensor id=[%s] uri=[%s]\n",
            serverappctx->config.sensorId, serverappctx->config.uri);
        stream_info->status = STREAM_ADD_SUCCESS;
        stream_info->err_info.code = StatusOk;
        stream_info->stream_log = "STREAM_ADD_SUCCESS";
      }
      gst_nvmultiurisrcbincreator_sync_children_states (serverappctx->
          nvmultiurisrcbinCreator);
      /** clean the config place-holders that we change for each source */
      serverappctx->config.uri = NULL;
      serverappctx->config.sensorId = NULL;

    } else if (g_strrstr (stream_info->value_change.c_str (), "remove")) {
      g_print ("stream_info->value_change is stream remove \n");
      /* First, find the GstDsNvUriSrcConfig object from nvmultiurisrcbinCreator
         for the provided sensorId and uri */
      GstDsNvUriSrcConfig const *sourceConfig =
          gst_nvmultiurisrcbincreator_get_source_config (serverappctx->
          nvmultiurisrcbinCreator,
          stream_info->value_camera_url.c_str (),
          stream_info->value_camera_id.c_str ());
      if (sourceConfig) {
        /* Remove the source */
        gboolean ret =
            gst_nvmultiurisrcbincreator_remove_source (serverappctx->
            nvmultiurisrcbinCreator,
            sourceConfig->source_id);
        /* NOTE: after call to gst_nvmultiurisrcbincreator_remove_source, sourceConfig will be invalid */
        gst_nvmultiurisrcbincreator_src_config_free((GstDsNvUriSrcConfig*)sourceConfig);
        if (ret == FALSE) {
          g_print ("Failed to remove sensor\n");
          stream_info->status = STREAM_REMOVE_FAIL;
          stream_info->stream_log = "STREAM_REMOVE_FAIL,Failed to remove source stream";
          stream_info->err_info.code = StatusInternalServerError;
        } else {
          g_print ("Successfully removed sensor\n");
          stream_info->status = STREAM_REMOVE_SUCCESS;
          stream_info->err_info.code = StatusOk;
          stream_info->stream_log = "STREAM_REMOVE_SUCCESS";
          gst_nvmultiurisrcbincreator_sync_children_states (serverappctx->
              nvmultiurisrcbinCreator);
        }
      } else {
        g_print ("No record found; Failed to remove sensor id=[%s] uri=[%s]\n",
            stream_info->value_camera_id.c_str (),
            stream_info->value_camera_url.c_str ());
      }
    } else {
      g_print ("stream_info->value_change string not supported \n");
      stream_info->stream_log = "Sensor API change string not supported";
      stream_info->err_info.code = StatusBadRequest;
    }
  } else {
    g_print("Unsupported REST API version\n");
  }
  g_mutex_unlock (&serverappctx->bincreator_lock);
  g_print ("Exiting s_stream_callback_impl callback \n");
  return;
}
