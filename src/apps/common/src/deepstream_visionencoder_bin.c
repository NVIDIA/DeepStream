/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * Vision encoder bin: SGIE-style separate bin with internal drop of empty/invalid
 * batches so nvvidconv_visionencoder_in is never called for removed-source buffers.
 * No app-level probe callbacks; cleanup is handled inside the bin (like SGIE).
 */

#include <string.h>
#include <ctype.h>
#include <dlfcn.h>
#include "deepstream_common.h"
#include "deepstream_visionencoder.h"
#include "gstnvdsmeta.h"
#include "nvbufsurface.h"

/**
 * Detect if the board is Thor using nvidia-ml (NVML) library.
 * Uses dlopen/dlsym to avoid link-time dependency (same pattern as nvds_obj_encode_common.c).
 * Thor requires VIC for hardware-accelerated transforms; all other platforms use GPU compute.
 *
 * @param gpu_id  GPU device index to query
 * @return 1 if board name contains "Thor", 0 otherwise
 */
static int
detect_is_thor (unsigned int gpu_id)
{
  typedef int (*nvmlInit_fn_t)(void);
  typedef int (*nvmlShutdown_fn_t)(void);
  typedef int (*nvmlDeviceGetHandleByIndex_fn_t)(unsigned int, void **);
  typedef int (*nvmlDeviceGetName_fn_t)(void *, char *, unsigned int);

  int is_thor = 0;
  void *nvml_lib = dlopen ("libnvidia-ml.so", RTLD_LAZY);
  if (!nvml_lib)
    nvml_lib = dlopen ("libnvidia-ml.so.1", RTLD_LAZY);
  if (!nvml_lib)
    return 0;

  nvmlInit_fn_t init_fn =
      (nvmlInit_fn_t) dlsym (nvml_lib, "nvmlInit");
  nvmlShutdown_fn_t shutdown_fn =
      (nvmlShutdown_fn_t) dlsym (nvml_lib, "nvmlShutdown");
  nvmlDeviceGetHandleByIndex_fn_t getHandle_fn =
      (nvmlDeviceGetHandleByIndex_fn_t) dlsym (nvml_lib, "nvmlDeviceGetHandleByIndex");
  nvmlDeviceGetName_fn_t getName_fn =
      (nvmlDeviceGetName_fn_t) dlsym (nvml_lib, "nvmlDeviceGetName");

  if (init_fn && shutdown_fn && getHandle_fn && getName_fn) {
    if (init_fn () == 0) {          /* NVML_SUCCESS */
      void *device = NULL;
      if (getHandle_fn (gpu_id, &device) == 0) {
        char name[96] = {0};       /* NVML_DEVICE_NAME_V2_BUFFER_SIZE */
        if (getName_fn (device, name, sizeof (name)) == 0) {
          /* Case-insensitive search for "thor" in board name */
          char name_lower[96];
          for (int i = 0; i < 95 && name[i]; i++)
            name_lower[i] = tolower ((unsigned char) name[i]);
          name_lower[95] = '\0';
          if (strstr (name_lower, "thor") != NULL) {
            is_thor = 1;
          }
        }
      }
      shutdown_fn ();
    }
  }

  dlclose (nvml_lib);
  return is_thor;
}

/* SGIE-style: drop buffers with no batch_meta or empty frame_meta_list so
 * nvvidconv_in is never invoked for them (same idea as nvinfer when num_filled==0). */
static GstPadProbeReturn
visionencoder_bin_drop_empty_batch_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
  GstBuffer *buf = (GstBuffer *) info->data;
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);
  (void) pad;
  (void) u_data;
  if (!batch_meta || !batch_meta->frame_meta_list)
    return GST_PAD_PROBE_DROP;
  return GST_PAD_PROBE_OK;
}

gboolean
create_visionencoder_bin (NvDsVisionEncoderConfig *config, NvDsVisionEncoderBin *bin)
{
  gboolean ret = FALSE;
  GstElement *ve_drop_identity = NULL;
  GstElement *nvvidconv_in = NULL;
  GstElement *nvvidconv_out = NULL;
  GstPad *drop_sink_pad = NULL;
  GstCaps *rgb_caps = NULL;
  const char *format_str = NULL;
  gchar *caps_str = NULL;

  /* Detect platform using NVML board name.
   * Thor → VIC (hardware-accelerated transforms, RGBA, SURFACE_ARRAY)
   * Everything else → GPU compute (RGB, CUDA unified memory) */
  int is_thor = detect_is_thor (config->gpu_id);
  bin->bin = gst_bin_new ("visionencoder_bin");
  if (!bin->bin) {
    NVGSTDS_ERR_MSG_V ("Failed to create 'visionencoder_bin'");
    goto done;
  }

  bin->queue = gst_element_factory_make (NVDS_ELEM_QUEUE, "queue_visionencoder");
  if (!bin->queue) {
    NVGSTDS_ERR_MSG_V ("Failed to create 'queue_visionencoder'");
    goto done;
  }

  /* SGIE-style: identity + probe inside bin to drop empty batches before nvvidconv_in */
  ve_drop_identity = gst_element_factory_make (NVDS_ELEM_IDENTITY, "ve_drop_empty_batch");
  if (!ve_drop_identity) {
    NVGSTDS_ERR_MSG_V ("Failed to create 've_drop_empty_batch' identity");
    goto done;
  }
  drop_sink_pad = gst_element_get_static_pad (ve_drop_identity, "sink");
  if (!drop_sink_pad) {
    NVGSTDS_ERR_MSG_V ("Failed to get sink pad of ve_drop_empty_batch");
    goto done;
  }
  gst_pad_add_probe (drop_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
      visionencoder_bin_drop_empty_batch_probe, NULL, NULL);
  gst_object_unref (drop_sink_pad);

  nvvidconv_in = gst_element_factory_make (NVDS_ELEM_VIDEO_CONV, "nvvidconv_visionencoder_in");
  if (!nvvidconv_in) {
    NVGSTDS_ERR_MSG_V ("Failed to create 'nvvidconv_visionencoder_in'");
    goto done;
  }
  /* Thor: VIC + surface array; everything else: GPU compute + CUDA unified */
  if (is_thor) {
    g_object_set (G_OBJECT (nvvidconv_in),
                  "nvbuf-memory-type", 4,  /* NVBUF_MEM_SURFACE_ARRAY */
                  "compute-hw", 0,  /* VIC */
                  NULL);
  } else {
    g_object_set (G_OBJECT (nvvidconv_in),
                  "nvbuf-memory-type", 3,  /* CUDA unified */
                  "compute-hw", 1,  /* GPU */
                  NULL);
  }

  bin->visionencoder = gst_element_factory_make ("nvdsvisionencoder", "nvdsvisionencoder0");
  if (!bin->visionencoder) {
    NVGSTDS_ERR_MSG_V ("Failed to create 'nvdsvisionencoder' element");
    goto done;
  }
  g_object_set (G_OBJECT (bin->visionencoder),
      "model-variant", config->model_variant,
      "batch-size", config->batch_size,
      "device", config->device,
      "min-crop-size", config->min_crop_size,
      "verbose", config->verbose,
      "backend", config->backend,
      "triton-url", config->triton_url,
      "triton-model", config->triton_model,
      "skip-interval", config->skip_interval,
      "embedding-classes", config->embedding_classes,
      NULL);
  if (config->onnx_model)
    g_object_set (G_OBJECT (bin->visionencoder), "onnx-model", config->onnx_model, NULL);
  if (config->tensorrt_engine)
    g_object_set (G_OBJECT (bin->visionencoder), "tensorrt-engine", config->tensorrt_engine, NULL);

  nvvidconv_out = gst_element_factory_make (NVDS_ELEM_VIDEO_CONV, "nvvidconv_visionencoder_out");
  if (!nvvidconv_out) {
    NVGSTDS_ERR_MSG_V ("Failed to create 'nvvidconv_visionencoder_out'");
    goto done;
  }
  /* Thor: surface array memory; everything else: CUDA unified */
  if (is_thor) {
    g_object_set (G_OBJECT (nvvidconv_out),
                  "nvbuf-memory-type", 4,  /* NVBUF_MEM_SURFACE_ARRAY */
                  NULL);
  } else {
    g_object_set (G_OBJECT (nvvidconv_out), "nvbuf-memory-type", 3, NULL);  /* CUDA unified */
  }

  gst_bin_add_many (GST_BIN (bin->bin), bin->queue, ve_drop_identity,
      nvvidconv_in, bin->visionencoder, nvvidconv_out, NULL);

  NVGSTDS_LINK_ELEMENT (bin->queue, ve_drop_identity);
  NVGSTDS_LINK_ELEMENT (ve_drop_identity, nvvidconv_in);

  /* Thor: RGBA for VIC performance; everything else: RGB */
  format_str = is_thor ? "RGBA" : "RGB";
  caps_str = g_strdup_printf ("video/x-raw(memory:NVMM), format=%s", format_str);
  rgb_caps = gst_caps_from_string (caps_str);
  g_free (caps_str);
  if (!gst_element_link_filtered (nvvidconv_in, bin->visionencoder, rgb_caps)) {
    NVGSTDS_ERR_MSG_V ("Failed to link nvvidconv_in to visionencoder with %s caps", format_str);
    gst_caps_unref (rgb_caps);
    goto done;
  }
  gst_caps_unref (rgb_caps);
  rgb_caps = NULL;

  NVGSTDS_LINK_ELEMENT (bin->visionencoder, nvvidconv_out);

  NVGSTDS_BIN_ADD_GHOST_PAD (bin->bin, bin->queue, "sink");
  NVGSTDS_BIN_ADD_GHOST_PAD (bin->bin, nvvidconv_out, "src");

  ret = TRUE;
done:
  if (!ret)
    NVGSTDS_ERR_MSG_V ("%s failed", __func__);
  if (rgb_caps)
    gst_caps_unref (rgb_caps);
  return ret;
}
