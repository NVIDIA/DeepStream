/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef _GST_NVDSUDP_COMMON_H_
#define _GST_NVDSUDP_COMMON_H_

#include <gst/gst.h>
#include <gio/gio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(err, err_str)                                        \
    do {                                                                \
      if ((err) != cudaSuccess) {                                       \
        GST_WARNING (err_str ", cuda err: %s", cudaGetErrorName (err)); \
        return FALSE;                                                   \
      }                                                                 \
    } while (0)

#define INVALID_STREAM_ID ((rmx_stream_id)-1L)
#define MAX_ST2022_7_STREAMS  2

#define round_up(num, round) \
    ((((num) + (round) - 1) / (round)) * (round))

typedef enum StreamType {
  VIDEO_2110_20_STREAM,
  VIDEO_2110_22_STREAM,
  AUDIO_2110_30_31_STREAM,
  ANCILLARY_2110_40_STREAM,
  APPLICATION_CUSTOM_STREAM
} StreamType;

typedef enum VideoType {
  PROGRESSIVE = 1,
  INTERLACE = 2
} VideoType;

typedef enum MemoryType {
  MEM_TYPE_HOST,
  MEM_TYPE_DEVICE,
  MEM_TYPE_SYSTEM,
  MEM_TYPE_UNKNOWN
} MemoryType;

gboolean
gst_udp_parse_uri (const gchar * uristr, gchar ** host, guint16 * port);

GInetAddress *
gst_udp_resolve_name (gpointer obj, const gchar * address);

#define LEAP_SECONDS               (37)
#define DEFAULT_PTP_SRC            NULL

// Define metadata structure
typedef struct _GstRTPTimestampMeta GstRTPTimestampMeta;

struct _GstRTPTimestampMeta {
  GstMeta meta;
  guint32 rtp_timestamp;
  gboolean leap_seconds_adjusted;
};

// Declare functions
GType gst_rtp_timestamp_meta_api_get_type(void);
const GstMetaInfo *gst_rtp_timestamp_meta_get_info(void);

// Helper functions for adding/getting metadata
GstRTPTimestampMeta *gst_buffer_add_rtp_timestamp_meta(GstBuffer *buffer, guint32 timestamp);
GstRTPTimestampMeta *gst_buffer_get_rtp_timestamp_meta(GstBuffer *buffer);

// GPU memory allocation related functions
void* gpu_allocate_memory (int gpuId, size_t size, size_t align);
size_t gpu_aligned_size (int gpuId, size_t allocSize);
gboolean cudaFreeMmap (uint64_t *ptr, size_t size);

#endif