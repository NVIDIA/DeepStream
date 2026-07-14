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

#ifndef _GST_NVDSUDP_COMMON_H_
#define _GST_NVDSUDP_COMMON_H_

#include <gst/gst.h>
#include <gio/gio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string.h>

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

/*
 * RFC 8331 / ST 2110-40 ancillary data constants
 */
#define RTP_HEADER_SIZE            (12)
#define ST2110_40_CLOCK_RATE       (90000)
#define RFC8331_PAYLOAD_HDR_SIZE   (8)
#define RTP_ST2110_40_HEADER_SIZE  (RTP_HEADER_SIZE + RFC8331_PAYLOAD_HDR_SIZE)
#define ANC_MAX_DATA_COUNT         (255)
#define ANC_MAX_PACKET_SIZE        (328)

typedef struct AncPacketInfo {
  gboolean c_not_y_channel_flag;
  guint16 line_number;
  guint16 horizontal_offset;
  guint16 did;
  guint16 sdid;
  guint8  data_count;
  guint16 user_data[ANC_MAX_DATA_COUNT];
  guint16 checksum;
} AncPacketInfo;

/*
 * Bit-level I/O helpers for RFC 8331 bit-packed ANC data
 */
typedef struct {
  const guint8 *data;
  gsize byte_len;
  gsize bit_pos;
} BitReader;

typedef struct {
  guint8 *data;
  gsize byte_len;
  gsize bit_pos;
} BitWriter;

static inline void
bit_reader_init (BitReader *br, const guint8 *data, gsize byte_len)
{
  br->data = data;
  br->byte_len = byte_len;
  br->bit_pos = 0;
}

static inline gboolean
bit_reader_has_bits (const BitReader *br, guint n)
{
  return (br->bit_pos + n) <= (br->byte_len * 8);
}

static inline guint32
bit_reader_read (BitReader *br, guint n)
{
  guint32 val = 0;
  for (guint i = 0; i < n; i++) {
    gsize byte_idx = br->bit_pos / 8;
    guint bit_idx = 7 - (br->bit_pos % 8);
    val = (val << 1) | ((br->data[byte_idx] >> bit_idx) & 1);
    br->bit_pos++;
  }
  return val;
}

static inline gboolean
bit_reader_is_byte_aligned (const BitReader *br)
{
  return (br->bit_pos % 8) == 0;
}

static inline gsize
bit_reader_byte_pos (const BitReader *br)
{
  return (br->bit_pos + 7) / 8;
}

static inline void
bit_writer_init (BitWriter *bw, guint8 *data, gsize byte_len)
{
  bw->data = data;
  bw->byte_len = byte_len;
  bw->bit_pos = 0;
  memset (data, 0, byte_len);
}

static inline void
bit_writer_init_at (BitWriter *bw, guint8 *data, gsize byte_len, gsize start_bit)
{
  bw->data = data;
  bw->byte_len = byte_len;
  bw->bit_pos = start_bit;
}

static inline gboolean
bit_writer_has_space (const BitWriter *bw, guint n)
{
  return (bw->bit_pos + n) <= (bw->byte_len * 8);
}

static inline void
bit_writer_write (BitWriter *bw, guint n, guint32 val)
{
  for (guint i = 0; i < n; i++) {
    gsize byte_idx = bw->bit_pos / 8;
    guint bit_idx = 7 - (bw->bit_pos % 8);
    if (val & (1u << (n - 1 - i)))
      bw->data[byte_idx] |= (1u << bit_idx);
    bw->bit_pos++;
  }
}

static inline gboolean
bit_writer_is_byte_aligned (const BitWriter *bw)
{
  return (bw->bit_pos % 8) == 0;
}

static inline gsize
bit_writer_byte_pos (const BitWriter *bw)
{
  return (bw->bit_pos + 7) / 8;
}

static inline void
bit_writer_pad_to_byte (BitWriter *bw, guint8 pad_val)
{
  while (!bit_writer_is_byte_aligned (bw)) {
    bit_writer_write (bw, 1, pad_val);
  }
}

static inline void
bit_writer_pad_to_u32 (BitWriter *bw)
{
  while (bw->bit_pos % 32 != 0) {
    bit_writer_write (bw, 1, 0);
  }
}

#endif