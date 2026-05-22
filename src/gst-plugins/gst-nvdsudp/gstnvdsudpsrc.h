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

#ifndef _GST_NVDSUDPSRC_H_
#define _GST_NVDSUDPSRC_H_

#include <gst/base/gstpushsrc.h>
#include <gio/gio.h>
#include <rivermax_api.h>

#include "gstnvdsudpcommon.h"

G_BEGIN_DECLS

#define GST_TYPE_NVDSUDPSRC   (gst_nvdsudpsrc_get_type())
#define GST_NVDSUDPSRC(obj)   (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_NVDSUDPSRC,GstNvDsUdpSrc))
#define GST_NVDSUDPSRC_CLASS(klass)   (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_NVDSUDPSRC,GstNvDsUdpSrcClass))
#define GST_IS_NVDSUDPSRC(obj)   (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_NVDSUDPSRC))
#define GST_IS_NVDSUDPSRC_CLASS(obj)   (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_NVDSUDPSRC))

typedef struct _GstNvDsUdpSrc GstNvDsUdpSrc;
typedef struct _GstNvDsUdpSrcClass GstNvDsUdpSrcClass;
typedef struct DstStreamInfo {
  char *ip;
  uint16_t port;
} DstStreamInfo;

struct _GstNvDsUdpSrc
{
  GstPushSrc parent;

  /* properties */
  gchar *uri;
  gchar *localIfaceIps;
  gint port;
  gint buffer_size;
  gchar *multi_iface;
  gboolean auto_multicast;
  GstCaps *caps;
  gchar *address;
  gchar *sourceAddresses;
  GSocket *used_socket;
  guint64 timeout;
  gboolean reuse;
  gboolean loop;
  guint headerSize;
  guint payloadSize;
  guint numPackets;
  guint payMultiple;
  gint gpuId;
  gboolean use_rtp_timestamp;
  gboolean adjust_leap_seconds;
  guint64 latency_offset;  /* Latency offset in nanoseconds */
  gchar *ptpSrc;
  gchar *st2022_7_streams;   /* List of IP:port pairs for ST2022-7 redundant streams */
  guint num_streams;         /* Number of active streams (primary + redundant) */
  guint num_source_addresses; /* Number of source addresses for filtering */
  guint num_local_interfaces; /* Number of local interfaces */

  StreamType streamType;
  VideoType videoType;
  MemoryType outputMemType;
  guint frameSize;
  guint stride;
  uint8_t *dataPtr1;
  uint8_t *dataPtr2;
  guint len1;
  guint len2;
  gboolean mBit;
  gboolean ffFound;
  guint packetCounter;

  GstBufferPool *pool;
  GThread *pThread;
  GQueue *bufferQ;
  GCond qCond;
  GMutex qLock;
  GCond flowCond;
  GMutex flowLock;
  gboolean isRunning;
  gboolean flushing;
  GInetSocketAddress *addr;
  GCancellable *cancellable;
  gint cancellableFd;
  gint pollfd;
  gint lastError;
  gboolean isRtpOut;
  /*For rtpticks based buffer pts/dts generation */
  gboolean first_rtp_packet;
  guint64 last_rtp_tick;
  guint64 rtp_tick_base;
  guint clock_rate;
  gboolean is_nvmm;

  /* Rivermax specific */
  guint flowId;
  struct sockaddr_in localNicAddr;
  rmx_stream_id streamId[MAX_ST2022_7_STREAMS];
  rmx_input_flow flow[MAX_ST2022_7_STREAMS];
  rmx_input_stream_params stream_params;
  rmx_mem_region *data;   /* Pointer to internal memory region in stream_params */
  rmx_mem_region *hdr;    /* Pointer to internal memory region in stream_params */
  rmx_device_iface device;
  rmx_input_chunk_handle chunk_handle[MAX_ST2022_7_STREAMS];
  rmx_mkey_id data_mkey;
  rmx_mkey_id hdr_mkey;
  size_t data_stride_size;
  size_t hdr_stride_size;
  size_t payload_mem_block_id;
  size_t header_mem_block_id;
  gboolean isGpuDirect;
  gpointer dataPtr;
  gpointer hdrPtr;
  size_t alignedMemSize;

  DstStreamInfo dstStream[MAX_ST2022_7_STREAMS];
  gchar *srcAddress[MAX_ST2022_7_STREAMS];
  gchar *localIfaceIp[MAX_ST2022_7_STREAMS];
};

struct _GstNvDsUdpSrcClass
{
  GstPushSrcClass parent_class;
};

GType gst_nvdsudpsrc_get_type (void);

G_END_DECLS

#endif
