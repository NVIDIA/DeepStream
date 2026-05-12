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

#ifndef _GST_NVDSUDPSINK_H_
#define _GST_NVDSUDPSINK_H_

#include <gst/base/gstbasesink.h>
#include <gst/base/gstadapter.h>
#include <rivermax_api.h>
#include <gio/gio.h>

#include "../gstnvdsudpcommon.h"

G_BEGIN_DECLS

#define GST_TYPE_NVDSUDPSINK   (gst_nvdsudpsink_get_type())
#define GST_NVDSUDPSINK(obj)   (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_NVDSUDPSINK,GstNvDsUdpSink))
#define GST_NVDSUDPSINK_CLASS(klass)   (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_NVDSUDPSINK,GstNvDsUdpSinkClass))
#define GST_IS_NVDSUDPSINK(obj)   (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_NVDSUDPSINK))
#define GST_IS_NVDSUDPSINK_CLASS(obj)   (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_NVDSUDPSINK))

typedef struct _GstNvDsUdpSink GstNvDsUdpSink;
typedef struct _GstNvDsUdpSinkClass GstNvDsUdpSinkClass;

typedef struct StreamParams {
  guint width;
  guint height;
  guint packetsPerFrame;
  guint chunksPerFrame;
  guint chunksPerMemblock;
  guint framesPerMemblock;
  guint payloadStride; // payload stride size.
  guint headerStride; // header stride size.
  guint chunkSize;  // Number of packets in a chunk.
  guint sampleRate;
  guint depth;
  guint audioChannels;
  guint64 ptime;   // duration of audio in a packet in nanoseconds.
  gdouble frameTimeInterval; // Time interval of frame in nanoseconds.
  gdouble firstPacketTime;
  guint64 pass_rtp_ts_offset;
  gdouble fps;
  StreamType streamType;
  VideoType videoType;
  gchar *format;
  guint chunkNum;
  guint64 frameCount;
  guint16 seq;
  guint16 extSeqNumber;
  gdouble timestampTick;
  guint ssrc;
  guint8 payloadType;
} StreamParams;

struct _GstNvDsUdpSink
{
  GstBaseSink parent;

  gboolean isRtpStream;
  GstAdapter *adapter;
  /* properties */
  gboolean auto_multicast;
  gboolean close_socket;
  gboolean loop;
  GSocket *socket;
  gchar *sdpFile;
  gchar *host;
  guint16 port;
  gchar *uri;
  gchar *localIfaceIp;
  gchar *ptpSrc;
  guint nChunks;
  guint packetsPerChunk;
  guint payloadSize;
  guint packetsPerLine;
  gboolean pass_rtp_timestamp;
  guint64 rtp_timestamp_offset;  /* RTP timestamp offset in nanoseconds */
  gint gpu_id;  /* GPU ID for GPUDirect, -1 means disabled */

  guint nextChunk;
  GThread *rThread;
  GQueue *bufferQ;
  GCond qCond;
  GMutex qLock;
  gboolean isRunning;
  gint lastError;
  gchar *renderThreadCore;

  /* Rivermax specific */
  void *memblock;
  void *header_block;
  rmax_stream_id streamId;
  struct sockaddr_in localNicAddr;
  struct rmax_chunk *chunks;
  rmax_mkey_id mKey[MAX_ST2022_7_STREAMS];
  rmax_mkey_id hKey[MAX_ST2022_7_STREAMS];
  StreamParams streamParams;
  gboolean isGenericApi;
  gint internalThreadCore;

  /* GPU direct specific */
  gboolean isGpuDirect;
  gboolean is_nvmm;
  size_t alignedMemSize;
  CUstream cuda_stream;

  gboolean is_dup;
  gchar* source_ips[MAX_ST2022_7_STREAMS];
  guint num_streams;
};

struct _GstNvDsUdpSinkClass
{
  GstBaseSinkClass parent_class;
};

GType gst_nvdsudpsink_get_type (void);

G_END_DECLS

#endif
