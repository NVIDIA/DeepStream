/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gst/gst.h>
#include <gst/base/gstbasesrc.h>
#include <gst/video/video.h>
#include "gstnvdsudpsrc.h"
#include "nvbufsurface.h"
#include "gstnvdsbufferpool.h"

#include <string.h>
#include <arpa/inet.h>

#include <sys/epoll.h>
#include <unistd.h>
#include <signal.h>


#define DEFAULT_PAYLOAD_SIZE            1500
#define DEFAULT_HEADER_SIZE             0
#define DEFAULT_NUM_PACKETS             10000
#define DEFAULT_MIN_PACKETS             0
#define DEFAULT_MAX_PACKETS             5000
#define DEFAULT_CHUNK_TIMEOUT_USEC      20000
#define UDP_DEFAULT_PORT                5004
#define UDP_DEFAULT_ADDRESS             "0.0.0.0"
#define UDP_DEFAULT_MULTICAST_IFACE     NULL
#define UDP_DEFAULT_LOCAL_IFACE_IP      NULL
#define UDP_DEFAULT_URI                 "udp://"UDP_DEFAULT_ADDRESS":"G_STRINGIFY(UDP_DEFAULT_PORT)
#define UDP_DEFAULT_BUFFER_SIZE         0
#define UDP_DEFAULT_TIMEOUT             0
#define UDP_DEFAULT_USED_SOCKET         NULL
#define UDP_DEFAULT_AUTO_MULTICAST      TRUE
#define UDP_DEFAULT_REUSE               TRUE
#define UDP_DEFAULT_LOOP                TRUE
#define ETH_TYPE_802_1Q                 0x8100
#define UDP_DEFAULT_SOURCE_ADDRESS      NULL
#define DEFAULT_GPU_DIRECT              FALSE
#define DEFAULT_PAYLOAD_MULTIPLE        1000

static void gst_nvdsudpsrc_set_property (GObject * object,
    guint property_id, const GValue * value, GParamSpec * pspec);
static void gst_nvdsudpsrc_get_property (GObject * object,
    guint property_id, GValue * value, GParamSpec * pspec);
static void gst_nvdsudpsrc_finalize (GObject * object);
static GstStateChangeReturn
gst_nvdsudpsrc_change_state (GstElement * element, GstStateChange transition);
static gboolean gst_nvdsudpsrc_open (GstNvDsUdpSrc * src);
static gboolean gst_nvdsudpsrc_close (GstNvDsUdpSrc * src);
static GstFlowReturn
gst_nvdsudpsrc_create (GstPushSrc * psrc, GstBuffer ** buf);
static gboolean gst_nvdsudpsrc_start (GstBaseSrc *psrc);
static gboolean gst_nvdsudpsrc_stop (GstBaseSrc *psrc);
static gboolean gst_nvdsudpsrc_unlock (GstBaseSrc * src);
static gboolean gst_nvdsudpsrc_unlock_stop (GstBaseSrc * src);
static GstClock* gst_nvdsudpsrc_provide_clock (GstElement * element);
static GstCaps *gst_nvdsudpsrc_get_caps (GstBaseSrc * src, GstCaps * filter);

static gboolean nvdsudpsrc_allocate_memory (GstNvDsUdpSrc *src);
static gpointer nvdsudpsr_data_fetch_loop (gpointer data);
static rmx_status
create_stream (GstNvDsUdpSrc *src, struct sockaddr_in *local_nic_addr,
               rmx_stream_id *stream_id, guint stream_idx);
static rmx_status destroy_stream (rmx_stream_id stream_id);
static rmx_status
attach_flow (rmx_stream_id stream_id, rmx_input_flow *in_flow);
static rmx_status
detach_flow (rmx_stream_id stream_id, rmx_input_flow *in_flow);

static void
gst_nvdsudpsrc_uri_handler_init (gpointer g_iface, gpointer iface_data);
static guint
parse_st2022_7_streams(const gchar* streams_str, DstStreamInfo* dstStream);
static guint
parse_ip_addresses(const gchar* addresses_str, gchar** address_array, const gchar* context);

static gboolean memcopy_2d(void *dst, guint dst_pitch, void *src_ptr, guint src_pitch,
                    guint width, guint height, GstNvDsUdpSrc *src_obj);
static gboolean memcopy_linear(void *dst, void *src_ptr, guint size, GstNvDsUdpSrc *src_obj);

enum
{
  PROP_0,

  PROP_PORT,
  PROP_PAYLOAD_SIZE,
  PROP_HEADER_SIZE,
  PROP_NUM_PACKETS,
  PROP_URI,
  PROP_LOCAL_IFACE_IP,
  PROP_CAPS,
  PROP_BUFFER_SIZE,
  PROP_REUSE,
  PROP_TIMEOUT,
  PROP_ADDRESS,
  PROP_USED_SOCKET,
  PROP_MULTICAST_IFACE,
  PROP_AUTO_MULTICAST,
  PROP_LOOP,
  PROP_SOURCE_ADDRESS,
  PROP_PAYLOAD_MULTIPLE,
  PROP_GPU_DEVICE_ID,
  PROP_USE_RTP_TIMESTAMP,
  PROP_ADJUST_LEAP_SECONDS,
  PROP_PTP_SOURCE,
  PROP_ST2022_7_STREAMS,
  PROP_BUFFER_PTS_OFFSET
};

static GstStaticPadTemplate src_template =
GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS_ANY);

GST_DEBUG_CATEGORY_STATIC (gst_nvdsudpsrc_debug_category);
#define GST_CAT_DEFAULT gst_nvdsudpsrc_debug_category

G_DEFINE_TYPE_WITH_CODE (GstNvDsUdpSrc, gst_nvdsudpsrc, GST_TYPE_PUSH_SRC,
  G_IMPLEMENT_INTERFACE (GST_TYPE_URI_HANDLER, gst_nvdsudpsrc_uri_handler_init));


/**  ********************* Memory allocator ***********************/

#define GST_NVDS_CUDA_MEMORY_TYPE "nvdscuda"

typedef struct _GstNvDsCudaMemoryAllocator GstNvDsCudaMemoryAllocator;
typedef struct _GstNvDsCudaMemoryAllocatorClass GstNvDsCudaMemoryAllocatorClass;

typedef struct GstNvDsCudaMemory {
  GstMemory mem;

  void *data;
} GstNvDsCudaMemory;

struct _GstNvDsCudaMemoryAllocator {
  GstAllocator parent;

  gboolean allocate_memory;
  gint gpuId;
};

struct _GstNvDsCudaMemoryAllocatorClass {
  GstAllocatorClass parent_class;
};

GType gst_nvds_cuda_memory_allocator_get_type (void);

G_DEFINE_TYPE (GstNvDsCudaMemoryAllocator, gst_nvds_cuda_memory_allocator,
    GST_TYPE_ALLOCATOR);

static gpointer
gst_nvds_cuda_memory_map (GstMemory *mem, gsize maxsize, GstMapFlags flags)
{
  GstNvDsCudaMemory *dsmem = (GstNvDsCudaMemory *) mem;
  GstNvDsCudaMemoryAllocator *dsAllocator = (GstNvDsCudaMemoryAllocator *) mem->allocator;

  g_return_val_if_fail (dsAllocator, NULL);

  // In case memory is not allocated internally,
  // return NULL if data is still not set by external entity.
  if (!dsAllocator->allocate_memory) {
    if (!dsmem->data)
      return NULL;
  }
  return dsmem->data;
}

static void
gst_nvds_cuda_memory_unmap (GstMemory *mem)
{
  // Nothing to do here.
}

static GstMemory *
gst_nvds_cuda_memory_share (GstMemory *mem, gssize offset, gssize size)
{
  /*
     Currently it won't be used because memory is non-shared.
   */
  g_assert_not_reached ();
  return NULL;
}

/**
 * @brief Initializes the CUDA memory allocator
 *
 * This function initializes the CUDA memory allocator by setting up the memory type,
 * memory mapping functions, and allocation flags. It configures the allocator to use
 * custom memory allocation and sets default values for memory allocation behavior.
 *
 * @param allocator Pointer to the GstNvDsCudaMemoryAllocator to initialize
 */
static void
gst_nvds_cuda_memory_allocator_init (GstNvDsCudaMemoryAllocator *allocator)
{
  GstAllocator *parent = GST_ALLOCATOR_CAST (allocator);

  parent->mem_type = GST_NVDS_CUDA_MEMORY_TYPE;
  parent->mem_map = gst_nvds_cuda_memory_map;
  parent->mem_unmap = gst_nvds_cuda_memory_unmap;
  parent->mem_share = gst_nvds_cuda_memory_share;

  allocator->allocate_memory = FALSE;
  allocator->gpuId = -1;

  /* We want to use default implementation of ->mem_copy which uses
     default allocator for memory allocation and then do memcpy().
     We are using this approach because by default our allocator doesn't
     allocate the memory it just initializes the mem structure.
     Null memory for destination buffer then will fail the memcpy().
   */

  GST_OBJECT_FLAG_SET (allocator, GST_ALLOCATOR_FLAG_CUSTOM_ALLOC);
}

/**
 * @brief Allocates CUDA memory using the custom allocator
 *
 * This function allocates memory using CUDA memory allocation functions based on the
 * allocator configuration. It can allocate either CUDA device memory or CUDA host memory
 * depending on the GPU ID setting.
 *
 * @param allocator Pointer to the GstAllocator instance
 * @param size Size of memory to allocate in bytes
 * @param params Additional allocation parameters (unused in this implementation)
 * @return GstMemory* Pointer to the allocated memory structure, or NULL on failure
 *
 * @note If gpuId is negative, allocates CUDA host memory (pinned memory)
 * @note If gpuId is non-negative, allocates CUDA device memory on the specified GPU
 */

static GstMemory *
gst_nvds_cuda_memory_allocator_alloc (GstAllocator *allocator,
        gsize size, GstAllocationParams *params)
{
  GstNvDsCudaMemory *mem = NULL;
  GstNvDsCudaMemoryAllocator *dsAllocator = (GstNvDsCudaMemoryAllocator *) allocator;
  GstMemoryFlags flags = GST_MEMORY_FLAG_NO_SHARE;
  cudaError_t err;

  mem = g_slice_new0 (GstNvDsCudaMemory);

  if (dsAllocator && dsAllocator->allocate_memory) {
    gint currentDevice = -1;
    gint gpuId = dsAllocator->gpuId;

    if (gpuId < 0) {
      err = cudaMallocHost (&mem->data, size);
      if (err != cudaSuccess) {
        GST_ERROR ("Failed to allocate cuda memory");
        goto error;
      }
    } else {
      err = cudaGetDevice (&currentDevice);
      if (err != cudaSuccess) {
        GST_ERROR ("Failed to get current GPU device, status=%d", err);
        goto error;
      }

      if (currentDevice != gpuId) {
        err = cudaSetDevice (gpuId);
        if (err != cudaSuccess) {
          GST_ERROR ("Failed to set CUDA device, status=%d", err);
          goto error;
        }
      }

      err = cudaMalloc (&mem->data, size);
      if (err != cudaSuccess) {
        GST_ERROR ("Failed to allocate cuda memory");
        goto error;
      }

      // restore the device of calling host thread.
      if (currentDevice != -1 && currentDevice != gpuId) {
        cudaSetDevice(currentDevice);
      }
    }
  }

  gst_memory_init (GST_MEMORY_CAST (mem), flags, allocator, NULL,
                   size, 0, 0, size);

  return GST_MEMORY_CAST (mem);

error:
  g_slice_free (GstNvDsCudaMemory, mem);
  return NULL;
}

/**
 * @brief Frees CUDA memory allocated by the allocator
 *
 * This function handles freeing CUDA memory that was previously allocated by the allocator.
 * It properly manages GPU device context switching when needed, ensuring the memory is freed
 * from the correct GPU device. For host-allocated memory (gpuId < 0), it uses cudaFreeHost.
 * For device-allocated memory, it uses cudaFree after ensuring the correct device context.
 *
 * @param allocator The allocator instance that allocated the memory
 * @param memory The memory to be freed
 */
static void
gst_nvds_cuda_memory_allocator_free (GstAllocator *allocator, GstMemory *memory)
{
  GstNvDsCudaMemory *dsmem = (GstNvDsCudaMemory *) memory;
  GstNvDsCudaMemoryAllocator *dsAllocator = (GstNvDsCudaMemoryAllocator *) allocator;

  if (dsAllocator && dsAllocator->allocate_memory) {
    cudaError_t err;
    void *data = dsmem->data;
    gint gpuId = dsAllocator->gpuId;
    gint currentDevice = -1;

    if (gpuId < 0) {
      cudaFreeHost (data);
    } else {
      err = cudaGetDevice (&currentDevice);
      if (err != cudaSuccess) {
        GST_ERROR ("Failed to get current GPU device, status=%d", err);
        goto error;
      }

      if (currentDevice != gpuId) {
        err = cudaSetDevice (gpuId);
        if (err != cudaSuccess) {
          GST_ERROR ("Failed to set CUDA device, status=%d", err);
          goto error;
        }
      }
      cudaFree (data);
      // restore the device of calling host thread.
      if (currentDevice != -1 && currentDevice != gpuId) {
        cudaSetDevice (currentDevice);
      }
    }
  }

error:
  g_slice_free (GstNvDsCudaMemory, dsmem);
}

static void
gst_nvds_cuda_memory_allocator_class_init (GstNvDsCudaMemoryAllocatorClass *klass)
{
  GstAllocatorClass *allocator_class;

  allocator_class = GST_ALLOCATOR_CLASS (klass);

  allocator_class->alloc = gst_nvds_cuda_memory_allocator_alloc;
  allocator_class->free = gst_nvds_cuda_memory_allocator_free;
}

/**  ********************* Buffer pool ***********************/

typedef struct _GstNvDsUdpBufferPool GstNvDsUdpBufferPool;
typedef struct _GstNvDsUdpBufferPoolClass GstNvDsUdpBufferPoolClass;
typedef struct _GstNvDsUdpBufferPoolPrivate GstNvDsUdpBufferPoolPrivate;

GType gst_nvds_udp_buffer_pool_get_type (void);

GstBufferPool* gst_nvds_udp_buffer_pool_new (void);

#define GST_TYPE_NVDS_UDP_BUFFER_POOL      (gst_nvds_udp_buffer_pool_get_type())
#define GST_IS_NVDS_UDP_BUFFER_POOL(obj)   (G_TYPE_CHECK_INSTANCE_TYPE ((obj), GST_TYPE_NVDS_UDP_BUFFER_POOL))
#define GST_NVDS_UDP_BUFFER_POOL(obj)      (G_TYPE_CHECK_INSTANCE_CAST ((obj), GST_TYPE_NVDS_UDP_BUFFER_POOL, GstNvDsUdpBufferPool))
#define GST_NVDS_UDP_BUFFER_POOL_CAST(obj) ((GstNvDsUdpBufferPool*)(obj))

struct _GstNvDsUdpBufferPoolPrivate
{
  GstAllocator *allocator;
  GstAllocationParams params;
  guint payloadSize;
  guint hdrSize;
  guint frameSize;
  gboolean isGpuDirect;
  gboolean isRtpOut;
};

struct _GstNvDsUdpBufferPool
{
  GstBufferPool bufferpool;

  GstNvDsUdpBufferPoolPrivate *priv;
};

struct _GstNvDsUdpBufferPoolClass
{
  GstBufferPoolClass parent_class;
};

#define gst_nvds_udp_buffer_pool_parent_class pool_parent_class
G_DEFINE_TYPE_WITH_CODE  (GstNvDsUdpBufferPool, gst_nvds_udp_buffer_pool,
    GST_TYPE_BUFFER_POOL, G_ADD_PRIVATE(GstNvDsUdpBufferPool));

static void
gst_nvds_udp_buffer_pool_finalize (GObject *object)
{
  GstNvDsUdpBufferPool *pool = GST_NVDS_UDP_BUFFER_POOL (object);
  GstNvDsUdpBufferPoolPrivate *priv = pool->priv;

  if (priv->allocator)
    gst_object_unref (priv->allocator);
  priv->allocator = NULL;

  G_OBJECT_CLASS (pool_parent_class)->finalize (object);
}

static GstFlowReturn
gst_nvds_udp_buffer_pool_alloc_buffer (GstBufferPool *bpool,
    GstBuffer **buffer, GstBufferPoolAcquireParams *params)
{
  GstNvDsUdpBufferPool *pool = GST_NVDS_UDP_BUFFER_POOL (bpool);
  GstNvDsUdpBufferPoolPrivate *priv = pool->priv;
  GstBuffer *buf = NULL;
  GstMemory *mem1 = NULL;
  GstMemory *mem2 = NULL;

  GST_DEBUG_OBJECT (pool, "alloc_buffer");

  if (priv->isRtpOut) {
    if (priv->hdrSize) {
      mem1 = gst_allocator_alloc (priv->allocator, priv->hdrSize, &priv->params);
      g_return_val_if_fail (mem1, GST_FLOW_ERROR);
    }

    mem2 = gst_allocator_alloc (priv->allocator, priv->payloadSize,
                                     &priv->params);

    g_return_val_if_fail (mem2, GST_FLOW_ERROR);
  } else {
    mem1 = gst_allocator_alloc (priv->allocator, priv->frameSize,
                                     &priv->params);

    g_return_val_if_fail (mem1, GST_FLOW_ERROR);
  }

  buf = gst_buffer_new ();

  if (mem1)
    gst_buffer_append_memory (buf, mem1);
  if (mem2)
    gst_buffer_append_memory (buf, mem2);

  *buffer = buf;

  return GST_FLOW_OK;
}

static void
gst_nvds_udp_buffer_pool_reset_buffer (GstBufferPool *pool, GstBuffer *buffer)
{
  GstMemory *mem;
  guint i, len;

  len = gst_buffer_n_memory (buffer);
  for (i = 0; i < len; i++) {
    mem = gst_buffer_peek_memory (buffer, i);
    if (!g_strcmp0 (mem->allocator->mem_type, GST_NVDS_CUDA_MEMORY_TYPE)) {
      mem->size = mem->maxsize;
    }
  }

  GST_BUFFER_POOL_CLASS (pool_parent_class)->reset_buffer (pool, buffer);
}

static void gst_nvds_udp_buffer_pool_init (GstNvDsUdpBufferPool *pool)
{
  GstNvDsUdpBufferPool *self = GST_NVDS_UDP_BUFFER_POOL (pool);
  pool->priv = (GstNvDsUdpBufferPoolPrivate *) gst_nvds_udp_buffer_pool_get_instance_private (self);

  memset (pool->priv, 0, sizeof(GstNvDsUdpBufferPoolPrivate));

  pool->priv->allocator = (GstAllocator *) g_object_new (
            gst_nvds_cuda_memory_allocator_get_type (), NULL);

  gst_allocation_params_init (&pool->priv->params);
}

static void
gst_nvds_udp_buffer_pool_class_init (GstNvDsUdpBufferPoolClass *klass)
{
  GObjectClass *gobject_class = (GObjectClass *) klass;
  GstBufferPoolClass *gstbufferpool_class = (GstBufferPoolClass *) klass;

  gobject_class->finalize = gst_nvds_udp_buffer_pool_finalize;
  gstbufferpool_class->alloc_buffer = gst_nvds_udp_buffer_pool_alloc_buffer;
  gstbufferpool_class->reset_buffer = gst_nvds_udp_buffer_pool_reset_buffer;
}

GstBufferPool * gst_nvds_udp_buffer_pool_new (void)
{
  GstNvDsUdpBufferPool *pool;
  pool = (GstNvDsUdpBufferPool *) g_object_new (GST_TYPE_NVDS_UDP_BUFFER_POOL, NULL);

  return GST_BUFFER_POOL (pool);
}

/*** GSTURIHANDLER INTERFACE **************************************/
static gboolean
gst_nvdsudpsrc_set_uri (GstNvDsUdpSrc * src, const gchar * uri, GError ** error)
{
  gchar *address;
  guint16 port;

  if (!src->localIfaceIps) {
    if (error != NULL) {
      g_set_error (error, GST_URI_ERROR, GST_URI_ERROR_BAD_STATE,
            "local interface ip not set");
    }
    return FALSE;
  }

  if (!gst_udp_parse_uri (uri, &address, &port))
    goto wrong_uri;

  if (port == (guint16) - 1)
    port = UDP_DEFAULT_PORT;

  g_free (src->address);
  src->address = address;
  src->port = port;

  g_free (src->uri);
  src->uri = g_strdup (uri);

  return TRUE;

  /* ERRORS */
wrong_uri:
  {
    GST_ELEMENT_ERROR (src, RESOURCE, READ, (NULL),
        ("error parsing uri %s", uri));
    g_set_error_literal (error, GST_URI_ERROR, GST_URI_ERROR_BAD_URI,
        "Could not parse UDP URI");
    return FALSE;
  }
}

static GstURIType
gst_nvdsudpsrc_uri_get_type (GType type)
{
  return GST_URI_SRC;
}

static const gchar *const *
gst_nvdsudpsrc_uri_get_protocols (GType type)
{
  static const gchar *protocols[] = { "udp", NULL };

  return protocols;
}

static gchar *
gst_nvdsudpsrc_uri_get_uri (GstURIHandler * handler)
{
  GstNvDsUdpSrc *src = GST_NVDSUDPSRC (handler);

  return g_strdup (src->uri);
}

static gboolean
gst_nvdsudpsrc_uri_set_uri (GstURIHandler * handler, const gchar * uri,
    GError ** error)
{
  return gst_nvdsudpsrc_set_uri (GST_NVDSUDPSRC (handler), uri, error);
}

static void
gst_nvdsudpsrc_uri_handler_init (gpointer g_iface, gpointer iface_data)
{
  GstURIHandlerInterface *iface = (GstURIHandlerInterface *) g_iface;

  iface->get_type = gst_nvdsudpsrc_uri_get_type;
  iface->get_protocols = gst_nvdsudpsrc_uri_get_protocols;
  iface->get_uri = gst_nvdsudpsrc_uri_get_uri;
  iface->set_uri = gst_nvdsudpsrc_uri_set_uri;
}

/**************** GSTURIHANDLER INTERFACE ************************/

static void
gst_nvdsudpsrc_class_init (GstNvDsUdpSrcClass * klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstBaseSrcClass *base_src_class = GST_BASE_SRC_CLASS (klass);
  GstPushSrcClass *gstpushsrc_class = GST_PUSH_SRC_CLASS (klass);
  GstElementClass *gstelement_class = GST_ELEMENT_CLASS (klass);

  gobject_class->set_property = gst_nvdsudpsrc_set_property;
  gobject_class->get_property = gst_nvdsudpsrc_get_property;
  gobject_class->finalize = gst_nvdsudpsrc_finalize;

  g_object_class_install_property (G_OBJECT_CLASS (klass), PROP_PORT,
      g_param_spec_int ("port", "Port",
      "The port to receive the packets from, 0=allocate", 0, G_MAXUINT16,
      UDP_DEFAULT_PORT, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (G_OBJECT_CLASS (klass), PROP_PAYLOAD_SIZE,
      g_param_spec_uint ("payload-size", "Payload size",
      "Size of payload in RTP / UDP packet", 0, G_MAXUINT16,
      DEFAULT_PAYLOAD_SIZE, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (G_OBJECT_CLASS (klass), PROP_HEADER_SIZE,
      g_param_spec_uint ("header-size", "Header size",
      "Size of RTP header", 0, G_MAXUINT16,
      DEFAULT_HEADER_SIZE, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (G_OBJECT_CLASS (klass), PROP_NUM_PACKETS,
      g_param_spec_uint ("num-packets", "Number of packets",
      "Number of packets for which memory to allocate.", 0, G_MAXINT,
      DEFAULT_NUM_PACKETS, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property(gobject_class, PROP_LOCAL_IFACE_IP,
      g_param_spec_string("local-iface-ip", "Local interface IP address",
      "Comma-separated list of IP addresses associated with network interfaces through which to receive data.\n"
      "\t\t\tFor ST2022-7 implementations, multiple IPs can be specified for interface binding.\n"
      "\t\t\tFormat: ip1,ip2,... (e.g., \"192.168.1.10,192.168.101.10\")\n",
      UDP_DEFAULT_LOCAL_IFACE_IP, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_CAPS,
      g_param_spec_boxed ("caps", "Caps",
      "The caps of the source pad", GST_TYPE_CAPS,
      G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_URI,
      g_param_spec_string ("uri", "URI",
      "URI in the form of udp://multicast_group:port", UDP_DEFAULT_URI,
      G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_USED_SOCKET,
      g_param_spec_object ("used-socket", "Socket Handle",
      "Socket currently in use for UDP reception. (NULL = no socket)",
      G_TYPE_SOCKET, G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_REUSE,
      g_param_spec_boolean ("reuse", "Reuse", "Enable reuse of the port\n"
      "\t\t\tsetting this property won't have any effect. Port will always be reused.\n"
      "\t\t\tIt is defined just to avoid warnings with rtspsrc",
      UDP_DEFAULT_REUSE, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (G_OBJECT_CLASS (klass), PROP_BUFFER_SIZE,
      g_param_spec_int ("buffer-size", "Buffer Size",
      "Size of the kernel receive buffer in bytes, 0=default", 0, G_MAXINT,
      UDP_DEFAULT_BUFFER_SIZE, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (G_OBJECT_CLASS (klass), PROP_TIMEOUT,
      g_param_spec_uint64 ("timeout", "Timeout",
      "Post a message after timeout nanoseconds (0 = disabled)", 0,
      G_MAXUINT64, UDP_DEFAULT_TIMEOUT,
      G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_MULTICAST_IFACE,
      g_param_spec_string ("multicast-iface", "Multicast Interface",
      "The network interface on which to join the multicast group."
      "This allows multiple interfaces seperated by comma. (\"eth0,eth1\")",
      UDP_DEFAULT_MULTICAST_IFACE,
      G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_AUTO_MULTICAST,
      g_param_spec_boolean ("auto-multicast", "Auto Multicast",
      "Automatically join/leave multicast groups",
      UDP_DEFAULT_AUTO_MULTICAST,
      G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_LOOP,
      g_param_spec_boolean ("loop", "Multicast Loopback",
          "Used for setting the multicast loop parameter. TRUE = enable,"
          " FALSE = disable", UDP_DEFAULT_LOOP,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_ADDRESS,
      g_param_spec_string ("address", "Address",
      "Address to receive packets for", UDP_DEFAULT_ADDRESS,
      G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_SOURCE_ADDRESS,
      g_param_spec_string ("source-address", "Source Address",
      "Comma-separated list of unicast addresses of senders. Used in source specific multicast to\n"
      "\t\t\t receive packets only from particular senders. For ST2022-7 implementations, multiple\n"
      "\t\t\t source addresses can be specified for source filtering.\n"
      "\t\t\t Format: ip1,ip2,... (e.g., \"192.168.1.100,192.168.101.100\")\n",
      UDP_DEFAULT_SOURCE_ADDRESS, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (G_OBJECT_CLASS (klass), PROP_PAYLOAD_MULTIPLE,
      g_param_spec_uint ("payload-multiple", "Payload Multiple",
      "Output buffer to be multiple of these number of packets.\n"
      "\t\t\tThis is applicable only for non RTP output mode\n"
      "\t\t\twhere mbit is not set in RTP header to decide the frame boundary",
      0, G_MAXUINT16, DEFAULT_PAYLOAD_MULTIPLE,
      G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (G_OBJECT_CLASS (klass), PROP_GPU_DEVICE_ID,
      g_param_spec_int ("gpu-id", "GPU Device id",
      "GPU device id to allocate the buffer.\n"
      "\t\t\tThis also enables the GPU Direct mode.",
      -1, G_MAXINT16, -1, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (G_OBJECT_CLASS (klass), PROP_USE_RTP_TIMESTAMP,
      g_param_spec_boolean ("use-rtp-timestamp", "Use RTP timestamp",
      "Parse RTP timestamp from rtp-header and attach as buffer PTS and DTS",
      FALSE, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (G_OBJECT_CLASS (klass), PROP_ADJUST_LEAP_SECONDS,
      g_param_spec_boolean ("adjust-leap-seconds", "Adjust leap seconds",
      "Adjust RTP timestamp for leap seconds when calculating running time",
      FALSE, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (G_OBJECT_CLASS (klass), PROP_PTP_SOURCE,
      g_param_spec_string ("ptp-src", "PTP source",
      "IP Address of PTP source.",
      DEFAULT_PTP_SRC, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (G_OBJECT_CLASS (klass), PROP_ST2022_7_STREAMS,
      g_param_spec_string ("st2022-7-streams", "ST2022-7 streams",
      "Comma-separated list of IP:port pairs for ST2022-7 redundant streams\n"
      "\t\t\tFormat: ip1:port1,ip2:port2,...\n"
      "\t\t\tSetting this property automatically enables ST2022-7 functionality",
      NULL, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (G_OBJECT_CLASS (klass), PROP_BUFFER_PTS_OFFSET,
      g_param_spec_uint64 ("latency-offset", "Latency Offset (ns)",
      "Offset in nanoseconds to add to buffer PTS/DTS when use-rtp-timestamp is enabled to account for pipeline latency (default: 0ns)",
      0, G_MAXUINT64, 0,
      G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  gst_element_class_add_static_pad_template (GST_ELEMENT_CLASS(klass),
      &src_template);

  gst_element_class_set_static_metadata (GST_ELEMENT_CLASS(klass),
      "UDP packet receiver", "Source/Network",
      "Receive data over the network via UDP using Mellanox Rivermax APIs",
      "NVIDIA Corporation. Post on Deepstream for Tesla forum for any queries "
      "@ https://devtalk.nvidia.com/default/board/209/");


  gstelement_class->change_state = gst_nvdsudpsrc_change_state;

  base_src_class->get_caps = GST_DEBUG_FUNCPTR (gst_nvdsudpsrc_get_caps);
  base_src_class->unlock = GST_DEBUG_FUNCPTR (gst_nvdsudpsrc_unlock);
  base_src_class->unlock_stop = GST_DEBUG_FUNCPTR (gst_nvdsudpsrc_unlock_stop);
  base_src_class->start = GST_DEBUG_FUNCPTR (gst_nvdsudpsrc_start);
  base_src_class->stop = GST_DEBUG_FUNCPTR (gst_nvdsudpsrc_stop);
  /* Use the provide_clock method to set REALTIME clock as pipeline clock.
     REALTIME clock is set only when "use_rtp_timestamp" is enabled. */
  gstelement_class->provide_clock = GST_DEBUG_FUNCPTR (gst_nvdsudpsrc_provide_clock);

  gstpushsrc_class->create = gst_nvdsudpsrc_create;

  GST_DEBUG_CATEGORY_INIT (gst_nvdsudpsrc_debug_category, "nvdsudpsrc", 0,
  "debug category for nvdsudpsrc element");
}

static void
gst_nvdsudpsrc_init (GstNvDsUdpSrc *src)
{
  src->localIfaceIps = g_strdup (g_getenv ("LOCAL_IFACE_IP"));
  src->port = UDP_DEFAULT_PORT;
  src->address = g_strdup ("0.0.0.0");
  src->buffer_size = UDP_DEFAULT_BUFFER_SIZE;
  src->timeout = UDP_DEFAULT_TIMEOUT;
  src->loop = UDP_DEFAULT_LOOP;
  src->payloadSize = DEFAULT_PAYLOAD_SIZE;
  src->headerSize = DEFAULT_HEADER_SIZE;
  src->numPackets = DEFAULT_NUM_PACKETS;
  src->uri = NULL;
  src->multi_iface = NULL;
  src->sourceAddresses = g_strdup (UDP_DEFAULT_SOURCE_ADDRESS);
  src->caps = NULL;
  src->use_rtp_timestamp = FALSE;
  src->adjust_leap_seconds = FALSE;
  src->latency_offset = 0;
  src->ptpSrc = NULL;
  src->st2022_7_streams = NULL;
  src->num_streams = 1;
  src->num_source_addresses = 0;
  src->num_local_interfaces = 0;
  src->first_rtp_packet = TRUE;
  src->last_rtp_tick = 0;
  src->rtp_tick_base = 0;
  src->clock_rate = 90000;  /* Default to video clock rate */

  src->flowId = 0;
  src->isGpuDirect = DEFAULT_GPU_DIRECT;
  src->reuse = TRUE;
  src->dataPtr = src->hdrPtr = NULL;
  src->lastError = 0;
  src->pollfd = -1;
  src->cancellableFd = -1;
  src->isRtpOut = TRUE;
  src->ffFound = FALSE;
  src->dataPtr1 = src->dataPtr2 = NULL;
  src->len1 = src->len2 = 0;
  src->mBit = 0;
  src->packetCounter = 0;
  src->payMultiple = DEFAULT_PAYLOAD_MULTIPLE;
  src->gpuId = -1;
  src->outputMemType = MEM_TYPE_UNKNOWN;
  src->is_nvmm = TRUE;
  for (guint i = 0; i < MAX_ST2022_7_STREAMS; i++) {
    src->streamId[i] = INVALID_STREAM_ID;
    src->dstStream[i].ip = NULL;
    src->srcAddress[i] = NULL;
    src->localIfaceIp[i] = NULL;
  }

  g_mutex_init (&src->qLock);
  g_cond_init (&src->qCond);
  g_mutex_init (&src->flowLock);
  g_cond_init (&src->flowCond);

  gst_base_src_set_live (GST_BASE_SRC (src), TRUE);
  gst_base_src_set_format (GST_BASE_SRC (src), GST_FORMAT_TIME);
  gst_base_src_set_do_timestamp (GST_BASE_SRC (src), TRUE);
  GST_OBJECT_FLAG_SET (src, GST_ELEMENT_FLAG_PROVIDE_CLOCK);
}

void
gst_nvdsudpsrc_set_property (GObject * object, guint property_id,
    const GValue * value, GParamSpec * pspec)
{
  GstNvDsUdpSrc *src = GST_NVDSUDPSRC (object);

  GST_DEBUG_OBJECT (src, "set_property");

  switch (property_id) {
    case PROP_PORT: {
      src->port = g_value_get_int (value);

      g_free (src->uri);
      src->uri =
          g_strdup_printf ("udp://%s:%u", src->address, src->port);
    }
      break;
    case PROP_PAYLOAD_SIZE:
      src->payloadSize = g_value_get_uint (value);
      break;
    case PROP_HEADER_SIZE:
      src->headerSize = g_value_get_uint (value);
      break;
    case PROP_NUM_PACKETS:
      src->numPackets = g_value_get_uint (value);
      break;
    case PROP_LOCAL_IFACE_IP:
      g_free (src->localIfaceIps);
      src->localIfaceIps = g_value_dup_string (value);
      g_strstrip (src->localIfaceIps);
      if (!g_strcmp0 (src->localIfaceIps, "")) {
        g_free (src->localIfaceIps);
        src->localIfaceIps = NULL;
      }
      break;
    case PROP_CAPS: {
      const GstCaps *new_caps_val = gst_value_get_caps (value);
      GstCaps *new_caps;
      GstCaps *old_caps;

      if (new_caps_val == NULL) {
        new_caps = gst_caps_new_any ();
      } else {
        new_caps = gst_caps_copy (new_caps_val);
      }

      GST_OBJECT_LOCK (src);
      old_caps = src->caps;
      src->caps = new_caps;
      GST_OBJECT_UNLOCK (src);
      if (old_caps)
        gst_caps_unref (old_caps);

      gst_pad_mark_reconfigure (GST_BASE_SRC_PAD (src));
      break;
    }
    case PROP_URI:
      gst_nvdsudpsrc_set_uri (src, g_value_get_string (value), NULL);
      break;
    case PROP_REUSE:
      // Ignore the value set by user. port will always be re-used.
      // "reuse" property is defined just to avoid warning prints because
      // "rtspsrc" sets that property.
      // src->reuse = g_value_get_boolean (value);
      break;
    case PROP_BUFFER_SIZE:
      src->buffer_size = g_value_get_int (value);
      break;
    case PROP_TIMEOUT:
      src->timeout = g_value_get_uint64 (value);
      break;
    case PROP_ADDRESS:
    {
      const gchar *group;

      g_free (src->address);
      if ((group = g_value_get_string (value)))
        src->address = g_strdup (group);
      else
        src->address = g_strdup (UDP_DEFAULT_ADDRESS);

      g_free (src->uri);
      src->uri =
          g_strdup_printf ("udp://%s:%u", src->address, src->port);
      break;
    }
    case PROP_MULTICAST_IFACE:
      g_free (src->multi_iface);

      if (g_value_get_string (value) == NULL)
        src->multi_iface = g_strdup (UDP_DEFAULT_MULTICAST_IFACE);
      else
        src->multi_iface = g_value_dup_string (value);
      break;
    case PROP_AUTO_MULTICAST:
      src->auto_multicast = g_value_get_boolean (value);
      break;
    case PROP_LOOP:
      src->loop = g_value_get_boolean (value);
      break;
    case PROP_SOURCE_ADDRESS:
      g_free (src->sourceAddresses);

      if (g_value_get_string (value) == NULL)
        src->sourceAddresses = NULL;
      else
        src->sourceAddresses = g_value_dup_string (value);
      break;
    case PROP_PAYLOAD_MULTIPLE:
      src->payMultiple = g_value_get_uint (value);
      break;
    case PROP_GPU_DEVICE_ID:
      src->gpuId = g_value_get_int (value);
      if (src->gpuId >= 0)
        src->isGpuDirect = TRUE;
      break;
    case PROP_USE_RTP_TIMESTAMP:
      src->use_rtp_timestamp = g_value_get_boolean (value);
      break;
    case PROP_ADJUST_LEAP_SECONDS:
      src->adjust_leap_seconds = g_value_get_boolean (value);
      break;
    case PROP_PTP_SOURCE:
      g_free (src->ptpSrc);
      src->ptpSrc = g_value_dup_string (value);
      break;
    case PROP_ST2022_7_STREAMS:
      g_free (src->st2022_7_streams);
      src->st2022_7_streams = g_value_dup_string (value);
      break;
    case PROP_BUFFER_PTS_OFFSET:
      src->latency_offset = g_value_get_uint64 (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);
      break;
  }
}

void
gst_nvdsudpsrc_get_property (GObject * object, guint property_id,
    GValue * value, GParamSpec * pspec)
{
  GstNvDsUdpSrc *src = GST_NVDSUDPSRC (object);

  GST_DEBUG_OBJECT (src, "get_property");

  switch (property_id) {
    case PROP_PORT:
      g_value_set_int (value, src->port);
      break;
    case PROP_PAYLOAD_SIZE:
      g_value_set_uint (value, src->payloadSize);
      break;
    case PROP_HEADER_SIZE:
      g_value_set_uint (value, src->headerSize);
      break;
    case PROP_NUM_PACKETS:
      g_value_set_uint (value, src->numPackets);
      break;
    case PROP_LOCAL_IFACE_IP:
      g_value_set_string (value, src->localIfaceIps);
      break;
    case PROP_CAPS:
      GST_OBJECT_LOCK (src);
      gst_value_set_caps (value, src->caps);
      GST_OBJECT_UNLOCK (src);
      break;
    case PROP_URI:
      g_value_set_string (value, src->uri);
      break;
    case PROP_USED_SOCKET:
      g_value_set_object (value, src->used_socket);
      break;
    case PROP_REUSE:
      g_value_set_boolean (value, src->reuse);
      break;
    case PROP_BUFFER_SIZE:
      g_value_set_int (value, src->buffer_size);
      break;
    case PROP_TIMEOUT:
      g_value_set_uint64 (value, src->timeout);
      break;
    case PROP_ADDRESS:
      g_value_set_string (value, src->address);
      break;
    case PROP_MULTICAST_IFACE:
      g_value_set_string (value, src->multi_iface);
      break;
    case PROP_AUTO_MULTICAST:
      g_value_set_boolean (value, src->auto_multicast);
      break;
    case PROP_LOOP:
      g_value_set_boolean (value, src->loop);
      break;
    case PROP_SOURCE_ADDRESS:
      g_value_set_string (value, src->sourceAddresses);
      break;
    case PROP_PAYLOAD_MULTIPLE:
      g_value_set_uint (value, src->payMultiple);
      break;
    case PROP_GPU_DEVICE_ID:
      g_value_set_int (value, src->gpuId);
      break;
    case PROP_USE_RTP_TIMESTAMP:
      g_value_set_boolean (value, src->use_rtp_timestamp);
      if (src->use_rtp_timestamp) {
        gst_base_src_set_do_timestamp (GST_BASE_SRC (src), FALSE);
      }
      break;
    case PROP_ADJUST_LEAP_SECONDS:
      g_value_set_boolean (value, src->adjust_leap_seconds);
      break;
    case PROP_PTP_SOURCE:
      g_value_set_string (value, src->ptpSrc);
      break;
    case PROP_ST2022_7_STREAMS:
      g_value_set_string (value, src->st2022_7_streams);
      break;
    case PROP_BUFFER_PTS_OFFSET:
      g_value_set_uint64 (value, src->latency_offset);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);
      break;
  }
}

void
gst_nvdsudpsrc_finalize (GObject * object)
{
  GstNvDsUdpSrc *src = GST_NVDSUDPSRC (object);

  GST_DEBUG_OBJECT (src, "finalize");

  if (src->caps)
    gst_caps_unref (src->caps);
  src->caps = NULL;

  g_free (src->multi_iface);
  src->multi_iface = NULL;

  g_free (src->uri);
  src->uri = NULL;

  g_free (src->address);
  src->address = NULL;

  g_free (src->localIfaceIps);
  src->localIfaceIps = NULL;

  for (guint i = 0; i < src->num_local_interfaces; i++) {
    if (src->localIfaceIp[i]) {
      g_free(src->localIfaceIp[i]);
      src->localIfaceIp[i] = NULL;
    }
  }

  g_free (src->sourceAddresses);
  src->sourceAddresses = NULL;

  for (guint i = 0; i < src->num_source_addresses; i++) {
    if (src->srcAddress[i]) {
      g_free(src->srcAddress[i]);
      src->srcAddress[i] = NULL;
    }
  }

  g_free (src->ptpSrc);
  src->ptpSrc = NULL;

  g_free (src->st2022_7_streams);
  src->st2022_7_streams = NULL;

  for (guint i = 0; i < src->num_streams; i++) {
    if (src->dstStream[i].ip) {
      g_free(src->dstStream[i].ip);
      src->dstStream[i].ip = NULL;
    }
  }

  if (src->used_socket)
    g_object_unref (src->used_socket);
  src->used_socket = NULL;

  g_mutex_clear (&src->qLock);
  g_cond_clear (&src->qCond);

  g_mutex_clear (&src->flowLock);
  g_cond_clear (&src->flowCond);

  G_OBJECT_CLASS (gst_nvdsudpsrc_parent_class)->finalize (object);
}

static GstStateChangeReturn
gst_nvdsudpsrc_change_state (GstElement * element, GstStateChange transition)
{
  GstNvDsUdpSrc *src;
  GstStateChangeReturn result;
  rmx_status status;

  src = GST_NVDSUDPSRC (element);

  switch (transition) {
    case GST_STATE_CHANGE_NULL_TO_READY:
      if (!gst_nvdsudpsrc_open (src))
        goto open_failed;
      break;
    default:
      break;
  }
  if ((result =
          GST_ELEMENT_CLASS (gst_nvdsudpsrc_parent_class)->change_state (element,
              transition)) == GST_STATE_CHANGE_FAILURE)
    goto failure;

  switch (transition) {
    case GST_STATE_CHANGE_PAUSED_TO_PLAYING:
      g_mutex_lock (&src->flowLock);

      if (src->use_rtp_timestamp) {
        /* For re-calculating RTP timestamp from rtp-ticks received in the packet header */
        src->first_rtp_packet = TRUE;
        src->rtp_tick_base    = 0;
        src->last_rtp_tick    = 0;
      }
      /* To start from the beginning of the stream */
      src->ffFound = FALSE;
      src->dataPtr1 = src->dataPtr2 = NULL;
      src->len1 = src->len2 = 0;
      src->packetCounter = 0;

      // Attach the flow(s) when transitioning from PAUSED to PLAYING.
      for (guint i = 0; i < src->num_streams; i++) {
        if (src->streamId[i] != INVALID_STREAM_ID) {
          status = attach_flow (src->streamId[i], &src->flow[i]);
          if (status != RMX_OK) {
            GST_ERROR_OBJECT (src, "Failed to attach flow for streamId[%u], status:%d", i, status);
            result = GST_STATE_CHANGE_FAILURE;
            break;
          } else {
            GST_DEBUG_OBJECT (src, "Successfully attached flow for streamId[%u]", i);
          }
        }
      }
      g_mutex_unlock (&src->flowLock);
      break;

    case GST_STATE_CHANGE_PLAYING_TO_PAUSED:
      g_mutex_lock (&src->flowLock);

      // Detach the flow(s) when transitioning from PLAYING to PAUSED.
      for (guint i = 0; i < src->num_streams; i++) {
        if (src->streamId[i] != INVALID_STREAM_ID) {
          status = detach_flow (src->streamId[i], &src->flow[i]);
          if (status != RMX_OK) {
            GST_ERROR_OBJECT (src, "Failed to detach flow for streamId[%u], status:%d", i, status);
            result = GST_STATE_CHANGE_FAILURE;
          } else {
            GST_DEBUG_OBJECT (src, "Successfully detached flow for streamId[%u]", i);
          }
        }
      }
      g_mutex_unlock (&src->flowLock);
      break;

    case GST_STATE_CHANGE_READY_TO_NULL:
      gst_nvdsudpsrc_close (src);
      break;
    default:
      break;
  }
  return result;
  /* ERRORS */
open_failed:
  {
    GST_DEBUG_OBJECT (src, "failed to open socket");
    return GST_STATE_CHANGE_FAILURE;
  }
failure:
  {
    GST_DEBUG_OBJECT (src, "parent failed state change");
    return result;
  }
}

/**
 * @brief Allocates GPU memory to be used for packet reception
 *
 * This function allocates GPU memory for packet reception with proper alignment.
 *
 * @param buffer_len Size of buffer to allocate in bytes
 * @param align Alignment requirement for the buffer
 * @param src Pointer to GstNvDsUdpSrc instance containing GPU configuration
 * @return void* Pointer to allocated GPU memory, or NULL on failure
 */
static void*
allocate_buffer_gpu (size_t buffer_len, size_t align, GstNvDsUdpSrc *src)
{
  void *buffer = NULL;
  cudaError_t cuda_ret;

  int gpuId = src->gpuId;
  struct cudaDeviceProp props;
  cuda_ret = cudaGetDeviceProperties (&props, gpuId);
  if (cuda_ret != cudaSuccess) {
    GST_ERROR ("failed to get device properties.");
    return NULL;
  }

  size_t size;
  if (props.integrated) {
    size = round_up (buffer_len, align);
  } else {
    size = gpu_aligned_size (gpuId, buffer_len);
  }
  src->alignedMemSize = size;
  buffer = gpu_allocate_memory (gpuId, size, align);

  if (!buffer)
    return NULL;

  if (props.integrated) {
    memset(buffer, 0, size);
  } else {
    cudaMemset(buffer, 0, size);
  }
  src->dataPtr = buffer;
  return buffer;
}

/**
 * @brief Allocates host memory with proper alignment for packet reception
 *
 * @param buffer_len Size of buffer to allocate in bytes
 * @param align Alignment requirement for the buffer
 * @param buf Pointer to store the original allocated buffer pointer
 * @return void* Pointer to aligned memory region, or NULL on failure
 */

static void*
allocate_buffer_host (size_t buffer_len, size_t align, void **buf)
{
  void *buffer;
  size_t buffer_len_aligned = buffer_len + align;
  uint64_t addr;
  void *ptr;

  buffer = g_malloc0 (buffer_len_aligned);
  if (!buffer) {
    GST_ERROR ("Host memory allocation failed");
    return NULL;
  }

  memset(buffer, 0, buffer_len_aligned);
  addr = (uint64_t) buffer;
  ptr = (void *) ((addr + align) & ~(align - 1));
  *buf = buffer;

  return ptr;
}

/**
 * @brief Deallocates memory allocated for packet reception
 *
 * This function deallocates memory allocated for packet reception. It handles both
 * GPU and host memory cases differently based on the isGpuDirect flag.
 *
 * @param src Pointer to GstNvDsUdpSrc instance containing memory information
 * @return void
 */
static void
deallocate_buffer (GstNvDsUdpSrc *src)
{
  g_return_if_fail (src);

  if (src->isGpuDirect && src->dataPtr) {
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, src->gpuId);

    if (prop.integrated) {
      cudaFreeHost (src->dataPtr);
    } else {
      cudaFreeMmap ((uint64_t *)&src->dataPtr, src->alignedMemSize);
    }
  } else {
    g_free (src->dataPtr);
  }

  g_free (src->hdrPtr);

  if (src->data) {
    src->data->addr = NULL;
  }
  src->data = NULL;
  src->dataPtr = NULL;
  if (src->hdr) {
    src->hdr->addr = NULL;
  }
  src->hdr = NULL;
  src->hdrPtr = NULL;
}

/**
 * @brief Creates a new Rivermax input stream
 *
 * This function creates a new Rivermax input stream with the specified parameters.
 * The stream is configured to use packet protocol and raw nanosecond timestamps.
 * Packet information is enabled for each packet received.
 *
 * @param local_nic_addr Pointer to sockaddr_in structure containing local NIC address
 * @param stream_id Pointer to store the created stream ID
 * @param buffer_attr Pointer to buffer attributes structure
 * @return rmax_status_t Status of stream creation operation
 *         - RMAX_OK on success
 *         - Other error codes on failure
 */
static rmx_status
create_stream (GstNvDsUdpSrc *src, struct sockaddr_in *local_nic_addr,
               rmx_stream_id *stream_id, guint stream_idx)
{
  rmx_status status;

  /* Set local NIC address (stream_params already configured in nvdsudpsrc_allocate_memory) */
  rmx_input_set_stream_nic_address(&src->stream_params, (struct sockaddr *)local_nic_addr);

  /* Enable packet info per packet */
  rmx_input_enable_stream_option(&src->stream_params, RMX_INPUT_STREAM_CREATE_INFO_PER_PACKET);

  /* Set timestamp format */
  rmx_input_set_timestamp_format(&src->stream_params, RMX_INPUT_TIMESTAMP_RAW_NANO);

  /* Create the stream */
  status = rmx_input_create_stream(&src->stream_params, stream_id);
  if (status != RMX_OK) {
    GST_ERROR("Failed to create stream, status = %d\n", status);
    return status;
  }

  /* Set completion moderation (CRITICAL: Required for Rivermax v1.80.24+) */
  status = rmx_input_set_completion_moderation(*stream_id, DEFAULT_MIN_PACKETS, DEFAULT_MAX_PACKETS, DEFAULT_CHUNK_TIMEOUT_USEC);
  if (status != RMX_OK) {
    GST_WARNING_OBJECT(src, "Failed to set completion moderation, status = %d (continuing anyway)\n", status);
  } else {
    GST_INFO_OBJECT(src, "Set completion moderation: min=%d, max=%d, timeout=%d us for stream %d\n",
             DEFAULT_MIN_PACKETS, DEFAULT_MAX_PACKETS, DEFAULT_CHUNK_TIMEOUT_USEC, *stream_id);
  }

  /* Initialize chunk handle */
  rmx_input_init_chunk_handle(&src->chunk_handle[stream_idx], *stream_id);

  GST_INFO("Created Rivermax input stream with ID %d\n", *stream_id);

  return status;
}

/**
 * @brief Destroys a Rivermax input stream
 *
 * This function destroys a Rivermax input stream with the specified stream ID.
 *
 * @param stream_id Stream ID to destroy
 * @return rmax_status_t Status of stream destruction operation
 *         - RMAX_OK on success
 *         - Other error codes on failure
 */
static rmx_status
destroy_stream (rmx_stream_id stream_id)
{
  rmx_status status;

  status = rmx_input_destroy_stream(stream_id);
  if (status != RMX_OK) {
    GST_ERROR("Failed to destroy stream, status = %d\n", status);
  }
  return status;
}

/**
 * @brief Attaches a flow to a Rivermax input stream
 *
 * This function attaches a flow to a Rivermax input stream with the specified stream ID.
 *
 * @param stream_id Stream ID to attach flow to
 * @param in_flow Pointer to flow attributes structure
 * @param flow_tag Flow tag identifier (new API only)
 * @return Status of flow attachment operation
 *         - RMAX_OK/RMX_OK on success
 *         - Other error codes on failure
 */
static rmx_status
attach_flow (rmx_stream_id stream_id, rmx_input_flow *in_flow)
{
  rmx_status status;

  status = rmx_input_attach_flow (stream_id, in_flow);
  if (status != RMX_OK) {
    GST_ERROR ("Failed to attach flow, status = %d", status);
  } else {
    GST_INFO ("Attached flow to stream %d", stream_id);
  }

  return status;
}

/**
 * @brief Detaches a flow from a Rivermax input stream
 *
 * This function detaches a flow from a Rivermax input stream with the specified stream ID.
 *
 * @param stream_id Stream ID to detach flow from
 * @param in_flow Pointer to flow attributes structure
 * @param flow_tag Flow tag identifier (new API only)
 * @return Status of flow detachment operation
 *         - RMAX_OK/RMX_OK on success
 *         - Other error codes on failure
 */
static rmx_status
detach_flow (rmx_stream_id stream_id, rmx_input_flow *in_flow)
{
  rmx_status status;

  status = rmx_input_detach_flow (stream_id, in_flow);
  if (status != RMX_OK && status != RMX_SIGNAL) {
    GST_ERROR("Failed to detach flow, status = %d", status);
  } else {
    GST_INFO ("Detached flow from stream %d", stream_id);
  }
  return status;
}

static uint16_t get_page_size(void)
{
    uint16_t size = (uint16_t)sysconf(_SC_PAGESIZE);

    return size;
}

static uint16_t get_cache_line_size (void)
{
  uint16_t size = (uint16_t) sysconf (_SC_LEVEL1_DCACHE_LINESIZE);

  return size;
}

/**
 * @brief Gets the RTP header pointer from the packet header
 *
 * This function extracts the RTP header pointer from the packet header based on the stream type.
 * For regular RTP packets, it skips the Ethernet, IP and UDP headers to get to the RTP header.
 * For 802.1Q tagged packets, it skips additional VLAN header bytes.
 *
 * @param hdr Pointer to the start of the packet header
 * @param type The stream type (RTP or application protocol)
 * @return Pointer to the RTP header within the packet
 */
static uint8_t* get_rtp_hdr_ptr (uint8_t* hdr, rmx_input_stream_params_type type)
{
  if (!hdr || RMX_INPUT_APP_PROTOCOL_PACKET == type) {
    return hdr;
  }

  uint16_t *eth_proto = (uint16_t *)(hdr + 12);
  if (ETH_TYPE_802_1Q == ntohs(*eth_proto)) {
    hdr += 46; // 802 + 802.1Q + IP + UDP
  } else {
    hdr += 42; // 802 + IP + UDP
  }
  return hdr;
}

/**
 * @brief Allocates memory for RTP packet header and payload to be used for packet reception
 *
 * If header data split is enabled, it allocates memory for RTP header and payload separately.
 *
 * For RTP header:
 * - Allocates host memory for RTP header data
 * - Ensures proper alignment for optimal CPU access
 *
 * For RTP payload:
 * - Allocates GPU memory for payload data if isGpuDirect is TRUE else system memory
 *
 */
static gboolean
nvdsudpsrc_allocate_memory (GstNvDsUdpSrc *src)
{
  rmx_status status = RMX_OK;
  size_t payLen;
  size_t hdrLen;
  gboolean is_page_aligned = false;

  memset(&src->localNicAddr, 0, sizeof(src->localNicAddr));
  src->localNicAddr.sin_family = AF_INET;
  src->localNicAddr.sin_addr.s_addr = inet_addr(src->localIfaceIp[0]);
  /* Initialize stream parameters*/
  rmx_input_init_stream(&src->stream_params, RMX_INPUT_APP_PROTOCOL_PACKET);

  /* Set memory configuration */
  rmx_input_set_mem_capacity_in_packets(&src->stream_params, src->numPackets);

  /* Configure memory blocks based on Header Data Split (HDS) */
  (src->headerSize > 0 ? (src->header_mem_block_id = 0, src->payload_mem_block_id = 1) : (src->payload_mem_block_id = 0));
  rmx_input_set_mem_sub_block_count(&src->stream_params, src->headerSize > 0 ? 2 : 1);
  if (src->headerSize > 0) {
    rmx_input_set_entry_size_range(&src->stream_params, src->header_mem_block_id, src->headerSize, src->headerSize);
  }
  rmx_input_set_entry_size_range(&src->stream_params, src->payload_mem_block_id, src->payloadSize, src->payloadSize);

  /* Set RTP extended sequence number flag for hardware-based ST2022-7 implementation with video streams */
  if (src->st2022_7_streams && (src->streamType == VIDEO_2110_20_STREAM)) {
    rmx_input_enable_stream_option(&src->stream_params, RMX_INPUT_STREAM_RTP_EXT_SEQN_PLACEMENT_ORDER);
    is_page_aligned = true;
  } else if (src->st2022_7_streams) {
    rmx_input_enable_stream_option(&src->stream_params, RMX_INPUT_STREAM_RTP_SEQN_PLACEMENT_ORDER);
    is_page_aligned = true;
  }

  rmx_input_set_stream_nic_address(&src->stream_params, (struct sockaddr*)(&src->localNicAddr));
  /* Determine memory layout using Rivermax API */
  status = rmx_input_determine_mem_layout(&src->stream_params);
  if (status != RMX_OK) {
    GST_ERROR_OBJECT (src, "Failed to determine memory layout, status: %d", status);
    return FALSE;
  }

  uint32_t buffer_elements = (uint32_t)rmx_input_get_mem_capacity_in_packets(&src->stream_params);

  /* Get pointers to internal memory regions within stream_params.
   * These point directly into stream_params and are used by Rivermax when creating the stream. */
  src->data = rmx_input_get_mem_block_buffer(&src->stream_params, src->payload_mem_block_id);
  src->hdr = (src->headerSize > 0) ?
             rmx_input_get_mem_block_buffer(&src->stream_params, src->header_mem_block_id) : NULL;

  payLen = src->data ? src->data->length : 0;
  hdrLen = src->hdr ? src->hdr->length : 0;

  GST_INFO_OBJECT (src, "=== Memory layout summary: payload=%lu bytes, header=%lu bytes, num_packets=%u ===",
                   payLen, hdrLen, buffer_elements);

  /* Store stride sizes for new API */
  src->data_stride_size = rmx_input_get_stride_size(&src->stream_params, src->payload_mem_block_id);
  if (src->headerSize > 0) {
    src->hdr_stride_size = rmx_input_get_stride_size(&src->stream_params, src->header_mem_block_id);
  }

  GST_INFO_OBJECT (src, "=== Stride sizes: data=%lu, header=%lu ===",
                   src->data_stride_size, src->hdr_stride_size);

  if (src->headerSize && !hdrLen) {
    GST_ERROR_OBJECT (src, "Header data split not supported");
    return FALSE;
  }

  size_t align = is_page_aligned ? (size_t) get_page_size() : (size_t) get_cache_line_size();

  /* Allocate payload memory and set it directly on the internal memory region */
  if (src->isGpuDirect) {
    GST_DEBUG_OBJECT (src, "Allocating %lu bytes of GPU memory for payload", payLen);
    src->data->addr = allocate_buffer_gpu (payLen, align, src);
    GST_DEBUG_OBJECT (src, "Allocated %lu bytes of GPU memory for payload", src->alignedMemSize);
    if (!src->data->addr) {
      GST_ERROR_OBJECT (src, "Failed to allocate payload GPU memory");
      return FALSE;
    }
    src->data->mkey = RMX_MKEY_INVALID;
  } else {
    GST_DEBUG_OBJECT (src, "Allocating %lu bytes of system memory for payload", payLen);
    src->data->addr = allocate_buffer_host (payLen, align, &src->dataPtr);
    if (!src->data->addr) {
      GST_ERROR_OBJECT (src, "Failed to allocate payload memory");
      return FALSE;
    }
    src->data->mkey = RMX_MKEY_INVALID;
  }

  /* Allocate header memory and set it directly on the internal memory region */
  if (hdrLen && src->hdr) {
    GST_DEBUG_OBJECT (src, "Allocating %lu bytes of system memory for header", hdrLen);
    src->hdr->addr = allocate_buffer_host (hdrLen, align, &src->hdrPtr);
    if (!src->hdr->addr) {
      GST_ERROR_OBJECT (src, "Failed to allocate header memory");
      deallocate_buffer (src);
      return FALSE;
    }
    src->hdr->mkey = RMX_MKEY_INVALID;
  }

  return TRUE;
}

/**
 * @brief Creates a cancellable object for the UDP source
 *
 * This function creates a GCancellable object and sets up an epoll event
 * to monitor the cancellable's file descriptor. This allows for asynchronous
 * cancellation of operations.
 *
 * @param src Pointer to the GstNvDsUdpSrc instance
 * @return TRUE if cancellable was created successfully, FALSE otherwise
 */
static gboolean
nvdsudpsrc_create_cancellable (GstNvDsUdpSrc *src)
{
  struct epoll_event ev;

  g_return_val_if_fail (src != NULL, FALSE);
  g_return_val_if_fail (src->pollfd >= 0, FALSE);

  src->cancellable = g_cancellable_new ();
  src->cancellableFd = g_cancellable_get_fd (src->cancellable);

  if (src->cancellableFd >= 0) {
     memset (&ev, 0, sizeof(ev));
     ev.events = EPOLLIN;
     ev.data.fd = src->cancellableFd;
     if (epoll_ctl (src->pollfd, EPOLL_CTL_ADD, src->cancellableFd, &ev)) {
       GST_ERROR_OBJECT(src, "Failed to add fd: %d to epoll, error: %d",
                        src->cancellableFd, errno);

       g_cancellable_release_fd (src->cancellable);
       src->cancellableFd = -1;
       g_object_unref (src->cancellable);
       src->cancellable = NULL;
       return FALSE;
     }
  } else {
    GST_ERROR_OBJECT (src, "Failed to get cancellable fd");
    g_object_unref (src->cancellable);
    src->cancellable = NULL;
    return FALSE;
  }

  return TRUE;
}

/**
 * @brief Frees the cancellable object and associated resources
 *
 * This function removes the cancellable's file descriptor from the epoll set,
 * releases the file descriptor, and unreferences the cancellable object.
 * It ensures proper cleanup of resources associated with the cancellable.
 *
 * @param src Pointer to the GstNvDsUdpSrc instance
 * @return TRUE if cancellable was freed successfully, FALSE otherwise
 */

static gboolean
nvdsudpsrc_free_cancellable (GstNvDsUdpSrc *src)
{
  g_return_val_if_fail (src != NULL, FALSE);

  if (src->cancellableFd > 0) {
    struct epoll_event ev;
    memset (&ev, 0, sizeof(ev));
    ev.events = EPOLLIN;
    ev.data.fd = src->cancellableFd;
    if (epoll_ctl (src->pollfd, EPOLL_CTL_DEL, src->cancellableFd, &ev)) {
      GST_ERROR_OBJECT(src, "Failed to remove fd: %d from epoll, error: %d",
                       src->cancellableFd, errno);
    }

    g_cancellable_release_fd (src->cancellable);
    src->cancellableFd = -1;
  }

  g_object_unref (src->cancellable);
  src->cancellable = NULL;

  return TRUE;
}

/**
 * @brief Initializes the event channel for stream monitoring
 *
 * This function sets up the event monitoring infrastructure for a Rivermax stream.
 * It creates an epoll instance if one doesn't exist, gets the event channel for the
 * specified stream, and adds it to the epoll monitoring set. It also ensures the
 * cancellable file descriptor is created for stream cancellation support.
 *
 * @param src Pointer to the GstNvDsUdpSrc instance
 * @param stream_id The Rivermax stream ID to monitor
 * @return TRUE if event channel was initialized successfully, FALSE otherwise
 *
 */
static gboolean
initialize_event_channel (GstNvDsUdpSrc *src, rmx_stream_id stream_id)
{
  rmx_status status;
  rmx_event_channel_params event_channel;
  rmx_event_channel_handle event_channel_handle;
  struct epoll_event ev;

  /* Create epoll instance only once */
  if (src->pollfd < 0) {
    src->pollfd = epoll_create1(0);
    if (src->pollfd < 0) {
      GST_ERROR_OBJECT(src, "Failed to create notification epoll file descriptor, error: %d", errno);
      return FALSE;
    }
  }

  /* Get event channel for this specific stream */
  rmx_init_event_channel(&event_channel, stream_id);
  rmx_set_event_channel_handle(&event_channel, &event_channel_handle);

  status = rmx_establish_event_channel(&event_channel);
  if (status != RMX_OK) {
    GST_ERROR_OBJECT(src, "Failed to establish event channel for stream, status: %d", status);
    close(src->pollfd);
    src->pollfd = -1;
    return FALSE;
  }

  /* Add the event channel to the epoll instance */
  memset(&ev, 0, sizeof(ev));
  ev.events = EPOLLIN | EPOLLOUT;
  ev.data.fd = event_channel_handle;
  if (epoll_ctl(src->pollfd, EPOLL_CTL_ADD, event_channel_handle, &ev)) {
    GST_ERROR_OBJECT(src, "Failed to add fd: %d to epoll, error: %d", event_channel_handle, errno);
    close(src->pollfd);
    src->pollfd = -1;
    return FALSE;
  }

  /* Create cancellableFd only once */
  if (src->cancellableFd < 0) {
    if (!nvdsudpsrc_create_cancellable(src)) {
      close(src->pollfd);
      src->pollfd = -1;
      return FALSE;
    }
  }

  return TRUE;
}

static GstCaps *
gst_nvdsudpsrc_get_caps (GstBaseSrc * bsrc, GstCaps * filter)
{
  GstNvDsUdpSrc *src;
  GstCaps *caps, *result;

  src = GST_NVDSUDPSRC (bsrc);

  GST_OBJECT_LOCK (src);
  caps = src->caps;
  GST_OBJECT_UNLOCK (src);

  if (caps) {
    if (filter) {
      result = gst_caps_intersect_full (filter, caps, GST_CAPS_INTERSECT_FIRST);
    } else {
      result = gst_caps_copy (caps);
    }

    GstStructure *structure = gst_caps_get_structure (result, 0);
    if (gst_structure_has_name (structure, "application/x-custom")) {
      gint memType = -1;
      gboolean ret = gst_structure_get_int (structure, "mem-type", &memType);
      if (ret && memType >= MEM_TYPE_HOST && memType < MEM_TYPE_UNKNOWN) {
        //we have user provided mem-type in caps.
        src->outputMemType = memType;
      } else {
        // we don't have mem-type in caps or there is incorrect value,
        // decide correct type based on the gpu-id property.
        if (src->gpuId < 0) {
          src->outputMemType = MEM_TYPE_HOST;
          memType = 0; // Host pinned memory as output.
        } else {
          src->outputMemType = MEM_TYPE_DEVICE;
          memType = 1; // Device memory as output.
        }
        gst_structure_set (structure, "mem-type", G_TYPE_INT, memType, NULL);
      }
    }
  } else {
    result = (filter) ? gst_caps_ref (filter) : gst_caps_new_any ();
  }
  return result;
}

static gboolean
nvdsudpsrc_parse_audio_params (GstNvDsUdpSrc *src, GstStructure *structure)
{
  const gchar *str;
  gint rate = 0, channels = 0, tmp = 0;
  gdouble ptime_in_ms = 0, tmp_d = 0;

  if ((str = gst_structure_get_string (structure, "rate"))) {
    rate = atoi (str);
  } else if (gst_structure_get_int (structure, "rate", &tmp)) {
    rate = tmp;
  } else {
    GST_ERROR_OBJECT (src, "no rate provided in caps");
    return FALSE;
  }
  src->clock_rate = rate;

  if ((str = gst_structure_get_string (structure, "channels"))) {
    channels = atoi (str);
  } else if (gst_structure_get_int (structure, "channels", &tmp)) {
    channels = tmp;
  } else {
    GST_ERROR_OBJECT (src, "no channels provided in caps");
    return FALSE;
  }

  if (!(str = gst_structure_get_string (structure, "format"))) {
    GST_ERROR_OBJECT (src, "No format in caps");
    return FALSE;
  }

  if (!g_strcmp0 (str, "S24BE")) {
    // frame size is one second of data.
    // It can be made configurable if required.
    src->frameSize = rate * channels * 3;
  } else if (!g_strcmp0 (str, "S16BE")) {
    src->frameSize = rate * channels * 2;
  } else {
    GST_ERROR_OBJECT (src, "format %s is not supported", str);
    return FALSE;
  }

  if ((str = gst_structure_get_string (structure, "ptime"))) {
    ptime_in_ms = strtof (str, NULL);
  } else if (gst_structure_get_int (structure, "ptime", &tmp)) {
    ptime_in_ms = tmp;
  } else if (gst_structure_get_double (structure, "ptime", &tmp_d)) {
    ptime_in_ms = tmp_d;
  } else {
    GST_WARNING_OBJECT (src, "No ptime provided in caps. framesize will be payloadSize * payMultiple. Adjust these params accordingly");
  }

  if (ptime_in_ms > 0) {
    /* 1 packet = <ptime> ms of audio data. */
    gdouble payload_size = src->frameSize * ((gdouble)ptime_in_ms / 1000);
    if (payload_size != (guint)payload_size) {
      GST_WARNING_OBJECT(src, "Payload size %f is not an integral multiple, so the data will be truncated", payload_size);
    }
    src->payloadSize = (guint)payload_size;
    /* payMultiple defines the number of packets to be sent downstream.
     Update frame size accordingly. */
    src->frameSize = src->payloadSize * src->payMultiple;
  } else {
    src->frameSize = src->payloadSize * src->payMultiple;
  }

  return TRUE;
}

static gboolean
nvdsudpsrc_parse_video_params (GstNvDsUdpSrc *src, GstStructure *structure)
{
  const gchar *str;
  gint width, height, tmp;
  guint stride;

  if ((str = gst_structure_get_string (structure, "width"))) {
    width = atoi (str);
  } else if (gst_structure_get_int (structure, "width", &tmp)) {
    width = tmp;
  } else {
    GST_ERROR_OBJECT (src, "no width in caps");
    return FALSE;
  }

  if ((str = gst_structure_get_string (structure, "height"))) {
    height = atoi (str);
  } else if (gst_structure_get_int (structure, "height", &tmp)) {
    height = tmp;
  } else {
    GST_ERROR_OBJECT (src, "no height in caps");
    return FALSE;
  }

  if (!(str = gst_structure_get_string (structure, "format"))) {
    GST_ERROR_OBJECT (src, "No format in caps");
    return FALSE;
  }

  if (!g_strcmp0 (str, "RGB")) {
    stride = width * 3;
  } else if (!g_strcmp0 (str, "UYVP")) {
    stride = width * 5 / 2;
  } else if (!g_strcmp0 (str, "UYVY")) {
    stride = width * 2;
  } else {
    GST_ERROR_OBJECT (src, "format %s is not supported", str);
    return FALSE;
  }

  if ((str = gst_structure_get_string (structure, "interlace-mode"))) {
    if (!g_strcmp0 (str, "interleaved"))
      src->videoType = INTERLACE;
  } else {
    src->videoType = PROGRESSIVE;
  }

  src->frameSize = stride * height;
  src->stride = stride;
  src->clock_rate = 90000;  /* Video clock rate is always 90kHz */
  return TRUE;
}

static gboolean
gst_nvdsudpsrc_start (GstBaseSrc *psrc)
{
  GstNvDsUdpSrc *src = (GstNvDsUdpSrc *) psrc;
  gboolean ret = TRUE;
  rmx_status status = RMX_OK;
  GstStructure *config;
  GstNvDsUdpBufferPoolPrivate *priv;
  guint size;

  src->isRtpOut = TRUE;

  GstCaps *caps = src->caps;
  if (caps) {
    GstStructure* structure = gst_caps_get_structure (caps, 0);
    const gchar* mimeType = gst_structure_get_name (structure);
    gboolean ret;

    if (!g_strcmp0 (mimeType, "video/x-raw") ||
        !g_strcmp0 (mimeType, "audio/x-raw") ||
        !g_strcmp0 (mimeType, "application/x-custom")) {

      if (!g_strcmp0 (mimeType, "video/x-raw")) {
        GstCapsFeatures *inFeature = gst_caps_features_new ("memory:NVMM", NULL);
        if (!gst_caps_features_is_equal (gst_caps_get_features (caps, 0), inFeature)) {
          src->is_nvmm = FALSE;
        }
        gst_caps_features_free (inFeature);

        src->streamType = VIDEO_2110_20_STREAM;
        src->videoType = PROGRESSIVE;
        ret = nvdsudpsrc_parse_video_params (src, structure);
      } else if (!g_strcmp0 (mimeType, "audio/x-raw")) {
        src->streamType = AUDIO_2110_30_31_STREAM;
        ret = nvdsudpsrc_parse_audio_params (src, structure);
      } else {
        src->streamType = APPLICATION_CUSTOM_STREAM;
        src->frameSize = src->payloadSize * src->payMultiple;
        if (src->outputMemType == MEM_TYPE_UNKNOWN) {
          if (src->gpuId >= 0) {
            src->outputMemType = MEM_TYPE_DEVICE;
          } else {
            src->outputMemType = MEM_TYPE_HOST;
          }
        }
        ret = TRUE;
      }
      if (!ret) {
        GST_ERROR_OBJECT (src, "failed to parse caps");
        return FALSE;
      }
      src->isRtpOut = FALSE;
    }
  }

  /* Handle ST2022-7 streams if specified */
  if (src->st2022_7_streams) {
    /* Parse streams string into DstStreamInfo structures */
    src->num_streams = parse_st2022_7_streams(src->st2022_7_streams, src->dstStream);
    if (src->num_streams == 0) {
      GST_ERROR_OBJECT (src, "Invalid number of streams: %u. Min: 1, Max: %u",
                        src->num_streams, MAX_ST2022_7_STREAMS);
      return FALSE;
    }
  }

  /* Handle multiple source addresses if specified */
  if (src->sourceAddresses) {
    /* Parse list of source addresses string into individual strings */
    src->num_source_addresses = parse_ip_addresses(src->sourceAddresses, src->srcAddress, "source");
    GST_DEBUG_OBJECT (src, "Parsed %u addresses for source filtering", src->num_source_addresses);
  }

  rmx_enable_system_signal_handling();
  status = rmx_init ();
  if (status != RMX_OK) {
    GST_ERROR_OBJECT (src, "Failed to initialize Rivermax - error %d", status);
    return FALSE;
  }

  if (src->ptpSrc) {
    rmx_ptp_clock_params clock;
    rmx_init_ptp_clock(&clock);
    struct in_addr ip;
    inet_pton(AF_INET, src->ptpSrc, &ip);
    status = rmx_retrieve_device_iface_ipv4(&src->device, &ip);
    if (status != RMX_OK) {
        GST_ERROR_OBJECT (src, "Failed to get device with ip: %s with status: %d", src->ptpSrc, status);
        return FALSE;
    }

    rmx_set_ptp_clock_device(&clock, &src->device);
    status = rmx_use_ptp_clock(&clock);

    GST_DEBUG_OBJECT (src, "rmax_set_clock(RIVERMAX_PTP_CLOCK) status: %d", status);
    /* If multiple instances are running, the clock return busy while trying to set the clock second time.
      Ignore the busy status in this case. */
    if ((status != RMX_OK) && (status != RMX_BUSY)) {
      GST_WARNING_OBJECT (src, "Failed to set PTP clock - status %d", status);
      GST_WARNING_OBJECT (src, "PTP clock is not supported, using SYSTEM clock");

      g_free (src->ptpSrc);
      src->ptpSrc = NULL;
    }
  }

  ret = nvdsudpsrc_allocate_memory (src);
  if (!ret) {
    GST_ERROR_OBJECT (src, "Failed to register memory");
    rmx_cleanup ();
    return FALSE;
  }

  struct sockaddr_in local_nic_addr;
  memset(&local_nic_addr, 0, sizeof(local_nic_addr));
  local_nic_addr.sin_family = AF_INET;

  /*If st2022-7 mode is enabled, num_streams=num of redundant streams provided, else it is 1 */
  for (guint i = 0; i < src->num_streams; i++)
  {
    local_nic_addr.sin_addr.s_addr = inet_addr(src->localIfaceIp[i]);
    if (src->st2022_7_streams) {
      local_nic_addr.sin_port = htons(src->dstStream[i].port);
    }

    /*HW 2022-7 implementation is only supported now.
      Same memory region will be used by redundant streams */
    status = create_stream(src, &local_nic_addr, &src->streamId[i], i);
    if (status != RMX_OK) {
      GST_ERROR_OBJECT(src, "Failed to create primary stream - error %d", status);
      deallocate_buffer(src);
      rmx_cleanup();
      return FALSE;
    }

    /* Initialize and configure flow for stream i using new API (following media_receiver.cpp pattern) */
    rmx_input_flow *recv_flow = &src->flow[i];
    rmx_input_init_flow(recv_flow);

    struct sockaddr_in flow_local_addr;
    memset(&flow_local_addr, 0, sizeof(flow_local_addr));
    flow_local_addr.sin_family = AF_INET;

    if (src->st2022_7_streams) {
      flow_local_addr.sin_addr.s_addr = inet_addr(src->dstStream[i].ip);
      flow_local_addr.sin_port = htons(src->dstStream[i].port);
    } else {
      if (!g_strcmp0(src->address, "0.0.0.0"))
        flow_local_addr.sin_addr.s_addr = inet_addr(src->localIfaceIp[i]);
      else
        flow_local_addr.sin_addr.s_addr = inet_addr(src->address);
      flow_local_addr.sin_port = htons((uint16_t)src->port);
    }

    rmx_input_set_flow_local_addr(recv_flow, (const struct sockaddr *)&flow_local_addr);

    if (src->srcAddress[i]) {
      struct sockaddr_in flow_remote_addr;
      memset(&flow_remote_addr, 0, sizeof(flow_remote_addr));
      flow_remote_addr.sin_family = AF_INET;
      flow_remote_addr.sin_addr.s_addr = inet_addr(src->srcAddress[i]);
      rmx_input_set_flow_remote_addr(recv_flow, (const struct sockaddr *)&flow_remote_addr);
    }

    rmx_input_set_flow_tag(recv_flow, src->flowId + i);

    /* Initialize event channel for this stream */
    ret = initialize_event_channel(src, src->streamId[i]);
    if (!ret) {
      GST_ERROR_OBJECT(src, "Failed to initialize event channel for stream %d", i);
      deallocate_buffer(src);
      rmx_cleanup();
      return FALSE;
    }
  }

  if (!src->isRtpOut && src->streamType == VIDEO_2110_20_STREAM) {
    if (src->is_nvmm) {
        src->pool = gst_nvds_buffer_pool_new ();
    } else {
        // Use regular GStreamer buffer pool for non-NVMM memory
        src->pool = gst_buffer_pool_new ();
    }
  } else {
    src->pool = gst_nvds_udp_buffer_pool_new ();
  }

  if (!src->pool) {
    GST_ERROR_OBJECT (src, "failed to create internal pool");
    deallocate_buffer (src);
    rmx_cleanup ();
    return FALSE;
  }

  config = gst_buffer_pool_get_config (src->pool);
  if (!src->isRtpOut && src->streamType == VIDEO_2110_20_STREAM) {
    gst_caps_ref (src->caps);

    if (src->is_nvmm) {
        gst_buffer_pool_config_set_params (config, src->caps, sizeof (NvBufSurface), 6, 0);
        gint gpuId = src->gpuId;

        if (gpuId == -1) {
            GST_WARNING_OBJECT (src, "gpu-id property is not set, using 0 as device id for output buffers");
            gpuId = 0;
        }
        gst_structure_set (config, "gpu-id", G_TYPE_UINT, gpuId, "memtype",
            G_TYPE_UINT, NVBUF_MEM_CUDA_DEVICE, "disable-pitch-padding", G_TYPE_BOOLEAN, TRUE, NULL);
    } else {
        // Configure regular GStreamer buffer pool for non-NVMM memory
        GST_DEBUG_OBJECT (src, "Using standard GStreamer buffer pool with system memory");
        gst_buffer_pool_config_set_params (config, src->caps, src->frameSize, 6, 0);
    }
  } else {
    priv = ((GstNvDsUdpBufferPool *) src->pool)->priv;
    priv->hdrSize = src->headerSize;
    priv->payloadSize = src->payloadSize;
    priv->isRtpOut = src->isRtpOut;

    if (src->isRtpOut) {
      size = src->payloadSize + src->headerSize;
    } else {
      size = src->frameSize;
      priv->frameSize = size;
      // need to allocate the memory for output frames.
      ((GstNvDsCudaMemoryAllocator *) priv->allocator)->allocate_memory = TRUE;
      if (src->streamType == APPLICATION_CUSTOM_STREAM &&
          src->outputMemType != MEM_TYPE_DEVICE) {
        // In case of custom stream, if user has asked for Host pinned memory
        // as output, we prefer that even if gpu-id property is set.
        ((GstNvDsCudaMemoryAllocator *) priv->allocator)->gpuId = -1;
      } else {
        ((GstNvDsCudaMemoryAllocator *) priv->allocator)->gpuId = src->gpuId;
      }
    }
    gst_buffer_pool_config_set_params (config, NULL, size, 10, 0);
    gst_buffer_pool_config_set_allocator (config, priv->allocator, &priv->params);
  }

  gst_buffer_pool_set_config (src->pool, config);
  gst_buffer_pool_set_active (src->pool, TRUE);

  src->bufferQ = g_queue_new ();
  src->isRunning = TRUE;
  src->pThread = g_thread_new (NULL, nvdsudpsr_data_fetch_loop, src);

  return TRUE;
}

static gboolean
gst_nvdsudpsrc_stop (GstBaseSrc *psrc)
{
  GstNvDsUdpSrc *src = (GstNvDsUdpSrc *) psrc;
  src->isRunning = FALSE;
  g_cancellable_cancel (src->cancellable);
  g_cond_signal (&src->qCond);
  gst_buffer_pool_set_active (src->pool, FALSE);

  if (src->pThread)
    g_thread_join (src->pThread);

  nvdsudpsrc_free_cancellable (src);

  if (src->pollfd > 0) {
    close (src->pollfd);
    src->pollfd = -1;
  }

  /* Destroy all streams */
  for (guint i = 0; i < src->num_streams; i++) {
    if (src->streamId[i] != INVALID_STREAM_ID) {
      destroy_stream(src->streamId[i]);
      src->streamId[i] = INVALID_STREAM_ID;
    }
  }

  deallocate_buffer (src);
  rmx_cleanup ();

  g_mutex_lock (&src->qLock);
  if (src->isRtpOut)
    g_queue_free_full (src->bufferQ, (GDestroyNotify) gst_buffer_list_unref);
  else
    g_queue_free_full (src->bufferQ, (GDestroyNotify) gst_buffer_unref);
  g_mutex_unlock (&src->qLock);

  src->dataPtr1 = src->dataPtr2 = NULL;
  src->len1 = src->len2 = 0;
  src->mBit = src->ffFound = 0;
  src->packetCounter = 0;

  return TRUE;
}

static GstFlowReturn
gst_nvdsudpsrc_create (GstPushSrc * psrc, GstBuffer ** buffer)
{
  GstNvDsUdpSrc *src = (GstNvDsUdpSrc *) psrc;
  GstBufferList *bufList = NULL;
  GstBuffer *buf = NULL;

  if (src->lastError) {
    return GST_FLOW_ERROR;
  }

  g_mutex_lock (&src->qLock);
  while (src->isRunning && !src->lastError && g_queue_is_empty (src->bufferQ)) {
    g_mutex_lock (&src->flowLock);
    if (src->flushing) {
      g_mutex_unlock (&src->flowLock);
      g_mutex_unlock (&src->qLock);
      return GST_FLOW_FLUSHING;
    }
    g_mutex_unlock (&src->flowLock);
    g_cond_wait (&src->qCond, &src->qLock);
  }

  if (!src->isRunning) {
    g_mutex_unlock (&src->qLock);
    return GST_FLOW_EOS;
  }

  if (src->lastError) {
    g_mutex_unlock (&src->qLock);
    return GST_FLOW_ERROR;
  }

  g_mutex_lock (&src->flowLock);
  if (src->flushing) {
    g_mutex_unlock (&src->flowLock);
    g_mutex_unlock (&src->qLock);
    return GST_FLOW_FLUSHING;
  }
  g_mutex_unlock (&src->flowLock);

  if (src->isRtpOut) {
    bufList = (GstBufferList *) g_queue_pop_head (src->bufferQ);
    gst_base_src_submit_buffer_list ((GstBaseSrc *)src, bufList);
  } else {
    buf = (GstBuffer *) g_queue_pop_head (src->bufferQ);
    *buffer = buf;
  }
  g_mutex_unlock (&src->qLock);
  return GST_FLOW_OK;
}

static gboolean
nvdsudpsrc_timed_wait (GstNvDsUdpSrc *src)
{
  rmx_status status[MAX_ST2022_7_STREAMS];
  rmx_notification_params notification[MAX_ST2022_7_STREAMS];
  guint rmax_ok_count = 0;
  guint rmax_err_count = 0;

  /* Request notifications from both streams */
  for (guint i = 0; i < src->num_streams; i++) {
    rmx_init_notification(&notification[i], src->streamId[i]);
    status[i] = rmx_request_notification(&notification[i]);
    if (status[i] == RMX_SIGNAL) {
      /* If any of the streams captured Ctrl-C from Rivermax, execute that on priority */
      g_mutex_lock (&src->flowLock);
      src->isRunning = FALSE; /* This will be handled by caller */
      g_mutex_unlock (&src->flowLock);
      return TRUE;
    }
    if (status[i] == RMX_OK) {
      rmax_ok_count++;
    } else if (status[i] != RMX_BUSY) {
      rmax_err_count++;
    }
  }

  /* If both of the streams triggered RMX_OK, wait for event. If any of the status is RMX_BUSY that means
     data is available on the channel. */
  if (rmax_ok_count == src->num_streams) {
    struct epoll_event ev;
    memset(&ev, 0, sizeof(ev));
    ev.events = EPOLLIN | EPOLLOUT;
    /*allocated memory for 1 ev instance i.e., 16 bytes but MAX_EVENTS=2 will cause buffer overflow.*/
    gint ret = epoll_wait(src->pollfd, &ev, 1, -1);
    if (ret < 0) {
      if (EINTR != errno) {
        GST_ERROR_OBJECT(src, "Error in epoll wait, error: %d", errno);
        return FALSE;
      }
    }
  } else if (rmax_err_count == src->num_streams) {
    /* If all of the streams failed to request notification then return false */
    GST_ERROR_OBJECT(src, "Error: in request notification for streamIds");
    return FALSE;
  }

  return TRUE;
}

/**
 * @brief Validates and processes RTP header information.
 *
 * This function analyzes the RTP header to determine packet boundaries and frame
 * structure for different stream types. It handles both video (ST2110-20) and audio
 * streams, tracking frame boundaries and packet ordering.
 *
 * @param rtp_hdr Pointer to RTP header data
 * @param rtp_data Pointer to RTP payload data
 * @param data_size Size of the payload data
 * @param src Pointer to GstNvDsUdpSrc instance
 * @return TRUE if header is valid and processing should continue, FALSE otherwise
 */
static gboolean
check_rtp_header (uint8_t *rtp_hdr, uint8_t *rtp_data, guint16 data_size, GstNvDsUdpSrc *src)
{
  gboolean mBit = 0;
  gboolean fBit = 0;
  guint16 size;
  // Reset mBit to default state
  src->mBit = FALSE;
  if (src->streamType == VIDEO_2110_20_STREAM) {
    mBit = !!(rtp_hdr[1] & 0x80);
    fBit = !!(rtp_hdr[16] & 0x80);
    size = g_ntohs (*(guint16 *)(rtp_hdr + 14));
  } else {
    size = data_size;
  }

  if (size != src->payloadSize)
      GST_WARNING ("received payload size %u not equal to defined payload size %u",
          size, src->payloadSize);

  // Accumulate packets for all stream types and all modes
  src->packetCounter++;

  if (src->packetCounter == 1) {
    src->dataPtr1 = rtp_data;
    src->len1 = size;
  } else {
    if (src->dataPtr2 == NULL && rtp_data < src->dataPtr1) {
      // wrap around case
      src->dataPtr2 = rtp_data;
      src->len2 = size;
    } else if (src->dataPtr2 == NULL) {
      src->len1 += size;
    } else {
      src->len2 += size;
    }
  }

  // In case of audio and custom type, m_bit might not be set.
  // Frame completion is based on the number of packets received.
  if (src->streamType != VIDEO_2110_20_STREAM) {
    if (src->packetCounter == src->payMultiple) {
      mBit = TRUE;
    }
  }

  // Check for excessive data accumulation (video streams only)
  if ((src->streamType == VIDEO_2110_20_STREAM) &&
      (src->len1 + src->len2) > src->frameSize) {
    GST_WARNING_OBJECT (src, "received (%u) bytes but no m_bit or f_bit in header",
          src->len1 + src->len2);
  }

  // Handle frame completion based on mBit
  if (mBit) {
    // Handle interlaced video: need both fields
    if (src->streamType == VIDEO_2110_20_STREAM &&
        src->videoType == INTERLACE && !fBit) {
      // This is end of first field, not complete frame yet
        GST_INFO_OBJECT (src, "First field end, waiting for second field");
      return TRUE;
    }

    // First frame vs subsequent frames handling
    if (!src->ffFound && src->streamType == VIDEO_2110_20_STREAM) {
      src->ffFound = TRUE;
      // For first frame, copy the packets to the downstream frame buffer only if complete frame is received.
      GST_INFO_OBJECT (src, "First frame mBit found, validating size");
      if ((src->len1 + src->len2) == src->frameSize) {
        src->mBit = TRUE;
        GST_INFO_OBJECT (src, "First valid frame found with correct size %u bytes",
            src->len1 + src->len2);
      } else {
        // Size mismatch - discard and start over
        GST_WARNING_OBJECT (src, "Partial frame received: (%u) bytes while (%u) bytes were expected, discarding",
          src->len1 + src->len2, src->frameSize);
        //Reset the frame pointer and length
        src->dataPtr1 = src->dataPtr2 = NULL;
        src->len1 = src->len2 = 0;
      }
    } else {
      // For subsequent frames or non-video type, print warning if size mismatch but still copy the packets to the downstream frame buffer.
      if ((src->len1 + src->len2) != src->frameSize) {
        GST_WARNING_OBJECT (src, "received (%u) bytes in a frame while (%u) bytes were expected",
          src->len1 + src->len2, src->frameSize);
      }
      src->mBit = TRUE;
    }
    // Reset the packet counter once mBit is found for the frame.
    src->packetCounter = 0;
  }
  return TRUE;
}

// Helper function for memory copying that handles both NVMM and standard memory
static gboolean memcopy_2d(void *dst, guint dst_pitch, void *src_ptr, guint src_pitch,
                    guint width, guint height, GstNvDsUdpSrc *src) {
  cudaError_t ret = cudaSuccess;

  if (src->is_nvmm || src->isGpuDirect) {
    ret = cudaMemcpy2D(dst, dst_pitch, src_ptr, src_pitch, width, height, cudaMemcpyDefault);
    if (ret != cudaSuccess) {
      GST_ERROR_OBJECT(src, "failed to copy memory: %s", cudaGetErrorString(ret));
      return FALSE;
    }
    ret = cudaDeviceSynchronize();
    if (ret != cudaSuccess) {
      GST_ERROR_OBJECT(src, "failed to synchronize: %s", cudaGetErrorString(ret));
      return FALSE;
    }
  } else {
    // For standard memory, use regular memcpy for each line
    uint8_t *d = (uint8_t*)dst;
    uint8_t *s = (uint8_t*)src_ptr;
    for (guint line = 0; line < height; line++) {
      memcpy(d, s, width);
      d += dst_pitch;
      s += src_pitch;
    }
  }

  return TRUE;
}

// Helper function for linear memory copying
static gboolean memcopy_linear(void *dst, void *src_ptr, guint size, GstNvDsUdpSrc *src) {
  cudaError_t ret = cudaSuccess;

  if (src->is_nvmm || src->isGpuDirect) {
    ret = cudaMemcpy(dst, src_ptr, size, cudaMemcpyDefault);
    if (ret != cudaSuccess) {
      GST_ERROR_OBJECT(src, "failed to copy memory: %s", cudaGetErrorString(ret));
      return FALSE;
    }
    ret = cudaDeviceSynchronize();
    if (ret != cudaSuccess) {
      GST_ERROR_OBJECT(src, "failed to synchronize: %s", cudaGetErrorString(ret));
      return FALSE;
    }
  } else {
    // For standard memory, use regular memcpy
    memcpy(dst, src_ptr, size);
  }

  return TRUE;
}

/**
 * @brief Copies an interlaced frame from source to destination buffer
 *
 * This function handles copying interlaced frame data from the source buffer to the destination buffer.
 * It supports both regular and wrap-around cases for frame copying. For interlaced frames, it copies
 * data in two fields - first field and second field. Each field is copied separately to maintain
 * the interlaced format.
 *
 * For wrap-around cases, it handles copying data that spans across the buffer boundary by copying
 * from both the initial buffer and overflow buffer.
 *
 * @param dataPtr Destination buffer pointer to copy frame data into
 * @param src Source element containing frame data to copy
 * @return TRUE if copy was successful, FALSE otherwise
 */

static gboolean
copy_interlace_frame (void *dataPtr, GstNvDsUdpSrc *src)
{
  guint size = 0;
  gpointer dstPtr = NULL;
  gpointer srcPtr = NULL;
  guint frameHeight = src->frameSize / src->stride;
  guint fieldHeight = frameHeight / 2;
  guint dstPitch = src->stride * 2;
  guint srcPitch = src->stride;
  guint widthInBytes = src->stride;

  if (src->len2) {
    // wrap around case
    guint row = src->len1 / src->stride;
    guint extraBytes = src->len1 % src->stride;
    guint remainingBytes = 0;
    guint remainingLines = 0;
    guint extraLine = extraBytes ? 1 : 0;

    // more lines than half frame + few packets
    if (row > fieldHeight) {
      size = fieldHeight * src->stride;
      dstPtr = dataPtr;
      srcPtr = src->dataPtr1;
      // first field copy from first pointer
      if (!memcopy_2d(dstPtr, dstPitch, srcPtr, srcPitch, widthInBytes, fieldHeight, src)) {
        GST_ERROR_OBJECT(src, "Failed in memcopy_2d: first field");
        return FALSE;
      }

      remainingLines = row - fieldHeight;
      dstPtr = (uint8_t *) dataPtr + src->stride;
      srcPtr = src->dataPtr1 + size;

      // second field copy from first pointer.
      if (!memcopy_2d(dstPtr, dstPitch, srcPtr, srcPitch, widthInBytes, remainingLines, src)) {
        GST_ERROR_OBJECT(src, "Failed in memcopy_2d: second field");
        return FALSE;
      }

      size = remainingLines * src->stride;
      if (extraBytes) {
        dstPtr = (uint8_t *) dataPtr + src->stride + size * 2;
        srcPtr = src->dataPtr1 + row * src->stride;

        // copy extra bytes of a next line from first pointer.
        if (!memcopy_linear(dstPtr, srcPtr, extraBytes, src)) {
          GST_ERROR_OBJECT(src, "Failed in memcopy_linear: second field : extra bytes");
          return FALSE;
        }

        // copy remaining bytes of same line from second pointer.
        remainingBytes = src->stride - extraBytes;
        dstPtr = (uint8_t *) dstPtr + extraBytes;
        srcPtr = src->dataPtr2;
        if (!memcopy_linear(dstPtr, srcPtr, remainingBytes, src)) {
          GST_ERROR_OBJECT(src, "Failed in memcopy_linear: second field : remaining bytes");
          return FALSE;
        }
      }

      // remaining lines of second field
      remainingLines = fieldHeight - remainingLines - extraLine;
      dstPtr = (uint8_t *) dataPtr + src->stride + (size + src->stride * extraLine) * 2;
      srcPtr = src->dataPtr2 + remainingBytes;

      // remaining second field copy from second pointer.
      if (remainingLines) {
        if (!memcopy_2d(dstPtr, dstPitch, srcPtr, srcPitch, widthInBytes, remainingLines, src)) {
          GST_ERROR_OBJECT(src, "Failed in memcopy_2d: second field : remaining lines");
          return FALSE;
        }
      }
    } else if (row < fieldHeight) {
      // less lines than half frame + few packets
      size = row * src->stride;
      remainingLines = fieldHeight - row;
      dstPtr = dataPtr;
      srcPtr = src->dataPtr1;

      // first field copy from first pointer
      if (!memcopy_2d(dstPtr, dstPitch, srcPtr, srcPitch, widthInBytes, row, src)) {
        GST_ERROR_OBJECT(src, "Failed in memcopy_2d: first field");
        return FALSE;
      }

      if (extraBytes) {
        // copy extra bytes of a next line from first pointer.
        dstPtr = (uint8_t *) dataPtr + size * 2;
        srcPtr = src->dataPtr1 + size;
        if (!memcopy_linear(dstPtr, srcPtr, extraBytes, src)) {
          GST_ERROR_OBJECT(src, "Failed in memcopy_linear: first field : extra bytes");
          return FALSE;
        }

        // copy remaining bytes of same line from second pointer.
        remainingBytes = src->stride - extraBytes;
        dstPtr = (uint8_t *) dataPtr + size * 2 + extraBytes;
        srcPtr = src->dataPtr2;
        if (!memcopy_linear(dstPtr, srcPtr, remainingBytes, src)) {
          GST_ERROR_OBJECT(src, "Failed in memcopy_linear: first field : remaining bytes");
          return FALSE;
        }
      }

      remainingLines = remainingLines - extraLine;
      if (remainingLines) {
        // remaining first field copy from second pointer
        dstPtr = (uint8_t *) dataPtr + (row + extraLine) * src->stride * 2;
        srcPtr = src->dataPtr2 + remainingBytes;
        if (!memcopy_2d(dstPtr, dstPitch, srcPtr, srcPitch, widthInBytes, remainingLines, src)) {
          GST_ERROR_OBJECT(src, "Failed in memcopy_2d: first field : remaining lines");
          return FALSE;
        }
      }

      // data that was copied for first field from second pointer.
      size = remainingLines * src->stride + remainingBytes;
      // second field copy from second pointer
      dstPtr = (uint8_t *) dataPtr + src->stride;
      srcPtr = src->dataPtr2 + size;
      if (!memcopy_2d(dstPtr, dstPitch, srcPtr, srcPitch, widthInBytes, fieldHeight, src)) {
        GST_ERROR_OBJECT(src, "Failed in memcopy_2d: second field");
        return FALSE;
      }
    } else {
      // same lines as half frame + few packets
      // copy first field from first pointer.
      size = row * src->stride;
      dstPtr = dataPtr;
      srcPtr = src->dataPtr1;
      if (!memcopy_2d(dstPtr, dstPitch, srcPtr, srcPitch, widthInBytes, fieldHeight, src)) {
        GST_ERROR_OBJECT(src, "Failed in memcopy_2d: first field");
        return FALSE;
      }

      if (extraBytes) {
        // copy bytes of second field from first pointer
        dstPtr = (uint8_t *) dataPtr + src->stride;
        srcPtr = src->dataPtr1 + size;
        if (!memcopy_linear(dstPtr, srcPtr, extraBytes, src)) {
          GST_ERROR_OBJECT(src, "Failed in memcopy_linear: second field: extra bytes");
          return FALSE;
        }

        // copy remaining bytes of same line from second pointer.
        remainingBytes = src->stride - extraBytes;
        dstPtr = (uint8_t *) dataPtr + src->stride + extraBytes;
        srcPtr = src->dataPtr2;
        if (!memcopy_linear(dstPtr, srcPtr, remainingBytes, src)) {
          GST_ERROR_OBJECT(src, "Failed in memcopy_linear: second field: remaining bytes");
          return FALSE;
        }
      }

      remainingLines = fieldHeight - extraLine;
      // copy remaining lines of second field from second pointer
      dstPtr = (uint8_t *) dataPtr + src->stride + src->stride * 2;
      srcPtr = src->dataPtr2 + remainingBytes;
      if (!memcopy_2d(dstPtr, dstPitch, srcPtr, srcPitch, widthInBytes, remainingLines, src)) {
        GST_ERROR_OBJECT(src, "Failed in memcopy_2d: second field : remaining lines");
        return FALSE;
      }
    }
  } else {
    size = fieldHeight * src->stride;
    dstPtr = dataPtr;
    srcPtr = src->dataPtr1;
    // copy of first field of the frame.
    if (!memcopy_2d(dstPtr, dstPitch, srcPtr, srcPitch, widthInBytes, fieldHeight, src)) {
      GST_ERROR_OBJECT(src, "Failed in memcopy_2d: first field of interlace frame");
      return FALSE;
    }

    // copy of second field of the frame.
    dstPtr = (uint8_t *) dataPtr + src->stride;
    srcPtr = src->dataPtr1 + size;
    if (!memcopy_2d(dstPtr, dstPitch, srcPtr, srcPitch, widthInBytes, fieldHeight, src)) {
      GST_ERROR_OBJECT(src, "Failed in memcopy_2d: second field of interlace frame");
      return FALSE;
    }
  }

  return TRUE;
}

/**
 * @brief Copies a frame from source to destination buffer
 *
 * This function handles copying frame data from the source buffer to the destination buffer.
 * It supports both regular and wrap-around cases for frame copying. For wrap-around cases,
 * it copies data in two parts - first from the initial buffer and then from the overflow buffer.
 *
 * @param dstBuf Destination buffer to copy frame data into
 * @param src Source element containing frame data to copy
 * @return TRUE if copy was successful, FALSE otherwise
 */
static gboolean
copy_frame (GstBuffer *dstBuf, GstNvDsUdpSrc *src)
{
  guint size = 0;
  void *dataPtr = NULL;
  GstMapInfo map_info = GST_MAP_INFO_INIT;

  GstMemory *mem = gst_buffer_peek_memory (dstBuf, 0);
  gst_memory_map (mem, &map_info, GST_MAP_WRITE);

  if (src->streamType == VIDEO_2110_20_STREAM) {
    if (src->is_nvmm) {
      NvBufSurface *surf;
      surf = (NvBufSurface *) map_info.data;
      dataPtr = surf->surfaceList[0].dataPtr;
    } else {
      // For standard GStreamer buffer, use the mapped data directly
      dataPtr = map_info.data;
    }
  } else {
    dataPtr = map_info.data;
  }
  gst_memory_unmap (mem, &map_info);

  if (src->streamType == VIDEO_2110_20_STREAM && src->videoType == INTERLACE) {
    return copy_interlace_frame (dataPtr, src);
  }

  if (src->len2) {
    if (src->len1 <= src->frameSize)
      size = src->len1;
    else
      size = src->frameSize;

    if (!memcopy_linear(dataPtr, src->dataPtr1, size, src)) {
      GST_ERROR_OBJECT(src, "Wrap around case: Failed in first memcopy_linear");
      return FALSE;
    }

    guint remainder = src->frameSize - size;
    if (remainder > 0) {
      if (!memcopy_linear(((uint8_t*)(dataPtr) + size), src->dataPtr2, remainder, src)) {
        GST_ERROR_OBJECT(src, "Wrap around case: Failed in second memcopy_linear");
        return FALSE;
      }
    }
  } else {
    if (src->len1 <= src->frameSize)
      size = src->len1;
    else
      size = src->frameSize;

    if (!memcopy_linear(dataPtr, src->dataPtr1, size, src)) {
      GST_ERROR_OBJECT(src, "Failed in memcopy_linear");
      return FALSE;
    }
  }

  return TRUE;
}

static GstClockTime
rtp_ticks_to_running_time(GstNvDsUdpSrc *src, guint32 rtp_tick, guint32 clock_rate)
{
    guint64 extended_rtp_tick=0;

    if (src->first_rtp_packet) {
        //GstClockTime now = get_realtime_in_ns();
        GstClock *clock = gst_system_clock_obtain();
        GstClockTime now = gst_clock_get_time (clock);
        gst_object_unref (clock);
        GST_DEBUG_OBJECT(src, " first_rtp_packet. now in HH:MM:SS.NS: %" GST_TIME_FORMAT,
                        GST_TIME_ARGS(now));
        // Convert current time to RTP ticks
        src->rtp_tick_base = gst_util_uint64_scale_int(now, clock_rate, GST_SECOND);
        // Mask out lower 32 bits.
        src->rtp_tick_base = (src->rtp_tick_base & ~((guint64)G_MAXUINT32));
        src->last_rtp_tick = rtp_tick;
        //This is the 64 bit RTP ticks.
        extended_rtp_tick = src->rtp_tick_base + rtp_tick;
        src->first_rtp_packet = FALSE;
    } else {
        // Handle wraparound
        if (rtp_tick < src->last_rtp_tick &&
            (src->last_rtp_tick - rtp_tick) > (G_MAXUINT32 / 2)) {
            // Timestamp wrapped around, increment high bits
            src->rtp_tick_base += ((guint64)1 << 32);
            src->last_rtp_tick = rtp_tick;
            extended_rtp_tick = src->rtp_tick_base + rtp_tick;
        } else if (rtp_tick > src->last_rtp_tick &&
                  (rtp_tick - src->last_rtp_tick) > (G_MAXUINT32 / 2)) {
            // Out of order packet from previous wrap
            extended_rtp_tick = rtp_tick + (src->rtp_tick_base - ((guint64)1 << 32));
        } else {
          //normal case
          src->last_rtp_tick = rtp_tick;
          extended_rtp_tick = src->rtp_tick_base + rtp_tick;
        }
    }
    //Base time of the element
    GstClockTime base_time = gst_element_get_base_time(GST_ELEMENT(src));
    // Convert to GstClockTime (nanoseconds)
    GstClockTime rtp_time_in_ns = gst_util_uint64_scale_int(extended_rtp_tick, GST_SECOND, clock_rate);
    GstClockTime running_time = GST_CLOCK_DIFF(base_time, rtp_time_in_ns);
    if (src->adjust_leap_seconds) {
        running_time -= (LEAP_SECONDS*GST_SECOND);
        GST_DEBUG_OBJECT(src, "Adjusted for leap seconds");
    }

    // Apply latency offset if configured
    if (src->latency_offset > 0) {
        running_time += src->latency_offset;
        GST_DEBUG_OBJECT(src, "Applied buffer PTS offset: %" G_GUINT64_FORMAT " ns, new buffer PTS: %" GST_TIME_FORMAT,
                        src->latency_offset, GST_TIME_ARGS(running_time));
    }

    GST_DEBUG_OBJECT(src, "RTP ticks received: %u, RTP time: %" GST_TIME_FORMAT ", Base time: %" GST_TIME_FORMAT ", Final buffer PTS: %" GST_TIME_FORMAT,
                    rtp_tick,
                    GST_TIME_ARGS(rtp_time_in_ns),
                    GST_TIME_ARGS(base_time),//GST_TIME_ARGS(src->real_base_time),
                    GST_TIME_ARGS(running_time));

    return running_time;
}

static gboolean
nvdsudpsrc_push_frame (const rmx_input_completion *comp, GstNvDsUdpSrc *src)
{
  guint i;
  gboolean ret;
  guint chunk_size = rmx_input_get_completion_chunk_size(comp);
  uint8_t *base_data_ptr = (uint8_t *)rmx_input_get_completion_ptr(comp, src->payload_mem_block_id);
  uint8_t *base_hdr_ptr = src->headerSize > 0 ?
                          (uint8_t *)rmx_input_get_completion_ptr(comp, src->header_mem_block_id) : NULL;

  for (i = 0; i < chunk_size; i++) {
    uint8_t *data_stride = base_data_ptr + i * src->data_stride_size;
    uint8_t *hdr_stride;
    uint8_t *app_hdr;
    uint8_t *app_data;
    GstBuffer *buf = NULL;

    const rmx_input_packet_info *packet_info = rmx_input_get_packet_info(&src->chunk_handle[0], i);

    if (base_hdr_ptr) {
      hdr_stride = base_hdr_ptr + i * src->hdr_stride_size;
      app_hdr = hdr_stride;
      app_data = data_stride;
    } else {
      hdr_stride = NULL;
      app_hdr = data_stride;
      app_data = get_rtp_hdr_ptr (data_stride, RMX_INPUT_APP_PROTOCOL_PACKET);
    }

    guint16 data_size = rmx_input_get_packet_size(packet_info, src->payload_mem_block_id);
    check_rtp_header (app_hdr, app_data, data_size, src);
    if (src->mBit) {
      // we have received all the packets of a frame.
      GstFlowReturn flowRet = gst_buffer_pool_acquire_buffer (src->pool, &buf, NULL);
      if (flowRet == GST_FLOW_FLUSHING || flowRet == GST_FLOW_EOS) {
        /* This is internal pool with unlimited buffers.
         * It will go in FLUSHING / EOS only in inactive state.
         * Pool is inactivated during stop. So come out of the loop.
         */
        return FALSE;
      }

      if (flowRet != GST_FLOW_OK || buf == NULL) {
        src->lastError = -1;
        GST_ERROR_OBJECT (src, "Failed to get buffer from pool:%d\n", flowRet);
        return FALSE;
      }

      ret = copy_frame (buf, src);
      if (!ret) {
        GST_WARNING_OBJECT (src, "frame copy failed, buffer might have garbage data");
      }

      if (src->use_rtp_timestamp) {
        /* Parse 32 bit RTP ticks from rtp header */
        guint32 rtp_time_ticks = GST_READ_UINT32_BE(app_hdr + 4);
        /* Calculate running time from RTP ticks */
        GstClockTime running_time = rtp_ticks_to_running_time(src, rtp_time_ticks, src->clock_rate);
        /* Set buffer timestamp = running time */
        GST_BUFFER_PTS(buf) = running_time;
        /* DTS is usually set to same as PTS */
        GST_BUFFER_DTS(buf) = GST_BUFFER_PTS(buf);

        /* Attach RTP ticks as metadata */
        GstRTPTimestampMeta *meta = gst_buffer_add_rtp_timestamp_meta(buf, rtp_time_ticks);
        if (!meta) {
          GST_WARNING_OBJECT(src, "Failed to add RTP timestamp metadata");
        } else {
          meta->leap_seconds_adjusted = src->adjust_leap_seconds;
        }
      }

      g_mutex_lock (&src->qLock);
      g_queue_push_tail (src->bufferQ, buf);
      g_cond_signal (&src->qCond);
      g_mutex_unlock (&src->qLock);

      // Existing frame copied, reset the state.
      src->dataPtr1 = src->dataPtr2 = NULL;
      src->len1 = src->len2 = 0;
    }
  }

  return TRUE;
}

static gboolean
nvdsudpsrc_push_rtp_packets (const rmx_input_completion *comp, GstNvDsUdpSrc *src)
{
  guint i;
  GstBufferList *bufList = NULL;
  guint chunk_size = rmx_input_get_completion_chunk_size(comp);
  uint8_t *base_data_ptr = (uint8_t *)rmx_input_get_completion_ptr(comp, src->payload_mem_block_id);
  uint8_t *base_hdr_ptr = src->headerSize > 0 ?
                          (uint8_t *)rmx_input_get_completion_ptr(comp, src->header_mem_block_id) : NULL;

  if (!chunk_size) {
    // no packet to handle
    return TRUE;
  }

  bufList = gst_buffer_list_new_sized (chunk_size);

  for (i = 0; i < chunk_size; i++) {
    uint8_t *data_stride = base_data_ptr + i * src->data_stride_size;
    uint8_t *hdr_stride;
    uint8_t *app_hdr;
    uint8_t *app_data;
    GstBuffer *buf = NULL;

    const rmx_input_packet_info *packet_info = rmx_input_get_packet_info(&src->chunk_handle[0], i);

    if (base_hdr_ptr) {
      hdr_stride = base_hdr_ptr + i * src->hdr_stride_size;
      app_hdr = hdr_stride;
      app_data = data_stride;
    } else {
      hdr_stride = NULL;
      app_hdr = data_stride;
      app_data = get_rtp_hdr_ptr (data_stride, RMX_INPUT_APP_PROTOCOL_PACKET);
    }

    GstFlowReturn flowRet = gst_buffer_pool_acquire_buffer (src->pool, &buf, NULL);
    if (flowRet == GST_FLOW_FLUSHING || flowRet == GST_FLOW_EOS) {
      /* This is internal pool with unlimited buffers.
       * It will go in FLUSHING / EOS only in inactive state.
       * Pool is inactivated during stop. So come out of the loop.
       */
      if (bufList)
        gst_buffer_list_unref (bufList);

      return FALSE;
    }

    if (flowRet != GST_FLOW_OK) {
      //TODO: should component be stopped?
      GST_ERROR_OBJECT (src, "Failed to get buffer from pool:%d\n", flowRet);
      continue;
    } else {
      GstNvDsCudaMemory *mem = NULL;
      guint16 hdr_size = src->headerSize > 0 ?
                         rmx_input_get_packet_size(packet_info, src->header_mem_block_id) : 0;
      guint16 data_size = rmx_input_get_packet_size(packet_info, src->payload_mem_block_id);
      if (hdr_stride) {
        // header data split case
        mem = (GstNvDsCudaMemory *) gst_buffer_peek_memory (buf, 0);
        mem->data = (gpointer) app_hdr;
        ((GstMemory *) mem)->size = hdr_size;

        mem = (GstNvDsCudaMemory *) gst_buffer_peek_memory (buf, 1);
        mem->data = (gpointer) app_data;
        ((GstMemory *) mem)->size = data_size;
      } else {
        mem = (GstNvDsCudaMemory *) gst_buffer_peek_memory (buf, 0);
        mem->data = (gpointer) app_data;
        ((GstMemory *) mem)->size = data_size;
      }
      gst_buffer_list_add (bufList, buf);
    }
  }

  g_mutex_lock (&src->qLock);
  g_queue_push_tail (src->bufferQ, bufList);
  g_cond_signal (&src->qCond);
  g_mutex_unlock (&src->qLock);

  return TRUE;
}

static gpointer
nvdsudpsr_data_fetch_loop (gpointer data)
{
  rmx_status status[MAX_ST2022_7_STREAMS];
  const rmx_input_completion *comp[MAX_ST2022_7_STREAMS];
  GstNvDsUdpSrc *src = (GstNvDsUdpSrc *) data;
  guint rmax_not_ok_count = 0, rmax_no_chunk_size_count = 0;
  gboolean ret = 0;


  while (src->isRunning) {
    // reset the counters
    rmax_not_ok_count = rmax_no_chunk_size_count = 0;

    if (!nvdsudpsrc_timed_wait (src)) {
      src->lastError = -1;
      g_cond_signal (&src->qCond);
      return NULL;
    }

    if (!src->isRunning) {
      g_cond_signal (&src->qCond);
      return NULL;
    }

    for (guint i=0; i<src->num_streams; i++) {
      status[i] = rmx_input_get_next_chunk(&src->chunk_handle[i]);
      if ((status[i] != RMX_OK) && (status[i] != RMX_SIGNAL)) {
        // TODO: should it be critical error?
        GST_DEBUG_OBJECT (src, "Failed to get next chunk for streamId[%d], status:%d\n", i, status[i]);
        rmax_not_ok_count++;
      } else if (status[i] == RMX_SIGNAL) {
        /* If any of the streams captured Ctrl-C from Rivermax, execute that on priority */
        g_mutex_lock (&src->flowLock);
        src->isRunning = FALSE;
        g_mutex_unlock (&src->flowLock);
        g_cond_signal (&src->qCond);
        return NULL;
      }

      comp[i] = rmx_input_get_chunk_completion(&src->chunk_handle[i]);

      if ((status[i] == RMX_OK) && (rmx_input_get_completion_chunk_size(comp[i]) == 0)) {
        rmax_no_chunk_size_count++;
      }

    }

    if ((rmax_not_ok_count == src->num_streams) || (rmax_no_chunk_size_count == src->num_streams)) {
      continue;
    }

    /* since all streams point to the same memory (st2022-7 hw), passing any one of the
       rmax_in_completion object is enough */
    if (!src->isRtpOut) {
      ret = nvdsudpsrc_push_frame ((rmx_input_get_completion_chunk_size(comp[0]) > 0) ? comp[0] : comp[1], src);
    } else {
      ret = nvdsudpsrc_push_rtp_packets (comp[0], src);
    }

    if (!ret) {
      // need to exit the loop
      return NULL;
    }
  }
  return NULL;
}

static gboolean
gst_nvdsudpsrc_close (GstNvDsUdpSrc * src)
{
  GError *err = NULL;

  if (src->used_socket) {
    if (src->auto_multicast
        &&
        g_inet_address_get_is_multicast (g_inet_socket_address_get_address
            (src->addr))) {
      GError *err = NULL;

      if (src->multi_iface) {
        GStrv multi_ifaces = g_strsplit (src->multi_iface, ",", -1);
        gchar **ifaces = multi_ifaces;
        while (*ifaces) {
          g_strstrip (*ifaces);
          GST_DEBUG_OBJECT (src, "leaving multicast group %s interface %s",
              src->address, *ifaces);
          if (!g_socket_leave_multicast_group (src->used_socket,
                  g_inet_socket_address_get_address (src->addr),
                  FALSE, *ifaces, &err)) {
            GST_ERROR_OBJECT (src, "Failed to leave multicast group: %s",
                err->message);
            g_clear_error (&err);
          }
          ifaces++;
        }
        g_strfreev (multi_ifaces);

      } else {
        GST_DEBUG_OBJECT (src, "leaving multicast group %s", src->address);
        if (!g_socket_leave_multicast_group (src->used_socket,
                g_inet_socket_address_get_address (src->addr), FALSE,
                NULL, &err)) {
          GST_ERROR_OBJECT (src, "Failed to leave multicast group: %s",
              err->message);
          g_clear_error (&err);
        }
      }
    }

    if (!g_socket_close (src->used_socket, &err)) {
      GST_ERROR_OBJECT(src, "Failed to close socket: %s", err->message);
      g_clear_error(&err);
    }

    g_object_unref (src->used_socket);
    src->used_socket = NULL;
    g_object_unref (src->addr);
    src->addr = NULL;
  }

  return TRUE;
}

/**
 * @brief Opens and initializes a UDP socket for the source element
 *
 * This function creates and configures a UDP socket for receiving data. It handles both
 * unicast and multicast scenarios, binding to the appropriate address and port.
 * For multicast addresses, it binds to ANY and prepares for joining the multicast group later.
 *
 * @param src Pointer to the GstNvDsUdpSrc instance
 * @return TRUE if socket was opened successfully, FALSE otherwise
 *
 */
static gboolean
gst_nvdsudpsrc_open (GstNvDsUdpSrc * src)
{
  GInetAddress *addr, *bind_addr;
  GSocketAddress *bind_saddr;
  GError *err = NULL;

  /* Handle multiple local interface IPs if specified */
  if (src->localIfaceIps) {
    /* Parse list of local interface IPs string into individual strings */
    src->num_local_interfaces = parse_ip_addresses(src->localIfaceIps, src->localIfaceIp, "local interface");
    if (src->num_local_interfaces == 0) {
      GST_ERROR_OBJECT (src, "Invalid number of local interface IPs: %u. Min: 1, Max: %u",
                        src->num_local_interfaces, MAX_ST2022_7_STREAMS);
      return FALSE;
    }
    GST_DEBUG_OBJECT (src, "Parsed %u local interface IPs", src->num_local_interfaces);
  }

  if (!src->localIfaceIp[0]) {
    GST_ERROR_OBJECT (src, "NULL ip of local interface");
    return FALSE;
  }

  GST_DEBUG_OBJECT(src, "allocating socket for %s:%d", src->address,
                   src->port);

  addr = gst_udp_resolve_name (src, src->address);
  if (!addr)
    goto name_resolve;

  if ((src->used_socket =
           g_socket_new(g_inet_address_get_family(addr),
                        G_SOCKET_TYPE_DATAGRAM, G_SOCKET_PROTOCOL_UDP, &err)) == NULL)
    goto no_socket;

  GST_DEBUG_OBJECT(src, "got socket %p", src->used_socket);

  if (src->addr)
    g_object_unref(src->addr);
  src->addr =
      G_INET_SOCKET_ADDRESS(g_inet_socket_address_new(addr, src->port));

  GST_DEBUG_OBJECT(src, "binding on port %d", src->port);

  /* For multicast, bind to ANY and join the multicast group later */
  if (g_inet_address_get_is_multicast (addr))
    bind_addr = g_inet_address_new_any (g_inet_address_get_family (addr));
  else
    bind_addr = G_INET_ADDRESS (g_object_ref (addr));

  g_object_unref(addr);

  bind_saddr = g_inet_socket_address_new (bind_addr, src->port);
  g_object_unref(bind_addr);
  if (!g_socket_bind (src->used_socket, bind_saddr, src->reuse, &err)) {
    GST_ERROR_OBJECT(src, "%s: error binding to %s:%d", err->message,
                    src->address, src->port);
    goto bind_error;
  }

  g_object_unref (bind_saddr);
  g_socket_set_multicast_loopback (src->used_socket, src->loop);

  gint val = 0;

  if (src->buffer_size != 0) {
    GError *opt_err = NULL;

    GST_INFO_OBJECT(src, "setting udp buffer of %d bytes", src->buffer_size);
    /* set buffer size, Note that on Linux this is typically limited to a
       * maximum of around 100K. Also a minimum of 128 bytes is required on
       * Linux. */
    if (!g_socket_set_option (src->used_socket, SOL_SOCKET, SO_RCVBUF,
                             src->buffer_size, &opt_err)) {
      GST_ELEMENT_WARNING (src, RESOURCE, SETTINGS, (NULL),
                          ("Could not create a buffer of requested %d bytes: %s",
                           src->buffer_size, opt_err->message));
      g_error_free(opt_err);
      opt_err = NULL;
    }
  }

  /* read the value of the receive buffer. Note that on linux this returns
     * 2x the value we set because the kernel allocates extra memory for
     * metadata. The default on Linux is about 100K (which is about 50K
     * without metadata) */
  if (g_socket_get_option (src->used_socket, SOL_SOCKET, SO_RCVBUF, &val,
                          NULL)) {
    GST_INFO_OBJECT (src, "have udp buffer of %d bytes", val);
  } else {
    GST_DEBUG_OBJECT (src, "could not get udp buffer size");
  }

  g_socket_set_broadcast (src->used_socket, TRUE);

  if (src->auto_multicast
      &&
      g_inet_address_get_is_multicast (g_inet_socket_address_get_address
          (src->addr))) {

    if (src->multi_iface) {
      GStrv multi_ifaces = g_strsplit (src->multi_iface, ",", -1);
      gchar **ifaces = multi_ifaces;
      while (*ifaces) {
        g_strstrip (*ifaces);
        GST_DEBUG_OBJECT (src, "joining multicast group %s interface %s",
            src->address, *ifaces);
        if (!g_socket_join_multicast_group (src->used_socket,
                g_inet_socket_address_get_address (src->addr),
                FALSE, *ifaces, &err)) {
          g_strfreev (multi_ifaces);
          goto membership;
        }

        ifaces++;
      }
      g_strfreev (multi_ifaces);
    } else {
      GST_DEBUG_OBJECT (src, "joining multicast group %s", src->address);
      if (!g_socket_join_multicast_group (src->used_socket,
              g_inet_socket_address_get_address (src->addr), FALSE, NULL, &err))
        goto membership;
    }
  }

  {
    GInetSocketAddress *addr;
    guint16 port;

    addr =
        G_INET_SOCKET_ADDRESS (g_socket_get_local_address (src->used_socket,
            &err));
    if (!addr)
      goto getsockname_error;

    port = g_inet_socket_address_get_port (addr);
    GST_DEBUG_OBJECT (src, "bound, on port %d", port);
    if (port != src->port) {
      src->port = port;
    }
    g_object_unref (addr);
  }

  return TRUE;

  /* ERRORS */
name_resolve:
  {
    return FALSE;
  }
no_socket:
  {
    GST_ELEMENT_ERROR (src, RESOURCE, OPEN_READ, (NULL),
        ("no socket error: %s", err->message));
    g_clear_error (&err);
    g_object_unref (addr);
    return FALSE;
  }
bind_error:
  {
    GST_ELEMENT_ERROR (src, RESOURCE, SETTINGS, (NULL),
        ("bind failed: %s", err->message));
    g_clear_error (&err);
    g_object_unref (bind_saddr);
    gst_nvdsudpsrc_close (src);
    return FALSE;
  }
membership:
  {
    GST_ELEMENT_ERROR (src, RESOURCE, SETTINGS, (NULL),
        ("could not add membership: %s", err->message));
    g_clear_error (&err);
    gst_nvdsudpsrc_close (src);
    return FALSE;
  }
getsockname_error:
  {
    GST_ELEMENT_ERROR (src, RESOURCE, SETTINGS, (NULL),
        ("getsockname failed: %s", err->message));
    g_clear_error (&err);
    gst_nvdsudpsrc_close (src);
    return FALSE;
  }
}

static gboolean
gst_nvdsudpsrc_unlock (GstBaseSrc * bsrc)
{
  GstNvDsUdpSrc *src;

  src = GST_NVDSUDPSRC (bsrc);
  g_mutex_lock (&src->flowLock);
  src->flushing = TRUE;
  g_mutex_unlock (&src->flowLock);

  g_cond_signal (&src->qCond);

  return TRUE;
}

static gboolean
gst_nvdsudpsrc_unlock_stop (GstBaseSrc * bsrc)
{
  GstNvDsUdpSrc *src;

  src = GST_NVDSUDPSRC (bsrc);
  g_mutex_lock (&src->flowLock);
  src->flushing = FALSE;
  g_mutex_unlock (&src->flowLock);

  return TRUE;
}

static GstClock*
gst_nvdsudpsrc_provide_clock (GstElement * element)
{
  GstNvDsUdpSrc *src = GST_NVDSUDPSRC (element);
  GstClock *clock = NULL;

  GST_OBJECT_LOCK (src);
  if (src->use_rtp_timestamp) {
    clock = gst_system_clock_obtain();
    if (clock) {
      g_object_set(G_OBJECT(clock), "clock-type", GST_CLOCK_TYPE_REALTIME, NULL);
      GST_DEBUG_OBJECT(src, "Providing REALTIME system clock");
      GST_OBJECT_UNLOCK (src);
      return clock;
    }
  }
  GST_OBJECT_UNLOCK (src);

  GST_DEBUG_OBJECT(src, "No clock provided when use_rtp_timestamp is disabled");
  return NULL;
}

/**
 * @brief Parse st2022_7_streams string and stores IP and port info in DstStreamInfo struct
 *
 * This function parses a comma-separated string of IP addresses and port numbers,
 * storing the information in a DstStreamInfo struct.
 *
 * @param streams_str String containing comma-separated IP:port pairs
 * @param dstStream Pointer to array of DstStreamInfo structs
 * @return Number of successfully parsed streams
 *
 * @note The caller is responsible for freeing the IP array with g_free()
 */
static guint
parse_st2022_7_streams(const gchar* streams_str, DstStreamInfo* dstStream)
{
  guint num_streams = 0;
  guint i = 0;

  if (streams_str == NULL) {
    GST_ERROR("Invalid ST2022-7 stream format. Expected ip:port");
    return 0;
  }

  gchar** stream_list = g_strsplit(streams_str, ",", -1);
  if (!stream_list) {
    GST_ERROR("Failed to split ST2022-7 streams string");
    return 0;
  }

  while (stream_list[i] != NULL && num_streams < MAX_ST2022_7_STREAMS) {
    gchar* stream = g_strstrip(stream_list[i]);
    gchar* ip_str_mem = NULL, *ip_str = NULL;
    gchar* port_str_mem = NULL, *port_str = NULL;

    /* Find the last colon to separate IP from port */
    gchar* last_colon = strrchr(stream, ':');
    if (last_colon) {
      ip_str_mem = g_strndup(stream, last_colon - stream);
      ip_str = g_strstrip(ip_str_mem);
      port_str_mem = g_strdup(last_colon + 1);
      port_str = g_strstrip(port_str_mem);
    } else {
      GST_ERROR("Invalid stream format. Expected ip:port, got %s", stream);
      i++;
      continue;
    }

    if (!ip_str || !port_str || strlen(ip_str) == 0 || strlen(port_str) == 0) {
      GST_ERROR("Invalid ST2022-7 stream format. Got IP: '%s', Port: '%s'",
                ip_str ? ip_str : "NULL", port_str ? port_str : "NULL");
      g_free(ip_str_mem);
      g_free(port_str_mem);
      i++;
      continue;
    }

    /* Parse port number */
    char *endptr = NULL;
    long port_val = strtol(port_str, &endptr, 10);
    if (*endptr != '\0' || port_val <= 0 || port_val > 65535) {
      GST_ERROR("Invalid port number: %s", port_str);
      g_free(ip_str_mem);
      g_free(port_str_mem);
      i++;
      continue;
    }

    /* Successfully parsed - assign to DstStreamInfo */
    dstStream[num_streams].ip = g_strdup(ip_str);
    dstStream[num_streams].port = (uint16_t)port_val;

    GST_DEBUG("ST2022-7 stream %d: %s:%d", num_streams,
              dstStream[num_streams].ip, dstStream[num_streams].port);

    g_free(ip_str_mem);
    g_free(port_str_mem);
    num_streams++;
    i++;
  }
  g_strfreev(stream_list);

  return num_streams;
}

/**
 * @brief Parse IP addresses string and stores IP addresses in char pointer array
 *
 * This function parses a comma-separated string of IP addresses,
 * storing the information in a char pointer array.
 *
 * @param addresses_str String containing comma-separated IP addresses
 * @param address_array Pointer to array of char pointers to store parsed IPs
 * @param context Context string for debug/error messages (e.g., "source", "local interface")
 * @return Number of successfully parsed IP addresses
 *
 * @note The caller is responsible for freeing the IP strings with g_free()
 */
static guint
parse_ip_addresses(const gchar* addresses_str, gchar** address_array, const gchar* context)
{
  guint num_addresses = 0;
  guint i = 0;

  if (addresses_str == NULL || strlen(addresses_str) == 0) {
    GST_DEBUG("Empty %s addresses string", context);
    return 0;
  }

  gchar** address_list = g_strsplit(addresses_str, ",", -1);
  if (!address_list) {
    GST_ERROR("Failed to split %s addresses string", context);
    return 0;
  }

  while (address_list[i] != NULL && num_addresses < MAX_ST2022_7_STREAMS) {
    gchar* ip_str = g_strstrip(address_list[i]);

    if (strlen(ip_str) == 0) {
      GST_ERROR("Empty IP address in %s addresses string", context);
      i++;
      continue;
    }

    /* Validate IP address format */
    struct in_addr addr;
    if (inet_aton(ip_str, &addr) == 0) {
      GST_ERROR("Invalid IP address format: %s", ip_str);
      i++;
      continue;
    }

    /* Successfully parsed - assign to char pointer array */
    address_array[num_addresses] = g_strdup(ip_str);

    GST_DEBUG("%s address %d: %s", context, num_addresses, address_array[num_addresses]);

    num_addresses++;
    i++;
  }
  g_strfreev(address_list);

  return num_addresses;
}
