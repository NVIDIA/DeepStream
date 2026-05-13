/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "gstnvdsudpcommon.h"
#include <string.h>

gboolean
gst_udp_parse_uri (const gchar * uristr, gchar ** host, guint16 * port)
{
  gchar *protocol, *location_start;
  gchar *location, *location_end;
  gchar *colptr;

  /* consider no protocol to be udp:// */
  protocol = gst_uri_get_protocol (uristr);
  if (!protocol)
    goto no_protocol;
  if (strcmp (protocol, "udp") != 0)
    goto wrong_protocol;
  g_free (protocol);

  location_start = gst_uri_get_location (uristr);
  if (!location_start)
    return FALSE;

  GST_DEBUG ("got location '%s'", location_start);

  /* VLC compatibility, strip everything before the @ sign. VLC uses that as the
   * remote address. */
  location = g_strstr_len (location_start, -1, "@");
  if (location == NULL)
    location = location_start;
  else
    location += 1;

  if (location[0] == '[') {
    GST_DEBUG ("parse IPV6 address '%s'", location);
    location_end = strchr (location, ']');
    if (location_end == NULL)
      goto wrong_address;

    *host = g_strndup (location + 1, location_end - location - 1);
    colptr = strrchr (location_end, ':');
  } else {
    GST_DEBUG ("parse IPV4 address '%s'", location);
    colptr = strrchr (location, ':');

    if (colptr != NULL) {
      *host = g_strndup(location, colptr - location);
    } else {
      *host = g_strdup(location);
    }
  }
  GST_DEBUG ("host set to '%s'", *host);

  if (colptr != NULL) {
    *port = g_ascii_strtoll (colptr + 1, NULL, 10);
  } else {
    *port = 0;
  }
  g_free (location_start);

  return TRUE;

  /* ERRORS */
no_protocol:
  {
    GST_ERROR ("error parsing uri %s: no protocol", uristr);
    return FALSE;
  }
wrong_protocol:
  {
    GST_ERROR ("error parsing uri %s: wrong protocol (%s != udp)", uristr,
        protocol);
    g_free (protocol);
    return FALSE;
  }
wrong_address:
  {
    GST_ERROR ("error parsing uri %s", uristr);
    g_free (location);
    return FALSE;
  }
}

GInetAddress *
gst_udp_resolve_name (gpointer obj, const gchar * address)
{
  GInetAddress *addr;
  GError *err = NULL;
  GResolver *resolver;

  addr = g_inet_address_new_from_string (address);
  if (!addr) {
    GList *results;

    GST_DEBUG_OBJECT (obj, "resolving IP address for host %s", address);
    resolver = g_resolver_get_default ();
    results =
        g_resolver_lookup_by_name (resolver, address, NULL, &err);
    if (!results)
      goto name_resolve;
    addr = G_INET_ADDRESS (g_object_ref (results->data));

    g_resolver_free_addresses (results);
    g_object_unref (resolver);
  }

  return addr;

name_resolve:
  {
    GST_WARNING_OBJECT (obj, "Failed to resolve %s: %s", address, err->message);
    g_clear_error (&err);
    g_object_unref (resolver);
    return NULL;
  }
}

/****************************************************
 * RTP TIMESTAMP METADATA RELATED FUNCTIONS
 * *************************************************/

static gboolean
gst_rtp_timestamp_meta_init(GstMeta *meta, gpointer params, GstBuffer *buffer)
{
    GstRTPTimestampMeta *rtp_meta = (GstRTPTimestampMeta *)meta;
    rtp_meta->rtp_timestamp = 0;
    rtp_meta->leap_seconds_adjusted = FALSE;
    return TRUE;
}

static gboolean
gst_rtp_timestamp_meta_transform(GstBuffer *transbuf, GstMeta *meta,
    GstBuffer *buffer, GQuark type, gpointer data)
{
    GstRTPTimestampMeta *rtp_meta = (GstRTPTimestampMeta *)meta;
    GstRTPTimestampMeta *new_meta = gst_buffer_add_rtp_timestamp_meta(transbuf,
        rtp_meta->rtp_timestamp);
    if (new_meta) {
        new_meta->leap_seconds_adjusted = rtp_meta->leap_seconds_adjusted;
    }
    return (new_meta != NULL);
}

GType
gst_rtp_timestamp_meta_api_get_type(void)
{
    static GType type = 0;
    static const gchar *tags[] = { NULL };

    if (g_once_init_enter(&type)) {
        GType _type = gst_meta_api_type_register("GstRTPTimestampMetaAPI", tags);
        g_once_init_leave(&type, _type);
    }
    return type;
}

const GstMetaInfo *
gst_rtp_timestamp_meta_get_info(void)
{
    static const GstMetaInfo *meta_info = NULL;

    if (g_once_init_enter(&meta_info)) {
        const GstMetaInfo *mi = gst_meta_register(
            gst_rtp_timestamp_meta_api_get_type(),
            "GstRTPTimestampMeta",
            sizeof(GstRTPTimestampMeta),
            gst_rtp_timestamp_meta_init,
            NULL,
            gst_rtp_timestamp_meta_transform);
        g_once_init_leave(&meta_info, mi);
    }
    return meta_info;
}

GstRTPTimestampMeta *
gst_buffer_add_rtp_timestamp_meta(GstBuffer *buffer, guint32 timestamp)
{
    GstRTPTimestampMeta *meta;

    g_return_val_if_fail(GST_IS_BUFFER(buffer), NULL);

    meta = (GstRTPTimestampMeta *)gst_buffer_add_meta(buffer,
        gst_rtp_timestamp_meta_get_info(), NULL);

    if (meta) {
        meta->rtp_timestamp = timestamp;
        meta->leap_seconds_adjusted = FALSE;
    }

    return meta;
}

GstRTPTimestampMeta *
gst_buffer_get_rtp_timestamp_meta(GstBuffer *buffer)
{
    g_return_val_if_fail(GST_IS_BUFFER(buffer), NULL);

    return (GstRTPTimestampMeta *)gst_buffer_get_meta(buffer,
        gst_rtp_timestamp_meta_api_get_type());
}

/************************************************** */

/***************************************************
 * GPU MEMORY ALLOCATION RELATED FUNCTIONS
 * *************************************************/


static void* cudaAllocateMmap (int gpuId, size_t size, size_t align)
{
  CUresult status = CUDA_SUCCESS;
  CUdeviceptr dptr = 0;
  int val = 0;

  // Setup the properties common for all the chunks
  // The allocations will be device pinned memory.
  // This property structure describes the physical location where the memory
  // will be allocated via cuMemCreate along with additional properties In this
  // case, the allocation will be pinned device memory local to a given device.
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = gpuId;

  status = cuDeviceGetAttribute (&val, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, gpuId);
  if (status != CUDA_SUCCESS || val == 0) {
    GST_ERROR ("Device does not support VA. err = %d", status);
    goto done;
  }

  status = cuDeviceGetAttribute (&val, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, gpuId);
  if (status != CUDA_SUCCESS || val == 0) {
    GST_ERROR ("RDMA is not supported or not enabled, err = %d val = %d", status, val);
    goto done;
  } else {
    prop.allocFlags.gpuDirectRDMACapable = 1;
  }

  // Reserve the required contiguous VA space for the allocations
  status = cuMemAddressReserve (&dptr, size, align, 0, 0);
  if (status != CUDA_SUCCESS) {
    GST_ERROR ("cuMemAddressReserve failed, err = %d", status);
    goto done;
  }

  // Create the allocation as a pinned allocation on this device
  CUmemGenericAllocationHandle allocationHandle;
  status = cuMemCreate (&allocationHandle, size, &prop, 0);
  if (status != CUDA_SUCCESS) {
    GST_ERROR ("cuMemCreate failed, err = %d", status);
    goto done;
  }

  // Assign the chunk to the appropriate VA range and release the handle.
  // After mapping the memory, it can be referenced by virtual address.
  // Since we do not need to make any other mappings of this memory or export
  // it, we no longer need and can release the allocationHandle. The
  // allocation will be kept live until it is unmapped.
  status = cuMemMap (dptr, size, 0, allocationHandle, 0);

  // the handle needs to be released even if the mapping failed.
  status = cuMemRelease (allocationHandle);
  if (status != CUDA_SUCCESS) {
    GST_ERROR ("cuMemRelease failed, err = %d", status);
    goto done;
  }
  // Each accessDescriptor will describe the mapping requirement for a single
  // device
  CUmemAccessDesc accessDescriptors;

  // Prepare the access descriptor array indicating where and how the backings
  // should be visible.
  // Specify which device we are adding mappings for.
  accessDescriptors.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDescriptors.location.id = gpuId;

  // Specify both read and write access.
  accessDescriptors.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  // Apply the access descriptors to the whole VA range.
  status = cuMemSetAccess (dptr, size, &accessDescriptors, 1);
  if (status != CUDA_SUCCESS) {
    GST_ERROR ("cuMemSetAccess failed, err = %d", status);
    goto done;
  }

done:
  if (status != CUDA_SUCCESS) {
    cudaFreeMmap ((uint64_t*) &dptr, size);
    return NULL;
  }
  return (void*) dptr;
}

gboolean cudaFreeMmap (uint64_t *ptr, size_t size)
{
  if (!ptr) {
    return TRUE;
  }

  CUdeviceptr dptr = *(CUdeviceptr*) ptr;
  CUresult status = CUDA_SUCCESS;

  // Unmap the mapped virtual memory region
  // Since the handles to the mapped backing stores have already been released
  // by cuMemRelease, and these are the only/last mappings referencing them,
  // The backing stores will be freed.
  // Since the memory has been unmapped after this call, accessing the specified
  // va range will result in a fault (until it is re-mapped).
  status = cuMemUnmap (dptr, size);
  if (status != CUDA_SUCCESS) {
    GST_ERROR ("CUDA cuMemUnmap failed, err = %d", status);
    return FALSE;
  }
  // Free the virtual address region. This allows the virtual address region
  // to be reused by future cuMemAddressReserve calls. This also allows the
  // virtual address region to be used by other allocation made through
  // Operating system calls like malloc & mmap.
  status = cuMemAddressFree (dptr, size);
  if (status != CUDA_SUCCESS) {
    GST_ERROR ("CUDA cuMemAddressFree failed, err = %d", status);
    return FALSE;
  }
  return TRUE;
}

void* gpu_allocate_memory (int gpuId, size_t size, size_t align)
{
  int count = -1;
  cudaError_t cuda_err = cudaGetDeviceCount (&count);
  if (cuda_err != cudaSuccess || count <= gpuId) {
    GST_ERROR ("Failed to allocate GPU memory; GPU %d not available.", gpuId);
    return NULL;
  }

  struct cudaDeviceProp props;
  cuda_err = cudaGetDeviceProperties (&props, gpuId);
  if (cuda_err != cudaSuccess || !props.canMapHostMemory) {
    GST_ERROR ("Failed to allocate GPU memory; host mapping not supported.");
    return NULL;
  }

  cuda_err = cudaSetDevice (gpuId);
  if (cuda_err != cudaSuccess) {
    GST_ERROR ("Failed to allocate GPU memory; failed to set device.");
    return NULL;
  }

  char* buffer = NULL;
  if (props.integrated) {
    cuda_err = cudaSetDeviceFlags (cudaDeviceMapHost);
    if (cuda_err != cudaSuccess) {
      GST_ERROR ("Failed to allocate GPU memory; failed to set device flags.");
      return NULL;
    }
    cuda_err = cudaMallocHost ((void**)&buffer, size);
  } else {
    buffer = (char*) cudaAllocateMmap (gpuId, size, align);
  }

  if (cuda_err != cudaSuccess || buffer == NULL) {
    GST_ERROR ("Failed to allocate GPU memory.");
    return NULL;
  }

  if (!props.integrated) {
    unsigned int flag = 1;
    cuPointerSetAttribute (&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, (CUdeviceptr) buffer);
  }

  cuda_err = cudaDeviceSynchronize();
  if (cuda_err != cudaSuccess) {
    GST_ERROR ("failed to synchronize. error: %d", cuda_err);
    if (props.integrated) {
      cudaFreeHost (buffer);
    } else {
      cudaFreeMmap ((uint64_t *)&buffer, size);
    }
    return NULL;
  }
  return buffer;
}

size_t gpu_aligned_size (int gpuId, size_t allocSize)
{
  CUresult status = CUDA_SUCCESS;
  size_t size = allocSize;
  size_t granularity = 0;
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = gpuId;

  status = cuMemGetAllocationGranularity (&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  if (status != CUDA_SUCCESS) {
    GST_WARNING ("cuMemGetAllocationGranularity failed, err = %d\n", status);
    return size;
  }
  size = round_up (allocSize, granularity);
  return size;
}
/*************************************************** */