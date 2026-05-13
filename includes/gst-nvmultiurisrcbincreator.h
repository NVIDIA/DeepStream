/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef _GST_NVMULTIURISRCBIN_CREATOR_H_
#define _GST_NVMULTIURISRCBIN_CREATOR_H_
#include <gst/gst.h>
#include "gst-nvdscommonconfig.h"
#include <unistd.h>


typedef struct NvDsServerDecInfo NvDsServerDecInfo;
typedef struct NvDsServerMuxInfo NvDsServerMuxInfo;
typedef struct NvDsServerConvInfo NvDsServerConvInfo;
typedef struct NvDsServerAppInstanceInfo NvDsServerAppInstanceInfo;

#ifdef __cplusplus
extern "C"
{
#endif


  typedef void *NvDst_Handle_NvMultiUriSrcCreator;

  typedef enum
  {
    NVDS_MULTIURISRCBIN_MODE_VIDEO,
    NVDS_MULTIURISRCBIN_MODE_AUDIO
  } NvDsMultiUriMode;

/**
 * API to check if sourceId related source exist
 *
 * @param[in] apiHandle  The nvmultiurisrcbincreator API handle returned by init
 * @param[in] sourceId sourceId information of the stream
 * @return  TRUE if source found, else FALSE
 */
  gboolean find_source (NvDst_Handle_NvMultiUriSrcCreator apiHandle,
      guint sourceId);

/**
 * API to set property value on nvv4l2decoder of nvmultiurisrcbin instance
 *
 * @param[in] apiHandle  The nvmultiurisrcbincreator API handle returned by init
 * @param[in] sourceId sourceId information of the stream
 * @param[in] dec_info decoder info structure with property value to be used to set on the nvv4l2decoder
 * @return  TRUE if successful, else FALSE
 */
  gboolean set_nvuribin_dec_prop (NvDst_Handle_NvMultiUriSrcCreator apiHandle,
      guint sourceId, NvDsServerDecInfo * dec_info);

/**
 * API to set property value on nvvideoconvert of nvmultiurisrcbin instance
 *
 * @param[in] apiHandle  The nvmultiurisrcbincreator API handle returned by init
 * @param[in] sourceId sourceId information of the stream
 * @param[in] conv_info converter info structure with property value to be used to set on the nvvideoconvert
 * @return  TRUE if successful, else FALSE
 */
  gboolean set_nvuribin_conv_prop (NvDst_Handle_NvMultiUriSrcCreator apiHandle,
      guint sourceId, NvDsServerConvInfo * conv_info);

/**
 * API to set property value on nvstreammux of nvmultiurisrcbin instance
 *
 * @param[in] apiHandle  The nvmultiurisrcbincreator API handle returned by init
 * @param[in] mux_info info structure with property value to be used to set on the nvstreammux
 * @return  TRUE if successful, else FALSE
 */
  gboolean set_nvuribin_mux_prop (NvDst_Handle_NvMultiUriSrcCreator apiHandle,
      NvDsServerMuxInfo * mux_info);

/**
 * API to set EOS related properties on nvstreammux of nvmultiurisrcbin instance
 *
 * @param[in] apiHandle  The nvmultiurisrcbincreator API handle returned by init
 * @param[in] appinstance_info info structure with activity to be used to set eos related properties on the nvstreammux
 * @return  TRUE if successful, else FALSE
 */
  gboolean s_force_eos_handle (NvDst_Handle_NvMultiUriSrcCreator apiHandle,
      NvDsServerAppInstanceInfo * appinstance_info);
/**
 * Initialize the API for nvmultiurisrcbin creation
 * Note: nvmultiurisrcbin is a collection of DeepStream plugins viz:
 *       nvurisrcbin X N -> nvstreammux
 *
 * @param[in] binName    A name to the nvmultiurisrcbin instance
 * @param[in] mode       The bin's mode can either be video-only or audio-only
 * @param[in] muxConfig  The nvstreammux config to use within nvmultiurisrcbin
 * @return  The handle to this API instance
 */
  NvDst_Handle_NvMultiUriSrcCreator gst_nvmultiurisrcbincreator_init (gchar *
      binName, NvDsMultiUriMode mode, GstDsNvStreammuxConfig * muxConfig);

/**
 * API to add a source/stream into the nvmultiurisrcbin instance
 * A new instance of nvurisrcbin will be created for this stream
 * and be linked with appropriate sink pad of the nvstreammux instance
 *
 * @param[in] apiHandle  The nvmultiurisrcbincreator API handle returned by init
 * @param[in] sourceConfig New stream/source configuration information
 * @return  TRUE if successful, else FALSE
 */
  gboolean
      gst_nvmultiurisrcbincreator_add_source (NvDst_Handle_NvMultiUriSrcCreator
      apiHandle, GstDsNvUriSrcConfig * sourceConfig);

/**
 * API to remove a source/stream from the nvmultiurisrcbin instance
 * The corresponding nvurisrcbin will be unlinked from nvstreammux and removed
 * from the parent nvmultiurisrcbin instance.
 * The book-keeping for this source will also be cleared.
 * Note: This API force the nvurisrcbin state to GST_STATE_NULL
 *
 * @param[in] apiHandle  The nvmultiurisrcbincreator API handle returned by init
 * @param[in] sourceId   stream/source ID matching the provided source
 *                       configuration information
 * @return  TRUE if successful, else FALSE
 */
  gboolean
      gst_nvmultiurisrcbincreator_remove_source
      (NvDst_Handle_NvMultiUriSrcCreator apiHandle, guint sourceId);

/**
 * API to remove a source/stream from the nvmultiurisrcbin instance
 * The corresponding nvurisrcbin will be unlinked from nvstreammux and removed
 * from the parent nvmultiurisrcbin instance.
 * The book-keeping for this source will also be cleared.
 * Note: This API does force the nvurisrcbin state to GST_STATE_NULL
 * Users may leverage this API when the state change happens externally
 * (example: when there is an ERROR)
 *
 * @param[in] apiHandle  The nvmultiurisrcbincreator API handle returned by init
 * @param[in] sourceId   stream/source ID matching the provided source
 *                       configuration information
 * @return  TRUE if successful, else FALSE
 */
  gboolean
      gst_nvmultiurisrcbincreator_remove_source_without_forced_state_change
      (NvDst_Handle_NvMultiUriSrcCreator apiHandle, guint sourceId);

/**
 * Get the nvmultiurisrcbin GstBin
 * User may:
 * a) add this Bin/Element into their pipeline and
 * b) link the bin's src pad to downstream element
 *    Note: This bin's src pad capability will be same as
 *    nvstreammux src pad capability
 *
 * @param[in] apiHandle  The nvmultiurisrcbincreator API handle returned by init
 * @return  [transfer none] A pointer to the nvmultiurisrcbin instance GstBin
 *          typecasted to the parent GstElement
 */
  GstElement
      *gst_nvmultiurisrcbincreator_get_bin (NvDst_Handle_NvMultiUriSrcCreator
      apiHandle);

/**
 * API to procure the source pad ("src") of this nvmultiurisrcbin instance
 * Note: User may link the bin's src pad to downstream element
 *    Note: This src pad is the nvstreammux instance's src pad.
 *    Thus, the src pad GstCaps capability will be same as
 *    nvstreammux src pad capability
 *
 * @param[in] apiHandle  The nvmultiurisrcbincreator API handle returned by init
 * @return  source pad for nvmultiurisrcbin
 */
  GstPad
      *gst_nvmultiurisrcbincreator_get_source_pad
      (NvDst_Handle_NvMultiUriSrcCreator apiHandle);

/**
 * @param[in] apiHandle  The nvmultiurisrcbincreator API handle returned by init
 * @return  TRUE if successful, else FALSE
 */
  gboolean
      gst_nvmultiurisrcbincreator_sync_children_states
      (NvDst_Handle_NvMultiUriSrcCreator apiHandle);

/**
 * Deinit the nvmultiurisrcbin creator API
 * Note: If this bin was added to a parent pipeline instance,
 * Use this API before calling gst_object_unref on the pipeline object.
 * @param[in] apiHandle  The nvmultiurisrcbincreator API handle returned by init
 */
  void gst_nvmultiurisrcbincreator_deinit (NvDst_Handle_NvMultiUriSrcCreator
      apiHandle);

/**
 * API to procure the source config for a particular sensor
 *
 * @param[in] apiHandle  The nvmultiurisrcbincreator API handle returned by init
 * @param[in] uri        The uri string (need to exactly match details provided
 *                       with gst_nvmultiurisrcbincreator_add_source()
 * @param[in] sensorId   The sensorId string (need to exactly match details provided
 *                       with gst_nvmultiurisrcbincreator_add_source()
 * @return  source config for the running sensor where uri and sensorId match
 *          [transfer-full] the pointer returned shall be freed using API
 *          gst_nvmultiurisrcbincreator_src_config_free(1, &returnedPointer)
 */
  GstDsNvUriSrcConfig
      *gst_nvmultiurisrcbincreator_get_source_config
      (NvDst_Handle_NvMultiUriSrcCreator apiHandle, gchar const *uri,
      gchar const *sensorId);


/**
 * API to create duplicate of GstDsNvUriSrcConfig
 */
  GstDsNvUriSrcConfig
      *gst_nvmultiurisrcbincreator_src_config_dup (GstDsNvUriSrcConfig *
      sourceConfig);

/**
 * API to free GstDsNvUriSrcConfig object obtained with API:
 * gst_nvmultiurisrcbincreator_src_config_dup
 */
  void gst_nvmultiurisrcbincreator_src_config_free (GstDsNvUriSrcConfig *
      config);

/**
 * API to procure the source config for a particular sensor
 *
 * @param[in] apiHandle  The nvmultiurisrcbincreator API handle returned by init
 * @param[in] sensorId   The sensorId string (need to exactly match details provided
 *                       with gst_nvmultiurisrcbincreator_add_source()
 * @return  source config for the running sensor where uri and sensorId match
 */
  GstDsNvUriSrcConfig
      *gst_nvmultiurisrcbincreator_get_source_config_by_sensorid
      (NvDst_Handle_NvMultiUriSrcCreator apiHandle, gchar const *sensorId);

/**
 * API to procure the list of sources
 *
 * @param[in] apiHandle   The nvmultiurisrcbincreator API handle returned by init
 * @param[in/out] count   The number of streams added to this bin using
 *                        API gst_nvmultiurisrcbincreator_add_source()
 * @param[in/out] configs An array of configs of size count
 *                       [transfer-full] Please free this configs array using
 *                        API gst_nvmultiurisrcbincreator_src_config_list_free()
 *                        API gst_nvmultiurisrcbincreator_add_source()
 * @return  TRUE if API was successful; FALSE otherwise
 */
  gboolean
      gst_nvmultiurisrcbincreator_get_active_sources_list
      (NvDst_Handle_NvMultiUriSrcCreator apiHandle, guint * count,
      GstDsNvUriSrcConfig *** configs);

/**
 * API to procure the list of active sources
 *
 * @param[in] apiHandle   The nvmultiurisrcbincreator API handle returned by init
 * @param[in/out] sensor_info_list A list which would contain the sensor realted info
 * @return  TRUE if API was successful; FALSE otherwise
 */

  gboolean
      gst_nvmultiurisrcbincreator_get_source_info_list
      (NvDst_Handle_NvMultiUriSrcCreator apiHandle, GList ** sensor_info_list);

/**
 * API to destroy the memory allocation for list of sources procured with API:
 * gst_nvmultiurisrcbincreator_get_active_sources_list()
 *
 * @param[in] apiHandle   The nvmultiurisrcbincreator API handle returned by init
 * @param[in] count       The size of configs array
 * @param[in] configs     array of configs returned by
 *                        API gst_nvmultiurisrcbincreator_get_active_sources_list()
 *                        API gst_nvmultiurisrcbincreator_add_source()
 */
  void gst_nvmultiurisrcbincreator_src_config_list_free
      (NvDst_Handle_NvMultiUriSrcCreator apiHandle, guint count,
      GstDsNvUriSrcConfig ** configs);

/**
 * API to get the sensor id from the source id
 *
 * @param[in] apiHandle  The nvmultiurisrcbincreator API handle returned by init
 * @param[in] sourceId   The source id
 * @return  The sensor id
 */
  gchar *gst_nvmultiurisrcbincreator_get_sensor_id_from_source_id(NvDst_Handle_NvMultiUriSrcCreator apiHandle, guint sourceId);

/**
 * API to get the GPU ID configured for the nvmultiurisrcbin creator
 *
 * This function retrieves the GPU ID that is being used by the nvstreammux
 * component within the nvmultiurisrcbin. The GPU ID determines which GPU
 * device is used for processing streams.
 *
 * @param[in] apiHandle  The nvmultiurisrcbincreator API handle returned by init
 *
 * @return  The GPU ID configured in the mux configuration
 *          Returns 0 if muxConfig is NULL or not set
 */
  guint get_gpu_id(NvDst_Handle_NvMultiUriSrcCreator apiHandle);
#ifdef __cplusplus
}
#endif

#endif
