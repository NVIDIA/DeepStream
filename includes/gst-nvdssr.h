/*
 * SPDX-FileCopyrightText: Copyright (c) 2019 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

/**
 * @file
 * <b>NVIDIA DeepStream: Smart recording API</b>
 */
/**
 * @defgroup  custom_gstreamer  Custom Gstreamer APIs
 *
 * This section defines custom Gstreamer APIs
 *
 */

#ifndef NVDSSR_H_
#define NVDSSR_H_

#include <gst/gst.h>

#define _DS_DEPRECATED_(STR) __attribute__ ((deprecated (STR)))

/**
 *
 * @defgroup  gstreamer_nvdssr  Smart Record
 *
 * Specifies APIs relating to smart recording.
 *
 * @ingroup custom_gstreamer
 * @{
 */
#ifdef __cplusplus
extern "C"
{
#endif

typedef struct NvDsSRRecordingInfo NvDsSRRecordingInfo;

typedef gpointer (*NvDsSRCallbackFunc) (NvDsSRRecordingInfo *info, gpointer userData);

typedef guint32 NvDsSRSessionId;

/**
 * Specifies container types.
 */
typedef enum {
  NVDSSR_CONTAINER_MP4,
  NVDSSR_CONTAINER_MKV
} NvDsSRContainerType;

/**
 * Specifies API return status.
 */
typedef enum {
  NVDSSR_STATUS_OK,
  NVDSSR_STATUS_INVALID_VAL,
  NVDSSR_STATUS_INVALID_OP,
  NVDSSR_STATUS_ERROR,

  NVDSSR_STATUS_CUSTOM1 = 100,
  NVDSSR_STATUS_CUSTOM2 = 101,
  NVDSSR_STATUS_CUSTOM3 = 102
} NvDsSRStatus;

/**
 * Holds initializtion paramerters required to create \ref NvDsSRContext.
 */
typedef struct NvDsSRInitParams
{
  /** callback function gets called once recording is complete */
  NvDsSRCallbackFunc callback;
  /** recording video container, MP4 / MKV */
  NvDsSRContainerType containerType;
  /** optional, recording video width, 0 means no transcode */
  guint width;
  /** optional, recording video height, 0 means no transcode */
  guint height;
  /** recorded file name prefix */
  gchar *fileNamePrefix;
  /** store recorded file under directory path */
  gchar *dirpath;
   /** default recording duration in seconds */
  guint defaultDuration;
  union {
    /** size of video cache in seconds. */
    guint videoCacheSize _DS_DEPRECATED_("videoCacheSize is deprecated. Use cacheSize instead");
    /** size of cache in seconds, applies to video and audio. */
    guint cacheSize;
  };
} NvDsSRInitParams;

/**
 * Holds information about smart record instance.
 */
typedef struct NvDsSRContext
{
  /** parent bin element. */
  GstElement *recordbin;
  /** queue element to cache the content. */
  GstElement *recordQue;
  /** child bin to save the content to file. */
  GstElement *encodebin;
  /** filesink element */
  GstElement *filesink;
  /** flag to check the key frame. */
  gboolean gotKeyFrame;
  /** flag to check if recording is on */
  gboolean recordOn;
  /** flag to check if encodebin is reset */
  gboolean resetDone;
  /** flag to check if encodebin is in playing state. */
  gboolean isPlaying;
  /** initialization parameters */
  NvDsSRInitParams initParams;
  /** mutex to control the flow */
  GMutex flowLock;
  /** thread to reset the encodebin */
  GThread *resetThread;
  /** pointer to user provided data */
  gpointer uData;
  /** pointer to private data */
  gpointer privData;
} NvDsSRContext;

/**
 * Hold information about video recorded.
 */
typedef struct NvDsSRRecordingInfo
{
  /** SR bin context */
  NvDsSRContext *ctx;
  /** recording session-id */
  NvDsSRSessionId sessionId;
  /** recorded file name */
  gchar *filename;
  /** recorded file dir path */
  gchar *dirpath;
  /** duration in milliseconds */
  guint64 duration;
  /** recorded video container, MP4 / MKV */
  NvDsSRContainerType containerType;
  /** recorded video width*/
  guint width;
  /** recorded video height*/
  guint height;
  /** Boolean indicating if recorded stream contains video. */
  gboolean containsVideo;
  /** channels in recorded audio */
  guint channels;
  /** sampling rate of recorded audio  in Hz */
  guint samplingRate;
  /** Boolean indicating if recorded stream contains audio. */
  gboolean containsAudio;
} NvDsSRRecordingInfo;

/**
 * \brief  Creates the instance of smart record.
 *
 * This function creates the instance of smart record and returns the pointer
 * to an allocated \ref NvDsSRContext. The \a params structure must be filled
 * with initialization parameters required to create the instance.
 *
 * recordbin of \ref NvDsSRContext is smart record bin which must be added
 * to the pipeline. It expects encoded frames which will be muxed and saved to
 * the file. Add this bin after parser element in the pipeline.
 *
 * Call NvDsSRDestroy() to free resources allocated by this function.
 *
 * @param[out] ctx         An indirect pointer to the smart record instance.
 * @param[in]  params      A pointer to a \ref NvDsSRInitParams structure.
 *
 * @return NVDSSR_STATUS_OK if successful, or corresponding error otherwise.
 */
NvDsSRStatus NvDsSRCreate (NvDsSRContext **ctx, NvDsSRInitParams *params);

/**
 * \brief  Starts the video recording.
 *
 * This function starts writing the cached video data to a file. It returns
 * the session id which later can be used in NvDsSRStop() to stop the
 * corresponding recording.
 *
 * Here startTime specifies the seconds before the current time and duration
 * specifies the seconds after the start of recording.
 * If current time is t1, content from t1 - startTime to t1 + duration will
 * be saved to file. Therefore a total of startTime + duration seconds of data
 * will be recorded.
 *
 * @param[in] ctx         A pointer to a \ref NvDsSRContext.
 * @param[out] sessionId  A pointer to a \ref NvDsSRSessionId.
 * @param[in] startTime   Seconds before the current time. Should be less than video cache size.
 * @param[in] duration    Duration value in seconds after the start of recording.
 * @param[in] userData    A pointer to user specified data.
 *
 * @return NVDSSR_STATUS_OK if successful, or corresponding error otherwise.
 */
NvDsSRStatus NvDsSRStart (NvDsSRContext *ctx, NvDsSRSessionId *sessionId,
                          guint startTime, guint duration, gpointer userData);

/**
 * \brief  Stops the previously started recording.
 *
 * @param[in] ctx         A pointer to a \ref NvDsSRContext.
 * @param[in] sessionId   Id of seesion to stop.
 *
 * @return NVDSSR_STATUS_OK if successful, or corresponding error otherwise.
 */
NvDsSRStatus NvDsSRStop (NvDsSRContext *ctx, NvDsSRSessionId sessionId);

/**
 * \brief  Destroys the instance of smart record.
 *
 * This function releases the resources previously allocated by NvDsSRCreate().
 *
 * @param[in] ctx         A pointer to a \ref NvDsSRContext to be freed.
 *
 * @return NVDSSR_STATUS_OK if successful, or corresponding error otherwise.
 */
NvDsSRStatus NvDsSRDestroy (NvDsSRContext *ctx);

#ifdef __cplusplus
}
#endif
#endif /* NVDSSR_H_ */

/** @} */
