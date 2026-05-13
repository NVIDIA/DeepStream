/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA GStreamer DeepStream: Custom Message Functions</b>
 *
 * @b Description: This file specifies the NVIDIA DeepStream GStreamer custom
 * message functions.
 *
 */
 /**
 * @defgroup gst_mess_evnt_qry Events, Messages and Query based APIs
 *
 * Defines Events, Messages and Query-based APIs
 *
 */

#ifndef __GST_NVDSCUSTOMMESSAGE_H__
#define __GST_NVDSCUSTOMMESSAGE_H__

#include <gst/gst.h>
#include "gst-nvdscommonconfig.h"
#include "deepstream_perf.h"
G_BEGIN_DECLS
/**
 * Creates a new Stream ADD message - denoting a new stream getting added to
 * nvmultiurisrcbin.
 *
 * params[in] obj           The GStreamer object creating the message.
 * params[in] sensor_info   Sensor info of the stream which is getting added
 *                          into nvmultiurisrcbin
 *
 * @return  A pointer to the new message.
 */
    GstMessage * gst_nvmessage_new_stream_add (GstObject * obj,
    NvDsSensorInfo * sensor_info);

/**
 * Determines whether a message is a stream ADD message.
 *
 * params[in] message   A pointer to the message to be checked.
 *
 * @return  A Boolean; true if the message is a stream ADD message.
 */
gboolean gst_nvmessage_is_stream_add (GstMessage * message);

/**
 * \brief  Parses the stream ID from a stream ADD message.
 *
 * The stream ID is the index of the stream which is getting added
 * to nvmultiurisrcbin
 *
 * params[in] message           A pointer to a stream ADD message.
 * params[out] stream_id        A pointer to @ref NvDsSensorInfo
 *    The string NvDsSensorInfo->sensor_id should not be modified,
 *    and remains valid until the next
 *    call to a gst_nvmessage_parse*() function with the given message.
 *    Please use or make copy within the bus callback scope.
 *
 * @return  A Boolean; true if the message was successfully parsed.
 */
gboolean gst_nvmessage_parse_stream_add (GstMessage * message,
    NvDsSensorInfo * sensor_info);

/**
 * Creates a new Stream REMOVE message - denoting a new stream getting removed
 * from nvmultiurisrcbin.
 *
 * params[in] obj           The GStreamer object creating the message.
 * params[in] sensor_info   Sensor info of the stream which is getting removed
 *                          from nvmultiurisrcbin
 *
 * @return  A pointer to the new message.
 */
GstMessage *gst_nvmessage_new_stream_remove (GstObject * obj,
    NvDsSensorInfo * sensor_info);

/**
 * Determines whether a message is a stream REMOVE message.
 *
 * params[in] message   A pointer to the message to be checked.
 *
 * @return  A Boolean; true if the message is a stream REMOVE message.
 */
gboolean gst_nvmessage_is_stream_remove (GstMessage * message);

/**
 * \brief  Parses the stream ID from a stream REMOVE message.
 *
 * The stream ID is the index of the stream which is getting removed
 * from nvmultiurisrcbin
 *
 * params[in] message           A pointer to a stream REMOVE message.
 * params[out] stream_id        A pointer to @ref NvDsSensorInfo
 *    The string NvDsSensorInfo->sensor_id should not be modified,
 *    and remains valid until the next
 *    call to a gst_nvmessage_parse*() function with the given message.
 *    Please use or make copy within the bus callback scope.
 *
 * @return  A Boolean; true if the message was successfully parsed.
 */
gboolean gst_nvmessage_parse_stream_remove (GstMessage * message,
    NvDsSensorInfo * sensor_info);

/**
 * \brief  Parses the stream ID from a stream ADD message.
 *
 * The stream ID is the index of the stream which is getting added
 * to nvmultiurisrcbin
 *
 * params[in] message           A pointer to a stream ADD message.
 * params[out] stream_id        A pointer to @ref NvDsSensorInfo
 *    The string NvDsFPSSensorInfo->sensor_id should not be modified,
 *    and remains valid until the next
 *    call to a gst_nvmessage_parse*() function with the given message.
 *    Please use or make copy within the bus callback scope.
 *
 * @return  A Boolean; true if the message was successfully parsed.
 */

gboolean gst_nvmessage_parse_fps_stream_add (GstMessage * message,
    NvDsFPSSensorInfo * sensor_info);

/**
 * \brief  Parses the stream ID from a stream REMOVE message.
 *
 * The stream ID is the index of the stream which is getting removed
 * from nvmultiurisrcbin
 *
 * params[in] message           A pointer to a stream REMOVE message.
 * params[out] stream_id        A pointer to @ref NvDsSensorInfo
 *    The string NvDsFPSSensorInfo->sensor_id should not be modified,
 *    and remains valid until the next
 *    call to a gst_nvmessage_parse*() function with the given message.
 *    Please use or make copy within the bus callback scope.
 *
 * @return  A Boolean; true if the message was successfully parsed.
 */

gboolean gst_nvmessage_parse_fps_stream_remove (GstMessage * message,
    NvDsFPSSensorInfo * sensor_info);

/**
 * \brief  Sends custom message to force-eos on the pipeline
 *
 * The element on which custom message related to eos is to be sent.
 *
 * params[in] object           A gboject on which custom message to be sent
 * params[in] force_eos        A force_eos variable 
 * @return  A GstMessage;      The GstMessage which was successfully sent
 */
GstMessage *gst_nvmessage_force_pipeline_eos (GstObject * obj,
    gboolean force_eos);


/**
 * \brief  Parses the force_eos from a force eos message.
 *
 *
 * params[in] message           A pointer to a force eos message.
 * params[out] force_eos        A pointer to force_eos variable
 * @return  A Boolean; true if the message was successfully parsed.
 */
gboolean
gst_nvmessage_parse_force_pipeline_eos (GstMessage * message,
    gboolean * force_eos);


/**
 * Determines whether a message is a force pipeline eos message.
 *
 * params[in] message   A pointer to the message to be checked.
 *
 * @return  A Boolean; true if the message is a force pipeline message.
 */
gboolean gst_nvmessage_is_force_pipeline_eos (GstMessage * message);

/**
 * Creates a new attempt-exceeded message - denoting a reconnection attempt
 * is exceeded for a rtsp uri
 *
 * params[in] obj           The GStreamer object creating the message.
 * params[in] rtsp_info     rtsp info of the stream which exceeds rtsp
 *                          reconnection attempt
 *
 * @return  A pointer to the new message.
 */
GstMessage *gst_nvmessage_reconnect_attempt_exceeded (GstObject * obj,
      NvDsRtspAttemptsInfo *rtsp_info);

/**
 * \brief  Parses the NvDsRtspAttemptsInfo from a attempt-exceeded message.
 *
 *
 * params[in] message                A pointer to a attempt-exceeded message.
 * params[out] NvDsRtspAttemptsInfo  A pointer to @ref NvDsRtspAttemptsInfo
 * @return  A Boolean; true if the message was successfully parsed.
 */
gboolean
gst_nvmessage_parse_reconnect_attempt_exceeded (GstMessage * message,
    NvDsRtspAttemptsInfo *rtsp_info);

/**
 * Determines whether a message is a attempt-exceeded message.
 *
 * params[in] message   A pointer to the message to be checked.
 *
 * @return  A Boolean; true if the message is a attempt exceeded message.
 */
gboolean gst_nvmessage_is_reconnect_attempt_exceeded (GstMessage * message);

/** @} */

G_END_DECLS
#endif
