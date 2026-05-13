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

#include "gst-nvdscustommessage.h"

#define STREAM_ADD_STRUCT_NAME "stream-add"
#define STREAM_REMOVE_STRUCT_NAME "stream-remove"
#define PIPELINE_EOS_STRUCT_NAME "force-pipeline-eos"
#define PIPELINE_RTSP_RECONNECT_ATTEMPT_STRUCT_NAME "attempt-exceeded"

#define CHECK_MESSAGE_TYPE(message,type) do { \
  const GstStructure *str; \
  if (GST_MESSAGE_TYPE (message) != GST_MESSAGE_ELEMENT) \
    return FALSE; \
  str = gst_message_get_structure (message); \
  return (str != NULL) && gst_structure_has_name (str,type); \
} while (0)

GstMessage *
gst_nvmessage_new_stream_add (GstObject * obj, NvDsSensorInfo * sensor_info)
{
  GstStructure *str =
      gst_structure_new (STREAM_ADD_STRUCT_NAME, "source-id", G_TYPE_UINT,
      sensor_info->source_id,
      "sensor-id", G_TYPE_STRING, sensor_info->sensor_id,
      "sensor-name", G_TYPE_STRING, sensor_info->sensor_name,
      "uri", G_TYPE_STRING, sensor_info->uri,  NULL);

  GstMessage *message = gst_message_new_custom (GST_MESSAGE_ELEMENT, obj, str);

  return message;
}

gboolean
gst_nvmessage_is_force_pipeline_eos (GstMessage * message)
{
  CHECK_MESSAGE_TYPE (message, PIPELINE_EOS_STRUCT_NAME);
}

GstMessage *
gst_nvmessage_force_pipeline_eos (GstObject * obj, gboolean force_eos)
{
  GstStructure *str = gst_structure_new (PIPELINE_EOS_STRUCT_NAME,
      "force_eos", G_TYPE_BOOLEAN, force_eos, NULL);

  GstMessage *message = gst_message_new_custom (GST_MESSAGE_ELEMENT, obj, str);

  return message;
}

gboolean
gst_nvmessage_parse_force_pipeline_eos (GstMessage * message,
    gboolean * force_eos)
{
  const GstStructure *str;

  if (!gst_nvmessage_is_force_pipeline_eos (message))
    return FALSE;

  str = gst_message_get_structure (message);

  gst_structure_get_boolean (str, "force_eos", force_eos);
  return TRUE;
}

gboolean
gst_nvmessage_is_stream_add (GstMessage * message)
{
  CHECK_MESSAGE_TYPE (message, STREAM_ADD_STRUCT_NAME);
}

gboolean
gst_nvmessage_parse_stream_add (GstMessage * message,
    NvDsSensorInfo * sensor_info)
{
  const GstStructure *str;

  if (!gst_nvmessage_is_stream_add (message))
    return FALSE;

  str = gst_message_get_structure (message);
  gst_structure_get_uint (str, "source-id", &sensor_info->source_id);
  sensor_info->sensor_id = gst_structure_get_string (str, "sensor-id");
  sensor_info->sensor_name = gst_structure_get_string (str, "sensor-name");
  sensor_info->uri = gst_structure_get_string (str, "uri");
  return TRUE;
}

gboolean
gst_nvmessage_parse_fps_stream_add (GstMessage * message,
    NvDsFPSSensorInfo * sensor_info)
{
  const GstStructure *str;

  if (!gst_nvmessage_is_stream_add (message))
    return FALSE;

  str = gst_message_get_structure (message);
  gst_structure_get_uint (str, "source-id", &sensor_info->source_id);
  sensor_info->sensor_id = gst_structure_get_string (str, "sensor-id");
  sensor_info->sensor_name = gst_structure_get_string (str, "sensor-name");
  sensor_info->uri = gst_structure_get_string (str, "uri");
  return TRUE;
}

GstMessage *
gst_nvmessage_new_stream_remove (GstObject * obj, NvDsSensorInfo * sensor_info)
{
  GstStructure *str =
      gst_structure_new (STREAM_REMOVE_STRUCT_NAME, "source-id", G_TYPE_UINT,
      sensor_info->source_id,
      "sensor-id", G_TYPE_STRING, sensor_info->sensor_id,
      "sensor-name", G_TYPE_STRING, sensor_info->sensor_name,
      "uri", G_TYPE_STRING, sensor_info->uri, NULL);

  GstMessage *message = gst_message_new_custom (GST_MESSAGE_ELEMENT, obj, str);

  return message;
}

gboolean
gst_nvmessage_is_stream_remove (GstMessage * message)
{
  CHECK_MESSAGE_TYPE (message, STREAM_REMOVE_STRUCT_NAME);
}

gboolean
gst_nvmessage_parse_stream_remove (GstMessage * message,
    NvDsSensorInfo * sensor_info)
{
  const GstStructure *str;

  if (!gst_nvmessage_is_stream_remove (message))
    return FALSE;

  str = gst_message_get_structure (message);
  gst_structure_get_uint (str, "source-id", &sensor_info->source_id);
  sensor_info->sensor_id = gst_structure_get_string (str, "sensor-id");
  sensor_info->sensor_name = gst_structure_get_string (str, "sensor-name");
  sensor_info->uri = gst_structure_get_string (str, "uri");
  return TRUE;
}

gboolean
gst_nvmessage_parse_fps_stream_remove (GstMessage * message,
    NvDsFPSSensorInfo * sensor_info)
{
  const GstStructure *str;

  if (!gst_nvmessage_is_stream_remove (message))
    return FALSE;

  str = gst_message_get_structure (message);
  gst_structure_get_uint (str, "source-id", &sensor_info->source_id);
  sensor_info->sensor_id = gst_structure_get_string (str, "sensor-id");
  sensor_info->sensor_name = gst_structure_get_string (str, "sensor-name");
  sensor_info->uri = gst_structure_get_string (str, "uri");
  return TRUE;
}


GstMessage *
gst_nvmessage_reconnect_attempt_exceeded (GstObject * obj, NvDsRtspAttemptsInfo * rtsp_info)
{
  GstStructure *str = gst_structure_new (PIPELINE_RTSP_RECONNECT_ATTEMPT_STRUCT_NAME,
      "source-id", G_TYPE_UINT, rtsp_info->source_id,
      "attempt_exceeded", G_TYPE_BOOLEAN, rtsp_info->attempt_exceeded, NULL);

  GstMessage *message = gst_message_new_custom (GST_MESSAGE_ELEMENT, obj, str);

  return message;
}


gboolean
gst_nvmessage_is_reconnect_attempt_exceeded (GstMessage * message)
{
  CHECK_MESSAGE_TYPE (message, PIPELINE_RTSP_RECONNECT_ATTEMPT_STRUCT_NAME);
}


gboolean
gst_nvmessage_parse_reconnect_attempt_exceeded (GstMessage * message,
    NvDsRtspAttemptsInfo * rtsp_info)
{
  const GstStructure *str;

  if (!gst_nvmessage_is_reconnect_attempt_exceeded (message))
    return FALSE;

  str = gst_message_get_structure (message);

  gst_structure_get_boolean (str, "attempt_exceeded", &rtsp_info->attempt_exceeded);
  gst_structure_get_uint (str, "source-id", &rtsp_info->source_id);
  return TRUE;
}
