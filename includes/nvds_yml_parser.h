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

/**
* @file nvds_yml_parser.h
* <b>NVIDIA DeepStream Yaml Parser API Specification </b>
*
* @b Description: This file specifies the APIs to set DeepStream GStreamer Element
* properties by parsing YAML file.
*/

/**
* @defgroup   yamlparser_api  DeepStream Yaml Parser API
* Defines an API for the GStreamer NvDsYaml plugin.
* @ingroup custom_gstreamer
* @{
*/

#ifndef _NVGSTDS_YAML_PARSER_H_
#define _NVGSTDS_YAML_PARSER_H_

#include <gst/gst.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * Enum for Yaml parsing status for the API call on a GstElement.
 */
typedef enum NvDsYamlParserStatus
{
    /** Properties were set correctly */
    NVDS_YAML_PARSER_SUCCESS,
    /** Property group was disabled, properties were not set. */
    NVDS_YAML_PARSER_DISABLED,
    /** Encountered an error while setting properties. */
    NVDS_YAML_PARSER_ERROR
} NvDsYamlParserStatus;

/**
 * Enum for codec type for the API call on a encoder.
 */
typedef struct NvDsYamlCodecStatus
{
    /** Type codec=1 h264, codec=2 h265 */
    guint codec_type;
    /** Enable status */
    gboolean enable;
} NvDsYamlCodecStatus;
/**
 * Enum for specifying the inference plugin type.
 */
typedef enum
{
    /** Inference using nvinfer GIE */
    NVDS_GIE_PLUGIN_INFER = 0,
    /** Inference using nvinferserver GIE */
    NVDS_GIE_PLUGIN_INFER_SERVER,
} NvDsGieType;

/**
* Set properties of a filesrc element from values specified in a YAML configuration file.
*
* @param[in]  element GStreamer element on which properties are to be set.
* @param[in]  cfg_file_path The YAML config file used by an application.
* @param[in]  group Group in the YAML config file to be parsed and
*             corresponding properties set on the filesrc element.
* @return Yaml parsing status for the API call.
*/
NvDsYamlParserStatus
nvds_parse_file_source (GstElement * element, gchar * cfg_file_path,
    const char *group);


/**
* Set properties of a uridecodebin element from values specified in a YAML configuration file.
*
* @param[in]  element GStreamer element on which properties are to be set.
* @param[in]  cfg_file_path The YAML config file used by an application.
* @param[in]  group Group in the YAML config file to be parsed and
*             corresponding properties set on the uridecodebin element.
* @return Yaml parsing status for the API call.
*/
NvDsYamlParserStatus
nvds_parse_uridecodebin (GstElement * element, gchar * cfg_file_path,
    const char *group);


/**
* Set properties of a rtspsrc element from values specified in a YAML configuration file.
*
* @param[in]  element GStreamer element on which properties are to be set.
* @param[in]  cfg_file_path The YAML config file used by an application.
* @param[in]  group Group in the YAML config file to be parsed and
*             corresponding properties set on the rtspsrc element.
* @return Yaml parsing status for the API call.
*/
NvDsYamlParserStatus
nvds_parse_rtsp_source (GstElement * element, gchar * cfg_file_path,
    const char *group);


/**
* Set properties of a nvarguscamerasrc element from values specified in a YAML configuration file.
*
* @param[in]  element GStreamer element on which properties are to be set.
* @param[in]  cfg_file_path The YAML config file used by an application.
* @param[in]  group Group in the YAML config file to be parsed and
*             corresponding properties set on the nvarguscamerasrc element.
* @return Yaml parsing status for the API call.
*/
NvDsYamlParserStatus
nvds_parse_nvarguscamerasrc (GstElement * element, gchar * cfg_file_path,
    const char *group);

/**
* Set properties of a v4l2src element from values specified in a YAML configuration file.
*
* @param[in]  element GStreamer element on which properties are to be set.
* @param[in]  cfg_file_path The YAML config file used by an application.
* @param[in]  group Group in the YAML config file to be parsed and
*             corresponding properties set on the v4l2src element.
* @return Yaml parsing status for the API call.
*/
NvDsYamlParserStatus
nvds_parse_v4l2src (GstElement * element, gchar * cfg_file_path,
    const char *group);

/**
* Set properties of a multifilesrc element from values specified in a YAML configuration file.
*
* @param[in]  element GStreamer element on which properties are to be set.
* @param[in]  The YAML config file used by an application.
* @param[in]  group Group in the YAML config file to be parsed and
*             corresponding properties set on the multifilesrc element.
* @return Yaml parsing status for the API call.
*/
NvDsYamlParserStatus
nvds_parse_multifilesrc (GstElement * element, gchar * cfg_file_path,
    const char *group);

/**
* Set properties of a alsasrc element from values specified in a YAML configuration file.
*
* @param[in]  element GStreamer element on which properties are to be set.
* @param[in]  cfg_file_path The YAML config file used by an application.
* @param[in]  group Group in the YAML config file to be parsed and
*             corresponding properties set on the alsasrc element.
* @return Yaml parsing status for the API call.
*/
NvDsYamlParserStatus
nvds_parse_alsasrc (GstElement * element, gchar * cfg_file_path,
    const char *group);


/**
* Parse semicolon separated uri(s) in the source-list group and store it in a GList
*
* @param[in]  src_list The empty GList address.
* @param[in]  cfg_file_path The YAML config file used by an application.
* @param[in]  group Group in the YAML config file on which parsing is done
*             A key "list" is present in the group which contains semicolon
*             separated uri(s). Once the API call finishes, the GList contains
*             the uris(s).
* @return Yaml parsing status for the API call.
*/
NvDsYamlParserStatus
nvds_parse_source_list (GList ** src_list, gchar * cfg_file_path,
    const char *group);

/**
* Set properties of a nvstreammux element from values specified in a YAML configuration file.
*
* @param[in]  element GStreamer element on which properties are to be set.
* @param[in]  cfg_file_path The YAML config file used by an application.
* @param[in]  group Group in the YAML config file to be parsed and
*             corresponding properties set on the nvstreammux element.
* @return Yaml parsing status for the API call.
*/
NvDsYamlParserStatus
nvds_parse_streammux (GstElement * element, gchar * cfg_file_path,
    const char *group);

/**
* Set properties of a nvtracker element from values specified in a YAML configuration file.
*
* @param[in]  element GStreamer element on which properties are to be set.
* @param[in]  cfg_file_path The YAML config file used by an application.
* @param[in]  group Group in the YAML config file to be parsed and
*             corresponding properties set on the nvtracker element.
* @return Yaml parsing status for the API call.
*/
NvDsYamlParserStatus
nvds_parse_tracker (GstElement * element, gchar * cfg_file_path,
    const char *group);

/**
* Set properties of a nvdsosd element from values specified in a YAML configuration file.
*
* @param[in]  element The gst element on which properties are to be set.
* @param[in]  cfg_file_path The YAML config file used by an application.
* @param[in]  group The group in the YAML config file on which parsing is done
*             and corresponding properties are set on the nvdsosd element.
* @return Yaml parsing status for the API call.
*/
NvDsYamlParserStatus
nvds_parse_osd (GstElement * element, gchar * cfg_file_path,
    const char *group);

/**
* Set properties of a nvtiler element from values specified in a YAML configuration file.
*
* @param[in]  element GStreamer element on which properties are to be set.
* @param[in]  cfg_file_path The YAML config file used by an application.
* @param[in]  group Group in the YAML config file to be parsed and
*             corresponding properties set on the nvmultistreamtiler element.
* @return Yaml parsing status for the API call.
*/
NvDsYamlParserStatus
nvds_parse_tiler (GstElement * element, gchar * cfg_file_path,
    const char *group);

/**
* Set properties of a nvmsgbroker element from values specified in a YAML configuration file.
*
* @param[in]  element GStreamer element on which properties are to be set.
* @param[in]  cfg_file_path The YAML config file used by an application.
* @param[in]  group Group in the YAML config file to be parsed and
*             corresponding properties set on the nmvsgbroker element.
* @return Yaml parsing status for the API call.
*/
NvDsYamlParserStatus
nvds_parse_msgbroker (GstElement * element, gchar * cfg_file_path,
    const char *group);

/**
* Set properties of a nvmsgconv element from values specified in a YAML configuration file.
*
* @param[in]  element GStreamer element on which properties are to be set.
* @param[in]  cfg_file_path The YAML config file used by an application.
* @param[in]  group Group in the YAML config file to be parsed and
*             corresponding properties set on the nvmsgconv element.
* @return Yaml parsing status for the API call.
*/
NvDsYamlParserStatus
nvds_parse_msgconv (GstElement * element, gchar * cfg_file_path,
    const char *group);

/**
* Set properties of a nvinfer element from values specified in a YAML configuration file.
*
* @param[in]  element GStreamer element on which properties are to be set.
* @param[in]  cfg_file_path The YAML config file used by an application.
* @param[in]  group Group in the YAML config file to be parsed and
*             corresponding properties set on the nvinfer element.
* @return Yaml parsing status for the API call.
*/
NvDsYamlParserStatus
nvds_parse_gie (GstElement * element, gchar * cfg_file_path,
    const char *group);

/**
* Get the GIE type, nvinfer or invinferserver, from the YAML configuration file.
*
* @param[out]  gie_type Pointer to variable of NvDSGieType to be updated if
*              the plugin type is specified.
* @param[in]   cfg_file_path The YAML config file used by an application.
* @param[in]   group Group in the YAML config file to be parsed and
*              corresponding properties set on the nvinfer element.
* @return Yaml parsing status for the API call.
*/
NvDsYamlParserStatus
nvds_parse_gie_type (NvDsGieType * gie_type, gchar * cfg_file_path,
    const char *group);

/**
* Set properties of a nveglglessink element from values specified in a YAML configuration file.
*
* @param[in]  element Gstreamer element on which properties are to be set.
* @param[in]  cfg_file_path The YAML config file used by an application.
* @param[in]  group Group in the YAML config file to be parsed and
*             corresponding properties set on the nveglglessink element.
* @return Yaml parsing status for the API call.
*/
NvDsYamlParserStatus
nvds_parse_egl_sink (GstElement * element, gchar * cfg_file_path,
    const char *group);

/**
* Set properties of a nv3dsink element from values specified in a YAML configuration file.
*
* @param[in]  element Gstreamer element on which properties are to be set.
* @param[in]  cfg_file_path The YAML config file used by an application.
* @param[in]  group Group in the YAML config file to be parsed and
*             corresponding properties set on the nv3dsink element.
* @return Yaml parsing status for the API call.
*/
NvDsYamlParserStatus
nvds_parse_3d_sink (GstElement * element, gchar * cfg_file_path,
    const char *group);

/**
* Set properties of a filesink element from values specified in a YAML configuration file.
*
* @param[in]  element GStreamer element on which properties are to be set.
* @param[in]  cfg_file_path The YAML config file used by an application.
* @param[in]  group Group in the YAML config file to be parsed and
*             corresponding properties set on the filesink element.
* @return Yaml parsing status for the API call.
*/
NvDsYamlParserStatus
nvds_parse_file_sink (GstElement * element, gchar * cfg_file_path,
    const char *group);

/**
* Set properties of a fakesink element from values specified in a YAML configuration file.
*
* @param[in]  element GStreamer element on which properties are to be set.
* @param[in]  cfg_file_path The YAML config file used by an application.
* @param[in]  group Group in the YAML config file to be parsed and
*             corresponding properties set on the fakesink element.
* @return Yaml parsing status for the API call.
*/
NvDsYamlParserStatus
nvds_parse_fake_sink (GstElement * element, gchar * cfg_file_path,
    const char *group);

/**
* Parses codec type and enable status values specified in a YAML configuration file.
*
* @param[in]  cfg_file_path The YAML config file used by an application.
* @param[in]  group Group in the YAML config file to be parsed.
*             The Group contains codec type parameter, based on which h264 or h265 encoder is used.
* @return codec type and enable status of encoder as pair for the API call.
*/
NvDsYamlParserStatus
nvds_parse_codec_status (gchar * cfg_file_path, const char *group,
    NvDsYamlCodecStatus * codec_status);

/**
* Set properties of a nvv4l2h264enc element from values specified in a YAML configuration file.
*
* @param[in]  element GStreamer element on which properties are to be set.
* @param[in]  cfg_file_path The YAML config file used by an application.
* @param[in]  group Group in the YAML config file to be parsed and
*             corresponding properties set on the nvv4l2h264enc element.
* @return Yaml parsing status for the API call.
*/
NvDsYamlParserStatus
nvds_parse_nvv4l2h264enc (GstElement * element, gchar * cfg_file_path,
    const char *group);


/**
* Set properties of a nvv4l2h265enc element from values specified in a YAML configuration file.
*
* @param[in]  element GStreamer element on which properties are to be set.
* @param[in]  cfg_file_path The YAML config file used by an application.
* @param[in]  group Group in the YAML config file to be parsed and
*             corresponding properties set on the nvv4l2h265enc element.
* @return Yaml parsing status for the API call.
*/
NvDsYamlParserStatus
nvds_parse_nvv4l2h265enc (GstElement * element, gchar * cfg_file_path,
    const char *group);

/**
* Set properties of a nvmultiurisrcbin element from values specified in a YAML configuration file.
*
* @param[in]  element GStreamer element on which properties are to be set.
* @param[in]  cfg_file_path The YAML config file used by an application.
* @param[in]  group Group in the YAML config file to be parsed and
*             corresponding properties set on the nvv4l2h265enc element.
* @return Yaml parsing status for the API call.
*/
NvDsYamlParserStatus
nvds_parse_multiurisrcbin (GstElement * element, gchar * cfg_file_path,
    const char *group);

/**
* Set properties of a nvdsxfer element from values specified in a YAML configuration file.
*
* @param[in]  element GStreamer element on which properties are to be set.
* @param[in]  cfg_file_path The YAML config file used by an application.
* @param[in]  group Group in the YAML config file to be parsed and
*             corresponding properties set on the nvdsxfer element.
* @return Yaml parsing status for the API call.
*/
NvDsYamlParserStatus
nvds_parse_nvxfer(GstElement *element, gchar *cfg_file_path, const char* group);


/**
* Parses nvdsxfer plugin position values specified in a YAML configuration file.
*
* @param[in]  cfg_file_path The YAML config file used by an application.
* @param[in]  group Group in the YAML config file to be parsed and
*             corresponding properties set on the nvdsxfer element.
* @param[out]  position Pointer to variable holding position value.
* @return Yaml parsing status for the API call.
*/
NvDsYamlParserStatus
nvds_parse_nvxfer_position (gchar *cfg_file_path, const char* group, guint* position);

/**
* Set properties of a nvdsanalytics element from values specified in a YAML configuration file.
*
* @param[in]  element GStreamer element on which properties are to be set.
* @param[in]  cfg_file_path The YAML config file used by an application.
* @param[in]  group Group in the YAML config file to be parsed and
*             corresponding properties set on the nvdsanalytics element.
* @return Yaml parsing status for the API call.
*/
NvDsYamlParserStatus
nvds_parse_nvdsanalytics(GstElement *element, gchar *cfg_file_path, const char* group);

/**
* Function definition to check if the rest server is enabled in the YAML config file.
*
* @param[in]  cfg_file_path YAML config file name/path.
* @param[in]  group Group in the YAML config file to be parsed and
*             corresponding properties set on the nvdsanalytics element.
* @return Boolean value on the basis of rest server status.
*/
gboolean
check_enable_status (gchar * cfg_file_path, const char* group);

#ifdef __cplusplus
}
#endif

#endif                          /* _NVGSTDS_YAML_PARSER_H_ */

/** @} */
