/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DeepStream: Message Schema payload Generation</b>
 *
 * @b Description: This file specifies the functions used to generate payload
 * based on NVIDIA Deepstream message schema either using eventMsg metadata
 * or the NvDSFrame(obj) metadata
 */

#ifndef NVEVENTMSGCONV_H_
#define NVEVENTMSGCONV_H_

#include "nvdsmeta.h"
#include "nvdsmeta_schema.h"
#include <iostream>
#include <string>
#include <unordered_map>

using namespace std;

#define CONFIG_GROUP_SENSOR "sensor"
#define CONFIG_GROUP_PLACE "place"
#define CONFIG_GROUP_ANALYTICS "analytics"

#define CONFIG_KEY_COORDINATE "coordinate"
#define CONFIG_KEY_DESCRIPTION "description"
#define CONFIG_KEY_ENABLE "enable"
#define CONFIG_KEY_ID "id"
#define CONFIG_KEY_LANE "lane"
#define CONFIG_KEY_LEVEL "level"
#define CONFIG_KEY_LOCATION "location"
#define CONFIG_KEY_NAME "name"
#define CONFIG_KEY_SOURCE "source"
#define CONFIG_KEY_TYPE "type"
#define CONFIG_KEY_VERSION "version"


#define CONFIG_KEY_PLACE_SUB_FIELD1 "place-sub-field1"
#define CONFIG_KEY_PLACE_SUB_FIELD2 "place-sub-field2"
#define CONFIG_KEY_PLACE_SUB_FIELD3 "place-sub-field3"

#define DEFAULT_CSV_FIELDS 10


#define CHECK_ERROR(error)                           \
    if (error) {                                     \
        cout << "Error: " << error->message << endl; \
        goto done;                                   \
    }

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Store data parsed from the config file in these structures
 */
struct NvDsPlaceSubObject {
    string field1;
    string field2;
    string field3;
};

struct NvDsSensorObject {
    string id;
    string type;
    string desc;
    gdouble location[3];
    gdouble coordinate[3];
};

struct NvDsPlaceObject {
    string id;
    string name;
    string type;
    gdouble location[3];
    gdouble coordinate[3];
    NvDsPlaceSubObject subObj;
};

struct NvDsAnalyticsObject {
    string id;
    string desc;
    string source;
    string version;
};

struct NvDs3dDatamap {
    string obj_key_2d;
    string obj_key_3d;
    string obj_key_fusion;
    string lidar_data_key;
    guint32 lidar_element_size = 4;
    guint32 lidar_element_max_points = 10;
};

struct NvDsPayloadPriv {
    unordered_map<int, NvDsSensorObject> sensorObj;
    unordered_map<int, NvDsPlaceObject> placeObj;
    unordered_map<int, NvDsAnalyticsObject> analyticsObj;
    NvDs3dDatamap datamapCfg;
};

gchar* generate_event_message(void* privData, NvDsEventMsgMeta* meta);
gchar* generate_event_message_minimal(void* privData, NvDsEvent* events, guint size);
gchar* generate_event_message_protobuf(void* privData, NvDsEvent* events, guint size, size_t& message_len);
gchar* generate_dsmeta_message(void* privData, void* frameMeta, void* objMeta);
gchar* generate_dsmeta_message_minimal(void* privData, void* frameMeta);
gchar* generate_dsmeta_message_protobuf(void* privData, void* frameMeta, size_t& message_len);
gchar* generate_dsmeta_message_ds3d(void* privData, void* ptrDataMap, gboolean addLidarData, size_t& message_len);
void* create_deepstream_schema_ctx();
void destroy_deepstream_schema_ctx(void* privData);
bool nvds_msg2p_parse_key_value(void* privData, const gchar* file);
bool nvds_msg2p_parse_csv(void* privData, const gchar* file);
bool nvds_msg2p_parse_yaml(void* privData, const gchar* file);

#ifdef __cplusplus
}
#endif
#endif /* NVEVENTMSGCONV_H_ */
