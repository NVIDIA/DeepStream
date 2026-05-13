/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <json-glib/json-glib.h>
#include <uuid.h>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <cstring>
#include <vector>
#include "deepstream_schema.h"
#include <google/protobuf/util/time_util.h>
//#include "nv-schema/src/main/c++/schema.pb.h"
#include "schema.pb.h"
#include "lidar_schema.pb.h"

#include <ds3d/common/func_utils.h>
#include <ds3d/common/hpp/datamap.hpp>
#include <ds3d/common/impl/impl_frames.h>
#include "ds3d/common/ds3d_analysis_datatype.h"

#include <cuda_runtime.h>

using namespace ds3d;

#define MAX_TIME_STAMP_LEN (64)

static void
generate_ts_rfc3339(char* buf, int buf_size)
{
    time_t tloc;
    struct tm tm_log;
    struct timespec ts;
    char strmsec[6];  //.nnnZ\0

    clock_gettime(CLOCK_REALTIME, &ts);
    memcpy(&tloc, (void*)(&ts.tv_sec), sizeof(time_t));
    gmtime_r(&tloc, &tm_log);
    strftime(buf, buf_size, "%Y-%m-%dT%H:%M:%S", &tm_log);
    int ms = ts.tv_nsec / 1000000;
    g_snprintf(strmsec, sizeof(strmsec), ".%.3dZ", ms);
    strncat(buf, strmsec, buf_size);
}

static JsonObject*
generate_place_object(void* privData, NvDsFrameMeta* frame_meta)
{
    NvDsPayloadPriv* privObj = NULL;
    NvDsPlaceObject* dsPlaceObj = NULL;
    JsonObject* placeObj;
    JsonObject* jobject;
    JsonObject* jobject2;

    privObj = (NvDsPayloadPriv*)privData;
    auto idMap = privObj->placeObj.find(frame_meta->source_id);

    if (idMap != privObj->placeObj.end()) {
        dsPlaceObj = &idMap->second;
    } else {
        cout << "No entry for " CONFIG_GROUP_PLACE << frame_meta->source_id << " in configuration file" << endl;
        return NULL;
    }

    /* place object
     * "place":
       {
         "id": "string",
         "name": "endeavor",
         “type”: “garage”,
         "location": {
           "lat": 30.333,
           "lon": -40.555,
           "alt": 100.00
         },
         "entrance/aisle": {
           "name": "walsh",
           "lane": "lane1",
           "level": "P2",
           "coordinate": {
             "x": 1.0,
             "y": 2.0,
             "z": 3.0
           }
         }
       }
     */

    placeObj = json_object_new();
    json_object_set_string_member(placeObj, "id", dsPlaceObj->id.c_str());
    json_object_set_string_member(placeObj, "name", dsPlaceObj->name.c_str());
    json_object_set_string_member(placeObj, "type", dsPlaceObj->type.c_str());

    // location sub object
    jobject = json_object_new();
    json_object_set_double_member(jobject, "lat", dsPlaceObj->location[0]);
    json_object_set_double_member(jobject, "lon", dsPlaceObj->location[1]);
    json_object_set_double_member(jobject, "alt", dsPlaceObj->location[2]);
    json_object_set_object_member(placeObj, "location", jobject);

    // place sub object (user to provide the name for sub place ex: parkingSpot/aisle/entrance..etc
    jobject = json_object_new();

    json_object_set_string_member(jobject, "id", dsPlaceObj->subObj.field1.c_str());
    json_object_set_string_member(jobject, "name", dsPlaceObj->subObj.field2.c_str());
    json_object_set_string_member(jobject, "level", dsPlaceObj->subObj.field3.c_str());
    json_object_set_object_member(placeObj, "place-sub-field", jobject);

    // coordinates for place sub object
    jobject2 = json_object_new();
    json_object_set_double_member(jobject2, "x", dsPlaceObj->coordinate[0]);
    json_object_set_double_member(jobject2, "y", dsPlaceObj->coordinate[1]);
    json_object_set_double_member(jobject2, "z", dsPlaceObj->coordinate[2]);
    json_object_set_object_member(jobject, "coordinate", jobject2);

    return placeObj;
}

static JsonObject*
generate_sensor_object(void* privData, NvDsFrameMeta* frame_meta)
{
    NvDsPayloadPriv* privObj = NULL;
    NvDsSensorObject* dsSensorObj = NULL;
    JsonObject* sensorObj;
    JsonObject* jobject;

    privObj = (NvDsPayloadPriv*)privData;
    auto idMap = privObj->sensorObj.find(frame_meta->source_id);

    if (idMap != privObj->sensorObj.end()) {
        dsSensorObj = &idMap->second;
    } else {
        cout << "No entry for " CONFIG_GROUP_SENSOR << frame_meta->source_id << " in configuration file" << endl;
        return NULL;
    }

    /* sensor object
     * "sensor": {
         "id": "string",
         "type": "Camera/Puck",
         "location": {
           "lat": 45.99,
           "lon": 35.54,
           "alt": 79.03
         },
         "coordinate": {
           "x": 5.2,
           "y": 10.1,
           "z": 11.2
         },
         "description": "Entrance of Endeavor Garage Right Lane"
       }
     */

    // sensor object
    sensorObj = json_object_new();
    json_object_set_string_member(sensorObj, "id", dsSensorObj->id.c_str());
    json_object_set_string_member(sensorObj, "type", dsSensorObj->type.c_str());
    json_object_set_string_member(sensorObj, "description", dsSensorObj->desc.c_str());

    // location sub object
    jobject = json_object_new();
    json_object_set_double_member(jobject, "lat", dsSensorObj->location[0]);
    json_object_set_double_member(jobject, "lon", dsSensorObj->location[1]);
    json_object_set_double_member(jobject, "alt", dsSensorObj->location[2]);
    json_object_set_object_member(sensorObj, "location", jobject);

    // coordinate sub object
    jobject = json_object_new();
    json_object_set_double_member(jobject, "x", dsSensorObj->coordinate[0]);
    json_object_set_double_member(jobject, "y", dsSensorObj->coordinate[1]);
    json_object_set_double_member(jobject, "z", dsSensorObj->coordinate[2]);
    json_object_set_object_member(sensorObj, "coordinate", jobject);

    return sensorObj;
}

static JsonObject*
generate_analytics_module_object(void* privData, NvDsFrameMeta* frame_meta)
{
    NvDsPayloadPriv* privObj = NULL;
    NvDsAnalyticsObject* dsObj = NULL;
    JsonObject* analyticsObj;

    privObj = (NvDsPayloadPriv*)privData;

    auto idMap = privObj->analyticsObj.find(frame_meta->source_id);

    if (idMap != privObj->analyticsObj.end()) {
        dsObj = &idMap->second;
    } else {
        cout << "No entry for " CONFIG_GROUP_ANALYTICS << frame_meta->source_id << " in configuration file" << endl;
        return NULL;
    }

    /* analytics object
     * "analyticsModule": {
         "id": "string",
         "description": "Vehicle Detection and License Plate Recognition",
         "confidence": 97.79,
         "source": "OpenALR",
         "version": "string"
       }
     */

    // analytics object
    analyticsObj = json_object_new();
    json_object_set_string_member(analyticsObj, "id", dsObj->id.c_str());
    json_object_set_string_member(analyticsObj, "description", dsObj->desc.c_str());
    json_object_set_string_member(analyticsObj, "source", dsObj->source.c_str());
    json_object_set_string_member(analyticsObj, "version", dsObj->version.c_str());

    return analyticsObj;
}

static JsonObject*
generate_object_object(void* privData, NvDsFrameMeta* frame_meta, NvDsObjectMeta* obj_meta)
{
    JsonObject* objectObj;
    JsonObject* jobject;
    gchar tracking_id[64];
    // GList *objectMask = NULL;

    // object object
    objectObj = json_object_new();
    if (snprintf(tracking_id, sizeof(tracking_id), "%lu", obj_meta->object_id) >= (int)sizeof(tracking_id))
        g_warning("Not enough space to copy trackingId");
    json_object_set_string_member(objectObj, "id", tracking_id);
    json_object_set_double_member(objectObj, "speed", 0);
    json_object_set_double_member(objectObj, "direction", 0);
    json_object_set_double_member(objectObj, "orientation", 0);

    jobject = json_object_new();
    json_object_set_double_member(jobject, "confidence", obj_meta->confidence);

    // Fetch object classifiers detected
    for (NvDsClassifierMetaList* cl = obj_meta->classifier_meta_list; cl; cl = cl->next) {
        NvDsClassifierMeta* cl_meta = (NvDsClassifierMeta*)cl->data;

        for (NvDsLabelInfoList* ll = cl_meta->label_info_list; ll; ll = ll->next) {
            NvDsLabelInfo* ll_meta = (NvDsLabelInfo*)ll->data;
            if (cl_meta->classifier_type != NULL && strcmp("", cl_meta->classifier_type))
                json_object_set_string_member(jobject, cl_meta->classifier_type, ll_meta->result_label);
        }
    }
    json_object_set_object_member(objectObj, obj_meta->obj_label, jobject);

    // bbox sub object
    float scaleW =
        (float)frame_meta->source_frame_width / (frame_meta->pipeline_width == 0) ? 1 : frame_meta->pipeline_width;
    float scaleH =
        (float)frame_meta->source_frame_height / (frame_meta->pipeline_height == 0) ? 1 : frame_meta->pipeline_height;

    float left = obj_meta->rect_params.left * scaleW;
    float top = obj_meta->rect_params.top * scaleH;
    float width = obj_meta->rect_params.width * scaleW;
    float height = obj_meta->rect_params.height * scaleH;

    jobject = json_object_new();
    json_object_set_int_member(jobject, "topleftx", left);
    json_object_set_int_member(jobject, "toplefty", top);
    json_object_set_int_member(jobject, "bottomrightx", left + width);
    json_object_set_int_member(jobject, "bottomrighty", top + height);
    json_object_set_object_member(objectObj, "bbox", jobject);

    // location sub object
    jobject = json_object_new();
    json_object_set_object_member(objectObj, "location", jobject);

    // coordinate sub object
    jobject = json_object_new();
    json_object_set_object_member(objectObj, "coordinate", jobject);

    return objectObj;
}

static JsonObject*
generate_event_object(NvDsObjectMeta* obj_meta)
{
    JsonObject* eventObj;
    uuid_t uuid;
    gchar uuidStr[37];

    /*
     * "event": {
         "id": "event-id",
         "type": "entry / exit"
       }
     */

    uuid_generate_random(uuid);
    uuid_unparse_lower(uuid, uuidStr);

    eventObj = json_object_new();
    json_object_set_string_member(eventObj, "id", uuidStr);
    json_object_set_string_member(eventObj, "type", "");
    return eventObj;
}

gchar*
generate_dsmeta_message(void* privData, void* frameMeta, void* objMeta)
{
    JsonNode* rootNode;
    JsonObject* rootObj;
    JsonObject* placeObj;
    JsonObject* sensorObj;
    JsonObject* analyticsObj;
    JsonObject* eventObj;
    JsonObject* objectObj;
    gchar* message;

    NvDsFrameMeta* frame_meta = (NvDsFrameMeta*)frameMeta;
    NvDsObjectMeta* obj_meta = (NvDsObjectMeta*)objMeta;

    uuid_t msgId;
    gchar msgIdStr[37];

    uuid_generate_random(msgId);
    uuid_unparse_lower(msgId, msgIdStr);

    // place object
    placeObj = generate_place_object(privData, frame_meta);

    // sensor object
    sensorObj = generate_sensor_object(privData, frame_meta);

    // analytics object
    analyticsObj = generate_analytics_module_object(privData, frame_meta);

    // object object
    objectObj = generate_object_object(privData, frame_meta, obj_meta);
    // event object
    eventObj = generate_event_object(obj_meta);

    char ts[MAX_TIME_STAMP_LEN + 1];
    generate_ts_rfc3339(ts, MAX_TIME_STAMP_LEN);

    // root object
    rootObj = json_object_new();
    json_object_set_string_member(rootObj, "messageid", msgIdStr);
    json_object_set_string_member(rootObj, "mdsversion", "1.0");
    json_object_set_string_member(rootObj, "@timestamp", ts);
    json_object_set_object_member(rootObj, "place", placeObj);
    json_object_set_object_member(rootObj, "sensor", sensorObj);
    json_object_set_object_member(rootObj, "analyticsModule", analyticsObj);
    json_object_set_object_member(rootObj, "object", objectObj);
    json_object_set_object_member(rootObj, "event", eventObj);

    json_object_set_string_member(rootObj, "videoPath", "");

    // Search for any custom message blob within frame usermeta list
    JsonArray* jArray = json_array_new();
    for (NvDsUserMetaList* l = frame_meta->frame_user_meta_list; l; l = l->next) {
        NvDsUserMeta* frame_usermeta = (NvDsUserMeta*)l->data;
        if (frame_usermeta && frame_usermeta->base_meta.meta_type == NVDS_CUSTOM_MSG_BLOB) {
            NvDsCustomMsgInfo* custom_blob = (NvDsCustomMsgInfo*)frame_usermeta->user_meta_data;
            string msg = string((const char*)custom_blob->message, custom_blob->size);
            json_array_add_string_element(jArray, msg.c_str());
        }
    }
    if (json_array_get_length(jArray) > 0)
        json_object_set_array_member(rootObj, "customMessage", jArray);
    else
        json_array_unref(jArray);

    rootNode = json_node_new(JSON_NODE_OBJECT);
    json_node_set_object(rootNode, rootObj);

    message = json_to_string(rootNode, TRUE);
    json_node_free(rootNode);
    json_object_unref(rootObj);

    return message;
}

gchar*
generate_dsmeta_message_minimal(void* privData, void* frameMeta)
{
    /*
    The JSON structure of the frame
    {
     "version": "4.0",
     "id": "frame-id",
     "@timestamp": "2018-04-11T04:59:59.828Z",
     "sensor": "sensor-id",
     "objects": [
        ".......object-1 attributes...........",
        ".......object-2 attributes...........",
        ".......object-3 attributes..........."
      ]
    }
    */

    /*
    An example object with Vehicle object-type
    {
      "version": "4.0",
      "id": "frame-id",
      "@timestamp": "2018-04-11T04:59:59.828Z",
      "sensorId": "sensor-id",
      "objects": [
          "957|1834|150|1918|215|Vehicle|#|sedan|Bugatti|M|blue|CA 444|California|0.8",
          "..........."
      ]
    }
     */

    JsonNode* rootNode;
    JsonObject* jobject;
    JsonArray* jArray;
    stringstream ss;
    gchar* message = NULL;

    jArray = json_array_new();

    NvDsFrameMeta* frame_meta = (NvDsFrameMeta*)frameMeta;
    for (NvDsObjectMetaList* obj_l = frame_meta->obj_meta_list; obj_l; obj_l = obj_l->next) {
        NvDsObjectMeta* obj_meta = (NvDsObjectMeta*)obj_l->data;
        if (obj_meta == NULL) {
            // Ignore Null object.
            continue;
        }

        // bbox sub object
        float scaleW =
            (float)frame_meta->source_frame_width / (frame_meta->pipeline_width == 0) ? 1 : frame_meta->pipeline_width;
        float scaleH = (float)frame_meta->source_frame_height / (frame_meta->pipeline_height == 0)
                           ? 1
                           : frame_meta->pipeline_height;

        float left = obj_meta->rect_params.left * scaleW;
        float top = obj_meta->rect_params.top * scaleH;
        float width = obj_meta->rect_params.width * scaleW;
        float height = obj_meta->rect_params.height * scaleH;

        ss.str("");
        ss.clear();
        ss << obj_meta->object_id << "|" << left << "|" << top << "|" << left + width << "|" << top + height << "|"
           << obj_meta->obj_label;

        if (g_list_length(obj_meta->classifier_meta_list) > 0) {
            ss << "|#";
            // Add classifiers for the object, if any
            for (NvDsClassifierMetaList* cl = obj_meta->classifier_meta_list; cl; cl = cl->next) {
                NvDsClassifierMeta* cl_meta = (NvDsClassifierMeta*)cl->data;
                for (NvDsLabelInfoList* ll = cl_meta->label_info_list; ll; ll = ll->next) {
                    NvDsLabelInfo* ll_meta = (NvDsLabelInfo*)ll->data;
                    ss << "|" << ll_meta->result_label;
                }
            }
            ss << "|" << obj_meta->confidence;
        }
        json_array_add_string_element(jArray, ss.str().c_str());
    }

    // generate timestamp
    char ts[MAX_TIME_STAMP_LEN + 1];
    generate_ts_rfc3339(ts, MAX_TIME_STAMP_LEN);

    // fetch sensor id
    string sensorId = "0";
    NvDsPayloadPriv* privObj = (NvDsPayloadPriv*)privData;
    auto idMap = privObj->sensorObj.find(frame_meta->source_id);
    if (idMap != privObj->sensorObj.end()) {
        NvDsSensorObject& obj = privObj->sensorObj[frame_meta->source_id];
        sensorId = obj.id;
    }

    jobject = json_object_new();
    json_object_set_string_member(jobject, "version", "4.0");
    json_object_set_string_member(jobject, "id", to_string(frame_meta->frame_num).c_str());
    json_object_set_string_member(jobject, "@timestamp", ts);
    json_object_set_string_member(jobject, "sensorId", sensorId.c_str());

    json_object_set_array_member(jobject, "objects", jArray);

    JsonArray* custMsgjArray = json_array_new();
    // Search for any custom message blob within frame usermeta list
    for (NvDsUserMetaList* l = frame_meta->frame_user_meta_list; l; l = l->next) {
        NvDsUserMeta* frame_usermeta = (NvDsUserMeta*)l->data;
        if (frame_usermeta && frame_usermeta->base_meta.meta_type == NVDS_CUSTOM_MSG_BLOB) {
            NvDsCustomMsgInfo* custom_blob = (NvDsCustomMsgInfo*)frame_usermeta->user_meta_data;
            string msg = string((const char*)custom_blob->message, custom_blob->size);
            json_array_add_string_element(custMsgjArray, msg.c_str());
        }
    }
    if (json_array_get_length(custMsgjArray) > 0)
        json_object_set_array_member(jobject, "customMessage", custMsgjArray);
    else
        json_array_unref(custMsgjArray);

    rootNode = json_node_new(JSON_NODE_OBJECT);
    json_node_set_object(rootNode, jobject);

    message = json_to_string(rootNode, TRUE);
    json_node_free(rootNode);
    json_object_unref(jobject);

    return message;
}

gchar*
generate_dsmeta_message_protobuf(void* privData, void* frameMeta, size_t& message_len)
{
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    nv::Frame pbFrame;

    pbFrame.set_version("4.0");

    // frameId
    NvDsFrameMeta* frame_meta = (NvDsFrameMeta*)frameMeta;
    pbFrame.set_id(std::to_string(frame_meta->frame_num));

    // generate timestamp
    char ts[MAX_TIME_STAMP_LEN + 1];
    generate_ts_rfc3339(ts, MAX_TIME_STAMP_LEN);
    std::string ts_string(ts);
    google::protobuf::Timestamp* timestamp = pbFrame.mutable_timestamp();
    if (!::google::protobuf::util::TimeUtil::FromString(ts_string, timestamp)) {
        message_len = 0;
        return NULL;
    }

    // fetch sensor id
    string sensorId = "0";
    NvDsPayloadPriv* privObj = (NvDsPayloadPriv*)privData;
    auto idMap = privObj->sensorObj.find(frame_meta->source_id);
    if (idMap != privObj->sensorObj.end()) {
        NvDsSensorObject& obj = privObj->sensorObj[frame_meta->source_id];
        sensorId = obj.id;
    }
    pbFrame.set_sensorid(sensorId);

    // objects
    for (NvDsObjectMetaList* obj_l = frame_meta->obj_meta_list; obj_l; obj_l = obj_l->next) {
        NvDsObjectMeta* obj_meta = (NvDsObjectMeta*)obj_l->data;
        if (obj_meta == NULL) {
            // Ignore Null object.
            continue;
        }

        // bbox sub object
        float scaleW =
            (float)frame_meta->source_frame_width / (frame_meta->pipeline_width == 0) ? 1 : frame_meta->pipeline_width;
        float scaleH = (float)frame_meta->source_frame_height / (frame_meta->pipeline_height == 0)
                           ? 1
                           : frame_meta->pipeline_height;

        float left = obj_meta->rect_params.left * scaleW;
        float top = obj_meta->rect_params.top * scaleH;
        float width = obj_meta->rect_params.width * scaleW;
        float height = obj_meta->rect_params.height * scaleH;

        nv::Object* object = pbFrame.add_objects();
        object->set_id(std::to_string(obj_meta->object_id));

        nv::Bbox* bbox = object->mutable_bbox();
        bbox->set_leftx(left);
        bbox->set_topy(top);
        bbox->set_rightx(left + width);
        bbox->set_bottomy(top + height);

        object->set_type(obj_meta->obj_label);
        object->set_confidence(obj_meta->confidence);
    }

    std::string msg_str;
    if (!pbFrame.SerializeToString(&msg_str)) {
        cout << "generate_event_message_protobuf : Failed to serialize protobuf message to string.\n";
        message_len = 0;
        return NULL;
    }

    message_len = msg_str.length();
    // Save the content of msg_str before the function returns which puts msg_str out of scope.
    gchar* message = (gchar*)g_memdup(msg_str.c_str(), message_len);
    return message;
}

static std::string
print_2d_obj(Object2DBbox* obj2D)
{
    stringstream ss;
    ss.str("");
    ss.clear();
    ss << obj2D->centerX << "|" << obj2D->centerY << "|" << obj2D->dx << "|" << obj2D->dy << "|" << obj2D->score << "|"
       << obj2D->labels;
    LOG_DEBUG(
        "[%5.3f, %5.3f, %5.3f, %5.3f], score: %2.3f, %s", obj2D->centerX, obj2D->centerY, obj2D->dx, obj2D->dy,
        obj2D->score, obj2D->labels);
    return ss.str();
}

static std::string
print_3d_obj(Lidar3DBbox* obj3D)
{
    stringstream ss;
    ss.str("");
    ss.clear();
    ss << obj3D->centerX << "|" << obj3D->centerY << "|" << obj3D->centerZ << "|" << obj3D->dx << "|" << obj3D->dy
       << "|" << obj3D->dz << "|" << obj3D->score << "|" << obj3D->labels;
    LOG_DEBUG(
        "[%5.3f, %5.3f, %5.3f, %5.3f, %5.3f, %5.3f], score: %2.3f, %s", obj3D->centerX, obj3D->centerY, obj3D->centerZ,
        obj3D->dx, obj3D->dy, obj3D->dz, obj3D->score, obj3D->labels);
    return ss.str();
}

static gchar*
ds3d_append_pointcloud(NvDsPayloadPriv* privObj, GuardDataMap datamap, gchar* msg, size_t& message_len)
{
    std::string message(msg ? msg : "", message_len);
    ds3dmsg::LidarPointCloud pc;

    FrameGuard lidarFrame;
    if (!isGood(datamap.getGuardData(privObj->datamapCfg.lidar_data_key, lidarFrame))) {
        LOG_DEBUG("No lidar data found in datamap from alignment filter\n");
        return msg;
    }
    Shape pShape = lidarFrame->shape();  // N x 3 | N x 4
    DS_ASSERT(pShape.numDims);
    // XYZ | XYZI
    uint32_t frameEleSize = pShape.d[pShape.numDims - 1];
    uint32_t numPoints = (uint32_t)(ShapeSize(pShape) / frameEleSize);
    numPoints = std::min<uint32_t>(numPoints, privObj->datamapCfg.lidar_element_max_points);
    if (!numPoints) {
        LOG_DEBUG("lidar data has 0 points\n");
        return msg;
    }
    LOG_DEBUG("adding lidar %d point data into message\n", (int32_t)numPoints);
    void *basePtr = lidarFrame->base();
    float* inputLidarPoints = (float*)basePtr;
    std::vector<float> hBuf;
    if (lidarFrame->memType() == MemType::kGpuCuda) {
        hBuf.resize(numPoints * frameEleSize);
        inputLidarPoints = &hBuf[0];
        cudaMemcpy((void*)inputLidarPoints, basePtr, hBuf.size() * sizeof(float), cudaMemcpyDeviceToHost);
    }

    for (uint32_t p = 0; p < numPoints; p++) {
        ds3dmsg::Point* point = pc.add_points();
        point->set_x(inputLidarPoints[0]);
        point->set_y(inputLidarPoints[1]);
        point->set_z(inputLidarPoints[2]);
        float w = ((frameEleSize == 4) ? inputLidarPoints[3] : 1.0);
        point->set_intensity(w);
        inputLidarPoints += frameEleSize;
    }
    std::string pc_str("");
    pc.SerializeToString(&pc_str);
    LOG_DEBUG("lidar data protobuf size:%d", (int32_t)pc_str.size());
    message += pc_str;
    message_len = message.size();
    gchar *out_msg = (gchar *)g_memdup2(message.c_str(), message_len);

    if (msg) {
        g_free(msg);
    }
    return out_msg;
}

uint32_t
add_2d_objects(FrameGuard& video2dBboxData, JsonObject* jobject)
{
    Shape obj2DShape = video2dBboxData->shape();
    JsonArray* jArray;
    DS_ASSERT(obj2DShape.numDims);
    uint32_t per2DObjBytes = obj2DShape.d[obj2DShape.numDims - 1];
    uint32_t num2DBbox = ShapeSize(obj2DShape) / per2DObjBytes;
    Object2DBbox* obj2D = (Object2DBbox*)video2dBboxData->base();
    LOG_DEBUG("msgconv Received %d 2D bbox.", num2DBbox);
    jArray = json_array_new();
    for (uint32_t i = 0; i < num2DBbox; ++i) {
        json_array_add_string_element(jArray, print_2d_obj(&obj2D[i]).c_str());
    }
    json_object_set_array_member(jobject, "2d_objects", jArray);
    return num2DBbox;
}

uint32_t
add_3d_objects(FrameGuard& lidar3dBboxData, JsonObject* jobject)
{
    JsonArray* jArray;
    Shape obj3DShape = lidar3dBboxData->shape();  // 1 X N X sizeof(Lidar3DBbox)
    DS_ASSERT(obj3DShape.numDims);
    uint32_t per3DObjBytes = obj3DShape.d[obj3DShape.numDims - 1];
    uint32_t num3DBbox = ShapeSize(obj3DShape) / per3DObjBytes;
    Lidar3DBbox* obj3D = (Lidar3DBbox*)lidar3dBboxData->base();
    LOG_DEBUG("msgconv Received %d 3D bbox.", num3DBbox);
    jArray = json_array_new();
    for (uint32_t i = 0; i < num3DBbox; ++i) {
        json_array_add_string_element(jArray, print_3d_obj(&obj3D[i]).c_str());
    }
    json_object_set_array_member(jobject, "3d_objects", jArray);
    return num3DBbox;
}

uint32_t
add_fused_objects(FrameGuard& fusedDetectionData, JsonObject* jobject)
{
    JsonArray* jArray;
    Shape objFusedShape = fusedDetectionData->shape();
    DS_ASSERT(objFusedShape.numDims);
    FusedDetection* objFused = (FusedDetection*)fusedDetectionData->base();
    uint32_t perFusedObjBytes = objFusedShape.d[objFusedShape.numDims - 1];
    uint32_t numFused = ShapeSize(objFusedShape) / perFusedObjBytes;

    jArray = json_array_new();
    for (uint32_t i = 0; i < numFused; ++i) {
        JsonObject* result = json_object_new();
        json_object_set_string_member(result, "2d", print_2d_obj(&objFused[i].obj2D).c_str());
        json_object_set_string_member(result, "3d", print_3d_obj(&objFused[i].obj3D).c_str());
        json_object_set_double_member(result, "score", objFused[i].score);
        json_array_add_object_element(jArray, result);
    }
    json_object_set_array_member(jobject, "fusion_results", jArray);
    return numFused;
}

gchar*
generate_dsmeta_message_ds3d(void* privData, void* ptrDataMap, gboolean addLidarData, size_t& message_len)
{
    const abiRefDataMap* refDataMap = (const abiRefDataMap*)ptrDataMap;
    NvDsPayloadPriv* privObj = (NvDsPayloadPriv*)privData;
    GuardDataMap inputData(*refDataMap);

    // extract 3d lidar and 2d video bbox from datamap
    GuardDataMap datamap(*refDataMap);
    FrameGuard lidar3dBboxData;
    FrameGuard video2dBboxData;
    FrameGuard fusedDetectionData;
    JsonNode* rootNode;
    JsonObject* jobject;
    gchar* message = NULL;
    message_len = 0;

    uint32_t num2DBbox = 0;
    uint32_t num3DBbox = 0;
    uint32_t numFused = 0;

    jobject = json_object_new();
    json_object_set_string_member(jobject, "version", "ds3d/1.0");

    TimeStamp ts{0};
    if(isGood(datamap.getData(kTimeStamp, ts))){
        json_object_set_string_member(jobject, "datamap@timestamp", std::to_string(ts.t0).c_str());
    }

    if (!isGood(datamap.getGuardData(privObj->datamapCfg.obj_key_2d, video2dBboxData))) {
        LOG_WARNING("No 2d bbox data: found in datamap in sensor fusion\n");
    } else {
        num2DBbox = add_2d_objects(video2dBboxData, jobject);
    }
    if (!isGood(datamap.getGuardData(privObj->datamapCfg.obj_key_3d, lidar3dBboxData))) {
        LOG_WARNING("No 3d bbox data: found in datamap in sensor fusion\n");
    } else {
        num3DBbox = add_3d_objects(lidar3dBboxData, jobject);
    }
    if (!isGood(datamap.getGuardData(privObj->datamapCfg.obj_key_fusion, fusedDetectionData))) {
        LOG_WARNING("No fusion data: found in datamap in sensor fusion\n");
    } else {
        numFused = add_fused_objects(fusedDetectionData, jobject);
    }

    rootNode = json_node_new(JSON_NODE_OBJECT);
    json_node_set_object(rootNode, jobject);

    if (num2DBbox + num3DBbox + numFused > 0) {
        message = json_to_string(rootNode, TRUE);
        message_len = strlen(message);
    }
    json_node_free(rootNode);
    json_object_unref(jobject);

    if (addLidarData) {
        message = ds3d_append_pointcloud(privObj, inputData, message, message_len);
    }

    return message;
}
