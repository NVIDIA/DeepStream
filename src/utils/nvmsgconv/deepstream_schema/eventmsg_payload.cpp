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

#include <json-glib/json-glib.h>
#include <uuid.h>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <cstring>
#include <vector>
#include "deepstream_schema.h"

#include <google/protobuf/util/time_util.h>
#include "schema.pb.h"

static const std::vector<std::vector<std::string>> _joint_maps = {
  {// Type 0 - pose2D
    std::string("nose"),
    std::string("neck"),
    std::string("right-shoulder"),
    std::string("right-elbow"),
    std::string("right-hand"),
    std::string("left-shoulder"),
    std::string("left-elbow"),
    std::string("left-hand"),
    std::string("right-hip"),
    std::string("right-knee"),
    std::string("right-foot"),
    std::string("left-hip"),
    std::string("left-knee"),
    std::string("left-foot"),
    std::string("right-eye"),
    std::string("left-eye"),
    std::string("right-ear"),
    std::string("left-ear"),
  },
  {// Type 1 - pose25D
    std::string("pelvis"),
    std::string("left-hip"),
    std::string("right-hip"),
    std::string("torso"),
    std::string("left-knee"),
    std::string("right-knee"),
    std::string("neck"),
    std::string("left-ankle"),
    std::string("right-ankle"),
    std::string("left-big-toe"),
    std::string("right-big-toe"),
    std::string("left-small-toe"),
    std::string("right-small-toe"),
    std::string("left-heel"),
    std::string("right-heel"),
    std::string("nose"),
    std::string("left-eye"),
    std::string("right-eye"),
    std::string("left-ear"),
    std::string("right-ear"),
    std::string("left-shoulder"),
    std::string("right-shoulder"),
    std::string("left-elbow"),
    std::string("right-elbow"),
    std::string("left-wrist"),
    std::string("right-wrist"),
    std::string("left-pinky-knuckle"),
    std::string("right-pinky-knuckle"),
    std::string("left-middle-tip"),
    std::string("right-middle-tip"),
    std::string("left-index-knuckle"),
    std::string("right-index-knuckle"),
    std::string("left-thumb-tip"),
    std::string("right-thumb-tip"),
  },
  {// Type 2 - pose3D
    std::string("pelvis"),
    std::string("left-hip"),
    std::string("right-hip"),
    std::string("torso"),
    std::string("left-knee"),
    std::string("right-knee"),
    std::string("neck"),
    std::string("left-ankle"),
    std::string("right-ankle"),
    std::string("left-big-toe"),
    std::string("right-big-toe"),
    std::string("left-small-toe"),
    std::string("right-small-toe"),
    std::string("left-heel"),
    std::string("right-heel"),
    std::string("nose"),
    std::string("left-eye"),
    std::string("right-eye"),
    std::string("left-ear"),
    std::string("right-ear"),
    std::string("left-shoulder"),
    std::string("right-shoulder"),
    std::string("left-elbow"),
    std::string("right-elbow"),
    std::string("left-wrist"),
    std::string("right-wrist"),
    std::string("left-pinky-knuckle"),
    std::string("right-pinky-knuckle"),
    std::string("left-middle-tip"),
    std::string("right-middle-tip"),
    std::string("left-index-knuckle"),
    std::string("right-index-knuckle"),
    std::string("left-thumb-tip"),
    std::string("right-thumb-tip"),
  }
};

static JsonObject*
generate_place_object (void *privData, NvDsEventMsgMeta *meta)
{
  NvDsPayloadPriv *privObj = NULL;
  NvDsPlaceObject *dsPlaceObj = NULL;
  JsonObject *placeObj;
  JsonObject *jobject;
  JsonObject *jobject2;

  privObj = (NvDsPayloadPriv *) privData;
  auto idMap = privObj->placeObj.find (meta->placeId);

  if (idMap != privObj->placeObj.end()) {
    dsPlaceObj = &idMap->second;
  } else {
    cout << "No entry for " CONFIG_GROUP_PLACE << meta->placeId
        << " in configuration file" << endl;
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

  placeObj = json_object_new ();
  json_object_set_string_member (placeObj, "id", dsPlaceObj->id.c_str());
  json_object_set_string_member (placeObj, "name", dsPlaceObj->name.c_str());
  json_object_set_string_member (placeObj, "type", dsPlaceObj->type.c_str());

  // location sub object
  jobject = json_object_new ();
  json_object_set_double_member (jobject, "lat", dsPlaceObj->location[0]);
  json_object_set_double_member (jobject, "lon", dsPlaceObj->location[1]);
  json_object_set_double_member (jobject, "alt", dsPlaceObj->location[2]);
  json_object_set_object_member (placeObj, "location", jobject);

  // parkingSpot / aisle /entrance sub object
  jobject = json_object_new ();

  switch (meta->type) {
    case NVDS_EVENT_MOVING:
    case NVDS_EVENT_STOPPED:
      json_object_set_string_member (jobject, "id", dsPlaceObj->subObj.field1.c_str());
      json_object_set_string_member (jobject, "name", dsPlaceObj->subObj.field2.c_str());
      json_object_set_string_member (jobject, "level", dsPlaceObj->subObj.field3.c_str());
      json_object_set_object_member (placeObj, "aisle", jobject);
      break;
    case NVDS_EVENT_EMPTY:
    case NVDS_EVENT_PARKED:
      json_object_set_string_member (jobject, "id", dsPlaceObj->subObj.field1.c_str());
      json_object_set_string_member (jobject, "type", dsPlaceObj->subObj.field2.c_str());
      json_object_set_string_member (jobject, "level", dsPlaceObj->subObj.field3.c_str());
      json_object_set_object_member (placeObj, "parkingSpot", jobject);
      break;
    case NVDS_EVENT_ENTRY:
    case NVDS_EVENT_EXIT:
      if (meta->objType == NVDS_OBJECT_TYPE_VEHICLE) {
        json_object_set_string_member (jobject, "id", dsPlaceObj->subObj.field1.c_str());
        json_object_set_string_member (jobject, "name", dsPlaceObj->subObj.field2.c_str());
        json_object_set_string_member (jobject, "level", dsPlaceObj->subObj.field3.c_str());
        json_object_set_object_member (placeObj, "aisle", jobject);
      } else {
        json_object_set_string_member (jobject, "name", dsPlaceObj->subObj.field1.c_str());
        json_object_set_string_member (jobject, "lane", dsPlaceObj->subObj.field2.c_str());
        json_object_set_string_member (jobject, "level", dsPlaceObj->subObj.field3.c_str());
        json_object_set_object_member (placeObj, "entrance", jobject);
      }
      break;
    default:
      cout << "Event type not implemented " << endl;
      break;
  }

  // coordinate sub sub object
  jobject2 = json_object_new ();
  json_object_set_double_member (jobject2, "x", dsPlaceObj->coordinate[0]);
  json_object_set_double_member (jobject2, "y", dsPlaceObj->coordinate[1]);
  json_object_set_double_member (jobject2, "z", dsPlaceObj->coordinate[2]);
  json_object_set_object_member (jobject, "coordinate", jobject2);

  return placeObj;
}

static JsonObject*
generate_sensor_object (void *privData, NvDsEventMsgMeta *meta)
{
  NvDsPayloadPriv *privObj = NULL;
  NvDsSensorObject *dsSensorObj = NULL;
  JsonObject *sensorObj;
  JsonObject *jobject;

  privObj = (NvDsPayloadPriv *) privData;
  auto idMap = privObj->sensorObj.find (meta->sensorId);

  if (idMap != privObj->sensorObj.end()) {
    dsSensorObj = &idMap->second;
  } else {
    cout << "No entry for " CONFIG_GROUP_SENSOR << meta->sensorId
         << " in configuration file" << endl;
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
  sensorObj = json_object_new ();
  json_object_set_string_member (sensorObj, "id", dsSensorObj->id.c_str());
  json_object_set_string_member (sensorObj, "type", dsSensorObj->type.c_str());
  json_object_set_string_member (sensorObj, "description", dsSensorObj->desc.c_str());

  // location sub object
  jobject = json_object_new ();
  json_object_set_double_member (jobject, "lat", dsSensorObj->location[0]);
  json_object_set_double_member (jobject, "lon", dsSensorObj->location[1]);
  json_object_set_double_member (jobject, "alt", dsSensorObj->location[2]);
  json_object_set_object_member (sensorObj, "location", jobject);

  // coordinate sub object
  jobject = json_object_new ();
  json_object_set_double_member (jobject, "x", dsSensorObj->coordinate[0]);
  json_object_set_double_member (jobject, "y", dsSensorObj->coordinate[1]);
  json_object_set_double_member (jobject, "z", dsSensorObj->coordinate[2]);
  json_object_set_object_member (sensorObj, "coordinate", jobject);

  return sensorObj;
}

static JsonObject*
generate_analytics_module_object (void *privData, NvDsEventMsgMeta *meta)
{
  NvDsPayloadPriv *privObj = NULL;
  NvDsAnalyticsObject *dsObj = NULL;
  JsonObject *analyticsObj;

  privObj = (NvDsPayloadPriv *) privData;

  auto idMap = privObj->analyticsObj.find (meta->moduleId);

  if (idMap != privObj->analyticsObj.end()) {
    dsObj = &idMap->second;
  } else {
    cout << "No entry for " CONFIG_GROUP_ANALYTICS << meta->moduleId
        << " in configuration file" << endl;
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
  analyticsObj = json_object_new ();
  json_object_set_string_member (analyticsObj, "id", dsObj->id.c_str());
  json_object_set_string_member (analyticsObj, "description", dsObj->desc.c_str());
  json_object_set_string_member (analyticsObj, "source", dsObj->source.c_str());
  json_object_set_string_member (analyticsObj, "version", dsObj->version.c_str());

  return analyticsObj;
}

static JsonObject*
generate_event_object (void *privData, NvDsEventMsgMeta *meta)
{
  JsonObject *eventObj;
  uuid_t uuid;
  gchar uuidStr[37];

  /*
   * "event": {
       "id": "event-id",
       "type": "entry / exit"
     }
   */

  uuid_generate_random (uuid);
  uuid_unparse_lower(uuid, uuidStr);

  eventObj = json_object_new ();
  json_object_set_string_member (eventObj, "id", uuidStr);

  switch (meta->type) {
    case NVDS_EVENT_ENTRY:
      json_object_set_string_member (eventObj, "type", "entry");
      break;
    case NVDS_EVENT_EXIT:
      json_object_set_string_member (eventObj, "type", "exit");
      break;
    case NVDS_EVENT_MOVING:
      json_object_set_string_member (eventObj, "type", "moving");
      break;
    case NVDS_EVENT_STOPPED:
      json_object_set_string_member (eventObj, "type", "stopped");
      break;
    case NVDS_EVENT_PARKED:
      json_object_set_string_member (eventObj, "type", "parked");
      break;
    case NVDS_EVENT_EMPTY:
      json_object_set_string_member (eventObj, "type", "empty");
      break;
    case NVDS_EVENT_RESET:
      json_object_set_string_member (eventObj, "type", "reset");
      break;
    default:
      cout << "Unknown event type " << endl;
      break;
  }

  return eventObj;
}

static JsonObject*
generate_object_object (void *privData, NvDsEventMsgMeta *meta)
{
  JsonObject *objectObj;
  JsonObject *jobject;
  guint i;
  gchar tracking_id[64];
  GList *objectMask = NULL;

  // object object
  objectObj = json_object_new ();
  if (snprintf (tracking_id, sizeof(tracking_id), "%lu", meta->trackingId)
      >= (int) sizeof(tracking_id))
    g_warning("Not enough space to copy trackingId");
  json_object_set_string_member (objectObj, "id", tracking_id);
  json_object_set_double_member (objectObj, "speed", 0);
  json_object_set_double_member (objectObj, "direction", 0);
  json_object_set_double_member (objectObj, "orientation", 0);

  switch (meta->objType) {
    case NVDS_OBJECT_TYPE_VEHICLE:
      // vehicle sub object
      jobject = json_object_new ();

      if (meta->extMsgSize) {
        NvDsVehicleObject *dsObj = (NvDsVehicleObject *) meta->extMsg;
        if (dsObj) {
          json_object_set_string_member (jobject, "type", dsObj->type);
          json_object_set_string_member (jobject, "make", dsObj->make);
          json_object_set_string_member (jobject, "model", dsObj->model);
          json_object_set_string_member (jobject, "color", dsObj->color);
          json_object_set_string_member (jobject, "licenseState", dsObj->region);
          json_object_set_string_member (jobject, "license", dsObj->license);
          json_object_set_double_member (jobject, "confidence", meta->confidence);
        }
      } else {
        // No vehicle object in meta data. Attach empty vehicle sub object.
        json_object_set_string_member (jobject, "type", "");
        json_object_set_string_member (jobject, "make", "");
        json_object_set_string_member (jobject, "model", "");
        json_object_set_string_member (jobject, "color", "");
        json_object_set_string_member (jobject, "licenseState", "");
        json_object_set_string_member (jobject, "license", "");
        json_object_set_double_member (jobject, "confidence", 1.0);
      }
      json_object_set_object_member (objectObj, "vehicle", jobject);
      break;
    case NVDS_OBJECT_TYPE_PERSON:
      // person sub object
      jobject = json_object_new ();

      if (meta->extMsgSize) {
        NvDsPersonObject *dsObj = (NvDsPersonObject *) meta->extMsg;
        if (dsObj) {
          json_object_set_int_member (jobject, "age", dsObj->age);
          json_object_set_string_member (jobject, "gender", dsObj->gender);
          json_object_set_string_member (jobject, "hair", dsObj->hair);
          json_object_set_string_member (jobject, "cap", dsObj->cap);
          json_object_set_string_member (jobject, "apparel", dsObj->apparel);
          json_object_set_double_member (jobject, "confidence", meta->confidence);
        }
      } else {
        // No person object in meta data. Attach empty person sub object.
        json_object_set_int_member (jobject, "age", 0);
        json_object_set_string_member (jobject, "gender", "");
        json_object_set_string_member (jobject, "hair", "");
        json_object_set_string_member (jobject, "cap", "");
        json_object_set_string_member (jobject, "apparel", "");
        json_object_set_double_member (jobject, "confidence", 1.0);
      }
      json_object_set_object_member (objectObj, "person", jobject);
      break;
    case NVDS_OBJECT_TYPE_FACE:
      // face sub object
      jobject = json_object_new ();

      if (meta->extMsgSize) {
        NvDsFaceObject *dsObj = (NvDsFaceObject *) meta->extMsg;
        if (dsObj) {
          json_object_set_int_member (jobject, "age", dsObj->age);
          json_object_set_string_member (jobject, "gender", dsObj->gender);
          json_object_set_string_member (jobject, "hair", dsObj->hair);
          json_object_set_string_member (jobject, "cap", dsObj->cap);
          json_object_set_string_member (jobject, "glasses", dsObj->glasses);
          json_object_set_string_member (jobject, "facialhair", dsObj->facialhair);
          json_object_set_string_member (jobject, "name", dsObj->name);
          json_object_set_string_member (jobject, "eyecolor", dsObj->eyecolor);
          json_object_set_double_member (jobject, "confidence", meta->confidence);
        }
      } else {
        // No face object in meta data. Attach empty face sub object.
        json_object_set_int_member (jobject, "age", 0);
        json_object_set_string_member (jobject, "gender", "");
        json_object_set_string_member (jobject, "hair", "");
        json_object_set_string_member (jobject, "cap", "");
        json_object_set_string_member (jobject, "glasses", "");
        json_object_set_string_member (jobject, "facialhair", "");
        json_object_set_string_member (jobject, "name", "");
        json_object_set_string_member (jobject, "eyecolor", "");
        json_object_set_double_member (jobject, "confidence", 1.0);
      }
      json_object_set_object_member (objectObj, "face", jobject);
      break;
    case NVDS_OBJECT_TYPE_VEHICLE_EXT:
      // vehicle sub object
      jobject = json_object_new ();

      if (meta->extMsgSize) {
        NvDsVehicleObjectExt *dsObj = (NvDsVehicleObjectExt *) meta->extMsg;
        if (dsObj) {
          json_object_set_string_member (jobject, "type", dsObj->type);
          json_object_set_string_member (jobject, "make", dsObj->make);
          json_object_set_string_member (jobject, "model", dsObj->model);
          json_object_set_string_member (jobject, "color", dsObj->color);
          json_object_set_string_member (jobject, "licenseState", dsObj->region);
          json_object_set_string_member (jobject, "license", dsObj->license);
          json_object_set_double_member (jobject, "confidence", meta->confidence);

          objectMask = dsObj->mask;
        }
      } else {
        // No vehicle object in meta data. Attach empty vehicle sub object.
        json_object_set_string_member (jobject, "type", "");
        json_object_set_string_member (jobject, "make", "");
        json_object_set_string_member (jobject, "model", "");
        json_object_set_string_member (jobject, "color", "");
        json_object_set_string_member (jobject, "licenseState", "");
        json_object_set_string_member (jobject, "license", "");
        json_object_set_double_member (jobject, "confidence", 1.0);
      }
      json_object_set_object_member (objectObj, "vehicle", jobject);
      break;
    case NVDS_OBJECT_TYPE_PERSON_EXT:
      // person sub object
      jobject = json_object_new ();

      if (meta->extMsgSize) {
        NvDsPersonObjectExt *dsObj = (NvDsPersonObjectExt *) meta->extMsg;
        if (dsObj) {
          json_object_set_int_member (jobject, "age", dsObj->age);
          json_object_set_string_member (jobject, "gender", dsObj->gender);
          json_object_set_string_member (jobject, "hair", dsObj->hair);
          json_object_set_string_member (jobject, "cap", dsObj->cap);
          json_object_set_string_member (jobject, "apparel", dsObj->apparel);
          json_object_set_double_member (jobject, "confidence", meta->confidence);

          objectMask = dsObj->mask;
        }
      } else {
        // No person object in meta data. Attach empty person sub object.
        json_object_set_int_member (jobject, "age", 0);
        json_object_set_string_member (jobject, "gender", "");
        json_object_set_string_member (jobject, "hair", "");
        json_object_set_string_member (jobject, "cap", "");
        json_object_set_string_member (jobject, "apparel", "");
        json_object_set_double_member (jobject, "confidence", 1.0);
      }
      json_object_set_object_member (objectObj, "person", jobject);
      break;
    case NVDS_OBJECT_TYPE_FACE_EXT:
      // face sub object
      jobject = json_object_new ();

      if (meta->extMsgSize) {
        NvDsFaceObjectExt *dsObj = (NvDsFaceObjectExt *) meta->extMsg;
        if (dsObj) {
          json_object_set_int_member (jobject, "age", dsObj->age);
          json_object_set_string_member (jobject, "gender", dsObj->gender);
          json_object_set_string_member (jobject, "hair", dsObj->hair);
          json_object_set_string_member (jobject, "cap", dsObj->cap);
          json_object_set_string_member (jobject, "glasses", dsObj->glasses);
          json_object_set_string_member (jobject, "facialhair", dsObj->facialhair);
          json_object_set_string_member (jobject, "name", dsObj->name);
          json_object_set_string_member (jobject, "eyecolor", dsObj->eyecolor);
          json_object_set_double_member (jobject, "confidence", meta->confidence);

          objectMask = dsObj->mask;
        }
      } else {
        // No face object in meta data. Attach empty face sub object.
        json_object_set_int_member (jobject, "age", 0);
        json_object_set_string_member (jobject, "gender", "");
        json_object_set_string_member (jobject, "hair", "");
        json_object_set_string_member (jobject, "cap", "");
        json_object_set_string_member (jobject, "glasses", "");
        json_object_set_string_member (jobject, "facialhair", "");
        json_object_set_string_member (jobject, "name", "");
        json_object_set_string_member (jobject, "eyecolor", "");
        json_object_set_double_member (jobject, "confidence", 1.0);
      }
      json_object_set_object_member (objectObj, "face", jobject);
      break;
    case NVDS_OBJECT_TYPE_PRODUCT:
      jobject = json_object_new ();
      if(meta->extMsgSize){
        NvDsProductObject *dsObj = (NvDsProductObject *) meta->extMsg;
        if(dsObj){
          json_object_set_string_member(jobject, "brand", dsObj->brand);
          json_object_set_string_member(jobject, "type", dsObj->type);
          json_object_set_string_member(jobject, "shape", dsObj->shape);
        }
      }
      else{
        json_object_set_string_member(jobject, "brand", "");
        json_object_set_string_member(jobject, "type", "");
        json_object_set_string_member(jobject, "shape", "");
      }
      json_object_set_object_member(objectObj, "product", jobject);
      break;
    case NVDS_OBJECT_TYPE_PRODUCT_EXT:
      jobject = json_object_new ();
      if(meta->extMsgSize){
        NvDsProductObjectExt *dsObj = (NvDsProductObjectExt *) meta->extMsg;
        if(dsObj){
          json_object_set_string_member(jobject, "brand", dsObj->brand);
          json_object_set_string_member(jobject, "type", dsObj->type);
          json_object_set_string_member(jobject, "shape", dsObj->shape);
        }
      }
      else{
        json_object_set_string_member(jobject, "brand", "");
        json_object_set_string_member(jobject, "type", "");
        json_object_set_string_member(jobject, "shape", "");
      }
      json_object_set_object_member(objectObj, "product", jobject);
      break;
    case NVDS_OBJECT_TYPE_UNKNOWN:
      if(!meta->objectId) {
        break;
      }
      /** No information to add; object type unknown within NvDsEventMsgMeta */
      jobject = json_object_new ();
      json_object_set_object_member (objectObj, meta->objectId, jobject);
      break;
    default:
      cout << "Object type not implemented" << endl;
  }

  // bbox sub object
  jobject = json_object_new ();
  json_object_set_int_member (jobject, "topleftx", meta->bbox.left);
  json_object_set_int_member (jobject, "toplefty", meta->bbox.top);
  json_object_set_int_member (jobject, "bottomrightx", meta->bbox.left + meta->bbox.width);
  json_object_set_int_member (jobject, "bottomrighty", meta->bbox.top + meta->bbox.height);
  json_object_set_object_member (objectObj, "bbox", jobject);

  if (objectMask) {
    GList *l;
    JsonArray *maskArray = json_array_sized_new (g_list_length(objectMask));

    for (l = objectMask; l != NULL; l = l->next) {
      GArray *polygon = (GArray *) l->data;
      JsonArray *polygonArray = json_array_sized_new (polygon->len);

      for (i = 0; i < polygon->len; i++) {
        gdouble value = g_array_index (polygon, gdouble, i);

        json_array_add_double_element (polygonArray, value);
      }

      json_array_add_array_element (maskArray, polygonArray);
    }

    json_object_set_array_member (objectObj, "maskoutline", maskArray);
  }

  // signature sub array
  if (meta->objSignature.size) {
    JsonArray *jArray = json_array_sized_new (meta->objSignature.size);

    for (i = 0; i < meta->objSignature.size; i++) {
      json_array_add_double_element (jArray, meta->objSignature.signature[i]);
    }
    json_object_set_array_member (objectObj, "signature", jArray);
  }

  // location sub object
  jobject = json_object_new ();
  json_object_set_double_member (jobject, "lat", meta->location.lat);
  json_object_set_double_member (jobject, "lon", meta->location.lon);
  json_object_set_double_member (jobject, "alt", meta->location.alt);
  json_object_set_object_member (objectObj, "location", jobject);

  // coordinate sub object
  jobject = json_object_new ();
  json_object_set_double_member (jobject, "x", meta->coordinate.x);
  json_object_set_double_member (jobject, "y", meta->coordinate.y);
  json_object_set_double_member (jobject, "z", meta->coordinate.z);
  json_object_set_object_member (objectObj, "coordinate", jobject);

  JsonObject *pobject = json_object_new ();
  int joint_index = 0;

  for (joint_index = 0;joint_index < meta->pose.num_joints; joint_index++)
  {
    if (meta->pose.joints[joint_index].confidence > 0.0)
    {
      jobject = json_object_new ();
      std::string s =_joint_maps[meta->pose.pose_type][joint_index];
      char *json_name = const_cast<char*>(s.c_str());

      if (meta->pose.pose_type==0) {
        json_object_set_int_member (jobject, "x", meta->pose.joints[joint_index].x);
        json_object_set_int_member (jobject, "y", meta->pose.joints[joint_index].y);
        json_object_set_double_member (jobject, "confidence", meta->pose.joints[joint_index].confidence);
        json_object_set_object_member (pobject, reinterpret_cast<const gchar*>(json_name), jobject);
      }
      else if (meta->pose.pose_type==1 || meta->pose.pose_type==2) {
        json_object_set_int_member (jobject, "x", meta->pose.joints[joint_index].x);
        json_object_set_int_member (jobject, "y", meta->pose.joints[joint_index].y);
        json_object_set_int_member (jobject, "z", meta->pose.joints[joint_index].z);
        json_object_set_double_member (jobject, "confidence", meta->pose.joints[joint_index].confidence);
        json_object_set_object_member (pobject, reinterpret_cast<const gchar*>(json_name), jobject);
      }
    }
  }

  json_object_set_object_member (objectObj, "pose", pobject);

  //===Embedding model full schema data population===
  if (meta->embedding.embedding_vector && meta->embedding.embedding_length) {
    jobject = json_object_new ();
    json_object_set_int_member(jobject, "embedding_length", meta->embedding.embedding_length);
    JsonArray *embeddingArray = json_array_sized_new (meta->embedding.embedding_length);
    for (guint idx = 0; idx<meta->embedding.embedding_length; idx++){
      json_array_add_double_element (embeddingArray, meta->embedding.embedding_vector[idx]);
    }
    json_object_set_array_member(jobject, "embedding_vector", embeddingArray);
    json_object_set_object_member (objectObj, "embedding", jobject);
  }

  // Single-view 3D Tracking metadata
  if(meta->has3DTracking) {
    jobject = json_object_new ();
    json_object_set_double_member (jobject, "visibility", meta->singleView3DTracking.visibility);

    JsonArray *footLoc2DArray = json_array_sized_new (2);
    json_array_add_double_element (footLoc2DArray, meta->singleView3DTracking.ptImgFeet[0]);
    json_array_add_double_element (footLoc2DArray, meta->singleView3DTracking.ptImgFeet[1]);
    json_object_set_array_member(jobject, "footLocation2D", footLoc2DArray);

    JsonArray *footLoc3DArray = json_array_sized_new (2);
    json_array_add_double_element (footLoc2DArray, meta->singleView3DTracking.ptWorldFeet[0]);
    json_array_add_double_element (footLoc2DArray, meta->singleView3DTracking.ptWorldFeet[1]);
    json_object_set_array_member(jobject, "footLocation3D", footLoc3DArray);

    JsonArray *convexHullArray = json_array_sized_new (meta->singleView3DTracking.convexHull.numFilled * 2);
    for (unsigned int idx = 0; idx<meta->singleView3DTracking.convexHull.numFilled * 2; idx++){
      json_array_add_double_element (convexHullArray, meta->singleView3DTracking.convexHull.points[idx]);
    }
    json_object_set_array_member(jobject, "convexHull", convexHullArray);

    json_object_set_object_member (objectObj, "SV3DT", jobject);
  }

  return objectObj;
}

gchar* generate_event_message (void *privData, NvDsEventMsgMeta *meta)
{
  JsonNode *rootNode;
  JsonObject *rootObj;
  JsonObject *placeObj;
  JsonObject *sensorObj;
  JsonObject *analyticsObj;
  JsonObject *eventObj;
  JsonObject *objectObj;
  gchar *message;

  uuid_t msgId;
  gchar msgIdStr[37];

  uuid_generate_random (msgId);
  uuid_unparse_lower(msgId, msgIdStr);

  // place object
  placeObj = generate_place_object (privData, meta);

  // sensor object
  sensorObj = generate_sensor_object (privData, meta);

  // analytics object
  analyticsObj = generate_analytics_module_object (privData, meta);

  // object object
  objectObj = generate_object_object (privData, meta);

  // event object
  eventObj = generate_event_object (privData, meta);

  // root object
  rootObj = json_object_new ();
  json_object_set_string_member (rootObj, "messageid", msgIdStr);
  json_object_set_string_member (rootObj, "mdsversion", "1.0");
  json_object_set_string_member (rootObj, "@timestamp", meta->ts);
  json_object_set_object_member (rootObj, "place", placeObj);
  json_object_set_object_member (rootObj, "sensor", sensorObj);
  json_object_set_object_member (rootObj, "analyticsModule", analyticsObj);
  json_object_set_object_member (rootObj, "object", objectObj);
  json_object_set_object_member (rootObj, "event", eventObj);

  if (meta->videoPath)
    json_object_set_string_member (rootObj, "videoPath", meta->videoPath);
  else
    json_object_set_string_member (rootObj, "videoPath", "");

  rootNode = json_node_new (JSON_NODE_OBJECT);
  json_node_set_object (rootNode, rootObj);

  message = json_to_string (rootNode, TRUE);
  json_node_free (rootNode);
  json_object_unref (rootObj);

  return message;
}

static const gchar*
object_enum_to_str (NvDsObjectType type, gchar* objectId)
{
  switch (type) {
    case NVDS_OBJECT_TYPE_VEHICLE:
      return "Vehicle";
    case NVDS_OBJECT_TYPE_FACE:
      return "Face";
    case NVDS_OBJECT_TYPE_PERSON:
      return "Person";
    case NVDS_OBJECT_TYPE_BAG:
      return "Bag";
    case NVDS_OBJECT_TYPE_BICYCLE:
      return "Bicycle";
    case NVDS_OBJECT_TYPE_ROADSIGN:
      return "RoadSign";
    case NVDS_OBJECT_TYPE_CUSTOM:
      return "Custom";
    case NVDS_OBJECT_TYPE_PRODUCT:
      return "Product";
    case NVDS_OBJECT_TYPE_UNKNOWN:
      return objectId ? objectId : "Unknown";
    default:
      return "Unknown";
  }
}

static const gchar*
to_str (gchar* cstr)
{
    return reinterpret_cast<const gchar*>(cstr) ? cstr : "";
}

static const gchar *
sensor_id_to_str (void *privData, gint sensorId)
{
  NvDsPayloadPriv *privObj = NULL;
  NvDsSensorObject *dsObj = NULL;

  g_return_val_if_fail (privData, NULL);

  privObj = (NvDsPayloadPriv *) privData;

  auto idMap = privObj->sensorObj.find (sensorId);
  if (idMap != privObj->sensorObj.end()) {
    dsObj = &idMap->second;
    return dsObj->id.c_str();
  } else {
    cout << "No entry for " CONFIG_GROUP_SENSOR << sensorId
        << " in configuration file" << endl;
    return NULL;
  }
}

static void
generate_mask_array (NvDsEventMsgMeta *meta, JsonArray *jArray, GList *mask)
{
  unsigned int i;
  GList *l;
  stringstream ss;
  bool started = false;

  ss << meta->trackingId << "|" << g_list_length(mask);

  for (l = mask; l != NULL; l = l->next) {
    GArray *polygon = (GArray *) l->data;

    if (started)
      ss << "|#";

    started = true;

    for (i = 0; i < polygon->len; i++) {
      gdouble value = g_array_index (polygon, gdouble, i);
      ss << "|" << value;
    }
  }
  json_array_add_string_element (jArray, ss.str().c_str());
}

gchar* generate_event_message_minimal (void *privData, NvDsEvent *events, guint size)
{
  /*
  The JSON structure of the frame
  {
   "version": "4.0",
   "id": "frame-id",
   "@timestamp": "2018-04-11T04:59:59.828Z",
   "sensorId": "sensor-id",
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

  JsonNode *rootNode;
  JsonObject *jobject;
  JsonArray *jArray;
  JsonArray *maskArray = NULL;
  guint i;
  stringstream ss;
  gchar *message = NULL;

  jArray = json_array_new ();

  for (i = 0; i < size; i++) {
    GList *objectMask = NULL;

    ss.str("");
    ss.clear();

    NvDsEventMsgMeta *meta = events[i].metadata;
    if(meta->objType!=NVDS_OBJECT_TYPE_DUMMY){
     ss << meta->trackingId << "|" << meta->bbox.left << "|" << meta->bbox.top
        << "|" << meta->bbox.left + meta->bbox.width << "|" << meta->bbox.top + meta->bbox.height
        << "|" << object_enum_to_str (meta->objType, meta->objectId);
     if (meta->extMsg && meta->extMsgSize) {
      // Attach secondary inference attributes.
      switch (meta->objType) {
        case NVDS_OBJECT_TYPE_VEHICLE: {
          NvDsVehicleObject *dsObj = (NvDsVehicleObject *) meta->extMsg;
          if (dsObj) {
            ss << "|#|" << to_str(dsObj->type) << "|" << to_str(dsObj->make) << "|"
               << to_str(dsObj->model) << "|" << to_str(dsObj->color) << "|" << to_str(dsObj->license)
               << "|" << to_str(dsObj->region);
          }
        }
          break;
        case NVDS_OBJECT_TYPE_PERSON: {
          NvDsPersonObject *dsObj = (NvDsPersonObject *) meta->extMsg;
          if (dsObj) {
            ss << "|#|" << to_str(dsObj->gender) << "|" << dsObj->age << "|"
                << to_str(dsObj->hair) << "|" << to_str(dsObj->cap) << "|" << to_str(dsObj->apparel);
            //===Adding pose data to stream for person object types===
            int joint_index = 0;
            if(meta->pose.num_joints){
              if (meta->pose.pose_type==0) {
                ss << "|#|pose2D|";
                for (joint_index = 0;joint_index < meta->pose.num_joints; joint_index++)
                {
                    std::string s = _joint_maps[meta->pose.pose_type][joint_index]
                                      + "," + to_string(meta->pose.joints[joint_index].x)
                                      + "," + to_string(meta->pose.joints[joint_index].y)
                                      + "," + to_string(meta->pose.joints[joint_index].confidence);
                    ss << s << "|";
                }
              }
              else if (meta->pose.pose_type==1) {
                ss << "|#|pose25D|";
                for (joint_index = 0;joint_index < meta->pose.num_joints; joint_index++)
                {
                    std::string s = _joint_maps[meta->pose.pose_type][joint_index]
                                      + "," + to_string(meta->pose.joints[joint_index].x)
                                      + "," + to_string(meta->pose.joints[joint_index].y)
                                      + "," + to_string(meta->pose.joints[joint_index].z)
                                      + "," + to_string(meta->pose.joints[joint_index].confidence);
                    ss << s << "|";
                }
              }
              else if (meta->pose.pose_type==2) {
                ss << "|#|pose3D|";
                for (joint_index = 0;joint_index < meta->pose.num_joints; joint_index++)
                {
                    std::string s = _joint_maps[meta->pose.pose_type][joint_index]
                                      + "," + to_string(meta->pose.joints[joint_index].x)
                                      + "," + to_string(meta->pose.joints[joint_index].y)
                                      + "," + to_string(meta->pose.joints[joint_index].z)
                                      + "," + to_string(meta->pose.joints[joint_index].confidence);
                    ss << s << "|";
                }
              }
            }
          }
        }
          break;
        case NVDS_OBJECT_TYPE_FACE: {
          NvDsFaceObject *dsObj = (NvDsFaceObject *) meta->extMsg;
          if (dsObj) {
            ss << "|#|" << to_str(dsObj->gender) << "|" << dsObj->age << "|"
                << to_str(dsObj->hair) << "|" << to_str(dsObj->cap) << "|" << to_str(dsObj->glasses)
                << "|" << to_str(dsObj->facialhair) << "|" << to_str(dsObj->name) << "|"
                << "|" << to_str(dsObj->eyecolor);
          }
        }
          break;
        case NVDS_OBJECT_TYPE_VEHICLE_EXT: {
          NvDsVehicleObjectExt *dsObj = (NvDsVehicleObjectExt *) meta->extMsg;
          if (dsObj) {
            ss << "|#|" << to_str(dsObj->type) << "|" << to_str(dsObj->make) << "|"
               << to_str(dsObj->model) << "|" << to_str(dsObj->color) << "|" << to_str(dsObj->license)
               << "|" << to_str(dsObj->region);

            if (dsObj->mask)
              objectMask = dsObj->mask;
          }
        }
          break;
        case NVDS_OBJECT_TYPE_PERSON_EXT: {
          NvDsPersonObjectExt *dsObj = (NvDsPersonObjectExt *) meta->extMsg;
          if (dsObj) {
            ss << "|#|" << to_str(dsObj->gender) << "|" << dsObj->age << "|"
                << to_str(dsObj->hair) << "|" << to_str(dsObj->cap) << "|" << to_str(dsObj->apparel);

            if (dsObj->mask)
              objectMask = dsObj->mask;
          }
        }
          break;
        case NVDS_OBJECT_TYPE_FACE_EXT: {
          NvDsFaceObjectExt *dsObj = (NvDsFaceObjectExt *) meta->extMsg;
          if (dsObj) {
            ss << "|#|" << to_str(dsObj->gender) << "|" << dsObj->age << "|"
                << to_str(dsObj->hair) << "|" << to_str(dsObj->cap) << "|" << to_str(dsObj->glasses)
                << "|" << to_str(dsObj->facialhair) << "|" << to_str(dsObj->name) << "|"
                << "|" << to_str(dsObj->eyecolor);

            if (dsObj->mask)
              objectMask = dsObj->mask;
          }
        }
          break;
        //===Product object to stream for minimal schema===
        case NVDS_OBJECT_TYPE_PRODUCT: {
          NvDsProductObject *dsObj = (NvDsProductObject *) meta->extMsg;
          if(dsObj) {
            ss << "|#|" << to_str(dsObj->brand) << "|" << to_str(dsObj->type) << "|"
                << to_str(dsObj->shape);
          }
        }
          break;
        case NVDS_OBJECT_TYPE_PRODUCT_EXT:{
          NvDsProductObjectExt *dsObj = (NvDsProductObjectExt *) meta->extMsg;
          if(dsObj) {
            ss << "|#|" << to_str(dsObj->brand) << "|" << to_str(dsObj->type) << "|"
                << to_str(dsObj->shape);
            if (dsObj->mask)
              objectMask = dsObj->mask;
          }
        }
          break;
        default:
          cout << "Object type (" << meta->objType << ") not implemented" << endl;
          break;
     }
    }
    else {
      if(meta->objType!=NVDS_OBJECT_TYPE_DUMMY){
        ss << "|#||||||" ;
      }
    }
   }
   if(meta->objType!=NVDS_OBJECT_TYPE_DUMMY){
    ss << "|" << meta->confidence;
   }

    //===Printing embedding model information to stream===
    if(meta->embedding.embedding_vector && meta->embedding.embedding_length) {
      ss << "|#|embedding|";
      for (unsigned int idx = 0; idx<meta->embedding.embedding_length; idx++){
        if(idx==meta->embedding.embedding_length-1){
          ss << (float)meta->embedding.embedding_vector[idx];
        }
        else{
          ss << (float)meta->embedding.embedding_vector[idx] << ",";
        }
      }
      ss << "|";
    }

    // Single-view 3D Tracking metadata
    if(meta->has3DTracking) {
      ss << "#|SV3DT|" << to_string(meta->singleView3DTracking.visibility) << "|"
        << to_string(meta->singleView3DTracking.ptImgFeet[0]) << "," << to_string(meta->singleView3DTracking.ptImgFeet[1]) << "|"
        << to_string(meta->singleView3DTracking.ptWorldFeet[0]) << "," << to_string(meta->singleView3DTracking.ptWorldFeet[1]) << "|";

      for (unsigned int idx = 0; idx<meta->singleView3DTracking.convexHull.numFilled * 2; idx++){
        if(idx==meta->singleView3DTracking.convexHull.numFilled * 2 - 1){
          ss << to_string(meta->singleView3DTracking.convexHull.points[idx]);
        }
        else{
          ss << to_string(meta->singleView3DTracking.convexHull.points[idx]) << ",";
        }
      }
      ss << "|";
    }

    if (objectMask) {
      if (maskArray == NULL)
        maskArray = json_array_new ();
      generate_mask_array (meta, maskArray, objectMask);
    }

    json_array_add_string_element (jArray, ss.str().c_str());
  }

  // It is assumed that all events / objects are associated with same frame.
  // Therefore ts / sensorId / frameId of first object can be used.

  jobject = json_object_new ();
  json_object_set_string_member (jobject, "version", "4.0");
  json_object_set_string_member (jobject, "id", to_string(events[0].metadata->frameId).c_str());
  json_object_set_string_member (jobject, "@timestamp", events[0].metadata->ts);
  if (events[0].metadata->sensorStr) {
    json_object_set_string_member (jobject, "sensorId", events[0].metadata->sensorStr);
  } else if ((NvDsPayloadPriv *) privData) {
    json_object_set_string_member (jobject, "sensorId",
        to_str((gchar *) sensor_id_to_str (privData, events[0].metadata->sensorId)));
  } else {
    json_object_set_string_member (jobject, "sensorId", "0");
  }

  json_object_set_array_member (jobject, "objects", jArray);
  if (maskArray && json_array_get_length (maskArray) > 0)
    json_object_set_array_member (jobject, "masks", maskArray);

  rootNode = json_node_new (JSON_NODE_OBJECT);
  json_node_set_object (rootNode, jobject);

  message = json_to_string (rootNode, TRUE);
  json_node_free (rootNode);
  json_object_unref (jobject);
  return message;
}

gchar* generate_event_message_protobuf (void *privData, NvDsEvent *events, guint size, size_t &message_len)
{
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  nv::Frame pbFrame;

  pbFrame.set_version("4.0");
  pbFrame.set_id(std::to_string(events[0].metadata->frameId));

  if (events[0].metadata->ts) {
    std::string ts_string(events[0].metadata->ts);
    google::protobuf::Timestamp *timestamp = pbFrame.mutable_timestamp();
    if (!::google::protobuf::util::TimeUtil::FromString(ts_string, timestamp)) {
      message_len = 0;
      return NULL;
    }
  }

  if (events[0].metadata->sensorStr) {
    pbFrame.set_sensorid(events[0].metadata->sensorStr);
  }
  else if ((NvDsPayloadPriv *) privData) {
    pbFrame.set_sensorid(to_str((gchar *) sensor_id_to_str (privData, events[0].metadata->sensorId)));
  }
  else {
    pbFrame.set_sensorid("0");
  }

  // objects
  for (guint i = 0; i < size; i++) {
    NvDsEventMsgMeta *meta = events[i].metadata;

   if(meta->objType!=NVDS_OBJECT_TYPE_DUMMY){
    nv::Object *object = pbFrame.add_objects();
    object->set_id(std::to_string(meta->trackingId));

    nv::Bbox *bbox = object->mutable_bbox();
    bbox->set_leftx(meta->bbox.left);
    bbox->set_topy(meta->bbox.top);
    bbox->set_rightx(meta->bbox.left + meta->bbox.width);
    bbox->set_bottomy(meta->bbox.top + meta->bbox.height);

    object->set_type(object_enum_to_str (meta->objType, meta->objectId));
    object->set_confidence(meta->confidence);

    // nv::Info
    auto *info = object->mutable_info();
    if (meta->extMsg && meta->extMsgSize) {
      // Attach secondary inference attributes.
      switch (meta->objType) {
        case NVDS_OBJECT_TYPE_VEHICLE:
        case NVDS_OBJECT_TYPE_VEHICLE_EXT: {
          NvDsVehicleObject *dsObj = (NvDsVehicleObject *) meta->extMsg;

          (*info)["type"] = to_str(dsObj->type);
          (*info)["make"] = to_str(dsObj->make);
          (*info)["model"] = to_str(dsObj->model);
          (*info)["color"] = to_str(dsObj->color);
          (*info)["licenseState"] = to_str(dsObj->region);
          (*info)["license"] = to_str(dsObj->license);
        }
          break;
        case NVDS_OBJECT_TYPE_PERSON:
        case NVDS_OBJECT_TYPE_PERSON_EXT: {
          NvDsPersonObject *dsObj = (NvDsPersonObject *) meta->extMsg;

          (*info)["age"] = std::to_string(dsObj->age);
          (*info)["gender"] = to_str(dsObj->gender);
          (*info)["hair"] = to_str(dsObj->hair);
          (*info)["cap"] = to_str(dsObj->cap);
          (*info)["apparel"] = to_str(dsObj->apparel);

          //===Adding pose data to stream for person object types===
          if(meta->pose.num_joints){
            nv::Pose *pose = object->mutable_pose();

            int joint_index = 0;
            if (meta->pose.pose_type == 0) {
              pose->set_type("pose2D");

              for (joint_index = 0; joint_index < meta->pose.num_joints; joint_index++) {
                nv::Pose_Keypoint *keypoint = pose->add_keypoints();

                keypoint->set_name(_joint_maps[meta->pose.pose_type][joint_index]);
                keypoint->add_coordinates(meta->pose.joints[joint_index].x);
                keypoint->add_coordinates(meta->pose.joints[joint_index].y);
                keypoint->add_coordinates(meta->pose.joints[joint_index].confidence);
              }
            }
            else if (meta->pose.pose_type==1) {
              pose->set_type("pose25D");

              for (joint_index = 0; joint_index < meta->pose.num_joints; joint_index++) {
                nv::Pose_Keypoint *keypoint = pose->add_keypoints();

                keypoint->set_name(_joint_maps[meta->pose.pose_type][joint_index]);
                keypoint->add_coordinates(meta->pose.joints[joint_index].x);
                keypoint->add_coordinates(meta->pose.joints[joint_index].y);
                keypoint->add_coordinates(meta->pose.joints[joint_index].z);
                keypoint->add_coordinates(meta->pose.joints[joint_index].confidence);
              }
            }
            else if (meta->pose.pose_type==2) {
              pose->set_type("pose3D");

              for (joint_index = 0; joint_index < meta->pose.num_joints; joint_index++) {
                nv::Pose_Keypoint *keypoint = pose->add_keypoints();

                keypoint->set_name(_joint_maps[meta->pose.pose_type][joint_index]);
                keypoint->add_coordinates(meta->pose.joints[joint_index].x);
                keypoint->add_coordinates(meta->pose.joints[joint_index].y);
                keypoint->add_coordinates(meta->pose.joints[joint_index].z);
                keypoint->add_coordinates(meta->pose.joints[joint_index].confidence);
              }
            }
          }
          //===Adding pose data to stream for person object types===
        }
          break;
        case NVDS_OBJECT_TYPE_FACE:
        case NVDS_OBJECT_TYPE_FACE_EXT: {
          NvDsFaceObject *dsObj = (NvDsFaceObject *) meta->extMsg;

          (*info)["age"] = std::to_string(dsObj->age);
          (*info)["gender"] = to_str(dsObj->gender);
          (*info)["hair"] = to_str(dsObj->hair);
          (*info)["cap"] = to_str(dsObj->cap);
          (*info)["glasses"] = to_str(dsObj->glasses);
          (*info)["facialhair"] = to_str(dsObj->facialhair);
          (*info)["name"] = to_str(dsObj->name);
          (*info)["eyecolor"] = to_str(dsObj->eyecolor);
        }
          break;
        //===Product object to stream for minimal schema===
        case NVDS_OBJECT_TYPE_PRODUCT:
        case NVDS_OBJECT_TYPE_PRODUCT_EXT:{
          NvDsProductObject *dsObj = (NvDsProductObject *) meta->extMsg;

          (*info)["brand"] = to_str(dsObj->brand);
          (*info)["type"] = to_str(dsObj->type);
          (*info)["shape"] = to_str(dsObj->shape);
        }
          break;
        default:
          cout << "Object type (" << meta->objType << ") not implemented" << endl;
          break;
      }
    }

    if((meta->embedding.embedding_vector != NULL && meta->embedding.embedding_length)) {
      nv::Embedding *embedding = object->mutable_embedding();
      for (unsigned idx = 0; idx<meta->embedding.embedding_length; idx++){
        embedding->add_vector((float)meta->embedding.embedding_vector[idx]);
      }
    }

    // Single-view 3D Tracking metadata
    if(meta->has3DTracking) {
      auto *info = object->mutable_info();
      (*info)["visibility"] = std::to_string(meta->singleView3DTracking.visibility);
      (*info)["footLocation2D"] = to_string(meta->singleView3DTracking.ptImgFeet[0]) + "," + to_string(meta->singleView3DTracking.ptImgFeet[1]);
      (*info)["footLocation3D"] = to_string(meta->singleView3DTracking.ptWorldFeet[0]) + "," + to_string(meta->singleView3DTracking.ptWorldFeet[1]);
      
      stringstream ss;
      ss.str("");
      ss.clear();

      for (unsigned int idx = 0; idx<meta->singleView3DTracking.convexHull.numFilled * 2; idx++){
        if(idx==meta->singleView3DTracking.convexHull.numFilled * 2 - 1){
          ss << to_string(meta->singleView3DTracking.convexHull.points[idx]);
        }
        else{
          ss << to_string(meta->singleView3DTracking.convexHull.points[idx]) << ",";
        }
      }

      (*info)["convexHull"] = ss.str();
    }
   }
  }
  std::string msg_str;
  if (!pbFrame.SerializeToString(&msg_str)) {
    cout<<"generate_event_message_protobuf : Failed to serialize protobuf message to string.\n";
    message_len = 0;
    return NULL;
  }

  message_len = msg_str.length();
  // Save the content of msg_str before the function returns which puts msg_str out of scope.
  gchar *message = (gchar *)g_memdup2(msg_str.c_str(), message_len);
  return message;
}
