/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "metadata.hpp"

#include "gstnvdsmeta.h"
#include "nvdsmeta_schema.h"

#include <map>

#define MAX_TIME_STAMP_LEN 32

#define MAX_DISPLAY_LEN 64
#define PGIE_CLASS_ID_VEHICLE 0
#define PGIE_CLASS_ID_PERSON 2
#define SECONDARY_GIE_VEHICLE_TYPE_UNIQUE_ID 4
#define SECONDARY_GIE_VEHICLE_COLOR_UNIQUE_ID 5
#define SECONDARY_GIE_VEHICLE_MAKE_UNIQUE_ID 6

namespace deepstream {

static gpointer
meta_copy_func (gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
  NvDsEventMsgMeta *srcMeta = (NvDsEventMsgMeta *) user_meta->user_meta_data;
  NvDsEventMsgMeta *dstMeta = NULL;

  dstMeta = (NvDsEventMsgMeta*) g_memdup ((gpointer)srcMeta, sizeof (NvDsEventMsgMeta));

  if (srcMeta->ts)
    dstMeta->ts = g_strdup (srcMeta->ts);

  if (srcMeta->sensorStr)
    dstMeta->sensorStr = g_strdup (srcMeta->sensorStr);

  if (srcMeta->objSignature.size > 0) {
    dstMeta->objSignature.signature = (gdouble *) g_memdup (
        (gpointer)srcMeta->objSignature.signature,
        srcMeta->objSignature.size);
    dstMeta->objSignature.size = srcMeta->objSignature.size;
  }

  if (srcMeta->objectId) {
    dstMeta->objectId = g_strdup (srcMeta->objectId);
  }

  if (srcMeta->extMsgSize > 0) {
    if (srcMeta->objType == NVDS_OBJECT_TYPE_VEHICLE) {
      NvDsVehicleObject *srcObj = (NvDsVehicleObject *) srcMeta->extMsg;
      NvDsVehicleObject *obj =
          (NvDsVehicleObject *) g_malloc0 (sizeof (NvDsVehicleObject));
      if (srcObj->type)
        obj->type = g_strdup (srcObj->type);
      if (srcObj->make)
        obj->make = g_strdup (srcObj->make);
      if (srcObj->model)
        obj->model = g_strdup (srcObj->model);
      if (srcObj->color)
        obj->color = g_strdup (srcObj->color);
      if (srcObj->license)
        obj->license = g_strdup (srcObj->license);
      if (srcObj->region)
        obj->region = g_strdup (srcObj->region);

      dstMeta->extMsg = obj;
      dstMeta->extMsgSize = sizeof (NvDsVehicleObject);
    } else if (srcMeta->objType == NVDS_OBJECT_TYPE_PERSON) {
      NvDsPersonObject *srcObj = (NvDsPersonObject *) srcMeta->extMsg;
      NvDsPersonObject *obj =
          (NvDsPersonObject *) g_malloc0 (sizeof (NvDsPersonObject));

      obj->age = srcObj->age;

      if (srcObj->gender)
        obj->gender = g_strdup (srcObj->gender);
      if (srcObj->cap)
        obj->cap = g_strdup (srcObj->cap);
      if (srcObj->hair)
        obj->hair = g_strdup (srcObj->hair);
      if (srcObj->apparel)
        obj->apparel = g_strdup (srcObj->apparel);
      dstMeta->extMsg = obj;
      dstMeta->extMsgSize = sizeof (NvDsPersonObject);
    }
  }

  return dstMeta;
}

static void
meta_free_func (gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
  NvDsEventMsgMeta *srcMeta = (NvDsEventMsgMeta *) user_meta->user_meta_data;

  g_free (srcMeta->ts);
  g_free (srcMeta->sensorStr);

  if (srcMeta->objSignature.size > 0) {
    g_free (srcMeta->objSignature.signature);
    srcMeta->objSignature.size = 0;
  }

  if (srcMeta->objectId) {
    g_free (srcMeta->objectId);
  }

  if (srcMeta->extMsgSize > 0) {
    if (srcMeta->objType == NVDS_OBJECT_TYPE_VEHICLE) {
      NvDsVehicleObject *obj = (NvDsVehicleObject *) srcMeta->extMsg;
      if (obj->type)
        g_free (obj->type);
      if (obj->color)
        g_free (obj->color);
      if (obj->make)
        g_free (obj->make);
      if (obj->model)
        g_free (obj->model);
      if (obj->license)
        g_free (obj->license);
      if (obj->region)
        g_free (obj->region);
    } else if (srcMeta->objType == NVDS_OBJECT_TYPE_PERSON) {
      NvDsPersonObject *obj = (NvDsPersonObject *) srcMeta->extMsg;

      if (obj->gender)
        g_free (obj->gender);
      if (obj->cap)
        g_free (obj->cap);
      if (obj->hair)
        g_free (obj->hair);
      if (obj->apparel)
        g_free (obj->apparel);
    }
    g_free (srcMeta->extMsg);
    srcMeta->extMsgSize = 0;
  }
  g_free (user_meta->user_meta_data);
  user_meta->user_meta_data = NULL;
}

#ifdef GENERATE_DUMMY_META_EXT
static void
generate_person_meta (gpointer data)
{
  NvDsPersonObject *obj = (NvDsPersonObject *) data;
  obj->age = 45;
  obj->cap = g_strdup ("none");
  obj->hair = g_strdup ("black");
  obj->gender = g_strdup ("male");
  obj->apparel = g_strdup ("formal");
}
#endif

static void
generate_ts_rfc3339 (char *buf, int buf_size)
{
  time_t tloc;
  struct tm tm_log;
  struct timespec ts;
  char strmsec[6];              //.nnnZ\0

  clock_gettime (CLOCK_REALTIME, &ts);
  memcpy (&tloc, (void *) (&ts.tv_sec), sizeof (time_t));
  gmtime_r (&tloc, &tm_log);
  strftime (buf, buf_size, "%Y-%m-%dT%H:%M:%S", &tm_log);
  int ms = ts.tv_nsec / 1000000;
  g_snprintf (strmsec, sizeof (strmsec), ".%.3dZ", ms);
  strncat (buf, strmsec, buf_size);
}

static void
generate_ts_rfc3339_from_ts (char *buf, int buf_size, GstClockTime ts)
{
  time_t tloc;
  struct tm tm_log;
  char strmsec[6];
  int ms;

  /** ts itself is UTC Time in ns */
  struct timespec timespec_current;
  GST_TIME_TO_TIMESPEC (ts, timespec_current);
  memcpy (&tloc, (void *) (&timespec_current.tv_sec), sizeof (time_t));
  ms = timespec_current.tv_nsec / 1000000;
  gmtime_r (&tloc, &tm_log);
  strftime (buf, buf_size, "%Y-%m-%dT%H:%M:%S", &tm_log);
  g_snprintf (strmsec, sizeof (strmsec), ".%.3dZ", ms);
  strncat (buf, strmsec, buf_size);
}

void EventMessageUserMetadata::generate(
  const ObjectMetadata& object,
  const FrameMetadata& frame,
  const std::string sensor,
  const std::string uri,
  const std::vector<std::string> labels) {
  NvDsUserMeta* user_meta = (NvDsUserMeta*) data_;
  NvDsEventMsgMeta* msg_meta = (NvDsEventMsgMeta*) user_meta->user_meta_data;
  float scaleW = 0;
  float scaleH = 0;
  std::map<std::string, unsigned int> class_id_map = {
    {"car", PGIE_CLASS_ID_VEHICLE}, 
    {"person", PGIE_CLASS_ID_PERSON}
  };

  object.iterate([msg_meta](const UserMetadata& user_meta) {
    msg_meta->has3DTracking = true;
    Object3DBBoxUserMetadata obj3d_meta(user_meta);
    NvDsObj3DBbox* p3DBbox = obj3d_meta.get3DBbox();
    msg_meta->singleView3DTracking.bbox3d.boxes_3d[0] = p3DBbox->xCentre;
    msg_meta->singleView3DTracking.bbox3d.boxes_3d[1] = p3DBbox->yCentre;
    msg_meta->singleView3DTracking.bbox3d.boxes_3d[2] = p3DBbox->zCentre;
    msg_meta->singleView3DTracking.bbox3d.boxes_3d[3] = p3DBbox->xLen;
    msg_meta->singleView3DTracking.bbox3d.boxes_3d[4] = p3DBbox->yLen;
    msg_meta->singleView3DTracking.bbox3d.boxes_3d[5] = p3DBbox->zLen;
  }, NVDS_OBJ_3D_META);

  /** override class ids for custom model*/
  if (!labels.empty()) {
    class_id_map.clear();
    for (unsigned int i = 0; i < labels.size(); i++) {
      class_id_map[labels[i]] = i;
    }
  }


  scaleW = (float)frame.sourceWidth() / frame.pipelineWidth();
  scaleH = (float)frame.sourceHeight() / frame.pipelineHeight();

  auto rect_params = object.rectParams();
  msg_meta->bbox.top = rect_params.top * scaleH;
  msg_meta->bbox.left = rect_params.left * scaleW;
  msg_meta->bbox.width = rect_params.width * scaleW;
  msg_meta->bbox.height = rect_params.height * scaleH;
  msg_meta->frameId = frame.frameNum();
  msg_meta->trackingId = object.objectId();
  msg_meta->confidence = object.confidence();

  msg_meta->sensorId = frame.sourceId();
  msg_meta->placeId = frame.sourceId();
  msg_meta->moduleId = frame.sourceId();
  msg_meta->sensorStr = g_strdup (sensor.c_str());

  msg_meta->objectId = (gchar *)g_malloc0(MAX_LABEL_SIZE + 1);
  strncpy (msg_meta->objectId, object.label().c_str(), MAX_LABEL_SIZE);

  msg_meta->ts = (gchar *) g_malloc0 (MAX_TIME_STAMP_LEN + 1);
  if (!uri.empty() && uri.rfind("ipc://", 0) == 0) {
    generate_ts_rfc3339_from_ts (msg_meta->ts, MAX_TIME_STAMP_LEN, frame.bufferPTS ());
  }
  else {
    generate_ts_rfc3339 (msg_meta->ts, MAX_TIME_STAMP_LEN);
  }

  /*
   * This demonstrates how to attach custom objects.
   * Any custom object as per requirement can be generated and attached
   * like NvDsVehicleObject / NvDsPersonObject. Then that object should
   * be handled in payload generator library (nvmsgconv.cpp) accordingly.
   */
  if (class_id_map.find("car") != class_id_map.end() && object.classId() == class_id_map["car"]) {
    msg_meta->type = NVDS_EVENT_MOVING;
    msg_meta->objType = NVDS_OBJECT_TYPE_VEHICLE;
    msg_meta->objClassId = class_id_map["car"];

    NvDsVehicleObject *obj =
        (NvDsVehicleObject *) g_malloc0 (sizeof (NvDsVehicleObject));

    obj->type = NULL;
    obj->make = NULL;
    obj->model = NULL;
    obj->color = NULL;
    obj->license = NULL;
    obj->region = NULL;

    ClassifierMetadata::Iterator classifier_itr;
    for (object.initiateIterator(classifier_itr); !classifier_itr->done(); classifier_itr->next())
    {
      switch ((*classifier_itr)->uniqueComponentId())
      {
      case SECONDARY_GIE_VEHICLE_TYPE_UNIQUE_ID:
        obj->type = g_strdup((*classifier_itr)->getLabel(0).c_str());
        break;
      case SECONDARY_GIE_VEHICLE_COLOR_UNIQUE_ID:
        obj->color = g_strdup((*classifier_itr)->getLabel(0).c_str());
        break;
      case SECONDARY_GIE_VEHICLE_MAKE_UNIQUE_ID:
        obj->make = g_strdup((*classifier_itr)->getLabel(0).c_str());
        break;
      default:
        break;
      }
    }

    msg_meta->extMsg = obj;
    msg_meta->extMsgSize = sizeof (NvDsVehicleObject);
  } else if (class_id_map.find("person") != class_id_map.end() && object.classId() == class_id_map["person"]) {
    msg_meta->type = NVDS_EVENT_ENTRY;
    msg_meta->objType = NVDS_OBJECT_TYPE_PERSON;
    msg_meta->objClassId = class_id_map["person"];

#ifdef GENERATE_DUMMY_META_EXT
    NvDsPersonObject *obj =
        (NvDsPersonObject *) g_malloc0 (sizeof (NvDsPersonObject));
    generate_person_meta (obj);
    msg_meta->extMsg = obj;
    msg_meta->extMsgSize = sizeof (NvDsPersonObject);
#else
    msg_meta->extMsg = NULL;
    msg_meta->extMsgSize = 0;
#endif


  }
  else if (class_id_map.find("face") != class_id_map.end() && object.classId() == class_id_map["face"]) {
    msg_meta->type = NVDS_EVENT_ENTRY;
    msg_meta->objType = NVDS_OBJECT_TYPE_FACE;
    msg_meta->objClassId = class_id_map["face"];
    msg_meta->extMsg = NULL;
    msg_meta->extMsgSize = 0;
  }
  else if (class_id_map.find("bag") != class_id_map.end() && object.classId() == class_id_map["bag"]) {
    msg_meta->type = NVDS_EVENT_ENTRY;
    msg_meta->objType = NVDS_OBJECT_TYPE_BAG;
    msg_meta->objClassId = class_id_map["bag"];
    msg_meta->extMsg = NULL;
    msg_meta->extMsgSize = 0;
  }
  else if (class_id_map.find("bg") != class_id_map.end() && object.classId() == class_id_map["bg"]) {
    msg_meta->type = NVDS_EVENT_ENTRY;
    msg_meta->objType = NVDS_OBJECT_TYPE_BAG;
    msg_meta->objClassId = class_id_map["bag"];
    msg_meta->extMsg = NULL;
    msg_meta->extMsgSize = 0;
  }
}

void EventMessageUserMetadata::generate(
  const AudioFrameMetadata& frame,
  const std::string sensor,
  const std::string uri) {
  NvDsUserMeta* user_meta = (NvDsUserMeta*) data_;
  NvDsEventMsgMeta* msg_meta = (NvDsEventMsgMeta*) user_meta->user_meta_data;

  std::string label = frame.classLabel();
  if (label.empty()) {
    frame.iterate([&label](const ClassifierMetadata& classifier) {
      if (label.empty() && classifier.nLabels() > 0) {
        label = classifier.getLabel(0);
      }
    });
  }

  msg_meta->type = static_cast<NvDsEventType>(frame.classId());
  msg_meta->objType = NVDS_OBJECT_TYPE_DUMMY;
  msg_meta->objClassId = frame.classId();
  msg_meta->frameId = frame.frameNum();
  msg_meta->confidence = frame.confidence();
  msg_meta->sensorId = frame.sourceId();
  msg_meta->placeId = frame.sourceId();
  msg_meta->moduleId = frame.sourceId();
  msg_meta->sensorStr = g_strdup(sensor.c_str());
  msg_meta->objectId = g_strdup(label.c_str());
  msg_meta->extMsg = NULL;
  msg_meta->extMsgSize = 0;

  msg_meta->ts = (gchar *) g_malloc0(MAX_TIME_STAMP_LEN + 1);
  if (!uri.empty() && uri.rfind("ipc://", 0) == 0) {
    generate_ts_rfc3339_from_ts(msg_meta->ts, MAX_TIME_STAMP_LEN, frame.bufferPTS());
  } else if (frame.ntpTimestamp() != 0) {
    generate_ts_rfc3339_from_ts(msg_meta->ts, MAX_TIME_STAMP_LEN, frame.ntpTimestamp());
  } else {
    generate_ts_rfc3339(msg_meta->ts, MAX_TIME_STAMP_LEN);
  }
}

EventMessageUserMetadata::EventMessageUserMetadata(void* data)
: UserMetadata(data) {
  if (data_) {
    NvDsUserMeta* user_meta = (NvDsUserMeta*) data_;
    user_meta->user_meta_data = g_malloc0 (sizeof (NvDsEventMsgMeta));
    user_meta->base_meta.meta_type = NVDS_EVENT_MSG_META;
    user_meta->base_meta.copy_func =
        (NvDsMetaCopyFunc) meta_copy_func;
    user_meta->base_meta.release_func =
        (NvDsMetaReleaseFunc) meta_free_func;
  }
}

EventMessageUserMetadata::~EventMessageUserMetadata()
{}
}