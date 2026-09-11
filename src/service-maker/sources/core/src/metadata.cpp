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

#include "nvdsmeta.h"
#include "nvds_audio_meta.h"
#include "nvds_analytics_meta.h"
#include "nvdspreprocess_meta.h"
#include "metadata.hpp"
#include "tensor.hpp"
#include <gst/gst.h>
#include "gstnvdsinfer.h"

namespace deepstream {

/**
 * @brief Release the custom preprocess batch meta
 *
 * @param data
 * @param user_data
 */
static void
release_preprocess_batch_meta(gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
  GstNvDsPreProcessBatchMeta *preprocess_batchmeta = (GstNvDsPreProcessBatchMeta *) user_meta->user_meta_data;
  NvDsPreProcessTensorMeta *tensor_meta = preprocess_batchmeta->tensor_meta;
  if (tensor_meta) {
    if (tensor_meta->private_data) {
      delete (Tensor*)(tensor_meta->private_data);
    }
    delete tensor_meta;
  }

  delete preprocess_batchmeta;
}

template<typename T, typename S>
class IteratorImpl : public AbstractIterator<T> {
 public:
  IteratorImpl(GList* list)
  :AbstractIterator<T>(list?T((S*)list->data):T(NULL)), p_(list) {}
  virtual ~IteratorImpl() {}

  virtual AbstractIterator<T>& next() {
    p_ = p_->next;
    this->data_= p_?T((S*)p_->data):T(NULL);
    return *this;
  }

  virtual bool done() {
    return p_ == NULL;
  }

  bool equals(const AbstractIterator<T>& other) const override {
    const IteratorImpl<T, S> &itr = dynamic_cast<const IteratorImpl<T, S>&>(other);
    return p_ == itr.p_;
  }

 protected:
  GList *p_;
};

class UsermetaIteratorImpl : public IteratorImpl<UserMetadata, NvDsUserMeta> {
 public:
  UsermetaIteratorImpl(GList* list, int meta_type) : IteratorImpl<UserMetadata, NvDsUserMeta>(nullptr), meta_type_(meta_type)
  {
    // Skip to first element matching meta_type
    p_ = list;
    int idx = 0;
    while (p_) {
      NvDsUserMeta* data = (NvDsUserMeta*)p_->data;
      if (data && data->base_meta.meta_type == this->meta_type_) {
        this->data_ = UserMetadata(data);
        return;
      }
      p_ = p_->next;
      idx++;
    }
  }

  virtual AbstractIterator<UserMetadata>& next() override {
    while (p_) {
      p_ = p_->next;
      NvDsUserMeta* data = p_ ? (NvDsUserMeta*)p_->data : NULL;
      if (data && data->base_meta.meta_type == this->meta_type_) {
        this->data_ = UserMetadata(data);
        break;
      }
    }
    return *this;
  }

 private:
  int meta_type_;
};

template<typename T, typename S>
class Iterable {
 public:
  Iterable(GList* list=NULL) : list_(list) {}
  virtual ~Iterable() {}

  virtual guint iterate(const std::function<void(const T&)>& func) {
    unsigned int total = 0;
    for (GList* l = list_; l != NULL; l = l->next) {
        S * data = (S*) l->data;
        func(T(data));
        total++;
    }
    return total;
  }
 protected:
  GList* list_;
};

class UsermetaIterable : public Iterable<UserMetadata, NvDsUserMeta> {
 public:
  UsermetaIterable(GList* list, int meta_type) : Iterable<UserMetadata, NvDsUserMeta>(list), meta_type_(meta_type) {}

  virtual guint iterate(const std::function<void(const UserMetadata&)>& func) {
    unsigned int total = 0;
    for (GList* l = list_; l != NULL; l = l->next) {
        NvDsUserMeta * data = (NvDsUserMeta*) l->data;
        if (data->base_meta.meta_type == this->meta_type_) {
          func(UserMetadata(data));
          total++;
        }
    }
    return total;
  }
 protected:
  int meta_type_;
};

Metadata::Metadata(void* data) : data_(data) {}
Metadata::~Metadata() {}

UserMetadata::UserMetadata(void* data) : Metadata(data) {}
UserMetadata::~UserMetadata() {}
void UserMetadata::get_(void*& pointer) {
    NvDsUserMeta *user_meta = (NvDsUserMeta *)data_;
    pointer = user_meta ? user_meta->user_meta_data : nullptr;
}

int UserMetadata::metaType() const {
  NvDsUserMeta* user_meta = (NvDsUserMeta*)data_;
  return user_meta ? user_meta->base_meta.meta_type : 0;
}

void UserMetadata::setMetaType(int type) {
  NvDsUserMeta* user_meta = (NvDsUserMeta*)data_;
  if (user_meta) {
    user_meta->base_meta.meta_type = (NvDsMetaType)type;
  }
}

void* UserMetadata::userData() const {
  NvDsUserMeta* user_meta = (NvDsUserMeta*)data_;
  return user_meta ? user_meta->user_meta_data : nullptr;
}

void UserMetadata::setUserData(void* data, void*(*copy)(void*, void*), void(*release)(void*, void*)) {
  NvDsUserMeta* user_meta = (NvDsUserMeta*)data_;
  if (user_meta) {
    user_meta->user_meta_data = data;
    user_meta->base_meta.copy_func = copy;
    user_meta->base_meta.release_func = release;
  }
}

SegmentationUserMetadata::SegmentationUserMetadata(void *data)
 : UserMetadata(data) {}

SegmentationUserMetadata::SegmentationUserMetadata(const UserMetadata& user_meta)
 : UserMetadata(user_meta) {
  if (!data_) {
    return;
  }

  NvDsUserMeta *nvds_user_meta = (NvDsUserMeta *) data_;
  if (nvds_user_meta->base_meta.meta_type != NVDSINFER_SEGMENTATION_META) {
    data_ = nullptr;
  }
}

SegmentationUserMetadata::~SegmentationUserMetadata() {}

unsigned int SegmentationUserMetadata::uniqueId() const {
  if (!data_) {
    return 0;
  }

  NvDsInferSegmentationMeta *seg_meta = (NvDsInferSegmentationMeta *)((NvDsUserMeta *)data_)->user_meta_data;
  return seg_meta->unique_id;
}

unsigned int SegmentationUserMetadata::getWidth() const {
  if (!data_) {
    return 0;
  }

  NvDsInferSegmentationMeta *seg_meta = (NvDsInferSegmentationMeta *)((NvDsUserMeta *)data_)->user_meta_data;
  return seg_meta->width;
}

unsigned int SegmentationUserMetadata::getHeight() const {
  if (!data_) {
    return 0;
  }

  NvDsInferSegmentationMeta *seg_meta = (NvDsInferSegmentationMeta *)((NvDsUserMeta *)data_)->user_meta_data;
  return seg_meta->height;
}

unsigned int SegmentationUserMetadata::getClasses() const {
  if (!data_) {
    return 0;
  }

  NvDsInferSegmentationMeta *seg_meta = (NvDsInferSegmentationMeta *)((NvDsUserMeta *)data_)->user_meta_data;
  return seg_meta->classes;
}

const float* SegmentationUserMetadata::getClassProbabilitiesMap() const {
  if (!data_) {
    return nullptr;
  }

  NvDsInferSegmentationMeta *seg_meta = (NvDsInferSegmentationMeta *)((NvDsUserMeta *)data_)->user_meta_data;
  return seg_meta->class_probabilities_map;
}

const int* SegmentationUserMetadata::getClassMap() const {
  if (!data_) {
    return nullptr;
  }

  NvDsInferSegmentationMeta *seg_meta = (NvDsInferSegmentationMeta *)((NvDsUserMeta *)data_)->user_meta_data;
  return seg_meta->class_map;
}

LabelInfo::LabelInfo(void* data) : data_(data) {}
std::string LabelInfo::label() const {
  NvDsLabelInfo* info = (NvDsLabelInfo*)data_;
  if (info->pResult_label != NULL && info->pResult_label[0] != '\0')
    return std::string(info->pResult_label);
  if (info->result_label[0] != '\0')
    return std::string(info->result_label);
  return std::string();
}
unsigned int LabelInfo::classId() const { return ((NvDsLabelInfo*)data_)->result_class_id; }
float        LabelInfo::prob() const { return ((NvDsLabelInfo*)data_)->result_prob; }
unsigned int LabelInfo::labelId() const { return ((NvDsLabelInfo*)data_)->label_id; }
unsigned int LabelInfo::numClasses() const { return ((NvDsLabelInfo*)data_)->num_classes; }

ClassifierMetadata::ClassifierMetadata(void* data) : Metadata(data) {}
ClassifierMetadata::~ClassifierMetadata() {}
unsigned int ClassifierMetadata::uniqueComponentId() const { return ((NvDsClassifierMeta*)data_)->unique_component_id; }
unsigned int ClassifierMetadata::nLabels() const { return ((NvDsClassifierMeta*)data_)->num_labels; }
std::string ClassifierMetadata::classifierType() const {
  const gchar* type = ((NvDsClassifierMeta*)data_)->classifier_type;
  return (type != nullptr) ? std::string(type) : std::string();
}
std::string ClassifierMetadata::getLabel(unsigned int nth) const {
  GList *n;
  unsigned int index = 0;
  for (n = ((NvDsClassifierMeta*)data_)->label_info_list; n != NULL; n = n->next) {
    if (index++ < nth) continue;
    NvDsLabelInfo *labelInfo = (NvDsLabelInfo *) (n->data);
    if (labelInfo->pResult_label != NULL && labelInfo->pResult_label[0] != '\0') {
      return std::string(labelInfo->pResult_label);
    }
    if (labelInfo->result_label[0] != '\0') {
      return std::string(labelInfo->result_label);
    }
  }
  return std::string();
}
LabelInfo ClassifierMetadata::getLabelInfo(unsigned int nth) const {
  unsigned int index = 0;
  for (GList* n = ((NvDsClassifierMeta*)data_)->label_info_list; n != NULL; n = n->next) {
    if (index++ == nth)
      return LabelInfo(n->data);
  }
  return LabelInfo(nullptr);
}

ObjectMetadata::ObjectMetadata(void* data) : Metadata(data) {}
ObjectMetadata::~ObjectMetadata() {}
unsigned int ObjectMetadata::uniqueComponentId() const { return ((NvDsObjectMeta*)data_)->unique_component_id; }
void ObjectMetadata::setUniqueComponentId(unsigned int value) { ((NvDsObjectMeta*)data_)->unique_component_id = value; }
unsigned int ObjectMetadata::classId() const { return ((NvDsObjectMeta*)data_)->class_id; }
void ObjectMetadata::setClassId(unsigned int value) { ((NvDsObjectMeta*)data_)->class_id = value; }
unsigned long int ObjectMetadata::objectId() const { return ((NvDsObjectMeta*)data_)->object_id; }
void ObjectMetadata::setObjectId(unsigned long value) { ((NvDsObjectMeta*)data_)->object_id = value; }
float ObjectMetadata::confidence() const { return ((NvDsObjectMeta*)data_)->confidence; }
void ObjectMetadata::setConfidence(float value) { ((NvDsObjectMeta*)data_)->confidence = value; }
float ObjectMetadata::trackerConfidence() const { return ((NvDsObjectMeta*)data_)->tracker_confidence; }
void ObjectMetadata::setTrackerConfidence(float value) { ((NvDsObjectMeta*)data_)->tracker_confidence = value; }
NvOSD_RectParams& ObjectMetadata::rectParams() const { return ((NvDsObjectMeta*)data_)->rect_params; }
void ObjectMetadata::setRectParams(const NvOSD_RectParams& params) { ((NvDsObjectMeta*)data_)->rect_params = params; }
NvOSD_MaskParams& ObjectMetadata::maskParams() const { return ((NvDsObjectMeta*)data_)->mask_params; }
void ObjectMetadata::setMaskParams(const NvOSD_MaskParams& params) { ((NvDsObjectMeta*)data_)->mask_params = params; }
NvOSD_TextParams& ObjectMetadata::textParams() const { return ((NvDsObjectMeta*)data_)->text_params; }
void ObjectMetadata::setTextParams(const NvOSD_TextParams& params) { ((NvDsObjectMeta*)data_)->text_params = params; }
std::string ObjectMetadata::label() const { return ((NvDsObjectMeta*)data_)->obj_label; }
void ObjectMetadata::setLabel(std::string value) {
  if (value.length() >= MAX_LABEL_SIZE) value[MAX_LABEL_SIZE] = 0;
  strcpy(((NvDsObjectMeta*)data_)->obj_label, value.c_str());
}
NvBbox_Coords& ObjectMetadata::nvBboxInfo() const { return ((NvDsObjectMeta*)data_)->tracker_bbox_info.org_bbox_coords; }
void ObjectMetadata::setNvBboxInfo(const NvBbox_Coords& params) { ((NvDsObjectMeta*)data_)->tracker_bbox_info.org_bbox_coords = params; }

unsigned int ObjectMetadata::iterate(
    const std::function<void(const ClassifierMetadata&)>& func) const {
  NvDsObjectMeta* object_meta = (NvDsObjectMeta*) data_;
  return Iterable<ClassifierMetadata, NvDsClassifierMeta>(object_meta->classifier_meta_list).iterate(func);
}

void ObjectMetadata::initiateIterator(ClassifierMetadata::Iterator& iterator) const {
  NvDsObjectMeta* object_meta = (NvDsObjectMeta*) data_;
  iterator = std::make_unique<IteratorImpl<ClassifierMetadata, NvDsClassifierMeta>>(object_meta->classifier_meta_list);
}

unsigned int ObjectMetadata::iterate(
    const std::function<void(const UserMetadata&)>& func, int meta_type) const {
  NvDsObjectMeta* object_meta = (NvDsObjectMeta*) data_;
  return UsermetaIterable(object_meta->obj_user_meta_list, meta_type).iterate(func);
}

void ObjectMetadata::initiateIterator(UserMetadata::Iterator& iterator, int meta_type) const {
  NvDsObjectMeta* object_meta = (NvDsObjectMeta*) data_;
  iterator = std::make_unique<UsermetaIteratorImpl>(object_meta->obj_user_meta_list, meta_type);
}

void ObjectMetadata::append(const UserMetadata& data) {
  NvDsObjectMeta* object_meta = (NvDsObjectMeta*) data_;
  NvDsUserMeta* user_meta = (NvDsUserMeta*) data.data_;

  // nvds_add_user_meta_to_obj(object_meta, user_meta);
  // Use proper DeepStream function with locking
  NvDsBatchMeta* batch_meta = user_meta->base_meta.batch_meta;
  nvds_acquire_meta_lock(batch_meta);
  object_meta->obj_user_meta_list = g_list_prepend(object_meta->obj_user_meta_list, user_meta);
  nvds_release_meta_lock(batch_meta);
}

RoiMetadata::RoiMetadata(void* data) : Metadata(data) {}
RoiMetadata::~RoiMetadata() {}

NvOSD_RectParams& RoiMetadata::rectParams() const {
  return ((NvDsRoiMeta*)data_)->roi;
}

FrameMetadata RoiMetadata::frameMetadata() const {
  return FrameMetadata(((NvDsRoiMeta*)data_)->frame_meta);
}

ObjectMetadata RoiMetadata::objectMetadata() const {
  return ObjectMetadata(((NvDsRoiMeta*)data_)->object_meta);
}

unsigned int RoiMetadata::iterate(const std::function<void(const UserMetadata&)>& func, int meta_type) const {
  return UsermetaIterable(((NvDsRoiMeta*)data_)->roi_user_meta_list, meta_type).iterate(func);
}

void RoiMetadata::initiateIterator(UserMetadata::Iterator& iterator, int meta_type) const {
  NvDsRoiMeta* roi_meta = (NvDsRoiMeta*) data_;
  iterator = std::make_unique<UsermetaIteratorImpl>(roi_meta->roi_user_meta_list, meta_type);
}

unsigned int RoiMetadata::iterate(const std::function<void(const ClassifierMetadata&)>& func) const {
  return Iterable<ClassifierMetadata, NvDsClassifierMeta>(((NvDsRoiMeta*)data_)->classifier_meta_list).iterate(func);
}

void RoiMetadata::initiateIterator(ClassifierMetadata::Iterator& iterator) const {
  NvDsRoiMeta* roi_meta = (NvDsRoiMeta*) data_;
  iterator = std::make_unique<IteratorImpl<ClassifierMetadata, NvDsClassifierMeta>>(roi_meta->classifier_meta_list);
}

ObjectVisibilityUserMetadata::ObjectVisibilityUserMetadata(void* data) : UserMetadata(data) {}
ObjectVisibilityUserMetadata::ObjectVisibilityUserMetadata(const UserMetadata& user_meta) : UserMetadata(user_meta) {
  if (!data_) {
    return;
  }

  NvDsUserMeta *nvds_user_meta = (NvDsUserMeta *) data_;
  if (nvds_user_meta->base_meta.meta_type != NVDS_OBJ_VISIBILITY) {
    data_ = nullptr;
  }
}
ObjectVisibilityUserMetadata::~ObjectVisibilityUserMetadata() {}

float ObjectVisibilityUserMetadata::getVisibility() const {
  if (!data_) {
    return 0.0f;
  }

  return *((float *) ((NvDsUserMeta *)data_)->user_meta_data);
}

ObjectImageFootLocationUserMetadata::ObjectImageFootLocationUserMetadata(void* data) : UserMetadata(data) {}
ObjectImageFootLocationUserMetadata::ObjectImageFootLocationUserMetadata(const UserMetadata& user_meta) : UserMetadata(user_meta) {
  if (!data_) {
    return;
  }

  NvDsUserMeta *nvds_user_meta = (NvDsUserMeta *) data_;
  if (nvds_user_meta->base_meta.meta_type != NVDS_OBJ_IMAGE_FOOT_LOCATION) {
    data_ = nullptr;
  }
}
ObjectImageFootLocationUserMetadata::~ObjectImageFootLocationUserMetadata() {}

std::pair<float, float> ObjectImageFootLocationUserMetadata::getImageFootLocation() const {
  if (!data_) {
    return {};
  }

  return std::make_pair(((float *) ((NvDsUserMeta *)data_)->user_meta_data)[0], ((float *) ((NvDsUserMeta *)data_)->user_meta_data)[1]);
}

ObjectWorldFootLocationUserMetadata::ObjectWorldFootLocationUserMetadata(void* data) : UserMetadata(data) {}
ObjectWorldFootLocationUserMetadata::ObjectWorldFootLocationUserMetadata(const UserMetadata& user_meta) : UserMetadata(user_meta) {
  if (!data_) {
    return;
  }

  NvDsUserMeta *nvds_user_meta = (NvDsUserMeta *) data_;
  if (nvds_user_meta->base_meta.meta_type != NVDS_OBJ_WORLD_FOOT_LOCATION) {
    data_ = nullptr;
  }
}
ObjectWorldFootLocationUserMetadata::~ObjectWorldFootLocationUserMetadata() {}

std::pair<float, float> ObjectWorldFootLocationUserMetadata::getWorldFootLocation() const {
  if (!data_) {
    return {};
  }

  return std::make_pair(((float *) ((NvDsUserMeta *)data_)->user_meta_data)[0], ((float *) ((NvDsUserMeta *)data_)->user_meta_data)[1]);
}

ObjectConvexHullUserMetadata::ObjectConvexHullUserMetadata(void* data) : UserMetadata(data) {}
ObjectConvexHullUserMetadata::ObjectConvexHullUserMetadata(const UserMetadata& user_meta) : UserMetadata(user_meta) {
  if (!data_) {
    return;
  }

  NvDsUserMeta *nvds_user_meta = (NvDsUserMeta *) data_;
  if (nvds_user_meta->base_meta.meta_type != NVDS_OBJ_IMAGE_CONVEX_HULL) {
    data_ = nullptr;
  }
}
ObjectConvexHullUserMetadata::~ObjectConvexHullUserMetadata() {}

std::vector<std::pair<int, int>> ObjectConvexHullUserMetadata::getConvexHull() const {
  if (!data_) {
    return {};
  }

  std::vector<std::pair<int, int>> points;
  NvDsObjConvexHull* hull = (NvDsObjConvexHull*) ((NvDsUserMeta *)data_)->user_meta_data;
  if (!hull || !hull->list) {
    return points;
  }
  points.reserve(hull->numPoints);
  for (unsigned int i = 0; i < hull->numPoints; i++) {
    points.emplace_back(hull->list[2 * i], hull->list[2 * i + 1]);
  }
  return points;
}

Object3DBBoxUserMetadata::Object3DBBoxUserMetadata(void* data) : UserMetadata(data) {}
Object3DBBoxUserMetadata::Object3DBBoxUserMetadata(const UserMetadata& user_meta) : UserMetadata(user_meta) {
  if (!data_) {
    return;
  }

  NvDsUserMeta *nvds_user_meta = (NvDsUserMeta *) data_;
  if (nvds_user_meta->base_meta.meta_type != NVDS_OBJ_3D_META) {
    data_ = nullptr;
  }
}
Object3DBBoxUserMetadata::~Object3DBBoxUserMetadata() {}

NvDsObj3DBbox* Object3DBBoxUserMetadata::get3DBbox() const {
  if (!data_) {
    return nullptr;
  }

  return (NvDsObj3DBbox*)((NvDsUserMeta *)data_)->user_meta_data;
}

TrackerPastFrameUserMetadata::TrackerPastFrameUserMetadata(void* data) : UserMetadata(data) {}
TrackerPastFrameUserMetadata::TrackerPastFrameUserMetadata(const UserMetadata& user_meta) : UserMetadata(user_meta) {
  if (!data_) {
    return;
  }

  NvDsUserMeta *nvds_user_meta = (NvDsUserMeta *) data_;
  if (nvds_user_meta->base_meta.meta_type != NVDS_TRACKER_PAST_FRAME_META) {
    data_ = nullptr;
  }
}
TrackerPastFrameUserMetadata::~TrackerPastFrameUserMetadata() {}

NvDsTargetMiscDataBatch* TrackerPastFrameUserMetadata::getTrackerMiscDataBatch() const {
  if (!data_) {
    return nullptr;
  }

  return (NvDsTargetMiscDataBatch*)((NvDsUserMeta *)data_)->user_meta_data;
}

StructureUserMetadata::StructureUserMetadata(void* data) : UserMetadata(data) {}
StructureUserMetadata::StructureUserMetadata(const UserMetadata& user_meta) : UserMetadata(user_meta) {}
StructureUserMetadata::~StructureUserMetadata() {}

int StructureUserMetadata::getStructureUserMetatype() const {
    NvDsUserMeta *user_meta = (NvDsUserMeta *) data_;
    return user_meta ? user_meta->base_meta.meta_type : -1;
}

void* StructureUserMetadata::getStructureUserMetadata() const  {
    NvDsUserMeta *user_meta = (NvDsUserMeta *) data_;
    return user_meta ? user_meta->user_meta_data : nullptr;
}

/**
 * @brief Copy function for ObjectReidUserMetadata
 *
 * Creates a deep copy of NvDsObjReid including the feature data.
 */
static gpointer copy_obj_reid_meta(gpointer data, gpointer user_data) {
  NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
  if (user_meta && user_meta->user_meta_data) {
    NvDsObjReid* pReidObj = (NvDsObjReid *) user_meta->user_meta_data;
    if ((pReidObj == NULL) || (pReidObj->featureSize == 0) || (pReidObj->ptr_host == NULL)) {
      return NULL;
    }
    NvDsObjReid *pReidObjNew = new NvDsObjReid;
    pReidObjNew->featureSize = pReidObj->featureSize;
    // Deep copy the feature vector
    pReidObjNew->ptr_host = new float[pReidObj->featureSize];
    memcpy(pReidObjNew->ptr_host, pReidObj->ptr_host, pReidObj->featureSize * sizeof(float));
    pReidObjNew->ptr_dev = NULL;  // GPU copy not supported from Python

    return (gpointer) pReidObjNew;
  }
  return NULL;
}

/**
 * @brief Release function for ObjectReidUserMetadata
 *
 * Frees NvDsObjReid and its feature data.
 */
static void free_obj_reid_meta(gpointer meta_data, gpointer user_data) {
  NvDsUserMeta *user_meta = (NvDsUserMeta *) meta_data;
  if (user_meta && user_meta->user_meta_data) {
    NvDsObjReid *pReidObj = (NvDsObjReid *) user_meta->user_meta_data;
    if (pReidObj->ptr_host) {
      delete[] pReidObj->ptr_host;
    }
    delete pReidObj;
  }
}

ObjectReidUserMetadata::ObjectReidUserMetadata(void* data) : UserMetadata(data) {}
ObjectReidUserMetadata::ObjectReidUserMetadata(const UserMetadata& user_meta) : UserMetadata(user_meta) {
  if (!data_) {
    return;
  }

  NvDsUserMeta *nvds_user_meta = (NvDsUserMeta *) data_;
  if (nvds_user_meta->base_meta.meta_type != NVDS_TRACKER_OBJ_REID_META) {
    data_ = nullptr;
  }
}

ObjectReidUserMetadata::~ObjectReidUserMetadata() {}

ObjectReidUserMetadata& ObjectReidUserMetadata::operator=(const ObjectReidUserMetadata& other) {
  if (this != &other) {
    data_ = other.data_;
  }
  return *this;
}

unsigned int ObjectReidUserMetadata::featureSize() const {
  if (!data_) {
    return 0;
  }

  NvDsUserMeta* user_meta = (NvDsUserMeta *)data_;
  NvDsObjReid* pReidObj = (NvDsObjReid*)user_meta->user_meta_data;

  if (!pReidObj) {
    return 0;
  }

  return pReidObj->featureSize;
}

const float* ObjectReidUserMetadata::featureVector() const {
  if (!data_) {
    return nullptr;
  }

  NvDsUserMeta* user_meta = (NvDsUserMeta *)data_;
  NvDsObjReid* pReidObj = (NvDsObjReid*)user_meta->user_meta_data;

  return pReidObj ? pReidObj->ptr_host : nullptr;
}

void ObjectReidUserMetadata::setFeatureVector(const float* features, unsigned int size) {
  if (!data_) {
    return;
  }

  NvDsUserMeta* user_meta = (NvDsUserMeta*)data_;
  NvDsObjReid* pReidObj = (NvDsObjReid*)user_meta->user_meta_data;

  if (!pReidObj) {
    return;
  }

  // Free existing data if any
  if (pReidObj->ptr_host) {
    delete[] pReidObj->ptr_host;
  }

  // Allocate and copy new data
  pReidObj->featureSize = size;
  pReidObj->ptr_host = new float[size];
  memcpy(pReidObj->ptr_host, features, size * sizeof(float));
  pReidObj->ptr_dev = NULL;  // GPU copy not supported from Python
}

PreprocessBatchUserMetadata::PreprocessBatchUserMetadata(void* data) : UserMetadata(data) {}
PreprocessBatchUserMetadata::PreprocessBatchUserMetadata(const UserMetadata& user_meta) : UserMetadata(user_meta) {
  if (!data_) {
    return;
  }

  NvDsUserMeta *nvds_user_meta = (NvDsUserMeta *) data_;
  if (nvds_user_meta->base_meta.meta_type != NVDS_PREPROCESS_BATCH_META) {
    data_ = nullptr;
  }
}
PreprocessBatchUserMetadata::~PreprocessBatchUserMetadata() {}

std::vector<RoiMetadata> PreprocessBatchUserMetadata::getRois() const {
  if (!data_) {
    return {};
  }

  GstNvDsPreProcessBatchMeta* preprocess_batch_meta = (GstNvDsPreProcessBatchMeta *) ((NvDsUserMeta *)data_)->user_meta_data;
  std::vector<RoiMetadata> rois;
  for (auto& roi : preprocess_batch_meta->roi_vector) {
    rois.push_back(RoiMetadata(&roi));
  }
  return rois;
}

std::vector<unsigned int> PreprocessBatchUserMetadata::getTargetUniqueIds() const {
  if (!data_) {
    return {};
  }

  GstNvDsPreProcessBatchMeta* preprocess_batch_meta = (GstNvDsPreProcessBatchMeta *) ((NvDsUserMeta *)data_)->user_meta_data;
  std::vector<unsigned int> target_unique_ids;
  for (auto& target_unique_id : preprocess_batch_meta->target_unique_ids) {
    target_unique_ids.push_back(target_unique_id);
  }
  return target_unique_ids;
}

PreprocessTensorMetadata PreprocessBatchUserMetadata::getPreprocessTensorMetadata() const {
  if (!data_) {
    return {};
  }

  GstNvDsPreProcessBatchMeta* preprocess_batch_meta = (GstNvDsPreProcessBatchMeta *) ((NvDsUserMeta *)data_)->user_meta_data;
  return PreprocessTensorMetadata(preprocess_batch_meta->tensor_meta);
}


bool PreprocessBatchUserMetadata::initPreprocessTensorMetadata(const std::string& name, unsigned int meta_id, unsigned int unique_id, Tensor* tensor) {
  if (!data_) {
    return false;
  }

  NvDsUserMeta *user_meta = (NvDsUserMeta *) data_;
  GstNvDsPreProcessBatchMeta *meta = (GstNvDsPreProcessBatchMeta *) user_meta->user_meta_data;
  if (meta->tensor_meta != nullptr) {
    throw std::runtime_error("Preprocess tensor metadata already initialized");
  }

  NvDsPreProcessTensorMeta *tensor_meta = new NvDsPreProcessTensorMeta;
  tensor_meta->private_data = tensor;
  tensor_meta->tensor_name = name;
  tensor_meta->meta_id = meta_id;
  tensor_meta->gpu_id = tensor->deviceId();
  tensor_meta->raw_tensor_buffer = tensor->data();
  tensor_meta->buffer_size = tensor->size();
  auto tensor_shape = tensor->shape();
  for (auto &dim : tensor_shape) {
    tensor_meta->tensor_shape.push_back((int)dim);
  }
  auto dtype = tensor->dtype();
  switch (dtype) {
    case Tensor::DataType::UNSIGNED:
      if (tensor->bits() == 8) {
        tensor_meta->data_type = NvDsDataType_UINT8;
      } else if (tensor->bits() == 32) {
        tensor_meta->data_type = NvDsDataType_UINT32;
      } else if (tensor->bits() == 64) {
        tensor_meta->data_type = NvDsDataType_UINT64;
      } else {
        throw std::runtime_error("Unsupported tensor data type");
      }
      break;
    case Tensor::DataType::SIGNED:
      if (tensor->bits() == 8) {
        tensor_meta->data_type = NvDsDataType_INT8;
      } else if (tensor->bits() == 32) {
        tensor_meta->data_type = NvDsDataType_INT32;
      } else if (tensor->bits() == 64) {
        tensor_meta->data_type = NvDsDataType_INT64;
      } else {
        throw std::runtime_error("Unsupported tensor data type");
      }
      break;
    case Tensor::DataType::FLOAT:
      if (tensor->bits() == 16) {
        tensor_meta->data_type = NvDsDataType_FP16;
      } else if (tensor->bits() == 32) {
        tensor_meta->data_type = NvDsDataType_FP32;
      } else {
        throw std::runtime_error("Unsupported tensor data type");
      }
      break;
    default:
      throw std::runtime_error("Unsupported tensor data type");
  }
  meta->tensor_meta = tensor_meta;
  meta->target_unique_ids.push_back((guint64)unique_id);
  return true;
}

DisplayMetadata::DisplayMetadata(void* data) : Metadata(data) {}
DisplayMetadata::~DisplayMetadata() {}

unsigned int DisplayMetadata::nRects() const {
  NvDsDisplayMeta * display_meta = (NvDsDisplayMeta*) data_;
  return display_meta->num_rects;
}
unsigned int DisplayMetadata::nLabels() const {
  NvDsDisplayMeta * display_meta = (NvDsDisplayMeta*) data_;
  return display_meta->num_labels;
}
unsigned int DisplayMetadata::nLines() const {
  NvDsDisplayMeta * display_meta = (NvDsDisplayMeta*) data_;
  return display_meta->num_lines;
}
unsigned int DisplayMetadata::nArrows() const {
  NvDsDisplayMeta * display_meta = (NvDsDisplayMeta*) data_;
  return display_meta->num_arrows;
}
unsigned int DisplayMetadata::nCircles() const {
  NvDsDisplayMeta * display_meta = (NvDsDisplayMeta*) data_;
  return display_meta->num_circles;
}

bool DisplayMetadata::add(NvOSD_TextParams& label) {
  NvDsDisplayMeta * display_meta = (NvDsDisplayMeta*) data_;
  if (display_meta->num_labels >= MAX_ELEMENTS_IN_DISPLAY_META) {
    return false;
  }
  NvOSD_TextParams *txt_params  =
    &display_meta->text_params[display_meta->num_labels];
  *txt_params = label;
  // re-allocate the string memory
  txt_params->display_text = g_strdup(label.display_text);
  display_meta->num_labels++;
  return true;
}

bool DisplayMetadata::add(NvOSD_RectParams& rect) {
  NvDsDisplayMeta * display_meta = (NvDsDisplayMeta*) data_;
  if (display_meta->num_rects >= MAX_ELEMENTS_IN_DISPLAY_META) {
    return false;
  }
  NvOSD_RectParams *rect_params  =
    &display_meta->rect_params[display_meta->num_rects];
  *rect_params = rect;
  display_meta->num_rects++;
  return true;
}

bool DisplayMetadata::add(NvOSD_LineParams& line) {
  NvDsDisplayMeta * display_meta = (NvDsDisplayMeta*) data_;
  if (display_meta->num_lines >= MAX_ELEMENTS_IN_DISPLAY_META) {
    return false;
  }
  NvOSD_LineParams *line_params  =
    &display_meta->line_params[display_meta->num_lines];
  *line_params = line;
  display_meta->num_lines++;
  return true;
}

bool DisplayMetadata::add(NvOSD_ArrowParams& arrow) {
  NvDsDisplayMeta * display_meta = (NvDsDisplayMeta*) data_;
  if (display_meta->num_arrows >= MAX_ELEMENTS_IN_DISPLAY_META) {
    return false;
  }
  NvOSD_ArrowParams *arrow_params  =
    &display_meta->arrow_params[display_meta->num_arrows];
  *arrow_params = arrow;
  display_meta->num_arrows++;
  return true;
}

bool DisplayMetadata::add(NvOSD_CircleParams& circle) {
  NvDsDisplayMeta * display_meta = (NvDsDisplayMeta*) data_;
  if (display_meta->num_circles >= MAX_ELEMENTS_IN_DISPLAY_META) {
    return false;
  }
  NvOSD_CircleParams *circle_params  =
    &display_meta->circle_params[display_meta->num_circles];
  *circle_params = circle;
  display_meta->num_circles++;
  return true;
}

FrameMetadata::FrameMetadata(void* data) : Metadata(data) {}
FrameMetadata::~FrameMetadata() {}

unsigned int FrameMetadata::iterate(
    const std::function<void(const ObjectMetadata&)>& func) const {
  NvDsFrameMeta* frame_meta = (NvDsFrameMeta*) data_;
  return Iterable<ObjectMetadata, NvDsObjectMeta>(frame_meta->obj_meta_list).iterate(func);
}

void FrameMetadata::initiateIterator(ObjectMetadata::Iterator& iterator) const {
  NvDsFrameMeta* frame_meta = (NvDsFrameMeta*) data_;
  iterator = std::make_unique<IteratorImpl<ObjectMetadata, NvDsObjectMeta>>(frame_meta->obj_meta_list);
}

unsigned int FrameMetadata::iterate(
    const std::function<void(const DisplayMetadata&)>& func) const {
  NvDsFrameMeta* frame_meta = (NvDsFrameMeta*) data_;
  return Iterable<DisplayMetadata, NvDsDisplayMeta>(frame_meta->display_meta_list).iterate(func);
}

void FrameMetadata::initiateIterator(DisplayMetadata::Iterator& iterator) const {
  NvDsFrameMeta* frame_meta = (NvDsFrameMeta*) data_;
  iterator = std::make_unique<IteratorImpl<DisplayMetadata, NvDsDisplayMeta>>(frame_meta->display_meta_list);
}

unsigned int FrameMetadata::iterate(
    const std::function<void(const UserMetadata&)>& func, int meta_type) const {
  NvDsFrameMeta* frame_meta = (NvDsFrameMeta*) data_;
  return UsermetaIterable(frame_meta->frame_user_meta_list, meta_type).iterate(func);
}

void FrameMetadata::initiateIterator(UserMetadata::Iterator& iterator, int meta_type) const {
  NvDsFrameMeta* frame_meta = (NvDsFrameMeta*) data_;
  iterator = std::make_unique<UsermetaIteratorImpl>(frame_meta->frame_user_meta_list, meta_type);
}

unsigned int FrameMetadata::padIndex() const {
  NvDsFrameMeta* frame_meta = (NvDsFrameMeta*) data_;
  return frame_meta->pad_index;
}

unsigned int FrameMetadata::batchId() const {
  NvDsFrameMeta* frame_meta = (NvDsFrameMeta*) data_;
  return frame_meta->batch_id;
}

int FrameMetadata::frameNum() const {
  NvDsFrameMeta* frame_meta = (NvDsFrameMeta*) data_;
  return frame_meta->frame_num;
}

unsigned int FrameMetadata::sourceId() const {
  NvDsFrameMeta* frame_meta = (NvDsFrameMeta*) data_;
  return frame_meta->source_id;
}

unsigned int FrameMetadata::sourceWidth() const {
  NvDsFrameMeta* frame_meta = (NvDsFrameMeta*) data_;
  return frame_meta->source_frame_width;
}

unsigned int FrameMetadata::sourceHeight() const {
  NvDsFrameMeta* frame_meta = (NvDsFrameMeta*) data_;
  return frame_meta->source_frame_height;
}

unsigned int FrameMetadata::pipelineWidth() const
{
  NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)data_;
  return frame_meta->pipeline_width;
}

unsigned int FrameMetadata::pipelineHeight() const
{
  NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)data_;
  return frame_meta->pipeline_height;
}

uint64_t FrameMetadata::bufferPTS() const
{
  NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)data_;
  return frame_meta->buf_pts;
}

uint64_t FrameMetadata::ntpTimestamp() const
{
  NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)data_;
  return frame_meta->ntp_timestamp;
}

void FrameMetadata::append(const DisplayMetadata& data) {
  NvDsFrameMeta* frame_meta = (NvDsFrameMeta*) data_;
  NvDsDisplayMeta* display_meta = (NvDsDisplayMeta*) data.data_;
  nvds_add_display_meta_to_frame(frame_meta, display_meta);
}

void FrameMetadata::append(const UserMetadata& data) {
  NvDsFrameMeta* frame_meta = (NvDsFrameMeta*) data_;
  NvDsUserMeta* user_meta = (NvDsUserMeta*) data.data_;
  nvds_add_user_meta_to_frame(frame_meta, user_meta);
}

void FrameMetadata::append(const ObjectMetadata& data) {
  NvDsFrameMeta* frame_meta = (NvDsFrameMeta*) data_;
  NvDsObjectMeta* obj_meta = (NvDsObjectMeta*) data.data_;
  nvds_add_obj_meta_to_frame(frame_meta, obj_meta, NULL);
}

AudioFrameMetadata::AudioFrameMetadata(void* data) : Metadata(data) {}
AudioFrameMetadata::~AudioFrameMetadata() {}

unsigned int AudioFrameMetadata::iterate(
    const std::function<void(const ClassifierMetadata&)>& func) const {
  NvDsAudioFrameMeta* frame_meta = (NvDsAudioFrameMeta*) data_;
  return Iterable<ClassifierMetadata, NvDsClassifierMeta>(frame_meta->classifier_meta_list).iterate(func);
}

void AudioFrameMetadata::initiateIterator(ClassifierMetadata::Iterator& iterator) const {
  NvDsAudioFrameMeta* frame_meta = (NvDsAudioFrameMeta*) data_;
  iterator = std::make_unique<IteratorImpl<ClassifierMetadata, NvDsClassifierMeta>>(frame_meta->classifier_meta_list);
}

unsigned int AudioFrameMetadata::iterate(
    const std::function<void(const UserMetadata&)>& func, int meta_type) const {
  NvDsAudioFrameMeta* frame_meta = (NvDsAudioFrameMeta*) data_;
  return UsermetaIterable(frame_meta->frame_user_meta_list, meta_type).iterate(func);
}

void AudioFrameMetadata::initiateIterator(UserMetadata::Iterator& iterator, int meta_type) const {
  NvDsAudioFrameMeta* frame_meta = (NvDsAudioFrameMeta*) data_;
  iterator = std::make_unique<UsermetaIteratorImpl>(frame_meta->frame_user_meta_list, meta_type);
}

void AudioFrameMetadata::append(const UserMetadata& data) {
  NvDsAudioFrameMeta* frame_meta = (NvDsAudioFrameMeta*) data_;
  NvDsUserMeta* user_meta = (NvDsUserMeta*) data.data_;
  nvds_add_user_meta_to_audio_frame(frame_meta, user_meta);
}

unsigned int AudioFrameMetadata::padIndex() const {
  NvDsAudioFrameMeta* frame_meta = (NvDsAudioFrameMeta*) data_;
  return frame_meta->pad_index;
}

unsigned int AudioFrameMetadata::batchId() const {
  NvDsAudioFrameMeta* frame_meta = (NvDsAudioFrameMeta*) data_;
  return frame_meta->batch_id;
}

int AudioFrameMetadata::frameNum() const {
  NvDsAudioFrameMeta* frame_meta = (NvDsAudioFrameMeta*) data_;
  return frame_meta->frame_num;
}

unsigned int AudioFrameMetadata::sourceId() const {
  NvDsAudioFrameMeta* frame_meta = (NvDsAudioFrameMeta*) data_;
  return frame_meta->source_id;
}

int AudioFrameMetadata::numSamplesPerFrame() const {
  NvDsAudioFrameMeta* frame_meta = (NvDsAudioFrameMeta*) data_;
  return frame_meta->num_samples_per_frame;
}

unsigned int AudioFrameMetadata::sampleRate() const {
  NvDsAudioFrameMeta* frame_meta = (NvDsAudioFrameMeta*) data_;
  return frame_meta->sample_rate;
}

unsigned int AudioFrameMetadata::numChannels() const {
  NvDsAudioFrameMeta* frame_meta = (NvDsAudioFrameMeta*) data_;
  return frame_meta->num_channels;
}

int AudioFrameMetadata::format() const {
  NvDsAudioFrameMeta* frame_meta = (NvDsAudioFrameMeta*) data_;
  return static_cast<int>(frame_meta->format);
}

int AudioFrameMetadata::layout() const {
  NvDsAudioFrameMeta* frame_meta = (NvDsAudioFrameMeta*) data_;
  return static_cast<int>(frame_meta->layout);
}

uint64_t AudioFrameMetadata::bufferPTS() const {
  NvDsAudioFrameMeta* frame_meta = (NvDsAudioFrameMeta*) data_;
  return frame_meta->buf_pts;
}

uint64_t AudioFrameMetadata::ntpTimestamp() const {
  NvDsAudioFrameMeta* frame_meta = (NvDsAudioFrameMeta*) data_;
  return frame_meta->ntp_timestamp;
}

bool AudioFrameMetadata::inferDone() const {
  NvDsAudioFrameMeta* frame_meta = (NvDsAudioFrameMeta*) data_;
  return frame_meta->bInferDone;
}

int AudioFrameMetadata::classId() const {
  NvDsAudioFrameMeta* frame_meta = (NvDsAudioFrameMeta*) data_;
  return frame_meta->class_id;
}

float AudioFrameMetadata::confidence() const {
  NvDsAudioFrameMeta* frame_meta = (NvDsAudioFrameMeta*) data_;
  return frame_meta->confidence;
}

std::string AudioFrameMetadata::classLabel() const {
  NvDsAudioFrameMeta* frame_meta = (NvDsAudioFrameMeta*) data_;
  return frame_meta->class_label;
}

BatchMetadata::BatchMetadata(void* data) : Metadata(data) {}
BatchMetadata::~BatchMetadata() {}

unsigned int BatchMetadata::iterate(
    const std::function<void(const FrameMetadata&)>& func) const {
  NvDsBatchMeta* batch_meta = (NvDsBatchMeta*) data_;
  return Iterable<FrameMetadata, NvDsFrameMeta>(batch_meta->frame_meta_list).iterate(func);
}

void BatchMetadata::initiateIterator(FrameMetadata::Iterator& iterator) const {
  NvDsBatchMeta* batch_meta = (NvDsBatchMeta*) data_;
  iterator = std::make_unique<IteratorImpl<FrameMetadata, NvDsFrameMeta>>(batch_meta->frame_meta_list);
}

unsigned int BatchMetadata::iterate(
    const std::function<void(const AudioFrameMetadata&)>& func) const {
  NvDsBatchMeta* batch_meta = (NvDsBatchMeta*) data_;
  return Iterable<AudioFrameMetadata, NvDsAudioFrameMeta>(batch_meta->frame_meta_list).iterate(func);
}

void BatchMetadata::initiateIterator(AudioFrameMetadata::Iterator& iterator) const {
  NvDsBatchMeta* batch_meta = (NvDsBatchMeta*) data_;
  iterator = std::make_unique<IteratorImpl<AudioFrameMetadata, NvDsAudioFrameMeta>>(batch_meta->frame_meta_list);
}

unsigned int BatchMetadata::iterate(
    const std::function<void(const UserMetadata&)>& func, int meta_type) const {
  NvDsBatchMeta* batch_meta = (NvDsBatchMeta*) data_;
  return UsermetaIterable(batch_meta->batch_user_meta_list, meta_type).iterate(func);
}

void BatchMetadata::initiateIterator(UserMetadata::Iterator& iterator, int meta_type) const {
  NvDsBatchMeta* batch_meta = (NvDsBatchMeta*) data_;
  iterator = std::make_unique<UsermetaIteratorImpl>(batch_meta->batch_user_meta_list, meta_type);
}


bool BatchMetadata::acquire(DisplayMetadata& data) {
  NvDsBatchMeta* batch_meta = (NvDsBatchMeta*) data_;
  NvDsDisplayMeta* display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
  if (display_meta == NULL) {
    return false;
  }
  data = DisplayMetadata(display_meta);
  return true;
}

bool BatchMetadata::acquire(FrameMetadata& data) {
  NvDsBatchMeta* batch_meta = (NvDsBatchMeta*) data_;
  NvDsFrameMeta* frame_meta = nvds_acquire_frame_meta_from_pool(batch_meta);
  if (frame_meta == NULL) {
    return false;
  }
  data = FrameMetadata(frame_meta);
  return true;
}

bool BatchMetadata::acquire(ObjectMetadata& data) {
  NvDsBatchMeta* batch_meta = (NvDsBatchMeta*) data_;
  NvDsObjectMeta* object_meta = nvds_acquire_obj_meta_from_pool(batch_meta);
  if (object_meta == NULL) {
    return false;
  }
  data = ObjectMetadata(object_meta);
  return true;
}

bool BatchMetadata::acquire(EventMessageUserMetadata& data) {
  NvDsBatchMeta* batch_meta = (NvDsBatchMeta*) data_;
  NvDsUserMeta* user_meta = nvds_acquire_user_meta_from_pool(batch_meta);
  if (user_meta == NULL) {
    return false;
  }
  data = EventMessageUserMetadata(user_meta);
  return true;
}

bool BatchMetadata::acquire(PreprocessBatchUserMetadata& data) {
  NvDsBatchMeta* batch_meta = (NvDsBatchMeta*) data_;
  NvDsUserMeta* user_meta = nvds_acquire_user_meta_from_pool(batch_meta);
  if (user_meta == NULL) {
    return false;
  }
  GstNvDsPreProcessBatchMeta* preprocess_batch_meta = new GstNvDsPreProcessBatchMeta;
  preprocess_batch_meta->tensor_meta = nullptr;
  preprocess_batch_meta->private_data = nullptr;
  user_meta->user_meta_data = preprocess_batch_meta;
  user_meta->base_meta.meta_type = (NvDsMetaType) NVDS_PREPROCESS_BATCH_META;
  user_meta->base_meta.copy_func = NULL;
  user_meta->base_meta.release_func =
      (NvDsMetaReleaseFunc) release_preprocess_batch_meta;
  user_meta->base_meta.batch_meta = batch_meta;
  data = PreprocessBatchUserMetadata(user_meta);
  return true;
}

bool BatchMetadata::acquire(UserMetadata& data) {
  NvDsBatchMeta* batch_meta = (NvDsBatchMeta*) data_;
  NvDsUserMeta* user_meta = nvds_acquire_user_meta_from_pool(batch_meta);
  if (user_meta == NULL) {
    return false;
  }
  data = UserMetadata(user_meta);
  return true;
}

bool BatchMetadata::acquire(ObjectReidUserMetadata& data) {
  NvDsBatchMeta* batch_meta = (NvDsBatchMeta*) data_;
  NvDsUserMeta* user_meta = nvds_acquire_user_meta_from_pool(batch_meta);
  if (user_meta == NULL) {
    return false;
  }
  // Initialize NvDsObjReid structure
  NvDsObjReid* pReidObj = new NvDsObjReid;
  pReidObj->featureSize = 0;
  pReidObj->ptr_host = NULL;
  pReidObj->ptr_dev = NULL;
  user_meta->user_meta_data = pReidObj;
  user_meta->base_meta.meta_type = (NvDsMetaType) NVDS_TRACKER_OBJ_REID_META;
  user_meta->base_meta.copy_func = (NvDsMetaCopyFunc) copy_obj_reid_meta;
  user_meta->base_meta.release_func = (NvDsMetaReleaseFunc) free_obj_reid_meta;
  user_meta->base_meta.batch_meta = batch_meta;
  data = ObjectReidUserMetadata(user_meta);
  return true;
}

void BatchMetadata::append(const FrameMetadata& data) {
  NvDsBatchMeta* batch_meta = (NvDsBatchMeta*) data_;
  NvDsFrameMeta* frame_meta = (NvDsFrameMeta*) data.data_;
  nvds_add_frame_meta_to_batch(batch_meta, frame_meta);
}

void BatchMetadata::append(const UserMetadata& data) {
  NvDsBatchMeta* batch_meta = (NvDsBatchMeta*) data_;
  NvDsUserMeta* user_meta = (NvDsUserMeta*) data.data_;
  nvds_add_user_meta_to_batch(batch_meta, user_meta);
}

UserMetadata BatchMetadata::acquireUserMetadata_(
  void* data, unsigned int type, void*(*copy)(void*, void*), void(*free)(void*, void*)
) {
  NvDsBatchMeta* batch_meta = (NvDsBatchMeta*) data_;
  NvDsUserMeta* user_meta = nvds_acquire_user_meta_from_pool(batch_meta);
  if (user_meta == NULL) {
    return UserMetadata(NULL);
  }
  user_meta->user_meta_data = (void*) data;
  user_meta->base_meta.meta_type = (NvDsMetaType) type;
  user_meta->base_meta.copy_func = copy;
  user_meta->base_meta.release_func = free;
  return UserMetadata(user_meta);
}

unsigned int BatchMetadata::nFrames() const {
  if (!data_) {
    return 0;
  }
  NvDsBatchMeta* batch_meta = (NvDsBatchMeta*) data_;
  return batch_meta->num_frames_in_batch;
}

bool BatchMetadata::isAudioBatch() const {
  if (!data_) {
    return false;
  }
  NvDsBatchMeta* batch_meta = (NvDsBatchMeta*) data_;
  return batch_meta->base_meta.meta_type == NVDS_AUDIO_BATCH_META;
}

PreprocessTensorMetadata::PreprocessTensorMetadata(void *data) : data_(data) {
}

PreprocessTensorMetadata::~PreprocessTensorMetadata() {}

std::string PreprocessTensorMetadata::getName() const {
  NvDsPreProcessTensorMeta *meta = (NvDsPreProcessTensorMeta *) data_;
  return meta->tensor_name;
}

unsigned int PreprocessTensorMetadata::getMetaId() const {
  NvDsPreProcessTensorMeta *meta = (NvDsPreProcessTensorMeta *) data_;
  return meta->meta_id;
}

Tensor* PreprocessTensorMetadata::getTensor() const {
  NvDsPreProcessTensorMeta *meta = (NvDsPreProcessTensorMeta *) data_;
  unsigned int rank = meta->tensor_shape.size();
  Tensor::DataType dtype = Tensor::DataType::INVALID;
  unsigned int bits = 0;
  int64_t shape[NVDSINFER_MAX_DIMS] = {0};
  int64_t strides[NVDSINFER_MAX_DIMS] = {0};
  void * data_ptr = meta->raw_tensor_buffer;
  Tensor::DeviceType device = Tensor::DeviceType::GPU;

  switch (meta->data_type) {
    case NvDsDataType_FP32:
      dtype = Tensor::DataType::FLOAT;
      bits = 32;
      break;
    case NvDsDataType_FP16:
      dtype = Tensor::DataType::FLOAT;
      bits = 16;
      break;
    case NvDsDataType_INT8:
      dtype = Tensor::DataType::SIGNED;
      bits = 8;
      break;
    case NvDsDataType_INT32:
      dtype = Tensor::DataType::SIGNED;
      bits = 32;
      break;
    case NvDsDataType_UINT8:
      dtype = Tensor::DataType::UNSIGNED;
      bits = 8;
      break;
    case NvDsDataType_UINT32:
      dtype = Tensor::DataType::UNSIGNED;
      bits = 32;
      break;
    default:
      break;
  }
  for (unsigned int i = 0; i < rank; i++) {
    shape[i] = meta->tensor_shape[i];
  }
  /* strides for flatten memory layout*/
  strides[rank-1] = 1;
  for (int j = rank-2; j >= 0; j--) {
    strides[j] = strides[j+1] * shape[j+1];
  }

  Tensor::Context* context = new Tensor::Context;
  return new Tensor(rank, dtype, bits, shape, strides, data_ptr, "", meta->gpu_id, device, context);
}

AnalyticsObjInfo::AnalyticsObjInfo(void* data) : UserMetadata(data) {}
AnalyticsObjInfo::AnalyticsObjInfo(const UserMetadata& user_meta) : UserMetadata(user_meta) {
  if (!data_) {
    return;
  }

  NvDsUserMeta *nvds_user_meta = (NvDsUserMeta *) data_;
  if (nvds_user_meta->base_meta.meta_type != NVDS_OBJ_META_NVDSANALYTICS) {
    data_ = nullptr;
  }
}

AnalyticsObjInfo::~AnalyticsObjInfo() {}

std::vector<std::string> AnalyticsObjInfo::getRoiStatus() const {
  if (!data_) {
    return {};
  }

  NvDsAnalyticsObjInfo *obj_info = (NvDsAnalyticsObjInfo *)((NvDsUserMeta *)data_)->user_meta_data;
  return obj_info->roiStatus;
}

std::vector<std::string> AnalyticsObjInfo::getOcStatus() const {
  if (!data_) {
    return {};
  }

  NvDsAnalyticsObjInfo *obj_info = (NvDsAnalyticsObjInfo *)((NvDsUserMeta *)data_)->user_meta_data;
  return obj_info->ocStatus;
}

std::vector<std::string> AnalyticsObjInfo::getLcStatus() const {
  if (!data_) {
    return {};
  }

  NvDsAnalyticsObjInfo *obj_info = (NvDsAnalyticsObjInfo *)((NvDsUserMeta *)data_)->user_meta_data;
  return obj_info->lcStatus;
}

std::string AnalyticsObjInfo::getDirStatus() const {
  if (!data_) {
    return {};
  }

  NvDsAnalyticsObjInfo *obj_info = (NvDsAnalyticsObjInfo *)((NvDsUserMeta *)data_)->user_meta_data;
  return obj_info->dirStatus;
}

unsigned int AnalyticsObjInfo::getUniqueId() const {
  if (!data_) {
    return 0;
  }

  NvDsAnalyticsObjInfo *obj_info = (NvDsAnalyticsObjInfo *)((NvDsUserMeta *)data_)->user_meta_data;
  return obj_info->unique_id;
}

std::string AnalyticsObjInfo::getObjStatus() const {
  if (!data_) {
    return {};
  }

  NvDsAnalyticsObjInfo *obj_info = (NvDsAnalyticsObjInfo *)((NvDsUserMeta *)data_)->user_meta_data;
  return obj_info->objStatus;
}

AnalyticsFrameMeta::AnalyticsFrameMeta(void* data) : UserMetadata(data) {}
AnalyticsFrameMeta::AnalyticsFrameMeta(const UserMetadata& user_meta) : UserMetadata(user_meta) {
  if (!data_) {
    return;
  }

  NvDsUserMeta *nvds_user_meta = (NvDsUserMeta *) data_;
  if (nvds_user_meta->base_meta.meta_type != NVDS_FRAME_META_NVDSANALYTICS) {
    data_ = nullptr;
  }
}

AnalyticsFrameMeta::~AnalyticsFrameMeta() {}

std::unordered_map<std::string, bool> AnalyticsFrameMeta::getOcStatus() const {
  if (!data_) {
    return {};
  }

  NvDsAnalyticsFrameMeta *frame_meta = (NvDsAnalyticsFrameMeta *)((NvDsUserMeta *)data_)->user_meta_data;
  return frame_meta->ocStatus;
}

std::unordered_map<std::string, uint32_t> AnalyticsFrameMeta::getObjInROIcnt() const {
  if (!data_) {
    return {};
  }

  NvDsAnalyticsFrameMeta *frame_meta = (NvDsAnalyticsFrameMeta *)((NvDsUserMeta *)data_)->user_meta_data;
  return frame_meta->objInROIcnt;
}

std::unordered_map<std::string, uint64_t> AnalyticsFrameMeta::getObjLCCurrCnt() const {
  if (!data_) {
    return {};
  }

  NvDsAnalyticsFrameMeta *frame_meta = (NvDsAnalyticsFrameMeta *)((NvDsUserMeta *)data_)->user_meta_data;
  return frame_meta->objLCCurrCnt;
}

std::unordered_map<std::string, uint64_t> AnalyticsFrameMeta::getObjLCCumCnt() const {
  if (!data_) {
    return {};
  }

  NvDsAnalyticsFrameMeta *frame_meta = (NvDsAnalyticsFrameMeta *)((NvDsUserMeta *)data_)->user_meta_data;
  return frame_meta->objLCCumCnt;
}

unsigned int AnalyticsFrameMeta::getUniqueId() const {
  if (!data_) {
    return 0;
  }

  NvDsAnalyticsFrameMeta *frame_meta = (NvDsAnalyticsFrameMeta *)((NvDsUserMeta *)data_)->user_meta_data;
  return frame_meta->unique_id;
}

std::unordered_map<int, uint32_t> AnalyticsFrameMeta::getObjCnt() const {
  if (!data_) {
    return {};
  }

  NvDsAnalyticsFrameMeta *frame_meta = (NvDsAnalyticsFrameMeta *)((NvDsUserMeta *)data_)->user_meta_data;
  return frame_meta->objCnt;
}

} // namespace deepstream