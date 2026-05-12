/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
#include "post_processor_bodypose.h"
#include <algorithm>
using namespace std;
#define MAX_TIME_STAMP_LEN 32

#define ACQUIRE_DISP_META(dmeta)  \
  if (dmeta->num_circles == MAX_ELEMENTS_IN_DISPLAY_META  || \
      dmeta->num_labels == MAX_ELEMENTS_IN_DISPLAY_META ||  \
      dmeta->num_lines == MAX_ELEMENTS_IN_DISPLAY_META) \
        { \
          dmeta = nvds_acquire_display_meta_from_pool(bmeta);\
          nvds_add_display_meta_to_frame(frame_meta, dmeta);\
        }\

#define GET_LINE(lparams) \
        ACQUIRE_DISP_META(dmeta)\
        lparams = &dmeta->line_params[dmeta->num_lines];\
        dmeta->num_lines++;\

static
void generate_ts_rfc3339 (char *buf, int buf_size)
{
  time_t tloc;
  struct tm tm_log;
  struct timespec ts;
  char strmsec[6]; //.nnnZ\0

  clock_gettime(CLOCK_REALTIME,  &ts);
  memcpy(&tloc, (void *)(&ts.tv_sec), sizeof(time_t));
  gmtime_r(&tloc, &tm_log);
  strftime(buf, buf_size,"%Y-%m-%dT%H:%M:%S", &tm_log);
  int ms = ts.tv_nsec/1000000;
  g_snprintf(strmsec, sizeof(strmsec),".%.3dZ", ms);
  strncat(buf, strmsec, buf_size);
}

static
gpointer copy_bodypose_meta (gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
  NvDsEventMsgMeta *srcMeta = (NvDsEventMsgMeta *) user_meta->user_meta_data;
  NvDsEventMsgMeta *dstMeta = NULL;
  NvDsPersonObject *srcExt = NULL;
  NvDsPersonObject *dstExt = NULL;

  dstMeta = (NvDsEventMsgMeta *)g_memdup2 ((gpointer)srcMeta, sizeof(NvDsEventMsgMeta));

  // pose
  dstMeta->pose.num_joints = srcMeta->pose.num_joints;
  dstMeta->pose.pose_type = srcMeta->pose.pose_type;
  dstMeta->pose.joints = (NvDsJoint *)g_memdup2 ((gpointer)srcMeta->pose.joints,
                                    sizeof(NvDsJoint)*srcMeta->pose.num_joints);

  if (srcMeta->ts)
    dstMeta->ts = g_strdup (srcMeta->ts);

  if (srcMeta->sensorStr)
    dstMeta->sensorStr = g_strdup (srcMeta->sensorStr);

  if (srcMeta->objSignature.size > 0) {
    dstMeta->objSignature.signature = (gdouble *)g_memdup2 ((gpointer)srcMeta->objSignature.signature,
                                                sizeof(gdouble)*srcMeta->objSignature.size);
    dstMeta->objSignature.size = srcMeta->objSignature.size;
  }

  if(srcMeta->objectId) {
    dstMeta->objectId = g_strdup (srcMeta->objectId);
  }

  if (srcMeta->extMsg){
    dstMeta->extMsg = g_memdup2(srcMeta->extMsg, srcMeta->extMsgSize);
    dstMeta->extMsgSize = srcMeta->extMsgSize;
    srcExt = (NvDsPersonObject *)srcMeta->extMsg;
    dstExt = (NvDsPersonObject *)dstMeta->extMsg;
    dstExt->gender = g_strdup(srcExt->gender);
    dstExt->hair = g_strdup(srcExt->hair);
    dstExt->cap = g_strdup(srcExt->cap);
    dstExt->apparel = g_strdup(srcExt->apparel);
    dstExt->age = srcExt->age;
  }
  return dstMeta;
}

static void
release_bodypose_meta (gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
  NvDsEventMsgMeta *srcMeta = (NvDsEventMsgMeta *) user_meta->user_meta_data;

  // pose
  g_free (srcMeta->pose.joints);
  g_free (srcMeta->ts);
  g_free (srcMeta->sensorStr);

  if (srcMeta->objSignature.size > 0) {
    g_free (srcMeta->objSignature.signature);
    srcMeta->objSignature.size = 0;
  }

  if(srcMeta->objectId) {
    g_free (srcMeta->objectId);
  }

  g_free (srcMeta->extMsg);
  srcMeta->extMsgSize = 0;
  srcMeta->extMsg = NULL;

  g_free (user_meta->user_meta_data);
  user_meta->user_meta_data = NULL;
}

NvDsPostProcessStatus
BodyPoseModelPostProcessor::initResource(NvDsPostProcessContextInitParams& initParams)
{
  ModelPostProcessor::initResource(initParams);
  m_ClassificationThreshold = initParams.classifierThreshold;
  return NVDSPOSTPROCESS_SUCCESS;
}

NvDsPostProcessStatus
BodyPoseModelPostProcessor::parseEachFrame(
    const std::vector<NvDsInferLayerInfo>& outputLayers,
    NvDsPostProcessFrameOutput& result)
{
    result.outputType = NvDsPostProcessNetworkType_BodyPose;
    fillBodyPoseOutput(outputLayers, result.bodyPoseOutput);
    return NVDSPOSTPROCESS_SUCCESS;
}

NvDsPostProcessStatus
BodyPoseModelPostProcessor::fillBodyPoseOutput(
    const std::vector<NvDsInferLayerInfo>& outputLayers,
    NvDsPostProcessBodyPoseOutput& output)
{
    movenetposeFromTensorMeta(outputLayers, output);
    return NVDSPOSTPROCESS_SUCCESS;
}

void
BodyPoseModelPostProcessor::attachMetadata     (NvBufSurface *surf, gint batch_idx,
    NvDsBatchMeta  *batch_meta,
    NvDsFrameMeta  *frame_meta,
    NvDsObjectMeta  *obj_meta,
    NvDsObjectMeta *parent_obj_meta,
    NvDsPostProcessFrameOutput & detection_output,
    NvDsPostProcessDetectionParams *all_params,
    std::set <gint> & filterOutClassIds,
    int32_t unique_id,
    gboolean output_instance_mask,
    gboolean process_full_frame,
    float segmentationThreshold,
    gboolean maintain_aspect_ratio,
    NvDsRoiMeta *roi_meta,
    gboolean symmetric_padding)
{
  NvDsEventMsgMeta *msg_meta = (NvDsEventMsgMeta *) g_malloc0 (sizeof (NvDsEventMsgMeta));
  NvDsPersonObject *msg_meta_ext = (NvDsPersonObject *) g_malloc0 (sizeof (NvDsPersonObject));
  NvDsBatchMeta *bmeta = frame_meta->base_meta.batch_meta;
  nvds_acquire_meta_lock (bmeta);
  NvDsDisplayMeta *dmeta = nvds_acquire_display_meta_from_pool(bmeta);
  const int numKeyPoints = 17;
  float keypoints[2* numKeyPoints];
  float keypoints_confidence[numKeyPoints];
  gint surf_width  = surf->surfaceList[batch_idx].width;
  gint surf_height = surf->surfaceList[batch_idx].height;
  float scale_x =
    (float)surf_width/(float)m_NetworkInfo.width;
  float scale_y =
    (float)surf_height/(float)m_NetworkInfo.height;

  nvds_add_display_meta_to_frame(frame_meta, dmeta);

  gint src_width  = surf_width;
  gint src_height = surf_height;
  gint src_top    = 0;
  gint src_left   = 0;
  gfloat pad_x = 0.0;
  gfloat pad_y = 0.0;
  if (obj_meta) {
    src_width  = obj_meta->rect_params.width;
    src_height = obj_meta->rect_params.height;
    src_top    = obj_meta->rect_params.top;
    src_left   = obj_meta->rect_params.left;
    scale_x =
    (float)src_width/(float)m_NetworkInfo.width;
    scale_y =
    (float)src_height/(float)m_NetworkInfo.height;
  }

  if (maintain_aspect_ratio){
    if (symmetric_padding){
      if (scale_x > scale_y){
        pad_y = ((m_NetworkInfo.height - src_height/scale_x)/2.0)/(float)m_NetworkInfo.height;
      }
      else {
        pad_x = ((m_NetworkInfo.width - src_width/scale_y)/2.0)/(float)m_NetworkInfo.width;
      }
    }
    if (scale_x > scale_y)
      scale_y = scale_x = src_width;
    else
      scale_x = scale_y = src_height;
  }
  else {
    scale_x = src_width;
    scale_y = src_height;
  }

  std::vector<OneEuroFilter> filter_vec;
  if (obj_meta && (obj_meta->object_id != G_MAXUINT64))
  {
    if (m_filter_pose.find(obj_meta->object_id) == m_filter_pose.end())
    {
      const float m_oneEuroSampleRate = 30.0f;
      // const float m_oneEuroMinCutoffFreq = 0.1f;
      // const float m_oneEuroCutoffSlope = 0.05f;
      const float m_oneEuroDerivCutoffFreq = 1.0f; // Hz

      // std::vector <SF1eFilter*> filter_vec;

      for (int j = 0; j < numKeyPoints * 2; j++)
      {
        // TODO:Pending delete especially when object goes out of view, or ID switch
        // will cause memleak, cleanup required wrap into class
        // filters for x and y
        // for (auto& fil : m_filterKeypoints2D) fil.reset(m_oneEuroSampleRate, 0.1f, 0.05, m_oneEuroDerivCutoffFreq);
        filter_vec.push_back(OneEuroFilter(m_oneEuroSampleRate, 0.1f, 0.05, m_oneEuroDerivCutoffFreq));
        filter_vec.push_back(OneEuroFilter(m_oneEuroSampleRate, 0.1f, 0.05, m_oneEuroDerivCutoffFreq));
      }
      m_filter_pose[obj_meta->object_id] = filter_vec;
    }
    else
      filter_vec = m_filter_pose[obj_meta->object_id];
  }
  int batchSize_offset = 0;
  // x,y,z,c
  float min_x = src_width;
  float min_y = src_height;
  float max_x = 0;
  float max_y = 0;
  float min_conf = 1;
  for (int i = 0; i < numKeyPoints; i++) {
    int index = batchSize_offset + i * 3;
    if (!obj_meta || obj_meta->object_id == G_MAXUINT64){
      keypoints[2*i] =((detection_output.bodyPoseOutput.data[index + 1 ]  -pad_x)*
                            scale_x
                          + (float)src_left);
      keypoints[2*i+1] = ((detection_output.bodyPoseOutput.data[index]  -pad_y)*
                            scale_y + (float)src_top);
    }
    else {
      keypoints[2*i] = (filter_vec[i*2].filter((detection_output.bodyPoseOutput.data[index + 1 ]-pad_x))*
                        scale_x + (float)src_left);
      keypoints[2*i+1] = (filter_vec[i*2+1].filter((detection_output.bodyPoseOutput.data[index]-pad_y))*
                          scale_y + (float)src_top);
    }

    min_x = std::min(keypoints[2*i], min_x);
    max_x = std::max(keypoints[2*i], max_x);
    min_y = std::min(keypoints[2*i+1], min_y);
    max_y = std::max(keypoints[2*i+1], max_y);
    keypoints_confidence[i] = detection_output.bodyPoseOutput.data[index + 2];
    min_conf = std::min(keypoints_confidence[i], min_conf);
  }

  osdBody(frame_meta, bmeta, dmeta, numKeyPoints,
    keypoints, keypoints_confidence);

  msg_meta->type = NVDS_EVENT_ENTRY; //Should this be ENTRY
  msg_meta->objType = (NvDsObjectType) NVDS_OBJECT_TYPE_PERSON;
  msg_meta->bbox.top = min_x;
  msg_meta->bbox.left = min_y;
  msg_meta->bbox.width = max_x - min_x +1;
  msg_meta->bbox.height = max_y - min_y + 1;
  msg_meta->extMsg = msg_meta_ext;
  msg_meta->extMsgSize  = sizeof(NvDsPersonObject);
  msg_meta_ext->gender  = g_strdup("");
  msg_meta_ext->hair    = g_strdup("");
  msg_meta_ext->cap     = g_strdup("");
  msg_meta_ext->apparel = g_strdup("");
  msg_meta_ext->age = 0;

  //---msg_meta->poses---
  msg_meta->pose.pose_type = 2;// pdatapose3D

  msg_meta->pose.num_joints = numKeyPoints;
  msg_meta->pose.joints = (NvDsJoint *)g_malloc0(sizeof(NvDsJoint) * numKeyPoints);
  for(int i = 0; i < msg_meta->pose.num_joints; i++){
    msg_meta->pose.joints[i].x = keypoints[2*i  ];
    msg_meta->pose.joints[i].y = keypoints[2*i+1];
    msg_meta->pose.joints[i].confidence = keypoints_confidence[i];
  }

  msg_meta->objectId = (gchar *) g_malloc0 (MAX_LABEL_SIZE);
  if (obj_meta){
    msg_meta->objClassId = obj_meta->class_id;
    msg_meta->trackingId = obj_meta->object_id;
    strncpy(msg_meta->objectId, obj_meta->obj_label, MAX_LABEL_SIZE);
  }
  else {
    msg_meta->objClassId = 0;
    msg_meta->trackingId = 0;
    strncpy(msg_meta->objectId, "Person", MAX_LABEL_SIZE);
  }
  msg_meta->frameId = frame_meta->frame_num;
  msg_meta->confidence = min_conf;
  msg_meta->ts = (gchar *) g_malloc0 (MAX_TIME_STAMP_LEN + 1);
  generate_ts_rfc3339(msg_meta->ts, MAX_TIME_STAMP_LEN);

  NvDsBatchMeta *batch_meta1 = frame_meta->base_meta.batch_meta;
  NvDsUserMeta *user_event_meta = nvds_acquire_user_meta_from_pool (batch_meta1);
  if (user_event_meta) {
    user_event_meta->user_meta_data = (void *) msg_meta;
    user_event_meta->base_meta.meta_type = NVDS_EVENT_MSG_META;
    user_event_meta->base_meta.copy_func = (NvDsMetaCopyFunc) copy_bodypose_meta;
    user_event_meta->base_meta.release_func = (NvDsMetaReleaseFunc)release_bodypose_meta;
    nvds_add_user_meta_to_frame(frame_meta, user_event_meta);
  } else {
    g_printerr("Error in attaching event meta to buffer\n");
  }
  nvds_release_meta_lock (bmeta);
}

void
BodyPoseModelPostProcessor::releaseFrameOutput(NvDsPostProcessFrameOutput& frameOutput)
{
    switch (frameOutput.outputType)
    {
        case NvDsPostProcessNetworkType_BodyPose:
          //Release if meta not attached
            //delete[] frameOutput.segmentationOutput.class_map;
            break;
        default:
            break;
    }
}

float BodyPoseModelPostProcessor::median(std::vector<float>& v) {
    size_t n = v.size() / 2;
    nth_element(v.begin(), v.begin() + n, v.end());
    return v[n];
}

void BodyPoseModelPostProcessor::osdBody(NvDsFrameMeta* frame_meta,
      NvDsBatchMeta *bmeta,
      NvDsDisplayMeta *dmeta,
      const int numKeyPoints,
      const float keypoints[],
      const float keypoints_confidence[])
{
  const int keypoint_radius = 3;//6;//3;
  const int keypoint_line_width = 2;//4;//2;

  const int num_joints = 17;
  const int num_bones = 18;
  const int idx_bones[] = { 0,1, 0,2, 1,3, 2,4,
                            0,5, 0,6, 5,6, 5,7,
                            7,9, 6,8, 8,10, 11,12,
                            5,11, 11,13, 13,15, 6,12,
                            12,14, 14,16};
  const NvOSD_ColorParams bone_colors[] = {
                          NvOSD_ColorParams{1.0, 0, 0, 1},
                          NvOSD_ColorParams{0, 0, 1.0, 1},
                          NvOSD_ColorParams{1.0, 0, 0, 1},
                          NvOSD_ColorParams{0, 0, 1.0, 1},
                          NvOSD_ColorParams{0, 0, 1.0, 1},
                          NvOSD_ColorParams{1.0, 0, 0, 1},
                          NvOSD_ColorParams{1.0, 0, 0, 1},
                          NvOSD_ColorParams{1.0, 0, 0, 1},
                          NvOSD_ColorParams{1.0, 0, 0, 1},
                          NvOSD_ColorParams{1.0, 0, 0, 1},
                          NvOSD_ColorParams{0, 0, 1.0, 1},
                          NvOSD_ColorParams{0, 0, 1.0, 1},
                          NvOSD_ColorParams{0, 0, 1.0, 1},
                          NvOSD_ColorParams{0, 0, 1.0, 1},
                          NvOSD_ColorParams{1.0, 0, 0, 1},
                          NvOSD_ColorParams{0, 0, 1.0, 1},
                          NvOSD_ColorParams{0, 1.0, 0, 1},
                          NvOSD_ColorParams{0, 1.0, 0, 1},
                          NvOSD_ColorParams{0, 1.0, 0, 1},
                          NvOSD_ColorParams{0, 1.0, 0, 1},
                          NvOSD_ColorParams{0, 1.0, 0, 1},
                          NvOSD_ColorParams{0, 1.0, 0, 1},
                          NvOSD_ColorParams{0, 0, 1.0, 1},
                          NvOSD_ColorParams{1.0, 0, 0, 1},
                          NvOSD_ColorParams{0, 1.0, 0, 1}};

  for (int ii = 0; ii < num_joints; ii++) {
    int i = ii;// idx_joints[ii];


    if (keypoints_confidence[i] < m_ClassificationThreshold)
      continue;

    ACQUIRE_DISP_META(dmeta);
    NvOSD_CircleParams &cparams = dmeta->circle_params[dmeta->num_circles];
    cparams.xc = keypoints[2 * i    ];
    cparams.yc = keypoints[2 * i + 1];
    cparams.radius = keypoint_radius;
    cparams.circle_color = NvOSD_ColorParams{1.0, 0, 0, 1};
    cparams.has_bg_color = 1;
    cparams.bg_color = NvOSD_ColorParams{1.0, 0, 0, 1};
    dmeta->num_circles++;
  }

  for (int i = 0; i < num_bones; i++) {
    int i0 = idx_bones[2 * i    ];
    int i1 = idx_bones[2 * i + 1];

    if ((keypoints_confidence[i0] < m_ClassificationThreshold) ||
        (keypoints_confidence[i1] < m_ClassificationThreshold))
        continue;

    ACQUIRE_DISP_META(dmeta);
    NvOSD_LineParams *lparams = &dmeta->line_params[dmeta->num_lines];
    lparams->x1 = keypoints[2 * i0];
    lparams->y1 = keypoints[2 * i0 + 1];
    lparams->x2 = keypoints[2 * i1];
    lparams->y2 = keypoints[2 * i1 + 1];
    lparams->line_width = keypoint_line_width;
    lparams->line_color = bone_colors[i];
    dmeta->num_lines++;
  }

  return;
}


void BodyPoseModelPostProcessor::movenetposeFromTensorMeta(
    const std::vector<NvDsInferLayerInfo>& outputLayers,
    NvDsPostProcessBodyPoseOutput& output)
{
  // const int pelvis = 0;
  // const int left_hip = 1;
  // const int right_hip = 2;
  // const int torso = 3;
  // const int left_knee = 4;
  // const int right_knee = 5;
  // const int neck = 6;
  // const int left_ankle = 7;
  // const int right_ankle = 8;
  // const int left_big_toe = 9;
  // const int right_big_toe = 10;
  // const int left_small_toe = 11;
  // const int right_small_toe = 12;
  // const int left_heel = 13;
  // const int right_heel = 14;
  // const int nose = 15;
  // const int left_eye = 16;
  // const int right_eye = 17;
  // const int left_ear = 18;
  // const int right_ear = 19;
  // const int left_shoulder = 20;
  // const int right_shoulder = 21;
  // const int left_elbow = 22;
  // const int right_elbow = 23;
  // const int left_wrist = 24;
  // const int right_wrist = 25;
  // const int left_pinky_knuckle = 26;
  // const int right_pinky_knuckle = 27;
  // const int left_middle_tip = 28;
  // const int right_middle_tip = 29;
  // const int left_index_knuckle = 30;
  // const int right_index_knuckle = 31;
  // const int left_thumb_tip = 32;
  // const int right_thumb_tip = 33;

  unsigned int numAttributes = outputLayers.size();
  for (unsigned int m=0; m < numAttributes;m++){
    const NvDsInferLayerInfo *info = &outputLayers[m];
    if (!strcmp(info->layerName, "output_0")) {
      output.data = (float *)info->buffer;

    }
  }
}


