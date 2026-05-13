/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <iostream>
#include <string>
#include <memory>
#include <vector>

#include "buffer_probe.hpp"

#include <gst/gst.h>
#include "nvdsmeta.h"
#include "gstnvdsinfer.h"
#include "nvdsinfer_custom_impl.h"
#include "nvdsinfer_dbscan.h"

using namespace std;

extern "C"
    bool NvDsInferParseCustomResnet (std::vector < NvDsInferLayerInfo >
    const &outputLayersInfo, NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector < NvDsInferObjectDetectionInfo > &objectList);

#define PGIE_CLASS_ID_VEHICLE 0
#define PGIE_CLASS_ID_PERSON 2

namespace deepstream
{

  class TensorMetaParser : public BufferProbe::IBatchMetadataOperator
  {
  public:
    TensorMetaParser() {
      dbscan_ = NvDsInferDBScanCreate();
    }
    virtual ~TensorMetaParser() {
      NvDsInferDBScanDestroy(dbscan_);
    }

    virtual probeReturn handleData(BufferProbe &probe, BatchMetadata &data) {
      FrameMetadata::Iterator frame_itr;
      for (data.initiateIterator(frame_itr); !frame_itr->done(); frame_itr->next()) {
        int network_width = 0;
        int network_height = 0;
        int stream_width = 0;
        int stream_height = 0;
        int n_classes = 0;
        NvDsInferParseDetectionParams detectionParams;
        detectionParams.numClassesConfigured = 4;
        detectionParams.perClassPreclusterThreshold = {0.2, 0.2, 0.2, 0.2};

        probe.getProperty("network-width", network_width)
             .getProperty("network-height", network_height)
             .getProperty("stream-width", stream_width)
             .getProperty("stream-height", stream_height)
             .getProperty("num-classes", n_classes);
        float scalers[] = {(float)stream_width/network_width, (float)stream_height/network_height };
        NvDsInferNetworkInfo networkInfo{(unsigned int)network_width, (unsigned int)network_height, 3};

        FrameMetadata &frame_meta = frame_itr->get();
        frame_meta.iterate([&](const UserMetadata& user_meta) {
          /* We know what tensor meta it is, so we directly cast it to the right type */
          UserMetadataTemplate<NvDsInferTensorMeta, NVDSINFER_TENSOR_OUTPUT_META> tensor_meta = user_meta;
          NvDsInferTensorMeta& output_tensor = tensor_meta.get();
          for (unsigned int i = 0; i < output_tensor.num_output_layers; i++) {
            NvDsInferLayerInfo &info = output_tensor.output_layers_info[i];
            /* fill in the buffer address */
            info.buffer = output_tensor.out_buf_ptrs_host[i];
          }
          /* Parse output tensor and fill detection results into objectList. */
          std::vector <NvDsInferLayerInfo> outputLayersInfo(output_tensor.output_layers_info, output_tensor.output_layers_info + output_tensor.num_output_layers);
          std::vector <NvDsInferObjectDetectionInfo> objectList;
          NvDsInferParseCustomResnet(outputLayersInfo, networkInfo, detectionParams, objectList);

          NvDsInferDBScanClusteringParams clusteringParams;
          clusteringParams.enableATHRFilter = true;
          clusteringParams.thresholdATHR = 60.0;
          clusteringParams.eps = 0.95;
          clusteringParams.minBoxes = 3;
          clusteringParams.minScore = 0.5;

          /* Create perClassObjectList: vector<vector<NvDsInferObjectDetectionInfo>>. Each vector is of same classID */
          std::vector <std::vector<NvDsInferObjectDetectionInfo>> perClassObjectList(n_classes);
          for (auto & obj:objectList) {
            perClassObjectList[obj.classId].emplace_back (obj);
          }

          /* Call NvDsInferDBScanCluster on each of the vector and resize it */
          for (unsigned int c = 0; c < perClassObjectList.size(); c++) {
            NvDsInferObjectDetectionInfo *objArray = (NvDsInferObjectDetectionInfo*) (perClassObjectList[c].data());
            size_t numObjects = perClassObjectList[c].size();

            /* Cluster together rectangles with similar locations and sizes since these rectangles might represent the same object using DBSCAN. */
            if (clusteringParams.minBoxes > 0) {
                NvDsInferDBScanCluster(dbscan_, &clusteringParams, objArray, &numObjects);
            }
            perClassObjectList[c].resize(numObjects);

            /* Generate object meta for downstream usage such as OSD display */
            for (unsigned int i = 0; i < numObjects; i++) {
              ObjectMetadata object_meta;
              if (data.acquire(object_meta)) {
                object_meta.setClassId(c);
                object_meta.setConfidence(objArray[i].detectionConfidence);
                object_meta.setRectParams(
                  NvOSD_RectParams{objArray[i].left*scalers[0], objArray[i].top*scalers[1], objArray[i].width*scalers[0], objArray[i].height*scalers[1], 1, NvOSD_ColorParams{1.0, 0, 0, 1.0}}
                );
                frame_meta.append(object_meta);
              }
            }
          }

          std::cout << "Object Counter: " <<  " Pad Idx = " << frame_meta.padIndex() <<  " Frame Number = " << frame_meta.frameNum() <<
                       " Vehicle Count = " << perClassObjectList[PGIE_CLASS_ID_VEHICLE].size() <<
                       " Person Count = " << perClassObjectList[PGIE_CLASS_ID_PERSON].size() << std::endl;


        }, NVDSINFER_TENSOR_OUTPUT_META);
      }

      return probeReturn::Probe_Ok;
    }
  protected:
   NvDsInferDBScan* dbscan_;
  };

}