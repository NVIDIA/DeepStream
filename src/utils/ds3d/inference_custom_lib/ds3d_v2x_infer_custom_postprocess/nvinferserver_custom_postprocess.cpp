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

#include "decoder.hpp"
#include "infer_custom_process.h"
#include "infer_datatypes.h"
#include "nvdsinfer.h"
#include "yaml-cpp/node/node.h"

// cuda
#include <cuda_runtime_api.h>
// inlcude all ds3d hpp header files
#include <ds3d/common/common.h>
#include <ds3d/common/ds3d_analysis_datatype.h>
#include <ds3d/common/hpp/datamap.hpp>
#include <ds3d/common/hpp/frame.hpp>
#include <ds3d/common/hpp/yaml_config.hpp>
#include <ds3d/common/helper/check.hpp>
#include <ds3d/common/impl/impl_frames.h>
#include <array>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace ds3d;
using namespace nvdsinferserver;

extern "C" IInferCustomProcessor *
CreateDs3dTritonInferV2xPostprocess(const char *config, uint32_t configLen);

class NvInferServerCustomProcess : public IInferCustomProcessor {
public:
  ~NvInferServerCustomProcess() {
    if (stream_) {
      checkCudaErrors(cudaStreamDestroy(stream_));
    }
  };

  void supportInputMemType(InferMemType &type) final {
    type = InferMemType::kGpuCuda;
  }

  bool requireInferLoop() const final { return false; }

  NvDsInferStatus extraInputProcess(
      const std::vector<IBatchBuffer *>
          &primaryInputs, // primary tensor(image) has been processed
      std::vector<IBatchBuffer *> &extraInputs,
      const IOptions *options) final {
    return NVDSINFER_SUCCESS;
  }

  NvDsInferStatus inferenceDone(const IBatchArray *batchArray,
                                const IOptions *inOptions) override {
    std::string filterConfigRaw = "";
    if (!configFileParser_) {
      if (inOptions->hasValue(kLidarInferenceParas)) {
        DS3D_INFER_ASSERT(inOptions->getValue<std::string>(kLidarInferenceParas,
                                                           filterConfigRaw) ==
                     NVDSINFER_SUCCESS);
      }
      customPostProcConfigParse(filterConfigRaw);
      configFileParser_ = true;
      LOG_INFO("infer result layerSize:%d", batchArray->getSize());
    }
    // init decoder
    if (decoder_ == nullptr) {
      bevfusion::decoder::Decoder::DecoderParameter param;
      param.voxel_size = ds3d::Float2(0.2, 0.2);
      param.pc_range   = ds3d::Float2(0, -51.2);
      param.post_center_range_start = ds3d::Float3(0.0, -61.2, -10.0);
      param.post_center_range_end   = ds3d::Float3(122.4, 61.2, 10.0);
      param.num_classes     = 5;
      param.out_size_factor = 4.0f;
      decoder_ = bevfusion::decoder::Decoder::createDecoder(param);
      if (decoder_ == nullptr) {
        LOG_ERROR("Failed to create Decoder.\n");
        return NVDSINFER_CUSTOM_LIB_FAILED;
      }
    }
    if (stream_ == nullptr) {
      checkCudaErrors(
          cudaStreamCreateWithFlags(&stream_, cudaStreamDefault));
    }
    TensorMap outTensors;
    for (uint32_t i = 0; i < batchArray->getSize(); i++) {
      auto buf = batchArray->getSafeBuf(i);
      outTensors[buf->getBufDesc().name] = buf;
    }
    std::vector<bevfusion::decoder::Head> heads;
    heads.reserve(2);
    const std::string layer[2][6] = {
      {"reg", "height", "rotation", "vel", "heatmap", "dim"},
      {"reg2", "height2", "rotation2", "vel2", "heatmap2", "dim2"}
    };
    for (int i = 0; i < 2; i++) {
      bevfusion::decoder::Head head;
      head.reg = outTensors[layer[i][0]]->getBufPtr(0);
      head.height = outTensors[layer[i][1]]->getBufPtr(0);
      head.rotation = outTensors[layer[i][2]]->getBufPtr(0);
      head.vel = outTensors[layer[i][3]]->getBufPtr(0);
      head.heatmap = outTensors[layer[i][4]]->getBufPtr(0);
      head.dim = outTensors[layer[i][5]]->getBufPtr(0);
      // reg
      head.fm_width = outTensors[layer[i][0]]->getBufDesc().dims.d[3];
      head.fm_area = outTensors[layer[i][0]]->getBufDesc().dims.d[2] * head.fm_width;
      head.batch = outTensors[layer[i][0]]->getBufDesc().dims.d[0];
      heads.emplace_back(head);
    }

    auto bboxes = decoder_->forward(heads, scoreThresh_, 4.0f, stream_);
    if (bboxes.empty()) {
      LOG_INFO("no bboxed");
      return NVDSINFER_SUCCESS;
    }
    // extract ThreeDBoxes and put to datamap
    abiRefDataMap *refDataMap = nullptr;
    if (inOptions->hasValue(kLidarRefDataMap)) {
      DS3D_INFER_ASSERT(inOptions->getObj(kLidarRefDataMap, refDataMap) ==
                   NVDSINFER_SUCCESS);
    }
    GuardDataMap dataMap(*refDataMap);
    using Lidar3DBboxSeries = std::vector<Lidar3DBbox>;
    std::vector<Lidar3DBboxSeries> seriesArray(batchSize_);
    for (const auto &bbox : bboxes) {
      auto lidar3dbox = toLidar3DBbox(bbox);
      lidar3dbox.dump();
      if (bbox.ibatch > batchSize_)
        break;
      seriesArray[bbox.ibatch].emplace_back(lidar3dbox);
    }
    std::string outputKey(kLidar3DBboxRawData);
    for (int i = 0; i < batchSize_; i++) {
      auto &series = seriesArray[i];
      if (series.empty()) {
        continue;
      }
      const auto key = outputKey + "_" + std::to_string(i);
      size_t bufBytes = sizeof(Lidar3DBbox) * series.size();
      void* bufBase = (void*)series.data();
      Shape shape{3, {1, (int)series.size(), sizeof(Lidar3DBbox)}};
      FrameGuard bboxFrame = impl::WrapFrame<uint8_t, FrameType::kCustom>(
          bufBase, bufBytes, shape, MemType::kCpu, 0, [outdata = std::move(series)](void*) {});
      ErrCode code = dataMap.setGuardData(key, bboxFrame);
      if (!isGood(code)) {
        LOG_WARNING("dataMap setGuardData %s failed", key.c_str());
      }
    }
    return NVDSINFER_SUCCESS;

  }

  /** override function
   * Receiving errors if anything wrong inside lowlevel lib
   */
  void notifyError(NvDsInferStatus s) final {}

  // TODO: nvinferserver custom postprocess configuration
  NvDsInferStatus customPostProcConfigParse(const std::string &config) {
    auto nodes = YAML::Load(config);
    if (!nodes["postprocess"].IsNull()) {
      auto postproc = nodes["postprocess"];
      if (!postproc["score_threshold"].IsNull()) {
        scoreThresh_ = postproc["score_threshold"].as<float>();
      }
      if (!postproc["batchSize"].IsNull()) {
        batchSize_ = postproc["batchSize"].as<int>();
      }
    }
    if (!nodes["labels"].IsNull()) {
      auto labelsNode = nodes["labels"];
      if (labelsNode.IsSequence()) {
        for (std::size_t i = 0; i < labelsNode.size(); ++i) {
          // Each element in the sequence is a map with a single key-value pair
          auto labelNode = labelsNode[i];
          if (labelNode.IsMap()) {
            for (auto it = labelNode.begin(); it != labelNode.end(); ++it) {
              auto label = it->first.as<std::string>();
              auto color = it->second["color"].as<std::array<int, 3>>();
              std::array<float, 4> out = {{ 1.0, 1.0, 1.0, 1.0 }};
              std::transform(color.begin(), color.end(), out.begin(), [](int x) { return (float)x / 255.f; });
              labels_.emplace_back(label);
              colorMap_.insert(std::make_pair(label, out));
              LOG_INFO("label %s color:%.3f %.3f %.3f %.3f", label.c_str(), out[0], out[1], out[2], out[3]);
            }
          }
        }
      }
    }
    return NVDSINFER_SUCCESS;
  }

private:
  using BD3DBbox = bevfusion::decoder::ThreeDBox;
  Lidar3DBbox toLidar3DBbox(const BD3DBbox &bbox) {
    Lidar3DBbox lidar3dbox;
    lidar3dbox.centerX = bbox.position.x;
    lidar3dbox.centerY = bbox.position.y;
    lidar3dbox.centerZ = bbox.position.z;
    lidar3dbox.dx = bbox.size.w;
    lidar3dbox.dy = bbox.size.l;
    lidar3dbox.dz = bbox.size.h;
    lidar3dbox.yaw = bbox.yaw;
    lidar3dbox.score = bbox.score;
    lidar3dbox.cid = bbox.id;
    if (labels_.size() > 0 && !labels_[bbox.category].empty()) {
      const auto& label = labels_[bbox.category];
      memcpy(lidar3dbox.labels, label.c_str(), label.size());
      auto color = colorMap_[label];
      if (!color.empty()) {
        lidar3dbox.bboxColor = {{color[0], color[1], color[2], 1.0f}};
      }
    }
    return lidar3dbox;
  }

  using TensorMap = std::unordered_map<std::string, SharedIBatchBuffer>;
  std::unique_ptr<bevfusion::decoder::Decoder> decoder_;
  float scoreThresh_ = 0.5;
  bool configFileParser_ = false;
  cudaStream_t stream_ = nullptr;
  int batchSize_ = 0;
  std::vector<std::string> labels_;
  std::unordered_map<std::string, std::array<float, 4>> colorMap_;
};

/** Implementation to Create a custom processor for DeepStream Triton
 * plugin(nvinferserver) to do custom extra input preprocess and custom
 * postprocess on triton based models.
 */
extern "C" {
IInferCustomProcessor *
CreateDs3dTritonInferV2xPostprocess(const char * /* config */,
                                    uint32_t /* configLen */) {
  return new NvInferServerCustomProcess();
}
}