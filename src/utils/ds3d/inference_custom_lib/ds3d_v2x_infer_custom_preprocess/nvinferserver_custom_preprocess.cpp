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

#include "infer_datatypes.h"
#include "nvdsinfer.h"
#include "nvdsinfer_custom_impl.h"
#include "tensor.hpp"

// cuda heade files
#include <cstddef>
#include <cuda_runtime_api.h>
// inlcude all ds3d hpp header files
#include <ds3d/common/hpp/datamap.hpp>
#include <ds3d/common/hpp/frame.hpp>
#include <ds3d/common/hpp/lidar_custom_process.hpp>
#include <ds3d/common/helper/check.hpp>
#include <ds3d/common/common.h>
#include <ds3d/common/defines.h>
#include <ds3d/common/hpp/yaml_config.hpp>
#include <string>

using namespace ds3d;
using namespace nvdsinferserver;

extern "C" IInferCustomPreprocessor* CreateInferServerCustomPreprocess();

class NvInferServerCustomPreProcess : public IInferCustomPreprocessor {
public:
  ~NvInferServerCustomPreProcess() = default;
  NvDsInferStatus preproc(GuardDataMap &datamap, SharedIBatchArray batchArray,
                          cudaStream_t stream) final {
    DS3D_INFER_ASSERT(batchArray->getSize() > 1);

    const IOptions *inOptions = batchArray->getOptions();
    std::string filterConfigRaw = "";
    if (!preProcConfFileParser_) {
      if (inOptions->hasValue(kDs3dInferenceParas)) {
        DS3D_INFER_ASSERT(inOptions->getValue<std::string>(kDs3dInferenceParas,
                                                           filterConfigRaw) ==
                     NVDSINFER_SUCCESS);
      }
      customPreProcConfParse(filterConfigRaw);
      preProcConfFileParser_ = true;

      if (intervalsFrom_.empty() || geometrysFrom_.empty()) {
        LOG_ERROR("interval or geometry path empty");
        return NVDSINFER_INVALID_PARAMS;
      }

      intervals_ = v2xinfer::Tensor::load(intervalsFrom_.c_str());
      geometrys_ = v2xinfer::Tensor::load(geometrysFrom_.c_str());
      if (!intervals_.empty() && !geometrys_.empty()) {
        intervals_.print();
        geometrys_.print();
      }
      LOG_INFO("infer result layerSize:%d", batchArray->getSize());
    }

    if (intervals_.empty() || geometrys_.empty()) {
      LOG_ERROR("Failed to load intervals & geometrys\n");
      return NVDSINFER_INVALID_PARAMS;
    }
    int num_intervals = intervals_.size(0);

    for (uint32_t i = 0; i < batchArray->getSize(); i++) {
      auto buf = batchArray->getSafeBuf(i);
      auto layerName = buf->getBufDesc().name;
      FrameGuard frame;
      if (layerName == "images") {
        DS3D_FAILED_RETURN(isGood(datamap.getGuardData(imagesFrom_, frame)),
                           NVDSINFER_TENSORRT_ERROR,
                           "No frame: %s found in datamap to copy into buf: %s",
                           imagesFrom_.c_str(), layerName.c_str());
        assert(frame);
        DS3D_FAILED_RETURN(isGood(copyFrameToTensor(frame, buf, stream)),
                           NVDSINFER_TENSORRT_ERROR,
                           "copy frame: %s to input tensor: %s failed.",
                           imagesFrom_.c_str(), layerName.c_str());
      } else if (layerName == "feats") {
        DS3D_FAILED_RETURN(isGood(datamap.getGuardData(featsFrom_, frame)),
                           NVDSINFER_TENSORRT_ERROR,
                           "No frame: %s found in datamap to copy into buf: %s",
                           featsFrom_.c_str(), layerName.c_str());
        assert(frame);
        DS3D_FAILED_RETURN(isGood(copyFrameToTensor(frame, buf, stream)),
                           NVDSINFER_TENSORRT_ERROR,
                           "copy frame: %s to input tensor: %s failed.",
                           featsFrom_.c_str(), layerName.c_str());
      } else if (layerName == "coords") {
        DS3D_FAILED_RETURN(isGood(datamap.getGuardData(coordsFrom_, frame)),
                           NVDSINFER_TENSORRT_ERROR,
                           "No frame: %s found in datamap to copy into buf: %s",
                           coordsFrom_.c_str(), layerName.c_str());
        assert(frame);
        DS3D_FAILED_RETURN(isGood(copyFrameToTensor(frame, buf, stream)),
                           NVDSINFER_TENSORRT_ERROR,
                           "copy frame: %s to input tensor: %s failed.",
                           coordsFrom_.c_str(), layerName.c_str());
      } else if (layerName == "N") {
        DS3D_FAILED_RETURN(isGood(datamap.getGuardData(NFrom_, frame)),
                           NVDSINFER_TENSORRT_ERROR,
                           "No frame: %s found in datamap to copy into buf: %s",
                           NFrom_.c_str(), layerName.c_str());
        assert(frame);
        DS3D_FAILED_RETURN(isGood(copyFrameToTensor(frame, buf, stream)),
                           NVDSINFER_TENSORRT_ERROR,
                           "copy frame: %s to input tensor: %s failed.",
                           NFrom_.c_str(), layerName.c_str());
      } else if (layerName == "intervals") {
        checkCudaErrors(cudaMemcpyAsync(buf->getBufPtr(0), intervals_.ptr(),
                                        intervals_.bytes(), cudaMemcpyDefault,
                                        stream));
      } else if (layerName == "geometry") {
        checkCudaErrors(cudaMemcpyAsync(buf->getBufPtr(0), geometrys_.ptr(),
                                        geometrys_.bytes(), cudaMemcpyDefault,
                                        stream));
      } else if (layerName == "num_intervals") {
        checkCudaErrors(cudaMemcpyAsync(buf->getBufPtr(0), &num_intervals,
                                        sizeof(int), cudaMemcpyDefault,
                                        stream));
      }
    }
    return NVDSINFER_SUCCESS;
  }

private:
  void customPreProcConfParse(const std::string &configRaw) {
    YAML::Node node = YAML::Load(configRaw);
    if (node["preprocess"]) {
      auto preproc = node["preprocess"];
      if (!preproc["intervalsFrom"].IsNull()) {
        intervalsFrom_ = preproc["intervalsFrom"].as<std::string>();
      }
      if (!preproc["geometrysFrom"].IsNull()) {
        geometrysFrom_ = preproc["geometrysFrom"].as<std::string>();
      }
      if (!preproc["imagesLayerFrom"].IsNull()) {
        imagesFrom_ = preproc["imagesFrom"].as<std::string>();
      }
      if (!preproc["featsFrom"].IsNull()) {
        featsFrom_ = preproc["featsFrom"].as<std::string>();
      }
      if (!preproc["coordsFrom"].IsNull()) {
        coordsFrom_ = preproc["coordsFrom"].as<std::string>();
      }
      if (!preproc["NFrom"].IsNull()) {
        NFrom_ = preproc["NFrom"].as<std::string>();
      }
    }
  }

  ErrCode getMemKind(MemType from, InferMemType to, cudaMemcpyKind &kind) {
    using Key = std::tuple<int, int>;
    static std::map<Key, cudaMemcpyKind> kMemCopyTable = {
        {std::make_tuple((int)MemType::kGpuCuda, (int)InferMemType::kGpuCuda),
         cudaMemcpyDeviceToDevice},
        {std::make_tuple((int)MemType::kGpuCuda, (int)InferMemType::kCpu),
         cudaMemcpyDeviceToHost},
        {std::make_tuple((int)MemType::kCpu, (int)InferMemType::kGpuCuda),
         cudaMemcpyHostToDevice},
        {std::make_tuple((int)MemType::kCpu, (int)InferMemType::kCpu),
         cudaMemcpyHostToHost},
    };

    auto fromTypeFun = [](MemType t) -> int {
      if (t == MemType::kCpuPinned) {
        return static_cast<int>(MemType::kCpu);
      }
      return static_cast<int>(t);
    };

    auto ToTypeFun = [](InferMemType t) -> int {
      if (t == InferMemType::kCpuCuda) {
        return static_cast<int>(InferMemType::kCpu);
      }
      if (t == InferMemType::kNvSurface || t == InferMemType::kNvSurfaceArray) {
        return static_cast<int>(InferMemType::kNone);
      }
      return static_cast<int>(t);
    };

    Key k = std::make_tuple(fromTypeFun(from), ToTypeFun(to));
    DS3D_FAILED_RETURN(kMemCopyTable.count(k), ErrCode::kParam,
                       "cuda mem copy type is not supported. from: %d to %d",
                       static_cast<int>(from), static_cast<int>(to));

    assert(kMemCopyTable.count(k));
    kind = kMemCopyTable[k];
    return ErrCode::kGood;
  }

  ErrCode copyFrameToTensor(FrameGuard& frame, SharedIBatchBuffer& tensor, cudaStream_t cudaStream) {
    assert(frame && tensor);
    void* framePtr = frame->base();
    void* tensorPtr = tensor->getBufPtr(0);
    const auto& tensorDesc = tensor->getBufDesc();
    uint64_t tensorBytes = tensor->getTotalBytes();
    uint64_t frameBytes = frame->bytes();
    // TODO: support different kind of device copy
    cudaMemcpyKind cpKind = cudaMemcpyHostToDevice;
    DS3D_ERROR_RETURN(
        getMemKind(frame->memType(), tensorDesc.memType, cpKind),
        "copy2DFrameToTensor get memory type failed");

    DS3D_FAILED_RETURN(
        frameBytes <= tensorBytes, ErrCode::kMem,
        "Frame bytes: %ld is larger than allocated input tensor: %s total bytes: %ld", frameBytes,
        tensorDesc.name.c_str(), tensorBytes);

    DS_ASSERT(cudaStream);
    cudaError_t cudaErr =
        cudaMemcpyAsync(tensorPtr, framePtr, frameBytes, cpKind, cudaStream);
    DS3D_FAILED_RETURN(
        cudaErr == cudaSuccess, ErrCode::kCuda, "copy2DFrameToTensor failed with cuda error: %s",
        cudaGetErrorString(cudaErr));
    return ErrCode::kGood;
  }

private:
  bool preProcConfFileParser_ = false;
  std::string intervalsFrom_;
  std::string geometrysFrom_;
  std::string imagesFrom_;
  std::string featsFrom_;
  std::string coordsFrom_;
  std::string NFrom_;
  v2xinfer::Tensor intervals_;
  v2xinfer::Tensor geometrys_;
};

extern "C" {
IInferCustomPreprocessor *CreateInferServerCustomPreprocess() {
  return new NvInferServerCustomPreProcess();
}
}