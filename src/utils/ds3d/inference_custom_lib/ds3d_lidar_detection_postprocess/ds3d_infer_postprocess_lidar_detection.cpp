/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <map>
#include "nvdsinfer.h"
#include "infer_custom_process.h"
//#include "infer_options.h"
#include "infer_datatypes.h"
#include <cuda_runtime_api.h>
#include <ds3d/common/common.h>
#include <ds3d/common/hpp/datamap.hpp>
#include <ds3d/common/hpp/frame.hpp>
#include <ds3d/common/hpp/yaml_config.hpp>
#include <ds3d/common/impl/impl_frames.h>
#include <ds3d/common/ds3d_analysis_datatype.h>

// #include "lidar_postprocess.hpp"

using namespace ds3d;
using namespace nvdsinferserver;

#ifndef INFER_ASSERT
#define INFER_ASSERT(expr)                                                     \
    do {                                                                       \
        if (!(expr)) {                                                         \
            fprintf(stderr, "%s:%d ASSERT(%s) \n", __FILE__, __LINE__, #expr); \
            std::abort();                                                      \
        }                                                                      \
    } while (0)
#endif


#define checkCudaErrors(status)                                                                    \
    {                                                                                              \
        if (status != 0) {                                                                         \
            std::cout << "Cuda failure: " << cudaGetErrorString(status) << " at line " << __LINE__ \
                      << " in file " << __FILE__ << " error status: " << status << std::endl;      \
            abort();                                                                               \
        }                                                                                          \
    }

extern "C" IInferCustomProcessor* Nvds3d_CreateLidarDetectionPostprocess(
    const char* config, uint32_t configLen);

static const std::string kLidar3DBboxOutputName = "output_3d_bbox";

struct ObjectLabel {
    uint32_t id = 0;
    std::string name;
    vec4f color = {{1.0f, 1.0f, 0, 1.0f}};
};

class DS3DTritonLidarInferCustomPostProcess : public IInferCustomProcessor {
private:
    using TensorMap = std::unordered_map<std::string, SharedIBatchBuffer>;
    bool _configFileParserFlag = false;
    std::vector<std::string> _modelOutputLayers;
    std::vector<ObjectLabel> _classLabels;
    float _scoreThresh = 0.5;
    std::string _3dBboxKey = kLidar3DBboxRawData;

public:
    ~DS3DTritonLidarInferCustomPostProcess() final = default;

    void supportInputMemType(::InferMemType& type) final { type = InferMemType::kGpuCuda; }

    bool requireInferLoop() const final { return false; }

    NvDsInferStatus extraInputProcess(
        const std::vector<IBatchBuffer*>&
            primaryInputs,  // primary tensor(image) has been processed
        std::vector<IBatchBuffer*>& extraInputs, const IOptions* options) override
    {
        return NVDSINFER_SUCCESS;
    }

    NvDsInferStatus inferenceDone(const IBatchArray* batchArray, const IOptions* inOptions) override
    {
        std::string filterConfigRaw = "";
        if (!_configFileParserFlag) {
            if (inOptions->hasValue(kLidarInferenceParas)) {
                INFER_ASSERT(
                    inOptions->getValue<std::string>(kLidarInferenceParas, filterConfigRaw) ==
                    NVDSINFER_SUCCESS);
            }
            CustomPostprocessConfigParse(filterConfigRaw);
            _configFileParserFlag = true;
        }
        LOG_INFO("infer result layerSize:%d", batchArray->getSize());

        NvDsInferStatus ret = NVDSINFER_SUCCESS;

        TensorMap outTensors;
        for (uint32_t i = 0; i < batchArray->getSize(); ++i) {
            auto buf = batchArray->getSafeBuf(i);
            outTensors[buf->getBufDesc().name] = buf;
        }
        std::vector<Lidar3DBbox> bboxes;
        ret = parseLidar3Dbbox(outTensors, bboxes);
        if (ret != NVDSINFER_SUCCESS) {
            LOG_ERROR("parse lidar 3d bbox failed");
            return ret;
        }

        // attach parsed bboxes into datamp
        abiRefDataMap* refDataMap = nullptr;
        if (inOptions->hasValue(kLidarRefDataMap)) {
            INFER_ASSERT(inOptions->getObj(kLidarRefDataMap, refDataMap) == NVDSINFER_SUCCESS);
        }
        GuardDataMap dataMap(*refDataMap);
        size_t bufBytes = sizeof(Lidar3DBbox) * bboxes.size();
        void* bufBase = (void*)bboxes.data();

        Shape shape{3, {1, (int)bboxes.size(), sizeof(Lidar3DBbox)}};
        FrameGuard bboxFrame = impl::WrapFrame<uint8_t, FrameType::kCustom>(
            bufBase, bufBytes, shape, MemType::kCpu, 0, [outdata = std::move(bboxes)](void*) {});

        ErrCode code = dataMap.setGuardData(_3dBboxKey, bboxFrame);
        if (!isGood(code)) {
            LOG_ERROR("lidar infer postprocess: dataMap setGuardData kLidar3DBboxRawData failed");
            return NVDSINFER_MEM_ERROR;
        }

        return ret;
    }

    NvDsInferStatus parseLidar3Dbbox(TensorMap tensors, std::vector<Lidar3DBbox>& bboxes)
    {
        if (!tensors.count(kLidar3DBboxOutputName)) {
            LOG_WARNING(
                "No output tensor %s found in inference response", kLidar3DBboxOutputName.c_str());
            return NVDSINFER_SUCCESS;
        }
        auto buf = tensors[kLidar3DBboxOutputName];
        int bufLen = buf->getTotalBytes();
        const InferBufferDescription& desc = buf->getBufDesc();
        LOG_DEBUG(
            "%s, %d, %d, %d\n", desc.name.c_str(), (int)desc.dataType, (int)desc.memType, bufLen);
        float* bboxPtr = (float*)buf->getBufPtr(0);
        INFER_ASSERT(desc.memType == InferMemType::kCpu || desc.memType == InferMemType::kCpuCuda);
        INFER_ASSERT(desc.dims.numDims >= 2);
        int32_t columns = desc.dims.d[desc.dims.numDims - 1];
        if (columns < 11) {
            LOG_ERROR(
                "output tensors shape is wrong. tensor: %s, dt: %d, mem: %d, len: %d\n",
                desc.name.c_str(), (int)desc.dataType, (int)desc.memType, bufLen);
            return NVDSINFER_TRITON_ERROR;
        }

        int32_t numBbox = std::accumulate(
            desc.dims.d, desc.dims.d + desc.dims.numDims - 1, 1,
            [](int s, int i) { return s * i; });

        if (numBbox <= 0) {
            return NVDSINFER_SUCCESS;
        }

        std::vector<Lidar3DBbox> res;
        res.reserve(numBbox);
        for (int32_t i = 0; i < numBbox; ++i) {
            float* row = bboxPtr + columns * i;
            auto box = Lidar3DBbox(
                row[0], row[1], row[2], row[3], row[4], row[5], row[6], (int)row[9], row[10]);
            if (box.score < _scoreThresh) {
                continue;
            }
            if (box.cid < (int)_classLabels.size()) {
                auto& label = _classLabels[box.cid];
                box.bboxColor = label.color;
                uint32_t strLen = std::min<uint32_t>(label.name.size(), DS3D_MAX_LABEL_SIZE - 1);
                strncpy(box.labels, label.name.c_str(), strLen);
            }
            res.emplace_back(box);
        }

        bboxes = std::move(res);
        for (size_t i = 0; i < bboxes.size(); i++) {
            LOG_INFO(
                "%s, %f, %f, %f, %f, %f, %f, %f, %f\n", bboxes[i].labels, bboxes[i].centerX,
                bboxes[i].centerY, bboxes[i].centerZ, bboxes[i].dx, bboxes[i].dy, bboxes[i].dz,
                bboxes[i].yaw, bboxes[i].score);
        }

        return NVDSINFER_SUCCESS;
    }

    /** override function
     * Receiving errors if anything wrong inside lowlevel lib
     */
    void notifyError(NvDsInferStatus s) final {}

private:
    NvDsInferStatus CustomPostprocessConfigParse(const std::string& filterConfigRaw)
    {
        YAML::Node node = YAML::Load(filterConfigRaw);

        auto yPost = node["postprocess"];
        if (yPost) {
            if (yPost["score_threshold"]) {
                _scoreThresh = yPost["score_threshold"].as<float>();
            }
            if (yPost["3d_bbox_key"]) {
                _3dBboxKey = yPost["3d_bbox_key"].as<std::string>();
            }
        }

        auto yamlClassLables = node["labels"];
        if (yamlClassLables) {
            uint32_t labelId = 0;
            for (auto item : yamlClassLables) {
                auto dict = item.begin();
                ObjectLabel label;
                label.id = labelId;
                ++labelId;
                label.name = dict->first.as<std::string>();
                auto ySecond = dict->second;
                if (ySecond && ySecond["color"]) {
                    auto yColor = ySecond["color"].as<std::vector<float>>();
                    for (size_t iC = 0; iC < std::min<size_t>(yColor.size(), 4); ++iC) {
                        label.color.data[iC] = yColor[iC] / 255.0f;
                    }
                }
                _classLabels.emplace_back(label);
            }
        }

        return NVDSINFER_SUCCESS;
    }
};

/** Implementation to Create a custom processor for DeepStream Triton
 * plugin(nvinferserver) to do custom extra input preprocess and custom
 * postprocess on triton based models.
 */
extern "C" {
IInferCustomProcessor*
Nvds3d_CreateLidarDetectionPostprocess(const char* config, uint32_t configLen)
{
    return new DS3DTritonLidarInferCustomPostProcess();
}
}