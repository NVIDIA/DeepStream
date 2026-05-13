/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "lidar_postprocess.hpp"

using namespace ds3d;
namespace dsis = nvdsinferserver;

#ifndef INFER_ASSERT
#define INFER_ASSERT(expr)                                                     \
    do {                                                                       \
        if (!(expr)) {                                                         \
            fprintf(stderr, "%s:%d ASSERT(%s) \n", __FILE__, __LINE__, #expr); \
            std::abort();                                                      \
        }                                                                      \
    } while (0)
#endif

#define checkCudaErrors(status)                                   \
{                                                                 \
  if (status != 0)                                                \
  {                                                               \
    std::cout << "Cuda failure: " << cudaGetErrorString(status)   \
              << " at line " << __LINE__                          \
              << " in file " << __FILE__                          \
              << " error status: " << status                      \
              << std::endl;                                       \
              abort();                                            \
    }                                                             \
}

extern "C" dsis::IInferCustomProcessor* CreateInferServerCustomProcess(
    const char* config, uint32_t configLen);

static const std::vector<vec4f> kClassColors = {
    {{1.0f, 1.0f, 0, 1.0f}},  // yellow
    {{1.0f, 0, 0, 1.0f}},  // red
    {{0, 0, 1.0f, 1.0f}},  // blue
};

class NvInferServerCustomProcess : public dsis::IInferCustomProcessor {
public:
    ~NvInferServerCustomProcess() final = default;

    void supportInputMemType(dsis::InferMemType& type) final { type = dsis::InferMemType::kGpuCuda; }

    bool requireInferLoop() const final { return false; }

    NvDsInferStatus extraInputProcess(
        const std::vector<dsis::IBatchBuffer*>&
            primaryInputs,  // primary tensor(image) has been processed
        std::vector<dsis::IBatchBuffer*>& extraInputs, const dsis::IOptions* options) override
    {
        return NVDSINFER_SUCCESS;
    }

    NvDsInferStatus inferenceDone(
        const dsis::IBatchArray* batchArray, const dsis::IOptions* inOptions) override
    {
        std::string filterConfigRaw = "";
        if(!_configFileParserFlag) {
            if (inOptions->hasValue(kLidarInferenceParas)) {
                INFER_ASSERT(
                    inOptions->getValue<std::string>(kLidarInferenceParas, filterConfigRaw) ==
                    NVDSINFER_SUCCESS);
            }
            CustomPostprocessConfigParse(filterConfigRaw);
            _configFileParserFlag = true;
        }
        LOG_INFO("infer result layerSize:%d", batchArray->getSize());
        float* boxOutput = NULL;
        int* boxNum = NULL;
        for (uint32_t i = 0; i < batchArray->getSize(); ++i) {
            const auto& buf = batchArray->getBuffer(i);
            assert(buf);
            int bufLen = buf->getTotalBytes();
            const dsis::InferBufferDescription& desc = buf->getBufDesc();
            LOG_INFO("%s, %d, %d, %d\n", desc.name.c_str(),
              (int)desc.dataType, (int)desc.memType, bufLen);
            if(desc.name == _modelOutputLayers[0]) {
                boxOutput = (float*)buf->getBufPtr(0);
            } else if(desc.name == _modelOutputLayers[1]) {
                boxNum = (int*)buf->getBufPtr(0);
            }
        }

        std::vector<Lidar3DBbox> nmsPred;
        int num_obj = *boxNum;
        LOG_INFO("infer box num:%d", num_obj);
        std::vector<Lidar3DBbox> res;
        for (int i = 0; i < num_obj; i++) {
            //score > scoreThreshold
            float boxScore = boxOutput[i * 9 + 8];
            if( boxScore - _scoreThresh > 1e-6) {
                auto Bb = Lidar3DBbox(
                boxOutput[i * 9],
                boxOutput[i * 9 + 1],
                boxOutput[i * 9 + 2],
                boxOutput[i * 9 + 3],
                boxOutput[i * 9 + 4],
                boxOutput[i * 9 + 5],
                boxOutput[i * 9 + 6],
                boxOutput[i * 9 + 7],
                boxOutput[i * 9 + 8]
                );
                res.push_back(Bb);
            }
        }

        ParseCustomBatchedNMS(res, _nmsIouThresh, nmsPred, _preNmsTopN);

        for(size_t i=0; i<nmsPred.size(); i++) {
            LOG_INFO(
                "%s, %f, %f, %f, %f, %f, %f, %f, %f\n", _classLabels[nmsPred[i].cid].c_str(),
                nmsPred[i].centerX, nmsPred[i].centerY, nmsPred[i].centerZ, nmsPred[i].dx,
                nmsPred[i].dy, nmsPred[i].dz, nmsPred[i].yaw, nmsPred[i].score);
            uint32_t strLen = _classLabels[nmsPred[i].cid].size();
            strLen = std::min<uint32_t>(strLen, DS3D_MAX_LABEL_SIZE - 1);
            strncpy(nmsPred[i].labels, _classLabels[nmsPred[i].cid].c_str(), strLen);
            if (nmsPred[i].cid < (int)kClassColors.size()) {
                nmsPred[i].bboxColor = kClassColors[nmsPred[i].cid];
            }
        }
        res.clear();


        //after nms, add output to datamap
        abiRefDataMap* refDataMap = nullptr;
        if (inOptions->hasValue(kLidarRefDataMap)) {
            INFER_ASSERT(inOptions->getObj(kLidarRefDataMap, refDataMap) == NVDSINFER_SUCCESS);
        }
        GuardDataMap dataMap(*refDataMap);
        int boxSize = sizeof(Lidar3DBbox);
        int outbufLen = boxSize * nmsPred.size();
        uint8_t *outBuf = new uint8_t[outbufLen];
        for(size_t i = 0; i < nmsPred.size(); i++){
            memcpy(outBuf + i * boxSize, &nmsPred[i], boxSize);
        }

        Shape shape{3, {1, (int)nmsPred.size(), sizeof(Lidar3DBbox)}};
        FrameGuard frame = impl::WrapFrame<uint8_t, FrameType::kCustom>(outBuf, outbufLen, shape,
                    MemType::kCpu, 0, [outBuf](void*) {delete []outBuf;});

        ErrCode code = dataMap.setGuardData(kLidar3DBboxRawData, frame);
        if(!isGood(code)){
            LOG_WARNING("dataMap setGuardData kLidarOutput failed");
        }

        return NVDSINFER_SUCCESS;
    }

    /** override function
     * Receiving errors if anything wrong inside lowlevel lib
     */
    void notifyError(NvDsInferStatus s) final
    {

    }

private:
    NvDsInferStatus CustomPostprocessConfigParse(const std::string& filterConfigRaw)
    {
        YAML::Node node = YAML::Load(filterConfigRaw);
        if(node["postprocess_nms_iou_thresh"]) {
            _nmsIouThresh = node["postprocess_nms_iou_thresh"].as<float>();
        }

        if(node["postprocess_pre_nms_top_n"]) {
            _preNmsTopN = node["postprocess_pre_nms_top_n"].as<int>();
        }

        if(node["postprocess_score_thresh"]) {
            _scoreThresh = node["postprocess_score_thresh"].as<float>();
        }

        auto outputModel = node["model_outputs"];
        if (outputModel) {
            for (auto item : outputModel) {
                if(item["name"]) {
                    std::string tmp = item["name"].as<std::string>().c_str();
                    _modelOutputLayers.push_back(tmp);
                }
            }
        }

        auto yamlClassLables = node["labels"];
        if (yamlClassLables) {
            for (YAML::const_iterator it=yamlClassLables.begin();it!=yamlClassLables.end();++it) {
                std::string seq = it->as<std::string>();
                _classLabels.push_back(seq);
            }
        }

        return NVDSINFER_SUCCESS;
    }
private:
    bool _configFileParserFlag = false;
    float _nmsIouThresh = 0.2;
    int _preNmsTopN = 4096;
    float _scoreThresh = 0.4;
    std::vector<std::string> _modelOutputLayers;
    std::vector<std::string> _classLabels;
};

/** Implementation to Create a custom processor for DeepStream Triton
 * plugin(nvinferserver) to do custom extra input preprocess and custom
 * postprocess on triton based models.
 */
extern "C" {
dsis::IInferCustomProcessor*
CreateInferServerCustomProcess(const char* config, uint32_t configLen)
{
    return new NvInferServerCustomProcess();
}
}