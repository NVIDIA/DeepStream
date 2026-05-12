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

#include <cuda_runtime_api.h>

// inlcude all ds3d hpp header files
#include <ds3d/common/hpp/datamap.hpp>
#include <ds3d/common/hpp/frame.hpp>
#include <ds3d/common/hpp/lidar_custom_process.hpp>
#include <ds3d/common/common.h>
#include <ds3d/common/hpp/yaml_config.hpp>
#include "infer_datatypes.h"
#include "nvdsinfer.h"
#include <string>
using namespace ds3d;

using namespace nvdsinferserver;

extern "C" IInferCustomPreprocessor* CreateInferServerCustomPreprocess();

#ifndef INFER_ASSERT
#define INFER_ASSERT(expr)                                                     \
    do {                                                                       \
        if (!(expr)) {                                                         \
            fprintf(stderr, "%s:%d ASSERT(%s) \n", __FILE__, __LINE__, #expr); \
            std::abort();                                                      \
        }                                                                      \
    } while (0)
#endif

#define checkCudaErrors(cudaErrorCode)                                                             \
    {                                                                                              \
        cudaError_t status = cudaErrorCode;                                                        \
        if (status != 0) {                                                                         \
            std::cout << "Cuda failure: " << cudaGetErrorString(status) << " at line " << __LINE__ \
                      << " in file " << __FILE__ << " error status: " << status << std::endl;      \
            abort();                                                                               \
        }                                                                                          \
    }

extern "C" void ds3dCustomCudaLidarNormalize(
    float* in, float* out, int points, float offset, float scale, cudaStream_t stream);

class NvInferServerCustomPreProcess : public IInferCustomPreprocessor {
public:
    ~NvInferServerCustomPreProcess() final = default;
    NvDsInferStatus preproc(
        GuardDataMap& dataMap, SharedIBatchArray batchArray, cudaStream_t stream) override
    {
        DS3D_UNUSED(stream);
        FrameGuard lidarFrame;
        const IOptions* inOptions = batchArray->getOptions();
        std::string key;
        if (inOptions && inOptions->hasValue(kLidarXYZI)) {
            INFER_ASSERT(inOptions->getValue<std::string>(kLidarXYZI, key)
                == NVDSINFER_SUCCESS);
        }
        ErrCode code = dataMap.getGuardData(key, lidarFrame);
        if(!isGood(code)){
            std::cout<< "dataMap getGuardData kLidarFrame failed" << std::endl;
            return NVDSINFER_TENSORRT_ERROR;
        }

        if (!_configParsed && inOptions->hasValue(kLidarInferenceParas)) {
            std::string inferConfigRaw;
            NvDsInferStatus ret =
                inOptions->getValue<std::string>(kLidarInferenceParas, inferConfigRaw);
            if (ret != NVDSINFER_SUCCESS) {
                std::cerr << "preprocess query key: " << kLidarInferenceParas << " failed.\n";
            }
            preprocessConfigParse(inferConfigRaw);
            _configParsed = true;
        }

        INFER_ASSERT(batchArray->getSize() > 1);
        /*
           Process all data into Model input tensors
           "points": FP32, dims [1, 204800, 4], GPU
           "num_points": INT32, dims [1], GPU
        */
        std::unordered_map<std::string, const IBatchBuffer*> tensorTable;
        for (uint32_t i = 0; i < batchArray->getSize(); ++i) {
            const IBatchBuffer* curBuf = batchArray->getBuffer(i);
            InferBufferDescription curDes = curBuf->getBufDesc();
            tensorTable[curDes.name] = curBuf;
        }
        const IBatchBuffer* buf = tensorTable["points"];
        //[0-255] to [0-1]
        InferBufferDescription des = buf->getBufDesc();
        int numPoints = std::accumulate(
            des.dims.d, des.dims.d + des.dims.numDims - 1, 1, [](int s, int i) { return s * i; });
        int elementSize = des.dims.d[des.dims.numDims - 1];
        INFER_ASSERT(elementSize == 4);

        // normalize intensity values
        float* frame = (float*)lidarFrame->base();
        if (isCpuMem(lidarFrame->memType())) {
            for (int j = 0; j < numPoints; j++) {
                float& val = frame[j * elementSize + 3];
                val = (val - _offsets) * _scaleFactor;
            }
            // copy preprocess data to GpuCuda or CpuCuda
            checkCudaErrors(cudaMemcpyAsync(
                buf->getBufPtr(0), lidarFrame->base(), lidarFrame->bytes(), cudaMemcpyDefault,
                stream));
        } else {
            ds3dCustomCudaLidarNormalize(
                frame, (float*)buf->getBufPtr(0), numPoints, _offsets, _scaleFactor, stream);
            checkCudaErrors(cudaGetLastError());
        }

        //add the second input.
        buf = tensorTable["num_points"];
        unsigned int points_size = numPoints;
        checkCudaErrors(cudaMemcpyAsync(
            buf->getBufPtr(0), &points_size, sizeof(unsigned int), cudaMemcpyDefault, stream));
        return NVDSINFER_SUCCESS;
    }

    void preprocessConfigParse(const std::string& configRaw)
    {
        YAML::Node node = YAML::Load(configRaw);
        if (node["preprocess"]) {
            auto preproc = node["preprocess"];
            if (preproc["scale_factor"]) {
                _scaleFactor = preproc["scale_factor"].as<float>();
            }
            if (preproc["offsets"]) {
                _offsets = preproc["offsets"].as<float>();
            }
        }
    }

private:
    bool _configParsed = false;
    float _scaleFactor = 1.0f;
    float _offsets = 0.0f;
};


extern "C" {
IInferCustomPreprocessor*
CreateInferServerCustomPreprocess()
{
    return new NvInferServerCustomPreProcess();
}
}
