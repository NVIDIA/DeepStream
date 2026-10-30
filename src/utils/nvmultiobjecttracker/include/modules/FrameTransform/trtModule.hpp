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

#ifndef TRTMODULE_HPP_
#define TRTMODULE_HPP_

#include <fstream>
#include <functional>
#include <memory>
#include <cstring>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include "nvdsinfer_context.h"
#include "modules/utility/cuda_common.hpp"
#include "modules/NvMultiObjectTracker/NvTrackerParams.hpp"

/** @brief Stores TensorRT model and engine configuration. */
class TRTConfigParams : public NvTrackerParams
{
public:
    /** @brief Construct a TRTConfigParams instance. */
    TRTConfigParams();

    /**
     * @brief Construct a TRTConfigParams instance.
     * @param[in] configFilePath Path to the tracker configuration file.
     * @param[in] moduleName Configuration module name.
     * @param[in] submoduleName Configuration submodule name.
     */
    TRTConfigParams(const std::string &configFilePath, const std::string moduleName, const std::string submoduleName);

    /**
     * @brief Construct a TRTConfigParams instance.
     * @param[in] fn Parsed YAML node.
     */
    TRTConfigParams(const YAML::Node &fn);

    /**
     * @brief Load configuration values from a YAML node.
     * @param[in] fn Parsed YAML node.
     */
    void read(const YAML::Node &fn) override;

    int m_maxBatchSize; //!< Maximum batch size for reid network.
    int m_inputOrder; //!< Input dimension order: 0: NCHW, 1: NHWC.
    int m_width; //!< Input image patch width.
    int m_height; //!< Input image patch height.
    int m_channel; //!< Input image patch channel.
    int m_gpuId; //!< Gpu id.
    std::vector<int> m_inferDims; //!< Reid input dimension.
    size_t m_workspaceSize; //!< Work space for reid network in MiB.
    std::string m_tltEncodedModelFilePath; //!< Reid network tao model path.
    std::string m_tltModelKey; //!< Tao model key.
    std::string m_onnxFilePath; //!< Reid network onnx model path.
    std::string m_engineFilePath; //!< Reid engine path.
    std::string m_calibrationTableFilePath; //!< Reid int8 calibration table file path.
    nvinfer1::DataType m_networkMode; //!< Reid network precision.
    //!< Integer 0: FP32 1: INT8 2: FP16.
    int m_builderOptimizationLevel; //!< Engine builder optimization level.
    int m_debugVerboseLevel; //!< Debug message verbose mode level.
};

/** @brief ReID int8 calibrator used for TensorRT int8 precision inference. */
class TrackerInt8EntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator
{
public:
    /**
     * @brief Construct a calibrator that reads a serialized calibration table.
     * @param[in] calibrationTableFile Path to the calibration-table file.
     */
    TrackerInt8EntropyCalibrator(std::string calibrationTableFile)
        : m_CalibrationTableFile(calibrationTableFile)
    {}

    /** @brief Release resources owned by the TrackerInt8EntropyCalibrator instance. */
    ~TrackerInt8EntropyCalibrator() {}

    /** @brief Report the calibrator batch size. */
    int getBatchSize() const noexcept override { return 0; }

    /**
     * @brief Provide a calibration batch when available.
     * @param[in,out] bindings TensorRT input and output bindings.
     * @param[in] names TensorRT binding names.
     * @param[in] nbBindings Number of TensorRT bindings.
     */
    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override { return false; }

    /**
     * @brief Read the calibration table file into a buffer and return a pointer to it.
     * @param[out] length Calibration-cache size to populate.
     * @return Pointer to the cache data, or null when the cache is empty.
     */
    const void*
    readCalibrationCache(size_t& length) noexcept override
    {
        m_CalibrationCache.clear();
        std::ifstream input(m_CalibrationTableFile, std::ios::binary);
        input >> std::noskipws;
        if (input.good())
            copy(
                std::istream_iterator<char>(input), std::istream_iterator<char>(),
                std::back_inserter(m_CalibrationCache)
            );

        length = m_CalibrationCache.size();
        return length ? m_CalibrationCache.data() : nullptr;
    }

    /**
     * @brief Accept a generated TensorRT calibration cache.
     * @param[in] cache Calibration-cache bytes to write.
     * @param[out] length Calibration-cache size to populate.
     */
    void writeCalibrationCache(const void* cache, size_t length) noexcept override {}

private:
    std::string m_CalibrationTableFile;
    std::vector<char> m_CalibrationCache;
};

/** @brief TRT module to perform various deep learning inference. */
class TrtModule
{
public:
    /** @brief Construct a TrtModule instance. */
    TrtModule();

    /**
     * @brief Create TensorRT inference engine; if engine does not exist, serialize from model.
     * @param[in] onnxProfileFunc Callback that sets the ONNX optimization profile before the engine is built.
     * @param[in] setRuntimeDimFunc Callback that sets runtime dimensions for dynamic input shapes.
     */
    void initializeModel(
        std::function<void(
            nvinfer1::INetworkDefinition *network, nvinfer1::IBuilder *builder,
            nvinfer1::IBuilderConfig *config, int batchSize
        )> onnxProfileFunc,
        std::function<void(TrtModule*, const std::vector<int>& sizes)> setRuntimeDimFunc
    );

    /** @brief Build a TensorRT engine from the configured network model. */
    void createInferEngine();

    /**
     * @brief Load ONNX model into network.
     * @param[out] network Network with ONNX model loaded.
     * @param[in] logger TensorRT logger.
     * @param[out] parser ONNX parser to populate.
     */
    void parseOnnxModel(
        nvinfer1::INetworkDefinition &network, nvinfer1::ILogger &logger, std::unique_ptr<nvonnxparser::IParser>& parser
    );

    /**
     * @brief Decode TAO encrypted model.
     * @param[in] builder TensorRT engine builder.
     * @param[in] config TensorRT builder configuration.
     * @return Decoded engine.
     */
    nvinfer1::ICudaEngine* decodeTltModel(nvinfer1::IBuilder* const builder, nvinfer1::IBuilderConfig* const config);

    /**
     * @brief Load reid TRT engine.
     * @return True if the engine file was loaded, otherwise the engine needs to be rebuilt.
     */
    bool loadTRTEngine();

    /**
     * @brief Submit network input for inference.
     * @param[in] bindings Input and output tensor addresses for the network.
     * @param[in] batchSize Runtime batch size for each dynamic input.
     * @param[in] cuStream CUDA stream that runs the GPU work.
     */
    void submitInput(std::vector<void *>& bindings, const std::vector<int>& batchSize, cudaStream_t& cuStream);

    /** @brief Print TRT engine information. */
    void printEngineInfo();

    /**
     * @brief Set the CUDA stream used for TensorRT inference.
     * @param[in] cuStream CUDA stream used for the operation.
     */
    void setCudaStream(cudaStream_t cuStream) { m_cuStream = cuStream; };

    TRTConfigParams m_params; //!< TRT module config.
    std::shared_ptr<nvinfer1::ICudaEngine> m_engine; //!< TensorRT engine for reid.
    std::shared_ptr<nvinfer1::IExecutionContext> m_context; //!< TensorRT execution context.
    cudaStream_t m_cuStream; //!< Cuda stream to run all GPU tasks.
protected:
    std::shared_ptr<nvinfer1::IRuntime> m_runtime; //!< TensorRT runtime used to deserialize engines.
    std::shared_ptr<TrackerInt8EntropyCalibrator> m_calibrator; //!< TensorRT int8 calibrator.
    std::string m_absModelFilePath; //!< Absolute model file path.
    std::function<void(
        nvinfer1::INetworkDefinition *network, nvinfer1::IBuilder *builder, nvinfer1::IBuilderConfig *config,
        int batchSize)> m_onnxProfileFunc; //!< Custom function to perform profiling for dynamic shapes required by TRT.
    std::function<void(TrtModule *, const std::vector<int>&)> m_setRuntimeDimFunc; //!< Custom function to set runtime dimension required by TRT.
};

#endif // TRTMODULE_HPP_
