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

#include "modules/FrameTransform/trtModule.hpp"
#include "nvdsinfer_logger.h"
#include "nvdsinfer_tlt.h"
#include "nvdsinfer_model_builder.h"

using std::cout;
using std::endl;

/**
 * @brief Resolve a path to an absolute path.
 * @param[in] path Path to resolve.
 * @param[out] absPath Resolved absolute path to populate.
 * @return True when the path was resolved successfully.
 */
static bool getAbsolutePath(const std::string& path, std::string& absPath)
{
    char buf[PATH_MAX + 1];
    if (realpath(path.c_str(), buf))
    {
        absPath = std::string(buf);
        return true;
    }
    else
    {
        return false;
    }
}

/** @brief Construct a TrtModule instance. */
TrtModule::TrtModule()
    : m_cuStream(nullptr)
{
}

/* Tracker configuration uses TensorRT's numbering (0 FP32, 1 FP16, 2 INT8),
 * which differs from NvDsInferNetworkMode. Never cast between these enums. */
static NvDsInferNetworkMode getInferNetworkMode(nvinfer1::DataType mode)
{
    switch (mode)
    {
        case nvinfer1::DataType::kFLOAT: return NvDsInferNetworkMode_FP32;
        case nvinfer1::DataType::kHALF: return NvDsInferNetworkMode_FP16;
        case nvinfer1::DataType::kINT8: return NvDsInferNetworkMode_INT8;
        default: throw std::runtime_error("Unsupported tracker network precision");
    }
}

static void validatePrecisionConfig(const TRTConfigParams& params)
{
    (void)getInferNetworkMode(params.m_networkMode);
#if NV_TENSORRT_MAJOR >= 11
    if (!params.m_calibrationTableFilePath.empty())
    {
        throw std::runtime_error("TensorRT 11 does not support tracker INT8 calibration caches. "
            "Use a prepared INT8 Q/DQ model and remove calibrationTableFile.");
    }
#endif
}

void TrtModule::resetModel()
{
    // Deleters also retain these dependencies when callers keep shared copies.
    m_context.reset();
    m_engine.reset();
    m_runtime.reset();
#if NV_TENSORRT_MAJOR < 11
    m_calibrator.reset();
#endif
}

void TrtModule::validateEngineIO() const
{
    if (!m_engine || m_engine->getNbIOTensors() == 0)
    {
        throw std::runtime_error("Tracker engine has no input/output tensors");
    }
    bool hasInput = false;
    bool hasOutput = false;
    for (int i = 0; i < m_engine->getNbIOTensors(); ++i)
    {
        const char* name = m_engine->getIOTensorName(i);
        if (!name || !*name)
        {
            throw std::runtime_error("Tracker engine has an unnamed input/output tensor");
        }
        const auto mode = m_engine->getTensorIOMode(name);
        hasInput |= mode == nvinfer1::TensorIOMode::kINPUT;
        hasOutput |= mode == nvinfer1::TensorIOMode::kOUTPUT;
        const auto dims = m_engine->getTensorShape(name);
        if (mode == nvinfer1::TensorIOMode::kNONE || dims.nbDims < 0 ||
            m_engine->getTensorDataType(name) != nvinfer1::DataType::kFLOAT ||
            m_engine->getTensorLocation(name) != nvinfer1::TensorLocation::kDEVICE)
        {
            throw std::runtime_error(std::string("Tracker requires named FP32 device I/O tensors: ") + name);
        }
        for (int profile = 0; profile < std::max(1, m_engine->getNbOptimizationProfiles()); ++profile)
        {
            if (m_engine->getTensorFormat(name, profile) != nvinfer1::TensorFormat::kLINEAR)
            {
                throw std::runtime_error(std::string("Tracker requires linear I/O tensors: ") + name);
            }
        }
        for (int d = 0; d < dims.nbDims; ++d)
        {
            if (dims.d[d] < -1)
            {
                throw std::runtime_error(std::string("Invalid tracker tensor shape: ") + name);
            }
        }
    }
    if (!hasInput || !hasOutput)
    {
        throw std::runtime_error("Tracker engine requires both input and output tensors");
    }
}

/** @brief Print TensorRT engine tensor metadata. */
void TrtModule::printEngineInfo()
{
    const std::map<nvinfer1::TensorIOMode, const char*> iomodeStr = {
        {nvinfer1::TensorIOMode::kINPUT, "kINPUT"},
        {nvinfer1::TensorIOMode::kOUTPUT, "kOUTPUT"},
        {nvinfer1::TensorIOMode::kNONE, "kNONE"}
    };

    if (m_params.m_debugVerboseLevel > 1)
    {
        std::vector<std::string> dataTypeNames = {"kFLOAT", "kHALF", "kINT8", "kINT32",
            "kBOOL", "kUINT8", "kFP8", "kBF16", "kINT64", "kINT4", "kFP4"};
        cout << "Printing TensorRT engine information" << endl;
        for (int32_t i = 0; i < m_engine->getNbIOTensors(); i++)
        {
            const char* tensorName = m_engine->getIOTensorName(i);
            const nvinfer1::TensorIOMode iomode = m_engine->getTensorIOMode(tensorName);
            const nvinfer1::Dims dims = m_engine->getTensorShape(tensorName);
            const nvinfer1::DataType dtype = m_engine->getTensorDataType(tensorName);
            int32_t totalSize = 1;
            cout << "Tensor index: " << i << ", IO: " << iomodeStr.at(iomode) << ", name: " << tensorName <<
                ", size: [";
            for (int32_t di=0; di < dims.nbDims; di++)
            {
                cout << dims.d[di];
                totalSize *= dims.d[di];
                if (di < dims.nbDims-1)
                {
                    cout << ", ";
                }
            }
            cout << "] = " << totalSize << ", data type: " << ( int(dtype)<int(dataTypeNames.size()) ? dataTypeNames.at(int(dtype)) : "others") << endl;
        }
    }
}

/**
 * @brief Create or load the configured inference model.
 * @param[in] onnxProfileFunc Callback that sets the ONNX optimization profile before the engine is built.
 * @param[in] setRuntimeDimeFunc Callback that sets runtime tensor dimensions.
 */
void TrtModule::initializeModel(
    std::function<void(
        nvinfer1::INetworkDefinition *network, nvinfer1::IBuilder *builder,
        nvinfer1::IBuilderConfig *config, int batchSize
    )> onnxProfileFunc,
    std::function<void(TrtModule*, const std::vector<int>&)> setRuntimeDimeFunc
)
{
    /// Load engine file and create context.
    /// TODO: Support a relative file path.
    /// Check existing engine if failed create a new one in constructor stage.
    m_onnxProfileFunc = onnxProfileFunc;
    m_setRuntimeDimFunc = setRuntimeDimeFunc;
    resetModel();
    // Validate cache requests too; TensorRT 11 must not silently ignore them.
    validatePrecisionConfig(m_params);
    try
    {
        if(!loadTRTEngine())
        {
            cout << "[NvMultiObjectTracker] Load engine failed. Create engine again." << endl;
            /// Engine file doesn't exist. Create from model file.
            createInferEngine();
        }

        validateEngineIO();
        auto engine = m_engine;
        m_context = std::shared_ptr<nvinfer1::IExecutionContext>(
            engine->createExecutionContext(), [engine](nvinfer1::IExecutionContext* context) {
                delete context;
            });
        if (!m_context)
        {
            throw std::runtime_error("Failed to create tracker execution context");
        }
        printEngineInfo();
    }
    catch (...)
    {
        resetModel();
        throw;
    }
}

/**
 * @brief Parse the configured ONNX model into a TensorRT network.
 * @param[out] network Network with ONNX model loaded.
 * @param[in] logger TensorRT logger.
 * @param[out] parser ONNX parser to populate.
 */
void TrtModule::parseOnnxModel(
    nvinfer1::INetworkDefinition &network, nvinfer1::ILogger &logger, std::unique_ptr<nvonnxparser::IParser>& parser
)
{
    if ( !getAbsolutePath(m_params.m_onnxFilePath, m_absModelFilePath) )
    {
        throw std::runtime_error("ONNX file does not exist");
    }

    parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(network, logger));
    if (!parser || !parser->parseFromFile(m_params.m_onnxFilePath.c_str(),
        (int)nvinfer1::ILogger::Severity::kWARNING))
    {
        throw std::runtime_error("Fail to parse ONNX network");
    }
}

/**
 * @brief Decode the configured TAO model into a TensorRT engine.
 * @param[in] builder TensorRT engine builder.
 * @param[in] config TensorRT builder configuration.
 */
std::shared_ptr<nvinfer1::ICudaEngine> TrtModule::decodeTltModel(
    nvinfer1::IBuilder* const builder, nvinfer1::IBuilderConfig* const config
)
{
    validatePrecisionConfig(m_params);
    /// Check if UFF file is provided.
    if ( !getAbsolutePath(m_params.m_tltEncodedModelFilePath, m_absModelFilePath) )
    {
        throw std::runtime_error("TAO model file does not exist");
    }

    NvDsInferContextInitParams initParams;
    memset(&initParams, 0, sizeof (initParams));
    initParams.gpuID = m_params.m_gpuId;
    initParams.dlaCore = -1;
    initParams.useDLA = 0;
    initParams.maxBatchSize = m_params.m_maxBatchSize;
    initParams.netInputOrder = (NvDsInferTensorOrder) m_params.m_inputOrder;
    /// CHW.
    initParams.inferInputDims = {(uint) m_params.m_channel, (uint) m_params.m_height, (uint) m_params.m_width};
    initParams.networkMode = getInferNetworkMode(m_params.m_networkMode);
    initParams.workspaceSize = m_params.m_workspaceSize;

    std::strncpy(initParams.int8CalibrationFilePath, m_params.m_calibrationTableFilePath.c_str(), _PATH_MAX-1);
    std::strncpy(initParams.tltEncodedModelFilePath, m_params.m_tltEncodedModelFilePath.c_str(), _PATH_MAX-1);
    std::strncpy(initParams.tltModelKey, m_params.m_tltModelKey.c_str(), _MAX_STR_LENGTH-1);

    // Keep the complete SDK wrapper as the owner of this aliasing shared pointer.
    // A raw-engine callback cannot transfer its TensorRT runtime ownership.
    std::shared_ptr<nvdsinfer::TrtEngine> owner =
        nvdsinfer::buildTltModelEngine(builder, config, &initParams);
    if (!owner)
    {
        throw std::runtime_error("Fail to decode TAO model");
    }

    return std::shared_ptr<nvinfer1::ICudaEngine>(owner, &owner->engine());
}

/** @brief Build and serialize a TensorRT inference engine. */
void TrtModule::createInferEngine()
{
    resetModel();
    validatePrecisionConfig(m_params);
    CUDA_CHECK(cudaGetDevice(&m_params.m_gpuId));

    /// Create engine builder.
    nvinfer1::ILogger* logger = getLogger();
    if (!logger)
    {
        logger = gTrtLogger.get();
    }
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(*logger));
    if (!builder)
    {
        throw std::runtime_error("Failed to create inference builder");
    }

    const std::map<nvinfer1::DataType, std::string> networkModeMap{
        {nvinfer1::DataType::kINT8, "int8"}, {nvinfer1::DataType::kHALF, "fp16"}, {nvinfer1::DataType::kFLOAT, "fp32"}
    };
#if NV_TENSORRT_MAJOR < 11
    if (m_params.m_networkMode == nvinfer1::DataType::kINT8 && !builder->platformHasFastInt8())
    {
        /// When int8 not supported, try fp16.
        /// If still not supported, use fp32.
        m_params.m_networkMode = builder->platformHasFastFp16() ? \
            nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT;
        CLOG(
            std::string(
                "Precision int8 is not supported. Using " + networkModeMap.at(m_params.m_networkMode) + " instead."
            ).c_str()
        );
    }
    else if (m_params.m_networkMode == nvinfer1::DataType::kHALF && !builder->platformHasFastFp16())
    {
        /// When fp16 not supported, use fp32.
        m_params.m_networkMode = nvinfer1::DataType::kFLOAT;
        CLOG("Precision fp16 is not supported. Using fp32 instead.");
    }
#endif

    std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());

    if (!config)
    {
        throw std::runtime_error("Failed to create inference builder configuration");
    }

    if (!m_params.m_tltEncodedModelFilePath.empty())
    {
        /// Use nvdsinferutils API to decode ETLT model and generate engine.
        m_engine = decodeTltModel(builder.get(), config.get());
    }
    else
    {
        /// Use native UFF or ONNX model to generate engine.
        nvinfer1::NetworkDefinitionCreationFlags netDefFlags = 0;

        /// UFF model uses implicit batch, and ONNX model uses explicit batch.
#if NV_TENSORRT_MAJOR < 11
        if (!m_params.m_onnxFilePath.empty())
            netDefFlags |= (1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
#endif

        /// Create network and load UFF/ONNX models into network.
        std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(netDefFlags));
        if (!network)
        {
            throw std::runtime_error("Failed to create inference network");
        }

        std::unique_ptr<nvonnxparser::IParser> onnxParser(nullptr);
        if (!m_params.m_onnxFilePath.empty())
        {
            parseOnnxModel(*network, *logger, onnxParser);
            if (m_onnxProfileFunc != nullptr)
            {
                m_onnxProfileFunc(network.get(), builder.get(), config.get(), m_params.m_maxBatchSize);
            }
        }
        else
        {
            throw std::runtime_error(
                "Network model file not provided. The supported format are ONNX, ETLT (Nvidia TAO model)."
            );
        }

#if NV_TENSORRT_MAJOR >= 11
        nvdsinfer::ExplicitBuildParams options;
        options.networkMode = getInferNetworkMode(m_params.m_networkMode);
        options.int8CalibrationFilePath = m_params.m_calibrationTableFilePath;
        if (nvdsinfer::validateTrt11Precision(*network, options) != NVDSINFER_SUCCESS)
        {
            throw std::runtime_error("Tracker model does not implement the requested TensorRT 11 precision; "
                "prepare a typed FP16/mixed source ONNX offline with NVIDIA ModelOpt AutoCast "
                "(tools/modelopt/prepare_model.py), preserving tracker I/O types");
        }
#else
        if (m_params.m_networkMode == nvinfer1::DataType::kINT8)
        {
            m_calibrator = std::make_shared<TrackerInt8EntropyCalibrator>(m_params.m_calibrationTableFilePath);
            if (!m_calibrator)
            {
                throw std::runtime_error("Invalid calibrator for INT8 precision");
            }
            config->setFlag(nvinfer1::BuilderFlag::kINT8);
            config->setInt8Calibrator(m_calibrator.get());
        }
        else if (m_params.m_networkMode == nvinfer1::DataType::kHALF)
        {
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
#endif

        config->setBuilderOptimizationLevel(m_params.m_builderOptimizationLevel);

        cout << "[NvMultiObjectTracker] Begin building engine for model: " << m_absModelFilePath << endl;

        std::shared_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(*logger));
        if(!runtime)
        {
            throw std::runtime_error("Failed to create runtime engine");
        }

        std::unique_ptr<nvinfer1::IHostMemory> serializedEngine(
            builder->buildSerializedNetwork(*network, *config));
        if (!serializedEngine)
        {
            throw std::runtime_error("Tracker serialized engine build failed");
        }
        m_engine = std::shared_ptr<nvinfer1::ICudaEngine>(
            runtime->deserializeCudaEngine(serializedEngine->data(), serializedEngine->size()),
            [runtime](nvinfer1::ICudaEngine* engine) { delete engine; });
        m_runtime = std::move(runtime);
    }

    if (!m_engine)
    {
        throw std::runtime_error("Engine build failed");
    }
    validateEngineIO();

    /// Save engine to hard drive.
    std::unique_ptr<nvinfer1::IHostMemory> serializedModel(m_engine->serialize());
    if (!serializedModel)
    {
        throw std::runtime_error("Engine serialization failed");
    }

    std::stringstream modelStream;
    modelStream.seekg(0, modelStream.beg);
    modelStream.write(static_cast<const char*>(serializedModel->data()), serializedModel->size());

    std::ofstream outFile;

    /// Engine file format: xxx.uff_bx_gpux_<mode>.engine.
    std::string absEngineFilePath = m_absModelFilePath + "_b" + std::to_string(m_params.m_maxBatchSize) + \
        "_gpu" + std::to_string(m_params.m_gpuId) + "_" + networkModeMap.at(m_params.m_networkMode) + ".engine";
    outFile.open(absEngineFilePath, std::ios::binary);
    outFile << modelStream.rdbuf();
    const bool saved = outFile.good();
    outFile.close();
    if (saved)
        cout << "[NvMultiObjectTracker] Serialized plan file cached at location: " << absEngineFilePath << endl;
    else
        CLOG(std::string("Unable to cache tracker engine at: " + absEngineFilePath).c_str());
}

/** @brief Load a serialized TensorRT engine. */
bool TrtModule::loadTRTEngine()
{
    resetModel();
    validatePrecisionConfig(m_params);
    std::string planFilePath;
    if ( !getAbsolutePath(m_params.m_engineFilePath, planFilePath) )
    {
        CLOG(std::string("Engine file does not exist: " + m_params.m_engineFilePath).c_str());
        return false;
    }

    /// Reading engine file into in memory.
    cout << "[NvMultiObjectTracker] Loading TRT Engine for model: " << planFilePath << endl;
    std::stringstream trtModelStream;
    trtModelStream.seekg(0, trtModelStream.beg);
    std::ifstream cache(planFilePath, std::ios::binary);
    if (!cache.good())
    {
        CLOG("Load TRT engine error");
        return false;
    }
    trtModelStream << cache.rdbuf();
    cache.close();

    /// Calculating model size.
    trtModelStream.seekg(0, std::ios::end);
    const std::streamoff modelSize = trtModelStream.tellg();
    if (modelSize <= 0)
    {
        CLOG("Tracker engine file is empty or unreadable");
        return false;
    }
    trtModelStream.seekg(0, std::ios::beg);
    std::shared_ptr<void> modelMem(malloc(modelSize), free);
    if (!modelMem)
    {
        CLOG("Failed to allocate memory to create engine");
        return false;
    }
    trtModelStream.read((char*) modelMem.get(), modelSize);
    if (!trtModelStream)
    {
        CLOG("Failed to read tracker engine file");
        return false;
    }

    /// Create engine.
    nvinfer1::ILogger* logger = getLogger();
    if (!logger)
    {
        logger = gTrtLogger.get();
    }
    std::shared_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(*logger));
    if (!runtime)
    {
        CLOG("Failed to create runtime engine");
        return false;
    }

    m_engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(
        modelMem.get(), modelSize), [runtime](nvinfer1::ICudaEngine* engine) { delete engine; });
    if (!m_engine)
    {
        CLOG("Engine build failed");
        return false;
    }
    m_runtime = std::move(runtime);
    try
    {
        validateEngineIO();
    }
    catch (...)
    {
        resetModel();
        throw;
    }

    cout << "[NvMultiObjectTracker] Loading Complete!" << endl;

    return true;
}

/**
 * @brief Bind tensors and enqueue TensorRT inference.
 * @param[in] bindings Input and output tensor addresses for the network.
 * @param[in] batchSize Runtime batch size for each dynamic input.
 * @param[in] cuStream CUDA stream that runs the GPU work.
 */
void TrtModule::submitInput(std::vector<void *>& bindings, const std::vector<int>& batchSize, cudaStream_t& cuStream)
{
    if (!m_engine || !m_context)
    {
        throw std::runtime_error("Tracker inference model is not initialized");
    }
    if (m_engine->getNbIOTensors() != int32_t(bindings.size()))
    {
        throw std::runtime_error("The number of input bindings does not match engine");
    }

    for (int32_t i = 0, e = m_engine->getNbIOTensors(); i < e; i++)
    {
        auto const name = m_engine->getIOTensorName(i);
        bool ok = m_context->setTensorAddress(name, bindings.data()[i]);
        if (!ok)
        {
            throw std::runtime_error("Fail to set tensor address");
        }
    }

    /// ONNX created engine has explicit batch and uses enqueueV3.
    /// When input batch size is not specified, specifyDimInputIndList is not empty, need to update batch size in input dim.
    if (m_setRuntimeDimFunc)
    {
        if (batchSize.empty())
            throw std::runtime_error("Tracker runtime shape parameters are empty");
        m_setRuntimeDimFunc(this, batchSize);
    }
    // allInputDimensionsSpecified does not support the name-based shape API.
    // Query resolved inputs without adding synchronization or shape inference.
    for (int32_t i = 0, e = m_engine->getNbIOTensors(); i < e; ++i)
    {
        const char* name = m_engine->getIOTensorName(i);
        if (m_engine->getTensorIOMode(name) != nvinfer1::TensorIOMode::kINPUT)
            continue;
        const auto dims = m_context->getTensorShape(name);
        if (dims.nbDims < 0)
            throw std::runtime_error("Invalid tracker input shape");
        for (int d = 0; d < dims.nbDims; ++d)
        {
            if (dims.d[d] < 0)
                throw std::runtime_error("Not all tracker input dimensions are specified");
        }
    }

    if (!m_context->enqueueV3(cuStream))
    {
        throw std::runtime_error("Tracker inference enqueue failed");
    }
}

/** @brief Construct a TRTConfigParams instance. */
TRTConfigParams::TRTConfigParams()
{
    m_maxBatchSize = 1;
    m_inputOrder = 0;
    m_width = 64;
    m_height = 128;
    m_channel = 3;
    m_inferDims = {3, 128, 64};
    m_gpuId = 0;
    m_workspaceSize = 20;
    m_networkMode = nvinfer1::DataType(0);
    m_builderOptimizationLevel = 3;
    m_debugVerboseLevel = 0;
}

/**
 * @brief Construct a TRTConfigParams instance.
 * @param[in] configFilePath Path to the tracker configuration file.
 * @param[in] moduleName Configuration module name.
 * @param[in] submoduleName Configuration submodule name.
 */
TRTConfigParams::TRTConfigParams(
    const std::string &configFilePath, const std::string moduleName, const std::string submoduleName
)
{
    /// Set the default values.
    *this = TRTConfigParams();

    YAML::Node configRoot = getConfigRoot(configFilePath);
    if (!submoduleName.empty())
    {
        /// The submodule contains model specific parameters to overwrite the basic module.
        this->read(configRoot[moduleName][submoduleName]);
    }
    else
    {
        /// Submodule is not used, so read from module.
        this->read(configRoot[moduleName]);
    }
    m_onnxFilePath = NvTrackerParams::resolvePathRelativeToConfig(configFilePath, m_onnxFilePath);
    m_tltEncodedModelFilePath =
        NvTrackerParams::resolvePathRelativeToConfig(configFilePath, m_tltEncodedModelFilePath);
    m_engineFilePath = NvTrackerParams::resolvePathRelativeToConfig(configFilePath, m_engineFilePath);
    m_calibrationTableFilePath =
        NvTrackerParams::resolvePathRelativeToConfig(configFilePath, m_calibrationTableFilePath);
    /// Interpret inferDims according to the input order.
    if (m_inputOrder == 0)
    {
        m_channel = m_inferDims[0], m_height = m_inferDims[1], m_width = m_inferDims[2];
    }
    else if(m_inputOrder == 1)
    {
        m_height = m_inferDims[0], m_width = m_inferDims[1], m_channel = m_inferDims[2];
    }
    if (m_channel != 3 || m_width <= 0 || m_height <= 0)
    {
        throw std::runtime_error("inferDims channel should be 3, widht and height should be positive int");
    }
}

/**
 * @brief Construct a TRTConfigParams instance.
 * @param[in] fn Parsed YAML node.
 */
TRTConfigParams::TRTConfigParams(const YAML::Node &fn)
{
    read(fn);
}

/**
 * @brief Load configuration values from a YAML node.
 * @param[in] fn Parsed YAML node.
 */
void TRTConfigParams::read(const YAML::Node &fn)
{
    if(!fn.size())
    {
        return;
    }

    for (YAML::const_iterator fit = fn.begin(); fit != fn.end(); ++fit)
    {
        std::string paramKey = fit->first.as<std::string>();

        if (paramKey == "debugVerboseLevel")
        {
            m_debugVerboseLevel = (int)fit->second.as<float>();
        }
        else if (paramKey == "batchSize")
        {
            int maxBatchSize = (int)fit->second.as<float>();
            if (maxBatchSize > 0)
            {
                m_maxBatchSize = maxBatchSize;
            }
            else
            {
                cout << "maxBatchSize should be positive int; use default batch size" <<endl;
            }
        }
        else if (paramKey == "workspaceSize")
        {
            int workspaceSize = (int)fit->second.as<float>();
            if (workspaceSize > 0)
            {
                /// Input is MiB, m_workspaceSize is B.
                m_workspaceSize = size_t(workspaceSize);
            }
            else
            {
                cout << "workspaceSize should be positive int; use default size" <<endl;
            }
        }
        else if (paramKey == "inferDims")
        {
            m_inferDims.clear();
            /// Interpret later because the input order is unknown at this time.
            for (std::size_t i=0; i<fit->second.size(); i++)
            {
                m_inferDims.push_back((int)fit->second[i].as<float>());
            }
            /// Check network dimension size.
            if (m_inferDims.size() != 3)
            {
                throw std::runtime_error("inferDims should be list of positive int with length 3");
            }
        }
        else if (paramKey == "inputOrder")
        {
            int inputOrder = (int)fit->second.as<float>();
            if (inputOrder == 0 || inputOrder == 1)
            {
                m_inputOrder = inputOrder;
            }
            else
            {
                throw std::runtime_error("inputOrder should be 0 or 1");
            }
        }
        else if (paramKey == "networkMode")
        {
            int networkMode = (int)fit->second.as<float>();
            if (networkMode >= 0 && networkMode <= 2)
            {
                m_networkMode = nvinfer1::DataType(networkMode);
            }
            else
            {
                throw std::runtime_error("networkMode should be 0, 1 or 2");
            }
        }
        else if (paramKey == "onnxFile")
        {
            m_onnxFilePath = fit->second.as<std::string>();
        }
        else if (paramKey == "tltEncodedModel")
        {
            m_tltEncodedModelFilePath = fit->second.as<std::string>();
        }
        else if (paramKey == "tltModelKey")
        {
            m_tltModelKey = fit->second.as<std::string>();
        }
        else if (paramKey == "modelEngineFile")
        {
            m_engineFilePath = fit->second.as<std::string>();
        }
        else if (paramKey == "calibrationTableFile")
        {
            m_calibrationTableFilePath = fit->second.as<std::string>();
        }
        else if (paramKey == "builderOptimizationLevel")
        {
            m_builderOptimizationLevel = (int)fit->second.as<float>();
        }

    }
}
