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
{
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

    if(!loadTRTEngine())
    {
        cout << "[NvMultiObjectTracker] Load engine failed. Create engine again." << endl;
        /// Engine file doesn't exist.
        /// Create from model file.
        createInferEngine();
    }

    m_context = std::shared_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());

    printEngineInfo();

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
    if (!parser->parseFromFile(m_params.m_onnxFilePath.c_str(),
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
nvinfer1::ICudaEngine* TrtModule::decodeTltModel(
    nvinfer1::IBuilder* const builder, nvinfer1::IBuilderConfig* const config
)
{
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
    initParams.networkMode = (NvDsInferNetworkMode) m_params.m_networkMode;
    initParams.workspaceSize = m_params.m_workspaceSize;

    std::strncpy(initParams.int8CalibrationFilePath, m_params.m_calibrationTableFilePath.c_str(), _PATH_MAX-1);
    std::strncpy(initParams.tltEncodedModelFilePath, m_params.m_tltEncodedModelFilePath.c_str(), _PATH_MAX-1);
    std::strncpy(initParams.tltModelKey, m_params.m_tltModelKey.c_str(), _MAX_STR_LENGTH-1);

    nvinfer1::ICudaEngine* engineTlt = nullptr;
    if (! NvDsInferCudaEngineGetFromTltModel(builder,
        config, &initParams, nvinfer1::DataType(0), engineTlt))
    {
        throw std::runtime_error("Fail to decode TAO model");
    }

    return engineTlt;
}

/** @brief Build and serialize a TensorRT inference engine. */
void TrtModule::createInferEngine()
{
    cudaGetDevice(&m_params.m_gpuId);

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

    std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());

    if (!config)
    {
        throw std::runtime_error("Failed to create inference builder configuration");
    }

    if (!m_params.m_tltEncodedModelFilePath.empty())
    {
        /// Use nvdsinferutils API to decode ETLT model and generate engine.
        m_engine = std::shared_ptr<nvinfer1::ICudaEngine>(decodeTltModel(builder.get(), config.get()));
    }
    else
    {
        /// Use native UFF or ONNX model to generate engine.
        nvinfer1::NetworkDefinitionCreationFlags netDefFlags = 0;

        /// UFF model uses implicit batch, and ONNX model uses explicit batch.
        if (!m_params.m_onnxFilePath.empty())
            netDefFlags |= (1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));

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

        config->setBuilderOptimizationLevel(m_params.m_builderOptimizationLevel);

        cout << "[NvMultiObjectTracker] Begin building engine for model: " << m_absModelFilePath << endl;

        std::shared_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(*logger));
        if(!runtime)
        {
            throw std::runtime_error("Failed to create runtime engine");
        }

        auto serializedEngine = builder->buildSerializedNetwork(*network, *config);

        m_engine.reset (runtime->deserializeCudaEngine(serializedEngine->data(), serializedEngine->size()));
    }

    if (!m_engine)
    {
        throw std::runtime_error("Engine build failed");
    }

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
    outFile.open(absEngineFilePath);
    outFile << modelStream.rdbuf();
    outFile.close();

    cout << "[NvMultiObjectTracker] Serialized plan file cached at location: " << absEngineFilePath << endl;
}

/** @brief Load a serialized TensorRT engine. */
bool TrtModule::loadTRTEngine()
{
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
    std::ifstream cache(planFilePath);
    if (!cache.good())
    {
        CLOG("Load TRT engine error");
        return false;
    }
    trtModelStream << cache.rdbuf();
    cache.close();

    /// Calculating model size.
    trtModelStream.seekg(0, std::ios::end);
    const int modelSize = trtModelStream.tellg();
    trtModelStream.seekg(0, std::ios::beg);
    std::shared_ptr<void> modelMem(malloc(modelSize), free);
    if (!modelMem)
    {
        CLOG("Failed to allocate memory to create engine");
        return false;
    }
    trtModelStream.read((char*) modelMem.get(), modelSize);

    /// Create engine.
    nvinfer1::ILogger* logger = getLogger();
    if (!logger)
    {
        logger = gTrtLogger.get();
    }
    m_runtime = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(*logger));
    if (!m_runtime)
    {
        CLOG("Failed to create runtime engine");
        return false;
    }

    m_engine = std::shared_ptr<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine( modelMem.get(), modelSize));
    if (!m_engine)
    {
        CLOG("Engine build failed");
        return false;
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
        m_setRuntimeDimFunc(this, batchSize);
    }
    /// Make sure all input dimensions are specified.
    if (!m_context->allInputDimensionsSpecified())
    {
        throw std::runtime_error("Not all dimensions are specified in explicit dimension mode");
    }

    m_context->enqueueV3(cuStream);
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
