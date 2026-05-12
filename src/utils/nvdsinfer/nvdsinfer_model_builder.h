/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef __NVDSINFER_MODEL_BUILDER_H__
#define __NVDSINFER_MODEL_BUILDER_H__

#include <stdarg.h>
#include <algorithm>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <unordered_map>

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>

#include <nvdsinfer_custom_impl.h>
#include "nvdsinfer_func_utils.h"
#include "nvdsinfer_tlt.h"

/* This file provides APIs for building models from ONNX files. It
 * also defines an interface where users can provide custom model parsers for
 * custom networks. A helper class (TrtEngine) written on top of TensorRT's
 * nvinfer1::ICudaEngine is also defined in this file.
 *
 * These interfaces/APIs are used by NvDsInferContextImpl class. */

namespace nvdsinfer {

using NvDsInferCudaEngineGetFcnDeprecated = decltype(&NvDsInferCudaEngineGet);

static const size_t kWorkSpaceSize = 450 * 1024 * 1024; // 450MB

/**
 * ModelParser base. Any model parser implementation must inherit from the
 * IModelParser interface.
 */
class BaseModelParser : public IModelParser
{
public:
    BaseModelParser(const NvDsInferContextInitParams& params,
        const std::shared_ptr<DlLibHandle>& dllib)
        : m_ModelParams(params), m_LibHandle(dllib) {}
    virtual ~BaseModelParser() {}
    virtual bool isValid() const = 0;

private:
    DISABLE_CLASS_COPY(BaseModelParser);

protected:
    NvDsInferContextInitParams m_ModelParams;
    std::shared_ptr<DlLibHandle> m_LibHandle;
};

/**
 * Implementation of ModelParser for ONNX models derived from BaseModelParser.
 * Manages resources internally required for parsing ONNX models.
 */
class OnnxModelParser : public BaseModelParser
{
public:
    OnnxModelParser(const NvDsInferContextInitParams& initParams,
        const std::shared_ptr<DlLibHandle>& handle = nullptr)
        : BaseModelParser(initParams, handle),
          m_ModelName(initParams.onnxFilePath) {}
    ~OnnxModelParser() override = default;
    bool isValid() const override { return !m_ModelName.empty(); }
    const char* getModelName() const override { return m_ModelName.c_str(); }
    NvDsInferStatus parseModel(nvinfer1::INetworkDefinition& network) override;
    bool hasFullDimsSupported() const override { return true; }

private:
    std::string m_ModelName;

protected:
    std::unique_ptr<nvonnxparser::IParser> m_OnnxParser;
};

/**
 * Implementation of ModelParser for custom models. This implementation will look
 * for the function symbol "NvDsInferCreateModelParser" in the custom library
 * handle passed to it. It will call the NvDsInferCreateModelParser to get
 * an instance of the IModelParser implementation required to parse the user's
 * custom model.
 */
class CustomModelParser : public BaseModelParser
{
public:
    CustomModelParser(const NvDsInferContextInitParams& initParams,
        const std::shared_ptr<DlLibHandle>& handle);

    ~CustomModelParser() {};

    bool isValid() const override
    {
        return (bool)m_CustomParser;
    }

    const char* getModelName() const override
    {
        return isValid() ? safeStr(m_CustomParser->getModelName()) : "";
    }

    NvDsInferStatus parseModel(nvinfer1::INetworkDefinition& network) override;
    bool hasFullDimsSupported() const override
    {
        return m_CustomParser->hasFullDimsSupported();
    }

private:
    std::unique_ptr<IModelParser> m_CustomParser;
};

/** Forward declaration of TrtModelBuilder class. */
class TrtModelBuilder;

/**
 * Holds build parameters common to implicit batch dimension/full dimension
 * networks.
 */
struct BuildParams
{
    using TensorIOFormat =
        std::tuple<nvinfer1::DataType, nvinfer1::TensorFormats>;
    using LayerDevicePrecision =
      std::tuple<nvinfer1::DataType, nvinfer1::DeviceType>;

    size_t workspaceSize = kWorkSpaceSize;
    NvDsInferNetworkMode networkMode = NvDsInferNetworkMode_FP32;
    std::string int8CalibrationFilePath;
    int dlaCore = -1;
    std::vector<TensorIOFormat> inputFormats;
    std::unordered_map<std::string, TensorIOFormat> outputFormats;
    std::unordered_map<std::string, LayerDevicePrecision> layerDevicePrecisions;

public:
    virtual ~BuildParams(){};
    virtual NvDsInferStatus configBuilder(TrtModelBuilder& builder) = 0;
    virtual bool sanityCheck() const;
};

/**
 * Holds build parameters required for implicit batch dimension network.
 */
struct ImplicitBuildParams : public BuildParams
{
    int maxBatchSize = 0;
    std::vector<nvinfer1::Dims> inputDims;

private:
    NvDsInferStatus configBuilder(TrtModelBuilder& builder) override;

    bool sanityCheck() const override;
};

using ProfileDims = std::array<nvinfer1::Dims,
    nvinfer1::EnumMax<nvinfer1::OptProfileSelector>()>;

/**
 * Holds build parameters required for full dimensions network.
 */
struct ExplicitBuildParams : public BuildParams
{
    // profileSelector, dims without batchSize
    // each input must have 3 selector MIN/OPT/MAX for profile0,
    // doesn't support multiple profiles
    std::vector<ProfileDims> inputProfileDims;
    int minBatchSize = 1;
    int optBatchSize = 1;
    int maxBatchSize = 1;
    NvDsInferTensorOrder inputOrder = NvDsInferTensorOrder_kNCHW;

private:
    NvDsInferStatus configBuilder(TrtModelBuilder& builder) override;
    bool sanityCheck() const override;
};

/**
 * Helper class written on top of nvinfer1::ICudaEngine.
 */
class TrtEngine
{
public:
    TrtEngine(std::unique_ptr<nvinfer1::ICudaEngine>&& engine, int dlaCore = -1)
        : m_Engine(std::move(engine)), m_DlaCore(dlaCore) {}

    TrtEngine(std::unique_ptr<nvinfer1::ICudaEngine>&& engine,
        const std::shared_ptr<nvinfer1::IRuntime>& runtime, int dlaCore = -1,
        const std::shared_ptr<DlLibHandle>& dlHandle = nullptr,
        nvinfer1::IPluginFactory* pluginFactory = nullptr);

    ~TrtEngine();

    bool hasDla() const { return m_DlaCore >= 0; }
    int getDlaCore() const { return m_DlaCore; }

    NvDsInferStatus getImplicitLayersInfo(
        std::vector<NvDsInferBatchDimsLayerInfo>& layersInfo);
    NvDsInferStatus getFullDimsLayersInfo(
        int profileIdx, std::vector<NvDsInferBatchDimsLayerInfo>& layersInfo);
    NvDsInferStatus getLayerInfo(int idx, NvDsInferLayerInfo& layer);

    void printEngineInfo();

    nvinfer1::ICudaEngine& engine()
    {
        assert(m_Engine);
        return *m_Engine;
    }

    nvinfer1::ICudaEngine* operator->()
    {
        assert(m_Engine);
        return m_Engine.get();
    }

private:
    DISABLE_CLASS_COPY(TrtEngine);

    std::shared_ptr<nvinfer1::IRuntime> m_Runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> m_Engine;
    std::shared_ptr<DlLibHandle> m_DlHandle;
    nvinfer1::IPluginFactory* m_RuntimePluginFactory = nullptr;
    int m_DlaCore = -1;

    friend bool ::NvDsInferCudaEngineGetFromTltModel( nvinfer1::IBuilder * const builder,
        nvinfer1::IBuilderConfig * const builderConfig,
        const NvDsInferContextInitParams * const initParams,
        nvinfer1::DataType dataType,
        nvinfer1::ICudaEngine *& cudaEngine);
};

/**
 * Helper class to build models and generate the TensorRT ICudaEngine required
 * for inference. This class will parse models using the nvdsinfer::IModelParser
 * interface and then build the model engine using nvinfer1::IBuilder's
 * BuilderConfig APIs based on initialization parameters passed to
 * NvDsInferContext. Alternatively, this class can also deserialize an existing
 * serialized engine to generate the ICudaEngine.
 */
class TrtModelBuilder
{
public:
    TrtModelBuilder(int gpuId, nvinfer1::ILogger& logger,
        const std::shared_ptr<DlLibHandle>& dlHandle = nullptr, bool isEngineFile = false);

    ~TrtModelBuilder() {
        m_Parser.reset();
    }

    void setInt8Calibrator(std::unique_ptr<nvinfer1::IInt8Calibrator>&& calibrator)
    {
        m_Int8Calibrator = std::move(calibrator);
    }

    /* Populate INetworkDefinition by parsing the model, build the engine and
     * return it as TrtEngine instance. Also, returns a suggested path for
     * writing the serialized engine to.
     *
     * Suggested path has the following format:
     * suggested path = [modelName]_b[#batchSize]_[#device]_[#dataType].engine
     */
    std::unique_ptr<TrtEngine> buildModel(
        const NvDsInferContextInitParams& initParams,
        std::string& suggestedPathName);

    /* Builds the engine from an already populated INetworkDefinition based on
     * the BuildParams passed to it. Returns the engine in the form of TrtEngine
     * instance.
     */
    std::unique_ptr<TrtEngine> buildEngine(
        nvinfer1::INetworkDefinition& network, BuildParams& options);

    /* Serialize engine to file
     */
    NvDsInferStatus serializeEngine(
        const std::string& path, nvinfer1::ICudaEngine& engine);

    /* Deserialize engine from file
     */
    std::unique_ptr<TrtEngine> deserializeEngine(
        const std::string& path, int dla = -1);

private:
    /* Parses a model file using an IModelParser implementation for
     * ONNX formats or from custom IModelParser implementation.
     */
    NvDsInferStatus buildNetwork(const NvDsInferContextInitParams& initParams);

    /* build cudaEngine from Netwwork, be careful for implicitBatch and
     * explicitBatch.
     */
    std::unique_ptr<TrtEngine> buildEngine();

    /* Calls a custom library's implementaion of NvDsInferCudaEngineGet function
     * to get a built ICudaEngine. */
    std::unique_ptr<TrtEngine> getCudaEngineFromCustomLib(
            NvDsInferCudaEngineGetFcnDeprecated cudaEngineGetDeprecatedFcn,
            NvDsInferEngineCreateCustomFunc cudaEngineGetFcn,
            const NvDsInferContextInitParams& initParams,
            NvDsInferNetworkMode &networkMode);


    /* config builder options */
    NvDsInferStatus configCommonOptions(BuildParams& params);
    NvDsInferStatus configImplicitOptions(ImplicitBuildParams& params);
    NvDsInferStatus configExplicitOptions(ExplicitBuildParams& params);

    std::unique_ptr<BuildParams> createImplicitParams(
        const NvDsInferContextInitParams& initParams);
    std::unique_ptr<BuildParams> createDynamicParams(
        const NvDsInferContextInitParams& initParams);
    void initCommonParams(
        BuildParams& params, const NvDsInferContextInitParams& initParams);

    DISABLE_CLASS_COPY(TrtModelBuilder);

    int m_GpuId = 0;
    nvinfer1::ILogger& m_Logger;
    std::shared_ptr<DlLibHandle> m_DlLib;
    std::shared_ptr<BaseModelParser> m_Parser;
    std::unique_ptr<BuildParams> m_Options;
    std::unique_ptr<nvinfer1::IBuilder> m_Builder;
    std::unique_ptr<nvinfer1::IBuilderConfig> m_BuilderConfig;
    std::unique_ptr<nvinfer1::INetworkDefinition> m_Network;
    std::shared_ptr<nvinfer1::IInt8Calibrator> m_Int8Calibrator;

    friend class BuildParams;
    friend class ImplicitBuildParams;
    friend class ExplicitBuildParams;

    friend bool ::NvDsInferCudaEngineGetFromTltModel( nvinfer1::IBuilder * const builder,
        nvinfer1::IBuilderConfig * const builderConfig,
        const NvDsInferContextInitParams * const initParams,
        nvinfer1::DataType dataType,
        nvinfer1::ICudaEngine *& cudaEngine);
};

} // end of namespace nvdsinfer

#endif
