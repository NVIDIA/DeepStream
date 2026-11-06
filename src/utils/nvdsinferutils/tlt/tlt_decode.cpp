/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <openssl/evp.h>
#include <sstream>
//WINDOWS
#ifndef _WIN32
#include <unistd.h>
#endif

#include "NvInferRuntime.h"

#include "nvdsinfer_context.h"
#include "nvdsinfer_model_builder.h"
#include "nvdsinfer_tlt.h"

using namespace nvdsinfer;

static const int __NONCE_SIZE = 16;
static const int __KEY_SIZE = 32;
static const int __CHUNK_SIZE = 1024;
static const int __PBKDF2_ITER = 100000;
static const int __SALT_LEN = 32;

/* Custom unique_ptr subclass with deleter functions for TensorRT objects. */
template <class T>
class NvDsInferUniquePtr : public std::unique_ptr<T, void (*)(T *)>
{
    public:
        NvDsInferUniquePtr(T * t = nullptr) :
            std::unique_ptr<T, void (*)(T *)>(t, [](T *t){if (t) t->destroy();})
            {}
};

#if NV_TENSORRT_MAJOR < 11
class NvDsInferInt8Calibrator : public nvinfer1::IInt8EntropyCalibrator2
{
public:
    NvDsInferInt8Calibrator(std::string calibrationTableFile)
        : m_CalibrationTableFile(calibrationTableFile) {}

    ~NvDsInferInt8Calibrator()
    {
    }

    int
    getBatchSize() const noexcept override
    {
        return 0;
    }

    bool
    getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override
    {
        return false;
    }

    /* Reads calibration table file contents into a buffer and returns a pointer
     * to the buffer.
     */
    const void*
    readCalibrationCache(size_t& length) noexcept override
    {
        m_CalibrationCache.clear();
        std::ifstream input(m_CalibrationTableFile, std::ios::binary);
        input >> std::noskipws;
        if (input.good())
            copy(std::istream_iterator<char>(input),
                std::istream_iterator<char>(),
                std::back_inserter(m_CalibrationCache));

        length = m_CalibrationCache.size();
        return length ? m_CalibrationCache.data() : nullptr;
    }

    void
    writeCalibrationCache(const void* cache, size_t length) noexcept override
    {
    }

private:
    std::string m_CalibrationTableFile;
    std::vector<char> m_CalibrationCache;
};
#endif

static int read_node_name_length(std::ifstream& in_stream)
{
    // Retrieve input node names baked into exported tlt file
    // Read input node name string length
    unsigned char* node_name_size_b = (unsigned char*) malloc(4);
    in_stream.read((char*)node_name_size_b, 4);
    int node_name_size_i = int(node_name_size_b[0]);
    for(int i = 1; i < 4; i++)
    {
        node_name_size_i += int(node_name_size_b[i] << i);
    }
    free(node_name_size_b);

    return node_name_size_i;
}


static bool
decode(std::ifstream& in_stream, std::ostream &out_stream, const char* password)
{
    // Retrieve input node names baked into exported tlt file
    // Read input node name string length
    int node_name_size_i = read_node_name_length(in_stream);

    // Read input node name and register with the parser
    unsigned char* input_node_name = (unsigned char*) malloc(node_name_size_i);
    in_stream.read((char*)input_node_name, node_name_size_i);

    // Derive key
    std::vector<unsigned char> key(__KEY_SIZE);
    PKCS5_PBKDF2_HMAC(password, strlen(password),
                      (const unsigned char*)("\xcb\x3c\xdc\x2e\x85\x86\x89\xda\x90\x85\x63\x05\x6a"
                                             "\x00\xdb\xed\x09\xac\x07\x6c\x78\x55\x43\xc4\x61\x0f"
                                             "\x06\xd3\x1b\x53\xa7\x32"),
                      __SALT_LEN, __PBKDF2_ITER, EVP_sha256(), __KEY_SIZE, key.data());

    // Read nonce value from file
    std::vector<unsigned char> nonce(__NONCE_SIZE);
    in_stream.read((char*)nonce.data(), __NONCE_SIZE);

    // Initialize cipher
    EVP_CIPHER_CTX *decrypt_ctx = EVP_CIPHER_CTX_new();
    EVP_DecryptInit_ex(decrypt_ctx, EVP_aes_256_ctr(), NULL, key.data(), nonce.data());

    // Read ciphertext chunk
    std::vector<unsigned char> ciphertext(__CHUNK_SIZE);
    in_stream.read((char*) ciphertext.data(), __CHUNK_SIZE);

    std::vector<unsigned char> plaintext(__CHUNK_SIZE);
    int p_len = 0;

    // Decode to file chunk by chunk, until end of encoded input file
    while(in_stream.gcount() == __CHUNK_SIZE)
    {
        EVP_DecryptUpdate(decrypt_ctx, plaintext.data(), &p_len, ciphertext.data(), __CHUNK_SIZE);

        out_stream.write((const char *) plaintext.data(), p_len);
        in_stream.read((char*) ciphertext.data(), __CHUNK_SIZE);
    }

    // Decode final chunk to file
    EVP_DecryptUpdate(decrypt_ctx, plaintext.data(), &p_len, ciphertext.data(), in_stream.gcount());
    out_stream.write((const char *) plaintext.data(), p_len);
    out_stream.flush();

    // Clean up cipher
    EVP_CIPHER_CTX_cleanup(decrypt_ctx);

    free(input_node_name);

    return true;
}

/**
 * Implementation of ModelParser for ONNX models in .etlt files.
 * Derived from OnnxModelParser.
 * Decodes the model from the input file and encyption key.
 */
class EtltOnnxModelParser : public OnnxModelParser
{
public:
    EtltOnnxModelParser(const NvDsInferContextInitParams& initParams,
            const std::shared_ptr<DlLibHandle>& handle)
        : OnnxModelParser(initParams, handle),
          m_ModelEtlt(initParams.tltEncodedModelFilePath),
          m_ModelKey(initParams.tltModelKey) {}

    NvDsInferStatus parseModel(nvinfer1::INetworkDefinition& network) override;

    const char* getModelName() const override { return m_ModelEtlt.c_str(); }
    bool isValid() const override { return !m_ModelEtlt.empty(); }

private:
    std::string m_ModelEtlt;
    std::string m_ModelKey;
};

NvDsInferStatus
EtltOnnxModelParser::parseModel(nvinfer1::INetworkDefinition& network)
{
    std::ifstream in_stream(m_ModelEtlt);
    if (!in_stream.good())
    {
        fprintf(stderr, "%s: Failed to open TLT encoded model file %s\n", __func__,
                safeStr(m_ModelEtlt));
        return NVDSINFER_CONFIG_FAILED;
    }

    std::ostringstream decoded_model_stream;

    if (!decode(in_stream, decoded_model_stream, m_ModelKey.c_str())) {
        fprintf(stderr, "%s: Failed to decode etlt file.\n", __func__);
        return NVDSINFER_CONFIG_FAILED;
    }

    m_OnnxParser.reset(nvonnxparser::createParser(network, *gTrtLogger));

    std::string decoded_model = decoded_model_stream.str();
    if (!m_OnnxParser->parse(decoded_model.c_str(), decoded_model.length()))
    {
        fprintf(stderr, "%s: Failed to parse ONNX model\n", __func__);
        return NVDSINFER_CONFIG_FAILED;
    }

    return NVDSINFER_SUCCESS;
}


namespace nvdsinfer {

#ifdef _WIN32
__declspec(dllexport)
#else
__attribute__ ((visibility ("default")))
#endif
std::unique_ptr<TrtEngine>
buildTltModelEngine(nvinfer1::IBuilder * const builder,
        nvinfer1::IBuilderConfig * const builderConfig,
        const NvDsInferContextInitParams * const initParams)
{
    /* Check if UFF file is provided. */
    if (string_empty(initParams->tltEncodedModelFilePath))
    {
        fprintf(stderr, "%s: TLT encoded model file path not provided\n",
                __func__);
        return nullptr;
    };

#if NV_TENSORRT_MAJOR >= 11
    if (!string_empty(initParams->int8CalibrationFilePath))
    {
        fprintf(stderr, "%s: TensorRT 11 does not support legacy INT8 calibration "
                "caches. Use an explicitly quantized ONNX model with Q/DQ nodes "
                "and remove int8-calib-file.\n", __func__);
        return nullptr;
    }
#endif

    std::shared_ptr<DlLibHandle> dlHandle;
    if (!string_empty(initParams->customLibPath))
    {
        dlHandle = std::make_shared<DlLibHandle>(initParams->customLibPath);
        if (dlHandle.get() == nullptr)
        {
            fprintf (stderr, "%s: Failed to open custom lib for parsing model\n",
                    __func__);
            return nullptr;
        }
    }

    std::unique_ptr<TrtModelBuilder> modelBuilder =
        std::make_unique<TrtModelBuilder>(initParams->gpuID, *gTrtLogger, dlHandle);
#if NV_TENSORRT_MAJOR < 11
    if (!string_empty(initParams->int8CalibrationFilePath) &&
        file_accessible(initParams->int8CalibrationFilePath))
    {
        auto calibrator = std::make_unique<NvDsInferInt8Calibrator>(
            initParams->int8CalibrationFilePath);
        modelBuilder->setInt8Calibrator(std::move(calibrator));
    }
#endif
    modelBuilder->m_BuilderConfig.reset(builderConfig);
    modelBuilder->m_Builder.reset(builder);
    // These objects belong to the caller. Release them on every exit, including
    // parser/configuration failures, before destroying the temporary builder.
    auto releaseBorrowed = [](TrtModelBuilder* temporary) {
        temporary->m_BuilderConfig.release();
        temporary->m_Builder.release();
    };
    std::unique_ptr<TrtModelBuilder, decltype(releaseBorrowed)> borrowedGuard(
        modelBuilder.get(), releaseBorrowed);

    /* Read input node name string length */
    std::ifstream in_stream(initParams->tltEncodedModelFilePath);
    if (!in_stream.good())
    {
        fprintf(stderr, "%s: Failed to open TLT encoded model file %s\n", __func__,
                safeStr(initParams->tltEncodedModelFilePath));
        return nullptr;
    }

    int node_name_size_i = read_node_name_length(in_stream);

    /* node_name_size_i indicates if the underlying model is onnx (0) or uff (> 0) */
    if(node_name_size_i)
    {
        fprintf (stderr, "%s: UFF model support has been deprecated.\n",
                 __func__);
        return nullptr;
    }
    else
    {
        auto parser = std::make_unique<EtltOnnxModelParser>(*initParams, dlHandle);
        assert(parser);

        if (!parser->isValid())
        {
            return nullptr;
        }

        std::unique_ptr<BuildParams> buildOptions;

        //onnx parser only supports explicit batch dim
#if NV_TENSORRT_MAJOR >= 11
        const uint32_t explicitBatch = 0;
#else
        const auto explicitBatch = 1U << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
#endif
        buildOptions = modelBuilder->createDynamicParams(*initParams);

        std::unique_ptr<nvinfer1::INetworkDefinition> network (modelBuilder->m_Builder->createNetworkV2(explicitBatch));
        assert(network);

        NvDsInferStatus status = parser->parseModel(*network);
        if (status != NVDSINFER_SUCCESS) {
            dsInferError("Failed to build network, error in model parsing.");
            return nullptr;
        }

        // Keep the decoded model's types. TRT 11 precision validation is shared
        // with normal ONNX builds; prepare the source ONNX offline for FP16.
        modelBuilder->m_Network = std::move(network);
        modelBuilder->m_Options = std::move(buildOptions);
        modelBuilder->m_Parser = std::move(parser);
    }

    std::unique_ptr<TrtEngine> engine = modelBuilder->buildEngine();

    if (!engine)
    {
        dsInferError("Failed to build engine from TLT model.");
        return nullptr;
    }

    return engine;
}

} // namespace nvdsinfer

extern "C"
#ifdef _WIN32
__declspec(dllexport)
#else
__attribute__ ((visibility ("default")))
#endif
bool NvDsInferCudaEngineGetFromTltModel(nvinfer1::IBuilder * const builder,
        nvinfer1::IBuilderConfig * const builderConfig,
        const NvDsInferContextInitParams * const initParams,
        nvinfer1::DataType dataType,
        nvinfer1::ICudaEngine *& cudaEngine)
{
#if NV_TENSORRT_MAJOR >= 11
    // This legacy API cannot transfer ownership of the runtime required by the
    // returned engine. The SDK's TLT path keeps both in the TrtEngine wrapper.
    cudaEngine = nullptr;
    fprintf(stderr, "%s: TensorRT 11 TLT models must be built through the "
            "DeepStream inference context to retain runtime ownership.\n",
            __func__);
    return false;
#else
    std::unique_ptr<TrtEngine> engine =
        buildTltModelEngine(builder, builderConfig, initParams);
    if (!engine)
    {
        cudaEngine = nullptr;
        return false;
    }
    cudaEngine = engine->m_Engine.release();
    return true;
#endif
}
