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

/**
 * @file trt_encoder.cpp
 * @brief TensorRT direct inference for RADIO-CLIP embeddings (no Triton, no gRPC)
 * 
 * Similar to the PyTorch direct path in cosmos-embed1, this uses TensorRT
 * engine directly for local inference without any server/client overhead.
 */

#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
#include <sys/stat.h>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <gst/gst.h>
#include "trt_encoder.h"
#if NV_TENSORRT_MAJOR >= 11
#include "nvdsinfer_model_builder.h"
#endif

using namespace nvinfer1;
using namespace nvonnxparser;

/* Use GStreamer debug category from main plugin */
GST_DEBUG_CATEGORY_EXTERN(gst_nvdsvisionencoder_debug);
#define GST_CAT_DEFAULT gst_nvdsvisionencoder_debug

/**
 * Logger for TensorRT
 */
class TRTLogger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            printf("[TRT-%d] %s\n", (int)severity, msg);
        }
    }
} gTRTLogger;

/**
 * Helper to check if file exists
 */
static bool FileExists(const std::string& path) {
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

/**
 * Get file modification time (seconds since epoch). Returns 0 on failure.
 */
static time_t GetFileModTime(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0)
        return 0;
    return st.st_mtime;
}

/**
 * Generate engine path from ONNX path following nvinfer convention:
 * TRT < 11: {onnx_base}_b{batchSize}_gpu{gpuId}_fp16.engine
 * TRT >= 11: version the offline-typed model policy so a previously automatic
 * conversion cache cannot bypass validation of the supplied model's types.
 */
static std::string GenerateEnginePath(const std::string& onnx_path, int batch_size) {
    std::string base = onnx_path;
    auto dot = base.rfind('.');
    if (dot != std::string::npos)
        base = base.substr(0, dot);

    int gpu_id = 0;
    cudaGetDevice(&gpu_id);

    return base + "_b" + std::to_string(batch_size)
               + "_gpu" + std::to_string(gpu_id)
#if NV_TENSORRT_MAJOR >= 11
               + "_trt" + std::to_string(NV_TENSORRT_MAJOR) + "_model_typed_v2.engine";
#else
               + "_fp16.engine";
#endif
}

/**
 * Sidecar metadata file for tracking which ONNX model an engine was built from.
 * Written alongside the engine (e.g. model.plan → model.plan.meta).
 * Format: single line with the absolute ONNX path used to build the engine.
 */
static std::string GetMetaPath(const std::string& engine_path) {
    return engine_path + ".meta";
}

static void WriteEngineMeta(const std::string& engine_path, const std::string& onnx_path) {
    std::string meta_path = GetMetaPath(engine_path);
    std::ofstream f(meta_path);
    if (f.is_open()) {
        f << onnx_path << std::endl;
        GST_DEBUG("Wrote engine meta: %s → source ONNX: %s",
                  meta_path.c_str(), onnx_path.c_str());
    }
}

static std::string ReadEngineMeta(const std::string& engine_path) {
    std::string meta_path = GetMetaPath(engine_path);
    std::ifstream f(meta_path);
    std::string stored_onnx;
    if (f.is_open())
        std::getline(f, stored_onnx);
    return stored_onnx;
}

/**
 * TensorRT encoder class
 */
class TRTEncoder {
public:
    TRTEncoder() : runtime_(nullptr), engine_(nullptr), context_(nullptr),
                   input_width_(0), input_height_(0), embedding_dim_(0),
                   max_batch_size_(1), d_output_buffer_(nullptr), stream_(nullptr) {}
    
    ~TRTEncoder() {
        Cleanup();
    }
    
    bool LoadEngine(const std::string& engine_path) {
        // Read engine file
        std::ifstream file(engine_path, std::ios::binary);
        if (!file.good()) {
            GST_ERROR("Failed to open engine file: %s", engine_path.c_str());
            return false;
        }
        
        file.seekg(0, std::ifstream::end);
        size_t size = file.tellg();
        file.seekg(0, std::ifstream::beg);
        
        std::vector<char> engine_data(size);
        file.read(engine_data.data(), size);
        file.close();
        
        GST_DEBUG("Loaded engine file: %s (%zu bytes)", engine_path.c_str(), size);
        
        // Create runtime and deserialize engine
        runtime_ = createInferRuntime(gTRTLogger);
        if (!runtime_) {
            GST_ERROR("Failed to create TensorRT runtime");
            return false;
        }
        
        engine_ = runtime_->deserializeCudaEngine(engine_data.data(), size);
        if (!engine_) {
            GST_ERROR("Failed to deserialize engine");
            return false;
        }
        
        // Create execution context
        context_ = engine_->createExecutionContext();
        if (!context_) {
            GST_ERROR("Failed to create execution context");
            return false;
        }
        
        // Parse input/output dimensions and tensor names
        if (!ParseEngineInfo()) {
            return false;
        }
        
        // Pre-allocate device output buffer (reused every Infer to avoid cudaMalloc/Free in hot path)
        size_t output_buf_size = (size_t)max_batch_size_ * embedding_dim_ * sizeof(float);
        if (cudaMalloc(&d_output_buffer_, output_buf_size) != cudaSuccess) {
            GST_ERROR("Failed to allocate device output buffer (%zu bytes)", output_buf_size);
            return false;
        }
        GST_DEBUG("Pre-allocated device output buffer: %zu bytes", output_buf_size);
        
        // Create CUDA stream
        cudaStreamCreate(&stream_);
        
        GST_DEBUG("Engine loaded successfully!");
        GST_DEBUG("  Input: %ux%u", input_width_, input_height_);
        GST_DEBUG("  Embedding dim: %u", embedding_dim_);
        
        return true;
    }
    
    bool BuildEngineFromONNX(const std::string& onnx_path, const std::string& engine_path, 
                              int batch_size = 1) {
        GST_DEBUG("Building TensorRT engine from ONNX...");
        GST_DEBUG("  ONNX: %s", onnx_path.c_str());
        GST_DEBUG("  Output: %s", engine_path.c_str());
        GST_DEBUG("  Batch size: %d", batch_size);

        // Create builder
        IBuilder* builder = createInferBuilder(gTRTLogger);
        if (!builder) {
            GST_ERROR("Failed to create builder");
            return false;
        }
        
        // TensorRT 11 networks are explicit-batch and strongly typed by default.
#if NV_TENSORRT_MAJOR >= 11
        INetworkDefinition* network = builder->createNetworkV2(0);
#else
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
#pragma GCC diagnostic pop
#endif
        if (!network) {
            GST_ERROR("Failed to create network");
            delete builder;
            return false;
        }
        
        // Create ONNX parser
        IParser* parser = createParser(*network, gTRTLogger);
        if (!parser) {
            std::cerr << "[TRTEncoder] Failed to create ONNX parser" << std::endl;
            delete network;
            delete builder;
            return false;
        }
        
        // Parse ONNX file
        if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int>(ILogger::Severity::kWARNING))) {
            GST_ERROR("Failed to parse ONNX file");
            for (int i = 0; i < parser->getNbErrors(); i++) {
                GST_ERROR("  Error %d: %s", i, parser->getError(i)->desc());
            }
            delete parser;
            delete network;
            delete builder;
            return false;
        }
        
        GST_DEBUG("ONNX parsed successfully");
        
        // For combined CLIP models (image+text): keep only the image
        // embedding output so TensorRT prunes the text encoder entirely.
        if (network->getNbOutputs() > 1) {
            g_print("nvdsvisionencoder [TRT]: Combined model detected "
                    "(%d outputs) — pruning to image_embedding only\n",
                    network->getNbOutputs());
            int kept = -1;
            for (int i = 0; i < network->getNbOutputs(); i++) {
                const char* name = network->getOutput(i)->getName();
                if (strstr(name, "image_embedding") || strstr(name, "image")) {
                    kept = i;
                    break;
                }
            }
            if (kept < 0) kept = 0;
            for (int i = network->getNbOutputs() - 1; i >= 0; i--) {
                if (i != kept) {
                    g_print("nvdsvisionencoder [TRT]:   unmark output '%s'\n",
                            network->getOutput(i)->getName());
                    network->unmarkOutput(*network->getOutput(i));
                }
            }
            g_print("nvdsvisionencoder [TRT]:   kept output '%s'\n",
                    network->getOutput(0)->getName());
        }
        
#if NV_TENSORRT_MAJOR >= 11
        // TensorRT 11 uses the model's declared types. Prepare the image graph
        // offline rather than rewriting it in this engine builder.
        nvdsinfer::ExplicitBuildParams precision;
        precision.networkMode = NvDsInferNetworkMode_FP16;
        if (nvdsinfer::validateTrt11Precision(*network, precision) != NVDSINFER_SUCCESS) {
            GST_ERROR("Vision encoder requires a qualified FP16/mixed ONNX model for TensorRT 11. "
                      "Prepare it offline with NVIDIA ModelOpt AutoCast "
                      "(samples/modelopt/prepare_model.py), preserving the required I/O types, "
                      "or supply an explicit compatible engine");
            delete parser;
            delete network;
            delete builder;
            return false;
        }
#endif

        // Create builder config
        IBuilderConfig* config = builder->createBuilderConfig();
        if (!config) {
            GST_ERROR("Failed to create builder config");
            delete parser;
            delete network;
            delete builder;
            return false;
        }
        
        // Set optimization profiles for ALL inputs (handles combined
        // CLIP models where input_ids is a second input).
        IOptimizationProfile* profile = builder->createOptimizationProfile();
        if (profile) {
            for (int i = 0; i < network->getNbInputs(); i++) {
                const char* input_name = network->getInput(i)->getName();
                Dims input_dims = network->getInput(i)->getDimensions();
                
                Dims min_dims = input_dims;
                Dims opt_dims = input_dims;
                Dims max_dims = input_dims;
                
                min_dims.d[0] = 1;
                opt_dims.d[0] = batch_size;
                max_dims.d[0] = batch_size;
                
                profile->setDimensions(input_name, OptProfileSelector::kMIN, min_dims);
                profile->setDimensions(input_name, OptProfileSelector::kOPT, opt_dims);
                profile->setDimensions(input_name, OptProfileSelector::kMAX, max_dims);
                
                g_print("nvdsvisionencoder [TRT]:   profile for input '%s' "
                        "[batch 1..%d]\n", input_name, batch_size);
            }
            config->addOptimizationProfile(profile);
        }
        
        // TensorRT 11 uses the offline-prepared graph tensor types. Keep the
        // original automatic FP16 builder preference on older TensorRT versions.
#if NV_TENSORRT_MAJOR >= 11
        g_print("nvdsvisionencoder [TRT]: Using the offline-prepared model's "
                "declared precision; no automatic graph conversion.\n");
#else
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        if (builder->platformHasFastFp16()) {
            config->setFlag(BuilderFlag::kFP16);
            GST_DEBUG("FP16 mode enabled");
        }
#pragma GCC diagnostic pop
#endif
        
        // Set max workspace size (e.g., 2GB)
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 2ULL << 30);
        
        // Build engine
        GST_DEBUG("Building engine (this may take a few minutes)...");
        IHostMemory* serialized_engine = builder->buildSerializedNetwork(*network, *config);
        if (!serialized_engine) {
            GST_ERROR("Failed to build engine");
            delete config;
            delete parser;
            delete network;
            delete builder;
            return false;
        }
        
        GST_DEBUG("Engine built successfully!");
        
        // Save engine to file
        std::ofstream engine_file(engine_path, std::ios::binary);
        if (!engine_file) {
            GST_ERROR("Failed to open engine file for writing: %s", engine_path.c_str());
            delete serialized_engine;
            delete config;
            delete parser;
            delete network;
            delete builder;
            return false;
        }
        
        engine_file.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size());
        engine_file.close();
        
        GST_DEBUG("Engine saved to: %s", engine_path.c_str());
        
        // Cleanup
        delete serialized_engine;
        delete config;
        delete parser;
        delete network;
        delete builder;
        
        return true;
    }
    
    bool Infer(float* input_data, int num_images, float* output_embeddings, bool input_is_device = false) {
        if (!context_ || num_images <= 0) {
            return false;
        }
        
        // Calculate sizes per image
        size_t single_input_size = 3 * input_height_ * input_width_ * sizeof(float);
        size_t single_output_size = embedding_dim_ * sizeof(float);
        
        bool all_success = true;
        int processed = 0;
        
        // Process images in batches; use engine tensor names and full bindings (SigLIP2 e.g. pixel_values)
        int num_io = engine_->getNbIOTensors();
        std::vector<void*> bindings(num_io, nullptr);
        bindings[output_binding_index_] = d_output_buffer_;
        for (const auto& ei : extra_inputs_) {
            if (ei.d_buffer)
                bindings[ei.binding_index] = ei.d_buffer;
        }

        while (processed < num_images) {
            // Determine current batch size
            int current_batch_size = std::min(max_batch_size_, num_images - processed);
            
            // Calculate batch buffer sizes
            size_t batch_input_size = current_batch_size * single_input_size;
            size_t batch_output_size = current_batch_size * single_output_size;
            
            // Allocate input buffer only when input is on host (device path reuses plugin buffer)
            void* d_input = nullptr;
            if (!input_is_device) {
                if (cudaMalloc(&d_input, batch_input_size) != cudaSuccess) {
                    GST_ERROR("Failed to allocate device input for batch");
                    all_success = false;
                    break;
                }
            }
            
            // Prepare input data
            void* input_ptr;
            float* current_input = input_data + (processed * 3 * input_height_ * input_width_);
            
            if (input_is_device) {
                input_ptr = current_input;
            } else {
                input_ptr = d_input;
                cudaMemcpy(d_input, current_input, batch_input_size, cudaMemcpyHostToDevice);
            }
            
            bindings[input_binding_index_] = input_ptr;
            
            // Set input shape (use engine tensor name, not hardcoded "input")
            nvinfer1::Dims input_dims;
            input_dims.nbDims = 4;
            input_dims.d[0] = current_batch_size;
            input_dims.d[1] = 3;
            input_dims.d[2] = input_height_;
            input_dims.d[3] = input_width_;
            
            if (!context_->setInputShape(input_tensor_name_.c_str(), input_dims)) {
                GST_ERROR("Failed to set input shape for batch size %d", current_batch_size);
                all_success = false;
                if (!input_is_device && d_input) cudaFree(d_input);
                break;
            }
            
            // Set shapes for extra inputs (e.g. input_ids, attention_mask)
            for (const auto& ei : extra_inputs_) {
                nvinfer1::Dims extra_dims;
                extra_dims.nbDims = 2;
                extra_dims.d[0] = current_batch_size;
                extra_dims.d[1] = ei.seq_len;
                context_->setInputShape(ei.tensor_name.c_str(), extra_dims);
            }
            
            // Execute inference (output written to pre-allocated d_output_buffer_)
            bool success = context_->executeV2(bindings.data());
            
            if (success) {
                float* current_output = output_embeddings + (processed * embedding_dim_);
                cudaMemcpy(current_output, d_output_buffer_, batch_output_size, cudaMemcpyDeviceToHost);
            } else {
                GST_ERROR("Inference failed for batch starting at image %d (batch size: %d)", 
                          processed, current_batch_size);
                all_success = false;
            }
            
            if (!input_is_device && d_input) {
                cudaFree(d_input);
            }
            
            if (!success) {
                break;
            }
            
            processed += current_batch_size;
        }
        
        return all_success;
    }
    
    unsigned int GetInputWidth() const { return input_width_; }
    unsigned int GetInputHeight() const { return input_height_; }
    unsigned int GetEmbeddingDim() const { return embedding_dim_; }
    int GetMaxBatchSize() const { return max_batch_size_; }
    
private:
    bool ParseEngineInfo() {
        int num_bindings = engine_->getNbIOTensors();
        
        GST_DEBUG("Engine has %d I/O tensors:", num_bindings);
        
        const char* input_name = nullptr;
        for (int i = 0; i < num_bindings; i++) {
            const char* name = engine_->getIOTensorName(i);
            auto dims = engine_->getTensorShape(name);
            bool is_input = engine_->getTensorIOMode(name) == TensorIOMode::kINPUT;
            
            char dims_str[256] = {0};
            char* p = dims_str;
            for (int d = 0; d < dims.nbDims; d++) {
                p += snprintf(p, sizeof(dims_str) - (p - dims_str), "%ld", (long)dims.d[d]);
                if (d < dims.nbDims - 1) {
                    p += snprintf(p, sizeof(dims_str) - (p - dims_str), "x");
                }
            }
            GST_DEBUG("  %s '%s': %s", is_input ? "Input " : "Output", name, dims_str);
            
            if (is_input && dims.nbDims == 4) {
                // Preprocessing and Infer pass float NCHW buffers directly.
                // Model-internal FP16 is supported; external HALF I/O is not.
                if (engine_->getTensorDataType(name) != DataType::kFLOAT ||
                    engine_->getTensorFormat(name) != TensorFormat::kLINEAR ||
                    engine_->getTensorLocation(name) != TensorLocation::kDEVICE) {
                    GST_ERROR("Image input '%s' must use FP32 linear device I/O", name);
                    return false;
                }
                // NCHW image input: [batch, channels, height, width]
                input_height_ = dims.d[2];
                input_width_ = dims.d[3];
                input_name = name;
                input_tensor_name_ = name;
                input_binding_index_ = i;
            } else if (is_input && dims.nbDims == 2) {
                // Non-image input (e.g. input_ids, attention_mask from
                // combined CLIP/SigLIP2 models).  We allocate a zero-filled
                // device buffer so executeV2 has a valid binding; TensorRT
                // should have pruned the text encoder if we unmarked
                // text_embedding during build.
                ExtraInput ei;
                ei.binding_index = i;
                ei.tensor_name = name;
                auto dtype = engine_->getTensorDataType(name);
                ei.elem_size = (dtype == DataType::kINT64) ? 8 : 4;
                ei.seq_len = 1;
                for (int d = 1; d < dims.nbDims; d++)
                    ei.seq_len *= dims.d[d];
                ei.d_buffer = nullptr;
                extra_inputs_.push_back(ei);

                g_print("nvdsvisionencoder [TRT]: Extra input '%s' detected "
                        "(dims=%s, elem_size=%zu) — will bind zeros\n",
                        name, dims_str, ei.elem_size);
            } else if (!is_input && dims.nbDims == 2) {
                if (engine_->getTensorDataType(name) != DataType::kFLOAT ||
                    engine_->getTensorFormat(name) != TensorFormat::kLINEAR ||
                    engine_->getTensorLocation(name) != TensorLocation::kDEVICE) {
                    GST_ERROR("Embedding output '%s' must use FP32 linear device I/O", name);
                    return false;
                }
                // Output: [batch, embedding_dim]
                embedding_dim_ = dims.d[1];
                output_tensor_name_ = name;
                output_binding_index_ = i;
            }
        }
        
        // Get max batch size from optimization profile (for dynamic engines)
        if (input_name && engine_->getNbOptimizationProfiles() > 0) {
            auto max_dims = engine_->getProfileShape(input_name, 0, OptProfileSelector::kMAX);
            max_batch_size_ = max_dims.d[0];
            GST_DEBUG("Max batch size from profile: %d", max_batch_size_);
        } else {
            max_batch_size_ = 1;
            GST_DEBUG("Static engine, using batch size: %d", max_batch_size_);
        }
        
        // Allocate dummy device buffers for extra inputs (e.g. input_ids, attention_mask)
        for (auto& ei : extra_inputs_) {
            size_t buf_size = (size_t)max_batch_size_ * ei.seq_len * ei.elem_size;
            if (cudaMalloc(&ei.d_buffer, buf_size) != cudaSuccess) {
                GST_ERROR("Failed to allocate extra input buffer '%s' (%zu bytes)",
                          ei.tensor_name.c_str(), buf_size);
                return false;
            }
            cudaMemset(ei.d_buffer, 0, buf_size);
            GST_DEBUG("Allocated zero-filled buffer for '%s': %zu bytes",
                      ei.tensor_name.c_str(), buf_size);
        }
        
        return (input_width_ > 0 && input_height_ > 0 && embedding_dim_ > 0);
    }
    
    void Cleanup() {
        for (auto& ei : extra_inputs_) {
            if (ei.d_buffer) {
                cudaFree(ei.d_buffer);
                ei.d_buffer = nullptr;
            }
        }
        extra_inputs_.clear();
        if (d_output_buffer_) {
            cudaFree(d_output_buffer_);
            d_output_buffer_ = nullptr;
        }
        if (stream_) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
        if (context_) {
            delete context_;
            context_ = nullptr;
        }
        if (engine_) {
            delete engine_;
            engine_ = nullptr;
        }
        if (runtime_) {
            delete runtime_;
            runtime_ = nullptr;
        }
    }
    
    IRuntime* runtime_;
    ICudaEngine* engine_;
    IExecutionContext* context_;
    
    unsigned int input_width_;
    unsigned int input_height_;
    unsigned int embedding_dim_;
    int max_batch_size_;
    
    std::string input_tensor_name_;
    std::string output_tensor_name_;
    int input_binding_index_{0};
    int output_binding_index_{1};
    float* d_output_buffer_;
    
    // Extra non-image inputs (e.g. input_ids, attention_mask from
    // combined CLIP/SigLIP2 models).  Zero-filled device buffers are
    // allocated at engine load and bound during executeV2.
    struct ExtraInput {
        int binding_index;
        std::string tensor_name;
        size_t elem_size;   // bytes per element (4 for float/int32, 8 for int64)
        int seq_len;        // elements per sample (e.g. 77 for CLIP, 64 for SigLIP2)
        void* d_buffer;
    };
    std::vector<ExtraInput> extra_inputs_;
    
    cudaStream_t stream_;
};

/**
 * C API wrapper for TensorRT encoder
 */
extern "C" {

void* nvds_triton_client_create(const char* url, const char* model_name, 
                                 int batch_size, const char* onnx_model) {
    std::string engine_path;
    std::string onnx_path = onnx_model ? onnx_model : "";
    bool explicit_engine = false;
    bool needs_build = false;

    // Priority 1: Explicit engine path (tensorrt-engine property)
    if (url && (strstr(url, ".plan") || strstr(url, ".engine"))) {
        engine_path = url;
        explicit_engine = true;
        GST_DEBUG("Using explicit engine path from tensorrt-engine property");
    }
    // Priority 2: model_name looks like a path
    else if (model_name && (model_name[0] == '/' || strstr(model_name, ".plan") || strstr(model_name, ".engine"))) {
        engine_path = model_name;
        explicit_engine = true;
        GST_DEBUG("Using model_name as engine path");
    }
    // Priority 3: Auto-generate engine path from ONNX (nvinfer convention)
    else if (!onnx_path.empty()) {
        engine_path = GenerateEnginePath(onnx_path, batch_size);
        GST_DEBUG("Auto-generated engine path: %s", engine_path.c_str());
    }
    // Priority 4: No engine path could be determined — error out
    else {
        GST_ERROR("No engine path could be determined. "
                  "Set tensorrt-engine=<path>.plan or onnx-model=<path>.onnx");
        g_print("nvdsvisionencoder [TRT]: ERROR — no engine path. "
                "Please set the tensorrt-engine or onnx-model property.\n");
        return nullptr;
    }

    g_print("nvdsvisionencoder [TRT]: Engine path: %s\n", engine_path.c_str());
    g_print("nvdsvisionencoder [TRT]: ONNX path:   %s\n",
            onnx_path.empty() ? "(none — engine regeneration disabled)" : onnx_path.c_str());
    g_print("nvdsvisionencoder [TRT]: Batch size:  %d\n", batch_size);
    GST_DEBUG("  Explicit engine path: %s", explicit_engine ? "yes" : "no (auto-generated)");

    /* ------------------------------------------------------------------ *
     *  Decide whether to (re)build the engine.                           *
     *  Three triggers, following the nvinfer pattern:                     *
     *    1. Engine file does not exist                                    *
     *    2. ONNX model is newer than the engine (model changed)          *
     *    3. Engine's max batch size < requested batch size                *
     * ------------------------------------------------------------------ */

    if (!FileExists(engine_path)) {
        GST_DEBUG("Engine file not found — build required");
        needs_build = true;
    }
    else if (!onnx_path.empty() && FileExists(onnx_path)) {
        time_t onnx_mtime  = GetFileModTime(onnx_path);
        time_t engine_mtime = GetFileModTime(engine_path);

        if (onnx_mtime > engine_mtime) {
            GST_DEBUG("ONNX model is newer than engine (onnx=%ld > engine=%ld) — rebuild required",
                      (long)onnx_mtime, (long)engine_mtime);
            needs_build = true;
        }

        std::string stored_onnx = ReadEngineMeta(engine_path);
        if (!stored_onnx.empty() && stored_onnx != onnx_path) {
            GST_DEBUG("ONNX source changed: engine was built from '%s', now '%s' — rebuild required",
                      stored_onnx.c_str(), onnx_path.c_str());
            needs_build = true;
        }
    }

    if (!needs_build && FileExists(engine_path)) {
        TRTEncoder* probe = new TRTEncoder();
        if (probe->LoadEngine(engine_path)) {
            if (probe->GetMaxBatchSize() < batch_size) {
                GST_DEBUG("Engine max batch size %d < requested %d — rebuild required",
                          probe->GetMaxBatchSize(), batch_size);
                needs_build = true;
            }
        } else {
            GST_DEBUG("Failed to deserialize existing engine — rebuild required");
            needs_build = true;
        }
        delete probe;
    }

    /* ------------------------------------------------------------------ *
     *  Build engine from ONNX if needed                                  *
     * ------------------------------------------------------------------ */
    if (needs_build) {
        g_print("nvdsvisionencoder [TRT]: Engine rebuild required\n");
        if (onnx_path.empty() || !FileExists(onnx_path)) {
            if (!FileExists(engine_path)) {
                g_print("nvdsvisionencoder [TRT]: ERROR — no ONNX model provided, cannot build engine. "
                        "Set onnx-model in [visionencoder] config.\n");
                if (!onnx_path.empty())
                    g_print("nvdsvisionencoder [TRT]: ONNX path does not exist: %s\n", onnx_path.c_str());
                return nullptr;
            }
            g_print("nvdsvisionencoder [TRT]: WARNING — no ONNX model provided, using existing engine as-is\n");
            needs_build = false;
        } else {
            if (!explicit_engine) {
                engine_path = GenerateEnginePath(onnx_path, batch_size);
                g_print("nvdsvisionencoder [TRT]: Auto-generated engine path: %s\n", engine_path.c_str());
            }

            g_print("nvdsvisionencoder [TRT]: Building engine from ONNX: %s (batch_size=%d)\n",
                    onnx_path.c_str(), batch_size);

            TRTEncoder* builder = new TRTEncoder();
            if (!builder->BuildEngineFromONNX(onnx_path, engine_path, batch_size)) {
                g_print("nvdsvisionencoder [TRT]: ERROR — failed to build engine from ONNX\n");
                delete builder;
                return nullptr;
            }
            delete builder;
            WriteEngineMeta(engine_path, onnx_path);
            g_print("nvdsvisionencoder [TRT]: Engine built and saved: %s\n", engine_path.c_str());
        }
    } else {
        g_print("nvdsvisionencoder [TRT]: Existing engine is compatible — skipping rebuild\n");
    }

    GST_DEBUG("==========================================");

    /* ------------------------------------------------------------------ *
     *  Load the (possibly freshly built) engine                          *
     * ------------------------------------------------------------------ */
    TRTEncoder* encoder = new TRTEncoder();
    if (!encoder->LoadEngine(engine_path)) {
        GST_ERROR("Failed to load engine from: %s", engine_path.c_str());
        delete encoder;
        return nullptr;
    }

    return encoder;
}

void nvds_triton_client_destroy(void* client) {
    if (client) {
        TRTEncoder* encoder = static_cast<TRTEncoder*>(client);
        delete encoder;
        GST_DEBUG("Encoder destroyed");
    }
}

int nvds_triton_client_get_input_size(void* client, unsigned int* width, unsigned int* height) {
    if (!client || !width || !height) {
        return -1;
    }
    
    TRTEncoder* encoder = static_cast<TRTEncoder*>(client);
    *width = encoder->GetInputWidth();
    *height = encoder->GetInputHeight();
    return 0;
}

int nvds_triton_client_get_embedding_dim(void* client) {
    if (!client) {
        return 0;
    }
    
    TRTEncoder* encoder = static_cast<TRTEncoder*>(client);
    return encoder->GetEmbeddingDim();
}

unsigned int nvds_triton_client_get_max_batch_size(void* client) {
    if (!client) return 0;
    return (unsigned int) reinterpret_cast<TRTEncoder*>(client)->GetMaxBatchSize();
}

unsigned int nvds_triton_client_get_input_width(void* client) {
    if (!client) {
        return 0;
    }
    
    TRTEncoder* encoder = static_cast<TRTEncoder*>(client);
    return encoder->GetInputWidth();
}

unsigned int nvds_triton_client_get_input_height(void* client) {
    if (!client) {
        return 0;
    }
    
    TRTEncoder* encoder = static_cast<TRTEncoder*>(client);
    return encoder->GetInputHeight();
}

int nvds_triton_client_infer(void* client, float* input_data, int num_images,
                              float* output_embeddings) {
    if (!client || !input_data || !output_embeddings || num_images <= 0) {
        return -1;
    }
    
    TRTEncoder* encoder = static_cast<TRTEncoder*>(client);
    
    if (encoder->Infer(input_data, num_images, output_embeddings, false)) {
        return 0;  // Success
    } else {
        return -1;  // Failure
    }
}

int nvds_triton_client_infer_device(void* client, float* d_input_data, int num_images,
                                     float* output_embeddings) {
    if (!client || !d_input_data || !output_embeddings || num_images <= 0) {
        return -1;
    }
    
    TRTEncoder* encoder = static_cast<TRTEncoder*>(client);
    
    if (encoder->Infer(d_input_data, num_images, output_embeddings, true)) {
        return 0;  // Success
    } else {
        return -1;  // Failure
    }
}

} // extern "C"
