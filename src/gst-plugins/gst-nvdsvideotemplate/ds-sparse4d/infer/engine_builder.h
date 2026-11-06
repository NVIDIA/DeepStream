/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef ENGINE_BUILDER_H
#define ENGINE_BUILDER_H

#include <string>
#include <fstream>
#include <iostream>
#include <memory>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include "../utils/log_utils.h"


// Smart pointer types for TensorRT objects to ensure proper cleanup
struct TRTDeleter {
    template <typename T>
    void operator()(T* obj) const {
        if (obj) {
            delete obj;
        }
    }
};

template <typename T>
using TRTUniquePtr = std::unique_ptr<T, TRTDeleter>;

// Logger for TensorRT with verbosity control
class Logger : public nvinfer1::ILogger {
public:
    // Default constructor sets verbosity to ERROR (only show errors and fatals)
    Logger(Severity verbosity = Severity::kERROR) : mVerbosity(verbosity) {}

    void log(Severity severity, const char* msg) noexcept override {
        // Only log messages at or above the configured verbosity level
        if (severity <= mVerbosity) {
            // Add a prefix to identify the source of the message
            const char* severityStr = "";
            switch (severity) {
                case Severity::kINTERNAL_ERROR: severityStr = "[F] "; break;
                case Severity::kERROR:          severityStr = "[E] "; break;
                case Severity::kWARNING:        severityStr = "[W] "; break;
                case Severity::kINFO:           severityStr = "[I] "; break;
                case Severity::kVERBOSE:        severityStr = "[V] "; break;
            }
            std::cerr << severityStr << msg << std::endl;
        }
    }

private:
    Severity mVerbosity;
} static logger;

/**
 * Builds a TensorRT engine from an ONNX model and saves it to the specified file path.
 * Compatible with TensorRT 10/11; external weights resolve beside the canonical
 * ONNX filename, independently of the application's working directory.
 *
 * @param onnxFilePath Path to the input ONNX model
 * @param engineFilePath Path where the output engine should be saved
 * @param batchSize Sparse4D model batch size
 * @param numSensors Number of sensors [Deepstream batch size]
 * @param fp16Mode Whether to enable FP16 mode for faster inference
 * @param logger Optional logger for structured logging
 * @return True if the engine was built successfully, false otherwise
 */
bool buildEngineFromOnnx(
    const std::string& onnxFilePath,
    const std::string& engineFilePath,
    int batchSize,
    int numSensors,
    bool fp16Mode,
    PrettyLogger* logger = nullptr
);

// Reuse only an engine whose complete ONNX/external-weight closure and
// plugin/TRT/GPU/shape identity still match. Optional CPU ONNX inspection uses
// SPARSE4D_MODEL_PYTHON/SPARSE4D_MODELOPT_PREPARE; without it, build a private
// uncached plan instead of trusting a protobuf-only model fingerprint.
bool prepareSparse4dEngine(const std::string& onnxFilePath,
    std::string& engineFilePath, int batchSize, int numSensors,
    int width, int height, int gpuId, bool fp16Mode, bool forceRebuild,
    PrettyLogger* logger);

#endif // ENGINE_BUILDER_H
