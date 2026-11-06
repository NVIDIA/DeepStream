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

#include "engine_builder.h"
#include <memory>
#include <vector>
#include <sys/stat.h>
#include <string>
#include <iostream>
#include <fstream>
#include <glib.h>
#include <dlfcn.h>
#include <cstdlib>
#include <iomanip>
#include <map>
#include <mutex>
#include <unistd.h>
#include <cstdio>
#include <cstring>
#include <set>
#include <sys/wait.h>
#include <link.h>
#include <sys/sysmacros.h>
#if NV_TENSORRT_MAJOR >= 11
#include "nvdsinfer_model_builder.h"
#endif

static std::string fileSha256(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) return {};
    std::unique_ptr<GChecksum, decltype(&g_checksum_free)> checksum(
        g_checksum_new(G_CHECKSUM_SHA256), g_checksum_free);
    char data[65536];
    while (input.read(data, sizeof(data)) || input.gcount()) {
        g_checksum_update(checksum.get(), reinterpret_cast<const guchar*>(data), input.gcount());
    }
    return input.eof() ? g_checksum_get_string(checksum.get()) : std::string();
}

struct ModelClosure {
    bool valid = true;
    bool identified = false;
    std::string hash;
    std::set<std::string> paths;
};

// Reuse the SDK's ONNX traversal/range/path validation rather than implementing
// another protobuf decoder here. Inspection is optional at runtime: deployments
// without the CPU model tools still parse/build, but never trust a model cache.
static ModelClosure inspectModelClosure(const std::string& path, PrettyLogger* logger) {
    ModelClosure result;
    result.paths.insert(path);
    const char* selectedHelper = std::getenv("SPARSE4D_MODELOPT_PREPARE");
    const std::string helper = selectedHelper && *selectedHelper ? selectedHelper :
        "/opt/nvidia/deepstream/deepstream/samples/modelopt/prepare_model.py";
    const char* selectedPython = std::getenv("SPARSE4D_MODEL_PYTHON");
    const std::string python = selectedPython && *selectedPython ? selectedPython : "python3";
    if (!g_file_test(helper.c_str(), G_FILE_TEST_IS_REGULAR)) {
        logger->warning("ONNX closure inspector is unavailable; building an uncached engine. "
            "Set SPARSE4D_MODELOPT_PREPARE and SPARSE4D_MODEL_PYTHON to enable verified reuse.");
        return result;
    }
    static const char script[] = R"PY(
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
try:
    import onnx
    spec = importlib.util.spec_from_file_location("sparse4d_model_preparation", sys.argv[2])
    preparation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preparation)
except (ImportError, FileNotFoundError) as error:
    print(f"ONNX closure inspector unavailable: {error}", file=sys.stderr)
    raise SystemExit(3)
try:
    source = Path(sys.argv[1]).resolve(strict=True)
    proto = onnx.load(str(source), load_external_data=False)
    external = preparation.external_files(proto, source)
    closure = {"model": preparation.identity(source), "external_data": external}
    print(hashlib.sha256(json.dumps(closure, sort_keys=True).encode()).hexdigest())
    paths = {source} | {(source.parent / name).resolve(strict=True) for name in external}
    for item in sorted(paths):
        print(base64.b64encode(str(item).encode()).decode())
except Exception as error:
    print(f"Invalid ONNX/external-weight closure: {error}", file=sys.stderr)
    raise SystemExit(2)
)PY";
    gchar* output = nullptr;
    gchar* errors = nullptr;
    gint status = 0;
    GError* error = nullptr;
    gchar* arguments[] = {const_cast<gchar*>(python.c_str()), const_cast<gchar*>("-B"),
        const_cast<gchar*>("-c"), const_cast<gchar*>(script), const_cast<gchar*>(path.c_str()),
        const_cast<gchar*>(helper.c_str()), nullptr};
    const bool started = g_spawn_sync(nullptr, arguments, nullptr, G_SPAWN_SEARCH_PATH,
        nullptr, nullptr, &output, &errors, &status, &error);
    std::unique_ptr<gchar, decltype(&g_free)> captured(output, g_free), diagnostics(errors, g_free);
    if (!started || (WIFEXITED(status) && WEXITSTATUS(status) == 3)) {
        logger->warning("ONNX closure inspector unavailable; building an uncached engine: "
            + std::string(error ? error->message : (errors ? errors : "unknown error")));
        if (error) g_error_free(error);
        return result;
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0 || !output) {
        logger->error(errors ? errors : "Failed to inspect ONNX/external-weight closure");
        result.valid = false;
        return result;
    }
    std::istringstream lines(output);
    std::getline(lines, result.hash);
    if (result.hash.size() != 64 || result.hash.find_first_not_of("0123456789abcdef") != std::string::npos) {
        result.valid = false;
        logger->error("Invalid ONNX closure inspector result");
        return result;
    }
    std::string encoded;
    while (std::getline(lines, encoded)) {
        if (encoded.empty()) continue;
        gsize size = 0;
        std::unique_ptr<guchar, decltype(&g_free)> decoded(g_base64_decode(encoded.c_str(), &size), g_free);
        if (!size || decoded.get()[0] != '/' || std::memchr(decoded.get(), '\0', size)) {
            result.valid = false;
            logger->error("Invalid input path in ONNX closure inspector result");
            return result;
        }
        result.paths.emplace(reinterpret_cast<const char*>(decoded.get()), size);
    }
    result.identified = true;
    return result;
}

static bool outputOverwritesInput(const std::string& output, const ModelClosure& closure) {
    // realpath resolves existing symlinks. For a new output, resolve its parent
    // too, so publication through an alias cannot overwrite an input payload.
    std::unique_ptr<gchar, decltype(&g_free)> resolved(realpath(output.c_str(), nullptr), g_free);
    if (resolved && closure.paths.count(resolved.get())) return true;
    std::unique_ptr<gchar, decltype(&g_free)> parent(g_path_get_dirname(output.c_str()), g_free);
    std::unique_ptr<gchar, decltype(&g_free)> base(g_path_get_basename(output.c_str()), g_free);
    std::unique_ptr<gchar, decltype(&g_free)> canonicalParent(realpath(parent.get(), nullptr), g_free);
    if (canonicalParent) {
        const std::string destination = std::string(canonicalParent.get()) + "/" + base.get();
        if (closure.paths.count(destination)) return true;
    }
    return false;
}

static std::string mappedPluginIdentity(const char* name) {
    void* handle = dlopen(name, RTLD_NOW | RTLD_NOLOAD);
    if (!handle) return {};
    struct link_map* mapping = nullptr;
    const bool known = dlinfo(handle, RTLD_DI_LINKMAP, &mapping) == 0 && mapping && mapping->l_name;
    const std::string path = known ? mapping->l_name : "";
    dlclose(handle);  // Drop only this RTLD_NOLOAD reference, not the plugin.
    struct stat file{};
    if (path.empty() || stat(path.c_str(), &file) != 0) return {};
    std::ifstream maps("/proc/self/maps");
    std::string line;
    while (std::getline(maps, line)) {
        unsigned int deviceMajor = 0, deviceMinor = 0;
        unsigned long long inode = 0;
        if (std::sscanf(line.c_str(), "%*s %*s %*s %x:%x %llu", &deviceMajor, &deviceMinor, &inode) == 3
            && deviceMajor == major(file.st_dev) && deviceMinor == minor(file.st_dev)
            && inode == file.st_ino) {
            const auto hash = fileSha256(path);
            return hash.empty() ? std::string() : path + ":" + hash + "\n";
        }
    }
    return {};  // A replaced on-disk path is not proof of the mapped plugin bytes.
}

bool prepareSparse4dEngine(const std::string& onnxFilePath,
    std::string& engineFilePath, int batchSize, int numSensors,
    int width, int height, int gpuId, bool fp16Mode, bool forceRebuild,
    PrettyLogger* logger) {
    // Serialize startup builds/cache publication across contexts in one process.
    static std::mutex engineMutex;
    std::lock_guard<std::mutex> engineLock(engineMutex);
    if (!logger || cudaSetDevice(gpuId) != cudaSuccess) return false;
    std::unique_ptr<gchar, decltype(&g_free)> canonicalModel(realpath(onnxFilePath.c_str(), nullptr), g_free);
    if (!canonicalModel) {
        logger->error("Cannot resolve ONNX model: " + onnxFilePath);
        return false;
    }
    const ModelClosure model = inspectModelClosure(canonicalModel.get(), logger);
    if (!model.valid) return false;
    std::string pluginIdentity;
    const char* pluginPath = std::getenv("MSDA_PLUGIN_PATH");
    static std::map<std::string, std::string> loadedPlugins;
    static std::mutex pluginMutex;
    if (pluginPath && *pluginPath) {
        const std::string hash = fileSha256(pluginPath);
        if (hash.empty()) {
            logger->error("Cannot identify MSDA_PLUGIN_PATH: " + std::string(pluginPath));
            return false;
        }
        // Plugin creators must remain loaded through parsing/deserialization.
        std::lock_guard<std::mutex> lock(pluginMutex);
        if (loadedPlugins.count(pluginPath) && loadedPlugins.at(pluginPath) != hash) {
            logger->error("MSDA plugin changed after loading; restart the process before using its new binary");
            return false;
        }
        if (!loadedPlugins.count(pluginPath)) {
            if (!dlopen(pluginPath, RTLD_NOW | RTLD_GLOBAL)) {
                logger->error("Cannot load MSDA plugin: " + std::string(dlerror()));
                return false;
            }
            loadedPlugins.emplace(pluginPath, hash);
        }
        pluginIdentity += mappedPluginIdentity(pluginPath);
    }
    // Preserve existing LD_PRELOAD launchers, without inventing a required field.
    // Unresolvable preload identities disable reuse rather than trust stale plans.
    const bool knownExplicitPlugin = !(pluginPath && *pluginPath) || !pluginIdentity.empty();
    bool identifiablePlugins = !pluginIdentity.empty();
    if (const char* preload = std::getenv("LD_PRELOAD")) {
        std::unique_ptr<gchar*, decltype(&g_strfreev)> paths(g_strsplit_set(preload, " :\t", -1), g_strfreev);
        for (gchar** path = paths.get(); path && *path; ++path) {
            if (!**path) continue;
            const auto identity = mappedPluginIdentity(*path);
            if (identity.empty()) { identifiablePlugins = false; break; }
            identifiablePlugins = true;
            pluginIdentity += identity;
        }
    }
    identifiablePlugins = identifiablePlugins && knownExplicitPlugin;
    cudaDeviceProp device{};
    if (cudaGetDeviceProperties(&device, gpuId) != cudaSuccess) return false;
    const std::string modelHash = fileSha256(canonicalModel.get());
    if (modelHash.empty()) return false;
    std::ostringstream identity;
    identity << "sparse4d-cache-v3-onnx-closure\nmodel=" << canonicalModel.get() << ':' << modelHash
             << "\nclosure=" << model.hash << "\nplugins=" << pluginIdentity
             << "\ntrt=" << NV_TENSORRT_MAJOR << '.' << NV_TENSORRT_MINOR << '.'
             << NV_TENSORRT_PATCH << '.' << NV_TENSORRT_BUILD
             << "\nruntime=" << getInferLibVersion()
#if NV_TENSORRT_MAJOR >= 11
             << '.' << getInferLibBuildVersion()
             << "\nprecision-policy=" << (fp16Mode ? "offline-mixed-v2" : "model-declared")
#endif
             << "\nshape=" << batchSize << ',' << numSensors << ',' << width << ',' << height
             << "\nfp16=" << fp16Mode << "\ngpu=" << device.name << ':' << device.major << '.' << device.minor << ':';
    for (const auto byte : device.uuid.bytes) identity << std::hex << std::setw(2)
        << std::setfill('0') << (static_cast<unsigned int>(static_cast<unsigned char>(byte)));
    identity << '\n';
    const bool cacheable = model.identified && identifiablePlugins;
    if (!cacheable) {
        std::unique_ptr<gchar, decltype(&g_free)> invocation(g_uuid_string_random(), g_free);
        identity << "uncached-invocation=" << invocation.get() << '\n';
    }
    // Different model/plugin/TRT/shape identities never replace the plan which
    // another process is about to deserialize. Same-identity rebuilds remain valid.
    std::unique_ptr<gchar, decltype(&g_free)> key(
        g_compute_checksum_for_string(G_CHECKSUM_SHA256, identity.str().c_str(), -1), g_free);
    engineFilePath += ".trt" + std::to_string(NV_TENSORRT_MAJOR) + "." + key.get() + ".engine";
    if (!identifiablePlugins) logger->warning("MSDA/preload identity is unavailable; rebuilding rather than reusing an unverified plan");
    const std::string markerPath = engineFilePath + ".sparse4d.identity";
    if (outputOverwritesInput(engineFilePath, model) || outputOverwritesInput(markerPath, model)) {
        logger->error("Sparse4D engine or identity output would overwrite an ONNX input file");
        return false;
    }
    if (cacheable && !forceRebuild) {
        const auto engineHash = fileSha256(engineFilePath);
        std::ifstream saved(markerPath);
        const std::string actual((std::istreambuf_iterator<char>(saved)), std::istreambuf_iterator<char>());
        if (!engineHash.empty() && actual == identity.str() + "engine=" + engineHash + "\n") {
            logger->info("Reusing Sparse4D engine with verified model/plugin/TRT/GPU/shape identity");
            return true;
        }
    }
    logger->info("Building fresh Sparse4D engine; old or unidentified cache is not reused");
    std::string temporary = engineFilePath + ".sparse4d-XXXXXX";
    const int descriptor = g_mkstemp(&temporary[0]);
    if (descriptor < 0) {
        logger->error("Cannot create a private Sparse4D engine output beside: " + engineFilePath);
        return false;
    }
    close(descriptor);
    if (!buildEngineFromOnnx(canonicalModel.get(), temporary, batchSize, numSensors, fp16Mode, logger)) {
        std::remove(temporary.c_str());
        return false;
    }
    // Do not publish a plan under an identity which changed while TensorRT was
    // reading/building it. Keep any previously published plan untouched.
    const ModelClosure current = model.identified ? inspectModelClosure(canonicalModel.get(), logger) : ModelClosure{};
    if (fileSha256(canonicalModel.get()) != modelHash || (model.identified
        && (!current.valid || !current.identified || current.hash != model.hash || current.paths != model.paths))) {
        logger->error("ONNX model/external weights changed during the Sparse4D build; keeping the previous engine");
        std::remove(temporary.c_str());
        return false;
    }
    if (outputOverwritesInput(engineFilePath, current) || outputOverwritesInput(markerPath, current)) {
        logger->error("Sparse4D output changed to an ONNX input path during the build");
        std::remove(temporary.c_str());
        return false;
    }
    const auto engineHash = fileSha256(temporary);
    if (engineHash.empty() || std::rename(temporary.c_str(), engineFilePath.c_str()) != 0) {
        std::remove(temporary.c_str());
        return false;
    }
    if (cacheable) {
        // The marker is replaced, not opened through an existing symlink.
        std::string temporaryMarker = markerPath + ".XXXXXX";
        const int markerDescriptor = g_mkstemp(&temporaryMarker[0]);
        if (markerDescriptor < 0) return false;
        close(markerDescriptor);
        std::ofstream marker(temporaryMarker, std::ios::trunc);
        marker << identity.str() << "engine=" << engineHash << '\n';
        marker.close();
        if (!marker || std::rename(temporaryMarker.c_str(), markerPath.c_str()) != 0) {
            std::remove(temporaryMarker.c_str());
            return false;
        }
    }
    return true;
}

// Function to read the entire file into a vector - marked static to limit scope to this file
static std::vector<char> readFile(const std::string& filePath) {
    std::ifstream file(filePath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filePath << std::endl;
        return {};
    }

    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(fileSize);
    if (!file.read(buffer.data(), fileSize)) {
        std::cerr << "Error reading file: " << filePath << std::endl;
        return {};
    }

    return buffer;
}

bool buildEngineFromOnnx(
    const std::string& onnxFilePath,
    const std::string& engineFilePath,
    int batchSize,
    int numSensors,
    bool fp16Mode,
    PrettyLogger* logger
) {
    if (batchSize <= 0 || numSensors <= 0 || !logger) {
        std::cerr << "Sparse4D requires positive batch/sensor dimensions and a logger" << std::endl;
        return false;
    }
    std::unique_ptr<gchar, decltype(&g_free)> canonicalModel(realpath(onnxFilePath.c_str(), nullptr), g_free);
    if (!canonicalModel) {
        logger->error("Cannot resolve ONNX model: " + onnxFilePath);
        return false;
    }
    // Create TensorRT logger with ERROR severity to hide verbose reshape messages
    static Logger trtLogger(nvinfer1::ILogger::Severity::kERROR);

    // Create builder
    TRTUniquePtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(trtLogger));
    if (!builder) {
        logger->error("Failed to create TensorRT builder");
        return false;
    }

    // Create network definition - in TensorRT 10.3, we don't use kEXPLICIT_BATCH flag
    // as it's now the default and the flag is deprecated
    TRTUniquePtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0));
    if (!network) {
        logger->error("Failed to create TensorRT network definition");
        return false;
    }

    // Create ONNX parser
    TRTUniquePtr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, trtLogger));
    if (!parser) {
        logger->error("Failed to create ONNX parser");
        return false;
    }

    // Parse ONNX model
    std::vector<char> modelData = readFile(canonicalModel.get());
    if (modelData.empty()) {
        return false;
    }

    logger->info("Parsing ONNX file: " + onnxFilePath);

    // TensorRT resolves external tensor locations relative to this filename,
    // never the process working directory or a symlink's containing directory.
    bool parseSuccess = parser->parse(modelData.data(), modelData.size(), canonicalModel.get());
    if (!parseSuccess) {
        logger->error("Failed to parse ONNX model");
        return false;
    }

    logger->info("Successfully Parsed ONNX file: " + onnxFilePath);

#if NV_TENSORRT_MAJOR >= 11
    // The surrounding preprocess/state/postprocess code uses FP32 buffers.
    // A typed internal FP16 graph must retain those external floating I/O types.
    for (int output = 0; output < 2; ++output) {
        const int count = output ? network->getNbOutputs() : network->getNbInputs();
        for (int i = 0; i < count; ++i) {
            auto* tensor = output ? network->getOutput(i) : network->getInput(i);
            const auto type = tensor->getType();
            if (type != nvinfer1::DataType::kFLOAT && type != nvinfer1::DataType::kINT32
                && type != nvinfer1::DataType::kINT64 && type != nvinfer1::DataType::kBOOL) {
                logger->error("Sparse4D requires FP32 external floating I/O; incompatible tensor: "
                    + std::string(tensor->getName()));
                return false;
            }
        }
    }
    nvdsinfer::ExplicitBuildParams precision;
    precision.networkMode = fp16Mode ? NvDsInferNetworkMode_FP16 : NvDsInferNetworkMode_FP32;
    if (nvdsinfer::validateTrt11Precision(*network, precision) != NVDSINFER_SUCCESS) {
        logger->error("Sparse4D model does not implement the requested TensorRT 11 precision. "
            "Prepare FP16/mixed ONNX offline with NVIDIA ModelOpt AutoCast "
            "(samples/modelopt/prepare_model.py), preserving FP32 external I/O and the "
            "matching MSDA custom-op contract; no automatic graph conversion is performed.");
        return false;
    }
    bool hasHalfCompute = false;
    for (int i = 0; i < network->getNbLayers(); ++i) {
        const auto* layer = network->getLayer(i);
        // Shape/reformat/plugin-only HALF tensors do not establish reduced
        // precision for the model's weighted numerical computation.
        if (layer->getType() != nvinfer1::LayerType::kCONVOLUTION
            && layer->getType() != nvinfer1::LayerType::kMATRIX_MULTIPLY) continue;
        for (int j = 0; j < layer->getNbOutputs(); ++j) {
            const auto* tensor = layer->getOutput(j);
            hasHalfCompute |= tensor && tensor->getType() == nvinfer1::DataType::kHALF;
        }
    }
    if (fp16Mode && !hasHalfCompute) {
        logger->error("enable_fp16 requires FP16 weighted computation in the prepared Sparse4D model on TensorRT 11. "
            "Preserve FP32 external I/O and validate the matching MSDA custom-op contract; "
            "prepare a qualified mixed ONNX model offline with NVIDIA ModelOpt AutoCast "
            "(samples/modelopt/prepare_model.py). Silent FP32 fallback is not supported.");
        return false;
    }
#endif

    // TensorRT's builder owns profiles (their destructor is protected).
    auto* profile = builder->createOptimizationProfile();
    if (!profile) return false;
    // Create config
    TRTUniquePtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    if (!config) {
        logger->error("Failed to create builder config");
        return false;
    }

    // Set workspace size - smaller workspace to avoid out of memory
    const size_t workspaceSizeMB = 512;
    const size_t workspace_size = static_cast<size_t>(workspaceSizeMB) * (1ULL << 20);
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, workspace_size);
    logger->info("Setting TensorRT builder workspace size to " + std::to_string(workspaceSizeMB) + "MB");

#if NV_TENSORRT_MAJOR < 11
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    if (fp16Mode) config->setFlag(nvinfer1::BuilderFlag::kFP16);
#pragma GCC diagnostic pop
    config->clearFlag(nvinfer1::BuilderFlag::kTF32);
#endif


    // Set builder optimization level to 3 (maximum) to reduce memory usage
    config->setBuilderOptimizationLevel(3);
    logger->info("Setting TensorRT builder optimization level to 3 (maximum)");

    // Create optimization profile with explicit tensor shapes to match trtexec command
    bool hasDynamicShapes = false;

    // For each input in the network, check and set dimensions
    for (int i = 0; i < network->getNbInputs(); i++) {
        auto input = network->getInput(i);
        auto dims = input->getDimensions();

        // Check if this input has dynamic dimensions
        bool isDynamicInput = false;
        for (int d = 0; d < dims.nbDims; d++) {
            if (dims.d[d] == -1) {
                isDynamicInput = true;
                hasDynamicShapes = true;
                break;
            }
        }
        // If this is a dynamic input, set dimensions in the profile
        if (isDynamicInput) {
            // Create the min, optimal and max dimensions
            nvinfer1::Dims minDims = dims;
            nvinfer1::Dims optDims = dims;
            nvinfer1::Dims maxDims = dims;

            // Set appropriate values for each dynamic dimension
            for (int d = 0; d < dims.nbDims; d++) {
                if (dims.d[d] == -1) {
                    // model batch size
                    if (d == 0) {
                        minDims.d[d] = 1;
                        optDims.d[d] = batchSize;
                        maxDims.d[d] = batchSize;
                    } else {
                        // model with numSensors
                        minDims.d[d] = 1;
                        optDims.d[d] = numSensors;
                        maxDims.d[d] = numSensors;
                    }
                }
            }

            std::stringstream ss;
            ss << "Input: " << input->getName() << ", Type: ";
            switch (input->getType()) {
                case nvinfer1::DataType::kFLOAT: ss << "FLOAT32"; break;
                case nvinfer1::DataType::kHALF: ss << "FLOAT16"; break;
                case nvinfer1::DataType::kINT8: ss << "INT8"; break;
                case nvinfer1::DataType::kINT32: ss << "INT32"; break;
                case nvinfer1::DataType::kBOOL: ss << "BOOL"; break;
                case nvinfer1::DataType::kUINT8: ss << "UINT8"; break;
                default: ss << "UNKNOWN"; break;
            }
            ss << ", Dims: [";
            for (int d = 0; d < dims.nbDims; d++) {
                ss << dims.d[d];
                if (d < dims.nbDims - 1) ss << ", ";
            }
            ss << "]";
            ss << "  MiDims: [";
            for (int d = 0; d < minDims.nbDims; d++) {
                ss << minDims.d[d];
                if (d < minDims.nbDims - 1) ss << ", ";
            }
            ss << "]";
            ss << "  OptDims: [";
            for (int d = 0; d < optDims.nbDims; d++) {
                ss << optDims.d[d];
                if (d < optDims.nbDims - 1) ss << ", ";
            }
            ss << "]";
            ss << "  MaxDims: [";
            for (int d = 0; d < maxDims.nbDims; d++) {
                ss << maxDims.d[d];
                if (d < maxDims.nbDims - 1) ss << ", ";
            }
            ss << "]";
            logger->info(ss.str());

            // Set dimensions in the profile
            if (!profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, minDims)
                || !profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, optDims)
                || !profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, maxDims)) {
                logger->error("Invalid Sparse4D optimization-profile dimensions");
                return false;
            }
        }
    }
    // Only add the profile if dynamic shapes were detected
    if (hasDynamicShapes) {
        if (config->addOptimizationProfile(profile) < 0) return false;
        logger->info("Added optimization profile to TensorRT builder");
    }

    logger->persistent_info("INFO: Building TensorRT engine from ONNX model " + onnxFilePath);

    // In TensorRT 10.3, use buildSerializedNetwork
    TRTUniquePtr<nvinfer1::IHostMemory> serializedEngine(builder->buildSerializedNetwork(*network, *config));
    if (!serializedEngine) {
        logger->error("Failed to build TensorRT engine");
        return false;
    } else {
        logger->info("Success: Built TensorRT Engine");
    }

    // Save engine to file
    std::ofstream engineFile(engineFilePath, std::ios::binary);
    if (!engineFile) {
        logger->error("Failed to open engine file for writing: " + engineFilePath);
        return false;
    }

    engineFile.write(static_cast<const char*>(serializedEngine->data()), serializedEngine->size());
    if (engineFile.fail()) {
        logger->error("Failed to write engine to file");
        return false;
    } else {
        logger->persistent_info("Success: Saved engine to file: " + engineFilePath);
    }

    return true;
}
