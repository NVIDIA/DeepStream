/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Build-only front end for the same model-declared precision validation as nvinfer.
 * This is deliberately not a trtexec benchmark or a general trtexec replacement.
 */
#include "nvdsinfer_model_builder.h"
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>

#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

namespace {
using namespace nvinfer1;
using ShapeMap = std::map<std::string, std::vector<int64_t>>;
mode_t outputMask = 0022; // Captured before TensorRT can start worker threads.

struct Options {
    std::string onnx, engine, inputFormats, outputFormats, layerInfo;
    ShapeMap shapes[3];
    std::vector<std::string> plugins;
    bool fp16 = false, int8 = false, noTF32 = false, verbose = false;
    bool dumpLayers = false, sparse = false;
    ProfilingVerbosity verbosity = ProfilingVerbosity::kLAYER_NAMES_ONLY;
    int device = 0, optimization = -1;
    size_t workspace = 0;
};

class Logger : public ILogger {
public:
    bool verbose = false;
    void log(Severity severity, const char* msg) noexcept override {
        if (verbose || severity <= Severity::kINFO)
            std::cerr << "[TensorRT] " << msg << '\n';
    }
};

std::vector<std::string> split(const std::string& text, char delimiter) {
    std::vector<std::string> result;
    std::istringstream stream(text);
    std::string part;
    while (std::getline(stream, part, delimiter)) {
        if (part.empty()) throw std::runtime_error("empty value in '" + text + "'");
        result.push_back(part);
    }
    if (result.empty() || text.back() == delimiter)
        throw std::runtime_error("empty value in '" + text + "'");
    return result;
}

int64_t integer(const std::string& text, int64_t minimum, int64_t maximum) {
    size_t end = 0;
    int64_t value = std::stoll(text, &end);
    if (end != text.size() || value < minimum || value > maximum)
        throw std::runtime_error("invalid integer: " + text);
    return value;
}

ShapeMap parseShapes(const std::string& text) {
    ShapeMap result;
    for (const auto& item : split(text, ',')) {
        // Tensor names may themselves contain a colon (e.g. input:0).
        auto colon = item.rfind(':');
        if (colon == std::string::npos || colon == 0)
            throw std::runtime_error("expected tensor:dimxdim in " + item);
        std::vector<int64_t> dims;
        for (const auto& dim : split(item.substr(colon + 1), 'x'))
            dims.push_back(integer(dim, std::numeric_limits<int64_t>::min(),
                                   std::numeric_limits<int64_t>::max()));
        if (!result.emplace(item.substr(0, colon), std::move(dims)).second)
            throw std::runtime_error("duplicate shape for " + item.substr(0, colon));
    }
    return result;
}

bool sameFile(const std::string& left, const std::string& right) {
    if (left == right) return true;
    // The outputs may not exist yet. Normalize existing parent symlinks and
    // lexical aliases as well as comparing existing inode identities below.
    auto canonical = [](const std::string& path) {
        auto slash = path.rfind('/');
        const auto parent = slash == std::string::npos ? "." :
            slash == 0 ? "/" : path.substr(0, slash);
        const auto name = slash == std::string::npos ? path : path.substr(slash + 1);
        std::unique_ptr<char, decltype(&std::free)> resolved(realpath(parent.c_str(), nullptr), &std::free);
        return resolved ? std::string(resolved.get()) + '/' + name : std::string();
    };
    const auto leftPath = canonical(left), rightPath = canonical(right);
    if (!leftPath.empty() && leftPath == rightPath) return true;
    struct stat a{}, b{};
    return stat(left.c_str(), &a) == 0 && stat(right.c_str(), &b) == 0 &&
        a.st_dev == b.st_dev && a.st_ino == b.st_ino;
}

void help() {
    std::cout << "Usage: nvdsinfer-engine-builder --onnx=MODEL --saveEngine=PLAN [options]\n"
        "Builds an engine only; does not benchmark or run inference.\n"
        "  --fp16                         require a prepared FP16/mixed model on TensorRT 11\n"
        "  --int8                         explicit INT8 Q/DQ boundary required on TensorRT 11\n"
        "    Prepare FP32 ONNX offline with samples/modelopt/prepare_model.py.\n"
        "    --fp16 alone requires HALF value computation; with --int8,\n"
        "    the explicitly quantized graph retains its declared floating-point types\n"
        "  --minShapes=NAME:1x3xHxW,...   all three profile selectors required\n"
        "  --optShapes=... --maxShapes=... (shape-tensor values also supported)\n"
        "  --inputIOFormats=fp32:chw,... --outputIOFormats=fp32:chw,...\n"
        "    linear only; input types must match ONNX; FLOAT/HALF output casts explicit\n"
        "  --plugins=LIB,... --staticPlugins=LIB,... --device=N --noTF32\n"
        "  --memPoolSize=workspace:MiB --builderOptimizationLevel=0..5\n"
        "  --sparsity=enable|disable --profilingVerbosity=detailed|layer_names_only|none\n"
        "  --dumpLayerInfo --exportLayerInfo=FILE --verbose --version --help\n"
        "  --skipInference --buildOnly --useCudaGraph --noDataTransfers are build-only hints\n"
        "Unknown options and legacy calibration are rejected, not ignored.\n";
}

Options parse(int argc, char** argv) {
    Options o;
    std::set<std::string> seen;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        auto eq = arg.find('=');
        std::string key = arg.substr(0, eq);
        std::string value = eq == std::string::npos ? "" : arg.substr(eq + 1);
        const bool flag = key == "--fp16" || key == "--int8" || key == "--noTF32" ||
            key == "--verbose" || key == "--dumpLayerInfo" || key == "--skipInference" ||
            key == "--buildOnly" || key == "--useCudaGraph" || key == "--noDataTransfers";
        if ((flag && eq != std::string::npos) || (!flag && value.empty()))
            throw std::runtime_error("invalid option syntax: " + arg);
        if (key != "--plugins" && key != "--staticPlugins" && !seen.insert(key).second)
            throw std::runtime_error("duplicate option: " + key);
        if (key == "--onnx") o.onnx = value;
        else if (key == "--saveEngine") o.engine = value;
        else if (key == "--fp16") o.fp16 = true;
        else if (key == "--int8") o.int8 = true;
        else if (key == "--noTF32") o.noTF32 = true;
        else if (key == "--verbose") o.verbose = true;
        else if (key == "--device") o.device = integer(value, 0, INT_MAX);
        else if (key == "--minShapes") o.shapes[0] = parseShapes(value);
        else if (key == "--optShapes") o.shapes[1] = parseShapes(value);
        else if (key == "--maxShapes") o.shapes[2] = parseShapes(value);
        else if (key == "--inputIOFormats") o.inputFormats = value;
        else if (key == "--outputIOFormats") o.outputFormats = value;
        else if (key == "--plugins" || key == "--staticPlugins") {
            auto libs = split(value, ',');
            o.plugins.insert(o.plugins.end(), libs.begin(), libs.end());
        } else if (key == "--memPoolSize") {
            if (value.compare(0, 10, "workspace:") != 0)
                throw std::runtime_error("only --memPoolSize=workspace:MiB is supported");
            o.workspace = static_cast<size_t>(integer(value.substr(10), 1,
                std::numeric_limits<size_t>::max() / (1024 * 1024))) * 1024 * 1024;
        } else if (key == "--builderOptimizationLevel") o.optimization = integer(value, 0, 5);
        else if (key == "--sparsity") {
            if (value != "enable" && value != "disable")
                throw std::runtime_error("--sparsity must be enable or disable (no weight rewriting)");
            o.sparse = value == "enable";
        } else if (key == "--profilingVerbosity") {
            if (value != "detailed" && value != "layer_names_only" && value != "none")
                throw std::runtime_error("unsupported profiling verbosity: " + value);
            o.verbosity = value == "detailed" ? ProfilingVerbosity::kDETAILED :
                value == "none" ? ProfilingVerbosity::kNONE : ProfilingVerbosity::kLAYER_NAMES_ONLY;
        } else if (key == "--dumpLayerInfo") o.dumpLayers = true;
        else if (key == "--exportLayerInfo") o.layerInfo = value;
        else if (key == "--skipInference" || key == "--buildOnly" || key == "--useCudaGraph" || key == "--noDataTransfers") {
            // This front end builds only. No inference/transfer/graph-capture measurement occurs.
        } else if (key == "--calib") {
            throw std::runtime_error("legacy calibration is unsupported; supply an explicitly quantized Q/DQ model");
        } else throw std::runtime_error("unknown option: " + key);
    }
    if (o.onnx.empty() || o.engine.empty()) throw std::runtime_error("--onnx and --saveEngine are required");
    if (sameFile(o.onnx, o.engine)) throw std::runtime_error("engine output must not overwrite the input ONNX");
    if (!o.layerInfo.empty() && (sameFile(o.onnx, o.layerInfo) || sameFile(o.engine, o.layerInfo)))
        throw std::runtime_error("layer-info output must be distinct from the ONNX and engine files");
    const bool profile = !o.shapes[0].empty() || !o.shapes[1].empty() || !o.shapes[2].empty();
    if (profile && (o.shapes[0].empty() || o.shapes[1].empty() || o.shapes[2].empty()))
        throw std::runtime_error("minShapes, optShapes and maxShapes must all be supplied");
    return o;
}

// Temporary file is in the destination directory; a failed build/write never
// truncates an existing engine. mkstemp avoids following an attacker-created link.
void atomicWrite(const std::string& path, const void* bytes, size_t size) {
    std::string temporary = path + ".tmp.XXXXXX";
    std::vector<char> name(temporary.begin(), temporary.end());
    name.push_back('\0');
    int fd = mkstemp(name.data());
    if (fd < 0) throw std::runtime_error("cannot create output beside " + path + ": " + std::strerror(errno));
    size_t done = 0;
    int error = 0;
    // mkstemp starts private. Published plans/JSON should retain an existing
    // regular file's permissions, or use the caller's normal creation umask.
    struct stat previous{};
    const mode_t mode = stat(path.c_str(), &previous) == 0 && S_ISREG(previous.st_mode)
        ? previous.st_mode & 0777 : 0666 & ~outputMask;
    if (fchmod(fd, mode) < 0) error = errno;
    while (!error && done < size) {
        ssize_t count = write(fd, static_cast<const char*>(bytes) + done, size - done);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) { error = count < 0 ? errno : EIO; break; }
        done += static_cast<size_t>(count);
    }
    if (!error && fsync(fd) < 0) error = errno;
    if (close(fd) < 0 && !error) error = errno;
    if (!error && rename(name.data(), path.c_str()) < 0) error = errno;
    if (error) {
        unlink(name.data());
        throw std::runtime_error("cannot write " + path + ": " + std::strerror(error));
    }
}

struct PluginLibraries {
    std::vector<void*> handles;
    ~PluginLibraries() { for (auto it = handles.rbegin(); it != handles.rend(); ++it) dlclose(*it); }
    void load(const std::string& path) {
        void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (!handle) throw std::runtime_error("cannot load plugin " + path + ": " + dlerror());
        handles.push_back(handle);
    }
};

void configureProfile(IBuilder& builder, IBuilderConfig& config, INetworkDefinition& n, const Options& o) {
    bool needed = !o.shapes[0].empty();
    for (int i = 0; i < n.getNbInputs(); ++i) {
        auto* input = n.getInput(i);
        needed |= input->isShapeTensor();
        auto dims = input->getDimensions();
        for (int j = 0; j < dims.nbDims; ++j) needed |= dims.d[j] < 0;
    }
    if (!needed) return;
    // TensorRT builder owns profiles; do not delete them.
    auto* profile = builder.createOptimizationProfile();
    if (!profile) throw std::runtime_error("cannot create optimization profile");
    std::set<std::string> names;
    for (int i = 0; i < n.getNbInputs(); ++i) {
        auto* input = n.getInput(i);
        const std::string name(input->getName());
        names.insert(name);
        auto dims = input->getDimensions();
        bool dynamic = input->isShapeTensor();
        for (int j = 0; j < dims.nbDims; ++j) dynamic |= dims.d[j] < 0;
        const bool supplied = o.shapes[0].count(name) || o.shapes[1].count(name) || o.shapes[2].count(name);
        if (!supplied && dynamic) throw std::runtime_error("missing optimization shapes for " + name);
        if (!supplied) continue;
        for (int s = 0; s < 3; ++s) {
            auto found = o.shapes[s].find(name);
            if (found == o.shapes[s].end()) throw std::runtime_error("incomplete profile for " + name);
            const auto& values = found->second;
            if (s && values.size() != o.shapes[0].at(name).size())
                throw std::runtime_error("inconsistent profile rank for " + name);
            for (size_t j = 0; j < values.size(); ++j)
                if (s && values[j] < o.shapes[s - 1].at(name)[j])
                    throw std::runtime_error("profile must satisfy min <= opt <= max for " + name);
            const auto selector = static_cast<OptProfileSelector>(s);
            bool ok = false;
            if (input->isShapeTensor()) {
#if NV_TENSORRT_MAJOR >= 11
                ok = profile->setShapeValuesV2(name.c_str(), selector, values.data(), values.size());
#else
                for (const auto value : values)
                    if (value < std::numeric_limits<int32_t>::min() ||
                        value > std::numeric_limits<int32_t>::max())
                        throw std::runtime_error("TensorRT 10 shape value is outside INT32 range for " + name);
                std::vector<int32_t> values32(values.begin(), values.end());
                ok = profile->setShapeValues(name.c_str(), selector, values32.data(), values32.size());
#endif
            } else {
                if (static_cast<int>(values.size()) != dims.nbDims)
                    throw std::runtime_error("profile rank does not match " + name);
                Dims selected = dims;
                for (int j = 0; j < dims.nbDims; ++j) {
                    if (values[j] <= 0 || (dims.d[j] >= 0 && dims.d[j] != values[j]))
                        throw std::runtime_error("profile disagrees with static/positive dimension for " + name);
                    selected.d[j] = values[j];
                }
                ok = profile->setDimensions(name.c_str(), selector, selected);
            }
            if (!ok) throw std::runtime_error("TensorRT rejected profile for " + name);
        }
    }
    for (const auto& shapes : o.shapes)
        for (const auto& shape : shapes)
            if (!names.count(shape.first)) throw std::runtime_error("unknown profile input: " + shape.first);
    if (!profile->isValid() || config.addOptimizationProfile(profile) < 0)
        throw std::runtime_error("invalid optimization profile");
}

std::vector<DataType> formats(const std::string& text, int count) {
    if (text.empty()) return {};
    std::vector<DataType> result;
    for (const auto& part : split(text, ',')) {
        auto tokens = split(part, ':');
        if (tokens.size() != 2 || (tokens[1] != "chw" && tokens[1] != "linear"))
            throw std::runtime_error("only linear/CHW I/O formats are supported: " + part);
        if (tokens[0] == "fp32") result.push_back(DataType::kFLOAT);
        else if (tokens[0] == "fp16") result.push_back(DataType::kHALF);
        else if (tokens[0] == "int32") result.push_back(DataType::kINT32);
        else if (tokens[0] == "int64") result.push_back(DataType::kINT64);
        else if (tokens[0] == "bool") result.push_back(DataType::kBOOL);
        else throw std::runtime_error("unsupported I/O type: " + tokens[0]);
    }
    if (result.size() == 1) result.resize(count, result[0]);
    if (static_cast<int>(result.size()) != count) throw std::runtime_error("I/O format count does not match network");
    return result;
}

bool floating(DataType t) { return t == DataType::kFLOAT || t == DataType::kHALF; }

void configureIO(INetworkDefinition& n, const Options& o) {
    const auto inputs = formats(o.inputFormats, n.getNbInputs());
    const auto outputs = formats(o.outputFormats, n.getNbOutputs());
    std::set<std::string> names;
    for (int i = 0; i < n.getNbLayers(); ++i)
        for (int j = 0; j < n.getLayer(i)->getNbOutputs(); ++j)
            names.insert(n.getLayer(i)->getOutput(j)->getName());
    for (int i = 0; i < n.getNbInputs(); ++i) names.insert(n.getInput(i)->getName());
    auto unique = [&names](std::string base) {
        while (!names.insert(base).second) base += "_";
        return base;
    };
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto* tensor = n.getInput(i);
        if (tensor->getType() != inputs[i]) {
#if NV_TENSORRT_MAJOR >= 11
            // Recreating parsed inputs can invalidate TensorRT shape intervals.
            // Keep the model's immutable input objects, names, order and types.
            // Internal mixed FP16 does not require HALF external inputs.
            throw std::runtime_error(std::string("input format type differs from ONNX for ") + tensor->getName() +
                "; TensorRT 11 requires a model exported with the requested input type. "
                "Omit the type-changing input format and prepare mixed computation offline with "
                "NVIDIA ModelOpt AutoCast, retaining the original I/O types");
#else
            if (!floating(tensor->getType()) || !floating(inputs[i]) || tensor->isShapeTensor())
                throw std::runtime_error(std::string("cannot change integer/shape input type: ") + tensor->getName());
            tensor->setType(inputs[i]);
#endif
        }
    }
    for (size_t i = 0; i < inputs.size(); ++i)
        n.getInput(i)->setAllowedFormats(1U << static_cast<uint32_t>(TensorFormat::kLINEAR));
    // Unmark/remap all outputs together to retain original order.
    std::vector<ITensor*> tensors;
    std::vector<std::string> outputNames;
    for (size_t i = 0; i < outputs.size(); ++i) {
        tensors.push_back(n.getOutput(i));
        outputNames.emplace_back(n.getOutput(i)->getName());
        if (tensors.back()->getType() != outputs[i] &&
            (!floating(tensors.back()->getType()) || !floating(outputs[i]) || tensors.back()->isShapeTensor()))
            throw std::runtime_error("cannot change integer/shape output type: " + outputNames.back());
        if (tensors.back()->isNetworkInput() && tensors.back()->getType() != outputs[i])
            throw std::runtime_error("changing I/O type on an output also marked as input is unsupported");
    }
    for (auto* tensor : tensors) n.unmarkOutput(*tensor);
    for (size_t i = 0; i < tensors.size(); ++i) {
        auto* tensor = tensors[i];
        if (tensor->getType() != outputs[i]) {
            tensor->setName(unique(outputNames[i] + "_model_output").c_str());
            auto* cast = n.addCast(*tensor, outputs[i]);
            if (!cast) throw std::runtime_error("cannot add output boundary cast");
            tensor = cast->getOutput(0);
            tensor->setName(outputNames[i].c_str());
        }
        n.markOutput(*tensor);
        tensor->setAllowedFormats(1U << static_cast<uint32_t>(TensorFormat::kLINEAR));
    }
}

int build(const Options& o) {
    Logger logger; logger.verbose = o.verbose;
    PluginLibraries plugins; // Must outlive parser, network, engine and runtime.
    for (const auto& path : o.plugins) plugins.load(path);
    auto error = cudaSetDevice(o.device);
    if (error != cudaSuccess) throw std::runtime_error(std::string("cudaSetDevice: ") + cudaGetErrorString(error));
    if (!initLibNvInferPlugins(&logger, "")) throw std::runtime_error("cannot initialize TensorRT plugins");
    std::unique_ptr<IBuilder> builder(createInferBuilder(logger));
    if (!builder) throw std::runtime_error("cannot create TensorRT builder");
#if NV_TENSORRT_MAJOR >= 11
    const uint32_t flags = 0;
#else
    const uint32_t flags = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
#endif
    std::unique_ptr<INetworkDefinition> network(builder->createNetworkV2(flags));
    if (!network) throw std::runtime_error("cannot create network");
    std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
    if (!parser || !parser->parseFromFile(o.onnx.c_str(), static_cast<int>(ILogger::Severity::kWARNING)))
        throw std::runtime_error("ONNX parsing failed: " + o.onnx);
    std::unique_ptr<IBuilderConfig> config(builder->createBuilderConfig());
    if (!config) throw std::runtime_error("cannot create builder configuration");
#if NV_TENSORRT_MAJOR >= 11
    // --int8 selects a genuinely quantized model, including prequantized weights
    // with DQ but no Q layer. A simultaneous --fp16 must not rewrite that model
    // or demand unrelated HALF arithmetic; its floating-point islands keep their
    // declared precision. --fp16 alone requires real HALF value computation,
    // not only HALF casts/constants or a disconnected low-precision branch.
    nvdsinfer::ExplicitBuildParams precision;
    precision.networkMode = o.int8 ? NvDsInferNetworkMode_INT8 :
        o.fp16 ? NvDsInferNetworkMode_FP16 : NvDsInferNetworkMode_FP32;
    if (nvdsinfer::validateTrt11Precision(*network, precision) != NVDSINFER_SUCCESS)
        throw std::runtime_error("model does not implement the requested TensorRT 11 precision; see preceding diagnostics");
#else
    if (o.fp16) config->setFlag(BuilderFlag::kFP16);
    if (o.int8) throw std::runtime_error("use trtexec for TensorRT 10 INT8 calibration; this tool accepts explicit Q/DQ on TensorRT 11");
#endif
    configureIO(*network, o);
    configureProfile(*builder, *config, *network, o);
    if (o.noTF32) config->clearFlag(BuilderFlag::kTF32);
    if (o.sparse) config->setFlag(BuilderFlag::kSPARSE_WEIGHTS);
    if (o.workspace) config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, o.workspace);
    if (o.optimization >= 0) config->setBuilderOptimizationLevel(o.optimization);
    config->setProfilingVerbosity(o.dumpLayers || !o.layerInfo.empty() ? ProfilingVerbosity::kDETAILED : o.verbosity);
    std::cout << "Build only: inference, CUDA graph capture and transfer benchmarks are not performed.\n";
    auto started = std::chrono::steady_clock::now();
    std::unique_ptr<IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
    if (!plan) throw std::runtime_error("TensorRT engine build failed; existing output was not modified");
    if (o.dumpLayers || !o.layerInfo.empty()) {
        std::unique_ptr<IRuntime> runtime(createInferRuntime(logger));
        std::unique_ptr<ICudaEngine> engine(runtime ? runtime->deserializeCudaEngine(plan->data(), plan->size()) : nullptr);
        std::unique_ptr<IEngineInspector> inspector(engine ? engine->createEngineInspector() : nullptr);
        if (!inspector) throw std::runtime_error("cannot inspect built engine");
        const char* text = inspector->getEngineInformation(LayerInformationFormat::kJSON);
        if (!text) throw std::runtime_error("engine inspector returned no layer information");
        if (o.dumpLayers) std::cout << text << '\n';
        if (!o.layerInfo.empty()) atomicWrite(o.layerInfo, text, std::strlen(text));
    }
    atomicWrite(o.engine, plan->data(), plan->size());
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    std::cout << "Saved engine: " << o.engine << " (" << plan->size() << " bytes); build/save " << seconds << " s\n";
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") { help(); return 0; }
    if (argc == 2 && std::string(argv[1]) == "--version") {
        std::cout << "TensorRT " << NV_TENSORRT_MAJOR << '.' << NV_TENSORRT_MINOR << '.'
                  << NV_TENSORRT_PATCH << '.' << NV_TENSORRT_BUILD << '\n';
        return 0;
    }
    // Read/restore once, before plugin loading or TensorRT thread creation.
    outputMask = umask(0);
    umask(outputMask);
    try { return build(parse(argc, argv)); }
    catch (const std::exception& e) { std::cerr << "ERROR: " << e.what() << '\n'; return 1; }
}
