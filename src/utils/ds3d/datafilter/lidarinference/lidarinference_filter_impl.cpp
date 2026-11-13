/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "lidarinference_filter_impl.h"
#include "ds3d/common/impl/impl_frames.h"

#include <unistd.h>

#include "infer_proto_utils.h"
#include "infer_options.h"
#include "infer_cuda_utils.h"

using namespace nvdsinferserver;

namespace ds3d { namespace impl { namespace filter {

void lidarInferenceLog(NvDsInferLogLevel log_level, const char* log_message);

ErrCode
LidarInferenceFilter::reserveInputMem(uint& devId, uint32_t count, int& batchSize) 
{
    for (auto iter : _config.inputLayersDes) {
        const auto& des = iter.second;
        auto pool = std::make_shared<BufferPool<UniqCudaTensorBuf>>(des.name);
        DS_ASSERT(pool);
        InferMemType memType = _config.inputTensorMemType;
        LOG_DEBUG("reserveInputMem mt:%d\n", (int)memType);
        for (uint32_t i = 0; i < count; ++i) {
            //createTensorBuf only support gpucuda  and cpucuda.
            UniqCudaTensorBuf ptr =
                createTensorBuf(des.dims, des.dataType, batchSize, des.name, memType, devId, false);
            pool->setBuffer(std::move(ptr));
        }
        DS3D_FAILED_RETURN(
            pool->size() == count, ErrCode::kMem, "allocate memory failed on %s", des.name.c_str());
        _inputBuferPoolMap[des.name] = pool;
    }
    return ErrCode::kGood;
}

LidarInferenceFilter::~LidarInferenceFilter()
{
    stop_i();
    _InferCtx.reset();
}

static ErrCode
getMemKind(MemType from, InferMemType to, cudaMemcpyKind& kind)
{
    using Key = std::tuple<int, int>;
    static std::map<Key, cudaMemcpyKind> kMemCopyTable = {
        {std::make_tuple((int)MemType::kGpuCuda, (int)InferMemType::kGpuCuda),
         cudaMemcpyDeviceToDevice},
        {std::make_tuple((int)MemType::kGpuCuda, (int)InferMemType::kCpu), cudaMemcpyDeviceToHost},
        {std::make_tuple((int)MemType::kCpu, (int)InferMemType::kGpuCuda), cudaMemcpyHostToDevice},
        {std::make_tuple((int)MemType::kCpu, (int)InferMemType::kCpu), cudaMemcpyHostToHost},
    };

    auto fromTypeFun = [](MemType t) -> int {
        if (t == MemType::kCpuPinned) {
            return static_cast<int>(MemType::kCpu);
        }
        return static_cast<int>(t);
    };

    auto ToTypeFun = [](InferMemType t) -> int {
        if (t == InferMemType::kCpuCuda) {
            return static_cast<int>(InferMemType::kCpu);
        }
        if (t == InferMemType::kNvSurface || t == InferMemType::kNvSurfaceArray) {
            return static_cast<int>(InferMemType::kNone);
        }
        return static_cast<int>(t);
    };

    Key k = std::make_tuple(fromTypeFun(from), ToTypeFun(to));
    DS3D_FAILED_RETURN(
        kMemCopyTable.count(k), ErrCode::kParam,
        "cuda mem copy type is not supported. from: %d to %d", static_cast<int>(from),
        static_cast<int>(to));

    assert(kMemCopyTable.count(k));
    kind = kMemCopyTable[k];
    return ErrCode::kGood;
}

ErrCode
LidarInferenceFilter::copy2DFrameToTensor(Frame2DGuard& frame, SharedBatchBuf& tensor)
{
    assert(frame && tensor);
    uint8_t* framePtr = (uint8_t*)frame->base();
    uint8_t* tensorPtr = (uint8_t*)tensor->getBufPtr(0);
    const auto& tensorDesc = tensor->getBufDesc();
    uint64_t totalBytes = tensor->getTotalBytes();
    uint64_t leftBytes = totalBytes;
    // TODO: support different kind of device copy
    cudaMemcpyKind cpKind = cudaMemcpyDeviceToDevice;
    DS3D_ERROR_RETURN(
        getMemKind(frame->memType(), tensorDesc.memType, cpKind),
        "copy2DFrameToTensor get memory type failed");

    DS_ASSERT(_cudaStream);
    for (uint32_t i = 0; i < frame->planes(); ++i) {
        const auto& plane = frame->getPlane(i);
        uint8_t* planePtr = framePtr + plane.offset;
        uint64_t validWidthBytes = plane.bytesPerPixel * plane.width;
        DS3D_FAILED_RETURN(
            validWidthBytes * plane.height <= leftBytes, ErrCode::kMem,
            "Frame pixel bytes is larger than allocated input tensor: %s total bytes: %ld",
            tensorDesc.name.c_str(), totalBytes);
        cudaError_t cudaErr = cudaMemcpy2DAsync(
            (void*)tensorPtr, validWidthBytes, (void*)planePtr, plane.pitchInBytes, validWidthBytes,
            plane.height, cpKind, _cudaStream->get());
        DS3D_FAILED_RETURN(
            cudaErr == cudaSuccess, ErrCode::kCuda,
            "copy2DFrameToTensor failed with cuda error: %s", cudaGetErrorString(cudaErr));
        tensorPtr += validWidthBytes * plane.height;
    }
    return ErrCode::kGood;
}

ErrCode
LidarInferenceFilter::copyFrameToTensor(FrameGuard& frame, SharedBatchBuf& tensor)
{
    assert(frame && tensor);
    void* framePtr = frame->base();
    void* tensorPtr = tensor->getBufPtr(0);
    const auto& tensorDesc = tensor->getBufDesc();
    uint64_t tensorBytes = tensor->getTotalBytes();
    uint64_t frameBytes = frame->bytes();
    // TODO: support different kind of device copy
    cudaMemcpyKind cpKind = cudaMemcpyHostToDevice;
    DS3D_ERROR_RETURN(
        getMemKind(frame->memType(), tensorDesc.memType, cpKind),
        "copy2DFrameToTensor get memory type failed");

    DS3D_FAILED_RETURN(
        frameBytes <= tensorBytes, ErrCode::kMem,
        "Frame bytes: %ld is larger than allocated input tensor: %s total bytes: %ld", frameBytes,
        tensorDesc.name.c_str(), tensorBytes);

    DS_ASSERT(_cudaStream);
    cudaError_t cudaErr =
        cudaMemcpyAsync(tensorPtr, framePtr, frameBytes, cpKind, _cudaStream->get());
    DS3D_FAILED_RETURN(
        cudaErr == cudaSuccess, ErrCode::kCuda, "copy2DFrameToTensor failed with cuda error: %s",
        cudaGetErrorString(cudaErr));
    return ErrCode::kGood;
}

ErrCode
LidarInferenceFilter::copyDatamapToInputs(GuardDataMap datamap, SharedBatchArray batchArray)
{
    for (auto input : batchArray->bufs()) {
        const auto& desc = input->getBufDesc();
        assert(_config.inputLayersDes.count(desc.name));
        auto& layer = _config.inputLayersDes[desc.name];
        DS3D_FAILED_RETURN(
            !layer.fromKey.empty(), ErrCode::kConfig,
            "ds3d inference failed since key not configured in model_inputs tensor: %s",
            layer.name.c_str());

        if (layer.is2DFrame) {
            Frame2DGuard frame2D;
            DS3D_ERROR_RETURN(
                datamap.getGuardData(layer.fromKey, frame2D),
                "No 2D frame: %s found in datamap to copy into input: %s", layer.fromKey.c_str(),
                desc.name.c_str());
            assert(frame2D);
            DS3D_ERROR_RETURN(
                copy2DFrameToTensor(frame2D, input),
                "copy 2D frame: %s to input tensor: %s failed.", layer.fromKey.c_str(),
                desc.name.c_str());
        } else {
            FrameGuard frame;
            DS3D_ERROR_RETURN(
                datamap.getGuardData(layer.fromKey, frame),
                "No frame: %s found in datamap to copy into input: %s", layer.fromKey.c_str(),
                desc.name.c_str());
            assert(frame);
            DS3D_ERROR_RETURN(
                copyFrameToTensor(frame, input), "copy frame: %s to input tensor: %s failed.",
                layer.fromKey.c_str(), desc.name.c_str());
        }
    }

    return ErrCode::kGood;
}

ErrCode
LidarInferenceFilter::processImpl(
    GuardDataMap datamap, OnGuardDataCBImpl outputDataCb, OnGuardDataCBImpl inputConsumedCb)
{
    LOG_INFO("lidarinference datafilter is starting to filter datamap");
    std::unique_lock<ImplMutex> locker(mutex());
    FrameGuard lidarFrame;

    // get data from datamap
    const std::string& filterInputDatamapKey = _config.filterInputDatamapKey;

    if (!filterInputDatamapKey.empty()) {
        DS3D_FAILED_RETURN(
            datamap.hasData(filterInputDatamapKey), ErrCode::kConfig,
            "lidarinference configured datamap key but no lidar frame found in datamap");

        DS3D_ERROR_RETURN(
            datamap.getGuardData(filterInputDatamapKey, lidarFrame),
            "No lidar data found in datamap from lidarinference filter");
    }

    //prepare buffer for inference
    SharedBatchArray batchArray = std::make_shared<BaseBatchArray>();
    for (auto iter : _config.inputLayersDes) {
        const auto& des = iter.second;
        auto bufferPool = _inputBuferPoolMap[des.name];
        auto buffer = bufferPool->acquireBuffer();
        batchArray->addBuf(std::move(buffer));
    }

    // attach option for preprocess and postprocess
    SharedBufOptions option = std::make_shared<BufOptions>();
    // keep dmInOptions alive untill inference done
    GuardDataMap dmInOptions = datamap;
    if(option){
        const std::string& inferConfig = _config.compConfig.configBody;
        // keep backward compatibility for kLidarInferenceParas
        option->setValue(kLidarInferenceParas, inferConfig);
        option->setValue(kDs3dInferenceParas, inferConfig);
        if (!filterInputDatamapKey.empty()) {
            option->setValue(kLidarXYZI, filterInputDatamapKey);
        }
        option->setValue(kLidarRefDataMap, dmInOptions.abiRef());
        batchArray->setOptions(option);
    }

    // do preprocess by upper layer
    if(_customPreprocess){
        DS3D_FAILED_RETURN(
            _customPreprocess->preproc(datamap, batchArray, _cudaStream->get()) ==
                NVDSINFER_SUCCESS,
            ErrCode::kUnknown, "ds3d inference custom preprocess failed");
    } else {
        DS3D_ERROR_RETURN(
            copyDatamapToInputs(datamap, batchArray),
            "ds3d inference copy inputs from datamap failed");
    }

    DS3D_CHECK_CUDA_ERROR(
        cudaStreamSynchronize(_cudaStream->get()), return ErrCode::kCuda,
        "Failed to synchronize cudaStream on ds3d inference filter for preprocess");

    //do inference
    auto promise = std::make_shared<std::promise<void>>();
    std::future<void> future = promise->get_future();
    NvDsInferStatus status = NVDSINFER_SUCCESS;
    std::shared_ptr<BaseBatchArray> bufArray;
    NvDsInferStatus runStatus = _InferCtx->run(
        std::move(batchArray), [&status, &bufArray, promise, dmInOptions](
                                   NvDsInferStatus s, SharedIBatchArray out) mutable {
            status = s;
            bufArray = std::dynamic_pointer_cast<BaseBatchArray>(out);
            dmInOptions.reset();
            out.reset();
            promise->set_value();
            LOG_DEBUG("inference done, ret: %d", status);
        });

    locker.unlock();
    if (runStatus == NVDSINFER_SUCCESS) {
        future.wait();
    }

    LOG_DEBUG(
        "inference runStatus: %d, doneStatus: %d, layerSize: %d", runStatus, status,
        bufArray->getSize());
    ErrCode ret = ErrCode::kGood;
    if (status != NVDSINFER_SUCCESS || runStatus != NVDSINFER_SUCCESS) {
        LOG_ERROR(
            "ds3d inference failed, runStatus: %d, doneStatus: %d, layerSize: %d", runStatus,
            status, bufArray->getSize());
        ret = ErrCode::kUnknown;
    } else {
        for(uint32_t i = 0; i < bufArray->getSize(); ++i) {
            const auto& buf = bufArray->getBuffer(i);
            assert(buf);
            int bufLen = buf->getTotalBytes();
            const InferBufferDescription& desc = buf->getBufDesc();
            LOG_DEBUG(
                "output layers, name:%s, dataType: %d, memType: %d, bufLen: %d", desc.name.c_str(),
                (int)desc.dataType, (int)desc.memType, bufLen);
         }
    }

    inputConsumedCb(ret, datamap);
    outputDataCb(ret, datamap);

    return ret;
}

ErrCode
LidarInferenceFilter::stopImpl()
{
    LOG_INFO("lidarinference datafilter is closing");
    std::unique_lock<ImplMutex> locker(mutex());
    if (_cudaStream) {
        DS3D_CHECK_CUDA_ERROR(
            cudaStreamSynchronize(_cudaStream->get()), return ErrCode::kCuda,
            "Failed to synchronize cuda stream on ds3d inference filter");
    }

    if (_InferCtx) {
        _InferCtx->deinit();
        // do not reset _InferCtx since custom postprocess lib still needed to destory buffers
    }
    _customPreprocess.reset();

    for(auto buffer:_inputBuferPoolMap){
        (buffer.second).reset();
    }
    _inputBuferPoolMap.clear();
    _cudaStream.reset();

    return ErrCode::kGood;
}

ErrCode
LidarInferenceFilter::flushImpl()
{
    LOG_INFO("lidarinference datafilter is flushing");

    std::unique_lock<ImplMutex> locker(mutex());

    return ErrCode::kGood;
}

void
 lidarInferenceLog(NvDsInferLogLevel log_level, const char* log_message)
 {
    switch (log_level) {
    case NVDSINFER_LOG_ERROR:
         LOG_ERROR("%s",log_message);
        return;
    case NVDSINFER_LOG_WARNING:
        LOG_WARNING("%s",log_message);
    return;
    case NVDSINFER_LOG_INFO:
        LOG_INFO("%s",log_message);
    return;
    case NVDSINFER_LOG_DEBUG:
        LOG_DEBUG("%s",log_message);
    return;
    }
}

ErrCode
LidarInferenceFilter::startImpl(const std::string& content, const std::string& path)
{
    NvDsInferStatus status = NVDSINFER_SUCCESS;

    LOG_INFO("lidarinference datafilter is starting");
    std::unique_lock<ImplMutex> locker(mutex());
    ErrCode code =
        config::CatchYamlCall([&, this]() { return parseConfig(content, path, _config); });
    DS3D_ERROR_RETURN(code, "parse datafilter lidarinference config: %s failed", path.c_str());
    DS3D_FAILED_RETURN(
        _config.check(), ErrCode::kConfig, "check datafilter lidarinference config: %s failed",
        path.c_str());

    //get preprocess callback function
    if(_config.customPreprocessLibHandle && !_config.customPreprocessFuncName.empty()){
        auto funcPtr = reinterpret_cast<CreateCustomProcessorFunc>(
            dlsym(_config.customPreprocessLibHandle, _config.customPreprocessFuncName.c_str()));
        _customPreprocess.reset(funcPtr());
        LOG_INFO("get preprocess callback successfully");
    }

    std::string prototxt;
    if (!readFile(_config.nvinferserverCfg, prototxt)) {
        DS3D_FAILED_RETURN(false, ErrCode::kConfig,
           "Configuration file read failed");
    }

    if (!parsePluginControl(prototxt, _PluginConfig)) {
        DS3D_FAILED_RETURN(false, ErrCode::kConfig,
           "Configuration file parsing failed");
    }

    //validate
    std::string contextStr;
    if (!formatInferenceConfig(_PluginConfig.infer_config(), contextStr)) {
        DS3D_FAILED_RETURN(false, ErrCode::kConfig,
           "Inference configuration serialization failed");
    }
    SharedInferContext context;

    if (_PluginConfig.infer_config().has_backend()) {
        const ic::BackendParams &params = _PluginConfig.infer_config().backend();
        if (params.has_triton()) {
            const ic::TritonParams& tritonParams = getTritonParam(params);
            if(tritonParams.has_grpc() && tritonParams.has_model_repo()) {
                DS3D_FAILED_RETURN(false, ErrCode::kConfig,
                   "Configuration either native or grpc should be provided");
            } else if (tritonParams.has_grpc()) {
                context = std::shared_ptr<dsis::IInferContext>(
                createInferTritonGrpcContext(contextStr.c_str(), contextStr.size()));
            } else {
                context = std::shared_ptr<dsis::IInferContext>(
                createInferTrtISContext(contextStr.c_str(), contextStr.size()));
            }
        } else {
            DS3D_FAILED_RETURN(false, ErrCode::kConfig,
                "Configuration config triton is missing");
        }
    } else {
        DS3D_FAILED_RETURN(false, ErrCode::kConfig,
            "Configuration config backend is missing");
    }

    auto logger = [ ]( NvDsInferLogLevel level, const char* msg) {
        lidarInferenceLog(level, msg);
    };
    status = context->initialize(contextStr, logger);
    if (status != NVDSINFER_SUCCESS) {
        DS3D_FAILED_RETURN(false, ErrCode::kConfig,
            "Failed to initialize InferTrtIsContext");
    }
    _InferCtx = context;

    _cudaStream.reset(new CudaStream(cudaStreamDefault, 0));
    DS_ASSERT(_cudaStream);

    //prepare bufferpool for model input data.
    int batchSize = 0;  //0 means no batching
    reserveInputMem(_config.gpuid, _config.memPoolSize, batchSize);

    setInputCaps(_config.compConfig.gstInCaps);
    setOutputCaps(_config.compConfig.gstOutCaps);

    return ErrCode::kGood;
}

}}}  // namespace ds3d::impl::filter

using namespace ds3d;

DS3D_EXTERN_C_BEGIN
DS3D_EXPORT_API abiRefDataFilter*
createLidarInferenceFilter()
{
    return NewAbiRef<abiDataFilter>(new impl::filter::LidarInferenceFilter);
}

DS3D_EXTERN_C_END
