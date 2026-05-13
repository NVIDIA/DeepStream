/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <fstream>
#include <iostream>

#include "infer_cuda_utils.h"
#include "infer_preprocess.h"
#include "infer_surface_bufs.h"
#include "infer_utils.h"

#include "nvtx3/nvToolsExt.h"
#include "nvtx3/nvToolsExtCudaRt.h"

#ifndef printError
#define printError InferError
#endif
namespace nvdsinferserver {

NetworkPreprocessor::NetworkPreprocessor(const NvDsInferNetworkInfo& info,
    InferMediaFormat networkFormat, InferDataType dt, int maxBatchSize)
    : m_NetworkInfo(info),
      m_NetworkFormat(networkFormat),
      m_MaxBatchSize(maxBatchSize),
      m_DstDataType(dt) {
    m_ChannelMeans.assign(info.channels, 0);
}

bool NetworkPreprocessor::setScaleOffsets(
    float scale, const std::vector<float>& offsets) {
    if (!offsets.empty() && m_NetworkInfo.channels > (uint32_t)offsets.size()) {
        return false;
    }

    m_Scale = scale;
    if (!offsets.empty()) {
        m_ChannelMeans.assign(
            offsets.begin(), offsets.begin() + m_NetworkInfo.channels);
    } else {
        m_ChannelMeans.assign(m_NetworkInfo.channels, 0.0f);
    }
    return true;
}

bool NetworkPreprocessor::setMeanFile(const std::string& file) {
    if (!file_accessible(file))
        return false;
    m_MeanFile = file;
    return true;
}

/* Read the mean image ppm file and copy the mean image data to the mean
 * data buffer allocated on the device memory.
 */
NvDsInferStatus NetworkPreprocessor::readMeanImageFile() {
    std::ifstream infile(m_MeanFile, std::ifstream::binary);
    size_t size =
        m_NetworkInfo.width * m_NetworkInfo.height * m_NetworkInfo.channels;
    if (!infile.good()) {
        printError("Could not open mean image file '%s'", safeStr(m_MeanFile));
        return NVDSINFER_CONFIG_FAILED;
    }

    std::string magic;
    uint32_t h, w, maxVal;
    infile >> magic >> w >> h >> maxVal;

    if ((magic != "P3" && magic != "P6") || maxVal > 255) {
        printError("Magic PPM identifier check failed");
        return NVDSINFER_CONFIG_FAILED;
    }

    if (w != m_NetworkInfo.width || h != m_NetworkInfo.height) {
        printError(
            "Mismatch between ppm mean image resolution(%d x %d) and "
            "network resolution(%d x %d)",
            w, h, m_NetworkInfo.width, m_NetworkInfo.height);
        return NVDSINFER_CONFIG_FAILED;
    }

    std::vector<uint8_t> tempMeanDataChar(size);
    infile.get();
    infile.read((char*)tempMeanDataChar.data(), size);
    if (infile.gcount() != (int)size || infile.fail()) {
        printError("Failed to read sufficient bytes from mean file");
        return NVDSINFER_CONFIG_FAILED;
    }

    std::vector<float> meanFloat(size);
    if (m_NetworkTensorOrder == InferTensorOrder::kNHWC) {
        for (size_t i = 0; i < size; i++) {
            meanFloat[i] = (float)tempMeanDataChar[i];
        }
    } else {  // Linear
        uint32_t imgSize = w * h;
        for (uint32_t k = 0; k < m_NetworkInfo.channels; ++k) {
            for (uint32_t j = 0; j < imgSize; j++) {
                meanFloat[k * imgSize + j] =
                    tempMeanDataChar[j * m_NetworkInfo.channels + k];
            }
        }
    }

    /* Mean Image File specified. Allocate the mean image buffer on device
     * memory. */
    m_MeanDataBuffer = std::make_unique<CudaDeviceMem>(size * sizeof(float));
    if (!m_MeanDataBuffer || !m_MeanDataBuffer->ptr()) {
        printError("Failed to allocate cuda buffer for mean image");
        return NVDSINFER_CUDA_ERROR;
    }

    assert(m_MeanDataBuffer);
    RETURN_CUDA_ERR(
        cudaMemcpy(
            m_MeanDataBuffer->ptr(), (const void*)meanFloat.data(),
            size * sizeof(float), cudaMemcpyHostToDevice),
        "Failed to copy mean data(%s) to cuda buffer", safeStr(m_MeanFile));

    return NVDSINFER_SUCCESS;
}

NvDsInferStatus NetworkPreprocessor::allocateResource(
    const std::vector<int>& devIds) {
    assert(devIds.size() == 1);
    RETURN_CUDA_ERR(
        cudaSetDevice(devIds[0]),
        "network preprocess failed to set cuda device(%d) during allocating "
        "resource",
        devIds[0]);
    RETURN_IF_FAILED(
        m_NetworkInfo.height && m_NetworkInfo.width && m_NetworkInfo.channels,
        NVDSINFER_CONFIG_FAILED,
        "network preprocess network_info check failed (h:%u, w:%u, c:%u)",
        m_NetworkInfo.height, m_NetworkInfo.width, m_NetworkInfo.channels);

    int toNCHW = (m_NetworkTensorOrder == InferTensorOrder::kNHWC ? 0 : 1);
    BatchSurfaceInfo dstInfo = {m_NetworkInfo.width, m_NetworkInfo.height,
        m_NetworkInfo.channels,
        (toNCHW ? m_NetworkInfo.width
                : m_NetworkInfo.width * m_NetworkInfo.channels),
        m_NetworkInfo.width * m_NetworkInfo.height * m_NetworkInfo.channels};
    m_DstInfo = dstInfo;

    /* Read the mean image file (PPM format) if specified and copy the
     * contents into the buffer. */
    if (!m_MeanFile.empty()) {
        if (!file_accessible(m_MeanFile)) {
            printError(
                "Cannot access mean image file '%s'", safeStr(m_MeanFile));
            return NVDSINFER_CONFIG_FAILED;
        }
        NvDsInferStatus status = readMeanImageFile();
        if (status != NVDSINFER_SUCCESS) {
            printError("Failed to read mean image file");
            return status;
        }
    }

    assert(m_PoolSize);
    /* Create Cuda Buffer Pool */
    InferDims bufDims = {3,
        {(int)m_NetworkInfo.channels, (int)m_NetworkInfo.height,
            (int)m_NetworkInfo.width},
        0};
    if (!toNCHW) {
        bufDims.d[0] = (int)m_NetworkInfo.height;
        bufDims.d[1] = (int)m_NetworkInfo.width;
        bufDims.d[2] = (int)m_NetworkInfo.channels;
    }
    normalizeDims(bufDims);
    m_BufPool = std::make_shared<BufferPool<UniqCudaTensorBuf>>(
        "NetworkPreprocBufPool");
    assert(m_BufPool);
    for (int iB = 0; iB < m_PoolSize; ++iB) {
        UniqCudaTensorBuf newBuf = createGpuTensorBuf(
            bufDims, m_DstDataType, m_MaxBatchSize, "", devIds[0], true);
        if (!newBuf || !newBuf->cuEvent()) {
            printError("Failed to create cuda tensor buffers");
            return NVDSINFER_CUDA_ERROR;
        }
        /* Cuda event to synchronize between completion of the pre-processing
         * kernels and enqueuing the next set of binding buffers for inference.
         */
        std::string nvtx_name =
            "infer_preprocess_buf_uid=" + std::to_string(uniqueId()) + "_" +
            std::to_string(iB);
        nvtxNameCudaEventA(*newBuf->cuEvent(), nvtx_name.c_str());
        m_BufPool->setBuffer(std::move(newBuf));
    }
    assert(m_BufPool->size() == m_PoolSize);

    /* Create the cuda stream on which pre-processing jobs will be executed. */
    m_PreProcessStream =
        std::make_unique<CudaStream>(cudaStreamNonBlocking, devIds[0]);
    if (!m_PreProcessStream || !m_PreProcessStream->ptr()) {
        printError("Failed to create preprocessor cudaStream");
        return NVDSINFER_CUDA_ERROR;
    }
    std::string nvtx_name =
        "nvdsinfer_preprocess_uid=" + std::to_string(uniqueId());
    nvtxNameCudaStreamA(*m_PreProcessStream, nvtx_name.c_str());

    return NVDSINFER_SUCCESS;
}

SharedBatchBuf NetworkPreprocessor::requestOutBuffer(SharedBatchBuf& inBuf) {
    assert(m_BufPool);
    SharedBatchBuf outBuf = m_BufPool->acquireBuffer();
    outBuf->setBatchSize(inBuf->getBatchSize());
    outBuf->mutableBufDesc().name = m_TensorName;
    return outBuf;
}

NvDsInferStatus NetworkPreprocessor::syncStream() {
    if (m_PreProcessStream) {
        cudaStreamSynchronize(*m_PreProcessStream);
    }
    return NVDSINFER_SUCCESS;
}

static void cudaCallbackReleaseBuf(
    cudaStream_t stream, cudaError_t status, void* userData) {
    SharedBatchBuf* buf = (SharedBatchBuf*)userData;
    delete buf;
}

NvDsInferStatus NetworkPreprocessor::transformImpl(
    SharedBatchBuf& src, SharedBatchBuf& dst, SharedCuStream& mainStream) {
    assert(src && dst);
    InferDebug("NetworkPreprocessor id:%d transform buffer", uniqueId());

    int devId = src->getBufDesc().devId;
    RETURN_CUDA_ERR(
        cudaSetDevice(devId),
        "NetworkPreprocessor failed to set cuda device(%d) during resize "
        "batch",
        devId);

    const InferBufferDescription& srcDesc = src->getBufDesc();
    InferDataType srcDType = srcDesc.dataType;
    BatchSurfaceInfo srcAlignInfo{0};
    InferMediaFormat srcFormat = m_NetworkFormat;
    if (srcDesc.memType == InferMemType::kNvSurface ||
        srcDesc.memType == InferMemType::kGpuCuda ||
        srcDesc.memType == InferMemType::kNvSurfaceArray) {
        assert(std::dynamic_pointer_cast<ImageAlignBuffer>(src));
        auto srcSurface = std::static_pointer_cast<ImageAlignBuffer>(src);
        assert(srcSurface);
        srcAlignInfo = srcSurface->getSurfaceAlignInfo(); // cuda
        srcFormat = srcSurface->getColorFormat();
    } else {
        assert(false);
    }
    // Wait for cuda processing of CropSurfaceConverter to complete
    src->waitForSyncObj();

    if (srcDesc.memType == InferMemType::kNvSurface) {
        for (size_t i = 0; i < src->getBatchSize(); i++) {
            void* srcBuf = src->getBufPtr(i);
            void* dstBuf = dst->getBufPtr(i);
            const InferBufferDescription& dstDesc = dst->getBufDesc();

            NvDsInferStatus status =
                cudaTransform(srcBuf, srcAlignInfo, srcFormat, srcDType, dstBuf,
                        m_DstInfo, dstDesc.dataType, 1, dstDesc.devId);
            if (status != NVDSINFER_SUCCESS) {
                printError("Failed to preprocess buffer during cudaTransform.");
                return status;
            }
        }
    } else {
        void* srcBuf = src->getBufPtr(0);
        void* dstBuf = dst->getBufPtr(0);
        const InferBufferDescription& dstDesc = dst->getBufDesc();

        NvDsInferStatus status =
            cudaTransform(srcBuf, srcAlignInfo, srcFormat, srcDType, dstBuf,
                    m_DstInfo, dstDesc.dataType, src->getBatchSize(), dstDesc.devId);
        if (status != NVDSINFER_SUCCESS) {
            printError("Failed to preprocess buffer during cudaTransform.");
            return status;
        }
    }

    /* Record CUDA event to synchronize the completion of pre-processing
     * kernels. */
    if (dst->cuEvent()) {
        RETURN_CUDA_ERR(cudaEventRecord(*dst->cuEvent(), *m_PreProcessStream),
            "Failed to record cuda network preprocess event");

        if (mainStream) {
            RETURN_CUDA_ERR(
                cudaStreamWaitEvent(*mainStream, *dst->cuEvent(), 0),
                "Failed to make mainstream wait for preprocess event");
        }
    }

    RETURN_CUDA_ERR(
        cudaStreamAddCallback(*m_PreProcessStream, cudaCallbackReleaseBuf,
            (void*)(new SharedBatchBuf(src)), 0),
        "Failed to add cudaStream callback for returning input buffers");

    return NVDSINFER_SUCCESS;
}

NvDsInferStatus NetworkPreprocessor::cudaTransform(void* srcBuf,
    const BatchSurfaceInfo& srcInfo, InferMediaFormat srcFormat,
    InferDataType srcDType, void* dstBuf, const BatchSurfaceInfo& dstInfo,
    InferDataType dstDType, int batchSize, int devId) {
    assert(srcFormat != InferMediaFormat::kUnknown);
    assert(
        srcDType == InferDataType::kInt8 || srcDType == InferDataType::kUint8);
    assert(dstDType != InferDataType::kNone);
    assert(devId >= 0);

    RETURN_CUDA_ERR(
        cudaSetDevice(devId), "network preprocess failed toset cuda device(%d)",
        devId);

    int fromChannelIndices[NVDSINFER_MAX_PREPROC_CHANNELS] = {0, 1, 2, 3};
    int toNCHW = (m_NetworkTensorOrder == InferTensorOrder::kLinear ? 1 : 0);

    /* Find the required conversion function. */
    switch (m_NetworkFormat) {
    case InferMediaFormat::kRGB:
        switch (srcFormat) {
        case InferMediaFormat::kRGB:
        case InferMediaFormat::kRGBA:
            break;
        case InferMediaFormat::kBGR:
        case InferMediaFormat::kBGRx:
            fromChannelIndices[0] = 2;
            fromChannelIndices[2] = 0;
            break;
        default:
            printError("Input format conversion is not supported");
            return NVDSINFER_INVALID_PARAMS;
        }
        break;
    case InferMediaFormat::kBGR:
        switch (srcFormat) {
        case InferMediaFormat::kRGB:
        case InferMediaFormat::kRGBA:
            fromChannelIndices[0] = 2;
            fromChannelIndices[2] = 0;
            break;
        case InferMediaFormat::kBGR:
        case InferMediaFormat::kBGRx:
            break;
        default:
            printError("Input format conversion is not supported");
            return NVDSINFER_INVALID_PARAMS;
        }
        break;
    case InferMediaFormat::kGRAY:
        if (srcFormat != InferMediaFormat::kGRAY) {
            printError("Input frame format is not GRAY.");
            return NVDSINFER_INVALID_PARAMS;
        }
        break;
    default:
        printError("Unsupported network input format");
        return NVDSINFER_INVALID_PARAMS;
    }

    /* For each frame in the input batch convert/copy to the input binding
     * buffer. */
    int cropH = m_NetworkInfo.height;
    int cropW = m_NetworkInfo.width;
    bool convertRes = true;
    if (m_MeanFile.empty()) {
        float meanOffsets[NVDSINFER_MAX_PREPROC_CHANNELS] = {0.0};
        for (size_t i = 0;
             i < m_ChannelMeans.size() && i < NVDSINFER_MAX_PREPROC_CHANNELS;
             ++i) {
            meanOffsets[i] = m_ChannelMeans[i];
        }
        convertRes = NvDsInferConvert_CxToPx(dstBuf, static_cast<int>(dstDType),
            dstInfo, srcBuf, static_cast<int>(srcDType), srcInfo, batchSize,
            m_NetworkInfo.channels, cropH, cropW, m_Scale, meanOffsets,
            fromChannelIndices, toNCHW, *m_PreProcessStream);
    } else {
        convertRes = NvDsInferConvert_CxToPxWithMeanBuffer(dstBuf,
            static_cast<int>(dstDType), dstInfo, srcBuf,
            static_cast<int>(srcDType), srcInfo, batchSize,
            m_NetworkInfo.channels, cropH, cropW, m_Scale,
            m_MeanDataBuffer->ptr<float>(), fromChannelIndices, toNCHW,
            *m_PreProcessStream);
    }
    if (!convertRes) {
        printError("Failed to convert input-format to network format");
        return NVDSINFER_CUDA_ERROR;
    }

    return NVDSINFER_SUCCESS;
}

CropSurfaceConverter::~CropSurfaceConverter() {
    delete[] m_TransformParam.src_rect;
    delete[] m_TransformParam.dst_rect;
}

void CropSurfaceConverter::setParams(
    int outW, int outH, InferMediaFormat outFormat, int maxBatchSize) {
    m_DstWidth = outW;
    m_DstHeight = outH;
    m_DstFormat = outFormat;
    m_MaxBatchSize = maxBatchSize;
}

NvDsInferStatus CropSurfaceConverter::allocateResource(
    const std::vector<int>& devIds) {
    assert(devIds.size() == 1);
    RETURN_CUDA_ERR(
        cudaSetDevice(devIds[0]),
        "network preprocess failed to set cuda device(%d) during allocating "
        "resource",
        devIds[0]);
    RETURN_IF_FAILED(
        m_DstWidth && m_DstHeight && m_MaxBatchSize &&
            m_DstFormat != InferMediaFormat::kUnknown,
        NVDSINFER_CONFIG_FAILED,
        "CropSurfaceConverter dst info check failed (w:%u, h:%u, b:%u)",
        m_DstWidth, m_DstHeight, m_MaxBatchSize);

    m_ConvertPool = createSurfaceBufPool(m_DstWidth, m_DstHeight, m_DstFormat,
        m_MaxBatchSize, devIds[0], m_ConvertPoolSize);
    assert(m_ConvertPool->size() == m_ConvertPoolSize);

    assert(m_MaxBatchSize);
    m_TransformParam.src_rect = new NvBufSurfTransformRect[m_MaxBatchSize];
    m_TransformParam.dst_rect = new NvBufSurfTransformRect[m_MaxBatchSize];
    m_TransformParam.transform_flag = NVBUFSURF_TRANSFORM_FILTER |
                                      NVBUFSURF_TRANSFORM_CROP_SRC |
                                      NVBUFSURF_TRANSFORM_CROP_DST;
    m_TransformParam.transform_flip = NvBufSurfTransform_None;

    m_ConverStream =
        std::make_shared<CudaStream>(cudaStreamNonBlocking, devIds[0]);
    assert(m_ConverStream);

    return NVDSINFER_SUCCESS;
}

SharedBatchBuf CropSurfaceConverter::requestOutBuffer(SharedBatchBuf& inBuf) {
    assert(inBuf);
    // bypass if inBuf is not a surface buf
    if (inBuf->getBufDesc().memType != InferMemType::kNvSurface &&
        inBuf->getBufDesc().memType != InferMemType::kNvSurfaceArray) {
        return inBuf;
    }
    auto dst = m_ConvertPool->acquireBuffer();
    assert(inBuf->getBatchSize() <= dst->getReservedSize());
    dst->setBatchSize(inBuf->getBatchSize());
    return std::move(dst);
}

NvDsInferStatus CropSurfaceConverter::transformImpl(
    SharedBatchBuf& src, SharedBatchBuf& dst, SharedCuStream& mainStream) {
    INFER_UNUSED(mainStream);
    // bypass if src is same as dst
    if (src.get() == dst.get()) {
        return NVDSINFER_SUCCESS;
    }
    return resizeBatch(src, dst);
}

NvDsInferStatus CropSurfaceConverter::resizeBatch(
    SharedBatchBuf& src, SharedBatchBuf& dst) {
    assert(m_ConverStream);
    InferDebug("NetworkPreprocessor id:%d resize batch buffer", uniqueId());

    int devId = dst->getBufDesc().devId;
    RETURN_CUDA_ERR(
        cudaSetDevice(devId),
        "CropSurfaceConverter failed to set cuda device(%d) during resize "
        "batch",
        devId);

    NvBufSurfTransformConfigParams configParams{
        m_ComputeHW, devId, m_ConverStream->ptr()};
    std::shared_ptr<BatchSurfaceBuffer> srcSurface =
        std::static_pointer_cast<BatchSurfaceBuffer>(src);
    assert(srcSurface);
    uint32_t frameCount = srcSurface->getBatchSize();
    assert(frameCount <= m_MaxBatchSize);
    std::shared_ptr<SurfaceBuffer> dstSurface =
        std::static_pointer_cast<SurfaceBuffer>(dst);
    assert(dstSurface);
    assert(frameCount <= dstSurface->getReservedSize());
    dstSurface->setBatchSize(frameCount);
    NvBufSurface* nvDstBuf = dstSurface->getBufSurface();
    NvBufSurfTransformSyncObj_t syncObj = nullptr;

    for (uint32_t i = 0; i < frameCount; ++i) {
        NvBufSurfTransformRect rect = srcSurface->getCropArea(i);
        int srcL = INFER_ROUND_UP(rect.left, 2);
        int srcT = INFER_ROUND_UP(rect.top, 2);
        int srcW = INFER_ROUND_DOWN(rect.width, 2);
        int srcH = INFER_ROUND_DOWN(rect.height, 2);
        m_TransformParam.src_rect[i].left = srcL;
        m_TransformParam.src_rect[i].top = srcT;
        m_TransformParam.src_rect[i].width = srcW;
        m_TransformParam.src_rect[i].height = srcH;
        int dstW = m_DstWidth, dstH = m_DstHeight;
        if (m_MaintainAspectRatio) {
            double hdest = m_DstWidth * srcH / (double)srcW;
            double wdest = m_DstHeight * srcW / (double)srcH;

            if (hdest <= m_DstHeight) {
                dstH = hdest;
            } else {
                dstW = wdest;
            }
        }
        dstW = INFER_ROUND_DOWN(dstW, 2);
        dstH = INFER_ROUND_DOWN(dstH, 2);

        double scaleX = (double)dstW / srcW;
        double scaleY = (double)dstH / srcH;
        uint32_t offsetLeft = 0, offsetRight = 0;
        uint32_t offsetTop = 0 , offsetBottom = 0;
        srcSurface->setScaleRatio(i, scaleX, scaleY);

        if(m_SymmetricPadding) {
            offsetLeft = (m_DstWidth - dstW)/2;
            offsetRight = m_DstWidth - dstW - offsetLeft;
            offsetTop = (m_DstHeight - dstH)/2;
            offsetBottom = m_DstHeight - dstH - offsetTop;
        }
        srcSurface->setOffsets(i, offsetLeft, offsetTop);

        NvBufSurfTransformRect dstRect{offsetTop, offsetLeft, (uint32_t)dstW, (uint32_t)dstH};
        m_TransformParam.dst_rect[i] = dstRect;

        int pixel_size = 1;
        switch (nvDstBuf->surfaceList[i].colorFormat) {
        case NVBUF_COLOR_FORMAT_RGBA:
            pixel_size = 4;
            break;
        case NVBUF_COLOR_FORMAT_RGB:
            pixel_size = 3;
            break;
        case NVBUF_COLOR_FORMAT_GRAY8:
        case NVBUF_COLOR_FORMAT_NV12:
            pixel_size = 1;
            break;
        default:
            assert(false);
            break;
        }
        if(!m_SymmetricPadding) {
            // set padding are to 0
            if (dstW < m_DstWidth) {
                RETURN_CUDA_ERR(
                    cudaMemset2DAsync(
                        (uint8_t*)dst->getBufPtr(i) + pixel_size * dstW,
                        nvDstBuf->surfaceList[i].planeParams.pitch[0], 0,
                        pixel_size * (m_DstWidth - dstW), dstH, *m_ConverStream),
                    "cudaMemset2DAsync failed to set 0 to scaled padding area");
            }

            if (dstH < m_DstHeight) {
                RETURN_CUDA_ERR(
                    cudaMemset2DAsync(
                        (uint8_t*)dst->getBufPtr(i) +
                            nvDstBuf->surfaceList[i].planeParams.pitch[0] * dstH,
                        nvDstBuf->surfaceList[i].planeParams.pitch[0], 0,
                        pixel_size * m_DstWidth, m_DstHeight - dstH, *m_ConverStream),
                    "cudaMemset2DAsync failed to set 0 to scaled padding area");
            }
        }
        else {
            /* Symmetric Padding */
            if (dstW < m_DstWidth) {
                RETURN_CUDA_ERR(
                    cudaMemset2DAsync(
                        (uint8_t*)dst->getBufPtr(i),
                        nvDstBuf->surfaceList[i].planeParams.pitch[0], 0,
                        pixel_size * offsetLeft, dstH, *m_ConverStream),
                    "cudaMemset2DAsync failed to set 0 to scaled padding area");

                RETURN_CUDA_ERR(
                    cudaMemset2DAsync(
                        (uint8_t*)dst->getBufPtr(i) +
                        pixel_size * (dstW + offsetLeft),
                        nvDstBuf->surfaceList[i].planeParams.pitch[0], 0,
                        pixel_size * offsetRight, dstH, *m_ConverStream),
                    "cudaMemset2DAsync failed to set 0 to scaled padding area");
            }

            if (dstH < m_DstHeight) {
                RETURN_CUDA_ERR(
                    cudaMemset2DAsync(
                        (uint8_t*)dst->getBufPtr(i),
                        nvDstBuf->surfaceList[i].planeParams.pitch[0], 0,
                        pixel_size * m_DstWidth, offsetTop, *m_ConverStream),
                    "cudaMemset2DAsync failed to set 0 to scaled padding area");

                RETURN_CUDA_ERR(
                    cudaMemset2DAsync(
                        (uint8_t*)dst->getBufPtr(i) +
                            nvDstBuf->surfaceList[i].planeParams.pitch[0] *
                            (dstH + offsetTop),
                        nvDstBuf->surfaceList[i].planeParams.pitch[0], 0,
                        pixel_size * m_DstWidth, offsetBottom, *m_ConverStream),
                    "cudaMemset2DAsync failed to set 0 to scaled padding area");
            }

        }
    }
    NvBufSurfTransform_Error err =
        NvBufSurfTransformSetSessionParams(&configParams);
    if (err != NvBufSurfTransformError_Success) {
        printError(
            "NvBufSurfTransformSetSessionParams failed with error %d", err);
        return NVDSINFER_INVALID_PARAMS;
    }

    err = NvBufSurfTransformAsync(srcSurface->getBufSurface(),
        dstSurface->getBufSurface(), &m_TransformParam, &syncObj);
    if (err != NvBufSurfTransformError_Success) {
        printError(
            "NvBufSurfTransform failed with error %d while converting buffer",
            err);
        return NVDSINFER_INVALID_PARAMS;
    }
    assert(syncObj != nullptr);
    dst->setSyncObj(syncObj);

    return NVDSINFER_SUCCESS;
}

} // namespace nvdsinferserver
