/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

/**
 * This is a header file for pre-processing cuda kernels with normalization and
 * mean subtraction required by nvdsinfer.
 */

#include "infer_surface_bufs.h"
#include "infer_cuda_utils.h"
#include "infer_iprocess.h"
#include "nvdsinfer.h"

#include <nvbufsurface.h>

namespace nvdsinferserver {

SurfaceBuffer::SurfaceBuffer(int batchSize)
    : ImageAlignBuffer(batchSize), m_ReservedSize(batchSize)
{
}

SurfaceBuffer::~SurfaceBuffer() {
#ifdef __aarch64__
    for (size_t i = 0; i < m_CudaResources.size(); i++) {
        cuGraphicsUnregisterResource(m_CudaResources[i]);
    }
#endif

    if (m_Surf) {
        NvBufSurfaceUnMapEglImage(m_Surf, -1);
        NvBufSurfaceDestroy(m_Surf);
    }
}

bool SurfaceBuffer::init(
    int width, int height, InferMediaFormat format, int gpuId) {
    NvBufSurfaceColorFormat color{NVBUF_COLOR_FORMAT_INVALID};
    int channel = 0;
    bool isNCHW = true;
    int is_nvgpu = 0;

    NvBufSurfaceDeviceInfo dev_info{};
    if (NvBufSurfaceGetDeviceInfo(&dev_info) == 0) {
      if (dev_info.driverType == NVBUF_DRIVER_TYPE_NVGPU) {
        is_nvgpu = 1;
      }
    }

    switch (format) {
    case InferMediaFormat::kRGB:
    case InferMediaFormat::kBGR:
        if (dev_info.isVicPresent) {
            color = NVBUF_COLOR_FORMAT_RGBA;
            channel = 4;
            m_ColorFormat = InferMediaFormat::kRGBA;
        } else {
            color = NVBUF_COLOR_FORMAT_RGB;
            m_ColorFormat = InferMediaFormat::kRGB;
            channel = 3;
        }
        isNCHW = false;
        break;
    case InferMediaFormat::kGRAY:
        if (dev_info.isVicPresent)
            color = NVBUF_COLOR_FORMAT_NV12;
        else
            color = NVBUF_COLOR_FORMAT_GRAY8;

        m_ColorFormat = InferMediaFormat::kGRAY;
        channel = 1;
        isNCHW = true;
        break;
    default:
        InferError("Unsupported network input format: %d", (int)format);
        return false;
    }

    NvBufSurfaceCreateParams params = {0};
    params.gpuId = gpuId;
    params.width = width;
    params.height = height;
    params.size = 0;
    params.isContiguous = 1;
    params.colorFormat = color;
    params.layout = NVBUF_LAYOUT_PITCH;
    if (is_nvgpu)
        params.memType = NVBUF_MEM_SURFACE_ARRAY;
    else
        params.memType = NVBUF_MEM_CUDA_DEVICE;
    assert(m_ReservedSize > 0 && m_ReservedSize == getBatchSize());
    if (NvBufSurfaceCreate(&m_Surf, m_ReservedSize, &params) != 0) {
        InferError("Error: Could not allocate surface buffer");
        return false;
    }

    if (is_nvgpu) {
        if (NvBufSurfaceMapEglImage(m_Surf, -1) != 0) {
            InferError(
                    "Error: Could not map EglImage from NvBufSurface for nvinfer");
            return false;
        }
    }
    m_BufPtrs.resize(m_ReservedSize, nullptr);
    setBatchSize(m_ReservedSize);

    if (is_nvgpu) {
#ifdef IS_TEGRA
        m_EglFrames.resize(m_ReservedSize);
        m_CudaResources.resize(m_ReservedSize);

        for (unsigned int i = 0; i < m_ReservedSize; i++) {
            if (cuGraphicsEGLRegisterImage(&m_CudaResources[i],
                        m_Surf->surfaceList[i].mappedAddr.eglImage,
                        CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE) != CUDA_SUCCESS) {
                InferError("Failed to register EGLImage in cuda");
                return false;
            }

            if (cuGraphicsResourceGetMappedEglFrame(
                        &m_EglFrames[i], m_CudaResources[i], 0, 0) != CUDA_SUCCESS) {
                InferError("Failed to get mapped EGL Frame");
                return false;
            }
            m_BufPtrs[i] = (void*)m_EglFrames[i].frame.pPitch[0];
        }
#endif
    } else {
        for (uint32_t i = 0; i < m_ReservedSize; i++) {
            m_BufPtrs[i] = (void*)m_Surf->surfaceList[i].dataPtr;
        }
    }
    InferDims dims{
        3, {(int)params.height, (int)params.width, (int)channel}, 0};
    if (isNCHW) {
        dims = InferDims{
            3, {(int)channel, (int)params.height, (int)params.width}, 0};
    }
    normalizeDims(dims);

    InferMemType memType = is_nvgpu ? InferMemType::kNvSurface :
                               InferMemType::kNvSurfaceArray;

    InferBufferDescription bufDesc{
        memType : memType,
        devId : params.gpuId,
        dataType : InferDataType::kUint8,
        dims : dims,
        elementSize : getElementSize(InferDataType::kUint8),
        name : "",
        isInput : false,
    };
    setBufDesc(bufDesc);

    m_AlignInfo.widthAlign = m_Surf->surfaceList[0].planeParams.width[0];
    m_AlignInfo.heightAlign = m_Surf->surfaceList[0].planeParams.height[0];
    m_AlignInfo.channelAlign = channel;
    m_AlignInfo.pitchPerRow = m_Surf->surfaceList[0].planeParams.pitch[0];
    m_AlignInfo.dataSizePerBatch = m_Surf->surfaceList[0].dataSize;
    return true;
}

const BatchSurfaceInfo& SurfaceBuffer::getSurfaceAlignInfo() const {
    return m_AlignInfo;
}

UniqSurfaceBuf createNvBufSurface(
    int width, int height, InferMediaFormat format, int batchSize, int gpuId) {
    UniqSurfaceBuf surfaceBuf(new SurfaceBuffer(batchSize));
    if (!surfaceBuf->init(width, height, format, gpuId)) {
        InferError("Failed to add surface buffer into pool");
        return nullptr;
    }
    return surfaceBuf;
}

SharedBufPool<UniqSurfaceBuf> createSurfaceBufPool(int width, int height,
    InferMediaFormat color, int batchSize, int gpuID, int poolSize) {
    auto pool = std::make_shared<BufferPool<UniqSurfaceBuf>>("SurfaceBufPool");

    assert(pool);
    for (int i = 0; i < poolSize; ++i) {
        UniqSurfaceBuf buf =
            createNvBufSurface(width, height, color, batchSize, gpuID);
        if (!buf) {
            InferError("Failed to creat nvbufsurface");
            return nullptr;
        }
        assert(buf);
        if (!pool->setBuffer(std::move(buf))) {
            InferError("Failed to add surface buffer into pool");
            return nullptr;
        }
    }
    assert(pool->size() == poolSize);
    return pool;
}

} // namespace nvdsinferserver
