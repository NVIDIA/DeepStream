/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef __NVDSINFERSERVER_SURFACE_BUFS_H__
#define __NVDSINFERSERVER_SURFACE_BUFS_H__

#include <stdarg.h>
#include <condition_variable>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <queue>

#ifdef IS_TEGRA
#include "cudaEGL.h"
#endif

#include <nvbufsurface.h>
#include <nvbufsurftransform.h>

#include "infer_batch_buffer.h"
#include "infer_cuda_utils.h"
#include "infer_datatypes.h"
#include "infer_preprocess_kernel.h"
#include "infer_utils.h"

namespace nvdsinferserver {

class SurfaceBuffer;
using UniqSurfaceBuf = std::unique_ptr<SurfaceBuffer>;

class ImageAlignBuffer : public BaseBatchBuffer {
public:
    ImageAlignBuffer(int batchSize) : BaseBatchBuffer(batchSize) {}
    virtual InferMediaFormat getColorFormat() const = 0;
    virtual const BatchSurfaceInfo& getSurfaceAlignInfo() const = 0;
};

class SurfaceBuffer : public ImageAlignBuffer {
    friend UniqSurfaceBuf createNvBufSurface(int width, int height,
        InferMediaFormat format, int batchSize, int gpuId);

protected:
    explicit SurfaceBuffer(int batchSize);
    bool init(int width, int height, InferMediaFormat format, int gpuId);

public:
    ~SurfaceBuffer() override;
    void* getBufPtr(uint32_t batchIdx) const override {
        assert(batchIdx < getBatchSize());
        return m_BufPtrs[batchIdx];
    }

    /**
     * @brief Get the offset from start of the memory allocation to the
     *        buffer pointer. Needed for sharing CUDA memory with Triton
     *        server.
     */
    size_t getBufOffset(uint32_t batchIdx) const override {
        assert(batchIdx < getBatchSize());
        if (batchIdx >= getBatchSize())
            return (size_t) -1;
        if (m_Surf->isContiguous) {
            return (char*)m_BufPtrs[batchIdx] - (char*)m_BufPtrs[0];
        } else {
            return 0;
        }
    }

    uint32_t getReservedSize() { return m_ReservedSize; }
    NvBufSurface* getBufSurface() { return m_Surf; }
    InferMediaFormat getColorFormat() const override { return m_ColorFormat; }

    // only for dGPU
    const BatchSurfaceInfo& getSurfaceAlignInfo() const override;

    // reuse for bufPool
    void reuse() { setBatchSize(m_ReservedSize); }

private:
    uint32_t m_ReservedSize{0};
    NvBufSurface* m_Surf{nullptr};
    std::vector<void*> m_BufPtrs;

    // only for dGPU
    BatchSurfaceInfo m_AlignInfo{0, 0, 0};
    InferMediaFormat m_ColorFormat{InferMediaFormat::kRGB};

#ifdef IS_TEGRA
    std::vector<CUgraphicsResource> m_CudaResources;
    std::vector<CUeglFrame> m_EglFrames;
#endif
};

SharedBufPool<UniqSurfaceBuf> createSurfaceBufPool(int width, int height,
    InferMediaFormat color, int batchSize, int gpuId, int poolSize);

class INFER_EXPORT_API BatchSurfaceBuffer : public BaseBatchBuffer {
public:
    BatchSurfaceBuffer(int devId, uint32_t maxBatchSize,
        NvBufSurfaceMemType memType) : BaseBatchBuffer(0) {
        mutableBufDesc().devId = devId;
        mutableBufDesc().memType = InferMemType::kNvSurface;
        mutableBufDesc().dataType = InferDataType::kUint8;
        mutableBufDesc().elementSize = 1;
        m_BufSurface.gpuId = devId;
        m_BufSurface.batchSize = maxBatchSize;
        m_BufSurface.memType = memType;
    }
    void* getBufPtr(uint32_t batchIdx) const override {
        assert(batchIdx < m_BufSurface.numFilled);
        if (batchIdx >= m_BufSurface.numFilled)
            return nullptr;
        return m_BufSurface.surfaceList[batchIdx].dataPtr;
    }

    /**
     * @brief Get the offset from start of the memory allocation to the
     *        buffer pointer. Needed for sharing CUDA memory with Triton
     *        server.
     */
    size_t getBufOffset(uint32_t batchIdx) const override {
        assert(batchIdx < m_BufSurface.numFilled);
        if (batchIdx >= m_BufSurface.numFilled)
            return (size_t) -1;
        if (m_BufSurface.isContiguous) {
            return (char *)m_BufSurface.surfaceList[batchIdx].dataPtr -
                (char *)m_BufSurface.surfaceList[0].dataPtr;
        } else {
            return 0;
        }
    }

    void append(
        const NvBufSurfaceParams& params, const NvBufSurfTransformRect& crop) {
        m_Params.push_back(params);
        m_CropRects.push_back(crop);
        m_ScaleRatios.emplace_back(0, 0);
        m_Offsets.emplace_back(0, 0);
        m_BufSurface.surfaceList = &m_Params[0];
        m_BufSurface.numFilled = (uint32_t)m_Params.size();
        assert(m_BufSurface.numFilled <= m_BufSurface.batchSize);
        setBatchSize(m_BufSurface.numFilled);
    }
    NvBufSurfaceParams& getSurfaceParams(int batchIdx) {
        assert(batchIdx < (int)m_Params.size());
        return m_Params.at(batchIdx);
    }
    NvBufSurfTransformRect getCropArea(int batchIdx) {
        assert(batchIdx < (int)m_CropRects.size());
        return m_CropRects.at(batchIdx);
    }
    NvBufSurface* getBufSurface() { return &m_BufSurface; }

    void getScaleRatio(uint32_t batchIdx, double& ratioX, double& ratioY) {
        assert(batchIdx < (uint32_t)m_ScaleRatios.size());
        std::tie(ratioX, ratioY) = m_ScaleRatios.at(batchIdx);
    }

    void setScaleRatio(uint32_t batchIdx, double ratioX, double ratioY) {
        assert(batchIdx < (uint32_t)m_ScaleRatios.size());
        m_ScaleRatios.at(batchIdx) = std::tie(ratioX, ratioY);
    }

    void getOffsets(uint32_t batchIdx, uint32_t& offsetLeft,
        uint32_t& offsetTop) {
        assert(batchIdx < (uint32_t)m_Offsets.size());
        std::tie(offsetLeft, offsetTop) = m_Offsets.at(batchIdx);
    }

    void setOffsets(uint32_t batchIdx, uint32_t offsetLeft, uint32_t offsetTop) {
        assert(batchIdx < (uint32_t)m_Offsets.size());
        m_Offsets.at(batchIdx) = std::tie(offsetLeft, offsetTop);
    }

private:
    NvBufSurface m_BufSurface{0, 0, 0, 0};
    std::vector<NvBufSurfaceParams> m_Params;
    std::vector<NvBufSurfTransformRect> m_CropRects;
    std::vector<std::tuple<double, double>> m_ScaleRatios;
    std::vector<std::tuple<uint32_t, uint32_t>> m_Offsets;
};

using SharedBatchSurface = std::shared_ptr<BatchSurfaceBuffer>;

} // namespace nvdsinferserver

#endif
