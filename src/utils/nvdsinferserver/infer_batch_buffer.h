/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file infer_batch_buffer.h
 *
 * @brief Header file of batch buffer related class declarations.
 */

#ifndef __NVDSINFERSERVER_BATCH_BUFFER_H__
#define __NVDSINFERSERVER_BATCH_BUFFER_H__

#include <algorithm>
#include <set>
#include <unordered_map>
#include "nvbufsurftransform.h"
#include "infer_common.h"
#include "infer_datatypes.h"

namespace nvdsinferserver {

/**
 * @brief The base class for batch buffers.
 */
class INFER_EXPORT_API BaseBatchBuffer : public IBatchBuffer {
protected:
    explicit BaseBatchBuffer(uint32_t batchSize) : m_BatchSize(batchSize) {}

public:
    ~BaseBatchBuffer() override = default;
    const InferBufferDescription& getBufDesc() const final { return m_Desc; }
    uint32_t getBatchSize() const final { return m_BatchSize; }
    uint64_t getTotalBytes() const override
    {
        assert(m_Desc.dataType != InferDataType::kNone);
        assert(m_Desc.elementSize >= 0);
        uint32_t b = getBatchSize();
        return m_Desc.elementSize * m_Desc.dims.numElements * (b ? b : 1);
    }
    void setBufDesc(const InferBufferDescription& desc) { m_Desc = desc; }
    InferBufferDescription& mutableBufDesc() { return m_Desc; }
    virtual void setBatchSize(uint32_t size) { m_BatchSize = size; }
    const SharedCuEvent& cuEvent() const { return m_CuEvent; }
    void setCuEvent(SharedCuEvent e) { m_CuEvent = std::move(e); }
    void setSyncObj(NvBufSurfTransformSyncObj_t SyncObj) { assert(m_SyncObj==nullptr); m_SyncObj = SyncObj; }
    NvBufSurfTransformSyncObj_t& getSyncObj() { return m_SyncObj; }
    void waitForSyncObj()
    {
        if (m_SyncObj)
        {
            NvBufSurfTransformSyncObjWait (m_SyncObj, -1);
            NvBufSurfTransformSyncObjDestroy (&m_SyncObj);
            m_SyncObj = nullptr;
        }
    }

    // attached buffers
    void attach(SharedBatchBuf buf)
    {
        m_Attaches.emplace_back(std::move(buf));
        assert(!hasAttachLoop());
    }
    void detach() { m_Attaches.clear(); }
    bool hasAttachedBufs() const { return !m_Attaches.empty();}
    const std::vector<SharedBatchBuf>& attachedBufs() const
    {
        return m_Attaches;
    }
    std::vector<SharedBatchBuf>& mutableAttachedBufs() { return m_Attaches; }
    bool hasAttachLoop() const
    {
        std::set<BaseBatchBuffer*> allAttached;
        for (auto const& buf : m_Attaches) {
            assert(buf);
            if (allAttached.count(buf.get())) {
                return true;
            }
            allAttached.emplace(buf.get());
        }
        return false;
    }

    void setBufId(uint64_t id) { m_BufId = id; }
    uint64_t bufId() const { return m_BufId; }

    /**
     * @brief Get the offset from start of the memory allocation to the
     *        buffer pointer. Needed for sharing CUDA memory with Triton
     *        server.
     */
    size_t getBufOffset(uint32_t batchIdx) const override
    {
        assert(getBufPtr(0) && getBufPtr(batchIdx));
        if (getBufPtr(0) == nullptr || getBufPtr(batchIdx) == nullptr) {
            return (size_t) -1;
        } else {
            return (uint8_t*)getBufPtr(batchIdx) - (uint8_t*)getBufPtr(0);
        }
    }

private:
    InferBufferDescription m_Desc{
        InferMemType::kNone, 0, InferDataType::kNone, {0}};
    uint32_t m_BatchSize = 0;
    SharedCuEvent m_CuEvent;
    std::vector<SharedBatchBuf> m_Attaches;
    uint64_t m_BufId = UINT64_C(0);  // for debug and track
    /** Sync object for allowing asynchronous call to nvbufsurftransform API
     * * Wait and Destroy to be done before network preprocess begins*/
    NvBufSurfTransformSyncObj_t m_SyncObj = nullptr;
};

/**
 * @brief The base class for array of batch buffers.
 */
class BaseBatchArray : public IBatchArray {
public:
    ~BaseBatchArray() override = default;
    uint32_t getSize() const final { return m_Bufs.size(); }
    const IBatchBuffer* getBuffer(uint32_t arrayIdx) const final
    {
        assert(arrayIdx < (uint32_t)m_Bufs.size());
        assert(m_Bufs.at(arrayIdx).get());
        return m_Bufs.at(arrayIdx).get();
    }
    SharedIBatchBuffer getSafeBuf(uint32_t arrayIdx) const final
    {
        assert(arrayIdx < (uint32_t)m_Bufs.size());
        assert(m_Bufs.at(arrayIdx).get());
        return buf(arrayIdx);
    }
    void appendIBatchBuf(SharedIBatchBuffer buf) final
    {
        SharedBatchBuf castBuf =
            std::dynamic_pointer_cast<BaseBatchBuffer>(buf);
        assert(castBuf);
        addBuf(castBuf);
    }
    const IOptions* getOptions() const final { return m_Options.get(); }
    SharedIOptions getSafeOptions() const { return m_Options; }
    void setIOptions(SharedIOptions o) final { setOptions(o); }

    const std::vector<SharedBatchBuf>& bufs() const { return m_Bufs; }
    std::vector<SharedBatchBuf>& mutableBufs() { return m_Bufs; }
    void addBuf(SharedBatchBuf buf) { assert(buf); m_Bufs.emplace_back(std::move(buf)); }
    const SharedBatchBuf& buf(uint32_t idx) const
    {
        assert(idx < (uint32_t)m_Bufs.size());
        return m_Bufs.at(idx);
    }
    SharedBatchBuf& buf(uint32_t idx)
    {
        assert(idx < (uint32_t)m_Bufs.size());
        return m_Bufs.at(idx);
    }
    const SharedCuEvent& cuEvent() const { return m_CuEvent; }
    void setCuEvent(SharedCuEvent e) { m_CuEvent = std::move(e); }
    int findFirstGpuId() const {
        int gpuId = -1;
        auto iter = std::find_if(m_Bufs.cbegin(), m_Bufs.cend(),
            [&gpuId](const SharedBatchBuf& buf) {
                assert(buf);
                const InferBufferDescription& desc = buf->getBufDesc();
                if (desc.memType == InferMemType::kGpuCuda) {
                    gpuId = desc.devId;
                    return true;
                }
                return false;
            });
        (void) iter;

        assert((iter != m_Bufs.cend() && gpuId >= 0) ||
               (iter == m_Bufs.cend() && gpuId == -1));
        return gpuId;
    }

    void setBufId(uint64_t id) { m_BufId = id; }
    uint64_t bufId() const { return m_BufId; }

    // set options
    void setOptions(SharedIOptions o) { m_Options = std::move(o); }

private:
    std::vector<SharedBatchBuf> m_Bufs;
    SharedCuEvent m_CuEvent;
    uint64_t m_BufId = UINT64_C(0);  // for debug and track

    SharedOptions m_Options;
};

extern void normalizeDims(InferDims& dims);

/**
 * @brief A batch buffer with allocated memory.
 */
class RefBatchBuffer : public BaseBatchBuffer {
public:
    RefBatchBuffer(
        void* bufBase, size_t offset, size_t bufBytes,
        const InferBufferDescription& desc, uint32_t batchSize)
        : BaseBatchBuffer(batchSize), m_BufBase(bufBase),
        m_BufBytes(bufBytes), m_BufOffset(offset)
    {
        setBufDesc(desc);
        normalizeDims(mutableBufDesc().dims);
    }
    void* getBufPtr(uint32_t batchIdx) const override {
        assert(batchIdx == 0 || batchIdx < getBatchSize());
        const InferBufferDescription& desc = getBufDesc();
        assert(batchIdx <= 0 || desc.dataType != InferDataType::kString);
        batchIdx = std::max(batchIdx, 0U);
        return (void*)((uint8_t*)m_BufBase +
                       desc.elementSize * desc.dims.numElements * batchIdx);
    }
    uint64_t getTotalBytes() const final { return m_BufBytes; }
    void* basePtr() { return m_BufBase; }

    /**
     * @brief Get the offset from start of the memory allocation to the
     *        buffer pointer. Needed for sharing CUDA memory with Triton
     *        server.
     */
    size_t getBufOffset(uint32_t batchIdx) const override {
        assert(batchIdx == 0 || batchIdx < getBatchSize());
        const InferBufferDescription& desc = getBufDesc();
        assert(batchIdx <= 0 || desc.dataType != InferDataType::kString);
        batchIdx = std::max(batchIdx, 0U);
        return m_BufOffset + desc.elementSize * desc.dims.numElements * batchIdx;
    }

private:
    mutable void* m_BufBase = nullptr;
    size_t m_BufBytes = 0;
    /**
     * @brief Offset of the buffer from the start of the memory allocation.
     */
    size_t m_BufOffset = 0;
};

class WrapCBatchBuffer : IBatchBuffer {
public:
    template <typename BufPtr>
    WrapCBatchBuffer(BufPtr buf) : m_Impl(std::move(buf)) {
        assert(m_Impl);
    }

    ~WrapCBatchBuffer() final = default;
    const InferBufferDescription& getBufDesc() const final {
        return m_Impl->getBufDesc();
    }
    void* getBufPtr(uint32_t batchIdx) const final {
        return m_Impl->getBufPtr(batchIdx);
    }
    uint32_t getBatchSize() const final { return m_Impl->getBatchSize(); }
    uint64_t getTotalBytes() const final { return m_Impl->getTotalBytes(); }

private:
    SharedBatchBuf m_Impl;
};

} // namespace nvdsinferserver

#endif
