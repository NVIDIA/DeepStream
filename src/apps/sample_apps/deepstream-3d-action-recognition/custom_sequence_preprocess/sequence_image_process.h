/*
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef __NVDS_SEQUENCE_IMAGE_PREPROCESS_H__
#define __NVDS_SEQUENCE_IMAGE_PREPROCESS_H__

#include "seq_process_common.h"
#include "sequence_preprocess_kernel.h"

// default values
constexpr static uint32_t kDefaultChannel = 3;
constexpr static uint32_t kDefaultStride = 1;
constexpr static uint32_t kDefaultSubSample = 0;

// Buffer manager convert and normalize all the source ROI input images into
// network model input datatype and format. It enables temporal batching per
// each ROI into sequence, and gather multiple ROI sequence into spacial batches.
// It supports `subsample` and `stride` to improve performance for model inference.
class BufferManager {
public:
    // constructor and destructor
    BufferManager(
        NvDsPreProcessAcquirer* allocator, const NvDsPreProcessTensorParams& params, uint32_t depth,
        uint32_t channel, cudaStream_t stream, uint32_t stride, uint32_t interval)
        : _allocator(allocator), _tensorParams(params), _seqStride(stride), _subsample(interval)
    {
        const auto& shape = params.network_input_shape;
        DSASSERT(shape.size() == 5 || shape.size() == 4);  // NCDHW or NCHW
        if (shape.size() == 5) {
            DSASSERT(shape[2] == (int)depth);  // NCDHW
        }
        _maxBatch = shape[0];
        _seqSize = depth;
        _channels = channel;
        _perBatchSize =
            std::accumulate(shape.begin() + 1, shape.end(), 1, [](int s, int i) { return s * i; });
        DSASSERT(_perBatchSize > 0);
        _cuStream = stream;
    }
    ~BufferManager() { clearAll(); }

    // allocator could be set later
    NvDsPreProcessAcquirer* allocator() const { return _allocator; }
    void setAllocator(NvDsPreProcessAcquirer* allocator) { _allocator = allocator; }

    // creat new sequence buffers on each new ROI information
    NvDsPreProcessStatus buildRoiBlocks(const std::vector<NvDsPreProcessUnit>& rois);
    // locate the output CUDA memory pointer for each coming ROI
    NvDsPreProcessStatus locateRoiDst(const NvDsPreProcessUnit& roi, void*& dstPatch);
    // collect all ready results of ROI sequence buffers
    NvDsPreProcessStatus collectReady();
    // pop a ready result
    bool popReady(ReadyResult& res);
    // clear before quit
    void clearAll();

private:
    NvDsPreProcessStatus addRoi(const SourceKey& key, const NvDsPreProcessUnit& unit);
    uint32_t perBatchBytes() const { return _perBatchSize * sizeof(float); }

    // HW: height * width
    uint32_t HWbytes() const { return perBatchBytes() / _seqSize / _channels; }

    uint32_t SHWbytes() const { return perBatchBytes() / _channels; }

    bool popOldestReady(ReadyResult& res)
    {
        if (_readyPendings.empty()) {
            return false;
        }
        res = _readyPendings.front();
        _readyPendings.pop();
        return true;
    }

private:
    NvDsPreProcessAcquirer* _allocator = nullptr;
    NvDsPreProcessTensorParams _tensorParams;
    cudaStream_t _cuStream = nullptr;
    std::vector<std::unique_ptr<FullBatchBlock>> _accumulateBlocks;
    SrcDestBufMap _src2dstMap;
    std::queue<ReadyResult> _readyPendings;
    uint32_t _maxBatch = 0;
    uint32_t _seqSize = 0;
    uint32_t _channels = kDefaultChannel;
    uint32_t _perBatchSize = 0;
    uint32_t _seqStride = kDefaultStride;
    uint32_t _subsample = kDefaultSubSample;
};

// Preprocess Context for all streams and ROIs. Custom lib symbols is cast into
// this context. It connects the buffer manager and input/output cuda processing
// kernels. Return the final processed buffer into Gst-nvdspreprocess plugin.
class SequenceImagePreprocess {
public:
    SequenceImagePreprocess(const CustomInitParams& initParams) { _initParams = initParams; }
    ~SequenceImagePreprocess() { deinit(); }

    // derives symbol of initLib
    NvDsPreProcessStatus init();
    // derives symbol of deInitLib
    NvDsPreProcessStatus deinit();
    // derives symbol of CustomSequenceTensorPreparation
    NvDsPreProcessStatus prepareTensorData(
        NvDsPreProcessBatch* batch, NvDsPreProcessCustomBuf*& buf, CustomTensorParams& tensorParam,
        NvDsPreProcessAcquirer* allocator);

    // internal cuda data conversion
    NvDsPreProcessStatus preprocessData(
        const NvDsPreProcessUnit& unit, NvDsPreProcessFormat inFormat, void* outPtr);

private:
    NvDsPreProcessStatus setDevice();
    NvDsPreProcessStatus parseUserConfig();

    CustomInitParams _initParams;
    std::unique_ptr<BufferManager> _bufManager;
    cudaStream_t _cuStream = nullptr;
    int _gpuId = -1;
    uint32_t _N = 0;
    uint32_t _C = kDefaultChannel;
    uint32_t _S = 0;
    uint32_t _H = 0;
    uint32_t _W = 0;
    Float4Vec _scales = {{1.0, 1.0, 1.0, 1.0}};
    Float4Vec _means = {{0.0, 0.0, 0.0, 0.0}};
    uint32_t _subsample = 0;
    uint32_t _stride = kDefaultStride;
};

// entrypoint symbols for Gst-nvpreprocess plugin to load
// custom lib: libnvds_custom_sequence_preprocess.so
extern "C" {
// entrypoint for each batched ROI buffers processing
PROCESS_EXPORT_API NvDsPreProcessStatus CustomSequenceTensorPreparation(
    CustomCtx* ctx, NvDsPreProcessBatch* batch, NvDsPreProcessCustomBuf*& buf,
    CustomTensorParams& tensorParam, NvDsPreProcessAcquirer* allocator);

// entrypoint to create and init custom lib context
PROCESS_EXPORT_API CustomCtx* initLib(CustomInitParams initparams);

// entrypoint to destroy custom lib context
PROCESS_EXPORT_API void deInitLib(CustomCtx* ctx);
}

#endif  // __NVDS_SEQUENCE_IMAGE_PREPROCESS_H__