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

#ifndef __NVDS_SEQ_PREPROCESS_COMMON_H__
#define __NVDS_SEQ_PREPROCESS_COMMON_H__

#include <inttypes.h>
#include <stdint.h>
#include <string.h>

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <unordered_map>

#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include "nvdspreprocess_interface.h"

// custom settings in [user-config]
#define CUSTOM_CONFIG_SUBSAMPLE "subsample"
#define CUSTOM_CONFIG_STRIDE "stride"
#define CUSTOM_CONFIG_CHANNEL_SCALE_FACTORS "channel-scale-factors"
#define CUSTOM_CONFIG_CHANNEL_MEANS "channel-mean-offsets"

// Misc macro definitions
#define DSASSERT assert

#ifndef UNUSED
#define UNUSED(var) (void)(var)
#endif

// log information definition for log print, debug, info, error.

#define LOG_PRINT_(out, level, fmt, ...) \
    fprintf(out, "%s:%d, [%s: CUSTOM_LIB] " fmt "\n", __FILE__, __LINE__, #level, ##__VA_ARGS__)

#define LOG_DEBUG(fmt, ...)                            \
    if (kEnableDebug) {                                \
        LOG_PRINT_(stdout, DEBUG, fmt, ##__VA_ARGS__); \
    }

#define LOG_INFO(fmt, ...) LOG_PRINT_(stdout, INFO, fmt, ##__VA_ARGS__)

#define LOG_ERROR(fmt, ...) LOG_PRINT_(stderr, ERROR, fmt, ##__VA_ARGS__)

// check preprocess errors and return error no.
#define CHECK_PROCESS_ERROR(err, fmt, ...)                                       \
    do {                                                                         \
        NvDsPreProcessStatus _sno_ = (err);                                      \
        if (_sno_ != NVDSPREPROCESS_SUCCESS) {                                   \
            LOG_ERROR(fmt ", seq_process error: %d", ##__VA_ARGS__, (int)_sno_); \
            return _sno_;                                                        \
        }                                                                        \
    } while (0)

// check CUDA errors errors and return process error
#define CHECK_CUDA_ERR(err, fmt, ...)                                               \
    do {                                                                            \
        cudaError_t _errnum_ = (err);                                               \
        if (_errnum_ != cudaSuccess) {                                              \
            LOG_ERROR(                                                              \
                fmt ", cuda err_no: %d, err_str: %s", ##__VA_ARGS__, (int)_errnum_, \
                cudaGetErrorName(_errnum_));                                        \
            return NVDSPREPROCESS_CUDA_ERROR;                                       \
        }                                                                           \
    } while (0)

// declare export APIs
#define PROCESS_EXPORT_API __attribute__((__visibility__("default")))

// skip frame: extend values of NvDsPreProcessStatus
#define NVDSPREPROCESS_SKIP_FRAME (NvDsPreProcessStatus)255

// default frame number value if not set
#define FRAME_NO_NOT_SET UINT64_MAX

// tuple<stream_id, roi.xy0, roi.xy1>
using SourceKey = std::tuple<uint64_t, int64_t, int64_t>;

// enable debug log
extern bool kEnableDebug;

// Block descriptions to store multiple sequenced buffers
struct FullBatchBlock {
    NvDsPreProcessCustomBuf* buf = nullptr;  // allocated from NvDsPreProcessAcquirer
    // parameters description of allocated `buf`
    NvDsPreProcessTensorParams param = {
        NvDsPreProcessNetworkInputOrder_CUSTOM, {}, NvDsPreProcessFormat_RGB};
    uint32_t maxBatchSize = 0;  // parsed from param.network_input_shape[0]
    uint32_t inUseBatchSize = 0;
};

// ROI based sequence processing buffer
struct RoiProcessedBuf {
    // pointer to batch buffer block
    FullBatchBlock* block = nullptr;
    // offsets in bytes for current sequence in batched block
    uint64_t blockOffset = 0;

    // starting sequence index of this buffer.
    // index loops between [0, seqence-size]
    uint32_t startSeqIdx = 0;
    // accumulation count of processed buffers.
    // accumulation is sliding by stride in user-config
    uint32_t processedSeq = 0;

    // ROI information of this sequence
    NvDsRoiMeta roiInfo;
    // latest processed buffer information
    NvDsPreProcessUnit latestUnit;
    // lastest updated frame number
    uint64_t latestFrameNo = FRAME_NO_NOT_SET;
    // frame number to indicate next sequence is ready
    uint64_t nextReadyFrameNo = 0;
};

// results storing batches of sequence processed buffers
struct ReadyResult {
    // allocated batches of processed sequence buffers
    NvDsPreProcessCustomBuf* readyBuf = nullptr;
    // ROI list of processed buffers
    std::vector<NvDsRoiMeta> rois;
    // updated parameter desciptions of `readyBuf`
    NvDsPreProcessTensorParams param = {
        NvDsPreProcessNetworkInputOrder_CUSTOM, {}, NvDsPreProcessFormat_RGB};
};

// mapping source/roi to specific sequence processing buffer
using SrcDestBufMap = std::map<SourceKey, std::unique_ptr<RoiProcessedBuf>>;

// definition of smart pointer
template <class T>
using UniqPtr = std::unique_ptr<T, std::function<void(T*)>>;

#endif  //  __NVDS_SEQ_PREPROCESS_COMMON_H__