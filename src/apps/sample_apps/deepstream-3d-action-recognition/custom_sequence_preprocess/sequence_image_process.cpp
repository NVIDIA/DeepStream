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

#include "sequence_image_process.h"

#include <algorithm>
#include <locale>

#include "seq_process_common.h"
#include "sequence_preprocess_kernel.h"

// enable debug log print from os environment variables
bool kEnableDebug = std::getenv("DS_CUSTOM_SEQUENC_DEBUG") ? true : false;

/* kEnableDumpROI only works when inter-pool-memory-type=3(NVBUF_MEM_CUDA_UNIFIED) */
bool kEnableDumpROI = std::getenv("DS_ENABLE_DUMP_ROI") ? true : false;

namespace {
// generate key for SrcDestBufMap
SourceKey
getKey(uint64_t src_id, float x, float y, float width, float height)
{
    constexpr int64_t kMaxX = 10000;
    constexpr int64_t kMaxY = kMaxX * kMaxX;
    int64_t xy0 = int64_t(x) * kMaxX + int64_t(y) * kMaxY;
    int64_t xy1 = int64_t(x + width) * kMaxX + int64_t(y * height) * kMaxY;
    return std::make_tuple(src_id, xy0, xy1);
}

uint32_t
getChannels(NvDsPreProcessFormat format)
{
    uint32_t channels = 0;
    switch (format) {
    case NvDsPreProcessFormat_RGB:
    case NvDsPreProcessFormat_BGR:
        channels = 3;
        break;
    case NvDsPreProcessFormat_RGBA:
    case NvDsPreProcessFormat_BGRx:
        channels = 4;
        break;
    case NvDsPreProcessFormat_GRAY:
        channels = 1;
    default:
        LOG_ERROR("channel format: %d is not supported.", (int)format);
        break;
    }
    return channels;
}

std::string
trimStr(std::string s)
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char c) {
                return !std::isspace(c);
            }));
    s.erase(
        std::find_if(s.rbegin(), s.rend(), [](unsigned char c) { return !std::isspace(c); }).base(),
        s.end());
    return s;
}

// parse string into float vector, e.g. "1.0;0.5;0.3"
std::vector<float>
parseNumList(const std::string& str)
{
    std::string delim{';'};
    size_t pos = 0, oldpos = 0;
    std::vector<float> ret;

    while ((pos = str.find(delim, oldpos)) != std::string::npos) {
        auto word = trimStr(str.substr(oldpos, pos - oldpos));
        if (word.empty()) {
            LOG_ERROR("parse num list failed");
            return {};
        }
        ret.push_back(std::stof(word));
        oldpos = pos + delim.length();
    }
    auto last = trimStr(str.substr(oldpos));
    if (!last.empty()) {
        ret.push_back(std::stof(last));
    }
    return ret;
}

// dump roi image if kEnableDumpROI enabled. converted_frame_ptr must be host readable
void
dumpRois(NvDsPreProcessBatch& batch, uint32_t h, uint32_t w, uint32_t c)
{
    for (guint i = 0; i < batch.units.size(); i++) {
        auto& unit = batch.units[i];
        uint32_t srcid = unit.frame_meta->source_id;
        uint32_t frameId = unit.frame_num;

        std::string fileName = std::string("ROI-src_") + std::to_string(srcid) + "-fid_" +
                               std::to_string(frameId) + "-roi_" + std::to_string(frameId) +
                               "hwc_" + std::to_string(h) + "x" + std::to_string(w) + "x" +
                               std::to_string(c) + ".bin";
        std::ofstream outfile(fileName);
        uint32_t pitch = batch.pitch;
        for (unsigned int j = 0; j < h; j++) {
            outfile.write((char*)unit.converted_frame_ptr + j * pitch, c * w);
        }
        outfile.close();
    }
}
}  // namespace

// resource clear up before buffer maanger get destroyed
void
BufferManager::clearAll()
{
    LOG_DEBUG(
        "Clearing ready pedning batches. size: %u",
        (uint32_t)_readyPendings.size());
    while (!_readyPendings.empty()) {
        ReadyResult res;
        popOldestReady(res);
        DSASSERT(res.readyBuf);
        DSASSERT(_allocator);
        _allocator->release(res.readyBuf);
    }

    LOG_DEBUG(
        "Clearing accumulated blocks. size: %u",
        (uint32_t)_accumulateBlocks.size());
    for (auto& block : _accumulateBlocks) {
        DSASSERT(block->buf);
        _allocator->release(block->buf);
        block.reset();
    }
    _accumulateBlocks.clear();
}

// add new ROI sequence buffers if not exist. Each block buffer could
// store at most `max-batch-size` ROIs. If ROI number is larger than that,
// it will allocate new block.
NvDsPreProcessStatus
BufferManager::addRoi(const SourceKey& key, const NvDsPreProcessUnit& unit)
{
    DSASSERT(!_src2dstMap.count(key));
    auto last = _accumulateBlocks.rbegin();
    if (_accumulateBlocks.empty() ||
        (*last)->inUseBatchSize >= (*last)->maxBatchSize) {
        auto newBlock = std::make_unique<FullBatchBlock>();
        DSASSERT(newBlock);
        newBlock->buf = _allocator->acquire();
        newBlock->param = _tensorParams;
        newBlock->maxBatchSize = _maxBatch;
        newBlock->inUseBatchSize = 0;
        _accumulateBlocks.emplace_back(std::move(newBlock));
        LOG_DEBUG(
            "Create New accumulating ROI block, batch size: %u, total block "
            "size: %u",
            _maxBatch, (uint32_t)_accumulateBlocks.size());
    }
    FullBatchBlock* block = (*_accumulateBlocks.rbegin()).get();
    auto newSrc = std::make_unique<RoiProcessedBuf>();
    DSASSERT(newSrc);
    newSrc->block = block;
    newSrc->roiInfo = unit.roi_meta;
    newSrc->blockOffset = perBatchBytes() * block->inUseBatchSize;
    newSrc->latestFrameNo = FRAME_NO_NOT_SET;
    newSrc->nextReadyFrameNo = unit.frame_num + (_seqSize - 1) * (_subsample + 1);
    _src2dstMap.emplace(key, std::move(newSrc));
    block->inUseBatchSize++;

    uint64_t srcId = std::get<0>(key);
    UNUSED(srcId);
    LOG_DEBUG(
        "Adding new ROI of source: %u mapped to dst blockId: %u, offset:%u, "
        "total ROI numbers: %d",
        (uint32_t)srcId, (uint32_t)_accumulateBlocks.size() - 1,
        block->inUseBatchSize - 1, (uint32_t)_src2dstMap.size());
    return NVDSPREPROCESS_SUCCESS;
}

// check and build ROI buffers on blocks
NvDsPreProcessStatus
BufferManager::buildRoiBlocks(const std::vector<NvDsPreProcessUnit>& rois)
{
    for (const auto& unit : rois) {
        uint64_t source_id = unit.frame_meta->source_id;
        const auto& roi = unit.roi_meta.roi;
        SourceKey key =
            getKey(source_id, roi.left, roi.top, roi.width, roi.height);
        // if allocate area is found, then run again.
        if (_src2dstMap.find(key) != _src2dstMap.end()) {
            continue;
        }
        CHECK_PROCESS_ERROR(
            addRoi(key, unit), "init roi block memory failed, src_id:%d",
            (int32_t)source_id);
    }
    return NVDSPREPROCESS_SUCCESS;
}

// locate the output CUDA memory pointer for each coming ROI.
// Before this call, all of the ROI information buffers should be created
// by `buildRoiBlocks`
NvDsPreProcessStatus
BufferManager::locateRoiDst(const NvDsPreProcessUnit& unit, void*& dstPatch)
{
    uint64_t source_id = unit.frame_meta->source_id;
    const auto& roi = unit.roi_meta.roi;
    SourceKey key = getKey(source_id, roi.left, roi.top, roi.width, roi.height);
    auto iter = _src2dstMap.find(key);
    if (iter == _src2dstMap.end()) {
        LOG_ERROR(
            "Could not located the new ROI source, src_id:%u",
            (uint32_t)source_id);
        return NVDSPREPROCESS_INVALID_PARAMS;
    }
    RoiProcessedBuf* roiDstBuf = iter->second.get();

    // If frame_num not in subsample rate, skip the frame
    // If frame_num far from next sequence range, skip the frame
    if (roiDstBuf->latestFrameNo != FRAME_NO_NOT_SET &&
        (unit.frame_num <= roiDstBuf->latestFrameNo + _subsample ||
         unit.frame_num + (_seqSize - 1) * (_subsample + 1) < roiDstBuf->nextReadyFrameNo)) {
        LOG_DEBUG("jump frame: %" PRIu64 " on subsample", (uint64_t)(unit.frame_num));
        return NVDSPREPROCESS_SKIP_FRAME;
    }
    uint8_t* basePtr =
        (uint8_t*)roiDstBuf->block->buf->memory_ptr + roiDstBuf->blockOffset;
    uint32_t curIdx = (roiDstBuf->startSeqIdx + roiDstBuf->processedSeq) % _seqSize;

    // current dstPatch memory pointer is for NCDHW(NCSHW) order type. for
    // other order type, User need to update accordingly.
    // e.g. for NSCHW, dstPatch = (void*)(basePtr + curIdx * CHWbytes())
    dstPatch = (void*)(basePtr + curIdx * HWbytes());
    roiDstBuf->processedSeq++;
    roiDstBuf->latestFrameNo = unit.frame_num;
    roiDstBuf->latestUnit = unit;
    roiDstBuf->roiInfo = unit.roi_meta;

    // Reject oldest processed buffers when it is larger than sequence size
    while (roiDstBuf->processedSeq > _seqSize) {
        roiDstBuf->startSeqIdx = (roiDstBuf->startSeqIdx + 1) % (_seqSize);
        roiDstBuf->processedSeq--;
        LOG_DEBUG(
            "reject 1 oldest processed bufer on receiving frame: %" PRIu64,
            unit.frame_num);
    }

    return NVDSPREPROCESS_SUCCESS;
}

// collect all ready results of ROI sequence buffers
NvDsPreProcessStatus
BufferManager::collectReady()
{
    UniqPtr<NvDsPreProcessCustomBuf> buf(
        nullptr,
        [this](NvDsPreProcessCustomBuf* b) { _allocator->release(b); });
    uint32_t filled = 0;
    uint64_t latestFrameNo = 0;
    ReadyResult res;
    for (auto& ipair : _src2dstMap) {
        RoiProcessedBuf* roiBuf = ipair.second.get();
        uint64_t srcId = std::get<0>(ipair.first);
        if (roiBuf->processedSeq < _seqSize || roiBuf->latestFrameNo < roiBuf->nextReadyFrameNo) {
            continue;
        }
        latestFrameNo = roiBuf->latestFrameNo;

        if (!buf) {
            buf.reset(_allocator->acquire());
            filled = 0;
            res.readyBuf = buf.get();
            res.rois.clear();
            res.param = _tensorParams;
        }
        UNUSED(srcId);
        LOG_DEBUG(
            "ROI sequence in src: %u, with last frameNo: %" PRIu64 " collected into batch",
            (uint32_t)srcId, latestFrameNo);
        DSASSERT(buf);
        uint8_t* dstPtr =
            (uint8_t*)(buf->memory_ptr) + filled * perBatchBytes();
        uint8_t* srcBasePtr =
            (uint8_t*)roiBuf->block->buf->memory_ptr + roiBuf->blockOffset;

        // Copy sequence ready rois/frames into batch buffer
        // this is for NCSHW(NCDHW) order only, user need replace this segment
        // copy block accordingly if other orders needed.
        uint32_t segment = _seqSize - roiBuf->startSeqIdx;
        DSASSERT(segment > 0);
        for (uint32_t i = 0; i < _channels; ++i) {
            uint32_t offsetC = i * SHWbytes();
            CHECK_CUDA_ERR(
                cudaMemcpyAsync(
                    (void*)(dstPtr + offsetC),
                    srcBasePtr + offsetC + roiBuf->startSeqIdx * HWbytes(), segment * HWbytes(),
                    cudaMemcpyDeviceToDevice, _cuStream),
                "Failed to copy ready sequence to batched buffer");
            if (segment < _seqSize) {
                CHECK_CUDA_ERR(
                    cudaMemcpyAsync(
                        (void*)(dstPtr + offsetC + segment * HWbytes()), srcBasePtr + offsetC,
                        (_seqSize - segment) * HWbytes(), cudaMemcpyDeviceToDevice, _cuStream),
                    "Failed to copy ready sequence to batched buffer");
            }
        }
        ++filled;
        res.rois.push_back(roiBuf->roiInfo);

        // update next ready frame number
        roiBuf->nextReadyFrameNo = roiBuf->latestFrameNo + _seqStride * (_subsample + 1);
        LOG_DEBUG(
            "update src: %d nextReadyFrameNo: %" PRIu64, (int)srcId, roiBuf->nextReadyFrameNo);
        // move buffers ahead.
        DSASSERT(_seqStride);
        uint32_t moveBufs = std::min<uint32_t>(_seqStride, roiBuf->processedSeq);
        roiBuf->startSeqIdx = (roiBuf->startSeqIdx + moveBufs) % _seqSize;
        roiBuf->processedSeq -= moveBufs;

        if (filled >= _maxBatch) {
            _readyPendings.push(res);
            buf.release();
            filled = 0;
            res.readyBuf = nullptr;
            LOG_DEBUG(
                "A full batched sequence tensor is ready on last frame: "
                "%" PRIu64,
                latestFrameNo);
        }
    }
    if (buf && filled) {
        DSASSERT(res.readyBuf);
        DSASSERT(filled <= _maxBatch);
        // update batch-size if not fully  filled
        res.param.network_input_shape[0] = filled;
        _readyPendings.push(res);
        buf.release();
        LOG_DEBUG(
            "A partial batched sequence tensor(filled: %u) is ready on last frame: "
            "%" PRIu64,
            filled, latestFrameNo);
    }
    return NVDSPREPROCESS_SUCCESS;
}

bool
BufferManager::popReady(ReadyResult& res)
{
    if (!_readyPendings.size()) {
        return false;
    }

    while (_readyPendings.size() > 1) {
        ReadyResult old;
        popOldestReady(old);
        DSASSERT(old.readyBuf);
        _allocator->release(old.readyBuf);
        LOG_ERROR(
            "INTERNAL ERROR: Multiple ready batches available, custom API need "
            "to improve.");
    }
    if (!popOldestReady(res)) {
        LOG_ERROR("Lasest ready batch pending but not able to pop.");
        return false;
    }
    return true;
}

// Convert and normalize each input cropped ROI image into NCSHW(NCDHW) format.
NvDsPreProcessStatus
SequenceImagePreprocess::preprocessData(
    const NvDsPreProcessUnit& unit, NvDsPreProcessFormat inFormat, void* outPtr)
{
    DSASSERT(unit.converted_frame_ptr);
    DSASSERT(unit.roi_meta.converted_buffer);
    bool swapRB = false;
    void* srcPtr = unit.converted_frame_ptr;
    if (inFormat == _initParams.tensor_params.network_color_format) {
        swapRB = false;
    } else {
        switch (_initParams.tensor_params.network_color_format) {
        case NvDsPreProcessFormat_RGB:
        case NvDsPreProcessFormat_RGBA:
            if (inFormat == NvDsPreProcessFormat_BGR || inFormat == NvDsPreProcessFormat_BGRx) {
                swapRB = true;
            }
            break;
        case NvDsPreProcessFormat_BGR:
        case NvDsPreProcessFormat_BGRx:
            if (inFormat == NvDsPreProcessFormat_RGB || inFormat == NvDsPreProcessFormat_RGBA) {
                swapRB = true;
            }
            break;
        default:
            LOG_ERROR(
                "network format: %d is not supported.",
                (int)_initParams.tensor_params.network_color_format);
            return NVDSPREPROCESS_CUSTOM_LIB_FAILED;
        }
    }

    uint32_t inChannel = getChannels(inFormat);
    if (inChannel <= 0 || inChannel > 4) {
        LOG_ERROR("input format is not supported");
        return NVDSPREPROCESS_CUSTOM_LIB_FAILED;
    }
    // User can replace and implement different cuda kernels for other order types.
    CHECK_CUDA_ERR(
        preprocessNCDHW(
            outPtr, _C, _H, _W, _S, srcPtr, inChannel, unit.roi_meta.converted_buffer->pitch,
            _scales, _means, swapRB, _cuStream),
        "preprocessNCDHW failed.");
    return NVDSPREPROCESS_SUCCESS;
}

NvDsPreProcessStatus
SequenceImagePreprocess::setDevice()
{
    if (_gpuId != -1) {
        CHECK_CUDA_ERR(cudaSetDevice(_gpuId), "failed to set dev-id:%d", _gpuId);
    }
    return NVDSPREPROCESS_SUCCESS;
}

// Implementation of cast context initLib.
// 3D models input_shape(NCSHW) has 5 dims and 2D models(NSHW) has 4 dims. 2D input
// shape is reshaped from 3D's NCSHW order shape.
// Parse channel-scale-factors, mean-offsets, subsample, stride from [user-config].
NvDsPreProcessStatus
SequenceImagePreprocess::init()
{
    bool is3D = true;
    auto const& input_shape = _initParams.tensor_params.network_input_shape;
    uint32_t bank = input_shape.size();
    if (bank == 5) {
        is3D = true;
        LOG_DEBUG("considered as 3d sequence input since network_input_shape size is 5");
    } else if (bank == 4) {
        is3D = false;
        LOG_DEBUG("considered as 2d sequence input since network_input_shape size is 5");
    } else {
        LOG_ERROR("network_input_shape size must be 4 or 5 dims. check config file");
        return NVDSPREPROCESS_CONFIG_FAILED;
    }

    _C = getChannels(_initParams.tensor_params.network_color_format);
    if (_C <= 0 || _C > 4) {
        LOG_ERROR(
            "get channels from network_color_format:%d failed",
            (int)_initParams.tensor_params.network_color_format);
        return NVDSPREPROCESS_CONFIG_FAILED;
    }
    _N = input_shape[0];
    _H = input_shape[bank - 2];
    _W = input_shape[bank - 1];
    if (is3D) {
        if (_C != (uint32_t)input_shape[1]) {
            LOG_ERROR(
                "3D channels: %d configured different from network-input-shape[1]: %d", (int)_C,
                input_shape[1]);
            return NVDSPREPROCESS_CONFIG_FAILED;
        }
        _S = (uint32_t)input_shape[2];
        DSASSERT(input_shape[2] > 0);
    } else {
        _S = input_shape[1] / _C;
        if (input_shape[1] % _C) {
            LOG_ERROR(
                "2D sequence network-input-shape[1]: %d must be mod channels: %d", input_shape[1],
                (int)_C);
            return NVDSPREPROCESS_CONFIG_FAILED;
        }
    }
    if (is3D) {
        LOG_INFO(
            "3D custom sequence network info(NCSHW), [N: %u, C: %u, S: %u, H: %u, W:%u]", _N, _C,
            _S, _H, _W);
    } else {
        auto const& dims = input_shape;
        LOG_INFO(
            "2D custom sequence network shape NSHW[%u, %u, %u, %u], reshaped as "
            "[N: %u, C: %u, S:%u, H: %u, W:%u]",
            dims[0], dims[1], dims[2], dims[3], _N, _C, _S, _H, _W);
    }
    if (_C > 4) {
        LOG_ERROR("custom sequence preprocess lib does not support channels larger than 4");
        return NVDSPREPROCESS_CONFIG_FAILED;
    }
    DSASSERT(_C && _H && _W && _S && _N);
    _means = {{0.0, 0.0, 0.0, 0.0}};
    _scales = {{1.0, 1.0, 1.0, 1.0}};
    CHECK_PROCESS_ERROR(parseUserConfig(), "custom-lib parse user-configs failed");

    CHECK_PROCESS_ERROR(setDevice(), "set gpu device failed");
    CHECK_CUDA_ERR(
        cudaStreamCreateWithPriority(&_cuStream, 0, 0),
        "cudaStreamCreateWithPriority failed");


    _bufManager = std::make_unique<BufferManager>(
        nullptr, _initParams.tensor_params, _S, _C, _cuStream, _stride, _subsample);
    DSASSERT(_bufManager);
    LOG_INFO(
        "Sequence preprocess buffer manager initialized with stride: %u, subsample: %u", _stride,
        _subsample);

    LOG_INFO("SequenceImagePreprocess initialized successfully");
    return NVDSPREPROCESS_SUCCESS;
}

// parse user-config
NvDsPreProcessStatus
SequenceImagePreprocess::parseUserConfig()
{
    auto const& tables = _initParams.user_configs;
    try {
        auto iSample = tables.find(CUSTOM_CONFIG_SUBSAMPLE);
        if (iSample != tables.end() && !trimStr(iSample->second).empty()) {
            _subsample = std::stoi(iSample->second);
        }
        auto iStride = tables.find(CUSTOM_CONFIG_STRIDE);
        if (iStride != tables.end() && !trimStr(iStride->second).empty()) {
            _stride = std::stoi(iStride->second);
        }
        auto iScales = tables.find(CUSTOM_CONFIG_CHANNEL_SCALE_FACTORS);
        if (iScales != tables.end() && !trimStr(iScales->second).empty()) {
            auto scales = parseNumList(iScales->second);
            if (scales.size()) {
                uint32_t c = 0;
                auto size_of_scales_arr = (sizeof(_scales.d))/(sizeof(_scales.d[0]));
                for (; c < _C && c < scales.size(); ++c) {
                    _scales.d[c] = scales[c];
                }
                for (; c > 0 && c < _C && c < size_of_scales_arr; ++c) {
                    _scales.d[c] = _scales.d[c - 1];
                }
            }
        }
        auto iMeans = tables.find(CUSTOM_CONFIG_CHANNEL_MEANS);
        if (iMeans != tables.end() && !trimStr(iMeans->second).empty()) {
            auto means = parseNumList(iMeans->second);
            if (means.size()) {
                uint32_t c = 0;
                auto size_of_means_arr = (sizeof(_means.d))/(sizeof(_means.d[0]));
                for (; c < _C && c < means.size(); ++c) {
                    _means.d[c] = means[c];
                }
                for (; c > 0 && c < _C && c < size_of_means_arr; ++c) {
                    _means.d[c] = _means.d[c - 1];
                }
            }
        }
    }
    catch (const std::exception& e) {
        LOG_ERROR("Catch exception parse preprocess config, error: %s", e.what());
        return NVDSPREPROCESS_CONFIG_FAILED;
    }
    catch (...) {
        LOG_ERROR("Catch unknown exception parse preprocess config");
        return NVDSPREPROCESS_CONFIG_FAILED;
    }
    return NVDSPREPROCESS_SUCCESS;
}

// Implementation of cast context deInitLib.
NvDsPreProcessStatus
SequenceImagePreprocess::deinit()
{
    LOG_INFO("SequenceImagePreprocess is deinitializing");
    if (_cuStream != nullptr) {
        setDevice();
        cudaStreamSynchronize(_cuStream);
        cudaStreamDestroy(_cuStream);
        _cuStream = nullptr;
    }
    if (_bufManager) {
        _bufManager.reset();
    }
    return NVDSPREPROCESS_SUCCESS;
}

// Implementation of cast context CustomSequenceTensorPreparation.
// - Check and build ROI blocks and sequence buffers for each incoming ROI.
// - Locate each ROI destination memory ptr in buffer blocks.
// - Cuda convert and normalize each cropped ROI image into dest cuda memory
// - Collect all ready sequence and sliding stride for next sequence.
// - pop ready batched buffers and update related parameters to Gst-nvdspreprocess plugin.
NvDsPreProcessStatus
SequenceImagePreprocess::prepareTensorData(
    NvDsPreProcessBatch* batchIn, NvDsPreProcessCustomBuf*& bufOut,
    CustomTensorParams& tensorParam, NvDsPreProcessAcquirer* allocator)
{
    LOG_DEBUG("preparing sequence TensorData in progress...");
    DSASSERT(batchIn);
    DSASSERT(allocator);
    DSASSERT(_bufManager);
    if (!_bufManager->allocator()) {
        _bufManager->setAllocator(allocator);
    }
    if (batchIn->scaling_pool_format != NvDsPreProcessFormat_RGB &&
        batchIn->scaling_pool_format != NvDsPreProcessFormat_BGR &&
        batchIn->scaling_pool_format != NvDsPreProcessFormat_RGBA &&
        batchIn->scaling_pool_format != NvDsPreProcessFormat_BGRx) {
        LOG_ERROR(
            "scaling_pool_format must be RGB based format, other format is not "
            "supported");
        return NVDSPREPROCESS_CONFIG_FAILED;
    }

    LOG_DEBUG(
        "preparing sequence tensor data received %u rois",
        (uint32_t)batchIn->units.size());
    CHECK_PROCESS_ERROR(
        _bufManager->buildRoiBlocks(batchIn->units), "build ROI blocks failed");

    CHECK_PROCESS_ERROR(setDevice(), "set gpu device failed");

    uint64_t frameId = 0;
    for (const auto& roi : batchIn->units) {
        frameId = roi.frame_num;
        DSASSERT(roi.roi_meta.converted_buffer->width >= _W);
        DSASSERT(roi.roi_meta.converted_buffer->height == _H);
        void* dstPatch = nullptr;
        NvDsPreProcessStatus s = _bufManager->locateRoiDst(roi, dstPatch);
        if (s == NVDSPREPROCESS_SKIP_FRAME) {
            continue;
        }
        CHECK_PROCESS_ERROR(s, "locate ROI dst buffer address failed");
        CHECK_PROCESS_ERROR(
            preprocessData(roi, batchIn->scaling_pool_format, dstPatch), "preprocessData failed");
    }
    if (kEnableDumpROI) {
        dumpRois(*batchIn, _H, _W, getChannels(batchIn->scaling_pool_format));
    }

    LOG_DEBUG("Trying to collect ready batches on frame: %" PRIu64, frameId);
    CHECK_PROCESS_ERROR(
        _bufManager->collectReady(), "collect ready buffers failed");

    CHECK_CUDA_ERR(
        cudaStreamSynchronize(_cuStream),
        "Failed to synch sequence process cuda stream");

    ReadyResult result;
    if (_bufManager->popReady(result)) {
        LOG_DEBUG(
            "preparing sequence batching is ready on frame: %" PRIu64
            ", batches: %u",
            frameId, (uint32_t)result.rois.size());
        bufOut = result.readyBuf;
        tensorParam.params = result.param;
        tensorParam.seq_params.roi_vector = result.rois;
        return NVDSPREPROCESS_SUCCESS;
    }

    LOG_DEBUG(
        "preparing sequence batching is not ready on frame: %" PRIu64, frameId);

    return NVDSPREPROCESS_TENSOR_NOT_READY;
}

// Implementation of entrypoint to process each ROI cropped images
NvDsPreProcessStatus
CustomSequenceTensorPreparation(
    CustomCtx* cctx, NvDsPreProcessBatch* batch, NvDsPreProcessCustomBuf*& buf,
    CustomTensorParams& tensorParam, NvDsPreProcessAcquirer* allocator)
{
    LOG_DEBUG("CustomSequenceTensorPreparation processing in progress");
    SequenceImagePreprocess* ctx =
        reinterpret_cast<SequenceImagePreprocess*>(cctx);
    DSASSERT(ctx);
    return ctx->prepareTensorData(batch, buf, tensorParam, allocator);
}

// Implementation of entrypoint to init custom lib context
CustomCtx*
initLib(CustomInitParams initparams)
{
    LOG_DEBUG("Initializing Custom sequence preprocessing lib");
    auto ctx = std::make_unique<SequenceImagePreprocess>(initparams);
    DSASSERT(ctx);
    if (ctx->init() != NVDSPREPROCESS_SUCCESS) {
        LOG_ERROR("SequenceImagePreprocess init failed");
        return nullptr;
    }
    return reinterpret_cast<CustomCtx*>(ctx.release());
}

// Implementation of entrypoint to destroy custom lib context
void
deInitLib(CustomCtx* cctx)
{
    LOG_DEBUG("Deinitializing Custom sequence preprocessing lib");
    SequenceImagePreprocess* ctx =
        reinterpret_cast<SequenceImagePreprocess*>(cctx);
    DSASSERT(ctx);
    delete ctx;
}