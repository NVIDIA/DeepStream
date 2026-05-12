/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file infer_iprocess.h
 *
 * @brief Preprocessing and postprocessing interface header file.
 */


#ifndef __NVDSINFERSERVER_IPROCESS_H__
#define __NVDSINFERSERVER_IPROCESS_H__

#include <stdarg.h>
#include <condition_variable>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <queue>
#include <tuple>

#include "infer_common.h"
#include "infer_datatypes.h"
#include "infer_utils.h"

namespace nvdsinferserver {

/**
 * @brief Preprocessor interface class.
 */
class IPreprocessor {
public:
    using PreprocessDone =
        std::function<void(NvDsInferStatus, SharedBatchArray)>;

    IPreprocessor() = default;
    virtual ~IPreprocessor() = default;

    virtual NvDsInferStatus transform(SharedBatchArray src,
        SharedCuStream mainStream, PreprocessDone done) = 0;

private:
    DISABLE_CLASS_COPY(IPreprocessor);
};

/**
 * @brief Post-processor interface class.
 */
class IPostprocessor {
public:
    using PostprocessDone =
        std::function<void(NvDsInferStatus, SharedBatchArray)>;
    IPostprocessor() = default;
    virtual ~IPostprocessor() = default;

    virtual NvDsInferStatus postCudaProcess(SharedBatchArray inBuf,
        SharedCuStream mainStream, PostprocessDone done) = 0;

    virtual NvDsInferStatus postHostProcess(SharedBatchArray inBuf,
        SharedCuStream mainStream, PostprocessDone done) = 0;

private:
    DISABLE_CLASS_COPY(IPostprocessor);
};

/**
 * @brief Base preprocessor class.
 */
class BasePreprocessor : public IPreprocessor {
public:
    /**
     * @brief Destructor default.
     */
    ~BasePreprocessor() override = default;

    /*
     * @brief Helper functions to access member variables.
     */
    /**@{*/
    void setUniqueId(int id) { m_UniqueId = id; }
    int uniqueId() const { return m_UniqueId; }
    void setTransformIdx(int idx) { m_TransformIdx = idx; }
    /**@}*/

    /**
     * @brief Perform the transformation on the input buffer from the buffer
     * array, indexed using m_TransformIdx.
     * @param[in] src        The input batch buffer array.
     * @param[in] mainStream The main processing CUDA stream.
     * @param[in] done       The call function to be called after transformation.
     * @return NVDSINFER_SUCCESS or NVDSINFER_RESOURCE_ERROR.
     */
    NvDsInferStatus transform(SharedBatchArray src, SharedCuStream mainStream,
        PreprocessDone done) override {
        assert(src && (int)src->getSize() > m_TransformIdx);
        SharedBatchBuf& from = src->buf(m_TransformIdx);
        SharedBatchBuf to = requestOutBuffer(from);
        if (!to) {
            return NVDSINFER_RESOURCE_ERROR;
        }
        to->setBufId(src->bufId());
        NvDsInferStatus status = transformImpl(from, to, mainStream);
        src->buf(m_TransformIdx) = to;
        mainStream.reset();
        if (status == NVDSINFER_SUCCESS) {
            done(status, std::move(src));
        }
        return status;
    }

    /**
     * @brief Allocate resource like output buffer pool.
     * @param[in] devIds List of device IDs.
     * @return Error status.
     */
    virtual NvDsInferStatus allocateResource(
        const std::vector<int>& devIds) = 0;

private:
    /**
     * @brief Request a buffer from the pool with same batch size as the
     * input buffer.
     * @param inBuf The input buffer being transformed.
     * @return Pointer to the acquired output buffer.
     */
    virtual SharedBatchBuf requestOutBuffer(SharedBatchBuf& inBuf) = 0;
    /**
     * @brief The core pre-process transform implementation.
     * @param[in] src        The source batch buffer.
     * @param[in] dst        The destination batch buffer.
     * @param[in] mainStream The main CUDA stream to synchronize with.
     * @return
     */
    virtual NvDsInferStatus transformImpl(SharedBatchBuf& src,
        SharedBatchBuf& dst, SharedCuStream& mainStream) = 0;

private:
    /**
     * @brief Index of the source buffer in the input buffer array to be
     * transformed.
     */
    int m_TransformIdx = 0;
    /**
     * @brief  A unique ID for the processor instance.
     */
    int m_UniqueId = 0;
};

/**
 * @brief Base post-processor class.
 */
class BasePostprocessor : public IPostprocessor {
public:
    /**
     * @brief Constructor, save process type and ID.
     */
    BasePostprocessor(InferPostprocessType type, int uid)
        : m_UniqueId(uid), m_ProcessType(type) {}
    /**
     * @brief Destructor default.
     */
    ~BasePostprocessor() override = default;

    /*
     * @brief Helper functions to access member variables.
     */
    /**@{*/
    void setUniqueId(int id) { m_UniqueId = id; }
    int uniqueId() const { return m_UniqueId; }
    InferPostprocessType networkType() const { return m_ProcessType; }
    /**@}*/

    /**
     * @brief Acquire an output buffer array and call CUDA post processing steps.
     * @param[in] inBuf      Pointer to the input batch array.
     * @param[in] mainStream The main CUDA stream to synchronize with.
     * @param[in] done       Callback function to be executed.
     * @return Error status.
     */
    NvDsInferStatus postCudaProcess(SharedBatchArray inBuf,
        SharedCuStream mainStream, PostprocessDone done) override {
        return genericPostProcess_(std::move(inBuf), std::move(mainStream),
            std::move(done), &BasePostprocessor::requestCudaOutBufs,
            &BasePostprocessor::postCudaImpl);
    }

    /**
     * @brief Acquire an output buffer array and call host side processing steps.
     * @param[in] inBuf      Pointer to the input batch array.
     * @param[in] mainStream The main CUDA stream to synchronize with.
     * @param[in] done       Callback function to be executed.
     * @return Error status.
     */
    NvDsInferStatus postHostProcess(SharedBatchArray inBuf,
        SharedCuStream mainStream, PostprocessDone done) override {
        return genericPostProcess_(std::move(inBuf), std::move(mainStream),
            std::move(done), &BasePostprocessor::requestHostOutBufs,
            &BasePostprocessor::postHostImpl);
    }

    /**
     * @brief Allocate resource like output buffer pool.
     * @param[in] devIds List of device IDs.
     * @return Error status.
     */
    virtual NvDsInferStatus allocateResource(
        const std::vector<int>& devIds) = 0;

private:
    /**
     * @brief CUDA post processing steps.
     * @param[in] inBuf      Pointer to the input batch array.
     * @param[out] outbuf    Pointer to the output batch array.
     * @param[in] mainStream The main CUDA stream to synchronize with.
     * @return Error status.
     */
    virtual NvDsInferStatus postCudaImpl(SharedBatchArray& inBuf,
        SharedBatchArray& outbuf, SharedCuStream& mainStream) = 0;
    /**
     * @brief Host side post processing steps.
     * @param[in] inBuf      Pointer to the input batch array.
     * @param[out] outbuf    Pointer to the output batch array.
     * @param[in] mainStream The main CUDA stream to synchronize with.
     * @return Error status.
     */
    virtual NvDsInferStatus postHostImpl(SharedBatchArray& inBuf,
        SharedBatchArray& outbuf, SharedCuStream& mainStream) = 0;

    /**
     * @brief Acquire an output batch buffer array for CUDA post processing
     * using the dimensions from the input buffers.
     * @param[in] inBuf Array of the input batch buffers.
     * @return Pointer to the acquired output buffer array.
     */
    virtual SharedBatchArray requestCudaOutBufs(const SharedBatchArray& inBuf) = 0;
    /**
     * @brief Acquire an output batch buffer array for host side post
     * processing using the dimensions from the input buffers.
     * @param[in] inBuf Array of the input batch buffers.
     * @return Pointer to the acquired output buffer array.
     */
    virtual SharedBatchArray requestHostOutBufs(const SharedBatchArray& inBuf) = 0;

    /**
     * @brief Generic post processing implementation calling function.
     * @param[in] inBuf      Array of input batch buffers.
     * @param[in] mainStream The main CUDA stream to synchronize with.
     * @param[in] done       Callback function to be executed after processing step.
     * @param[in] reqBuf     The function to acquire output buffers.
     * @param[in] impl       The post processing step implementation.
     * @return
     */
    template <typename RequestBuf, typename Impl>
    NvDsInferStatus genericPostProcess_(SharedBatchArray inBuf,
        SharedCuStream mainStream, PostprocessDone done, RequestBuf reqBuf,
        Impl impl) {
        assert(inBuf);
        SharedBatchArray outBuf = (this->*reqBuf)(inBuf);
        if (!outBuf) {
            return NVDSINFER_UNKNOWN_ERROR;
        }
        outBuf->setBufId(inBuf->bufId());
        NvDsInferStatus status = (this->*impl)(inBuf, outBuf, mainStream);
        inBuf.reset();
        mainStream.reset();
        if (status == NVDSINFER_SUCCESS) {
            done(status, std::move(outBuf));
        }
        return status;
    }

private:
    /**
     * @brief  A unique ID for the processor instance.
     */
    int m_UniqueId = 0;
    /**
     * @brief Type of the post-processing instance.
     */
    InferPostprocessType m_ProcessType = InferPostprocessType::kOther;
};

/**
 * @brief Preprocessor thread queue class template.
 * @tparam BasePreprocessorT The preprocessor class used by the thread loop.
 */
template <class BasePreprocessorT>
class ThreadPreprocessor : public BasePreprocessorT {
private:
    using Item = std::tuple<SharedBatchArray, SharedCuStream,
        IPreprocessor::PreprocessDone>;
    enum { kBuf, kStream, kDone };

public:
    template <typename... Args>
    ThreadPreprocessor(Args&&... args)
        : BasePreprocessorT(std::forward<Args>(args)...),
          m_Worker(
              [this](Item i) -> bool {
                  IPreprocessor::PreprocessDone done =
                      std::move(std::get<kDone>(i));
                  NvDsInferStatus s = BasePreprocessorT::transform(
                      std::move(std::get<kBuf>(i)),
                      std::move(std::get<kStream>(i)), done);
                  if (s != NVDSINFER_SUCCESS) {
                      done(s, nullptr);
                  }
                  return true;
              },
              "Preproc")
    {
    }
    ~ThreadPreprocessor() override { m_Worker.join(); }
    void setThreadName(const std::string &name) {
        m_Worker.setThreadName(name);
    }
    NvDsInferStatus transform(SharedBatchArray src, SharedCuStream mainStream,
        IPreprocessor::PreprocessDone done) final {
        Item item = std::make_tuple(
            std::move(src), std::move(mainStream), std::move(done));
        if (m_Worker.queueItem(std::move(item))) {
            return NVDSINFER_SUCCESS;
        }
        return NVDSINFER_RESOURCE_ERROR;
    }

private:
    QueueThread<std::vector<Item>> m_Worker;
};

/**
 * @brief A CUDA post processor thread queue template class.
 * @tparam BasePostprocessorT The post processor used by the thread loop.
 */
template <class BasePostprocessorT>
class ThreadCudaPostprocessor : public BasePostprocessorT {
    using ItemCuda = std::tuple<SharedBatchArray, SharedCuStream,
        IPostprocessor::PostprocessDone>;
    enum { kBuf, kStream, kDone };

public:
    template <typename... Args>
    ThreadCudaPostprocessor(Args&&... args)
        : BasePostprocessorT(std::forward<Args>(args)...),
          m_CudaWorker(
              [this](ItemCuda i) -> bool {
                  IPostprocessor::PostprocessDone done =
                      std::move(std::get<kDone>(i));
                  NvDsInferStatus s = BasePostprocessorT::postCudaProcess(
                      std::move(std::get<kBuf>(i)),
                      std::move(std::get<kStream>(i)), done);
                  if (s != NVDSINFER_SUCCESS) {
                      done(s, nullptr);
                  }
                  return true;
              },
              "PostCuda")
    {}
    ~ThreadCudaPostprocessor() { m_CudaWorker.join(); }
    NvDsInferStatus postCudaProcess(SharedBatchArray inBuf,
        SharedCuStream mainStream, IPostprocessor::PostprocessDone done) final {
        ItemCuda item = std::make_tuple(
            std::move(inBuf), std::move(mainStream), std::move(done));
        if (m_CudaWorker.queueItem(std::move(item))) {
            return NVDSINFER_SUCCESS;
        }
        return NVDSINFER_RESOURCE_ERROR;
    }

private:
    QueueThread<std::vector<ItemCuda>> m_CudaWorker;
};

/**
 * @brief A host post processor thread queue template class.
 * @tparam BasePostprocessorT The post processor used by the thread loop.
 */
template <typename BasePostprocessorT>
class ThreadHostPostprocessor : public BasePostprocessorT {
    using ItemHost = std::tuple<SharedBatchArray, SharedCuStream,
        IPostprocessor::PostprocessDone>;
    enum { kBuf, kStream, kDone };

public:
    template <typename... Args>
    ThreadHostPostprocessor(Args&&... args)
        : BasePostprocessorT(std::forward<Args>(args)...),
          m_HostWorker(
              [this](ItemHost i) -> bool {
                  IPostprocessor::PostprocessDone done =
                      std::move(std::get<kDone>(i));
                  NvDsInferStatus s = BasePostprocessorT::postHostProcess(
                      std::move(std::get<kBuf>(i)),
                      std::move(std::get<kStream>(i)), done);
                  if (s != NVDSINFER_SUCCESS) {
                      done(s, nullptr);
                  }
                  return true;
              },
              "PostHost")
    {
    }

    ~ThreadHostPostprocessor() { m_HostWorker.join(); }

    NvDsInferStatus postHostProcess(SharedBatchArray inBuf,
        SharedCuStream mainStream, IPostprocessor::PostprocessDone done) final {
        ItemHost item = std::make_tuple(
            std::move(inBuf), std::move(mainStream), std::move(done));
        if (m_HostWorker.queueItem(std::move(item))) {
            return NVDSINFER_SUCCESS;
        }
        return NVDSINFER_RESOURCE_ERROR;
    }

private:
    QueueThread<std::vector<ItemHost>> m_HostWorker;
};

} // namespace nvdsinferserver

#endif
