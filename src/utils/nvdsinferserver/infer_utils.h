/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file infer_utils.h
 *
 * @brief Header file containing utility functions and classes used by
 * the nvinferserver low level library.
 */

#ifndef __NVDSINFER_SERVER_INFER_UTILS_H__
#define __NVDSINFER_SERVER_INFER_UTILS_H__

#include <infer_batch_buffer.h>
#include <infer_common.h>
#include <infer_datatypes.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>

namespace nvdsinferserver INFER_EXPORT_API {

/**
 * @brief Print the nvinferserver log messages as per the configured
 * log level.
 */
void dsInferLogPrint__(NvDsInferLogLevel level, const char* fmt, ...);

/**
 * @brief Helper function to print the nvinferserver logs.
 *
 * This functions prints the log message to stderr or stdout depending
 * on the input level. If the input level is more that the log
 * level configured by the environment variable NVDSINFERSERVER_LOG_LEVEL
 * (default level NVDSINFER_LOG_INFO), the log message is discarded.
 * Messages of level NVDSINFER_LOG_ERROR are output to stderr others to
 * stdout. A global mutex is used to guard concurrent prints from multiple
 * threads.
 *
 * @param[in] level Log level of the message.
 * @param[in] fmt   The fprintf format string of the log message.
 * @param[in] args  The variable argument list for fprintf.
 */
void dsInferLogVPrint__(NvDsInferLogLevel level, const char* fmt, va_list args);


/**
 * @brief Helper functions to get a safe C-string representation for the
 * input string. Returns an empty string if the input pointer is null.
 */
/**@{*/
inline const char* safeStr(const char* str) {
    return !str ? "" : str;
}

inline const char* safeStr(const std::string& str) {
    return str.c_str();
}
/**@}*/


/**
 * @brief Helper function, returns true if the input C string is empty or null.
 */
inline bool string_empty(const char* str) {
    return !str || strlen(str) == 0;
}

/**
 * @brief Helper functions to check if the input file path
 * is valid and accessible.
 */
/**@{*/
inline bool file_accessible(const char* path) {
    assert(path);
    return (access(path, F_OK) != -1);
}

inline bool file_accessible(const std::string& path) {
    return (!path.empty()) && file_accessible(path.c_str());
}
/**@}*/

/**
 * @brief Checks if the input batch size is zero.
 */
template <typename T>
inline bool
isNonBatch(T b)
{
    return b == 0;
}

/**
 * @brief Helper functions to convert the various data types to string values
 * for debug, log information.
 */
/**@{*/
std::string dims2Str(const InferDims& d);
std::string batchDims2Str(const InferBatchDims& d);
std::string dataType2Str(const InferDataType type);
std::string dataType2GrpcStr(const InferDataType type);
InferDataType grpcStr2DataType(const std::string &type);
NvDsInferNetworkInfo dims2ImageInfo(
    const InferDims& d, InferTensorOrder order);
std::string tensorOrder2Str(InferTensorOrder order);
/**@}*/

/**
 * @brief Check if the two floating point values are equal, the difference
 * is less than or equal to the epsilon value.
 */
bool fEqual(float a, float b);

/**
 * @brief Helper class for dynamic loading of custom library.
 */
class DlLibHandle {
public:
    /**
     * @brief Constructor. Open the shared library with the given mode.
     * @param path Filename of the dynamic library.
     * @param mode Flags to be passed to dlopen call.
     */
    DlLibHandle(const std::string& path, int mode = RTLD_LAZY);

    /*
     * @brief Destructor. Close the dynamically loaded library.
     */
    ~DlLibHandle();

    /*
     * @brief Check that the library handle is valid.
     */
    bool isValid() const { return m_LibHandle; }

    /*
     * @brief Get the filename of the library.
     */
    const std::string& getPath() const { return m_LibPath; }

    /*
     * @brief Get the function pointer from the library for given function
     * name.
     */
    /**@{*/
    template <typename FuncPtr>
    FuncPtr symbol(const char* func) {
        assert(!string_empty(func));
        if (!m_LibHandle)
            return nullptr;
        InferDebug("lib: %s dlsym :%s", safeStr(m_LibPath), safeStr(func));
        return (FuncPtr)dlsym(m_LibHandle, func);
    }

    template <typename FuncPtr>
    FuncPtr symbol(const std::string& func) {
        return symbol<FuncPtr>(func.c_str());
    }
    /**@}*/

private:
    /*
     * @brief Handle for the library returned by dlopen().
     */
    void* m_LibHandle{nullptr};
    /*
     * @brief Filename of the dynamically loaded library.
     */
    const std::string m_LibPath;
};

/**
 * @brief Wrapper class for handling exception.
 */
class WakeupException : public std::exception {
    std::string m_Msg;

public:
    WakeupException(const std::string& s) : m_Msg(s) {}
    const char* what() const noexcept override { return m_Msg.c_str(); }
};

/**
 * @brief Template class for creating a thread safe queue for the given
 * container class.
 *
 * @tparam Container The container class for the queue, e.g. std::queue, std::list.
 */
template <typename Container>
class GuardQueue {
public:
    typedef typename Container::value_type T;
    /**
     * @brief Push an item to the queue.
     */
    void push(T data) {
        std::unique_lock<std::mutex> lock(m_Mutex);
        m_Queue.emplace_back(std::move(data));
        m_Cond.notify_one();
    }
    /**
     * @brief Pop an item from the queue.
     *
     * Blocking call. Returns when there is an item in queue or on wakeup trigger.
     * Throws exception on wakeup trigger.
     */
    T pop() {
        std::unique_lock<std::mutex> lock(m_Mutex);
        m_Cond.wait(
            lock, [this]() { return m_WakeupOnce || !m_Queue.empty(); });
        if (m_WakeupOnce) {
            m_WakeupOnce = false;
            InferDebug("GuardQueue pop end on wakeup signal");
            throw WakeupException("GuardQueue stopped");
        }
        assert(!m_Queue.empty());
        T ret = std::move(*m_Queue.begin());
        m_Queue.erase(m_Queue.begin());
        return ret;
    }
    /**
     * @brief Send the wakeup trigger to the queue thread.
     */
    void wakeupOnce() {
        InferDebug("GuardQueue trigger wakeup once");
        std::unique_lock<std::mutex> lock(m_Mutex);
        m_WakeupOnce = true;
        m_Cond.notify_all();
    }
    /**
     * @brief Clear the queue.
     */
    void clear() {
        InferDebug("GuardQueue clear");
        std::unique_lock<std::mutex> lock(m_Mutex);
        m_Queue.clear();
        m_WakeupOnce = false;
    }
    /**
     * @brief Current size of the queue.
     */
    int size() {
        std::unique_lock<std::mutex> lock(m_Mutex);
        return m_Queue.size();
    }

private:
    /**
     * @brief Mutex for access lock.
     */
    std::mutex m_Mutex;
    /**
     * @brief Condition variable for thread synchronization.
     */
    std::condition_variable m_Cond;
    /**
     * @brief The queue element of the container class.
     */
    Container m_Queue;
    /**
     * @brief Wake up trigger flag.
     */
    bool m_WakeupOnce = false;
};

/**
 * @brief Template class for running the specified function on the queue items
 * in a separate thread.
 *
 * @tparam Container The container class for the queue, e.g. std::queue,
 * std::list.
 */
template <typename Container>
class QueueThread {
public:
    using Item = typename Container::value_type;
    using RunFunc = std::function<bool(Item)>;

    /**
     * @brief Create a new thread that runs the specified function over the
     * queued items in a loop.
     *
     * @param[in] runFunc The processing function of the thread.
     * @param[in] name Name for the thread.
     */
    QueueThread(RunFunc runFunc, const std::string& name) : m_Run(runFunc) {
        std::promise<void> p;
        std::future<void> f = p.get_future();
        InferDebug("QueueThread starting new thread");
        m_Thread = std::thread([&p, this]() {
            p.set_value();
            this->threadLoop();
        });
        setThreadName(name);
        f.wait();
    }
    /**
     * @brief Set the internal (m_Name) name of the thread and system name
     * using pthread_setname_np().
     */
    void setThreadName(const std::string &name) {
        assert(!name.empty());
        m_Name = name;
        if (m_Thread.joinable()) {
            const int kMakLen = 16;
            char cName[kMakLen];
            strncpy(cName, name.c_str(), kMakLen);
            cName[kMakLen - 1] = 0;
            if (pthread_setname_np(m_Thread.native_handle(), cName) != 0) {
                InferError("set thread name: %s failed", safeStr(name));
                return;
            }
            InferDebug("QueueThread set new thread name:%s", cName);
        }
    }
    /**
     * @brief Destructor. Send a wake up trigger to the queue, wait for the
     * thread to join and clear the queue.
     */
    ~QueueThread() { join(); }
    void join() {
        InferDebug("QueueThread: %s join", safeStr(m_Name));
        if (m_Thread.joinable()) {
            m_Queue.wakeupOnce();
            m_Thread.join();
        }
        m_Queue.clear();
    }
    /**
     * @brief Add an item to the queue for processing.
     */
    bool queueItem(Item item) {
        m_Queue.push(std::move(item));
        return true;
    }

private:
    /**
     * @brief Processing loop of the thread.
     *
     * This function calls the processing function on the queued items in
     * a loop. It exits the loop on receiving exception from the wake up
     * trigger or if the processing returns fail.
     */
    void threadLoop() {
        while (true) {
            try {
                Item item = m_Queue.pop();
                if (!m_Run(std::move(item))) {
                    InferDebug("QueueThread:%s return and stop", safeStr(m_Name));
                    return;
                }
            }
            catch (const WakeupException& e) {
                InferDebug("QueueThread:%s stopped", safeStr(m_Name));
                return;
            }
            catch (...) {  // unexpected
                InferError(
                    "QueueThread:%s internal unexpected error, may cause stop",
                    safeStr(m_Name));
                // Usually can move on to next, but need developer to check
                continue;
            }
        }
    }

private:
    /**
     * @brief The processing thread.
     */
    std::thread m_Thread;
    /**
     * @brief Name of the thread.
     */
    std::string m_Name;
    /**
     * @brief The processing function for the thread.
     */
    RunFunc m_Run;
    /**
     * @brief Queue of items for processing.
     */
    GuardQueue<Container> m_Queue;
};

/**
 * @brief Template class for buffer pool of the specified buffer type.
 * @tparam UniPtr Unique pointer type of the buffers in the pool.
 */
template <class UniPtr>
class BufferPool : public std::enable_shared_from_this<BufferPool<UniPtr> > {
public:
    using ItemType = typename UniPtr::element_type;
    using RecylePtr = std::unique_ptr<ItemType, std::function<void(ItemType*)>>;
    /**
     * @brief Constructor. Name the pool.
     */
    BufferPool(const std::string &name)
        : m_Name(name) {}
    /**
     * @brief Destructor. Print debug message of number of free buffers.
     */
    virtual ~BufferPool()
    {
        InferDebug(
            "BufferPool: %s deleted with free buffer size:%d", safeStr(m_Name),
            m_FreeBuffers.size());
    }
    /**
     * @brief Add a buffer to the pool.
     * @param[in] buf Unique pointer to a buffer to be added to the pool.
     * @return Boolean error status.
     */
    bool setBuffer(UniPtr buf) {
        assert(buf);
        buf->reuse();
        m_FreeBuffers.push(std::move(buf));
        InferDebug("BufferPool: %s set buf to free, available size:%d",
            safeStr(m_Name), m_FreeBuffers.size());
        return true;
    }
    /**
     * @brief Get the number of free buffers.
     */
    int size() { return m_FreeBuffers.size(); }

    /**
     * @brief Acquire a buffer from the pool.
     *
     * This function pops a buffer from the pool queue. A deleter function is
     * associate with the buffer pointer to return the buffer to the pool when
     * it is no longer needed.
     *
     * @return A unique pointer with corresponding deleter function.
     */
    RecylePtr acquireBuffer() {
        try {
            UniPtr p = m_FreeBuffers.pop();
            auto deleter = p.get_deleter();
            std::weak_ptr<BufferPool<UniPtr>> poolPtr =
                this->shared_from_this();
            RecylePtr recBuf(
                p.release(), [poolPtr, d = deleter](ItemType* buf) {
                assert(buf);
                UniPtr data(buf, d);
                auto pool = poolPtr.lock();
                if (pool) {
                    InferDebug("BufferPool: %s release a buffer", safeStr(pool->m_Name));
                    pool->setBuffer(std::move(data));
                } else {
                    InferError("BufferPool is deleted, check internal error.");
                    assert(false);
                }
            });
            InferDebug("BufferPool: %s acquired buffer, available free buffer left:%d",
                safeStr(m_Name), m_FreeBuffers.size());
            return recBuf;
        } catch (...) {
            InferDebug(
                "BufferPool: %s acquired buffer failed, queue maybe waked up.",
                safeStr(m_Name));
            assert(false);
            return nullptr;
        }
    }

private:
    /**
     * @brief Guarded queue holding the unique pointers for the free buffers.
     */
    GuardQueue<std::deque<UniPtr>> m_FreeBuffers;
    /**
     * @brief Name of the buffer.
     */
    const std::string m_Name;
};

template <class UniPtr>
using SharedBufPool = std::shared_ptr<BufferPool<UniPtr>>;

/**
 * @brief Template class for a map of buffer pools.
 * @tparam Key Type of the map key.
 * @tparam UniqBuffer Type of the unique pointers to the buffers.
 */
template<typename Key, typename UniqBuffer>
class MapBufferPool {
public:
    using SharedPool = SharedBufPool<UniqBuffer>;
    using RecylePtr = typename BufferPool<UniqBuffer>::RecylePtr;
public:
    /**
     * @brief Construct the buffer pool map with a name.
     */
    MapBufferPool(const std::string &name): m_Name(name) {}
    /**
     * @brief Destructor. Print a debug message of number of pools in the map.
     */
    virtual ~MapBufferPool()
    {
        InferDebug(
            "MapBufferPool: %s deleted with buffer pool size:%d",
            safeStr(m_Name), (int)m_MapPool.size());
    }

    /** Disable copy operations */
    /**@{*/
    MapBufferPool(const MapBufferPool& other) = delete;
    MapBufferPool& operator=(const MapBufferPool& other) = delete;
    /**@}*/

    /**
     * @brief Add a buffer to the pool map.
     *
     * This function adds a new buffer to the pool specified by the key.
     * If the pool for the key is not found, a new one is created.
     *
     * @param key Map key of the pool in the map.
     * @param buf Unique pointer to the buffer to be added.
     * @return Boolean error status. True if success.
     */
    bool setBuffer(const Key &key, UniqBuffer buf) {
        std::unique_lock<std::shared_timed_mutex> uniqLock(m_MapPoolMutex);
        assert(buf);
        SharedPool &pool = m_MapPool[key];
        if(!pool) {
            uint32_t id = m_MapPool.size() - 1;
            std::string poolName = m_Name + std::to_string(id);
            pool = std::make_shared<BufferPool<UniqBuffer>>(poolName);
            assert(pool);
            InferDebug("MapBufferPool: %s create new pool id:%d",
                safeStr(m_Name), id);
        }
        if(!pool) {
            return false;
        }
        return pool->setBuffer(std::move(buf));
    }
    /**
     * @brief Get the size of a pool from the map.
     * @param key Map key of the pool.
     * @return Number of buffers in the pool if found, 0 otherwise.
     */
    uint32_t getPoolSize(const Key &key) {
        SharedPool pool = findPool(key);
        if (!pool)
            return 0;
        return pool->size();
    }

    /**
     * @brief Acquire a buffer from the selected pool.
     * @param key Map key to identify the pool.
     * @return Unique pointer to the buffer with deleter function to return
     * the buffer to the pool.
     */
    RecylePtr acquireBuffer(const Key &key) {
        SharedPool pool = findPool(key);
        assert(pool);
        if (!pool) {
            InferWarning(
                "MapBufferPool: %s acquire buffer failed, no key found",
                safeStr(m_Name));
            return nullptr;
        }
        InferDebug("MapBufferPool: %s acquire buffer", safeStr(m_Name));
        return pool->acquireBuffer();
    }
    /**
     * @brief Remove all pools from the map.
     */
    void clear() {
        InferDebug("MapBufferPool: %s clear all buffers", safeStr(m_Name));
        std::unique_lock<std::shared_timed_mutex> uniqLock(m_MapPoolMutex);
        m_MapPool.clear();
    }

private:
    /**
     * @brief Find the pool for the given key.
     */
    SharedPool findPool(const Key& key) {
        std::shared_lock<std::shared_timed_mutex> sharedLock(m_MapPoolMutex);
        auto iter = m_MapPool.find(key);
        if (iter != m_MapPool.end()) {
            assert(iter->second);
            return iter->second;
        }
        return nullptr;
    }

private:
    /**
     * @brief Map of the buffer pools.
     */
    std::map<Key, SharedPool> m_MapPool;
    /**
     * @brief A shared mutex with timeout to access the map.
     */
    std::shared_timed_mutex m_MapPoolMutex;
    /**
     * @brief Name of the buffer pool map.
     */
    const std::string m_Name;
};

/**
 * @brief Get the size of the element from the data type.
 */
inline uint32_t getElementSize(InferDataType t) {
    switch (t) {
    case InferDataType::kInt32:
    case InferDataType::kUint32:
    case InferDataType::kFp32:
        return 4;
    case InferDataType::kFp16:
    case InferDataType::kInt16:
    case InferDataType::kUint16:
        return 2;
    case InferDataType::kInt8:
    case InferDataType::kUint8:
    case InferDataType::kBool:
        return 1;
    case InferDataType::kString:
        return 0;
    case InferDataType::kFp64:
    case InferDataType::kInt64:
    case InferDataType::kUint64:
        return 8;
    default:
        InferError("Failed to get element size on Unknown datatype:%d",
            static_cast<int>(t));
        return 0;
    }
}

/**
 * @brief Check if any of the InferDims dimensions are of dynamic size
 * (-1 or negative values).
 */
inline bool
hasWildcard(const InferDims& dims)
{
    return std::any_of(
        dims.d, dims.d + dims.numDims,
        [](int d) { return d <= INFER_WILDCARD_DIM_VALUE; });
}

/**
 * @brief Calculate the total number of elements for the given dimensions.
 *
 * @param dims  Input Input dimensions.
 * @return Total number of elements, 0 in case of dynamic size.
 */
inline size_t
dimsSize(const InferDims& dims)
{
    if (hasWildcard(dims) || !dims.numDims) {
        return 0;
    } else {
        return std::accumulate(
            dims.d, dims.d + dims.numDims, 1,
            [](int s, int i) { return s * i; });
    }
}

/**
 * @brief Recalculates the total number of elements for the dimensions.
 * @param dims Input dimensions.
 */
inline void
normalizeDims(InferDims& dims)
{
    dims.numElements = dimsSize(dims);
}

/**
 * @brief Comparison operators for the InferDims type.
 */
/**@{*/
bool operator<=(const InferDims& a, const InferDims& b);
bool operator>(const InferDims& a, const InferDims& b);
bool operator==(const InferDims& a, const InferDims& b);
bool operator!=(const InferDims& a, const InferDims& b);
/**@}*/

struct LayerDescription;

/**
 * @brief Convert the layer description and buffer pointer to
 * NvDsInferLayerInfo of the interface.
 */
NvDsInferLayerInfo toCapi(const LayerDescription& desc, void* bufPtr);

/**
 * @brief Convert the InferDims to NvDsInferDims of the library
 * interface.
 */
NvDsInferDims toCapi(const InferDims &dims);

/**
 * @brief Generate NvDsInferLayerInfo of the interface from the buffer
 * description and buffer pointer.
 */
NvDsInferLayerInfo toCapiLayerInfo(
    const InferBufferDescription& desc, void* buf = nullptr);

/**
 * @brief Convert the InferDataType to NvDsInferDataType of the library
 * interface.
 */
NvDsInferDataType toCapiDataType(InferDataType dt);

/**
 * @brief Get the intersection of the two input dimensions.
 *
 * This functions derives the intersections of the two input dimensions by
 * replacing the wild card dimensions (dynamic sized) with the corresponding
 * value from the other input. The functions returns failure if the two inputs
 * have different number of dimensions or if the two corresponding dimensions
 * are of fixed size but different.
 *
 * @param[in]  a First input dimensions.
 * @param[in]  b Second input dimensions.
 * @param[out] c The derived output intersection.
 * @return True if the intersection could be found, false otherwise.
 */
bool intersectDims(
    const InferDims& a, const InferDims& b, InferDims& c);

/**
 * @brief Check if the given tensor is marked as private (contains
 * INFER_SERVER_PRIVATE_BUF in the name). Private tensors are skipped
 * in inference output processing.
 */
bool isPrivateTensor(const std::string &tensorName);

/**
 * @brief Helper functions for parsing the configuration file.
 */
/**@{*/
std::string joinPath(const std::string& a, const std::string& b);
std::string dirName(const std::string& path);
bool isAbsolutePath(const std::string& path);
bool realPath(const std::string &inPath, std::string &absPath);
/**@}*/

/**
 * @brief Check if the memory type uses CPU memory (kCpu or kCpuCuda).
 */
bool isCpuMem(InferMemType type);

/**
 * @brief Returns a string object corresponding to the InferMemType name.
 */
std::string memType2Str(InferMemType type);

/**
 * @brief Extend the dimensions to include batch size.
 * @param[in] batchSize Input batch size.
 * @param[in] in Input dimensions.
 * @return Extended dimensions with batch size added as first dimension.
 */
InferDims fullDims(int batchSize, const InferDims& in);

/**
 * @brief Separates batch size from given dimensions.
 * @param[in] full Input full dimensions with batch size.
 * @param[out] debatched Output dimensions without the batch size.
 * @param[out] batch Batch size of the input dimensions.
 * @return True if batch size could be derived (number dimensions >= 1),
 *         false otherwise.
 */
bool debatchFullDims(
    const InferDims& full, InferDims& debatched, uint32_t& batch);

/**
 * @brief Check that the two dimensions are equal ignoring single element
 *        values.
 *
 * @param[in] a First set of inference dimensions.
 * @param[in] b Second set of inference dimensions.
 * @return True if the two dimensions match.
 */
bool squeezeMatch(const InferDims& a, const InferDims& b);

/**
 * @brief Update the buffer dimensions as per provided new dimensions.
 *
 * @param[in] in          Input batch buffer.
 * @param[in] batch       Expected batch size.
 * @param[in] dims        New buffer dimensions.
 * @param[in] reCalcBytes Flag to enable recalculation of total number of
 *                        bytes in buffer based on expected batch size and new
 *                        dimensions.
 * @return                Shared pointer to new batch buffer pointing to
 *                        same memory with updated dimensions.
 */
SharedBatchBuf ReshapeBuf(
    const SharedBatchBuf& in, uint32_t batch, const InferDims& dims,
    bool reCalcBytes = false);

/**
 * @brief Reshape the buffer dimensions with batch size added as new dimension.
 *
 * @param[in] buf         Input batch buffer.
 * @param[in] reCalcBytes Flag to enable recalculation of total number of
 *                        bytes in buffer.
 * @return                Shared pointer to new batch buffer pointing to
 *                        same memory with updated dimensions.
 */
SharedBatchBuf reshapeToFullDimsBuf(const SharedBatchBuf& buf,
        bool reCalcBytes = false);

/**
 * @brief Copy one tensor buffer to another.
 *
 * This functions copies the data from one batch buffer to another.
 * Both the buffers must have the same total number of bytes. In case of
 * copy to or from device memory cudaMemcpyAsync is used with the provided
 * CUDA stream and GPU ID. For copy between host memory buffers memcpy is
 * used.
 *
 * @param[in] in Source batch buffer.
 * @param[out] out Destination btach buffer.
 * @param[in] stream CUDA stream for use with cudaMemcpyAsync
 * @return NVDSINFER_SUCCESS on success or NVDSINFER_CUDA_ERROR on error.
 */
NvDsInferStatus tensorBufferCopy(
    const SharedBatchBuf& in, const SharedBatchBuf& out,
    const SharedCuStream& stream);

} // namespace nvdsinferserver

extern "C" {

/**
 * @brief Returns the NvDsInferStatus enum name as a string.
 */
INFER_EXPORT_API const char* NvDsInferStatus2Str(NvDsInferStatus status);

/**
 * @brief Validates the provided nvinferserver configuration string.
 *
 * This function parses the input configuration string into the InferenceConfig
 * configuration protobuf message and validates it. It provides an updated
 * configuration string with the validated and if required modified
 * configuration.
 *
 * @param[in]  configStr Input string read from the configuration file.
 * @param[in]  path      Path of the configuration file.
 * @param[out] updated   Validated and updated configuration string.
 * @return Boolean indicating if the validation passed or failed.
 */
INFER_EXPORT_API bool validateInferConfigStr(
    const std::string& configStr, const std::string& path, std::string& updated);
}

#endif
