/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef __NVDSINFER_FUNC_UTILS_H__
#define __NVDSINFER_FUNC_UTILS_H__

#include <dlfcn.h>
#include <stdarg.h>
#include <string.h>
#include <unistd.h>
#include <cassert>
#include <condition_variable>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_set>

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <nvdsinfer.h>
#include <nvdsinfer_context.h>
#include <nvdsinfer_logger.h>

/* This file provides APIs/macros for some frequently used functionality. */

#define DISABLE_CLASS_COPY(NoCopyClass)       \
    NoCopyClass(const NoCopyClass&) = delete; \
    void operator=(const NoCopyClass&) = delete

#define SIMPLE_MOVE_COPY(Cls)    \
    Cls& operator=(Cls&& o) {    \
        move_copy(std::move(o)); \
        return *this;            \
    }                            \
    Cls(Cls&& o) { move_copy(std::move(o)); }

#define CHECK_NVINFER_ERROR(err, action, fmt, ...)                         \
    do {                                                                   \
        NvDsInferStatus ifStatus = (err);                                  \
        if (ifStatus != NVDSINFER_SUCCESS) {                               \
            auto errStr = NvDsInferStatus2Str(ifStatus);                   \
            dsInferError(fmt ", nvinfer error:%s", ##__VA_ARGS__, errStr); \
            action;                                                        \
        }                                                                  \
    } while (0)

#define RETURN_NVINFER_ERROR(err, fmt, ...) \
    CHECK_NVINFER_ERROR(err, return ifStatus, fmt, ##__VA_ARGS__)

#define CHECK_CUDA_ERR_W_ACTION(err, action, fmt, ...)                      \
    do {                                                                    \
        cudaError_t errnum = (err);                                         \
        if (errnum != cudaSuccess) {                                        \
            dsInferError(fmt ", cuda err_no:%d, err_str:%s", ##__VA_ARGS__, \
                (int)errnum, cudaGetErrorName(errnum));                     \
            action;                                                         \
        }                                                                   \
    } while (0)

#define CHECK_CUDA_ERR_NO_ACTION(err, fmt, ...) \
    CHECK_CUDA_ERR_W_ACTION(err, , fmt, ##__VA_ARGS__)

#define RETURN_CUDA_ERR(err, fmt, ...) \
    CHECK_CUDA_ERR_W_ACTION(           \
        err, return NVDSINFER_CUDA_ERROR, fmt, ##__VA_ARGS__)

#define READ_SYMBOL(lib, func_name) \
    lib->symbol<decltype(&func_name)>(#func_name)

namespace nvdsinfer {

inline const char* safeStr(const char* str)
{
    return !str ? "" : str;
}

inline const char* safeStr(const std::string& str)
{
    return str.c_str();
}

inline bool string_empty(const char* str)
{
    return !str || strlen(str) == 0;
}

inline bool file_accessible(const char* path)
{
    assert(path);
    return (access(path, F_OK) != -1);
}

inline bool file_accessible(const std::string& path)
{
    return (!path.empty()) && file_accessible(path.c_str());
}

std::string dims2Str(const nvinfer1::Dims& d);
std::string dims2Str(const NvDsInferDims& d);
std::string batchDims2Str(const NvDsInferBatchDims& d);

std::string dataType2Str(const nvinfer1::DataType type);
std::string dataType2Str(const NvDsInferDataType type);
std::string networkMode2Str(const NvDsInferNetworkMode type);

/* Custom unique_ptr subclass with deleter functions for TensorRT objects. */
template <class T>
class UniquePtrWDestroy : public std::unique_ptr<T, void (*)(T*)>
{
public:
    UniquePtrWDestroy(T* t = nullptr)
        : std::unique_ptr<T, void (*)(T*)>(t, [](T* t) {
              if (t)
                  t->destroy();
          }) {}
};

template <class T>
class SharedPtrWDestroy : public std::shared_ptr<T>
{
public:
    SharedPtrWDestroy(T* t = nullptr)
        : std::shared_ptr<T>(t, [](T* t) {
              if (t)
                  t->destroy();
          }) {}
};

class DlLibHandle
{
public:
    DlLibHandle(const std::string& path, int mode = RTLD_LAZY);
    ~DlLibHandle();

    bool isValid() const { return m_LibHandle; }
    const std::string& getPath() const { return m_LibPath; }

    template <typename FuncPtr>
    FuncPtr symbol(const char* func)
    {
        assert(!string_empty(func));
        if (!m_LibHandle)
            return nullptr;
        return (FuncPtr)dlsym(m_LibHandle, func);
    }

    template <typename FuncPtr>
    FuncPtr symbol(const std::string& func)
    {
        return symbol<FuncPtr>(func.c_str());
    }

private:
    void* m_LibHandle{nullptr};
    const std::string m_LibPath;
};


template <typename Container>
class GuardQueue
{
public:
    typedef typename Container::value_type T;
    void push(const T& data)
    {
        std::unique_lock<std::mutex> lock(m_Mutex);
        m_Queue.push_back(data);
        m_Cond.notify_one();
    }
    T pop()
    {
        std::unique_lock<std::mutex> lock(m_Mutex);
        m_Cond.wait(lock, [this]() { return !m_Queue.empty(); });
        assert(!m_Queue.empty());
        T ret = std::move(*m_Queue.begin());
        m_Queue.erase(m_Queue.begin());
        return ret;
    }
    bool isEmpty()
    {
      std::unique_lock<std::mutex> lock(m_Mutex);
      return m_Queue.empty();
    }
    void clear()
    {
        std::unique_lock<std::mutex> lock(m_Mutex);
        m_Queue.clear();
    }

private:
    std::mutex m_Mutex;
    std::condition_variable m_Cond;
    Container m_Queue;
};

/**
 * Get the size of the element from the data type
 */
inline uint32_t
getElementSize(NvDsInferDataType t)
{
    switch (t)
    {
    case INT64:
        return 8;
    case INT32:
    case FLOAT:
        return 4;
    case HALF:
        return 2;
    case INT8:
    case UINT8:
        return 1;
    default:
        dsInferError(
            "Failed to get element size on Unknown datatype:%d", (int)t);
        return 0;
    }
}

/* Convert between TRT's nvinfer1::Dims representation and DeepStream's
 * NvDsInferDimsCHW/NvDsInferDims representation. */
nvinfer1::Dims ds2TrtDims(const NvDsInferDimsCHW& dims);
nvinfer1::Dims ds2TrtDims(const NvDsInferDims& dims);
NvDsInferDims trt2DsDims(const nvinfer1::Dims& dims);

/* Add batch size to provided dims to get full dims as nvinfer1::Dims. */
nvinfer1::Dims CombineDimsBatch(const NvDsInferDims& dims, int batch);
/* Split full dims provided in the form of nvinfer1::Dims into batch size and
 * layer dims. */
void SplitFullDims(
    const nvinfer1::Dims& fullDims, NvDsInferDims& dims, int& batch);

/* Convert from TRT's nvinfer1::Dims representation to DeepStream's
 * NvDsInferBatchDims representation. */
inline void
convertFullDims(const nvinfer1::Dims& fullDims, NvDsInferBatchDims& batchDims)
{
    SplitFullDims(fullDims, batchDims.dims, batchDims.batchSize);
}

void normalizeDims(NvDsInferDims& dims);

bool hasWildcard(const nvinfer1::Dims& dims);
bool hasWildcard(const NvDsInferDims& dims);

/* Equality / inequality operators implementation for nvinfer1::Dims */
bool operator<=(const nvinfer1::Dims& a, const nvinfer1::Dims& b);
bool operator>(const nvinfer1::Dims& a, const nvinfer1::Dims& b);
bool operator==(const nvinfer1::Dims& a, const nvinfer1::Dims& b);
bool operator!=(const nvinfer1::Dims& a, const nvinfer1::Dims& b);

/* Equality / inequality operators implementation for NvDsInferDims */
bool operator<=(const NvDsInferDims& a, const NvDsInferDims& b);
bool operator>(const NvDsInferDims& a, const NvDsInferDims& b);
bool operator==(const NvDsInferDims& a, const NvDsInferDims& b);
bool operator!=(const NvDsInferDims& a, const NvDsInferDims& b);


bool isValidOutputFormat(const std::string& fmt);
bool isValidOutputDataType(const std::string& dataType);
nvinfer1::DataType str2DataType(const std::string& dataType);
uint32_t str2TensorFormat(const std::string& fmt);

struct BuildParams;
bool validateIOTensorNames(const BuildParams& params,
                           const  nvinfer1::INetworkDefinition& network);
bool isValidDeviceType(const std::string& fmt);
bool isValidPrecisionType(const std::string& dataType);
nvinfer1::DataType str2PrecisionType(const std::string& dataType);
nvinfer1::DeviceType str2DeviceType(const std::string& deviceType);

} // namespace nvdsinfer

#endif
