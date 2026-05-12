/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file infer_common.h
 *
 * @brief Header file of the common declarations for the nvinferserver library.
 */


#ifndef __NVDSINFERSERVER_COMMON_H__
#define __NVDSINFERSERVER_COMMON_H__

#include <dlfcn.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdarg.h>
#include <string.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <condition_variable>
#include <future>
#include <map>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>

#include <infer_datatypes.h>
#include <infer_defines.h>

namespace nvdsinferserver {

class BaseBatchBuffer;
class BaseBatchArray;
class RefBatchBuffer;

class SysMem;
class CudaStream;
class CudaEvent;
class CudaTensorBuf;

class IPreprocessor;
class IPostprocessor;
class IBackend;
class BasePreprocessor;
class BasePostprocessor;
class BaseBackend;

class TrtISBackend;

/**
 * Common buffer interfaces (internal).
 */
/**@{*/
using SharedBatchBuf = std::shared_ptr<BaseBatchBuffer>;
using UniqBatchBuf = std::unique_ptr<BaseBatchBuffer>;
using SharedOptions = std::shared_ptr<IOptions>;

using SharedBatchArray = std::shared_ptr<BaseBatchArray>;
using UniqBatchArray = std::unique_ptr<BaseBatchArray>;
using SharedRefBatchBuf = std::shared_ptr<RefBatchBuffer>;
/**@}*/

/**
 * Cuda based pointers.
 */
/**@{*/
using SharedCuStream = std::shared_ptr<CudaStream>;
using UniqCuStream = std::unique_ptr<CudaStream>;
using SharedCuEvent = std::shared_ptr<CudaEvent>;
using UniqCuEvent = std::unique_ptr<CudaEvent>;
using SharedSysMem = std::shared_ptr<SysMem>;
using UniqSysMem = std::unique_ptr<SysMem>;
using UniqCudaTensorBuf = std::unique_ptr<CudaTensorBuf>;
using SharedCudaTensorBuf = std::shared_ptr<CudaTensorBuf>;
/**@}*/

/**
 * Processor interfaces.
 */
/**@{*/
using UniqPostprocessor = std::unique_ptr<BasePostprocessor>;
using UniqPreprocessor = std::unique_ptr<BasePreprocessor>;

using UniqTrtISBackend = std::unique_ptr<TrtISBackend>;
/**@}*/

/**
 * Miscellaneous declarations.
 */
/**@{*/
using UniqLock = std::unique_lock<std::mutex>;

class DlLibHandle;
using SharedDllHandle = std::shared_ptr<DlLibHandle>;

template <typename T>
using UniqTritonT = std::unique_ptr<T, std::function<void(T*)>>;

template <typename T>
using ShrTritonT = std::shared_ptr<T>;

class TrtISServer;
class TrtServerAllocator;
using TrtServerPtr = std::shared_ptr<TrtISServer>;

using UniqTritonAllocator = std::unique_ptr<TrtServerAllocator>;
using ShrTritonAllocator = std::shared_ptr<TrtServerAllocator>;
using WeakTritonAllocator = std::weak_ptr<TrtServerAllocator>;

class LstmController;
using UniqLstmController = std::unique_ptr<LstmController>;

class StreamManager;
using UniqStreamManager = std::unique_ptr<StreamManager>;
/**@}*/

/**
 * Extra and custom processor.
 */
/**@{*/
class InferExtraProcessor;
using UniqInferExtraProcessor = std::unique_ptr<InferExtraProcessor>;
class IInferCustomProcessor;
using InferCustomProcessorPtr = std::shared_ptr<IInferCustomProcessor>;
/**@}*/

}  // namespace nvdsinferserver

#endif
