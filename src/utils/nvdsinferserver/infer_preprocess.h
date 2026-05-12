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
 * @file infer_preprocess.h
 *
 * @brief Header file for the preprocessor classes for scaling and cropping.
 *
 * This is a header file for pre-processing CUDA kernels with normalization and
 * mean subtraction required by the nvdsinfererver library.
 */

#ifndef __NVDSINFERSERVER_PREPROCESS_H__
#define __NVDSINFERSERVER_PREPROCESS_H__

#include <stdarg.h>
#include <condition_variable>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <queue>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "infer_cuda_utils.h"
#include "infer_datatypes.h"
#include "infer_iprocess.h"
#include "infer_preprocess_kernel.h"
#include "infer_utils.h"

#include "nvbufsurface.h"
#include "nvbufsurftransform.h"

namespace nvdsinferserver {

/**
 * @brief Preprocessor for scaling and normalization of the input and conversion
 * to network media format.
 */
class NetworkPreprocessor : public BasePreprocessor {
public:
    NetworkPreprocessor(const NvDsInferNetworkInfo& info,
        InferMediaFormat networkFormat, InferDataType dt, int maxBatchSize);
    virtual ~NetworkPreprocessor() = default;

    bool setScaleOffsets(float scale, const std::vector<float>& offsets = {});
    bool setMeanFile(const std::string& file);
    void setNetworkTensorOrder(InferTensorOrder order) {
        m_NetworkTensorOrder = order;
    }
    void setPoolSize(int size) { m_PoolSize = size; }
    int poolSize() const { return m_PoolSize; }
    void setNetworkTensorName(std::string name) { m_TensorName = name; }
    const BatchSurfaceInfo& getDstSurfaceInfo() const { return m_DstInfo; }

    NvDsInferStatus allocateResource(const std::vector<int>& devIds) override;
    NvDsInferStatus syncStream();

private:
    SharedBatchBuf requestOutBuffer(SharedBatchBuf& inBuf) override;
    NvDsInferStatus transformImpl(SharedBatchBuf& src, SharedBatchBuf& dst,
        SharedCuStream& mainStream) override;

    NvDsInferStatus cudaTransform(void* srcBuf, const BatchSurfaceInfo& srcInfo,
        InferMediaFormat srcFormat, InferDataType srcDType, void* dstBuf,
        const BatchSurfaceInfo& dstInfo, InferDataType dstDType, int batchSize,
        int devId);

protected:
    NvDsInferStatus readMeanImageFile();
    DISABLE_CLASS_COPY(NetworkPreprocessor);

private:
    NvDsInferNetworkInfo m_NetworkInfo = {0};
    InferTensorOrder m_NetworkTensorOrder = InferTensorOrder::kLinear;
    /** Input format for the network. */
    InferMediaFormat m_NetworkFormat = InferMediaFormat::kRGB;
    float m_Scale = 1.0f;
    std::vector<float> m_ChannelMeans; // same as channels
    std::string m_MeanFile;

    std::string m_TensorName;

    /** Hold output buffers */
    SharedBufPool<UniqCudaTensorBuf> m_BufPool;
    int m_MaxBatchSize = 1;
    int m_PoolSize = 1;
    InferDataType m_DstDataType = InferDataType::kFp32;

    BatchSurfaceInfo m_DstInfo = {0};

    std::unique_ptr<CudaStream> m_PreProcessStream;
    std::unique_ptr<CudaDeviceMem> m_MeanDataBuffer;
};

class SurfaceBuffer;

/**
 * @brief Preprocessor for cropping, scaling and padding the inference input
 * to required height, width.
 */
class CropSurfaceConverter : public BasePreprocessor {
public:
    CropSurfaceConverter(int32_t convertPoolSize)
        : m_ConvertPoolSize(convertPoolSize) {}
    ~CropSurfaceConverter() override;
    void setParams(
        int outW, int outH, InferMediaFormat outFormat, int maxBatchSize);
    void setMaintainAspectRatio(bool enable) { m_MaintainAspectRatio = enable; }
    void setSymmetricPadding(bool enable) { m_SymmetricPadding = enable; }
    void setScalingHW(NvBufSurfTransform_Compute compute_hw) {
        m_ComputeHW = compute_hw;
    }
    void setScalingFilter(NvBufSurfTransform_Inter filter) {
        m_TransformParam.transform_filter = filter;
    }

private:
    NvDsInferStatus allocateResource(const std::vector<int>& devIds) override;
    SharedBatchBuf requestOutBuffer(SharedBatchBuf& inBuf);
    NvDsInferStatus transformImpl(
        SharedBatchBuf& src, SharedBatchBuf& dst, SharedCuStream& mainStream);

private:
    NvDsInferStatus resizeBatch(SharedBatchBuf& src, SharedBatchBuf& dst);

private:
    int32_t m_ConvertPoolSize = 4;

    int32_t m_DstWidth = 0;
    int32_t m_DstHeight = 0;
    InferMediaFormat m_DstFormat = InferMediaFormat::kRGB;
    uint32_t m_MaxBatchSize = 0;
    bool m_MaintainAspectRatio = false;
    bool m_SymmetricPadding = false;
    NvBufSurfTransform_Compute m_ComputeHW = NvBufSurfTransformCompute_Default;

    SharedBufPool<std::unique_ptr<SurfaceBuffer>> m_ConvertPool;
    SharedCuStream m_ConverStream;
    NvBufSurfTransformParams m_TransformParam = {
        0, NvBufSurfTransform_None, NvBufSurfTransformInter_Default};
};

} // namespace nvdsinferserver

#endif /* __NVDSINFER_CONVERSION_H__ */
