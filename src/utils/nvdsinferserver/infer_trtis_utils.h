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
 * @file infer_trtis_utils.h
 *
 * @brief Triton Inference Server utilies header file.
 */

#ifndef __NVDSINFER_TRTIS_UTILS_H__
#define __NVDSINFER_TRTIS_UTILS_H__

#include <string>
#include <thread>
#include <vector>
#include <memory>
#include <functional>

#include "infer_common.h"
#include "infer_datatypes.h"
#include "infer_utils.h"

#include "model_config.pb.h"
#include "tritonserver.h"

/**
 * @brief Checks the TRITONSERVER_Error object returned by the trtisExpr.
 * In case of error, performs the specified action after logging the error
 * message.
 */
#define CHECK_TRTIS_ERR_W_ACTION(trtisExpr, action, fmt, ...)              \
    do {                                                                   \
        UniqTritonT<TRITONSERVER_Error> errPtr(                            \
            (trtisExpr), TRITONSERVER_ErrorDelete);                        \
        if (errPtr) {                                                      \
            InferError(                                                    \
                "Triton: " fmt ", triton_err_str:%s, err_msg:%s",          \
                ##__VA_ARGS__, TRITONSERVER_ErrorCodeString(errPtr.get()), \
                TRITONSERVER_ErrorMessage(errPtr.get()));                  \
            action;                                                        \
        }                                                                  \
    } while (0)

/**
 * @brief Checks the TRITONSERVER_Error object returned by the trtisExpr.
 * In case of error, logs the error and returns with NVDSINFER_TRITON_ERROR.
 */
#define RETURN_TRTIS_ERROR(trtisExpr, fmt, ...) \
    CHECK_TRTIS_ERR_W_ACTION(trtisExpr, return NVDSINFER_TRITON_ERROR, fmt, ##__VA_ARGS__)

/**
 * @brief Check the TRITONSERVER_Error object returned by the trtisExpr.
 * In case of error, log the error and continue.
 */
#define CONTINUE_TRTIS_ERROR(trtisExpr, fmt, ...) \
    CHECK_TRTIS_ERR_W_ACTION(trtisExpr, , fmt, ##__VA_ARGS__)

namespace ni = inference;

namespace nvdsinferserver {

/**
 * @brief Maps the TRITONSERVER_DataType to the InferDataType.
 * @param type TRITONSERVER_DataType
 * @return InferDataType corresponding to the input type.
 */
InferDataType DataTypeFromTriton(TRITONSERVER_DataType type);

/**
 * @brief Maps the InferDataType to TRITONSERVER_DataType.
 * @param type InferDataType
 * @return TRITONSERVER_DataType corresponding to the input type.
 */
TRITONSERVER_DataType DataTypeToTriton(InferDataType type);

/**
 * @brief Maps the data type from Triton model configuration proto definition
 * to InferDataType.
 * @param type The protobuf datatype.
 * @return InferDataType corresponding to the protobuf type.
 */
InferDataType DataTypeFromTritonPb(ni::DataType type);

/**
 * @brief Maps the tensor order from Triton model configuration proto
 * definition to the InferTensorOrder type.
 * @param[in] order The protobuf tensor order type.
 * @return  InferTensorOrder type corresponding to the protobuf order.
 */
InferTensorOrder TensorOrderFromTritonPb(ni::ModelInput::Format order);

/**
 * @brief Maps the tensor order from Triton metadata string to
 * the InferTensorOrder type.
 * @param[in] format The Triton metadata tensor order string.
 * @return InferTensorOrder type corresponding to the protobuf order.
 */
InferTensorOrder TensorOrderFromTritonMeta(const std::string& format);

/**
 * @brief Maps the InferMemType to the TRITONSERVER_MemoryType.
 */
TRITONSERVER_MemoryType MemTypeToTriton(InferMemType type);
/**
 * @brief Maps the TRITONSERVER_MemoryType to the InferMemType.
 */
InferMemType MemTypeFromTriton(TRITONSERVER_MemoryType type);

/**
 * @brief Converts the input shape vector from Triton to InferDims type.
 *
 * This functions provides a InferDims structure created based on the
 * input dimensions vector. Any negative dimensions are mapped to -1.
 * The numElements value is updated if there is no dynamic size dimension.
 *
 * @tparam VecDims Input dimensions vector type.
 * @param shape  Input dimensions vector.
 * @return The InferDims structure object created based on the input.
 */
template <typename VecDims>
InferDims
DimsFromTriton(const VecDims& shape)
{
    InferDims ret{0};
    assert((int)shape.size() <= NVDSINFER_MAX_DIMS);
    uint32_t i = 0;
    for (const auto &v : shape) {
        if (v < 0) {
            ret.d[i] = -1;
        } else {
            ret.d[i] = v;
        }
        ++i;
    }
    ret.numDims = i;
    if (!hasWildcard(ret)) {
        normalizeDims(ret);
    }
    return ret;
}

/**
 * @brief Returns a string describing the TRITONSERVER_ModelControlMode:
 * none, explicit or poll.
 */
const char* TritonControlModeToStr(int32_t mode);

} // namespace nvdsinferserver

#endif /* __NVDSINFER_TRTIS_UTILS_H__ */
