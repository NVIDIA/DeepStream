/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "metadata.hpp"
#include "tensor.hpp"

#include <gst/gst.h>
#include <nvdsmeta.h>
#include <gstnvdsinfer.h>

#include "nvdspreprocess_meta.h"

using namespace deepstream;

TensorOutputUserMetadata::TensorOutputUserMetadata(void *data)
 : UserMetadata(data) {}

TensorOutputUserMetadata::TensorOutputUserMetadata(const UserMetadata& user_meta)
 : UserMetadata(user_meta) {
  if (!data_) {
    return;
  }

  NvDsUserMeta *nvds_user_meta = (NvDsUserMeta *) data_;
  if (nvds_user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META) {
    data_ = nullptr;
  }
 }

TensorOutputUserMetadata::~TensorOutputUserMetadata() {}

unsigned int TensorOutputUserMetadata::uniqueId() const {
  if (!data_) {
    return 0;
  }

  NvDsUserMeta *nvds_user_meta = (NvDsUserMeta *) data_;
  NvDsInferTensorMeta *meta = (NvDsInferTensorMeta *) nvds_user_meta->user_meta_data;
  return meta->unique_id;
}

std::unordered_map<std::string, Tensor*> TensorOutputUserMetadata::getLayers() {
  std::unordered_map<std::string, Tensor*> layers;
  if (!data_) {
    return layers;
  }

  NvDsUserMeta *nvds_user_meta = (NvDsUserMeta *) data_;
  NvDsInferTensorMeta *meta = (NvDsInferTensorMeta *) nvds_user_meta->user_meta_data;
  for (unsigned int i = 0; i < meta->num_output_layers; i++) {
    NvDsInferLayerInfo *info = &meta->output_layers_info[i];
    std::string name = info->layerName;
    unsigned int rank = info->inferDims.numDims;
    NvDsInferDataType nvds_dtype = info->dataType;
    Tensor::DataType dtype = Tensor::DataType::INVALID;
    unsigned int bits = 0;
    int64_t shape[NVDSINFER_MAX_DIMS] = {0};
    int64_t strides[NVDSINFER_MAX_DIMS] = {0};
    void * data_ptr = nullptr;
    Tensor::DeviceType device = Tensor::DeviceType::NONE;
    switch (nvds_dtype) {
      case NvDsInferDataType::FLOAT:
        dtype = Tensor::DataType::FLOAT;
        bits = 32;
        break;
      case NvDsInferDataType::HALF:
        dtype = Tensor::DataType::FLOAT;
        bits = 16;
        break;
      case NvDsInferDataType::INT8:
        dtype = Tensor::DataType::SIGNED;
        bits = 8;
        break;
      case NvDsInferDataType::INT32:
        dtype = Tensor::DataType::SIGNED;
        bits = 32;
        break;
      case NvDsInferDataType::INT64:
        dtype = Tensor::DataType::SIGNED;
        bits = 64;
        break;
      default:
        break;
    }
    for (unsigned int j = 0; j < info->inferDims.numDims; j++) {
      shape[j] = info->inferDims.d[j];
    }
    /* strides for flatten memory layout*/
    strides[rank-1] = 1;
    for (int j = rank-2; j >= 0; j--) {
      strides[j] = strides[j+1] * shape[j+1];
    }
    if (meta->out_buf_ptrs_dev[i]) {
      data_ptr = meta->out_buf_ptrs_dev[i];
      device = Tensor::DeviceType::GPU;
    } else {
      data_ptr = meta->out_buf_ptrs_host[i];
      device = Tensor::DeviceType::CPU;
    }
    Tensor::Context* context = new Tensor::Context;
    layers[name] = new Tensor(rank, dtype, bits, shape, strides, data_ptr, "CHW", meta->gpu_id, device, context);
  }
  return layers;
}
