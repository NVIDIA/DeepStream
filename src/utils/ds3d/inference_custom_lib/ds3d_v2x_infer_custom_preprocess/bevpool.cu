/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <assert.h>
#include <cuda_fp16.h>

#include "bevpool.h"
#include "check.hpp"
#include "tensor.hpp"
#include "timer.hpp"

using namespace nvinfer1;
using nvinfer1::plugin::BEVPoolPlugin;
using nvinfer1::plugin::BEVPoolPluginCreator;

static const char* PLUGIN_VERSION{"1"};
static const char* PLUGIN_NAME{"BEVPooling"};

#define tile_size 10

typedef struct __align__(4) {
  half val[tile_size];
}
combined_half;

static __global__ void bevpool_half_pack10_kernel(const half* camera_feature, const half* depth_weights, unsigned int nchannel,
                                                  const int3* intervals, unsigned int n_intervals, const unsigned int* indices,
                                                  unsigned int out_h, unsigned int out_w, unsigned int ndepth, unsigned int farea,
                                                  half* output_bevfeat, size_t volumn_output) {
  int interval_index = blockIdx.y * blockDim.y + threadIdx.y;
  int feature_block = threadIdx.x * tile_size;

  if (interval_index >= n_intervals) return;
  int3 interval = intervals[interval_index];
  if (interval.z >= volumn_output) return;

  float accumulate[tile_size] = {0.0f};

  // camera_feature: B H W C
  // depth_weights: B D H W
  // indices: B Z H W
  for (int i = interval.x; i < interval.y; i++) {
    int indice = indices[i];
    half depth_weight = __half2float(depth_weights[indice]);
    int ibn = indice / (ndepth * farea);
    int inner_area = indice % farea;
    unsigned int camera_feature_offset = (ibn * farea + inner_area) * nchannel + feature_block;
    combined_half feature = *(combined_half*)(camera_feature + camera_feature_offset);

#pragma unroll
    for (int j = 0; j < tile_size; j++) {
      accumulate[j] = fma(__half2float(feature.val[j]), depth_weight, accumulate[j]);
    }
  }

  // C x H x W
#pragma unroll
  for (int j = 0; j < tile_size; j++) {
    unsigned int output_offset = (feature_block + j) * out_h * out_w + interval.z;
    output_bevfeat[output_offset] = __float2half(accumulate[j]);
  }
}

// Helper function for serializing plugin
template <typename T>
void writeToBuffer(char*& buffer, const T& val) {
  *reinterpret_cast<T*>(buffer) = val;
  buffer += sizeof(T);
}

// Helper function for deserializing plugin
template <typename T>
T readFromBuffer(const char*& buffer) {
  T val = *reinterpret_cast<const T*>(buffer);
  buffer += sizeof(T);
  return val;
}

inline size_t volume(const nvinfer1::Dims& d) {
  return std::accumulate(d.d, d.d + d.nbDims, size_t{1}, std::multiplies<size_t>{});
}

BEVPoolPlugin::~BEVPoolPlugin(){
  checkRuntime(cudaFreeHost(num_intervals_));
}

BEVPoolPlugin::BEVPoolPlugin(size_t h, size_t w) : bev_h_(h), bev_w_(w) {
  checkRuntime(cudaMallocHost(&num_intervals_, sizeof(int)));
}

BEVPoolPlugin::BEVPoolPlugin(const void* data, size_t length) {
  (void)length;
  const char* d = reinterpret_cast<const char*>(data);
  bev_h_ = readFromBuffer<size_t>(d);
  bev_w_ = readFromBuffer<size_t>(d);
  checkRuntime(cudaMallocHost(&num_intervals_, sizeof(int)));
}

nvinfer1::IPluginV2DynamicExt* BEVPoolPlugin::clone() const noexcept {
  auto* plugin = new BEVPoolPlugin(bev_h_, bev_w_);
  plugin->initialize();
  plugin->setPluginNamespace(mNamespace.c_str());
  return plugin;
}

nvinfer1::DimsExprs BEVPoolPlugin::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
                                                       nvinfer1::IExprBuilder& exprBuilder) noexcept {
  (void)nbInputs;
  assert(outputIndex == 0);
  nvinfer1::DimsExprs output;
  output.nbDims = 4;

  // input0: NHWC
  // output: N C BEVH BEVW
  output.d[0] = inputs[0].d[0];
  output.d[1] = inputs[0].d[3];
  output.d[2] = exprBuilder.constant(bev_h_);
  output.d[3] = exprBuilder.constant(bev_w_);
  return output;
}

bool BEVPoolPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs,
                                              int nbOutputs) noexcept {
  assert(nbInputs == 5);
  assert(nbOutputs == 1);
  const PluginTensorDesc& in = inOut[pos];
  if (pos == 0) return (in.type == nvinfer1::DataType::kHALF) && (in.format == TensorFormat::kLINEAR);
  if (pos == 1) return (in.type == nvinfer1::DataType::kHALF) && (in.format == TensorFormat::kLINEAR);
  if (pos == 2) return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
  if (pos == 3) return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
  if (pos == 4) return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
  if (pos == 5) return (in.type == nvinfer1::DataType::kHALF) && (in.format == TensorFormat::kLINEAR);
  return false;
}

void BEVPoolPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                                    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept {
  (void)in;
  (void)nbInputs;
  (void)out;
  (void)nbOutputs;
  return;
}

size_t BEVPoolPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
                                       const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept {
  return 0;
}

int BEVPoolPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
                           const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept {
  auto feats = inputDesc[0];
  auto depth = inputDesc[1];
  // auto intervals = inputDesc[2];
  // auto geom_feats = inputDesc[3];
  auto output = outputDesc[0];
  int H = feats.dims.d[1];
  int W = feats.dims.d[2];
  int C = feats.dims.d[3];
  int D = depth.dims.d[1];

  int batch_size = feats.dims.d[0];
  checkRuntime(cudaMemcpyAsync(num_intervals_, inputs[4], sizeof(int), cudaMemcpyDeviceToHost, stream));
  checkRuntime(cudaStreamSynchronize(stream));
  int num_intervals = *num_intervals_;

  // feats, depth, interval_starts, interval_lengths, geom_feats
  size_t volumn_output = volume(output.dims);
  // int num_intervals = intervals.dims.d[0];
  half* feats_ptr = (half*)inputs[0];
  half* depth_ptr = (half*)inputs[1];
  int3* intervals_ptr = (int3*)inputs[2];
  unsigned int* geom_feats_ptr = (unsigned int*)inputs[3];
  half* output_ptr = (half*)outputs[0];

  int thread_x = C / tile_size;
  int thread_y = 1024 / thread_x;
  dim3 threads(thread_x, thread_y);
  dim3 blocks(1, int((num_intervals + thread_y - 1) / thread_y));
  checkRuntime(cudaMemsetAsync(output_ptr, 0x00, volumn_output * sizeof(half), stream));
  checkKernel(bevpool_half_pack10_kernel<<<blocks, threads, 0, stream>>>(feats_ptr, depth_ptr, C, intervals_ptr, num_intervals,
                                                                         geom_feats_ptr, bev_h_, bev_w_, D, W * H, output_ptr, volumn_output));
  return 0;
}

nvinfer1::DataType BEVPoolPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
                                                    int nbInputs) const noexcept {
  (void)index;
  (void)nbInputs;
  return inputTypes[0];
}

const char* BEVPoolPlugin::getPluginType() const noexcept { return PLUGIN_NAME; }

const char* BEVPoolPlugin::getPluginVersion() const noexcept { return PLUGIN_VERSION; }

int BEVPoolPlugin::getNbOutputs() const noexcept { return 1; }

int BEVPoolPlugin::initialize() noexcept { return 0; }

void BEVPoolPlugin::terminate() noexcept {}

size_t BEVPoolPlugin::getSerializationSize() const noexcept { return 2 * sizeof(size_t); }

void BEVPoolPlugin::serialize(void* buffer) const noexcept {
  char* d = reinterpret_cast<char*>(buffer);
  writeToBuffer<size_t>(d, bev_h_);
  writeToBuffer<size_t>(d, bev_w_);
}

void BEVPoolPlugin::destroy() noexcept { delete this; }

void BEVPoolPlugin::setPluginNamespace(const char* libNamespace) noexcept { mNamespace = libNamespace; }

const char* BEVPoolPlugin::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

BEVPoolPluginCreator::BEVPoolPluginCreator() {
  mPluginAttributes = {PluginField("H", nullptr, PluginFieldType::kINT32, 1),
                       PluginField("W", nullptr, PluginFieldType::kINT32, 1)};
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char* BEVPoolPluginCreator::getPluginName() const noexcept { return PLUGIN_NAME; }

const char* BEVPoolPluginCreator::getPluginVersion() const noexcept { return PLUGIN_VERSION; }

const PluginFieldCollection* BEVPoolPluginCreator::getFieldNames() noexcept { return &mFC; }

IPluginV2* BEVPoolPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept {
  (void)name;
  const PluginField* fields = fc->fields;
  int nbFields = fc->nbFields;
  int target_h = 0;
  int target_w = 0;
  for (int i = 0; i < nbFields; ++i) {
    const char* attr_name = fields[i].name;
    if (strcmp(attr_name, "H") == 0) target_h = *static_cast<const int*>(fields[i].data);
    if (strcmp(attr_name, "W") == 0) target_w = *static_cast<const int*>(fields[i].data);
  }
  auto* plugin = new BEVPoolPlugin(target_h, target_w);
  return plugin;
}

IPluginV2* BEVPoolPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept {
  // This object will be deleted when the network is destroyed,
  (void)name;
  auto* plugin = new BEVPoolPlugin(serialData, serialLength);
  plugin->initialize();
  return plugin;
}

void BEVPoolPluginCreator::setPluginNamespace(const char* libNamespace) noexcept { mNamespace = libNamespace; }

const char* BEVPoolPluginCreator::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

REGISTER_TENSORRT_PLUGIN(BEVPoolPluginCreator);