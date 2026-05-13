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

#include "check.hpp"
#include "launch.cuh"
#include "pillars-scatter.h"
#include "tensor.hpp"

using namespace nvinfer1;
using nvinfer1::plugin::PillarsScatterPlugin;
using nvinfer1::plugin::PillarsScatterPluginCreator;

static const char* PLUGIN_VERSION{"1"};
static const char* PLUGIN_NAME{"PillarsScatter"};

static __global__ void pillars_scatter_kernel(unsigned int N, unsigned int W, unsigned int H, unsigned int C,
                                              const half* pillar_features_data, const uint4* coords_data,
                                              const unsigned int* num_pillars, half* spatial_feature_data) {
  unsigned int idx = cuda_linear_index;
  if (idx >= *num_pillars) return;

  // loc is B, X, Y, Z
  auto loc = coords_data[idx];
  int loc_ib = loc.x;
  int loc_ix = loc.y;
  int loc_iy = loc.z;
  for (unsigned int ic = 0; ic < C; ic += 1) {
    half* pout = (spatial_feature_data + (((loc_ib * C + ic) * W + loc_ix) * H + loc_iy));
    *pout = *(pillar_features_data + idx * C + ic);
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

inline int64_t volume(const nvinfer1::Dims& d) {
  return std::accumulate(d.d, d.d + d.nbDims, int64_t{1}, std::multiplies<int64_t>{});
}

PillarsScatterPlugin::PillarsScatterPlugin(size_t h, size_t w) : bev_h_(h), bev_w_(w) {}

PillarsScatterPlugin::PillarsScatterPlugin(const void* data, size_t length) {
  (void)length;
  const char* d = reinterpret_cast<const char*>(data);
  bev_h_ = readFromBuffer<size_t>(d);
  bev_w_ = readFromBuffer<size_t>(d);
}

nvinfer1::IPluginV2DynamicExt* PillarsScatterPlugin::clone() const noexcept {
  auto* plugin = new PillarsScatterPlugin(bev_h_, bev_w_);
  plugin->initialize();
  plugin->setPluginNamespace(mNamespace.c_str());
  return plugin;
}

nvinfer1::DimsExprs PillarsScatterPlugin::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
                                                              nvinfer1::IExprBuilder& exprBuilder) noexcept {
  (void)nbInputs;
  assert(outputIndex == 0);
  nvinfer1::DimsExprs output;
  output.nbDims = 4;
  output.d[0] = inputs[1].d[0];
  output.d[1] = inputs[0].d[1];
  output.d[2] = exprBuilder.constant(bev_h_);
  output.d[3] = exprBuilder.constant(bev_w_);
  return output;
}

bool PillarsScatterPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs,
                                                     int nbOutputs) noexcept {
  assert(nbInputs == 3);
  assert(nbOutputs == 1);
  const PluginTensorDesc& in = inOut[pos];

  // Feat, Coords, N
  if (pos == 0) return (in.type == nvinfer1::DataType::kHALF) && (in.format == TensorFormat::kLINEAR);
  if (pos == 1) return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
  if (pos == 2) return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
  if (pos == 3) return (in.type == nvinfer1::DataType::kHALF) && (in.format == TensorFormat::kLINEAR);
  return false;
}

void PillarsScatterPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                                           const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept {
  (void)in;
  (void)nbInputs;
  (void)out;
  (void)nbOutputs;
  return;
}

size_t PillarsScatterPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
                                              const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept {
  (void)nbOutputs;
  (void)nbInputs;
  (void)outputs;
  nvinfer1::DataType inputType = inputs[0].type;
  size_t inputVolume = volume(inputs[0].dims);

  if (inputType == nvinfer1::DataType::kHALF) {
    return inputVolume * sizeof(half);
  }

  assert(inputType == nvinfer1::DataType::kFLOAT);
  return inputVolume * sizeof(float);
}

int PillarsScatterPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
                                  const void* const* inputs, void* const* outputs, void* workspace,
                                  cudaStream_t stream) noexcept {
  auto pillar_features_data = static_cast<const half*>(inputs[0]);
  auto coords_data = static_cast<const unsigned int*>(inputs[1]);
  auto num_pillars = static_cast<const unsigned int*>(inputs[2]);
  auto spatial_feature_data = static_cast<half*>(outputs[0]);
  int C = inputDesc[0].dims.d[1];
  int batch = inputDesc[1].dims.d[0];

  checkRuntime(cudaMemsetAsync(spatial_feature_data, 0, batch * C * bev_h_ * bev_w_ * sizeof(half), stream));
  cuda_linear_launch(pillars_scatter_kernel, stream, bev_w_ * bev_h_ * batch, bev_w_, bev_h_, C, pillar_features_data,
                     (const uint4*)coords_data, num_pillars, spatial_feature_data);
  return 0;
}

nvinfer1::DataType PillarsScatterPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
                                                           int nbInputs) const noexcept {
  (void)index;
  (void)nbInputs;
  return inputTypes[0];
}

const char* PillarsScatterPlugin::getPluginType() const noexcept { return PLUGIN_NAME; }

const char* PillarsScatterPlugin::getPluginVersion() const noexcept { return PLUGIN_VERSION; }

int PillarsScatterPlugin::getNbOutputs() const noexcept { return 1; }

int PillarsScatterPlugin::initialize() noexcept { return 0; }

void PillarsScatterPlugin::terminate() noexcept {}

size_t PillarsScatterPlugin::getSerializationSize() const noexcept { return 3 * sizeof(size_t); }

void PillarsScatterPlugin::serialize(void* buffer) const noexcept {
  char* d = reinterpret_cast<char*>(buffer);
  writeToBuffer<size_t>(d, bev_h_);
  writeToBuffer<size_t>(d, bev_w_);
}

void PillarsScatterPlugin::destroy() noexcept { delete this; }

void PillarsScatterPlugin::setPluginNamespace(const char* libNamespace) noexcept { mNamespace = libNamespace; }

const char* PillarsScatterPlugin::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

PillarsScatterPluginCreator::PillarsScatterPluginCreator() {
  mPluginAttributes = {
      PluginField("H", nullptr, PluginFieldType::kINT32, 1),
      PluginField("W", nullptr, PluginFieldType::kINT32, 1)
  };
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char* PillarsScatterPluginCreator::getPluginName() const noexcept { return PLUGIN_NAME; }

const char* PillarsScatterPluginCreator::getPluginVersion() const noexcept { return PLUGIN_VERSION; }

const PluginFieldCollection* PillarsScatterPluginCreator::getFieldNames() noexcept { return &mFC; }

IPluginV2* PillarsScatterPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept {
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
  return new PillarsScatterPlugin(target_h, target_w);
}

IPluginV2* PillarsScatterPluginCreator::deserializePlugin(const char* name, const void* serialData,
                                                          size_t serialLength) noexcept {
  // This object will be deleted when the network is destroyed,
  (void)name;
  auto* plugin = new PillarsScatterPlugin(serialData, serialLength);
  plugin->initialize();
  return plugin;
}

void PillarsScatterPluginCreator::setPluginNamespace(const char* libNamespace) noexcept { mNamespace = libNamespace; }

const char* PillarsScatterPluginCreator::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

REGISTER_TENSORRT_PLUGIN(PillarsScatterPluginCreator);