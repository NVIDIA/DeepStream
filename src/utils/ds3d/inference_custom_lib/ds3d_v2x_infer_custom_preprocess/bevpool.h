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

#ifndef _BEV_POOL_H_
#define _BEV_POOL_H_

#include <NvInferPlugin.h>

#include <numeric>
#include <string>
#include <vector>

namespace nvinfer1 {
namespace plugin {

class BEVPoolPlugin : public nvinfer1::IPluginV2DynamicExt {
 public:
  BEVPoolPlugin() = delete;
  BEVPoolPlugin(const void* data, size_t length);
  BEVPoolPlugin(size_t h, size_t w);
  virtual ~BEVPoolPlugin();
  // IPluginV2DynamicExt Methods
  virtual nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
  virtual nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
                                          nvinfer1::IExprBuilder& exprBuilder) noexcept override;
  virtual bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;
  virtual void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs, const nvinfer1::DynamicPluginTensorDesc* out,
                       int nbOutputs) noexcept override;
  virtual size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs, const nvinfer1::PluginTensorDesc* outputs,
                          int nbOutputs) const noexcept override; 
  virtual int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
              const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
  // IPluginV2Ext Methods
  virtual nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;
  // IPluginV2 Methods
  virtual const char* getPluginType() const noexcept override;
  virtual const char* getPluginVersion() const noexcept override;
  virtual int getNbOutputs() const noexcept override;
  virtual int initialize() noexcept override;
  virtual void terminate() noexcept override;
  virtual size_t getSerializationSize() const noexcept override;
  virtual void serialize(void* buffer) const noexcept override;
  virtual void destroy() noexcept override;
  virtual void setPluginNamespace(const char* pluginNamespace) noexcept override;
  virtual const char* getPluginNamespace() const noexcept override;

 private:
  std::string mNamespace;
  size_t bev_h_;
  size_t bev_w_;
  int* num_intervals_ = nullptr;
};

class BEVPoolPluginCreator : public nvinfer1::IPluginCreator {
 public:
  BEVPoolPluginCreator();
  const char* getPluginName() const noexcept override;
  const char* getPluginVersion() const noexcept override;
  const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;
  nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;
  nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;
  void setPluginNamespace(const char* pluginNamespace) noexcept override;
  const char* getPluginNamespace() const noexcept override;

 private:
  nvinfer1::PluginFieldCollection mFC;
  std::vector<nvinfer1::PluginField> mPluginAttributes;
  std::string mNamespace;
};

}  // namespace plugin
}  // namespace nvinfer1

#endif