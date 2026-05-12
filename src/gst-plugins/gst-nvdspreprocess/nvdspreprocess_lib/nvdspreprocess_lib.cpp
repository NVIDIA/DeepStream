/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <iostream>
#include <fstream>
#include <thread>
#include <string.h>
#include <queue>
#include <mutex>
#include <stdexcept>
#include <condition_variable>

#include "nvbufsurface.h"
#include "nvbufsurftransform.h"

#include "nvdspreprocess_lib.h"
#include "nvdspreprocess_impl.h"

struct CustomCtx
{
  /** Custom initialization parameters */
  CustomInitParams initParams;
  /** Custom mean subtraction and normalization parameters */
  CustomMeanSubandNormParams custom_mean_norm_params;
  /** unique pointer to tensor_impl class instance */
  std::unique_ptr <NvDsPreProcessTensorImpl> tensor_impl;
};

/* Get the absolute path of a file mentioned in the config given a
 * file path absolute/relative to the config file. */
static gboolean
get_absolute_file_path (
    const gchar * cfg_file_path, const gchar * file_path,
    char *abs_path_str)
{
  gchar abs_cfg_path[_PATH_MAX + 1];
  gchar abs_real_file_path[_PATH_MAX + 1];
  gchar *abs_file_path;
  gchar *delim;

  /* Absolute path. No need to resolve further. */
  if (file_path[0] == '/') {
    /* Check if the file exists, return error if not. */
    if (!realpath (file_path, abs_real_file_path)) {
      return FALSE;
    }
    g_strlcpy (abs_path_str, abs_real_file_path, _PATH_MAX);
    return TRUE;
  }

  /* Get the absolute path of the config file. */
  if (!realpath (cfg_file_path, abs_cfg_path)) {
    return FALSE;
  }

  /* Remove the file name from the absolute path to get the directory of the
   * config file. */
  delim = g_strrstr (abs_cfg_path, "/");
  *(delim + 1) = '\0';

  /* Get the absolute file path from the config file's directory path and
   * relative file path. */
  abs_file_path = g_strconcat (abs_cfg_path, file_path, nullptr);

  /* Resolve the path.*/
  if (realpath (abs_file_path, abs_real_file_path) == nullptr) {
    /* Ignore error if file does not exist and use the unresolved path. */
    if (errno == ENOENT)
      g_strlcpy (abs_real_file_path, abs_file_path, _PATH_MAX);
    else {
      g_free (abs_file_path);
      return FALSE;
    }
  }

  g_free (abs_file_path);

  g_strlcpy (abs_path_str, abs_real_file_path, _PATH_MAX);
  return TRUE;
}

NvDsPreProcessStatus
CustomTensorPreparation(CustomCtx *ctx, NvDsPreProcessBatch *batch, NvDsPreProcessCustomBuf *&buf,
                        CustomTensorParams &tensorParam, NvDsPreProcessAcquirer *acquirer)
{
  NvDsPreProcessStatus status = NVDSPREPROCESS_TENSOR_NOT_READY;

  /** acquire a buffer from tensor pool */
  buf = acquirer->acquire();

  /** Prepare Tensor */
  status = ctx->tensor_impl->prepare_tensor(batch, tensorParam, buf->memory_ptr);
  if (status != NVDSPREPROCESS_SUCCESS) {
    printf ("Custom Lib: Tensor Preparation failed\n");
    acquirer->release(buf);
  }

  /** synchronize cuda stream */
  status = ctx->tensor_impl->syncStream();
  if (status != NVDSPREPROCESS_SUCCESS) {
    printf ("Custom Lib: Cuda Stream Synchronization failed\n");
    acquirer->release(buf);
  }

  tensorParam.params.network_input_shape[0] = (int)batch->units.size();

  return status;
}

NvDsPreProcessStatus
CustomTransformation(NvBufSurface *in_surf, NvBufSurface *out_surf, CustomTransformParams &params)
{
  NvBufSurfTransform_Error err;

  err = NvBufSurfTransformSetSessionParams(&params.transform_config_params);
  if (err != NvBufSurfTransformError_Success)
  {
      printf("NvBufSurfTransformSetSessionParams failed with error %d\n", err);
      return NVDSPREPROCESS_CUSTOM_TRANSFORMATION_FAILED;
  }

  /* Batched tranformation. */
  err = NvBufSurfTransform(in_surf, out_surf, &params.transform_params);

  if (err != NvBufSurfTransformError_Success)
  {
      printf("NvBufSurfTransform failed with error %d\n", err);
      return NVDSPREPROCESS_CUSTOM_TRANSFORMATION_FAILED;
  }

  return NVDSPREPROCESS_SUCCESS;
}

NvDsPreProcessStatus
CustomAsyncTransformation(NvBufSurface *in_surf, NvBufSurface *out_surf, CustomTransformParams &params)
{
  NvBufSurfTransform_Error err;

  err = NvBufSurfTransformSetSessionParams(&params.transform_config_params);
  if (err != NvBufSurfTransformError_Success)
  {
      printf("NvBufSurfTransformSetSessionParams failed with error %d\n", err);
      return NVDSPREPROCESS_CUSTOM_TRANSFORMATION_FAILED;
  }

  /* Async Batched tranformation. */
  err = NvBufSurfTransformAsync(in_surf, out_surf, &params.transform_params, &params.sync_obj);

  if (err != NvBufSurfTransformError_Success)
  {
      printf("NvBufSurfTransform failed with error %d\n", err);
      return NVDSPREPROCESS_CUSTOM_TRANSFORMATION_FAILED;
  }

  return NVDSPREPROCESS_SUCCESS;
}

CustomCtx *initLib(CustomInitParams initparams)
{
  auto ctx = std::make_unique<CustomCtx>();
  NvDsPreProcessStatus status;

  ctx->custom_mean_norm_params.pixel_normalization_factor =
      std::stof(initparams.user_configs[NVDSPREPROCESS_USER_CONFIGS_PIXEL_NORMALIZATION_FACTOR]);

  if (!initparams.user_configs[NVDSPREPROCESS_USER_CONFIGS_MEAN_FILE].empty()) {
    char abs_path[_PATH_MAX] = {0};
    if (!get_absolute_file_path (initparams.config_file_path,
          initparams.user_configs[NVDSPREPROCESS_USER_CONFIGS_MEAN_FILE].c_str(), abs_path)) {
      printf("Error: Could not parse mean image file path\n");
      return nullptr;
    }
    if (!ctx->custom_mean_norm_params.meanImageFilePath.empty()) {
      ctx->custom_mean_norm_params.meanImageFilePath.clear();
    }
    ctx->custom_mean_norm_params.meanImageFilePath.append(abs_path);
  }

  std::string offsets_str = initparams.user_configs[NVDSPREPROCESS_USER_CONFIGS_OFFSETS];

  if (!offsets_str.empty()) {
    std::string delimiter = ";";
    size_t pos = 0;
    std::string token;

    while ((pos = offsets_str.find(delimiter)) != std::string::npos) {
        token = offsets_str.substr(0, pos);
        ctx->custom_mean_norm_params.offsets.push_back(std::stof(token));
        offsets_str.erase(0, pos + delimiter.length());
    }
    ctx->custom_mean_norm_params.offsets.push_back(std::stof(offsets_str));

    printf("Using offsets : %f,%f,%f\n", ctx->custom_mean_norm_params.offsets[0],
          ctx->custom_mean_norm_params.offsets[1], ctx->custom_mean_norm_params.offsets[2]);
  }

  status = normalization_mean_subtraction_impl_initialize(&ctx->custom_mean_norm_params,
          &initparams.tensor_params, ctx->tensor_impl, initparams.unique_id);

  if (status != NVDSPREPROCESS_SUCCESS) {
    printf("normalization_mean_subtraction_impl_initialize failed\n");
    return nullptr;
  }

  ctx->initParams = initparams;

  return ctx.release();
}

void deInitLib(CustomCtx *ctx)
{
  delete ctx;
}
