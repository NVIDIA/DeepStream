/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef CUDCFLIB_CUDCF_H_
#define CUDCFLIB_CUDCF_H_

#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <cuComplex.h>
#include "modules/NvMultiObjectTracker/localizer.hpp"

typedef float ftype1;
typedef float2 ftype2;

/** @brief Detector bbox in correlation-response coordinates. */
typedef struct
{
    int x, y; //!< Coordinates in the correlation-response matrix.
    uint dBboxInd; //!< Index to the corresponding detector bbox.
} detectorBbox_t;

namespace cuDCF
{
/**
 * @brief Read the correlation response near each detector bbox and write the confidence and response.
 * @param[in] stream CUDA stream.
 * @param[in] num_of_targets The number of targets processed in this batch.
 * @param[in] maxTargetsPerStream The maximum number of targets in each stream.
 * @param[in] detectedPts Detector bboxes in correlation-response coordinates.
 * @param[in] detectedPts_size The number of detector bboxes for each target.
 * @param[in] detCorrRespSearchWindowSize The search radius around each detector point in the correlation response.
 * @param[in] corr_responses The correlation response after applying the filter.
 * @param[in] maxCorrResponse_ma The moving-averaged max correlation response.
 * @param[in] img_element_size The number of elements in one feature matrix.
 * @param[in] img_width The feature-matrix width.
 * @param[out] confidenceAtDetectedPt The tracking confidence at each detector bbox.
 * @param[out] corrRespAtDetectedPt The correlation response at each detector bbox.
 */
void getDetectorBboxCorrelationResponses(
    const cudaStream_t &stream, const int num_of_targets, const int maxTargetsPerStream,
    const detectorBbox_t *detectedPts, const uint *detectedPts_size, const int detCorrRespSearchWindowSize,
    const float *corr_responses, const float *maxCorrResponse_ma, const int img_element_size,
    const int img_width, float *confidenceAtDetectedPt, float *corrRespAtDetectedPt
);

/**
 * @brief Multiply a 2D high-precision feature by a masking window elementwise.
 * @param[in] stream CUDA stream.
 * @param[in] ftrs The input high-precision feature.
 * @param[in] masking_window The filter multiplied with the feature.
 * @param[in] srcWidth The width of the input feature.
 * @param[in] srcPitch The pitch of the input feature.
 * @param[in] srcHeight The height of the input feature.
 * @param[in] outBatchingGrid The 2D size of the VPI feature grid.
 * @param[in] num_of_channels The number of channels in a feature vector.
 * @param[in] num_of_targets The number of targets processed in this batch.
 * @param[in] useColorNames Whether ColorNames features are included.
 * @param[in] num_of_colorNames_channels The number of leading ColorNames channels.
 * @param[out] ftrs_masked The feature after the filter is applied.
 */
void applyFilter2D_U16(
    const cudaStream_t &stream, const uint16_t *ftrs, const float *masking_window, const uint srcWidth,
    const uint srcPitch, const uint srcHeight, const uint2 outBatchingGrid, const uint num_of_channels,
    const uint num_of_targets, const bool useColorNames, const uint num_of_colorNames_channels, float *ftrs_masked
);

/**
 * @brief Multiply a 2D low-precision feature by a masking window elementwise.
 * @param[in] stream CUDA stream.
 * @param[in] ftrs The input low-precision feature.
 * @param[in] masking_window The filter multiplied with the feature.
 * @param[in] srcWidth The width of the input feature.
 * @param[in] srcPitch The pitch of the input feature.
 * @param[in] srcHeight The height of the input feature.
 * @param[in] outBatchingGrid The 2D size of the VPI feature grid.
 * @param[in] num_of_channels The number of channels in a feature vector.
 * @param[in] num_of_targets The number of targets processed in this batch.
 * @param[in] useColorNames Whether ColorNames features are included.
 * @param[in] num_of_colorNames_channels The number of leading ColorNames channels.
 * @param[out] ftrs_masked The feature after the filter is applied.
 */
void applyFilter2D_U8(
    const cudaStream_t &stream, const uchar *ftrs, const float *masking_window, const uint srcWidth,
    const uint srcPitch, const uint srcHeight, const uint2 outBatchingGrid, const uint num_of_channels,
    const uint num_of_targets, const bool useColorNames, const uint num_of_colorNames_channels, float *ftrs_masked
);

/**
 * @brief Create a DCF filter and update the learned filter with an exponential moving average.
 * @param[in] stream CUDA stream.
 * @param[in] cufft_handle_C2R The cuFFT C2R handler for batch processing.
 * @param[in] cufft_handle_R2C The cuFFT R2C handler for batch processing.
 * @param[in] ftrs_F The extracted features in Fourier space.
 * @param[in] desiredResp_F The desired response in Fourier space.
 * @param[in] masking_window_model The masking window for focused windowing in image space.
 * @param[in] img_width The feature-matrix width.
 * @param[in] img_element_size The number of elements in one feature matrix.
 * @param[in] batch_element_size The number of feature-vector elements for all current trackers.
 * @param[in] lambda The regularization constant used in the ridge-regression solver.
 * @param[in] num_of_channels The number of channels in a feature vector.
 * @param[in] num_of_targets The number of targets processed in this batch.
 * @param[out] work_F A temporary workspace for complex arrays.
 * @param[out] work_I A temporary workspace for float arrays.
 * @param[in] tracker_age The age of each tracker.
 * @param[in] trackerDoProcessing_d Whether to process each target.
 * @param[in] learn_rate The learning rate for the moving average of DCF filters.
 * @param[out] learned_filter_F The newly created DCF filter.
 * @param[in,out] result_filter_F The moving-averaged DCF filter in Fourier space.
 */
void createDcfFilter(
    const cudaStream_t &stream, const cufftHandle &cufft_handle_C2R, const cufftHandle &cufft_handle_R2C,
    const cuComplex *ftrs_F, const cuComplex *desiredResp_F, const float *masking_window_model, const int img_width,
    const int img_element_size, const int batch_element_size, const float lambda, const int num_of_channels,
    const int num_of_targets, cuComplex *work_F, float *work_I, const uint32_t *tracker_age,
    const bool *trackerDoProcessing_d, const float learn_rate, cuComplex *learned_filter_F, cuComplex *result_filter_F
);

/**
 * @brief Crop and rescale each target's segmentation mask to the visual tracker region size.
 * @param[in] stream CUDA stream.
 * @param[in] src_mask_frame_batch The source mask from the segmenter output.
 * @param[in] src_frame_width The width of the source mask.
 * @param[in] src_frame_element_size The number of pixels in the source mask.
 * @param[in] feat_width The destination filter width.
 * @param[in] feat_element_size The number of elements in the destination filter.
 * @param[in] batch_element_size The number of filter elements across all channels and supported targets.
 * @param[in] num_of_targets The number of targets processed in this batch.
 * @param[in] num_of_channels The number of channels in a feature vector.
 * @param[in] src_rois Each target's region in segmenter-output coordinates.
 * @param[in] trackerDoProcessing_d Whether to process each target.
 * @param[out] dst_mask_patch_batch The cropped and rescaled destination masks.
 */
void createSegmentationFilterMask(
    const cudaStream_t &stream, const float *src_mask_frame_batch, const int src_frame_width,
    const int src_frame_element_size, const int feat_width, const int feat_element_size,
    const int batch_element_size, const int num_of_targets, const int num_of_channels,
    const float4 *src_rois, const bool *trackerDoProcessing_d, float *dst_mask_patch_batch
);

/**
 * @brief Compute and update the channel weights via exponential moving average.
 * @param[in] stream CUDA stream.
 * @param[in] cufft_handle_many The cuFFT handler for batch processing.
 * @param[in] ftrs_F The extracted features in Fourier space.
 * @param[in] img_element_size The number of elements in one feature matrix.
 * @param[in] img_width The feature-matrix width.
 * @param[in] batch_element_size The number of feature-vector elements for all current trackers.
 * @param[in] batch_size The number of feature matrices across channels and target trackers.
 * @param[in] num_of_channels The number of channels in a feature vector.
 * @param[in] num_of_targets The number of targets processed in this batch.
 * @param[in] new_filter_F The newly created DCF filter in Fourier space.
 * @param[out] work_F A temporary workspace for complex arrays.
 * @param[out] work_I A temporary workspace for float arrays.
 * @param[in] tracker_age The age of each tracker.
 * @param[in] trackerDoProcessing_d Whether to process each target.
 * @param[in] weights_lr The learning rate for the moving average of channel weights.
 * @param[in,out] channel_weights The adaptive filter weights for the channels.
 * @param[out] weight_sum Filter-weight sums used as workspace during normalization.
 */
void computeChannelWeights(
    const cudaStream_t &stream, const cufftHandle &cufft_handle_many, const cuComplex *ftrs_F,
    const int img_element_size, const int img_width, const int batch_element_size,
    const int batch_size, const int num_of_channels, const int num_of_targets,
    const cuComplex *new_filter_F, cuComplex *work_F, float *work_I, const uint32_t *tracker_age,
    const bool *trackerDoProcessing_d, const float weights_lr, float *channel_weights, float *weight_sum
);

/**
 * @brief Compute the moving average of the max correlation response.
 * @param[in] stream CUDA stream.
 * @param[in] num_of_channels The number of channels in a feature vector.
 * @param[in] num_of_targets The number of targets processed in this batch.
 * @param[in] learn_rate The learning rate for the moving average.
 * @param[in] channel_weights The adaptive filter weights for the channels.
 * @param[in] tracker_age The age of each tracker.
 * @param[in] trackerDoProcessing_d Whether to process each target.
 * @param[in,out] maxCorrResponse_ma The moving-averaged max correlation response.
 */
void calcuateExpectedPeakCorrelation(
    const cudaStream_t &stream, const int num_of_channels, const int num_of_targets,
    const float learn_rate, const float *channel_weights, const uint32_t *tracker_age,
    const bool *trackerDoProcessing_d, float *maxCorrResponse_ma
);

/**
 * @brief Given the Fourier-transformed extracted features and the learned filters, compute the correlation response.
 * @param[in] stream CUDA stream.
 * @param[in] cufft_handle_many_small The smaller cuFFT handle used for one correlation response per target.
 * @param[in] img_width The feature-matrix width.
 * @param[in] img_element_size The number of elements in one feature matrix.
 * @param[in] batch_element_size The number of feature-vector elements for all current trackers.
 * @param[in] num_of_channels The number of channels in a feature vector.
 * @param[in] num_of_targets The number of targets processed in this batch.
 * @param[in] ftrs_F The extracted features in Fourier space.
 * @param[in] result_filter_F The moving-averaged DCF filter in Fourier space.
 * @param[in] channel_weights The adaptive filter weights for the channels.
 * @param[out] work_F A temporary workspace for complex arrays.
 * @param[out] corr_responses The resulting correlation response in image space.
 * @param[out] maxResp The peak response values.
 * @param[out] targetDisplacement The target displacement from the center to the peak correlation response.
 */
void calculateResponse(
    const cudaStream_t &stream, const cufftHandle &cufft_handle_many_small, const int img_width,
    const int img_element_size, const int batch_element_size, const int num_of_channels, const int num_of_targets,
    const cuComplex *ftrs_F, const cuComplex *result_filter_F, const float *channel_weights,
    cuComplex *work_F, float *corr_responses, float *maxResp, float2 *targetDisplacement
);

/**
 * @brief Calculate tracking confidence from moving-average and instantaneous peak correlation responses.
 * @param[in] stream CUDA stream.
 * @param[in] num_of_targets The number of targets processed in this batch.
 * @param[in] maxCorrResponse_ma The moving-averaged max correlation response.
 * @param[in] maxCorrResponse The instantaneous max correlation response.
 * @param[out] trackingConfidence The resulting tracking confidence.
 */
void calculateTrackingConfidence(
    const cudaStream_t &stream, const int num_of_targets, const float *maxCorrResponse_ma,
    const float *maxCorrResponse, float *trackingConfidence
);
} // end of namespace cuDCF

#endif // CUDCFLIB_CUDCF_H_
