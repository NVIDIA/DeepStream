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

#include <iostream>
#include <cufft.h>
#include <cufftXt.h>
#include "modules/utility/cuda_common.hpp"
#include "modules/cuDCFv2/cuda_utility.cuh"
#include "modules/cuDCFv2/cuDCF.h"

using std::cout;
using std::endl;

namespace cuDCF
{
#define USE_DETECTOR_ERROR_HANDLING

#define CUDA_BLOCK_SIZE_1D 1024
#define CUDA_BLOCK_SIZE_2D 32
#define CUDCF_SCALE_FACTOR 1

#define FIXED_SCALE_FACTOR 256
#define LEARN_RATE_UPPER_LIMIT 0.5

/**
 * @brief Flatten wrapped feature coordinates into a target-batch index.
 * @param[in] target_ind_i The target index.
 * @param[in] x The horizontal coordinate.
 * @param[in] y The vertical coordinate.
 * @param[in] feat_width The feature-matrix width.
 * @param[in] feat_element_size The number of elements in one feature matrix.
 */
__device__ static inline int getFlattenedIndFromModuloXY(
	const int target_ind_i, const int x, const int y, const int feat_width, const int feat_element_size
)
{
	const int firstIndPerTarget = target_ind_i*feat_element_size;

	return firstIndPerTarget + CUDA_UTIL::modul(y, feat_width)*feat_width + CUDA_UTIL::modul(x, feat_width);
}

/**
 * @brief Reduce one CUDA warp to its maximum value.
 * @param[in,out] reduction_data The shared reduction values.
 * @param[in] i The element or thread index.
 */
__device__ static void warpReduce(volatile float* reduction_data, int i)
{
    reduction_data[i] = reduction_data[i + 32] > reduction_data[i] ? reduction_data[i + 32] : reduction_data[i];
    reduction_data[i] = reduction_data[i + 16] > reduction_data[i] ? reduction_data[i + 16] : reduction_data[i];
    reduction_data[i] = reduction_data[i + 8] > reduction_data[i] ? reduction_data[i + 8] : reduction_data[i];
    reduction_data[i] = reduction_data[i + 4] > reduction_data[i] ? reduction_data[i + 4] : reduction_data[i];
    reduction_data[i] = reduction_data[i + 2] > reduction_data[i] ? reduction_data[i + 2] : reduction_data[i];
    reduction_data[i] = reduction_data[i + 1] > reduction_data[i] ? reduction_data[i + 1] : reduction_data[i];
}

/**
 * @brief Multiply a float feature batch by its masking window.
 * @param[in] src_mat The source feature matrix.
 * @param[in] masking_window The masking window.
 * @param[in] scale The normalization scale factor.
 * @param[out] dst_mat The destination feature matrix.
 * @param[in] batch_element_size The number of elements in the current batch.
 */
static __global__ void cuda_applyMaskingWindow(
	const float *src_mat, const float *masking_window, const float scale, float *dst_mat, const int batch_element_size
)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < batch_element_size)
	{
		dst_mat[i] = (src_mat[i] * masking_window[i]) / scale;
	}
}

/**
 * @brief Normalize high-precision VPI features and apply their masking windows.
 * @param[in] src_mat The source feature matrix.
 * @param[in] masking_window The masking window.
 * @param[in] scale The normalization scale factor.
 * @param[in] srcWidth The source feature width.
 * @param[in] srcPitch The source feature pitch.
 * @param[in] srcHeight The source feature height.
 * @param[in] outBatchingGrid The two-dimensional output batching grid.
 * @param[in] num_of_channels The number of channels in a feature vector.
 * @param[in] num_of_targets The number of targets processed in this batch.
 * @param[in] totalImgNum The total number of feature images.
 * @param[in] useColorNames Whether ColorNames channels are present.
 * @param[in] num_of_colorNames_channels The number of leading ColorNames channels.
 * @param[out] dst_mat The destination feature matrix.
 */
static __global__ void cuda_applyMaskingWindow2D_U16(
    const uint16_t *src_mat, const float *masking_window, const float scale, const uint srcWidth, const uint srcPitch,
    const uint srcHeight, const uint2 outBatchingGrid, const uint num_of_channels, const uint num_of_targets,
    const uint totalImgNum, const bool useColorNames, const uint num_of_colorNames_channels, float *dst_mat
)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    /// The index of each sub feature image.
    const int imgNo = int(i/(srcWidth*srcHeight));

    /// Relative origin within each sub feature image.
    const int relativeOrigin = i - imgNo*srcWidth*srcHeight;
    /// Relative Y within each sub feature image.
    const int yid = relativeOrigin / srcWidth;
    /// Relative X within each sub feature image.
    const int xid = relativeOrigin - yid*srcWidth;

    /// Target index, consistent with dst_mat.
    const int target_ind_i = imgNo / num_of_channels;
	/// Channel index, consistent with dst_mat.
	const uint channelInd = imgNo - target_ind_i*num_of_channels;

    if(imgNo >= totalImgNum)
		return;

    if(xid >= 0 && xid < srcWidth && yid >= 0 && yid < srcHeight)
    {
        int2 bboxLoc;

        bboxLoc.y = int(target_ind_i/outBatchingGrid.x);
        bboxLoc.x = int(target_ind_i%outBatchingGrid.x);

        /// Channel offset.
        const uint32_t src_addr_offset = channelInd*outBatchingGrid.y*srcHeight*srcPitch
            + bboxLoc.y*srcHeight*srcPitch // Y offset in Grid.
            + bboxLoc.x*srcWidth // X offset in Grid (pitch is for 1-ch uint16_t image).
            + yid*srcPitch; // Y offset in img patch.

        float srcVal = (float)src_mat[src_addr_offset + xid];

		/// Normalize to [0, 1).
		/// VPI maps ColorNames from [-1, 1) into uint16_t by shifting one and multiplying by 1<<15.
		/// VPI maps HOG values from [0, 1) into uint16_t by multiplying by 1<<16.
		/// ColorNames and HOG are converted back to their original ranges.
		if(useColorNames && channelInd < num_of_colorNames_channels)
		{
			srcVal /= (float)(1<<15);
			srcVal -= 1.f;
		}
		else
		{
			srcVal /= (float)(1<<16);
		}

		/// Reorder features by target and then channel to align memory access.
		dst_mat[i] = (srcVal * masking_window[i]) / scale;
    }
}

/**
 * @brief Normalize low-precision VPI features and apply their masking windows.
 * @param[in] src_mat The source feature matrix.
 * @param[in] masking_window The masking window.
 * @param[in] scale The normalization scale factor.
 * @param[in] srcWidth The source feature width.
 * @param[in] srcPitch The source feature pitch.
 * @param[in] srcHeight The source feature height.
 * @param[in] outBatchingGrid The two-dimensional output batching grid.
 * @param[in] num_of_channels The number of channels in a feature vector.
 * @param[in] num_of_targets The number of targets processed in this batch.
 * @param[in] totalImgNum The total number of feature images.
 * @param[in] useColorNames Whether ColorNames channels are present.
 * @param[in] num_of_colorNames_channels The number of leading ColorNames channels.
 * @param[out] dst_mat The destination feature matrix.
 */
static __global__ void cuda_applyMaskingWindow2D_U8(
    const uchar *src_mat, const float *masking_window, const float scale, const uint srcWidth, const uint srcPitch,
    const uint srcHeight, const uint2 outBatchingGrid, const uint num_of_channels, const uint num_of_targets,
    const uint totalImgNum, const bool useColorNames, const uint num_of_colorNames_channels, float *dst_mat
)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    /// The index of each sub feature image.
    const int imgNo = int(i/(srcWidth*srcHeight));

    /// Relative origin within each sub feature image.
    const int relativeOrigin = i - imgNo*srcWidth*srcHeight;
    /// Relative Y within each sub feature image.
    const int yid = relativeOrigin / srcWidth;
    /// Relative X within each sub feature image.
    const int xid = relativeOrigin - yid*srcWidth;

    /// Target index, consistent with dst_mat.
    const int target_ind_i = imgNo / num_of_channels;
    /// Channel index, consistent with dst_mat.
    const uint channelInd = imgNo - target_ind_i*num_of_channels;

    if(imgNo >= totalImgNum)
		return;

    if(xid >= 0 && xid < srcWidth && yid >= 0 && yid < srcHeight)
    {
        int2 bboxLoc;

        bboxLoc.y = int(target_ind_i/outBatchingGrid.x);
        bboxLoc.x = int(target_ind_i%outBatchingGrid.x);

        /// Channel offset.
        const uint32_t src_addr_offset = channelInd*outBatchingGrid.y*srcHeight*srcPitch
            + bboxLoc.y*srcHeight*srcPitch // Y offset in Grid.
            + bboxLoc.x*srcWidth // X offset in Grid (pitch is for 1-ch uint8_t image).
            + yid*srcPitch; // Y offset in img patch.

        float srcVal = (float)src_mat[src_addr_offset + xid];

		/// Normalize to [0, 1).
		/// VPI maps ColorNames from [-1, 1) into uint8_t by shifting one and multiplying by 1<<7.
		/// VPI maps HOG values from [0, 1) into uint8_t by multiplying by 1<<8.
		/// ColorNames and HOG are converted back to their original ranges.
		if(useColorNames && channelInd < num_of_colorNames_channels)
		{
			srcVal /= (float)(1<<7);
			srcVal -= 1.f;
		}
		else
		{
			srcVal /= (float)(1<<8);
		}

		/// Reorder features by target and then channel to align memory access.
		dst_mat[i] = (srcVal * masking_window[i]) / scale;
    }
}

/**
 * @brief Prepare the regularized Fourier-domain terms used to learn a DCF filter.
 * @param[in] F The extracted Fourier-domain features.
 * @param[in] Y The desired Fourier-domain response.
 * @param[in] lambda The regularization factor.
 * @param[out] Sxx The regularized feature autocorrelation workspace.
 * @param[out] Sxy The feature-response cross-correlation workspace.
 * @param[in] batch_element_size The number of elements in the current batch.
 */
static __global__ void cuda_matrixPrep4LearnFilter(
	const cuComplex *F, const cuComplex *Y, const float lambda,
	cuComplex *Sxx, cuComplex *Sxy, const int batch_element_size
)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	/// A = Sxy[i].x.
	/// B = Sxy[i].y.
	/// C = Sxx[i].x.
	/// D = Sxx[i].y.
	if(i < batch_element_size)
	{
		const cuComplex F_i = F[i];
		/// A and B.
		const cuComplex Sxy_i = CUDA_UTIL::ComplexMulConjB(F_i, Y[i]);
		/// C and D.
		cuComplex Sxx_i = CUDA_UTIL::ComplexAdd(CUDA_UTIL::ComplexMulConjB(F_i, F_i), lambda);

		const float a = Sxy_i.x;
		const float b = Sxy_i.y;
		const float c = Sxx_i.x;
		const float d = Sxx_i.y;

		float div = c*c + d*d;
		float realPart = a*c + b*d;
		float imPart = b*c - a*d;

		/// Compute Sxy / (Sxx + lambda).
		realPart /= div;
		imPart /= div;

		Sxx_i.x = realPart;
		Sxx_i.y = imPart;

		Sxx[i] = Sxx_i;
		Sxy[i] = Sxy_i;
	}
}

/**
 * @brief Multiply complex arrays elementwise using the conjugate of the second input.
 * @param[in] A The first complex input array.
 * @param[in] B The second complex input array.
 * @param[out] C The output complex array.
 * @param[in] size The number of elements processed.
 */
static __global__ void complexMatMulConjB(const cuComplex *A, const cuComplex *B, cuComplex *C, const int size)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < size)
	{
		C[i] = CUDA_UTIL::ComplexMulConjB(A[i], B[i]);
	}
}

/**
 * @brief Update learned filters with an age-adjusted exponential moving average.
 * @param[in] tracker_age The age of each tracker.
 * @param[in] trackerDoProcessing_d Whether each target should be processed.
 * @param[in,out] result_filter_F The moving-averaged DCF filter in Fourier space.
 * @param[in] learned_filter_F The newly learned DCF filter.
 * @param[in] learn_rate The learning rate for the moving average of DCF filters.
 * @param[in] num_of_channels The number of channels in a feature vector.
 * @param[in] num_of_targets The number of targets processed in this batch.
 * @param[in] feat_element_size The number of elements in one feature matrix.
 * @param[in] size The number of elements processed.
 */
static __global__ void cuda_exponentialMovingAverage(
    const uint32_t *tracker_age, const bool *trackerDoProcessing_d, cuComplex *result_filter_F,
    const cuComplex *learned_filter_F, float learn_rate, const int num_of_channels,
    const int num_of_targets, const int feat_element_size, const int size
)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < size)
	{
		const int img_ind_i = floorf((float)i/feat_element_size);
		/// Target index.
		const int target_ind_i = floorf((float)img_ind_i/num_of_channels);
		/// Channel index.
		const int channel_ind_i = img_ind_i - target_ind_i*num_of_channels;

		if(target_ind_i >= num_of_targets || channel_ind_i >= num_of_channels)
			return;

		cuComplex resultFilter = result_filter_F[i];
		const cuComplex learnedFilter = learned_filter_F[i];
		const uint32_t age = tracker_age[target_ind_i];

		if(trackerDoProcessing_d[target_ind_i] == true)
		{
			if(age <= 1)
			{
				result_filter_F[i] = learnedFilter;
			}
			else
			{
				/// Taper the initially aggressive learning rate as the target ages.
				const float alpha = 1.f/age;
				learn_rate = (1.f-alpha)*learn_rate + alpha*LEARN_RATE_UPPER_LIMIT;

				resultFilter.x = (1. - (double)learn_rate)*(double)resultFilter.x + (double)learn_rate*(double)learnedFilter.x;
				resultFilter.y = (1. - (double)learn_rate)*(double)resultFilter.y + (double)learn_rate*(double)learnedFilter.y;

				result_filter_F[i] = resultFilter;
			}
		}
	}
}

/**
 * @brief Scale inverse-DFT response values.
 * @param[in,out] resp The correlation-response values.
 * @param[in] feat_element_size The number of elements in one feature matrix.
 * @param[in] size The number of elements processed.
 */
static __global__ void cuda_dftScale(float *resp, const int feat_element_size, const int size)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < size)
	{
		resp[i] /= (float)feat_element_size*FIXED_SCALE_FACTOR;
	}
}

/**
 * @brief Compute per-channel filter weights and their per-target sums.
 * @param[in] resp_I The image-space response for each feature channel.
 * @param[out] channel_weights The adaptive filter weights for the channels.
 * @param[out] weight_sum The per-target sum of channel weights.
 * @param[in] feat_element_size The number of elements in one feature matrix.
 * @param[in] num_of_channels The number of channels in a feature vector.
 * @param[in] num_of_targets The number of targets processed in this batch.
 */
static __global__ void getFilterWeights(
	const float *resp_I, float *channel_weights, float *weight_sum,
	const int feat_element_size, const int num_of_channels, const int num_of_targets
)
{
	/// Target index.
	const int target_ind_i = blockIdx.x * blockDim.x + threadIdx.x;

	int img_ind_i = 0;
	float ch_weight_normalized = 0;
	float weightSum = 0;

	if(target_ind_i < num_of_targets)
	{
		#pragma unroll
		for(uint ch=0; ch<num_of_channels; ch++)
		{
			img_ind_i = ch + target_ind_i*num_of_channels;

			/// Normalize the response value because IDFT scaling has not been applied yet.
			ch_weight_normalized = fmax(0.f, resp_I[img_ind_i*feat_element_size]/(float)feat_element_size);

			channel_weights[img_ind_i] = ch_weight_normalized;

			weightSum += ch_weight_normalized;
		}

		weight_sum[target_ind_i] = weightSum;
	}
}

/**
 * @brief Update channel weights with an age-adjusted exponential moving average.
 * @param[in] tracker_age The age of each tracker.
 * @param[in] trackerDoProcessing_d Whether each target should be processed.
 * @param[in,out] channel_weights The adaptive filter weights for the channels.
 * @param[in] new_channel_weights The newly computed channel weights.
 * @param[in] weight_sum The per-target sum of channel weights.
 * @param[in] weights_lr The learning rate for the moving average of channel weights.
 * @param[in] num_of_channels The number of channels in a feature vector.
 * @param[in] num_of_targets The number of targets processed in this batch.
 * @param[in] size The number of elements processed.
 */
static __global__ void computeMovingAverageWeights(
	const uint32_t *tracker_age, const bool *trackerDoProcessing_d,
	float *channel_weights, const float *new_channel_weights, float *weight_sum,
	float weights_lr, const int num_of_channels, const int num_of_targets, const int size
)
{
	const int img_ind_i = blockIdx.x * blockDim.x + threadIdx.x;

	/// Target index.
	const int target_ind_i = floorf((float)img_ind_i / num_of_channels);

	if(img_ind_i < size && target_ind_i < num_of_targets)
	{
		const uint32_t age = tracker_age[target_ind_i];

		/// Taper the initially aggressive learning rate as the target ages.
		const float alpha = 1.f/age;
		weights_lr = (1.f-alpha)*weights_lr + alpha*LEARN_RATE_UPPER_LIMIT;

		if(trackerDoProcessing_d[target_ind_i] == true && weight_sum[target_ind_i] > 0)
		{
			if(age <= 1)
			{
				channel_weights[img_ind_i] = new_channel_weights[img_ind_i];
			}
			else
			{
				channel_weights[img_ind_i] = (1. - (double)weights_lr)*(double)channel_weights[img_ind_i] + (double)weights_lr*(double)new_channel_weights[img_ind_i];
			}
		}
	}
}

/**
 * @brief Accumulate weighted per-channel Fourier responses for each target.
 * @param[in] ftrs_F The extracted features in Fourier space.
 * @param[in] result_filter_F The moving-averaged DCF filter in Fourier space.
 * @param[in] channel_weights The adaptive filter weights for the channels.
 * @param[out] corr_responses The correlation responses.
 * @param[in] feat_element_size The number of elements in one feature matrix.
 * @param[in] num_of_channels The number of channels in a feature vector.
 * @param[in] num_of_targets The number of targets processed in this batch.
 * @param[in] size The number of elements processed.
 */
static __global__ void cuda_calculateResponse(
    const cuComplex *ftrs_F, const cuComplex *result_filter_F, const float *channel_weights, cuComplex *corr_responses,
    const int feat_element_size, const int num_of_channels, const int num_of_targets, const int size
)
{
	const int i = blockIdx.x*blockDim.x + threadIdx.x;
	const int target_ind_i = floorf((float)i/feat_element_size);
	const int relativeInd = i - target_ind_i*feat_element_size;

	if(target_ind_i < num_of_targets && i < size)
	{
		int srcInd;
		int ch_ind_for_target;

		cuComplex corr_resp;
		corr_resp.x = 0;
		corr_resp.y = 0;

		#pragma unroll
		for(int ch=0; ch<num_of_channels; ++ch)
		{
			ch_ind_for_target = target_ind_i*num_of_channels + ch;
			srcInd = ch_ind_for_target*feat_element_size + relativeInd;

			corr_resp = CUDA_UTIL::ComplexAdd(
				corr_resp,
				CUDA_UTIL::ComplexScale(
					CUDA_UTIL::ComplexMulConjB(ftrs_F[srcInd], result_filter_F[srcInd]),
					channel_weights[ch_ind_for_target]*FIXED_SCALE_FACTOR
				)
			);
		}

		corr_responses[i] = corr_resp;
	}
}

/**
 * @brief Find the maximum correlation response for each target.
 * @param[in] resp The correlation-response values.
 * @param[out] maxResp The peak response values.
 * @param[in] feat_width The feature-matrix width.
 * @param[in] feat_element_size The number of elements in one feature matrix.
 * @param[in] num_of_channels The number of channels in a feature vector.
 * @param[in] num_of_targets The number of targets processed in this batch.
 * @param[in] size The number of elements processed.
 */
static __global__ void cuda_findMaxResponseBatched(
    float *resp, float *maxResp, const int feat_width, const int feat_element_size,
    const int num_of_channels, const int num_of_targets, const int size
)
{
    __shared__ float sharedResp[CUDA_BLOCK_SIZE_1D];
    int tid = threadIdx.x;
    const int num_of_blocks_per_target = (feat_element_size + blockDim.x - 1)/blockDim.x;
    /// Target index.
    const int target_i = blockIdx.x / num_of_blocks_per_target;
    /// Index of this target's starting block.
    const int target_start_block = num_of_blocks_per_target * target_i;
    /// Number of elements in resp that are processed by this block.
    const int size_this_block = min(
        int(feat_element_size - (blockIdx.x - target_start_block)*blockDim.x), CUDA_BLOCK_SIZE_1D
    );

    /// Copy resp to shared memory.
    if (tid < size_this_block)
    {
        /// The index in resp for this thread.
        const int relativeInd = feat_element_size*target_i + (blockIdx.x - target_start_block)*blockDim.x + tid;
        sharedResp[tid] = resp[relativeInd];
    }
    else
    {
        sharedResp[tid] = 0.0;
    }
    __syncthreads();

    /// Perform max reduction in each block.
    for(unsigned int stride=blockDim.x/2; stride > 32; stride>>=1)
    {
        if (tid < stride && tid + stride < size_this_block) {
            sharedResp[tid] = sharedResp[tid + stride] > sharedResp[tid] ? sharedResp[tid + stride] : sharedResp[tid];
        }

        __syncthreads();
    }
    if (tid < 32)
    {
        cuDCF::warpReduce(sharedResp, tid);
    }
    __syncthreads();

    /// Find max among blocks per target.
    if (tid == 0)
    {
        CUDA_UTIL::atomicMaxFloat(&maxResp[target_i], sharedResp[0]);
    }
}

/**
 * @brief Find each target response peak and its subpixel displacement.
 * @param[in] resp The correlation-response values.
 * @param[in] maxResp The peak response values.
 * @param[out] targetDisplacement The target displacement from the response center.
 * @param[in] feat_width The feature-matrix width.
 * @param[in] img_element_size The number of elements in a single feature matrix.
 * @param[in] num_of_channels The number of channels in a feature vector.
 * @param[in] num_of_targets The number of targets processed in this batch.
 * @param[in] size The number of elements processed.
 */
static __global__ void cuda_findMaxLocBatched(
    const float *resp, float *maxResp, float2 *targetDisplacement, const int feat_width,
    const int img_element_size, const int num_of_channels, const int num_of_targets, const int size
)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	/// Target index.
	const int target_ind_i = floorf((float)i / img_element_size);

	const float maxResp4Target = maxResp[target_ind_i];
	const float respI = resp[i];

	if(target_ind_i < num_of_targets && i < size)
	{
		/// For the first channel for each target, where the correlation response is stored.
		{
			/// Find the location that has the max resp value and perform a 5-pt interpolation to get the subpixel accuracy.
			if(maxResp4Target == respI && maxResp4Target > 0)
			{
				const int relativeInd = i - target_ind_i*img_element_size;

				float2 dLoc, dLocSubPixel;
				dLoc.y = (float)(relativeInd / feat_width);
				dLoc.x = floorf(relativeInd - dLoc.y*feat_width + 0.5f);
				dLocSubPixel.x = 0;
				dLocSubPixel.y = 0;

				const int stride = 1;

				/// Find the sub-pixel peak to take into account also subpixel accuracy.
				{
					float leftPt = resp[getFlattenedIndFromModuloXY(target_ind_i, dLoc.x-stride, dLoc.y, feat_width, img_element_size)];
					float rightPt = resp[getFlattenedIndFromModuloXY(target_ind_i, dLoc.x+stride, dLoc.y, feat_width, img_element_size)];
					float topPt = resp[getFlattenedIndFromModuloXY(target_ind_i, dLoc.x, dLoc.y-stride, feat_width, img_element_size)];
					float bottomPt = resp[getFlattenedIndFromModuloXY(target_ind_i, dLoc.x, dLoc.y+stride, feat_width, img_element_size)];
					float centerPt = respI;

					float denom = 2*centerPt - rightPt - leftPt;
					dLocSubPixel.x = (denom != 0) ? (0.5f*(rightPt - leftPt)/denom) : 0;

					denom = 2*centerPt - bottomPt - topPt;
					dLocSubPixel.y = (denom != 0) ? (0.5f*(bottomPt - topPt)/denom) : 0;
				}

				dLoc.x += dLocSubPixel.x;
				dLoc.y += dLocSubPixel.y;

				/// Take into account the wrapping.
				if(dLoc.y + 1 > floorf((float)(feat_width*0.5f))) {
					dLoc.y -= feat_width;
				}
				if(dLoc.x + 1 > floorf((float)(feat_width*0.5f))) {
					dLoc.x -= feat_width;
				}

				targetDisplacement[target_ind_i] = dLoc;
			}
		}
	}
}

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
)
{
	uint totalImgNum = num_of_channels*num_of_targets;

	/// Launch one thread per feature element using the ftrs_masked indexing order.
	dim3 dimBlock(CUDA_BLOCK_SIZE_1D);
	dim3 dimGrid((srcWidth*srcHeight*totalImgNum + dimBlock.x - 1) / dimBlock.x);

	/// Apply the masking window for model: H = H.mul(P).
	cuda_applyMaskingWindow2D_U16<<< dimGrid, dimBlock, 0, stream >>>(
		reinterpret_cast<const uint16_t *>(ftrs), reinterpret_cast<const float*>(masking_window),
		1.f, srcWidth, srcPitch, srcHeight, outBatchingGrid, num_of_channels, num_of_targets,
		totalImgNum, useColorNames, num_of_colorNames_channels, reinterpret_cast<float *>(ftrs_masked)
	);
}

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
)
{
	uint totalImgNum = num_of_channels*num_of_targets;

	/// Launch one thread per feature element using the ftrs_masked indexing order.
	dim3 dimBlock(CUDA_BLOCK_SIZE_1D);
	dim3 dimGrid((srcWidth*srcHeight*totalImgNum + dimBlock.x - 1) / dimBlock.x);

	/// Apply the masking window for model: H = H.mul(P).
	cuda_applyMaskingWindow2D_U8<<< dimGrid, dimBlock, 0, stream >>>(
		reinterpret_cast<const uchar *>(ftrs), reinterpret_cast<const float *>(masking_window), 1.f,
		srcWidth, srcPitch, srcHeight, outBatchingGrid, num_of_channels, num_of_targets, totalImgNum,
		useColorNames, num_of_colorNames_channels, reinterpret_cast<float *>(ftrs_masked)
	);
}

/**
 * @brief Sample correlation responses around detector bounding-box centers.
 * @param[in] num_of_targets The number of targets processed in this batch.
 * @param[in] maxTargetsPerStream The maximum number of targets in each stream.
 * @param[in] detectedPts Detector bounding boxes in correlation-response coordinates.
 * @param[in] detectedPts_size The number of detector bounding boxes for each target.
 * @param[in] searchWindowSize The search radius around each detector point.
 * @param[in] corr_responses The correlation responses.
 * @param[in] maxCorrResponse_ma The moving-averaged maximum correlation response.
 * @param[in] feat_element_size The number of elements in one feature matrix.
 * @param[in] batchElementSize The total number of response elements in the batch.
 * @param[in] feat_width The feature-matrix width.
 * @param[out] confidenceAtDetectedPt The tracking confidence at each detector bounding box.
 * @param[out] corrRespAtDetectedPt The correlation response at each detector bounding box.
 */
static __global__ void cuda_getDetectorBboxCorrelationResponses(
    const int num_of_targets, const int maxTargetsPerStream, const detectorBbox_t *detectedPts,
    const uint *detectedPts_size, const int searchWindowSize, const float *corr_responses,
    const float *maxCorrResponse_ma, const int feat_element_size, const int batchElementSize,
    const int feat_width, float *confidenceAtDetectedPt, float *corrRespAtDetectedPt
)
{
	const int trackerInd = blockIdx.x * blockDim.x + threadIdx.x;

	if(trackerInd < num_of_targets)
	{
		const uint dPtSize = detectedPts_size[trackerInd];
		int corrRespInd;
		for(int j=0; j<dPtSize && j<maxTargetsPerStream; j++)
		{
			const uint srcInd = trackerInd*maxTargetsPerStream + j;
			const detectorBbox_t pt = detectedPts[srcInd];

			/// To avoid accessing out-of-range memory.
			if(pt.dBboxInd >= maxTargetsPerStream)
				continue;

			const uint dstInd = trackerInd*maxTargetsPerStream + pt.dBboxInd;
			float maxDetPtResp = 0;
			float corrResp = 0;

			#ifdef USE_DETECTOR_ERROR_HANDLING
				detectorBbox_t pt_near;
				/// Detector bounding boxes may be offset from the target center by several pixels.
				/// Thus, get the max value within 5x5 window.
				for(int m=-searchWindowSize; m<=searchWindowSize; m++)
				{
					pt_near.y = pt.y+m;
					pt_near.y = pt_near.y < 0 ? pt_near.y + feat_width : pt_near.y;
					pt_near.y = pt_near.y >= feat_width ? pt_near.y - feat_width : pt_near.y;

					#pragma unroll
					for(int n=-searchWindowSize; n<=searchWindowSize; n++)
					{
						pt_near.x = pt.x+n;
						pt_near.x = pt_near.x < 0 ? pt_near.x + feat_width : pt_near.x;
						pt_near.x = pt_near.x >= feat_width ? pt_near.x - feat_width : pt_near.x;

						corrRespInd = trackerInd*feat_element_size + pt_near.y*feat_width + pt_near.x;
						corrResp = corr_responses[corrRespInd];

						/// Get the max value within 5x5 window.
						if(maxDetPtResp < corrResp)
						{
							maxDetPtResp = corrResp;
						}
					}
				}

				/// Compute the tracking confidence score at the detector pt.
				confidenceAtDetectedPt[dstInd] = maxDetPtResp / maxCorrResponse_ma[trackerInd];
				corrRespAtDetectedPt[dstInd] = maxDetPtResp;

			#else
				corrRespInd = trackerInd*feat_element_size + pt.y*feat_width + pt.x;

				/// Compute the tracking confidence score at a detector pt.
				confidenceAtDetectedPt[dstInd] = corr_responses[corrRespInd] / maxCorrResponse_ma[trackerInd];
				corrRespAtDetectedPt[dstInd] = corr_responses[corrRespInd];
			#endif
		}
	}
}

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
 * @param[in] feat_element_size The number of elements in one feature matrix.
 * @param[in] feat_width The feature-matrix width.
 * @param[out] confidenceAtDetectedPt The tracking confidence at each detector bbox.
 * @param[out] corrRespAtDetectedPt The correlation response at each detector bbox.
 */
void getDetectorBboxCorrelationResponses(
    const cudaStream_t &stream, const int num_of_targets, const int maxTargetsPerStream,
    const detectorBbox_t *detectedPts, const uint *detectedPts_size, const int detCorrRespSearchWindowSize,
    const float *corr_responses, const float *maxCorrResponse_ma, const int feat_element_size,
    const int feat_width, float *confidenceAtDetectedPt, float *corrRespAtDetectedPt
)
{
	/// Set the number of threads per block.
	const dim3 Threads(CUDA_BLOCK_SIZE_1D);
	const dim3 Blocks((num_of_targets + Threads.x - 1)/Threads.x);

	if(Blocks.x > 0)
	{
		cuda_getDetectorBboxCorrelationResponses<<< Blocks, Threads, 0, stream >>>(
			num_of_targets, maxTargetsPerStream, detectedPts, detectedPts_size,
			detCorrRespSearchWindowSize, corr_responses, maxCorrResponse_ma, feat_element_size,
			num_of_targets*feat_element_size, feat_width, confidenceAtDetectedPt, corrRespAtDetectedPt
		);
	}
}

/**
 * @brief Create a DCF filter and update the learned filter with an exponential moving average.
 * @param[in] stream CUDA stream.
 * @param[in] cufft_handle_C2R The cuFFT C2R handler for batch processing.
 * @param[in] cufft_handle_R2C The cuFFT R2C handler for batch processing.
 * @param[in] ftrs_F The extracted features in Fourier space.
 * @param[in] desiredResp_F The desired response in Fourier space.
 * @param[in] masking_window_model The masking window for focused windowing in image space.
 * @param[in] feat_width The feature-matrix width.
 * @param[in] feat_element_size The number of elements in one feature matrix.
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
    const cuComplex *ftrs_F, const cuComplex *desiredResp_F, const float *masking_window_model, const int feat_width,
    const int feat_element_size, const int batch_element_size, const float lambda, const int num_of_channels,
    const int num_of_targets, cuComplex *work_F, float *work_I, const uint32_t *tracker_age,
    const bool *trackerDoProcessing_d, const float learn_rate, cuComplex *learned_filter_F, cuComplex *result_filter_F
)
{
	const int feat_height = feat_element_size/feat_width;
	/// Half the size.
	const int feat_width_F = feat_width/2 + 1;
	const int feat_element_size_F = feat_height*feat_width_F;
	const int batch_element_size_F = batch_element_size/feat_element_size*feat_element_size_F;

	/// Set the number of threads per block.
	const dim3 Threads(CUDA_BLOCK_SIZE_1D);
	const dim3 Blocks((batch_element_size + Threads.x - 1)/Threads.x);
	const dim3 Blocks_F((batch_element_size_F + Threads.x - 1)/Threads.x);

	cuComplex *Sxx = work_F;
	cuComplex *Sxy = learned_filter_F;

	/// Compute Sxx and Sxy.
	/// The results are stored back at Sxx.
	cuda_matrixPrep4LearnFilter<<< Blocks_F, Threads, 0, stream >>>(
		ftrs_F, desiredResp_F, lambda, Sxx, Sxy, batch_element_size_F
	);

	cuComplex *SxyDivSxx_F = Sxx;
	float *SxyDivSxx_I = work_I;

	/// Inverse DFT.
	CUFFT_CHECK( cufftExecC2R(cufft_handle_C2R, (cufftComplex *)SxyDivSxx_F, (cufftReal *)SxyDivSxx_I) );

	float *maskedFilter_I = work_I;

	/// Normalize the result_filter and then apply the masking window for model: H = H.mul(P).
	cuda_applyMaskingWindow<<< Blocks, Threads, 0, stream >>>(
		reinterpret_cast<const float*>(SxyDivSxx_I), reinterpret_cast<const float *>(masking_window_model),
		(float)feat_element_size/(float)CUDCF_SCALE_FACTOR, maskedFilter_I, batch_element_size
	);

	/// DFT again to get the filter in Fourier domain.
	CUFFT_CHECK( cufftExecR2C(cufft_handle_R2C, (cufftReal *)maskedFilter_I, (cufftComplex *)learned_filter_F) );

	/// Perform moving average.
	cuda_exponentialMovingAverage<<< Blocks_F, Threads, 0, stream >>>(
		tracker_age, trackerDoProcessing_d, result_filter_F, learned_filter_F, learn_rate,
		num_of_channels, num_of_targets, feat_element_size_F, batch_element_size_F
	);
}

/**
 * @brief Crop and rescale segmentation masks into each target feature grid.
 * @param[in] src_mask_frame_batch The source masks from segmenter output.
 * @param[in] src_frame_width The source-mask width.
 * @param[in] src_frame_element_size The number of pixels in one source mask.
 * @param[in] feat_width The feature-matrix width.
 * @param[in] feat_element_size The number of elements in one feature matrix.
 * @param[in] batch_element_size The number of elements in the current batch.
 * @param[in] num_of_targets The number of targets processed in this batch.
 * @param[in] num_of_channels The number of channels in a feature vector.
 * @param[in] src_rois Each target region in segmenter-output coordinates.
 * @param[in] trackerDoProcessing_d Whether each target should be processed.
 * @param[out] dst_mask_patch_batch The cropped and rescaled destination masks.
 */
static __global__ void cuda_createSegmentationFilterMask(
	const float* src_mask_frame_batch, const int src_frame_width, const int src_frame_element_size,
	const int feat_width, const int feat_element_size, const int batch_element_size, const int num_of_targets,
	const int num_of_channels, const float4 src_rois[], const bool *trackerDoProcessing_d, float *dst_mask_patch_batch
)
{
	const int i = blockIdx.x*blockDim.x + threadIdx.x;
	/// Each thread writes one target's one mask pixel in all channels.
	if (i >= batch_element_size)
		return;

	const int target_ind_i = i / feat_element_size;
	if (target_ind_i >= num_of_targets || (!trackerDoProcessing_d[target_ind_i]))
	{
		return;
	}

	/// Index in current target's dst mask.
	const int relativeInd = i - target_ind_i*feat_element_size;

	/// X and y in current target's dst mask.
	const int feat_x = relativeInd % feat_width;
	const int feat_y = relativeInd / feat_width;

	/// Src (mask with decoder output size) roi.
	float4 src_roi = src_rois[target_ind_i];

	/// Location in src whole frame.
	int src_x = int(float(feat_x) / feat_width * src_roi.w + src_roi.x + 0.5);
	int src_y = int(float(feat_y) / feat_width * src_roi.z + src_roi.y + 0.5);

	src_x = min(max(0, src_x), src_frame_width-1);
	src_y = min(max(0, src_y), src_frame_element_size / src_frame_width - 1);

	/// Pixel index in src batch.
	const int src_ind = target_ind_i * src_frame_element_size + src_y * src_frame_width + src_x;

	for (int cid = 0; cid < num_of_channels; cid++)
	{
		dst_mask_patch_batch[target_ind_i * feat_element_size * num_of_channels + feat_element_size * cid + relativeInd] = (src_mask_frame_batch[src_ind]>=0);
	}
}

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
)
{
	/// Set the number of threads per block.
	const dim3 Threads(CUDA_BLOCK_SIZE_1D);
	/// Each thread writes one target's one mask pixel in all channels.
	const dim3 Blocks((num_of_targets * feat_element_size + Threads.x - 1)/Threads.x);

	cuda_createSegmentationFilterMask<<< Blocks, Threads, 0, stream >>>(
		src_mask_frame_batch, src_frame_width, src_frame_element_size, feat_width, feat_element_size,
		batch_element_size, num_of_targets, num_of_channels, src_rois, trackerDoProcessing_d, dst_mask_patch_batch
	);
}

/**
 * @brief Compute and update the channel weights via exponential moving average.
 * @param[in] stream CUDA stream.
 * @param[in] cufft_handle_many The cuFFT handler for batch processing.
 * @param[in] ftrs_F The extracted features in Fourier space.
 * @param[in] feat_element_size The number of elements in one feature matrix.
 * @param[in] feat_width The feature-matrix width.
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
    const int feat_element_size, const int feat_width, const int batch_element_size,
    const int batch_size, const int num_of_channels, const int num_of_targets,
    const cuComplex *new_filter_F, cuComplex *work_F, float *work_I, const uint32_t *tracker_age,
    const bool *trackerDoProcessing_d, const float weights_lr, float *channel_weights, float *weight_sum
)
{
	const int feat_height = feat_element_size/feat_width;
	const int feat_width_F = feat_width/2 + 1;
	const int feat_element_size_F = feat_height*feat_width_F;
	const int batch_element_size_F = batch_element_size/feat_element_size*feat_element_size_F;

	/// Set the number of threads per block.
	const dim3 Threads(CUDA_BLOCK_SIZE_1D);
	const dim3 Blocks((batch_element_size + Threads.x - 1)/Threads.x);
	const dim3 Blocks_F((batch_element_size_F + Threads.x - 1)/Threads.x);

	cuComplex *resp_F = work_F;

	/// Multiply elementwise with a conjugate at half size because the C2R inverse transform recovers the other half.
	complexMatMulConjB<<< Blocks_F, Threads, 0, stream >>>(ftrs_F, new_filter_F, resp_F, batch_element_size_F);

	float *resp_I = work_I;

	/// Inverse DFT.
	/// Scaling is done in the next kernel launch (i.e., at getFilterWeights()).
	CUFFT_CHECK( cufftExecC2R(cufft_handle_many, (cufftComplex *)resp_F, (cufftReal *)resp_I) );

	/// For per-channel processing.
	const dim3 Threads_small(std::min(CUDA_BLOCK_SIZE_1D, (num_of_channels/32+1)*32));
	const dim3 Blocks_small((batch_size + Threads_small.x - 1)/Threads_small.x);

	/// For per-target processing.
	const dim3 Threads_tiny(std::min(CUDA_BLOCK_SIZE_1D, (num_of_targets/32+1)*32));
	const dim3 Blocks_tiny((num_of_targets + Threads_tiny.x - 1)/Threads_tiny.x);

	float *new_channel_weights = reinterpret_cast<float*>(work_F);

	/// Initialize weight_sum before adding stuffs.
	CUDA_CHECK( cudaMemsetAsync(weight_sum, 0, num_of_targets*sizeof(float), stream) );
	CUDA_CHECK( cudaMemsetAsync(new_channel_weights, 0, batch_size*sizeof(float), stream) );

	if(Blocks_tiny.x > 0)
	{
		/// Calculate the filter weights and accumulate the weights per target.
		getFilterWeights<<< Blocks_tiny, Threads_tiny, 0, stream >>>(
			resp_I, new_channel_weights, weight_sum, feat_element_size, num_of_channels, num_of_targets
		);
	}

	if(Blocks_small.x > 0)
	{
		/// Compute exponential moving average of the channel weights.
		computeMovingAverageWeights<<< Blocks_small, Threads_small, 0, stream >>>(
			tracker_age, trackerDoProcessing_d, channel_weights, new_channel_weights,
			weight_sum, weights_lr, num_of_channels, num_of_targets, batch_size
		);
	}
}

/**
 * @brief Compute the moving average of the max correlation response.
 * @param[in] trackerDoProcessing_d Whether to process each target.
 * @param[in] tracker_age The age of each tracker.
 * @param[in] channel_weights The adaptive filter weights for the channels.
 * @param[in] num_of_channels The number of channels in a feature vector.
 * @param[in] num_of_targets The number of targets processed in this batch.
 * @param[in] learn_rate The learning rate for the moving average.
 * @param[in,out] maxCorrResponse_ma The moving-averaged max correlation response.
 */
static __global__ void calcuateExpectedPeakCorrelation(
    const bool *trackerDoProcessing_d, const uint32_t *tracker_age, const float *channel_weights,
    const int num_of_channels, const int num_of_targets, const float learn_rate, float *maxCorrResponse_ma
)
{
	/// Target index.
	const int target_ind_i = blockIdx.x*blockDim.x + threadIdx.x;
	int img_ind_i = 0;

	if(target_ind_i < num_of_targets)
	{
		if(trackerDoProcessing_d[target_ind_i] == true)
		{
			float expectedPeakCorr = 0, ch_weight = 0;

			#pragma unroll
			for(uint ch=0; ch<num_of_channels; ch++)
			{
				img_ind_i = ch + target_ind_i*num_of_channels;
				ch_weight = channel_weights[img_ind_i];
				expectedPeakCorr += ch_weight*ch_weight;
			}

			if(tracker_age[target_ind_i] <= 1)
			{
				maxCorrResponse_ma[target_ind_i] = expectedPeakCorr > 0 ? expectedPeakCorr : 0;
			}
			else
			{
				maxCorrResponse_ma[target_ind_i] = (1.f - (float)learn_rate)*maxCorrResponse_ma[target_ind_i] + learn_rate*expectedPeakCorr;
			}
		}
	}
}

/**
 * @brief Calculate tracking confidence from moving-average and instantaneous peak correlation responses.
 * @param[in] num_of_targets The number of targets processed in this batch.
 * @param[in] maxCorrResponse_ma The moving-averaged max correlation response.
 * @param[in] maxCorrResponse The instantaneous max correlation response.
 * @param[out] trackingConfidence The resulting tracking confidence.
 */
static __global__ void calculateTrackingConfidence(
    const int num_of_targets, const float *maxCorrResponse_ma, const float *maxCorrResponse, float *trackingConfidence
)
{
	/// Target index.
	const int target_ind_i = blockIdx.x*blockDim.x + threadIdx.x;

	if(target_ind_i < num_of_targets)
	{
		const float ma = maxCorrResponse_ma[target_ind_i];

		if(ma > 0)
		{
			trackingConfidence[target_ind_i] = maxCorrResponse[target_ind_i] / ma;
		}
		else
		{
			trackingConfidence[target_ind_i] = 0;
		}
	}
}

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
)
{
	/// For per-target processing.
	const dim3 Threads_tiny(std::min(CUDA_BLOCK_SIZE_1D, (num_of_targets/32+1)*32));
	const dim3 Blocks_tiny((num_of_targets + Threads_tiny.x - 1)/Threads_tiny.x);

	if(Blocks_tiny.x > 0)
	{
		/// Compute the moving average of the max corr resp.
		calcuateExpectedPeakCorrelation<<< Blocks_tiny, Threads_tiny, 0, stream >>>(
			trackerDoProcessing_d, tracker_age, channel_weights,
			num_of_channels, num_of_targets, learn_rate, maxCorrResponse_ma
		);
	}
}

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
)
{
	/// For per-target processing.
	const dim3 Threads_tiny(std::min(CUDA_BLOCK_SIZE_1D, (num_of_targets/32+1)*32));
	const dim3 Blocks_tiny((num_of_targets + Threads_tiny.x - 1)/Threads_tiny.x);

	if(Blocks_tiny.x > 0)
	{
		/// Compute the tracking confidence from the moving average and the instantaneous max corr resp.
		calculateTrackingConfidence<<< Blocks_tiny, Threads_tiny, 0, stream >>>(
			num_of_targets, maxCorrResponse_ma, maxCorrResponse, trackingConfidence
		);
	}
}

/**
 * @brief Given the Fourier-transformed extracted features and the learned filters, compute the correlation response.
 * @param[in] cuStream CUDA stream.
 * @param[in] cufft_handle_many_small The smaller cuFFT handle used for one correlation response per target.
 * @param[in] feat_width The feature-matrix width.
 * @param[in] feat_element_size The number of elements in one feature matrix.
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
    const cudaStream_t &cuStream, const cufftHandle &cufft_handle_many_small, const int feat_width,
    const int feat_element_size, const int batch_element_size, const int num_of_channels, const int num_of_targets,
    const cuComplex *ftrs_F, const cuComplex *result_filter_F, const float *channel_weights,
    cuComplex *work_F, float *corr_responses, float *maxResp, float2 *targetDisplacement
)
{
	/// For a single-channel processing per target.
	const int feat_height = feat_element_size/feat_width;

	/// In Fourier domain, features and filters are conjugate symmetric: elements in the right half are conjugates of the left half.
	/// So feat_width_F is half width. ftrs_F and result_filter_F has size [feat_width_F, feat_height].
	const int feat_width_F = feat_width/2 + 1;
	const int feat_element_size_F = feat_height*feat_width_F;
	const int batch_element_size_F = batch_element_size/feat_element_size*feat_element_size_F;

	const dim3 Threads_small(CUDA_BLOCK_SIZE_1D);
	const dim3 Blocks_small((batch_element_size + Threads_small.x - 1)/Threads_small.x);
	const dim3 Blocks_small_F((batch_element_size_F + Threads_small.x - 1)/Threads_small.x);
	const dim3 Threads_tiny(std::max(std::min(CUDA_BLOCK_SIZE_1D, (feat_element_size + 31)/32*32), 64));
	const dim3 Blocks_tiny((feat_element_size + Threads_tiny.x - 1)/Threads_tiny.x*num_of_targets);

	if(Blocks_small.x > 0 && Blocks_tiny.x > 0)
	{
		/// Fourier response uses a temp work space.
		cuComplex *corr_responses_F = work_F;

		/// Perform element-wise multiplication w/ conjugate on half size of the image due to conjugate symmetry.
		/// Aggregate to a single-channel correlation response per target.
		cuda_calculateResponse<<< Blocks_small_F, Threads_small, 0, cuStream >>>(
			ftrs_F, result_filter_F, channel_weights, corr_responses_F,
			feat_element_size_F, num_of_channels, num_of_targets, batch_element_size_F
		);

		float *corr_responses_I = corr_responses;
		/// Inverse DFT, convert from half size to full img size, save results in corr_responses_I.
		/// Scale the output in the next kernel launch.
		CUFFT_CHECK(
			cufftExecC2R(cufft_handle_many_small, (cufftComplex *)corr_responses_F, (cufftReal *)corr_responses_I)
		);

		/// Below operations are in image domain.
		/// Scale the idft result.
		cuda_dftScale<<< Blocks_small, Threads_small, 0, cuStream >>>(
			corr_responses_I, feat_element_size, batch_element_size
		);

		/// Initialize the max response and the displacement before searching for the peak.
		CUDA_CHECK( cudaMemsetAsync(maxResp, 0, num_of_targets*sizeof(float), cuStream) );
		CUDA_CHECK( cudaMemsetAsync(targetDisplacement, 0, num_of_targets*sizeof(float2), cuStream) );

		/// Find the max response value in batch.
		cuda_findMaxResponseBatched<<< Blocks_tiny, Threads_tiny, 0, cuStream >>>(
			corr_responses_I, maxResp, feat_width, feat_element_size,
			num_of_channels, num_of_targets, batch_element_size
		);

		/// Find the location corresponding to the max location in batch.
		cuda_findMaxLocBatched<<< Blocks_small, Threads_small, 0, cuStream >>>(
			corr_responses_I, maxResp, targetDisplacement, feat_width,
			feat_element_size, num_of_channels, num_of_targets, batch_element_size
		);
	}
}

} // end of namespace cuDCF
