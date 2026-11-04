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
#include "modules/utility/cuda_common.hpp"
#include "modules/ReID/cuReID.h"

namespace cuReID
{
#define CUDA_BLOCK_SIZE_1D 1024
#define DET_THREAD_NUM_PER_TARGET 4

/**
 * @brief Reduce partial feature sums within a CUDA warp.
 * @param[in,out] sdata Shared-memory values reduced in place.
 * @param[in] i Lane index within the warp.
 */
__device__ static void warpReduce(volatile float *sdata, int i)
{
    sdata[i] += sdata[i + 32];
    sdata[i] += sdata[i + 16];
    sdata[i] += sdata[i + 8];
    sdata[i] += sdata[i + 4];
    sdata[i] += sdata[i + 2];
    sdata[i] += sdata[i + 1];
}

/**
 * @brief Find the highest gallery similarity for each target and detection.
 * @param[in] numOfTargets Number of targets to compare.
 * @param[in] maxDetsPerStream Maximum number of detections in one stream.
 * @param[in] divide Number of threads assigned to each target.
 * @param[in] historySize Maximum number of gallery features retained per target.
 * @param[in] maxReidProductSize Maximum number of similarity values per stream.
 * @param[in] targetReidFeatureInds Gallery indices retained for each target.
 * @param[in] reidProduct Detection-to-gallery similarity values.
 * @param[in] reidProductSizeStreams Gallery and detection counts for each stream.
 * @param[in] streamInds Stream index for each target.
 * @param[in] featureIndSize Number of retained gallery features for each target.
 * @param[out] confidenceReidDetTargets Highest similarity for each target and detection.
 * @param[out] nearestIndReidDetTargets Nearest gallery index for each target and detection.
 */
static __global__ void cuda_getNearestReIDSimilarity(
    const int numOfTargets, const int maxDetsPerStream, const int divide,
    const int historySize, const int maxReidProductSize, const int* targetReidFeatureInds,
    const float* reidProduct, const int2* reidProductSizeStreams, const int* streamInds,
    const int* featureIndSize, float* confidenceReidDetTargets, int* nearestIndReidDetTargets
)
{
    const int threadInd = blockIdx.x * blockDim.x + threadIdx.x;
    const int trackerInd = threadInd / divide, streamInd = streamInds[trackerInd];
    if (trackerInd >= numOfTargets)
	{
        return;
    }

    /// Size of this stream's reid product, x: number of gallery features, y: number of detections.
    int2 reidProductSize = reidProductSizeStreams[streamInd];

    /// No detection or target reid features in this stream.
    if (reidProductSize.x == 0 || reidProductSize.y == 0)
    {
        return;
    }

    /// Determine detection range for this thread.
    int didStart = (reidProductSize.y + divide) / divide * (threadInd % divide);
    int didEnd = min(didStart + (reidProductSize.y + divide) / divide, reidProductSize.y);

    for (int did=didStart; did<didEnd && did<maxDetsPerStream; did++)
    {
        float conf = 0.0;
        int nearestInd = -1;
        // Iterate all its history features and find nearest neighbor (max product).
        #pragma unroll
        for (int hid=0; hid<historySize && hid<featureIndSize[trackerInd]; hid++)
        {
            int featGalleryInd = targetReidFeatureInds[trackerInd * historySize + hid];
            float newConf = reidProduct[streamInd * maxReidProductSize + did * reidProductSize.x + featGalleryInd];
            if (newConf > conf)
            {
                conf = newConf;
                nearestInd = featGalleryInd;
            }
        }
        confidenceReidDetTargets[trackerInd * maxDetsPerStream + did] = conf;
        nearestIndReidDetTargets[trackerInd * maxDetsPerStream + did] = nearestInd;
    }
}

/**
 * @brief Copy matched detection features into the target gallery.
 * @param[in] numOfTargets Number of tracked targets to process.
 * @param[in] reidFeatureSize Number of values in each ReID feature vector.
 * @param[in] detInds Matched detection index for each target.
 * @param[in] reidFeatureDet Detection feature vectors.
 * @param[in] galleryInds Destination gallery index for each target.
 * @param[in] trackerDoProcessing Processing-enable flag for each target.
 * @param[in,out] reidFeatureGallery Gallery receiving the matched detection features.
 */
static __global__ void cuda_copyDetReIDFeatureToGallery(
    const int numOfTargets, const int reidFeatureSize, const int* detInds, const float* reidFeatureDet,
    const int* galleryInds, const int* trackerDoProcessing, float* reidFeatureGallery
)
{
    const int trackerInd = blockIdx.x * blockDim.x + threadIdx.x;
    if (trackerInd >= numOfTargets)
	{
        return;
    }

    /// Find source index in detection features and destination index in feature gallery.
    int detInd = detInds[trackerInd], galleryInd = galleryInds[trackerInd];

    if (trackerDoProcessing[trackerInd] == 0 || detInd < 0 || galleryInd < 0)
    {
        return;
    }

	int galleryOffset = galleryInd * reidFeatureSize, detOffset = detInd * reidFeatureSize;

    #pragma unroll
    for (int i=0; i<reidFeatureSize; i++)
    {
        reidFeatureGallery[galleryOffset + i] = reidFeatureDet[detOffset + i];
    }
}

/**
 * @brief Apply segmentation masks to cropped ReID patches.
 * @param[in] src_mask_frame_batch Source segmentation masks for all targets.
 * @param[in] src_frame_width Source mask width.
 * @param[in] src_frame_element_size Number of elements in the source mask.
 * @param[in] img_width Width of each destination image patch in pixels.
 * @param[in] img_element_size Number of pixels in each destination image patch.
 * @param[in] num_of_bboxes Number of image patches to process.
 * @param[in] target_inds Source-mask target index for each image patch.
 * @param[in] src_rois Source region of interest for each image patch.
 * @param[in,out] img_patch_batch Image patches masked in place.
 */
static __global__ void cuda_applySegmentaitonToImagePatches(
    const float* src_mask_frame_batch, const int src_frame_width, const int src_frame_element_size,
    const int img_width, const int img_element_size, const int num_of_bboxes,
    const int target_inds[], const float4 src_rois[], float *img_patch_batch
)
{
	const int i = blockIdx.x*blockDim.x + threadIdx.x;

    /// Each thread writes one target's one mask pixel.
	const int bbox_ind_i = i / img_element_size;
	if (bbox_ind_i >= num_of_bboxes)
	{
		return;
	}

	/// Index in current target's dst mask.
	const int relativeInd = i - bbox_ind_i*img_element_size;

	/// X and y in current target's dst mask.
	const int img_x = relativeInd % img_width;
	const int img_y = relativeInd / img_width;
    const int img_height = img_element_size / img_width;

	/// Src (mask with decoder output size) roi.
	float4 src_roi = src_rois[bbox_ind_i];

	/// Location in src whole frame.
	int src_x = int(float(img_x) / img_width * src_roi.w + src_roi.x + 0.5);
	int src_y = int(float(img_y) / img_height * src_roi.z + src_roi.y + 0.5);

	src_x = min(max(0, src_x), src_frame_width-1);
	src_y = min(max(0, src_y), src_frame_element_size / src_frame_width - 1);

	/// Pixel index in src batch.
    int target_ind = target_inds[bbox_ind_i];
    if (target_ind < 0)
        return;

	const int src_ind = target_ind * src_frame_element_size + src_y * src_frame_width + src_x;
	img_patch_batch[i] *= (src_mask_frame_batch[src_ind]>=0);
}

/**
 * @brief Normalize ReID feature vectors on the GPU.
 * @param[in] numOfFeatures Number of feature vectors to normalize.
 * @param[in] reidFeatureSize Number of values in each ReID feature vector.
 * @param[in,out] reidFeatureDet Feature vectors normalized in place.
 */
static __global__ void cuda_l2NormalizeReIDFeatures(
    const int numOfFeatures, const int reidFeatureSize, float* reidFeatureDet
)
{
    const int tid = threadIdx.x;
    const int vec_i = blockIdx.x;

    __shared__ float sharedFeat[CUDA_BLOCK_SIZE_1D];

    sharedFeat[tid] = 0;
    for (int offset=tid; offset < reidFeatureSize; offset += blockDim.x)
    {
        float ele = reidFeatureDet[reidFeatureSize * vec_i + offset];
        sharedFeat[tid] += ele * ele;
    }
    __syncthreads();

    /// Perform reduction sum in each block.
    for(unsigned int stride=blockDim.x/2; stride > 32; stride>>=1)
    {
        if (tid < stride)
        {
            sharedFeat[tid] += sharedFeat[tid + stride];
        }
        __syncthreads();
    }
    if (tid < 32)
    {
        cuReID::warpReduce(sharedFeat, tid);
    }
    __syncthreads();

    float norm = sqrt(sharedFeat[0]);

    #pragma unroll
    for (int offset=tid; offset < reidFeatureSize; offset += CUDA_BLOCK_SIZE_1D)
    {
        reidFeatureDet[reidFeatureSize * vec_i + offset] /= norm;
    }
}

/**
 * @brief Launch nearest-gallery ReID similarity computation.
 * @param[in] stream CUDA stream used for the operation.
 * @param[in] numOfTargets Actual number of targets now.
 * @param[in] maxDetsPerStream Maximum number of detections in one stream.
 * @param[in] historySize Maximum number of gallery features retained per target.
 * @param[in] maxReidProductSize Max number of reid dot product elements per stream := maxNumofDet x maxFeatGallerySize.
 * @param[in] targetReidFeatureInds Feature gallery indices for all targets := numOfTargets x historySize.
 * @param[in] reidProduct Reid dot product elements for all streams := numOfStreams x numOfDet x featGallerySize.
 * @param[in] reidProductSizePerStream Actual number of features in gallery and detection features for each stream. x: featureGallerySizeThisStream, y: numOfDetThisStream.
 * @param[in] streamInds Stream index for each target to find detected features and feature gallery for that stream.
 * @param[in] featureIndSize Number of retained gallery features for each target.
 * @param[out] confidenceReidDetTargets Similarity score between each det and target after finding nearest neighbor.
 * @param[out] nearestIndReidDetTargets Gallery index of the nearest history feature for each target and detection.
 */
void getNearestReIDSimilarity(
    const cudaStream_t &stream, const int numOfTargets, const int maxDetsPerStream,
    const int historySize, const int maxReidProductSize, const int* targetReidFeatureInds,
    const float* reidProduct, const int2* reidProductSizePerStream, const int* streamInds,
    const int* featureIndSize, float* confidenceReidDetTargets, int* nearestIndReidDetTargets
)
{
    /// DET_THREAD_NUM_PER_TARGET threads per detection-target pair.
    const dim3 Threads( std::min(CUDA_BLOCK_SIZE_1D, (numOfTargets * DET_THREAD_NUM_PER_TARGET / 32 + 1) * 32) );
	const dim3 Blocks((numOfTargets * DET_THREAD_NUM_PER_TARGET + Threads.x - 1)/Threads.x);

	if(Blocks.x > 0)
	{
		cuda_getNearestReIDSimilarity<<< Blocks, Threads, 0, stream >>>(
			numOfTargets, maxDetsPerStream, DET_THREAD_NUM_PER_TARGET, historySize,
			maxReidProductSize, targetReidFeatureInds, reidProduct, reidProductSizePerStream,
			streamInds, featureIndSize, confidenceReidDetTargets, nearestIndReidDetTargets
		);
	}
}

/**
 * @brief Copy matched detection features into the ReID gallery.
 * @param[in] stream CUDA stream used for the operation.
 * @param[in] numOfTargets Number of tracked targets to process.
 * @param[in] reidFeatureSize Number of values in each ReID feature vector.
 * @param[in] detInds Matched detection index for each target.
 * @param[in] reidFeatureDet Detection feature vectors.
 * @param[in] galleryInds Destination gallery index for each target.
 * @param[in] trackerDoProcessing Processing-enable flag for each target.
 * @param[in,out] reidFeatureGallery Gallery receiving the matched detection features.
 */
void copyDetReIDFeatureToGallery(
    const cudaStream_t &stream, const int numOfTargets, const int reidFeatureSize, const int* detInds,
    const float* reidFeatureDet, const int* galleryInds, const int* trackerDoProcessing, float* reidFeatureGallery
)
{
    /// One thread per target.
    const dim3 Threads( std::min(CUDA_BLOCK_SIZE_1D, (numOfTargets/32+1)*32) );
	const dim3 Blocks((numOfTargets + Threads.x - 1)/Threads.x);

	if(Blocks.x > 0)
	{
		cuda_copyDetReIDFeatureToGallery<<< Blocks, Threads, 0, stream >>>(
			numOfTargets, reidFeatureSize, detInds, reidFeatureDet,
			galleryInds, trackerDoProcessing, reidFeatureGallery
		);
	}
}

/**
 * @brief Normalize a batch of ReID feature vectors.
 * @param[in] stream CUDA stream used for the operation.
 * @param[in] numOfFeatures Number of feature vectors to normalize.
 * @param[in] reidFeatureSize Number of values in each ReID feature vector.
 * @param[in] blockSize Number of CUDA threads requested per feature vector.
 * @param[in,out] reidFeatureDet Feature vectors normalized in place.
 */
void l2NormalizeReIDFeatures(
    const cudaStream_t &stream, const int numOfFeatures,
    const int reidFeatureSize, const int blockSize, float* reidFeatureDet
)
{
    const dim3 Threads( std::max(std::min(CUDA_BLOCK_SIZE_1D, blockSize), 64) );
    const dim3 Blocks(numOfFeatures);
    if (Blocks.x > 0)
    {
        cuda_l2NormalizeReIDFeatures<<< Blocks, Threads, 0, stream >>>(numOfFeatures, reidFeatureSize, reidFeatureDet);
    }
}

/**
 * @brief Apply target masks to cropped ReID image patches.
 * @param[in] stream CUDA stream used for the operation.
 * @param[in] srcMaskFrameBatch Source segmentation masks for all targets.
 * @param[in] srcFrameWidth Width of each source mask in pixels.
 * @param[in] srcFrameElementSize Number of elements in each source mask.
 * @param[in] imgWidth Width of each destination image patch in pixels.
 * @param[in] imgElementSize Number of pixels in each destination image patch.
 * @param[in] numOfBboxes Number of image patches to process.
 * @param[in] targetInds Source-mask target index for each image patch.
 * @param[in] srcRois Source region of interest for each image patch.
 * @param[in,out] imgPatchBatch Image patches masked in place.
 */
void applySegmentaitonToImagePatches(
    const cudaStream_t &stream, const float *srcMaskFrameBatch, const int srcFrameWidth,
    const int srcFrameElementSize, const int imgWidth, const int imgElementSize,
    const int numOfBboxes, const int *targetInds, const float4 *srcRois, float *imgPatchBatch
)
{
    /// Use one CUDA block of threads for each feature vector.
    const dim3 Threads(CUDA_BLOCK_SIZE_1D);
	/// Each thread writes one target's one mask pixel in all channels.
	const dim3 Blocks((numOfBboxes * imgElementSize + Threads.x - 1)/Threads.x);

	cuda_applySegmentaitonToImagePatches<<< Blocks, Threads, 0, stream >>>(
		srcMaskFrameBatch, srcFrameWidth, srcFrameElementSize, imgWidth,
		imgElementSize, numOfBboxes, targetInds, srcRois, imgPatchBatch
	);
}
} // end of namespace cuReID
