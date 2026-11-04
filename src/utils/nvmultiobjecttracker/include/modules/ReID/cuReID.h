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

#ifndef CUREIDLIB_CUREID_H_
#define CUREIDLIB_CUREID_H_

namespace cuReID
{
/**
 * @brief Given ReID dot product score between each detection and each target's feature gallery, find nearest neighbor (highest score) for each detection and target pair.
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
);

/**
 * @brief Copy detection's reid features to target's feature gallery based on data association matched pairs.
 * @param[in] stream CUDA stream used for the operation.
 * @param[in] numOfTargets Number of tracked targets to process.
 * @param[in] reidFeatureSize Number of values in each ReID feature vector.
 * @param[in] detInds Matched detection index for each target.
 * @param[in] reidFeatureDet Detection feature vectors.
 * @param[in] reidFeatureGalleryInds Destination gallery index for each target.
 * @param[in] trackerDoProcessing Processing-enable flag for each target.
 * @param[in,out] reidFeatureGallery Gallery receiving the matched detection features.
 */
void copyDetReIDFeatureToGallery(
    const cudaStream_t &stream, const int numOfTargets, const int reidFeatureSize,
    const int* detInds, const float* reidFeatureDet, const int* reidFeatureGalleryInds,
    const int* trackerDoProcessing, float* reidFeatureGallery
);

/**
 * @brief Perform l2 normalization for ReID features.
 * @param[in] stream CUDA stream used for the operation.
 * @param[in] numOfFeatures Number of feature vectors to normalize.
 * @param[in] reidFeatureSize Number of values in each ReID feature vector.
 * @param[in] blockSize Number of CUDA threads requested per feature vector.
 * @param[in,out] reidFeatureDet Feature vectors normalized in place.
 */
void l2NormalizeReIDFeatures(
    const cudaStream_t &stream, const int numOfFeatures,
    const int reidFeatureSize, const int blockSize, float* reidFeatureDet
);

/**
 * @brief Crop and resize segmentation mask into each ReID input patch's size.
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
);
} // end of namespace cuReID

#endif // CUREIDLIB_CUREID_H_
