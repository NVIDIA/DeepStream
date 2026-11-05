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

#ifndef CUDCFLIB_FEATUREEXTRACTOR_HPP_
#define CUDCFLIB_FEATUREEXTRACTOR_HPP_

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuComplex.h>

#if defined(VIZ_DEBUG)
#include <opencv2/core.hpp>
#endif

// VPI.
#include "vpi/Stream.h"
#include "vpi/Image.h"
#include "vpi/experimental/ColorNames.h"
#include "vpi/experimental/HOG.h"
#include "vpi/CUDAInterop.h"
#include "modules/utility/cuda_common.hpp"
#include "modules/cuDCFv2/cuDCFFrameTransformTexture.hpp"

/** @brief Extracts batched ColorNames and HOG features with VPI. */
class FeatureExtractor
{
public:
	/** @brief Construct an uninitialized feature extractor. */
	FeatureExtractor();

	/** @brief Release feature-extraction resources. */
	virtual ~FeatureExtractor();

	enum FEATURE_TYPES : uint
	{
		COLOR_NAMES = 0x1, //!< ColorNames feature.
		HOG = 0x10 //!< HOG feature.
	};
	static const uint NUM_CHANNELS_COLORNAMES = 10; //!< Number of channels in the ColorNames feature.
	static const uint NUM_CHANNELS_HOG_SENSITIVE = 18; //!< Number of channels in the contrast-sensitive HOG feature.
	static const uint NUM_CHANNELS_HOG_INSENSITIVE = NUM_CHANNELS_HOG_SENSITIVE/2; //!< Number of channels in the contrast-insensitive HOG feature.

	/** @brief Report whether GPU feature extraction is initialized. */
	bool isInitialized() { return m_initGpu; }

	/** @brief Return the number of channels in each extracted feature vector. */
	uint getNumChannels() { return m_numFeatureChannelsOut;}

	/** @brief Allocate the device memory for the LUT on the device memory and copy the LUT from host to device. */
	void initialize_ColorNames();

	/**
	 * @brief Extract batched visual features from source-image regions with VPI.
	 * @param[in] nvFramesInBatch The input images from which features are extracted.
	 * @param[in] roisInBatch The ROIs for each stream, as pairs of bbox index and rectangle.
	 * @param[in] lastBboxInd The last bbox index in the batch.
	 * @param[in] doUpdateImages Whether to store input images on the first extraction call for a batch.
	 * @return True when feature extraction succeeds.
	 */
	bool extractFeatures_VPI(
		const std::map<NvMOTStreamId, NvMOTFrame*> &nvFramesInBatch,
		const std::map<NvMOTStreamId, std::vector<std::pair<uint, NvMotDataTypes::Rect2f> > > &roisInBatch,
		const uint lastBboxInd, const bool doUpdateImages
	);

	/**
	 * @brief Initialize feature extraction and allocate its host and device memory.
	 * @param[in] debugVerboseLevel The debug level that selects which intermediate information is printed.
	 * @param[in] maxStreamNum The maximum number of streams to extract.
	 * @param[in] maxBufSurfAddrSize The maximum number of possible surface buffer addresses.
	 * @param[in] maxfeatureImgNum The maximum number of feature images.
	 * @param[in] srcImgSize The input image size.
	 * @param[in] srcImgPitch The input image pitch.
	 * @param[in] colorFormatType The color format of the input image.
	 * @param[in] templateImgSize The size of the tracker templates.
	 * @param[in] hogCellSize The cell size of the HOG feature.
	 * @param[in] featureTypes The feature types to extract, such as ColorNames or HOG.
	 * @param[in] vpiDevType The VPI backend, CUDA or CPU.
	 * @param[in] vpiImgOutputType The VPI image format of the extracted features.
	 * @return True when initialization completes.
	 */
	bool initialize(
		const int debugVerboseLevel, const int maxStreamNum, const int maxBufSurfAddrSize, const int maxfeatureImgNum,
		const NvMotDataTypes::Size srcImgSize, const uint srcImgPitch, const NvBufSurfaceColorFormat colorFormatType,
		const NvMotDataTypes::Size templateImgSize, const int hogCellSize, const uint featureTypes,
		const VPIBackend vpiDevType = VPI_BACKEND_CUDA, const VPIImageFormat vpiImgOutputType = VPI_IMAGE_FORMAT_U16
	);

	/**
	 * @brief Get the metadata for the extracted features by VPI.
	 * @param[out] width The feature width.
	 * @param[out] height The feature height, including the NV12 plane layout when applicable.
	 * @param[out] outBatchingGrid The dimensions of the VPI feature grid.
	 * @param[out] channels The number of channels in each image patch.
	 * @param[out] pitch The pitch of a source image patch.
	 * @param[out] imgNum The number of image patches in the batch.
	 * @return The device pointer to the feature-vector data.
	 */
	uchar *getOutputFeaturePtrs_VPI(
		uint &width, uint &height, uint2 &outBatchingGrid, uint &channels, uint &pitch, uint &imgNum
	);

	#if defined(VIZ_DEBUG)
		/**
		 * @brief Get VPI features copied to host for debugging or visualization.
		 * @return The feature vectors copied to host memory.
		 */
		std::vector<std::vector<cv::Mat> > getExtractedFeaturesAtHost_VPI() const;

		/**
		 * @brief Get image patches at host for debugging or visualization.
		 * @return The image patches copied to host memory.
		 */
		std::map<NvMOTStreamId, std::map<uint, cv::Mat > > getSrcImgPatches()
		{
			assert(m_subImageCropper2 != nullptr && m_subImageCropper2->isInitialized());

			if(m_colorFormatType == NvBufSurfaceColorFormat::NVBUF_COLOR_FORMAT_NV12)
			{
				return m_subImageCropper2->getImagePatches2Host_OCV_mosaic();
			}
			else
			{
				return m_subImageCropper2->getImagePatches2Host_OCV_batch();
			}
		}
	#endif

	/** @brief Return the number of source image patches in the current batch. */
	uint getSrcImgNum() { return m_srcImgPatchNum; }

	/** @brief Return the two-dimensional output batching grid. */
	uint2 getOutBatchingGrid() const { return m_outBatchingGrid; }

	/** @brief Return the number of channels in each output feature vector. */
	uint getOutputFeatureChannelSize() const { return m_numFeatureChannelsOut; }

	/** @brief Report whether VPI produces 16-bit feature values. */
	bool isVpiOutputInHighPrecision() const { return m_vpiImgOutputType == VPI_IMAGE_FORMAT_U16; }

	#ifdef VIZ_DEBUG
		/** @brief Return source images cached in host memory for visualization. */
		const std::map<NvMOTStreamId, cv::Mat> &getSourceImagesAtHost() const
		{
			return m_subImageCropper2->getSourceImagesAtHost();
		}
	#endif

protected:

	/** @brief Initialize and allocate the input/output memories for VPI. */
	void initialize_VPI();

	const uint m_featureTypeHog = VPI_HOG_CONTRAST_SENSITIVE; //!< Selected VPI HOG feature type.
	//!< Contrast-sensitive HOG adds numOrientations channels, contrast-insensitive HOG adds half as many, and texture HOG adds four.
	const uint m_numOfChannelsByHog = FeatureExtractor::NUM_CHANNELS_HOG_SENSITIVE; //!< Number of output feature channels extracted by HOG.
	const uint m_numOfChannelsByColorNames = FeatureExtractor::NUM_CHANNELS_COLORNAMES; //!< Number of output feature channels extracted by ColorNames.
	uint8_t *m_srcImgPatchPtr_d[2]; //!< Device pointers to externally allocated source image storage.
	uint m_srcSubImgWidth[2]; //!< Source image-patch widths for each plane.
	uint m_srcSubImgHeight[2]; //!< Source image-patch heights for each plane, including the NV12 plane layout.
	uint m_dstSubImgHeight[2]; //!< Destination image-patch heights for each plane.
	uint2 m_outBatchingGrid; //!< Grid dimensions used to compose outputs in memory.
	uint m_srcImgColorChannels; //!< Number of color channels in a source image patch.
	NvBufSurfaceColorFormat m_colorFormatType; //!< Source image color format.
	uint m_srcImgPatchPitch[2]; //!< Source image-patch pitches for each plane.
	uint m_srcImgPatchNum; //!< Number of source image patches processed in a batch.
	uint m_maxfeatureImgNum; //!< Maximum number of feature images processed in a batch.
	uint m_maxNumFramesInBatch; //!< Maximum number of video streams processed in a batch.
	uint m_numFeatureChannelsOut; //!< Total # of feature channels, including other types of features, so extra channel space can be reserved.
	uint m_featureTypes; //!< The types of visual features to be used (e.g., ColorNames and/or HOG).
	cuComplex *m_batch_ftrs_device; //!< Device memory to hold the extracted features.
	size_t m_batch_ftrs_device_pitch; //!< Pitch of the device feature buffer.
	void *m_batch_array_host_real; //!< Host memory to download the extracted features.
	std::shared_ptr<cuDCFFrameTransformTexture> m_subImageCropper2; //!< Cropper & Scaler.
	bool m_initGpu; //!< Indicates whether GPU feature extraction is initialized.

	// [Configuration parameters]
	int m_debugVerboseLevel; //!< Debug verbosity level.
	const int m_cudaBlockSize = 32; //!< Tile/block size.
	const bool m_showGpuTiming = false; //!< Controls whether GPU timing is displayed.
	int m_hogCellSize; //!< Cell size of HOG feature.

	// [CUDA timer]
	cudaEvent_t m_eventStart; //!< Event start point for GPU timers.
	cudaEvent_t m_eventStop; //!< Event stop point for GPU timers.

	// [VPI]
	bool m_isVpiInitialized; //!< Indicates whether VPI resources are initialized.
	VPIImage m_vpiInputImage[2]; //!< For input image source.
	VPIImageData m_vpiInputImageData[2]; //!< VPI metadata for input image planes.
	std::vector<VPIImage> m_vpiOutFeatures; //!< VPI output feature images.
	std::vector<VPIImageData> m_vpiOutFeaturesData; //!< Metadata for VPI output feature images.
	VPIStream m_vpiStream; //!< Stream used for VPI operations.
	VPIPayload m_vpiColorNames; //!< VPI ColorNames payload.
	VPIPayload m_vpiHOG; //!< VPI HOG payload.
	void *m_batch_vpi_ftrs; //!< CUDA output buffer where VPI lib will write the extracted features.
	size_t m_batch_vpi_ftrs_pitch; //!< Pitch of the VPI feature output buffer.
	VPIBackend m_vpiDevType; //!< VPI backend used for feature extraction.
	const int m_vpiImgFlags = VPI_BACKEND_CUDA | VPI_RESTRICT_MEM_USAGE; //!< Flags used to wrap CUDA image memory for VPI.
	//!< Source image patches are on CUDA memory.
	VPIImageFormat m_vpiImgOutputType; //!< VPI image format used for feature output.
	//!< Unsigned 8-bit or 16-bit integer, so the range is [0, 255] or [0, 65535], whereas ColorNames values are in [-1, 1).
};

#endif // CUDCFLIB_FEATUREEXTRACTOR_HPP_
