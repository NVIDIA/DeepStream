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

#ifndef cuDCFFrameTransform_HPP_
#define cuDCFFrameTransform_HPP_

#include "modules/FrameTransform/FrameTransform.hpp"
#include "modules/utility/dataTypes.hpp"

/** @brief Crops, scales, and converts tracker image patches with CUDA textures. */
class cuDCFFrameTransformTexture : public FrameTransformTexture
{
public:
	/** @brief Construct an uninitialized CUDA texture frame transform. */
	cuDCFFrameTransformTexture();

	/** @brief Release CUDA texture frame-transform resources. */
	~cuDCFFrameTransformTexture();

    /**
     * @brief Initialize for image cropping and scaling.
     * @param[in] debugVerboseLevel The debug level that selects which intermediate information is printed.
     * @param[in] maxStreamNum The maximum number of streams to extract.
     * @param[in] maxBufSurfAddrSize The maximum number of possible surface buffer addresses.
     * @param[in] maxTargetNum The maximum number of targets.
     * @param[in] srcImgSize The input image size.
     * @param[in] srcImgPitch The input image pitch.
     * @param[in] colorFormatType The color format of the input image.
     * @param[in] dstImgSize The image size after resizing.
     * @param[in] needSecondaryScale Whether the image needs a secondary scale level.
     * @param[in] secondaryScaleFactor The secondary scaling factor.
     */
    void initialize(
        const int debugVerboseLevel, const int maxStreamNum, const int maxBufSurfAddrSize,
        const int maxTargetNum, const NvMotDataTypes::Size srcImgSize, const uint srcImgPitch,
        const NvBufSurfaceColorFormat colorFormatType, const NvMotDataTypes::Size dstImgSize,
        const bool needSecondaryScale = false, const float secondaryScaleFactor = 1.0
    );

    /**
     * @brief Transform (crop, scale and color conversion) a set of bboxes for the given single frame.
     * @param[in] roisInBatch The ROIs for each stream, as pairs of bbox index and rectangle.
     * @param[in] lastBboxInd The last bbox index in the batch.
     * @return True when the batch transform finishes.
     */
    bool frameTransformInBatch(
        const std::map<NvMOTStreamId, std::vector<std::pair<uint, NvMotDataTypes::Rect2f> > > &roisInBatch,
        const uint lastBboxInd
    );

    /**
     * @brief Transform (crop, scale and color conversion) a set of bboxes for the given single frame.
     * @param[in] roisInBatch The ROIs for each stream, as pairs of bbox index and rectangle.
     * @param[in] lastBboxInd The last bbox index in the batch.
     * @param[in] doMosaicing Whether to mosaic the cropped patches.
     * @return True when the batch transform finishes.
     */
    bool frameTransformInBatch_NV12(
        const std::map<NvMOTStreamId, std::vector<std::pair<uint, NvMotDataTypes::Rect2f> > > &roisInBatch,
        const uint lastBboxInd, bool doMosaicing
    );

    /**
     * @brief Get the pointer to the output image patches on device memory with metadata.
     * @param[out] dataPtr_d The pointer to the beginning of the device data.
     * @param[out] width The width of an image patch.
     * @param[out] height The height of an image patch.
     * @param[out] outBatchingGrid The dimensions of the batching grid.
     * @param[out] channels The number of image channels.
     * @param[out] pitch The byte length of one image line.
     * @param[out] imgNum The number of images in the batch.
     * @param[in] multiResInd The multi-resolution scale index.
     */
    void getOutputImagePatchPtrs(
        uint8_t* &dataPtr_d, uint &width, uint &height, uint2 &outBatchingGrid,
        uint &channels, uint &pitch, uint &imgNum, uint multiResInd
    );

	/** @brief Return the number of output scale levels. */
	int getMultiResScaleNum() const { return m_multiResScaleNum; }

	/** @brief Return the two-dimensional output batching grid. */
	uint2 getOutBatchingGrid() const { return m_outBatchingGrid; }

	/** @brief Report whether frame-transform initialization completed. */
	bool isInitialized() const { return m_isInit; }

	#if defined(VIZ_DEBUG)
		/** @brief Return source images cached in host memory for visualization. */
		const std::map<NvMOTStreamId, cv::Mat> &getSourceImagesAtHost() const
		{
			if(!m_frameInput)
			{
				throw std::runtime_error("frameInput is invalid");
			}
			const std::map<NvMOTStreamId, cv::Mat> & srcFramesAtHost = m_frameInput->getSrcFramesAtHost();
			if(srcFramesAtHost.size() == 0)
			{
				throw std::runtime_error("srcFramesAtHost is not prepared");
			}
			else
			{
				return srcFramesAtHost;
			}
		}

		/**
		 * @brief Copy a set of output images from the device to the host memories.
		 * @param[in] multiResInd The multi-resolution scale index.
		 * @return Cropped and scaled output images on the host, keyed by stream and target.
		 */
		std::map<NvMOTStreamId, std::map<uint, cv::Mat > > getImagePatches2Host_OCV_mosaic(uint multiResInd = 0);

		/**
		 * @brief Copy batched output image patches from device to host memory.
		 * @param[in] multiResInd The multi-resolution scale index.
		 * @return Cropped and scaled output images on the host, keyed by stream and target.
		 */
		std::map<NvMOTStreamId, std::map<uint, cv::Mat > > getImagePatches2Host_OCV_batch(uint multiResInd = 0);
	#endif

private:
    /**
     * @brief Set output config and allocate the device and host memories for the output frame patches.
     * @param[in] dstSize The output patch size.
     */
    void initialize4Output(const NvMotDataTypes::Size dstSize);

	/** @brief Allocate the device and host memories needed to perform transform. */
	void allocateOutputMemories();

    /** @brief Release the device and host memories needed to perform transform. */
    void deAllocateOutputMemories();

	bool m_isInit; //!< Indicates whether initialization completed.
    bool m_doMosaicing; //!< Indicates whether cropped patches are mosaiced.

	// [Output configuration specific to cuDCF]
    int m_width_out[CUDA_TEXTURE_MAX_MULTI_SCALE_NUM], m_height_out[CUDA_TEXTURE_MAX_MULTI_SCALE_NUM]; //!< Width and height of output image (feature extraction).
    uint2 m_outBatchingGrid; //!< Grid dimensions used to compose outputs in memory.
    int m_multiResScaleNum; //!< Number of output scale levels.
    float m_multiResScaleFactors[CUDA_TEXTURE_MAX_MULTI_SCALE_NUM]; //!< Scale factor for each output level.

	// [Output memory specific to cuDCF]
    uint8_t* m_odataPL_d[CUDA_TEXTURE_MAX_MULTI_SCALE_NUM]; //!< Regular device memory for output.
    size_t m_odataPL_d_pitch[CUDA_TEXTURE_MAX_MULTI_SCALE_NUM]; //!< Pitch at the device memory.
    uint8_t *m_odata_host[CUDA_TEXTURE_MAX_MULTI_SCALE_NUM]; //!< Host memory for output.
};

#endif // cuDCFFrameTransform_HPP_
