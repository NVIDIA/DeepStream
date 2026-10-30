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

#ifndef ReIDFrameTransform_HPP_
#define ReIDFrameTransform_HPP_

#if defined VIZ_DEBUG_REID || defined VIZ_DEBUG_SEG || defined VIZ_DEBUG_POSE
#include <opencv2/core.hpp>
#endif

#include "modules/FrameTransform/FrameTransform.hpp"

/** @brief Crops and rescales target regions through CUDA texture transforms. */
class ReIDFrameTransformTexture : public FrameTransformTexture
{
public:
	/** @brief Construct a ReIDFrameTransformTexture instance. */
	ReIDFrameTransformTexture();

	/** @brief Release resources owned by the ReIDFrameTransformTexture instance. */
	~ReIDFrameTransformTexture();

    /**
     * @brief Initialize for image cropping and scaling.
     * @param[in] debugVerboseLevel Debug verbosity level.
     * @param[in] maxStreamNum Maximum number of input streams.
     * @param[in] maxBufSurfAddrSize Maximum number of cached surface-buffer addresses.
     * @param[in] maxTargetNum Maximum number of targets in a batch.
     * @param[in] srcImgSize Input image dimensions.
     * @param[in] srcImgPitch Input image pitch in bytes.
     * @param[in] colorFormatType Input surface color format.
     * @param[in] dstImgSize Size of image after resizing.
     * @param[in] dstColorFormatType Output color format RGB or BGR.
     * @param[in] dimOrder Output tensor dimension order.
     */
    void initialize(
        const int debugVerboseLevel, const int maxStreamNum, const int maxBufSurfAddrSize, const int maxTargetNum,
        const NvMotDataTypes::Size srcImgSize, const uint srcImgPitch, const NvBufSurfaceColorFormat colorFormatType,
        const NvMotDataTypes::Size dstImgSize, const NvBufSurfaceColorFormat dstColorFormatType, const int dimOrder
    );

    /**
     * @brief Transform (crop, scale and color conversion) a set of bboxes for the given single frame, all the bboxes will be scaled to the same size.
     * @param[in] roisInBatch The set of ROIs of interest {videoStreamID, a vector of {bboxInd, bbox} }.
     * @param[in] lastBboxInd Last bbox index.
     * @param[in] outputScaleFactor Scale applied to output pixel values.
     * @param[in] outputOffsets Parameter of each channel's offset.
     * @return Status.
     */
    bool frameTransformInBatch(
        const std::map<NvMOTStreamId, std::vector<std::pair<uint, NvMotDataTypes::Rect2f> > > &roisInBatch,
        const uint lastBboxInd, const float outputScaleFactor, const std::vector<float> outputOffsets
    );

    /**
     * @brief Transform (crop, scale and color conversion) a set of bboxes for the given single frame, all the bboxes will be scaled to the same size.
     * @param[in] roisInBatch The set of ROIs of interest {videoStreamID, a vector of {bboxInd, bbox} }.
     * @param[in] lastBboxInd Last bbox index.
     * @param[in] outputScaleFactor Scale applied to output pixel values.
     * @param[in] outputOffsets Parameter of each channel's offset.
     * @return Status.
     */
    bool frameTransformInBatch_NV12(
        const std::map<NvMOTStreamId, std::vector<std::pair<uint, NvMotDataTypes::Rect2f> > > &roisInBatch,
        const uint lastBboxInd, const float outputScaleFactor, const std::vector<float> outputOffsets
    );

    /** @brief Get the pointer to the output image patches on device memory with metadata. */
    float* getOutputImagePatchPtrs() { return m_odata_d; }

    #if defined VIZ_DEBUG_REID || defined VIZ_DEBUG_SEG || defined VIZ_DEBUG_POSE
        /**
         * @brief Get patches to host for visualization debugging.
         * @param[in] outputScaleFactor Scale applied to output pixel values.
         * @param[in] outputOffsets Per-channel offsets restored in the host image.
         * @return Host image patches keyed by stream and target.
         */
        std::map<NvMOTStreamId, std::map<uint, cv::Mat > > getImagePatches2Host_OCV_batch(
            const float outputScaleFactor, const std::vector<float> outputOffsets
        );
    #endif

private:
    /**
     * @brief Set output config and allocate the device and host memories for the output frame patches.
     * @param[in] dstSize Output patch size.
     * @param[in] dstColorFormat Output color format, RGB or BGR.
     * @param[in] dimOrder Output tensor dimension order, 0 for NCHW and 1 for NHWC.
     */
    void initialize4Output(
        const NvMotDataTypes::Size dstSize, const NvBufSurfaceColorFormat dstColorFormat, const int dimOrder
    );

	/** @brief Allocate the device and host memories needed to perform transform. */
	void allocateOutputMemories();

    /** @brief Release the device and host memories needed to perform transform. */
    void deAllocateOutputMemories();

	bool m_isInit; //!< If initialized.

	// [Output configuration specific to ReID]
	int m_width_out, m_height_out, m_channels_out; //!< Width, height and channel of output image (feature extraction).
	int m_dimOrder; //!< Infer dimension order, 0: NCHW; 1: NHWC.
	NvBufSurfaceColorFormat m_dstColorFormatType; //!< Output NvBufSurfaceColorFormat.

	// [Output memory specific to ReID]
	float *m_odata_d; //!< Regular device memory for output for inference.
	float *m_odata_host; //!< Host memory for output for inference.
	#if defined VIZ_DEBUG_REID || defined VIZ_DEBUG_SEG || defined VIZ_DEBUG_POSE
		float *m_odata_host_nchw; //!< Host memory nchw output for inference.
	#endif
};

#endif // ReIDFrameTransform_HPP_
