/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef ReIDCropScaler_HPP_
#define ReIDCropScaler_HPP_

#include <map>
#include <vector>
#include <unordered_map>
#include "nvbufsurface.h"
#include "vpi/Stream.h"
#include "vpi/Image.h"
#include "vpi/Array.h"
#include "vpi/algo/CropScaler.h"
#include "modules/utility/dataTypes.hpp"
#include "modules/NvMultiObjectTracker/nvdstracker.h"

#if defined(__aarch64__)
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include "cudaEGL.h"
#endif

/** @brief Crops and rescales target regions with VPI for network input. */
class ReIDCropScaler
{
public:
	/** @brief Construct a ReIDCropScaler instance. */
	ReIDCropScaler();

	/** @brief Release resources owned by the ReIDCropScaler instance. */
	~ReIDCropScaler();

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
     * @param[in] inputMeta Input frames keyed by stream ID.
     * @return Status.
     */
    bool frameTransformInBatch(
        const std::map<NvMOTStreamId, std::vector<std::pair<uint, NvMotDataTypes::Rect2f> > > &roisInBatch,
        const int lastBboxInd, const float outputScaleFactor, const std::vector<float> outputOffsets,
        const std::map<NvMOTStreamId, NvMOTFrame*> &inputMeta
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

	/**
	 * @brief Allocate the device and host memories for the input source image.
	 * @param[in] numFramesInBatch Number of frames in the batch.
	 * @param[in] nx Input image width in pixels.
	 * @param[in] ny Input image height in pixels.
	 * @param[in] pitch_in Input image pitch in bytes.
	 */
	void allocateInputMemoriesBatch(const uint numFramesInBatch, const int nx, const int ny, const uint pitch_in);

	/** @brief Release the device and host memories for the input images. */
	void deAllocateInputMemoriesBatch();

	/**
	 * @brief Set input image size and allocate the device and host memories for the input images.
	 * @param[in] numFramesInBatch Number of frames in the batch.
	 * @param[in] frameSize Input frame dimensions.
	 * @param[in] srcImgPitch Input image pitch in bytes.
	 * @param[in] frameChannels Number of input image channels.
	 */
	void initialize4Input(
		uint numFramesInBatch, const NvMotDataTypes::Size frameSize, const uint srcImgPitch, const uint frameChannels
	);

	/**
	 * @brief Check the input and create a frame sequence for VPI crop scaler.
	 * @param[in] inputMeta Input frames keyed by stream ID.
	 */
	void checkInput(const std::map<NvMOTStreamId, NvMOTFrame*> &inputMeta);

	/**
	 * @brief Get the cuda memory pointers for all the planes in a NvBufSurface.
	 * @param[in] baseSurface Surface whose CUDA plane pointers are retrieved.
	 */
	inline void getCudaMemPointers(NvBufSurfaceParams *baseSurface);

	bool m_isInit; //!< If initialized.

	// [Output configuration specific to ReID]
	int m_width_out, m_height_out, m_channels_out; //!< Width, height and channel of output image (feature extraction).
	int m_dimOrder; //!< Infer dimension order, 0: NCHW; 1: NHWC.
	NvBufSurfaceColorFormat m_dstColorFormatType; //!< Output NvBufSurfaceColorFormat.
	VPIImageFormat m_vpiImageFormat; //!< Output image format for VPI crop scaler.
	cudaStream_t m_cuStream; //!< CUDA Stream to be used.
	const int m_cudaBlockSize_2D = 32; //!< CUDA block size.
	int m_debugVerboseLevel; //!< Debug message verbose mode level.
	int m_maxTargetsToTrack; //!< Max # of targets to track.
	uint m_lastInd; //!< The # of targets of interest.
	uint m_maxBufSurfAddrSize; //!< Max # of egl mapped addresses to be stored.
	uint m_numFramesInBatchMax; //!< The max # of frames to be processed in batch.

	// [Input image format]
	int m_width_in, m_height_in, m_pitch_in, m_channels_in; //!< Width, height, pitch and channels of source image.
	NvBufSurfaceColorFormat m_colorFormatType; //!< NvBufSurfaceColorFormat for source image.

	// [Output memory specific to ReID]
	float *m_odata_d; //!< Regular device memory for output for inference.
	float *m_odata_host; //!< Host memory for output for inference.
	#if defined VIZ_DEBUG_REID || defined VIZ_DEBUG_SEG || defined VIZ_DEBUG_POSE
		float *m_odata_host_nchw; //!< Host memory nchw output for inference.
	#endif
	uint8_t *m_odata_unit8_d; //!< Int8 device memory for VPI output that contains cropped and scaled object patches.
	VPIImageData m_vpiObjPatchesData; //!< VPIImageData corresponding to m_odata_unit8_d.
	VPIImage m_vpiObjPatches; //!< VPIImage corresponding to m_odata_unit8_d.

	// [Getting CUDA pointers from NvBufSurfaces]
	int m_integrated_gpu; //!< Whether to run on iGPU.
	int m_deviceId; //!< GPU device ID.
	uint8_t *m_surfCudaMemPtr[NVBUF_MAX_PLANES]; //!< Pointers to cuda memories of all the planes in a surface.
	#if defined(__aarch64__)
		std::unordered_map<const NvBufSurfaceParams *, CUgraphicsResource> m_graphicsResourceAddrMap; //!< Graphics resources keyed by surface address.
		//!< DeepStream reuses a finite frame-buffer pool, so resources are cached by surface address.
		std::unordered_map<const NvBufSurfaceParams *, CUeglFrame> m_eglFrameAddrMap; //!< Map storing (BufSurfaceParams address, CUeglFrame) pairs.
	#endif

	// [VPI]
	VPIStream m_vpiStream; //!< VPI stream used for all VPI operations.
	VPIBackend m_vpiBackend4CropnScale = VPI_BACKEND_CUDA; //!< VPI backend for Crop Scaler.
	VPIPayload m_vpiCropPayload; //!< VPI payload for Crop scaler.
	int m_vpiMemFlags = VPI_BACKEND_CUDA | VPI_RESTRICT_MEM_USAGE; //!< Flags used for creating VPI Image/Array wrappers.

	// [Sequence of input frames]
	VPIImage *m_vpiInputFrameSequence; //!< Sequence of VPIImage pointers to input frames.
	VPIImageData *m_vpiInputFrameDataList; //!< Sequence of VPIImageData pointers to input frames.
	std::vector<bool> m_isFrameCreated; //!< Flags indicating which sequence images have been created.
	//!< A true value indicates that vpiImageCreateWrapper created the image at that sequence index.

	// [List of tracked objects]
	std::vector<VPIDCFTrackedBoundingBox> m_vpiDCFBboxes; //!< List of tracked objects.
	//!< Reserved size is "maxTargetsToTrack".
	VPIArrayData m_vpiDCFBboxesData; //!< VPIArrayData for m_vpiDCFBboxes.
	VPIArray m_vpiDCFBboxesArray; //!< VPIArray for m_vpiDCFBboxes.
	int32_t m_vpiBboxesArraySize; //!< Number of active entries in m_vpiDCFBboxes.
	//!< The value supports dynamic compute adjustment.

	#if defined VIZ_DEBUG_REID || defined VIZ_DEBUG_SEG || defined VIZ_DEBUG_POSE
		std::map<uint, std::pair<NvMOTStreamId, uint>> m_targetNumInd2RoisInBatchMapper; //!< Map the target index to the corresponding video stream ID and the bbox ind {targetInd, { videoStreamID, bboxInd} }.
	#endif
};

#endif // ReIDCropScaler_HPP_
