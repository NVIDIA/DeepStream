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

#ifndef FrameTransform_HPP_
#define FrameTransform_HPP_

#include <stdio.h>
#include <memory>
#include <vector>
#include <unordered_map>
#include <map>

#if defined(VIZ_DEBUG)
#include <opencv2/core.hpp>
#endif

#include "modules/utility/dataTypes.hpp"
#include "modules/NvMultiObjectTracker/nvdstracker.h"
#include <cuda.h>
#include <cuda_runtime.h>

#if defined(__aarch64__)
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include "cudaEGL.h"
#endif

#define CUDA_TEXTURE_MAX_MULTI_SCALE_NUM	2

/// Maximum width and height of a 2D texture bound to linear memory.
#define CUDA_LIMIT_ON_2D_TEXTURE_WIDTH		65000
#define CUDA_LIMIT_ON_2D_TEXTURE_HEIGHT		65000

/** @brief Holds luma and chroma texture objects for an NV12 frame. */
struct TextureObjects
{
	cudaTextureObject_t luma; //!< Texture object for the luma plane.
	cudaTextureObject_t chroma; //!< Texture object for the chroma plane.
};

/** @brief Manages CUDA texture objects for batched input frames. */
class FrameInputTexture
{
public:
	/** @brief Construct an uninitialized input-texture manager. */
	FrameInputTexture();

	/** @brief Release resources owned by the FrameInputTexture instance. */
	~FrameInputTexture();

	/**
	 * @brief Initialize input texture state and allocate frame memory.
	 * @param[in] debugVerboseLevel Debug verbosity level.
	 * @param[in] maxStreamNum Maximum number of input streams.
	 * @param[in] maxBufSurfAddrSize Maximum number of cached surface-buffer addresses.
	 * @param[in] srcImgSize Input image dimensions.
	 * @param[in] srcImgPitch Input image pitch in bytes.
	 * @param[in] colorFormatType Input surface color format.
	 */
	void initialize(
		const int debugVerboseLevel, const int maxStreamNum, const int maxBufSurfAddrSize,
		const NvMotDataTypes::Size srcImgSize, const uint srcImgPitch, const NvBufSurfaceColorFormat colorFormatType
	);

	/**
	 * @brief Cache a batch of source frames and create their texture objects.
	 * @param[in] nvFramesInBatch Source frames keyed by stream ID.
	 * @return True when texture objects were created for the batch.
	 */
	bool create_batch(const std::map<NvMOTStreamId, NvMOTFrame*> &nvFramesInBatch);

	/** @brief Create texture objects for BGRA or grayscale frames in the current batch. */
	void createTexObjInBatch();

	/** @brief Create texture objects for NV12 frames in the current batch. */
	void createTexObjInBatch_NV12();

	/** @brief Destroy all texture objects for the current batch. */
	void destroyTextureObjects();

	/**
	 * @brief Remove the requested number of cached BGRA or grayscale texture objects.
	 * @param[in] cleanUpNum Number of cached texture objects to remove.
	 */
	void cleanUpTextureObjects(int cleanUpNum);

	/**
	 * @brief Remove the requested number of cached NV12 texture objects.
	 * @param[in] cleanUpNum Number of cached texture objects to remove.
	 */
	void cleanUpTextureObjects_NV12(int cleanUpNum);

	/**
	 * @brief Get the BGRA or grayscale texture-object pointer for a stream.
	 * @param[in] streamId Video stream identifier.
	 * @return Pointer to the stream's texture object, or null when unavailable.
	 */
	cudaTextureObject_t *getTexturePointer(NvMOTStreamId streamId);

	/**
	 * @brief Get the NV12 texture-object pair for a stream.
	 * @param[in] streamId Video stream identifier.
	 * @return Pointer to the stream's texture-object pair, or null when unavailable.
	 */
	TextureObjects *getTexturePointer_NV12(NvMOTStreamId streamId);

	/** @brief Get the input image width in pixels. */
	int getWidthIn() { return m_width_in; }

	/** @brief Get the input image height in pixels. */
	int getHeightIn() { return m_height_in; }

	/** @brief Get the number of input image channels. */
	int getChannelsIn() { return m_channels_in; }

	/** @brief Get the input surface color format. */
	NvBufSurfaceColorFormat getColorFormat() const { return m_colorFormatType; }
	#if defined(VIZ_DEBUG)
		/** @brief Get host copies of the input source frames. */
		const std::map<NvMOTStreamId, cv::Mat>& getSrcFramesAtHost() const { return m_srcFramesAtHost; }
	#endif

protected:
	/**
	 * @brief Set the input image layout and allocate batch memory.
	 * @param[in] numFramesInBatch Number of frames in the batch.
	 * @param[in] frameSize Input frame dimensions.
	 * @param[in] srcImgPitch Input image pitch in bytes.
	 * @param[in] frameChannels Number of input image channels.
	 */
	void initialize4Input(
		uint numFramesInBatch, const NvMotDataTypes::Size frameSize, const uint srcImgPitch, const uint frameChannels
	);

	/** @brief Initialize CUDA resource and texture descriptors. */
	void initializeTextureDescriptors();

	/**
	 * @brief Allocate device and host memory for a batch of input images.
	 * @param[in] numFramesInBatch Number of frames in the batch.
	 * @param[in] nx Input image width in pixels.
	 * @param[in] ny Input image height in pixels.
	 * @param[in] pitch_in Input image pitch in bytes.
	 */
	void allocateInputMemoriesBatch(const uint numFramesInBatch, const int nx, const int ny, const uint pitch_in);

	/** @brief Release device and host memory for the input images. */
	void deAllocateInputMemoriesBatch();

	/**
	 * @brief Upload a batch of source frames to pitch-linear CUDA memory.
	 * @param[in] nvFramesInBatch Source frames keyed by stream ID.
	 */
	void uploadDeviceData2DeviceMemories(const std::map<NvMOTStreamId, NvMOTFrame*> &nvFramesInBatch);

	bool m_isInit; //!< Indicates whether input texture state is initialized.

	// [Input frames and textures]
	std::map<NvMOTStreamId, NvMOTFrame*> m_srcNvFrames; //!< Input source frames keyed by stream ID.
	std::map<NvMOTStreamId, cudaTextureObject_t*> m_pTex; //!< BGRA or grayscale texture pointers keyed by stream ID.
	std::map<NvMOTStreamId, TextureObjects*> m_pTex_NV12; //!< NV12 texture-object pairs keyed by stream ID.
	std::map<const NvBufSurfaceParams*, cudaTextureObject_t> m_texInputAddrMap; //!< BGRA or grayscale texture objects keyed by surface address.
	std::map<const NvBufSurfaceParams*, TextureObjects> m_texInputAddrMap_NV12; //!< NV12 texture-object pairs keyed by surface address.
	cudaResourceDesc m_resDesc_luma; //!< CUDA resource descriptor for NV12 luma data.
	cudaResourceDesc m_resDesc_chroma; //!< CUDA resource descriptor for NV12 chroma data.
	cudaResourceDesc m_resDesc; //!< CUDA resource descriptor for RGBA data.
	cudaTextureDesc m_texDesc; //!< CUDA texture descriptor shared by input formats.
	int m_maxTexAddrSize; //!< Maximum number of cached texture addresses.
	uint m_numFramesInBatchMax; //!< Maximum number of frames processed in a batch.

	#if defined(VIZ_DEBUG)
		std::map<NvMOTStreamId, cv::Mat> m_srcFramesAtHost; //!< Host copies of input source frames.
		uint8_t *m_idata_host; //!< Host memory for input frames.
		float* m_idataPL_d; //!< Pitch-linear device memory for input frames.
		size_t m_idataPL_d_pitch; //!< Pitch of the input device memory in bytes.
		cudaStream_t m_cuStream; //!< CUDA stream shared with FrameTransformTexture.
	#endif

	#if defined(__aarch64__)
		std::unordered_map<const NvBufSurfaceParams*, CUgraphicsResource> m_graphicsResourceAddrMap; //!< Graphics resources keyed by surface address.
		//!< DeepStream reuses a finite frame-buffer pool, so graphics resources are cached by surface address.
		std::unordered_map<const NvBufSurfaceParams*, CUeglFrame> m_eglFrameAddrMap; //!< EGL frames keyed by surface address.
	#endif

	// [Input image format]
	int m_width_in, m_height_in, m_pitch_in, m_channels_in; //!< Width, height, pitch, and channel count of the source image.
	NvBufSurfaceColorFormat m_colorFormatType; //!< Color format of the source surface.

	// [CUDA configuration]
	int m_integrated_gpu; //!< Indicates whether execution uses an integrated GPU.
	int m_deviceId; //!< GPU device identifier.
	int m_debugVerboseLevel; //!< Debug verbosity level.
};

/** @brief Provides common CUDA texture state for frame transformations. */
class FrameTransformTexture
{
public:
	/**
	 * @brief Construct common frame-transform state.
	 * @param[in] cudaBlockSize_2D CUDA block width and height for two-dimensional kernels.
	 */
	FrameTransformTexture(const int cudaBlockSize_2D = 32);

	/** @brief Release resources owned by the FrameTransformTexture instance. */
	~FrameTransformTexture();

	/**
	 * @brief Initialize common transform state and allocate output memory.
	 * @param[in] debugVerboseLevel Debug verbosity level.
	 * @param[in] maxStreamNum Maximum number of input streams.
	 * @param[in] maxBufSurfAddrSize Maximum number of cached surface-buffer addresses.
	 * @param[in] maxTargetNum Maximum number of targets in a batch.
	 * @param[in] srcImgSize Input image dimensions.
	 * @param[in] srcImgPitch Input image pitch in bytes.
	 * @param[in] colorFormatType Input surface color format.
	 */
	void initialize_common(
		const int debugVerboseLevel, const int maxStreamNum, const int maxBufSurfAddrSize, const int maxTargetNum,
		const NvMotDataTypes::Size srcImgSize, const uint srcImgPitch, const NvBufSurfaceColorFormat colorFormatType
	);

	/**
	 * @brief Update texture objects from a batch of source frames.
	 * @param[in] inputMeta Source frames keyed by stream ID.
	 */
	void updateTextureObject(const std::map<NvMOTStreamId, NvMOTFrame*> &inputMeta);

	/** @brief Get the input surface color format. */
	NvBufSurfaceColorFormat getColorFormat() const { return m_frameInput->getColorFormat(); }

	/** @brief Get the maximum number of video streams in a batch. */
	uint getMaxVideoStreams() const { return m_numFramesInBatchMax; }

	/** @brief Report whether common frame-transform state is initialized. */
	bool isInitialized_common() const { return m_isInit_common; }

protected:
	/** @brief Allocate device and host memory for transform output. */
	void allocateOutputMemories_common();

	/** @brief Release device and host memory for transform output. */
	void deAllocateOutputMemories_common();

	bool m_isInit_common; //!< Indicates whether common transform state is initialized.

	// [Input texture]
	std::shared_ptr<FrameInputTexture> m_frameInput; //!< Shared input-frame texture manager.

	// [Frame ROIs to transform]
	float4 *m_rois_h, *m_rois_d; //!< Host and device regions of interest in x, y, width, and height order.
	uint2 *m_dstOffsets_h, *m_dstOffsets_d; //!< Host and device byte offsets into transform output memory.
	std::map<uint, std::pair<NvMOTStreamId, uint> > m_targetNumInd2RoisInBatchMapper; //!< Stream and bounding-box indices keyed by batch target index.

	// [Transform variables]
	int m_scaleFactor; //!< Scale that restores values read through normalized CUDA texture mode.
	uint m_lastInd; //!< Number of targets in the latest transform.
	uint m_numFramesInBatchMax; //!< Maximum number of frames processed in a batch.
	int m_maxTargetsToTrack; //!< Maximum number of targets in a batch.

	// [CUDA variables]
	const int m_cudaBlockSize_2D; //!< CUDA block width and height for two-dimensional kernels.
	cudaStream_t m_cuStream; //!< CUDA stream used for frame transformations.
	int m_debugVerboseLevel; //!< Debug verbosity level.
};

#endif // FrameTransform_HPP_
