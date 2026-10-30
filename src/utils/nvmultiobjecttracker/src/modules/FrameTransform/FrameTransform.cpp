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

#include <cstdio>
#include <cstdlib>
#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "modules/utility/debug_common.hpp"
#include "modules/utility/cuda_common.hpp"
#include "modules/FrameTransform/FrameTransform.hpp"
#include "modules/cuDCFv2/helper_math.h"

using std::cout;
using std::endl;

/** @brief Construct a FrameInputTexture instance. */
FrameInputTexture::FrameInputTexture()
{
	cudaGetDevice(&m_deviceId);

	m_debugVerboseLevel = 0;

	m_width_in = 0;
	m_height_in = 0;
	m_channels_in = 0;
	m_numFramesInBatchMax = 0;
	m_isInit = false;
	m_integrated_gpu = 0;

	NvBufSurfaceDeviceInfo dev_info{};
	if (NvBufSurfaceGetDeviceInfo(&dev_info) == 0) {
		if (dev_info.driverType == NVBUF_DRIVER_TYPE_NVGPU) {
			m_integrated_gpu = 1;
		}
	}
}

/** @brief Release resources owned by the FrameInputTexture instance. */
FrameInputTexture::~FrameInputTexture()
{
	if (m_isInit)
	{
		deAllocateInputMemoriesBatch();
	}

	destroyTextureObjects();

	#if defined(VIZ_DEBUG)
		CUDA_CHECK(cudaStreamDestroy(m_cuStream));
	#endif
}

/**
 * @brief Configure input-frame geometry and allocate texture backing memory.
 * @param[in] debugVerboseLevel Debug verbosity level.
 * @param[in] maxStreamNum Maximum number of input streams.
 * @param[in] maxBufSurfAddrSize Maximum number of cached surface-buffer addresses.
 * @param[in] srcImgSize Input image dimensions.
 * @param[in] srcImgPitch Input image pitch in bytes.
 * @param[in] colorFormatType Input surface color format.
 */
void FrameInputTexture::initialize(
	const int debugVerboseLevel, const int maxStreamNum, const int maxBufSurfAddrSize,
	const NvMotDataTypes::Size srcImgSize, const uint srcImgPitch, const NvBufSurfaceColorFormat colorFormatType
)
{
	#if defined(VIZ_DEBUG)
		CUDA_CHECK( cudaStreamCreateWithFlags(&m_cuStream, cudaStreamNonBlocking) );
	#endif

	m_debugVerboseLevel = debugVerboseLevel;
	m_colorFormatType = colorFormatType;
	m_numFramesInBatchMax = maxStreamNum;
	m_maxTexAddrSize = maxBufSurfAddrSize;

	if(m_colorFormatType == NVBUF_COLOR_FORMAT_BGRA
			|| m_colorFormatType == NVBUF_COLOR_FORMAT_RGBA
			)
	{
		initialize4Input(maxStreamNum, srcImgSize, srcImgPitch, 4);
	}
	else if(m_colorFormatType == NVBUF_COLOR_FORMAT_NV12)
	{
		initialize4Input(maxStreamNum, srcImgSize, srcImgPitch, 1);
	}
	else
	{
		throw std::runtime_error("Invalid Color Formats");
	}

	initializeTextureDescriptors();

	m_isInit = true;
}

/**
 * @brief Configure the input layout and allocate frame memory.
 * @param[in] numFramesInBatch Number of frames in the batch.
 * @param[in] frameSize Input frame dimensions.
 * @param[in] srcImgPitch Input image pitch in bytes.
 * @param[in] frameChannels Number of input image channels.
 */
void FrameInputTexture::initialize4Input(
	uint numFramesInBatch, const NvMotDataTypes::Size frameSize, const uint srcImgPitch, const uint frameChannels
)
{
	/// Check if the cuda pitch-linear memory is already created and if so, it is compatible.
	/// (Assuming all the images in batch are homogeneous in terms of the size and channels).
	if(m_numFramesInBatchMax < numFramesInBatch || m_width_in != frameSize.width || m_height_in != frameSize.height || m_channels_in != (int)frameChannels)
	{
		if((m_width_in != 0 && m_height_in != 0) || (m_isInit &&  m_numFramesInBatchMax < numFramesInBatch))
		{

			deAllocateInputMemoriesBatch();
		}

		m_width_in = frameSize.width;
		m_height_in = frameSize.height;
		m_pitch_in = srcImgPitch;
		m_channels_in = frameChannels;

		m_numFramesInBatchMax = numFramesInBatch;

		/// Allocate memory for 1.5x height, because NV12 has 1.5x height total.
		allocateInputMemoriesBatch(numFramesInBatch, m_width_in, m_height_in*1.5f, m_pitch_in);

	}
}

/** @brief Initialize CUDA resource and texture descriptors. */
void FrameInputTexture::initializeTextureDescriptors()
{
	memset(&m_resDesc_luma, 0, sizeof(m_resDesc_luma));
	/// Set memory params for texture registration.
	m_resDesc_luma.resType = cudaResourceTypePitch2D;

	m_resDesc_luma.res.pitch2D.desc = cudaCreateChannelDesc<uchar1>();
	m_resDesc_luma.res.pitch2D.width = m_width_in;
	m_resDesc_luma.res.pitch2D.height = m_height_in;

	memset(&m_resDesc_chroma, 0, sizeof(m_resDesc_chroma));
	/// Set memory params for texture registration.
	m_resDesc_chroma.resType = cudaResourceTypePitch2D;

	m_resDesc_chroma.res.pitch2D.desc = cudaCreateChannelDesc<uchar2>();
	m_resDesc_chroma.res.pitch2D.width = m_width_in * 1/2;
	m_resDesc_chroma.res.pitch2D.height = m_height_in * 1/2;

	/// Define m_resDesc and m_texDesc.
	memset(&m_resDesc, 0, sizeof(m_resDesc));
	/// Set memory params for texture registration.
	m_resDesc.resType = cudaResourceTypePitch2D;

	m_resDesc.res.pitch2D.width = m_width_in;
	m_resDesc.res.pitch2D.height = m_height_in;

	memset(&m_texDesc, 0, sizeof(m_texDesc));
	/// Set texture parameters input and output memories are in a different type, so need to use NormalizedFloat read mode.
	m_texDesc.readMode = cudaReadModeNormalizedFloat;
	/// Clamp mode := Reflection.
	m_texDesc.addressMode[0] = cudaAddressModeClamp;
	/// Clamp mode := Reflection.
	m_texDesc.addressMode[1] = cudaAddressModeClamp;
	m_texDesc.filterMode = cudaFilterModeLinear;
	/// Access with normalized texture coordinates.
	m_texDesc.normalizedCoords = true;
}

/**
 * @brief Allocate input memory for a frame batch.
 * @param[in] numFramesInBatch Number of frames in the batch.
 * @param[in] nx Input image width in pixels.
 * @param[in] ny Input image height in pixels.
 * @param[in] pitch_in Input image pitch in bytes.
 */
void FrameInputTexture::allocateInputMemoriesBatch(
	const uint numFramesInBatch, const int nx, const int ny, const uint pitch_in
)
{
	#if defined(VIZ_DEBUG)
		CUDA_CHECK( cudaMallocHost( reinterpret_cast<void**>( &m_idata_host ), pitch_in*(ny*1.5)*numFramesInBatch ));

		CUDA_CHECK(
			cudaMallocPitch(
				reinterpret_cast<void **>(&m_idataPL_d), &m_idataPL_d_pitch,
				nx * sizeof(float)*m_channels_in, (ny*1.5) * numFramesInBatch
			)
		);
	#endif
}

/** @brief Release input memory for the frame batch. */
void FrameInputTexture::deAllocateInputMemoriesBatch()
{
	#if defined(VIZ_DEBUG)
		CUDA_CHECK( cudaFreeHost(m_idata_host) );
		CUDA_CHECK( cudaFree(m_idataPL_d) );
	#endif
}

/** @brief Destroy all cached texture objects. */
void FrameInputTexture::destroyTextureObjects()
{
	for(auto it = m_texInputAddrMap_NV12.begin(); it != m_texInputAddrMap_NV12.end(); ++it)
	{
		CUDA_CHECK( cudaDestroyTextureObject(it->second.luma) );
		CUDA_CHECK( cudaDestroyTextureObject(it->second.chroma) );
	}

	m_texInputAddrMap_NV12.clear();
	m_pTex_NV12.clear();

	for(auto it = m_texInputAddrMap.begin(); it != m_texInputAddrMap.end(); ++it)
	{
		CUDA_CHECK( cudaDestroyTextureObject(it->second) );
	}

	m_texInputAddrMap.clear();
	m_pTex.clear();

	#if defined(__aarch64__)
		if(m_integrated_gpu)
		{
			for(auto it = m_graphicsResourceAddrMap.begin(); it != m_graphicsResourceAddrMap.end(); ++it)
			{
				CUDA_CHECK( cuGraphicsUnregisterResource(it->second) );
			}
			m_graphicsResourceAddrMap.clear();

			m_eglFrameAddrMap.clear();
		}
	#endif
}

/**
 * @brief Cache an input batch and create its texture objects.
 * @param[in] inputMeta Source frames keyed by stream ID.
 */
bool FrameInputTexture::create_batch(const std::map<NvMOTStreamId, NvMOTFrame*> &inputMeta)
{
	if(inputMeta.size() == 0)
		return false;

	if(!m_isInit)
	{
		throw std::runtime_error("Texture input uninitialized");
	}

	/// Upload to cuda pitch-linear memory.
	uploadDeviceData2DeviceMemories(inputMeta);

	return true;
}

/**
 * @brief Upload input frames to pitch-linear device memory.
 * @param[in] inputMeta Source frames keyed by stream ID.
 */
void FrameInputTexture::uploadDeviceData2DeviceMemories(const std::map<NvMOTStreamId, NvMOTFrame*> &inputMeta)
{

	// Keep cropped patches on the host when visual debugging is enabled.
	#if defined(VIZ_DEBUG)
		m_srcFramesAtHost.clear();

		/// Copy host data to device Pitch linear.
		uint cnt = 0;
		for(auto it = inputMeta.begin(); it != inputMeta.end(); ++it, ++cnt)
		{
			const NvBufSurfaceParams *baseSurface = it->second->bufferList[0];

			const cv::Size srcImageSize = cv::Size(baseSurface->width, baseSurface->height);
			const uint pitchBytes_src = baseSurface->pitch;

			if(m_debugVerboseLevel > 2)
			{
				printf(
					"[FrameInputTexture::uploadDeviceData2DeviceMemories()] (%d out of %d) baseSurface->width, pitchBytes_src*m_height_in, sizeof(baseSurface), sizeof(m_idata_host) = %d, %d, %d, %d\n",
					(int)cnt, (int)inputMeta.size(), (int)baseSurface->width,
					(int)pitchBytes_src*m_height_in, int(pitchBytes_src*baseSurface->planeParams.height[0]),
					int(pitchBytes_src*m_height_in*inputMeta.size())
				);
			}

			/// Copy data to the host memory.
			if(baseSurface->colorFormat == NVBUF_COLOR_FORMAT_NV12)
			{
				assert(baseSurface->planeParams.num_planes == 2);

				/// In bytes.
				size_t hPos = pitchBytes_src*(m_height_in*1.5f)*cnt;

				for(uint ch=0; ch<baseSurface->planeParams.num_planes; ch++)
				{
					cv::Size srcPlaneSize = cv::Size(
						baseSurface->planeParams.width[ch], baseSurface->planeParams.height[ch]
					);
					uint pitchBytes_plane = baseSurface->planeParams.pitch[ch];
					uint offset = baseSurface->planeParams.offset[ch];

					if(pitchBytes_plane != pitchBytes_src)
					{
						printf(
							"[ERROR] pitchBytes_plane[%d] != pitchBytes_src[%d]\n", pitchBytes_plane, pitchBytes_src
						);
						assert(pitchBytes_plane == pitchBytes_src);
					}

					if(m_debugVerboseLevel > 3)
					{
						cout << "[FrameInputTexture::uploadDeviceData2DeviceMemories] (hPos, offset) = (" << hPos << ", " << offset << ")" << endl;
					}

					CUDA_CHECK(
						cudaMemcpyAsync(
							reinterpret_cast<uint8_t*>(m_idata_host) + hPos + offset,
							reinterpret_cast<uint8_t*>(baseSurface->dataPtr) + offset,
							pitchBytes_src*srcPlaneSize.height, cudaMemcpyHostToHost, m_cuStream
						)
					);
				}

				/// Src frame.
				{
					/// Create an empty NV12 image.
					m_srcFramesAtHost[it->first] = cv::Mat::zeros(m_height_in*1.5f, m_width_in, CV_8UC1);

					cv::Mat y_ch = cv::Mat(
						m_height_in, m_width_in, CV_8UC1,
						reinterpret_cast<uint8_t*>(m_idata_host) + hPos + baseSurface->planeParams.offset[0],
						pitchBytes_src
					);
					cv::Mat uv_ch = cv::Mat(
						m_height_in*0.5, m_width_in, CV_8UC1,
						reinterpret_cast<uint8_t*>(m_idata_host) + hPos + baseSurface->planeParams.offset[1],
						pitchBytes_src
					);

					y_ch.copyTo(m_srcFramesAtHost[it->first](cv::Rect(0,0,m_width_in,m_height_in)));
					uv_ch.copyTo(m_srcFramesAtHost[it->first](cv::Rect(0,m_height_in,m_width_in,m_height_in*0.5)));
				}

			}
			else if(baseSurface->colorFormat == NVBUF_COLOR_FORMAT_RGBA)
			{
				/// In bytes.
				size_t hPos = pitchBytes_src*m_height_in*cnt;

				CUDA_CHECK(
					cudaMemcpyAsync(
						reinterpret_cast<uint8_t*>(m_idata_host) + hPos, baseSurface->dataPtr,
						pitchBytes_src*srcImageSize.height, cudaMemcpyHostToHost, m_cuStream
					)
				);
				/// Src frame.
				m_srcFramesAtHost[it->first] = cv::Mat(
					m_height_in, m_width_in, CV_8UC4, reinterpret_cast<uint8_t*>(m_idata_host) + hPos, pitchBytes_src
				);
			}
			else
			{
				std::runtime_error("Invalid color format!");
			}
		}

		cnt = 0;
		for(auto it = inputMeta.begin(); it != inputMeta.end(); ++it, ++cnt)
		{
			const NvBufSurfaceParams *baseSurface = it->second->bufferList[0];
			const cv::Size srcImageSize = cv::Size(baseSurface->width, baseSurface->height);
			const uint pitchBytes_src = baseSurface->pitch;

			/// Copy data to the host memory.
			if(baseSurface->colorFormat == NVBUF_COLOR_FORMAT_NV12)
			{
				assert(baseSurface->planeParams.num_planes == 2);

				size_t dPos = m_idataPL_d_pitch*(m_height_in*1.5f)*cnt;

				for(uint ch=0; ch<baseSurface->planeParams.num_planes; ch++)
				{
					cv::Size srcPlaneSize = cv::Size(
						baseSurface->planeParams.width[ch], baseSurface->planeParams.height[ch]
					);
					uint pitchBytes_plane = baseSurface->planeParams.pitch[ch];
					uint offset = baseSurface->planeParams.offset[ch];

					assert(pitchBytes_plane == pitchBytes_src);

					CUDA_CHECK(
						cudaMemcpy2DAsync(
							reinterpret_cast<uint8_t*>(m_idataPL_d) + dPos + m_idataPL_d_pitch*m_height_in*ch,
							m_idataPL_d_pitch, reinterpret_cast<uint8_t*>(baseSurface->dataPtr) + offset,
							pitchBytes_plane, pitchBytes_plane, srcPlaneSize.height, cudaMemcpyHostToDevice, m_cuStream
						)
					);
				}
			}
			else if(baseSurface->colorFormat == NVBUF_COLOR_FORMAT_RGBA)
			{
				size_t dPos = m_idataPL_d_pitch*m_height_in*cnt;

				CUDA_CHECK(
					cudaMemcpy2DAsync(
						reinterpret_cast<uint8_t*>(m_idataPL_d) + dPos, m_idataPL_d_pitch, baseSurface->dataPtr,
						pitchBytes_src, pitchBytes_src, srcImageSize.height, cudaMemcpyHostToDevice, m_cuStream
					)
				);
			}
			else
			{
				std::runtime_error("Invalid color format!");
			}
		}

		CUDA_CHECK( cudaStreamSynchronize(m_cuStream) );
		CUDA_CHECK( cudaPeekAtLastError() );

	#endif

	m_srcNvFrames = inputMeta;

}

/** @brief Create texture objects for NV12 frames in the current batch. */
void FrameInputTexture::createTexObjInBatch_NV12()
{
	#if defined(VIZ_DEBUG)
		/// Destroy every cuda texture object in debug mode.
		destroyTextureObjects();
	#else
		/// Otherwise clear streamid addr mapping.
		m_pTex_NV12.clear();
		/// If texture addr map is full, clean up elements such that it has enough space to accommodate the current batch.
		if (int(m_texInputAddrMap_NV12.size()) > m_maxTexAddrSize)
		{
			if (m_debugVerboseLevel > 2)
			{
				printf(
					"[FrameInputTexture::createTexObjInBatch_NV12] m_texInputAddrMap_NV12 size exceeds limit: %lu / %d\n",
					m_texInputAddrMap_NV12.size(), m_maxTexAddrSize
				);
			}
			cleanUpTextureObjects_NV12(
				int(m_texInputAddrMap_NV12.size()) - m_maxTexAddrSize + int(m_srcNvFrames.size())
			);
		}
	#endif

	uint videoStramInd = 0;
	for(auto it = m_srcNvFrames.begin(); it != m_srcNvFrames.end(); ++it, ++videoStramInd)
	{
		const NvMOTStreamId streamID = it->first;
		const NvBufSurfaceParams *baseSurface = it->second->bufferList[0];

		if (m_texInputAddrMap_NV12.find(baseSurface) == m_texInputAddrMap_NV12.end())
		{
			#if defined(VIZ_DEBUG)
				size_t dPos = m_idataPL_d_pitch*(m_height_in*1.5f)*videoStramInd;

				m_resDesc_luma.res.pitch2D.devPtr = reinterpret_cast<uint8_t*>(m_idataPL_d) + dPos;
				m_resDesc_chroma.res.pitch2D.devPtr = reinterpret_cast<uint8_t*>(m_resDesc_luma.res.pitch2D.devPtr) + m_idataPL_d_pitch*m_height_in;

				m_resDesc_luma.res.pitch2D.pitchInBytes = m_idataPL_d_pitch;
				m_resDesc_chroma.res.pitch2D.pitchInBytes = m_idataPL_d_pitch;
			#else
				#if defined(__aarch64__)
					if(m_integrated_gpu)
					{
						/// Register EGL-mapped image to CUDA.
						if (baseSurface->mappedAddr.eglImage == NULL)
						{
							throw std::runtime_error("EGL mapped image is null");
						}

						/// EGLImage is in GPU memory address.
						EGLImageKHR eglimage_src = baseSurface->mappedAddr.eglImage;
						/// Register EGLImage to graphics resource.
						CUDA_CHECK(
							cuGraphicsEGLRegisterImage(
								&m_graphicsResourceAddrMap[baseSurface],
								eglimage_src, CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY
							)
						);
						/// Map to CUeglFrame, for direct CUDA use.
						CUeglFrame eglFrame = {};
						CUDA_CHECK(
							cuGraphicsResourceGetMappedEglFrame(
								&eglFrame, m_graphicsResourceAddrMap.at(baseSurface), 0, 0
							)
						);
						m_resDesc_luma.res.pitch2D.devPtr = reinterpret_cast<uint8_t*>(eglFrame.frame.pPitch[0]);
						m_resDesc_chroma.res.pitch2D.devPtr = reinterpret_cast<uint8_t*>(eglFrame.frame.pPitch[1]);
						/// Save eglFrame in map.
						m_eglFrameAddrMap[baseSurface] = std::move(eglFrame);

						m_resDesc_luma.res.pitch2D.pitchInBytes = baseSurface->planeParams.pitch[0];
						m_resDesc_chroma.res.pitch2D.pitchInBytes = baseSurface->planeParams.pitch[1];
					}
					else
				#endif
				{
					m_resDesc_luma.res.pitch2D.devPtr = reinterpret_cast<uint8_t*>(baseSurface->dataPtr) + baseSurface->planeParams.offset[0];
					m_resDesc_chroma.res.pitch2D.devPtr = reinterpret_cast<uint8_t*>(baseSurface->dataPtr) + baseSurface->planeParams.offset[1];

					m_resDesc_luma.res.pitch2D.pitchInBytes = baseSurface->planeParams.pitch[0];
					m_resDesc_chroma.res.pitch2D.pitchInBytes = baseSurface->planeParams.pitch[1];
				}
			#endif
			/// Texture objects are created only when new surface arrived, not every batch.
			TextureObjects texObj;
			m_texInputAddrMap_NV12[baseSurface] = std::move(texObj);
			CUDA_CHECK(
				cudaCreateTextureObject(
					&(m_texInputAddrMap_NV12[baseSurface].luma), &m_resDesc_luma, &m_texDesc, nullptr
				)
			);
			CUDA_CHECK(
				cudaCreateTextureObject(
					&(m_texInputAddrMap_NV12[baseSurface].chroma), &m_resDesc_chroma, &m_texDesc, nullptr
				)
			);
		}

		/// Sanity check for textureObjects format.
		if (m_texInputAddrMap_NV12.find(baseSurface) == m_texInputAddrMap_NV12.end())
		{
			throw std::runtime_error("TextureObjects not found");
		}

		/// Acquire (streamID, TextureObjects) mapping.
		m_pTex_NV12[streamID] = &(m_texInputAddrMap_NV12.at(baseSurface));
	}
}

/**
 * @brief Remove cached NV12 texture objects.
 * @param[in] cleanUpNum Number of cached texture objects to remove.
 */
void FrameInputTexture::cleanUpTextureObjects_NV12(int cleanUpNum)
{
	/// If cleanup num is larger than map size, empty the map.
	if(cleanUpNum > (int)(m_texInputAddrMap_NV12.size()))
	{
		cleanUpNum = m_texInputAddrMap_NV12.size();
	}

	/// Delete the first cleanUpNum entries in the map.
	int cleaned = 0;
	for(auto it = m_texInputAddrMap_NV12.begin(); it != m_texInputAddrMap_NV12.end() && cleaned < cleanUpNum; cleaned++)
	{
		#if defined(__aarch64__)
			if(m_integrated_gpu)
			{
				/// Unregister corresponding Egl mapping.
				auto jt = m_graphicsResourceAddrMap.find(it->first);
				if(jt != m_graphicsResourceAddrMap.end())
				{
					CUDA_CHECK( cuGraphicsUnregisterResource(jt->second) );
					m_graphicsResourceAddrMap.erase(jt);
				}
				else
				{
					if (m_debugVerboseLevel > 2)
					{
						printf(
							"[FrameInputTexture::cleanUpTextureObjects_NV12] Address found in m_texInputAddrMap_NV12 but not int m_graphicsResourceAddrMap!\n"
						);
					}
				}
			}
		#endif
		CUDA_CHECK( cudaDestroyTextureObject(it->second.luma) );
		CUDA_CHECK( cudaDestroyTextureObject(it->second.chroma) );
		it = m_texInputAddrMap_NV12.erase(it);

	}
}

/** @brief Create texture objects for BGRA or grayscale frames. */
void FrameInputTexture::createTexObjInBatch()
{
	#if defined(VIZ_DEBUG)
		/// Destroy every cuda texture object in debug mode.
		destroyTextureObjects();
	#else
		/// Otherwise clear streamid addr mapping.
		m_pTex.clear();
		/// If texture addr map is full, clean up elements such that it has enough space to accommodate the current batch.
		if (int(m_texInputAddrMap.size()) > m_maxTexAddrSize)
		{
			if (m_debugVerboseLevel > 2)
			{
				printf(
					"[createTexObjInBatch] m_texInputAddrMapsize exceeds limit: %lu / %d\n",
					m_texInputAddrMap.size(), m_maxTexAddrSize
				);
			}
			cleanUpTextureObjects(int(m_texInputAddrMap.size()) - m_maxTexAddrSize + int(m_srcNvFrames.size()));
		}
	#endif

	uint videoStramInd = 0;
	for(auto it = m_srcNvFrames.begin(); it != m_srcNvFrames.end(); ++it, ++videoStramInd)
	{
		const NvMOTStreamId streamID = it->first;
		const NvBufSurfaceParams *baseSurface = it->second->bufferList[0];

		if (m_texInputAddrMap.find(baseSurface) == m_texInputAddrMap.end())
		{
			#if defined(VIZ_DEBUG)
				size_t dPos = m_idataPL_d_pitch*m_height_in*videoStramInd;
				m_resDesc.res.pitch2D.devPtr = reinterpret_cast<uint8_t*>(m_idataPL_d) + dPos;
				m_resDesc.res.pitch2D.pitchInBytes = m_idataPL_d_pitch;
			#else
				#if defined(__aarch64__)
					if(m_integrated_gpu)
					{
						/// Register EGLImage to CUDA.
						if (baseSurface->mappedAddr.eglImage == NULL)
						{
							throw std::runtime_error("EGL mapped image is null");
						}

						/// EGLImage is in GPU memory address.
						EGLImageKHR eglimage_src = baseSurface->mappedAddr.eglImage;

						/// Register EGLImage to graphics resource.
						CUDA_CHECK(
							cuGraphicsEGLRegisterImage(
								&m_graphicsResourceAddrMap[baseSurface],
								eglimage_src, CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY
							)
						);
						/// Map to CUeglFrame, for direct CUDA use.
						CUeglFrame eglFrame = {};
						CUDA_CHECK(
							cuGraphicsResourceGetMappedEglFrame(
								&eglFrame, m_graphicsResourceAddrMap.at(baseSurface), 0, 0
							)
						);
						m_resDesc.res.pitch2D.devPtr = reinterpret_cast<uint8_t*>(eglFrame.frame.pPitch[0]);
						/// Store the mapped EGL frame for this surface.
						m_eglFrameAddrMap[baseSurface] = std::move(eglFrame);
						m_resDesc.res.pitch2D.pitchInBytes = baseSurface->planeParams.pitch[0];
					}
					else
				#endif
				{
					m_resDesc.res.pitch2D.devPtr = reinterpret_cast<uint8_t*>(baseSurface->dataPtr);
					m_resDesc.res.pitch2D.pitchInBytes = baseSurface->pitch;
				}
			#endif

			if(m_channels_in == 1)
			{
				m_resDesc.res.pitch2D.desc = cudaCreateChannelDesc<uchar1>();
			}
			else if(m_channels_in == 4)
			{
				m_resDesc.res.pitch2D.desc = cudaCreateChannelDesc<uchar4>();
			}
			else
			{
				throw std::runtime_error("Invalid image channels");
			}

			/// Texture objects are created only when new surface arrived, not every batch.
			CUDA_CHECK( cudaCreateTextureObject(&m_texInputAddrMap[baseSurface], &m_resDesc, &m_texDesc, nullptr) );
		}

		/// Sanity check for textureObjects format.
		if (m_texInputAddrMap.find(baseSurface) == m_texInputAddrMap.end())
		{
			throw std::runtime_error("TextureObjects not found");
		}

		/// Acquire (streamID, TextureObjects) mapping.
		m_pTex[streamID] = &(m_texInputAddrMap.at(baseSurface));
	}
}

/**
 * @brief Remove cached BGRA or grayscale texture objects.
 * @param[in] cleanUpNum Number of cached texture objects to remove.
 */
void FrameInputTexture::cleanUpTextureObjects(int cleanUpNum)
{
	/// If cleanup num is larger than map size, empty the map.
	if(cleanUpNum > (int)(m_texInputAddrMap.size()))
	{
		cleanUpNum = m_texInputAddrMap.size();
	}

	/// Delete the first cleanUpNum entries in the map.
	int cleaned = 0;
	for(auto it = m_texInputAddrMap.begin(); it != m_texInputAddrMap.end() && cleaned < cleanUpNum; cleaned++)
	{
		#if defined(__aarch64__)
			if(m_integrated_gpu)
			{
				/// Unregister corresponding Egl mapping.
				auto jt = m_graphicsResourceAddrMap.find(it->first);
				if(jt != m_graphicsResourceAddrMap.end())
				{
					CUDA_CHECK( cuGraphicsUnregisterResource(jt->second) );
					m_graphicsResourceAddrMap.erase(jt);
				}
				else
				{
					if (m_debugVerboseLevel > 2)
					{
						printf(
							"[cleanUpTextureObjects] Address found in m_texInputAddrMap_NV12 but not int m_graphicsResourceAddrMap!\n"
						);
					}
				}
			}
		#endif
		CUDA_CHECK( cudaDestroyTextureObject(it->second) );
		it = m_texInputAddrMap.erase(it);
	}
}

/**
 * @brief Get a stream texture-object pointer.
 * @param[in] streamId Video stream identifier.
 */
cudaTextureObject_t *FrameInputTexture::getTexturePointer(NvMOTStreamId streamId)
{
	if (m_pTex.find(streamId) != m_pTex.end())
	{
		return m_pTex.at(streamId);
	}
	else
	{
		return nullptr;
	}
}

/**
 * @brief Get a stream NV12 texture-object pair.
 * @param[in] streamId Video stream identifier.
 */
TextureObjects *FrameInputTexture::getTexturePointer_NV12(NvMOTStreamId streamId)
{
	if (m_pTex_NV12.find(streamId) != m_pTex_NV12.end())
	{
		return m_pTex_NV12.at(streamId);
	}
	else
	{
		return nullptr;
	}
}

/**
 * @brief Construct a FrameTransformTexture instance.
 * @param[in] cudaBlockSize_2D CUDA block width and height for two-dimensional kernels.
 */
FrameTransformTexture::FrameTransformTexture(const int cudaBlockSize_2D) : m_cudaBlockSize_2D(cudaBlockSize_2D)
{
	m_frameInput = std::make_shared<FrameInputTexture>();

	m_debugVerboseLevel = 0;
	m_lastInd = 0;
	m_numFramesInBatchMax = 0;
	m_maxTargetsToTrack = 0;
	m_isInit_common = false;
}

/** @brief Release resources owned by the FrameTransformTexture instance. */
FrameTransformTexture::~FrameTransformTexture()
{
	if (m_isInit_common)
	{
		deAllocateOutputMemories_common();
		CUDA_CHECK(cudaStreamDestroy(m_cuStream));
	}

}

/**
 * @brief Initialize shared frame-transform resources.
 * @param[in] debugVerboseLevel Debug verbosity level.
 * @param[in] maxStreamNum Maximum number of input streams.
 * @param[in] maxBufSurfAddrSize Maximum number of cached surface-buffer addresses.
 * @param[in] maxTargetNum Maximum number of targets in a batch.
 * @param[in] srcImgSize Input image dimensions.
 * @param[in] srcImgPitch Input image pitch in bytes.
 * @param[in] colorFormatType Input surface color format.
 */
void FrameTransformTexture::initialize_common(
	const int debugVerboseLevel, const int maxStreamNum, const int maxBufSurfAddrSize, const int maxTargetNum,
	const NvMotDataTypes::Size srcImgSize, const uint srcImgPitch, const NvBufSurfaceColorFormat colorFormatType
)
{
	/// Initialize the input texture manager.
	m_frameInput->initialize(
		debugVerboseLevel, maxStreamNum, maxBufSurfAddrSize, srcImgSize, srcImgPitch, colorFormatType
	);

	if(maxTargetNum <= 0)
	{
		throw std::runtime_error("[FrameTransformTexture::initialize] Make sure maxTargetNum > 0");
	}

	CUDA_CHECK( cudaStreamCreateWithFlags(&m_cuStream, cudaStreamNonBlocking) );

	m_debugVerboseLevel = debugVerboseLevel;
	m_maxTargetsToTrack = maxTargetNum;

	/// Note: The readMode of cudaReadModeNormalizedFloat always make the value normalized so that the range is either [-1, 1] or [0, 1], where the max possible value to be mapped to 1.
	/// Thus, need to scale it to preserve the value float: 1; int8: 1<<8; int16: 1<<16.
	m_scaleFactor = 1<<8;

	/// The memory size of source images should be multiples of the unit CUDA block, which is m_cudaBlockSize_2D-by-m_cudaBlockSize_2D.
	if(m_frameInput->getWidthIn() % m_cudaBlockSize_2D != 0 )
	{
		std::string msg = "tracker-width(" + std::to_string(m_frameInput->getWidthIn()) + ") must be a multiple of "
						+ std::to_string(m_cudaBlockSize_2D);
		throw std::runtime_error(msg);
	}
	else if(m_frameInput->getHeightIn() % int(m_cudaBlockSize_2D) != 0)
	{
		std::string msg = "tracker-height(" + std::to_string(m_frameInput->getHeightIn()) +  ") must be a multiple of "
						+ std::to_string(int(m_cudaBlockSize_2D));
		throw std::runtime_error(msg);
	}

	allocateOutputMemories_common();

	m_isInit_common = true;
}

/** @brief Allocate common frame-transform output memory. */
void FrameTransformTexture::allocateOutputMemories_common()
{
	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &m_rois_d ), m_maxTargetsToTrack * sizeof(float4) ));
	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &m_dstOffsets_d ), m_maxTargetsToTrack * sizeof(uint2) ));

	CUDA_CHECK( cudaMallocHost( reinterpret_cast<void**>( &m_rois_h ), m_maxTargetsToTrack * sizeof(float4) ));
	CUDA_CHECK( cudaMallocHost( reinterpret_cast<void**>( &m_dstOffsets_h ), m_maxTargetsToTrack * sizeof(uint2) ));
}

/** @brief Release common frame-transform output memory. */
void FrameTransformTexture::deAllocateOutputMemories_common()
{
	CUDA_CHECK( cudaFree(m_rois_d) );
	CUDA_CHECK( cudaFree(m_dstOffsets_d) );

	CUDA_CHECK( cudaFreeHost(m_rois_h) );
	CUDA_CHECK( cudaFreeHost(m_dstOffsets_h) );
}

/**
 * @brief Update texture objects from an input frame batch.
 * @param[in] inputMeta Source frames keyed by stream ID.
 */
void FrameTransformTexture::updateTextureObject(const std::map<NvMOTStreamId, NvMOTFrame*> &inputMeta)
{
	m_frameInput->create_batch(inputMeta);

	if (m_frameInput->getColorFormat() == NvBufSurfaceColorFormat::NVBUF_COLOR_FORMAT_NV12)
	{
		m_frameInput->createTexObjInBatch_NV12();
	}
	else if (m_frameInput->getColorFormat() == NvBufSurfaceColorFormat::NVBUF_COLOR_FORMAT_RGBA)
	{
		m_frameInput->createTexObjInBatch();
	}
	else
	{
		throw std::runtime_error("[FAILED] Invalid color format");
	}
}
