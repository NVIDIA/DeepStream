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

#if defined VIZ_DEBUG_REID || defined VIZ_DEBUG_SEG || defined VIZ_DEBUG_POSE
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#endif

#include "modules/utility/debug_common.hpp"
#include "modules/utility/cuda_common.hpp"
#include "modules/ReID/ReIDFrameTransformTexture.hpp"
#include "modules/FrameTransform/cuda_Gaussian.cuh"
#include "modules/cuDCFv2/helper_math.h"

#if defined(__aarch64__) && defined(USE_GAUSSIAN_SMOOTHING_FOR_SCALING)
/// Without this launch_bounds, this kernel may have launch failures with `too many resources requested for launch at..` error in gdb on Nano.
__launch_bounds__(1024)
#endif
/**
 * @brief Crop and transform four-channel texture regions.
 * @param[in] texInput Source four-channel CUDA texture.
 * @param[out] odata Destination network-input tensor.
 * @param[in] oPitch Pitched width of each output row or plane.
 * @param[in] width_out Width of each output patch in pixels.
 * @param[in] outBatchingGrid Grid dimensions used to arrange output patches.
 * @param[in] height_out Height of each output patch in pixels.
 * @param[in] width_in Width of the source frame in pixels.
 * @param[in] height_in Height of the source frame in pixels.
 * @param[in] rois Source region of interest for each target.
 * @param[in] targetNum Number of target patches to transform.
 * @param[in] colorFormat Required output color ordering.
 * @param[in] channels Number of output color channels.
 * @param[in] dimOrder Output tensor dimension order selector.
 * @param[in] texScale Scale that converts texture samples to pixel values.
 * @param[in] outScale Scale applied to each output value.
 * @param[in] offset Per-channel offset subtracted before scaling.
 */
__global__ void cuda_ReIDFrameTransform_4CH(
    cudaTextureObject_t texInput, float *odata, uint oPitch, int width_out, uint2 outBatchingGrid,
    int height_out, int width_in, int height_in, float4 rois[], const int targetNum,
    const NvBufSurfaceColorFormat colorFormat, const int channels, const int dimOrder = 0,
    const float texScale = 1.0, const float outScale = 1.0, const float3 offset = {0.0, 0.0, 0.0}
)
{
    int xid = blockIdx.x * blockDim.x + threadIdx.x;
    int yid = blockIdx.y * blockDim.y + threadIdx.y;

    int2 bboxLoc;

    bboxLoc.y = int(yid/height_out);
    bboxLoc.x = int(xid/width_out);

    int bboxId = bboxLoc.y*outBatchingGrid.x + bboxLoc.x;

    yid -= bboxLoc.y*height_out;
    xid -= bboxLoc.x*width_out;

    if (bboxId >= targetNum)
		return;

    if (xid >= 0 && xid < width_out && yid >= 0 && yid < height_out)
    {
		float2 pt;

		/// Starting point in normalized coord.
		pt.x = (rois[bboxId].x)/(float)width_in
			+ (((xid+0.5f)/(float)width_out) // Steps.
			*((rois[bboxId].w)/(float)width_in)); // Total increment in normalized coord.
		/// Starting point in normalized coord.
		pt.y = (rois[bboxId].y)/(float)height_in
			+ (((yid+0.5f)/(float)height_out) // Steps.
			*((rois[bboxId].z)/(float)height_in)); // Total increment in normalized coord.

		// Note: to allow filtering mode, the textures should be configured to return floating-point data.
		// tex2D<float> is used even for uint8 or uint16 input images.
		#ifdef USE_GAUSSIAN_SMOOTHING_FOR_SCALING
			float4 valFloat = cuda_GaussianSmoothing_3x3<float4>(texInput, pt);
		#else
			float4 valFloat = tex2D<float4>(texInput, pt.x, pt.y);
		#endif

		/// Convert the color, leaving out alpha.
		float3 rgb{
			max(0.f, min(255.f, valFloat.x*texScale+0.5f)),
			max(0.f, min(255.f, valFloat.y*texScale+0.5f)),
			max(0.f, min(255.f, valFloat.z*texScale+0.5f))
		};

		int ind;
		/// NCHW.
		if (dimOrder == 0)
		{
			/// To place the image patch according to the bbox index.
			ind = bboxId * height_out * oPitch * channels
				+ yid * oPitch // Y location within the image patch.
				+ xid;
			if(colorFormat == NVBUF_COLOR_FORMAT_RGB){
				odata[ind] = (rgb.x - offset.x) * outScale;
				odata[ind + height_out * oPitch] = (rgb.y - offset.y) * outScale;
				odata[ind + height_out * oPitch * 2] = (rgb.z - offset.z) * outScale;
			}
			else if(colorFormat == NVBUF_COLOR_FORMAT_BGR){
				odata[ind] = (rgb.z - offset.x) * outScale;
				odata[ind + height_out * oPitch] = (rgb.y - offset.y) * outScale;
				odata[ind + height_out * oPitch * 2] = (rgb.x - offset.z) * outScale;
			}
		}
		/// NHWC.
		else if (dimOrder == 1)
		{
			/// To place the image patch according to the bbox index.
			ind = (bboxId * height_out * oPitch
				+ yid * oPitch + xid) * channels; // Y location within the image patch.

			if(colorFormat == NVBUF_COLOR_FORMAT_RGB){
				odata[ind] = (rgb.x - offset.x) * outScale;
				odata[ind + 1] = (rgb.y - offset.y) * outScale;
				odata[ind + 2] = (rgb.z - offset.z) * outScale;
			}
			else if(colorFormat == NVBUF_COLOR_FORMAT_BGR){
				/// TODO: Check the offset order.
				odata[ind] = (rgb.z - offset.x) * outScale;
				odata[ind + 1] = (rgb.y - offset.y) * outScale;
				odata[ind + 2] = (rgb.x - offset.z) * outScale;
			}
		}
	}
}

/// Color conversion equation.
#define PUT_YUV2RGB_FN(suby, subuv, scaley, e, cdB, aeB, d) \
__device__ __inline__ float3 yuv2rgb (float y, float u, float v, float low, float high) \
{ \
	float3 rgb; \
	rgb.x = max(min((y - suby) * scaley + e * (v - subuv), high), low); \
	rgb.y = max(min((y - suby) * scaley - cdB * (u - subuv) - aeB * (v - subuv), high), low); \
	rgb.z = max(min((y - suby) * scaley + d  * (u - subuv), high), low); \
	return rgb; \
}
PUT_YUV2RGB_FN (16.0f, 128.0f, 1.1644f, 1.5960f, 0.3918f, 0.8130f, 2.0172f)

#if defined(__aarch64__) && defined(USE_GAUSSIAN_SMOOTHING_FOR_SCALING)
/// Without this launch_bounds, this kernel may have launch failures with `too many resources requested for launch at..` error in gdb on Nano.
__launch_bounds__(1024)
#endif
/**
 * @brief Crop and transform NV12 texture regions.
 * @param[in] luma Source luma-plane CUDA texture.
 * @param[in] chroma Source chroma-plane CUDA texture.
 * @param[out] odata Destination network-input tensor.
 * @param[in] oPitch Pitched width of each output row or plane.
 * @param[in] width_out Width of each output patch in pixels.
 * @param[in] outBatchingGrid Grid dimensions used to arrange output patches.
 * @param[in] height_out Height of each output patch in pixels.
 * @param[in] width_in Width of the source frame in pixels.
 * @param[in] height_in Height of the source frame in pixels.
 * @param[in] rois Source region of interest for each target.
 * @param[in] targetNum Number of target patches to transform.
 * @param[in] colorFormat Required output color ordering.
 * @param[in] channels Number of output color channels.
 * @param[in] dimOrder Output tensor dimension order selector.
 * @param[in] texScale Scale that converts texture samples to pixel values.
 * @param[in] outScale Scale applied to each output value.
 * @param[in] offset Per-channel offset subtracted before scaling.
 */
__global__ void cuda_ReIDFrameTransform_batch_NV12(
    cudaTextureObject_t luma, cudaTextureObject_t chroma, float *odata, uint oPitch, int width_out,
    uint2 outBatchingGrid, int height_out, int width_in, int height_in, float4 rois[], const int targetNum,
    const NvBufSurfaceColorFormat colorFormat, const int channels, const int dimOrder = 0,
    const float texScale = 1.0, const float outScale = 1.0, const float3 offset = {0.0, 0.0, 0.0}
)
{
    int xid = blockIdx.x * blockDim.x + threadIdx.x;
    int yid = blockIdx.y * blockDim.y + threadIdx.y;

    int2 bboxLoc;

    bboxLoc.y = int(yid/height_out);
    bboxLoc.x = int(xid/width_out);

    int bboxId = bboxLoc.y*outBatchingGrid.x + bboxLoc.x;

    yid -= bboxLoc.y*height_out;
    xid -= bboxLoc.x*width_out;

    if (bboxId >= targetNum)
		return;

    if (xid >= 0 && xid < width_out && yid >= 0 && yid < height_out)
    {
		float2 pt, uv;
		float y;

		/// For Luma.
		{
			/// Starting point in normalized coord.
			pt.x = (rois[bboxId].x)/(float)width_in
				+ (((xid+0.5f)/(float)width_out) // Steps.
				*((rois[bboxId].w)/(float)width_in)); // Total increment in normalized coord.
			/// Starting point in normalized coord.
			pt.y = (rois[bboxId].y)/(float)height_in
				+ (((yid+0.5f)/(float)height_out) // Steps.
				*((rois[bboxId].z)/(float)height_in)); // Total increment in normalized coord.

			// Note: to allow filtering mode, the textures should be configured to return floating-point data.
			// tex2D<float> is used even for uint8 or uint16 input images.
			#ifdef USE_GAUSSIAN_SMOOTHING_FOR_SCALING
				y = cuda_GaussianSmoothing_3x3<float>(luma, pt);
			#else
				y = tex2D<float>(luma, pt.x, pt.y);
			#endif
		}

		/// For Chroma.
		/// Luma coord (xid, yid) matches Chroma (xid/2, yid/2).
		{
			/// Starting point in normalized coord.
			pt.x = (rois[bboxId].x)/(float)width_in
				+ (((xid/2+0.5f)/(float)(width_out*0.5)) // Steps.
				*((rois[bboxId].w)/(float)width_in) // Total increment in normalized coord.
				);
			/// Starting point in normalized coord.
			pt.y = (rois[bboxId].y)/(float)height_in
				+ (((yid/2+0.5f)/(float)(height_out*0.5)) // Steps.
				*((rois[bboxId].z)/(float)height_in) // Total increment in normalized coord.
				);

			// Note: to allow filtering mode, the textures should be configured to return floating-point data.
			// tex2D<float> is used even for uint8 or uint16 input images.
			#ifdef USE_GAUSSIAN_SMOOTHING_FOR_SCALING
				uv = cuda_GaussianSmoothing_3x3<float2>(chroma, pt);
			#else
				uv = tex2D<float2>(chroma, pt.x, pt.y);
			#endif
		}

		/// Color conversion.
		float3 rgb = yuv2rgb (y * texScale, uv.x * texScale, uv.y * texScale, 0.0, 255.0);
		int ind;
		/// NCHW.
		if (dimOrder == 0)
		{
			/// To place the image patch according to the bbox index.
			ind = bboxId * height_out * oPitch * channels
				+ yid * oPitch // Y location within the image patch.
				+ xid;
			if(colorFormat == NVBUF_COLOR_FORMAT_RGB){
				odata[ind] = (rgb.x - offset.x) * outScale;
				odata[ind + height_out * oPitch] = (rgb.y - offset.y) * outScale;
				odata[ind + height_out * oPitch * 2] = (rgb.z - offset.z) * outScale;
			}
			else if(colorFormat == NVBUF_COLOR_FORMAT_BGR){
				odata[ind] = (rgb.z - offset.x) * outScale;
				odata[ind + height_out * oPitch] = (rgb.y - offset.y) * outScale;
				odata[ind + height_out * oPitch * 2] = (rgb.x - offset.z) * outScale;
			}
		}
		/// NHWC.
		else if (dimOrder == 1)
		{
			/// To place the image patch according to the bbox index.
			ind = (bboxId * height_out * oPitch
				+ yid * oPitch + xid) * channels; // Y location within the image patch.

			if(colorFormat == NVBUF_COLOR_FORMAT_RGB){
				odata[ind] = (rgb.x - offset.x) * outScale;
				odata[ind + 1] = (rgb.y - offset.y) * outScale;
				odata[ind + 2] = (rgb.z - offset.z) * outScale;
			}
			else if(colorFormat == NVBUF_COLOR_FORMAT_BGR){
				/// TODO: Check the offset order.
				odata[ind] = (rgb.z - offset.x) * outScale;
				odata[ind + 1] = (rgb.y - offset.y) * outScale;
				odata[ind + 2] = (rgb.x - offset.z) * outScale;
			}
		}
	}
}

/** @brief Construct a ReIDFrameTransformTexture instance. */
ReIDFrameTransformTexture::ReIDFrameTransformTexture()
{
    m_width_out = 0;
    m_height_out = 0;
    m_channels_out = 3;
	m_debugVerboseLevel = 0;
	m_isInit = false;
}

/** @brief Release resources owned by the ReIDFrameTransformTexture instance. */
ReIDFrameTransformTexture::~ReIDFrameTransformTexture()
{
	if (m_isInit)
	{
		deAllocateOutputMemories();
	}
}

/**
 * @brief Configure texture-based cropping and allocate output buffers.
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
void ReIDFrameTransformTexture::initialize(
    const int debugVerboseLevel, const int maxStreamNum, const int maxBufSurfAddrSize, const int maxTargetNum,
    const NvMotDataTypes::Size srcImgSize, const uint srcImgPitch, const NvBufSurfaceColorFormat colorFormatType,
    const NvMotDataTypes::Size dstImgSize, const NvBufSurfaceColorFormat dstColorFormatType, const int dimOrder
)
{
    FrameTransformTexture::initialize_common(
        debugVerboseLevel, maxStreamNum, maxBufSurfAddrSize, maxTargetNum, srcImgSize, srcImgPitch, colorFormatType
    );
	m_debugVerboseLevel = debugVerboseLevel;

    if(colorFormatType == NVBUF_COLOR_FORMAT_BGRA
		|| colorFormatType == NVBUF_COLOR_FORMAT_RGBA
        || colorFormatType == NVBUF_COLOR_FORMAT_NV12)
    {
        initialize4Output(dstImgSize, dstColorFormatType, dimOrder);
    }
    else
	{
		throw std::runtime_error("Invalid Color Formats");
	}

	m_isInit = true;
}

/**
 * @brief Configure the output layout and allocate patch memory.
 * @param[in] dstInferSize Network input dimensions.
 * @param[in] dstColorFormat Output color format, RGB or BGR.
 * @param[in] dimOrder Output tensor dimension order, 0 for NCHW and 1 for NHWC.
 */
void ReIDFrameTransformTexture::initialize4Output(
    const NvMotDataTypes::Size dstInferSize, const NvBufSurfaceColorFormat dstColorFormat, const int dimOrder
)
{
	/// Create output host memory if not created.
	if(m_width_out != 0 && m_height_out != 0)
	{
		deAllocateOutputMemories();
	}

	if (dstColorFormat != NvBufSurfaceColorFormat::NVBUF_COLOR_FORMAT_BGR &&
		dstColorFormat != NvBufSurfaceColorFormat::NVBUF_COLOR_FORMAT_RGB)
	{
		throw std::runtime_error("Only RGB or BGR format is supported for ReID inference");
	}
	m_dstColorFormatType = dstColorFormat;
	m_width_out = dstInferSize.width;
	m_height_out = dstInferSize.height;

	if (dimOrder != 0 && dimOrder != 1)
	{
		throw std::runtime_error("ReID inference dimension order must be 0 or 1");
	}
	m_dimOrder = dimOrder;

	/// TODO: Decide the output-size check.
	allocateOutputMemories();
}

/** @brief Allocate transform output memory. */
void ReIDFrameTransformTexture::allocateOutputMemories()
{
	CUDA_CHECK(
		cudaMalloc(
			reinterpret_cast<void **>(&m_odata_d),
			m_maxTargetsToTrack * m_channels_out * m_height_out * m_width_out * sizeof(float)
		)
	);

	#if defined VIZ_DEBUG_REID || defined VIZ_DEBUG_SEG || defined VIZ_DEBUG_POSE
		/// Declare the host memory large enough to accommodate the output value in device memory.
		CUDA_CHECK(
			cudaMallocHost(
				reinterpret_cast<void **>(&m_odata_host),
				m_maxTargetsToTrack * m_channels_out * m_height_out * m_width_out * sizeof(float)
			)
		);
		CUDA_CHECK(
			cudaMallocHost(
				reinterpret_cast<void **>(&m_odata_host_nchw),
				m_maxTargetsToTrack * m_channels_out * m_height_out * m_width_out * sizeof(float)
			)
		);
	#endif
}

/** @brief Release transform output memory. */
void ReIDFrameTransformTexture::deAllocateOutputMemories()
{
	CUDA_CHECK( cudaFree(m_odata_d) );
	#if defined VIZ_DEBUG_REID || defined VIZ_DEBUG_SEG || defined VIZ_DEBUG_POSE
		CUDA_CHECK( cudaFreeHost(m_odata_host) );
		CUDA_CHECK( cudaFreeHost(m_odata_host_nchw) );
	#endif
}

/**
 * @brief Crop and rescale target regions from NV12 frames.
 * @param[in] roisInBatch The set of ROIs of interest {videoStreamID, a vector of {bboxInd, bbox} }.
 * @param[in] lastBboxInd Last bbox index.
 * @param[in] outputScaleFactor Scale applied to output pixel values.
 * @param[in] outputOffsets Parameter of each channel's offset.
 * @return Status.
 */
bool ReIDFrameTransformTexture::frameTransformInBatch_NV12(
    const std::map<NvMOTStreamId, std::vector<std::pair<uint, NvMotDataTypes::Rect2f> > > &roisInBatch,
    const uint lastBboxInd, const float outputScaleFactor, const std::vector<float> outputOffsets
)
{
	if (outputOffsets.size() != 3)
	{
		throw std::runtime_error("image channel offsets should be length of 3");
	}

	/// Store how many targets are present in each frame ind.
	std::unordered_map<NvMOTStreamId, int> perStreamTargetNum;

	m_targetNumInd2RoisInBatchMapper.clear();
	m_lastInd = lastBboxInd;
	int targetNumCnt = 0;

	/// For videoStreams from all existing targets.
	for(auto it = roisInBatch.begin(); it != roisInBatch.end(); ++it)
	{
		const NvMOTStreamId &streamID = it->first;

		/// A vector of {trackerInd, bbox}.
		const std::vector<std::pair<uint, NvMotDataTypes::Rect2f> > &rois = roisInBatch.at(streamID);

		for(uint i=0; i<rois.size(); i++, targetNumCnt++)
		{
			const NvMotDataTypes::Rect2f &bbox = rois[i].second;

			m_rois_h[targetNumCnt].x = bbox.min()(0);
			m_rois_h[targetNumCnt].y = bbox.min()(1);
			m_rois_h[targetNumCnt].w = bbox.sizes()(0);
			m_rois_h[targetNumCnt].z = bbox.sizes()(1);

			/// Map the target index to its video stream and per-stream bounding-box index.
			m_targetNumInd2RoisInBatchMapper[targetNumCnt] = std::make_pair(streamID, i);
		}

		/// For videoStreams from existing targets.
		perStreamTargetNum[streamID] = rois.size();
	}

	if((int)targetNumCnt > m_maxTargetsToTrack)
	{
		std::string msg = "Too many objects to track. Max target num = " + std::to_string(m_maxTargetsToTrack) + ", but total rois = " + std::to_string(targetNumCnt) ;
		CLOG(msg.c_str());

		throw std::runtime_error(msg);
	}

	/// Sync host and device.
	CUDA_CHECK(
		cudaMemcpyAsync(m_rois_d, m_rois_h, targetNumCnt * sizeof(float4), cudaMemcpyHostToDevice, m_cuStream)
	);

	int roiInd = 0;

	/// Launch CUDA kernels for processing.
	for(auto it = roisInBatch.begin(); it != roisInBatch.end(); ++it)
	{
		const NvMOTStreamId &streamID = it->first;

		TextureObjects *pTex_NV12 = m_frameInput->getTexturePointer_NV12(streamID);
		if(!pTex_NV12)
		{
			CLOG(
				std::string("[frameTransformInBatch_NV12] !! StreamID[" + std::to_string(streamID) + "] NOT found in pTex_NV12 !!").c_str()
			);
			continue;
		}

		if(perStreamTargetNum[streamID] == 0)
			continue;

		/// Setup execution configuration parameters.
		dim3 Threads(m_cudaBlockSize_2D, m_cudaBlockSize_2D);
		dim3 Blocks(
			(m_width_out + Threads.x - 1) / Threads.x,
			(m_height_out * perStreamTargetNum[streamID] + Threads.x - 1) / Threads.x
		);
		float *odata_d = m_odata_d + m_channels_out * m_width_out * m_height_out * roiInd;

		/// Crop the source image and scale to the dst size.
		cuda_ReIDFrameTransform_batch_NV12<<<Blocks, Threads, 0, m_cuStream>>>(
			pTex_NV12->luma, pTex_NV12->chroma, odata_d, m_width_out, m_width_out, uint2{1, (uint)m_maxTargetsToTrack},
			m_height_out, m_frameInput->getWidthIn(), m_frameInput->getHeightIn(), &m_rois_d[roiInd],
			perStreamTargetNum[streamID], m_dstColorFormatType, m_channels_out, m_dimOrder, m_scaleFactor,
			outputScaleFactor, float3{outputOffsets[0], outputOffsets[1], outputOffsets[2]}
		);

		roiInd += perStreamTargetNum[streamID];
	}

	CUDA_CHECK( cudaStreamSynchronize(m_cuStream) );
	CUDA_CHECK( cudaPeekAtLastError() );

	return true;
}

/**
 * @brief Crop and rescale target regions in a frame batch.
 * @param[in] roisInBatch The set of ROIs of interest {videoStreamID, a vector of {bboxInd, bbox} }.
 * @param[in] lastBboxInd Last bbox index.
 * @param[in] outputScaleFactor Scale applied to output pixel values.
 * @param[in] outputOffsets Parameter of each channel's offset.
 * @return Status.
 */
bool ReIDFrameTransformTexture::frameTransformInBatch(
    const std::map<NvMOTStreamId, std::vector<std::pair<uint, NvMotDataTypes::Rect2f> > > &roisInBatch,
    const uint lastBboxInd, const float outputScaleFactor, const std::vector<float> outputOffsets
)
{
	if (outputOffsets.size() != 3)
	{
		throw std::runtime_error("image channel offsets should be length of 3");
	}

	/// Store how many targets are present in each frame ind.
	std::unordered_map<NvMOTStreamId, int> perStreamTargetNum;

	m_targetNumInd2RoisInBatchMapper.clear();
	m_lastInd = lastBboxInd;
	int targetNumCnt = 0;

	/// For videoStreams from all existing targets.
	for(auto it = roisInBatch.begin(); it != roisInBatch.end(); ++it)
	{
		const NvMOTStreamId &streamID = it->first;

		/// A vector of {trackerInd, bbox}.
		const std::vector<std::pair<uint, NvMotDataTypes::Rect2f> > &rois = roisInBatch.at(streamID);

		for(uint i=0; i<rois.size(); i++, targetNumCnt++)
		{
			const NvMotDataTypes::Rect2f &bbox = rois[i].second;

			m_rois_h[targetNumCnt].x = bbox.min()(0);
			m_rois_h[targetNumCnt].y = bbox.min()(1);
			m_rois_h[targetNumCnt].w = bbox.sizes()(0);
			m_rois_h[targetNumCnt].z = bbox.sizes()(1);

			/// Map the target index to its video stream and per-stream bounding-box index.
			m_targetNumInd2RoisInBatchMapper[targetNumCnt] = std::make_pair(streamID, i);
		}

		/// For videoStreams from existing targets.
		perStreamTargetNum[streamID] = rois.size();
	}

	if((int)targetNumCnt > m_maxTargetsToTrack)
	{
		std::string msg = "Too many objects to track. Max target num = " + std::to_string(m_maxTargetsToTrack) + ", but total rois = " + std::to_string(targetNumCnt) ;
		CLOG(msg.c_str());

		throw std::runtime_error(msg);
	}

	/// Sync host and device.
	CUDA_CHECK(
		cudaMemcpyAsync(m_rois_d, m_rois_h, targetNumCnt * sizeof(float4), cudaMemcpyHostToDevice, m_cuStream)
	);

	int roiInd = 0;

	/// Launch CUDA kernels for processing.
	for(auto it = roisInBatch.begin(); it != roisInBatch.end(); ++it)
	{
		const NvMOTStreamId &streamID = it->first;
		cudaTextureObject_t *pTex = m_frameInput->getTexturePointer(streamID);
		if(!pTex)
		{
			CLOG(
				std::string("[frameTransformInBatch] !! StreamID[" + std::to_string(streamID) + "] NOT found in m_texInput !!").c_str()
			);
			continue;
		}

		if(perStreamTargetNum[streamID] == 0)
			continue;

		/// Setup execution configuration parameters.
		dim3 Threads(m_cudaBlockSize_2D, m_cudaBlockSize_2D);
		dim3 Blocks(
			(m_width_out + Threads.x - 1) / Threads.x,
			(m_height_out * perStreamTargetNum[streamID] + Threads.x - 1) / Threads.x
		);
		float *odata_d = m_odata_d + m_channels_out * m_width_out * m_height_out * roiInd;

		/// Crop the source image and scale to the dst size.
		cuda_ReIDFrameTransform_4CH<<<Blocks, Threads, 0, m_cuStream>>>(
			*pTex, odata_d, m_width_out, m_width_out, uint2{1, (uint)m_maxTargetsToTrack},
			m_height_out, m_frameInput->getWidthIn(), m_frameInput->getHeightIn(), &m_rois_d[roiInd],
			perStreamTargetNum[streamID], m_dstColorFormatType, m_channels_out, m_dimOrder,
			m_scaleFactor, outputScaleFactor, float3{outputOffsets[0], outputOffsets[1], outputOffsets[2]}
		);

		roiInd += perStreamTargetNum[streamID];
	}

	CUDA_CHECK( cudaStreamSynchronize(m_cuStream) );
	CUDA_CHECK( cudaPeekAtLastError() );

	return true;
}

#if defined VIZ_DEBUG_REID || defined VIZ_DEBUG_SEG || defined VIZ_DEBUG_POSE
/**
 * @brief Convert network tensor layout into host image layout.
 * @param[in] input_image Input tensor data.
 * @param[out] output_image Output image data to populate.
 * @param[in] batch_size Number of images in the batch.
 * @param[in] num_channels Number of image channels.
 * @param[in] height Image height in pixels.
 * @param[in] width Image width in pixels.
 * @param[in] nchw_to_nhwc Whether to convert NCHW layout to NHWC layout.
 * @param[in] outputScaleFactor Scale applied to output pixel values.
 * @param[in] outputOffsets Per-channel offsets restored in the host image.
 */
static inline void transform2ocv(
    float* input_image, float* output_image, int batch_size, int num_channels, int height, int width,
    bool nchw_to_nhwc, const float outputScaleFactor, const std::vector<float> outputOffsets
)
{
    for (int i = 0; i < batch_size; i++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                for (int c = 0; c < num_channels; c++) {
                    /// Calculate indices for NCHW and NHWC.
                    int nchw_idx = i * num_channels * height * width + c * height * width + h * width + w;
                    int nhwc_idx = i * height * width * num_channels + h * width * num_channels + w * num_channels + c;

                    /// Copy data from NCHW to NHWC.
					if (nchw_to_nhwc)
					{
						output_image[nhwc_idx] = input_image[nchw_idx] / outputScaleFactor + outputOffsets.at(c);
					}
					else
					{
						output_image[nhwc_idx] = input_image[nhwc_idx] / outputScaleFactor + outputOffsets.at(c);
					}
				}
            }
        }
    }
}

/**
 * @brief Copy transformed image patches to host OpenCV matrices.
 * @param[in] outputScaleFactor Scale applied to output pixel values.
 * @param[in] outputOffsets Per-channel offsets restored in the host image.
 * @return Host image patches keyed by stream and target.
 */
std::map<NvMOTStreamId, std::map<uint, cv::Mat > > ReIDFrameTransformTexture::getImagePatches2Host_OCV_batch(
    const float outputScaleFactor, const std::vector<float> outputOffsets
)
{
	/// Get the data from device to host.
	CUDA_CHECK(
		cudaMemcpyAsync (
			m_odata_host, m_odata_d, (m_lastInd + 1) * m_channels_out * m_width_out * m_height_out * sizeof(float),
			cudaMemcpyDeviceToHost, m_cuStream
		)
	);
	CUDA_CHECK( cudaStreamSynchronize(m_cuStream) );
	CUDA_CHECK( cudaPeekAtLastError() );
	bool nchw_to_nhwc = !m_dimOrder;

	/// Convert from nchw to nhwc.
	transform2ocv(
		m_odata_host, m_odata_host_nchw, (m_lastInd + 1), m_channels_out,
		m_height_out, m_width_out, nchw_to_nhwc, outputScaleFactor, outputOffsets
	);

	cv::Mat originalImg, outImg;

	/// Map each video stream and target to its cropped bounding-box image.
	std::map<NvMOTStreamId, std::map<uint, cv::Mat > >  outputRoiImages;

	if(m_channels_out == 3 &&
		(m_dstColorFormatType == NVBUF_COLOR_FORMAT_BGR ||
		m_dstColorFormatType == NVBUF_COLOR_FORMAT_RGB))
	{
		for(uint bboxInd=0; bboxInd <= m_lastInd; bboxInd++)
		{
			/// Declare cv::Mat with proper pitch value.
			originalImg = cv::Mat(
				m_height_out, m_width_out, CV_32FC3,
				reinterpret_cast<float*>(m_odata_host_nchw) + bboxInd * m_width_out * m_height_out * m_channels_out
			);

			originalImg.convertTo(outImg, CV_8UC3);

			/// OpenCV shows BGR format.
			if (m_dstColorFormatType == NVBUF_COLOR_FORMAT_RGB)
			{
				cv::cvtColor(outImg, outImg, cv::COLOR_RGB2BGR);
			}

			outputRoiImages[m_targetNumInd2RoisInBatchMapper[bboxInd].first][bboxInd] = \
						outImg.clone();
		}
	}
	else
		throw std::runtime_error("Invalid image channels");

	return std::move(outputRoiImages);
}
#endif
