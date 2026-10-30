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

#if defined(VIZ_DEBUG)
#include "opencv2/imgproc.hpp"
#endif

#include "modules/utility/debug_common.hpp"
#include "modules/utility/cuda_common.hpp"
#include "modules/cuDCFv2/cuDCFFrameTransformTexture.hpp"
#include "modules/FrameTransform/cuda_Gaussian.cuh"
#include "modules/cuDCFv2/helper_math.h"

// The launch bound avoids excessive-resource launch failures during debugging on embedded platforms.
#if defined(__aarch64__) && defined(USE_GAUSSIAN_SMOOTHING_FOR_SCALING)
__launch_bounds__(1024)
#endif
/**
 * @brief Concatenate cropped NV12 patches while preserving their plane layout.
 * @param[in] luma The luma texture.
 * @param[in] chroma The interleaved chroma texture.
 * @param[out] odata The destination image buffer.
 * @param[in] oPitch The destination image pitch.
 * @param[in] dstOffsets The destination offset for each target patch.
 * @param[in] width_out The output patch width.
 * @param[in] outBatchingGrid The output batching-grid dimensions.
 * @param[in] height_out The output patch height.
 * @param[in] width_in The input image width.
 * @param[in] height_in The input image height.
 * @param[in] rois The source region for each target.
 * @param[in] targetNum The number of target patches.
 * @param[in] scale The output-value scale factor.
 */
__global__ void cuda_cuDCFFrameTransform_batch_NV12(
    cudaTextureObject_t luma, cudaTextureObject_t chroma, uint8_t *odata, uint oPitch,
    uint2 dstOffsets[], int width_out, uint2 outBatchingGrid, int height_out,
    int width_in, int height_in, float4 rois[], const int targetNum, const int scale = 1
)
{
    int xid = blockIdx.x * blockDim.x + threadIdx.x;
    int yid = blockIdx.y * blockDim.y + threadIdx.y;

    int2 bboxLoc;

    bboxLoc.y = int(yid/height_out);
    bboxLoc.x = int(xid/width_out);

    int bboxId = bboxLoc.y*outBatchingGrid.x + bboxLoc.x;

    /// Correct to relative xid and yid.
    yid -= bboxLoc.y*height_out;
    xid -= bboxLoc.x*width_out;

    if(bboxId >= targetNum)
		return;

    if(xid >= 0 && xid < width_out && yid >= 0 && yid < height_out)
    {
		float2 pt;

		/// For Luma.
		{
			/// Starting point in normalized coord.
			pt.x =  (rois[bboxId].x)/(float)width_in
					+ (((xid+0.5f)/(float)width_out) // Steps.
					*((rois[bboxId].w)/(float)width_in)); // Total increment in normalized coord.
			/// Starting point in normalized coord.
			pt.y =  (rois[bboxId].y)/(float)height_in
					+ (((yid+0.5f)/(float)height_out) // Steps.
					*((rois[bboxId].z)/(float)height_in)); // Total increment in normalized coord.

			// To allow filtering mode, the textures should be configured to return floating-point data.
			// Tex2D<float> is used for uint8 and uint16 images so filtering can return floating-point samples.
			#ifdef USE_GAUSSIAN_SMOOTHING_FOR_SCALING
				float y = cuda_GaussianSmoothing_3x3<float>(luma, pt);
			#else
				float y = tex2D<float>(luma, pt.x, pt.y);
			#endif

			/// To place the image patch according to the bbox index.
			uint32_t ind = dstOffsets[bboxId].y + dstOffsets[bboxId].x
							+ yid*oPitch // Y location within the image patch.
							+ xid; // X location within the image patch.

			odata[ind] = max(0, min(255, int(y*scale +0.5f)));

		}

		/// For Chroma.
		if(xid< width_out*0.5 && yid < height_out*0.5)
		{
			/// Starting point in normalized coord.
			pt.x =  (rois[bboxId].x)/(float)width_in
						 + (((xid+0.5f)/(float)(width_out*0.5)) // Steps.
						 *((rois[bboxId].w)/(float)width_in) // Total increment in normalized coord.
						 );
			/// Starting point in normalized coord.
			pt.y =  (rois[bboxId].y)/(float)height_in
						 + (((yid+0.5f)/(float)(height_out*0.5)) // Steps.
						 *((rois[bboxId].z)/(float)height_in) // Total increment in normalized coord.
						 );

			// To allow filtering mode, the textures should be configured to return floating-point data.
			// Tex2D<float> is used for uint8 and uint16 images so filtering can return floating-point samples.
			#ifdef USE_GAUSSIAN_SMOOTHING_FOR_SCALING
				float2 uv = cuda_GaussianSmoothing_3x3<float2>(chroma, pt);
			#else
				float2 uv = tex2D<float2>(chroma, pt.x, pt.y);
			#endif
			/// To place the image patch according to the bbox index.
			uint32_t ind = dstOffsets[bboxId].y + dstOffsets[bboxId].x
							+ height_out*oPitch // Because Chroma components (i.e., u & v) are placed from the end of Luma component.
							+ yid*oPitch // Y location within the image patch.
							+ xid*2; // X location within the image patch.

			odata[ind    ] = max(0, min(255, int(uv.x*scale+0.5f)));
			odata[ind + 1] = max(0, min(255, int(uv.y*scale+0.5f)));

		}
    }
}

// The launch bound avoids excessive-resource launch failures during debugging on embedded platforms.
#if defined(__aarch64__) && defined(USE_GAUSSIAN_SMOOTHING_FOR_SCALING)
__launch_bounds__(1024)
#endif
/**
 * @brief Mosaic cropped NV12 patches into a single NV12 image.
 * @param[in] luma The luma texture.
 * @param[in] chroma The interleaved chroma texture.
 * @param[out] odata The destination image buffer.
 * @param[in] oPitch The destination image pitch.
 * @param[in] dstOffsets The destination offset for each target patch.
 * @param[in] width_out The output patch width.
 * @param[in] outBatchingGrid The output batching-grid dimensions.
 * @param[in] height_out The output patch height.
 * @param[in] width_in The input image width.
 * @param[in] height_in The input image height.
 * @param[in] rois The source region for each target.
 * @param[in] targetNum The number of target patches in the current stream.
 * @param[in] totalTargetNum The total number of target patches in the batch.
 * @param[in] scale The output-value scale factor.
 */
__global__ void cuda_cuDCFFrameTransform_mosaic_NV12(
    cudaTextureObject_t luma, cudaTextureObject_t chroma, uint8_t *odata, uint oPitch,
    uint2 dstOffsets[], int width_out, uint2 outBatchingGrid, int height_out, int width_in,
    int height_in, float4 rois[], const int targetNum, const int totalTargetNum, const int scale = 1
)
{
    int xid = blockIdx.x * blockDim.x + threadIdx.x;
    int yid = blockIdx.y * blockDim.y + threadIdx.y;

    int2 bboxLoc;

    bboxLoc.y = int(yid/height_out);
    bboxLoc.x = int(xid/width_out);

    int bboxId = bboxLoc.y*outBatchingGrid.x + bboxLoc.x;

    /// Correct to relative xid and yid.
    yid -= bboxLoc.y*height_out;
    xid -= bboxLoc.x*width_out;

    if(bboxId >= targetNum)
		return;

    if(xid >= 0 && xid < width_out && yid >= 0 && yid < height_out)
    {
		float2 pt;

		/// For Luma.
		{
			/// Starting point in normalized coord.
			pt.x =  (rois[bboxId].x)/(float)width_in
						 + (((xid+0.5f)/(float)width_out) // Steps.
						 *((rois[bboxId].w)/(float)width_in)); // Total increment in normalized coord.
			/// Starting point in normalized coord.
			pt.y =  (rois[bboxId].y)/(float)height_in
						 + (((yid+0.5f)/(float)height_out) // Steps.
						 *((rois[bboxId].z)/(float)height_in)); // Total increment in normalized coord.

			// To allow filtering mode, the textures should be configured to return floating-point data.
			// Tex2D<float> is used for uint8 and uint16 images so filtering can return floating-point samples.
			#ifdef USE_GAUSSIAN_SMOOTHING_FOR_SCALING
				float y = cuda_GaussianSmoothing_3x3<float>(luma, pt);
			#else
				float y = tex2D<float>(luma, pt.x, pt.y);
			#endif

			/// To place the image patch according to the bbox index.
			uint32_t ind = dstOffsets[bboxId].y + dstOffsets[bboxId].x
							+ yid*oPitch // Y location within the image patch.
							+ xid; // X location within the image patch.

			odata[ind] = max(0, min(255, int(y*scale +0.5f)));

		}

		/// For Chroma.
		if(xid< width_out*0.5 && yid < height_out*0.5)
		{
			/// Starting point in normalized coord.
			pt.x =  (rois[bboxId].x)/(float)width_in
						 + (((xid+0.5f)/(float)(width_out*0.5)) // Steps.
						 *((rois[bboxId].w)/(float)width_in) // Total increment in normalized coord.
						 );
			/// Starting point in normalized coord.
			pt.y =  (rois[bboxId].y)/(float)height_in
						 + (((yid+0.5f)/(float)(height_out*0.5)) // Steps.
						 *((rois[bboxId].z)/(float)height_in) // Total increment in normalized coord.
						 );

			// To allow filtering mode, the textures should be configured to return floating-point data.
			// Tex2D<float> is used for uint8 and uint16 images so filtering can return floating-point samples.
			#ifdef USE_GAUSSIAN_SMOOTHING_FOR_SCALING
				float2 uv = cuda_GaussianSmoothing_3x3<float2>(chroma, pt);
			#else
				float2 uv = tex2D<float2>(chroma, pt.x, pt.y);
			#endif

			/// Because Chroma components (i.e., u & v) are placed from the end of Luma component.
			uint32_t ind =	uint32_t(outBatchingGrid.y*height_out*oPitch)
							+ uint32_t(dstOffsets[bboxId].y/2 + dstOffsets[bboxId].x) // To place the image patch according the bbox index.
							+ yid*oPitch // Y location within the image patch.
							+ xid*2; // Because of the interleaved uv channels.

			odata[ind    ] = max(0, min(255, int(uv.x*scale+0.5f)));
			odata[ind + 1] = max(0, min(255, int(uv.y*scale+0.5f)));

		}
    }
}

// The launch bound avoids excessive-resource launch failures during debugging on embedded platforms.
#if defined(__aarch64__) && defined(USE_GAUSSIAN_SMOOTHING_FOR_SCALING)
__launch_bounds__(1024)
#endif
/**
 * @brief Crop and scale single-channel texture regions into batched image patches.
 * @param[in] texInput The input texture.
 * @param[out] odata The destination image buffer.
 * @param[in] oPitch The destination image pitch.
 * @param[in] dstOffsets The destination offset for each target patch.
 * @param[in] width_out The output patch width.
 * @param[in] outBatchingGrid The output batching-grid dimensions.
 * @param[in] height_out The output patch height.
 * @param[in] width_in The input image width.
 * @param[in] height_in The input image height.
 * @param[in] rois The source region for each target.
 * @param[in] targetNum The number of target patches.
 * @param[in] scale The output-value scale factor.
 */
__global__ void cuda_cuDCFFrameTransform_batch_1CH(
    cudaTextureObject_t texInput, uint8_t *odata, uint oPitch, uint2 dstOffsets[], int width_out, uint2 outBatchingGrid,
    int height_out, int width_in, int height_in, float4 rois[], const int targetNum, const int scale = 1
)
{
    int xid = blockIdx.x * blockDim.x + threadIdx.x;
    int yid = blockIdx.y * blockDim.y + threadIdx.y;

    int2 bboxLoc;

    bboxLoc.y = int(yid/height_out);
    bboxLoc.x = int(xid/width_out);

    int bboxId = bboxLoc.y*outBatchingGrid.x + bboxLoc.x;

    /// Correct to relative xid and yid.
    yid -= bboxLoc.y*height_out;
    xid -= bboxLoc.x*width_out;

    if(bboxId >= targetNum)
		return;

    if(xid >= 0 && xid < width_out && yid >= 0 && yid < height_out)
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

		// To allow filtering mode, the textures should be configured to return floating-point data.
		// Tex2D<float> is used for uint8 and uint16 images so filtering can return floating-point samples.
		#ifdef USE_GAUSSIAN_SMOOTHING_FOR_SCALING
			float valFloat = cuda_GaussianSmoothing_3x3<float>(texInput, pt);
		#else
			float valFloat = tex2D<float>(texInput, pt.x, pt.y);
		#endif

		/// To place the image patch according to the bbox index.
		uint32_t ind = dstOffsets[bboxId].y + dstOffsets[bboxId].x
						+ yid*oPitch // Y location within the image patch.
						+ xid; // X location within the image patch.

		odata[ind] = max(0, min(255, int(valFloat*scale + 0.5f)));
    }
}

// The launch bound avoids excessive-resource launch failures during debugging on embedded platforms.
#if defined(__aarch64__) && defined(USE_GAUSSIAN_SMOOTHING_FOR_SCALING)
__launch_bounds__(1024)
#endif
/**
 * @brief Crop and scale four-channel texture regions into batched image patches.
 * @param[in] texInput The input texture.
 * @param[out] odata The destination image buffer.
 * @param[in] oPitch The destination image pitch.
 * @param[in] dstOffsets The destination offset for each target patch.
 * @param[in] width_out The output patch width.
 * @param[in] outBatchingGrid The output batching-grid dimensions.
 * @param[in] height_out The output patch height.
 * @param[in] width_in The input image width.
 * @param[in] height_in The input image height.
 * @param[in] rois The source region for each target.
 * @param[in] targetNum The number of target patches.
 * @param[in] scale The output-value scale factor.
 */
__global__ void cuda_cuDCFFrameTransform_batch_4CH(
    cudaTextureObject_t texInput, uint8_t *odata, uint oPitch, uint2 dstOffsets[], int width_out, uint2 outBatchingGrid,
    int height_out, int width_in, int height_in, float4 rois[], const int targetNum, const int scale = 1
)
{
    int xid = blockIdx.x * blockDim.x + threadIdx.x;
    int yid = blockIdx.y * blockDim.y + threadIdx.y;

    int2 bboxLoc;

    bboxLoc.y = int(yid/height_out);
    bboxLoc.x = int(xid/width_out);

    int bboxId = bboxLoc.y*outBatchingGrid.x + bboxLoc.x;

    /// Correct to relative xid and yid.
    yid -= bboxLoc.y*height_out;
    xid -= bboxLoc.x*width_out;

    if(bboxId >= targetNum)
		return;

    if(xid >= 0 && xid < width_out && yid >= 0 && yid < height_out)
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

		// To allow filtering mode, the textures should be configured to return floating-point data.
		// Tex2D<float> is used for uint8 and uint16 images so filtering can return floating-point samples.
		#ifdef USE_GAUSSIAN_SMOOTHING_FOR_SCALING
			float4 valFloat = cuda_GaussianSmoothing_3x3<float4>(texInput, pt);
		#else
			float4 valFloat = tex2D<float4>(texInput, pt.x, pt.y);
		#endif

		/// To place the image patch according to the bbox index.
		uint32_t ind = dstOffsets[bboxId].y + dstOffsets[bboxId].x
						+ yid*oPitch // Y location within the image patch.
						+ xid*4; // X location within the image patch.

		odata[ind + 0] = max(0, min(255, int(valFloat.x*scale+0.5f)));
		odata[ind + 1] = max(0, min(255, int(valFloat.y*scale+0.5f)));
		odata[ind + 2] = max(0, min(255, int(valFloat.z*scale+0.5f)));
		odata[ind + 3] = max(0, min(255, int(valFloat.w*scale+0.5f)));
    }
}

/** @brief Construct an uninitialized CUDA texture frame transform. */
cuDCFFrameTransformTexture::cuDCFFrameTransformTexture()
{
	m_doMosaicing = false;
	m_multiResScaleNum = 1;

	for(uint multiResInd = 0; multiResInd < CUDA_TEXTURE_MAX_MULTI_SCALE_NUM; multiResInd++)
	{
		m_width_out[multiResInd] = 0;
		m_height_out[multiResInd] = 0;
	}

	m_isInit = false;
}

/** @brief Release CUDA texture frame-transform resources. */
cuDCFFrameTransformTexture::~cuDCFFrameTransformTexture()
{
	if (m_isInit)
	{
		deAllocateOutputMemories();
	}
}

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
void cuDCFFrameTransformTexture::initialize(
    const int debugVerboseLevel, const int maxStreamNum, const int maxBufSurfAddrSize, const int maxTargetNum,
    const NvMotDataTypes::Size srcImgSize, const uint srcImgPitch, const NvBufSurfaceColorFormat colorFormatType,
    const NvMotDataTypes::Size dstImgSize, const bool needSecondaryScale, const float secondaryScaleFactor
)
{
	FrameTransformTexture::initialize_common(
		debugVerboseLevel, maxStreamNum, maxBufSurfAddrSize, maxTargetNum, srcImgSize, srcImgPitch, colorFormatType
	);

	m_multiResScaleFactors[0] = 1.f;

	if(needSecondaryScale)
	{
		m_multiResScaleNum = 2;
		m_multiResScaleFactors[1] = secondaryScaleFactor;
	}

	if(colorFormatType == NVBUF_COLOR_FORMAT_BGRA
			|| colorFormatType == NVBUF_COLOR_FORMAT_RGBA
			)
	{
		initialize4Output(dstImgSize);
	}
	else if(colorFormatType == NVBUF_COLOR_FORMAT_NV12)
	{
		initialize4Output(NvMotDataTypes::Size(dstImgSize.width, dstImgSize.height*1.5f));
	}
	else
	{
		throw std::runtime_error("Invalid Color Formats");
	}

	m_isInit = true;
}

/**
 * @brief Set output config and allocate the device and host memories for the output frame patches.
 * @param[in] dstSize The output patch size.
 */
void cuDCFFrameTransformTexture::initialize4Output(const NvMotDataTypes::Size dstSize)
{
	/// Create output host memory if not created.
	if(m_width_out[0] != 0 && m_height_out[0] != 0)
	{

		deAllocateOutputMemories();
	}

	for(int multiResInd=0; multiResInd<m_multiResScaleNum; multiResInd++)
	{
		m_width_out[multiResInd] = dstSize.width * m_multiResScaleFactors[multiResInd];
		m_height_out[multiResInd] = dstSize.height * m_multiResScaleFactors[multiResInd];

	}

	/// Determine output memory layout.
	{
		/// Find max output width.
		int max_width_out = 0;
		for(int multiResInd=0; multiResInd<m_multiResScaleNum; multiResInd++)
		{
			max_width_out = m_width_out[multiResInd] > max_width_out ? m_width_out[multiResInd] : max_width_out;
		}

		/// CUDA memory layout would be square-shaped.
		m_outBatchingGrid.x = std::min(
			ceilf((float)sqrt(m_maxTargetsToTrack)), CUDA_LIMIT_ON_2D_TEXTURE_WIDTH / (float)max_width_out
		);
		m_outBatchingGrid.y = ceilf(m_maxTargetsToTrack/(float)m_outBatchingGrid.x);

		if(m_debugVerboseLevel > 0)
			printf("[cuDCFFrameTransformTexture::initialize()] max_width_out = %d, outBatchingGrid (rows, cols) = (%d, %d)\n", max_width_out, m_outBatchingGrid.y, m_outBatchingGrid.x);

		if(m_outBatchingGrid.y*max_width_out >= CUDA_LIMIT_ON_2D_TEXTURE_HEIGHT)
		{
			std::runtime_error("Hit CUDA limit on 2D texture size, due to too many number of image patches");
		}
	}

	allocateOutputMemories();
}

/** @brief Allocate the device and host memories needed to perform transform. */
void cuDCFFrameTransformTexture::allocateOutputMemories()
{
	for(int multiResInd=0; multiResInd<m_multiResScaleNum; multiResInd++)
	{
		CUDA_CHECK(
			cudaMallocPitch(
				reinterpret_cast<void **>(&m_odataPL_d[multiResInd]), &m_odataPL_d_pitch[multiResInd],
				m_width_out[multiResInd]*sizeof(uint8_t)*(m_frameInput->getChannelsIn())*m_outBatchingGrid.x,
				m_height_out[multiResInd]*m_outBatchingGrid.y
			)
		);

		if(m_debugVerboseLevel > 0)
		{
			printf(
				"m_odataPL_d[%d] = %.2f [KB]\n", multiResInd,
				m_odataPL_d_pitch[multiResInd]*m_height_out[multiResInd]*m_maxTargetsToTrack / 1024.f
			);
			printf(
				"[cuDCFFrameTransformTexture::allocateOutputMemories] m_odataPL_d_pitch[%d] = %d (c.f., width_out*m_outBatchingGrid.x = %d) --> %.2f %%\n",
				multiResInd, (int)m_odataPL_d_pitch[multiResInd], (int)m_width_out[multiResInd]*m_outBatchingGrid.x,
				m_odataPL_d_pitch[multiResInd] / float(m_width_out[multiResInd]*m_outBatchingGrid.x) * 100
			);
		}

		/// Declare the host memory large enough to accomodate the pitch value in device memory.
		CUDA_CHECK(
			cudaMallocHost(
				reinterpret_cast<void**>( &m_odata_host[multiResInd] ),
				m_odataPL_d_pitch[multiResInd] * m_height_out[multiResInd]*m_outBatchingGrid.y
			)
		);
	}
}

/** @brief Release the device and host memories needed to perform transform. */
void cuDCFFrameTransformTexture::deAllocateOutputMemories()
{
	for(int multiResInd=0; multiResInd<m_multiResScaleNum; multiResInd++)
	{
		CUDA_CHECK( cudaFreeHost(m_odata_host[multiResInd]) );
		CUDA_CHECK( cudaFree(m_odataPL_d[multiResInd]) );
	}
}

/**
 * @brief Transform a batch of NV12 bounding-box regions with optional mosaicing.
 * @param[in] roisInBatch The ROIs for each stream, as pairs of bbox index and rectangle.
 * @param[in] lastBboxInd The last bbox index in the batch.
 * @param[in] doMosaicing Whether to mosaic the cropped patches.
 * @return True when the batch transform finishes.
 */
bool cuDCFFrameTransformTexture::frameTransformInBatch_NV12(
    const std::map<NvMOTStreamId, std::vector<std::pair<uint, NvMotDataTypes::Rect2f> > > &roisInBatch,
    const uint lastBboxInd, bool doMosaicing
)
{
	m_doMosaicing = doMosaicing;
	m_lastInd = lastBboxInd;

	/// Store how many targets are present in each frame ind.
	std::unordered_map<NvMOTStreamId, int> perStreamTargetNum;

	/// Make sure the number of ROIs does not exceed the maximum number of targets to track.
	if((int)m_lastInd+1 > m_maxTargetsToTrack)
	{
		std::string msg = "Too many objects to track. Max target num = " + std::to_string(m_maxTargetsToTrack) + ", but total rois = " + std::to_string(m_lastInd) ;
		CLOG(msg.c_str());

		throw std::runtime_error(msg);
	}

	for(int multiResInd = 0; multiResInd<m_multiResScaleNum; multiResInd++)
	{
		m_targetNumInd2RoisInBatchMapper.clear();

		int targetNumCnt = 0;

		/// For videoStreams from all existing targets.
		for(auto it = roisInBatch.begin(); it != roisInBatch.end(); ++it)
		{
			const NvMOTStreamId &streamID = it->first;

			/// A vector of {trackerInd, bbox}.
			const std::vector<std::pair<uint, NvMotDataTypes::Rect2f> > &rois = roisInBatch.at(streamID);

			for(uint i=0; i<rois.size(); i++, targetNumCnt++)
			{
				/// Same as targetInd.
				const uint bboxInd = rois[i].first;
				const NvMotDataTypes::Rect2f &rect = rois[i].second;
				const Eigen::Vector2f min = rect.min();
				const Eigen::Vector2f max = rect.max();

				m_rois_h[targetNumCnt].x = min(0);
				m_rois_h[targetNumCnt].y = min(1);
				m_rois_h[targetNumCnt].w = max(0) - min(0);
				m_rois_h[targetNumCnt].z = max(1) - min(1);

				/// Define the dst memory offsets based on the targetInd location.
				/// Y-axis offset.
				m_dstOffsets_h[targetNumCnt].y = uint(bboxInd/m_outBatchingGrid.x) * uint((m_height_out[multiResInd]/1.5f)) * m_odataPL_d_pitch[multiResInd];
				/// X-axis offset.
				m_dstOffsets_h[targetNumCnt].x = uint(bboxInd%m_outBatchingGrid.x) * m_width_out[multiResInd]*sizeof(uint8_t)*m_frameInput->getChannelsIn();

				/// Map the target index to the corresponding video stream ID and the bbox ind.
				/// {targetInd, { videoStreamID, bboxInd Per Stream} }.
				m_targetNumInd2RoisInBatchMapper[bboxInd] = std::make_pair(streamID, i);
			}

			/// For videoStreams from existing targets.
			perStreamTargetNum[streamID] = rois.size();
		}

		/// Sync host and device.
		CUDA_CHECK(
			cudaMemcpyAsync(
				m_dstOffsets_d, m_dstOffsets_h, (m_lastInd+1) * sizeof(uint2), cudaMemcpyHostToDevice, m_cuStream
			)
		);
		CUDA_CHECK(
			cudaMemcpyAsync(m_rois_d, m_rois_h, (m_lastInd+1) * sizeof(float4), cudaMemcpyHostToDevice, m_cuStream)
		);

		int roiInd = 0;

		/// roisInBatch and perStreamTargetNum follow tracker streams from prior batches.
		/// m_texInput_NV12 follows streams in the current batch, so a partial batch can make the indices disagree.
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
				(m_width_out[multiResInd]*m_outBatchingGrid.x + Threads.x - 1) / Threads.x,
				((m_height_out[multiResInd]/1.5f) * (perStreamTargetNum[streamID]/m_outBatchingGrid.x+1) + Threads.x -1) / Threads.x
			);

			if(!(roiInd <= targetNumCnt - perStreamTargetNum[streamID]))
			{
				throw std::runtime_error(
					"[frameTransformInBatch_NV12] Make sure (roiInd <= targetNumCnt - perStreamTargetNum[videoStreamInd])"
				);
			}

			if(m_doMosaicing == true)
			{
				/// Crop the source image and scale to the dst size.
				cuda_cuDCFFrameTransform_mosaic_NV12<<<Blocks, Threads, 0, m_cuStream>>>(
					pTex_NV12->luma, pTex_NV12->chroma, (uint8_t*)(m_odataPL_d[multiResInd]),
					m_odataPL_d_pitch[multiResInd], &m_dstOffsets_d[roiInd],
					m_width_out[multiResInd], m_outBatchingGrid, (m_height_out[multiResInd]/1.5f),
					m_frameInput->getWidthIn(), m_frameInput->getHeightIn(),
					&m_rois_d[roiInd], perStreamTargetNum[streamID], m_lastInd+1, m_scaleFactor
				);
			}
			else
			{
				/// Crop the source image and scale to the dst size.
				cuda_cuDCFFrameTransform_batch_NV12<<<Blocks, Threads, 0, m_cuStream>>>(
					pTex_NV12->luma, pTex_NV12->chroma, (uint8_t*)(m_odataPL_d[multiResInd]),
					m_odataPL_d_pitch[multiResInd], &m_dstOffsets_d[roiInd], m_width_out[multiResInd],
					m_outBatchingGrid, (m_height_out[multiResInd]/1.5f), m_frameInput->getWidthIn(),
					m_frameInput->getHeightIn(), &m_rois_d[roiInd], perStreamTargetNum[streamID], m_scaleFactor
				);
			}

			roiInd += perStreamTargetNum[streamID];
		}

		CUDA_CHECK( cudaStreamSynchronize(m_cuStream) );
		CUDA_CHECK( cudaPeekAtLastError() );
	}

	return true;
}

/**
 * @brief Transform a batch of bounding-box regions into cropped and scaled patches.
 * @param[in] roisInBatch The ROIs for each stream, as pairs of bbox index and rectangle.
 * @param[in] lastBboxInd The last bbox index in the batch.
 * @return True when the batch transform finishes.
 */
bool cuDCFFrameTransformTexture::frameTransformInBatch(
    const std::map<NvMOTStreamId, std::vector<std::pair<uint, NvMotDataTypes::Rect2f> > > &roisInBatch,
    const uint lastBboxInd
)
{
	m_lastInd = lastBboxInd;

	/// Store how many targets are present in each frame ind.
	std::unordered_map<NvMOTStreamId, int> perStreamTargetNum;

	/// Make sure the number of ROIs does not exceed the maximum number of targets to track.
	if((int)m_lastInd+1 > m_maxTargetsToTrack)
	{
		std::string msg = "Too many objects to track. Max target num = " + std::to_string(m_maxTargetsToTrack) + ", but total rois = " + std::to_string(m_lastInd) ;
		CLOG(msg.c_str());

		throw std::runtime_error(msg);
	}

	for(uint multiResInd = 0; multiResInd<(uint)m_multiResScaleNum; multiResInd++)
	{
		m_targetNumInd2RoisInBatchMapper.clear();
		int targetNumPerFrame = 0;

		for(auto it = roisInBatch.begin(); it != roisInBatch.end(); ++it)
		{
			const NvMOTStreamId &streamID = it->first;

			const std::vector<std::pair<uint, NvMotDataTypes::Rect2f> > &rois = roisInBatch.at(streamID);

			for(uint i=0; i<rois.size(); i++, targetNumPerFrame++)
			{
				/// Same as targetInd.
				const uint bboxInd = rois[i].first;
				const NvMotDataTypes::Rect2f &rect = rois[i].second;
				const Eigen::Vector2f &min = rect.min();
				const Eigen::Vector2f &max = rect.max();

				m_rois_h[targetNumPerFrame].x = min(0);
				m_rois_h[targetNumPerFrame].y = min(1);
				m_rois_h[targetNumPerFrame].w = max(0) - min(0);
				m_rois_h[targetNumPerFrame].z = max(1) - min(1);

				/// Define the dst memory offsets based on the targetInd location.
				/// Y-axis offset.
				m_dstOffsets_h[targetNumPerFrame].y = uint(bboxInd/m_outBatchingGrid.x) * m_height_out[multiResInd] * m_odataPL_d_pitch[multiResInd];
				/// X-axis offset.
				m_dstOffsets_h[targetNumPerFrame].x = uint(bboxInd%m_outBatchingGrid.x) * m_width_out[multiResInd]*sizeof(uint8_t)*m_frameInput->getChannelsIn();

				/// Map the target index to the corresponding video stream ID and the targetInd.
				/// {targetInd, { videoStreamID, bboxInd} }.
				m_targetNumInd2RoisInBatchMapper[rois[i].first] = std::make_pair(streamID, i);
			}

			perStreamTargetNum[streamID] = rois.size();
		}

		/// Sync host and device.
		CUDA_CHECK(
			cudaMemcpyAsync(
				m_dstOffsets_d, m_dstOffsets_h, targetNumPerFrame * sizeof(uint2), cudaMemcpyHostToDevice, m_cuStream
			)
		);
		CUDA_CHECK(
			cudaMemcpyAsync(m_rois_d, m_rois_h, targetNumPerFrame * sizeof(float4), cudaMemcpyHostToDevice, m_cuStream)
		);

		CUDA_CHECK( cudaStreamSynchronize(m_cuStream) );
		CUDA_CHECK( cudaPeekAtLastError() );

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
				(m_width_out[multiResInd]*m_outBatchingGrid.x + Threads.x - 1) / Threads.x,
				(m_height_out[multiResInd] * (perStreamTargetNum[streamID]/m_outBatchingGrid.x+1) + Threads.y -1) / Threads.y
			);

			if(!(roiInd <= targetNumPerFrame - perStreamTargetNum[streamID]))
			{
				throw std::runtime_error(
					"[frameTransformInBatch] Make sure (roiInd <= targetNumPerFrame - perStreamTargetNum[videoStreamInd])"
				);
			}

			/// Crop the source image and scale to the dst size.
			if(m_frameInput->getChannelsIn() == 1)
			{
				cuda_cuDCFFrameTransform_batch_1CH<<<Blocks, Threads, 0, m_cuStream>>>(
					*pTex, (uint8_t*)(m_odataPL_d[multiResInd]), m_odataPL_d_pitch[multiResInd],
					&m_dstOffsets_d[roiInd], m_width_out[multiResInd], m_outBatchingGrid,
					m_height_out[multiResInd], m_frameInput->getWidthIn(), m_frameInput->getHeightIn(),
					&m_rois_d[roiInd], perStreamTargetNum[streamID], m_scaleFactor
				);
			}
			else if(m_frameInput->getChannelsIn() == 4)
			{
				cuda_cuDCFFrameTransform_batch_4CH<<<Blocks, Threads, 0, m_cuStream>>>(
					(*pTex), (uint8_t*)(m_odataPL_d[multiResInd]), m_odataPL_d_pitch[multiResInd],
					&m_dstOffsets_d[roiInd], m_width_out[multiResInd], m_outBatchingGrid,
					m_height_out[multiResInd], m_frameInput->getWidthIn(), m_frameInput->getHeightIn(),
					&m_rois_d[roiInd], perStreamTargetNum[streamID], m_scaleFactor
				);
			}

			roiInd += perStreamTargetNum[streamID];
		}

		CUDA_CHECK( cudaStreamSynchronize(m_cuStream) );
		CUDA_CHECK( cudaPeekAtLastError() );
	}

	return true;
}

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
void cuDCFFrameTransformTexture::getOutputImagePatchPtrs(
	uint8_t* &dataPtr_d, uint &width, uint &height, uint2 &outBatchingGrid,
	uint &channels, uint &pitch, uint &imgNum, uint multiResInd
)
{
	if(multiResInd >= CUDA_TEXTURE_MAX_MULTI_SCALE_NUM)
	{
		throw std::runtime_error(
			"[cuDCFFrameTransformTexture::getOutputImagePatchPtrs] Make sure (multiResInd < CUDA_TEXTURE_MAX_MULTI_SCALE_NUM)"
		);
	}

	dataPtr_d = m_odataPL_d[multiResInd];
	width = m_width_out[multiResInd];
	height = m_height_out[multiResInd];
	outBatchingGrid = m_outBatchingGrid;
	channels = m_frameInput->getChannelsIn();
	pitch = m_odataPL_d_pitch[multiResInd];
	imgNum = m_lastInd+1;

	CUDA_CHECK( cudaStreamSynchronize(m_cuStream) );
	CUDA_CHECK( cudaPeekAtLastError() );
}

#if defined(VIZ_DEBUG)

/**
 * @brief Copy mosaiced output images from the device to host memory and split them into patches.
 * @param[in] multiResInd The multi-resolution scale index.
 * @return Cropped and scaled output images on the host, keyed by stream and target.
 */
std::map<NvMOTStreamId, std::map<uint, cv::Mat > > cuDCFFrameTransformTexture::getImagePatches2Host_OCV_mosaic(
	uint multiResInd
)
{
	/// Make sure the image color format is NV12 and crop/scale is carried out with doMosaicing.
	if(!(m_doMosaicing == true && m_frameInput->getChannelsIn() == 1))
	{
		throw std::runtime_error(
			"[cuDCFFrameTransformTexture::getImagePatches2Host_OCV_mosaic] Make sure (m_doMosaicing == true && m_channels_in == 1)"
		);
	}

	/// Get the data from device to host.
	CUDA_CHECK(
		cudaMemcpy2DAsync(
			reinterpret_cast<uint8_t*>(m_odata_host[multiResInd]),
			m_odataPL_d_pitch[multiResInd], reinterpret_cast<uint8_t*>(m_odataPL_d[multiResInd]),
			m_odataPL_d_pitch[multiResInd], m_odataPL_d_pitch[multiResInd],
			m_height_out[multiResInd]*m_outBatchingGrid.y, cudaMemcpyDeviceToHost, m_cuStream
		)
	);

	CUDA_CHECK( cudaStreamSynchronize(m_cuStream) );
	CUDA_CHECK( cudaPeekAtLastError() );

	cv::Mat croppedImg_mosaic_BGR;

	/// Declare cv::Mat with proper pitch value.
	if(m_frameInput->getColorFormat() == NVBUF_COLOR_FORMAT_NV12)
	{
		cv::Mat croppedImg_mosaic = cv::Mat(
			m_height_out[multiResInd]*m_outBatchingGrid.y, m_width_out[multiResInd]*m_outBatchingGrid.x,
			CV_8UC1, reinterpret_cast<uint8_t*>(m_odata_host[multiResInd]), m_odataPL_d_pitch[multiResInd]
		);

		/// Convert NV12 to BGR image.
		cv::cvtColor(croppedImg_mosaic, croppedImg_mosaic_BGR, cv::COLOR_YUV2BGR_NV12);
	}
	else
		throw std::runtime_error("Invalid image channels");

	/// {videoStreamID, {targetInd, cropped bbox images} }.
	std::map<NvMOTStreamId, std::map<uint, cv::Mat > > outputRoiImages;

	uint bboxInd = 0;
	for(uint y=0; y<m_outBatchingGrid.y; y++)
	{
		for(uint x=0; x<m_outBatchingGrid.x && bboxInd<=m_lastInd; x++, bboxInd++)
		{
			cv::Mat croppedImg;

			/// Declare cv::Mat with proper pitch value.
			if(m_frameInput->getColorFormat() == NVBUF_COLOR_FORMAT_NV12)
			{
				croppedImg = cv::Mat(
					croppedImg_mosaic_BGR,
					cv::Rect(
						x*m_width_out[multiResInd], y*(m_height_out[multiResInd]/1.5f),
						m_width_out[multiResInd], (m_height_out[multiResInd]/1.5f)
					)
				);
			}
			else
				throw std::runtime_error("Invalid image channels");

			outputRoiImages[m_targetNumInd2RoisInBatchMapper[bboxInd].first][bboxInd] = croppedImg.clone();
		}
	}

	return std::move(outputRoiImages);
}

/**
 * @brief Copy batched output image patches from device to host memory.
 * @param[in] multiResInd The multi-resolution scale index.
 * @return Cropped and scaled output images on the host, keyed by stream and target.
 */
std::map<NvMOTStreamId, std::map<uint, cv::Mat > > cuDCFFrameTransformTexture::getImagePatches2Host_OCV_batch(
	uint multiResInd
)
{
	/// Get the data from device to host.
	CUDA_CHECK(
		cudaMemcpy2DAsync(
			reinterpret_cast<uint8_t*>(m_odata_host[multiResInd]),
			m_odataPL_d_pitch[multiResInd], reinterpret_cast<uint8_t*>(m_odataPL_d[multiResInd]),
			m_odataPL_d_pitch[multiResInd], m_odataPL_d_pitch[multiResInd],
			m_height_out[multiResInd]*m_outBatchingGrid.y, cudaMemcpyDeviceToHost, m_cuStream
		)
	);

	CUDA_CHECK( cudaStreamSynchronize(m_cuStream) );
	CUDA_CHECK( cudaPeekAtLastError() );

	cv::Mat croppedImg;

	/// {videoStreamID, {targetInd, cropped bbox images} }.
	std::map<NvMOTStreamId, std::map<uint, cv::Mat > >  outputRoiImages;

	uint bboxInd = 0;
	for(uint y=0; y<m_outBatchingGrid.y; y++)
	{
		for(uint x=0; x<m_outBatchingGrid.x && bboxInd<=m_lastInd; x++, bboxInd++)
		{
			/// Declare cv::Mat with proper pitch value.
			if(m_frameInput->getChannelsIn() == 1)
			{
				croppedImg = cv::Mat(
					m_height_out[multiResInd], m_width_out[multiResInd], CV_8UC1,
					reinterpret_cast<uint8_t*>(m_odata_host[multiResInd]) + y*m_odataPL_d_pitch[multiResInd]*m_height_out[multiResInd] + x*m_width_out[multiResInd]*sizeof(uint8_t)*m_frameInput->getChannelsIn(),
					m_odataPL_d_pitch[multiResInd]
				);
			}
			else if(m_frameInput->getChannelsIn() == 4)
			{
				croppedImg = cv::Mat(
					m_height_out[multiResInd], m_width_out[multiResInd], CV_8UC4,
					reinterpret_cast<uint8_t*>(m_odata_host[multiResInd]) + y*m_odataPL_d_pitch[multiResInd]*m_height_out[multiResInd] + x*m_width_out[multiResInd]*sizeof(uint8_t)*m_frameInput->getChannelsIn(),
					m_odataPL_d_pitch[multiResInd]
				);
			}
			else
				throw std::runtime_error("Invalid image channels");

			outputRoiImages[m_targetNumInd2RoisInBatchMapper[bboxInd].first][bboxInd] = croppedImg.clone();
		}
	}

	return std::move(outputRoiImages);
}
#endif
