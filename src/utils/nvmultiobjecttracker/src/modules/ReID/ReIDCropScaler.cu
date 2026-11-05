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
#include "modules/ReID/ReIDCropScaler.hpp"
#include "modules/FrameTransform/cuda_Gaussian.cuh"
#include "modules/cuDCFv2/helper_math.h"

#if defined(__aarch64__) && defined(USE_GAUSSIAN_SMOOTHING_FOR_SCALING)
/// Without this launch_bounds, this kernel may have launch failures with `too many resources requested for launch at..` error in gdb on Nano.
__launch_bounds__(1024)
#endif
/**
 * @brief Crop and transform frame regions into NCHW network input.
 * @param[in] idata Source image-patch batch in planar layout.
 * @param[out] odata Destination network-input tensor.
 * @param[in] oPitch Pitched width of each output plane.
 * @param[in] width_out Width of each output patch in pixels.
 * @param[in] outBatchingGrid Grid dimensions used to arrange output patches.
 * @param[in] height_out Height of each output patch in pixels.
 * @param[in] height_in Height of each source patch in pixels.
 * @param[in] targetNum Number of target patches to transform.
 * @param[in] channels Number of output color channels.
 * @param[in] outScale Scale applied to each output value.
 * @param[in] offset Per-channel offset subtracted before scaling.
 */
__global__ void cuda_ReIDFrameTransform_nchw(
    uint8_t *idata, float *odata, uint oPitch, int width_out, uint2 outBatchingGrid, int height_out, int height_in,
    const int targetNum, const int channels, const float outScale = 1.0, const float3 offset = {0.0, 0.0, 0.0}
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
		/// Input index : CNHW.
		int ind_in;
		/// Output index : NCHW.
		int ind_out;
		ind_in = bboxId * height_out * oPitch + yid * oPitch + xid;
		/// To place the image patch according to the bbox index.
		ind_out = bboxId * height_out * oPitch * channels
			+ yid * oPitch // Y location within the image patch.
			+ xid;

		odata[ind_out] = (idata[ind_in] - offset.x) * outScale;
		odata[ind_out + height_out * oPitch] = (idata[ind_in + height_in * oPitch] - offset.y) * outScale;
		odata[ind_out + height_out * oPitch * 2] = (idata[ind_in + height_in * oPitch * 2] - offset.z) * outScale;
	}
}

#if defined(__aarch64__) && defined(USE_GAUSSIAN_SMOOTHING_FOR_SCALING)
/// Without this launch_bounds, this kernel may have launch failures with `too many resources requested for launch at..` error in gdb on Nano.
__launch_bounds__(1024)
#endif
/**
 * @brief Crop and transform frame regions into NHWC network input.
 * @param[in] idata Source image-patch batch in interleaved layout.
 * @param[out] odata Destination network-input tensor.
 * @param[in] oPitch Pitched width of each output row.
 * @param[in] width_out Width of each output patch in pixels.
 * @param[in] outBatchingGrid Grid dimensions used to arrange output patches.
 * @param[in] height_out Height of each output patch in pixels.
 * @param[in] targetNum Number of target patches to transform.
 * @param[in] channels Number of output color channels.
 * @param[in] outScale Scale applied to each output value.
 * @param[in] offset Per-channel offset subtracted before scaling.
 */
__global__ void cuda_ReIDFrameTransform_nhwc(
    uint8_t *idata, float *odata, uint oPitch, int width_out, uint2 outBatchingGrid, int height_out,
    const int targetNum, const int channels, const float outScale = 1.0, const float3 offset = {0.0, 0.0, 0.0}
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
		int ind;
		/// To place the image patch according to the bbox index.
		ind = (bboxId * height_out * oPitch
			+ yid * oPitch + xid) * channels; // Y location within the image patch.

		odata[ind] = (idata[ind] - offset.x) * outScale;
		odata[ind + 1] = (idata[ind + 1] - offset.y) * outScale;
		odata[ind + 2] = (idata[ind + 2] - offset.z) * outScale;
	}
}

/** @brief Construct a ReIDCropScaler instance. */
ReIDCropScaler::ReIDCropScaler()
{
	m_width_out = 0;
	m_height_out = 0;
	m_channels_out = 3;
	m_width_in = 0;
	m_height_in = 0;
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

/** @brief Release resources owned by the ReIDCropScaler instance. */
ReIDCropScaler::~ReIDCropScaler()
{
	if (m_isInit)
	{
		deAllocateInputMemoriesBatch();
		deAllocateOutputMemories();
		vpiPayloadDestroy(m_vpiCropPayload);
		vpiStreamDestroy(m_vpiStream);
		m_vpiDCFBboxes.clear();
		vpiArrayDestroy(m_vpiDCFBboxesArray);
		CUDA_CHECK( cudaStreamDestroy(m_cuStream) );

		#if defined(__aarch64__)
			if (m_integrated_gpu)
			{
				for (auto it = m_graphicsResourceAddrMap.begin(); it != m_graphicsResourceAddrMap.end(); ++it)
				{
					CUDA_CHECK(cuGraphicsUnregisterResource(it->second));
				}
				m_graphicsResourceAddrMap.clear();

				m_eglFrameAddrMap.clear();
			}
		#endif
	}
}

/**
 * @brief Configure image scaling and allocate input and output buffers.
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
void ReIDCropScaler::initialize(
    const int debugVerboseLevel, const int maxStreamNum, const int maxBufSurfAddrSize, const int maxTargetNum,
    const NvMotDataTypes::Size srcImgSize, const uint srcImgPitch, const NvBufSurfaceColorFormat colorFormatType,
    const NvMotDataTypes::Size dstImgSize, const NvBufSurfaceColorFormat dstColorFormatType, const int dimOrder
)
{
	CUDA_CHECK( cudaStreamCreateWithFlags(&m_cuStream, cudaStreamNonBlocking) );
	m_debugVerboseLevel = debugVerboseLevel;
	m_maxTargetsToTrack = maxTargetNum;
	m_colorFormatType = colorFormatType;
	m_maxBufSurfAddrSize = maxBufSurfAddrSize;

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

	/// Initialization of VPI related parameters.
	{
		VPI_CHECK(vpiStreamCreate((m_vpiBackend4CropnScale | VPI_BACKEND_CPU), &m_vpiStream));
		VPI_CHECK(
			vpiCreateCropScaler(m_vpiBackend4CropnScale, m_numFramesInBatchMax, m_maxTargetsToTrack, &m_vpiCropPayload)
		);
		m_vpiMemFlags = m_vpiMemFlags | m_vpiBackend4CropnScale;
		/// Reserve max required space in m_vpiDCFBboxes.
		m_vpiDCFBboxes.reserve(m_maxTargetsToTrack);
		memset(&m_vpiDCFBboxes[0], 0, m_maxTargetsToTrack * sizeof(m_vpiDCFBboxes[0]));
		/// Create array for m_vpiDCFBboxes.
		m_vpiBboxesArraySize = 0;
		m_vpiDCFBboxesData = {};
		m_vpiDCFBboxesData.bufferType = VPI_ARRAY_BUFFER_HOST_AOS;
		m_vpiDCFBboxesData.buffer.aos.type = VPI_ARRAY_TYPE_DCF_TRACKED_BOUNDING_BOX;
		m_vpiDCFBboxesData.buffer.aos.capacity = m_vpiDCFBboxes.capacity();
		m_vpiDCFBboxesData.buffer.aos.sizePointer = &m_vpiBboxesArraySize;
		m_vpiDCFBboxesData.buffer.aos.data = &m_vpiDCFBboxes[0];
		VPI_CHECK(vpiArrayCreateWrapper(&m_vpiDCFBboxesData, m_vpiMemFlags, &m_vpiDCFBboxesArray));
	}

	m_isInit = true;
}

/**
 * @brief Configure the input layout and allocate frame memory.
 * @param[in] numFramesInBatch Number of frames in the batch.
 * @param[in] frameSize Input frame dimensions.
 * @param[in] srcImgPitch Input image pitch in bytes.
 * @param[in] frameChannels Number of input image channels.
 */
void ReIDCropScaler::initialize4Input(
	uint numFramesInBatch, const NvMotDataTypes::Size frameSize, const uint srcImgPitch, const uint frameChannels
)
{
	/// Check if the cuda pitch-linear memory is already created and compatible, assuming all the images in the batch are homogeneous in size and channels.
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

/**
 * @brief Allocate input memory for a frame batch.
 * @param[in] numFramesInBatch Number of frames in the batch.
 * @param[in] nx Input image width in pixels.
 * @param[in] ny Input image height in pixels.
 * @param[in] pitch_in Input image pitch in bytes.
 */
void ReIDCropScaler::allocateInputMemoriesBatch(
	const uint numFramesInBatch, const int nx, const int ny, const uint pitch_in
)
{
	/// Allocate memory for VPIImage frame list.
	CUDA_CHECK(
		cudaMallocHost(reinterpret_cast<void **>(&m_vpiInputFrameSequence), m_numFramesInBatchMax * sizeof(VPIImage *))
	);
	memset(m_vpiInputFrameSequence, 0, m_numFramesInBatchMax * sizeof(VPIImage *));
	CUDA_CHECK(
		cudaMallocHost(
			reinterpret_cast<void **>(&m_vpiInputFrameDataList), m_numFramesInBatchMax * sizeof(VPIImageData)
		)
	);
	memset(m_vpiInputFrameDataList, 0, m_numFramesInBatchMax * sizeof(VPIImageData));
	m_isFrameCreated.reserve(m_numFramesInBatchMax);
	m_isFrameCreated.resize(m_numFramesInBatchMax, false);
}

/** @brief Release input memory for the frame batch. */
void ReIDCropScaler::deAllocateInputMemoriesBatch()
{
	for (unsigned int i = 0; i < m_isFrameCreated.size(); i++)
	{
		if (m_isFrameCreated[i] == true)
			vpiImageDestroy(m_vpiInputFrameSequence[i]);
	}
	CUDA_CHECK(cudaFreeHost(m_vpiInputFrameSequence));
	CUDA_CHECK(cudaFreeHost(m_vpiInputFrameDataList));
	m_isFrameCreated.clear();
}

/**
 * @brief Configure the output layout and allocate patch memory.
 * @param[in] dstInferSize Network input dimensions.
 * @param[in] dstColorFormat Output color format, RGB or BGR.
 * @param[in] dimOrder Output tensor dimension order, 0 for NCHW and 1 for NHWC.
 */
void ReIDCropScaler::initialize4Output(
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

	/// Set VPI Image for crop scaler int8 output.
	/// NCHW : planar.
	if(m_dstColorFormatType == NvBufSurfaceColorFormat::NVBUF_COLOR_FORMAT_RGB && m_dimOrder == 0)
	{
		m_vpiImageFormat = VPI_IMAGE_FORMAT_RGB8p;
	}
	/// NHWC : interleaved.
	else if(m_dstColorFormatType == NvBufSurfaceColorFormat::NVBUF_COLOR_FORMAT_RGB && m_dimOrder == 1)
	{
		m_vpiImageFormat = VPI_IMAGE_FORMAT_RGB8;
	}
	/// NCHW : planar.
	else if(m_dstColorFormatType == NvBufSurfaceColorFormat::NVBUF_COLOR_FORMAT_BGR && m_dimOrder == 0)
	{
		m_vpiImageFormat = VPI_IMAGE_FORMAT_BGR8p;
		throw std::runtime_error(
			"inputOrder: 0 (i.e. NCHW) & colorFormat: 1 (BGR) " "is not supported by crop scaler in current VPI version. \n" "Switch to built-in crop scaler i.e. Set useVPICropScaler: 0"
		);
	}
	/// NHWC : interleaved.
	else if(m_dstColorFormatType == NvBufSurfaceColorFormat::NVBUF_COLOR_FORMAT_BGR && m_dimOrder == 1)
	{
		m_vpiImageFormat = VPI_IMAGE_FORMAT_BGR8;
	}
	else
	{
		throw std::runtime_error("Unsupported combination of NvBufSurfaceColorFormat and dimOrder.");
	}
	m_vpiObjPatchesData = {};
	m_vpiObjPatchesData.bufferType = VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR;
	m_vpiObjPatchesData.buffer.pitch.format = m_vpiImageFormat;
	if((m_vpiImageFormat == VPI_IMAGE_FORMAT_RGB8) || (m_vpiImageFormat == VPI_IMAGE_FORMAT_BGR8))
	{
		/// Interleaved format.
		/// Single plane.
		m_vpiObjPatchesData.buffer.pitch.numPlanes = 1;
		m_vpiObjPatchesData.buffer.pitch.planes[0].pixelType = VPI_PIXEL_TYPE_3U8;
		/// Use the raw output pointer as pitch-linear CUDA memory.
		m_vpiObjPatchesData.buffer.pitch.planes[0].pBase = m_odata_unit8_d;
		m_vpiObjPatchesData.buffer.pitch.planes[0].offsetBytes = 0;
		m_vpiObjPatchesData.buffer.pitch.planes[0].width = m_width_out;
		/// Consider the stacked image patches as a single image.
		m_vpiObjPatchesData.buffer.pitch.planes[0].height = m_height_out*m_maxTargetsToTrack;
		m_vpiObjPatchesData.buffer.pitch.planes[0].pitchBytes = m_width_out*m_channels_out;
	}
	else
	{
		/// Planar format.
		m_vpiObjPatchesData.buffer.pitch.numPlanes = 3;

		/// Channel 1.
		m_vpiObjPatchesData.buffer.pitch.planes[0].pixelType = VPI_PIXEL_TYPE_U8;
		/// Use the raw output pointer as pitch-linear CUDA memory.
		m_vpiObjPatchesData.buffer.pitch.planes[0].pBase = m_odata_unit8_d;
		m_vpiObjPatchesData.buffer.pitch.planes[0].offsetBytes = 0;
		m_vpiObjPatchesData.buffer.pitch.planes[0].width = m_width_out;
		/// Consider the stacked image patches as a single image.
		m_vpiObjPatchesData.buffer.pitch.planes[0].height = m_height_out*m_maxTargetsToTrack;
		m_vpiObjPatchesData.buffer.pitch.planes[0].pitchBytes = m_width_out;

		/// Channel 2.
		/// Store the second planar channel as single-byte samples.
		m_vpiObjPatchesData.buffer.pitch.planes[1].pixelType = VPI_PIXEL_TYPE_U8;
		m_vpiObjPatchesData.buffer.pitch.planes[1].pBase = reinterpret_cast<uint8_t *>(m_odata_unit8_d) + uint(
			m_width_out*m_height_out*m_maxTargetsToTrack
		);
		m_vpiObjPatchesData.buffer.pitch.planes[1].offsetBytes = 0;
		m_vpiObjPatchesData.buffer.pitch.planes[1].width = m_width_out;
		m_vpiObjPatchesData.buffer.pitch.planes[1].height = m_height_out*m_maxTargetsToTrack;
		m_vpiObjPatchesData.buffer.pitch.planes[1].pitchBytes = m_width_out;

		/// Channel 3.
		/// Store the third planar channel as single-byte samples.
		m_vpiObjPatchesData.buffer.pitch.planes[2].pixelType = VPI_PIXEL_TYPE_U8;
		m_vpiObjPatchesData.buffer.pitch.planes[2].pBase = reinterpret_cast<uint8_t *>(m_odata_unit8_d) + uint(
			m_width_out*m_height_out*m_maxTargetsToTrack*2
		);
		m_vpiObjPatchesData.buffer.pitch.planes[2].offsetBytes = 0;
		m_vpiObjPatchesData.buffer.pitch.planes[2].width = m_width_out;
		m_vpiObjPatchesData.buffer.pitch.planes[2].height = m_height_out*m_maxTargetsToTrack;
		m_vpiObjPatchesData.buffer.pitch.planes[2].pitchBytes = m_width_out;
	}
	VPI_CHECK( vpiImageCreateWrapper (&m_vpiObjPatchesData, nullptr, m_vpiMemFlags, &m_vpiObjPatches) );
}

/** @brief Allocate transform output memory. */
void ReIDCropScaler::allocateOutputMemories()
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

	CUDA_CHECK(
		cudaMalloc(
			reinterpret_cast<void **>(&m_odata_unit8_d),
			m_maxTargetsToTrack * m_channels_out * m_height_out * m_width_out * sizeof(uint8_t)
		)
	);

	if(m_debugVerboseLevel > 0)
	{
		printf(
			"[cudaCropScaleInTexture2D<T>::allocateInferOutputMemories] m_odata_unit8_d = %.2f [KB]\n",
			m_maxTargetsToTrack * m_channels_out * m_height_out * m_width_out * sizeof(uint8_t) / 1024.f
		);
	}
}

/** @brief Release transform output memory. */
void ReIDCropScaler::deAllocateOutputMemories()
{
	CUDA_CHECK( cudaFree(m_odata_d) );
	#if defined VIZ_DEBUG_REID || defined VIZ_DEBUG_SEG || defined VIZ_DEBUG_POSE
		CUDA_CHECK( cudaFreeHost(m_odata_host) );
		CUDA_CHECK( cudaFreeHost(m_odata_host_nchw) );
	#endif
	CUDA_CHECK( cudaFree(m_odata_unit8_d) );
	vpiImageDestroy(m_vpiObjPatches);
}

/**
 * @brief Crop and rescale target regions in a frame batch.
 * @param[in] roisInBatch The set of ROIs of interest {videoStreamID, a vector of {bboxInd, bbox} }.
 * @param[in] lastBboxInd Last bbox index.
 * @param[in] outputScaleFactor Scale applied to output pixel values.
 * @param[in] outputOffsets Parameter of each channel's offset.
 * @param[in] inputMeta Input frames keyed by stream ID.
 * @return Status.
 */
bool ReIDCropScaler::frameTransformInBatch(
    const std::map<NvMOTStreamId, std::vector<std::pair<uint, NvMotDataTypes::Rect2f> > > &roisInBatch,
    const int lastBboxInd, const float outputScaleFactor, const std::vector<float> outputOffsets,
    const std::map<NvMOTStreamId, NvMOTFrame*> &inputMeta
)
{
	if (outputOffsets.size() != 3)
	{
		throw std::runtime_error("image channel offsets should be length of 3");
	}
	if (lastBboxInd < 0)
		/// No detected objects in this call.
		return true;

	checkInput(inputMeta);

	/// Store how many targets are present in each frame ind.
	std::unordered_map<NvMOTStreamId, int> perStreamTargetNum;

	#if defined VIZ_DEBUG_REID || defined VIZ_DEBUG_SEG || defined VIZ_DEBUG_POSE
		m_targetNumInd2RoisInBatchMapper.clear();
	#endif
	m_lastInd = lastBboxInd;
	int targetNumCnt = 0;
	/// Lock the arrays since the data buffers will be accessed outside VPI.
	VPI_CHECK(vpiArrayLock(m_vpiDCFBboxesArray, VPI_LOCK_WRITE));
	memset(&m_vpiDCFBboxes[0], 0, (lastBboxInd + 1) * sizeof(m_vpiDCFBboxes[0]));

	/// For videoStreams from all existing targets.
	for(auto it = roisInBatch.begin(); it != roisInBatch.end(); ++it)
	{
		const NvMOTStreamId &streamID = it->first;
		const NvMOTFrame *srcMotFrame = inputMeta.at(streamID);
		uint32_t seq_index = srcMotFrame->seq_index;

		/// A vector of {trackerInd, bbox}.
		const std::vector<std::pair<uint, NvMotDataTypes::Rect2f> > &rois = roisInBatch.at(streamID);

		for(uint i=0; i<rois.size(); i++, targetNumCnt++)
		{
			const NvMotDataTypes::Rect2f &bbox = rois[i].second;
			m_vpiDCFBboxes[targetNumCnt].bbox.left = bbox.min()(0);
			m_vpiDCFBboxes[targetNumCnt].bbox.top = bbox.min()(1);
			m_vpiDCFBboxes[targetNumCnt].bbox.width = bbox.sizes()(0);
			m_vpiDCFBboxes[targetNumCnt].bbox.height = bbox.sizes()(1);
			m_vpiDCFBboxes[targetNumCnt].seqIndex = seq_index;
			m_vpiDCFBboxes[targetNumCnt].state = VPI_TRACKING_STATE_TRACKED;

			#if defined VIZ_DEBUG_REID || defined VIZ_DEBUG_SEG || defined VIZ_DEBUG_POSE
				/// Map the target index to its video stream and per-stream bounding-box index.
				m_targetNumInd2RoisInBatchMapper[targetNumCnt] = std::make_pair(streamID, i);
			#endif
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
	else if(targetNumCnt == 0)
	{
		/// No detected objects in this call.
		return true;
	}
	/// Set the valid number of entries in the tracked objects array.
	/// This gets used for dynamic compute adjustments by VPI library.
	assert(targetNumCnt == (lastBboxInd +1));
	m_vpiBboxesArraySize = targetNumCnt;

	/// Unlock the arrays since it's needed by VPI task.
	VPI_CHECK(vpiArrayUnlock(m_vpiDCFBboxesArray));

	/// Crop tracked objects from the input frames and rescale them all to the same dimensions.
	/// Input frames provided as m_vpiInputFrameSequence and tracked objects information is in m_vpiDCFBboxesArray.
	/// Cropped and scaled object patches are stacked on top of each other and provided as output in m_vpiObjPatches.
	VPI_CHECK(
		vpiSubmitCropScalerBatch(
			m_vpiStream, m_vpiBackend4CropnScale, m_vpiCropPayload, m_vpiInputFrameSequence,
			m_numFramesInBatchMax, m_vpiDCFBboxesArray, m_width_out, m_height_out, m_vpiObjPatches
		)
	);

	/// Wait until VPI processing is done.
	VPI_CHECK(vpiStreamSync(m_vpiStream));

	/// Apply three postprocessing operations to the crop-scaler output.
	/// Apply offset and normalization.
	/// Convert int8 to float.
	/// If m_dimOrder=0 i.e. NHWC output is expected : VPI Crop scaler generated a RGB8p images which is CNHW.
	/// Convert it to NHWC.
	/// Lock the VPI Image for reading.
	VPI_CHECK(vpiImageLock(m_vpiObjPatches, VPI_LOCK_READ));

	/// Setup execution configuration parameters.
	dim3 Threads(m_cudaBlockSize_2D, m_cudaBlockSize_2D);
	dim3 Blocks((m_width_out + Threads.x - 1) / Threads.x, (m_height_out * targetNumCnt + Threads.x - 1) / Threads.x);

	if(m_dimOrder == 0)
	{
		cuda_ReIDFrameTransform_nchw<<<Blocks, Threads, 0, m_cuStream>>>(
			m_odata_unit8_d, m_odata_d, m_width_out, m_width_out, uint2{1, (uint)m_maxTargetsToTrack}, m_height_out,
			/// Height of complete image for CNHW input.
			m_height_out*m_maxTargetsToTrack, targetNumCnt, m_channels_out,
			outputScaleFactor, float3{outputOffsets[0], outputOffsets[1], outputOffsets[2]}
		);
	}
	else if(m_dimOrder == 1)
	{
		/// Crop the source image and scale to the dst size.
		cuda_ReIDFrameTransform_nhwc<<<Blocks, Threads, 0, m_cuStream>>>(
			m_odata_unit8_d, m_odata_d, m_width_out, m_width_out,
			uint2{1, (uint)m_maxTargetsToTrack}, m_height_out, targetNumCnt, m_channels_out,
			outputScaleFactor, float3{outputOffsets[0], outputOffsets[1], outputOffsets[2]}
		);
	}
	else
	{
		throw std::runtime_error("Incorrect inputOrder");
	}

	CUDA_CHECK( cudaStreamSynchronize(m_cuStream) );
	CUDA_CHECK( cudaPeekAtLastError() );
	VPI_CHECK(vpiImageUnlock(m_vpiObjPatches));
	return true;
}

/**
 * @brief Retrieve CUDA plane pointers from a surface.
 * @param[in] baseSurface Surface whose CUDA plane pointers are retrieved.
 */
inline void ReIDCropScaler::getCudaMemPointers(NvBufSurfaceParams *baseSurface)
{
	if(!((baseSurface->colorFormat == NvBufSurfaceColorFormat::NVBUF_COLOR_FORMAT_NV12) ||
	(baseSurface->colorFormat == NvBufSurfaceColorFormat::NVBUF_COLOR_FORMAT_RGBA)))
	{
		throw std::runtime_error("[FAILED] Invalid color format");
	}

	#if defined(__aarch64__)
		if(m_integrated_gpu)
		{
			/// Get the cuda mem pointers by EGL mapping.
			if (m_graphicsResourceAddrMap.find(baseSurface) == m_graphicsResourceAddrMap.end())
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
						&m_graphicsResourceAddrMap[baseSurface], eglimage_src, CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY
					)
				);
				/// Map to CUeglFrame, for direct CUDA use.
				CUeglFrame eglFrame;
				CUDA_CHECK(
					cuGraphicsResourceGetMappedEglFrame(&eglFrame, m_graphicsResourceAddrMap.at(baseSurface), 0, 0)
				);
				/// Save eglFrame in map.
				m_eglFrameAddrMap[baseSurface] = std::move(eglFrame);
				/// If the saved egl mapped entries exceed the configured value, erase the oldest entry.
				if(m_graphicsResourceAddrMap.size() > m_maxBufSurfAddrSize)
				{
					if (m_debugVerboseLevel > 1)
					{
						printf(
							"[DCFTracker::getCudaMemPointers] m_graphicsResourceAddrMap size exceeds limit: %lu / %d\n",
							m_graphicsResourceAddrMap.size(), m_maxBufSurfAddrSize
						);
					}
					/// Erase first entry in the map.
					auto it = m_graphicsResourceAddrMap.begin();
					auto jt = m_eglFrameAddrMap.find(it->first);
					CUDA_CHECK( cuGraphicsUnregisterResource(it->second) );
					m_graphicsResourceAddrMap.erase(it);
					m_eglFrameAddrMap.erase(jt);
				}
			}
			m_surfCudaMemPtr[0] = reinterpret_cast<uint8_t*>(m_eglFrameAddrMap[baseSurface].frame.pPitch[0]);
			if(baseSurface->colorFormat == NvBufSurfaceColorFormat::NVBUF_COLOR_FORMAT_NV12)
			{
				/// UV plane for NV12.
				m_surfCudaMemPtr[1] = reinterpret_cast<uint8_t*>(m_eglFrameAddrMap[baseSurface].frame.pPitch[1]);
			}
		}
		else
	#endif
	{
		m_surfCudaMemPtr[0] = reinterpret_cast<uint8_t *>(baseSurface->dataPtr)+uint(
			baseSurface->planeParams.offset[0]
		);
		if(baseSurface->colorFormat == NvBufSurfaceColorFormat::NVBUF_COLOR_FORMAT_NV12)
		{
			/// UV plane for NV12.
			m_surfCudaMemPtr[1] = reinterpret_cast<uint8_t *>(baseSurface->dataPtr)+uint(
				baseSurface->planeParams.offset[1]
			);
		}
	}
}

/**
 * @brief Validate input frames and build the VPI frame sequence.
 * @param[in] inputMeta Input frames keyed by stream ID.
 */
void ReIDCropScaler::checkInput(const std::map<NvMOTStreamId, NvMOTFrame*> &inputMeta)
{
	for (auto it = inputMeta.begin(); it != inputMeta.end(); ++it)
	{
		const NvMOTStreamId &videoStreamId = it->first;
		const NvMOTFrame *srcMotFrame = it->second;

		/// Create VPIImage for the frame in srcMotFrame.
		{
			NvBufSurfaceParams *surfparams = srcMotFrame->bufferList[0];
			uint32_t seq_index = srcMotFrame->seq_index;
			if ((m_width_in < (int)srcMotFrame->bufferList[0]->width) ||
				(m_height_in < (int)srcMotFrame->bufferList[0]->height) ||
				(m_pitch_in < (int)srcMotFrame->bufferList[0]->pitch) ||
				(m_colorFormatType != srcMotFrame->bufferList[0]->colorFormat))
			{
				std::string msg = "Image specifications in configuration paarameters are lower than actual image characteristics! \n  Configured specification = (" +
				std::to_string(m_width_in) + "," + std::to_string(m_height_in) + "," +
				std::to_string(m_pitch_in) + "," + std::to_string(m_colorFormatType) + ") \n"
				"Actual image specifications = (" +
				std::to_string(srcMotFrame->bufferList[0]->width) + "," + std::to_string(srcMotFrame->bufferList[0]->height) + "," +
				std::to_string(srcMotFrame->bufferList[0]->pitch) + "," + std::to_string(srcMotFrame->bufferList[0]->colorFormat) + ")";
				throw std::runtime_error(msg);
			}
			if(seq_index >= m_numFramesInBatchMax)
			{
				throw std::runtime_error("[FAILED] Frame sequence index should be in the range 0 to (batch size - 1)");
			}

			/// Get the cuda memory pointers.
			getCudaMemPointers(surfparams);

			if(m_isFrameCreated[seq_index] == false)
			{
				/// Populate VPIImage fields using the entries in NvBufSurfaceParams structure.
				if (m_colorFormatType == NvBufSurfaceColorFormat::NVBUF_COLOR_FORMAT_NV12)
				{
					m_vpiInputFrameDataList[seq_index] = {};
					m_vpiInputFrameDataList[seq_index].bufferType = VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR;

					/// For NV12.
					m_vpiInputFrameDataList[seq_index].buffer.pitch.format = VPI_IMAGE_FORMAT_NV12_ER;
					m_vpiInputFrameDataList[seq_index].buffer.pitch.numPlanes = 2;

					/// For Y channel.
					m_vpiInputFrameDataList[seq_index].buffer.pitch.planes[0].pixelType = VPI_PIXEL_TYPE_U8;
					/// Use the mapped luma pointer as pitch-linear CUDA memory.
					m_vpiInputFrameDataList[seq_index].buffer.pitch.planes[0].pBase = m_surfCudaMemPtr[0];
					m_vpiInputFrameDataList[seq_index].buffer.pitch.planes[0].offsetBytes = 0;
					m_vpiInputFrameDataList[seq_index].buffer.pitch.planes[0].width = surfparams->planeParams.width[0];
					m_vpiInputFrameDataList[seq_index].buffer.pitch.planes[0].height = surfparams->planeParams.height[0];
					m_vpiInputFrameDataList[seq_index].buffer.pitch.planes[0].pitchBytes = surfparams->planeParams.pitch[0];

					/// For UV channel.
					/// Store interleaved chroma as two-byte samples.
					m_vpiInputFrameDataList[seq_index].buffer.pitch.planes[1].pixelType = VPI_PIXEL_TYPE_2U8;
					m_vpiInputFrameDataList[seq_index].buffer.pitch.planes[1].pBase = m_surfCudaMemPtr[1];
					m_vpiInputFrameDataList[seq_index].buffer.pitch.planes[1].offsetBytes = 0;
					m_vpiInputFrameDataList[seq_index].buffer.pitch.planes[1].width = surfparams->planeParams.width[1];
					m_vpiInputFrameDataList[seq_index].buffer.pitch.planes[1].height = surfparams->planeParams.height[1];
					m_vpiInputFrameDataList[seq_index].buffer.pitch.planes[1].pitchBytes = surfparams->planeParams.pitch[1];
				}
				else if (m_colorFormatType == NvBufSurfaceColorFormat::NVBUF_COLOR_FORMAT_RGBA)
				{
					m_vpiInputFrameDataList[seq_index] = {};
					m_vpiInputFrameDataList[seq_index].bufferType = VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR;

					/// For RGBA.
					m_vpiInputFrameDataList[seq_index].buffer.pitch.format = VPI_IMAGE_FORMAT_RGBA8;
					m_vpiInputFrameDataList[seq_index].buffer.pitch.numPlanes = 1;
					m_vpiInputFrameDataList[seq_index].buffer.pitch.planes[0].pixelType = VPI_PIXEL_TYPE_4U8;
					/// Use the mapped RGBA pointer as pitch-linear CUDA memory.
					m_vpiInputFrameDataList[seq_index].buffer.pitch.planes[0].pBase = m_surfCudaMemPtr[0];
					m_vpiInputFrameDataList[seq_index].buffer.pitch.planes[0].offsetBytes = 0;
					m_vpiInputFrameDataList[seq_index].buffer.pitch.planes[0].width = surfparams->width;
					m_vpiInputFrameDataList[seq_index].buffer.pitch.planes[0].height = surfparams->height;
					m_vpiInputFrameDataList[seq_index].buffer.pitch.planes[0].pitchBytes = surfparams->pitch;
				}
				else
				{
					throw std::runtime_error("[FAILED] Invalid color format");
				}
				VPI_CHECK(
					vpiImageCreateWrapper(
						&m_vpiInputFrameDataList[seq_index], nullptr, m_vpiMemFlags, &m_vpiInputFrameSequence[seq_index]
					)
				);
				m_isFrameCreated[seq_index] = true;
			}
			else
			{
				if (m_colorFormatType == NvBufSurfaceColorFormat::NVBUF_COLOR_FORMAT_NV12)
				{
					/// Refresh the pitch-linear CUDA memory pointers for the NV12 planes.
					m_vpiInputFrameDataList[seq_index].buffer.pitch.planes[0].pBase = m_surfCudaMemPtr[0];
					m_vpiInputFrameDataList[seq_index].buffer.pitch.planes[0].offsetBytes = 0;
					m_vpiInputFrameDataList[seq_index].buffer.pitch.planes[1].pBase = m_surfCudaMemPtr[1];
					m_vpiInputFrameDataList[seq_index].buffer.pitch.planes[1].offsetBytes = 0;
				}
				else if (m_colorFormatType == NvBufSurfaceColorFormat::NVBUF_COLOR_FORMAT_RGBA)
				{
					/// Refresh the pitch-linear CUDA memory pointer for the RGBA plane.
					m_vpiInputFrameDataList[seq_index].buffer.pitch.planes[0].pBase = m_surfCudaMemPtr[0];
					m_vpiInputFrameDataList[seq_index].buffer.pitch.planes[0].offsetBytes = 0;
				}
				else
				{
					throw std::runtime_error("[FAILED] Invalid color format");
				}
				VPI_CHECK(vpiImageSetWrapper(m_vpiInputFrameSequence[seq_index], &m_vpiInputFrameDataList[seq_index]));
			}
		}
	}
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
std::map<NvMOTStreamId, std::map<uint, cv::Mat > > ReIDCropScaler::getImagePatches2Host_OCV_batch(
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

	if(m_debugVerboseLevel > 1)
	{
		printf(
			"[cudaCropScaleInTexture2D<T>::getInferImagePatches2Host_OCV_batch]\nm_odataInfer_host: [%.1f %.1f %.1f ... %.1f %.1f %.1f]\n",
			m_odata_host_nchw[0], m_odata_host_nchw[1], m_odata_host_nchw[2],
			m_odata_host_nchw[(m_lastInd + 1) * m_channels_out * m_width_out * m_height_out-3],
			m_odata_host_nchw[(m_lastInd + 1) * m_channels_out * m_width_out * m_height_out-2],
			m_odata_host_nchw[(m_lastInd + 1) * m_channels_out * m_width_out * m_height_out-1]
		);
	}

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
