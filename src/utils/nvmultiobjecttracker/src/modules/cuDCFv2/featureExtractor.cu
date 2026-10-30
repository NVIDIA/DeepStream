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
#include <memory>
#include <mutex>

#if defined(VIZ_DEBUG)
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#if !defined(__aarch64__) && !defined(DEEPSTREAM_SDK)
#include <opencv2/highgui/highgui.hpp>
#endif
#endif

#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>
#include "modules/cuDCFv2/featureExtractor.hpp"

using std::cout;
using std::endl;

/** @brief Construct an uninitialized feature extractor. */
FeatureExtractor::FeatureExtractor()
{
	m_initGpu = false;
	m_isVpiInitialized = false;
	m_debugVerboseLevel = 0;

	m_maxfeatureImgNum = 0;

	for(uint i=0; i<2; i++)
	{
		m_srcSubImgWidth[i] = 0;
		m_srcSubImgHeight[i] = 0;
		m_srcImgPatchPitch[i] = 0;
		m_dstSubImgHeight[i] = 0;
	}

	m_vpiDevType = VPI_BACKEND_CUDA;

	m_srcImgColorChannels = 0;
	m_srcImgPatchNum = 0;

	m_numFeatureChannelsOut = 0;
	m_featureTypes = 0;

	m_batch_ftrs_device = nullptr;
	m_batch_array_host_real = nullptr;
	m_batch_vpi_ftrs = nullptr;
	m_vpiColorNames = nullptr;
	m_vpiHOG = nullptr;

	/// Instantiate CUDA Texture-memory based cropper/scaler.
	m_subImageCropper2 = std::make_shared<cuDCFFrameTransformTexture>();

	/// Start timer.
	if(m_showGpuTiming)
	{
		cudaEventCreate(&m_eventStart);
		cudaEventCreate(&m_eventStop);
	}
}

/** @brief Release feature-extraction resources. */
FeatureExtractor::~FeatureExtractor()
{
	if(m_initGpu)
	{
		CUDA_CHECK(cudaFree( m_batch_vpi_ftrs ));
		CUDA_CHECK(cudaFree( m_batch_ftrs_device ));
		CUDA_CHECK(cudaFreeHost( m_batch_array_host_real ));
	}

	if(m_showGpuTiming)
	{
		cudaEventDestroy(m_eventStart);
		cudaEventDestroy(m_eventStop);
	}

	if(m_isVpiInitialized)
	{
		vpiStreamDestroy(m_vpiStream);

		if(m_vpiColorNames)
		{
			vpiPayloadDestroy(m_vpiColorNames);
		}
		if(m_vpiHOG)
		{
			vpiPayloadDestroy(m_vpiHOG);
		}

		for(size_t i=0; i<m_vpiOutFeatures.size(); i++)
		{
			vpiImageDestroy(m_vpiOutFeatures[i]);
		}

		/// Free VPIImage resources.
		if(m_featureTypes & FEATURE_TYPES::COLOR_NAMES)
		{
			vpiImageDestroy(m_vpiInputImage[0]);
		}

		/// Free VPIImage resources.
		if(m_featureTypes & FEATURE_TYPES::HOG)
		{
			vpiImageDestroy(m_vpiInputImage[1]);
		}
	}
}

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
bool FeatureExtractor::initialize(
    const int debugVerboseLevel, const int maxStreamNum, const int maxBufSurfAddrSize,
    const int maxfeatureImgNum, const NvMotDataTypes::Size srcImgSize, const uint srcImgPitch,
    const NvBufSurfaceColorFormat colorFormatType, const NvMotDataTypes::Size templateImgSize, const int hogCellSize,
    const uint featureTypes, const VPIBackend vpiDevType, const VPIImageFormat vpiImgOutputType
)
{
	m_debugVerboseLevel = debugVerboseLevel;
	m_srcImgColorChannels = 0;
	m_colorFormatType = colorFormatType;
	m_vpiDevType = vpiDevType;
	m_vpiImgOutputType = vpiImgOutputType;

	if(m_colorFormatType == NVBUF_COLOR_FORMAT_BGRA || m_colorFormatType == NVBUF_COLOR_FORMAT_RGBA)
	{
		m_srcImgColorChannels = 4;
	}
	else if(m_colorFormatType == NVBUF_COLOR_FORMAT_NV12)
	{
		m_srcImgColorChannels = 1;
	}
	else
	{
		throw std::runtime_error("Invalid Color Formats\n");
	}

	m_hogCellSize = hogCellSize;

	NvMotDataTypes::Size featureImgSize = NvMotDataTypes::Size(
		templateImgSize.width / hogCellSize, templateImgSize.height / hogCellSize
	);

	m_srcSubImgWidth[0] = featureImgSize.width;
	m_srcSubImgHeight[0] = featureImgSize.height;
	m_featureTypes = featureTypes;

	m_maxfeatureImgNum = maxfeatureImgNum;

	m_numFeatureChannelsOut = 0;

	if(m_featureTypes & FEATURE_TYPES::COLOR_NAMES)
	{
		m_numFeatureChannelsOut += m_numOfChannelsByColorNames;
	}
	if(m_featureTypes & FEATURE_TYPES::HOG)
	{
		m_numFeatureChannelsOut += m_numOfChannelsByHog;
	}

	if((int)m_subImageCropper2->getMaxVideoStreams() < maxStreamNum)
	{
		if(m_featureTypes & FEATURE_TYPES::HOG)
		{
			bool needSecondaryScale = true;
			float secondaryScaleFactor = m_hogCellSize;

			m_subImageCropper2->initialize(
				m_debugVerboseLevel, maxStreamNum, maxBufSurfAddrSize, maxfeatureImgNum, srcImgSize,
				srcImgPitch, m_colorFormatType, featureImgSize, needSecondaryScale, secondaryScaleFactor
			);
		}
		else
		{
			m_subImageCropper2->initialize(
				m_debugVerboseLevel, maxStreamNum, maxBufSurfAddrSize, maxfeatureImgNum,
				srcImgSize, srcImgPitch, m_colorFormatType, featureImgSize
			);
		}
	}

	if(m_debugVerboseLevel > 0)
	{
		size_t free, total;
		CUDA_CHECK( cudaMemGetInfo( &free, &total ) );

		printf(
			"[Tracker][GPU] 	[after cropper/scaler initialized] Free mem = %d [MB], Total = %d [MB]\n",
			int(free/1024/1024), int(total/1024/1024)
		);
	}

	m_outBatchingGrid = m_subImageCropper2->getOutBatchingGrid();

	if(m_debugVerboseLevel > 0)
		cout << "[FeatureExtractor::initialize()] Total # of channels out = " << m_numFeatureChannelsOut << endl;

	if ((m_srcSubImgHeight[0] % 2 != 0) || (m_srcSubImgHeight[0] % 3 != 0))
	{
		throw std::runtime_error("srcHeight must be multiples of 2 and 3\n");
	}

	if(m_batch_vpi_ftrs != nullptr)
	{

		CUDA_CHECK(cudaFree( m_batch_vpi_ftrs ));
	}

	if(m_vpiImgOutputType == VPI_IMAGE_FORMAT_U8)
	{
		/// For extracted features using VPI.
		CUDA_CHECK(
			cudaMallocPitch(
				reinterpret_cast<void **>(&m_batch_vpi_ftrs), &m_batch_vpi_ftrs_pitch,
				m_srcSubImgWidth[0] * sizeof(uint8_t) * m_outBatchingGrid.x,
				m_srcSubImgHeight[0] * m_numFeatureChannelsOut * m_outBatchingGrid.y
			)
		);
	}
	else if(m_vpiImgOutputType == VPI_IMAGE_FORMAT_U16)
	{
		/// For extracted features using VPI.
		CUDA_CHECK(
			cudaMallocPitch(
				reinterpret_cast<void **>(&m_batch_vpi_ftrs), &m_batch_vpi_ftrs_pitch,
				m_srcSubImgWidth[0] * sizeof(uint16_t) * m_outBatchingGrid.x,
				m_srcSubImgHeight[0] * m_numFeatureChannelsOut * m_outBatchingGrid.y * 2
			)
		);
	}
	else
	{
		throw std::runtime_error("Invalid image output type!");
	}

	CUDA_CHECK(
		cudaMallocHost(
			reinterpret_cast<void**>( &m_batch_array_host_real ),
			m_batch_vpi_ftrs_pitch*m_srcSubImgHeight[0]*m_numFeatureChannelsOut*m_outBatchingGrid.y
		)
	);

	m_initGpu = true;

	return true;
}

/** @brief Initialize and allocate the input/output memories for VPI. */
void FeatureExtractor::initialize_VPI()
{
	if(m_subImageCropper2->getColorFormat() == NvBufSurfaceColorFormat::NVBUF_COLOR_FORMAT_NV12)
	{
		for(uint multiResInd = 0; multiResInd < (uint)m_subImageCropper2->getMultiResScaleNum(); multiResInd++)
		{
			/// Set the destination height to two-thirds of the NV12 buffer height to exclude the plane layout overhead.
			m_dstSubImgHeight[multiResInd] = m_srcSubImgHeight[multiResInd]*2/3;

			m_vpiInputImageData[multiResInd].bufferType = VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR;

			/// For NV12.
			m_vpiInputImageData[multiResInd].buffer.pitch.format = VPI_IMAGE_FORMAT_NV12_ER;
			m_vpiInputImageData[multiResInd].buffer.pitch.numPlanes = 2;

			/// VPI treats stacked image patches as one large image rather than a batch.
			/// Stacking works for interleaved RGBA but not for the separate NV12 planes.
			/// A naive two-patch stack assumes Y1 Y2 UV1 UV2 ordering, while NV12 requires Y1 UV1 Y2 UV2.
			/// The cropper therefore composes a mosaiced NV12 image.
			/// For Y channel.
			/// 1-ch Y.
			m_vpiInputImageData[multiResInd].buffer.pitch.planes[0].pixelType = VPI_PIXEL_TYPE_U8;
			/// Use m_srcImgPatchPtr_d as a raw pointer to pitch-linear CUDA memory.
			m_vpiInputImageData[multiResInd].buffer.pitch.planes[0].pBase = m_srcImgPatchPtr_d[multiResInd];
			m_vpiInputImageData[multiResInd].buffer.pitch.planes[0].offsetBytes = 0;
			m_vpiInputImageData[multiResInd].buffer.pitch.planes[0].width = m_srcSubImgWidth[multiResInd]*m_outBatchingGrid.x;
			/// Consider the stacked image patches as a single image.
			m_vpiInputImageData[multiResInd].buffer.pitch.planes[0].height = m_dstSubImgHeight[multiResInd]*m_outBatchingGrid.y;
			m_vpiInputImageData[multiResInd].buffer.pitch.planes[0].pitchBytes = m_srcImgPatchPitch[multiResInd];

			/// For UV channel.
			/// 2-ch due to interleaved UV channel.
			m_vpiInputImageData[multiResInd].buffer.pitch.planes[1].pixelType = VPI_PIXEL_TYPE_2U8;
			m_vpiInputImageData[multiResInd].buffer.pitch.planes[1].pBase = reinterpret_cast<uint8_t *>(m_srcImgPatchPtr_d[multiResInd])+uint(m_srcImgPatchPitch[multiResInd]*m_dstSubImgHeight[multiResInd]*m_outBatchingGrid.y);
			m_vpiInputImageData[multiResInd].buffer.pitch.planes[1].offsetBytes = 0;
			m_vpiInputImageData[multiResInd].buffer.pitch.planes[1].width = (m_srcSubImgWidth[multiResInd]*m_outBatchingGrid.x)/2;
			m_vpiInputImageData[multiResInd].buffer.pitch.planes[1].height = (m_dstSubImgHeight[multiResInd]*m_outBatchingGrid.y)/2;
			m_vpiInputImageData[multiResInd].buffer.pitch.planes[1].pitchBytes = m_srcImgPatchPitch[multiResInd];
		}
	}
	else if(m_subImageCropper2->getColorFormat() == NvBufSurfaceColorFormat::NVBUF_COLOR_FORMAT_RGBA)
	{
		for(uint multiResInd = 0; multiResInd < (uint)m_subImageCropper2->getMultiResScaleNum(); multiResInd++)
		{
			m_dstSubImgHeight[multiResInd] = m_srcSubImgHeight[multiResInd];

			m_vpiInputImageData[multiResInd].bufferType = VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR;

			/// For RGBA.
			m_vpiInputImageData[multiResInd].buffer.pitch.format = VPI_IMAGE_FORMAT_RGBA8;
			m_vpiInputImageData[multiResInd].buffer.pitch.numPlanes = 1;
			/// 4-ch because RGBA.
			m_vpiInputImageData[multiResInd].buffer.pitch.planes[0].pixelType = VPI_PIXEL_TYPE_4U8;
			/// Use m_srcImgPatchPtr_d as a raw pointer to pitch-linear CUDA memory.
			m_vpiInputImageData[multiResInd].buffer.pitch.planes[0].pBase = m_srcImgPatchPtr_d[multiResInd];
			m_vpiInputImageData[multiResInd].buffer.pitch.planes[0].offsetBytes = 0;
			m_vpiInputImageData[multiResInd].buffer.pitch.planes[0].width = m_srcSubImgWidth[multiResInd]*m_outBatchingGrid.x;
			/// Consider the stacked image patches as a single image.
			m_vpiInputImageData[multiResInd].buffer.pitch.planes[0].height = m_dstSubImgHeight[multiResInd]*m_outBatchingGrid.y;
			m_vpiInputImageData[multiResInd].buffer.pitch.planes[0].pitchBytes = m_srcImgPatchPitch[multiResInd];
		}
	}
	else
	{
		throw std::runtime_error("[FAILED] Invalid color format");
	}

	/// Create VPIImage for ColorNames feature extraction.
	if(m_featureTypes & FEATURE_TYPES::COLOR_NAMES)
	{

		VPI_CHECK( vpiImageCreateWrapper (&m_vpiInputImageData[0], nullptr, m_vpiImgFlags, &m_vpiInputImage[0]) );
	}

	/// Create VPIImage for HOG feature extraction.
	if(m_featureTypes & FEATURE_TYPES::HOG)
	{
		if(m_subImageCropper2->getMultiResScaleNum() != 2)
		{
			throw std::runtime_error(
				"[FeatureExtractor::extractFeatures_VPI] Make sure m_subImageCropper2->getMultiResScaleNum() == 2"
			);
		}

		VPI_CHECK( vpiImageCreateWrapper (&m_vpiInputImageData[1], nullptr, m_vpiImgFlags, &m_vpiInputImage[1]) );
	}

	{
		m_vpiOutFeatures.clear();
		m_vpiOutFeaturesData.clear();

		m_vpiOutFeatures.resize(m_numFeatureChannelsOut);
		m_vpiOutFeaturesData.resize(m_numFeatureChannelsOut);

		for(uint ch=0; ch<m_numFeatureChannelsOut; ch++)
		{
			m_vpiOutFeaturesData[ch].bufferType = VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR;
			m_vpiOutFeaturesData[ch].buffer.pitch.numPlanes = 1;

			if(m_vpiImgOutputType == VPI_IMAGE_FORMAT_U8)
			{
				m_vpiOutFeaturesData[ch].buffer.pitch.format = VPI_IMAGE_FORMAT_U8;
				m_vpiOutFeaturesData[ch].buffer.pitch.planes[0].pixelType = VPI_PIXEL_TYPE_U8;
			}
			else if(m_vpiImgOutputType == VPI_IMAGE_FORMAT_U16)
			{
				m_vpiOutFeaturesData[ch].buffer.pitch.format = VPI_IMAGE_FORMAT_U16;
				m_vpiOutFeaturesData[ch].buffer.pitch.planes[0].pixelType = VPI_PIXEL_TYPE_U16;
			}
			else
			{
				throw std::runtime_error("Invalid image output type!");
			}

			/// A raw pointer to pitch-linear CUDA memory.
			m_vpiOutFeaturesData[ch].buffer.pitch.planes[0].pBase = reinterpret_cast<uchar*>(m_batch_vpi_ftrs) + ch*m_batch_vpi_ftrs_pitch*m_dstSubImgHeight[0]*m_outBatchingGrid.y;
			m_vpiOutFeaturesData[ch].buffer.pitch.planes[0].offsetBytes = 0;
			m_vpiOutFeaturesData[ch].buffer.pitch.planes[0].width = m_srcSubImgWidth[0]*m_outBatchingGrid.x;
			/// Consider the stacked image patches as a single image.
			m_vpiOutFeaturesData[ch].buffer.pitch.planes[0].height = m_dstSubImgHeight[0]*m_outBatchingGrid.y;
			m_vpiOutFeaturesData[ch].buffer.pitch.planes[0].pitchBytes = m_batch_vpi_ftrs_pitch;

			VPI_CHECK(vpiImageCreateWrapper (&m_vpiOutFeaturesData[ch], nullptr, m_vpiImgFlags, &m_vpiOutFeatures[ch]));
		}
	}

	{
		VPI_CHECK( vpiStreamCreate(m_vpiDevType, &m_vpiStream) );

		if(m_featureTypes & FEATURE_TYPES::COLOR_NAMES)
		{
			VPI_CHECK( vpiCreateExtractColorNameFeatures(m_vpiDevType, m_vpiImgOutputType, &m_vpiColorNames) );
		}

		if(m_featureTypes & FEATURE_TYPES::HOG)
		{
			VPI_CHECK(
				vpiCreateExtractHOGFeaturesBatch(
					m_vpiDevType, m_outBatchingGrid.x, m_outBatchingGrid.y, m_srcSubImgWidth[1],
					m_dstSubImgHeight[1], m_featureTypeHog, m_hogCellSize, m_numOfChannelsByHog, NULL, &m_vpiHOG
				)
			);

		}
	}

	m_isVpiInitialized = true;
}

/**
 * @brief Extract batched visual features from source-image regions with VPI.
 * @param[in] nvFramesInBatch The input images from which features are extracted.
 * @param[in] roisInBatch The ROIs for each stream, as pairs of bbox index and rectangle.
 * @param[in] lastBboxInd The last bbox index in the batch.
 * @param[in] doUpdateImages Whether to store input images on the first extraction call for a batch.
 * @return True when feature extraction succeeds.
 */
bool FeatureExtractor::extractFeatures_VPI(
    const std::map<NvMOTStreamId, NvMOTFrame*> &nvFramesInBatch,
    const std::map<NvMOTStreamId, std::vector<std::pair<uint, NvMotDataTypes::Rect2f> > > &roisInBatch,
    const uint lastBboxInd, const bool doUpdateImages
)
{
	if(m_initGpu != true)
	{
		throw std::runtime_error("[FeatureExtractor::extractFeatures_VPI] Make sure m_initGpu == true");
	}

	/// Upload a set of source images to device memory.
	if(doUpdateImages == true)
	{
		m_subImageCropper2->updateTextureObject(nvFramesInBatch);
	}

	if(m_subImageCropper2->getColorFormat() == NvBufSurfaceColorFormat::NVBUF_COLOR_FORMAT_NV12)
	{
		/// Given the source image in Texture, generate a cropped and scaled image on device memory based on the input ROI and template size.
		if(!m_subImageCropper2->frameTransformInBatch_NV12(roisInBatch, lastBboxInd, true))
		{
			throw std::runtime_error("[FAILED] m_subImageCropper2->cropAndScale_batch_NV12()");
		}

		for(uint multiResInd = 0; multiResInd < (uint)m_subImageCropper2->getMultiResScaleNum(); multiResInd++)
		{
			/// Get the output feature metadata.
			m_subImageCropper2->getOutputImagePatchPtrs(
				m_srcImgPatchPtr_d[multiResInd], m_srcSubImgWidth[multiResInd], m_srcSubImgHeight[multiResInd],
				m_outBatchingGrid, m_srcImgColorChannels, m_srcImgPatchPitch[multiResInd], m_srcImgPatchNum, multiResInd
			);
		}
		// DEBUG: Show cropped/scaled sub-images.
		#if defined(VIZ_DEBUG) && !defined(__aarch64__) && !defined(DEEPSTREAM_SDK)
			if(0)
			{
				std::map<NvMOTStreamId, std::map<uint, cv::Mat > > subImgs = m_subImageCropper2->getImagePatches2Host_OCV_mosaic(0);

				for(auto it = subImgs.begin(); it != subImgs.end(); ++it)
				{
					NvMOTStreamId streamId = it->first;
					for(auto jt = it->second.begin(); jt != it->second.end(); ++jt)
					{
						uint targetInd = jt->first;

						cv::Mat subImg = jt->second;

						cv::resize(subImg, subImg, cv::Size(120,120));

						std::string label = "V[" + std::to_string(targetInd) + "]";
						cv::imshow(label, subImg);
					}
				}
			}
		#endif
	}
	else if(m_subImageCropper2->getColorFormat() == NvBufSurfaceColorFormat::NVBUF_COLOR_FORMAT_RGBA)
	{
		/// Given the source image in Texture, generate a cropped and scaled image on device memory based on the input ROI and template size.
		if(!m_subImageCropper2->frameTransformInBatch(roisInBatch, lastBboxInd))
		{
			throw std::runtime_error("[FAILED] m_subImageCropper2->cropAndScale_batch()");
		}

		for(uint multiResInd = 0; multiResInd < (uint)m_subImageCropper2->getMultiResScaleNum(); multiResInd++)
		{
			/// Get the output feature metadata.
			m_subImageCropper2->getOutputImagePatchPtrs(
				m_srcImgPatchPtr_d[multiResInd], m_srcSubImgWidth[multiResInd], m_srcSubImgHeight[multiResInd],
				m_outBatchingGrid, m_srcImgColorChannels, m_srcImgPatchPitch[multiResInd], m_srcImgPatchNum, multiResInd
			);

			// DEBUG: Show cropped/scaled sub-images.
			#if defined(VIZ_DEBUG) && !defined(__aarch64__) && !defined(DEEPSTREAM_SDK)
				if(0)
				{

					std::map<NvMOTStreamId, std::map<uint, cv::Mat > > subImgs = m_subImageCropper2->getImagePatches2Host_OCV_batch(multiResInd);

					for(auto it = subImgs.begin(); it != subImgs.end(); ++it)
					{
						NvMOTStreamId streamId = it->first;
						for(auto jt=it->second.begin(); jt != it->second.end(); ++jt)
						{
							uint targetInd = jt->first;

							cv::Mat subImg = jt->second;
							cvtColor(subImg, subImg, cv::COLOR_RGBA2BGR);

							cv::resize(subImg, subImg, cv::Size(240,240));

							std::string label = "V[" + std::to_string(targetInd) + "]";
							cv::imshow(label, subImg);
						}
					}
				}
			#endif
		}
	}
	else
	{
		throw std::runtime_error("[FAILED] Invalid color format");
	}

	/// In case of no ROI, stop the process.
	if(roisInBatch.size() == 0)
		return false;

	if(!m_isVpiInitialized)
	{
		/// Initialize and allocate memories for intputs/outputs for VPI calls.
		initialize_VPI();
	}
	else
	{
		if(m_featureTypes & FEATURE_TYPES::COLOR_NAMES)
		{
			/// Set VPIImage for ColorNames feature extraction.
			VPI_CHECK( vpiImageSetWrapper (m_vpiInputImage[0], &m_vpiInputImageData[0]) );
		}

		if(m_featureTypes & FEATURE_TYPES::HOG)
		{
			/// Set VPIImage for HOG feature extraction.
			VPI_CHECK( vpiImageSetWrapper (m_vpiInputImage[1], &m_vpiInputImageData[1]) );
		}
	}

	uint numFeatureChannels = 0;

	if(m_featureTypes & FEATURE_TYPES::COLOR_NAMES)
	{
		/// Process for ColorNames.
		VPI_CHECK(
			vpiSubmitExtractColorNameFeatures(
				m_vpiStream, m_vpiDevType, m_vpiColorNames, m_vpiInputImage[0],
				m_vpiOutFeatures.data(), m_numOfChannelsByColorNames
			)
		);

		numFeatureChannels += m_numOfChannelsByColorNames;
	}

	if(m_featureTypes & FEATURE_TYPES::HOG)
	{
		/// Process for HOG.
		VPI_CHECK(
			vpiSubmitExtractHOGFeatures(
				m_vpiStream, m_vpiDevType, m_vpiHOG, m_vpiInputImage[1],
				m_vpiOutFeatures.data() + numFeatureChannels, m_numOfChannelsByHog
			)
		);

		numFeatureChannels += m_numOfChannelsByHog;
	}

	/// Wait until VPI processing is done.
	VPI_CHECK( vpiStreamSync(m_vpiStream) );

	/// To catch any CUDA error in VPI in case it's not caught inside VPI.
	CUDA_CHECK( cudaPeekAtLastError() );

	return true;
}

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
uchar *FeatureExtractor::getOutputFeaturePtrs_VPI(
	uint &width, uint &height, uint2 &outBatchingGrid, uint &channels, uint &pitch, uint &imgNum
)
{
	width = m_srcSubImgWidth[0];
	height = m_dstSubImgHeight[0];
	outBatchingGrid = m_outBatchingGrid;
	channels = m_numFeatureChannelsOut;
	pitch = m_batch_vpi_ftrs_pitch;
	imgNum = m_srcImgPatchNum;

	return reinterpret_cast<uchar*>( m_batch_vpi_ftrs );
}

#if defined(VIZ_DEBUG)
/**
 * @brief Get VPI features copied to host for debugging or visualization.
 * @return The feature vectors copied to host memory.
 */
std::vector<std::vector<cv::Mat> > FeatureExtractor::getExtractedFeaturesAtHost_VPI() const
{
	if(m_batch_array_host_real == nullptr)
	{
		throw std::runtime_error(
			"[FeatureExtractor::getExtractedFeaturesAtHost_VPI] Make sure m_batch_array_host_real != nullptr"
		);
	}

	/// Host vector, real part.
	uchar *real_vec = reinterpret_cast<uchar*>(m_batch_array_host_real);
	/// Device vector, single-precision complex.
	uchar *ftrs_d = reinterpret_cast<uchar*>( m_batch_vpi_ftrs );

	size_t totalSize =  m_batch_vpi_ftrs_pitch * m_dstSubImgHeight[0] * m_outBatchingGrid.y * m_vpiOutFeatures.size();

	assert(m_numFeatureChannelsOut == m_vpiOutFeatures.size());

	/// Get the cuComplex-type data from device to host, but only the real part.
	CUDA_CHECK(
		cudaMemcpy2D(
			(void*)real_vec, m_batch_vpi_ftrs_pitch, (void*)ftrs_d, m_batch_vpi_ftrs_pitch, m_batch_vpi_ftrs_pitch,
			m_outBatchingGrid.y*m_numFeatureChannelsOut*m_dstSubImgHeight[0], cudaMemcpyDeviceToHost
		)
	);

	std::vector<std::vector<cv::Mat> > featuresAtHost(m_srcImgPatchNum);

	if(!(m_dstSubImgHeight[0] > 0 && (m_dstSubImgHeight[0] == m_srcSubImgHeight[0] || m_dstSubImgHeight[0] == m_srcSubImgHeight[0]*2/3)))
	{
		throw std::runtime_error(
			"[FeatureExtractor::getExtractedFeaturesAtHost_VPI] Make sure (m_dstSubImgHeight[0] > 0 && (m_dstSubImgHeight[0] == m_srcSubImgHeight[0] || m_dstSubImgHeight[0] == m_srcSubImgHeight[0]*2/3))"
		);
	}

	cv::Mat featNormalized;
	cv::Mat featNormConv;

	/// For N targets with M feature channels, the extracted features on CUDA memory are ordered as M_{0} through M_{M-1}, and each block consists of N_{0} through N_{N-1}.
	/// This structure is re-ordered per target in cuda_calculateResponse().
	for(uint ch=0; ch < m_vpiOutFeatures.size(); ++ch)
	{
		uint bboxInd=0;
		for(uint y=0; y < m_outBatchingGrid.y; y++)
		{
			for(uint x=0; x < m_outBatchingGrid.x && bboxInd < m_srcImgPatchNum; x++, bboxInd++)
			{
				if(m_vpiImgOutputType == VPI_IMAGE_FORMAT_U8)
				{
					featNormalized = cv::Mat(
						m_dstSubImgHeight[0], m_srcSubImgWidth[0], CV_8UC1,
						reinterpret_cast<uchar*>(real_vec)
							+ m_batch_vpi_ftrs_pitch*m_dstSubImgHeight[0]*m_outBatchingGrid.y*ch // Offset for channel [Bytes].
							+ m_batch_vpi_ftrs_pitch*m_dstSubImgHeight[0]*y // Offset for Y direction [Bytes].
							+ m_srcSubImgWidth[0]*sizeof(uint8_t)*x, // Offset for X direction [Bytes].
						m_batch_vpi_ftrs_pitch
					);

					featNormalized.convertTo(featNormConv, CV_32FC1);

					/// Normalize the range to [-1, 1).
					/// From [0, 255) to [-127, 128].
					featNormConv -= (float)(1<<7);
					/// From [-127, 128] to (-1, 1).
					featNormConv /= (float)(1<<7);
				}
				else if(m_vpiImgOutputType == VPI_IMAGE_FORMAT_U16)
				{
					featNormalized = cv::Mat(
						m_dstSubImgHeight[0], m_srcSubImgWidth[0], CV_16UC1,
						reinterpret_cast<uchar*>(real_vec)
							+ m_batch_vpi_ftrs_pitch*m_dstSubImgHeight[0]*m_outBatchingGrid.y*ch // Offset for channel [Bytes].
							+ m_batch_vpi_ftrs_pitch*m_dstSubImgHeight[0]*y // Offset for Y direction [Bytes].
							+ m_srcSubImgWidth[0]*sizeof(uint16_t)*x, // Offset for X direction [Bytes].
						m_batch_vpi_ftrs_pitch
					);

					featNormalized.convertTo(featNormConv, CV_32FC1);

					/// Normalize the range to [-1, 1).
					/// From [0, 65536) to [-32767, 32768].
					featNormConv -= (float)(1<<15);
					/// From [-32767, 32768] to (-1, 1).
					featNormConv /= (float)(1<<15);
				}
				else
				{
					throw std::runtime_error(
						"[FeatureExtractor::getExtractedFeaturesAtHost_VPI] Invalid m_vpiImgOutputType"
					);
				}

				featuresAtHost[bboxInd].push_back(std::move(featNormConv));
			}
		}
	}

	return std::move(featuresAtHost);
}
#endif
