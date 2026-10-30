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

#include "modules/utility/cuda_common.hpp"
#include "modules/utility/debug_common.hpp"
#include "modules/utility/cv_functions.hpp"
#include "modules/VisualTracker/DCFTracker.hpp"

#ifdef VIZ_DEBUG
#include "modules/VisualTracker/vizOutput.hpp"
#endif

using std::cout;
using std::endl;

/// Upper limit for DCF filter and channel weights learning rate.
#define LEARN_RATE_UPPER_LIMIT 0.5
/// Check if a number is nan or inf.
#define IS_NAN(a)  (isnan(a) || isinf(a))

/**
 * @brief Construct from a configuration file path.
 * @param[in] configFilePath The path to the configuration file.
 */
DCFTracker::DCFTracker(const std::string &configFilePath)
{
	m_isInit = false;

	/// Parse the visual tracker configuration.
	m_params = DCFTrackerConfigParams(configFilePath, "VisualTracker");

	m_integrated_gpu = 0;
	NvBufSurfaceDeviceInfo dev_info{};
	if (NvBufSurfaceGetDeviceInfo(&dev_info) == 0) {
		if (dev_info.driverType == NVBUF_DRIVER_TYPE_NVGPU) {
			m_integrated_gpu = 1;
		}
	}
}

/** @brief Release VPI DCF tracker resources. */
DCFTracker::~DCFTracker()
{
	#ifdef VIZ_DEBUG
		if(m_targetManager.get() != nullptr && m_targetManager.use_count() > 0)
		{
			m_targetManager->removeAllStreams();

			if(m_vizOutput != nullptr && isInitialized())
			{
				if(m_params.m_vizWriteAllTrajectoriesOnEveryFrame)
				{
					cout << "[DCFTracker][Debug] Wrapping-up visualization" << endl;

					m_vizOutput->writeAllTrajectoriesOnEveryFrame(VIZ_OUTPUT_TYPES::TRACKER_STATES_VIZ);
				}

				if(m_params.m_vizWriteEachCompletedTrajectory)
				{
					cout << "[DCFTracker][Debug] Wrapping-up visualization" << endl;

					m_vizOutput->writeEachCompletedTrajectory(VIZ_OUTPUT_TYPES::TRACKER_STATES_VIZ);
				}

				if(m_params.m_vizWriteBirdsEyeView)
				{
					cout << "[DCFTracker][Debug] Wrapping-up visualization" << endl;

					m_vizOutput->writeBirdsEyeView(VIZ_OUTPUT_TYPES::TRACKER_STATES_VIZ);
				}
			}
		}
	#endif

	if(m_isInit)
	{
		CUDA_CHECK(cudaFree( m_corrResponses_d ));
		CUDA_CHECK(cudaFree( m_maskingWindow4Src ));
		CUDA_CHECK(cudaFree( m_maskingWindow4Model ));

		CUDA_CHECK(cudaFree(m_maxCorrResponse_d));
		CUDA_CHECK(cudaFree(m_maxCorrResponse_ma_d));
		CUDA_CHECK(cudaFree(m_trackingConfidence_d));

		CUDA_CHECK(cudaFreeHost(m_maxCorrResponse_h));
		CUDA_CHECK(cudaFreeHost(m_maxCorrResponse_ma_h));
		CUDA_CHECK(cudaFreeHost(m_trackingConfidence_h));

		CUDA_CHECK(cudaFree(m_trackerAge_d));
		CUDA_CHECK(cudaFreeHost(m_trackerAge_h));

		CUDA_CHECK(cudaFree(m_trackerDoProcessing_d));
		CUDA_CHECK(cudaFreeHost(m_trackerDoProcessing_h));

		CUDA_CHECK(cudaFree(m_detectorBboxes_d));
		CUDA_CHECK(cudaFreeHost(m_detectorBboxes_h));

		CUDA_CHECK(cudaFree(m_detectorBboxes_size_d));
		CUDA_CHECK(cudaFreeHost(m_detectorBboxes_size_h));

		CUDA_CHECK(cudaFree(m_confidenceAtDetectedPt_d));
		CUDA_CHECK(cudaFreeHost(m_confidenceAtDetectedPt_h));

		CUDA_CHECK(cudaFree(m_corrResponseAtDetectedPt_d));
		CUDA_CHECK(cudaFreeHost(m_corrResponseAtDetectedPt_h));

		#ifdef VIZ_DEBUG
			CUDA_CHECK(cudaFreeHost(m_batchCudaArray_host));
			CUDA_CHECK( cudaFreeHost(m_idata_host) );
			CUDA_CHECK(cudaFreeHost(m_targetDisplacement2D_h));
		#endif

		CUDA_CHECK(cudaStreamDestroy(m_cudaStream));
		vpiStreamDestroy(m_vpiStream);
		vpiPayloadDestroy(m_vpiDcfPayload);
		vpiPayloadDestroy(m_vpiCropPayload);

		vpiImageDestroy(m_vpiMaskingWindow4SrcImage);
		vpiImageDestroy(m_vpiMaskingWindow4ModelImage);
		for(unsigned int i = 0; i < m_isFrameCreated.size(); i++)
		{
			if(m_isFrameCreated[i] == true)
				vpiImageDestroy(m_vpiInputFrameSequence[i]);
		}
		CUDA_CHECK(cudaFreeHost(m_vpiInputFrameSequence));
		CUDA_CHECK(cudaFreeHost(m_vpiInputFrameDataList));
		m_isFrameCreated.clear();
		CUDA_CHECK(cudaFreeHost(m_enabledSequences));

		m_vpiDCFBboxes.clear();
		vpiArrayDestroy(m_vpiDCFBboxesArray);
		m_vpiDCFBboxesLocalized.clear();
		vpiArrayDestroy(m_vpiDCFBboxesLocalizedArray);

		CUDA_CHECK(cudaFree(m_vpiObjPatches_d));
		#ifdef VIZ_DEBUG
			CUDA_CHECK(cudaFreeHost(m_vpiObjPatches_h));
		#endif
		vpiImageDestroy(m_vpiObjPatches);

		vpiImageDestroy(m_vpiCorrResponsesImage);
		vpiArrayDestroy(m_vpiMaxCorrResponseArray);
		vpiArrayDestroy(m_vpiChannelWeightsArray);

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

		if(m_params.m_debugVerboseLevel > 1)
			cout << "[DCFTracker] De-initialized" << endl;
	}

	if(m_params.m_showGpuTiming)
	{
		cudaEventDestroy(m_eventStart);
		cudaEventDestroy(m_eventStop);
	}

	#ifdef VIZ_DEBUG
		m_vizOutput.reset();
	#endif
}

/**
 * @brief Initialize variables and allocate memories.
 * @param[in] frameInfo Metadata about the source image (e.g., size, color).
 * @param[in] maxTargetsToTrack The maximum number of targets to track.
 * @param[in] maxDetectorBboxesPerFrame The maximum number of detector bounding boxes per frame.
 * @param[in] targetManager The target manager through which all the targets can be accessed.
 */
void DCFTracker::initialize(
	const SourceFrameInfo &frameInfo, const uint maxTargetsToTrack,
	const uint maxDetectorBboxesPerFrame, const std::shared_ptr<NvTrackedObjectManager> &targetManager
)
{
	m_targetManager = targetManager;

	#ifdef VIZ_DEBUG
		m_vizOutput = std::make_shared<TrackerVizOutput>(this, targetManager);

		if(m_params.m_generateTrackerVisualization && !m_vizOutput->isImageOutputEnabled(TRACKER_STATES_VIZ))
		{
			m_vizOutput->enableImageOutput(
				m_params.m_vizOutDirectory, m_params.m_vizJpegImageQuality, TRACKER_STATES_VIZ
			);
		}
	#endif

	if(m_targetManager == nullptr)
	{
		throw std::runtime_error("[DCFTracker::initialize()] m_targetManager == nullptr");
	}

	m_paramsRt.maxTargetsToTrack = maxTargetsToTrack;
	m_paramsRt.maxDetectorBboxesPerFrame = maxDetectorBboxesPerFrame;
	m_paramsRt.maxStreams = frameInfo.maxStreams;
	m_paramsRt.m_maxBufSurfAddrSize = frameInfo.maxBufSurfAddrSize;
	m_paramsRt.originalSrcImageSize = frameInfo.originalSrcImageSize;
	m_paramsRt.srcImageSize = frameInfo.srcImageSize;
	m_paramsRt.srcImagePitch = frameInfo.srcImagePitch;
	m_paramsRt.colorFormat = frameInfo.colorFormat;
	m_paramsRt.skipBatchUpdate = true;

	/// If m_useHog is set to false, rectify m_hogCellSize & m_featureImgSize.
	if(!m_params.m_useHog)
	{
		m_params.m_featureImgSize = m_params.m_featureImgSize/m_params.m_hogCellSize;
		m_params.m_hogCellSize = 1;
	}

	m_paramsRt.templateSize = NvMotDataTypes::Size(m_params.m_featureImgSize, m_params.m_featureImgSize);

	if(m_params.m_debugVerboseLevel > 1)
	{
		cout << "[DCFTracker::initialize] OriginalSrcImageSize = " << m_paramsRt.originalSrcImageSize.width << "x" << m_paramsRt.originalSrcImageSize.height << endl;
		cout << "[DCFTracker::initialize] SrcImageSize = " << m_paramsRt.srcImageSize.width << "x" << m_paramsRt.srcImageSize.height << endl;
	}

	/// Configure the search window used to tolerate detector jitter.
	/// A little heuristics on the search window size for finding the corr resp value for a detector bbox.
	/// The width and height of the search window size will be (searchWindowSize*2+1).
	m_paramsRt.detCorrRespSearchWindowSize = (m_params.m_featureImgSize/m_params.m_hogCellSize*m_params.m_gaussianSigma/5.f - 1) + 0.5f;
	/// Make the range be [1, 2].
	m_paramsRt.detCorrRespSearchWindowSize = std::max(1, std::min(2, m_paramsRt.detCorrRespSearchWindowSize));

	/// This is the feature matrix size for a single object for non-HOG feature like ColorNames.
	m_paramsRt.singleFeatureMatSize = NvMotDataTypes::Size(
		m_paramsRt.templateSize.width / m_params.m_hogCellSize,
		m_paramsRt.templateSize.height / m_params.m_hogCellSize
	);

	if(m_paramsRt.singleFeatureMatSize.width == 0 || m_paramsRt.singleFeatureMatSize.height == 0)
	{
		throw std::runtime_error(
			"[DCFTracker::initialize()] Make sure m_paramsRt.singleFeatureMatSize.width > 0 && m_paramsRt.singleFeatureMatSize.height > 0"
		);
	}

	/// Require a square image so inversion preserves its dimensions.
	if(m_paramsRt.singleFeatureMatSize.width != m_paramsRt.singleFeatureMatSize.height)
	{
		throw std::runtime_error(
			"[DCFTracker::initialize()] Make sure m_paramsRt.singleFeatureMatSize.width == m_paramsRt.singleFeatureMatSize.height"
		);
	}

	size_t freeGpuMemory_init = 0, freeGpuMemory_prev = 0;

	if(m_params.m_debugVerboseLevel_gpumemory > 0)
	{
		size_t free, total;
		CUDA_CHECK( cudaMemGetInfo( &free, &total ) );

		freeGpuMemory_init = free;
		freeGpuMemory_prev = free;

		printf(
			"[DCFTracker][GPU] Free mem = %d [MB] among Total = %d [MB]\n", int(free/1024/1024), int(total/1024/1024)
		);
	}

	m_paramsRt.singleFeatureChannelElementSize =  static_cast<int>(
		m_paramsRt.singleFeatureMatSize.width*m_paramsRt.singleFeatureMatSize.height
	);
	if(m_params.m_debugVerboseLevel > 1)
	{
		printf(
			"[DCFTracker::initialize()] m_paramsRt.singleFeatureMatSize = (%d, %d)\n",
			m_paramsRt.singleFeatureMatSize.width, m_paramsRt.singleFeatureMatSize.height
		);
	}

	{
		/// Initialize VPI contexts.
		VPI_CHECK(
			vpiStreamCreate(
				(m_params.m_vpiBackend4CropnScale | m_params.m_vpiBackend4DcfTracker | VPI_BACKEND_CPU), &m_vpiStream
			)
		);
		m_vpiMemFlags = m_vpiMemFlags | m_params.m_vpiBackend4CropnScale | m_params.m_vpiBackend4DcfTracker;
		VPI_CHECK(vpiInitDCFTrackerCreationParams(&m_vpiCreateParams));
		m_vpiCreateParams.flags = 0;
		if (m_params.m_useColorNames)
		{
			m_vpiCreateParams.flags |= VPI_DCFTRACKER_USE_COLORNAMES;
		}

		if (m_params.m_useHog)
		{
			m_vpiCreateParams.flags |= VPI_DCFTRACKER_USE_HOG;
		}

		if (m_params.m_useHighPrecisionFeature)
		{
			m_vpiCreateParams.flags |= VPI_DCFTRACKER_USE_HIGH_PRECISION_FEATURES;
		}

		m_vpiCreateParams.featurePatchSize = m_paramsRt.singleFeatureMatSize.width;
		m_vpiCreateParams.hogCellSize = m_params.m_hogCellSize;
		m_vpiCreateParams.featureFocusVertOffsetFactor = m_params.m_featureFocusOffsetFactor_y;
		m_vpiCreateParams.gaussianSigma = m_params.m_gaussianSigma;

		VPI_CHECK(
			vpiCreateDCFTracker(
				m_params.m_vpiBackend4DcfTracker, m_paramsRt.maxStreams,
				m_paramsRt.maxTargetsToTrack, &m_vpiCreateParams, &m_vpiDcfPayload
			)
		);
		if (m_params.m_debugVerboseLevel_gpumemory > 0)
		{
			size_t free, total;
			CUDA_CHECK(cudaMemGetInfo(&free, &total));
			printf(
				"[DCFTracker][GPU] [vpiCreateDCFTracker] Free mem = %d [MB] among Total = %d [MB] --> memory increase = %d [MB]\n",
				int(free/1024/1024), int(total/1024/1024), int((freeGpuMemory_prev-free)/1024/1024)
			);
			freeGpuMemory_prev = free;
		}
		VPI_CHECK(
			vpiCreateCropScaler(
				m_params.m_vpiBackend4CropnScale, m_paramsRt.maxStreams, m_paramsRt.maxTargetsToTrack, &m_vpiCropPayload
			)
		);
		if (m_params.m_debugVerboseLevel_gpumemory > 0)
		{
			size_t free, total;
			CUDA_CHECK(cudaMemGetInfo(&free, &total));
			printf(
				"[DCFTracker][GPU] [vpiCreateCropScaler] Free mem = %d [MB] among Total = %d [MB] --> memory increase = %d [MB]\n",
				int(free/1024/1024), int(total/1024/1024), int((freeGpuMemory_prev-free)/1024/1024)
			);
			freeGpuMemory_prev = free;
		}

		/// Make a dummy call to get number of feature channels.
		VPI_CHECK(vpiDCFTrackerGetChannelWeights(m_vpiDcfPayload, &m_vpiChannelWeightsArray, &m_featureChannelSize));
		if(m_featureChannelSize == 0)
		{
			throw std::runtime_error(
				"[DCFTracker::initialize()] numFeatureChannels returned by vpiDCFTrackerGetChannelWeights() is 0"
			);
		}
		m_cudaKernelMaxBatchSize = m_featureChannelSize*m_paramsRt.maxTargetsToTrack;

		/// Populate VPI runtime params.
		m_vpiRuntimeParams.dcfRegFactor = m_params.m_lambda;

		/// Set the number of channels for m_vpiObjPatches based on DCF tracker backend.
		if(m_params.m_vpiBackend4DcfTracker == VPI_BACKEND_CUDA)
		{
			/// VPI_IMAGE_FORMAT_RGBA8.
			m_vpiObjPatchesNumChannels = 4;
		}
		else if(m_params.m_vpiBackend4DcfTracker == VPI_BACKEND_PVA)
		{
			/// VPI_IMAGE_FORMAT_RGB8p.
			m_vpiObjPatchesNumChannels = 3;
		}
		else
			throw std::runtime_error("Unsupported config for vpiBackend4DcfTracker");
	}

	/// Allocate custom CUDA kernel memories.
	allocateCudaMemories();

	if(m_params.m_debugVerboseLevel_gpumemory > 0)
	{
		size_t free, total;
		CUDA_CHECK( cudaMemGetInfo( &free, &total ) );
		printf(
			"[DCFTracker][GPU]	[after all custom CUDA memories allocated] Free mem = %d [MB] among Total = %d [MB] --> memory increase = %d [MB]\n",
			int(free/1024/1024), int(total/1024/1024), int((freeGpuMemory_prev-free)/1024/1024)
		);
		freeGpuMemory_prev = free;
	}

	/// Initialize the tracker states.
	{
		CUDA_CHECK( cudaStreamCreateWithFlags(&m_cudaStream, cudaStreamNonBlocking) );

		CUDA_CHECK( cudaMemsetAsync(m_trackerAge_d, 0, m_paramsRt.maxTargetsToTrack*sizeof(uint32_t), m_cudaStream) );
		memset(m_trackerAge_h, 0, m_paramsRt.maxTargetsToTrack*sizeof(uint32_t));

		CUDA_CHECK(
			cudaMemsetAsync(m_trackerDoProcessing_d, false, m_paramsRt.maxTargetsToTrack*sizeof(bool), m_cudaStream)
		);
		memset(	m_trackerDoProcessing_h, false, m_paramsRt.maxTargetsToTrack*sizeof(bool));

		CUDA_CHECK( cudaStreamSynchronize(m_cudaStream) );
		CUDA_CHECK( cudaPeekAtLastError() );
	}

	/// Create a Hanning window for smoother image borders.
	NvMotDataTypes::RowMajorMatrixXf maskingWindow4Src = CV_FUNCS::getHanningWindow(
		m_paramsRt.singleFeatureMatSize.width,
		m_paramsRt.singleFeatureMatSize.height, m_params.m_featureFocusOffsetFactor_y
	);

	/// Narrower window for model update for Focused Windowing.
	NvMotDataTypes::RowMajorMatrixXf maskingWindow4Model = pow(
		maskingWindow4Src.array(), (float)m_params.m_modelMaskingWindowFactor
	);

	{
		/// Allocate the host batch buffer used to download and de-batch GPU matrices.
		float *workarea_host;

		/// Allocate a temporary memory in the host the batch-array.
		CUDA_CHECK(
			cudaMallocHost(
				reinterpret_cast<void **>(&workarea_host),
				m_paramsRt.singleFeatureChannelElementSize * sizeof(cuComplex)
			)
		);

		/// Create VPI image for masking window for model.
		makeBatchAndSendFloatMatToDevice(maskingWindow4Model, m_maskingWindow4Model, 1, workarea_host);
		m_vpiMaskingWindow4ModelData = {};
		m_vpiMaskingWindow4ModelData.bufferType = VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR;
		m_vpiMaskingWindow4ModelData.buffer.pitch.format = VPI_IMAGE_FORMAT_F32;
		m_vpiMaskingWindow4ModelData.buffer.pitch.numPlanes = 1;
		m_vpiMaskingWindow4ModelData.buffer.pitch.planes[0].pixelType = VPI_PIXEL_TYPE_F32;
		m_vpiMaskingWindow4ModelData.buffer.pitch.planes[0].pBase = reinterpret_cast<VPIByte *>(m_maskingWindow4Model);
		m_vpiMaskingWindow4ModelData.buffer.pitch.planes[0].offsetBytes = 0;
		m_vpiMaskingWindow4ModelData.buffer.pitch.planes[0].width = m_paramsRt.singleFeatureMatSize.width;
		m_vpiMaskingWindow4ModelData.buffer.pitch.planes[0].height = m_paramsRt.singleFeatureMatSize.height;
		m_vpiMaskingWindow4ModelData.buffer.pitch.planes[0].pitchBytes = m_paramsRt.singleFeatureMatSize.width * sizeof(
			float
		);
		VPI_CHECK(
			vpiImageCreateWrapper(&m_vpiMaskingWindow4ModelData, nullptr, m_vpiMemFlags, &m_vpiMaskingWindow4ModelImage)
		);

		/// Create VPI image for masking window for src.
		makeBatchAndSendFloatMatToDevice(maskingWindow4Src, m_maskingWindow4Src, 1, workarea_host);
		m_vpiMaskingWindow4SrcData = {};
		m_vpiMaskingWindow4SrcData.bufferType = VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR;
		m_vpiMaskingWindow4SrcData.buffer.pitch.format = VPI_IMAGE_FORMAT_F32;
		m_vpiMaskingWindow4SrcData.buffer.pitch.numPlanes = 1;
		m_vpiMaskingWindow4SrcData.buffer.pitch.planes[0].pixelType = VPI_PIXEL_TYPE_F32;
		m_vpiMaskingWindow4SrcData.buffer.pitch.planes[0].pBase = reinterpret_cast<VPIByte *>(m_maskingWindow4Src);
		m_vpiMaskingWindow4SrcData.buffer.pitch.planes[0].offsetBytes = 0;
		m_vpiMaskingWindow4SrcData.buffer.pitch.planes[0].width = m_paramsRt.singleFeatureMatSize.width;
		m_vpiMaskingWindow4SrcData.buffer.pitch.planes[0].height = m_paramsRt.singleFeatureMatSize.height;
		m_vpiMaskingWindow4SrcData.buffer.pitch.planes[0].pitchBytes = m_paramsRt.singleFeatureMatSize.width * sizeof(
			float
		);
		VPI_CHECK(
			vpiImageCreateWrapper(&m_vpiMaskingWindow4SrcData, nullptr, m_vpiMemFlags, &m_vpiMaskingWindow4SrcImage)
		);

		/// Create VPI image for object patches.
		if(m_params.m_vpiBackend4DcfTracker == VPI_BACKEND_CUDA)
		{
			m_vpiObjPatchesData = {};
			m_vpiObjPatchesData.bufferType = VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR;
			m_vpiObjPatchesData.buffer.pitch.format = VPI_IMAGE_FORMAT_RGBA8;
			m_vpiObjPatchesData.buffer.pitch.numPlanes = 1;
			/// 4-ch.
			m_vpiObjPatchesData.buffer.pitch.planes[0].pixelType = VPI_PIXEL_TYPE_4U8;
			/// Use m_vpiObjPatches_d as a raw pointer to pitch-linear CUDA memory.
			m_vpiObjPatchesData.buffer.pitch.planes[0].pBase = m_vpiObjPatches_d;
			m_vpiObjPatchesData.buffer.pitch.planes[0].offsetBytes = 0;
			m_vpiObjPatchesData.buffer.pitch.planes[0].width = m_params.m_featureImgSize;
			/// Consider the stacked image patches as a single image.
			m_vpiObjPatchesData.buffer.pitch.planes[0].height = m_params.m_featureImgSize * m_paramsRt.maxTargetsToTrack;
			m_vpiObjPatchesData.buffer.pitch.planes[0].pitchBytes = m_params.m_featureImgSize * m_vpiObjPatchesNumChannels;
			VPI_CHECK(vpiImageCreateWrapper(&m_vpiObjPatchesData, nullptr, m_vpiMemFlags, &m_vpiObjPatches));
		}
		else
		{
			if(m_params.m_vpiBackend4DcfTracker != VPI_BACKEND_PVA)
			{
				throw std::invalid_argument("Invalid vpiBackend4DcfTracker");
			}

			m_vpiObjPatchesData = {};
			m_vpiObjPatchesData.bufferType = VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR;
			m_vpiObjPatchesData.buffer.pitch.format = VPI_IMAGE_FORMAT_RGB8p;
			m_vpiObjPatchesData.buffer.pitch.numPlanes = 3;
			/// R plane.
			/// 1-ch.
			m_vpiObjPatchesData.buffer.pitch.planes[0].pixelType = VPI_PIXEL_TYPE_U8;
			/// Use m_vpiObjPatches_d as a raw pointer to pitch-linear CUDA memory.
			m_vpiObjPatchesData.buffer.pitch.planes[0].pBase = m_vpiObjPatches_d;
			m_vpiObjPatchesData.buffer.pitch.planes[0].offsetBytes = 0;
			m_vpiObjPatchesData.buffer.pitch.planes[0].width = m_params.m_featureImgSize;
			/// Consider the stacked image patches as a single image.
			m_vpiObjPatchesData.buffer.pitch.planes[0].height = m_params.m_featureImgSize * m_paramsRt.maxTargetsToTrack;
			m_vpiObjPatchesData.buffer.pitch.planes[0].pitchBytes = m_params.m_featureImgSize;
			/// G plane.
			/// 1-ch.
			m_vpiObjPatchesData.buffer.pitch.planes[1].pixelType = VPI_PIXEL_TYPE_U8;
			/// Use m_vpiObjPatches_d as a raw pointer to pitch-linear CUDA memory.
			m_vpiObjPatchesData.buffer.pitch.planes[1].pBase = reinterpret_cast<uint8_t *>(m_vpiObjPatches_d)+uint(
				m_params.m_featureImgSize*m_params.m_featureImgSize*m_paramsRt.maxTargetsToTrack
			);
			m_vpiObjPatchesData.buffer.pitch.planes[1].offsetBytes = 0;
			m_vpiObjPatchesData.buffer.pitch.planes[1].width = m_params.m_featureImgSize;
			/// Consider the stacked image patches as a single image.
			m_vpiObjPatchesData.buffer.pitch.planes[1].height = m_params.m_featureImgSize * m_paramsRt.maxTargetsToTrack;
			m_vpiObjPatchesData.buffer.pitch.planes[1].pitchBytes = m_params.m_featureImgSize;
			/// B plane.
			/// 1-ch.
			m_vpiObjPatchesData.buffer.pitch.planes[2].pixelType = VPI_PIXEL_TYPE_U8;
			/// Use m_vpiObjPatches_d as a raw pointer to pitch-linear CUDA memory.
			m_vpiObjPatchesData.buffer.pitch.planes[2].pBase = reinterpret_cast<uint8_t *>(m_vpiObjPatches_d)+uint(
				m_params.m_featureImgSize*m_params.m_featureImgSize*m_paramsRt.maxTargetsToTrack*2
			);
			m_vpiObjPatchesData.buffer.pitch.planes[2].offsetBytes = 0;
			m_vpiObjPatchesData.buffer.pitch.planes[2].width = m_params.m_featureImgSize;
			/// Consider the stacked image patches as a single image.
			m_vpiObjPatchesData.buffer.pitch.planes[2].height = m_params.m_featureImgSize * m_paramsRt.maxTargetsToTrack;
			m_vpiObjPatchesData.buffer.pitch.planes[2].pitchBytes = m_params.m_featureImgSize;

			VPI_CHECK(vpiImageCreateWrapper(&m_vpiObjPatchesData, nullptr, m_vpiMemFlags, &m_vpiObjPatches));
		}

		CUDA_CHECK( cudaStreamSynchronize(m_cudaStream) );
		CUDA_CHECK( cudaPeekAtLastError() );

		/// Free the temporary memory.
		CUDA_CHECK( cudaFreeHost(workarea_host) );
	}
	/// Setup VPI related image and array wrappers.
	{
		/// Allocate memory for VPIImage frame list.
		CUDA_CHECK(
			cudaMallocHost(
				reinterpret_cast<void **>(&m_vpiInputFrameSequence), m_paramsRt.maxStreams * sizeof(VPIImage *)
			)
		);
		memset(m_vpiInputFrameSequence, 0, m_paramsRt.maxStreams * sizeof(VPIImage *));
		CUDA_CHECK(
			cudaMallocHost(
				reinterpret_cast<void **>(&m_vpiInputFrameDataList), m_paramsRt.maxStreams * sizeof(VPIImageData)
			)
		);
		memset(m_vpiInputFrameDataList, 0, m_paramsRt.maxStreams * sizeof(VPIImageData));
		m_isFrameCreated.reserve(m_paramsRt.maxStreams);
		m_isFrameCreated.resize(m_paramsRt.maxStreams, false);

		/// Allocate an array to maintain a list of enabled & disabled sequences.
		CUDA_CHECK(
			cudaMallocHost(reinterpret_cast<void **>(&m_enabledSequences), m_paramsRt.maxStreams * sizeof(int32_t))
		);
		memset(m_enabledSequences, 0, m_paramsRt.maxStreams * sizeof(int32_t));

		/// Reserve max required space in m_vpiDCFBboxes.
		m_vpiDCFBboxes.reserve(m_paramsRt.maxTargetsToTrack);
		/// NOTE: memset is used here instead of std::fill since the vector holds VPIDCFTrackedBoundingBox values.
		/// Wrap VPIDCFTrackedBoundingBox values in a VPI array.
		memset(&m_vpiDCFBboxes[0], 0, m_paramsRt.maxTargetsToTrack * sizeof(m_vpiDCFBboxes[0]));
		/// Create array for m_vpiDCFBboxes.
		m_vpiBboxesArraySize = 0;
		m_vpiDCFBboxesData           = {};
		m_vpiDCFBboxesData.bufferType = VPI_ARRAY_BUFFER_HOST_AOS;
		m_vpiDCFBboxesData.buffer.aos.type = VPI_ARRAY_TYPE_DCF_TRACKED_BOUNDING_BOX;
		m_vpiDCFBboxesData.buffer.aos.capacity = m_vpiDCFBboxes.capacity();
		m_vpiDCFBboxesData.buffer.aos.sizePointer = &m_vpiBboxesArraySize;
		m_vpiDCFBboxesData.buffer.aos.data = &m_vpiDCFBboxes[0];
		VPI_CHECK(vpiArrayCreateWrapper(&m_vpiDCFBboxesData, m_vpiMemFlags, &m_vpiDCFBboxesArray));

		/// Reserve max required space in m_vpiDCFBboxesLocalized.
		m_vpiDCFBboxesLocalized.reserve(m_paramsRt.maxTargetsToTrack);
		/// NOTE: memset is used here instead of std::fill since the vector holds VPIDCFTrackedBoundingBox values.
		/// Wrap localized VPIDCFTrackedBoundingBox values in a VPI array.
		memset(&m_vpiDCFBboxesLocalized[0], 0, m_paramsRt.maxTargetsToTrack * sizeof(m_vpiDCFBboxesLocalized[0]));
		/// Create array for m_vpiDCFBboxesLocalized.
		m_vpiDCFBboxesLocalizedData           = {};
		m_vpiDCFBboxesLocalizedData.bufferType = VPI_ARRAY_BUFFER_HOST_AOS;
		m_vpiDCFBboxesLocalizedData.buffer.aos.type = VPI_ARRAY_TYPE_DCF_TRACKED_BOUNDING_BOX;
		m_vpiDCFBboxesLocalizedData.buffer.aos.capacity = m_vpiDCFBboxesLocalized.capacity();
		m_vpiDCFBboxesLocalizedData.buffer.aos.sizePointer = &m_vpiBboxesArraySize;
		m_vpiDCFBboxesLocalizedData.buffer.aos.data = &m_vpiDCFBboxesLocalized[0];
		VPI_CHECK(vpiArrayCreateWrapper(&m_vpiDCFBboxesLocalizedData, m_vpiMemFlags, &m_vpiDCFBboxesLocalizedArray));

		/// Allocate VPIImage in which correlation response will be returned.
		/// It is a stack of images each of size (m_paramsRt.singleFeatureMatSize.width X m_paramsRt.singleFeatureMatSize.height).
		m_vpiCorrResponsesData = {};
		m_vpiCorrResponsesData.bufferType = VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR;
		m_vpiCorrResponsesData.buffer.pitch.format = VPI_IMAGE_FORMAT_F32;
		m_vpiCorrResponsesData.buffer.pitch.numPlanes = 1;
		m_vpiCorrResponsesData.buffer.pitch.planes[0].pixelType = VPI_PIXEL_TYPE_F32;
		/// Use m_corrResponses_d as a raw pointer to pitch-linear CUDA memory.
		m_vpiCorrResponsesData.buffer.pitch.planes[0].pBase = reinterpret_cast<VPIByte *>(m_corrResponses_d);
		m_vpiCorrResponsesData.buffer.pitch.planes[0].offsetBytes = 0;
		m_vpiCorrResponsesData.buffer.pitch.planes[0].width = m_paramsRt.singleFeatureMatSize.width;
		/// Consider the stacked image patches as a single image.
		m_vpiCorrResponsesData.buffer.pitch.planes[0].height = m_paramsRt.singleFeatureMatSize.height * m_paramsRt.maxTargetsToTrack;
		m_vpiCorrResponsesData.buffer.pitch.planes[0].pitchBytes = m_paramsRt.singleFeatureMatSize.width * sizeof(
			float
		);
		VPI_CHECK(vpiImageCreateWrapper(&m_vpiCorrResponsesData, nullptr, m_vpiMemFlags, &m_vpiCorrResponsesImage));

		/// Allocate VPIArray in which Max correlation response will be returned.
		/// It is one value per tracked object.
		/// Hence the size of the array is m_paramsRt.maxTargetsToTrack.
		m_vpiMaxCorrResponseArrayData = {};
		m_vpiMaxCorrResponseArrayData.bufferType = VPI_ARRAY_BUFFER_CUDA_AOS;
		m_vpiMaxCorrResponseArrayData.buffer.aos.type = VPI_ARRAY_TYPE_F32;
		m_vpiMaxCorrResponseArrayData.buffer.aos.capacity = m_paramsRt.maxTargetsToTrack;
		m_vpiMaxCorrResponseArrayData.buffer.aos.data = reinterpret_cast<void *>(m_maxCorrResponse_d);
		VPI_CHECK(vpiArrayCreateWrapper(&m_vpiMaxCorrResponseArrayData, m_vpiMemFlags, &m_vpiMaxCorrResponseArray));
	}

	if(m_params.m_debugVerboseLevel_gpumemory > 0)
	{
		size_t free, total;
		CUDA_CHECK( cudaMemGetInfo( &free, &total ) );

		printf(
			"[DCFTracker][GPU] Free mem = %d [MB], Total = %d [MB] ===> Total allocated memory = %d [MB]\n",
			int(free/1024/1024), int(total/1024/1024), int((freeGpuMemory_init - free)/1024/1024)
		);
	}

	m_paramsRt.detectorInputInterval = -1;
	m_isInit = true;

	if(m_params.m_debugVerboseLevel > 1)
		cout << "[DCFTracker] Initialized" << endl;
}

/**
 * @brief Populate the visual tracker's capability query.
 * @param[out] paramsQuery The output query parameters.
 */
void DCFTracker::queryParams(NvMOTQuery &paramsQuery)
{
	paramsQuery.numTransforms = 1;
}

/**
 * @brief Select a cuFFT plan large enough for the requested target count.
 * @param[in] num_target The number of targets to process.
 * @return The identifier of the selected cuFFT plan.
 */
inline int DCFTracker::getCufftPlanId(int num_target)
{
	float ind = num_target/float(1<<m_params.m_cufftHandleQuantFactor);

	return floorf(ind);
}

/** @brief Pre-allocate required cuda memories on GPU. */
void DCFTracker::allocateCudaMemories()
{
	size_t freeGpuMemory_prev = 0;
	if(m_params.m_debugVerboseLevel_gpumemory > 0)
	{
		size_t free, total;
		CUDA_CHECK( cudaMemGetInfo( &free, &total ) );
		freeGpuMemory_prev = free;
	}

	#ifdef VIZ_DEBUG
		/// Create in the host the batch-array.
		int featureVecMaxBatchElementSize = m_paramsRt.singleFeatureChannelElementSize*m_cudaKernelMaxBatchSize;
		CUDA_CHECK(
			cudaMallocHost(
				reinterpret_cast<void**>( &m_batchCudaArray_host ), featureVecMaxBatchElementSize*sizeof(cuComplex)
			)
		);
		CUDA_CHECK(
			cudaMallocHost(
				reinterpret_cast<void**>( &m_idata_host ),
				m_paramsRt.srcImagePitch*(m_paramsRt.srcImageSize.height*1.5)*m_paramsRt.maxStreams
			)
		);
		CUDA_CHECK(
			cudaMallocHost(
				reinterpret_cast<void**>(&m_targetDisplacement2D_h), m_paramsRt.maxTargetsToTrack*sizeof(float2)
			)
		);
	#endif

	CUDA_CHECK(
		cudaMalloc(
			reinterpret_cast<void **>(&m_maskingWindow4Src), m_paramsRt.singleFeatureChannelElementSize * sizeof(float)
		)
	);

	if (m_params.m_debugVerboseLevel_gpumemory > 0)
	{
		size_t free, total;
		CUDA_CHECK(cudaMemGetInfo(&free, &total));

		printf(
			"[DCFTracker][GPU]		[m_maskingWindow4Src] Free mem = %d [MB] among Total = %d [MB] --> memory increase = %d [MB]\n",
			int(free / 1024 / 1024), int(total / 1024 / 1024), int((freeGpuMemory_prev - free) / 1024 / 1024)
		);
		freeGpuMemory_prev = free;
	}

	CUDA_CHECK(
		cudaMalloc(
			reinterpret_cast<void **>(&m_maskingWindow4Model),
			m_paramsRt.singleFeatureChannelElementSize * sizeof(float)
		)
	);

	if (m_params.m_debugVerboseLevel_gpumemory > 0)
	{
		size_t free, total;
		CUDA_CHECK(cudaMemGetInfo(&free, &total));

		printf(
			"[DCFTracker][GPU]		[m_maskingWindow4Model] Free mem = %d [MB] among Total = %d [MB] --> memory increase = %d [MB]\n",
			int(free / 1024 / 1024), int(total / 1024 / 1024), int((freeGpuMemory_prev - free) / 1024 / 1024)
		);
		freeGpuMemory_prev = free;
	}

	CUDA_CHECK(
		cudaMalloc(
			reinterpret_cast<void **>(&m_vpiObjPatches_d),
			m_paramsRt.maxTargetsToTrack * m_vpiObjPatchesNumChannels * m_params.m_featureImgSize * m_params.m_featureImgSize * sizeof(uint8_t)
		)
	);
	#if defined (VIZ_DEBUG)
		CUDA_CHECK(
			cudaMallocHost(
				reinterpret_cast<void **>(&m_vpiObjPatches_h),
				m_paramsRt.maxTargetsToTrack * m_vpiObjPatchesNumChannels * m_params.m_featureImgSize * m_params.m_featureImgSize * sizeof(uint8_t)
			)
		);
	#endif

	if (m_params.m_debugVerboseLevel_gpumemory > 0)
	{
		size_t free, total;
		CUDA_CHECK(cudaMemGetInfo(&free, &total));

		printf(
			"[DCFTracker][GPU] [m_vpiObjPatches_d] Free mem = %d [MB] among Total = %d [MB] --> memory increase = %d [MB]\n",
			int(free / 1024 / 1024), int(total / 1024 / 1024), int((freeGpuMemory_prev - free) / 1024 / 1024)
		);
		freeGpuMemory_prev = free;
	}

	CUDA_CHECK(
		cudaMalloc(
			reinterpret_cast<void**>(&m_corrResponses_d),
			m_paramsRt.maxTargetsToTrack*m_paramsRt.singleFeatureChannelElementSize*sizeof(float)
		)
	);

	if(m_params.m_debugVerboseLevel_gpumemory > 0)
	{
		size_t free, total;
		CUDA_CHECK( cudaMemGetInfo( &free, &total ) );

		printf(
			"[DCFTracker][GPU]		[m_corrResponses_d] Free mem = %d [MB] among Total = %d [MB] --> memory increase = %d [MB]\n",
			int(free/1024/1024), int(total/1024/1024), int((freeGpuMemory_prev-free)/1024/1024)
		);
		freeGpuMemory_prev = free;
	}

	CUDA_CHECK(
		cudaMalloc(reinterpret_cast<void**>(&m_maxCorrResponse_d), m_paramsRt.maxTargetsToTrack*sizeof(float))
	);
	CUDA_CHECK(
		cudaMalloc(reinterpret_cast<void**>(&m_maxCorrResponse_ma_d), m_paramsRt.maxTargetsToTrack*sizeof(float))
	);
	CUDA_CHECK(
		cudaMalloc(reinterpret_cast<void**>(&m_trackingConfidence_d), m_paramsRt.maxTargetsToTrack*sizeof(float))
	);

	CUDA_CHECK(
		cudaMallocHost(reinterpret_cast<void**>(&m_maxCorrResponse_h), m_paramsRt.maxTargetsToTrack*sizeof(float))
	);
	CUDA_CHECK(
		cudaMallocHost(reinterpret_cast<void**>(&m_maxCorrResponse_ma_h), m_paramsRt.maxTargetsToTrack*sizeof(float))
	);
	CUDA_CHECK(
		cudaMallocHost(reinterpret_cast<void**>(&m_trackingConfidence_h), m_paramsRt.maxTargetsToTrack*sizeof(float))
	);

	CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>(&m_trackerAge_d), m_paramsRt.maxTargetsToTrack*sizeof(uint32_t)) );
	CUDA_CHECK(
		cudaMallocHost(reinterpret_cast<void**>(&m_trackerAge_h), m_paramsRt.maxTargetsToTrack*sizeof(uint32_t))
	);

	CUDA_CHECK(
		cudaMalloc(reinterpret_cast<void**>(&m_trackerDoProcessing_d), m_paramsRt.maxTargetsToTrack*sizeof(bool))
	);
	CUDA_CHECK(
		cudaMallocHost(reinterpret_cast<void**>(&m_trackerDoProcessing_h), m_paramsRt.maxTargetsToTrack*sizeof(bool))
	);

	CUDA_CHECK(
		cudaMallocHost(
			reinterpret_cast<void**>(&m_detectorBboxes_h),
			m_paramsRt.maxTargetsToTrack*m_paramsRt.maxDetectorBboxesPerFrame*sizeof(detectorBbox_t)
		)
	);
	CUDA_CHECK(
		cudaMalloc(
			reinterpret_cast<void**>(&m_detectorBboxes_d),
			m_paramsRt.maxTargetsToTrack*m_paramsRt.maxDetectorBboxesPerFrame*sizeof(detectorBbox_t)
		)
	);

	CUDA_CHECK(
		cudaMallocHost(reinterpret_cast<void**>(&m_detectorBboxes_size_h), m_paramsRt.maxTargetsToTrack*sizeof(uint))
	);
	CUDA_CHECK(
		cudaMalloc(reinterpret_cast<void**>(&m_detectorBboxes_size_d), m_paramsRt.maxTargetsToTrack*sizeof(uint))
	);

	CUDA_CHECK(
		cudaMallocHost(
			reinterpret_cast<void**>(&m_confidenceAtDetectedPt_h),
			m_paramsRt.maxTargetsToTrack*m_paramsRt.maxDetectorBboxesPerFrame *sizeof(float)
		)
	);
	CUDA_CHECK(
		cudaMalloc(
			reinterpret_cast<void**>(&m_confidenceAtDetectedPt_d),
			m_paramsRt.maxTargetsToTrack*m_paramsRt.maxDetectorBboxesPerFrame *sizeof(float)
		)
	);

	CUDA_CHECK(
		cudaMallocHost(
			reinterpret_cast<void**>(&m_corrResponseAtDetectedPt_h),
			m_paramsRt.maxTargetsToTrack*m_paramsRt.maxDetectorBboxesPerFrame *sizeof(float)
		)
	);
	CUDA_CHECK(
		cudaMalloc(
			reinterpret_cast<void**>(&m_corrResponseAtDetectedPt_d),
			m_paramsRt.maxTargetsToTrack*m_paramsRt.maxDetectorBboxesPerFrame *sizeof(float)
		)
	);

	if(m_params.m_debugVerboseLevel_gpumemory > 0)
	{
		size_t free, total;
		CUDA_CHECK( cudaMemGetInfo( &free, &total ) );

		printf(
			"[DCFTracker][GPU]		[all other misc. memories] Free mem = %d [MB] among Total = %d [MB] --> memory increase = %d [MB]\n",
			int(free/1024/1024), int(total/1024/1024), int((freeGpuMemory_prev-free)/1024/1024)
		);
		freeGpuMemory_prev = free;
	}

	if(m_params.m_debugVerboseLevel > 1)
	{
		printf(
			"[DCFTracker::allocateCudaMemories()] Allocated CUDA memories. m_paramsRt.maxTargetsToTrack = %d\n ",
			m_paramsRt.maxTargetsToTrack
		);
	}
}

/**
 * @brief Duplicate one host matrix into a batched float buffer and copy it to the device.
 * @param[in] srcMat The source matrix for a single tracker.
 * @param[out] devMem The device buffer that receives the duplicated matrix.
 * @param[in] batchSize The number of copies of the source matrix.
 * @param[in,out] workarea_host The host workspace filled with the duplicated matrix before the device copy.
 */
void DCFTracker::makeBatchAndSendFloatMatToDevice(
	const NvMotDataTypes::RowMajorMatrixXf &srcMat, void *devMem, uint batchSize, float *workarea_host
)
{
	if(devMem == nullptr)
	{
		throw std::runtime_error("[DCFTracker::makeBatchAndSendFloatMatToDevice] Make sure devMem != nullptr");
	}

	if(srcMat.rows()*srcMat.cols() != m_paramsRt.singleFeatureChannelElementSize)
	{
		throw std::runtime_error(
			"[DCFTracker::makeBatchAndSendFloatMatToDevice] Make sure srcMat.rows*srcMat.cols == m_paramsRt.singleFeatureChannelElementSize"
		);
	}

	uint totalElementSize = m_paramsRt.singleFeatureChannelElementSize*batchSize;

	memset(workarea_host, 0, totalElementSize*sizeof(float));

	const float* srcData = reinterpret_cast<const float*>(srcMat.data());
	float* dstData = reinterpret_cast<float*>(workarea_host);

	int index = 0;
	for(uint i=0; i<batchSize; i++, index += m_paramsRt.singleFeatureChannelElementSize)
	{
		std::copy_n( srcData, m_paramsRt.singleFeatureChannelElementSize, dstData + index );
	}

	CUDA_CHECK(
		cudaMemcpyAsync( (float*)devMem, dstData, totalElementSize*sizeof(float), cudaMemcpyHostToDevice, m_cudaStream)
	);

	CUDA_CHECK( cudaStreamSynchronize(m_cudaStream) );
	CUDA_CHECK( cudaPeekAtLastError() );
}

/**
 * @brief Compute the scale that maps a search region to the fixed feature-image size.
 * @param[in] roi Search region whose area determines the scale.
 * @return Factor that scales the search region to the feature image size.
 */
float DCFTracker::getTemplateRescaleRatio(const NvMotDataTypes::Rect2f &roi) const
{
	/// Scale the search region size to be static to fit to the featureImgSizeLevel, so that all the templates have the same dimensions.
	/// Don't make this dynamic unless you have a proper way to address irregular template size in batch processing.
	float templateRescaleRatio = static_cast<float>(
		sqrt(static_cast<float>(pow(m_params.m_featureImgSize,2) / roi.volume()))
	);

	return templateRescaleRatio;
}

/**
 * @brief Map a source-image point into a tracker's correlation-response coordinates.
 * @param[in] roi The source-frame region that defines the relative coordinate system.
 * @param[in] pt_img The point in source-image coordinates.
 * @return The corresponding point in the tracker's correlation-response matrix.
 */
Eigen::Vector2f DCFTracker::convertSrcImgCoordToCorrRespCoord(
	const NvMotDataTypes::Rect2f &roi, const Eigen::Vector2f pt_img
) const
{
	/// Scale the search region size to be static to fit to the featureImgSizeLevel, so that all the templates have the same dimensions.
	/// Don't make this dynamic unless you have a proper way to address irregular template size in batch processing.
	float templateRescaleRatio = getTemplateRescaleRatio(roi);

	/// The center point of the corr resp mat in corr resp mat.
	Eigen::Vector2f centerPtInCorrResp; centerPtInCorrResp << float(m_paramsRt.singleFeatureMatSize.width*0.5f), float(m_paramsRt.singleFeatureMatSize.height*0.5f);

	/// The center point of the search region used for localization.
	const Eigen::Vector2f roiCenter = (roi.min() + roi.max())*0.5f;

	/// The corresponding point in corr resp mat after coordinate system transformation (e.g., translation and scaling).
	Eigen::Vector2f ptInCorrResp = centerPtInCorrResp + (pt_img - roiCenter)*templateRescaleRatio/(float)m_params.m_hogCellSize;

	/// Take into account the wrapping.
	ptInCorrResp(1) = ptInCorrResp(1)+(float)m_paramsRt.singleFeatureMatSize.height*0.5f;
	ptInCorrResp(0) = ptInCorrResp(0)+(float)m_paramsRt.singleFeatureMatSize.width*0.5f;
	ptInCorrResp(1) = ptInCorrResp(1) >= (float)m_paramsRt.singleFeatureMatSize.height ? (ptInCorrResp(1) - (float)m_paramsRt.singleFeatureMatSize.height) : ptInCorrResp(1);
	ptInCorrResp(0) = ptInCorrResp(0) >= (float)m_paramsRt.singleFeatureMatSize.width ? (ptInCorrResp(0) - (float)m_paramsRt.singleFeatureMatSize.width) : ptInCorrResp(0);

	return ptInCorrResp;
}

/**
 * @brief Retrieve each tracker's correlation response at nearby detector bounding boxes.
 * @param[in] detectedObjects The detected objects grouped by video stream.
 */
void DCFTracker::getDetectorBboxCorrelationResponses(
	const std::map<NvMOTStreamId, std::deque<DetectedObjInfo>> &detectedObjects
)
{
	int lastTargetInd = m_targetManager->getLastTargetInd();

	if(lastTargetInd < 0 || detectedObjects.size() == 0)
		return;

	memset(m_detectorBboxes_size_h, 0, (lastTargetInd+1)*sizeof(uint));
	memset(m_detectorBboxes_h, 0, (lastTargetInd+1)*m_paramsRt.maxDetectorBboxesPerFrame*sizeof(detectorBbox_t));

	for(auto it = detectedObjects.begin(); it != detectedObjects.end(); ++it)
	{
		const NvMOTStreamId &videoStreamId = it->first;
		const std::deque<DetectedObjInfo> &detObjs = it->second;
		const NvTrackedObjectList targetList = m_targetManager->getTrackedObjectsInUseList(videoStreamId);

		if(targetList.size() == 0)
			continue;

		/// For each detector bounding box in a video stream.
		for(uint dBboxInd=0; dBboxInd < detObjs.size(); dBboxInd++)
		{
			const DetectedObjInfo &detObj = detObjs[dBboxInd];

			/// For each valid tracker.
			for(auto targetIt = targetList.begin(); targetIt != targetList.end(); ++targetIt)
			{
				std::shared_ptr<NvTrackedObject> target = targetIt->second;
				const uint32_t targetInd = target->getTargetInd();

				/// Get the search region used to perform localization.
				NvMotDataTypes::Rect2f predictedSearchRegion = target->getPredictedSearchRegion();

				/// Check if this detector bbox is within the search region.
				if(predictedSearchRegion.contains(detObj.loc))
				{
					/// Convert the center point of the detector bbox into the tracker corr resp's coordinate system.
					const Eigen::Vector2f relPt = convertSrcImgCoordToCorrRespCoord(predictedSearchRegion, detObj.loc);

					/// Check if the point is within the corr resp matrix.
					if(relPt(0) >=0 && relPt(1) >= 0
							&& relPt(0) < m_paramsRt.singleFeatureMatSize.width && relPt(1) < m_paramsRt.singleFeatureMatSize.height)
					{
						uint ind = targetInd*m_paramsRt.maxDetectorBboxesPerFrame + m_detectorBboxes_size_h[targetInd];

						/// Put it into a storage so that later can be accessed in CUDA kernel.
						m_detectorBboxes_h[ind].x = relPt(0);
						m_detectorBboxes_h[ind].y = relPt(1);
						m_detectorBboxes_h[ind].dBboxInd = dBboxInd;
						m_detectorBboxes_size_h[targetInd]++;
					}
				}
			}
		}
	}

	/// Sync host and device.
	CUDA_CHECK(
		cudaMemcpyAsync(
			m_detectorBboxes_d, m_detectorBboxes_h,
			(lastTargetInd+1)*m_paramsRt.maxDetectorBboxesPerFrame*sizeof(detectorBbox_t),
			cudaMemcpyHostToDevice, m_cudaStream
		)
	);
	CUDA_CHECK(
		cudaMemcpyAsync(
			m_detectorBboxes_size_d, m_detectorBboxes_size_h,
			(lastTargetInd+1)*sizeof(uint), cudaMemcpyHostToDevice, m_cudaStream
		)
	);

	/// Initialize the tracker states.
	CUDA_CHECK(
		cudaMemsetAsync(
			m_confidenceAtDetectedPt_d, 0.f,
			(lastTargetInd+1)*m_paramsRt.maxDetectorBboxesPerFrame*sizeof(float), m_cudaStream
		)
	);
	CUDA_CHECK(
		cudaMemsetAsync(
			m_corrResponseAtDetectedPt_d, 0.f,
			(lastTargetInd+1)*m_paramsRt.maxDetectorBboxesPerFrame*sizeof(float), m_cudaStream
		)
	);

	VPI_CHECK(vpiImageLock(m_vpiCorrResponsesImage, VPI_LOCK_READ));
	/// Given the Fourier-transformed extracted features and the learned filters, compute the correlation response.
	cuDCF::getDetectorBboxCorrelationResponses(
		m_cudaStream, (lastTargetInd+1), m_paramsRt.maxDetectorBboxesPerFrame,
		m_detectorBboxes_d, m_detectorBboxes_size_d, m_paramsRt.detCorrRespSearchWindowSize,
		m_corrResponses_d, m_maxCorrResponse_ma_d, m_paramsRt.singleFeatureChannelElementSize,
		m_paramsRt.singleFeatureMatSize.width, m_confidenceAtDetectedPt_d, m_corrResponseAtDetectedPt_d
	);
	VPI_CHECK(vpiImageUnlock(m_vpiCorrResponsesImage));
	/// Sync host and device.
	CUDA_CHECK(
		cudaMemcpyAsync(
			m_confidenceAtDetectedPt_h, m_confidenceAtDetectedPt_d,
			(lastTargetInd+1)*m_paramsRt.maxDetectorBboxesPerFrame*sizeof(float), cudaMemcpyDeviceToHost, m_cudaStream
		)
	);
	CUDA_CHECK(
		cudaMemcpyAsync(
			m_corrResponseAtDetectedPt_h, m_corrResponseAtDetectedPt_d,
			(lastTargetInd+1)*m_paramsRt.maxDetectorBboxesPerFrame*sizeof(float), cudaMemcpyDeviceToHost, m_cudaStream
		)
	);
	CUDA_CHECK( cudaStreamSynchronize(m_cudaStream) );
	CUDA_CHECK( cudaPeekAtLastError() );

	if(m_params.m_debugVerboseLevel > 4)
	{
		cout << "[DCFTracker::getDetectorBboxCorrelationResponses]" << endl;
		for(uint targetInd = 0; targetInd <= (uint)lastTargetInd; targetInd++)
		{
			for(uint detInd = 0; detInd < m_paramsRt.maxDetectorBboxesPerFrame; detInd++)
			{
				const float confidenceAtDetPt = m_confidenceAtDetectedPt_h[targetInd*m_paramsRt.maxDetectorBboxesPerFrame + detInd];
				const float corrResponseAtDetPt = m_corrResponseAtDetectedPt_h[targetInd*m_paramsRt.maxDetectorBboxesPerFrame + detInd];

				if(confidenceAtDetPt > 0)
				{
					cout << "- targetInd[" << targetInd << "] --> detInd[" << detInd << "] conf score = " << confidenceAtDetPt << endl;
					cout << "- targetInd[" << targetInd << "] --> detInd[" << detInd << "] corr Response = " << corrResponseAtDetPt << endl;
				}
			}
		}
	}
}

/** @brief Given the cropped and scaled object patches and the learned filters, compute the correlation response. */
void DCFTracker::cudaCalculateResponse(void)
{
	int lastTargetInd = m_targetManager->getLastTargetInd();
	if (lastTargetInd < 0)
		return;
	/// Localize each tracked object using DCF.
	/// The scaled and cropped objects are provided as an image in m_vpiObjPatches.
	/// Information about each tracked object is present in m_vpiDCFBboxesArray and is modified after localization in this call.
	/// Correlation response of each object being tracked is provided as output in m_vpiCorrResponsesImage.
	VPI_CHECK(
		vpiSubmitDCFTrackerLocalizeBatch(
			m_vpiStream, m_params.m_vpiBackend4DcfTracker, m_vpiDcfPayload, m_enabledSequences,
			m_paramsRt.maxStreams, m_vpiMaskingWindow4SrcImage, m_vpiObjPatches, m_vpiDCFBboxesArray,
			m_vpiDCFBboxesLocalizedArray, m_vpiCorrResponsesImage, m_vpiMaxCorrResponseArray, &m_vpiRuntimeParams
		)
	);
	/// Wait until VPI processing is done.
	VPI_CHECK(vpiStreamSync(m_vpiStream));

	/// Lock all the arrays/images since the data buffers will be accessed outside VPI.
	VPI_CHECK(vpiArrayLock(m_vpiMaxCorrResponseArray, VPI_LOCK_READ));

	/// Compute the moving average of the max corr resp.
	cuDCF::calculateTrackingConfidence(
		m_cudaStream, (lastTargetInd+1), m_maxCorrResponse_ma_d, m_maxCorrResponse_d, m_trackingConfidence_d
	);

	/// Sync host and device.
	CUDA_CHECK(
		cudaMemcpyAsync(
			(float*)m_trackingConfidence_h, m_trackingConfidence_d,
			(lastTargetInd+1)*sizeof(float), cudaMemcpyDeviceToHost, m_cudaStream
		)
	);

	if(m_params.m_debugVerboseLevel > 2)
	{
		CUDA_CHECK(
			cudaMemcpyAsync(
				(float*)m_maxCorrResponse_h, m_maxCorrResponse_d,
				(lastTargetInd+1)*sizeof(float), cudaMemcpyDeviceToHost, m_cudaStream
			)
		);
		CUDA_CHECK(
			cudaMemcpyAsync(
				(float*)m_maxCorrResponse_ma_h, m_maxCorrResponse_ma_d,
				(lastTargetInd+1)*sizeof(float), cudaMemcpyDeviceToHost, m_cudaStream
			)
		);

		CUDA_CHECK( cudaStreamSynchronize(m_cudaStream) );
		CUDA_CHECK( cudaPeekAtLastError() );

		printf("\n");
		for(int i=0; i<=lastTargetInd; i++)
		{
			int targetInd = i;
			if(!m_trackerDoProcessing_h[targetInd])
				continue;

			cout << "@@[DCFTracker] targetInd[" << targetInd << "]: (maxCorrResponse_ma, maxCorrResponse, trackingConfidence) = ("
						<< m_maxCorrResponse_ma_h[targetInd] << ", " << m_maxCorrResponse_h[targetInd]
						<< ", " << m_trackingConfidence_h[targetInd] << ")" << endl;
		}
	}
	VPI_CHECK(vpiArrayUnlock(m_vpiMaxCorrResponseArray));
}

/**
 * @brief Update target states from their correlation responses.
 * @param[in] inputMeta A batch of input frames.
 */
void DCFTracker::updateLocalizationResults(const std::map<NvMOTStreamId, NvMOTFrame*> &inputMeta)
{
	/// Lock the VPI arrays for reading bboxes.
	VPI_CHECK(vpiArrayLock(m_vpiDCFBboxesArray, VPI_LOCK_READ));
	VPI_CHECK(vpiArrayLock(m_vpiDCFBboxesLocalizedArray, VPI_LOCK_READ));
	/// Run detection and localization only for video streams that provided a source frame in this batch.
	for(auto streamIt = inputMeta.begin(); streamIt != inputMeta.end(); ++streamIt)
	{
		const NvMOTStreamId &videoStreamId = streamIt->first;
		const NvMOTFrame *srcMotFrame = streamIt->second;
		const uint32_t frameNum = srcMotFrame->frameNum;
		const NvTrackedObjectList targetList = m_targetManager->getTrackedObjectsInUseList(videoStreamId);

		for(auto targetIt = targetList.begin(); targetIt != targetList.end(); ++targetIt)
		{
			std::shared_ptr<NvTrackedObject> target = targetIt->second;
			const uint32_t targetInd = target->getTargetInd();
			/// For debugging.
			const NvMotDataTypes::Rect2f original_bbox = target->getTargetBbox();

			/// Get the search region used to perform localization.
			const NvMotDataTypes::Rect2f predictedSearchRegion = target->getPredictedSearchRegion();

			/// Bounding box displacement is the difference between the input and output bounding boxes
			/// of VPI call vpiSubmitDCFTrackerLocalizeBatch().
			VPIAxisAlignedBoundingBoxF32 vpiOriginalBbox = m_vpiDCFBboxes[targetInd].bbox;
			VPIAxisAlignedBoundingBoxF32 vpiUpdatedBbox = m_vpiDCFBboxesLocalized[targetInd].bbox;
			if(m_params.m_debugVerboseLevel > 2)
			{
				std::cout << "vpiOriginalBbox : " << vpiOriginalBbox.left << "X" << vpiOriginalBbox.top << "=>" <<\
				vpiOriginalBbox.width << "X" << vpiOriginalBbox.height <<std::endl ;
				std::cout << "vpiUpdatedBbox : " << vpiUpdatedBbox.left << "X" << vpiUpdatedBbox.top << "=>" <<\
				vpiUpdatedBbox.width << "X" << vpiUpdatedBbox.height <<std::endl ;
			}

			if(IS_NAN(vpiOriginalBbox.width) || IS_NAN(vpiUpdatedBbox.width) ||
				IS_NAN(vpiOriginalBbox.height) || IS_NAN(vpiUpdatedBbox.height))
			{
				/// Log a warning when localization returns a non-finite box.
				CLOG(
					std::string("!!![WARNING] Width and height of input and output bounding boxes from " "vpiSubmitDCFTrackerLocalizeBatch() are NAN!!").c_str()
				);
			}

			Eigen::Vector2f displacementInImg(
				(vpiUpdatedBbox.left-vpiOriginalBbox.left), (vpiUpdatedBbox.top-vpiOriginalBbox.top)
			);

			#if defined(VIZ_DEBUG)
				float templateRescaleRatio = getTemplateRescaleRatio(predictedSearchRegion);
				m_targetDisplacement2D_h[targetInd].x = displacementInImg(0)*templateRescaleRatio/m_params.m_hogCellSize;
				m_targetDisplacement2D_h[targetInd].y = displacementInImg(1)*templateRescaleRatio/m_params.m_hogCellSize;
			#endif

			/// Set the new location from the instant peak correlation location response value.
			target->setLocalizationResults(
				frameNum, m_trackingConfidence_h[targetInd],
				CV_FUNCS::displaceBbox( target->getTargetBbox(), displacementInImg(0), displacementInImg(1))
			);
			/// Set the tracking confidence score at nearby detected pts.
			target->setConfidenceCorrRespAtDetectedPts(
				&m_confidenceAtDetectedPt_h[targetInd*m_paramsRt.maxDetectorBboxesPerFrame],
				m_paramsRt.maxDetectorBboxesPerFrame
			);
			target->setCorrRespAtDetectedPts(
				&m_corrResponseAtDetectedPt_h[targetInd*m_paramsRt.maxDetectorBboxesPerFrame],
				m_paramsRt.maxDetectorBboxesPerFrame
			);

			if(m_params.m_debugVerboseLevel > 2)
			{
				const NvMotDataTypes::Rect2f localizationBbox = target->getTargetBbox();
				cout << "[DCFTracker] Stream[" << (videoStreamId>>32) << "] Frame[" << frameNum << "] TargetInd[" << targetInd
						<< "] ID[" << target->getTargetID()
						<< "] (maxCorrResp, dX, dY) = (" << m_maxCorrResponse_h[targetInd] << ", "
						<< displacementInImg(0) << ", "
						<< displacementInImg(1) << ")" << endl;
				std::cout << "Original : [" << original_bbox.sizes()(0) << " x " << original_bbox.sizes()(1) << " from (" << original_bbox.min()(0) << ", " << original_bbox.min()(1) << ")]"  << std::endl;
				std::cout << "Displaced : [" << localizationBbox.sizes()(0) << " x " << localizationBbox.sizes()(1) << " from (" << localizationBbox.min()(0) << ", " << localizationBbox.min()(1) << ")]"  << std::endl;
			}
		}
	}
	/// Unlock the VPI arrays.
	VPI_CHECK(vpiArrayUnlock(m_vpiDCFBboxesArray));
	VPI_CHECK(vpiArrayUnlock(m_vpiDCFBboxesLocalizedArray));
}

/**
 * @brief Localize tracked targets and evaluate nearby detections across an input batch.
 * @param[in] inputMeta A batch of input frames.
 * @param[in] detectedObjects The list of newly-detected objects.
 */
void DCFTracker::localize(
	const std::map<NvMOTStreamId, NvMOTFrame *> inputMeta,
	const std::map<NvMOTStreamId, std::deque<DetectedObjInfo>> &detectedObjects
)
{
	/// Set the input frame sequence for VPI DCF tracker.
	setVpiInputFrameSequence(inputMeta);
	#if defined(VIZ_DEBUG)
		uploadDeviceData2HostMemories(inputMeta);
	#endif

	/// Compute the correlation response.
	performCropAndScale(inputMeta, true);
	cudaCalculateResponse();

	/// Retrieve the correlation response of the detector bboxes from each tracker.
	getDetectorBboxCorrelationResponses(detectedObjects);

	/// Propagate the localization results.
	updateLocalizationResults(inputMeta);

	#if defined(VIZ_DEBUG)
		if(m_vizOutput != nullptr && isInitialized() && m_targetManager->getLastTargetInd() >= 0)
		{
			if(m_params.m_vizWriteSrcImgPatchesAndCorrResp)
			{
				m_vizOutput->writeSrcImagePatchesWithCorrResp(VIZ_OUTPUT_TYPES::TRACKER_STATES_VIZ);
			}
		}
	#endif
}

/**
 * @brief Retrieve CUDA memory pointers for all planes of an NvBufSurface.
 * @param[in] baseSurface The surface parameters whose plane pointers are retrieved.
 */
inline void DCFTracker::getCudaMemPointers(NvBufSurfaceParams *baseSurface)
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
				if(m_graphicsResourceAddrMap.size() > m_paramsRt.m_maxBufSurfAddrSize)
				{
					if (m_params.m_debugVerboseLevel > 2)
					{
						printf(
							"[DCFTracker::getCudaMemPointers] m_graphicsResourceAddrMap size exceeds limit: %lu / %d\n",
							m_graphicsResourceAddrMap.size(), m_paramsRt.m_maxBufSurfAddrSize
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
 * @brief Check the input and find the detection interval.
 * @param[in] inputMeta A batch of input frames.
 */
void DCFTracker::checkInputOutput(const std::map<NvMOTStreamId, NvMOTFrame*> &inputMeta)
{
	m_paramsRt.skipBatchUpdate = true;

	for (auto it = inputMeta.begin(); it != inputMeta.end(); ++it)
	{
		const NvMOTStreamId &videoStreamId = it->first;
		const NvMOTFrame *srcMotFrame = it->second;
		const uint32_t seq_index = srcMotFrame->seq_index;

		if(m_paramsRt.detectorInputInterval < 0
				&& m_paramsRt.isDetectionDonePerStream.find(videoStreamId) != m_paramsRt.isDetectionDonePerStream.end()
				&& srcMotFrame->objectsIn.detectionDone == true)
		{
			m_paramsRt.detectorInputInterval = m_paramsRt.isDetectionDonePerStream[videoStreamId];

			{
				/// Scaling factor for learning rates against the inference interval.
				/// 3.f is used here to make the value of m_filterLr grows naturally.
				const float lrAdaptRate = 3.f;
				const float upperLimit = 0.5f;
				float alpha = m_paramsRt.detectorInputInterval/(float)(lrAdaptRate+m_paramsRt.detectorInputInterval);

				/// When detector interval > 0, in case of match, increase the learning rate (LR) by taking an intermediate point between the base LR and 1.f.
				m_params.m_filterLr = (1.f-alpha)*m_params.m_filterLr + alpha*upperLimit;
				m_params.m_filterChannelWeightsLr = (1.f-alpha)*m_params.m_filterChannelWeightsLr + alpha*upperLimit;
				m_params.m_peakCorrLr = (1.f-alpha)*m_params.m_peakCorrLr + alpha*upperLimit;
			}

			if(m_params.m_debugVerboseLevel > 1)
			{
				cout << "[DCFTracker::checkInputOutput()] Found detection interval = " << m_paramsRt.detectorInputInterval << "]" << endl;
				cout << "[DCFTracker::checkInputOutput()] (filterLr, channelWeightsLr, peakCorrLr) = ("
						<< m_params.m_filterLr << ", " << m_params.m_filterChannelWeightsLr << ", " << m_params.m_peakCorrLr << ")" << endl;
			}
		}

		if(srcMotFrame->objectsIn.detectionDone == true)
		{
			m_paramsRt.isDetectionDonePerStream[videoStreamId] = 0;
			m_paramsRt.skipBatchUpdate = false;
		}
		else
		{
			m_paramsRt.isDetectionDonePerStream[videoStreamId]++;
		}

		/// Only enable to update streams with detection input.
		m_enabledSequences[seq_index] = int(srcMotFrame->objectsIn.detectionDone);
	}
}

/**
 * @brief Populate the VPI input frame sequence from MOT frames.
 * @param[in] inputMeta A batch of input frames.
 */
void DCFTracker::setVpiInputFrameSequence(const std::map<NvMOTStreamId, NvMOTFrame*> &inputMeta)
{
	/// Following array will be populated in this routine.
	memset(m_enabledSequences, 0, m_paramsRt.maxStreams * sizeof(int32_t));

	for (auto it = inputMeta.begin(); it != inputMeta.end(); ++it)
	{
		const NvMOTFrame *srcMotFrame = it->second;

		/// Create VPIImage for the frame in srcMotFrame.
		{
			NvBufSurfaceParams *surfparams = srcMotFrame->bufferList[0];
			uint32_t seq_index = srcMotFrame->seq_index;
			if ((m_paramsRt.srcImageSize.width < (int)srcMotFrame->bufferList[0]->width) ||
				(m_paramsRt.srcImageSize.height < (int)srcMotFrame->bufferList[0]->height) ||
				(m_paramsRt.srcImagePitch < srcMotFrame->bufferList[0]->pitch) ||
				(m_paramsRt.colorFormat != srcMotFrame->bufferList[0]->colorFormat))
			{
				std::string msg = "Image specification in configuration paarameters do not match with actual image characteristics! \n  Configured specification = (" +
				std::to_string(m_paramsRt.srcImageSize.width) + "," + std::to_string(m_paramsRt.srcImageSize.height) + "," +
				std::to_string(m_paramsRt.srcImagePitch) + "," + std::to_string(m_paramsRt.colorFormat) + ") \n"
				"Actual image specifications = (" +
				std::to_string(srcMotFrame->bufferList[0]->width) + "," + std::to_string(srcMotFrame->bufferList[0]->height) + "," +
				std::to_string(srcMotFrame->bufferList[0]->pitch) + "," + std::to_string(srcMotFrame->bufferList[0]->colorFormat) + ")";
				throw std::runtime_error(msg);
			}
			if(seq_index >= m_paramsRt.maxStreams)
			{
				throw std::runtime_error("[FAILED] Frame sequence index should be in the range 0 to (batch size - 1)");
			}

			/// Get the cuda memory pointers.
			getCudaMemPointers(surfparams);

			if(m_isFrameCreated[seq_index] == false)
			{
				/// Populate VPIImage fields using the entries in NvBufSurfaceParams structure.
				if (m_paramsRt.colorFormat == NvBufSurfaceColorFormat::NVBUF_COLOR_FORMAT_NV12)
				{
					m_vpiInputFrameDataList[seq_index] = {};
					m_vpiInputFrameDataList[seq_index].bufferType = VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR;

					/// For NV12.
					m_vpiInputFrameDataList[seq_index].buffer.pitch.format = VPI_IMAGE_FORMAT_NV12_ER;
					m_vpiInputFrameDataList[seq_index].buffer.pitch.numPlanes = 2;

					/// For Y channel.
					/// 1-ch Y.
					m_vpiInputFrameDataList[seq_index].buffer.pitch.planes[0].pixelType = VPI_PIXEL_TYPE_U8;
					/// Use m_srcImgPatchPtr_d as a raw pointer to pitch-linear CUDA memory.
					m_vpiInputFrameDataList[seq_index].buffer.pitch.planes[0].pBase = m_surfCudaMemPtr[0];
					m_vpiInputFrameDataList[seq_index].buffer.pitch.planes[0].offsetBytes = 0;
					m_vpiInputFrameDataList[seq_index].buffer.pitch.planes[0].width = surfparams->planeParams.width[0];
					m_vpiInputFrameDataList[seq_index].buffer.pitch.planes[0].height = surfparams->planeParams.height[0];
					m_vpiInputFrameDataList[seq_index].buffer.pitch.planes[0].pitchBytes = surfparams->planeParams.pitch[0];

					/// For UV channel.
					/// 2-ch due to interleaved UV channel.
					m_vpiInputFrameDataList[seq_index].buffer.pitch.planes[1].pixelType = VPI_PIXEL_TYPE_2U8;
					m_vpiInputFrameDataList[seq_index].buffer.pitch.planes[1].pBase = m_surfCudaMemPtr[1];
					m_vpiInputFrameDataList[seq_index].buffer.pitch.planes[1].offsetBytes = 0;
					m_vpiInputFrameDataList[seq_index].buffer.pitch.planes[1].width = surfparams->planeParams.width[1];
					m_vpiInputFrameDataList[seq_index].buffer.pitch.planes[1].height = surfparams->planeParams.height[1];
					m_vpiInputFrameDataList[seq_index].buffer.pitch.planes[1].pitchBytes = surfparams->planeParams.pitch[1];
				}
				else if (m_paramsRt.colorFormat == NvBufSurfaceColorFormat::NVBUF_COLOR_FORMAT_RGBA)
				{
					m_vpiInputFrameDataList[seq_index] = {};
					m_vpiInputFrameDataList[seq_index].bufferType = VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR;

					/// For RGBA.
					m_vpiInputFrameDataList[seq_index].buffer.pitch.format = VPI_IMAGE_FORMAT_RGBA8;
					m_vpiInputFrameDataList[seq_index].buffer.pitch.numPlanes = 1;
					/// 4-ch because RGBA.
					m_vpiInputFrameDataList[seq_index].buffer.pitch.planes[0].pixelType = VPI_PIXEL_TYPE_4U8;
					/// Use m_srcImgPatchPtr_d as a raw pointer to pitch-linear CUDA memory.
					m_vpiInputFrameDataList[seq_index].buffer.pitch.planes[0].pBase = m_surfCudaMemPtr[0];
					m_vpiInputFrameDataList[seq_index].buffer.pitch.planes[0].offsetBytes = 0;
					m_vpiInputFrameDataList[seq_index].buffer.pitch.planes[0].width = surfparams->width;
					/// Consider the stacked image patches as a single image.
					m_vpiInputFrameDataList[seq_index].buffer.pitch.planes[0].height = surfparams->height;
					m_vpiInputFrameDataList[seq_index].buffer.pitch.planes[0].pitchBytes = surfparams->pitch;
					#if 0
						{
						unsigned char *temp_data = (unsigned char *)malloc(surfparams->height*surfparams->pitch);
						CUDA_CHECK(
							cudaMemcpyAsync(
								temp_data, m_surfCudaMemPtr[0], surfparams->height*surfparams->pitch,
								cudaMemcpyDeviceToHost, m_cudaStream
							)
						);
						CUDA_CHECK( cudaStreamSynchronize(m_cudaStream) );
						CUDA_CHECK( cudaPeekAtLastError() );
						cv::Mat tempImg = cv::Mat(
							surfparams->height, surfparams->width, CV_8UC4, reinterpret_cast<uint8_t*>(temp_data)
						);
						cv::cvtColor(tempImg, tempImg, CV_RGBA2BGR);
						cv::imwrite("inputImg.png", tempImg);
						}
					#endif
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
				if (m_paramsRt.colorFormat == NvBufSurfaceColorFormat::NVBUF_COLOR_FORMAT_NV12)
				{
					/// Use m_srcImgPatchPtr_d as a raw pointer to pitch-linear CUDA memory.
					m_vpiInputFrameDataList[seq_index].buffer.pitch.planes[0].pBase = m_surfCudaMemPtr[0];
					m_vpiInputFrameDataList[seq_index].buffer.pitch.planes[0].offsetBytes = 0;
					m_vpiInputFrameDataList[seq_index].buffer.pitch.planes[1].pBase = m_surfCudaMemPtr[1];
					m_vpiInputFrameDataList[seq_index].buffer.pitch.planes[1].offsetBytes = 0;
				}
				else if (m_paramsRt.colorFormat == NvBufSurfaceColorFormat::NVBUF_COLOR_FORMAT_RGBA)
				{
					/// Use m_srcImgPatchPtr_d as a raw pointer to pitch-linear CUDA memory.
					m_vpiInputFrameDataList[seq_index].buffer.pitch.planes[0].pBase = m_surfCudaMemPtr[0];
					m_vpiInputFrameDataList[seq_index].buffer.pitch.planes[0].offsetBytes = 0;
				}
				else
				{
					throw std::runtime_error("[FAILED] Invalid color format");
				}
				VPI_CHECK(vpiImageSetWrapper(m_vpiInputFrameSequence[seq_index], &m_vpiInputFrameDataList[seq_index]));
			}
			/// Indicate to VPI library that there is a valid frame at "seq_index".
			m_enabledSequences[seq_index] = 1;
		}
	}
}

/**
 * @brief Update each target's correlation-filter model from an input batch.
 * @param[in] inputMeta A batch of input frames.
 */
void DCFTracker::update(const std::map<NvMOTStreamId, NvMOTFrame *> inputMeta)
{
	/// Check the input.
	checkInputOutput(inputMeta);

	/// Create a new optimal filter and update the moving-avg filter.
	/// Will skip the full batch if no stream needs update.
	if (!m_paramsRt.skipBatchUpdate)
	{
		performCropAndScale(inputMeta, false);
		cudaFilterUpdate();
	}

	#if defined(VIZ_DEBUG)
		if(m_vizOutput != nullptr && isInitialized() && m_targetManager->getLastTargetInd() >= 0)
		{
			if(m_params.m_vizWriteAllTrajectoriesOnEveryFrame)
			{
				m_vizOutput->writeAllTrajectoriesOnEveryFrame(VIZ_OUTPUT_TYPES::TRACKER_STATES_VIZ);
			}

			if(m_params.m_vizWriteEachCompletedTrajectory)
			{
				m_vizOutput->writeEachCompletedTrajectory(VIZ_OUTPUT_TYPES::TRACKER_STATES_VIZ);
			}
			if (m_params.m_vizWriteBirdsEyeView)
			{
				m_vizOutput->writeBirdsEyeView(VIZ_OUTPUT_TYPES::TRACKER_STATES_VIZ);
			}
		}
	#endif
}

/**
 * @brief Crop and scale object regions for localization or model update.
 * @param[in] inputMeta A batch of input frames.
 * @param[in] isForLocalization Whether the operation is for localization rather than model update.
 */
void DCFTracker::performCropAndScale(
	const std::map<NvMOTStreamId, NvMOTFrame *> &inputMeta, const bool isForLocalization
)
{
	int lastTargetInd = m_targetManager->getLastTargetInd();
	if (lastTargetInd < 0)
		return;

	/// Mark the tracker so it is not processed for this batch.
	/// And, Construct ROIs and offsets to the dst device pointers.
	/// Construct regions from the video streams present in the current batch.
	{
		/// {videoStreamID, a vector of {targetInd, bbox} }.
		m_featureRoisInBatch.clear();
		memset(m_trackerDoProcessing_h, false, (lastTargetInd + 1) * sizeof(bool));
		memset(m_trackerAge_h, 0, (lastTargetInd + 1) * sizeof(uint32_t));
		/// Lock all the arrays/images since the data buffers will be accessed outside VPI.
		VPI_CHECK(vpiArrayLock(m_vpiDCFBboxesArray, VPI_LOCK_WRITE));
		VPI_CHECK(vpiArrayLock(m_vpiDCFBboxesLocalizedArray, VPI_LOCK_WRITE));
		/// NOTE: memset is used here instead of std::fill since the vectors hold VPIDCFTrackedBoundingBox values.
		/// Populate VPIDCFTrackedBoundingBox values for VPI processing.
		memset(&m_vpiDCFBboxes[0], 0, (lastTargetInd + 1) * sizeof(m_vpiDCFBboxes[0]));
		memset(&m_vpiDCFBboxesLocalized[0], 0, (lastTargetInd + 1) * sizeof(m_vpiDCFBboxesLocalized[0]));
		#if defined(VIZ_DEBUG)
			m_targetNumInd2RoisInBatchMapper.clear();
		#endif

		/// Update target states with no detector inputs.
		for (auto streamIt = inputMeta.begin(); streamIt != inputMeta.end(); ++streamIt)
		{
			const NvMOTStreamId &videoStreamId = streamIt->first;
			const NvMOTFrame *srcMotFrame = streamIt->second;
			const NvTrackedObjectList targetList = m_targetManager->getTrackedObjectsInUseList(videoStreamId);

			/// If detection is not done for the current frame, there is no need to update DCF, so
			/// leave feature regions, VPI bounding boxes, processing flags, and tracker ages unfilled.
			/// In this way, all targets' states are VPI_TRACKING_STATE_LOST in m_vpiDCFBboxes. vpiSubmitCropScalerBatch automatically skips them.
			if ( (!srcMotFrame->objectsIn.detectionDone) && (!isForLocalization) )
			{
				continue;
			}

			for (auto targetIt = targetList.begin(); targetIt != targetList.end(); ++targetIt)
			{
				std::shared_ptr<NvTrackedObject> target = targetIt->second;
				const uint32_t targetInd = target->getTargetInd();
				NvMotDataTypes::Rect2f search_region = {};

				/// Quasi-active targets: use current bbox as ROI (search region not set), skip filter update.
				const bool quasiActive = (target->getTrackerState() == TRACKER_STATE::QUASIACTIVE);
				if (quasiActive)
				{
					search_region = target->getTargetBbox();
					m_featureRoisInBatch[videoStreamId].push_back(std::make_pair(targetInd, search_region));
					if (!isForLocalization)
					{
						m_trackerDoProcessing_h[targetInd] = false;
						m_trackerAge_h[targetInd] = target->getAge();
					}
				}
				else if (isForLocalization)
				{
					search_region = target->getPredictedSearchRegion();
					/// Get the ROI from which the visual features are to be extracted.
					m_featureRoisInBatch[videoStreamId].push_back(std::make_pair(targetInd, search_region));
				}
				else
				{
					search_region = target->getSearchRegion();
					/// Get the ROI from which the visual features are to be extracted.
					m_featureRoisInBatch[videoStreamId].push_back(std::make_pair(targetInd, search_region));

					/// Perform filter update only when there was a successful match with a detector.
					m_trackerDoProcessing_h[targetInd] = (target->getTargetInfo().matchedDetObjInfo != nullptr);
					m_trackerAge_h[targetInd] = target->getAge();
				}
				/// Now that the search region is defined, pass it to VPI library for crop and scale.
				const Eigen::Vector2f min = search_region.min();
				const Eigen::Vector2f max = search_region.max();
				m_vpiDCFBboxes[targetInd].bbox.left = min(0);
				m_vpiDCFBboxes[targetInd].bbox.top = min(1);
				m_vpiDCFBboxes[targetInd].bbox.width = max(0) - min(0);
				m_vpiDCFBboxes[targetInd].bbox.height = max(1) - min(1);
				m_vpiDCFBboxes[targetInd].seqIndex = srcMotFrame->seq_index;
				if (quasiActive)
				{
					m_vpiDCFBboxes[targetInd].state = VPI_TRACKING_STATE_LOST;
				}
				else
				{
					const uint32_t age = target->getAge();
					const bool isObjectTracked = (target->getTargetInfo().matchedDetObjInfo != nullptr);
					if(age <= 0)
					{
						throw std::runtime_error("Age of an object is <= 0");
					}
					if (!isForLocalization && (age == 1))
					{
						m_vpiDCFBboxes[targetInd].state = VPI_TRACKING_STATE_NEW;
						/// New object : Set learning rates as 1.0.
						m_vpiDCFBboxes[targetInd].filterLR = 1.0f;
						m_vpiDCFBboxes[targetInd].filterChannelWeightsLR = 1.0f;
					}
					else
					{
						/// The object is marked as tracked only when there was a successful match with a detector.
						/// Else it is marked as shadow tracked.
						if (isObjectTracked)
						{
							m_vpiDCFBboxes[targetInd].state = VPI_TRACKING_STATE_TRACKED;
						}
						else
						{
							m_vpiDCFBboxes[targetInd].state = VPI_TRACKING_STATE_SHADOW_TRACKED;
						}

						/// Taper the initially aggressive learning rate as the target ages.
						const float alpha = 1.f / age;
						m_vpiDCFBboxes[targetInd].filterLR = (1.f - alpha) * m_params.m_filterLr + alpha * LEARN_RATE_UPPER_LIMIT;
						m_vpiDCFBboxes[targetInd].filterChannelWeightsLR = (1.f - alpha) * m_params.m_filterChannelWeightsLr + alpha * LEARN_RATE_UPPER_LIMIT;
					}
				}
				#if defined(VIZ_DEBUG)
					/// Map the target index to the corresponding video stream ID and the bbox ind.
					/// {targetInd, { videoStreamID, bboxInd Per Stream} }.
					m_targetNumInd2RoisInBatchMapper[targetInd] = std::make_pair(videoStreamId, targetInd);
				#endif
			}
		}

		if (!isForLocalization)
		{
			/// Sync host and device.
			CUDA_CHECK(
				cudaMemcpyAsync(
					(uint32_t *)m_trackerAge_d, m_trackerAge_h,
					(lastTargetInd + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice, m_cudaStream
				)
			);
			CUDA_CHECK(
				cudaMemcpyAsync(
					m_trackerDoProcessing_d, m_trackerDoProcessing_h,
					(lastTargetInd + 1) * sizeof(bool), cudaMemcpyHostToDevice, m_cudaStream
				)
			);
		}
	}
	/// Set the valid number of entries in the tracked objects array.
	/// This gets used for dynamic compute adjustments by VPI library.
	m_vpiBboxesArraySize = lastTargetInd + 1;

	/// Unlock all the arrays/images that are needed by VPI task.
	VPI_CHECK(vpiArrayUnlock(m_vpiDCFBboxesArray));
	VPI_CHECK(vpiArrayUnlock(m_vpiDCFBboxesLocalizedArray));

	/// Crop tracked objects from the input frames and rescale them all to the same dimensions.
	/// Input frames provided as m_vpiInputFrameSequence and tracked objects information is in m_vpiDCFBboxesArray.
	/// Cropped and scaled object patches are stacked on top of each other and provided as output in m_vpiObjPatches.
	VPI_CHECK(
		vpiSubmitCropScalerBatch(
			m_vpiStream, m_params.m_vpiBackend4CropnScale, m_vpiCropPayload,
			m_vpiInputFrameSequence, m_paramsRt.maxStreams, m_vpiDCFBboxesArray,
			m_params.m_featureImgSize, m_params.m_featureImgSize, m_vpiObjPatches
		)
	);

	#if 0
		std::map<NvMOTStreamId, std::map<uint, cv::Mat > > subImgs = getImagePatches2Host_OCV_batch();
		for (auto it = subImgs.begin(); it != subImgs.end(); ++it)
		{
			NvMOTStreamId streamId = it->first;
			/// Use m_featureRoisInBatch only for visualization debugging.
			if (m_featureRoisInBatch.find(streamId) == m_featureRoisInBatch.end())
				continue;
			const std::vector<std::pair<uint, NvMotDataTypes::Rect2f>> &roisPerStream = m_featureRoisInBatch.at(
				streamId
			);
			int temp_cnt = 0;
			for (auto jt = it->second.begin(); jt != it->second.end(); ++jt)
			{
				uint targetInd = jt->first;
				NvMotDataTypes::Rect2f roi = std::get<1>(roisPerStream[temp_cnt]);
				if (targetInd != std::get<0>(roisPerStream[temp_cnt]))
				{
					std::cout << targetInd << "v/s" << std::get<0>(roisPerStream[temp_cnt]) << endl;
					exit(1);
				}
				temp_cnt++;
				std::cout << "[" << roi.sizes()(0) << " x " << roi.sizes()(1) << " from (" << roi.min()(0) << ", " << roi.min()(1) << ")]" << std::endl;

				cv::Mat subImg = jt->second;
				cvtColor(subImg, subImg, cv::COLOR_RGBA2BGR);

				std::string outDir = "VPI_DCF_patches/";
				#if 1
					std::string label = "s" + std::to_string((int)roi.min()(0)) +
										"X" + std::to_string((int)roi.min()(1)) +
										"_" + std::to_string((int)roi.sizes()(0)) +
										"_" + std::to_string((int)roi.sizes()(1)) + ".png";
				#endif
				cv::imwrite(outDir + label, subImg);
			}
		}
	#endif
	return;
}

#if defined(VIZ_DEBUG)
/** @brief Get patches to host for visualization debugging. */
std::map<NvMOTStreamId, std::map<uint, cv::Mat > > DCFTracker::getImagePatches2Host_OCV_batch()
{
	VPI_CHECK(vpiStreamSync(m_vpiStream));
	CUDA_CHECK( cudaStreamSynchronize(m_cudaStream) );
	CUDA_CHECK( cudaPeekAtLastError() );
	cv::Mat originalImg, outImg;
	/// {videoStreamID, {targetInd, cropped bbox images} }.
	std::map<NvMOTStreamId, std::map<uint, cv::Mat > >  outputRoiImages;
	CUDA_CHECK(
		cudaMemcpyAsync(
			m_vpiObjPatches_h, m_vpiObjPatches_d,
			m_params.m_featureImgSize*m_vpiBboxesArraySize*m_params.m_featureImgSize*m_vpiObjPatchesNumChannels,
			cudaMemcpyDeviceToHost, m_cudaStream
		)
	);
	CUDA_CHECK( cudaStreamSynchronize(m_cudaStream) );
	CUDA_CHECK( cudaPeekAtLastError() );

	for (uint bboxInd = 0; bboxInd < m_vpiBboxesArraySize; bboxInd++)
	{
		originalImg = cv::Mat(
			m_params.m_featureImgSize, m_params.m_featureImgSize, CV_8UC4,
			reinterpret_cast<uint8_t*>(m_vpiObjPatches_h) + (int)(bboxInd * m_params.m_featureImgSize * m_params.m_featureImgSize * 4)
		);
		originalImg.convertTo(outImg, CV_8UC3);
		cv::cvtColor(outImg, outImg, cv::COLOR_RGB2BGR);

		outputRoiImages[m_targetNumInd2RoisInBatchMapper[bboxInd].first][bboxInd] = outImg.clone();
	}

	return std::move(outputRoiImages);
}
#endif

/** @brief Create and update the DCF filter. */
void DCFTracker::cudaFilterUpdate()
{
	int32_t numFeatureChannels = 0;

	uint lastTargetInd = m_targetManager->getLastTargetInd();

	/// Update object tracking information based on its state and its bounding box on the input frame.
	/// The tracking information is internal to the VPI context.
	VPI_CHECK(
		vpiSubmitDCFTrackerUpdateBatch(
			m_vpiStream, m_params.m_vpiBackend4DcfTracker, m_vpiDcfPayload,
			m_enabledSequences, m_paramsRt.maxStreams, m_vpiMaskingWindow4SrcImage,
			m_vpiMaskingWindow4ModelImage, m_vpiObjPatches, m_vpiDCFBboxesArray, &m_vpiRuntimeParams
		)
	);
	/// Wait until VPI processing is done.
	VPI_CHECK(vpiStreamSync(m_vpiStream));
	/// Get the array with channel weight information for each tracked object.
	VPI_CHECK(vpiDCFTrackerGetChannelWeights(m_vpiDcfPayload, &m_vpiChannelWeightsArray, &numFeatureChannels));
	if(numFeatureChannels != m_featureChannelSize)
	{
		throw std::runtime_error(
			"numFeatureChannels returned by vpiDCFTrackerGetChannelWeights() is not the same as that returned at the initialization stage of the tracker."
		);
	}
	VPI_CHECK(
		vpiArrayLockData(
			m_vpiChannelWeightsArray, VPI_LOCK_READ, VPI_ARRAY_BUFFER_CUDA_AOS, &m_vpiChannelWeightsArrayData
		)
	);
	m_channelWeights = (float *)m_vpiChannelWeightsArrayData.buffer.aos.data;

	/// Recompute correlation responses and confidence after the filter update to refresh the moving peak average.
	cuDCF::calcuateExpectedPeakCorrelation(
		m_cudaStream, numFeatureChannels, (lastTargetInd+1), m_params.m_peakCorrLr,
		m_channelWeights, m_trackerAge_d, m_trackerDoProcessing_d, m_maxCorrResponse_ma_d
	);

	if(m_params.m_debugVerboseLevel > 3)
	{
		#if 0

			/// Sync host and device.
			CUDA_CHECK(
				cudaMemcpyAsync(
					(float*)m_maxCorrResponse_ma_h, m_maxCorrResponse_ma_d,
					(lastTargetInd+1)*sizeof(float), cudaMemcpyDeviceToHost, m_cudaStream
				)
			);

			CUDA_CHECK( cudaStreamSynchronize(m_cudaStream) );
			CUDA_CHECK( cudaPeekAtLastError() );

			printf("\n");
			for(int i=0; i<m_paramsRt.cudaKernelBatchSize/m_paramsRt.featureChannelSize; i++)
			{
				uint targetInd = i;
				if(!m_trackerDoProcessing_h[targetInd])
					continue;

				float expectedPeakCorr = 0;
				for(int ch = 0; ch < m_paramsRt.featureChannelSize; ch++)
				{
					expectedPeakCorr += std::pow(m_channelWeights[i*m_paramsRt.featureChannelSize + ch], 2);
				}

				cout << "@@[DCFTracker] targetInd[" << targetInd << "]: expectedPeakCorr[" << targetInd << "] = " << expectedPeakCorr << ", "  << m_maxCorrResponse_ma_h[targetInd] << endl;
			}

		#endif
	}

	VPI_CHECK(vpiArrayUnlock(m_vpiChannelWeightsArray));
}

#ifdef VIZ_DEBUG
/**
 * @brief Fetch batched device matrices as one host matrix per object.
 * @param[in] deviceData The device buffer containing batched matrices.
 */
std::vector<cv::Mat> DCFTracker::fetchBatchDataToHost_perObj(const float *deviceData)
{
	int lastTargetInd = m_targetManager->getLastTargetInd();

	/// Get the cuComplex-type data from device to host, but only the real part.
	CUDA_CHECK(
		cudaMemcpy2DAsync(
			reinterpret_cast<float*>(m_batchCudaArray_host), sizeof(float),
			reinterpret_cast<const float*>( deviceData ), sizeof(float), sizeof(float),
			(lastTargetInd+1)*m_paramsRt.singleFeatureChannelElementSize, cudaMemcpyDeviceToHost, m_cudaStream
		)
	);

	CUDA_CHECK( cudaStreamSynchronize(m_cudaStream) );
	CUDA_CHECK( cudaPeekAtLastError() );

	/// {targetInd, corrResp}.
	std::vector<cv::Mat>  dataAtHosts(lastTargetInd+1);

	const std::map<NvMOTStreamId, std::shared_ptr<NvTrackedObjectList>> targets = m_targetManager->getTrackedObjectsInUseList();

	for(auto streamIt = targets.begin(); streamIt != targets.end(); ++streamIt)
	{
		for(auto targetIt = streamIt->second->begin(); targetIt != streamIt->second->end(); ++targetIt)
		{
			std::shared_ptr<NvTrackedObject> target = targetIt->second;
			const uint32_t targetInd = target->getTargetInd();
			const uint32_t index = targetInd*m_paramsRt.singleFeatureChannelElementSize;

			if(	target->getTrackerState() == TRACKER_STATE::EMPTY)
			{
				continue;
			}

			cv::Mat dataAtHost(
				m_paramsRt.singleFeatureMatSize.height, m_paramsRt.singleFeatureMatSize.width,
				CV_32FC1, reinterpret_cast<float*>(m_batchCudaArray_host) + index,
				m_paramsRt.singleFeatureMatSize.width*sizeof(float)
			);

			dataAtHosts[targetInd] = dataAtHost.clone();
		}
	}

	return dataAtHosts;
}

/** @brief Return source images copied into host memory. */
std::map<NvMOTStreamId, cv::Mat> DCFTracker::getSrcImagesAtHost()
{
	/// Get source image patches to Host.
	const std::map<NvMOTStreamId, cv::Mat> &srcImgPatches = getSrcFramesAtHost();

	std::map<NvMOTStreamId, cv::Mat> srcImgPatchesBgr;

	for(auto it = srcImgPatches.begin(); it != srcImgPatches.end(); ++it)
	{
		if(m_paramsRt.colorFormat == NVBUF_COLOR_FORMAT_NV12)
		{
			/// Convert NV12 to BGR.
			cvtColor(it->second, srcImgPatchesBgr[it->first], cv::COLOR_YUV2BGR_NV12);
		}
		else if(m_paramsRt.colorFormat == NVBUF_COLOR_FORMAT_RGBA)
		{
			cvtColor(it->second, srcImgPatchesBgr[it->first], cv::COLOR_RGBA2BGR);
		}
		else if(m_paramsRt.colorFormat == NVBUF_COLOR_FORMAT_BGRA)
		{
			cvtColor(it->second, srcImgPatchesBgr[it->first], cv::COLOR_BGRA2BGR);
		}
		else
		{
			throw std::invalid_argument("Invalid color format");
		}
	}

	return srcImgPatchesBgr;
}

/**
 * @brief Copy input frames and tracker data from device to host memory.
 * @param[in] inputMeta A batch of input frames.
 */
void DCFTracker::uploadDeviceData2HostMemories(const std::map<NvMOTStreamId, NvMOTFrame*> &inputMeta)
{
	m_srcFramesAtHost.clear();

	/// Copy host data to device.
	/// Pitch linear.
	uint cnt = 0;
	for(auto it = inputMeta.begin(); it != inputMeta.end(); ++it, ++cnt)
	{
		const NvBufSurfaceParams *baseSurface = it->second->bufferList[0];

		const cv::Size srcImageSize = cv::Size(baseSurface->width, baseSurface->height);
		const uint pitchBytes_src = baseSurface->pitch;

		if(m_params.m_debugVerboseLevel > 2)
		{
			printf(
				"[DCFTracker::uploadDeviceData2HostMemories()] (%d out of %d) baseSurface->width, pitchBytes_src*m_height_in, sizeof(baseSurface), sizeof(m_idata_host) = %d, %d, %d, %d\n",
				(int)cnt, (int)inputMeta.size(), (int)baseSurface->width,
				(int)pitchBytes_src*baseSurface->height, int(pitchBytes_src*baseSurface->planeParams.height[0]),
				int(pitchBytes_src*baseSurface->height*inputMeta.size())
			);
		}

		/// Copy data to the host memory.
		if(baseSurface->colorFormat == NVBUF_COLOR_FORMAT_NV12)
		{
			assert(baseSurface->planeParams.num_planes == 2);

			/// In bytes.
			size_t hPos = pitchBytes_src*(baseSurface->height*1.5f)*cnt;

			for(uint ch=0; ch<baseSurface->planeParams.num_planes; ch++)
			{
				cv::Size srcPlaneSize = cv::Size(
					baseSurface->planeParams.width[ch], baseSurface->planeParams.height[ch]
				);
				uint pitchBytes_plane = baseSurface->planeParams.pitch[ch];
				uint offset = baseSurface->planeParams.offset[ch];

				if(pitchBytes_plane != pitchBytes_src)
				{
					printf("[ERROR] pitchBytes_plane[%d] != pitchBytes_src[%d]\n", pitchBytes_plane, pitchBytes_src);
					assert(pitchBytes_plane == pitchBytes_src);
				}

				if(m_params.m_debugVerboseLevel > 3)
				{
					cout << "[DCFTracker::uploadDeviceData2HostMemories] (hPos, offset) = (" << hPos << ", " << offset << ")" << endl;
				}

				CUDA_CHECK(
					cudaMemcpyAsync(
						reinterpret_cast<uint8_t*>(m_idata_host) + hPos + offset,
						reinterpret_cast<uint8_t*>(baseSurface->dataPtr) + offset,
						pitchBytes_src*srcPlaneSize.height, cudaMemcpyHostToHost, m_cudaStream
					)
				);
			}

			/// Src frame.
			{
				/// Create an empty NV12 image.
				m_srcFramesAtHost[it->first] = cv::Mat::zeros(baseSurface->height*1.5f, baseSurface->width, CV_8UC1);

				cv::Mat y_ch = cv::Mat(
					baseSurface->height, baseSurface->width, CV_8UC1,
					reinterpret_cast<uint8_t*>(m_idata_host) + hPos + baseSurface->planeParams.offset[0], pitchBytes_src
				);
				cv::Mat uv_ch = cv::Mat(
					baseSurface->height*0.5, baseSurface->width, CV_8UC1,
					reinterpret_cast<uint8_t*>(m_idata_host) + hPos + baseSurface->planeParams.offset[1], pitchBytes_src
				);

				y_ch.copyTo(m_srcFramesAtHost[it->first](cv::Rect(0,0,baseSurface->width,baseSurface->height)));
				uv_ch.copyTo(
					m_srcFramesAtHost[it->first](
						cv::Rect(0,baseSurface->height,baseSurface->width,baseSurface->height*0.5)
					)
				);
			}
		}
		else if(baseSurface->colorFormat == NVBUF_COLOR_FORMAT_RGBA)
		{
			/// In bytes.
			size_t hPos = pitchBytes_src*baseSurface->height*cnt;

			CUDA_CHECK(
				cudaMemcpyAsync(
					reinterpret_cast<uint8_t*>(m_idata_host) + hPos, baseSurface->dataPtr,
					pitchBytes_src*srcImageSize.height, cudaMemcpyHostToHost, m_cudaStream
				)
			);
			/// Src frame.
			m_srcFramesAtHost[it->first] = cv::Mat(
				baseSurface->height, baseSurface->width, CV_8UC4,
				reinterpret_cast<uint8_t*>(m_idata_host) + hPos, pitchBytes_src
			);
		}
		else
		{
			std::runtime_error("Invalid color format!");
		}
	}
	CUDA_CHECK( cudaStreamSynchronize(m_cudaStream) );
	CUDA_CHECK( cudaPeekAtLastError() );
}
#endif

/** @brief Construct the default configuration. */
DCFTrackerConfigParams::DCFTrackerConfigParams()
{
	m_showGpuTiming = false;
	m_debugVerboseLevel = 0;
	m_debugVerboseLevel_gpumemory = 0;
	m_debugVerboseLevel_cufft = 0;
	m_visualTrackerType = VISUAL_TRACKER_TYPES::NvDCF_VPI_VOT;

	/// Visual feature types.
	m_useHog = false;
	m_useColorNames = true;
	m_useGray = false;
	m_useRgb = false;

	// [Feature extraction]
	m_featureImgSize = 72;
	m_featureFocusOffsetFactor_y = 0.0f;
	m_hogCellSize = 4;
	m_modelMaskingWindowFactor = 2;

	// [VPI]
	m_useHighPrecisionFeature = false;

	// [DCF][Core]
	m_filterLr = 0.075f;
	m_filterChannelWeightsLr = 0.1f;
	m_gaussianSigma = 0.75f;
	m_lambda = 5.f / 100.0f;
	m_peakCorrLr = 0.1f;

	// [FFT]
	m_cufftHandleQuantFactor = 7;

	// [NvDCF_VPI]
	m_vpiBackend4CropnScale = VPI_BACKEND_CUDA;
	m_vpiBackend4DcfTracker = VPI_BACKEND_CUDA;

	// [Trajectory Visualization]
	m_generateTrackerVisualization = false;
	m_vizWriteSrcImgPatchesAndCorrResp = false;
	m_vizWriteBirdsEyeView = false;
	m_vizWriteAllTrajectoriesOnEveryFrame = false;
	m_vizWriteEachCompletedTrajectory = false;
	m_vizJpegImageQuality = 50;
	m_vizOutDirectory = "trackerViz";
	m_vizDebugStreamId = -1;
}

/**
 * @brief Construct configuration parameters from a named module in a file.
 * @param[in] configFilePath The path to the configuration file.
 * @param[in] moduleName The configuration module name.
 */
DCFTrackerConfigParams::DCFTrackerConfigParams(const std::string &configFilePath, const std::string moduleName)
{
	YAML::Node configRoot = getConfigRoot(configFilePath);
	this->read(configRoot[moduleName]);
}

/**
 * @brief Construct configuration parameters from a YAML node.
 * @param[in] fn Parsed YAML structure containing the configuration.
 */
DCFTrackerConfigParams::DCFTrackerConfigParams(const YAML::Node &fn)
{
	this->read(fn);
}

/**
 * @brief Read visual tracker parameters from a YAML node.
 * @param[in] fn Parsed YAML structure of configuration parameters.
 */
void DCFTrackerConfigParams::read(const YAML::Node &fn)
{
	if(!m_isInit)
	{
		/// Set the default values.
		*this = DCFTrackerConfigParams();
	}

	if(!fn.size())
	{
		m_isInit = true;
		return;
	}

	for (YAML::const_iterator fit = fn.begin(); fit != fn.end(); ++fit)
	{
		std::string paramKey = fit->first.as<std::string>();

		if (paramKey == "debugVerboseLevel")
		{
			m_debugVerboseLevel = (int)fit->second.as<float>();
		}
		else if (paramKey == "showGpuTiming")
		{
			m_showGpuTiming = (int)fit->second.as<float>();
		}
		else if (paramKey == "debugVerboseLevel_gpumemory")
		{
			m_debugVerboseLevel_gpumemory = (int)fit->second.as<float>();
		}
		else if (paramKey == "debugVerboseLevel_cufft")
		{
			m_debugVerboseLevel_cufft = (int)fit->second.as<float>();
		}
		else if (paramKey == "visualTrackerType")
		{
			/// Only support setting this parameter on startup.
			if(!m_isInit)
				m_visualTrackerType = (VISUAL_TRACKER_TYPES)((int)fit->second.as<float>());
			else
				CLOG(std::string("!!![WARNING] m_visualTrackerType is not supported for dynamic updates").c_str());
		}
		else if (paramKey == "useColorNames")
		{
			/// Only support setting this parameter on startup.
			if(!m_isInit)
				m_useColorNames = (int)fit->second.as<float>();
			else
				CLOG(std::string("!!![WARNING] m_useColorNames is not supported for dynamic updates").c_str());
		}
		else if (paramKey == "useHog")
		{
			/// Only support setting this parameter on startup.
			if(!m_isInit)
				m_useHog = (int)fit->second.as<float>();
			else
				CLOG(std::string("!!![WARNING] m_useHog is not supported for dynamic updates").c_str());
		}
		else if (paramKey == "useHighPrecisionFeature")
		{
			/// Only support setting this parameter on startup.
			if(!m_isInit)
				m_useHighPrecisionFeature = (int)fit->second.as<float>();
			else
				CLOG(std::string("!!![WARNING] m_useHighPrecisionFeature is not supported for dynamic updates").c_str());
		}
		else if(paramKey == "featureImgSizeLevel")
		{
			/// Only support setting this parameter on startup.
			if(!m_isInit)
			{
				int featureImgSizeLevel = (int)fit->second.as<float>();

				/// Prefer a multiple of 24 so dividing m_featureImgSize by m_hogCellSize produces the intended feature size.
				if(featureImgSizeLevel >= 1 && featureImgSizeLevel <= 5)
				{
					m_featureImgSize = 6*m_hogCellSize*(featureImgSizeLevel+1);
				}
				else
				{
					throw std::runtime_error("featureImgSizeLevel should be one in {1, 2, 3, 4, 5}");
				}
			}
			else
				CLOG(std::string("!!![WARNING] featureImgSizeLevel is not supported for dynamic updates").c_str());
		}
		else if (paramKey == "featureFocusOffsetFactor_y")
		{
			m_featureFocusOffsetFactor_y = fit->second.as<float>();
		}
		else if (paramKey == "filterLr")
		{
			m_filterLr = fit->second.as<float>();
		}
		else if (paramKey == "filterChannelWeightsLr")
		{
			m_filterChannelWeightsLr = fit->second.as<float>();
		}
		else if (paramKey == "gaussianSigma")
		{
			m_gaussianSigma = fit->second.as<float>();
		}
		else if (paramKey == "lambda")
		{
			m_lambda = fit->second.as<float>();
		}
		else if (paramKey == "peakCorrLr")
		{
			m_peakCorrLr = fit->second.as<float>();
		}
		else if (paramKey == "vpiBackend4CropnScale")
		{
			/// Only support setting this parameter on startup.
			if(!m_isInit)
			{
				int backend = (int)fit->second.as<float>();
				if(backend >= 1 && backend <= 2)
				{
					m_vpiBackend4CropnScale = (VPIBackend)(1ULL << backend);
				}
				else
				{
					throw std::runtime_error("vpiBackend4CropnScale should be one in {1, 2}");
				}
			}
			else
				CLOG(std::string("!!![WARNING] m_useHighPrecisionFeature is not supported for dynamic updates").c_str());
		}
		else if (paramKey == "vpiBackend4DcfTracker")
		{
			/// Only support setting this parameter on startup.
			if(!m_isInit)
			{
				int backend = (int)fit->second.as<float>();
				if(backend >= 1 && backend <= 2)
				{
					m_vpiBackend4DcfTracker = (VPIBackend)(1ULL << backend);
				}
				else
				{
					throw std::runtime_error("vpiBackend4DcfTracker should be one in {1, 2}");
				}
			}
			else
				CLOG(std::string("!!![WARNING] m_useHighPrecisionFeature is not supported for dynamic updates").c_str());
		}
		#ifdef VIZ_DEBUG
			else if(paramKey == "visualization")
			{
				if(fn["visualization"].size() > 0)
				{
					YAML::Node fn_s = fn["visualization"];

					m_generateTrackerVisualization = true;

					for (YAML::const_iterator fjt = fn_s.begin(); fjt != fn_s.end(); ++fjt)
					{
						std::string paramKey_s = fjt->first.as<std::string>();

						if(paramKey_s == "vizWriteSrcImgPatchesAndCorrResp")
						{
							m_vizWriteSrcImgPatchesAndCorrResp = (int)fjt->second.as<float>();
						}
						else if(paramKey_s == "vizWriteBirdsEyeView")
						{
							m_vizWriteBirdsEyeView = (int)fjt->second.as<float>();
						}
						else if(paramKey_s == "vizWriteAllTrajectoriesOnEveryFrame")
						{
							m_vizWriteAllTrajectoriesOnEveryFrame = (int)fjt->second.as<float>();
						}
						else if(paramKey_s == "vizWriteEachCompletedTrajectory")
						{
							m_vizWriteEachCompletedTrajectory = (int)fjt->second.as<float>();
						}
						else if(paramKey_s == "vizJpegImageQuality")
						{
							m_vizJpegImageQuality = (int)fjt->second.as<float>();
						}
						else if(paramKey_s == "vizOutDirectory")
						{
							m_vizOutDirectory = fjt->second.as<std::string>();
						}
						else if(paramKey_s == "vizDebugStreamId")
						{
							m_vizDebugStreamId = (int)fjt->second.as<float>();
						}
						else
						{
							cout << "!! [WARNING][DCFTrackerConfigParams][Visualization] Unknown param found: " << paramKey_s << endl;
						}
					}
				}
			}
		#endif
		else if(paramKey == "debugParamInt0")
		{
			m_debugParamInt0 = (int)fit->second.as<float>();
		}
		else if(paramKey == "debugParamInt1")
		{
			m_debugParamInt1 = (int)fit->second.as<float>();
		}
		else if(paramKey == "debugParamInt2")
		{
			m_debugParamInt2 = (int)fit->second.as<float>();
		}
		else if(paramKey == "debugParamFloat0")
		{
			m_debugParamFloat0 = fit->second.as<float>();
		}
		else if(paramKey == "debugParamFloat1")
		{
			m_debugParamFloat1 = fit->second.as<float>();
		}
		else if(paramKey == "debugParamFloat2")
		{
			m_debugParamFloat2 = fit->second.as<float>();
		}
		else
		{
			cout << "!! [WARNING][DCFTrackerConfigParams] Unknown param found: " << paramKey << endl;
		}
	}

	if(m_debugVerboseLevel > 1)
	{
		cout << "[VisualTracker] visualTrackerType = " << m_visualTrackerType << endl;
	}

	/// Set init flag after initialization to prevent updates from changing.
	m_isInit = true;
}
