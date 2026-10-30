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

#include "modules/utility/cuda_common.hpp"
#include "modules/utility/debug_common.hpp"
#include "modules/utility/cv_functions.hpp"
#include "modules/VisualTracker/VisualTracker.hpp"

#ifdef VIZ_DEBUG
#include "modules/VisualTracker/vizOutput.hpp"
#endif

using std::cout;
using std::endl;

/**
 * @brief Construct from a configuration file path.
 * @param[in] configFilePath The path to the configuration file.
 */
VisualTracker::VisualTracker(const std::string &configFilePath)
{
	m_isInit = false;

	/// Parse the visual tracker configuration.
	m_params = VisualTrackerConfigParams(configFilePath, "VisualTracker");

	m_cufftWorkspce = nullptr;
}

/** @brief Release CUDA DCF tracker resources. */
VisualTracker::~VisualTracker()
{
	#ifdef VIZ_DEBUG
		if(m_targetManager.get() != nullptr && m_targetManager.use_count() > 0)
		{
			m_targetManager->removeAllStreams();

			if(m_vizOutput != nullptr && isInitialized())
			{
				if(m_params.m_vizWriteAllTrajectoriesOnEveryFrame)
				{
					cout << "[VisualTracker][Debug] Wrapping-up visualization" << endl;

					m_vizOutput->writeAllTrajectoriesOnEveryFrame(VIZ_OUTPUT_TYPES::TRACKER_STATES_VIZ);
				}

				if(m_params.m_vizWriteEachCompletedTrajectory)
				{
					cout << "[VisualTracker][Debug] Wrapping-up visualization" << endl;

					m_vizOutput->writeEachCompletedTrajectory(VIZ_OUTPUT_TYPES::TRACKER_STATES_VIZ);
				}

				if(m_params.m_vizWriteBirdsEyeView)
				{
					cout << "[VisualTracker][Debug] Wrapping-up visualization" << endl;

					m_vizOutput->writeBirdsEyeView(VIZ_OUTPUT_TYPES::TRACKER_STATES_VIZ);
				}
			}
		}
	#endif

	if(m_isInit)
	{
		for(uint i=0; i<m_cufftPlanSize; ++i)
		{
			CUFFT_CHECK(cufftDestroy(m_cufftHandle_C2R_perCh[i]));
			CUFFT_CHECK(cufftDestroy(m_cufftHandle_C2R_perObj[i]));
			CUFFT_CHECK(cufftDestroy(m_cufftHandle_R2C_perCh[i]));
		}

		delete [] m_cufftHandle_C2R_perCh;
		delete [] m_cufftHandle_C2R_perObj;
		delete [] m_cufftHandle_R2C_perCh;

		CUDA_CHECK(cudaFree( m_cufftWorkspce ));

		CUDA_CHECK(cudaFree( m_ftrs_F ));
		CUDA_CHECK(cudaFree( m_desiredResp_F ));
		CUDA_CHECK(cudaFree( m_dcfFilter_F ));
		CUDA_CHECK(cudaFree( m_corrResponses_d ));

		CUDA_CHECK(cudaFree( m_maskingWindow4Src ));
		CUDA_CHECK(cudaFree( m_maskingWindow4Model ));
		CUDA_CHECK(cudaFree( m_maskingSegWindow4Model ));

		CUDA_CHECK(cudaFree(m_work_F_1));
		CUDA_CHECK(cudaFree(m_work_F_2));
		CUDA_CHECK(cudaFree(m_work_I));

		CUDA_CHECK(cudaFree(m_channelWeights));
		CUDA_CHECK(cudaFree(m_channelWeightSum));

		CUDA_CHECK(cudaFree(m_targetDisplacement2D_d));
		CUDA_CHECK(cudaFree(m_maxCorrResponse_d));
		CUDA_CHECK(cudaFree(m_maxCorrResponse_ma_d));
		CUDA_CHECK(cudaFree(m_trackingConfidence_d));

		CUDA_CHECK(cudaFreeHost(m_targetDisplacement2D_h));
		CUDA_CHECK(cudaFreeHost(m_maxCorrResponse_h));
		CUDA_CHECK(cudaFreeHost(m_maxCorrResponse_ma_h));
		CUDA_CHECK(cudaFreeHost(m_trackingConfidence_h));

		CUDA_CHECK(cudaFree(m_trackerAge_d));
		CUDA_CHECK(cudaFreeHost(m_trackerAge_h));

		CUDA_CHECK(cudaFree(m_trackerDoProcessing_d));
		CUDA_CHECK(cudaFreeHost(m_trackerDoProcessing_h));

		CUDA_CHECK(cudaFree(m_trackerUseMaskSeg_d));
		CUDA_CHECK(cudaFreeHost(m_trackerUseMaskSeg_h));

		CUDA_CHECK(cudaFree(m_detectorBboxes_d));
		CUDA_CHECK(cudaFreeHost(m_detectorBboxes_h));

		CUDA_CHECK(cudaFree(m_detectorBboxes_size_d));
		CUDA_CHECK(cudaFreeHost(m_detectorBboxes_size_h));

		CUDA_CHECK(cudaFree(m_confidenceAtDetectedPt_d));
		CUDA_CHECK(cudaFreeHost(m_confidenceAtDetectedPt_h));

		CUDA_CHECK(cudaFree(m_corrResponseAtDetectedPt_d));
		CUDA_CHECK(cudaFreeHost(m_corrResponseAtDetectedPt_h));
		if (m_params.m_minConfidence4SegmentationFilter >= 0 && m_params.m_minConfidence4SegmentationFilter <= 1.0)
		{
			CUDA_CHECK(cudaFree(m_maskSegRois_d));
			CUDA_CHECK(cudaFreeHost(m_maskSegRois_h));
		}

		#ifdef VIZ_DEBUG
			CUDA_CHECK(cudaFreeHost(m_batchCudaArray_host));
		#endif

		CUDA_CHECK(cudaStreamDestroy(m_cudaStream));

		if(m_params.m_debugVerboseLevel > 1)
			cout << "[VisualTracker] De-initialized" << endl;
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
void VisualTracker::initialize(
	const SourceFrameInfo &frameInfo, const uint maxTargetsToTrack,
	const uint maxDetectorBboxesPerFrame, const std::shared_ptr<NvTrackedObjectManager> &targetManager
)
{
	m_targetManager = targetManager;

	#ifdef VIZ_DEBUG
		m_vizOutput = std::make_shared<TrackerVizOutput>(this, targetManager);

		if(m_params.m_generateTrackerVisualization && !m_vizOutput->isImageOutputEnabled(VIZ_OUTPUT_TYPES::TRACKER_STATES_VIZ))
		{
			m_vizOutput->enableImageOutput(
				m_params.m_vizOutDirectory, m_params.m_vizJpegImageQuality, VIZ_OUTPUT_TYPES::TRACKER_STATES_VIZ
			);
		}

	#endif

	if(m_targetManager == nullptr)
	{
		throw std::runtime_error("[VisualTracker::initialize()] m_targetManager == nullptr");
	}

	m_paramsRt.maxTargetsToTrack = maxTargetsToTrack;
	m_paramsRt.maxDetectorBboxesPerFrame = maxDetectorBboxesPerFrame;
	m_paramsRt.maxStreams = frameInfo.maxStreams;
	m_paramsRt.originalSrcImageSize = frameInfo.originalSrcImageSize;
	m_paramsRt.srcImageSize = frameInfo.srcImageSize;
	m_paramsRt.srcImagePitch = frameInfo.srcImagePitch;
	m_paramsRt.colorFormat = frameInfo.colorFormat;
	m_paramsRt.templateSize = NvMotDataTypes::Size(m_params.m_featureImgSize, m_params.m_featureImgSize);

	if(m_params.m_debugVerboseLevel > 1)
	{
		cout << "[VisualTracker::initialize] OriginalSrcImageSize = " << m_paramsRt.originalSrcImageSize.width << "x" << m_paramsRt.originalSrcImageSize.height << endl;
		cout << "[VisualTracker::initialize] SrcImageSize = " << m_paramsRt.srcImageSize.width << "x" << m_paramsRt.srcImageSize.height << endl;
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
			"[VisualTracker::initialize()] Make sure m_paramsRt.singleFeatureMatSize.width > 0 && m_paramsRt.singleFeatureMatSize.height > 0"
		);
	}

	/// Require a square image so inversion preserves its dimensions.
	if(m_paramsRt.singleFeatureMatSize.width != m_paramsRt.singleFeatureMatSize.height)
	{
		throw std::runtime_error(
			"[VisualTracker::initialize()] Make sure m_paramsRt.singleFeatureMatSize.width == m_paramsRt.singleFeatureMatSize.height"
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
			"[VisualTracker][GPU] Free mem = %d [MB] among Total = %d [MB]\n", int(free/1024/1024), int(total/1024/1024)
		);
	}

	/// Initialize and allocate GPU memories in feature extractor.
	{
		/// Get the # of channels in feature vector.
		uint featureTypes = 0;

		if (m_params.m_useColorNames)
		{
			featureTypes += FeatureExtractor::FEATURE_TYPES::COLOR_NAMES;
		}

		if (m_params.m_useHog)
		{
			featureTypes += FeatureExtractor::FEATURE_TYPES::HOG;
		}

		if(featureTypes == 0)
		{
			throw std::invalid_argument("[VisualTracker::initialize()] No visual feature types specified");
		}

		/// Initialize the feature extractor.
		m_featureExtractor.initialize(
			m_params.m_debugVerboseLevel, m_paramsRt.maxStreams, frameInfo.maxBufSurfAddrSize,
			m_paramsRt.maxTargetsToTrack, frameInfo.srcImageSize, frameInfo.srcImagePitch, frameInfo.colorFormat,
			NvMotDataTypes::Size(m_paramsRt.templateSize.width, m_paramsRt.templateSize.height),
			m_params.m_hogCellSize, featureTypes, m_params.m_useCpuBackends4VPI ? VPI_BACKEND_CPU : VPI_BACKEND_CUDA,
			m_params.m_useHighPrecisionFeature ? VPI_IMAGE_FORMAT_U16 : VPI_IMAGE_FORMAT_U8
		);

		/// Get the number of output feature channels.
		m_paramsRt.featureChannelSize = m_featureExtractor.getOutputFeatureChannelSize();

		if(m_paramsRt.featureChannelSize == 0)
		{
			throw std::runtime_error("[VisualTracker::initialize()] Make sure m_paramsRt.featureChannelSize > 0");
		}
	}

	if(m_params.m_debugVerboseLevel_gpumemory > 0)
	{
		size_t free, total;
		CUDA_CHECK( cudaMemGetInfo( &free, &total ) );

		printf(
			"[VisualTracker][GPU]	[after feature extractor initialized] Free mem = %d [MB] among Total = %d [MB] --> memory increase = %d [MB]\n",
			int(free/1024/1024), int(total/1024/1024), int((freeGpuMemory_prev-free)/1024/1024)
		);
		freeGpuMemory_prev = free;
	}

	m_paramsRt.singleFeatureChannelElementSize =  static_cast<int>(
		m_paramsRt.singleFeatureMatSize.width*m_paramsRt.singleFeatureMatSize.height
	);
	m_paramsRt.singleFeatureChannelElementSize_F =  static_cast<int>(
		(m_paramsRt.singleFeatureMatSize.width / 2 + 1)*m_paramsRt.singleFeatureMatSize.height
	);
	m_paramsRt.cudaKernelMaxBatchSize = m_paramsRt.featureChannelSize*m_paramsRt.maxTargetsToTrack;
	m_paramsRt.featureVecMaxBatchElementSize =  static_cast<int>(
		m_paramsRt.singleFeatureChannelElementSize*m_paramsRt.cudaKernelMaxBatchSize
	);
	m_paramsRt.featureVecMaxBatchElementSize_F =  static_cast<int>(
		m_paramsRt.singleFeatureChannelElementSize_F*m_paramsRt.cudaKernelMaxBatchSize
	);

	if(m_params.m_debugVerboseLevel > 1)
	{
		printf(
			"[VisualTracker::initialize()] m_paramsRt.singleFeatureMatSize = (%d, %d)\n",
			m_paramsRt.singleFeatureMatSize.width, m_paramsRt.singleFeatureMatSize.height
		);
		printf(
			"[VisualTracker::initialize()] m_paramsRt.cudaKernelMaxBatchSize = m_paramsRt.featureChannelSize*m_paramsRt.maxTargetsToTrack: %d = %d*%d\n",
			m_paramsRt.cudaKernelMaxBatchSize, m_paramsRt.featureChannelSize, m_paramsRt.maxTargetsToTrack
		);
		printf(
			"[VisualTracker::initialize()] m_paramsRt.featureVecMaxBatchElementSize = m_paramsRt.singleFeatureChannelElementSize*m_paramsRt.cudaKernelMaxBatchSize: %d = %d*%d\n",
			m_paramsRt.featureVecMaxBatchElementSize,
			m_paramsRt.singleFeatureChannelElementSize, m_paramsRt.cudaKernelMaxBatchSize
		);
		printf(
			"[VisualTracker::initialize()] m_paramsRt.featureVecMaxBatchElementSize_F = m_paramsRt.singleFeatureChannelElementSize_F*m_paramsRt.cudaKernelMaxBatchSize: %d = %d*%d\n",
			m_paramsRt.featureVecMaxBatchElementSize_F,
			m_paramsRt.singleFeatureChannelElementSize_F, m_paramsRt.cudaKernelMaxBatchSize
		);
	}

	/// Update some metadata.
	updateData4Processing(m_paramsRt.maxTargetsToTrack-1);

	/// Allocate custom CUDA kernel memories.
	allocateCudaMemories();

	if(m_params.m_debugVerboseLevel_gpumemory > 0)
	{
		size_t free, total;
		CUDA_CHECK( cudaMemGetInfo( &free, &total ) );

		printf(
			"[VisualTracker][GPU]	[after all custom CUDA memories allocated] Free mem = %d [MB] among Total = %d [MB] --> memory increase = %d [MB]\n",
			int(free/1024/1024), int(total/1024/1024), int((freeGpuMemory_prev-free)/1024/1024)
		);
		freeGpuMemory_prev = free;
	}

	/// FFT.
	{
		int shape[2] = { m_paramsRt.singleFeatureMatSize.height, m_paramsRt.singleFeatureMatSize.width};

		CUDA_CHECK( cudaStreamCreateWithFlags(&m_cudaStream, cudaStreamNonBlocking) );

		/// Higher GPU memory usage as m_cufftPlanSize gets larger.
		m_cufftPlanSize = m_paramsRt.maxTargetsToTrack/(1<<m_params.m_cufftHandleQuantFactor) + 1;

		m_cufftHandle_C2R_perCh = new cufftHandle[m_cufftPlanSize];
		m_cufftHandle_C2R_perObj = new cufftHandle[m_cufftPlanSize];
		m_cufftHandle_R2C_perCh = new cufftHandle[m_cufftPlanSize];

		size_t workSizeMax = 0;

		/// Starts from the larger plan.
		for(int i=m_cufftPlanSize-1; i>=0; --i)
		{
			int batchSizePerTarget = std::min(
				m_paramsRt.maxTargetsToTrack, (uint32_t)(i+1)*(1<<m_params.m_cufftHandleQuantFactor)
			);
			int batchSizePerChannel = m_paramsRt.featureChannelSize*batchSizePerTarget;

			if(m_params.m_debugVerboseLevel_cufft > 0)
			{
				cout << "[VisualTracker::initialize()] cuFFT Plan[" << i << "] batch size for per-target cuFFT = " << batchSizePerTarget << ", among total # of objects = " << m_paramsRt.maxTargetsToTrack << endl;
				cout << "[VisualTracker::initialize()] cuFFT Plan[" << i << "] batch size for per-channel cuFFT = " << batchSizePerChannel << ", among total (# of channels)*(# of objects) = " << m_paramsRt.featureChannelSize*m_paramsRt.maxTargetsToTrack << endl;
			}

			/// Create cuFFT plans.
			CUFFT_CHECK( cufftCreate(&m_cufftHandle_C2R_perCh[i]) );
			CUFFT_CHECK( cufftCreate(&m_cufftHandle_C2R_perObj[i]) );
			CUFFT_CHECK( cufftCreate(&m_cufftHandle_R2C_perCh[i]) );

			/// Set auto allocation off to let cufft not allocate any memory automatically.
			CUFFT_CHECK( cufftSetAutoAllocation(m_cufftHandle_R2C_perCh[i], false) );
			CUFFT_CHECK( cufftSetAutoAllocation(m_cufftHandle_C2R_perCh[i], false) );
			CUFFT_CHECK( cufftSetAutoAllocation(m_cufftHandle_C2R_perObj[i], false) );

			size_t workSize_C2R_perCh = 0, workSize_C2R_perObj = 0, workSize_R2C_perCh = 0;

			/// Plan N batches of 2D matrices of size H*W.
			CUFFT_CHECK(
				cufftMakePlanMany(
					m_cufftHandle_C2R_perObj[i], 2, shape, NULL, 1, 0, NULL,
					1, 0, CUFFT_C2R, batchSizePerTarget, &workSize_C2R_perObj
				)
			);

			if(m_params.m_debugVerboseLevel_gpumemory > 0)
			{
				size_t free, total;
				CUDA_CHECK( cudaMemGetInfo( &free, &total ) );

				printf(
					"[VisualTracker][GPU]	[after cufftMakePlanMany C2R_perObj created] Free mem = %d [MB] among Total = %d [MB] --> memory increase = %d [MB]\n",
					int(free/1024/1024), int(total/1024/1024), int((freeGpuMemory_prev-free)/1024/1024)
				);
				freeGpuMemory_prev = free;
			}

			/// Plan N*C batches of 2D matrices of size H*W.
			CUFFT_CHECK(
				cufftMakePlanMany(
					m_cufftHandle_R2C_perCh[i], 2, shape, NULL, 1, 0, NULL,
					1, 0, CUFFT_R2C, batchSizePerChannel, &workSize_R2C_perCh
				)
			);

			if(m_params.m_debugVerboseLevel_gpumemory > 0)
			{
				size_t free, total;
				CUDA_CHECK( cudaMemGetInfo( &free, &total ) );

				printf(
					"[VisualTracker][GPU]	[after cufftMakePlanMany R2C_perCh created] Free mem = %d [MB] among Total = %d [MB] --> memory increase = %d [MB]\n",
					int(free/1024/1024), int(total/1024/1024), int((freeGpuMemory_prev-free)/1024/1024)
				);
				freeGpuMemory_prev = free;
			}

			/// Plan N*C batches of 2D matrices of size H*W.
			CUFFT_CHECK(
				cufftMakePlanMany(
					m_cufftHandle_C2R_perCh[i], 2, shape, NULL, 1, 0, NULL,
					1, 0, CUFFT_C2R, batchSizePerChannel, &workSize_C2R_perCh
				)
			);

			if(m_params.m_debugVerboseLevel_gpumemory > 0)
			{
				size_t free, total;
				CUDA_CHECK( cudaMemGetInfo( &free, &total ) );

				printf(
					"[VisualTracker][GPU]	[after cufftMakePlanMany C2R_perCh created] Free mem = %d [MB] among Total = %d [MB] --> memory increase = %d [MB]\n",
					int(free/1024/1024), int(total/1024/1024), int((freeGpuMemory_prev-free)/1024/1024)
				);
				freeGpuMemory_prev = free;
			}

			if(m_params.m_debugVerboseLevel_cufft > 0)
			{
				printf(
					"[VisualTracker::initialize()][cuFFT] cufftPlan[%d]: workSize for (C2R_perCh, R2C_perCH, C2R_perObj) = (%d, %d, %d) [MB]\n",
					i, int(workSize_C2R_perCh/1024/1024),
					int(workSize_R2C_perCh/1024/1024), int(workSize_C2R_perObj/1024/1024)
				);
			}

			workSizeMax = std::max(
				workSizeMax, std::max(workSize_C2R_perObj, std::max(workSize_C2R_perCh, workSize_R2C_perCh))
			);

			/// Manually set the CUDA stream.
			CUFFT_CHECK( cufftSetStream(m_cufftHandle_C2R_perCh[i], m_cudaStream) );
			CUFFT_CHECK( cufftSetStream(m_cufftHandle_C2R_perObj[i], m_cudaStream) );
			CUFFT_CHECK( cufftSetStream(m_cufftHandle_R2C_perCh[i], m_cudaStream) );
		}

		CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &m_cufftWorkspce ), workSizeMax));

		if(m_params.m_debugVerboseLevel_gpumemory > 0)
		{
			size_t free, total;
			CUDA_CHECK( cudaMemGetInfo( &free, &total ) );

			printf(
				"[VisualTracker][GPU]	[after all three m_cufftWorkspce allocated] Free mem = %d [MB] among Total = %d [MB] --> memory increase = %d [MB]\n",
				int(free/1024/1024), int(total/1024/1024), int((freeGpuMemory_prev-free)/1024/1024)
			);
			freeGpuMemory_prev = free;
		}

		/// Manually set the work area.
		for(int i=m_cufftPlanSize-1; i>=0; --i)
		{
			CUFFT_CHECK( cufftSetWorkArea(m_cufftHandle_C2R_perCh[i], m_cufftWorkspce) );
			CUFFT_CHECK( cufftSetWorkArea(m_cufftHandle_C2R_perObj[i], m_cufftWorkspce) );
			CUFFT_CHECK( cufftSetWorkArea(m_cufftHandle_R2C_perCh[i], m_cufftWorkspce) );
		}
	}

	if(m_params.m_debugVerboseLevel_cufft > 0)
	{
		printf("[VisualTracker::initialize()] m_cufftPlanSize = %d (x2)\n", m_cufftPlanSize);
	}

	/// Initialize the tracker states.
	{
		CUDA_CHECK( cudaMemsetAsync(m_trackerAge_d, 0, m_paramsRt.maxTargetsToTrack*sizeof(uint32_t), m_cudaStream) );
		memset(m_trackerAge_h, 0, m_paramsRt.maxTargetsToTrack*sizeof(uint32_t));

		CUDA_CHECK(
			cudaMemsetAsync(m_trackerDoProcessing_d, false, m_paramsRt.maxTargetsToTrack*sizeof(bool), m_cudaStream)
		);
		memset(m_trackerDoProcessing_h, false, m_paramsRt.maxTargetsToTrack*sizeof(bool));

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

	/// Create a Gaussian for the desired response.
	NvMotDataTypes::RowMajorMatrixXf desiredResp = CV_FUNCS::get2dGaussianPdf(
		m_params.m_gaussianSigma, m_paramsRt.singleFeatureMatSize.width, m_paramsRt.singleFeatureMatSize.height
	);

	{
		/// Allocate the host batch buffer used to download and de-batch GPU matrices.
		float *workarea_host;
		/// Desired response in image space (a Batch CUDA memory buffer).
		float *desiredResp_I = m_work_I;

		/// Allocate a temporary memory in the host the batch-array.
		CUDA_CHECK(
			cudaMallocHost(
				reinterpret_cast<void**>( &workarea_host ), m_paramsRt.featureVecMaxBatchElementSize*sizeof(cuComplex)
			)
		);

		makeBatchAndSendFloatMatToDevice(
			maskingWindow4Model, m_maskingWindow4Model,
			m_paramsRt.featureChannelSize*m_paramsRt.maxTargetsToTrack, workarea_host
		);
		makeBatchAndSendFloatMatToDevice(
			maskingWindow4Src, m_maskingWindow4Src,
			m_paramsRt.featureChannelSize*m_paramsRt.maxTargetsToTrack, workarea_host
		);
		makeBatchAndSendFloatMatToDevice(
			desiredResp, desiredResp_I, m_paramsRt.featureChannelSize*m_paramsRt.maxTargetsToTrack, workarea_host
		);
		CUDA_CHECK(
			cudaMemcpyAsync(
				m_maskingSegWindow4Model, m_maskingWindow4Model,
				m_paramsRt.featureVecMaxBatchElementSize*sizeof(float), cudaMemcpyDeviceToDevice, m_cudaStream
			)
		);

		/// Store Fourier-transformed desired response. m_desiredResp_F has half size due to conjugate symmetry.
		CUFFT_CHECK(
			cufftExecR2C(
				m_cufftHandle_R2C_perCh[m_cufftPlanSize - 1],
				(cufftReal *)desiredResp_I, (cufftComplex *)m_desiredResp_F
			)
		);

		CUDA_CHECK( cudaStreamSynchronize(m_cudaStream) );
		CUDA_CHECK( cudaPeekAtLastError() );

		/// Free the temporary memory.
		CUDA_CHECK( cudaFreeHost(workarea_host) );
	}

	if(m_params.m_debugVerboseLevel_gpumemory > 0)
	{
		size_t free, total;
		CUDA_CHECK( cudaMemGetInfo( &free, &total ) );

		printf(
			"[VisualTracker][GPU] Free mem = %d [MB], Total = %d [MB] ===> Total allocated memory = %d [MB]\n",
			int(free/1024/1024), int(total/1024/1024), int((freeGpuMemory_init - free)/1024/1024)
		);
	}

	m_paramsRt.detectorInputInterval = -1;
	m_isInit = true;

	if(m_params.m_debugVerboseLevel > 1)
		cout << "[VisualTracker] Initialized" << endl;
}

/**
 * @brief Populate the visual tracker's capability query.
 * @param[out] paramsQuery The output query parameters.
 */
void VisualTracker::queryParams(NvMOTQuery &paramsQuery)
{
	paramsQuery.numTransforms = 1;
}

/**
 * @brief Select a cuFFT plan large enough for the requested target count.
 * @param[in] num_target The number of targets to process.
 * @return The identifier of the selected cuFFT plan.
 */
inline int VisualTracker::getCufftPlanId(int num_target)
{
	float ind = num_target/float(1<<m_params.m_cufftHandleQuantFactor);

	return floorf(ind);
}

/**
 * @brief Adjust processing dimensions after the active tracker count changes.
 * @param[in] lastTargetInd The index of the last tracker in the active batch.
 */
void VisualTracker::updateData4Processing(const int lastTargetInd)
{
	if(lastTargetInd < 0)
		return;

	/// Total batch size would be (The # of channels per target)*(The # of target trackers).
	m_paramsRt.cudaKernelBatchSize = m_paramsRt.featureChannelSize*(lastTargetInd+1);

	if(m_paramsRt.cudaKernelBatchSize > m_paramsRt.cudaKernelMaxBatchSize)
	{
		std::ostringstream msg;
		msg << "[ERROR] m_paramsRt.cudaKernelBatchSize > m_paramsRt.cudaKernelMaxBatchSize: " << m_paramsRt.cudaKernelBatchSize << " > " << m_paramsRt.cudaKernelMaxBatchSize;
		throw std::runtime_error(msg.str());
	}

	m_paramsRt.featureVecBatchElementSize = static_cast<int>(
		m_paramsRt.singleFeatureChannelElementSize*m_paramsRt.cudaKernelBatchSize
	);

}

/** @brief Pre-allocate required cuda memories on GPU. */
void VisualTracker::allocateCudaMemories()
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
		CUDA_CHECK(
			cudaMallocHost(
				reinterpret_cast<void**>( &m_batchCudaArray_host ),
				m_paramsRt.featureVecMaxBatchElementSize*sizeof(cuComplex)
			)
		);
	#endif

	CUDA_CHECK(
		cudaMalloc(
			reinterpret_cast<void**>( &m_ftrs_F ), m_paramsRt.featureVecMaxBatchElementSize_F*sizeof(cuComplex)
		)
	);

	if(m_params.m_debugVerboseLevel_gpumemory > 0)
	{
		size_t free, total;
		CUDA_CHECK( cudaMemGetInfo( &free, &total ) );

		printf(
			"[VisualTracker][GPU]		[m_ftrs_F] Free mem = %d [MB] among Total = %d [MB] --> memory increase = %d [MB]\n",
			int(free/1024/1024), int(total/1024/1024), int((freeGpuMemory_prev-free)/1024/1024)
		);
		freeGpuMemory_prev = free;
	}

	CUDA_CHECK(
		cudaMalloc(
			reinterpret_cast<void**>( &m_dcfFilter_F ), m_paramsRt.featureVecMaxBatchElementSize_F*sizeof(cuComplex)
		)
	);

	if(m_params.m_debugVerboseLevel_gpumemory > 0)
	{
		size_t free, total;
		CUDA_CHECK( cudaMemGetInfo( &free, &total ) );

		printf(
			"[VisualTracker][GPU]		[m_dcfFilter_F] Free mem = %d [MB] among Total = %d [MB] --> memory increase = %d [MB]\n",
			int(free/1024/1024), int(total/1024/1024), int((freeGpuMemory_prev-free)/1024/1024)
		);
		freeGpuMemory_prev = free;
	}

	CUDA_CHECK(
		cudaMalloc(
			reinterpret_cast<void**>( &m_desiredResp_F ),
			m_paramsRt.featureVecMaxBatchElementSize_F*sizeof(cuComplex)
		)
	);

	if(m_params.m_debugVerboseLevel_gpumemory > 0)
	{
		size_t free, total;
		CUDA_CHECK( cudaMemGetInfo( &free, &total ) );

		printf(
			"[VisualTracker][GPU]		[m_desiredResp_F] Free mem = %d [MB] among Total = %d [MB] --> memory increase = %d [MB]\n",
			int(free/1024/1024), int(total/1024/1024), int((freeGpuMemory_prev-free)/1024/1024)
		);
		freeGpuMemory_prev = free;
	}

	CUDA_CHECK(
		cudaMalloc(
			reinterpret_cast<void**>( &m_maskingWindow4Src ), m_paramsRt.featureVecMaxBatchElementSize*sizeof(float)
		)
	);

	if(m_params.m_debugVerboseLevel_gpumemory > 0)
	{
		size_t free, total;
		CUDA_CHECK( cudaMemGetInfo( &free, &total ) );

		printf(
			"[VisualTracker][GPU]		[m_maskingWindow4Src] Free mem = %d [MB] among Total = %d [MB] --> memory increase = %d [MB]\n",
			int(free/1024/1024), int(total/1024/1024), int((freeGpuMemory_prev-free)/1024/1024)
		);
		freeGpuMemory_prev = free;
	}

	CUDA_CHECK(
		cudaMalloc(
			reinterpret_cast<void**>( &m_maskingWindow4Model ),
			m_paramsRt.featureVecMaxBatchElementSize*sizeof(float)
		)
	);

	if(m_params.m_debugVerboseLevel_gpumemory > 0)
	{
		size_t free, total;
		CUDA_CHECK( cudaMemGetInfo( &free, &total ) );

		printf(
			"[VisualTracker][GPU]		[m_maskingWindow4Model] Free mem = %d [MB] among Total = %d [MB] --> memory increase = %d [MB]\n",
			int(free/1024/1024), int(total/1024/1024), int((freeGpuMemory_prev-free)/1024/1024)
		);
		freeGpuMemory_prev = free;
	}

	CUDA_CHECK(
		cudaMalloc(
			reinterpret_cast<void**>( &m_maskingSegWindow4Model ),
			m_paramsRt.featureVecMaxBatchElementSize*sizeof(float)
		)
	);

	if(m_params.m_debugVerboseLevel_gpumemory > 0)
	{
		size_t free, total;
		CUDA_CHECK( cudaMemGetInfo( &free, &total ) );

		printf(
			"[VisualTracker][GPU]		[m_maskingSegWindow4Model] Free mem = %d [MB] among Total = %d [MB] --> memory increase = %d [MB]\n",
			int(free/1024/1024), int(total/1024/1024), int((freeGpuMemory_prev-free)/1024/1024)
		);
		freeGpuMemory_prev = free;
	}

	/// Allocate for one target and share.
	CUDA_CHECK(
		cudaMalloc(
			reinterpret_cast<void**>( &m_work_F_1 ), m_paramsRt.featureVecMaxBatchElementSize_F*sizeof(cuComplex)
		)
	);

	if(m_params.m_debugVerboseLevel_gpumemory > 0)
	{
		size_t free, total;
		CUDA_CHECK( cudaMemGetInfo( &free, &total ) );

		printf(
			"[VisualTracker][GPU]		[m_work_F_1] Free mem = %d [MB] among Total = %d [MB] --> memory increase = %d [MB]\n",
			int(free/1024/1024), int(total/1024/1024), int((freeGpuMemory_prev-free)/1024/1024)
		);
		freeGpuMemory_prev = free;
	}

	CUDA_CHECK(
		cudaMalloc(
			reinterpret_cast<void**>( &m_work_F_2 ), m_paramsRt.featureVecMaxBatchElementSize_F*sizeof(cuComplex)
		)
	);

	if(m_params.m_debugVerboseLevel_gpumemory > 0)
	{
		size_t free, total;
		CUDA_CHECK( cudaMemGetInfo( &free, &total ) );

		printf(
			"[VisualTracker][GPU]		[m_work_F_2] Free mem = %d [MB] among Total = %d [MB] --> memory increase = %d [MB]\n",
			int(free/1024/1024), int(total/1024/1024), int((freeGpuMemory_prev-free)/1024/1024)
		);
		freeGpuMemory_prev = free;
	}

	CUDA_CHECK(
		cudaMalloc( reinterpret_cast<void**>( &m_work_I ), m_paramsRt.featureVecMaxBatchElementSize*sizeof(float) )
	);

	if(m_params.m_debugVerboseLevel_gpumemory > 0)
	{
		size_t free, total;
		CUDA_CHECK( cudaMemGetInfo( &free, &total ) );

		printf(
			"[VisualTracker][GPU]		[m_work_I] Free mem = %d [MB] among Total = %d [MB] --> memory increase = %d [MB]\n",
			int(free/1024/1024), int(total/1024/1024), int((freeGpuMemory_prev-free)/1024/1024)
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
			"[VisualTracker][GPU]		[m_corrResponses_d] Free mem = %d [MB] among Total = %d [MB] --> memory increase = %d [MB]\n",
			int(free/1024/1024), int(total/1024/1024), int((freeGpuMemory_prev-free)/1024/1024)
		);
		freeGpuMemory_prev = free;
	}

	#if !defined(__aarch64__) && defined(VIZ_DEBUG)
		CUDA_CHECK(
			cudaMallocManaged(
				reinterpret_cast<void**>( &m_channelWeights ), m_paramsRt.cudaKernelMaxBatchSize*sizeof(float)
			)
		);

		/// Define the sum of weights for each target for later weight normalization.
		CUDA_CHECK(
			cudaMallocManaged(
				reinterpret_cast<void**>( &m_channelWeightSum ), m_paramsRt.maxTargetsToTrack*sizeof(float)
			)
		);

	#else
		CUDA_CHECK(
			cudaMalloc( reinterpret_cast<void**>( &m_channelWeights ), m_paramsRt.cudaKernelMaxBatchSize*sizeof(float) )
		);

		/// Define the sum of weights for each target for later weight normalization.
		CUDA_CHECK(
			cudaMalloc( reinterpret_cast<void**>( &m_channelWeightSum ),  m_paramsRt.maxTargetsToTrack*sizeof(float) )
		);
	#endif

	CUDA_CHECK(
		cudaMalloc(reinterpret_cast<void**>(&m_targetDisplacement2D_d), m_paramsRt.maxTargetsToTrack*sizeof(float2))
	);
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
		cudaMallocHost(reinterpret_cast<void**>(&m_targetDisplacement2D_h), m_paramsRt.maxTargetsToTrack*sizeof(float2))
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
		cudaMalloc(reinterpret_cast<void**>(&m_trackerUseMaskSeg_d), m_paramsRt.maxTargetsToTrack*sizeof(bool))
	);
	CUDA_CHECK(
		cudaMallocHost(reinterpret_cast<void**>(&m_trackerUseMaskSeg_h), m_paramsRt.maxTargetsToTrack*sizeof(bool))
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

	if (m_params.m_minConfidence4SegmentationFilter >= 0 && m_params.m_minConfidence4SegmentationFilter <= 1.0)
	{
		CUDA_CHECK(
			cudaMalloc(reinterpret_cast<void**>(&m_maskSegRois_d), m_paramsRt.maxTargetsToTrack*sizeof(float4))
		);
		CUDA_CHECK(
			cudaMallocHost(reinterpret_cast<void**>(&m_maskSegRois_h), m_paramsRt.maxTargetsToTrack*sizeof(float4))
		);
	}

	if(m_params.m_debugVerboseLevel_gpumemory > 0)
	{
		size_t free, total;
		CUDA_CHECK( cudaMemGetInfo( &free, &total ) );

		printf(
			"[VisualTracker][GPU]		[all other misc. memories] Free mem = %d [MB] among Total = %d [MB] --> memory increase = %d [MB]\n",
			int(free/1024/1024), int(total/1024/1024), int((freeGpuMemory_prev-free)/1024/1024)
		);
		freeGpuMemory_prev = free;
	}

	if(m_params.m_debugVerboseLevel > 1)
	{
		printf(
			"[VisualTracker::allocateCudaMemories()] Allocated CUDA memories. m_paramsRt.maxTargetsToTrack = %d\n ",
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
void VisualTracker::makeBatchAndSendFloatMatToDevice(
	const NvMotDataTypes::RowMajorMatrixXf &srcMat, void *devMem, uint batchSize, float *workarea_host
)
{
	if(devMem == nullptr)
	{
		throw std::runtime_error("[VisualTracker::makeBatchAndSendFloatMatToDevice] Make sure devMem != nullptr");
	}

	if(srcMat.rows()*srcMat.cols() != m_paramsRt.singleFeatureChannelElementSize)
	{
		throw std::runtime_error(
			"[VisualTracker::makeBatchAndSendFloatMatToDevice] Make sure srcMat.rows*srcMat.cols == m_paramsRt.singleFeatureChannelElementSize"
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
float VisualTracker::getTemplateRescaleRatio(const NvMotDataTypes::Rect2f &roi) const
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
Eigen::Vector2f VisualTracker::convertSrcImgCoordToCorrRespCoord(
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
void VisualTracker::getDetectorBboxCorrelationResponses(
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

	/// Given the Fourier-transformed extracted features and the learned filters, compute the correlation response.
	cuDCF::getDetectorBboxCorrelationResponses(
		m_cudaStream, (lastTargetInd+1), m_paramsRt.maxDetectorBboxesPerFrame,
		m_detectorBboxes_d, m_detectorBboxes_size_d, m_paramsRt.detCorrRespSearchWindowSize,
		m_corrResponses_d, m_maxCorrResponse_ma_d, m_paramsRt.singleFeatureChannelElementSize,
		m_paramsRt.singleFeatureMatSize.width, m_confidenceAtDetectedPt_d, m_corrResponseAtDetectedPt_d
	);

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

		cout << "[VisualTracker::getDetectorBboxCorrelationResponses]" << endl;
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

/**
 * @brief Given the Fourier-transformed extracted features and the learned filters, compute the correlation response.
 * @param[in] ftrs The device buffer containing the extracted features.
 */
void VisualTracker::cudaCalculateResponse(float *ftrs)
{
	if(ftrs == nullptr)
		return;

	int lastTargetInd = m_targetManager->getLastTargetInd();
	uint cufftPlanId = getCufftPlanId(lastTargetInd+1);

	/// Perform DFT on GPU to have features in Fourier space. m_ftrs_F has half img size due to conjugate symmetry.
	CUFFT_CHECK(cufftExecR2C(m_cufftHandle_R2C_perCh[cufftPlanId], (cufftReal *)ftrs, (cufftComplex *)m_ftrs_F));

	/// Given the Fourier-transformed extracted features and the learned filters, compute the correlation response.
	cuDCF::calculateResponse(
		m_cudaStream, m_cufftHandle_C2R_perObj[cufftPlanId],
		m_paramsRt.singleFeatureMatSize.width, m_paramsRt.singleFeatureChannelElementSize,
		(lastTargetInd+1)*m_paramsRt.singleFeatureChannelElementSize, m_paramsRt.featureChannelSize,
		(lastTargetInd+1), reinterpret_cast<cuComplex*>(m_ftrs_F), reinterpret_cast<cuComplex*>(m_dcfFilter_F),
		m_channelWeights, m_work_F_1, m_corrResponses_d, m_maxCorrResponse_d, m_targetDisplacement2D_d
	);

	/// Compute the moving average of the max corr resp.
	cuDCF::calculateTrackingConfidence(
		m_cudaStream, (lastTargetInd+1), m_maxCorrResponse_ma_d, m_maxCorrResponse_d, m_trackingConfidence_d
	);

	/// Sync host and device.
	CUDA_CHECK(
		cudaMemcpyAsync(
			(float2*)m_targetDisplacement2D_h, m_targetDisplacement2D_d,
			(lastTargetInd+1)*sizeof(float2), cudaMemcpyDeviceToHost, m_cudaStream
		)
	);
	CUDA_CHECK(
		cudaMemcpyAsync(
			(float*)m_maxCorrResponse_h, m_maxCorrResponse_d,
			(lastTargetInd+1)*sizeof(float), cudaMemcpyDeviceToHost, m_cudaStream
		)
	);
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

			cout << "@@[VisualTracker] targetInd[" << targetInd << "]: (maxCorrResponse_ma, maxCorrResponse, trackingConfidence) = ("
						<< m_maxCorrResponse_ma_h[targetInd] << ", " << m_maxCorrResponse_h[targetInd]
						<< ", " << m_trackingConfidence_h[targetInd] << ")" << endl;
		}
	}

}

/**
 * @brief Update target states from their correlation responses.
 * @param[in] inputMeta A batch of input frames.
 */
void VisualTracker::updateLocalizationResults(const std::map<NvMOTStreamId, NvMOTFrame*> &inputMeta)
{
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

			/// Get the search region used to perform localization.
			const NvMotDataTypes::Rect2f predictedSearchRegion = target->getPredictedSearchRegion();

			/// Scale the search region size to be static to fit to the featureImgSizeLevel, so that all the templates have the same dimensions.
			/// Don't make this dynamic unless you have a proper way to address irregular template size in batch processing.
			float templateRescaleRatio = getTemplateRescaleRatio(predictedSearchRegion);

			Eigen::Vector2f displacementInImg(
				m_targetDisplacement2D_h[targetInd].x/templateRescaleRatio*m_params.m_hogCellSize,
				m_targetDisplacement2D_h[targetInd].y/templateRescaleRatio*m_params.m_hogCellSize
			);

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
				cout << "[VisualTracker] Stream[" << (videoStreamId>>32) << "] Frame[" << frameNum << "] TargetInd[" << targetInd
						<< "] ID[" << target->getTargetID()
						<< "] (maxCorrResp, dX, dY) = (" << m_maxCorrResponse_h[targetInd] << ", "
						<< displacementInImg(0) << ", "
						<< displacementInImg(1) << ")" << endl;
			}
		}
	}
}

/**
 * @brief Localize tracked targets and evaluate nearby detections across an input batch.
 * @param[in] inputMeta A batch of input frames.
 * @param[in] detectedObjects The list of newly-detected objects.
 */
void VisualTracker::localize(
	const std::map<NvMOTStreamId, NvMOTFrame *> inputMeta,
	const std::map<NvMOTStreamId, std::deque<DetectedObjInfo>> &detectedObjects
)
{
	m_inputMeta = inputMeta;

	/// Compute the correlation response.
	cudaCalculateResponse( extractVisualFeatures(inputMeta, true) );

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
 * @brief Check the input and find the detection interval.
 * @param[in] inputMeta A batch of input frames.
 */
void VisualTracker::checkInputOutput(const std::map<NvMOTStreamId, NvMOTFrame*> &inputMeta)
{
	for (auto it = inputMeta.begin(); it != inputMeta.end(); ++it)
	{
		const NvMOTStreamId &videoStreamId = it->first;
		const NvMOTFrame *srcMotFrame = it->second;

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
				cout << "[VisualTracker::checkInputOutput()] Found detection interval = " << m_paramsRt.detectorInputInterval << "]" << endl;
				cout << "[VisualTracker::checkInputOutput()] (filterLr, channelWeightsLr, peakCorrLr) = ("
						<< m_params.m_filterLr << ", " << m_params.m_filterChannelWeightsLr << ", " << m_params.m_peakCorrLr << ")" << endl;
			}
		}

		if(srcMotFrame->objectsIn.detectionDone == true)
		{
			m_paramsRt.isDetectionDonePerStream[videoStreamId] = 0;
		}
		else
		{
			m_paramsRt.isDetectionDonePerStream[videoStreamId]++;
		}
	}
}

/**
 * @brief Update each target's correlation-filter model from an input batch.
 * @param[in] inputMeta A batch of input frames.
 */
void VisualTracker::update(const std::map<NvMOTStreamId, NvMOTFrame *> inputMeta)
{
	m_inputMeta = inputMeta;

	/// Check the input.
	checkInputOutput(inputMeta);

	/// Create a new optimal filter and update the moving-avg filter.
	cudaFilterUpdate(extractVisualFeatures(inputMeta, false));

	#if defined(VIZ_DEBUG)
		if(m_vizOutput != nullptr && isInitialized())
		{

			if(m_targetManager->getLastTargetInd() >= 0)
			{
				if(m_params.m_vizWriteAllTrajectoriesOnEveryFrame)
				{
					m_vizOutput->writeAllTrajectoriesOnEveryFrame(VIZ_OUTPUT_TYPES::TRACKER_STATES_VIZ);
				}

				if(m_params.m_vizWriteEachCompletedTrajectory)
				{
					m_vizOutput->writeEachCompletedTrajectory(VIZ_OUTPUT_TYPES::TRACKER_STATES_VIZ);
				}

				if(m_params.m_vizWriteBirdsEyeView)
				{
					m_vizOutput->writeBirdsEyeView(VIZ_OUTPUT_TYPES::TRACKER_STATES_VIZ);
				}

				if(m_params.m_vizWriteSegMask)
				{
					m_vizOutput->writeSrcImagePatchesWithMaskFilter(VIZ_OUTPUT_TYPES::TRACKER_STATES_VIZ);
				}
			}
		}
	#endif
}

/**
 * @brief Extract a batched visual feature vector for localization or model update.
 * @param[in] inputMeta A batch of input frames.
 * @param[in] isForLocalization Whether the operation is for localization rather than model update.
 * @return The device buffer containing the extracted feature vectors.
 */
float *VisualTracker::extractVisualFeatures(
	const std::map<NvMOTStreamId, NvMOTFrame*> &inputMeta, const bool isForLocalization
)
{
	bool doUpdateImages = false;

	int lastTargetInd = m_targetManager->getLastTargetInd();
	if(!isForLocalization && lastTargetInd < 0)
		return nullptr;

	/// Mark the tracker so it is not processed for this batch.
	/// And, Construct ROIs and offsets to the dst device pointers.
	/// Construct regions from the video streams present in the current batch.
	{
		/// {videoStreamID, a vector of {targetInd, bbox} }.
		m_featureRoisInBatch.clear();
		memset(m_trackerDoProcessing_h, false, (lastTargetInd+1)*sizeof(bool));
		memset(m_trackerAge_h, 0, (lastTargetInd+1)*sizeof(uint32_t));

		/// Update target states with no detector inputs.
		for(auto streamIt = inputMeta.begin(); streamIt != inputMeta.end(); ++streamIt)
		{
			const NvMOTStreamId &videoStreamId = streamIt->first;
			const NvMOTFrame *srcMotFrame = streamIt->second;
			const NvTrackedObjectList targetList = m_targetManager->getTrackedObjectsInUseList(videoStreamId);

			/// If this is the very first frame, then need to update the source frame in the feature extractor module.
			if(srcMotFrame->frameNum == 0
					|| isForLocalization == true) // If this is for localization, then it should be done for the new frame, so update the source frame.
			{
				doUpdateImages = true;
			}

			/// If detection is not done for the current frame, there is no need to update DCF, so leave feature regions, processing flags, and tracker ages unfilled.
			if ( (!srcMotFrame->objectsIn.detectionDone) && (!isForLocalization) )
			{
				continue;
			}

			for(auto targetIt = targetList.begin(); targetIt != targetList.end(); ++targetIt)
			{
				std::shared_ptr<NvTrackedObject> target = targetIt->second;
				const uint32_t targetInd = target->getTargetInd();

				/// Quasi-active targets: use current bbox as ROI (search region not set), skip filter update.
				if (target->getTrackerState() == TRACKER_STATE::QUASIACTIVE)
				{
					m_featureRoisInBatch[videoStreamId].push_back(std::make_pair(targetInd, target->getTargetBbox()));
					if (!isForLocalization)
					{
						m_trackerDoProcessing_h[targetInd] = false;
						m_trackerAge_h[targetInd] = target->getAge();
					}
				}
				else if(isForLocalization)
				{
					/// Get the ROI from which the visual features are to be extracted.
					m_featureRoisInBatch[videoStreamId].push_back(
						std::make_pair(targetInd, target->getPredictedSearchRegion())
					);
				}
				else
				{
					/// Get the ROI from which the visual features are to be extracted.
					m_featureRoisInBatch[videoStreamId].push_back(std::make_pair(targetInd, target->getSearchRegion()));

					/// Perform filter update only when there was a successful match with a detector.
					m_trackerDoProcessing_h[targetInd] = (target->getTargetInfo().matchedDetObjInfo != nullptr);

					m_trackerAge_h[targetInd] = target->getAge();
				}
			}
		}

		if(!isForLocalization)
		{
			/// Sync host and device.
			CUDA_CHECK(
				cudaMemcpyAsync(
					(uint32_t*)m_trackerAge_d, m_trackerAge_h,
					(lastTargetInd+1)*sizeof(uint32_t), cudaMemcpyHostToDevice, m_cudaStream
				)
			);
			CUDA_CHECK(
				cudaMemcpyAsync(
					m_trackerDoProcessing_d, m_trackerDoProcessing_h,
					(lastTargetInd+1)*sizeof(bool), cudaMemcpyHostToDevice, m_cudaStream
				)
			);
		}
	}

	/// Perform feature extraction on GPU in batch.
	/// Given the source image in CUDA Texture memory, generate a cropped and scaled image on device memory based on the input ROI and template size.
	/// If there are no ROIs in current batch, feature extraction returns false, return nullptr.
	if(m_featureExtractor.extractFeatures_VPI(inputMeta, m_featureRoisInBatch, lastTargetInd, doUpdateImages) == false)
	{
		return nullptr;
	}

	/// Make sure the target states are synced to device while performing VPI operations.
	CUDA_CHECK( cudaStreamSynchronize(m_cudaStream) );
	CUDA_CHECK( cudaPeekAtLastError() );

	/// Get the extracted features and Apply a masking window to all the image feature.
	/// Feature extraction keeps pitch-linear memory, and applying the 2D mask leaves memory that is no longer pitch-linear.
	{
		uint featPatchWidth, featPatchHeight, featChannelNum, featPitch, srcSubImgNum;
		uint2 featOutBatchingGrid;

		/// Get the extracted feature pointers and metadata on device.
		uchar *featureVecAtDevice = m_featureExtractor.getOutputFeaturePtrs_VPI(
			featPatchWidth, featPatchHeight, featOutBatchingGrid, featChannelNum, featPitch, srcSubImgNum
		);

		if(!((int)featPatchWidth == m_paramsRt.singleFeatureMatSize.width && (int)featPatchHeight == m_paramsRt.singleFeatureMatSize.height
				&& (int)featChannelNum == m_paramsRt.featureChannelSize
				&& (int)srcSubImgNum <= lastTargetInd+1
				))
		{
			printf(
				"[VisualTracker::extractVisualFeatures] Make sure ((int)featWidth [%d] == m_paramsRt.singleFeatureMatSize.width [%d] " "&& (int)featHeight [%d] == m_paramsRt.singleFeatureMatSize.height [%d]" "&& (int)featChannelNum [%d] == m_paramsRt.featureChannelSize [%d]" "&& (int)srcSubImgNum [%d] <= lastTargetInd+1 [%d])\n",
				(int)featPatchWidth, m_paramsRt.singleFeatureMatSize.width, (int)featPatchHeight,
				m_paramsRt.singleFeatureMatSize.height, (int)featChannelNum,
				m_paramsRt.featureChannelSize, (int)srcSubImgNum, lastTargetInd+1
			);

			throw std::runtime_error(
				"[VisualTracker::extractVisualFeatures] Make sure ((int)featWidth == m_paramsRt.singleFeatureMatSize.width && (int)featHeight == m_paramsRt.singleFeatureMatSize.height" "&& (int)featChannelNum == m_paramsRt.featureChannelSize" "&& (int)srcSubImgNum <= lastTargetInd+1)"
			);
		}

		if(m_featureExtractor.isVpiOutputInHighPrecision() == true)
		{
			/// Apply masking window to all the image features.
			cuDCF::applyFilter2D_U16(
				m_cudaStream, reinterpret_cast<const uint16_t*>(featureVecAtDevice),
				reinterpret_cast<const float*>(m_maskingWindow4Src), featPatchWidth, featPitch,
				featPatchHeight, featOutBatchingGrid, featChannelNum, srcSubImgNum, m_params.m_useColorNames,
				FeatureExtractor::NUM_CHANNELS_COLORNAMES, reinterpret_cast<float*>(m_work_I)
			);
		}
		else
		{
			/// Apply the masking window to all image features.
			cuDCF::applyFilter2D_U8(
				m_cudaStream, reinterpret_cast<const uchar*>(featureVecAtDevice),
				reinterpret_cast<const float*>(m_maskingWindow4Src), featPatchWidth, featPitch,
				featPatchHeight, featOutBatchingGrid, featChannelNum, srcSubImgNum, m_params.m_useColorNames,
				FeatureExtractor::NUM_CHANNELS_COLORNAMES, reinterpret_cast<float*>(m_work_I)
			);
		}

		return m_work_I;
	}
}

/**
 * @brief Create and update the DCF filter.
 * @param[in] ftrs The image-space feature vector.
 */
void VisualTracker::cudaFilterUpdate(const float *ftrs)
{
	if(ftrs == nullptr)
		return;

	uint lastTargetInd = m_targetManager->getLastTargetInd();
	uint cufftPlanId = getCufftPlanId(lastTargetInd+1);

	/// Store m_ftrs_F at half image size because of conjugate symmetry.
	CUFFT_CHECK(cufftExecR2C(m_cufftHandle_R2C_perCh[cufftPlanId], (cufftReal *)ftrs, (cufftComplex *)m_ftrs_F));

	/// Newly learned filter for this round.
	cuComplex *learned_filter_new = m_work_F_2;

	if (m_params.m_minConfidence4SegmentationFilter >= 0 && m_params.m_minConfidence4SegmentationFilter <= 1.0)
	{
		const SegmentationInBatch& segInfo = m_targetManager->getSegmentationBatchInfo();

		if (lastTargetInd >= 0)
		{
			memset( m_trackerUseMaskSeg_h, 0, (lastTargetInd+1)*sizeof(bool));

			for (auto it = m_featureRoisInBatch.begin(); it != m_featureRoisInBatch.end(); it++)
			{
				const std::map<uint32_t, std::shared_ptr<NvTrackedObject>> &targetList = m_targetManager->getTrackedObjectsInUseList(it->first);

				auto& featureRoisInBatchPeStream = it->second;
				for (auto jt = featureRoisInBatchPeStream.begin(); jt != featureRoisInBatchPeStream.end(); jt++)
				{
					const uint32_t targetInd = jt->first;
					const auto& roi = jt->second;
					std::shared_ptr<NvTrackedObject> target = targetList.at(targetInd);

					m_maskSegRois_h[targetInd].x = roi.min()(0) / m_paramsRt.srcImageSize.width * segInfo.size.width;
					m_maskSegRois_h[targetInd].y = roi.min()(1) / m_paramsRt.srcImageSize.height * segInfo.size.height;
					m_maskSegRois_h[targetInd].w = roi.sizes()(0) / m_paramsRt.srcImageSize.width * segInfo.size.width;
					m_maskSegRois_h[targetInd].z = roi.sizes()(1) / m_paramsRt.srcImageSize.height * segInfo.size.height;

					if (m_trackerDoProcessing_h[targetInd] && target->getTrackerConfidence() > m_params.m_minConfidence4SegmentationFilter)
					{
						m_trackerUseMaskSeg_h[targetInd] = true;
					}
				}
			}

			CUDA_CHECK(
				cudaMemcpyAsync(
					m_maskSegRois_d, m_maskSegRois_h, (lastTargetInd+1)*sizeof(float4),
					cudaMemcpyHostToDevice, m_cudaStream
				)
			);
			CUDA_CHECK(
				cudaMemcpyAsync(
					m_trackerUseMaskSeg_d, m_trackerUseMaskSeg_h,
					(lastTargetInd+1)*sizeof(bool), cudaMemcpyHostToDevice, m_cudaStream
				)
			);
			CUDA_CHECK(
				cudaMemcpyAsync(
					m_maskingSegWindow4Model, m_maskingWindow4Model,
					(lastTargetInd+1) * m_paramsRt.singleFeatureChannelElementSize * m_paramsRt.featureChannelSize * sizeof(float),
					cudaMemcpyDeviceToDevice, m_cudaStream
				)
			);

			int maskElementSize = segInfo.size.height * segInfo.size.width;
			cuDCF::createSegmentationFilterMask(
				m_cudaStream, segInfo.segMaskInBatch_d, segInfo.size.width, maskElementSize,
				m_paramsRt.singleFeatureMatSize.width, m_paramsRt.singleFeatureChannelElementSize,
				m_paramsRt.featureVecBatchElementSize, lastTargetInd+1, m_paramsRt.featureChannelSize,
				m_maskSegRois_d, m_trackerUseMaskSeg_d, m_maskingSegWindow4Model
			);
		}
	}

	/// Create the current optimal filter in learned_filter_new and update the moving-average filters in m_dcfFilter_F.
	cuDCF::createDcfFilter(
		m_cudaStream, m_cufftHandle_C2R_perCh[cufftPlanId], m_cufftHandle_R2C_perCh[cufftPlanId],
		reinterpret_cast<cuComplex*>(m_ftrs_F), reinterpret_cast<cuComplex*>(m_desiredResp_F),
		reinterpret_cast<float*>(m_maskingSegWindow4Model), m_paramsRt.singleFeatureMatSize.width,
		m_paramsRt.singleFeatureChannelElementSize, m_paramsRt.featureVecBatchElementSize,
		m_params.m_lambda, m_paramsRt.featureChannelSize, (lastTargetInd+1),
		/// Just to provide a work space.
		reinterpret_cast<cuComplex*>(m_work_F_1),
		/// Just to provide a work space.
		reinterpret_cast<float*>(m_work_I), m_trackerAge_d, m_trackerDoProcessing_d, m_params.m_filterLr,
		/// Newly learned filter for this batch round.
		reinterpret_cast<cuComplex*>(learned_filter_new),
		/// Moving-averaged filter.
		reinterpret_cast<cuComplex*>(m_dcfFilter_F)
	);

	/// Compute moving-average channel weights.
	cuDCF::computeChannelWeights(
		m_cudaStream, m_cufftHandle_C2R_perCh[cufftPlanId],
		reinterpret_cast<cuComplex*>(m_ftrs_F), m_paramsRt.singleFeatureChannelElementSize,
		m_paramsRt.singleFeatureMatSize.width, m_paramsRt.featureVecBatchElementSize,
		m_paramsRt.cudaKernelBatchSize, m_paramsRt.featureChannelSize, (lastTargetInd+1),
		/// Newly learned filter for this round.
		reinterpret_cast<cuComplex*>(learned_filter_new),
		/// Just to provide a work space.
		reinterpret_cast<cuComplex*>(m_work_F_1),
		/// Just to provide a work space.
		reinterpret_cast<float*>(m_work_I), m_trackerAge_d, m_trackerDoProcessing_d, m_params.m_filterChannelWeightsLr,
		/// Moving-averaged channel weights.
		m_channelWeights, m_channelWeightSum
	);

	if(m_params.m_debugVerboseLevel > 3)
	{
		#if defined(VIZ_DEBUG)

			CUDA_CHECK( cudaStreamSynchronize(m_cudaStream) );
			CUDA_CHECK( cudaPeekAtLastError() );

			printf("\n");
			for(int i=0; i<m_paramsRt.cudaKernelBatchSize; i++)
			{
				uint targetInd = i/m_paramsRt.featureChannelSize;
				if(!m_trackerDoProcessing_h[targetInd])
					continue;

				if(i%m_paramsRt.featureChannelSize == 0)
				{
					cout << "@@[VisualTracker] targetInd[" << targetInd << "]: weight_sum = " << m_channelWeightSum[targetInd] << ", when m_paramsRt.singleFeatureChannelElementSize = " << m_paramsRt.singleFeatureChannelElementSize << endl;
				}

				cout << "      @@[VisualTracker] channel_weights[" << i%m_paramsRt.featureChannelSize << "] = " << m_channelWeights[i] << endl;
			}

		#endif
	}

	/// Recompute correlation responses and confidence after the filter update to refresh the moving peak average.
	cuDCF::calcuateExpectedPeakCorrelation(
		m_cudaStream, m_paramsRt.featureChannelSize, (lastTargetInd+1), m_params.m_peakCorrLr,
		m_channelWeights, m_trackerAge_d, m_trackerDoProcessing_d, m_maxCorrResponse_ma_d
	);

	if(m_params.m_debugVerboseLevel > 3)
	{
		#if defined(VIZ_DEBUG)

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

				cout << "@@[VisualTracker] targetInd[" << targetInd << "]: expectedPeakCorr[" << targetInd << "] = " << expectedPeakCorr << ", "  << m_maxCorrResponse_ma_h[targetInd] << endl;
			}

		#endif
	}

}

#ifdef VIZ_DEBUG
/**
 * @brief Fetch batched device matrices as one host matrix per object.
 * @param[in] deviceData The device buffer containing batched matrices.
 * @param[in] multiChannelInput Whether each object contains multiple feature channels.
 */
std::vector<cv::Mat> VisualTracker::fetchBatchDataToHost_perObj(const float *deviceData, const bool multiChannelInput)
{
	int lastTargetInd = m_targetManager->getLastTargetInd();

	/// Get the cuComplex-type data from device to host, but only the real part.
	const int numElements = multiChannelInput ? (lastTargetInd+1)*m_paramsRt.singleFeatureChannelElementSize*m_paramsRt.featureChannelSize : \
		(lastTargetInd+1)*m_paramsRt.singleFeatureChannelElementSize;
	CUDA_CHECK(
		cudaMemcpy2DAsync ( reinterpret_cast<float*>(m_batchCudaArray_host), sizeof(float), reinterpret_cast<const float*>( deviceData ), sizeof(float), sizeof(float), numElements, cudaMemcpyDeviceToHost, m_cudaStream)
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
			const uint32_t index = multiChannelInput ? targetInd*m_paramsRt.singleFeatureChannelElementSize*m_paramsRt.featureChannelSize : \
				targetInd*m_paramsRt.singleFeatureChannelElementSize;

			if(target->getTrackerState() == TRACKER_STATE::EMPTY)
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
std::map<NvMOTStreamId, cv::Mat> VisualTracker::getSrcImagesAtHost()
{
	/// Get source image patches to Host.
	const std::map<NvMOTStreamId, cv::Mat> &srcImgPatches = m_featureExtractor.getSourceImagesAtHost();

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
#endif

/** @brief Construct the default configuration. */
VisualTrackerConfigParams::VisualTrackerConfigParams()
{
	m_showGpuTiming = false;
	m_debugVerboseLevel = 0;
	m_debugVerboseLevel_gpumemory = 0;
	m_debugVerboseLevel_cufft = 0;
	m_visualTrackerType = VISUAL_TRACKER_TYPES::NvDCF_VOT;

	// [Visual feature types]
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
	m_useCpuBackends4VPI = false;

	// [DCF][Core]
	m_filterLr = 0.075f;
	m_filterChannelWeightsLr = 0.1f;
	m_gaussianSigma = 0.75f;
	m_lambda = 5.f / 100.0f;
	m_peakCorrLr = 0.1f;
	m_minConfidence4SegmentationFilter = -0.1;

	// [FFT]
	m_cufftHandleQuantFactor = 7;

	// [Trajectory Visualization]
	m_generateTrackerVisualization = false;
	m_vizWriteSrcImgPatchesAndCorrResp = false;
	m_vizWriteBirdsEyeView = false;
	m_vizWriteSegMask = false;
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
VisualTrackerConfigParams::VisualTrackerConfigParams(const std::string &configFilePath, const std::string moduleName)
{
	YAML::Node configRoot = getConfigRoot(configFilePath);
	this->read(configRoot[moduleName]);
}

/**
 * @brief Construct configuration parameters from a YAML node.
 * @param[in] fn Parsed YAML structure containing the configuration.
 */
VisualTrackerConfigParams::VisualTrackerConfigParams(const YAML::Node &fn)
{
	this->read(fn);
}

/**
 * @brief Read visual tracker parameters from a YAML node.
 * @param[in] fn Parsed YAML structure of configuration parameters.
 */
void VisualTrackerConfigParams::read(const YAML::Node &fn)
{
	if(!m_isInit)
	{
		/// Set the default values.
		*this = VisualTrackerConfigParams();
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
		else if (paramKey == "useCpuBackends4VPI")
		{
			/// Only support setting this parameter on startup.
			if(!m_isInit)
				m_useCpuBackends4VPI = (int)fit->second.as<float>();
			else
				CLOG(std::string("!!![WARNING] m_useCpuBackends4VPI is not supported for dynamic updates").c_str());
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
				CLOG(std::string("!!![WARNING] m_useCpuBackends4VPI is not supported for dynamic updates").c_str());
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
		else if (paramKey == "minConfidence4SegmentationFilter")
		{
			m_minConfidence4SegmentationFilter = fit->second.as<float>();
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
						else if(paramKey_s == "vizWriteSegMask")
						{
							m_vizWriteSegMask = (int)fjt->second.as<float>();
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
							cout << "!! [WARNING][VisualTrackerConfigParams][Visualization] Unknown param found: " << paramKey_s << endl;
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
			cout << "!! [WARNING][VisualTrackerConfigParams] Unknown param found: " << paramKey << endl;
		}
	}

	/// Set init flag after initialization to prevent updates from changing.
	m_isInit = true;

	if(m_debugVerboseLevel > 1)
	{
		cout << "[VisualTracker] visualTrackerType = " << m_visualTrackerType << endl;

	}
}
