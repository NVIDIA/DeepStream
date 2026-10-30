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

#if defined VIZ_DEBUG_REID || defined VIZ_DEBUG_SEG || defined VIZ_DEBUG
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/stacktrace.hpp>
#endif

#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#include "modules/utility/cuda_common.hpp"
#include "modules/utility/dataTypes.hpp"
#include "modules/NvMultiObjectTracker/context.hpp"
#include "modules/NvMultiObjectTracker/localizer_factory.hpp"

using namespace std;

/**
 * @brief Initialize a tracking context from the negotiated configuration.
 * @param[in] config Tracker configuration supplied by the client.
 * @param[out] configResponse Configuration response populated with initialization status.
 */
NvMOTContext::NvMOTContext(const NvMOTConfig &config, NvMOTConfigResponse& configResponse)
{
	/// Make a deep-copy of the config.
	{
		/// Copy the non-pointer type values.
		m_Config = config;

		m_Config.perTransformBatchConfig = new NvMOTPerTransformBatchConfig;
		memcpy(m_Config.perTransformBatchConfig, config.perTransformBatchConfig, sizeof(NvMOTPerTransformBatchConfig));

		m_Config.customConfigFilePath = new char[config.customConfigFilePathSize+1];
		strncpy(m_Config.customConfigFilePath, config.customConfigFilePath, config.customConfigFilePathSize);
		m_Config.customConfigFilePath[config.customConfigFilePathSize] = '\0';
	}

	try
	{
		/// Set CUDA device.
		{
			int deviceCount = 0;
			cudaGetDeviceCount(&deviceCount);
			if((int)m_Config.miscConfig.gpuId < deviceCount)
			{
				CUDA_CHECK( cudaSetDevice(m_Config.miscConfig.gpuId) );
				CUDA_CHECK( cudaPeekAtLastError() );
			}
			else
			{
				throw std::runtime_error(
					"[NvMOTContext::NvMOTContext()] invalid CUDA device chosen: " + std::to_string(m_Config.miscConfig.gpuId) + " !!"
				);
			}
		}

		/// FIXME: This assignment hardcodes success before the status check.
		configResponse.summaryStatus = NvMOTConfigStatus_OK;

		if (NvMOTConfigStatus_OK != configResponse.summaryStatus)
			throw std::invalid_argument("Failed to configure context");

		/// Instantiate an appropriate localizer.
		m_pLocalizer = LocalizerFactory::getInstance().makeLocalizer(m_Config.customConfigFilePath);

		/// Configure the maximum number of supported streams.
		reinterpret_cast<NvMultiObjectTrackerBase*>(m_pLocalizer.get())->setMaxStreams(m_Config.maxStreams);
		/// Set max # of texture addresses to be supported.
		reinterpret_cast<NvMultiObjectTrackerBase*>(m_pLocalizer.get())->setMaxBufSurfAddrSize(
			m_Config.maxBufSurfAddrSize
		);

		/// Initialize MOT tracker.
		if(!m_pLocalizer->isInitialized())
		{
			uint32_t frameHeight = m_Config.perTransformBatchConfig[0].maxHeight;
			uint32_t frameWidth = m_Config.perTransformBatchConfig[0].maxWidth;
			uint32_t framePitch = m_Config.perTransformBatchConfig[0].maxPitch;

			// Adjust one-channel NV12 input height for stand-alone and unit-test layouts.
			#if  defined(STANDALONE_APP) || defined(UNIT_TESTS)
				if(reinterpret_cast<NvMultiObjectTrackerBase*>(m_pLocalizer.get())->getColorFormatType() == NVBUF_COLOR_FORMAT_NV12)
				{
					frameHeight = m_Config.perTransformBatchConfig[0].maxHeight/1.5f;
				}
			#endif

			m_pLocalizer->initialize(NvMotDataTypes::Size(frameWidth, frameHeight), framePitch);
		}
	}
	catch (exception& e)
	{
		cout << "\n!![ERROR] " << e.what() << '\n';

		std::exception_ptr pExcept = std::current_exception();
		std::rethrow_exception (pExcept);
	}
	catch (...)
	{
		cout << "\n!![ERROR] Default Exception\n";
	}
}

/** @brief Release the batch localizer owned by the context. */
NvMOTContext::~NvMOTContext()
{
	/// The m_Config was deep-copied, so its memory needs to be deallocated.
	delete m_Config.perTransformBatchConfig;
	delete [] m_Config.customConfigFilePath;

}

/**
 * @brief Populate tracker capability and memory requirement fields.
 * @param[out] paramsQuery Query structure populated with tracker requirements.
 */
NvMOTStatus NvMOTContext::queryParams(NvMOTQuery &paramsQuery)
{
	m_pLocalizer->queryParams(paramsQuery);

	/// Set the expected color format of the input source images.
	reinterpret_cast<NvMultiObjectTrackerBase*>(m_pLocalizer.get())->setColorFormatType(paramsQuery.colorFormats[0]);
	m_Config.computeConfig = paramsQuery.computeConfig;
	m_Config.numTransforms = paramsQuery.numTransforms;

	return NvMOTStatus_OK;
}

/**
 * @brief Process a batch of frames.
 * @param[in] params Parameters for the frames to process.
 * @param[out] pTrackedObjectsBatch Object tracks produced for the batch.
 */
NvMOTStatus NvMOTContext::processFrame(const NvMOTProcessParams *params, NvMOTTrackedObjBatch *pTrackedObjectsBatch)
{
	#ifdef VIZ_DEBUG
		cudaProfilerStart();
	#endif
	try
	{
		if(params->numFrames == 0)
		{
			throw std::runtime_error("[NvMOTContext::processFrame()] params->numFrames == 0 !!");
		}
		else if(params->numFrames > m_Config.maxStreams)
		{
			CLOG(
				std::string( "!!![WARNING] params->numFrames = " + std::to_string(params->numFrames) + ", m_Config.maxStreams = " + std::to_string(m_Config.maxStreams) ).c_str()
			);

			throw std::runtime_error("[NvMOTContext::processFrame()] params->numFrames > m_Config.maxStreams !!");
		}

		/// Make sure the input frame is valid according to the MOT Config used to create this context.
		if (m_Config.numTransforms > 1)
		{
			throw std::runtime_error("[NvMOTContext::processFrame()] numTransforms can only be 0 or 1 !!");
		}
		else if (m_Config.numTransforms == 1)
		{
			for(uint streamInd=0; streamInd<params->numFrames; streamInd++)
			{
				NvMOTFrame *motFrame = &params->frameList[streamInd];
				if (motFrame->numBuffers != 1)
				{
					throw std::runtime_error(
						"[NvMOTContext::processFrame()] motFrame->numBuffers must be 1 when numTransforms is 1 !!"
					);

				}
				if((*motFrame->bufferList)->width > m_Config.perTransformBatchConfig->maxWidth)
				{
					throw std::runtime_error(
						"[NvMOTContext::processFrame()] motFrame->bufferList[i]->width > m_Config.perTransformBatchConfig[i].maxWidth "
					);
				}
				if((*motFrame->bufferList)->height > m_Config.perTransformBatchConfig->maxHeight)
				{
					throw std::runtime_error(
						"[NvMOTContext::processFrame()] motFrame->bufferList[i]->height > m_Config.perTransformBatchConfig[i].maxHeight"
					);
				}
				if((*motFrame->bufferList)->pitch > m_Config.perTransformBatchConfig->maxPitch)
				{
					throw std::runtime_error(
						"[NvMOTContext::processFrame()] motFrame->bufferList[i]->pitch > m_Config.perTransformBatchConfig[i].maxPitch"
					);
				}
				if((*motFrame->bufferList)->colorFormat != reinterpret_cast<NvMultiObjectTrackerBase*>(m_pLocalizer.get())->getColorFormatType())
				{
					throw std::runtime_error(
						"[NvMOTContext::processFrame()] motFrame->bufferList[i]->colorFormat != m_Config.perTransformBatchConfig[i].colorFormat"
					);
				}
			}
		}

		/// Set CUDA device.
		CUDA_CHECK( cudaSetDevice(m_Config.miscConfig.gpuId) );
		CUDA_CHECK( cudaPeekAtLastError() );

		/// Construct the mot input frames.
		std::map<NvMOTStreamId, NvMOTFrame*> nvFramesInBatch;
		for(NvMOTStreamId streamInd=0; streamInd<params->numFrames; streamInd++)
		{
			NvMOTFrame *motFrame = &params->frameList[streamInd];
			nvFramesInBatch[motFrame->streamID] = motFrame;
		}

		if(nvFramesInBatch.size() > 0)
		{
			/// Perform update and construct the output data inside.
			m_pLocalizer->update(nvFramesInBatch, pTrackedObjectsBatch);
		}

	}
	catch (std::exception& e)
	{
		cout << "\n!![Exception] " << e.what() << '\n';
		#if defined VIZ_DEBUG_REID || defined VIZ_DEBUG_SEG || defined VIZ_DEBUG
			cout << boost::stacktrace::stacktrace() << std::endl;
		#endif
		std::exception_ptr pExcept = std::current_exception();
		std::rethrow_exception (pExcept);
	}
	catch (...)
	{
		cout << "\n!![ERROR] Default Exception\n";
	}
	#ifdef VIZ_DEBUG
		cudaProfilerStop();
	#endif
	return NvMOTStatus_OK;
}

/**
 * @brief Output miscellaneous past-frame data when it is available.
 * @param[in] params Parameters for the frames to process.
 * @param[out] pTrackerMiscData Miscellaneous tracker output.
 */
NvMOTStatus NvMOTContext::retrieveMiscData(const NvMOTProcessParams *params, NvMOTTrackerMiscData *pTrackerMiscData)
{
	try
	{
		std::set<NvMOTStreamId> videoStreamIdList;

		/// Indicate which streams to fetch past-frame data for.
		for(NvMOTStreamId streamInd=0; streamInd<params->numFrames; streamInd++)
		{
			videoStreamIdList.insert(params->frameList[streamInd].streamID);
		}

		if (pTrackerMiscData && pTrackerMiscData->pPastFrameObjBatch)
		{
			m_pLocalizer->outputPastFrameObjs(videoStreamIdList, pTrackerMiscData->pPastFrameObjBatch);
		}

		if (pTrackerMiscData && pTrackerMiscData->pTerminatedTrackBatch)
		{
			m_pLocalizer->outputTerminatedObjs(videoStreamIdList, pTrackerMiscData->pTerminatedTrackBatch);
		}

		if(pTrackerMiscData && pTrackerMiscData->pShadowTrackBatch)
		{
			m_pLocalizer->outputShadowObj(videoStreamIdList, pTrackerMiscData->pShadowTrackBatch);
		}

	}
	catch (std::exception& e)
	{
		cout << "\n!![Exception] " << e.what() << '\n';
		std::exception_ptr pExcept = std::current_exception();
		std::rethrow_exception (pExcept);
	}
	catch (...)
	{
		cout << "\n!![ERROR] Default Exception\n";
	}

	return  NvMOTStatus_OK;
}

/**
 * @brief Terminate trackers and release resources for streams selected by a mask.
 * @param[in] streamIdMask Stream ID mask selecting streams to remove.
 */
NvMOTStatus NvMOTContext::removeStream(const NvMOTStreamId streamIdMask)
{
	try
	{
		m_pLocalizer->deleteRemovedStreamTrackers(streamIdMask);
	}
	catch (std::exception& e)
	{
		cout << "\n!![Exception] " << e.what() << '\n';
		std::exception_ptr pExcept = std::current_exception();
		std::rethrow_exception (pExcept);
	}
	catch (...)
	{
		cout << "\n!![ERROR] Default Exception\n";
	}
	return  NvMOTStatus_OK;
}

/**
 * @brief Update the subset of tracker parameters that supports dynamic configuration.
 * @param[in] configStr Stringified YAML configuration parameters to update.
 */
NvMOTStatus NvMOTContext::updateParams(const std::string &configStr)
{
	try
	{
		m_pLocalizer->updateParams(configStr);
	}
	catch (std::exception& e)
	{
		cout << "\n!![Exception] " << e.what() << '\n';
		std::exception_ptr pExcept = std::current_exception();
		std::rethrow_exception (pExcept);
	}
	catch (...)
	{
		cout << "\n!![ERROR] Default Exception\n";
	}
	return  NvMOTStatus_OK;
}
