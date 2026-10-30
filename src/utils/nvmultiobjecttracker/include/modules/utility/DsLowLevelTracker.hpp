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

#ifndef CV_FUNCSLIB_DSLOWLEVELTRACKER_HPP_
#define CV_FUNCSLIB_DSLOWLEVELTRACKER_HPP_

#include <iostream>
#include <map>

#include "modules/NvMultiObjectTracker/nvdstracker.h"

/// Maps each stream ID to tuples of bounding-box index, bounding box, class ID, and detection confidence.
typedef std::map<NvMOTStreamId, std::vector<std::tuple<int, cv::Rect2f, uint, float > > > RoiBatch;

/// Maps each stream ID to a frame number and source image.
typedef std::map<NvMOTStreamId, std::pair<uint32_t, cv::Mat> > FrameBatch;

/** @brief Adapts test frame and ROI batches to the low-level tracker API. */
class DsLowLevelTracker
{
public:
	/** @brief Initialize empty low-level tracker input and output storage. */
	DsLowLevelTracker()
	{
		/// Reset the MOT context.
		mp_MotCtx = nullptr;
		m_motOutputObjectsMeta.list = nullptr;
		m_motInputParams.frameList = nullptr;

		m_computeType = NVMOTCOMP_ANY;

		m_isIntialized = false;
	}

	/** @brief Deinitialize the tracker context and release allocated metadata. */
	~DsLowLevelTracker()
	{
		/// Deinitialize NvMOT.
		if(mp_MotCtx != nullptr)
			NvMOT_DeInit(mp_MotCtx);

		/// De-allocate the output to the tracker.
		deAllocateMotOutputData();

		/// De-allocate the input to the tracker.
		deAllocateMotInputData();
	}

	/** @brief Report whether the low-level tracker is initialized. */
	bool isInitialized() { return m_isIntialized; }

	/**
	 * @brief Allocate tracker metadata and initialize a low-level tracking context.
	 * @param[in] gpuId GPU identifier used by the tracker.
	 * @param[in] maxVideoStreams Maximum number of video streams in output storage.
	 * @param[in] numVideoStreams Number of video streams in input storage.
	 * @param[in] maxObjectsPerFrame Maximum number of objects allocated per frame.
	 * @param[in] numTransforms Number of input transforms per frame.
	 * @param[in] maxWidth Maximum transformed frame width.
	 * @param[in] maxHeight Maximum transformed frame height.
	 * @param[in] maxPitch Maximum transformed frame row pitch.
	 * @param[in] srcImgColorFormat Source image color format.
	 * @param[in] lowLevelTrackerConfigFilePath Path to the low-level tracker configuration file.
	 * @param[in] debugVerboseLevel Debug verbosity level.
	 */
	void initialize(
		const uint32_t gpuId, const int maxVideoStreams, const int numVideoStreams,
		const int maxObjectsPerFrame, const int numTransforms, const uint maxWidth,
		const uint maxHeight, const uint maxPitch, const NvBufSurfaceColorFormat srcImgColorFormat,
		std::string lowLevelTrackerConfigFilePath = "", const int debugVerboseLevel = 0
	)
	{
		/// Allocate mot output for current frame data.
		allocateMotOutputData(maxVideoStreams, maxObjectsPerFrame);

		/// Allocate MOT input data.
		allocateMotInputData(numTransforms, numVideoStreams, maxObjectsPerFrame);

		m_srcImgColorFormat = srcImgColorFormat;

		m_debugVerboseLevel = debugVerboseLevel;

		/// Configure tracker component limits.
		{
			m_motMiscConfig.gpuId = gpuId;
			/// Leave the per-stream target limit unrestricted.
			m_motMiscConfig.maxObjPerStream = 0;
			/// Leave the per-batch target limit unrestricted.
			m_motMiscConfig.maxObjPerBatch = 0;
		}

		/// Initialize the low-level tracker.
		initializeLowLeveTracker(
			maxWidth, maxHeight, maxPitch, m_srcImgColorFormat,
			numVideoStreams, numTransforms, lowLevelTrackerConfigFilePath
		);

		m_isIntialized = true;
	}

	/**
	 * @brief Process a batch of frames and return the current tracker output.
	 * @param[in] framesInBatch Source images and frame numbers keyed by stream ID.
	 * @param[in] numTransforms Number of input transforms per frame.
	 * @param[in] rois Detection regions keyed by stream ID.
	 * @param[in] isDetectionDone Indicates whether detection completed for this batch.
	 * @param[in] frameBatchNo Frame-batch number used when batch-based frame IDs are enabled.
	 */
	const NvMOTTrackedObjBatch &update(
		const FrameBatch &framesInBatch, const int numTransforms, RoiBatch rois = RoiBatch(),
		const bool isDetectionDone = false, const uint32_t frameBatchNo = 0
	)
	{
		m_motOutputObjectsMeta.numFilled = 0;

		/// Construct the input data.
		if(generateMotInputParams(framesInBatch, numTransforms, rois, isDetectionDone, frameBatchNo) == true)
		{
			/// Process a frame.
			NvMOTStatus nvtstatus = NvMOT_Process(mp_MotCtx, &m_motInputParams, &m_motOutputObjectsMeta);

			if (NvMOTStatus_OK != nvtstatus)
			{
				std::cerr << "NvMOT_Process() returned error " << nvtstatus << std::endl;
			}
		}

		return m_motOutputObjectsMeta;
	}

	/**
	 * @brief Remove a stream from the low-level tracker context.
	 * @param[in] videoStreamID Video stream ID to remove.
	 */
	const void removeStream(const NvMOTStreamId videoStreamID)
	{
		NvMOT_RemoveStreams(mp_MotCtx, videoStreamID);
	}

	/** @brief Return input frame metadata keyed by video stream ID. */
	const std::map<NvMOTStreamId, NvMOTFrame *> getMotInputParams()
	{
		std::map<NvMOTStreamId, NvMOTFrame *> inputParam;

		for(uint i=0; i<m_motInputParams.numFrames; i++)
		{
			NvMOTFrame *frame = &m_motInputParams.frameList[i];

			inputParam[frame->streamID] = frame;
		}

		return inputParam;
	}

protected:

	/**
	 * @brief Recreate the low-level tracker context for configured transforms and streams.
	 * @param[in] maxWidth Maximum transformed frame width.
	 * @param[in] maxHeight Maximum transformed frame height.
	 * @param[in] maxPitch Maximum transformed frame row pitch.
	 * @param[in] colorFormat Transformed frame color format.
	 * @param[in] maxStream Maximum number of video streams.
	 * @param[in] numTransforms Number of input transforms per frame.
	 * @param[in] lowLevelTrackerConfigFilePath Path to the low-level tracker configuration file.
	 */
	void initializeLowLeveTracker(
		uint maxWidth, uint maxHeight, uint maxPitch, NvBufSurfaceColorFormat colorFormat,
		uint maxStream, uint numTransforms, std::string lowLevelTrackerConfigFilePath = ""
	)
	{
		NvMOTPerTransformBatchConfig perTransformBatchConfig;
		perTransformBatchConfig.bufferType = NVBUF_MEM_CUDA_UNIFIED;
		perTransformBatchConfig.maxWidth = maxWidth;
		perTransformBatchConfig.maxHeight = maxHeight;
		perTransformBatchConfig.maxPitch = maxPitch;
		/// Store the allocation size in bytes.
		perTransformBatchConfig.maxSize = maxPitch * maxHeight;
		perTransformBatchConfig.colorFormat = colorFormat;

		/// Configure the NvMOT context.
		NvMOTConfig motLlLibConfig;
		motLlLibConfig.maxStreams = maxStream;
		motLlLibConfig.numTransforms = numTransforms;
		motLlLibConfig.computeConfig = m_computeType;
		motLlLibConfig.miscConfig = m_motMiscConfig;
		motLlLibConfig.perTransformBatchConfig = new NvMOTPerTransformBatchConfig[motLlLibConfig.numTransforms];
		for(uint i=0; i<motLlLibConfig.numTransforms; i++)
		{
			motLlLibConfig.perTransformBatchConfig[i] = perTransformBatchConfig;
		}

		motLlLibConfig.customConfigFilePathSize = lowLevelTrackerConfigFilePath.size();
		motLlLibConfig.customConfigFilePath = new char[motLlLibConfig.customConfigFilePathSize+1];
		strncpy(
			motLlLibConfig.customConfigFilePath, (char *)(lowLevelTrackerConfigFilePath.c_str()),
			motLlLibConfig.customConfigFilePathSize
		);
		motLlLibConfig.customConfigFilePath[motLlLibConfig.customConfigFilePathSize] = '\0';

		/// Deinitialize NvMOT.
		if(mp_MotCtx != nullptr)
			NvMOT_DeInit(mp_MotCtx);

		/// Initialize NvMOT.
		NvMOTStatus nvtstatus = NvMOT_Init(&motLlLibConfig, &mp_MotCtx, &m_motConfigResponse);

		assert(mp_MotCtx != nullptr);

		/// Release temporary configuration storage.
		delete motLlLibConfig.perTransformBatchConfig;
		delete motLlLibConfig.customConfigFilePath;

		if (NvMOTStatus_OK != nvtstatus)
		{
			std::cerr << "NvMOTInit returned error " << nvtstatus << std::endl;
		}
	}

	/**
	 * @brief Allocate current-frame tracker output storage.
	 * @param[in] maxVideoStreams Maximum number of video streams.
	 * @param[in] maxObjectsPerFrame Maximum number of tracked objects per frame.
	 */
	void allocateMotOutputData(const int maxVideoStreams, const int maxObjectsPerFrame)
	{
		if(m_motOutputObjectsMeta.list != nullptr)
		{
			deAllocateMotOutputData();
		}

		/// This should be equal to or larger than (# of targets)*(max length of shadow tracking)*(# of video streams).
		m_motOutputObjectsMeta.list = new NvMOTTrackedObjList[maxVideoStreams];

		m_motOutputObjectsMeta.numAllocated = maxVideoStreams;
		m_motOutputObjectsMeta.numFilled = 0;

		for(int i=0; i<maxVideoStreams; i++)
		{
			NvMOTTrackedObjList *objList = &m_motOutputObjectsMeta.list[m_motOutputObjectsMeta.numFilled++];

			objList->streamID = i;
			objList->frameNum = 0;
			objList->valid = false;
			objList->list = new NvMOTTrackedObj[maxObjectsPerFrame];
			objList->numAllocated = maxObjectsPerFrame;
			objList->numFilled = 0;
		}
	}

	/** @brief Release current-frame tracker output storage. */
	void deAllocateMotOutputData()
	{
		for(uint i=0; i<m_motOutputObjectsMeta.numAllocated; i++)
		{
			delete [] m_motOutputObjectsMeta.list[i].list;
		}

		if(m_motOutputObjectsMeta.list != nullptr)
		{
			delete [] m_motOutputObjectsMeta.list;

			m_motOutputObjectsMeta.list = nullptr;
		}
	}

	/**
	 * @brief Allocate low-level tracker input storage.
	 * @param[in] numTransforms Number of input transforms per frame.
	 * @param[in] numVideoStreams Number of video streams.
	 * @param[in] maxObjectsPerFrame Maximum number of objects per frame.
	 */
	void allocateMotInputData(const int numTransforms, const int numVideoStreams, const int maxObjectsPerFrame)
	{
		if(m_motInputParams.frameList != nullptr)
		{
			deAllocateMotInputData();
		}

		m_motInputParams.numFrames = numVideoStreams;
		m_motInputParams.frameList = new NvMOTFrame[numVideoStreams];

		for(int i=0; i<numVideoStreams; i++)
		{
			NvMOTFrame *newFrame = &m_motInputParams.frameList[i];
			newFrame->streamID = i;
			newFrame->frameNum = 0;
			newFrame->timeMeta = NvMOTTimeMeta{};
			newFrame->doTracking = false;
			newFrame->reset = true;

			newFrame->numBuffers = numTransforms;
			newFrame->bufferList = new NvBufSurfaceParams*;

			for(int i=0; i<numTransforms; i++)
			{
				newFrame->bufferList[i] = new NvBufSurfaceParams;
			}

			newFrame->objectsIn.list = new NvMOTObjToTrack[maxObjectsPerFrame];
			newFrame->objectsIn.numAllocated = maxObjectsPerFrame;
			newFrame->objectsIn.numFilled = 0;
		}
	}

	/** @brief Release low-level tracker input storage. */
	void deAllocateMotInputData()
	{
		for(uint i=0; i<m_motInputParams.numFrames; i++)
		{
			delete [] m_motInputParams.frameList[i].objectsIn.list;
			delete [] m_motInputParams.frameList[i].bufferList;
		}

		if(m_motInputParams.frameList != nullptr)
		{
			delete [] m_motInputParams.frameList;

			m_motInputParams.frameList = nullptr;
		}
	}

	/**
	 * @brief Populate low-level tracker input metadata from frame and ROI batches.
	 * @param[in] framesInBatch Source images and frame numbers keyed by stream ID.
	 * @param[in] numTransforms Number of input transforms per frame.
	 * @param[in] rois Detection regions keyed by stream ID.
	 * @param[in] isDetectionDone Indicates whether detection completed for this batch.
	 * @param[in] frameBatchNo Frame-batch number used when batch-based frame IDs are enabled.
	 * @return True when at least one input frame is populated.
	 */
	bool generateMotInputParams(
		const FrameBatch &framesInBatch, const int numTransforms, RoiBatch &rois,
		const bool isDetectionDone, const uint32_t frameBatchNo = 0
	)
	{
		m_motInputParams.numFrames = 0;

		for(auto it = framesInBatch.begin(); it != framesInBatch.end(); ++it)
		{
			NvMOTStreamId videoStreamId = it->first;
			const uint32_t frameNumPerStream = it->second.first;
			const uint32_t resolvedFrameNum = m_useBatchNumForFrameId ? frameBatchNo : frameNumPerStream;
			const cv::Mat cvFrame = it->second.second;

			assert(cvFrame.type() == CV_8UC1 || cvFrame.type() == CV_8UC4);

			if(rois.find(videoStreamId) == rois.end())
				continue;

			const std::vector<std::tuple<int, cv::Rect2f, uint, float > > &roisPerStream = rois.at(videoStreamId);

			NvMOTFrame *frame = &m_motInputParams.frameList[m_motInputParams.numFrames++];

			frame->streamID = videoStreamId;
			frame->frameNum = resolvedFrameNum;
			frame->sensorName = nullptr;
			frame->timeMeta.frameNum = resolvedFrameNum;
			frame->timeMeta.frameTimeStamp = 0;
			frame->timeMeta.frameTimeValid = false;
			frame->timeMeta.localTimestamp = Time().getUnixTimestampNanoSeconds();

			if(isDetectionDone)
			{
				frame->doTracking = true;
			}
			else
			{
				frame->doTracking = false;
			}

			if(resolvedFrameNum == 0)
			{
				frame->reset = true;
			}
			else
			{
				frame->reset = false;
			}

			frame->numBuffers = numTransforms;

			for(int j=0; j<numTransforms; j++)
			{
				NvBufSurfaceParams *transform = frame->bufferList[j];
				/// # of bytes.
				transform->dataSize = cvFrame.step*cvFrame.rows;
				transform->width = cvFrame.cols;
				transform->pitch = cvFrame.step;
				transform->colorFormat = m_srcImgColorFormat;

				transform->dataPtr = (void *)cvFrame.data;
				transform->mappedAddr.addr[0] = (void *)cvFrame.data;

				if(m_srcImgColorFormat == NVBUF_COLOR_FORMAT_NV12)
				{
					/// In case of NV12 image, the input cv::Mat image would have 1.5x height in CV_8UC1 format.
					transform->height = cvFrame.rows/1.5;

					/// There are two planes: Y-ch and UV-ch.
					transform->planeParams.num_planes = 2;

					transform->planeParams.width[0] = transform->width;
					transform->planeParams.height[0] = transform->height;
					transform->planeParams.pitch[0] = transform->pitch;
					transform->planeParams.offset[0] = 0;
					transform->planeParams.psize[0] = transform->pitch * transform->planeParams.height[0];
					/// Y-ch in NV12 is uchar.
					transform->planeParams.bytesPerPix[0] = 1;

					transform->planeParams.width[1] = transform->width;
					/// UV-ch have half of the height of Y-ch.
					transform->planeParams.height[1] = transform->height/2;
					transform->planeParams.pitch[1] = transform->pitch;
					/// UV-ch would start after Y-ch.
					transform->planeParams.offset[1] = transform->planeParams.psize[0];
					transform->planeParams.psize[1] = transform->pitch * transform->planeParams.height[1];
					/// UV-ch in NV12 is uchar.
					transform->planeParams.bytesPerPix[1] = 1;

				}
				else if(m_srcImgColorFormat == NVBUF_COLOR_FORMAT_RGBA)
				{
					transform->height = cvFrame.rows;

					/// There is only one plane where RGBA pixels are interleaved.
					transform->planeParams.num_planes = 1;

					transform->planeParams.width[0] = transform->width;
					transform->planeParams.height[0] = transform->height;
					transform->planeParams.pitch[0] = transform->pitch;
					transform->planeParams.offset[0] = 0;
					transform->planeParams.psize[0] = transform->pitch * transform->planeParams.height[0];
					/// RGBA is in uchar4 format.
					transform->planeParams.bytesPerPix[0] = 4;
				}
				else
				{
					std::runtime_error("Invalid color format!");
				}
			}

			NvMOTObjToTrackList *objs = &frame->objectsIn;

			if(objs->numAllocated < roisPerStream.size())
			{
				printf(
					"(objs->numAllocated, roisPerStream.size() = (%d, %d)\n",
					objs->numAllocated, (int)roisPerStream.size()
				);
				assert(objs->numAllocated > roisPerStream.size());
			}

			objs->numFilled = 0;
			objs->detectionDone = isDetectionDone;

			if(isDetectionDone && roisPerStream.size() > 0)
			{
				for(uint i=0; i<roisPerStream.size(); i++, objs->numFilled++)
				{
					/// {bboxInd, bbox, classId, detectionConfidence}.
					int classId = std::get<2>(roisPerStream[i]);
					cv::Rect2f bbox = std::get<1>(roisPerStream[i]);
					float detectionConfidence = std::get<3>(roisPerStream[i]);

					objs->list[i].classId = classId;
					objs->list[i].confidence = detectionConfidence;

					objs->list[i].doTracking = true;

					objs->list[i].bbox.x = bbox.x;
					objs->list[i].bbox.y = bbox.y;
					objs->list[i].bbox.width = bbox.width;
					objs->list[i].bbox.height = bbox.height;

				}
			}

			if(m_debugVerboseLevel > 0)
				printf(
					"[DsLowLevelTracker::generateMotInputParams()] videoStreamId[%lx], frameNum[%d]: feeding %d objects \n",
					videoStreamId, resolvedFrameNum, objs->numFilled
				);
		}

		return m_motInputParams.numFrames > 0;
	}

	NvMOTTrackedObjBatch m_motOutputObjectsMeta; //!< Current-frame output from the tracker.
	NvMOTProcessParams m_motInputParams; //!< Input batch submitted to the tracker.
	NvMOTMiscConfig m_motMiscConfig; //!< Miscellaneous tracker configuration.
	NvMOTContext *mp_MotCtx; //!< Low-level tracker context.
	NvMOTConfigResponse m_motConfigResponse; //!< Response to the tracker configuration request.
	NvMOTCompute m_computeType; //!< Requested tracker compute target.
	NvBufSurfaceColorFormat m_srcImgColorFormat; //!< Source image color format.
	int m_debugVerboseLevel; //!< Debug verbosity level.
	bool m_isIntialized; //!< Indicates whether the low-level tracker is initialized.
	bool m_useBatchNumForFrameId = false; //!< Selects frame-batch numbers instead of per-stream frame numbers.
};

#endif // CV_FUNCSLIB_DSLOWLEVELTRACKER_HPP_
