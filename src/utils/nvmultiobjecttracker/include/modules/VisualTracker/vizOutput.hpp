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

#ifndef VISUALTRACKERLIB_VIZOUTPUT_HPP_
#define VISUALTRACKERLIB_VIZOUTPUT_HPP_

#include <memory>
#include <deque>
#include <map>
#include <iostream>
#include <iomanip>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/core/eigen.hpp>
#include "modules/utility/FrameOutput.hpp"
#include "modules/NvMultiObjectTracker/NvTrackedObject.hpp"
#include "modules/VisualTracker/VisualTracker.hpp"
#include "modules/VisualTracker/DCFTracker.hpp"

enum VIZ_OUTPUT_TYPES : uint
{
	TRACKER_STATES_VIZ = 1, //!< Visualize the tracker states.
	TRAJECTORY_VIZ = 2 //!< Visualize the trajectories.
};

/** @brief Renders and writes visual tracker debugging output. */
class TrackerVizOutput : public FrameOutput
{
	/** @brief Stores an image and its output naming metadata until a target ID is available. */
	struct WriteBuffer_t
	{
		const cv::Mat img; //!< Image waiting to be written.
		const uint frameNum; //!< Frame number associated with the image.
		const uint outputDirectoryKey; //!< Output directory selector.
		const std::string subFolderName; //!< Output subfolder name.
		const std::string fileNamePrefix; //!< Output filename prefix.

		/**
		 * @brief Construct a buffered image output entry.
		 * @param[in] img_ The image to buffer.
		 * @param[in] frameNum_ The associated frame number.
		 * @param[in] outputDirectoryKey_ The output directory selector.
		 * @param[in] subFolderName_ The output subfolder name.
		 * @param[in] fileNamePrefix_ The output filename prefix.
		 */
		WriteBuffer_t(
			const cv::Mat &img_, uint frameNum_, uint outputDirectoryKey_,
			const std::string &subFolderName_, const std::string &fileNamePrefix_
		)
			: img(img_), frameNum(frameNum_), outputDirectoryKey(outputDirectoryKey_),
			  subFolderName(subFolderName_), fileNamePrefix(fileNamePrefix_)
		{
		}

		/**
		 * @brief Copy a buffered image output entry.
		 */
		WriteBuffer_t(const WriteBuffer_t&) = default;

		/**
		 * @brief Copy buffered image output metadata into this entry.
		 */
		WriteBuffer_t& operator=(const WriteBuffer_t&) = default;

		/**
		 * @brief Move a buffered image output entry.
		 */
		WriteBuffer_t(WriteBuffer_t&&) noexcept = default;

		/**
		 * @brief Move buffered image output metadata into this entry.
		 */
		WriteBuffer_t& operator=(WriteBuffer_t&&) noexcept = default;
	};

protected:
	VisualTracker *m_visualTracker = NULL; //!< CUDA visual tracker that supplies visualization data.
	DCFTracker *m_dcfTracker = NULL; //!< VPI DCF tracker that supplies visualization data.
	const std::shared_ptr<NvTrackedObjectManager> m_targets; //!< Manager that supplies tracked objects.

	// [Configuration parameters]
	const int m_displayWidth = 1920; //!< Display/screen width.
	const int m_srcImgWH = 240; //!< Visualization width and height for source image patches.
	const int m_featImgWH = 160; //!< Visualization width and height for feature images.
	std::map<uint32_t, std::deque<WriteBuffer_t>> m_writeBuffer; //!< Buffered outputs grouped by target index.
	std::map<NvMOTStreamId, uint64_t> m_lastFrameMatchedInTrajectoryArchive; //!< The target ID of the latest trajectory archived.
	std::map<NvMOTStreamId, cv::Mat> m_lastFrames; //!< The last source frame kept.

public:
	/**
	 * @brief Construct output rendering for a CUDA visual tracker.
	 * @param[in] visTracker The CUDA visual tracker that supplies visualization data.
	 * @param[in] targets The manager that supplies tracked objects.
	 */
	TrackerVizOutput(VisualTracker *visTracker, const std::shared_ptr<NvTrackedObjectManager> targets)
	: m_visualTracker(visTracker), m_targets(targets)
	{
	}

	/**
	 * @brief Construct output rendering for a VPI DCF tracker.
	 * @param[in] dcfTracker The VPI DCF tracker that supplies visualization data.
	 * @param[in] targets The manager that supplies tracked objects.
	 */
	TrackerVizOutput(DCFTracker *dcfTracker, const std::shared_ptr<NvTrackedObjectManager> targets)
	: m_dcfTracker(dcfTracker), m_targets(targets)
	{
	}

	/** @brief Destroy the visualization output writer. */
	~TrackerVizOutput() {}

	/**
	 * @brief Generate visually distinct BGR colors by distributing shuffled hues.
	 * @param[in] nb The number of colors to generate.
	 * @return A single-row BGR matrix containing the generated colors.
	 */
	cv::Mat make_dist_colors(unsigned nb)
	{
		/// Compute step to cover the whole hue range in OpenCV.
		float step = 180.0f / nb;
		std::vector<int> hues(nb);
		/// Generate hue values.
		for (size_t i = 0; i < nb; i++) {
			hues[i] = i * step;
		}
		/// Shuffle the hue values randomly.
		std::random_device rd;
		std::mt19937 g(1);
		std::shuffle(hues.begin(), hues.end(), g);
		/// Build the HSV matrix with shuffled hues.
		/// High saturation and value for vivid colors.
		cv::Mat hsv(1, nb, CV_8UC3, cv::Scalar(0, 255, 255));
		for (size_t i = 0; i < nb; i++) {
			auto &color = hsv.at<cv::Vec3b>(0, i);
			color[0] = hues[i];
		}
		cv::Mat rgb;
		cv::cvtColor(hsv, rgb, cv::COLOR_HSV2BGR);
		return rgb;
	}

	/**
	 * @brief Render and write a bird's-eye view for each available stream.
	 * @param[in] outputDirectoryKey The output directory selector.
	 */
	void writeBirdsEyeView(const uint outputDirectoryKey = 0)
	{
		/// {videoStreamID, src image frame}.
		std::map<NvMOTStreamId, cv::Mat> framesBgr;
		/// {videoStreamID, world map coordinates}.
		static std::map<NvMOTStreamId, cv::Rect> worldMapRegions;
		/// {videoStreamID, world-to-frame scale}.
		std::map<NvMOTStreamId, cv::Point2d> worldMapsScale;
		const float worldMapMargin = 40;
		cv::Point frameSize;
		if(m_visualTracker)
			/// Both frame dimensions use the source height.
			frameSize = cv::Point(
				m_visualTracker->m_paramsRt.srcImageSize.height, m_visualTracker->m_paramsRt.srcImageSize.height
			);
		else
			/// Both frame dimensions use the source height.
			frameSize = cv::Point(
				m_dcfTracker->m_paramsRt.srcImageSize.height, m_dcfTracker->m_paramsRt.srcImageSize.height
			);

		const std::map<NvMOTStreamId, std::shared_ptr<NvTrackedObjectList>>& targetLists = m_targets->getTrackedObjectsInUseList();

		/// Define the world map size.
		for(auto it = targetLists.begin(); it != targetLists.end(); ++it)
		{
			const NvMOTStreamId &videoStreamID = it->first;

			if(m_targets->m_objModelProjectors.find(videoStreamID) == m_targets->m_objModelProjectors.end()
					|| !m_targets->m_objModelProjectors.at(videoStreamID)->isInitialized())
			{
				continue;
			}

			framesBgr[videoStreamID] = cv::Mat::zeros(frameSize,CV_8UC3);

			if(worldMapRegions.find(videoStreamID) == worldMapRegions.end())
			{
				worldMapRegions[videoStreamID] = cv::Rect(0,0,0,0);
			}

			cv::Rect &worldMapRegion = worldMapRegions[videoStreamID];

			if(targetLists.find(videoStreamID) != targetLists.end())
			{
				const NvTrackedObjectList &targetList = *targetLists.at(videoStreamID);

				if(targetList.size() > 0)
				{
					for(auto targetIt = targetList.begin(); targetIt != targetList.end(); ++targetIt)
					{
						std::shared_ptr<NvTrackedObject> tracker = targetIt->second;

						const std::list<TargetInfo> trajectory = tracker->getTrajectoryBuffer();

						if(trajectory.size() == 0)
						{
							continue;
						}

						/// If the last target state is not Active, then don't draw.
						{
							auto pt = trajectory.rbegin();

							if(!((*pt).trackerState == TRACKER_STATE::ACTIVE
									|| ((*pt).trackerState == TRACKER_STATE::TENTATIVE && (*pt).matchedDetObjInfo != nullptr)))
							{
								continue;
							}
						}

						for(auto pt = trajectory.begin(); pt != trajectory.end(); ++pt)
						{
							cv::Rect2f tBbox = CV_FUNCS::convertMotRect2f2CvRect2f((*pt).tBbox);
							cv::Point bboxCenter = (tBbox.tl()+tBbox.br())*0.5;
							Eigen::Vector2d eigenVector = m_targets->m_objModelProjectors.at(videoStreamID)->getWorldFeetPtOnGroundFromCamWaistPt(Eigen::Vector2d(bboxCenter.x, bboxCenter.y), (*pt).classId);
							cv::Point2d pt_world_feet(eigenVector(0), eigenVector(1));

							/// Check if a point in world coordinate is within the world map size.
							/// If not, adjust world map size to include the new point.
							if(worldMapRegion.width == 0 || worldMapRegion.x > pt_world_feet.x-worldMapMargin)
							{
								worldMapRegion.x = pt_world_feet.x-worldMapMargin;
							}
							if(worldMapRegion.width == 0 || worldMapRegion.x+worldMapRegion.width < pt_world_feet.x+worldMapMargin)
							{
								worldMapRegion.width = pt_world_feet.x+worldMapMargin-worldMapRegion.x;
							}
							if(worldMapRegion.height == 0 || worldMapRegion.y > pt_world_feet.y-worldMapMargin)
							{
								worldMapRegion.y = pt_world_feet.y-worldMapMargin;
							}
							if(worldMapRegion.height == 0 || worldMapRegion.y+worldMapRegion.height < pt_world_feet.y+worldMapMargin)
							{
								worldMapRegion.height = pt_world_feet.y+worldMapMargin-worldMapRegion.y;
							}

						}
					}
				}
			}

			const double mapWidth = std::max(worldMapRegion.width, worldMapRegion.height);

			worldMapsScale[videoStreamID] = cv::Point2d(frameSize.x/mapWidth, frameSize.y/mapWidth);

		}

		/// Param to be configured.
		const int radius = 3;

		/// Draw the trajectories of each stream on its bird's-eye view frame.
		for(auto it = targetLists.begin(); it != targetLists.end(); ++it)
		{
			const NvMOTStreamId &videoStreamID = it->first;

			if(worldMapRegions.find(videoStreamID) == worldMapRegions.end()
					|| framesBgr.find(videoStreamID) == framesBgr.end()
					|| worldMapsScale.find(videoStreamID) == worldMapsScale.end())
				continue;

			const cv::Rect worldMapRegion = worldMapRegions.at(videoStreamID);
			const cv::Point2d worldMapScale = worldMapsScale.at(videoStreamID);
			const cv::Point2d mapCenter(
				worldMapRegion.x+worldMapRegion.width/2, worldMapRegion.y+worldMapRegion.height/2
			);
			cv::Mat frame = framesBgr.at(videoStreamID);
			const double mapWidth = std::max(worldMapRegion.width, worldMapRegion.height);

			if(targetLists.find(videoStreamID) != targetLists.end())
			{
				const NvTrackedObjectList &targetList = *targetLists.at(videoStreamID);

				if(targetList.size() > 0)
				{
					/// Show frame number.
					int frameNum = targetList.begin()->second->getTargetInfo().frameNum;
					cv::putText(
						frame, std::string("Frame: " + std::to_string(frameNum)),
						cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,0,255), 2, 8
					);

					for(auto targetIt = targetList.begin(); targetIt != targetList.end(); ++targetIt)
					{
						std::shared_ptr<NvTrackedObject> tracker = targetIt->second;

						const std::list<TargetInfo> trajectory_ = tracker->getTrajectoryBuffer();
						std::list<TargetInfo> trajectory = trajectory_;

						if(trajectory.size() == 0)
						{
							continue;
						}

						/// If the last target state is not Active, then don't draw.
						{
							auto pt = trajectory.rbegin();

							if(!((*pt).trackerState == TRACKER_STATE::ACTIVE
									|| ((*pt).trackerState == TRACKER_STATE::TENTATIVE && (*pt).matchedDetObjInfo != nullptr)))
							{
								continue;
							}
						}

						uint32_t targetID = tracker->getTargetID();
						uint32_t targetAge = tracker->getAge();
						cv::Rect2f lastBbox = CV_FUNCS::convertMotRect2f2CvRect2f(trajectory.rbegin()->tBbox);
						cv::Point bboxCenter_last = (lastBbox.tl()+lastBbox.br())*0.5;
						Eigen::Vector2d eigenVector = m_targets->m_objModelProjectors.at(videoStreamID)->getWorldFeetPtOnGroundFromCamWaistPt(Eigen::Vector2d(bboxCenter_last.x, bboxCenter_last.y), tracker->getClassId());
						cv::Point2d pt_world_feet_last(eigenVector(0), eigenVector(1));
						cv::Point pt_map_last_(
							(pt_world_feet_last.x-mapCenter.x+mapWidth/2)*worldMapScale.x,
							(pt_world_feet_last.y-mapCenter.y+mapWidth/2)*worldMapScale.y
						);

						/// Make the image coordinate axis align with the world coordinate axis in terms of orientations.
						/// The origin of the image coordinate is at the top-left, while the world coordinate origin is at the bottom-right.
						cv::Point pt_map_last(frameSize.x - pt_map_last_.x, frameSize.y - pt_map_last_.y);

						const cv::Scalar targetColor = tracker->getColor();

						uint i=0;
						for(auto pt = trajectory.begin(); pt != trajectory.end(); ++pt, ++i)
						{
							cv::Rect2f tBbox = CV_FUNCS::convertMotRect2f2CvRect2f((*pt).tBbox);
							cv::Point bboxCenter = (tBbox.tl()+tBbox.br())*0.5;
							Eigen::Vector2d eigenVector = m_targets->m_objModelProjectors.at(videoStreamID)->getWorldFeetPtOnGroundFromCamWaistPt(Eigen::Vector2d(bboxCenter.x, bboxCenter.y), (*pt).classId);
							cv::Point2d pt_world_feet(eigenVector(0), eigenVector(1));
							cv::Point pt_map_(
								(pt_world_feet.x-mapCenter.x+mapWidth/2)*worldMapScale.x,
								(pt_world_feet.y-mapCenter.y+mapWidth/2)*worldMapScale.y
							);

							/// Make the image coordinate axis align with the world coordinate axis in terms of orientations.
							/// The origin of the image coordinate is at the top-left, while the world coordinate origin is at the bottom-right.
							cv::Point pt_map(frameSize.x - pt_map_.x, frameSize.y - pt_map_.y);

							float fading = std::max((float)0, (i*1.5f)/((float)trajectory.size()-1));
							cv::Scalar color = targetColor*fading + cv::Scalar(100,100,100)*(1.f - fading);

							if(!((*pt).trackerState == TRACKER_STATE::ACTIVE
									|| ((*pt).trackerState == TRACKER_STATE::TENTATIVE && (*pt).matchedDetObjInfo != nullptr)))
							{
								color = cv::Scalar(50,50,50);
							}

							if(std::next(pt) == trajectory.end())
							{
								/// Draw the bbox center.
								cv::circle(frame, pt_map, radius, color, -1);
								cv::circle(frame, pt_map, radius+2, CV_RGB(255,0,0), 2);
							}
							else
							{
								cv::circle(frame, pt_map, radius, color, -1);
							}

							/// Show target label (ID & Age).
							std::string label = std::to_string(targetID) + "(" + std::to_string(targetAge) + ")";
							cv::putText(
								frame, label, pt_map_last + cv::Point(5,-5),
								cv::FONT_HERSHEY_SIMPLEX, 0.6, targetColor, 2, 8
							);
						}
					}

					std::ostringstream imgName;

					imgName	<< "BirdsEye_StreamID[" << std::hex << videoStreamID << "]";
					imgName	<< "_Frame[" << std::dec << std::setfill('0') << std::setw(10) << frameNum << "]";

					writeImageOutput(frame, imgName.str(), outputDirectoryKey);
				}
			}
		}
	}

	/**
	 * @brief Draw all current trajectories on each available source frame and write the results.
	 * @param[in] outputDirectoryKey The output directory selector.
	 */
	void writeAllTrajectoriesOnEveryFrame(const uint outputDirectoryKey = 0)
	{
		bool showRecentTrajectory = true;
		bool showCurrentDetBbox = true;
		bool showCurrentTargetBbox = true;
		bool showPrediction = false;
		bool showPredictionUncertainty = false;

		bool showHumanModelCylinder = true;
		bool showTrajectoryOnFeet = true;
		bool showPeerTrajectory = true;

		/// Get source image frames available at Host (made available only when m_params.m_doTrajectoryAnalysis is enabled).
		/// {videoStreamID, src image frame}.
		std::map<NvMOTStreamId, cv::Mat> srcImgFramesBgr;

		/// Get source image patches to Host.
		if(m_visualTracker)
			srcImgFramesBgr = m_visualTracker->getSrcImagesAtHost();
		else
			srcImgFramesBgr = m_dcfTracker->getSrcImagesAtHost();

		if(srcImgFramesBgr.size() == 0)
		{
			CLOG(std::string( "!!![WARNING] No video stream exist in srcImgFramesBgr" ).c_str());

			return;
		}

		const std::map<NvMOTStreamId, std::shared_ptr<NvTrackedObjectList>>& targetLists = m_targets->getTrackedObjectsInUseList();

		/// Write all the recent trajectories on the source frame.
		for(auto it = srcImgFramesBgr.begin(); it != srcImgFramesBgr.end(); ++it)
		{
			const NvMOTStreamId &videoStreamID = it->first;

			if(targetLists.find(videoStreamID) == targetLists.end())
				continue;

			cv::Mat frame = it->second.clone();

			const NvTrackedObjectList &targetList = *targetLists.at(videoStreamID);

			if(targetList.size() == 0)
				continue;

			/// Show frame number.
			int frameNum = targetList.begin()->second->getTargetInfo().frameNum;
			cv::putText(
				frame, std::string("Frame: " + std::to_string(frameNum)),
				cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,0,255), 2, 8
			);

			std::map<TargetId, cv::Scalar> targetColorList;
			for(auto targetIt = targetList.begin(); targetIt != targetList.end(); ++targetIt)
			{
				std::shared_ptr<NvTrackedObject> target = targetIt->second;
				targetColorList[target->getTargetID()] = target->getColor();
				/// Draw recent trajectories.
				if(showRecentTrajectory)
				{
					drawRecentTrajectory(
						target, frame, showCurrentDetBbox, showCurrentTargetBbox,
						showPrediction, showPredictionUncertainty, showTrajectoryOnFeet
					);
				}

				/// Draw projected human models on the camera frame.
				if(showHumanModelCylinder
						&& m_targets->m_objModelProjectors.find(videoStreamID) != m_targets->m_objModelProjectors.end()
						&& m_targets->m_objModelProjectors.at(videoStreamID)->isInitialized())
				{
					drawProjectedTargetModel(target, frame);
				}
			}

			if (showPeerTrajectory)
			{
				const PeerTargetDB peerTargetDB = m_targets->getPeerTargets(videoStreamID);
				drawProjectedPeerTrajectory(peerTargetDB, frame, videoStreamID, frameNum, targetColorList);
			}

			std::ostringstream imgName;

			imgName	<< "TrajPerFrame_StreamID[" << std::hex << videoStreamID << "]";
			imgName	<< "_Frame[" << std::dec << std::setfill('0') << std::setw(10) << frameNum << "]";

			writeImageOutput(frame, imgName.str(), outputDirectoryKey);
		}
	}

	/**
	 * @brief Draw and write each trajectory after its track completes.
	 * @param[in] outputDirectoryKey The output directory selector.
	 */
	void writeEachCompletedTrajectory(const uint outputDirectoryKey = 0)
	{
		/// Get source image frames available at Host (made available only when m_params.m_doTrajectoryAnalysis is enabled).
		/// {videoStreamID, src image frame}.
		std::map<NvMOTStreamId, cv::Mat> srcImgFramesBgr;

		/// Get source image patches to Host.
		if(m_visualTracker)
			srcImgFramesBgr = m_visualTracker->getSrcImagesAtHost();
		else
			srcImgFramesBgr = m_dcfTracker->getSrcImagesAtHost();

		if(srcImgFramesBgr.size() == 0)
		{
			CLOG(std::string( "!!![WARNING] No video stream exist in srcImgFramesBgr" ).c_str());

			return;
		}

		/// Keep the latest copy of the source frame.
		for(auto it = srcImgFramesBgr.begin(); it != srcImgFramesBgr.end(); ++it)
		{
			const NvMOTStreamId &videoStreamID = it->first;

			m_lastFrames[videoStreamID] = it->second.clone();
		}

		/// Write a complete trajectory of a target when terminated.
		for(auto it = m_lastFrames.begin(); it != m_lastFrames.end(); ++it)
		{
			const NvMOTStreamId &videoStreamID = it->first;

			if(m_targets->m_trajectoryManager->m_trajectoryArchive.find(videoStreamID) == m_targets->m_trajectoryManager->m_trajectoryArchive.end())
				continue;

			/// {targetId, Trajectory4Archive}.
			for(auto trajIt = m_targets->m_trajectoryManager->m_trajectoryArchive.at(videoStreamID)->begin(); trajIt != m_targets->m_trajectoryManager->m_trajectoryArchive.at(videoStreamID)->end(); ++trajIt)
			{
				const uint64_t targetId = trajIt->first & 0x0000ffff;
				const std::list<TargetInfo> &tracklet = trajIt->second.tracklet;
				bool &isVisualized = trajIt->second.isVisualized;

				if(!isVisualized)
				{
					cv::Mat frame = it->second.clone();

					/// Write the processed tracklet for visualization.
					drawFullTrajectory(targetId, tracklet, frame);

					std::ostringstream imgName;

					imgName	<< "CompletedTraj_StreamID[" << std::hex << videoStreamID << "]";
					imgName	<< "_TargetID[" << std::dec << targetId << "]_Frame[" << tracklet.begin()->frameNum
							<< "->"<< tracklet.rbegin()->frameNum << "]";

					writeImageOutput(frame, imgName.str(), outputDirectoryKey);

					std::cout << "[VizOutput] Wrote archived tracklet of TargetID[" << targetId << "]" << std::endl;

					isVisualized = true;
				}
			}
		}
	}

	/**
	 * @brief Write source image patches alongside their correlation-response visualizations.
	 * @param[in] outputDirectoryKey The output directory selector.
	 */
	void writeSrcImagePatchesWithCorrResp(const uint outputDirectoryKey = 0)
	{
		/// Get source image patches to Host.
		std::map<NvMOTStreamId, std::map<uint, cv::Mat > > srcImgPatches;
		std::vector<cv::Mat> corrRespMats;
		int m_vizDebugStreamId;
		if(m_visualTracker)
		{
			srcImgPatches = m_visualTracker->m_featureExtractor.getSrcImgPatches();
			corrRespMats = m_visualTracker->fetchBatchDataToHost_perObj(m_visualTracker->m_corrResponses_d);
			m_vizDebugStreamId = m_visualTracker->m_params.m_vizDebugStreamId;
		}
		else
		{
			srcImgPatches = m_dcfTracker->getImagePatches2Host_OCV_batch();
			corrRespMats = m_dcfTracker->fetchBatchDataToHost_perObj(m_dcfTracker->m_corrResponses_d);
			m_vizDebugStreamId = m_dcfTracker->m_params.m_vizDebugStreamId;
		}

		for(auto it = m_targets->getTrackedObjectsInUseList().begin(); it != m_targets->getTrackedObjectsInUseList().end(); ++it)
		{
			NvMOTStreamId streamId = it->first;

			if(m_vizDebugStreamId < 0 || m_vizDebugStreamId == (int)(streamId>>32))
			{
				for(auto jt = it->second->begin(); jt != it->second->end(); ++jt)
				{
					uint trackerInd = jt->first;

					writeSrcImagePatchesWithCorrResp_perTarget(
						streamId, trackerInd, outputDirectoryKey, srcImgPatches[streamId][trackerInd].clone(),
						prepareCorrelationResponseImg(streamId, trackerInd, corrRespMats, cv::Size(m_srcImgWH,m_srcImgWH))
					);
				}
			}
		}
	}

	/**
	 * @brief Write source image patches alongside their segmentation-filter masks.
	 * @param[in] outputDirectoryKey The output directory selector.
	 */
	void writeSrcImagePatchesWithMaskFilter(const uint outputDirectoryKey = 0)
	{
		/// Get source image patches to Host.
		std::map<NvMOTStreamId, std::map<uint, cv::Mat > > srcImgPatches;
		std::vector<cv::Mat> maskMats;
		int m_vizDebugStreamId;
		if(m_visualTracker)
		{
			srcImgPatches = m_visualTracker->m_featureExtractor.getSrcImgPatches();
			maskMats = m_visualTracker->fetchBatchDataToHost_perObj(m_visualTracker->m_maskingSegWindow4Model, true);
			m_vizDebugStreamId = m_visualTracker->m_params.m_vizDebugStreamId;
		}
		else
		{
			std::cout << "Mask visualization is not supported for VPI NvDCF yet" << std::endl;
			return;
		}

		for(auto it = m_targets->getTrackedObjectsInUseList().begin(); it != m_targets->getTrackedObjectsInUseList().end(); ++it)
		{
			NvMOTStreamId streamId = it->first;

			if(m_vizDebugStreamId < 0 || m_vizDebugStreamId == (int)(streamId>>32))
			{
				for(auto jt = it->second->begin(); jt != it->second->end(); ++jt)
				{
					uint trackerInd = jt->first;

					writeSrcImagePatchesWithMaskFilter_perTarget(
						streamId, trackerInd, outputDirectoryKey, srcImgPatches[streamId][trackerInd].clone(),
						prepareMaskFilterImg(streamId, trackerInd, maskMats, cv::Size(m_srcImgWH,m_srcImgWH))
					);
				}
			}
		}
	}

protected:

	/**
	 * @brief Buffer a source patch paired with its correlation-response image for one target.
	 * @param[in] streamId_ The video stream identifier.
	 * @param[in] trackerInd The target's tracker index.
	 * @param[in] outputDirectoryKey The output directory selector.
	 * @param[in] srcImgPatch The source image patch.
	 * @param[in] corrRespMat The correlation-response visualization.
	 */
	void writeSrcImagePatchesWithCorrResp_perTarget(
		const NvMOTStreamId streamId_, const int trackerInd,
		const uint outputDirectoryKey, cv::Mat srcImgPatch, cv::Mat corrRespMat
	)
	{
		const std::shared_ptr<NvTrackedObject> target = m_targets->getTrackedObjectsInUseList(streamId_).at(trackerInd);

		if(	target->getTrackerState() == TRACKER_STATE::EMPTY)
			return;

		int frameNum = target->getTargetInfo().frameNum;
		int streamId = streamId_ >> 32;

		assert(srcImgPatch.size().width > 0 && srcImgPatch.size().height > 0);

		if(m_visualTracker && m_visualTracker->m_paramsRt.colorFormat == NvBufSurfaceColorFormat::NVBUF_COLOR_FORMAT_RGBA)
		{
			cvtColor(srcImgPatch, srcImgPatch, cv::COLOR_RGBA2BGR);
		}
		cv::resize(srcImgPatch, srcImgPatch, cv::Size(m_srcImgWH,m_srcImgWH));

		assert(srcImgPatch.type() == corrRespMat.type());

		/// Stack the two.
		cv::Mat fusedImg(cv::Size(m_srcImgWH*2,m_srcImgWH), srcImgPatch.type());
		srcImgPatch.copyTo(fusedImg(cv::Rect(0,0,m_srcImgWH,m_srcImgWH)));
		corrRespMat.copyTo(fusedImg(cv::Rect(m_srcImgWH,0,m_srcImgWH,m_srcImgWH)));

		/// Put it into a buffer first because if target state is TENTATIVE, there is no ID assigned yet.
		addToWriteBuffer(
			target, fusedImg, frameNum, outputDirectoryKey,
			std::string("Stream_" + std::to_string((int)streamId)), "CorrResp_"
		);

		/// Write into file if ID is assigned.
		writeFromWriteBuffer(target);
	}

	/**
	 * @brief Buffer a source patch paired with its segmentation-filter mask for one target.
	 * @param[in] streamId_ The video stream identifier.
	 * @param[in] trackerInd The target's tracker index.
	 * @param[in] outputDirectoryKey The output directory selector.
	 * @param[in] srcImgPatch The source image patch.
	 * @param[in] maskMat The segmentation-filter mask visualization.
	 */
	void writeSrcImagePatchesWithMaskFilter_perTarget(
		const NvMOTStreamId streamId_, const int trackerInd,
		const uint outputDirectoryKey, cv::Mat srcImgPatch, cv::Mat maskMat
	)
	{
		const std::shared_ptr<NvTrackedObject> target = m_targets->getTrackedObjectsInUseList(streamId_).at(trackerInd);

		if(	target->getTrackerState() == TRACKER_STATE::EMPTY)
			return;

		int frameNum = target->getTargetInfo().frameNum;
		int streamId = streamId_ >> 32;

		assert(srcImgPatch.size().width > 0 && srcImgPatch.size().height > 0);

		if(m_visualTracker && m_visualTracker->m_paramsRt.colorFormat == NvBufSurfaceColorFormat::NVBUF_COLOR_FORMAT_RGBA)
		{
			cvtColor(srcImgPatch, srcImgPatch, cv::COLOR_RGBA2BGR);
		}
		cv::resize(srcImgPatch, srcImgPatch, cv::Size(m_srcImgWH,m_srcImgWH));

		assert(srcImgPatch.type() == maskMat.type());

		/// Stack the two.
		cv::Mat fusedImg(cv::Size(m_srcImgWH*2,m_srcImgWH), srcImgPatch.type());
		srcImgPatch.copyTo(fusedImg(cv::Rect(0,0,m_srcImgWH,m_srcImgWH)));
		maskMat.copyTo(fusedImg(cv::Rect(m_srcImgWH,0,m_srcImgWH,m_srcImgWH)));

		/// Put it into a buffer first because if target state is TENTATIVE, there is no ID assigned yet.
		addToWriteBuffer(
			target, fusedImg, frameNum, outputDirectoryKey,
			std::string("Stream_" + std::to_string((int)streamId)), "mask_"
		);

		/// Write into file if ID is assigned.
		writeFromWriteBuffer(target);
	}

	/**
	 * @brief Add an image to a target's pending output buffer.
	 * @param[in] target The tracked object that owns the output.
	 * @param[in] img The image to buffer.
	 * @param[in] frameNum The associated frame number.
	 * @param[in] outputDirectoryKey The output directory selector.
	 * @param[in] subFolderName The output subfolder name.
	 * @param[in] fileNamePrefix The output filename prefix.
	 */
	void addToWriteBuffer(
		const std::shared_ptr<NvTrackedObject> target, const cv::Mat &img, const uint frameNum,
		const uint outputDirectoryKey, const std::string subFolderName, const std::string fileNamePrefix = ""
	)
	{
		const uint targetInd = target->getTargetInd();
		const uint32_t targetAge = target->getAge();

		/// Clear any existing ones.
		if(targetAge == 1)
			m_writeBuffer[targetInd].clear();

		WriteBuffer_t buf(img, frameNum, outputDirectoryKey, subFolderName, fileNamePrefix);

		m_writeBuffer[targetInd].push_back(std::move(buf));
	}

	/**
	 * @brief Write and clear pending images after a target receives a persistent ID.
	 * @param[in] target The tracked object whose pending images are written.
	 */
	void writeFromWriteBuffer(const std::shared_ptr<NvTrackedObject> target)
	{
		if(target->getTrackerState() == TRACKER_STATE::ACTIVE
				|| target->getTrackerState() == TRACKER_STATE::INACTIVE
				)
		{
			const uint targetInd = target->getTargetInd();
			uint64_t targetID = target->getTargetID();

			if(m_writeBuffer[targetInd].size() > 0)
			{
				for(auto it = m_writeBuffer[targetInd].begin(); it != m_writeBuffer[targetInd].end(); ++it)
				{
					const WriteBuffer_t &buf = *it;

					/// FileName: Target_{targetID}_Frame_{FrameNum}.
					std::string outFileName = buf.fileNamePrefix + "Target_" + std::to_string((int)targetID) + "_Frame_" + std::to_string(buf.frameNum);

					/// Out folder name: Stream_{StreamID}.
					writeImageOutput(buf.img, outFileName, buf.outputDirectoryKey, buf.subFolderName);
				}

				m_writeBuffer[targetInd].clear();
			}
		}
	}

	/**
	 * @brief Prepare a color-mapped correlation response with optional target and detection markers.
	 * @param[in] streamId_ The video stream identifier.
	 * @param[in] trackerInd The target's tracker index.
	 * @param[in] corrRespMats The correlation-response matrices for the tracker batch.
	 * @param[in] dstSize The requested output image size.
	 * @param[in] showPeakPoint Whether to draw the localized peak.
	 * @param[in] showDetBboxes Whether to draw nearby detector bounding-box centers.
	 * @return The prepared BGR correlation-response image.
	 */
	cv::Mat prepareCorrelationResponseImg(
		const NvMOTStreamId streamId_, const uint trackerInd, std::vector<cv::Mat> &corrRespMats,
		const cv::Size dstSize, const bool showPeakPoint = true, const bool showDetBboxes = true
	)
	{
		const NvTrackedObjectList &targetList = m_targets->getTrackedObjectsInUseList(streamId_);

		if(targetList.size() == 0)
		{
			return cv::Mat::zeros(corrRespMats[trackerInd].size(), corrRespMats[trackerInd].type());
		}

		const std::shared_ptr<NvTrackedObject> target = targetList.at(trackerInd);

		if(target->getTrackerState() == TRACKER_STATE::EMPTY)
		{
			return cv::Mat::zeros(corrRespMats[trackerInd].size(), corrRespMats[trackerInd].type());
		}

		assert(corrRespMats[trackerInd].size().width > 0 && corrRespMats[trackerInd].size().height > 0);

		cv::Mat response_viz_gray, response_viz_rgb;

		/// Normalize to a high-enough magnitude for better visualization.
		cv::normalize(
			corrRespMats[trackerInd], response_viz_gray,
			corrRespMats[trackerInd].cols*corrRespMats[trackerInd].rows, 0, cv::NORM_L2
		);

		/// Wrap around with the circular shifting.
		response_viz_gray = CV_FUNCS::circshift(
			response_viz_gray, cvFloor(response_viz_gray.cols/2), cvFloor(response_viz_gray.rows/2)
		);
		response_viz_gray.convertTo(response_viz_gray, CV_8UC1);

		cv::resize(response_viz_gray, response_viz_gray, dstSize);

		float wScale = dstSize.width / (float)corrRespMats[trackerInd].size().width;
		float hScale = dstSize.height / (float)corrRespMats[trackerInd].size().height;

		cv::applyColorMap(response_viz_gray, response_viz_rgb, cv::COLORMAP_JET);

		double maxVal;
		cv::Point maxLoc;
		cv::minMaxLoc(response_viz_gray, NULL, &maxVal, NULL, &maxLoc);

		/// Draw the peak point.
		if(showPeakPoint)
		{
			cv::Point2f loc;
			if(m_visualTracker)
				loc = cv::Point2f(m_visualTracker->m_targetDisplacement2D_h[trackerInd].x, m_visualTracker->m_targetDisplacement2D_h[trackerInd].y)
							+ cv::Point2f(corrRespMats[trackerInd].size().width, corrRespMats[trackerInd].size().height)*0.5f;
			else
				loc = cv::Point2f(m_dcfTracker->m_targetDisplacement2D_h[trackerInd].x, m_dcfTracker->m_targetDisplacement2D_h[trackerInd].y)
							+ cv::Point2f(corrRespMats[trackerInd].size().width, corrRespMats[trackerInd].size().height)*0.5f;

			loc.x *= wScale;
			loc.y *= hScale;

			cv::drawMarker(
				response_viz_rgb, loc+cv::Point2f(0.5f,0.5f), CV_RGB(255,255,0), cv::MARKER_CROSS, 1.5*wScale, 2
			);
		}

		/// Draw the detector bbox points within the search region.
		if(showDetBboxes)
		{
			if(m_visualTracker)
			{
				for(uint i=0; i<m_visualTracker->m_detectorBboxes_size_h[trackerInd]; i++)
				{
					cv::Point2f dBboxCenterPtInCorrResp = cv::Point2f(
						m_visualTracker->m_detectorBboxes_h[trackerInd*m_targets->getMaxTargetsPerStream() + i].x*wScale,
						m_visualTracker->m_detectorBboxes_h[trackerInd*m_targets->getMaxTargetsPerStream() + i].y*hScale
					);

					/// Convert the center point of the detector bbox into the tracker corr resp's coordinate system.
					cv::Point ptInCorrResp_shifted(
						int(response_viz_gray.cols/2 + dBboxCenterPtInCorrResp.x +0.5f)%(response_viz_gray.cols),
						int(response_viz_gray.rows/2 + dBboxCenterPtInCorrResp.y +0.5f)%(response_viz_gray.rows)
					);
					cv::drawMarker(
						response_viz_rgb, ptInCorrResp_shifted,
						CV_RGB(255,0,255), cv::MARKER_TILTED_CROSS, 1.5*wScale, 3
					);
				}
			}
			else
			{
				for(uint i=0; i<m_dcfTracker->m_detectorBboxes_size_h[trackerInd]; i++)
				{
					cv::Point2f dBboxCenterPtInCorrResp = cv::Point2f(
						m_dcfTracker->m_detectorBboxes_h[trackerInd*m_targets->getMaxTargetsPerStream() + i].x*wScale,
						m_dcfTracker->m_detectorBboxes_h[trackerInd*m_targets->getMaxTargetsPerStream() + i].y*hScale
					);

					/// Convert the center point of the detector bbox into the tracker corr resp's coordinate system.
					cv::Point ptInCorrResp_shifted(
						int(response_viz_gray.cols/2 + dBboxCenterPtInCorrResp.x +0.5f)%(response_viz_gray.cols),
						int(response_viz_gray.rows/2 + dBboxCenterPtInCorrResp.y +0.5f)%(response_viz_gray.rows)
					);
					cv::drawMarker(
						response_viz_rgb, ptInCorrResp_shifted,
						CV_RGB(255,0,255), cv::MARKER_TILTED_CROSS, 1.5*wScale, 3
					);
				}

			}
		}

		return response_viz_rgb;
	}

	/**
	 * @brief Prepare an 8-bit BGR visualization of a target's segmentation-filter mask.
	 * @param[in] streamId_ The video stream identifier.
	 * @param[in] trackerInd The target's tracker index.
	 * @param[in] maskMats The segmentation-filter masks for the tracker batch.
	 * @param[in] dstSize The requested output image size.
	 * @return The prepared BGR mask image.
	 */
	cv::Mat prepareMaskFilterImg(
		const NvMOTStreamId streamId_, const uint trackerInd, std::vector<cv::Mat> &maskMats, const cv::Size dstSize
	)
	{
		const std::map<uint32_t, std::shared_ptr<NvTrackedObject>> &targetList = m_targets->getTrackedObjectsInUseList(
			streamId_
		);

		if(targetList.size() == 0)
		{
			return cv::Mat::zeros(maskMats[trackerInd].size(), CV_8UC3);
		}

		const std::shared_ptr<NvTrackedObject> target = targetList.at(trackerInd);

		if(target->getTrackerState() == TRACKER_STATE::EMPTY)
		{
			return cv::Mat::zeros(maskMats[trackerInd].size(), CV_8UC3);
		}

		assert(maskMats[trackerInd].size().width > 0 && maskMats[trackerInd].size().height > 0);

		/// Scale the mask from [0.0, 1.0] into 8-bit values.
		cv::Mat mask_viz_gray = maskMats[trackerInd] * 255;
		cv::Mat mask_viz_rgb;

		mask_viz_gray.convertTo(mask_viz_gray, CV_8UC1);
		cvtColor(mask_viz_gray, mask_viz_rgb, cv::COLOR_GRAY2BGR);

		cv::resize(mask_viz_rgb, mask_viz_rgb, dstSize);
		return mask_viz_rgb;

	}

	/**
	 * @brief Draw a target's complete trajectory on a frame.
	 * @param[in] targetID The persistent target identifier.
	 * @param[in] trajectory The complete target trajectory.
	 * @param[in,out] frame The frame updated with the rendered trajectory.
	 */
	void drawFullTrajectory(const uint64_t targetID, const std::list<TargetInfo> &trajectory, cv::Mat &frame)
	{
		/// Param to be configured.
		const int radius = 5;

		if(trajectory.size() == 0)
		{
			return;
		}

		cv::Point objLoc_curr, objLoc_prev;
		uint i=0;

		cv::Rect2f lastBbox = CV_FUNCS::convertMotRect2f2CvRect2f(trajectory.rbegin()->tBbox);

		/// Show target ID.
		cv::putText(
			frame, std::to_string(targetID), cv::Point(lastBbox.tl()) + cv::Point(lastBbox.width*0.5),
			cv::FONT_HERSHEY_SIMPLEX, 0.6, CV_RGB(155,255,55), 2, 8
		);

		for(auto pt=trajectory.begin(); pt != trajectory.end(); ++pt, ++i)
		{
			cv::Rect2f bbox = CV_FUNCS::convertMotRect2f2CvRect2f((*pt).tBbox);
			objLoc_curr = (bbox.tl()+bbox.br())*0.5;

			float fading = std::max((float)0, (i*1.5f)/((float)trajectory.size()-1));
			cv::Scalar color = cv::Scalar(0,255,255)*fading + cv::Scalar(255,255,0)*(1.f - fading);

			if((*pt).trackerState == TRACKER_STATE::PROJECTED)
			{
				color = cv::Scalar(0,0,255);
			}
			else if((*pt).trackerState != TRACKER_STATE::ACTIVE)
			{
				color = cv::Scalar(0,0,0);
			}

			if(i == trajectory.size()-1)
			{
				cv::circle(frame, objLoc_curr, radius, color, -1);
				cv::circle(frame, objLoc_curr, radius+2, CV_RGB(255,0,0), 2);

				cv::rectangle(frame, bbox, color, 1, 1);
			}
			else
			{
				cv::circle(frame, objLoc_curr, radius, color, 1);
			}

			if(i>0)
				cv::line(frame, objLoc_prev, objLoc_curr, color, 1);

			objLoc_prev = objLoc_curr;
		}
	}

	/**
	 * @brief Draw a target's recent trajectory, current boxes, and optional predictions on a frame.
	 * @param[in] tracker The tracked object whose trajectory is drawn.
	 * @param[in,out] frame The frame updated with the rendered trajectory.
	 * @param[in] showCurrentDetBbox Whether to draw the current detector bounding box.
	 * @param[in] showCurrentTargetBbox Whether to draw the current target bounding box.
	 * @param[in] showPrediction Whether to draw predicted target positions.
	 * @param[in] showPredictionUncertainty Whether to draw prediction uncertainty.
	 * @param[in] showTrajectoryOnFeet Whether to anchor trajectory points at projected feet locations.
	 */
	void drawRecentTrajectory(
		const std::shared_ptr<NvTrackedObject> tracker, cv::Mat &frame, const bool showCurrentDetBbox = true,
		const bool showCurrentTargetBbox = true, const bool showPrediction = false,
		const bool showPredictionUncertainty = false, const bool showTrajectoryOnFeet = false
	)
	{
		/// Param to be configured.
		const int radius = 3;
		const int tailLength = 300;

		cv::Point objLoc_curr, objLoc_prev;

		const std::list<TargetInfo> trajectory = tracker->getTrajectoryBuffer();

		if(trajectory.size() == 0)
		{
			return;
		}

		/// If the last target state is not Active, then don't draw.
		{
			auto pt = trajectory.rbegin();

			if(!((*pt).trackerState == TRACKER_STATE::ACTIVE
					|| ((*pt).trackerState == TRACKER_STATE::TENTATIVE && (*pt).matchedDetObjInfo != nullptr)))
			{
				return;
			}
		}

		uint32_t targetID = tracker->getTargetID();
		cv::Rect2f lastBbox = CV_FUNCS::convertMotRect2f2CvRect2f(trajectory.rbegin()->tBbox);
		const cv::Scalar targetColor = tracker->getColor();

		/// Show target ID.
		cv::putText(
			frame, std::to_string(targetID), cv::Point(lastBbox.tl()) + cv::Point(lastBbox.width*0.5),
			cv::FONT_HERSHEY_SIMPLEX, 0.6, targetColor, 2, 8
		);

		auto pt = trajectory.begin();
		uint totalPts = trajectory.size();

		/// Consider only the latest N points.
		if(trajectory.size() > tailLength)
		{
			std::advance(pt, trajectory.size() - tailLength);
			totalPts = tailLength;
		}

		uint i=0;
		for(; pt != trajectory.end(); ++pt, ++i)
		{
			cv::Rect2f tBbox = CV_FUNCS::convertMotRect2f2CvRect2f((*pt).tBbox);

			if(showTrajectoryOnFeet)
			{
				if(tracker->m_objModelProjector != nullptr && tracker->m_objModelProjector->isInitialized())
				{
					/// Feet location.
					objLoc_curr = cv::Point((*pt).pt_img_feet(0), (*pt).pt_img_feet(1));
				}
				else
				{
					/// Bbox center.
					objLoc_curr = tBbox.br() - cv::Point2f(tBbox.width/2, 0);
				}
			}
			else
			{
				/// Bbox center.
				objLoc_curr = (tBbox.tl()+tBbox.br())*0.5;
			}

			float fading = std::max((float)0, (i*1.5f)/((float)totalPts-1));
			cv::Scalar color = targetColor*fading + cv::Scalar(100,100,100)*(1.f - fading);

			if(!((*pt).trackerState == TRACKER_STATE::ACTIVE
					|| ((*pt).trackerState == TRACKER_STATE::TENTATIVE && (*pt).matchedDetObjInfo != nullptr)))
			{
				color = cv::Scalar(0,0,0);
			}

			if(std::next(pt) == trajectory.end())
			{
				if(showCurrentDetBbox)
				{
					/// Draw original detection bbox if there is.
					if(pt->matchedDetObjInfo != nullptr && pt->matchedDetObjInfo->objIn != nullptr)
					{
						const NvMOTObjToTrack *objIn = pt->matchedDetObjInfo->objIn;
						NvMotDataTypes::Rect2f detBbox = CV_FUNCS::constructRect2f(
							objIn->bbox.x, objIn->bbox.y, objIn->bbox.width, objIn->bbox.height
						);
						cv::Rect2f originalDetBbox = CV_FUNCS::convertMotRect2f2CvRect2f(detBbox);

						cv::rectangle(frame, originalDetBbox, cv::Scalar(100,100,100), 1, 1);
					}
				}

				if(showCurrentTargetBbox && (*pt).convexHull.size() == 0)
				{
					/// Draw current bbox.
					cv::rectangle(frame, tBbox, color, 1, 1);
				}

				/// Draw the bbox center.
				cv::circle(frame, objLoc_curr, radius, color, -1);
				cv::circle(frame, objLoc_curr, radius+2, CV_RGB(255,0,0), 2);

				if(showPrediction)
				{

					if(tracker->m_objModelProjector != nullptr && tracker->m_objModelProjector->isInitialized())
					{
						/// Time intervals for prediction.
						const std::deque<float> dT = {10.f, 20.f};
						/// The drawn radius is a fixed stand-in for uncertainty.
						const std::deque<float> radius = {20.f, 40.f};

						/// Two ellipses to show the prediction and uncertainty.
						for(uint i=0; i<dT.size(); i++)
						{
							const Eigen::VectorXd pt_world_pred_states = tracker->m_stateEstimator->getPredictedStates(
								dT[i]
							);
							Eigen::Vector3d pt_world_feet_pred; pt_world_feet_pred << pt_world_pred_states(0), pt_world_pred_states(1), 0;
							Eigen::Vector2d eigenVector  = tracker->m_objModelProjector->projectWorldPt2ImgPt(
								pt_world_feet_pred
							);
							cv::Point2d pt_img_feet_pred(eigenVector(0), eigenVector(1));

							cv::Point predLoc(int(pt_img_feet_pred.x+0.5f), int(pt_img_feet_pred.y+0.5f));

							/// Draw the projected circle.
							{
								std::vector<Eigen::Vector2d> projCircle = tracker->m_objModelProjector->calculateProjectedCircleFromWorldFeet(Eigen::Vector2d(pt_world_feet_pred(0), pt_world_feet_pred(1)), radius[i]);

								for(uint i=1; i<projCircle.size(); i++)
								{
									cv::line(
										frame, cv::Point(projCircle[i-1](0), projCircle[i-1](1)),
										cv::Point(projCircle[i](0), projCircle[i](1)), targetColor, 2, cv::LINE_AA
									);
								}
								cv::line(
									frame, cv::Point(projCircle[0](0),projCircle[0](1)),
									cv::Point(projCircle[projCircle.size()-1](0),projCircle[projCircle.size()-1](1)),
									targetColor, 2, cv::LINE_AA
								);
							}
						}

					}
					else
					{
						/// Time intervals for prediction.
						const std::deque<float> dT = {10.f, 20.f};

						Eigen::VectorXd P;
						Eigen::MatrixXd Pcov;
						P = tracker->m_stateEstimator->getPredictedStates(*(dT.rbegin()));
						cv::Point predLoc(int(P(0)+0.5f), int(P(1)+0.5f));

						cv::line(frame, objLoc_curr, predLoc, CV_RGB(255,0,0), 2);

						if(showPredictionUncertainty)
						{
							/// Two ellipses to show the prediction and uncertainty.
							for(uint i=0; i<dT.size(); i++)
							{
								std::tie(P, Pcov) = tracker->m_stateEstimator->getPredictedStatesWithCov(dT[i]);

								{
									cv::Mat P_ocv(P.rows(), P.cols(), CV_64F);
									cv::Mat Pcov_ocv(Pcov.rows(), Pcov.cols(), CV_64F);

									cv::eigen2cv(P, P_ocv);
									cv::eigen2cv(Pcov, Pcov_ocv);

									drawUncertaintyEllipse(frame, P_ocv, Pcov_ocv, color, 0.5);
								}
							}
						}
					}
				}
			}
			else
			{
				cv::circle(frame, objLoc_curr, radius, color, -1);
			}

			if(i>0)
			{
				cv::line(frame, objLoc_prev, objLoc_curr, color, 1);
			}

			objLoc_prev = objLoc_curr;
		}
	}

	/**
	 * @brief Draw a covariance-derived uncertainty ellipse around a predicted position.
	 * @param[in,out] frame The frame updated with the rendered ellipse.
	 * @param[in] P The predicted position vector.
	 * @param[in] Pcov The predicted position covariance matrix.
	 * @param[in] color The ellipse color.
	 * @param[in] scale The ellipse-size scale factor.
	 */
	void drawUncertaintyEllipse(
		cv::Mat &frame, const cv::Mat P, const cv::Mat Pcov, cv::Scalar color, float scale = 1.f
	)
	{
		/// Compute the vanishing correlation coefficient.
		double rho;
		double sigma[2];

		sigma[0] = (double)std::sqrt(Pcov.at<double>(0,0));
		sigma[1] = (double)std::sqrt(Pcov.at<double>(1,1));
		rho = (double)(Pcov.at<double>(0,1)/(sigma[0]*sigma[1]));

		/// Compute the principal semi-diameters of the ellipse.
		double width, height;
		/// Compute the angle between the x1 axis and the semi-diameter of length p1.
		double a;

		if(sigma[0] != sigma[1])
			a = atan((double)(2.*rho*sigma[0]*sigma[1] / (std::pow(sigma[0],2)-std::pow(sigma[1],2))))/2.0;
		else
			a = 0;

		/// Convert from Radian to Degree.
		a *= 180/M_PI;

		width = std::pow(sigma[0]*sigma[1],2) * (1.-std::pow(rho,2)) / (std::pow(sigma[1]*cos(a),2) - 2.0*rho*sigma[0]*sigma[1]*sin(a)*cos(a) + std::pow(sigma[0]*sin(a),2));
		height = std::pow(sigma[0]*sigma[1],2) * (1.-std::pow(rho,2)) / (std::pow(sigma[1]*sin(a),2) + 2.0*rho*sigma[0]*sigma[1]*sin(a)*cos(a) + std::pow(sigma[0]*cos(a),2));

		width = std::pow(width, 0.7);
		height = std::pow(height, 0.7);

		width *= scale;
		height *= scale;

		cv::Point2i loc(P.at<double>(0), P.at<double>(1));

		/// Drawing the ellipse.
		cv::ellipse(frame, loc, cv::Size(width, height), a, 0, 360, color, 1, cv::LINE_AA);
	}

	/**
	 * @brief Draw a target's projected human model on the camera frame.
	 * @param[in] target The tracked object whose model is drawn.
	 * @param[in,out] frame The camera frame updated with the projected model.
	 */
	void drawProjectedTargetModel(const std::shared_ptr<NvTrackedObject> target, cv::Mat &frame)
	{
		const std::list<TargetInfo> trajectory = target->getTrajectoryBuffer();

		if(trajectory.size() == 0)
		{
			return;
		}

		/// If the last target state is not Active, then don't draw.
		{
			auto pt = trajectory.rbegin();

			if(!((*pt).trackerState == TRACKER_STATE::ACTIVE
					|| ((*pt).trackerState == TRACKER_STATE::TENTATIVE && (*pt).matchedDetObjInfo != nullptr)))
			{
				return;
			}
		}

		const cv::Scalar targetColor = target->getColor();
		const NvMOTStreamId streamId = target->getVideoStreamId();
		const TargetInfo &lastTargetInfo = *trajectory.rbegin();

		/// Draw the cylinderic human model.
		{
			const std::vector<Eigen::Vector2f> &convexHull = lastTargetInfo.convexHull;
			const Eigen::Vector2f &tLoc = lastTargetInfo.loc;
			const Eigen::Vector2f &pt_img_feet = lastTargetInfo.pt_img_feet;

			if(convexHull.size() == 0)
				return;

			std::vector<cv::Point> projected_cylinder;
			for(auto &pt : convexHull)
			{
				projected_cylinder.push_back(cv::Point(pt(0), pt(1)));
			}

			for(uint i=1; i<projected_cylinder.size(); i++)
			{
				cv::line(frame, projected_cylinder[i-1], projected_cylinder[i], targetColor, 2, cv::LINE_AA);
			}
			cv::line(
				frame, projected_cylinder[0],
				projected_cylinder[projected_cylinder.size()-1], targetColor, 2, cv::LINE_AA
			);

			/// Draw the feet location.
			const uint humanFeetRadius = 3;
			cv::circle(frame, cv::Point2d(pt_img_feet(0), pt_img_feet(1)) , humanFeetRadius, targetColor, -1);
		}
	}

	/**
	 * @brief Draw peer-camera target boxes and trajectories projected into the current camera frame.
	 * @param[in] peerTargetDB The peer-camera target database.
	 * @param[in,out] frame The camera frame updated with peer trajectories.
	 * @param[in] videoStreamID The current video stream identifier.
	 * @param[in] frameNum The current frame number.
	 * @param[in] targetColorList Colors already assigned to local target identifiers.
	 */
	void drawProjectedPeerTrajectory(
		const PeerTargetDB& peerTargetDB, cv::Mat &frame, const NvMOTStreamId &videoStreamID,
		const int& frameNum, const std::map<TargetId, cv::Scalar>& targetColorList
	)
	{
		for (auto &peerTargets : peerTargetDB) {
			CamId camId = peerTargets.first;
			for (auto &peerTargetIt : peerTargets.second) {
				TargetId targetID = peerTargetIt.first;

				const PeerTarget& peerTarget = peerTargetIt.second;
				/// Skip peers whose last frame is more than two frames behind.
				if (peerTarget.frameNum + 2 < frameNum)
					continue;

				Eigen::Vector2d peer_world_feet(peerTarget.nextState(0), peerTarget.nextState(1));

				Eigen::Vector2d pt_img_feet = m_targets->m_objModelProjectors.at(videoStreamID)->getProjectedWorldGroundPts({peer_world_feet}).front();

				/// Set target trajectory color.
				unsigned camNum = 100;
				cv::Scalar color = make_dist_colors(camNum).at<cv::Vec3b>(0, camId);
				if (targetColorList.find(targetID) != targetColorList.end())
				{
					color = targetColorList.at(targetID);
				}

				NvMotDataTypes::Rect2f peerMotBbox = m_targets->m_objModelProjectors.at(videoStreamID)->getBboxOfProjectedCylinderFromWorldFeet(Eigen::Vector2d(peer_world_feet(0), peer_world_feet(1)), -1);
				cv::Rect2f bbox = CV_FUNCS::convertMotRect2f2CvRect2f(peerMotBbox);

				cv::rectangle(frame, bbox, color, 2, 1);
				cv::putText(
					frame, std::string("ID " + std::to_string(int(targetID)) + " / Cam " + std::to_string(camId)),
					cv::Point(int(bbox.x), int(bbox.y-10)), cv::FONT_HERSHEY_SIMPLEX, 1.2, color, 2, 8
				);

				const Tracklet& tracklet = peerTarget.tracklet;
				if (tracklet.empty())
					continue;

				cv::Point objLoc_curr, objLoc_prev;
				for (auto it = tracklet.rbegin(); it!= tracklet.rend(); ++it)
				{
					Eigen::Vector2d pt_world((*it).second(0), (*it).second(1));
					Eigen::Vector2d pt_img = m_targets->m_objModelProjectors.at(videoStreamID)->getProjectedWorldGroundPts({pt_world}).front();

					objLoc_curr = cv::Point(int(pt_img(0)), int(pt_img(1)));
					if (it != tracklet.rbegin())
					{
						/// Not the first point.
						cv::line(frame, objLoc_curr, objLoc_prev, color, 2, cv::LINE_AA);
					}

					objLoc_prev = objLoc_curr;
				}
			}
		}
	}
};

#endif // VISUALTRACKERLIB_VIZOUTPUT_HPP_
