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

#include <algorithm>
#include <iostream>
#include <random>

#include "modules/utility/debug_common.hpp"
#include "modules/TrajectoryManager/TrajectoryOperations.hpp"
#include "modules/NvMultiObjectTracker/NvTrackedObject.hpp"
#include "modules/TrajectoryManager/NvTrajectoryManagerOnline.hpp"
#include "modules/MultiViewAssociator/MultiViewAssociator_factory.hpp"

using std::cout;
using std::endl;

/**
 * @brief Initialize a tracked target and its estimator.
 * @param[in] configFilePath Path to the tracker configuration file.
 * @param[in] params Parsed target-manager parameters.
 * @param[in] trackerInd Index of this target in the allocated memory layout.
 */
NvTrackedObject::NvTrackedObject(
	const std::string &configFilePath, const NvTrackerParamsTargetManager *params, const TrackerIndex trackerInd
)
: m_params(params), m_targetInd(trackerInd)
{
	/// Instantiate an estimator based on the config.
	m_stateEstimator = EstimatorFactory2::getInstance().makeEstimator(configFilePath);

	if(m_stateEstimator == nullptr)
	{
		throw std::runtime_error("[NvTrackedObject][ERROR] Cannot instantiate a state estimator!!");
	}

	reset();
	/// Obtain the multi-view associator parameters.
	m_multiViewParams = MultiViewAssociatorParams(configFilePath, "MultiViewAssociator");
}

/** @brief Release resources owned by the tracked target. */
NvTrackedObject::~NvTrackedObject()
{
}

/** @brief Reset the target states for later use. */
void NvTrackedObject::reset()
{
	m_targetInfo = TargetInfo();
	m_stateEstimator->reset();
	m_objModelProjector.reset();
	m_pastFrameDataBuffer.clear();

	m_trajectoryBuffer.clear();
	m_trajectorySkipInterval = 1;
	m_needToSetTargetID = false;
	m_ready4ReAssoc = false;
	m_targetID = -1;
	m_becameActive = false;
	/// Persistent; true once track was ever ACTIVE.
	m_hasBeenActive = false;
	m_acquiredID = false;
	m_peakCorrResp_long = -1.f;
	m_peakCorrResp_short = -1.f;
	m_progBbox.setEmpty();
	m_maskOriginal = NvMOTMask();
	m_needToReAssocFromReidService = true;
	m_adoptedReidServiceID = false;

	#ifdef VIZ_DEBUG
		/// Assign a random, unique color.
		static std::mt19937 gen{std::random_device{}()};
		std::uniform_int_distribution<int> dist(0, 254);
		int r = dist(gen);
		int g = dist(gen);
		int b = 255 - (r+g)*0.5;

		m_color = CV_RGB(r,g,b);
	#endif
}

/**
 * @brief Expand a target bounding box into its search region.
 * @param[in] bbox Target bounding box to expand.
 */
const NvMotDataTypes::Rect2f NvTrackedObject::defineSearchRegion(const NvMotDataTypes::Rect2f bbox) const
{
	return CV_FUNCS::computeSearchRegion(bbox, m_params->m_searchRegionPaddingScale);
}

/**
 * @brief Store a deep copy of the object model projector.
 * @param[in] objModelProjector Object model projector to copy.
 */
void NvTrackedObject::setObjModelProjector(std::shared_ptr<ObjectModelProjector> objModelProjector)
{
	m_objModelProjector.reset();

	/// Make a deep copy.
	m_objModelProjector = std::make_shared<ObjectModelProjector>(*(objModelProjector.get()));
}

/**
 * @brief Check whether the object model projector is initialized.
 * @return True when the object model projector is initialized.
 */
bool NvTrackedObject::isObjModelProjectorInitialized()
{
	if (m_objModelProjector == nullptr || (!m_objModelProjector->isInitialized()))
	{
		return false;
	}
	return true;
}

/**
 * @brief Set 3D bounding box information.
 * @param[in] pt_world_feet World coordinates of the feet point.
 * @param[out] bbox3DWorld 3D bounding box in world coordinates.
 * @param[out] bbox3DImg 2D projection of the 3D bounding box corners, eight corners with x and y coordinates.
 */
void NvTrackedObject::setBbox3D(const Eigen::Vector2d& pt_world_feet, NvDsObj3DBbox& bbox3DWorld, float (*bbox3DImg)[2])
{
	bbox3DWorld = m_objModelProjector->getWorld3DBboxFromWorldFeet(pt_world_feet, getClassId(), getTargetInd());
	std::vector<Eigen::Vector2d> projCorners = m_objModelProjector->calculateProjCornersOf3DBboxFromWorldFeet(
		pt_world_feet, getClassId(), getTargetInd()
	);

	if(projCorners.size() != 8)
	{
		throw std::runtime_error("[NvTrackedObject::setBbox3D] Projected corners size is not 8");
	}
	for(uint i=0; i < projCorners.size(); i++)
	{
		bbox3DImg[i][0] = (int) (projCorners.at(i)(0) + 0.5);
		bbox3DImg[i][1] = (int) (projCorners.at(i)(1) + 0.5);
	}
}

/**
 * @brief Adopt and fuse a newer peer tracklet into this target.
 * @param[in] tracklet_peer Newer tracklet fused into this target's trajectory.
 */
void NvTrackedObject::adoptTracklet(const std::list<TargetInfo> &tracklet_peer)
{
	std::list<TargetInfo> tracklet = getTrajectoryBuffer();
	const TargetInfo lastTargetInfo_peer = *tracklet_peer.rbegin();
	const TargetInfo firstTargetInfo_peer = *tracklet_peer.begin();
	const NvMOTFrameNum firstFrameNum_peer = tracklet_peer.begin()->frameNum;

	/// Remove invalid points after the tracklet's last ACTIVE state.
	auto it = tracklet.rbegin();
	while(it != tracklet.rend()
			&& (it->trackerState != TRACKER_STATE::ACTIVE || (!m_isOffline && (it->frameNum >= firstFrameNum_peer))))
	{

		it = decltype(it)(tracklet.erase( std::next(it).base() ));
	}

	/// Skip fusion when stripping leaves no usable history to interpolate.
	if(tracklet.empty())
	{
		return;
	}

	if(tracklet.rbegin()->frameNum >= firstFrameNum_peer)
	{
		/// Fuse overlapping tracklets.
		assert(m_isOffline == true);
		std::list<TargetInfo> newTracklet = fuseOverlappingTracklets(tracklet, tracklet_peer);
		setTrajectoryBuffer(newTracklet);
		setTargetInfo(*newTracklet.rbegin());
	}
	else
	{
	/// Append projected trajectory points for non-overlapping tracklets with the long-term Kalman filter.
	if(tracklet.size() > 0)
	{
		const TargetInfo lastTargetInfo = *it;

		setTrajectoryBuffer(tracklet);
		setTargetInfo(lastTargetInfo);

		/// Reset the state estimator to start fresh again.
		m_stateEstimator->reset();

		/// Get all the interpolated target points.
		std::deque<TargetInfo> interpolatedTargets;
		if(m_isOffline)
		{
			/// For offline tracking, interpolate the missing points between the last tracked point of current target and first point in the peer.
			interpolatedTargets = TrajOps::generateInterpolatedTargets(lastTargetInfo, firstTargetInfo_peer);
		}
		else
		{
			/// For online tracking, interpolate the last tracked points from the two objects.
			interpolatedTargets = TrajOps::generateInterpolatedTargets(lastTargetInfo, lastTargetInfo_peer);
			/// Add the last one.
			interpolatedTargets.push_back(lastTargetInfo_peer);
		}
		const NvMOTClassId classId = getClassId();

		for(uint i=0; i<interpolatedTargets.size(); i++)
		{
			NvMotDataTypes::Rect2f newTargetBbox = interpolatedTargets[i].tBbox;
			const NvMOTFrameNum newFrameNum = interpolatedTargets[i].frameNum;

			/// Check if 3D model is used.
			if(m_objModelProjector != nullptr && m_objModelProjector->isInitialized())
			{
				Eigen::Vector2d tLoc; tLoc << newTargetBbox.min()(0)+newTargetBbox.sizes()(0)/2, newTargetBbox.min()(1)+newTargetBbox.sizes()(1)/2;

				Eigen::Vector2d world_feet = m_objModelProjector->getWorldFeetPtOnGroundFromCamWaistPt(
					Eigen::Vector2d(tLoc(0), tLoc(1)), classId, (int)m_targetInd
				);

				/// Update the state estimator with the measurement.
				std::vector<std::tuple<Eigen::VectorXd, Eigen::Matrix2d, bool>> measurements = {{world_feet, -Eigen::Matrix2d::Identity(), true}};

				m_stateEstimator->update(measurements);

				const Eigen::VectorXd &pt_world_feet_est = m_stateEstimator->getStates();

				newTargetBbox = m_objModelProjector->getBboxOfProjectedCylinderFromWorldFeet(
					Eigen::Vector2d(pt_world_feet_est(0), pt_world_feet_est(1)), classId, (int)m_targetInd
				);

				if(m_params->m_enableBboxUnClipping && getAge() > 0)
				{
					newTargetBbox = CV_FUNCS::unclipBbox(newTargetBbox, m_params->srcImageBox, m_targetInfo.tBbox);
				}
				const float imgBorderCroppingVisibility_ = CV_FUNCS::computeVisibility(
					newTargetBbox, m_params->srcImageBox
				);

				/// Update the target states with the estimated state back.
				m_targetInfo.update(newFrameNum, newTargetBbox, imgBorderCroppingVisibility_);

				m_targetInfo.pt_world_feet = m_objModelProjector->getWorldFeetPtOnGroundFromCamWaistPt(Eigen::Vector2d(m_targetInfo.loc(0), m_targetInfo.loc(1)), classId, (int)m_targetInd).cast<float>();
				m_targetInfo.pt_world_feet_cov = getLocCovariance().cast<float>();
				m_targetInfo.pt_img_feet = m_objModelProjector->getCamFeetPtFromCamWaistPt(Eigen::Vector2d(m_targetInfo.loc(0), m_targetInfo.loc(1)), classId, (int)m_targetInd).cast<float>();
			}
			else
			{
				/// Update the state estimator with the measurement.
				std::vector<std::tuple<Eigen::VectorXd, Eigen::Matrix2d, bool>> measurements = {{CV_FUNCS::convertBboxToZ(newTargetBbox, m_stateEstimator->useAspectRatio()), -Eigen::Matrix2d::Identity(), true}};

				m_stateEstimator->update(measurements);

				/// Use the state estimator.
				newTargetBbox = CV_FUNCS::convertXToBbox(
					m_stateEstimator->getStates(), m_stateEstimator->useAspectRatio()
				);

				if(m_params->m_enableBboxUnClipping && getAge() > 0)
				{
					newTargetBbox = CV_FUNCS::unclipBbox(newTargetBbox, m_params->srcImageBox, m_targetInfo.tBbox);
				}
				const float imgBorderCroppingVisibility = CV_FUNCS::computeVisibility(
					newTargetBbox, m_params->srcImageBox
				);

				/// Update the target states with the estimated state back.
				m_targetInfo.update(newFrameNum, newTargetBbox, imgBorderCroppingVisibility);
			}

				if(m_params->m_debugVerboseLevel > 3)
				{
					cout << "[NvTrackedObject::adoptTracklet] frameNum[" << newFrameNum << "] newTargetBbox (x,y,w,h) = (" << newTargetBbox.min()(0) << ", " << newTargetBbox.min()(1) << ", " << newTargetBbox.sizes()(0) << ", " << newTargetBbox.sizes()(1) << ")" << endl;
				}

			updateTrajectoryBuffer();
		}
		if(m_isOffline)
		{
			/// Append the remaining peer points during offline metadata fusion.
			for (auto it = tracklet_peer.begin(); it != tracklet_peer.end(); it++)
			{
				TargetInfo targetInfo_peer = *it;
				NvMotDataTypes::Rect2f newTargetBbox = targetInfo_peer.tBbox;
				const NvMOTFrameNum newFrameNum = targetInfo_peer.frameNum;
				const float imgBorderCroppingVisibility = targetInfo_peer.visibility;
				m_targetInfo.update(
					newFrameNum, targetInfo_peer.matchedDetObjInfo, newTargetBbox, imgBorderCroppingVisibility
				);
				m_targetInfo.pt_world_feet_cov = targetInfo_peer.pt_world_feet_cov;
				/// Preserve the peer tracker's confidence because update() treats its input as a detection.
				m_targetInfo.confidence = targetInfo_peer.confidence;
				if (m_params->m_debugVerboseLevel > 3)
				{
					cout << "[NvTrackedObjectOffline::adoptTracklet] frameNum[" << newFrameNum << "] peerTargetBbox (x,y,w,h) = (" << newTargetBbox.min()(0) << ", " << newTargetBbox.min()(1) << ", " << newTargetBbox.sizes()(0) << ", " << newTargetBbox.sizes()(1) << ")" << endl;
				}

				/// Update the trajectory buffer.
				updateTrajectoryBuffer();
			}
		}
	}
	}
}

/**
 * @brief Fuse overlapping portions of this target and a peer tracklet.
 * @param[in] selfTracklet This target's tracklet.
 * @param[in] peerTracklet Peer tracklet fused with the target tracklet.
 * @return Fused tracklet, or the target tracklet when either input is empty.
 */
std::list<TargetInfo> NvTrackedObject::fuseOverlappingTracklets(
	const std::list<TargetInfo> &selfTracklet, const std::list<TargetInfo> &peerTracklet
)
{
	std::list<TargetInfo> newTracklet;

	if (selfTracklet.size() <= 0 || peerTracklet.size() <= 0)
	{
		CLOG("Self tracklet and peer tracklet should not be empty");
		return selfTracklet;
	}

	const NvMOTFrameNum firstFrameNum_self = selfTracklet.begin()->frameNum;
	const NvMOTFrameNum lastFrameNum_self = selfTracklet.rbegin()->frameNum;
	const NvMOTFrameNum firstFrameNum_peer = peerTracklet.begin()->frameNum;
	const NvMOTFrameNum lastFrameNum_peer = peerTracklet.rbegin()->frameNum;

	if(m_params->m_debugVerboseLevel > 2)
	{
		cout << "[NvTrackedObjectOffline::fuseOverlappingTracklets()]"
				<< "FrameNum (first, last) = (" << firstFrameNum_self << ", " << lastFrameNum_self
				<< ")"
				<< "FrameNum (first, last) = (" << firstFrameNum_peer << ", " << lastFrameNum_peer
				<< ")" << endl;
	}

	/// Extend the interpolation range.
	const NvMOTFrameNum firstFrameNum = std::min(firstFrameNum_self, firstFrameNum_peer);
	const NvMOTFrameNum lastFrameNum = std::max(lastFrameNum_self, lastFrameNum_peer);

	/// Start from the earliest frameNum and fuse frame-by-frame.
	auto it_self = selfTracklet.begin();
	auto it_peer = peerTracklet.begin();

	NvMOTFrameNum currFrameNum = firstFrameNum;

	/// Get to the start position.
	while(it_self->frameNum < currFrameNum && std::next(it_self) != selfTracklet.end())
	{
		++it_self;
	}

	while(it_peer->frameNum < currFrameNum && std::next(it_peer) != peerTracklet.end())
	{
		++it_peer;
	}

	/// Iterate until both tracklets are all covered.
	while(it_self != selfTracklet.end() || it_peer != peerTracklet.end())
	{
		if(it_self != selfTracklet.end() && it_peer != peerTracklet.end())
		{
			currFrameNum = std::min(it_self->frameNum, it_peer->frameNum);
		}
		else if(it_self != selfTracklet.end())
		{
			currFrameNum = it_self->frameNum;
		}
		else
		{
			currFrameNum = it_peer->frameNum;
		}

		if(it_self != selfTracklet.end() && it_self->frameNum == currFrameNum
				&& it_peer != peerTracklet.end() && it_peer->frameNum == currFrameNum)
		{
			TargetInfo fusedTarget = TrajOps::fuseTargets(*it_self, *it_peer);
			newTracklet.push_back(fusedTarget);
			it_self++;
			it_peer++;
		}
		else if(it_self != selfTracklet.end() && it_self->frameNum == currFrameNum)
		{
			newTracklet.push_back(*it_self);

			it_self++;
		}
		else if(it_peer != peerTracklet.end() && it_peer->frameNum == currFrameNum)
		{
			newTracklet.push_back(*it_peer);

			it_peer++;
		}
		else
		{
			std::stringstream msg;
			msg << "[NvTrackedObjectOffline::fuseOverlappingTracklets][ERROR] Invalid case!!"
					<< "newTracket Age = " << currFrameNum - firstFrameNum
					<< ", tracketLength (self, peer) = (" << selfTracklet.size() << ", " << peerTracklet.size() << "), "
					<< "(first, last) frameNum = ("
					<< firstFrameNum << ", " << lastFrameNum << "), "
					<< "(frameNum, selfFrameNum, peerFrameNum) = ("
					<< currFrameNum << ", " << it_self->frameNum << ", " << it_peer->frameNum
					<< ")";

			CLOG(msg.str().c_str());
			continue;
		}

		newTracklet.rbegin()->age = currFrameNum - firstFrameNum;

	}

	if(m_params->m_debugVerboseLevel > 2)
	{
		cout << " Fusion complete ---> "
				<< " FrameNum (first, last) = (" << newTracklet.begin()->frameNum
				<< ", " << newTracklet.rbegin()->frameNum
				<< ")" << endl;
	}

	return std::move(newTracklet);
}

/**
 * @brief Update target state from its own prediction on a no-inference frame.
 * @param[in] frameNum Frame number of the prediction update.
 */
void NvTrackedObject::updateTargetStates(const NvMOTFrameNum frameNum)
{
	/// Add new recovered model variables.
	NvMotDataTypes::Rect2f newTargetBbox;
	const NvMOTClassId classId = getClassId();

	/// Check if 3D model is used.
	if(m_objModelProjector != nullptr && m_objModelProjector->isInitialized())
	{
		RecoveredObjModel recovered;
		if (!m_targetInfo.pose25d.empty())
		{
			Eigen::Vector2d \
				pt_img_head((m_targetInfo.pose25d[16*4] + m_targetInfo.pose25d[17*4]) / 2.0f, (m_targetInfo.pose25d[16*4 + 1] + m_targetInfo.pose25d[17*4 + 1]) / 2.0f), \
				pt_img_waist(m_targetInfo.pose25d[0*4], m_targetInfo.pose25d[0*4+1]), \
				pt_img_foot(
					(m_targetInfo.pose25d[9*4] + m_targetInfo.pose25d[10*4] + m_targetInfo.pose25d[11*4] + m_targetInfo.pose25d[12*4]) / 4.0f,
					(m_targetInfo.pose25d[9*4 + 1] + m_targetInfo.pose25d[10*4 + 1] + m_targetInfo.pose25d[11*4 + 1] + m_targetInfo.pose25d[12*4 + 1]) / 4.0f
				);
			float \
				conf_head = (m_targetInfo.pose25d[16*4 + 3] + m_targetInfo.pose25d[17*4 + 3]) / 2.0f, \
				conf_waist = m_targetInfo.pose25d[0*4 + 3], \
				conf_foot = (m_targetInfo.pose25d[9*4 + 3] + m_targetInfo.pose25d[10*4 + 3] + m_targetInfo.pose25d[11*4 + 3] + m_targetInfo.pose25d[12*4 + 3]) / 4.0f;
			const float minPoseConf = getMinPoseConfidence();
			if ((conf_head > minPoseConf && conf_waist > minPoseConf) ||
				(conf_head > minPoseConf && conf_foot > minPoseConf))
			{
				/// Compute model with adaptive height (keypoints from frame N-1).
				recovered = m_objModelProjector->getRecoveredObjectModelWithAdaptiveHeight(
					pt_img_head, pt_img_waist, pt_img_foot, m_targetInfo.origBbox,
					(int)m_targetInd, classId, conf_head, conf_waist, conf_foot
				);
				/// Update targetInfo with pose-corrected values (N-1).
				m_targetInfo.pt_world_feet = recovered.pt_world_feet.cast<float>();
				m_targetInfo.pt_world_feet_cov = getLocCovariance().cast<float>();
				m_targetInfo.pt_img_feet = recovered.pt_img_feet.cast<float>();
				m_targetInfo.convexHull = recovered.convexHull;
				if (!m_trajectoryBuffer.empty()) m_trajectoryBuffer.back() = m_targetInfo;
				/// Keep state estimator in sync: override position with N-1 correction, preserving the current Kalman velocity so prediction forward to frame N continues from the corrected footpoint.
				{
					const Eigen::VectorXd adaptSt = m_stateEstimator->getStates();
					m_stateEstimator->overrideState(recovered.pt_world_feet, Eigen::Vector2d(adaptSt(2), adaptSt(3)));
				}
			}
		}
		float maxPeerDist = m_multiViewParams.m_maxPeerToPredDistance4Fusion;
		/// Override maxPeerDist to a large value for LOCATION_ESTIMATOR_PROJECTIVE.
		if (m_stateEstimator->getStateEstimatorType() == STATE_ESTIMATOR_TYPES::LOCATION_ESTIMATOR_PROJECTIVE) {
			maxPeerDist = 100;
		}
		/// Use peer targets if available - select height with highest confidence.
		if (m_multiViewParams.m_multiViewAssociatorType == MULTI_VIEW_ASSOCIATOR && m_peerTargets.size() > 0) {
			/// Use estimated state since peer already predicted forward to current frame.
			Eigen::VectorXd egoState = m_stateEstimator->getStates();
			Eigen::Vector2d ego_world_feet(egoState(0), egoState(1));
			for (const auto& peerTarget : m_peerTargets) {
				if (peerTarget.heightConfidence > getHeightConfidence()) {
					setTargetHeight(peerTarget.height, peerTarget.heightConfidence);
				}
				Eigen::Vector2d peer_world_feet(peerTarget.nextState(0), peerTarget.nextState(1));
				float distancePeerToPred = (peer_world_feet - ego_world_feet).norm();
				/// Fusion criteria for non-inference frames.
				if (m_objModelProjector->isTargetVisibleInImage(peer_world_feet, peerTarget.height)
					&& peerTarget.frameNum + m_multiViewParams.m_maxPeerFrameDiff4NoDet >= frameNum
					&& distancePeerToPred < maxPeerDist
					)
				{
					m_measList.push_back(std::make_tuple(peer_world_feet, peerTarget.feetCov, true));
				}
			}
		}

		/// Update the state estimator with prediction, visual tracker measurement and/or peer measurements.
		m_stateEstimator->update(m_measList);
		m_measList.clear();
		const Eigen::VectorXd &pt_world_feet_est = m_stateEstimator->getStates();
		newTargetBbox = m_objModelProjector->getBboxOfProjectedCylinderFromWorldFeet(
			Eigen::Vector2d(pt_world_feet_est(0), pt_world_feet_est(1)), classId, -1
		);
#ifdef _WIN32
		NvMOTObjToTrack objToTrack{};
		objToTrack.classId = classId;
		objToTrack.bbox.x = newTargetBbox.min()(0);
		objToTrack.bbox.y = newTargetBbox.min()(1);
		objToTrack.bbox.width = newTargetBbox.sizes()(0);
		objToTrack.bbox.height = newTargetBbox.sizes()(1);
#else
		const NvMOTObjToTrack objToTrack = {
			.classId = classId,
			.bbox = {
				.x = newTargetBbox.min()(0),
				.y = newTargetBbox.min()(1),
				.width = newTargetBbox.sizes()(0),
				.height = newTargetBbox.sizes()(1)
			}
		};
#endif
		recovered = m_objModelProjector->getRecoveredObjectModelFromBbox(
			&objToTrack, frameNum, (int)m_targetInd, false
		);

		newTargetBbox = recovered.bbox;

		if(m_params->m_enableBboxUnClipping && getAge() > 0)
		{
			newTargetBbox = CV_FUNCS::unclipBbox(newTargetBbox, m_params->srcImageBox, m_targetInfo.tBbox);
		}
		const float imgBorderCroppingVisibility = CV_FUNCS::computeVisibility(newTargetBbox, m_params->srcImageBox);

		/// Update the target states.
		m_targetInfo.update(frameNum, newTargetBbox, imgBorderCroppingVisibility);
		m_targetInfo.pt_world_feet = recovered.pt_world_feet.cast<float>();
		m_targetInfo.pt_world_feet_cov = getLocCovariance().cast<float>();
		m_targetInfo.pt_img_feet = recovered.pt_img_feet.cast<float>();
		m_targetInfo.convexHull = recovered.convexHull;
	}
	else
	{
		/// Fall back to the 2D state-estimator model.
		/// Update the state estimator with prediction and/or visual tracker measurement.
		m_stateEstimator->update(m_measList);
		m_measList.clear();

		newTargetBbox = CV_FUNCS::convertXToBbox(m_stateEstimator->getStates(), m_stateEstimator->useAspectRatio());

		if(m_params->m_enableBboxUnClipping && getAge() > 0)
		{
			newTargetBbox = CV_FUNCS::unclipBbox(newTargetBbox, m_params->srcImageBox, m_targetInfo.tBbox);
		}
		const float imgBorderCroppingVisibility = CV_FUNCS::computeVisibility(newTargetBbox, m_params->srcImageBox);

		/// Update the target states with the estimated state back.
		m_targetInfo.update(frameNum, newTargetBbox, imgBorderCroppingVisibility);
	}

	/// Define the search region from the current target location.
	m_searchRegion = defineSearchRegion(m_targetInfo.tBbox);

	/// Check if 3D model is used.
	if(m_objModelProjector != nullptr && m_objModelProjector->isInitialized())
	{
		const Eigen::VectorXd &pt_world_feet_pred = m_stateEstimator->getPredictedStates();

		const NvMotDataTypes::Rect2f newBbox = m_objModelProjector->getBboxOfProjectedCylinderFromWorldFeet(
			Eigen::Vector2d(pt_world_feet_pred(0), pt_world_feet_pred(1)), classId, -1
		);

		m_predictedSearchRegion = defineSearchRegion(newBbox);
	}
	else
	{
		/// Define the predicted search region from the current target location.
		m_predictedSearchRegion = defineSearchRegion(
			CV_FUNCS::convertXToBbox(m_stateEstimator->getPredictedStates(), m_stateEstimator->useAspectRatio())
		);
	}

	/// If this target has not been matched before, then increment it for this non-inference frame as well.
	if(getShadowTrackingAge() > 0 || getTrackerConfidence() < m_params->m_minTrackerConfidence)
	{
		/// Set the shadow tracking age.
		if (getTrackerState() != TRACKER_STATE::QUASIACTIVE)
		{
			incrementShadowTrackingAge();
		}
	}

	if(getTrackerConfidence() < m_params->m_minTrackerConfidence)
	{
		/// Keep the TENTATIVE mode until it gets ACTIVE.
		if(getTrackerState() != TRACKER_STATE::TENTATIVE && getTrackerState() != TRACKER_STATE::QUASIACTIVE)
		{
			setTrackerState(TRACKER_STATE::INACTIVE);

			/// Prepare projected tracklet points once shadow age reaches probation age.
			if((int)getShadowTrackingAge() >= m_params->m_probationAge)
			{
				m_ready4ReAssoc = true;
			}
		}
	}

	m_targetInfo.pose25d.clear();
}

/**
 * @brief Update target state from a new detector measurement.
 * @param[in] frameNum Current frame number.
 * @param[in] detObj Detected object from the input metadata.
 * @param[in] newState New tracker state.
 */
void NvTrackedObject::updateTargetStates(
	const NvMOTFrameNum frameNum, const DetectedObjInfo *detObj, TRACKER_STATE newState
)
{
	/// Add new recovered model variables.
	NvMotDataTypes::Rect2f newTargetBbox;
	NvMOTClassId classId = getClassId();

	/// Update the target states.
	if(detObj != nullptr)
	{
		/// When a new target is added, "m_targetInfo" is not yet initialized So, getClassId() above returns incorrect value Hence, whenever detected object is present take the classId from detected object.
		classId = detObj->classId;

		/// Check if 3D model is used.
		if(m_objModelProjector != nullptr && m_objModelProjector->isInitialized())
		{
			RecoveredObjModel recovered, detRecovered;
			bool useAdaptiveHeight = false;
#ifdef _WIN32
			NvMOTObjToTrack objToTrack{};
			objToTrack.classId = classId;
			objToTrack.bbox.x = detObj->origBbox.min()(0);
			objToTrack.bbox.y = detObj->origBbox.min()(1);
			objToTrack.bbox.width = detObj->origBbox.sizes()(0);
			objToTrack.bbox.height = detObj->origBbox.sizes()(1);
#else
			const NvMOTObjToTrack objToTrack = {
				.classId = classId,
				.bbox = {
					.x = detObj->origBbox.min()(0),
					.y = detObj->origBbox.min()(1),
					.width = detObj->origBbox.sizes()(0),
					.height = detObj->origBbox.sizes()(1)
				}
			};
			#endif
			if (!m_targetInfo.pose25d.empty())
			{
				Eigen::Vector2d \
					pt_img_head((m_targetInfo.pose25d[16*4] + m_targetInfo.pose25d[17*4]) / 2.0f, (m_targetInfo.pose25d[16*4 + 1] + m_targetInfo.pose25d[17*4 + 1]) / 2.0f), \
					pt_img_waist(m_targetInfo.pose25d[0*4], m_targetInfo.pose25d[0*4+1]), \
					pt_img_foot(
						(m_targetInfo.pose25d[9*4] + m_targetInfo.pose25d[10*4] + m_targetInfo.pose25d[11*4] + m_targetInfo.pose25d[12*4]) / 4.0f,
						(m_targetInfo.pose25d[9*4 + 1] + m_targetInfo.pose25d[10*4 + 1] + m_targetInfo.pose25d[11*4 + 1] + m_targetInfo.pose25d[12*4 + 1]) / 4.0f
					);
				float \
					conf_head = (m_targetInfo.pose25d[16*4 + 3] + m_targetInfo.pose25d[17*4 + 3]) / 2.0f, \
					conf_waist = m_targetInfo.pose25d[0*4 + 3], \
					conf_foot = (m_targetInfo.pose25d[9*4 + 3] + m_targetInfo.pose25d[10*4 + 3] + m_targetInfo.pose25d[11*4 + 3] + m_targetInfo.pose25d[12*4 + 3]) / 4.0f;
				const float minPoseConf = getMinPoseConfidence();
				if ((conf_head > minPoseConf && conf_waist > minPoseConf) ||
					(conf_head > minPoseConf && conf_foot > minPoseConf))
				{
					/// Compute model with adaptive height (keypoints from frame N-1).
					recovered = m_objModelProjector->getRecoveredObjectModelWithAdaptiveHeight(
						pt_img_head, pt_img_waist, pt_img_foot, m_targetInfo.origBbox,
						(int)m_targetInd, classId, conf_head, conf_waist, conf_foot
					);

					/// Get the recovered object model from the detection bbox with recovered height.
					detRecovered = m_objModelProjector->getRecoveredObjectModelFromBbox(
						&objToTrack, frameNum, (int)m_targetInd, false
					);
					/// Update trajectory buffer with N-1's pose-corrected values.
					m_targetInfo.pt_world_feet = recovered.pt_world_feet.cast<float>();
					m_targetInfo.pt_world_feet_cov = getLocCovariance().cast<float>();
					m_targetInfo.pt_img_feet = recovered.pt_img_feet.cast<float>();
					m_targetInfo.convexHull = recovered.convexHull;
					if (!m_trajectoryBuffer.empty()) m_trajectoryBuffer.back() = m_targetInfo;
					/// Override Kalman state with N-1 position and better velocity estimate (N-1 to Detection).
					m_stateEstimator->overrideState(
						recovered.pt_world_feet, detRecovered.pt_world_feet - recovered.pt_world_feet
					);
					/// New recovered object model from detection bbox with recovered height.
					recovered = detRecovered;
					/// If foot is visible, clear the (possibly noisy due to wrong height) visual tracker measurement.
					useAdaptiveHeight = true;
				}
			}
			if (!useAdaptiveHeight)
			{
				recovered = m_objModelProjector->getRecoveredObjectModelFromBbox(
					&objToTrack, frameNum, (int)m_targetInd, false
				);
			}
			/// Compute ground covariance and override maxPeerDist to a large value for LOCATION_ESTIMATOR_PROJECTIVE.
			Eigen::Matrix2d measCov = Eigen::Matrix2d::Identity();
			float maxPeerDist = m_multiViewParams.m_maxPeerToPredDistance4Fusion;
			if (usesProjectiveEstimator())
			{
				Eigen::Vector2d bbox_bottom_center(
					(detObj->origBbox.min()(0) + detObj->origBbox.max()(0)) / 2.0, detObj->origBbox.max()(1)
				);
				float pixel_sigma = std::max(
					1.0f, static_cast<float>((recovered.pt_img_feet - bbox_bottom_center).norm())
				);
				measCov = m_objModelProjector->computeGroundCovariance(recovered.pt_img_feet, pixel_sigma);
				maxPeerDist = 100;
			}
			m_measList.push_back(std::make_tuple(recovered.pt_world_feet, measCov, true));

			/// Use peer targets if available - select height with highest confidence.
			if (m_multiViewParams.m_multiViewAssociatorType == MULTI_VIEW_ASSOCIATOR && m_peerTargets.size() > 0) {
				/// Use estimated state since peer already predicted forward to current frame.
				Eigen::VectorXd egoState = m_stateEstimator->getStates();
				Eigen::Vector2d ego_world_feet(egoState(0), egoState(1));
				for (const auto& peerTarget : m_peerTargets) {
					if (peerTarget.heightConfidence > getHeightConfidence()) {
						setTargetHeight(peerTarget.height, peerTarget.heightConfidence);
					}
					Eigen::Vector2d peer_world_feet(peerTarget.nextState(0), peerTarget.nextState(1));
					float distancePeerToPred = (peer_world_feet - ego_world_feet).norm();
					/// Fusion criteria for matched targets.
					if (m_objModelProjector->isTargetVisibleInImage(peer_world_feet, peerTarget.height)
						&& m_targetInfo.visibility >= 0
						&& peerTarget.visibility > m_multiViewParams.m_minPeerVisibility4Fusion
						&& peerTarget.visibility > detObj->visibility
						&& peerTarget.distFromCam < detObj->distFromCam
						&& peerTarget.frameNum + m_multiViewParams.m_maxPeerFrameDiff4NoDet >= frameNum
						&& distancePeerToPred < maxPeerDist
						)
					{
						m_measList.push_back(std::make_tuple(peer_world_feet, peerTarget.feetCov, true));
					}
				}
			}

			m_stateEstimator->update(m_measList);

			const Eigen::VectorXd &pt_world_feet_est = m_stateEstimator->getStates();

			newTargetBbox = recovered.bbox;

			if(m_params->m_enableBboxUnClipping && getAge() > 0)
			{
				newTargetBbox = CV_FUNCS::unclipBbox(newTargetBbox, m_params->srcImageBox, m_targetInfo.tBbox);
			}
			const float imgBorderCroppingVisibility = CV_FUNCS::computeVisibility(newTargetBbox, m_params->srcImageBox);

			/// Update the target states with the estimated state back.
			m_targetInfo.update(frameNum, detObj, newTargetBbox, imgBorderCroppingVisibility);
			/// Update the targetInfo with the estimated world feet and covariance.
			m_targetInfo.pt_world_feet = Eigen::Vector2f(pt_world_feet_est(0), pt_world_feet_est(1));
			m_targetInfo.pt_world_feet_cov = getLocCovariance().cast<float>();
			if (m_params->m_debugVerboseLevel > 4)
			{
				std::cout << "StreamInd: " << (getVideoStreamId() >> 32) <<  ", target ID: " << getTargetID() << ", world feet: ["
					<< pt_world_feet_est(0) << ", " << pt_world_feet_est(1) << "]" << std::endl;
			}
			m_targetInfo.pt_img_feet = recovered.pt_img_feet.cast<float>();
			m_targetInfo.convexHull = recovered.convexHull;
			/// Store the OMP-estimated height, using zero to disable the archive reassociation height gate.
			m_targetInfo.height = (m_objModelProjector->getHeightConfidence() > 0.0f)
								  ? m_objModelProjector->getTargetHeight() : 0.0f;
		}
		else
		{
			/// Fall back to the 2D state-estimator model.
			/// Convert the detection bbox into measurement states.
			if (!m_params->m_useVOTBasedTargetUpdate)
			{
				Eigen::VectorXd detBbox = CV_FUNCS::convertBboxToZ(detObj->bbox, m_stateEstimator->useAspectRatio());

				m_measList.push_back(std::make_tuple(detBbox, -Eigen::Matrix2d::Identity(), true));
			}
			/// Update the state estimator with the measurement.
			m_stateEstimator->update(m_measList);

			newTargetBbox = CV_FUNCS::convertXToBbox(m_stateEstimator->getStates(), m_stateEstimator->useAspectRatio());

			if(m_params->m_enableBboxUnClipping && getAge() > 0)
			{
				newTargetBbox = CV_FUNCS::unclipBbox(newTargetBbox, m_params->srcImageBox, m_targetInfo.tBbox);
			}
			const float imgBorderCroppingVisibility = CV_FUNCS::computeVisibility(newTargetBbox, m_params->srcImageBox);

			if(m_isOffline)
			{
				/// Preserve the tracked object's bounding box and visibility during offline processing.
				m_targetInfo.update(frameNum, detObj, detObj->bbox, detObj->visibility);
				/// Preserve the tracked object's confidence because update() treats its input as a detection.
				m_targetInfo.confidence = detObj->confidence;
			}
			else
			{
				/// Update the target states with the estimated state back.
				m_targetInfo.update(frameNum, detObj, newTargetBbox, imgBorderCroppingVisibility);
			}
		}
	}
	else
	{
		/// This is the case when the target is not detected: detObj == nullptr.
		if(getAge() == 0)
		{
			throw std::runtime_error("Invalid update attempt trackerInd = " + std::to_string(m_targetInd));
		}

		/// Check if 3D model is used.
		if(m_objModelProjector != nullptr && m_objModelProjector->isInitialized())
		{
			RecoveredObjModel recovered;
			if (!m_targetInfo.pose25d.empty())
			{
				Eigen::Vector2d \
					pt_img_head((m_targetInfo.pose25d[16*4] + m_targetInfo.pose25d[17*4]) / 2.0f, (m_targetInfo.pose25d[16*4 + 1] + m_targetInfo.pose25d[17*4 + 1]) / 2.0f), \
					pt_img_waist(m_targetInfo.pose25d[0*4], m_targetInfo.pose25d[0*4+1]), \
					pt_img_foot(
						(m_targetInfo.pose25d[9*4] + m_targetInfo.pose25d[10*4] + m_targetInfo.pose25d[11*4] + m_targetInfo.pose25d[12*4]) / 4.0f,
						(m_targetInfo.pose25d[9*4 + 1] + m_targetInfo.pose25d[10*4 + 1] + m_targetInfo.pose25d[11*4 + 1] + m_targetInfo.pose25d[12*4 + 1]) / 4.0f
					);
				float \
					conf_head = (m_targetInfo.pose25d[16*4 + 3] + m_targetInfo.pose25d[17*4 + 3]) / 2.0f, \
					conf_waist = m_targetInfo.pose25d[0*4 + 3], \
					conf_foot = (m_targetInfo.pose25d[9*4 + 3] + m_targetInfo.pose25d[10*4 + 3] + m_targetInfo.pose25d[11*4 + 3] + m_targetInfo.pose25d[12*4 + 3]) / 4.0f;
				const float minPoseConf = getMinPoseConfidence();
				if ((conf_head > minPoseConf && conf_waist > minPoseConf) ||
					(conf_head > minPoseConf && conf_foot > minPoseConf))
				{
					/// Compute model with adaptive height (keypoints from frame N-1).
					recovered = m_objModelProjector->getRecoveredObjectModelWithAdaptiveHeight(
						pt_img_head, pt_img_waist, pt_img_foot, m_targetInfo.origBbox,
						(int)m_targetInd, classId, conf_head, conf_waist, conf_foot
					);
					/// Update targetInfo with pose-corrected values (N-1).
					m_targetInfo.pt_world_feet = recovered.pt_world_feet.cast<float>();
					m_targetInfo.pt_world_feet_cov = getLocCovariance().cast<float>();
					m_targetInfo.pt_img_feet = recovered.pt_img_feet.cast<float>();
					m_targetInfo.convexHull = recovered.convexHull;
					if (!m_trajectoryBuffer.empty()) m_trajectoryBuffer.back() = m_targetInfo;
					/// Keep state estimator in sync with the N-1 corrected position, preserving current Kalman velocity so prediction to frame N is consistent with the corrected footpoint.
					{
						const Eigen::VectorXd adaptSt = m_stateEstimator->getStates();
						m_stateEstimator->overrideState(
							recovered.pt_world_feet, Eigen::Vector2d(adaptSt(2), adaptSt(3))
						);
					}
				}
			}
			float maxPeerDist = m_multiViewParams.m_maxPeerToPredDistance4Fusion;
			/// Override maxPeerDist to a large value for LOCATION_ESTIMATOR_PROJECTIVE.
			if (usesProjectiveEstimator()) {
				maxPeerDist = 100;
			}
			/// Use peer targets if available - select height with highest confidence.
			if (m_multiViewParams.m_multiViewAssociatorType == MULTI_VIEW_ASSOCIATOR && m_peerTargets.size() > 0) {
				/// Use estimated state since peer already predicted forward to current frame.
				Eigen::VectorXd egoState = m_stateEstimator->getStates();
				Eigen::Vector2d ego_world_feet(egoState(0), egoState(1));
				/// For projective estimator, disable distance check (using estimated state, peer already predicted).
				for (const auto& peerTarget : m_peerTargets) {
					if (peerTarget.heightConfidence > getHeightConfidence()) {
						setTargetHeight(peerTarget.height, peerTarget.heightConfidence);
					}
					Eigen::Vector2d peer_world_feet(peerTarget.nextState(0), peerTarget.nextState(1));
					float distancePeerToPred = (peer_world_feet - ego_world_feet).norm();
					/// Fusion criteria for unmatched targets.
					if (m_objModelProjector->isTargetVisibleInImage(peer_world_feet, peerTarget.height)
						&& peerTarget.frameNum + m_multiViewParams.m_maxPeerFrameDiff4NoDet >= frameNum
						&& distancePeerToPred < maxPeerDist
						)
					{
						m_measList.push_back(std::make_tuple(peer_world_feet, peerTarget.feetCov, true));
						newState = QUASIACTIVE;
					}
				}
			}

			/// Update the state estimator with prediction, visual tracker measurement and/or peer measurements.
			m_stateEstimator->update(m_measList);
			const Eigen::VectorXd &pt_world_feet_est = m_stateEstimator->getStates();
			newTargetBbox = m_objModelProjector->getBboxOfProjectedCylinderFromWorldFeet(
				Eigen::Vector2d(pt_world_feet_est(0), pt_world_feet_est(1)), classId, -1
			);
#ifdef _WIN32
			NvMOTObjToTrack objToTrack{};
			objToTrack.classId = classId;
			objToTrack.bbox.x = newTargetBbox.min()(0);
			objToTrack.bbox.y = newTargetBbox.min()(1);
			objToTrack.bbox.width = newTargetBbox.sizes()(0);
			objToTrack.bbox.height = newTargetBbox.sizes()(1);
#else
			const NvMOTObjToTrack objToTrack = {
				.classId = classId,
				.bbox = {
					.x = newTargetBbox.min()(0),
					.y = newTargetBbox.min()(1),
					.width = newTargetBbox.sizes()(0),
					.height = newTargetBbox.sizes()(1)
				}
			};
#endif
			recovered = m_objModelProjector->getRecoveredObjectModelFromBbox(
				&objToTrack, frameNum, (int)m_targetInd, false
			);

			newTargetBbox = recovered.bbox;

			if(m_params->m_enableBboxUnClipping && getAge() > 0)
			{
				newTargetBbox = CV_FUNCS::unclipBbox(newTargetBbox, m_params->srcImageBox, m_targetInfo.tBbox);
			}
			const float imgBorderCroppingVisibility = CV_FUNCS::computeVisibility(newTargetBbox, m_params->srcImageBox);

			/// Update the target states with the estimated state back.
			m_targetInfo.update(frameNum, newTargetBbox, imgBorderCroppingVisibility);

			/// Update the targetInfo with the estimated world feet and covariance.
			if (m_objModelProjector->isTargetVisibleInImage(recovered.pt_world_feet, getTargetHeight()))
			{
				m_targetInfo.pt_world_feet = recovered.pt_world_feet.cast<float>();
				m_targetInfo.pt_world_feet_cov = getLocCovariance().cast<float>();
				m_targetInfo.pt_img_feet = recovered.pt_img_feet.cast<float>();
				m_targetInfo.convexHull = recovered.convexHull;
			}
			else
			{
				m_targetInfo.visibility = -1.f;
			}
		}
		else
		{
			/// Fall back to the 2D state-estimator model.
			/// Update the state estimator with its predicted one.
			m_stateEstimator->update(m_measList);

			newTargetBbox = CV_FUNCS::convertXToBbox(m_stateEstimator->getStates(), m_stateEstimator->useAspectRatio());

			if(m_params->m_enableBboxUnClipping && getAge() > 0)
			{
				newTargetBbox = CV_FUNCS::unclipBbox(newTargetBbox, m_params->srcImageBox, m_targetInfo.tBbox);
			}
			const float imgBorderCroppingVisibility = CV_FUNCS::computeVisibility(newTargetBbox, m_params->srcImageBox);

			/// Update the target states with the estimated state back.
			m_targetInfo.update(frameNum, newTargetBbox, imgBorderCroppingVisibility);
		}
	}

	m_measList.clear();
	m_targetInfo.pose25d.clear();

	/// Define search region from the current target location.
	m_searchRegion = defineSearchRegion(m_targetInfo.tBbox);

	/// Check if 3D model is used.
	if(m_objModelProjector != nullptr && m_objModelProjector->isInitialized())
	{
		const Eigen::VectorXd &pt_world_feet_pred = m_stateEstimator->getPredictedStates();

		const NvMotDataTypes::Rect2f newBbox = m_objModelProjector->getBboxOfProjectedCylinderFromWorldFeet(
			Eigen::Vector2d(pt_world_feet_pred(0), pt_world_feet_pred(1)), classId, -1
		);

		m_predictedSearchRegion = defineSearchRegion(newBbox);
	}
	else
	{
		/// Define the predicted search region from the current target location.
		m_predictedSearchRegion = defineSearchRegion(
			CV_FUNCS::convertXToBbox(m_stateEstimator->getPredictedStates(), m_stateEstimator->useAspectRatio())
		);
	}

	/// Update the tracker states Note: This should be consistent with the state transiation diagram depicted in DeepStream Doc.
	{
		/// This is the case only when a new target is added.
		if(newState == TRACKER_STATE::TENTATIVE)
		{
			/// Set to the TENTATIVE mode.
			if((int)getAge() <= m_params->m_probationAge)
			{
				setTrackerState(TRACKER_STATE::TENTATIVE);
			}
			else if(m_params->m_probationAge <= 1)
			{
				/// In case m_params->m_probationAge <= 1, then set the tracker state to ACTIVE from the beginning.
				setTrackerState(TRACKER_STATE::ACTIVE);
				resetShadowTrackingAge();

				/// Request to assign a target ID.
				m_needToSetTargetID = true;
			}
			else
			{
				throw std::runtime_error(
					"Invalid tracker state update attempt trackerInd = " + std::to_string(m_targetInd) + ", (age, probationAge) = " + std::to_string(getAge()) + ", " + std::to_string(m_params->m_probationAge)
				);
			}
		}
		else if(newState == TRACKER_STATE::ACTIVE)
		{
			if((getTrackerState() == TRACKER_STATE::TENTATIVE)
				|| (m_isOffline && (getTrackerState() == TRACKER_STATE::EMPTY)))
			{
				/// TODO: Determine whether visibility should gate unique ID assignment.
				if((int)getAge() > m_params->m_probationAge)
				{
					/// Request to assign a unique ID.
					m_needToSetTargetID = true;
					/// Indicate that this tracker has become active.
					m_becameActive = true;
					/// Persistent; DCF initialized here.
					m_hasBeenActive = true;

					setTrackerState(TRACKER_STATE::ACTIVE);
				}
				else
				{
				} // keep it to be in TENTATIVE mode
			}
			else
			{
				/// It was active, inactive or quasi-active.
				/// Indicate that this tracker has become active.
				if (getTrackerState() != TRACKER_STATE::ACTIVE)
				{
					m_becameActive = true;
					/// Persistent; once active, always flagged.
					m_hasBeenActive = true;
				}

				setTrackerState(TRACKER_STATE::ACTIVE);

				/// Remove the target from pending reassociation after it becomes ACTIVE.
				if((int)getShadowTrackingAge() >= m_params->m_probationAge)
				{
					m_needToRemoveFromPendingReAssocDB = true;
				}
			}

			/// Set the shadow tracking age.
			resetShadowTrackingAge();
		}
		else if(newState == TRACKER_STATE::INACTIVE)
		{
			/// Set the shadow tracking age.
			incrementShadowTrackingAge();

			/// Avoid tracklet projection to ReAssocDB if transitioned from quasi-active to inactive.
			if (getTrackerState() == TRACKER_STATE::QUASIACTIVE)
			{
				incrementShadowTrackingAge(m_params->m_maxShadowTrackingAge);
				setTrackerState(TRACKER_STATE::INACTIVE);
			}
			/// If TENTATIVE, keep the TENTATIVE mode until it gets ACTIVE.
			else if(getTrackerState() != TRACKER_STATE::TENTATIVE)
			{
				setTrackerState(TRACKER_STATE::INACTIVE);

				/// Prepare projected tracklet points once shadow age reaches probation age.
				if((int)getShadowTrackingAge() >= m_params->m_probationAge)
					m_ready4ReAssoc = true;
			}
		}
		/// Transition to quasi-active.
		else if(newState == TRACKER_STATE::QUASIACTIVE)
		{
			resetShadowTrackingAge();
			setTrackerState(TRACKER_STATE::QUASIACTIVE);
		}
		else
		{
			throw std::runtime_error("Invalid tracker state update attempt targetInd = " + std::to_string(m_targetInd));
		}
	}
}

/**
 * @brief Initialize a target tracked by a peer camera in QUASIACTIVE state.
 * @param[in] frameNum Current frame number.
 * @param[in] peerTarget Peer-camera target used to initialize target information.
 */
void NvTrackedObject::initializeQuasiActiveStates(const NvMOTFrameNum frameNum, const PeerTarget *peerTarget)
{
	NvMOTClassId classId = peerTarget->classId;
	m_targetIDTimeStamp = peerTarget->targetIDTimeStamp;
	m_targetID = peerTarget->targetId;
	m_acquiredID = true;

	Eigen::Vector2d peer_world_feet(peerTarget->nextState(0), peerTarget->nextState(1));
	m_measList.push_back(std::make_tuple(peer_world_feet, peerTarget->feetCov, true));

	/// Update the state estimator with the measurement.
	m_stateEstimator->update(m_measList);
	m_measList.clear();
	setTargetHeight(peerTarget->height, peerTarget->heightConfidence, true);

	const Eigen::Vector2d pt_world_feet_est(m_stateEstimator->getStates()(0), m_stateEstimator->getStates()(1));
	NvMotDataTypes::Rect2f newTargetBbox = m_objModelProjector->getBboxOfProjectedCylinderFromWorldFeet(
		pt_world_feet_est, classId, -1
	);

#ifdef _WIN32
	NvMOTObjToTrack objToTrack{};
	objToTrack.classId = classId;
	objToTrack.bbox.x = newTargetBbox.min()(0);
	objToTrack.bbox.y = newTargetBbox.min()(1);
	objToTrack.bbox.width = newTargetBbox.sizes()(0);
	objToTrack.bbox.height = newTargetBbox.sizes()(1);
#else
	const NvMOTObjToTrack objToTrack = {
		.classId = classId,
		.bbox = {
			.x = newTargetBbox.min()(0),
			.y = newTargetBbox.min()(1),
			.width = newTargetBbox.sizes()(0),
			.height = newTargetBbox.sizes()(1)
		}
	};
#endif

	RecoveredObjModel recovered = m_objModelProjector->getRecoveredObjectModelFromBbox(
		&objToTrack, frameNum, (int)m_targetInd, false
	);

	newTargetBbox = recovered.bbox;
	Eigen::Vector2d newImgFeet = recovered.pt_img_feet;
	std::vector<Eigen::Vector2f> newConvexHull = recovered.convexHull;

	/// Update the target states with the estimated state back.
	m_targetInfo.classId = classId;
	m_targetInfo.frameNum = frameNum;
	m_targetInfo.tBbox = newTargetBbox;
	m_targetInfo.origBbox = newTargetBbox;
	m_targetInfo.loc = Eigen::Vector2f(
		newTargetBbox.min()(0) + newTargetBbox.sizes()(0)/2.f, newTargetBbox.min()(1) + newTargetBbox.sizes()(1)/2.f
	);
	m_targetInfo.visibility = CV_FUNCS::computeVisibility(newTargetBbox, m_params->srcImageBox);
	m_targetInfo.imgBorderCroppingVisibility = m_targetInfo.visibility;
	m_targetInfo.distFromCam = std::numeric_limits<float>::max();
	m_targetInfo.shadowTrackingAge = 0;
	m_targetInfo.age = peerTarget->age;
	m_targetInfo.trackerState = TRACKER_STATE::QUASIACTIVE;
	m_targetInfo.pt_world_feet = Eigen::Vector2f(pt_world_feet_est(0), pt_world_feet_est(1));
	m_targetInfo.pt_world_feet_cov = getLocCovariance().cast<float>();
	m_targetInfo.pt_img_feet = newImgFeet.cast<float>();
	m_targetInfo.convexHull = newConvexHull;

	m_peerTargets.push_back(*peerTarget);

	if (m_params->m_debugVerboseLevel > 4)
	{
		std::cout << "[NvTrackedObject::initializeQuasiActiveStates()] StreamInd["
				  << (getVideoStreamId() >> 32) <<  "], target ID[" << getTargetID()
				  << "], world feet: [" << pt_world_feet_est(0) << ", " << pt_world_feet_est(1) << "]\n";
	}
}

/**
 * @brief Update localization state from the current tracker result.
 * @param[in] frameNum Current frame number.
 * @param[in] trackingConfidence Current tracking confidence.
 * @param[in] localizationBbox Localized bounding box.
 */
void NvTrackedObject::setLocalizationResults(
	const NvMOTFrameNum frameNum, const float trackingConfidence, const NvMotDataTypes::Rect2f& localizationBbox
)
{
	m_targetInfo.confidence = trackingConfidence;

	/// Check if the correlation response is above a threshold.
	if(trackingConfidence >= m_params->m_minTrackerConfidence)
	{
		/// Update the target location with the newly found location and confidence.
		m_targetInfo.setLocation(localizationBbox);

		/// Check if 3D model is used.
		if(m_objModelProjector != nullptr && m_objModelProjector->isInitialized())
		{
			Eigen::Vector2d tLoc; tLoc << m_targetInfo.loc(0), m_targetInfo.loc(1);
			Eigen::Vector2d tFeet = m_objModelProjector->getWorldFeetPtOnGroundFromCamWaistPt(
				Eigen::Vector2d(tLoc(0), tLoc(1)), getClassId(), (int)m_targetInd
			);

			/// Add the localization result as a measurement to the state estimator.
			m_measList.push_back(std::make_tuple(tFeet, -Eigen::Matrix2d::Identity(), false));
		}
		else
		{
			/// Convert the detection bbox into measurement states.
			Eigen::VectorXd trackBbox = CV_FUNCS::convertBboxToZ(localizationBbox, m_stateEstimator->useAspectRatio());

			/// Add the localization result as a measurement to the state estimator.
			m_measList.push_back(std::make_tuple(trackBbox, -Eigen::Matrix2d::Identity(), false));
		}
	}
	else
	{
	} // in case the tracking confidence is lower than a threshold, let it become INACTIVE
}

/** @brief Add current target information to the past-frame reporting buffer. */
void NvTrackedObject::addPastFrameData()
{
	TargetInfo targetInfo = m_targetInfo;

	/// The pointer to a detector object will be invalid in the later frame So, need to set to nullptr.
	targetInfo.matchedDetObjInfo = nullptr;

	while (!m_pastFrameDataBuffer.empty() && int(m_pastFrameDataBuffer.size()) >= m_params->m_maxShadowTrackingAge)
	{
		m_pastFrameDataBuffer.pop_front();
	}

	m_pastFrameDataBuffer.push_back(std::move(targetInfo));
}

/**
 * @brief Append adopted past-frame objects to this target's buffer.
 * @param[in] pastFrameBuffer Past-frame objects appended to this target's buffer.
 */
void NvTrackedObject::adoptPastFrameData(const std::deque<TargetInfo> &pastFrameBuffer)
{
	while (!m_pastFrameDataBuffer.empty() && int(m_pastFrameDataBuffer.size()) >= m_params->m_maxShadowTrackingAge - int(pastFrameBuffer.size()))
	{
		m_pastFrameDataBuffer.pop_front();
	}

	for (uint ind = 0; ind < pastFrameBuffer.size(); ind++) {
		m_pastFrameDataBuffer.push_back(pastFrameBuffer.at(ind));
	}
}

/**
 * @brief Remove trajectory points newer than a specified frame.
 * @param[in] frameNum Newest frame number that is kept; later points are removed.
 */
void NvTrackedObject::purgeTrajectoryBufferUntilSpecified(const NvMOTFrameNum frameNum)
{
	auto rit = m_trajectoryBuffer.rbegin();
	while(rit != m_trajectoryBuffer.rend())
	{
		if(rit->frameNum <= frameNum)
		{
			break;
		}

		++rit;
	}

	if(rit != m_trajectoryBuffer.rend())
	{
		auto it = rit.base();
		while(it != m_trajectoryBuffer.end())
		{
			if(m_params->m_debugVerboseLevel > 1 && it->trackerState == TRACKER_STATE::ACTIVE)
			{
				printf(
					"[NvTrackedObject::purgeTrajectoryBufferUntilSpecified()][Warning] Trying to purge ACTIVE trajectory points  !!\n"
				);
			}

			it = m_trajectoryBuffer.erase(it);
		}
	}
}

/** @brief Add the current target information to the trajectory buffer. */
void NvTrackedObject::updateTrajectoryBuffer()
{
	/// Make sure we keep the latest trajectory point.
	if(m_params->m_maxTrajectoryBufferLength >= 0
			&& m_trajectoryBuffer.size() > 1
			&& m_trajectorySkipInterval > 1
			&& ((m_trajectoryBuffer.rbegin()->frameNum - m_trajectoryBuffer.begin()->frameNum) % m_trajectorySkipInterval) != 0)
		m_trajectoryBuffer.pop_back();

	m_trajectoryBuffer.push_back(m_targetInfo);

	/// Subsample trajectories longer than the configured limit to reduce their length by half.
	if(m_params->m_maxTrajectoryBufferLength >= 0 && (int)m_trajectoryBuffer.size() > m_params->m_maxTrajectoryBufferLength)
	{
		m_trajectorySkipInterval *= 2;

		auto it = m_trajectoryBuffer.begin() ;

		if(m_params->m_debugVerboseLevel > 2)
		{
			cout << "[NvTrackedObject::updateTrajectoryBuffer()][Trajectory Sub-Sampling] TargetID[" << getTargetID()
				<< "] FrameNum[" << m_targetInfo.frameNum << "] (first frame, last frame, length) = ("
				<< m_trajectoryBuffer.begin()->frameNum << ", " << m_trajectoryBuffer.rbegin()->frameNum
				<< ", " << m_trajectoryBuffer.size() << ") --> ";
		}

		/// Check if the erased one is in ACTIVE state. If so, mark the next sub-sampled one as the ACTIVE one , because otherwise we don't know if it is a validated track point and it may be discarded during the trajectory post-processing.
		TRACKER_STATE trackerState_prev = TRACKER_STATE::EMPTY;

		/// Don't erase the first and the last element.
		++it;
		while( it != m_trajectoryBuffer.end() && std::next(it) != m_trajectoryBuffer.end() )
		{
			if(((it->frameNum - m_trajectoryBuffer.begin()->frameNum) % m_trajectorySkipInterval) != 0)
			{
				if(trackerState_prev == TRACKER_STATE::EMPTY && it->trackerState == TRACKER_STATE::ACTIVE)
					trackerState_prev = it->trackerState;

				it = m_trajectoryBuffer.erase(it) ;
			}
			else
			{
				/// Carry the ACTIVE state when sub-sampled.
				if(it->trackerState != TRACKER_STATE::ACTIVE && trackerState_prev == TRACKER_STATE::ACTIVE)
					(*it).trackerState = trackerState_prev;

				trackerState_prev = TRACKER_STATE::EMPTY;

				++it;
			}
		}

		if(m_params->m_debugVerboseLevel > 2)
		{
			cout << " (" << m_trajectoryBuffer.begin()->frameNum << ", " << m_trajectoryBuffer.rbegin()->frameNum
					<< ", " << m_trajectoryBuffer.size() << "): " << endl;

			if(m_params->m_debugVerboseLevel > 3)
			{
				cout << "  -- frameNums: ";
				for(auto it=m_trajectoryBuffer.begin(); it != m_trajectoryBuffer.end(); ++it)
				{
					cout << it->frameNum << " ";
				}
				cout << endl;
			}
		}
	}
}

/**
 * @brief Initialize the target manager from a tracker configuration.
 * @param[in] configFilePath Path to the tracker configuration file.
 */
NvTrackedObjectManager::NvTrackedObjectManager(const std::string &configFilePath)
: m_configFilePath(configFilePath)
{
	/// Initialize Eigen for multithreaded use.
	Eigen::initParallel();

	/// Parse config file for base tracker module.
	m_params = NvTrackerParamsTargetManager(m_configFilePath, "TargetManagement");
	/// Obtain the multi-view associator parameters.
	m_multiViewParams = MultiViewAssociatorParams(configFilePath, "MultiViewAssociator");

	m_defaultModelProjectors.clear();

	m_trajectoryManager = std::make_shared<NvTrajectoryManagerOnline>(m_configFilePath);

	memset(&m_segInfo, 0, sizeof(SegmentationInBatch));

	m_isInit = false;
}

/** @brief Release all managed targets and stream submodules. */
NvTrackedObjectManager::~NvTrackedObjectManager()
{
	removeAllStreams();

	m_trajectoryManager.reset();

	if (m_params.m_debugVerboseLevel >= 1) {
		fprintf(
			stderr,
			"[NvTrackedObjectManager] term_reg_total=%llu term_reg_was_active=%llu" " shadow_1=%llu shadow_2=%llu shadow_3=%llu shadow_4=%llu shadow_5=%llu" " shadow_6=%llu shadow_7=%llu shadow_8=%llu shadow_9plus=%llu" " young=%llu mature=%llu" " rescue_6=%llu rescue_7=%llu rescue_8plus=%llu" " tent_total=%llu tent_le2=%llu tent_3_4=%llu tent_5_7=%llu tent_8_10=%llu tent_11plus=%llu\n",
			(unsigned long long)m_tm_stats.term_reg_total, (unsigned long long)m_tm_stats.term_reg_was_active,
			(unsigned long long)m_tm_stats.term_reg_shadow_1, (unsigned long long)m_tm_stats.term_reg_shadow_2,
			(unsigned long long)m_tm_stats.term_reg_shadow_3, (unsigned long long)m_tm_stats.term_reg_shadow_4,
			(unsigned long long)m_tm_stats.term_reg_shadow_5, (unsigned long long)m_tm_stats.term_reg_shadow_6,
			(unsigned long long)m_tm_stats.term_reg_shadow_7, (unsigned long long)m_tm_stats.term_reg_shadow_8,
			(unsigned long long)m_tm_stats.term_reg_shadow_9plus, (unsigned long long)m_tm_stats.term_reg_young,
			(unsigned long long)m_tm_stats.term_reg_mature, (unsigned long long)m_tm_stats.shadow_rescue_6,
			(unsigned long long)m_tm_stats.shadow_rescue_7, (unsigned long long)m_tm_stats.shadow_rescue_8plus,
			(unsigned long long)m_tm_stats.term_tent_total, (unsigned long long)m_tm_stats.term_tent_age_le2,
			(unsigned long long)m_tm_stats.term_tent_age_3_4, (unsigned long long)m_tm_stats.term_tent_age_5_7,
			(unsigned long long)m_tm_stats.term_tent_age_8_10, (unsigned long long)m_tm_stats.term_tent_age_11plus
		);
	}

	if(m_params.m_debugVerboseLevel > 1)
		cout << "[NvTrackedObjectManager] De-initialized" << endl;
}

/**
 * @brief Allocate the target pool and initialize stream-independent submodules.
 * @param[in] frameInfo Source frame sizes and stream limits used to allocate the target pool.
 */
void NvTrackedObjectManager::initialize(const SourceFrameInfo &frameInfo)
{
	m_params.originalSrcImageBox = CV_FUNCS::constructRect2f(
		0, 0, frameInfo.originalSrcImageSize.width, frameInfo.originalSrcImageSize.height
	);
	m_params.srcImageBox = CV_FUNCS::constructRect2f(0,0,frameInfo.srcImageSize.width, frameInfo.srcImageSize.height);
	m_params.m_maxTargetsToTrack = frameInfo.maxStreams*m_params.m_maxTargetsPerStream;

	m_trajectoryManager->setSrcFrameInfo(m_params.originalSrcImageBox, m_params.srcImageBox);

	if(m_params.m_maxTargetsPerStream == 0)
	{
		throw std::invalid_argument(
			"[NvTrackedObjectManager] Invalid maxTargetsPerStream: " + std::to_string(m_params.m_maxTargetsPerStream)
		);
	}

	/// Create a pool of tracked objects.
	for(TrackerIndex targetInd=0; targetInd<m_params.m_maxTargetsToTrack; targetInd++)
	{
		std::shared_ptr<NvTrackedObject> target = std::make_shared<NvTrackedObject>(
			m_configFilePath, &m_params, targetInd
		);
		m_TrackedObjectsInPool[targetInd] = target;
	}

	/// Multi-view associators are created when a stream is added.
	m_lastTrackerInd = -1;
	m_targetCapacityUtilization = 0.f;

	m_isInit = true;
	if(m_params.m_debugVerboseLevel > 1)
		cout << "[NvTrackedObjectManager] Initialized" << endl;
}

/**
 * @brief Populate target-manager memory requirements.
 * @param[out] paramsQuery Query structure populated with target-manager requirements.
 */
void NvTrackedObjectManager::queryParams(NvMOTQuery &paramsQuery)
{
	paramsQuery.maxTargetsPerStream = m_params.m_maxTargetsPerStream;
	paramsQuery.maxShadowTrackingAge = m_params.m_maxShadowTrackingAge;
	paramsQuery.outputTerminatedTracks = m_params.m_saveTerminatedTracks;
	paramsQuery.maxTrajectoryBufferLength = m_params.m_maxTrajectoryBufferLength;
	paramsQuery.outputShadowTracks = m_params.m_saveShadowTracks;
	paramsQuery.outputVisibility = m_params.m_outputVisibility;
}

/** @brief Remove all trackers for all streams. */
void NvTrackedObjectManager::removeAllStreams()
{
	/// Remove all the streams and terminate all the trackers accordingly.
	for(auto it = m_TrackedObjectsInUse.begin(); it != m_TrackedObjectsInUse.end();)
	{
		auto it_next = std::next(it);
		removeStream(it->first);
		it = it_next;
	}
}

/** @brief Update target capacity utilization. */
void NvTrackedObjectManager::updateTargetCapacityUtilization()
{
	m_targetCapacityUtilization = std::max(0, getLastTargetInd()+1) / (float)m_params.m_maxTargetsToTrack * 100.f;
}

/**
 * @brief Assign IDs, perform reassociation, and terminate targets for one stream.
 * @param[in] assocOut Association results for the stream whose trajectories are updated.
 */
void NvTrackedObjectManager::updateTrajectoryManagerPerStream(const assocDataOut_t& assocOut)
{

	const NvMOTStreamId streamId = assocOut.videoStreamId;
	const NvMOTFrameNum frameNum = assocOut.frameNum;

	/// The list of targets to be terminated because they are fused.
	std::map<std::shared_ptr<NvTrackedObject>, std::shared_ptr<NvTrackedObject>> toBeFused;

	/// Assign new Target IDs if needed and fuse them if re-assoc is done.
	m_trajectoryManager->assignTargetIDs(getTrackedObjectsInUseList(streamId), toBeFused, assocOut.timeMeta);

	/// Terminate fused trackers and record their peers for reid feature gallery fusion.
	if(toBeFused.size() > 0)
	{
		for(auto it = toBeFused.begin(); it != toBeFused.end(); ++it)
		{
			terminateTracker(it->first, FUSED_BY_REASSOCIATION, it->second);
		}
	}

	/// Discard too old tracklets from the archive.
	m_trajectoryManager->cleanUpArchivedTrajectories_perStream(streamId, frameNum);

	/// Sync the MV associators and communicators.
	if (m_multiViewAssociators.find(streamId) != m_multiViewAssociators.end())
	{
		MEAS_TIME(
			m_multiViewAssociators.at(streamId)->update(assocOut.timeMeta), m_params.m_debugVerboseLevel > 1,
			(std::string("update stream ") + std::to_string(streamId >> 32)).c_str()
		);
	}
	else
	{
		std::cout << "[NvTrackedObjectManager::updateTrajectoryManagerPerStream][Warning] Stream ID not found in multi-view associator: " << streamId << std::endl;
	}
}

/**
 * @brief Update targets and trajectories for one stream in a worker thread.
 * @param[in] assocOut Association output for one stream.
 * @param[in] reid4reAssocState Re-identification state that selects target update, trajectory update, or both.
 */
void NvTrackedObjectManager::updatePerStream(const assocDataOut_t& assocOut, const REID4REASSOC_STATE reid4reAssocState)
{
	/// If preserving update order, store association results for main thread to update later.
	/// Reid for reassoc NOT_NEEDED: call this function only once.
	/// Reid for reassoc needed: call this function twice; association is performed in the first call (reid4reAssocState == TO_DO), so only store that.
	if (m_params.m_preserveStreamUpdateOrder)
	{
		if (reid4reAssocState == NOT_NEEDED || reid4reAssocState == TO_DO)
		{
			std::unique_lock<std::mutex> assocLock(m_trackedObjectsLock);
			m_assocOuts[assocOut.videoStreamId] = assocOut;
			assocLock.unlock();
		}
	}

	else
	{
		/// If not preserving update order, update targets and trajectory in each thread.
		/// Reid for reassoc NOT_NEEDED: update targets and traj in one call.
		/// Reid for reassoc needed: update targets in the first call; then do reid in nvmultiobjecttracker; then update traj in the second call.
		if (reid4reAssocState == NOT_NEEDED || reid4reAssocState == TO_DO)
		{
			updateTargetsPerStream(assocOut);
		}

		if (reid4reAssocState == NOT_NEEDED || reid4reAssocState == DONE)
		{
			updateTrajectoryManagerPerStream(assocOut);
		}
	}
}

/** @brief Update targets for all streams on the main thread. */
void NvTrackedObjectManager::updateTargets()
{
	/// Update targets before ReID on the main thread when stream update order is preserved.
	if (m_params.m_preserveStreamUpdateOrder)
	{
		for (auto streamIt = m_assocOuts.begin(); streamIt != m_assocOuts.end(); streamIt++)
		{
			updateTargetsPerStream(streamIt->second);
		}
	}
}

/** @brief Update trajectories for all streams on the main thread. */
void NvTrackedObjectManager::updateTrajectoryManager()
{
	/// Update trajectories after ReID on the main thread when stream update order is preserved.
	if (m_params.m_preserveStreamUpdateOrder)
	{
		for (auto streamIt = m_assocOuts.begin(); streamIt != m_assocOuts.end(); streamIt++)
		{
			updateTrajectoryManagerPerStream(streamIt->second);
		}
	}
	/// Update global capacity utilization after target and trajectory updates are done.
	/// Save the shadow-track information.
	saveShadowTrackInfo();
	updateTargetCapacityUtilization();
}

/**
 * @brief Perform target updates for one stream.
 * @param[in] assocOut Association results used to update targets on one stream.
 */
void NvTrackedObjectManager::updateTargetsPerStream(const assocDataOut_t& assocOut)
{
	dumpIntermediateTargetInfo(assocOut);

	if (assocOut.isAssocDone)
	{
		/// For matched pairs, update target states.
		updateMatched(assocOut.videoStreamId, assocOut.frameNum, assocOut.matchedPairs);

		/// For unmatched targets, update target states.
		updateUnMatched(assocOut.videoStreamId, assocOut.frameNum, assocOut.unmatchedTargets);

		/// For new targets, create new target trackers.
		addTargetsBatch(assocOut.videoStreamId, assocOut.frameNum, assocOut.unmatchedDetObjs);

		/// Add see-through targets in quasi-active mode.
		addSeeThroughTargets(assocOut.videoStreamId, assocOut.frameNum);
	}
	else
	{
		const NvTrackedObjectList targetList = getTrackedObjectsInUseList(assocOut.videoStreamId);

		for(auto targetIt = targetList.begin(); targetIt != targetList.end(); ++targetIt)
		{
			std::shared_ptr<NvTrackedObject> target = targetIt->second;

			target->updateTargetStates(assocOut.frameNum);

			/// Check and terminate tracker if needed.
			checkToTerminateTracker(target);
		}
	}
	/// Update the trajectory buffer.
	updateTrajectoryBuffers(assocOut.videoStreamId);
}

/** @brief Save current-batch shadow-track objects. */
void NvTrackedObjectManager::saveShadowTrackInfo()
{
	if(! m_params.m_saveShadowTracks )
	{
		return;
	}

	for (auto streamIt = m_TrackedObjectsInUse.begin(); streamIt != m_TrackedObjectsInUse.end(); streamIt++)
	{
		auto targetList = streamIt->second;
		if (targetList)
		{
			for( auto targetIt = targetList->begin(); targetIt != targetList->end(); ++targetIt)
			{
				addTargetInShadowTrack(targetIt->second);
			}
		}
	}
}

/**
 * @brief Save an inactive target in the current batch's shadow-track set.
 * @param[in] target Inactive target recorded when its shadow-tracking age is positive.
 */
void NvTrackedObjectManager::addTargetInShadowTrack(std::shared_ptr<NvTrackedObject> target)
{
	if( (target->getTrackerState() == TRACKER_STATE::INACTIVE) &&
		(target->getShadowTrackingAge() > 0)
	)
	{
		m_shadowTracks[target->getVideoStreamId()].insert(target);
	}
}

/** @brief Clear target-manager data retained from the previous batch. */
void NvTrackedObjectManager::cleanPreviousBatchData()
{
	m_trackedObjectsTerminated.clear();
	m_trackedObjectsFused.clear();

	clearOutputTerminatedTracks();
	clearShadowTracks();
}

/**
 * @brief Update per-stream input information before target-manager processing.
 * @param[in] inputMeta Batch of input frames keyed by stream ID.
 */
void NvTrackedObjectManager::updateInputInfo(const std::map<NvMOTStreamId, NvMOTFrame *> &inputMeta)
{
	/// If preserving stream update order, clean previous assoc results.
	if (m_params.m_preserveStreamUpdateOrder)
	{
		m_assocOuts.clear();
	}

	/// If there are new streams, add to m_TrackedObjectsInUse and m_trajectoryManager.
	std::vector<NvMOTStreamId> streamIdsToAdd;
	for(auto streamIt = inputMeta.begin(); streamIt != inputMeta.end(); streamIt++)
	{
		NvMOTStreamId videoStreamId = streamIt->first;
		if (m_TrackedObjectsInUse.find(videoStreamId) == m_TrackedObjectsInUse.end() || (!m_TrackedObjectsInUse.at(videoStreamId)))
		{
			m_TrackedObjectsInUse[videoStreamId] = std::make_shared<NvTrackedObjectList>();
		}
		streamIdsToAdd.push_back(videoStreamId);
	}
	m_trajectoryManager->updateStreamIds(streamIdsToAdd);
}

/**
 * @brief Append current target information to each trajectory on a stream.
 * @param[in] videoStreamId Stream whose in-use targets append the current frame.
 */
void NvTrackedObjectManager::updateTrajectoryBuffers(const NvMOTStreamId videoStreamId)
{
	const NvTrackedObjectList targetList = getTrackedObjectsInUseList(videoStreamId);

	if(targetList.size() > 0)
		for(auto targetIt = targetList.begin(); targetIt != targetList.end(); ++targetIt)
		{
			std::shared_ptr<NvTrackedObject> target = targetIt->second;

			target->updateTrajectoryBuffer();
		}
}

/**
 * @brief Update object model projectors and associators when streams are added or deleted.
 * @param[in] inputMeta Input frames keyed by stream ID.
 * @param[in] srcImageSize Source image size.
 * @return True when at least one new stream is added.
 */
bool NvTrackedObjectManager::addStreamSubModules(
	const std::map<NvMOTStreamId, NvMOTFrame *> &inputMeta, const NvMotDataTypes::Size& srcImageSize
)
{
	bool newStreamAdded = false;
	for(auto streamIt = inputMeta.begin(); streamIt != inputMeta.end(); ++streamIt)
	{
		const NvMOTStreamId &videoStreamId = streamIt->first;

		/// When a new stream is added, initialize obj model projector.
		if(m_defaultModelProjectors.find(videoStreamId) == m_defaultModelProjectors.end())
		{
			std::string sensorName;
			if (streamIt->second && streamIt->second->sensorName)
			{
				sensorName = streamIt->second->sensorName;
			}
			m_defaultModelProjectors[videoStreamId] = std::make_shared<ObjectModelProjector>(
				m_configFilePath, srcImageSize, videoStreamId, sensorName
			);
			newStreamAdded = true;
		}
	}

	if (newStreamAdded)
	{
		createMultiViewAssociators();
	}

	return newStreamAdded;
}

/**
 * @brief Recover an object model from a detection bounding box using the default projector.
 * @param[in] videoStreamId Video stream ID.
 * @param[in] frameNum Current frame number.
 * @param[in,out] detObjInfo Detected-object information updated with the recovered model.
 */
void NvTrackedObjectManager::recoverDefaultModelFromBbox(
	const NvMOTStreamId videoStreamId, const NvMOTFrameNum frameNum, DetectedObjInfo &detObjInfo
)
{
	if (m_defaultModelProjectors.find(videoStreamId) == m_defaultModelProjectors.end()
		|| (!m_defaultModelProjectors.at(videoStreamId)->isInitialized()))
	{
		return;
	}

	/// Deep copy so we don't modify the default projector (it holds multi-class object models for the stream).
	const std::shared_ptr<ObjectModelProjector> &defaultProj = m_defaultModelProjectors.at(videoStreamId);
	std::shared_ptr<ObjectModelProjector> proj = std::make_shared<ObjectModelProjector>(*defaultProj);
	if (detObjInfo.objIn->classId != proj->getTargetClassId())
	{
		if (m_params.m_debugVerboseLevel >= 5)
			std::cout << "[setTargetModelForClass] from NvTrackedObjectManager::recoverDefaultModelFromBbox, classId before=" << proj->getTargetClassId() << " after=" << detObjInfo.objIn->classId << std::endl;
		proj->setTargetModelForClass(detObjInfo.objIn->classId);
	}

	RecoveredObjModel objRecovered = proj->getRecoveredObjectModelFromBbox(detObjInfo.objIn, frameNum, -1, true);

	detObjInfo.bbox = objRecovered.bbox;
	detObjInfo.loc = objRecovered.pt_img_center.cast<float>();
	detObjInfo.pt_img_feet = objRecovered.pt_img_feet.cast<float>();
	detObjInfo.pt_world_feet = objRecovered.pt_world_feet.cast<float>();
	detObjInfo.visibility = objRecovered.visibility;
	detObjInfo.distFromCam = objRecovered.distFromCam;
	detObjInfo.convexHull = objRecovered.convexHull;
}

/** @brief Replace multi-view associators after stream addition or deletion. */
void NvTrackedObjectManager::createMultiViewAssociators()
{
	/// Make a deep copy.
	std::map<NvMOTStreamId, std::shared_ptr<ICommunicator>> commMap;
	system_clock::time_point start = system_clock::now();
	m_multiViewAssociators.clear();
	for (auto it = m_defaultModelProjectors.begin(); it != m_defaultModelProjectors.end(); ++it)
	{
		const NvMOTStreamId &videoStreamId = it->first;
		/// Initialize the m_multiViewAssociators from the factory.
		m_multiViewAssociators[videoStreamId] = MultiViewAssociatorFactory::getInstance().makeMultiViewAssociator(
			m_configFilePath
		);
		/// Initialize m_TrackedObjectsInUse as empty lists.
		if (m_TrackedObjectsInUse.find(videoStreamId) == m_TrackedObjectsInUse.end() || m_TrackedObjectsInUse.at(videoStreamId) == nullptr)
		m_TrackedObjectsInUse[videoStreamId] = std::make_shared<NvTrackedObjectList>();
		commMap[videoStreamId] = m_multiViewAssociators[videoStreamId]->getCommunicator();
	}
	/// Initialize the multi-view associators.
	for (auto it = m_defaultModelProjectors.begin(); it != m_defaultModelProjectors.end(); ++it)
	{
		const NvMOTStreamId &videoStreamId = it->first;
		m_multiViewAssociators[videoStreamId]->initialize(
			videoStreamId, start, commMap, m_TrackedObjectsInUse.at(videoStreamId),
			m_defaultModelProjectors.at(videoStreamId)
		);
	}

	m_trajectoryManager->setMultiViewAssociators(m_multiViewAssociators);
}

/**
 * @brief Apply supported runtime target-manager configuration updates.
 * @param[in] fn Parsed YAML structure of config to update.
 */
void NvTrackedObjectManager::updateParams(const YAML::Node &fn)
{

	// [Target management]
	if (fn["TargetManagement"] && fn["TargetManagement"].IsMap())
	{
		m_params.read(fn["TargetManagement"]);
	}

	// [Trajectory management]
	if (fn["TrajectoryManagement"] && fn["TrajectoryManagement"].IsMap())
	{
		m_trajectoryManager->updateParams(fn["TrajectoryManagement"]);
	}
}

/**
 * @brief Update targets matched to detector objects.
 * @param[in] videoStreamId Video stream ID.
 * @param[in] frameNum Current frame number.
 * @param[in] matchedPairs Matched pairs as detected object and target index.
 */
void NvTrackedObjectManager::updateMatched(
	const NvMOTStreamId videoStreamId, const NvMOTFrameNum frameNum,
	const std::deque<std::pair<const DetectedObjInfo *, TrackerIndex>> &matchedPairs
)
{
	if(matchedPairs.size() == 0)
		return;

	/// For matched pairs, update target states.
	for(auto it = matchedPairs.begin(); it != matchedPairs.end(); ++it)
	{
		const DetectedObjInfo *detObj = (*it).first;
		const TrackerIndex targetInd = (*it).second;
		const std::shared_ptr<NvTrackedObject> target = m_TrackedObjectsInUse.at(videoStreamId)->at(targetInd);

		TRACKER_STATE newState = TRACKER_STATE::ACTIVE;
		if (m_params.m_useVOTBasedTargetUpdate)
		{
			newState = target->getTrackerConfidence() < m_params.m_minTrackerConfidence ? TRACKER_STATE::INACTIVE : TRACKER_STATE::ACTIVE;
			if(m_params.m_debugVerboseLevel > 2)
			{
				CLOG(
					std::string( "Stream[" + std::to_string(videoStreamId) + "] frameNum[" + std::to_string(frameNum) + "] Update matched target: (trackerInd) = (" + std::to_string(targetInd) + ") tracker confidence: " + std::to_string(target->getTrackerConfidence()) ).c_str()
				);
			}
		}
		else
		{
			newState = ACTIVE;
		}
		/// Capture INACTIVE-to-ACTIVE rescues at extended shadow ages before updateTargetStates resets the age.
		const bool prevWasInactive    = (target->getTrackerState() == TRACKER_STATE::INACTIVE);
		const bool prevHasBeenActive  = target->hasBeenActive();
		const uint32_t prevShadowAge  = prevWasInactive ? (uint32_t)target->getShadowTrackingAge() : 0u;
		target->updateTargetStates(frameNum, detObj, newState);
		if (prevWasInactive && prevHasBeenActive && prevShadowAge >= 6) {
			if      (prevShadowAge == 6) m_tm_stats.shadow_rescue_6++;
			else if (prevShadowAge == 7) m_tm_stats.shadow_rescue_7++;
			else                         m_tm_stats.shadow_rescue_8plus++;
		}
	}
}

/**
 * @brief Update targets without matched detector objects.
 * @param[in] videoStreamId Stream that owns the unmatched targets.
 * @param[in] frameNum Current frame number.
 * @param[in] unmatchedTargets Target indexes with no matched detection.
 */
void NvTrackedObjectManager::updateUnMatched(
	const NvMOTStreamId videoStreamId, const NvMOTFrameNum frameNum, const std::set<TrackerIndex> &unmatchedTargets
)
{
	if(unmatchedTargets.size() == 0)
		return;

	/// For unmatched targets, update target states.
	for(auto it = unmatchedTargets.begin(); it != unmatchedTargets.end(); ++it)
	{
		const TrackerIndex targetInd = *it;
		const std::shared_ptr<NvTrackedObject> target = m_TrackedObjectsInUse.at(videoStreamId)->at(targetInd);

		TRACKER_STATE newState = TRACKER_STATE::INACTIVE;
		if (m_params.m_useVOTBasedTargetUpdate)
		{
			if (target->getTrackerState() == TRACKER_STATE::TENTATIVE)
			{
				newState = TRACKER_STATE::INACTIVE;
			}
			else
			{
				newState = target->getTrackerConfidence() < m_params.m_minTrackerConfidence ? TRACKER_STATE::INACTIVE : TRACKER_STATE::ACTIVE;
			}
		}
		else
		{
			newState = INACTIVE;
		}

		target->updateTargetStates(frameNum, nullptr, newState);

		/// Check and terminate tracker if needed.
		checkToTerminateTracker(target);
	}
}

/**
 * @brief Check a target's termination criteria and terminate it when required.
 * @param[in] target Target whose shadow age is checked against the termination limits.
 */
void NvTrackedObjectManager::checkToTerminateTracker(const std::shared_ptr<NvTrackedObject> target)
{
	/// Terminate if the criteria is met.
	/// ACTIVE-lineage tracks (hasBeenActive=true) use the larger of m_maxShadowTrackingAge and m_maxActiveShadowTrackingAge as their shadow cap.
	const int shadowCap = (target->hasBeenActive() &&
						   m_params.m_maxActiveShadowTrackingAge > m_params.m_maxShadowTrackingAge)
		? m_params.m_maxActiveShadowTrackingAge
		: m_params.m_maxShadowTrackingAge;
	if((int)target->getShadowTrackingAge() > shadowCap)
	{
		terminateTracker(target, REGULAR_TERMINATION);
	}
	else if(m_params.m_earlyTerminationAge > 0 && target->getTrackerState() == TRACKER_STATE::TENTATIVE && (int)target->getShadowTrackingAge() >= m_params.m_earlyTerminationAge)
	{
		terminateTracker(target, EARLY_TERMINATION);
	}
}

/**
 * @brief Add new targets for unmatched detector objects.
 * @param[in] videoStreamId Stream that owns the new detections.
 * @param[in] frameNum Current frame number.
 * @param[in] unmatchedDetObjs Detections that were not matched to an existing target.
 */
void NvTrackedObjectManager::addTargetsBatch(
	const NvMOTStreamId videoStreamId, const NvMOTFrameNum frameNum,
	const std::map<const TrackerIndex, const DetectedObjInfo*> &unmatchedDetObjs
)
{
	if(unmatchedDetObjs.size() == 0)
		return;

	/// For new targets, create new target trackers.
	for(auto detIt = unmatchedDetObjs.begin(); detIt != unmatchedDetObjs.end(); ++detIt)
	{
		const DetectedObjInfo *detObj = detIt->second;

		/// Add a new target.
		if(detObj->confidence >= m_params.m_minDetectorConfidence && addTarget(videoStreamId, frameNum, detObj))
		{
		} // failed to add a new target
	}
}

/**
 * @brief Add see-through targets from peer cameras.
 * @param[in] videoStreamId Video stream ID.
 * @param[in] frameNum Current frame number.
 */
void NvTrackedObjectManager::addSeeThroughTargets(const NvMOTStreamId videoStreamId, const NvMOTFrameNum frameNum)
{
	/// Check if there is a multi-view associator entry for the given stream.
	if (m_multiViewAssociators.find(videoStreamId) == m_multiViewAssociators.end()) return;
	if (!m_multiViewParams.m_enableSeeThrough) return;

	/// Get the multi-view associator.
	std::shared_ptr<IMultiViewAssociator> mvAssociator = m_multiViewAssociators.at(videoStreamId);

	/// Check if the multi-view associator is valid.
	if (mvAssociator == nullptr) return;

	std::map<TargetId, std::shared_ptr<NvTrackedObject>> egoTargetByID;

	/// Get the ego target IDs.
	for (auto target : getTrackedObjectsInUseList(videoStreamId))
	{
		egoTargetByID[target.second->getTargetID()] = target.second;
	}
	/// Do not consider TENTATIVE targets (if any).
	egoTargetByID.erase(-1);

	/// Collect valid peer targets with their scores.
	std::vector<std::pair<float, const PeerTarget*>> scoredPeerTargets;

	PeerTargetDB peerTargetDB = mvAssociator->getPeerTargetDB();
	for(auto peerTargetIts = peerTargetDB.begin(); peerTargetIts != peerTargetDB.end(); ++peerTargetIts)
	{
		for (auto peerTargetIt = peerTargetIts->second.begin(); peerTargetIt != peerTargetIts->second.end(); ++peerTargetIt)
		{
			/// Check if the target is not an ego target and the target is not too old.
			if(egoTargetByID.find(peerTargetIt->second.targetId) == egoTargetByID.end()
			   && peerTargetIt->second.frameNum + m_multiViewParams.m_maxPeerFrameDiff4NoDet >= frameNum)
			{
				Eigen::Vector2d pt_world_feet(peerTargetIt->second.nextState(0), peerTargetIt->second.nextState(1));

				if(m_defaultModelProjectors.at(videoStreamId)->isTargetVisibleInImage(pt_world_feet, peerTargetIt->second.height))
				{
					addTarget(videoStreamId, frameNum, nullptr, &peerTargetIt->second);
				}
			}
		}
	}
}

/**
 * @brief Add a new target from a detection or peer-camera target.
 * @param[in] videoStreamId Stream that receives the new target.
 * @param[in] frameNum Current frame number.
 * @param[in] detObj Detected object for the new target, or null when a peer target is supplied.
 * @param[in] peerTarget Peer-camera target used to initialize target information.
 * @return False when the new target overlaps an existing target or no tracker slot is available.
 */
bool NvTrackedObjectManager::addTarget(
	const NvMOTStreamId videoStreamId, const NvMOTFrameNum frameNum,
	const DetectedObjInfo *detObj, const PeerTarget *peerTarget
)
{
	/// Check the overlap with the existing targets and discard if there a high overlap one.
	const NvTrackedObjectList targetLists = getTrackedObjectsInUseList(videoStreamId);
	for(auto tIt = targetLists.begin(); tIt != targetLists.end(); ++tIt)
	{
		const std::shared_ptr<NvTrackedObject> target = tIt->second;
		NvMotDataTypes::Rect2f targetBbox;
		NvMOTClassId targetClassId = 0;
		if (detObj) {
			targetBbox = detObj->bbox;
			targetClassId = detObj->classId;
		} else {
			Eigen::Vector2d pt_world_feet(peerTarget->nextState(0), peerTarget->nextState(1));
			targetBbox = m_defaultModelProjectors.at(videoStreamId)->getBboxOfProjectedCylinderFromWorldFeet(
				pt_world_feet, peerTarget->classId, -1
			);
			targetClassId = peerTarget->classId;
		}
		const float iou = CV_FUNCS::computeIOU(target->getTargetInfo().tBbox, targetBbox);

		/// Apply the new-target IoU threshold to the matching class or all classes according to association settings.
		if ((m_params.m_checkClassMatch && targetClassId == target->getTargetInfo().classId)
				|| (!m_params.m_checkClassMatch))
		{
			if(iou > m_params.m_minIouDiff4NewTarget)
			{
				if(m_params.m_debugVerboseLevel > 2)
				{
					cout << "[NvTrackedObjectManager::addTarget()] Failed to add new traget (IOU to an existing target = " << iou << ", which should be larger than " << m_params.m_minIouDiff4NewTarget << ")" << endl;
				}
				return false;
			}
		}
	}

	/// Exclusive access to m_TrackedObjectsInPool and m_lastTrackerInd.
	std::unique_lock<std::mutex> addLock(m_trackedObjectsLock);

	/// Allow to borrow from other streams 10% more per-stream capacity than what is allocated to take advantage of heterogeneous video contents.
	if((int)(*m_TrackedObjectsInUse.at(videoStreamId)).size() >= (int)(m_params.m_maxTargetsPerStream*1.1f)
			|| m_TrackedObjectsInPool.size() == 0
			)
	{
		if(m_params.m_debugVerboseLevel > 2)
			CLOG(
				std::string( "Stream[" + std::to_string(videoStreamId >> 32) + "] frameNum[" + std::to_string(frameNum) + "] Failed to add a new tracker, because of Max Target reached" ).c_str()
			);
		addLock.unlock();
		return false;
	}

	/// Get a tracker from the pool that has the lowest index.
	auto it = m_TrackedObjectsInPool.begin();
	TrackerIndex trackerInd = it->first;
	std::shared_ptr<NvTrackedObject> target = it->second;
	m_TrackedObjectsInPool.erase(it);

	/// Track the last tracker index.
	if(m_lastTrackerInd < (int)trackerInd)
	{
		m_lastTrackerInd = (int)trackerInd;
	}

	if(m_params.m_debugVerboseLevel > 2)
	{
		CLOG(
			std::string( "Stream[" + std::to_string(videoStreamId >> 32) + "] frameNum[" + std::to_string(frameNum) + "] Added a new target: (trackerInd) = (" + std::to_string(trackerInd) + ") --> last trackerInd = " + std::to_string((int)getLastTargetInd()) ).c_str()
		);
	}

	addLock.unlock();

	/// Initialize and then update the target states.
	target->reset();
	target->setVideoStreamId(videoStreamId);

	if (!m_defaultModelProjectors.empty())
	{
		addLock.lock();
		/// Deep-copy of the model projector for the corresponding video stream.
		if(m_defaultModelProjectors.find(videoStreamId) != m_defaultModelProjectors.end() && m_defaultModelProjectors.at(videoStreamId) != nullptr && m_defaultModelProjectors.at(videoStreamId)->isInitialized())
		{
			target->setObjModelProjector(m_defaultModelProjectors.at(videoStreamId));
			if (detObj) {
				if (detObj->classId != target->getObjModelProjector()->getTargetClassId()) {
					if (m_params.m_debugVerboseLevel >= 5)
						std::cout << "[setTargetModelForClass] from NvTrackedObjectManager::addTarget (detObj), classId before=" << target->getObjModelProjector()->getTargetClassId() << " after=" << detObj->classId << std::endl;
					target->getObjModelProjector()->setTargetModelForClass(detObj->classId);
				}
			} else {
				if (peerTarget->classId != target->getObjModelProjector()->getTargetClassId()) {
					if (m_params.m_debugVerboseLevel >= 5)
						std::cout << "[setTargetModelForClass] from NvTrackedObjectManager::addTarget (peerTarget), classId before=" << target->getObjModelProjector()->getTargetClassId() << " after=" << peerTarget->classId << std::endl;
					target->getObjModelProjector()->setTargetModelForClass(peerTarget->classId);
				}
			}

		}
		addLock.unlock();
	}

	if(peerTarget)
	{
		target->initializeQuasiActiveStates(frameNum, peerTarget);
	}
	else
	{
		target->updateTargetStates(frameNum, detObj, TRACKER_STATE::TENTATIVE);
	}
	if(m_params.m_debugVerboseLevel > 3)
	{
		CLOG(
			std::string( "Stream[" + std::to_string(videoStreamId >> 32) + "] frameNum[" + std::to_string(frameNum) + "] class ID of new target at: (trackerInd) = (" + std::to_string(trackerInd) + ") is = " + std::to_string((int)target->getClassId()) ).c_str()
		);
	}
	/// Move the target from the candidate pool to the in-use pool.
	(*m_TrackedObjectsInUse.at(videoStreamId))[trackerInd] = target;

	return true;
}

/**
 * @brief Terminate a target and update all associated output and reassociation state.
 * @param[in] target Target to terminate.
 * @param[in] deleteMode Reason the tracker is terminated.
 * @param[in] peer Peer that absorbed this track on re-association, or null.
 */
void NvTrackedObjectManager::terminateTracker(
	const std::shared_ptr<NvTrackedObject> target,
	const TRACKER_DELETE_MODE deleteMode, const std::shared_ptr<NvTrackedObject> peer
)
{
	const TrackerIndex trackerInd = target->getTargetInd();
	const TargetId targetId = target->getTargetID();
	const NvMOTStreamId videoStreamId = target->getVideoStreamId();

	if(m_TrackedObjectsInUse.find(videoStreamId) == m_TrackedObjectsInUse.end())
	{
		if(m_params.m_debugVerboseLevel > 2)
			CLOG(
				std::string( "Stream[" + std::to_string(videoStreamId >> 32) + "] 1 Invalid target termination attempt !! targetInd = " + std::to_string(trackerInd) ).c_str()
			);

		return;
	}

	auto it = m_TrackedObjectsInUse.at(videoStreamId)->find(trackerInd);

	if(it == m_TrackedObjectsInUse.at(videoStreamId)->end())
	{
		if(m_params.m_debugVerboseLevel > 2)
			CLOG(
				std::string( "Stream[" + std::to_string(videoStreamId >> 32) + "] 2 Invalid target termination attempt !! targetInd = " + std::to_string(trackerInd) ).c_str()
			);

		return;
	}

	if(m_params.m_debugVerboseLevel > 2)
	{
		const TargetInfo &tInfo = target->getTargetInfo();

		std::stringstream deleteMsg;

		deleteMsg << "Stream[" << (videoStreamId >> 32) << "]: frameNum[" << tInfo.frameNum << "] Terminated a tracker (Ind, ID) = (" << trackerInd << ", " << targetId
				<< "), (Age, shadowTrackingAge) = (" << (int)tInfo.age << ", " << (int)target->getShadowTrackingAge()
				<< "), when Last trackerInd = " << (int)getLastTargetInd();

		switch(deleteMode)
		{
		case REGULAR_TERMINATION:
			deleteMsg << " <-- REGULAR_TERMINATION";
			break;

		case EARLY_TERMINATION:
			deleteMsg << " <-- EARLY_TERMINATION";
			break;

		case INSUFFICIENT_VISIBILITY:
			deleteMsg << " <-- INSUFFICIENT_VISIBILITY_DUE_TO_IMAGE_BORDER_CROPPING";
			break;

		case DUPLICATE_TRACKER:
			deleteMsg << " <-- DUPLICATE_TRACKER";
			break;

		case CORRUPTED_TRACKER:
			deleteMsg << " <-- CORRUPTED_TRACKER";
			break;

		case TOO_SMALL_OBJECT:
			deleteMsg << " <-- TOO_SMALL_OBJECT";
			break;

		case TOO_LARGE_OBJECT:
			deleteMsg << " <-- TOO_LARGE_OBJECT";
			break;

		case REMOVED_STREAM_TRACKER:
			deleteMsg << " <-- REMOVED_STREAM_TRACKER";
			break;

		case FUSED_BY_REASSOCIATION:
			deleteMsg << " <-- FUSED_BY_REASSOCIATION";
			break;

		default:
			throw std::runtime_error("Invalid tracker delete mode: " + std::to_string(deleteMode));
		}

		CLOG(deleteMsg.str().c_str());
	}

	TerminatedTrack terminatedTrack = createTerminatedTrack(target);

	/// Count ACTIVE track deaths by shadow-age distribution.
	if (deleteMode == REGULAR_TERMINATION) {
		m_tm_stats.term_reg_total++;
		if (target->hasBeenActive()) {
			m_tm_stats.term_reg_was_active++;
			uint32_t shadow = (uint32_t)target->getShadowTrackingAge();
			if      (shadow <= 1) m_tm_stats.term_reg_shadow_1++;
			else if (shadow == 2) m_tm_stats.term_reg_shadow_2++;
			else if (shadow == 3) m_tm_stats.term_reg_shadow_3++;
			else if (shadow == 4) m_tm_stats.term_reg_shadow_4++;
			else if (shadow == 5) m_tm_stats.term_reg_shadow_5++;
			else if (shadow == 6) m_tm_stats.term_reg_shadow_6++;
			else if (shadow == 7) m_tm_stats.term_reg_shadow_7++;
			else if (shadow == 8) m_tm_stats.term_reg_shadow_8++;
			else                  m_tm_stats.term_reg_shadow_9plus++;
			uint32_t age = (uint32_t)target->getAge();
			if (age < 30) m_tm_stats.term_reg_young++;
			else          m_tm_stats.term_reg_mature++;
		}
	} else if (deleteMode == EARLY_TERMINATION) {
		/// Count TENTATIVE early terminations by inference-frame lifetime to distinguish near-promotion from short-lived tracks.
		m_tm_stats.term_tent_total++;
		uint32_t age = (uint32_t)target->getAge();
		if      (age <= 2)  m_tm_stats.term_tent_age_le2++;
		else if (age <= 4)  m_tm_stats.term_tent_age_3_4++;
		else if (age <= 7)  m_tm_stats.term_tent_age_5_7++;
		else if (age <= 10) m_tm_stats.term_tent_age_8_10++;
		else                m_tm_stats.term_tent_age_11plus++;
	}

	/// Archive the trajectory data for later use and visual debugging.
	if(deleteMode != FUSED_BY_REASSOCIATION
		&& deleteMode != EARLY_TERMINATION)
	{
		m_trajectoryManager->archiveTrajectory(target);
	}

	/// Reset the target states, tracker states.
	target->reset();

	/// Remove from the in-use list.
	m_TrackedObjectsInUse.at(videoStreamId)->erase(it);

	std::unique_lock<std::mutex> terminateLock(m_trackedObjectsLock);

	/// Add it to the not-in-use pool.
	m_TrackedObjectsInPool[trackerInd] = target;

	/// Update terminated list.
	m_trackedObjectsTerminated[videoStreamId].push_back(target);

	/// Update tracker fusion list.
	if (deleteMode == FUSED_BY_REASSOCIATION && peer)
	{
		m_trackedObjectsFused[videoStreamId][peer] = {target, target->getMatchedDetObjInd()};
	}

	/// Keep track of the last tracker index.
	if(m_lastTrackerInd == (int)trackerInd)
	{
		/// Find the new last tracker index.
		int tInd = (int)trackerInd-1;
		for(; tInd >= 0; tInd--)
		{
			if(m_TrackedObjectsInPool.find(tInd) == m_TrackedObjectsInPool.end())
				break;
		}

		m_lastTrackerInd = tInd;
	}
	else if(m_lastTrackerInd < (int)trackerInd)
	{
		throw std::runtime_error(
			"Invalid m_lastTrackerInd = " + std::to_string(m_lastTrackerInd) + ", while the to-be-terminated trackerInd = " + std::to_string(trackerInd)
		);
	}

	/// Only record the terminated track if it is not a false track.
	if(deleteMode != FUSED_BY_REASSOCIATION
	&& deleteMode != EARLY_TERMINATION)
	{
		addTerminatedTrackToBufferPerStream(terminatedTrack);
	}

	terminateLock.unlock();
}

/** @brief Format all active target states as structured debug text. */
const std::string NvTrackedObjectManager::dumpAllTargetStates()
{
	std::stringstream msg;

	msg << "== Targets ==" << endl;
	for(auto streamIt = m_TrackedObjectsInUse.begin(); streamIt != m_TrackedObjectsInUse.end(); ++streamIt)
	{
		const NvMOTStreamId videoStreamId = streamIt->first;
		const NvTrackedObjectList &targetList = *(streamIt->second);

		if(targetList.size() == 0)
			continue;

		msg << "- Stream[" << std::to_string(videoStreamId >> 32) << "] frameNum[" << std::to_string(targetList.begin()->second->getTargetInfo().frameNum) << "]" << endl;

		for(auto targetIt = targetList.begin(); targetIt != targetList.end(); ++targetIt)
		{
			const TrackerIndex targetInd = targetIt->first;
			std::shared_ptr<NvTrackedObject> target = targetIt->second;
			const TargetInfo &tInfo = target->getTargetInfo();

			msg << DEBUG_UTIL::to_string_fmt(
				"  -- targetInd[%d]: trackerState = %d", (int)targetInd, (int)target->getTrackerState()
			);
			msg << ", bbox = " << CV_FUNCS::getEigenRectString(tInfo.tBbox);
			msg << DEBUG_UTIL::to_string_fmt(
				", (classId, confidence, shadowTrackingAge, age) = (%d, %.3f, %d, %d)\n",
				(int)tInfo.classId, tInfo.confidence, (int)target->getShadowTrackingAge(), (int)tInfo.age
			);
		}
	}

	return msg.str();
}

/**
 * @brief Remove all trackers for a stream while retaining its submodules.
 * @param[in] videoStreamId Stream whose trackers are removed while the stream is kept.
 */
void NvTrackedObjectManager::resetStream(const NvMOTStreamId videoStreamId)
{
	if(m_params.m_debugVerboseLevel > 1)
	{
		cout << "[NvTrackedObjectManager::clearTrackedObjects] Removing All ("  << m_TrackedObjectsInUse.at(videoStreamId)->size() <<") tracks from stream: " << (videoStreamId>>32) << endl;
	}

	const NvTrackedObjectList trackerList = getTrackedObjectsInUseList(videoStreamId);
	m_trackedObjectsTerminated.clear();
	m_trackedObjectsFused.clear();

	/// Remove any active targets for the stream.
	while(m_TrackedObjectsInUse.find(videoStreamId) != m_TrackedObjectsInUse.end()
			&& m_TrackedObjectsInUse.at(videoStreamId)->size() > 0)
	{
		const std::shared_ptr<NvTrackedObject> target = (*m_TrackedObjectsInUse.at(videoStreamId)).begin()->second;
		terminateTracker(target, REMOVED_STREAM_TRACKER);
	}

	/// Remove any tracklets in DB for the stream.
	m_trajectoryManager->cleanUpArchivedTrajectories_perStream(videoStreamId, 0, true);

	m_trajectoryManager->resetIDs();
}

/** @brief Reset all trackers for all streams while retaining stream submodules. */
void NvTrackedObjectManager::resetAllStreams()
{
	/// Remove all active tracks from every stream.
	for(auto it = m_TrackedObjectsInUse.begin(); it != m_TrackedObjectsInUse.end();)
	{
		auto it_next = std::next(it);
		resetStream(it->first);
		it = it_next;
	}
}

/**
 * @brief Remove all trackers and submodules for a stream.
 * @param[in] videoStreamId Stream whose trackers and per-stream modules are removed.
 */
void NvTrackedObjectManager::removeStream(const NvMOTStreamId videoStreamId)
{
	if(m_params.m_debugVerboseLevel > 1)
	{
		cout << "[NvTrackedObjectManager::removeStream] Removing stream ID: " << (videoStreamId>>32) << endl;
	}

	/// Reset all tracks for the stream.
	resetStream(videoStreamId);

	/// Remove stream from manager.
	m_trajectoryManager->updateStreamIds({}, {videoStreamId});

	dumpTerminatedTracks();

	/// Remove stream submodules.
	removeStreamSubModules(videoStreamId);
}

/**
 * @brief Remove per-stream model-projector and multi-view association state.
 * @param[in] videoStreamId Stream whose model projector and multi-view associator are removed.
 */
void NvTrackedObjectManager::removeStreamSubModules(const NvMOTStreamId videoStreamId)
{
	if (m_defaultModelProjectors.find(videoStreamId) != m_defaultModelProjectors.end())
	{
		m_defaultModelProjectors.at(videoStreamId).reset();
		m_defaultModelProjectors.erase(videoStreamId);
	}

	/// TODO: Notify peer-camera associators when this shallow stream removal occurs.
	if (m_multiViewAssociators.find(videoStreamId) != m_multiViewAssociators.end())
	{
		m_multiViewAssociators.at(videoStreamId).reset();
		m_multiViewAssociators.erase(videoStreamId);
	}

}

/**
 * @brief Select targets for ReID feature extraction during reassociation.
 * @param[in] inputMeta Input frames keyed by video stream ID.
 */
std::map<NvMOTStreamId, std::deque<DetectedObjInfo>> NvTrackedObjectManager::generateTrackedObjects4Reid(
	const std::map<NvMOTStreamId, NvMOTFrame *> &inputMeta
)
{
	std::map<NvMOTStreamId, std::deque<DetectedObjInfo>> objects4Reid;

	for (auto it = inputMeta.begin(); it != inputMeta.end(); ++it)
	{
		const NvMOTStreamId &videoStreamId = it->first;

		/// Whether there's no detected objects, we need to mark that this streams as valid.
		objects4Reid[videoStreamId] = m_trajectoryManager->generateTrackedObjects4ReidPerStream(
			videoStreamId, getTrackedObjectsInUseList(videoStreamId)
		);
	}

	return objects4Reid;
}

/**
 * @brief Return in-use targets for one stream.
 * @param[in] streamId Video stream ID whose targets are requested.
 */
NvTrackedObjectList NvTrackedObjectManager::getTrackedObjectsInUseList(const NvMOTStreamId streamId)
{
	if(m_TrackedObjectsInUse.find(streamId) != m_TrackedObjectsInUse.end() && m_TrackedObjectsInUse.at(streamId))
		return *m_TrackedObjectsInUse.at(streamId);
	else
		return NvTrackedObjectList();
}

/** @brief Return the minimum visibility required for trajectory dumping. */
const float NvTrackedObjectManager::getMinVisibility4TrajDump()
{
	if (m_trajectoryManager != nullptr)
	{
		return m_trajectoryManager->getMinVisibility4TrajDump();
	}
	else
	{
		return 0.0;
	}
}

/** @brief Report whether detector bounding boxes are included in debug dumps. */
const bool NvTrackedObjectManager::getDumpDetectorBboxes()
{
	if(m_trajectoryManager != nullptr)
	{
		return m_trajectoryManager->getDumpDetectorBboxes();
	}
	else
	{
		return false;
	}
}

/**
 * @brief Return peer targets visible to a stream.
 * @param[in] streamId Video stream ID whose peer targets are requested.
 */
const PeerTargetDB NvTrackedObjectManager::getPeerTargets(const NvMOTStreamId streamId)
{
	return m_multiViewAssociators[streamId]->getPeerTargetDB();
}

/** @brief Initialize target-manager parameters with default values. */
NvTrackerParamsTargetManager::NvTrackerParamsTargetManager()
{

	// [Default configuration]
	m_debugVerboseLevel = 0;
	m_checkClassMatch = true;
	m_enableBboxUnClipping = false;
	m_outputVisibility = false;
	m_preserveStreamUpdateOrder = false;
	m_saveTerminatedTracks = false;
	m_saveShadowTracks = false;
	m_useVOTBasedTargetUpdate = false;
	m_maxTargetsPerStream = 150;
	m_maxTargetsToTrack = 1000;

	m_minDetectorConfidence = 0.0f;
	m_minTrackerConfidence = 0.3f;
	m_probationAge = 5;
	m_maxShadowTrackingAge = 50;
	m_maxActiveShadowTrackingAge = 0;
	m_earlyTerminationAge = 2;
	m_minIouDiff4NewTarget = 0.5f;
	m_searchRegionPaddingScale = 1;

	// [Trajectory Management]
	m_maxTrajectoryBufferLength = 2000;

	// [Debug parameters]
	m_debugParamInt0 = 0;
	m_debugParamInt1 = 0;
	m_debugParamInt2 = 0;
	m_debugParamFloat0 = 0.f;
	m_debugParamFloat1 = 0.f;
	m_debugParamFloat2 = 0.f;

}

/**
 * @brief Append a terminated track to its stream output buffer.
 * @param[in] newTermTrack Terminated track appended to that stream's output buffer.
 */
void NvTrackedObjectManager::addTerminatedTrackToBufferPerStream( TerminatedTrack& newTermTrack)
{
	if( (!m_params.m_saveTerminatedTracks) || newTermTrack.targetHistory.size() == 0 )
		return;

	if (m_terminatedTrackOutputMap_.find(newTermTrack.videoStreamId) != m_terminatedTrackOutputMap_.end() )
	{
		/// Insert into existing list for this stream.
		std::list<TerminatedTrack>& termTrackList = m_terminatedTrackOutputMap_.at(newTermTrack.videoStreamId);
		termTrackList.push_back(newTermTrack);
	}
	else
	{
		std::list<TerminatedTrack> termTrackList;
		termTrackList.push_back(newTermTrack);
		m_terminatedTrackOutputMap_[newTermTrack.videoStreamId] = termTrackList;
	}
}

/**
 * @brief Create a terminated-track output record from a managed target.
 * @param[in] terminatedTarget Target whose identity and trajectory are copied.
 * @return Terminated-track record for the target, or an empty record when terminated tracks are not saved.
 */
TerminatedTrack NvTrackedObjectManager::createTerminatedTrack( const std::shared_ptr<NvTrackedObject> terminatedTarget)
{
	TerminatedTrack newTermTrack;

	if ( !m_params.m_saveTerminatedTracks )
	{
		return newTermTrack;
	}

	newTermTrack.targetID = terminatedTarget->getTargetID();
	newTermTrack.classID = terminatedTarget->getClassId();
	newTermTrack.state = terminatedTarget->getTrackerState();
	newTermTrack.videoStreamId = terminatedTarget->getVideoStreamId();

	std::list<TargetInfo> & trajInfo = terminatedTarget->getTrajectoryBuffer();

	for( TargetInfo curFrame : trajInfo)
	{
			newTermTrack.targetHistory.push_back(curFrame);
	}

	/// Trim off the last shadowTracks prior to the track dying.
	auto it = newTermTrack.targetHistory.rbegin();
	while(it != newTermTrack.targetHistory.rend()
		&& it->trackerState != TRACKER_STATE::QUASIACTIVE
		&& it->trackerState != TRACKER_STATE::ACTIVE) // Archive only validated target histories.
		{
			it = decltype(it)(newTermTrack.targetHistory.erase( std::next(it).base() ));
	}

	return newTermTrack;
}

/**
 * @brief Write intermediate detector and target information for one stream.
 * @param[in] assocOut Association output for the stream being dumped.
 */
void NvTrackedObjectManager::dumpIntermediateTargetInfo(const assocDataOut_t& assocOut)
{
	/// No file provided do not save to file.
	if(m_params.m_intermediateInfoFileName.empty())
	{
		return;
	}

	{
		const NvMOTStreamId streamId = assocOut.videoStreamId;
		const int streamIntId = streamId >> 32;
		NvTrackerParamsTrajectoryManager trajParams = m_trajectoryManager->getParams();

		std::vector< std::shared_ptr<NvTrackedObject> > targetList;
		std::vector< DetectedObjInfo* > detList;

			for(auto it = assocOut.unmatchedTargets.begin(); it != assocOut.unmatchedTargets.end(); ++it)
		{
			const TrackerIndex targetInd = *it;
			std::shared_ptr<NvTrackedObject> target = m_TrackedObjectsInUse.at(streamId)->at(targetInd);
			targetList.push_back(target);

			cout << "Stream[" << (assocOut.videoStreamId >> 32) << "] frameNum[" << assocOut.frameNum
							<< "] Unmatched target: " << targetInd << endl;
		}
		for(auto detIt = assocOut.unmatchedDetObjs.begin(); detIt != assocOut.unmatchedDetObjs.end(); ++detIt)
		{
			DetectedObjInfo *detObj = (DetectedObjInfo *) detIt->second;
			detList.push_back(detObj);
			cout << "Stream[" << (assocOut.videoStreamId >> 32) << "] frameNum[" << assocOut.frameNum
							<< "] Unmatched detection: " << detObj->confidence << endl;
		}
		for(auto it = assocOut.matchedPairs.begin(); it != assocOut.matchedPairs.end(); ++it)
		{
			const TrackerIndex targetInd = (*it).second;
			std::shared_ptr<NvTrackedObject> target = m_TrackedObjectsInUse.at(streamId)->at(targetInd);
			targetList.push_back(target);

			DetectedObjInfo *detObj = (DetectedObjInfo *) (*it).first;
			detList.push_back(detObj);
			cout << "Stream[" << (assocOut.videoStreamId >> 32) << "] frameNum[" << assocOut.frameNum
							<< "] Matched (target, detection): (" << targetInd << ", " << detObj->confidence << ")" << endl;
		}

		std::string fileName =  m_params.m_intermediateInfoFileName + std::to_string(streamIntId) + "_targetInfo.txt";

		/// Open for append.
		if(m_fileIntermediateTrackDump.find(streamId) == m_fileIntermediateTrackDump.end())
		{
			m_fileIntermediateTrackDump[streamId].open(fileName, std::ofstream::out);
		}
		else
		{
			m_fileIntermediateTrackDump[streamId].open(fileName, std::ofstream::app);
		}

		if(!m_fileIntermediateTrackDump[streamId])
		{
			throw std::runtime_error(
				"[NvTrackedObjectManager::dumpIntermediateTargetInfo()] fileIntermediateTrackDump is not opened !!!"
			);
		}

		for(auto target : targetList)
		{
			auto& tBbox = target->getTargetInfo().tBbox;
			if (target->getTrackerState() == TRACKER_STATE::TENTATIVE){
				continue;
			}
			m_fileIntermediateTrackDump[streamId] << assocOut.frameNum + 1 << ","
							<< target->getTargetID() << ","
							<< (int)(tBbox.min()(0)*trajParams.m_scale2OriginalSourceCoordinates(0) + 0.5f) << ","
							<< (int)(tBbox.min()(1)*trajParams.m_scale2OriginalSourceCoordinates(1) + 0.5f) << ","
							<< (int)(tBbox.sizes()(0)*trajParams.m_scale2OriginalSourceCoordinates(0) + 0.5f) << ","
							<< (int)(tBbox.sizes()(1)*trajParams.m_scale2OriginalSourceCoordinates(1) + 0.5f) << ","
							<< std::setprecision(3) << std::fixed
							<< target->getTargetInfo().confidence << ",-1,-1,-1" << endl;
		}
		m_fileIntermediateTrackDump[streamId].close();

		// [Detections]
		fileName =  m_params.m_intermediateInfoFileName + std::to_string(streamIntId) + "_det.txt";

		/// Open for append.
		if(m_fileIntermediateDetDump.find(streamId) == m_fileIntermediateDetDump.end())
		{
			m_fileIntermediateDetDump[streamId].open(fileName, std::ofstream::out);
		}
		else
		{
			m_fileIntermediateDetDump[streamId].open(fileName, std::ofstream::app);
		}

		if(!m_fileIntermediateDetDump[streamId])
		{
			throw std::runtime_error(
				"[NvTrackedObjectManager::dumpIntermediateTargetInfo()] fileIntermediateDetDump is not opened !!!"
			);
		}

		for(auto detObj : detList)
		{
			auto& bbox = detObj->bbox;

			m_fileIntermediateDetDump[streamId] << assocOut.frameNum + 1 << ","
							<< -1 << ","
							<< (int)(bbox.min()(0)*trajParams.m_scale2OriginalSourceCoordinates(0) + 0.5f) << ","
							<< (int)(bbox.min()(1)*trajParams.m_scale2OriginalSourceCoordinates(1) + 0.5f) << ","
							<< (int)(bbox.sizes()(0)*trajParams.m_scale2OriginalSourceCoordinates(0) + 0.5f) << ","
							<< (int)(bbox.sizes()(1)*trajParams.m_scale2OriginalSourceCoordinates(1) + 0.5f) << ","
							<< std::setprecision(3) << std::fixed
							<< detObj->confidence << ",-1,-1,-1" << endl;
		}

		m_fileIntermediateDetDump[streamId].close();

	}
}

/** @brief Write all buffered terminated tracks to the configured debug output. */
void NvTrackedObjectManager::dumpTerminatedTracks()
{
	/// No file provided do not save to file.
	if(!(m_params.m_terminatedTrackFilename.size() > 0))
	{
		return;
	}

	for( auto curStreamIter = m_terminatedTrackOutputMap_.begin(); curStreamIter != m_terminatedTrackOutputMap_.end(); curStreamIter++ )
	{
		const NvMOTStreamId streamId = curStreamIter->first;
		std::list<TerminatedTrack>& curTrackOutputBuffer = curStreamIter->second;

		const int streamIntId = streamId >> 32;

		const std::string fileName =  m_params.m_terminatedTrackFilename + std::to_string(streamIntId) + ".txt";

		/// Open for append.
		if(m_isFileTerminatedDumpCreated.find(streamId) == m_isFileTerminatedDumpCreated.end() || !m_isFileTerminatedDumpCreated[streamId])
		{
			m_fileTerminatedDump[streamId].open(fileName, std::ofstream::out);
			m_isFileTerminatedDumpCreated[streamId] = true;
		}
		else
		{
			m_fileTerminatedDump[streamId].open(fileName, std::ofstream::app);
		}

		if(!m_fileTerminatedDump[streamId])
		{
			throw std::runtime_error(
				"[NvTrackedObjectManager::dumpTerminatedTracks()] fileTerminatedDump is not opened !!!"
			);
		}

		/// Write the data into file.
		NvTrackerParamsTrajectoryManager trajParams = m_trajectoryManager->getParams();
		for(auto curTarget = curTrackOutputBuffer.begin(); curTarget != curTrackOutputBuffer.end(); ++curTarget)
		{

			for(auto curFrame : curTarget->targetHistory )
			{
				m_fileTerminatedDump[streamId] << curFrame.frameNum + 1 << ","
						<< (curTarget->targetID & 0xffff) << ","
						<< (int)(curFrame.tBbox.min()(0)*trajParams.m_scale2OriginalSourceCoordinates(0) + 0.5f) << ","
						<< (int)(curFrame.tBbox.min()(1)*trajParams.m_scale2OriginalSourceCoordinates(1) + 0.5f) << ","
						<< (int)(curFrame.tBbox.sizes()(0)*trajParams.m_scale2OriginalSourceCoordinates(0) + 0.5f) << ","
						<< (int)(curFrame.tBbox.sizes()(1)*trajParams.m_scale2OriginalSourceCoordinates(1) + 0.5f) << ","
						<< std::setprecision(3) << std::fixed
						<< curFrame.confidence << ","
						<< curFrame.pt_world_feet(0) << ","
						<< curFrame.pt_world_feet(1) << ","
						<< "-1,"
						<< curTarget->classID << ","
						<< static_cast<int>(curTarget->state) << ","
						<< curFrame.visibility << ","
						<< int(curFrame.pt_img_feet(0)*trajParams.m_scale2OriginalSourceCoordinates(0) + 0.5f) << ","
						<< int(curFrame.pt_img_feet(1)*trajParams.m_scale2OriginalSourceCoordinates(1) + 0.5f);

				/// Save convex hull.
				if (curFrame.convexHull.size() == 0)
				{
					m_fileTerminatedDump[streamId] << ",-1";
				}
				else
				{
					for (uint i=0; i<curFrame.convexHull.size(); i++)
					{
						m_fileTerminatedDump[streamId] << (i == 0 ? "," : "|")
							<< int(curFrame.convexHull.at(i)(0)*trajParams.m_scale2OriginalSourceCoordinates(0) + 0.5f)
							<< "|"
							<< int(curFrame.convexHull.at(i)(1)*trajParams.m_scale2OriginalSourceCoordinates(1) + 0.5f);
					}
				}

				/// Save 2.5D pose.
				if (curFrame.pose25d.size() == 0)
				{
					m_fileTerminatedDump[streamId] << ",-1";
				}
				else
				{
					for (uint i=0; i<curFrame.pose25d.size()/4; i++)
					{
						m_fileTerminatedDump[streamId] << (i == 0 ? "," : "|")
							<< int(curFrame.pose25d.at(i*4)*trajParams.m_scale2OriginalSourceCoordinates(0) + 0.5f)
							<< "|"
							<< int(curFrame.pose25d.at(i*4+1)*trajParams.m_scale2OriginalSourceCoordinates(1) + 0.5f)
							<< "|" << curFrame.pose25d.at(i*4+2) << "|" << curFrame.pose25d.at(i*4+3);
					}
				}

				/// Save 3D pose.
				if (curFrame.pose3d.size() == 0)
				{
					m_fileTerminatedDump[streamId] << ",-1";
				}
				else
				{
					for (uint i=0; i<curFrame.pose3d.size()/3; i++)
					{
						m_fileTerminatedDump[streamId] << (i == 0 ? "," : "|")
							<< curFrame.pose3d.at(i*3) << "|" << curFrame.pose3d.at(i*3+1) << "|" << curFrame.pose3d.at(i*3+2);
					}
				}

				m_fileTerminatedDump[streamId] << endl;
			}
		}
		m_fileTerminatedDump[streamId].close();
	}

}

/**
 * @brief Load target-manager parameters from a configuration module.
 * @param[in] configFilePath Path to the tracker configuration file.
 * @param[in] moduleName Name of the configuration module to load.
 */
NvTrackerParamsTargetManager::NvTrackerParamsTargetManager(
	const std::string &configFilePath, const std::string moduleName
)
{
	YAML::Node configRoot = getConfigRoot(configFilePath);
	this->read(configRoot[moduleName]);
	/// Read cross-cutting BaseConfig params (startup-only; overrides in-section default).
	const YAML::Node &baseNode = configRoot["BaseConfig"];
	if (baseNode && baseNode["checkClassMatch"])
		m_checkClassMatch = (bool)baseNode["checkClassMatch"].as<int>();
}

/**
 * @brief Load target-manager parameters from a YAML node.
 * @param[in] fn YAML node containing the target-manager configuration.
 */
NvTrackerParamsTargetManager::NvTrackerParamsTargetManager(const YAML::Node &fn)
{
	this->read(fn);
}

/**
 * @brief Read target-manager parameters from a YAML node.
 * @param[in] fn YAML node containing the target-manager configuration.
 */
void NvTrackerParamsTargetManager::read(const YAML::Node &fn)
{
	if(!m_isInit)
	{
		/// Set the default values.
		*this = NvTrackerParamsTargetManager();
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
			m_debugVerboseLevel = fit->second.as<float>();
		}
		else if (paramKey == "outputTerminatedTracks")
		{
			/// Only support setting this parameter on startup.
			if(!m_isInit)
				m_saveTerminatedTracks = (int)fit->second.as<float>();
			else
				CLOG(std::string("!!![WARNING] m_saveTerminatedTracks is not supported for dynamic updates").c_str());
		}
		else if (paramKey == "useVOTBasedTargetUpdate")
		{
			m_useVOTBasedTargetUpdate = (int)fit->second.as<float>();
		}
		else if (paramKey == "outputShadowTracks")
		{
			/// Only support setting this parameter on startup.
			if(!m_isInit)
				m_saveShadowTracks = (int)fit->second.as<float>();
			else
				CLOG(std::string("!!![WARNING] m_saveShadowTracks is not supported for dynamic updates").c_str());
		}
		else if (paramKey == "terminatedTrackFilename")
		{
			/// Only support setting this parameter on startup.
			if(!m_isInit)
				m_terminatedTrackFilename = fit->second.as<std::string>();
			else
				CLOG(std::string("!!![WARNING] m_terminatedTrackFilename is not supported for dynamic updates").c_str());
		}
		else if (paramKey == "intermediateInfoFileName")
		{
			m_intermediateInfoFileName = fit->second.as<std::string>();
		}
		else if (paramKey == "preserveStreamUpdateOrder")
		{
			/// Only support setting this parameter on startup.
			if(!m_isInit)
				m_preserveStreamUpdateOrder = (int)fit->second.as<float>();
			else
				CLOG(std::string("!!![WARNING] m_preserveStreamUpdateOrder is not supported for dynamic updates").c_str());
		}
		else if (paramKey == "enableBboxUnClipping")
		{
			m_enableBboxUnClipping = (int)fit->second.as<float>();
		}
		else if (paramKey == "outputVisibility")
		{
			m_outputVisibility = (int)fit->second.as<float>();
		}
		else if (paramKey == "maxTargetsPerStream")
		{
			/// Only support setting this parameter on startup.
			if(!m_isInit)
				m_maxTargetsPerStream = fit->second.as<float>();
			else
				CLOG(std::string("!!![WARNING] m_maxTargetsPerStream is not supported for dynamic updates").c_str());
		}
		else if (paramKey == "minDetectorConfidence")
		{
			m_minDetectorConfidence = fit->second.as<float>();
		}
		else if (paramKey == "minTrackerConfidence")
		{
			m_minTrackerConfidence = fit->second.as<float>();
		}
		else if (paramKey == "maxShadowTrackingAge")
		{
			m_maxShadowTrackingAge = fit->second.as<float>();
		}
		else if (paramKey == "probationAge")
		{
			m_probationAge = fit->second.as<float>();
		}
		else if (paramKey == "maxActiveShadowTrackingAge")
		{
			/// Per-state shadow cap override — extends ACTIVE-lineage track survival.
			m_maxActiveShadowTrackingAge = fit->second.as<int>();
			if (m_debugVerboseLevel >= 1) {
				fprintf(
					stderr, "[NvTrackedObjectManager] maxActiveShadowTrackingAge=%d\n", m_maxActiveShadowTrackingAge
				);
			}
		}
		else if (paramKey == "earlyTerminationAge")
		{
			m_earlyTerminationAge = fit->second.as<float>();
		}
		else if (paramKey == "minIouDiff4NewTarget")
		{
			m_minIouDiff4NewTarget = fit->second.as<float>();
		}
		else if (paramKey == "searchRegionPaddingScale")
		{
			m_searchRegionPaddingScale = fit->second.as<float>();
		}
		else if (paramKey == "maxTrajectoryBufferLength")
		{
			/// Only support setting this parameter on startup.
			if(!m_isInit)
				m_maxTrajectoryBufferLength = fit->second.as<float>();
			else
				CLOG(std::string("!!![WARNING] m_maxTrajectoryBufferLength is not supported for dynamic updates").c_str());
		}
		else if(paramKey == "debugParamInt0")
		{
			m_debugParamInt0 = fit->second.as<float>();
		}
		else if(paramKey == "debugParamInt1")
		{
			m_debugParamInt1 = fit->second.as<float>();
		}
		else if(paramKey == "debugParamInt2")
		{
			m_debugParamInt2 = fit->second.as<float>();
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
			cout << "!! [WARNING][NvTrackerParamsTargetManager] Unknown param found: " << paramKey << endl;
		}

	}

	/// Mark parameter initialization complete.
	m_isInit = true;

	// [Post-processing]
	/// Ensure the shadow window extends at least ten frames beyond probation so tracks can become ACTIVE after minor occlusions.
	if(m_maxShadowTrackingAge < m_probationAge + 10)
	{
		m_maxShadowTrackingAge = m_probationAge + 10;
	}

	/// Preserve enough trajectory history to support reassociation after probation.
	if(m_maxTrajectoryBufferLength >=0 && m_maxTrajectoryBufferLength <= m_probationAge + 10)
	{
		m_maxTrajectoryBufferLength = m_probationAge + 10;
	}
}
