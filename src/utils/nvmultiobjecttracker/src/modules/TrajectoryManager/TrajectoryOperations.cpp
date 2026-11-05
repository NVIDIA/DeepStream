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

#include "modules/utility/debug_common.hpp"
#include "modules/TrajectoryManager/TrajectoryOperations.hpp"

namespace TrajOps {

/**
 * @brief Combine same-frame target records from overlapping tracklets.
 * @param[in] prevTarget Target record from the earlier tracklet.
 * @param[in] nextTarget Target record from the later tracklet.
 */
TargetInfo fuseTargets(const TargetInfo prevTarget, const TargetInfo nextTarget)
{
	if((prevTarget.videoStreamId != nextTarget.videoStreamId) ||
		(prevTarget.frameNum != nextTarget.frameNum) ||
		(prevTarget.trackerState != TRACKER_STATE::ACTIVE) ||
		(nextTarget.trackerState != TRACKER_STATE::ACTIVE) ||
		(prevTarget.classId != nextTarget.classId))
		{
			CLOG("[WARNING][NvTrajectoryManager::fuseTargets] prevTarget & nextTarget do not have appropriate values");
			std::cout << "prevTarget (videoStreamId, frameNum, trackerState, classId): (" <<
			prevTarget.videoStreamId << "," << prevTarget.frameNum << "," <<
			prevTarget.trackerState << "," << prevTarget.classId << ")" << std::endl;
			std::cout << "nextTarget (videoStreamId, frameNum, trackerState, classId): (" <<
			nextTarget.videoStreamId << "," << nextTarget.frameNum << "," <<
			nextTarget.trackerState << "," << nextTarget.classId << ")" << std::endl;
		}
	const float alpha = nextTarget.confidence/(nextTarget.confidence + prevTarget.confidence);
	const Eigen::Vector2f newLoc =  alpha*nextTarget.loc + (1.f - alpha)*prevTarget.loc;
	const float width = alpha*nextTarget.tBbox.sizes()(0) + (1.f - alpha)*prevTarget.tBbox.sizes()(0);
	const float height = alpha*nextTarget.tBbox.sizes()(1) + (1.f - alpha)*prevTarget.tBbox.sizes()(1);
	const NvMotDataTypes::Rect2f newBbox = CV_FUNCS::constructRect2f(
		newLoc(0) - width/2, newLoc(1) - height/2, width, height
	);

	/// Initializing new target with nextTarget will copy the DetectedObjInfo of nextTarget to newTarget.
	TargetInfo newTarget = nextTarget;
	newTarget.tBbox = newBbox;
	newTarget.age = prevTarget.age;

	newTarget.confidence = alpha * nextTarget.confidence + (1.f - alpha) * prevTarget.confidence;

	/// In case of 3D tracking enabled, Interpolate the pt_img_feet, pt_world_feet, etc.
	newTarget.pt_img_feet = alpha * nextTarget.pt_img_feet + (1.f - alpha) * prevTarget.pt_img_feet;
	newTarget.pt_world_feet = alpha * nextTarget.pt_world_feet + (1.f - alpha) * prevTarget.pt_world_feet;
	newTarget.visibility = alpha * nextTarget.visibility + (1.f - alpha) * prevTarget.visibility;
	newTarget.imgBorderCroppingVisibility = alpha * nextTarget.imgBorderCroppingVisibility + (1.f - alpha) * prevTarget.imgBorderCroppingVisibility;

	return newTarget;
}

/**
 * @brief Interpolate target records between two observations.
 * @param[in] prevTarget Earlier target observation.
 * @param[in] nextTarget Later target observation.
 */
std::deque<TargetInfo> generateInterpolatedTargets(const TargetInfo prevTarget, const TargetInfo nextTarget)
{
	std::deque<TargetInfo> midTargets;

	const int interpolationLength = nextTarget.frameNum - prevTarget.frameNum;

	if(interpolationLength > 1)
	{
		for(int i=1; i<interpolationLength; i++)
		{
			const float alpha = i/(float)interpolationLength;
			const Eigen::Vector2f newLoc =  alpha*nextTarget.loc + (1.f - alpha)*prevTarget.loc;
			const float width = alpha*nextTarget.tBbox.sizes()(0) + (1.f - alpha)*prevTarget.tBbox.sizes()(0);
			const float height = alpha*nextTarget.tBbox.sizes()(1) + (1.f - alpha)*prevTarget.tBbox.sizes()(1);

			const NvMotDataTypes::Rect2f newBbox = CV_FUNCS::constructRect2f(
				newLoc(0) - width/2, newLoc(1) - height/2, width, height
			);

			TargetInfo newTarget = prevTarget;

			newTarget.matchedDetObjInfo = nullptr;
			newTarget.frameNum += i;
			newTarget.tBbox = newBbox;
			newTarget.age += i;
			newTarget.shadowTrackingAge = (nextTarget.trackerState == TRACKER_STATE::ACTIVE ? 0 : prevTarget.shadowTrackingAge + i);
			newTarget.trackerState = nextTarget.trackerState;

			newTarget.confidence = alpha*nextTarget.confidence + (1.f - alpha)*prevTarget.confidence;

			/// In case of 3D tracking enabled, Interpolate the pt_img_feet, pt_world_feet, etc.
			newTarget.pt_img_feet = alpha*nextTarget.pt_img_feet + (1.f - alpha)*prevTarget.pt_img_feet;
			newTarget.pt_world_feet = alpha*nextTarget.pt_world_feet + (1.f - alpha)*prevTarget.pt_world_feet;
			newTarget.visibility = alpha*nextTarget.visibility + (1.f - alpha)*prevTarget.visibility;
			newTarget.imgBorderCroppingVisibility = alpha*nextTarget.imgBorderCroppingVisibility + (1.f - alpha)*prevTarget.imgBorderCroppingVisibility;

			midTargets.push_back(std::move(newTarget));
		}

		if(midTargets.begin()->frameNum != prevTarget.frameNum + 1
				|| midTargets.rbegin()->frameNum != nextTarget.frameNum - 1)
		{
			CLOG(
				"[WARNING] midTargets.begin()->frameNum != prevTarget.frameNum + 1 || midTargets.rbegin()->frameNum != nextTarget.frameNum - 1"
			);
		}
	}

	return midTargets;
}

/**
 * @brief Generate a per-frame tracklet over an optional frame range.
 * @param[in] traj Sparse trajectory to interpolate.
 * @param[in] startFrameNum First frame to include, or zero for the complete trajectory.
 * @param[in] endFrameNum Last frame to include, or zero for the complete trajectory.
 */
std::list<TargetInfo> generateFullTracklet(
	const std::list<TargetInfo> &traj, const int startFrameNum, const int endFrameNum
)
{
	std::list<TargetInfo> recovTraj;

	/// Check if the caller want to generate the whole tracklet.
	const bool isFullFull = (startFrameNum == 0 && endFrameNum == 0) ? true : false;

	if(!isFullFull && endFrameNum < startFrameNum)
	{
		CLOG(
			std::string("[TrajOps::generateFullTracklet] [ERROR] endFrameNum[" + std::to_string(endFrameNum) + "] < startFrameNum[" + std::to_string(startFrameNum) + "] !!!").c_str()
		);
		return recovTraj;
	}

	/// Return the empty tracklet when there is no trajectory.
	if(traj.size() == 0)
	{
		return recovTraj;
	}

	auto it = traj.begin();

	/// Move to the closest traj pt to startFrameNum.
	while(!isFullFull && std::next(it) != traj.end() && (int)std::next(it)->frameNum <= startFrameNum)
	{

		++it;
	}

	for(; it != traj.end() && (isFullFull || (int)it->frameNum <= endFrameNum); ++it)
	{
		std::deque<TargetInfo> targets;

		auto it_next = std::next(it);
		if(it_next != traj.end())
		{
			/// Get all the interpolated target points.
			targets = TrajOps::generateInterpolatedTargets(*it, *it_next);
		}
		/// Add the current target point at the front.
		targets.push_front(*it);

		/// Insert the interpolated traj points to the list.
		for(auto tIt = targets.begin(); tIt != targets.end(); ++tIt)
		{
			if(isFullFull || ((int)tIt->frameNum >= startFrameNum && (int)tIt->frameNum <= endFrameNum))
			{
				recovTraj.push_back(*tIt);
			}
		}
	}

	return recovTraj;
}

}
