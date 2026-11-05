/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "modules/utility/cv_functions.hpp"
#include "modules/TrajectoryManager/NvTrajectoryManagerOnline.hpp"
#include "modules/TrajectoryManager/TrajectoryOperations.hpp"

using std::cout;
using std::endl;

/**
 * @brief Construct an online trajectory manager from configuration.
 * @param[in] configFilePath Path to the configuration file.
 */
NvTrajectoryManagerOnline::NvTrajectoryManagerOnline(const std::string &configFilePath)
	: NvTrajectoryManager(configFilePath)
{
	if(m_params.m_debugVerboseLevel > 1)
		cout << "[NvTrajectoryManagerOnline] Instantiated" << endl;

}

/** @brief Destroy the online trajectory manager. */
NvTrajectoryManagerOnline::~NvTrajectoryManagerOnline()
{
}

/**
 * @brief Generate target detections that require ReID extraction for one stream.
 * @param[in] streamId Source stream identifier.
 * @param[in] targetList Tracked targets to inspect.
 */
std::deque<DetectedObjInfo> NvTrajectoryManagerOnline::generateTrackedObjects4ReidPerStream(
	const NvMOTStreamId streamId, const std::map<TrackerIndex, std::shared_ptr<NvTrackedObject>> &targetList
)
{
	std::deque<DetectedObjInfo> targetReidList;
	int bboxInd = 0;

	for (auto targetIt = targetList.begin(); targetIt != targetList.end(); targetIt++)
	{
		std::shared_ptr<NvTrackedObject> target = targetIt->second;
		if (target->needTargetIDAssigned() ||
			(m_params.m_reidExtractionInterval >= 0 && target->getAge() % (m_params.m_reidExtractionInterval + 1) == 0) )
		{
			const NvMOTObjToTrack *detObj = target->getTargetInfo().getMatchedDetObjIn();
			if (detObj)
			{
				targetReidList.push_back(DetectedObjInfo(detObj, target->getMatchedDetObjInd()));

				/// Store each target's index in reid vector.
				(*m_trackedObjectsReidInd.at(streamId))[target] = bboxInd++;
			}
		}
	}

	return targetReidList;
}
