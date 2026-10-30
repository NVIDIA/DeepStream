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

#include <iostream>
#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "modules/utility/cuda_common.hpp"
#include "modules/utility/debug_common.hpp"
#include "modules/utility/cv_functions.hpp"
#include "modules/DataAssociator/DataAssociator.hpp"

/// Infinity cost for distance matrix.
#define MIN_SCORE -1e+5

using std::cout;
using std::endl;

/**
 * @brief Construct a data associator from a configuration file.
 * @param[in] configFilePath The path to a config file.
 */
DataAssociator::DataAssociator(const std::string &configFilePath)
{
	m_params = NvTrackerParamsAssoc(configFilePath, "DataAssociator");

	m_isInit = false;
}

/** @brief Report accumulated association statistics during destruction. */
DataAssociator::~DataAssociator()
{
	/// Dump per-frame match funnel on teardown.
	if (m_params.m_debugVerboseLevel >= 1) {
		fprintf(
			stderr,
			"[DataAssociator] total=%llu gate_hard=%llu near_lt04=%llu near_04_05=%llu near_ge50=%llu passed=%llu" " qactive_rescue_attempted=%llu qactive_iou_hardgate=%llu qactive_size_rescued=%llu" " stage2_eligible=%llu stage2_matched=%llu stage2_no_dets=%llu" " active_iou_hardgate=%llu" " stage2_has_dets=%llu stage2_pair_total=%llu stage2_wp_gate=%llu stage2_size_gate=%llu" " stage2_noworld_track=%llu stage2_noworld_det=%llu stage2_hard_world=%llu stage2_size_rescued=%llu\n",
			(unsigned long long)m_da_stats.total, (unsigned long long)m_da_stats.gate_hard,
			(unsigned long long)m_da_stats.near_lt04, (unsigned long long)m_da_stats.near_04_05,
			(unsigned long long)m_da_stats.near_ge50, (unsigned long long)m_da_stats.passed,
			(unsigned long long)m_da_stats.qactive_rescue_attempted,
			(unsigned long long)m_da_stats.qactive_iou_hardgate, (unsigned long long)m_da_stats.qactive_size_rescued,
			(unsigned long long)m_da_stats.stage2_eligible, (unsigned long long)m_da_stats.stage2_matched,
			(unsigned long long)m_da_stats.stage2_no_dets, (unsigned long long)m_da_stats.active_iou_hardgate,
			(unsigned long long)m_da_stats.stage2_has_dets, (unsigned long long)m_da_stats.stage2_pair_total,
			(unsigned long long)m_da_stats.stage2_wp_gate, (unsigned long long)m_da_stats.stage2_size_gate,
			(unsigned long long)m_da_stats.stage2_noworld_track, (unsigned long long)m_da_stats.stage2_noworld_det,
			(unsigned long long)m_da_stats.stage2_hard_world, (unsigned long long)m_da_stats.stage2_size_rescued
		);
	}
}

/**
 * @brief Initialize the data associator for a source stream.
 * @param[in] frameInfo Metadata about the source image (e.g., size, color).
 * @param[in] maxTargetsPerStream The max number of targets to track per stream.
 * @param[in] useVisualTracker Whether visual-tracker similarity is used.
 * @param[in] useReid Whether ReID similarity is used.
 */
void DataAssociator::initialize(
	const SourceFrameInfo &frameInfo, const uint32_t maxTargetsPerStream,
	const bool useVisualTracker, const bool useReid
)
{
	m_maxTargetsPerStream = maxTargetsPerStream;

	/// Store so updateParams can rebuild similarity lists without re-supplying these flags.
	m_useVisualTracker = useVisualTracker;
	m_useReid = useReid;

	initializeSimilarityParams(useVisualTracker, useReid);

	/// Report the ReID association initialization when debugging.
	if(m_params.m_debugVerboseLevel > 1)
		cout << "[DataAssociator] Initialized reidAssoc" << endl;

	m_isInit = true;

	if(m_params.m_debugVerboseLevel > 1)
		cout << "[DataAssociator] Initialized" << endl;
}

/**
 * @brief Rebuild per-stage similarity parameters for the enabled features.
 * @param[in] useVisualTracker Whether visual-tracker similarity is used.
 * @param[in] useReid Whether ReID similarity is used.
 */
void DataAssociator::initializeSimilarityParams(const bool useVisualTracker, const bool useReid)
{
	/// Take an exclusive write lock so computeDistanceMatrix cannot read the list while it is cleared and rebuilt.
	std::unique_lock<std::mutex> wlock(m_similarityParamsMutex);
	m_similarityParamLists.clear();

	/// First stage association params.
	std::vector<SimilarityParam> similarityParamFirst={
		{SimilarityParam::TYPES::IOU, m_params.m_minMatchingScore4Iou, m_params.m_matchingScoreWeight4Iou},
		{
			SimilarityParam::TYPES::SIZE_SIMILARITY, m_params.m_minMatchingScore4SizeSimilarity,
			m_params.m_matchingScoreWeight4SizeSimilarity
		}
	};

	/// Only uses visual similarity scores when visual tracker enabled.
	if (useVisualTracker)
	{
		/// Print warnings if visual similarity scores are not set in config.
		if (!(m_params.m_matchingScoreSet & NvTrackerParamsAssoc::MATCHING_SCORE_SET_STATUS::MIN_VISUAL))
		{
			CLOG(
				"[DataAssociator][Warning] Visual tracker enabled, but minMatchingScore4VisualSimilarity is not set. Using 0.0 by default"
			);
			m_params.m_minMatchingScore4VisualSimilarity = 0.0f;
		}
		if (!(m_params.m_matchingScoreSet & NvTrackerParamsAssoc::MATCHING_SCORE_SET_STATUS::WEIGHT_VISUAL))
		{
			CLOG(
				"[DataAssociator][Warning] Visual tracker enabled, but matchingScoreWeight4VisualSimilarity is not set. Using 1.0 by default"
			);
			m_params.m_matchingScoreWeight4VisualSimilarity = 1.0f;
		}

		similarityParamFirst.push_back(
			{
				SimilarityParam::TYPES::VISUAL_SIMILARITY, m_params.m_minMatchingScore4VisualSimilarity,
				m_params.m_matchingScoreWeight4VisualSimilarity
			}
		);
	}
	else
	{
		if (m_params.m_matchingScoreSet & NvTrackerParamsAssoc::MATCHING_SCORE_SET_STATUS::MIN_VISUAL)
		{
			CLOG("[DataAssociator][Warning] Visual tracker disabled. minMatchingScore4VisualSimilarity is not used");
		}
		if (m_params.m_matchingScoreSet & NvTrackerParamsAssoc::MATCHING_SCORE_SET_STATUS::WEIGHT_VISUAL)
		{
			CLOG("[DataAssociator][Warning] Visual tracker disabled. matchingScoreWeight4VisualSimilarity is not used");
		}
	}

	/// Only uses reid similarity scores when reid enabled.
	if (useReid)
	{
		/// Print warnings if reid scores are not set in config.
		if (!(m_params.m_matchingScoreSet & NvTrackerParamsAssoc::MATCHING_SCORE_SET_STATUS::MIN_REID))
		{
			CLOG(
				"[DataAssociator][Warning] Reid enabled, but minMatchingScore4ReidSimilarity is not set. Using 0.0 by default"
			);
			m_params.m_minMatchingScore4ReidSimilarity = 0.0f;
		}
		if (!(m_params.m_matchingScoreSet & NvTrackerParamsAssoc::MATCHING_SCORE_SET_STATUS::WEIGHT_REID))
		{
			CLOG(
				"[DataAssociator][Warning] Reid enabled, but matchingScoreWeight4ReidSimilarity is not set. Using 1.0 by default"
			);
			m_params.m_matchingScoreWeight4ReidSimilarity = 1.0f;
		}

		similarityParamFirst.push_back(
			{
				SimilarityParam::TYPES::REID_SIMILARITY, m_params.m_minMatchingScore4ReidSimilarity,
				m_params.m_matchingScoreWeight4ReidSimilarity
			}
		);
	}
	else
	{
		if (m_params.m_matchingScoreSet & NvTrackerParamsAssoc::MATCHING_SCORE_SET_STATUS::MIN_REID)
		{
			CLOG("[DataAssociator][Warning] Reid disabled. minMatchingScore4ReidSimilarity is not used");
		}
		if (m_params.m_matchingScoreSet & NvTrackerParamsAssoc::MATCHING_SCORE_SET_STATUS::WEIGHT_REID)
		{
			CLOG("[DataAssociator][Warning] Reid disabled. matchingScoreWeight4ReidSimilarity is not used");
		}
	}

	/// Normalize score weights.
	float matchingScoreWeightSum = 0.0f;
	for (auto it = similarityParamFirst.begin(); it != similarityParamFirst.end(); it++)
	{
		matchingScoreWeightSum += it->weight;
	}
	if (matchingScoreWeightSum != 0)
	{
		for (auto it = similarityParamFirst.begin(); it != similarityParamFirst.end(); it++)
		{
			it->weight /= matchingScoreWeightSum;
		}
	}
	m_similarityParamLists.push_back(similarityParamFirst);

	/// Tentative detection association params.
	if (m_params.m_associationMatcherType == ASSOCIATION_MATCHER_TYPE::CASCADED)
	{
		m_similarityParamLists.push_back(
			{{SimilarityParam::TYPES::IOU, m_params.m_minMatchingScore4TentativeIou, 1.0}}
		);

		/// Add a world-proximity stage only when maxWorldDist4PrimaryAssoc is positive.
		if (m_params.m_maxWorldDist4PrimaryAssoc > 0.0f)
		{
			float w_wp = m_params.m_matchingScoreWeight4Iou;
			float w_s2 = m_params.m_matchingScoreWeight4SizeSimilarity;
			const float wSum2 = w_wp + w_s2;
			if (wSum2 > 0.0f) { w_wp /= wSum2; w_s2 /= wSum2; }
			/// WORLD_PROXIMITY uses a minimum of 0.1, and SIZE_SIMILARITY uses the primary size-similarity minimum.
			m_similarityParamLists.push_back({
				{SimilarityParam::TYPES::WORLD_PROXIMITY, 0.1f, w_wp},
				{SimilarityParam::TYPES::SIZE_SIMILARITY, m_params.m_minMatchingScore4SizeSimilarity, w_s2}
			});
		}
	}
}

/**
 * @brief Compute the association score between a detected object and a tracked object.
 * @param[in] target Tracked object.
 * @param[in] detObj Detected object.
 * @param[in] similarityParamList Similarity parameters and their weights.
 * @param[in] sumCorrRespAtDetectedPt Sum of tracker correlation responses at the detected point.
 * @return The similarity score.
 */
float DataAssociator::assocDistanceFunc(
	const std::shared_ptr<NvTrackedObject> target, const DetectedObjInfo &detObj,
	const std::vector<SimilarityParam> &similarityParamList, const float sumCorrRespAtDetectedPt
) const
{
	/// Count all invocations before any gate.
	m_da_stats.total++;
	/// Detect Stage Second (stage 2) by checking for WORLD_PROXIMITY in params.
	bool isS2 = false;
	for (const auto& sp : similarityParamList) {
		if (sp.type == SimilarityParam::TYPES::WORLD_PROXIMITY) { isS2 = true; break; }
	}
	/// Detect Stage 3 (IoU-only stage): stage 3 uses m_similarityParamLists[1] = {IOU only}.
	const bool isIouOnlyStage = (similarityParamList.size() == 1
		&& similarityParamList[0].type == SimilarityParam::TYPES::IOU);
	/// Check class match and Mahalanobis distance within threshold if enabled.
	if ( (m_params.m_checkClassMatch && detObj.classId != target->getTargetInfo().classId) \
		|| (m_params.m_thresholdMahalanobis > 0 && target->getDisMahalanobisDets()[detObj.bboxInd] > m_params.m_thresholdMahalanobis) )
	{
		/// Class/mahalanobis gate.
		m_da_stats.gate_hard++;
		return 0.f;
	}

	/// World-distance gate: reject detection-track pairs whose ground-plane feet are farther apart than the configured threshold.
	float worldDistSaved = std::numeric_limits<float>::max();
	{
		const TargetInfo &tInfo = target->getTargetInfo();
		if (tInfo.pt_world_feet.x() >= 0.0f && detObj.pt_world_feet.x() >= 0.0f) {
			worldDistSaved = (tInfo.pt_world_feet - detObj.pt_world_feet).norm();
		}
		/// Primary world-distance gate (only when threshold is configured > 0).
		if (m_params.m_maxWorldDist4PrimaryAssoc > 0.0f) {
			if (worldDistSaved < std::numeric_limits<float>::max() / 2.0f) {
				if (worldDistSaved > m_params.m_maxWorldDist4PrimaryAssoc) {
					/// S2 global hard gate.
					if (isS2) m_da_stats.stage2_hard_world++;
					/// World-distance hard gate rejection.
					m_da_stats.gate_hard++;
					return 0.f;
				}
			} else if (isS2) {
				/// Stage Second pairs where world pos is unavailable.
				if (tInfo.pt_world_feet.x() < 0.0f)  m_da_stats.stage2_noworld_track++;
				if (detObj.pt_world_feet.x() < 0.0f) m_da_stats.stage2_noworld_det++;
			}
		}
	}

	NvMotDataTypes::Rect2f tBbox;
	float totalScore = 0.f;

	if ( m_params.m_usePrediction4Assoc && target->getTrackerState() != TRACKER_STATE::TENTATIVE )
	{
		/// Use the predicted bbox for matching in case of no visual tracker enabled.
		tBbox = target->getPredictedTargetBbox();
	}
	else
	{
		/// Use the bbox (1) from the previous frame if there's no visual tracker or (2) from the localization if there is a visual tracker.
		tBbox = target->getTargetInfo().tBbox;
	}
	float iouScore = 0, sizeSimilarityScore = 0, visualSimilarityScore = 0;
	/// Set when IOU rescue fires, checked in SIZE_SIMILARITY gate.
	bool iouRescueFired = false;
	for (auto it = similarityParamList.begin(); it != similarityParamList.end(); it++)
	{
		const SimilarityParam& similarityParam = *it;
		float score;
		bool isInterTrackerVisualSimilarityScore = false;
		switch (similarityParam.type)
		{
			case SimilarityParam::TYPES::IOU:
				score = CV_FUNCS::computeIOU(tBbox, detObj.bbox);
				iouScore = score;
				break;
			case SimilarityParam::TYPES::SIZE_SIMILARITY:
				score = CV_FUNCS::computeSizeScore(tBbox, detObj.bbox);
				sizeSimilarityScore = score;
				break;
			case SimilarityParam::TYPES::VISUAL_SIMILARITY:
				if(sumCorrRespAtDetectedPt > 0)
				{
					/// Consider inter tracker visual similarity measure.
					float interTrackerScore = (target->getCorrRespAtDetectedPts().size() > detObj.bboxInd) ? \
						target->getCorrRespAtDetectedPts().at(detObj.bboxInd) / sumCorrRespAtDetectedPt : 0.f;
					float intraTrackerScore = (target->getConfidenceCorrRespAtDetectedPts().size() > detObj.bboxInd) ? \
						target->getConfidenceCorrRespAtDetectedPts().at(detObj.bboxInd) / target->getTrackerConfidence() : 0.f;
					score = (intraTrackerScore > similarityParam.minVal) ? interTrackerScore : 0.f;
					isInterTrackerVisualSimilarityScore = true;
				}
				else
				{
					score = (target->getConfidenceCorrRespAtDetectedPts().size() > detObj.bboxInd) ? \
						target->getConfidenceCorrRespAtDetectedPts().at(detObj.bboxInd) / target->getTrackerConfidence() : 0.f;
				}
				visualSimilarityScore = score;
				break;
			case SimilarityParam::TYPES::REID_SIMILARITY:
				score = (target->getConfidenceReidDets().size() > detObj.bboxInd) ? \
					target->getConfidenceReidDets().at(detObj.bboxInd) : 0.f;
				break;
			case SimilarityParam::TYPES::WORLD_PROXIMITY:
				/// Stage Second world-proximity score: score = 1 - dist/maxWorldDist.
				if (worldDistSaved >= std::numeric_limits<float>::max() / 2.0f
					|| m_params.m_maxWorldDist4PrimaryAssoc <= 0.0f) {
					score = -1.0f;
				} else if (target->getTrackerState() == TRACKER_STATE::INACTIVE
						&& m_params.m_maxWorldDist4ActiveIouRescue > 0.0f
						&& worldDistSaved >= m_params.m_maxWorldDist4ActiveIouRescue) {
					/// Tight gate for INACTIVE: reject beyond maxWorldDist4ActiveIouRescue.
					score = -1.0f;
				} else {
					score = std::max(0.0f, 1.0f - worldDistSaved / m_params.m_maxWorldDist4PrimaryAssoc);
				}
				break;
			default:
				score = 0.f;
		}
		if (!isInterTrackerVisualSimilarityScore && (score < similarityParam.minVal))
		{
			/// Partial-overlap rescue for INACTIVE Stage 3 pairs.
			if (similarityParam.type == SimilarityParam::TYPES::IOU
				&& isIouOnlyStage
				&& m_params.m_minMatchingScore4TentativeIouInactive > 0.0f
				&& target->getTrackerState() == TRACKER_STATE::INACTIVE
				&& score >= m_params.m_minMatchingScore4TentativeIouInactive
				&& (worldDistSaved >= std::numeric_limits<float>::max() / 2.0f // No world pos → accept IoU-only.
				|| worldDistSaved < m_params.m_maxWorldDist4ActiveIouRescue)) // Require world proximity.
			{
				totalScore += similarityParam.weight * score;
				/// Partial IoU overlap accepted: bypass hard gate.
				continue;
			}
			/// QUASIACTIVE world-space rescue for IOU gate failures.
			if (similarityParam.type == SimilarityParam::TYPES::IOU
				&& target->getTrackerState() == TRACKER_STATE::QUASIACTIVE
				&& worldDistSaved < m_params.m_maxWorldDist4QuasiactiveRescue)
			{
				score = (1.0f - worldDistSaved / m_params.m_maxWorldDist4QuasiactiveRescue) * similarityParam.minVal;
				totalScore += similarityParam.weight * score;
				iouRescueFired = true;
				m_da_stats.qactive_rescue_attempted++;
			} else if (similarityParam.type == SimilarityParam::TYPES::IOU
					&& isIouOnlyStage
					&& m_params.m_maxWorldDist4ActiveIouRescue > 0.0f
					&& (target->getTrackerState() == TRACKER_STATE::ACTIVE
						|| target->getTrackerState() == TRACKER_STATE::INACTIVE))
			{
				/// ACTIVE/INACTIVE IoU rescue in Stage 3 (IoU-only stage).
				const int shadowAge371 = (int)target->getShadowTrackingAge();
				const float ageMult371 = (target->getTrackerState() == TRACKER_STATE::INACTIVE)
					? std::min(1.0f + shadowAge371 * 0.2f, 2.0f)
					: 1.0f;
				const float rescueGate371 = m_params.m_maxWorldDist4ActiveIouRescue * ageMult371;
				if (worldDistSaved < rescueGate371) {
					score = (1.0f - worldDistSaved / rescueGate371) * similarityParam.minVal;
					totalScore += similarityParam.weight * score;
					iouRescueFired = true;
				} else {
					m_da_stats.gate_hard++;
					return 0.f;
				}
			} else {
				/// Count IOU hard gate for QUASIACTIVE when rescue doesn't fire.
				if (similarityParam.type == SimilarityParam::TYPES::IOU
					&& target->getTrackerState() == TRACKER_STATE::QUASIACTIVE) {
					m_da_stats.qactive_iou_hardgate++;
				}
				/// Counts IoU hard-gate failures for projective ACTIVE tracks and INACTIVE tracks with shadow age at most 2.
				if (similarityParam.type == SimilarityParam::TYPES::IOU
					&& target->usesProjectiveEstimator()
					&& (target->getTrackerState() == TRACKER_STATE::ACTIVE
						|| (target->getTrackerState() == TRACKER_STATE::INACTIVE
							&& (int)target->getShadowTrackingAge() <= 2))) {
					m_da_stats.active_iou_hardgate++;
				}
				/// Bypasses the size gate when worldDist is below m_maxWorldDist4SizeBypass after a QUASIACTIVE IoU rescue within m_maxWorldDist4QuasiactiveRescue.
				if (similarityParam.type == SimilarityParam::TYPES::SIZE_SIMILARITY && iouRescueFired) {
					m_da_stats.qactive_size_rescued++;
					if (worldDistSaved < m_params.m_maxWorldDist4SizeBypass) {
						score = (1.0f - worldDistSaved / m_params.m_maxWorldDist4SizeBypass) * similarityParam.minVal;
						totalScore += similarityParam.weight * score;
						/// Bypass gate_hard++/return 0.f: proceed to next param.
						continue;
					}
				}
				/// Count which Stage Second feature rejected the pair.
				if (isS2) {
					if (similarityParam.type == SimilarityParam::TYPES::WORLD_PROXIMITY)
						m_da_stats.stage2_wp_gate++;
					else if (similarityParam.type == SimilarityParam::TYPES::SIZE_SIMILARITY) {
						m_da_stats.stage2_size_gate++;
						/// Stage Second SIZE bypass at worldDist<m_maxWorldDist4SizeBypass: wP already passed (valid world positions).
						if (worldDistSaved < m_params.m_maxWorldDist4SizeBypass
							&& target->getTrackerState() != TRACKER_STATE::INACTIVE) {
							m_da_stats.stage2_size_rescued++;
							score = (1.0f - worldDistSaved / m_params.m_maxWorldDist4SizeBypass) * similarityParam.minVal;
							totalScore += similarityParam.weight * score;
							/// Bypass SIZE hard gate at worldDist<threshold: s2 unmatched dets only.
							continue;
						}
					}
				}
				/// Individual-score hard gate (IoU/size/visual min).
				m_da_stats.gate_hard++;
				return 0.f;
			}
		}
		else
		{
			totalScore += similarityParam.weight * score;
		}
	}

	/// DEBUG.
	if(m_params.m_debugVerboseLevel > 3)
	{
		const TargetInfo &tInfo = target->getTargetInfo();

		std::stringstream msg;
		msg << "[DataAssociator::AssocDistanceFunc] prev-frameNum[" << tInfo.frameNum
				<< "] detObjInd[" << detObj.bboxInd << "] (confidence=" << detObj.confidence << ") --> TrackerInd[" << target->getTargetInd() << "],ID[" << target->getTargetID()
				<< "]: Scores (iou, size, visual similarity) = ("
				<< iouScore << ", " << sizeSimilarityScore << ", " << visualSimilarityScore << ")"
				<< " --> Total score = " << totalScore;

		cout << msg.str();
	}

	/// DEBUG.
	if(m_params.m_debugVerboseLevel > 3)
	{
		cout << " ---> " << totalScore << endl;
	}

	if (totalScore < m_params.m_minMatchingScore4Overall)
	{
		/// Near-miss bucketing: passed all individual gates but below overall threshold.
		if      (totalScore >= 0.5f) m_da_stats.near_ge50++;
		else if (totalScore >= 0.4f) m_da_stats.near_04_05++;
		else                         m_da_stats.near_lt04++;
		return 0.f;
	}
	else
	{
		m_da_stats.passed++;
		return totalScore;
	}
}

/**
 * @brief Compute the association score matrix.
 * @param[in] in Association inputs.
 * @param[in] stage Association stage.
 * @return The computed score matrix, where higher values indicate closer matches.
 */
std::vector<std::vector<float>> DataAssociator::computeDistanceMatrix(const assocDataIn_t &in, const uint stage) const
{
	std::vector<std::vector<float>> distMat(in.detObjs.size(), std::vector<float>(in.targets.size(), 0.f));

	if(in.detObjs.size() == 0 || in.targets.size() == 0)
		return distMat;

	/// Copy the stage params under lock, then release before the O(D×T) scoring loop so per-stream association workers can run concurrently.
	std::vector<SimilarityParam> stageParams;
	{
		std::unique_lock<std::mutex> rlock(m_similarityParamsMutex);
		if (m_similarityParamLists.size() <= stage)
		{
			throw std::runtime_error("[DataAssociator::computeDistanceMatrix] Invalid similarity params");
		}
		stageParams = m_similarityParamLists.at(stage);
	}

	/// Construct distance matrix.
	for(uint detInd = 0; detInd < in.detObjs.size(); detInd++)
	{
		const DetectedObjInfo *detObj = in.detObjs[detInd];
		uint inTargetListInd = 0;
		/// Sum of all the tracker correlations at detected object index.
		float sumCorrRespAtDetectedPt = 0;
		if(in.sumsCorrRespAtDetectedPt.size() > 0)
		{
			if(in.sumsCorrRespAtDetectedPt.find(detObj->bboxInd) != in.sumsCorrRespAtDetectedPt.end())
			{
				sumCorrRespAtDetectedPt = in.sumsCorrRespAtDetectedPt.at(detObj->bboxInd);
				if(sumCorrRespAtDetectedPt <= 0)
				{
					throw std::runtime_error(
						"[DataAssociator::computeDistanceMatrix] sumsCorrRespAtDetectedPt should be a non-zero positive value!!"
					);
				}
			}
			else
			{
				throw std::runtime_error(
					"[DataAssociator::computeDistanceMatrix] sumsCorrRespAtDetectedPt doesn't contain a required entry!!"
				);
			}
		}
		for(auto tIt = in.targets.begin(); tIt != in.targets.end(); ++tIt, ++inTargetListInd)
		{
			const std::shared_ptr<NvTrackedObject> target = tIt->second;
			/// Matching score function NOTE: You can define your own matching score function.
			distMat[detInd][inTargetListInd] = assocDistanceFunc(target, *detObj, stageParams, sumCorrRespAtDetectedPt);
		}
	}

	/// DEBUG: Print out the data association matrix.
	if(m_params.m_debugVerboseLevel > 2)
	{
		dumpAssocDistanceMatrix(in, distMat);
	}

	return distMat;
}

/**
 * @brief Sort potential matches by association score.
 * @param[in] in Association inputs.
 * @param[in] distMat Score matrix, where higher values indicate closer matches.
 * @return The sorted list of potential match pairs { matching score, {inDetBboxListInd, inTargetListInd}}.
 */
std::multimap<float, std::pair<uint, int>> DataAssociator::sortMatching(
	const assocDataIn_t &in, const std::vector<std::vector<float>> &distMat
) const
{
	std::multimap<float, std::pair<uint, int>> sortedMatching;

	for(uint inDetBboxListInd = 0; inDetBboxListInd < in.detObjs.size(); inDetBboxListInd++)
	{
		for(uint inTargetListInd = 0; inTargetListInd < in.targets.size(); inTargetListInd++)
		{
			float assocScore = distMat[inDetBboxListInd][inTargetListInd];
			if(assocScore > 0 && assocScore >= m_params.m_minMatchingScore4Overall)
			{
				sortedMatching.insert(std::make_pair(assocScore, std::make_pair(inDetBboxListInd, inTargetListInd)));
			}
		}
	}

	return sortedMatching;
}

/**
 * @brief Find the best matches between detections and existing targets.
 * @param[in] in Association inputs.
 * @param[in] stage Association stage.
 * @return The association results.
 */
assocDataOut_t DataAssociator::associatePerStream_greedy(const assocDataIn_t& in, const uint stage)
{
	/// Data association output.
	assocDataOut_t out;
	out.isAssocDone = true;
	out.videoStreamId = in.videoStreamId;
	out.frameNum = in.frameNum;
	out.timeMeta = in.timeMeta;

	/// 2D distance matrix {detBboxInd, {targetInd, distance}}.
	std::vector<std::vector<float>> distMat = computeDistanceMatrix(in, stage);

	/// { matching score, {detBboxInd, inTargetListInd}}.
	std::multimap<float, std::pair<uint, int>> sortedMatching = sortMatching(in, distMat);

	/// A mapping from the index for in.targets[] to targetInd.
	std::map<uint, uint> mappingInTargetListInd2TargetInd;
	{
		uint inTargetListInd = 0;
		for(auto tIt = in.targets.begin(); tIt != in.targets.end(); ++tIt, ++inTargetListInd)
		{
			const uint32_t targetInd = tIt->first;
			mappingInTargetListInd2TargetInd[inTargetListInd] = targetInd;
		}
	}

	/// Index in.detObjs: note: This is different from bboxInd for the inputMeta.
	std::set<uint32_t> matchedDetBboxListInds;
	/// Index in.detObjs: note: This is different from bboxInd for the inputMeta.
	std::set<uint32_t> unmatchedDetBboxListInds;
	/// Index in.targets: note: This is different from targetInd for targets.
	std::set<uint32_t> unmatchedTargetListInds;

	/// Fill unmatchedTargetListInds list.
	{
		uint32_t inTargetListInd = 0;
		for(auto targetIt = in.targets.begin(); targetIt != in.targets.end(); ++targetIt, ++inTargetListInd)
		{
			unmatchedTargetListInds.insert(inTargetListInd);
		}
	}

	/// Fill unmatchedDetBboxListInds list.
	{
		uint32_t inDetBboxListInd = 0;
		for(auto dIt = in.detObjs.begin(); dIt != in.detObjs.end(); ++dIt, ++inDetBboxListInd)
		{
			unmatchedDetBboxListInds.insert(inDetBboxListInd);
		}
	}

	/// Iterate from the highest score.
	for(auto rit = sortedMatching.rbegin(); rit != sortedMatching.rend(); ++rit)
	{
		const float assocScore = rit->first;
		const uint32_t inDetBboxListInd = rit->second.first;
		const int inTargetListInd = rit->second.second;
		const uint32_t targetInd = mappingInTargetListInd2TargetInd[inTargetListInd];

		/// Stop processing of the too-low-score matches.
		if(assocScore <= 0 || assocScore < m_params.m_minMatchingScore4Overall)
			break;

		/// Once a detector bbox is matched, don't need to try to match.
		if(matchedDetBboxListInds.find(inDetBboxListInd) != matchedDetBboxListInds.end())
			continue;

		/// Check if the matched target is a valid one and if the matched target has not been matched yet.
		if( inTargetListInd >= 0
				&& unmatchedTargetListInds.find(inTargetListInd) != unmatchedTargetListInds.end())
		{
			/// If the max target is still available to be matched, then match them.
			out.matchedPairs.push_back(std::make_pair(in.detObjs[inDetBboxListInd], targetInd));
			unmatchedTargetListInds.erase(inTargetListInd);

			matchedDetBboxListInds.insert(inDetBboxListInd);
			unmatchedDetBboxListInds.erase(inDetBboxListInd);
		}
	}

	/// In case there are still some unmatched bboxes and some unmatched targets, try to match them.
	if(unmatchedDetBboxListInds.size() > 0 && unmatchedTargetListInds.size() > 0)
	{
		bool isSecondPassMatchFound = false;

		/// DEBUG.
		if(m_params.m_debugVerboseLevel > 2)
		{
			cout << "Stream[" << in.videoStreamId << "] frameNum[" << in.frameNum
					<< "] (# of unmatchedDetObjs, # of unmatchedTargets) = ("
					<< unmatchedDetBboxListInds.size() << ", " << unmatchedTargetListInds.size() << ")" << endl;

			/// Fill out the output results for proper printing the results.
			for(auto dIt = unmatchedDetBboxListInds.begin(); dIt != unmatchedDetBboxListInds.end(); ++dIt)
			{
				const uint32_t inDetBboxListInd = *dIt;
				out.unmatchedDetObjs[inDetBboxListInd] = in.detObjs[inDetBboxListInd];
			}

			for(auto tIt = unmatchedTargetListInds.begin(); tIt != unmatchedTargetListInds.end(); ++tIt)
			{
				const uint32_t inTargetListInd = *tIt;
				const uint32_t targetInd = mappingInTargetListInd2TargetInd[inTargetListInd];
				out.unmatchedTargets.insert(targetInd);
			}
		}

		/// For each unmatched detector bbox, try to find a qualified match to an unmatched target.
		for(auto dIt = unmatchedDetBboxListInds.begin(); dIt != unmatchedDetBboxListInds.end(); )
		{
			const uint32_t inDetBboxListInd = *dIt;
			auto nextdIt = std::next(dIt, 1);

			float maxAssocScore = -9999.f;
			uint32_t maxInTargetListInd = 0;

			for(auto tIt = unmatchedTargetListInds.begin(); tIt != unmatchedTargetListInds.end(); ++tIt)
			{
				const TrackerIndex inTargetListInd = *tIt;

				float assocScore = distMat[inDetBboxListInd][inTargetListInd];
				if(maxAssocScore < assocScore)
				{
					maxAssocScore = assocScore;
					maxInTargetListInd = inTargetListInd;
				}
			}

			if(maxAssocScore > 0 && maxAssocScore >= m_params.m_minMatchingScore4Overall)
			{
				const DetectedObjInfo *detObj = in.detObjs[inDetBboxListInd];
				const TrackerIndex targetInd = mappingInTargetListInd2TargetInd[maxInTargetListInd];

				out.matchedPairs.push_back(std::make_pair(detObj, targetInd));
				unmatchedTargetListInds.erase(maxInTargetListInd);
				unmatchedDetBboxListInds.erase(dIt);

				/// DEBUG.
				if(m_params.m_debugVerboseLevel > 3)
				{
					cout << " [Second Pass] New Matched pairs (detBboxInd, targetInd): ("
						<< detObj->bboxInd << ", " << targetInd << ") w/ assocScore = " << maxAssocScore << endl;
				}

				isSecondPassMatchFound = true;
			}

			dIt = nextdIt;
		}

		/// DEBUG.
		if(m_params.m_debugVerboseLevel > 2)
			if(isSecondPassMatchFound)
			{
				cout << "Stream[" << in.videoStreamId << "] frameNum[" << in.frameNum
					<< "] (# of unmatchedDetObjs, # of unmatchedTargets) = ("
					<< unmatchedDetBboxListInds.size() << ", " << unmatchedTargetListInds.size() << ")" << endl;

				dumpDataAssociationResults(out);
			}
	}

	/// Fill out the output results.
	out.unmatchedDetObjs.clear();
	for(auto dIt = unmatchedDetBboxListInds.begin(); dIt != unmatchedDetBboxListInds.end(); ++dIt)
	{
		const uint32_t inDetBboxListInd = *dIt;
		out.unmatchedDetObjs[inDetBboxListInd] = in.detObjs[inDetBboxListInd];
	}
	out.unmatchedTargets.clear();
	for(auto tIt = unmatchedTargetListInds.begin(); tIt != unmatchedTargetListInds.end(); ++tIt)
	{
		const uint32_t inTargetListInd = *tIt;
		const uint32_t targetInd = mappingInTargetListInd2TargetInd[inTargetListInd];
		out.unmatchedTargets.insert(targetInd);
	}

	return out;
}

/**
 * @brief Run the cascaded association algorithm.
 * @param[in] in Association inputs.
 * @return The association results.
 */
assocDataOut_t DataAssociator::associatePerStream_cascaded(const assocDataIn_t& in)
{
	std::deque<const DetectedObjInfo *> detObjsZeroth, detObjsFirst, detObjsThird, detObjsFourth;
	NvTrackedObjectList targetsZeroth, targetsFirst, targetsThird, targetsFourth;

	/// Split detection inputs into valid and tentative based on tentativeDetectorConfidence.
	for (auto dIt = in.detObjs.begin(); dIt != in.detObjs.end(); ++dIt)
	{
		if ((*dIt)->confidence > m_params.m_tentativeDetectorConfidence)
		{
			detObjsFirst.push_back(*dIt);
		}
		else
		{
			detObjsThird.push_back(*dIt);
		}
	}

	/// Splits targets into ACTIVE, INACTIVE, and QUASIACTIVE versus TENTATIVE.
	for (auto targetIt = in.targets.begin(); targetIt != in.targets.end(); targetIt++)
	{
		TrackerIndex targetInd = targetIt->first;
		if	(	in.targets.at(targetInd)->getTrackerState() == TRACKER_STATE::ACTIVE ||
				in.targets.at(targetInd)->getTrackerState() == TRACKER_STATE::INACTIVE ||
				in.targets.at(targetInd)->getTrackerState() == TRACKER_STATE::QUASIACTIVE
		   	)
		{
			targetsFirst[targetInd] = in.targets.at(targetInd);
		}
		else
		{
			/// Tentative state.
			targetsFourth[targetInd] = in.targets.at(targetInd);
		}
	}

	/// Add a zeroth stage to address the case: "multiple overlapping trackers and one (or fewer) detector bbox" Zeroth stage: detections: valid & (correlation response > 0 with more than one targets) Subset of detObjsFirst targets: active and inactive & correlated to detections at detObjsZeroth Subset of targetsFirst similarity: visual + motion similarity.
	std::map<uint16_t, float> sumsCorrRespAtDetectedPt;
	for (auto dIt = detObjsFirst.begin(); dIt != detObjsFirst.end(); ++dIt)
	{
		/// All trackers with (correlation response > 0) at detected object.
		std::vector<uint32_t> trackersList = {};
		/// Sum of all the tracker correlation responses at detected object.
		float sumCorrRespAtDetectedPt = 0;
		for (auto tIt = targetsFirst.begin(); tIt != targetsFirst.end(); ++tIt)
		{
			uint32_t targetInd = tIt->first;
			std::shared_ptr<NvTrackedObject> target =  tIt->second;
			float corrResp = (target->getCorrRespAtDetectedPts().size() > (*dIt)->bboxInd) ? target->getCorrRespAtDetectedPts().at((*dIt)->bboxInd) : 0.f;
			if(corrResp > 0)
			{
				sumCorrRespAtDetectedPt += corrResp;
				trackersList.push_back(targetInd);
			}
		}
		/// If more than one trackers have a correlation with detected object, add it to zeroth stage.
		if(trackersList.size() > 1)
		{
			detObjsZeroth.push_back(*dIt);
			sumsCorrRespAtDetectedPt[(*dIt)->bboxInd] = sumCorrRespAtDetectedPt;
			for (uint32_t targetCnt = 0; targetCnt < trackersList.size(); targetCnt++)
			{
				uint32_t targetInd = trackersList[targetCnt];
				targetsZeroth[targetInd] = in.targets.at(targetInd);
			}
		}
	}
	assocDataIn_t inZeroth = {
		in.videoStreamId,
		in.frameNum,
		in.timeMeta,
		in.isDetDone,
		std::move(detObjsZeroth),
		std::move(targetsZeroth),
		std::move(sumsCorrRespAtDetectedPt),
		nullptr
	};
	assocDataOut_t outZeroth = associatePerStream_greedy(inZeroth);
	/// Delete the matched detected objects and targets from first stage inputs.
	for (auto matchedPairs = outZeroth.matchedPairs.begin(); matchedPairs != outZeroth.matchedPairs.end(); matchedPairs++)
	{
		const DetectedObjInfo *detObjInfo = matchedPairs->first;
		uint32_t targetInd = matchedPairs->second;
		/// Remove detected object from the list.
		auto removeEntry = std::remove(detObjsFirst.begin(), detObjsFirst.end(), detObjInfo);
		detObjsFirst.erase(removeEntry, detObjsFirst.end());
		/// Remove tracker from the list.
		targetsFirst.erase(targetInd);
	}

	/// First stage: detections: valid targets: active and inactive similarity: visual + motion similarity.
	assocDataIn_t inFirst = {
		in.videoStreamId,
		in.frameNum,
		in.timeMeta,
		in.isDetDone,
		std::move(detObjsFirst),
		std::move(targetsFirst),
		{},
		nullptr
	};
	assocDataOut_t outFirst = associatePerStream_greedy(inFirst);

	/// Stage Second: QUASIACTIVE-only uncontested second chance using WORLD_PROXIMITY.
	if (m_params.m_maxWorldDist4PrimaryAssoc > 0.0f)
	{
		NvTrackedObjectList targetsSecond;
		for (TrackerIndex tInd : outFirst.unmatchedTargets)
		{
			if (in.targets.at(tInd)->getTrackerState() == TRACKER_STATE::QUASIACTIVE)
				targetsSecond[tInd] = in.targets.at(tInd);
		}
		/// Count the QUASIACTIVE targets eligible for Stage Second.
		m_da_stats.stage2_eligible += (uint64_t)targetsSecond.size();
		if (!targetsSecond.empty() && !outFirst.unmatchedDetObjs.empty())
		{
			/// Count frames where Stage Second actually runs + total pair surface area.
			m_da_stats.stage2_has_dets++;
			m_da_stats.stage2_pair_total += (uint64_t)targetsSecond.size() * (uint64_t)outFirst.unmatchedDetObjs.size();
			std::deque<const DetectedObjInfo *> detObjsSecond;
			for (auto& kv : outFirst.unmatchedDetObjs)
				detObjsSecond.push_back(kv.second);
			assocDataIn_t inSecond = {
				in.videoStreamId, in.frameNum, in.timeMeta, in.isDetDone,
				std::move(detObjsSecond), std::move(targetsSecond), {}, nullptr
			};
			assocDataOut_t outSecond = associatePerStream_greedy(inSecond, 2);
			m_da_stats.stage2_matched += (uint64_t)outSecond.matchedPairs.size();
			for (auto& mp : outSecond.matchedPairs)
			{
				const TrackerIndex tInd = mp.second;
				const DetectedObjInfo* detPtr = mp.first;
				outFirst.unmatchedTargets.erase(tInd);
				for (auto dIt = outFirst.unmatchedDetObjs.begin(); dIt != outFirst.unmatchedDetObjs.end(); ++dIt)
				{
					if (dIt->second == detPtr) { outFirst.unmatchedDetObjs.erase(dIt); break; }
				}
				outFirst.matchedPairs.push_back(mp);
			}
		}
		else if (!targetsSecond.empty())
		{
			/// Pool non-empty but no unmatched dets.
			m_da_stats.stage2_no_dets++;
		}
	}

	/// Regarding unmatched targets from Stage First, fill them as input to Stage Third.
	for (auto targetIt = outFirst.unmatchedTargets.begin(); targetIt != outFirst.unmatchedTargets.end(); targetIt++)
	{
		TrackerIndex targetInd = *targetIt;
		targetsThird[targetInd] = in.targets.at(targetInd);
	}

	/// Third stage: detections: tentative targets: all unmatched targets from Stage First (after Stage Second) similarity: iou only.
	assocDataIn_t inThird = {
		in.videoStreamId,
		in.frameNum,
		in.timeMeta,
		in.isDetDone,
		std::move(detObjsThird),
		std::move(targetsThird),
		{},
		nullptr
	};
	assocDataOut_t outThird = associatePerStream_greedy(inThird, 1);

	for (auto dIt = outFirst.unmatchedDetObjs.begin(); dIt != outFirst.unmatchedDetObjs.end(); ++dIt)
	{
		detObjsFourth.push_back(dIt->second);
	}

	/// Fourth stage: detections: valid remain from first stage targets: tentative similarity: if useSimpleIou4Tentative, iou only.
	assocDataIn_t inFourth = {
		in.videoStreamId,
		in.frameNum,
		in.timeMeta,
		in.isDetDone,
		std::move(detObjsFourth),
		std::move(targetsFourth),
		{},
		nullptr
	};
	assocDataOut_t outFourth = associatePerStream_greedy(inFourth, m_params.m_useSimpleIou4Tentative ? 1 : 0);

	/// Construct final output from all stages.
	assocDataOut_t out;
	out.videoStreamId = outFirst.videoStreamId;
	out.frameNum = outFirst.frameNum;
	out.isAssocDone = outFirst.isAssocDone;
	out.timeMeta = in.timeMeta;

	int dInd = 0;
	/// DetectorInd, unmatched detector objects (i.e., new objects).
	std::deque<const DetectedObjInfo*> peerAssocDetObjsIn;
	for (auto it = outFourth.unmatchedDetObjs.begin(); it != outFourth.unmatchedDetObjs.end(); it++)
	{
		if (it->second->confidence >= m_params.m_tentativeDetectorConfidence)
			out.unmatchedDetObjs[dInd++] = it->second;
		else
			peerAssocDetObjsIn.push_back(it->second);
	}

	if (peerAssocDetObjsIn.size() > 0 && m_params.m_minMatchingScore4PeerAssocIou < 1.0 && peerAssocDetObjsIn.size() > 0)
	{
		/// For unmatched tentative detections, try associate them with peer targets.
		peerAssocDetObjsIn = in.multiViewAssociator->tryPeerDataAssoc(
			peerAssocDetObjsIn, m_params.m_minMatchingScore4PeerAssocIou
		);

		dInd = out.unmatchedDetObjs.size();
		for (auto& det : peerAssocDetObjsIn)
		{
			out.unmatchedDetObjs[dInd++] = det;
		}
	}

	out.matchedPairs = outZeroth.matchedPairs;
	out.matchedPairs.insert(out.matchedPairs.end(), outFirst.matchedPairs.begin(), outFirst.matchedPairs.end());
	out.matchedPairs.insert(out.matchedPairs.end(), outThird.matchedPairs.begin(), outThird.matchedPairs.end());
	out.matchedPairs.insert(out.matchedPairs.end(), outFourth.matchedPairs.begin(), outFourth.matchedPairs.end());

	for (auto targetIt = outFirst.unmatchedTargets.begin(); targetIt != outFirst.unmatchedTargets.end(); targetIt++)
	{
		TrackerIndex targetInd = *targetIt;
		if ( in.targets.at(targetInd)->getTrackerState() == TRACKER_STATE::INACTIVE )
		{
			out.unmatchedTargets.insert(targetInd);
		}
	}
	out.unmatchedTargets.insert(outThird.unmatchedTargets.begin(), outThird.unmatchedTargets.end());
	out.unmatchedTargets.insert(outFourth.unmatchedTargets.begin(), outFourth.unmatchedTargets.end());

	return out;
}

/**
 * @brief Run the legacy two-stage association algorithm.
 * @param[in] in Association inputs.
 * @return The association results.
 */
assocDataOut_t DataAssociator::associatePerStream_cascadedLegacy(const assocDataIn_t& in)
{
	/// Split assoc inputs for two stages First stage: detection confidence > m_thresholdDetectorConfidenceSecond Remain low confidence detections go into second stage.
	std::deque<const DetectedObjInfo *> detObjsFirst, detObjsSecond;
	for (auto dIt = in.detObjs.begin(); dIt != in.detObjs.end(); ++dIt)
	{
		if ((*dIt)->confidence > m_params.m_tentativeDetectorConfidence)
		{
			detObjsFirst.push_back(*dIt);
		}
		else
		{
			detObjsSecond.push_back(*dIt);
		}
	}

	/// First stage: detections: confidence > m_thresholdDetectorConfidenceSecond targets: all distance: original NvDCF or NvDeepSORT distance.
	assocDataIn_t inFirst = {
		in.videoStreamId,
		in.frameNum,
		in.timeMeta,
		in.isDetDone,
		std::move(detObjsFirst),
		in.targets,
		{},
		nullptr
	};
	assocDataOut_t outFirst = associatePerStream_greedy(inFirst);

	/// Regarding unmatched targets from first stage, fill them as input to second stage.
	NvTrackedObjectList targetsSecond;
	for (auto targetIt = outFirst.unmatchedTargets.begin(); targetIt != outFirst.unmatchedTargets.end(); targetIt++)
	{
		uint32_t targetInd = *targetIt;
		targetsSecond[targetInd] = in.targets.at(targetInd);
	}

	/// Second stage: detections: confidence <= m_thresholdDetectorConfidenceSecond targets: remain from first stage distance: iou only.
	assocDataIn_t inSecond = {
		in.videoStreamId,
		in.frameNum,
		in.timeMeta,
		in.isDetDone,
		std::move(detObjsSecond),
		std::move(targetsSecond),
		{},
		nullptr
	};
	assocDataOut_t outSecond = associatePerStream_greedy(inSecond, true);

	/// Construct final output.
	assocDataOut_t out = std::move(outFirst);
	/// Merge matched pairs in both stages.
	out.matchedPairs.insert(out.matchedPairs.end(), outSecond.matchedPairs.begin(), outSecond.matchedPairs.end());
	/// UnmatchedDetObjs are first stage only, so no need to change unmatchedTargets are from second stage.
	out.unmatchedTargets = std::move(outSecond.unmatchedTargets);
	return out;
}

/**
 * @brief Print data-association results for debugging.
 * @param[in] out Association results to print.
 */
void DataAssociator::dumpDataAssociationResults(const assocDataOut_t &out) const
{
	std::stringstream msg;
	msg << DEBUG_UTIL::to_string_fmt("- For video stream [%ld] frameNum[%d]:\n", out.videoStreamId, out.frameNum);

	/// Show matched pairs.
	msg << " - Matched pairs (detBboxInd, targetInd): ";
	for(auto it = out.matchedPairs.begin(); it != out.matchedPairs.end(); ++it)
	{
		const uint detInd = (*it).first->bboxInd;
		const uint targetInd = (*it).second;

		msg << DEBUG_UTIL::to_string_fmt("(%d, %d) ", detInd, targetInd);
	}
	msg << endl;

	/// Show unmatched detector bboxes.
	msg << " - Unmatched detector bboxInd: ";
	for(auto it = out.unmatchedDetObjs.begin(); it != out.unmatchedDetObjs.end(); ++it)
	{
		const uint detInd = it->second->bboxInd;
		msg << detInd << " ";
	}
	msg << endl;

	/// Show unmatched targets.
	msg << " - Unmatched target Inds: ";
	for(auto it = out.unmatchedTargets.begin(); it != out.unmatchedTargets.end(); ++it)
	{
		const uint targetInd = *it;
		msg << targetInd << " ";
	}
	msg << endl;

	cout << msg.str();
}

/**
 * @brief Print an association score matrix for debugging.
 * @param[in] in Association inputs used to label the matrix.
 * @param[in] distMat Score matrix to print.
 */
void DataAssociator::dumpAssocDistanceMatrix(const assocDataIn_t &in, std::vector<std::vector<float>> distMat) const
{
	std::stringstream msg;
	msg << DEBUG_UTIL::to_string_fmt("- For video stream [%ld] frameNum[%d]:\n", in.videoStreamId, in.frameNum);

	{
		msg << "     ";
		for(auto tIt = in.targets.begin(); tIt != in.targets.end(); ++tIt)
		{
			const uint trackerInd = tIt->first;
			msg << DEBUG_UTIL::to_string_fmt(" t[%2d]", trackerInd);
		}
		msg << endl;

		/// For each detector bounding box in a video stream.
		for(uint detInd = 0; detInd < distMat.size(); detInd++)
		{
			msg << DEBUG_UTIL::to_string_fmt("d[%2d]", detInd);
			/// For each tracker.
			uint inTargetListInd = 0;
			for(auto tIt = in.targets.begin(); tIt != in.targets.end(); ++tIt, ++inTargetListInd)
			{
				float score = distMat[detInd][inTargetListInd];
				if(score > 0)
					msg << DEBUG_UTIL::to_string_fmt(" %.3f", score);
				else
					msg << "     0";
			}
			msg << endl;
		}
		msg << endl;
	}

	cout << msg.str();
}

/**
 * @brief Associate detections with existing targets for one stream.
 * @param[in] inData Newly detected object bounding boxes and existing targets.
 * @return The association results.
 */
assocDataOut_t DataAssociator::associatePerStream(assocDataIn_t inData)
{
	/// Output will be an empty one.
	assocDataOut_t assocOutput;
	assocOutput.videoStreamId = inData.videoStreamId;
	assocOutput.frameNum = inData.frameNum;
	assocOutput.timeMeta = inData.timeMeta;

	/// Submit the matching job only if det is done and there are valid objects in both sides.
	if(!inData.isDetDone)
	{
		/// Move det not done case here.
		assocOutput.isAssocDone = false;
	}
	else if(inData.targets.size() == 0 && inData.detObjs.size() == 0)
	{
		assocOutput.isAssocDone = true;
	}
	else if(inData.targets.size() == 0)
	{
		assocOutput.isAssocDone = true;

		/// Fill new object info.
		uint32_t detInd = 0;
		for(auto dIt = inData.detObjs.begin(); dIt != inData.detObjs.end(); ++dIt, ++detInd)
		{
			assocOutput.unmatchedDetObjs[detInd] = *dIt;
		}
	}
	else if(inData.detObjs.size() == 0)
	{
		assocOutput.isAssocDone = true;

		/// Fill the unmatched tracker info.
		for(auto tIt = inData.targets.begin(); tIt != inData.targets.end(); ++tIt)
		{
			assocOutput.unmatchedTargets.insert(tIt->first);
		}
	}
	else
	{
		/// Run data association.
		if (m_params.m_associationMatcherType == ASSOCIATION_MATCHER_TYPE::CASCADED)
		{
			assocOutput = associatePerStream_cascaded(inData);
		}
		else
		{
			assocOutput = associatePerStream_greedy(inData);
		}
	}
	return assocOutput;
}

/** @brief Initialize association parameters with default values. */
NvTrackerParamsAssoc::NvTrackerParamsAssoc()
{
	m_debugVerboseLevel = 0;
	m_dataAssociatorType = DATA_ASSOCIATOR_TYPES::DEFAULT_ASSOCIATOR;
	m_associationMatcherType = ASSOCIATION_MATCHER_TYPE::GREEDY;
	m_usePrediction4Assoc = false;
	m_checkClassMatch = true;

	m_minMatchingScore4Overall = 0.0f;
	m_minMatchingScore4Iou = 0.0f;
	m_minMatchingScore4SizeSimilarity = 0.0f;
	m_minMatchingScore4VisualSimilarity = 0.0f;
	m_minMatchingScore4ReidSimilarity = 0.0f;

	m_matchingScoreWeight4Iou = 1.0f;
	m_matchingScoreWeight4SizeSimilarity = 0.0f;
	m_matchingScoreWeight4VisualSimilarity = 0.0f;
	m_matchingScoreWeight4ReidSimilarity = 0.0f;

	m_matchingScoreSet = 0;

	m_tentativeDetectorConfidence = 0.5f;
	m_minMatchingScore4TentativeIou = 0.0f;
	m_minMatchingScore4TentativeIouInactive = 0.0f;
	m_useSimpleIou4Tentative = true;
	m_minMatchingScore4PeerAssocIou = 1.0f;

	m_thresholdMahalanobis = -1.0;

	m_maxWorldDist4PrimaryAssoc = 0.0f;

	m_maxWorldDist4QuasiactiveRescue = 2.0f;

	m_maxWorldDist4SizeBypass = 1.0f;

	m_maxWorldDist4ActiveIouRescue = 0.0f;
}

/**
 * @brief Initialize association parameters from a configuration module.
 * @param[in] configFilePath The path to a configuration file.
 * @param[in] moduleName The configuration module name.
 */
NvTrackerParamsAssoc::NvTrackerParamsAssoc(const std::string &configFilePath, const std::string moduleName)
{
	YAML::Node configRoot = getConfigRoot(configFilePath);
	this->read(configRoot[moduleName]);
	const YAML::Node &baseNode = configRoot["BaseConfig"];
	if (baseNode && baseNode["checkClassMatch"])
		m_checkClassMatch = (bool)baseNode["checkClassMatch"].as<int>();
}

/**
 * @brief Initialize association parameters from a YAML node.
 * @param[in] fn Parsed YAML structure of config to update.
 */
NvTrackerParamsAssoc::NvTrackerParamsAssoc(const YAML::Node &fn)
{
	this->read(fn);
}

/**
 * @brief Update association parameters from a YAML node.
 * @param[in] fn Parsed YAML structure of config to update.
 */
void NvTrackerParamsAssoc::read(const YAML::Node &fn)
{
	if(!m_isInit)
	{
		/// Set the default values.
		*this = NvTrackerParamsAssoc();
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
		else if (paramKey == "dataAssociatorType")
		{
			/// Only support setting this parameter on startup.
			if(!m_isInit)
				m_dataAssociatorType = (DATA_ASSOCIATOR_TYPES)((int)fit->second.as<float>()) ;
			else
				CLOG(std::string("!!![WARNING] m_dataAssociatorType is not supported for dynamic updates").c_str());
		}
		else if (paramKey == "associationMatcherType")
		{
			/// Only support setting this parameter on startup.
			if(!m_isInit)
				m_associationMatcherType = (ASSOCIATION_MATCHER_TYPE)((int)fit->second.as<float>());
			else
				CLOG(std::string("!!![WARNING] m_associationMatcherType is not supported for dynamic updates").c_str());
		}
		else if (paramKey == "checkClassMatch")
		{
			CLOG(
				std::string("!!![DEPRECATED] DataAssociator:checkClassMatch is ignored. Use BaseConfig:checkClassMatch instead.").c_str()
			);
		}
		else if (paramKey == "usePrediction4Assoc")
		{
			m_usePrediction4Assoc = (int)fit->second.as<float>();
		}
		else if (paramKey == "minMatchingScore4Overall")
		{
			m_minMatchingScore4Overall = fit->second.as<float>();
		}
		else if (paramKey == "minMatchingScore4Iou")
		{
			m_minMatchingScore4Iou = fit->second.as<float>();
		}
		else if (paramKey == "minMatchingScore4SizeSimilarity")
		{
			m_minMatchingScore4SizeSimilarity = fit->second.as<float>();
		}
		else if (paramKey == "minMatchingScore4VisualSimilarity")
		{
			m_minMatchingScore4VisualSimilarity = fit->second.as<float>();
			m_matchingScoreSet |= NvTrackerParamsAssoc::MATCHING_SCORE_SET_STATUS::MIN_VISUAL;
		}
		else if (paramKey == "minMatchingScore4ReidSimilarity")
		{
			m_minMatchingScore4ReidSimilarity = fit->second.as<float>();
			m_matchingScoreSet |= NvTrackerParamsAssoc::MATCHING_SCORE_SET_STATUS::MIN_REID;
		}
		else if (paramKey == "matchingScoreWeight4Iou")
		{
			m_matchingScoreWeight4Iou = fit->second.as<float>();
		}
		else if (paramKey == "matchingScoreWeight4SizeSimilarity")
		{
			m_matchingScoreWeight4SizeSimilarity = fit->second.as<float>();
		}
		else if (paramKey == "matchingScoreWeight4VisualSimilarity")
		{
			m_matchingScoreWeight4VisualSimilarity = fit->second.as<float>();
			m_matchingScoreSet |= NvTrackerParamsAssoc::MATCHING_SCORE_SET_STATUS::WEIGHT_VISUAL;
		}
		else if (paramKey == "matchingScoreWeight4ReidSimilarity")
		{
			m_matchingScoreWeight4ReidSimilarity = fit->second.as<float>();
			m_matchingScoreSet |= NvTrackerParamsAssoc::MATCHING_SCORE_SET_STATUS::WEIGHT_REID;
		}
		else if (paramKey == "tentativeDetectorConfidence")
		{
			m_tentativeDetectorConfidence = fit->second.as<float>();
		}
		else if (paramKey == "minMatchingScore4TentativeIou")
		{
			m_minMatchingScore4TentativeIou = fit->second.as<float>();
		}
		else if (paramKey == "minMatchingScore4TentativeIouInactive")
			m_minMatchingScore4TentativeIouInactive = std::max(0.0f, fit->second.as<float>());
		else if (paramKey == "useSimpleIou4Tentative")
		{
			m_useSimpleIou4Tentative = fit->second.as<float>();
		}
		else if (paramKey == "minMatchingScore4PeerAssocIou")
		{
			m_minMatchingScore4PeerAssocIou = fit->second.as<float>();
		}
		else if (paramKey == "thresholdMahalanobis")
		{
			m_thresholdMahalanobis = fit->second.as<float>();
		}
		else if (paramKey == "maxWorldDist4PrimaryAssoc")
		{
			m_maxWorldDist4PrimaryAssoc = fit->second.as<float>();
		}
		else if (paramKey == "maxWorldDist4QuasiactiveRescue")
		{
			m_maxWorldDist4QuasiactiveRescue = fit->second.as<float>();
		}
		else if (paramKey == "maxWorldDist4SizeBypass")
		{
			m_maxWorldDist4SizeBypass = fit->second.as<float>();
		}
		else if (paramKey == "maxWorldDist4ActiveIouRescue")
			m_maxWorldDist4ActiveIouRescue = std::max(0.0f, fit->second.as<float>());
		else
		{
			cout << "!! [WARNING][NvTrackerParamsAssoc] Unknown param found: " << paramKey << endl;
		}
	}

	/// Set init flag after initialization to prevent updates from changing static parameters.
	m_isInit = true;
}
