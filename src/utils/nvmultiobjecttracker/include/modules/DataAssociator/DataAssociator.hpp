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

#ifndef NVMULTIOBJECTTRACKERLIB_DATAASSOCIATOR_HPP_
#define NVMULTIOBJECTTRACKERLIB_DATAASSOCIATOR_HPP_

#include <atomic>
#include <mutex>
#include <deque>
#include <queue>
#include <map>
#include <unordered_map>
#include <limits>
#include "cublas_v2.h"
#include "modules/NvMultiObjectTracker/NvTrackerParams.hpp"
#include "modules/DataAssociator/IDataAssociator.hpp"

enum ASSOCIATION_MATCHER_TYPE : int
{
	GREEDY=0, //!< Matches pairs in a single greedy pass.
	CASCADED //!< Matches pairs through cascaded stages.
};

/** @brief Config params for the custom distance function for the data association, including the min criteria and weights for each score factor. */
class NvTrackerParamsAssoc : public NvTrackerParams
{
public:
	/** @brief Initialize association parameters with default values. */
	NvTrackerParamsAssoc();

	/**
	 * @brief Initialize association parameters from a configuration module.
	 * @param[in] configFilePath The path to a configuration file.
	 * @param[in] moduleName The configuration module name.
	 */
	NvTrackerParamsAssoc(const std::string &configFilePath, const std::string moduleName);

	/**
	 * @brief Initialize association parameters from a YAML node.
	 * @param[in] fn Parsed YAML structure of config to update.
	 */
	NvTrackerParamsAssoc(const YAML::Node &fn);

	/**
	 * @brief Update association parameters from a YAML node.
	 * @param[in] fn Parsed YAML structure of config to update.
	 */
	void read(const YAML::Node &fn) override;

	int m_debugVerboseLevel; //!< Controls debug message verbosity.
	ASSOCIATION_MATCHER_TYPE m_associationMatcherType; //!< Selects the GREEDY or CASCADED matcher.
	bool m_checkClassMatch; //!< Indicates whether association requires matching object classes.
	bool m_usePrediction4Assoc; //!< Indicates whether association uses predicted states instead of previous states.

	// [Minimum matching scores]
	float m_minMatchingScore4Overall; //!< Specifies the minimum total matching score for a candidate.
	float m_minMatchingScore4Iou; //!< Specifies the minimum IOU matching score for a candidate.
	float m_minMatchingScore4SizeSimilarity; //!< Specifies the minimum bounding-box size-similarity score for a candidate.
	float m_minMatchingScore4VisualSimilarity; //!< Specifies the minimum visual-similarity score for a candidate.
	float m_minMatchingScore4ReidSimilarity; //!< Specifies the minimum ReID-similarity score for a candidate.

	// [Matching score weights]
	float m_matchingScoreWeight4Iou; //!< Specifies the weight of the IOU score.
	float m_matchingScoreWeight4SizeSimilarity; //!< Specifies the weight of the bounding-box size-similarity score.
	float m_matchingScoreWeight4VisualSimilarity; //!< Specifies the weight of visual similarity measured by the correlation-response ratio.
	float m_matchingScoreWeight4ReidSimilarity; //!< Specifies the weight of ReID similarity.

	// [Optional matching-score configuration]
	uint m_matchingScoreSet; //!< Records which optional matching scores were configured.

	// [Cascaded matching]
	float m_tentativeDetectorConfidence; //!< Specifies the confidence ceiling for tentative detections above the minimum detector confidence.
	float m_minMatchingScore4TentativeIou; //!< Specifies the minimum IOU score for matching tentative objects.
	float m_minMatchingScore4TentativeIouInactive; //!< Lower IoU floor for INACTIVE Stage 3 pairs; 0 disables it.
	bool m_useSimpleIou4Tentative; //!< Indicates whether tentative targets use simple IOU instead of the valid-object similarity calculation.
	float m_minMatchingScore4PeerAssocIou; //!< Specifies the minimum IOU score for matching peer targets.

	// [Mahalanobis-distance association]
	double m_thresholdMahalanobis; //!< Specifies the Mahalanobis-distance threshold.

	// [World-space distance gates]
	float m_maxWorldDist4PrimaryAssoc; //!< Specifies the maximum world-space feet distance in meters for primary association; a nonpositive value disables the limit.
	float m_maxWorldDist4QuasiactiveRescue; //!< Specifies the maximum world-space distance in meters for QUASIACTIVE IOU rescue.
	float m_maxWorldDist4SizeBypass; //!< Specifies the maximum world-space distance in meters for bypassing the size gate.
	float m_maxWorldDist4ActiveIouRescue; //!< Specifies the maximum world-space distance for ACTIVE or INACTIVE Stage 3 IOU rescue; zero disables it.

	// [Associator selection]
	DATA_ASSOCIATOR_TYPES m_dataAssociatorType; //!< Selects the data-associator implementation to construct.

	typedef enum {
		MIN_VISUAL = 1 << 0, //!< Indicates that minMatchingScore4VisualSimilarity was configured.
		WEIGHT_VISUAL = 1 << 1, //!< Indicates that matchingScoreWeight4VisualSimilarity was configured.
		MIN_REID = 1 << 2, //!< Indicates that minMatchingScore4ReidSimilarity was configured.
		WEIGHT_REID = 1 << 3 //!< Indicates that matchingScoreWeight4ReidSimilarity was configured.
	} MATCHING_SCORE_SET_STATUS;

	bool m_isInit = false; //!< Indicates whether the parameters have been initialized.
	//!< This prevents static parameters from changing during dynamic updates.
};

/** @brief Similarity params. */
struct SimilarityParam
{
	typedef enum {
		IOU, //!< Intersection over union score.
		SIZE_SIMILARITY, //!< Bbox size similarity score.
		VISUAL_SIMILARITY, //!< Visual similarity score.
		REID_SIMILARITY, //!< Reid similarity score.
		WORLD_PROXIMITY //!< World-space proximity score used in Stage Second.
	} TYPES;

	TYPES type; //!< Similarity type.
	float minVal; //!< Specifies the minimum matching score, below which a pair is not matched.
	float weight; //!< Specifies this score's contribution to the total score.
};

/** @brief Data association module. */
class DataAssociator : public IDataAssociator
{
public:

	/**
	 * @brief Construct a data associator from a configuration file.
	 * @param[in] configFilePath The path to a config file.
	 */
	DataAssociator(const std::string &configFilePath);

	/** @brief Report accumulated association statistics during destruction. */
	~DataAssociator();

	/**
	 * @brief Initialize the data associator for a source stream.
	 * @param[in] frameInfo Metadata about the source image (e.g., size, color).
	 * @param[in] maxTargetsPerStream The max number of targets to track per stream.
	 * @param[in] useVisualTracker Whether visual-tracker similarity is used.
	 * @param[in] useReid Whether ReID similarity is used.
	 */
	void initialize(
		const SourceFrameInfo &frameInfo, const uint32_t maxTargetsPerStream,
		const bool useVisualTracker, const bool useReid
	) override;

	/**
	 * @brief Associate detections with existing targets for one stream.
	 * @param[in] in Newly detected object bounding boxes and existing targets.
	 * @return (1) matched pairs, (2) unmatched detector bboxes, and (3) unmatched targets.
	 */
	assocDataOut_t associatePerStream(const assocDataIn_t in) override;

	/** @brief Return whether the data associator has been initialized. */
	bool isInitialized() override { return m_isInit; }

	/** @brief Return the configured data-associator type. */
	const DATA_ASSOCIATOR_TYPES getDataAssociatorType() override { return m_params.m_dataAssociatorType; }

	/**
	 * @brief Update dynamic parameters during runtime.
	 * @param[in] fn Parsed YAML structure of config to update.
	 */
	void updateParams(const YAML::Node &fn)
	{
		m_params.read(fn);
		if (m_isInit) initializeSimilarityParams(m_useVisualTracker, m_useReid);
	}

protected:

	bool m_isInit; //!< Indicates whether the data associator has been initialized.
	NvTrackerParamsAssoc m_params; //!< Stores parsed configuration parameters.
	std::vector<std::vector<SimilarityParam>> m_similarityParamLists; //!< Per-stage similarity parameters and weights.
	mutable std::mutex m_similarityParamsMutex; //!< Guards similarity parameters during reads and updates.
	int m_maxTargetsPerStream; //!< Specifies the maximum number of targets to track per stream.
	bool m_useVisualTracker{false}; //!< Records whether visual-tracker similarity is enabled for parameter rebuilds.
	bool m_useReid{false}; //!< Records whether ReID similarity is enabled for parameter rebuilds.

	/**
	 * @brief Rebuild per-stage similarity parameters for the enabled features.
	 * @param[in] useVisualTracker Whether visual-tracker similarity is used.
	 * @param[in] useReid Whether ReID similarity is used.
	 */
	void initializeSimilarityParams(const bool useVisualTracker, const bool useReid);

	/**
	 * @brief Find the best matches between detections and existing targets.
	 * @param[in] in Association inputs.
	 * @param[in] stage Association stage.
	 * @return The association results.
	 */
	assocDataOut_t associatePerStream_greedy(const assocDataIn_t& in, const uint stage = 0);

	/**
	 * @brief Run the legacy two-stage association algorithm.
	 * @param[in] in Association inputs.
	 * @return The association results.
	 */
	assocDataOut_t associatePerStream_cascadedLegacy(const assocDataIn_t& in);

	/**
	 * @brief Run the cascaded association algorithm.
	 * @param[in] in Association inputs.
	 * @return The association results.
	 */
	assocDataOut_t associatePerStream_cascaded(const assocDataIn_t& in);

	/**
	 * @brief Function to compute the association score between a detector bbox and a tracked bbox.
	 * @param[in] trajObj Tracked object.
	 * @param[in] detObj Detected object.
	 * @param[in] similarityParamList Similarity parameters and their weights.
	 * @param[in] sumCorrRespAtDetectedPt Sum of tracker correlation responses at the detected point.
	 * @return The similarity score.
	 */
	virtual float assocDistanceFunc(
		const std::shared_ptr<NvTrackedObject> trajObj, const DetectedObjInfo &detObj,
		const std::vector<SimilarityParam> &similarityParamList, const float sumCorrRespAtDetectedPt
	) const;

	/**
	 * @brief Compute the association score matrix.
	 * @param[in] in Association inputs.
	 * @param[in] stage Association stage.
	 * @return The computed score matrix, where higher values indicate closer matches.
	 */
	std::vector<std::vector<float>> computeDistanceMatrix(const assocDataIn_t &in, const uint stage = 0) const;

	/**
	 * @brief Given the distance matrix, get the sorted list of potential match pairs based on the association cost.
	 * @param[in] in Association inputs.
	 * @param[in] distMat Score matrix, where higher values indicate closer matches.
	 * @return The sorted list of potential match pairs { matching score, {inDetBboxListInd, inTargetListInd}}.
	 */
	std::multimap<float, std::pair<uint, int>> sortMatching(
		const assocDataIn_t &in, const std::vector<std::vector<float>> &distMat
	) const;

	/**
	 * @brief Dump-print the data association results.
	 * @param[in] out Association results to print.
	 */
	void dumpDataAssociationResults(const assocDataOut_t &out) const;

	/**
	 * @brief Dump-print the scores in the distance matrix for association.
	 * @param[in] in Association inputs used to label the matrix.
	 * @param[in] distMat Score matrix to print.
	 */
	void dumpAssocDistanceMatrix(const assocDataIn_t &in, std::vector<std::vector<float>> distMat) const;

	/** @brief Store thread-safe per-session counters for the association-scoring funnel. */
	struct DA_Stats {
		std::atomic<uint64_t> total{0}; //!< Counts all evaluated target-detection pairs.
		std::atomic<uint64_t> gate_hard{0}; //!< Counts pairs rejected by a class, Mahalanobis, or component-score gate.
		std::atomic<uint64_t> near_lt04{0}; //!< Counts pairs below a 0.4 overall score after passing component gates.
		std::atomic<uint64_t> near_04_05{0}; //!< Counts pairs with an overall score in the range [0.4, 0.5).
		std::atomic<uint64_t> near_ge50{0}; //!< Counts pairs above 0.5 but below the minimum overall matching score.
		std::atomic<uint64_t> passed{0}; //!< Counts pairs that meet the minimum overall matching score.
		std::atomic<uint64_t> qactive_rescue_attempted{0}; //!< Counts QUASIACTIVE rescue attempts in Stage Second.
		std::atomic<uint64_t> qactive_iou_hardgate{0}; //!< Counts QUASIACTIVE pairs rejected by the IOU hard gate.
		std::atomic<uint64_t> qactive_size_rescued{0}; //!< Counts QUASIACTIVE pairs rescued from the size gate.
		std::atomic<uint64_t> stage2_eligible{0}; //!< Counts candidates eligible for Stage Second.
		std::atomic<uint64_t> stage2_matched{0}; //!< Counts candidates matched in Stage Second.
		std::atomic<uint64_t> stage2_no_dets{0}; //!< Counts Stage Second runs without detections.
		std::atomic<uint64_t> active_iou_hardgate{0}; //!< Counts ACTIVE or INACTIVE pairs rejected by the IOU hard gate.
		std::atomic<uint64_t> stage2_has_dets{0}; //!< Counts Stage Second runs with detections.
		std::atomic<uint64_t> stage2_pair_total{0}; //!< Counts target-detection pairs evaluated in Stage Second.
		std::atomic<uint64_t> stage2_wp_gate{0}; //!< Counts Stage Second pairs rejected by the world-proximity gate.
		std::atomic<uint64_t> stage2_size_gate{0}; //!< Counts Stage Second pairs rejected by the size gate.
		std::atomic<uint64_t> stage2_noworld_track{0}; //!< Counts Stage Second targets without world coordinates.
		std::atomic<uint64_t> stage2_noworld_det{0}; //!< Counts Stage Second detections without world coordinates.
		std::atomic<uint64_t> stage2_hard_world{0}; //!< Counts Stage Second pairs rejected by the world-distance hard gate.
		std::atomic<uint64_t> stage2_size_rescued{0}; //!< Counts Stage Second pairs rescued from the size gate.
	};
	mutable DA_Stats m_da_stats; //!< Stores association-scoring counters.

};

#endif // NVMULTIOBJECTTRACKERLIB_DATAASSOCIATOR_HPP_
