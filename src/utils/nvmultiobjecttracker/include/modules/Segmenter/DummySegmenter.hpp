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

#ifndef SEGMENTERLIB_DUMMYSEGMENTER_HPP_
#define SEGMENTERLIB_DUMMYSEGMENTER_HPP_

#include "modules/Segmenter/ISegmenter.hpp"
#include "modules/NvMultiObjectTracker/NvTrackerParams.hpp"
#include "modules/NvMultiObjectTracker/NvTrackedObject.hpp"

/** @brief Stores configuration for a segmenter implementation. */
class DummySegmenterConfigParams : public NvTrackerParams
{
public:
	/** @brief Construct a DummySegmenterConfigParams instance. */
	DummySegmenterConfigParams();

	/**
	 * @brief Construct a DummySegmenterConfigParams instance.
	 * @param[in] configFilePath Path to the tracker configuration file.
	 * @param[in] moduleName Configuration module name.
	 */
	DummySegmenterConfigParams(const std::string &configFilePath, const std::string moduleName);

	/**
	 * @brief Construct a DummySegmenterConfigParams instance.
	 * @param[in] fn Parsed YAML node.
	 */
	DummySegmenterConfigParams(const YAML::Node &fn);

	/**
	 * @brief Load configuration values from a YAML node.
	 * @param[in] fn Parsed YAML node.
	 */
	void read(const YAML::Node &fn) override;

	int m_debugVerboseLevel; //!< Debug message verbose mode level.
	SEGMENTER_TYPES m_segmenterType; //!< The type of Segmenter.
};

/** @brief Provides a no-op target segmenter implementation. */
class DummySegmenter : public ISegmenter
{
public:
	/**
	 * @brief Construct a DummySegmenter instance from configuration.
	 * @param[in] configFilePath Path to the tracker configuration file.
	 */
	DummySegmenter(const std::string &configFilePath);

	/** @brief Release resources owned by the DummySegmenter instance. */
	virtual ~DummySegmenter();

	/**
	 * @brief Initialize variables and allocate memories.
	 * @param[in] frameInfo Metadata about the source image, such as size and color.
	 * @param[in] maxTargetsPerStream Maximum number of targets to track in each stream.
	 * @param[in] targetManager Target manager of the tracked objects.
	 */
	void initialize(
		const SourceFrameInfo &frameInfo, const uint maxTargetsPerStream,
		const std::shared_ptr<NvTrackedObjectManager> &targetManager
	) override;

	/**
	 * @brief Populate the capabilities reported to the tracker.
	 * @param[out] paramsQuery The output query parameters.
	 */
	void queryParams(NvMOTQuery &paramsQuery) override;

	/**
	 * @brief Localize and segment target in a new frame.
	 * @param[in] inputMeta A batch of input frames.
	 */
	void localize(const std::map<NvMOTStreamId, NvMOTFrame *>& inputMeta) override;

	/**
	 * @brief Update Segmenter after data association.
	 * @param[in] inputMeta A batch of input frames.
	 */
	void update(const std::map<NvMOTStreamId, NvMOTFrame *>& inputMeta) override;

	/**
	 * @brief Output segmentation mask of a target.
	 * @param[in] target The target.
	 * @param[out] mask The output mask.
	 */
	void outputMaskObj(const std::shared_ptr<NvTrackedObject> target, NvMOTMask& mask) override;

	/**
	 * @brief Remove a stream's Segmenter features.
	 * @param[in] videoStreamId Video stream id.
	 */
	void removeStream(const NvMOTStreamId videoStreamId) override;

	/** @brief Report whether the module is initialized. */
	bool isInitialized() override { return m_isInit; }

	/** @brief Get the configured segmenter type. */
	const SEGMENTER_TYPES getSegmenterType() override { return m_params.m_segmenterType; }

protected:
	bool m_isInit; //!< Segmenter status if initialized.
	DummySegmenterConfigParams m_params; //!< Config params for Segmenter module.
};

#endif // SEGMENTERLIB_DUMMYSEGMENTER_HPP_
