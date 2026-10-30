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

#ifndef MULTIVIEWASSOCIATORLIB_MULTIVIEWASSOCIATOR_FACTORY_HPP_
#define MULTIVIEWASSOCIATORLIB_MULTIVIEWASSOCIATOR_FACTORY_HPP_

// Multi-view associator implementations.
#include "modules/MultiViewAssociator/DummyMultiViewAssociator.hpp"
#ifndef _WIN32
#include "modules/MultiViewAssociator/MultiViewAssociator.hpp"
#endif

#ifdef _WIN32
// IOU tracking does not implement the multi-view backends.  Keep the parameter
// surface used by shared tracker classes while rejecting non-dummy use at the
// factory boundary below.
class MultiViewAssociatorParams : public NvTrackerParams
{
public:
    MultiViewAssociatorParams() { reset(); }
    MultiViewAssociatorParams(const std::string &, const std::string &) { reset(); }
    MultiViewAssociatorParams(const YAML::Node &) { reset(); }
    void read(const YAML::Node &) override { reset(); }
    MULTI_VIEW_ASSOCIATOR_TYPES m_multiViewAssociatorType;
    int m_debugVerboseLevel;
    bool m_enableLatePeerReAssoc;
    bool m_enableIDCorrection;
    bool m_enableSeeThrough;
    bool m_enableMsgSync;
    bool m_checkClassMatch;
    int m_maxPeerTrackletSize;
    int m_recentlyActiveAge;
    int m_minCommonFrames4MatchScore;
    int m_maxTrackletMatchingTimeSearchRange;
    float m_maxPeerToPredDistance4Fusion;
    float m_minPeerVisibility4Fusion;
    float m_minPeerTrackletMatchScore;
    int m_maxPeerFrameDiff4NoDet;
    int m_communicatorInitSleepTime;
    float m_minReidServiceMatchConfidence;
private:
    void reset()
    {
        m_multiViewAssociatorType = DUMMY_MULTI_VIEW_ASSOCIATOR;
        m_debugVerboseLevel = 0;
        m_enableLatePeerReAssoc = false;
        m_enableIDCorrection = false;
        m_enableSeeThrough = false;
        m_enableMsgSync = false;
        m_checkClassMatch = true;
        m_maxPeerTrackletSize = 0;
        m_recentlyActiveAge = 0;
        m_minCommonFrames4MatchScore = 0;
        m_maxTrackletMatchingTimeSearchRange = 0;
        m_maxPeerToPredDistance4Fusion = 0.0f;
        m_minPeerVisibility4Fusion = 0.0f;
        m_minPeerTrackletMatchScore = 0.0f;
        m_maxPeerFrameDiff4NoDet = 0;
        m_communicatorInitSleepTime = 0;
        m_minReidServiceMatchConfidence = 0.0f;
    }
};
#endif

/** @brief Store multi-view-associator factory parameters. */
class MultiViewAssociatorFactoryParams : public NvTrackerParams
{
public:
	/** @brief Initialize factory parameters with the dummy associator type. */
	MultiViewAssociatorFactoryParams()
	{
		m_multiViewAssociatorType = DUMMY_MULTI_VIEW_ASSOCIATOR;
	}

	/**
	 * @brief Initialize factory parameters from a configuration module.
	 * @param[in] configFilePath The path to a configuration file.
	 * @param[in] moduleName The configuration module name.
	 */
	MultiViewAssociatorFactoryParams(const std::string &configFilePath, const std::string moduleName)
	{
		YAML::Node configRoot = getConfigRoot(configFilePath);
		this->read(configRoot[moduleName]);
	}

	/**
	 * @brief Initialize factory parameters from a YAML node.
	 * @param[in] fn Parsed YAML structure of config to update.
	 */
	MultiViewAssociatorFactoryParams(const YAML::Node &fn)
	{
		read(fn);
	}

	/**
	 * @brief Update factory parameters from a YAML node.
	 * @param[in] fn Parsed YAML structure of config to update.
	 */
	void read(const YAML::Node &fn) override
	{
		*this = MultiViewAssociatorFactoryParams();
		if(!fn.size()) { return;}

		for (YAML::const_iterator fit = fn.begin(); fit != fn.end(); ++fit)
		{
			std::string paramKey = fit->first.as<std::string>();
			if (paramKey == "multiViewAssociatorType")
			{
				m_multiViewAssociatorType = (MULTI_VIEW_ASSOCIATOR_TYPES)((int)fit->second.as<float>());
			}
		}
	}

	MULTI_VIEW_ASSOCIATOR_TYPES m_multiViewAssociatorType; //!< Selects the multi-view-associator implementation.
};

/** @brief Provide a singleton factory for multi-view associator objects. */
class MultiViewAssociatorFactory
{
public:
    /** @brief Return the static instance reference to the factory. */
    static MultiViewAssociatorFactory& getInstance()
    {
        static MultiViewAssociatorFactory instance;
        return instance;
    }

    /**
     * @brief Create the configured multi-view-associator implementation.
     * @param[in] configFilePath The path to a configuration file.
     */
    std::shared_ptr<IMultiViewAssociator> makeMultiViewAssociator(const std::string configFilePath)
	{

#ifdef _WIN32
        m_params = MultiViewAssociatorFactoryParams(configFilePath, "MultiViewAssociator");
        if (m_params.m_multiViewAssociatorType != DUMMY_MULTI_VIEW_ASSOCIATOR) {
            throw std::runtime_error("Multi-view association is not available in the Windows IOU-only tracker build");
        }
        return std::make_shared<DummyMultiViewAssociator>(configFilePath);
#else
		m_params = MultiViewAssociatorFactoryParams(configFilePath, "MultiViewAssociator");
		switch (m_params.m_multiViewAssociatorType)
		{
			case DUMMY_MULTI_VIEW_ASSOCIATOR:
			{
				return std::make_shared<DummyMultiViewAssociator>(configFilePath);
			}
			case MULTI_VIEW_ASSOCIATOR:
			{
				return std::make_shared<MultiViewAssociator>(configFilePath);
			}
			default:
			{
				std::cerr << "WARNING: Invalid MultiViewAssociator type: "
						  << m_params.m_multiViewAssociatorType << std::endl;
				return std::make_shared<DummyMultiViewAssociator>(configFilePath);
			}
		}
#endif
	}

private:
    MultiViewAssociatorFactoryParams m_params; //!< Stores multi-view-associator factory parameters.
};

#endif // MULTIVIEWASSOCIATORLIB_MULTIVIEWASSOCIATOR_FACTORY_HPP_
