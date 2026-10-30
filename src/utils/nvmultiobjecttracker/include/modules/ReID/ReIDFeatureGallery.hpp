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

#ifndef REIDLIB_REIDFEATUREGALLERY_HPP_
#define REIDLIB_REIDFEATUREGALLERY_HPP_

#include <map>
#include <deque>
#include <vector>
#include "modules/utility/cuda_common.hpp"
#include "modules/NvMultiObjectTracker/NvMultiObjectTracker_common.hpp"

/** @brief Stores and updates per-target ReID feature histories. */
class ReIDFeatureGallery
{
public:
    /** @brief Construct a ReIDFeatureGallery instance. */
    ReIDFeatureGallery();

    /** @brief Release resources owned by the ReIDFeatureGallery instance. */
    ~ReIDFeatureGallery();

    /**
     * @brief Allocate feature-gallery storage and initialize stream partitions.
     * @param[in] reidFeatureSize Number of values in each ReID feature.
     * @param[in] reidHistorySize Maximum number of ReID features per object.
     * @param[in] maxStreams Maximum number of streams.
     * @param[in] maxTargetsPerStream Maximum number of targets to track in each stream.
     * @param[in] debugVerboseLevel Debug verbosity level.
     */
    void initialize(
        const int reidFeatureSize, const int reidHistorySize, const uint32_t maxStreams,
        const int maxTargetsPerStream, const int debugVerboseLevel
    );

    /**
     * @brief For a new reid feature to add, determine its position in feature gallery and return the global index of that position.
     * @param[in] videoStreamId Video stream identifier.
     * @param[in] targetInd Index of the target that receives the feature.
     * @return Global index in the feature gallery, or -1 when the feature cannot be added.
     */
    int getReidFeatureIndGlobal(const NvMOTStreamId videoStreamId, uint targetInd);

    /**
     * @brief Get the global-gallery offset of a stream's partition.
     * @param[in] videoStreamId Video stream id.
     * @return The stream's offset into the global gallery, or -1 when the stream has no gallery partition.
     */
    int getReidFeatureOffset(const NvMOTStreamId videoStreamId) const
    {
        auto it = m_reidFeatureOffsets.find(videoStreamId);
        return (it == m_reidFeatureOffsets.end()) ? -1 : it->second;
    }

    /**
     * @brief Remove reid feature of given indices.
     * @param[in] videoStreamId Video stream identifier.
     * @param[in] reidFeatureInds Indices of the features to remove.
     */
    void remove(const NvMOTStreamId videoStreamId, const std::vector<int>& reidFeatureInds);

    /**
     * @brief Remove a stream's all reid features.
     * @param[in] videoStreamId Video stream id.
     */
    void removeStream(const NvMOTStreamId videoStreamId);

    /**
     * @brief Update reid feature gallery and each target's record.
     * @param[in] videoStreamId Video stream id.
     * @param[in] detObj Detection that provides the new feature.
     * @param[in] target The target associated with this detection.
     */
    bool updateReIDFeature(
        const NvMOTStreamId videoStreamId, const DetectedObjInfo *detObj, const std::shared_ptr<NvTrackedObject> target
    );

    /**
     * @brief Remove reid feature gallery and each target's record.
     * @param[in] videoStreamId Video stream id.
     */
    void removeStreamReIDFeature(const NvMOTStreamId videoStreamId);

    /**
     * @brief Remove a target's reid features from the gallery.
     * @param[in] videoStreamId Video stream id.
     * @param[in] target Tracker associated with the detection.
     */
    void removeTrackerReIDFeature(const NvMOTStreamId videoStreamId, const std::shared_ptr<NvTrackedObject> target);

    /**
     * @brief Merge fused targets' reid features into its peer.
     * @param[in] videoStreamId Video stream identifier.
     * @param[in] targetIndKept Index of the target that keeps the merged features.
     * @param[in] targetIndFused Index of the target whose features are merged into the kept target.
     */
    void fuseReidFeatureInd(const NvMOTStreamId videoStreamId, uint targetIndKept, uint targetIndFused);

    /**
     * @brief Get the last feature gallery index in use for a stream.
     * @param[in] streamId Video stream identifier.
     * @return The last index in use, or -1 when the stream has no gallery partition.
     */
    const int getLastInd(const NvMOTStreamId streamId) const;

    /**
     * @brief Get the device pointer to a stream's feature gallery partition.
     * @param[in] streamId Video stream identifier.
     * @return The device pointer to the stream's features, or nullptr when the stream has no gallery partition.
     */
    const float* getReIDFeaturePtrs(const NvMOTStreamId streamId) const;

    /** @brief Get the start of the complete feature gallery. */
    float* getFirstReIDFeaturePtrs() { return m_reidFeatureGallery_d; }

    /** @brief Get the configured per-target feature history size. */
    const int getReIDHistorySize() const { return m_reidHistorySize; }

    /**
     * @brief Get the gallery feature indices retained for a target.
     * @param[in] targetInd Target index.
     */
    const std::vector<int>& getTargetReIDFeatureInds(uint targetInd) const \
    {
        return m_reidFeatureIndsTargets.at(targetInd);
    }

    /**
     * @brief Get the next circular-history slot for a target.
     * @param[in] targetInd Target index.
     */
    const int getTargetReIDFeatureNextInd(uint targetInd) const \
    {
        return m_reidFeatureNextIndTargets.at(targetInd);
    }

    /**
     * @brief Reset the feature indices retained for a target.
     * @param[in] targetInd Target index.
     */
    void resetTargetReIDFeatureInds(uint targetInd)
    {
        m_reidFeatureIndsTargets.at(targetInd).clear();
        m_reidFeatureNextIndTargets[targetInd] = 0;
    }

    /**
     * @brief Record a feature index for a target.
     * @param[in] targetInd Target index.
     * @param[in] featureInd Gallery feature index to retain for the target.
     */
    void updateTargetReIDFeatureInd(uint targetInd, int featureInd)
    {
        if ( int(m_reidFeatureIndsTargets.at(targetInd).size()) < m_reidHistorySize )
            m_reidFeatureIndsTargets.at(targetInd).push_back(featureInd);
        else
            m_reidFeatureIndsTargets.at(targetInd)[m_reidFeatureNextIndTargets.at(targetInd)] = featureInd;

        m_reidFeatureNextIndTargets[targetInd] = (m_reidFeatureNextIndTargets[targetInd] + 1) % m_reidHistorySize;
    }

    /**
     * @brief Print the feature indices retained for a target.
     * @param[in] targetInd Target index.
     */
    void printFeatureInds(uint targetInd) const
    {
        for (auto featureInd : m_reidFeatureIndsTargets.at(targetInd))
        {
            std::cout << featureInd << ", ";
        }
    }

    #ifdef VIZ_DEBUG
        /**
         * @brief Write the feature gallery to a file.
         * @param[in] filename Output file name.
         */
        void dumpReIDFeatureGallery(std::string filename);
    #endif

protected:
    int m_debugVerboseLevel; //!< Debug message verbose mode level.
    int m_maxTargetsPerStream; //!< Max targets per stream.
    int m_reidHistorySize; //!< Max number of reid features per object.
    int m_reidFeatureSize; //!< Size of each reid feature.
    std::map<NvMOTStreamId, int> m_reidFeatureOffsets; //!< Reid feature gallery indices being used by each stream, {streamID, ptr}.
    std::deque<int> m_reidFeatureOffsetsInPool; //!< Reid feature pointers in pool when unused.
    std::map<NvMOTStreamId, std::set<int>> m_reidFeatureIndsInPool; //!< Feature gallery indices in pool.
    std::map<NvMOTStreamId, int> m_lastInds; //!< Index of last feature gallery being used in each stream.
    std::map<uint, std::vector<int>> m_reidFeatureIndsTargets; //!< Each target's reid feature indices.
    std::map<uint, uint> m_reidFeatureNextIndTargets; //!< Each target's next circular-history slot for a reid feature.
    float* m_reidFeatureGallery_d, *m_reidFeatureGallery_h; //!< CUDA array saving target features in past frames.
    cudaStream_t m_cuStream; //!< CUDA stream.
    bool m_isInit; //!< If initialized.
};

#endif // REIDLIB_REIDFEATUREGALLERY_HPP_
