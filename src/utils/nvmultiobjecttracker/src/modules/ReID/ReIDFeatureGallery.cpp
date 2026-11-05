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

#include "cuda_runtime.h"
#ifdef VIZ_DEBUG
#include <fstream>
#endif

#include "modules/ReID/ReIDFeatureGallery.hpp"

using std::cout;
using std::endl;

/**
 * @brief Reserve a global feature-gallery index for a target.
 * @param[in] videoStreamId Video stream identifier.
 * @param[in] targetInd Index of the target that receives the feature.
 */
int ReIDFeatureGallery::getReidFeatureIndGlobal(const NvMOTStreamId videoStreamId, uint targetInd)
{
    const std::vector<int>& reidFeatureInds = m_reidFeatureIndsTargets.at(targetInd);

    int reidFeatureInd;

    /// Reuse the oldest feature slot once the number of features reaches the size limit.
    if ( int(reidFeatureInds.size()) == m_reidHistorySize )
    {
        if (m_debugVerboseLevel > 3)
        {
            cout << "[NvTrackedObjectManager::updateReIDFeature] target " << targetInd <<
                ", reid feature exceeds history: "
                << int(reidFeatureInds.size()) << endl;
        }
        reidFeatureInd = reidFeatureInds.at(getTargetReIDFeatureNextInd(targetInd));
    }
    else if ( int(reidFeatureInds.size()) > m_reidHistorySize )
    {
        throw std::runtime_error("reid feature number exceeds feature gallery size");
    }
    else
    {
        /// If target is in a new stream, create a feature gallery pointer for the new stream.
        if(m_reidFeatureOffsets.find(videoStreamId) == m_reidFeatureOffsets.end())
        {
            /// Check if there is a spare feature gallery pointer in the pool to assign.
            if(m_reidFeatureOffsetsInPool.size() == 0)
            {
                return -1;
            }
            m_reidFeatureOffsets[videoStreamId] = m_reidFeatureOffsetsInPool.front();
            m_reidFeatureOffsetsInPool.pop_front();

            /// Initialize the feature index pool.
            for (int fid = 0; fid < m_reidHistorySize * m_maxTargetsPerStream; fid++)
            {
                m_reidFeatureIndsInPool[videoStreamId].insert(fid);
            }

            /// Set last feature ind as -1.
            m_lastInds[videoStreamId] = -1;
        }

        /// Add new feature to gallery.
        /// Find spare feature ind in pool.
        std::set<int>& reidFeatureIndsInPool = m_reidFeatureIndsInPool[videoStreamId];

        /// Check if pool is full and if pointer is valid.
        if (reidFeatureIndsInPool.size() == 0)
        {
            return -1;
        }

        /// Take the first spare feature index from the pool.
        reidFeatureInd = *(reidFeatureIndsInPool.begin());

        /// Update pool.
        reidFeatureIndsInPool.erase(reidFeatureIndsInPool.begin());

        if (m_debugVerboseLevel > 2)
        {
            cout << "[ReIDFeatureGallery::update] lastInd: " << m_lastInds.at(videoStreamId) <<
                " assigned reidFeatureInd: " << reidFeatureInd << endl;
        }

        /// Update last ind.
        if(reidFeatureInd > m_lastInds.at(videoStreamId))
        {
            m_lastInds[videoStreamId] = reidFeatureInd;
        }
    }

    /// Add featureInd to target.
    updateTargetReIDFeatureInd(targetInd, reidFeatureInd);
    if (m_debugVerboseLevel > 2 )
    {
        cout << "[NvTrackedObjectManager::updateReIDFeature] Added featured to target " <<
            targetInd <<
            ". Last index: " << getLastInd(videoStreamId) << endl;
    }

    /// Global index = index in current stream + stream offset.
    return reidFeatureInd + m_reidFeatureOffsets.at(videoStreamId);
}

/**
 * @brief Merge feature-gallery indices from fused targets.
 * @param[in] videoStreamId Video stream identifier.
 * @param[in] targetIndKept Index of the target that keeps the merged features.
 * @param[in] targetIndFused Index of the target whose features are merged into the kept target.
 */
void ReIDFeatureGallery::fuseReidFeatureInd(const NvMOTStreamId videoStreamId, uint targetIndKept, uint targetIndFused)
{
    std::vector<int>& reidFeatureIndsFused = m_reidFeatureIndsTargets.at(targetIndFused);

    /// TODO: Do not assume feature ages in reidFeatureIndsFused are descending.
    /// If its not, there's no mem leak, but the age order in reidFeatureIndsKept is switched.

    /// If reidFeatureIndsKept.size() + reidFeatureIndsFused() > history size, pop oldest features in reidFeatureIndsKept.
    std::vector<int> reidFeatureIndsToPop;
    for (auto reidFeatureInd : reidFeatureIndsFused)
    {
        int nextInd = getTargetReIDFeatureNextInd(targetIndKept);
        if (nextInd < int(m_reidFeatureIndsTargets.at(targetIndKept).size()))
        {
            reidFeatureIndsToPop.push_back(m_reidFeatureIndsTargets.at(targetIndKept).at(nextInd));
        }
        updateTargetReIDFeatureInd(targetIndKept, reidFeatureInd);
    }

    reidFeatureIndsFused.clear();

    remove(videoStreamId, reidFeatureIndsToPop);
}

/**
 * @brief Remove selected features from the gallery.
 * @param[in] videoStreamId Video stream identifier.
 * @param[in] reidFeatureInds Indices of the features to remove.
 */
void ReIDFeatureGallery::remove(const NvMOTStreamId videoStreamId, const std::vector<int>& reidFeatureInds)
{
    if (reidFeatureInds.empty())
    {
        return;
    }

    /// Return to pool.
    int maxInd = *std::max_element(reidFeatureInds.begin(), reidFeatureInds.end());

    m_reidFeatureIndsInPool[videoStreamId].insert(reidFeatureInds.begin(), reidFeatureInds.end());

    if (m_debugVerboseLevel > 2 && !m_reidFeatureIndsInPool[videoStreamId].empty())
        cout << "[ReIDFeatureGallery::remove] insert: " <<
        reidFeatureInds[0] << " pool begin: " << *(m_reidFeatureIndsInPool[videoStreamId].begin())
        <<  " last ind: " << m_lastInds.at(videoStreamId) << endl;
    if (m_debugVerboseLevel > 2 && !reidFeatureInds.empty())
    {
        cout << "Returned feature inds:";
        for (auto it = reidFeatureInds.begin(); it != reidFeatureInds.end(); it++)
            cout << *it << " ";
        cout << endl;
    }

    if(m_lastInds.at(videoStreamId) == (int)maxInd)
    {
        /// Find the new last feat index.
        int tInd = (int)maxInd-1;
        for(; tInd >= 0; tInd--)
        {
            if(m_reidFeatureIndsInPool[videoStreamId].find(tInd) ==
                m_reidFeatureIndsInPool[videoStreamId].end())
                break;
        }
        m_lastInds[videoStreamId] = tInd;
    }
}

/**
 * @brief Remove all state associated with a video stream.
 * @param[in] videoStreamId Video stream id.
 */
void ReIDFeatureGallery::removeStream(const NvMOTStreamId videoStreamId)
{
    if(m_reidFeatureOffsets.find(videoStreamId) != m_reidFeatureOffsets.end() &&
        m_reidFeatureIndsInPool.find(videoStreamId) != m_reidFeatureIndsInPool.end() &&
        m_lastInds.find(videoStreamId) != m_lastInds.end())
    {
        m_reidFeatureOffsetsInPool.push_back(m_reidFeatureOffsets[videoStreamId]);
        m_reidFeatureOffsets.erase(videoStreamId);

        /// Will init when a new stream received.
        m_reidFeatureIndsInPool.erase(videoStreamId);

        m_lastInds.erase(videoStreamId);
    }
}

/**
 * @brief Get the last feature gallery index in use for a stream.
 * @param[in] streamId Video stream identifier.
 * @return The last index in use, or -1 when the stream has no gallery partition.
 */
const int ReIDFeatureGallery::getLastInd(const NvMOTStreamId streamId) const
{
    if (m_lastInds.find(streamId) != m_lastInds.end())
        return m_lastInds.at(streamId);
    else
        return -1;
}

/**
 * @brief Get the device pointer to a stream's feature gallery partition.
 * @param[in] streamId Video stream identifier.
 * @return The device pointer to the stream's features, or nullptr when the stream has no gallery partition.
 */
const float* ReIDFeatureGallery::getReIDFeaturePtrs(const NvMOTStreamId streamId) const
{
    if (m_reidFeatureOffsets.find(streamId) != m_reidFeatureOffsets.end())
        return m_reidFeatureGallery_d + m_reidFeatureOffsets.at(streamId) * m_reidFeatureSize;
    else
        return nullptr;
}

#ifdef VIZ_DEBUG
/**
 * @brief Write the feature gallery to a file.
 * @param[in] filename Output file name.
 */
void ReIDFeatureGallery::dumpReIDFeatureGallery(std::string filename)
{
    std::ofstream outFile;
    outFile.open(filename, std::ofstream::out);

    if (m_lastInds.find(0) != m_lastInds.end())
    {
        int num_vectors = m_lastInds.at(0);
        CUDA_CHECK(
            cudaMemcpyAsync(
                m_reidFeatureGallery_h, m_reidFeatureGallery_d,
                num_vectors * m_reidFeatureSize * sizeof(float), cudaMemcpyDeviceToHost, m_cuStream
            )
        );
        CUDA_CHECK(cudaStreamSynchronize(m_cuStream));
        CUDA_CHECK(cudaPeekAtLastError());
        float *ptr = m_reidFeatureGallery_h;

        outFile << "Printing all ReID feature vectors in gallery" << endl;
        for (int cnt = 0; cnt < num_vectors; cnt++)
        {
            outFile << "Vector No. = " << cnt << endl;
            for (int i = 0; i < m_reidFeatureSize; i++)
            {
                outFile << *ptr++ << "\t";
            }
            outFile << endl;
        }
    }
    outFile.close();
}
#endif

/**
 * @brief Allocate feature-gallery storage and initialize stream partitions.
 * @param[in] reidFeatureSize Number of values in each ReID feature.
 * @param[in] reidHistorySize Maximum number of ReID features per object.
 * @param[in] maxStreams Maximum number of streams.
 * @param[in] maxTargetsPerStream Maximum number of targets to track in each stream.
 * @param[in] debugVerboseLevel Debug verbosity level.
 */
void ReIDFeatureGallery::initialize(
    const int reidFeatureSize, const int reidHistorySize, const uint32_t maxStreams,
    const int maxTargetsPerStream, const int debugVerboseLevel
)
{
    if (reidFeatureSize <= 0 || reidHistorySize <= 0 || maxTargetsPerStream <= 0 || maxStreams <= 0)
    {
        throw std::runtime_error("Invalid ReID feature gallery configuration");
    }

    m_reidFeatureSize = reidFeatureSize;
    m_reidHistorySize = reidHistorySize;
    m_maxTargetsPerStream = maxTargetsPerStream;
    m_debugVerboseLevel = debugVerboseLevel;

    /// Malloc feature gallery CUDA array.
    int reidFeatureGallerySizePerStream = m_reidHistorySize * maxTargetsPerStream;
    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(&m_reidFeatureGallery_d),
            reidFeatureGallerySizePerStream * m_reidFeatureSize * maxStreams * sizeof(float)
        )
    );
    #ifdef VIZ_DEBUG
        CUDA_CHECK(
            cudaMallocHost(
                reinterpret_cast<void **>(&m_reidFeatureGallery_h),
                reidFeatureGallerySizePerStream * m_reidFeatureSize * maxStreams * sizeof(float)
            )
        );
    #endif
    CUDA_CHECK( cudaStreamCreateWithFlags(&m_cuStream, cudaStreamNonBlocking) );

    /// For each stream put its starting feature gallery pointer into ptrs pool.
    for (uint32_t streamInd = 0; streamInd < maxStreams; streamInd++)
    {
        m_reidFeatureOffsetsInPool.push_back(streamInd * reidFeatureGallerySizePerStream);
    }

    /// Initialize reid feature indices for each target.
    for (uint targetInd=0; targetInd < maxTargetsPerStream * maxStreams; targetInd++)
    {
        m_reidFeatureIndsTargets[targetInd] = std::vector<int>();
        m_reidFeatureNextIndTargets[targetInd] = 0;
    }

    m_isInit = true;
}

/** @brief Construct a ReIDFeatureGallery instance. */
ReIDFeatureGallery::ReIDFeatureGallery()
{
    m_reidFeatureSize = 0;
    m_reidHistorySize = 0;
    m_maxTargetsPerStream = 0;
    m_debugVerboseLevel = 0;
    m_reidFeatureGallery_d = nullptr;
    m_reidFeatureGallery_h = nullptr;
    m_isInit = false;
}

/** @brief Release resources owned by the ReIDFeatureGallery instance. */
ReIDFeatureGallery::~ReIDFeatureGallery()
{
    if (m_isInit)
    {
        CUDA_CHECK( cudaFree(m_reidFeatureGallery_d) );
        #ifdef VIZ_DEBUG
            CUDA_CHECK(cudaFreeHost(m_reidFeatureGallery_h));
        #endif
        CUDA_CHECK( cudaStreamDestroy(m_cuStream) );
    }
}
