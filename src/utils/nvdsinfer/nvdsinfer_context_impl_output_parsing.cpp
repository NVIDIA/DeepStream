/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <cassert>

#include "nvdsinfer_context_impl.h"

#include <algorithm>

static const bool ATHR_ENABLED = true;
static const float ATHR_THRESHOLD = 60.0;

using namespace std;

#define DIVIDE_AND_ROUND_UP(a, b) ((a + b - 1) / b)

namespace nvdsinfer {

/* Parse all object bounding boxes for the class `classIndex` in the frame
 * meeting the minimum threshold criteria.
 *
 * This parser function has been specifically written for the sample resnet10
 * model provided with the SDK. Other models will require this function to be
 * modified.
 */
bool
DetectPostprocessor::parseBoundingBox(vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    vector<NvDsInferObjectDetectionInfo>& objectList)
{

    int outputCoverageLayerIndex = -1;
    int outputBBoxLayerIndex = -1;


    for (unsigned int i = 0; i < outputLayersInfo.size(); i++)
    {
        if (strstr(outputLayersInfo[i].layerName, "bbox") != nullptr)
        {
            outputBBoxLayerIndex = i;
        }
        if (strstr(outputLayersInfo[i].layerName, "cov") != nullptr)
        {
            outputCoverageLayerIndex = i;
        }
    }

    if (outputCoverageLayerIndex == -1)
    {
        printError("Could not find output coverage layer for parsing objects");
        return false;
    }
    if (outputBBoxLayerIndex == -1)
    {
        printError("Could not find output bbox layer for parsing objects");
        return false;
    }

    float *outputCoverageBuffer =
        (float *)outputLayersInfo[outputCoverageLayerIndex].buffer;
    float *outputBboxBuffer =
        (float *)outputLayersInfo[outputBBoxLayerIndex].buffer;

    NvDsInferDimsCHW outputCoverageDims;
    NvDsInferDimsCHW outputBBoxDims;

    getDimsCHWFromDims(outputCoverageDims,
        outputLayersInfo[outputCoverageLayerIndex].inferDims);
    getDimsCHWFromDims(
        outputBBoxDims, outputLayersInfo[outputBBoxLayerIndex].inferDims);

    unsigned int targetShape[2] = { outputCoverageDims.w, outputCoverageDims.h };
    float bboxNorm[2] = { 35.0, 35.0 };
    float gcCenters0[targetShape[0]];
    float gcCenters1[targetShape[1]];
    int gridSize = outputCoverageDims.w * outputCoverageDims.h;
    int strideX = DIVIDE_AND_ROUND_UP(networkInfo.width, outputBBoxDims.w);
    int strideY = DIVIDE_AND_ROUND_UP(networkInfo.height, outputBBoxDims.h);

    for (unsigned int i = 0; i < targetShape[0]; i++)
    {
        gcCenters0[i] = (float)(i * strideX + 0.5);
        gcCenters0[i] /= (float)bboxNorm[0];
    }
    for (unsigned int i = 0; i < targetShape[1]; i++)
    {
        gcCenters1[i] = (float)(i * strideY + 0.5);
        gcCenters1[i] /= (float)bboxNorm[1];
    }

    unsigned int numClasses =
        std::min(outputCoverageDims.c, detectionParams.numClassesConfigured);
    for (unsigned int classIndex = 0; classIndex < numClasses; classIndex++)
    {

        /* Pointers to memory regions containing the (x1,y1) and (x2,y2) coordinates
         * of rectangles in the output bounding box layer. */
        float *outputX1 = outputBboxBuffer
            + classIndex * sizeof (float) * outputBBoxDims.h * outputBBoxDims.w;

        float *outputY1 = outputX1 + gridSize;
        float *outputX2 = outputY1 + gridSize;
        float *outputY2 = outputX2 + gridSize;

        /* Iterate through each point in the grid and check if the rectangle at that
         * point meets the minimum threshold criteria. */
        for (unsigned int h = 0; h < outputCoverageDims.h; h++)
        {
            for (unsigned int w = 0; w < outputCoverageDims.w; w++)
            {
                int i = w + h * outputCoverageDims.w;
                float confidence = outputCoverageBuffer[classIndex * gridSize + i];

                if (confidence < detectionParams.perClassPreclusterThreshold[classIndex])
                    continue;

                float rectX1Float, rectY1Float, rectX2Float, rectY2Float;

                /* Centering and normalization of the rectangle. */
                rectX1Float =
                    outputX1[w + h * outputCoverageDims.w] - gcCenters0[w];
                rectY1Float =
                    outputY1[w + h * outputCoverageDims.w] - gcCenters1[h];
                rectX2Float =
                    outputX2[w + h * outputCoverageDims.w] + gcCenters0[w];
                rectY2Float =
                    outputY2[w + h * outputCoverageDims.w] + gcCenters1[h];

                rectX1Float *= -bboxNorm[0];
                rectY1Float *= -bboxNorm[1];
                rectX2Float *= bboxNorm[0];
                rectY2Float *= bboxNorm[1];

                /* Clip parsed rectangles to frame bounds. */
                if (rectX1Float >= (int)m_NetworkInfo.width)
                    rectX1Float = m_NetworkInfo.width - 1;
                if (rectX2Float >= (int)m_NetworkInfo.width)
                    rectX2Float = m_NetworkInfo.width - 1;
                if (rectY1Float >= (int)m_NetworkInfo.height)
                    rectY1Float = m_NetworkInfo.height - 1;
                if (rectY2Float >= (int)m_NetworkInfo.height)
                    rectY2Float = m_NetworkInfo.height - 1;

                if (rectX1Float < 0)
                    rectX1Float = 0;
                if (rectX2Float < 0)
                    rectX2Float = 0;
                if (rectY1Float < 0)
                    rectY1Float = 0;
                if (rectY2Float < 0)
                    rectY2Float = 0;

                //Prevent underflows
                if(((rectX2Float - rectX1Float) < 0) || ((rectY2Float - rectY1Float) < 0))
                    continue;

                objectList.push_back({ classIndex, rectX1Float,
                         rectY1Float, (rectX2Float - rectX1Float),
                         (rectY2Float - rectY1Float), confidence});
            }
        }
    }
    return true;
}

/**
 * Filter out objects which have been specificed to be removed from the metadata
 * prior to clustering operation
 */
void DetectPostprocessor::preClusteringThreshold(
                           NvDsInferParseDetectionParams const &detectionParams,
                           std::vector<NvDsInferObjectDetectionInfo> &objectList)
{
    objectList.erase(std::remove_if(objectList.begin(), objectList.end(),
               [detectionParams](const NvDsInferObjectDetectionInfo& obj)
               { return (obj.classId >= detectionParams.numClassesConfigured) ||
                        (obj.detectionConfidence <
                        detectionParams.perClassPreclusterThreshold[obj.classId])
                        ? true : false;}),objectList.end());
}

/**
 * Filter out the top k objects with the highest probability and ignore the
 * rest
 */
void DetectPostprocessor::filterTopKOutputs(const int topK,
                          std::vector<NvDsInferObjectDetectionInfo> &objectList)
{
    if(topK < 0)
        return;

    std::stable_sort(objectList.begin(), objectList.end(),
                    [](const NvDsInferObjectDetectionInfo& obj1, const NvDsInferObjectDetectionInfo& obj2) {
                        return obj1.detectionConfidence > obj2.detectionConfidence; });
    objectList.resize(static_cast<size_t>(topK) <= objectList.size() ? topK : objectList.size());
}

std::vector<int>
DetectPostprocessor::nonMaximumSuppression(std::vector<std::pair<float, int>>& scoreIndex,
                                           std::vector<NvDsInferParseObjectInfo>& bbox,
                                           const float nmsThreshold)
{
    auto overlap1D = [](float x1min, float x1max, float x2min, float x2max) -> float {
        if (x1min > x2min)
        {
            std::swap(x1min, x2min);
            std::swap(x1max, x2max);
        }
        return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
    };

    auto computeIoU
        = [&overlap1D](NvDsInferParseObjectInfo& bbox1, NvDsInferParseObjectInfo& bbox2) -> float {
        float overlapX
            = overlap1D(bbox1.left, bbox1.left + bbox1.width, bbox2.left, bbox2.left + bbox2.width);
        float overlapY
            = overlap1D(bbox1.top, bbox1.top + bbox1.height, bbox2.top, bbox2.top + bbox2.height);
        float area1 = (bbox1.width) * (bbox1.height);
        float area2 = (bbox2.width) * (bbox2.height);
        float overlap2D = overlapX * overlapY;
        float u = area1 + area2 - overlap2D;
        return u == 0 ? 0 : overlap2D / u;
    };

    std::vector<int> indices;
    for (auto i : scoreIndex)
    {
        const int idx = i.second;
        bool keep = true;
        for (unsigned k = 0; k < indices.size(); ++k)
        {
            if (keep)
            {
                const int kept_idx = indices[k];
                float overlap = computeIoU(bbox.at(idx), bbox.at(kept_idx));
                keep = overlap <= nmsThreshold;
            }
            else
            {
                break;
            }
        }
        if (keep)
        {
            indices.push_back(idx);
        }
    }
    return indices;
}

/** Cluster objects using Non Max Suppression */
void
DetectPostprocessor::clusterAndFillDetectionOutputNMS(NvDsInferDetectionOutput &output)
{
    auto maxComp = [](const std::vector<NvDsInferObjectDetectionInfo>& c1,
                    const std::vector<NvDsInferObjectDetectionInfo>& c2) -> bool
                    { return c1.size() < c2.size(); };

    std::vector<std::pair<float, int>> scoreIndex;
    std::vector<NvDsInferObjectDetectionInfo> clusteredBboxes;
    auto maxElement = *std::max_element(m_PerClassObjectList.begin(),
                            m_PerClassObjectList.end(), maxComp);
    clusteredBboxes.reserve(maxElement.size() * m_NumDetectedClasses);

    for (unsigned int c = 0; c < m_NumDetectedClasses; c++)
    {
        if(!m_PerClassObjectList[c].empty())
        {
            scoreIndex.reserve(m_PerClassObjectList[c].size());
            scoreIndex.clear();
            for (size_t r = 0; r < m_PerClassObjectList[c].size(); ++r)
            {
                scoreIndex.emplace_back(std::make_pair(m_PerClassObjectList[c][r].detectionConfidence, r));
            }
            std::stable_sort(scoreIndex.begin(), scoreIndex.end(),
                            [](const std::pair<float, int>& pair1, const std::pair<float, int>& pair2) {
                                return pair1.first > pair2.first; });
            // Apply NMS algorithm
            const std::vector<int> indices = nonMaximumSuppression(scoreIndex, m_PerClassObjectList[c],
                            m_PerClassDetectionParams[c].nmsIOUThreshold);

            std::vector<NvDsInferObjectDetectionInfo> postNMSBboxes;
            for(auto idx : indices) {
                if(m_PerClassObjectList[c][idx].detectionConfidence >
                m_PerClassDetectionParams[c].postClusterThreshold)
                {
                    postNMSBboxes.emplace_back(m_PerClassObjectList[c][idx]);
                }
            }
            filterTopKOutputs(m_PerClassDetectionParams.at(c).topK, postNMSBboxes);
            clusteredBboxes.insert(clusteredBboxes.end(),postNMSBboxes.begin(), postNMSBboxes.end());
        }
    }

    output.objects = new NvDsInferObject[clusteredBboxes.size()];
    output.numObjects = 0;

    for(uint i=0; i < clusteredBboxes.size(); ++i)
    {
        NvDsInferObject &object = output.objects[output.numObjects];
        object.left = clusteredBboxes[i].left;
        object.top = clusteredBboxes[i].top;
        object.width = clusteredBboxes[i].width;
        object.height = clusteredBboxes[i].height;
        object.classIndex = clusteredBboxes[i].classId;
        object.label = nullptr;
        object.mask = nullptr;
        if (object.classIndex < static_cast<int>(m_Labels.size()) && m_Labels[object.classIndex].size() > 0)
                object.label = strdup(m_Labels[object.classIndex][0].c_str());
        object.confidence = clusteredBboxes[i].detectionConfidence;
        object.rotation_angle = clusteredBboxes[i].rotation_angle;
        output.numObjects++;
    }
}

#ifdef WITH_OPENCV
/**
 * Cluster objects using OpenCV groupRectangles and fill the output structure.
 */
void
DetectPostprocessor::clusterAndFillDetectionOutputCV(NvDsInferDetectionOutput& output)
{
    size_t totalObjects = 0;

    for (auto & list:m_PerClassCvRectList)
        list.clear();

    /* The above functions will add all objects in the m_ObjectList vector.
     * Need to seperate them per class for grouping. */
    for (auto & object:m_ObjectList)
    {
        m_PerClassCvRectList[object.classId].emplace_back(object.left,
                object.top, object.width, object.height);
    }

    for (unsigned int c = 0; c < m_NumDetectedClasses; c++)
    {
        /* Cluster together rectangles with similar locations and sizes
         * since these rectangles might represent the same object. Refer
         * to opencv documentation of groupRectangles for more
         * information about the tuning parameters for grouping. */
        if (m_PerClassDetectionParams[c].groupThreshold > 0)
            cv::groupRectangles(m_PerClassCvRectList[c],
                    m_PerClassDetectionParams[c].groupThreshold,
                    m_PerClassDetectionParams[c].eps);
        totalObjects += m_PerClassCvRectList[c].size();
    }

    output.objects = new NvDsInferObject[totalObjects];
    output.numObjects = 0;

    for (unsigned int c = 0; c < m_NumDetectedClasses; c++)
    {
        /* Add coordinates and class ID and the label of all objects
         * detected in the frame to the frame output. */
        for (auto & rect:m_PerClassCvRectList[c])
        {
            NvDsInferObject &object = output.objects[output.numObjects];
            object.left = rect.x;
            object.top = rect.y;
            object.width = rect.width;
            object.height = rect.height;
            object.classIndex = c;
            object.label = nullptr;
            object.mask = nullptr;
            if (c < m_Labels.size() && m_Labels[c].size() > 0)
                object.label = strdup(m_Labels[c][0].c_str());
            object.confidence = -0.1;
            output.numObjects++;
        }
    }
}
#endif

/**
 * Cluster objects using DBSCAN and fill the output structure.
 */
void
DetectPostprocessor::clusterAndFillDetectionOutputDBSCAN(NvDsInferDetectionOutput& output)
{
    size_t totalObjects = 0;
    NvDsInferDBScanClusteringParams clusteringParams;
    clusteringParams.enableATHRFilter = ATHR_ENABLED;
    clusteringParams.thresholdATHR = ATHR_THRESHOLD;
    assert(m_DBScanHandle);

    for (unsigned int c = 0; c < m_NumDetectedClasses; c++)
    {
        NvDsInferObjectDetectionInfo *objArray = m_PerClassObjectList[c].data();
        size_t numObjects = m_PerClassObjectList[c].size();
        NvDsInferDetectionParams detectionParams = m_PerClassDetectionParams[c];

        clusteringParams.eps = detectionParams.eps;
        clusteringParams.minBoxes = detectionParams.minBoxes;
        clusteringParams.minScore = detectionParams.minScore;

        /* Cluster together rectangles with similar locations and sizes
         * since these rectangles might represent the same object using
         * DBSCAN. */
        if (detectionParams.minBoxes > 0) {
            NvDsInferDBScanCluster(
                m_DBScanHandle.get(), &clusteringParams, objArray, &numObjects);
        }
        m_PerClassObjectList[c].resize(numObjects);
        m_PerClassObjectList[c].erase(std::remove_if(m_PerClassObjectList[c].begin(),
               m_PerClassObjectList[c].end(),
               [detectionParams](const NvDsInferObjectDetectionInfo& obj)
               { return (obj.detectionConfidence <
                        detectionParams.postClusterThreshold)
                        ? true : false;}),m_PerClassObjectList[c].end());
        filterTopKOutputs(m_PerClassDetectionParams.at(c).topK, m_PerClassObjectList.at(c));
        totalObjects += m_PerClassObjectList[c].size();
    }

    output.objects = new NvDsInferObject[totalObjects];
    output.numObjects = 0;

    for (unsigned int c = 0; c < m_NumDetectedClasses; c++)
    {
        /* Add coordinates and class ID and the label of all objects
         * detected in the frame to the frame output. */
        for (size_t i = 0; i < m_PerClassObjectList[c].size(); i++)
        {
            NvDsInferObject &object = output.objects[output.numObjects];
            object.left = m_PerClassObjectList[c][i].left;
            object.top = m_PerClassObjectList[c][i].top;
            object.width = m_PerClassObjectList[c][i].width;
            object.height = m_PerClassObjectList[c][i].height;
            object.classIndex = c;
            object.label = nullptr;
            object.mask = nullptr;
            if (c < m_Labels.size() && m_Labels[c].size() > 0)
                object.label = strdup(m_Labels[c][0].c_str());
            object.confidence = m_PerClassObjectList[c][i].detectionConfidence;
            object.rotation_angle = m_PerClassObjectList[c][i].rotation_angle;
            output.numObjects++;
        }
    }
}

/**
 * Cluster objects using a hybrid algorithm of DBSCAN + NMS
 * and fill the output structure.
 */
void
DetectPostprocessor::clusterAndFillDetectionOutputHybrid(NvDsInferDetectionOutput& output)
{
    NvDsInferDBScanClusteringParams clusteringParams;
    clusteringParams.enableATHRFilter = ATHR_ENABLED;
    clusteringParams.thresholdATHR = ATHR_THRESHOLD;
    assert(m_DBScanHandle);

    for (unsigned int c = 0; c < m_NumDetectedClasses; c++)
    {
        NvDsInferObjectDetectionInfo *objArray = m_PerClassObjectList[c].data();
        size_t numObjects = m_PerClassObjectList[c].size();

        clusteringParams.eps = m_PerClassDetectionParams[c].eps;
        clusteringParams.minBoxes = m_PerClassDetectionParams[c].minBoxes;
        clusteringParams.minScore = m_PerClassDetectionParams[c].minScore;

        /* Cluster together rectangles with similar locations and sizes
         * since these rectangles might represent the same object using
         * DBSCAN. */
        if (m_PerClassDetectionParams[c].minBoxes > 0) {
            NvDsInferDBScanClusterHybrid(
                m_DBScanHandle.get(), &clusteringParams, objArray, &numObjects);
        }
        m_PerClassObjectList[c].resize(numObjects);
    }

    return clusterAndFillDetectionOutputNMS(output);
}

/**
 * full the output structure without performing any clustering operations
 */

void
DetectPostprocessor::fillUnclusteredOutput(NvDsInferDetectionOutput& output)
{
    for (auto & object:m_ObjectList)
    {
        m_PerClassObjectList[object.classId].emplace_back(object);
    }

    unsigned int totalObjects = 0;
    for (unsigned int c = 0; c < m_NumDetectedClasses; c++)
    {
        filterTopKOutputs(m_PerClassDetectionParams.at(c).topK, m_PerClassObjectList.at(c));
        totalObjects += m_PerClassObjectList.at(c).size();
    }

    output.objects = new NvDsInferObject[totalObjects];
    output.numObjects = 0;
    for(const auto& perClassList : m_PerClassObjectList)
    {
        for(const auto& obj: perClassList)
        {
            NvDsInferObject &object = output.objects[output.numObjects];
            object.left = obj.left;
            object.top = obj.top;
            object.width = obj.width;
            object.height = obj.height;
            object.classIndex = obj.classId;
            object.label = nullptr;
            object.mask = nullptr;
            if(obj.classId < m_Labels.size() && m_Labels[obj.classId].size() > 0)
                object.label = strdup(m_Labels[obj.classId][0].c_str());
            object.confidence = obj.detectionConfidence;
            object.rotation_angle = obj.rotation_angle;
            ++output.numObjects;
        }
    }
}

/**
 * full the output structure without performing any clustering operations
 */

void
InstanceSegmentPostprocessor::fillUnclusteredOutput(NvDsInferDetectionOutput& output)
{
    for (auto & object:m_InstanceMaskList)
    {
        m_PerClassInstanceMaskList[object.classId].emplace_back(object);
    }

    unsigned int totalObjects = 0;
    for (unsigned int c = 0; c < m_NumDetectedClasses; c++)
    {
        filterTopKOutputs(m_PerClassDetectionParams.at(c).topK, m_PerClassInstanceMaskList.at(c));
        totalObjects += m_PerClassInstanceMaskList.at(c).size();
    }

    output.objects = new NvDsInferObject[totalObjects];
    output.numObjects = 0;
    for(const auto& perClassList : m_PerClassInstanceMaskList)
    {
        for(const auto& obj: perClassList)
        {
            NvDsInferObject &object = output.objects[output.numObjects];
            object.left = obj.left;
            object.top = obj.top;
            object.width = obj.width;
            object.height = obj.height;
            object.classIndex = obj.classId;
            object.label = nullptr;
            if(obj.classId < m_Labels.size() && m_Labels[obj.classId].size() > 0)
                object.label = strdup(m_Labels[obj.classId][0].c_str());
            object.confidence = obj.detectionConfidence;
            object.rotation_angle = 0.0f;
            object.mask = nullptr;
            if (obj.mask) {
                object.mask = std::move(obj.mask);
                object.mask_width = obj.mask_width;
                object.mask_height = obj.mask_height;
                object.mask_size = obj.mask_size;
            }
            ++output.numObjects;
        }
    }
}

/**
 * Filter out objects which have been specificed to be removed from the metadata
 * prior to clustering operation
 */
void InstanceSegmentPostprocessor::preClusteringThreshold(
                           NvDsInferParseDetectionParams const &detectionParams,
                           std::vector<NvDsInferInstanceMaskInfo> &objectList)
{
    objectList.erase(std::remove_if(objectList.begin(), objectList.end(),
               [detectionParams](const NvDsInferInstanceMaskInfo& obj)
               { return (obj.classId >= detectionParams.numClassesConfigured) ||
                        (obj.detectionConfidence <
                        detectionParams.perClassPreclusterThreshold[obj.classId])
                        ? true : false;}),objectList.end());
}

/**
 * Filter out the top k objects with the highest probability and ignore the
 * rest
 */
void InstanceSegmentPostprocessor::filterTopKOutputs(const int topK,
                          std::vector<NvDsInferInstanceMaskInfo> &objectList)
{
    if(topK < 0)
        return;

    std::stable_sort(objectList.begin(), objectList.end(),
                    [](const NvDsInferInstanceMaskInfo& obj1, const NvDsInferInstanceMaskInfo& obj2) {
                        return obj1.detectionConfidence > obj2.detectionConfidence; });
    objectList.resize(static_cast<size_t>(topK) <= objectList.size() ? topK : objectList.size());
}

bool
ClassifyPostprocessor::parseAttributesFromSoftmaxLayers(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo, float classifierThreshold,
    std::vector<NvDsInferAttribute>& attrList, std::string& attrString)
{
    /* Get the number of attributes supported by the classifier. */
    unsigned int numAttributes = m_OutputLayerInfo.size();

    /* Iterate through all the output coverage layers of the classifier.
    */
    for (unsigned int l = 0; l < numAttributes; l++)
    {
        /* outputCoverageBuffer for classifiers is usually a softmax layer.
         * The layer is an array of probabilities of the object belonging
         * to each class with each probability being in the range [0,1] and
         * sum all probabilities will be 1.
         */
        NvDsInferDimsCHW dims;

        getDimsCHWFromDims(dims, m_OutputLayerInfo[l].inferDims);
        unsigned int numClasses = dims.c;
        float *outputCoverageBuffer =
            (float *)m_OutputLayerInfo[l].buffer;
        float maxProbability = 0;
        bool attrFound = false;
        NvDsInferAttribute attr;

        /* Iterate through all the probabilities that the object belongs to
         * each class. Find the maximum probability and the corresponding class
         * which meets the minimum threshold. */
        for (unsigned int c = 0; c < numClasses; c++)
        {
            float probability = outputCoverageBuffer[c];
            if (probability > m_ClassifierThreshold
                    && probability > maxProbability)
            {
                maxProbability = probability;
                attrFound = true;
                attr.attributeIndex = l;
                attr.attributeValue = c;
                attr.attributeConfidence = probability;
            }
        }
        if (attrFound)
        {
            if (m_Labels.size() > attr.attributeIndex &&
                    attr.attributeValue < m_Labels[attr.attributeIndex].size())
                attr.attributeLabel =
                    strdup(m_Labels[attr.attributeIndex][attr.attributeValue].c_str());
            else
                attr.attributeLabel = nullptr;
            attrList.push_back(attr);
            if (attr.attributeLabel)
                attrString.append(attr.attributeLabel).append(" ");
        }
    }

    return true;
}

NvDsInferStatus
InstanceSegmentPostprocessor::fillDetectionOutput(
    const std::vector<NvDsInferLayerInfo>& outputLayers,
    NvDsInferDetectionOutput& output)
{
    /* Clear the object lists. */
    m_InstanceMaskList.clear();

    /* Clear all per class object lists */
    for (auto & list:m_PerClassInstanceMaskList)
        list.clear();

    /* Call custom parsing function if specified otherwise use the one
     * written along with this implementation. */
    if (m_CustomParseFunc)
    {
        if (!m_CustomParseFunc(outputLayers, m_NetworkInfo,
                    m_DetectionParams, m_InstanceMaskList))
        {
            printError("Failed to parse bboxes and instance mask using custom parse function");
            return NVDSINFER_CUSTOM_LIB_FAILED;
        }
    }
    else
    {
        printError("Failed to find custom parse function");
        return NVDSINFER_OUTPUT_PARSING_FAILED;
    }

    preClusteringThreshold(m_DetectionParams, m_InstanceMaskList);

    switch (m_ClusterMode)
    {
        case NVDSINFER_CLUSTER_NONE:
            fillUnclusteredOutput(output);
            break;
        default:
            printError("Invalid cluster mode for instance mask detection");
            return NVDSINFER_OUTPUT_PARSING_FAILED;
    }
    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
DetectPostprocessor::fillDetectionOutput(
    const std::vector<NvDsInferLayerInfo>& outputLayers,
    NvDsInferDetectionOutput& output)
{
    /* Clear the object lists. */
    m_ObjectList.clear();

    /* Clear all per class object lists */
    for (auto & list:m_PerClassObjectList)
        list.clear();

    /* Call custom parsing function if specified otherwise use the one
     * written along with this implementation. */
    if (m_CustomBBoxParseFunc)
    {
        if (!m_CustomBBoxParseFunc(outputLayers, m_NetworkInfo,
                    m_DetectionParams, m_ObjectList))
        {
            printError("Failed to parse bboxes using custom parse function");
            return NVDSINFER_CUSTOM_LIB_FAILED;
        }
    }
    else
    {
        if (!parseBoundingBox(outputLayers, m_NetworkInfo,
                    m_DetectionParams, m_ObjectList))
        {
            printError("Failed to parse bboxes");
            return NVDSINFER_OUTPUT_PARSING_FAILED;
        }
    }

    preClusteringThreshold(m_DetectionParams, m_ObjectList);

    /* The above functions will add all objects in the m_ObjectList vector.
     * Need to seperate them per class for grouping. */
#ifndef WITH_OPENCV
    if(m_ClusterMode != NVDSINFER_CLUSTER_NONE)
#else
    if((m_ClusterMode != NVDSINFER_CLUSTER_GROUP_RECTANGLES) &&
        (m_ClusterMode != NVDSINFER_CLUSTER_NONE))
#endif
    {
        for (auto & object:m_ObjectList)
        {
            m_PerClassObjectList[object.classId].emplace_back(object);
        }
    }

    switch (m_ClusterMode)
    {
        case NVDSINFER_CLUSTER_NMS:
            clusterAndFillDetectionOutputNMS(output);
            break;

        case NVDSINFER_CLUSTER_DBSCAN:
            clusterAndFillDetectionOutputDBSCAN(output);
            break;

#ifdef WITH_OPENCV
        case NVDSINFER_CLUSTER_GROUP_RECTANGLES:
            clusterAndFillDetectionOutputCV(output);
            break;
#endif

        case NVDSINFER_CLUSTER_DBSCAN_NMS_HYBRID:
            clusterAndFillDetectionOutputHybrid(output);
            break;

        case NVDSINFER_CLUSTER_NONE:
            fillUnclusteredOutput(output);
            break;

        default:
            break;
    }

    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
ClassifyPostprocessor::fillClassificationOutput(
    const std::vector<NvDsInferLayerInfo>& outputLayers,
    NvDsInferClassificationOutput& output)
{
    string attrString;
    vector<NvDsInferAttribute> attributes;

    /* Call custom parsing function if specified otherwise use the one
     * written along with this implementation. */
    if (m_CustomClassifierParseFunc)
    {
        if (!m_CustomClassifierParseFunc(outputLayers, m_NetworkInfo,
                m_ClassifierThreshold, attributes, attrString))
        {
            printError("Failed to parse classification attributes using "
                    "custom parse function");
            return NVDSINFER_CUSTOM_LIB_FAILED;
        }
        // apply labels to attributes if not set
        for (auto &attr : attributes) {
            if (attr.attributeLabel == nullptr && m_Labels.size() > attr.attributeIndex &&
                    attr.attributeValue < m_Labels[attr.attributeIndex].size()) {
                attr.attributeLabel =
                    strdup(m_Labels[attr.attributeIndex][attr.attributeValue].c_str());
                attrString.append(attr.attributeLabel).append(" ");
            }
        }
    }
    else
    {
        if (!parseAttributesFromSoftmaxLayers(outputLayers, m_NetworkInfo,
                m_ClassifierThreshold, attributes, attrString))
        {
            printError("Failed to parse bboxes");
            return NVDSINFER_OUTPUT_PARSING_FAILED;
        }
    }

    /* Fill the output structure with the parsed attributes. */
    output.label = strdup(attrString.c_str());
    output.numAttributes = attributes.size();
    output.attributes = new NvDsInferAttribute[output.numAttributes];
    for (size_t i = 0; i < output.numAttributes; i++)
    {
        output.attributes[i].attributeIndex = attributes[i].attributeIndex;
        output.attributes[i].attributeValue = attributes[i].attributeValue;
        output.attributes[i].attributeConfidence = attributes[i].attributeConfidence;
        output.attributes[i].attributeLabel = attributes[i].attributeLabel;
    }
    return NVDSINFER_SUCCESS;
}

bool SegmentPostprocessor::parseSemanticSegmentationOutput(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    float segmentationThreshold,
    unsigned int numClasses,
    int* classificationMap,
    float*& classProbabilityMap) {
    assert(classificationMap);

    for (auto const& layer : outputLayersInfo) {
        if (layer.dataType != FLOAT) {
            printError(
                "Default segment parsing function supports datatype"
                "FP32 only but received output tensor: %s with datatype: %s",
                safeStr(layer.layerName),
                safeStr(
                    dataType2Str(static_cast<NvDsInferDataType>(layer.dataType))));
            return false;
        }
    }

    std::function<unsigned int(unsigned int, unsigned int, unsigned int)> indAlongChannel = nullptr;
    NvDsInferDimsCHW outputDimsCHW = {0};

    if (m_SegmentationOutputOrder == NvDsInferTensorOrder_kNCHW) {
        getDimsCHWFromDims(outputDimsCHW, outputLayersInfo[0].inferDims);
        indAlongChannel = [&outputDimsCHW](int x, int y, int c)-> int
            { return c * outputDimsCHW.w * outputDimsCHW.h + y * outputDimsCHW.w + x;};
    }
    else if (m_SegmentationOutputOrder == NvDsInferTensorOrder_kNHWC) {
        getDimsHWCFromDims(outputDimsCHW, outputLayersInfo[0].inferDims);
        indAlongChannel = [&outputDimsCHW](int x, int y, int c)-> int
            {return  outputDimsCHW.c * ( y * outputDimsCHW.w + x) + c;};
    }
    if (numClasses != outputDimsCHW.c) {
        printError(
                "Configured number for classes %u differs from"
                " number of channels in output %u",
                 numClasses, outputDimsCHW.c);
        return false;
    }

    classProbabilityMap = (float*)outputLayersInfo[0].buffer;

    for (unsigned int y = 0; y < networkInfo.height; y++) {
        for (unsigned int x = 0; x < networkInfo.width; x++) {
            float max_prob = -1;
            int& cls = classificationMap[y * networkInfo.width + x] = -1;
            for (unsigned int c = 0; c < numClasses; c++) {
                float prob = classProbabilityMap[indAlongChannel(x,y,c)];
                if (prob > max_prob && prob > segmentationThreshold) {
                    cls = c;
                    max_prob = prob;
                }
            }
        }
    }
    return true;
}

NvDsInferStatus SegmentPostprocessor::fillSegmentationOutput(
    const std::vector<NvDsInferLayerInfo>& outputLayers,
    NvDsInferSegmentationOutput& output) {

    output.width = m_NetworkInfo.width;
    output.height = m_NetworkInfo.height;
    output.classes = m_NumSegmentationClasses;
    output.class_map = new int[output.width * output.height];
    output.class_probability_map = nullptr;


    /* Call custom parsing function if specified otherwise use the one
     * written along with this implementation. */
    if (m_CustomSegmentationParseFunc) {
        if (!m_CustomSegmentationParseFunc(outputLayers, m_NetworkInfo,
                m_SegmentationThreshold, m_NumSegmentationClasses,
                output.class_map, output.class_probability_map)) {
            printError(
                "Failed to parse semantic segmentation output using "
                "custom parse function");
            return NVDSINFER_CUSTOM_LIB_FAILED;
        }
    } else {
        if (!parseSemanticSegmentationOutput(outputLayers, m_NetworkInfo,
                m_SegmentationThreshold, m_NumSegmentationClasses,
                output.class_map, output.class_probability_map)) {
            printError(
                "Failed to parse semantic segmentation output.");
            return NVDSINFER_OUTPUT_PARSING_FAILED;
        }
    }

    return NVDSINFER_SUCCESS;
}

void
InferPostprocessor::releaseFrameOutput(NvDsInferFrameOutput& frameOutput)
{
    switch (frameOutput.outputType)
    {
        case NvDsInferNetworkType_Detector:
            for (unsigned int j = 0; j < frameOutput.detectionOutput.numObjects;
                    j++)
            {
                free(frameOutput.detectionOutput.objects[j].label);
            }
            delete[] frameOutput.detectionOutput.objects;
            break;
        case NvDsInferNetworkType_Classifier:
            for (unsigned int j = 0; j < frameOutput.classificationOutput.numAttributes;
                    j++)
            {
                if (frameOutput.classificationOutput.attributes[j].attributeLabel)
                    free(frameOutput.classificationOutput.attributes[j].attributeLabel);
            }
            free(frameOutput.classificationOutput.label);
            delete[] frameOutput.classificationOutput.attributes;
            break;
        case NvDsInferNetworkType_Segmentation:
            delete[] frameOutput.segmentationOutput.class_map;
            break;
        case NvDsInferNetworkType_InstanceSegmentation:
            for (unsigned int j = 0; j < frameOutput.detectionOutput.numObjects;
                    j++)
            {
                free(frameOutput.detectionOutput.objects[j].label);
                delete[] frameOutput.detectionOutput.objects[j].mask;
            }
            delete[] frameOutput.detectionOutput.objects;
            break;
        default:
            break;
    }
}

} // namespace nvdsinfer
