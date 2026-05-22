/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef __NVDSINFERSERVER_POST_PROCESS_BUF_H__
#define __NVDSINFERSERVER_POST_PROCESS_BUF_H__

#include <condition_variable>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <queue>

#include "infer_batch_buffer.h"
#include "infer_cuda_utils.h"
#include "infer_datatypes.h"
#include "infer_iprocess.h"
#include "infer_post_datatypes.h"
#include "nvdsinfer.h"
#include "nvdsinfer_custom_impl.h"
#include "nvdsinferserver_config.pb.h"

namespace ic = nvdsinferserver::config;

struct NvDsInferDBScan;

namespace nvdsinferserver {

class DetectionOutput : public BaseBatchBuffer {
public:
    DetectionOutput() : BaseBatchBuffer(0)
    {
        InferBufferDescription desc{
            memType : InferMemType::kCpu,
            devId : 0,
            dataType : InferDataType::kNone,
            dims : {1, {1}, 1},
            elementSize : sizeof(NvDsInferDetectionOutput),
            name : INFER_SERVER_DETECTION_BUF_NAME,
            isInput : false,
        };
        setBufDesc(desc);
    }
    ~DetectionOutput()
    {
        for (auto& batch : m_Objects)
            for (auto& object : batch)
                if (object.label) {
                    free(object.label);
                }
    }
    void swapObjects(std::vector<std::vector<NvDsInferObject>>& objs)
    {
        setBatchSize(objs.size());
        m_Objects.swap(objs);
        m_BufPtrs.resize(m_Objects.size());
        for (size_t i = 0; i < m_BufPtrs.size(); ++i) {
            NvDsInferDetectionOutput& bufPtr = m_BufPtrs[i];
            bufPtr.numObjects = m_Objects.at(i).size();
            bufPtr.objects = m_Objects.at(i).data();
        }
    }
    /// return @a NvDsInferDetectionOutput* pointer
    void* getBufPtr(uint32_t batchIdx) const override
    {
        assert(batchIdx < (uint32_t)m_BufPtrs.size());
        return (void*)&m_BufPtrs.at(batchIdx);
    }

private:
    std::vector<std::vector<NvDsInferObject>> m_Objects;
    std::vector<NvDsInferDetectionOutput> m_BufPtrs;
};

class ClassificationOutput : public BaseBatchBuffer {
public:
    ClassificationOutput(int batchSize) : BaseBatchBuffer(batchSize)
    {
        InferBufferDescription desc{
            memType : InferMemType::kCpu,
            devId : 0,
            dataType : InferDataType::kNone,
            dims : {1, {1}, 1},
            elementSize : sizeof(InferClassificationOutput),
            name : INFER_SERVER_CLASSIFICATION_BUF_NAME,
            isInput : false,
        };
        setBufDesc(desc);
        m_BufPtrs.resize(batchSize);
    }
    ~ClassificationOutput() {}
    InferClassificationOutput& mutableOutput(uint32_t idx)
    {
        assert(idx < m_BufPtrs.size());
        return m_BufPtrs[idx];
    }
    /// return @a InferClassificationOutput* pointer
    void* getBufPtr(uint32_t batchIdx) const override
    {
        assert(batchIdx < (uint32_t)m_BufPtrs.size());
        return (void*)&m_BufPtrs.at(batchIdx);
    }

    /// copy all lables to safe place
    void finalize()
    {
        for (uint32_t i = 0; i < m_BufPtrs.size(); ++i) {
            InferClassificationOutput& data = mutableOutput(i);
            for (size_t k = 0; k < data.attributes.size(); ++k) {
                auto& attrib = data.attributes[k];
                if (attrib.safeAttributeLabel.empty()) {
                    attrib.safeAttributeLabel = safeStr(attrib.attributeLabel);
                }
                attrib.attributeLabel = (char *) attrib.safeAttributeLabel.c_str();
            }
        }
    }

private:
    std::vector<InferClassificationOutput> m_BufPtrs;
};

using SharedClassOutput = std::shared_ptr<ClassificationOutput>;

class SegmentationOutput : public BaseBatchBuffer {
public:
    SegmentationOutput(int batchSize) : BaseBatchBuffer(batchSize)
    {
        InferBufferDescription desc{
            memType : InferMemType::kCpu,
            devId : 0,
            dataType : InferDataType::kNone,
            dims : {1, {1}, 1},
            elementSize : sizeof(NvDsInferSegmentationOutput),
            name : INFER_SERVER_SEGMENTATION_BUF_NAME,
            isInput : false,
        };
        setBufDesc(desc);
        m_BufPtrs.resize(batchSize, NvDsInferSegmentationOutput{0});
    }
    ~SegmentationOutput()
    {
        for (NvDsInferSegmentationOutput& out : m_BufPtrs) {
            if (out.class_map) {
                delete[] out.class_map;
            }
        }
    }
    NvDsInferSegmentationOutput& mutableOutput(uint32_t idx)
    {
        assert(idx < m_BufPtrs.size());
        return m_BufPtrs[idx];
    }
    /// return @a NvDsInferSegmentationOutput* pointer
    void* getBufPtr(uint32_t batchIdx) const override
    {
        assert(batchIdx < (uint32_t)m_BufPtrs.size());
        return (void*)&m_BufPtrs.at(batchIdx);
    }

private:
    std::vector<NvDsInferSegmentationOutput> m_BufPtrs;
};

}  // namespace nvdsinferserver

#endif  // __NVDSINFERSERVER_POST_PROCESS_BUF_H__