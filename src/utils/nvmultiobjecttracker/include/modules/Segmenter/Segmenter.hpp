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

#ifndef SEGMENTERLIB_SEGMENTER_HPP_
#define SEGMENTERLIB_SEGMENTER_HPP_

#include <deque>
#include <map>
#include <set>

#ifdef VIZ_DEBUG_SEG
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#endif

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include "cuda.h"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include "modules/Segmenter/ISegmenter.hpp"
#include "modules/FrameTransform/trtModule.hpp"
#include "modules/ReID/ReIDFrameTransformTexture.hpp"
#include "modules/ReID/ReIDCropScaler.hpp"

enum SAM_POINT_TYPE : int
{
    BACKGROUND = 0, //!< Negative point prompt.
    FOREGROUND = 1, //!< Positive point prompt.
    TOP_LEFT = 2, //!< Top-left corner of a bounding-box prompt.
    BOTTOM_RIGHT = 3 //!< Bottom-right corner of a bounding-box prompt.
};

typedef std::shared_ptr<std::vector<float>> TensorPtr;

/** @brief Holds per-frame segmentation features retained in memory. */
struct SegmentMemoryInfo
{
    TensorPtr pred_masks; //!< Predicted masks.
    TensorPtr pred_masks_high_res; //!< Predicted high-resolution masks.
    TensorPtr obj_ptr; //!< Object pointer.
    TensorPtr maskmem_features; //!< Mask memory features.
    TensorPtr maskmem_pos_enc; //!< Mask memory position encoding.
    TensorPtr temporal_code; //!< Temporal code.
};

/** @brief Records targets awaiting memory-encoder processing. */
struct MemoryEncoderStatus
{
    TensorPtr maskmem_features; //!< Mask memory features.
    TensorPtr maskmem_pos_enc; //!< Mask memory position encoding.
    TensorPtr temporal_code; //!< Temporal code.
};

/** @brief Tracks pending segmentation inference for one stream. */
struct InferenceStatus
{
    NvMOTFrameNum currentFrameNum; //!< Target age at the latest memory update.
    std::vector<TensorPtr> objPtrCond; //!< Object pointers of the conditional frames.
    std::vector<MemoryEncoderStatus> statusCond; //!< Memory encoder status of the conditional frames.
    std::vector<TensorPtr> objPtrUncond; //!< Object pointers of the unconditional frames.
    std::vector<MemoryEncoderStatus> statusUncond; //!< Memory encoder status of the unconditional frames.
};

/** @brief Tracks batched CUDA inference work across streams. */
struct CudaInferenceStatus
{
    NvMOTFrameNum currentFrameNum; //!< Target age at the latest memory update.
    int numStatusUncond; //!< Number of unconditional memory entries stored.
    int numObjPtrUncond; //!< Number of object pointers stored.
};

/** @brief Stores dimensions and batch limits used at segmenter runtime. */
struct SegmenterRunTimeParams
{
    int featuresEncoderSize; //!< Image encoder basic resolution output feature size (single image).
    int highResFeaturesEncoder0Size; //!< Image encoder high resolution (0) output feature size (single image).
    int highResFeaturesEncoder1Size; //!< Image encoder high resolution (1) output feature size (single image).
    int pixFeatWithMemSize; //!< Memory attention output feature size in a batch.
    int maxConcatMemFeatSize; //!< Memory attention input feature max size in a batch.
    int maskMemFeatSize; //!< Memory encoder output memory feature size in a batch.
    int temporalCodeSize; //!< Temporal code size.
    int objPtrSize; //!< Mask decoder output obj ptr size in a batch.
    int basicMaskSize; //!< Mask decoder basic resolution output mask size in a batch.
    int highResMaskSize; //!< Mask decoder high resolution output mask size in a batch.
    int maxPointNum; //!< Max number of points for each object.
    int memChannel; //!< Number of channels in memory feature.
    int batchSize; //!< Batch size.
};

/** @brief Stores configuration for segmentation inference. */
class SegmenterConfigParams : public NvTrackerParams
{
public:
    /** @brief Construct a SegmenterConfigParams instance. */
    SegmenterConfigParams();

    /**
     * @brief Construct a SegmenterConfigParams instance.
     * @param[in] configFilePath Path to the tracker configuration file.
     * @param[in] moduleName Configuration module name.
     */
    SegmenterConfigParams(const std::string &configFilePath, const std::string moduleName);

    /**
     * @brief Construct a SegmenterConfigParams instance.
     * @param[in] fn Parsed YAML node.
     */
    SegmenterConfigParams(const YAML::Node &fn);

    /**
     * @brief Load configuration values from a YAML node.
     * @param[in] fn Parsed YAML node.
     */
    void read(const YAML::Node &fn) override;

    SEGMENTER_TYPES m_segmenterType; //!< Type of segmenter.
    int m_hiddenDim; //!< Hidden feature dimension.
    std::vector<int> m_inferDims; //!< Network input dimension.
    int m_inputOrder; //!< Input dimension order: 0: NCHW, 1: NHWC.
    int m_encoderFeatureDim; //!< Encoder feature dimension.
    int m_maxMemVecSize_legacy; //!< Maximum memory vector size of the legacy work spaces.
    int m_width; //!< Input image patch width.
    int m_height; //!< Input image patch height.
    int m_channel; //!< Input image patch channel.
    int m_gpuId; //!< Gpu id.
    float m_netScaleFactor; //!< Input normalization factor.
    std::vector<float> m_offsets; //!< List of mean values of color components to be subtracted from input.
    NvBufSurfaceColorFormat m_colorFormat; //!< Color format of input to segmentation network.
    int m_promptType; //!< Prompt type.
    //!< A value of 0 uses a foreground point, while 1 uses a box.
    int m_outputLabelNum; //!< Number of decoder output labels for each target.
    //!< Each label corresponds to a mask, and the implementation currently uses only label 0.
    bool m_useVPICropScaler; //!< Use VPI crop scaler instead of in built crop scaler.
    std::string m_vizDir; //!< Dir to save visualization dumps.
    bool m_saveFrameViz; //!< Save overlay masks for each target in each frame.
    bool m_saveIDMask; //!< Enables saving encoded target-ID masks.
    //!< Each frame stores all targets in one four-channel RGBA image with one four-byte target ID per pixel.
    bool m_outputMask; //!< Output mask to tracker plugin.
    bool m_useMemBank; //!< Use memory bank.
    bool m_useCudaMemBank; //!< Use cuda memory bank.
    bool m_updateMemBankPrompt; //!< Update memory bank using bbox prompt.
    int m_numMaskMem; //!< Number of memory bank masks.
    int m_debugVerboseLevel; //!< Debug message verbose mode level.
    float m_promptBboxPaddingScale; //!< Padding of the bounding-box prompt as a fraction of the box height.
    int m_numObjPtr; //!< Number of object pointers.
    int m_numConditionalMem; //!< Number of conditional frames to use for memory bank update.
    int m_maskMemoryUpdateInterval; //!< Frame interval to update memory bank.
    float m_minIouDiff4MemoryUpdate; //!< IOU threshold to update memory bank.
    float m_minIouPrevFrame4ProgBbox; //!< IOU threshold to use detection bboxes as the prompt for mask decoder.
    float m_minMemoryUpdateConfidence; //!< Mask confidence to update memory bank.
    int m_maskPromptUpdateInterval; //!< Frame interval to use detection bboxes as the prompt for mask decoder. (Memory bank is also used except in the first frame).
    float m_minPromptUpdateConfidence; //!< Minimum confidence of detection bbox to use as the prompt for mask decoder.

private:
    std::string m_trackerConfigPathForPathResolve; //!< When set before read(), segmenterConfigPath and nested paths resolve against this tracker YAML path.
};

/** @brief Manages CUDA-resident temporal features for SAM2 inference. */
class SegmenterCudaMemBank
{
public:
    /** @brief Construct a SegmenterCudaMemBank instance from configuration. */
    SegmenterCudaMemBank();

    /** @brief Release resources owned by the SegmenterCudaMemBank instance. */
    ~SegmenterCudaMemBank();

    /**
     * @brief Initialize memory bank.
     * @param[in] frameInfo Metadata about the source image, such as size and color.
     * @param[in] maxTargetsPerStream Maximum number of targets to track in each stream.
     * @param[in] paramsRt Runtime parameters for the segmenter.
     * @param[in] params Configuration parameters for the segmenter.
     * @param[in] cuStream CUDA stream that runs the GPU work.
     */
    void initialize(
        const SourceFrameInfo &frameInfo, const uint maxTargetsPerStream,
        const SegmenterRunTimeParams &paramsRt, const SegmenterConfigParams &params, cudaStream_t& cuStream
    );

    /**
     * @brief Update memory bank.
     * @param[in] maskMemFeatures_d Mask memory features to append.
     * @param[in] maskMemPosEnc_d Mask memory position encoding to append.
     * @param[in] temporalCode_d Device temporal-code tensor.
     * @param[in] objPtr_d Device object-pointer tensor.
     * @param[in] target Target whose memory bank is updated.
     * @param[in] indInBatch Index of this target in the inference batch.
     * @param[in,out] inferStatus CUDA inference status whose stored-tensor counts are updated.
     */
    void update(
        float* maskMemFeatures_d, float* maskMemPosEnc_d, float* temporalCode_d, float* objPtr_d,
        std::shared_ptr<NvTrackedObject> target, int indInBatch, CudaInferenceStatus& inferStatus
    );

    /**
     * @brief Clear retained memory for one target.
     * @param[in] targetInd Target index.
     */
    void reset(const TrackerIndex targetInd);

    /**
     * @brief Get CUDA memory-bank pointers for a target.
     * @param[in] targetInd Target index.
     */
    std::vector<float*> getCudaMemBank(const TrackerIndex targetInd);

private:
    bool m_isInit; //!< Whether the memory bank is initialized.
    cudaStream_t m_cuStream; //!< Cuda stream.
    SegmenterRunTimeParams m_paramsRt; //!< Runtime parameters.
    SegmenterConfigParams m_params; //!< Config parameters.
    float* m_targetsMaskMemFeatures_d; //!< Mask memory features.
    float* m_targetsMaskMemPosEnc_d; //!< Mask memory position encoding.
    float* m_targetsTemporalCode_d; //!< Temporal code.
    float* m_targetsObjPtr_d; //!< Object pointer.
    float* m_work_0_d; //!< Work tensor 0.

    /**
     * @brief Update tensor bank.
     * @param[in] srcBatch Source batch containing the new tensor.
     * @param[in,out] dst Tensor bank that receives the new tensor.
     * @param[in] numFilled Number of populated tensors in the destination bank.
     * @param[in] maxNumPerTarget Maximum number of tensors stored per target.
     * @param[in] numElementsPerTensor Number of elements in each tensor.
     * @param[in] targetInd Index of the target.
     * @param[in] indInBatch Index of this target in the inference batch.
     * @param[in] keepFirst When the bank is full, keep the first tensor and drop the second oldest.
     */
    void updateTensorBank(
        float* srcBatch, float* dst, int& numFilled, const int maxNumPerTarget, const int numElementsPerTensor,
        const TrackerIndex targetInd, const int indInBatch, const bool keepFirst=false
    );
};

/** @brief Performs target segmentation and maintains temporal model state. */
class Segmenter: public ISegmenter
{
public:
    /**
     * @brief Construct a Segmenter instance from configuration.
     * @param[in] configFilePath Path to the tracker configuration file.
     */
    Segmenter(const std::string &configFilePath);

    /**
     * @brief Construct a Segmenter instance from configuration.
     * @param[in] configFilePath Path to the tracker configuration file.
     * @param[in] modelName Configured model name.
     */
    Segmenter(const std::string &configFilePath, const std::string &modelName);

    /** @brief Release resources owned by the Segmenter instance. */
    virtual ~Segmenter();

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
    bool m_isForLocalization; //!< Whether the inference is for localization.
    SegmenterConfigParams m_params; //!< Config params for Segmenter module.
    SegmenterRunTimeParams m_paramsRt; //!< Runtime parameters.
    float *m_directly_add_no_mem_embed_d; //!< Non-memory embeddings added to image-encoder output.
    //!< These embeddings are used when memory attention does not process the first or only frame.
    cudaStream_t m_cuStream; //!< Cuda stream to run all GPU tasks.
    cublasHandle_t m_cublasHandle; //!< Cublas handle.

    // [TRT inference modules]
    TrtModule m_imageEncoder; //!< Encoder to extract features from each image.
    TrtModule m_maskDecoder; //!< Decoder taking in prompts (points, bboxes, etc.) to create segmentation mask.
    TrtModule m_memoryAttention; //!< Memory attention module.
    TrtModule m_memoryEncoder; //!< Memory encoder module.

    // [Image encoder]
    float *m_inputBuf_d, *m_inputBuf_h; //!< Input patches to encoder.
    float *m_featuresEncoder_d, *m_highResFeaturesEncoder0_d, *m_highResFeaturesEncoder1_d; //!< Image encoder output feature.
    float *m_visionPosEmbed_d, *m_visionFeats_d, *m_highResImgPosEmbed1_d; //!< Image encoder output position embedding.

    // [Memory attention]
    float *m_work_0_d, *m_work_1_d, *m_work_2_d; //!< Scratch buffers for memory attention.
    float *m_memory_obj_ptr_d, *m_memory_maskmem_features_d, *m_memory_d, *m_memory_pos_embed_d, *m_cond_frame_id_diff_d; //!< Memory attention inputs: object pointers, mask memory features, memory, positional embeddings, and condition-frame ID differences.
    float *m_pix_feat_with_mem_d; //!< Attention output features with memory for one target.
    float *m_batchPixFeatWithMem_d; //!< Attention output features with memory for all targets in all streams.

    // [Mask decoder]
    float *m_points_d; //!< Input a list of points (prompt) to decoder.
    float *m_pointLabels_d; //!< Input a list of point types (single point, bounding box corners) to decoder.
    float *m_inputMask_d; //!< Input mask prompt, currently unused and set as 0.
    float *m_inputMaskLabel_d; //!< Input mask label, currently unused and set as 0.
    float *m_targetMaskDecSize_d, *m_targetMaskDecSize_h; //!< Output mask basic resolution.
    float *m_maskForMem_d; //!< Output mask high resolution.
    float *m_obj_ptr_d; //!< Output obj ptrs.
    float *m_outputIou_d; //!< Output IOU score.
    float *m_outputOccLogit_d; //!< Output occlusion logit; <2 means target is occluded.
    //!< A value greater than 6 indicates high confidence that the object is present.
    std::vector<float> m_outputIou_h; //!< Output IOU score on host.
    std::vector<float> m_outputOccLogit_h; //!< Output occlusion logit on host.
    int *m_outputSize_d; //!< Output size.

    // [Memory encoder]
    float *m_maskmem_features_d, *m_maskmem_pos_enc_d, *m_temporal_code_d; //!< Output mask memory features and pos encoding.

    // [Mask post-processing]
    float *m_batchMaskDecSize_d; //!< Mask output for all targets in all streams.
    float *m_targetMaskOriginalSrcSize_d; //!< Mask for each target resized to original streammux size.

    // [Image size]
    Eigen::Vector2f m_scaleMask2Src; //!< Scale from mask output to the source image.
    Eigen::Vector2f m_scaleInput2Original; //!< Scale from decoder output to original frame size decided by streammux.
    NvMotDataTypes::Size m_srcImageSize; //!< Input to low level tracker.
    NvMotDataTypes::Size m_originalSrcImageSize; //!< Input to tracker plugin, usually from streammux.

    // [Target related]
    std::shared_ptr<NvTrackedObjectManager> m_targetManager; //!< Target manager providing target status to update feature gallery.
    std::shared_ptr<ReIDFrameTransformTexture> m_subImageCropper2; //!< Image cropper, may share with visual tracker in the future.
    std::shared_ptr<ReIDCropScaler> m_subImageCropScaler; //!< Image crop scaler.
    //!< Uses VPI crop scaler.
    std::map<NvMOTStreamId, std::vector<std::pair<uint, NvMotDataTypes::Rect2f>>> m_inputRoisInBatch; //!< Input rois to crop.
    std::map<TrackerIndex, std::map<NvMOTFrameNum, SegmentMemoryInfo>> m_condFrameOutputs; //!< Outputs of the conditional frames.
    std::map<TrackerIndex, std::map<NvMOTFrameNum, SegmentMemoryInfo>> m_nonCondFrameOutputs; //!< Outputs of the non-conditional frames.
    std::map<TrackerIndex, InferenceStatus> m_inferStatus; //!< Inference status.
    std::map<TrackerIndex, CudaInferenceStatus> m_cudaInferenceStatus; //!< Cuda inference status.
    std::map<TrackerIndex, std::vector<float>> m_targetScores; //!< Target IOU and occlusion scores.
    SegmenterCudaMemBank m_cudaMemBank; //!< Cuda memory bank.
    std::vector<void *> m_bindingsImgEncoder, m_bindingsMaskDecoder, m_bindingsMemAtten, m_bindingsMemEncoder; //!< TensorRT bindings.

    /**
     * @brief Run segmentation inference for targets in the input batch.
     * @param[in] inputMeta A batch of input frames.
     */
    void doInference(const std::map<NvMOTStreamId, NvMOTFrame *>& inputMeta);

    /**
     * @brief Rescale low-level tracker frames into encoder input.
     * @param[in] inputMeta Input frames keyed by stream ID.
     */
    void extractImagePatchesBatch(const std::map<NvMOTStreamId, NvMOTFrame *> &inputMeta);

    /**
     * @brief Construct memory attention input.
     * @param[in] targetInd Index of the target.
     * @param[in] indInBatch Index of this target in the inference batch.
     */
    void fillMemoryConditionedFeatures(TrackerIndex targetInd, int indInBatch);

    /**
     * @brief Fill memory conditioned features for cuda memory bank.
     * @param[in] targetInd Index of the target.
     * @param[in] indInBatch Index of this target in the inference batch.
     */
    void fillCudaMemoryConditionedFeatures(TrackerIndex targetInd, int indInBatch);

    // [TensorRT inference functions]
    /** @brief Define IO bindings for each TRT engine. */
    void initializeTrtBindings();

    /**
     * @brief Update TRT bindings per stream.
     * @param[in] inputRoiInd Region index within the input batch.
     */
    void updateTrtBindingsPerStream(const uint inputRoiInd);

    /**
     * @brief Batch inference.
     * @param[in] targetList Targets included in the inference operation.
     * @param[in] usePrompt Whether decoder prompt data is enabled.
     * @param[in] outputLabelIndMax Highest decoder output-label index to process.
     * @param[in] frameNum Frame number associated with the inference result.
     * @param[in] inputRoiInd Region index within the input batch.
     */
    void batchInfer(
        const std::map<uint32_t, std::shared_ptr<NvTrackedObject>> &targetList, bool usePrompt,
        int outputLabelIndMax, const NvMOTFrameNum frameNum, const uint inputRoiInd
    );

    /**
     * @brief Run segmentation for one target.
     * @param[in] targetList Targets included in the inference operation.
     * @param[in] usePrompt Whether decoder prompt data is enabled.
     * @param[in] outputLabelIndMax Highest decoder output-label index to process.
     * @param[in] frameNum Frame number associated with the inference result.
     * @param[in] inputRoiInd Region index within the input batch.
     */
    void singleInfer(
        const std::map<uint32_t, std::shared_ptr<NvTrackedObject>> &targetList, bool usePrompt,
        int outputLabelIndMax, const NvMOTFrameNum frameNum, const uint inputRoiInd
    );

    /** @brief Run the image encoder for the current input batch. */
    void imgEncoderInfer();

    /**
     * @brief Run mask-decoder inference for the current target batch.
     * @param[in] usePrompt Whether decoder prompt data is enabled.
     * @param[in] targetList Targets included in the inference operation.
     * @param[out] outputLabelIndMax Index of the output label with the highest IoU.
     * @param[in] inputStreamInd Index of the input stream being processed.
     * @param[in] frameNum Frame number associated with the inference result.
     */
    void maskDecoderInfer(
        bool usePrompt, std::vector<std::shared_ptr<NvTrackedObject>>& targetList,
        int& outputLabelIndMax, const uint inputStreamInd, const NvMOTFrameNum frameNum
    );

    /**
     * @brief Run memory attention for one tracked target.
     * @param[in] target Tracked target to process.
     * @param[in] indInBatch Target index within the inference batch.
     * @param[in] inputStreamInd Index of the input stream being processed.
     */
    void memAttenInfer(const std::shared_ptr<NvTrackedObject> target, const int indInBatch, const int inputStreamInd);

    /**
     * @brief Assemble and run batched memory-attention input.
     * @param[in] targetList Targets included in the inference operation.
     * @param[in] inputStreamInd Index of the input stream being processed.
     */
    void memAttenInferBatchParser(std::vector<std::shared_ptr<NvTrackedObject>>& targetList, const int inputStreamInd);

    /**
     * @brief Run the memory encoder for the current target batch.
     * @param[in] batchSizeActual Number of targets in the current batch.
     */
    void memEncoderInfer(const int batchSizeActual);

    /**
     * @brief Populate mask-decoder labels for points or bounding boxes.
     * @param[in] promptType Prompt representation selected for the decoder.
     */
    void setMaskDecoderPointLabels(int promptType);

    /**
     * @brief Mark a single decoder prompt point as negative.
     * @param[in] pointNumPerTarget Number of prompt points allocated to each target.
     * @param[in] indInBatch Target index within the inference batch.
     */
    void setMaskDecoderSinglePointLabelAsNegative(int pointNumPerTarget, int indInBatch);

    /**
     * @brief Populate mask-decoder prompt coordinates for a target batch.
     * @param[in] usePrompt Whether decoder prompt data is enabled.
     * @param[in] targetList Targets included in the inference operation.
     */
    void setMaskDecoderPoints(const bool usePrompt, std::vector<std::shared_ptr<NvTrackedObject>> targetList);

    /**
     * @brief Convert one inference result into retained target state.
     * @param[in] indInBatch Index of this target in the inference batch.
     * @param[in] target Tracked target to process.
     * @param[in] frameNum Frame number associated with the inference result.
     */
    void postProcessing(
        const int indInBatch, const std::shared_ptr<NvTrackedObject> target, const NvMOTFrameNum frameNum
    );

    /**
     * @brief Record a target in the pending inference batch.
     * @param[in] indInBatch Index of this target in the inference batch.
     * @param[in] target Tracked target to process.
     * @param[in] frameNum Frame number associated with the inference result.
     */
    void updateInferStatus(
        const int indInBatch, const std::shared_ptr<NvTrackedObject> target, const NvMOTFrameNum frameNum
    );

    /** @brief Remove terminated memory bank. */
    void removeInferStatus();

    /**
     * @brief Copy a CUDA buffer into a TensorRT tensor.
     * @param[in] src The source tensor.
     * @param[out] dst Destination tensor to populate.
     * @param[in] numElements Number of tensor elements to copy.
     * @param[in] cuStream CUDA stream that runs the GPU work.
     */
    void copyFromCuda(float* src, TensorPtr& dst, int numElements, cudaStream_t& cuStream);

    /**
     * @brief Check whether a target has no retained inference state.
     * @param[in] targetInd Index of the target.
     * @return True if the infer status is empty, false otherwise.
     */
    inline bool isInferStatusEmpty(TrackerIndex targetInd)
    {
        if (m_params.m_useCudaMemBank)
        {
            if (m_cudaInferenceStatus.find(targetInd) != m_cudaInferenceStatus.end())
            {
                CudaInferenceStatus& targetStatus = m_cudaInferenceStatus.at(targetInd);
                return targetStatus.numStatusUncond == 0;
            }
            else
            {
                return true;
            }
        }
        else
        {
            if (m_inferStatus.find(targetInd) != m_inferStatus.end())
            {
                InferenceStatus& targetStatus = m_inferStatus.at(targetInd);
                return targetStatus.statusCond.empty() && targetStatus.statusUncond.empty();
            }
            else
            {
                return true;
            }
        }
    }

    /**
     * @brief Determine whether segmentation can be skipped for a target.
     * @param[in] target Tracked target to process.
     * @param[in] frameNum Frame number associated with the inference result.
     */
    bool skipInfer(std::shared_ptr<NvTrackedObject> target, const NvMOTFrameNum frameNum);

    /**
     * @brief Check whether a propagated bounding box is reliable.
     * @param[in] target Tracked target to process.
     * @param[in] confidence Target confidence used to validate propagation.
     * @param[in] bboxProg Propagated target bounding box.
     * @param[in] iouThresh Minimum intersection-over-union threshold.
     */
    inline bool isGoodBboxProg(
        const std::shared_ptr<NvTrackedObject> target, const float confidence,
        const NvMotDataTypes::Rect2f bboxProg, const float iouThresh=0.5
    );

    #ifdef VIZ_DEBUG_SEG
        cv::Mat m_imageSumMask; //!< Accumulated masks of all targets in the frame.
        cv::Mat m_imageIDMask; //!< Target-ID masks of all targets in the frame.
        std::map<NvMOTStreamId, std::map<uint, cv::Mat>> m_vizSrcImages; //!< Source images used for visualization.

        /**
         * @brief Update visualization masks for one target.
         * @param[in] target Tracked target to process.
         * @param[in] initMask Whether the target mask is being initialized.
         */
        void updatePerTargetMaskInFrame(const std::shared_ptr<NvTrackedObject> target, bool initMask);

        /**
         * @brief Write accumulated frame masks and overlays.
         * @param[in] videoStreamId Video stream identifier.
         * @param[in] frameNum Frame number associated with the inference result.
         * @param[in] targetList Targets included in the inference operation.
         */
        void savePerFrameMask(
            NvMOTStreamId videoStreamId, NvMOTFrameNum frameNum,
            const std::map<uint32_t, std::shared_ptr<NvTrackedObject>> &targetList
        );
    #endif
};

#endif // SEGMENTERLIB_SEGMENTER_HPP_
