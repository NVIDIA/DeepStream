/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef VISUALTRACKERLIB_DCFTRACKER_HPP_
#define VISUALTRACKERLIB_DCFTRACKER_HPP_

#include <map>
#include <unordered_map>
#include "modules/cuDCFv2/cuDCF.h"
#include "modules/VisualTracker/IVisualTracker.hpp"
#include "modules/NvMultiObjectTracker/NvTrackerParams.hpp"
#include "modules/NvMultiObjectTracker/NvTrackedObject.hpp"
#include "modules/utility/dataTypes.hpp"

// VPI.
#include "vpi/Stream.h"
#include "vpi/Image.h"
#include "vpi/Array.h"
#include "vpi/algo/CropScaler.h"
#include "vpi/algo/DCFTracker.h"

#if defined(__aarch64__)
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include "cudaEGL.h"
#endif

#ifdef VIZ_DEBUG
/** @brief Writes visual tracker debugging output. */
class TrackerVizOutput;
#endif

/** @brief Configuration parameters for the VPI DCF tracker. */
class DCFTrackerConfigParams : public NvTrackerParams
{
public:
	/** @brief Construct the default configuration. */
	DCFTrackerConfigParams();

	/**
	 * @brief Construct configuration parameters from a named module in a file.
	 * @param[in] configFilePath The path to the configuration file.
	 * @param[in] moduleName The configuration module name.
	 */
	DCFTrackerConfigParams(const std::string &configFilePath, const std::string moduleName);

	/**
	 * @brief Construct configuration parameters from a YAML node.
	 * @param[in] fn Parsed YAML structure containing the configuration.
	 */
	DCFTrackerConfigParams(const YAML::Node &fn);

	/**
	 * @brief Read visual tracker parameters from a YAML node.
	 * @param[in] fn Parsed YAML structure of configuration parameters.
	 */
	void read(const YAML::Node &fn) override;

	int m_showGpuTiming; //!< Controls whether GPU timing is displayed.
	int m_debugVerboseLevel; //!< Debug message verbose mode level.
	int m_debugVerboseLevel_gpumemory; //!< GPU memory debug verbosity level.
	int m_debugVerboseLevel_cufft; //!< Debug verbosity level for cuFFT.

	// [Visual feature types]
	bool m_useColorNames; //!< Enables the ColorNames feature.
	bool m_useHog; //!< Enables the HOG feature.
	bool m_useGray; //!< Enables the grayscale pixel-intensity feature.
	bool m_useRgb; //!< Enables the RGB feature.

	// [VPI]
	bool m_useHighPrecisionFeature; //!< Enables high-precision 16-bit VPI output instead of low-precision 8-bit output.
	VPIBackend m_vpiBackend4CropnScale; //!< VPI backend for the crop scaler.
	VPIBackend m_vpiBackend4DcfTracker; //!< VPI backend for the DCF tracker.

	// [Feature extraction]
	float m_featureImgSize; //!< Maximum template length in one dimension.
	//!< The maximum template area is m_featureImgSize squared, and the actual feature size is divided by m_hogCellSize.
	float m_featureFocusOffsetFactor_y; //!< Vertical Hanning-window center offset relative to the feature height.
	float m_modelMaskingWindowFactor; //!< Masking-window factor used for focused filter learning.
	int m_hogCellSize; //!< Cell size of HOG feature.

	// [DCF][Core]
	float m_filterLr; //!< Learning rate for DCF filter in exponential moving average.
	float m_filterChannelWeightsLr; //!< Learning rate for DCF filter channel weights in exponential moving average.
	float m_gaussianSigma; //!< Sigma for Gaussian for desired response.
	float m_lambda; //!< Regularization factor used in DCF filter creation.
	float m_peakCorrLr; //!< Learning rate for peak correlation response in exponential moving average.

	// [FFT]
	int m_cufftHandleQuantFactor; //!< Clustering size for cuFFT plans.
	//!< Each plan covers a range of input sizes, and this sets how fine that range is.

	// [Viz Debug]
	bool m_generateTrackerVisualization; //!< Enables tracker visualization generation.
	bool m_vizWriteSrcImgPatchesAndCorrResp; //!< Write the cropped/scaled image patches and correlation response.
	bool m_vizWriteBirdsEyeView; //!< Write the bird's-eye view.
	bool m_vizWriteAllTrajectoriesOnEveryFrame; //!< Write the whole frame with trajectories of all targets in the scene.
	bool m_vizWriteEachCompletedTrajectory; //!< Write each completed trajectory.
	int m_vizJpegImageQuality; //!< JPEG output quality in the range [1, 100].
	std::string m_vizOutDirectory; //!< The directory to write the outputs.
	int m_vizDebugStreamId; //!< Video stream ID selected for visualization debugging.
	//!< A negative ID such as -1 selects all streams.

	// [Custom parameters for debugging]
	int m_debugParamInt0; //!< First integer debugging parameter.
	int m_debugParamInt1; //!< Second integer debugging parameter.
	int m_debugParamInt2; //!< Third integer debugging parameter.
	float m_debugParamFloat0; //!< First floating-point debugging parameter.
	float m_debugParamFloat1; //!< Second floating-point debugging parameter.
	float m_debugParamFloat2; //!< Third floating-point debugging parameter.

	// [Configuration parameters]
	VISUAL_TRACKER_TYPES m_visualTrackerType; //!< The type of visual tracker.
	bool m_isInit = false; //!< Indicates whether the parameters have been initialized.
	//!< Prevents changing static params in dynamic updates.
};

/** @brief Stores runtime dimensions and processing state for the VPI DCF tracker. */
struct DCFTrackerRunTimeParams
{
	uint maxStreams; //!< Maximum number of supported video streams.
	uint maxTargetsToTrack; //!< Maximum number of targets tracked across all streams.
	uint maxDetectorBboxesPerFrame; //!< Maximum number of detector bounding boxes per frame.
	uint m_maxBufSurfAddrSize; //!< Maximum number of stored EGL-mapped addresses.
	NvMotDataTypes::Size originalSrcImageSize; //!< The size of the original video source image.
	NvMotDataTypes::Size srcImageSize; //!< The size of the source image.
	size_t srcImagePitch; //!< Memory pitch of the source frame.
	NvBufSurfaceColorFormat colorFormat; //!< Input source image color format.
	NvMotDataTypes::Size templateSize; //!< Size of the rescaled search-region template.
	NvMotDataTypes::Size singleFeatureMatSize; //!< Dimensions of a single feature matrix processed on the GPU.
	int detectorInputInterval; //!< Interval between external detector inputs.
	std::map<NvMOTStreamId, uint32_t> isDetectionDonePerStream; //!< Indicates whether inference completed for each stream.
	bool skipBatchUpdate; //!< Indicates whether to skip the update for the whole batch.
	int singleFeatureChannelElementSize; //!< Number of elements in a single feature matrix.
	int detCorrRespSearchWindowSize; //!< Correlation-response search radius used to tolerate detector jitter.
	//!< The resulting search-window width and height are twice this value plus one.
};

/** @brief Tracks objects with the VPI discriminative correlation filter implementation. */
class DCFTracker : public IVisualTracker
{
public:

	/**
	 * @brief Construct from a configuration file path.
	 * @param[in] configFilePath The path to the configuration file.
	 */
	DCFTracker(const std::string &configFilePath);

	/** @brief Release VPI DCF tracker resources. */
	virtual ~DCFTracker();

	/**
	 * @brief Initialize variables and allocate memories.
	 * @param[in] frameInfo Metadata about the source image (e.g., size, color).
	 * @param[in] maxTargetsToTrack The maximum number of targets to track.
	 * @param[in] maxDetectorBboxesPerFrame The maximum number of detector bounding boxes per frame.
	 * @param[in] targetManager The target manager through which all the targets can be accessed.
	 */
	void initialize(
		const SourceFrameInfo &frameInfo, const uint maxTargetsToTrack,
		const uint maxDetectorBboxesPerFrame, const std::shared_ptr<NvTrackedObjectManager> &targetManager
	) override;

	/**
	 * @brief Populate the visual tracker's capability query.
	 * @param[out] paramsQuery The output query parameters.
	 */
	void queryParams(NvMOTQuery &paramsQuery);

	/**
	 * @brief Localize tracked targets and evaluate nearby detections across an input batch.
	 * @param[in] inputMeta A batch of input frames.
	 * @param[in] detectedObjects The list of newly-detected objects.
	 */
	void localize(
		const std::map<NvMOTStreamId, NvMOTFrame *> inputMeta,
		const std::map<NvMOTStreamId, std::deque<DetectedObjInfo>> &detectedObjects
	) override;

	/**
	 * @brief Update each target's correlation-filter model from an input batch.
	 * @param[in] inputMeta A batch of input frames.
	 */
	void update(const std::map<NvMOTStreamId, NvMOTFrame *> inputMeta) override;

	/** @brief Report whether tracker initialization completed. */
	bool isInitialized() override { return m_isInit; }

	/** @brief Return the configured visual tracker type. */
	const VISUAL_TRACKER_TYPES getVisualTrackerType() override { return m_params.m_visualTrackerType; }

	/**
	 * @brief Update any Dynamic Params as provided during runtime by the user.
	 * @param[in] fn Parsed YAML structure of config to update.
	 */
	void updateParams(const YAML::Node &fn) { m_params.read(fn); }

protected:

	bool m_isInit; //!< Indicates whether tracker initialization completed.
	DCFTrackerConfigParams m_params; //!< Config params for visual tracker module.
	DCFTrackerRunTimeParams m_paramsRt; //!< Runtime params for visual tracker module.
	std::shared_ptr<NvTrackedObjectManager> m_targetManager; //!< Target manager.
	std::map<NvMOTStreamId, std::vector<std::pair<uint, NvMotDataTypes::Rect2f>>> m_featureRoisInBatch; //!< Trackers' search regions as ROI for feature extraction.
	//!< Supposed to be updated after data association.
	//!< Maps each videoStreamID to a vector of trackerInd and bbox pairs.

	// [cuFFT operations]
	cudaStream_t m_cudaStream; //!< CUDA stream used for tracker operations.

	// [CUDA memory buffers for pixel processing]
	float *m_corrResponses_d; //!< The correlation response maps for targets (a Batch CUDA memory buffer).
	float *m_channelWeights; //!< The adaptive filter weights for the channels on device (a Batch CUDA memory buffer).

	// [CUDA and host memory buffers for miscellaneous data]
	bool *m_trackerDoProcessing_h, *m_trackerDoProcessing_d; //!< True when this tracker is updated for this batch.
	uint *m_detectorBboxes_size_h, *m_detectorBboxes_size_d; //!< The number of detector bbox candidate for each target.
	uint32_t *m_trackerAge_h, *m_trackerAge_d; //!< Tracker age.
	detectorBbox_t *m_detectorBboxes_h, *m_detectorBboxes_d; //!< The detector bbox center points to be looked up at each tracker's correlation response.
	float *m_confidenceAtDetectedPt_h, *m_confidenceAtDetectedPt_d; //!< Tracking confidence at each detector bbox center.
	float *m_corrResponseAtDetectedPt_h, *m_corrResponseAtDetectedPt_d; //!< Correlation response at each detector bbox center.
	float *m_maxCorrResponse_h, *m_maxCorrResponse_d; //!< The max correlation response for targets (a Batch CUDA memory buffer).
	float *m_maxCorrResponse_ma_d, *m_maxCorrResponse_ma_h; //!< The moving average of the max corr resp.
	float *m_trackingConfidence_d, *m_trackingConfidence_h; //!< The tracking confidence derived from the instant and the moving average of the max corr resp.
	float *m_maskingWindow4Src; //!< Masking window for input image feature (a CUDA memory buffer).
	float *m_maskingWindow4Model; //!< Masking window for the model for filter learning (a CUDA memory buffer).
	int m_featureChannelSize; //!< Number of channels in a feature vector.
	int m_cudaKernelMaxBatchSize; //!< Maximum CUDA batch size across feature channels and target trackers.

	// [VPI library interactions]
	VPIDCFTrackerCreationParams m_vpiCreateParams; //!< VPI DCF Tracker creation params.
	VPIDCFTrackerParams m_vpiRuntimeParams; //!< VPI DCF Tracker runtime params.
	VPIPayload m_vpiDcfPayload; //!< VPI payload for DCF Tracker.
	VPIPayload m_vpiCropPayload; //!< VPI payload for Crop scaler.
	VPIStream m_vpiStream; //!< VPI stream used for all VPI operations.
	int m_vpiMemFlags = VPI_BACKEND_CUDA | VPI_RESTRICT_MEM_USAGE; //!< Flags used for creating VPI Image/Array wrappers.

	// [Input frame sequences]
	VPIImage *m_vpiInputFrameSequence; //!< Sequence of VPIImage pointers to input frames.
	VPIImageData *m_vpiInputFrameDataList; //!< Sequence of VPIImageData pointers to input frames.
	std::vector<bool> m_isFrameCreated; //!< List indicating whether each frame wrapper has been created.
	//!< An entry is true when vpiImageCreateWrapper created the frame at that sequence index.
	int32_t *m_enabledSequences; //!< An array marking the entries in input frame list as valid(1) or invalid(0).

	// [Tracked objects]
	std::vector<VPIDCFTrackedBoundingBox> m_vpiDCFBboxes; //!< List of tracked objects.
	//!< Its reserved size is maxTargetsToTrack.
	VPIArrayData m_vpiDCFBboxesData; //!< VPIArrayData for m_vpiDCFBboxes.
	VPIArray m_vpiDCFBboxesArray; //!< VPIArray for m_vpiDCFBboxes.
	int32_t m_vpiBboxesArraySize; //!< Number of active entries in m_vpiDCFBboxes used for dynamic compute adjustment.
	std::vector<VPIDCFTrackedBoundingBox> m_vpiDCFBboxesLocalized; //!< List of localized objects.
	//!< Its reserved size is maxTargetsToTrack.
	VPIArrayData m_vpiDCFBboxesLocalizedData; //!< VPIArrayData for m_vpiDCFBboxesLocalized.
	VPIArray m_vpiDCFBboxesLocalizedArray; //!< VPIArray for m_vpiDCFBboxesLocalized.

	// [Cropped and scaled object patches]
	uint8_t *m_vpiObjPatches_d; //!< Int8 device memory for VPI output that contains cropped and scaled object patches.
	#if defined (VIZ_DEBUG)
		uint8_t *m_vpiObjPatches_h; //!< Host memory corresponding to m_vpiObjPatches_d.
	#endif
	int m_vpiObjPatchesNumChannels; //!< Object patches are of format VPI_IMAGE_FORMAT_RGBA8 for CUDA backend.
	//!< And of format VPI_IMAGE_FORMAT_RGB8p for PVA backend.
	VPIImageData m_vpiObjPatchesData; //!< VPIImageData corresponding to m_vpiObjPatches_d.
	VPIImage m_vpiObjPatches; //!< Image of cropped and scaled object patches.
	//!< The VPI crop scaler generates this image.

	// [VPI images and arrays wrapping CUDA memory]
	VPIImageData m_vpiMaskingWindow4SrcData; //!< VPIImageData for m_maskingWindow4Src.
	VPIImage m_vpiMaskingWindow4SrcImage; //!< VPIImage for m_maskingWindow4Src.
	VPIImageData m_vpiMaskingWindow4ModelData; //!< VPIImageData for m_maskingWindow4Model.
	VPIImage m_vpiMaskingWindow4ModelImage; //!< VPIImage for m_maskingWindow4Model.
	VPIImageData m_vpiCorrResponsesData; //!< VPIImageData for m_corrResponses_d.
	VPIImage m_vpiCorrResponsesImage; //!< VPIImage for m_corrResponses_d.
	VPIArrayData m_vpiMaxCorrResponseArrayData; //!< VPIArrayData for m_maxCorrResponse_d.
	VPIArray m_vpiMaxCorrResponseArray; //!< VPIArray for m_maxCorrResponse_d.
	VPIArrayData m_vpiChannelWeightsArrayData; //!< VPIArrayData for m_channelWeights.
	VPIArray m_vpiChannelWeightsArray; //!< VPIArray for m_channelWeights.

	// [CUDA pointers from NvBufSurface objects]
	int m_integrated_gpu; //!< Indicates whether the tracker runs on an integrated GPU.
	int m_deviceId; //!< GPU device ID.
	uint8_t *m_surfCudaMemPtr[NVBUF_MAX_PLANES]; //!< Pointers to cuda memories of all the planes in a surface.
	#if defined(__aarch64__)

		// [Cached EGL mappings]
		std::unordered_map<const NvBufSurfaceParams *, CUgraphicsResource> m_graphicsResourceAddrMap; //!< Map storing (BufSurfaceParams address, CUgraphicsResource) pairs.
		//!< The cached mappings avoid remapping each incoming NvBufSurface.
		std::unordered_map<const NvBufSurfaceParams *, CUeglFrame> m_eglFrameAddrMap; //!< Map storing (BufSurfaceParams address, CUeglFrame) pairs.
	#endif

	cudaEvent_t m_eventStart; //!< Event start point for GPU timers.
	cudaEvent_t m_eventStop; //!< Event stop point for GPU timers.

	/**
	 * @brief Check the input and find the detection interval.
	 * @param[in] inputMeta A batch of input frames.
	 */
	void checkInputOutput(const std::map<NvMOTStreamId, NvMOTFrame*> &inputMeta);

	/**
	 * @brief Populate the VPI input frame sequence from MOT frames.
	 * @param[in] inputMeta A batch of input frames.
	 */
	void setVpiInputFrameSequence(const std::map<NvMOTStreamId, NvMOTFrame*> &inputMeta);

	/** @brief Pre-allocate required cuda memories on GPU. */
	void allocateCudaMemories();

	/**
	 * @brief Duplicate one host matrix into a batched float buffer and copy it to the device.
	 * @param[in] srcMat The source matrix for a single tracker.
	 * @param[out] devMem The device buffer that receives the duplicated matrix.
	 * @param[in] batchSize The number of copies of the source matrix.
	 * @param[in,out] workarea_host The host workspace filled with the duplicated matrix before the device copy.
	 */
	void makeBatchAndSendFloatMatToDevice(
		const NvMotDataTypes::RowMajorMatrixXf &srcMat, void *devMem, uint batchSize, float *workarea_host
	);

	/**
	 * @brief Update target states from their correlation responses.
	 * @param[in] inputMeta A batch of input frames.
	 */
	void updateLocalizationResults(const std::map<NvMOTStreamId, NvMOTFrame*> &inputMeta);

	/**
	 * @brief Map a source-image point into a tracker's correlation-response coordinates.
	 * @param[in] roi The source-frame region that defines the relative coordinate system.
	 * @param[in] pt_img The point in source-image coordinates.
	 * @return The corresponding point in the tracker's correlation-response matrix.
	 */
	Eigen::Vector2f convertSrcImgCoordToCorrRespCoord(
		const NvMotDataTypes::Rect2f &roi, const Eigen::Vector2f pt_img
	) const;

	/** @brief Given the cropped and scaled object patches and the learned filters, compute the correlation response. */
	void cudaCalculateResponse(void);

	/**
	 * @brief Retrieve each tracker's correlation response at nearby detector bounding boxes.
	 * @param[in] detectedObjects The detected objects grouped by video stream.
	 */
	void getDetectorBboxCorrelationResponses(
		const std::map<NvMOTStreamId, std::deque<DetectedObjInfo>> &detectedObjects
	);

	/** @brief Create and update the DCF filter. */
	void cudaFilterUpdate();

	/**
	 * @brief Select a cuFFT plan large enough for the requested target count.
	 * @param[in] num_target The number of targets to process.
	 * @return The identifier of the selected cuFFT plan.
	 */
	inline int getCufftPlanId(int num_target);

	/**
	 * @brief Compute the scale that maps a search region to the fixed feature-image size.
	 * @param[in] roi Search region whose area determines the scale.
	 * @return Factor that scales the search region to the feature image size.
	 */
	float getTemplateRescaleRatio(const NvMotDataTypes::Rect2f &roi) const;

	/**
	 * @brief Crop and scale object regions for localization or model update.
	 * @param[in] inputMeta A batch of input frames.
	 * @param[in] isForLocalization Whether the operation is for localization rather than model update.
	 */
	void performCropAndScale(const std::map<NvMOTStreamId, NvMOTFrame *> &inputMeta, const bool isForLocalization);

	/**
	 * @brief Retrieve CUDA memory pointers for all planes of an NvBufSurface.
	 * @param[in] baseSurface The surface parameters whose plane pointers are retrieved.
	 */
	inline void getCudaMemPointers(NvBufSurfaceParams *baseSurface);

	#ifdef VIZ_DEBUG
		// [Source images in host memory]
		std::map<NvMOTStreamId, cv::Mat> m_srcFramesAtHost; //!< Input source frames.
		uint8_t *m_idata_host; //!< Host memory for input.

		/** @brief Get patches to host for visualization debugging. */
		std::map<NvMOTStreamId, std::map<uint, cv::Mat > > getImagePatches2Host_OCV_batch();

		std::shared_ptr<TrackerVizOutput> m_vizOutput; //!< Visualization output writer.
		float *m_batchCudaArray_host; //!< Temporary batch memory buffer in host memory.
		//!< GPU-processed matrices are downloaded here before host-side de-batching.
		std::map<uint, std::pair<NvMOTStreamId, uint> > m_targetNumInd2RoisInBatchMapper; //!< Map the target index to the corresponding video stream ID and the bbox ind {targetInd, { videoStreamID, bboxInd} }.
		float2 *m_targetDisplacement2D_h; //!< The target displacement in correlation response.

		/**
		 * @brief Fetch batched device matrices as one host matrix per object.
		 * @param[in] deviceData The device buffer containing batched matrices.
		 */
		std::vector<cv::Mat> fetchBatchDataToHost_perObj(const float *deviceData);

		/** @brief Return source images copied into host memory. */
		std::map<NvMOTStreamId, cv::Mat> getSrcImagesAtHost();

		/**
		 * @brief Copy input frames and tracker data from device to host memory.
		 * @param[in] inputMeta A batch of input frames.
		 */
		void uploadDeviceData2HostMemories(const std::map<NvMOTStreamId, NvMOTFrame*> &inputMeta);

		/** @brief Return the cached source frames in host memory. */
		const std::map<NvMOTStreamId, cv::Mat>& getSrcFramesAtHost() const { return m_srcFramesAtHost; }

		friend class TrackerVizOutput;
	#endif
};

#endif // VISUALTRACKERLIB_DCFTRACKER_HPP_
