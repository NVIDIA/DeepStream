/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "customlib_impl.h"
#include "utils/dump_utils.h"
#include "utils/tensor_init.h"
#include "preprocess/preprocessor.h"
#include "postprocess/postprocessor.h"
#include <set>

#define GOTO_ERROR(msg) do { errorMessage = (msg); goto error; } while(0)

#define GOTO_SKIP(msg) do { skipMessage = (msg); goto skip; } while(0)

// Helper function to check if file exists
static bool fileExists(const std::string& path) {
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

/* This quark is required to identify NvDsMeta when iterating through
 * the buffer metadatas */
GQuark _dsmeta_quark = g_quark_from_static_string (NVDS_META_STRING);

// Create Custom Algorithm / Library Context
extern "C" IDSCustomLibrary *CreateCustomAlgoCtx(DSCustom_CreateParams *params)
{
  return new SampleAlgorithm();
}

// Initialize the custom library
void SampleAlgorithm::Initialize()
{
  // Load configuration from config.yaml
  m_configParser = std::make_unique<ConfigParser>("/opt/nvidia/deepstream/deepstream/sources/sparse4d/configs/config.yaml");
  if (m_configParser->isLoaded()) {
    m_ConfigProperties = m_configParser->getProperties();
    m_configParser->displayProperties();
  } else {
    m_Logger->error("No configuration found at /opt/nvidia/deepstream/deepstream/sources/sparse4d/configs/config.yaml");
  }

  // Check if ONNX file exists
  if (!fileExists(m_ConfigProperties.onnx_file)) {
      m_Logger->error("ONNX file does not exist: " + m_ConfigProperties.onnx_file);
      exit(1);
  }

  // Check if engine is empty use default
  if (m_ConfigProperties.engine_file.empty()) {
    m_ConfigProperties.engine_file = "root/sparse4d/model.engine";
  }

  // Check if calibration file exists
  if (m_ConfigProperties.calib_file_path.empty()) {
    m_Logger->error("Calibration file does not exist: " + m_ConfigProperties.calib_file_path);
    exit(1);
  }

  // Check if labels file exists
  if (!fileExists(m_ConfigProperties.labels_file)) {
    m_Logger->error("Labels file does not exist: " + m_ConfigProperties.labels_file);
    exit(1);
  }

  // Check if anchor file exists
  if (!fileExists(m_ConfigProperties.anchor)) {
    m_Logger->error("Anchor file does not exist: " + m_ConfigProperties.anchor);
    exit(1);
  }

  // If interval is negative, skip all frames
  if (m_ConfigProperties.interval < 0) {
    m_Logger->error("Interval is negative, skipping all frames");
    exit(1);
  }

  // Parse calibration file to get lidar-to-image tensor and sensor names
  std::tuple<std::vector<Matrix4f>, std::vector<std::string>, std::vector<float>> pytorch_lidar2img_tensor_raw_init =
      parseCalibFile(m_ConfigProperties.calib_file_path, m_ConfigProperties.bev_group_name,
                    m_ConfigProperties.calib_mode, m_ConfigProperties.use_camera_groups, m_ConfigProperties.recentering);
  std::vector<Matrix4f> calibMatrices = std::get<0>(pytorch_lidar2img_tensor_raw_init);
  m_SensorNames = std::get<1>(pytorch_lidar2img_tensor_raw_init);
  std::vector<float> groupOrigin = std::get<2>(pytorch_lidar2img_tensor_raw_init);

  std::tuple<std::vector<Matrix4f>, std::vector<std::string>, std::vector<float>> org_pytorch_lidar2img_tensor_raw_init =
      parseCalibFile(m_ConfigProperties.calib_file_path, m_ConfigProperties.bev_group_name,
                    m_ConfigProperties.calib_mode, m_ConfigProperties.use_camera_groups, false);
  std::vector<Matrix4f> orgCalibMatrices = std::get<0>(org_pytorch_lidar2img_tensor_raw_init);

  // Print information about the BEV scene
  std::string separator = "+ " + std::string(146, '-') + " +";
  std::cout << separator << std::endl;
  std::cout << "| " << std::string(73 - 10, ' ') << "Calibration File Information" << std::string(73 - 18, ' ') << " |" << std::endl;
  std::cout << separator << std::endl;
  std::cout << "Calib File Path: " << m_ConfigProperties.calib_file_path << std::endl;
  std::cout << "BEV Group Name => " << m_ConfigProperties.bev_group_name << std::endl;
  std::cout << "Group Origin => " << groupOrigin[0] << ", " << groupOrigin[1] << std::endl;
  std::cout << "Recentering => " << m_ConfigProperties.recentering << std::endl;
  std::cout << "Total Sensors => " << m_SensorNames.size() << std::endl;

  std::cout << "Sensors List => [";
  bool first = true;
  for (const auto &cam : m_SensorNames) {
      if (!first) std::cout << ", ";
      std::cout << cam;
      first = false;
  }
  std::cout << "]" << std::endl;
  std::cout << separator << std::endl;



  // Create AugConfig map from the augmentation configuration in m_ConfigProperties
  AugConfig augConfig;
  augConfig["resize"] = std::variant<float, std::vector<int>, bool>(m_ConfigProperties.aug_config.resize);
  augConfig["resize_dims"] = std::variant<float, std::vector<int>, bool>(m_ConfigProperties.aug_config.resize_dims);
  augConfig["crop"] = std::variant<float, std::vector<int>, bool>(m_ConfigProperties.aug_config.crop);
  augConfig["flip"] = std::variant<float, std::vector<int>, bool>(m_ConfigProperties.aug_config.flip);
  augConfig["rotate"] = std::variant<float, std::vector<int>, bool>(m_ConfigProperties.aug_config.rotate);
  augConfig["rotate_3d"] = std::variant<float, std::vector<int>, bool>(m_ConfigProperties.aug_config.rotate_3d);

  // Transform the calibration matrices using the augmentation configuration
  std::vector<Matrix4f> transformedCalibMatrices = transformLidarInfo(calibMatrices, augConfig);

  // Set logger properties based on config
  m_Logger->setLogLevel(m_ConfigProperties.log_level);

  // File existence alone cannot identify a plan after a TRT/model/plugin upgrade.
  m_Logger->processStart("Build TensorRT Engine", -1, true);
#if NV_TENSORRT_MAJOR >= 11
  {
    bool success = prepareSparse4dEngine(
      m_ConfigProperties.onnx_file,
      m_ConfigProperties.engine_file,
      m_ConfigProperties.batch_size,
      m_ConfigProperties.num_sensors,
      m_ConfigProperties.preprocessed_width,
      m_ConfigProperties.preprocessed_height,
      m_ConfigProperties.gpu_id,
      m_ConfigProperties.enable_fp16,
      m_ConfigProperties.force_engine_rebuild,
      m_Logger
    );
#else
  struct stat buffer;
  if (stat(m_ConfigProperties.engine_file.c_str(), &buffer) != 0 || m_ConfigProperties.force_engine_rebuild) {
    bool success = buildEngineFromOnnx(
      m_ConfigProperties.onnx_file, m_ConfigProperties.engine_file,
      m_ConfigProperties.batch_size, m_ConfigProperties.num_sensors,
      m_ConfigProperties.enable_fp16, m_Logger);
#endif
    if (!success) {
      m_Logger->processComplete("Build TensorRT Engine", -1, false, true);
      m_Logger->error("Failed to build TensorRT engine. Exiting...");
      exit(1);
    } else {
      m_Logger->processComplete("Build TensorRT Engine", -1, true, true);
    }
  }
#if NV_TENSORRT_MAJOR < 11
  else {
    m_Logger->processComplete("Build TensorRT Engine", -1, true, true);
    m_Logger->info("Using existing TensorRT engine file: " + m_ConfigProperties.engine_file);
  }
#endif

  // Deserialize the engine file and prepare for inference
  m_Logger->processStart("Deserialize TensorRT Engine", -1, true);

  // Initialize TensorRT engine
  m_TensorRTEngine = std::make_unique<TensorRTEngine>();

  // Configure the TensorRT engine with all properties
  m_TensorRTEngine->configure(m_ConfigProperties);

  // Deserialize the engine after plugin loading
  if (!m_TensorRTEngine->deserializeEngine(m_ConfigProperties.engine_file, m_TensorCache, m_ConfigProperties.batch_size, m_ConfigProperties.num_sensors, m_Logger)) {
    m_Logger->processComplete("Deserialize TensorRT Engine", -1, false, true);
    m_Logger->error("Failed to deserialize TensorRT engine. Exiting...");
    m_TensorRTEngine.reset();
    exit(1);
  } else {
    m_Logger->processComplete("Deserialize TensorRT Engine", -1, true, true);
  }

  // Read tensor from file [Fixed input tensor]
  if (!m_ConfigProperties.fixed_input_tensor_path.empty()) {
    m_Logger->info("Using fixed input tensor from file: " + m_ConfigProperties.fixed_input_tensor_path);
    Tensor* fixed_input_tensor = read_tensor_from_file( m_ConfigProperties.fixed_input_tensor_path, m_ConfigProperties.gpu_id);
    m_TensorRTEngine->updateBindingPointer("img", fixed_input_tensor->buffer);
    m_TensorCache->updateTensorBuffer("img", fixed_input_tensor->buffer);
  }

  // Store projection matrices by camera name for partial batch support
  if (m_SensorNames.size() != orgCalibMatrices.size()) {
    m_Logger->error("Number of sensor names (" + std::to_string(m_SensorNames.size()) + 
                  ") does not match number of calibration matrices (" + 
                  std::to_string(orgCalibMatrices.size()) + ")");
    exit(1);
  }

  // Populate the camera projection matrices map
  for (size_t i = 0; i < m_SensorNames.size(); i++) {
    m_CameraProjectionMatrices[m_SensorNames[i]] = orgCalibMatrices[i];
    m_CameraTransformedProjectionMatrices[m_SensorNames[i]] = transformedCalibMatrices[i];
  }

  // Initialize the pre-allocated tensors using tensor_utils functions to fill the required data in input tensors
  if (m_TensorRTEngine && m_TensorRTEngine->isInitialized()) {
    initOrgProjectionMatTensor(m_TensorCache, m_ConfigProperties.batch_size, m_ConfigProperties.num_sensors, orgCalibMatrices, -1);
    initProjectionMatTensor(m_TensorCache, m_ConfigProperties.batch_size, m_ConfigProperties.num_sensors, transformedCalibMatrices);
    initImageWHTensor(m_TensorCache, m_ConfigProperties.preprocessed_width, m_ConfigProperties.preprocessed_height);
    initInputCachedAnchorTensor(m_TensorCache);
    initInputCachedFeatureTensor(m_TensorCache);
    intiPrevExistsTensor(m_TensorCache, 0.0f);
    initIntervalMaskTensor(m_TensorCache, 0.0f);
    initGroupOriginTensor(m_TensorCache, groupOrigin);
    m_Logger->info("Initialized input tensors with default values using pre-allocated binding memory");

  }
}

// Set Init Parameters
bool SampleAlgorithm::SetInitParams(DSCustom_CreateParams *params)
{
  DSCustomLibraryBase::SetInitParams(params);

  BufferPoolConfig pool_config = {0};
  GstStructure *s1 = NULL;
  GstCapsFeatures *feature;
  GstStructure *config = NULL;

  s1 = gst_caps_get_structure(m_inCaps, 0);

  feature = gst_caps_get_features (m_outCaps, 0);
  if (gst_caps_features_contains (feature, GST_CAPS_FEATURE_MEMORY_NVMM))
  {
      hw_caps = true;
  }
  else
  {
      hw_caps = false;
  }

   // Create buffer pool
  // Set the params properly based on this custom library requirement
  pool_config.cuda_mem_type = NVBUF_MEM_DEFAULT;
  if (hw_caps == false)
  {
      //pool_config.cuda_mem_type = NVBUF_MEM_SYSTEM;
      m_swbufpool = gst_buffer_pool_new ();
      config = gst_buffer_pool_get_config (m_swbufpool);
      gfloat scale_factor = 1.5;
      if (m_outVideoFmt == GST_VIDEO_FORMAT_RGBA)
          scale_factor = 4;
      swbuffersize = m_outVideoInfo.width * m_outVideoInfo.height * scale_factor;
      gst_buffer_pool_config_set_params (config, m_outCaps, swbuffersize, 4, 4);
      if (!gst_buffer_pool_set_config (m_swbufpool, config))
      {
          printf ("FAILED TO SET CONFIG SW BUFFER POOL\n");
          return false;
      }
      gboolean is_active = gst_buffer_pool_set_active (m_swbufpool, TRUE);
      if (!is_active)
      {
          GST_WARNING (" Failed to allocate the buffers inside the output pool");
          return false;
      }
      else
      {
          GST_DEBUG_OBJECT (m_element, " Output SW buffer pool (%p) successfully created", m_swbufpool);
      }
  }
  else
  {
      pool_config.gpu_id = params->m_gpuId;
      pool_config.max_buffers = 4;
      gst_structure_get_int (s1, "batch-size", &pool_config.batch_size);

      if (pool_config.batch_size == 0) {
          // If this component is placed before mux, batch-size value is not set
          // In this case make batch_size = 1
          pool_config.batch_size = 1;
      }

      m_dsBufferPool = CreateBufferPool (&pool_config, m_outCaps);
      if (!m_dsBufferPool) {
          throw std::runtime_error("Custom Buffer Pool Creation failed");
      }
      m_ConfigParams.compute_mode = NvBufSurfTransformCompute_GPU;
      m_ConfigParams.gpu_id = params->m_gpuId;
      m_ConfigParams.cuda_stream = params->m_cudaStream;
  }

  return true;
}

// Return Compatible Output Caps based on input caps
GstCaps* SampleAlgorithm::GetCompatibleCaps (GstPadDirection direction,
        GstCaps* in_caps, GstCaps* othercaps)
{
  GstStructure *s1, *s2;
  gint width = 0, height = 0;
  gint i = 0, num = 0, denom = 0;
  const gchar *inputFmt = NULL;
  const gchar *outputFmt = NULL;
  gint w = 0;
  gint h = 0;

  GST_DEBUG_OBJECT (m_element, "\n----------\ndirection = %d (1=Src, 2=Sink) -> %s:\nCAPS = %s\n",
   direction, __func__, gst_caps_to_string(in_caps));
  GST_DEBUG_OBJECT (m_element, "%s : OTHERCAPS = %s\n", __func__, gst_caps_to_string(othercaps));

  GST_DEBUG_OBJECT (m_element, "trying to fixate othercaps %" GST_PTR_FORMAT
      " based on caps %" GST_PTR_FORMAT, othercaps, in_caps);

  othercaps = gst_caps_truncate(othercaps);
  othercaps = gst_caps_make_writable(othercaps);


  int num_output_caps = gst_caps_get_size (othercaps);

  s1 = gst_caps_get_structure(in_caps, 0);
  for (i=0; i<num_output_caps; i++)
  {
    s2 = gst_caps_get_structure(othercaps, i);
    inputFmt = gst_structure_get_string (s1, "format");
    outputFmt = gst_structure_get_string (s2, "format");

    GST_DEBUG_OBJECT (m_element, "InputFMT = %s \n\n", inputFmt);

    //Set these output caps
    gst_structure_get_int (s1, "width", &width);
    gst_structure_get_int (s1, "height", &height);

    /* otherwise the dimension of the output heatmap needs to be fixated */

    // Here change the width and height on output caps based on the information provided
    // byt the custom library
    gst_structure_fixate_field_nearest_int(s2, "width", width);
    gst_structure_fixate_field_nearest_int(s2, "height", height);
    if (gst_structure_get_fraction(s1, "framerate", &num, &denom))
    {
        gst_structure_fixate_field_nearest_fraction(s2, "framerate", num, denom);
    }

    // TODO: Get width, height, coloutformat, and framerate from customlibrary API
    // set the new properties accordingly
    gst_structure_get_int (s2, "width", &w);
    gst_structure_get_int (s2, "height", &h);

    if ((w != 0) && (h != 0))
    {
        gst_structure_set (s2, "width",  G_TYPE_INT, w, NULL);
        gst_structure_set (s2, "height", G_TYPE_INT, h, NULL);
    }

    if (!outputFmt)
    {
        gst_structure_set (s2, "format", G_TYPE_STRING, inputFmt, NULL);
    }
    gst_structure_set (s2, "pixel-aspect-ratio", GST_TYPE_FRACTION, 1, 1, NULL);

    GST_DEBUG_OBJECT (m_element, "%s : Updated OTHERCAPS = %s \n\n", __func__, gst_caps_to_string(othercaps));

    break;
  }
  return othercaps;
}

char *SampleAlgorithm::QueryProperties ()
{
    char *str = new char[1000];
    strcpy (str, "CUSTOM LIBRARY PROPERTIES\n \t\t\tcustomlib-props=");
    return str;
}

bool SampleAlgorithm::HandleEvent (GstEvent *event)
{
  switch (GST_EVENT_TYPE(event))
  {
       case GST_EVENT_EOS:
           m_eosReceived = true;
           break;
       default:
           break;
  }
  if ((GstNvEventType)GST_EVENT_TYPE(event) == GST_NVEVENT_STREAM_EOS)
  {
      gst_nvevent_parse_stream_eos (event, &source_id);
      m_eosReceived = true;
  }
  if ((GstNvEventType)GST_EVENT_TYPE(event) == GST_NVEVENT_PAD_ADDED)
  {
      gst_nvevent_parse_pad_added (event, &source_id);
  }
  if ((GstNvEventType)GST_EVENT_TYPE(event) == GST_NVEVENT_PAD_DELETED)
  {
      gst_nvevent_parse_pad_deleted (event, &source_id);
  }
  return true;
}

// Set Custom Library Specific Properties
bool SampleAlgorithm::SetProperty(Property &prop)
{
  m_VectorProperty.emplace_back(prop.key, prop.value);
  return true;
}

/* Deinitialize the Custom Lib context */
SampleAlgorithm::~SampleAlgorithm()
{
  if (m_dsBufferPool) {
      gst_buffer_pool_set_active (m_dsBufferPool, FALSE);
      gst_object_unref(m_dsBufferPool);
      m_dsBufferPool = NULL;
  }
  if (m_swbufpool) {
      gst_buffer_pool_set_active (m_swbufpool, FALSE);
      gst_object_unref (m_swbufpool);
      m_swbufpool = NULL;
  }
  if (m_scratchNvBufSurface)
  {
    cudaFreeHost (m_scratchNvBufSurface);
    m_scratchNvBufSurface = NULL;
  }

  // Clean up tensor cache
  if (m_TensorCache) {
    delete m_TensorCache;
    m_TensorCache = nullptr;
  }

  // Clean up logger
  if (m_Logger) {
    delete m_Logger;
    m_Logger = nullptr;
  }

  // Clean up TensorRT engine (will be handled by unique_ptr destructor)
  m_TensorRTEngine.reset();
}

// Returns NvDsBatchMeta if present in the gstreamer buffer else NULL
NvDsBatchMeta *SampleAlgorithm::GetNVDS_BatchMeta (GstBuffer *buffer)
{
  gpointer state = NULL;
  GstMeta *gst_meta = NULL;
  NvDsBatchMeta *batch_meta = NULL;

  while ((gst_meta = gst_buffer_iterate_meta(buffer, &state))) {
    if (!gst_meta_api_type_has_tag (gst_meta->info->api, _dsmeta_quark)) {
      continue;
    }
    NvDsMeta *dsmeta = (NvDsMeta *) gst_meta;

    if (dsmeta->meta_type == NVDS_BATCH_GST_META) {
      if (batch_meta != NULL) {
        GST_WARNING("Multiple NvDsBatchMeta found on buffer %p", buffer);
      }
      batch_meta = (NvDsBatchMeta *) dsmeta->meta_data;
    }
  }

  return batch_meta;
}

/* Process Buffer */
BufferResult SampleAlgorithm::ProcessBuffer (GstBuffer *inbuf)
{
  GstMapInfo in_map_info;
  NvBufSurface *in_surf;
  NvDsBatchMeta *batch_meta = NULL;
  NvDsMetaList* l_frame = NULL;
  NvDsFrameMeta *frame_meta = NULL;
  Tensor* preprocessed_tensor = new Tensor();
  int num_filled = 0;
  bool inference_success = false;

  // Initial ProcessBuffer start
  m_Logger->processStart("ProcessBuffer", m_FrameNum, true);

  GST_DEBUG_OBJECT (m_element, "CustomLib: ---> Inside %s frame_num = %d\n", __func__, m_FrameNum);

  // TODO: End of Stream Handling
  memset (&in_map_info, 0, sizeof (in_map_info));

  /* Map the buffer contents and get the pointer to NvBufSurface. */
  if (!gst_buffer_map (inbuf, &in_map_info, GST_MAP_READ)) {
    GST_ELEMENT_ERROR (m_element, STREAM, FAILED,
        ("%s:gst buffer map to get pointer to NvBufSurface failed", __func__), (NULL));
    return BufferResult::Buffer_Error;
  }

  // Assuming that the plugin uses DS NvBufSurface data structure
  in_surf = (NvBufSurface *) in_map_info.data;

  gst_buffer_unmap(inbuf, &in_map_info);

  //batch_meta = gst_buffer_get_nvds_batch_meta (inbuf);
  batch_meta = GetNVDS_BatchMeta (inbuf);

  // Batch meta is attached when nvstreammux plugin is present in the pipeline and
  // it should be upstream plugin in the pipeline
  if (batch_meta) {
    num_filled = batch_meta->num_frames_in_batch;
  } else {
    num_filled = 1;
  }

  GST_DEBUG ("\tCustomLib: Element : %s in_surf = %p\n", GST_ELEMENT_NAME(m_element), in_surf);
  GST_DEBUG ("\tCustomLib: Num of frames in batch: %d\n", num_filled);

  // Check for empty batch after EOS
  if (num_filled == 0 && !m_eosReceived) {
    GOTO_SKIP("Empty batch received - no frames to process");
  } else if (num_filled == 0 && m_eosReceived) {
    GOTO_SKIP("Empty batch received after EOS - passing through");
  }

  // Skip batch if interval is set and no frames match
  if (batch_meta && m_ConfigProperties.interval > 0) {
    // Check if any frame in the batch should be processed
    // NOTE: This assumes homogeneous batches where all frames have the same frame number
    // This is typically true when:
    // 1. Using synchronized sources with sync-inputs-ntp=1
    // 2. Sort-batch enabled in nvstreammux
    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
      frame_meta = (NvDsFrameMeta *) (l_frame->data);
      if (frame_meta->frame_num % (m_ConfigProperties.interval + 1) > 0) {
        GOTO_SKIP("Skipping Batch " + std::to_string(m_FrameNum) + " [interval=" + 
                            std::to_string(m_ConfigProperties.interval) + "]");
      }
      break;
    }
  }

  // Enable for dumping the input frame, using dump_frames and dump_max_frames from config
  if (m_ConfigProperties.dump_frames && m_ConfigProperties.dump_max_frames > 0) {
    m_Logger->processStart("Dump input_frame", m_FrameNum);

    // Dump the input frame
    DumpNvBufSurface(in_surf, batch_meta, "input_frame", m_FrameNum, m_inVideoFmt, m_scratchNvBufSurface, true, m_Logger);

    m_Logger->processComplete("Dump input_frame", m_FrameNum, true);
  }

  // Update the frame number for the cached tensors (use processed frame count for consecutive numbering)
  for (const auto& [name, tensor_ptr] : m_TensorCache->getTensors()) {
    tensor_ptr->frame_num = m_ProcessedFrameNum;
  }

  /*
  * Preprocessing Tensors
  *
  * This section is responsible for preprocessing the input tensors.
  * It will be called only if the NvDsBatchMeta is present in the buffer.
  *
  * The preprocessing steps include:
  * 1. Parsing the preprocessed tensor data from batch meta
  * 2. Dumping the preprocessed tensor
  * 3. Updating the TensorRT binding for the input tensor
  * 4. Logging the preprocessed tensor
  */
  if (batch_meta) {
    m_Logger->processStart("PreProcess", m_FrameNum);

    // Parse the preprocessed tensor data
    preprocessed_tensor = parse_preprocess_batch_meta(batch_meta, m_Logger);

    if (preprocessed_tensor && preprocessed_tensor->buffer != NULL && preprocessed_tensor->size > 0) {
      // Dump preprocessed tensor
      if (m_ConfigProperties.dump_frames && m_ConfigProperties.dump_max_frames > 0) {
        m_Logger->processStart("Dump preprocessed_frame", m_FrameNum);
        dump_tensor(*preprocessed_tensor, "preprocessed_frame", true, m_Logger);
        m_Logger->processComplete("Dump preprocessed_frame", m_FrameNum, true);
      }

      // Update the binding pointer to use our preprocessed buffer directly (zero copy)
      if (m_ConfigProperties.fixed_input_tensor_path.empty()) {
        bool success = m_TensorRTEngine->updateBindingPointer("img", preprocessed_tensor->buffer)
                      && m_TensorCache->updateTensorBuffer("img", preprocessed_tensor->buffer);
        if (success) {
          m_Logger->info("Frame " + std::to_string(m_FrameNum) + 
                         " - Updated TensorRT binding for input tensor[img] with preprocessed buffer");
        } else {
          GOTO_ERROR("Failed to update TensorRT binding for input tensor[img] with preprocessed buffer");
        }
      }

      if (!m_ConfigProperties.partial_batch && preprocessed_tensor->shape[1] != m_ConfigProperties.num_sensors) {
        GOTO_ERROR("Number of sensors mismatch: Sparse4D model is configured with " +
                         std::to_string(m_ConfigProperties.num_sensors) + " sources but provided " +
                         std::to_string(preprocessed_tensor->shape[1]));
      }

      if (m_ConfigProperties.partial_batch && preprocessed_tensor->shape[1] > m_ConfigProperties.num_sensors) {
        GOTO_ERROR("Number of sensors mismatch: Sparse4D model is configured with " +
                         std::to_string(m_ConfigProperties.num_sensors) + " sources but provided " +
                         std::to_string(preprocessed_tensor->shape[1]));
      }

      /* Update the projection matrices based on the current sensors in the batch */
      std::string sensor_names_str;
      std::string absent_sensor_names_str;
      std::vector<std::string> current_sensor_names;
      std::set<std::string> absent_sensor_names(m_SensorNames.begin(), m_SensorNames.end());
      for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;l_frame = l_frame->next) {
        frame_meta = (NvDsFrameMeta *) (l_frame->data);
        current_sensor_names.push_back(frame_meta->sensorInfo_meta.sensor_name);
        sensor_names_str += frame_meta->sensorInfo_meta.sensor_name;
        absent_sensor_names.erase(frame_meta->sensorInfo_meta.sensor_name);
        if (l_frame->next != NULL) {
          sensor_names_str += ",";
        }
      }

      if (preprocessed_tensor->shape[1] != m_ConfigProperties.num_sensors && m_ConfigProperties.display_partial_batch_info) {
        for (auto it = absent_sensor_names.begin(); it != absent_sensor_names.end(); ++it) {
          absent_sensor_names_str += *it;
          if (std::next(it) != absent_sensor_names.end()) {
            absent_sensor_names_str += ",";
          }
        }
        m_Logger->persistent_info("Partial Batch [Frame " + std::to_string(m_FrameNum) + "] Dropped " +
                          std::to_string(absent_sensor_names.size()) + " sensors out of " +
                          std::to_string(m_ConfigProperties.num_sensors) + " sensors: [" +
                          absent_sensor_names_str + "]");
      }

      // Update the projection matrix binding with only the matrices for cameras in this batch
      updateProjectionMatrixForBatch(m_TensorCache,
                                    m_CameraTransformedProjectionMatrices,
                                    current_sensor_names,
                                    m_ConfigProperties.batch_size,
                                    m_ConfigProperties.num_sensors);
      // Update the original projection matrix binding with only the matrices for cameras in this batch
      if (m_ConfigProperties.generate_3d_bbox) {
        updateOrgProjectionMatrixForBatch(m_TensorCache,
                                      m_CameraProjectionMatrices,
                                      current_sensor_names,
                                      m_ConfigProperties.batch_size,
                                      m_ConfigProperties.num_sensors);
      }
      // Ensure TensorRT engine uses updated projection matrix buffer
      void* updated_proj_buffer = m_TensorCache->getTensorBuffer("projection_mat");
      if (updated_proj_buffer) {
        bool success = m_TensorRTEngine->updateBindingPointer("projection_mat", updated_proj_buffer);
        if (success) {
          m_Logger->info("Updated TensorRT binding for input tensor[projection_mat] with partial batch camera matrices");
        } else {
          m_Logger->warning("Failed to update TensorRT binding for input tensor[projection_mat]");
        }
      }

      m_Logger->processComplete("PreProcess", m_FrameNum, true);
    } else {
      GOTO_SKIP("No NVDS_PREPROCESS_BATCH_META found in NvDsBatchMeta");
    }
  } else {
    GOTO_SKIP("No NvDsBatchMeta found in the buffer");
  }

  /*
  * TensorRT Inference
  * 
  * This section is responsible for running the inference on the preprocessed tensor.
  * It will be called only if the preprocessed tensor is available.
  *
  * The inference steps include:
  * 1. Running the inference on the preprocessed tensor
  * 2. Logging the inference results
  */
  if (m_TensorRTEngine && m_TensorRTEngine->isInitialized()) {
    m_Logger->processStart("TensorRT Inference", m_FrameNum);

    // Run inference
    inference_success = m_TensorRTEngine->runInference(
        m_ConfigProperties.batch_size,
        preprocessed_tensor->shape[1],
        m_Logger);

    m_Logger->processComplete("TensorRT Inference", m_FrameNum, inference_success);
    if (!inference_success) {
      GOTO_ERROR("TensorRT Inference failed");
    }
  } else {
    GOTO_ERROR("TensorRT Engine is not initialized");
  }

  /*
  * Postprocessing Tensors
  *
  * This section is responsible for postprocessing the output tensors.
  * It will be called only if inference is successful.
  *
  * The postprocessing steps include:
  * 1. Cloning the output tensors to CPU PyTorch tensors
  * 2. Applying the postprocessing operations
  * 3. Feeding the postprocessed tensors to the next stage
  * 4. World coordinates to pixel coordinates conversion
  * 5. Adding the postprocessed tensors to the batch meta as a user meta
  * 6. Logging the postprocessed tensors
  */
  if (inference_success) {
    m_Logger->processStart("Postprocessing Tensors", m_FrameNum);
    bool postprocess_success = postprocess_tensors(m_TensorCache, batch_meta, m_ConfigProperties, m_Logger);
    if (!postprocess_success) {
      GOTO_ERROR("Postprocessing Tensors failed");
    }
    m_Logger->processComplete("Postprocessing Tensors", m_FrameNum, postprocess_success);
  }

  m_Logger->processComplete("ProcessBuffer", m_FrameNum, true, true);

  // Display the TensorCache
  if (m_ConfigProperties.display_tensor_info) {
    m_TensorCache->display(true);
  }

  // Update frame counters
  if (m_ConfigProperties.dump_max_frames > 0) {
    m_ConfigProperties.dump_max_frames--;
  }
  m_FrameNum++;
  m_ProcessedFrameNum++;  // Increment processed frame count (only for actually processed frames)
  return BufferResult::Buffer_Ok;

  // Skip frame processing
  skip:
    if (m_ConfigProperties.dump_max_frames > 0) {
      m_ConfigProperties.dump_max_frames--;
    }
    m_Logger->info(skipMessage);
    m_Logger->processComplete("ProcessBuffer", m_FrameNum, true, true);
    m_FrameNum++;
    return BufferResult::Buffer_Drop;

  // Error handling
  error :
    m_Logger->error(errorMessage);
    return BufferResult::Buffer_Error;
}
