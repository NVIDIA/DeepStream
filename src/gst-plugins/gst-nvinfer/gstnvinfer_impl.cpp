/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <vector>
#include <sstream>
#include <cassert>
#include <string>

#include "gstnvinfer_impl.h"
#include "gstnvinfer.h"
#include "gstnvinfer_yaml_parser.h"
#include "gstnvinfer_property_parser.h"

GST_DEBUG_CATEGORY_EXTERN (gst_nvinfer_debug);
#define GST_CAT_DEFAULT gst_nvinfer_debug

using namespace nvdsinfer;

namespace gstnvinfer
{

void
LockGMutex::lock ()
{
  assert (!locked);
  if (!locked)
    g_mutex_lock (&m);
  locked = true;
}

void
LockGMutex::unlock ()
{
  assert (locked);
  if (locked)
    g_mutex_unlock (&m);
  locked = false;
}

void
LockGMutex::wait (GCond & cond)
{
  assert (locked);
  if (locked)
    g_cond_wait (&cond, &m);
}

DsNvInferImpl::DsNvInferImpl (GstNvInfer * infer)
  : m_InitParams (new NvDsInferContextInitParams),
    m_GstInfer (infer)
{
  NvDsInferContext_ResetInitParams (m_InitParams.get ());
}

DsNvInferImpl::~DsNvInferImpl ()
{
  if (m_InitParams) {
    delete[]m_InitParams->perClassDetectionParams;
    g_strfreev (m_InitParams->outputLayerNames);
    g_strfreev (m_InitParams->outputIOFormats);
    g_strfreev (m_InitParams->layerDevicePrecisions);
  }
}

/** Initialize the ModelLoadThread instance - start the model load thread. */
DsNvInferImpl::ModelLoadThread::ModelLoadThread (DsNvInferImpl & impl)
  : m_Impl (impl)
{
  m_Thread = std::thread (&DsNvInferImpl::ModelLoadThread::Run, this);
}

/** Destroy the ModelLoadThread instance - stop and join the model load thread. */
DsNvInferImpl::ModelLoadThread::~ModelLoadThread ()
{
  if (m_Thread.joinable ()) {
    //push stop signal
    m_PendingModels.push (ModelItem ("", MODEL_LOAD_STOP));
    m_Thread.join ();
  }
  m_PendingModels.clear ();
}

/** Callable function for the thread. */
void
DsNvInferImpl::ModelLoadThread::Run ()
{
  while (true) {
    /* Pop a pending model update item. This is a blocking call. */
    ModelItem item = m_PendingModels.pop ();
    std::string file;
    ModelLoadType type;
    std::tie (file, type) = item;

    // receive thread stop signal
    if (type == MODEL_LOAD_STOP)
      break;

    if (file.empty ())
      continue;

    // Load the new model
    m_Impl.loadModel (file, type);
  }
}

NvDsInferStatus
DsNvInferImpl::start ()
{
  m_ModelLoadThread.reset (new ModelLoadThread (*this));
  return NVDSINFER_SUCCESS;
}

void
DsNvInferImpl::stop ()
{
  m_ModelLoadThread.reset ();
  m_InferCtx.reset ();
}

/* Queue a new model update. */
bool
DsNvInferImpl::triggerNewModel (const std::string & modelPath,
    ModelLoadType loadType)
{
  if (!m_ModelLoadThread.get ()) {
    return false;
  }
  m_ModelLoadThread->queueModel (modelPath, loadType);
  return true;
}

/* Load the new model - Try to create a new NvDsInferContext instance with new
 * parameters and store the instance in m_NextContextReplacement. */
void
DsNvInferImpl::loadModel (const std::string & modelPath, ModelLoadType loadType)
{
  NvDsInferContextInitParams curParam;
  bool needUpdate = true;

  /* Check if a model is loaded but not yet being used for inferencing. */
  {
    LockGMutex lock (m_GstInfer->process_lock);
    curParam = *m_InitParams;
    needUpdate = !m_NextContextReplacement;
  }

  if (!needUpdate) {
    notifyLoadModelStatus (ModelStatus {
        NVDSINFER_UNKNOWN_ERROR, modelPath, "Trying to update model too frequently"}
    );
    return;
  }

  NvDsInferContextInitParamsPtr newParamsPtr (new NvDsInferContextInitParams);
  NvDsInferContextInitParams & newParams = *newParamsPtr;

  /* Initialize the NvDsInferContextInitParams struct with the new model
   * config. */
  if (!initNewInferModelParams (newParams, modelPath, loadType, curParam)) {
    notifyLoadModelStatus (ModelStatus {
        NVDSINFER_CONFIG_FAILED, modelPath,
          "Initialization of new model params failed"}
    );
    return;
  }

  NvDsInferContextHandle newContext;
  /* Create new NvDsInferContext instance. */
  NvDsInferStatus err =
      createNvDsInferContext (&newContext, newParams, m_GstInfer,
      gst_nvinfer_logger);

  if (err != NVDSINFER_SUCCESS) {
    /* Notify application if the model load failed. */
    notifyLoadModelStatus (ModelStatus {
        err, modelPath, "Creation new model context failed"}
    );
    return;
  }
  NvDsInferContextPtr newCtxPtr (newContext);
  assert (newCtxPtr.get ());

  /* Check that the input of the newly loaded model parameter is compatible
   * with gst-nvinfer instance. */
  if (!isNewContextValid (*newCtxPtr, newParams)) {
    notifyLoadModelStatus (ModelStatus {
        NVDSINFER_CONFIG_FAILED, modelPath,
          "New model's settings doesn't match current model"}
    );
    return;
  }

  /* Store the new NvDsInferContext instance so that it can be used for
   * inferencing after ensuring synchronization with existing buffers
   * being processed. */
  if (!triggerContextReplace (newCtxPtr, std::move (newParamsPtr), modelPath)) {
    notifyLoadModelStatus (ModelStatus {
        NVDSINFER_UNKNOWN_ERROR, modelPath, "trigger new model replace failed"}
    );
    return;
  }
}

/* Initialize NvDsInferContextInitParams with a combination of current
 * init params and new init params for the new model. */
bool
DsNvInferImpl::initNewInferModelParams (NvDsInferContextInitParams & newParams,
    const std::string & newModelPath, ModelLoadType loadType,
    const NvDsInferContextInitParams & oldParams)
{
  static const std::string string_yml = ".yml";
  static const std::string string_yaml = ".yaml";
  static const std::string string_txt = ".txt";
  NvDsInferContext_ResetInitParams (&newParams);
  assert (!newModelPath.empty ());

  switch (loadType) {
    case MODEL_LOAD_FROM_CONFIG:
      if(!newModelPath.compare (newModelPath.length() - 4, 4, string_yml) ||
              !newModelPath.compare (newModelPath.length() - 5, 5, string_yaml)) {
        if (!gst_nvinfer_parse_context_params_yaml (&newParams, newModelPath.c_str ())) {
          GST_WARNING_OBJECT (m_GstInfer,
              "[UID %d]: parse new model config file: %s failed.",
              m_GstInfer->unique_id, newModelPath.c_str ());
          return false;
        }
      } else if (!newModelPath.compare (newModelPath.length() - 4, 4, string_txt)) {
        if (!gst_nvinfer_parse_context_params (&newParams, newModelPath.c_str ())) {
          GST_WARNING_OBJECT (m_GstInfer,
              "[UID %d]: parse new model config file: %s failed.",
              m_GstInfer->unique_id, newModelPath.c_str ());
          return false;
        }
      }

      break;
    case MODEL_LOAD_FROM_ENGINE:
      g_strlcpy (newParams.modelEngineFilePath, newModelPath.c_str (),
          sizeof (newParams.modelEngineFilePath));
      newParams.networkMode = oldParams.networkMode;
      memcpy (newParams.meanImageFilePath, oldParams.meanImageFilePath,
          sizeof (newParams.meanImageFilePath));
      newParams.networkScaleFactor = oldParams.networkScaleFactor;
      newParams.networkInputFormat = oldParams.networkInputFormat;
      newParams.numOffsets = oldParams.numOffsets;
      memcpy (newParams.offsets, oldParams.offsets, sizeof (newParams.offsets));
      break;

    default:

      GST_WARNING_OBJECT (m_GstInfer,
          "[UID %d]: unsupported model load type (:%s), internal error.",
          m_GstInfer->unique_id, newModelPath.c_str ());
      return false;
  }

  newParams.maxBatchSize = oldParams.maxBatchSize;
  newParams.gpuID = oldParams.gpuID;
  newParams.uniqueID = oldParams.uniqueID;
  newParams.networkType = oldParams.networkType;
  newParams.useDBScan = oldParams.useDBScan;
  newParams.classifierThreshold = oldParams.classifierThreshold;
  newParams.segmentationThreshold = oldParams.segmentationThreshold;
  newParams.copyInputToHostBuffers = oldParams.copyInputToHostBuffers;
  newParams.outputBufferPoolSize = oldParams.outputBufferPoolSize;
  if (string_empty (newParams.labelsFilePath)
      && !string_empty (oldParams.labelsFilePath)) {
      g_strlcpy (newParams.labelsFilePath, oldParams.labelsFilePath,
        sizeof (newParams.labelsFilePath));
  }

  if (string_empty (newParams.customLibPath)
      && !string_empty (oldParams.customLibPath)) {
      g_strlcpy (newParams.customLibPath, oldParams.customLibPath,
        sizeof (newParams.customLibPath));
  }
  if (string_empty (newParams.customBBoxParseFuncName)
      && !string_empty (oldParams.customBBoxParseFuncName)) {
      g_strlcpy (newParams.customBBoxParseFuncName, oldParams.customBBoxParseFuncName,
        sizeof (newParams.customBBoxParseFuncName));
  }

  if ((oldParams.networkType == NvDsInferNetworkType_Detector
      || oldParams.networkType == NvDsInferNetworkType_InstanceSegmentation)
      && oldParams.numDetectedClasses) {
    newParams.numDetectedClasses = oldParams.numDetectedClasses;
    delete[]newParams.perClassDetectionParams;
    newParams.perClassDetectionParams =
        new NvDsInferDetectionParams[oldParams.numDetectedClasses];
    memcpy (newParams.perClassDetectionParams,
        oldParams.perClassDetectionParams,
        sizeof (newParams.perClassDetectionParams[0]) *
        newParams.numDetectedClasses);
  }

  return true;
}

/* Check that the input of the newly loaded model is compatible with
 * gst-nvinfer instance. */
bool
DsNvInferImpl::isNewContextValid (INvDsInferContext & newCtx,
    NvDsInferContextInitParams & newParam)
{
  if (newParam.maxBatchSize < m_GstInfer->max_batch_size) {
    GST_WARNING_OBJECT (m_GstInfer,
        "[UID %d]: New model batch-size[in config] (%d) is smaller then gst-nvinfer's"
        " configured batch-size (%d).", m_GstInfer->unique_id,
        newParam.maxBatchSize, m_GstInfer->max_batch_size);
    return false;
  }

  NvDsInferNetworkInfo networkInfo;
  newCtx.getNetworkInfo (networkInfo);
  if (m_GstInfer->network_width != (gint) networkInfo.width ||
      m_GstInfer->network_height != (gint) networkInfo.height) {
    GST_WARNING_OBJECT (m_GstInfer,
        "[UID %d]: New model input resolution (%dx%d) is not compatible with "
        "gst-nvinfer's current resolution (%dx%d).", m_GstInfer->unique_id,
        networkInfo.width, networkInfo.height, m_GstInfer->network_width,
        m_GstInfer->network_height);
    return false;
  }

  return true;
}

/* Notify the app of model load status using GObject signal. */
void
DsNvInferImpl::notifyLoadModelStatus (const ModelStatus & res)
{
  assert (!res.cfg_file.empty ());
  if (res.status == NVDSINFER_SUCCESS) {
    GST_INFO_OBJECT (m_GstInfer, "[UID %d]: Load new model:%s sucessfully",
        m_GstInfer->unique_id, res.cfg_file.c_str ());
  } else {
    GST_ELEMENT_WARNING (m_GstInfer, LIBRARY, SETTINGS,
        ("[UID %d]: Load new model:%s failed, reason: %s",
            m_GstInfer->unique_id, res.cfg_file.c_str (),
            (res.message.empty ()? "unknown" : res.message.c_str ())),
        (nullptr));
  }
  g_signal_emit (m_GstInfer, gst_nvinfer_signals[SIGNAL_MODEL_UPDATED], 0,
      (int) res.status, res.cfg_file.c_str ());
}

bool
DsNvInferImpl::triggerContextReplace (NvDsInferContextPtr ctx,
    NvDsInferContextInitParamsPtr params, const std::string & path)
{
  std::string lastConfig;
  LockGMutex lock (m_GstInfer->process_lock);
  m_NextContextReplacement.reset (new ContextReplacementPtr::element_type (ctx,
          std::move (params), path));
  return true;
}

DsNvInferImpl::ContextReplacementPtr
DsNvInferImpl::getNextReplacementUnlock ()
{
  if (!m_NextContextReplacement.get ()) {
    return nullptr;
  }
  ContextReplacementPtr next;
  // get next replacement, meanwhile empty m_NextContextReplacement
  next.swap (m_NextContextReplacement);
  return next;
}

/* Wait till all the buffers with gst-nvinfer are inferred/post-processed and
 * pushed downstream. */
NvDsInferStatus
DsNvInferImpl::flushDataUnlock (LockGMutex & lock)
{
  GstNvInferBatch *batch = new GstNvInferBatch;
  batch->event_marker = TRUE;

  /* Push the event marker batch to ensure all data processed. */
  g_queue_push_tail (m_GstInfer->input_queue, batch);
  g_cond_broadcast (&m_GstInfer->process_cond);

  /* Wait till all the items in the two queues are handled. */
  while (!g_queue_is_empty (m_GstInfer->input_queue)) {
    lock.wait (m_GstInfer->process_cond);
  }
  while (!g_queue_is_empty (m_GstInfer->process_queue)) {
    lock.wait (m_GstInfer->process_cond);
  }

  g_cond_broadcast (&m_GstInfer->process_cond);

  for (auto & si:*(m_GstInfer->source_info)) {
    si.second.object_history_map.clear ();
  }

  return NVDSINFER_SUCCESS;
}

/* Actually replace the current NvDsInferContext used for inferencing with
 * the newly created NvDsInferContext. Also query updated information about
 * the model. */
NvDsInferStatus
DsNvInferImpl::resetContextUnlock (NvDsInferContextPtr ctx,
    NvDsInferContextInitParamsPtr params, const std::string & path)
{
  assert (ctx.get () && params.get () && !path.empty ());
  m_InferCtx = ctx;
  m_InitParams = std::move (params);

  GstNvInfer *nvinfer = m_GstInfer;
  g_free (nvinfer->config_file_path);
  nvinfer->config_file_path = g_strdup (path.c_str ());
  /* Get the network resolution. */
  ctx->getNetworkInfo (nvinfer->network_info);
  nvinfer->network_width = nvinfer->network_info.width;
  nvinfer->network_height = nvinfer->network_info.height;

  /* Get information on all the bound layers. */
  ctx->fillLayersInfo (*nvinfer->layers_info);

  nvinfer->output_layers_info->clear ();
  for (auto & layer:*(nvinfer->layers_info)) {
    if (!layer.isInput)
      nvinfer->output_layers_info->push_back (layer);
  }

  return NVDSINFER_SUCCESS;
}

/* Check if a new model has been loaded. If yes, ensure synchronization by
 * waiting till all the queued buffers are processed and then do the actual
 * model replacement. */
NvDsInferStatus
DsNvInferImpl::ensureReplaceNextContext ()
{
  NvDsInferStatus err = NVDSINFER_SUCCESS;

  LockGMutex lock (m_GstInfer->process_lock);
  /* Get any newly loaded model. */
  ContextReplacementPtr nextReplacement = getNextReplacementUnlock ();
  if (!nextReplacement.get ())
    return NVDSINFER_SUCCESS;

  NvDsInferContextPtr nextCtx = std::get <0> (*nextReplacement);
  NvDsInferContextInitParamsPtr nextParams;
  nextParams.swap (std::get <1> (*nextReplacement));
  assert (nextParams.get ());
  const std::string path = std::get <2> (*nextReplacement);

  /* Wait for current processing to finish. */
  err = flushDataUnlock (lock);
  if (err != NVDSINFER_SUCCESS) {
    notifyLoadModelStatus (ModelStatus {
        err, path, "Model update failed while flushing data"}
    );
    return err;
  }

  /* Replace the model to be used for inferencing. */
  err = resetContextUnlock (nextCtx, std::move (nextParams), path);
  if (err != NVDSINFER_SUCCESS) {
    notifyLoadModelStatus (ModelStatus {
        err, path, "Model update failed while replacing the current context"}
    );
    return err;
  }

  /* Notify application of successful model load. */
  notifyLoadModelStatus (ModelStatus {
      NVDSINFER_SUCCESS, path, "New Model updated succefully"}
  );
  return NVDSINFER_SUCCESS;
}

}
