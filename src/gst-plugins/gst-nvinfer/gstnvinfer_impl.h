/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2020 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef __GSTNVINFER_IMPL_H__
#define __GSTNVINFER_IMPL_H__

#include <string.h>
#include <sys/time.h>
#include <glib.h>
#include <gst/gst.h>

#include <vector>
#include <list>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>

#include "nvbufsurftransform.h"
#include "nvdsinfer_context.h"
#include "nvdsinfer_func_utils.h"
#include "nvdsmeta.h"
#include "nvdspreprocess_meta.h"
#include "nvtx3/nvToolsExt.h"

G_BEGIN_DECLS
typedef struct _GstNvInfer GstNvInfer;

void gst_nvinfer_logger(NvDsInferContextHandle handle, unsigned int unique_id,
    NvDsInferLogLevel log_level, const char* log_message, void* user_ctx);

G_END_DECLS

using NvDsInferContextInitParamsPtr = std::unique_ptr<NvDsInferContextInitParams>;
using NvDsInferContextPtr = std::shared_ptr<INvDsInferContext>;

typedef struct _GstNvInferObjectHistory GstNvInferObjectHistory;

/**
 * Holds info about one frame in a batch for inferencing.
 */
typedef struct {
  /** Ratio by which the frame / object crop was scaled in the horizontal
   * direction. Required when scaling the detector boxes from the network
   * resolution to input resolution. Not required for classifiers. */
  gdouble scale_ratio_x = 0;
  /** Ratio by which the frame / object crop was scaled in the vertical
   * direction. Required when scaling the detector boxes from the network
   * resolution to input resolution. Not required for classifiers. */
  gdouble scale_ratio_y = 0;
  /** Offsets if symmetric padding was performed whille scaling objects to
   * network resolition. */
  guint offset_left = 0;
  guint offset_top = 0;
  /** roi left and top for preprocessed tensor */
  guint roi_left = 0;
  guint roi_top = 0;
  /** NvDsObjectParams belonging to the object to be classified. */
  NvDsObjectMeta *obj_meta = nullptr;
  NvDsFrameMeta *frame_meta = nullptr;
  NvDsRoiMeta *roi_meta = nullptr;
  /** Index of the frame in the batched input GstBuffer. Not required for
   * classifiers. */
  guint batch_index = 0;
  /** Frame number of the frame from the source. */
  gulong frame_num = 0;
  /** The buffer structure the object / frame was converted from. */
  NvBufSurfaceParams *input_surf_params = nullptr;
  /** Pointer to the converted frame memory. This memory contains the frame
   * converted to RGB/RGBA and scaled to network resolution. This memory is
   * given to NvDsInferContext as input for pre-processing and inferencing. */
  gpointer converted_frame_ptr = nullptr;
  /** Pointer to the structure holding inference history for the object. Should
   * be NULL when inferencing on frames. */
  std::weak_ptr<GstNvInferObjectHistory> history;

} GstNvInferFrame;

using GstNvInferObjHistory_MetaPair =
    std::pair<std::weak_ptr<GstNvInferObjectHistory>, NvDsObjectMeta *>;

/**
 * Holds information about the batch of frames to be inferred.
 */
typedef struct {
  /** Vector of frames in the batch. */
  std::vector<GstNvInferFrame> frames;
  /** Pointer to the input GstBuffer. */
  GstBuffer *inbuf = nullptr;
  /** Batch number of the input batch. */
  gulong inbuf_batch_num = 0;
  /** Boolean indicating that the output thread should only push the buffer to
   * downstream element. If set to true, a corresponding batch has not been
   * queued at the input of NvDsInferContext and hence dequeuing of output is
   * not required. */
  gboolean push_buffer = FALSE;
  /** Boolean marking this batch as an event marker. This is only used for
   * synchronization. The output loop does not process on the batch.
   */
  gboolean event_marker = FALSE;
  /** Buffer containing the intermediate conversion output for the batch. */
  GstBuffer *conv_buf = nullptr;
  nvtxRangeId_t nvtx_complete_buf_range = 0;

  /** Sync object for allowing asynchronous call to nvbufsurftransform API
   * Wait and Destroy to be done before preprocess call of nvinfer */
  NvBufSurfTransformSyncObj_t sync_obj = NULL;

  /** List of objects not inferred on in the current batch but pending
   * attachment of lastest available classification metadata. */
  std::vector <GstNvInferObjHistory_MetaPair> objs_pending_meta_attach;
} GstNvInferBatch;


/**
 * Data type used for the refcounting and managing the usage of NvDsInferContext's
 * batch output and the output buffers contained in it. This is especially required
 * when the tensor output is flowed along with buffers as metadata or when the
 * segmentation output containing pointer to the NvDsInferContext allocated
 * memory is attached to buffers as metadata. Whenever the last ref on the buffer
 * is dropped, the callback to free the GstMiniObject-inherited GstNvInferTensorOutputObject
 * is called and the batch_output can be released back to the NvDsInferContext.
 */
typedef struct
{
  /** Parent type. Allows easy refcounting and destruction. Refcount will be
   * increased by 1 for each frame/object for which NvDsInferTensorMeta will be
   * generated. */
  GstMiniObject mini_object;
  /** NvDsInferContext pointer which hold the resource */
  NvDsInferContextPtr infer_context;
  /** NvDsInferContextBatchOutput instance whose output tensor buffers are being
   * sent as meta data. This batch output will be released back to the
   * NvDsInferContext when the last ref on the mini_object is removed. */
  NvDsInferContextBatchOutput batch_output;
} GstNvInferTensorOutputObject;

namespace gstnvinfer {

/** Holds runtime model update status along with the error message if any. */
struct ModelStatus
{
  /** Status of the model update. */
  NvDsInferStatus status;
  /** Config file used for model update. */
  std::string cfg_file;
  /* Error message string if any. */
  std::string message;
};

/** C++ helper class written on top of GMutex/GCond. */
class LockGMutex
{
public:
  LockGMutex (GMutex &mutex)
    :m (mutex) {
    lock ();
  }
  ~LockGMutex () {
    if (locked)
      unlock();
  }
  void lock ();
  void unlock ();
  void wait (GCond &cond);

private:
  GMutex &m;
  bool locked = false;
};

/** Enum for type of model update required. */
enum ModelLoadType
{
  /** Load a new model by just replacing the model engine assuming no network
   * architecture changes. */
  MODEL_LOAD_FROM_ENGINE,
  /** Load a new model with other configuration changes. This option will only
   * update the NvDsInferContext, any filtering/post-processing/pre-processing
   * done in gst-nvinfer will not be updated. An important requirement is that
   * the network input layer resolution should not changes. */
  MODEL_LOAD_FROM_CONFIG,
  /** Request the model load thread to stop. */
  MODEL_LOAD_STOP,
};

/* Helper class to manage the NvDsInferContext and runtime model update. The
 * model can be updated at runtime by setting "config-file-path" and/or
 * "model-engine-file" properties with the new config file/model engine file.
 *
 * The runtime update implementation would basically create and initialize a
 * new NvDsInferContext with new parameters and if successful will replace the
 * current NvDsInferContext instance with the new instance while taking care of
 * processing synchronization.
 *
 * Constraints of runtime model update:
 *   - Model input resolution and channels should not change
 *   - Batch-size of new model engine should be equal or greater than
 *     gst-nvinfer's batch-size
 *   - Type of the model (Detection/Classification/Segmentation) should not
 *     change.
 *
 * Check deepstream-test5-app README for more details on OTA and runtime model
 * update and sample test steps.*/
class DsNvInferImpl
{
public:
  using ContextReplacementPtr =
      std::unique_ptr<std::tuple<NvDsInferContextPtr, NvDsInferContextInitParamsPtr, std::string>>;

  DsNvInferImpl (GstNvInfer *infer);
  ~DsNvInferImpl ();
  /* Start the model load thread. */
  NvDsInferStatus start ();
  /* Stop the model load thread. Release the NvDsInferContext. */
  void stop ();

  bool isContextReady () const { return m_InferCtx.get(); }

  /** Load new model in separate thread */
  bool triggerNewModel (const std::string &modelPath, ModelLoadType loadType);

  /** replace context, action in submit_input_buffer */
  NvDsInferStatus ensureReplaceNextContext ();
  void notifyLoadModelStatus (const ModelStatus &res);

  /** NvDsInferContext to be used for inferencing. */
  NvDsInferContextPtr m_InferCtx;

  /** NvDsInferContext initialization params. */
  NvDsInferContextInitParamsPtr m_InitParams;

private:
  /** Class implementation of separate thread for runtime model load. */
  class ModelLoadThread
  {
  public:
    using ModelItem = std::tuple<std::string, ModelLoadType> ;

    ModelLoadThread (DsNvInferImpl &impl);
    ~ModelLoadThread ();
    void queueModel (const std::string &modelPath, ModelLoadType type) {
      m_PendingModels.push (ModelItem(modelPath, type));
    }
  private:
    void Run();

    DsNvInferImpl &m_Impl;
    std::thread m_Thread;
    nvdsinfer::GuardQueue<std::list<ModelItem>> m_PendingModels;
  };

  bool initNewInferModelParams (
      NvDsInferContextInitParams &newParams,
      const std::string &newModelPath, ModelLoadType loadType,
      const NvDsInferContextInitParams &oldParams);
  bool isNewContextValid (
      INvDsInferContext &newCtx, NvDsInferContextInitParams &newParam);
  bool triggerContextReplace (
      NvDsInferContextPtr ctx, NvDsInferContextInitParamsPtr params,
      const std::string &path);
  void loadModel (const std::string &path, ModelLoadType loadType);

  ContextReplacementPtr getNextReplacementUnlock ();
  NvDsInferStatus flushDataUnlock (LockGMutex &lock);
  NvDsInferStatus resetContextUnlock (
      NvDsInferContextPtr ctx, NvDsInferContextInitParamsPtr params,
      const std::string &path);

  GstNvInfer *m_GstInfer = nullptr;
  /** Updating model thread. */
  std::unique_ptr<ModelLoadThread> m_ModelLoadThread;
  ContextReplacementPtr m_NextContextReplacement;
};

}

#endif
