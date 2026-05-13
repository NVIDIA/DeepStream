/*
 * SPDX-FileCopyrightText: Copyright (c) 2020 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

/**
 * @file
 * <b>NVIDIA DeepStream Audio Metadata Structures </b>
 *
 * @b Description: This file defines DeepStream audio metadata structures.
 */

/**
 * @defgroup  metadata_structures  Metadata Structures
 *
 * Define structures that hold metadata.
 * @ingroup NvDsMetaApi
 * @{
 */

#ifndef _NVDS_AUDIO_META_H_
#define _NVDS_AUDIO_META_H_

#include "glib.h"
#include "gmodule.h"

#include <nvdsmeta.h>
#include <nvbufaudio.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * Holds metadata for a audio frame in a batch.
 */
typedef struct _NvDsAudioFrameMeta {
  /** Holds the base metadata for the frame. */
  NvDsBaseMeta base_meta;
  /** Holds the pad or port index of the Gst-streammux plugin for the frame
   in the batch. */
  guint pad_index;
  /** Holds the location of the frame in the batch. */
  guint batch_id;
  /** Holds the current frame number of the source. */
  gint frame_num;
  /** Holds the presentation timestamp (PTS) of the frame. */
  guint64 buf_pts;
  /** Holds the ntp timestamp. */
  guint64 ntp_timestamp;
  /** Holds the source IDof the frame in the batch, e.g. the camera ID.
   It need not be in sequential order. */
  guint source_id;
  /** Holds the number of samples in the frame */
  gint num_samples_per_frame;
  /* Holds the sample rate for audio stream */
  guint sample_rate;
  /* Holds the number of channels in audio stream. */
  guint num_channels;
  /* Holds the audio format type. */
  NvBufAudioFormat format;
  /* Holds layout information indicating whether audio channels are interleaved
   * or non-interleaved */
  NvBufAudioLayout layout;
  /** Holds a Boolean indicating whether inference is performed on the frame. */
  gboolean bInferDone;
  /** Holds the index of the last object class inferred by the primary
   detector/classifier. */
  gint class_id;
  /** Hold confidence for last event detected (last NvDsClassifierMeta) */
  gfloat confidence;
  /** Holds a string describing the class of the detected event. */
  gchar class_label[MAX_LABEL_SIZE];
  /** Holds a pointer to a list of pointers of type @ref NvDsClassifierMeta
   in use for the frame. */
  NvDsClassifierMetaList *classifier_meta_list;
  /** Holds a pointer to a list of pointers of type @ref NvDsUserMeta
   in use for the frame. */
  NvDsUserMetaList *frame_user_meta_list;
  /** Holds additional user-defined frame information. */
  gint64 misc_frame_info[MAX_USER_FIELDS];
  /** For internal use. */
  gint64 reserved[MAX_RESERVED_FIELDS];
} NvDsAudioFrameMeta;

/**
 * Creates a batch metadata structure for a audio batch of specified size.
 *
 * @param[in] max_batch_size    The maximum number of frames in the batch.
 * @ return  A pointer to the created structure.
 */
NvDsBatchMeta *nvds_create_audio_batch_meta(guint max_batch_size);

/**
 * Destroys a batch metadata structure.
 *
 * @param[in] batch_meta    A pointer to audio batch metadata structure
 *                          to be destroyed.
 * @returns  True if the object was successfully destroyed, or false otherwise.
 */
gboolean nvds_destroy_audio_batch_meta(NvDsBatchMeta *batch_meta);

/**
 * \brief  Acquires a audio frame meta from a batch's audio frame meta pool.
 *
 * You must acquire a audio frame meta before you can fill it with audio frame metadata.
 *
 * @param[in] batch_meta    A pointer to batch meta from which to acquire
 *                          a audio frame meta.
 *
 * @return  A pointer to the acquired audio frame meta.
 */
NvDsAudioFrameMeta *nvds_acquire_audio_frame_meta_from_pool (NvDsBatchMeta *batch_meta);

/**
 * Adds a audio frame meta to a batch meta.
 *
 * @param[in] batch_meta    A pointer to the NvDsBatchMeta to which
 *                          @a frame_meta is to be added.
 * @param[in] frame_meta    A pointer to a frame meta to be added to
 *                          @a batch_meta.
 */
void nvds_add_audio_frame_meta_to_audio_batch(NvDsBatchMeta * batch_meta,
    NvDsAudioFrameMeta * frame_meta);

/**
 * Removes a audio frame meta from a batch meta.
 *
 * @param[in] batch_meta    A pointer to the batch meta from which @a frame_meta
 *                          is to be removed.
 * @param[in] frame_meta    A pointer to the frame meta to be removed from
 *                          @a batch_meta.
 */
void nvds_remove_audio_frame_meta_from_audio_batch (NvDsBatchMeta *batch_meta,
    NvDsAudioFrameMeta * frame_meta);

/**
 * @brief  Adds a classifier meta the audio frame meta.
 *
 * You must acquire a classifier meta with
 * nvds_acquire_classifier_meta_from_pool() and fill it with
 * classifier metadata before you add it to the audio frame metadata.
 *
 * @param[in] frame_meta        A pointer to the frame meta to which
 *                              @a classifier_meta is to be added.
 * @param[in] classifier_meta   A pointer to the classifier meta to be added
 *                              to @a obj_meta.
 */
void nvds_add_classifier_meta_to_audio_frame(NvDsAudioFrameMeta *frame_meta,
    NvDsClassifierMeta * classifier_meta);

/**
 * Removes a classifier meta from the audio frame meta to which it is attached.
 *
 * @param[in] frame_meta          A pointer to the frame meta from which
 *                              @a classifier_meta is to be removed.
 * @param[in] classifier_meta   A pointer to the classifier meta to be removed
 *                              from @a frame_meta.
 */
void nvds_remove_classifier_meta_from_audio_frame (NvDsAudioFrameMeta * frame_meta,
    NvDsClassifierMeta *classifier_meta);

/**
 * Add a user meta to a audio batch meta.
 *
 * @param[in] batch_meta    A pointer to batch meta to which @a user_meta
 *                          is to be added.
 * @param[in] user_meta     A pointer to a user meta to be added to
 *                          @a batch_meta.
 */
void nvds_add_user_meta_to_audio_batch(NvDsBatchMeta * batch_meta,
    NvDsUserMeta * user_meta);

/**
 * Add a user meta to a audio frame meta.
 *
 * @param[in] frame_meta    A pointer to the frame meta to which @a user_meta
 *                          is to be added.
 * @param[in] user_meta     A pointer to a user meta to be added to
 *                          @a frame_meta.
 */
void nvds_add_user_meta_to_audio_frame(NvDsAudioFrameMeta * frame_meta,
    NvDsUserMeta * user_meta);

/**
 * Removes a user meta from a audio batch meta to which it is attached.
 *
 * @param[in] batch_meta    A pointer to the audio batch meta from which @a user_meta
 *                          is to be removed.
 * @param[in] user_meta     A pointer to the user meta to be removed from
 *                          @a batch_meta.
 *
 * returns acquired @ref NvDsUserMeta pointer from user meta pool
 */
void nvds_remove_user_meta_from_audio_batch(NvDsBatchMeta * batch_meta,
    NvDsUserMeta * user_meta);

/**
 * Removes a user meta from a audio frame meta to which it is attached.
 *
 * @param[in] frame_meta    A pointer to the frame meta from which @a user_meta
 *                          is to be removed.
 * @param[in] user_meta     A pointer to the user meta to be removed from
 *                          @a frame_meta.
 */
void nvds_remove_user_meta_from_audio_frame(NvDsAudioFrameMeta * frame_meta,
    NvDsUserMeta * user_meta);

/**
 * @brief  Copies or transforms meta data from one buffer to another.
 *
 * @param[in] data      A pointer to a batch meta (of type @ref NvDsBatchMeta),
 *                      cast to @c gpointer.
 * @param[in] user_data Currently not in use and should be set to NULL.
 *
 * @return A pointer to a metadata structure, to be cast to type NvDsBatchMeta.
 */
gpointer nvds_audio_batch_meta_copy_func (gpointer data, gpointer user_data);

/**
 * Releases metadata from a batch meta.
 *
 * @param[in] data      A pointer to a batch meta (type @ref NvDsBatchMeta),
 *                      cast to @c gpointer.
 * @param[in] user_data Currently not in use and should be set to NULL.
 */
void nvds_audio_batch_meta_release_func(gpointer data, gpointer user_data);

/**
 * Returns a pointer to a specified frame meta in the frame meta list.
 *
 * @param[in] frame_meta_list   A pointer to a list of pointers to frame metas.
 * @param[in] index             The index of the frame meta to be returned.
 *
 * @return  A pointer to the @a index'th frame meta in the frame meta list.
 */
NvDsAudioFrameMeta *nvds_get_nth_audio_frame_meta (NvDsFrameMetaList *frame_meta_list,
    guint index);

/**
 * Removes all of the frame metadata attached to a batch meta.
 *
 * @param[in] batch_meta    A pointer to the batch whose frame meta list
 *                          is to be cleared.
 * @param[in] meta_list     A pointer to the frame meta list to be cleared.
 */
void nvds_clear_audio_frame_meta_list(NvDsBatchMeta *batch_meta,
    NvDsFrameMetaList *meta_list);

/**
 * Removes all of the classifier metadata attached to an audio frame meta.
 *
 * @param[in] frame_meta A pointer to @ref NvDsAudioFrameMeta from which @a
 *            NvDsClassifierMetaList needs to be cleared
 * @param[in] meta_list A pointer to @ref NvDsClassifierMetaList which needs to
 *            be cleared
 */
void nvds_clear_audio_classifier_meta_list(NvDsAudioFrameMeta *frame_meta,
    NvDsClassifierMetaList *meta_list);

/**
 * Removes all of the user metadata attached to the audio batch meta.
 *
 * @param[in] batch_meta    A pointer to the audio batch meta whose
 *                          user meta list is to be cleared.
 * @param[in] meta_list     A pointer to the user meta list to be
 *            cleared
 */
void nvds_clear_audio_batch_user_meta_list(NvDsBatchMeta *batch_meta,
    NvDsUserMetaList *meta_list);

/**
 * Removes all of the user metadata attached to the audio frame meta.
 *
 * @param[in] frame_meta    A pointer to the audio frame meta whose
 *                          user meta list is to be cleared.
 * @param[in] meta_list     A pointer to the user meta list to be cleared.
 */
void nvds_clear_audio_frame_user_meta_list(NvDsAudioFrameMeta *frame_meta,
    NvDsUserMetaList *meta_list);

/**
 * \brief  Makes a deep copy of a user meta list to the user meta list
 * in a specified audio batch meta.
 *
 * @param[in] src_user_meta_list    A pointer to the source user meta list.
 * @param[in] dst_batch_meta        A pointer to the destination batch meta.
 */
void nvds_copy_audio_batch_user_meta_list(NvDsUserMetaList *src_user_meta_list,
    NvDsBatchMeta *dst_batch_meta);

/**
 * \brief  Makes a deep copy of a frame meta to another frame meta.
 *
 * @param[in] src_frame_meta    A pointer to the source frame meta.
 * @param[in] dst_frame_meta    A pointer to the destination frame meta.
 */
void nvds_copy_audio_frame_meta(NvDsAudioFrameMeta *src_frame_meta,
    NvDsAudioFrameMeta *dst_frame_meta);

/**
 * \brief  Makes a deep copy of a source user meta list to the user meta list
 * in a specified audio frame meta.
 *
 * @param[in] src_user_meta_list    A pointer to the source user meta list.
 * @param[in] dst_frame_meta        A pointer to the destination audio frame meta.
 */
void nvds_copy_audio_frame_user_meta_list(NvDsUserMetaList *src_user_meta_list,
    NvDsAudioFrameMeta *dst_frame_meta);

/**
 * \brief  Makes a deep copy of a source frame meta list to the frame meta list
 *  in a specified batch meta.
 *
 * @param[in] src_frame_meta_list   A pointer to the source frame meta list.
 * @param[in] dst_batch_meta        A pointer to the destination batch meta.
 */
void nvds_copy_audio_frame_meta_list (NvDsFrameMetaList *src_frame_meta_list,
    NvDsBatchMeta *dst_batch_meta);

/**
 * \brief  Makes a deep copy of a source classifier meta list to the
 *  classifier meta list in a specified object meta.
 *
 * @param[in] src_classifier_meta_list  A pointer to the source
 *                                      classifier meta list.
 * @param[in] dst_frame_meta           A pointer to the destination
 *                                      audio frame meta.
 */
void nvds_copy_audio_classification_list(NvDsClassifierMetaList *src_classifier_meta_list,
    NvDsAudioFrameMeta *dst_frame_meta);

#ifdef __cplusplus
}
#endif
#endif

/** @} */
