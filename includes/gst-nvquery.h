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

/**
 * @file
 * <b>NVIDIA GStreamer DeepStream: Helper Queries</b>
 *
 * @b Description: This file specifies the NVIDIA DeepStream GStreamer helper
 * query functions.
 *
 */
#ifndef __GST_NVQUERY_H__
#define __GST_NVQUERY_H__

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup gst_query_plugin Query Functions
 * Gets information such as the batch size and the number of streams.
 * @ingroup gst_mess_evnt_qry
 * @{
 */

/**
 * Creates a new batch-size query, which can be used by elements to query
 * the number of buffers in upstream elements' batched buffers.
 *
 * @return A pointer to the new batch size query.
 */
GstQuery * gst_nvquery_batch_size_new (void);

/**
 * Determines whether a query is a batch size query.
 *
 * params[in] query     A pointer to the query to be checked.
 *
 * @return  True if the query is a batch size query.
 */
gboolean gst_nvquery_is_batch_size (GstQuery * query);

/**
 * Sets the batch size, used by the elements responding to the batch size query.
 *
 * This function fails if the query is not a batch size query.
 *
 * params[in] query         A pointer to a batch size query.
 * params[in] batch_size    The batch size to be set.
 */
void gst_nvquery_batch_size_set (GstQuery * query, guint batch_size);

/**
 * Parses batch size from a batch size query.
 *
 * params[in] query         A pointer to a batch size query.
 * params[out] batch_size   A pointer to an unsigned integer in which the
 *                          batch size is stored.
 *
 * @return  True if the query was successfully parsed.
 */
gboolean gst_nvquery_batch_size_parse (GstQuery * query, guint * batch_size);

/**
 * Creates a number of streams query, used by elements to query
 * upstream the number of input sources.
 *
 * @return  A pointer to the new query.
 */
GstQuery * gst_nvquery_numStreams_size_new (void);

/**
 * Determines whether a query is a number-of-streams query.
 *
 * params[in] query     A pointer to the query to be checked.
 *
 * @return  A Boolean; true if the query is a number of streams query.
 */
gboolean gst_nvquery_is_numStreams_size (GstQuery * query);

/**
 * \brief  Sets the number of input sources.
 *
 * This function is used by elements responding to
 * a number of streams query. It fails if the query is not of the correct type.
 *
 * params[in] query             A pointer to a number-of-streams query.
 * params[in] numStreams_size   The number of input sources.
 */
void gst_nvquery_numStreams_size_set (GstQuery * query, guint numStreams_size);

/**
 * Parses the number of streams from a number of streams query.
 *
 * params[in] query         A pointer to a number-of-streams query.
 * params[out] batch_size   A pointer to an unsigned integer in which
 *                          the number of streams is stored.
 *
 * @return  True if the query was successfully parsed.
 */
gboolean gst_nvquery_numStreams_size_parse (GstQuery * query, guint * numStreams_size);

/**
 * Creates a preprocess poolsize query, used by elements to query
 * preprocess element for the size of buffer pool.
 *
 * params[in] gieid    An unsigned integer in which
 *                     the preprocess gie id is stored.
 * @return  A pointer to the new query.
 */
GstQuery * gst_nvquery_preprocess_poolsize_new (guint gieid);

/**
 * Determines whether a query is a preprocess poolsize query.
 *
 * params[in] query     A pointer to the query to be checked.
 *
 * @return  A Boolean; true if the query is a preprocess poolsize query.
 */
gboolean gst_nvquery_is_preprocess_poolsize (GstQuery * query);

/**
 * \brief  Sets the preprocess poolsize as a reponse to query.
 *
 * This function is used by elements responding to
 * a number of streams query. It fails if the query is not of the correct type.
 *
 * params[in] query             A pointer to a nv-preprocess-poolsize query.
 * params[in] preprocess_poolsize   The preprocess poolsize to be set.
 */
void gst_nvquery_preprocess_poolsize_set (GstQuery * query, guint preprocess_poolsize);

/**
 * Parses the preprocess poolsize from a preprocess poolsize query.
 *
 * params[in] query         A pointer to a nv-preprocess-poolsize query.
 * params[out] preprocess_poolsize   A pointer to an unsigned integer in which
 *                          the preprocess poolsize is stored.
 *
 * @return  True if the query was successfully parsed.
 */
gboolean gst_nvquery_preprocess_poolsize_parse (GstQuery * query, guint * preprocess_poolsize);

/**
 * Parses the preprocess gie id from a preprocess poolsize query.
 *
 * params[in] query     A pointer to a nv-preprocess-poolsize query.
 * params[out] gieid    A pointer to an unsigned integer in which
 *                          the preprocess gie id is stored.
 *
 * @return  True if the query was successfully parsed.
 */
gboolean gst_nvquery_preprocess_poolsize_gieid_parse (GstQuery * query, guint *gieId);

/**
 * Checks if a query is update_caps query.
 *
 * params[in] query     A pointer to a query.
 *
 * @return  True if the query was update_caps query.
 */
gboolean gst_nvquery_is_update_caps (GstQuery * query);

/**
 * Parses the update_caps query.
 *
 * params[in] query     A pointer to a update_caps query.
 * params[out] stream_index    A pointer to an unsigned integer in which
 *                          the stream_index is stored.
 * params[out] frame_rate    A pointer to an GValue string in which
 *                          the frame_rate is stored.
 *
 * @return  Void.
 */
void gst_nvquery_parse_update_caps (GstQuery *query, guint *stream_index, const GValue *frame_rate);

/**
 * Heterogeneous batching query for new streammux.
 *
 * params[in] srcpad     A pointer to a srcpad.
 * params[in] str     A pointer to a str of update_caps query.
 *
 * @return  True if the query was successfully pushed.
 */
gboolean gst_nvquery_update_caps_peer_query (GstPad *srcpad, GstStructure *str);

/**
 * Creates a new text embedding query, which can be used to request
 * text embedding information from the nvinfer plugin.
 *
 * params[in] text_input    A pointer to the input text string.
 * params[in] model         A pointer to the model name string.
 *
 * @return  A pointer to the new text embedding query.
 */
GstQuery * gst_nvquery_text_embedding_new (const gchar *text_input, const gchar *model);

/**
 * Determines whether a query is a text embedding query.
 *
 * params[in] query     A pointer to the query to be checked.
 *
 * @return  True if the query is a text embedding query.
 */
gboolean gst_nvquery_is_text_embedding (GstQuery * query);

/**
 * Sets the text embedding response data, used by elements responding
 * to the text embedding query.
 *
 * params[in] query     A pointer to a text embedding query.
 * params[in] id        A pointer to the unique identifier string.
 * params[in] created   The creation timestamp (epoch).
 * params[in] model     A pointer to the model name string.
 * params[in] data      A pointer to a GValue containing the embedding data array.
 */
void gst_nvquery_text_embedding_set (GstQuery * query, const gchar *id,
                                      guint64 created, const gchar *model,
                                      const GValue *data);

/**
 * Parses the text embedding response from a text embedding query.
 *
 * params[in] query     A pointer to a text embedding query.
 * params[out] id       A pointer to store the unique identifier string.
 * params[out] created  A pointer to store the creation timestamp.
 * params[out] model    A pointer to store the model name string.
 * params[out] data     A pointer to store the GValue containing embedding data.
 *
 * @return  True if the query was successfully parsed.
 */
gboolean gst_nvquery_text_embedding_parse (GstQuery * query, const gchar **id,
                                            guint64 *created, const gchar **model,
                                            const GValue **data);

/**
 * Parses the text embedding request parameters from a text embedding query.
 *
 * params[in] query         A pointer to a text embedding query.
 * params[out] text_input   A pointer to store the input text string.
 * params[out] model        A pointer to store the model name string.
 *
 * @return  True if the query was successfully parsed.
 */
gboolean gst_nvquery_text_embedding_parse_request (GstQuery * query,
                                                     const gchar **text_input,
                                                     const gchar **model);

/**
 * Creates a new image embedding query, which can be used to request
 * image embedding generation from the nvdsvisionencoder plugin.
 *
 * params[in] image_path   Filesystem path to the input image.
 * params[in] model        Model name string (for response metadata).
 * params[in] has_bbox     Whether a bounding box crop is specified.
 * params[in] bbox_left    Bounding box left coordinate (ignored if !has_bbox).
 * params[in] bbox_top     Bounding box top coordinate.
 * params[in] bbox_width   Bounding box width.
 * params[in] bbox_height  Bounding box height.
 *
 * @return  A pointer to the new image embedding query.
 */
GstQuery * gst_nvquery_image_embedding_new (const gchar *image_path,
                                             const gchar *model,
                                             gboolean has_bbox,
                                             gdouble bbox_left,
                                             gdouble bbox_top,
                                             gdouble bbox_width,
                                             gdouble bbox_height);

/**
 * Determines whether a query is an image embedding query.
 *
 * params[in] query     A pointer to the query to be checked.
 *
 * @return  True if the query is an image embedding query.
 */
gboolean gst_nvquery_is_image_embedding (GstQuery * query);

/**
 * Sets the image embedding response data, used by elements responding
 * to the image embedding query.
 *
 * params[in] query     A pointer to an image embedding query.
 * params[in] id        A pointer to the unique identifier string.
 * params[in] created   The creation timestamp (epoch).
 * params[in] model     A pointer to the model name string.
 * params[in] data      A pointer to a GValue containing the embedding data array.
 */
void gst_nvquery_image_embedding_set (GstQuery * query, const gchar *id,
                                       guint64 created, const gchar *model,
                                       const GValue *data);

/**
 * Parses the image embedding response from an image embedding query.
 *
 * params[in] query     A pointer to an image embedding query.
 * params[out] id       A pointer to store the unique identifier string.
 * params[out] created  A pointer to store the creation timestamp.
 * params[out] model    A pointer to store the model name string.
 * params[out] data     A pointer to store the GValue containing embedding data.
 *
 * @return  True if the query was successfully parsed.
 */
gboolean gst_nvquery_image_embedding_parse (GstQuery * query, const gchar **id,
                                             guint64 *created, const gchar **model,
                                             const GValue **data);

/**
 * Parses the image embedding request parameters from an image embedding query.
 *
 * params[in] query         A pointer to an image embedding query.
 * params[out] image_path   A pointer to store the image path string.
 * params[out] model        A pointer to store the model name string.
 * params[out] has_bbox     A pointer to store the bbox flag.
 * params[out] bbox_left    A pointer to store left coordinate.
 * params[out] bbox_top     A pointer to store top coordinate.
 * params[out] bbox_width   A pointer to store width.
 * params[out] bbox_height  A pointer to store height.
 *
 * @return  True if the query was successfully parsed.
 */
gboolean gst_nvquery_image_embedding_parse_request (GstQuery * query,
                                                      const gchar **image_path,
                                                      const gchar **model,
                                                      gboolean *has_bbox,
                                                      gdouble *bbox_left,
                                                      gdouble *bbox_top,
                                                      gdouble *bbox_width,
                                                      gdouble *bbox_height);

/** @} */

#ifdef __cplusplus
}
#endif

#endif
