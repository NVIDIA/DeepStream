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
 * <b>NVIDIA GStreamer DeepStream: Custom Message Functions</b>
 *
 * @b Description: This file specifies the NVIDIA DeepStream GStreamer custom
 * message functions.
 *
 */
 /**
 * @defgroup gst_mess_evnt_qry Events, Messages and Query based APIs
 *
 * Defines Events, Messages and Query-based APIs
 *
 */

#ifndef __GST_NVMESSAGE_H__
#define __GST_NVMESSAGE_H__

#include <gst/gst.h>

G_BEGIN_DECLS


/**
 * @defgroup gst_message_plugin Messages: Custom Message Functions API
 * Helper functions for custom GStreamer messages posted by DeepStream GStreamer
 * plugins.
 *
 * DeepStream GStreamer plugins post the following custom messages:
 * - Stream EOS - Posted by the `NvStreamMuxer` element when it recieves EOS
 *                on one of its sink pads.
 *
 * @ingroup gst_mess_evnt_qry
 * @{
 */

/**
 * Creates a new Stream EOS message.
 *
 * params[in] obj           The GStreamer object creating the message.
 * params[in] eos_stream_id Stream ID of the stream for which EOS
 *                          has been received.
 *
 * @return  A pointer to the new message.
 */
GstMessage * gst_nvmessage_new_stream_eos (GstObject *obj, guint eos_stream_id);

/**
 * Determines whether a message is a stream EOS message.
 *
 * params[in] message   A pointer to the message to be checked.
 *
 * @return  A Boolean; true if the message is a stream EOS message.
 */
gboolean gst_nvmessage_is_stream_eos (GstMessage * message);

/**
 * \brief  Parses the stream ID from a stream EOS message.
 *
 * The stream ID is the index of the stream which sent the EOS event to
 * Gst-streammux.
 *
 * params[in] message           A pointer to a stream EOS message.
 * params[out] eos_stream_id    A pointer to an unsigned integer in which
 *                              the stream ID is stored.
 *
 * @return  A Boolean; true if the message was successfully parsed.
 */
gboolean gst_nvmessage_parse_stream_eos (GstMessage * message, guint * eos_stream_id);

/** @} */

G_END_DECLS

#endif
