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

#ifndef __NVC_HELPER_H__
#define __NVC_HELPER_H__

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>

G_BEGIN_DECLS

/* Debug category declaration for the helper functions */
GST_DEBUG_CATEGORY_EXTERN (gst_nvvideoconvert_debug);

/**
 * find_peer_element:
 * @pad: The pad to find the peer element for
 * @caps_filter: Pointer to store a capsfilter element if found
 *
 * Finds the peer element connected to the given pad, recursively looking
 * through capsfilters.
 *
 * Returns: The peer element if found, NULL otherwise
 */
GstElement* find_peer_element(GstPad *pad, GstElement **caps_filter);

/**
 * get_format_from_peer_srcpad:
 * @pad: The pad whose peer element's source pad format to check
 * @format: Pointer to a string pointer that will be set to the format string if found
 *
 * Gets the format from the source pad of the peer element connected to the given pad.
 * The caller is responsible for freeing the returned format string with g_free().
 *
 * Returns: TRUE if format is found, FALSE otherwise
 */
gboolean get_format_from_peer_srcpad(GstPad *pad, gchar **format);

/**
 * check_capsfilter_has_format_field:
 * @capsfilter: The capsfilter element to check
 * @format: Pointer to a string pointer that will be set to the format string if found
 *
 * Checks if the capsfilter has a format field specified and returns the format.
 * The caller is responsible for freeing the returned format string with g_free().
 *
 * Returns: TRUE if format field is found, FALSE otherwise
 */
gboolean check_capsfilter_has_format_field(GstElement *capsfilter, gchar **format);

/**
 * remove_format_from_caps:
 * @btrans: The GstBaseTransform element
 * @caps: The caps to modify
 * @format_to_remove: The format string to remove
 *
 * Creates a new caps structure with the specified format removed.
 *
 * Returns: A new caps structure with the format removed
 */
GstCaps *remove_format_from_caps(GstBaseTransform *btrans, GstCaps *caps, const gchar *format_to_remove);

/**
 * is_format_bgra64_or_uyvp:
 * @format: The format string to check
 *
 * Checks if the format is either BGRA64_LE or UYVP.
 *
 * Returns: TRUE if the format is BGRA64_LE or UYVP, FALSE otherwise
 */
gboolean is_format_bgra64_or_uyvp(const gchar *format);

/**
 * is_format_bgra64:
 * @format: The format string to check
 *
 * Checks if the format is BGRA64_LE.
 *
 * Returns: TRUE if the format is BGRA64_LE, FALSE otherwise
 */
gboolean
is_format_bgra64(const gchar *format);

/**
 * get_format_from_src_pad_caps:
 * @btrans: The GstBaseTransform element
 *
 * Retrieves the format from the source pad capsfilter.
 *
 * Returns: The format string if found, NULL otherwise
 */
gchar *
get_format_from_src_pad_capsfilter(GstBaseTransform *btrans);

/**
 * inspect_caps:
 * @btrans: The GstBaseTransform element
 * @direction: The pad direction (sink or src)
 * @caps: The caps to inspect
 *
 * Inspects the caps and modifies them based on the presence of BGRA64_LE format.
 *
 * Returns: Modified caps
 */
GstCaps *inspect_caps(GstBaseTransform *btrans, GstPadDirection direction, GstCaps *caps);

G_END_DECLS

#endif /* __NVC_HELPER_H__ */