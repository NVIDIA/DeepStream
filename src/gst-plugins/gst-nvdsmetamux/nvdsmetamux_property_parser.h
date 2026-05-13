/**
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef NVDSMETAMUX_PROPERTY_FILE_PARSER_H_
#define NVDSMETAMUX_PROPERTY_FILE_PARSER_H_

#include <gst/gst.h>
#include "gstnvdsmetamux.h"

/**
 * This file describes the Macro defined for config file property parser.
 */

/** max string length */
#define _PATH_MAX 4096

#define NVDSMETAMUX_PROPERTY "property"
#define NVDSMETAMUX_PROPERTY_ENABLE "enable"
#define NVDSMETAMUX_PROPERTY_ACTIVE_PAD "active-pad"
#define NVDSMETAMUX_PROPERTY_PTS_TOLERANCE "pts-tolerance"

#define NVDSMETAMUX_USER_CONFIGS "user-configs"

#define NVDSMETAMUX_GROUP "group-"
#define NVDSMETAMUX_GROUP_SRC_IDS_MODEL "src-ids-model"

/**
 * Get GstNvDsMetaMuxMemory structure associated with buffer allocated using
 * GstNvDsMetaMuxAllocator.
 *
 * @param nvdsmetamux pointer to GstNvDsMetaMux structure
 *
 * @param cfg_file_path config file path
 *
 * @return boolean denoting if successfully parsed config file
 */
gboolean
nvdsmetamux_parse_config_file (GstNvDsMetaMux *nvdsmetamux, gchar *cfg_file_path);

#endif /* NVDSMETAMUX_PROPERTY_FILE_PARSER_H_ */
