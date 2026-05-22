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
 * <b>NVIDIA DeepStream logging API</b>
 *
 * @b Description: This file defines an API for logging and debugging
 * DeepStream applications.
 */

/**
 * @defgroup  ee_logging_group  Logging API
 *
 * Defines an API for logging and debugging DeepStream applications.
 *
 * @ingroup NvDsLoggerApi
 * @{
 */

#ifndef NVDS_LOGGER_H
#define NVDS_LOGGER_H

#include <syslog.h>

#define DSLOG_SYSLOG_IDENT "DSLOG"

//define additional categories here
#define DSLOG_CAT_CR "DSLOG:CR"
#define DSLOG_CAT_SG "DSLOG:SG"


#ifdef __cplusplus
extern "C" {
#endif


/**
 * @brief  Opens a connection to the logger.
 *
 * This function must be called once per DeepStream application execution,
 * prior to use of the logger.
 */
void nvds_log_open();


/**
 * Closes a connection to the logger.
 */
void nvds_log_close();

// Category is user (application) defined; priority is based on syslog levels
//     data is message to be logged
/**
 * Logs a message to a location determined by the setup script.
 *
 * @param[in] category  A pointer to a string which specifies the category of
 *                      this message. Categories are user-defined.
 * @param[in] priority  Severity of the event to be logged, based on syslog
 *                      levels. For more information, see
 *                      ./src/utils/nvds_logger/README relative to the
 *                      directory that contains the DeepStream SDK.
 *
 * @param[in] data      A pointer to a string containing the message. The
 *                      message may contain the format specifiers recognized
 *                      by %printf() in C/C++. @a data may be followeded by an
 *                      arbitary number of parameters that supply values for
 *                      the format specifiers.
 */
void nvds_log(const char *category, int priority, const char *data, ...);
#ifdef __cplusplus
}
#endif

#endif

/** @} */
