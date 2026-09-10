/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <stdio.h>
#include <syslog.h>
#include <stdarg.h>
#include <string.h>
#include "nvds_logger.h"

extern "C" {

void nvds_log_open()
{
}

void nvds_log_close()
{
}

void nvds_log(const char *category,  int priority, const char *data, ...)
{
  char logmsg[2048];
  char *logptr;

  /* this logic is necessary to prepend category in front of the formatted string */
  snprintf(logmsg, sizeof(logmsg), "%s: ", category);
  char catsize = strlen(logmsg);
  logptr = (char *)(logmsg + catsize);
  va_list args;
  va_start(args, data);
  vsnprintf(logptr, (sizeof(logmsg) - catsize), data, args);

  syslog(priority|LOG_LOCAL1, "%s\n", logmsg);
  va_end(args);
}


}

