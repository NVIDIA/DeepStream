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
#include <dlfcn.h>
#include "nvtx_helper.h"

#define NVTX_LIBRARY "libnvToolsExt.so.1"

/* NVTX related functions */
typedef void (*fnnvtxRangePushA) (const char *);
typedef void (*fnnvtxRangePop)(void);
typedef unsigned long (*fnnvtxRangeStartA) (const char *);
typedef void (*fnnvtxRangeEnd) (unsigned  long);
static void *nvtx_lib = NULL;
static fnnvtxRangePushA nvtxRangePushA;
static fnnvtxRangePop nvtxRangePop;
static fnnvtxRangeStartA nvtxRangeStartA;
static fnnvtxRangeEnd nvtxRangeEnd;

void __attribute__((constructor)) nvtx_helper_init(void);
void __attribute__((destructor)) nvtx_helper_deinit(void);

/* constructor for auto initiating the handle for nvtools */
void nvtx_helper_init(void)
{
  nvtx_lib = dlopen (NVTX_LIBRARY, RTLD_NOW);
  if (nvtx_lib)
  {
    nvtxRangePushA = (fnnvtxRangePushA)dlsym (nvtx_lib, "nvtxRangePushA");
    nvtxRangePop = (fnnvtxRangePop)dlsym (nvtx_lib, "nvtxRangePop");
    nvtxRangeStartA = (fnnvtxRangeStartA)dlsym (nvtx_lib, "nvtxRangeStartA");
    nvtxRangeEnd = (fnnvtxRangeEnd)dlsym (nvtx_lib, "nvtxRangeEnd");
  }
  if (nvtx_lib && (!nvtxRangePushA || !nvtxRangePop || !nvtxRangeStartA
        || !nvtxRangeEnd)) {

    dlclose (nvtx_lib);
    nvtx_lib = NULL; nvtxRangePushA = NULL; nvtxRangePop = NULL;
    nvtxRangeStartA = NULL; nvtxRangeEnd = NULL;
  }
}

void nvtx_helper_deinit(void)
{
  if (nvtx_lib)
  {
    dlclose(nvtx_lib);
    nvtx_lib = NULL; nvtxRangePushA = NULL; nvtxRangePop = NULL;
    nvtxRangeStartA = NULL; nvtxRangeEnd = NULL;
  }
}

void nvtx_helper_push_pop (char * context)
{
  /* if library present */
  if (nvtx_lib)
  {
    /* If arguments present call push else pop */
    if(context)
    {
      nvtxRangePushA(context);
    }
    else
    {
      nvtxRangePop();
    }
  }
}

/* Call this function if start is called in one thread and stop is called
 * in other thread
 */
void nvtx_helper_start_end (char * context, unsigned long *id)
{
  /* if library present */
  if (nvtx_lib)
  {
    /* If arguments present call StartA to obtain id, which will be used
     * for stop */
    if(context)
    {
      *id = nvtxRangeStartA(context);
    }
    else
    {
      nvtxRangeEnd(*id);
    }
  }
}
