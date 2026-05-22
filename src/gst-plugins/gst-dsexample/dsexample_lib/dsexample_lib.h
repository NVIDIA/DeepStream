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

#ifndef __DSEXAMPLE_LIB__
#define __DSEXAMPLE_LIB__

#define MAX_LABEL_SIZE 128
#ifdef __cplusplus
extern "C" {
#endif

typedef struct DsExampleCtx DsExampleCtx;

// Init parameters structure as input, required for instantiating dsexample_lib
typedef struct
{
  // Width at which frame/object will be scaled
  int processingWidth;
  // height at which frame/object will be scaled
  int processingHeight;
  // Flag to indicate whether operating on crops of full frame
  int fullFrame;
} DsExampleInitParams;

// Detected/Labelled object structure, stores bounding box info along with label
typedef struct
{
  float left;
  float top;
  float width;
  float height;
  char label[MAX_LABEL_SIZE];
} DsExampleObject;

// Output data returned after processing
typedef struct
{
  int numObjects;
  DsExampleObject object[4];
} DsExampleOutput;

// Initialize library context
DsExampleCtx * DsExampleCtxInit (DsExampleInitParams *init_params);

// Dequeue processed output
DsExampleOutput *DsExampleProcess (DsExampleCtx *ctx, unsigned char *data);

// Deinitialize library context
void DsExampleCtxDeinit (DsExampleCtx *ctx);

#ifdef __cplusplus
}
#endif

#endif
