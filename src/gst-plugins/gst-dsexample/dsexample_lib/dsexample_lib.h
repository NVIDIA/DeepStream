/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2020 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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
