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

#include "dsexample_lib.h"
#include <stdio.h>
#include <stdlib.h>

struct DsExampleCtx
{
    DsExampleInitParams initParams;
};

DsExampleCtx *
DsExampleCtxInit (DsExampleInitParams * initParams)
{
    DsExampleCtx *ctx = (DsExampleCtx *) calloc (1, sizeof (DsExampleCtx));
    ctx->initParams = *initParams;
    return ctx;
}

// In case of an actual processing library, processing on data wil be completed
// in this function and output will be returned
DsExampleOutput *
DsExampleProcess (DsExampleCtx * ctx, unsigned char *data)
{
    DsExampleOutput *out =
        (DsExampleOutput*)calloc (1, sizeof (DsExampleOutput));

    if (data != NULL)
    {
        // Process your data here
    }
    // Fill output structure using processed output
    // Here, we fake some detected objects and labels
    if (ctx->initParams.fullFrame)
    {
        out->numObjects = 2;
        out->object[0] = (DsExampleObject)
        {
            (float)(ctx->initParams.processingWidth) / 8,
                (float)(ctx->initParams.processingHeight) / 8,
                (float)(ctx->initParams.processingWidth) / 8,
                (float)(ctx->initParams.processingHeight) / 8, "Obj0"
        };

        out->object[1] = (DsExampleObject)
        {
            (float)(ctx->initParams.processingWidth) / 2,
                (float)(ctx->initParams.processingHeight) / 2,
                (float)(ctx->initParams.processingWidth) / 8,
                (float)(ctx->initParams.processingHeight) / 8, "Obj1"
        };
    }
    else
    {
        out->numObjects = 1;
        out->object[0] = (DsExampleObject)
        {
            (float)(ctx->initParams.processingWidth) / 8,
                (float)(ctx->initParams.processingHeight) / 8,
                (float)(ctx->initParams.processingWidth) / 8,
                (float)(ctx->initParams.processingHeight) / 8, ""
        };
        // Set the object label
        snprintf (out->object[0].label, 64, "Obj_label");
    }

    return out;
}

void
DsExampleCtxDeinit (DsExampleCtx * ctx)
{
    free (ctx);
}
