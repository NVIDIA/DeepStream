//! @file NVWarpCubeMap.cpp
//! Create a cubic environment map.
//!
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

#include "NVWarpCubeMap.h"
#define F_PI_2  1.5707963267948966192f


nvwarpResult nvwarpIntoCubemap(nvwarpHandle han, cudaStream_t stream, cudaTextureObject_t srcTex, const nvwarpFaceDescription_t desc[6],
    unsigned faceDim, int interpolateEdges, void *dstAddr, size_t dstRowBytes)
{
    static const signed char recipe[6][6][3] = {    /* Illegal combinations yield the canonical view for that direction */
        {   /* Front */
            {  0,  0,  0 },    /* F */
            {  0,  0, +1 },    /* R */
            {  0,  0,  0 },    /* B */
            {  0,  0, -1 },    /* L */
            {  0,  0,  0 },    /* T */
            {  0,  0, +2 }     /* G */
        },
        {   /* Right */
            { +1,  0, -1 },    /* F */
            { +1,  0,  0 },    /* R */
            { +1,  0, +1 },    /* B */
            { +1,  0,  0 },    /* L */
            { +1,  0,  0 },    /* T */
            { +1,  0, +2 }     /* G */
        },
        {   /* Back */
            { +2,  0,  0 },    /* F */
            { +2,  0, -1 },    /* R */
            { +2,  0,  0 },    /* B */
            { +2,  0, +1 },    /* L */
            { +2,  0,  0 },    /* T */
            { +2,  0, +2 }     /* G */
        },
        {   /* Left */
            { -1,  0, +1 },    /* F */
            { -1,  0,  0 },    /* R */
            { -1,  0, -1 },    /* B */
            { -1,  0,  0 },    /* L */
            { -1,  0,  0 },    /* T */
            { -1,  0, +2 }     /* G */
        },
        {   /* Top */
            {  0, +1, +2 },    /* F */
            {  0, +1, +1 },    /* R */
            {  0, +1,  0 },    /* B */
            {  0, +1, -1 },    /* L */
            {  0, +1,  0 },    /* T */
            {  0, +1,  0 }     /* G */
        },
        {   /* Bottom (Ground) */
            {  0, -1,  0 },    /* F */
            {  0, -1, +1 },    /* R */
            {  0, -1, +2 },    /* B */
            {  0, -1, -1 },    /* L */
            {  0, -1,  0 },    /* T */
            {  0, -1,  0 }     /* G */
        },
    };
    const nvwarpFaceDescription_t *const faceEnd = desc + 6;
    const nvwarpFaceDescription_t *face;
    nvwarpResult err = NVWARP_SUCCESS, e;

    nvwarpSetDstWidthHeight(han, faceDim, faceDim);
    nvwarpSetDstPrincipalPoint(han, NULL, 1);
    nvwarpSetDstFocalLengths(han, (faceDim - interpolateEdges) * .5f, 0.f);

    for (face = desc; face < faceEnd; ++face)
    {
        const signed char *dir = recipe[face->to][face->up];
        float angles[3] = { dir[0] * F_PI_2, dir[1] * F_PI_2, dir[2] * F_PI_2 };
        nvwarpSetEulerRotation(han, angles, "YXZ");
        if (NVWARP_SUCCESS != (e = nvwarpWarpBuffer(han, stream, srcTex, ((char*)dstAddr + face->y * dstRowBytes + face->x * 4), dstRowBytes)))
            err = e;
    }
    return err;
}
