//! @file NVWarpCubeMap.h
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

#ifndef __NVWARP_CUBEMAP__
#define __NVWARP_CUBEMAP__

#include "NVWarp360.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus


//! This ID is used to identify both the look-at direction and the up direction for each face.
typedef enum nvwarpFaceID_t {
    NVWARP_FRONT    = 0,    //! Toward the front face.
    NVWARP_RIGHT    = 1,    //! Toward the right face.
    NVWARP_BACK     = 2,    //! Toward the back face.
    NVWARP_LEFT     = 3,    //! Toward the left face.
    NVWARP_TOP      = 4,    //! Toward the top face.
    NVWARP_BOTTOM   = 5     //! Toward the bottom face.
} nvwarpFaceID_t;


//! There is a description for each of the 6 faces.
//! This can describe all of the formats and their rotational variants: H-cross, V-cross, 3x2, 2x3, 1x6, 6x1.
typedef struct nvwarpFaceDescription_t {
    unsigned        x;      //! The left edge of the face in the buffer.
    unsigned        y;      //! The top  edge of the face in the buffer.
    nvwarpFaceID_t  to;     //! The ID of the look-at direction for the face.
    nvwarpFaceID_t  up;     //! The ID of the   up    direction for the face.
} nvwarpFaceDescription_t;


//! Generate a cube map. 
//! \param[in]      han                 The Warp360 instance. This should be initialized properly for the source and warp type
//!                                     (typically NVWARP_EQUIRECT_PERSPECTIVE, but NVWARP_FISHEYE_PERSPECTIVE can be used for a 360 fisheye).
//! \param[in,out]  stream              The CUDA stream on which the rendering is to be performed, or 0 for the default stream.
//! \param[in,out]  srcTex              The source texture, appropriately initialized and buffered. Clamp the edges if the source wraps around.
//! \param[in,out]  faceDesc            An array of six descriptors for each face.
//! \param[in,out]  faceDim             The dimension, in linear pixels, for each face, which will be (faceDim X faceDim).
//! \param[in,out]  interpolateEdges    If 1, the cube map will interpolate the edges; this necessarily duplicates the pixels on adjacent faces.
//!                                     If 0, the pixels will be sampled a half pixel away from the edges, a more uniform distribution.
//!                                     Greater values will extrapolate beyond the edges by half pixels; this can be useful for motion estimation.
//! \param[in,out]  dstAddr             A pointer to pixel (0,0) in the destination CUDA buffer.
//! \param[in,out]  dstRowBytes         The byte stride between pixels vertically in the destination CUDA buffer.
//! \return         NVWARP_SUCCESS      if successful.
nvwarpResult nvwarpIntoCubemap(nvwarpHandle han, cudaStream_t stream, cudaTextureObject_t srcTex,
    const nvwarpFaceDescription_t faceDesc[6], unsigned faceDim, int interpolateEdges, void *dstAddr, size_t dstRowBytes);


// Face descriptions for various popular cube map formats.
//    nvwarpFaceDescription_t HCross[6] = {
//        { faceDim * 0, faceDim * 1, NVWARP_LEFT,   NVWARP_TOP   },
//        { faceDim * 1, faceDim * 1, NVWARP_FRONT,  NVWARP_TOP   },
//        { faceDim * 2, faceDim * 1, NVWARP_RIGHT,  NVWARP_TOP   },
//        { faceDim * 3, faceDim * 1, NVWARP_BACK,   NVWARP_TOP   },
//        { faceDim * 1, faceDim * 0, NVWARP_TOP,    NVWARP_BACK  },
//        { faceDim * 1, faceDim * 2, NVWARP_BOTTOM, NVWARP_FRONT },
//    };
//    nvwarpFaceDescription_t Continuous2x3[6] = {
//        { faceDim * 0, faceDim * 0, NVWARP_LEFT,   NVWARP_TOP  },
//        { faceDim * 1, faceDim * 0, NVWARP_FRONT,  NVWARP_TOP  },
//        { faceDim * 2, faceDim * 0, NVWARP_RIGHT,  NVWARP_TOP  },
//        { faceDim * 0, faceDim * 1, NVWARP_TOP,    NVWARP_LEFT },
//        { faceDim * 1, faceDim * 1, NVWARP_BACK,   NVWARP_LEFT },
//        { faceDim * 2, faceDim * 1, NVWARP_BOTTOM, NVWARP_LEFT },
//    };
//    nvwarpFaceDescription_t Pano2VR2x3[6] = {
//        { faceDim * 0, faceDim * 0, NVWARP_LEFT,   NVWARP_TOP   },
//        { faceDim * 1, faceDim * 0, NVWARP_FRONT,  NVWARP_TOP   },
//        { faceDim * 2, faceDim * 0, NVWARP_RIGHT,  NVWARP_TOP   },
//        { faceDim * 0, faceDim * 1, NVWARP_BACK,   NVWARP_TOP   },
//        { faceDim * 1, faceDim * 1, NVWARP_TOP,    NVWARP_BACK  },
//        { faceDim * 2, faceDim * 1, NVWARP_BOTTOM, NVWARP_FRONT },
//    };
//    nvwarpFaceDescription_t F2x3[6] = {
//        { faceDim * 0, faceDim * 0, NVWARP_FRONT,   NVWARP_TOP   },
//        { faceDim * 1, faceDim * 0, NVWARP_BACK,    NVWARP_TOP   },
//        { faceDim * 2, faceDim * 0, NVWARP_TOP,     NVWARP_BACK  },
//        { faceDim * 0, faceDim * 1, NVWARP_BOTTOM,  NVWARP_FRONT },
//        { faceDim * 1, faceDim * 1, NVWARP_LEFT,    NVWARP_TOP   },
//        { faceDim * 2, faceDim * 1, NVWARP_RIGHT,   NVWARP_TOP   },
//    };


#ifdef __cplusplus
}
#endif // __cplusplus
#endif // __NVWARP_CUBEMAP__
