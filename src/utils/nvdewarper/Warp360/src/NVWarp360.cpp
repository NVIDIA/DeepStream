/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


#include "NVWarp360.h"
#include "Warp360.h"
#include "Warp360Priv.h"
#include "NVWarp360Kernel.h"
#include <string.h>
#include <math.h>

#define WARP_DEBUG 0
#if WARP_DEBUG
    #include <stdio.h>
#endif /* WARP_DEBUG */

#ifndef   M_PI
    #define M_PI        3.1415926535897932385
#endif /* M_PI */
#ifndef M_2PI
    #define M_2PI       6.2831853071795864769
#endif /* M_2PI */
#ifndef M_PI_2
    #define M_PI_2      1.5707963267948966192
#endif /* M_PI_2 */
#define F_PI            ((float)M_PI)
#define F_PI_2          ((float)M_PI_2)
#define F_2PI           ((float)M_2PI)

#define MAJOR_VERSION   2   /**< Major version. */
#define MINOR_VERSION   0   /**< Minor Version. */
#define REVISION        1   /**< Revision. */
#define PATCH           3  /**< Development build. */
#define WARP360VERSION()    ((MAJOR_VERSION << 24) + (MINOR_VERSION << 16) + (REVISION << 8) + (PATCH << 0))

#define WARP_TYPE(srcType, dstType) ((nvwarpType_t)(((srcType) << NVSURF_BITS) | (dstType)))
#define GET_SRC_TYPE(warpType)      (((unsigned)(warpType) >> NVSURF_BITS) & NVSURF_MASK)
#define GET_DST_TYPE(warpType)      (((unsigned)(warpType)               ) & NVSURF_MASK)

#ifdef _MSC_VER
    #define strncpy(d, s, n) strncpy_s(d, n, s, n)
#endif /* _MSC_VER */

/********************************************************************************
 ********************************************************************************
 ********************************************************************************
 *****                                  UTILITIES                           *****
 ********************************************************************************
 ********************************************************************************
 ********************************************************************************/


/* Forward declaration */
namespace {
    float  EvaluatePade6(float r, const float *pade);
    double KEFindRootOfDistortionPolynomial(unsigned n, const double *cf, double maxX, double y);
}

static const char gDefaultAxes[] = { 'Y', 'X', 'Z', 0 };


/********************************************************************************
 * SET_TRANSFORM
 ********************************************************************************/

#define SET_TRANSFORM(T, t0, t1, t2, t3, t4, t5, t6, t7, t8) \
  do { T[0] = t0; T[1] = t1; T[2] = t2; T[3] = t3; T[4] = t4; T[5] = t5; T[6] = t6; T[7] = t7; T[8] = t8; } while(0)


/********************************************************************************
 * transformVector3
 ********************************************************************************/

static float* transformVector3(const float T[9], const float vin[3], float vout[3])
{
    vout[0] = (float)((double)T[0] * vin[0] + (double)T[3] * vin[1] + (double)T[6] * vin[2]);
    vout[1] = (float)((double)T[1] * vin[0] + (double)T[4] * vin[1] + (double)T[7] * vin[2]);
    vout[2] = (float)((double)T[2] * vin[0] + (double)T[5] * vin[1] + (double)T[8] * vin[2]);
    return vout;
}


/********************************************************************************
 * concatenateTransformations
 ********************************************************************************/

static float* concatenateTransformations(const float T1[9], const float T2[9], float T12[9])
{
    transformVector3(T2, T1 + 0, T12 + 0);
    transformVector3(T2, T1 + 3, T12 + 3);
    transformVector3(T2, T1 + 6, T12 + 6);
    return T12;
}


/********************************************************************************
 * yawTransformYdown
 ********************************************************************************/

static float* yawTransformYdown(float yaw, float T[9])
{
    const float c = cosf(yaw);
    const float s = sinf(yaw);
    SET_TRANSFORM(T, c, 0, -s, 0, 1, 0, s, 0, c);
    return T;
}


/********************************************************************************
 * pitchTransformYdown
 ********************************************************************************/

static float* pitchTransformYdown(float pitch, float T[9])
{
    const float c = cosf(pitch);
    const float s = sinf(pitch);
    SET_TRANSFORM(T, 1, 0, 0, 0, c, s, 0, -s, c);
    return T;
}


/********************************************************************************
 * rollTransformYdown
 ********************************************************************************/

static float* rollTransformYdown(float roll, float T[9])
{
    const float c = cosf(roll);
    const float s = sinf(roll);
    SET_TRANSFORM(T, c, s, 0, -s, c, 0, 0, 0, 1);
    return T;
}


/********************************************************************************
 * SetSrcRotationMatrixYdown
 ********************************************************************************/

static void SetSrcRotationMatrixYdown(float yaw, float pitch, float roll, float M[3 * 3])
{
    float A[9], B[9];
    (void)concatenateTransformations(yawTransformYdown(yaw, M), pitchTransformYdown(pitch, A), B);
    (void)concatenateTransformations(B, rollTransformYdown(roll, A), M);
}


/********************************************************************************
 * SetRotationAxis
 ********************************************************************************/

static void SetRotationAxis(float angle, char axis, float M[9])
{
    if ('x' <= axis && axis <= 'z')
    {
        angle = -angle;
        axis += 'A' - 'a';
    }
    switch (axis)
    {
        default:
        case 'Z': (void)rollTransformYdown (angle, M); break;
        case 'Y': (void)yawTransformYdown  (angle, M); break;
        case 'X': (void)pitchTransformYdown(angle, M); break;
    }
}


/********************************************************************************
 * SetEulerRotation
 ********************************************************************************/

static void SetEulerRotation(const float *angles, const char *axes, float R[9])
{
    float A[9], B[9], C[9];
    unsigned d;
    SetRotationAxis(*angles++, *axes++, C);    /* Assume at least one axis */
    for (d = 0; *axes; d = 1 - d)
    {
        SetRotationAxis(*angles++, *axes++, A);
        if (d)
            (void)concatenateTransformations(A, B, C);
        else
            (void)concatenateTransformations(A, C, B);
    }
    memcpy(R, (d ? B : C), sizeof(B));
}


/********************************************************************************
 * AnglesFromRotationMatrixYdown
 ********************************************************************************/

static bool AnglesFromRotationMatrixYdown(const float T[9], float ypr[3], const char *axes)
{
    float r;
    if (!strcmp(axes, gDefaultAxes))    /* "YXZ" */
    {
        r = hypotf(T[6], T[8]);
        if (r)                              /* Normal, non-singular case */
        {
            ypr[2] = atan2f(T[1], T[4]);    /* roll */
            ypr[0] = atan2f(T[6], T[8]);    /* yaw */
            ypr[1] = atan2f(-T[7], r);      /* pitch */
        }
        else                                /* Straight up or down: gimbal lock loses one degree of freedom */
        {
            ypr[2] = atan2f(-T[3], T[0]);   /* roll */
            ypr[1] = asinf(-T[7]);          /* pitch */
            ypr[0] = 0;                     /* we choose the yaw to be zero */
        }
        return true;
    }
    else if (!strcmp(axes, "ZXY"))
    {
        r = hypotf(T[3], T[4]);
        if (r)                              /* Normal, non-singular case */
        {
            ypr[0] = atan2f(-T[3], T[4]);   /* roll */
            ypr[2] = atan2f(-T[2], T[8]);   /* yaw */
            ypr[1] = atan2f(+T[5], r);      /* pitch */
        }
        else                                /* Straight up or down: gimbal lock loses one degree of freedom */
        {
            ypr[0] = atan2f(T[1], T[0]);    /* roll */
            ypr[1] = asinf(T[5]);           /* pitch */
            ypr[2] = 0;                     /* we choose the yaw to be zero */
        }
        return true;
    }
   else if (!strcmp(axes, "ZXZ"))
    {
        if (fabs(T[8]) < 1.f)                           /* Normal, non-singular case */
        {
            ypr[0] = atan2f(T[6], -T[7]);               /* first roll */
            ypr[2] = atan2f(T[2],  T[5]);               /* pitch */
            ypr[1] = atan2f(hypotf(T[2], T[5]), T[8]);  /* final roll */
        }
        else                                            /* Straight up or down: gimbal lock loses one degree of freedom */
        {
            ypr[2] = 0.f;                               /* we choose the final roll to be 0 */
            ypr[1] = (T[8] > 0.f) ? 0.f : F_PI;         /* pitch straight up or down */
            ypr[0] = atan2f(T[1], T[0]);                /* initial roll has it all */
        }
        return true;
    }
    return false;
}


/********************************************************************************
 * convertTransformBetweenYUpandYDown
 ********************************************************************************/

float* Warp360::convertTransformBetweenYUpandYDown(const float fr[9], float to[9])
{
    float M[9] = { +fr[0], -fr[1], -fr[2], -fr[3], +fr[4], +fr[5], -fr[6], +fr[7], +fr[8] };
    memcpy(to, M, sizeof(M));
    return to;
}


/********************************************************************************
 * ComputeSrcFocalLength
 ********************************************************************************/

static float ComputeSrcFocalLength(unsigned srcType, float radius, float angle, float d[4])
{
    float foclen;

    switch (srcType)
    {
        case NVSURF_PERSPECTIVE:
        {   float a = tanf(angle);
            double a2 = (double)a * (double)a;
            a *= (float)(((((d[3] * a2 + d[2]) * a2 + d[1]) * a2 + d[0]) * a2) + 1.);
            foclen = radius / a;
            // TODO: how is this affected by tangential distortion?
        }   break;
        case NVSURF_FISHEYE:
        {   double a2 = (double)angle * (double)angle;
            angle *= (float)(((((d[3] * a2 + d[2]) * a2 + d[1]) * a2 + d[0]) * a2) + 1.);
            foclen = radius / angle;
        }   break;
        case NVSURF_EQUIRECT:
            foclen = radius / angle;
            break;
        default:
            foclen = NAN;
    }
    return foclen;
}


/********************************************************************************
 * Warp360Params DistortFourAngles
 ********************************************************************************/

static void DistortFourAngles(float *angles, const float *srcDist)
{
    if (srcDist && (srcDist[0] || srcDist[1] || srcDist[2] || srcDist[3]))
    {
        float padeInverse[6];
        Compute6InverseRadialPadeCoefficients(srcDist, padeInverse);
        for (unsigned i = 0; i < 4u; ++i)
        {
            angles[i] = EvaluatePade6(fabsf(angles[i]), padeInverse) * ((angles[i] < 0.f) ? -1.f : +1.f);
            // TODO: Accommodate tangential distortion if perspective
        }
    }
}


/********************************************************************************
 * ComputeAxialAngleRange
 ********************************************************************************/

nvwarpResult ComputeAxialAngleRange(unsigned srcType, float srcFocalLen, float srcX0, float srcY0, const float *srcDist, uint32_t srcWidth, uint32_t srcHeight, float minMaxXY[4])
{
    int w, h;
    switch (srcType)
    {
        case NVSURF_PERSPECTIVE:
            minMaxXY[0] =  (0.f           - srcX0) / srcFocalLen;
            minMaxXY[1] =  (srcWidth  - 1 - srcX0) / srcFocalLen;
            minMaxXY[3] = -(0.f           - srcY0) / srcFocalLen;
            minMaxXY[2] = -(srcHeight - 1 - srcY0) / srcFocalLen;
            DistortFourAngles(minMaxXY, srcDist);
            for (w = 0; w < 4; ++w)
            {
                minMaxXY[w] = atanf(minMaxXY[w]);
            }
            break;

        case NVSURF_FISHEYE:
            minMaxXY[0] =  (0.f           - srcX0) / srcFocalLen;
            minMaxXY[1] =  (srcWidth  - 1 - srcX0) / srcFocalLen; /* Assuming that the fisheye is not 360 degrees */
            minMaxXY[3] = -(0.f           - srcY0) / srcFocalLen;
            minMaxXY[2] = -(srcHeight - 1 - srcY0) / srcFocalLen;
            DistortFourAngles(minMaxXY, srcDist);
            break;

        case NVSURF_EQUIRECT:
            w = (int)srcWidth;
            h = (int)srcHeight;
            if (!(fabs(h / srcFocalLen - F_PI)  <= 1.e-5f)) /* If not close to 90 degrees vertically, ... */
                --h;                                        /* ... measure angles between extreme pixels, rather than wrapping around */
            if (!(fabs(w / srcFocalLen - F_2PI) <= 1.e-5f)) /* If not close to 180 degrees horizontally, ... */
                --w;                                        /* ... measure angles between extreme pixels, rather than wrapping around */
            minMaxXY[0] =  (0.f - srcX0) / srcFocalLen;
            minMaxXY[1] =  (w   - srcX0) / srcFocalLen;
            minMaxXY[3] = -(0.f - srcY0) / srcFocalLen;
            minMaxXY[2] = -(h   - srcY0) / srcFocalLen;
            break;

        default:
            memset(minMaxXY, -1, sizeof(*minMaxXY) * 4);    /* Set to NaN */
            return NVWARP_ERR_UNIMPLEMENTED;
    }
    return NVWARP_SUCCESS;
}


/********************************************************************************
 * ComputeAxialAngleRange
 ********************************************************************************/

nvWarpResult ComputeAxialAngleRange(unsigned srcType, float srcFocalLen, float srcX0, float srcY0, const float *srcDist, const float srcBoundingBox[4], float minMaxXY[4])
{
    int w;

    float leftTopX     = srcBoundingBox[0];
    float leftTopY     = srcBoundingBox[1];
    float bottomRightX = srcBoundingBox[2];
    float bottomRightY = srcBoundingBox[3];

    switch (srcType)
    {
        case NVSURF_PERSPECTIVE:
            minMaxXY[0] =  (leftTopX     - srcX0) / srcFocalLen; /* left   fov */
            minMaxXY[1] =  (bottomRightX - srcX0) / srcFocalLen; /* right  fov */
            minMaxXY[3] = -(leftTopY     - srcY0) / srcFocalLen; /* top    fov */
            minMaxXY[2] = -(bottomRightY - srcY0) / srcFocalLen; /* bottom fov */
            DistortFourAngles(minMaxXY, srcDist);
            for (w = 0; w < 4; ++w)
            {
                minMaxXY[w] = atanf(minMaxXY[w]);
            }
            break;

        case NVSURF_FISHEYE:
            minMaxXY[0] =  (leftTopX     - srcX0) / srcFocalLen;
            minMaxXY[1] =  (bottomRightX - srcX0) / srcFocalLen; /* Assuming that the fisheye is not 360 degrees */
            minMaxXY[3] = -(leftTopY     - srcY0) / srcFocalLen;
            minMaxXY[2] = -(bottomRightY - srcY0) / srcFocalLen;
            DistortFourAngles(minMaxXY, srcDist);
            break;

        case NVSURF_EQUIRECT:
            minMaxXY[0] =  (leftTopX     - srcX0) / srcFocalLen;
            minMaxXY[1] =  (bottomRightX - srcX0) / srcFocalLen;
            minMaxXY[3] = -(leftTopY     - srcY0) / srcFocalLen;
            minMaxXY[2] = -(bottomRightY - srcY0) / srcFocalLen;
            break;

        default:
            memset(minMaxXY, -1, sizeof(*minMaxXY) * 4);    /* Set to NaN */
            return NVWARP_ERR_UNIMPLEMENTED;
    }
    return NVWARP_SUCCESS;
}


/********************************************************************************
 * ComputeOutputResolution
 ********************************************************************************/

nvwarpResult ComputeOutputResolution(unsigned dstType, float topAngle, float botAngle, float focLen, float control, float aspectRatio, uint32_t wh[2])
{
    float dPix, dAng, dTan;

    switch (dstType)
    {
        case NVSURF_PERSPECTIVE:
        case NVSURF_CYLINDER:
        case NVSURF_PANINI:
        case NVSURF_PUSHBROOM:
            dTan = tanf(topAngle) - tanf(botAngle);
            dPix = focLen * dTan;
            break;

        case NVSURF_FISHEYE:
        case NVSURF_EQUIRECT:
        case NVSURF_ROTCYLINDER:
            dAng = topAngle - botAngle;
            dPix = focLen * dAng;
            break;

        case NVSURF_STEREOGRAPHIC:
            dTan  = sinf(topAngle) / (cosf(topAngle) + control)
                  - sinf(botAngle) / (cosf(botAngle) + control);
            dPix = focLen * dTan;
            break;

        default:
            return NVWARP_ERR_UNIMPLEMENTED;
    }

    wh[1] = (unsigned)(dPix + 1.f) & ~1;                 /* Make it even */
    wh[0] = (unsigned)(wh[1] * aspectRatio + 1.f) & ~1;

    return NVWARP_SUCCESS;
}


/********************************************************************************
 ********************************************************************************
 ********************************************************************************
 *****                              VRWARPERPARAMS                          *****
 ********************************************************************************
 ********************************************************************************
 ********************************************************************************/


/********************************************************************************
 * Warp360Params constructor
 ********************************************************************************/

Warp360Params::Warp360Params()
{
    type        = static_cast<WarpType>(~0u);                    /* Default for uninitialized variables is NaN */
    srcWidth    = ~0u;
    srcHeight   = ~0u;
    srcX0       = NAN;
    srcY0       = NAN;
    srcFocalLen = NAN;
    srcRadius   = 0;                                            /* No circular clipping */
    memset(srcDist, 0, sizeof(srcDist));                        /* Default is no distortion */
    dstWidth    = ~0u;
    dstHeight   = ~0u;
    yaw         = pitch = roll = 0;                             /* Default is no rotation */
    topAngle    = +F_PI_2;                                      /* Default is full angular range, north pole ... */
    bottomAngle = -F_PI_2;                                      /* ... to south pole */
    userData    = nullptr;                                      /* No user data default */
    control[0]  = control[1] = control[2] = control[3] = NAN;
}


/********************************************************************************
 * Warp360Params computeSrcFocalLength
 ********************************************************************************/

bool Warp360Params::computeSrcFocalLength(float angle, float radius)
{
    const unsigned srcType = GET_SRC_TYPE(type);
    srcFocalLen = ComputeSrcFocalLength(srcType, radius, angle, srcDist);
    return srcFocalLen == srcFocalLen;
}


/********************************************************************************
 * Warp360Params computeOutputResolution
 ********************************************************************************/

bool Warp360Params::computeOutputResolution(float aspectRatio)
{
    uint32_t        wh[2];
    const unsigned  dstType = GET_DST_TYPE(type);
    nvwarpResult    err     = ComputeOutputResolution(dstType, topAngle, bottomAngle, srcFocalLen, control[0], aspectRatio, wh);
    dstWidth  = wh[0];
    dstHeight = wh[1];
    return NVWARP_SUCCESS == err;
}


/********************************************************************************
 * Warp360Params computeAxialAngleRange
 ********************************************************************************/

bool Warp360Params::computeAxialAngleRange(float minMaxXY[4]) const
{
    const unsigned srcType = GET_SRC_TYPE(type);
    return NVWARP_SUCCESS == ComputeAxialAngleRange(srcType, srcFocalLen, srcX0, srcY0, srcDist, srcWidth, srcHeight, minMaxXY);
}


/********************************************************************************
 * Warp360Params computeAxialAngleRange
 ********************************************************************************/

bool Warp360Params::computeAxialAngleRange(const float srcBoundingBox[4], float minMaxXY[4]) const
{
    const unsigned srcType = GET_SRC_TYPE(type);
    return NVWARP_SUCCESS == ComputeAxialAngleRange(srcType, srcFocalLen, srcX0, srcY0, srcDist, srcBoundingBox, minMaxXY);
}


/********************************************************************************
 ********************************************************************************
 ********************************************************************************
 *****                                 WARP360                              *****
 ********************************************************************************
 ********************************************************************************
 ********************************************************************************/


/****************************************************************************//**
 * Compute the flat focal length from vertical view angles and the distance between the pixels that correspond to the two angles.
 * @param[in]   topAngle    the   top  angle of the measurement.
 * @param[in]   botAngle    the bottom angle of the measurement.
 * @param[in]   diffPixels  the difference, in pixels, between the pixel corresponding to the top angle and the pixel
 *                          corresponding to the bottom angle, where there is a 1 pixel difference between adjacent pixels, not 2.
 * @param[out]  y0          a place to store the distance, in pixels, from the pixel corresponding to the top angle
 *                          to the pixel coordinate where the angle is 0.
 * @return                  the focal length.
 ********************************************************************************/

static float ComputeFlatFocalLengthFromVerticalViewAngles(float topAngle, float botAngle, uint32_t diffPixels, float *y0)
{
    float tanTop = tanf(topAngle),
          tanBot = tanf(botAngle),
          focLen = diffPixels / (tanTop - tanBot);
    *y0 = focLen * tanTop;
    return  focLen;
}


/****************************************************************************//**
 * Compute the round focal length from vertical view angles and the distance between the pixels that correspond to the two angles.
 * @param[in]   topAngle    the   top  angle of the measurement.
 * @param[in]   botAngle    the bottom angle of the measurement.
 * @param[in]   diffPixels  the difference, in pixels, between the pixel corresponding to the top angle and the pixel
 *                          corresponding to the bottom angle, where there is a 1 pixel difference between adjacent pixels, not 2.
 * @param[out]  y0          a place to store the distance, in pixels, from the pixel corresponding to the top angle
 *                          to the pixel coordinate where the angle is 0.
 * @return                  the focal length.
 ********************************************************************************/

static float ComputeRoundFocalLengthFromVerticalViewAngles(float topAngle, float botAngle, uint32_t diffPixels, float *y0)
{
    float focLen = diffPixels / (topAngle - botAngle);
    *y0 = focLen * topAngle;
    return  focLen;
}


/****************************************************************************//**
 * Compute the equirectangular focal length from vertical view angles and the distance between the pixels that correspond to the two angles.
 * @param[in]   topAngle    the   top  angle of the measurement.
 * @param[in]   botAngle    the bottom angle of the measurement.
 * @param[in]   diffPixels  the difference, in pixels, between the pixel corresponding to the top angle and the pixel
 *                          corresponding to the bottom angle, where there is a 1 pixel difference between adjacent pixels, not 2.
 * @param[out]  y0          a place to store the distance, in pixels, from the pixel corresponding to the top angle
 *                          to the pixel coordinate where the angle is 0.
 * @return                  the focal length.
 ********************************************************************************/

static float ComputeEquirectFocalLengthFromVerticalViewAngles(float topAngle, float botAngle, uint32_t diffPixels, float *y0, uint32_t width = 0, float *x0 = nullptr)
{
    float   dAng    = topAngle - botAngle,
            focLen  = ((fabs(dAng - F_PI) < 1.e-6) ? (diffPixels + 1) : diffPixels) / dAng;
    *y0 = focLen * topAngle - .5f;
    if (x0)
        *x0 = ((fabs(focLen * F_2PI - width) < 1.e-3) ? width : (width - 1)) * .5f - .5f;
    return  focLen;
}


/****************************************************************************//**
 * Compute the stereographic focal length from vertical view angles and the distance between the pixels that correspond to the two angles.
 * @param[in]   topAngle    the   top  angle of the measurement.
 * @param[in]   botAngle    the bottom angle of the measurement.
 * @param[in]   diffPixels  the difference, in pixels, between the pixel corresponding to the top angle and the pixel
 *                          corresponding to the bottom angle, where there is a 1 pixel difference between adjacent pixels, not 2.
 * @param[out]  y0          a place to store the distance, in pixels, from the pixel corresponding to the top angle
 *                          to the pixel coordinate where the angle is 0.
 * @return                  the focal length.
 ********************************************************************************/

static float ComputeStereographicFocalLengthFromVerticalViewAngles(float topAngle, float botAngle, uint32_t diffPixels, float *y0, float control)
{
    float tanTop = sinf(topAngle) / (cosf(topAngle) + control) *  (1.f + control),
          tanBot = sinf(botAngle) / (cosf(botAngle) + control) *  (1.f + control),
          focLen = diffPixels / (tanTop - tanBot);
    *y0 = focLen * tanTop;
    return  focLen;
}


/********************************************************************************
 * ComputeDimGrid
 ********************************************************************************/

static void ComputeDimGrid(uint32_t dstWidth, uint32_t dstHeight, const dim3& dimBlock, dim3& dimGrid)
{
    dimGrid.x  = (dstWidth  + dimBlock.x - 1) / dimBlock.x; /* ceil */
    dimGrid.y  = (dstHeight + dimBlock.y - 1) / dimBlock.y;
    dimGrid.z  = 1;
}


/********************************************************************************
 * DispatchEntry
 ********************************************************************************/

struct DispatchEntry
{
    nvwarpType_t  type;
    __host__ cudaError_t (*warpSurface)(dim3 dim_grid, dim3 dim_block, cudaStream_t stream, const Warp360KernelParams *params, cudaTextureObject_t srcTex, cudaSurfaceObject_t dstSurface);
    __host__ cudaError_t (*warpBuffer)(dim3 dim_grid, dim3 dim_block, cudaStream_t stream, const Warp360KernelParams *params, cudaTextureObject_t srcTex, void *dstAddr, size_t dstRowBytes);
};
#define DISPATCH_ENTRY(src, dst) { NVWARP_##src##_##dst, Warp<NVSURF_##src, NVSURF_##dst>, Warp<NVSURF_##src, NVSURF_##dst> }


/********************************************************************************
 * GetDispatch
 ********************************************************************************/

static const DispatchEntry* GetDispatch(nvwarpType_t type)
{
    static const DispatchEntry dispatcher[] = {
        DISPATCH_ENTRY(EQUIRECT,    PUSHBROOM),
        DISPATCH_ENTRY(EQUIRECT,    CYLINDER),
        DISPATCH_ENTRY(EQUIRECT,    EQUIRECT),
        DISPATCH_ENTRY(EQUIRECT,    FISHEYE),
        DISPATCH_ENTRY(EQUIRECT,    PANINI),
        DISPATCH_ENTRY(EQUIRECT,    PERSPECTIVE),
        DISPATCH_ENTRY(EQUIRECT,    STEREOGRAPHIC),
        DISPATCH_ENTRY(EQUIRECT,    ROTCYLINDER),
        DISPATCH_ENTRY(FISHEYE,     PUSHBROOM),
        DISPATCH_ENTRY(FISHEYE,     CYLINDER),
        DISPATCH_ENTRY(FISHEYE,     EQUIRECT),
        DISPATCH_ENTRY(FISHEYE,     FISHEYE),
        DISPATCH_ENTRY(FISHEYE,     PANINI),
        DISPATCH_ENTRY(FISHEYE,     PERSPECTIVE),
        DISPATCH_ENTRY(FISHEYE,     ROTCYLINDER),
        DISPATCH_ENTRY(PERSPECTIVE, EQUIRECT),
        DISPATCH_ENTRY(PERSPECTIVE, PANINI),
        DISPATCH_ENTRY(PERSPECTIVE, PERSPECTIVE),
    };
    const DispatchEntry *p;
    for (p = dispatcher; p != &dispatcher[sizeof(dispatcher) / sizeof(dispatcher[0])]; ++p)
        if (p->type == type)
            return p;
    return nullptr;
}


/********************************************************************************
 * nvwarpObject
 ********************************************************************************/

struct nvwarpObject
{
    dim3                    dimBlock;       /**< Block dimensions. */
    float                   pixelPhase;

    nvwarpType_t            type;           /**< The type of the warp. */
    uint32_t                srcType;
    uint32_t                dstType;
    uint32_t                srcWidth;       /**< The width  of the source image. */
    uint32_t                srcHeight;      /**< The height of the source image. */
    float                   srcRadius;      /**< Source circular clipping radius. */    /* (default 0 means no clipping) (unimplemented) */

    float                   control;        /**< Projection-specific controls. */

    float                   dstFocLenX;     /**< Reciprocal of the destination horizontal focal length */
    float                   dstFocLenY;     /**< Reciprocal of the destination  vertical   focal length */

    float                   topAngle;       /**< Top    angle of view */                /* (default +pi/2 */
    float                   botAngle;       /**< Bottom angle of view */                /* (default -pi/2 */

    void*                   userData;       /**< Pointer supplied by the user. */       /* (default NULL) */

    Warp360KernelParams     cuParams;       /**< Parameters to be fed to CUDA */
    float                   padeInverse[6]; /** Pade coefficients for the inverse of the distortion function. */

    const DispatchEntry     *dispatch;

    nvwarpObject()  { init(); }
    ~nvwarpObject() {}
    void            init()
                    {
                        memset(static_cast<void *>(this), -1, sizeof(*this));    /* NaN unless otherwise specified */
                        setWarpType(NVWARP_NONE);           /* No specified warp */
                        setRotationMatrix(nullptr);         /* Identity rotation */
                        setDistortion(nullptr);             /* No distortion */
                        setControl(0, 0.f);                 /* Zero control */
                        setDimBlock(dim3(8, 8, 1));         /* 8x8 blocks */
                        pixelPhase = 0.f;                   /* Zero pixel phase */
                        srcRadius  = 0.f;                   /* No circular clipping */
                        userData   = nullptr;               /* NULL user data */
                    }
    void            setDimBlock(dim3 dim_block) { dimBlock = dim_block; }
    dim3            getDimBlock() const         { return dimBlock; }
    void*           getUserData() const         { return userData; }
    void            setRotationMatrix(const float M[9])
                    {
                        if (M)
                            memcpy(cuParams.srcRot, M, sizeof(cuParams.srcRot));    /* Set to the specified matrix, ... */
                        else
                            rollTransformYdown(0.f, cuParams.srcRot);               /* ... or identity if NULL */
                    }
    void            getRotationMatrix(float M[9]) const
                    {
                        memcpy(M, cuParams.srcRot, sizeof(cuParams.srcRot));
                    }
    void            setEulerRotation(const float *angles, const char *axes);
    cudaError_t     setWarp360Params(const Warp360Params *params);
    nvwarpResult    setWarp360Params(const nvwarpParams_t *params);
    nvwarpResult    warp(uint32_t numPts, const float *inPtsXY, float *outPtsXY) const;
    nvwarpResult    inverseWarp(uint32_t numPts, const float *inPtsXY, float *outPtsXY) const;
    nvwarpResult    setDstFocalLengthCOP(float topAngle, float bottomAngle, uint32_t dstWidth, uint32_t dstHeight);
    void            setSrcFocalLength(float fx, float fy)
                    {
                        cuParams.srcFocLenX = fx;
                        cuParams.srcFocLenY = fy ? fy : fx;
                    }
    void            setDstFocalLength(float fx, float fy)
                    {
                        dstFocLenX = fx;
                        dstFocLenY = fy ? fy : fx;
                        cuParams.dstInvFocLenX = 1.f / dstFocLenX;
                        cuParams.dstInvFocLenY = 1.f / dstFocLenY;
                    }
    void            setSrcPrincipalPoint(const float xy[2], uint32_t relToCenter)
                    {
                        const float zero[2] = { 0.f, 0.f };
                        if (!xy)
                        {
                            cuParams.srcX0 = 0.f;   /* NULL xy implies (0,0) */
                            cuParams.srcY0 = 0.f;
                            xy = zero;
                        }
                        if (!relToCenter)
                        {
                            cuParams.srcX0 = xy[0] + 0.5f - pixelPhase; /* srcX0 & srcY0 appropriate for sampling on integers-plus-one-half */
                            cuParams.srcY0 = xy[1] + 0.5f - pixelPhase;
                        }
                        else
                        {
                            cuParams.srcX0 = xy[0] + srcWidth  * 0.5f;
                            cuParams.srcY0 = xy[1] + srcHeight * 0.5f;
                        }
                    }
    void            getSrcPrincipalPoint(float xy[2], uint32_t relToCenter) const
                    {
                        if (!relToCenter)
                        {
                            xy[0] = cuParams.srcX0 + pixelPhase - 0.5f; /* srcX0 & srcY0 appropriate for sampling on integers-plus-one-half */
                            xy[1] = cuParams.srcY0 + pixelPhase - 0.5f;
                        }
                        else
                        {
                            xy[0] = cuParams.srcX0 - srcWidth  * 0.5f;
                            xy[1] = cuParams.srcY0 - srcHeight * 0.5f;
                        }
                    }
    void            setDstPrincipalPoint(const float xy[2], uint32_t relToCenter)
                    {
                        const float zero[2] = { 0.f, 0.f };
                        if (!xy)
                        {
                            cuParams.dstX0 = 0.f;   /* NULL xy implies (0,0) */
                            cuParams.dstY0 = 0.f;
                            xy = zero;
                        }
                        if (!relToCenter)
                        {
                            cuParams.dstX0 = xy[0] - pixelPhase;        /* dstX0 & dstY0 appropriate for sampling on integers */
                            cuParams.dstY0 = xy[1] - pixelPhase;
                        }
                        if (relToCenter)
                        {
                            cuParams.dstX0 = xy[0] + (cuParams.dstWidth  - 1) * .5f;
                            cuParams.dstY0 = xy[1] + (cuParams.dstHeight - 1) * .5f;
                        }
                    }
    void            getDstPrincipalPoint(float xy[2], uint32_t relToCenter) const
                    {
                        if (!relToCenter)
                        {
                            xy[0] = cuParams.dstX0 + pixelPhase;        /* dstX0 & dstY0 appropriate for sampling on integers */
                            xy[1] = cuParams.dstY0 + pixelPhase;
                        }
                        if (relToCenter)
                        {
                            xy[0] = cuParams.dstX0 - (cuParams.dstWidth  - 1) * 0.5f;
                            xy[1] = cuParams.dstY0 - (cuParams.dstHeight - 1) * 0.5f;
                        }
                    }
    void            setDistortion(const float d[5])
                    {
                        if (d)  memcpy(cuParams.srcDist, d, sizeof(cuParams.srcDist));      /* Set specified distortion, ... */
                        else    memset(cuParams.srcDist, 0, sizeof(cuParams.srcDist));      /* or (0,0,0,0,0) if NULL */
                        Compute6InverseRadialPadeCoefficients(cuParams.srcDist, padeInverse);
                    }
    void            getDistortion(float d[5]) const
                    {
                        memcpy(d, cuParams.srcDist, sizeof(cuParams.srcDist));
                    }
    void            setControl(uint32_t index, float value)
                    {
                        if (index != 0)
                            return;
                        control = value;
                        cuParams.control = (NVSURF_PUSHBROOM == dstType) ? control * control : control;
                    }
    float           getControl(uint32_t index) const
                    {
                        if (index > 0)
                            return NAN;
                        return control;
                    }
    nvwarpResult    setWarpType(nvwarpType_t t)
                    {
                        type     = t;
                        dispatch = GetDispatch(t);
                        srcType  = GET_SRC_TYPE(type);
                        dstType  = GET_DST_TYPE(type);
                        return dispatch ? NVWARP_SUCCESS : NVWARP_ERR_UNIMPLEMENTED;
                    }
    cudaError_t     warp(cudaStream_t stream, cudaTextureObject_t srcTex, cudaSurfaceObject_t dstSurface) const
                    {
                        dim3 dimGrid;

                        if (!dispatch)
                            return cudaErrorInvalidSymbol;  /* The warper does not have a valid warp type */
                        ComputeDimGrid(cuParams.dstWidth, cuParams.dstHeight, dimBlock, dimGrid);
                        return dispatch->warpSurface(dimGrid, dimBlock, stream, &cuParams, srcTex, dstSurface);
                    }
    cudaError_t     warp(cudaStream_t stream, cudaTextureObject_t srcTex, void *dstAddr, size_t dstRowBytes) const
                    {
                        dim3 dimGrid;

                        if (!dispatch)
                            return cudaErrorInvalidSymbol;  /* The warper does not have a valid warp type */
                        ComputeDimGrid(cuParams.dstWidth, cuParams.dstHeight, dimBlock, dimGrid);
                        return dispatch->warpBuffer(dimGrid, dimBlock, stream, &cuParams, srcTex, dstAddr, dstRowBytes);
                    }
    nvwarpResult    srcToRay(uint32_t numPts, const float *pts2D, float *rays3D) const;
    nvwarpResult    dstToRay(uint32_t numPts, const float *pts2D, float *rays3D) const;
    nvwarpResult    srcFromRay(uint32_t numRays, const float *rays3D, float *pts2D) const;
    nvwarpResult    dstFromRay(uint32_t numRays, const float *rays3D, float *pts2D) const;
};


/********************************************************************************
 * Set Euler rotation
 ********************************************************************************/

void nvwarpObject::setEulerRotation(const float *angles, const char *axes)
{
    if (!angles || !axes)
    {
        setRotationMatrix(nullptr);             /* Identity matrix if angles or axes is NULL */
        return;
    }
    SetEulerRotation(angles, axes, cuParams.srcRot);
}


/********************************************************************************
 * Set destination focal length and center of projection
 ********************************************************************************/

nvwarpResult nvwarpObject::setDstFocalLengthCOP(float topAng, float botAng, uint32_t dstWidth, uint32_t dstHeight)
{
    float focLen;
    topAngle = topAng;
    botAngle = botAng;
    cuParams.dstWidth  = dstWidth;
    cuParams.dstHeight = dstHeight;
    cuParams.dstX0 = (cuParams.dstWidth - 1) * .5f; /* This is valid for all but equirect, which is corrected later */
    switch (dstType)
    {   /* Compute the focal length from the view angles */
        case NVSURF_PUSHBROOM:
        case NVSURF_PANINI:
        case NVSURF_CYLINDER:
        case NVSURF_PERSPECTIVE:
            focLen = ComputeFlatFocalLengthFromVerticalViewAngles(topAngle, botAngle, cuParams.dstHeight - 1, &cuParams.dstY0);
            break;
        case NVSURF_ROTCYLINDER:
        case NVSURF_FISHEYE:
            focLen = ComputeRoundFocalLengthFromVerticalViewAngles(topAngle, botAngle, cuParams.dstHeight - 1, &cuParams.dstY0);
            break;
        case NVSURF_EQUIRECT:
            focLen = ComputeEquirectFocalLengthFromVerticalViewAngles(topAngle, botAngle, cuParams.dstHeight - 1, &cuParams.dstY0,
                cuParams.dstWidth, &cuParams.dstX0);
            break;
        case NVSURF_STEREOGRAPHIC:
            focLen = ComputeStereographicFocalLengthFromVerticalViewAngles(topAngle, botAngle, cuParams.dstHeight - 1, &cuParams.dstY0, control);
            break;
        default:
            return NVWARP_ERR_UNIMPLEMENTED;
    }
    setDstFocalLength(focLen, focLen);
    return NVWARP_SUCCESS;
}


/********************************************************************************
 * Set from nvwarpParams_t
 ********************************************************************************/

nvwarpResult nvwarpObject::setWarp360Params(const nvwarpParams_t *params)
{
    if (!params)
    {
        init();                         /* Initialize if NULL passed as a parameter */
        return NVWARP_ERR_PARAMETER;
    }
    setWarpType(params->type);
    srcWidth            = params->srcWidth;
    srcHeight           = params->srcHeight;
    srcRadius           = params->srcRadius;
    cuParams.srcX0      = params->srcX0 + .5f - pixelPhase;   /* The 0.5 offset accounts for the coordinate system used for bilinear interpolation */
    cuParams.srcY0      = params->srcY0 + .5f - pixelPhase;
    cuParams.srcFocLenX = params->srcFocalLen;
    cuParams.srcFocLenY = params->srcFocalLen;
    userData            = params->userData;
    setControl(0, params->control[0]);
    if (params->rotAxes[3] != 0)
        (const_cast<nvwarpParams_t *>(params))->rotAxes[3] = 0;
    SetEulerRotation(params->rotAngles, params->rotAxes, cuParams.srcRot);
    memset(cuParams.srcDist, 0, sizeof(cuParams.srcDist));
    memcpy(cuParams.srcDist, params->dist, sizeof(params->dist));
    Compute6InverseRadialPadeCoefficients(cuParams.srcDist, padeInverse);
    setDstFocalLengthCOP(params->topAngle, params->bottomAngle, params->dstWidth, params->dstHeight);

    return NVWARP_SUCCESS;
}


/********************************************************************************
 * Set from Warp360Params
 ********************************************************************************/

cudaError_t nvwarpObject::setWarp360Params(const Warp360Params *params)
{
    if (!params)
    {
        init();                         /* Initialize if NULL passed as a parameter */
        return cudaErrorMisalignedAddress;
    }
    setWarpType((nvwarpType_t)params->type);
    srcWidth            = params->srcWidth;
    srcHeight           = params->srcHeight;
    srcRadius           = params->srcRadius;
    cuParams.srcX0      = params->srcX0 + .5f - pixelPhase;   /* The 0.5 offset accounts for the coordinate system used for bilinear interpolation */
    cuParams.srcY0      = params->srcY0 + .5f - pixelPhase;
    cuParams.srcFocLenX = params->srcFocalLen;
    cuParams.srcFocLenY = params->srcFocalLen;
    userData            = params->userData;
    setControl(0, params->control[0]);
    float rpy[3] = { params->roll, params->pitch, params->yaw };
    setEulerRotation(rpy, "ZXY");
    memset(cuParams.srcDist, 0, sizeof(cuParams.srcDist));
    memcpy(cuParams.srcDist, params->srcDist, sizeof(params->srcDist));
    Compute6InverseRadialPadeCoefficients(cuParams.srcDist, padeInverse);
    setDstFocalLengthCOP(params->topAngle, params->bottomAngle, params->dstWidth, params->dstHeight);

    return cudaSuccess;
}



#define USE_PADE_INVERSE_DISTORTION
/********************************************************************************
 * Warp coordinates
 ********************************************************************************/

nvwarpResult nvwarpObject::warp(uint32_t numPts, const float *inPtsXY, float *outPtsXY) const
{
    nvwarpResult    err         = NVWARP_SUCCESS;
    const float     invSrcFLX   = 1.f / cuParams.srcFocLenX,
                    invSrcFLY   = 1.f / cuParams.srcFocLenY,
                    dstX0       = cuParams.dstX0 + pixelPhase,          /* Since we store dstXY0 appropriate for pixel phase = 0, ... */
                    dstY0       = cuParams.dstY0 + pixelPhase,          /* ... adjust the phase here */
                    *M          = cuParams.srcRot;
    float           srcX0       = cuParams.srcX0 + pixelPhase - .5f,    /* Since we store srcXY0 appropriate for pixel phase = 0.5, ... */
                    srcY0       = cuParams.srcY0 + pixelPhase - .5f;    /* ... adjust the phase here */
#ifndef USE_PADE_INVERSE_DISTORTION
    const double    coeff[4]    = { _userParams.srcDist[0], _userParams.srcDist[1], _userParams.srcDist[2], _userParams.srcDist[3] };
#endif /* USE_PADE_INVERSE_DISTORTION */
    unsigned        numCoeff    = 0;
    bool            isNorm      = false;

    switch (srcType)
    {
        case NVSURF_FISHEYE:
            if (cuParams.srcDist[3] || cuParams.srcDist[2] || cuParams.srcDist[1] || cuParams.srcDist[0])
                numCoeff = 4;
            break;
        case NVSURF_PERSPECTIVE:
            if (cuParams.srcDist[2] || cuParams.srcDist[1] || cuParams.srcDist[0])
                numCoeff = 3;
            if (cuParams.srcDist[3] || cuParams.srcDist[4]) /* If tangential distortion is used, ... */
                err = NVWARP_ERR_APPROXIMATE;               /* ... signal that the result is not accurate */
            break;
        default:
            break;
    }

    for (; numPts--; inPtsXY += 2, outPtsXY += 2)
    {
        float ray[3], xyz[3], r, t;
        bool isOutOfRange = false;
        ray[0] = (inPtsXY[0] - srcX0) * invSrcFLX;
        ray[1] = (inPtsXY[1] - srcY0) * invSrcFLY;
        r = hypotf(ray[0], ray[1]);

        switch (srcType)
        {
            case NVSURF_PERSPECTIVE:
                isNorm = false;
                if (r)
                {
                    #ifndef USE_PADE_INVERSE_DISTORTION
                        t = numCoeff ? (float)KEFindRootOfDistortionPolynomial(numCoeff, coeff, 10., r) / r : 1.f;
                    #else /* USE_PADE_INVERSE_DISTORTION */
                        t = numCoeff ? EvaluatePade6(r, padeInverse) / r : 1.f;   /* error < 6.3e-4 @ 1.873 radians. */
                    #endif /* USE_PADE_INVERSE_DISTORTION */
                    // TODO: Accommodate tangential distortion
                    ray[0] *= t;
                    ray[1] *= t;
                }
                ray[2] = 1.f;
                break;
            case NVSURF_FISHEYE:
                isNorm = true;
                if (r)
                {
                    #ifndef USE_PADE_INVERSE_DISTORTION
                        t = numCoeff ? (float)KEFindRootOfDistortionPolynomial(numCoeff, coeff, 10., r) : r;
                    #else /* USE_PADE_INVERSE_DISTORTION */
                        t = numCoeff ? EvaluatePade6(r, padeInverse) : r;   /* error < 6.3e-4 @ 1.873 radians. */
                    #endif /* USE_PADE_INVERSE_DISTORTION */
                    r = sinf(t) / r;
                    ray[0] *= r;
                    ray[1] *= r;
                    ray[2] = cosf(t);
                }
                else
                {
                    ray[2] = 1.f;
                }
                break;
            case NVSURF_EQUIRECT:
                isNorm = true;
                t = cosf(ray[1]);
                ray[2] = cosf(ray[0]) * t;
                ray[0] = sinf(ray[0]) * t;
                ray[1] = sinf(ray[1]);
                break;
            default:
                ray[2] = NAN;	/* This will propagate NaN to all of xyz */
                err = NVWARP_ERR_UNIMPLEMENTED;
        }

        xyz[0] = (float)((double)ray[0] * M[0] + (double)ray[1] * M[1] + (double)ray[2] * M[2]); /* Apply inverse rotation */
        xyz[1] = (float)((double)ray[0] * M[3] + (double)ray[1] * M[4] + (double)ray[2] * M[5]);
        xyz[2] = (float)((double)ray[0] * M[6] + (double)ray[1] * M[7] + (double)ray[2] * M[8]);

        switch (dstType)
        {
            case NVSURF_PERSPECTIVE:
                if (!(xyz[2] > 0.f))
                {
                    err = NVWARP_ERR_DOMAIN;
                    isOutOfRange = true;
                    break;
                }
                outPtsXY[0] = xyz[0] / xyz[2];
                outPtsXY[1] = xyz[1] / xyz[2];
                break;
            case NVSURF_EQUIRECT:
                outPtsXY[1] = atan2f(xyz[1], sqrtf(xyz[0] * xyz[0] + xyz[2] * xyz[2]));
                outPtsXY[0] = atan2f(xyz[0], xyz[2]);
                break;
            case NVSURF_ROTCYLINDER:
                 outPtsXY[0] = xyz[0] / hypotf(xyz[1], xyz[2]);
                 outPtsXY[1] = atan2f(xyz[1], xyz[2]);
                 break;
            case NVSURF_CYLINDER:
                 outPtsXY[0] = atan2f(xyz[0], xyz[2]);
                 outPtsXY[1] = xyz[1] / hypotf(xyz[0], xyz[2]);
                 break;
            case NVSURF_FISHEYE:
                if (0 != (r = hypotf(xyz[0], xyz[1])))
                    r = atan2f(r, xyz[2]) / r;
                outPtsXY[0] = xyz[0] * r;
                outPtsXY[1] = xyz[1] * r;
                break;
            case NVSURF_PANINI:
                if (!isNorm)
                {
                    t = 1.f / sqrtf(xyz[0] * xyz[0] + xyz[1] * xyz[1] + xyz[2] * xyz[2]);
                    xyz[0] *= t;    xyz[1] *= t;    xyz[2] *= t;
                }   /* xyz = { sin(lon) * cos(lat), sin(lat), cos(lon) * cos(lat) } */
                r = (1.f + control) / (xyz[2] + control * sqrtf(xyz[0] * xyz[0] + xyz[2] * xyz[2]));
                outPtsXY[0] = xyz[0] * r;
                outPtsXY[1] = xyz[1] * r;
                break;
            case NVSURF_PUSHBROOM:
                if (!(xyz[2] > 0.f))
                {
                    err = NVWARP_ERR_DOMAIN;
                    isOutOfRange = true;
                    break;
                }
                outPtsXY[1] = xyz[1] / xyz[2];
                t = control * xyz[0] / xyz[2];
                outPtsXY[0] = (float)(sqrt(sqrt((double)t * t + .25) - .5) / control);
                if (t < 0)
                    outPtsXY[0] = -outPtsXY[0];
                break;
            case NVSURF_STEREOGRAPHIC:
                r = hypotf(xyz[0], xyz[1]);     /* Radius */
                t = atan2f(r, xyz[2]);          /* Inclination angle */
                if (r)
                    r = (1.f + control) * sinf(t) / ((cosf(t) + control) * r);
                outPtsXY[0] = xyz[0] * r;       /* Adjust scale of X and Y */
                outPtsXY[1] = xyz[1] * r;
                break;
            default:
                err = NVWARP_ERR_UNIMPLEMENTED;
                isOutOfRange = true;
                break;
        }
        if (!isOutOfRange)
        {
            outPtsXY[0] = outPtsXY[0] * dstFocLenX + dstX0;
            outPtsXY[1] = outPtsXY[1] * dstFocLenY + dstY0;
        }
        if (isOutOfRange || (0&&!(0.f <= outPtsXY[0] && outPtsXY[0] < cuParams.dstWidth && 0.f <= outPtsXY[1] && outPtsXY[1] < cuParams.dstHeight)))
        {
            outPtsXY[0] = NAN;
            outPtsXY[1] = NAN;
        }
    }
    return err;
}


/********************************************************************************
 * Inverse Warp of coordinates
 ********************************************************************************/

nvwarpResult nvwarpObject::inverseWarp(uint32_t numPts, const float *inPtsXY, float *outPtsXY) const
{
    nvwarpResult    err         = NVWARP_SUCCESS;
    const float     dstX0       = cuParams.dstX0 + pixelPhase,          /* Since we store dstXY0 without a bias, we add the appropriate bias here */
                    dstY0       = cuParams.dstY0 + pixelPhase,
                    *M          = cuParams.srcRot,
                    *coeff      = cuParams.srcDist,
                    srcFocLenX  = cuParams.srcFocLenX,
                    srcFocLenY  = cuParams.srcFocLenY;
    float           srcX0       = cuParams.srcX0 + pixelPhase - .5f,    /* Since we store srcXY0 biased for a 0.5 pixel phase, ... */
                    srcY0       = cuParams.srcY0 + pixelPhase - .5f;    /* ... remove it for phase=0 */

    for (; numPts--; inPtsXY += 2, outPtsXY += 2)
    {
        float ray[3], xyz[3], r, t;
        bool isOutOfRange = false;
        ray[0] = (inPtsXY[0] - dstX0) * cuParams.dstInvFocLenX;
        ray[1] = (inPtsXY[1] - dstY0) * cuParams.dstInvFocLenY;

        switch (dstType)
        {
            case NVSURF_PERSPECTIVE:
                ray[2] = 1.f;
                break;
            case NVSURF_FISHEYE:
                t = hypotf(ray[0], ray[1]);
                ray[2] = cosf(t);
                r = t ? (sinf(t) / t) : 1.f;
                ray[0] *= r;
                ray[1] *= r;
                break;
            case NVSURF_EQUIRECT:
                t = ray[0];                                                     /* Azimuth angle */
                ray[0] = sinf(t);                                               /* Radial vector */
                ray[2] = cosf(t);
                ray[1] = tanf(ray[1]);                                          /* Tilt angle */
                break;
            case NVSURF_ROTCYLINDER:
                t = ray[1];                                                     /* Radial angle */
                ray[1] = sinf(t);                                               /* Vertical coordinate */
                ray[2] = cosf(t);                                               /* Depth coordinate */
                break;
            case NVSURF_CYLINDER:
                t = ray[0];                                                     /* Temporary azimuth angle */
                ray[0] = sinf(t);                                               /* Radial vector */
                ray[2] = cosf(t);
                break;
            case NVSURF_PANINI:
            {   float scale, cosLon;
                scale = 1.f / (control + 1.f);
                ray[0] *= scale;
                ray[1] *= scale;
                if (ray[0] == 0)
                {
                    cosLon = 1.f;
                }
                else
                {
                    double  kk = (double)ray[0] * ray[0],
                            del = (1. - (double)control * control) * kk + 1.;
                    if (del < 0)
                        err = NVWARP_ERR_DOMAIN;                                            /* NANs will be generated below */
                    cosLon = (float)((-kk * control + sqrt(del)) / (kk + 1.));
                }

                scale = control + cosLon;                                                   /* Project to the unit circle */
                t = ray[1] * scale;                                                         /* tan(latitude) */
                r = 1.f / sqrtf(1.f + t * t);                                               /* cos(latitude) */
                ray[1] = t * r;                                                             /* sin(latitude) */
                ray[0] = ray[0] * scale * r;                                                /* sin(longitude) * cos(latitude) */
                ray[2] = cosLon * r;                                                        /* cos(longitude) * cos(latitude) */
            }   break;
            case NVSURF_PUSHBROOM:
                ray[0] *= sqrtf(ray[0] * ray[0] * control * control + 1.f);
                ray[2] = 1.f;
                break;
            case NVSURF_STEREOGRAPHIC:
            {   t = 1.f / (control + 1.f);                                                  /* Inverse viewing distance */
                ray[0] *= t;                                                                /* Adjust ray coordinates for the distance */
                ray[1] *= t;
                r = ray[0] * ray[0] + ray[1] * ray[1];                                      /* Magnitude squared */
                if (r)
                {
                    double d = (1.0 - control * control) * r + 1.0;                         /* Discriminant */
                    if (d < 0.0)
                        err = NVWARP_ERR_DOMAIN;                                            /* Out-of-bounds */
                    ray[2] = (float)((sqrt(d) - control * r) / (1.0 + r));                  /* Cosine of inclination angle */
                    r = sqrtf((1.f - ray[2] * ray[2]) / r);                                 /* Compute the sine and divide by the magnitude */
                    ray[0] *= r;                                                            /* Normalized ray */
                    ray[1] *= r;
                }
                else
                {
                    ray[2] = 1.f;                                                           /* Exactly the center */
                }
            }   break;
            default:
                ray[2] = NAN;	/* This NaN will be propagated to all components of xyz */
                err = NVWARP_ERR_UNIMPLEMENTED;
        }

        xyz[0] = (float)((double)ray[0] * M[0] + (double)ray[1] * M[3] + (double)ray[2] * M[6]);    /* Apply rotation */
        xyz[1] = (float)((double)ray[0] * M[1] + (double)ray[1] * M[4] + (double)ray[2] * M[7]);
        xyz[2] = (float)((double)ray[0] * M[2] + (double)ray[1] * M[5] + (double)ray[2] * M[8]);

        switch (srcType)
        {
            case NVSURF_PERSPECTIVE:
                if (!(xyz[2] > 0.))
                {
                    err = NVWARP_ERR_DOMAIN;
                    isOutOfRange = true;
                    break;
                }
                t = 1.f / xyz[2];
                xyz[0] *= t;     /* Project to the plane */
                xyz[1] *= t;
                {
                    float r2 = xyz[0] * xyz[0] + xyz[1] * xyz[1];
                    r = ((coeff[2] * r2 + coeff[1]) * r2 + coeff[0]) * r2 + 1.f,
                    t = xyz[0] * xyz[1] * 2.f;
                    outPtsXY[0] = r * xyz[0] + coeff[3] * t + coeff[4] * (r2 + 2.f * xyz[0] * xyz[0]);
                    outPtsXY[1] = r * xyz[1] + coeff[4] * t + coeff[3] * (r2 + 2.f * xyz[1] * xyz[1]);
                }
                break;
            case NVSURF_FISHEYE:
                t = r = hypotf(xyz[0], xyz[1]);                                             /* Radius */
                if (r)
                {
                    t = atan2f(t, xyz[2]);                                                  /* Inclination angle */
                    float t2 = t * t;
                    t *= ((((coeff[3] * t2 + coeff[2]) * t2 + coeff[1]) * t2 + coeff[0]) * t2 + 1.f) / r; /* Distort */
                }
                outPtsXY[0] = xyz[0] * t;
                outPtsXY[1] = xyz[1] * t;
                break;
            case NVSURF_EQUIRECT:
                outPtsXY[1] = atan2f(xyz[1], sqrtf(xyz[0] * xyz[0] + xyz[2] * xyz[2]));
                outPtsXY[0] = atan2f(xyz[0], xyz[2]);
                break;
            default:
                err = NVWARP_ERR_UNIMPLEMENTED;
        }
        if (!isOutOfRange)
        {
            outPtsXY[0] = outPtsXY[0] * srcFocLenX + srcX0;
            outPtsXY[1] = outPtsXY[1] * srcFocLenY + srcY0;
        }
        if (isOutOfRange || (0&&!(0.f <= outPtsXY[0] && outPtsXY[0] < srcWidth && 0.f <= outPtsXY[1] && outPtsXY[1] < srcHeight)))
        {
            outPtsXY[0] = NAN;
            outPtsXY[1] = NAN;
        }
    }
    return err;
}


/********************************************************************************
 * NormalizeRay
 ********************************************************************************/

static void NormalizeRay(float ray[3])
{
    float r;
    if (0.f != (r = ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2]))
    {
        r = 1.f / sqrtf(r);
        ray[0] *= r;
        ray[1] *= r;
        ray[2] *= r;
    }
}


// TODO: It would be nice to refactor these and coordinate warps to avoid duplication of common code.

/********************************************************************************
 * srcToRay
 ********************************************************************************/

nvwarpResult nvwarpObject::srcToRay(uint32_t numPts, const float *pt, float *ray) const
{
    nvwarpResult    err         = NVWARP_SUCCESS;
    float           srcX0       = cuParams.srcX0,
                    srcY0       = cuParams.srcY0;
#ifndef USE_PADE_INVERSE_DISTORTION
    const double    coeff[4]    = { _userParams.srcDist[0], _userParams.srcDist[1], _userParams.srcDist[2], _userParams.srcDist[3] };
#endif /* USE_PADE_INVERSE_DISTORTION */


    if (!(NVSURF_EQUIRECT == srcType && srcHeight * 2 == srcWidth))
    {
        srcX0 += pixelPhase - .5f;
        srcY0 += pixelPhase - .5f;
    }
    if (NVSURF_PERSPECTIVE == srcType && (cuParams.srcDist[3] || cuParams.srcDist[4]))
        err = NVWARP_ERR_APPROXIMATE;

    for (; numPts--; pt += 2, ray += 3)
    {
        float r, t;
        ray[0] = (pt[0] - srcX0) / cuParams.srcFocLenX;
        ray[1] = (pt[1] - srcY0) / cuParams.srcFocLenY;
        r = hypotf(ray[0], ray[1]);

        switch (srcType)
        {
            case NVSURF_PERSPECTIVE:
                if (r)
                {
                    #ifndef USE_PADE_INVERSE_DISTORTION
                        t = numCoeff ? (float)KEFindRootOfDistortionPolynomial(numCoeff, coeff, 10., r) / r : 1.f;
                    #else /* USE_PADE_INVERSE_DISTORTION */
                        t = EvaluatePade6(r, padeInverse) / r;   /* error < 6.3e-4 @ 1.873 radians. */
                    #endif /* USE_PADE_INVERSE_DISTORTION */
                    ray[0] *= t;
                    ray[1] *= t;
                }
                ray[2] = 1.f;
                // TODO: Add tangential distortion
                NormalizeRay(ray);
                break;
            case NVSURF_FISHEYE:
                if (r)
                {
                    #ifndef USE_PADE_INVERSE_DISTORTION
                        t = numCoeff ? (float)KEFindRootOfDistortionPolynomial(numCoeff, coeff, 10., r) : r;
                    #else /* USE_PADE_INVERSE_DISTORTION */
                        t = EvaluatePade6(r, padeInverse) ;   /* error < 6.3e-4 @ 1.873 radians. */
                    #endif /* USE_PADE_INVERSE_DISTORTION */
                    r = sinf(t) / r;
                    ray[0] *= r;
                    ray[1] *= r;
                    ray[2] = cosf(t);
                }
                else
                {
                    ray[2] = 1.f;
                }
                break;
            case NVSURF_EQUIRECT:
                t = cosf(ray[1]);
                ray[2] = cosf(ray[0]) * t;
                ray[0] = sinf(ray[0]) * t;
                ray[1] = sinf(ray[1]);
                break;
            default:
                ray[0] = ray[1] = ray[2] = NAN;
                err = NVWARP_ERR_UNIMPLEMENTED;
                break;
        }
    }
    return err;
}


/********************************************************************************
 * srcFromRay
 ********************************************************************************/

nvwarpResult nvwarpObject::srcFromRay(uint32_t numRays, const float *ray, float *pt) const
{
    nvwarpResult    err     = NVWARP_SUCCESS;
    const float     *coeff  = cuParams.srcDist;
    float           srcX0   = cuParams.srcX0,
                    srcY0   = cuParams.srcY0;
    float           r, t;

    if (!(NVSURF_EQUIRECT == srcType && srcHeight * 2 == srcWidth))
    {
        srcX0 += pixelPhase - .5f;
        srcY0 += pixelPhase - .5f;
    }

    for (; numRays--; ray += 3, pt += 2)
    {
        switch (srcType)
        {
            case NVSURF_PERSPECTIVE:
                if (!(ray[2] > 0.))
                {
                    pt[0] = NAN;
                    pt[1] = NAN;
                    err = NVWARP_ERR_DOMAIN;
                    break;
                }
                {
                    float xy[2], r2;
                    t = 1.f / ray[2];
                    xy[0] = ray[0] * t;     /* Project to the plane */
                    xy[1] = ray[1] * t;
                    r2 = xy[0] * xy[0] + xy[1] * xy[1];
                    r = ((coeff[2] * r2 + coeff[1]) * r2 + coeff[0]) * r2 + 1.f,
                    t = xy[0] * xy[1] * 2.f;
                    pt[0] = r * xy[0] + coeff[3] * t + coeff[4] * (r2 + 2.f * xy[0] * xy[0]);
                    pt[1] = r * xy[1] + coeff[4] * t + coeff[3] * (r2 + 2.f * xy[1] * xy[1]);
                }
                break;
            case NVSURF_FISHEYE:
                t = r = hypotf(ray[0], ray[1]);                                             /* Radius */
                if (r)
                {
                    t = atan2f(t, ray[2]);                                                  /* Inclination angle */
                    float t2 = t * t;
                    t *= ((((coeff[3] * t2 + coeff[2]) * t2 + coeff[1]) * t2 + coeff[0]) * t2 + 1.f) / r; /* Distort */
                }
                pt[0] = ray[0] * t;
                pt[1] = ray[1] * t;
                break;
            case NVSURF_EQUIRECT:
                pt[1] = atan2f(ray[1], sqrtf(ray[0] * ray[0] + ray[2] * ray[2]));
                pt[0] = atan2f(ray[0], ray[2]);
                break;
            default:
                pt[0] = NAN;
                pt[1] = NAN;
                err = NVWARP_ERR_DOMAIN;
                break;
        }
        pt[0] = pt[0] * cuParams.srcFocLenX + srcX0;
        pt[1] = pt[1] * cuParams.srcFocLenY + srcY0;
    }

    return err;
}


/********************************************************************************
 * dstToRay
 ********************************************************************************/

nvwarpResult nvwarpObject::dstToRay(uint32_t numPts, const float *pt, float *ray) const
{
    nvwarpResult    err     = NVWARP_SUCCESS;
    const float     dstX0   = cuParams.dstX0 + pixelPhase,
                    dstY0   = cuParams.dstY0 + pixelPhase;
    float           r, t;

    for (; numPts--; pt += 2, ray += 3)
    {
        ray[0] = (pt[0] - dstX0) * cuParams.dstInvFocLenX;
        ray[1] = (pt[1] - dstY0) * cuParams.dstInvFocLenY;

        switch (dstType)
        {
            case NVSURF_PERSPECTIVE:
                ray[2] = 1.f;
                NormalizeRay(ray);
                break;
            case NVSURF_FISHEYE:
                t = hypotf(ray[0], ray[1]);
                ray[2] = cosf(t);
                r = t ? (sinf(t) / t) : 1.f;
                ray[0] *= r;
                ray[1] *= r;
                break;
            case NVSURF_EQUIRECT:
                t = cosf(ray[1]);                                               /* Tilt angle cos */
                ray[2] = cosf(ray[0]) * t;
                ray[0] = sinf(ray[0]) * t;                                      /* Radial vector */
                ray[1] = sinf(ray[1]);                                          /* Tilt angle sin */
                break;
            case NVSURF_ROTCYLINDER:
                t = ray[1];                                                     /* Radial angle */
                ray[1] = sinf(t);                                               /* Vertical coordinate */
                ray[2] = cosf(t);                                               /* Depth coordinate */
                NormalizeRay(ray);
                break;
            case NVSURF_CYLINDER:
                t = ray[0];                                                     /* Temporary azimuth angle */
                ray[0] = sinf(t);                                               /* Radial vector */
                ray[2] = cosf(t);
                NormalizeRay(ray);
                break;
            case NVSURF_PANINI:
            {   float scale, cosLon;
                scale = 1.f / (control + 1.f);
                ray[0] *= scale;
                ray[1] *= scale;
                if (ray[0] == 0)
                {
                    cosLon = 1.f;
                }
                else
                {
                    double  kk = (double)ray[0] * ray[0],
                            del = (1. - (double)control * control) * kk + 1.;
                    if (del < 0)
                        err = NVWARP_ERR_DOMAIN;                                            /* NaNs will be generated below */
                    cosLon = (float)((-kk * control + sqrt(del)) / (kk + 1.));
                }

                scale = control + cosLon;                                                   /* Project to the unit circle */
                t = ray[1] * scale;                                                         /* tan(latitude) */
                r = 1.f / sqrtf(1.f + t * t);                                               /* cos(latitude) */
                ray[1] = t * r;                                                             /* sin(latitude) */
                ray[0] = ray[0] * scale * r;                                                /* sin(longitude) * cos(latitude) */
                ray[2] = cosLon * r;                                                        /* cos(longitude) * cos(latitude) */
            }   break;
            case NVSURF_PUSHBROOM:
                ray[0] *= sqrtf(ray[0] * ray[0] * control * control + 1.f);
                ray[2] = 1.f;
                NormalizeRay(ray);
                break;
            case NVSURF_STEREOGRAPHIC:
            {   t = 1.f / (control + 1.f);                                                  /* Inverse viewing distance */
                ray[0] *= t;                                                                /* Adjust ray coordinates for the distance */
                ray[1] *= t;
                r = ray[0] * ray[0] + ray[1] * ray[1];                                      /* Magnitude squared */
                if (r)
                {
                    double d = (1.0 - control * control) * r + 1.0;                         /* Discriminant */
                    if (d < 0.0)
                        err = NVWARP_ERR_DOMAIN;                                            /* Out-of-bounds: NaNs will be generated below */
                    ray[2] = (float)((sqrt(d) - control * r) / (1.0 + r));                  /* Cosine of inclination angle */
                }
                else
                {
                    ray[2] = 1.f;                                                           /* Exactly the center */
                }
                r = sqrtf((1.f - ray[2] * ray[2]) / r);                                     /* Compute the sine and divide by the magnitude */
                ray[0] *= r;                                                                /* Normalized ray */
                ray[1] *= r;
            }   break;
            default:
                ray[0] = ray[1] = ray[2] = NAN;
                err = NVWARP_ERR_UNIMPLEMENTED;
                break;
        }
    }

    return err;
}


/********************************************************************************
 * dstFromRay
 ********************************************************************************/

nvwarpResult nvwarpObject::dstFromRay(uint32_t numRays, const float *ray, float *pt) const
{
    nvwarpResult    err         = NVWARP_SUCCESS;
    const float     dstX0       = cuParams.dstX0 + pixelPhase,
                    dstY0       = cuParams.dstY0 + pixelPhase;
    float           r, t;
#ifndef USE_PADE_INVERSE_DISTORTION
    const double    coeff[4]    = { _userParams.srcDist[0], _userParams.srcDist[1], _userParams.srcDist[2], _userParams.srcDist[3] };
#endif /* USE_PADE_INVERSE_DISTORTION */

    for ( ; numRays--; ray += 3, pt += 2)
    {
        bool isOutOfRange = false;
        switch (dstType)
        {
            case NVSURF_PERSPECTIVE:
                if (!(ray[2] > 0.f))
                {
                    isOutOfRange = true;
                    break;
                }
                pt[0] = ray[0] / ray[2];
                pt[1] = ray[1] / ray[2];
                break;
            case NVSURF_EQUIRECT:
                pt[0] = atan2f(ray[0], ray[2]);
                pt[1] = atan2f(ray[1], sqrtf(ray[0] * ray[0] + ray[2] * ray[2]));
                break;
            case NVSURF_ROTCYLINDER:
                pt[0] = ray[0] / hypotf(ray[1], ray[2]);
                pt[1] = atan2f(ray[1], ray[2]);
                break;
            case NVSURF_CYLINDER:
                pt[0] = atan2f(ray[0], ray[2]);
                pt[1] = ray[1] / hypotf(ray[0], ray[2]);
                break;
            case NVSURF_FISHEYE:
                if (0 != (r = hypotf(ray[0], ray[1])))
                    r = atan2f(r, ray[2]) / r;
                pt[0] = ray[0] * r;
                pt[1] = ray[1] * r;
                break;
            case NVSURF_PANINI:
            {
                t = 1.f / sqrtf(ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2]);
                float xyz[3] = { ray[0] * t, ray[1] * t, ray[2] * t };
                r = (1.f + control) / (xyz[2] + control * sqrtf(xyz[0] * xyz[0] + xyz[2] * xyz[2]));
                pt[0] = ray[0] * r;
                pt[1] = ray[1] * r;
            }    break;
            case NVSURF_PUSHBROOM:
                if (!(ray[2] > 0.f))
                {
                    isOutOfRange = true;
                    break;
                }
                pt[1] = ray[1] / ray[2];
                t = control * ray[0] / ray[2];
                pt[0] = (float)(sqrt(sqrt((double)t * t + .25) - .5) / control);
                if (t < 0)
                    pt[0] = -pt[0];
                break;
            case NVSURF_STEREOGRAPHIC:
                r = hypotf(ray[0], ray[1]);     /* Radius */
                t = atan2f(r, ray[2]);          /* Inclination angle */
                if (r)
                    r = (1.f + control) * sinf(t) / ((cosf(t) + control) * r);
                pt[0] = ray[0] * r;             /* Adjust scale of X and Y */
                pt[1] = ray[1] * r;
                break;
            default:
                pt[0] = pt[1] = NAN;
                err = NVWARP_ERR_UNIMPLEMENTED;
                continue;
        }
        if (isOutOfRange)
        {
            pt[0] = NAN;
            pt[1] = NAN;
            err = NVWARP_ERR_DOMAIN;
        }
        else
        {
            pt[0] = pt[0] * dstFocLenX + dstX0;
            pt[1] = pt[1] * dstFocLenY + dstY0;
        }
    }

    return err;
}


/********************************************************************************
 * Constructor
 ********************************************************************************/

Warp360::Warp360()
{
    _impl = new nvwarpObject;
}


/********************************************************************************
 * Destructor
 ********************************************************************************/

Warp360::~Warp360()
{
    if (_impl)
        delete _impl;
}


/********************************************************************************
 * Binary copyright
 ********************************************************************************/

extern const char Warp360Copyright[];
const char Warp360Copyright[] = "\0Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.";


/********************************************************************************
 * set Params
 ********************************************************************************/

cudaError_t Warp360::setParams(const Warp360Params *params)
{
    return _impl->setWarp360Params(params);
}


/********************************************************************************
 * get Params
 ********************************************************************************/

void Warp360::getParams(Warp360Params *params) const
{
    params->type        = (Warp360Params::WarpType)_impl->type;
    params->srcWidth    = _impl->srcWidth;
    params->srcHeight   = _impl->srcHeight;
    params->srcFocalLen = srcFocalLength();
    params->srcRadius   = _impl->srcRadius;
    params->dstWidth    = _impl->cuParams.dstWidth;
    params->dstHeight   = _impl->cuParams.dstHeight;
    params->topAngle    = _impl->topAngle;
    params->bottomAngle = _impl->botAngle;
    params->userData    = _impl->userData;
    params->control[0]  = getControl(0);
    memcpy(params->srcDist, _impl->cuParams.srcDist, sizeof(params->srcDist));
    _impl->getSrcPrincipalPoint(&params->srcX0, false);
    AnglesFromRotationMatrixYdown(_impl->cuParams.srcRot, &params->yaw, "ZXY");
    float t = params->yaw; params->yaw = params->roll; params->roll = t;
}


/********************************************************************************
 ********************************************************************************
 ********************************************************************************
 *****                              YUV --> RGB                             *****
 ********************************************************************************
 ********************************************************************************
 ********************************************************************************/


/********************************************************************************
 * ComputeYCbCr2RgbMatrix
 ********************************************************************************/

void ComputeYCbCr2RgbMatrix(int matrix_coefficients, int video_full_range_flag, int bit_depth, bool normalized_input, bool normalized_output, YUVRGBParams *p)
{
    nvwarpComputeYCbCr2RgbMatrix((nvwarpYUVRGBParams_t*)p, (uint32_t)matrix_coefficients, (uint32_t)video_full_range_flag, (uint32_t)bit_depth, (uint32_t)normalized_input, (uint32_t)normalized_output);
}


/********************************************************************************
 * YUVRGBParams::computeMatrix
 ********************************************************************************/

void YUVRGBParams::computeMatrix(int matrix_coefficients, int video_full_range_flag)
{
    nvwarpComputeYCbCr2RgbMatrix((nvwarpYUVRGBParams_t*)this, (uint32_t)matrix_coefficients, (uint32_t)video_full_range_flag, 8u, 0u, 1u);
}


/********************************************************************************
 * ConvertYUVNV12ToRGBA
 ********************************************************************************/

cudaError_t ConvertYUVNV12ToRGBA(cudaStream_t stream, const YUVRGBParams *params, const void *yuv, size_t yuvRowBytes, void *dst, size_t dstRowBytes)
{
    NV12RGBABuffer(stream, (nvwarpYUVRGBParams_t*)params, (const unsigned char*)yuv, yuvRowBytes, (uchar4*)dst, dstRowBytes);
    return cudaGetLastError();
}



/********************************************************************************
 ********************************************************************************
 ********************************************************************************
 *****      UTILITIES FOR FINDING THE ROOT OF A DISTORTION POLYNOMIAL       *****
 ********************************************************************************
 ********************************************************************************
 ********************************************************************************/

/********************************************************************************
 * Input odd order coefficients, get odd/even Pade rational polynomial.
 * @param[in]   dist  r + d0 * r^3 + d1 * r^5 + d2 * r^7 + d3 * r^9
 * @param[out]  pade  (r + p0 * r^3 + p2 * r^5 + p4 * r^7) / (1 + p1 * r^2 + p3 * r^4 + p5 * r^6)
 ********************************************************************************/
void Compute6InverseRadialPadeCoefficients(const float *dist, float *pade)
{
    double b[6];
    {
        double d02 = (double)dist[0] * dist[0];
        double d03 = (double)dist[0] * d02;
        double d04 = d02     * d02;
        double d05 = d02     * d03;
        double d06 = d03     * d03;
        double d12 = (double)dist[1] * dist[1];
        double d13 = (double)dist[1] * d12;
        double d22 = (double)dist[2] * dist[2];

        b[0] = -dist[0];
        b[1] = 3 * d02 - dist[1];
        b[2] = -12 * d03 + 8 * dist[0] * dist[1] - dist[2];
        b[3] = 55 * d04 - 55 * d02 * dist[1] + 5 * d12 + 10 * dist[0] * dist[2] - dist[3];
        b[4] = -273 * d05 + 364 * d03 * dist[1] - 78 * dist[0] * d12 - 78 * d02 * dist[2]
            + 12 * dist[1] * dist[2] + 12 * dist[0] * dist[3];
        b[5] = 7 * (204 * d06 - 340 * d04 * dist[1] - 5 * d13 + 80 * d03 * dist[2]
            - 30 * dist[0] * dist[1] * dist[2] + d22 + 15 * d02 * (8 * d12 - dist[3]) + 2 * dist[1] * dist[3]);
    }
    {
        double b02 = b[0] * b[0];
        double b12 = b[1] * b[1];
        double b13 = b[1] * b12;
        double b22 = b[2] * b[2];
        double b23 = b[2] * b22;
        double b32 = b[3] * b[3];
        double b33 = b[3] * b32;
        double b42 = b[4] * b[4];
        double den = b23 - 2 * b[1] * b[2] * b[3] + b[0] * b32 + b12 * b[4] - b[0] * b[2] * b[4];

        if (!den)
        {
            memset(pade, 0, sizeof(*pade) * 6);
            return;
        }

        /* Numerator coefficients */
        pade[0] = (float)((
            -(b22 * b[3]) + b[1] * b[2] * b[4] + b02 * (b32 - b[2] * b[4]) +
            b[1] * (b32 - b[1] * b[5]) + b[0] * (b23 + (b12 - b[3]) * b[4] + b[2] * (-2 * b[1] * b[3] + b[5]))
            ) / den);

        pade[2] = (float)((
            b13 * b[4] + b[2] * (b32 - b[2] * b[4]) - b12 * (2 * b[2] * b[3] + b[0] * b[5]) +
            b[1] * (b23 + b[3] * (2 * b[0] * b[3] - b[4]) + b[2] * b[5]) + b02 * (-(b[3] * b[4]) + b[2] * b[5]) +
            b[0] * (-(b22 * b[3]) + b42 - b[3] * b[5])
            ) / den);

        pade[4] = (float)((
            b22 * b22 + b12 * b32 - b33 +
            b02 * b42 - b13 * b[5] - b02 * b[3] * b[5] -
            b22 * (3 * b[1] * b[3] + 2 * b[0] * b[4] + b[5]) - b[1] * (2 * b[0] * b[3] * b[4] + b42 - b[3] * b[5]) +
            2 * b[2] * ((b12 + b[3]) * b[4] + b[0] * (b32 + b[1] * b[5]))
            ) / den);

        /* Denominator coefficients */
        pade[1] = (float)((
            -(b22 * b[3]) + b[1] * b32 + b[1] * b[2] * b[4] - b[0] * b[3] * b[4] - b12 * b[5] + b[0] * b[2] * b[5]
            ) / den);

        pade[3] = (float)((
            -(b22 * b[4]) - b[1] * b[3] * b[4] + b[2] * (b32 + b[1] * b[5]) + b[0] * (b42 - b[3] * b[5])
            ) / den);

        pade[5] = (float)((
            -b33 - b[1] * b42 - b22 * b[5] + b[3] * (2 * b[2] * b[4] + b[1] * b[5])
            ) / den);
    }
}




/********************************************************************************
 ********************************************************************************
 ********************************************************************************
 *****                                  C API                               *****
 ********************************************************************************
 ********************************************************************************
 ********************************************************************************/

static nvwarpResult nvwarpFromCUDAError(cudaError cuErr)
{
    struct Cunv { cudaError_t cuErr; nvwarpResult nvErr; };
    static const Cunv lut[] = {
        { cudaSuccess,                          NVWARP_SUCCESS              },
        { cudaErrorLaunchFileScopedSurf,        NVWARP_ERR_CUDA_LAUNCH      },
        { cudaErrorLaunchFileScopedTex,         NVWARP_ERR_CUDA_LAUNCH      },
        { cudaErrorLaunchMaxDepthExceeded,      NVWARP_ERR_CUDA_LAUNCH      },
        { cudaErrorLaunchOutOfResources,        NVWARP_ERR_CUDA_LAUNCH      },
        { cudaErrorLaunchPendingCountExceeded,  NVWARP_ERR_CUDA_LAUNCH      },
        { cudaErrorLaunchTimeout,               NVWARP_ERR_CUDA_LAUNCH      },
        { cudaErrorMemoryAllocation,            NVWARP_ERR_CUDA_MEMORY      },
        { cudaErrorMemoryValueTooLarge,         NVWARP_ERR_CUDA_MEMORY      },
        { cudaErrorInsufficientDriver,          NVWARP_ERR_CUDA_DRIVER      },
        { cudaErrorNoKernelImageForDevice,      NVWARP_ERR_CUDA_NO_KERNEL   },
        { cudaErrorNotSupported,                NVWARP_ERR_UNIMPLEMENTED    },
        { cudaErrorNotYetImplemented,           NVWARP_ERR_UNIMPLEMENTED    },
        { cudaErrorStartupFailure,              NVWARP_ERR_INITIALIZATION   },
    };
    #if WARP_DEBUG
        printf("CUDA error code %d\n", (int)cuErr);
    #endif /* WARP_DEBUG */
    for (const Cunv *p = lut; p < &lut[sizeof(lut) / sizeof(lut[0])]; ++p)
        if (p->cuErr == cuErr)
            return p->nvErr;
    return NVWARP_ERR_CUDA;
}


/********************************************************************************
* nvwarpErrorStringFromCode
********************************************************************************/

const char* nvwarpErrorStringFromCode(nvwarpResult err)
{
    struct WrErrStr { nvwarpResult err; const char *str; };
    static const WrErrStr lut[] = {
        { NVWARP_ERR_GENERAL,               "An otherwise unspecified failure has occurred"                         },
        { NVWARP_ERR_UNIMPLEMENTED,         "The requested feature has not yet been implemented"                    },
        { NVWARP_ERR_DOMAIN,                "The coordinates are outside of the domain"                             },
        { NVWARP_ERR_MISSING_PARAMETERS,    "Some required parameters have not been specified"                      },
        { NVWARP_ERR_PARAMETER,             "A parameter has an invalid value"                                      },
        { NVWARP_ERR_INITIALIZATION,        "Initialization has not completed successfully"                         },
        { NVWARP_ERR_CUDA_MEMORY,           "There is not enough CUDA memory for the operation specified"           },
        { NVWARP_ERR_CUDA_LAUNCH,           "CUDA was not able to launch the specified kernel"                      },
        { NVWARP_ERR_CUDA_DRIVER,           "CUDA driver version is insufficient for CUDA runtime version"          },
        { NVWARP_ERR_CUDA_NO_KERNEL,        "No CUDA kernel image has been found for this GPU compute version"      },
        { NVWARP_ERR_CUDA,                  "An otherwise unspecified error has been reported by the CUDA runtime"  },
        { NVWARP_ERR_APPROXIMATE,           "An accurate calculation is not available; approximation returned"      },
        { NVWARP_SUCCESS,                   "Operation was successful"                                              },
    };
    for (const WrErrStr *p = lut; p < &lut[sizeof(lut) / sizeof(lut[0])]; ++p)
        if (p->err == err)
            return p->str;
    return "UNKNOWN";
}


void nvwarpInitParams(nvwarpParams_t *params)
{
    memset(static_cast<void *>(params), -1, sizeof(*params));                        /* Default for uninitialized variables is NaN */
    memset(params->dist, 0, sizeof(params->dist));              /* Default is no distortion */
    params->topAngle    = +F_PI_2;                              /* Default is full angular range, north pole ... */
    params->bottomAngle = -F_PI_2;                              /* ... to south pole */
    params->rotAngles[0] = params->rotAngles[1] = params->rotAngles[2] = 0.f;
    strncpy(params->rotAxes, gDefaultAxes, sizeof(params->rotAxes));
    params->srcRadius   = 0.f;                                  /* No circular clipping */
    params->userData    = nullptr;                              /* No user data default */
}


nvwarpResult nvwarpComputeParamsSrcFocalLength(nvwarpParams_t *params, float angle, float radius)
{
    const unsigned srcType = GET_SRC_TYPE(params->type);
    params->srcFocalLen = ComputeSrcFocalLength(srcType, radius, angle, params->dist);
    return (params->srcFocalLen == params->srcFocalLen) ? NVWARP_SUCCESS : NVWARP_ERR_UNIMPLEMENTED;
}


nvwarpResult  nvwarpComputeParamsAxialAngleRange(const nvwarpParams_t *params, float minMaxXY[4])
{
    const unsigned srcType = GET_SRC_TYPE(params->type);
    return ComputeAxialAngleRange(srcType, params->srcFocalLen, params->srcX0, params->srcY0, params->dist, params->srcWidth, params->srcHeight, minMaxXY);
}


nvwarpResult nvwarpComputeBoxAxialAngleRange(const nvwarpParams_t *params, const float srcBoundingBox[4], float minMaxXY[4])
{
    const unsigned srcType = GET_SRC_TYPE(params->type);
    return ComputeAxialAngleRange(srcType, params->srcFocalLen, params->srcX0, params->srcY0, params->dist, srcBoundingBox, minMaxXY);
}


nvwarpResult nvwarpComputeParamsOutputResolution(nvwarpParams_t *params, float aspectRatio)
{
    uint32_t        wh[2];
    const unsigned  dstType = GET_DST_TYPE(params->type);
    nvwarpResult    err     = ComputeOutputResolution(dstType, params->topAngle, params->bottomAngle, params->srcFocalLen, params->control[0], aspectRatio, wh);
    params->dstWidth  = wh[0];
    params->dstHeight = wh[1];
    return err;
}


/********************************************************************************
 * nvwarpCreateInstance
 ********************************************************************************/

nvwarpResult nvwarpCreateInstance(nvwarpHandle *han)
{
    if (!han)
        return NVWARP_ERR_PARAMETER;
    *han = new nvwarpObject();
    return (*han) ? NVWARP_SUCCESS : NVWARP_ERR_GENERAL;
}


/********************************************************************************
 * nvwarpDestroyInstance
 ********************************************************************************/

void nvwarpDestroyInstance(nvwarpHandle han)
{
    if (han)
        delete han;
}


/********************************************************************************
 * nvwarpSetParams
 ********************************************************************************/

nvwarpResult nvwarpSetParams(nvwarpHandle han, const nvwarpParams_t *params)
{
    return han->setWarp360Params(params);
}


/********************************************************************************
 * nvwarpGetParams
 ********************************************************************************/

void nvwarpGetParams(const nvwarpHandle han, nvwarpParams_t *params)
{
    params->type        = han->type;
    params->srcWidth    = han->srcWidth;
    params->srcHeight   = han->srcHeight;
    params->srcFocalLen = han->cuParams.srcFocLenX;
    params->srcRadius   = han->srcRadius;
    params->dstWidth    = han->cuParams.dstWidth;
    params->dstHeight   = han->cuParams.dstHeight;
    params->topAngle    = han->topAngle;
    params->bottomAngle = han->botAngle;
    params->userData    = han->userData;
    params->control[0]  = han->getControl(0);
    memcpy(params->dist, han->cuParams.srcDist, sizeof(params->dist));
    han->getSrcPrincipalPoint(&params->srcX0, false);
    if (!AnglesFromRotationMatrixYdown(han->cuParams.srcRot, params->rotAngles, params->rotAxes))
    {
        strncpy(params->rotAxes, gDefaultAxes, sizeof(params->rotAxes));
        AnglesFromRotationMatrixYdown(han->cuParams.srcRot, params->rotAngles, params->rotAxes);
    }
}


/********************************************************************************
 * nvwarpSetBlock
 ********************************************************************************/

void nvwarpSetBlock(nvwarpHandle han, dim3 dim_block)
{
    han->setDimBlock(dim_block);
}


/********************************************************************************
 * nvwarpGetBlock
 ********************************************************************************/

void nvwarpGetBlock(const nvwarpHandle han, dim3 *dim_block)
{
    *dim_block = han->getDimBlock();
}


/********************************************************************************
 * nvwarpVersion
 ********************************************************************************/

uint32_t nvwarpVersion()
{
    return WARP360VERSION();
}


/********************************************************************************
 * Warp type
 ********************************************************************************/

nvwarpResult nvwarpSetWarpType(nvwarpHandle han, nvwarpType_t type)
{
    return han->setWarpType(type);
}

nvwarpType_t nvwarpGetWarpType(const nvwarpHandle han)
{
    return han->type;
}


/********************************************************************************
 * Pixel phase
 ********************************************************************************/

void nvwarpSetPixelPhase(nvwarpHandle han, uint32_t phase)
{
    han->pixelPhase = phase ? 0.5f : 0.0f;
}

uint32_t nvwarpGetPixelPhase(const nvwarpHandle han)
{
    return han->pixelPhase ? 1u : 0u;
}


/********************************************************************************
 * Source focal length
 ********************************************************************************/

void nvwarpSetSrcFocalLengths(nvwarpHandle han, float fl, float fy)
{
    han->setSrcFocalLength(fl, fy);
}

float nvwarpGetSrcFocalLength(const nvwarpHandle han, float *fy)
{
    if (fy) *fy = han->cuParams.srcFocLenY;
    return han->cuParams.srcFocLenX;
}

nvwarpResult nvwarpComputeSrcFocalLength(nvwarpHandle han, nvwarpType_t warpType, float radius, float angle, float dist[5])
{
    float foclen, d[5];

    han->setWarpType(warpType);
    han->setDistortion(dist);
    han->getDistortion(d);
    foclen = ComputeSrcFocalLength(han->srcType, radius, angle, d);
    han->setSrcFocalLength(foclen, foclen);
    return (foclen == foclen) ? NVWARP_SUCCESS : NVWARP_ERR_UNIMPLEMENTED;
}


/********************************************************************************
 * Source width & height
 ********************************************************************************/

void nvwarpSetSrcWidthHeight(nvwarpHandle han, uint32_t w, uint32_t h)
{
    han->srcWidth  = w;
    han->srcHeight = h;
}

void nvwarpGetSrcWidthHeight(const nvwarpHandle han, uint32_t wh[2])
{
    wh[0] = han->srcWidth;
    wh[1] = han->srcHeight;
}


/********************************************************************************
 * Source radius
 ********************************************************************************/

void nvwarpSetSrcRadius(nvwarpHandle han, float r)
{
    han->srcRadius = (r > 0.f) ? r : 0.f;
}

float nvwarpGetSrcRadius(const nvwarpHandle han)
{
    return han->srcRadius;
}


/********************************************************************************
 * Rotation
 ********************************************************************************/

void nvwarpSetRotation(nvwarpHandle han, const float R[9])
{
    han->setRotationMatrix(R);
}

void nvwarpGetRotation(const nvwarpHandle han, float R[9])
{
    han->getRotationMatrix(R);
}

void nvwarpSetEulerRotation(nvwarpHandle han, const float *angles, const char *axes)
{
    han->setEulerRotation(angles, axes);
}


/********************************************************************************
 * nvwarpSetSrcPrincipalPoint
 ********************************************************************************/

void nvwarpSetSrcPrincipalPoint(nvwarpHandle han, const float xy[2], uint32_t relToCenter)
{
    han->setSrcPrincipalPoint(xy, relToCenter);
}

void nvwarpGetSrcPrincipalPoint(const nvwarpHandle han, float xy[2], uint32_t relToCenter)
{
    han->getSrcPrincipalPoint(xy, relToCenter);
}


/********************************************************************************
 * Distortion
 ********************************************************************************/

void nvwarpSetDistortion(nvwarpHandle han, const float d[5])
{
    han->setDistortion(d);
}

void nvwarpGetDistortion(const nvwarpHandle han, float d[5])
{
    han->getDistortion(d);
}


/********************************************************************************
 * Destination width & height
 ********************************************************************************/

void nvwarpSetDstWidthHeight(nvwarpHandle han, uint32_t w, uint32_t h)
{
    han->cuParams.dstWidth  = w;
    han->cuParams.dstHeight = h;
}

void nvwarpGetDstWidthHeight(const nvwarpHandle han, uint32_t wh[2])
{
    wh[0] = han->cuParams.dstWidth;
    wh[1] = han->cuParams.dstHeight;
}


/********************************************************************************
 * Destination principal point
 ********************************************************************************/

void nvwarpSetDstPrincipalPoint(nvwarpHandle han, const float xy[2], uint32_t relToCenter)
{
    han->setDstPrincipalPoint(xy, relToCenter);
}

void nvwarpGetDstPrincipalPoint(const nvwarpHandle han, float xy[2], uint32_t relToCenter)
{
    han->getDstPrincipalPoint(xy, relToCenter);
}


/********************************************************************************
 * Destination focal length
 ********************************************************************************/

void nvwarpSetDstFocalLengths(nvwarpHandle han, float fl, float fy)
{
    han->setDstFocalLength(fl, fy);
}

float nvwarpGetDstFocalLength(const nvwarpHandle han, float *fy)
{
    if (fy) *fy = han->dstFocLenY;
    return han->dstFocLenX;
}

void nvwarpComputeDstFocalLength(nvwarpHandle han, float topAngle, float bottomAngle, uint32_t dstWidth, uint32_t dstHeight)
{
    han->setDstFocalLengthCOP(topAngle, bottomAngle, dstWidth, dstHeight);
    han->topAngle = topAngle;
    han->botAngle = bottomAngle;
}


/********************************************************************************
 * Control
 ********************************************************************************/

void nvwarpSetControl(nvwarpHandle han, uint32_t index, float control)
{
    han->setControl(index, control);
}

float nvwarpGetControl(const nvwarpHandle han, uint32_t index)
{
    return han->getControl(index);
}


/********************************************************************************
 * User data
 ********************************************************************************/

void nvwarpSetUserData(nvwarpHandle han, void *userData)
{
    han->userData = userData;
}

void* nvwarpGetUserData(nvwarpHandle han)
{
    return han->userData;
}

/********************************************************************************
 * nvwarpWarpSurface
 ********************************************************************************/

nvwarpResult nvwarpWarpSurface(const nvwarpHandle han, cudaStream_t stream, cudaTextureObject_t srcTex, cudaSurfaceObject_t dstSurface)
{
    return nvwarpFromCUDAError(han->warp(stream, srcTex, dstSurface));
}


/********************************************************************************
 * nvwarpWarpBuffer
 ********************************************************************************/

nvwarpResult nvwarpWarpBuffer(const nvwarpHandle han, cudaStream_t stream, cudaTextureObject_t srcTex, void *dstAddr, size_t dstRowBytes)
{
    return nvwarpFromCUDAError(han->warp(stream, srcTex, dstAddr, dstRowBytes));
}


/********************************************************************************
 * nvwarpWarpCoordinates
 ********************************************************************************/

nvwarpResult nvwarpWarpCoordinates(const nvwarpHandle han, uint32_t numPts, const float *inPtsXY, float *outPtsXY)
{
    return han->warp(numPts, inPtsXY, outPtsXY);
}


/********************************************************************************
 * nvwarpInverseWarpCoordinates
 ********************************************************************************/

nvwarpResult nvwarpInverseWarpCoordinates(const nvwarpHandle han, uint32_t numPts, const float *inPtsXY, float *outPtsXY)
{
    return han->inverseWarp(numPts, inPtsXY, outPtsXY);
}


/********************************************************************************
 * nvwarpSrcToRay
 ********************************************************************************/

nvwarpResult nvwarpSrcToRay(const nvwarpHandle han, uint32_t numPts, const float *pts2D, float *rays3D)
{
    return han->srcToRay(numPts, pts2D, rays3D);
}


/********************************************************************************
 * nvwarpDstToRay
 ********************************************************************************/

nvwarpResult nvwarpDstToRay(const nvwarpHandle han, uint32_t numPts, const float *pts2D, float *rays3D)
{
    return han->dstToRay(numPts, pts2D, rays3D);
}


/********************************************************************************
 * nvwarpSrcFromRay
 ********************************************************************************/

nvwarpResult nvwarpSrcFromRay(const nvwarpHandle han, uint32_t numRays, const float *rays3D, float *pts2D)
{
    return han->srcFromRay(numRays, rays3D, pts2D);
}


/********************************************************************************
 * nvwarpDstFromRay
 ********************************************************************************/

nvwarpResult nvwarpDstFromRay(const nvwarpHandle han, uint32_t numRays, const float *rays3D, float *pts2D)
{
    return han->dstFromRay(numRays, rays3D, pts2D);
}


/********************************************************************************
 * nvwarpComputeSrcAngularFromPixelCoordinates
 ********************************************************************************/

nvwarpResult NVWARPAPI nvwarpComputeSrcAngularFromPixelCoordinates(const nvwarpHandle han, uint32_t numPts, const float *pts2D, float *ang2D)
{
    nvwarpResult    err     = NVWARP_SUCCESS;
    nvwarpHandle    angWarp = nullptr;
    nvwarpType_t    type    = WARP_TYPE(han->srcType, NVSURF_EQUIRECT);

    if (NVWARP_SUCCESS != (err = nvwarpCreateInstance(&angWarp)))
        goto bail;
    *angWarp = *han;
    err = angWarp->setWarpType(type);
    angWarp->setDstFocalLength(1.f, -1.f);          /* Destination is a unit sphere ... */
    angWarp->setDstPrincipalPoint(nullptr, 0);      /* ... centered at the origin */
    err = angWarp->warp(numPts, pts2D, ang2D);      /* Warp from the source to equirect */
bail:
    nvwarpDestroyInstance(angWarp);
    return err;
}


/********************************************************************************
 * nvwarpComputeDstAngularFromPixelCoordinates
 ********************************************************************************/

nvwarpResult NVWARPAPI nvwarpComputeDstAngularFromPixelCoordinates(const nvwarpHandle han, uint32_t numPts, const float *pts2D, float *ang2D)
{
    nvwarpResult    err     = NVWARP_SUCCESS;
    nvwarpHandle    angWarp = nullptr;
    nvwarpType_t    type    = WARP_TYPE(NVSURF_EQUIRECT, han->dstType);

    if (NVWARP_SUCCESS != (err = nvwarpCreateInstance(&angWarp)))
        goto bail;
    *angWarp = *han;
    err = angWarp->setWarpType(type);
    angWarp->pixelPhase = 0.f;
    angWarp->cuParams.dstX0 += han->pixelPhase;
    angWarp->cuParams.dstY0 += han->pixelPhase;
    angWarp->setSrcFocalLength(1.f, -1.f);              /* Source is a unit sphere ... */
    angWarp->setSrcPrincipalPoint(nullptr, 0);          /* ... centered at the origin */
    err = angWarp->inverseWarp(numPts, pts2D, ang2D);   /* Warp from the destination to equirect */
bail:
    nvwarpDestroyInstance(angWarp);
    return err;
}


/********************************************************************************
 * nvwarpComputeSrcPixelFromAngularCoordinates
 ********************************************************************************/

nvwarpResult NVWARPAPI nvwarpComputeSrcPixelFromAngularCoordinates(const nvwarpHandle han, uint32_t numPts, const float *ang2D, float *pts2D)
{
    nvwarpResult    err     = NVWARP_SUCCESS;
    nvwarpHandle    angWarp = nullptr;
    nvwarpType_t    type    = WARP_TYPE(han->srcType, NVSURF_EQUIRECT);

    if (NVWARP_SUCCESS != (err = nvwarpCreateInstance(&angWarp)))
        goto bail;
    *angWarp = *han;
    err = angWarp->setWarpType(type);
    angWarp->pixelPhase = 0.f;                          /* Make sure that angles are unbiased */
    angWarp->setDstFocalLength(1.f, -1.f);              /* Destination is the unit sphere ... */
    angWarp->setDstPrincipalPoint(nullptr, 0);          /* ... centered at the origin */
    angWarp->cuParams.srcX0 += han->pixelPhase;
    angWarp->cuParams.srcY0 += han->pixelPhase;
    err = angWarp->inverseWarp(numPts, ang2D, pts2D);   /* Warp from equirect to source */
bail:
    nvwarpDestroyInstance(angWarp);
    return err;
}


/********************************************************************************
 * nvwarpComputeDstPixelFromAngularCoordinates
 ********************************************************************************/

nvwarpResult NVWARPAPI nvwarpComputeDstPixelFromAngularCoordinates(const nvwarpHandle han, uint32_t numPts, const float *ang2D, float *pts2D)
{
    nvwarpResult    err     = NVWARP_SUCCESS;
    nvwarpHandle    angWarp = nullptr;
    nvwarpType_t    type    = WARP_TYPE(NVSURF_EQUIRECT, han->dstType);

    if (NVWARP_SUCCESS != (err = nvwarpCreateInstance(&angWarp)))
        goto bail;
    *angWarp = *han;
    err = angWarp->setWarpType(type);
    angWarp->pixelPhase = 0.f;                          /* Make sure that angles are unbiased */
    angWarp->cuParams.dstX0 += han->pixelPhase;
    angWarp->cuParams.dstY0 += han->pixelPhase;
    angWarp->setSrcFocalLength(1.f, -1.f);              /* Src is the unit sphere ... */
    angWarp->setSrcPrincipalPoint(nullptr, 0);          /* ... centered at the origin */
    err = angWarp->warp(numPts, ang2D, pts2D);          /* Warp from equirect to source */
bail:
    nvwarpDestroyInstance(angWarp);
    return err;
}


/********************************************************************************
 * nvwarpConvertTransformBetweenYUpandYDown
 ********************************************************************************/

nvwarpResult NVWARPAPI nvwarpComputeDstFocalLengths(nvwarpHandle han, float topAxialAngle, float botAxialAngle, float leftAxialAngle, float rightAxialAngle, uint32_t dstWidth, uint32_t dstHeight)
{
    han->setDstFocalLengthCOP(topAxialAngle, botAxialAngle, dstWidth, dstHeight);
    float pts[] = { leftAxialAngle, 0.f, rightAxialAngle, 0.f };                        /* { (lonL, latL), (lonR, latR) } here */
    nvwarpResult err = nvwarpComputeDstPixelFromAngularCoordinates(han, 2, pts, pts);   /* { (  xL,   yL), (  xR,   yR) } here */
    float fs = (pts[2] - pts[0]) / (dstWidth - 1.f);                                    /* Focal scale */
    han->cuParams.dstX0 = pts[0] + han->cuParams.dstX0 * fs;                            /* New offset, independent of the pixel phase */
    han->dstFocLenX *= fs;                                                              /* New focal length */
    han->cuParams.dstInvFocLenX = 1.f / han->dstFocLenX;
    return err;
}


/********************************************************************************
 * nvwarpConvertTransformBetweenYUpandYDown
 ********************************************************************************/

void nvwarpConvertTransformBetweenYUpandYDown(const float fr[9], float to[9])
{
    float M[9] = { +fr[0], -fr[1], -fr[2], -fr[3], +fr[4], +fr[5], -fr[6], +fr[7], +fr[8] };
    memcpy(to, M, sizeof(M));
}


/********************************************************************************
 * nvwarpComputeYCbCr2RgbMatrix
 ********************************************************************************/

void nvwarpComputeYCbCr2RgbMatrix(nvwarpYUVRGBParams_t *p, uint32_t matrix_coefficients, uint32_t video_full_range_flag, uint32_t bit_depth, uint32_t normalized_input, uint32_t normalized_output)
{
    double m[3][3];
    double Kr, Kb, d1, d2;

    switch (matrix_coefficients)
    {
        default:
        case 2:  // If unspecified use 709
        case 709:
        case 1:  Kr = 0.2126; Kb = 0.0722; break;  // BT.709
        case 0xFCC:
        case 4:  Kr = 0.30;   Kb = 0.11;   break;  // FCC
        case 5:
        case 601:
        case 6:  Kr = 0.299;  Kb = 0.114;  break;  // BT.601
        case 240:
        case 7:  Kr = 0.212;  Kb = 0.087;  break;  // SMPTE 240M
        case 9:
        case 2020:
        case 10: Kr = 0.2627; Kb = 0.0593; break;  // BT.2020
    }

    m[0][0] = Kr;                       // Y
    m[0][1] = 1.0 - Kr - Kb;
    m[0][2] = Kb;
    m[1][0] = -0.5 * Kr / (1.0 - Kb);   // U
    m[1][1] = -0.5 - m[1][0];
    m[1][2] = 0.5;
    m[2][0] = 0.5;                      // V
    m[2][2] = -0.5 * Kb / (1.0 - Kr);
    m[2][1] = -0.5 - m[2][2];

    // full vs limited range (d1,d2 are initialized for 8-bit scale (adjusted later on if output is 10-bit)
    if (video_full_range_flag)
    {
        d1 = 0.0;
        d2 = 128.0;
    }
    else
    {
        d1 = 16.0;
        d2 = 128.0;

        double yRange  = pow(2, bit_depth - 8) * 219.0 / (pow(2, bit_depth) - 1);
        double uvRange = pow(2, bit_depth - 8) * 224.0 / (pow(2, bit_depth) - 1);
        if (matrix_coefficients == 8)
            uvRange = yRange;

        for (int i = 0; i < 3; i++)
        {
            m[0][i] *= yRange;
            m[1][i] *= uvRange;
            m[2][i] *= uvRange;
        }
    }
    if (bit_depth != 8)
    {
        d1 *= pow(2, bit_depth - 8);
        d2 *= pow(2, bit_depth - 8);
    }

    // For normalize output multiply the matrix by maxValue, so the inverse divides by it.
    if (normalized_output)
    {
        const double maxValue = pow(2, bit_depth) - 1.0;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                m[i][j] *= maxValue;
    }
    // For normalized input divide the matrix and constants so the inverse multiplies the output by maxValue
    if (normalized_input)
    {
        const double maxValue = pow(2, bit_depth) - 1.0;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                m[i][j] /= maxValue;
    }
    // Compute inverse matrix
    const double det =
        m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2]) -
        m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
        m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    const double invdet = 1 / det;
    p->ry  = (float)((m[1][1] * m[2][2] - m[2][1] * m[1][2]) * invdet);
    p->rcb = (float)((m[0][2] * m[2][1] - m[0][1] * m[2][2]) * invdet);
    p->rcr = (float)((m[0][1] * m[1][2] - m[0][2] * m[1][1]) * invdet);
    p->gy  = (float)((m[1][2] * m[2][0] - m[1][0] * m[2][2]) * invdet);
    p->gcb = (float)((m[0][0] * m[2][2] - m[0][2] * m[2][0]) * invdet);
    p->gcr = (float)((m[1][0] * m[0][2] - m[0][0] * m[1][2]) * invdet);
    p->by  = (float)((m[1][0] * m[2][1] - m[2][0] * m[1][1]) * invdet);
    p->bcb = (float)((m[2][0] * m[0][1] - m[0][0] * m[2][1]) * invdet);
    p->bcr = (float)((m[0][0] * m[1][1] - m[1][0] * m[0][1]) * invdet);
    p->yOffset = (float)d1;
    p->cOffset = (float)d2;
    #if WARP_DEBUG
        printf("%d %s\n", matrix_coefficients, (video_full_range_flag ? "Full" : ""));
        printf("Y: %+16.7g %+16.7g %+16.7g\nU: %+16.7g %+16.7g %+16.7g\nV: %+16.7g %+16.7g %+16.7g\n",
            m[0][0], m[0][1], m[0][2],  m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2]);
        printf("R: %+16.7g %+16.7g %+16.7g\nG: %+16.7g %+16.7g %+16.7g\nB: %+16.7g %+16.7g %+16.7g\nO: %+16.7g %+16.7g\n",
            p->ry, p->rcb, p->rcr, p->gy, p->gcb, p->gcr, p->by, p->bcb, p->bcr, p->yOffset, p->cOffset);
    #endif /* WARP_DEBUG */
}


/********************************************************************************
 * nvwarpConvertYUVNV12ToRGBA
 ********************************************************************************/

nvwarpResult nvwarpConvertYUVNV12ToRGBA(cudaStream_t stream, const nvwarpYUVRGBParams_t *params, const void *yuv, size_t yuvRowBytes, void *dst, size_t dstRowBytes)
{
    NV12RGBABuffer(stream, params, (const unsigned char*)yuv, yuvRowBytes, (uchar4*)dst, dstRowBytes);
    return nvwarpFromCUDAError(cudaGetLastError());
}


namespace { // anonymous

#ifdef USE_PADE_INVERSE_DISTORTION

/********************************************************************************
 * (r + p0 * r^3 + p2 * r^5 + p4 * r^7) / (1. + p1 * r^2 + p3 * r^4 + p5 * r^6)
 ********************************************************************************/
float EvaluatePade6(float r, const float *pade)
{
    double r2 = (double)r * r;
    return (float)((((pade[4] * r2 + pade[2]) * r2 + pade[0]) * r2 + 1.) * r
                 / (((pade[5] * r2 + pade[3]) * r2 + pade[1]) * r2 + 1.));
}

#else /* !USE_PADE_INVERSE_DISTORTION */


/********************************************************************************
 ****************************************************************************//**
 *** Scalar Function class, primarily for root-finding.
 ********************************************************************************
 ********************************************************************************/

class ScalarFunction {
public:
    /// Constructor
    ScalarFunction() {}

    /// Destructor (don't really need this to be virtual)
    virtual ~ScalarFunction() {}

    /// Evaluate the scalar function.
    /// @param[in]  x the location at which the scalar function is to be evaluated.
    /// @return     the value of the scalar function evaluated at x.
    virtual double evaluate(double x) const = 0;

    /// Evaluate the derivative of the scalar function (not required).
    /// @param[in]  x the location at which the derivative of the scalar function is to be evaluated.
    /// @return     the derivative of the scalar function evaluated at x.
    virtual double derivative(double x) const { return NAN; }

    /// Solve the continuous scalar function y = f(x) for x, given y.
    /// The function evaluated at the bounds should surround the desired value y, i.e.
    ///     (f(left) - y) * (f(right) - y) <= 0,
    /// otherwise NAN is returned. This is a sufficient condition to guarantee convergence, but not necessary,
    /// so it is desirable to bracket the interval as tightly as possible to contain only one root.
    /// @param[in]  y     the value of the scalar function for which to solve.
    /// @param[in]  left  the left  end of the interval in which to search for the solution.
    /// @param[in]  right the right end of the interval in which to search for the solution.
    /// @return     the parameter x for which y = f(x), or
    ///             NAN, if no root was found in the specified interval.
    double solve(double y, double left, double right) const;
};


/********************************************************************************
 * ScalarFunction::solve
 ********************************************************************************/

double ScalarFunction::solve(double y, double x0, double x1) const
{
    unsigned      k = 60;                             /* Max iterations, though 8 is typical */
    const double  eps = 2.2204460492503130808e-16;      /* 1 ULP */
    double  x, f0, f1, f, d, dx;

    /* Assure that x0 and x1 are distinct, and evaluate at those points */
    if ((f0 = evaluate(x0) - y) == 0)
        return x0;                                        /* Solution at left end of interval */
    if ((f1 = evaluate(x1) - y) == 0)
        return x1;                                        /* Solution at right end of interval */
    if (f0 * f1 > 0)
        return NAN;                                       /* No solution in interval */

    if (x0 > x1)
    {
        f = f0; f0 = f1; f1 = f;                          /* Assure x0 < x1 */
        x = x0; x0 = x1; x1 = x;
    }

    if (fabs(f0) < fabs(f1))
    {                                                   /* f(x0) is closer to 0 */
        x = x0;
        f = f0;
    }
    else
    {                                                   /* f(x1) is closer to 0 */
        x = x1;
        f = f1;
    }

    if ((d = derivative(0.)) != d || (d = derivative(1.)) != d)
    { /* No derivative supplied: use secant or regula falsi */
        return NAN;
    }
    else
    { /* Use Newton or bisection */
        do {  // TODO: Brent's method to avoid worst case Newton?
            if ((d = derivative(x)) == 0 || !(x0 < (x -= f / d) && (x < x1)))  /* If Newton fails, ... */
                x = (x0 + x1) * .5;                                              /* ... bisect */
            if ((f = evaluate(x) - y) == 0)
                return x;
            if (f * f0 > 0)
            {                                                                 /* Squeeze left end of interval */
                f0 = f;
                dx = x - x0;
                x0 = x;
            }
            else
            {                                                                 /* Squeeze right end of interval */
                f1 = f;
                dx = x1 - x;
                x1 = x;
            }
        } while (dx > eps * fabs(x) && --k);
        return x;
    }
}



/********************************************************************************
 ********************************************************************************
 *** Polynomial
 ********************************************************************************
 ********************************************************************************/


#define MAXORDER 6

 /********************************************************************************
  * KEEvaluatePolynomial
  ********************************************************************************/

double KEEvaluatePolynomial(unsigned n, const double *cf, double x)
{
    double y;
    for (cf += n, y = 0; n--;)
    { /* Horner's method */
        y = y * x + *--cf;
    }
    return y;
}


/********************************************************************************
 * KEEvaluatePolynomialDerivative
 ********************************************************************************/

double KEEvaluatePolynomialDerivative(unsigned n, const double *cf, double x)
{
    double y;
    for (cf += n, y = 0; --n;)
    { /* Horner's method */
        y = y * x + *--cf * n;
    }
    return y;
}


/********************************************************************************
 * Minimize Polynomial Order.
 * This removes high-order zero coefficients.
 * @param[in,out] n    the order of the polynomial, adjusted downward on output to remove high-order zero coefficients.
 * @param[in]     cf  the coefficients of the polynomial.
 ********************************************************************************/

static void MinimizePolynomialOrder(unsigned *pn, const double *cf)
{
    unsigned n = *pn;
    for (cf += n - 1; 0 == *cf && n; --cf)
        --n;
    *pn = n;
}


/********************************************************************************
 * Bound Polynomial Roots, algorithm due to Cauchy.
 * @param[in]   n       the order of the polynomial (number of coefficients).
 * @param[in]   cf      the coefficients of the polynomial.
 * @param[out]  bounds  the resultant bounds.
 * @param[in]   y       find bounds for roots of y == poly(x), rather than 0 == poly(x).
 ********************************************************************************/

void KEBoundPolynomialRoots(unsigned n, const double *cf, double bounds[2], double y)
{
    double max, m0, m1;
    int i;

    if (n < 2)
    {
        bounds[0] = bounds[1] = NAN;
        return;
    }

    for (i = n - 2, max = 0; i > 0; --i)
    { /* Find max of the middle coefficients */
        if (max < (m1 = fabs(cf[i])))
            max = m1;
    }
    m0 = fabs(cf[0] - y);
    m1 = fabs(cf[n - 1]);
    bounds[0] = m0 / (m0 + ((max >= m1) ? max : m1));
    bounds[1] = 1. + ((max >= m0) ? max : m0) / m1;
}


class PolynomialFunction: public ScalarFunction {
public:
    PolynomialFunction(unsigned numCoeff, const double *coeff) { m_numCoeff = numCoeff; MinimizePolynomialOrder(&m_numCoeff, m_coeff = coeff); };
    double   evaluate(double x) const override { return KEEvaluatePolynomial(m_numCoeff, m_coeff, x); }
    double derivative(double x) const override { return KEEvaluatePolynomialDerivative(m_numCoeff, m_coeff, x); }
private:
    unsigned      m_numCoeff;
    const double  *m_coeff;
};


double KEFindRootOfPolynomial(unsigned n, const double *cf, double x0, double x1, double y)
{
    PolynomialFunction poly(n, cf);
    return poly.solve(y, x0, x1);
}



/********************************************************************************
 ********************************************************************************
 *** Distortion Polynomial
 ********************************************************************************
 ********************************************************************************/


 /********************************************************************************
  * EvaluateDistortionPolynomial
  ********************************************************************************/

double KEEvaluateDistortionPolynomial(unsigned n, const double *cf, double x)
{
    double f, x2;

    for (f = 0, cf += n, x2 = x * x; n--;)
    { /* Horner's method */
        f += *--cf;
        f *= x2;
    }
    f += 1;
    f *= x;
    return f;
}


/********************************************************************************
 * EvaluateDistortionPolynomialDerivative
 ********************************************************************************/

double KEEvaluateDistortionPolynomialDerivative(unsigned n, const double *cf, double x)
{
    double d;
    int j;

    for (d = 0, cf += n, j = 2 * n + 1, x *= x; j > 1; j -= 2)
    { /* Horner's method */
        d += *--cf * j;
        d *= x;
    }
    d += 1;
    return d;
}


/********************************************************************************
 * DistortionPolynomialFunction
 ********************************************************************************/

class DistortionPolynomialFunction: public ScalarFunction {
public:
    DistortionPolynomialFunction(unsigned numCoeff, const double *coeff) { m_numCoeff = numCoeff; m_coeff = coeff; };
    double evaluate(double x)   const override { return KEEvaluateDistortionPolynomial(m_numCoeff, m_coeff, x); }
    double derivative(double x) const override { return KEEvaluateDistortionPolynomialDerivative(m_numCoeff, m_coeff, x); }
private:
    unsigned      m_numCoeff;
    const double  *m_coeff;
};


/****************************************************************************//**
 * Bracket the First Zero Crossing of the polynomial representing the derivative of the distortion polynomial.
 * Such a polynomial is even with the constant coefficient equal to 1.
 * we know that the first negative coefficient will probably be responsible for the zero crossing, so we
 ********************************************************************************/

static bool BracketFirstDistortionDerivativeZeroCrossing(unsigned n, const double *cf, double bounds[2])
{
    unsigned i;

    for (i = 2; i <= n; ++i)
    {
        if (cf[i - 1] < 0)                                                                            /* Find the first negative coefficients */
        {
            KEBoundPolynomialRoots(i, cf, bounds, 0);                                                    /* Determine bounds for a lower-order polynomial */
            if (KEEvaluatePolynomial(n, cf, bounds[0]) * KEEvaluatePolynomial(n, cf, bounds[1]) <= 0) /* Check that the bounds bracket the roots */
                return true;
        }
    }
    return false;
}


/********************************************************************************
 * KEFindFlatOfDistortionPolynomial
 ********************************************************************************/

double KEFindFlatOfDistortionPolynomial(unsigned n, const double *cf, double flatValue)
{
    double    x;
    unsigned  i;
    double    der[MAXORDER + 1], bounds[2];

    MinimizePolynomialOrder(&n, cf);

    /* Compute derivative polynomial: polynomial of order n+1 in x^2 */
    if (n >= sizeof(der) / sizeof(der[0]))
        return NAN;     /* If we get here, we need to enlarge the der array above */
    for (i = n; i; i--)
    {
        der[i] = cf[i - 1] * (i * 2 + 1);
    }
    der[0] = 1. - flatValue;
    ++n;

    if (BracketFirstDistortionDerivativeZeroCrossing(n, der, bounds))
    {
        x = KEFindRootOfPolynomial(n, der, bounds[0], bounds[1], 0);   /* This could return a NaN */
        x = sqrt(x);
    }
    else
    {
        x = INFINITY; // HUGE_VAL might be more portable
    }
    return x;
}


/********************************************************************************
 * BoundDistortionPolynomialRoots, algorithm due to Cauchy.
 * @param[in]   n       the number of coefficients in the distortion polynomial.
 * @param[in]   cf      the coefficients of the distortion polynomial.
 * @param[out]  bounds  the resultant bounds.
 * @param[in]   y       find bounds for roots of y == poly(x), rather than 0 == poly(x).
 ********************************************************************************/

static void BoundDistortionPolynomialRoots(unsigned n, const double *cf, double bounds[2], double y = 0)
{
    double fake[MAXORDER + 2];

    if (n == 0)
    {
        bounds[0] = bounds[1] = y;
        return;
    }

    for (unsigned i = n; n--;)                    /* Make fake poly = { 0, 1, cf[0], cf[1], ... } */
    {
        fake[i + 2] = cf[i];
    }
    fake[1] = 1;
    fake[0] = 0;
    KEBoundPolynomialRoots(n + 2, fake, bounds, y);
    bounds[0] = sqrt(bounds[0]);
    bounds[1] = sqrt(bounds[1]);
}


/********************************************************************************
 * KEFindRootOfDistortionPolynomial
 ********************************************************************************/

double KEFindRootOfDistortionPolynomial(unsigned n, const double *cf, double maxX, double y)
{
    double f0, f1, x0, x1;
    if (y < 0)                                                    /* The argument should be positive, ... */
    {
        return -KEFindRootOfDistortionPolynomial(n, cf, maxX, -y);  /* ... but we can get a valid solution since the polynomial is odd */
    }                                                             /* Hereafter, y >= 0 */

    /* Since undistorted is the identity and distortion should be slight, the root should be close to y */
    f1 = KEEvaluateDistortionPolynomial(n, cf, x1 = y) - y;       /* Let us find the value at y */
    if (f1 >= 0)                                                  /* We have a bracket in [0, y] */
    {
        f0 = -y;                                                    /* We know the [negative] value ... */
        x0 = 0;                                                     /* ... at 0 */
    }
    else
    {
        x0 = x1;
        f0 = f1;
        f1 = KEEvaluateDistortionPolynomial(n, cf, x1 = maxX) - y;    /* Let us find the value at maxX, to bracket between [y, maxX] */
        if (!(f1 >= 0))                                             /* No bracket in the interval */
            return NAN;
    }
    DistortionPolynomialFunction poly(n, cf);
    return poly.solve(y, x0, x1);
}

#endif /* !USE_PADE_INVERSE_DISTORTION */

} // namespace anonymous
