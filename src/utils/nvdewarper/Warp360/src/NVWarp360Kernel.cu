/** @file Warp360Kernel.cu
 * Various projections.
 *
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

#include "NVWarp360Kernel.h"
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <texture_indirect_functions.h>
#include <math.h>


/********************************************************************************
 ********************************************************************************
 **                             RENDERING UTILITIES                            **
 ********************************************************************************
 ********************************************************************************/


#define   F_PI_2    ((float)(1.5707963267948966192))


/********************************************************************************
 * Convert from a float4 to a uchar4 pixel
 ********************************************************************************/
__device__ inline uchar4 PackColor(float4 input)
{
    // Saturate each channel and scale to output range
    uchar4 output;
    output.x = __float2uint_rn(__saturatef(input.x) * 255.f);
    output.y = __float2uint_rn(__saturatef(input.y) * 255.f);
    output.z = __float2uint_rn(__saturatef(input.z) * 255.f);
    output.w = __float2uint_rn(__saturatef(input.w) * 255.f);
    return output;
}

__device__ inline uchar4 PackColor(float3 input)
{
    // Saturate each channel and scale to output range
    uchar4 output;
    output.x = __float2uint_rn(__saturatef(input.x) * 255.f);
    output.y = __float2uint_rn(__saturatef(input.y) * 255.f);
    output.z = __float2uint_rn(__saturatef(input.z) * 255.f);
    output.w = 255;
    return output;
}


/********************************************************************************
 ********************************************************************************
 **                                GET PIXEL FROM RAY                          **
 ********************************************************************************
 ********************************************************************************/

template<nvwarpSurface_t> uchar4 GetPixel(float3& worldRay, Warp360KernelParams& p, cudaTextureObject_t srcTex);


/********************************************************************************
 * Project from a fisheye.
 ********************************************************************************/
template<> __device__ inline uchar4 GetPixel<NVSURF_FISHEYE>(float3& worldRay, Warp360KernelParams& p, cudaTextureObject_t srcTex)
{
    float3 ray;                                             /* Transform ray from world to camera space */
    ray.x = worldRay.x * p.srcRot[0] + worldRay.y * p.srcRot[3] + worldRay.z * p.srcRot[6];
    ray.y = worldRay.x * p.srcRot[1] + worldRay.y * p.srcRot[4] + worldRay.z * p.srcRot[7];
    ray.z = worldRay.x * p.srcRot[2] + worldRay.y * p.srcRot[5] + worldRay.z * p.srcRot[8];
    float r = rnorm3df(ray.x, ray.y, ray.z);
    ray.x *= r;
    ray.y *= r;
    ray.z *= r;

    r = hypotf(ray.x, ray.y);                               /* Project ray into fisheye */
    float t = r;
    if (r)
    {
        t = atan2f(r, ray.z);
        float t2 = t * t;
        t = ((((p.srcDist[3] * t2 + p.srcDist[2]) * t2 + p.srcDist[1]) * t2 + p.srcDist[0]) * t2 + 1.f) * t / r;
    }
    ray.x = ray.x * t * p.srcFocLenX + p.srcX0;
    ray.y = ray.y * t * p.srcFocLenY + p.srcY0;

    return PackColor(tex2D<float4>(srcTex, ray.x, ray.y));  /* Bilinearly interpolate the desired pixel value */
}


/********************************************************************************
 * Project from an equirectangular.
 ********************************************************************************/
template<> __device__ inline uchar4 GetPixel<NVSURF_EQUIRECT>(float3& worldRay, Warp360KernelParams& p, cudaTextureObject_t srcTex)
{
    float3 ray;                                             /* Transform ray from world to camera space */
    ray.x = worldRay.x * p.srcRot[0] + worldRay.y * p.srcRot[3] + worldRay.z * p.srcRot[6];
    ray.y = worldRay.x * p.srcRot[1] + worldRay.y * p.srcRot[4] + worldRay.z * p.srcRot[7];
    ray.z = worldRay.x * p.srcRot[2] + worldRay.y * p.srcRot[5] + worldRay.z * p.srcRot[8];
    ray.y = atan2f(ray.y, hypotf(ray.x, ray.z));            /* Bipolar */
    ray.x = atan2f(ray.x, ray.z);                           /* Bipolar */
    ray.x =  ray.x * p.srcFocLenX + p.srcX0;                /* Positive */
    ray.y =  ray.y * p.srcFocLenY + p.srcY0;                /* Positive */
    return PackColor(tex2D<float4>(srcTex, ray.x, ray.y));  /* Bilinearly interpolate the desired pixel value */
}


/********************************************************************************
 * Project from a perspective.
 ********************************************************************************/

/* Without tangential distortion */
__device__ inline uchar4 GetPixel_PERSPECTIVE_NOTAN(float3& worldRay, Warp360KernelParams& p, cudaTextureObject_t srcTex)
{
    float3 ray;                                             /* Transform ray from world to camera space */
    ray.x = worldRay.x * p.srcRot[0] + worldRay.y * p.srcRot[3] + worldRay.z * p.srcRot[6];
    ray.y = worldRay.x * p.srcRot[1] + worldRay.y * p.srcRot[4] + worldRay.z * p.srcRot[7];
    ray.z = worldRay.x * p.srcRot[2] + worldRay.y * p.srcRot[5] + worldRay.z * p.srcRot[8];
    if (ray.z <= 0)
        return uchar4{0, 0, 0, 0};

    float   t = 1.f / ray.z,
            r = hypotf(ray.x, ray.y) * t;
    if (r)
    {
        float r2 = r * r;
        t *= (((p.srcDist[2] * r2 + p.srcDist[1]) * r2 + p.srcDist[0]) * r2 + 1.f);
    }
    ray.x = ray.x * t * p.srcFocLenX + p.srcX0;
    ray.y = ray.y * t * p.srcFocLenY + p.srcY0;
    return PackColor(tex2D<float4>(srcTex, ray.x, ray.y));  /* Bilinearly interpolate the desired pixel value */
}

/* With tangential distortion */
template<> __device__ inline uchar4 GetPixel<NVSURF_PERSPECTIVE>(float3& worldRay, Warp360KernelParams& p, cudaTextureObject_t srcTex)
{
    float3 ray;                                             /* Transform ray from world to camera space */
    ray.x = worldRay.x * p.srcRot[0] + worldRay.y * p.srcRot[3] + worldRay.z * p.srcRot[6];
    ray.y = worldRay.x * p.srcRot[1] + worldRay.y * p.srcRot[4] + worldRay.z * p.srcRot[7];
    ray.z = worldRay.x * p.srcRot[2] + worldRay.y * p.srcRot[5] + worldRay.z * p.srcRot[8];
    if (ray.z <= 0)
        return uchar4{0, 0, 0, 0};

    float2 q = { ray.x / ray.z, ray.y / ray.z };     /* Project to the plane */
    float r2        = q.x * q.x + q.y * q.y,
          rs        = ((p.srcDist[2] * r2 + p.srcDist[1]) * r2 + p.srcDist[0]) * r2 + 1.f,
          twiceXY   = q.x * q.y * 2.f;
    /*      -radial-   -------------------------tangential--------------------------- */
    ray.x = rs * q.x + p.srcDist[3] * twiceXY + p.srcDist[4] * (r2 + 2.f * q.x * q.x);
    ray.y = rs * q.y + p.srcDist[4] * twiceXY + p.srcDist[3] * (r2 + 2.f * q.y * q.y);
    ray.x = ray.x * p.srcFocLenX + p.srcX0;
    ray.y = ray.y * p.srcFocLenY + p.srcY0;

    return PackColor(tex2D<float4>(srcTex, ray.x, ray.y));  /* Bilinearly interpolate the desired pixel value */
}


/********************************************************************************
 ********************************************************************************
 **                     GET RAY FROM DESTINATION COORDINATE                    **
 ********************************************************************************
 ********************************************************************************/

template<nvwarpSurface_t> bool GetRay(unsigned x, unsigned y, Warp360KernelParams& p, float3 &ray);

/********************************************************************************
 * Get the ray for a pushbroom projection
 ********************************************************************************/

template<> __device__ inline bool GetRay<NVSURF_PUSHBROOM>(unsigned x, unsigned y, Warp360KernelParams& p, float3 &ray)
{
    ray.x = ((float)x - p.dstX0) * p.dstInvFocLenX;         /* Compute coordinates (u,v) on plane w=1 */
    ray.y = ((float)y - p.dstY0) * p.dstInvFocLenY;
    ray.x *= sqrtf(ray.x * ray.x * p.control + 1.f);
    ray.z = 1.f;
    return true;
}


/********************************************************************************
 * Get the ray for a vertically panned radial cylinder projection
 ********************************************************************************/

template<> __device__ inline bool GetRay<NVSURF_ROTCYLINDER>(unsigned x, unsigned y, Warp360KernelParams& p, float3 &ray)
{
    ray.x = ((float)y - p.dstY0) * p.dstInvFocLenX;         /* Radial angle */
    ray.y = sinf(ray.x);                                    /* Vertical coordinate TODO: sincos() */
    ray.z = cosf(ray.x);                                    /* Depth coordinate */
    ray.x = ((float)x - p.dstX0) * p.dstInvFocLenY;         /* Axial coordinate */
    return true;
}


/********************************************************************************
 * Get the ray for a Panini projection
 * This is not normalized.
 ********************************************************************************/

template<> __device__ inline bool GetRay<NVSURF_PANINI>(unsigned x, unsigned y, Warp360KernelParams& p, float3 &ray)
{
    float scale, cosLon;
    scale = 1.f / (p.control + 1.f);
    ray.x = (x - p.dstX0) * p.dstInvFocLenX * scale;
    ray.y = (y - p.dstY0) * p.dstInvFocLenY * scale;
    if (ray.x == 0.f)
    {
        cosLon = 1.f;
    }
    else
    {
        double  kk = (double)ray.x * ray.x,
                del = (1. - (double)p.control * p.control) * kk + 1.;
        if (del < 0.)
            return false;
        cosLon = (float)((-kk * p.control + sqrt(del)) / (kk + 1.));
    }
    scale = p.control + cosLon;
    ray.y *= scale;                                     /* tan(latitude) */
    ray.z = 1.f / sqrtf(1.f + ray.y * ray.y);           /* cos(latitude) */
    ray.y *= ray.z;                                     /* sin(latitude) */
    ray.x *= scale * ray.z;                             /* sin(longitude) * cos(latitude) */
    ray.z *= cosLon;                                    /* cos(longitude) * cos(latitude) */
    return true;
}


/********************************************************************************
 * Get the ray for a Perspective projection.
 * This is not normalized.
 ********************************************************************************/

template<> __device__ inline bool GetRay<NVSURF_PERSPECTIVE>(unsigned x, unsigned y, Warp360KernelParams& p, float3 &ray)
{
    ray.x = (x - p.dstX0) * p.dstInvFocLenX;
    ray.y = (y - p.dstY0) * p.dstInvFocLenY;
    ray.z = 1.f;
    return true;
}


/********************************************************************************
 * Get the ray for a Cylindrical projection
 * This is not normalized.
 ********************************************************************************/

template<> __device__ inline bool GetRay<NVSURF_CYLINDER>(unsigned x, unsigned y, Warp360KernelParams& p, float3 &ray)
{
    ray.y = (x - p.dstX0) * p.dstInvFocLenX;            /* Temporary azimuth angle */
    ray.x = sinf(ray.y);                                /* Radial vector TODO: sincos() */
    ray.z = cosf(ray.y);
    ray.y = (y - p.dstY0) * p.dstInvFocLenY;            /* Axial coordinate */
    return true;
}


/********************************************************************************
 * Get the ray for an equirectangular projection
 * This is not normalized.
 ********************************************************************************/

template<> __device__ inline bool GetRay<NVSURF_EQUIRECT>(unsigned x, unsigned y, Warp360KernelParams& p, float3 &ray)
{
    ray.y = (x - p.dstX0) * p.dstInvFocLenX;            /* Temporary azimuth angle */
    ray.x = sinf(ray.y);                                /* Radial vector TODO: sincos() */
    ray.z = cosf(ray.y);
    ray.y = tanf((y - p.dstY0) * p.dstInvFocLenY);      /* Tilt angle */
    return true;
}


/********************************************************************************
 * Get the ray for a fisheye projection
 * This is normalized.
 ********************************************************************************/

template<> __device__ inline bool GetRay<NVSURF_FISHEYE>(unsigned x, unsigned y, Warp360KernelParams& p, float3 &ray)
{
    ray.x = (x - p.dstX0) * p.dstInvFocLenX;
    ray.y = (y - p.dstY0) * p.dstInvFocLenY;
    ray.z = hypotf(ray.x, ray.y);
    if (0.f != ray.z)
    {
        float t = sinf(ray.z) / ray.z;
        ray.x *= t;
        ray.y *= t;
    }
    ray.z = cosf(ray.z);
    return true;
}


/********************************************************************************
 * Get the ray for a stereographic projection
 * This is normalized.
 ********************************************************************************/

template<> __device__ inline bool GetRay<NVSURF_STEREOGRAPHIC>(unsigned x, unsigned y, Warp360KernelParams& p, float3 &ray)
{
    float rr;
    rr = 1.f / (p.control + 1.f);
    ray.x = (x - p.dstX0) * p.dstInvFocLenX * rr;           /* Normalized coordinates */
    ray.y = (y - p.dstY0) * p.dstInvFocLenY * rr;
    rr = ray.x * ray.x + ray.y * ray.y;                     /* Magnitude */
    double d = (1.0 - p.control * p.control) * rr + 1.0;    /* Discriminant */
    if (d < 0.0)
        return false;                                       /* Out-of-bounds */
    d = (sqrt(d) - p.control * rr) / (1.0 + rr);            /* Cosine of the inclination angle */
    ray.z = (float)d;                                       /* The z component of the ray *is* the cosine */
    rr = sqrt((1.0 - d * d) / rr);                          /* Compute the sine and divide by the magnitude */
    ray.x *= rr;                                            /* Normalized ray */
    ray.y *= rr;
    return true;
}


/********************************************************************************
 ********************************************************************************
 **                               PROJECTIONS                                  **
 ********************************************************************************
 ********************************************************************************/

/* Surface template */

template<nvwarpSurface_t Tsrc, nvwarpSurface_t Tdst> __global__ void
DeviceWarp(Warp360KernelParams p, cudaTextureObject_t srcTex, cudaSurfaceObject_t dstSurface)
{
    unsigned x = blockIdx.x * blockDim.x + threadIdx.x;             /* Get destination coordinates */
    unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.dstWidth || y >= p.dstHeight)
        return;
    float3 ray;                                                     /* Compute ray */
    uchar4 pix = { 0, 0, 0, 0 };
    if (GetRay<Tdst>(x, y, p, ray))
        pix = GetPixel<Tsrc>(ray, p, srcTex);
    surf2Dwrite(pix, dstSurface, x * sizeof(uchar4), y);
}

template<nvwarpSurface_t Tsrc, nvwarpSurface_t Tdst> __host__ cudaError_t
Warp(dim3 dim_grid, dim3 dim_block, cudaStream_t stream, const Warp360KernelParams *params, cudaTextureObject_t srcTex, cudaSurfaceObject_t dstSurface)
{
    DeviceWarp<Tsrc, Tdst><<<dim_grid, dim_block, 0, stream>>>(*params, srcTex, dstSurface);
    return cudaGetLastError();
}


/* Buffer template */

template<nvwarpSurface_t Tsrc, nvwarpSurface_t Tdst> __global__ void
DeviceWarp(Warp360KernelParams p, cudaTextureObject_t srcTex, void *dstAddr, size_t dstRowBytes)
{
    unsigned x = blockIdx.x * blockDim.x + threadIdx.x;             /* Get destination coordinates */
    unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.dstWidth || y >= p.dstHeight)
        return;
    float3 ray;                                                     /* Compute ray */
    uchar4 pix = { 0, 0, 0, 0 };
    if (GetRay<Tdst>(x, y, p, ray))
        pix = GetPixel<Tsrc>(ray, p, srcTex);
    *((uchar4*)((char*)dstAddr + y * dstRowBytes + x * sizeof(uchar4))) = pix;
}

template<nvwarpSurface_t Tsrc, nvwarpSurface_t Tdst> __host__ cudaError_t
Warp(dim3 dim_grid, dim3 dim_block, cudaStream_t stream,
    const Warp360KernelParams *params, cudaTextureObject_t srcTex, void *dstAddr, size_t dstRowBytes)
{
    DeviceWarp<Tsrc, Tdst><<<dim_grid, dim_block, 0, stream>>>(*params, srcTex, dstAddr, dstRowBytes);
    return cudaGetLastError();
}


/********************************************************************************
 *                                INSTANTIATIONS                                *
 ********************************************************************************/

// Equirectangular --> Cyl
template __host__ cudaError_t Warp<NVSURF_EQUIRECT, NVSURF_CYLINDER>
    (dim3 dim_grid, dim3 dim_block, cudaStream_t stream, const Warp360KernelParams *params, cudaTextureObject_t srcTex, cudaSurfaceObject_t dstSurface);
template __host__ cudaError_t Warp<NVSURF_EQUIRECT, NVSURF_CYLINDER>
    (dim3 dim_grid, dim3 dim_block, cudaStream_t stream, const Warp360KernelParams *params, cudaTextureObject_t srcTex, void *dstAddr, size_t dstRowBytes);

// Equirectangular --> Equirectangular
template __host__ cudaError_t Warp<NVSURF_EQUIRECT, NVSURF_EQUIRECT>
    (dim3 dim_grid, dim3 dim_block, cudaStream_t stream, const Warp360KernelParams *params, cudaTextureObject_t srcTex, cudaSurfaceObject_t dstSurface);
template __host__ cudaError_t Warp<NVSURF_EQUIRECT, NVSURF_EQUIRECT>
    (dim3 dim_grid, dim3 dim_block, cudaStream_t stream, const Warp360KernelParams *params, cudaTextureObject_t srcTex, void *dstAddr, size_t dstRowBytes);

// Equirectangular --> Fisheye
template __host__ cudaError_t Warp<NVSURF_EQUIRECT, NVSURF_FISHEYE>
    (dim3 dim_grid, dim3 dim_block, cudaStream_t stream, const Warp360KernelParams *params, cudaTextureObject_t srcTex, cudaSurfaceObject_t dstSurface);
template __host__ cudaError_t Warp<NVSURF_EQUIRECT, NVSURF_FISHEYE>
    (dim3 dim_grid, dim3 dim_block, cudaStream_t stream, const Warp360KernelParams *params, cudaTextureObject_t srcTex, void *dstAddr, size_t dstRowBytes);

// Equirectangular --> Panini
template __host__ cudaError_t Warp<NVSURF_EQUIRECT, NVSURF_PANINI>
    (dim3 dim_grid, dim3 dim_block, cudaStream_t stream, const Warp360KernelParams *params, cudaTextureObject_t srcTex, cudaSurfaceObject_t dstSurface);
template __host__ cudaError_t Warp<NVSURF_EQUIRECT, NVSURF_PANINI>
    (dim3 dim_grid, dim3 dim_block, cudaStream_t stream, const Warp360KernelParams *params, cudaTextureObject_t srcTex, void *dstAddr, size_t dstRowBytes);

// Equirectangular --> Perspective
template __host__ cudaError_t Warp<NVSURF_EQUIRECT, NVSURF_PERSPECTIVE>
    (dim3 dim_grid, dim3 dim_block, cudaStream_t stream, const Warp360KernelParams *params, cudaTextureObject_t srcTex, cudaSurfaceObject_t dstSurface);
template __host__ cudaError_t Warp<NVSURF_EQUIRECT, NVSURF_PERSPECTIVE>
    (dim3 dim_grid, dim3 dim_block, cudaStream_t stream, const Warp360KernelParams *params, cudaTextureObject_t srcTex, void *dstAddr, size_t dstRowBytes);

// Equirectangular --> Pushbroom
template __host__ cudaError_t Warp<NVSURF_EQUIRECT, NVSURF_PUSHBROOM>
    (dim3 dim_grid, dim3 dim_block, cudaStream_t stream, const Warp360KernelParams *params, cudaTextureObject_t srcTex, cudaSurfaceObject_t dstSurface);
template __host__ cudaError_t Warp<NVSURF_EQUIRECT, NVSURF_PUSHBROOM>
    (dim3 dim_grid, dim3 dim_block, cudaStream_t stream, const Warp360KernelParams *params, cudaTextureObject_t srcTex, void *dstAddr, size_t dstRowBytes);

// Equirectangular --> Stereographic
template __host__ cudaError_t Warp<NVSURF_EQUIRECT, NVSURF_STEREOGRAPHIC>
    (dim3 dim_grid, dim3 dim_block, cudaStream_t stream, const Warp360KernelParams *params, cudaTextureObject_t srcTex, cudaSurfaceObject_t dstSurface);
template __host__ cudaError_t Warp<NVSURF_EQUIRECT, NVSURF_STEREOGRAPHIC>
    (dim3 dim_grid, dim3 dim_block, cudaStream_t stream, const Warp360KernelParams *params, cudaTextureObject_t srcTex, void *dstAddr, size_t dstRowBytes);

// Equirectangular --> RotCyl
template __host__ cudaError_t Warp<NVSURF_EQUIRECT, NVSURF_ROTCYLINDER>
    (dim3 dim_grid, dim3 dim_block, cudaStream_t stream, const Warp360KernelParams *params, cudaTextureObject_t srcTex, cudaSurfaceObject_t dstSurface);
template __host__ cudaError_t Warp<NVSURF_EQUIRECT, NVSURF_ROTCYLINDER>
    (dim3 dim_grid, dim3 dim_block, cudaStream_t stream, const Warp360KernelParams *params, cudaTextureObject_t srcTex, void *dstAddr, size_t dstRowBytes);


// Fisheye --> Equirectangular
template __host__ cudaError_t Warp<NVSURF_FISHEYE, NVSURF_EQUIRECT>
    (dim3 dim_grid, dim3 dim_block, cudaStream_t stream, const Warp360KernelParams *params, cudaTextureObject_t srcTex, cudaSurfaceObject_t dstSurface);
template __host__ cudaError_t Warp<NVSURF_FISHEYE, NVSURF_EQUIRECT>
    (dim3 dim_grid, dim3 dim_block, cudaStream_t stream, const Warp360KernelParams *params, cudaTextureObject_t srcTex, void *dstAddr, size_t dstRowBytes);

// Fisheye --> Cylinder
template __host__ cudaError_t Warp<NVSURF_FISHEYE, NVSURF_CYLINDER>
    (dim3 dim_grid, dim3 dim_block, cudaStream_t stream, const Warp360KernelParams *params, cudaTextureObject_t srcTex, cudaSurfaceObject_t dstSurface);
template __host__ cudaError_t Warp<NVSURF_FISHEYE, NVSURF_CYLINDER>
    (dim3 dim_grid, dim3 dim_block, cudaStream_t stream, const Warp360KernelParams *params, cudaTextureObject_t srcTex, void *dstAddr, size_t dstRowBytes);

// Fisheye --> Fisheye
template __host__ cudaError_t Warp<NVSURF_FISHEYE, NVSURF_FISHEYE>
    (dim3 dim_grid, dim3 dim_block, cudaStream_t stream, const Warp360KernelParams *params, cudaTextureObject_t srcTex, cudaSurfaceObject_t dstSurface);
template __host__ cudaError_t Warp<NVSURF_FISHEYE, NVSURF_FISHEYE>
    (dim3 dim_grid, dim3 dim_block, cudaStream_t stream, const Warp360KernelParams *params, cudaTextureObject_t srcTex, void *dstAddr, size_t dstRowBytes);

// Fisheye --> Panini
template __host__ cudaError_t Warp<NVSURF_FISHEYE, NVSURF_PANINI>
    (dim3 dim_grid, dim3 dim_block, cudaStream_t stream, const Warp360KernelParams *params, cudaTextureObject_t srcTex, cudaSurfaceObject_t dstSurface);
template __host__ cudaError_t Warp<NVSURF_FISHEYE, NVSURF_PANINI>
    (dim3 dim_grid, dim3 dim_block, cudaStream_t stream, const Warp360KernelParams *params, cudaTextureObject_t srcTex, void *dstAddr, size_t dstRowBytes);

// Fisheye --> Pushbroom
template __host__ cudaError_t Warp<NVSURF_FISHEYE, NVSURF_PUSHBROOM>
    (dim3 dim_grid, dim3 dim_block, cudaStream_t stream, const Warp360KernelParams *params, cudaTextureObject_t srcTex, cudaSurfaceObject_t dstSurface);
template __host__ cudaError_t Warp<NVSURF_FISHEYE, NVSURF_PUSHBROOM>
    (dim3 dim_grid, dim3 dim_block, cudaStream_t stream, const Warp360KernelParams *params, cudaTextureObject_t srcTex, void *dstAddr, size_t dstRowBytes);

// Fisheye --> Perspective
template __host__ cudaError_t Warp<NVSURF_FISHEYE, NVSURF_PERSPECTIVE>
    (dim3 dim_grid, dim3 dim_block, cudaStream_t stream, const Warp360KernelParams *params, cudaTextureObject_t srcTex, cudaSurfaceObject_t dstSurface);
template __host__ cudaError_t Warp<NVSURF_FISHEYE, NVSURF_PERSPECTIVE>
    (dim3 dim_grid, dim3 dim_block, cudaStream_t stream, const Warp360KernelParams *params, cudaTextureObject_t srcTex, void *dstAddr, size_t dstRowBytes);

// Fisheye --> Vertically-panned Cylinder
template __host__ cudaError_t Warp<NVSURF_FISHEYE, NVSURF_ROTCYLINDER>
    (dim3 dim_grid, dim3 dim_block, cudaStream_t stream, const Warp360KernelParams *params, cudaTextureObject_t srcTex, cudaSurfaceObject_t dstSurface);
template __host__ cudaError_t Warp<NVSURF_FISHEYE, NVSURF_ROTCYLINDER>
    (dim3 dim_grid, dim3 dim_block, cudaStream_t stream, const Warp360KernelParams *params, cudaTextureObject_t srcTex, void *dstAddr, size_t dstRowBytes);


// Perspective --> Equirectangular
template __host__ cudaError_t Warp<NVSURF_PERSPECTIVE, NVSURF_EQUIRECT>
    (dim3 dim_grid, dim3 dim_block, cudaStream_t stream, const Warp360KernelParams *params, cudaTextureObject_t srcTex, cudaSurfaceObject_t dstSurface);
template __host__ cudaError_t Warp<NVSURF_PERSPECTIVE, NVSURF_EQUIRECT>
    (dim3 dim_grid, dim3 dim_block, cudaStream_t stream, const Warp360KernelParams *params, cudaTextureObject_t srcTex, void *dstAddr, size_t dstRowBytes);

// Perspective --> Panini
template __host__ cudaError_t Warp<NVSURF_PERSPECTIVE, NVSURF_PANINI>
    (dim3 dim_grid, dim3 dim_block, cudaStream_t stream, const Warp360KernelParams *params, cudaTextureObject_t srcTex, cudaSurfaceObject_t dstSurface);
template __host__ cudaError_t Warp<NVSURF_PERSPECTIVE, NVSURF_PANINI>
    (dim3 dim_grid, dim3 dim_block, cudaStream_t stream, const Warp360KernelParams *params, cudaTextureObject_t srcTex, void *dstAddr, size_t dstRowBytes);

// Perspective --> Perspective
//#define OPTIMIZE_PERSPECTIVE
#ifndef OPTIMIZE_PERSPECTIVE
template __host__ cudaError_t Warp<NVSURF_PERSPECTIVE, NVSURF_PERSPECTIVE>
    (dim3 dim_grid, dim3 dim_block, cudaStream_t stream, const Warp360KernelParams *params, cudaTextureObject_t srcTex, cudaSurfaceObject_t dstSurface);
template __host__ cudaError_t Warp<NVSURF_PERSPECTIVE, NVSURF_PERSPECTIVE>
    (dim3 dim_grid, dim3 dim_block, cudaStream_t stream, const Warp360KernelParams *params, cudaTextureObject_t srcTex, void *dstAddr, size_t dstRowBytes);
#else /* OPTIMIZE_PERSPECTIVE */

__global__ void
DeviceWarp_PERSPECTIVE_PERSPECTIVE_NOTAN(Warp360KernelParams p, cudaTextureObject_t srcTex, cudaSurfaceObject_t dstSurface)
{
    unsigned x = blockIdx.x * blockDim.x + threadIdx.x;             /* Get destination coordinates */
    unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.dstWidth || y >= p.dstHeight)
        return;
    float3 ray;                                                     /* Compute ray */
    uchar4 pix = { 0, 0, 0, 0 };
    if (GetRay<NVSURF_PERSPECTIVE>(x, y, p, ray))
        pix = GetPixel_PERSPECTIVE_NOTAN(ray, p, srcTex);
    surf2Dwrite(pix, dstSurface, x * sizeof(uchar4), y);
}

template<> __host__ cudaError_t
Warp<NVSURF_PERSPECTIVE, NVSURF_PERSPECTIVE>(dim3 dim_grid, dim3 dim_block, cudaStream_t stream, const Warp360KernelParams *params, cudaTextureObject_t srcTex, cudaSurfaceObject_t dstSurface)
{
    if (params->srcDist[3] || params->srcDist[4])
        DeviceWarp<NVSURF_PERSPECTIVE, NVSURF_PERSPECTIVE><<<dim_grid, dim_block, 0, stream>>>(*params, srcTex, dstSurface);
    else
        DeviceWarp_PERSPECTIVE_PERSPECTIVE_NOTAN<<<dim_grid, dim_block, 0, stream>>>(*params, srcTex, dstSurface);
    return cudaGetLastError();
}

__global__ void
DeviceWarp_PERSPECTIVE_PERSPECTIVE_NOTAN(Warp360KernelParams p, cudaTextureObject_t srcTex, void *dstAddr, size_t dstRowBytes)
{
    unsigned x = blockIdx.x * blockDim.x + threadIdx.x;             /* Get destination coordinates */
    unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.dstWidth || y >= p.dstHeight)
        return;
    float3 ray;                                                     /* Compute ray */
    uchar4 pix = { 0, 0, 0, 0 };
    if (GetRay<NVSURF_PERSPECTIVE>(x, y, p, ray))
        pix = GetPixel_PERSPECTIVE_NOTAN(ray, p, srcTex);
    *((uchar4*)((char*)dstAddr + y * dstRowBytes + x * sizeof(uchar4))) = pix;
}

template<> __host__ cudaError_t
Warp<NVSURF_PERSPECTIVE, NVSURF_PERSPECTIVE>(dim3 dim_grid, dim3 dim_block, cudaStream_t stream,
    const Warp360KernelParams *params, cudaTextureObject_t srcTex, void *dstAddr, size_t dstRowBytes)
{
    if (params->srcDist[3] || params->srcDist[4])
        DeviceWarp<NVSURF_PERSPECTIVE, NVSURF_PERSPECTIVE><<<dim_grid, dim_block, 0, stream>>>(*params, srcTex, dstAddr, dstRowBytes);
    else
        DeviceWarp_PERSPECTIVE_PERSPECTIVE_NOTAN<<<dim_grid, dim_block, 0, stream>>>(*params, srcTex, dstAddr, dstRowBytes);
    return cudaGetLastError();
}

#endif /* OPTIMIZE_PERSPECTIVE */


/********************************************************************************
 ********************************************************************************
 **                     CONVERT FROM YUV 4:2:0 NV12 TO RGBA                    **
 ********************************************************************************
 ********************************************************************************/

__global__ void NV12RGBADeviceBuffer(nvwarpYUVRGBParams_t params, const unsigned char *srcY, const unsigned char *srcC, size_t srcRowBytes, uchar4 *dst, size_t dstRowBytes)
{
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;             /* Get destination coordinates */
    unsigned iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= params.width || iy >= params.height)
        return;

    float y = (float)srcY[iy * srcRowBytes + ix];
    unsigned ic = (iy >> 1) * srcRowBytes + (ix & ~1);
    float cb = (float)srcC[ic + 0];
    float cr = (float)srcC[ic + 1];
    float3 rgb;
    if ((params.cLocation == 0) && (ix & 1) && ((ix + 1) < params.width))
    {
        float cb1 = (float)srcC[ic + 2];
        float cr1 = (float)srcC[ic + 3];
        cb = 0.5f * cb + 0.5f * cb1;
        cr = 0.5f * cr + 0.5f * cr1;
    }
    y  -= params.yOffset;
    cb -= params.cOffset;
    cr -= params.cOffset;
    rgb.x = params.ry * y + params.rcb * cb + params.rcr * cr;
    rgb.y = params.gy * y + params.gcb * cb + params.gcr * cr;
    rgb.z = params.by * y + params.bcb * cb + params.bcr * cr;
    *((uchar4*)((char*)dst + iy * dstRowBytes + ix * sizeof(uchar4))) = PackColor(rgb);
}


__host__ void NV12RGBABuffer(cudaStream_t stream, const nvwarpYUVRGBParams_t *params, const unsigned char *yuv, size_t yuvRowBytes, uchar4 *dst, size_t dstRowBytes)
{
    dim3 dimBlock(16, 8);
    dim3 dimGrid((params->width + 0xf) >> 4, (params->height + 7) >> 3);
    const unsigned char *srcY = (const unsigned char*)yuv;
    const unsigned char *srcC = yuv + params->height * yuvRowBytes;

    NV12RGBADeviceBuffer<<<dimGrid, dimBlock, 0, stream>>>(*params, srcY, srcC, yuvRowBytes, dst,  dstRowBytes);
}


