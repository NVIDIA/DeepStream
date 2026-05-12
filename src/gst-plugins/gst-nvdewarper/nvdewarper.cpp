/**
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
#include <string.h>

#include <sstream>
#include <vector>

#include "gstnvdewarper.h"
#include "gstnvdsmeta.h"

#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include "nvdewarper.h"
#include "nvdewarper_property_parser.h"

#include "NVWarp360.h"

#if defined(__aarch64__)
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include "cudaEGL.h"
#endif

#define USE_CUDA_STREAM

/* Dewarper #defines */
#ifndef   M_PI
#define M_PI                3.1415926535897932385
#endif /* M_PI */
#ifndef   M_2PI
#define M_2PI               6.2831853071795864769
#endif /* M_2PI */
#define F_PI                  ((gfloat)M_PI)
#define F_2PI                 ((gfloat)M_2PI)
#define F_PI_6                ((gfloat)(M_PI/6.))
#define D_RADIANS_PER_DEGREE  (M_PI / 180.)
#define D_DEGREES_PER_RADIAN  (180. / M_PI)
#define F_RADIANS_PER_DEGREE  ((gfloat)D_RADIANS_PER_DEGREE)
#define F_DEGREES_PER_RADIAN  ((gfloat)D_DEGREES_PER_RADIAN)

#define ITER_FACTOR 1.01
#define NUM_WARP_TYPES  (NVDS_META_SURFACE_EQUIRECT_VERTCYLINDER+1)

// This array maps the enum "NvDsSurfaceType" to enum "nvwarpType_t"
const nvwarpType_t  NvDsSurfaceType_To_nvwarpType_t[NUM_WARP_TYPES]
{
  NVWARP_NONE,                    //NVDS_META_SURFACE_NONE=0,
  NVWARP_FISHEYE_PUSHBROOM,       //NVDS_META_SURFACE_FISH_PUSHBROOM=1,
  NVWARP_FISHEYE_ROTCYLINDER,     //NVDS_META_SURFACE_FISH_VERTCYL=2,
  NVWARP_PERSPECTIVE_PERSPECTIVE, //NVDS_META_SURFACE_PERSPECTIVE_PERSPECTIVE=3,
  NVWARP_FISHEYE_PERSPECTIVE,     //NVDS_META_SURFACE_FISH_PERSPECTIVE=4,
  NVWARP_FISHEYE_FISHEYE,         //NVDS_META_SURFACE_FISH_FISH=5,
  NVWARP_FISHEYE_CYLINDER,        //NVDS_META_SURFACE_FISH_CYL=6,
  NVWARP_FISHEYE_EQUIRECT,        //NVDS_META_SURFACE_FISH_EQUIRECT=7,
  NVWARP_FISHEYE_PANINI ,         //NVDS_META_SURFACE_FISH_PANINI=8,
  NVWARP_PERSPECTIVE_EQUIRECT,    //NVDS_META_SURFACE_PERSPECTIVE_EQUIRECT=9,
  NVWARP_PERSPECTIVE_PANINI,      //NVDS_META_SURFACE_PERSPECTIVE_PANINI=10,
  NVWARP_EQUIRECT_CYLINDER ,      //NVDS_META_SURFACE_EQUIRECT_CYLINDER=11,
  NVWARP_EQUIRECT_EQUIRECT,       //NVDS_META_SURFACE_EQUIRECT_EQUIRECT=12,
  NVWARP_EQUIRECT_FISHEYE,        //NVDS_META_SURFACE_EQUIRECT_FISHEYE=13,
  NVWARP_EQUIRECT_PANINI,         //NVDS_META_SURFACE_EQUIRECT_PANINI=14,
  NVWARP_EQUIRECT_PERSPECTIVE,    //NVDS_META_SURFACE_EQUIRECT_PERSPECTIVE=15,
  NVWARP_EQUIRECT_PUSHBROOM,      //NVDS_META_SURFACE_EQUIRECT_PUSHBROOM=16,
  NVWARP_EQUIRECT_STEREOGRAPHIC,  //NVDS_META_SURFACE_EQUIRECT_STEREOGRAPHIC=17,
  NVWARP_EQUIRECT_ROTCYLINDER     //NVDS_META_SURFACE_EQUIRECT_VERTCYLINDER=18
};

struct Buffer
{
  const unsigned *ptr;
  unsigned width;
  unsigned height;
  unsigned rowBytes;
};

/**
 * @brief Wrapper over the Warp360 library calls.
 */
struct WarpWrapper
{
    WarpWrapper()
    {
        nvwarpCreateInstance(&_warper);
    }

    ~WarpWrapper()
    {
        if (_warper) nvwarpDestroyInstance(_warper);
    }

    nvwarpResult setParams(const nvwarpParams_t *params)
    {
      return nvwarpSetParams(_warper, params);
    }

    nvwarpResult warp(cudaStream_t stream, cudaTextureObject_t srcTex,
        void *dstAddr, size_t dstRowBytes)
    {
      return nvwarpWarpBuffer(_warper, stream, srcTex, dstAddr, dstRowBytes);
    }

    void getSrcPrincipalPoint(float xy[2], bool relToCenter) const
    {
      nvwarpGetSrcPrincipalPoint(_warper, xy, relToCenter);
    }

    void setSrcPrincipalPoint(const float xy[2], bool relToCenter)
    {
      nvwarpSetSrcPrincipalPoint(_warper, xy, relToCenter);
    }

    void getDstPrincipalPoint(float xy[2], bool relToCenter) const
    {
      nvwarpGetDstPrincipalPoint(_warper, xy, relToCenter);
    }

    void setDstPrincipalPoint(const float xy[2], bool relToCenter)
    {
      nvwarpSetDstPrincipalPoint(_warper, xy, relToCenter);
    }

    void setDstFocalLength(float fl, float fy)
    {
      nvwarpSetDstFocalLengths(_warper, fl, fy);
    }

    void setSrcFocalLength(float fx, float fy)
    {
      nvwarpSetSrcFocalLengths(_warper, fx, fy);
    }

    void getSrcFocalLength(float *fx, float *fy)
    {
      *fx = nvwarpGetSrcFocalLength(_warper, fy);
    }
    void SetRotation(const float *R)
    {
      nvwarpSetRotation(_warper, R);
    }

    nvwarpHandle _warper; /**< Opaque pointer to the Warp360 library handle. Populated by the constructor of the class */
};

/********************************************************************************
 * Dewarp_Buffer
 * Initialize the warper, set advanced configurations and call the core warp library function
 ********************************************************************************/

static cudaError Dewarp_Buffer(const Gstnvdewarper *nvdewarper,
    const Buffer &src, const Buffer &dst,
    const nvwarpParams_t &warparams, const NvDewarperParams *surfaceParams
)
{
  cudaChannelFormatDesc   formatDesc  = { 8, 8, 8, 8, cudaChannelFormatKindUnsigned };   // format descriptor for uchar4
  cudaResourceDesc        srcResDesc  = {},
      dstResDesc  = {};
  cudaTextureDesc         srcTexDesc  = {};
  void *srcBuffer=nullptr, *dstBuffer=nullptr;
  cudaTextureObject_t     srcTex      = 0;
  gint                     err = 0;
  cudaError_t             cudaErr;
  size_t                  cuSrcRowBytes,  cuDstRowBytes;
  dim3                    dimGrid,        dimBlock;
  WarpWrapper             warper;

  err = err;

  /* Set warper parameters */
  warper.setParams(&warparams);

  // If focal lengths are specified for both X & Y seperately, set them here
  if((surfaceParams->dewarpFocalLength[0]) && (surfaceParams->dewarpFocalLength[1]))
  {
    warper.setSrcFocalLength(surfaceParams->dewarpFocalLength[0], surfaceParams->dewarpFocalLength[1]);
  }

    // In case viewing angles are not provided keep same Focal Length and PPoint (preserves detail and symmetry)
    if (warparams.topAngle == 0 && warparams.bottomAngle == 0){
      float xy[2];
      warper.getSrcPrincipalPoint(xy,true);
      warper.setDstPrincipalPoint(xy,true);
      warper.getSrcFocalLength(&xy[0], &xy[1]);
      warper.setDstFocalLength(xy[0], xy[1]);
    }
  if(surfaceParams->rot_matrix_valid)
    warper.SetRotation(surfaceParams->rot_matrix);

  if(surfaceParams->dstFocalLength[0])
     warper.setDstFocalLength(surfaceParams->dstFocalLength[0], surfaceParams->dstFocalLength[1]);
  if(surfaceParams->dstPrincipalPoint[0] && surfaceParams->dstPrincipalPoint[1])
    warper.setDstPrincipalPoint(surfaceParams->dstPrincipalPoint,false);

  /* Allocate src Buffer and texture */
  cuSrcRowBytes = src.rowBytes;

  srcBuffer = (void *) src.ptr;
  srcResDesc.resType                  = cudaResourceTypePitch2D;
  srcResDesc.res.pitch2D.devPtr       = srcBuffer;
  srcResDesc.res.pitch2D.desc         = formatDesc;
  srcResDesc.res.pitch2D.width        = src.width;
  srcResDesc.res.pitch2D.height       = src.height;
  srcResDesc.res.pitch2D.pitchInBytes = cuSrcRowBytes;
  srcTexDesc.addressMode[0]           = (surfaceParams->addressMode == 1)? cudaAddressModeBorder : cudaAddressModeClamp;
  srcTexDesc.addressMode[1]           = srcTexDesc.addressMode[0];

  if (surfaceParams->addressMode == 1)
    srcTexDesc.borderColor[0]         = srcTexDesc.borderColor[1] = srcTexDesc.borderColor[2] = srcTexDesc.borderColor[3] = 0;

  srcTexDesc.filterMode               = cudaFilterModeLinear;
  srcTexDesc.readMode                 = cudaReadModeNormalizedFloat;
  srcTexDesc.normalizedCoords         = false;

  cudaErr = cudaCreateTextureObject(&srcTex, &srcResDesc, &srcTexDesc, nullptr);
  BAIL_IF_FALSE(cudaSuccess == cudaErr, err, (gint)cudaErr);

  /* Allocate dst array */
  cuDstRowBytes = dst.rowBytes;
  dstBuffer = (void *) dst.ptr;
  dstResDesc.resType                  = cudaResourceTypePitch2D;
  dstResDesc.res.pitch2D.devPtr       = dstBuffer;
  dstResDesc.res.pitch2D.desc         = formatDesc;
  dstResDesc.res.pitch2D.width        = warparams.dstWidth;
  dstResDesc.res.pitch2D.height       = warparams.dstHeight;
  dstResDesc.res.pitch2D.pitchInBytes = cuDstRowBytes;

  /* Test measurement with 10 iterations */
#ifdef USE_CUDA_STREAM
  warper.warp(nvdewarper->stream, srcTex, dstBuffer, cuDstRowBytes);
  cudaErr = cudaStreamSynchronize(nvdewarper->stream);
#else
  warper.warp(0, srcTex, dstBuffer, cuDstRowBytes);
  cudaErr = cudaDeviceSynchronize();
#endif

  bail:
  /* Dispose */
  if (srcTex)
    cudaDestroyTextureObject(srcTex);

  return cudaErr;
}

/********************************************************************************
 * Dewarp
 ********************************************************************************/

static cudaError Dewarp(Gstnvdewarper *nvdewarper,
    NvBufSurface *in_surface, const NvDewarperParams * surfaceParams)
{
  nvwarpParams_t warparams;
  cudaError_t cudaErr = cudaSuccess;
  Buffer src, dst;

  src.ptr = (const unsigned int*)in_surface->surfaceList[0].dataPtr;
  src.width = in_surface->surfaceList[0].planeParams.width[0];
  src.height = in_surface->surfaceList[0].planeParams.height[0];
  src.rowBytes = in_surface->surfaceList[0].planeParams.pitch[0];

  dst.ptr = (const guint*) (surfaceParams->surface);
  dst.width = (surfaceParams->dewarpWidth == 0) ? src.width : surfaceParams->dewarpWidth;
  dst.height = (surfaceParams->dewarpHeight == 0) ? src.height : surfaceParams->dewarpHeight;
  dst.rowBytes = surfaceParams->dewarpPitch;

#if defined(__aarch64__)
  CUresult status;
  CUeglFrame eglFrame = {};
  CUgraphicsResource pResource = NULL;
  EGLImageKHR eglimage_src = NULL;

  if(in_surface->memType == NVBUF_MEM_SURFACE_ARRAY) {
    if (in_surface->surfaceList[0].mappedAddr.eglImage == NULL)
    {
      NvBufSurfaceMapEglImage (in_surface, 0);
    }
    eglimage_src = in_surface->surfaceList[0].mappedAddr.eglImage;

    status = cuGraphicsEGLRegisterImage(&pResource, eglimage_src,
        CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
    if (status != CUDA_SUCCESS)
    {
      printf("cuGraphicsEGLRegisterImage failed: %d, cuda process stop\n",
          status);
      exit (-1);
    }

    status = cuGraphicsResourceGetMappedEglFrame(&eglFrame, pResource, 0, 0);
    if (status != CUDA_SUCCESS)
    {
      printf("cuGraphicsSubResourceGetMappedArray failed\n");
    }

    src.ptr = (const unsigned *) eglFrame.frame.pPitch[0];
  }
#endif

  warparams.type          = NvDsSurfaceType_To_nvwarpType_t[surfaceParams->projection_type];
  warparams.srcWidth      = src.width;
  warparams.srcHeight     = src.height;
  warparams.srcX0         = (surfaceParams->src_x0 == 0)? (src.width  - 1) * .5f : surfaceParams->src_x0;
  warparams.srcY0         = (surfaceParams->src_y0 == 0)? (src.height - 1) * .5f : surfaceParams->src_y0;
  if (surfaceParams->dewarpFocalLength[0] == 0 && surfaceParams->srcFov > 0){
    float ang = surfaceParams->srcFov * (.5f * F_RADIANS_PER_DEGREE);
    float rad = ((surfaceParams->srcFov == 180.f) ? src.height : (src.height - 1)) * .5F;

    if (nvwarpComputeParamsSrcFocalLength(&warparams, ang,rad)){ //Computes and sets srcFocalLen
      GST_INFO_OBJECT (nvdewarper,
          "Computing source Focal Length from source Field of View failed. "
          "Setting Focal Length to zero\n");
      warparams.srcFocalLen = 0.f;
    }

  } else{
    warparams.srcFocalLen = surfaceParams->dewarpFocalLength[0];
  }

  /* Set warper parameters */
  warparams.dist[0]    = surfaceParams->distortion[0];
  warparams.dist[1]    = surfaceParams->distortion[1];
  warparams.dist[2]    = surfaceParams->distortion[2];
  warparams.dist[3]    = surfaceParams->distortion[3];
  warparams.dist[4]    = surfaceParams->distortion[4];

  warparams.dstWidth = dst.width;
  warparams.dstHeight = dst.height;

  if(surfaceParams->rot_axes[0])
  {
    if((strcmp(surfaceParams->rot_axes, "XYZ")==0) || (strcmp(surfaceParams->rot_axes, "XZY")==0) ||
    (strcmp(surfaceParams->rot_axes, "YXZ")==0) || (strcmp(surfaceParams->rot_axes, "YZX")==0) ||
    (strcmp(surfaceParams->rot_axes, "ZXY")==0) || (strcmp(surfaceParams->rot_axes, "ZYX")==0))
    {
      strcpy(warparams.rotAxes, surfaceParams->rot_axes);
    }
    else
    {
      GST_WARNING_OBJECT (nvdewarper,
        "rot-axes setting is incorrect. Using the default setting : %s", warparams.rotAxes);
    }
  }

  // Map Yaw, pitch and roll to appropriate position in "rotAngles" based on "rot_axes"
  for(int i=0; i<3; i++)
  {
    switch (warparams.rotAxes[i])
    {
        default:
        case 'X': warparams.rotAngles[i] = surfaceParams->pitch * F_RADIANS_PER_DEGREE; break;
        case 'Y': warparams.rotAngles[i] = surfaceParams->yaw * F_RADIANS_PER_DEGREE; break;
        case 'Z': warparams.rotAngles[i] = surfaceParams->roll * F_RADIANS_PER_DEGREE; break;
    }
  }

  warparams.topAngle      = surfaceParams->top_angle  * F_RADIANS_PER_DEGREE;
  warparams.bottomAngle   = surfaceParams->bottom_angle * F_RADIANS_PER_DEGREE;

  warparams.control[0]    = surfaceParams->control;

  cudaErr = Dewarp_Buffer(nvdewarper, src, dst, warparams, surfaceParams);

  if (nvdewarper->dump_frames)
    // Dump output
  {
    guint size = 0;

    if (!nvdewarper->output)
    {
      cuda_ck (cudaMallocHost (&nvdewarper->output,
          (dst.rowBytes * dst.height)));
    }

#ifdef USE_CUDA_STREAM
    cudaErr = cudaStreamSynchronize(nvdewarper->stream);
    GST_INFO_OBJECT (nvdewarper,
        "SPOT %s  DumpFrames i=%d Frame=%d cudaStreamSynchronize cudaErr=%d Stream=%p Completed",
        __func__, surfaceParams->id, nvdewarper->frame_num, cudaErr, nvdewarper->stream);
#endif

    std::ostringstream elem;
    elem << (void*)nvdewarper;

    std::string idx_str = std::to_string (surfaceParams->surface_index);
    std::string tmp = "_" + elem.str() + "_" + std::to_string(dst.rowBytes>>2) + "x"
        + std::to_string(dst.height) + "_" + idx_str;
    std::string fname;

    fname = "Dewarper_Output" + tmp + "_interleaved.rgba";

    size = dst.rowBytes * dst.height;

    cudaMemcpy2D (nvdewarper->output, dst.rowBytes, dst.ptr, dst.rowBytes,
        dst.rowBytes, dst.height, cudaMemcpyDeviceToHost);

    std::ofstream outfile1;
    outfile1.open (fname, std::ofstream::out | std::ofstream::app);
    outfile1.write(reinterpret_cast<gchar*>(nvdewarper->output), size);
    outfile1.close();
  }
#if defined(__aarch64__)
  if(in_surface->memType == NVBUF_MEM_SURFACE_ARRAY) {
    status = cuGraphicsUnregisterResource(pResource);
    if (status != CUDA_SUCCESS) {
      printf ("cuGraphicsEGLUnRegisterResource failed: %d \n", status);
    }
  }
#endif
  GST_INFO_OBJECT (nvdewarper,
      " %s Frame=%d Dewarp for Views=%d cudaErr=%d "
      "Stream=%p Completed", __func__, nvdewarper->frame_num, surfaceParams->id,
      cudaErr, nvdewarper->stream);
  return cudaErr;
}

cudaError gst_nvdewarper_do_dewarp (Gstnvdewarper *nvdewarper,
    NvBufSurface *in_surface, NvBufSurface *out_surface)
{
  gchar context_name[100];
  cudaError cudaErr = cudaSuccess;
  std::vector<NvDewarperParams>::iterator it;
  NvDewarperParams * dewarpParams = NULL;

  for (it = nvdewarper->priv->vecDewarpSurface.begin();
      it != nvdewarper->priv->vecDewarpSurface.end(); it++)
  {
    dewarpParams = &(*it);
    //cout << it->projection_type << " " << it->dewarpWidth << " " << it->dewarpHeight << endl;
    if((it->projection_type >= NVDS_META_SURFACE_FISH_PUSHBROOM)&&(it->projection_type <= NVDS_META_SURFACE_EQUIRECT_VERTCYLINDER))
    {
        if (it->isValid)
        {
          snprintf(context_name, sizeof(context_name), "%s_(Frame=%u)",
              GST_ELEMENT_NAME(nvdewarper), nvdewarper->frame_num);
          //nvtx_helper_push_pop(strcat(context_name,"Dewarp"));

          cuda_ck(Dewarp(nvdewarper, in_surface, dewarpParams));

          // To maintain legacy prints
          if(it->projection_type == NVDS_META_SURFACE_FISH_PUSHBROOM)
          {
            GST_LOG_OBJECT (nvdewarper, "Called Dewarp for Pushbroom GPU=%d Frame=%d Views=%d\n",
              nvdewarper->gpu_id, nvdewarper->frame_num, nvdewarper->num_spot_views);
          }
          else if(it->projection_type == NVDS_META_SURFACE_FISH_VERTCYL)
          {
            GST_LOG_OBJECT (nvdewarper, "Called Dewarp for VertRadCyl GPU=%d Frame=%d Views=%d\n",
              nvdewarper->gpu_id, nvdewarper->frame_num, nvdewarper->num_aisle_views);
          }
          else
          {
            GST_LOG_OBJECT (nvdewarper, "Called Dewarp GPU=%d Frame=%d\n",
              nvdewarper->gpu_id, nvdewarper->frame_num);
          }
          //nvtx_helper_push_pop(NULL);
        }
    }
    else
    {
        g_print ("\n%s: Invalid Projection type (%d) selected \n",
            GST_ELEMENT_NAME(nvdewarper), it->projection_type);
        g_assert(it->projection_type <= NVDS_META_SURFACE_EQUIRECT_VERTCYLINDER);
    }
  }
  cudaStreamSynchronize(nvdewarper->stream);
  return cudaErr;
}

uint32_t gst_nvdewarper_version()
{
  return nvwarpVersion();
}
