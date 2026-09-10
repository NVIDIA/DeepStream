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
//! \file Warp360.h
//! 360 Image and Coordinate Warp SDK.

#ifndef __WARP360_H__
#define __WARP360_H__

#include <cuda.h>
#include <cuda_runtime.h>
#include "NVWarp360.h"

#ifdef __cplusplus


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////                                Warp360                                 ////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


//! Pass parameters via this struct.
//! Many parameters are common. Some may not be used.
struct Warp360Params
{
    //! Warp type enumeration.
    enum WarpType
    {
        warpEquirectBroom           = NVWARP_EQUIRECT_PUSHBROOM,        //!< Equirectangular to pushbroom.
        warpEquirectCyl             = NVWARP_EQUIRECT_CYLINDER,         //!< Equirectangular to cylindrical.
        warpEquirectEquirect        = NVWARP_EQUIRECT_EQUIRECT,         //!< Equirectangular to equirectangular.
        warpEquirectFish            = NVWARP_EQUIRECT_FISHEYE,          //!< Equirectangular to fisheye.
        warpEquirectPanini          = NVWARP_EQUIRECT_PANINI,           //!< Equirectangular to Panini.
        warpEquirectPerspective     = NVWARP_EQUIRECT_PERSPECTIVE,      //!< Equirectangular to perspective.
        warpEquirectStereographic   = NVWARP_EQUIRECT_STEREOGRAPHIC,    //!< Equirectangular to generalized stereographic.
        warpEquirectVertCyl         = NVWARP_EQUIRECT_ROTCYLINDER,      //!< Equirectangular to vertical cylindrical.
        warpFishBroom               = NVWARP_FISHEYE_PUSHBROOM,         //!< Fisheye to pushbroom.
        warpFishCyl                 = NVWARP_FISHEYE_CYLINDER,          //!< Fisheye to horizontally panned cylinder.
        warpFishEquirect            = NVWARP_FISHEYE_EQUIRECT,          //!< Fisheye to equirectangular.
        warpFishFish                = NVWARP_FISHEYE_FISHEYE,           //!< Fisheye to fisheye.
        warpFishPanini              = NVWARP_FISHEYE_PANINI,            //!< Fisheye to Panini.
        warpFishPerspective         = NVWARP_FISHEYE_PERSPECTIVE,       //!< Fisheye to perspective.
        warpFishVertCyl             = NVWARP_FISHEYE_ROTCYLINDER,       //!< Fisheye to vertically panned radial cylinder.
        warpPerspectiveEquirect     = NVWARP_PERSPECTIVE_EQUIRECT,      //!< Perspective to equirectangular.
        warpPerspectivePanini       = NVWARP_PERSPECTIVE_PANINI,        //!< Perspective to Panini.
        warpPerspectivePerspective  = NVWARP_PERSPECTIVE_PERSPECTIVE,   //!< Perspective to perspective.
        warpNone                    = NVWARP_NONE                       //!< sizeof(int)
    };

    // Warp type selector
    WarpType    type;                       //!< The type of the warp.

    // Source specification
    unsigned    srcWidth;                   //!< The width  of the source image.
    unsigned    srcHeight;                  //!< The height of the source image.
    float       srcX0;                      //!< Source center of projection X; frequently (srcWidth  - 1) * 0.5, but srcWidth  * .5 for wraparounds (equirect).
    float       srcY0;                      //!< Source center of projection Y; frequently (srcHeight - 1) * 0.5, but srcHeight * .5 for wraparounds (equirect).
    float       srcFocalLen;                //!< Source focal length.
    float       srcRadius;                  //!< Source circular clipping radius. (default 0 means no clipping) (unimplemented).
    float       srcDist[4];                 //!< Source distortion. (default {0,0,0,0} means no distortion).

    // Destination specification
    unsigned    dstWidth;                   //!< The width  of the destination.
    unsigned    dstHeight;                  //!< The height of the destination.

    // View specification
    float       yaw;                        //!< Yaw   angle. (default 0).
    float       pitch;                      //!< Pitch angle. (default 0).
    float       roll;                       //!< Roll  angle. (default 0).
    float       topAngle;                   //!< Top    angle of view. (default +pi/2)
    float       bottomAngle;                //!< Bottom angle of view. (default -pi/2)

    void*       userData;                   //!< Pointer supplied by the user. (default NULL)

    /* Warp controls */
    float       control[4];                 //!< Projection-specific controls.

    Warp360Params();                        //!< Constructor.
    ~Warp360Params(){}                      //!< Destructor.

    //! Compute and set the srcFocalLen given the srcDist distortion coefficients, plus an angle and corresponding radius.
    //! The srcDist[] coefficients must be initialized prior to the call (suggest {0,0,0,0} for ideal lens).
    //! The focal length converts from angles to pixels at the center of projection,
    //! and is a measure of spherical image resolution also known as angular pixel density (in pixels/radian).
    //! It is only as accurate as the parameters that were supplied.
    //! It can be adjusted by hand, for example to straighten out lines that appeared bowed, but distortion does that as well.
    //! \param[in]   angle   the angle in radians.
    //! \param[in]   radius  the radius, in pixels, corresponding to the angle above.
    //! \return      true if the srcFocalLen was computed successfully, false otherwise.
    //! \note    The focal length is one of the properties of the source, and this API provides a way to convert
    //!          angle and distortion measurements into a focal length. Once calibrated for a given lens and camera,
    //!          it is fixed, and not considered to be a warp control parameter.
    //! \note    The focal length can also be acquired from the image EXIF data without the use of this API.
    //!          In particular, the focalLength tag (37386) yields the focal length in millimeters. Then the
    //!          FocalPlaneXResolution tag (41486), the FocalPlaneXResolution tag (41487), and FocalPlaneResolutionUnit tag (41488)
    //!          can be used to compute the X and Y focal lengths by converting mm to pixels. If the X and Y focal lengths differ,
    //!          it is suggested to use the geometric average focLen = sqrt(focLenX * focLenY).
    bool        computeSrcFocalLength(float angle, float radius);

    //! Compute the angular limits on the source image along the primary axes going through the center of projection.
    //! The Warp360Params must have the following fields filled in:
    //! type, srcWidth, srcHeight, srcX0, srcY0, srcFocalLen, srcDist[4];
    //! \param[out]  minMaxXY    an array of length 4, where the values {minX, maxX, minY, maxY} angles, in radians, are returned.
    //! \return      true if successful; otherwise false.
    //! \note        At the moment, only fisheye, perspective and equirectangular sources are accommodated.
    //! \note        Even though this guarantees that this angular range contains valid pixels along the horizontal and vertical axes
    //!              through the center of projection, this does not guarantee that the same holds true at the corners.
    bool        computeAxialAngleRange(float minMaxXY[4]) const;

    //! Compute the angular limits on the source bounding box along the primary axes going through the center of projection.
    //! The Warp360Params must have the following fields filled in:
    //! type, srcX0, srcY0, srcFocalLen, srcDist[4];
    //! \param[in]   srcBoundingBox  an array of length 4 for describing bounding box in source image {leftTopX, leftTopY, bottomRightX, bottomRightY}.
    //! \param[out]  minMaxXY        an array of length 4, where the values {minX, maxX, minY, maxY} angles, in radians, are returned.
    //! \return      true if successful; otherwise false.
    //! \note        At the moment, only fisheye, perspective and equirectangular sources are accommodated.
    //! \note        Even though this guarantees that this angular range contains valid pixels along the horizontal and vertical axes
    //!              through the center of projection, this does not guarantee that the same holds true at the corners.
    bool        computeAxialAngleRange(const float srcBoundingBox[4], float minMaxXY[4]) const;

    //! Compute the output resolution that matches the source focal length and desired aspect ratio.
    //! The dimensions are computed from the source focal length, so it must have the appropriate value.
    //! \param[in]   aspectRatio the ratio of width/height for the output.
    //! \return      true        if the dstWidth and dstHeight were updated successfully, false if not.
    //! \note    This API provides a suggestion that should be tweaked to meet the needs of the application.
    //!          Enlarging the dimensions will produce a bigger image, but not introduce any new details.
    //!          Reducing the dimensions will not only produce a smaller image, but will lose more details as it shrinks.
    //!          It is not recommended to reduce these dimensions smaller than 1/3, or aliasing will be introduced.
    bool        computeOutputResolution(float aspectRatio);
};


//! Opaque definition.
struct nvwarpObject;

//! The object Warp360.
class Warp360
{
public:

    //! Constructor.
    Warp360();

    ~Warp360();

    //! Set parameters for a warp.
    //! \param[in]   params  pointer to the desired parameters for the warp.
    cudaError_t setParams(const Warp360Params *params);

    //! Get the current values of the warp parameters.
    //! \param[out]  params  pointer to the location where the parameters are to be stored.
    void        getParams(Warp360Params *params) const;

    //! Set the CUDA block size.
    //! \param[in]   dim_block   the desired block size (default 8x8).
    void        setBlock(dim3 dim_block);

    //! Get the current value of the CUDA block size.
    //! \param[out]  dim_block   pointer to a location where the block size is to be stored.
    void        getBlock(dim3 *dim_block) const;

    //! Get the current value of the user data pointer.
    //! \return  the current value of the user data pointer.
    void*       getUserData() const;

    //! Warp an image texture to a surface.
    //! \param[in]  stream      the stream on which to execute the warp.
    //! \param[in]  srcTex      the source texture.
    //! \param[out] dstSurface  the destination surface.
    //! \return     cudaSuccess if successful.
    cudaError_t warp(cudaStream_t stream, cudaTextureObject_t srcTex, cudaSurfaceObject_t dstSurface) const;

    //! Warp an image texture to a buffer.
    //! \param[in]  stream      the stream on which to execute the warp.
    //! \param[in]  srcTex      the source texture.
    //! \param[out] dstAddr     the destination buffer address.
    //! \param[in]  dstRowBytes the byte stride between pixels in the buffer vertically.
    //! \return     cudaSuccess if successful.
    cudaError_t warp(cudaStream_t stream, cudaTextureObject_t srcTex, void *dstAddr, size_t dstRowBytes) const;

    //! Transform coordinates from the input space to the output space.
    //! This works in-place.
    //! \param[in]  numPts      the number of points to be transformed.
    //! \param[in]  inPtsXY     an array of 2D points to be transformed.
    //! \param[out] outPtsXY    a 2D point array of where the transformed 2D points are to be placed.
    //!                         This can be the same as inPtsXY.
    //! \return     NVWARP_SUCCESS,             if the conversion was successful,
    //! \return     NVWARP_ERR_DOMAIN           if it fails due to any coordinate being out of domain.
    //! \return     NVWARP_ERR_UNIMPLEMENTED    if it is an unsupported warp type.
    nvwarpResult warp(unsigned numPts, const float *inPtsXY, float *outPtsXY) const;

    //! Transform coordinates from the destination space to the source space.
    //! This works in-place.
    //! \param[in]  numPts      the number of points to be transformed.
    //! \param[in]  inPtsXY     an array of 2D points to be transformed.
    //! \param[out] outPtsXY    a 2D point array of where the transformed 2D points are to be placed.
    //!                         This can be the same as inPtsXY.
    //! \return     NVWARP_SUCCESS,             if the conversion was successful,
    //! \return     NVWARP_ERR_DOMAIN           if it fails due to any coordinate being out of domain.
    //! \return     NVWARP_ERR_UNIMPLEMENTED    if it is an unsupported warp type.
    nvwarpResult inverseWarp(unsigned numPts, const float *inPtsXY, float *outPtsXY) const;


    //! Get the version number, encoded as (major_version * 16777216u + minor * 65536u + revision * 256u + developer_build).
    //! For example,
    //!    version 1.0.0   is represented as 0x01000000 = 16777216,
    //!    version 0.9.0   is represented as 0x00090000 = 589824,
    //!    version 0.9.0d1 is represented as 0x00090001 = 589825,
    //! Typically, the major version is incremented when the API, other major changes, or backwards incompatibilities occur,
    //!            the minor version is incremented when minor functionality changes such as new projections are added,
    //!            the   revision    is incremented when a bug fix has been released.
    //! and the patch is incremented for particular bug fixes.
    //! Versions with nonzero developer builds are never released to the public, and not supported; these are primarily used
    //! for testing internally or perhaps with third parties with whom we have a close relationship. The rest of the version is set appropriate
    //! for the target release, e.g. in preparation for version 1.0.0, we may have 1.0.0d1, 1.0.0d2, and eventually 1.0.0 when it is releasable.
    //! At most 256 developer builds can occur before it is necessary to increment the    revision   number,
    //! at most 256    revisions     can occur before it is necessary to increment the minor version number, and
    //! at most 256  minor versions  can occur before it is necessary to increment the major version number.
    //! \return the version number.
    unsigned version() const;


    ////////////////////////////////////////////////////////////////////////////////
    //                              Advanced API                                  //
    ////////////////////////////////////////////////////////////////////////////////

    //! Set the warp type.
    //! \param[in]   type    the warp type.
    void setWarpType(Warp360Params::WarpType type);

    //! Set the pixel phase.
    //! \param[in]  phase   0 = pixels are sampled on the integers, with valid pixel coordinates [0, width-1].
    //!                     1 = pixels are sampled on the integers-plus-one-half, with valid pixel coordinates [0.5, width-0.5].
    //!                     Any nonzero value has the same effect as 1.
    void setPixelPhase(int phase);

    //! Set the source focal length. Typically, the same focal length is used for X and Y, but these can be different,
    //! if a second focal length is supplied. Focal length is a measure of the angular pixel density at the principal
    //! point, so the effect of different focal lengths is anisotropic sampling, or rectangular rather than square pixels.
    //! \param[in]  fl  the X focal length.
    //! \param[in]  fy  the X focal length; if 0, the X focal length is used for Y as well.
    //! \note       negative focal lengths have the effect of mirroring, but are not recommended.
    void setSrcFocalLength(float fl, float fy = 0.f);

    //! Set the source dimensions.
    //! \param[in]   w   the source width,  in pixels.
    //! \param[in]   h   the source height, in pixels.
    void setSrcWidthHeight(unsigned w, unsigned h);

    //! Set the source fisheye clipping radius.
    //! \param[in] r the fisheye clipping radius.
    //! \note      this is not [yet] used for circular clipping. No clipping is specified by r<=0.
    void setSrcRadius(float r);

    //! Specify the rotation. By suitable construction, this can be used either as a projection (viewing) matrix,
    //! or an embedding (placement) matrix. The rotations are specified in a coordinate system where Y is down and
    //! Z is out. You can use the function convertTransformBetweenYUpandYDown() to convert representations to a
    //! coordinate system where Y is up and Z is in. The same function is used in either direction of conversion,
    //! and is used both for a projection or embedding matrix.
    //! \param[in]   R   the rotation transformation.
    void setRotation(const float R[9]);

    //! Specify the rotation, using a list of angles and their respective axes of rotation.
    //! \param[in]   angles  a list of angles, typically 3 for traditional Euler angles, but can be any size greater than 0.
    //! \param[in]   axes    the list of axes of rotation: upper-case 'X', 'Y', and 'Z' for the positive X-, Y- and Z-axes,
    //!                      and lower-case 'x', 'y', and 'z' for the negative X-, Y-, and Z- axes. The same axis may appear
    //!                      more than once, e.g. "ZXZ". This specification is for a coordinate system where the Y-axis is down
    //!                      and the Z-axis is out. For a coordinate system where Y goes up and Z comes in, invert the case of all
    //!                      'Y' and 'Z' axis specifications. This string is 0-terminated, like any C-string; the length
    //!                      of the string determines how many rotations are concatenated.
    void setEulerRotation(const float *angles, const char *axes);

    //! Specify the principal point of the source image.
    //! \note        the Y axis is always considered to point downward.
    //! \param[in]   xy          the principal point.
    //! \param[in]   relToCenter 0 = the principal point is specified relative to the upper left corner of the image;
    //!                          1 = the principal point is specified relative to the center of the image.
    void setSrcPrincipalPoint(const float xy[2], bool relToCenter);

    //! Set the distortion coefficients.
    //! \param[in]   d   the list of distortion coefficients. Though only 4 are used at the moment, 5 are anticipated to be
    //!                  used to accommodate tangential distortion in the Brown perspective distortion model. For future
    //!                  compatibility, set d[3] = d[4] = 0.
    void setDistortion(const float d[5]);

    //! Set the destination width and height.
    //! \param[in]   w   the desired destination width.
    //! \param[in]   h   the desired destination height.
    void setDstWidthHeight(unsigned w, unsigned h);

    //! Set the destination principal point.
    //! \note        the Y axis is always considered to point downward.
    //! \param[in]   xy  the principal point.
    //! \param[in]   relToCenter 0 = the principal point is specified relative to the upper left corner of the image;
    //!                          1 = the principal point is specified relative to the center of the image.
    void setDstPrincipalPoint(const float xy[2], bool relToCenter);

    //! Set the destination focal length. Typically, the same focal length is used for X and Y, but these can be different,
    //! if a second focal length is supplied. Focal length is a measure of the angular pixel density (in pixels/radian)
    //! at the principal point, so the effect of different focal lengths is anisotropic sampling, or rectangular rather than
    //! square pixels. To keep the same magnification, set the destination focal length equal to the source focal length.
    //! To zoom in by a factor of 2, double the focal length. Be careful when decreasing the focal length to avoid aliasing;
    //! you are probably safe down to a factor of 1/2, but you are most certainly going to manifest aliasing when going
    //! smaller than a factor of 1/3.
    //! \param[in]   fl  the desired X focal length.
    //! \param[in]   fy  the desired Y focal length. If not specified, it is set to be identical to the X focal length.
    //! \note       negative focal lengths have the effect of mirroring, but are not recommended.
    void setDstFocalLength(float fl, float fy = 0.f);

    //! From the vertical view angles, this sets the identical destination focal lengths, plus width and height.
    //! The view angles are measured in the center of a symmetric view.
    //! The warp type must already be chosen before calling.
    //! \param[in]   topAngle        the top view angle.
    //! \param[in]   bottomAngle     the bottom view angle.
    //! \param[in]   dstWidth        the width of the destination.
    //! \param[in]   dstHeight       the height of the destination.
    void setDstFocalLength(float topAngle, float bottomAngle, unsigned dstWidth, unsigned dstHeight);

    //! Set a control parameter. Most warps do not have one. At the current time,
    //! no warps have more than 1 control parameter.
    //! \param[in]   index       the index of the control to be set.
    //! \param[in]   control     the desired control value.
    void setControl(unsigned index, float control);

    //! Set the user data pointer.
    //! \param[in]   userData    pointer to the user data.
    void setUserData(void *userData);

    //! Get the type of the warp.
    //! \return  the type of the warp.
    Warp360Params::WarpType warpType() const;

    //! Get the pixel phase.
    //! \return 0 if pixels are sampled on the integers,
    //!         1 if pixels are sampled on the integers-plus-one-half.
    int pixelPhase() const;

    //! Get the source focal length, or the X focal length if two were specified.
    //! \return  the source focal length.
    float srcFocalLength() const;

    //! Get both source focal lengths.
    //! \param[out]  fl  an array to place the source's X focal length in fl[0] and the Y focal length in fl[1].
    void getSrcFocalLengths(float fl[2]) const;

    //! Get the rotation matrix. This is one in which the Y-axis is directed downward, and the Z axis out.
    void getRotation(float R[9]) const;

    //! Get the source principal point.
    //! \param[out]  xy  a place to store the source principal point.
    //! \param[in]   relToCenter 0 = relative to the upper left corner of the image,
    //!                          1 = relative to the center of the image.
    void getSrcPrincipalPoint(float xy[2], bool relToCenter) const;

    //! Get the source fisheye clipping radius.
    //! \return  the fisheye clipping radius.
    float srcRadius() const;

    //! Get the source image dimensions.
    //! \param[out]  wh  a place to store the source width (wh[0]) and height (wh[1]).
    void getSrcWidthHeight(unsigned wh[2]) const;

    //! Get the distortion coefficients. Note: an array of 5 must be supplied, even though only 4 are currently used.
    //! The fifth is reserved to implement the full Brown model for perspective images.
    //! \param[out]  d   the array where the distortion coefficients are to be stored.
    void getDistortion(float d[5]) const;

    //! Get the destination width and height.
    //! \param[out]  wh  an array in which to store the width and height.
    void getDstWidthHeight(unsigned wh[2]) const;

    //! Get the destination principal point.
    //! \param[out]  xy  a place to store the destination principal point.
    //! \param[in]   relToCenter 0 = relative to the upper left corner of the image,
    //!                          1 = relative to the center of the image.
    void getDstPrincipalPoint(float xy[2], bool relToCenter) const;

    //! Get the destination focal length, or the X focal length if two were specified.
    //! \return the destination focal length.
    float dstFocalLength() const;

    //! Get both source focal lengths.
    //! \param[in]   fl  a place to store the destination focal length for X in fl[0] and Y in fl[1].
    void getDstFocalLengths(float fl[2]) const;

    //! Get the value for the control parameters.
    //! \param[in]   index       the index of the control parameter to retrieve.
    //! \return      the value of the specified control parameter, or NaN if there is no such parameter (i.e. index > 0 at the moment).
    //! \note        Only control[0] is implemented, but only for a few warps.
    float getControl(unsigned index) const;


    //! Convert source coordinates into normalized rays.
    //! \param[in]  numPts  the number of points to be converted into rays.
    //! \param[in]  pts2D   the array of 2D points to be transformed into 3D rays.
    //! \param[out] rays3D  the array into which the 3D rays are to be placed.
    //! \return     NVWARP_SUCCESS,             if the conversion was successful,
    //! \return     NVWARP_ERR_DOMAIN           if it fails due to any coordinate being out of domain.
    //! \return     NVWARP_ERR_UNIMPLEMENTED    if it is an unsupported warp type.
    //! \note    The srcToRay(), dstToRay(), srcFromRay() and dstFromRay() functions can be used to warp coordinates as follows:
    //! \code
    //! float srcPt[2], dstPt[2], srcRay[3], dstRay[3], M[9];
    //! ...
    //! warper.getRotation(M);
    //! warper.srcToRay(1, srcPt, srcRay); // Warp
    //! dstRay[0] = M[0] * srcRay[0] + M[1] * srcRay[1] + M[2] * srcRay[2];
    //! dstRay[1] = M[3] * srcRay[0] + M[4] * srcRay[1] + M[5] * srcRay[2];
    //! dstRay[2] = M[6] * srcRay[0] + M[7] * srcRay[1] + M[8] * srcRay[2];
    //! warper.dstFromRay(1, dstRay, dstPt);
    //! ...
    //! warper.dstToRay(1, dstPt, dstRay); // Inverse Warp
    //! srcRay[0] = M[0] * dstRay[0] + M[3] * dstRay[1] + M[6] * dstRay[2];
    //! srcRay[1] = M[1] * dstRay[0] + M[4] * dstRay[1] + M[7] * dstRay[2];
    //! srcRay[2] = M[2] * dstRay[0] + M[5] * dstRay[1] + M[8] * dstRay[2];
    //! warper.srcFromRay(1, srcRay, srcPt);
    //! \endcode
    nvwarpResult srcToRay(unsigned numPts, const float *pts2D, float *rays3D) const;

    //! Convert destination coordinates into normalized rays.
    //! \param[in]   numPts  the number of points to be converted into rays.
    //! \param[in]   pts2D   the array of 2D points to be transformed into 3D rays.
    //! \param[out]  rays3D  the array into which the 3D rays are to be placed.
    //! \return     NVWARP_SUCCESS,             if the conversion was successful,
    //! \return     NVWARP_ERR_DOMAIN           if it fails due to any coordinate being out of domain.
    //! \return     NVWARP_ERR_UNIMPLEMENTED    if it is an unsupported warp type.
    nvwarpResult dstToRay(unsigned numPts, const float *pts2D, float *rays3D) const;

    //! Convert rays into source coordinates. The rays do not need to be normalized.
    //! \param[in]   numRays the number of rays to be converted into points.
    //! \param[in]   rays3D  the array of 3D rays to be transformed into 2D points.
    //! \param[out]  pts2D   the array into which the 2D points are to be placed.
    //! \return     NVWARP_SUCCESS,             if the conversion was successful,
    //! \return     NVWARP_ERR_DOMAIN           if it fails due to any coordinate being out of domain.
    //! \return     NVWARP_ERR_UNIMPLEMENTED    if it is an unsupported warp type.
    nvwarpResult srcFromRay(unsigned numRays, const float *rays3D, float *pts2D) const;

    //! Convert rays into destination coordinates. The rays do not need to be normalized.
    //! \param[in]   numRays the number of rays to be converted into points.
    //! \param[in]   rays3D  the array of 3D rays to be transformed into 2D points.
    //! \param[out]  pts2D   the array into which the 2D points are to be placed.
    //! \return     NVWARP_SUCCESS,             if the conversion was successful,
    //! \return     NVWARP_ERR_DOMAIN           if it fails due to any coordinate being out of domain.
    //! \return     NVWARP_ERR_UNIMPLEMENTED    if it is an unsupported warp type.
    nvwarpResult dstFromRay(unsigned numRays, const float *rays3D, float *pts2D) const;

    //! Invert the sense of the Y- and Z-axes in the specified transformation,
    //! keeping the X axis pointing in the same direction, i.e.
    //! converting between {X-right, Y-up, Z-in} and {X-right, Y-down, Z-out}.
    //! \param[in]   fr  the initial transformation to be converted.
    //! \param[out]  to  a place to store the converted transformation (can the same as fr).
    //! \return      the "to" transformation, to allow its use in-line.
    static float* convertTransformBetweenYUpandYDown(const float fr[9], float to[9]);


protected:

    struct nvwarpObject *_impl;   //!< private implementation.
};


inline unsigned Warp360::version() const                            { return nvwarpVersion(); }
inline void     Warp360::setBlock(dim3 dim_block)                   { nvwarpSetBlock(_impl, dim_block); }
inline void     Warp360::getBlock(dim3 *dim_block)            const { nvwarpGetBlock(_impl, dim_block); }
inline void     Warp360::setWarpType(Warp360Params::WarpType type)  { nvwarpSetWarpType(_impl, (nvwarpType_t)type); }
inline Warp360Params::WarpType Warp360::warpType()            const { return (Warp360Params::WarpType)nvwarpGetWarpType(_impl); }
inline void     Warp360::setPixelPhase(int phase)                   { nvwarpSetPixelPhase(_impl, (uint32_t)phase); }
inline int      Warp360::pixelPhase()                         const { return (int)nvwarpGetPixelPhase(_impl); }
inline void     Warp360::setSrcFocalLength(float fl, float fy)      { nvwarpSetSrcFocalLengths(_impl, fl, fy); }
inline float    Warp360::srcFocalLength()                     const { return nvwarpGetSrcFocalLength(_impl, nullptr); }
inline void     Warp360::getSrcFocalLengths(float fl[2])      const { fl[0] = nvwarpGetSrcFocalLength(_impl, &fl[1]); }
inline void     Warp360::setSrcWidthHeight(unsigned w, unsigned h)  { nvwarpSetSrcWidthHeight(_impl, w, h); }
inline void     Warp360::getSrcWidthHeight(unsigned wh[2])    const { nvwarpGetSrcWidthHeight(_impl, wh); }
inline void     Warp360::setSrcRadius(float r)                      { nvwarpSetSrcRadius(_impl, r); }
inline float    Warp360::srcRadius()                          const { return nvwarpGetSrcRadius(_impl); }
inline void     Warp360::setRotation(const float R[9])              { nvwarpSetRotation(_impl, R); }
inline void     Warp360::getRotation(float R[9])              const { nvwarpGetRotation(_impl, R); }
inline void     Warp360::setEulerRotation(const float *angles, const char *axes)
                                                                    { nvwarpSetEulerRotation(_impl, angles, axes); }
inline void     Warp360::setSrcPrincipalPoint(const float xy[2], bool relToCenter)
                                                                    { nvwarpSetSrcPrincipalPoint(_impl, xy, (uint32_t)relToCenter); }
inline void     Warp360::getSrcPrincipalPoint(float xy[2], bool relToCenter)
                                                              const { nvwarpGetSrcPrincipalPoint(_impl, xy, (uint32_t)relToCenter); }
inline void     Warp360::setDistortion(const float d[5])            { nvwarpSetDistortion(_impl, d); }
inline void     Warp360::getDistortion(float d[5])            const { nvwarpGetDistortion(_impl, d); }
inline void     Warp360::setDstWidthHeight(unsigned w, unsigned h)  { nvwarpSetDstWidthHeight(_impl, w, h); }
inline void     Warp360::getDstWidthHeight(unsigned wh[2])    const { nvwarpGetDstWidthHeight(_impl, wh); }
inline void     Warp360::setDstPrincipalPoint(const float xy[2], bool relToCenter)
                                                                    { nvwarpSetDstPrincipalPoint(_impl, xy, (uint32_t)relToCenter); }
inline void     Warp360::getDstPrincipalPoint(float xy[2], bool relToCenter)
                                                              const { nvwarpGetDstPrincipalPoint(_impl, xy, (uint32_t)relToCenter); }
inline void     Warp360::setDstFocalLength(float fl, float fy)      { nvwarpSetDstFocalLengths(_impl, fl, fy); }
inline void     Warp360::setDstFocalLength(float topAngle, float bottomAngle, unsigned dstWidth, unsigned dstHeight)
                                                                    { nvwarpComputeDstFocalLength(_impl, topAngle, bottomAngle, dstWidth, dstHeight); }
inline float    Warp360::dstFocalLength()                     const { return nvwarpGetDstFocalLength(_impl, nullptr); }
inline void     Warp360::getDstFocalLengths(float fl[2])      const { fl[0] = nvwarpGetDstFocalLength(_impl, &fl[1]); }
inline void     Warp360::setControl(unsigned index, float control)  { nvwarpSetControl(_impl, index, control); }
inline float    Warp360::getControl(unsigned index)           const { return nvwarpGetControl(_impl, index); }
inline void     Warp360::setUserData(void *userData)                { nvwarpSetUserData(_impl, userData); }
inline void*    Warp360::getUserData() const                        { return nvwarpGetUserData(_impl); }
inline cudaError_t Warp360::warp(cudaStream_t stream, cudaTextureObject_t srcTex, cudaSurfaceObject_t dstSurface) const
                                                                    { return (cudaError_t)nvwarpWarpSurface(_impl, stream, srcTex, dstSurface); }
inline cudaError_t Warp360::warp(cudaStream_t stream, cudaTextureObject_t srcTex, void *dstAddr, size_t dstRowBytes) const
                                                                    { return (cudaError_t)nvwarpWarpBuffer(_impl, stream, srcTex, dstAddr, dstRowBytes); }
inline nvwarpResult Warp360::warp(unsigned numPts, const float *inPtsXY, float *outPtsXY) const
                                                                    { return nvwarpWarpCoordinates(_impl, numPts, inPtsXY, outPtsXY); }
inline nvwarpResult Warp360::inverseWarp(unsigned numPts, const float *inPtsXY, float *outPtsXY) const
                                                                    { return nvwarpInverseWarpCoordinates(_impl, numPts, inPtsXY, outPtsXY); }
inline nvwarpResult Warp360::srcToRay(unsigned numPts, const float *pts2D, float *rays3D)
                                                              const { return nvwarpSrcToRay(_impl, numPts, pts2D, rays3D); }
inline nvwarpResult Warp360::dstToRay(unsigned numPts, const float *pts2D, float *rays3D)
                                                              const { return nvwarpDstToRay(_impl, numPts, pts2D, rays3D); }
inline nvwarpResult Warp360::srcFromRay(unsigned numRays, const float *rays3D, float *pts2D)
                                                              const { return nvwarpSrcFromRay(_impl, numRays, rays3D, pts2D); }
inline nvwarpResult Warp360::dstFromRay(unsigned numRays, const float *rays3D, float *pts2D)
                                                              const { return nvwarpDstFromRay(_impl, numRays, rays3D, pts2D); }


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////                Conversion from YUV 4:2:0 NV12 to RGBA                  ////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////



//! Parameters for YUV:420:NV12 --> RGBA conversion.
struct YUVRGBParams
{
    unsigned width;     //!< The width of the Y and RGB channels (chroma has half the width, but is interleaved in NV12).
    unsigned height;    //!< The height of the Y and RGB channels (chroma has half the height, so this must be even).
    unsigned cLocation; //!< 0 for chroma sampled cosited horizontally with luma; 1 for chroma sampled halfway between luma samples horizontally. Get the the video header.
    float   ry,         //!< Coefficients for R with respect to Y,  including normalization scaling.
            rcb,        //!< Coefficients for R with respect to Cb, including normalization scaling.
            rcr,        //!< Coefficients for R with respect to Cr, including normalization scaling.
            gy,         //!< Coefficients for G with respect to Y,  including normalization scaling.
            gcb,        //!< Coefficients for G with respect to Cb, including normalization scaling.
            gcr,        //!< Coefficients for G with respect to Cr, including normalization scaling.
            by,         //!< Coefficients for B with respect to Y,  including normalization scaling.
            bcb,        //!< Coefficients for B with respect to Cb, including normalization scaling.
            bcr;        //!< Coefficients for B with respect to Cr, including normalization scaling.
    float   yOffset,    //!< Offset of luma,   typically 16.
            cOffset;    //!< Offset of chroma, typically 128.

    //! Method to compute all values except for {width, height, cLocation}.
    //! \param[in]   matrix_coefficients     One of {709, 1, 2} for BT.709, {601, 5, 6} for BT.601, {2020, 9, 10} for BT.2020, {4, 4044, 0xFCC} for FCC, {240, 7} FOR SMPTE 240M.
    //! \param[in]   video_full_range_flag   0 for standard video range (16-240), 1 for full range (0-255).
    void computeMatrix(int matrix_coefficients, int video_full_range_flag);
};


//! General method to compute all values except for {width, height, cLocation}.
//! \param[in]   matrix_coefficients     One of {709, 1, 2} for BT.709, {601, 5, 6} for BT.601, {2020, 9, 10} for BT.2020, {4} for FCC, {240, 7} FOR SMPTE 240M.
//! \param[in]   video_full_range_flag   0 for standard video range (16-240), 1 for full range (0-255).
//! \param[in]   bit_depth               8 is the only depth that is supported by ConvertYUVNV12ToRGBA().
//! \param[in]   normalized_input        1 if the input  is normalized, 0 if not.
//! \param[in]   normalized_output       1 if the output is normalized, 0 if not.
//! \param[out]  p                       the location of the parameters that will be set by this method.
void ComputeYCbCr2RgbMatrix(int matrix_coefficients, int video_full_range_flag, int bit_depth, bool normalized_input, bool normalized_output, YUVRGBParams *p);


//! Perform YUV 4:2:0 NV12 --> RGBA conversion.
//! \param[in]   stream      The stream on which the computation is to be performed.
//! \param[in]   params      The parameters controlling the YUV-->RGB conversion; typically set with YUVRGBParams.computeMatrix().
//! \param[in]   yuv         Pointer to the YV CUDA pitched memory buffer.
//! \param[in]   yuvRowBytes Byte stride between pixels vertically in the luminance (and chrominance) of YUV.
//! \param[in]   dst         Pointer to the RGBA CUDA pitched memory buffer.
//! \param[in]   dstRowBytes Byte stride between pixels vertically in the RGBA.
//! \return      cudaSuccess if successful.
cudaError_t ConvertYUVNV12ToRGBA(cudaStream_t stream, const YUVRGBParams *params, const void *yuv, size_t yuvRowBytes, void *dst, size_t dstRowBytes);



////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////                                                                        ////
////                               MultiWarp                                ////
////                                                                        ////
////            The implementation is distributed as sample code,           ////
////            to optimize better for specific applications.               ////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////



//! Perform an optional YUV:420:NV12 --> RGBA conversion, followed by a suite of warps from that conversion.
//! This uses an array of parameter blocks.
//! \param[in]   stream      Stream on which to perform this computation.
//! \param[in]   yuvParams   Parameters for the YUV-->RGB conversion (can be NULL).
//! \param[in]   yuvBuffer   The buffer containing the YUV data in 420 NV12 format.
//! \param[in]   yuvRowBytes The byte stride between luminance (and chroma) pixels vertically.
//! \param[in]   rgbBuffer   The buffer where the YUV-->RGB conversion is to be placed.
//! \param[in]   rgbRowBytes The byte stride between RGBA pixels vertically.
//! \param[in]   rgbTex      The texture associated with the RGB buffer, initialized appropriately.
//! \param[in]   numWarps    The number of warps to be executed on using the RGB buffer texture as a source.
//! \param[in]   paramArray  The array of parameter blocks for each warp.
//! \param[out]  dstBuffers  The array of pointers to the destination buffers.
//! \param[in]   dstRowBytes The array of byte strides between pixels vertically, one for each warp.
//! \return      cudaSuccess if successful.
cudaError_t MultiWarp360(cudaStream_t stream,
    const YUVRGBParams *yuvParams, const void *yuvBuffer, size_t yuvRowBytes,
    void *rgbBuffer, size_t rgbRowBytes, cudaTextureObject_t rgbTex,
    unsigned numWarps, const Warp360Params *paramArray, void **dstBuffers, const size_t *dstRowBytes);




#ifdef MULTIWARP_IMPLEMENTATION
/********************************************************************************
 * MultiWarp360 - with array of parameter blocks.
 ********************************************************************************/

cudaError_t MultiWarp360(cudaStream_t stream,
    const YUVRGBParams *yuvParams, const void *yuvBuffer, size_t yuvRowBytes,
    void *rgbBuffer, size_t rgbRowBytes, cudaTextureObject_t rgbTex,
    unsigned numWarps, const Warp360Params *paramArray, void **dstBuffers, const size_t *dstRowBytes)
{
    cudaError_t cuErr   = cudaSuccess;
    Warp360     warper;

    if (yuvParams)
    {
        cuErr = ConvertYUVNV12ToRGBA(stream, yuvParams, yuvBuffer,  yuvRowBytes, rgbBuffer, rgbRowBytes);
        if (cudaSuccess != cuErr)
            return cuErr;
    }
    for (; numWarps--; ++paramArray, ++dstBuffers, ++dstRowBytes)
    {
        cuErr = warper.setParams(paramArray);
        if (cudaSuccess != cuErr)
            return cuErr;
        cuErr = warper.warp(stream, rgbTex, *dstBuffers, *dstRowBytes);
        if (cudaSuccess != cuErr)
            return cuErr;
    }

    return cuErr;
}
#endif /* MULTIWARP_IMPLEMENTATION */





#endif // __cplusplus
#endif // __WARP360_H__
