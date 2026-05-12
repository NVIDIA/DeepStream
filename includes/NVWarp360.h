/**
 * SPDX-FileCopyrightText: Copyright (c) 2017-18 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

//! @file NVWarp360.h
//! 360 Image and Coordinate Warp SDK.
//!

#ifndef __NVWARP360_H__
#define __NVWARP360_H__

#include <cuda_runtime.h>
#include <stdint.h>


#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(_WIN64)
    #define NVWARPAPI __stdcall //!< export tag for Windows
#else /* !_WINxx */
    #define NVWARPAPI           //!< export tag
#endif /* _WINxx */

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                              NVWarp360                               /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

//! Error code enumerations for NVWarp360
typedef enum nvwarpResult
{
    NVWARP_SUCCESS                  = 0,            //!< The operation was successful.
    NVWARP_ERR_GENERAL              = 1,            //!< An otherwise unspecified failure has occurred.
    NVWARP_ERR_UNIMPLEMENTED        = 2,            //!< The requested feature has not yet been implemented.
    NVWARP_ERR_DOMAIN               = 3,            //!< The coordinates are outside of the domain.
    NVWARP_ERR_MISSING_PARAMETERS   = 4,            //!< Some required parameters have not been specified.
    NVWARP_ERR_PARAMETER            = 5,            //!< A parameter has an invalid value.
    NVWARP_ERR_INITIALIZATION       = 6,            //!< Initialization has not completed successfully.
    NVWARP_ERR_CUDA_MEMORY          = 7,            //!< There is not enough CUDA memory for the operation specified.
    NVWARP_ERR_CUDA_LAUNCH          = 8,            //!< CUDA was not able to launch the specified kernel.
    NVWARP_ERR_CUDA_DRIVER          = 9,            //!< CUDA driver version is insufficient for CUDA runtime version.
    NVWARP_ERR_CUDA_NO_KERNEL       = 10,           //!< No suitable CUDA kernel image has been found for this GPU.
    NVWARP_ERR_CUDA                 = 11,           //!< An otherwise unspecified error has been reported by the CUDA runtime.
    NVWARP_ERR_APPROXIMATE          = 12,           //!< An accurate calculation is not available; approximation returned.
    NVWARP_ERR_MAX                  = 0x7fffffff    //!< This makes it 32 bits.
} nvWarpResult; //!< Error code typedef for NVWarp360.

// These are not used directly by the API.
typedef enum nvwarpSurface_t
{
    NVSURF_PERSPECTIVE      = 0,                        //!< Perspective, projective, rectilinear.
    NVSURF_FISHEYE          = 1,                        //!< Fisheye, equidistant.
    NVSURF_EQUIRECT         = 2,                        //!< Equirectangular spherical.
    NVSURF_CYLINDER         = 3,                        //!< Cylindrical, panned horizontally with vertical axis.
    NVSURF_ROTCYLINDER      = 4,                        //!< Cylindrical, panned vertically, with horizontal axis.
    NVSURF_PUSHBROOM        = 5,                        //!< Simulated pushbroom. Control[0] typically in (0, 1].
    NVSURF_PANINI           = 6,                        //!< Panini. Control[0] typically in (0, 1].
    NVSURF_STEREOGRAPHIC    = 7,                        //!< Generalized Stereographic. Normal stereographic has control[0] = 1.

    NVSURF_BITS             = 8,                        //!< The width of the field used for each image type.
    NVSURF_UNKNOWN          = ((1 << NVSURF_BITS) - 1), //!< Unknown        (currently 255).
    NVSURF_MASK             = ((1 << NVSURF_BITS) - 1), //!< The NVWARP_360 mask (currently 0xFF).
    NVSURF_MAX              = 0x7fffffff                //!< This makes it 32 bits.
} nvwarpSurface_t;

//! Warp type enumeration.
typedef enum nvwarpType_t
{
    NVWARP_EQUIRECT_CYLINDER        = (NVSURF_EQUIRECT    << NVSURF_BITS) | NVSURF_CYLINDER,        //!< Equirectangular to cylindrical.
    NVWARP_EQUIRECT_EQUIRECT        = (NVSURF_EQUIRECT    << NVSURF_BITS) | NVSURF_EQUIRECT,        //!< Equirectangular to equirectangular.
    NVWARP_EQUIRECT_FISHEYE         = (NVSURF_EQUIRECT    << NVSURF_BITS) | NVSURF_FISHEYE,         //!< Equirectangular to fisheye.
    NVWARP_EQUIRECT_PANINI          = (NVSURF_EQUIRECT    << NVSURF_BITS) | NVSURF_PANINI,          //!< Equirectangular to Panini.
    NVWARP_EQUIRECT_PERSPECTIVE     = (NVSURF_EQUIRECT    << NVSURF_BITS) | NVSURF_PERSPECTIVE,     //!< Equirectangular to perspective.
    NVWARP_EQUIRECT_PUSHBROOM       = (NVSURF_EQUIRECT    << NVSURF_BITS) | NVSURF_PUSHBROOM,       //!< Equirectangular to pushbroom.
    NVWARP_EQUIRECT_STEREOGRAPHIC   = (NVSURF_EQUIRECT    << NVSURF_BITS) | NVSURF_STEREOGRAPHIC,   //!< Equirectangular to generalized stereographic.
    NVWARP_EQUIRECT_ROTCYLINDER     = (NVSURF_EQUIRECT    << NVSURF_BITS) | NVSURF_ROTCYLINDER,     //!< Equirectangular to vertical cylindrical.
    NVWARP_FISHEYE_CYLINDER         = (NVSURF_FISHEYE     << NVSURF_BITS) | NVSURF_CYLINDER,        //!< Fisheye to horizontally panned cylinder.
    NVWARP_FISHEYE_EQUIRECT         = (NVSURF_FISHEYE     << NVSURF_BITS) | NVSURF_EQUIRECT,        //!< Fisheye to equirectangular.
    NVWARP_FISHEYE_FISHEYE          = (NVSURF_FISHEYE     << NVSURF_BITS) | NVSURF_FISHEYE,         //!< Fisheye to fisheye.
    NVWARP_FISHEYE_PANINI           = (NVSURF_FISHEYE     << NVSURF_BITS) | NVSURF_PANINI,          //!< Fisheye to Panini.
    NVWARP_FISHEYE_PERSPECTIVE      = (NVSURF_FISHEYE     << NVSURF_BITS) | NVSURF_PERSPECTIVE,     //!< Fisheye to perspective.
    NVWARP_FISHEYE_PUSHBROOM        = (NVSURF_FISHEYE     << NVSURF_BITS) | NVSURF_PUSHBROOM,       //!< Fisheye to pushbroom.
    NVWARP_FISHEYE_ROTCYLINDER      = (NVSURF_FISHEYE     << NVSURF_BITS) | NVSURF_ROTCYLINDER,     //!< Fisheye to vertically panned radial cylinder.
    NVWARP_PERSPECTIVE_EQUIRECT     = (NVSURF_PERSPECTIVE << NVSURF_BITS) | NVSURF_EQUIRECT,        //!< Perspective to equirectangular.
    NVWARP_PERSPECTIVE_PANINI       = (NVSURF_PERSPECTIVE << NVSURF_BITS) | NVSURF_PANINI,          //!< Perspective to Panini.
    NVWARP_PERSPECTIVE_PERSPECTIVE  = (NVSURF_PERSPECTIVE << NVSURF_BITS) | NVSURF_PERSPECTIVE,     //!< Perspective to perspective.
    NVWARP_NONE                     = 0xffffffff                                                    //!< This makes it 32 bits.
} nvwarpType_t; //!< Warp type typedef.




////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////                              Parameter Block                           ////
////                                                                        ////
//// One way to set parameters for a warp is via a parameter block,         ////
//// described here. The other way uses individual setters and getters,     ////
//// described afterward. The parameter block is useful and convenient for  ////
//// most cases, but the setters/getters give more control and access to    ////
//// advanced features. You can mix and match, e.g. start setting with the  ////
//// parameter block and tweak with a specific setter. The rotation setters ////
//// are particularly useful.                                               ////
////                                                                        ////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


//! Pass parameters via this struct.
//! Many parameters are common. Some may not be used.
struct nvwarpParams_t;
typedef struct nvwarpParams_t nvwarpParams_t;


//! Initialize nvwarpParams_t to defaults or NaN if a particular parameter must be supplied.
//! \param[out]     params  the parameter block to initialize.
void NVWARPAPI nvwarpInitParams(nvwarpParams_t *params);


//! Parameter structure.
typedef struct nvwarpParams_t
{
    // Warp type selector
    nvwarpType_t    type;                       //!< The type of the warp.

    // Source specification
    uint32_t        srcWidth;                   //!< The width  of the source image.
    uint32_t        srcHeight;                  //!< The height of the source image.
    float           srcX0;                      //!< Source center of projection X; frequently (srcWidth  - 1) * 0.5, but srcWidth  * .5 for wraparounds (equirect).
    float           srcY0;                      //!< Source center of projection Y; frequently (srcHeight - 1) * 0.5, but srcHeight * .5 for wraparounds (equirect).
    float           srcFocalLen;                //!< Source focal length.
    float           srcRadius;                  //!< Source circular clipping radius. (default 0 means no clipping) (unimplemented).

    float           dist[5];                    //!< Distortion, typically for the source. (default {0,0,0,0} means no distortion).

    // Destination specification
    uint32_t        dstWidth;                   //!< The width  of the destination.
    uint32_t        dstHeight;                  //!< The height of the destination.

    // View specification
    float           rotAngles[3];               //!< Rotation angles in radians, corresponding to the list of rotation axes below..
    char            rotAxes[4];                 //!< 3 rotation axes: upper case 'X', 'Y', and 'Z' are rotation about the positive X, Y and Z axes,
                                                //!< whereas lower case 'x', 'y', and 'z' are rotation about the negative axes. Set the 4th to '\0'.
                                                //!< X rotation rotates the view upward, Y rightward, and Z clockwise. To specify an embedding
                                                //!< of the image in 3D rather than a view of a straightahead image, specify the angles in the opposite order.
                                                //!< The default is "YXZ", a.k.a. yaw, pitch, roll (as set by nvwarpInitParams()).
                                                //!< Other characters are treated as 'Z'.

    float           topAngle;                   //!< Top    angle of view. (default +pi/2)
    float           bottomAngle;                //!< Bottom angle of view. (default -pi/2)

    void*           userData;                   //!< Pointer supplied by the user. (default NULL)

    // Warp controls
    float           control[4];                 //!< Projection-specific controls.

#ifdef __cplusplus
    //! Constructor for C++ automatically initializes nvwarpParams_t.
    //! C users are recommended to call nvwarpInitParams() explicitly, for consistent initialization.
    nvwarpParams_t() { nvwarpInitParams(this); }
#endif // __cplusplus

} nvwarpParams_t;   //!< Parameters typedef.


//! Compute and set the srcFocalLen given the warp type and srcDist distortion coefficients, plus an angle and corresponding radius.
//! The srcDist[] coefficients must be initialized prior to the call (suggest {0,0,0,0} for ideal lens).
//! The focal length converts from angles to pixels at the center of projection,
//! and is a measure of spherical image resolution also known as angular pixel density (in pixels/radian).
//! It is only as accurate as the parameters that were supplied.
//! It can be adjusted by hand, for example to straighten out lines that appeared bowed, but distortion does that as well.
//! \param[in,out]  params  the Warp360 parameters to be set.
//! \param[in]      angle   the angle in radians.
//! \param[in]      radius  the radius, in pixels, corresponding to the angle above.
//! \return         true if the srcFocalLen was computed successfully, false otherwise.
//! \note           The focal length is one of the properties of the source, and this API provides a way to convert
//!                 angle and distortion measurements into a focal length. Once calibrated for a given lens and camera,
//!                 it is fixed, and not considered to be a warp control parameter.
//! \note           The focal length can also be acquired from the image EXIF data without the use of this API.
//!                 In particular, the focalLength tag (37386) yields the focal length in millimeters. Then the
//!                 FocalPlaneXResolution tag (41486), the FocalPlaneXResolution tag (41487), and FocalPlaneResolutionUnit tag (41488)
//!                 can be used to compute the X and Y focal lengths by converting mm to pixels. If the X and Y focal lengths differ,
//!                 it is suggested to use the geometric average focLen = sqrt(focLenX * focLenY).
nvwarpResult NVWARPAPI nvwarpComputeParamsSrcFocalLength(nvwarpParams_t *params, float angle, float radius);

//! Compute the output resolution that matches the source focal length and desired aspect ratio.
//! The dimensions are computed from the source focal length, so it must already have the appropriate value.
//! \param[in,out]  params  the Warp360 parameters.
//! \param[in]      aspectRatio the ratio of width/height for the output.
//! \return         true        if the dstWidth and dstHeight were updated successfully, false if not.
//! \note           This API provides a suggestion that should be tweaked to meet the needs of the application.
//!                 Enlarging the dimensions will produce a bigger image, but not introduce any new details.
//!                 Reducing the dimensions will not only produce a smaller image, but will lose more details as it shrinks.
//!                 It is not recommended to reduce these dimensions smaller than 1/3, or aliasing will be introduced.
nvwarpResult NVWARPAPI nvwarpComputeParamsOutputResolution(nvwarpParams_t *params, float aspectRatio);

//! Compute the angular limits on the source image along the primary axes going through the center of projection.
//! The nvwarpParams_t must have the following fields filled in:
//! type, srcWidth, srcHeight, srcX0, srcY0, srcFocalLen, srcDist[4];
//! \param[in]      params  the Warp360 parameters.
//! \param[out]     minMaxXY    an array of length 4, where the values {minX, maxX, minY, maxY} angles, in radians, are returned.
//! \return         true if successful; otherwise false.
//! \note           At the moment, only fisheye, perspective and equirectangular sources are accommodated.
//! \note           Even though this guarantees that this angular range contains valid pixels along the horizontal and vertical axes
//!                 through the center of projection, this does not guarantee that the same holds true at the corners.
nvwarpResult NVWARPAPI nvwarpComputeParamsAxialAngleRange(const nvwarpParams_t *params, float minMaxXY[4]);

//! Compute the angular limits on the source bounding box along the primary axes going through the center of projection.
//! The nvwarpParams_t must have the following fields filled in:
//! type, srcX0, srcY0, srcFocalLen, srcDist[4];
//! \param[in]      params  the Warp360 parameters.
//! \param[in]      srcBoundingBox  an array of length 4 for describing bounding box in source image {leftTopX, leftTopY, bottomRightX, bottomRightY}.
//! \param[out]     minMaxXY        an array of length 4, where the values {minX, maxX, minY, maxY} angles, in radians, are returned.
//! \return         true if successful; otherwise false.
//! \note           At the moment, only fisheye, perspective and equirectangular sources are accommodated.
//! \note           Even though this guarantees that this angular range contains valid pixels along the horizontal and vertical axes
//!                 through the center of projection, this does not guarantee that the same holds true at the corners.
nvwarpResult NVWARPAPI nvwarpComputeBoxAxialAngleRange(const nvwarpParams_t *params, const float srcBoundingBox[4], float minMaxXY[4]);



////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////                                                                        ////
////                                Warp Object                             ////
////                                                                        ////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


//! Opaque definition of the Warp360 object.
struct nvwarpObject;

//! Opaque definition of a pointer to the Warp360 state.
typedef struct nvwarpObject *nvwarpHandle;

//! Constructor for a new instance of a Warp360 object.
//! \param[out]     han a place to store the new instance of Warp360 object.
//!	\return         NVWARP_SUCCESS  if successful.
nvwarpResult NVWARPAPI nvwarpCreateInstance(nvwarpHandle *han);

//! Destructor for a Warp360 object.
//! \param[in]      han handle to a Warp360 object. Can be NULL.
void         NVWARPAPI nvwarpDestroyInstance(nvwarpHandle han);


//! Get the version number, encoded as (major_version * 16777216u + minor * 65536u + revision * 256u).
//! For example,
//!   version 1.0.3   is represented as 0x01000300 = 16777984,
//!   version 2.0.0   is represented as 0x00090000 = 536870912,
//! Typically, the major version is incremented when the API, other major changes, or backwards incompatibilities occur,
//!            the minor version is incremented when minor functionality changes such as new projections are added,
//!            the   revision    is incremented when a bug fix has been released.
//! At most 256   revisions     can occur before it is necessary to increment the minor version number, and
//! at most 256 minor versions  can occur before it is necessary to increment the major version number.
//! The LS byte is for internal use only and should be 0.
//! The macros below can be useful to extract the different parts of the version number.
//! \return         the version number.
uint32_t NVWARPAPI nvwarpVersion();

//! Helper function to extract the major part of the version returned by nvwarpVersion().
//! \param[in]      version the version, as returned by nvwarpVersion().
//! \return         the major part of the version number.
#define NVWARP_VERSION_MAJOR(version)       (((v) >> 24) & 0xff)

//! Helper function to extract the minor part of the version returned by nvwarpVersion().
//! \param[in]      version the version, as returned by nvwarpVersion().
//! \return         the minor part of the version number.
#define NVWARP_VERSION_MINOR(version)       (((v) >> 16) & 0xff)

//! Helper function to extract the revision part of the version returned by nvwarpVersion().
//! \param[in]      version the version, as returned by nvwarpVersion().
//! \return         the revision part of the version number.
#define NVWARP_VERSION_REVISION(version)    (((v) >>  8) & 0xff)

//! Get the error string corresponding to the given result code.
//! \param[in]      err     the error code.
//! \return         the C-string corresponding to the given error code.
const char* NVWARPAPI nvwarpErrorStringFromCode(nvwarpResult err);


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////                                                                        ////
////                            Setters and Getters                         ////
////                                                                        ////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


    // Set the Warp Object from the Parameter Block //

//! Set parameters for a warp.
//! \param[out]     han     the Warp360 instance.
//! \param[in]      params  pointer to the desired parameters for the warp. NULL initializes the parameters to NaN or default values as appropriate.
//!	\return         NVWARP_SUCCESS  if successful.
nvwarpResult NVWARPAPI nvwarpSetParams(nvwarpHandle han, const nvwarpParams_t *params);

//! Get the current values of the warp parameters.
//! \param[in]      han     the Warp360 instance.
//! \param[out]     params  pointer to the location where the parameters are to be stored.
//! \note           An equivalent series of rotation angles and axes may be returned instead of the one supplied;
//!                 it still results in the same rotation. In particular, "YXZ", "ZXY" and "ZXZ" are returned as given,
//!                 and all else are returned as "YXZ".
void         NVWARPAPI nvwarpGetParams(const nvwarpHandle han, nvwarpParams_t *params);


    // Warp Type //

//! Set the warp type.
//! \param[in,out]  han     the Warp360 instance.
//! \param[in]      type    the warp type.
//! \return         NVWARP_SUCCESS              if successful,
//! \return         NVWARP_ERR_UNIMPLEMENTED    if the specified warp type is unimplemented or unknown.
nvwarpResult NVWARPAPI nvwarpSetWarpType(nvwarpHandle han, nvwarpType_t type);

//! Get the type of the warp.
//! \param[in]      han the Warp360 instance.
//! \return         the type of the warp.
nvwarpType_t NVWARPAPI nvwarpGetWarpType(const nvwarpHandle han);


    // CUDA Block Size //

//! Set the CUDA block size.
//! \param[in,out]  han         the Warp360 instance.
//! \param[in]      dim_block   the desired block size (default 8x8).
void NVWARPAPI nvwarpSetBlock(nvwarpHandle han, dim3 dim_block);

//! Get the current value of the CUDA block size.
//! \param[in]      han         the Warp360 instance.
//! \param[out]     dim_block   pointer to a location where the block size is to be stored.
void NVWARPAPI nvwarpGetBlock(const nvwarpHandle han, dim3 *dim_block);


    // Pixel Phase //

//! Set the pixel phase.
//! \param[in,out]  han     the Warp360 instance.
//! \param[in]      phase   zero    = pixels are sampled on the integers, with valid pixel coordinates [0, width-1].
//!                         nonzero = pixels are sampled on the integers-plus-one-half, with valid pixel coordinates [0.5, width-0.5].
void NVWARPAPI nvwarpSetPixelPhase(nvwarpHandle han, uint32_t phase);

//! Get the pixel phase.
//! \param[in]      han the Warp360 instance.
//! \return         0 if pixels are sampled on the integers (default),
//!                 1 if pixels are sampled on the integers-plus-one-half.
uint32_t NVWARPAPI nvwarpGetPixelPhase(const nvwarpHandle han);


    // Source Dimensions //

//! Set the source dimensions.
//! \param[in,out]  han the Warp360 instance.
//! \param[in]      w   the source width,  in pixels.
//! \param[in]      h   the source height, in pixels.
void NVWARPAPI nvwarpSetSrcWidthHeight(nvwarpHandle han, uint32_t w, uint32_t h);

//! Get the source image dimensions.
//! \param[in]      han the Warp360 instance.
//! \param[out]     wh  a place to store the source width (wh[0]) and height (wh[1]).
void NVWARPAPI nvwarpGetSrcWidthHeight(const nvwarpHandle han, uint32_t wh[2]);


    // Source focal length //

//! Set the source focal length. Typically, the same focal length is used for X and Y, but these can be different,
//! if a second focal length is supplied. Focal length is a measure of the angular pixel density at the principal
//! point, so the effect of different focal lengths is anisotropic sampling, or rectangular rather than square pixels.
//! \param[in,out]  han     the Warp360 instance.
//! \param[in]      fl      the X focal length.
//! \param[in]      fy      the Y focal length; if zero, the X focal length is used for Y as well.
//! \note           Negative focal lengths will cause reflection, but are not recommended.
void NVWARPAPI nvwarpSetSrcFocalLengths(nvwarpHandle han, float fl, float fy);

//! Compute the source focal length, primarily from a radius and corresponding angle (e.g. fov/2), and the warp type.
//! The distortion also plays a part in with perspective and fisheye sources.
//! The focal length converts from angles to pixels at the center of projection,
//! and is a measure of spherical image resolution also known as angular pixel density (in pixels/radian).
//! It is only as accurate as the parameters that were supplied.
//! It can be adjusted by hand, for example to straighten out lines that appeared bowed, but distortion does that as well.
//! This will set all of these parameters:
//!     srcFocalLengthX
//!     srcFocalLengthY
//!     warp type
//!     distortion
//! \param[in,out]  han         the Warp360 instance.
//! \param[in]      warpType    the warp type.
//! \param[in]      radius      a radius, i.e. distance from the principal point, e.g. to an edge.
//! \param[in]      angle       the angle corresponding to the above radius, i.e. angular distance from the optical axis.
//! \param[in]      dist        the distortion coefficients. (NULL implies (0,0,0,0,0).
//! \note           The focal length is one of the properties of the source, and this API provides a way to convert
//!                 angle and distortion measurements into a focal length. Once calibrated for a given lens and camera,
//!                 it is fixed, and not considered to be a warp control parameter.
//! \note           The focal length can also be acquired from the image EXIF data without the use of this API.
//!                 In particular, the focalLength tag (37386) yields the focal length in millimeters. Then the
//!                 FocalPlaneXResolution tag (41486), the FocalPlaneXResolution tag (41487), and FocalPlaneResolutionUnit tag (41488)
//!                 can be used to compute the X and Y focal lengths by converting mm to pixels. If the X and Y focal lengths differ,
//!                 it is suggested to use the geometric average focLen = sqrt(focLenX * focLenY).
nvwarpResult NVWARPAPI nvwarpComputeSrcFocalLength(nvwarpHandle han, nvwarpType_t warpType, float radius, float angle, float dist[5]);

//! Get one or both source focal lengths.
//! In most cases, there will only be one focal length, so this will be called with NULL as the second parameter.
//! \param[in]      han the Warp360 instance.
//! \param[out]     fy  a place to store the Y focal length. (can be NULL if only one focal length is desired).
//! \return         the X source focal length.
float NVWARPAPI nvwarpGetSrcFocalLength(const nvwarpHandle han, float *fy);


    // Source Principal Point //

//! Specify the principal point of the source image.
//! \param[in,out]  han         the Warp360 instance.
//! \param[in]      xy          the principal point. NULL implies (0,0).
//! \param[in]      relToCenter zero    = the principal point is specified relative to the upper left corner of the image;
//!                             nonzero = the principal point is specified relative to the center of the image.
//! \note           The Y axis is always considered to point downward.
//! \note           The source width and height must be specified prior to calling nvwarpSetSrcPrincipalPoint().
//! \note           A good default is nvwarpSetDstPrincipalPoint(han, NULL, 1);
void NVWARPAPI nvwarpSetSrcPrincipalPoint(nvwarpHandle han, const float xy[2], uint32_t relToCenter);

//! Get the source principal point.
//! \param[in]      han         the Warp360 instance.
//! \param[out]     xy          a place to store the source principal point.
//! \param[in]      relToCenter zero    = relative to the upper left corner of the image,
//!                             nonzero = relative to the center of the image.
void NVWARPAPI nvwarpGetSrcPrincipalPoint(const nvwarpHandle han, float xy[2], uint32_t relToCenter);


    // Source Fisheye Radius //

//! set the source fisheye clipping radius.
//! \param[in,out]  han the Warp360 instance.
//! \param[in]      r   the fisheye clipping radius. No clipping is specified by r<=0.
//! \note           this is not [yet] used for circular clipping.
void NVWARPAPI nvwarpSetSrcRadius(nvwarpHandle han, float r);

//! Get the source fisheye clipping radius.
//! \param[in]      han the Warp360 instance.
//! \return         the fisheye clipping radius. Zero indicates no circular clipping.
float NVWARPAPI nvwarpGetSrcRadius(const nvwarpHandle han);


    // Distortion //

//! Set the distortion coefficients.
//! \param[in,out]  han the Warp360 instance.
//! \param[in]      d   the list of distortion coefficients. Though only 4 are used at the moment, 5 are anticipated to be
//!                     used to accommodate tangential distortion in the Brown perspective distortion model. For future
//!                     compatibility, set d[3] = d[4] = 0 for perspective and d[4] = 0 for fisheye.
//!                     If NULL is supplied for d, all coefficients are set to zero.
void NVWARPAPI nvwarpSetDistortion(nvwarpHandle han, const float d[5]);

//! Get the distortion coefficients. Note: an array of 5 must be supplied, even though only 4 are currently used.
//! The fifth is reserved to implement the full Brown model for perspective images.
//! \param[in]      han the Warp360 instance.
//! \param[out]     d   the array where the distortion coefficients are to be stored.
void NVWARPAPI nvwarpGetDistortion(const nvwarpHandle han, float d[5]);


    // Rotation //

//! Specify the rotation. By suitable construction, this can be used either as a projection (viewing) matrix,
//! or an embedding (placement) matrix. The rotations are specified in a coordinate system where Y is down and
//! Z is out. You can use the function convertTransformBetweenYUpandYDown() to convert representations to a
//! coordinate system where Y is up and Z is in. The same function is used in either direction of conversion,
//! and is used both for a projection or embedding matrix.
//! \param[in,out]  han the Warp360 instance.
//! \param[in]      R   the rotation transformation. This should be an orthonormal transformation, but is not enforced. NULL implies identity.
void NVWARPAPI nvwarpSetRotation(nvwarpHandle han, const float R[9]);

//! Specify the view rotation, using a list of angles and their respective axes of rotation.
//! The image is considered to be straight ahead, and the sequence of rotations are applied to the viewer (camera).
//! Alternatively one could maintain the camera facing forward, and instead rotate the image to embed it into 3D,
//! by supplying the angles and axes in reverse order.
//! Upward    is the direction of positive rotation about the rightward-pointing X axis,
//! Rightward is the direction of positive rotation about the  downward-pointing Y axis, and
//! Clockwise is the direction of positive rotation about the   outward-pointing Z axis.
//! \param[in,out]  han     the Warp360 instance.
//! \param[in]      angles  a list of angles, typically 3 for traditional Euler angles, but can be any size greater than 0.
//! \param[in]      axes    the list of axes of rotation: upper-case 'X', 'Y', and 'Z' for the positive X-, Y- and Z-axes,
//!                         and lower-case 'x', 'y', and 'z' for the negative X-, Y-, and Z- axes. The same axis may appear
//!                         more than once, e.g. "ZXZ". This specification is for a coordinate system where the Y-axis is down
//!                         and the Z-axis is out. For a coordinate system where Y goes up and Z comes in, invert the case of all
//!                         'Y' and 'Z' axis specifications. This string is 0-terminated, like any C-string; the length
//!                         of the string determines how many rotations are concatenated.
void NVWARPAPI nvwarpSetEulerRotation(nvwarpHandle han, const float *angles, const char *axes);

//! Get the rotation matrix. This is one in which the Y-axis is directed downward, and the Z axis out.
//! \param[in]      han the Warp360 instance.
//! \param[out]     R   a place to store the current rotation matrix.
void NVWARPAPI nvwarpGetRotation(const nvwarpHandle han, float R[9]);


    // Destination dimensions //

//! Set the destination width and height.
//! \param[in,out]  han the Warp360 instance.
//! \param[in]      w   the desired destination width.
//! \param[in]      h   the desired destination height.
void NVWARPAPI nvwarpSetDstWidthHeight(nvwarpHandle han, uint32_t w, uint32_t h);

//! Get the destination width and height.
//! \param[in]      han the Warp360 instance.
//! \param[out]     wh  an array in which to store the width and height.
void NVWARPAPI nvwarpGetDstWidthHeight(const nvwarpHandle han, uint32_t wh[2]);


    // Destination focal length //

//! Set the destination focal length. Typically, the same focal length is used for X and Y, but these can be different,
//! if a second focal length is supplied. Focal length is a measure of the angular pixel density (in pixels/radian)
//! at the principal point, so the effect of different focal lengths is anisotropic sampling, or rectangular rather than
//! square pixels. To keep the same magnification, set the destination focal length equal to the source focal length.
//! To zoom in by a factor of 2, double the focal length. Be careful when decreasing the focal length to avoid aliasing;
//! you are probably safe down to a factor of 1/2, but you are most certainly going to manifest aliasing when going
//! smaller than a factor of 1/3.
//! \param[in,out]  han the Warp360 instance.
//! \param[in]      fl  the desired X focal length.
//! \param[in]      fy  the desired Y focal length. If zero, it is set to be identical to the X focal length.
//! \note           Negative focal lengths have the effect of reflection, but are not recommended.
void NVWARPAPI nvwarpSetDstFocalLengths(nvwarpHandle han, float fl, float fy);

//! From the vertical view angles, this sets the identical destination focal lengths, plus width and height.
//! The view angles are measured in the center of a symmetric view.
//! The warp type must already be chosen before calling.
//! \param[in,out]  han             the Warp360 instance.
//! \param[in]      topAngle        the top view angle.
//! \param[in]      bottomAngle     the bottom view angle.
//! \param[in]      dstWidth        the width  of the destination.
//! \param[in]      dstHeight       the height of the destination.
//! \note           If (topAngle < bottomAngle), a negative focal length will result, and is not recommended.
void NVWARPAPI nvwarpComputeDstFocalLength(nvwarpHandle han, float topAngle, float bottomAngle, uint32_t dstWidth, uint32_t dstHeight);


//! From the vertical and horizontal axial view angles, this sets the anisotropic destination focal lengths,
//! asymmetric center of projection, width and height.
//! The view angles are measured through the center of projection.
//! The warp type must already be chosen before calling.
//! \param[in,out]  han             the Warp360 instance.
//! \param[in]      topAxialAngle   the  top   view angle through the center of projection.
//! \param[in]      botAxialAngle   the bottom view angle through the center of projection.
//! \param[in]      leftAxialAngle  the  left  view angle through the center of projection.
//! \param[in]      rightAxialAngle the right  view angle through the center of projection.
//! \param[in]      dstWidth        the width  of the destination.
//! \param[in]      dstHeight       the height of the destination.
//! \note           If (topAngle < bottomAngle), a negative focal length will result, and is not recommended.
nvwarpResult NVWARPAPI nvwarpComputeDstFocalLengths(nvwarpHandle han, float topAxialAngle, float botAxialAngle, float leftAxialAngle, float rightAxialAngle,
    uint32_t dstWidth, uint32_t dstHeight);

//! Get one or both destination focal lengths.
//! In most cases, only a single focal length will be used, so NULL would be specified for the second parameter.
//! \param[in]      han the Warp360 instance.
//! \param[out]     fy  a place to store the Y focal length (can be NULL if it is not desired).
//! \return         the destination focal length.
float NVWARPAPI nvwarpGetDstFocalLength(const nvwarpHandle han, float *fy);


    // Destination Principal Point //

//! Set the destination principal point.
//! \param[in,out]  han         the Warp360 instance.
//! \param[in]      xy          the principal point. NULL implies (0,0).
//! \param[in]      relToCenter zero    = the principal point is specified relative to the upper left corner of the image;
//!                             nonzero = the principal point is specified relative to the center of the image.
//! \note           The Y axis is always considered to point downward.
//! \note           The destination width and height must be specified prior to calling nvwarpSetDstPrincipalPoint().
//! \note           A good default is nvwarpSetDstPrincipalPoint(han, NULL, 1);
void NVWARPAPI nvwarpSetDstPrincipalPoint(nvwarpHandle han, const float xy[2], uint32_t relToCenter);

//! Get the destination principal point.
//! \param[in]      han the Warp360 instance.
//! \param[out]     xy  a place to store the destination principal point.
//! \param[in]      relToCenter 0 = relative to the upper left corner of the image,
//!                         1 = relative to the center of the image.
void NVWARPAPI nvwarpGetDstPrincipalPoint(const nvwarpHandle han, float xy[2], uint32_t relToCenter);


    // Control Parameters //

//! Set a control parameter. Most warps do not have one. At the current time,
//! no warps have more than 1 control parameter.
//! \param[in,out]  han         the Warp360 instance.
//! \param[in]      index       the index of the control to be set.
//! \param[in]      control     the desired control value.
//! \note           If a particular warp does not use a control parameter, it is ignored.
void NVWARPAPI nvwarpSetControl(nvwarpHandle han, uint32_t index, float control);

//! Get the value for the control parameters.
//! \param[in]      han     the Warp360 instance.
//! \param[in]      index   the index of the control parameter to retrieve.
//! \return         the value of the specified control parameter, or NaN if there is no such parameter (i.e. index > 0 at the moment).
//! \note           Only control[0] is implemented, but only for a few warps.
float NVWARPAPI nvwarpGetControl(const nvwarpHandle han, uint32_t index);


    // User Data Pointer //

//! Set the user data pointer.
//! \param[in,out]  han         the Warp360 instance.
//! \param[in]      userData    pointer to the user data.
void NVWARPAPI nvwarpSetUserData(nvwarpHandle han, void *userData);

//! Get the current value of the user data pointer.
//! \param[in]      han the Warp360 instance.
//! \return         the current value of the user data pointer.
void* NVWARPAPI nvwarpGetUserData(nvwarpHandle han);



////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////                                                                        ////
////                                Warp functions                          ////
////                                                                        ////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


    // Warp Images //

//! Warp an image texture to a surface.
//! \param[in]      han         the Warp360 instance.
//! \param[in]      stream      the stream on which to execute the warp.
//! \param[in]      srcTex      the source texture.
//! \param[out]     dstSurface  the destination surface.
//! \return         NVWARP_SUCCESS if successful.
nvwarpResult NVWARPAPI nvwarpWarpSurface(const nvwarpHandle han, cudaStream_t stream, cudaTextureObject_t srcTex, cudaSurfaceObject_t dstSurface);

//! Warp an image texture to a buffer.
//! \param[in]      han         the Warp360 instance.
//! \param[in]      stream      the stream on which to execute the warp.
//! \param[in]      srcTex      the source texture.
//! \param[out]     dstAddr     the destination buffer address.
//! \param[in]      dstRowBytes the byte stride between pixels in the buffer vertically.
//! \return         NVWARP_SUCCESS if successful.
nvwarpResult NVWARPAPI nvwarpWarpBuffer(const nvwarpHandle han, cudaStream_t stream, cudaTextureObject_t srcTex, void *dstAddr, size_t dstRowBytes);


    // Warp Coordinates Between Source and Destination //

//! Transform coordinates from the input space to the output space.
//! This works in-place.
//! \param[in]      han         the Warp360 instance.
//! \param[in]      numPts      the number of points to be transformed.
//! \param[in]      inPtsXY     an array of 2D points to be transformed.
//! \param[out]     outPtsXY    a 2D point array of where the transformed 2D points are to be placed.
//!                             This can be the same as inPtsXY.
//! \return         NVWARP_SUCCESS,             if the conversion was successful,
//! \return         NVWARP_ERR_DOMAIN           if it fails due to any coordinate being out of domain.
//! \return         NVWARP_ERR_APPROXIMATE      if tangential distortion is used for a perspective source, due to lack of implementation;
//!                                             only the radial distortion is accommodated.
//! \return         NVWARP_ERR_UNIMPLEMENTED    if it is an unsupported warp type.
nvwarpResult NVWARPAPI nvwarpWarpCoordinates(const nvwarpHandle han, uint32_t numPts, const float *inPtsXY, float *outPtsXY);

//! Transform coordinates from the destination space to the source space.
//! This works in-place.
//! \param[in]      han         the Warp360 instance.
//! \param[in]      numPts      the number of points to be transformed.
//! \param[in]      inPtsXY     an array of 2D points to be transformed.
//! \param[out]     outPtsXY    a 2D point array of where the transformed 2D points are to be placed.
//!                             This can be the same as inPtsXY.
//! \return         NVWARP_SUCCESS,             if the conversion was successful,
//! \return         NVWARP_ERR_DOMAIN           if it fails due to any coordinate being out of domain.
//! \return         NVWARP_ERR_UNIMPLEMENTED    if it is an unsupported warp type.
//! \note           Tangential distortion is implemented for perspective.
nvwarpResult NVWARPAPI nvwarpInverseWarpCoordinates(const nvwarpHandle han, uint32_t numPts, const float *inPtsXY, float *outPtsXY);


    // Ray <--> Coordinate Conversion //

//! Convert source coordinates into normalized rays.
//! \param[in]      han     the Warp360 instance.
//! \param[in]      numPts  the number of points to be converted into rays.
//! \param[in]      pts2D   the array of 2D points to be transformed into 3D rays.
//! \param[out]     rays3D  the array into which the 3D rays are to be placed. NANs are returned for unsuccessful conversions.
//! \return         NVWARP_SUCCESS,             if the conversion was successful,
//! \return         NVWARP_ERR_DOMAIN           if it fails due to being out of domain.
//! \return         NVWARP_ERR_APPROXIMATE      if tangential distortion is used for a perspective source, due to lack of implementation;
//!                                             only the radial distortion is accommodated.
//! \return         NVWARP_ERR_UNIMPLEMENTED    if it is an unsupported warp type.
//! \note           The nvwarpSrcToRay(), nvwarpDstToRay(), nvwarpSrcFromRay() and nvwarpDstFromRay() functions
//!                 can be used to warp coordinates as follows:
//! \code
//!                 float srcPt[2], dstPt[2], srcRay[3], dstRay[3], M[9];
//!                 ...
//!                 nvwarpGetRotation(warper, M);
//!                 nvwarpSrcToRay(warper, 1, srcPt, srcRay); // Warp
//!                 dstRay[0] = M[0] * srcRay[0] + M[1] * srcRay[1] + M[2] * srcRay[2];
//!                 dstRay[1] = M[3] * srcRay[0] + M[4] * srcRay[1] + M[5] * srcRay[2];
//!                 dstRay[2] = M[6] * srcRay[0] + M[7] * srcRay[1] + M[8] * srcRay[2];
//!                 nvwarpDstFromRay(warper, 1, dstRay, dstPt);
//!                 ...
//!                 nvwarpDstToRay(warper, 1, dstPt, dstRay); // Inverse Warp
//!                 srcRay[0] = M[0] * dstRay[0] + M[3] * dstRay[1] + M[6] * dstRay[2];
//!                 srcRay[1] = M[1] * dstRay[0] + M[4] * dstRay[1] + M[7] * dstRay[2];
//!                 srcRay[2] = M[2] * dstRay[0] + M[5] * dstRay[1] + M[8] * dstRay[2];
//!                 nvwarpSrcFromRay(warper, 1, srcRay, srcPt);
//! \endcode
nvwarpResult NVWARPAPI nvwarpSrcToRay(const nvwarpHandle han, uint32_t numPts, const float *pts2D, float *rays3D);

//! Convert destination coordinates into normalized rays.
//! \param[in]      han     the Warp360 instance.
//! \param[in]      numPts  the number of points to be converted into rays.
//! \param[in]      pts2D   the array of 2D points to be transformed into 3D rays.
//! \param[out]     rays3D  the array into which the 3D rays are to be placed. NANs are returned for unsuccessful conversions.
//! \return         NVWARP_SUCCESS,             if the conversion was successful,
//! \return         NVWARP_ERR_DOMAIN           if it fails due to any coordinate being out of domain.
//! \return         NVWARP_ERR_UNIMPLEMENTED    if it is an unsupported warp type.
nvwarpResult NVWARPAPI nvwarpDstToRay(const nvwarpHandle han, uint32_t numPts, const float *pts2D, float *rays3D);

//! Convert rays into source coordinates. The rays do not need to be normalized.
//! \param[in]      han     the Warp360 instance.
//! \param[in]      numRays the number of rays to be converted into points.
//! \param[in]      rays3D  the array of 3D rays to be transformed into 2D points.
//! \param[out]     pts2D   the array into which the 2D points are to be placed. NANs are returned for unsuccessful conversions.
//! \return         NVWARP_SUCCESS,             if the conversion was successful,
//! \return         NVWARP_ERR_DOMAIN           if it fails due to being out of domain.
//! \return         NVWARP_ERR_UNIMPLEMENTED    if it is an unsupported warp type.
nvwarpResult NVWARPAPI nvwarpSrcFromRay(const nvwarpHandle han, uint32_t numRays, const float *rays3D, float *pts2D);

//! Convert rays into destination coordinates. The rays do not need to be normalized.
//! \param[in]      han     the Warp360 instance.
//! \param[in]      numRays the number of rays to be converted into points.
//! \param[in]      rays3D  the array of 3D rays to be transformed into 2D points.
//! \param[out]     pts2D   the array into which the 2D points are to be placed. NANs are returned for unsuccessful conversions.
//! \return         NVWARP_SUCCESS,             if the conversion was successful,
//! \return         NVWARP_ERR_DOMAIN           if it fails due to being out of domain.
//! \return         NVWARP_ERR_UNIMPLEMENTED    if it is an unsupported warp type.
nvwarpResult NVWARPAPI nvwarpDstFromRay(const nvwarpHandle han, uint32_t numRays, const float *rays3D, float *pts2D);


    // Angle <--> Coordinate Conversion //

//! Compute the angular coordinates {longitude, latitude} of for the given points in the source.
//! \param[in]      han     the Warp360 instance.
//! \param[in]      numPts  the number of 2D points to be converted into angles.
//! \param[in]      pts2D   the array of 2D points to be transformed into 2D angles.
//! \param[out]     ang2D   the array into which the 2D angles are to be placed. NANs are returned for unsuccessful conversions.
//! \return         NVWARP_SUCCESS,             if the conversion was successful,
//! \return         NVWARP_ERR_DOMAIN           if it fails due to any coordinate being out of domain.
//! \return         NVWARP_ERR_UNIMPLEMENTED    if it is an unsupported warp type.
nvwarpResult NVWARPAPI nvwarpComputeSrcAngularFromPixelCoordinates(const nvwarpHandle han, uint32_t numPts, const float *pts2D, float *ang2D);

//! Compute the pixel coordinates for the given angular coordinates {longitude, latitude} in the source.
//! \param[in]      han     the Warp360 instance.
//! \param[in]      numPts  the number of angle pairs to be converted into 2D points.
//! \param[in]      ang2D   the array of 2D angles to be converted into 2D points.
//! \param[out]     pts2D   the array into which the 2D points are to be placed. NANs are returned for unsuccessful conversions.
//! \return         NVWARP_SUCCESS,             if the conversion was successful,
//! \return         NVWARP_ERR_DOMAIN           if it fails due to any coordinate being out of domain.
//! \return         NVWARP_ERR_UNIMPLEMENTED    if it is an unsupported warp type.
nvwarpResult NVWARPAPI nvwarpComputeSrcPixelFromAngularCoordinates(const nvwarpHandle han, uint32_t numPts, const float *ang2D, float *pts2D);

//! Compute the angular coordinates {longitude, latitude} of for the given points in the destination.
//! \param[in]      han     the Warp360 instance.
//! \param[in]      numPts  the number of points to be converted into rays.
//! \param[in]      pts2D   the array of 2D points to be transformed into 3D rays.
//! \param[out]     ang2D   the array into which the 2D angles are to be placed. NANs are returned for unsuccessful conversions.
//! \return         NVWARP_SUCCESS,             if the conversion was successful,
//! \return         NVWARP_ERR_DOMAIN           if it fails due to any coordinate being out of domain.
//! \return         NVWARP_ERR_UNIMPLEMENTED    if it is an unsupported warp type.
nvwarpResult NVWARPAPI nvwarpComputeDstAngularFromPixelCoordinates(const nvwarpHandle han, uint32_t numPts, const float *pts2D, float *ang2D);

//! Compute the pixel coordinates for the given angular coordinates {longitude, latitude} in the destination.
//! \param[in]      han     the Warp360 instance.
//! \param[in]      numPts  the number of angle pairs to be converted into 2D points.
//! \param[in]      ang2D   the array of 2D angles to be converted into 2D points.
//! \param[out]     pts2D   the array into which the 2D points are to be placed. NANs are returned for unsuccessful conversions.
//! \return         NVWARP_SUCCESS,             if the conversion was successful,
//! \return         NVWARP_ERR_DOMAIN           if it fails due to any coordinate being out of domain.
//! \return         NVWARP_ERR_UNIMPLEMENTED    if it is an unsupported warp type.
nvwarpResult NVWARPAPI nvwarpComputeDstPixelFromAngularCoordinates(const nvwarpHandle han, uint32_t numPts, const float *ang2D, float *pts2D);


    // Miscellaneous //

//! Invert the sense of the Y- and Z-axes in the specified transformation,
//! keeping the X axis pointing in the same direction, i.e.
//! converting between {X-right, Y-up, Z-in} and {X-right, Y-down, Z-out}.
//! \param[in]      fr  the initial transformation to be converted.
//! \param[out]     to  a place to store the converted transformation (can the same as fr).
void NVWARPAPI nvwarpConvertTransformBetweenYUpandYDown(const float fr[9], float to[9]);



////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////              Conversion from YUV 4:2:0 NV12 to RGBA                  /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////



//! Parameters for YUV:420:NV12 --> RGBA conversion.
typedef struct nvwarpYUVRGBParams_t
{
    uint32_t width;     //!< The width of the Y and RGB channels (chroma has half the width, but is interleaved in NV12).
    uint32_t height;    //!< The height of the Y and RGB channels (chroma has half the height, so this must be even).
    uint32_t cLocation; //!< 0 for chroma sampled cosited horizontally with luma; 1 for chroma sampled halfway between luma samples horizontally. Get the the video header.
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
} nvwarpYUVRGBParams_t;


//! General method to compute all values except for {width, height, cLocation}.
//! \param[out] p                       the location of the parameters that will be set by this method.
//! \param[in]  matrix_coefficients     One of {709, 1, 2} for BT.709, {601, 5, 6} for BT.601, {2020, 9, 10} for BT.2020, {4} for FCC, {240, 7} FOR SMPTE 240M.
//! \param[in]  video_full_range_flag   0 for standard video range (16-240), 1 for full range (0-255).
//! \param[in]  bit_depth               8 is the only depth that is supported by nvwarpConvertYUVNV12ToRGBA().
//! \param[in]  normalized_input        1 if the input  is normalized, 0 if not.
//! \param[in]  normalized_output       1 if the output is normalized, 0 if not.
void NVWARPAPI nvwarpComputeYCbCr2RgbMatrix(nvwarpYUVRGBParams_t *p, uint32_t matrix_coefficients, uint32_t video_full_range_flag, uint32_t bit_depth, uint32_t normalized_input, uint32_t normalized_output);


//! Perform YUV 4:2:0 NV12 --> RGBA conversion.
//! \param[in]  stream      The stream on which the computation is to be performed.
//! \param[in]  params      The parameters controlling the YUV-->RGB conversion; typically set with YUVRGBParams.computeMatrix().
//! \param[in]  yuv         Pointer to the YV CUDA pitched memory buffer.
//! \param[in]  yuvRowBytes Byte stride between pixels vertically in the luminance (and chrominance) of YUV.
//! \param[in]  dst         Pointer to the RGBA CUDA pitched memory buffer.
//! \param[in]  dstRowBytes Byte stride between pixels vertically in the RGBA.
//! \return NVWARP_SUCCESS if successful.
nvwarpResult NVWARPAPI nvwarpConvertYUVNV12ToRGBA(cudaStream_t stream, const nvwarpYUVRGBParams_t *params, const void *yuv, size_t yuvRowBytes, void *dst, size_t dstRowBytes);



////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/////                                                                      /////
/////                              MultiWarp                               /////
/////                                                                      /////
/////           The implementation is distributed as sample code,          /////
/////           to optimize better for specific applications.              /////
/////                                                                      /////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////



//! Perform an optional YUV:420:NV12 --> RGBA conversion, followed by a suite of warps from that conversion.
//! This uses an array of parameter blocks.
//! \param[in]  stream      Stream on which to perform this computation.
//! \param[in]  yuvParams   Parameters for the YUV-->RGB conversion (can be NULL).
//! \param[in]  yuvBuffer   The buffer containing the YUV data in 420 NV12 format.
//! \param[in]  yuvRowBytes The byte stride between luminance (and chroma) pixels vertically.
//! \param[in]  rgbBuffer   The buffer where the YUV-->RGB conversion is to be placed.
//! \param[in]  rgbRowBytes The byte stride between RGBA pixels vertically.
//! \param[in]  rgbTex      The texture associated with the RGB buffer, initialized appropriately.
//! \param[in]  numWarps    The number of warps to be executed on using the RGB buffer texture as a source.
//! \param[in]  paramArray  The array of parameter blocks for each warp.
//! \param[out] dstBuffers  The array of pointers to the destination buffers.
//! \param[in]  dstRowBytes The array of byte strides between pixels vertically, one for each warp.
//! \return NVWARP_SUCCESS if successful.
nvwarpResult NVWARPAPI nvwarpMultiWarp360(cudaStream_t stream,
    const nvwarpYUVRGBParams_t *yuvParams, const void *yuvBuffer, size_t yuvRowBytes,
    void *rgbBuffer, size_t rgbRowBytes, cudaTextureObject_t rgbTex,
    uint32_t numWarps, const nvwarpParams_t *paramArray, void **dstBuffers, const size_t *dstRowBytes);


#ifdef __cplusplus
}
#endif // __cplusplus


#endif /* __NVWARP360_H__ */
