/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

/**
 * @file
 * <b>NVIDIA Multimedia Utilities: On-Screen Display Manager</b>
 *
 * This file defines the NvOSD library used to draw rectangles and text
 * over the frame.
 */

/**
 * @defgroup ee_nvosd_group On-Screen Display Manager
 * Defines the NvOSD library to be used to draw rectangles and text
 * over the frame.
 * @ingroup NvDsOsdApi
 * @{
 */

#ifndef __NVLL_OSD_STRUCT_DEFS__
#define __NVLL_OSD_STRUCT_DEFS__

#include <stdbool.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * Defines modes used to overlay boxes and text.
 */
typedef enum {
    MODE_CPU, /**< Specifies using the CPU for OSD processing.
                Works with RGBA data only */
    MODE_GPU, /**< Specifies using the GPU for OSD processing.
                Currently not implemented. */
    MODE_NONE /**< Invalid mode. Instead GPU mode will be used. */
} NvOSD_Mode;

/**
 * Specifies arrow head positions.
 */
typedef enum
{
  /** Specifies an arrow head only at start = 0. */
  START_HEAD,
  /** Specifies an arrow head only at end = 1. */
  END_HEAD,
  /** Specifies arrow heads at both start and end = 2. */
  BOTH_HEAD
} NvOSD_Arrow_Head_Direction;

/**
 * Holds unclipped bounding box coordinates of the object.
 */
typedef struct _NvBbox_Coords {
  float left;            /**< Holds the box's left coordinate
                                     in pixels. */

  float top;             /**< Holds the box's top coordinate
                                     in pixels. */

  float width;           /**< Holds the box's width in pixels. */

  float height;          /**< Holds the box's height in pixels. */
} NvBbox_Coords;

/**
 * Holds the color parameters of the box or text to be overlayed.
 */
typedef struct _NvOSD_ColorParams {
  double red;     /**< Holds the red component of color.
                   Value must be in the range 0.0-1.0. */

  double green;   /**< Holds the green component of color.
                   Value must be in the range 0.0-1.0.*/

  double blue;    /**< Holds the blue component of color.
                   Value must be in the range 0.0-1.0.*/

  double alpha;   /**< Holds the alpha component of color.
                   Value must be in the range 0.0-1.0.*/
} NvOSD_ColorParams;

/**
 * Holds the font parameters of the text to be overlayed.
 */
typedef struct _NvOSD_FontParams {
  char * font_name;             /**< Holds a pointer to the string containing
                                 the font name. To display a list of
                                 supported fonts, run the fc-list command. */

//  char font_name[64];         /**< Holds a pointer to a string containing
//                               the font name. */

  unsigned int font_size;       /**< Holds the size of the font. */

  NvOSD_ColorParams font_color; /**< Holds the font color. */
} NvOSD_FontParams;


/**
 * Holds parameters of text to be overlayed.
 */
typedef struct _NvOSD_TextParams {
  char * display_text;          /**< Holds the text to be overlayed. */

  unsigned int x_offset;        /**< Holds the text's horizontal offset from
                                 the top left pixel of the frame. */
  unsigned int y_offset;        /**< Holds the text's vertical offset from the
                                 top left pixel of the frame. */
  NvOSD_FontParams font_params; /**< Holds the font parameters of the text
                                 to be overlaid. */
  int set_bg_clr;               /**< Holds a Boolean; true if the text has a
                                 background color. */

  NvOSD_ColorParams text_bg_clr;/**< Holds the text's background color, if
                                 specified. */

} NvOSD_TextParams;

typedef struct _NvOSD_Color_info {
    int id;
    NvOSD_ColorParams color;
}NvOSD_Color_info;

/**
 * Holds the box parameters of the box to be overlayed.
 */
typedef struct _NvOSD_RectParams {
  float left;            /**< Holds the box's left coordinate
                                     in pixels. */

  float top;             /**< Holds the box's top coordinate
                                     in pixels. */

  float width;           /**< Holds the box's width in pixels. */

  float height;          /**< Holds the box's height in pixels. */

  unsigned int border_width;    /**< Holds the box's border width in pixels. */

  NvOSD_ColorParams border_color;
                                /**< Holds the box's border color. */

  unsigned int has_bg_color;    /**< Holds a Boolean; true if the box has a
                                 background color. */

  unsigned int reserved;        /**< Holds a field reserved for future use. */

  NvOSD_ColorParams bg_color;   /**< Holds the box's background color. */

  int has_color_info;
  int color_id;

  float rotation_angle;           /**< Holds the box's rotation angle in degrees in case of OBB. */
} NvOSD_RectParams;

/**
 * Holds the mask parameters of the segment to be overlayed
 */
typedef struct _NvOSD_MaskParams {
  float *data;                   /** Mask buffer */
  unsigned int size;             /** Size of mask buffer in bytes */
  float threshold;               /** Threshold for binarization */
  unsigned int width;            /** Mask width */
  unsigned int height;           /** Mask height */
} NvOSD_MaskParams;

/**
 * Holds the box parameters of a line to be overlayed.
 */
typedef struct _NvOSD_LineParams {
  unsigned int x1;              /**< Holds the box's left coordinate
                                 in pixels. */

  unsigned int y1;              /**< Holds the box's top coordinate
                                 in pixels. */

  unsigned int x2;              /**< Holds the box's width in pixels. */

  unsigned int y2;              /**< Holds the box's height in pixels. */

  unsigned int line_width;      /**< Holds the box's border width in pixels. */

  NvOSD_ColorParams line_color; /**< Holds the box's border color. */
} NvOSD_LineParams;

/**
 * Holds arrow parameters to be overlaid.
 */
typedef struct _NvOSD_ArrowParams {
  unsigned int x1;   /**< Holds the start horizontal coordinate in pixels. */

  unsigned int y1;    /**< Holds the start vertical coordinate in pixels. */

  unsigned int x2;  /**< Holds the end horizontal coordinate in pixels. */

  unsigned int y2; /**< Holds the end vertical coordinate in pixels. */

  unsigned int arrow_width; /**< Holds the arrow shaft width in pixels. */

  NvOSD_Arrow_Head_Direction arrow_head;
                            /**< Holds the arrowhead position. */

  NvOSD_ColorParams arrow_color;
                            /**< Holds color parameters of the arrow box. */

  unsigned int reserved;    /**< Reserved for future use; currently
                               for internal use only. */
} NvOSD_ArrowParams;

/**
 * Holds circle parameters to be overlayed.
 */
typedef struct _NvOSD_CircleParams {
  unsigned int xc;   /**< Holds the start horizontal coordinate in pixels. */

  unsigned int yc;    /**< Holds the start vertical coordinate in pixels. */

  unsigned int radius;  /**< Holds the radius of circle in pixels. */

  NvOSD_ColorParams circle_color;
                        /**< Holds the color parameters of the arrow box. */

  unsigned int has_bg_color;   /*< Holds a Boolean value indicating whether
                                   the circle has a background color. */

  NvOSD_ColorParams bg_color;  /*< Holds the circle's background color. */

  unsigned int circle_width;  /**< Holds the circle's border width in pixels. */

  unsigned int reserved; /**< Reserved for future use; currently
                            for internal use only. */
} NvOSD_CircleParams;

#ifdef __cplusplus
}
#endif
/** @} */
#endif
