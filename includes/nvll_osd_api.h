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

/**
 * @file
 * <b>NVIDIA Multimedia Utilities: On-Screen Display API</b>
 *
 * This file defines the NvOSD library, used to draw rectangles and text
 * over the frame.
 */
/**
 * @defgroup NvDsOsdApi On-Screen Display API
 *
 * Defines the functions of On-screen Display APIs
 *
 */
/**
 * @defgroup ee_nvosd_api_group On-Screen Display API
 * Defines the NvOSD library to be used to draw rectangles and text
 * over the frame.
 * @ingroup NvDsOsdApi
 * @{
 */

#ifndef __NVLL_OSD_API_DEFS__
#define __NVLL_OSD_API_DEFS__

#include "nvll_osd_struct.h"
#include "nvbufsurface.h"

#define NVOSD_PRINT_E(f_, ...) \
  printf("libnvosd (%d):(ERROR) : " f_, __LINE__, ##__VA_ARGS__)

#define MAX_IN_BUF 16
#define MAX_BORDER_WIDTH 32

#ifdef __cplusplus
extern "C"
{
#endif

typedef void * NvOSDCtxHandle;
/**
 * Holds information about the text in a frame.
 */
typedef struct _NvOSD_FrameTextParams
{
  /** Holds a pointer to a buffer containing a frame. */
  /** buf_ptr is deprecated. Use NvBufSurface *surf instead */
  NvBufSurfaceParams *buf_ptr;
  /** Holds the OSD mode to be used for processing. */
  NvOSD_Mode mode;
  /** Holds the number of strings. */
  int num_strings;
  /** Holds the strings' text parameters. */
  NvOSD_TextParams *text_params_list;
  /** Holds information about batched buffers. */
  NvBufSurface *surf;
} NvOSD_FrameTextParams;

/** Holds information about the rectangles in a frame. */
typedef struct _NvOSD_FrameRectParams
{
  /** Holds a pointer to the buffer containing the frame. */
  /** buf_ptr is deprecated. Use NvBufSurface *surf instead */
  NvBufSurfaceParams *buf_ptr;
  /** Holds the OSD mode to be used for processing. */
  NvOSD_Mode mode;
  /** Holds the number of rectangles. */
  int num_rects;
  /** Holds the rectangles' parameters. */
  NvOSD_RectParams *rect_params_list;
  /** Holds mask parameters */
  NvOSD_MaskParams *mask_params_list;
  /** Holds information about batched buffers. */
  NvBufSurface *surf;
} NvOSD_FrameRectParams;

/** Holds information about the rectangles in a frame. */
typedef struct _NvOSD_FrameSegmentMaskParams
{
  /** Holds a pointer to the buffer containing the frame. */
  /** buf_ptr is deprecated. Use NvBufSurface *surf instead */
  NvBufSurfaceParams *buf_ptr;
  /** Holds the OSD mode to be used for processing. */
  NvOSD_Mode mode;
  /** Holds the number of rectangles. */
  int num_segments;
  /** Holds the rectangles' parameters. */
  NvOSD_RectParams *rect_params_list;
  /** Holds mask parameters */
  NvOSD_MaskParams *mask_params_list;
  /** Holds information about batched buffers. */
  NvBufSurface *surf;
} NvOSD_FrameSegmentMaskParams;

/** Holds information about the lines in a frame. */
typedef struct _NvOSD_FrameLineParams
{
  /** Holds a pointer to the buffer containing the frame. */
  /** buf_ptr is deprecated. Use NvBufSurface *surf instead */
  NvBufSurfaceParams *buf_ptr;
  /** Holds the OSD mode to be used for processing. */
  NvOSD_Mode mode;
  /** Holds the number of lines. */
  int num_lines;
  /** Holds the lines' parameters. */
  NvOSD_LineParams *line_params_list;
  /** Holds information about batched buffers. */
  NvBufSurface *surf;
} NvOSD_FrameLineParams;

/** Holds information about the arrows in a frame. */
typedef struct _NvOSD_FrameArrowParams
{
  /** Holds a pointer to the buffer containing the frame. */
  /** buf_ptr is deprecated. Use NvBufSurface *surf instead */
  NvBufSurfaceParams *buf_ptr;
  /** Holds the OSD mode to be used for processing. */
  NvOSD_Mode mode;
  /** Holds the number of arrows. */
  int num_arrows;
  /** Holds the parameters of the arrows. */
  NvOSD_ArrowParams *arrow_params_list;
  /** Holds information about batched buffers. */
  NvBufSurface *surf;
} NvOSD_FrameArrowParams;

/** Holds information about the circles in a frame. */
typedef struct _NvOSD_FrameCircleParams
{
  /** Holds a pointer to the buffer containing the frame. */
  /** buf_ptr is deprecated. Use NvBufSurface *surf instead */
  NvBufSurfaceParams *buf_ptr;
  /** Holds the OSD mode to be used for processing. */
  NvOSD_Mode mode;
  /** Holds the number of circles. */
  int num_circles;
  /** Holds the parameters of the circles. */
  NvOSD_CircleParams *circle_params_list;
  /** Holds information about batched buffers. */
  NvBufSurface *surf;
} NvOSD_FrameCircleParams;

/**
 * Creates an NvOSD context.
 *
 * @returns A handle for an NvOSD context if successful, or NULL otherwise.
 */
NvOSDCtxHandle nvll_osd_create_context(void);

/**
 * Destroys an NvOSD context.
 *
 * @param[in] nvosd_ctx A handle for the NvOSD context to be destroyed.
 */
void nvll_osd_destroy_context(NvOSDCtxHandle nvosd_ctx);

/**
 * \brief  Sets clock parameters for a given context.
 *
 * The clock is overlayed when nvll_osd_put_text() is called.
 * If no other text is to be overlayed, %nvll_osd_put_text() must be called with
 * @a num_strings as 0 and @a text_params_list as NULL.
 *
 * @param[in] nvosd_ctx     A handle for an NvOSD context.
 * @param[in] clk_params    A pointer to a structure for the clock
 *                           to be overlayed; NULL to disable the clock.
 */
void nvll_osd_set_clock_params(NvOSDCtxHandle nvosd_ctx, NvOSD_TextParams *clk_params);


/**
 * /brief  Overlays clock and given text at a given location in a buffer.
 *
 * To overlay the clock, you must call nvll_osd_set_clock_params().
 * You must also ensure that the length of @a text_params_list is at least
 * @ref NvOSD_FrameTextParams::num_strings.
 *
 * @note Currently only the NvOSD_Mode value @ref MODE_CPU is supported.
 * Specifying other modes has no effect.
 *
 * @param[in] nvosd_ctx         A handle for an NvOSD context.
 * @param[in] frame_text_params A pointer to a structure containing
 *                               information about rectangles to be overlayed.
 *
 * @returns 0 for success, or -1 for failure.
 */
int nvll_osd_put_text(NvOSDCtxHandle nvosd_ctx, NvOSD_FrameTextParams *frame_text_params);

/**
 * \brief  Overlays segment masks at a given location in a buffer.
 *
 * You must ensure that the length of @a mask_params_list is at least
 * @a num_segments.
 *
 * @param[in] nvosd_ctx A handle for an NvOSD context.
 * @param[in] frame_mask_params A pointer to the FrameSegmentMaskParams struct
 *            containing mask information to be overlayed.
 *
 * @returns 0 for success, -1 for failure.
 */
int nvll_osd_draw_segment_masks(NvOSDCtxHandle nvosd_ctx, NvOSD_FrameSegmentMaskParams *frame_mask_params);

/**
 * \brief  Blur boxes at a given location in a buffer.
 *
 * You must ensure that the length of @a rect_params_list is at least
 * @a num_rects.
 *
 * @param[in] nvosd_ctx A handle for an NvOSD context.
 * @param[in] frame_rect_params A pointer to the FrameRectParams struct
 *            containing rectangles information to be overlayed.
 *
 * @returns 0 for success, -1 for failure.
 */
int nvll_osd_blur_rectangles(NvOSDCtxHandle nvosd_ctx, NvOSD_FrameRectParams *frame_rect_params);

/**
 * \brief  Overlays boxes at a given location in a buffer.
 *
 * Boxes can be configured with:
 * a. Border only:
 *    Uou must set @a border_width and set
 *    @a has_bg_color to 0 for the given box.
 * b. Border and background color
 *    To draw boxes with border and background color, you must set @a
 *    border_width and set @a has_bg_color to 1, and specify background color
 *    parameters for the given box.
 * c. Solid fill acting as mask region
 *    To draw boxes with solid fill acting as mask region, you must set @a
 *    border_width to 0 and @a has_bg_color to 1 for the given box.
 *
 *
 * You must ensure that the length of @a rect_params_list is at least
 * @a num_rects.
 *
 * @param[in] nvosd_ctx A handle for an NvOSD context.
 * @param[in] frame_rect_params A pointer to the FrameRectParams struct
 *            containing rectangles information to be overlayed.
 *
 * @returns 0 for success, -1 for failure.
 */
int nvll_osd_draw_rectangles(NvOSDCtxHandle nvosd_ctx, NvOSD_FrameRectParams *frame_rect_params);

/**
 * Overlays lines on the buffer.
 *
 * The length of @a line_params_list must be equal to @a num_lines.
 * The client is responsible for allocating this array.
 *
 * @note Currently only MODE_CPU is supported. Specifying other modes will have
 * no effect.
 *
 * @param[in] nvosd_ctx A handle for an NvOSD context.
 * @param[in] frame_line_params A pointer to the FrameLineParams struct
 *            containing line information to be overlayed.
 *
 * @returns 0 if successful, or -1 otherwise.
 */
int nvll_osd_draw_lines(NvOSDCtxHandle nvosd_ctx, NvOSD_FrameLineParams *frame_line_params);

/**
 * Overlays arrows on the buffer.
 *
 * The length of @a arrow_params_list must equal @a num_arrows.
 * The client is responsible for allocating this array.
 *
 * @note Currently only @ref MODE_CPU is supported. Specifying other modes has no effect.
 *
 * @param[in] nvosd_ctx A handle for an NvOSD context.
 * @param[in] frame_arrow_params A pointer to the FrameArrowParams struct
 *            containing arrow information to be overlayed.
 *
 * @returns 0 if successful, or -1 otherwise.
 */
int nvll_osd_draw_arrows(NvOSDCtxHandle nvosd_ctx, NvOSD_FrameArrowParams *frame_arrow_params);

/**
 * Overlays circles on the buffer.
 *
 * You must ensure that the length of @a circle_params_list equals @a num_circles.
 *
 * @note Currently only @ref MODE_CPU is supported. Specifying other modes has no effect.
 *
 * @param[in] nvosd_ctx A handle for an NvOSD context.
 * @param[in] frame_circle_params A pointer to the FrameCircleParams struct
 *            containing circle information to be overlayed.
 *
 * @returns 0 if successful, or -1 otherwise.
 */
int nvll_osd_draw_circles(NvOSDCtxHandle nvosd_ctx, NvOSD_FrameCircleParams *frame_circle_params);

/**
 * Sets the resolution of the frames on which the NvOSDContext is to operate.
 *
 * @param[in] nvosd_ctx A handle for the NvOSD context.
 * @param[in] width Width of the input frames.
 * @param[in] height Height of the input frames.
 *
 * @returns 0 if successful, or -1 otherwise.
 */
void *nvll_osd_set_params (NvOSDCtxHandle nvosd_ctx, int width, int height);

/**
 * Initializes colors for HW based blending.
 *
 * Applicable ONLY for Jetson.
 *
 * @param[in] nvosd_ctx A pointer to NvOSD context.
 * @param[in] color_info A pointer to the Color_info struct
 *            containing color information.
 * @param[in] num_classes Number of classes.
 *
 * @returns A pointer to the internally allocated Host Memory.
 */
int nvll_osd_init_colors_for_hw_blend(void *nvosd_ctx, NvOSD_Color_info * color_info, int num_classes);

/**
 * Apply OSD to the video frame for GPU backend.
 *
 * @param[in] nvosd_ctx A handle for the NvOSD context.
 * @param[in] buf_ptr Surface which the OSD apply.
 *            This is now deprecated. 'NvBufSurface *surf' is used instead.
 * @param[in] surf Holds information of batched buffer
 */
int nvll_osd_apply (NvOSDCtxHandle nvosd_ctx, NvBufSurfaceParams *buf_ptr, NvBufSurface *surf);

#ifdef __cplusplus
}
#endif
/** @} */
#endif
