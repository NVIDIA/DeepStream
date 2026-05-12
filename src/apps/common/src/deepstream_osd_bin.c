/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2019 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "deepstream_common.h"
#include "deepstream_osd.h"

gboolean
create_osd_bin (NvDsOSDConfig * config, NvDsOSDBin * bin)
{
  gboolean ret = FALSE;
  guint clk_color =
      ((((guint) (config->clock_color.red * 255)) & 0xFF) << 24) |
      ((((guint) (config->clock_color.green * 255)) & 0xFF) << 16) |
      ((((guint) (config->clock_color.blue * 255)) & 0xFF) << 8) | 0xFF;

  bin->bin = gst_bin_new ("osd_bin");
  if (!bin->bin) {
    NVGSTDS_ERR_MSG_V ("Failed to create 'osd_bin'");
    goto done;
  }

  bin->nvvidconv = gst_element_factory_make (NVDS_ELEM_VIDEO_CONV, "osd_conv");
  if (!bin->nvvidconv) {
    NVGSTDS_ERR_MSG_V ("Failed to create 'osd_conv'");
    goto done;
  }

  bin->queue = gst_element_factory_make (NVDS_ELEM_QUEUE, "osd_queue");
  if (!bin->queue) {
    NVGSTDS_ERR_MSG_V ("Failed to create 'osd_queue'");
    goto done;
  }

  bin->conv_queue =
      gst_element_factory_make (NVDS_ELEM_QUEUE, "osd_conv_queue");
  if (!bin->conv_queue) {
    NVGSTDS_ERR_MSG_V ("Failed to create 'osd_conv_queue'");
    goto done;
  }

  bin->nvosd = gst_element_factory_make (NVDS_ELEM_OSD, "nvosd0");
  if (!bin->nvosd) {
    NVGSTDS_ERR_MSG_V ("Failed to create 'nvosd0'");
    goto done;
  }

  g_object_set (G_OBJECT (bin->nvosd), "display-clock", config->enable_clock,
      "clock-font", config->font, "x-clock-offset", config->clock_x_offset,
      "y-clock-offset", config->clock_y_offset, "clock-color", clk_color,
      "clock-font-size", config->clock_text_size, "process-mode", config->mode,
      NULL);

  gst_bin_add_many (GST_BIN (bin->bin), bin->queue,
      bin->nvvidconv, bin->conv_queue, bin->nvosd, NULL);

  g_object_set (G_OBJECT (bin->nvvidconv), "gpu-id", config->gpu_id, NULL);

  g_object_set (G_OBJECT (bin->nvvidconv), "nvbuf-memory-type",
      config->nvbuf_memory_type, NULL);

  g_object_set (G_OBJECT (bin->nvosd), "gpu-id", config->gpu_id, NULL);
  g_object_set (G_OBJECT (bin->nvosd), "display-text", config->draw_text, NULL);
  g_object_set (G_OBJECT (bin->nvosd), "display-bbox", config->draw_bbox, NULL);
  g_object_set (G_OBJECT (bin->nvosd), "display-mask", config->draw_mask, NULL);
  if (config->mode == 2 && config->hw_blend_color_attr)
    g_object_set (G_OBJECT (bin->nvosd), "hw-blend-color-attr",
        config->hw_blend_color_attr, NULL);

  NVGSTDS_LINK_ELEMENT (bin->queue, bin->nvvidconv);

  NVGSTDS_LINK_ELEMENT (bin->nvvidconv, bin->conv_queue);

  NVGSTDS_LINK_ELEMENT (bin->conv_queue, bin->nvosd);

  NVGSTDS_BIN_ADD_GHOST_PAD (bin->bin, bin->queue, "sink");

  NVGSTDS_BIN_ADD_GHOST_PAD (bin->bin, bin->nvosd, "src");

  ret = TRUE;
done:
  if (!ret) {
    NVGSTDS_ERR_MSG_V ("%s failed", __func__);
  }
  return ret;
}
