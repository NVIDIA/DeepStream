/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "gstnvstreampad.h"

G_DEFINE_TYPE (GstNvStreamPad, gst_nvstream_pad, GST_TYPE_PAD);

static void
gst_nvstream_pad_class_init (GstNvStreamPadClass * klass)
{
}

static void
gst_nvstream_pad_init (GstNvStreamPad * pad)
{
  pad->got_eos = FALSE;
}
