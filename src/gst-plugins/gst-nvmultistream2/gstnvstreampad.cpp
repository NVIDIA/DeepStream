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
