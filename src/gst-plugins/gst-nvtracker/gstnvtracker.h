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

#ifndef __GST_NVTRACKER_H__
#define __GST_NVTRACKER_H__

#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/video/gstvideofilter.h>
#include <sys/time.h>

#include "gstnvdsmeta.h"
#include "invtracker_proc.h"

using namespace std;

G_BEGIN_DECLS

/* #defines don't like whitespacey bits */
#define GST_TYPE_NVTRACKER \
  (gst_nv_tracker_get_type())
#define GST_NVTRACKER(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_NVTRACKER,GstNvTracker))
#define GST_NVTRACKER_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_NVTRACKER,GstNvTrackerClass))
#define GST_IS_NVTRACKER(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_NVTRACKER))
#define GST_IS_NVTRACKER_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_NVTRACKER))


/* Version number of package */
#define VERSION "2.0.0"
#define PACKAGE_DESCRIPTION "Gstreamer plugin to track the objects"
/* Define under which licence the package has been released */
#define PACKAGE_LICENSE "Proprietary"
#define PACKAGE_NAME "GStreamer nvtracker Plugin"
/* Define to the home page for this package. */
#define PACKAGE_URL "http://nvidia.com/"

typedef struct _GstNvTracker      GstNvTracker;
typedef struct _GstNvTrackerClass GstNvTrackerClass;

/** Basic GStreamer class for tracker. */
struct _GstNvTracker
{
  GstBaseTransform parent_instance;

  GstPad *sinkpad, *srcpad;

  /** < private > */
  gboolean running;
  TrackerConfig trackerConfig;
  INvTrackerProc *trackerIface;

  GCond eventCondition;
  GMutex eventLock;

  GThread *output_loop;
};

struct _GstNvTrackerClass
{
    GstBaseTransformClass parent_class;
};

GType gst_nv_tracker_get_type (void);

G_END_DECLS

#endif /* __GST_NVTRACKER_H__ */
