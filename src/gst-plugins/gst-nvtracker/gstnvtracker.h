/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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
