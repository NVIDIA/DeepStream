/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "gstnvvideotestsrc.h"
#include "patterns.h"

#include <gstnvdsbufferpool.h>

#include <glib.h>
#include <glib/gstdio.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <iostream>
#include <random>
#include <sstream>

#define struct_stat struct stat

GST_DEBUG_CATEGORY_STATIC(nv_video_test_src_debug);
#define GST_CAT_DEFAULT nv_video_test_src_debug

#define DEFAULT_WIDTH          (1280)
#define DEFAULT_HEIGHT         (720)
#define DEFAULT_FRAMERATE      (60)
#define DEFAULT_PATTERN        (GST_NV_VIDEO_TEST_SRC_SMPTE)
#define DEFAULT_ANIMATION_MODE (GST_NV_VIDEO_TEST_SRC_FRAMES)
#define DEFAULT_GPU_ID         (0)
#define DEFAULT_MEMTYPE        (NVBUF_MEM_CUDA_DEVICE)
#define DEFAULT_IS_LIVE        (FALSE)
#define DEFAULT_FILE_LOOP       (FALSE)
#define DEFAULT_MAX_JITTER      (0)
#define DEFAULT_FIXED_JITTER    ""
#define DEFAULT_HORIZONTAL_SPEED (0)

enum {
    PROP_0,
    PROP_PATTERN,
    PROP_ANIMATION_MODE,
    PROP_GPU_ID,
    PROP_MEMTYPE,
    PROP_IS_LIVE,
    PROP_LOCATION,
    PROP_FILE_LOOP,
    PROP_MAX_JITTER,
    PROP_FIXED_JITTER,
    PROP_HORIZONTAL_SPEED,
};

static GstStaticPadTemplate gst_nv_video_test_src_template =
    GST_STATIC_PAD_TEMPLATE("src",
        GST_PAD_SRC,
        GST_PAD_ALWAYS,
        GST_STATIC_CAPS(GST_VIDEO_CAPS_MAKE_WITH_FEATURES(
            "memory:NVMM", "{I420, NV12, RGBA, UYVP, BGRA64_LE, RGB10A2_LE}")));

#define parent_class gst_nv_video_test_src_parent_class
G_DEFINE_TYPE(GstNvVideoTestSrc, gst_nv_video_test_src, GST_TYPE_PUSH_SRC)

static void gst_nv_video_test_src_set_pattern(GstNvVideoTestSrc *self,
    int pattern);
static void gst_nv_video_test_src_set_property(GObject *object,
    guint property_id, const GValue *value, GParamSpec *pspec);
static void gst_nv_video_test_src_get_property(
    GObject *object, guint property_id, GValue *value, GParamSpec *pspec);

static gboolean gst_nv_video_test_src_set_caps(GstBaseSrc *bsrc, GstCaps *caps);
static GstCaps* gst_nv_video_test_src_fixate(GstBaseSrc *bsrc, GstCaps *caps);
static gboolean gst_nv_video_test_src_decide_allocation(
    GstBaseSrc *bsrc, GstQuery *query);
static void gst_nv_video_test_src_get_times (GstBaseSrc * basesrc,
    GstBuffer * buffer, GstClockTime * start, GstClockTime * end);
static gboolean gst_nv_video_test_src_start(GstBaseSrc *self);
static gboolean gst_nv_video_test_src_stop(GstBaseSrc *self);

static GstFlowReturn gst_nv_video_test_src_fill(
    GstPushSrc *psrc, GstBuffer *buffer);

#define GST_TYPE_NV_VIDEO_TEST_SRC_PATTERN (gst_nv_video_test_src_pattern_get_type())
static GType
gst_nv_video_test_src_pattern_get_type(void)
{
    static gsize id = 0;
    static const GEnumValue patterns[] =
    {
        {GST_NV_VIDEO_TEST_SRC_SMPTE,      "SMPTE color bars", "smpte"},
        {GST_NV_VIDEO_TEST_SRC_MANDELBROT, "Mandelbrot set",   "mandelbrot"},
        {GST_NV_VIDEO_TEST_SRC_GRADIENT,   "Gradient",         "gradient"},
        {0,                                 NULL,               NULL}
    };

    if (g_once_init_enter(&id))
    {
        GType tmp = g_enum_register_static("GstNvVideoTestSrcPattern", patterns);
        g_once_init_leave(&id, tmp);
    }

    return (GType)id;
}

#define GST_TYPE_NV_VIDEO_TEST_SRC_ANIMATION_MODE (gst_nv_video_test_src_animation_mode_get_type())
static GType
gst_nv_video_test_src_animation_mode_get_type (void)
{
    static gsize id = 0;
    static const GEnumValue modes[] =
    {
        {GST_NV_VIDEO_TEST_SRC_FRAMES,       "Frame count",     "frames"},
        {GST_NV_VIDEO_TEST_SRC_WALL_TIME,    "Wall clock time", "wall-time"},
        {GST_NV_VIDEO_TEST_SRC_RUNNING_TIME, "Running time",    "running-time"},
        {0,                                   NULL,              NULL}
    };

    if (g_once_init_enter(&id))
    {
        GType tmp = g_enum_register_static("GstNvVideoTestSrcAnimationMode", modes);
        g_once_init_leave(&id, tmp);
    }

    return (GType)id;
}

#define GST_TYPE_NVMM_MEMTYPE (gst_nvmm_memtype_get_type())
static GType
gst_nvmm_memtype_get_type(void)
{
    static gsize id = 0;
    static const GEnumValue modes[] =
    {
        {NVBUF_MEM_CUDA_PINNED,  "host",    "Host/Pinned CUDA memory"},
        {NVBUF_MEM_CUDA_DEVICE,  "device",  "Device CUDA memory"},
        {NVBUF_MEM_CUDA_UNIFIED, "unified", "Unified CUDA memory"},
        {0,                       NULL,      NULL}
    };

    if (g_once_init_enter(&id))
    {
        GType tmp = g_enum_register_static("NvBufSurfaceMemType", modes);
        g_once_init_leave(&id, tmp);
    }

    return (GType)id;
}

static void
gst_nv_video_test_src_class_init(GstNvVideoTestSrcClass *klass)
{
    GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
    GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
    GstBaseSrcClass *basesrc_class = GST_BASE_SRC_CLASS(klass);
    GstPushSrcClass *pushsrc_class = GST_PUSH_SRC_CLASS(klass);

    gobject_class->set_property = GST_DEBUG_FUNCPTR(gst_nv_video_test_src_set_property);
    gobject_class->get_property = GST_DEBUG_FUNCPTR(gst_nv_video_test_src_get_property);

    g_object_class_install_property(gobject_class, PROP_PATTERN,
        g_param_spec_enum (
            "pattern", "Pattern",
            "Type of test pattern to generate",
            GST_TYPE_NV_VIDEO_TEST_SRC_PATTERN, DEFAULT_PATTERN,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(gobject_class, PROP_ANIMATION_MODE,
        g_param_spec_enum (
            "animation-mode", "Animation mode",
            "For animating patterns, the counter that controls the animation.",
            GST_TYPE_NV_VIDEO_TEST_SRC_ANIMATION_MODE, DEFAULT_ANIMATION_MODE,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(gobject_class, PROP_GPU_ID,
        g_param_spec_uint(
            "gpu-id", "GPU ID",
            "ID of the GPU where the buffers are allocated",
            0, G_MAXINT, DEFAULT_GPU_ID,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(gobject_class, PROP_MEMTYPE,
        g_param_spec_enum(
            "memtype", "Memory type",
            "Type of the memory used for the allocated buffers",
            GST_TYPE_NVMM_MEMTYPE, DEFAULT_MEMTYPE,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_IS_LIVE,
      g_param_spec_boolean ("is-live", "Is Live",
          "Whether to act as a live source", DEFAULT_IS_LIVE,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_LOCATION,
      g_param_spec_string ("location", "File Location",
          "Location of the file to read", NULL,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
          GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_FILE_LOOP,
      g_param_spec_boolean ("file-loop", "Enable file looping",
          "Loop through the input file infinitely", DEFAULT_FILE_LOOP,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_MAX_JITTER,
      g_param_spec_uint ("max-jitter", "Max jitter to be inserted",
          "Maximun jitter (in milliseconds) to be inserted (only when insert-jitter is enabled)",
           0, G_MAXINT, DEFAULT_MAX_JITTER,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property (gobject_class, PROP_FIXED_JITTER,
        g_param_spec_string ("fixed-jitter", "List of fixed jitter values",
            "List of fixed jitter values in milliseconds\n"
            "\t\t\tThe list will be looped through for the entire sequence. \n"
            "\t\t\t e.g. 30;40;100;0;0",
            DEFAULT_FIXED_JITTER,
            (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property (gobject_class, PROP_HORIZONTAL_SPEED,
        g_param_spec_int ("horizontal-speed", "Horizontal Speed",
            "Horizontal speed of the SMPTE pattern in pixels per frame, "
            "positive values move to the left, negative to the right",
            G_MININT, G_MAXINT, DEFAULT_HORIZONTAL_SPEED,
            (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    gst_element_class_set_static_metadata(element_class,
        GST_PACKAGE_NAME, "Source/Video",
        GST_PACKAGE_DESCRIPTION, GST_PACKAGE_AUTHOR);

    gst_element_class_add_pad_template(element_class,
        gst_static_pad_template_get(&gst_nv_video_test_src_template));

    basesrc_class->set_caps = GST_DEBUG_FUNCPTR(gst_nv_video_test_src_set_caps);
    basesrc_class->fixate = GST_DEBUG_FUNCPTR(gst_nv_video_test_src_fixate);
    basesrc_class->get_times = GST_DEBUG_FUNCPTR(gst_nv_video_test_src_get_times);
    basesrc_class->decide_allocation = GST_DEBUG_FUNCPTR(gst_nv_video_test_src_decide_allocation);
    basesrc_class->start = GST_DEBUG_FUNCPTR(gst_nv_video_test_src_start);
    basesrc_class->stop = GST_DEBUG_FUNCPTR(gst_nv_video_test_src_stop);

    pushsrc_class->fill = GST_DEBUG_FUNCPTR(gst_nv_video_test_src_fill);
}

class Jitter_Vector {
public:
    std::vector<guint64> list;
    int current_position = 0;
    int get_current_value()
    {
        int current_value = list[current_position];
        current_position++;
        if(current_position >= (int)list.size())
            current_position = 0;
        return current_value;
    }
};

static void
gst_nv_video_test_src_init(GstNvVideoTestSrc *self)
{
    gst_nv_video_test_src_set_pattern(self, DEFAULT_PATTERN);

    self->animation_mode = DEFAULT_ANIMATION_MODE;
    self->gpu_id         = DEFAULT_GPU_ID;
    self->memtype        = DEFAULT_MEMTYPE;
    self->horizontal_speed = DEFAULT_HORIZONTAL_SPEED;

    self->filename = NULL;
    self->file_handle = NULL;
    self->file_loop = DEFAULT_FILE_LOOP;

    self->max_jitter = DEFAULT_MAX_JITTER;
    self->p_fixed_jitter_list = (void *)new Jitter_Vector;

    gst_base_src_set_format(GST_BASE_SRC_CAST(self), GST_FORMAT_TIME);
    gst_base_src_set_live(GST_BASE_SRC_CAST(self), DEFAULT_IS_LIVE);
}

static void
gst_nv_video_test_src_set_pattern(GstNvVideoTestSrc *self, int pattern)
{
    self->pattern = (GstNvVideoTestSrcPattern) pattern;

    switch (pattern)
    {
        case GST_NV_VIDEO_TEST_SRC_SMPTE:
            self->cuda_fill_image = gst_nv_video_test_src_smpte;
            break;

        case GST_NV_VIDEO_TEST_SRC_MANDELBROT:
            self->cuda_fill_image = gst_nv_video_test_src_mandelbrot;
            break;

        case GST_NV_VIDEO_TEST_SRC_GRADIENT:
            self->cuda_fill_image = gst_nv_video_test_src_gradient;
            break;

        default:
            g_assert_not_reached();
    }
}

static gboolean
gst_nv_video_file_src_set_location (GstNvVideoTestSrc * src, const gchar * location)
{
  GstState state;

  /* the element must be stopped in order to do this */
  GST_OBJECT_LOCK (src);
  state = GST_STATE (src);
  if (state != GST_STATE_READY && state != GST_STATE_NULL)
  {
    GST_WARNING_OBJECT (src, "Pipeline is not in NULL or READY state. Can't set file location.");
    GST_OBJECT_UNLOCK (src);
    return FALSE;
  }
  GST_OBJECT_UNLOCK (src);

  if(src->filename)
  {
    g_free (src->filename);
    src->filename = NULL;
  }

  src->filename = g_strdup (location);
  GST_INFO ("filename : %s", src->filename);
  g_object_notify (G_OBJECT (src), "location");

  return TRUE;

}

static void
gst_nv_video_test_src_set_property(GObject *object, guint property_id,
    const GValue *value, GParamSpec *pspec)
{
    GstNvVideoTestSrc *self = GST_NV_VIDEO_TEST_SRC(object);

    switch (property_id)
    {
        case PROP_PATTERN:
            gst_nv_video_test_src_set_pattern(self, g_value_get_enum(value));
            break;

        case PROP_ANIMATION_MODE:
            self->animation_mode = (GstNvVideoTestSrcAnimationMode)g_value_get_enum(value);
            break;

        case PROP_GPU_ID:
            self->gpu_id = g_value_get_uint(value);
            break;

        case PROP_MEMTYPE:
            self->memtype = (NvBufSurfaceMemType)g_value_get_enum(value);
            break;
        case PROP_IS_LIVE:
            gst_base_src_set_live (GST_BASE_SRC (self), g_value_get_boolean (value));
            break;
        case PROP_LOCATION:
            gst_nv_video_file_src_set_location (self, g_value_get_string (value));
            break;
        case PROP_FILE_LOOP:
            self->file_loop = g_value_get_boolean (value);
            break;
        case PROP_MAX_JITTER:
            self->max_jitter = g_value_get_uint (value);
            break;
        case PROP_FIXED_JITTER:
        {
            std::stringstream str(g_value_get_string(value)? g_value_get_string(value) : "");
            Jitter_Vector *jitter_vector = (Jitter_Vector *)self->p_fixed_jitter_list;
            jitter_vector->list.clear();
            while(str.peek() != EOF) {
                gint jitter_value;
                str >> jitter_value;
                jitter_vector->list.push_back(jitter_value);
                str.get();
                }
        }
        break;
        case PROP_HORIZONTAL_SPEED:
            self->horizontal_speed = g_value_get_int (value);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, pspec);
            break;
    }
}

static void
gst_nv_video_test_src_get_property(GObject *object, guint property_id,
    GValue *value, GParamSpec *pspec)
{
    GstNvVideoTestSrc *self = GST_NV_VIDEO_TEST_SRC(object);

    switch (property_id)
    {
        case PROP_PATTERN:
            g_value_set_enum(value, self->pattern);
            break;

        case PROP_ANIMATION_MODE:
            g_value_set_enum(value, self->animation_mode);
            break;

        case PROP_GPU_ID:
            g_value_set_uint(value, self->gpu_id);
            break;

        case PROP_MEMTYPE:
            g_value_set_enum(value, self->memtype);
            break;

        case PROP_IS_LIVE:
            g_value_set_boolean (value, gst_base_src_is_live (GST_BASE_SRC (self)));
            break;

        case PROP_LOCATION:
            g_value_set_string (value, self->filename);
            break;

        case PROP_FILE_LOOP:
            g_value_set_boolean (value, self->file_loop);
            break;

        case PROP_MAX_JITTER:
            g_value_set_uint(value, self->max_jitter);
            break;

        case PROP_FIXED_JITTER:
        {
            std::stringstream str;
            Jitter_Vector *jitter_vector = (Jitter_Vector *)self->p_fixed_jitter_list;
            for(const auto value : jitter_vector->list)
                str << value << ";";
            g_value_set_string (value, str.str ().c_str ());
        }
        break;

        case PROP_HORIZONTAL_SPEED:
            g_value_set_int (value, self->horizontal_speed);
            break;

        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, pspec);
            break;
    }
}

static gboolean
gst_nv_video_test_src_set_caps(GstBaseSrc *bsrc, GstCaps *caps)
{
    GstNvVideoTestSrc *self = GST_NV_VIDEO_TEST_SRC(bsrc);

    GST_OBJECT_LOCK(self);

    if (!gst_video_info_from_caps(&self->info, caps))
        return FALSE;

    self->caps = caps;

    gst_nv_video_test_src_cuda_init(self);

    GST_OBJECT_UNLOCK(self);

    return TRUE;
}

static GstCaps*
gst_nv_video_test_src_fixate(GstBaseSrc *bsrc, GstCaps *caps)
{
    GstStructure *structure;

    caps = gst_caps_make_writable(caps);
    structure = gst_caps_get_structure(caps, 0);

    gst_structure_fixate_field_nearest_int(structure, "width", DEFAULT_WIDTH);
    gst_structure_fixate_field_nearest_int(structure, "height", DEFAULT_HEIGHT);

    if (gst_structure_has_field(structure, "framerate"))
    {
        gst_structure_fixate_field_nearest_fraction(structure,
            "framerate", DEFAULT_FRAMERATE, 1);
    }
    else
    {
        gst_structure_set(structure, "framerate",
            GST_TYPE_FRACTION, DEFAULT_FRAMERATE, 1, NULL);
    }

    caps = GST_BASE_SRC_CLASS(parent_class)->fixate(bsrc, caps);

    return caps;
}

static gboolean
gst_nv_video_test_src_decide_allocation(GstBaseSrc *bsrc, GstQuery *query)
{
    GstNvVideoTestSrc *self = GST_NV_VIDEO_TEST_SRC(bsrc);

    // Remove any downstream-proposed allocation pools and params, since these
    // may not meet the requirements of our pool (e.g. memory type and GPU ID).
    while (gst_query_get_n_allocation_pools(query) > 0)
        gst_query_remove_nth_allocation_pool(query, 0);
    while (gst_query_get_n_allocation_params(query) > 0)
        gst_query_remove_nth_allocation_param(query, 0);

    // Allocate and configure a new GstNvDsBufferPool.
    GstBufferPool *pool = gst_nvds_buffer_pool_new();
    GstStructure *config = gst_buffer_pool_get_config(pool);

    // Note that the size of each GstBuffer allocated by the GstNvmmBufferPool
    // is equal to the size of the NvBufSurface struct. The GstBuffer memory
    // simply contains the NvBufSurface descriptor which in turn points to the
    // actual NVMM GPU buffer allocation(s).
    gst_buffer_pool_config_set_params(config, self->caps,
        sizeof(NvBufSurface), 0, 0);

    // Configure the GstNvDsBufferPool allocator.
    gst_structure_set(config,
        "memtype", G_TYPE_UINT, self->memtype,
        "gpu-id", G_TYPE_UINT, self->gpu_id, NULL);

    if (!gst_buffer_pool_set_config(pool, config))
    {
        GST_ERROR("Failed to set buffer pool config");
        return FALSE;
    }

    gst_query_add_allocation_pool(query, pool, sizeof(NvBufSurface), 0, 0);

    gst_object_unref(pool);

    return GST_BASE_SRC_CLASS(parent_class)->decide_allocation(bsrc, query);
}

static float get_random_jitter(guint max_jitter)
{
    float jitter;
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::normal_distribution<float> distribution((max_jitter>>1), (max_jitter>>2));
    while (true) {
        jitter = distribution(generator);
        if (jitter >= 0 && jitter <= (float)max_jitter)
            return jitter;
    }
}

static void
gst_nv_video_test_src_get_times (GstBaseSrc * basesrc, GstBuffer * buffer,
    GstClockTime * start, GstClockTime * end)
{
    GstNvVideoTestSrc *self = GST_NV_VIDEO_TEST_SRC(basesrc);

  /* for live sources, sync on the timestamp of the buffer */
  if (gst_base_src_is_live (basesrc)) {
    GstClockTime timestamp = GST_BUFFER_PTS (buffer);

    if (GST_CLOCK_TIME_IS_VALID (timestamp)) {
      /* get duration to calculate end time */
      GstClockTime duration = GST_BUFFER_DURATION (buffer);

      // Introduce a jitter
      Jitter_Vector *jitter_vector = (Jitter_Vector *)self->p_fixed_jitter_list;
      bool insert_jitter = (jitter_vector->list.size() > 0) || (self->max_jitter > 0);
      if(insert_jitter && (timestamp != 0))
      {

        float jitter;
        // If fixed jitter list is provided, use it
        // Else introduce random jitter
        if(jitter_vector->list.size() > 0)
        {
            jitter = jitter_vector->get_current_value();
        }
        else
        {
            jitter = get_random_jitter(self->max_jitter);
        }
        GST_DEBUG_OBJECT (self, "Buffer PTS(ms) = %lu Jitter introduced(ms) = %f", (timestamp/1000000), jitter);
        timestamp = timestamp + (GstClockTime)(jitter*1000000);
        // Make sure the buffers are in order
        if(timestamp < self->last_buffer_start_timestamp)
            timestamp = self->last_buffer_start_timestamp;
      }
      if (GST_CLOCK_TIME_IS_VALID (duration)) {
        *end = timestamp + duration;
      }
      *start = timestamp;
      self->last_buffer_start_timestamp = timestamp;
    }
  } else {
    *start = -1;
    *end = -1;
  }
}

/* open the file, necessary to go to READY state */
static gboolean
gst_nv_video_file_src_start (GstNvVideoTestSrc *src)
{

  if (src->filename == NULL || src->filename[0] == '\0')
    goto no_filename;

  GST_INFO_OBJECT (src, "opening file %s", src->filename);

  /* open the file */
  src->file_handle = fopen(src->filename, "rb");

  if (!src->file_handle)
    goto open_failed;

  {
    struct_stat file_stat;

    stat(src->filename, &file_stat);

    if (!S_ISREG (file_stat.st_mode))
      goto not_regular;

  }


  src->read_position = 0;

  return TRUE;

  /* ERROR */
no_filename:
  {
    GST_ERROR_OBJECT (src, "No file name specified for reading.");
    goto error_exit;
  }
open_failed:
  {
    GST_ERROR_OBJECT (src, "Could not open file %s for reading.", src->filename);
    goto error_exit;
  }
not_regular:
  {
    GST_ERROR_OBJECT (src, "%s is not a regular file.", src->filename);
    goto error_exit;
  }

error_exit:
  if(src->file_handle)
    fclose (src->file_handle);
  src->file_handle = NULL;
  return FALSE;
}

static gboolean
gst_nv_video_test_src_start(GstBaseSrc *bsrc)
{
    GstNvVideoTestSrc *self = GST_NV_VIDEO_TEST_SRC(bsrc);

    GST_OBJECT_LOCK(self);

    gst_video_info_init(&self->info);

    if(self->filename)
        gst_nv_video_file_src_start(self);

    self->running_time = 0;
    self->filled_frames = 0;

    GST_OBJECT_UNLOCK(self);

    return TRUE;
}

static gboolean
gst_nv_video_test_src_stop(GstBaseSrc *bsrc)
{
    GstNvVideoTestSrc *self = GST_NV_VIDEO_TEST_SRC(bsrc);

    GST_OBJECT_LOCK(self);

    gst_nv_video_test_src_cuda_free(self);

    if(self->filename)
    {
        g_free (self->filename);
        self->filename = NULL;
    }
    if(self->file_handle)
    {
        fclose (self->file_handle);
        self->file_handle = NULL;
    }
    if (self->file_read_surface) {
        NvBufSurfaceDestroy (self->file_read_surface);
        self->file_read_surface = NULL;
    }

    if(self->p_fixed_jitter_list)
    {
        delete (Jitter_Vector *)self->p_fixed_jitter_list;
        self->p_fixed_jitter_list = NULL;
    }

    GST_OBJECT_UNLOCK(self);

    return TRUE;
}

static GstFlowReturn
gst_nv_video_file_src_fill (GstNvVideoTestSrc *src)
{
  guint to_read = 0;
  int ret;
  NvBufSurface *surf = src->file_read_surface;
  NvBufSurfaceParams *params = surf->surfaceList;
  guint8 *data =  (guint8*)params->dataPtr;
  switch (surf->surfaceList[0].colorFormat){
      case NVBUF_COLOR_FORMAT_RGBA:
      to_read = 4*params->width*params->height;
      break;
      case NVBUF_COLOR_FORMAT_YUV420:
      case NVBUF_COLOR_FORMAT_YUV420_709:
      to_read = (3*params->width*params->height)/2;
      break;
      case NVBUF_COLOR_FORMAT_NV12:
      case NVBUF_COLOR_FORMAT_NV12_709:
      to_read = (3*params->width*params->height)/2;
      break;
      case NVBUF_COLOR_FORMAT_UYVP:
      case NVBUF_COLOR_FORMAT_UYVP_ER:
      // UYVP is 10-bit 4:2:2, packed format
      // Each group of 2 pixels (U|Y, V|Y) takes 5 bytes (10 bits per component)
      to_read = (params->width * params->height * 5) / 2;
      break;
      case NVBUF_COLOR_FORMAT_BGRA64_LE:
      // BGRA64_LE is 64 bits per pixel (16 bits per component)
      to_read = 8 * params->width * params->height;
      break;
      case NVBUF_COLOR_FORMAT_RGBA_10_10_10_2_709:
      case NVBUF_COLOR_FORMAT_RGBA_10_10_10_2_2020:
      // RGB10A2_LE is 32 bits per pixel (10 bits for R,G,B and 2 bits for A)
      to_read = 4 * params->width * params->height;
      break;
      default:
      GST_ERROR_OBJECT (src, "ColorFormat %d not supported", params->colorFormat);
      return GST_FLOW_ERROR;
      break;
  }

  ret = fread(data, 1, to_read, src->file_handle);
  if (G_UNLIKELY (ret < 0)) {
    GST_ELEMENT_ERROR (src, RESOURCE, READ, (NULL), GST_ERROR_SYSTEM);
    return GST_FLOW_ERROR;
  }

  /* files should eos if they read 0 and more was requested */
  if (G_UNLIKELY (ret == 0))
  {
    // If file-loop is enabled, loop through the file
    if(src->file_loop)
    {
        ret = fseek (src->file_handle, 0, SEEK_SET);
        if (G_UNLIKELY (ret != 0))
        {
            GST_ELEMENT_ERROR (src, RESOURCE, READ, (NULL), GST_ERROR_SYSTEM);
            return GST_FLOW_ERROR;
        }
        src->read_position = 0;
        ret = fread(data, 1, to_read, src->file_handle);
        if (G_UNLIKELY (ret < 0)) {
            GST_ELEMENT_ERROR (src, RESOURCE, READ, (NULL), GST_ERROR_SYSTEM);
            return GST_FLOW_ERROR;
        }
    }
    else
    {
        GST_DEBUG ("EOS");
        return GST_FLOW_EOS;
    }
  }

  src->read_position += ret;

  return GST_FLOW_OK;

}

static GstFlowReturn
gst_nv_video_test_src_fill(GstPushSrc *psrc, GstBuffer *buffer)
{
    GstNvVideoTestSrc *self = GST_NV_VIDEO_TEST_SRC(psrc);
    GstMapInfo map = GST_MAP_INFO_INIT;

    // If 0 framerate and we've returned a frame, EOS.
    if (G_UNLIKELY(self->info.fps_n == 0 && self->filled_frames == 1))
        return GST_FLOW_EOS;

    gst_buffer_map(buffer, &map, GST_MAP_READWRITE);

    // The memory of a GstBuffer allocated by an NvDsBufferPool contains
    // the NvBufSurface descriptor which then describes the actual GPU
    // buffer allocation(s) in its surfaceList.
    NvBufSurface *surf = (NvBufSurface*)map.data;

    if(self->file_handle)
    {
        int status = 0;
        // Crate a file read surface if it doesn't exist
        if(self->file_read_surface == NULL)
        {
            NvBufSurfaceAllocateParams surf_params = {{0}};
            surf_params.params.gpuId = self->gpu_id;
            surf_params.params.width = surf->surfaceList[0].width;
            surf_params.params.height = surf->surfaceList[0].height;
            surf_params.params.colorFormat = surf->surfaceList[0].colorFormat;
            surf_params.params.memType = NVBUF_MEM_SYSTEM;
            surf_params.disablePitchPadding = true;
            status = NvBufSurfaceAllocate (&self->file_read_surface, 1, &surf_params);
            if (status < 0) {
              GST_ERROR_OBJECT (self, "NvBufSurfaceAllocate Failed");
              goto error;
            }
        }
        // Fill the surface
        GstFlowReturn gst_status = gst_nv_video_file_src_fill(self);
        if(gst_status == GST_FLOW_EOS)
            goto eos;
        else if(gst_status == GST_FLOW_ERROR)
            goto error;
        else
            {g_assert(gst_status == GST_FLOW_OK);}

        // Copy to cuda buffer
        status = NvBufSurfaceCopy(self->file_read_surface, surf);
        if (status < 0) {
            GST_ERROR_OBJECT (self, "NvBufSurfaceCopy Failed");
            goto error;
        }
    }
    else
    {
    // Use CUDA to fill the GPU buffer with a test pattern.
    //
    // NOTE: In this source, we currently only fill the GPU buffer using CUDA.
    //       This source could be modified to fill the buffer instead with other
    //       mechanisms such as mapped CPU writes or RDMA transfers:
    //
    //       1) To use mapped CPU writes, the GPU buffer could be mapped into
    //          the CPU address space using NvBufSurfaceMap.
    //
    //       2) To use RDMA transfers from another hardware device, the GPU
    //          address for the buffer(s) (i.e. the `dataPtr` member of the
    //          NvBufSurfaceParams) could be passed here to the device driver
    //          that is responsible for performing the RDMA transfer into the
    //          buffer. Details on how RDMA to NVIDIA GPUs can be performed by
    //          device drivers is provided in a demonstration application
    //          available at https://github.com/NVIDIA/jetson-rdma-picoevb.
    //
    //       For more details on the NvBufSurface API, see nvbufsurface.h
    //
    gst_nv_video_test_src_cuda_prepare(self, surf->surfaceList);
    self->cuda_fill_image(self);
    }

    // Set the numFilled field of the surface with the number of surfaces
    // that have been filled (always 1 in this example).
    // This metadata is required by other downstream DeepStream plugins.
    surf->numFilled = 1;

    gst_buffer_unmap(buffer, &map);

    // Set the buffer timestamps and duration.
    GST_BUFFER_PTS(buffer) = self->running_time;
    GST_BUFFER_DTS(buffer) = GST_CLOCK_TIME_NONE;
    GST_BUFFER_OFFSET(buffer) = self->filled_frames;
    self->filled_frames++;
    GST_BUFFER_OFFSET_END(buffer) = self->filled_frames;

    if (self->info.fps_n != 0)
    {
        GstClockTime next_time = gst_util_uint64_scale(self->filled_frames,
            self->info.fps_d * GST_SECOND, self->info.fps_n);
        GST_BUFFER_DURATION(buffer) = next_time - self->running_time;
        self->running_time = next_time;
    }
    else
    {
        GST_BUFFER_DURATION(buffer) = GST_CLOCK_TIME_NONE;
    }

    gst_object_sync_values(GST_OBJECT(psrc), GST_BUFFER_PTS(buffer));

    return GST_FLOW_OK;

error:
 {
    gst_buffer_unmap(buffer, &map);
    gst_buffer_resize (buffer, 0, 0);
    return GST_FLOW_ERROR;
 }
eos:
 {
    gst_buffer_unmap(buffer, &map);
    gst_buffer_resize (buffer, 0, 0);
    return GST_FLOW_EOS;
 }
}

static gboolean
plugin_init(GstPlugin *plugin)
{
    GST_DEBUG_CATEGORY_INIT(nv_video_test_src_debug, "nvvideotestsrc", 0,
        GST_PACKAGE_NAME);

    return gst_element_register(plugin, "nvvideotestsrc", GST_RANK_PRIMARY,
        GST_TYPE_NV_VIDEO_TEST_SRC);
}

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR,
                  nvdsgst_videotestsrc, GST_PACKAGE_DESCRIPTION, plugin_init,
                  GST_PACKAGE_VERSION, GST_PACKAGE_LICENSE,
                  GST_PACKAGE_NAME, GST_PACKAGE_ORIGIN)
