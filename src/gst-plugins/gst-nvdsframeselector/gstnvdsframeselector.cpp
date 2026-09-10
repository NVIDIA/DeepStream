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

#include <string.h>
#include <string>
#include <sstream>
#include <iostream>
#include <ostream>
#include <fstream>
#include <sys/time.h>
#include <dlfcn.h>
#include <memory>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <utility>

#include "gstnvdsbufferpool.h"
#include "gstnvdsframeselector.h"
#include "gstnvdsmeta.h"

#include "nvbufsurface.h"

#include "nvtx_helper.h"
#include <cuda_runtime.h>
#include "motion_detector.h"

/* Optical flow interval will be a property now */

/* External SAD function declarations */
#ifdef __cplusplus
extern "C" {
#endif

/* CUDA SAD initialization and cleanup functions */
int initSAD_CUDA(cudaStream_t* stream,
                 unsigned int** d_partial,
                 unsigned int** h_partial,
                 int max_width, int max_height);

void cleanupSAD_CUDA(cudaStream_t stream,
                     unsigned int* d_partial,
                     unsigned int* h_partial);

/* CUDA SAD computation function implemented in sad_kernel.cu */
unsigned int runSAD_CUDA(const unsigned char* frameA,
                         const unsigned char* frameB,
                         int width, int height,
                         cudaStream_t stream,
                         unsigned int* d_partial,
                         unsigned int* h_partial,
                         int max_blocks);

/* CPU SAD computation function implemented in sad_cpu.cpp */
unsigned int runSAD_CPU(const unsigned char* frameA,
						const unsigned char* frameB,
						int width, int height);


#ifdef __cplusplus
}
#endif

/* Helper function to select appropriate SAD implementation based on environment variable */
static unsigned int runSAD_adaptive(GstNvDsFrameSelector *nvdsframeselector,
									const unsigned char* frameA,
									const unsigned char* frameB,
									int width, int height,
									bool is_device_memory = false) {
	/* Check environment variable NVDS_FRAME_SELECTOR_USE_SAD_CPU */
	/* Default to CUDA, use CPU if flag is set to "1" */
	const char* use_cpu_env = getenv("NVDS_FRAME_SELECTOR_USE_SAD_CPU");
	bool use_cpu = false;  // Default to CUDA

	if (use_cpu_env && strcmp(use_cpu_env, "1") == 0) {
		use_cpu = true;  // Use CPU if explicitly set to "1"
	}

	if (use_cpu) {
		/* If data is in device memory, copy to host first */
		if (is_device_memory) {

			int totalPixels = width * height;
			size_t bytes = totalPixels * sizeof(unsigned char);

			/* Allocate host memory */
			unsigned char* hostFrameA = (unsigned char*)malloc(bytes);
			unsigned char* hostFrameB = (unsigned char*)malloc(bytes);

			if (!hostFrameA || !hostFrameB) {
				GST_ERROR_OBJECT (nvdsframeselector, "Error: Failed to allocate host memory for CPU SAD computation");
				if (hostFrameA) free(hostFrameA);
				if (hostFrameB) free(hostFrameB);
				return 0;
			}

			/* Copy data from device to host */
			cudaError_t copyResult1 = cudaMemcpy(hostFrameA, frameA, bytes, cudaMemcpyDeviceToHost);
			cudaError_t copyResult2 = cudaMemcpy(hostFrameB, frameB, bytes, cudaMemcpyDeviceToHost);

			if (copyResult1 != cudaSuccess || copyResult2 != cudaSuccess) {
				GST_ERROR_OBJECT (nvdsframeselector, "Error: cudaMemcpy failed - frameA: %s, frameB: %s",
					   cudaGetErrorString(copyResult1), cudaGetErrorString(copyResult2));
				free(hostFrameA);
				free(hostFrameB);
				return 0;
			}
			/* Perform CPU SAD computation on host data */
			unsigned int sad_result = runSAD_CPU(hostFrameA, hostFrameB, width, height);

			/* Clean up host memory */
			free(hostFrameA);
			free(hostFrameB);
			return sad_result;
		} else {
			/* Data is already in host memory, use directly */
			return runSAD_CPU(frameA, frameB, width, height);
		}
	} else {
		/* Use CUDA SAD computation with pre-allocated resources */
		if (!nvdsframeselector->cuda_stream || !nvdsframeselector->d_partial_sums || 
			!nvdsframeselector->h_partial_sums) {
			GST_ERROR_OBJECT (nvdsframeselector, "Error: CUDA resources not initialized");
			return 0;
		}
		return runSAD_CUDA(frameA, frameB, width, height,
		                   nvdsframeselector->cuda_stream,
		                   nvdsframeselector->d_partial_sums,
		                   nvdsframeselector->h_partial_sums,
		                   nvdsframeselector->max_blocks);
	}
}


GST_DEBUG_CATEGORY_STATIC (gst_nvds_frame_selector_debug);
#define GST_CAT_DEFAULT gst_nvds_frame_selector_debug

static GQuark _dsmeta_quark = 0;

/* Enum to identify properties */
enum
{
	PROP_0,
	PROP_GPU_DEVICE_ID,
	PROP_ENABLE_SELECTOR,
	PROP_CACHE_SIZE,
	PROP_SELECTION_COUNT,
	PROP_PERCENTAGE_THRESHOLD,
	PROP_AVERAGING_FRAMES,
	PROP_EXCLUSION_RANGE,
	PROP_FRAME_SELECTION_ALGORITHM,
	PROP_BYPASS_MODE,
	PROP_OPTICAL_FLOW_INTERVAL,
	PROP_ENABLE_MOTION_DETECTION,
	PROP_SOURCE_FPS,
	PROP_OF_TIGHTEN,
	PROP_EQUIDISTANT_OUTPUT
};

/* Default values for properties */
#define DEFAULT_GPU_ID             0
#define DEFAULT_ENABLE_SELECTOR    TRUE
#define DEFAULT_CACHE_SIZE         100
#define DEFAULT_SELECTION_COUNT    10
#define DEFAULT_PERCENTAGE_THRESHOLD 0.0
#define DEFAULT_AVERAGING_FRAMES   5
#define DEFAULT_EXCLUSION_RANGE    0
#define DEFAULT_BYPASS_MODE        FALSE
#define DEFAULT_OPTICAL_FLOW_INTERVAL 0   // 0 = AUTO
                                          // OF gap stays ~0.5s (threshold-calibration point) at any
                                          // input fps (2..60). Set >=1 to force an explicit stride.
#define DEFAULT_EQUIDISTANT_OUTPUT    FALSE

/* Frames to emit (equidistant) for a diffuse-motion chunk: low optical-flow
 * velocity and no distinct peaks, but not provably empty. Capped by selection_count. */
#define DIFFUSE_MOTION_FRAME_FLOOR 3

/* OF+SAD optimistic-blend selection (deployment: cache-size=20 equidistant frames
 * from a 10s chunk, optical-flow-interval=1 => 19 consecutive pairs 0.5s apart).
 * For each consecutive pair we mark the transition "active" if OF, SAD, OR moving-
 * cells indicates motion (optimistic union => maximize event recall). Output is
 * DYNAMIC: a fully-static window yields 1 frame; otherwise EVERY active frame is
 * emitted, with the only ceiling being the configurable `selection-count` budget
 * (no hardcoded min/max frame count). */
#define OF_SAD_MC_ABS_FLOOR     150          /* moving-cell noise floor (1080p-referenced). A chunk whose
                                              * peak moving-cells stays below this is treated as static.
                                              * 150 is recall-first: faint distant events (e.g. a person's
                                              * gate APPROACH measures ~215 moving-cells) sit in the same band
                                              * as compression noise (~200-300), so a higher floor clips them.
                                              * Resolution-scaled at runtime via OF_REF_GRID_CELLS; override
                                              * per-deployment with env NVDS_FSELECT_OF_FLOOR. */

/* ---- OF-ONLY selection (now the DEFAULT path) ----
 * Default ON; set env NVDS_FSELECT_OF_ONLY=0 to fall back to the SAD+OF blend.
 * Count follows ABSOLUTE optical-flow motion (moving-cells / largest-blob), frames are
 * the most-active ones. See gst_nvds_frame_selection_of_only(). */
#define OF_ONLY_MC_FULL     3000   /* moving-cells at which a chunk earns ALL selection_count frames.
                                    * Env: NVDS_FSELECT_OF_MC_FULL. */
#define OF_ONLY_MIN_ACTIVE  5      /* min frames for any active chunk (peak mc >= OF_SAD_MC_ABS_FLOOR).
                                    * Env: NVDS_FSELECT_OF_MIN. */
#define OF_ONLY_TIGHTEN     0.5    /* tighten factor: shrink the motion-scaled count (e.g. 20->10, 12->6)
                                    * before the MIN clamp. Since selection is motion-ranked, the dropped
                                    * frames are the weakest/most-redundant; the MIN floor protects sparse
                                    * events. Env: NVDS_FSELECT_OF_TIGHTEN. */
#define OF_ONLY_TIGHTEN_EXTREME 0.8 /* for EXTREME-motion chunks (peak OF level >= MOTION_VERY_HIGH) don't
                                     * tighten so hard — relax the fraction up to this so busy real events
                                     * keep more frames. Env: NVDS_FSELECT_OF_TIGHTEN_EXTREME. */

/* Moving-cell COUNT thresholds (OF_SAD_MC_ABS_FLOOR, OF_ONLY_MC_FULL, and the env
 * MC_FULL) are expressed in units of the 1080p reference OF grid (480x270 @ grid-size 4
 * = 129600 cells). At runtime they are scaled by (actual_grid_cells / this reference)
 * so the same "fraction of the frame is moving" holds at any resolution. */
#define OF_REF_GRID_CELLS   129600

/* Frame selection algorithm types */
#define FRAME_SELECTION_ALGO_BASIC            0   /* equidistant */
#define FRAME_SELECTION_ALGO_RANGE_BASED      1   /* SAD / range-based */
#define FRAME_SELECTION_ALGO_OF               2   /* optical-flow motion-based (default) */
#define DEFAULT_FRAME_SELECTION_ALGO          FRAME_SELECTION_ALGO_OF

GType gst_nvds_frame_selection_algorithm_get_type (void);
GType gst_nvds_frame_selection_algorithm_get_type (void)
{
  static GType qtype2 = 0;
  if (qtype2 == 0) {
	  static const GEnumValue frame_selection_algorithms[] = {
		{ FRAME_SELECTION_ALGO_BASIC, "Basic frame selection (equidistant)", "BASIC"},
		{ FRAME_SELECTION_ALGO_RANGE_BASED, "Range-based selection with exclusion (SAD)", "RANGE_BASED"},
		{ FRAME_SELECTION_ALGO_OF, "Optical-flow motion-based selection", "OF"},
		{0, NULL, NULL}
	  };
	qtype2 = g_enum_register_static ("GstNvDsFrameSelectionAlgorithm", frame_selection_algorithms);
  }
  return qtype2;
}
#define GST_TYPE_NVDS_FRAME_SELECTION_ALGORITHM (gst_nvds_frame_selection_algorithm_get_type ())

/* By default NVIDIA Hardware allocated memory flows through the pipeline. We
 * will be processing on this type of memory only. */
#define GST_CAPS_FEATURE_MEMORY_NVMM "memory:NVMM"
static GstStaticPadTemplate gst_nvds_frame_selector_sink_template =
	GST_STATIC_PAD_TEMPLATE("sink",
							GST_PAD_SINK,
							GST_PAD_ALWAYS,
							GST_STATIC_CAPS(GST_VIDEO_CAPS_MAKE_WITH_FEATURES(
							"memory:NVMM",
							"{ NV12 }")));

static GstStaticPadTemplate gst_nvds_frame_selector_src_template =
	GST_STATIC_PAD_TEMPLATE("src",
							GST_PAD_SRC,
							GST_PAD_ALWAYS,
							GST_STATIC_CAPS(GST_VIDEO_CAPS_MAKE_WITH_FEATURES(
							"memory:NVMM",
							"{ NV12 }")));

/* Define our element type. Standard GObject/GStreamer boilerplate stuff */
#define gst_nvds_frame_selector_parent_class parent_class
G_DEFINE_TYPE (GstNvDsFrameSelector, gst_nvds_frame_selector, GST_TYPE_BASE_TRANSFORM);

static void gst_nvds_frame_selector_set_property (GObject * object, guint prop_id,
	const GValue * value, GParamSpec * pspec);
static void gst_nvds_frame_selector_get_property (GObject * object, guint prop_id,
	GValue * value, GParamSpec * pspec);
static void gst_nvds_frame_selector_finalize (GObject * object);

static gboolean gst_nvds_frame_selector_transform_size(GstBaseTransform* btrans,
		GstPadDirection dir, GstCaps *caps, gsize size, GstCaps* othercaps, gsize* othersize);

static gboolean gst_nvds_frame_selector_accept_caps (GstBaseTransform * btrans,
	GstPadDirection direction, GstCaps * caps);

static GstCaps* gst_nvds_frame_selector_fixate_caps(GstBaseTransform* btrans,
		GstPadDirection direction, GstCaps* caps, GstCaps* othercaps);

static gboolean gst_nvds_frame_selector_set_caps (GstBaseTransform * btrans,
	GstCaps * incaps, GstCaps * outcaps);

static GstCaps* gst_nvds_frame_selector_transform_caps(GstBaseTransform* btrans, GstPadDirection dir,
	GstCaps* caps, GstCaps* filter);

static gboolean gst_nvds_frame_selector_start (GstBaseTransform * btrans);
static gboolean gst_nvds_frame_selector_stop (GstBaseTransform * btrans);

static GstFlowReturn gst_nvds_frame_selector_transform_ip(GstBaseTransform* btrans,
	GstBuffer* inbuf);

static gboolean gst_nvds_frame_selector_sink_event (GstBaseTransform * btrans, GstEvent * event);

/* Install properties, set sink and src pad capabilities, override the required
 * functions of the base class, These are common to all instances of the
 * element.
 */
static void
gst_nvds_frame_selector_class_init (GstNvDsFrameSelectorClass * klass)
{
	GObjectClass *gobject_class;
	GstElementClass *gstelement_class;
	GstBaseTransformClass *gstbasetransform_class;
	gobject_class = (GObjectClass *) klass;
	gstelement_class = (GstElementClass *) klass;
	gstbasetransform_class = (GstBaseTransformClass *) klass;

	/* Overide base class functions */
	gobject_class->set_property = GST_DEBUG_FUNCPTR (gst_nvds_frame_selector_set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR (gst_nvds_frame_selector_get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR (gst_nvds_frame_selector_finalize);

	gstbasetransform_class->transform_size = GST_DEBUG_FUNCPTR(gst_nvds_frame_selector_transform_size);
	gstbasetransform_class->accept_caps = GST_DEBUG_FUNCPTR(gst_nvds_frame_selector_accept_caps);
	gstbasetransform_class->fixate_caps = GST_DEBUG_FUNCPTR(gst_nvds_frame_selector_fixate_caps);
	gstbasetransform_class->set_caps = GST_DEBUG_FUNCPTR (gst_nvds_frame_selector_set_caps);
	gstbasetransform_class->transform_caps = GST_DEBUG_FUNCPTR(gst_nvds_frame_selector_transform_caps);

	gstbasetransform_class->start = GST_DEBUG_FUNCPTR (gst_nvds_frame_selector_start);
	gstbasetransform_class->stop = GST_DEBUG_FUNCPTR (gst_nvds_frame_selector_stop);

	gstbasetransform_class->transform_ip = GST_DEBUG_FUNCPTR (gst_nvds_frame_selector_transform_ip);
	gstbasetransform_class->sink_event = GST_DEBUG_FUNCPTR (gst_nvds_frame_selector_sink_event);

	/* Install properties */
	g_object_class_install_property (gobject_class, PROP_GPU_DEVICE_ID,
		g_param_spec_uint ("gpu-id",
			"Set GPU Device ID",
			"Set GPU Device ID", 0,
			G_MAXUINT, 0,
			GParamFlags
			(G_PARAM_READWRITE |
				G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

	g_object_class_install_property (gobject_class, PROP_ENABLE_SELECTOR,
		g_param_spec_boolean ("enable",
			"Enable Frame Selector",
			"Enable or disable frame selection", DEFAULT_ENABLE_SELECTOR,
			GParamFlags
			(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

	g_object_class_install_property (gobject_class, PROP_CACHE_SIZE,
		g_param_spec_uint ("cache-size",
			"Cache Size",
			"Number of frames to cache before selection", 1,
			G_MAXUINT, DEFAULT_CACHE_SIZE,
			GParamFlags
			(G_PARAM_READWRITE |
				G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

	g_object_class_install_property (gobject_class, PROP_SELECTION_COUNT,
		g_param_spec_uint ("selection-count",
			"Selection Count",
			"Number of random frames to select from cache", 1,
			G_MAXUINT, DEFAULT_SELECTION_COUNT,
			GParamFlags
			(G_PARAM_READWRITE |
				G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

	g_object_class_install_property (gobject_class, PROP_PERCENTAGE_THRESHOLD,
		g_param_spec_double ("percentage-threshold",
			"Percentage Threshold",
			"Percentage threshold for frame filtering (0.0-100.0%, where 100% = max pixel difference of 255)", 0.0,
			100.0, DEFAULT_PERCENTAGE_THRESHOLD,
			GParamFlags
			(G_PARAM_READWRITE |
				G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

	g_object_class_install_property (gobject_class, PROP_AVERAGING_FRAMES,
		g_param_spec_uint ("averaging-frames",
			"Averaging Frames",
			"Number of frames to average for normalization", 1,
			G_MAXUINT, DEFAULT_AVERAGING_FRAMES,
			GParamFlags
			(G_PARAM_READWRITE |
				G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

	g_object_class_install_property (gobject_class, PROP_EXCLUSION_RANGE,
		g_param_spec_uint ("exclusion-range",
			"Exclusion Range",
			"+/- range value for frame exclusion in selection algorithm", 0,
			G_MAXUINT, DEFAULT_EXCLUSION_RANGE,
			GParamFlags
			(G_PARAM_READWRITE |
				G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

	g_object_class_install_property (gobject_class, PROP_FRAME_SELECTION_ALGORITHM,
		g_param_spec_enum ("frame-selection-algorithm",
			"Frame Selection Algorithm",
			"Algorithm to use for frame selection (0=BASIC equidistant, 1=RANGE_BASED SAD, 2=OF optical-flow; default OF)",
			GST_TYPE_NVDS_FRAME_SELECTION_ALGORITHM, DEFAULT_FRAME_SELECTION_ALGO,
			GParamFlags
			(G_PARAM_READWRITE |
				G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

	g_object_class_install_property (gobject_class, PROP_BYPASS_MODE,
		g_param_spec_boolean ("bypass-mode",
			"Bypass Mode",
			"Enable bypass mode to pass all frames through without selection", DEFAULT_BYPASS_MODE,
			GParamFlags
			(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

	g_object_class_install_property (gobject_class, PROP_OPTICAL_FLOW_INTERVAL,
		g_param_spec_uint ("optical-flow-interval",
			"Optical Flow Interval",
			"Analyze every Nth frame for optical flow motion detection. "
			"0 = AUTO: derive N from the received frame rate so the OF gap stays ~0.5s "
			"(the threshold-calibration point) at any input fps. "
			"1=every frame, 15=every 15th frame, etc.",
			0, G_MAXUINT, DEFAULT_OPTICAL_FLOW_INTERVAL,
			GParamFlags
			(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

	g_object_class_install_property (gobject_class, PROP_ENABLE_MOTION_DETECTION,
		g_param_spec_boolean ("enable-motion-detection",
			"Enable Motion Detection",
			"Enable optical flow based motion detection for adaptive frame selection (TRUE=enabled, FALSE=disabled)", TRUE,
			GParamFlags
			(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

	g_object_class_install_property (gobject_class, PROP_SOURCE_FPS,
		g_param_spec_double ("source-fps",
			"Source FPS",
			"Native stream frame rate. Used to normalize optical-flow motion to "
			"px-per-source-frame via frame PTS, making motion classification invariant "
			"to input decimation (e.g. feeding every Nth frame). 0 = unknown -> fall "
			"back to dividing by optical-flow-interval (received-frame count).",
			0.0, G_MAXDOUBLE, 0.0,
			GParamFlags
			(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

	g_object_class_install_property (gobject_class, PROP_OF_TIGHTEN,
		g_param_spec_double ("of-tighten",
			"OF Tighten",
			"Tighten factor applied to the motion-scaled frame count before the "
			"minimum clamp (e.g. 0.5 halves the count, dropping the weakest frames). "
			"Range (0.0, 1.0]. Env NVDS_FSELECT_OF_TIGHTEN overrides at runtime.",
			0.0, 1.0, OF_ONLY_TIGHTEN,
			GParamFlags
			(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

	g_object_class_install_property (gobject_class, PROP_EQUIDISTANT_OUTPUT,
		g_param_spec_boolean ("equidistant-output",
			"Equidistant Output",
			"When TRUE, output equidistant frames from the chunk instead of "
			"motion-ranked frames. The count is still determined by the selection "
			"algorithm (OF/BASIC/RANGE_BASED); only the positions become evenly spaced.",
			DEFAULT_EQUIDISTANT_OUTPUT,
			GParamFlags
			(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

	/* Set sink and src pad capabilities */
	gst_element_class_add_pad_template (gstelement_class,
		gst_static_pad_template_get (&gst_nvds_frame_selector_src_template));
	gst_element_class_add_pad_template (gstelement_class,
		gst_static_pad_template_get (&gst_nvds_frame_selector_sink_template));

	/* Set metadata describing the element */
	gst_element_class_set_details_simple (gstelement_class,
		"NvDsFrameSelector",
		"Video/Filter",
		DESCRIPTION, "NVIDIA Corporation. Post on Deepstream for Tesla forum for any queries "
		"@ https://devtalk.nvidia.com/default/board/209/");
}

/* Function called when object is being destroyed. Free up resources. */
static void
gst_nvds_frame_selector_finalize (GObject * object)
{
	GstNvDsFrameSelector *nvdsframeselector = GST_NVDS_FRAME_SELECTOR (object);

	GST_DEBUG_OBJECT (nvdsframeselector, "Finalizing frame selector");

	/* Free cached frame buffers */
	if (nvdsframeselector->cached_frames) {
		guint total_cache_size = nvdsframeselector->cache_size + nvdsframeselector->selection_count;
		for (guint i = 0; i < total_cache_size; i++) {
			if (nvdsframeselector->cached_frames[i]) {
				NvBufSurfaceDestroy (nvdsframeselector->cached_frames[i]);
				nvdsframeselector->cached_frames[i] = NULL;
			}
		}
		g_free (nvdsframeselector->cached_frames);
		nvdsframeselector->cached_frames = NULL;
	}

	/* Free SAD values array */
	if (nvdsframeselector->sad_values) {
		g_free (nvdsframeselector->sad_values);
		nvdsframeselector->sad_values = NULL;
	}

	/* Free timestamps array */
	if (nvdsframeselector->cached_timestamps) {
		g_free (nvdsframeselector->cached_timestamps);
		nvdsframeselector->cached_timestamps = NULL;
	}

	/* Free frame numbers array */
	if (nvdsframeselector->cached_frame_numbers) {
		g_free (nvdsframeselector->cached_frame_numbers);
		nvdsframeselector->cached_frame_numbers = NULL;
	}

	/* Free selected indices array */
	if (nvdsframeselector->selected_indices) {
		g_free (nvdsframeselector->selected_indices);
		nvdsframeselector->selected_indices = NULL;
	}

	/* Free previous frame buffer */
	if (nvdsframeselector->previous_frame) {
		NvBufSurfaceDestroy (nvdsframeselector->previous_frame);
		nvdsframeselector->previous_frame = NULL;
	}

	/* Cleanup CUDA resources */
	if (nvdsframeselector->cuda_stream || nvdsframeselector->d_partial_sums ||
		nvdsframeselector->h_partial_sums) {
		cleanupSAD_CUDA(nvdsframeselector->cuda_stream,
		                nvdsframeselector->d_partial_sums,
		                nvdsframeselector->h_partial_sums);
		nvdsframeselector->cuda_stream = NULL;
		nvdsframeselector->d_partial_sums = NULL;
		nvdsframeselector->h_partial_sums = NULL;
		nvdsframeselector->max_blocks = 0;
	}

	/* Cleanup motion detector */
	if (nvdsframeselector->motion_detector_initialized && nvdsframeselector->motion_detector_handle) {
		cleanupMotionDetector((MotionDetectorHandle)nvdsframeselector->motion_detector_handle);
		nvdsframeselector->motion_detector_handle = NULL;
		nvdsframeselector->motion_detector_initialized = FALSE;
	}

	/* Chain up to parent class */
	G_OBJECT_CLASS (gst_nvds_frame_selector_parent_class)->finalize (object);
}

static void
gst_nvds_frame_selector_init (GstNvDsFrameSelector * nvdsframeselector)
{
	GstBaseTransform *btrans = GST_BASE_TRANSFORM (nvdsframeselector);

	/* We will not be generating a new buffer. Just adding / updating
	 * metadata. */
	gst_base_transform_set_in_place (GST_BASE_TRANSFORM (btrans), TRUE);
	/* Don't set passthrough since we need to control which frames pass */
	gst_base_transform_set_passthrough (GST_BASE_TRANSFORM (btrans), FALSE);

	/* Initialize all property variables to default values */
	nvdsframeselector->unique_id = 0;
	nvdsframeselector->frame_num = 0;
	nvdsframeselector->input_frame_count = 0;
	nvdsframeselector->processing_width = 0;
	nvdsframeselector->processing_height = 0;
	nvdsframeselector->gpu_id = DEFAULT_GPU_ID;
	nvdsframeselector->enable_selector = DEFAULT_ENABLE_SELECTOR;

	/* Initialize new configurable properties */
	nvdsframeselector->cache_size = DEFAULT_CACHE_SIZE;
	nvdsframeselector->selection_count = DEFAULT_SELECTION_COUNT;
	nvdsframeselector->percentage_threshold = DEFAULT_PERCENTAGE_THRESHOLD;
	nvdsframeselector->averaging_frames = DEFAULT_AVERAGING_FRAMES;
	nvdsframeselector->exclusion_range = DEFAULT_EXCLUSION_RANGE;
	nvdsframeselector->frame_selection_algorithm = DEFAULT_FRAME_SELECTION_ALGO;
	nvdsframeselector->user_frame_selection_algorithm = DEFAULT_FRAME_SELECTION_ALGO;
	nvdsframeselector->bypass_mode = DEFAULT_BYPASS_MODE;
	nvdsframeselector->optical_flow_interval = DEFAULT_OPTICAL_FLOW_INTERVAL;
	nvdsframeselector->enable_motion_detection = TRUE;  // Enabled by default
	nvdsframeselector->source_fps = 0.0;  // 0 = unknown → fall back to frame-count normalization
	nvdsframeselector->of_tighten = OF_ONLY_TIGHTEN;
	nvdsframeselector->equidistant_output = DEFAULT_EQUIDISTANT_OUTPUT;
	nvdsframeselector->chunk_first_in_us = 0;  // no chunk in progress
	nvdsframeselector->chunk_eos_us = 0;
	nvdsframeselector->chunk_sad_us = 0;

	/* Allocate dynamic arrays based on properties - extended cache size for sliding window */
	guint total_cache_size = nvdsframeselector->cache_size + nvdsframeselector->selection_count;
	nvdsframeselector->cached_frames = g_new0(NvBufSurface *, total_cache_size);
	nvdsframeselector->sad_values = g_new0(guint, total_cache_size);
	nvdsframeselector->cached_timestamps = g_new0(GstClockTime, total_cache_size);
	nvdsframeselector->cached_frame_numbers = g_new0(guint64, total_cache_size);
	nvdsframeselector->selected_indices = g_new0(guint, nvdsframeselector->selection_count);

	/* Initialize frame cache */
	for (guint i = 0; i < total_cache_size; i++) {
		nvdsframeselector->cached_frames[i] = NULL;
		nvdsframeselector->sad_values[i] = 0;
		nvdsframeselector->cached_timestamps[i] = GST_CLOCK_TIME_NONE;
		nvdsframeselector->cached_frame_numbers[i] = 0;
	}
	nvdsframeselector->cached_frame_count = 0;
	nvdsframeselector->cache_index = 0;
	nvdsframeselector->is_outputting = FALSE;
	nvdsframeselector->output_count = 0;

	/* Initialize CUDA resources to NULL - will be allocated in set_caps */
	nvdsframeselector->cuda_stream = NULL;
	nvdsframeselector->d_partial_sums = NULL;
	nvdsframeselector->h_partial_sums = NULL;
	nvdsframeselector->max_blocks = 0;
	nvdsframeselector->motion_detector_initialized = FALSE;
	nvdsframeselector->motion_detector_handle = NULL;
	nvdsframeselector->actual_selected_count = 0;
	nvdsframeselector->previous_frame = NULL;

	/* Initialize selected indices array */
	for (guint i = 0; i < nvdsframeselector->selection_count; i++) {
		nvdsframeselector->selected_indices[i] = 0;
	}

	/* This quark is required to identify NvDsMeta when iterating through
	 * the buffer metadatas */
	if (!_dsmeta_quark)
		_dsmeta_quark = g_quark_from_static_string (NVDS_META_STRING);
}

/* Function called when a property of the element is set. Standard boilerplate.
 */
static void
gst_nvds_frame_selector_set_property (GObject * object, guint prop_id,
	const GValue * value, GParamSpec * pspec)
{
	GstNvDsFrameSelector *nvdsframeselector = GST_NVDS_FRAME_SELECTOR (object);
	switch (prop_id) {
	  case PROP_GPU_DEVICE_ID:
		nvdsframeselector->gpu_id = g_value_get_uint (value);
		break;
	  case PROP_ENABLE_SELECTOR:
		nvdsframeselector->enable_selector = g_value_get_boolean (value);
		break;
	  case PROP_CACHE_SIZE:
		{
		  guint new_cache_size = g_value_get_uint (value);
		  if (new_cache_size != nvdsframeselector->cache_size) {
			/* Destroy existing NvBufSurface objects before freeing arrays */
			guint old_total_cache_size = nvdsframeselector->cache_size + nvdsframeselector->selection_count;
			if (nvdsframeselector->cached_frames) {
				for (guint i = 0; i < old_total_cache_size; i++) {
					if (nvdsframeselector->cached_frames[i]) {
						NvBufSurfaceDestroy(nvdsframeselector->cached_frames[i]);
						nvdsframeselector->cached_frames[i] = NULL;
					}
				}
			}
			if (nvdsframeselector->previous_frame) {
				NvBufSurfaceDestroy(nvdsframeselector->previous_frame);
				nvdsframeselector->previous_frame = NULL;
			}

			/* Reallocate cached_frames, sad_values, timestamps, and frame_numbers arrays with new total size */
			guint new_total_cache_size = new_cache_size + nvdsframeselector->selection_count;
			g_free (nvdsframeselector->cached_frames);
			g_free (nvdsframeselector->sad_values);
			g_free (nvdsframeselector->cached_timestamps);
			g_free (nvdsframeselector->cached_frame_numbers);
			nvdsframeselector->cached_frames = g_new0(NvBufSurface *, new_total_cache_size);
			nvdsframeselector->sad_values = g_new0(guint, new_total_cache_size);
			nvdsframeselector->cached_timestamps = g_new0(GstClockTime, new_total_cache_size);
			nvdsframeselector->cached_frame_numbers = g_new0(guint64, new_total_cache_size);
			nvdsframeselector->cache_size = new_cache_size;
			/* Reset cache state */
			nvdsframeselector->cached_frame_count = 0;
			nvdsframeselector->is_outputting = FALSE;
			nvdsframeselector->output_count = 0;
			GST_INFO_OBJECT (nvdsframeselector, "Updated cache size to %u (total cache: %u)", new_cache_size, new_total_cache_size);
		  }
		}
		break;
	  case PROP_SELECTION_COUNT:
		{
		  guint new_selection_count = g_value_get_uint (value);
		  if (new_selection_count != nvdsframeselector->selection_count) {
			/* Destroy existing NvBufSurface objects before freeing arrays */
			guint old_total_cache_size = nvdsframeselector->cache_size + nvdsframeselector->selection_count;
			if (nvdsframeselector->cached_frames) {
				for (guint i = 0; i < old_total_cache_size; i++) {
					if (nvdsframeselector->cached_frames[i]) {
						NvBufSurfaceDestroy(nvdsframeselector->cached_frames[i]);
						nvdsframeselector->cached_frames[i] = NULL;
					}
				}
			}
			if (nvdsframeselector->previous_frame) {
				NvBufSurfaceDestroy(nvdsframeselector->previous_frame);
				nvdsframeselector->previous_frame = NULL;
			}

			/* Reallocate cached_frames, sad_values, timestamps, frame_numbers and selected_indices arrays with new sizes */
			guint new_total_cache_size = nvdsframeselector->cache_size + new_selection_count;
			g_free (nvdsframeselector->cached_frames);
			g_free (nvdsframeselector->sad_values);
			g_free (nvdsframeselector->cached_timestamps);
			g_free (nvdsframeselector->cached_frame_numbers);
			g_free (nvdsframeselector->selected_indices);
			nvdsframeselector->cached_frames = g_new0(NvBufSurface *, new_total_cache_size);
			nvdsframeselector->sad_values = g_new0(guint, new_total_cache_size);
			nvdsframeselector->cached_timestamps = g_new0(GstClockTime, new_total_cache_size);
			nvdsframeselector->cached_frame_numbers = g_new0(guint64, new_total_cache_size);
			nvdsframeselector->selected_indices = g_new0(guint, new_selection_count);
			nvdsframeselector->selection_count = new_selection_count;
			/* Reset output state */
			nvdsframeselector->cached_frame_count = 0;
			nvdsframeselector->is_outputting = FALSE;
			nvdsframeselector->output_count = 0;
			GST_INFO_OBJECT (nvdsframeselector, "Updated selection count to %u (total cache: %u)", new_selection_count, new_total_cache_size);
		  }
		}
		break;
	  case PROP_PERCENTAGE_THRESHOLD:
		nvdsframeselector->percentage_threshold = g_value_get_double (value);
		break;
	  case PROP_AVERAGING_FRAMES:
		nvdsframeselector->averaging_frames = g_value_get_uint (value);
		break;
	  case PROP_EXCLUSION_RANGE:
		nvdsframeselector->exclusion_range = g_value_get_uint (value);
		break;
	  case PROP_FRAME_SELECTION_ALGORITHM:
		nvdsframeselector->frame_selection_algorithm = g_value_get_enum (value);
		nvdsframeselector->user_frame_selection_algorithm = g_value_get_enum (value);
		break;
	  case PROP_BYPASS_MODE:
		nvdsframeselector->bypass_mode = g_value_get_boolean (value);
		break;
	  case PROP_OPTICAL_FLOW_INTERVAL:
		/* 0 = AUTO (derive from received fps to hold ~0.5s OF gap); >=1 = explicit stride. */
		nvdsframeselector->optical_flow_interval = g_value_get_uint (value);
		break;
	  case PROP_ENABLE_MOTION_DETECTION:
		nvdsframeselector->enable_motion_detection = g_value_get_boolean (value);
		break;
	  case PROP_SOURCE_FPS:
		nvdsframeselector->source_fps = g_value_get_double (value);
		break;
	  case PROP_OF_TIGHTEN:
		nvdsframeselector->of_tighten = g_value_get_double (value);
		break;
	  case PROP_EQUIDISTANT_OUTPUT:
		nvdsframeselector->equidistant_output = g_value_get_boolean (value);
		break;
	  default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
		break;
	}
}

/* Function called when a property of the element is requested. Standard
 * boilerplate.
 */
static void
gst_nvds_frame_selector_get_property (GObject * object, guint prop_id,
	GValue * value, GParamSpec * pspec)
{
	GstNvDsFrameSelector *nvdsframeselector = GST_NVDS_FRAME_SELECTOR (object);
	switch (prop_id) {
	  case PROP_GPU_DEVICE_ID:
		g_value_set_uint (value, nvdsframeselector->gpu_id);
		break;
	  case PROP_ENABLE_SELECTOR:
		g_value_set_boolean (value, nvdsframeselector->enable_selector);
		break;
	  case PROP_CACHE_SIZE:
		g_value_set_uint (value, nvdsframeselector->cache_size);
		break;
	  case PROP_SELECTION_COUNT:
		g_value_set_uint (value, nvdsframeselector->selection_count);
		break;
	  case PROP_PERCENTAGE_THRESHOLD:
		g_value_set_double (value, nvdsframeselector->percentage_threshold);
		break;
	  case PROP_AVERAGING_FRAMES:
		g_value_set_uint (value, nvdsframeselector->averaging_frames);
		break;
	  case PROP_EXCLUSION_RANGE:
		g_value_set_uint (value, nvdsframeselector->exclusion_range);
		break;
	  case PROP_FRAME_SELECTION_ALGORITHM:
		g_value_set_enum (value, nvdsframeselector->frame_selection_algorithm);
		break;
	  case PROP_BYPASS_MODE:
		g_value_set_boolean (value, nvdsframeselector->bypass_mode);
		break;
	  case PROP_OPTICAL_FLOW_INTERVAL:
		g_value_set_uint (value, nvdsframeselector->optical_flow_interval);
		break;
	  case PROP_ENABLE_MOTION_DETECTION:
		g_value_set_boolean (value, nvdsframeselector->enable_motion_detection);
		break;
	  case PROP_SOURCE_FPS:
		g_value_set_double (value, nvdsframeselector->source_fps);
		break;
	  case PROP_OF_TIGHTEN:
		g_value_set_double (value, nvdsframeselector->of_tighten);
		break;
	  case PROP_EQUIDISTANT_OUTPUT:
		g_value_set_boolean (value, nvdsframeselector->equidistant_output);
		break;
	  default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
		break;
	}
}

/**
 * Initialize all resources and start the output thread
 */
static gboolean
gst_nvds_frame_selector_start(GstBaseTransform *btrans)
{
	// TODO: Implement start function
	return TRUE;
}

/**
 * Stop the output thread and free up all the resources
 */
static gboolean
gst_nvds_frame_selector_stop (GstBaseTransform * btrans)
{
	GstNvDsFrameSelector *nvdsframeselector = GST_NVDS_FRAME_SELECTOR (btrans);
	/* Clean up cached frames */
	guint total_cache_size = nvdsframeselector->cache_size + nvdsframeselector->selection_count;
	for (guint i = 0; i < total_cache_size; i++) {
		if (nvdsframeselector->cached_frames[i]) {
			NvBufSurfaceDestroy(nvdsframeselector->cached_frames[i]);
			nvdsframeselector->cached_frames[i] = NULL;
		}
	}
	/* Clean up previous frame buffer */
	if (nvdsframeselector->previous_frame) {
		NvBufSurfaceDestroy(nvdsframeselector->previous_frame);
		nvdsframeselector->previous_frame = NULL;
	}
	/* Cleanup CUDA resources */
	if (nvdsframeselector->cuda_stream || nvdsframeselector->d_partial_sums ||
		nvdsframeselector->h_partial_sums) {
		cleanupSAD_CUDA(nvdsframeselector->cuda_stream,
		                nvdsframeselector->d_partial_sums,
		                nvdsframeselector->h_partial_sums);
		nvdsframeselector->cuda_stream = NULL;
		nvdsframeselector->d_partial_sums = NULL;
		nvdsframeselector->h_partial_sums = NULL;
		nvdsframeselector->max_blocks = 0;
	}
	/* Cleanup motion detector */
	if (nvdsframeselector->motion_detector_initialized && nvdsframeselector->motion_detector_handle) {
		cleanupMotionDetector((MotionDetectorHandle)nvdsframeselector->motion_detector_handle);
		nvdsframeselector->motion_detector_handle = NULL;
		nvdsframeselector->motion_detector_initialized = FALSE;
	}
	/* Reset state counters */
	nvdsframeselector->cached_frame_count = 0;
	nvdsframeselector->cache_index = 0;
	nvdsframeselector->is_outputting = FALSE;
	nvdsframeselector->output_count = 0;
	nvdsframeselector->actual_selected_count = 0;
	nvdsframeselector->input_frame_count = 0;

	return TRUE;
}

static gboolean
gst_nvds_frame_selector_transform_size(GstBaseTransform* btrans,
		GstPadDirection dir, GstCaps *caps, gsize size, GstCaps* othercaps, gsize* othersize)
{
	// TODO: Implement transform_size function
	*othersize = size;
	return TRUE;
}

/**
 * Check if caps can be accepted
 */
static gboolean
gst_nvds_frame_selector_accept_caps (GstBaseTransform * btrans,
	GstPadDirection direction, GstCaps * caps)
{
	// TODO: Implement accept_caps function
	return TRUE;
}

/**
 * Fixate caps
 */
static GstCaps*
gst_nvds_frame_selector_fixate_caps(GstBaseTransform* btrans,
		GstPadDirection direction, GstCaps* caps, GstCaps* othercaps)
{
	// TODO: Implement fixate_caps function
	return gst_caps_ref(othercaps);
}

/**
 * Transform caps
 */
static GstCaps*
gst_nvds_frame_selector_transform_caps(GstBaseTransform* btrans, GstPadDirection dir,
	GstCaps* caps, GstCaps* filter)
{
	// TODO: Implement transform_caps function
	return gst_caps_ref(caps);
}

/**
 * Called when source / sink pad capabilities have been negotiated.
 */
static gboolean
gst_nvds_frame_selector_set_caps (GstBaseTransform * btrans, GstCaps * incaps,
	GstCaps * outcaps)
{
	GstNvDsFrameSelector *nvdsframeselector = GST_NVDS_FRAME_SELECTOR (btrans);
	GstVideoInfo in_info = {0};
	if (!gst_video_info_from_caps (&in_info, incaps))
    {
      GST_ERROR ("invalid input caps");
      return FALSE;
    }

	guint width = GST_VIDEO_INFO_WIDTH (&in_info);
	guint height = GST_VIDEO_INFO_HEIGHT (&in_info);

	/* Initialize CUDA resources if not already initialized */
	if (!nvdsframeselector->cuda_stream) {
		int max_blocks = initSAD_CUDA(&nvdsframeselector->cuda_stream,
		                              &nvdsframeselector->d_partial_sums,
		                              &nvdsframeselector->h_partial_sums,
		                              width, height);
		if (max_blocks < 0) {
			GST_ERROR_OBJECT (nvdsframeselector, "Failed to initialize CUDA resources for SAD computation");
			return FALSE;
		}
		nvdsframeselector->max_blocks = max_blocks;
	}

	/* Initialize motion detector if enabled and not already initialized */
	if (nvdsframeselector->enable_motion_detection && !nvdsframeselector->motion_detector_initialized) {
		MotionDetectorHandle handle = NULL;
		if (initMotionDetector(NULL, width, height, &handle) == 0) {
			nvdsframeselector->motion_detector_handle = handle;
			nvdsframeselector->motion_detector_initialized = TRUE;
		} else {
			nvdsframeselector->motion_detector_handle = NULL;
			GST_WARNING_OBJECT (nvdsframeselector, "Failed to initialize Motion Detector - will skip motion analysis");
			nvdsframeselector->motion_detector_initialized = FALSE;
		}
	} else if (!nvdsframeselector->enable_motion_detection) {
		GST_DEBUG_OBJECT (nvdsframeselector, "Motion Detection is disabled (enable-motion-detection=false)");
	}
	return TRUE;
}

/**
 * Basic frame selection algorithm - selects equidistant frames from cache
 */
static void
gst_nvds_frame_selection_basic_algo (GstNvDsFrameSelector * nvdsframeselector)
{
	guint cached_frame_count = nvdsframeselector->cached_frame_count;
	guint selection_count = nvdsframeselector->selection_count;

	/* Ensure we don't try to select more frames than we have cached */
	if (selection_count > cached_frame_count) {
		selection_count = cached_frame_count;
		GST_WARNING_OBJECT (nvdsframeselector,
						  "Selection count (%u) > cached frame count (%u), limiting to cached frame count",
						  nvdsframeselector->selection_count, cached_frame_count);
	}

	if (selection_count == 0) {
		return;
	}

	if (selection_count == 1) {
		/* Special case: select the middle frame */
		nvdsframeselector->selected_indices[0] = cached_frame_count / 2;
	} else if (selection_count >= cached_frame_count) {
		/* Select all frames */
		for (guint i = 0; i < cached_frame_count; i++) {
			nvdsframeselector->selected_indices[i] = i;
		}
	} else {
		/* Calculate step size for equidistant selection */
		/* Distribute frames evenly across the entire cache */
		gdouble step = (gdouble)(cached_frame_count) / (gdouble)(selection_count);

		for (guint i = 0; i < selection_count; i++) {
			guint index = (guint)(i * step + 0.5);  /* Round to nearest */
			if (index >= cached_frame_count) {
				index = cached_frame_count - 1;
			}
			nvdsframeselector->selected_indices[i] = index;
		}
	}

	/* Store the actual number of frames selected */
	nvdsframeselector->actual_selected_count = selection_count;
}


/* Return TRUE if the OF selection path should run for this chunk. OF is now a
 * first-class algorithm (enum value FRAME_SELECTION_ALGO_OF, the default). It runs
 * when the selected algorithm is OF. Legacy override: NVDS_FSELECT_OF_ONLY=0 forces
 * the fallback (RANGE_BASED/BASIC) even when OF is selected. */
static gboolean
gst_nvds_frame_selection_of_only_enabled (GstNvDsFrameSelector * self)
{
	const char *e = getenv ("NVDS_FSELECT_OF_ONLY");
	if (e && e[0] == '0' && e[1] == '\0')
		return FALSE;
	return (self->frame_selection_algorithm == FRAME_SELECTION_ALGO_OF);
}

/* OF-ONLY frame selection — a simple, self-contained alternative to the SAD blend.
 * Everything is driven by optical-flow moving-cells (motion-specific, not fooled by
 * compression noise):
 *   Step 1 (how many): target count scales with the chunk's ABSOLUTE motion — the
 *     peak moving-cell count. Below the noise floor -> 1 frame; grows linearly to
 *     selection_count at OF_ONLY_MC_FULL. Active chunks get at least OF_ONLY_MIN_ACTIVE.
 *   Step 2 (which): keep the `target` frames with the most motion, with a light
 *     min-gap so they don't all bunch into one instant.
 * No SAD, no cumulative-diversity, no TAU. Reads only sample_moving_cells[] (strided
 * by optical-flow-interval; a sample at i covers span [i, i+N]). Tunable via env
 * NVDS_FSELECT_OF_MC_FULL and NVDS_FSELECT_OF_MIN. Leaves the blend path untouched. */
static void
gst_nvds_frame_selection_of_only (GstNvDsFrameSelector * nvdsframeselector,
    MotionLevel * sample_levels, int * sample_moving_cells)
{
	guint n = nvdsframeselector->cached_frame_count;
	if (n == 0) { nvdsframeselector->actual_selected_count = 0; return; }
	if (n == 1) {
		nvdsframeselector->selected_indices[0] = 0;
		nvdsframeselector->actual_selected_count = 1;
		return;
	}

	guint N = nvdsframeselector->eff_of_interval;   /* effective interval used for analysis */
	if (N < 1) N = 1;
	guint SC = nvdsframeselector->selection_count;
	if (SC < 1) SC = 1;

	/* Per-frame motion score from OF moving-cells; a sample at i covers span [i, i+N].
	 * Also track the chunk's peak OF motion LEVEL (whole-frame magnitude class) — used to
	 * relax tightening on extreme-motion chunks. */
	std::vector<int> score(n, 0);
	int peak = 0;
	MotionLevel peak_level = MOTION_STATIC;
	for (guint i = 0; i + N < n; i += N) {
		int mc = sample_moving_cells ? sample_moving_cells[i] : 0;
		if (mc > peak) peak = mc;
		if (sample_levels && sample_levels[i] > peak_level) peak_level = sample_levels[i];
		guint hi = i + N; if (hi >= n) hi = n - 1;
		for (guint j = i; j <= hi; j++)
			if (mc > score[j]) score[j] = mc;
	}

	/* Env-tunable knobs (expressed in 1080p-reference moving-cell units). */
	double mc_full = OF_ONLY_MC_FULL;
	guint min_active = OF_ONLY_MIN_ACTIVE;
	double tighten = nvdsframeselector->of_tighten;   /* property; env overrides below */
	double tighten_extreme = OF_ONLY_TIGHTEN_EXTREME;
	{
		const char *e = getenv ("NVDS_FSELECT_OF_MC_FULL");
		if (e && *e) { double v = atof (e); if (v >= 1.0) mc_full = v; }
		e = getenv ("NVDS_FSELECT_OF_MIN");
		if (e && *e) { int v = atoi (e); if (v >= 1) min_active = (guint) v; }
		e = getenv ("NVDS_FSELECT_OF_TIGHTEN");
		if (e && *e) { double v = atof (e); if (v > 0.0 && v <= 1.0) tighten = v; }
		e = getenv ("NVDS_FSELECT_OF_TIGHTEN_EXTREME");
		if (e && *e) { double v = atof (e); if (v > 0.0 && v <= 1.0) tighten_extreme = v; }
	}
	/* Extreme-motion override: a chunk with very high whole-frame motion shouldn't be
	 * tightened as hard — relax the fraction up to tighten_extreme so big real events
	 * keep more frames. */
	if (peak_level >= MOTION_VERY_HIGH && tighten < tighten_extreme)
		tighten = tighten_extreme;

	/* Resolution-invariant scaling: the moving-cell COUNT thresholds are referenced to
	 * the 1080p OF grid; scale them by (actual grid cells / reference) so a given
	 * fraction-of-frame moving means the same at any resolution. */
	double res_scale = (nvdsframeselector->of_grid_cells > 0)
		? ((double) nvdsframeselector->of_grid_cells / (double) OF_REF_GRID_CELLS) : 1.0;
	double base_floor = OF_SAD_MC_ABS_FLOOR;
	{ const char *ef = getenv ("NVDS_FSELECT_OF_FLOOR"); if (ef && *ef) { double v = atof (ef); if (v >= 0) base_floor = v; } }
	int mc_floor = (int) (base_floor * res_scale + 0.5);
	mc_full *= res_scale;

	/* Static (nothing above the noise floor): 3 EQUIDISTANT representative frames
	 * (each covers a third of the chunk ~ n/6, n/2, 5n/6). 3 keeps the max intra-chunk
	 * gap ~3.3s (vs ~5s for 2), so a faint/distant subject transiting a below-floor chunk
	 * is less likely to be under-sampled. Clamped to n when the chunk is very short. */
	if (peak < mc_floor) {
		guint k = (n >= 3) ? 3 : n;   /* 3 when we have >=3 frames, else all we have */
		for (guint i = 0; i < k; i++)
			nvdsframeselector->selected_indices[i] = (guint) (((i + 0.5) * (double) n) / (double) k);
		nvdsframeselector->actual_selected_count = k;
		return;
	}

	/* Step 1: target count from absolute motion (moving-cell peak). */
	guint target;
	{
		double frac = (double) peak / mc_full;
		if (frac > 1.0) frac = 1.0;
		target = (guint) (SC * frac * tighten + 0.5);   /* tighten: fewer frames, drop the weakest */
		if (target < min_active) target = min_active;
		if (target > SC) target = SC;
	}
	if (target >= n) {                               /* want everything we have */
		for (guint j = 0; j < n; j++) nvdsframeselector->selected_indices[j] = j;
		nvdsframeselector->actual_selected_count = n;
		return;
	}

	/* Step 2: keep the `target` highest-motion frames, with a light min-gap so they
	 * spread out instead of clustering in a single instant. */
	std::vector<guint> order(n);
	for (guint j = 0; j < n; j++) order[j] = j;
	std::stable_sort (order.begin (), order.end (),
		[&](guint a, guint b) { return score[a] > score[b]; });

	guint min_gap = n / (2 * target);
	if (min_gap < 1) min_gap = 1;
	std::vector<char> picked (n, 0);
	guint kept = 0;
	for (guint t = 0; t < n && kept < target; t++) {          /* pass 1: spaced */
		guint idx = order[t];
		gboolean too_close = FALSE;
		guint lo = (idx > min_gap) ? (idx - min_gap) : 0;
		guint hi = idx + min_gap; if (hi >= n) hi = n - 1;
		for (guint j = lo; j <= hi; j++)
			if (picked[j]) { too_close = TRUE; break; }
		if (too_close) continue;
		picked[idx] = 1; kept++;
	}
	for (guint t = 0; t < n && kept < target; t++) {          /* pass 2: fill if short */
		guint idx = order[t];
		if (!picked[idx]) { picked[idx] = 1; kept++; }
	}

	/* Sparse-chunk context: ONLY when the chunk is active but FEWER THAN 6 frames were
	 * selected. A chunk that already yields >=6 frames gives the model enough temporal
	 * coverage, so we leave it alone. For a 1-5 frame event we add a little context so it
	 * isn't isolated, kept deliberately light (+/-1, not a full window):
	 *   - one frame BEFORE the first selected frame,
	 *   - one frame AFTER  the last selected frame,
	 *   - a MIDPOINT frame between any two far-apart selected frames.
	 * Enabled by default; disable with NVDS_FSELECT_OF_CONTEXT=0. */
	gboolean motion = (peak >= mc_floor);
	gboolean ctx_on = TRUE;
	{
		const char *e = getenv ("NVDS_FSELECT_OF_CONTEXT");
		if (e && *e && atoi (e) == 0) ctx_on = FALSE;
	}
	if (motion && ctx_on && kept < 6) {
		std::vector<guint> sel;
		for (guint j = 0; j < n; j++) if (picked[j]) sel.push_back (j);
		if (!sel.empty ()) {
			guint first = sel.front (), last = sel.back ();
			if (first > 0)        picked[first - 1] = 1;   /* one frame before the first */
			if (last + 1 < n)     picked[last + 1]  = 1;   /* one frame after the last */
			for (guint s = 0; s + 1 < sel.size (); s++) {  /* midpoint of far-apart pairs */
				if (sel[s + 1] - sel[s] >= 4)
					picked[(sel[s] + sel[s + 1]) / 2] = 1;
			}
		}
	}

	/* Collect, then cap at the selection_count budget (even-subsample if context
	 * pushed us over). */
	std::vector<guint> sel;
	for (guint j = 0; j < n; j++)
		if (picked[j]) sel.push_back (j);
	if ((guint) sel.size () > SC) {
		std::vector<guint> reduced;
		reduced.reserve (SC);
		for (guint i = 0; i < SC; i++) {
			guint idx = (guint) (((double) i * (sel.size () - 1)) / (SC - 1) + 0.5);
			if (idx >= sel.size ()) idx = (guint) sel.size () - 1;
			if (reduced.empty () || reduced.back () != sel[idx])
				reduced.push_back (sel[idx]);
		}
		sel.swap (reduced);
	}

	for (guint i = 0; i < sel.size (); i++)
		nvdsframeselector->selected_indices[i] = sel[i];
	nvdsframeselector->actual_selected_count = (guint) sel.size ();
}


static void gst_nvds_frame_selection_algo_range_based (GstNvDsFrameSelector * nvdsframeselector)
{
	/* Use actual cached frame count instead of cache_size to handle EOS case */
	guint cached_frame_count = nvdsframeselector->cached_frame_count;
	guint selection_count = nvdsframeselector->selection_count;
	guint exclusion_range = nvdsframeselector->exclusion_range;

	/* Calculate exclusion range dynamically if not set (i.e., if it's 0) */
	if (exclusion_range == 0 && selection_count > 0) {
		gdouble temp = ceil(cached_frame_count / selection_count);
		exclusion_range = ceil(temp / 2);
	}
	if (exclusion_range == 0) {
		exclusion_range = 1;
	}
	GST_DEBUG_OBJECT (nvdsframeselector, "Exclusion range: %u", exclusion_range);

	/* Ensure we don't try to select more frames than we have cached */
	if (selection_count > cached_frame_count) {
		selection_count = cached_frame_count;
		GST_WARNING_OBJECT (nvdsframeselector,
						  "Selection count (%u) > cached frame count (%u), limiting to cached frame count",
						  nvdsframeselector->selection_count, cached_frame_count);
	}

	/* Step 1: Calculate AVG SAD for 10 consecutive frames */
	guint *avg_sad_values = g_new0(guint, cached_frame_count);
	guint window_size = nvdsframeselector->averaging_frames;

	guint64 orig_sad_value[cached_frame_count];
	for (guint i = 0; i < cached_frame_count; i++) {
		orig_sad_value[i] = nvdsframeselector->sad_values[i];
	}

	/* First 10 frames get 0 SAD - also overwrite in sad_values array */
	for (guint i = 0; i < window_size && i < cached_frame_count; i++) {
		avg_sad_values[i] = 0;
		nvdsframeselector->sad_values[i] = 0;  /* Overwrite original SAD values */
	}

	/* Calculate AVG SAD for remaining frames and overwrite sad_values */
	for (guint i = window_size; i < cached_frame_count; i++) {
		guint64 sum = 0;
		for (guint j = i - window_size; j < i; j++) {
			sum += orig_sad_value[j];
		}
		avg_sad_values[i] = (guint)(sum / window_size);
		nvdsframeselector->sad_values[i] = avg_sad_values[i];  /* Overwrite original SAD values */
	}

	/* Step 2: Create pairs of (AVG SAD value, frame index) for sorting */
	typedef struct {
		guint avg_sad_value;
		guint frame_index;
		gboolean selected;
	} AvgSadFramePair;

	AvgSadFramePair *sad_pairs = g_new0(AvgSadFramePair, cached_frame_count);

	for (guint i = 0; i < cached_frame_count; i++) {
		sad_pairs[i].avg_sad_value = avg_sad_values[i];
		sad_pairs[i].frame_index = i;
		sad_pairs[i].selected = FALSE;
	}

	/* Step 3: Sort based on AVG SAD Values (descending order - highest first) */
	for (guint i = 0; i < cached_frame_count - 1; i++) {
		for (guint j = i + 1; j < cached_frame_count; j++) {
			if (sad_pairs[i].avg_sad_value < sad_pairs[j].avg_sad_value) {
				AvgSadFramePair temp = sad_pairs[i];
				sad_pairs[i] = sad_pairs[j];
				sad_pairs[j] = temp;
			}
		}
	}

	/* Step 4: Use all sorted frames as candidates (no top-50% filter) */
	guint top_half_count = cached_frame_count;


	/* Step 5: Range-based selection from top 50% */
	guint selected_count = 0;
	guint *temp_selected = g_new0(guint, selection_count);

	/* Helper function to check if a frame is in exclusion range of any selected frame */
	auto is_in_exclusion_range = [&](guint frame_idx) -> gboolean {
		for (guint i = 0; i < selected_count; i++) {
			gint diff = (gint)frame_idx - (gint)temp_selected[i];
			if (diff >= -((gint)exclusion_range) && diff <= (gint)exclusion_range) {
				return TRUE;
			}
		}
		return FALSE;
	};

	/* Select frames from top 50% with range constraint */
	for (guint i = 0; i < top_half_count && selected_count < selection_count; i++) {
		guint frame_idx = sad_pairs[i].frame_index;
		if (!is_in_exclusion_range(frame_idx)) {
			temp_selected[selected_count] = frame_idx;
			selected_count++;
		}
	}

	/* Step 6: If we couldn't select enough frames, keep slots empty (don't fill) */
	if (selected_count < selection_count) {
		GST_DEBUG_OBJECT (nvdsframeselector, "Note: Selected only %u frames out of requested %u frames (keeping empty slots)",
			   selected_count, selection_count);
	}

	/* Step 7: Copy to selected_indices and sort in increasing order */
	for (guint i = 0; i < selected_count; i++) {
		nvdsframeselector->selected_indices[i] = temp_selected[i];
	}

	/* Sort selected indices in increasing order */
	for (guint i = 0; i < selected_count - 1; i++) {
		for (guint j = i + 1; j < selected_count; j++) {
			if (nvdsframeselector->selected_indices[i] > nvdsframeselector->selected_indices[j]) {
				guint temp = nvdsframeselector->selected_indices[i];
				nvdsframeselector->selected_indices[i] = nvdsframeselector->selected_indices[j];
				nvdsframeselector->selected_indices[j] = temp;
			}
		}
	}
	/* Store the actual number of frames selected */
	nvdsframeselector->actual_selected_count = selected_count;

	/* Clean up */
	g_free(avg_sad_values);
	g_free(sad_pairs);
	g_free(temp_selected);
}


/* When equidistant-output is enabled, replace whatever indices the selection
 * algorithm chose with evenly-spaced positions spanning the full chunk.
 * actual_selected_count (HOW MANY) is preserved — only the WHICH changes. */
static void
gst_nvds_frame_selector_apply_equidistant (GstNvDsFrameSelector * self)
{
	guint k = self->actual_selected_count;
	guint n = self->cached_frame_count;
	if (k == 0 || n == 0) return;
	for (guint i = 0; i < k; i++)
		self->selected_indices[i] =
			(guint)(((i + 0.5) * (double) n) / (double) k);
}

/* Analyze + select the current cached chunk (cached_frames[0 .. cached_frame_count-1]),
 * setting selected_indices / actual_selected_count. */
/* Effective OF sample interval for a chunk of `count` cached frames. An explicit
 * optical-flow-interval (>=1) is honored as-is. When it is 0 (AUTO), derive it from the
 * RECEIVED frame rate (measured from cached PTS) so the OF gap lands near ~0.5s at any
 * input fps (2..60): interval = round(0.5 * received_fps) = round(0.5 / received_period).
 * This keeps OF at the high-SNR threshold-calibration point (gfac~1.0). Clamped to
 * [1, count/2] so at least two OF samples fit. Falls back to 1 when PTS is unavailable. */
static guint
gst_nvds_effective_of_interval (GstNvDsFrameSelector * self, guint count)
{
	if (self->optical_flow_interval >= 1)
		return self->optical_flow_interval;      /* explicit user value */

	guint interval = 1;
	if (count >= 2 && self->cached_timestamps) {
		GstClockTime t0 = self->cached_timestamps[0];
		GstClockTime t1 = self->cached_timestamps[count - 1];
		if (t0 != GST_CLOCK_TIME_NONE && t1 != GST_CLOCK_TIME_NONE && t1 > t0) {
			double span_sec = (double) (t1 - t0) / 1e9;
			double period = span_sec / (double) (count - 1);   /* sec / received frame */
			if (period > 1e-9) {
				double want = 0.5 / period;                    /* frames per 0.5s */
				interval = (guint) (want + 0.5);
				if (interval < 1) interval = 1;
				guint cap = count / 2;
				if (cap >= 1 && interval > cap) interval = cap;
			}
		}
	}
	return interval;
}

static void
gst_nvds_frame_selector_select_chunk (GstNvDsFrameSelector * nvdsframeselector)
{
	nvdsframeselector->frame_selection_algorithm = nvdsframeselector->user_frame_selection_algorithm;
	gboolean blend_done = FALSE;
	if (nvdsframeselector->enable_motion_detection &&
	    nvdsframeselector->motion_detector_initialized &&
	    nvdsframeselector->cached_frame_count >= 2) {
		ChunkMotionVerdict motionVerdict;
		std::vector<MotionLevel> sample_levels(nvdsframeselector->cached_frame_count, MOTION_STATIC);
		std::vector<int> sample_mcells(nvdsframeselector->cached_frame_count, 0);
		nvdsframeselector->eff_of_interval =
			gst_nvds_effective_of_interval (nvdsframeselector, nvdsframeselector->cached_frame_count);
		if (analyzeChunkMotion((MotionDetectorHandle)nvdsframeselector->motion_detector_handle,
		                       nvdsframeselector->cached_frames,
		                       nvdsframeselector->cached_frame_count,
		                       nvdsframeselector->eff_of_interval,
		                       (const unsigned long long *)nvdsframeselector->cached_timestamps,
		                       nvdsframeselector->source_fps,
		                       &motionVerdict, sample_levels.data(), sample_mcells.data()) == 0) {
			GST_INFO_OBJECT (nvdsframeselector, "MOTION avg=%.2f px/frame of_peak=%.2f level=%s highMotionCount=%d maxMovingCells=%d ofN=%u%s",
				motionVerdict.avgMagnitude,
				motionVerdict.maxSampleMagnitude, motionLevelToString(motionVerdict.overallMotionLevel),
				motionVerdict.highMotionCount, motionVerdict.maxMovingCells,
				nvdsframeselector->eff_of_interval,
				nvdsframeselector->optical_flow_interval == 0 ? " (auto)" : "");
			nvdsframeselector->of_grid_cells = motionVerdict.gridTotalCells;
			if (gst_nvds_frame_selection_of_only_enabled (nvdsframeselector)) {
				gst_nvds_frame_selection_of_only (nvdsframeselector, sample_levels.data(), sample_mcells.data());
				blend_done = TRUE;
				GST_INFO_OBJECT (nvdsframeselector, "OF-only -> %u frame(s)",
					nvdsframeselector->actual_selected_count);
			}
			/* else: OF-only disabled (NVDS_FSELECT_OF_ONLY=0) -> leave blend_done FALSE so the
			 * RANGE_BASED/BASIC fallback below runs. */
		} else {
			GST_WARNING_OBJECT (nvdsframeselector, "Motion analysis failed, using fallback algorithm");
		}
	}

	if (!blend_done) {
		GST_INFO_OBJECT (nvdsframeselector, "algorithm selected = %d", nvdsframeselector->frame_selection_algorithm);
		switch (nvdsframeselector->frame_selection_algorithm) {
			case FRAME_SELECTION_ALGO_BASIC:
				gst_nvds_frame_selection_basic_algo (nvdsframeselector);
				break;
			case FRAME_SELECTION_ALGO_RANGE_BASED:
			default:
				gst_nvds_frame_selection_algo_range_based (nvdsframeselector);
				break;
		}
	}
	if (nvdsframeselector->equidistant_output)
		gst_nvds_frame_selector_apply_equidistant (nvdsframeselector);
}


/* Push one selected cached frame downstream at EOS.
 *
 * The normal (steady-state) output path in transform_ip hands downstream a real,
 * independently-owned NvBufSurface (it deep-copies the cached frame into the
 * incoming buffer via NvBufSurfaceCopy). At EOS there is no incoming buffer, so
 * the old code shallow-copied only the NvBufSurface *header* and pushed a buffer
 * that still referenced the selector's persistent cache GPU surfaces. That aliases
 * under an async, zero-copy-importing HW encoder (e.g. nvv4l2h264enc, whose input
 * buffer pool is ~4 deep): distinct logged frames came out as the SAME image every
 * ~4th frame. Here we mirror the steady-state path: allocate a fresh, solely-owned
 * NVMM surface, deep-copy the cached pixels into it, and wrap it in a buffer that
 * destroys the surface on unref — so the encoder gets a normal, independent input. */
static void
gst_nvds_frame_selector_push_cached_frame (GstNvDsFrameSelector * nvdsframeselector,
    GstPad * srcpad, guint cache_index, const char * tag)
{
	NvBufSurface *cached_surf = nvdsframeselector->cached_frames[cache_index];
	if (!cached_surf)
		return;

	/* Optional percentage-of-change gate (same semantics as the other paths). */
	if (nvdsframeselector->percentage_threshold > 0.0) {
		guint width = cached_surf->surfaceList[0].width;
		guint height = cached_surf->surfaceList[0].height;
		guint max_possible_sad = width * height * 255;
		gdouble sad_percentage =
			((gdouble) nvdsframeselector->sad_values[cache_index] / (gdouble) max_possible_sad) * 100.0;
		if (sad_percentage < nvdsframeselector->percentage_threshold)
			return;
	}

	/* Allocate an independent output surface and DEEP-copy the cached frame. */
	NvBufSurface *out_surf = NULL;
	NvBufSurfaceCreateParams create_params;
	memset (&create_params, 0, sizeof (create_params));
	create_params.gpuId = nvdsframeselector->gpu_id;
	create_params.width = cached_surf->surfaceList[0].width;
	create_params.height = cached_surf->surfaceList[0].height;
	create_params.size = 0;
	create_params.colorFormat = cached_surf->surfaceList[0].colorFormat;
	create_params.layout = cached_surf->surfaceList[0].layout;
	create_params.memType = cached_surf->memType;

	if (NvBufSurfaceCreate (&out_surf, 1, &create_params) != 0 || !out_surf) {
		GST_ERROR_OBJECT (nvdsframeselector, "EOS[%s]: failed to allocate output surface", tag);
		return;
	}
	if (NvBufSurfaceCopy (cached_surf, out_surf) != 0) {
		GST_ERROR_OBJECT (nvdsframeselector, "EOS[%s]: NvBufSurfaceCopy failed", tag);
		NvBufSurfaceDestroy (out_surf);
		return;
	}
	out_surf->numFilled = 1;

	/* Wrap the surface as the buffer payload; free it when the buffer is released.
	 * Size is EXACTLY sizeof(NvBufSurface) — surfaceList is a separate allocation the
	 * struct points to, NOT inlined here. nvvideoconvert asserts inmap.size ==
	 * sizeof(NvBufSurface), so appending sizeof(NvBufSurfaceParams) would fail it. */
	gsize surf_size = sizeof (NvBufSurface);
	GstBuffer *out_buf = gst_buffer_new_wrapped_full ((GstMemoryFlags) 0, out_surf, surf_size,
		0, surf_size, out_surf, (GDestroyNotify) NvBufSurfaceDestroy);
	if (!out_buf) {
		NvBufSurfaceDestroy (out_surf);
		return;
	}

	GST_BUFFER_PTS (out_buf) = nvdsframeselector->cached_timestamps[cache_index];
	GST_INFO_OBJECT (nvdsframeselector, "OUT [%s] frame=%lu PTS=%" GST_TIME_FORMAT,
		tag,
		nvdsframeselector->cached_frame_numbers[cache_index], GST_TIME_ARGS (GST_BUFFER_PTS (out_buf)));

	GstFlowReturn flow_ret = gst_pad_push (srcpad, out_buf);
	if (flow_ret != GST_FLOW_OK)
		GST_WARNING_OBJECT (nvdsframeselector, "EOS[%s]: failed to push frame (flow %d)", tag, flow_ret);
}

/**
 * Handle sink events including EOS
 */
static gboolean
gst_nvds_frame_selector_sink_event (GstBaseTransform * btrans, GstEvent * event)
{
	GstNvDsFrameSelector *nvdsframeselector = GST_NVDS_FRAME_SELECTOR (btrans);
	gboolean ret = TRUE;

	switch (GST_EVENT_TYPE (event)) {
		case GST_EVENT_EOS:
			GST_INFO_OBJECT (nvdsframeselector, "Received EOS event. Total input frames received: %" G_GUINT64_FORMAT,
					nvdsframeselector->input_frame_count);
			/* fselect residency split: stamp EOS arrival (= last input frame in). */
			nvdsframeselector->chunk_eos_us = g_get_monotonic_time();
			if (nvdsframeselector->chunk_first_in_us != 0) {
				GST_INFO_OBJECT (nvdsframeselector, "FSELECT input-phase first-in->EOS = %.1f ms",
					(nvdsframeselector->chunk_eos_us - nvdsframeselector->chunk_first_in_us) / 1000.0);
			}
			if (nvdsframeselector->bypass_mode) {
				goto bypass_mode;
			}

			GST_DEBUG_OBJECT (nvdsframeselector, "output frame count: %u, actual_selected_count: %u",
					nvdsframeselector->output_count, nvdsframeselector->actual_selected_count);

			/* Handle EOS in outputting mode - check actual_selected_count because
			 * selection may have run but output_count could still be 0 if EOS arrives
			 * before any frames were output in this cycle */
			if (nvdsframeselector->actual_selected_count > 0) {
				GST_INFO_OBJECT (nvdsframeselector, "Instance %p EOS *** EOS received in OUTPUTTING mode (output_count=%u, actual_selected_count=%u) ***",
						nvdsframeselector, nvdsframeselector->output_count, nvdsframeselector->actual_selected_count);

				GstPad *srcpad = GST_BASE_TRANSFORM_SRC_PAD (btrans);

				/* Step 1: Output remaining selected frames that haven't been output yet */
				GST_INFO_OBJECT (nvdsframeselector, "Instance %p EOS Step 1: Outputting remaining %u selected frames ***",
						nvdsframeselector, nvdsframeselector->actual_selected_count - nvdsframeselector->output_count);

				for (guint i = nvdsframeselector->output_count; i < nvdsframeselector->actual_selected_count; i++) {
					gst_nvds_frame_selector_push_cached_frame (nvdsframeselector, srcpad,
						nvdsframeselector->selected_indices[i], "eos_remain");
				}

				/* Step 2: the frames buffered during output mode (at cache_size+i) are an
				 * incomplete NEXT chunk. Run them through the SAME motion-based selection as a
				 * normal chunk (static->1 frame, active->several) instead of dumping them ALL
				 * raw — raw-dumping a slow/static tail produced a burst of near-identical
				 * "repeating" frames at EOS. */
				guint tail = nvdsframeselector->output_count;
				if (tail >= 1) {
					/* Move buffered frames to the front of the cache (swap to keep GPU allocs). */
					for (guint i = 0; i < tail; i++) {
						NvBufSurface *tmp = nvdsframeselector->cached_frames[i];
						nvdsframeselector->cached_frames[i] = nvdsframeselector->cached_frames[nvdsframeselector->cache_size + i];
						nvdsframeselector->cached_frames[nvdsframeselector->cache_size + i] = tmp;
						nvdsframeselector->sad_values[i] = nvdsframeselector->sad_values[nvdsframeselector->cache_size + i];
						nvdsframeselector->cached_timestamps[i] = nvdsframeselector->cached_timestamps[nvdsframeselector->cache_size + i];
						nvdsframeselector->cached_frame_numbers[i] = nvdsframeselector->cached_frame_numbers[nvdsframeselector->cache_size + i];
					}
					nvdsframeselector->cached_frame_count = tail;
					nvdsframeselector->actual_selected_count = 0;
					gboolean tail_done = FALSE;
					if (nvdsframeselector->enable_motion_detection &&
					    nvdsframeselector->motion_detector_initialized && tail >= 2) {
						ChunkMotionVerdict mv;
						std::vector<MotionLevel> lv(tail, MOTION_STATIC);
						std::vector<int> mcv(tail, 0);
						nvdsframeselector->eff_of_interval =
							gst_nvds_effective_of_interval (nvdsframeselector, tail);
						if (analyzeChunkMotion((MotionDetectorHandle)nvdsframeselector->motion_detector_handle,
						                       nvdsframeselector->cached_frames, tail,
						                       nvdsframeselector->eff_of_interval,
						                       (const unsigned long long *)nvdsframeselector->cached_timestamps,
						                       nvdsframeselector->source_fps, &mv, lv.data(), mcv.data()) == 0) {
							nvdsframeselector->of_grid_cells = mv.gridTotalCells;
							if (gst_nvds_frame_selection_of_only_enabled (nvdsframeselector)) {
								gst_nvds_frame_selection_of_only(nvdsframeselector, lv.data(), mcv.data());
								tail_done = TRUE;
							}
							/* else: OF-only disabled -> tail_done stays FALSE -> single middle frame below */
						}
					}
					if (!tail_done) {
						nvdsframeselector->selected_indices[0] = tail / 2;
						nvdsframeselector->actual_selected_count = 1;
					}
					if (nvdsframeselector->equidistant_output)
						gst_nvds_frame_selector_apply_equidistant (nvdsframeselector);
					for (guint i = 0; i < nvdsframeselector->actual_selected_count; i++) {
						gst_nvds_frame_selector_push_cached_frame (nvdsframeselector, srcpad,
							nvdsframeselector->selected_indices[i], "eos_tail");
					}
				}

				/* fselect residency: first-frame-in → last-frame-out for this chunk. */
				if (nvdsframeselector->chunk_first_in_us != 0) {
					gint64 now_us = g_get_monotonic_time();
					gint64 total_us = now_us - nvdsframeselector->chunk_first_in_us;
					gint64 sel_us = (nvdsframeselector->chunk_eos_us != 0)
						? (now_us - nvdsframeselector->chunk_eos_us) : 0;
					gint64 in_us = total_us - sel_us;
					GST_INFO_OBJECT (nvdsframeselector, "FSELECT residency first-in->last-out = %.1f ms "
						"(input-phase first-in->EOS = %.1f ms [SAD %.1f ms], "
						"select-phase EOS->last-out = %.1f ms, output %u frames)",
						total_us / 1000.0,
						in_us / 1000.0, nvdsframeselector->chunk_sad_us / 1000.0,
						sel_us / 1000.0, nvdsframeselector->actual_selected_count);
					nvdsframeselector->chunk_first_in_us = 0;
					nvdsframeselector->chunk_eos_us = 0;
				}
				/* Reset state for potential new stream */
				nvdsframeselector->cached_frame_count = 0;
				nvdsframeselector->is_outputting = FALSE;
				nvdsframeselector->output_count = 0;
				nvdsframeselector->actual_selected_count = 0;

				/* Reset SAD values and timestamps for next stream */
				guint total_cache_size = nvdsframeselector->cache_size + nvdsframeselector->selection_count;
				for (guint i = 0; i < total_cache_size; i++) {
					nvdsframeselector->sad_values[i] = 0;
					nvdsframeselector->cached_timestamps[i] = GST_CLOCK_TIME_NONE;
					nvdsframeselector->cached_frame_numbers[i] = 0;
				}
			}
			/* Handle EOS in caching mode - If we have cached frames when EOS arrives, process them with SAD algorithm */
			else if (nvdsframeselector->cached_frame_count > 0) {
				GST_DEBUG_OBJECT (nvdsframeselector, "EOS RECEIVED, cached_frame_count: %u", nvdsframeselector->cached_frame_count);
				GST_INFO_OBJECT (nvdsframeselector,
					"EOS received with %u cached frames, processing with SAD algorithm",
					nvdsframeselector->cached_frame_count);

				/* Adjust selection count if we have fewer cached frames */
				guint actual_selection_count = nvdsframeselector->selection_count;
				if (actual_selection_count > nvdsframeselector->cached_frame_count) {
					actual_selection_count = nvdsframeselector->cached_frame_count;
				}

				/* Calculate FPS based on cached timestamps at EOS */
				if (nvdsframeselector->cached_frame_count > 1) {
					GstClockTime first_timestamp = nvdsframeselector->cached_timestamps[0];
					GstClockTime last_timestamp = nvdsframeselector->cached_timestamps[nvdsframeselector->cached_frame_count - 1];

					if (first_timestamp != GST_CLOCK_TIME_NONE && last_timestamp != GST_CLOCK_TIME_NONE && last_timestamp > first_timestamp) {
						GstClockTime timestamp_diff = last_timestamp - first_timestamp;
						/* Duration per frame = total duration / number of cached frames */
						gdouble duration_per_frame_sec = (gdouble)timestamp_diff / (gdouble)nvdsframeselector->cached_frame_count / GST_SECOND;
						gdouble fps = 1.0 / duration_per_frame_sec;
						GST_DEBUG_OBJECT(nvdsframeselector, "FPS: %f", fps);
					}
				}
				/* Reset algorithm to user-configured value before motion analysis for this chunk */
				nvdsframeselector->frame_selection_algorithm = nvdsframeselector->user_frame_selection_algorithm;

				/* OF-only selection (default). Falls back to the user-configured algorithm
				 * (RANGE_BASED/BASIC) if OF-only is disabled, motion detection is off, or
				 * analysis fails. */
				gboolean eos_blend_done = FALSE;
				if (nvdsframeselector->enable_motion_detection &&
				    nvdsframeselector->motion_detector_initialized &&
				    nvdsframeselector->cached_frame_count >= 2) {
					ChunkMotionVerdict motionVerdict;
					std::vector<MotionLevel> sample_levels(nvdsframeselector->cached_frame_count, MOTION_STATIC);
					std::vector<int> sample_mcells(nvdsframeselector->cached_frame_count, 0);
					nvdsframeselector->eff_of_interval =
						gst_nvds_effective_of_interval (nvdsframeselector, nvdsframeselector->cached_frame_count);
					if (analyzeChunkMotion((MotionDetectorHandle)nvdsframeselector->motion_detector_handle,
					                       nvdsframeselector->cached_frames,
					                       nvdsframeselector->cached_frame_count,
					                       nvdsframeselector->eff_of_interval,
					                       (const unsigned long long *)nvdsframeselector->cached_timestamps,
					                       nvdsframeselector->source_fps,
					                       &motionVerdict, sample_levels.data(), sample_mcells.data()) == 0) {
						GST_INFO_OBJECT (nvdsframeselector, "EOS MOTION avg=%.2f px/frame of_peak=%.2f level=%s highMotionCount=%d maxMovingCells=%d ofN=%u%s",
							motionVerdict.avgMagnitude,
							motionVerdict.maxSampleMagnitude, motionLevelToString(motionVerdict.overallMotionLevel),
							motionVerdict.highMotionCount, motionVerdict.maxMovingCells,
							nvdsframeselector->eff_of_interval,
							nvdsframeselector->optical_flow_interval == 0 ? " (auto)" : "");
						nvdsframeselector->of_grid_cells = motionVerdict.gridTotalCells;
						if (gst_nvds_frame_selection_of_only_enabled (nvdsframeselector)) {
							gst_nvds_frame_selection_of_only (nvdsframeselector, sample_levels.data(), sample_mcells.data());
							eos_blend_done = TRUE;
							GST_INFO_OBJECT (nvdsframeselector, "EOS OF-only -> %u frame(s)",
								nvdsframeselector->actual_selected_count);
						}
						/* else: OF-only disabled -> leave eos_blend_done FALSE -> RANGE_BASED/BASIC fallback below */
					} else {
						GST_WARNING_OBJECT (nvdsframeselector, "EOS motion analysis failed, using fallback algorithm");
					}
				}

				/* Report SAD range across the chunk */
				if (!eos_blend_done) {
					GST_INFO_OBJECT (nvdsframeselector, "EOS algorithm selected = %d", nvdsframeselector->frame_selection_algorithm);
					switch (nvdsframeselector->frame_selection_algorithm) {
						case FRAME_SELECTION_ALGO_BASIC:
							gst_nvds_frame_selection_basic_algo (nvdsframeselector);
							break;
						case FRAME_SELECTION_ALGO_RANGE_BASED:
						default:
							gst_nvds_frame_selection_algo_range_based (nvdsframeselector);
							break;
					}
				}

			/* Push selected cached frames downstream */
				GstPad *srcpad = GST_BASE_TRANSFORM_SRC_PAD (btrans);
				if (nvdsframeselector->equidistant_output)
					gst_nvds_frame_selector_apply_equidistant (nvdsframeselector);
				for (guint i = 0; i < nvdsframeselector->actual_selected_count; i++) {
					gst_nvds_frame_selector_push_cached_frame (nvdsframeselector, srcpad,
						nvdsframeselector->selected_indices[i], "eos_cached");
				}
			}

bypass_mode:
			/* fselect residency: first-frame-in → last-frame-out for this chunk. */
			if (nvdsframeselector->chunk_first_in_us != 0) {
				gint64 now_us = g_get_monotonic_time();
				gint64 total_us = now_us - nvdsframeselector->chunk_first_in_us;
				gint64 sel_us = (nvdsframeselector->chunk_eos_us != 0)
					? (now_us - nvdsframeselector->chunk_eos_us) : 0;
				gint64 in_us = total_us - sel_us;
				GST_INFO_OBJECT (nvdsframeselector, "FSELECT residency first-in->last-out = %.1f ms "
					"(input-phase first-in->EOS = %.1f ms [SAD %.1f ms], "
					"select-phase EOS->last-out = %.1f ms, output %u frames)",
					total_us / 1000.0,
					in_us / 1000.0, nvdsframeselector->chunk_sad_us / 1000.0,
					sel_us / 1000.0, nvdsframeselector->actual_selected_count);
				nvdsframeselector->chunk_first_in_us = 0;
				nvdsframeselector->chunk_eos_us = 0;
			}
			/* Reset cache state so the next chunk starts fresh.
			 * The outputting-mode EOS handler already does this,
			 * but the caching-mode EOS handler does not — causing the next
			 * chunk to inherit a stale cached_frame_count and trigger a
			 * spurious mid-chunk selection. */
			nvdsframeselector->cached_frame_count = 0;
			nvdsframeselector->is_outputting = FALSE;
			nvdsframeselector->output_count = 0;
			nvdsframeselector->actual_selected_count = 0;
			{
				guint total_cache_size = nvdsframeselector->cache_size + nvdsframeselector->selection_count;
				for (guint i = 0; i < total_cache_size; i++) {
					nvdsframeselector->sad_values[i] = 0;
					nvdsframeselector->cached_timestamps[i] = GST_CLOCK_TIME_NONE;
					nvdsframeselector->cached_frame_numbers[i] = 0;
				}
			}

			/* Chain to parent class to handle EOS properly */
			GST_INFO_OBJECT (nvdsframeselector, "EOS sent downstream");
			ret = GST_BASE_TRANSFORM_CLASS (parent_class)->sink_event (btrans, event);
			nvdsframeselector->frame_num = 0;
			nvdsframeselector->input_frame_count = 0;
			break;

		default:
			/* Chain to parent class for all other events */
			ret = GST_BASE_TRANSFORM_CLASS (parent_class)->sink_event (btrans, event);
			break;
	}

	return ret;
}

/**
 * Called when element recieves an input buffer from upstream element.
 */
static GstFlowReturn
gst_nvds_frame_selector_transform_ip (GstBaseTransform * btrans, GstBuffer * inbuf)
{
	GstNvDsFrameSelector *nvdsframeselector = GST_NVDS_FRAME_SELECTOR (btrans);
	nvdsframeselector->input_frame_count++;
	/* fselect residency timing: stamp the first input frame of each chunk. */
	if (nvdsframeselector->chunk_first_in_us == 0) {
		nvdsframeselector->chunk_first_in_us = g_get_monotonic_time();
		nvdsframeselector->chunk_sad_us = 0;  // reset per-chunk SAD accumulator
		GST_INFO_OBJECT (nvdsframeselector, "FSELECT first-frame-in PTS=%" GST_TIME_FORMAT,
			GST_TIME_ARGS(GST_BUFFER_PTS(inbuf)));
	}
	if (nvdsframeselector->bypass_mode) {
		GST_INFO_OBJECT (nvdsframeselector, "OUT [bypass] PTS=%" GST_TIME_FORMAT, GST_TIME_ARGS(GST_BUFFER_PTS(inbuf)));
		return GST_FLOW_OK;
	}
	NvBufSurface *in_surf = NULL;
	GstMapInfo in_map_info;

	nvdsframeselector->frame_num++;
	/* If frame selector is disabled, pass all frames through */
	if (!nvdsframeselector->enable_selector) {
		GST_INFO_OBJECT (nvdsframeselector, "OUT [disabled] frame=%lu PTS=%" GST_TIME_FORMAT,
			nvdsframeselector->frame_num, GST_TIME_ARGS(GST_BUFFER_PTS(inbuf)));
		return GST_FLOW_OK;
	}

	/* Get the NvBufSurface from the input buffer */
	if (!gst_buffer_map (inbuf, &in_map_info, GST_MAP_READWRITE)) {
		GST_ERROR_OBJECT (nvdsframeselector, "Failed to map input buffer");
		return GST_FLOW_ERROR;
	}

	in_surf = (NvBufSurface *) in_map_info.data;

	/* Validate NvBufSurface */
	if (!in_surf || in_surf->numFilled == 0) {
		GST_ERROR_OBJECT (nvdsframeselector, "Invalid NvBufSurface");
		gst_buffer_unmap (inbuf, &in_map_info);
		return GST_FLOW_ERROR;
	}

	{
		/* We're in caching mode - cache the incoming frame */
		guint sad_value = 0;

		/* Skip SAD calculation for BASIC algorithm - it only needs equidistant frames */
		/* Skip SAD for BASIC (equidistant) and for OF-only (default) — OF-only never reads
		 * sad_values, so computing per-frame SAD is wasted work. (If OF-only is disabled or
		 * OF analysis fails, the RANGE_BASED fallback would then have no SAD; acceptable
		 * since OF-only is the default path.) */
		gboolean skip_sad = (nvdsframeselector->frame_selection_algorithm == FRAME_SELECTION_ALGO_BASIC)
			|| gst_nvds_frame_selection_of_only_enabled (nvdsframeselector);

		/* Compute SAD against the previous frame (already in the cache) only if
		 * needed (not for BASIC). PERF: previously this used a dedicated
		 * previous_frame buffer that was deep-copied (NvBufSurfaceCopy) for EVERY
		 * input frame — ~150 full-NV12 GPU copies/chunk. The previous frame is
		 * already cached at index (count-1), so we SAD against it directly and
		 * skip the redundant copy. SAD values are identical. */
		if (!skip_sad && nvdsframeselector->cached_frame_count > 0 &&
		    nvdsframeselector->cached_frames[nvdsframeselector->cached_frame_count - 1]) {
			NvBufSurface *prev_surf = nvdsframeselector->cached_frames[nvdsframeselector->cached_frame_count - 1];
			unsigned char *current_pixels = (unsigned char*)in_surf->surfaceList[0].dataPtr;
			unsigned char *previous_pixels = (unsigned char*)prev_surf->surfaceList[0].dataPtr;

			if (current_pixels && previous_pixels) {
				bool is_device_memory = (in_surf->memType == NVBUF_MEM_CUDA_DEVICE ||
										in_surf->memType == NVBUF_MEM_CUDA_UNIFIED);
				gint64 _sad_t0 = g_get_monotonic_time();
				sad_value = runSAD_adaptive(nvdsframeselector, current_pixels, previous_pixels,
										in_surf->surfaceList[0].width, in_surf->surfaceList[0].height,
										is_device_memory);
				nvdsframeselector->chunk_sad_us += (g_get_monotonic_time() - _sad_t0);
			}
		}

		/* Create buffer for caching if not already created */
		if (!nvdsframeselector->cached_frames[nvdsframeselector->cached_frame_count]) {
			NvBufSurfaceCreateParams create_params;
			memset(&create_params, 0, sizeof(create_params));
			create_params.gpuId = nvdsframeselector->gpu_id;
			create_params.width = in_surf->surfaceList[0].width;
			create_params.height = in_surf->surfaceList[0].height;
			create_params.size = 0;
			create_params.colorFormat = in_surf->surfaceList[0].colorFormat;
			create_params.layout = in_surf->surfaceList[0].layout;
			create_params.memType = in_surf->memType;

			if (NvBufSurfaceCreate(&nvdsframeselector->cached_frames[nvdsframeselector->cached_frame_count], 1, &create_params) != 0) {
				GST_ERROR_OBJECT (nvdsframeselector, "Failed to create cached frame buffer");
				gst_buffer_unmap (inbuf, &in_map_info);
				return GST_FLOW_ERROR;
			}
		}

		/* Copy current frame to cache */
		if (NvBufSurfaceCopy(in_surf, nvdsframeselector->cached_frames[nvdsframeselector->cached_frame_count]) != 0) {
			GST_ERROR_OBJECT (nvdsframeselector, "Failed to copy frame to cache");
			gst_buffer_unmap (inbuf, &in_map_info);
			return GST_FLOW_ERROR;
		}

		/* Store SAD value, timestamp, and frame number */
		nvdsframeselector->sad_values[nvdsframeselector->cached_frame_count] = sad_value;
		nvdsframeselector->cached_timestamps[nvdsframeselector->cached_frame_count] = GST_BUFFER_PTS(inbuf);
		nvdsframeselector->cached_frame_numbers[nvdsframeselector->cached_frame_count] = nvdsframeselector->frame_num;

		GST_DEBUG_OBJECT (nvdsframeselector, "Cached frame #%lu at index %u with SAD %u and timestamp %" GST_TIME_FORMAT,
						 nvdsframeselector->frame_num, nvdsframeselector->cached_frame_count, sad_value,
						 GST_TIME_ARGS(GST_BUFFER_PTS(inbuf)));

		/* NOTE: no separate previous_frame copy — SAD for the next frame uses the
		 * frame just cached at index cached_frame_count (see SAD block above). */

		/* Increment cached frame count */
		nvdsframeselector->cached_frame_count++;

		/* Check if we have reached the cache_size limit for caching phase */
		if (nvdsframeselector->cached_frame_count == nvdsframeselector->cache_size) {
			/* Calculate FPS based on cached timestamps */
			if (nvdsframeselector->cache_size > 1) {
				GstClockTime first_timestamp = nvdsframeselector->cached_timestamps[0];
				GstClockTime last_timestamp = nvdsframeselector->cached_timestamps[nvdsframeselector->cache_size - 1];

				if (first_timestamp != GST_CLOCK_TIME_NONE && last_timestamp != GST_CLOCK_TIME_NONE && last_timestamp > first_timestamp) {
					GstClockTime timestamp_diff = last_timestamp - first_timestamp;
					/* Duration per frame = total duration / number of cached frames */
					gdouble duration_per_frame_sec = (gdouble)timestamp_diff / (gdouble)nvdsframeselector->cache_size / GST_SECOND;
					gdouble fps = 1.0 / duration_per_frame_sec;
					GST_DEBUG_OBJECT(nvdsframeselector, "FPS: %f", fps);
				}
		    }

		/* Analyze + select this full chunk. */
		gst_nvds_frame_selector_select_chunk (nvdsframeselector);

		/* Low-latency immediate push: emit ALL selected frames now, each via
		 * push_cached_frame() (which allocates its own output buffer), instead of
		 * dribbling one pick per subsequently-arriving input. The old drain overwrote
		 * each arriving in_surf to carry a single pick, so on a real-time / RTSP feed it
		 * added ~(picks x frame-interval) latency (~4-8 s) before this chunk's frames
		 * were all out. Pushing now removes that; the next window is collected normally
		 * at the head as frames arrive. Chunk boundaries and per-frame PTS are unchanged. */
		{
			GstPad *srcpad = GST_BASE_TRANSFORM_SRC_PAD (btrans);
			for (guint i = 0; i < nvdsframeselector->actual_selected_count; i++) {
				gst_nvds_frame_selector_push_cached_frame (nvdsframeselector, srcpad,
					nvdsframeselector->selected_indices[i], "output");
			}
		}

		/* Per-chunk residency (steady state, no EOS): first-frame-in -> all-picks-out. */
		if (nvdsframeselector->chunk_first_in_us != 0) {
			gint64 total_us = g_get_monotonic_time () - nvdsframeselector->chunk_first_in_us;
			GST_INFO_OBJECT (nvdsframeselector, "FSELECT chunk residency first-in->picks-out = %.1f ms (output %u frames)",
				total_us / 1000.0,
				nvdsframeselector->actual_selected_count);
			nvdsframeselector->chunk_first_in_us = 0;   /* re-stamp on the next chunk's first frame */
			nvdsframeselector->chunk_sad_us = 0;
		}

		/* Reset to caching mode for the next window. actual_selected_count returns to 0
		 * so a subsequent EOS is handled as a partial caching-mode chunk. */
		nvdsframeselector->cached_frame_count = 0;
		nvdsframeselector->actual_selected_count = 0;
		nvdsframeselector->output_count = 0;
		nvdsframeselector->is_outputting = FALSE;
		{
			guint total_cache_size = nvdsframeselector->cache_size + nvdsframeselector->selection_count;
			for (guint i = 0; i < total_cache_size; i++) {
				nvdsframeselector->sad_values[i] = 0;
				nvdsframeselector->cached_timestamps[i] = GST_CLOCK_TIME_NONE;
				nvdsframeselector->cached_frame_numbers[i] = 0;
			}
		}
		}

		GST_DEBUG_OBJECT (nvdsframeselector, "Cached and dropping frame %lu (frame %u of %u, SAD:%u)",
						 nvdsframeselector->frame_num, nvdsframeselector->cached_frame_count,
						 nvdsframeselector->cache_size, sad_value);

		gst_buffer_unmap (inbuf, &in_map_info);
		return GST_BASE_TRANSFORM_FLOW_DROPPED;
	}
}

/**
 * Boiler plate for registering a plugin and an element.
 */
static gboolean
nvds_frame_selector_plugin_init (GstPlugin * plugin)
{
	GST_DEBUG_CATEGORY_INIT (gst_nvds_frame_selector_debug, "nvdsframeselector", 0,
	  "nvdsframeselector plugin");

	return gst_element_register (plugin, "nvdsframeselector", GST_RANK_PRIMARY,
		  GST_TYPE_NVDS_FRAME_SELECTOR);
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
	GST_VERSION_MINOR,
	nvdsgst_frameselector,
	DESCRIPTION, nvds_frame_selector_plugin_init, "4.0", LICENSE, BINARY_PACKAGE, URL)
