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

#ifndef __MOTION_DETECTOR_H__
#define __MOTION_DETECTOR_H__

#include <cuda.h>
#include <cuda_runtime.h>
#include "nvbufsurface.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Motion level classification
 */
typedef enum {
    MOTION_STATIC = 0,      // No motion (< 0.5 px)
    MOTION_VERY_LOW,        // Minimal motion (0.5 - 1.0 px)
    MOTION_LOW,             // Low motion (2 - 5 px)
    MOTION_MEDIUM,          // Medium motion (5 - 15 px)
    MOTION_HIGH,            // High motion (15 - 40 px)
    MOTION_VERY_HIGH,       // Very high motion (40 - 100 px)
    MOTION_EXTREME          // Extreme motion / likely scene cut (> 100 px)
} MotionLevel;

/**
 * Chunk motion verdict
 */
typedef struct {
    MotionLevel overallMotionLevel;     // Overall motion classification
    double avgMagnitude;                 // Average motion magnitude across all samples
    double maxMagnitude;                 // Maximum motion detected
    double maxSampleMagnitude;           // Peak per-sample mean magnitude (px/frame, excl scene cuts)
    double minMagnitude;                 // Minimum motion detected
    int numSamples;                      // Number of frame pairs analyzed
    int maxMovingCells;                  // Peak coherent moving-cell count in any pair (localized subject size)
    int gridTotalCells;                  // Total OF grid cells (outWidth*outHeight) — for resolution-invariant thresholds
    int highMotionCount;                 // Number of high motion samples
    int lowMotionCount;                  // Number of low/static motion samples
    bool isHighMotionVideo;              // Final verdict: high motion video?
    bool hasSceneCuts;                   // Detected scene cuts?
} ChunkMotionVerdict;

/**
 * Opaque handle for motion detector context (per-instance)
 */
typedef struct MotionDetectorContext_s* MotionDetectorHandle;

/**
 * Initialize optical flow motion detector
 * Call once during plugin initialization
 * 
 * @param cuda_ctx CUDA context (can be NULL to use current context)
 * @param width Frame width
 * @param height Frame height
 * @param handle Output handle to motion detector instance
 * @return 0 on success, -1 on failure
 */
int initMotionDetector(CUcontext cuda_ctx, int width, int height, MotionDetectorHandle *handle);

/**
 * Cleanup optical flow motion detector
 * Call during plugin cleanup
 * 
 * @param handle Motion detector handle to cleanup
 */
void cleanupMotionDetector(MotionDetectorHandle handle);

/**
 * Analyze motion in chunk of cached frames
 * Samples every Nth frame and aggregates motion analysis
 * 
 * @param handle Motion detector handle
 * @param cached_frames Array of cached NvBufSurface frames (already on device)
 * @param num_cached Total number of cached frames
 * @param sample_interval Analyze every Nth frame (e.g., 15)
 * @param timestamps_ns Per-cached-frame PTS in nanoseconds. When provided (with
 *        source_fps>0), motion magnitude is normalized to px-per-SOURCE-frame
 *        using the real elapsed time between the two sampled frames, so the
 *        result is invariant to input decimation (e.g. RTVI feeding every Nth
 *        frame) and to the configured sample_interval. NULL → fall back to
 *        dividing by sample_interval (received-frame count).
 * @param source_fps Native stream frame rate. <=0 → fall back to frame-count.
 * @param verdict Output structure with motion analysis results
 * @param out_sample_levels Optional caller array (size >= num_cached) or NULL. When
 *        provided, out_sample_levels[i] receives the per-pair OF motion level for the
 *        transition starting at cached frame i (scene cuts reported as MOTION_HIGH).
 *        With sample_interval==1 this is a dense per-frame OF signal the caller can
 *        blend with per-frame SAD. Entries not written by a sample are left untouched.
 * @return 0 on success, -1 on failure
 */
int analyzeChunkMotion(
    MotionDetectorHandle handle,
    NvBufSurface **cached_frames,
    int num_cached,
    int sample_interval,
    const unsigned long long *timestamps_ns,
    double source_fps,
    ChunkMotionVerdict *verdict,
    MotionLevel *out_sample_levels,
    int *out_sample_moving_cells);

/**
 * Get string representation of motion level
 */
const char* motionLevelToString(MotionLevel level);

#ifdef __cplusplus
}
#endif

#endif /* __MOTION_DETECTOR_H__ */

