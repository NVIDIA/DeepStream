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

#include <cstdlib>
#include <cstdio>

/* Forward declaration */
extern "C" unsigned int runSAD_CPU(const unsigned char* frameA,
                                   const unsigned char* frameB,
                                   int width, int height);

// CPU-based SAD computation for fallback/testing
extern "C" unsigned int runSAD_CPU(const unsigned char* frameA,
                                   const unsigned char* frameB,
                                   int width, int height) {
    if (!frameA || !frameB || width <= 0 || height <= 0) {
        printf("Error: Invalid input parameters to runSAD\n");
        return 0;
    }

    int totalPixels = width * height;
    unsigned int sad = 0;

    // Simple CPU-based SAD computation
    for (int i = 0; i < totalPixels; i++) {
        int diff = (int)frameA[i] - (int)frameB[i];
        sad += (diff < 0) ? -diff : diff;  // abs(diff)
    }

    return sad;
}
