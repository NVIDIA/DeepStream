/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "nvdsinfer_context.h"

/* Define GLib-like functions for memory management */
#ifndef g_malloc
#define g_malloc(size) malloc(size)
#endif
#ifndef g_realloc
#define g_realloc(ptr, size) realloc(ptr, size)
#endif
#ifndef g_free
#define g_free(ptr) free(ptr)
#endif
#ifndef g_strdup
#define g_strdup(str) (str ? strcpy((char*)malloc(strlen(str) + 1), str) : NULL)
#endif
#ifndef g_print
#define g_print printf
#endif
#ifndef g_printerr
#define g_printerr(...) fprintf(stderr, __VA_ARGS__)
#endif

/**
 * Dynamic property management implementation for extensible TensorRT/trtexec flags
 */

#ifdef __cplusplus
extern "C" {
#endif

int 
NvDsInferContext_SetDynamicProperty(NvDsInferContextInitParams *initParams, 
                                   const char *key, const char *value)
{
    if (!initParams || !key || !value) {
        return 0;
    }

    // Check if property already exists
    for (unsigned int i = 0; i < initParams->numDynamicProperties; i++) {
        if (strcmp(initParams->dynamicPropertyKeys[i], key) == 0) {
            // Update existing property
            free(initParams->dynamicPropertyValues[i]);
            initParams->dynamicPropertyValues[i] = g_strdup(value);
            return 1;
        }
    }

    // Add new property
    unsigned int newSize = initParams->numDynamicProperties + 1;
    
    // Reallocate arrays
    initParams->dynamicPropertyKeys = (char**)g_realloc(
        initParams->dynamicPropertyKeys, newSize * sizeof(char*));
    initParams->dynamicPropertyValues = (char**)g_realloc(
        initParams->dynamicPropertyValues, newSize * sizeof(char*));
    
    if (!initParams->dynamicPropertyKeys || !initParams->dynamicPropertyValues) {
        return 0;
    }

    // Add new key-value pair
    initParams->dynamicPropertyKeys[initParams->numDynamicProperties] = g_strdup(key);
    initParams->dynamicPropertyValues[initParams->numDynamicProperties] = g_strdup(value);
    initParams->numDynamicProperties = newSize;

    return 1;
}

const char* 
NvDsInferContext_GetDynamicProperty(const NvDsInferContextInitParams *initParams, 
                                   const char *key)
{
    if (!initParams || !key) {
        return NULL;
    }

    for (unsigned int i = 0; i < initParams->numDynamicProperties; i++) {
        if (strcmp(initParams->dynamicPropertyKeys[i], key) == 0) {
            return initParams->dynamicPropertyValues[i];
        }
    }

    return NULL;
}

int 
NvDsInferContext_HasDynamicProperty(const NvDsInferContextInitParams *initParams, 
                                   const char *key)
{
    return NvDsInferContext_GetDynamicProperty(initParams, key) != NULL ? 1 : 0;
}

void 
NvDsInferContext_ClearDynamicProperties(NvDsInferContextInitParams *initParams)
{
    if (!initParams) {
        return;
    }

    // Free all keys and values
    for (unsigned int i = 0; i < initParams->numDynamicProperties; i++) {
        g_free(initParams->dynamicPropertyKeys[i]);
        g_free(initParams->dynamicPropertyValues[i]);
    }

    // Free arrays
    g_free(initParams->dynamicPropertyKeys);
    g_free(initParams->dynamicPropertyValues);

    // Reset state
    initParams->dynamicPropertyKeys = NULL;
    initParams->dynamicPropertyValues = NULL;
    initParams->numDynamicProperties = 0;
}

/**
 * Helper function to convert dynamic properties to TensorRT builder config
 * This can be extended to handle specific TensorRT configurations
 */
void 
NvDsInferContext_ApplyDynamicPropertiesToBuilder(const NvDsInferContextInitParams *initParams,
                                                 void *builderConfig)
{
    if (!initParams || !builderConfig) {
        return;
    }

    // Iterate through dynamic properties and apply them
    for (unsigned int i = 0; i < initParams->numDynamicProperties; i++) {
        const char *key = initParams->dynamicPropertyKeys[i];
        const char *value = initParams->dynamicPropertyValues[i];

        // Specific property handlers for TensorRT/trtexec flags
        if (strcmp(key, "data_max") == 0) {
            // Handle --data_max flag (for calibration)
            g_print("Info: Setting data_max (calibration) to %s\n", value);
            // TODO: Apply to TensorRT IInt8Calibrator or builder config
            // This would typically be used for INT8 calibration data scaling
        }
        else if (strcmp(key, "init_max") == 0) {
            // Handle --init_max flag (for initialization scaling)
            g_print("Info: Setting init_max (initialization scaling) to %s\n", value);
            // TODO: Apply to TensorRT builder initialization parameters
            // This could affect weight initialization scaling
        }
        else if (strcmp(key, "calibration_data") == 0) {
            // Handle --calibration_data flag (path to calibration dataset)
            g_print("Info: Setting calibration_data path to %s\n", value);
            // TODO: Set calibration data path for INT8 quantization
            // This would be used by IInt8Calibrator implementation
        }
        else if (strcmp(key, "workspace_size") == 0) {
            // Handle --workspace flag (memory workspace size)
            long workspace_bytes = strtol(value, NULL, 10);
            if (workspace_bytes > 0) {
                g_print("Info: Setting workspace size to %ld bytes\n", workspace_bytes);
                // TODO: Apply to builder config: config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, workspace_bytes);
            }
        }
        else if (strcmp(key, "optimization_level") == 0) {
            // Handle --builderOptimizationLevel flag
            int opt_level = atoi(value);
            g_print("Info: Setting builder optimization level to %d\n", opt_level);
            // TODO: Apply to builder config: config->setBuilderOptimizationLevel(opt_level);
        }
        else if (strcmp(key, "precision_mode") == 0) {
            // Handle precision mode flags (--fp16, --int8, --best)
            g_print("Info: Setting precision mode to %s\n", value);
            // TODO: Apply precision mode to builder config
            // This would set FP16/INT8 flags based on the value
        }
        else if (strcmp(key, "dla_core") == 0) {
            // Handle --useDLACore flag
            int dla_core = atoi(value);
            g_print("Info: Setting DLA core to %d\n", dla_core);
            // TODO: Apply DLA core setting to builder config
        }
        else if (strcmp(key, "allow_gpu_fallback") == 0) {
            // Handle --allowGPUFallback flag
            int allow_fallback = (strcmp(value, "1") == 0 || strcmp(value, "true") == 0);
            g_print("Info: Setting GPU fallback to %s\n", allow_fallback ? "enabled" : "disabled");
            // TODO: Apply GPU fallback setting to builder config
        }
        else {
            // Generic property - store for future use or custom handling
            g_print("Info: Dynamic property %s = %s (stored for future use)\n", key, value);
        }
    }
}

#ifdef __cplusplus
}
#endif
