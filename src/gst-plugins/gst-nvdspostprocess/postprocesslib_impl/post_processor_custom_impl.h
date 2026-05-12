/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef __POST_PROCESSOR_CUSTOM_IMPL_HPP__
#define __POST_PROCESSOR_CUSTOM_IMPL_HPP__
#include "post_processor_struct.h"

/*
 * C interfaces
 */

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * Type definition for the custom bounding box parsing function.
 *
 * @param[in]  outputLayersInfo A vector containing information on the output
 *                              layers of the model.
 * @param[in]  networkInfo      Network information.
 * @param[in]  detectionParams  Detection parameters required for parsing
 *                              objects.
 * @param[out] objectList       A reference to a vector in which the function
 *                              is to add parsed objects.
 */
typedef bool (* NvDsPostProcessParseCustomFunc) (
        std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        NvDsPostProcessParseDetectionParams const &detectionParams,
        std::vector<NvDsPostProcessObjectDetectionInfo> &objectList);

/**
 * Validates a custom parser function definition. Must be called
 * after defining the function.
 */
#define CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(customParseFunc) \
    static void checkFunc_ ## customParseFunc (NvDsPostProcessParseCustomFunc func = customParseFunc) \
        { checkFunc_ ## customParseFunc (); }; \
    extern "C" bool customParseFunc (std::vector<NvDsInferLayerInfo> const &outputLayersInfo, \
           NvDsInferNetworkInfo  const &networkInfo, \
           NvDsPostProcessParseDetectionParams const &detectionParams, \
           std::vector<NvDsPostProcessObjectDetectionInfo> &objectList);

/**
 * Type definition for the custom bounding box and instance mask parsing function.
 *
 * @param[in]  outputLayersInfo A vector containing information on the output
 *                              layers of the model.
 * @param[in]  networkInfo      Network information.
 * @param[in]  detectionParams  Detection parameters required for parsing
 *                              objects.
 * @param[out] objectList       A reference to a vector in which the function
 *                              is to add parsed objects and instance mask.
 */
typedef bool (* NvDsPostProcessInstanceMaskParseCustomFunc) (
        std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        NvDsPostProcessParseDetectionParams const &detectionParams,
        std::vector<NvDsPostProcessInstanceMaskInfo> &objectList);

/**
 * Validates a custom parser function definition. Must be called
 * after defining the function.
 */
#define CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(customParseFunc) \
    static void checkFunc_ ## customParseFunc (NvDsPostProcessInstanceMaskParseCustomFunc func = customParseFunc) \
        { checkFunc_ ## customParseFunc (); }; \
    extern "C" bool customParseFunc (std::vector<NvDsInferLayerInfo> const &outputLayersInfo, \
           NvDsInferNetworkInfo  const &networkInfo, \
           NvDsPostProcessParseDetectionParams const &detectionParams, \
           std::vector<NvDsPostProcessInstanceMaskInfo> &objectList);

/**
 * Type definition for the custom classifier output parsing function.
 *
 * @param[in]  outputLayersInfo  A vector containing information on the
 *                               output layers of the model.
 * @param[in]  networkInfo       Network information.
 * @param[in]  classifierThreshold
                                 Classification confidence threshold.
 * @param[out] attrList          A reference to a vector in which the function
 *                               is to add the parsed attributes.
 * @param[out] descString        A reference to a string object in which the
 *                               function may place a description string.
 */
typedef bool (* NvDsPostProcessClassiferParseCustomFunc) (
        std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        float classifierThreshold,
        std::vector<NvDsPostProcessAttribute> &attrList,
        std::string &descString);

/**
 * Validates the classifier custom parser function definition. Must be called
 * after defining the function.
 */
#define CHECK_CUSTOM_CLASSIFIER_PARSE_FUNC_PROTOTYPE(customParseFunc) \
    static void checkFunc_ ## customParseFunc (NvDsPostProcessClassiferParseCustomFunc func = customParseFunc) \
        { checkFunc_ ## customParseFunc (); }; \
    extern "C" bool customParseFunc (std::vector<NvDsInferLayerInfo> const &outputLayersInfo, \
           NvDsInferNetworkInfo  const &networkInfo, \
           float classifierThreshold, \
           std::vector<NvDsPostProcessAttribute> &attrList, \
           std::string &descString);

#ifdef __cplusplus
}
#endif
#endif
