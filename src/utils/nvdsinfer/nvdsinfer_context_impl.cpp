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

#include <dlfcn.h>
#include <unistd.h>
#include <array>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>

#include "nvtx3/nvToolsExtCudaRt.h"

#include <NvInferPlugin.h>
#include <NvOnnxParser.h>

#include "nvdsinfer_context_impl.h"
#include "nvdsinfer_conversion.h"
#include "nvdsinfer_func_utils.h"
#include "nvdsinfer_model_builder.h"

#include "cuda.h"
#include "cuda_runtime_api.h"

/* This file contains the implementation of the INvDsInferContext implementation.
 * The pre- and post- processing implementations are also in this file.
 */

/* Pair data type for returning input back to caller. */
using NvDsInferReturnInputPair = std::pair<NvDsInferContextReturnInputAsyncFunc, void *>;

static const int WORKSPACE_SIZE = 450 * 1024 * 1024;

namespace nvdsinfer {

/*
 * TensorRT INT8 Calibration implementation. This implementation requires
 * pre-generated INT8 Calibration Tables. Please refer TensorRT documentation
 * for information on the calibration tables and the procedure for creating the
 * tables.
 *
 * Since this implementation only reads from pre-generated calibration tables,
 * readCalibrationCache is requires to be implemented.
 */
class NvDsInferInt8Calibrator : public nvinfer1::IInt8EntropyCalibrator2
{
public:
    NvDsInferInt8Calibrator(std::string calibrationTableFile)
        : m_CalibrationTableFile(calibrationTableFile) {}

    ~NvDsInferInt8Calibrator()
    {
    }

    int
    getBatchSize() const noexcept override
    {
        return 0;
    }

    bool
    getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override
    {
        return false;
    }

    /* Reads calibration table file contents into a buffer and returns a pointer
     * to the buffer.
     */
    const void*
    readCalibrationCache(size_t& length) noexcept override
    {
        m_CalibrationCache.clear();
        std::ifstream input(m_CalibrationTableFile, std::ios::binary);
        input >> std::noskipws;
        if (input.good())
            copy(std::istream_iterator<char>(input),
                std::istream_iterator<char>(),
                std::back_inserter(m_CalibrationCache));

        length = m_CalibrationCache.size();
        return length ? m_CalibrationCache.data() : nullptr;
    }

    void
    writeCalibrationCache(const void* cache, size_t length) noexcept override
    {
    }

private:
    std::string m_CalibrationTableFile;
    std::vector<char> m_CalibrationCache;
};

/* Cuda callback function for returning input back to client. */
static void
returnInputCudaCallback(cudaStream_t stream,  cudaError_t status, void*  userData)
{
    NvDsInferReturnInputPair *pair = (NvDsInferReturnInputPair  *) userData;
    pair->first(pair->second);
    delete pair;
}

InferPreprocessor::InferPreprocessor(const NvDsInferNetworkInfo& info,
    NvDsInferFormat format, const NvDsInferBatchDimsLayerInfo& layerInfo,
    int id)
    : m_UniqueID(id),
      m_NetworkInfo(info),
      m_NetworkInputFormat(format),
      m_NetworkInputLayer(layerInfo)
{
}

bool
InferPreprocessor::setScaleOffsets(float scale, const std::vector<float>& offsets)
{
    if (!offsets.empty() &&
        m_NetworkInfo.channels != (uint32_t)offsets.size())
    {
        return false;
    }

    m_Scale = scale;
    if (!offsets.empty())
    {
        m_ChannelMeans.assign(
            offsets.begin(), offsets.begin() + m_NetworkInfo.channels);
    }
    return true;
}

bool
InferPreprocessor::setMeanFile(const std::string& file)
{
    if (!file_accessible(file))
        return false;
    m_MeanFile = file;
    return true;
}

bool
InferPreprocessor::setInputOrder(const NvDsInferTensorOrder order)
{
    m_InputOrder = order;
    return true;
}

/* Read the mean image ppm file and copy the mean image data to the mean
 * data buffer allocated on the device memory.
 */
NvDsInferStatus
InferPreprocessor::readMeanImageFile()
{
    std::ifstream infile(m_MeanFile, std::ifstream::binary);
    size_t size =
        m_NetworkInfo.width * m_NetworkInfo.height * m_NetworkInfo.channels;
    uint8_t tempMeanDataChar[size];
    float tempMeanDataFloat[size];
    cudaError_t cudaReturn;

    if (!infile.good())
    {
        printError("Could not open mean image file '%s'", safeStr(m_MeanFile));
        return NVDSINFER_CONFIG_FAILED;
    }

    std::string magic, max;
    unsigned int h, w;
    infile >> magic >> w >> h >> max;

    if (magic != "P3" && magic != "P6")
    {
        printError("Magic PPM identifier check failed");
        return NVDSINFER_CONFIG_FAILED;
    }

    if (w != m_NetworkInfo.width || h != m_NetworkInfo.height)
    {
        printError(
            "Mismatch between ppm mean image resolution(%d x %d) and "
            "network resolution(%d x %d)",
            w, h, m_NetworkInfo.width, m_NetworkInfo.height);
        return NVDSINFER_CONFIG_FAILED;
    }

    infile.get();
    infile.read((char*)tempMeanDataChar, size);
    if (infile.gcount() != (int)size || infile.fail())
    {
        printError("Failed to read sufficient bytes from mean file");
        return NVDSINFER_CONFIG_FAILED;
    }

    for (size_t i = 0; i < size; i++)
    {
        tempMeanDataFloat[i] = (float)tempMeanDataChar[i];
    }

    assert(m_MeanDataBuffer);
    cudaReturn = cudaMemcpy(m_MeanDataBuffer->ptr(), tempMeanDataFloat,
        size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaReturn != cudaSuccess)
    {
        printError("Failed to copy mean data to mean data buffer (%s)",
            cudaGetErrorName(cudaReturn));
        return NVDSINFER_CUDA_ERROR;
    }

    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
InferPreprocessor::allocateResource()
{
    if (!m_MeanFile.empty() || m_ChannelMeans.size() > 0)
    {
        /* Mean Image File specified. Allocate the mean image buffer on device
         * memory. */
        m_MeanDataBuffer = std::make_unique<CudaDeviceBuffer>(
               (size_t) m_NetworkInfo.width * m_NetworkInfo.height * m_NetworkInfo.channels *
                sizeof(float));

        if (!m_MeanDataBuffer || !m_MeanDataBuffer->ptr())
        {
            printError("Failed to allocate cuda buffer for mean image");
            return NVDSINFER_CUDA_ERROR;
        }
    }

    /* Read the mean image file (PPM format) if specified and copy the
     * contents into the buffer. */
    if (!m_MeanFile.empty())
    {
        if (!file_accessible(m_MeanFile))
        {
            printError(
                "Cannot access mean image file '%s'", safeStr(m_MeanFile));
            return NVDSINFER_CONFIG_FAILED;
        }
        NvDsInferStatus status = readMeanImageFile();
        if (status != NVDSINFER_SUCCESS)
        {
            printError("Failed to read mean image file");
            return status;
        }
    }
    /* Create the mean data buffer from per-channel offsets. */
    else if (m_ChannelMeans.size() > 0)
    {
        /* Make sure the number of offsets are equal to the number of input
         * channels. */
        if ((uint32_t)m_ChannelMeans.size() != m_NetworkInfo.channels)
        {
            printError(
                "Number of offsets(%d) not equal to number of input "
                "channels(%d)",
                (int)m_ChannelMeans.size(), m_NetworkInfo.channels);
            return NVDSINFER_CONFIG_FAILED;
        }

        std::vector<float> meanData(m_NetworkInfo.channels *
                                    m_NetworkInfo.width * m_NetworkInfo.height);
        for (size_t j = 0; j < m_NetworkInfo.width * m_NetworkInfo.height;
             j++)
        {
            for (size_t i = 0; i < m_NetworkInfo.channels; i++)
            {
                meanData[j * m_NetworkInfo.channels + i] = m_ChannelMeans[i];
            }
        }
        cudaError_t cudaReturn =
            cudaMemcpy(m_MeanDataBuffer->ptr(), meanData.data(),
                meanData.size() * sizeof(float), cudaMemcpyHostToDevice);
        if (cudaReturn != cudaSuccess)
        {
            printError("Failed to copy mean data to mean data cuda buffer(%s)",
                cudaGetErrorName(cudaReturn));
            return NVDSINFER_CUDA_ERROR;
        }
    }

    /* Create the cuda stream on which pre-processing jobs will be executed. */
    m_PreProcessStream = std::make_unique<CudaStream>(cudaStreamNonBlocking);
    if (!m_PreProcessStream || !m_PreProcessStream->ptr())
    {
        printError("Failed to create preprocessor cudaStream");
        return NVDSINFER_TENSORRT_ERROR;
    }
    std::string nvtx_name =
        "nvdsinfer_preprocess_uid=" + std::to_string(m_UniqueID);
    nvtxNameCudaStreamA(*m_PreProcessStream, nvtx_name.c_str());

    /* Cuda event to synchronize between completion of the pre-processing
     * kernels and enqueuing the next set of binding buffers for inference. */
    m_PreProcessCompleteEvent =
        std::make_shared<CudaEvent>(cudaEventDisableTiming);
    if (!m_PreProcessCompleteEvent || !m_PreProcessCompleteEvent->ptr())
    {
        printError("Failed to create cuda preprocessing complete event");
        return NVDSINFER_CUDA_ERROR;
    }
    nvtx_name =
        "nvdsinfer_preprocess_complete_uid=" + std::to_string(m_UniqueID);
    nvtxNameCudaEventA(*m_PreProcessCompleteEvent, nvtx_name.c_str());

    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
InferPreprocessor::syncStream()
{
    if (m_PreProcessStream)
    {
        if (cudaSuccess != cudaStreamSynchronize(*m_PreProcessStream))
            return NVDSINFER_CUDA_ERROR;
    }
    return NVDSINFER_SUCCESS;
}

NvDsInferStatus InferPreprocessor::transform(
    NvDsInferContextBatchInput& batchInput, void* devBuf,
    CudaStream& mainStream, CudaEvent* waitingEvent)
{
    unsigned int batchSize = batchInput.numInputFrames;
    NvDsInferConvertFcn convertFcn = nullptr;
    NvDsInferConvertFcnFloat convertFcnFloat = nullptr;

    /* Make the future jobs on the stream wait till the infer engine consumes
     * the previous contents of the input binding buffer. */
    if (waitingEvent)
    {
        RETURN_CUDA_ERR(
            cudaStreamWaitEvent(*m_PreProcessStream, *waitingEvent, 0),
            "Failed to make stream wait on event");
    }

    /* Find the required conversion function. */
    switch (m_NetworkInputFormat)
    {
        case NvDsInferFormat_RGB:
            switch (batchInput.inputFormat)
            {
                case NvDsInferFormat_RGB:
                    switch (m_InputOrder)
                    {
                        case NvDsInferTensorOrder_kNCHW:
                            convertFcn = NvDsInferConvert_C3ToP3Float;
                            break;
                        case NvDsInferTensorOrder_kNHWC:
                            convertFcn = NvDsInferConvert_C3ToL3Float;
                            break;
                        default:
                            break;
                    }
                    break;
                case NvDsInferFormat_BGR:
                    switch (m_InputOrder)
                    {
                        case NvDsInferTensorOrder_kNCHW:
                            convertFcn = NvDsInferConvert_C3ToP3RFloat;
                            break;
                        case NvDsInferTensorOrder_kNHWC:
                            convertFcn = NvDsInferConvert_C3ToL3RFloat;
                            break;
                        default:
                            break;
                    }
                    break;
                case NvDsInferFormat_RGBA:
                    switch (m_InputOrder)
                    {
                        case NvDsInferTensorOrder_kNCHW:
                            convertFcn = NvDsInferConvert_C4ToP3Float;
                            break;
                        case NvDsInferTensorOrder_kNHWC:
                            convertFcn = NvDsInferConvert_C4ToL3Float;
                            break;
                        default:
                            break;
                    }
                    break;
                case NvDsInferFormat_BGRx:
                    switch (m_InputOrder)
                    {
                        case NvDsInferTensorOrder_kNCHW:
                            convertFcn = NvDsInferConvert_C4ToP3RFloat;
                            break;
                        case NvDsInferTensorOrder_kNHWC:
                            convertFcn = NvDsInferConvert_C4ToL3RFloat;
                            break;
                        default:
                            break;
                    }
                    break;
                default:
                    printError("Input format conversion is not supported");
                    return NVDSINFER_INVALID_PARAMS;
            }
            break;
        case NvDsInferFormat_BGR:
            switch (batchInput.inputFormat)
            {
                case NvDsInferFormat_RGB:
                    switch (m_InputOrder)
                    {
                        case NvDsInferTensorOrder_kNCHW:
                            convertFcn = NvDsInferConvert_C3ToP3RFloat;
                            break;
                        case NvDsInferTensorOrder_kNHWC:
                            convertFcn = NvDsInferConvert_C3ToL3RFloat;
                            break;
                        default:
                            break;
                    }
                    break;
                case NvDsInferFormat_BGR:
                    switch (m_InputOrder)
                    {
                        case NvDsInferTensorOrder_kNCHW:
                            convertFcn = NvDsInferConvert_C3ToP3Float;
                            break;
                        case NvDsInferTensorOrder_kNHWC:
                            convertFcn = NvDsInferConvert_C3ToL3Float;
                            break;
                        default:
                            break;
                    }
                    break;
                case NvDsInferFormat_RGBA:
                    switch (m_InputOrder)
                    {
                        case NvDsInferTensorOrder_kNCHW:
                            convertFcn = NvDsInferConvert_C4ToP3RFloat;
                            break;
                        case NvDsInferTensorOrder_kNHWC:
                            convertFcn = NvDsInferConvert_C4ToL3RFloat;
                            break;
                        default:
                            break;
                    }
                    break;
                case NvDsInferFormat_BGRx:
                    switch (m_InputOrder)
                    {
                        case NvDsInferTensorOrder_kNCHW:
                            convertFcn = NvDsInferConvert_C4ToP3Float;
                            break;
                        case NvDsInferTensorOrder_kNHWC:
                            convertFcn = NvDsInferConvert_C4ToL3Float;
                            break;
                        default:
                            break;
                    }
                    break;
                default:
                    printError("Input format conversion is not supported");
                    return NVDSINFER_INVALID_PARAMS;
            }
            break;
        case NvDsInferFormat_GRAY:
            if (batchInput.inputFormat != NvDsInferFormat_GRAY)
            {
                printError("Input frame format is not GRAY.");
                return NVDSINFER_INVALID_PARAMS;
            }
            convertFcn = NvDsInferConvert_C1ToP1Float;
            break;
        case NvDsInferFormat_Tensor:
            if (batchInput.inputFormat != NvDsInferFormat_Tensor)
            {
                printError("Input frame format is not Tensor.");
                return NVDSINFER_INVALID_PARAMS;
            }
            convertFcnFloat = NvDsInferConvert_FtFTensor;
            break;
        default:
            printError("Unsupported network input format");
            return NVDSINFER_INVALID_PARAMS;
    }

    /* For each frame in the input batch convert/copy to the input binding
     * buffer. */
    for (unsigned int i = 0; i < batchSize; i++)
    {
        float* outPtr =
            (float*)devBuf + i * m_NetworkInputLayer.inferDims.numElements;

        if (convertFcn) {
            /* Input needs to be pre-processed. */
            convertFcn(outPtr, (unsigned char*)batchInput.inputFrames[i],
                m_NetworkInfo.width, m_NetworkInfo.height, batchInput.inputPitch,
                m_Scale, m_MeanDataBuffer.get() ? m_MeanDataBuffer->ptr<float>() : nullptr,
                *m_PreProcessStream);
        } else if (convertFcnFloat) {
            /* Input needs to be pre-processed. */
            convertFcnFloat(outPtr, (float *)batchInput.inputFrames[i],
                m_NetworkInfo.width, m_NetworkInfo.height, batchInput.inputPitch,
                m_Scale, m_MeanDataBuffer.get() ? m_MeanDataBuffer->ptr<float>() : nullptr,
                *m_PreProcessStream);
        }
    }

    /* Inputs can be returned back once pre-processing is complete. */
    if (batchInput.returnInputFunc)
    {
        RETURN_CUDA_ERR(
            cudaStreamAddCallback(*m_PreProcessStream, returnInputCudaCallback,
                new NvDsInferReturnInputPair(
                    batchInput.returnInputFunc, batchInput.returnFuncData),
                0),
            "Failed to add cudaStream callback for returning input buffers");
    }

    /* Record CUDA event to synchronize the completion of pre-processing
     * kernels. */
    RETURN_CUDA_ERR(
        cudaEventRecord(*m_PreProcessCompleteEvent, *m_PreProcessStream),
        "Failed to record cuda event");

    RETURN_CUDA_ERR(
        cudaStreamWaitEvent(mainStream, *m_PreProcessCompleteEvent, 0),
        "Failed to make mainstream wait for preprocess event");
    return NVDSINFER_SUCCESS;
}

/* Parse the labels file and extract the class label strings. For format of
 * the labels file, please refer to the custom models section in the
 * DeepStreamSDK documentation.
 */
NvDsInferStatus
InferPostprocessor::parseLabelsFile(const std::string& labelsFilePath)
{
    std::ifstream labels_file(labelsFilePath);
    std::string delim{';'};
    if (!labels_file.is_open())
    {
        printError("Could not open labels file:%s", safeStr(labelsFilePath));
        return NVDSINFER_CONFIG_FAILED;
    }
    while (labels_file.good() && !labels_file.eof())
    {
        std::string line, word;
        std::vector<std::string> l;
        size_t pos = 0, oldpos = 0;

        std::getline(labels_file, line, '\n');
        if (line.empty())
            continue;

        while ((pos = line.find(delim, oldpos)) != std::string::npos)
        {
            word = line.substr(oldpos, pos - oldpos);
            l.push_back(word);
            oldpos = pos + delim.length();
        }
        l.push_back(line.substr(oldpos));
        m_Labels.push_back(l);
    }

    if (labels_file.bad())
    {
        printError("Failed to parse labels file:%s, iostate:%d",
            safeStr(labelsFilePath), (int)labels_file.rdstate());
        return NVDSINFER_CONFIG_FAILED;
    }
    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
InferPostprocessor::allocDeviceResource()
{
    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
InferPostprocessor::initResource(const NvDsInferContextInitParams& initParams)
{
    m_CopyInputToHostBuffers = initParams.copyInputToHostBuffers;

    m_disableOutputHostCopy = initParams.disableOutputHostCopy;
    m_DumpOpTensor = initParams.dumpOpTensor;
    m_OverwriteOpTensor = initParams.overwriteOpTensor;
    if (m_OverwriteOpTensor) {
        for (unsigned int i = 0; i < initParams.numOutputLayers; i++) {
            std::pair<std::string, int> file_pair;
            std::string file_name = initParams.opTensorFilePath[i];
            std::ifstream *input_file = new std::ifstream(file_name, std::ifstream::binary);
            file_pair = std::make_pair(file_name, i);
            m_OverwriteOpTensorFilePairs.push_back(file_pair);
            m_OverwriteOpTensorFiles.push_back(input_file);
        }
    }

    if (!string_empty(initParams.labelsFilePath))
    {
        RETURN_NVINFER_ERROR(parseLabelsFile(initParams.labelsFilePath),
            "parse label file:%s failed", initParams.labelsFilePath);
    }

    RETURN_NVINFER_ERROR(
        allocDeviceResource(), "allocate device resource failed");
    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
InferPostprocessor::copyBuffersToHostMemory(NvDsInferBatch& batch, CudaStream& mainStream)
{
    assert(m_AllLayerInfo.size());
    /* Queue the copy of output contents from device to host memory after the
     * infer completion event. */
    for (size_t i = 0; i < m_AllLayerInfo.size(); i++)
    {
        NvDsInferLayerInfo& info = m_AllLayerInfo[i];
        assert(info.inferDims.numElements > 0);

        if (!info.isInput && needOutputCopyB4Processing())
        {
            RETURN_CUDA_ERR(
                cudaMemcpyAsync(batch.m_HostBuffers[info.bindingIndex]->ptr(),
                    batch.m_DeviceBuffers[info.bindingIndex],
                    getElementSize(info.dataType) * info.inferDims.numElements *
                        batch.m_BatchSize,
                    cudaMemcpyDeviceToHost, mainStream),
                "postprocessing cudaMemcpyAsync for output buffers failed");
        }
        else if (needInputCopy()&&info.isInput)
        {
            RETURN_CUDA_ERR(
                cudaMemcpyAsync(batch.m_HostBuffers[info.bindingIndex]->ptr(),
                    batch.m_DeviceBuffers[info.bindingIndex],
                    getElementSize(info.dataType) * info.inferDims.numElements *
                        batch.m_BatchSize,
                    cudaMemcpyDeviceToHost, mainStream),
                "postprocessing cudaMemcpyAsync for input buffers failed");
        }
    }

    /* Record CUDA event to later synchronize for the copy to actually
     * complete. */
    if (batch.m_OutputCopyDoneEvent)
    {
        RETURN_CUDA_ERR(cudaEventRecord(*batch.m_OutputCopyDoneEvent, mainStream),
            "Failed to record batch cuda copy-complete-event");
    }

    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
InferPostprocessor::postProcessHost(NvDsInferBatch& batch,
        NvDsInferContextBatchOutput& batchOutput)
{
    batchOutput.frames = new NvDsInferFrameOutput[batch.m_BatchSize];
    batchOutput.numFrames = batch.m_BatchSize;

    /* For each frame in the current batch, parse the output and add the frame
     * output to the batch output. The number of frames output in one batch
     * will be equal to the number of frames present in the batch during queuing
     * at the input.
     */
    for (unsigned int index = 0; index < batch.m_BatchSize; index++)
    {
        NvDsInferFrameOutput& frameOutput = batchOutput.frames[index];
        frameOutput.outputType = NvDsInferNetworkType_Other;

        /* Calculate the pointer to the output for each frame in the batch for
         * each output layer buffer. The NvDsInferLayerInfo vector for output
         * layers is passed to the output parsing function. */
        for (unsigned int i = 0; i < m_OutputLayerInfo.size(); i++)
        {
            NvDsInferLayerInfo& info = m_OutputLayerInfo[i];
            if(needOutputCopyB4Processing()){
                info.buffer =
                    (void*)(batch.m_HostBuffers[info.bindingIndex]->ptr<uint8_t>() +
                            info.inferDims.numElements *
                                getElementSize(info.dataType) * index);
            }
            else{
                info.buffer =
                    (void*)((uint8_t *)(batch.m_DeviceBuffers[info.bindingIndex]) +
                            info.inferDims.numElements *
                                getElementSize(info.dataType) * index);
            }
            if (m_DumpOpTensor) {
                uint32_t dump_size = m_NetworkInfo.width*m_NetworkInfo.height*m_NetworkInfo.channels;
                std::string file_path, layer_name;
                layer_name = info.layerName;
                for (auto & element : m_DumpOpTensorFiles)
                {
                    if (layer_name == element.first) {
                        file_path = element.second;
                        break;
                    }
                }
                if (file_path.empty()) {
                    std::pair<std::string, std::string> file_pair;
                    file_path = layer_name + "_op_tensor.bin";
                    std::replace(file_path.begin(), file_path.end(), '/', '-');
                    file_pair = std::make_pair(layer_name, file_path);
                    m_DumpOpTensorFiles.push_back(file_pair);
                }
                std::ofstream dump_op_file(file_path, std::ios_base::app);
                dump_op_file.write((char*) info.buffer, dump_size*4);
                dump_op_file.close();
            }
            if (m_OverwriteOpTensor) {
                uint32_t dump_size = m_NetworkInfo.width*m_NetworkInfo.height*m_NetworkInfo.channels;
                std::string layer_name = info.layerName;
                for (auto & element:m_OverwriteOpTensorFilePairs)
                {
                    std::replace(element.first.begin(), element.first.end(), '-', '/');
                    std::string sub_file_name = element.first.substr(0, layer_name.size());
                    if (layer_name == sub_file_name) {
                        int index = element.second;
                        m_OverwriteOpTensorFiles[index]->read((char *) info.buffer, dump_size*4);
                        break;
                    }
                }
            }
        }

        RETURN_NVINFER_ERROR(parseEachBatch(m_OutputLayerInfo, frameOutput),
            "Infer context initialize inference info failed");
    }

    /* Fill the host buffers information in the output. */
    batchOutput.numHostBuffers = m_AllLayerInfo.size();
    batchOutput.hostBuffers = new void*[m_AllLayerInfo.size()];
    for (size_t i = 0; i < batchOutput.numHostBuffers; i++)
    {
        batchOutput.hostBuffers[i] =
            batch.m_HostBuffers[i] ? batch.m_HostBuffers[i]->ptr() : nullptr;
    }

    batchOutput.numOutputDeviceBuffers = m_OutputLayerInfo.size();
    batchOutput.outputDeviceBuffers = new void*[m_OutputLayerInfo.size()];
    for (size_t i = 0; i < batchOutput.numOutputDeviceBuffers; i++)
    {
        batchOutput.outputDeviceBuffers[i] =
            batch.m_DeviceBuffers[m_OutputLayerInfo[i].bindingIndex];
    }

    /* Mark the set of host buffers as not with the context. */
    batch.m_BuffersWithContext = false;

    return NVDSINFER_SUCCESS;
}

void
InferPostprocessor::freeBatchOutput(NvDsInferContextBatchOutput& batchOutput)
{
    /* Free memory allocated in dequeueOutputBatch */
    for (unsigned int i = 0; i < batchOutput.numFrames; i++)
    {
        releaseFrameOutput(batchOutput.frames[i]);
    }

    delete[] batchOutput.frames;
    delete[] batchOutput.hostBuffers;
    delete[] batchOutput.outputDeviceBuffers;
}

/**
 * Clean up and free all resources
 */
InferPostprocessor::~InferPostprocessor() {
    while (!m_OverwriteOpTensorFiles.empty()) {
        std::ifstream *ptr = m_OverwriteOpTensorFiles.back();
        m_OverwriteOpTensorFiles.pop_back();
        ptr->close();
    }
}

NvDsInferStatus
DetectPostprocessor::initResource(const NvDsInferContextInitParams& initParams)
{
    RETURN_NVINFER_ERROR(InferPostprocessor::initResource(initParams),
        "init post processing resource failed");

    m_UseDBScan = initParams.useDBScan;
    if(m_UseDBScan)
        printError(" 'useDBScan' parameter has been deprecated. Use 'clusterMode' instead.");

    m_ClusterMode = initParams.clusterMode;

    m_NumDetectedClasses = initParams.numDetectedClasses;
    if (initParams.numDetectedClasses > 0 &&
        initParams.perClassDetectionParams == nullptr)
    {
        printError(
            "NumDetectedClasses > 0 but PerClassDetectionParams array not "
            "specified");
        return NVDSINFER_CONFIG_FAILED;
    }

    m_PerClassDetectionParams.assign(initParams.perClassDetectionParams,
        initParams.perClassDetectionParams + m_NumDetectedClasses);
    m_DetectionParams.numClassesConfigured = initParams.numDetectedClasses;
    m_DetectionParams.perClassPreclusterThreshold.resize(initParams.numDetectedClasses);
    m_DetectionParams.perClassPostclusterThreshold.resize(initParams.numDetectedClasses);

    /* Resize the per class vector to the number of detected classes. */
    m_PerClassObjectList.resize(initParams.numDetectedClasses);
#ifdef WITH_OPENCV
    if (m_ClusterMode == NVDSINFER_CLUSTER_GROUP_RECTANGLES)
    {
        m_PerClassCvRectList.resize(initParams.numDetectedClasses);
    }
#endif

    /* Fill the class thresholds in the m_DetectionParams structure. This
     * will be required during parsing. */
    for (unsigned int i = 0; i < initParams.numDetectedClasses; i++)
    {
        m_DetectionParams.perClassPreclusterThreshold[i] =
            m_PerClassDetectionParams[i].preClusterThreshold;
        m_DetectionParams.perClassPostclusterThreshold[i] =
            m_PerClassDetectionParams[i].postClusterThreshold;
    }

    /* If custom parse function is specified get the function address from the
     * custom library. */
    if (m_CustomLibHandle &&
        !string_empty(initParams.customBBoxParseFuncName))
    {
        m_CustomBBoxParseFunc =
            m_CustomLibHandle->symbol<NvDsInferParseCustomFunc>(
                initParams.customBBoxParseFuncName);
        if (!m_CustomBBoxParseFunc)
        {
            printError(
                "Detect-postprocessor failed to init resource "
                "because dlsym failed to get func %s pointer",
                safeStr(initParams.customBBoxParseFuncName));
            return NVDSINFER_CUSTOM_LIB_FAILED;
        }
    }

    if (m_ClusterMode == NVDSINFER_CLUSTER_DBSCAN || m_ClusterMode == NVDSINFER_CLUSTER_DBSCAN_NMS_HYBRID)
    {
        m_DBScanHandle.reset(
            NvDsInferDBScanCreate(), [](NvDsInferDBScanHandle handle) {
                if (handle)
                    NvDsInferDBScanDestroy(handle);
            });
        if (!m_DBScanHandle)
        {
            printError("Detect-postprocessor failed to create dbscan handle");
            return NVDSINFER_RESOURCE_ERROR;
        }
    }

    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
DetectPostprocessor::parseEachBatch(
    const std::vector<NvDsInferLayerInfo>& outputLayers,
    NvDsInferFrameOutput& result)
{
    result.outputType = NvDsInferNetworkType_Detector;
    fillDetectionOutput(outputLayers, result.detectionOutput);
    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
InstanceSegmentPostprocessor::initResource(const NvDsInferContextInitParams& initParams)
{
    RETURN_NVINFER_ERROR(InferPostprocessor::initResource(initParams),
        "init post processing resource failed");

    m_ClusterMode = initParams.clusterMode;
    if (m_ClusterMode != NVDSINFER_CLUSTER_NONE) {
        printError(" cluster mode %d not supported with instance segmentation", m_ClusterMode);
        return NVDSINFER_CONFIG_FAILED;
    }

    m_NumDetectedClasses = initParams.numDetectedClasses;
    if (initParams.numDetectedClasses > 0 &&
        initParams.perClassDetectionParams == nullptr)
    {
        printError(
            "NumDetectedClasses > 0 but PerClassDetectionParams array not "
            "specified");
        return NVDSINFER_CONFIG_FAILED;
    }

    m_PerClassDetectionParams.assign(initParams.perClassDetectionParams,
        initParams.perClassDetectionParams + m_NumDetectedClasses);
    m_DetectionParams.numClassesConfigured = initParams.numDetectedClasses;
    m_DetectionParams.perClassPreclusterThreshold.resize(initParams.numDetectedClasses);
    m_DetectionParams.perClassPostclusterThreshold.resize(initParams.numDetectedClasses);

    /* Resize the per class vector to the number of detected classes. */
    m_PerClassInstanceMaskList.resize(initParams.numDetectedClasses);

    /* Fill the class thresholds in the m_DetectionParams structure. This
     * will be required during parsing. */
    for (unsigned int i = 0; i < initParams.numDetectedClasses; i++)
    {
        m_DetectionParams.perClassPreclusterThreshold[i] =
            m_PerClassDetectionParams[i].preClusterThreshold;
        m_DetectionParams.perClassPostclusterThreshold[i] =
            m_PerClassDetectionParams[i].postClusterThreshold;
    }

    /* If custom parse function is specified get the function address from the
     * custom library. */
    if (m_CustomLibHandle &&
        !string_empty(initParams.customBBoxInstanceMaskParseFuncName))
    {
        m_CustomParseFunc =
            m_CustomLibHandle->symbol<NvDsInferInstanceMaskParseCustomFunc>(
                initParams.customBBoxInstanceMaskParseFuncName);
        if (!m_CustomParseFunc)
        {
            printError(
                "InstanceSegment-postprocessor failed to init resource "
                "because dlsym failed to get func %s pointer",
                safeStr(initParams.customBBoxInstanceMaskParseFuncName));
            return NVDSINFER_CUSTOM_LIB_FAILED;
        }
    } else {
        printError("Custom parse function not found for InstanceSegment-postprocessor");
        return NVDSINFER_RESOURCE_ERROR;
    }

    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
InstanceSegmentPostprocessor::parseEachBatch(
    const std::vector<NvDsInferLayerInfo>& outputLayers,
    NvDsInferFrameOutput& result)
{
    result.outputType = NvDsInferNetworkType_InstanceSegmentation;
    fillDetectionOutput(outputLayers, result.detectionOutput);
    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
ClassifyPostprocessor::initResource(const NvDsInferContextInitParams& initParams)
{
    RETURN_NVINFER_ERROR(InferPostprocessor::initResource(initParams),
        "init post processing resource failed");

    m_ClassifierThreshold = initParams.classifierThreshold;

    /* If custom parse function is specified get the function address from the
     * custom library. */
    if (m_CustomLibHandle &&
        !string_empty(initParams.customClassifierParseFuncName))
    {
        m_CustomClassifierParseFunc =
            m_CustomLibHandle->symbol<NvDsInferClassiferParseCustomFunc>(
                initParams.customClassifierParseFuncName);
        if (!m_CustomClassifierParseFunc)
        {
            printError(
                "Failed to init classify-postprocessor "
                "because dlsym failed to get func %s pointer",
                safeStr(initParams.customClassifierParseFuncName));
            return NVDSINFER_CUSTOM_LIB_FAILED;
        }
    }
    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
ClassifyPostprocessor::parseEachBatch(
    const std::vector<NvDsInferLayerInfo>& outputLayers,
    NvDsInferFrameOutput& result)
{
    result.outputType = NvDsInferNetworkType_Classifier;
    fillClassificationOutput(outputLayers, result.classificationOutput);
    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
SegmentPostprocessor::initResource(const NvDsInferContextInitParams& initParams)
{
    RETURN_NVINFER_ERROR(InferPostprocessor::initResource(initParams),
        "init post processing resource failed");

    m_SegmentationThreshold = initParams.segmentationThreshold;
    m_SegmentationOutputOrder = initParams.segmentationOutputOrder;
    m_NumSegmentationClasses = initParams.numDetectedClasses;
    /* If custom parse function is specified get the function address from the
     * custom library. */
    if (m_CustomLibHandle &&
        !string_empty(initParams.customSegmentationParseFuncName))
    {
        m_CustomSegmentationParseFunc =
            m_CustomLibHandle->symbol<NvDsInferSemSegmentationParseCustomFunc>(
                initParams.customSegmentationParseFuncName);
        if (!m_CustomSegmentationParseFunc)
        {
            printError("Segment-postprocessor failed to init resource "
                "because dlsym failed to get func %s pointer",
                safeStr(initParams.customSegmentationParseFuncName));
            return NVDSINFER_CUSTOM_LIB_FAILED;
        }
    }

    return NVDSINFER_SUCCESS;
}
NvDsInferStatus
SegmentPostprocessor::parseEachBatch(
    const std::vector<NvDsInferLayerInfo>& outputLayers,
    NvDsInferFrameOutput& result)
{
    result.outputType = NvDsInferNetworkType_Segmentation;
    fillSegmentationOutput(outputLayers, result.segmentationOutput);

    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
OtherPostprocessor::initResource(const NvDsInferContextInitParams& initParams)
{
    return NVDSINFER_SUCCESS;
}

/* Default constructor. */
NvDsInferContextImpl::NvDsInferContextImpl()
    : INvDsInferContext(), m_Batches(0) {}

NvDsInferStatus
NvDsInferContextImpl::preparePreprocess(const NvDsInferContextInitParams& initParams)
{
    assert(!m_Preprocessor);
    assert(
            m_NetworkInfo.channels && m_NetworkInfo.width && m_NetworkInfo.height);

    switch (initParams.networkInputFormat)
    {
        case NvDsInferFormat_RGB:
        case NvDsInferFormat_BGR:
            if (m_NetworkInfo.channels != 3)
            {
                printError(
                        "RGB/BGR input format specified but network input"
                        " channels is not 3");
                return NVDSINFER_CONFIG_FAILED;
            }
            break;
        case NvDsInferFormat_GRAY:
            if (m_NetworkInfo.channels != 1)
            {
                printError(
                        "GRAY input format specified but network input "
                        "channels is not 1.");
                return NVDSINFER_CONFIG_FAILED;
            }
            break;
        case NvDsInferFormat_Tensor:
            break;
        default:
            printError("Unknown input format");
            return NVDSINFER_CONFIG_FAILED;
    }

    std::unique_ptr<InferPreprocessor> processor =
        std::make_unique<InferPreprocessor>(m_NetworkInfo,
                initParams.networkInputFormat, m_InputImageLayerInfo, m_UniqueID);
    assert(processor);

    processor->setLoggingFunc(m_LoggingFunc);

    if (initParams.networkScaleFactor > 0.0f)
    {
        std::vector<float> offsets(
                initParams.offsets, initParams.offsets + initParams.numOffsets);
        if (!processor->setScaleOffsets(
                    initParams.networkScaleFactor, offsets)) {
            printError("Preprocessor set scale and offsets failed.");
            return NVDSINFER_CONFIG_FAILED;
        }
    }

    if (!string_empty(initParams.meanImageFilePath) &&
            !processor->setMeanFile(initParams.meanImageFilePath))
    {
        printError("Cannot access mean image file '%s'",
                safeStr(initParams.meanImageFilePath));
        return NVDSINFER_CONFIG_FAILED;
    }

    if (!processor->setInputOrder(initParams.netInputOrder))
    {
        printError("Cannot set network order '%s'",
               (initParams.netInputOrder == 0) ? "NCHW" : "NHWC");
        return NVDSINFER_CONFIG_FAILED;
    }

    NvDsInferStatus status = processor->allocateResource();
    if (status != NVDSINFER_SUCCESS)
    {
        printError("preprocessor allocate resource failed");
        return status;
    }

    m_Preprocessor = std::move(processor);
    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
NvDsInferContextImpl::preparePostprocess(const NvDsInferContextInitParams& initParams)
{
    assert(!m_Postprocessor);
    std::unique_ptr<InferPostprocessor> processor;
    switch (initParams.networkType)
    {
        case NvDsInferNetworkType_Detector:
            processor = std::make_unique<DetectPostprocessor>(m_UniqueID, m_GpuID);
            break;
        case NvDsInferNetworkType_Classifier:
            processor =
                std::make_unique<ClassifyPostprocessor>(m_UniqueID, m_GpuID);
            break;
        case NvDsInferNetworkType_Segmentation:
            processor = std::make_unique<SegmentPostprocessor>(m_UniqueID, m_GpuID);
            break;
        case NvDsInferNetworkType_InstanceSegmentation:
            processor = std::make_unique<InstanceSegmentPostprocessor>(m_UniqueID, m_GpuID);
            break;
        case NvDsInferNetworkType_Other:
            processor = std::make_unique<OtherPostprocessor>(m_UniqueID, m_GpuID);
            break;
        default:
            printError(
                    "Failed to preprare post processing because of unknown network "
                    "type:%d",
                    (int)(initParams.networkType));
            return NVDSINFER_CONFIG_FAILED;
    }

    processor->setDlHandle(m_CustomLibHandle);
    processor->setNetworkInfo(m_NetworkInfo);
    processor->setAllLayerInfo(m_AllLayerInfo);
    processor->setOutputLayerInfo(m_OutputLayerInfo);
    processor->setLoggingFunc(m_LoggingFunc);

    RETURN_NVINFER_ERROR(processor->initResource(initParams),
        "Infer Context failed to initialize post-processing resource");

    m_Postprocessor = std::move(processor);
    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
NvDsInferContextImpl::initInferenceInfo(
    const NvDsInferContextInitParams& initParams, BackendContext& ctx)
{
    /* Get information on all bound layers. */
    m_MaxBatchSize = ctx.getMaxBatchDims(INPUT_LAYER_INDEX).batchSize;
    int layerSize = ctx.getNumBoundLayers();
    assert(layerSize);
    assert(INPUT_LAYER_INDEX < layerSize);

    /* Get properties of bound layers like the name, dimension, datatype and
     * fill the m_AllLayerInfo and m_OutputLayerInfo vectors.
     */
    for (int i = 0; i < layerSize; i++)
    {
        NvDsInferBatchDimsLayerInfo info = ctx.getLayerInfo(i);

        if (i == INPUT_LAYER_INDEX)
        {
            if (!info.isInput)
            {
                printError(
                    "Infer Context default image layer[%d] is not a input "
                    "layer",
                    i);
                return NVDSINFER_TENSORRT_ERROR;
            }

            m_InputImageLayerInfo = info;
        }

        m_AllLayerInfo.push_back(info);
        if (!info.isInput)
            m_OutputLayerInfo.push_back(info);
    }

    /* Get the network input dimensions. */

    if (!initParams.inputFromPreprocessedTensor && m_InputImageLayerInfo.inferDims.numDims != 3)
    {
        printError("Infer Context default input_layer is not a image[CHW]");
        return NVDSINFER_TENSORRT_ERROR;
    }

    if (initParams.inferInputDims.c && initParams.inferInputDims.h &&
        initParams.inferInputDims.w)
    {
        m_NetworkInfo.width = initParams.inferInputDims.w;
        m_NetworkInfo.height = initParams.inferInputDims.h;
        m_NetworkInfo.channels = initParams.inferInputDims.c;
    }
    else if (m_InputImageLayerInfo.inferDims.numDims == 3)
    {
        m_NetworkInfo.width = m_InputImageLayerInfo.inferDims.d[2];
        m_NetworkInfo.height = m_InputImageLayerInfo.inferDims.d[1];
        m_NetworkInfo.channels = m_InputImageLayerInfo.inferDims.d[0];
    }

    m_MaxBatchSize = m_InputImageLayerInfo.profileDims[kSELECTOR_MAX].batchSize;

    return NVDSINFER_SUCCESS;
}

/* The function performs all the initialization steps required by the inference
 * engine. */
NvDsInferStatus
NvDsInferContextImpl::initialize(NvDsInferContextInitParams& initParams,
        void* userCtx, NvDsInferContextLoggingFunc logFunc)
{
    m_UniqueID = initParams.uniqueID;
    m_MaxBatchSize = initParams.maxBatchSize;
    m_GpuID = initParams.gpuID;
    m_OutputBufferPoolSize = initParams.outputBufferPoolSize;
    m_AutoIncMem = initParams.autoIncMem;
    m_MaxGPUMem = initParams.maxGPUMemPer;
    m_DumpIpTensor = initParams.dumpIpTensor;
    if (m_DumpIpTensor) {
        m_DumpIpTensorFilePath = "ip_tensor_dump.bin";
    }
    m_OverwriteIpTensor = initParams.overwriteIpTensor;
    if (m_OverwriteIpTensor) {
        m_OverwriteIpTensorFilePath = initParams.ipTensorFilePath;
        m_OverwriteIpTensorFile.open (m_OverwriteIpTensorFilePath, std::ifstream::binary);
    }

    uint32_t uniqueID = initParams.uniqueID;
    m_LoggingFunc = [this, userCtx, logFunc, uniqueID](
                        NvDsInferLogLevel level, const char* msg) {
        logFunc(this, uniqueID, level, msg, userCtx);
    };

#ifndef WITH_OPENCV
    if (initParams.clusterMode == NVDSINFER_CLUSTER_GROUP_RECTANGLES &&
            initParams.networkType == NvDsInferNetworkType_Detector) {
        initParams.clusterMode = NVDSINFER_CLUSTER_NMS;
        for (unsigned int i = 0; i < initParams.numDetectedClasses && initParams.perClassDetectionParams; i++) {
            initParams.perClassDetectionParams[i].topK = 20;
            initParams.perClassDetectionParams[i].nmsIOUThreshold = 0.5;
        }
        printWarning ("Warning, OpenCV has been deprecated. Using NMS for clustering instead of cv::groupRectangles with topK = 20 and NMS Threshold = 0.5");
    }
#endif

    /* Synchronization using once_flag and call_once to ensure TensorRT plugin
     * initialization function is called only once in case of multiple instances
     * of this constructor being called from different threads. */
    {
        static std::once_flag pluginInitFlag;
        std::call_once(pluginInitFlag,
            [this]() { initLibNvInferPlugins(gTrtLogger.get(), ""); });
    }

    if (m_UniqueID == 0)
    {
        printError("Unique ID not set");
        return NVDSINFER_CONFIG_FAILED;
    }

    if (m_MaxBatchSize == 0)
    {
        printError("maxBatchSize should be not be zero");
        return NVDSINFER_CONFIG_FAILED;
    }

    if (m_MaxBatchSize > NVDSINFER_MAX_BATCH_SIZE)
    {
        printError("Batch-size (%d) more than maximum allowed batch-size (%d)",
            initParams.maxBatchSize, NVDSINFER_MAX_BATCH_SIZE);
        return NVDSINFER_CONFIG_FAILED;
    }

    if (m_MaxGPUMem <= 0) {
        printf("WARNING: Max GPU memory (%f) is less than or equal to 0, falling back to 99\n",
            initParams.maxGPUMemPer);
        m_MaxGPUMem = 99.0;
    }

    if(m_MaxGPUMem > 100) {
        printf("WARNING: Max GPU memory (%f) cannot be greater than 100, setting to 100\n",
            initParams.maxGPUMemPer);
        m_MaxGPUMem = 100.0;
    }

    if(initParams.uffDimsCHW.c != 0 || initParams.uffDimsCHW.h !=0 ||
       initParams.uffDimsCHW.w != 0)
    {
      printError("Deprecated params uffDimsCHW is being used");
      return NVDSINFER_INVALID_PARAMS;
    }

    if(initParams.inputDims.c != 0 || initParams.inputDims.h != 0 ||
       initParams.inputDims.w != 0)
    {
      printError("Deprecated params inputDims is being used");
      return NVDSINFER_INVALID_PARAMS;
    }

    if (initParams.numOutputLayers > 0 &&
        initParams.outputLayerNames == nullptr)
    {
        printError(
            "numOutputLayers > 0 but outputLayerNames array not specified");
        return NVDSINFER_CONFIG_FAILED;
    }

    if(initParams.numOutputIOFormats > 0 &&
       initParams.outputIOFormats == nullptr)
    {
        printError("numOutputIOFormats >0 but outputIOFormats array not specified");
        return NVDSINFER_CONFIG_FAILED;
    }

    if (initParams.numLayerDevicePrecisions > 0 &&
       initParams.layerDevicePrecisions == nullptr)
    {
        printError("numLayerDevicePrecisions >0 but layerDevicePrecisions array not specified");
        return NVDSINFER_CONFIG_FAILED;
    }

    if (m_OutputBufferPoolSize < NVDSINFER_MIN_OUTPUT_BUFFERPOOL_SIZE)
    {
        printError(
            "Output buffer pool size (%d) less than minimum required(%d)",
            m_OutputBufferPoolSize, NVDSINFER_MIN_OUTPUT_BUFFERPOOL_SIZE);
        return NVDSINFER_CONFIG_FAILED;
    }

    /* Set the cuda device to be used. */
    RETURN_CUDA_ERR(
        cudaSetDevice(m_GpuID), "Failed to set cuda device (%d).", m_GpuID);

    /* Load the custom library if specified. */
    if (!string_empty(initParams.customLibPath))
    {
        std::unique_ptr<DlLibHandle> dlHandle =
            std::make_unique<DlLibHandle>(initParams.customLibPath, RTLD_LAZY);
        if (!dlHandle->isValid())
        {
            printError("Could not open custom lib: %s", dlerror());
            return NVDSINFER_CUSTOM_LIB_FAILED;
        }
        m_CustomLibHandle = std::move(dlHandle);
    }

    m_BackendContext = generateBackendContext(initParams);
    if (!m_BackendContext)
    {
        printError("generate backend failed, check config file settings");
        return NVDSINFER_CONFIG_FAILED;
    }

    RETURN_NVINFER_ERROR(initInferenceInfo(initParams, *m_BackendContext),
        "Infer context initialize inference info failed");
    assert(m_AllLayerInfo.size());

    if (!initParams.inputFromPreprocessedTensor) {
        RETURN_NVINFER_ERROR(preparePreprocess(initParams),
            "Infer Context prepare preprocessing resource failed.");
        assert(m_Preprocessor);
    }

    RETURN_NVINFER_ERROR(preparePostprocess(initParams),
        "Infer Context prepare postprocessing resource failed.");
    assert(m_Postprocessor);

    /* Allocate binding buffers on the device and the corresponding host
     * buffers. */
    NvDsInferStatus status = allocateBuffers();
    if (status != NVDSINFER_SUCCESS)
    {
        printError("Failed to allocate buffers");
        return status;
    }

    /* If there are more than one input layers (non-image input) and custom
     * library is specified, try to initialize these layers. */
    if (m_InputDeviceBuffers.size() > 1)
    {
        NvDsInferStatus status = initNonImageInputLayers();
        if (status != NVDSINFER_SUCCESS)
        {
            printError("Failed to initialize non-image input layers");
            return status;
        }
    }

    /* Run TRT engine warmup if enabled. This exercises enqueueV3() for
     * each batch size from 1 to maxBatchSize, pre-warming TRT's internal
     * CUDA kernel caches and eliminating first-inference latency spikes
     * caused by partial batches during pipeline startup. */
    if (initParams.warmupEngine)
    {
        NvDsInferStatus status = warmupEngine();
        if (status != NVDSINFER_SUCCESS)
        {
            printWarning("Engine warmup failed (non-fatal), continuing "
                "initialization");
        }
    }

    m_Initialized = true;
    return NVDSINFER_SUCCESS;
}

/* Get the network input resolution. This is required since this implementation
 * requires that the caller supplies an input buffer having the network
 * resolution.
 */
void
NvDsInferContextImpl::getNetworkInfo(NvDsInferNetworkInfo &networkInfo)
{
    networkInfo = m_NetworkInfo;
}

/* Allocate/re-size the batch vector (bufferpool) according to the inferences
 * made by the gie. "numBuffers" should ideally be a low number to avoid
 * allocation of extra cuda memory.
 */
NvDsInferStatus
NvDsInferContextImpl::resizeOutputBufferpool(uint32_t numBuffers)
{
  /* Check to see the amount of utilized memory */
  size_t free, total;
  cudaSetDevice(m_GpuID);
  int id;
  cudaGetDevice(&id);
  cudaMemGetInfo(&free, &total);

  std::unique_lock<std::mutex> lock(m_BatchesMutex);

  int buffer_pool_size = m_Batches.size();

  if (buffer_pool_size) {
    if(free <= (1 - m_MaxGPUMem/100) * total) {
      printf("Warning: Not increasing the pool size as CUDA memory is already at %f utilisation\n", m_MaxGPUMem);
      return NVDSINFER_MEM_ERROR;
    }
  }

  m_Batches.resize(buffer_pool_size + numBuffers);

  /* Initialize the batch vector, allocate host memory for the layers,
   * add all the free indexes to the free queue. */
  for (size_t iB = buffer_pool_size; iB < m_Batches.size(); iB++)
  {
      m_Batches[iB] = std::make_shared<NvDsInferBatch>();
      NvDsInferBatch& batch = *m_Batches[iB];

      /* Resize the host buffers vector to the number of bound layers. */
      batch.m_HostBuffers.resize(m_AllLayerInfo.size());
      batch.m_DeviceBuffers.assign(m_AllLayerInfo.size(), nullptr);

      for (unsigned int jL = 0; jL < m_AllLayerInfo.size(); jL++)
      {
          const NvDsInferBatchDimsLayerInfo& layerInfo = m_AllLayerInfo[jL];
          const NvDsInferDims& bindingDims = layerInfo.inferDims;
          assert(bindingDims.numElements > 0);
          size_t size = m_MaxBatchSize *
                        bindingDims.numElements *
                        getElementSize(layerInfo.dataType);

          if (layerInfo.isInput)
          {
              /* Reuse input binding buffer pointers. */
              batch.m_DeviceBuffers[jL] = m_BindingBuffers[jL];
          }
          else
          {
              /* Allocate device memory for output layers here. */
              auto outputBuf = std::make_unique<CudaDeviceBuffer>(size);
              if (!outputBuf || !outputBuf->ptr())
              {
                  printError(
                      "Failed to allocate cuda output buffer during context "
                      "initialization");
                  return NVDSINFER_CUDA_ERROR;
              }
              batch.m_DeviceBuffers[jL] = outputBuf->ptr();
              batch.m_OutputDeviceBuffers.emplace_back(std::move(outputBuf));
          }

          /* Allocate host memory for input layers only if application
            * needs access to the input layer contents. */
          if (layerInfo.isInput && !m_Postprocessor->needInputCopy())
              continue;

          auto hostBuf = std::make_unique<CudaHostBuffer>(size);
          if (!hostBuf || !hostBuf->ptr())
          {
              printError(
                  "Failed to allocate cuda host buffer during context "
                  "initialization");
              return NVDSINFER_CUDA_ERROR;
          }
          batch.m_HostBuffers[jL] = std::move(hostBuf);
      }

      batch.m_OutputCopyDoneEvent = std::make_unique<CudaEvent>(
          cudaEventDisableTiming | cudaEventBlockingSync);
      if (!batch.m_OutputCopyDoneEvent || !batch.m_OutputCopyDoneEvent->ptr())
      {
          printError("Failed to create cuda event");
          return NVDSINFER_CUDA_ERROR;
      }

      /* Add all the indexes to the free queue initially. */
      m_FreeBatchQueue.push(&batch);
  }
  return NVDSINFER_SUCCESS;
}


/* Allocate binding buffers for all bound layers on the device memory. The size
 * of the buffers allocated is calculated from the dimensions of the layers, the
 * data type of the layer and the max batch size of the infer cuda engine.
 *
 * NvDsInfer enqueue API requires an array of (void *) buffer pointers. The length
 * of the array is equal to the number of bound layers. The buffer corresponding
 * to a layer is placed at an index equal to the layer's binding index.
 *
 * Also allocate corresponding host buffers for output layers in system memory.
 *
 * Multiple sets of the device and host buffers are allocated so that (inference +
 * device to host copy) and output layers parsing can be parallelized.
 */
NvDsInferStatus
NvDsInferContextImpl::allocateBuffers()
{
    /* Create the cuda stream on which inference jobs will be executed. */
    m_InferStream = std::make_unique<CudaStream>(cudaStreamNonBlocking);
    if (!m_InferStream || !m_InferStream->ptr())
    {
        printError("Failed to create infer cudaStream");
        return NVDSINFER_CUDA_ERROR;
    }
    std::string nvtx_name = "nvdsinfer_infer_uid=" + std::to_string(m_UniqueID);
    nvtxNameCudaStreamA(*m_InferStream, nvtx_name.c_str());

    /* Create the cuda stream on which post-processing jobs will be
     * executed. */
    m_PostprocessStream = std::make_unique<CudaStream>(cudaStreamNonBlocking);
    if (!m_PostprocessStream || !m_PostprocessStream->ptr())
    {
        printError("Failed to create cudaStream");
        return NVDSINFER_CUDA_ERROR;
    }
    nvtx_name = "nvdsinfer_postproc_uid=" + std::to_string(m_UniqueID);
    nvtxNameCudaStreamA(*m_InferStream, nvtx_name.c_str());

    /* Cuda event to synchronize between consumption of input binding buffer by
     * the cuda engine and the pre-processing kernel which writes to the input
     * binding buffer. */
    m_InputConsumedEvent = std::make_unique<CudaEvent>(cudaEventDisableTiming);
    if ((cudaEvent_t)(*m_InputConsumedEvent) == nullptr)
    {
        printError("Failed to create input consume cuda event");
        return NVDSINFER_CUDA_ERROR;
    }
    nvtx_name =
        "nvdsinfer_TRT_input_consumed_uid=" + std::to_string(m_UniqueID);
    nvtxNameCudaEventA(*m_InputConsumedEvent, nvtx_name.c_str());

    /* Cuda event to synchronize between completion of inference on a batch
     * and copying the output contents from device to host memory. */
    m_InferCompleteEvent = std::make_unique<CudaEvent>(cudaEventDisableTiming);
    if ((cudaEvent_t)(*m_InferCompleteEvent) == nullptr)
    {
        printError("Failed to create cuda event");
        return NVDSINFER_CUDA_ERROR;
    }
    nvtx_name = "nvdsinfer_infer_complete_uid=" + std::to_string(m_UniqueID);
    nvtxNameCudaEventA(*m_InferCompleteEvent, nvtx_name.c_str());

    /* Resize the binding buffers vector to the number of bound layers. */
    m_BindingBuffers.assign(m_AllLayerInfo.size(), nullptr);

    /* allocate input/output layers buffers */
    for (size_t iL = 0; iL < m_AllLayerInfo.size(); iL++)
    {
        const NvDsInferBatchDimsLayerInfo& layerInfo = m_AllLayerInfo[iL];
        /* Do not allocate device memory for output layers here. */
        if (!layerInfo.isInput)
            continue;

        const NvDsInferDims& layerDims = layerInfo.inferDims;
        assert(layerDims.numElements > 0);
        size_t size = m_MaxBatchSize * layerDims.numElements *
                      getElementSize(layerInfo.dataType);

        auto inputBuf = std::make_unique<CudaDeviceBuffer>(size);
        if (!inputBuf || !inputBuf->ptr())
        {
            printError(
                "Failed to allocate cuda input buffer during context "
                "initialization");
            return NVDSINFER_CUDA_ERROR;
        }
        m_BindingBuffers[iL] = inputBuf->ptr();
        m_InputDeviceBuffers.emplace_back(std::move(inputBuf));
    }

    NvDsInferStatus status = resizeOutputBufferpool(m_OutputBufferPoolSize);
    if (status != NVDSINFER_SUCCESS)
    {
        printError("Failed to allocate output bufferpool\n");
        return status;
    }

    return NVDSINFER_SUCCESS;
}

/* Initialize non-image input layers if the custom library has implemented
 * the interface. */
NvDsInferStatus
NvDsInferContextImpl::initNonImageInputLayers()
{
    cudaError_t cudaReturn;

    /* Needs the custom library to be specified. */
    if (!m_CustomLibHandle)
    {
        printWarning("More than one input layers but custom initialization "
            "function not implemented");
        return NVDSINFER_CUSTOM_LIB_FAILED;
    }

    /* Check if the interface to initialize the layers has been implemented. */
    auto initInputFcn =
        READ_SYMBOL(m_CustomLibHandle, NvDsInferInitializeInputLayers);
    if (initInputFcn == nullptr)
    {
        printWarning("More than one input layers but custom initialization "
            "function not implemented");
        return NVDSINFER_CUSTOM_LIB_FAILED;
    }

    /* Interface implemented.  */
    /* Vector of NvDsInferLayerInfo for non-image input layers. */
    std::vector<NvDsInferLayerInfo> inputLayers;
    for (const auto& layer : m_AllLayerInfo)
    {
        if (layer.isInput && layer.bindingIndex != INPUT_LAYER_INDEX)
        {
            inputLayers.push_back(layer);
        }
    }

    /* Vector of host memories that can be initialized using CPUs. */
    std::vector<std::vector<uint8_t>> initBuffers(inputLayers.size());

    for (size_t i = 0; i < inputLayers.size(); i++)
    {
        /* For each layer calculate the size required for the layer, allocate
         * the host memory and assign the pointer to layer info structure. */
        assert(inputLayers[i].inferDims.numElements > 0);
        size_t size = inputLayers[i].inferDims.numElements *
                      getElementSize(inputLayers[i].dataType) * m_MaxBatchSize;
        initBuffers[i].resize(size);
        inputLayers[i].buffer = (void *) initBuffers[i].data();
    }

    /* Call the input layer initialization function. */
    if (!initInputFcn(inputLayers, m_NetworkInfo, m_MaxBatchSize))
    {
        printError("Failed to initialize input layers using "
                "NvDsInferInitializeInputLayers() in custom lib");
        return NVDSINFER_CUSTOM_LIB_FAILED;
    }

    /* Memcpy the initialized contents from the host memory to device memory for
     * layer binding buffers. */
    for (size_t i = 0; i < inputLayers.size(); i++)
    {
        cudaReturn =
            cudaMemcpyAsync(m_BindingBuffers[inputLayers[i].bindingIndex],
                initBuffers[i].data(), initBuffers[i].size(),
                cudaMemcpyHostToDevice, *m_InferStream);
        if (cudaReturn != cudaSuccess)
        {
            printError("Failed to copy from host to device memory (%s)",
                    cudaGetErrorName(cudaReturn));
            return NVDSINFER_CUDA_ERROR;
        }

        /* Application has requested access to the bound buffer contents. Copy
         * the contents to all sets of host buffers. */
        if (m_Postprocessor->needInputCopy()) {
            std::unique_lock<std::mutex> lock(m_BatchesMutex);
            for (size_t j = 0; j < m_Batches.size(); j++)
            {
                auto& buf =
                    m_Batches[j]->m_HostBuffers[inputLayers[i].bindingIndex];
                assert(buf && buf->bytes() >= initBuffers[i].size());
                memcpy(
                    buf->ptr(), initBuffers[i].data(), initBuffers[i].size());
            }
        }
    }
    cudaReturn = cudaStreamSynchronize(*m_InferStream);
    if (cudaReturn != cudaSuccess)
    {
        printError("Failed to synchronize cuda stream(%s)",
                cudaGetErrorName(cudaReturn));
        return NVDSINFER_CUDA_ERROR;
    }

    return NVDSINFER_SUCCESS;
}

namespace {

class BackendBatchBuffer : public InferBatchBuffer
{
public:
    BackendBatchBuffer(std::vector<void*>& bindings,
        std::vector<NvDsInferBatchDimsLayerInfo>& info, int batch)
        : m_BindingBufs(bindings), m_LayersInfo(info), m_BatchSize(batch) {}
    ~BackendBatchBuffer() override {}

private:
    std::vector<void*>& getDeviceBuffers() override { return m_BindingBufs; }
    NvDsInferDataType getDataType(int bindingIndex) const override
    {
        assert(bindingIndex < (int)m_LayersInfo.size());
        return m_LayersInfo.at(bindingIndex).dataType;
    }
    NvDsInferBatchDims getBatchDims(int bindingIndex) const override
    {
        assert(bindingIndex < (int)m_LayersInfo.size());
        return {m_BatchSize, m_LayersInfo.at(bindingIndex).inferDims};
    }

private:
    std::vector<void*> m_BindingBufs;
    std::vector<NvDsInferBatchDimsLayerInfo>& m_LayersInfo;
    int m_BatchSize = 0;
};
}

/* Run TensorRT engine warmup by executing dummy inference for batch sizes
 * 1 through m_MaxBatchSize. This pre-warms TRT's internal CUDA kernel caches,
 * workspace allocations, and execution graphs for every possible batch size.
 *
 * Without warmup, the first inference at a previously unseen batch size
 * triggers expensive TRT internal recompilation (setInputShape + enqueueV3
 * JIT overhead). During pipeline startup with RTSP sources, partial batches
 * arrive before all sources are active, causing accumulated latency that
 * never recovers in the streaming FIFO pipeline.
 *
 * This method reuses already-allocated input/output device buffers from
 * m_Batches[0] -- no extra GPU memory is needed. Input data is uninitialized
 * (garbage); output is discarded. Only the TRT execution path is exercised.
 */
NvDsInferStatus
NvDsInferContextImpl::warmupEngine()
{
    assert(m_BackendContext);
    assert(m_InferStream && m_InferStream->ptr());
    assert(m_InputConsumedEvent);
    assert(!m_Batches.empty());

    /* Use the device buffers from the first output batch slot. Input buffers
     * are shared across all batch slots (m_BindingBuffers), and output
     * buffers are per-slot (m_Batches[i]->m_DeviceBuffers). */
    NvDsInferBatch& batchSlot = *m_Batches[0];
    std::vector<void*>& bindings = batchSlot.m_DeviceBuffers;

    for (uint32_t bs = 1; bs <= m_MaxBatchSize; bs++)
    {
        auto warmupBuffer = std::make_shared<BackendBatchBuffer>(
            bindings, m_AllLayerInfo, bs);

        NvDsInferStatus status = m_BackendContext->enqueueBuffer(
            warmupBuffer, *m_InferStream, m_InputConsumedEvent.get());
        if (status != NVDSINFER_SUCCESS)
        {
            printWarning("Warmup failed for batch size %d, skipping "
                "remaining warmup iterations", bs);
            return NVDSINFER_SUCCESS;
        }

        /* Standard practice: sync so engine is truly warm before first real
         * batch (JIT done, caches hot). This is init-only; inference path
         * remains async. */
        cudaError_t cudaReturn = cudaStreamSynchronize(*m_InferStream);
        if (cudaReturn != cudaSuccess)
        {
            printWarning("Warmup cudaStreamSynchronize failed for "
                "batch size %d: %s", bs,
                cudaGetErrorName(cudaReturn));
            return NVDSINFER_SUCCESS;
        }
    }

    printInfo("TRT engine warmup completed");

    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
NvDsInferContextImpl::queueInputBatch(NvDsInferContextBatchInput &batchInput)
{
    assert(m_Initialized);
    uint32_t batchSize = batchInput.numInputFrames;

    /* Check that current batch size does not exceed max batch size. */
    if (batchSize > m_MaxBatchSize)
    {
        printError("Not inferring on batch since it's size(%d) exceeds max batch"
                " size(%d)", batchSize, m_MaxBatchSize);
        return NVDSINFER_INVALID_PARAMS;
    }

    /* Set the cuda device to be used. */
    RETURN_CUDA_ERR(cudaSetDevice(m_GpuID),
        "queue buffer failed to set cuda device(%s)", m_GpuID);

    std::shared_ptr<CudaEvent> preprocWaitEvent = m_InputConsumedEvent;

    assert(m_Preprocessor && m_InputConsumedEvent);
    RETURN_NVINFER_ERROR(m_Preprocessor->transform(batchInput,
                             m_BindingBuffers[INPUT_LAYER_INDEX],
                             *m_InferStream, preprocWaitEvent.get()),
        "Preprocessor transform input data failed.");


    if (m_DumpIpTensor) {
        uint32_t dump_size = m_NetworkInfo.width*m_NetworkInfo.height*m_NetworkInfo.channels;
        char *dump_cpu_buffer = (char *) calloc(dump_size*4, sizeof(char));
        cudaError_t cudaReturn;
        cudaReturn = cudaStreamSynchronize(*m_InferStream);
        if (cudaReturn != cudaSuccess)
        {
            printError("Failed to synchronize cuda stream(%s)",
                    cudaGetErrorName(cudaReturn));
            free(dump_cpu_buffer);
            return NVDSINFER_CUDA_ERROR;
        }
        cudaMemcpy(dump_cpu_buffer, m_BindingBuffers[INPUT_LAYER_INDEX], dump_size*4, cudaMemcpyDeviceToHost);
        std::ofstream dump_ip_file(m_DumpIpTensorFilePath, std::ios_base::app);
        dump_ip_file.write((char*) dump_cpu_buffer, dump_size*4);
        dump_ip_file.close();
        free(dump_cpu_buffer);
    }

    if (m_OverwriteIpTensor) {
        uint32_t dump_size = m_NetworkInfo.width*m_NetworkInfo.height*m_NetworkInfo.channels;
        char *dump_cpu_buffer = (char *) calloc(dump_size*4, sizeof(char));
        cudaError_t cudaReturn;
        cudaReturn = cudaStreamSynchronize(*m_InferStream);
        if (cudaReturn != cudaSuccess)
        {
            printError("Failed to synchronize cuda stream(%s)",
                    cudaGetErrorName(cudaReturn));
	    free(dump_cpu_buffer);
            return NVDSINFER_CUDA_ERROR;
        }
        m_OverwriteIpTensorFile.read (dump_cpu_buffer, dump_size*4);
        cudaMemcpy(m_BindingBuffers[INPUT_LAYER_INDEX], dump_cpu_buffer, dump_size*4, cudaMemcpyHostToDevice);
        free(dump_cpu_buffer);
    }

    /* We may use multiple sets of the output device and host buffers since
     * while the output of one batch is being parsed on the CPU, we can queue
     * pre-processing and inference of another on the GPU. Pop an index from the
     * free queue. Wait if queue is empty. */
    auto recyleFunc = [this](NvDsInferBatch* batch) {
        if (batch)
            m_FreeBatchQueue.push(batch);
    };

    if (m_FreeBatchQueue.isEmpty() && m_AutoIncMem) {
      NvDsInferStatus status = resizeOutputBufferpool(6);
      if (status != NVDSINFER_SUCCESS)
      {
          printError("Dropping the batch as output bufferpool resize failed\n");
          return status;
      }
    }

    std::unique_ptr<NvDsInferBatch, decltype(recyleFunc)> safeRecyleBatch(
        m_FreeBatchQueue.pop(), recyleFunc);

    assert(safeRecyleBatch);
    safeRecyleBatch->m_BatchSize = batchSize;

    /* Fill the array of binding buffers for the current batch. */
    std::vector<void*>& bindings = safeRecyleBatch->m_DeviceBuffers;
    auto backendBuffer = std::make_shared<BackendBatchBuffer>(
        bindings, m_AllLayerInfo, batchSize);
    assert(m_BackendContext && backendBuffer);
    assert(m_InferStream && m_InputConsumedEvent && m_InferCompleteEvent);

    RETURN_NVINFER_ERROR(m_BackendContext->enqueueBuffer(backendBuffer,
                             *m_InferStream, m_InputConsumedEvent.get()),
        "Infer context enqueue buffer failed");

    /* Record event on m_InferStream to indicate completion of inference on the
     * current batch. */
    RETURN_CUDA_ERR(cudaEventRecord(*m_InferCompleteEvent, *m_InferStream),
        "Failed to record cuda infer-complete-event ");

    assert(m_PostprocessStream && m_InferCompleteEvent);
    /* Make future jobs on the postprocessing stream wait on the infer
     * completion event. */
    RETURN_CUDA_ERR(
        cudaStreamWaitEvent(*m_PostprocessStream, *m_InferCompleteEvent, 0),
        "postprocessing cuda waiting event failed ");
    RETURN_NVINFER_ERROR(m_Postprocessor->copyBuffersToHostMemory(
                             *safeRecyleBatch, *m_PostprocessStream),
        "post cuda process failed.");

    m_ProcessBatchQueue.push(safeRecyleBatch.release());
    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
NvDsInferContextImpl::queueInputBatchPreprocessed(NvDsInferContextBatchPreprocessedInput &batchInput)
{
    assert(m_Initialized);

    unsigned int batchSize = batchInput.tensors->dims.d[0];

    /* Set the cuda device to be used. */
    RETURN_CUDA_ERR(cudaSetDevice(m_GpuID),
        "queue buffer failed to set cuda device(%s)", m_GpuID);

    /* We may use multiple sets of the output device and host buffers since
     * while the output of one batch is being parsed on the CPU, we can queue
     * pre-processing and inference of another on the GPU. Pop an index from the
     * free queue. Wait if queue is empty. */
    auto recyleFunc = [this](NvDsInferBatch* batch) {
        if (batch)
            m_FreeBatchQueue.push(batch);
    };

    if (m_FreeBatchQueue.isEmpty() && m_AutoIncMem) {
      NvDsInferStatus status = resizeOutputBufferpool(6);
      if (status != NVDSINFER_SUCCESS)
      {
          printError("Dropping the batch as output bufferpool resize failed\n");
          return status;
      }
    }

    std::unique_ptr<NvDsInferBatch, decltype(recyleFunc)> safeRecyleBatch(
        m_FreeBatchQueue.pop(), recyleFunc);
    assert(safeRecyleBatch);
    safeRecyleBatch->m_BatchSize = batchSize;

    auto allLayerInfo = m_AllLayerInfo;

    /* Fill the array of binding buffers for the current batch. */
    std::vector<void*>& bindings = safeRecyleBatch->m_DeviceBuffers;
    for (unsigned int i = 0; i < batchInput.numInputTensors; i++) {
        for (auto &layerInfo: allLayerInfo) {
            if (strcmp(layerInfo.layerName, batchInput.tensors[i].layerName) == 0) {
                bindings[layerInfo.bindingIndex] = batchInput.tensors[i].buffer;
                layerInfo.buffer = batchInput.tensors[i].buffer;
                if (layerInfo.dataType != batchInput.tensors[i].dataType) {
                    // Warn user once
                }
                layerInfo.dims.numDims = batchInput.tensors[i].dims.numDims - 1;
                layerInfo.dims.numElements = batchInput.tensors[i].dims.numElements / batchSize;
                memcpy(layerInfo.dims.d, batchInput.tensors[i].dims.d + 1, layerInfo.dims.numDims * sizeof(unsigned int));
            }
        }
    }

    auto backendBuffer = std::make_shared<BackendBatchBuffer>(bindings, allLayerInfo, batchSize);
    assert(m_BackendContext && backendBuffer);
    assert(m_InferStream && m_InputConsumedEvent && m_InferCompleteEvent);

    RETURN_NVINFER_ERROR(m_BackendContext->enqueueBuffer(backendBuffer,
                             *m_InferStream, m_InputConsumedEvent.get()),
        "Infer context enqueue buffer failed");

    if (batchInput.returnInputFunc)
    {
        RETURN_CUDA_ERR(
            cudaStreamAddCallback(*m_InferStream, returnInputCudaCallback,
                new NvDsInferReturnInputPair(
                    batchInput.returnInputFunc, batchInput.returnFuncData),
                0),
            "Failed to add cudaStream callback for returning input buffers");
    }

    /* Record event on m_InferStream to indicate completion of inference on the
     * current batch. */
    RETURN_CUDA_ERR(cudaEventRecord(*m_InferCompleteEvent, *m_InferStream),
        "Failed to record cuda infer-complete-event ");

    assert(m_PostprocessStream && m_InferCompleteEvent);
    /* Make future jobs on the postprocessing stream wait on the infer
     * completion event. */
    RETURN_CUDA_ERR(
        cudaStreamWaitEvent(*m_PostprocessStream, *m_InferCompleteEvent, 0),
        "postprocessing cuda waiting event failed ");
    RETURN_NVINFER_ERROR(m_Postprocessor->copyBuffersToHostMemory(
                             *safeRecyleBatch, *m_PostprocessStream),
        "post cuda process failed.");

    m_ProcessBatchQueue.push(safeRecyleBatch.release());
    return NVDSINFER_SUCCESS;
}

/* Dequeue batch output of the inference engine for each batch input. */
NvDsInferStatus
NvDsInferContextImpl::dequeueOutputBatch(NvDsInferContextBatchOutput &batchOutput)
{
    assert(m_Initialized);
    auto recyleFunc = [this](NvDsInferBatch* batch) {
        if (batch)
            m_FreeBatchQueue.push(batch);
    };
    std::unique_ptr<NvDsInferBatch, decltype(recyleFunc)> recyleBatch(
        m_ProcessBatchQueue.pop(), recyleFunc);
    assert(recyleBatch);

    /* Set the cuda device */
    RETURN_CUDA_ERR(cudaSetDevice(m_GpuID),
        "dequeue buffer failed to set cuda device(%s)", m_GpuID);

    /* Wait for the copy to the current set of host buffers to complete. */
    RETURN_CUDA_ERR(cudaEventSynchronize(*recyleBatch->m_OutputCopyDoneEvent),
        "Failed to synchronize on cuda copy-coplete-event");

    assert(m_Postprocessor);
    /* Fill the host buffers information in the output. */
    RETURN_NVINFER_ERROR(
        m_Postprocessor->postProcessHost(*recyleBatch, batchOutput),
        "postprocessing host buffers failed.");

    /* Hold batch private data */
    batchOutput.priv = (void*)recyleBatch.release();
    return NVDSINFER_SUCCESS;
}

/**
 * Release a set of host buffers back to the context.
 */
void
NvDsInferContextImpl::releaseBatchOutput(NvDsInferContextBatchOutput &batchOutput)
{
    NvDsInferBatch *batch = (NvDsInferBatch *) batchOutput.priv;

    std::unique_lock<std::mutex> lock(m_BatchesMutex);

    /* Check for a valid id */
    if (std::find_if(m_Batches.begin(), m_Batches.end(),
            [batch](const std::shared_ptr<NvDsInferBatch> b) { return b.get() == batch; }) ==
        m_Batches.end())
    {
        printWarning("Tried to release an unknown outputBatchID");
        return;
    }

    /* And if the batch is not already with the context. */
    if (batch->m_BuffersWithContext)
    {
        printWarning("Tried to release an outputBatchID which is"
            " already with the context");
        return;
    }
    batch->m_BuffersWithContext = true;
    m_FreeBatchQueue.push(batch);

    assert(m_Postprocessor);
    m_Postprocessor->freeBatchOutput(batchOutput);
}

/**
 * Fill all the bound layers information in the vector.
 */
void
NvDsInferContextImpl::fillLayersInfo(std::vector<NvDsInferLayerInfo>& layersInfo)
{
    layersInfo.resize(m_AllLayerInfo.size());
    std::copy(m_AllLayerInfo.begin(), m_AllLayerInfo.end(), layersInfo.begin());
}

const std::vector<std::vector<std::string>>&
NvDsInferContextImpl::getLabels()
{
    assert(m_Postprocessor);
    return m_Postprocessor->getLabels();
}

/* Check if the runtime backend is compatible with requested configuration. */
NvDsInferStatus
NvDsInferContextImpl::checkBackendParams(BackendContext& ctx,
        const NvDsInferContextInitParams& initParams)
{
    NvDsInferBatchDims maxDims = ctx.getMaxBatchDims (INPUT_LAYER_INDEX);
    if (maxDims.batchSize < (int) initParams.maxBatchSize)
    {
        printWarning("Backend has maxBatchSize %d whereas %d has been requested",
                maxDims.batchSize, initParams.maxBatchSize);
        return NVDSINFER_CONFIG_FAILED;
    }

    if (initParams.inferInputDims.c && initParams.inferInputDims.h &&
        initParams.inferInputDims.w)
    {
        NvDsInferDims inputDims = {3,
            {initParams.inferInputDims.c, initParams.inferInputDims.h,
                initParams.inferInputDims.w},
            0};

        if (initParams.netInputOrder == NvDsInferTensorOrder_kNHWC)
        {
            inputDims.d[0] = initParams.inferInputDims.h;
            inputDims.d[1] = initParams.inferInputDims.w;
            inputDims.d[2] = initParams.inferInputDims.c;
        }

        NvDsInferBatchDims requestBatchDims{
            (int)initParams.maxBatchSize, inputDims};
        if (!ctx.canSupportBatchDims(INPUT_LAYER_INDEX, requestBatchDims))
        {
            printWarning("backend can not support dims:%s",
                safeStr(dims2Str(inputDims)));
            return NVDSINFER_CONFIG_FAILED;
        }
    }

    for (unsigned int i = 0; i < initParams.numOutputLayers; i++)
    {
        if(!ctx.isOutputLayer(initParams.outputLayerNames[i])) {
            printWarning("Could not find output layer '%s' in engine",
                initParams.outputLayerNames[i]);
        }
    }

    return NVDSINFER_SUCCESS;
}

bool
NvDsInferContextImpl::deserializeEngineAndBackend(const std::string enginePath,
        int dla, std::shared_ptr<TrtEngine>& engine,
        std::unique_ptr<BackendContext>& backend)
{
    auto builder = std::make_unique<TrtModelBuilder>(
        m_GpuID, *gTrtLogger, m_CustomLibHandle, true);
    assert(builder);

    std::shared_ptr<TrtEngine> newEngine =
        builder->deserializeEngine(enginePath, dla);
    if (!newEngine)
    {
        printWarning(
            "deserialize engine from file :%s failed", safeStr(enginePath));
        return false;
    }
    auto newBackend = createBackendContext(newEngine);
    if (!newBackend)
    {
        printWarning("create backend context from engine from file :%s failed",
            safeStr(enginePath));
        return false;
    }

    printInfo("deserialized trt engine from :%s", safeStr(enginePath));
    newEngine->printEngineInfo();

    engine = std::move(newEngine);
    backend = std::move(newBackend);
    return true;
}

/* Create engine and backend context for the model from the init params
 * (caffemodel & prototxt/uff/onnx, int8 calibration tables, etc) and return the
 * backend */
std::unique_ptr<BackendContext>
NvDsInferContextImpl::buildModel(NvDsInferContextInitParams& initParams)
{
    printInfo("Trying to create engine from model files");

    std::unique_ptr<TrtModelBuilder> builder =
        std::make_unique<TrtModelBuilder>(
            initParams.gpuID, *gTrtLogger, m_CustomLibHandle);
    assert(builder);

    if (!string_empty(initParams.int8CalibrationFilePath) &&
        file_accessible(initParams.int8CalibrationFilePath))
    {
        auto calibrator = std::make_unique<NvDsInferInt8Calibrator>(
            initParams.int8CalibrationFilePath);
        builder->setInt8Calibrator(std::move(calibrator));
    }

    std::string enginePath;
    std::shared_ptr<TrtEngine> engine =
        builder->buildModel(initParams, enginePath);
    if (!engine)
    {
        printError("build engine file failed");
        return nullptr;
    }

    if (builder->serializeEngine(enginePath, engine->engine()) !=
        NVDSINFER_SUCCESS)
    {
        printWarning(
            "failed to serialize cude engine to file: %s", safeStr(enginePath));
    }
    else
    {
        printInfo("serialize cuda engine to file: %s successfully",
            safeStr(enginePath));
    }

    std::unique_ptr<BackendContext> backend;
    auto newBackend = createBackendContext(engine);
    if (!newBackend)
    {
        printWarning("create backend context from engine failed");
        return nullptr;
    }

    engine->printEngineInfo();

    backend = std::move(newBackend);

    if (checkBackendParams(*backend, initParams) != NVDSINFER_SUCCESS)
    {
        printError(
            "deserialized backend context :%s failed to match config params",
            safeStr(enginePath));
        return nullptr;
    }

    builder.reset();

    return backend;
}

/* Deserialize engine and create backend context for the model from the init
 * params (caffemodel & prototxt/uff/onnx/etlt&key/custom-parser, int8
 * calibration tables, etc) and return the backend */

std::unique_ptr<BackendContext>
NvDsInferContextImpl::generateBackendContext(NvDsInferContextInitParams& initParams)
{
    int dla = -1;
    if (initParams.useDLA && initParams.dlaCore >= 0)
        dla = initParams.dlaCore;

    std::shared_ptr<TrtEngine> engine;
    std::unique_ptr<BackendContext> backend;
    if (!string_empty(initParams.modelEngineFilePath))
    {
        if (!deserializeEngineAndBackend(
                initParams.modelEngineFilePath, dla, engine, backend))
        {
            printWarning(
                "deserialize backend context from engine from file :%s failed, "
                "try rebuild",
                safeStr(initParams.modelEngineFilePath));
        }
    }

    if (backend &&
        checkBackendParams(*backend, initParams) == NVDSINFER_SUCCESS)
    {
        printInfo("Use deserialized engine model: %s",
            safeStr(initParams.modelEngineFilePath));
        return backend;
    }
    else if (backend)
    {
        printWarning(
            "deserialized backend context :%s failed to match config params, "
            "trying rebuild",
            safeStr(initParams.modelEngineFilePath));
        backend.reset();
        engine.reset();
    }

    backend = buildModel(initParams);
    if (!backend)
    {
        printError("build backend context failed");
        return nullptr;
    }

    return backend;
}

/**
 * Clean up and free all resources
 */
NvDsInferContextImpl::~NvDsInferContextImpl()
{
    /* Set the cuda device to be used. */
    cudaError_t cudaReturn = cudaSetDevice(m_GpuID);
    if (cudaReturn != cudaSuccess)
    {
        printError("Failed to set cuda device %d (%s).", m_GpuID,
                cudaGetErrorName(cudaReturn));
        return;
    }

    if (m_Preprocessor)
        m_Preprocessor->syncStream();

    /* Clean up other cuda resources. */
    if (m_InferStream)
    {
        cudaStreamSynchronize(*m_InferStream);
    }
    if (m_PostprocessStream)
    {
        cudaStreamSynchronize(*m_PostprocessStream);
    }

    m_BackendContext.reset();

    m_InferStream.reset();
    m_PostprocessStream.reset();
    m_InputConsumedEvent.reset();
    m_InferCompleteEvent.reset();

    m_Preprocessor.reset();
    m_Postprocessor.reset();

    m_OverwriteIpTensorFile.close();

    bool warn = false;

    std::unique_lock<std::mutex> lock(m_BatchesMutex);

    for (auto & batch:m_Batches)
    {
        if (!batch->m_BuffersWithContext && !warn)
        {
            warn = true;
            printWarning ("Not all output batches released back to the context "
                    "before destroy. Memory associated with the outputs will "
                    "no longer be valid.");
        }
        if (batch->m_OutputCopyDoneEvent)
        {
            cudaEventSynchronize(*batch->m_OutputCopyDoneEvent);
            batch->m_OutputCopyDoneEvent.reset();
        }
        batch->m_OutputDeviceBuffers.clear();
        batch->m_HostBuffers.clear();
    }
    m_Batches.clear();
    m_InputDeviceBuffers.clear();
    m_CustomLibHandle.reset();
}

/*
 * Destroy the context to release all resources.
 */
void
NvDsInferContextImpl::destroy()
{
    delete this;
}

} // namespace nvdsinfer

using namespace nvdsinfer;

/*
 * Factory function to create an NvDsInferContext instance and initialize it with
 * supplied parameters.
 */
NvDsInferStatus
createNvDsInferContext(NvDsInferContextHandle *handle,
        NvDsInferContextInitParams &initParams, void *userCtx,
        NvDsInferContextLoggingFunc logFunc)
{
    NvDsInferStatus status;
    NvDsInferContextImpl *ctx = new NvDsInferContextImpl();

    status = ctx->initialize(initParams, userCtx, logFunc);
    if (status == NVDSINFER_SUCCESS)
    {
        *handle = ctx;
    }
    else
    {
        static_cast<INvDsInferContext *>(ctx)->destroy();
    }
    return status;
}

/*
 * Reset the members inside the initParams structure to default values.
 */
void
NvDsInferContext_ResetInitParams (NvDsInferContextInitParams *initParams)
{
    if (initParams == nullptr)
    {
        fprintf(stderr, "Warning. NULL initParams passed to "
                "NvDsInferContext_ResetInitParams()\n");
        return;
    }

    memset(initParams, 0, sizeof (*initParams));

    initParams->networkMode = NvDsInferNetworkMode_FP32;
    initParams->networkInputFormat = NvDsInferFormat_Unknown;
    initParams->uffInputOrder = NvDsInferTensorOrder_kNCHW;
    initParams->netInputOrder = NvDsInferTensorOrder_kNCHW;
    initParams->maxBatchSize = 1;
    initParams->networkScaleFactor = 1.0;
    initParams->networkType = NvDsInferNetworkType_Detector;
    initParams->outputBufferPoolSize = NVDSINFER_MIN_OUTPUT_BUFFERPOOL_SIZE;
    initParams->autoIncMem = 1;
    initParams->maxGPUMemPer = 99.0;
}

const char *
NvDsInferContext_GetStatusName (NvDsInferStatus status)
{
    return NvDsInferStatus2Str(status);
}
