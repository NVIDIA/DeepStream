/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */


#ifndef DS3D_DATAFILTER_LIDAR_PREPROCESS_CONFIG_H
#define DS3D_DATAFILTER_LIDAR_PREPROCESS_CONFIG_H

#include <ds3d/common/common.h>
#include <ds3d/common/func_utils.h>

#include "ds3d/common/hpp/yaml_config.hpp"

#include "infer_utils.h"

using namespace nvdsinferserver;


namespace ds3d { namespace impl { namespace filter {

struct ModelInputDesc : public InferBufferDescription {
    std::string fromKey;
    bool is2DFrame = false;
};

struct Config {
    config::ComponentConfig compConfig;
    std::string filterInputDatamapKey;
    uint32_t memPoolSize = 8;
    std::map<size_t, std::string> modelInputs;
    uint gpuid;
    InferMemType inputTensorMemType = InferMemType::kGpuCuda;
    std::map<std::string, ModelInputDesc> inputLayersDes;
    //app's cfg path
    std::string configPath;
    std::vector<std::string> lidarDataFrom;

    bool check()
    {
        if (!memPoolSize) {
            LOG_ERROR("lidarpreprocess configed wrong mem_pool_size");
            return false;
        }
        return true;
    }
};

inline ErrCode
parseConfig(const std::string& content, const std::string& path, Config& config)
{
    DS3D_ERROR_RETURN(
        config::parseComponentConfig(content, path, config.compConfig),
        "parse dataloader component content failed");
    YAML::Node node = YAML::Load(config.compConfig.configBody);
    auto filterInputDatamapKey = node["filter_input_datamap_key"];
    auto yMemPoolSize = node["mem_pool_size"];
    auto modelInputs = node["model_inputs"];
    auto gpuid = node["gpu_id"];
    auto inputTensorMemType = node["input_tensor_mem_type"];
    auto configPath = node["config_path"];
    auto lidarDataFrom = node["lidar_data_from"];

    if (filterInputDatamapKey) {
        config.filterInputDatamapKey =  filterInputDatamapKey.as<std::string>();
    }
    if (yMemPoolSize) {
        config.memPoolSize = yMemPoolSize.as<uint32_t>();
    }
    if (modelInputs) {
        for (const auto& item :modelInputs) {
            ModelInputDesc des;
            if (item["name"])
                des.name = item["name"].as<std::string>();  // name
            std::string dataType;
            if (item["datatype"]){
                dataType = item["datatype"].as<std::string>();
                des.dataType = grpcStr2DataType(dataType);  //datatype
            }
            std::vector<std::string> strStreams = item["shape"].as<std::vector<std::string>>();
            des.dims.numDims = strStreams.size();
            for (size_t i = 0; i < des.dims.numDims; i++){
                des.dims.d[i] = std::stoi(strStreams[i]);
            }
            des.dims.numElements = std::accumulate(des.dims.d, des.dims.d + des.dims.numDims, 1,
                [](int s, int i) { return s * i; });
            if (item["from"]) {
                des.fromKey = item["from"].as<std::string>();
            }
            if (item["is_2d_frame"]) {
                des.is2DFrame = item["is_2d_frame"].as<bool>();
            }
            LOG_INFO(
                "modelInputs name:%s, dataType:%s, ndataType:%d, numDims:%d, numElements:%d, from: "
                "%s",
                des.name.c_str(), dataType.c_str(), (int)des.dataType, des.dims.numDims,
                des.dims.numElements, des.fromKey.c_str());
            config.inputLayersDes.emplace(des.name, des);
        }
    }
    if (gpuid) {
        config.gpuid = gpuid.as<uint>();
    }
    if (inputTensorMemType) {
        std::string value = inputTensorMemType.as<std::string>();
        if(!value.compare("GpuCuda")){
            config.inputTensorMemType = InferMemType::kGpuCuda;
        } else if (!value.compare("CpuCuda") || !value.compare("Cpu")) {
            config.inputTensorMemType = InferMemType::kCpuCuda;
        } else {
            LOG_WARNING("%s not supported, use defualt value GpuCuda", value.c_str());
        }
    }
    if (configPath) {
        config.configPath = configPath.as<std::string>();
    }
    if (lidarDataFrom) {
        config.lidarDataFrom = lidarDataFrom.as<std::vector<std::string>>();
    }

    LOG_INFO("memPoolSize: %d", config.memPoolSize);
    LOG_INFO("gpuid: %d", config.gpuid);
    LOG_INFO("filterInputDatamapKey: %s", config.filterInputDatamapKey.c_str());
    LOG_INFO("inputTensorMemType: %d", (int)config.inputTensorMemType);
    LOG_INFO("configPath: %s", config.configPath.c_str());

    //config data checking
    if (config.inputLayersDes.size() == 0) {
       LOG_ERROR("model_inputs parse failed, abort");
       abort();
    }
    if (config.lidarDataFrom.size() == 0) {
       LOG_ERROR("no lidar data, abort");
       abort();
    }

    return ErrCode::kGood;
}

}}}  // namespace ds3d::impl::filter

#endif  // DS3D_DATAFILTER_LIDAR_PREPROCESS_CONFIG_H
