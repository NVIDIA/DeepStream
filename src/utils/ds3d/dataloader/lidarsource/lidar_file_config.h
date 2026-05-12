/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */


#ifndef _DS3D_DATALOADER_LIDARSOURCE_CONFIG_H
#define _DS3D_DATALOADER_LIDARSOURCE_CONFIG_H

#include <ds3d/common/common.h>
#include <ds3d/common/func_utils.h>

#include "ds3d/common/hpp/yaml_config.hpp"
#include "ds3d/common/idatatype.h"

#define _PATH_MAX 4096

namespace ds3d { namespace impl { namespace lidarsource {

struct DataParas {
    std::string location;
    uint64_t pts;
};

struct Config {
    config::ComponentConfig compConfig;
    std::string dataConfigFilePath = "";
    std::string realPath = "";
    MemType memType = MemType::kCpu;
    int gpuId = 0;
    std::vector<std::string> datamapKey;
    uint32_t pointNums = 0;
    uint32_t elementSize = 4;
    uint32_t elementStride = 0;
    bool fixedPointsNum = true;
    DataType dataType = DataType::kFp32;
    std::vector<std::deque<std::map<uint64_t, std::string>>> dataParas;
    uint64_t lastFileTimestamp = 0;
    uint32_t memPoolSize = 4;
    bool fileLoop = false;
    uint64_t frameDuration = 0;
    uint32_t sourceId = 0;
};

inline DataType getDataType(const std::string& dataType)
{
    DataType ret = DataType::kFp32;
    if(dataType == "FP32") {
        ret = DataType::kFp32;
    } else {
        LOG_WARNING("unsupported datatype: %s, fallback to FP32", dataType.c_str());
    }

    return ret;
}

inline ErrCode parseDataConfig(Config& config)
{
    char absRealFilePath[_PATH_MAX + 1];
    std::map<uint64_t, std::string> dataParas;
    std::deque<std::map<uint64_t, std::string>> dataParasQueue;
    YAML::Node node = YAML::LoadFile(config.realPath);
    const YAML::Node& listNode = node["source-list"];
    uint64_t timestampPrev = 0;
    for (std::size_t i = 0; i < listNode.size(); i++) {
        const YAML::Node& kvs = listNode[i];
        for (const auto& kv : kvs) {
            uint64_t timestamp = kv.first.as<uint64_t>();
            std::string filename = kv.second.as<std::string>();
            std::string absFilepath = "";
            if (filename[0] == '/') {
                absFilepath = filename;
            } else {
                int pos = config.realPath.find_last_of("/");
                std::string tmpPath = config.realPath.substr(0, pos+1);
                tmpPath = tmpPath + filename;
                if (!realpath(tmpPath.c_str(), absRealFilePath)) {
                    if (errno != ENOENT) {
                        LOG_WARNING("Your config file path is not right!");
                        return ErrCode::kNotFound;
                    }
                }
                absFilepath = absRealFilePath;
                LOG_DEBUG("lidar data path %s",absFilepath.c_str());
            }
            dataParas[timestamp] =  absFilepath;
            timestampPrev = config.lastFileTimestamp;
            config.lastFileTimestamp = timestamp;
        }
        dataParasQueue.push_back(dataParas);
    }

    config.dataParas.push_back(dataParasQueue);

    if (listNode.size() > 1) {
        config.frameDuration = config.lastFileTimestamp - timestampPrev;
    } else {
        // default to 100ms
        config.frameDuration = 100;
    }

    return ErrCode::kGood;
}

inline ErrCode
parseConfig(const std::string& content, const std::string& path, Config& config)
{
    DS3D_ERROR_RETURN(
        config::parseComponentConfig(content, path, config.compConfig),
        "parse lidarsource component content failed");
    char absRealFilePath[_PATH_MAX + 1];
    YAML::Node node = YAML::Load(config.compConfig.configBody);

    if (node["data_config_file"]) {
        auto lidarDataNode = node["data_config_file"];
        std::vector<std::string> lidarDataPaths;
        if (lidarDataNode.IsSequence()) {
            lidarDataPaths = lidarDataNode.as<std::vector<std::string>>();
        } else {
            // There is only one lidar data
            lidarDataPaths.resize(1);
            lidarDataPaths[0] = lidarDataNode.as<std::string>();
        }
        for (const auto& item : lidarDataPaths) {
            config.dataConfigFilePath = item;
            if (config.dataConfigFilePath[0] == '/') {
                config.realPath = config.dataConfigFilePath;
                LOG_DEBUG("config path %s",config.realPath.c_str());
            } else {
                if (!realpath(path.c_str(), absRealFilePath)) {
                    if (errno != ENOENT) {
                        LOG_WARNING("Your config file path is not right!");
                        return ErrCode::kNotFound;
                    }
                }
                config.realPath = absRealFilePath;
                int pos = config.realPath.find_last_of("/");
                std::string tmpPath = config.realPath.substr(0, pos+1);
                config.realPath = tmpPath + config.dataConfigFilePath;
                LOG_DEBUG("config path %s",config.realPath.c_str());
            }
            parseDataConfig(config);
        }
    }

    if (node["mem_type"]) {
        auto strType = node["mem_type"].as<std::string>();
        if (strncasecmp(strType.c_str(), "cpu", strType.size()) == 0) {
            config.memType = MemType::kCpu;
        } else if (strncasecmp(strType.c_str(), "gpu", strType.size()) == 0) {
            config.memType = MemType::kGpuCuda;
        } else {
            LOG_WARNING(
                "unknown mem_type: %s in lidar_file_source config parsing", strType.c_str());
        }
    }
    if (node["gpu_id"]) {
        config.gpuId = node["gpu_id"].as<int32_t>();
    }
    if (node["fixed_points_num"]) {
        config.fixedPointsNum = node["fixed_points_num"].as<bool>();
    }
    if (node["mem_pool_size"]) {
        config.memPoolSize = node["mem_pool_size"].as<uint32_t>();
    }
    if (node["data_type"]) {
        std::string dataType = node["data_type"].as<std::string>();
        config.dataType = getDataType(dataType);
    }
    if (node["points_num"]) {
        config.pointNums = node["points_num"].as<uint32_t>();
    }
    if (node["element_stride"]) {
        config.elementStride = node["element_stride"].as<uint32_t>();
    }
    if (node["element_size"]) {
        config.elementSize = node["element_size"].as<uint32_t>();
    }
    if (node["output_datamap_key"]) {
        auto keyNode = node["output_datamap_key"];
        if (keyNode.IsSequence()) {
            config.datamapKey = keyNode.as<std::vector<std::string>>();
        } else {
            config.datamapKey.resize(1);
            config.datamapKey[0] = keyNode.as<std::string>();
        }
    }
    if (node["file_loop"]) {
        config.fileLoop = node["file_loop"].as<bool>();
    }

    if (node["source_id"]) {
        config.sourceId = node["source_id"].as<uint32_t>();
    }

    if (config.elementStride == 0) {
        config.elementStride = config.elementSize;
    }

    DS3D_FAILED_RETURN(
        config.elementSize == 3 || config.elementSize == 4, ErrCode::kConfig,
        "lidar data config element_size: %d must be [3, 4].", config.elementSize);
    assert(config.elementStride >= config.elementSize);

    return ErrCode::kGood;
}

}}}  // namespace ds3d::impl::lidarsource

#endif  // _DS3D_DATALOADER_LIDARSOURCE_CONFIG_H
