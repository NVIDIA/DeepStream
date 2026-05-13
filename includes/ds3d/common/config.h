/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */


#ifndef DS3D_COMMON_HPP_CONFIG_HPP
#define DS3D_COMMON_HPP_CONFIG_HPP

#include <ds3d/common/common.h>
#include <ds3d/common/idatatype.h>
#include <ds3d/common/type_trait.h>

namespace ds3d { namespace config {

enum class ComponentType : int {
    kNone = 0,
    kDataLoader = 1,
    kDataFilter = 2,
    kDataRender = 3,
    kUserApp = 4,
    kDataBridge = 5,
    kDataMixer = 6,
    kGstParseBin = 10,
};

struct ComponentConfig {
    std::string name;
    ComponentType type = ComponentType::kNone;
    std::string gstInCaps;
    std::string gstOutCaps;
    std::string linkTo;
    std::string linkFrom;
    std::string withQueue;  // choose from [src, sink]
    std::string customLibPath;
    std::string customCreateFunction;
    std::string configBody;

    // raw data
    std::string rawContent;
    std::string filePath;
};

inline ComponentType
componentType(const std::string& strType)
{
    const static std::unordered_map<std::string, ComponentType> kTypeMap = {
        {"ds3d::dataloader", ComponentType::kDataLoader},
        {"ds3d::datafilter", ComponentType::kDataFilter},
        {"ds3d::datarender", ComponentType::kDataRender},
        {"ds3d::databridge", ComponentType::kDataBridge},
        {"ds3d::datamixer", ComponentType::kDataMixer},
        {"ds3d::userapp", ComponentType::kUserApp},
        {"ds3d::gstparsebin", ComponentType::kGstParseBin},
    };
    DS_ASSERT(!strType.empty());
    auto tpIt = kTypeMap.find(strType);
    DS3D_FAILED_RETURN(
        tpIt != kTypeMap.end(), ComponentType::kNone, "component type: %s is not supportted.",
        strType.c_str());
    return tpIt->second;
}

inline const char*
componentTypeStr(ComponentType type)
{
    switch(type) {
    case ComponentType::kDataLoader: return "ds3d::dataloader";
    case ComponentType::kDataFilter: return "ds3d::datafilter";
    case ComponentType::kDataRender: return "ds3d::datarender";
    case ComponentType::kDataBridge: return "ds3d::databridge";
    case ComponentType::kDataMixer: return "ds3d::datamixer";
    case ComponentType::kUserApp:
        return "ds3d::userapp";
    case ComponentType::kGstParseBin:
        return "ds3d::gstparsebin";
    default: return "ds3d::unknown::component";
    }
}

}}  // namespace ds3d::config

#endif  // DS3D_COMMON_HPP_CONFIG_HPP
