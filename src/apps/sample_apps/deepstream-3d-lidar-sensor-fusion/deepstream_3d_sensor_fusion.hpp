/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef DEEPSTREAM_DS3D_SENSOR_FUSION_HPP
#define DEEPSTREAM_DS3D_SENSOR_FUSION_HPP

#include <unistd.h>

// inlcude all ds3d hpp header files
#include <ds3d/common/hpp/dataloader.hpp>
#include <ds3d/common/hpp/datamap.hpp>
#include <ds3d/common/hpp/frame.hpp>
#include <ds3d/common/hpp/yaml_config.hpp>
#include <ds3d/common/hpp/profiling.hpp>

// inlucde nvds3d Gst header files
#include <ds3d/gst/nvds3d_gst_plugin.h>
#include <ds3d/gst/nvds3d_gst_ptr.h>
#include <ds3d/gst/nvds3d_meta.h>
#include <ds3d/gst/nvds3d_pipeline_context.h>

#include "gstnvdsmeta.h"

using namespace ds3d;

#undef CHECK_ERROR
#define CHECK_ERROR(statement, fmt, ...) DS3D_FAILED_RETURN(statement, -1, fmt, ##__VA_ARGS__)

#undef RETURN_ERROR
#define RETURN_ERROR(statement, fmt, ...) DS3D_ERROR_RETURN(statement, fmt, ##__VA_ARGS__)

using ConfigList = std::vector<config::ComponentConfig>;

struct Component {
public:
    virtual ~Component() = default;
    bool isGstPlugin() const;
    config::ComponentType type() const { return config.type; }

    config::ComponentConfig config;
    gst::ElePtr gstElement;
};

struct AppProfiler {
public:
    config::ComponentConfig config;
    bool enableDebug = false;
    bool eosAutoStop = true;
    bool probeBuffer = false;

    AppProfiler() = default;
    AppProfiler(const AppProfiler&) = delete;
    void operator=(const AppProfiler&) = delete;
    ~AppProfiler() {}

    ErrCode parse(const config::ComponentConfig& compConf)
    {
        DS_ASSERT(compConf.type == config::ComponentType::kUserApp);
        this->config = compConf;
        YAML::Node node = YAML::Load(compConf.rawContent);
        auto debugNode = node["enable_debug"];
        auto eosNode = node["eos_auto_stop"];
        auto probeNode = node["probe_buffer"];
        if (debugNode) {
            enableDebug = debugNode.as<bool>();
        }

        if (eosNode) {
            eosAutoStop = eosNode.as<bool>();
        }
        if (probeNode) {
            probeBuffer = probeNode.as<bool>();
        }
        return ErrCode::kGood;
    }
};

class SensorFusionApp : public gst::PipelineContext {
public:
    SensorFusionApp() = default;
    ~SensorFusionApp()
    {
        stop();
        deinit();
    }

    ErrCode setup(const std::string& configPath, std::function<void()> windowClosed);
    ErrCode stop() override;
    void deinit() override;

    AppProfiler& profiler() { return _appProfiler; }

private:
    ErrCode init(const std::string& name) override;
    ErrCode buildComponents(const ConfigList& componentConfigs);
    ErrCode LinkComponents();

    std::unique_ptr<Component> createLoader(const config::ComponentConfig& c);
    std::unique_ptr<Component> createRender(const config::ComponentConfig& c);
    std::unique_ptr<Component> createFilterBridge(const config::ComponentConfig& c);
    std::unique_ptr<Component> createMixer(const config::ComponentConfig& c);
    std::unique_ptr<Component> createGstParseBin(const config::ComponentConfig& c);
    ErrCode setupProfiling();

private:
    std::string _configPath;
    std::map<std::string, Ptr<Component>> _components;
    std::function<void()> _windowClosedCb;
    AppProfiler _appProfiler;
};

#endif
