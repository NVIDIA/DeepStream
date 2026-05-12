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


#ifndef NVDS3D_GST_GST_PLUGINS_H
#define NVDS3D_GST_GST_PLUGINS_H

#include <ds3d/common/func_utils.h>
#include <ds3d/common/config.h>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>

#include <ds3d/common/hpp/obj.hpp>
#include <ds3d/common/hpp/dataloader.hpp>
#include <ds3d/common/hpp/datarender.hpp>
#include <ds3d/common/hpp/datafilter.hpp>
#include <ds3d/common/hpp/yaml_config.hpp>

#include <ds3d/gst/custom_lib_factory.h>
#include <ds3d/gst/nvds3d_gst_ptr.h>
#include <ds3d/gst/nvds3d_meta.h>

namespace ds3d {

constexpr uint32_t kDataProcessUserDataMagic = NVDS3D_MAGIC_ID('D', '3', 'U', 'D');
struct DataProcessUserData {
    uint32_t pluginMagicUD = kDataProcessUserDataMagic;
    CustomLibFactory* customlib = nullptr;
    std::string configContent;
    std::string configPath;
};

}  // namespace ds3d

DS3D_EXTERN_C_BEGIN

DS3D_EXPORT_API ds3d::ErrCode NvDs3D_GstAppSrcSetDataloader(
    GstAppSrc* src, ds3d::abiRefDataLoader* refLoader);

DS3D_EXPORT_API ds3d::ErrCode NvDs3D_GstAppSinkSetDataRender(
    GstAppSink* sink, ds3d::abiRefDataRender* refRender);

DS3D_EXTERN_C_END

namespace ds3d { namespace gst {

template<class GuardProcess>
struct DataProcessInfo {
    ElePtr gstElement;
    GuardProcess customProcessor;
    config::ComponentConfig config;
    Ptr<CustomLibFactory> customlib;

    DataProcessInfo() = default;
    ~DataProcessInfo() { reset(); }
    void reset()
    {
        gstElement.reset();
        customProcessor.reset();
        customlib.reset();
    }
};

using DataLoaderSrc = DataProcessInfo<GuardDataLoader>;
using DataRenderSink = DataProcessInfo<GuardDataRender>;

template <class GuardProcess>
inline ErrCode
loadCustomProcessor(
    const config::ComponentConfig& compConfig, GuardProcess& customProcessor,
    Ptr<CustomLibFactory>& lib)
{
    using abiRefType = typename GuardProcess::abiRefType;
    Ptr<CustomLibFactory> customLib(new CustomLibFactory);
    DS_ASSERT(customLib);
    GuardProcess processor;
    processor.reset(customLib->CreateCtx<abiRefType>(
        compConfig.customLibPath, compConfig.customCreateFunction));
    DS3D_FAILED_RETURN(
        processor, ErrCode::kLoadLib, "create custom processor: %s from lib: %s failed",
        componentTypeStr(compConfig.type), compConfig.customLibPath.c_str());

    // holder a customlib pointer to keep dataloader safe
    Ptr<DataProcessUserData> uData(new DataProcessUserData);
    uData->customlib = customLib.get();
    uData->configContent = compConfig.rawContent;
    uData->configPath = compConfig.filePath;
    processor.setUserData(uData.get(), [holder = customLib, uData = uData](void*) mutable {
        uData.reset();
        if (holder && holder.use_count() == 1) {
            holder->keepOpen(true);
        }
        holder.reset();
    });
    customProcessor = std::move(processor);
    lib = std::move(customLib);
    return ErrCode::kGood;
}

/**
 * @brief Generate DataLoaderSrc from config file
 *
 * @param configStr [in] config file content
 * @param configFilePath [in] config file path
 * @param DataLoaderSrc [out] new created dataloader and appsrc from config files
 * @param start [in] start dataloader immediately
 * @return ds3d::ErrCode
 */
inline ErrCode
NvDs3D_CreateDataLoaderSrc(
    const config::ComponentConfig& compConfig, DataLoaderSrc& loaderSrc, bool start)
{
    GuardDataLoader loader;
    DS3D_ERROR_RETURN(
        loadCustomProcessor(compConfig, loader, loaderSrc.customlib),
        "load custom dataloader failed");

    loaderSrc.config = compConfig;
    loaderSrc.customProcessor = loader;
    ElePtr loaderEle = elementMake("appsrc", compConfig.name);
    DS3D_FAILED_RETURN(loaderEle, ErrCode::kGst, "create appsrc failed.");
    // update loaderSrc appsrc
    loaderSrc.gstElement = loaderEle;

    // set gst element properties
    constexpr static size_t kPoolSize = 6;
    std::string caps = compConfig.gstOutCaps.empty() ? loader.getOutputCaps()
                                                     : compConfig.gstOutCaps;
    DS3D_FAILED_RETURN(
        !caps.empty(), ErrCode::kConfig, "caps must be configure for dataloader source");
    CapsPtr srcCaps(gst_caps_from_string(caps.c_str()));
    DS3D_FAILED_RETURN(srcCaps, ErrCode::kGst, "gst_caps_from_string: %s failed", caps.c_str());
    g_object_set(
        G_OBJECT(loaderEle.get()), "do-timestamp", TRUE, "stream-type", GST_APP_STREAM_TYPE_STREAM,
        "max-bytes", (uint64_t)(kPoolSize * sizeof(NvDs3DBuffer)), "min-percent", 80, "caps",
        srcCaps.get(), NULL);

    if (start) {
        DS3D_ERROR_RETURN(loader.start(compConfig.rawContent, compConfig.filePath),
        "Dataloader start config: %s failed", compConfig.filePath.c_str());
    }

    GstAppSrc* appSrc = GST_APP_SRC(loaderEle.get());
    DS_ASSERT(appSrc);
    DS3D_ERROR_RETURN(
        NvDs3D_GstAppSrcSetDataloader(appSrc, loader.release()),
        "Set dataloader into GstAppsrc failed.");

    return ErrCode::kGood;
}

inline ErrCode
NvDs3D_CreateDataRenderSink(
    const config::ComponentConfig& compConfig, DataRenderSink& renderSink, bool start)
{
    GuardDataRender render;
    DS3D_ERROR_RETURN(
        loadCustomProcessor(compConfig, render, renderSink.customlib),
        "load custom datarender failed");

    renderSink.config = compConfig;
    renderSink.customProcessor = render;
    ElePtr renderEle = elementMake("appsink", compConfig.name);
    DS3D_FAILED_RETURN(renderEle, ErrCode::kGst, "create appsink failed.");
    // update renderSink appsink
    renderSink.gstElement = renderEle;

    // set gst element properties
    uint32_t maxBuffers = 4;
    std::string caps = compConfig.gstInCaps.empty() ? render.getInputCaps()
                                                     : compConfig.gstInCaps;
    DS3D_FAILED_RETURN(
        !caps.empty(), ErrCode::kConfig, "caps must be configure for datarender source");
    CapsPtr sinkCaps(gst_caps_from_string(caps.c_str()));
    DS3D_FAILED_RETURN(sinkCaps, ErrCode::kGst, "gst_caps_from_string: %s failed", caps.c_str());
    GObject* eleObj = G_OBJECT(renderEle.get());
    g_object_set(
        eleObj, "wait-on-eos", TRUE, "max-buffers", (uint32_t)maxBuffers, "caps", sinkCaps.get(),
        nullptr);

    auto setGstProperties = [eleObj, &compConfig]() {
        YAML::Node node = YAML::Load(compConfig.rawContent);
        auto properties = node["gst_properties"];
        if (properties) {
            auto sync = properties["sync"];
            auto async = properties["async"];
            auto drop = properties["drop"];
            if (sync) {
                g_object_set(eleObj, "sync", sync.as<bool>(), nullptr);
            }
            if (async) {
                g_object_set(eleObj, "async", async.as<bool>(), nullptr);
            }
            if (drop) {
                g_object_set(eleObj, "drop", drop.as<bool>(), nullptr);
            }
        }
        return ErrCode::kGood;
    };
    DS3D_ERROR_RETURN(
        config::CatchYamlCall(setGstProperties), "parse gst_properties failed for datareander");

    if (start) {
        DS3D_ERROR_RETURN(render.start(compConfig.rawContent, compConfig.filePath),
        "Dataloader start config: %s failed", compConfig.filePath.c_str());
    }

    GstAppSink* appSink = GST_APP_SINK(renderEle.get());
    DS_ASSERT(appSink);
    DS3D_ERROR_RETURN(
        NvDs3D_GstAppSinkSetDataRender(appSink, render.release()),
        "Set datarender into GstAppSink failed.");

    return ErrCode::kGood;
}

}}  // namespace ds3d::gst

#endif  // NVDS3D_GST_GST_PLUGINS_H