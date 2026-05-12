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

#ifndef DS3D_APP_DEEPSTREAM_3D_CONTEXT_APP_H
#define DS3D_APP_DEEPSTREAM_3D_CONTEXT_APP_H

#include "gstnvdsmeta.h"

// inlcude all ds3d hpp header files
#include <ds3d/common/hpp/dataloader.hpp>
#include <ds3d/common/hpp/datamap.hpp>
#include <ds3d/common/hpp/frame.hpp>
#include <ds3d/common/hpp/yaml_config.hpp>

// inlucde nvds3d Gst header files
#include <ds3d/gst/nvds3d_gst_plugin.h>
#include <ds3d/gst/nvds3d_gst_ptr.h>
#include <ds3d/gst/nvds3d_meta.h>
#include <gst/gst.h>

namespace ds3d { namespace app {

class Ds3dAppContext {
public:
    Ds3dAppContext() {}
    virtual ~Ds3dAppContext() { deinit(); }

    void setMainloop(GMainLoop* loop) { _mainLoop.reset(loop); }

    ErrCode init(const std::string& name)
    {
        DS_ASSERT(_mainLoop);
        DS_ASSERT(!_pipeline);
        _pipeline.reset(gst_pipeline_new(name.c_str()));
        DS3D_FAILED_RETURN(pipeline(), ErrCode::kGst, "create pipeline: %s failed", name.c_str());
        _pipeline.setName(name);
        _bus.reset(gst_pipeline_get_bus(pipeline()));
        DS3D_FAILED_RETURN(bus(), ErrCode::kGst, "get bus from pipeline: %s failed", name.c_str());
        _busWatchId = gst_bus_add_watch(bus(), sBusCall, this);
        return ErrCode::kGood;
    }

    Ds3dAppContext& add(const gst::ElePtr& ele)
    {
        DS_ASSERT(_pipeline);
        DS3D_THROW_ERROR(
            gst_bin_add(GST_BIN(pipeline()), ele.copy()), ErrCode::kGst, "add element failed");
        _elementList.emplace_back(ele);
        return *this;
    }

    ErrCode play()
    {
        DS_ASSERT(_pipeline);
        return setPipelineState(GST_STATE_PLAYING);
    }

    virtual ErrCode stop()
    {
        DS_ASSERT(_pipeline);
        ErrCode c = setPipelineState(GST_STATE_NULL);
        if (!isGood(c)) {
            LOG_WARNING("set pipeline state to GST_STATE_NULL failed");
        }
        if (!isGood(c)) {
            LOG_WARNING("set pipeline state to GST_STATE_NULL failed");
        }
        GstState end = GST_STATE_NULL;
        c = getState(_pipeline.get(), &end, nullptr, 3000);
        if (!isGood(c) || end != GST_STATE_NULL) {
            LOG_WARNING("waiting for pipeline state to null failed, force to quit");
        }
        for (auto& each : _elementList) {
            if (each) {
                c = setState(each.get(), GST_STATE_NULL);
            }
        }
        return c;
    }

    /* timeout: milliseconds, 0 means never timeout */
    bool isRunning(size_t timeout = 0)
    {
        DS_ASSERT(pipeline());
        GstState state = GST_STATE_NULL;
        GstState pending = GST_STATE_NULL;
        if (gst_element_get_state(
                GST_ELEMENT(pipeline()), &state, &pending,
                (timeout ? timeout * 1000000 : GST_CLOCK_TIME_NONE)) == GST_STATE_CHANGE_FAILURE) {
            return false;
        }
        if (state == GST_STATE_PLAYING || pending == GST_STATE_PLAYING) {
            return true;
        }
        return false;
    }

    void quitMainLoop()
    {
        if (mainLoop()) {
            g_main_loop_quit(mainLoop());
        }
    }

    void runMainLoop()
    {
        if (mainLoop()) {
            g_main_loop_run(mainLoop());
        }
    }

    virtual void deinit()
    {
        if (bus()) {
            gst_bus_remove_watch(bus());
        }
        _bus.reset();
        _pipeline.reset();
        _elementList.clear();
        _mainLoop.reset();
    }

    ErrCode sendEOS()
    {
        DS3D_FAILED_RETURN(
            gst_element_send_event(GST_ELEMENT(pipeline()), gst_event_new_eos()), ErrCode::kGst,
            "send EOS failed");
        return ErrCode::kGood;
    }

    GstPipeline* pipeline() const { return GST_PIPELINE_CAST(_pipeline.get()); }
    GstBus* bus() const { return _bus.get(); }
    GMainLoop* mainLoop() const { return _mainLoop.get(); }

private:
    // no need to free msg
    virtual bool busCall(GstMessage* msg) = 0;

protected:
    ErrCode setPipelineState(GstState state)
    {
        DS_ASSERT(_pipeline);
        return setState(_pipeline.get(), state);
    }

    ErrCode setState(GstElement* ele, GstState state)
    {
        DS_ASSERT(ele);
        DS3D_FAILED_RETURN(
            gst_element_set_state(ele, state) != GST_STATE_CHANGE_FAILURE, ErrCode::kGst,
            "element set state: %d failed", state);
        return ErrCode::kGood;
    }
    /* get element states. timeout in milliseconds.
     */
    ErrCode getState(
        GstElement* ele, GstState* state, GstState* pending = nullptr, size_t timeout = 0)
    {
        DS_ASSERT(ele);
        GstStateChangeReturn ret = gst_element_get_state(
            ele, state, pending, (timeout ? timeout * 1000000 : GST_CLOCK_TIME_NONE));
        switch (ret) {
        case GST_STATE_CHANGE_FAILURE:
            return ErrCode::kGst;
        case GST_STATE_CHANGE_SUCCESS:
        case GST_STATE_CHANGE_NO_PREROLL:
            return ErrCode::kGood;
        default:
            return ErrCode::kUnknown;
        }
        return ErrCode::kGood;
    }

    static gboolean sBusCall(GstBus* bus, GstMessage* msg, gpointer data)
    {
        Ds3dAppContext* ctx = static_cast<Ds3dAppContext*>(data);
        DS_ASSERT(ctx->bus() == bus);
        return ctx->busCall(msg);
    }

    // members
    gst::ElePtr _pipeline;
    gst::BusPtr _bus;
    uint32_t _busWatchId = 0;
    std::vector<gst::ElePtr> _elementList;
    ds3d::UniqPtr<GMainLoop> _mainLoop{nullptr, g_main_loop_unref};
    DS3D_DISABLE_CLASS_COPY(Ds3dAppContext);
};

}}  // namespace ds3d::app

#endif  // DS3D_APP_DEEPSTREAM_3D_CONTEXT_APP_H
