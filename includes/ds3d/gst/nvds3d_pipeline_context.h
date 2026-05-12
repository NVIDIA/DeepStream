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


#ifndef NVDS3D_GST_NVDS3D_PIPELINE_CONTEXT_H
#define NVDS3D_GST_NVDS3D_PIPELINE_CONTEXT_H

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

namespace ds3d { namespace gst {

constexpr const char* kDs3dFilterPluginName = "nvds3dfilter";
constexpr const char* kDs3dBridgePluginName = "nvds3dbridge";
constexpr const char* kDs3dMixerPluginName = "nvds3dmixer";

extern "C" {

static gboolean
SendEosOnSrc(GstElement* element, GstPad* pad, gpointer user_data)
{
    GstPad* peer = gst_pad_get_peer(pad);
    if (!peer) {
        LOG_WARNING(
            "send EOS downstream [elem:%s] skipped; not linked\n",
            GST_ELEMENT_NAME(GST_ELEMENT(gst_pad_get_parent(pad))));
        return TRUE;
    }
    LOG_DEBUG(
        "sending EOS downstream [elem:%s->%s]\n", GST_ELEMENT_NAME(GST_ELEMENT(gst_pad_get_parent(pad))),
        GST_ELEMENT_NAME(GST_ELEMENT(gst_pad_get_parent(peer))));
    if (gst_pad_send_event(peer, gst_event_new_eos()) == FALSE) {
        LOG_WARNING(
            "send EOS downstream [elem:%s->%s] failed\n", GST_ELEMENT_NAME(GST_ELEMENT(gst_pad_get_parent(pad))),
            GST_ELEMENT_NAME(GST_ELEMENT(gst_pad_get_parent(peer))));
    }
    gst_object_unref(peer);
    return TRUE;
}
}

class PipelineContext {
public:
    PipelineContext() {}
    virtual ~PipelineContext() { deinit(); }

    void setMainloop(GMainLoop* loop) { _mainLoop.reset(loop); }
    void setEosAutoQuit(bool enable) { _eosAutoQuit = enable; }

    virtual ErrCode init(const std::string& name)
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

    PipelineContext& add(const gst::ElePtr& ele)
    {
        DS_ASSERT(_pipeline);
        DS3D_THROW_ERROR(
            gst_bin_add(GST_BIN(pipeline()), ele.copy()), ErrCode::kGst, "add element failed");
        _elementList.emplace_back(ele);
        return *this;
    }

    virtual ErrCode start(std::function<void()> loopQuitCb)
    {
        LOG_DEBUG("starting");
        DS3D_ERROR_RETURN(playPipeline(), "failed to start playing the pipeline");
        DS3D_ERROR_RETURN(
            runMainLoop(std::move(loopQuitCb)), "failed to run main loop on the pipeline");
        return ErrCode::kGood;
    }

    virtual ErrCode stop()
    {
        LOG_DEBUG("stopping");
        if (mainLoop() && isRunning(1000)) {
            LOG_DEBUG("start sending EOS");
            sendEOS();
            std::unique_lock<std::mutex> lock(mutex());
            if (!_StatusCond.wait_for(lock, std::chrono::seconds(3), [this]() {
                    return _mainStopped || _eosReceived;
                })) {
                LOG_DEBUG("waiting for EOS timed out, force to stop");
            }
        }

        quitMainLoop();
        waitLoopQuit();
        return stopPipeline();
    }

    virtual void deinit()
    {
        LOG_DEBUG("deinit");
        if (bus()) {
            gst_bus_remove_watch(bus());
        }
        _bus.reset();
        _pipeline.reset();
        _mainLoop.reset();
        std::for_each(_elementList.rbegin(), _elementList.rend(), [](auto& e) { e.reset(); });
        _elementList.clear();
    }

    /* timeout: milliseconds, 0 means never timeout */
    bool isRunning(size_t timeout = 0)
    {
        std::unique_lock<std::mutex> locker(mutex());
        if (!mainLoop() || !pipeline() || _mainStopped || (_eosAutoQuit && _eosReceived)) {
            return false;
        }
        if (!g_main_loop_is_running(mainLoop())) {
            return false;
        }
        locker.unlock();

        GstState state = GST_STATE_NULL;
        GstState pending = GST_STATE_NULL;
        GstStateChangeReturn ret = gst_element_get_state(
            GST_ELEMENT(pipeline()), &state, &pending,
            (timeout ? timeout * 1000000 : GST_CLOCK_TIME_NONE));

        // multi-times try on get_state in case gstreamer is not maintening states well.
        uint32_t times = 1;
        while (ret == GST_STATE_CHANGE_FAILURE && times++ < 3) {
            ret = gst_element_get_state(GST_ELEMENT(pipeline()), &state, &pending, 0);
        }
        if (ret == GST_STATE_CHANGE_FAILURE) {
            return false;
        }
        if (state == GST_STATE_PLAYING || pending == GST_STATE_PLAYING) {
            return true;
        }
        return false;
    }

    void quitMainLoop()
    {
        std::unique_lock<std::mutex> locker(mutex());
        if (mainLoop()) {
            g_main_loop_quit(mainLoop());
        }
        _StatusCond.notify_all();
    }

    void waitLoopQuit()
    {
        std::unique_lock<std::mutex> locker(mutex());
        if (mainLoop() && !_mainStopped && _mainLoopThread) {
            if (_StatusCond.wait_for(locker, std::chrono::milliseconds(3000)) ==
                std::cv_status::timeout) {
                LOG_DEBUG("waiting loop timed out, force loop to stop");
            }
        }
        _mainStopped = true;
        if (_mainLoopThread) {
            auto swapThread = std::move(_mainLoopThread);
            _mainLoopThread.reset();
            locker.unlock();
            swapThread->join();
        }
    }

    ErrCode playPipeline()
    {
        DS_ASSERT(_pipeline);
        {
            std::unique_lock<std::mutex> locker(mutex());
            _eosReceived = false;
        }
        auto c = setPipelineState(GST_STATE_PLAYING);
        return c;
    }

    ErrCode stopPipeline()
    {
        if (!_pipeline) {
            return ErrCode::kGood;
        }
        ErrCode c = setPipelineState(GST_STATE_NULL);
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

    static gboolean GSourceCb(gpointer user_data)
    {
        std::function<bool()>* f = (std::function<bool()>*)(user_data);
        DS_ASSERT(f);
        return (*f)();
    }

    ErrCode runMainLoop(std::function<void()> loopQuitCb)
    {
        std::unique_lock<std::mutex> locker(mutex());
        DS3D_FAILED_RETURN(
            mainLoop() && !_mainLoopThread, ErrCode::kUnknown,
            "failed to run main loop due to loop might not set or thread already running.");

        _mainStopped = false;
        auto loopThread = std::make_unique<std::thread>([this, quitCb = std::move(loopQuitCb)]() {
            g_main_loop_run(mainLoop());
            quitCb();
            std::unique_lock<std::mutex> locker(mutex());
            _mainStopped = true;
            _StatusCond.notify_all();
        });
        DS_ASSERT(loopThread);

        // check g_main_loop_run started
        std::atomic_bool loopStarted{false};
        std::function<bool()> loopCheck = [&loopStarted, this]() -> bool {
            std::unique_lock<std::mutex> locker(mutex());
            loopStarted = true;
            _StatusCond.notify_all();
            return false;
        };
        g_idle_add(GSourceCb, &loopCheck);

        // TODO, find better timeout
        // set a larger timeout value since some model load taking long time
        if (!_StatusCond.wait_for(locker, std::chrono::milliseconds(20000), [this, &loopStarted]() {
                return loopStarted || _mainStopped;
            })) {
            locker.unlock();
            LOG_WARNING("Starting main loop timed out");
            quitMainLoop();
            loopThread->join();
            return ErrCode::kTimeOut;
        }

        // run main loop stopped too fast
        if (_mainStopped) {
            LOG_ERROR("Run main loop stopped too fast, please check.");
            locker.unlock();
            loopThread->join();
            return ErrCode::kUnknown;
        }
        _mainLoopThread = std::move(loopThread);

        return ErrCode::kGood;
    }

    ErrCode sendEOS()
    {
        LOG_INFO("pipeline sending EOS");
        GstIterator* itr = nullptr;
        GValue data = {
            0,
        };
        for (itr = gst_bin_iterate_sources(GST_BIN(pipeline())); gst_iterator_next(itr, &data) == GST_ITERATOR_OK;) {
            GstElement* elem = GST_ELEMENT_CAST(g_value_get_object(&data));
            LOG_DEBUG("sending EOS downstream from src element %s\n", GST_ELEMENT_NAME(elem));
            // operating pads directly might lose element's state lock, each element's function must
            // be thread-safe
            gst_element_foreach_src_pad(elem, SendEosOnSrc, NULL);
            g_value_reset(&data);
        }
        g_value_unset(&data);
        gst_iterator_free(itr);

        return ErrCode::kGood;
    }

    GstPipeline* pipeline() const { return GST_PIPELINE_CAST(_pipeline.get()); }
    GstBus* bus() const { return _bus.get(); }
    GMainLoop* mainLoop() const { return _mainLoop.get(); }

private:
    // default bus callback
    virtual bool busCall(GstMessage* msg)
    {
        DS_ASSERT(mainLoop());
        switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_EOS:
            LOG_INFO("End of stream received");
            {
                std::unique_lock<std::mutex> locker(mutex());
                _eosReceived = true;
                _StatusCond.notify_all();
            }
            if (_eosAutoQuit) {
                quitMainLoop();
            }
            break;
        case GST_MESSAGE_ERROR: {
            gchar* debug = nullptr;
            GError* error = nullptr;
            gst_message_parse_error(msg, &error, &debug);
            LOG_ERROR(
                "ERROR from element %s: %s, details: %s", GST_OBJECT_NAME(msg->src), error->message,
                (debug ? debug : ""));
            g_free(debug);
            g_error_free(error);

            quitMainLoop();
            break;
        }
        case GST_MESSAGE_STATE_CHANGED: {
            GstState oldState, newState, pendingState;

            gst_message_parse_state_changed(msg, &oldState, &newState, &pendingState);
            LOG_DEBUG(
                "Element %s changed state from %s to %s, pending: %s.", GST_OBJECT_NAME(msg->src),
                gst_element_state_get_name(oldState), gst_element_state_get_name(newState),
                gst_element_state_get_name(pendingState));
            break;
        }
        default:
            break;
        }
        return TRUE;
    }

protected:
    ErrCode setPipelineState(GstState state)
    {
        DS_ASSERT(_pipeline);
        return setState(_pipeline.get(), state);
    }

    ErrCode setState(GstElement* ele, GstState state)
    {
        DS_ASSERT(ele);
        GstStateChangeReturn ret = gst_element_set_state(ele, state);
        DS3D_FAILED_RETURN(
            ret != GST_STATE_CHANGE_FAILURE, ErrCode::kGst, "element set state: %d failed", state);
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
        PipelineContext* ctx = static_cast<PipelineContext*>(data);
        DS_ASSERT(ctx->bus() == bus);
        return ctx->busCall(msg);
    }

    std::mutex& mutex() const { return _pipelineMutex; }

    // members
    gst::ElePtr _pipeline;
    gst::BusPtr _bus;
    uint32_t _busWatchId = 0;
    std::vector<gst::ElePtr> _elementList;
    ds3d::UniqPtr<GMainLoop> _mainLoop{nullptr, g_main_loop_unref};
    bool _eosAutoQuit = false;
    std::unique_ptr<std::thread> _mainLoopThread;
    std::atomic_bool _mainStopped{false};
    std::atomic_bool _eosReceived{false};
    mutable std::mutex _pipelineMutex;
    std::condition_variable _StatusCond;
    DS3D_DISABLE_CLASS_COPY(PipelineContext);
};

}}  // namespace ds3d::gst

#endif  // NVDS3D_GST_NVDS3D_PIPELINE_CONTEXT_H
