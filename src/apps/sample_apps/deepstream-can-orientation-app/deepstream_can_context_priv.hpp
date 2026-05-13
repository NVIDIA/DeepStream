/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef DS_APP_DEEPSTREAM_CAN_CONTEXT_PRIV_H
#define DS_APP_DEEPSTREAM_CAN_CONTEXT_PRIV_H

#include "deepstream_can_context.hpp"
#include <future>

#include <cuda_runtime_api.h>

#undef DS_CAN_FPS_INTERVAL
#define DS_CAN_FPS_INTERVAL 30

class Ds3dAppContext {
public:
    Ds3dAppContext() {}
    virtual ~Ds3dAppContext() { deinit(); }

    void setMainloop(GMainLoop* loop) { _mainLoop.reset(loop); }

    ErrCode init(const std::string& name)
    {
        int curDev = -1;
        cudaGetDevice(&curDev);
        struct cudaDeviceProp gpuProp;
        cudaGetDeviceProperties(&gpuProp, curDev);
        _isdGPU = (gpuProp.integrated ? false : true);

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
        {
            std::unique_lock<std::mutex> locker(mutex());
            _eosReceived = false;
        }
        auto c = setPipelineState(GST_STATE_PLAYING);
        return c;
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
        std::unique_lock<std::mutex> locker(mutex());
        if (!mainLoop() || !pipeline() || _mainStopped || _eosReceived) {
            return false;
        }
        locker.unlock();

        GstState state = GST_STATE_NULL;
        GstState pending = GST_STATE_NULL;
        GstStateChangeReturn ret = gst_element_get_state(
            GST_ELEMENT(pipeline()), &state, &pending,
            (timeout ? timeout * 1000000 : GST_CLOCK_TIME_NONE));

        // basler camera has change state issue, try multi-times
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
    }

    void waitLoopQuit()
    {
        std::unique_lock<std::mutex> locker(mutex());
        if (mainLoop() && !_mainStopped && _mainLoopThread) {
            _stoppedCond.wait_for(
                locker, std::chrono::milliseconds(100), [this]() { return _mainStopped; });
        }
        _mainStopped = true;
        if (_mainLoopThread) {
            auto swapThread = std::move(_mainLoopThread);
            _mainLoopThread.reset();
            locker.unlock();
            swapThread->join();
        }
    }

    void runMainLoop(std::function<void()> loopQuitCb)
    {
        std::unique_lock<std::mutex> locker(mutex());
        if (mainLoop() && !_mainLoopThread) {
            _mainStopped = false;
            _mainLoopThread = std::make_unique<std::thread>([this, cb = std::move(loopQuitCb)]() {
                g_main_loop_run(mainLoop());
                cb();
                std::unique_lock<std::mutex> locker(mutex());
                _mainStopped = true;
                _stoppedCond.notify_all();
            });
            DS_ASSERT(_mainLoopThread);
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
        Ds3dAppContext* ctx = static_cast<Ds3dAppContext*>(data);
        DS_ASSERT(ctx->bus() == bus);
        return ctx->busCall(msg);
    }

    std::mutex& mutex() const { return _streamMutex; }

    // members
    gst::ElePtr _pipeline;
    gst::BusPtr _bus;
    uint32_t _busWatchId = 0;
    std::vector<gst::ElePtr> _elementList;
    ds3d::UniqPtr<GMainLoop> _mainLoop{nullptr, g_main_loop_unref};
    bool _eosAutoQuit = false;
    std::unique_ptr<std::thread> _mainLoopThread;
    bool _mainStopped = false;
    bool _eosReceived = false;
    mutable std::mutex _streamMutex;
    std::condition_variable _stoppedCond;
    bool _isdGPU = true;
    DS3D_DISABLE_CLASS_COPY(Ds3dAppContext);
};

class CameraCanApp : public Ds3dAppContext {
public:
    CameraCanApp() = default;
    ~CameraCanApp() { deinit(); }

    void setConfig(const NvDsCanContextConfig& config) { _config = config; }
    const NvDsCanContextConfig& config() const { return _config; }

    ErrCode buildPipeline();

    ErrCode stop() override;

    void deinit() override
    {
        Ds3dAppContext::deinit();
        _src.reset();
        _appsrcFrameIdx = 0;
    }

    ErrCode processFrame(
        const NvDsCanContextFrame* frame, std::function<void(GstBuffer*)> callback);

    NvDsCanSrcType srcType() const { return _config.srcType; }

    static GstPadProbeReturn lastSinkBufferProbe(
        GstPad* pad, GstPadProbeInfo* info, gpointer udata);
    static GstPadProbeReturn processedBufferProbe(
        GstPad* pad, GstPadProbeInfo* info, gpointer udata);

private:
    bool busCall(GstMessage* msg) final;
    void outputBuffer(GstBuffer* buf);
    void visualMetaUpdate(GstBuffer* buf);
    void printOutputResult(GstBuffer* buf);

    gst::ElePtr createAppSrc(const NvDsCanFrameInfo& info);
    gst::ElePtr createSourceBin(const std::string& uri);
    gst::ElePtr createCameraSrc(const NvDsCanCameraInfo& conf);
    gst::ElePtr createSink();

private:
    NvDsCanContextConfig _config;
    gst::ElePtr _src;
    std::function<void(GstBuffer* buf)> _outputCallback = nullptr;
    uint64_t _appsrcFrameIdx = 0;
    profiling::FpsCalculation _fpsCalculator{DS_CAN_FPS_INTERVAL};
    uint64_t _bufId = 0;
};

#endif  // DS_APP_DEEPSTREAM_CAN_CONTEXT_PRIV_H
