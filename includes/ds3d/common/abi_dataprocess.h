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


#ifndef _DS3D_COMMON_ABI_PROCESS_H
#define _DS3D_COMMON_ABI_PROCESS_H

#include <ds3d/common/common.h>
#include <ds3d/common/abi_obj.h>
#include <ds3d/common/abi_window.h>

namespace ds3d {

/**
 * @brief All custom-libs need create the abi reference for DataLoader,
 *   DataRender, and DataFilter. Take dataloader for example,
 *   custom-lib: libnvds_custom_loader.so has exported
 *   function: NvDsCreateFakeDataloader
 *   Users can get this loader through
 *      typedef abiRefDataLoader*(*LoaderCreateFunc)();
 *      void* handle = dlopen("libnvds_custom_loader.so");
 *      auto creator = (LoaderCreateFunc)dlsym(handle, "NvDsCreateFakeDataloader");
 *      abiRefDataLoader* loader = creator();
 *      GuardDataLoader guardLoader(loader, true); // true for take ownership
 *      loader.setUserData(ctx, [ctx](void*){ delete ctx });
 *      loader.setErrorCallback([](ErrCode c, const char* msg){ printf("error msg: %s", msg); });
 *      ErrCode c = loader.start(config, configpath);
 *      DS_ASSERT(isGood(c));
 *      DS_ASSERT(loader.getState() == State::kRunning)
 *      while(...) {
 *          GuardDataMap datamap;
 *          ErrCode c = loader.readData(datamap);
 *          // process datamap
 *      }
 *      c = loader.stop();
 *      DS_ASSERT(loader.getState() == State::kStopped)
 *      loader.reset(); // destroy dataloader reference, when last reference
 *                      // destroyed, actual derived dataLoader will be deleted.
 */

// abiDataProcess state in runtime
enum class State : int {
    kNone = 0,
    kStarting,
    kRunning,
    kStopped,
};

enum class MixerUpdate : int { kPortAdded = 0, kPortRemoved = 1 };

// abiDataProcess port for input and output directions
enum class CapsPort : int {
    kInput = 0,
    kOutput,
};

// Process Interface, all abi processor must derived from abiProcess
class abiProcess {
public:
    abiProcess() = default;
    virtual ~abiProcess() = default;
    // set application user data
    virtual void setUserData_i(const abiRefAny* userdata) = 0;
    // get application user data back, abiProcess continues keeping the ref copy
    // untill another setuserData_i called or abiProcess destroyed
    virtual const abiRefAny* getUserData_i() const = 0;
    // set callback function when error happens or status changed.
    virtual void setErrorCallback_i(const abiErrorCB& cb) = 0;
    // check process state, e.g. whether it is in kRunning or kStopped
    virtual State state_i() const = 0;
    // start the process, before start, the state is kNone,
    // during the start progress, the state is kStarting.
    // Once start successfully, the state would became kRunning.
    virtual ErrCode start_i(const char* configStr, uint32_t strLen, const char* path) = 0;
    // stop the whole process and stop all resources.
    // implementation will check the state, if it is not in kRunning/kStarting mode, will
    // do nothing and return ErrCode::kGood
    virtual ErrCode stop_i() = 0;
    // get caps string for CapsPort::kInput or CapsPort::kOutput
    virtual const char* getCaps_i(CapsPort p) const = 0;

    // flush all unprocessed data, specificall when End-Of-Stream received
    virtual ErrCode flush_i() = 0;

private:
    DS3D_DISABLE_CLASS_COPY(abiProcess);
};

// ABI Interface for dataloader
class abiDataLoader : public abiProcess {
public:
    abiDataLoader() = default;
    ~abiDataLoader() override = default;
    // read data in sync mode
    virtual ErrCode readData_i(abiRefDataMap*& datamap) = 0;
    // read data in async mode, datamap would be invoked in dataReadyCb once it is ready.
    virtual ErrCode readDataAsync_i(const abiOnDataCB* dataReadyCb) = 0;
};

// ABI Interface for dataFilter
class abiDataFilter : public abiProcess {
public:
    abiDataFilter() = default;
    ~abiDataFilter() override = default;
    // render_i could be in sync or async mode
    virtual ErrCode process_i(
        const abiRefDataMap* inputData, const abiOnDataCB* outputDataCb,
        const abiOnDataCB* dataConsumedCb) = 0;
};

// ABI Interface for dataBridge
class abiDataBridge : public abiProcess {
public:
    abiDataBridge() = default;
    ~abiDataBridge() override = default;
    // render_i could be in sync or async mode
    virtual ErrCode process_i(
        const struct VideoBridge2dInput* inputData, const abiOnDataCB* outputDataCb,
        const abiOnBridgeDataCB* dataConsumedCb) = 0;
};

// ABI Interface for dataRender
class abiDataRender : public abiProcess {
public:
    abiDataRender() = default;
    ~abiDataRender() override = default;
    // get native created window, user can get window when state_i() == State::kRunning
    virtual const abiRefWindow* getWindow_i() const = 0;

    // prepare some dynamic resource when 1st buffer is coming
    virtual ErrCode preroll_i(const abiRefDataMap* inputData) = 0;
    // render_i could be in sync or async mode
    virtual ErrCode render_i(const abiRefDataMap* inputData, const abiOnDataCB* dataDoneCb) = 0;
};

// ABI Interface for dataMixer
class abiDataMixer : public abiProcess {
public:
    abiDataMixer() = default;
    ~abiDataMixer() override = default;
    virtual ErrCode process_i(
        const int portId,
        const abiRefDataMap* inputData,
        const abiOnDataCB* dataConsumedCb) = 0;
    /**
     * @brief to configure mixer framerate and output callback
     * @param outputDataCb The output datamap generated by mixer will be
     *        passed back to the caller using this registered callback.
     */
    virtual ErrCode setOutputCb_i(
        const abiOnDataCB* outputDataCb) = 0;
    virtual ErrCode updateInput_i(int portId, MixerUpdate updateType) = 0;
};

// raw reference pointer for abiDataLoader
typedef abiRefT<abiDataLoader> abiRefDataLoader;

// raw reference pointer for abiDataFilter
typedef abiRefT<abiDataFilter> abiRefDataFilter;

// raw reference pointer for abiDataBridge
typedef abiRefT<abiDataBridge> abiRefDataBridge;

// raw reference pointer for abiDataRender
typedef abiRefT<abiDataRender> abiRefDataRender;

// raw reference pointer for abiDataMixer
typedef abiRefT<abiDataMixer> abiRefDataMixer;

}  // namespace ds3d

#endif  // _DS3D_COMMON_ABI_PROCESS_H
