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


#ifndef _DS3D_COMMON_ABI_DATA_OBJ__H
#define _DS3D_COMMON_ABI_DATA_OBJ__H

#include <ds3d/common/common.h>
#include <ds3d/common/idatatype.h>
#include <ds3d/common/type_trait.h>

namespace ds3d {

struct abiRefObj {
    virtual void destroy() = 0;
    virtual abiRefObj* refCopy_i() const = 0;
    virtual ~abiRefObj() = default;
};

enum class DataMapPolicy : int {
    kCopyPolicyNone = 0,
    /** copy from one datamap to another with
     * key_string manipulated with %d+key_string+%d
     * where %d is an integer Id passed with copy_i() @policyData (int32_t*)
     */
    kCopyWithIntegerId = 1,
};

template <class T>
struct abiRefT : public abiRefObj {
    virtual T* data() const = 0;
    abiRefT* refCopy() const { return dynamic_cast<abiRefT*>(refCopy_i()); }
};

typedef abiRefT<void> abiRefAny;

class abiDataMap {
public:
    abiDataMap() = default;
    virtual ~abiDataMap() = default;
    virtual ErrCode setBuf_i(const char* key, TIdType tid, abiRefAny* data) = 0;
    virtual ErrCode getBuf_i(const char* key, TIdType tid, const abiRefAny*& data) const = 0;
    virtual ErrCode removeBuf_i(const char* key) = 0;
    virtual bool has_i(const char* key) const = 0;
    virtual ErrCode clear_i() = 0;
    // clone input datamap into this datamap
    // existing keys' values in this datamap will be updated with values from input
    virtual ErrCode copy_i(abiDataMap* input, DataMapPolicy policy, char* policyData) = 0;
    // key provided will be used to selectively copy just that entry from input
    virtual ErrCode copy_i(abiDataMap* input, const char* key, DataMapPolicy policy, char* policyData) = 0;
    //list all entries in a datamap
    virtual int32_t getSize_i() = 0;
    // print list of keys in the map
    virtual void printDebug_i() const = 0;
    DS3D_DISABLE_CLASS_COPY(abiDataMap);
};

typedef abiRefT<abiDataMap> abiRefDataMap;

// define Callback abi interfaces
template <typename... Args>
struct abiCallBackT : public abiRefObj {
    using CppFunc = std::function<void(Args...)>;
    virtual void notify(Args... args) = 0;
    abiCallBackT* refCopy() const { return dynamic_cast<abiCallBackT*>(refCopy_i()); }
};

typedef abiCallBackT<ErrCode, const char*> abiErrorCB;
typedef abiCallBackT<ErrCode, const abiRefDataMap*> abiOnDataCB;
typedef abiCallBackT<ErrCode, const struct VideoBridge2dInput*> abiOnBridgeDataCB;

}  // namespace ds3d

#endif  // _DS3D_COMMON_ABI_DATA_OBJ__H
