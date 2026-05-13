/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <iostream>

#include "data_receiver.hpp"

#define PGIE_CLASS_ID_VEHICLE 0
#define PGIE_CLASS_ID_PERSON 2

namespace deepstream {

class ObjectCounter : public DataReceiver::IDataConsumer {
 public:
  virtual int consume(DataReceiver& receiver, Buffer buffer) {
    VideoBuffer video_buffer = buffer;
    BatchMetadata batch_meta = video_buffer.getBatchMetadata();
    batch_meta.iterate([](const FrameMetadata& frame_data) {
      auto vehicle_count = 0;
      auto person_count = 0;
      frame_data.iterate([&](const ObjectMetadata& object_data) {
        auto class_id = object_data.classId();
        if (class_id == PGIE_CLASS_ID_VEHICLE) {
          vehicle_count++;
        } else if (class_id == PGIE_CLASS_ID_PERSON ) {
          person_count++;
        }
      });
      std::cout << "Object Counter: " <<
        " Pad Idx = " << frame_data.padIndex() <<
        " Frame Number = " << frame_data.frameNum() <<
        " Vehicle Count = " << vehicle_count <<
        " Person Count = " << person_count << std::endl;
    });
    return 1;
  }
};
}