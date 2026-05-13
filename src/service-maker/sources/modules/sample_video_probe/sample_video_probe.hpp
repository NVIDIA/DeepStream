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
#include <string>
#include <memory>
#include <vector>

#include "buffer_probe.hpp"

using namespace std;

#define MAX_DISPLAY_LEN 64
#define PGIE_CLASS_ID_VEHICLE 0
#define PGIE_CLASS_ID_PERSON 2

namespace deepstream
{

  class CountMarker : public BufferProbe::IBatchMetadataOperator
  {
  public:
    CountMarker() {}
    virtual ~CountMarker() {}

    virtual probeReturn handleData(BufferProbe &probe, BatchMetadata &data)
    {
      FrameMetadata::Iterator frame_itr;
      for (data.initiateIterator(frame_itr); !frame_itr->done(); frame_itr->next())
      {
        auto vehicle_count = 0;
        auto person_count = 0;
        FrameMetadata &frame_meta = frame_itr->get();
        ObjectMetadata::Iterator obj_itr;
        for (frame_meta.initiateIterator(obj_itr); !obj_itr->done(); obj_itr->next())
        {
          ObjectMetadata &object_meta = obj_itr->get();
          auto class_id = object_meta.classId();
          if (class_id == PGIE_CLASS_ID_VEHICLE)
          {
            vehicle_count++;
          }
          else if (class_id == PGIE_CLASS_ID_PERSON)
          {
            person_count++;
          }
        }
        DisplayMetadata display_meta;
        if (data.acquire(display_meta))
        {
          std::stringstream ss;
          ss << "Person=" << person_count << ",Vehicle=" << vehicle_count;
          std::string str = ss.str();
          int font_size = 0;
          char font[] = "Serif";
          probe.getProperty("font-size", font_size);
          NvOSD_TextParams label = NvOSD_TextParams{
              (char *)str.c_str(),
              10,                                      //< x offset
              12,                                      //< y offset
              {font, (unsigned int)font_size, {1.0, 1.0, 1.0, 1.0}}, //< font and color
              1,
              {0.0, 0.0, 0.0, 1.0} //< background
          };
          display_meta.add(label);
          (*frame_itr)->append(display_meta);
        }
      }

      return probeReturn::Probe_Ok;
    }
  };

}