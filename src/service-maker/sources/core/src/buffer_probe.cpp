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

#include "buffer_probe.hpp"
#include "element.hpp"
#include "pipeline.hpp"

#include <gst/gst.h>
#include "gst/buffer_probe.h"
#include "gstnvdsmeta.h"

namespace deepstream
{

  static GQuark _dsmeta_quark = g_quark_from_static_string(NVDS_META_STRING);

  GstPadProbeReturn
  genric_probe_callback(GstPad *pad, GstPadProbeInfo *info, gpointer u_data);

  /**
   * @brief Generic pad probe callback
   *
   * @param pad
   * @param info
   * @param u_data
   * @return GstPadProbeReturn
   */
  GstPadProbeReturn
  genric_probe_callback(GstPad *pad, GstPadProbeInfo *info, gpointer u_data) {
    BufferProbe *probe = (BufferProbe *)u_data;
    Element* element = probe->getTarget();
    const Pipeline& pipeline = element->getPipeline();
    OpaqueBuffer *inbuf = (OpaqueBuffer *)info->data;
    int pad_direction = gst_pad_get_direction(pad);
    GstMeta *gst_meta = NULL;
    gpointer state = NULL;

    if (pad_direction != GST_PAD_SRC) {
      // only support src pad probe for now
      return GST_PAD_PROBE_OK;
    }

    if (probe == NULL) {
      // unlikely, error
      throw std::runtime_error("Empty probe attached");
    }

    if (!pipeline.isRunning()) {
      return GST_PAD_PROBE_OK;
    }

    BufferProbe::IMetadataHandler *metadata_handler;
    BufferProbe::IBufferHandler *buffer_handler;
    probeReturn ret = probeReturn::Probe_Ok;
    if (probe->query(buffer_handler)) {
      Buffer buffer(inbuf);
      BufferProbe::IBufferObserver *buffer_observer;
      if (probe->query(buffer_observer)) {
        ret = buffer_observer->handleBuffer(*probe, buffer);
      }
      BufferProbe::IBufferOperator *buffer_operator;
      if (probe->query(buffer_operator)) {
        ret = buffer_operator->handleBuffer(*probe, buffer);
      }
      // this is a buffer probe
    }
    else if (probe->query(metadata_handler)) {
      // this is a metadata handler
      while ((gst_meta = gst_buffer_iterate_meta(inbuf, &state))) {
        if (gst_meta_api_type_has_tag(gst_meta->info->api, _dsmeta_quark)) {
          NvDsMeta *nvds_meta = (NvDsMeta *)gst_meta;
          if (nvds_meta->meta_type == NVDS_BATCH_GST_META) {
            BatchMetadata metadata(nvds_meta->meta_data);
            BufferProbe::IBatchMetadataObserver *batchmeta_handler;
            if (probe->query(batchmeta_handler)) {
              ret = batchmeta_handler->handleData(*probe, metadata);
            }
            BufferProbe::IBatchMetadataOperator *batchmeta_operator;
            if (probe->query(batchmeta_operator)) {
              ret = batchmeta_operator->handleData(*probe, metadata);
            }
          }
        }
      }
    }

    return ret == probeReturn::Probe_Ok ? GST_PAD_PROBE_OK:GST_PAD_PROBE_DROP;
  }

  BufferProbe::BufferProbe(const std::string &name, IHandler *handler)
      : CustomObject(BufferProbe::type(), nullptr, name), metadata_handler_(handler), target_(nullptr)
  {
  }

  BufferProbe::BufferProbe(const std::string &name, const char* factory, IHandler *handler)
      : CustomObject(BufferProbe::type(), factory, name), metadata_handler_(handler), target_(nullptr)
  {
  }

  BufferProbe::~BufferProbe()
  {
  }

  BufferProbe &BufferProbe::attach(Element *target, const Pad pad)
  {
    this->target_ = target;
    this->pad_ = pad;
    return *this;
  }

  unsigned long BufferProbe::type()
  {
    return GST_TYPE_BUFFER_PROBE;
  }

}