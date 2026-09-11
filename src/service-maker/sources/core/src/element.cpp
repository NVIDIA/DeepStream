/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "element.hpp"
#include "buffer_probe.hpp"
#include "common_factory.hpp"
#include "signal_handler.hpp"
#include "signal_emitter.hpp"

#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <cstdarg>
#include <gst/gst.h>

#include <dlfcn.h>


using namespace std;
using namespace deepstream;


namespace deepstream {
extern GstPadProbeReturn
genric_probe_callback (GstPad * pad, GstPadProbeInfo * info, gpointer u_data);
}

static std::map<Element::State, GstState> gstStateMap =
{
    { Element::INVALID, GST_STATE_VOID_PENDING },
    { Element::EMPTY  , GST_STATE_NULL         },
    { Element::READY  , GST_STATE_READY        },
    { Element::PAUSED , GST_STATE_PAUSED       },
    { Element::PLAYING, GST_STATE_PLAYING      }
};

// Get Source/Sink Pad Template List for an Element
static std::vector<GstPadTemplate*> get_element_pad_template_list (GstElement* element, GstPadDirection pad_direction) {
  std::vector<GstPadTemplate*> pad_template_list;
  GList* pads = gst_element_get_pad_template_list (element);
  for (; pads != NULL; pads = g_list_next(pads)) {
    GstPadTemplate *pad_template;
    pad_template = (GstPadTemplate *) (pads->data);
    if (pad_template->direction == pad_direction){
      pad_template_list.push_back (pad_template);
    }
  }
  return pad_template_list;
}

// Check whether a specific pad name (e.g. "sink_0") is an instance of a
// pad-template pattern (e.g. "sink_%u").  Also returns true on exact match.
static bool pad_name_matches_template(const std::string& pad_name,
                                      const char* name_template) {
  if (pad_name == name_template) return true;

  std::string tmpl(name_template);
  size_t pct = tmpl.find('%');
  if (pct == std::string::npos) return false;

  if (pad_name.size() <= pct) return false;
  if (pad_name.compare(0, pct, tmpl, 0, pct) != 0) return false;

  // skip the format specifier character after '%' (u, d, s, …)
  size_t spec_end = pct + 2;
  std::string suffix = (spec_end < tmpl.size()) ? tmpl.substr(spec_end) : "";

  if (!suffix.empty()) {
    if (pad_name.size() < pct + 1 + suffix.size()) return false;
    if (pad_name.compare(pad_name.size() - suffix.size(),
                         suffix.size(), suffix) != 0)
      return false;
  }
  return true;
}

// Get the available on-request pad for an element based on the pad template.
// When |specific_name| is provided (e.g. "sink_0"), that exact pad is
// requested instead of letting GStreamer auto-assign the next index.
static GstPad* get_element_request_pad (GstElement* element,
                                        GstPadTemplate* pad_template,
                                        const std::string& specific_name = "") {
  if (!specific_name.empty() && specific_name != pad_template->name_template) {
    GstPad* pad = gst_element_get_request_pad(element, specific_name.c_str());
    if (pad) return pad;
  }

  GstPad* pad =  gst_element_get_request_pad(element, pad_template->name_template);
      if (!pad) {
        gchar pad_name[64];
        for (int i = 0; i < 1024; i++) {
          g_snprintf(pad_name, sizeof(pad_name), pad_template->name_template, i);
          GstPad* test_pad = gst_element_get_static_pad(element, pad_name);
          if (!test_pad) {
            pad = gst_element_get_request_pad(element, pad_name);
            if (pad) break;
          }
          gst_object_unref (test_pad);
        }
      }

    return pad;
}


// Call Back Function for dynamic pad linking
static void cb_newpad (GstElement * element, GstPad * pad, gpointer data) {
  GstCaps *caps = gst_pad_query_caps (pad, NULL);
  const GstStructure *str = gst_caps_get_structure (caps, 0);
  const gchar *name = gst_structure_get_name (str);

  g_print("*** Inside %s name=%s\n", __func__, name);
  // gst_pad_can_link is used here to handle cases when more than one call back is attached to the element
  // example nvurisrcbin->audiconvert & nvurisrcbin->videoconvert
  // one call back will be attached for audio & one for video
  // whenever a pad is created both call backs will be called
  if (gst_pad_can_link (pad, (GstPad *) data)) {
    if (gst_pad_link (pad, (GstPad *) data) != GST_PAD_LINK_OK) {
      g_printerr("Failed to link  %s to pipeline", gst_element_get_name (element));
    }
  }
}

Element::Element(const std::string& type_name, std::string name)
: Object(), objects_(std::make_shared<std::unordered_map<std::string, std::unique_ptr<CustomObject>>>()), pipeline_(nullptr) {
  object_ = GST_OBJECT(gst_element_factory_make(
    type_name.c_str(), name.empty() ? NULL:name.c_str()
  ));
  if (!object_) {
    throw std::runtime_error("Element creation failed");
  }
}

Element::Element(const Object& other) : Object(other) {
  if (!object_) {
    throw std::runtime_error("Up cast to Element failed");
  }
  if (!GST_IS_ELEMENT(object_)) {
    throw std::runtime_error("Up cast to Element failed");
  }
}

Element::Element(Object&& other) : Object(other) {
  if (!object_ || !GST_IS_ELEMENT(object_)) {
    throw std::runtime_error("Up cast to Element failed");
  }
}

Element::Element(GstObject* object):
Object(), objects_(std::make_shared<std::unordered_map<std::string, std::unique_ptr<CustomObject>>>()) {
  object_ = object;
}

Element::~Element()
{
}

Element & Element::link(Element &target, std::pair<std::string, std::string> hint)
{
  // Source Element
  GstElement * src = GST_ELEMENT(this->object_);
  // Destination Element
  GstElement * dst = GST_ELEMENT(target.object_);

  cout<<"LINKING: "<<"Source: "<<GST_ELEMENT_NAME(src)<<" Target: "<<GST_ELEMENT_NAME(dst)<<endl;
  // Source -> Destination

  // Source Element Source Pad Template List
  std::vector<GstPadTemplate*> src_pad_template_list = get_element_pad_template_list (src, GST_PAD_SRC);
  // Destination Element Sink Pad Template List
  std::vector<GstPadTemplate*> sink_pad_template_list = get_element_pad_template_list (dst, GST_PAD_SINK);

  // To save index of compatible caps b/w source pad and sink pad
  int i,j;

  int src_pad_template_list_size = src_pad_template_list.size();
  int sink_pad_template_list_size = sink_pad_template_list.size();

  // set to true if compatible caps are found
  bool caps_intersect = false;

  // Logic to find comaptible caps
  for (i=0; i<src_pad_template_list_size; i++) {
    // Makes sure only user specified pad template is used for checking caps compatibility
    if (hint.first != "" && !pad_name_matches_template(hint.first, src_pad_template_list[i]->name_template))
      continue;

    for (j=0; j<sink_pad_template_list_size; j++) {
      if (hint.second != "" && !pad_name_matches_template(hint.second, sink_pad_template_list[j]->name_template))
        continue;

      if (gst_caps_can_intersect (src_pad_template_list[i]->caps, sink_pad_template_list[j]->caps)) {
        caps_intersect = true;
        break;
      }
    }
    if (caps_intersect) break;
  }

  if (!caps_intersect) {
    g_printerr("Source Element and Destination Element Caps Not Compatible. Exiting.\n");
    throw std::runtime_error("Elements linking  failed");
  }

  // Get Compatible pad template for Source and Sink Pad
  GstPadTemplate *src_pad_template = src_pad_template_list[i];
  GstPadTemplate *sink_pad_template = sink_pad_template_list[j];

  // Handling Different Pad Linking Cases

  // static -> static
  if (src_pad_template->presence==GST_PAD_ALWAYS && sink_pad_template->presence==GST_PAD_ALWAYS) {
    if (gst_element_link (src, dst) != TRUE) {
      throw std::runtime_error("Element linking failed");
    }
  }

  // static -> request
  if (src_pad_template->presence==GST_PAD_ALWAYS && sink_pad_template->presence==GST_PAD_REQUEST) {
    GstPad *srcpad, *sinkpad;

    srcpad = gst_element_get_static_pad (src, src_pad_template->name_template);
    if (!srcpad) {
      g_printerr("Unable to  get Source Element Static Pad. Exiting.\n");
      throw std::runtime_error("Element creation failed");
    }

    sinkpad = get_element_request_pad (dst, sink_pad_template, hint.second);
    if (!sinkpad) {
      g_printerr("Failed to create on-request sinkpad. Exiting.\n");
      throw std::runtime_error("Element creation failed");
    }

    if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) {
      g_printerr("Failed to link static srcpad to on-request sinkpad. Exiting.\n");
      throw std::runtime_error("Element linking failed");
    }

    gst_object_unref (sinkpad);
    gst_object_unref (srcpad);
  }

  // dynamic -> static
  if (src_pad_template->presence==GST_PAD_SOMETIMES && sink_pad_template->presence==GST_PAD_ALWAYS) {
    // Element class member variable used for Element with dynamic pads
    GstPad* sinkpad = gst_element_get_static_pad (dst, sink_pad_template->name_template);
    if (!sinkpad) {
      g_printerr("Unable to get Destination Element Static Pad. Exiting.\n");
      throw std::runtime_error("Element creation failed");
    }
    g_signal_connect(G_OBJECT(this->object_), "pad-added", G_CALLBACK(cb_newpad), sinkpad);
    gst_object_unref (sinkpad);
  }

  // dynamic -> request
  if (src_pad_template->presence==GST_PAD_SOMETIMES && sink_pad_template->presence==GST_PAD_REQUEST) {
    // Element class member variable used for Element with dynamic pads
    GstPad* sinkpad =  get_element_request_pad (dst, sink_pad_template, hint.second);
    if (!sinkpad) {
      g_printerr("Failed to create on-request sinkpad. Exiting.\n");
      throw std::runtime_error("Element creation failed");
    }
    g_signal_connect(G_OBJECT(this->object_), "pad-added", G_CALLBACK(cb_newpad), sinkpad);
    gst_object_unref (sinkpad);
  }

  // request -> static
  if (src_pad_template->presence==GST_PAD_REQUEST && sink_pad_template->presence==GST_PAD_ALWAYS) {
    GstPad *srcpad, *sinkpad;

    srcpad =  get_element_request_pad (src, src_pad_template, hint.first);
    if (!srcpad) {
      g_printerr("Failed to create on-request srcpad. Exiting.\n");
      throw std::runtime_error("Element creation failed");
    }

    sinkpad = gst_element_get_static_pad (dst, sink_pad_template->name_template);
    if (!sinkpad) {
      g_printerr("Unable to get Destination Element Static Pad. Exiting.\n");
      throw std::runtime_error("Element creation failed");
    }

    if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) {
      g_printerr("Failed to link on-request srcpad to static sinkpad. Exiting.\n");
      throw std::runtime_error("Element linking failed");
    }

    gst_object_unref (sinkpad);
    gst_object_unref (srcpad);
  }

  // request -> request
  if (src_pad_template->presence==GST_PAD_REQUEST && sink_pad_template->presence==GST_PAD_REQUEST) {
    GstPad *srcpad, *sinkpad;

    srcpad =  get_element_request_pad (src, src_pad_template, hint.first);
    if (!srcpad) {
      g_printerr("Failed to create on-request srcpad. Exiting.\n");
      throw std::runtime_error("Element creation failed");
    }

    sinkpad =  get_element_request_pad (dst, sink_pad_template, hint.second);
    if (!sinkpad) {
      g_printerr("Failed to create on-request sinkpad. Exiting.\n");
      throw std::runtime_error("Element creation failed");
    }

    if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) {
      g_printerr("Failed to link on-request srcpad to on-request sinkpad. Exiting.\n");
      throw std::runtime_error("Element linking failed");
    }

    gst_object_unref (sinkpad);
    gst_object_unref (srcpad);
  }

  return target;
}

Element & Element::link(Element& other) {
  return link(other, {"", ""});
}

Element& Element::addProbe(const std::string& plugin_name,
                           const std::string& name,
                           const std::string tip) {
  BufferProbe* probe = dynamic_cast<BufferProbe*>(
    CommonFactory::getInstance().createObject(plugin_name, name).release()
  );
  if (!probe) {
    g_printerr("Failed to create probe from %s\n", plugin_name.c_str());
    throw std::runtime_error("Probe creation failed");
  }
  return this->addProbe(probe, tip);
}

Element& Element::addProbe(BufferProbe* probe, const std::string tip) {
  GstElement * element = GST_ELEMENT(this->object_);
  GstPad* pad = NULL;
  GValue item = G_VALUE_INIT;
  if (tip.empty()) {
    GstIterator *it = gst_element_iterate_src_pads(element);
    bool done = false;
    while (!done) {
      switch (gst_iterator_next(it, (GValue *)&item)) {
        case GST_ITERATOR_OK:
          pad = GST_PAD(g_value_get_object(&item));
          break;
        case GST_ITERATOR_RESYNC:
          gst_iterator_resync(it);
          break;
        case GST_ITERATOR_ERROR:
          g_printerr("Iterator error\n");
          done = TRUE;
          break;
        case GST_ITERATOR_DONE:
          done = TRUE;
          break;
      }
    }
    gst_iterator_free(it);
  } else {
    pad = gst_element_get_static_pad(element, tip.c_str());
    if (pad == NULL) {
      // error
      g_printerr("Pad %s not found from %s\n", tip.c_str(), this->getName().c_str());
      throw std::runtime_error("Probe failure");
    } else if (gst_pad_get_direction(pad) != GST_PAD_SRC) {
      // error
      g_printerr("Pad %s is not an output\n", tip.c_str());
      throw std::runtime_error("Probe failure");
    }
  }

  if (pad) {
    const gchar* pad_name = gst_pad_get_name(pad);
    gst_pad_add_probe (pad, GST_PAD_PROBE_TYPE_BUFFER,
          genric_probe_callback, probe, NULL);
    probe->attach(this, Pad(Object().seize(GST_OBJECT(pad))));
    add_(probe);
    g_free((gpointer)pad_name);
  } else {
    g_printerr("Error: unable to add probe %s\n", probe->getName().c_str());
    throw std::runtime_error("Probe failure");
  }

  return *this;
}

Element& Element::connectSignal(const std::string& signal_name, SignalHandler* handler) {
  if (!Object::connectSignal(signal_name, *handler)) {
    throw std::runtime_error("Signal connect failure");
  }
  if (!find_<SignalHandler>(handler->getName())){
    add_(handler);
  }

  return *this;
}

Element& Element::connectSignal(const std::string& plugin_name,
                                 const std::string& handler_name,
                                 const std::string& signal_names) {
  SignalHandler* handler = dynamic_cast<SignalHandler*>(
    CommonFactory::getInstance().createObject(plugin_name, handler_name).release()
  );
  if (!handler) {
    g_printerr("Failed to create signal handler from %s\n", plugin_name.c_str());
    throw std::runtime_error("Signal creation failed");
  }
  size_t begin = 0;
  size_t end = 0;
  while (end != std::string::npos) {
    end = signal_names.find("/", begin);
    std::string signal = signal_names.substr(begin, end);
    this->connectSignal(signal, handler);
    begin = end+1;
  }

  return *this;
}

Element& Element::add_(CustomObject* object) {
  auto& name = object->getName();
  if (objects_->find(name) != objects_->end()) {
    g_printerr("Custom object %s already exists! Element: %s\n", name.c_str(), this->getName().c_str());
    throw std::runtime_error("Custom Object duplicated");
  }
  objects_->insert({object->getName(), std::unique_ptr<CustomObject>(object)});
  return *this;
}

bool  Element::setState(Element::State state) {
  bool ret = true;
  auto it = gstStateMap.find(state);

  if (it != gstStateMap.end()) {
    GstState gst_state = it->second;
    if (gst_element_set_state (GST_ELEMENT(object_),
            gst_state) == GST_STATE_CHANGE_FAILURE) {
      GST_ERROR_OBJECT (object_, "Can't set object to %d", state);
      ret = false;
    }
  } else {
    GST_ERROR_OBJECT (object_, "Invalid state %d", state);
    ret = false;
  }
  return ret;
}
