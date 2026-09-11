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

#include <iostream>
#include <algorithm>
#include <gst/gst.h>
#include <gst/rtsp-server/rtsp-server.h>
#include <unistd.h>

#include "pipeline.hpp"
#include "ds_yaml_parser.hpp"
#include "common_factory.hpp"
#include "signal_emitter.hpp"

#include "gst-nvdscustommessage.h"
#include "gst-nvdssr.h"

using namespace std;
using namespace deepstream;

GST_DEBUG_CATEGORY (NVDS_APP);

namespace deepstream {
void init_gst();
bool _gst_initialized = false;
}


// Class used to pass linking data
typedef struct LinkInfo {
  std::pair<std::string, std::string> link;
  std::pair<std::string, std::string> tip;
} LinkInfo;

struct BusCallData {
  void* loop;
  Pipeline* pipeline;
  std::function<void(Pipeline&, const Pipeline::Message&)> listener;
};

struct SrDoneData {
  std::function<void(const RecordingInfo &)> callback;
};

typedef enum
{
  NV_DS_ENCODER_H264 = 1,
  NV_DS_ENCODER_H265,
  NV_DS_ENCODER_MPEG4
} NvDsEncoderType;

static const char* _default_module_path = "/opt/nvidia/deepstream/deepstream/service-maker/modules";
static GstRTSPServer *server[MAX_SINK_BINS];
static guint server_count = 0;
static GMutex server_cnt_lock;

RecordingInfo::RecordingInfo(void* data) : data_(data) {}
RecordingInfo::~RecordingInfo() {}

unsigned int RecordingInfo::getSessionId() const {
  NvDsSRRecordingInfo *recording_info = (NvDsSRRecordingInfo *) data_;
  return recording_info->sessionId;
}

std::string RecordingInfo::getFileName() const {
  NvDsSRRecordingInfo *recording_info = (NvDsSRRecordingInfo *) data_;
  return recording_info->filename;
}

std::string RecordingInfo::getFileDirectory() const {
  NvDsSRRecordingInfo *recording_info = (NvDsSRRecordingInfo *) data_;
  return recording_info->dirpath;
}

unsigned int RecordingInfo::getDuration() const {
  NvDsSRRecordingInfo *recording_info = (NvDsSRRecordingInfo *) data_;
  return recording_info->duration;
}

std::string RecordingInfo::getContainerType() const {
  NvDsSRRecordingInfo *recording_info = (NvDsSRRecordingInfo *) data_;
  return recording_info->containerType == NVDSSR_CONTAINER_MP4 ? "MP4" : "MKV";
}

unsigned int RecordingInfo::getWidth() const {
  NvDsSRRecordingInfo *recording_info = (NvDsSRRecordingInfo *) data_;
  return recording_info->width;
}

unsigned int RecordingInfo::getHeight() const {
  NvDsSRRecordingInfo *recording_info = (NvDsSRRecordingInfo *) data_;
  return recording_info->height;
}

bool RecordingInfo::containsVideo() const {
  NvDsSRRecordingInfo *recording_info = (NvDsSRRecordingInfo *) data_;
  return recording_info->containsVideo;
}

bool RecordingInfo::containsAudio() const {
  NvDsSRRecordingInfo *recording_info = (NvDsSRRecordingInfo *) data_;
  return recording_info->containsAudio;
}

unsigned int RecordingInfo::getChannels() const {
  NvDsSRRecordingInfo *recording_info = (NvDsSRRecordingInfo *) data_;
  return recording_info->channels;
}

unsigned int RecordingInfo::getSamplingRate() const {
  NvDsSRRecordingInfo *recording_info = (NvDsSRRecordingInfo *) data_;
  return recording_info->samplingRate;
}

static gboolean
start_rtsp_streaming (guint rtsp_port_num, guint updsink_port_num,
    NvDsEncoderType enctype, guint64 udp_buffer_size);

static GstElement* get_element_by_factory_name(GstBin *bin, const gchar *factory_name) {
    GstElement *element = NULL;
    GstIterator *iterator = gst_bin_iterate_all_by_element_factory_name(GST_BIN(bin), factory_name);
    GValue item = G_VALUE_INIT;

    // Get first matching element
    if (gst_iterator_next(iterator, &item) == GST_ITERATOR_OK) {
        element = GST_ELEMENT(g_value_get_object(&item));
        gst_object_ref(element); // Increase ref count since we'll be using it
        g_value_reset(&item);
    }

    gst_iterator_free(iterator);
    return element;
}

static gboolean load_file(GstElement *dynamicsrcbin, gchar *current_file) {
    //Add new filesrc and queue element
    GstElement *filesrc = gst_element_factory_make("filesrc", "source");
    GstElement *queue = gst_element_factory_make("queue", "queue");
    GstElement *parsebin = gst_element_factory_make("parsebin", "parsebin");

    if (!filesrc || !queue || !parsebin) {
        g_print("One of the elements not successfully created\n \n");
        return FALSE;
    }

    // Update filesrc location
    g_object_set(filesrc, "location", current_file, NULL);
    g_print("Playing next file: %s\n", current_file);

    // Add the elements in dynamicsrcbin
    gst_bin_add_many(GST_BIN(dynamicsrcbin), filesrc, queue, parsebin, NULL);

    // Link elements
    gst_element_link_many(filesrc, queue, parsebin, NULL);

    // Change their state to PLAYING
    gst_element_set_state(parsebin, GST_STATE_PLAYING);
    gst_element_set_state(queue, GST_STATE_PLAYING);
    gst_element_set_state(filesrc, GST_STATE_PLAYING);

    return TRUE;
}

static gboolean try_load_file(gpointer data) {
  GstElement *dynamicsrcbin = GST_ELEMENT(data);
  int current_id = -1;
  gchar *current_file = NULL;
  g_object_get(G_OBJECT(dynamicsrcbin), "current-id", &current_id, NULL);
  g_object_get(G_OBJECT(dynamicsrcbin), "current-file", &current_file, NULL);
  if (!current_file || current_id < 0) {
    g_print("Waiting for file to come...\n");
    return TRUE;
  }
  if (!load_file(dynamicsrcbin, current_file)) {
    g_print("Failed to load file: %s\n", current_file);
  }
  g_free(current_file);
  // File loaded, no need to try again
  return FALSE;
}

static gboolean handle_dynamic_source_message(GstElement *dynamicsrcbin) {
  int current_id = -1;
  gchar *current_file = NULL;
  GstElement *filesrc = get_element_by_factory_name(GST_BIN(dynamicsrcbin), "filesrc");
  GstElement *queue = get_element_by_factory_name(GST_BIN(dynamicsrcbin), "queue");
  GstElement *parsebin = get_element_by_factory_name(GST_BIN(dynamicsrcbin), "parsebin");

  if (!filesrc || !queue || !parsebin) {
      g_print("One of the elements not found\n \n");
  }
  else {
      gst_element_set_state(parsebin, GST_STATE_NULL);
      gst_element_set_state(queue, GST_STATE_NULL);
      gst_element_set_state(filesrc, GST_STATE_NULL);

      gst_bin_remove(GST_BIN(dynamicsrcbin), filesrc);
      gst_bin_remove(GST_BIN(dynamicsrcbin), queue);
      gst_bin_remove(GST_BIN(dynamicsrcbin), parsebin);
      g_object_get(G_OBJECT(dynamicsrcbin), "current-id", &current_id, NULL);
      g_object_get(G_OBJECT(dynamicsrcbin), "current-file", &current_file, NULL);
      g_print("Current ID: %d, Current File: %s\n", current_id, current_file);
      if (current_file && current_id >= 0) {
          gboolean ret = load_file(dynamicsrcbin, current_file);
          g_free(current_file);
          return ret;
      }

      g_timeout_add(1000, try_load_file, dynamicsrcbin);
  }
  return TRUE;
}

void deepstream::init_gst() {
    g_print("Initializing GStreamer Backend...!\n");
    /* update environment variable */
    std::string module_path = _default_module_path;
    char* plugin_path = getenv("GST_PLUGIN_PATH");
    if (plugin_path) {
      module_path += ":";
      module_path += std::string(plugin_path);
    }
    char* custom_path = getenv("NVDS_MODULE_PATH");
    if (custom_path) {
      module_path += ":";
      module_path += std::string(custom_path);
    }
    setenv("GST_PLUGIN_PATH", module_path.c_str(), 1);

    /* Initialize GStreamer */
    gst_init(NULL, NULL);
}

Pipeline::EOSMessage::EOSMessage(void* message)
: Message(GST_MESSAGE_TYPE(message)) {
  if (this->type_ != GST_MESSAGE_EOS) {
    throw std::runtime_error("Invalid type for EOSMessage");
  }
}

Pipeline::StateTransitionMessage::StateTransitionMessage(void* message)
: Message(GST_MESSAGE_TYPE(message)) {
  GstMessage* gst_msg = (GstMessage*) message;
  char* object_name = gst_object_get_name(GST_OBJECT(GST_MESSAGE_SRC(gst_msg)));
  if (this->type_ != GST_MESSAGE_STATE_CHANGED) {
    throw std::runtime_error("Invalid type for StateTransitionMessage");
  }
  GstState oldstate = GST_STATE_NULL, newstate = GST_STATE_NULL, pending = GST_STATE_NULL;
  gst_message_parse_state_changed (gst_msg, &oldstate, &newstate,
            &pending);
  old_state_ = static_cast<State>(oldstate);
  new_state_ = static_cast<State>(newstate);;
  name_ = object_name;
}

Pipeline::DynamicSourceMessage::DynamicSourceMessage(void* message)
: Message(GST_MESSAGE_TYPE(message)){
  GstMessage* gst_msg = (GstMessage*) message;
  NvDsSensorInfo sensorInfo = {0};
  if (gst_nvmessage_is_stream_add(gst_msg)) {
    gst_nvmessage_parse_stream_add(gst_msg, &sensorInfo);
    g_print("new stream added [%d:%s:%s]\n\n\n\n", sensorInfo.source_id, sensorInfo.sensor_id, sensorInfo.sensor_name);
    source_added_ = true;
  } else if (gst_nvmessage_is_stream_remove(gst_msg)) {
    gst_nvmessage_parse_stream_remove(gst_msg, &sensorInfo);
    g_print("new stream removed [%d:%s]\n\n\n\n", sensorInfo.source_id, sensorInfo.sensor_id);
    source_added_ = false;
  }
  source_id_ = sensorInfo.source_id;
  sensor_id_ = sensorInfo.sensor_id?string(sensorInfo.sensor_id):"N/A";
  sensor_name_ = sensorInfo.sensor_name?string(sensorInfo.sensor_name):"N/A";
  uri_ = sensorInfo.uri ? string(sensorInfo.uri) : "N/A";
}

Pipeline::DynamicSourceMessage::DynamicSourceMessage(uint32_t source_id, bool source_added)
: Message(GST_MESSAGE_APPLICATION),
  source_added_(source_added),
  source_id_(source_id),
  sensor_id_("N/A"),
  sensor_name_("N/A"),
  uri_("N/A") {
}

Pipeline::Pipeline(const char* name, const std::string& config_file): Pipeline(name)
{
  if (config_file.empty()) {
    // fall back to standard constructor
    return;
  }

  std::string err = "YAML Parsing Error: ";
  err = err + __PRETTY_FUNCTION__ + " ";
  std::vector<BufferProbe*> probes;
  std::vector<SignalHandler*> signal_handlers;
  std::vector<LinkInfo> links;

  try
  {
    YAML_DSConfig *dsConfig = new YAML_DSConfig(config_file);
    dsConfig->Parse();
    dsConfig->PrintElements();

    // DeepStream Elements Vector in YAML
    for (auto it = dsConfig->getElementsVector().begin(); it < dsConfig->getElementsVector().end(); ++it)
    {
      g_print("Element: %s, Name: %s\n", it->m_elementName.c_str(), it->m_linkName.c_str());

      string delimiter = ".";
      size_t pos = 0;
      if ((pos = it->m_elementName.find(delimiter)) == string::npos) {
        // this is a standard GST element
        Element element = Element(it->m_elementName, it->m_linkName);
        // DeepStream Elements Property Vector in YAML
        if (!it->properties_.IsNull()) {
          element.set(it->properties_);
        }
        this->add(element);
      } else {
        string plugin_name = it->m_elementName.substr(0, pos);
        string factory_name = it->m_elementName.substr(pos+1);
        /* we creates the Elements and Buffer Probes separately for now, however
          we want them all created through common factory */
        Object* object = CommonFactory::getInstance().createObject(factory_name, it->m_linkName).release();
        if (!object) {
          std::string error = "Failed to create object:";
          error += it->m_linkName;
          throw std::runtime_error(error);
        }
        if (!it->properties_.IsNull()) {
          object->set(it->properties_);
        }
        BufferProbe* probe = dynamic_cast<BufferProbe*>(object);
        SignalHandler* handler = dynamic_cast<SignalHandler*>(object);
        SignalEmitter* emitter = dynamic_cast<SignalEmitter*>(object);
        if (probe) {
          probes.push_back(probe);
        }
        if (handler) {
          signal_handlers.push_back(handler);
        }
        if (emitter) {
          action_owners_.insert(
            {emitter->getName(), std::unique_ptr<SignalEmitter>(emitter)}
          );
        }
      }
      // Create Vector of src -> dst Link Pair
      for (size_t i=0; i<it->m_linkToElement.size(); i++){
        // Saving Link Data in linkVector source, target along with user provided pad templates
        links.push_back(
          LinkInfo({
            {it->m_linkName, it->m_linkToElement[i].target},
            {it->m_linkToElement[i].source_element_src_pad_template, it->m_linkToElement[i].target_element_sink_pad_template}
          }));
      }
    }

    // Link created elements
    for (auto& l : links)
    {
      Element *src = NULL;
      Element *dest = NULL;

      if (l.link.second.size())
      {
        auto itr = elements_.find(l.link.first);
        if (itr != elements_.end()) {
          src = &itr->second;
        }
        itr = elements_.find(l.link.second);
        if (itr != elements_.end()) {
          dest = &itr->second;
        }
        if (src && dest) {
          src->link(*dest, {l.tip.first, l.tip.second});
        } else if (src) {
            auto probe_itr = std::find_if(
              probes.begin(),
              probes.end(),
              [&](BufferProbe* p) { return p->getName() == l.link.second; }
            );
            if (probe_itr != probes.end()) {
              g_print("Adding probe: %s\n", (*probe_itr)->getName().c_str());
              src->addProbe(*probe_itr);
            }
            auto handler_itr = std::find_if(
              signal_handlers.begin(),
              signal_handlers.end(),
              [&](SignalHandler* p) { return p->getName() == l.link.second; }
            );
            if (handler_itr != signal_handlers.end()) {
              g_print("Connecting signal: %s\n", (*handler_itr)->getName().c_str());
              src->connectSignal(l.tip.first, *handler_itr);
            }
        } else if (dest) {
          auto action_itr = action_owners_.find(l.link.first);
          if (action_itr != action_owners_.end()) {
            action_itr->second->attach(l.tip.first, *dest);
          }
        } else {
          g_printerr("Error, either source or target not found");
        }
      }
    }
  }
  catch (exception &e)
  {
    std::throw_with_nested( std::runtime_error(err + e.what()));
  }
}

Pipeline::Pipeline(const char *name) : Object() {
  if (!_gst_initialized) {
    init_gst();
    _gst_initialized = true;
  }

  object_ = GST_OBJECT(gst_pipeline_new(name));
  // Create GMainLoop
  loop_ = g_main_loop_new(NULL, FALSE);
}

Pipeline::~Pipeline() {}

Pipeline & Pipeline::add(Element element)
{
  cout << "Add Element ... " << element.getName() << endl;
  element.pipeline_ = this;
  elements_.insert({element.getName(), element});
  GstObject* object = GST_OBJECT(element.give());
  gst_bin_add_many (GST_BIN(object_), GST_ELEMENT(object), NULL);
  return *this;
}

Element* Pipeline::find(const std::string& name) {
  auto itr = elements_.find(name);
  if (itr == elements_.end()) {
    return nullptr;
  }
  return &(itr->second);
}

Pipeline& Pipeline::link(
  std::pair<std::string, std::string> pair,
  std::pair<std::string, std::string> tips
) {
  auto source = (*this)[pair.first];
  auto target = (*this)[pair.second];
  source.link(target, std::move(tips));
  return *this;
}

Pipeline& Pipeline::attach(const std::string& element_name, CustomObject* object, const std::string tip) {
  auto& element = (*this)[element_name];
  if (auto probe = dynamic_cast<BufferProbe*>(object)) {
    element.addProbe(probe, tip);
  } else if (auto handler = dynamic_cast<SignalHandler*>(object)) {
    size_t begin = 0;
    size_t end = 0;
    while (end != std::string::npos) {
      end = tip.find("/", begin);
      std::string signal = tip.substr(begin, end);
      element.connectSignal(signal, handler);
      begin = end+1;
    }
  } else {
    g_printerr("Attach: Wront custom object type\n");
    throw std::runtime_error("Object type invalid");
  }
  return *this;
}

Pipeline& Pipeline::attach(
  const std::string& element_name,
  const std::string& plugin_name,
  const std::string& object_name,
  const std::string tip ){
  auto obj = CommonFactory::getInstance().createObject(plugin_name, object_name).release();
  if (!obj) {
    g_printerr("Failed to create object from plugin %s\n", plugin_name.c_str());
    throw std::runtime_error("Factory failure");
  }
  return attach(element_name, obj, tip);
}

static gboolean
bus_call (GstBus * bus, GstMessage * msg, gpointer data)
{
  struct BusCallData *busData = (BusCallData *)data;
  GMainLoop *loop = (GMainLoop *) busData->loop;
  gchar *debuginfo = NULL;
  GError *error = NULL;

  switch (GST_MESSAGE_TYPE(msg)) {
    case GST_MESSAGE_EOS:
      g_print ("End of stream\n");
      if (busData->listener) {
        Pipeline::EOSMessage eos_msg(msg);
        busData->listener(*busData->pipeline, eos_msg);
      }
      g_main_loop_quit (loop);
      break;
    case GST_MESSAGE_STATE_CHANGED:
      if (busData->listener) {
        Pipeline::StateTransitionMessage state_transition_msg(msg);
        busData->listener(*busData->pipeline, state_transition_msg);
      }
      break;
    case GST_MESSAGE_ELEMENT:
      if (gst_nvmessage_is_stream_add(msg) || gst_nvmessage_is_stream_remove(msg)) {
        Pipeline::DynamicSourceMessage dynamic_source_msg(msg);
        if (busData->listener) {
          busData->listener(*busData->pipeline, dynamic_source_msg);
        }
      }
      break;
    case GST_MESSAGE_INFO:
      error = NULL;
      debuginfo = NULL;
      gst_message_parse_info(msg, &error, &debuginfo);
      g_printerr ("INFO from %s: %s\n",
          GST_OBJECT_NAME(msg->src), error->message);
      if (debuginfo) {
        g_printerr("Debug info: %s\n", debuginfo);
      }
      g_error_free(error);
      g_free(debuginfo);
      break;
    case GST_MESSAGE_WARNING:
      error = NULL;
      debuginfo = NULL;
      gst_message_parse_warning(msg, &error, &debuginfo);
      g_printerr("WARNING from %s: %s\n",
          GST_OBJECT_NAME(msg->src), error->message);
      if (debuginfo) {
        g_printerr("Debug info: %s\n", debuginfo);
      }
      g_error_free(error);
      g_free(debuginfo);
      break;
    case GST_MESSAGE_ERROR:
      error = NULL;
      debuginfo = NULL;
      gst_message_parse_error(msg, &error, &debuginfo);

      {
        const gchar *attempts_error =
          "Reconnection attempts exceeded for all sources or EOS received.";
        const gchar *window_closed = "Output window was closed";
        if (
          strstr(error->message, attempts_error) ||
          strstr(error->message, window_closed)
        ) {
          g_main_loop_quit (loop);
        }
      }

      g_printerr ("ERROR from %s: %s\n",
          GST_OBJECT_NAME (msg->src), error->message);
      if (debuginfo) {
        g_printerr ("Debug info: %s\n", debuginfo);
      }
      g_error_free (error);
      g_free (debuginfo);
      break;
    case GST_MESSAGE_APPLICATION: {
        // To Handle EOS message
        const GstStructure *str = gst_message_get_structure(msg);
        if (str && gst_structure_has_name(str, "dynamic-src-bin-file-change")) {
            gint source_id = -1;
            if (gst_structure_get_int(str, "source-id", &source_id) && source_id >= 0 && busData->listener) {
              Pipeline::DynamicSourceMessage dynamic_source_msg(source_id, false);
              busData->listener(*busData->pipeline, dynamic_source_msg);
            }
            GstElement *dynamicsrcbin = get_element_by_factory_name(GST_BIN((*busData->pipeline).getGObject()), "nvdsdynamicsrcbin");
            if (!dynamicsrcbin) {
              g_printerr("Failed to get dynamicsrcbin\n");
              return FALSE;
            }
            handle_dynamic_source_message(dynamicsrcbin);
            gst_object_unref(dynamicsrcbin);
        }
        break;
    }
    default:
      break;
  }
  return TRUE;
}

void Pipeline::handleKey(int key) {
  if (keyboard_listener_) {
    keyboard_listener_(*this, key);
  }
}

static gboolean
kbhit (void)
{
  struct timeval tv;
  fd_set rdfs;

  tv.tv_sec = 0;
  tv.tv_usec = 0;

  FD_ZERO (&rdfs);
  FD_SET (STDIN_FILENO, &rdfs);

  select (STDIN_FILENO + 1, &rdfs, NULL, NULL, &tv);
  return FD_ISSET (STDIN_FILENO, &rdfs);
}

/**
 * Loop function to check keyboard inputs and status of each pipeline.
 */
static gboolean
event_thread_func (gpointer arg) {
  Pipeline* pipeline = (Pipeline*) arg;

  if (!pipeline->isRunning()) {
    return FALSE;
  }

  // Check for keyboard input
  if (!kbhit ()) {
    //continue;
    return TRUE;
  }

  int c = fgetc (stdin);
  g_print ("\n");
  if (feof(stdin)) {
      g_print("End of file reached on stdin.\n");
      return FALSE;
  } else if (ferror(stdin)) {
      g_printerr("An error occurred while reading from stdin.\n");
  } else {
    pipeline->handleKey(c);
  }

  return TRUE;
}

int Pipeline::prepare()
{
  bus_data_ = new BusCallData{loop_, this};
  GstBus *bus = NULL;
  GstStateChangeReturn ret;

  bus = gst_pipeline_get_bus(GST_PIPELINE(object_));
  bus_watch_id_ = gst_bus_add_watch(bus, bus_call, bus_data_);
  gst_object_unref(bus);

  GST_DEBUG_BIN_TO_DOT_FILE_WITH_TS(GST_BIN(object_),
                                    GST_DEBUG_GRAPH_SHOW_ALL, "ds_cpp_app_running");

  /* Start playing */
  ret = gst_element_set_state(GST_ELEMENT(object_), GST_STATE_PAUSED);
  if (ret == GST_STATE_CHANGE_FAILURE)
  {
    g_printerr("Unable to set the pipeline to the paused state for initialization.\n");
    return -1;
  }

  return 1;
}

int Pipeline::prepare(std::function<void(Pipeline &, const Message &)> listener)
{
  bus_data_ = new BusCallData{loop_, this, std::move(listener)};
  GstBus *bus = NULL;
  GstStateChangeReturn ret;

  bus = gst_pipeline_get_bus(GST_PIPELINE(object_));
  bus_watch_id_ = gst_bus_add_watch(bus, bus_call, bus_data_);
  gst_object_unref(bus);

  GST_DEBUG_BIN_TO_DOT_FILE_WITH_TS(GST_BIN(object_),
                                    GST_DEBUG_GRAPH_SHOW_ALL, "ds_cpp_app_running");

  /* Start playing */
  ret = gst_element_set_state(GST_ELEMENT(object_), GST_STATE_PAUSED);
  if (ret == GST_STATE_CHANGE_FAILURE)
  {
    g_printerr("Unable to set the pipeline to the paused state for initialization.\n");
    return -1;
  }

  return 1;
}

int Pipeline::run() {
  GstBus *bus = NULL;
  GstStateChangeReturn ret;

  bus = gst_pipeline_get_bus (GST_PIPELINE(object_));
  bus_watch_id_ = gst_bus_add_watch (bus, bus_call, bus_data_);
  gst_object_unref (bus);

  GST_DEBUG_BIN_TO_DOT_FILE_WITH_TS(GST_BIN(object_),
                                    GST_DEBUG_GRAPH_SHOW_ALL, "ds_cpp_app_running");

   if (start_pts_ > 0) {
    ret = gst_element_set_state(GST_ELEMENT(object_), GST_STATE_PAUSED);
    if (ret == GST_STATE_CHANGE_FAILURE) {
      g_printerr ("Unable to set the pipeline to the playing state.\n");
      return -1;
    }
    GstState state, pending;
    gst_element_get_state(GST_ELEMENT(object_), &state, &pending, GST_CLOCK_TIME_NONE);
    if (!gst_element_seek_simple(GST_ELEMENT(object_), GST_FORMAT_TIME,
                                 (GstSeekFlags)(GST_SEEK_FLAG_FLUSH | GST_SEEK_FLAG_KEY_UNIT | GST_SEEK_FLAG_SNAP_BEFORE),
                                 start_pts_)
    ) {
      g_printerr("Pipeline::seek failed\n");
    }
  }

  /* Start playing */
  ret = gst_element_set_state (GST_ELEMENT(object_), GST_STATE_PLAYING);
  if (ret == GST_STATE_CHANGE_FAILURE) {
    g_printerr ("Unable to set the pipeline to the playing state.\n");
    return -1;
  }
  guint gid = g_timeout_add (40, event_thread_func, this);
  g_print("Event Thread Enabled...\n");
  g_print("Main Loop Running...\n");
  g_main_loop_run((GMainLoop*) loop_);
  g_print("Main Loop Exited...\n");
  g_source_remove(gid);

  return 1;
}

int Pipeline::run_after_prepare()
{
  GstStateChangeReturn ret;

  if (start_pts_ > 0)
  {
    ret = gst_element_set_state(GST_ELEMENT(object_), GST_STATE_PAUSED);
    if (ret == GST_STATE_CHANGE_FAILURE)
    {
      g_printerr("Unable to set the pipeline to the playing state.\n");
      return -1;
    }
    GstState state, pending;
    gst_element_get_state(GST_ELEMENT(object_), &state, &pending, GST_CLOCK_TIME_NONE);
    if (!gst_element_seek_simple(GST_ELEMENT(object_), GST_FORMAT_TIME,
                                 (GstSeekFlags)(GST_SEEK_FLAG_FLUSH | GST_SEEK_FLAG_KEY_UNIT | GST_SEEK_FLAG_SNAP_BEFORE),
                                 start_pts_))
    {
      g_printerr("Pipeline::seek failed\n");
    }
  }

  /* Start playing */
  ret = gst_element_set_state(GST_ELEMENT(object_), GST_STATE_PLAYING);
  if (ret == GST_STATE_CHANGE_FAILURE)
  {
    g_printerr("Unable to set the pipeline to the playing state.\n");
    return -1;
  }
  guint gid = g_timeout_add(40, event_thread_func, this);
  g_print("Event Thread Enabled...\n");
  g_print("Main Loop Running...\n");
  g_main_loop_run((GMainLoop *)loop_);
  g_print("Main Loop Exited...\n");
  g_source_remove(gid);

  return 1;
}

Pipeline& Pipeline::wait() {
  if (thread_.joinable()) {
    thread_.join();
  }

  GstStateChangeReturn ret = gst_element_set_state(GST_ELEMENT(object_), GST_STATE_NULL);
  if (ret == GST_STATE_CHANGE_FAILURE) {
      g_printerr("Failed to change state\n");
  }
  loop_ = NULL;
  delete (BusCallData*) bus_data_;
  bus_data_ = nullptr;
  g_print("Pipeline %s stopped\n", this->getName().c_str());
  return *this;
}

Pipeline& Pipeline::start() {
  bus_data_ = new BusCallData{loop_, this};
  thread_ = std::thread(&Pipeline::run, this);
  return *this;
}

Pipeline& Pipeline::start(std::function<void(Pipeline&, const Message&)> listener) {
  bus_data_ = new BusCallData{loop_, this, std::move(listener)};
  thread_ = std::thread(&Pipeline::run, this);
  return *this;
}

Pipeline &Pipeline::activate()
{
  thread_ = std::thread(&Pipeline::run_after_prepare, this);
  return *this;
}

Pipeline& Pipeline::stop() {
  GstState cur = GST_STATE_NULL;
  GstState pending = GST_STATE_NULL;
  GstStateChangeReturn ret;
  GstClockTime timeout = 5 * GST_SECOND / 1000;

  ret = gst_element_get_state(GST_ELEMENT(object_), &cur, &pending, timeout);

  if (ret == GST_STATE_CHANGE_SUCCESS && cur == GST_STATE_NULL) {
    return *this;
  }

  gst_element_send_event(GST_ELEMENT(object_), gst_event_new_eos());
  g_print("Posted EOS to stop pipeline\n");

  return *this;
}

Pipeline& Pipeline::pause() {
  GstState cur = GST_STATE_NULL;
  GstState pending = GST_STATE_NULL;
  GstStateChangeReturn ret;
  GstClockTime timeout = 5 * GST_SECOND / 1000;

  ret =
      gst_element_get_state(GST_ELEMENT(object_), &cur, &pending,
      timeout);

  if (ret == GST_STATE_CHANGE_ASYNC) {
    return *this;
  }

  if (cur == GST_STATE_PLAYING) {
    gst_element_set_state(GST_ELEMENT(object_), GST_STATE_PAUSED);
    gst_element_get_state(GST_ELEMENT(object_), &cur, &pending,
        GST_CLOCK_TIME_NONE);
  }

  return *this;
}

Pipeline& Pipeline::resume() {
  GstState cur = GST_STATE_NULL;
  GstState pending = GST_STATE_NULL;
  GstStateChangeReturn ret;
  GstClockTime timeout = 5 * GST_SECOND / 1000;

  ret =
      gst_element_get_state(GST_ELEMENT(object_), &cur, &pending,
      timeout);

  if (ret == GST_STATE_CHANGE_ASYNC) {
    return *this;
  }

  if (cur == GST_STATE_PAUSED) {
    gst_element_set_state(GST_ELEMENT(object_), GST_STATE_PLAYING);
    gst_element_get_state(GST_ELEMENT(object_), &cur, &pending,
        GST_CLOCK_TIME_NONE);
  }

  return *this;
}

Pipeline& Pipeline::seek(uint64_t timestamp) {
  if (!thread_.joinable()) {
    // don't seek before mainloop is started
    start_pts_ = timestamp;
  } else if (!gst_element_seek_simple(
    GST_ELEMENT(object_), GST_FORMAT_TIME,
    (GstSeekFlags)(GST_SEEK_FLAG_FLUSH | GST_SEEK_FLAG_KEY_UNIT | GST_SEEK_FLAG_SNAP_BEFORE),
    timestamp
  )) {
    g_printerr("Pipeline::seek failed\n");
  }
  return *this;
}

bool Pipeline::isRunning() const {
  return loop_ && g_main_loop_is_running((GMainLoop*) loop_);
}

static gboolean
start_rtsp_streaming (guint rtsp_port_num, guint updsink_port_num,
    NvDsEncoderType enctype, guint64 udp_buffer_size)
{
  GstRTSPMountPoints *mounts;
  GstRTSPMediaFactory *factory;
  char udpsrc_pipeline[512];

  char port_num_Str[64] = { 0 };
  const char *encoder_name;

  if (enctype == NV_DS_ENCODER_H264) {
    encoder_name = "H264";
  } else if (enctype == NV_DS_ENCODER_H265) {
    encoder_name = "H265";
  } else {
    g_printerr("%s failed", __func__);
    return FALSE;
  }

  if (udp_buffer_size == 0)
    udp_buffer_size = 512 * 1024;

  sprintf (udpsrc_pipeline,
      "( udpsrc name=pay0 port=%d buffer-size=%lu caps=\"application/x-rtp, media=video, "
      "clock-rate=90000, encoding-name=%s, payload=96 \" )",
      updsink_port_num, udp_buffer_size, encoder_name);

  sprintf (port_num_Str, "%d", rtsp_port_num);

  g_mutex_lock (&server_cnt_lock);

  server[server_count] = gst_rtsp_server_new ();
  g_object_set (server[server_count], "service", port_num_Str, NULL);

  mounts = gst_rtsp_server_get_mount_points (server[server_count]);

  factory = gst_rtsp_media_factory_new ();
  gst_rtsp_media_factory_set_shared (factory, TRUE);
  gst_rtsp_media_factory_set_launch (factory, udpsrc_pipeline);

  gst_rtsp_mount_points_add_factory (mounts, "/ds-test", factory);

  g_object_unref (mounts);

  gst_rtsp_server_attach (server[server_count], NULL);

  server_count++;

  g_mutex_unlock (&server_cnt_lock);

  g_print
      ("\n *** DeepStream: Launched RTSP Streaming at rtsp://localhost:%d/ds-test ***\n\n",
      rtsp_port_num);

  return TRUE;
}

uint32_t Pipeline::startRTSP(uint16_t rtsp_port, uint16_t udp_port, uint32_t buffer_size) {
  start_rtsp_streaming(rtsp_port, udp_port, NV_DS_ENCODER_H264, buffer_size);
  g_mutex_lock(&server_cnt_lock);
  uint32_t index = server_count - 1;
  g_mutex_unlock(&server_cnt_lock);
  return index;
}

static void
sr_done_cb (GstElement * src, NvDsSRRecordingInfo* recordingInfo, void* data, void* user_data) {
  SrDoneData* srDoneData = static_cast<SrDoneData*>(user_data);
  if (srDoneData) {
    if (srDoneData->callback) {
      srDoneData->callback(RecordingInfo(recordingInfo));
    } else {
      g_printerr("Callback is null for %s\n", GST_ELEMENT_NAME(src));
    }
  }
}

uint32_t Pipeline::startRecording(const std::string& element_name,
                              uint32_t startTime,
                              uint32_t duration,
                              std::function<void(const RecordingInfo &)> callback,
                              void* userData) {
  Element* element = find(element_name);
  if (!element) {
    g_printerr("Element %s not found\n", element_name.c_str());
    return 0;
  }
  GstElement *gst_elem = GST_ELEMENT(element->getGObject());
  GstElementFactory *fac = gst_element_get_factory(gst_elem);
  bool is_nvurisrcbin = fac && g_strcmp0(gst_plugin_feature_get_name(GST_PLUGIN_FEATURE(fac)), "nvurisrcbin") == 0;
  if (!is_nvurisrcbin) {
    g_printerr("Element %s is not a nvurisrcbin\n", element_name.c_str());
    return 0;
  }
  if (element_to_session_id.find(element_name) != element_to_session_id.end()) {
    g_printerr("Element %s is already recording\n", element_name.c_str());
    return 0;
  }

  SrDoneData* data = new SrDoneData{std::move(callback)};
  if (!g_signal_connect_data(gst_elem, "sr-done", (GCallback)sr_done_cb, data,
      [](gpointer user_data, GClosure*) { delete static_cast<SrDoneData*>(user_data); }, (GConnectFlags)0)) {
    g_printerr("Failed to connect sr-done signal\n");
    delete data;
    return 0;
  }
  NvDsSRSessionId sessionId = 0;
  GST_DEBUG("Start recording %s\n", element_name.c_str());
  g_signal_emit_by_name(gst_elem, "start-sr", &sessionId, (guint)startTime, (guint)duration, (gpointer)userData);
  element_to_session_id[element_name] = (uint32_t)sessionId;
  session_id_to_element[(uint32_t)sessionId] = element_name;
  return (uint32_t)sessionId;
}

bool Pipeline::stopRecording(const std::string& element_name) {
  Element* element = find(element_name);
  if (!element) {
    g_printerr("Element %s not found\n", element_name.c_str());
    return false;
  }
  GstElement *gst_elem = GST_ELEMENT(element->getGObject());
  GstElementFactory *fac = gst_element_get_factory(gst_elem);
  bool is_nvurisrcbin = fac && g_strcmp0(gst_plugin_feature_get_name(GST_PLUGIN_FEATURE(fac)), "nvurisrcbin") == 0;
  if (!is_nvurisrcbin) {
    g_printerr("Element %s is not a nvurisrcbin\n", element_name.c_str());
    return false;
  }
  if (element_to_session_id.find(element_name) == element_to_session_id.end()) {
    g_printerr("Element %s is not recording\n", element_name.c_str());
    return false;
  }
  uint32_t sid = element_to_session_id[element_name];
  GST_DEBUG("Stop recording %s\n", element_name.c_str());
  g_signal_emit_by_name(gst_elem, "stop-sr", (guint)sid);
  element_to_session_id.erase(element_name);
  session_id_to_element.erase(sid);

  return true;
}

bool Pipeline::stopRecording(uint32_t sessionId) {
  if (session_id_to_element.find(sessionId) == session_id_to_element.end()) {
    g_printerr("Session %d not found\n", sessionId);
    return false;
  }
  std::string element_name = session_id_to_element[sessionId];
  return stopRecording(element_name);
}
