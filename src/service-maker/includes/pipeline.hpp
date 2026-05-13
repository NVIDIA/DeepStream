/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

/**
 * @file
 * <b>Pipeline definition </b>
 *
 * The Pipeline class serves as the foundation of Deepstream-based AI streaming applications.
 * Media streams flow through interconnected elements within a Pipeline instance, processed
 * via buffers and metadata.
 *
 */
#ifndef PIPELINE_HPP
#define PIPELINE_HPP

#include <string>
#include <memory>
#include <thread>
#include <unordered_map>

#include "object.hpp"
#include "element.hpp"
#include "buffer_probe.hpp"
#include "signal_handler.hpp"
#include "metadata.hpp"

namespace deepstream
{


/**
 * @brief Holds information about a recording session.
 *
 * This struct is used to hold information about a recording session.
 */

struct RecordingInfo {
    RecordingInfo(void* data=nullptr);
    virtual ~RecordingInfo();
  
    /** @brief Get the recording session id */
    unsigned int getSessionId() const;
    /** @brief Get the recording file name */
    std::string getFileName() const;
    /** @brief Get the recording file directory */
    std::string getFileDirectory() const;
    /** @brief Get the recording file duration */
    unsigned int getDuration() const;
    /** @brief Get the recording file container type */
    std::string getContainerType() const;
    /** @brief Get the recording file width */
    unsigned int getWidth() const;
    /** @brief Get the recording file height */
    unsigned int getHeight() const;
    /** @brief Get the recording file contains video */
    bool containsVideo() const;
    /** @brief Get the recording file contains audio */
    bool containsAudio() const;
    /** @brief Get the recording file channels */
    unsigned int getChannels() const;
    /** @brief Get the recording file sampling rate */
    unsigned int getSamplingRate() const;

    operator bool() const {
      return data_ != nullptr;
    }
  
   protected:
    void * data_;
    
};

/**
 * @brief Pipeline class definition
 *
 * Pipelie class provide the high level interface for most of the functionalities.
 * It is recommended to use Pipeline API for creating applications unless there is
 * some specific features that are only supported in lower level APIs from Element,
 * Object, etc.
 *
 */
class Pipeline : public Object {
 public:

  typedef enum {
    INVALID,
    EMPTY,
    READY,
    PAUSED,
    PLAYING
  } State;

  /**
   * @brief base class for pipeline message
   *
   * Pipeline message provides a mechanism for pipeline to
   * notify the application of something happening
  */
  class Message {
    public:
      virtual ~Message() {}
      uint32_t type() { return type_; }
    protected:
      uint32_t type_;

      Message(uint32_t type): type_(type) {}
  };

  /**
   * @brief Pipeline message on EOS
   */
  class EOSMessage : public Message {
    public:
      EOSMessage(void*);
  };

  /**
   * @brief Pipeline message on source add/remove
  */
  class DynamicSourceMessage : public Message {
    public:
      DynamicSourceMessage(void*);

      inline bool isSourceAdded() const { return source_added_; }
      inline uint32_t getSourceId() const { return source_id_; }
      inline const std::string& getSensorId() const { return sensor_id_; }
      inline const std::string& getSensorName() const { return sensor_name_; }
      inline const std::string& getUri() const { return uri_; }

    protected:
      bool source_added_ = false;
      uint32_t source_id_ = 0;
      std::string sensor_id_;
      std::string sensor_name_;
      std::string uri_;
  };

  /**
   * @brief Pipeline message on state transition
  */
  class StateTransitionMessage : public Message {
    public:
      StateTransitionMessage(void*);
      inline const std::string& getName() const { return name_; };
      inline void getState(State& old_state, State& new_state) const {
        old_state = old_state_;
        new_state = new_state_;
      }
    protected:
      State old_state_;
      State new_state_;
      std::string name_;
  };

  Pipeline(const char* name);

  /** @brief Constructor with name and a description file */
  Pipeline(const char* name, const std::string& config_file);

  /** @brief Destructor */
  virtual ~Pipeline();

  /** @brief  Template function for creating and adding element with properties */
  template<typename... Args>
  Pipeline& add(const std::string& element_type, const std::string& element_name, const Args&... args) {
    Element element = Element(element_type, element_name);
    if constexpr (sizeof...(args) > 0) {
      element.set(args...);
    }
    return this->add(element);
  }

  /** @brief Add a given element to the pipeline */
  Pipeline& add(Element element);

  /** @brief Find an element within the pipeline by name */
  Element* find(const::std::string& name);

  /** @brief Operator for accessing elements in a pipeline */
  Element& operator[](const std::string& name) {
    Element* e = this->find(name);
    if (e == nullptr) {
      throw std::runtime_error(name + " Not found");
    }
    return *e;
  }

  /**
   * @brief Link two elements within the pipeline
   *
   * @param[in] route   a pair with source element name and target element name
   * @param[in] tips    extra pair with source pad name and target pad name
   */
  Pipeline& link(
    std::pair<std::string, std::string> route,
    std::pair<std::string, std::string> tips
  );

  /** @brief Template function for linking elements in the simplest way */
  template <typename... Args>
  Pipeline &link(const std::string &arg1, const std::string arg2, const Args &...args)
  {
    link(std::make_pair(arg1, arg2), std::make_pair("", ""));
    if constexpr (sizeof...(args) > 0) {
        return this->link(arg2, args...);
    } else {
        return *this;
    }
  }

  /**
   * @brief Attach a custom object to an element within the pipeline
   *
   * Supported custom objects can be buffer probes, signal handlers
   * Once the object is attached, the element takes the ownership of it
   *
   * @param[in] element_name   name of the elment to which the object attaches
   * @param[in] object         pointer to a custom object
   * @param[in] tips           extra information. pad name for buffer probes, signal name for signal handlers
   */
  Pipeline& attach(const std::string& elmenent_name, CustomObject* object, const std::string tip="");

  /**
   * @brief Create and attach a custom object to an element within the pipeline
   *
   * Supported custom objects can be buffer probes, signal handlers
   * The custom object will be created through factory
   *
   * @param[in] element_name   name of the elment to which the object attaches
   * @param[in] plugin_name    name of the plugin where the custom object factory is defined
   * @param[in] object_name    name of the new custome object
   * @param[in] tips           extra information. pad name for buffer probes, signal name for signal handlers
   */
  Pipeline& attach(
    const std::string& element_name,
    const std::string& plugin_name,
    const std::string& object_name,
    const std::string tip="");

  /** @brief Template function for creating and attaching custom object with properties */
  template<typename... Args>
  Pipeline& attach(
    const std::string& element_name,
    const std::string& plugin_name,
    const std::string& object_name,
    const std::string tip,
    const Args&... args) {
    attach(element_name, plugin_name, object_name, tip);
    if (sizeof...(args) > 0) {
      auto& element = (*this)[element_name];
      if (auto handler = element.getSignalHandler(object_name)) {
        handler->set(args...);
      } else if (auto probe = element.getProbe(object_name)) {
        probe->set(args...);
      }
    }
    return *this;
  }

  /** @brief install a callback to capture keyboard events */
  Pipeline& install(std::function<void(Pipeline&, int key)> keyboard_listener) {
    keyboard_listener_ = keyboard_listener;
    return *this;
  }

  /** @brief Intialize the pipeline */
  int prepare();
  /** @brief Intialize the pipeline with a callback to capture the messages */
  int prepare(std::function<void(Pipeline &, const Message &)> listener);

  /** @brief Start the pipeline */
  Pipeline& start();
  /** @brief Start the pipeline with a callback to capture the messages */
  Pipeline& start(std::function<void(Pipeline&, const Message&)> listener);
  /** @brief Start the pipeline after it is already intialized*/
  Pipeline &activate();
  /** @brief Wait until the pipeline ends */
  Pipeline& wait();

  /** @brief Stop the pipeline */
  Pipeline& stop();

  bool isRunning() const;

  /** @brief Pause the pipeline */
  Pipeline& pause();
  /** @brief Resume the pipeline */
  Pipeline& resume();
  /** @brief Seek to a specified position for processing data within the pipeline */
  Pipeline& seek(uint64_t timestamp);

  /** @brief Start an RTSP server*/
  uint32_t startRTSP(uint16_t rtsp_port, uint16_t udp_port, uint32_t buffer_size=0);

  /**
   * @brief Send Smart Record start signal to an nvurisrcbin element
   *
   * @param[in] element_name  name of the nvurisrcbin element
   * @param[in] startTime     seconds of cached data to include before trigger (0 = from now)
   * @param[in] duration      seconds to record (0 = element default)
   * @param[in] userData      optional user data pointer passed to callback
   */
  uint32_t startRecording(const std::string& element_name,
                      uint32_t startTime = 0,
                      uint32_t duration = 0,
                      std::function<void(const RecordingInfo &)> callback = [](const RecordingInfo &info) {
                        printf("\nSource recorded at %s/%s. Duration %.2f sec\n",
                          info.getFileDirectory().c_str(), info.getFileName().c_str(),
                          info.getDuration() / 1000.0);
                      },
                      void* userData = nullptr);

  /**
   * @brief Send Smart Record stop signal to an nvurisrcbin element
   *
   * @param[in] element_name  name of the nvurisrcbin element
   */
  bool stopRecording(const std::string& element_name);

  /**
   * @brief Send Smart Record stop signal to an nvurisrcbin element
   *
   * @param[in] sessionId  session id returned by startRecording
   */
  bool stopRecording(uint32_t sessionId);

  void handleKey(int key);

protected:
  int run();
  int run_after_prepare();

  // This is GMainLoop*
  void *loop_ = NULL;
  // gstreamer bus watch id
  uint bus_watch_id_ = 0;
  // gstreamer bus data
  void *bus_data_ = NULL;
  std::function<void(Pipeline&, int key)> keyboard_listener_;

  // Thread
  std::thread thread_;
  // Hold all the element references here
  std::map<std::string, Element> elements_;
  // Hold the signal emitters for action control
  std::map<std::string, std::unique_ptr<SignalEmitter>> action_owners_;
  // start timestamp for the whole pipeline
  uint64_t start_pts_ = 0;
  // Smart Record session id per element name
  std::unordered_map<std::string, uint32_t> element_to_session_id;
  std::unordered_map<uint32_t, std::string> session_id_to_element;
};

} // namespace deepstream

#endif
