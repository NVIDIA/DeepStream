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

/**
 * @file
 * <b>Element class </b>
 *
 * An element is the basic processing Unit in a pipeline created by Service Maker
 * Defined within various Deepstream plugins, the functionalities of elements cover
 * a wide range including audio/video decoding, batching, inference, tracking,
 * rendering, post-processing, pre-processing, etc.
 *
 */

#ifndef ELEMENT_HPP
#define ELEMENT_HPP

#include <string>
#include <memory>
#include <vector>
#include <typeinfo>
#include <algorithm>

#include "yaml-cpp/yaml.h"
#include "object.hpp"
#include "buffer_probe.hpp"
#include "signal_emitter.hpp"
#include "signal_handler.hpp"

namespace deepstream {

class Pipeline;

/**
 * @brief Element class definition
 *
 * Element class derives from the Object class and is a reference based wrapper,
 * supporting copying and moving.
 */
class Element : public Object {
 public:

   typedef enum {
    INVALID,
    EMPTY,
    READY,
    PAUSED,
    PLAYING
  } State;

  /**
   * @brief Constructor
   *
   * @param[in] type_name  ehe type name for a specific element defined with Deepstream SDK
   * @param[in] name       name given to the Element instance
   */
  Element(const std::string& type_name, std::string name = std::string());

  /** @brief Downcast copy constructor from Object class */
  Element(const Object&);

  /** @brief Downcast move constructor from Object class */
  Element(Object&&);

  /** @brief Destructor */
  virtual ~Element();

  /**
   * @brief Link two Element instances using hint
   *
   * @param[in] dst   target element to which this element links
   * @param[in] hint  hint for the link, providing the source pad name and target pad name
   * @return          target element
   */
  Element & link(Element &dst, std::pair<std::string, std::string> hint);

  /**
   * @brief Link two Element instances directly
   *
   * @param[in] dst   target element to which this element links
   * @return          target element
   */
  Element& link(Element &other);

  /** @brief Find the buffer probe attached to the elmenent by name */
  BufferProbe* getProbe(const std::string& name) {
    return find_<BufferProbe>(name);
  }

  /**
   * @brief Create and add a buffer probe to the element
   *
   * A buffer probe will be automatically created by the factory in the plugin.
   *
   * @param[in] plugin_name   name of the plugin where the factory for the buffer probe is defined
   * @param[in] probe_name    name of the buffer probe to be added
   * @param[in] probe_tip     extra information for adding the probe, e.g name of the pad
   *
   */
  Element& addProbe(const std::string& plugin_name,
                    const std::string& probe_name,
                    const std::string probe_tip = "");

  /** @brief Template function for creating and adding buffer probe with properties */
  template<typename... Args>
  Element& addProbe(const std::string& plugin_name,
                    const std::string& probe_name,
                    const std::string probe_tip = "",
                    const Args&... args) {
    addProbe(plugin_name, probe_name, probe_tip);
    auto probe = getProbe(probe_name);
    if (probe && (sizeof...(args) > 0)) {
      probe->set(args...);
    }
    return *this;
  }

  /**
   * @brief Add a BufferProbe instance to the element
   *
   * Once the probe is added, the element will take the ownership
   *
   * @param[in] probe         pointer to the buffer probe
   * @param[in] probe_tip     extra information for adding the probe, e.g name of the pad
   *
   */
  Element& addProbe(BufferProbe* probe, const std::string probe_tip  = "");

  /** @brief Find the signal handler attached to the element by name */
  SignalHandler* getSignalHandler(const std::string& name) {
    return find_<SignalHandler>(name);
  }

  /**
   * @brief Connect a signal handler to the element
   *
   * A signal handler is responsible for handling the asynchronous signals emitted from
   * the element. Signals behave like events and are defined by their names. Detailed
   * information of signal types available within an element can be found from the Plugin
   * Manual of Deepstream.
   *
   * Once connected, the element takes the ownership of the signal handler
   *
   * @param[in] signal_name   name of the signal to which the handler is connected
   * @param[in] probe_tip     pointer to the signal handler
   *
   */
  Element& connectSignal(const std::string&signal_name, SignalHandler* handler);

  /**
   * @brief Create and connect a handler to element's signal
   *
   * A signal handler will be created automatically by the factory defined in the plugin
   *
   * @param[in] plugin_name   name of the plugin where the factory of the signal handler is defined
   * @param[in] handler_name  name of the signal handler to be added
   * @param[in] signal_names  names of the signals for the handler to connect, separated by "/"
   *
   */
  Element& connectSignal(const std::string& plugin_name,
                         const std::string& handler_name,
                         const std::string& signal_names);

  /** @brief Template function for creating and connecting signal with properties */
  template<typename... Args>
  Element& connectSignal(const std::string& plugin_name,
                         const std::string& handler_name,
                         const std::string& signal_names,
                         const Args&... args) {
    connectSignal(plugin_name, handler_name, signal_names);
    auto handler = getSignalHandler(handler_name);
    if (handler && (sizeof...(args) > 0)) {
      handler->set(args...);
    }
    return *this;
  }

  const Pipeline& getPipeline() {
    return *pipeline_;
  }

  /** @brief Set the Element in particular state */
  bool setState (Element::State state);

protected:
  Element(GstObject* object);

  Element& add_(CustomObject* object);

  // buffer probes, signal handlers, signal emitters are held by its target
  // element as shared pointers.
  std::shared_ptr<std::unordered_map<std::string, std::unique_ptr<CustomObject>>> objects_;
  // pointer to the pipeline
  Pipeline* pipeline_=nullptr;

  template<class T>
  T* find_(const std::string& name) {
    auto itr = find_if(
      objects_->begin(),
      objects_->end(),
      [&](const auto& pair) {
        return pair.first == name && dynamic_cast<T*>(pair.second.get());
      }
    );
    if (itr != objects_->end()) return dynamic_cast<T*>(itr->second.get());
    return nullptr;
  }

friend class Pipeline;
};

} // namespace deepstream

#endif
