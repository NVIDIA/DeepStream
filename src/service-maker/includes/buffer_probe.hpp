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
 * <b>Service maker buffer probe definitions </b>
 *
 * @b Description: Buffer probe offers a mechnism for peeking the output buffers.
 * Both the data and the metadata carried by the buffer is accessible through
 * buffer probe.
 *
 * Yet it is not recommended to perform complex processing within a probe, which
 * could potentially disrupt the running pipeline. For data processing purpose,
 * data receiver is the right choice. @see DataReceiver.
 */

#ifndef NVIDIA_DEEPSTREAM_BUFFER_PROBE
#define NVIDIA_DEEPSTREAM_BUFFER_PROBE

#include "custom_object.hpp"
#include "factory_metadata.h"
#include "metadata.hpp"
#include "buffer.hpp"
#include "pad.hpp"

namespace deepstream {

class Element;

/**
 * Return values from user implemented probe interfaces
 */
enum class probeReturn {
  /** Nothing abnormal, the buffer will be treated as usual */
  Probe_Ok,
  /** Something wrong, indicating the pipeline to drop the buffer */
  Probe_Drop
};

/**
 * @brief Represent a custom object for the purpose of probing output buffers
 *
 * Appropriate interface must be implemented and assigned to an BufferProbe
 * instance for it to work.
 *
 * BufferProbe instances are not copyable/movable.
 *
 */
class BufferProbe : public CustomObject {
public:
  /**
   * @brief  Root interface required by a BufferProbe instance
   */
  class IHandler {
   public:
    virtual ~IHandler() {}
  };

  /**
   * @brief  Derived interface for handling metadata.
   */
  class IMetadataHandler : public IHandler {
   public:
    virtual ~IMetadataHandler() {}
  };

  /**
   * @brief  Derived interface for handling buffer itself.
   */
  class IBufferHandler : public IHandler {
   public:
    virtual ~IBufferHandler() {}
  };

  /**
   * @brief  Readonly interface for handling buffer.
   */
  class IBufferObserver : public IBufferHandler {
   public:
    virtual probeReturn handleBuffer(BufferProbe& probe, const Buffer&) = 0;
  };

  /**
   * @brief  Read/write interface for handling buffer.
   */
  class IBufferOperator : public IBufferHandler {
   public:
    virtual probeReturn handleBuffer(BufferProbe& probe, Buffer&) = 0;
  };

  /**
   * @brief  Readonly interface for handling batch metadata.
   */
  class IBatchMetadataObserver : public IMetadataHandler {
   public:
    virtual probeReturn handleData(BufferProbe& probe, const BatchMetadata& data) = 0;
  };

  /**
   * @brief  Read/write interface for handling batch metadata.
   */
  class IBatchMetadataOperator : public IMetadataHandler {
   public:
    virtual probeReturn handleData(BufferProbe& probe, BatchMetadata& data) = 0;
  };

  /**
   * @brief  Constructor
   *
   * Create a BufferProbe instance with user implemented handler interface
   *
   * @param[in] name        name of the instance
   * @param[in] handler     implementation of the IHandler interface
   */
  BufferProbe(const std::string& name, IHandler* handler);

  /**
   * @brief  Constructor for factory
   *
   * Create a BufferProbe instance with user implemented handler interface
   *
   * @param[in] name        name of the instance
   * @param[in] handler     implementation of the IHandler interface
   */
  BufferProbe(const std::string& name, const char* factory, IHandler* handler);

  /**
   * @brief Destructor
  */
  virtual ~BufferProbe();

  /**
   * @brief  Get the unique id associated with this type of BufferProbe
   *
   * An unique number is assigned for each BufferProbe class defined within
   * a custom plugin.
   */
  static unsigned long type();

  /**
   * @brief  Attach the BufferProbe instance to an Element instance
   *
   * The probe must be attached to the output.
   *
   * @param[in] target   pointer to the Element instance
   * @param[in] pad      target pad from which the probe takes buffers
   */
  BufferProbe& attach(Element* target, Pad pad);

  /**
   * @brief  Return the pointer to the element where the probe is attached
   *
   */
  Element* getTarget() { return target_; }

  /**
   * @brief  Return the pad from which the probe takes buffers
   *
   */
  const Pad& getPad() { return pad_; }

  /**
   * @brief  Template class to query the type of the handler inteface
   *
   */
  template<typename T>
  bool query(T*& interface) const {
    auto ptr = metadata_handler_.get();
    interface = dynamic_cast<T*>(ptr);
    return interface != nullptr;
  }

protected:
  /**
   * Buffer probe takes the ownership of the handler instance assigned to it
   * during construction.
   */
  std::unique_ptr<IHandler> metadata_handler_;
  /** the Pad from which the probe takes buffers, @see Pad */
  Pad pad_;
  /**
   * weak reference to the target element to which it is attached,
   * @see Element
   */
  Element* target_;
};

}


#endif