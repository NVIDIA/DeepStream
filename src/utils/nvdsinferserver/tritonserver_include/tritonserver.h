// Copyright 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#pragma once

/// \file

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _COMPILING_TRITONSERVER
#if defined(_MSC_VER)
#define TRITONSERVER_DECLSPEC __declspec(dllexport)
#elif defined(__GNUC__)
#define TRITONSERVER_DECLSPEC __attribute__((__visibility__("default")))
#else
#define TRITONSERVER_DECLSPEC
#endif
#else
#if defined(_MSC_VER)
#define TRITONSERVER_DECLSPEC __declspec(dllimport)
#else
#define TRITONSERVER_DECLSPEC
#endif
#endif

struct TRITONSERVER_BufferAttributes;
struct TRITONSERVER_Error;
struct TRITONSERVER_InferenceRequest;
struct TRITONSERVER_InferenceResponse;
struct TRITONSERVER_InferenceTrace;
struct TRITONSERVER_Message;
struct TRITONSERVER_Metrics;
struct TRITONSERVER_Parameter;
struct TRITONSERVER_ResponseAllocator;
struct TRITONSERVER_Server;
struct TRITONSERVER_ServerOptions;
struct TRITONSERVER_Metric;
struct TRITONSERVER_MetricFamily;
struct TRITONSERVER_MetricArgs;

///
/// TRITONSERVER API Version
///
/// The TRITONSERVER API is versioned with major and minor version
/// numbers. Any change to the API that does not impact backwards
/// compatibility (for example, adding a non-required function)
/// increases the minor version number. Any change that breaks
/// backwards compatibility (for example, deleting or changing the
/// behavior of a function) increases the major version number. A
/// client should check that the API version used to compile the
/// client is compatible with the API version of the Triton shared
/// library that it is linking against. This is typically done by code
/// similar to the following which makes sure that the major versions
/// are equal and that the minor version of the Triton shared library
/// is >= the minor version used to build the client.
///
///   uint32_t api_version_major, api_version_minor;
///   TRITONSERVER_ApiVersion(&api_version_major, &api_version_minor);
///   if ((api_version_major != TRITONSERVER_API_VERSION_MAJOR) ||
///       (api_version_minor < TRITONSERVER_API_VERSION_MINOR)) {
///     return TRITONSERVER_ErrorNew(
///       TRITONSERVER_ERROR_UNSUPPORTED,
///       "triton server API version does not support this client");
///   }
///
#define TRITONSERVER_API_VERSION_MAJOR 1
#define TRITONSERVER_API_VERSION_MINOR 34

/// Get the TRITONBACKEND API version supported by the Triton shared
/// library. This value can be compared against the
/// TRITONSERVER_API_VERSION_MAJOR and TRITONSERVER_API_VERSION_MINOR
/// used to build the client to ensure that Triton shared library is
/// compatible with the client.
///
/// \param major Returns the TRITONSERVER API major version supported
/// by Triton.
/// \param minor Returns the TRITONSERVER API minor version supported
/// by Triton.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error* TRITONSERVER_ApiVersion(
    uint32_t* major, uint32_t* minor);

/// TRITONSERVER_DataType
///
/// Tensor data types recognized by TRITONSERVER.
///
typedef enum TRITONSERVER_datatype_enum {
  TRITONSERVER_TYPE_INVALID,
  TRITONSERVER_TYPE_BOOL,
  TRITONSERVER_TYPE_UINT8,
  TRITONSERVER_TYPE_UINT16,
  TRITONSERVER_TYPE_UINT32,
  TRITONSERVER_TYPE_UINT64,
  TRITONSERVER_TYPE_INT8,
  TRITONSERVER_TYPE_INT16,
  TRITONSERVER_TYPE_INT32,
  TRITONSERVER_TYPE_INT64,
  TRITONSERVER_TYPE_FP16,
  TRITONSERVER_TYPE_FP32,
  TRITONSERVER_TYPE_FP64,
  TRITONSERVER_TYPE_BYTES,
  TRITONSERVER_TYPE_BF16
} TRITONSERVER_DataType;

/// Get the string representation of a data type. The returned string
/// is not owned by the caller and so should not be modified or freed.
///
/// \param datatype The data type.
/// \return The string representation of the data type.
TRITONSERVER_DECLSPEC const char* TRITONSERVER_DataTypeString(
    TRITONSERVER_DataType datatype);

/// Get the Triton datatype corresponding to a string representation
/// of a datatype.
///
/// \param dtype The datatype string representation.
/// \return The Triton data type or TRITONSERVER_TYPE_INVALID if the
/// string does not represent a data type.
TRITONSERVER_DECLSPEC TRITONSERVER_DataType
TRITONSERVER_StringToDataType(const char* dtype);

/// Get the size of a Triton datatype in bytes. Zero is returned for
/// TRITONSERVER_TYPE_BYTES because it have variable size. Zero is
/// returned for TRITONSERVER_TYPE_INVALID.
///
/// \param dtype The datatype.
/// \return The size of the datatype.
TRITONSERVER_DECLSPEC uint32_t
TRITONSERVER_DataTypeByteSize(TRITONSERVER_DataType datatype);

/// TRITONSERVER_MemoryType
///
/// Types of memory recognized by TRITONSERVER.
///
typedef enum TRITONSERVER_memorytype_enum {
  TRITONSERVER_MEMORY_CPU,
  TRITONSERVER_MEMORY_CPU_PINNED,
  TRITONSERVER_MEMORY_GPU
} TRITONSERVER_MemoryType;

/// Get the string representation of a memory type. The returned
/// string is not owned by the caller and so should not be modified or
/// freed.
///
/// \param memtype The memory type.
/// \return The string representation of the memory type.
TRITONSERVER_DECLSPEC const char* TRITONSERVER_MemoryTypeString(
    TRITONSERVER_MemoryType memtype);

/// TRITONSERVER_ParameterType
///
/// Types of parameters recognized by TRITONSERVER.
///
typedef enum TRITONSERVER_parametertype_enum {
  TRITONSERVER_PARAMETER_STRING,
  TRITONSERVER_PARAMETER_INT,
  TRITONSERVER_PARAMETER_BOOL,
  TRITONSERVER_PARAMETER_DOUBLE,
  TRITONSERVER_PARAMETER_BYTES
} TRITONSERVER_ParameterType;

/// Get the string representation of a parameter type. The returned
/// string is not owned by the caller and so should not be modified or
/// freed.
///
/// \param paramtype The parameter type.
/// \return The string representation of the parameter type.
TRITONSERVER_DECLSPEC const char* TRITONSERVER_ParameterTypeString(
    TRITONSERVER_ParameterType paramtype);

/// Create a new parameter object. The caller takes ownership of the
/// TRITONSERVER_Parameter object and must call TRITONSERVER_ParameterDelete to
/// release the object. The object will maintain its own copy of the 'value'
///
/// \param name The parameter name.
/// \param type The parameter type.
/// \param value The pointer to the value.
/// \return A new TRITONSERVER_Parameter object. 'nullptr' will be returned if
/// 'type' is 'TRITONSERVER_PARAMETER_BYTES'. The caller should use
/// TRITONSERVER_ParameterBytesNew to create parameter with bytes type.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Parameter* TRITONSERVER_ParameterNew(
    const char* name, const TRITONSERVER_ParameterType type, const void* value);

/// Create a new parameter object with type TRITONSERVER_PARAMETER_BYTES.
/// The caller takes ownership of the TRITONSERVER_Parameter object and must
/// call TRITONSERVER_ParameterDelete to release the object. The object only
/// maintains a shallow copy of the 'byte_ptr' so the data content must be
/// valid until the parameter object is deleted.
///
/// \param name The parameter name.
/// \param byte_ptr The pointer to the data content.
/// \param size The size of the data content.
/// \return A new TRITONSERVER_Error object.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Parameter*
TRITONSERVER_ParameterBytesNew(
    const char* name, const void* byte_ptr, const uint64_t size);

/// Delete an parameter object.
///
/// \param parameter The parameter object.
TRITONSERVER_DECLSPEC void TRITONSERVER_ParameterDelete(
    struct TRITONSERVER_Parameter* parameter);

/// TRITONSERVER_InstanceGroupKind
///
/// Kinds of instance groups recognized by TRITONSERVER.
///
typedef enum TRITONSERVER_instancegroupkind_enum {
  TRITONSERVER_INSTANCEGROUPKIND_AUTO,
  TRITONSERVER_INSTANCEGROUPKIND_CPU,
  TRITONSERVER_INSTANCEGROUPKIND_GPU,
  TRITONSERVER_INSTANCEGROUPKIND_MODEL
} TRITONSERVER_InstanceGroupKind;

/// Get the string representation of an instance-group kind. The
/// returned string is not owned by the caller and so should not be
/// modified or freed.
///
/// \param kind The instance-group kind.
/// \return The string representation of the kind.
TRITONSERVER_DECLSPEC const char* TRITONSERVER_InstanceGroupKindString(
    TRITONSERVER_InstanceGroupKind kind);

/// TRITONSERVER_Logging
///
/// Types/levels of logging.
///
typedef enum TRITONSERVER_loglevel_enum {
  TRITONSERVER_LOG_INFO,
  TRITONSERVER_LOG_WARN,
  TRITONSERVER_LOG_ERROR,
  TRITONSERVER_LOG_VERBOSE
} TRITONSERVER_LogLevel;

/// Logging Formats
///
/// The TRITONSERVER API offers two logging formats. The formats have
/// a common set of fields but differ in how the timestamp for a log
/// entry is represented. Messages are serialized according to JSON
/// encoding rules by default. This behavior can be disabled by
/// setting the environment variable TRITON_SERVER_ESCAPE_LOG_MESSAGES
/// to "0".
///
///
/// 1. TRITONSERVER_LOG_DEFAULT
///
/// <level><month><day><hour>:<min>:<sec>.<usec> <pid> <file>:<line>] <msg>
///
/// Example:
///
/// I0520 20:03:25.829575 3355 model_lifecycle.cc:441] "AsyncLoad() 'simple'"
///
/// 2. TRITONSERVER_LOG_ISO8601
///
/// <year>-<month>-<day>T<hour>:<min>:<sec>Z <level> <pid> <file>:<line>] <msg>
///
/// Example:
///
/// 2024-05-20T20:03:26Z I 3415 model_lifecycle.cc:441] "AsyncLoad() 'simple'"
///
typedef enum TRITONSERVER_logformat_enum {
  TRITONSERVER_LOG_DEFAULT,
  TRITONSERVER_LOG_ISO8601
} TRITONSERVER_LogFormat;

/// Is a log level enabled?
///
/// \param level The log level.
/// \return True if the log level is enabled, false if not enabled.
TRITONSERVER_DECLSPEC bool TRITONSERVER_LogIsEnabled(
    TRITONSERVER_LogLevel level);

/// Log a message at a given log level if that level is enabled.
///
/// \param level The log level.
/// \param filename The file name of the location of the log message.
/// \param line The line number of the log message.
/// \param msg The log message.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error* TRITONSERVER_LogMessage(
    TRITONSERVER_LogLevel level, const char* filename, const int line,
    const char* msg);

/// TRITONSERVER_Error
///
/// Errors are reported by a TRITONSERVER_Error object. A NULL
/// TRITONSERVER_Error indicates no error, a non-NULL TRITONSERVER_Error
/// indicates error and the code and message for the error can be
/// retrieved from the object.
///
/// The caller takes ownership of a TRITONSERVER_Error object returned by
/// the API and must call TRITONSERVER_ErrorDelete to release the object.
///

/// The TRITONSERVER_Error error codes
typedef enum TRITONSERVER_errorcode_enum {
  TRITONSERVER_ERROR_UNKNOWN,
  TRITONSERVER_ERROR_INTERNAL,
  TRITONSERVER_ERROR_NOT_FOUND,
  TRITONSERVER_ERROR_INVALID_ARG,
  TRITONSERVER_ERROR_UNAVAILABLE,
  TRITONSERVER_ERROR_UNSUPPORTED,
  TRITONSERVER_ERROR_ALREADY_EXISTS,
  TRITONSERVER_ERROR_CANCELLED
} TRITONSERVER_Error_Code;

/// Create a new error object. The caller takes ownership of the
/// TRITONSERVER_Error object and must call TRITONSERVER_ErrorDelete to
/// release the object.
///
/// \param code The error code.
/// \param msg The error message.
/// \return A new TRITONSERVER_Error object.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error* TRITONSERVER_ErrorNew(
    TRITONSERVER_Error_Code code, const char* msg);

/// Delete an error object.
///
/// \param error The error object.
TRITONSERVER_DECLSPEC void TRITONSERVER_ErrorDelete(
    struct TRITONSERVER_Error* error);

/// Get the error code.
///
/// \param error The error object.
/// \return The error code.
TRITONSERVER_DECLSPEC TRITONSERVER_Error_Code
TRITONSERVER_ErrorCode(struct TRITONSERVER_Error* error);

/// Get the string representation of an error code. The returned
/// string is not owned by the caller and so should not be modified or
/// freed. The lifetime of the returned string extends only as long as
/// 'error' and must not be accessed once 'error' is deleted.
///
/// \param error The error object.
/// \return The string representation of the error code.
TRITONSERVER_DECLSPEC const char* TRITONSERVER_ErrorCodeString(
    struct TRITONSERVER_Error* error);

/// Get the error message. The returned string is not owned by the
/// caller and so should not be modified or freed. The lifetime of the
/// returned string extends only as long as 'error' and must not be
/// accessed once 'error' is deleted.
///
/// \param error The error object.
/// \return The error message.
TRITONSERVER_DECLSPEC const char* TRITONSERVER_ErrorMessage(
    struct TRITONSERVER_Error* error);

/// TRITONSERVER_ResponseAllocator
///
/// Object representing a memory allocator for output tensors in an
/// inference response.
///

/// Type for allocation function that allocates a buffer to hold an
/// output tensor.
///
/// \param allocator The allocator that is provided in the call to
/// TRITONSERVER_InferenceRequestSetResponseCallback.
/// \param tensor_name The name of the output tensor to allocate for.
/// \param byte_size The size of the buffer to allocate.
/// \param memory_type The type of memory that the caller prefers for
/// the buffer allocation.
/// \param memory_type_id The ID of the memory that the caller prefers
/// for the buffer allocation.
/// \param userp The user data pointer that is provided as
/// 'response_allocator_userp' in the call to
/// TRITONSERVER_InferenceRequestSetResponseCallback.
/// \param buffer Returns a pointer to the allocated memory.
/// \param buffer_userp Returns a user-specified value to associate
/// with the buffer, or nullptr if no user-specified value should be
/// associated with the buffer. This value will be provided in the
/// call to TRITONSERVER_ResponseAllocatorReleaseFn_t when the buffer
/// is released and will also be returned by
/// TRITONSERVER_InferenceResponseOutput.
/// \param actual_memory_type Returns the type of memory where the
/// allocation resides. May be different than the type of memory
/// requested by 'memory_type'.
/// \param actual_memory_type_id Returns the ID of the memory where
/// the allocation resides. May be different than the ID of the memory
/// requested by 'memory_type_id'.
/// \return a TRITONSERVER_Error object if a failure occurs while
/// attempting an allocation. If an error is returned all other return
/// values will be ignored.
typedef struct TRITONSERVER_Error* (*TRITONSERVER_ResponseAllocatorAllocFn_t)(
    struct TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id, void* userp, void** buffer, void** buffer_userp,
    TRITONSERVER_MemoryType* actual_memory_type,
    int64_t* actual_memory_type_id);

/// Type for allocation function that allocates a buffer to hold an
/// output tensor with buffer attributes. The callback function must fill in the
/// appropriate buffer attributes information related to this buffer. If set,
/// this function is always called after TRITONSERVER_ResponseAllocatorAllocFn_t
/// function.
///
/// \param allocator The allocator that is provided in the call to
/// TRITONSERVER_InferenceRequestSetResponseCallback.
/// \param tensor_name The name of the output tensor to allocate for.
/// \param buffer_attributes The buffer attributes associated with the buffer.
/// \param userp The user data pointer that is provided as
/// 'response_allocator_userp' in the call to
/// TRITONSERVER_InferenceRequestSetResponseCallback.
/// \param buffer_userp Returns a user-specified value to associate
/// with the buffer, or nullptr if no user-specified value should be
/// associated with the buffer. This value will be provided in the
/// call to TRITONSERVER_ResponseAllocatorReleaseFn_t when the buffer
/// is released and will also be returned by
/// TRITONSERVER_InferenceResponseOutput.
/// \return a TRITONSERVER_Error object if a failure occurs while
/// attempting an allocation. If an error is returned all other return
/// values will be ignored.
typedef struct TRITONSERVER_Error* (
    *TRITONSERVER_ResponseAllocatorBufferAttributesFn_t)(
    struct TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    struct TRITONSERVER_BufferAttributes* buffer_attributes, void* userp,
    void* buffer_userp);

/// Type for function that is called to query the allocator's preferred memory
/// type and memory type ID. As much as possible, the allocator should attempt
/// to return the same memory_type and memory_type_id values that will be
/// returned by the subsequent call to TRITONSERVER_ResponseAllocatorAllocFn_t.
/// But the allocator is not required to do so.
///
/// \param allocator The allocator that is provided in the call to
/// TRITONSERVER_InferenceRequestSetResponseCallback.
/// \param userp The user data pointer that is provided as
/// 'response_allocator_userp' in the call to
/// TRITONSERVER_InferenceRequestSetResponseCallback.
/// \param tensor_name The name of the output tensor. This is optional
/// and it should be set to nullptr to indicate that the tensor name has
/// not determined.
/// \param byte_size The expected size of the buffer. This is optional
/// and it should be set to nullptr to indicate that the byte size has
/// not determined.
/// \param memory_type Acts as both input and output. On input gives
/// the memory type preferred by the caller. Returns memory type preferred
/// by the allocator, taken account of the caller preferred type.
/// \param memory_type_id Acts as both input and output. On input gives
/// the memory type ID preferred by the caller. Returns memory type ID preferred
/// by the allocator, taken account of the caller preferred type ID.
/// \return a TRITONSERVER_Error object if a failure occurs.
typedef struct TRITONSERVER_Error* (*TRITONSERVER_ResponseAllocatorQueryFn_t)(
    struct TRITONSERVER_ResponseAllocator* allocator, void* userp,
    const char* tensor_name, size_t* byte_size,
    TRITONSERVER_MemoryType* memory_type, int64_t* memory_type_id);

/// Type for function that is called when the server no longer holds
/// any reference to a buffer allocated by
/// TRITONSERVER_ResponseAllocatorAllocFn_t. In practice this function
/// is typically called when the response object associated with the
/// buffer is deleted by TRITONSERVER_InferenceResponseDelete.
///
/// \param allocator The allocator that is provided in the call to
/// TRITONSERVER_InferenceRequestSetResponseCallback.
/// \param buffer Pointer to the buffer to be freed.
/// \param buffer_userp The user-specified value associated
/// with the buffer in TRITONSERVER_ResponseAllocatorAllocFn_t.
/// \param byte_size The size of the buffer.
/// \param memory_type The type of memory holding the buffer.
/// \param memory_type_id The ID of the memory holding the buffer.
/// \return a TRITONSERVER_Error object if a failure occurs while
/// attempting the release. If an error is returned Triton will not
/// attempt to release the buffer again.
typedef struct TRITONSERVER_Error* (*TRITONSERVER_ResponseAllocatorReleaseFn_t)(
    struct TRITONSERVER_ResponseAllocator* allocator, void* buffer,
    void* buffer_userp, size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id);

/// Type for function that is called to indicate that subsequent
/// allocation requests will refer to a new response.
///
/// \param allocator The allocator that is provided in the call to
/// TRITONSERVER_InferenceRequestSetResponseCallback.
/// \param userp The user data pointer that is provided as
/// 'response_allocator_userp' in the call to
/// TRITONSERVER_InferenceRequestSetResponseCallback.
/// \return a TRITONSERVER_Error object if a failure occurs.
typedef struct TRITONSERVER_Error* (*TRITONSERVER_ResponseAllocatorStartFn_t)(
    struct TRITONSERVER_ResponseAllocator* allocator, void* userp);

/// Create a new response allocator object.
///
/// The response allocator object is used by Triton to allocate
/// buffers to hold the output tensors in inference responses. Most
/// models generate a single response for each inference request
/// (TRITONSERVER_TXN_ONE_TO_ONE). For these models the order of
/// callbacks will be:
///
///   TRITONSERVER_ServerInferAsync called
///    - start_fn : optional (and typically not required)
///    - alloc_fn : called once for each output tensor in response
///   TRITONSERVER_InferenceResponseDelete called
///    - release_fn: called once for each output tensor in response
///
/// For models that generate multiple responses for each inference
/// request (TRITONSERVER_TXN_DECOUPLED), the start_fn callback can be
/// used to determine sets of alloc_fn callbacks that belong to the
/// same response:
///
///   TRITONSERVER_ServerInferAsync called
///    - start_fn
///    - alloc_fn : called once for each output tensor in response
///    - start_fn
///    - alloc_fn : called once for each output tensor in response
///      ...
///   For each response, TRITONSERVER_InferenceResponseDelete called
///    - release_fn: called once for each output tensor in the response
///
/// In all cases the start_fn, alloc_fn and release_fn callback
/// functions must be thread-safe. Typically making these functions
/// thread-safe does not require explicit locking. The recommended way
/// to implement these functions is to have each inference request
/// provide a 'response_allocator_userp' object that is unique to that
/// request with TRITONSERVER_InferenceRequestSetResponseCallback. The
/// callback functions then operate only on this unique state. Locking
/// is required only when the callback function needs to access state
/// that is shared across inference requests (for example, a common
/// allocation pool).
///
/// \param allocator Returns the new response allocator object.
/// \param alloc_fn The function to call to allocate buffers for result
/// tensors.
/// \param release_fn The function to call when the server no longer
/// holds a reference to an allocated buffer.
/// \param start_fn The function to call to indicate that the
/// subsequent 'alloc_fn' calls are for a new response. This callback
/// is optional (use nullptr to indicate that it should not be
/// invoked).
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_ResponseAllocatorNew(
    struct TRITONSERVER_ResponseAllocator** allocator,
    TRITONSERVER_ResponseAllocatorAllocFn_t alloc_fn,
    TRITONSERVER_ResponseAllocatorReleaseFn_t release_fn,
    TRITONSERVER_ResponseAllocatorStartFn_t start_fn);

/// Set the buffer attributes function for a response allocator object.
/// The function will be called after alloc_fn to set the buffer attributes
/// associated with the output buffer.
///
/// The thread-safy requirement for buffer_attributes_fn is the same as other
/// allocator callbacks.
///
/// \param allocator The response allocator object.
/// \param buffer_attributes_fn The function to call to get the buffer
/// attributes information for an allocated buffer.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_ResponseAllocatorSetBufferAttributesFunction(
    struct TRITONSERVER_ResponseAllocator* allocator,
    TRITONSERVER_ResponseAllocatorBufferAttributesFn_t buffer_attributes_fn);

/// Set the query function to a response allocator object. Usually the
/// function will be called before alloc_fn to understand what is the
/// allocator's preferred memory type and memory type ID at the current
/// situation to make different execution decision.
///
/// The thread-safy requirement for query_fn is the same as other allocator
/// callbacks.
///
/// \param allocator The response allocator object.
/// \param query_fn The function to call to query allocator's preferred memory
/// type and memory type ID.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_ResponseAllocatorSetQueryFunction(
    struct TRITONSERVER_ResponseAllocator* allocator,
    TRITONSERVER_ResponseAllocatorQueryFn_t query_fn);

/// Delete a response allocator.
///
/// \param allocator The response allocator object.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_ResponseAllocatorDelete(
    struct TRITONSERVER_ResponseAllocator* allocator);

/// TRITONSERVER_Message
///
/// Object representing a Triton Server message.
///

/// Create a new message object from serialized JSON string.
///
/// \param message The message object.
/// \param base The base of the serialized JSON.
/// \param byte_size The size, in bytes, of the serialized message.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_MessageNewFromSerializedJson(
    struct TRITONSERVER_Message** message, const char* base, size_t byte_size);

/// Delete a message object.
///
/// \param message The message object.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error* TRITONSERVER_MessageDelete(
    struct TRITONSERVER_Message* message);

/// Get the base and size of the buffer containing the serialized
/// message in JSON format. The buffer is owned by the
/// TRITONSERVER_Message object and should not be modified or freed by
/// the caller. The lifetime of the buffer extends only as long as
/// 'message' and must not be accessed once 'message' is deleted.
///
/// \param message The message object.
/// \param base Returns the base of the serialized message.
/// \param byte_size Returns the size, in bytes, of the serialized
/// message.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_MessageSerializeToJson(
    struct TRITONSERVER_Message* message, const char** base, size_t* byte_size);

/// TRITONSERVER_Metrics
///
/// Object representing metrics.
///

/// Metric format types
typedef enum tritonserver_metricformat_enum {
  TRITONSERVER_METRIC_PROMETHEUS
} TRITONSERVER_MetricFormat;

/// Delete a metrics object.
///
/// \param metrics The metrics object.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error* TRITONSERVER_MetricsDelete(
    struct TRITONSERVER_Metrics* metrics);

/// Get a buffer containing the metrics in the specified format. For
/// each format the buffer contains the following:
///
///   TRITONSERVER_METRIC_PROMETHEUS: 'base' points to a single multiline
///   string (char*) that gives a text representation of the metrics in
///   prometheus format. 'byte_size' returns the length of the string
///   in bytes.
///
/// The buffer is owned by the 'metrics' object and should not be
/// modified or freed by the caller. The lifetime of the buffer
/// extends only as long as 'metrics' and must not be accessed once
/// 'metrics' is deleted.
///
/// \param metrics The metrics object.
/// \param format The format to use for the returned metrics.
/// \param base Returns a pointer to the base of the formatted
/// metrics, as described above.
/// \param byte_size Returns the size, in bytes, of the formatted
/// metrics.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error* TRITONSERVER_MetricsFormatted(
    struct TRITONSERVER_Metrics* metrics, TRITONSERVER_MetricFormat format,
    const char** base, size_t* byte_size);

/// TRITONSERVER_InferenceTrace
///
/// Object that represents tracing for an inference request.
///

/// Trace levels. The trace level controls the type of trace
/// activities that are reported for an inference request.
///
/// Trace level values are power-of-2 and can be combined to trace
/// multiple types of activities. For example, use
/// (TRITONSERVER_TRACE_LEVEL_TIMESTAMPS |
/// TRITONSERVER_TRACE_LEVEL_TENSORS) to trace both timestamps and
/// tensors for an inference request.
///
/// TRITONSERVER_TRACE_LEVEL_MIN and TRITONSERVER_TRACE_LEVEL_MAX are
/// deprecated and should not be used.
typedef enum tritonserver_tracelevel_enum {
  /// Tracing disabled. No trace activities are reported.
  TRITONSERVER_TRACE_LEVEL_DISABLED = 0,
  /// Deprecated. Use TRITONSERVER_TRACE_LEVEL_TIMESTAMPS.
  TRITONSERVER_TRACE_LEVEL_MIN = 1,
  /// Deprecated. Use TRITONSERVER_TRACE_LEVEL_TIMESTAMPS.
  TRITONSERVER_TRACE_LEVEL_MAX = 2,
  /// Record timestamps for the inference request.
  TRITONSERVER_TRACE_LEVEL_TIMESTAMPS = 0x4,
  /// Record input and output tensor values for the inference request.
  TRITONSERVER_TRACE_LEVEL_TENSORS = 0x8
} TRITONSERVER_InferenceTraceLevel;

/// Get the string representation of a trace level. The returned
/// string is not owned by the caller and so should not be modified or
/// freed.
///
/// \param level The trace level.
/// \return The string representation of the trace level.
TRITONSERVER_DECLSPEC const char* TRITONSERVER_InferenceTraceLevelString(
    TRITONSERVER_InferenceTraceLevel level);

/// Trace activities
typedef enum tritonserver_traceactivity_enum {
  TRITONSERVER_TRACE_REQUEST_START = 0,
  TRITONSERVER_TRACE_QUEUE_START = 1,
  TRITONSERVER_TRACE_COMPUTE_START = 2,
  TRITONSERVER_TRACE_COMPUTE_INPUT_END = 3,
  TRITONSERVER_TRACE_COMPUTE_OUTPUT_START = 4,
  TRITONSERVER_TRACE_COMPUTE_END = 5,
  TRITONSERVER_TRACE_REQUEST_END = 6,
  TRITONSERVER_TRACE_TENSOR_QUEUE_INPUT = 7,
  TRITONSERVER_TRACE_TENSOR_BACKEND_INPUT = 8,
  TRITONSERVER_TRACE_TENSOR_BACKEND_OUTPUT = 9,
  TRITONSERVER_TRACE_CUSTOM_ACTIVITY = 10
} TRITONSERVER_InferenceTraceActivity;

/// Get the string representation of a trace activity. The returned
/// string is not owned by the caller and so should not be modified or
/// freed.
///
/// \param activity The trace activity.
/// \return The string representation of the trace activity.
TRITONSERVER_DECLSPEC const char* TRITONSERVER_InferenceTraceActivityString(
    TRITONSERVER_InferenceTraceActivity activity);

/// Type for trace timeline activity callback function. This callback function
/// is used to report activity occurring for a trace. This function
/// does not take ownership of 'trace' and so any information needed
/// from that object must be copied before returning. The 'userp' data
/// is the same as what is supplied in the call to
/// TRITONSERVER_InferenceTraceNew.
typedef void (*TRITONSERVER_InferenceTraceActivityFn_t)(
    struct TRITONSERVER_InferenceTrace* trace,
    TRITONSERVER_InferenceTraceActivity activity, uint64_t timestamp_ns,
    void* userp);

/// Type for trace tensor activity callback function. This callback function
/// is used to report tensor activity occurring for a trace. This function
/// does not take ownership of 'trace' and so any information needed
/// from that object must be copied before returning. The 'userp' data
/// is the same as what is supplied in the call to
/// TRITONSERVER_InferenceTraceTensorNew.
typedef void (*TRITONSERVER_InferenceTraceTensorActivityFn_t)(
    struct TRITONSERVER_InferenceTrace* trace,
    TRITONSERVER_InferenceTraceActivity activity, const char* name,
    TRITONSERVER_DataType datatype, const void* base, size_t byte_size,
    const int64_t* shape, uint64_t dim_count,
    TRITONSERVER_MemoryType memory_type, int64_t memory_type_id, void* userp);

/// Type for trace release callback function. This callback function
/// is called when all activity for the trace has completed. The
/// callback function takes ownership of the
/// TRITONSERVER_InferenceTrace object. The 'userp' data is the same
/// as what is supplied in the call to TRITONSERVER_InferenceTraceNew.
typedef void (*TRITONSERVER_InferenceTraceReleaseFn_t)(
    struct TRITONSERVER_InferenceTrace* trace, void* userp);

/// Create a new inference trace object. The caller takes ownership of
/// the TRITONSERVER_InferenceTrace object and must call
/// TRITONSERVER_InferenceTraceDelete to release the object.
///
/// The activity callback function will be called to report activity
/// for 'trace' as well as for any child traces that are spawned by
/// 'trace', and so the activity callback must check the trace object
/// to determine specifically what activity is being reported.
///
/// The release callback is called for both 'trace' and for any child
/// traces spawned by 'trace'.
///
/// \param trace Returns the new inference trace object.
/// \param level The tracing level.
/// \param parent_id The parent trace id for this trace. A value of 0
/// indicates that there is not parent trace.
/// \param activity_fn The callback function where activity for the
/// trace is reported.
/// \param release_fn The callback function called when all activity
/// is complete for the trace.
/// \param trace_userp User-provided pointer that is delivered to
/// the activity and release callback functions.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error* TRITONSERVER_InferenceTraceNew(
    struct TRITONSERVER_InferenceTrace** trace,
    TRITONSERVER_InferenceTraceLevel level, uint64_t parent_id,
    TRITONSERVER_InferenceTraceActivityFn_t activity_fn,
    TRITONSERVER_InferenceTraceReleaseFn_t release_fn, void* trace_userp);

/// Create a new inference trace object. The caller takes ownership of
/// the TRITONSERVER_InferenceTrace object and must call
/// TRITONSERVER_InferenceTraceDelete to release the object.
///
/// The timeline and tensor activity callback function will be called to report
/// activity for 'trace' as well as for any child traces that are spawned by
/// 'trace', and so the activity callback must check the trace object
/// to determine specifically what activity is being reported.
///
/// The release callback is called for both 'trace' and for any child
/// traces spawned by 'trace'.
///
/// \param trace Returns the new inference trace object.
/// \param level The tracing level.
/// \param parent_id The parent trace id for this trace. A value of 0
/// indicates that there is not parent trace.
/// \param activity_fn The callback function where timeline activity for the
/// trace is reported.
/// \param tensor_activity_fn The callback function where tensor activity for
/// the trace is reported.
/// \param release_fn The callback function called when all activity
/// is complete for the trace.
/// \param trace_userp User-provided pointer that is delivered to
/// the activity and release callback functions.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_InferenceTraceTensorNew(
    struct TRITONSERVER_InferenceTrace** trace,
    TRITONSERVER_InferenceTraceLevel level, uint64_t parent_id,
    TRITONSERVER_InferenceTraceActivityFn_t activity_fn,
    TRITONSERVER_InferenceTraceTensorActivityFn_t tensor_activity_fn,
    TRITONSERVER_InferenceTraceReleaseFn_t release_fn, void* trace_userp);

/// Report a trace activity. All the traces reported using this API will be
/// using TRITONSERVER_TRACE_CUSTOM_ACTIVITY type.
///
/// \param trace The trace object.
/// \param timestamp The timestamp associated with the trace activity.
/// \param name The trace activity name.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_InferenceTraceReportActivity(
    TRITONSERVER_InferenceTrace* trace, uint64_t timestamp,
    const char* activity_name);

/// Delete a trace object.
///
/// \param trace The trace object.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_InferenceTraceDelete(struct TRITONSERVER_InferenceTrace* trace);

/// Get the id associated with a trace. Every trace is assigned an id
/// that is unique across all traces created for a Triton server.
///
/// \param trace The trace.
/// \param id Returns the id associated with the trace.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error* TRITONSERVER_InferenceTraceId(
    struct TRITONSERVER_InferenceTrace* trace, uint64_t* id);

/// Get the parent id associated with a trace. The parent id indicates
/// a parent-child relationship between two traces. A parent id value
/// of 0 indicates that there is no parent trace.
///
/// \param trace The trace.
/// \param id Returns the parent id associated with the trace.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_InferenceTraceParentId(
    struct TRITONSERVER_InferenceTrace* trace, uint64_t* parent_id);

/// Get the name of the model associated with a trace. The caller does
/// not own the returned string and must not modify or delete it. The
/// lifetime of the returned string extends only as long as 'trace'.
///
/// \param trace The trace.
/// \param model_name Returns the name of the model associated with
/// the trace.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_InferenceTraceModelName(
    struct TRITONSERVER_InferenceTrace* trace, const char** model_name);

/// Get the version of the model associated with a trace.
///
/// \param trace The trace.
/// \param model_version Returns the version of the model associated
/// with the trace.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_InferenceTraceModelVersion(
    struct TRITONSERVER_InferenceTrace* trace, int64_t* model_version);

/// Get the request id associated with a trace. The caller does
/// not own the returned string and must not modify or delete it. The
/// lifetime of the returned string extends only as long as 'trace'.
///
/// \param trace The trace.
/// \param request_id Returns the version of the model associated
/// with the trace.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_InferenceTraceRequestId(
    struct TRITONSERVER_InferenceTrace* trace, const char** request_id);

/// Get the child trace, spawned from the parent trace. The caller owns
/// the returned object and must call TRITONSERVER_InferenceTraceDelete
/// to release the object, unless ownership is transferred through
/// other APIs (see TRITONSERVER_ServerInferAsync).
///
/// \param trace The trace.
/// \param child_trace Returns the child trace, spawned from the trace.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_InferenceTraceSpawnChildTrace(
    struct TRITONSERVER_InferenceTrace* trace,
    struct TRITONSERVER_InferenceTrace** child_trace);

/// Set TRITONSERVER_InferenceTrace context.
///
/// \param trace The trace.
/// \param trace_context A new trace context to associate with the trace.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_InferenceTraceSetContext(
    struct TRITONSERVER_InferenceTrace* trace, const char* trace_context);

/// Get TRITONSERVER_InferenceTrace context.
///
/// \param trace The trace.
/// \param trace_context Returns the context associated with the trace.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_InferenceTraceContext(
    struct TRITONSERVER_InferenceTrace* trace, const char** trace_context);

/// TRITONSERVER_InferenceRequest
///
/// Object representing an inference request. The inference request
/// provides the meta-data and input tensor values needed for an
/// inference and returns the inference result meta-data and output
/// tensors. An inference request object can be modified and reused
/// multiple times.
///

/// Inference request flags. The enum values must be power-of-2 values.
typedef enum tritonserver_requestflag_enum {
  TRITONSERVER_REQUEST_FLAG_SEQUENCE_START = 1,
  TRITONSERVER_REQUEST_FLAG_SEQUENCE_END = 2
} TRITONSERVER_RequestFlag;

/// Inference request release flags. The enum values must be
/// power-of-2 values.
typedef enum tritonserver_requestreleaseflag_enum {
  TRITONSERVER_REQUEST_RELEASE_ALL = 1,
  TRITONSERVER_REQUEST_RELEASE_RESCHEDULE = 2
} TRITONSERVER_RequestReleaseFlag;

/// Inference response complete flags. The enum values must be
/// power-of-2 values.
typedef enum tritonserver_responsecompleteflag_enum {
  TRITONSERVER_RESPONSE_COMPLETE_FINAL = 1
} TRITONSERVER_ResponseCompleteFlag;

/// Type for inference request release callback function. The callback
/// indicates what type of release is being performed on the request
/// and for some of these the callback function takes ownership of the
/// TRITONSERVER_InferenceRequest object. The 'userp' data is the data
/// provided as 'request_release_userp' in the call to
/// TRITONSERVER_InferenceRequestSetReleaseCallback.
///
/// One or more flags will be specified when the callback is invoked,
/// and the callback must take the following actions:
///
///   - TRITONSERVER_REQUEST_RELEASE_ALL: The entire inference request
///     is being released and ownership is passed to the callback
///     function. Triton will not longer access the 'request' object
///     itself nor any input tensor data associated with the
///     request. The callback should free or otherwise manage the
///     'request' object and all associated tensor data.
///   - TRITONSERVER_REQUEST_RELEASE_RESCHEDULE: This flag is currently being
///     consumed internally and the callback is not expected to receive nor
///     process this kind of release. The backend will call
///     TRITONBACKEND_RequestRelease with this flag when it wishes to reschedule
///     the request back to the model. An example is that the model is
///     recursively performing inference of the request and use the rescheduling
///     to proceed the recursive execution.
///
/// Note that currently TRITONSERVER_REQUEST_RELEASE_ALL should always
/// be set when the callback is invoked but in the future that may
/// change, so the callback should explicitly check for the flag
/// before taking ownership of the request object.
///
typedef void (*TRITONSERVER_InferenceRequestReleaseFn_t)(
    struct TRITONSERVER_InferenceRequest* request, const uint32_t flags,
    void* userp);

/// Type for callback function indicating that an inference response
/// has completed. The callback function takes ownership of the
/// TRITONSERVER_InferenceResponse object. The 'userp' data is the
/// data provided as 'response_userp' in the call to
/// TRITONSERVER_InferenceRequestSetResponseCallback.
///
/// One or more flags may be specified when the callback is invoked:
///
///   - TRITONSERVER_RESPONSE_COMPLETE_FINAL: Indicates that no more
///     responses will be generated for a given request (more
///     specifically, that no more responses will be generated for the
///     inference request that set this callback and 'userp'). When
///     this flag is set 'response' may be a response object or may be
///     nullptr. If 'response' is not nullptr, then 'response' is the
///     last response that Triton will produce for the request. If
///     'response' is nullptr then Triton is indicating that no more
///     responses will be produced for the request.
typedef void (*TRITONSERVER_InferenceResponseCompleteFn_t)(
    struct TRITONSERVER_InferenceResponse* response, const uint32_t flags,
    void* userp);

/// Create a new inference request object.
///
/// \param inference_request Returns the new request object.
/// \param server the inference server object.
/// \param model_name The name of the model to use for the request.
/// \param model_version The version of the model to use for the
/// request. If -1 then the server will choose a version based on the
/// model's policy.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_InferenceRequestNew(
    struct TRITONSERVER_InferenceRequest** inference_request,
    struct TRITONSERVER_Server* server, const char* model_name,
    const int64_t model_version);

/// Delete an inference request object.
///
/// \param inference_request The request object.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_InferenceRequestDelete(
    struct TRITONSERVER_InferenceRequest* inference_request);

/// Get the ID for a request. The returned ID is owned by
/// 'inference_request' and must not be modified or freed by the
/// caller.
///
/// \param inference_request The request object.
/// \param id Returns the ID.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_InferenceRequestId(
    struct TRITONSERVER_InferenceRequest* inference_request, const char** id);

/// Set the ID for a request.
///
/// \param inference_request The request object.
/// \param id The ID.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_InferenceRequestSetId(
    struct TRITONSERVER_InferenceRequest* inference_request, const char* id);

/// Get the flag(s) associated with a request. On return 'flags' holds
/// a bitwise-or of all flag values, see TRITONSERVER_RequestFlag for
/// available flags.
///
/// \param inference_request The request object.
/// \param flags Returns the flags.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_InferenceRequestFlags(
    struct TRITONSERVER_InferenceRequest* inference_request, uint32_t* flags);

/// Set the flag(s) associated with a request. 'flags' should hold a
/// bitwise-or of all flag values, see TRITONSERVER_RequestFlag for
/// available flags.
///
/// \param inference_request The request object.
/// \param flags The flags.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_InferenceRequestSetFlags(
    struct TRITONSERVER_InferenceRequest* inference_request, uint32_t flags);

/// Get the correlation ID of the inference request as an unsigned integer.
/// Default is 0, which indicates that the request has no correlation ID.
/// If the correlation id associated with the inference request is a string,
/// this function will return a failure. The correlation ID is used
/// to indicate two or more inference request are related to each other.
/// How this relationship is handled by the inference server is determined by
/// the model's scheduling policy.
///
/// \param inference_request The request object.
/// \param correlation_id Returns the correlation ID.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_InferenceRequestCorrelationId(
    struct TRITONSERVER_InferenceRequest* inference_request,
    uint64_t* correlation_id);

/// Get the correlation ID of the inference request as a string.
/// Default is empty "", which indicates that the request has no correlation ID.
/// If the correlation id associated with the inference request is an unsigned
/// integer, then this function will return a failure. The correlation ID
/// is used to indicate two or more inference request are related to each other.
/// How this relationship is handled by the inference server is determined by
/// the model's scheduling policy.
///
/// \param inference_request The request object.
/// \param correlation_id Returns the correlation ID.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_InferenceRequestCorrelationIdString(
    struct TRITONSERVER_InferenceRequest* inference_request,
    const char** correlation_id);

/// Set the correlation ID of the inference request to be an unsigned integer.
/// Default is 0, which indicates that the request has no correlation ID.
/// The correlation ID is used to indicate two or more inference request
/// are related to each other. How this relationship is handled by the
/// inference server is determined by the model's scheduling policy.
///
/// \param inference_request The request object.
/// \param correlation_id The correlation ID.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_InferenceRequestSetCorrelationId(
    struct TRITONSERVER_InferenceRequest* inference_request,
    uint64_t correlation_id);

/// Set the correlation ID of the inference request to be a string.
/// The correlation ID is used to indicate two or more inference
/// request are related to each other. How this relationship is
/// handled by the inference server is determined by the model's
/// scheduling policy.
///
/// \param inference_request The request object.
/// \param correlation_id The correlation ID.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_InferenceRequestSetCorrelationIdString(
    struct TRITONSERVER_InferenceRequest* inference_request,
    const char* correlation_id);

/// Cancel an inference request. Requests are canceled on a best
/// effort basis and no guarantee is provided that cancelling a
/// request will result in early termination. Note that the
/// inference request cancellation status will be reset after
/// TRITONSERVER_InferAsync is run. This means that if you cancel
/// the request before calling TRITONSERVER_InferAsync
/// the request will not be cancelled.
///
/// \param inference_request The request object.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_InferenceRequestCancel(
    struct TRITONSERVER_InferenceRequest* inference_request);

/// Query whether the request is cancelled or not.
///
/// If possible the backend should terminate any processing and
/// send an error response with cancelled status.
///
/// \param inference_request The request object.
/// \param is_cancelled Returns whether the inference request is cancelled or
/// not.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_InferenceRequestIsCancelled(
    struct TRITONSERVER_InferenceRequest* inference_request,
    bool* is_cancelled);

/// Deprecated. See TRITONSERVER_InferenceRequestPriorityUInt64 instead.
///
/// Get the priority for a request. The default is 0 indicating that
/// the request does not specify a priority and so will use the
/// model's default priority.
///
/// \param inference_request The request object.
/// \param priority Returns the priority level.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_InferenceRequestPriority(
    struct TRITONSERVER_InferenceRequest* inference_request,
    uint32_t* priority);

/// Get the priority for a request. The default is 0 indicating that
/// the request does not specify a priority and so will use the
/// model's default priority.
///
/// \param inference_request The request object.
/// \param priority Returns the priority level.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_InferenceRequestPriorityUInt64(
    struct TRITONSERVER_InferenceRequest* inference_request,
    uint64_t* priority);

/// Deprecated. See TRITONSERVER_InferenceRequestSetPriorityUInt64 instead.
///
/// Set the priority for a request. The default is 0 indicating that
/// the request does not specify a priority and so will use the
/// model's default priority.
///
/// \param inference_request The request object.
/// \param priority The priority level.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_InferenceRequestSetPriority(
    struct TRITONSERVER_InferenceRequest* inference_request, uint32_t priority);

/// Set the priority for a request. The default is 0 indicating that
/// the request does not specify a priority and so will use the
/// model's default priority.
///
/// \param inference_request The request object.
/// \param priority The priority level.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_InferenceRequestSetPriorityUInt64(
    struct TRITONSERVER_InferenceRequest* inference_request, uint64_t priority);

/// Get the timeout for a request, in microseconds. The default is 0
/// which indicates that the request has no timeout.
///
/// \param inference_request The request object.
/// \param timeout_us Returns the timeout, in microseconds.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_InferenceRequestTimeoutMicroseconds(
    struct TRITONSERVER_InferenceRequest* inference_request,
    uint64_t* timeout_us);

/// Set the timeout for a request, in microseconds. The default is 0
/// which indicates that the request has no timeout.
///
/// \param inference_request The request object.
/// \param timeout_us The timeout, in microseconds.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_InferenceRequestSetTimeoutMicroseconds(
    struct TRITONSERVER_InferenceRequest* inference_request,
    uint64_t timeout_us);

/// Add an input to a request.
///
/// \param inference_request The request object.
/// \param name The name of the input.
/// \param datatype The type of the input. Valid type names are BOOL,
/// UINT8, UINT16, UINT32, UINT64, INT8, INT16, INT32, INT64, FP16,
/// FP32, FP64, and BYTES.
/// \param shape The shape of the input.
/// \param dim_count The number of dimensions of 'shape'.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_InferenceRequestAddInput(
    struct TRITONSERVER_InferenceRequest* inference_request, const char* name,
    const TRITONSERVER_DataType datatype, const int64_t* shape,
    uint64_t dim_count);

/// Add a raw input to a request. The name recognized by the model, data type
/// and shape of the input will be deduced from model configuration.
/// This function must be called at most once on request with no other input to
/// ensure the deduction is accurate.
///
/// \param inference_request The request object.
/// \param name The name of the input. This name is only used as a reference
/// of the raw input in other Tritonserver APIs. It doesn't associate with the
/// name used in the model.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_InferenceRequestAddRawInput(
    struct TRITONSERVER_InferenceRequest* inference_request, const char* name);

/// Remove an input from a request.
///
/// \param inference_request The request object.
/// \param name The name of the input.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_InferenceRequestRemoveInput(
    struct TRITONSERVER_InferenceRequest* inference_request, const char* name);

/// Remove all inputs from a request.
///
/// \param inference_request The request object.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_InferenceRequestRemoveAllInputs(
    struct TRITONSERVER_InferenceRequest* inference_request);

/// Assign a buffer of data to an input. The buffer will be appended
/// to any existing buffers for that input. The 'inference_request'
/// object takes ownership of the buffer and so the caller should not
/// modify or free the buffer until that ownership is released by
/// 'inference_request' being deleted or by the input being removed
/// from 'inference_request'.
///
/// \param inference_request The request object.
/// \param name The name of the input.
/// \param base The base address of the input data.
/// \param byte_size The size, in bytes, of the input data.
/// \param memory_type The memory type of the input data.
/// \param memory_type_id The memory type id of the input data.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_InferenceRequestAppendInputData(
    struct TRITONSERVER_InferenceRequest* inference_request, const char* name,
    const void* base, size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id);

/// Assign a buffer of data to an input for execution on all model instances
/// with the specified host policy. The buffer will be appended to any existing
/// buffers for that input on all devices with this host policy. The
/// 'inference_request' object takes ownership of the buffer and so the caller
/// should not modify or free the buffer until that ownership is released by
/// 'inference_request' being deleted or by the input being removed from
/// 'inference_request'. If the execution is scheduled on a device that does not
/// have a input buffer specified using this function, then the input buffer
/// specified with TRITONSERVER_InferenceRequestAppendInputData will be used so
/// a non-host policy specific version of data must be added using that API.
/// \param inference_request The request object.
/// \param name The name of the input.
/// \param base The base address of the input data.
/// \param byte_size The size, in bytes, of the input data.
/// \param memory_type The memory type of the input data.
/// \param memory_type_id The memory type id of the input data.
/// \param host_policy_name All model instances executing with this host_policy
/// will use this input buffer for execution.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_InferenceRequestAppendInputDataWithHostPolicy(
    struct TRITONSERVER_InferenceRequest* inference_request, const char* name,
    const void* base, size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id, const char* host_policy_name);

/// Assign a buffer of data to an input. The buffer will be appended
/// to any existing buffers for that input. The 'inference_request'
/// object takes ownership of the buffer and so the caller should not
/// modify or free the buffer until that ownership is released by
/// 'inference_request' being deleted or by the input being removed
/// from 'inference_request'.
///
/// \param inference_request The request object.
/// \param name The name of the input.
/// \param base The base address of the input data.
/// \param buffer_attributes The buffer attributes of the input.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_InferenceRequestAppendInputDataWithBufferAttributes(
    struct TRITONSERVER_InferenceRequest* inference_request, const char* name,
    const void* base, struct TRITONSERVER_BufferAttributes* buffer_attributes);

/// Clear all input data from an input, releasing ownership of the
/// buffer(s) that were appended to the input with
/// TRITONSERVER_InferenceRequestAppendInputData or
/// TRITONSERVER_InferenceRequestAppendInputDataWithHostPolicy
/// \param inference_request The request object.
/// \param name The name of the input.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_InferenceRequestRemoveAllInputData(
    struct TRITONSERVER_InferenceRequest* inference_request, const char* name);

/// Add an output request to an inference request.
///
/// \param inference_request The request object.
/// \param name The name of the output.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_InferenceRequestAddRequestedOutput(
    struct TRITONSERVER_InferenceRequest* inference_request, const char* name);

/// Remove an output request from an inference request.
///
/// \param inference_request The request object.
/// \param name The name of the output.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_InferenceRequestRemoveRequestedOutput(
    struct TRITONSERVER_InferenceRequest* inference_request, const char* name);

/// Remove all output requests from an inference request.
///
/// \param inference_request The request object.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_InferenceRequestRemoveAllRequestedOutputs(
    struct TRITONSERVER_InferenceRequest* inference_request);

/// Set the release callback for an inference request. The release
/// callback is called by Triton to return ownership of the request
/// object.
///
/// \param inference_request The request object.
/// \param request_release_fn The function called to return ownership
/// of the 'inference_request' object.
/// \param request_release_userp User-provided pointer that is
/// delivered to the 'request_release_fn' callback.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_InferenceRequestSetReleaseCallback(
    struct TRITONSERVER_InferenceRequest* inference_request,
    TRITONSERVER_InferenceRequestReleaseFn_t request_release_fn,
    void* request_release_userp);

/// Set the allocator and response callback for an inference
/// request. The allocator is used to allocate buffers for any output
/// tensors included in responses that are produced for this
/// request. The response callback is called to return response
/// objects representing responses produced for this request.
/// Typically 'response_allocator_userp' and 'response_userp' will no
/// longer be referenced after 'response_fn' is invoked with
/// 'TRITONSERVER_RESPONSE_COMPLETE_FINAL' flag, therefore the user may
/// release 'response_allocator_userp' and 'response_userp' at that point.
///
/// \param inference_request The request object.
/// \param response_allocator The TRITONSERVER_ResponseAllocator to use
/// to allocate buffers to hold inference results.
/// \param response_allocator_userp User-provided pointer that is
/// delivered to the response allocator's start and allocation functions.
/// \param response_fn The function called to deliver an inference
/// response for this request.
/// \param response_userp User-provided pointer that is delivered to
/// the 'response_fn' callback.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_InferenceRequestSetResponseCallback(
    struct TRITONSERVER_InferenceRequest* inference_request,
    struct TRITONSERVER_ResponseAllocator* response_allocator,
    void* response_allocator_userp,
    TRITONSERVER_InferenceResponseCompleteFn_t response_fn,
    void* response_userp);

/// Set a string parameter in the request.
///
/// \param request The request.
/// \param key The name of the parameter.
/// \param value The value of the parameter.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_InferenceRequestSetStringParameter(
    struct TRITONSERVER_InferenceRequest* request, const char* key,
    const char* value);

/// Set an integer parameter in the request.
///
/// \param request The request.
/// \param key The name of the parameter.
/// \param value The value of the parameter.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_InferenceRequestSetIntParameter(
    struct TRITONSERVER_InferenceRequest* request, const char* key,
    const int64_t value);

/// Set a boolean parameter in the request.
///
/// \param request The request.
/// \param key The name of the parameter.
/// \param value The value of the parameter.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_InferenceRequestSetBoolParameter(
    struct TRITONSERVER_InferenceRequest* request, const char* key,
    const bool value);

/// Set a double parameter in the request.
///
/// \param request The request.
/// \param key The name of the parameter.
/// \param value The value of the parameter.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_InferenceRequestSetDoubleParameter(
    struct TRITONSERVER_InferenceRequest* request, const char* key,
    const double value);

/// TRITONSERVER_InferenceResponse
///
/// Object representing an inference response. The inference response
/// provides the meta-data and output tensor values calculated by the
/// inference.
///

/// Delete an inference response object.
///
/// \param inference_response The response object.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_InferenceResponseDelete(
    struct TRITONSERVER_InferenceResponse* inference_response);

/// Return the error status of an inference response. Return a
/// TRITONSERVER_Error object on failure, return nullptr on success.
/// The returned error object is owned by 'inference_response' and so
/// should not be deleted by the caller.
///
/// \param inference_response The response object.
/// \return a TRITONSERVER_Error indicating the success or failure
/// status of the response.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_InferenceResponseError(
    struct TRITONSERVER_InferenceResponse* inference_response);

/// Get model used to produce a response. The caller does not own the
/// returned model name value and must not modify or delete it. The
/// lifetime of all returned values extends until 'inference_response'
/// is deleted.
///
/// \param inference_response The response object.
/// \param model_name Returns the name of the model.
/// \param model_version Returns the version of the model.
/// this response.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_InferenceResponseModel(
    struct TRITONSERVER_InferenceResponse* inference_response,
    const char** model_name, int64_t* model_version);

/// Get the ID of the request corresponding to a response. The caller
/// does not own the returned ID and must not modify or delete it. The
/// lifetime of all returned values extends until 'inference_response'
/// is deleted.
///
/// \param inference_response The response object.
/// \param request_id Returns the ID of the request corresponding to
/// this response.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_InferenceResponseId(
    struct TRITONSERVER_InferenceResponse* inference_response,
    const char** request_id);

/// Get the number of parameters available in the response.
///
/// \param inference_response The response object.
/// \param count Returns the number of parameters.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_InferenceResponseParameterCount(
    struct TRITONSERVER_InferenceResponse* inference_response, uint32_t* count);

/// Get all information about a parameter. The caller does not own any
/// of the returned values and must not modify or delete them. The
/// lifetime of all returned values extends until 'inference_response'
/// is deleted.
///
/// The 'vvalue' returns a void* pointer that must be cast
/// appropriately based on 'type'. For example:
///
///   void* vvalue;
///   TRITONSERVER_ParameterType type;
///   TRITONSERVER_InferenceResponseParameter(
///                     response, index, &name, &type, &vvalue);
///   switch (type) {
///     case TRITONSERVER_PARAMETER_BOOL:
///       bool value = *(reinterpret_cast<bool*>(vvalue));
///       ...
///     case TRITONSERVER_PARAMETER_INT:
///       int64_t value = *(reinterpret_cast<int64_t*>(vvalue));
///       ...
///     case TRITONSERVER_PARAMETER_STRING:
///       const char* value = reinterpret_cast<const char*>(vvalue);
///       ...
///
/// \param inference_response The response object.
/// \param index The index of the parameter, must be 0 <= index <
/// count, where 'count' is the value returned by
/// TRITONSERVER_InferenceResponseParameterCount.
/// \param name Returns the name of the parameter.
/// \param type Returns the type of the parameter.
/// \param vvalue Returns a pointer to the parameter value.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_InferenceResponseParameter(
    struct TRITONSERVER_InferenceResponse* inference_response,
    const uint32_t index, const char** name, TRITONSERVER_ParameterType* type,
    const void** vvalue);

/// Get the number of outputs available in the response.
///
/// \param inference_response The response object.
/// \param count Returns the number of output tensors.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_InferenceResponseOutputCount(
    struct TRITONSERVER_InferenceResponse* inference_response, uint32_t* count);

/// Get all information about an output tensor.  The tensor data is
/// returned as the base pointer to the data and the size, in bytes,
/// of the data. The caller does not own any of the returned values
/// and must not modify or delete them. The lifetime of all returned
/// values extends until 'inference_response' is deleted.
///
/// \param inference_response The response object.
/// \param index The index of the output tensor, must be 0 <= index <
/// count, where 'count' is the value returned by
/// TRITONSERVER_InferenceResponseOutputCount.
/// \param name Returns the name of the output.
/// \param datatype Returns the type of the output.
/// \param shape Returns the shape of the output.
/// \param dim_count Returns the number of dimensions of the returned
/// shape.
/// \param base Returns the tensor data for the output.
/// \param byte_size Returns the size, in bytes, of the data.
/// \param memory_type Returns the memory type of the data.
/// \param memory_type_id Returns the memory type id of the data.
/// \param userp The user-specified value associated with the buffer
/// in TRITONSERVER_ResponseAllocatorAllocFn_t.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_InferenceResponseOutput(
    struct TRITONSERVER_InferenceResponse* inference_response,
    const uint32_t index, const char** name, TRITONSERVER_DataType* datatype,
    const int64_t** shape, uint64_t* dim_count, const void** base,
    size_t* byte_size, TRITONSERVER_MemoryType* memory_type,
    int64_t* memory_type_id, void** userp);

/// Get a classification label associated with an output for a given
/// index.  The caller does not own the returned label and must not
/// modify or delete it. The lifetime of all returned label extends
/// until 'inference_response' is deleted.
///
/// \param inference_response The response object.
/// \param index The index of the output tensor, must be 0 <= index <
/// count, where 'count' is the value returned by
/// TRITONSERVER_InferenceResponseOutputCount.
/// \param class_index The index of the class.
/// \param name Returns the label corresponding to 'class_index' or
/// nullptr if no label.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_InferenceResponseOutputClassificationLabel(
    struct TRITONSERVER_InferenceResponse* inference_response,
    const uint32_t index, const size_t class_index, const char** label);

/// TRITONSERVER_BufferAttributes
///
/// API to create, modify, or retrieve attributes associated with a buffer.
///

/// Create a new buffer attributes object. The caller takes ownership of
/// the TRITONSERVER_BufferAttributes object and must call
/// TRITONSERVER_BufferAttributesDelete to release the object.
///
/// \param buffer_attributes Returns the new buffer attributes object.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_BufferAttributesNew(
    struct TRITONSERVER_BufferAttributes** buffer_attributes);

/// Delete a buffer attributes object.
///
/// \param buffer_attributes The buffer_attributes object.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_BufferAttributesDelete(
    struct TRITONSERVER_BufferAttributes* buffer_attributes);

/// Set the memory type id field of the buffer attributes.
///
/// \param buffer_attributes The buffer attributes object.
/// \param memory_type_id Memory type id to assign to the buffer attributes
/// object.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_BufferAttributesSetMemoryTypeId(
    struct TRITONSERVER_BufferAttributes* buffer_attributes,
    int64_t memory_type_id);

/// Set the memory type field of the buffer attributes.
///
/// \param buffer_attributes The buffer attributes object.
/// \param memory_type Memory type to assign to the buffer attributes object.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_BufferAttributesSetMemoryType(
    struct TRITONSERVER_BufferAttributes* buffer_attributes,
    TRITONSERVER_MemoryType memory_type);

/// Set the CudaIpcHandle field of the buffer attributes.
///
/// \param buffer_attributes The buffer attributes object.
/// \param cuda_ipc_handle The CudaIpcHandle to assign to the buffer attributes
/// object.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_BufferAttributesSetCudaIpcHandle(
    struct TRITONSERVER_BufferAttributes* buffer_attributes,
    void* cuda_ipc_handle);

/// Set the byte size field of the buffer attributes.
///
/// \param buffer_attributes The buffer attributes object.
/// \param byte_size Byte size to assign to the buffer attributes object.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_BufferAttributesSetByteSize(
    struct TRITONSERVER_BufferAttributes* buffer_attributes, size_t byte_size);

/// Get the memory type id field of the buffer attributes.
///
/// \param buffer_attributes The buffer attributes object.
/// \param memory_type_id Returns the memory type id associated with the buffer
/// attributes object.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_BufferAttributesMemoryTypeId(
    struct TRITONSERVER_BufferAttributes* buffer_attributes,
    int64_t* memory_type_id);

/// Get the memory type field of the buffer attributes.
///
/// \param buffer_attributes The buffer attributes object.
/// \param memory_type Returns the memory type associated with the buffer
/// attributes object.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_BufferAttributesMemoryType(
    struct TRITONSERVER_BufferAttributes* buffer_attributes,
    TRITONSERVER_MemoryType* memory_type);

/// Get the CudaIpcHandle field of the buffer attributes object.
///
/// \param buffer_attributes The buffer attributes object.
/// \param cuda_ipc_handle Returns the memory type associated with the buffer
/// attributes object. If the cudaIpcHandle does not exist for the buffer,
/// nullptr will be returned.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_BufferAttributesCudaIpcHandle(
    struct TRITONSERVER_BufferAttributes* buffer_attributes,
    void** cuda_ipc_handle);

/// Get the byte size field of the buffer attributes.
///
/// \param buffer_attributes The buffer attributes object.
/// \param byte_size Returns the byte size associated with the buffer attributes
/// object.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_BufferAttributesByteSize(
    struct TRITONSERVER_BufferAttributes* buffer_attributes, size_t* byte_size);


/// TRITONSERVER_ServerOptions
///
/// Options to use when creating an inference server.
///

/// Model control modes
typedef enum tritonserver_modelcontrolmode_enum {
  TRITONSERVER_MODEL_CONTROL_NONE,
  TRITONSERVER_MODEL_CONTROL_POLL,
  TRITONSERVER_MODEL_CONTROL_EXPLICIT
} TRITONSERVER_ModelControlMode;

/// Rate limit modes
typedef enum tritonserver_ratelimitmode_enum {
  TRITONSERVER_RATE_LIMIT_OFF,
  TRITONSERVER_RATE_LIMIT_EXEC_COUNT
} TRITONSERVER_RateLimitMode;

/// Create a new server options object. The caller takes ownership of
/// the TRITONSERVER_ServerOptions object and must call
/// TRITONSERVER_ServerOptionsDelete to release the object.
///
/// \param options Returns the new server options object.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error* TRITONSERVER_ServerOptionsNew(
    struct TRITONSERVER_ServerOptions** options);

/// Delete a server options object.
///
/// \param options The server options object.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_ServerOptionsDelete(struct TRITONSERVER_ServerOptions* options);

/// Set the textual ID for the server in a server options. The ID is a
/// name that identifies the server.
///
/// \param options The server options object.
/// \param server_id The server identifier.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetServerId(
    struct TRITONSERVER_ServerOptions* options, const char* server_id);

/// Set the model repository path in a server options. The path must be
/// the full absolute path to the model repository. This function can be called
/// multiple times with different paths to set multiple model repositories.
/// Note that if a model is not unique across all model repositories
/// at any time, the model will not be available.
///
/// \param options The server options object.
/// \param model_repository_path The full path to the model repository.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetModelRepositoryPath(
    struct TRITONSERVER_ServerOptions* options,
    const char* model_repository_path);

/// Set the model control mode in a server options. For each mode the models
/// will be managed as the following:
///
///   TRITONSERVER_MODEL_CONTROL_NONE: the models in model repository will be
///   loaded on startup. After startup any changes to the model repository will
///   be ignored. Calling TRITONSERVER_ServerPollModelRepository will result in
///   an error.
///
///   TRITONSERVER_MODEL_CONTROL_POLL: the models in model repository will be
///   loaded on startup. The model repository can be polled periodically using
///   TRITONSERVER_ServerPollModelRepository and the server will load, unload,
///   and updated models according to changes in the model repository.
///
///   TRITONSERVER_MODEL_CONTROL_EXPLICIT: the models in model repository will
///   not be loaded on startup. The corresponding model control APIs must be
///   called to load / unload a model in the model repository.
///
/// \param options The server options object.
/// \param mode The mode to use for the model control.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetModelControlMode(
    struct TRITONSERVER_ServerOptions* options,
    TRITONSERVER_ModelControlMode mode);

/// Set the model to be loaded at startup in a server options. The model must be
/// present in one, and only one, of the specified model repositories.
/// This function can be called multiple times with different model name
/// to set multiple startup models.
/// Note that it only takes affect on TRITONSERVER_MODEL_CONTROL_EXPLICIT mode.
///
/// \param options The server options object.
/// \param mode_name The name of the model to load on startup.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetStartupModel(
    struct TRITONSERVER_ServerOptions* options, const char* model_name);

/// Enable or disable strict model configuration handling in a server
/// options.
///
/// \param options The server options object.
/// \param strict True to enable strict model configuration handling,
/// false to disable.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetStrictModelConfig(
    struct TRITONSERVER_ServerOptions* options, bool strict);

/// Set the custom model configuration name to load for all models.
/// Fall back to default config file if empty.
///
/// \param options The server options object.
/// \param config_name The name of the config file to load for all models.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetModelConfigName(
    struct TRITONSERVER_ServerOptions* options, const char* model_config_name);

/// Set the rate limit mode in a server options.
///
///   TRITONSERVER_RATE_LIMIT_EXEC_COUNT: The rate limiting prioritizes the
///   inference execution using the number of times each instance has got a
///   chance to run. The execution gets to run only when its resource
///   constraints are satisfied.
///
///   TRITONSERVER_RATE_LIMIT_OFF: The rate limiting is turned off and the
///   inference gets executed whenever an instance is available.
///
/// \param options The server options object.
/// \param mode The mode to use for the rate limiting. By default, execution
/// count is used to determine the priorities.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetRateLimiterMode(
    struct TRITONSERVER_ServerOptions* options,
    TRITONSERVER_RateLimitMode mode);

/// Add resource count for rate limiting.
///
/// \param options The server options object.
/// \param name The name of the resource.
/// \param count The count of the resource.
/// \param device The device identifier for the resource. A value of -1
/// indicates that the specified number of resources are available on every
/// device. The device value is ignored for a global resource. The server
/// will use the rate limiter configuration specified for instance groups
/// in model config to determine whether resource is global. In case of
/// conflicting resource type in different model configurations, server
/// will raise an appropriate error while loading model.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_ServerOptionsAddRateLimiterResource(
    struct TRITONSERVER_ServerOptions* options, const char* resource_name,
    const size_t resource_count, const int device);

/// Set the total pinned memory byte size that the server can allocate
/// in a server options. The pinned memory pool will be shared across
/// Triton itself and the backends that use
/// TRITONBACKEND_MemoryManager to allocate memory.
///
/// \param options The server options object.
/// \param size The pinned memory pool byte size.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetPinnedMemoryPoolByteSize(
    struct TRITONSERVER_ServerOptions* options, uint64_t size);

/// Set the total CUDA memory byte size that the server can allocate
/// on given GPU device in a server options. The pinned memory pool
/// will be shared across Triton itself and the backends that use
/// TRITONBACKEND_MemoryManager to allocate memory.
///
/// \param options The server options object.
/// \param gpu_device The GPU device to allocate the memory pool.
/// \param size The CUDA memory pool byte size.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetCudaMemoryPoolByteSize(
    struct TRITONSERVER_ServerOptions* options, int gpu_device, uint64_t size);

/// Set the size of the virtual address space that will be used
/// for growable memory in implicit state.
///
/// \param options The server options object.
/// \param gpu_device The GPU device to set the CUDA virtual address space size
/// \param size The size of the CUDA virtual address space.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetCudaVirtualAddressSize(
    TRITONSERVER_ServerOptions* options, int gpu_device,
    size_t cuda_virtual_address_size);

/// Deprecated. See TRITONSERVER_ServerOptionsSetCacheConfig instead.
///
/// Set the total response cache byte size that the server can allocate in CPU
/// memory. The response cache will be shared across all inference requests and
/// across all models.
///
/// \param options The server options object.
/// \param size The total response cache byte size.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetResponseCacheByteSize(
    struct TRITONSERVER_ServerOptions* options, uint64_t size);

/// Set the cache config that will be used to initialize the cache
/// implementation for "cache_name".
///
/// It is expected that the "cache_name" provided matches a directory inside
/// the "cache_dir" used for TRITONSERVER_ServerOptionsSetCacheDirectory. The
/// default "cache_dir" is "/opt/tritonserver/caches", so for a "cache_name" of
/// "local", Triton would expect to find the "local" cache implementation at
/// "/opt/tritonserver/caches/local/libtritoncache_local.so"
///
/// Altogether an example for the "local" cache implementation would look like:
///   std::string cache_name = "local";
///   std::string config_json = R"({"size": 1048576})"
///   auto err = TRITONSERVER_ServerOptionsSetCacheConfig(
///     options, cache_name, config_json);
///
/// \param options The server options object.
/// \param cache_name The name of the cache. Example names would be
/// "local", "redis", or the name of a custom cache implementation.
/// \param config_json The string representation of config JSON that is
/// used to initialize the cache implementation.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetCacheConfig(
    struct TRITONSERVER_ServerOptions* options, const char* cache_name,
    const char* config_json);

/// Set the directory containing cache shared libraries. This
/// directory is searched when looking for cache implementations.
///
/// \param options The server options object.
/// \param cache_dir The full path of the cache directory.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetCacheDirectory(
    struct TRITONSERVER_ServerOptions* options, const char* cache_dir);

/// Set the minimum support CUDA compute capability in a server
/// options.
///
/// \param options The server options object.
/// \param cc The minimum CUDA compute capability.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetMinSupportedComputeCapability(
    struct TRITONSERVER_ServerOptions* options, double cc);

/// Enable or disable exit-on-error in a server options.
///
/// \param options The server options object.
/// \param exit True to enable exiting on initialization error, false
/// to continue.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetExitOnError(
    struct TRITONSERVER_ServerOptions* options, bool exit);

/// Enable or disable strict readiness handling in a server options.
///
/// \param options The server options object.
/// \param strict True to enable strict readiness handling, false to
/// disable.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetStrictReadiness(
    struct TRITONSERVER_ServerOptions* options, bool strict);

/// Set the exit timeout, in seconds, for the server in a server
/// options.
///
/// \param options The server options object.
/// \param timeout The exit timeout, in seconds.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetExitTimeout(
    struct TRITONSERVER_ServerOptions* options, unsigned int timeout);

/// Set the number of threads used in buffer manager in a server options.
///
/// \param options The server options object.
/// \param thread_count The number of threads.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetBufferManagerThreadCount(
    struct TRITONSERVER_ServerOptions* options, unsigned int thread_count);

/// Set the number of threads to concurrently load models in a server options.
///
/// \param options The server options object.
/// \param thread_count The number of threads.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetModelLoadThreadCount(
    struct TRITONSERVER_ServerOptions* options, unsigned int thread_count);

/// Set the number of retry to load a model in a server options.
///
/// \param options The server options object.
/// \param retry_count The number of retry.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetModelLoadRetryCount(
    struct TRITONSERVER_ServerOptions* options, unsigned int retry_count);

/// Enable model namespacing to allow serving models with the same name if
/// they are in different namespaces.
///
/// \param options The server options object.
/// \param enable_namespace Whether to enable model namespacing or not.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetModelNamespacing(
    struct TRITONSERVER_ServerOptions* options, bool enable_namespace);

/// Enable peer access to allow GPU device to directly access the memory of
/// another GPU device. Note that even when this option is set to True, Triton
/// will only try to enable peer access and might fail to enable it if the
/// underlying system doesn't support peer access.
///
/// \param options The server options object.
/// \param enable_peer_access Whether to enable peer access or not.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetEnablePeerAccess(
    struct TRITONSERVER_ServerOptions* options, bool enable_peer_access);

/// Provide a log output file.
///
/// \param options The server options object.
/// \param file a string defining the file where the log outputs will be saved.
/// An empty string for the file name will cause triton to direct logging
/// facilities to the console
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetLogFile(
    struct TRITONSERVER_ServerOptions* options, const char* file);

/// Enable or disable info level logging.
///
/// \param options The server options object.
/// \param log True to enable info logging, false to disable.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetLogInfo(
    struct TRITONSERVER_ServerOptions* options, bool log);

/// Enable or disable warning level logging.
///
/// \param options The server options object.
/// \param log True to enable warning logging, false to disable.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetLogWarn(
    struct TRITONSERVER_ServerOptions* options, bool log);

/// Enable or disable error level logging.
///
/// \param options The server options object.
/// \param log True to enable error logging, false to disable.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetLogError(
    struct TRITONSERVER_ServerOptions* options, bool log);

/// Set the logging format.
///
/// \param options The server options object.
/// \param format The logging format.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetLogFormat(
    struct TRITONSERVER_ServerOptions* options,
    const TRITONSERVER_LogFormat format);

/// Set verbose logging level. Level zero disables verbose logging.
///
/// \param options The server options object.
/// \param level The verbose logging level.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetLogVerbose(
    struct TRITONSERVER_ServerOptions* options, int level);

/// Enable or disable metrics collection in a server options.
///
/// \param options The server options object.
/// \param metrics True to enable metrics, false to disable.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetMetrics(
    struct TRITONSERVER_ServerOptions* options, bool metrics);

/// Enable or disable GPU metrics collection in a server options. GPU
/// metrics are collected if both this option and
/// TRITONSERVER_ServerOptionsSetMetrics are true.
///
/// \param options The server options object.
/// \param gpu_metrics True to enable GPU metrics, false to disable.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetGpuMetrics(
    struct TRITONSERVER_ServerOptions* options, bool gpu_metrics);

/// Enable or disable CPU metrics collection in a server options. CPU
/// metrics are collected if both this option and
/// TRITONSERVER_ServerOptionsSetMetrics are true.
///
/// \param options The server options object.
/// \param cpu_metrics True to enable CPU metrics, false to disable.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetCpuMetrics(
    struct TRITONSERVER_ServerOptions* options, bool cpu_metrics);

/// Set the interval for metrics collection in a server options.
/// This is 2000 milliseconds by default.
///
/// \param options The server options object.
/// \param metrics_interval_ms The time interval in ms between
/// successive metrics updates.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetMetricsInterval(
    struct TRITONSERVER_ServerOptions* options, uint64_t metrics_interval_ms);

/// Set the directory containing backend shared libraries. This
/// directory is searched last after the version and model directory
/// in the model repository when looking for the backend shared
/// library for a model. If the backend is named 'be' the directory
/// searched is 'backend_dir'/be/libtriton_be.so.
///
/// \param options The server options object.
/// \param backend_dir The full path of the backend directory.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetBackendDirectory(
    struct TRITONSERVER_ServerOptions* options, const char* backend_dir);

/// Set the directory containing repository agent shared libraries. This
/// directory is searched when looking for the repository agent shared
/// library for a model. If the repo agent is named 'ra' the directory
/// searched is 'repoagent_dir'/ra/libtritonrepoagent_ra.so.
///
/// \param options The server options object.
/// \param repoagent_dir The full path of the repository agent directory.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetRepoAgentDirectory(
    struct TRITONSERVER_ServerOptions* options, const char* repoagent_dir);

/// Specify the limit on memory usage as a fraction on the device identified by
/// 'kind' and 'device_id'. If model loading on the device is requested and the
/// current memory usage exceeds the limit, the load will be rejected. If not
/// specified, the limit will not be set.
///
/// Currently support TRITONSERVER_INSTANCEGROUPKIND_GPU
///
/// \param options The server options object.
/// \param kind The kind of the device.
/// \param device_id The id of the device.
/// \param fraction The limit on memory usage as a fraction
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetModelLoadDeviceLimit(
    struct TRITONSERVER_ServerOptions* options,
    const TRITONSERVER_InstanceGroupKind kind, const int device_id,
    const double fraction);

/// Set a configuration setting for a named backend in a server
/// options.
///
/// \param options The server options object.
/// \param backend_name The name of the backend.
/// \param setting The name of the setting.
/// \param value The setting value.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetBackendConfig(
    struct TRITONSERVER_ServerOptions* options, const char* backend_name,
    const char* setting, const char* value);

/// Set a host policy setting for a given policy name in a server options.
///
/// \param options The server options object.
/// \param policy_name The name of the policy.
/// \param setting The name of the setting.
/// \param value The setting value.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetHostPolicy(
    struct TRITONSERVER_ServerOptions* options, const char* policy_name,
    const char* setting, const char* value);

/// Set a configuration setting for metrics in server options.
///
/// \param options The server options object.
/// \param name The name of the configuration group. An empty string indicates
///             a global configuration option.
/// \param setting The name of the setting.
/// \param value The setting value.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetMetricsConfig(
    struct TRITONSERVER_ServerOptions* options, const char* name,
    const char* setting, const char* value);

/// TRITONSERVER_Server
///
/// An inference server.
///

/// Model batch flags. The enum values must be power-of-2 values.
typedef enum tritonserver_batchflag_enum {
  TRITONSERVER_BATCH_UNKNOWN = 1,
  TRITONSERVER_BATCH_FIRST_DIM = 2
} TRITONSERVER_ModelBatchFlag;

/// Model index flags. The enum values must be power-of-2 values.
typedef enum tritonserver_modelindexflag_enum {
  TRITONSERVER_INDEX_FLAG_READY = 1
} TRITONSERVER_ModelIndexFlag;

/// Model transaction policy flags. The enum values must be
/// power-of-2 values.
typedef enum tritonserver_txn_property_flag_enum {
  TRITONSERVER_TXN_ONE_TO_ONE = 1,
  TRITONSERVER_TXN_DECOUPLED = 2
} TRITONSERVER_ModelTxnPropertyFlag;

/// Create a new server object. The caller takes ownership of the
/// TRITONSERVER_Server object and must call TRITONSERVER_ServerDelete
/// to release the object.
///
/// \param server Returns the new inference server object.
/// \param options The inference server options object.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error* TRITONSERVER_ServerNew(
    struct TRITONSERVER_Server** server,
    struct TRITONSERVER_ServerOptions* options);

/// Delete a server object. If server is not already stopped it is
/// stopped before being deleted.
///
/// \param server The inference server object.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error* TRITONSERVER_ServerDelete(
    struct TRITONSERVER_Server* server);

/// Stop a server object. A server can't be restarted once it is
/// stopped.
///
/// \param server The inference server object.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error* TRITONSERVER_ServerStop(
    struct TRITONSERVER_Server* server);

/// Set the exit timeout on the server object. This value overrides the value
/// initially set through server options and provides a mechanism to update the
/// exit timeout while the serving is running.
///
/// \param server The inference server object.
/// \param timeout The exit timeout, in seconds.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_ServerSetExitTimeout(
    struct TRITONSERVER_Server* server, unsigned int timeout);

/// Register a new model repository. Not available in polling mode.
///
/// \param server The inference server object.
/// \param repository_path The full path to the model repository.
/// \param name_mapping List of name_mapping parameters. Each mapping has
/// the model directory name as its key, overridden model name as its value.
/// \param model_count Number of mappings provided.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_ServerRegisterModelRepository(
    struct TRITONSERVER_Server* server, const char* repository_path,
    const struct TRITONSERVER_Parameter** name_mapping,
    const uint32_t mapping_count);

/// Unregister a model repository. Not available in polling mode.
///
/// \param server The inference server object.
/// \param repository_path The full path to the model repository.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_ServerUnregisterModelRepository(
    struct TRITONSERVER_Server* server, const char* repository_path);

/// Check the model repository for changes and update server state
/// based on those changes.
///
/// \param server The inference server object.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_ServerPollModelRepository(struct TRITONSERVER_Server* server);

/// Is the server live?
///
/// \param server The inference server object.
/// \param live Returns true if server is live, false otherwise.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error* TRITONSERVER_ServerIsLive(
    struct TRITONSERVER_Server* server, bool* live);

/// Is the server ready?
///
/// \param server The inference server object.
/// \param ready Returns true if server is ready, false otherwise.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error* TRITONSERVER_ServerIsReady(
    struct TRITONSERVER_Server* server, bool* ready);

/// Is the model ready?
///
/// \param server The inference server object.
/// \param model_name The name of the model to get readiness for.
/// \param model_version The version of the model to get readiness
/// for.  If -1 then the server will choose a version based on the
/// model's policy.
/// \param ready Returns true if server is ready, false otherwise.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_ServerModelIsReady(
    struct TRITONSERVER_Server* server, const char* model_name,
    const int64_t model_version, bool* ready);

/// Get the batch properties of the model. The properties are
/// communicated by a flags value and an (optional) object returned by
/// 'voidp'.
///
///   - TRITONSERVER_BATCH_UNKNOWN: Triton cannot determine the
///     batching properties of the model. This means that the model
///     does not support batching in any way that is usable by
///     Triton. The returned 'voidp' value is nullptr.
///
///   - TRITONSERVER_BATCH_FIRST_DIM: The model supports batching
///     along the first dimension of every input and output
///     tensor. Triton schedulers that perform batching can
///     automatically batch inference requests along this dimension.
///     The returned 'voidp' value is nullptr.
///
/// \param server The inference server object.
/// \param model_name The name of the model.
/// \param model_version The version of the model.  If -1 then the
/// server will choose a version based on the model's policy.
/// \param flags Returns flags indicating the batch properties of the
/// model.
/// \param voidp If non-nullptr, returns a point specific to the
/// 'flags' value.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_ServerModelBatchProperties(
    struct TRITONSERVER_Server* server, const char* model_name,
    const int64_t model_version, uint32_t* flags, void** voidp);

/// Get the transaction policy of the model. The policy is
/// communicated by a flags value.
///
///   - TRITONSERVER_TXN_ONE_TO_ONE: The model generates exactly
///     one response per request.
///
///   - TRITONSERVER_TXN_DECOUPLED: The model may generate zero
///     to many responses per request.
///
/// \param server The inference server object.
/// \param model_name The name of the model.
/// \param model_version The version of the model.  If -1 then the
/// server will choose a version based on the model's policy.
/// \param txn_flags Returns flags indicating the transaction policy of the
/// model.
/// \param voidp If non-nullptr, returns a point specific to the 'flags' value.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_ServerModelTransactionProperties(
    struct TRITONSERVER_Server* server, const char* model_name,
    const int64_t model_version, uint32_t* txn_flags, void** voidp);

/// Get the metadata of the server as a TRITONSERVER_Message object.
/// The caller takes ownership of the message object and must call
/// TRITONSERVER_MessageDelete to release the object.
///
/// \param server The inference server object.
/// \param server_metadata Returns the server metadata message.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error* TRITONSERVER_ServerMetadata(
    struct TRITONSERVER_Server* server,
    struct TRITONSERVER_Message** server_metadata);

/// Get the metadata of a model as a TRITONSERVER_Message
/// object.  The caller takes ownership of the message object and must
/// call TRITONSERVER_MessageDelete to release the object.
///
/// \param server The inference server object.
/// \param model_name The name of the model.
/// \param model_version The version of the model.
/// If -1 then the server will choose a version based on the model's
/// policy.
/// \param model_metadata Returns the model metadata message.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_ServerModelMetadata(
    struct TRITONSERVER_Server* server, const char* model_name,
    const int64_t model_version, struct TRITONSERVER_Message** model_metadata);

/// Get the statistics of a model as a TRITONSERVER_Message
/// object. The caller takes ownership of the object and must call
/// TRITONSERVER_MessageDelete to release the object.
///
/// \param server The inference server object.
/// \param model_name The name of the model.
/// If empty, then statistics for all available models will be returned,
/// and the server will choose a version based on those models' policies.
/// \param model_version The version of the model.  If -1 then the
/// server will choose a version based on the model's policy.
/// \param model_stats Returns the model statistics message.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_ServerModelStatistics(
    struct TRITONSERVER_Server* server, const char* model_name,
    const int64_t model_version, struct TRITONSERVER_Message** model_stats);

/// Get the configuration of a model as a TRITONSERVER_Message object.
/// The caller takes ownership of the message object and must call
/// TRITONSERVER_MessageDelete to release the object.
///
/// \param server The inference server object.
/// \param model_name The name of the model.
/// \param model_version The version of the model.  If -1 then the
/// server will choose a version based on the model's policy.
/// \param config_version The model configuration will be returned in
/// a format matching this version. If the configuration cannot be
/// represented in the requested version's format then an error will
/// be returned. Currently only version 1 is supported.
/// \param model_config Returns the model config message.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error* TRITONSERVER_ServerModelConfig(
    struct TRITONSERVER_Server* server, const char* model_name,
    const int64_t model_version, const uint32_t config_version,
    struct TRITONSERVER_Message** model_config);

/// Get the index of all unique models in the model repositories as a
/// TRITONSERVER_Message object. The caller takes ownership of the
/// message object and must call TRITONSERVER_MessageDelete to release
/// the object.
///
/// If TRITONSERVER_INDEX_FLAG_READY is set in 'flags' only the models
/// that are loaded into the server and ready for inferencing are
/// returned.
///
/// \param server The inference server object.
/// \param flags TRITONSERVER_ModelIndexFlag flags that control how to
/// collect the index.
/// \param model_index Return the model index message that holds the
/// index of all models contained in the server's model repository(s).
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error* TRITONSERVER_ServerModelIndex(
    struct TRITONSERVER_Server* server, uint32_t flags,
    struct TRITONSERVER_Message** model_index);

/// Load the requested model or reload the model if it is already
/// loaded. The function does not return until the model is loaded or
/// fails to load. Returned error indicates if model loaded
/// successfully or not.
///
/// \param server The inference server object.
/// \param model_name The name of the model.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error* TRITONSERVER_ServerLoadModel(
    struct TRITONSERVER_Server* server, const char* model_name);

/// Load the requested model or reload the model if it is already
/// loaded, with load parameters provided. The function does not return until
/// the model is loaded or fails to load. Returned error indicates if model
/// loaded successfully or not.
/// Currently the below parameter names are recognized:
/// - "config" : string parameter that contains a JSON representation of the
/// model configuration. This config will be used for loading the model instead
/// of the one in the model directory.
///
/// \param server The inference server object.
/// \param model_name The name of the model.
/// \param parameters The array of load parameters.
/// \param parameter_count The number of parameters.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_ServerLoadModelWithParameters(
    struct TRITONSERVER_Server* server, const char* model_name,
    const struct TRITONSERVER_Parameter** parameters,
    const uint64_t parameter_count);

/// Unload the requested model. Unloading a model that is not loaded
/// on server has no affect and success code will be returned.
/// The function does not wait for the requested model to be fully unload
/// and success code will be returned.
/// Returned error indicates if model unloaded successfully or not.
///
/// \param server The inference server object.
/// \param model_name The name of the model.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error* TRITONSERVER_ServerUnloadModel(
    struct TRITONSERVER_Server* server, const char* model_name);

/// Unload the requested model, and also unload any dependent model that
/// was loaded along with the requested model (for example, the models composing
/// an ensemble). Unloading a model that is not loaded
/// on server has no affect and success code will be returned.
/// The function does not wait for the requested model and all dependent
/// models to be fully unload and success code will be returned.
/// Returned error indicates if model unloaded successfully or not.
///
/// \param server The inference server object.
/// \param model_name The name of the model.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_ServerUnloadModelAndDependents(
    struct TRITONSERVER_Server* server, const char* model_name);

/// Get the current metrics for the server. The caller takes ownership
/// of the metrics object and must call TRITONSERVER_MetricsDelete to
/// release the object.
///
/// \param server The inference server object.
/// \param metrics Returns the metrics.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error* TRITONSERVER_ServerMetrics(
    struct TRITONSERVER_Server* server, struct TRITONSERVER_Metrics** metrics);

/// Perform inference using the meta-data and inputs supplied by the
/// 'inference_request'. If the function returns success, then the
/// caller releases ownership of 'inference_request' and must not
/// access it in any way after this call, until ownership is returned
/// via the 'request_release_fn' callback registered in the request
/// object with TRITONSERVER_InferenceRequestSetReleaseCallback.
///
/// The function unconditionally takes ownership of 'trace' and so the
/// caller must not access it in any way after this call (except in
/// the trace activity callbacks) until ownership is returned via the
/// trace's release_fn callback.
///
/// Responses produced for this request are returned using the
/// allocator and callback registered with the request by
/// TRITONSERVER_InferenceRequestSetResponseCallback.
///
/// \param server The inference server object.
/// \param inference_request The request object.
/// \param trace The trace object for this request, or nullptr if no
/// tracing.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error* TRITONSERVER_ServerInferAsync(
    struct TRITONSERVER_Server* server,
    struct TRITONSERVER_InferenceRequest* inference_request,
    struct TRITONSERVER_InferenceTrace* trace);

/// TRITONSERVER_MetricKind
///
/// Types of metrics recognized by TRITONSERVER.
///
typedef enum TRITONSERVER_metrickind_enum {
  TRITONSERVER_METRIC_KIND_COUNTER,
  TRITONSERVER_METRIC_KIND_GAUGE,
  TRITONSERVER_METRIC_KIND_HISTOGRAM
} TRITONSERVER_MetricKind;

/// Create a new metric family object. The caller takes ownership of the
/// TRITONSERVER_MetricFamily object and must call
/// TRITONSERVER_MetricFamilyDelete to release the object.
///
/// \param family Returns the new metric family object.
/// \param kind The type of metric family to create.
/// \param name The name of the metric family seen when calling the metrics
/// endpoint.
/// \param description The description of the metric family seen when
/// calling the metrics endpoint.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error* TRITONSERVER_MetricFamilyNew(
    struct TRITONSERVER_MetricFamily** family,
    const TRITONSERVER_MetricKind kind, const char* name,
    const char* description);

/// Delete a metric family object.
/// A TRITONSERVER_MetricFamily* object should be deleted AFTER its
/// corresponding TRITONSERVER_Metric* objects have been deleted.
/// Attempting to delete a family before its metrics will return an error.
///
/// \param family The metric family object to delete.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_MetricFamilyDelete(struct TRITONSERVER_MetricFamily* family);

/// Get the TRITONSERVER_MetricKind of the metric family.
///
/// \param family The metric family object to query.
/// \param kind Returns the TRITONSERVER_MetricKind of metric.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_GetMetricFamilyKind(
    struct TRITONSERVER_MetricFamily* family, TRITONSERVER_MetricKind* kind);

/// Create a new metric args object. The caller takes ownership of the
/// TRITONSERVER_MetricArgs object and must call TRITONSERVER_MetricArgsDelete
/// to release the object.
///
/// \param args Returns the new metric args object.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error* TRITONSERVER_MetricArgsNew(
    struct TRITONSERVER_MetricArgs** args);

/// Set metric args with histogram metric parameter.
///
/// \param args The metric args object to set.
/// \param buckets The array of bucket boundaries for the expected range of
/// observed values.
///
/// \param buckets_count The number of bucket boundaries.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error*
TRITONSERVER_MetricArgsSetHistogram(
    struct TRITONSERVER_MetricArgs* args, const double* buckets,
    const uint64_t buckets_count);

/// Delete a metric args object.
///
/// \param args The metric args object.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error* TRITONSERVER_MetricArgsDelete(
    struct TRITONSERVER_MetricArgs* args);

/// Create a new metric object. The caller takes ownership of the
/// TRITONSERVER_Metric object and must call
/// TRITONSERVER_MetricDelete to release the object. The caller is also
/// responsible for ownership of the labels passed in. Each label can be deleted
/// immediately after creating the metric with TRITONSERVER_ParameterDelete
/// if not re-using the labels.
///
/// \param metric Returns the new metric object.
/// \param family The metric family to add this new metric to.
/// \param labels The array of labels to associate with this new metric.
/// \param label_count The number of labels.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error* TRITONSERVER_MetricNew(
    struct TRITONSERVER_Metric** metric,
    struct TRITONSERVER_MetricFamily* family,
    const struct TRITONSERVER_Parameter** labels, const uint64_t label_count);

/// Create a new metric object. The caller takes ownership of the
/// TRITONSERVER_Metric object and must call
/// TRITONSERVER_MetricDelete to release the object. The caller is also
/// responsible for ownership of the labels passed in.
/// Each label can be deleted immediately after creating the metric with
/// TRITONSERVER_ParameterDelete if not re-using the labels.
/// Metric args can be deleted immediately after creating the metric with
/// TRITONSERVER_MetricArgsDelete if not re-using the metric args.
///
/// \param metric Returns the new metric object.
/// \param family The metric family to add this new metric to.
/// \param labels The array of labels to associate with this new metric.
/// \param label_count The number of labels.
/// \param args Metric args that store additional arguments to construct
/// particular metric types, e.g. histogram.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error* TRITONSERVER_MetricNewWithArgs(
    struct TRITONSERVER_Metric** metric,
    struct TRITONSERVER_MetricFamily* family,
    const struct TRITONSERVER_Parameter** labels, const uint64_t label_count,
    const struct TRITONSERVER_MetricArgs* args);

/// Delete a metric object.
/// All TRITONSERVER_Metric* objects should be deleted BEFORE their
/// corresponding TRITONSERVER_MetricFamily* objects have been deleted.
/// If a family is deleted before its metrics, an error will be returned.
///
/// \param metric The metric object to delete.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error* TRITONSERVER_MetricDelete(
    struct TRITONSERVER_Metric* metric);

/// Get the current value of a metric object.
/// Supports metrics of kind TRITONSERVER_METRIC_KIND_COUNTER
/// and TRITONSERVER_METRIC_KIND_GAUGE, and returns
/// TRITONSERVER_ERROR_UNSUPPORTED for unsupported TRITONSERVER_MetricKind.
///
/// \param metric The metric object to query.
/// \param value Returns the current value of the metric object.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error* TRITONSERVER_MetricValue(
    struct TRITONSERVER_Metric* metric, double* value);

/// Increment the current value of metric by value.
/// Supports metrics of kind TRITONSERVER_METRIC_KIND_GAUGE for any value,
/// and TRITONSERVER_METRIC_KIND_COUNTER for non-negative values. Returns
/// TRITONSERVER_ERROR_UNSUPPORTED for unsupported TRITONSERVER_MetricKind
/// and TRITONSERVER_ERROR_INVALID_ARG for negative values on a
/// TRITONSERVER_METRIC_KIND_COUNTER metric.
///
/// \param metric The metric object to update.
/// \param value The amount to increment the metric's value by.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error* TRITONSERVER_MetricIncrement(
    struct TRITONSERVER_Metric* metric, double value);

/// Set the current value of metric to value.
/// Supports metrics of kind TRITONSERVER_METRIC_KIND_GAUGE and returns
/// TRITONSERVER_ERROR_UNSUPPORTED for unsupported TRITONSERVER_MetricKind.
///
/// \param metric The metric object to update.
/// \param value The amount to set metric's value to.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error* TRITONSERVER_MetricSet(
    struct TRITONSERVER_Metric* metric, double value);

/// Sample an observation and count it to the appropriate bucket of a metric.
/// Supports metrics of kind TRITONSERVER_METRIC_KIND_HISTOGRAM and returns
/// TRITONSERVER_ERROR_UNSUPPORTED for unsupported TRITONSERVER_MetricKind.
///
/// \param metric The metric object to update.
/// \param value The amount for metric to sample observation.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error* TRITONSERVER_MetricObserve(
    struct TRITONSERVER_Metric* metric, double value);

/// Get the TRITONSERVER_MetricKind of metric of its corresponding family.
///
/// \param metric The metric object to query.
/// \param kind Returns the TRITONSERVER_MetricKind of metric.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC struct TRITONSERVER_Error* TRITONSERVER_GetMetricKind(
    struct TRITONSERVER_Metric* metric, TRITONSERVER_MetricKind* kind);

#ifdef __cplusplus
}
#endif
