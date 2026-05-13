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
 * <b>NVIDIA DeepStream: Metadata Extension Structures</b>
 *
 * @b Description: This file defines the NVIDIA DeepStream metadata structures
 * used to describe metadata objects.
 */

/**
 * @defgroup  metadata_extensions  Metadata Extension Structures
 *
 * Defines metadata structures used to describe metadata objects.
 *
 * @ingroup NvDsMetaApi
 * @{
 */

#ifndef NVDSMETA_H_
#define NVDSMETA_H_

#include <glib.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * Defines event type flags.
 */
typedef enum NvDsEventType {
  NVDS_EVENT_ENTRY,
  NVDS_EVENT_EXIT,
  NVDS_EVENT_MOVING,
  NVDS_EVENT_STOPPED,
  NVDS_EVENT_EMPTY,
  NVDS_EVENT_PARKED,
  NVDS_EVENT_RESET,

  /** Reserved for future use. Custom events must be assigned values
   greater than this. */
  NVDS_EVENT_RESERVED = 0x100,
  /** Specifies a custom event. */
  NVDS_EVENT_CUSTOM = 0x101,
  NVDS_EVENT_FORCE32 = 0x7FFFFFFF
} NvDsEventType;

/**
 * Defines object type flags.
 */
typedef enum NvDsObjectType {
  NVDS_OBJECT_TYPE_VEHICLE,
  NVDS_OBJECT_TYPE_PERSON,
  NVDS_OBJECT_TYPE_FACE,
  NVDS_OBJECT_TYPE_BAG,
  NVDS_OBJECT_TYPE_BICYCLE,
  NVDS_OBJECT_TYPE_ROADSIGN,
  NVDS_OBJECT_TYPE_VEHICLE_EXT,
  NVDS_OBJECT_TYPE_PERSON_EXT,
  NVDS_OBJECT_TYPE_FACE_EXT,
  NVDS_OBJECT_TYPE_3D,
  NVDS_OBJECT_TYPE_DUMMY,
  NVDS_OBJECT_TYPE_PRODUCT,	/** New Object Type for Product */
  NVDS_OBJECT_TYPE_PRODUCT_EXT, /** New Object Type for Product */
  /** Reserved for future use. Custom objects must be assigned values
   greater than this. */
  NVDS_OBJECT_TYPE_RESERVED = 0x100,
  /** Specifies a custom object. */
  NVDS_OBJECT_TYPE_CUSTOM = 0x101,
  /** "object" key will be missing in the schema */
  NVDS_OBJECT_TYPE_UNKNOWN = 0x102,
  NVDS_OBEJCT_TYPE_FORCE32 = 0x7FFFFFFF
} NvDsObjectType;

/**
 * Defines payload type flags.
 */
typedef enum NvDsPayloadType {
  NVDS_PAYLOAD_DEEPSTREAM,
  NVDS_PAYLOAD_DEEPSTREAM_MINIMAL,
  NVDS_PAYLOAD_DEEPSTREAM_PROTOBUF,
  /** Reserved for future use. Custom payloads must be assigned values
   greater than this. */
  NVDS_PAYLOAD_RESERVED = 0x100,
  /** Specifies a custom payload. You must implement the nvds_msg2p_*
   interface. */
  NVDS_PAYLOAD_CUSTOM = 0x101,
  NVDS_PAYLOAD_FORCE32 = 0x7FFFFFFF
} NvDsPayloadType;

/**
 * Holds a rectangle's position and size.
 */
typedef struct NvDsRect {
  float top;     /**< Holds the position of rectangle's top in pixels. */
  float left;    /**< Holds the position of rectangle's left side in pixels. */
  float width;   /**< Holds the rectangle's width in pixels. */
  float height;  /**< Holds the rectangle's height in pixels. */
} NvDsRect;

/**
 * Holds geolocation parameters.
 */
typedef struct NvDsGeoLocation {
  gdouble lat;      /**< Holds the location's latitude. */
  gdouble lon;      /**< Holds the location's longitude. */
  gdouble alt;      /**< Holds the location's altitude. */
} NvDsGeoLocation;

/**
 * Hold a coordinate's position.
 */
typedef struct NvDsCoordinate {
  gdouble x;        /**< Holds the coordinate's X position. */
  gdouble y;        /**< Holds the coordinate's Y position. */
  gdouble z;        /**< Holds the coordinate's Z position. */
} NvDsCoordinate;

/**
 * Holds an object's signature.
 */
typedef struct NvDsObjectSignature {
  /** Holds a pointer to an array of signature values. */
  gdouble *signature;
  /** Holds the number of signature values in @a signature. */
  guint size;
} NvDsObjectSignature;

/**
* Holds a product object's parameters.
*/
typedef struct NvDsProductObjectExt {
  gchar *brand; /**< Holds the product brand */
  gchar *type; /**< Holds the product type */
  gchar *shape; /**< Holds the product shape */
  GList *mask;
} NvDsProductObjectExt;

/**
 * Holds a vehicle object's parameters.
 */
typedef struct NvDsVehicleObject {
  gchar *type;      /**< Holds a pointer to the type of the vehicle. */
  gchar *make;      /**< Holds a pointer to the make of the vehicle. */
  gchar *model;     /**< Holds a pointer to the model of the vehicle. */
  gchar *color;     /**< Holds a pointer to the color of the vehicle. */
  gchar *region;    /**< Holds a pointer to the region of the vehicle. */
  gchar *license;   /**< Holds a pointer to the license number of the vehicle.*/
} NvDsVehicleObject;

/**
 * Holds a person object's parameters.
 */
typedef struct NvDsPersonObject {
  gchar *gender;    /**< Holds a pointer to the person's gender. */
  gchar *hair;      /**< Holds a pointer to the person's hair color. */
  gchar *cap;       /**< Holds a pointer to the type of cap the person is
                     wearing, if any. */
  gchar *apparel;   /**< Holds a pointer to a description of the person's
                     apparel. */
  guint age;        /**< Holds the person's age. */
} NvDsPersonObject;

/**
 * Holds a face object's parameters.
 */
typedef struct NvDsFaceObject {
  gchar *gender;    /**< Holds a pointer to the person's gender. */
  gchar *hair;      /**< Holds a pointer to the person's hair color. */
  gchar *cap;       /**< Holds a pointer to the type of cap the person
                     is wearing, if any. */
  gchar *glasses;   /**< Holds a pointer to the type of glasses the person
                     is wearing, if any. */
  gchar *facialhair;/**< Holds a pointer to the person's facial hair color. */
  gchar *name;      /**< Holds a pointer to the person's name. */
  gchar *eyecolor;  /**< Holds a pointer to the person's eye color. */
  guint age;        /**< Holds the person's age. */
} NvDsFaceObject;

/**
 * Holds a vehicle object's parameters.
 */
typedef struct NvDsVehicleObjectExt {
  gchar *type;      /**< Holds a pointer to the type of the vehicle. */
  gchar *make;      /**< Holds a pointer to the make of the vehicle. */
  gchar *model;     /**< Holds a pointer to the model of the vehicle. */
  gchar *color;     /**< Holds a pointer to the color of the vehicle. */
  gchar *region;    /**< Holds a pointer to the region of the vehicle. */
  gchar *license;   /**< Holds a pointer to the license number of the vehicle.*/

  GList *mask;      /**< Holds a list of polygons for vehicle mask. */
} NvDsVehicleObjectExt;

/**
 * Holds a person object's parameters.
 */
typedef struct NvDsPersonObjectExt {
  gchar *gender;    /**< Holds a pointer to the person's gender. */
  gchar *hair;      /**< Holds a pointer to the person's hair color. */
  gchar *cap;       /**< Holds a pointer to the type of cap the person is
                     wearing, if any. */
  gchar *apparel;   /**< Holds a pointer to a description of the person's
                     apparel. */
  guint age;        /**< Holds the person's age. */

  GList *mask;      /**< Holds a list of polygons for person mask. */
} NvDsPersonObjectExt;

/**
 * Holds a face object's parameters.
 */
typedef struct NvDsFaceObjectWithExt {
  gchar *gender;    /**< Holds a pointer to the person's gender. */
  gchar *hair;      /**< Holds a pointer to the person's hair color. */
  gchar *cap;       /**< Holds a pointer to the type of cap the person
                     is wearing, if any. */
  gchar *glasses;   /**< Holds a pointer to the type of glasses the person
                     is wearing, if any. */
  gchar *facialhair;/**< Holds a pointer to the person's facial hair color. */
  gchar *name;      /**< Holds a pointer to the person's name. */
  gchar *eyecolor;  /**< Holds a pointer to the person's eye color. */
  guint age;        /**< Holds the person's age. */

  GList *mask;      /**< Holds a list of polygons for face mask. */
} NvDsFaceObjectExt;

/**
 * Holds a joint's position and confidence.
 */
typedef struct NvDsJoint {
  float x;     /**< Holds the joint x position in pixels.*/
  float y;     /**< Holds the joint y position in pixels.*/
  float z;     /**< Holds the joint z position in pixels.*/
  float confidence;   /**< Holds the confidence of the joint. */
} NvDsJoint;

/**
 * Holds a body pose's joint points.
 */
typedef struct NvDsJoints {
  NvDsJoint *joints;   /**< Holds the joints of the person.*/
  int num_joints;      /**< Holds the number of joints.*/
  int pose_type;       /**< Holds the type of pose 0 - 2D, 1 - 3D , integer to support 2.5D in future*/
} NvDsJoints;

/**
* Holds an embedding model's parameters.
*/
typedef struct NvDsEmbedding {
  float *embedding_vector; /**< Holds the embedding vector */
  guint embedding_length; /**< Holds the length of embedding vector */
} NvDsEmbedding;

/**
* Holds a product object's parameters.
*/
typedef struct NvDsProductObject {
  gchar *brand; /**< Holds the product brand */
  gchar *type; /**< Holds the product type */
  gchar *shape; /**< Holds the product shape */
} NvDsProductObject;

/**
 * Holds 2D convex hull metadata.
 */
typedef struct NvDsConvexHull {
  /** Holds a pointer a list of convex hull points. */
  gint* points;
  /** Holds the number of points in the convex hull. */
  guint numFilled;
} NvDsConvexHull;

typedef struct _NvDsBbox3dObject {
    float boxes_3d[12];
    float scores_3d;
    gchar *labels_3d;
    gchar *embedding_filter;
    float cls_scores; // confidence
    guint instance_ids;
    NvDsEmbedding *instance_feats;
    gchar *ocr_label;
} NvDsBbox3dObject;

#define MAX_SOURCE_ID_LEN 128  // Define a maximum length for source_id
typedef struct {
    gchar source_id[MAX_SOURCE_ID_LEN];
    guint64 timestamp;
} SourceTimePair;

#define MAX_NUM_OBJECTS 600  // Define a maximum number of objects
#define MAX_ENTRIES 100      //Define max batchsize for sparse4D
typedef struct _NvDsBbox3dObjectList {
    NvDsBbox3dObject objects[MAX_NUM_OBJECTS];
    guint num_objects;
    guint64 timestamp;
    gint frame_num;
    guint camera_id;
    gchar *sensorStr;
    SourceTimePair entries[MAX_ENTRIES];
    int count;
} NvDsBbox3dObjectList;

/**
 * Holds Single View 3D Tracking metadata.
 */
 typedef struct NvDs3DTracking {
  /** Holds the object's estimated foot location in 2D coordinates on the estimated world ground. */
  gfloat ptWorldFeet[2];
  /** Holds the object's estimated foot location in frame coordinates. */
  gfloat ptImgFeet[2];
  /** Holds the object's visibility. */
  gfloat visibility;
  /** Holds the object's convex hull information */
  NvDsConvexHull convexHull;
  NvDsBbox3dObject bbox3d;
} NvDs3DTracking;


typedef struct {
    gchar *id;
    gchar *query;
    gchar *response;
} NvDsConversationObject;

typedef struct {
    GHashTable *info;
    NvDsConversationObject *conversations;
    guint num_conversations;
    NvDsEmbedding *embeddings;
    guint num_embeddings;
} NvDsLLMObject;

typedef struct {
    gchar *id;
    gchar **objectIds;
    guint num_objectIds;
    NvDsCoordinate *coordinates;
    guint num_coordinates;
    gchar *description;
    GHashTable *info;
    gint64 timestamp;  // New field added
} NvDsInteractionObject;

typedef struct {
    gchar *id;
    gchar **objectIds;
    guint num_objectIds;
    float amount;
    GHashTable *info;
    gint64 timestamp;  // New field added
} NvDsCongestionObject;

typedef struct {
    gchar *id;
    gchar *type;
    gint32 count;
    NvDsCoordinate *coordinates;
    guint num_coordinates;
    gchar **objectIds;
    guint num_objectIds;
    GHashTable *info;
} NvDsTypeMetricsObject;

typedef struct {
    gint32 *mask;
    guint mask_size;
    GHashTable *info;
} NvDsSegmentationObject;

/**
 * Holds event message meta data.
 *
 * You can attach various types of objects (vehicle, person, face, etc.)
 * to an event by setting a pointer to the object in @a extMsg.
 *
 * Similarly, you can attach a custom object to an event by setting a pointer to the object in @a extMsg.
 * A custom object must be handled by the metadata parsing module accordingly.
 */
typedef struct NvDsEventMsgMeta {
  /** Holds the event's type. */
  NvDsEventType type;
  /** Holds the object's type. */
  NvDsObjectType objType;
  /** Holds the object's bounding box. */
  NvDsRect bbox;
  /** Holds the object's geolocation. */
  NvDsGeoLocation location;
  /** Holds the object's coordinates. */
  NvDsCoordinate coordinate;
  /** Holds the object's signature. */
  NvDsObjectSignature objSignature;
  /** Holds the object's class ID. */
  gint objClassId;
  /** Holds the ID of the sensor that generated the event. */
  gint sensorId;
  /** Holds the ID of the analytics module that generated the event. */
  gint moduleId;
  /** Holds the ID of the place related to the object. */
  gint placeId;
  /** Holds the ID of the component (plugin) that generated this event. */
  gint componentId;
  /** Holds the video frame ID of this event. */
  gint frameId;
  /** Holds the confidence level of the inference. */
  gdouble confidence;
  /** Holds the object's tracking ID. */
  guint64 trackingId;
  /** Holds a pointer to the generated event's timestamp. */
  gchar *ts;
  /** Holds a pointer to the detected or inferred object's ID. */
  gchar *objectId;
  /** Holds a pointer to a string containing the sensor's identity. */
  gchar *sensorStr;
  /** Holds a pointer to a string containing other attributes associated with
   the object. */

  SourceTimePair entries[MAX_ENTRIES];
  int count;
  NvDsBbox3dObject *bbox3d;
  NvDsLLMObject *llm;
  gint num_interactions;
  NvDsInteractionObject *interactions;
  gint num_congestions;
  NvDsCongestionObject *congestions;
  gint num_fov;
  NvDsTypeMetricsObject *fov;
  NvDsSegmentationObject *segmentation;

  gchar *otherAttrs;
  /** Holds a pointer to the name of the video file. */
  gchar *videoPath;
  /** Holds a pointer to event message meta data. This can be used to hold
   data that can't be accommodated in the existing fields, or an associated
   object (representing a vehicle, person, face, etc.). */
  gpointer extMsg;
  /** Holds the size of the custom object at @a extMsg. */
  guint extMsgSize;
  /** Holds the object's pose information */
  NvDsJoints pose;
  /** Holds the object's embedding information */
  NvDsEmbedding embedding;
  /** Holds the flag for whether single view 3D tracking metadata is available. */
  bool has3DTracking;
  /** Holds the single view 3D tracking metadata if available. */
  NvDs3DTracking singleView3DTracking;
} NvDsEventMsgMeta;

/**
 * Holds event information.
 */
typedef struct _NvDsEvent {
  /** Holds the type of event. */
  NvDsEventType eventType;
  /** Holds a pointer to event metadata. */
  NvDsEventMsgMeta *metadata;
} NvDsEvent;

/**
 * Holds data for any user defined custom message to be attached to the payload
 * message : custom message to be attached
 * size    : size of the custom message
 */
typedef struct _NvDsCustomMsgInfo {
  void *message;
  guint size;
}NvDsCustomMsgInfo;

/**
 * Holds payload metadata.
 */
typedef struct NvDsPayload {
  /** Holds a pointer to the payload. */
  gpointer payload;
  /** Holds the size of the payload. */
  guint payloadSize;
  /** Holds the ID of the component (plugin) which attached the payload
   (optional). */
  guint componentId;
} NvDsPayload;

#ifdef __cplusplus
}
#endif
#endif /* NVDSMETA_H_ */

/** @} */
