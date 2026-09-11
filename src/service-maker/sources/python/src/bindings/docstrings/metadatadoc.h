/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

namespace pydeepstreamdoc
{
    namespace nvmeta
    {
        namespace BatchMetaDoc
        {
            constexpr const char* descr = R"pydeepstream(
                Holds information about a formed batch containing frames from different sources.

                NOTE: Both Video and Audio metadata uses the same :class:`BatchMetadata` type.

                NOTE: Audio batch metadata is formed within nvinferaudio plugin and will not be corresponding to any one buffer output from nvinferaudio.
                The batch metadata for audio is attached to the last input buffer when the audio batch buffering reach configurable threshold (audio frame length)
                and this is when inference output is available.

                :ivar n_frames: *int* Number of frames present in the current batch.
                :ivar is_audio: *bool* True when the batch contains :class:`AudioFrameMetadata` entries.
                :ivar frame_items: *iterator* Iterator over items of type :class:`FrameMetadata` in use in the current batch.
                :ivar audio_frame_items: *iterator* Iterator over items of type :class:`AudioFrameMetadata` in use in the current audio batch.
                :ivar preprocess_batch_items: *iterator* Iterator over preprocess batch metadata in use in the current batch as items of type :class:`UserMetadata`
                    Each item *user_meta* can be called as :class:`PreprocessBatchUserMetadata` by calling *user_meta.as_preprocess_batch()*.)pydeepstream";
            constexpr const char* acquire_display_meta = R"pydeepstream(Acquires a default :class:`DisplayMetadata` object from the pool.)pydeepstream";
            constexpr const char* acquire_frame_meta = R"pydeepstream(Acquires a default :class:`FrameMetadata` object from the pool.)pydeepstream";
            constexpr const char* acquire_object_meta = R"pydeepstream(Acquires a default :class:`ObjectMeta` object from the pool.)pydeepstream";
            constexpr const char* acquire_event_message_meta = R"pydeepstream(Acquires a default :class:`EventMessageUserMetadata` object from the pool.)pydeepstream";
            constexpr const char* acquire_preprocess_batch_meta = R"pydeepstream(Acquires a default :class:`PreprocessBatchUserMetadata` object from the pool.)pydeepstream";
            constexpr const char* acquire_user_meta = R"pydeepstream(Acquires a :class:`UserMetadata` object from the batch meta's user meta pool.

                The acquired user meta can be filled with custom data and then attached to a batch, frame, or object metadata.)pydeepstream";
            constexpr const char* acquire_obj_reid_meta = R"pydeepstream(Acquires an :class:`ObjectReidUserMetadata` object from the batch meta's user meta pool.

                The acquired ReID metadata can be filled with feature vectors and attached to object metadata
                for use in re-identification tasks.

                :returns: :class:`ObjectReidUserMetadata` object ready to be configured.)pydeepstream";
            constexpr const char* user_meta_items = R"pydeepstream(
                Gets an iterator over user metadata items of a specific type attached to the batch.

                This generic method allows accessing any user metadata type without requiring
                specific bindings for each type.

                :arg meta_type: *int*, The NvDsMetaType value (e.g., NVDS_TRACKER_OBJ_REID_META).
                :returns: Iterator over :class:`UserMetadata` items matching the specified type.)pydeepstream";
            constexpr const char* append = R"pydeepstream(Append a :class:`PreprocessBatchUserMetadata` object to the current batch metadata.)pydeepstream";
            constexpr const char* extract = R"pydeepstream(
                Extracts metadata from the batch into a list of dictionaries.

                This method efficiently extracts all metadata (bounding boxes, labels, confidences, segmentation maps, audio labels, etc.)
                from all frames in the batch. The extraction, data copying, and type conversions are performed in C++ for optimal performance.

                :arg max_items: *int* - Maximum number of items in the returned list. The list will have this capacity,
                    indexed by frame *pad_index*.

                :returns: *list of dict* - A list with size equal to *max_items*. Each element corresponds to a frame
                    indexed by *pad_index*. Elements without data will be *None*.

                Each dictionary contains:
                    - **shape**: *list of int* - Frame dimensions [height, width]
                    - **bboxes**: *list of list of int* - Bounding boxes in [left, top, right, bottom] format
                    - **probs**: *list of float* - Confidence scores for each detected object
                    - **labels**: *list of list of str* - Labels for each object (including classifier outputs)
                    - **objects**: *list of int* - Object IDs for tracking
                    - **seg_maps**: *list of numpy.ndarray* - Segmentation maps (instance masks or semantic segmentation) as int arrays
                    - **timestamp**: *int* - Frame presentation timestamp (PTS)

                Audio dictionaries also include **media_type**, **sample_rate**, **num_channels**,
                **num_samples_per_frame**, **format**, **layout**, **infer_done**, **class_id**,
                **class_label**, and **confidence**.
            )pydeepstream";
        }

        namespace FrameMetaDoc
        {
            constexpr const char* descr = R"pydeepstream(
                Holds metadata for a frame in a batch.

                :ivar frame_number: *int*, Current frame number of the source.
                :ivar pad_index: *int*, Pad or port index of stream muxer component for the frame in the batch.
                :ivar batch_id: *int*, Location of the frame in the batch.
                :ivar source_id: *int*, ID of the source to which the frame belongs, generated through source config
                :ivar source_width: *int*, Original width of the frame at input to Gst-streammux.
                :ivar source_height: *int*, Original height of the frame at input to Gst-streammux.
                :ivar pipeline_width: *int*, Width of the frame after batching.
                :ivar pipeline_height: *int*, Height of the frame after batching.
                :ivar buffer_pts: *int*, Presentation timestamp (PTS) of the frame.
                :ivar ntp_timestamp: *int*, Network Time Protocol (NTP) timestamp.
                :ivar object_items: *iterator* Iterator over items of type :class:`ObjectMetadata` in use in the current frame.
                :ivar tensor_items: *iterator* Iterator over items of type :class:`UserMetadata` in use in the current frame as items of type :class:`TensorOutputUserMetadata`
                    Each item *user_meta* can be called as :class:`TensorOutputUserMetadata` by calling *user_meta.as_tensor_output()*.
                :ivar segmentation_items: *iterator* Iterator over items of type :class:`UserMetadata` in use in the current frame as items of type :class:`SegmentationUserMetadata`
                    Each item *user_meta* can be called as :class:`SegmentationUserMetadata` by calling *user_meta.as_segmentation()*.)pydeepstream";

            constexpr const char* append = R"pydeepstream(Append a :class:`DisplayMetadata`, :class:`EventMessageUserMetadata`, :class:`ObjectMetadata` or :class:`TensorOutputUserMetadata` object to the current frame metadata.)pydeepstream";
            constexpr const char* user_meta_items = R"pydeepstream(
                Gets an iterator over user metadata items of a specific type attached to the frame.

                This generic method allows accessing any user metadata type without requiring
                specific bindings for each type.

                :arg meta_type: *int*, The NvDsMetaType value (e.g., NVDS_TRACKER_OBJ_REID_META).
                :returns: Iterator over :class:`UserMetadata` items matching the specified type.)pydeepstream";
        }

        namespace AudioFrameMetaDoc
        {
            constexpr const char* descr = R"pydeepstream(
                Holds metadata for an audio frame in a batch.

                :ivar frame_number: *int*, Current frame number of the source.
                :ivar pad_index: *int*, Pad or port index of stream muxer component for the audio frame in the batch.
                :ivar batch_id: *int*, Location of the audio frame in the batch.
                :ivar source_id: *int*, ID of the source to which the audio frame belongs.
                :ivar sample_rate: *int*, Audio sample rate.
                :ivar num_channels: *int*, Number of audio channels.
                :ivar num_samples_per_frame: *int*, Number of audio samples in this frame.
                :ivar format: *int*, Audio sample format.
                :ivar layout: *int*, Audio channel layout.
                :ivar buffer_pts: *int*, Presentation timestamp (PTS) of the audio frame.
                :ivar ntp_timestamp: *int*, Network Time Protocol (NTP) timestamp.
                :ivar infer_done: *bool*, Whether inference has been performed on this audio frame.
                :ivar class_id: *int*, Class ID of the last classified event.
                :ivar confidence: *float*, Confidence of the last classified event.
                :ivar class_label: *str*, Label of the last classified event.
                :ivar classifier_items: *iterator* Iterator over :class:`ClassifierMetadata` items attached by nvinferaudio.
                :ivar tensor_items: *iterator* Iterator over tensor output user metadata items attached to the audio frame.)pydeepstream";
            constexpr const char* append = R"pydeepstream(Append a :class:`UserMetadata` object to the current audio frame metadata.)pydeepstream";
            constexpr const char* user_meta_items = R"pydeepstream(
                Gets an iterator over user metadata items of a specific type attached to the audio frame.

                :arg meta_type: *int*, The NvDsMetaType value.
                :returns: Iterator over :class:`UserMetadata` items matching the specified type.)pydeepstream";
        }

        namespace ObjectMetaDoc
        {
            constexpr const char* descr = R"pydeepstream(
                Holds information of object metadata in the frame.

                :ivar class_id: *int*, Index of the object class infered by the primary detector/classifier
                :ivar object_id: *int*, Unique ID for tracking the object.
                :ivar unique_component_id: *int*, Unique id of the component that generates the object metadata.
                :ivar confidence: *float*, Holds a confidence value for the object, set by the inference component.
                    Confidence will be set to -0.1, if "Group Rectangles" mode of clustering is chosen since the algorithm does not preserve confidence values.
                    Also, for objects found by tracker and not inference component, confidence will be set to -0.1
                :ivar tracker_confidence: *float*, Holds a confidence value for the object, set by the tracker component.
                :ivar rect_params: :class:`NvOSD_RectParams`, Structure containing the positional parameters of the object in the frame.
                    Can also be used to overlay borders / semi-transparent boxes on top of objects. See :class:`NvOSD_RectParams`
                :ivar mask_params: :class:`NvOSD_MaskParams`, Mask parameters for the object. This mask is overlaid over the object See :class:`NvOSD_MaskParams`
                :ivar text_params: :class:`NvOSD_TextParams`, Text for describing the object. See :class:`NvOSD_TextParams`
                :ivar label: *str*, Label describing the class of the detected object.
                :ivar classifier_items: *iterator* Iterator over items of type :class:`ClassifierMetadata` in use in the current object.
                :ivar tensor_items: *iterator* Iterator over items of type :class:`UserMetadata` in use in the current object as items of type :class:`TensorOutputUserMetadata`
                    Each item *user_meta* can be called as :class:`TensorOutputUserMetadata` by calling *user_meta.as_tensor_output()*.
                :ivar obj_reid_items: *iterator* Iterator over ReID user metadata items of type :class:`UserMetadata`.
                    Each item *user_meta* can be cast to :class:`ObjectReidUserMetadata` by calling *user_meta.as_obj_reid()*.)pydeepstream";

            constexpr const char* append = R"pydeepstream(Append a :class:`UserMetadata` object to the current object metadata.)pydeepstream";
            constexpr const char* user_meta_items = R"pydeepstream(
                Gets an iterator over user metadata items of a specific type attached to the object.

                This generic method allows accessing any user metadata type without requiring
                specific bindings for each type.

                :arg meta_type: *int*, The NvDsMetaType value (e.g., NVDS_TRACKER_OBJ_REID_META).
                :returns: Iterator over :class:`UserMetadata` items matching the specified type.)pydeepstream";
        }

        namespace LabelInfoDoc
        {
            constexpr const char* descr = R"pydeepstream(
                Holds all fields of a single label entry produced by a classifier.

                :ivar label: *str*, Label string of the result.
                :ivar class_id: *int*, Class ID of the best result.
                :ivar prob: *float*, Probability of the best result.
                :ivar label_id: *int*, Label ID for multi-label classifiers.
                :ivar num_classes: *int*, Number of classes for this label.)pydeepstream";
        }

        namespace ClassifierMetaDoc
        {
            constexpr const char* descr = R"pydeepstream(
                Holds classifier metadata for an object.

                :ivar n_labels: *int*, Number of output labels of the classifier.
                :ivar unique_component_id: *int*, Unique id of the component that generates the classifier metadata.
                :ivar classifier_type: *str*, Type of the classifier. Empty string if not set.)pydeepstream";

            constexpr const char* classifier_type = R"pydeepstream(Type of the classifier. Returns an empty string if the classifier type was not set.)pydeepstream";
            constexpr const char* get_label_info = R"pydeepstream(Get all fields of the nth label as a LabelInfo object.)pydeepstream";
            constexpr const char* get_label = R"pydeepstream(Get the label string of the nth label.)pydeepstream";
        }

        namespace DisplayMetaDoc
        {
            constexpr const char* descr = R"pydeepstream(
                Holds information of display metadata that user can specify in the frame.

                :ivar n_rects: *int*, Number of rectangles present in display meta.
                :ivar n_labels: *int*, Number of labels/strings present in display meta.
                :ivar n_lines: *int*, Number of lines present in display meta.
                :ivar n_arrows: *int*, Number of arrows present in display meta.
                :ivar n_circles: *int*, Number of circles present in display meta.)pydeepstream";
            constexpr const char* add_text = R"pydeepstream(Adds a :class:`Text` object to the display meta.)pydeepstream";
            constexpr const char* add_rect = R"pydeepstream(Adds a :class:`NvOSD_RectParams` object to the display meta.)pydeepstream";
            constexpr const char* add_line = R"pydeepstream(Adds a :class:`NvOSD_LineParams` object to the display meta.)pydeepstream";
            constexpr const char* add_arrow = R"pydeepstream(Adds a :class:`NvOSD_ArrowParams` object to the display meta.)pydeepstream";
            constexpr const char* add_circle = R"pydeepstream(Adds a :class:`NvOSD_CircleParams` object to the display meta.)pydeepstream";
        }

        namespace UserMetaDoc
        {
            constexpr const char* descr = R"pydeepstream(
                Holds the user metadata. Can be casted to :class:`TensorOutputUserMetadata`, :class:`SegmentationUserMetadata`, or :class:`PreprocessBatchUserMetadata`.

                :ivar meta_type: *int*, The metadata type identifier for this user metadata.)pydeepstream";
            constexpr const char* set_user_data_json = R"pydeepstream(
                Sets a Python object as JSON-serialized user data for this metadata.

                The Python object is serialized to a JSON string and stored as a C string (char*).
                This allows C/C++ components to access the data by reading the string directly
                and parsing it with any JSON library.

                :arg data: Any JSON-serializable Python object (dict, list, str, int, float, bool, None).
                :arg meta_type: *int*, The metadata type identifier (e.g., NVDS_USER_META starting from 4096+4096+1).

                Example C/C++ access:
                    const char* json_str = (const char*)user_meta->user_meta_data;
                    // Parse with nlohmann/json, rapidjson, etc.)pydeepstream";
            constexpr const char* get_user_data_json = R"pydeepstream(
                Gets the JSON-serialized user data as a Python object.

                Deserializes the JSON string stored in user_meta_data back to a Python object.

                :returns: The deserialized Python object, or None if no data is set.)pydeepstream";
            constexpr const char* as_tensor_output = R"pydeepstream(Casts the user metadata to :class:`TensorOutputUserMetadata` object.)pydeepstream";
            constexpr const char* as_segmentation = R"pydeepstream(Casts the user metadata to :class:`SegmentationUserMetadata` object.)pydeepstream";
            constexpr const char* as_preprocess_batch = R"pydeepstream(Casts the user metadata to :class:`PreprocessBatchUserMetadata` object.)pydeepstream";
            constexpr const char* as_nvdsanalytics_obj = R"pydeepstream(Casts the user metadata to :class:`NvDsAnalyticsObjInfo` object.)pydeepstream";
            constexpr const char* as_nvdsanalytics_frame = R"pydeepstream(Casts the user metadata to :class:`NvDsAnalyticsFrameMeta` object.)pydeepstream";
            constexpr const char* as_structure_str = R"pydeepstream(Returns the string representation of the underlying GstStructure of the Structure user metadata.)pydeepstream";
            constexpr const char* as_obj_reid = R"pydeepstream(Casts the user metadata to :class:`ObjectReidUserMetadata` object for ReID features.)pydeepstream";
        }

        namespace ObjectReidUserMetaDoc
        {
            constexpr const char* descr = R"pydeepstream(
                User metadata for object Re-Identification (ReID) features.

                This metadata type (NVDS_TRACKER_OBJ_REID_META) holds ReID feature vectors
                that can be used to identify/re-identify objects across frames.

                The structure is interoperable with C/C++ components using NvDsObjReid.

                :ivar feature_size: *int*, Length of the ReID feature vector.
                :ivar feature_vector: *numpy.ndarray*, The ReID feature vector as float array.)pydeepstream";
            constexpr const char* set_feature_vector = R"pydeepstream(
                Sets the ReID feature vector.

                :arg features: numpy array of float32 values representing the ReID embedding.)pydeepstream";
        }

        namespace EventMessageUserMetaDoc
        {
            constexpr const char* descr = R"pydeepstream(
                User metadata for event messages.)pydeepstream";
            constexpr const char* generate = R"pydeepstream(
                Fills the event message user metadata with the given :class:`ObjectMetadata`, :class:`FrameMetadata`, sensor, uri, and labels.

                :arg object_meta: :class:`ObjectMetadata` to be used to fill the event message user metadata.
                :arg frame_meta: :class:`FrameMetadata` to be used to fill the event message user metadata.
                :arg sensor: *str*, optional, Sensor name.
                :arg uri: *str*, optional, URI of the source, if applicable.
                :arg labels: *list of str*, optional, List of labels for custom model.)pydeepstream";
            constexpr const char* generate_audio = R"pydeepstream(
                Fills the event message user metadata from :class:`AudioFrameMetadata`.

                This copies the audio classification label, class id, confidence, source id,
                and timestamp into :class:`NvDsEventMsgMeta` so it can be consumed by the
                DeepStream audio message converter.

                :arg audio_frame_meta: :class:`AudioFrameMetadata` to be used to fill the event message user metadata.
                :arg sensor: *str*, optional, Sensor name.
                :arg uri: *str*, optional, URI of the source, if applicable.)pydeepstream";
        }

        namespace TensorOutputUserMetaDoc
        {
            constexpr const char* descr = R"pydeepstream(
                User metadata for tensor output.

                :ivar unique_id: *int*, Unique id of the component that generates tensor output.)pydeepstream";
            constexpr const char* get_layers = R"pydeepstream(Returns a map of tensors where key is the layer name and value is tensor output.)pydeepstream";
        }

        namespace SegmentationUserMetaDoc
        {
            constexpr const char* descr = R"pydeepstream(
                User metadata for segmentation output.

                :ivar unique_id: *int*, Unique id of the component that generates segmentation output.
                :ivar classes: *int*, Number of classes in the segmentation output.
                :ivar width: *int*, Width of the segmentation mask.
                :ivar height: *int*, Height of the segmentation mask.
                :ivar class_map: *list of int* Class map of the segmentation output.
                :ivar class_probabilities_map: *list of float* Class probabilities map of the segmentation output.)pydeepstream";
        }

        namespace RoiMetaDoc
        {
            constexpr const char* descr = R"pydeepstream(
                Holds information of a region of interest.

                :ivar roi: :class:`NvOSD_RectParams`, Region of interest bounding box.
                :ivar frame_meta: :class:`FrameMetadata`, Frame metadata which contains the region of interest.
                :ivar tensor_items: *iterator* Iterator over items of type :class:`UserMetadata` in use in the current frame as items of type :class:`TensorOutputUserMetadata`
                    Each item *user_meta* can be called as :class:`TensorOutputUserMetadata` by calling *user_meta.as_tensor_output()*.
                :ivar classifier_items: *iterator* Iterator over items of type :class:`ClassifierMetadata` in use in the current object.
                    Each item *classifier_meta* can be called as :class:`ClassifierMetadata` by calling *classifier_meta.as_classifier()*.
                :ivar segmentation_items: *iterator* Iterator over items of type :class:`UserMetadata` in use in the current object as items of type :class:`SegmentationUserMetadata`
                    Each item *user_meta* can be called as :class:`SegmentationUserMetadata` by calling *user_meta.as_segmentation()*.
                :ivar object_meta: :class:`ObjectMetadata`, Object metadata of the region of interest.)pydeepstream";
        }

        namespace PreprocessTensorMetaDoc
        {
            constexpr const char* descr = R"pydeepstream(
                Tensor meta containing prepared tensor and related info inside preprocess user meta which is attached at batch level.

                :ivar name: *str*, Name of the preprocess tensor.
                :ivar meta_id: *int*, Meta id of the preprocess tensor.
                :ivar tensor: :class:`Tensor`, Tensor object.)pydeepstream";
        }

        namespace PreprocessBatchUserMetaDoc
            {
                constexpr const char* descr = R"pydeepstream(
                    Batch metadata structure for preprocessing operations.

                    :ivar rois: *list of :class:`RoiMetadata`*, List of roi metadata.
                    :ivar preprocess_tensor_meta: :class:`PreprocessTensorMetadata`, Preprocess tensor metadata.)pydeepstream";

            constexpr const char* set_preprocessed_tensor = R"pydeepstream(
                Initialize the preprocess tensor meta from given name, meta_id, unique_id, and tensor.

                :arg name: *str*, Name of the preprocess tensor.
                :arg meta_id: *int*, Meta id of the preprocess tensor.
                :arg unique_id: *int*, Unique target ID.
                :arg tensor: :class:`Tensor`, Tensor object.

                :returns: True if the preprocess tensor meta is initialized successfully, False otherwise.)pydeepstream";
            constexpr const char* extract = R"pydeepstream(
                Extracts metadata from preprocess batch ROIs into a list of dictionaries.

                This method efficiently extracts metadata from all ROIs in the preprocess batch, including segmentation maps,
                object detections, and classifier outputs. The extraction and data copying are performed in C++ for optimal performance.

                :arg max_items: *int* - Maximum number of items in the returned list. The list will have this capacity,
                    indexed by frame *pad_index*.

                :returns: *list of dict* - A list with size equal to *max_items*. Each element corresponds to the
                    metadata extracted from ROIs belonging to that frame. Elements without data will be *None*.

                Each dictionary contains:
                    - **shape**: *list of int* - Frame dimensions [height, width]
                    - **bboxes**: *list of list of int* - Bounding boxes in [left, top, right, bottom] format
                    - **probs**: *list of float* - Confidence scores for each detected object
                    - **labels**: *list of list of str* - Labels from objects and ROI-level classifiers
                    - **objects**: *list of int* - Object IDs for tracking
                    - **seg_maps**: *list of numpy.ndarray* - Segmentation maps as int arrays
                    - **timestamp**: *int* - Frame presentation timestamp (PTS)
            )pydeepstream";
        }

        namespace AnalyticsObjInfoDoc
        {
            constexpr const char* descr = R"pydeepstream(
                Holds a set of nvdsanalytics object level metadata.

                :ivar roi_status: *list of str*, Holds the array of ROI labels in which object is present.
                :ivar oc_status: *list of str*, Holds the array of OverCrowding labels in which object is present.
                :ivar lc_status: *list of str*, Holds the array of line crossing labels which object has crossed.
                :ivar dir_status: *str*, Holds the direction string for the tracked object.
                :ivar unique_id: *int*, Holds unique identifier for nvdsanalytics instance.
                :ivar obj_status: *str*, Holds the status string for the tracked object.
            )pydeepstream";
        }

        namespace AnalyticsFrameMetaDoc
        {
            constexpr const char* descr = R"pydeepstream(
                Holds a set of nvdsanalytics framelevel metadata.

                :ivar oc_status: *dict of str to bool*, Holds a map of boolean status of overcrowding for configured ROIs, which can be accessed using key, value pair; where key is the ROI label.
                :ivar obj_in_roi_cnt: *dict of str to int*, Holds a map of total count of valid objects in ROI  for configured ROIs, which can be accessed using key, value pair; where key is the ROI label.
                :ivar obj_lc_curr_cnt: *dict of str to int*, Holds a map of total count of Line crossing in current frame for configured lines, which can be accessed using key, value pair; where key is the ROI label.
                :ivar obj_lc_cum_cnt: *dict of str to int*, Holds a map of total cumulative count of Line crossing  for configured lines, which can be accessed using key, value pair; where key is the ROI label.
                :ivar unique_id: *int*, Holds unique identifier for nvdsanalytics instance.
                :ivar obj_cnt: *dict of int to int*, Holds a map of total count of objects for each class ID, which can be accessed using key, value pair; where key is class ID.
            )pydeepstream";
        }
    }
}
