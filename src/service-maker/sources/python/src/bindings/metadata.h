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

#include "metadatadoc.h"
#include <gst/gst.h>

struct BBox {
    float x;
    float y;
    float w;
    float h;
};

template<class T, class ITERATOR>
class IteratorWrapper {
 public:
   IteratorWrapper() {}

   T& operator*() const {
     return *(*iterator_);
   }

   IteratorWrapper& operator++() {
     iterator_->next();
     return *this;
   }

   bool operator != (const IteratorWrapper& other) const {
     return *iterator_ != *other.iterator_;
   }

   bool operator == (const IteratorWrapper& other) const {
     return *iterator_ == *other.iterator_;
   }

   ITERATOR& get() { return iterator_; }

 protected:
   ITERATOR iterator_;
};

template<class T, class META_CLS, class ITERATOR>
py::iterable generate_iterator(META_CLS& metadata) {
    if (!metadata) {
        return py::cast(std::vector<T>());
    }
    IteratorWrapper<T, ITERATOR> begin;
    IteratorWrapper<T, ITERATOR> end;
    metadata.initiateIterator(begin.get());
    metadata.initiateIterator(end.get());
    while (!end.get()->done()) ++end;

    return py::make_iterator(std::move(begin), std::move(end));
}

template<class META_CLS, int N>
py::iterable generate_usermeta_iterator(META_CLS& metadata) {
    if (!metadata) {
        return py::cast(std::vector<UserMetadata>());
    }
    IteratorWrapper<UserMetadata, UserMetadata::Iterator> begin;
    IteratorWrapper<UserMetadata, UserMetadata::Iterator> end;
    metadata.initiateIterator(begin.get(), N);
    metadata.initiateIterator(end.get(), N);
    while (!end.get()->done()) ++end;

    return py::make_iterator(std::move(begin), std::move(end));
}

template<class META_CLS>
std::vector<deepstream::UserMetadata> retrieve_structure_usermeta(META_CLS& self, const std::string& type_name) {
    int type = NVDS_START_USER_META + g_quark_from_string(type_name.c_str());
    std::vector<deepstream::UserMetadata> results = {};
    self.iterate([type, &results](const UserMetadata& user_meta) {
        StructureUserMetadata meta(user_meta);
        if (meta.getStructureUserMetadata() && meta.getStructureUserMetatype() == type) {
            results.emplace_back(user_meta);
        }
    }, type);
    return results;
}

py::list preprocess_batch_metadata_extract(PreprocessBatchUserMetadata& self, unsigned int max_items);
py::list batch_metadata_extract(BatchMetadata& self, unsigned int max_items);
py::list audio_batch_metadata_extract(BatchMetadata& self, unsigned int max_items);

py::list preprocess_batch_metadata_extract(PreprocessBatchUserMetadata& self, unsigned int max_items) {
    // Check if preprocess batch metadata is valid
    if (!self) {
        return py::list();
    }

    // Get ROIs from preprocess batch
    std::vector<RoiMetadata> rois = self.getRois();
    if (rois.empty()) {
        return py::list();
    }

    // Create list with capacity of max_items
    py::list received(max_items);

    // Initialize all elements with None
    for (unsigned int i = 0; i < max_items; i++) {
        received[i] = py::none();
    }

    // Iterate through ROIs
    for (const RoiMetadata& roi : rois) {
        py::dict metadata;
        std::vector<int> shape;
        std::vector<std::vector<int>> bboxes;
        std::vector<float> probs;
        std::vector<std::vector<std::string>> labels;
        std::vector<py::array_t<int>> seg_maps;
        std::vector<int> objects;
        int64_t timestamp = 0;

        FrameMetadata frame_meta = roi.frameMetadata();
        if (!frame_meta) continue;

        // Extract shape from frame metadata (pipeline dimensions)
        shape = {static_cast<int>(frame_meta.pipelineHeight()),
                 static_cast<int>(frame_meta.pipelineWidth())};

        // Iterate through segmentation user metadata in ROI
        roi.iterate([&](const UserMetadata& user_meta) {
            SegmentationUserMetadata seg_meta(user_meta);

            if (seg_meta) {
                shape = {static_cast<int>(seg_meta.getHeight()),
                        static_cast<int>(seg_meta.getWidth())};

                const int* class_map_data = seg_meta.getClassMap();
                if (class_map_data) {
                    // Allocate new array and copy data
                    unsigned int height = seg_meta.getHeight();
                    unsigned int width = seg_meta.getWidth();
                    py::array_t<int> class_map({height, width});
                    auto buf = class_map.mutable_unchecked<2>();

                    // Copy data in C++ for performance
                    for (unsigned int i = 0; i < height; i++) {
                        for (unsigned int j = 0; j < width; j++) {
                            buf(i, j) = class_map_data[i * width + j];
                        }
                    }
                    seg_maps.push_back(class_map);
                }
            }
        }, NVDSINFER_SEGMENTATION_META);

        // Iterate through objects in the ROI's frame
        frame_meta.iterate([&](const ObjectMetadata& object_meta) {
            // Extract labels
            std::vector<std::string> obj_labels;
            if (!object_meta.label().empty()) {
                obj_labels.push_back(object_meta.label());
            }

            // Extract bbox coordinates
            const NvOSD_RectParams& rect = object_meta.rectParams();
            int left = static_cast<int>(rect.left);
            int top = static_cast<int>(rect.top);
            int width = static_cast<int>(rect.width);
            int height = static_cast<int>(rect.height);

            // Set shape
            shape = {static_cast<int>(frame_meta.pipelineHeight()),
                     static_cast<int>(frame_meta.pipelineWidth())};

            // Add bbox in [left, top, left+width, top+height] format
            bboxes.push_back({left, top, left + width, top + height});

            // Add confidence
            probs.push_back(object_meta.confidence());

            // Extract classifier labels
            object_meta.iterate([&](const ClassifierMetadata& classifier) {
                for (unsigned int i = 0; i < classifier.nLabels(); i++) {
                    obj_labels.push_back(classifier.getLabel(i));
                }
            });

            labels.push_back(obj_labels);
            objects.push_back(static_cast<int>(object_meta.objectId()));
        });

        // Iterate through classifier metadata at ROI level
        roi.iterate([&](const ClassifierMetadata& classifier) {
            std::vector<std::string> cls_labels;
            for (unsigned int i = 0; i < classifier.nLabels(); i++) {
                cls_labels.push_back(classifier.getLabel(i));
            }
            if (!cls_labels.empty()) {
                labels.push_back(cls_labels);
            }
        });

        // Set timestamp
        timestamp = static_cast<int64_t>(frame_meta.bufferPTS());

        // Build dictionary
        metadata["shape"] = shape;
        metadata["bboxes"] = bboxes;
        metadata["probs"] = probs;
        metadata["labels"] = labels;
        metadata["seg_maps"] = seg_maps;
        metadata["objects"] = objects;
        metadata["timestamp"] = timestamp;

        // Store in list indexed by pad_index
        unsigned int pad_index = frame_meta.padIndex();
        if (pad_index < max_items) {
            received[pad_index] = metadata;
        }
    }

    return received;
}

py::list batch_metadata_extract(BatchMetadata& self, unsigned int max_items) {
    // Check if batch metadata is valid
    if (!self) {
        return py::list();  // Return empty list if batch metadata is null
    }

    if (self.isAudioBatch()) {
        return audio_batch_metadata_extract(self, max_items);
    }

    // Create list with capacity of max_items
    py::list received(max_items);

    // Initialize all elements with None
    for (unsigned int i = 0; i < max_items; i++) {
        received[i] = py::none();
    }

    // Iterate through frames in batch
    self.iterate([&](const FrameMetadata& frame_meta) {
        py::dict metadata;
        std::vector<int> shape;
        std::vector<std::vector<int>> bboxes;
        std::vector<float> probs;
        std::vector<std::vector<std::string>> labels;
        std::vector<py::array_t<int>> seg_maps;
        std::vector<int> objects;
        int64_t timestamp = 0;

        // Extract shape from frame metadata (pipeline dimensions)
        shape = {static_cast<int>(frame_meta.pipelineHeight()),
                 static_cast<int>(frame_meta.pipelineWidth())};

        // Iterate through objects in frame
        frame_meta.iterate([&](const ObjectMetadata& object_meta) {
            // Extract labels
            std::vector<std::string> obj_labels;
            if (!object_meta.label().empty()) {
                obj_labels.push_back(object_meta.label());
            }

            // Extract bbox coordinates
            const NvOSD_RectParams& rect = object_meta.rectParams();
            int left = static_cast<int>(rect.left);
            int top = static_cast<int>(rect.top);
            int width = static_cast<int>(rect.width);
            int height = static_cast<int>(rect.height);

            // Add bbox in [left, top, left+width, top+height] format
            bboxes.push_back({left, top, left + width, top + height});

            // Add confidence
            probs.push_back(object_meta.confidence());

            // Extract classifier labels
            object_meta.iterate([&](const ClassifierMetadata& classifier) {
                for (unsigned int i = 0; i < classifier.nLabels(); i++) {
                    obj_labels.push_back(classifier.getLabel(i));
                }
            });

            labels.push_back(obj_labels);
            objects.push_back(static_cast<int>(object_meta.objectId()));

            // Handle instance mask
            const NvOSD_MaskParams& mask_params = object_meta.maskParams();
            if (mask_params.data != nullptr && mask_params.size > 0) {
                // Allocate new array and copy/convert data (for data persistence and type conversion)
                py::array_t<int> mask_array_int({mask_params.height, mask_params.width});
                auto mask_buf = mask_array_int.mutable_unchecked<2>();

                // Copy and convert float to int in C++ for performance
                for (unsigned int i = 0; i < mask_params.height; i++) {
                    for (unsigned int j = 0; j < mask_params.width; j++) {
                        mask_buf(i, j) = static_cast<int>(mask_params.data[i * mask_params.width + j]);
                    }
                }
                seg_maps.push_back(mask_array_int);
            }
        });

        // Iterate through segmentation user metadata
        frame_meta.iterate([&](const UserMetadata& user_meta) {
            SegmentationUserMetadata seg_meta(user_meta);

            if (seg_meta) {
                shape = {static_cast<int>(seg_meta.getHeight()),
                        static_cast<int>(seg_meta.getWidth())};

                const int* class_map_data = seg_meta.getClassMap();
                if (class_map_data) {
                    // Allocate new array and copy data (for data persistence beyond callback)
                    unsigned int height = seg_meta.getHeight();
                    unsigned int width = seg_meta.getWidth();
                    py::array_t<int> class_map({height, width});
                    auto buf = class_map.mutable_unchecked<2>();

                    // Copy data in C++ for performance
                    for (unsigned int i = 0; i < height; i++) {
                        for (unsigned int j = 0; j < width; j++) {
                            buf(i, j) = class_map_data[i * width + j];
                        }
                    }
                    seg_maps.push_back(class_map);
                }
            }
        }, NVDSINFER_SEGMENTATION_META);

        // Set timestamp
        timestamp = static_cast<int64_t>(frame_meta.bufferPTS());

        // Build dictionary
        metadata["shape"] = shape;
        metadata["bboxes"] = bboxes;
        metadata["probs"] = probs;
        metadata["labels"] = labels;
        metadata["seg_maps"] = seg_maps;
        metadata["objects"] = objects;
        metadata["timestamp"] = timestamp;

        // Store in list indexed by pad_index
        unsigned int pad_index = frame_meta.padIndex();
        if (pad_index < max_items) {
            received[pad_index] = metadata;
        }
    });

    return received;
}

py::list audio_batch_metadata_extract(BatchMetadata& self, unsigned int max_items) {
    if (!self) {
        return py::list();
    }

    py::list received(max_items);
    for (unsigned int i = 0; i < max_items; i++) {
        received[i] = py::none();
    }

    self.iterate([&](const AudioFrameMetadata& frame_meta) {
        py::dict metadata;
        std::vector<int> shape = {
            static_cast<int>(frame_meta.numSamplesPerFrame()),
            static_cast<int>(frame_meta.numChannels())
        };
        std::vector<std::vector<int>> bboxes;
        std::vector<float> probs;
        std::vector<std::vector<std::string>> labels;
        std::vector<py::array_t<int>> seg_maps;
        std::vector<int> objects;

        if (!frame_meta.classLabel().empty()) {
            labels.push_back({frame_meta.classLabel()});
            probs.push_back(frame_meta.confidence());
            objects.push_back(frame_meta.classId());
        }

        frame_meta.iterate([&](const ClassifierMetadata& classifier) {
            std::vector<std::string> cls_labels;
            for (unsigned int i = 0; i < classifier.nLabels(); i++) {
                cls_labels.push_back(classifier.getLabel(i));
            }
            if (!cls_labels.empty()) {
                labels.push_back(cls_labels);
            }
        });

        metadata["media_type"] = "audio";
        metadata["shape"] = shape;
        metadata["bboxes"] = bboxes;
        metadata["probs"] = probs;
        metadata["labels"] = labels;
        metadata["seg_maps"] = seg_maps;
        metadata["objects"] = objects;
        metadata["timestamp"] = static_cast<int64_t>(frame_meta.bufferPTS());
        metadata["sample_rate"] = frame_meta.sampleRate();
        metadata["num_channels"] = frame_meta.numChannels();
        metadata["num_samples_per_frame"] = frame_meta.numSamplesPerFrame();
        metadata["format"] = frame_meta.format();
        metadata["layout"] = frame_meta.layout();
        metadata["infer_done"] = frame_meta.inferDone();
        metadata["class_id"] = frame_meta.classId();
        metadata["class_label"] = frame_meta.classLabel();
        metadata["confidence"] = frame_meta.confidence();

        unsigned int pad_index = frame_meta.padIndex();
        if (pad_index < max_items) {
            received[pad_index] = metadata;
        }
    });

    return received;
}

/**
 * @brief Copy function for C string user data
 *
 * Creates a duplicate of the C string.
 */
static void* copy_cstring_user_data(void* data, void* user_data) {
    const char* str = static_cast<const char*>(data);
    return strdup(str);
}

/**
 * @brief Release function for C string user data
 *
 * Frees the C string.
 */
static void release_cstring_user_data(void* data, void* user_data) {
    free(data);
}

void module_metadata_bind(py::module &m);

void module_metadata_bind(py::module &m) {
    py::class_<BatchMetadata>(m, "BatchMetadata", pydeepstreamdoc::nvmeta::BatchMetaDoc::descr)
        .def_property_readonly("n_frames", [](const BatchMetadata& self) {
            return self.nFrames();
        })
        .def_property_readonly("is_audio", &BatchMetadata::isAudioBatch)
        .def_property_readonly("frame_items", [](BatchMetadata& self) -> py::object {
            if (self.isAudioBatch()) {
                return py::cast(std::vector<FrameMetadata>());
            }
            return generate_iterator<FrameMetadata, BatchMetadata, FrameMetadata::Iterator>(self);
        }, py::keep_alive<0, 1>())
        .def_property_readonly("audio_frame_items", [](BatchMetadata& self) -> py::object {
            if (!self.isAudioBatch()) {
                return py::cast(std::vector<AudioFrameMetadata>());
            }
            return generate_iterator<AudioFrameMetadata, BatchMetadata, AudioFrameMetadata::Iterator>(self);
        }, py::keep_alive<0, 1>())
        .def_property_readonly("preprocess_batch_items", [](BatchMetadata& self) {
            return generate_usermeta_iterator<BatchMetadata, NVDS_PREPROCESS_BATCH_META>(self);
        }, py::keep_alive<0, 1>())
        .def("retrieve_structure_usermeta", [](BatchMetadata& self, const std::string& type_name) {
            return retrieve_structure_usermeta<BatchMetadata>(self, type_name);
        })
        .def("user_meta_items", [](BatchMetadata& self, int meta_type) {
            IteratorWrapper<UserMetadata, UserMetadata::Iterator> begin;
            IteratorWrapper<UserMetadata, UserMetadata::Iterator> end;
            self.initiateIterator(begin.get(), meta_type);
            self.initiateIterator(end.get(), meta_type);
            while (!end.get()->done()) ++end;
            return py::make_iterator(std::move(begin), std::move(end));
        }, pydeepstreamdoc::nvmeta::BatchMetaDoc::user_meta_items,
           py::arg("meta_type"), py::keep_alive<0, 1>())
        .def("acquire_display_meta", [](BatchMetadata& self) {
            DisplayMetadata display_meta;
            self.acquire(display_meta);
            // basic initialization for convenience
            return display_meta;
        }, pydeepstreamdoc::nvmeta::BatchMetaDoc::acquire_display_meta)
        .def("acquire_frame_meta", [](BatchMetadata& self) {
            if (self.isAudioBatch()) {
                throw py::type_error("Use audio_frame_items to access audio frame metadata");
            }
            FrameMetadata frame_meta;
            self.acquire(frame_meta);
            // basic initialization for convenience
            return frame_meta;
        }, pydeepstreamdoc::nvmeta::BatchMetaDoc::acquire_frame_meta)
        .def("acquire_object_meta", [](BatchMetadata& self) {
            ObjectMetadata object_meta;
            self.acquire(object_meta);
            // basic initialization for convenience
            return object_meta;
        }, pydeepstreamdoc::nvmeta::BatchMetaDoc::acquire_object_meta)
        .def("acquire_event_message_meta", [](BatchMetadata& self) {
            EventMessageUserMetadata event_meta;
            self.acquire(event_meta);
            return event_meta;
        }, pydeepstreamdoc::nvmeta::BatchMetaDoc::acquire_event_message_meta)
        .def("acquire_preprocess_batch_meta", [](BatchMetadata& self) {
            PreprocessBatchUserMetadata preprocess_meta;
            self.acquire(preprocess_meta);
            return preprocess_meta;
        }, pydeepstreamdoc::nvmeta::BatchMetaDoc::acquire_preprocess_batch_meta)
        .def("acquire_user_meta", [](BatchMetadata& self) {
            UserMetadata user_meta(nullptr);
            self.acquire(user_meta);
            return user_meta;
        }, pydeepstreamdoc::nvmeta::BatchMetaDoc::acquire_user_meta)
        .def("acquire_obj_reid_meta", [](BatchMetadata& self) {
            ObjectReidUserMetadata reid_meta;
            self.acquire(reid_meta);
            return reid_meta;
        }, pydeepstreamdoc::nvmeta::BatchMetaDoc::acquire_obj_reid_meta)
        .def("append", [](BatchMetadata& self, py::object object) {
            if (py::isinstance<PreprocessBatchUserMetadata>(object)) {
                self.append(object.cast<PreprocessBatchUserMetadata>());
            } else {
                throw py::type_error("Invalid type for BatchMetadata.append");
            }
        }, pydeepstreamdoc::nvmeta::BatchMetaDoc::append)
        .def("extract", &batch_metadata_extract, pydeepstreamdoc::nvmeta::BatchMetaDoc::extract, py::arg("max_items"))
        .def("__repr__", [](BatchMetadata& self) {
            std::string frame_items_repr = "[]";
            std::string audio_frame_items_repr = "[]";
            if (self.isAudioBatch()) {
                audio_frame_items_repr =
                    py::repr(generate_iterator<AudioFrameMetadata, BatchMetadata, AudioFrameMetadata::Iterator>(self)).cast<std::string>();
            } else {
                frame_items_repr =
                    py::repr(generate_iterator<FrameMetadata, BatchMetadata, FrameMetadata::Iterator>(self)).cast<std::string>();
            }
            return "BatchMetadata(n_frames=" + std::to_string(self.nFrames()) +
                ", is_audio=" + std::string(self.isAudioBatch() ? "True" : "False") +
                ", frame_items=" + frame_items_repr +
                ", audio_frame_items=" + audio_frame_items_repr +
                ", preprocess_batch_items=" + py::repr(generate_usermeta_iterator<BatchMetadata, NVDS_PREPROCESS_BATCH_META>(self)).cast<std::string>() + ")";
        });
    py::class_<FrameMetadata>(m, "FrameMetadata", pydeepstreamdoc::nvmeta::FrameMetaDoc::descr)
        .def_property_readonly("frame_number", [](const FrameMetadata& self) {
            return self.frameNum();
        })
        .def_property_readonly("pad_index", [](const FrameMetadata& self) {
            return self.padIndex();
        })
        .def_property_readonly("batch_id", [](const FrameMetadata& self) {
            return self.batchId();
        })
        .def_property_readonly("source_id", [](const FrameMetadata& self) {
            return self.sourceId();
        })
        .def_property_readonly("source_width", [](const FrameMetadata& self) {
            return self.sourceWidth();
        })
        .def_property_readonly("source_height", [](const FrameMetadata& self) {
            return self.sourceHeight();
        })
        .def_property_readonly("pipeline_width", [](const FrameMetadata& self) {
            return self.pipelineWidth();
        })
        .def_property_readonly("pipeline_height", [](const FrameMetadata& self) {
            return self.pipelineHeight();
        })
        .def_property_readonly("buffer_pts", [](const FrameMetadata& self) {
            return self.bufferPTS();
        })
        .def_property_readonly("ntp_timestamp", [](const FrameMetadata& self) {
            return self.ntpTimestamp();
        })
        .def_property_readonly("object_items", [](FrameMetadata& self) {
            return generate_iterator<ObjectMetadata, FrameMetadata, ObjectMetadata::Iterator>(self);
        })
        .def_property_readonly("tensor_items", [](FrameMetadata& self) {
            return generate_usermeta_iterator<FrameMetadata, NVDSINFER_TENSOR_OUTPUT_META>(self);
        })
        .def_property_readonly("segmentation_items", [](FrameMetadata& self) {
            return generate_usermeta_iterator<FrameMetadata, NVDSINFER_SEGMENTATION_META>(self);
        })
        .def_property_readonly("nvdsanalytics_frame_items", [](FrameMetadata& self) {
            return generate_usermeta_iterator<FrameMetadata, NVDS_USER_FRAME_META_NVDSANALYTICS>(self);
        })
        .def("retrieve_structure_usermeta", [](FrameMetadata& self, const std::string& type_name){
            return retrieve_structure_usermeta<FrameMetadata>(self, type_name);
        })
        .def("user_meta_items", [](FrameMetadata& self, int meta_type) {
            IteratorWrapper<UserMetadata, UserMetadata::Iterator> begin;
            IteratorWrapper<UserMetadata, UserMetadata::Iterator> end;
            self.initiateIterator(begin.get(), meta_type);
            self.initiateIterator(end.get(), meta_type);
            while (!end.get()->done()) ++end;
            return py::make_iterator(std::move(begin), std::move(end));
        }, pydeepstreamdoc::nvmeta::FrameMetaDoc::user_meta_items,
           py::arg("meta_type"), py::keep_alive<0, 1>())
        .def("append", [](FrameMetadata& self, py::object object) {
            if (py::isinstance<DisplayMetadata>(object)){
                self.append(object.cast<DisplayMetadata>());
            } else if (py::isinstance<EventMessageUserMetadata>(object)) {
                self.append(object.cast<EventMessageUserMetadata>());
            } else if (py::isinstance<ObjectMetadata>(object)) {
                self.append(object.cast<ObjectMetadata>());
            } else {
                throw py::type_error("Invalid type for FrameMetadata.append");
            }
        }, pydeepstreamdoc::nvmeta::FrameMetaDoc::append)
        .def("__repr__", [](FrameMetadata& self) {
            return "FrameMetadata(frame_number=" + std::to_string(self.frameNum()) +
                ", pad_index=" + std::to_string(self.padIndex()) +
                ", batch_id=" + std::to_string(self.batchId()) +
                ", source_id=" + std::to_string(self.sourceId()) +
                ", source_width=" + std::to_string(self.sourceWidth()) +
                ", source_height=" + std::to_string(self.sourceHeight()) +
                ", pipeline_width=" + std::to_string(self.pipelineWidth()) +
                ", pipeline_height=" + std::to_string(self.pipelineHeight()) +
                ", buffer_pts=" + std::to_string(self.bufferPTS()) +
                ", ntp_timestamp=" + std::to_string(self.ntpTimestamp()) +
                ", object_items=" + py::repr(generate_iterator<ObjectMetadata, FrameMetadata, ObjectMetadata::Iterator>(self)).cast<std::string>() +
                ", tensor_items=" + py::repr(generate_usermeta_iterator<FrameMetadata, NVDSINFER_TENSOR_OUTPUT_META>(self)).cast<std::string>() +
                ", segmentation_items=" + py::repr(generate_usermeta_iterator<FrameMetadata, NVDSINFER_SEGMENTATION_META>(self)).cast<std::string>() +
                ", nvdsanalytics_frame_items=" + py::repr(generate_usermeta_iterator<FrameMetadata, NVDS_USER_FRAME_META_NVDSANALYTICS>(self)).cast<std::string>() + ")";
        });
    py::class_<AudioFrameMetadata>(m, "AudioFrameMetadata", pydeepstreamdoc::nvmeta::AudioFrameMetaDoc::descr)
        .def_property_readonly("frame_number", [](const AudioFrameMetadata& self) {
            return self.frameNum();
        })
        .def_property_readonly("pad_index", [](const AudioFrameMetadata& self) {
            return self.padIndex();
        })
        .def_property_readonly("batch_id", [](const AudioFrameMetadata& self) {
            return self.batchId();
        })
        .def_property_readonly("source_id", [](const AudioFrameMetadata& self) {
            return self.sourceId();
        })
        .def_property_readonly("sample_rate", [](const AudioFrameMetadata& self) {
            return self.sampleRate();
        })
        .def_property_readonly("num_channels", [](const AudioFrameMetadata& self) {
            return self.numChannels();
        })
        .def_property_readonly("num_samples_per_frame", [](const AudioFrameMetadata& self) {
            return self.numSamplesPerFrame();
        })
        .def_property_readonly("format", [](const AudioFrameMetadata& self) {
            return self.format();
        })
        .def_property_readonly("layout", [](const AudioFrameMetadata& self) {
            return self.layout();
        })
        .def_property_readonly("buffer_pts", [](const AudioFrameMetadata& self) {
            return self.bufferPTS();
        })
        .def_property_readonly("ntp_timestamp", [](const AudioFrameMetadata& self) {
            return self.ntpTimestamp();
        })
        .def_property_readonly("infer_done", [](const AudioFrameMetadata& self) {
            return self.inferDone();
        })
        .def_property_readonly("class_id", [](const AudioFrameMetadata& self) {
            return self.classId();
        })
        .def_property_readonly("confidence", [](const AudioFrameMetadata& self) {
            return self.confidence();
        })
        .def_property_readonly("class_label", [](const AudioFrameMetadata& self) {
            return self.classLabel();
        })
        .def_property_readonly("classifier_items", [](AudioFrameMetadata& self) {
            return generate_iterator<ClassifierMetadata, AudioFrameMetadata, ClassifierMetadata::Iterator>(self);
        })
        .def_property_readonly("tensor_items", [](AudioFrameMetadata& self) {
            return generate_usermeta_iterator<AudioFrameMetadata, NVDSINFER_TENSOR_OUTPUT_META>(self);
        })
        .def("user_meta_items", [](AudioFrameMetadata& self, int meta_type) {
            IteratorWrapper<UserMetadata, UserMetadata::Iterator> begin;
            IteratorWrapper<UserMetadata, UserMetadata::Iterator> end;
            self.initiateIterator(begin.get(), meta_type);
            self.initiateIterator(end.get(), meta_type);
            while (!end.get()->done()) ++end;
            return py::make_iterator(std::move(begin), std::move(end));
        }, pydeepstreamdoc::nvmeta::AudioFrameMetaDoc::user_meta_items,
           py::arg("meta_type"), py::keep_alive<0, 1>())
        .def("append", [](AudioFrameMetadata& self, py::object object) {
            if (py::isinstance<EventMessageUserMetadata>(object)) {
                self.append(object.cast<EventMessageUserMetadata>());
            } else if (py::isinstance<UserMetadata>(object)) {
                self.append(object.cast<UserMetadata>());
            } else {
                throw py::type_error("Invalid type for AudioFrameMetadata.append");
            }
        }, pydeepstreamdoc::nvmeta::AudioFrameMetaDoc::append)
        .def("__repr__", [](AudioFrameMetadata& self) {
            return "AudioFrameMetadata(frame_number=" + std::to_string(self.frameNum()) +
                ", pad_index=" + std::to_string(self.padIndex()) +
                ", batch_id=" + std::to_string(self.batchId()) +
                ", source_id=" + std::to_string(self.sourceId()) +
                ", sample_rate=" + std::to_string(self.sampleRate()) +
                ", num_channels=" + std::to_string(self.numChannels()) +
                ", num_samples_per_frame=" + std::to_string(self.numSamplesPerFrame()) +
                ", class_id=" + std::to_string(self.classId()) +
                ", class_label=" + (self.classLabel().empty() ? "''" : self.classLabel()) +
                ", confidence=" + std::to_string(self.confidence()) +
                ", classifier_items=" + py::repr(generate_iterator<ClassifierMetadata, AudioFrameMetadata, ClassifierMetadata::Iterator>(self)).cast<std::string>() + ")";
        });
    py::class_<DisplayMetadata>(m, "DisplayMetadata", pydeepstreamdoc::nvmeta::DisplayMetaDoc::descr)
        .def_property_readonly("n_rects", [](const DisplayMetadata& self) {
            return self.nRects();
        })
        .def_property_readonly("n_labels", [](const DisplayMetadata& self) {
            return self.nLabels();
        })
        .def_property_readonly("n_lines", [](const DisplayMetadata& self) {
            return self.nLines();
        })
        .def_property_readonly("n_arrows", [](const DisplayMetadata& self) {
            return self.nArrows();
        })
        .def_property_readonly("n_circles", [](const DisplayMetadata& self) {
            return self.nCircles();
        })
        .def("add_text", [](DisplayMetadata& self, Text& item) {
            self.add(item);
        }, pydeepstreamdoc::nvmeta::DisplayMetaDoc::add_text)
        .def("add_rect", [](DisplayMetadata& self, NvOSD_RectParams& item) {
            self.add(item);
        }, pydeepstreamdoc::nvmeta::DisplayMetaDoc::add_rect)
        .def("add_line", [](DisplayMetadata& self, NvOSD_LineParams& item) {
            self.add(item);
        }, pydeepstreamdoc::nvmeta::DisplayMetaDoc::add_line)
        .def("add_arrow", [](DisplayMetadata& self, NvOSD_ArrowParams& item) {
            self.add(item);
        }, pydeepstreamdoc::nvmeta::DisplayMetaDoc::add_arrow)
        .def("add_circle", [](DisplayMetadata& self, NvOSD_CircleParams& item) {
            self.add(item);
        }, pydeepstreamdoc::nvmeta::DisplayMetaDoc::add_circle)
        .def("__repr__", [](DisplayMetadata& self) {
            return "DisplayMetadata(n_rects=" + std::to_string(self.nRects()) +
                ", n_labels=" + std::to_string(self.nLabels()) +
                ", n_lines=" + std::to_string(self.nLines()) +
                ", n_arrows=" + std::to_string(self.nArrows()) +
                ", n_circles=" + std::to_string(self.nCircles()) + ")";
        });
    py::class_<LabelInfo>(m, "LabelInfo", pydeepstreamdoc::nvmeta::LabelInfoDoc::descr)
        .def_property_readonly("label", &LabelInfo::label)
        .def_property_readonly("class_id", &LabelInfo::classId)
        .def_property_readonly("prob", &LabelInfo::prob)
        .def_property_readonly("label_id", &LabelInfo::labelId)
        .def_property_readonly("num_classes", &LabelInfo::numClasses)
        .def("__repr__", [](LabelInfo& self) {
            return "LabelInfo(label=" + self.label() +
                ", class_id=" + std::to_string(self.classId()) +
                ", prob=" + std::to_string(self.prob()) +
                ", label_id=" + std::to_string(self.labelId()) +
                ", num_classes=" + std::to_string(self.numClasses()) + ")";
        });
    py::class_<ClassifierMetadata>(m, "ClassifierMetadata", pydeepstreamdoc::nvmeta::ClassifierMetaDoc::descr)
        .def_property_readonly("n_labels", &ClassifierMetadata::nLabels)
        .def_property_readonly("unique_component_id", &ClassifierMetadata::uniqueComponentId)
        .def_property_readonly("classifier_type", &ClassifierMetadata::classifierType, pydeepstreamdoc::nvmeta::ClassifierMetaDoc::classifier_type)
        .def("get_label_info", &ClassifierMetadata::getLabelInfo, pydeepstreamdoc::nvmeta::ClassifierMetaDoc::get_label_info)
        .def("get_n_label", &ClassifierMetadata::getLabel, pydeepstreamdoc::nvmeta::ClassifierMetaDoc::get_label)
        .def("__repr__", [](ClassifierMetadata& self) {
            return "ClassifierMetadata(n_labels=" + std::to_string(self.nLabels()) +
                ", unique_component_id=" + std::to_string(self.uniqueComponentId()) +
                ", classifier_type=" + self.classifierType() + ")";
        });
    py::class_<ObjectMetadata>(m, "ObjectMetadata", pydeepstreamdoc::nvmeta::ObjectMetaDoc::descr)
        .def_property("class_id", &ObjectMetadata::classId, &ObjectMetadata::setClassId)
        .def_property("object_id", &ObjectMetadata::objectId, &ObjectMetadata::setObjectId)
        .def_property("unique_component_id", &ObjectMetadata::uniqueComponentId, &ObjectMetadata::setUniqueComponentId)
        .def_property("confidence", &ObjectMetadata::confidence, &ObjectMetadata::setConfidence)
        .def_property("tracker_confidence", &ObjectMetadata::trackerConfidence, &ObjectMetadata::setTrackerConfidence)
        .def_property("rect_params", &ObjectMetadata::rectParams, &ObjectMetadata::setRectParams, py::return_value_policy::reference_internal)
        .def_property("mask_params", &ObjectMetadata::maskParams, &ObjectMetadata::setMaskParams, py::return_value_policy::reference_internal)
        .def_property("text_params", &ObjectMetadata::textParams, &ObjectMetadata::setTextParams, py::return_value_policy::reference_internal)
        .def_property("label", &ObjectMetadata::label, &ObjectMetadata::setLabel)
        .def_property_readonly("classifier_items", [](ObjectMetadata& self) {
            return generate_iterator<ClassifierMetadata, ObjectMetadata, ClassifierMetadata::Iterator>(self);
        })
        .def_property_readonly("tensor_items", [](ObjectMetadata& self) {
            return generate_usermeta_iterator<ObjectMetadata, NVDSINFER_TENSOR_OUTPUT_META>(self);
        })
        .def_property_readonly("nvdsanalytics_obj_items", [](ObjectMetadata& self) {
            return generate_usermeta_iterator<ObjectMetadata, NVDS_USER_OBJ_META_NVDSANALYTICS>(self);
        })
        .def_property_readonly("obj_reid_items", [](ObjectMetadata& self) {
            return generate_usermeta_iterator<ObjectMetadata, NVDS_TRACKER_OBJ_REID_META>(self);
        })
        .def("user_meta_items", [](ObjectMetadata& self, int meta_type) {
            IteratorWrapper<UserMetadata, UserMetadata::Iterator> begin;
            IteratorWrapper<UserMetadata, UserMetadata::Iterator> end;
            self.initiateIterator(begin.get(), meta_type);
            self.initiateIterator(end.get(), meta_type);
            while (!end.get()->done()) ++end;
            return py::make_iterator(std::move(begin), std::move(end));
        }, pydeepstreamdoc::nvmeta::ObjectMetaDoc::user_meta_items,
           py::arg("meta_type"), py::keep_alive<0, 1>())
        .def("append", [](ObjectMetadata& self, UserMetadata& user_meta) {
            self.append(user_meta);
        }, pydeepstreamdoc::nvmeta::ObjectMetaDoc::append)
        .def_property("nv_bbox_info", [](const ObjectMetadata& self) {
            const NvBbox_Coords& b = self.nvBboxInfo();
            py::dict d;
            d["left"]   = b.left;
            d["top"]    = b.top;
            d["width"]  = b.width;
            d["height"] = b.height;
            return d;
        }, [](ObjectMetadata& self, py::dict d) {
            NvBbox_Coords b;
            b.left   = d["left"].cast<float>();
            b.top    = d["top"].cast<float>();
            b.width  = d["width"].cast<float>();
            b.height = d["height"].cast<float>();
            self.setNvBboxInfo(b);
        })
        .def("__repr__", [](ObjectMetadata& self) {
            return "ObjectMetadata(class_id=" + std::to_string(self.classId()) +
                ", object_id=" + std::to_string(self.objectId()) +
                ", unique_component_id=" + std::to_string(self.uniqueComponentId()) +
                ", confidence=" + std::to_string(self.confidence()) +
                ", rect_params=" + py::repr(py::cast(self.rectParams())).cast<std::string>() +
                ", mask_params=" + py::repr(py::cast(self.maskParams())).cast<std::string>() +
                ", text_params=" + [&]() {
                    const NvOSD_TextParams* text_ptr = &self.textParams();
                    if (text_ptr == nullptr) {
                        return std::string("Text(null)");
                    }
                    const NvOSD_TextParams& text = *text_ptr;
                    std::string display_text_str = text.display_text ? std::string(text.display_text) : "null";
                    std::string font_name_str = text.font_params.font_name ? std::string(text.font_params.font_name) : "null";

                    return "Text(display_text='" + display_text_str +
                           "', x_offset=" + std::to_string(text.x_offset) +
                           ", y_offset=" + std::to_string(text.y_offset) +
                           ", font_name='" + font_name_str +
                           "', font_size=" + std::to_string(text.font_params.font_size) +
                           ", bg_color=" + py::repr(py::cast(text.text_bg_clr)).cast<std::string>() + ")";
                }() +
                ", label=" + (self.label().empty() ? "''" : self.label()) +
                ", classifier_items=" + py::repr(generate_iterator<ClassifierMetadata, ObjectMetadata, ClassifierMetadata::Iterator>(self)).cast<std::string>() +
                ", tensor_items=" + py::repr(generate_usermeta_iterator<ObjectMetadata, NVDSINFER_TENSOR_OUTPUT_META>(self)).cast<std::string>() +
                ", nvdsanalytics_obj_items=" + py::repr(generate_usermeta_iterator<ObjectMetadata, NVDS_USER_OBJ_META_NVDSANALYTICS>(self)).cast<std::string>() + ")";
        });
    py::class_<UserMetadata>(m, "UserMetadata", pydeepstreamdoc::nvmeta::UserMetaDoc::descr)
        .def_property("meta_type", &UserMetadata::metaType, &UserMetadata::setMetaType)
        .def("set_user_data_json", [](UserMetadata& self, py::object obj, int meta_type) {
            // Import json module and serialize object
            py::module_ json = py::module_::import("json");
            py::str json_str = json.attr("dumps")(obj);
            std::string str = json_str.cast<std::string>();
            // Allocate C string copy
            char* c_str = strdup(str.c_str());
            self.setUserData(c_str, copy_cstring_user_data, release_cstring_user_data);
            self.setMetaType(meta_type);
        }, pydeepstreamdoc::nvmeta::UserMetaDoc::set_user_data_json,
           py::arg("data"), py::arg("meta_type"))
        .def("get_user_data_json", [](UserMetadata& self) -> py::object {
            void* data = self.userData();
            if (data) {
                const char* json_str = static_cast<const char*>(data);
                // Import json module and deserialize string
                py::module_ json = py::module_::import("json");
                return json.attr("loads")(py::str(json_str));
            }
            return py::none();
        }, pydeepstreamdoc::nvmeta::UserMetaDoc::get_user_data_json)
        .def("as_tensor_output", [](UserMetadata& self) {
            return TensorOutputUserMetadata(self);
        }, pydeepstreamdoc::nvmeta::UserMetaDoc::as_tensor_output)
        .def("as_segmentation", [](UserMetadata& self) {
            return SegmentationUserMetadata(self);
        }, pydeepstreamdoc::nvmeta::UserMetaDoc::as_segmentation)
        .def("as_preprocess_batch", [](UserMetadata& self) {
            return PreprocessBatchUserMetadata(self);
        }, pydeepstreamdoc::nvmeta::UserMetaDoc::as_preprocess_batch)
        .def("as_nvdsanalytics_obj", [](UserMetadata& self) {
            return AnalyticsObjInfo(self);
        }, pydeepstreamdoc::nvmeta::UserMetaDoc::as_nvdsanalytics_obj)
        .def("as_nvdsanalytics_frame", [](UserMetadata& self) {
            return AnalyticsFrameMeta(self);
        }, pydeepstreamdoc::nvmeta::UserMetaDoc::as_nvdsanalytics_frame)
        .def("as_structure_str", [](UserMetadata& self) -> std::string {
           StructureUserMetadata meta(self);
           const auto *structure = (const GstStructure*)meta.getStructureUserMetadata();
           if (!structure) return "";

           // If Structure no name, return empty string
           auto name = gst_structure_get_name(structure);
           if (name == nullptr) return "";

           // Create a copy and convert gpointer fields to guint64
           GstStructure *converted = gst_structure_new_empty(name);
           gint n_fields = gst_structure_n_fields(structure);
           for (gint i = 0; i < n_fields; i++) {
               const gchar *field_name = gst_structure_nth_field_name(structure, i);
               const GValue *value = gst_structure_get_value(structure, field_name);
               if (value && G_VALUE_TYPE(value) == G_TYPE_POINTER) {
                   // Convert pointer to uint64
                   gpointer ptr = g_value_get_pointer(value);
                   guint64 ptr_as_uint64 = (guint64)(uintptr_t)ptr;
                   gst_structure_set(converted, field_name, G_TYPE_UINT64, ptr_as_uint64, NULL);
               } else if (value) {
                   // Copy other fields as-is
                   gst_structure_set_value(converted, field_name, value);
               }
           }

           gchar* str = gst_structure_to_string(converted);
           gst_structure_free(converted);

           if (!str) return "";
           std::string result(str);
           g_free(str);
           return result;
        })
        .def("as_obj_reid", [](UserMetadata& self) {
            return ObjectReidUserMetadata(self);
        }, pydeepstreamdoc::nvmeta::UserMetaDoc::as_obj_reid)
        .def("__bool__", [](UserMetadata& self) { return (bool)self; });
    py::class_<ObjectReidUserMetadata, UserMetadata>(m, "ObjectReidUserMetadata", pydeepstreamdoc::nvmeta::ObjectReidUserMetaDoc::descr)
        .def_property_readonly("feature_size", &ObjectReidUserMetadata::featureSize)
        .def_property_readonly("feature_vector", [](ObjectReidUserMetadata& self) {
            const float* data = self.featureVector();
            unsigned int size = self.featureSize();
            if (!data || size == 0) {
                return py::array_t<float>();
            }
            return py::array_t<float>(
                {size},
                {sizeof(float)},
                data
            );
        })
        .def("set_feature_vector", [](ObjectReidUserMetadata& self, py::array_t<float> features) {
            auto buf = features.request();
            if (buf.ndim != 1) {
                throw std::runtime_error("Feature vector must be 1-dimensional");
            }
            self.setFeatureVector(static_cast<float*>(buf.ptr), buf.shape[0]);
        }, pydeepstreamdoc::nvmeta::ObjectReidUserMetaDoc::set_feature_vector,
           py::arg("features"))
        .def("__repr__", [](ObjectReidUserMetadata& self) {
            return "ObjectReidUserMetadata(feature_size=" + std::to_string(self.featureSize()) + ")";
        });
    py::class_<EventMessageUserMetadata, UserMetadata>(m, "EventMessageUserMetadata", pydeepstreamdoc::nvmeta::EventMessageUserMetaDoc::descr)
        .def("generate", [](EventMessageUserMetadata& self,
                            const ObjectMetadata& object_meta,
                            const FrameMetadata& frame_meta,
                            const std::string& sensor,
                            const std::string& uri,
                            const std::vector<std::string>& labels) {
            self.generate(object_meta, frame_meta, sensor, uri, labels);
        }, pydeepstreamdoc::nvmeta::EventMessageUserMetaDoc::generate,
           py::arg("object_meta"), py::arg("frame_meta"), py::arg("sensor")="N/A", py::arg("uri")="N/A", py::arg("labels")=std::vector<std::string>())
        .def("generate", [](EventMessageUserMetadata& self,
                            const AudioFrameMetadata& audio_frame_meta,
                            const std::string& sensor,
                            const std::string& uri) {
            self.generate(audio_frame_meta, sensor, uri);
        }, pydeepstreamdoc::nvmeta::EventMessageUserMetaDoc::generate_audio,
           py::arg("audio_frame_meta"), py::arg("sensor")="N/A", py::arg("uri")="N/A");
    py::class_<TensorOutputUserMetadata, UserMetadata>(m, "TensorOutputUserMetadata", pydeepstreamdoc::nvmeta::TensorOutputUserMetaDoc::descr)
        .def_property_readonly("unique_id", [](const TensorOutputUserMetadata& self) {
            return self.uniqueId();
        })
        .def("get_layers", [](TensorOutputUserMetadata& self) {
            std::unordered_map<std::string, TensorWrapper> outputs;
            auto layers = self.getLayers();
            for (auto it = layers.begin(); it != layers.end(); it++) {
                outputs[it->first] = TensorWrapper(it->second);
            }
            return outputs;
        }, pydeepstreamdoc::nvmeta::TensorOutputUserMetaDoc::get_layers)
        .def("__repr__", [](TensorOutputUserMetadata& self) {
            return "TensorOutputUserMetadata(unique_id=" + std::to_string(self.uniqueId()) + ")";
        });
    py::class_<SegmentationUserMetadata, UserMetadata>(m, "SegmentationUserMetadata", pydeepstreamdoc::nvmeta::SegmentationUserMetaDoc::descr)
        .def_property_readonly("unique_id", [](const SegmentationUserMetadata& self) {
            return self.uniqueId();
        })
        .def_property_readonly("classes", [](SegmentationUserMetadata& self) {
            return self.getClasses();
        })
        .def_property_readonly("width", [](SegmentationUserMetadata& self) {
            return self.getWidth();
        })
        .def_property_readonly("height", [](SegmentationUserMetadata& self) {
            return self.getHeight();
        })
        .def_property_readonly("class_map", [](SegmentationUserMetadata& self) {
            const int* data = self.getClassMap();
            if (!data) {
                return py::array_t<int>();
            }
            return py::array_t<int>(
                {self.getHeight(), self.getWidth()},
                {self.getWidth() * sizeof(int), sizeof(int)},
                data
            );
        })
        .def_property_readonly("class_probabilities_map", [](SegmentationUserMetadata& self) {
            const float* data = self.getClassProbabilitiesMap();
            if (!data) {
                return py::array_t<float>();
            }
            return py::array_t<float>(
                {self.getHeight(), self.getWidth(), self.getClasses()},
                {self.getWidth() * self.getClasses() * sizeof(float), self.getClasses() * sizeof(float), sizeof(float)},
                data
            );
        })
        .def("__repr__", [](SegmentationUserMetadata& self) {
            const int* class_map_data = self.getClassMap();
            py::array_t<int> class_map;
            if (!class_map_data) {
                class_map = py::array_t<int>();
            } else {
                class_map = py::array_t<int>(
                    {self.getHeight(), self.getWidth()},
                    {self.getWidth() * sizeof(int), sizeof(int)},
                    class_map_data
                );
            }
            const float* class_probabilities_map_data = self.getClassProbabilitiesMap();
            py::array_t<float> class_probabilities_map;
            if (!class_probabilities_map_data) {
                class_probabilities_map = py::array_t<float>();
            } else {
                class_probabilities_map = py::array_t<float>(
                    {self.getHeight(), self.getWidth(), self.getClasses()},
                    {self.getWidth() * self.getClasses() * sizeof(float), self.getClasses() * sizeof(float), sizeof(float)},
                    class_probabilities_map_data
                );
            }

            return "SegmentationUserMetadata(unique_id=" + std::to_string(self.uniqueId()) +
                ", classes=" + std::to_string(self.getClasses()) +
                ", width=" + std::to_string(self.getWidth()) +
                ", height=" + std::to_string(self.getHeight()) +
                ", class_map=" + py::repr(class_map).cast<std::string>() +
                ", class_probabilities_map=" + py::repr(class_probabilities_map).cast<std::string>() + ")";
        });
    py::class_<RoiMetadata>(m, "RoiMetadata", pydeepstreamdoc::nvmeta::RoiMetaDoc::descr)
        .def_property_readonly("roi", [](const RoiMetadata& self) {
            return self.rectParams();
        })
        .def_property_readonly("frame_meta", [](const RoiMetadata& self) {
            return self.frameMetadata();
        })
        .def_property_readonly("tensor_items", [](RoiMetadata& self) {
            return generate_usermeta_iterator<RoiMetadata, NVDSINFER_TENSOR_OUTPUT_META>(self);
        })
        .def_property_readonly("classifier_items", [](RoiMetadata& self) {
            return generate_iterator<ClassifierMetadata, RoiMetadata, ClassifierMetadata::Iterator>(self);
        })
        .def_property_readonly("segmentation_items", [](RoiMetadata& self) {
            return generate_usermeta_iterator<RoiMetadata, NVDSINFER_SEGMENTATION_META>(self);
        })
        .def_property_readonly("object_meta", [](const RoiMetadata& self) {
            return self.objectMetadata();
        })
        .def("__repr__", [](RoiMetadata& self) {
            std::string object_meta_str = "null";
            auto object_meta = self.objectMetadata();
            if (object_meta) {
                object_meta_str = py::repr(py::cast(object_meta)).cast<std::string>();
            }
            return "RoiMetadata(roi=" + py::repr(py::cast(self.rectParams())).cast<std::string>() +
                ", frame_meta=" + py::repr(py::cast(self.frameMetadata())).cast<std::string>() +
                ", classifier_items=" + py::repr(generate_iterator<ClassifierMetadata, RoiMetadata, ClassifierMetadata::Iterator>(self)).cast<std::string>() +
                ", segmentation_items=" + py::repr(generate_usermeta_iterator<RoiMetadata, NVDSINFER_SEGMENTATION_META>(self)).cast<std::string>() +
                ", object_meta=" + object_meta_str + ")";
        });
    py::class_<PreprocessTensorMetadata>(m, "PreprocessTensorMetadata", pydeepstreamdoc::nvmeta::PreprocessTensorMetaDoc::descr)
        .def_property_readonly("name", [](PreprocessTensorMetadata& self) {
            return self.getName();
        })
        .def_property_readonly("meta_id", [](PreprocessTensorMetadata& self) {
            return self.getMetaId();
        })
        .def_property_readonly("tensor", [](PreprocessTensorMetadata& self) {
            return TensorWrapper(self.getTensor());
        })
        .def("__repr__", [](PreprocessTensorMetadata& self) {
            std::string tensor_str = "null";
            std::unique_ptr<Tensor> tensor_ptr(self.getTensor());
            if (tensor_ptr != nullptr) {
                TensorWrapper temp_wrapper(tensor_ptr.get());
                tensor_str = py::repr(py::cast(temp_wrapper, py::return_value_policy::reference)).cast<std::string>();
                temp_wrapper.release();
            }
            return "PreprocessTensorMetadata(name=" + self.getName() +
                ", meta_id=" + std::to_string(self.getMetaId()) +
                ", tensor=" + tensor_str + ")";
        });
    py::class_<PreprocessBatchUserMetadata, UserMetadata>(m, "PreprocessBatchUserMetadata", pydeepstreamdoc::nvmeta::PreprocessBatchUserMetaDoc::descr)
        .def_property_readonly("rois", [](PreprocessBatchUserMetadata& self) {
            return self.getRois();
        })
        .def_property_readonly("preprocess_tensor_meta", [](PreprocessBatchUserMetadata& self) {
            return self.getPreprocessTensorMetadata();
        })
        .def("set_preprocessed_tensor", [](PreprocessBatchUserMetadata& self, std::string name, unsigned int meta_id, unsigned int unique_id, TensorWrapper& tensor) {
            self.initPreprocessTensorMetadata(name, meta_id, unique_id, tensor.release());
        }, pydeepstreamdoc::nvmeta::PreprocessBatchUserMetaDoc::set_preprocessed_tensor)
        .def("extract", &preprocess_batch_metadata_extract, pydeepstreamdoc::nvmeta::PreprocessBatchUserMetaDoc::extract,
             py::arg("max_items"))
        .def("__repr__", [](PreprocessBatchUserMetadata& self) {
            std::string preprocess_tensor_meta_str = "null";
            auto preprocess_tensor_meta = self.getPreprocessTensorMetadata();
            if (preprocess_tensor_meta) {
                preprocess_tensor_meta_str = py::repr(py::cast(preprocess_tensor_meta)).cast<std::string>();
            }
            return "PreprocessBatchUserMetadata(rois=" + py::repr(py::cast(self.getRois())).cast<std::string>() +
                ", preprocess_tensor_meta=" + preprocess_tensor_meta_str + ")";
        });
    py::class_<AnalyticsObjInfo, UserMetadata>(m, "AnalyticsObjInfo", pydeepstreamdoc::nvmeta::AnalyticsObjInfoDoc::descr)
        .def_property_readonly("roi_status", [](AnalyticsObjInfo& self) {
            return self.getRoiStatus();
        })
        .def_property_readonly("oc_status", [](AnalyticsObjInfo& self) {
            return self.getOcStatus();
        })
        .def_property_readonly("lc_status", [](AnalyticsObjInfo& self) {
            return self.getLcStatus();
        })
        .def_property_readonly("dir_status", [](AnalyticsObjInfo& self) {
            return self.getDirStatus();
        })
        .def_property_readonly("unique_id", [](AnalyticsObjInfo& self) {
            return self.getUniqueId();
        })
        .def_property_readonly("obj_status", [](AnalyticsObjInfo& self) {
            return self.getObjStatus();
        })
        .def("__repr__", [](AnalyticsObjInfo& self) {
            return "AnalyticsObjInfo(roi_status=" + py::repr(py::cast(self.getRoiStatus())).cast<std::string>() +
                ", oc_status=" + py::repr(py::cast(self.getOcStatus())).cast<std::string>() +
                ", lc_status=" + py::repr(py::cast(self.getLcStatus())).cast<std::string>() +
                ", dir_status=" + self.getDirStatus() +
                ", unique_id=" + std::to_string(self.getUniqueId()) +
                ", obj_status=" + self.getObjStatus() + ")";
        });
    py::class_<AnalyticsFrameMeta, UserMetadata>(m, "AnalyticsFrameMeta", pydeepstreamdoc::nvmeta::AnalyticsFrameMetaDoc::descr)
        .def_property_readonly("oc_status", [](AnalyticsFrameMeta& self) {
            return self.getOcStatus();
        })
        .def_property_readonly("obj_in_roi_cnt", [](AnalyticsFrameMeta& self) {
            return self.getObjInROIcnt();
        })
        .def_property_readonly("obj_lc_curr_cnt", [](AnalyticsFrameMeta& self) {
            return self.getObjLCCurrCnt();
        })
        .def_property_readonly("obj_lc_cum_cnt", [](AnalyticsFrameMeta& self) {
            return self.getObjLCCumCnt();
        })
        .def_property_readonly("unique_id", [](AnalyticsFrameMeta& self) {
            return self.getUniqueId();
        })
        .def_property_readonly("obj_cnt", [](AnalyticsFrameMeta& self) {
            return self.getObjCnt();
        })
        .def("__repr__", [](AnalyticsFrameMeta& self) {
            return "AnalyticsFrameMeta(oc_status=" + py::repr(py::cast(self.getOcStatus())).cast<std::string>() +
                ", obj_in_roi_cnt=" + py::repr(py::cast(self.getObjInROIcnt())).cast<std::string>() +
                ", obj_lc_curr_cnt=" + py::repr(py::cast(self.getObjLCCurrCnt())).cast<std::string>() +
                ", obj_lc_cum_cnt=" + py::repr(py::cast(self.getObjLCCumCnt())).cast<std::string>() +
                ", unique_id=" + std::to_string(self.getUniqueId()) +
                ", obj_cnt=" + py::repr(py::cast(self.getObjCnt())).cast<std::string>() + ")";
        });
    py::class_<ObjectVisibilityUserMetadata, UserMetadata>(m, "ObjectVisibilityUserMetadata")
        .def(py::init<const UserMetadata&>())
        .def_property_readonly("visibility", &ObjectVisibilityUserMetadata::getVisibility);
    py::class_<ObjectImageFootLocationUserMetadata, UserMetadata>(m, "ObjectImageFootLocationUserMetadata")
        .def(py::init<const UserMetadata&>())
        .def_property_readonly("image_foot_location", [](ObjectImageFootLocationUserMetadata& self) {
            auto loc = self.getImageFootLocation();
            py::dict d;
            d["x"] = loc.first;
            d["y"] = loc.second;
            return d;
        });
    py::class_<ObjectWorldFootLocationUserMetadata, UserMetadata>(m, "ObjectWorldFootLocationUserMetadata")
        .def(py::init<const UserMetadata&>())
        .def_property_readonly("world_foot_location", [](ObjectWorldFootLocationUserMetadata& self) {
            auto loc = self.getWorldFootLocation();
            py::dict d;
            d["x"] = loc.first;
            d["y"] = loc.second;
            return d;
        });
    py::class_<ObjectConvexHullUserMetadata, UserMetadata>(m, "ObjectConvexHullUserMetadata")
        .def(py::init<const UserMetadata&>())
        .def_property_readonly("convex_hull", [](ObjectConvexHullUserMetadata& self) {
            py::list points;
            for (auto& pt : self.getConvexHull()) {
                points.append(py::make_tuple(pt.first, pt.second));
            }
            return points;
        });
    py::class_<Object3DBBoxUserMetadata, UserMetadata>(m, "Object3DBBoxUserMetadata")
        .def(py::init<const UserMetadata&>())
        .def_property_readonly("bbox_3d", [](Object3DBBoxUserMetadata& self) {
            py::dict d;
            NvDsObj3DBbox* bbox = self.get3DBbox();
            if (!bbox) return d;
            d["x_centre"] = bbox->xCentre;
            d["y_centre"] = bbox->yCentre;
            d["z_centre"] = bbox->zCentre;
            d["x_len"] = bbox->xLen;
            d["y_len"] = bbox->yLen;
            d["z_len"] = bbox->zLen;
            d["x_rot"] = bbox->xRot;
            d["y_rot"] = bbox->yRot;
            d["z_rot"] = bbox->zRot;
            d["x_vel"] = bbox->xVel;
            d["y_vel"] = bbox->yVel;
            d["z_vel"] = bbox->zVel;
            return d;
        });
    py::class_<TrackerPastFrameUserMetadata, UserMetadata>(m, "TrackerPastFrameUserMetadata")
        .def(py::init<const UserMetadata&>())
        .def_property_readonly("past_frame_data", [](TrackerPastFrameUserMetadata& self) {
            py::list streams;
            NvDsTargetMiscDataBatch* batch = self.getTrackerMiscDataBatch();
            if (!batch) return streams;
            for (uint32_t si = 0; si < batch->numFilled; si++) {
                NvDsTargetMiscDataStream* s = batch->list + si;
                py::list objects;
                for (uint32_t li = 0; li < s->numFilled; li++) {
                    NvDsTargetMiscDataObject* obj = s->list + li;
                    py::list frames;
                    for (uint32_t oi = 0; oi < obj->numObj; oi++) {
                        NvDsTargetMiscDataFrame* f = obj->list + oi;
                        py::dict bbox;
                        bbox["left"]   = f->tBbox.left;
                        bbox["top"]    = f->tBbox.top;
                        bbox["width"]  = f->tBbox.width;
                        bbox["height"] = f->tBbox.height;
                        py::dict fd;
                        fd["frame_num"]  = f->frameNum;
                        fd["bbox"]       = bbox;
                        fd["confidence"] = f->confidence;
                        fd["age"] = f->age;
                        fd["tracker_state"] = static_cast<int>(f->trackerState);
                        fd["visibility"] = f->visibility;
                        fd["world_foot_location"] = py::make_tuple(f->ptWorldFeet[0], f->ptWorldFeet[1]);
                        frames.append(fd);
                    }
                    py::dict od;
                    od["unique_id"] = obj->uniqueId;
                    od["label"]     = std::string(obj->objLabel);
                    od["frames"]    = frames;
                    objects.append(od);
                }
                py::dict sd;
                sd["stream_id"] = s->streamID;
                sd["objects"]   = objects;
                streams.append(sd);
            }
            return streams;
        });

}
