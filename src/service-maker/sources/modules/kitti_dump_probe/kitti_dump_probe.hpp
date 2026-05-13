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

#include <string>
#include <sys/stat.h>
#include <fstream>

#include "buffer_probe.hpp"

#define UNTRACKED_OBJECT_ID 0xFFFFFFFFFFFFFFFF

#define NVDS_OBJ_VISIBILITY 20
#define NVDS_OBJ_IMAGE_FOOT_LOCATION 21
#define NVDS_TRACKER_PAST_FRAME_META 15

namespace deepstream {
    class NvDsKittiDump: public BufferProbe::IBatchMetadataOperator {
        public:
            void generateInferenceKittiDump(BatchMetadata& data, std::string output_dir) {
                char bbox_file[1024] = {0};
                FILE *bbox_params_dump_file = NULL;
                FrameMetadata::Iterator frame_itr;
                for (data.initiateIterator(frame_itr); !frame_itr->done(); frame_itr->next()) {
                    uint stream_id = (*frame_itr)->padIndex();
                        snprintf(bbox_file, sizeof(bbox_file) - 1, "%s/%03u_%06lu.txt",
                            output_dir.c_str(), stream_id,
                            (ulong)(*frame_itr)->frameNum());
                    bbox_params_dump_file = fopen(bbox_file, "w");
                    if (!bbox_params_dump_file) continue;

                    FrameMetadata& frame_meta = frame_itr->get();
                    ObjectMetadata::Iterator obj_itr;

                    for (frame_meta.initiateIterator(obj_itr); !obj_itr->done(); obj_itr->next()) {
                        ObjectMetadata& object_meta = obj_itr->get();
                        float confidence = object_meta.confidence();
                        float left = object_meta.rectParams().left;
                        float top = object_meta.rectParams().top;
                        float right = left + object_meta.rectParams().width;
                        float bottom = top + object_meta.rectParams().height;

                        if (object_meta.objectId() == UNTRACKED_OBJECT_ID) {
                            fprintf(bbox_params_dump_file,
                                    "%s 0.0 0 0.0 %f %f %f %f 0.0 0.0 0.0 0.0 0.0 0.0 0.0 %f\n",
                                    object_meta.label().c_str(), left, top, right, bottom, confidence);
                        } else {
                            fprintf(bbox_params_dump_file,
                                    "%s %lu 0.0 0 0.0 %f %f %f %f 0.0 0.0 0.0 0.0 0.0 0.0 0.0 %f\n",
                                    object_meta.label().c_str(), object_meta.objectId(), left, top, right, bottom,
                                    confidence);
                        }
                    }
                    fclose(bbox_params_dump_file);
                }
            }

            void generateTrackerKittiDump(BatchMetadata& data, std::string output_dir) {
                char bbox_file[1024] = {0};
                FILE *bbox_params_dump_file = NULL;
                FrameMetadata::Iterator frame_itr;
                for (data.initiateIterator(frame_itr); !frame_itr->done(); frame_itr->next()) {
                    uint stream_id = (*frame_itr)->padIndex();
                    snprintf(bbox_file, sizeof(bbox_file) - 1, "%s/%03u_%06lu.txt",
                            output_dir.c_str(), stream_id,
                            (ulong)(*frame_itr)->frameNum());
                    bbox_params_dump_file = fopen(bbox_file, "w");
                    if (!bbox_params_dump_file) continue;

                    FrameMetadata& frame_meta = frame_itr->get();
                    ObjectMetadata::Iterator obj_itr;

                    for (frame_meta.initiateIterator(obj_itr); !obj_itr->done(); obj_itr->next()) {
                        ObjectMetadata& object_meta = obj_itr->get();
                        float confidence = object_meta.trackerConfidence();
                        uint64_t id = object_meta.objectId();
                        bool write_proj_info = false;
                        float visibility = -1.0, x_img_foot = -1.0, y_img_foot = -1.0;

                        float left = object_meta.nvBboxInfo().left;
                        float top = object_meta.nvBboxInfo().top;
                        float right = left + object_meta.nvBboxInfo().width;
                        float bottom = top + object_meta.nvBboxInfo().height;

                        object_meta.iterate([&visibility, &write_proj_info](const UserMetadata& user_meta) {
                            ObjectVisibilityUserMetadata visibility_meta(user_meta);
                            visibility = visibility_meta.getVisibility();
                            write_proj_info = true;
                        }, NVDS_OBJ_VISIBILITY);

                        object_meta.iterate([&x_img_foot, &y_img_foot, &write_proj_info](const UserMetadata& user_meta) {
                            ObjectImageFootLocationUserMetadata foot_meta(user_meta);
                            x_img_foot = foot_meta.getImageFootLocation().first;
                            y_img_foot = foot_meta.getImageFootLocation().second;
                            write_proj_info = true;
                        }, NVDS_OBJ_IMAGE_FOOT_LOCATION);


                        if (write_proj_info)
                        {
                            fprintf (bbox_params_dump_file,
                                "%s %lu 0.0 0 0.0 %f %f %f %f 0.0 0.0 0.0 0.0 0.0 0.0 0.0 %f %f %f %f\n",
                                object_meta.label().c_str(), id, left, top, right, bottom, confidence, visibility, x_img_foot, y_img_foot);
                        }
                        else
                        {
                            fprintf (bbox_params_dump_file,
                                "%s %lu 0.0 0 0.0 %f %f %f %f 0.0 0.0 0.0 0.0 0.0 0.0 0.0 %f\n",
                                object_meta.label().c_str(), id, left, top, right, bottom, confidence);
                        }
                    }
                    fclose(bbox_params_dump_file);
                }
            }

            void generateTrackerPastKittiDump(BatchMetadata& data, std::string output_dir) {
                data.iterate([&output_dir](const UserMetadata &user_meta) { 
                    TrackerPastFrameUserMetadata past_frame_meta(user_meta);
                    NvDsTargetMiscDataBatch* pPastFrameObjBatch = past_frame_meta.getTrackerMiscDataBatch();
                    
                    if (!pPastFrameObjBatch) {
                        return; // Skip if batch is null
                    }
                    
                    for (uint si = 0; si < pPastFrameObjBatch->numFilled; si++) {
                        NvDsTargetMiscDataStream *objStream = (pPastFrameObjBatch->list) + si;
                        uint stream_id = (uint) (objStream->streamID);
                        for (uint li = 0; li < objStream->numFilled; li++) {
                            NvDsTargetMiscDataObject *objList = (objStream->list) + li;
                            for (uint oi = 0; oi < objList->numObj; oi++) {
                                NvDsTargetMiscDataFrame *obj = (objList->list) + oi;
                                
                                char bbox_file[1024] = { 0 };
                                snprintf(bbox_file, sizeof(bbox_file) - 1, "%s/%03u_%06lu.txt",
                                    output_dir.c_str(), stream_id, (ulong)obj->frameNum);

                                float left = obj->tBbox.left;
                                float right = left + obj->tBbox.width;
                                float top = obj->tBbox.top;
                                float bottom = top + obj->tBbox.height;
                                // Past frame object confidence given by tracker
                                float confidence = obj->confidence;

                                FILE *bbox_params_dump_file = fopen(bbox_file, "a");
                                if (!bbox_params_dump_file) {
                                    continue; // Skip this iteration if file open fails
                                }
                                
                                fprintf(bbox_params_dump_file,
                                    "%s %lu 0.0 0 0.0 %f %f %f %f 0.0 0.0 0.0 0.0 0.0 0.0 0.0 %f\n",
                                    objList->objLabel, objList->uniqueId, left, top, right, bottom,
                                    confidence);
                                fclose(bbox_params_dump_file);
                            }
                        }
                    }
                }, NVDS_TRACKER_PAST_FRAME_META);
            }

            NvDsKittiDump () {}
            virtual probeReturn handleData(BufferProbe& probe, BatchMetadata& data) {
                std::string output_dir = "/tmp/kitti/";
                probe.getProperty("kitti-dir", output_dir);

                bool tracker_kitti_output = false;
                probe.getProperty("tracker-kitti-output", tracker_kitti_output);

                if (tracker_kitti_output) {
                    generateTrackerKittiDump(data, output_dir);
                    generateTrackerPastKittiDump(data, output_dir);
                } else {
                    generateInferenceKittiDump(data, output_dir);
                }

                return probeReturn::Probe_Ok;
            }
    };

}

