###################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###################################################################################################

from pyservicemaker import Pipeline, Probe, BatchMetadataOperator, osd
from multiprocessing import Process
import sys
import platform
import os

PIPELINE_NAME = "deepstream-nvdsanalytics-test"
CONFIG_FILE_PATH = "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-nvdsanalytics-test/nvdsanalytics_pgie_config.txt"
ANALYTICS_CONFIG_FILE_PATH = "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-nvdsanalytics-test/config_nvdsanalytics.txt"
TRACKER_LL_CONFIG_FILE = "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml"
TRACKER_LL_LIB_FILE = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"
BATCHED_PUSH_TIMEOUT = 33000
MUXER_WIDTH = 1920
MUXER_HEIGHT = 1080
TILER_WIDTH = 1280
TILER_HEIGHT = 720

class ObjectCounterMarker(BatchMetadataOperator):
    def handle_metadata(self, batch_meta):
        for frame_meta in batch_meta.frame_items:
            vehicle_count = 0
            person_count = 0
            for object_meta in frame_meta.object_items:
                class_id = object_meta.class_id
                if class_id == 0:
                    vehicle_count += 1
                elif class_id == 2:
                    person_count += 1
                for user_meta in object_meta.nvdsanalytics_obj_items:
                    nvdsanalytics_obj_info = user_meta.as_nvdsanalytics_obj()
                    print("Object {0} moving in direction: {1}".format(object_meta.object_id, nvdsanalytics_obj_info.dir_status))
                    print("Object {0} line crossing status: {1}".format(object_meta.object_id, nvdsanalytics_obj_info.lc_status))
                    print("Object {0} overcrowding status: {1}".format(object_meta.object_id, nvdsanalytics_obj_info.oc_status))
                    print("Object {0} ROI status: {1}".format(object_meta.object_id, nvdsanalytics_obj_info.roi_status))
                    print("Object {0} status: {1}".format(object_meta.object_id, nvdsanalytics_obj_info.obj_status))
                    print("Object {0} unique ID: {1}".format(object_meta.object_id, nvdsanalytics_obj_info.unique_id))
            for user_meta in frame_meta.nvdsanalytics_frame_items:
                nvdsanalytics_frame_meta = user_meta.as_nvdsanalytics_frame()
                print("Frame {0} overcrowding status: {1}".format(frame_meta.frame_number, nvdsanalytics_frame_meta.oc_status))
                print("Frame {0} object in ROI count: {1}".format(frame_meta.frame_number, nvdsanalytics_frame_meta.obj_in_roi_cnt))
                print("Frame {0} object line crossing current count: {1}".format(frame_meta.frame_number, nvdsanalytics_frame_meta.obj_lc_curr_cnt))
                print("Frame {0} object line crossing cumulative count: {1}".format(frame_meta.frame_number, nvdsanalytics_frame_meta.obj_lc_cum_cnt))
                print("Frame {0} unique ID: {1}".format(frame_meta.frame_number, nvdsanalytics_frame_meta.unique_id))
                print("Frame {0} object count: {1}".format(frame_meta.frame_number, nvdsanalytics_frame_meta.obj_cnt))
            print(f"Object Counter: Pad Idx={frame_meta.pad_index},"
                  f"Frame Number={frame_meta.frame_number},"
                  f"Vehicle Count={vehicle_count}, Person Count={person_count}")
            display_text = f"Person={person_count},Vehicle={vehicle_count}"
            display_meta = batch_meta.acquire_display_meta()
            text = osd.Text()
            text.display_text = display_text.encode('ascii')
            text.x_offset = 10
            text.y_offset = 12
            text.font.name = osd.FontFamily.Serif
            text.font.size = 12
            text.font.color = osd.Color(1.0, 1.0, 1.0, 1.0)
            text.set_bg_color = True
            text.bg_color = osd.Color(0.0, 0.0, 0.0, 1.0)
            display_meta.add_text(text)
            frame_meta.append(display_meta)

def main(file_path):
    if isinstance(file_path, list):
        file_list = file_path if isinstance(file_path, list) else [file_path]
        pipeline = Pipeline(PIPELINE_NAME)
        pipeline.add("nvstreammux", "mux", {"batch-size": len(file_list), "batched-push-timeout": BATCHED_PUSH_TIMEOUT, "width": MUXER_WIDTH, "height": MUXER_HEIGHT, "compute-hw": 1, "nvbuf-memory-type": 2})
        for i, file in enumerate(file_list):
            pipeline.add("uridecodebin", f"src_{i}", {"uri": file})
            pipeline.link((f"src_{i}", "mux"), ("", "sink_%u"))
        pipeline.add("nvinfer", "infer", {"config-file-path": CONFIG_FILE_PATH, "batch-size": len(file_list)})
        pipeline.add("nvtracker", "tracker", {"ll-config-file": TRACKER_LL_CONFIG_FILE, "ll-lib-file": TRACKER_LL_LIB_FILE})
        pipeline.add("nvdsanalytics", "analytics", {"config-file": ANALYTICS_CONFIG_FILE_PATH})
        pipeline.add("nvmultistreamtiler", "tiler", {"width": TILER_WIDTH, "height": TILER_HEIGHT})
        pipeline.add("nvosdbin", "osd").add("nv3dsink" if platform.processor() == "aarch64" else "nveglglessink", "sink")
        pipeline.link("mux", "infer", "tracker", "analytics", "tiler", "osd", "sink")
        pipeline.attach("tiler", Probe("counter", ObjectCounterMarker()))
        pipeline.start().wait()

if __name__ == '__main__':
    # Check input arguments
    if len(sys.argv) < 2:
        sys.stderr.write("usage: %s <uri1> [uri2] ... [uriN]\n" % sys.argv[0])
        sys.exit(1)

    # pipeline.wait() in the main function is a blocking call due to which the KeyboardInterrupt may not be processed immediately.
    # we use Process from multiprocessing which runs the main function in a different process and processes KeyboardInterrupt immediately.
    process = Process(target=main, args=(sys.argv[1:],))
    try:
        process.start()
        process.join()
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Terminating process...")
        process.terminate()
