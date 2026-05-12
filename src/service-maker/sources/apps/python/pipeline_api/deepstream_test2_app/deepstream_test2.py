#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

from pyservicemaker import Pipeline, Probe, BatchMetadataOperator, osd
from multiprocessing import Process
import sys
import platform
import os

PIPELINE_NAME = "deepstream-test2"
PGIE_CONFIG_FILE_PATH = "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-test2/dstest2_pgie_config.yml"
TRACKER_LL_CONFIG_FILE = "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml"
TRACKER_LL_LIB_FILE = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"
SGIE1_CONFIG_FILE_PATH = "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-test2/dstest2_sgie1_config.yml"
SGIE2_CONFIG_FILE_PATH = "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-test2/dstest2_sgie2_config.yml"

class ObjectCounterMarker(BatchMetadataOperator):
    def handle_metadata(self, batch_meta):
        for frame_meta in batch_meta.frame_items:
            vehcle_count = 0
            person_count = 0
            for object_meta in frame_meta.object_items:
                class_id = object_meta.class_id
                if class_id == 0:
                    vehcle_count += 1
                elif class_id == 2:
                    person_count += 1
            print(f"Object Counter: Pad Idx={frame_meta.pad_index},"
                  f"Frame Number={frame_meta.frame_number},"
                  f"Vehicle Count={vehcle_count}, Person Count={person_count}")
            display_text = f"Person={person_count},Vehicle={vehcle_count}"
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
    file_ext = os.path.splitext(file_path)[1]

    if file_ext in [".yaml", ".yml"]:
        Pipeline(PIPELINE_NAME, file_path).attach("pgie", Probe("counter", ObjectCounterMarker())).start().wait()
    else:
        (Pipeline(PIPELINE_NAME).add("filesrc", "src", {"location": file_path}).add("h264parse", "parser").add("nvv4l2decoder", "decoder")
            .add("nvstreammux", "mux", {"batch-size": 1, "width": 1280, "height": 720})
            .add("nvinfer", "pgie", {"config-file-path": PGIE_CONFIG_FILE_PATH})
            .add("nvinfer", "sgie1", {"config-file-path": SGIE1_CONFIG_FILE_PATH})
            .add("nvinfer", "sgie2", {"config-file-path": SGIE2_CONFIG_FILE_PATH})
            .add("nvtracker", "tracker", {"ll-config-file": TRACKER_LL_CONFIG_FILE, "ll-lib-file": TRACKER_LL_LIB_FILE})
            .add("nvosdbin", "osd").add("nv3dsink" if platform.processor() == "aarch64" else "nveglglessink", "sink")
            .link("src", "parser", "decoder").link(("decoder", "mux"), ("", "sink_%u")).link("mux","pgie", "tracker", "sgie1", "sgie2", "osd", "sink")
            .attach("pgie", Probe("counter", ObjectCounterMarker()))
            .attach("pgie", "measure_fps_probe", "my probe")
            .start().wait())

if __name__ == '__main__':
    # Check input arguments
    if len(sys.argv) != 2:
        sys.stderr.write("usage: %s <H264 filename> OR <YAML config file>\n" % sys.argv[0])
        sys.exit(1)

    # pipeline.wait() in the main function is a blocking call due to which the KeyboardInterrupt may not be processed immediately.
    # we use Process from multiprocessing which runs the main function in a different process and processes KeyboardInterrupt immediately.
    process = Process(target=main, args=(sys.argv[1],))
    try:
        process.start()
        process.join()
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Terminating process...")
        process.terminate()