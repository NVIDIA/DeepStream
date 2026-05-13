#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

PIPELINE_NAME = "deepstream-test3"
CONFIG_FILE_PATH = "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-test3/dstest3_pgie_config.yml"
BATCHED_PUSH_TIMEOUT = 33000
MUXER_WIDTH = 1920
MUXER_HEIGHT = 1080
TILER_WIDTH = 1280
TILER_HEIGHT = 720

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
    if isinstance(file_path, list) or not os.path.splitext(file_path)[1] in [".yaml", ".yml"]:
        file_list = file_path if isinstance(file_path, list) else [file_path]
        pipeline = Pipeline(PIPELINE_NAME)
        pipeline.add("nvstreammux", "mux", {"batch-size": len(file_list), "batched-push-timeout": BATCHED_PUSH_TIMEOUT, "width": MUXER_WIDTH, "height": MUXER_HEIGHT})
        for i, file in enumerate(file_list):
            pipeline.add("uridecodebin", f"src_{i}", {"uri": file})
            pipeline.link((f"src_{i}", "mux"), ("", "sink_%u"))
        pipeline.add("nvinfer", "infer", {"config-file-path": CONFIG_FILE_PATH, "batch-size": len(file_list)})
        pipeline.add("nvmultistreamtiler", "tiler", {"width": TILER_WIDTH, "height": TILER_HEIGHT})
        pipeline.add("nvosdbin", "osd").add("nv3dsink" if platform.processor() == "aarch64" else "nveglglessink", "sink")
        pipeline.link("mux", "infer", "tiler", "osd", "sink")
        pipeline.attach("infer", Probe("counter", ObjectCounterMarker()))
        pipeline.start().wait()
    else:
        Pipeline(PIPELINE_NAME, file_path).attach("infer", Probe("counter", ObjectCounterMarker())).start().wait()

if __name__ == '__main__':
    # Check input arguments
    if len(sys.argv) < 2:
        sys.stderr.write("usage: %s <YAML config file> OR <uri1> [uri2] ... [uriN]\n" % sys.argv[0])
        sys.exit(1)

    # pipeline.wait() in the main function is a blocking call due to which the KeyboardInterrupt may not be processed immediately.
    # we use Process from multiprocessing which runs the main function in a different process and processes KeyboardInterrupt immediately.
    process = Process(target=main, args=(sys.argv[1] if len(sys.argv) == 2 else sys.argv[1:],))
    try:
        process.start()
        process.join()
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Terminating process...")
        process.terminate()